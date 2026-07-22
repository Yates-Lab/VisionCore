"""Unified data loader for figure 3's analysis row (panels C/D/E).

Within-model ablations on fixRSVP: hold the trained concat-model weights fixed
and, at inference, remove one of the two FEM information routes:

  - intact     : full retinal stimulus + full behavior input (reference)
  - zeroed     : behavior set to 0 (extraretinal route removed; "retinal only")
  - stabilized : retinal input frozen at ONE session-global centroid gaze so the
                 image no longer moves with the eye (reafferent route removed;
                 "extraretinal only"), behavior intact.

The two ablations are symmetric counterfactuals. The fixRSVP stimulus is rendered
in retinal (eye-referenced) coordinates, so fixational eye movements are already
in the visual stream; the behavior tensor is the only *separate* extraretinal
route. `zeroed` removes the extraretinal route; `stabilized` removes the reafferent
(retinal-image-motion) route. If single-trial prediction survives `zeroed` but
collapses under `stabilized`, the trial-to-trial variability the twin captures is
carried by the moving retinal image, not by extraretinal modulation of V1.

The stabilized retinal input is rendered pixel-exactly with DataYatesV1
`FixRsvpTrial.get_rois` at a constant session-global centroid ROI (validated
against the native grid_sample renderer), in the raw 240 Hz frame, then decimated
to the model's
120 Hz frame exactly as the training pipeline does (`downsample_stimulus` =
decimation). A per-session alignment gate asserts decimate(raw stored stim) ==
embedded `dset['stim']` bit-exactly, so the substitution is frame-aligned and the
intact re-render carries zero rendering artifact.

This cache is the single source for all three analysis panels, so every panel
draws on the same sessions, neurons, and `good` mask:

  - panel C : trial-averaged held-out prediction, `ccnorm[intact]` vs
              `ccnorm[zeroed]` (normalized correlation; ccmax is shared).
  - panel D : single-trial r^2, `ve_psth` vs `ve[intact]` vs `ve[zeroed]`.
  - panel E : empirical FEM modulation (1 - `alpha`) vs single-trial r^2 gain
              over the PSTH baseline (`ve[zeroed]` / `ve_psth`).

Self-contained: owns cache `outputs/cache/fig3_bottomrow_ablation.pkl` (no
dependency on the behavior-vs-vision within-model cache or the fig3 top-row
cache). 1-alpha is read from the covariance-decomposition cache via
`_fig3_data._load_fig2_alpha_by_session`.
"""
import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import rate_variance_components

from _fig3_data import (
    DT, VALID_TIME_BINS, MIN_FIX_DUR, MIN_TOTAL_SPIKES, CCMAX_THRESHOLD,
    SUBJECTS, CHECKPOINT_PATH,
    COVDECOMP_CACHE_PATH, COVDECOMP_TARGET,
    subject_from_session, _load_fig2_alpha_by_session,
    _load_fig2_included_sessions,
)
from _fig3_helpers import (
    order_single_neuron_by_seriation,
    PANEL_B_SESSION, PANEL_B_NEURON_ID, PANEL_B_MIN_BINS, N_BINS_B,
    PANEL_B_WINDOW_S,
)


CACHE_PATH = CACHE_DIR / "fig3_bottomrow_ablation.pkl"

CONDS = ["intact", "zeroed", "stabilized"]
ABLATIONS = ["zeroed", "stabilized"]          # extraretinal-route first
COND_LABEL = {"intact": "intact", "zeroed": "zeroed", "stabilized": "stabilized"}
BEHAVIOR_CONDS = ["intact", "zeroed"]         # conditions that keep the stored stim
STIM_CONDS = ["stabilized"]                   # conditions that replace the stim
FIX_RADIUS = 1.0                              # deg; fixation = hypot(eyepos) < 1.0
CENTROID_RADIUS = 0.5                          # deg; central window for the global
                                              # stabilization centroid (independent
                                              # of FIX_RADIUS; matches fig2's frame)

CCNORM_N_SPLITS = 200
MIN_TRIALS_PER_PHASE = 10


def build_behavior_modifiers():
    """Return {cond: behavior_modifier or None} for the behavior-route conditions.
    Each modifier maps (behavior_tensor, itrial) -> tensor of same shape."""
    import torch

    def zeroed(b, _itrial):
        return torch.zeros_like(b)

    return {"intact": None, "zeroed": zeroed}


def build_stabilized_stim(session_name, embedded_stim, factor):
    """Return (stab_stim, align_maxabs, n_trials_stab) for the reafferent ablation.

    stab_stim: float32 array shaped like `embedded_stim` (N_emb, 1, 51, 51),
    pixel-normalized ((raw-127)/255), a drop-in for dset['stim']; the retinal image
    is frozen at ONE common (session-global) gaze for every trial while the RSVP
    images still flash.

    A per-trial medoid would anchor each trial to a different frozen point, so the
    frozen image would still vary trial-to-trial and leave a spurious across-trial
    signal at matched eye positions (inflating the estimated FEM modulation). To get
    a true extraretinal-only control we freeze every trial at a SINGLE gaze: the
    centroid of `dpi_pix` over all valid samples inside the central CENTROID_RADIUS
    deg (independent of FIX_RADIUS), realized as the ROI of the one session bin whose
    `dpi_pix` is nearest that centroid. That ROI is reused for all trials.

    The stim is rendered pixel-exactly via DataYatesV1 `FixRsvpTrial.get_rois` with a
    constant global ROI in the raw 240 Hz frame, then decimated to the model's
    120 Hz frame (the pipeline's `downsample_stimulus` is decimation). Because the
    other covariates are average-pooled at downsample time, the render must use the
    RAW covariates, not the embedded ones.

    align_maxabs: max |decimate(raw stored stim) - embedded_stim| over all bins;
    MUST be 0 for the substitution to be frame-aligned (also proves the intact
    re-render is artifact-free).
    """
    from DataYatesV1.utils.io import YatesV1Session
    from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
    from DataYatesV1.utils.general import get_clock_functions
    from DataYatesV1.utils.data.datasets import DictDataset

    sess = YatesV1Session(session_name)
    exp = sess.exp
    ptb2ephys, _ = get_clock_functions(exp)
    raw = DictDataset.load(sess.sess_dir / "datasets" / "fixrsvp.dset")

    raw_stim = raw["stim"].numpy()                       # (Nraw,51,51) uint8
    trial_inds = raw["trial_inds"].numpy().astype(int)
    t_bins = raw["t_bins"].numpy()
    roi_all = raw["roi"].numpy()
    dpi_pix = raw["dpi_pix"].numpy()
    dpi_valid = raw["dpi_valid"].numpy() > 0
    eyepos = raw["eyepos"].numpy()
    fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < FIX_RADIUS

    Nraw = raw_stim.shape[0]
    keep = (Nraw // factor) * factor

    # alignment gate: decimate(raw stored) must equal embedded stim
    emb = np.asarray(embedded_stim)
    emb_px = np.rint(emb.reshape(emb.shape[0], *emb.shape[-2:]) * 255 + 127).astype(int)
    dec = raw_stim[:keep:factor].astype(int)
    n = min(len(dec), len(emb_px))
    align_maxabs = int(np.abs(dec[:n] - emb_px[:n]).max())

    # Session-global stabilization gaze: centroid of dpi_pix over all valid samples
    # inside the central CENTROID_RADIUS deg, realized as the ROI of the single bin
    # nearest that centroid. Reused for every trial so the frozen retinal image is
    # identical across trials (true extraretinal-only control). Falls back to the
    # fixation window only if no sample lands inside CENTROID_RADIUS.
    central = np.hypot(eyepos[:, 0], eyepos[:, 1]) < CENTROID_RADIUS
    global_valid = central & dpi_valid
    if not np.any(global_valid):
        global_valid = fixation & dpi_valid
    gidx = np.where(global_valid)[0]
    centroid = dpi_pix[gidx].mean(axis=0)
    med_global = int(gidx[np.argmin(((dpi_pix[gidx] - centroid) ** 2).sum(1))])
    roi_global = roi_all[med_global]

    # Global-centroid-stabilized render (raw frame): freeze every trial at roi_global.
    stab_raw = raw_stim.copy()
    n_trials_stab = 0
    for iT in np.unique(trial_inds):
        m = trial_inds == iT
        if not np.any(m & fixation & dpi_valid):
            continue
        trial = FixRsvpTrial(exp["D"][iT], exp["S"])
        start_idx = np.where(trial.image_ids == 2)[0][0]
        flip_times = ptb2ephys(trial.flip_times[start_idx:])
        hist_idx = np.searchsorted(flip_times, t_bins[m], side="right") - 1 + start_idx
        roi_const = np.repeat(roi_global[None], m.sum(), axis=0)
        stab_raw[m] = trial.get_rois(hist_idx, roi=roi_const)
        n_trials_stab += 1

    stab_dec = stab_raw[:keep:factor].astype(np.float32)
    stab_stim = ((stab_dec - 127.0) / 255.0)[:, None]    # (Nemb,1,51,51)
    stab_stim = stab_stim[:emb.shape[0]]
    return stab_stim.astype(np.float32), align_maxabs, n_trials_stab


def _var_explained(pred, true, axis=None):
    return 1 - np.nanvar(pred - true, axis=axis) / np.nanvar(true, axis=axis)


def _compute_ccnorm_by_condition(robs, rhat_rs, dfs, n_splits=CCNORM_N_SPLITS):
    """Per-neuron trial-averaged ccnorm for each behavior condition.

    ccmax is the split-half reliability of the observed responses, so it is a
    property of `robs` alone and identical across conditions; we compute it once
    and return it alongside a {cond: ccnorm} dict. Each condition's ccnorm is
    averaged over two split-half seeds and neurons whose two estimates disagree
    (squared diff > 0.01) are dropped, mirroring the fig3 top-row loader.
    """
    from eval.eval_stack_utils import ccnorm_split_half_variable_trials

    ccnorm = {}
    ccmax = None
    for c, r in rhat_rs.items():
        cn1, _, cm1, _, _ = ccnorm_split_half_variable_trials(
            robs, r, dfs, n_splits=n_splits, return_components=True, rng=42)
        cn2, _, cm2, _, _ = ccnorm_split_half_variable_trials(
            robs, r, dfs, n_splits=n_splits, return_components=True, rng=43)
        cn = 0.5 * (cn1 + cn2)
        cn[(cn1 - cn2) ** 2 > 0.01] = np.nan
        ccnorm[c] = cn
        if ccmax is None:
            ccmax = 0.5 * (cm1 + cm2)
    return ccnorm, ccmax


def _compute_model_one_minus_alpha_by_condition(rhat_rs, dfs):
    """Per-neuron model 1-alpha for each behavior condition."""
    n_neurons = dfs.shape[2]
    out = {c: np.full(n_neurons, np.nan, dtype=float) for c in CONDS}
    for c, rates in rhat_rs.items():
        for ni in range(n_neurons):
            comp = rate_variance_components(
                rates[:, :, ni],
                valid=dfs[:, :, ni] != 0,
                min_trials_per_phase=MIN_TRIALS_PER_PHASE,
            )
            out[c][ni] = comp["one_minus_alpha"]
    return out


def _run_inference():
    """Run the concat model on every Allen/Logan fixRSVP session under all three
    behavior conditions. Returns per-session result dicts and writes CACHE_PATH."""
    import torch
    from tqdm import tqdm
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import (
        load_single_dataset, run_model, rescale_rhat,
    )

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    fig2_alpha_by_session = _load_fig2_alpha_by_session()

    device = get_free_device()
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")

    results = []
    for dataset_idx, session_name in enumerate(model.names):
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            continue
        print(f"\n--- {session_name} ({subject}) ---")

        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
            fixrsvp_inds = torch.cat([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp'),
            ], dim=0)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        dset_idx_local = fixrsvp_inds[:, 0].unique().item()
        dset = train_data.dsets[dset_idx_local]

        trial_inds = np.asarray(dset.covariates['trial_inds']).ravel()
        psth_inds_flat = np.asarray(dset.covariates['psth_inds']).ravel()
        robs_flat = np.asarray(dset['robs'])
        eyepos_flat = np.asarray(dset['eyepos'])
        fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < 1.0

        trials = np.unique(trial_inds)
        NT, NC = len(trials), robs_flat.shape[1]
        T = int(psth_inds_flat.max()) + 1
        stim_lags = np.array(dataset_config['keys_lags']['stim'])

        beh_mod = build_behavior_modifiers()

        # Reafferent-ablation stim: retinal image frozen at one session-global
        # centroid gaze, rendered from raw data and decimated to the model frame.
        # The alignment gate must pass (== 0) or the substitution is not frame-aligned.
        samp = dataset_config.get('sampling', {})
        factor = (int(samp['source_rate']) // int(samp['target_rate'])) if samp else 1
        stab_stim_np, align_maxabs, n_tr_stab = build_stabilized_stim(
            session_name, dset['stim'].numpy(), factor)
        print(f"  stabilized render: {n_tr_stab} trials, factor={factor}, "
              f"alignment max-abs(decimate(raw)-embedded)={align_maxabs}")
        if align_maxabs != 0:
            print(f"  Skipping: stabilized-stim alignment gate failed "
                  f"(max-abs={align_maxabs})")
            continue
        stab_stim = torch.from_numpy(stab_stim_np)

        robs = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
        fix_dur = np.full(NT, np.nan)
        rhat = {c: np.full((NT, T, NC), np.nan) for c in CONDS}

        for itrial in tqdm(range(NT), desc=f"  {session_name}"):
            ix = (trial_inds == trials[itrial]) & fixation
            if not np.any(ix):
                continue
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            stim_stab = stab_stim[stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior0 = dset['behavior'][ix]
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
            for c in CONDS:
                if c in STIM_CONDS:            # replace stim, keep behavior intact
                    batch = {'stim': stim_stab, 'behavior': behavior0}
                else:                          # keep stored stim, modify behavior
                    behavior = (behavior0 if beh_mod[c] is None
                                else beh_mod[c](behavior0, itrial))
                    batch = {'stim': stim, 'behavior': behavior}
                out = run_model(model, batch, dataset_idx=dataset_idx)
                rhat[c][itrial, t_inds] = out['rhat'].detach().cpu().numpy()

        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < 10:
            print(f"  Skipping: only {good_trials.sum()} good trials")
            continue
        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[good_trials][:, iix]
        dfs = dfs[good_trials][:, iix]
        rhat = {c: r[good_trials][:, iix] for c, r in rhat.items()}

        neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
        if len(neuron_mask) < 3:
            print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
            continue
        robs = robs[:, :, neuron_mask]
        dfs = dfs[:, :, neuron_mask]
        rhat = {c: r[:, :, neuron_mask] for c, r in rhat.items()}
        n_trials, n_time, n_neurons = robs.shape
        print(f"  {n_trials} trials, {n_time} bins, {n_neurons} neurons")

        def rescale(r):
            rr, _ = rescale_rhat(
                torch.from_numpy(robs.reshape(-1, n_neurons)),
                torch.from_numpy(r.reshape(-1, n_neurons)),
                torch.from_numpy(dfs.reshape(-1, n_neurons)),
                mode='affine',
            )
            return rr.reshape(n_trials, n_time, n_neurons).detach().cpu().numpy()

        rhat_rs = {c: rescale(r) for c, r in rhat.items()}

        robs_m = robs.copy(); robs_m[dfs == 0] = np.nan
        rhat_m = {c: r.copy() for c, r in rhat_rs.items()}
        for c in CONDS:
            rhat_m[c][dfs == 0] = np.nan

        ve = {c: _var_explained(rhat_m[c], robs_m, axis=(0, 1)) for c in CONDS}
        model_one_minus_alpha = _compute_model_one_minus_alpha_by_condition(rhat_rs, dfs)

        rbar = np.zeros_like(robs_m)
        for i in range(n_trials):
            other = np.setdiff1d(np.arange(n_trials), i)
            rbar[i] = np.nanmean(robs_m[other], axis=0)
        ve_psth = _var_explained(rbar, robs_m, axis=(0, 1))

        ccnorm, ccmax = _compute_ccnorm_by_condition(robs, rhat_rs, dfs)

        alpha = np.full(n_neurons, np.nan)
        if session_name in fig2_alpha_by_session:
            f2 = fig2_alpha_by_session[session_name]
            for i, nidx in enumerate(neuron_mask):
                loc = np.where(f2["neuron_mask"] == nidx)[0]
                if len(loc) == 1:
                    alpha[i] = f2["alpha"][loc[0]]
        else:
            print(f"  Warning: {session_name} not in fig2 cache (no alpha)")

        example = None
        if session_name == PANEL_B_SESSION:
            matches = np.where(neuron_mask == PANEL_B_NEURON_ID)[0]
            if len(matches):
                ni = int(matches[0])
                dfs_n = dfs[:, :, ni]
                robs_n = robs[:, :, ni]
                _, _, order, _ = order_single_neuron_by_seriation(
                    robs_n, rhat_rs['intact'][:, :, ni], dfs_n)
                any_valid = (dfs_n > 0).any(axis=0)
                fb = int(np.argmax(any_valid))
                end = fb + PANEL_B_MIN_BINS
                keep = (dfs_n[:, fb:end] > 0).sum(axis=1) >= PANEL_B_MIN_BINS

                def prep(arr2d):
                    k = arr2d[keep, fb:end].astype(float).copy()
                    kv = dfs_n[keep, fb:end] > 0
                    k[~kv] = np.nan
                    return (k[order] / DT)[:, :N_BINS_B]

                example = {
                    "neuron_id": PANEL_B_NEURON_ID,
                    "obs_rate": prep(robs_n),
                    "rate": {c: prep(rhat_rs[c][:, :, ni]) for c in CONDS},
                    "window_s": PANEL_B_WINDOW_S,
                }
                print(f"  example neuron {PANEL_B_NEURON_ID}: "
                      f"{example['obs_rate'].shape[0]} raster trials")

        results.append({
            "session": session_name, "subject": subject,
            "neuron_mask": neuron_mask, "n_neurons": n_neurons,
            "ve": ve, "ve_psth": ve_psth, "ccnorm": ccnorm, "ccmax": ccmax,
            "alpha": alpha,
            "model_one_minus_alpha": model_one_minus_alpha,
            "example": example,
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        dill.dump(results, f)
    print(f"\nCached {len(results)} sessions to {CACHE_PATH}")
    return results


def aggregate(results):
    """Flatten per-cell arrays across sessions; `good` = ccmax > threshold."""
    ve = {c: [] for c in CONDS}
    ccnorm = {c: [] for c in CONDS}
    model_one_minus_alpha = {c: [] for c in CONDS}
    ve_psth, ccmax, alpha, subjects = [], [], [], []
    for r in results:
        for c in CONDS:
            ve[c].append(r["ve"][c])
            if "ccnorm" in r:
                ccnorm[c].append(r["ccnorm"][c])
            if "model_one_minus_alpha" in r:
                model_one_minus_alpha[c].append(r["model_one_minus_alpha"][c])
        ve_psth.append(r["ve_psth"])
        ccmax.append(r["ccmax"])
        alpha.append(r["alpha"])
        subjects.extend([r["subject"]] * r["n_neurons"])
    agg = {"ve": {c: np.concatenate(ve[c]) for c in CONDS}}
    if all(ccnorm[c] for c in CONDS):
        agg["ccnorm"] = {c: np.concatenate(ccnorm[c]) for c in CONDS}
    if all(model_one_minus_alpha[c] for c in CONDS):
        agg["model_one_minus_alpha"] = {
            c: np.concatenate(model_one_minus_alpha[c]) for c in CONDS
        }
    agg["ve_psth"] = np.concatenate(ve_psth)
    agg["ccmax"] = np.concatenate(ccmax)
    agg["alpha"] = np.concatenate(alpha)
    agg["subjects"] = np.array(subjects)
    agg["good"] = agg["ccmax"] > CCMAX_THRESHOLD
    return agg


def _raw_one_minus_alpha(results):
    """Unclipped fig2 1-alpha, aligned to `aggregate`'s flattened cell order.

    Reads the covariance-decomposition cache (first counting window,
    target='full') and returns 1 - diag(Cpsth)/diag(Crate) WITHOUT clipping,
    exactly as fig2's `derive.py` computes it (NaN where diag(Crate) <= 0). The
    stored `alpha` in this cache is `_load_fig2_alpha_by_session`'s clipped
    version, which folds every diag_psth/diag_rate > 1 cell onto 1-alpha=0 and
    keeps it; fig2 instead *excludes* those cells (0 <= 1-alpha <= 1). Panel E
    uses this unclipped value plus the `fem_include` mask so its 1-alpha axis
    describes the exact population fig2 reports (no clip pile-up at 0).

    Ordering matches `aggregate`: results-order x neuron_mask-order.
    """
    with open(COVDECOMP_CACHE_PATH, "rb") as f:
        srs = dill.load(f)
    raw = {}
    for sr in srs:
        if sr["subject"] not in SUBJECTS:
            continue
        block = sr["windows"][0]["targets"][COVDECOMP_TARGET]
        diag_psth = np.diag(block["Cpsth"])
        diag_rate = np.diag(block["Crate"])
        with np.errstate(divide="ignore", invalid="ignore"):
            oma = 1.0 - diag_psth / diag_rate
        oma[~(diag_rate > 0)] = np.nan
        for nid, v in zip(sr["neuron_mask"], oma):
            raw[(sr["session"], int(nid))] = float(v)
    out = []
    for r in results:
        for nid in r["neuron_mask"]:
            out.append(raw.get((r["session"], int(nid)), np.nan))
    return np.asarray(out, dtype=float)


def select_ablation_example(results):
    """Return the example-neuron payload (pinned PANEL_B session) or None."""
    return next((r["example"] for r in results if r.get("example")), None)


_cached_data = None


def load_ablation_data(recompute=False):
    """Return a dict with flattened per-cell arrays, `good` mask, and the
    example-neuron payload. Cached in-process after the first call."""
    global _cached_data
    if _cached_data is not None and not recompute:
        return _cached_data

    if CACHE_PATH.exists() and not recompute:
        print(f"Loading cached ablation results from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            results = dill.load(f)
    else:
        results = _run_inference()

    # Restrict to fig2's floored population (>=10 analyzed units/session) so
    # panels C/D/E describe the exact same sessions/neurons fig2 reports. The
    # only discrepancy is session-level (fig3 otherwise keeps one sub-floor
    # session with zero unit leakage in shared sessions).
    included = _load_fig2_included_sessions()
    kept = [r for r in results if r["session"] in included]
    dropped = [r["session"] for r in results if r["session"] not in included]
    if dropped:
        print(f"Session floor (fig2 population): dropping {len(dropped)} "
              f"sub-floor session(s): {dropped}")
    results = kept

    agg = aggregate(results)
    # Unclipped fig2 1-alpha + fig2's inclusion mask (0 <= 1-alpha <= 1). The
    # panel-E 1-alpha axis uses these instead of the clipped `agg["alpha"]`, so
    # the FEM axis describes the exact population fig2 reports (no clip pile-up
    # at 0). Aligned to the same flattened cell order as `agg`.
    oma_raw = _raw_one_minus_alpha(results)
    agg["one_minus_alpha"] = oma_raw
    agg["fem_include"] = (
        np.isfinite(oma_raw) & (oma_raw >= 0.0) & (oma_raw <= 1.0)
    )
    n_good = int(agg["good"].sum())
    n_excl = int((agg["good"] & ~agg["fem_include"]).sum())
    print(f"Ablation data: {len(results)} sessions, {len(agg['good'])} cells "
          f"({n_good} good, ccmax > {CCMAX_THRESHOLD}); "
          f"FEM-axis excludes {n_excl} good cell(s) with 1-alpha out of [0,1]")

    _cached_data = {
        **agg,
        "results": results,
        "example": select_ablation_example(results),
    }
    return _cached_data


def print_ablation_stats(data=None):
    """Ablation cost in single-trial r^2, normalized two ways:
      - cost/intact   : fraction of the twin's OWN single-trial r^2 lost (always
                        well-defined; matches the 'does the prediction change?' framing)
      - cost/gainPSTH : fraction of the twin's gain over the leave-one-out PSTH
                        (undefined where the twin does not beat the PSTH, e.g. Logan)
    """
    if data is None:
        data = load_ablation_data()
    good = data["good"]
    print("\n=== Fig 3 bottom row — ablation cost (good cells, single-trial r^2) ===")
    print(f"{'cond':<10}{'subject':<8}{'N':<6}{'cost Δr²':<12}{'intact r²':<12}"
          f"{'cost/intact%':<14}{'cost/gainPSTH%':<15}")
    for cond in ABLATIONS:
        for subj in ["All"] + SUBJECTS:
            m = good & np.isfinite(data["ve"]["intact"]) & np.isfinite(data["ve"][cond])
            if subj != "All":
                m = m & (data["subjects"] == subj)
            cost = np.median(data["ve"]["intact"][m] - data["ve"][cond][m])
            intact = np.median(data["ve"]["intact"][m])
            gain = np.median(data["ve"]["intact"][m] - data["ve_psth"][m])
            ci = 100 * cost / intact if abs(intact) > 1e-9 else np.nan
            cg = 100 * cost / gain if abs(gain) > 1e-9 else np.nan
            print(f"{cond:<10}{subj:<8}{m.sum():<6}{cost:<+12.4f}{intact:<12.4f}"
                  f"{ci:<14.1f}{cg:<15.1f}")
