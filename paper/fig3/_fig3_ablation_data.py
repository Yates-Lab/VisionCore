"""Ablation data loader for figure 3 bottom row (panels G/H/I).

Within-model behavior ablation on fixRSVP: hold the trained concat-model weights
fixed and, at inference, ablate only the extraretinal `behavior` input
(eye_vel x20, eye_pos x2 feeding the modulator):

  - intact    : full behavior input (reference)
  - zeroed    : behavior set to 0 (extraretinal route removed; committed condition)
  - permuted  : behavior shuffled across trials (in-distribution adversarial control)

The fixRSVP stimulus is rendered in retinal (eye-referenced) coordinates, so eye
movements are already in the visual stream; the behavior tensor is the only
*separate* extraretinal route. If single-trial prediction is preserved under
ablation, the trial-to-trial variability the twin captures is explained by the
moving retinal image, not by extraretinal modulation of V1.

Self-contained: owns cache `outputs/cache/fig4_bottomrow_ablation.pkl` (no
dependency on the behavior-vs-vision within-model cache or the fig3 top-row
cache). Single-trial r^2 (ve) and 1-alpha definitions match panels E/F; 1-alpha
is read from `fig2_decomposition.pkl` via `_fig3_data._load_fig2_alpha_by_session`.
"""
import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import rate_variance_components

from _fig3_data import (
    DT, VALID_TIME_BINS, MIN_FIX_DUR, MIN_TOTAL_SPIKES, CCMAX_THRESHOLD,
    SUBJECTS, CHECKPOINT_PATH,
    subject_from_session, _load_fig2_alpha_by_session,
)
from _fig3_helpers import (
    order_single_neuron_by_seriation,
    PANEL_B_SESSION, PANEL_B_NEURON_ID, PANEL_B_MIN_BINS, N_BINS_B,
    PANEL_B_WINDOW_S,
)


CACHE_PATH = CACHE_DIR / "fig4_bottomrow_ablation.pkl"

CONDS = ["intact", "zeroed", "permuted"]
ABLATIONS = ["zeroed", "permuted"]            # committed lead first
COND_LABEL = {"intact": "intact", "zeroed": "zeroed", "permuted": "permuted"}

PERM_SEED = 42
CCMAX_N_SPLITS = 200
MIN_TRIALS_PER_PHASE = 10


def build_modifiers(dset, trials, trial_inds, fixation, NT, seed=PERM_SEED):
    """Return {cond: behavior_modifier or None}. Each modifier maps
    (behavior_tensor, itrial) -> tensor of same shape."""
    import torch

    src = []
    for t in trials:
        ix = (trial_inds == t) & fixation
        src.append(dset['behavior'][ix] if ix.any() else None)

    valid_idx = np.where([s is not None for s in src])[0]
    rng = np.random.default_rng(seed)
    perm = np.arange(NT)
    vperm = rng.permutation(valid_idx)
    for _ in range(5):
        if np.any(vperm != valid_idx):
            break
        vperm = rng.permutation(valid_idx)
    perm[valid_idx] = vperm

    def zeroed(b, _itrial):
        return torch.zeros_like(b)

    def permuted(b, itrial):
        s = src[perm[itrial]]
        if s is None:
            return torch.zeros_like(b)
        bn = b.clone()
        n = min(s.shape[0], b.shape[0])
        bn[:n] = s[:n]
        if n < b.shape[0]:
            bn[n:] = s.mean(dim=0, keepdim=True)
        return bn

    return {"intact": None, "zeroed": zeroed, "permuted": permuted}


def _var_explained(pred, true, axis=None):
    return 1 - np.nanvar(pred - true, axis=axis) / np.nanvar(true, axis=axis)


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
        ccnorm_split_half_variable_trials,
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

        modifiers = build_modifiers(dset, trials, trial_inds, fixation, NT)

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
            behavior0 = dset['behavior'][ix]
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
            for c in CONDS:
                behavior = behavior0 if modifiers[c] is None else modifiers[c](behavior0, itrial)
                out = run_model(model, {'stim': stim, 'behavior': behavior},
                                dataset_idx=dataset_idx)
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

        _, _, ccmax, _, _ = ccnorm_split_half_variable_trials(
            robs, rhat_rs['intact'], dfs,
            n_splits=CCMAX_N_SPLITS, return_components=True,
        )

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
            "ve": ve, "ve_psth": ve_psth, "ccmax": ccmax, "alpha": alpha,
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
    model_one_minus_alpha = {c: [] for c in CONDS}
    ve_psth, ccmax, alpha, subjects = [], [], [], []
    for r in results:
        for c in CONDS:
            ve[c].append(r["ve"][c])
            if "model_one_minus_alpha" in r:
                model_one_minus_alpha[c].append(r["model_one_minus_alpha"][c])
        ve_psth.append(r["ve_psth"])
        ccmax.append(r["ccmax"])
        alpha.append(r["alpha"])
        subjects.extend([r["subject"]] * r["n_neurons"])
    agg = {"ve": {c: np.concatenate(ve[c]) for c in CONDS}}
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

    agg = aggregate(results)
    n_good = int(agg["good"].sum())
    print(f"Ablation data: {len(results)} sessions, {len(agg['good'])} cells "
          f"({n_good} good, ccmax > {CCMAX_THRESHOLD})")

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
