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
draws on the same sessions, neurons, and `cd_population` mask (fig2 inclusion):

  - panel C : trial-averaged held-out prediction, `ccnorm[intact]` vs
              `ccnorm[zeroed]` (normalized correlation; ccmax is shared).
  - panel D : captured count variance on Figure 2-matched, model-valid windows
              divided by Figure 2's own diag(Crate) at the one-bin window, for
              the leave-one-out PSTH and all three twin conditions.
  - panel E : empirical and within-model FEM-modulation fractions from the
              Figure-2-matched close-pair estimator, on the same aligned
              sessions and neurons as the other analysis panels.

Two stages, deliberately split so the expensive one is not held hostage by the
cheap one:

  1. INFERENCE (`_run_inference`, cache `outputs/cache/fig3_ablation_inference.pkl`)
     Everything that needs the model or the raw dataset: `ve`, `ve_psth`,
     `ccnorm`/`ccmax`, `captured_variance`, `matched_var_y`, the `matched_*`
     rate-variance diagnostic, `model_one_minus_alpha`, the Figure-2-matched
     `femfraction` decomposition for every condition, and the example payload.
     Tens of minutes on a GPU. Re-run only when the checkpoint, the conditions,
     the rendering, or the scoring windows change -- bump
     `INFERENCE_SCHEMA_VERSION` when the stored schema or its semantics change.

  2. DERIVE (`_attach_fig2_derived` + `aggregate`, no cache)
     Everything that is a function of the Figure 2 covariance decomposition:
     `alpha`, `fig2_c_rate`, `fig2_c_total`, `explainable_fraction`, and (in
     `aggregate`) `total_variance_ratio` and `explainable_fraction_matched`.
     Recomputed on EVERY load from the current caches. A change to Figure 2's
     estimator, weighting, or inclusion criteria therefore costs a
     `generate_figure3.py` run, not an inference sweep, and cannot leave a stale
     Figure 2 quantity frozen inside the inference cache.

The one asymmetry worth knowing: the `matched_*` diagnostic needs `dfs`, so it
lives in stage 1 even though it uses no model. Changing the covariance
*estimator convention* does invalidate it. It feeds only the
`matched_denominator_sensitivity` entry in the figure manifest, never a
manuscript number and never the score.

No dependency on the behavior-vs-vision within-model cache or the fig3 top-row
cache.
"""
import os
import sys
from pathlib import Path

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import rate_variance_components

from _fig3_data import (
    DT, VALID_TIME_BINS, MIN_FIX_DUR, MIN_TOTAL_SPIKES,
    SUBJECTS, CHECKPOINT_PATH,
    CACHE_PATH as FIG3_DATA_CACHE_PATH,
    COVDECOMP_CACHE_PATH, COVDECOMP_TARGET,
    subject_from_session, analysis_endpoint_mask_and_psth,
    analysis_endpoint_block_filter, analysis_endpoint_block_mean,
    analysis_endpoint_block_sum, analysis_model_indices,
    analysis_reduce_model_output, figure3_analysis_grid,
    load_reference_sessions, align_native_trial_arrays_to_reference,
    _load_fig2_alpha_by_session,
    _load_fig2_included_sessions,
)
from _fig3_helpers import (
    order_single_neuron_by_seriation,
    PANEL_B_SESSION, PANEL_B_NEURON_ID, PANEL_B_MIN_BINS, N_BINS_B,
    PANEL_B_WINDOW_S,
)
from _fig3_explainable_variance import (
    compute_matched_captured_variance,
    estimate_matched_rate_variance,
    explainable_fraction,
    total_variance_ratio,
)


CACHE_PATH = Path(
    os.environ.get(
        "FIG3_ABLATION_CACHE_PATH",
        str(CACHE_DIR / "fig3_ablation_inference.pkl"),
    )
)


def _partial_cache_path(cache_path):
    """Return the resumable sidecar used by the expensive inference pass."""
    cache_path = Path(cache_path)
    return cache_path.with_name(f"{cache_path.stem}.partial{cache_path.suffix}")


def _inference_cache_payload(results, *, complete):
    """Build one versioned cache payload for atomic progress/final writes."""
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "checkpoint_path": CHECKPOINT_PATH,
        "femfraction_count_bins": FIG2_REPORTED_WINDOW_BINS,
        "complete": bool(complete),
        "n_sessions": int(len(results)),
        "results": results,
    }


def _write_inference_cache_atomic(path, payload):
    """Write a cache without exposing a truncated pickle to another process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "wb") as stream:
        dill.dump(payload, stream)
    os.replace(temporary, path)


def _load_partial_inference_cache(cache_path):
    """Load a compatible schema-v7 progress sidecar, if one exists."""
    partial_path = _partial_cache_path(cache_path)
    if not partial_path.exists():
        return [], partial_path
    with open(partial_path, "rb") as stream:
        payload = dill.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid partial Figure-3 cache: {partial_path}")
    expected_checkpoint = str(Path(CHECKPOINT_PATH).resolve())
    observed_checkpoint = str(Path(payload.get("checkpoint_path", "")).resolve())
    if (
        int(payload.get("schema_version", -1)) != INFERENCE_SCHEMA_VERSION
        or int(payload.get("femfraction_count_bins", -1))
        != FIG2_REPORTED_WINDOW_BINS
        or observed_checkpoint != expected_checkpoint
    ):
        raise ValueError(
            "Partial Figure-3 cache is incompatible with the requested "
            f"checkpoint/schema: {partial_path}"
        )
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Partial Figure-3 cache lacks a results list: {partial_path}")
    sessions = [str(record.get("session")) for record in results]
    if len(sessions) != len(set(sessions)):
        raise ValueError(f"Partial Figure-3 cache has duplicate sessions: {partial_path}")
    return results, partial_path

# Bumped when the *inference* stage's output schema or semantics change (new
# condition, different rendering, different scoring windows). Figure 2-derived
# fields are NOT part of this schema -- they are recomputed on every load by
# `_attach_fig2_derived`, so a Figure 2 convention change does not invalidate
# this cache and must not bump this number.
# v2: scored windows now carry Figure 2's fixed 3-bin matching history (was 1 bin
# at W=1), and every window in `SCORED_COUNT_BINS` is stored under
# `scored_by_window`. A v1 cache's numerator is not comparable to a v2 one.
# v3: native-240-Hz twins use causal block-start coordinates, reject blocks that
# cross trial boundaries, average continuous covariates over each supervised
# pair, and align observations/neurons/validity to the canonical Figure-3 cache.
# v4: true native-240-Hz counts and predictions are summed on the exact causal
# 120-Hz grid, and CCnorm eligibility uses a shared, data-only ceiling mask.
# v5: affine fits and every downstream metric use that same finite Boolean
# support, and per-condition CCabs is retained for an explicit identity audit.
# v6: all ablation conditions are divided by the selected intact trace's exact
# CCmax and use its data-only stability mask.
# v7: retain the Figure-2-matched FEM-fraction decomposition for every
# condition.  Panel E can now consume the same inference sweep as panels C/D
# instead of depending on a second, easily stale prediction cache.
INFERENCE_SCHEMA_VERSION = 7

# Counting window (in 120 Hz bins) that panel D's numerator is scored on, and
# the denominator window `_attach_fig2_derived` reads to match it. Panel D uses
# the twin's native resolution; see covariance_decomposition/fig3_windows.py for
# why it differs from the window panel E and Figure 2 report.
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "covariance_decomposition"))
from fig3_windows import (  # noqa: E402
    FIG2_REPORTED_WINDOW_BINS,
    FIG3_SINGLETRIAL_WINDOW_BINS,
)
from model_decompose import decompose_model_session  # noqa: E402

PRODUCTION_COUNT_BINS = FIG3_SINGLETRIAL_WINDOW_BINS
# Every window scored in the same inference pass, on identical predictions, so
# the counting-window choice can be revisited without re-running inference.
# PRODUCTION_COUNT_BINS must be one of these.
SCORED_COUNT_BINS = (1, 3)

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


def _center_crop_spatial(array, target_hw):
    """Center-crop the final two dimensions to a model's embedded aperture."""
    arr = np.asarray(array)
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    height, width = arr.shape[-2:]
    if target_h > height or target_w > width:
        raise ValueError(
            f"Cannot crop spatial shape {(height, width)} to {(target_h, target_w)}."
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return arr[..., top : top + target_h, left : left + target_w]


def build_stabilized_stim(session_name, embedded_stim, factor):
    """Return (stab_stim, align_maxabs, n_trials_stab) for the reafferent ablation.

    stab_stim: float32 array shaped like `embedded_stim` (N_emb, 1, H, W),
    pixel-normalized ((raw-127)/255), a drop-in for dset['stim']; the retinal image
    is frozen at ONE common (session-global) gaze for every trial while the RSVP
    images still flash.

    A per-trial medoid would anchor each trial to a different frozen point, so the
    frozen image would still vary trial-to-trial and leave a spurious across-trial
    signal at matched eye positions (inflating the estimated FEM modulation). To get
    a stabilized-retina control we freeze every trial at a SINGLE gaze: the
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
    embedded_hw = emb.shape[-2:]
    emb_px = np.rint(emb.reshape(emb.shape[0], *emb.shape[-2:]) * 255 + 127).astype(int)
    dec = _center_crop_spatial(raw_stim[:keep:factor], embedded_hw).astype(int)
    n = min(len(dec), len(emb_px))
    align_maxabs = int(np.abs(dec[:n] - emb_px[:n]).max())

    # Session-global stabilization gaze: centroid of dpi_pix over all valid samples
    # inside the central CENTROID_RADIUS deg, realized as the ROI of the single bin
    # nearest that centroid. Reused for every trial so the frozen retinal image is
    # identical across trials (stabilized-retina control). Falls back to the
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

    stab_dec = _center_crop_spatial(stab_raw[:keep:factor], embedded_hw).astype(np.float32)
    stab_stim = ((stab_dec - 127.0) / 255.0)[:, None]    # (Nemb,1,H,W)
    stab_stim = stab_stim[:emb.shape[0]]
    return stab_stim.astype(np.float32), align_maxabs, n_trials_stab


def _var_explained(pred, true, axis=None):
    return 1 - np.nanvar(pred - true, axis=axis) / np.nanvar(true, axis=axis)


def _compute_ccnorm_by_condition(robs, rhat_rs, dfs, n_splits=CCNORM_N_SPLITS):
    """Per-neuron trial-averaged ccnorm for each behavior condition.

    ccmax is the split-half reliability of the observed responses, so it is a
    property of `robs` alone and identical across conditions; we compute it once
    and return it alongside the per-condition CCabs values. Eligibility is
    determined only by disagreement between the two data-only CCmax estimates;
    every reported CCnorm is then formed explicitly as CCabs / CCmax.
    """
    from eval.eval_stack_utils import ccnorm_split_half_variable_trials

    support = np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)
    ccnorm = {}
    ccabs_by_condition = {}
    ccmax = None
    unstable = None
    for c, r in rhat_rs.items():
        r = np.asarray(r)
        missing = support & ~np.isfinite(r)
        if missing.any():
            raise RuntimeError(
                f"{c}: {int(missing.sum())} predictions missing on CCnorm support"
            )
        _, ca1, cm1, _, _ = ccnorm_split_half_variable_trials(
            robs, r, support, n_splits=n_splits, return_components=True, rng=42)
        _, ca2, cm2, _, _ = ccnorm_split_half_variable_trials(
            robs, r, support, n_splits=n_splits, return_components=True, rng=43)
        if not np.allclose(ca1, ca2, rtol=0, atol=1e-12, equal_nan=True):
            raise AssertionError(f"{c}: CCabs changed across split-half seeds")
        if ccmax is None:
            ccmax = 0.5 * (cm1 + cm2)
            unstable = (cm1 - cm2) ** 2 > 0.01
        else:
            if not np.allclose(ccmax, 0.5 * (cm1 + cm2), rtol=0, atol=1e-12, equal_nan=True):
                raise AssertionError(f"{c}: data-only CCmax changed across conditions")
        ccabs = 0.5 * (ca1 + ca2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cn = ccabs / ccmax
        cn[unstable] = np.nan
        ccnorm[c] = cn
        ccabs_by_condition[c] = ccabs
    return ccnorm, ccmax, ccabs_by_condition, unstable


def _compute_model_one_minus_alpha_by_condition(rhat_rs, dfs):
    """Per-neuron model 1-alpha for each behavior condition."""
    n_neurons = dfs.shape[2]
    valid = np.isfinite(dfs) & (dfs != 0)
    out = {c: np.full(n_neurons, np.nan, dtype=float) for c in CONDS}
    for c, rates in rhat_rs.items():
        for ni in range(n_neurons):
            comp = rate_variance_components(
                rates[:, :, ni],
                valid=valid[:, :, ni],
                min_trials_per_phase=MIN_TRIALS_PER_PHASE,
            )
            out[c][ni] = comp["one_minus_alpha"]
    return out


def _renormalize_ccnorm_to_intact_anchor(ccabs, anchor):
    """Put every counterfactual on one exact, data-only noise ceiling."""
    ccmax = np.asarray(anchor["ccmax"], dtype=np.float64).copy()
    unstable = np.asarray(
        anchor.get("ccnorm_unstable", ~np.isfinite(anchor["ccnorm"])),
        dtype=bool,
    )
    anchored_ccabs = {
        condition: np.asarray(value, dtype=np.float64).copy()
        for condition, value in ccabs.items()
    }
    anchored_ccabs["intact"] = np.asarray(anchor["ccabs"], dtype=np.float64).copy()
    ccnorm = {}
    for condition, value in anchored_ccabs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ccnorm[condition] = value / ccmax
        ccnorm[condition][unstable] = np.nan
    if not np.allclose(
        ccnorm["intact"],
        np.asarray(anchor["ccnorm"]),
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError("anchored CCnorm identity does not reproduce intact trace")
    return ccnorm, ccmax, anchored_ccabs, unstable


def _run_inference(session_filter=None, cache_path=CACHE_PATH):
    """Run the twin under all three conditions and write a summary cache.

    ``session_filter`` and ``cache_path`` support an off-cache alignment smoke
    test before a full production sweep.
    """
    import torch
    from tqdm import tqdm
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import (
        load_single_dataset, run_model, rescale_rhat,
    )

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    # Which sessions the Figure 2 decomposition covers. Read from the aligned
    # cache (72 MB, ~0.1 s) rather than the multi-GB stage-1 cache: the
    # decomposition emits exactly one record per aligned session, so the names
    # agree, and `_attach_fig2_derived` -- which raises on a session missing from
    # Figure 2 -- stays the only place that opens the big cache.
    covdecomp = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
    if covdecomp not in sys.path:
        sys.path.insert(0, covdecomp)
    from data_loading import load_cache as load_aligned_cache
    fig2_sessions = {a["session"] for a in load_aligned_cache()}

    # See _fig3_data._run_inference: allow production jobs to avoid racing an
    # independent GPU workload while preserving automatic selection by default.
    device = get_free_device(os.environ.get("FIG3_GPU"))
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")
    # Loaded lazily only if this checkpoint uses native-rate supervision.
    native_reference = None
    # The selected model's independently evaluated intact trace is the metric
    # anchor for Panel C.  The ablation forward pass is numerically equivalent
    # in VE, but changing batch shape can amplify tiny mixed-precision changes
    # in the split-half normalized-correlation estimator.
    with open(FIG3_DATA_CACHE_PATH, "rb") as stream:
        selected_intact = {row["session"]: row for row in dill.load(stream)}

    cache_path = Path(cache_path)
    resume_enabled = str(
        os.environ.get("FIG3_ABLATION_RESUME", "1")
    ).strip().lower() not in {"0", "false", "no"}
    if resume_enabled:
        results, partial_cache_path = _load_partial_inference_cache(cache_path)
    else:
        results = []
        partial_cache_path = _partial_cache_path(cache_path)
    completed_sessions = {str(record["session"]) for record in results}
    if completed_sessions:
        print(
            f"Resuming schema-v{INFERENCE_SCHEMA_VERSION} ablation inference "
            f"with {len(completed_sessions)} completed sessions from "
            f"{partial_cache_path}"
        )

    eligible_sessions = {
        session_name
        for session_name in model.names
        if subject_from_session(session_name) in SUBJECTS
        and (session_filter is None or session_name in session_filter)
        and session_name in fig2_sessions
        and session_name in selected_intact
    }
    for dataset_idx, session_name in enumerate(model.names):
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            continue
        if session_filter is not None and session_name not in session_filter:
            continue
        if session_name not in fig2_sessions:
            print(f"Skipping {session_name}: absent from the Figure 2 decomposition")
            continue
        if session_name not in selected_intact:
            print(f"Skipping {session_name}: absent from the canonical Figure-3 population")
            continue
        if session_name in completed_sessions:
            print(f"Skipping {session_name}: already present in resumable schema-v7 cache")
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

        analysis_grid = figure3_analysis_grid(dataset_config)
        if analysis_grid["align_to_reference"] and native_reference is None:
            native_reference = load_reference_sessions()

        dset_idx_local = fixrsvp_inds[:, 0].unique().item()
        dset = train_data.dsets[dset_idx_local]

        trial_inds = np.asarray(dset.covariates['trial_inds']).ravel()
        psth_inds_flat = np.asarray(dset.covariates['psth_inds']).ravel()
        analysis_endpoints, psth_inds_analysis = analysis_endpoint_mask_and_psth(
            dataset_config, psth_inds_flat, trial_inds
        )
        robs_flat = np.asarray(dset['robs'])
        dfs_flat = np.asarray(dset['dfs'])
        robs_analysis = analysis_endpoint_block_sum(
            dataset_config, robs_flat, analysis_endpoints
        )
        dfs_analysis = analysis_endpoint_block_filter(
            dataset_config, dfs_flat, analysis_endpoints
        )
        eyepos_flat = np.asarray(dset['eyepos'])
        eyepos_analysis = analysis_endpoint_block_mean(
            dataset_config, eyepos_flat, analysis_endpoints
        )
        fixation = np.hypot(
            eyepos_analysis[:, 0], eyepos_analysis[:, 1]
        ) < 1.0

        trials = np.unique(trial_inds)
        NT, NC = len(trials), robs_flat.shape[1]
        T = int(psth_inds_analysis[analysis_endpoints].max()) + 1
        stim_lags = np.array(dataset_config['keys_lags']['stim'])

        beh_mod = build_behavior_modifiers()

        # Reafferent-ablation stim: retinal image frozen at one session-global
        # centroid gaze, rendered from raw data and decimated to the model frame.
        # The alignment gate must pass (== 0) or the substitution is not frame-aligned.
        samp = dataset_config.get('sampling', {})
        render_factor = (
            int(samp['source_rate']) // int(samp['target_rate']) if samp else 1
        )
        stab_stim_np, align_maxabs, n_tr_stab = build_stabilized_stim(
            session_name, dset['stim'].numpy(), render_factor)
        print(f"  stabilized render: {n_tr_stab} trials, factor={render_factor}, "
              f"alignment max-abs(decimate(raw)-embedded)={align_maxabs}")
        if align_maxabs != 0:
            print(f"  Skipping: stabilized-stim alignment gate failed "
                  f"(max-abs={align_maxabs})")
            continue
        stab_stim = torch.from_numpy(stab_stim_np)

        robs = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
        eyepos = np.full((NT, T, 2), np.nan)
        fix_dur = np.full(NT, np.nan)
        rhat = {c: np.full((NT, T, NC), np.nan) for c in CONDS}

        for itrial in tqdm(range(NT), desc=f"  {session_name}"):
            ix_obs = (trial_inds == trials[itrial]) & fixation & analysis_endpoints
            if not np.any(ix_obs):
                continue
            t_obs = psth_inds_analysis[ix_obs].astype(int)
            fix_dur[itrial] = len(t_obs)
            robs[itrial, t_obs] = robs_analysis[ix_obs]
            dfs[itrial, t_obs] = dfs_analysis[ix_obs]
            eyepos[itrial, t_obs] = eyepos_analysis[ix_obs]

            endpoint_indices, model_indices = analysis_model_indices(
                dataset_config, ix_obs, int(stim_lags.max(initial=0))
            )
            if not len(endpoint_indices):
                continue
            stim_lag_indices = model_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            stim_stab = stab_stim[stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior0 = dset['behavior'][model_indices]
            output_behavior0 = (
                dset['output_behavior'][model_indices]
                if 'output_behavior' in dset
                else None
            )
            t_inds = psth_inds_analysis[endpoint_indices].astype(int)
            for c in CONDS:
                if c in STIM_CONDS:            # replace stim, keep behavior intact
                    batch = {'stim': stim_stab, 'behavior': behavior0}
                    if output_behavior0 is not None:
                        batch['output_behavior'] = output_behavior0
                else:                          # keep stored stim, modify behavior
                    behavior = (behavior0 if beh_mod[c] is None
                                else beh_mod[c](behavior0, itrial))
                    batch = {'stim': stim, 'behavior': behavior}
                    if output_behavior0 is not None:
                        output_behavior = (
                            output_behavior0
                            if beh_mod[c] is None
                            else beh_mod[c](output_behavior0, itrial)
                        )
                        batch['output_behavior'] = output_behavior
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    out = run_model(model, batch, dataset_idx=dataset_idx)
                prediction = analysis_reduce_model_output(
                    dataset_config,
                    out['rhat'].detach().cpu().numpy(),
                    len(endpoint_indices),
                )
                rhat[c][itrial, t_inds] = prediction

        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < 10:
            print(f"  Skipping: only {good_trials.sum()} good trials")
            continue
        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[good_trials][:, iix]
        dfs = dfs[good_trials][:, iix]
        eyepos = eyepos[good_trials][:, iix]
        rhat = {c: r[good_trials][:, iix] for c, r in rhat.items()}

        if analysis_grid["align_to_reference"]:
            if native_reference is None or session_name not in native_reference:
                # The production figure is defined on the canonical reference
                # population.  Newer training manifests can include sessions
                # with FixRSVP trials that are not members of that population,
                # so there is no native neuron/time intersection to score.
                print(
                    "  Skipping: missing from canonical Figure-3 reference "
                    "cache"
                )
                continue
            robs, rhat, dfs, neuron_mask = align_native_trial_arrays_to_reference(
                robs,
                rhat,
                native_reference[session_name],
                label=f"{session_name} ablation",
            )
        else:
            neuron_mask = np.where(
                np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES
            )[0]
            if len(neuron_mask) < 3:
                print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
                continue
            robs = robs[:, :, neuron_mask]
            dfs = dfs[:, :, neuron_mask]
            rhat = {c: r[:, :, neuron_mask] for c, r in rhat.items()}
        n_trials, n_time, n_neurons = robs.shape
        print(f"  {n_trials} trials, {n_time} bins, {n_neurons} neurons")
        score_support = np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)
        score_filter = score_support.astype(np.float32)
        for condition, prediction in rhat.items():
            missing = score_support & ~np.isfinite(prediction)
            if missing.any():
                raise RuntimeError(
                    f"{session_name}/{condition}: {int(missing.sum())} "
                    "predictions missing on data-only support"
                )

        def rescale(r):
            rr, _ = rescale_rhat(
                torch.from_numpy(robs.reshape(-1, n_neurons)),
                torch.from_numpy(r.reshape(-1, n_neurons)),
                torch.from_numpy(score_filter.reshape(-1, n_neurons)),
                mode='affine',
            )
            return rr.reshape(n_trials, n_time, n_neurons).detach().cpu().numpy()

        rhat_rs = {c: rescale(r) for c, r in rhat.items()}

        # Figure 3 panel E reports the same matched close-pair FEM fraction as
        # Figure 2, on the central |eye| < 0.5-deg frame.  Compute it while the
        # exact condition predictions are resident rather than requiring a
        # second inference cache with an independent masking contract.
        fem_valid_mask = (
            np.isfinite(eyepos).all(axis=-1)
            & (np.hypot(eyepos[..., 0], eyepos[..., 1]) < CENTROID_RADIUS)
        )
        femfraction = {
            condition: decompose_model_session(
                prediction,
                robs,
                eyepos,
                fem_valid_mask,
                score_filter,
                count_bins=FIG2_REPORTED_WINDOW_BINS,
            )
            for condition, prediction in rhat_rs.items()
        }
        observed_anchor = femfraction["intact"]
        for condition in ABLATIONS:
            for key in ("B_obs", "B_obs_uncl"):
                if not np.allclose(
                    femfraction[condition][key],
                    observed_anchor[key],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                ):
                    raise AssertionError(
                        f"{session_name}/{condition}: data-only {key} changed "
                        "across model conditions"
                    )

        robs_m = np.where(score_support, robs, np.nan)
        rhat_m = {
            c: np.where(score_support, r, np.nan) for c, r in rhat_rs.items()
        }

        ve = {c: _var_explained(rhat_m[c], robs_m, axis=(0, 1)) for c in CONDS}
        model_one_minus_alpha = _compute_model_one_minus_alpha_by_condition(
            rhat_rs, score_filter
        )

        rbar = np.zeros_like(robs_m)
        for i in range(n_trials):
            other = np.setdiff1d(np.arange(n_trials), i)
            rbar[i] = np.nanmean(robs_m[other], axis=0)
        ve_psth = _var_explained(rbar, robs_m, axis=(0, 1))

        ccnorm, ccmax, ccabs, ccnorm_unstable = _compute_ccnorm_by_condition(
            robs, rhat_rs, score_filter
        )
        if analysis_grid["align_to_reference"]:
            anchor = selected_intact.get(session_name)
            if anchor is None or not np.array_equal(
                neuron_mask, np.asarray(anchor["neuron_mask"])
            ):
                raise ValueError(
                    f"{session_name}: selected intact metric anchor population mismatch."
                )
            # Every counterfactual must use the exact same data-only ceiling
            # and stability mask as the selected intact trace. Mixing the
            # ablation sweep's 200-split ceiling with the intact evaluator's
            # 500-split ceiling makes CCnorm differ even when CCabs does not.
            ccnorm, ccmax, ccabs, ccnorm_unstable = (
                _renormalize_ccnorm_to_intact_anchor(ccabs, anchor)
            )
            ve["intact"] = np.asarray(anchor["ve_model"]).copy()
            ve_psth = np.asarray(anchor["ve_psth"]).copy()

        for condition in CONDS:
            with np.errstate(divide="ignore", invalid="ignore"):
                identity = ccabs[condition] / ccmax
            identity[ccnorm_unstable] = np.nan
            if not np.allclose(
                ccnorm[condition], identity, rtol=0, atol=1e-12, equal_nan=True
            ):
                raise AssertionError(
                    f"{session_name}/{condition}: CCnorm != CCabs / shared CCmax"
                )

        # Captured count variance on Figure 2's one-bin windows intersected with
        # the twin's valid support. Only the numerator is computed here; the
        # Figure 2 denominators (diag(Crate), diag(Ctotal)) and everything
        # derived from them are attached at load time by `_attach_fig2_derived`,
        # so they track the current decomposition cache without re-inference.
        scored_by_window = {
            cb: compute_matched_captured_variance(
                robs,
                {"psth": rbar, **rhat_m},
                eyepos,
                score_filter,
                count_bins=cb,
            )
            for cb in SCORED_COUNT_BINS
        }
        scored = scored_by_window[PRODUCTION_COUNT_BINS]
        for cb in SCORED_COUNT_BINS:
            s = scored_by_window[cb]
            n_scored = int(np.isfinite(s["var_y"]).sum())
            tag = " (production)" if cb == PRODUCTION_COUNT_BINS else ""
            print(
                f"  captured variance [W={cb}, {cb * 1000 / 120:.1f} ms]{tag}: "
                f"{n_scored}/{n_neurons} units scored on "
                f"{s['n_base_windows']} Figure 2 windows (median "
                f"{int(np.median(s['n_windows']))} model-valid/unit, "
                f"{s['n_units_below_floor']} below floor)"
            )

        # Diagnostic only: the abandoned matched denominator, recorded so the
        # figure can report how far it drifts from Figure 2's estimate. This one
        # needs `dfs`, so it stays inference-side; its drift-vs-Figure-2 ratio is
        # reported by `_attach_fig2_derived` once the denominators are attached.
        matched = estimate_matched_rate_variance(robs, eyepos, score_filter)
        print(
            f"  [diagnostic] matched Crate: "
            f"{int(np.sum(matched['c_rate'] > 0))}/{n_neurons} units positive, "
            f"{matched['n_validity_groups']} validity groups "
            f"({matched['n_validity_groups_excluded']} unusable), median "
            f"{int(np.median(matched['n_close_pairs']))} close pairs"
        )

        example = None
        if session_name == PANEL_B_SESSION:
            matches = np.where(neuron_mask == PANEL_B_NEURON_ID)[0]
            if len(matches):
                ni = int(matches[0])
                dfs_n = score_filter[:, :, ni]
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
            "ve": ve, "ve_psth": ve_psth,
            "ccnorm": ccnorm, "ccabs": ccabs, "ccmax": ccmax,
            "ccnorm_unstable": ccnorm_unstable,
            "captured_variance": scored["captured_variance"],
            "var_residual": scored["var_residual"],
            "matched_var_y": scored["var_y"],
            "matched_n_windows": scored["n_windows"],
            "n_base_windows": scored["n_base_windows"],
            # Same quantities at every screened counting window, computed on the
            # SAME predictions, so the windows can be compared pairwise per unit.
            "scored_by_window": {
                cb: {
                    "captured_variance": s["captured_variance"],
                    "var_residual": s["var_residual"],
                    "var_y": s["var_y"],
                    "n_windows": s["n_windows"],
                    "n_base_windows": s["n_base_windows"],
                    "n_units_below_floor": s["n_units_below_floor"],
                }
                for cb, s in scored_by_window.items()
            },
            "matched_c_rate": matched["c_rate"],
            "matched_c_total": matched["c_total"],
            "matched_n_close_pairs": matched["n_close_pairs"],
            "matched_n_validity_groups": matched["n_validity_groups"],
            "matched_n_validity_groups_excluded": matched[
                "n_validity_groups_excluded"
            ],
            "model_one_minus_alpha": model_one_minus_alpha,
            "femfraction": femfraction,
            "example": example,
        })
        completed_sessions.add(session_name)
        _write_inference_cache_atomic(
            partial_cache_path,
            _inference_cache_payload(results, complete=False),
        )

    completed_sessions = {str(record["session"]) for record in results}
    if completed_sessions != eligible_sessions:
        missing = sorted(eligible_sessions - completed_sessions)
        extra = sorted(completed_sessions - eligible_sessions)
        raise RuntimeError(
            "Figure-3 ablation inference did not complete its exact production "
            f"session set; missing={missing}, extra={extra}. Resumable progress "
            f"is retained at {partial_cache_path}."
        )
    _write_inference_cache_atomic(
        cache_path,
        _inference_cache_payload(results, complete=True),
    )
    print(f"\nCached {len(results)} sessions to {cache_path}")
    return results


def aggregate(results):
    """Flatten per-cell arrays across sessions.

    The population masks are built downstream in `load_ablation_data`:
    `cd_population` (fig2 inclusion: rate > 2 Hz & split-half PSTH R^2) for
    panels C/D and `fem_include` for panel E. `ccmax` (split-half reliability)
    is carried per cell for reference but no longer gates any panel."""
    score_conditions = ["psth", *CONDS]
    required = (
        "explainable_fraction", "captured_variance", "matched_var_y",
        "matched_n_windows", "fig2_c_rate", "fig2_c_total", "matched_c_rate",
    )
    if any(any(key not in r for key in required) for r in results):
        raise RuntimeError(
            "The Figure 3 ablation cache predates the Figure 2-denominator "
            "explainable-variance score. Rebuild it with `--recompute`."
        )

    ve = {c: [] for c in CONDS}
    ccnorm = {c: [] for c in CONDS}
    model_one_minus_alpha = {c: [] for c in CONDS}
    fraction = {c: [] for c in score_conditions}
    captured_variance = {c: [] for c in score_conditions}
    ve_psth, ccmax, alpha = [], [], []
    matched_var_y, matched_n_windows = [], []
    fig2_c_rate, fig2_c_total = [], []
    matched_c_rate, matched_c_total, matched_n_close_pairs = [], [], []
    subjects, sessions = [], []
    for r in results:
        for c in CONDS:
            ve[c].append(r["ve"][c])
            if "ccnorm" in r:
                ccnorm[c].append(r["ccnorm"][c])
            if "model_one_minus_alpha" in r:
                model_one_minus_alpha[c].append(r["model_one_minus_alpha"][c])
        for c in score_conditions:
            fraction[c].append(r["explainable_fraction"][c])
            captured_variance[c].append(r["captured_variance"][c])
        ve_psth.append(r["ve_psth"])
        ccmax.append(r["ccmax"])
        alpha.append(r["alpha"])
        matched_var_y.append(r["matched_var_y"])
        matched_n_windows.append(r["matched_n_windows"])
        fig2_c_rate.append(r["fig2_c_rate"])
        fig2_c_total.append(r["fig2_c_total"])
        matched_c_rate.append(r["matched_c_rate"])
        matched_c_total.append(r["matched_c_total"])
        matched_n_close_pairs.append(r["matched_n_close_pairs"])
        subjects.extend([r["subject"]] * r["n_neurons"])
        sessions.extend([r["session"]] * r["n_neurons"])
    agg = {
        "ve": {c: np.concatenate(ve[c]) for c in CONDS},
        "explainable_fraction": {
            c: np.concatenate(fraction[c]) for c in score_conditions
        },
        "captured_variance": {
            c: np.concatenate(captured_variance[c]) for c in score_conditions
        },
    }
    if all(ccnorm[c] for c in CONDS):
        agg["ccnorm"] = {c: np.concatenate(ccnorm[c]) for c in CONDS}
    if all(model_one_minus_alpha[c] for c in CONDS):
        agg["model_one_minus_alpha"] = {
            c: np.concatenate(model_one_minus_alpha[c]) for c in CONDS
        }
    agg["ve_psth"] = np.concatenate(ve_psth)
    agg["ccmax"] = np.concatenate(ccmax)
    agg["alpha"] = np.concatenate(alpha)
    agg["matched_var_y"] = np.concatenate(matched_var_y)
    agg["matched_n_windows"] = np.concatenate(matched_n_windows)
    agg["fig2_c_rate"] = np.concatenate(fig2_c_rate)
    agg["fig2_c_total"] = np.concatenate(fig2_c_total)
    agg["matched_c_rate"] = np.concatenate(matched_c_rate)
    agg["matched_c_total"] = np.concatenate(matched_c_total)
    agg["matched_n_close_pairs"] = np.concatenate(matched_n_close_pairs)
    # Sensitivity view: the same numerators over the abandoned matched
    # denominator. Reported beside the panel, never plotted.
    agg["explainable_fraction_matched"] = explainable_fraction(
        agg["captured_variance"], agg["matched_c_rate"]
    )
    agg["total_variance_ratio"] = total_variance_ratio(
        agg["matched_var_y"], agg["fig2_c_total"]
    )
    agg["subjects"] = np.array(subjects)
    agg["sessions"] = np.array(sessions)
    return agg


def _attach_fig2_derived(results, window_bins=PRODUCTION_COUNT_BINS):
    """Attach the Figure 2-derived per-session fields, recomputed from the
    CURRENT decomposition cache.

    ``window_bins`` selects the Figure 2 counting window supplying the
    denominators. It must match the window the cached `captured_variance`
    numerator was scored on, or the explainable fraction mixes two windows.

    These fields are pure functions of the session name, `neuron_mask`, the
    Figure 2 covariance cache, and the already-cached `captured_variance` --
    no model and no dataset. Keeping them out of the inference cache is what
    makes a Figure 2 convention change (estimator, weighting, inclusion) cost a
    `generate_figure3.py` run instead of a full GPU inference sweep.

    Mutates and returns ``results``. Ordering is untouched, so the flattened
    cell order assumed by `aggregate` / `_raw_one_minus_alpha` still holds.
    """
    fig2_info_by_session = _load_fig2_alpha_by_session(window_bins=window_bins)
    drifts = []
    for r in results:
        session_name = r["session"]
        n_neurons = int(r["n_neurons"])
        alpha = np.full(n_neurons, np.nan)
        fig2_c_rate = np.full(n_neurons, np.nan)
        fig2_c_total = np.full(n_neurons, np.nan)
        if session_name not in fig2_info_by_session:
            raise KeyError(f"{session_name} is missing from the Figure 2 cache")
        f2 = fig2_info_by_session[session_name]
        for i, nidx in enumerate(r["neuron_mask"]):
            loc = np.where(f2["neuron_mask"] == nidx)[0]
            if len(loc) == 1:
                j = int(loc[0])
                alpha[i] = f2["alpha"][j]
                fig2_c_rate[i] = f2["c_rate"][j]
                fig2_c_total[i] = f2["c_total"][j]
        r["alpha"] = alpha
        r["fig2_c_rate"] = fig2_c_rate
        r["fig2_c_total"] = fig2_c_total
        r["explainable_fraction"] = explainable_fraction(
            r["captured_variance"], fig2_c_rate
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            drift = r["matched_c_rate"] / fig2_c_rate
        if np.any(np.isfinite(drift)):
            drifts.append(np.nanmedian(drift))

    # Sanity figures only, over every cached session and unit -- i.e. BEFORE the
    # fig2 session floor and the C/D population mask. They are deliberately not
    # the reported values: `generate_figure3.py` prints the population-restricted
    # Var(y)/Ctotal_fig2 that the manuscript quotes, and it will read lower.
    ratio = total_variance_ratio(
        np.concatenate([r["matched_var_y"] for r in results]),
        np.concatenate([r["fig2_c_total"] for r in results]),
    )
    print(
        f"Figure 2 derived fields attached for {len(results)} session(s) "
        f"[all cached sessions, pre-floor]: Var(y)/Ctotal_fig2 "
        f"median={np.nanmedian(ratio):.3f}; matched/Figure 2 Crate "
        f"median={np.nanmedian(drifts):.3f}"
    )
    return results


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


def _fig2_cd_population(results):
    """Per-cell bool (aligned to `aggregate`'s flattened cell order): the cell
    passes fig2 inclusion (rate > 2 Hz & split-half PSTH R^2 > 0.10) in the
    aligned covariance cache.

    Panels C/D use this so their population matches fig2 (and the panel-E f_FEM
    population). Ordering mirrors `_raw_one_minus_alpha`: results-order x
    neuron_mask-order.
    """
    covdecomp = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
    if covdecomp not in sys.path:
        sys.path.insert(0, covdecomp)
    import derive
    from data_loading import load_cache as load_aligned_cache
    aligned_by = {a["session"]: a for a in load_aligned_cache()}
    incl = {}
    for a in aligned_by.values():
        nm = np.asarray(a["neuron_mask"])
        rate = np.asarray(a["rate_hz"], float)
        psth = np.asarray(a["psth_r2"], float)
        keep = (np.isfinite(rate) & (rate > derive.MIN_RATE_HZ)
                & np.isfinite(psth) & (psth > derive.MIN_PSTH_R2))
        for o, k in zip(nm, keep):
            incl[(a["session"], int(o))] = bool(k)
    out = []
    for r in results:
        for nid in r["neuron_mask"]:
            out.append(incl.get((r["session"], int(nid)), False))
    return np.asarray(out, dtype=bool)


def select_ablation_example(results):
    """Return the example-neuron payload (pinned PANEL_B session) or None."""
    return next((r["example"] for r in results if r.get("example")), None)


_cached_data = None


def load_ablation_data(recompute=False):
    """Return a dict with flattened per-cell arrays, the `cd_population` /
    `fem_include` masks, and the example-neuron payload. Cached in-process after
    the first call."""
    global _cached_data
    if _cached_data is not None and not recompute:
        return _cached_data

    if CACHE_PATH.exists() and not recompute:
        print(f"Loading cached inference results from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            payload = dill.load(f)
        if isinstance(payload, dict) and "results" in payload:
            results = payload["results"]
            version = payload.get("schema_version")
            if version != INFERENCE_SCHEMA_VERSION:
                raise ValueError(
                    f"{CACHE_PATH} has inference schema_version {version}, "
                    f"expected {INFERENCE_SCHEMA_VERSION}. Re-run inference "
                    f"(load_ablation_data(recompute=True))."
                )
            if payload.get("checkpoint_path") != CHECKPOINT_PATH:
                raise ValueError(
                    f"{CACHE_PATH} was produced from "
                    f"{payload.get('checkpoint_path')}, but CHECKPOINT_PATH is "
                    f"{CHECKPOINT_PATH}. Re-run inference."
                )
        else:  # pre-split cache: a bare list, no provenance recorded
            print("  (legacy cache format: no schema/checkpoint provenance)")
            results = payload
    else:
        results = _run_inference()

    # Figure 2-derived fields are recomputed from the current decomposition
    # cache on every load, never read from the inference cache.
    results = _attach_fig2_derived(results)

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
    # Panels C/D population: fig2 inclusion (rate > 2 Hz & PSTH R^2 > 0.10), so
    # they describe the same cells fig2 (and panel E) report.
    agg["cd_population"] = _fig2_cd_population(results)
    n_cd = int(agg["cd_population"].sum())
    n_excl = int((agg["cd_population"] & ~agg["fem_include"]).sum())
    print(f"Ablation data: {len(results)} sessions, {len(agg['cd_population'])} cells "
          f"({n_cd} in fig2 C/D population); "
          f"FEM-axis excludes {n_excl} C/D cell(s) with 1-alpha out of [0,1]")

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
    pop = data["cd_population"]
    print("\n=== Fig 3 bottom row — ablation cost (fig2 C/D population, single-trial r^2) ===")
    print(f"{'cond':<10}{'subject':<8}{'N':<6}{'cost Δr²':<12}{'intact r²':<12}"
          f"{'cost/intact%':<14}{'cost/gainPSTH%':<15}")
    for cond in ABLATIONS:
        for subj in ["All"] + SUBJECTS:
            m = pop & np.isfinite(data["ve"]["intact"]) & np.isfinite(data["ve"][cond])
            if subj != "All":
                m = m & (data["subjects"] == subj)
            cost = np.median(data["ve"]["intact"][m] - data["ve"][cond][m])
            intact = np.median(data["ve"]["intact"][m])
            gain = np.median(data["ve"]["intact"][m] - data["ve_psth"][m])
            ci = 100 * cost / intact if abs(intact) > 1e-9 else np.nan
            cg = 100 * cost / gain if abs(gain) > 1e-9 else np.nan
            print(f"{cond:<10}{subj:<8}{m.sum():<6}{cost:<+12.4f}{intact:<12.4f}"
                  f"{ci:<14.1f}{cg:<15.1f}")
