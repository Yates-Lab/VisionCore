"""Shared data loader for figure 3.

Exposes `load_fig3_data()`, which lazily loads (or recomputes) the digital
twin inference cache, joins the figure 2 α decomposition, and flattens
per-neuron arrays across sessions. The model is only loaded when the
inference cache is missing or `recompute=True` — panels D/E/F can therefore
be iterated without a GPU once the cache exists.
"""
import os
import sys
from pathlib import Path
import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
DT = 1 / 120                # seconds per bin
VALID_TIME_BINS = 120        # max within-trial time bins
MIN_FIX_DUR = 20             # minimum fixation duration (bins)
# Neuron inclusion threshold, in total spikes over the inference trials.
#
# This was lowered to 0 on 2026-08-04 to match fig2's `align_fixrsvp_trials`,
# recovering the 28 of fig2's 1022 analyzed cells (2.7%) that it alone dropped
# from the fig3 population. It was reverted to 200 the same day, because the
# recovered cells are not free: panel D scores every cell in a session on one
# *shared* window set, built by `_fig3_explainable_variance.figure2_valid_mask`
# as `np.isfinite(robs).all(axis=2)` -- a bin survives only where every unit in
# the array is finite. A cell that was isolated for only part of a session
# therefore deletes its absent bins for all of its neighbours, and low-spike
# cells are exactly the cells with that pattern.
#
# Measured over the two caches: 10 of 24 sessions lost base windows, 8 of them
# by more than a quarter (Allen_2022-04-01 1260 -> 539; Allen_2022-03-30
# 1429 -> 679; Logan_2020-02-28 5034 -> 2626 on just 2 added units), and the
# per-session median windows/unit equals n_base_windows, so every cell in an
# affected session takes the full hit. That cost up to half of panel D's
# scored windows to gain 25 panel C cells.
#
# The superset invariant this was meant to serve (fig2_analyzed subset of
# twin_readout, TWIN_IMPROVEMENTS item 5) holds at the readout level regardless
# of this analysis threshold, so nothing depends on it being 0. Fixing the cost
# properly means making the base mask per-unit, which changes the fig2/fig3
# shared-window contract and needs its own verification pass.
MIN_TOTAL_SPIKES = 200
CCNORM_N_SPLITS = 500        # split-half iterations for ccnorm

SUBJECTS = ["Allen", "Logan"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}

# Model checkpoint
CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_SUBDIR = "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian"
EXPERIMENT_SUBDIR = "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
BEST_CKPT = "epoch=374-val_bps_overall=0.6395.ckpt"
CHECKPOINT_PATH = os.environ.get(
    "FIG3_TWIN_CHECKPOINT",
    f"{CHECKPOINT_DIR}/{CHECKPOINT_SUBDIR}/{EXPERIMENT_SUBDIR}/{BEST_CKPT}",
)

DATASET_CONFIGS_PATH = os.environ.get(
    "FIG3_DATASET_CONFIGS",
    str(VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"),
)

FIG_DIR = FIGURES_DIR / "fig3"
STAT_DIR = STATS_DIR / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = Path(
    os.environ.get("FIG3_CACHE_PATH", str(CACHE_DIR / "fig3_digitaltwin.pkl"))
)
REFERENCE_CACHE_PATH = Path(
    os.environ.get(
        "FIG3_REFERENCE_CACHE",
        str(VISIONCORE_ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"),
    )
)
# Empirical covariance-decomposition cache (shared package). Per-session schema:
#   sr["windows"][w]["targets"]["full"]["Cpsth"/"Crate"], sr["neuron_mask"].
COVDECOMP_CACHE_PATH = CACHE_DIR / "covdecomp_empirical.pkl"
COVDECOMP_TARGET = "full"
# Derived (floored) fig2 bundle: `session_names` is the population fig2 actually
# reports, after the >=10-analyzed-unit session floor (covariance_decomposition/
# derive.py MIN_SESSION_UNITS). Used to keep fig3's C/D/E population identical.
COVDECOMP_DERIVED_CACHE_PATH = CACHE_DIR / "covdecomp_derived.pkl"


def configure_matplotlib():
    """Apply publication rcParams (PDF type 42, Arial)."""
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def subject_from_session(session_name):
    return session_name.split("_")[0]


def figure3_analysis_grid(dataset_config):
    """Describe how stored samples map onto Figure 3's 120-Hz count grid.

    ``sampling.target_rate`` is the rate of the arrays exposed by the dataset.
    A native-input/120-Hz-supervision model already stores a two-frame count at
    each configured supervision endpoint. A genuinely native-240-Hz model has
    no such supervision block, so Figure 3 must sum two model outputs and two
    observed counts itself.
    """
    sampling = dataset_config.get("sampling") or {}
    stored_rate = int(sampling.get("target_rate", round(1.0 / DT)))
    supervision = dataset_config.get("supervision") or {}
    analysis_rate = int(supervision.get("target_rate", round(1.0 / DT)))
    if stored_rate <= 0 or analysis_rate <= 0 or stored_rate % analysis_rate:
        raise ValueError(
            f"Invalid stored/analysis rates: stored={stored_rate}, "
            f"analysis={analysis_rate}."
        )
    factor = stored_rate // analysis_rate
    phase = int(supervision.get("phase", factor - 1))
    if factor < 1 or not 0 <= phase < factor:
        raise ValueError(
            f"Invalid Figure-3 block factor/phase: factor={factor}, phase={phase}."
        )
    return {
        "stored_rate": stored_rate,
        "analysis_rate": analysis_rate,
        "factor": factor,
        "phase": phase,
        "sum_native_counts": bool(factor > 1 and not supervision),
        "align_to_reference": bool(factor > 1),
    }


def analysis_endpoint_mask_and_psth(dataset_config, psth_inds, trial_inds=None):
    """Return analysis endpoints and PSTH bins on Figure 3's 120-Hz grid.

    Legacy twins globally downsample the whole dataset to 120 Hz, so every
    stored sample is already an analysis endpoint.  Native-240-Hz twins keep
    stimulus/behavior at 240 Hz. They either supervise on causal 120-Hz count
    pairs or predict each 240-Hz count; both cases use the same pair endpoints,
    while the latter is summed explicitly downstream.
    """
    psth = np.asarray(psth_inds).ravel().astype(np.int64, copy=False)
    grid = figure3_analysis_grid(dataset_config)
    factor = int(grid["factor"])
    if factor == 1:
        keep = np.ones(psth.size, dtype=bool)
        return keep, psth
    phase = int(grid["phase"])
    endpoints = np.arange(psth.size, dtype=np.int64)
    keep = endpoints % factor == phase

    # The supervised count at an endpoint is the causal block ending there,
    # but Figure 3 names that block by its *start* coordinate.  Using the
    # endpoint PSTH coordinate works only when every trial happens to begin on
    # the same global sampling phase.  Real FixRSVP trials have both parities,
    # which otherwise produces a trial-dependent one-bin displacement between
    # native-240-Hz twins and the canonical 120-Hz Figure-3 cache.
    block_start = endpoints - factor + 1
    keep &= block_start >= 0
    if trial_inds is not None:
        trial = np.asarray(trial_inds).ravel()
        if trial.shape != psth.shape:
            raise ValueError(
                f"trial_inds shape {trial.shape} does not match psth_inds {psth.shape}."
            )
        valid = keep.copy()
        valid[keep] = trial[block_start[keep]] == trial[endpoints[keep]]
        keep = valid

    analysis_psth = np.floor_divide(psth, factor)
    analysis_psth[keep] = np.floor_divide(psth[block_start[keep]], factor)
    return keep, analysis_psth


def analysis_endpoint_block_mean(dataset_config, values, endpoint_mask):
    """Average native-rate continuous covariates over 120-Hz blocks.

    The legacy Figure-3 data path first average-pools eye position from 240 Hz
    to 120 Hz. Native-rate twins keep the two source samples in memory and put
    their causal count target at the block endpoint, so using the raw endpoint
    eye position would change both the fixation mask and the trajectory used by
    the covariance decomposition. Return a shape-preserving array whose valid
    endpoints contain the same block mean as the legacy downsampled path.

    ``endpoint_mask`` must come from :func:`analysis_endpoint_mask_and_psth`;
    that helper already rejects incomplete and cross-trial blocks.
    """
    values = np.asarray(values)
    endpoint_mask = np.asarray(endpoint_mask, dtype=bool).ravel()
    if values.ndim == 0 or values.shape[0] != endpoint_mask.size:
        raise ValueError(
            f"values leading dimension {values.shape if values.ndim else ()} "
            f"does not match endpoint mask length {endpoint_mask.size}."
        )
    factor = int(figure3_analysis_grid(dataset_config)["factor"])
    if factor == 1:
        return values.copy()
    endpoints = np.flatnonzero(endpoint_mask)
    if not len(endpoints) or factor == 1:
        return values.copy()
    offsets = np.arange(factor - 1, -1, -1, dtype=np.int64)
    block_indices = endpoints[:, None] - offsets[None, :]
    if block_indices.min(initial=0) < 0:
        raise ValueError("endpoint_mask contains an incomplete supervision block.")
    # Continuous covariates are normally floating point, but promote integer
    # inputs so this helper cannot silently truncate a fractional block mean.
    averaged = values.astype(np.result_type(values.dtype, np.float64), copy=True)
    averaged[endpoints] = values[block_indices].mean(axis=1)
    return averaged


def analysis_endpoint_block_sum(dataset_config, values, endpoint_mask):
    """Sum genuinely native counts/predictions into 120-Hz causal blocks.

    For native-input models trained with 120-Hz supervision, the endpoint is
    already a block count and is returned unchanged. Only unsupervised
    native-240-Hz outputs are summed here.
    """
    values = np.asarray(values)
    endpoint_mask = np.asarray(endpoint_mask, dtype=bool).ravel()
    if values.ndim == 0 or values.shape[0] != endpoint_mask.size:
        raise ValueError("values and endpoint_mask must share their leading dimension.")
    grid = figure3_analysis_grid(dataset_config)
    if not grid["sum_native_counts"]:
        return values.copy()
    factor = int(grid["factor"])
    endpoints = np.flatnonzero(endpoint_mask)
    offsets = np.arange(factor - 1, -1, -1, dtype=np.int64)
    block_indices = endpoints[:, None] - offsets[None, :]
    summed = values.copy()
    summed[endpoints] = values[block_indices].sum(axis=1)
    return summed


def analysis_endpoint_block_filter(dataset_config, values, endpoint_mask):
    """Require every native count in a 120-Hz block to be data-valid."""
    values = np.asarray(values)
    endpoint_mask = np.asarray(endpoint_mask, dtype=bool).ravel()
    if values.ndim == 0 or values.shape[0] != endpoint_mask.size:
        raise ValueError("values and endpoint_mask must share their leading dimension.")
    grid = figure3_analysis_grid(dataset_config)
    if not grid["sum_native_counts"]:
        return values.copy()
    factor = int(grid["factor"])
    endpoints = np.flatnonzero(endpoint_mask)
    offsets = np.arange(factor - 1, -1, -1, dtype=np.int64)
    block_indices = endpoints[:, None] - offsets[None, :]
    valid = np.isfinite(values[block_indices]) & (values[block_indices] != 0)
    filtered = values.copy()
    filtered[endpoints] = valid.all(axis=1).astype(values.dtype)
    return filtered


def analysis_model_indices(dataset_config, endpoint_mask, minimum_index):
    """Return scored endpoints and ordered model rows needed for each block."""
    endpoint_mask = np.asarray(endpoint_mask, dtype=bool).ravel()
    endpoints = np.flatnonzero(endpoint_mask)
    grid = figure3_analysis_grid(dataset_config)
    factor = int(grid["factor"])
    if grid["sum_native_counts"]:
        offsets = np.arange(factor - 1, -1, -1, dtype=np.int64)
        blocks = endpoints[:, None] - offsets[None, :]
        endpoints = endpoints[blocks[:, 0] >= int(minimum_index)]
        blocks = endpoints[:, None] - offsets[None, :]
        return endpoints, blocks.reshape(-1)
    endpoints = endpoints[endpoints >= int(minimum_index)]
    return endpoints, endpoints.copy()


def analysis_reduce_model_output(dataset_config, prediction, n_endpoints):
    """Sum ordered true-240 model outputs, or preserve endpoint predictions."""
    prediction = np.asarray(prediction)
    grid = figure3_analysis_grid(dataset_config)
    if grid["sum_native_counts"]:
        factor = int(grid["factor"])
        expected = int(n_endpoints) * factor
        if prediction.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} native prediction rows, got {prediction.shape[0]}."
            )
        return prediction.reshape(int(n_endpoints), factor, *prediction.shape[1:]).sum(axis=1)
    if prediction.shape[0] != int(n_endpoints):
        raise ValueError(
            f"Expected {int(n_endpoints)} endpoint rows, got {prediction.shape[0]}."
        )
    return prediction.copy()


def load_reference_sessions(path=REFERENCE_CACHE_PATH):
    """Load the canonical Ryan/Figure-3 observation frame by session."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical Figure-3 reference cache does not exist: {path}"
        )
    with path.open("rb") as stream:
        rows = dill.load(stream)
    return {row["session"]: row for row in rows}


def align_native_trial_arrays_to_reference(
    robs,
    predictions,
    reference,
    *,
    robs_key="robs_used",
    dfs_key="dfs_used",
    label="native twin",
):
    """Put native-rate predictions on an already validated analysis frame.

    Native twins retain a different history/data-filter support even after
    their causal count pairs have been assigned the correct 120-Hz labels.
    Model comparisons must not let that support change the observed trials,
    cells, or reliability denominator.  This gate checks the overlapping spike
    counts exactly, selects the reference neuron population, masks predictions
    to its finite support, and returns the canonical observations/filters.
    """
    robs = np.asarray(robs)
    ref_robs = np.asarray(reference[robs_key])
    neuron_mask = np.asarray(reference["neuron_mask"], dtype=np.int64)
    if robs.ndim != 3 or ref_robs.ndim != 3:
        raise ValueError(f"{label}: expected trial x time x neuron observations.")
    if robs.shape[:2] != ref_robs.shape[:2]:
        raise ValueError(
            f"{label}: native trial/time shape {robs.shape[:2]} does not match "
            f"reference {ref_robs.shape[:2]}."
        )
    if neuron_mask.max(initial=-1) >= robs.shape[2]:
        raise ValueError(
            f"{label}: reference neuron index exceeds native width {robs.shape[2]}."
        )
    native_robs = robs[:, :, neuron_mask]
    if native_robs.shape != ref_robs.shape:
        raise ValueError(
            f"{label}: selected native observations {native_robs.shape} do not "
            f"match reference {ref_robs.shape}."
        )
    overlap = np.isfinite(native_robs) & np.isfinite(ref_robs)
    max_abs = (
        float(np.max(np.abs(native_robs[overlap] - ref_robs[overlap])))
        if overlap.any()
        else float("inf")
    )
    if max_abs != 0:
        raise ValueError(
            f"{label}: native/reference spike alignment failed (max abs {max_abs})."
        )

    finite = np.isfinite(ref_robs)

    def _select(prediction):
        prediction = np.asarray(prediction)
        if prediction.shape[:2] != robs.shape[:2] or prediction.shape[2] != robs.shape[2]:
            raise ValueError(
                f"{label}: prediction shape {prediction.shape} is incompatible "
                f"with native observations {robs.shape}."
            )
        selected = prediction[:, :, neuron_mask]
        return np.where(finite, selected, np.nan)

    if isinstance(predictions, dict):
        aligned_predictions = {key: _select(value) for key, value in predictions.items()}
    else:
        aligned_predictions = _select(predictions)
    if dfs_key is not None and dfs_key in reference:
        dfs = np.asarray(reference[dfs_key]).copy()
    else:
        dfs = finite.astype(np.float32)
    return ref_robs.copy(), aligned_predictions, dfs, neuron_mask


def _load_fig2_alpha_by_session(window_bins=None):
    """Load Figure 2 per-unit rate-variance quantities for inference.

    Reads the shared covariance-decomposition cache (target='full') at the
    ``window_bins``-bin counting window. The returned Ctotal and Crate diagonals
    let Figure 3 verify and normalize its matched single-trial variance
    calculation, so this window must be the one Figure 3 scores its numerator on.
    """
    if window_bins is None:
        covd = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
        if covd not in sys.path:
            sys.path.insert(0, covd)
        from fig3_windows import FIG3_SINGLETRIAL_WINDOW_BINS
        window_bins = FIG3_SINGLETRIAL_WINDOW_BINS
    if not COVDECOMP_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Covariance-decomposition cache not found at {COVDECOMP_CACHE_PATH}. "
            "Run `uv run python paper/covariance_decomposition/decompose.py` first."
        )
    print(f"Loading covariance-decomposition cache from {COVDECOMP_CACHE_PATH}")
    with open(COVDECOMP_CACHE_PATH, "rb") as f:
        session_results = dill.load(f)

    out = {}
    for sr in session_results:
        sess_name = sr["session"]
        subject = sr["subject"]
        if subject not in SUBJECTS:
            continue
        window = next((w for w in sr["windows"]
                       if int(w["window_bins"]) == int(window_bins)), None)
        if window is None:
            raise ValueError(
                f"{sess_name} has no {window_bins}-bin Figure 2 counting window "
                f"(available: {[int(w['window_bins']) for w in sr['windows']]})"
            )
        block = window["targets"][COVDECOMP_TARGET]
        diag_psth = np.diag(block["Cpsth"])
        diag_rate = np.diag(block["Crate"])
        diag_total = np.diag(window["Ctotal"])
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.clip(diag_psth / diag_rate, 0, 1)
        out[sess_name] = {
            "alpha": alpha,
            "c_rate": diag_rate,
            "c_total": diag_total,
            "neuron_mask": sr["neuron_mask"],
            "subject": subject,
        }
    print(f"  Loaded {window_bins}-bin Figure 2 variances for {len(out)} sessions")
    return out


def _load_fig2_included_sessions():
    """Return the set of session names fig2 reports, i.e. the population after
    fig2's >=10-analyzed-unit session floor.

    Reads the derived (floored) covariance-decomposition bundle's
    `session_names`. Falls back to computing it via the shared loader if the
    derived cache is missing. Fig3 intersects its sessions with this set so
    panels C/D/E describe the exact same population fig2 does.
    """
    if COVDECOMP_DERIVED_CACHE_PATH.exists():
        with open(COVDECOMP_DERIVED_CACHE_PATH, "rb") as f:
            bundle = dill.load(f)
        return set(bundle["session_names"])

    # Fallback: derive it (heavier; only if the cache was never built).
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "covariance_decomposition"))
    from derive import load_empirical_data
    return set(load_empirical_data()["session_names"])


def _run_inference(session_filter=None, cache_path=CACHE_PATH):
    """Load the model and run forward passes for every Allen/Logan session.

    Returns per-session results and writes them to ``cache_path``.
    ``session_filter`` supports an isolated alignment smoke test before a full
    production sweep.
    """
    import torch
    from tqdm import tqdm
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import (
        load_single_dataset,
        run_model,
        rescale_rhat,
        ccnorm_split_half_variable_trials,
    )

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    fig2_alpha_by_session = _load_fig2_alpha_by_session()

    # ``get_free_device`` queries physical GPU indices through nvidia-smi.  A
    # caller using CUDA_VISIBLE_DEVICES can therefore receive an index that is
    # invalid inside the remapped process.  Production regeneration may pin a
    # physical device explicitly with FIG3_GPU; the default remains automatic.
    device = get_free_device(os.environ.get("FIG3_GPU"))
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")
    print(f"  {len(model.names)} datasets: {model.names}")
    # Loaded lazily only when a native-supervision config is encountered. This
    # keeps a first-ever legacy Ryan cache build independent of a pre-existing
    # reference file.
    native_reference = None

    session_results = []
    for dataset_idx in range(len(model.names)):
        session_name = model.names[dataset_idx]
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            print(f"Skipping {session_name} (subject {subject} not in {SUBJECTS})")
            continue
        if session_filter is not None and session_name not in session_filter:
            continue
        print(f"\n--- {session_name} ({subject}) [{dataset_idx+1}/{len(model.names)}] ---")

        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        analysis_grid = figure3_analysis_grid(dataset_config)
        if analysis_grid["align_to_reference"] and native_reference is None:
            native_reference = load_reference_sessions()

        try:
            fixrsvp_inds = torch.cat([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp'),
            ], dim=0)
        except (ValueError, KeyError):
            print("  Skipping: no fixrsvp data")
            continue

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

        trials = np.unique(trial_inds)
        NT = len(trials)
        NC = robs_flat.shape[1]
        T = int(psth_inds_analysis[analysis_endpoints].max()) + 1

        fixation = np.hypot(
            eyepos_analysis[:, 0], eyepos_analysis[:, 1]
        ) < 1.0

        robs = np.full((NT, T, NC), np.nan)
        rhat = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
        eyepos = np.full((NT, T, 2), np.nan)
        fix_dur = np.full(NT, np.nan)

        stim_lags = np.array(dataset_config['keys_lags']['stim'])

        for itrial in tqdm(range(NT), desc=f"  Inference {session_name}"):
            ix_obs = (trial_inds == trials[itrial]) & fixation & analysis_endpoints
            if not np.any(ix_obs):
                continue
            t_obs = psth_inds_analysis[ix_obs].astype(int)
            fix_dur[itrial] = len(t_obs)
            robs[itrial, t_obs] = robs_analysis[ix_obs]
            dfs[itrial, t_obs] = dfs_analysis[ix_obs]
            eyepos[itrial, t_obs] = eyepos_analysis[ix_obs]

            # Never let negative NumPy lag indices wrap into the end of the
            # recording. The canonical reference filter excludes this prefix,
            # but leaving wrapped predictions in memory obscures parity audits.
            endpoint_indices, model_indices = analysis_model_indices(
                dataset_config, ix_obs, int(stim_lags.max(initial=0))
            )
            if not len(endpoint_indices):
                continue
            stim_lag_indices = model_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior = dset['behavior'][model_indices]
            batch = {'stim': stim, 'behavior': behavior}
            if 'output_behavior' in dset:
                batch['output_behavior'] = dset['output_behavior'][model_indices]
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
            t_inds = psth_inds_analysis[endpoint_indices].astype(int)
            rhat[itrial, t_inds] = prediction

        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < 10:
            print(f"  Skipping: only {good_trials.sum()} good trials")
            continue

        robs = robs[good_trials]
        rhat = rhat[good_trials]
        dfs = dfs[good_trials]
        eyepos = eyepos[good_trials]

        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[:, iix]
        rhat = rhat[:, iix]
        dfs = dfs[:, iix]
        eyepos = eyepos[:, iix]

        if analysis_grid["align_to_reference"]:
            if native_reference is None or session_name not in native_reference:
                # The canonical Figure-3 cache defines the displayed
                # population.  A recording can contain FixRSVP trials and be
                # present in a newer training manifest without belonging to
                # that population (Logan_2020-01-06 is the current example).
                # There is no reference neuron/time intersection to align in
                # that case, so exclude the session rather than aborting the
                # complete production rebuild.
                print(
                    "  Skipping: missing from canonical Figure-3 reference "
                    "cache"
                )
                continue
            robs_used, rhat_used, dfs_used, neuron_mask = (
                align_native_trial_arrays_to_reference(
                    robs,
                    rhat,
                    native_reference[session_name],
                    label=session_name,
                )
            )
        else:
            neuron_mask = np.where(
                np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES
            )[0]
            if len(neuron_mask) < 3:
                print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
                continue
            robs_used = robs[:, :, neuron_mask]
            rhat_used = rhat[:, :, neuron_mask]
            dfs_used = dfs[:, :, neuron_mask]
        # Eye trajectory aligned to the same trials/bins (no neuron axis) and a
        # per-(trial, bin) validity mask, used by the covariance decomposition
        # in the panel-D simulation control.
        eyepos_used = eyepos
        valid_mask = np.isfinite(eyepos_used).all(axis=-1)

        n_trials, n_time, n_neurons = robs_used.shape
        print(f"  {n_trials} trials, {n_time} time bins, {n_neurons} neurons")

        # One model-independent support for affine fitting, CCnorm, PSTHs, and
        # variance metrics. In particular, NaN-valued numeric data filters are
        # invalid rather than truthy.
        ccnorm_support = (
            np.isfinite(robs_used)
            & np.isfinite(dfs_used)
            & (dfs_used != 0)
        )
        missing_prediction = ccnorm_support & ~np.isfinite(rhat_used)
        if missing_prediction.any():
            raise RuntimeError(
                f"{session_name}: {int(missing_prediction.sum())} predictions "
                "missing on Figure-3 support"
            )

        # Affine-rescale model predictions to match observed spike counts.
        rhat_flat = rhat_used.reshape(n_trials * n_time, n_neurons)
        robs_flat_used = robs_used.reshape(n_trials * n_time, n_neurons)
        dfs_flat = ccnorm_support.astype(np.float32).reshape(
            n_trials * n_time, n_neurons
        )
        rhat_rescaled, _ = rescale_rhat(
            torch.from_numpy(robs_flat_used),
            torch.from_numpy(rhat_flat),
            torch.from_numpy(dfs_flat),
            mode='affine',
        )
        rhat_used = rhat_rescaled.reshape(n_trials, n_time, n_neurons).detach().cpu().numpy()

        # CCnorm via split-half (run twice, average, drop unstable).
        missing_prediction = ccnorm_support & ~np.isfinite(rhat_used)
        if missing_prediction.any():
            raise RuntimeError(
                f"{session_name}: {int(missing_prediction.sum())} predictions "
                "missing on Figure-3 CCnorm support"
            )
        ccnorm1, ccabs1, ccmax1, _, _ = ccnorm_split_half_variable_trials(
            robs_used, rhat_used, ccnorm_support,
            n_splits=CCNORM_N_SPLITS, return_components=True, rng=42,
        )
        ccnorm2, ccabs2, ccmax2, _, _ = ccnorm_split_half_variable_trials(
            robs_used, rhat_used, ccnorm_support,
            n_splits=CCNORM_N_SPLITS, return_components=True, rng=43,
        )
        if not np.allclose(ccabs1, ccabs2, rtol=0, atol=1e-12, equal_nan=True):
            raise AssertionError("CCabs changed across data-only split-half seeds")
        # Unit eligibility must depend only on the recorded responses.  A gate
        # on CCnorm disagreement is model-dependent because the shared CCmax
        # uncertainty is multiplied by each model's own CCabs numerator.
        unstable = (ccmax1 - ccmax2) ** 2 > 0.01
        ccabs = 0.5 * (ccabs1 + ccabs2)
        ccmax = 0.5 * (ccmax1 + ccmax2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ccnorm = ccabs / ccmax
        ccnorm[unstable] = np.nan
        finite_cc = np.isfinite(ccnorm)
        if finite_cc.any() and not np.allclose(
            ccnorm[finite_cc],
            (ccabs / ccmax)[finite_cc],
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("CCnorm != CCabs / CCmax")

        valid_samples = (
            np.isfinite(robs_used)
            & np.isfinite(rhat_used)
            & np.isfinite(dfs_used)
            & (dfs_used != 0)
        )
        rhat_masked = np.where(valid_samples, rhat_used, np.nan)
        robs_masked = np.where(valid_samples, robs_used, np.nan)

        rhat_mean = np.nanmean(rhat_masked, axis=0)
        robs_mean = np.nanmean(robs_masked, axis=0)
        n_valid = valid_samples.sum(axis=0)

        rhos = np.array([
            np.corrcoef(
                rhat_mean[n_valid[:, cc] > 10, cc],
                robs_mean[n_valid[:, cc] > 10, cc],
            )[0, 1]
            for cc in range(n_neurons)
        ])

        def var_explained(pred, true, axis=None):
            residuals = pred - true
            return 1 - np.nanvar(residuals, axis=axis) / np.nanvar(true, axis=axis)

        # Leave-one-out PSTH baseline.
        rbar = np.zeros_like(robs_masked)
        for i in range(n_trials):
            other = np.setdiff1d(np.arange(n_trials), i)
            rbar[i] = np.nanmean(robs_masked[other], axis=0)

        ve_model = var_explained(rhat_masked, robs_masked, axis=(0, 1))
        ve_psth = var_explained(rbar, robs_masked, axis=(0, 1))

        alpha_vec = np.full(n_neurons, np.nan)
        if session_name in fig2_alpha_by_session:
            fig2_info = fig2_alpha_by_session[session_name]
            fig2_nmask = fig2_info["neuron_mask"]
            fig2_alpha = fig2_info["alpha"]
            for i, nidx in enumerate(neuron_mask):
                loc = np.where(fig2_nmask == nidx)[0]
                if len(loc) == 1:
                    alpha_vec[i] = fig2_alpha[loc[0]]
        else:
            print(f"  Warning: session {session_name} not in figure 2 cache")

        session_results.append({
            "session": session_name,
            "subject": subject,
            "neuron_mask": neuron_mask,
            "n_trials": n_trials,
            "n_time": n_time,
            "n_neurons": n_neurons,
            "rhat_mean": rhat_mean,
            "robs_mean": robs_mean,
            "robs_used": robs_used,
            "rhat_used": rhat_used,
            "dfs_used": dfs_used,
            "eyepos_used": eyepos_used,
            "valid_mask": valid_mask,
            "rhos": rhos,
            "ccnorm": ccnorm,
            "ccabs": ccabs,
            "ccmax": ccmax,
            "ve_model": ve_model,
            "ve_psth": ve_psth,
            "alpha": alpha_vec,
        })

        print(f"  ccnorm: median={np.nanmedian(ccnorm):.3f}, "
              f"rho: median={np.nanmedian(rhos):.3f}")

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {cache_path}")
    return session_results


_cached_data = None


def load_fig3_data(recompute=False):
    """Return a dict of per-neuron flattened arrays and per-session results.

    Cached in-process after the first call (within one Python session) so
    repeated panel renders share the same arrays.
    """
    global _cached_data
    if _cached_data is not None and not recompute:
        return _cached_data

    if CACHE_PATH.exists() and not recompute:
        print(f"Loading cached results from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            session_results = dill.load(f)
    else:
        session_results = _run_inference()

    # The exact native-rate evaluator intentionally stores model/observation
    # metrics only. Figure 3 also carries Ryan's model-independent Figure-2
    # alpha annotation; recover it from the canonical cache when importing an
    # evaluator trace bundle as the production digital-twin cache.
    if any("alpha" not in row for row in session_results):
        reference = load_reference_sessions()
        for row in session_results:
            if "alpha" in row:
                continue
            ref = reference.get(row["session"])
            if ref is None or "alpha" not in ref:
                raise KeyError(
                    f"Missing canonical alpha annotation for {row['session']}."
                )
            if not np.array_equal(row["neuron_mask"], ref["neuron_mask"]):
                raise ValueError(
                    f"Canonical alpha population does not match {row['session']}."
                )
            row["alpha"] = np.asarray(ref["alpha"]).copy()

    all_rhos, all_ccnorm, all_ccmax = [], [], []
    all_ve_model, all_ve_psth, all_alpha = [], [], []
    all_subjects, all_session_idx = [], []
    all_rhat_mean, all_robs_mean = [], []
    all_trace_neuron_session = []

    for i, sr in enumerate(session_results):
        n = sr["n_neurons"]
        all_rhos.append(sr["rhos"])
        all_ccnorm.append(sr["ccnorm"])
        all_ccmax.append(sr["ccmax"])
        all_ve_model.append(sr["ve_model"])
        all_ve_psth.append(sr["ve_psth"])
        all_alpha.append(sr["alpha"])
        all_subjects.extend([sr["subject"]] * n)
        all_session_idx.extend([i] * n)
        for j in range(n):
            all_rhat_mean.append(sr["rhat_mean"][:, j])
            all_robs_mean.append(sr["robs_mean"][:, j])
            all_trace_neuron_session.append((i, j))

    rhos = np.concatenate(all_rhos)
    ccnorm = np.concatenate(all_ccnorm)
    ccmax = np.concatenate(all_ccmax)
    ve_model = np.concatenate(all_ve_model)
    ve_psth = np.concatenate(all_ve_psth)
    alpha = np.concatenate(all_alpha)
    subjects = np.array(all_subjects)

    valid = np.isfinite(rhos)
    rhos = rhos[valid]
    ccnorm = ccnorm[valid]
    ccmax = ccmax[valid]
    ve_model = ve_model[valid]
    ve_psth = ve_psth[valid]
    alpha = alpha[valid]
    subjects = subjects[valid]
    valid_indices = np.where(valid)[0]

    print(f"\nTotal neurons: {len(rhos)} ({(subjects == 'Allen').sum()} Allen, "
          f"{(subjects == 'Logan').sum()} Logan)")

    _cached_data = {
        "session_results": session_results,
        "rhos": rhos, "ccnorm": ccnorm, "ccmax": ccmax,
        "ve_model": ve_model, "ve_psth": ve_psth, "alpha": alpha,
        "subjects": subjects,
        "valid_indices": valid_indices,
        "all_rhat_mean": all_rhat_mean,
        "all_robs_mean": all_robs_mean,
        "all_trace_neuron_session": all_trace_neuron_session,
    }
    return _cached_data
