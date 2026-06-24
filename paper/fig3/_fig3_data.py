"""Shared data loader for figure 3.

Exposes `load_fig3_data()`, which lazily loads (or recomputes) the digital
twin inference cache, joins the figure 2 α decomposition, and flattens
per-neuron arrays across sessions. The model is only loaded when the
inference cache is missing or `recompute=True` — panels D/E/F can therefore
be iterated without a GPU once the cache exists.
"""
import sys
import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
DT = 1 / 120                # seconds per bin
VALID_TIME_BINS = 120        # max within-trial time bins
MIN_FIX_DUR = 20             # minimum fixation duration (bins)
MIN_TOTAL_SPIKES = 200       # neuron inclusion threshold
CCNORM_N_SPLITS = 500        # split-half iterations for ccnorm
CCMAX_THRESHOLD = 0.85       # reliability threshold for "good" neurons

SUBJECTS = ["Allen", "Logan"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}

# Model checkpoint
CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_SUBDIR = "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian"
EXPERIMENT_SUBDIR = "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
BEST_CKPT = "epoch=374-val_bps_overall=0.6395.ckpt"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_SUBDIR}/{EXPERIMENT_SUBDIR}/{BEST_CKPT}"

DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)

FIG_DIR = FIGURES_DIR / "fig3"
STAT_DIR = STATS_DIR / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = CACHE_DIR / "fig3_digitaltwin.pkl"
# Empirical covariance-decomposition cache (shared package). Per-session schema:
#   sr["windows"][w]["targets"]["full"]["Cpsth"/"Crate"], sr["neuron_mask"].
COVDECOMP_CACHE_PATH = CACHE_DIR / "covdecomp_empirical.pkl"
COVDECOMP_TARGET = "full"


def configure_matplotlib():
    """Apply publication rcParams (PDF type 42, Arial)."""
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def subject_from_session(session_name):
    return session_name.split("_")[0]


def _load_fig2_alpha_by_session():
    """Load empirical α-per-session lookup (only needed during inference).

    Reads the shared covariance-decomposition cache (target='full') and the
    first counting window, mirroring the empirical 1-α the fig2 panels report.
    """
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
        block = sr["windows"][0]["targets"][COVDECOMP_TARGET]  # first counting window
        diag_psth = np.diag(block["Cpsth"])
        diag_rate = np.diag(block["Crate"])
        alpha = np.clip(diag_psth / diag_rate, 0, 1)
        out[sess_name] = {
            "alpha": alpha,
            "neuron_mask": sr["neuron_mask"],
            "subject": subject,
        }
    print(f"  Loaded 1-α for {len(out)} sessions")
    return out


def _run_inference():
    """Load the model and run forward passes for every Allen/Logan session.

    Returns a list of per-session result dicts and writes them to CACHE_PATH.
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

    device = get_free_device()
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")
    print(f"  {len(model.names)} datasets: {model.names}")

    session_results = []
    for dataset_idx in range(len(model.names)):
        session_name = model.names[dataset_idx]
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            print(f"Skipping {session_name} (subject {subject} not in {SUBJECTS})")
            continue
        print(f"\n--- {session_name} ({subject}) [{dataset_idx+1}/{len(model.names)}] ---")

        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

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
        robs_flat = np.asarray(dset['robs'])
        eyepos_flat = np.asarray(dset['eyepos'])

        trials = np.unique(trial_inds)
        NT = len(trials)
        NC = robs_flat.shape[1]
        T = int(psth_inds_flat.max()) + 1

        fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < 1.0

        robs = np.full((NT, T, NC), np.nan)
        rhat = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
        eyepos = np.full((NT, T, 2), np.nan)
        fix_dur = np.full(NT, np.nan)

        stim_lags = np.array(dataset_config['keys_lags']['stim'])

        for itrial in tqdm(range(NT), desc=f"  Inference {session_name}"):
            ix = (trial_inds == trials[itrial]) & fixation
            if not np.any(ix):
                continue
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior = dset['behavior'][ix]
            out = run_model(model, {'stim': stim, 'behavior': behavior},
                            dataset_idx=dataset_idx)
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            rhat[itrial, t_inds] = out['rhat'].detach().cpu().numpy()
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
            eyepos[itrial, t_inds] = eyepos_flat[ix]

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

        neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
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

        # Affine-rescale model predictions to match observed spike counts.
        rhat_flat = rhat_used.reshape(n_trials * n_time, n_neurons)
        robs_flat_used = robs_used.reshape(n_trials * n_time, n_neurons)
        dfs_flat = dfs_used.reshape(n_trials * n_time, n_neurons)
        rhat_rescaled, _ = rescale_rhat(
            torch.from_numpy(robs_flat_used),
            torch.from_numpy(rhat_flat),
            torch.from_numpy(dfs_flat),
            mode='affine',
        )
        rhat_used = rhat_rescaled.reshape(n_trials, n_time, n_neurons).detach().cpu().numpy()

        # ccnorm via split-half (run twice, average, drop unstable).
        ccnorm1, ccabs1, ccmax1, _, _ = ccnorm_split_half_variable_trials(
            robs_used, rhat_used, dfs_used,
            n_splits=CCNORM_N_SPLITS, return_components=True, rng=42,
        )
        ccnorm2, ccabs2, ccmax2, _, _ = ccnorm_split_half_variable_trials(
            robs_used, rhat_used, dfs_used,
            n_splits=CCNORM_N_SPLITS, return_components=True, rng=43,
        )
        unstable = (ccnorm1 - ccnorm2) ** 2 > 0.01
        ccnorm = 0.5 * (ccnorm1 + ccnorm2)
        ccabs = 0.5 * (ccabs1 + ccabs2)
        ccmax = 0.5 * (ccmax1 + ccmax2)
        ccnorm[unstable] = np.nan

        rhat_masked = rhat_used.copy()
        robs_masked = robs_used.copy()
        rhat_masked[dfs_used == 0] = np.nan
        robs_masked[dfs_used == 0] = np.nan

        rhat_mean = np.nanmean(rhat_masked, axis=0)
        robs_mean = np.nanmean(robs_masked, axis=0)
        n_valid = np.nansum(dfs_used, axis=0)

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

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {CACHE_PATH}")
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

    good = ccmax > CCMAX_THRESHOLD
    print(f"Good neurons (ccmax > {CCMAX_THRESHOLD}): {good.sum()}")

    _cached_data = {
        "session_results": session_results,
        "rhos": rhos, "ccnorm": ccnorm, "ccmax": ccmax,
        "ve_model": ve_model, "ve_psth": ve_psth, "alpha": alpha,
        "subjects": subjects, "good": good,
        "valid_indices": valid_indices,
        "all_rhat_mean": all_rhat_mean,
        "all_robs_mean": all_robs_mean,
        "all_trace_neuron_session": all_trace_neuron_session,
    }
    return _cached_data
