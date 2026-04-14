# %% Imports and configuration
"""
Figure 3: Digital Twin Performance

Flat, cell-based script. Each cell computes stats and plots its panel(s).
Run interactively with IPython (#%% cells) or as a script with uv run.

Panels:
  A  Architecture schematic (fig3-schematic.svg, placed manually)
  B  Example neuron PSTH overlay (observed + twin)
  C  Single-trial rasters: observed | twin (same neuron as B)
  D  Histogram of normalized correlation (ccnorm)
  E  Single-trial r^2 scatter: model vs PSTH
  F  Improvement over PSTH vs FEM modulation (1-alpha)
"""
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d
import dill
import torch
from tqdm import tqdm

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR

# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
RECOMPUTE = False  # set True to rerun model inference from scratch

DT = 1 / 120                # seconds per bin
VALID_TIME_BINS = 120        # max within-trial time bins
MIN_FIX_DUR = 20             # minimum fixation duration (bins)
MIN_TOTAL_SPIKES = 200       # neuron inclusion threshold
CCNORM_N_SPLITS = 500        # split-half iterations for ccnorm
CCMAX_THRESHOLD = 0.80       # reliability threshold for "good" neurons
PANEL_B_MIN_DUR_S = 0.5      # minimum trial length for seriation + raster (sec).
                             # Bounded by the model's temporal warmup (~32 bins),
                             # which leaves ~88 bins (~0.73s) of valid data max.
PANEL_B_SERIATION = "olo"    # "olo" (hierarchical clustering + optimal leaf
                             # ordering on correlation distance of smoothed
                             # rates) or "pc1" (sort by PC1 score).
PANEL_B_SMOOTH_SIGMA_S = 0.015  # Gaussian sigma (sec) for smoothing trial rates
                             # before computing distances. Only used for "olo".
# Panel B example neuron. Set both to None to auto-pick the highest-ccnorm
# reliable neuron across all sessions. Otherwise pin a specific example:
# PANEL_B_NEURON_ID is the original neuron index (matches sr["neuron_mask"][ni]).
#PANEL_B_SESSION = "Allen_2022-04-08"
#PANEL_B_NEURON_ID = 62
PANEL_B_SESSION = "Logan_2019-12-26"
PANEL_B_NEURON_ID = 20
SUBJECTS = ["Allen", "Logan"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}

# Model checkpoint
CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_SUBDIR = "2026-03-31_12-03-23_learned_resnet_concat_convgru_gaussian"
EXPERIMENT_SUBDIR = "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga1"
BEST_CKPT = "epoch=193-val_bps_overall=0.6000.ckpt"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_SUBDIR}/{EXPERIMENT_SUBDIR}/{BEST_CKPT}"

# Dataset config (same sessions used for training)
DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"

# Output directories
FIG_DIR = FIGURES_DIR / "fig3"
STAT_DIR = STATS_DIR / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Detect interactive IPython session
try:
    get_ipython()  # type: ignore[name-defined]
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False


def show_or_close(fig):
    """Show figure in interactive sessions, close otherwise."""
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


def subject_from_session(session_name):
    """Extract subject name from session string, e.g. 'Allen_2022-02-16' -> 'Allen'."""
    return session_name.split("_")[0]


# %% Load digital twin model
from DataYatesV1 import get_free_device

DEVICE = get_free_device()

if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from eval.eval_stack_multidataset import load_model

print(f"Loading model from: {CHECKPOINT_PATH}")
model, model_info = load_model(
    checkpoint_path=CHECKPOINT_PATH,
    device=str(DEVICE),
)
model.model.eval()
model.model.convnet.use_checkpointing = False
print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")
print(f"  {len(model.names)} datasets: {model.names}")


# %% Load 1-alpha from figure 2 cache
# The covariance decomposition (figure 2) computes alpha = Var(PSTH) / Var(rate)
# per neuron per session. We load these cached results rather than recomputing.
# Alpha represents the fraction of rate variance due to the stimulus (PSTH);
# 1-alpha is the fraction attributable to fixational eye movements (FEMs).

fig2_cache_path = CACHE_DIR / "fig2_decomposition.pkl"
if not fig2_cache_path.exists():
    raise FileNotFoundError(
        f"Figure 2 cache not found at {fig2_cache_path}. "
        "Run generate_figure2.py first to compute the covariance decomposition."
    )

print(f"Loading figure 2 cache from {fig2_cache_path}")
with open(fig2_cache_path, "rb") as f:
    fig2_session_results = dill.load(f)

# Build a lookup: session_name -> (alpha_array, neuron_mask)
# We use window index 0 (first counting window) to match figure 2's primary result.
fig2_alpha_by_session = {}
for sr in fig2_session_results:
    sess_name = sr["session"]
    subject = sr["subject"]
    if subject not in SUBJECTS:
        continue
    res_w0 = sr["results"][0]  # first window
    mats_w0 = sr["mats"][0]
    neuron_mask = sr["neuron_mask"]
    # alpha = diag(Cpsth) / diag(Crate), clipped to [0, 1]
    diag_psth = np.diag(mats_w0["PSTH"])
    diag_rate = np.diag(mats_w0["Intercept"])
    alpha = np.clip(diag_psth / diag_rate, 0, 1)
    fig2_alpha_by_session[sess_name] = {
        "alpha": alpha,
        "neuron_mask": neuron_mask,
        "subject": subject,
    }

print(f"  Loaded 1-alpha for {len(fig2_alpha_by_session)} sessions")


# %% Run model inference on fixRSVP data (or load from cache)
# For each session, we:
# 1. Load fixRSVP stimulus trials via the eval utilities
# 2. Run the digital twin forward on each trial to get predicted rates (rhat)
# 3. Trial-align robs and rhat into (n_trials, n_time, n_neurons) arrays
# 4. Affine-rescale rhat to match observed spike counts (gain + offset correction)
# 5. Compute ccnorm via split-half resampling (noise-ceiling normalization)
# 6. Compute single-trial variance explained for both model and leave-one-out PSTH

from eval.eval_stack_utils import (
    load_single_dataset,
    run_model,
    rescale_rhat,
    ccnorm_split_half_variable_trials,
)
from models.data import prepare_data

cache_path = CACHE_DIR / "fig3_digitaltwin.pkl"

if cache_path.exists() and not RECOMPUTE:
    print(f"Loading cached results from {cache_path}")
    with open(cache_path, "rb") as f:
        session_results = dill.load(f)
else:
    session_results = []

    for dataset_idx in range(len(model.names)):
        session_name = model.names[dataset_idx]
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            print(f"Skipping {session_name} (subject {subject} not in {SUBJECTS})")
            continue

        print(f"\n--- {session_name} ({subject}) [{dataset_idx+1}/{len(model.names)}] ---")

        # 1. Load fixRSVP data
        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Combine train + val fixRSVP indices (fixRSVP is never truly "held out")
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
        robs_flat = np.asarray(dset['robs'])        # (T_flat, NC)
        eyepos_flat = np.asarray(dset['eyepos'])    # (T_flat, 2)

        trials = np.unique(trial_inds)
        NT = len(trials)
        NC = robs_flat.shape[1]
        T = int(psth_inds_flat.max()) + 1

        # Fixation mask: eye within 1 degree of center
        fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < 1.0

        # 2. Run model trial-by-trial and trial-align
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

            # Build stimulus tensor with temporal lags
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            stim = dset['stim'][stim_lag_indices].permute(0, 2, 1, 3, 4)
            behavior = dset['behavior'][ix]

            # Forward pass
            out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)

            # Place into trial-aligned arrays
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            rhat[itrial, t_inds] = out['rhat'].detach().cpu().numpy()
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])
            eyepos[itrial, t_inds] = eyepos_flat[ix]

        # Filter trials by fixation duration
        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < 10:
            print(f"  Skipping: only {good_trials.sum()} good trials")
            continue

        robs = robs[good_trials]
        rhat = rhat[good_trials]
        dfs = dfs[good_trials]
        eyepos = eyepos[good_trials]

        # Truncate to valid time bins
        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[:, iix]
        rhat = rhat[:, iix]
        dfs = dfs[:, iix]

        # Neuron inclusion: minimum total spikes
        neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
        if len(neuron_mask) < 3:
            print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
            continue

        robs_used = robs[:, :, neuron_mask]
        rhat_used = rhat[:, :, neuron_mask]
        dfs_used = dfs[:, :, neuron_mask]

        n_trials, n_time, n_neurons = robs_used.shape
        print(f"  {n_trials} trials, {n_time} time bins, {n_neurons} neurons")

        # 3. Affine-rescale model predictions to match observed spike counts.
        # The model outputs are in Poisson rate space but may have a gain/offset
        # mismatch with the actual spike counts due to training on mixed conditions.
        # Rescaling corrects this so that r^2 and visualization are meaningful.
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

        # 4. Compute ccnorm via split-half resampling (Schoppe et al. 2016).
        # CCnorm = CCabs / CCmax, where:
        #   CCabs = correlation between model prediction and observed PSTH
        #   CCmax = estimated ceiling (max achievable correlation given neural noise),
        #           estimated by correlating split-half PSTHs and Spearman-Brown correcting.
        # This normalization accounts for the fact that noisy neurons have lower
        # achievable correlations, so a raw r=0.5 on a noisy neuron may actually
        # represent near-perfect prediction.
        # We run it twice and average to stabilize the stochastic estimate,
        # discarding neurons where the two estimates disagree substantially.
        ccnorm1, ccabs1, ccmax1, _, _ = ccnorm_split_half_variable_trials(
            robs_used, rhat_used, dfs_used,
            n_splits=CCNORM_N_SPLITS, return_components=True,
        )
        ccnorm2, ccabs2, ccmax2, _, _ = ccnorm_split_half_variable_trials(
            robs_used, rhat_used, dfs_used,
            n_splits=CCNORM_N_SPLITS, return_components=True,
        )
        # Flag neurons where the two ccnorm estimates diverge
        unstable = (ccnorm1 - ccnorm2) ** 2 > 0.01
        ccnorm = 0.5 * (ccnorm1 + ccnorm2)
        ccabs = 0.5 * (ccabs1 + ccabs2)
        ccmax = 0.5 * (ccmax1 + ccmax2)
        ccnorm[unstable] = np.nan

        # 5. Compute per-neuron correlation (rho) on trial-averaged traces
        # Mask invalid time bins before averaging
        rhat_masked = rhat_used.copy()
        robs_masked = robs_used.copy()
        rhat_masked[dfs_used == 0] = np.nan
        robs_masked[dfs_used == 0] = np.nan

        rhat_mean = np.nanmean(rhat_masked, axis=0)  # (n_time, n_neurons)
        robs_mean = np.nanmean(robs_masked, axis=0)
        n_valid = np.nansum(dfs_used, axis=0)  # (n_time, n_neurons)

        # Per-neuron Pearson r (only on time bins with sufficient trials)
        rhos = np.array([
            np.corrcoef(
                rhat_mean[n_valid[:, cc] > 10, cc],
                robs_mean[n_valid[:, cc] > 10, cc],
            )[0, 1]
            for cc in range(n_neurons)
        ])

        # 6. Single-trial variance explained
        # For the model: r^2 = 1 - Var(rhat - robs) / Var(robs)
        # For the PSTH baseline: we use leave-one-out PSTH to avoid overfitting.
        # The LOO-PSTH for trial i is the mean of all OTHER trials, so it's an
        # unbiased predictor for that trial's response.
        def var_explained(pred, true, axis=None):
            residuals = pred - true
            return 1 - np.nanvar(residuals, axis=axis) / np.nanvar(true, axis=axis)

        # Leave-one-out PSTH: for each trial, average all other trials
        rbar = np.zeros_like(robs_masked)
        for i in range(n_trials):
            other = np.setdiff1d(np.arange(n_trials), i)
            rbar[i] = np.nanmean(robs_masked[other], axis=0)

        # r^2 averaged over trials and time, per neuron
        ve_model = var_explained(rhat_masked, robs_masked, axis=(0, 1))
        ve_psth = var_explained(rbar, robs_masked, axis=(0, 1))

        # Look up 1-alpha from figure 2 cache
        alpha_vec = np.full(n_neurons, np.nan)
        if session_name in fig2_alpha_by_session:
            fig2_info = fig2_alpha_by_session[session_name]
            fig2_nmask = fig2_info["neuron_mask"]
            fig2_alpha = fig2_info["alpha"]
            # Align: our neuron_mask indexes into the full NC neurons,
            # and fig2's neuron_mask also indexes into the full NC neurons.
            # Find the intersection.
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
            # Traces (trial-averaged, for Panel B)
            "rhat_mean": rhat_mean,  # (n_time, n_neurons)
            "robs_mean": robs_mean,
            # Per-trial data (for Panel B raster/heatmaps)
            "robs_used": robs_used,   # (n_trials, n_time, n_neurons)
            "rhat_used": rhat_used,   # (n_trials, n_time, n_neurons)
            "dfs_used": dfs_used,     # (n_trials, n_time, n_neurons)
            # Performance metrics (per-neuron)
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

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {cache_path}")

# ---------------------------------------------------------------------------
# Flatten per-neuron arrays across sessions, tracking subject identity
# ---------------------------------------------------------------------------
all_rhos = []
all_ccnorm = []
all_ccmax = []
all_ve_model = []
all_ve_psth = []
all_alpha = []
all_subjects = []
all_session_idx = []

# Also keep traces for Panel B example selection
all_rhat_mean = []
all_robs_mean = []
all_trace_neuron_session = []  # (session_idx, local_neuron_idx) for each entry

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

# Filter to neurons with finite rho
valid = np.isfinite(rhos)
rhos = rhos[valid]
ccnorm = ccnorm[valid]
ccmax = ccmax[valid]
ve_model = ve_model[valid]
ve_psth = ve_psth[valid]
alpha = alpha[valid]
subjects = subjects[valid]
# Keep trace index mapping (filtered)
valid_indices = np.where(valid)[0]

print(f"\nTotal neurons: {len(rhos)} ({(subjects == 'Allen').sum()} Allen, "
      f"{(subjects == 'Logan').sum()} Logan)")

# "Good" neurons: reliable enough for quantitative analysis
good = ccmax > CCMAX_THRESHOLD
print(f"Good neurons (ccmax > {CCMAX_THRESHOLD}): {good.sum()}")


# %% Helper: filter long trials and seriate by observed trial pattern
# For the single-neuron raster plots, we restrict to trials with at least
# PANEL_B_MIN_DUR_S seconds of valid data, truncate to that window, and
# seriate the trials so neighbors in the display are maximally similar.
# Applying the same ordering to the model predictions lets the raster reveal
# whether the twin captures the same trial-to-trial structure.

PANEL_B_MIN_BINS = int(round(PANEL_B_MIN_DUR_S / DT))
tbins = np.arange(VALID_TIME_BINS) * DT  # time axis in seconds (pre-shift)


def order_single_neuron_by_seriation(robs_trials, rhat_trials, dfs_trials,
                                      method=PANEL_B_SERIATION,
                                      min_bins=PANEL_B_MIN_BINS,
                                      smooth_sigma_s=PANEL_B_SMOOTH_SIGMA_S):
    """Filter trials with ≥ min_bins valid bins (in the plotting window),
    truncate to the window, and seriate the trials so adjacent rows in the
    raster are as similar as possible.

    Background
    ----------
    "Seriation" is the problem of finding a 1-D ordering of n items that
    minimizes total dissimilarity between neighbors — the same objective as
    an open-path TSP on the pairwise distance matrix. It is NP-hard in
    general, so practical methods either solve a tractable relaxation or
    use a good heuristic. Two options are exposed here:

      - "pc1": sort trials by their score on the first principal component
        of the observed (trials × time) matrix. Equivalent to a 1-D PCA
        embedding. Captures the dominant global mode but tends to blur out
        finer structure that is orthogonal to PC1.

      - "olo": hierarchical clustering with optimal leaf ordering
        (Bar-Joseph et al. 2001) on correlation distance between
        Gaussian-smoothed trial rates. The dendrogram is built with average
        linkage, then subtrees are flipped to minimize the sum of distances
        between adjacent leaves. This keeps local continuity while also
        respecting cluster structure, and is the standard choice for
        ordering heatmap rows in the genomics/neuroscience literature.

    Leading bins that are NaN across all trials (model temporal-context
    warmup) are skipped first, so the returned window starts at the first
    bin with any valid data. Missing bins within the window (brief fixation
    dropouts) are per-timepoint-mean imputed for the seriation step only;
    display arrays keep NaN.

    Parameters
    ----------
    robs_trials, rhat_trials, dfs_trials : (n_trials, n_time) arrays
    method : {"olo", "pc1"}
    min_bins : int, required valid bins per trial in the plotting window
    smooth_sigma_s : float, Gaussian smoothing sigma (sec) applied to rates
        before computing pairwise correlation distance. Only used for "olo".

    Returns
    -------
    robs_sorted, rhat_sorted : (n_kept, min_bins) with NaN at invalid bins
    order : indices into filtered set giving the seriation
    first_bin : int, leading bin of the returned window in the full trial axis
    """
    # Drop leading bins with no valid data in any trial (model warmup).
    any_valid = (dfs_trials > 0).any(axis=0)
    if not any_valid.any():
        empty = np.empty((0, min_bins))
        return empty, empty, np.arange(0), 0
    first_bin = int(np.argmax(any_valid))
    end_bin = first_bin + min_bins
    if end_bin > dfs_trials.shape[1]:
        empty = np.empty((0, min_bins))
        return empty, empty, np.arange(0), first_bin

    # Keep trials whose valid-bin count in the plotting window meets the minimum.
    window_valid = dfs_trials[:, first_bin:end_bin] > 0
    trial_valid_count = window_valid.sum(axis=1)
    keep = trial_valid_count >= min_bins

    robs_k = robs_trials[keep, first_bin:end_bin].astype(float).copy()
    rhat_k = rhat_trials[keep, first_bin:end_bin].astype(float).copy()
    dfs_k = dfs_trials[keep, first_bin:end_bin]
    valid_k = dfs_k > 0

    if robs_k.shape[0] < 2:
        robs_k[~valid_k] = np.nan
        rhat_k[~valid_k] = np.nan
        return robs_k, rhat_k, np.arange(robs_k.shape[0]), first_bin

    # Impute missing bins with per-timepoint mean across retained trials
    # so the seriation isn't distorted by zeros / NaNs at dropped fixation bins.
    obs_masked = np.where(valid_k, robs_k, np.nan)
    col_mean = np.nanmean(obs_masked, axis=0, keepdims=True)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    obs_filled = np.where(valid_k, robs_k, np.broadcast_to(col_mean, robs_k.shape))

    if method == "pc1":
        obs_centered = obs_filled - obs_filled.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(obs_centered, full_matrices=False)
        pc1_scores = U[:, 0] * S[0]
        order = np.argsort(pc1_scores)
    elif method == "olo":
        # Gaussian-smooth each trial's rate in time, then use correlation
        # distance (1 - Pearson r) between trials. Correlation distance
        # emphasizes pattern similarity over overall rate.
        sigma_bins = max(smooth_sigma_s / DT, 1e-6)
        obs_smooth = gaussian_filter1d(obs_filled, sigma=sigma_bins, axis=1,
                                       mode="nearest")
        # Guard against zero-variance rows (e.g., flat silent trials), which
        # would produce NaN correlations. Add a tiny jitter to any such row.
        row_std = obs_smooth.std(axis=1)
        flat = row_std < 1e-12
        if flat.any():
            rng = np.random.default_rng(0)
            obs_smooth = obs_smooth.copy()
            obs_smooth[flat] += rng.normal(scale=1e-9, size=(flat.sum(),
                                                              obs_smooth.shape[1]))
        dists = pdist(obs_smooth, metric="correlation")
        Z = linkage(dists, method="average")
        Z = optimal_leaf_ordering(Z, dists)
        order = np.asarray(leaves_list(Z), dtype=int)
    else:
        raise ValueError(f"Unknown seriation method: {method!r} "
                         f"(expected 'olo' or 'pc1')")

    robs_k[~valid_k] = np.nan
    rhat_k[~valid_k] = np.nan
    return robs_k[order], rhat_k[order], order, first_bin


# Hard-coded display window for Panel B (and candidates). PSTH and raster
# halves are both this wide so everything lines up.
PANEL_B_WINDOW_S = 0.5
n_bins_b = int(round(PANEL_B_WINDOW_S / DT))


def _draw_raster_pair(ax, robs_rate, rhat_rate, *, window_s, vmin, vmax,
                      scale_len_s=0.1, n_trials_scale=10,
                      label_fontsize=9, scale_fontsize=8):
    """Concatenated observed|twin raster on one axes with a vertical divider,
    top 'Observed'/'Twin' labels, no tick marks/spines, and time + trial
    scale bars. Returns the AxesImage for colorbar construction.

    Both halves are rendered in a single imshow so they have identical aspect
    ratios and pixel sizes — subplot-spacing jitter can't desync them.
    """
    combined = np.concatenate([robs_rate, rhat_rate], axis=1)
    n_trials_local = combined.shape[0]
    im = ax.imshow(
        combined, aspect="auto", origin="upper",
        extent=[0, 2 * window_s, n_trials_local, 0],
        vmin=vmin, vmax=vmax, cmap="binary", interpolation="none",
    )
    ax.axvline(window_s, color="k", linewidth=0.8)
    ax.text(0.25, 1.02, "Observed", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=label_fontsize)
    ax.text(0.75, 1.02, "Twin", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=label_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # Time scale bar below the image: x in data, y in axes fraction.
    trans_x = ax.get_xaxis_transform()
    ax.plot([0.0, scale_len_s], [-0.06, -0.06], "k-", linewidth=2,
            transform=trans_x, clip_on=False)
    ax.text(scale_len_s / 2, -0.11, f"{int(round(scale_len_s * 1000))} ms",
            transform=trans_x, ha="center", va="top",
            fontsize=scale_fontsize, clip_on=False)
    # Trials scale bar to the left: x in axes fraction, y in data.
    # Image is origin='upper' with ylim=[n_trials, 0], so bottom of image
    # is y_data=n_trials; 10 trials up from the bottom = n_trials - 10.
    trans_y = ax.get_yaxis_transform()
    n_scale = min(n_trials_scale, n_trials_local)
    y0, y1 = n_trials_local, n_trials_local - n_scale
    ax.plot([-0.02, -0.02], [y0, y1], "k-", linewidth=2,
            transform=trans_y, clip_on=False)
    ax.text(-0.04, (y0 + y1) / 2, f"{n_scale} trials",
            transform=trans_y, ha="right", va="center", rotation=90,
            fontsize=scale_fontsize, clip_on=False)
    return im


# Panel B candidates: top 30 neurons by ccnorm
# Quick visual survey to pick the best-looking example for Panel B.
# Each row is one neuron: PSTH overlay (left) and concatenated observed|twin
# raster (right). Trials are filtered to ≥ PANEL_B_WINDOW_S and sorted by PC1
# of the observed trial-by-time matrix; the same sort is applied to the twin.

mask_all_cand = good & np.isfinite(ccnorm)
candidates_ranked = np.where(mask_all_cand)[0]
candidates_ranked = candidates_ranked[np.argsort(ccnorm[candidates_ranked])[::-1]]
n_show = min(50, len(candidates_ranked))

for rank in range(n_show):
    idx_local = candidates_ranked[rank]
    idx_global = valid_indices[idx_local]
    si_c, ni_c = all_trace_neuron_session[idx_global]
    sr_c = session_results[si_c]

    # Per-trial data for this neuron
    robs_t_full = sr_c["robs_used"][:, :, ni_c]
    rhat_t_full = sr_c["rhat_used"][:, :, ni_c]
    dfs_t_full = sr_c["dfs_used"][:, :, ni_c]

    robs_sorted, rhat_sorted, _, first_bin_c = order_single_neuron_by_seriation(
        robs_t_full, rhat_t_full, dfs_t_full
    )
    if robs_sorted.shape[0] < 2:
        continue
    robs_rate = (robs_sorted / DT)[:, :n_bins_b]
    rhat_rate = (rhat_sorted / DT)[:, :n_bins_b]

    # Trial-averaged traces shifted so first_bin_c → t=0, then clipped to window.
    robs_tr = all_robs_mean[idx_global] / DT
    rhat_tr = all_rhat_mean[idx_global] / DT
    tt = (np.arange(len(robs_tr)) - first_bin_c) * DT
    window_c = (np.isfinite(robs_tr) & np.isfinite(rhat_tr)
                & (tt >= 0) & (tt <= PANEL_B_WINDOW_S))

    vm = 0
    vx = np.nanpercentile(
        np.concatenate([robs_rate.ravel(), rhat_rate.ravel()]), 97
    )

    fig_cand = plt.figure(figsize=(8, 3))
    gs_cand = fig_cand.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35)
    ax_psth_c = fig_cand.add_subplot(gs_cand[0, 0])
    ax_rast_c = fig_cand.add_subplot(gs_cand[0, 1])

    fig_cand.suptitle(
        f"Rank {rank+1}: {sr_c['session']} neuron {sr_c['neuron_mask'][ni_c]} "
        f"({sr_c['subject']}) — ccnorm={ccnorm[idx_local]:.3f}, "
        f"N_trials={robs_sorted.shape[0]}",
        fontsize=9,
    )

    ax_psth_c.plot(tt[window_c], robs_tr[window_c], 'k', linewidth=1,
                   label="Observed")
    ax_psth_c.plot(tt[window_c], rhat_tr[window_c], 'tab:red', linewidth=1,
                   label="Twin")
    ax_psth_c.set_xlim(0, PANEL_B_WINDOW_S)
    ax_psth_c.set_xlabel("Time (s)")
    ax_psth_c.set_ylabel("Rate (sp/s)")
    ax_psth_c.legend(frameon=False, fontsize=7)
    ax_psth_c.spines["top"].set_visible(False)
    ax_psth_c.spines["right"].set_visible(False)

    im_cand = _draw_raster_pair(
        ax_rast_c, robs_rate, rhat_rate,
        window_s=PANEL_B_WINDOW_S, vmin=vm, vmax=vx,
    )
    fig_cand.colorbar(im_cand, ax=ax_rast_c, shrink=0.8, pad=0.02, label="sp/s")
    plt.show()
    plt.close(fig_cand)


# %% Panels B and C: Example neuron — PSTH overlay (B) + single-trial rasters (C)
# Select the neuron with the highest ccnorm among reliable neurons across all
# animals (or honor the PANEL_B_SESSION / PANEL_B_NEURON_ID override). Panel B
# is the trial-averaged observed (black) and twin (red) rates; Panel C is the
# concatenated observed|twin single-trial raster. Same 0.5 s window in both.

# Select best neuron across all subjects, unless a specific example is pinned.
if PANEL_B_SESSION is not None and PANEL_B_NEURON_ID is not None:
    si = next(
        (i for i, sr in enumerate(session_results)
         if sr["session"] == PANEL_B_SESSION),
        None,
    )
    if si is None:
        raise ValueError(
            f"PANEL_B_SESSION={PANEL_B_SESSION!r} not found in session_results"
        )
    nmask = session_results[si]["neuron_mask"]
    matches = np.where(np.asarray(nmask) == PANEL_B_NEURON_ID)[0]
    if len(matches) == 0:
        raise ValueError(
            f"PANEL_B_NEURON_ID={PANEL_B_NEURON_ID} not in session "
            f"{PANEL_B_SESSION} (neurons passing spike threshold: "
            f"{sorted(int(x) for x in nmask)})"
        )
    ni = int(matches[0])
    best_global = all_trace_neuron_session.index((si, ni))
    loc = np.where(valid_indices == best_global)[0]
    if len(loc) == 0:
        raise ValueError(
            f"Pinned neuron (session={PANEL_B_SESSION}, id={PANEL_B_NEURON_ID}) "
            "was filtered out (non-finite rho)"
        )
    best_local = int(loc[0])
else:
    mask_all = good & np.isfinite(ccnorm)
    candidates_all = np.where(mask_all)[0]
    best_local = candidates_all[np.nanargmax(ccnorm[candidates_all])]
    best_global = valid_indices[best_local]
    si, ni = all_trace_neuron_session[best_global]

best_sr = session_results[si]
best_subj = best_sr["subject"]

print(f"Panel B — example neuron: {best_sr['session']}, "
      f"neuron {best_sr['neuron_mask'][ni]}, "
      f"ccnorm={ccnorm[best_local]:.2f}")

# Extract per-trial data for this neuron and seriate by observed pattern
robs_trials = best_sr["robs_used"][:, :, ni]   # (n_trials, n_time)
rhat_trials = best_sr["rhat_used"][:, :, ni]
dfs_trials = best_sr["dfs_used"][:, :, ni]

robs_sorted_b, rhat_sorted_b, _, first_bin_b = order_single_neuron_by_seriation(
    robs_trials, rhat_trials, dfs_trials
)
# Truncate to the hard-coded display window so PSTH and rasters match length.
robs_trials_rate = (robs_sorted_b / DT)[:, :n_bins_b]
rhat_trials_rate = (rhat_sorted_b / DT)[:, :n_bins_b]

print(f"  Panel B raster: {robs_sorted_b.shape[0]} trials "
      f"(≥ {PANEL_B_WINDOW_S:.1f}s) ordered by '{PANEL_B_SERIATION}' seriation "
      f"of observed, starting at bin {first_bin_b} "
      f"({tbins[first_bin_b]*1000:.0f} ms → t=0)")

# Trial-averaged traces (full window, shifted so first_bin_b → t=0)
robs_trace = all_robs_mean[best_global] / DT
rhat_trace = all_rhat_mean[best_global] / DT
t_valid = np.isfinite(robs_trace) & np.isfinite(rhat_trace)
t = (np.arange(len(robs_trace)) - first_bin_b) * DT
psth_window = t_valid & (t >= 0) & (t <= PANEL_B_WINDOW_S)

# Shared color limits for raster plots
vmin = 0
vmax = np.nanpercentile(
    np.concatenate([robs_trials_rate.ravel(), rhat_trials_rate.ravel()]), 97
)


# --- Panel B: PSTH overlay (its own figure) ---
fig_b, ax_psth_b = plt.subplots(figsize=(3, 2.5))
ax_psth_b.plot(t[psth_window], robs_trace[psth_window], 'k',
               linewidth=1, label="Observed")
ax_psth_b.plot(t[psth_window], rhat_trace[psth_window], color='tab:red',
               linewidth=1, label="Twin")
ax_psth_b.set_xlim(0, PANEL_B_WINDOW_S)
ax_psth_b.set_xlabel("Time (s)")
ax_psth_b.set_ylabel("Rate (sp/s)")
ax_psth_b.set_title(f"ccnorm = {ccnorm[best_local]:.2f}")
ax_psth_b.legend(frameon=False, fontsize=8)
ax_psth_b.spines["top"].set_visible(False)
ax_psth_b.spines["right"].set_visible(False)
fig_b.tight_layout()
fig_b.savefig(FIG_DIR / "panel_b_psth.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_b)

# --- Panel C: concatenated observed|twin single-trial raster ---
fig_c_rast, ax_rast_c_fig = plt.subplots(figsize=(4.5, 2.5))
im = _draw_raster_pair(
    ax_rast_c_fig, robs_trials_rate, rhat_trials_rate,
    window_s=PANEL_B_WINDOW_S, vmin=vmin, vmax=vmax,
)
fig_c_rast.colorbar(im, ax=ax_rast_c_fig, shrink=0.8, pad=0.02, label="sp/s")
fig_c_rast.savefig(FIG_DIR / "panel_c_rasters.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_c_rast)


# %% Panel D: Histogram of normalized correlation (ccnorm)
# ccnorm = CCabs / CCmax normalizes each neuron's prediction accuracy by its
# noise ceiling. A value of 1.0 means the model explains all explainable variance.
# We report per-animal distributions with median and IQR.

fig_d, ax_d_hist = plt.subplots(figsize=(3.5, 2.5))

# Shared bins across subjects
valid_ccnorm = ccnorm[good & np.isfinite(ccnorm)]
bins = np.linspace(0, 1, 21)

for subj in SUBJECTS:
    mask = (subjects == subj) & good & np.isfinite(ccnorm)
    if not mask.any():
        continue
    vals = ccnorm[mask]
    color = SUBJECT_COLORS[subj]
    med = np.nanmedian(vals)
    q25, q75 = np.nanpercentile(vals, [25, 75])

    ax_d_hist.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.5)
    ax_d_hist.axvline(med, color=color, linewidth=2, ls=(0, (1, 1)),
                      label=f"{subj}: {med:.2f} [{q25:.2f}, {q75:.2f}]")

    print(f"Panel D — {subj} (N={mask.sum()}): "
          f"median ccnorm={med:.2f}, IQR=[{q25:.2f}, {q75:.2f}]")

ax_d_hist.set_xlabel("Normalized correlation (ccnorm)")
ax_d_hist.set_ylabel("Count")
ax_d_hist.legend(frameon=False, fontsize=8)
ax_d_hist.spines["top"].set_visible(False)
ax_d_hist.spines["right"].set_visible(False)

fig_d.tight_layout()
fig_d.savefig(FIG_DIR / "panel_d_ccnorm_hist.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_d)


# %% Panel E: Single-trial r^2 — model vs PSTH
# Each point is one neuron. X-axis = r^2 using leave-one-out PSTH as predictor.
# Y-axis = r^2 using the digital twin as predictor. Points above the unity line
# indicate the model captures more trial-by-trial variance than the PSTH alone.

fig_e, ax_e_scat = plt.subplots(figsize=(3, 2.5))

for subj in SUBJECTS:
    mask = (subjects == subj) & good
    if not mask.any():
        continue
    ax_e_scat.scatter(
        ve_psth[mask], ve_model[mask],
        s=5, alpha=0.5, color=SUBJECT_COLORS[subj], label=subj,
    )

lims = [0, max(0.4, np.nanmax(ve_model[good]) * 1.1)]
ax_e_scat.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)
ax_e_scat.set_xlim(lims)
ax_e_scat.set_ylim(lims)
ax_e_scat.set_xlabel("Single-trial $r^2$ (PSTH)")
ax_e_scat.set_ylabel("Single-trial $r^2$ (Model)")
ax_e_scat.legend(frameon=False, fontsize=8)
ax_e_scat.spines["top"].set_visible(False)
ax_e_scat.spines["right"].set_visible(False)

fig_e.tight_layout()
fig_e.savefig(FIG_DIR / "panel_e_r2_scatter.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_e)

# Stats: Wilcoxon signed-rank (model r^2 > PSTH r^2)
from scipy.stats import wilcoxon

for subj in SUBJECTS + ["All"]:
    mask = good.copy()
    if subj != "All":
        mask = mask & (subjects == subj)
    x = ve_model[mask]
    y = ve_psth[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = x - y
    stat, p = wilcoxon(d, alternative='greater')
    print(f"Panel E — {subj} (N={len(d)}): "
          f"median model r^2={np.median(x):.3f}, PSTH r^2={np.median(y):.3f}, "
          f"Wilcoxon stat={stat:.1f}, p={p:.3g}")


# %% Panel F: Improvement over PSTH vs FEM modulation (1-alpha)
# The digital twin captures eye-movement-driven variability that the PSTH misses.
# We expect the largest improvement for neurons most modulated by FEMs (high 1-alpha).
# Improvement ratio = r^2(model) / r^2(PSTH). Values > 1 mean the model is better.

fig_f, ax_f = plt.subplots(figsize=(3, 2.5))

# Only neurons with valid alpha and positive PSTH r^2
has_alpha = good & np.isfinite(alpha) & (ve_psth > 0)

for subj in SUBJECTS:
    mask = has_alpha & (subjects == subj)
    if not mask.any():
        continue
    fem_mod = 1 - alpha[mask]
    improvement = ve_model[mask] / ve_psth[mask]
    ax_f.scatter(fem_mod, improvement, s=5, alpha=0.5,
                 color=SUBJECT_COLORS[subj], label=subj)

ax_f.axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax_f.set_xlabel("FEM modulation (1 - α)")
ax_f.set_ylabel("$r^2$ improvement (Model / PSTH)")
ax_f.set_ylim(0, 5)
ax_f.legend(frameon=False, fontsize=8)
ax_f.spines["top"].set_visible(False)
ax_f.spines["right"].set_visible(False)

fig_f.tight_layout()
fig_f.savefig(FIG_DIR / "panel_f_improvement_vs_fem.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_f)

# Stats: Spearman correlation between 1-alpha and improvement ratio
for subj in SUBJECTS + ["All"]:
    mask = has_alpha.copy()
    if subj != "All":
        mask = mask & (subjects == subj)
    fem_mod = 1 - alpha[mask]
    improvement = ve_model[mask] / ve_psth[mask]
    ok = np.isfinite(fem_mod) & np.isfinite(improvement)
    r_s, p_s = sp_stats.spearmanr(fem_mod[ok], improvement[ok])
    print(f"Panel F — {subj} (N={ok.sum()}): "
          f"Spearman r={r_s:.3f}, p={p_s:.3g}")

# %% Composite figure
# Layout (2 rows × 3 cols; PSTH and rasters get their own cells):
#   Row 1: [A: schematic] [B: PSTH] [C: rasters]   (width_ratios 1, 1, 2)
#   Row 2: [D: ccnorm hist] [E: r^2 scatter] [F: improvement vs FEM]

import cairosvg
from PIL import Image
import io

fig_comp = plt.figure(figsize=(10, 5), constrained_layout=True)
gs_outer = fig_comp.add_gridspec(2, 1)
gs_top = gs_outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.1)
gs_bot = gs_outer[1].subgridspec(1, 3, wspace=0.15)

# --- Panel A: schematic from SVG ---
ax_a = fig_comp.add_subplot(gs_top[0, 0])
svg_path = str(VISIONCORE_ROOT / "ryan" / "fig3" / "fig3-schematic.svg")
png_data = cairosvg.svg2png(url=svg_path, output_width=800)
img = Image.open(io.BytesIO(png_data))
ax_a.imshow(img)
ax_a.set_title("A", fontweight="bold", loc="left")
ax_a.axis("off")

# --- Panel B: example neuron PSTH overlay ---
ax_b_comp = fig_comp.add_subplot(gs_top[0, 1])
ax_b_comp.plot(t[psth_window], robs_trace[psth_window],
               'k', linewidth=1, label="Observed")
ax_b_comp.plot(t[psth_window], rhat_trace[psth_window],
               color='tab:red', linewidth=1, label="Twin")
ax_b_comp.set_xlim(0, PANEL_B_WINDOW_S)
ax_b_comp.set_xlabel("Time (s)")
ax_b_comp.set_ylabel("Rate (sp/s)")
ax_b_comp.set_title("B", fontweight="bold", loc="left")
ax_b_comp.legend(frameon=False, fontsize=7)
ax_b_comp.spines["top"].set_visible(False)
ax_b_comp.spines["right"].set_visible(False)

# --- Panel C: concatenated observed|twin raster ---
ax_c_comp = fig_comp.add_subplot(gs_top[0, 2])
im_comp = _draw_raster_pair(
    ax_c_comp, robs_trials_rate, rhat_trials_rate,
    window_s=PANEL_B_WINDOW_S, vmin=vmin, vmax=vmax,
    label_fontsize=8, scale_fontsize=7,
)
ax_c_comp.set_title("C", fontweight="bold", loc="left")
fig_comp.colorbar(im_comp, ax=ax_c_comp, shrink=0.8, pad=0.02, label="sp/s")

# --- Panel D: ccnorm histogram ---
ax = fig_comp.add_subplot(gs_bot[0, 0])
bins_comp = np.linspace(0, 1, 21)
for subj in SUBJECTS:
    mask = (subjects == subj) & good & np.isfinite(ccnorm)
    if not mask.any():
        continue
    vals = ccnorm[mask]
    color = SUBJECT_COLORS[subj]
    med = np.nanmedian(vals)
    q25, q75 = np.nanpercentile(vals, [25, 75])
    ax.hist(vals, bins=bins_comp, color=color, edgecolor="white", alpha=0.5)
    ax.axvline(med, color=color, linewidth=2, ls=(0, (1, 1)),
               label=f"{subj}: {med:.2f} [{q25:.2f}, {q75:.2f}]")
ax.set_xlabel("Normalized correlation (ccnorm)")
ax.set_ylabel("Count")
ax.set_title("D", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel E: single-trial r^2 scatter ---
ax = fig_comp.add_subplot(gs_bot[0, 1])
for subj in SUBJECTS:
    mask = (subjects == subj) & good
    if not mask.any():
        continue
    ax.scatter(ve_psth[mask], ve_model[mask], s=5, alpha=0.5,
               color=SUBJECT_COLORS[subj], label=subj)
lims_e = [0, max(0.4, np.nanmax(ve_model[good]) * 1.1)]
ax.plot(lims_e, lims_e, 'k--', linewidth=0.5, alpha=0.5)
ax.set_xlim(lims_e)
ax.set_ylim(lims_e)
ax.set_xlabel("Single-trial $r^2$ (PSTH)")
ax.set_ylabel("Single-trial $r^2$ (Model)")
ax.set_title("E", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel F: improvement vs FEM modulation ---
ax = fig_comp.add_subplot(gs_bot[0, 2])
for subj in SUBJECTS:
    mask = has_alpha & (subjects == subj)
    if not mask.any():
        continue
    fem_mod = 1 - alpha[mask]
    improvement = ve_model[mask] / ve_psth[mask]
    ax.scatter(fem_mod, improvement, s=5, alpha=0.5,
               color=SUBJECT_COLORS[subj], label=subj)
ax.axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel("FEM modulation (1 - α)")
ax.set_ylabel("$r^2$ improvement (Model / PSTH)")
ax.set_ylim(0, 5)
ax.set_title("F", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig_comp.savefig(FIG_DIR / "fig3_composite.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_comp)

print(f"\nAll panel figures saved to: {FIG_DIR}")
print("Done.")
