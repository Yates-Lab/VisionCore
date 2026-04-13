# %% Imports and configuration
"""
Figure 3: Digital Twin Performance

Flat, cell-based script. Each cell computes stats and plots its panel(s).
Run interactively with IPython (#%% cells) or as a script with uv run.

Panels:
  A  Architecture schematic (fig3-schematic.svg, placed manually)
  B  Example neuron PSTHs with twin predictions (1 per animal)
  C  Histogram of normalized correlation (ccnorm)
  D  Single-trial r^2 scatter: model vs PSTH
  E  Improvement over PSTH vs FEM modulation (1-alpha)
"""
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
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


# %% Panel B: Example neuron PSTHs with twin predictions
# For each animal, select the neuron with the highest ccnorm among reliable
# neurons (ccmax > threshold). Plot the trial-averaged observed firing rate
# (black) and model prediction (colored) over time.

tbins = np.arange(VALID_TIME_BINS) * DT  # time axis in seconds

fig_b, axs_b = plt.subplots(1, 2, figsize=(6, 2.5), sharey=False)

for ax, subj in zip(axs_b, SUBJECTS):
    mask = (subjects == subj) & good & np.isfinite(ccnorm)
    if not mask.any():
        ax.set_title(f"{subj}: no good neurons")
        continue

    # Best neuron by ccnorm
    candidates = np.where(mask)[0]
    best_local = candidates[np.nanargmax(ccnorm[candidates])]
    best_global = valid_indices[best_local]
    si, ni = all_trace_neuron_session[best_global]

    robs_trace = all_robs_mean[best_global] / DT  # convert to spikes/sec
    rhat_trace = all_rhat_mean[best_global] / DT

    # Only plot time bins with valid data
    t_valid = np.isfinite(robs_trace) & np.isfinite(rhat_trace)
    t = tbins[:len(robs_trace)]

    ax.plot(t[t_valid], robs_trace[t_valid], 'k', linewidth=1, label="Observed")
    ax.plot(t[t_valid], rhat_trace[t_valid], color=SUBJECT_COLORS[subj],
            linewidth=1, label="Twin")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (sp/s)")
    ax.set_title(f"{subj} (ccnorm={ccnorm[best_local]:.2f})")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig_b.tight_layout()
fig_b.savefig(FIG_DIR / "panel_b_example_traces.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_b)


# %% Panel C: Histogram of normalized correlation (ccnorm)
# ccnorm = CCabs / CCmax normalizes each neuron's prediction accuracy by its
# noise ceiling. A value of 1.0 means the model explains all explainable variance.
# We report per-animal distributions with median and IQR.

fig_c, ax_c = plt.subplots(figsize=(3.5, 2.5))

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

    ax_c.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.5)
    ax_c.axvline(med, color=color, linewidth=2, ls=(0, (1, 1)),
                 label=f"{subj}: {med:.2f} [{q25:.2f}, {q75:.2f}]")

    print(f"Panel C — {subj} (N={mask.sum()}): "
          f"median ccnorm={med:.2f}, IQR=[{q25:.2f}, {q75:.2f}]")

ax_c.set_xlabel("Normalized correlation (ccnorm)")
ax_c.set_ylabel("Count")
ax_c.legend(frameon=False, fontsize=8)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

fig_c.tight_layout()
fig_c.savefig(FIG_DIR / "panel_c_ccnorm_hist.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_c)


# %% Panel D: Single-trial r^2 — model vs PSTH
# Each point is one neuron. X-axis = r^2 using leave-one-out PSTH as predictor.
# Y-axis = r^2 using the digital twin as predictor. Points above the unity line
# indicate the model captures more trial-by-trial variance than the PSTH alone.

fig_d, ax_d = plt.subplots(figsize=(3, 2.5))

for subj in SUBJECTS:
    mask = (subjects == subj) & good
    if not mask.any():
        continue
    ax_d.scatter(
        ve_psth[mask], ve_model[mask],
        s=5, alpha=0.5, color=SUBJECT_COLORS[subj], label=subj,
    )

lims = [0, max(0.4, np.nanmax(ve_model[good]) * 1.1)]
ax_d.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)
ax_d.set_xlim(lims)
ax_d.set_ylim(lims)
ax_d.set_xlabel("Single-trial $r^2$ (PSTH)")
ax_d.set_ylabel("Single-trial $r^2$ (Model)")
ax_d.legend(frameon=False, fontsize=8)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)

fig_d.tight_layout()
fig_d.savefig(FIG_DIR / "panel_d_r2_scatter.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_d)

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
    print(f"Panel D — {subj} (N={len(d)}): "
          f"median model r^2={np.median(x):.3f}, PSTH r^2={np.median(y):.3f}, "
          f"Wilcoxon stat={stat:.1f}, p={p:.3g}")


# %% Panel E: Improvement over PSTH vs FEM modulation (1-alpha)
# The digital twin captures eye-movement-driven variability that the PSTH misses.
# We expect the largest improvement for neurons most modulated by FEMs (high 1-alpha).
# Improvement ratio = r^2(model) / r^2(PSTH). Values > 1 mean the model is better.

fig_e, ax_e = plt.subplots(figsize=(3, 2.5))

# Only neurons with valid alpha and positive PSTH r^2
has_alpha = good & np.isfinite(alpha) & (ve_psth > 0)

for subj in SUBJECTS:
    mask = has_alpha & (subjects == subj)
    if not mask.any():
        continue
    fem_mod = 1 - alpha[mask]
    improvement = ve_model[mask] / ve_psth[mask]
    ax_e.scatter(fem_mod, improvement, s=5, alpha=0.5,
                 color=SUBJECT_COLORS[subj], label=subj)

ax_e.axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
ax_e.set_xlabel("FEM modulation (1 - α)")
ax_e.set_ylabel("$r^2$ improvement (Model / PSTH)")
ax_e.set_ylim(0, 5)
ax_e.legend(frameon=False, fontsize=8)
ax_e.spines["top"].set_visible(False)
ax_e.spines["right"].set_visible(False)

fig_e.tight_layout()
fig_e.savefig(FIG_DIR / "panel_e_improvement_vs_fem.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_e)

# Stats: Spearman correlation between 1-alpha and improvement ratio
for subj in SUBJECTS + ["All"]:
    mask = has_alpha.copy()
    if subj != "All":
        mask = mask & (subjects == subj)
    fem_mod = 1 - alpha[mask]
    improvement = ve_model[mask] / ve_psth[mask]
    ok = np.isfinite(fem_mod) & np.isfinite(improvement)
    r_s, p_s = sp_stats.spearmanr(fem_mod[ok], improvement[ok])
    print(f"Panel E — {subj} (N={ok.sum()}): "
          f"Spearman r={r_s:.3f}, p={p_s:.3g}")

# %% Composite figure
# Layout:
#   Row 1: [A: schematic]  [B: example traces (2 subplots)]
#   Row 2: [C: ccnorm hist] [D: r^2 scatter] [E: improvement vs FEM]

import cairosvg
from PIL import Image
import io

fig_comp = plt.figure(figsize=(10, 7))
gs = fig_comp.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.35)

# --- Panel A: schematic from SVG ---
ax_a = fig_comp.add_subplot(gs[0, 0])
svg_path = str(VISIONCORE_ROOT / "ryan" / "fig3" / "fig3-schematic.svg")
png_data = cairosvg.svg2png(url=svg_path, output_width=800)
img = Image.open(io.BytesIO(png_data))
ax_a.imshow(img)
ax_a.set_title("A", fontweight="bold", loc="left")
ax_a.axis("off")

# --- Panel B: example neuron PSTHs (2 subplots in columns 1-2 of row 0) ---
gs_b = gs[0, 1:].subgridspec(1, 2, wspace=0.3)
for idx, subj in enumerate(SUBJECTS):
    ax = fig_comp.add_subplot(gs_b[0, idx])
    mask = (subjects == subj) & good & np.isfinite(ccnorm)
    if not mask.any():
        ax.set_title(f"{subj}: no good neurons")
        continue
    candidates = np.where(mask)[0]
    best_local = candidates[np.nanargmax(ccnorm[candidates])]
    best_global = valid_indices[best_local]
    si, ni = all_trace_neuron_session[best_global]
    robs_trace = all_robs_mean[best_global] / DT
    rhat_trace = all_rhat_mean[best_global] / DT
    t_valid = np.isfinite(robs_trace) & np.isfinite(rhat_trace)
    t = tbins[:len(robs_trace)]
    ax.plot(t[t_valid], robs_trace[t_valid], 'k', linewidth=1, label="Observed")
    ax.plot(t[t_valid], rhat_trace[t_valid], color=SUBJECT_COLORS[subj],
            linewidth=1, label="Twin")
    ax.set_xlabel("Time (s)")
    if idx == 0:
        ax.set_ylabel("Rate (sp/s)")
    ax.set_title(f"{'B' if idx == 0 else ''}", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- Panel C: ccnorm histogram ---
ax = fig_comp.add_subplot(gs[1, 0])
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
ax.set_title("C", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel D: single-trial r^2 scatter ---
ax = fig_comp.add_subplot(gs[1, 1])
for subj in SUBJECTS:
    mask = (subjects == subj) & good
    if not mask.any():
        continue
    ax.scatter(ve_psth[mask], ve_model[mask], s=5, alpha=0.5,
               color=SUBJECT_COLORS[subj], label=subj)
lims_d = [0, max(0.4, np.nanmax(ve_model[good]) * 1.1)]
ax.plot(lims_d, lims_d, 'k--', linewidth=0.5, alpha=0.5)
ax.set_xlim(lims_d)
ax.set_ylim(lims_d)
ax.set_xlabel("Single-trial $r^2$ (PSTH)")
ax.set_ylabel("Single-trial $r^2$ (Model)")
ax.set_title("D", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel E: improvement vs FEM modulation ---
ax = fig_comp.add_subplot(gs[1, 2])
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
ax.set_title("E", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig_comp.savefig(FIG_DIR / "fig3_composite.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_comp)

print(f"\nAll panel figures saved to: {FIG_DIR}")
print("Done.")
