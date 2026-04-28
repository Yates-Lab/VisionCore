# %% Imports and configuration
"""
Compare behavior vs vision-only digital twin models on the fixRSVP condition.

Follows the fig3 methodology: for each session, run both models on trial-aligned
fixRSVP trials, affine-rescale each model's predicted rates to match observed
spike counts (per-neuron gain+offset, fit by Poisson MLE), then compute matched
performance metrics for a fair within-condition comparison.

Metrics (per neuron):
  - BPS: bits-per-spike vs the mean-rate null, computed on rescaled rates
  - ccnorm: split-half noise-corrected correlation (Schoppe et al. 2016)
  - single-trial r^2: 1 - Var(pred - obs) / Var(obs)

Panels:
  A  ΔBPS histogram (behavior - vision) per subject
  B  ccnorm scatter: behavior vs vision per neuron
  C  single-trial r^2 scatter: behavior vs vision per neuron
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill
import torch
from tqdm import tqdm

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Analysis parameters (match fig3 where applicable)
# ---------------------------------------------------------------------------
RECOMPUTE = True

DT = 1 / 120
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 200
CCNORM_N_SPLITS = 500
CCMAX_THRESHOLD = 0.85

SUBJECTS = ["Allen", "Logan"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}

BEHAVIOR_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)
VISION_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-04-13_11-10-15_learned_resnet_none_convgru_gaussian/"
    "learned_resnet_none_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)

FIG_DIR = FIGURES_DIR / "behavior-vs-vision"
STAT_DIR = STATS_DIR / "behavior-vs-vision"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = CACHE_DIR / "behavior_vs_vision_fixrsvp.pkl"

try:
    get_ipython()  # type: ignore[name-defined]
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False


def show_or_close(fig):
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


def subject_from_session(session_name):
    return session_name.split("_")[0]


def find_best_ckpt(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("epoch=*-val_bps_overall=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No val_bps_overall checkpoints in {ckpt_dir}")
    return max(ckpts, key=lambda p: float(p.stem.split("val_bps_overall=")[1]))


# %% Load both digital twin models
from DataYatesV1 import get_free_device

DEVICE = get_free_device()

if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from eval.eval_stack_multidataset import load_model

beh_ckpt = find_best_ckpt(BEHAVIOR_DIR)
vis_ckpt = find_best_ckpt(VISION_DIR)
print(f"Behavior ckpt:    {beh_ckpt.name}")
print(f"Vision-only ckpt: {vis_ckpt.name}")

beh_model, beh_info = load_model(checkpoint_path=str(beh_ckpt), device=str(DEVICE))
beh_model.model.eval()
beh_model.model.convnet.use_checkpointing = False
print(f"Behavior model loaded: {beh_info['experiment']}, epoch {beh_info['epoch']}")

vis_model, vis_info = load_model(checkpoint_path=str(vis_ckpt), device=str(DEVICE))
vis_model.model.eval()
vis_model.model.convnet.use_checkpointing = False
print(f"Vision-only model loaded: {vis_info['experiment']}, epoch {vis_info['epoch']}")

assert list(beh_model.names) == list(vis_model.names), (
    "Behavior and vision models have different datasets — cannot align cells."
)


# %% Run fixRSVP inference for both models, per session
from eval.eval_stack_utils import (
    load_single_dataset,
    run_model,
    rescale_rhat,
    ccnorm_split_half_variable_trials,
    bits_per_spike,
)


def _compute_metrics(robs_used, rhat_used, dfs_used):
    """Affine-rescale rhat, then compute BPS, ccnorm, single-trial r^2."""
    n_trials, n_time, n_neurons = robs_used.shape
    rhat_flat = rhat_used.reshape(n_trials * n_time, n_neurons)
    robs_flat = robs_used.reshape(n_trials * n_time, n_neurons)
    dfs_flat = dfs_used.reshape(n_trials * n_time, n_neurons)

    rhat_rescaled, _ = rescale_rhat(
        torch.from_numpy(robs_flat),
        torch.from_numpy(rhat_flat),
        torch.from_numpy(dfs_flat),
        mode="affine",
    )
    rhat_rescaled_np = rhat_rescaled.reshape(n_trials, n_time, n_neurons).detach().cpu().numpy()

    # BPS on rescaled rates (same null model across models -> fair comparison).
    # Sanitize dfs to a clean 0/1 mask — NaN entries from trial-aligned init
    # would otherwise propagate through the sums in bits_per_spike.
    dfs_mask = (np.nan_to_num(dfs_flat, nan=0.0) > 0.5).astype(np.float32)
    bps = bits_per_spike(
        rhat_rescaled.float(),
        torch.from_numpy(robs_flat).float(),
        torch.from_numpy(dfs_mask),
    ).detach().cpu().numpy()

    ccnorm1, ccabs1, ccmax1, _, _ = ccnorm_split_half_variable_trials(
        robs_used, rhat_rescaled_np, dfs_used,
        n_splits=CCNORM_N_SPLITS, return_components=True,
    )
    ccnorm2, ccabs2, ccmax2, _, _ = ccnorm_split_half_variable_trials(
        robs_used, rhat_rescaled_np, dfs_used,
        n_splits=CCNORM_N_SPLITS, return_components=True,
    )
    unstable = (ccnorm1 - ccnorm2) ** 2 > 0.01
    ccnorm = 0.5 * (ccnorm1 + ccnorm2)
    ccabs = 0.5 * (ccabs1 + ccabs2)
    ccmax = 0.5 * (ccmax1 + ccmax2)
    ccnorm[unstable] = np.nan

    # Single-trial r^2 (mask invalid bins)
    rhat_masked = rhat_rescaled_np.copy()
    robs_masked = robs_used.copy()
    rhat_masked[dfs_used == 0] = np.nan
    robs_masked[dfs_used == 0] = np.nan
    residuals = rhat_masked - robs_masked
    ve = 1 - np.nanvar(residuals, axis=(0, 1)) / np.nanvar(robs_masked, axis=(0, 1))

    return {
        "bps": bps,
        "ccnorm": ccnorm,
        "ccabs": ccabs,
        "ccmax": ccmax,
        "ve": ve,
    }


if CACHE_PATH.exists() and not RECOMPUTE:
    print(f"Loading cached results from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        session_results = dill.load(f)
else:
    session_results = []

    for dataset_idx in range(len(beh_model.names)):
        session_name = beh_model.names[dataset_idx]
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            print(f"Skipping {session_name} (subject {subject} not in {SUBJECTS})")
            continue

        print(f"\n--- {session_name} ({subject}) [{dataset_idx+1}/{len(beh_model.names)}] ---")

        try:
            train_data, val_data, dataset_config = load_single_dataset(beh_model, dataset_idx)
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
        rhat_beh = np.full((NT, T, NC), np.nan)
        rhat_vis = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
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

            batch = {'stim': stim, 'behavior': behavior}
            out_beh = run_model(beh_model, batch, dataset_idx=dataset_idx)
            out_vis = run_model(vis_model, batch, dataset_idx=dataset_idx)

            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            rhat_beh[itrial, t_inds] = out_beh['rhat'].detach().cpu().numpy()
            rhat_vis[itrial, t_inds] = out_vis['rhat'].detach().cpu().numpy()
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])

        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < 10:
            print(f"  Skipping: only {good_trials.sum()} good trials")
            continue

        robs = robs[good_trials]
        rhat_beh = rhat_beh[good_trials]
        rhat_vis = rhat_vis[good_trials]
        dfs = dfs[good_trials]

        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[:, iix]
        rhat_beh = rhat_beh[:, iix]
        rhat_vis = rhat_vis[:, iix]
        dfs = dfs[:, iix]

        neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
        if len(neuron_mask) < 3:
            print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
            continue

        robs_used = robs[:, :, neuron_mask]
        rhat_beh_used = rhat_beh[:, :, neuron_mask]
        rhat_vis_used = rhat_vis[:, :, neuron_mask]
        dfs_used = dfs[:, :, neuron_mask]

        n_trials, n_time, n_neurons = robs_used.shape
        print(f"  {n_trials} trials, {n_time} time bins, {n_neurons} neurons")

        print("  Rescaling + metrics: behavior")
        m_beh = _compute_metrics(robs_used, rhat_beh_used, dfs_used)
        print("  Rescaling + metrics: vision")
        m_vis = _compute_metrics(robs_used, rhat_vis_used, dfs_used)

        session_results.append({
            "session": session_name,
            "subject": subject,
            "neuron_mask": neuron_mask,
            "n_trials": n_trials,
            "n_time": n_time,
            "n_neurons": n_neurons,
            "behavior": m_beh,
            "vision": m_vis,
        })

        print(f"  [beh] ccnorm median={np.nanmedian(m_beh['ccnorm']):.3f}, "
              f"bps median={np.nanmedian(m_beh['bps']):.3f}")
        print(f"  [vis] ccnorm median={np.nanmedian(m_vis['ccnorm']):.3f}, "
              f"bps median={np.nanmedian(m_vis['bps']):.3f}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {CACHE_PATH}")


# ---------------------------------------------------------------------------
# Flatten per-neuron arrays across sessions, tracking subject identity
# ---------------------------------------------------------------------------
def _stack(key_model, key_metric):
    return np.concatenate([sr[key_model][key_metric] for sr in session_results])


bps_beh = _stack("behavior", "bps")
bps_vis = _stack("vision", "bps")
ccnorm_beh = _stack("behavior", "ccnorm")
ccnorm_vis = _stack("vision", "ccnorm")
ccmax_beh = _stack("behavior", "ccmax")
ccmax_vis = _stack("vision", "ccmax")
ve_beh = _stack("behavior", "ve")
ve_vis = _stack("vision", "ve")

subjects = np.concatenate([
    np.full(sr["n_neurons"], sr["subject"]) for sr in session_results
])

# Reliability: either model's ccmax passes threshold
ccmax_best = np.fmax(ccmax_beh, ccmax_vis)
good = ccmax_best > CCMAX_THRESHOLD

print(f"\nTotal neurons: {len(bps_beh)} "
      f"({(subjects == 'Allen').sum()} Allen, {(subjects == 'Logan').sum()} Logan)")
print(f"Good neurons (ccmax > {CCMAX_THRESHOLD}): {good.sum()}")


# %% Panel A: ΔBPS histogram per subject
from scipy.stats import wilcoxon

fig_a, axes_a = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
bins = np.linspace(-0.2, 0.5, 31)

for ax, subj in zip(axes_a, SUBJECTS):
    mask = (subjects == subj) & np.isfinite(bps_beh) & np.isfinite(bps_vis)
    diff = bps_beh[mask] - bps_vis[mask]
    med = np.median(diff)
    stat, p = wilcoxon(diff, alternative="greater")
    ax.hist(diff, bins=bins, color=SUBJECT_COLORS[subj],
            edgecolor="white", alpha=0.7)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(med, color=SUBJECT_COLORS[subj], lw=2, ls=(0, (1, 1)),
               label=f"median={med:.3f}")
    ax.set_xlabel(r"$\Delta$BPS (behavior − vision)")
    ax.set_title(f"{subj} (N={mask.sum()})")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"Panel A — {subj} (N={mask.sum()}): ΔBPS median={med:.3f}, "
          f"Wilcoxon stat={stat:.1f}, p={p:.3g}")

axes_a[0].set_ylabel("Count")
fig_a.suptitle("fixRSVP: ΔBPS (behavior − vision) on rescaled predictions")
fig_a.tight_layout()
fig_a.savefig(FIG_DIR / "fixrsvp_panel_a_delta_bps.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_a)


# %% Panel B: ccnorm scatter (behavior vs vision)
fig_b, ax_b = plt.subplots(figsize=(3.5, 3.5))

for subj in SUBJECTS:
    mask = (subjects == subj) & good & np.isfinite(ccnorm_beh) & np.isfinite(ccnorm_vis)
    if not mask.any():
        continue
    ax_b.scatter(ccnorm_vis[mask], ccnorm_beh[mask],
                 s=8, alpha=0.5, color=SUBJECT_COLORS[subj], label=subj)

ax_b.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)
ax_b.set_xlabel("ccnorm (vision)")
ax_b.set_ylabel("ccnorm (behavior)")
ax_b.set_title("fixRSVP ccnorm")
ax_b.legend(frameon=False, fontsize=8)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.set_aspect("equal")

fig_b.tight_layout()
fig_b.savefig(FIG_DIR / "fixrsvp_panel_b_ccnorm_scatter.pdf",
              bbox_inches="tight", dpi=300)
show_or_close(fig_b)

for subj in SUBJECTS + ["All"]:
    mask = good & np.isfinite(ccnorm_beh) & np.isfinite(ccnorm_vis)
    if subj != "All":
        mask = mask & (subjects == subj)
    d = ccnorm_beh[mask] - ccnorm_vis[mask]
    stat, p = wilcoxon(d, alternative="greater")
    print(f"Panel B — {subj} (N={mask.sum()}): "
          f"median ccnorm_beh={np.median(ccnorm_beh[mask]):.3f}, "
          f"vis={np.median(ccnorm_vis[mask]):.3f}, "
          f"Δ median={np.median(d):.3f}, Wilcoxon stat={stat:.1f}, p={p:.3g}")


# %% Panel C: single-trial r^2 scatter (behavior vs vision)
fig_c, ax_c = plt.subplots(figsize=(3.5, 3.5))

for subj in SUBJECTS:
    mask = (subjects == subj) & good & np.isfinite(ve_beh) & np.isfinite(ve_vis)
    if not mask.any():
        continue
    ax_c.scatter(ve_vis[mask], ve_beh[mask],
                 s=8, alpha=0.5, color=SUBJECT_COLORS[subj], label=subj)

lims = [0, max(0.4, np.nanmax(np.concatenate([ve_beh[good], ve_vis[good]])) * 1.1)]
ax_c.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
ax_c.set_xlim(lims)
ax_c.set_ylim(lims)
ax_c.set_xlabel("single-trial $r^2$ (vision)")
ax_c.set_ylabel("single-trial $r^2$ (behavior)")
ax_c.set_title("fixRSVP single-trial $r^2$")
ax_c.legend(frameon=False, fontsize=8)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.set_aspect("equal")

fig_c.tight_layout()
fig_c.savefig(FIG_DIR / "fixrsvp_panel_c_r2_scatter.pdf",
              bbox_inches="tight", dpi=300)
show_or_close(fig_c)

for subj in SUBJECTS + ["All"]:
    mask = good & np.isfinite(ve_beh) & np.isfinite(ve_vis)
    if subj != "All":
        mask = mask & (subjects == subj)
    d = ve_beh[mask] - ve_vis[mask]
    stat, p = wilcoxon(d, alternative="greater")
    print(f"Panel C — {subj} (N={mask.sum()}): "
          f"median r^2_beh={np.median(ve_beh[mask]):.3f}, "
          f"vis={np.median(ve_vis[mask]):.3f}, "
          f"Δ median={np.median(d):.3f}, Wilcoxon stat={stat:.1f}, p={p:.3g}")


print(f"\nAll panels saved to: {FIG_DIR}")
print("Done.")
