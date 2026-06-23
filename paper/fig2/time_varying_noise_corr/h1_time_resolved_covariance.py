"""
Time-resolved covariance analysis.

Tests whether PSTH covariance and noise covariance change systematically
as a function of temporal position within fixations. Early time bins
contain all trials while late bins contain only long-fixation trials,
so any systematic trend would indicate that covariance structure depends
on trial composition or time within fixation.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats

from VisionCore.covariance import (
    align_fixrsvp_trials,
    extract_valid_segments,
    extract_windows,
    bagged_split_half_psth_covariance,
    cov_to_corr,
    get_upper_triangle,
)
from VisionCore.subspace import project_to_psd
from VisionCore.stats import fisher_z_mean, fisher_z
from VisionCore.paths import VISIONCORE_ROOT

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parameters
DT = 1 / 240
T_COUNT = 2
T_HIST = 10
MIN_SEG_LEN = 36
MIN_TOTAL_SPIKES = 500
SESSION_NAME = "Allen_2022-04-13"
DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
SLIDING_WINDOW = 10  # number of time bins per sliding window
MIN_TRIALS_PER_TIME = 10


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        free_mem = []
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                free_mem.append(free)
            except Exception:
                free_mem.append(0)
        best_gpu = int(np.argmax(free_mem))
        if free_mem[best_gpu] < 1e9:
            print("No GPU with enough free memory, using CPU")
            return "cpu"
        dev = f"cuda:{best_gpu}"
        print(f"Selected GPU {best_gpu} with {free_mem[best_gpu]/1e9:.1f} GB free")
        return dev
    return "cpu"


def load_real_session():
    sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs
    from models.data import prepare_data

    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    cfg = None
    for c in dataset_configs:
        if c["session"] == SESSION_NAME:
            cfg = c
            break
    assert cfg is not None, f"Session {SESSION_NAME} not found"
    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    print(f"Loading {SESSION_NAME}...")
    train_data, val_data, cfg = prepare_data(cfg, strict=False)
    dset_idx = train_data.get_dataset_index("fixrsvp")
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset, valid_time_bins=120, min_fix_dur=20,
        min_total_spikes=MIN_TOTAL_SPIKES,
    )
    print(f"Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
          f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")
    return robs, eyepos, valid_mask, neuron_mask, meta


def extract_data(robs, eyepos, valid_mask, device):
    robs_clean = np.nan_to_num(robs, nan=0.0)
    eyepos_clean = np.nan_to_num(eyepos, nan=0.0)
    device_obj = torch.device(device)
    robs_t = torch.tensor(robs_clean, dtype=torch.float32, device=device_obj)
    eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32, device=device_obj)

    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)
    print(f"Found {len(segments)} valid segments")

    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_windows(
        robs_t, eyepos_t, segments, T_COUNT, T_HIST, device=device
    )
    n_samples, n_cells = SpikeCounts.shape
    print(f"Extracted {n_samples} windows, {n_cells} neurons")
    return SpikeCounts, EyeTraj, T_idx, idx_tr


def off_diag_mean(C):
    n = C.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(np.nanmean(C[mask]))


def fz_from_cov(C_noise, use_psd=True):
    if use_psd:
        C_psd = project_to_psd(C_noise)
    else:
        C_psd = C_noise.copy()
    R = cov_to_corr(C_psd)
    tri = get_upper_triangle(R)
    return fisher_z_mean(tri)


# ---------------------------------------------------------------------------
# Report collector
# ---------------------------------------------------------------------------

report_lines = []


def report(msg=""):
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def get_valid_time_bins(T_idx, min_trials=MIN_TRIALS_PER_TIME):
    """Get sorted list of valid time bins (those with >= min_trials)."""
    T_np = T_idx.detach().cpu().numpy()
    unique_times = np.unique(T_np)
    valid = []
    counts = []
    for t in sorted(unique_times):
        n = np.sum(T_np == t)
        if n >= min_trials:
            valid.append(int(t))
            counts.append(int(n))
    return np.array(valid), np.array(counts)


def filter_to_time_bins(SpikeCounts, T_idx, time_bins):
    """Filter SpikeCounts and T_idx to only include specified time bins."""
    T_np = T_idx.detach().cpu().numpy()
    mask = np.isin(T_np, time_bins)
    S_sub = SpikeCounts[mask]
    T_sub = T_idx[mask]
    return S_sub, T_sub


def compute_total_cov(S_sub, T_sub):
    """Compute total sample covariance for a subset of data."""
    S_np = S_sub.detach().cpu().numpy().astype(np.float64)
    S_centered = S_np - S_np.mean(axis=0, keepdims=True)
    n = S_np.shape[0]
    C_total = (S_centered.T @ S_centered) / (n - 1)
    return C_total


def analyze_subset(SpikeCounts, T_idx, time_bins, label):
    """Compute PSTH cov, total cov, noise cov, noise corr for a subset of time bins."""
    S_sub, T_sub = filter_to_time_bins(SpikeCounts, T_idx, time_bins)
    n_windows = S_sub.shape[0]

    # Trial counts per time bin in this subset
    T_np = T_sub.detach().cpu().numpy()
    unique_t = np.unique(T_np)
    trial_counts = [np.sum(T_np == t) for t in unique_t]

    # Mean firing rate
    S_np = S_sub.detach().cpu().numpy().astype(np.float64)
    mean_rate = S_np.mean(axis=0)  # per neuron
    pop_mean_rate = float(mean_rate.mean())

    # PSTH covariance (split-half)
    C_psth, PSTH_mean = bagged_split_half_psth_covariance(
        S_sub, T_sub, n_boot=20, min_trials_per_time=MIN_TRIALS_PER_TIME,
        weighting='pair_count'
    )
    psth_cov_offdiag = off_diag_mean(C_psth)

    # Total covariance
    C_total = compute_total_cov(S_sub, T_sub)
    total_cov_offdiag = off_diag_mean(C_total)

    # Noise covariance
    C_noise = C_total - C_psth
    noise_cov_offdiag = off_diag_mean(C_noise)

    # Noise correlation (Fisher z)
    fz = fz_from_cov(C_noise, use_psd=True)

    return {
        'label': label,
        'n_time_bins': len(time_bins),
        'n_windows': n_windows,
        'trial_count_mean': float(np.mean(trial_counts)),
        'trial_count_min': int(np.min(trial_counts)),
        'trial_count_max': int(np.max(trial_counts)),
        'pop_mean_rate': pop_mean_rate,
        'mean_rates': mean_rate,
        'psth_cov_offdiag': psth_cov_offdiag,
        'total_cov_offdiag': total_cov_offdiag,
        'noise_cov_offdiag': noise_cov_offdiag,
        'fisher_z_noise_corr': fz,
        'C_psth': C_psth,
        'C_noise': C_noise,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(
        robs, eyepos, valid_mask, device
    )

    valid_times, trial_counts = get_valid_time_bins(T_idx)
    n_valid = len(valid_times)
    n_cells = SpikeCounts.shape[1]

    report("=" * 70)
    report("H1: Time-Resolved Covariance Analysis")
    report("=" * 70)
    report(f"Session: {SESSION_NAME}")
    report(f"Neurons: {n_cells}")
    report(f"Valid time bins: {n_valid}")
    report(f"Trial counts: min={trial_counts.min()}, max={trial_counts.max()}, "
           f"mean={trial_counts.mean():.1f}, CV={trial_counts.std()/trial_counts.mean():.3f}")

    # ----- PART 1: Tercile analysis -----
    report("\n" + "=" * 70)
    report("PART 1: Tercile Analysis")
    report("=" * 70)

    tercile_size = n_valid // 3
    tercile_bins = {
        'early': valid_times[:tercile_size],
        'mid': valid_times[tercile_size:2*tercile_size],
        'late': valid_times[2*tercile_size:],
    }

    tercile_results = {}
    for label in ['early', 'mid', 'late']:
        bins = tercile_bins[label]
        result = analyze_subset(SpikeCounts, T_idx, bins, label)
        tercile_results[label] = result
        report(f"\n--- {label.upper()} tercile ---")
        report(f"  Time bins: {len(bins)} (indices {bins[0]}..{bins[-1]})")
        report(f"  Trial counts: mean={result['trial_count_mean']:.1f}, "
               f"range=[{result['trial_count_min']}, {result['trial_count_max']}]")
        report(f"  Pop mean rate: {result['pop_mean_rate']:.4f}")
        report(f"  PSTH cov (off-diag): {result['psth_cov_offdiag']:.6f}")
        report(f"  Total cov (off-diag): {result['total_cov_offdiag']:.6f}")
        report(f"  Noise cov (off-diag): {result['noise_cov_offdiag']:.6f}")
        report(f"  Fisher-z noise corr: {result['fisher_z_noise_corr']:.6f}")

    # Tercile trend
    report("\n--- Tercile trends ---")
    for metric in ['psth_cov_offdiag', 'noise_cov_offdiag', 'fisher_z_noise_corr', 'pop_mean_rate']:
        vals = [tercile_results[k][metric] for k in ['early', 'mid', 'late']]
        report(f"  {metric}: early={vals[0]:.6f} -> mid={vals[1]:.6f} -> late={vals[2]:.6f}")
        pct_change = (vals[2] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else float('nan')
        report(f"    early-to-late change: {pct_change:+.1f}%")

    # ----- PART 2: Sliding window analysis -----
    report("\n" + "=" * 70)
    report("PART 2: Sliding Window Analysis")
    report("=" * 70)
    report(f"Window size: {SLIDING_WINDOW} time bins, step: 1")

    n_windows_slide = n_valid - SLIDING_WINDOW + 1
    report(f"Number of sliding windows: {n_windows_slide}")

    slide_centers = []
    slide_psth_cov = []
    slide_noise_cov = []
    slide_fz = []
    slide_mean_rate = []
    slide_mean_rate_per_neuron = []

    for i in range(n_windows_slide):
        bins = valid_times[i:i + SLIDING_WINDOW]
        center = float(np.mean(bins))
        slide_centers.append(center)

        S_sub, T_sub = filter_to_time_bins(SpikeCounts, T_idx, bins)
        S_np = S_sub.detach().cpu().numpy().astype(np.float64)

        # Mean firing rate
        mean_rate = S_np.mean(axis=0)
        slide_mean_rate.append(float(mean_rate.mean()))
        slide_mean_rate_per_neuron.append(mean_rate)

        # PSTH covariance
        C_psth, _ = bagged_split_half_psth_covariance(
            S_sub, T_sub, n_boot=20, min_trials_per_time=MIN_TRIALS_PER_TIME,
            weighting='pair_count'
        )
        slide_psth_cov.append(off_diag_mean(C_psth))

        # Total covariance
        C_total = compute_total_cov(S_sub, T_sub)
        total_offdiag = off_diag_mean(C_total)

        # Noise covariance
        C_noise = C_total - C_psth
        slide_noise_cov.append(off_diag_mean(C_noise))

        # Noise correlation
        fz = fz_from_cov(C_noise, use_psd=True)
        slide_fz.append(fz)

        if (i + 1) % 10 == 0 or i == n_windows_slide - 1:
            print(f"  Window {i+1}/{n_windows_slide} done")

    slide_centers = np.array(slide_centers)
    slide_psth_cov = np.array(slide_psth_cov)
    slide_noise_cov = np.array(slide_noise_cov)
    slide_fz = np.array(slide_fz)
    slide_mean_rate = np.array(slide_mean_rate)
    slide_mean_rate_per_neuron = np.array(slide_mean_rate_per_neuron)  # (n_windows, n_cells)

    # Spearman correlations
    report("\n--- Spearman correlations with temporal position ---")
    for name, vals in [
        ('PSTH cov (off-diag)', slide_psth_cov),
        ('Noise cov (off-diag)', slide_noise_cov),
        ('Fisher-z noise corr', slide_fz),
        ('Mean firing rate', slide_mean_rate),
    ]:
        rho, pval = sp_stats.spearmanr(slide_centers, vals)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        report(f"  {name}: rho={rho:.4f}, p={pval:.4e} {sig}")

    # ----- PART 3: Figure -----
    report("\n" + "=" * 70)
    report("PART 3: Generating figure")
    report("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: Trial count vs time bin index
    ax = axes[0, 0]
    ax.bar(valid_times, trial_counts, color='steelblue', alpha=0.7, width=1.0)
    # Mark tercile boundaries
    for label, color in [('early', 'green'), ('mid', 'orange'), ('late', 'red')]:
        bins = tercile_bins[label]
        ax.axvspan(bins[0] - 0.5, bins[-1] + 0.5, alpha=0.1, color=color, label=label)
    ax.set_xlabel('Time bin index')
    ax.set_ylabel('Trial count (n_t)')
    ax.set_title('A. Trial counts across time bins')
    ax.legend(fontsize=8)

    # Panel B: Mean firing rate vs time bin index
    ax = axes[0, 1]
    # Individual neurons as light lines
    T_np = T_idx.detach().cpu().numpy()
    S_np = SpikeCounts.detach().cpu().numpy().astype(np.float64)
    neuron_rates = np.zeros((n_valid, n_cells))
    for i, t in enumerate(valid_times):
        mask_t = T_np == t
        neuron_rates[i] = S_np[mask_t].mean(axis=0)

    for j in range(n_cells):
        ax.plot(valid_times, neuron_rates[:, j], color='gray', alpha=0.2, linewidth=0.5)
    ax.plot(valid_times, neuron_rates.mean(axis=1), color='black', linewidth=2, label='Pop mean')
    ax.set_xlabel('Time bin index')
    ax.set_ylabel('Mean spike count')
    ax.set_title('B. Mean firing rate vs time')
    ax.legend(fontsize=8)

    # Panel C: Sliding-window PSTH covariance
    ax = axes[1, 0]
    ax.plot(slide_centers, slide_psth_cov, color='tab:blue', linewidth=1.5)
    rho_p, pval_p = sp_stats.spearmanr(slide_centers, slide_psth_cov)
    ax.set_xlabel('Time bin index (window center)')
    ax.set_ylabel('PSTH cov (off-diag mean)')
    ax.set_title(f'C. PSTH covariance vs time\n(Spearman rho={rho_p:.3f}, p={pval_p:.3e})')

    # Panel D: Sliding-window noise correlation
    ax = axes[1, 1]
    ax.plot(slide_centers, slide_fz, color='tab:red', linewidth=1.5)
    rho_n, pval_n = sp_stats.spearmanr(slide_centers, slide_fz)
    ax.set_xlabel('Time bin index (window center)')
    ax.set_ylabel('Fisher-z noise correlation')
    ax.set_title(f'D. Noise correlation vs time\n(Spearman rho={rho_n:.3f}, p={pval_n:.3e})')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'h1_time_resolved_covariance.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    report(f"Figure saved to {fig_path}")

    # ----- Save report -----
    report_path = os.path.join(SAVE_DIR, 'h1_report.md')
    with open(report_path, 'w') as f:
        f.write("# H1: Time-Resolved Covariance Analysis\n\n")
        f.write("```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
