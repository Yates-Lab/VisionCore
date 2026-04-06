"""
FEM-corrected time-resolved noise correlations.

Extends h1 by adding the trajectory-matching correction:
  CnoiseU = Ctotal - Cpsth   (uncorrected)
  CnoiseC = Ctotal - Crate   (FEM-corrected via estimate_rate_covariance)
  Dz      = fz(CnoiseC) - fz(CnoiseU)

Analyzes both terciles and sliding windows to test whether the FEM
correction magnitude changes with temporal position within fixation.
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
    compute_conditional_second_moments,
    estimate_rate_covariance,
    bagged_split_half_psth_covariance,
    cov_to_corr,
    get_upper_triangle,
    fit_intercept_linear,
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
N_BINS = 15
MIN_TOTAL_SPIKES = 500
SESSION_NAME = "Allen_2022-04-13"
DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
SLIDING_WINDOW = 15
MIN_TRIALS_PER_TIME = 10


# ---------------------------------------------------------------------------
# Utility (same as h1)
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
# Core helpers
# ---------------------------------------------------------------------------

def get_valid_time_bins(T_idx, min_trials=MIN_TRIALS_PER_TIME):
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


def filter_to_time_bins(SpikeCounts, EyeTraj, T_idx, time_bins):
    """Filter SpikeCounts, EyeTraj, and T_idx to only include specified time bins."""
    T_np = T_idx.detach().cpu().numpy()
    mask = np.isin(T_np, time_bins)
    idx = torch.from_numpy(np.where(mask)[0]).to(SpikeCounts.device)
    return SpikeCounts[idx], EyeTraj[idx], T_idx[idx]


def compute_total_cov(S_sub, T_sub):
    S_np = S_sub.detach().cpu().numpy().astype(np.float64)
    S_centered = S_np - S_np.mean(axis=0, keepdims=True)
    n = S_np.shape[0]
    C_total = (S_centered.T @ S_centered) / (n - 1)
    return C_total


def analyze_subset_corrected(SpikeCounts, EyeTraj, T_idx, time_bins, label):
    """
    Compute uncorrected and FEM-corrected noise correlations for a subset.

    Returns dict with fz_U, fz_C, Dz, and supporting quantities.
    """
    S_sub, Eye_sub, T_sub = filter_to_time_bins(SpikeCounts, EyeTraj, T_idx, time_bins)
    n_windows = S_sub.shape[0]

    T_np = T_sub.detach().cpu().numpy()
    unique_t = np.unique(T_np)
    trial_counts = [np.sum(T_np == t) for t in unique_t]

    S_np = S_sub.detach().cpu().numpy().astype(np.float64)
    pop_mean_rate = float(S_np.mean())

    # Total covariance
    C_total = compute_total_cov(S_sub, T_sub)

    # --- Uncorrected: CnoiseU = Ctotal - Cpsth ---
    C_psth, _ = bagged_split_half_psth_covariance(
        S_sub, T_sub, n_boot=20, min_trials_per_time=MIN_TRIALS_PER_TIME,
        weighting='pair_count'
    )
    C_noiseU = C_total - C_psth
    fz_U = fz_from_cov(C_noiseU, use_psd=True)

    # --- Corrected: CnoiseC = Ctotal - Crate ---
    Crate, Erate, Ceye, bin_centers, count_e, bin_edges = estimate_rate_covariance(
        S_sub, Eye_sub, T_sub, n_bins=N_BINS, Ctotal=C_total,
        intercept_mode='linear'
    )
    C_noiseC = C_total - Crate
    fz_C = fz_from_cov(C_noiseC, use_psd=True)

    Dz = fz_C - fz_U

    return {
        'label': label,
        'n_time_bins': len(time_bins),
        'n_windows': n_windows,
        'trial_count_mean': float(np.mean(trial_counts)),
        'trial_count_min': int(np.min(trial_counts)),
        'trial_count_max': int(np.max(trial_counts)),
        'pop_mean_rate': pop_mean_rate,
        'fz_U': fz_U,
        'fz_C': fz_C,
        'Dz': Dz,
        'psth_cov_offdiag': off_diag_mean(C_psth),
        'rate_cov_offdiag': off_diag_mean(Crate),
        'noise_covU_offdiag': off_diag_mean(C_noiseU),
        'noise_covC_offdiag': off_diag_mean(C_noiseC),
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
    report("H1b: FEM-Corrected Time-Resolved Noise Correlations")
    report("=" * 70)
    report(f"Session: {SESSION_NAME}")
    report(f"Neurons: {n_cells}")
    report(f"Valid time bins: {n_valid}")
    report(f"Trial counts: min={trial_counts.min()}, max={trial_counts.max()}, "
           f"mean={trial_counts.mean():.1f}")

    # ===== PART 1: Tercile analysis =====
    report("\n" + "=" * 70)
    report("PART 1: Tercile Analysis (Uncorrected vs FEM-Corrected)")
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
        report(f"\nComputing {label.upper()} tercile ({len(bins)} time bins, "
               f"indices {bins[0]}..{bins[-1]})...")
        result = analyze_subset_corrected(SpikeCounts, EyeTraj, T_idx, bins, label)
        tercile_results[label] = result

        report(f"  Windows: {result['n_windows']}")
        report(f"  Trial counts: mean={result['trial_count_mean']:.1f}, "
               f"range=[{result['trial_count_min']}, {result['trial_count_max']}]")
        report(f"  PSTH cov (off-diag):  {result['psth_cov_offdiag']:.6f}")
        report(f"  Rate cov (off-diag):  {result['rate_cov_offdiag']:.6f}")
        report(f"  fz_U (uncorrected):   {result['fz_U']:.6f}")
        report(f"  fz_C (corrected):     {result['fz_C']:.6f}")
        report(f"  Dz (correction):      {result['Dz']:.6f}")

    # Tercile summary table
    report("\n--- Tercile Summary ---")
    report(f"{'Tercile':<10} {'fz_U':>10} {'fz_C':>10} {'Dz':>10}")
    report("-" * 42)
    for label in ['early', 'mid', 'late']:
        r = tercile_results[label]
        report(f"{label:<10} {r['fz_U']:>10.6f} {r['fz_C']:>10.6f} {r['Dz']:>10.6f}")

    # Tercile trends
    report("\n--- Tercile trends ---")
    for metric in ['fz_U', 'fz_C', 'Dz']:
        vals = [tercile_results[k][metric] for k in ['early', 'mid', 'late']]
        pct = (vals[2] - vals[0]) / abs(vals[0]) * 100 if vals[0] != 0 else float('nan')
        report(f"  {metric}: early={vals[0]:.6f} -> mid={vals[1]:.6f} -> late={vals[2]:.6f} "
               f"({pct:+.1f}%)")

    # ===== PART 2: Sliding window analysis =====
    report("\n" + "=" * 70)
    report("PART 2: Sliding Window Analysis")
    report("=" * 70)
    report(f"Window size: {SLIDING_WINDOW} time bins, step: 1")

    n_windows_slide = n_valid - SLIDING_WINDOW + 1
    report(f"Number of sliding windows: {n_windows_slide}")

    slide_centers = []
    slide_fz_U = []
    slide_fz_C = []
    slide_Dz = []
    slide_trial_count = []

    for i in range(n_windows_slide):
        bins = valid_times[i:i + SLIDING_WINDOW]
        center = float(np.mean(bins))
        slide_centers.append(center)

        S_sub, Eye_sub, T_sub = filter_to_time_bins(
            SpikeCounts, EyeTraj, T_idx, bins
        )

        # Trial count for this window
        T_np = T_sub.detach().cpu().numpy()
        slide_trial_count.append(S_sub.shape[0])

        # Total covariance
        C_total = compute_total_cov(S_sub, T_sub)

        # Uncorrected
        C_psth, _ = bagged_split_half_psth_covariance(
            S_sub, T_sub, n_boot=20, min_trials_per_time=MIN_TRIALS_PER_TIME,
            weighting='pair_count'
        )
        C_noiseU = C_total - C_psth
        fz_U = fz_from_cov(C_noiseU, use_psd=True)
        slide_fz_U.append(fz_U)

        # FEM-corrected
        Crate, Erate, Ceye, bin_c, count_e, bin_e = estimate_rate_covariance(
            S_sub, Eye_sub, T_sub, n_bins=N_BINS, Ctotal=C_total,
            intercept_mode='linear'
        )
        C_noiseC = C_total - Crate
        fz_C = fz_from_cov(C_noiseC, use_psd=True)
        slide_fz_C.append(fz_C)

        slide_Dz.append(fz_C - fz_U)

        if (i + 1) % 5 == 0 or i == n_windows_slide - 1:
            print(f"  Window {i+1}/{n_windows_slide} done  "
                  f"(fz_U={fz_U:.4f}, fz_C={fz_C:.4f}, Dz={fz_C - fz_U:.4f})")

    slide_centers = np.array(slide_centers)
    slide_fz_U = np.array(slide_fz_U)
    slide_fz_C = np.array(slide_fz_C)
    slide_Dz = np.array(slide_Dz)
    slide_trial_count = np.array(slide_trial_count)

    # Spearman correlations
    report("\n--- Spearman correlations with temporal position ---")
    spearman_results = {}
    for name, vals in [
        ('fz_U (uncorrected)', slide_fz_U),
        ('fz_C (corrected)', slide_fz_C),
        ('Dz (correction magnitude)', slide_Dz),
    ]:
        rho, pval = sp_stats.spearmanr(slide_centers, vals)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        report(f"  {name}: rho={rho:.4f}, p={pval:.4e} {sig}")
        spearman_results[name] = (rho, pval, sig)

    # ===== PART 3: Figure =====
    report("\n" + "=" * 70)
    report("PART 3: Generating figure")
    report("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: Trial count vs time
    ax = axes[0, 0]
    ax.bar(valid_times, trial_counts, color='steelblue', alpha=0.7, width=1.0)
    for label, color in [('early', 'green'), ('mid', 'orange'), ('late', 'red')]:
        bins = tercile_bins[label]
        ax.axvspan(bins[0] - 0.5, bins[-1] + 0.5, alpha=0.1, color=color, label=label)
    ax.set_xlabel('Time bin index')
    ax.set_ylabel('Trial count')
    ax.set_title('A. Trial counts across time bins')
    ax.legend(fontsize=8)

    # Panel B: Sliding-window uncorrected fz_U
    ax = axes[0, 1]
    ax.plot(slide_centers, slide_fz_U, color='tab:red', linewidth=1.5)
    rho_U, pval_U = sp_stats.spearmanr(slide_centers, slide_fz_U)
    ax.set_xlabel('Time bin index (window center)')
    ax.set_ylabel('Fisher-z noise corr (uncorrected)')
    ax.set_title(f'B. Uncorrected fz_U vs time\n'
                 f'(Spearman rho={rho_U:.3f}, p={pval_U:.3e})')

    # Panel C: Sliding-window corrected fz_C
    ax = axes[1, 0]
    ax.plot(slide_centers, slide_fz_C, color='tab:blue', linewidth=1.5)
    rho_C, pval_C = sp_stats.spearmanr(slide_centers, slide_fz_C)
    ax.set_xlabel('Time bin index (window center)')
    ax.set_ylabel('Fisher-z noise corr (FEM-corrected)')
    ax.set_title(f'C. FEM-corrected fz_C vs time\n'
                 f'(Spearman rho={rho_C:.3f}, p={pval_C:.3e})')

    # Panel D: Sliding-window Dz
    ax = axes[1, 1]
    ax.plot(slide_centers, slide_Dz, color='tab:purple', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    rho_D, pval_D = sp_stats.spearmanr(slide_centers, slide_Dz)
    ax.set_xlabel('Time bin index (window center)')
    ax.set_ylabel('Dz = fz_C - fz_U')
    ax.set_title(f'D. FEM correction magnitude vs time\n'
                 f'(Spearman rho={rho_D:.3f}, p={pval_D:.3e})')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'h1b_fem_corrected.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    report(f"Figure saved to {fig_path}")

    # ===== Save report =====
    report_path = os.path.join(SAVE_DIR, 'h1b_report.md')
    with open(report_path, 'w') as f:
        f.write("# H1b: FEM-Corrected Time-Resolved Noise Correlations\n\n")
        f.write(f"**Session:** {SESSION_NAME}  \n")
        f.write(f"**Neurons:** {n_cells}  \n")
        f.write(f"**Valid time bins:** {n_valid}  \n\n")

        f.write("## Tercile Analysis\n\n")
        f.write("| Tercile | fz_U | fz_C | Dz |\n")
        f.write("|---------|------|------|----|\n")
        for label in ['early', 'mid', 'late']:
            r = tercile_results[label]
            f.write(f"| {label} | {r['fz_U']:.6f} | {r['fz_C']:.6f} | {r['Dz']:.6f} |\n")
        f.write("\n")

        f.write("## Sliding Window Spearman Correlations\n\n")
        f.write(f"Window size: {SLIDING_WINDOW} time bins, {n_windows_slide} windows\n\n")
        f.write("| Metric | rho | p-value | sig |\n")
        f.write("|--------|-----|---------|-----|\n")
        for name, (rho, pval, sig) in spearman_results.items():
            f.write(f"| {name} | {rho:.4f} | {pval:.4e} | {sig} |\n")
        f.write("\n")

        f.write("## Key Findings\n\n")
        # Auto-generate findings
        dz_early = tercile_results['early']['Dz']
        dz_late = tercile_results['late']['Dz']
        dz_rho, dz_pval, dz_sig = spearman_results['Dz (correction magnitude)']

        f.write(f"- Dz (correction magnitude) early={dz_early:.6f}, late={dz_late:.6f}\n")
        if dz_pval < 0.05:
            direction = "increases" if dz_rho > 0 else "decreases"
            f.write(f"- FEM correction magnitude significantly {direction} "
                    f"with time (rho={dz_rho:.4f}, p={dz_pval:.4e})\n")
        else:
            f.write(f"- FEM correction magnitude does NOT significantly change "
                    f"with time (rho={dz_rho:.4f}, p={dz_pval:.4e})\n")

        fzU_rho, fzU_pval, _ = spearman_results['fz_U (uncorrected)']
        fzC_rho, fzC_pval, _ = spearman_results['fz_C (corrected)']
        f.write(f"- Uncorrected noise corr trend: rho={fzU_rho:.4f}, p={fzU_pval:.4e}\n")
        f.write(f"- Corrected noise corr trend: rho={fzC_rho:.4f}, p={fzC_pval:.4e}\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
