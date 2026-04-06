"""
H2: Multi-session replication of time-resolved FEM-corrected noise correlations.

Replicates the h1b sliding-window analysis across all 30 sessions (14 Allen, 16 Logan).
Produces per-subject summaries and a combined figure.
"""
import sys
import os
import warnings
import time

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats
from scipy.interpolate import interp1d

from VisionCore.covariance import (
    align_fixrsvp_trials,
    extract_valid_segments,
    extract_windows,
    estimate_rate_covariance,
    bagged_split_half_psth_covariance,
    cov_to_corr,
    get_upper_triangle,
)
from VisionCore.subspace import project_to_psd
from VisionCore.stats import fisher_z_mean
from VisionCore.paths import VISIONCORE_ROOT

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parameters
DT = 1 / 240
T_COUNT = 2
T_HIST = 10
MIN_SEG_LEN = 36
N_BINS = 15
MIN_TOTAL_SPIKES = 500
SLIDING_WINDOW = 15
MIN_TRIALS_PER_TIME = 10
MIN_VALID_BINS = 15
INTERP_GRID = 50  # common normalized x-axis grid
DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)

ALLEN_SESSIONS = [
    "Allen_2022-02-16", "Allen_2022-02-18", "Allen_2022-02-24", "Allen_2022-03-02",
    "Allen_2022-03-04", "Allen_2022-03-30", "Allen_2022-04-01", "Allen_2022-04-06",
    "Allen_2022-04-08", "Allen_2022-04-13", "Allen_2022-04-15", "Allen_2022-06-01",
    "Allen_2022-06-10", "Allen_2022-08-05",
]

LOGAN_SESSIONS = [
    "Logan_2019-12-20", "Logan_2019-12-23", "Logan_2019-12-24", "Logan_2019-12-26",
    "Logan_2019-12-30", "Logan_2019-12-31", "Logan_2020-01-06", "Logan_2020-01-07",
    "Logan_2020-01-09", "Logan_2020-01-10", "Logan_2020-01-15", "Logan_2020-02-28",
    "Logan_2020-02-29", "Logan_2020-03-02", "Logan_2020-03-04", "Logan_2020-03-06",
]


# ---------------------------------------------------------------------------
# Utilities
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
            return "cpu"
        print(f"Selected GPU {best_gpu} with {free_mem[best_gpu]/1e9:.1f} GB free")
        return f"cuda:{best_gpu}"
    return "cpu"


def load_session(session_name, dataset_configs):
    cfg = None
    for c in dataset_configs:
        if c["session"] == session_name:
            cfg = c
            break
    if cfg is None:
        return None

    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    from models.data import prepare_data

    # Suppress verbose data loading output
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        train_data, val_data, cfg = prepare_data(cfg, strict=False)

    dset_idx = train_data.get_dataset_index("fixrsvp")
    if dset_idx is None:
        return None
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset, valid_time_bins=120, min_fix_dur=20,
        min_total_spikes=MIN_TOTAL_SPIKES,
    )
    return robs, eyepos, valid_mask, neuron_mask, meta


def extract_data(robs, eyepos, valid_mask, device):
    robs_clean = np.nan_to_num(robs, nan=0.0)
    eyepos_clean = np.nan_to_num(eyepos, nan=0.0)
    device_obj = torch.device(device)
    robs_t = torch.tensor(robs_clean, dtype=torch.float32, device=device_obj)
    eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32, device=device_obj)
    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_windows(
        robs_t, eyepos_t, segments, T_COUNT, T_HIST, device=device
    )
    return SpikeCounts, EyeTraj, T_idx, idx_tr


def fz_from_cov(C_noise, use_psd=True):
    if use_psd:
        C_psd = project_to_psd(C_noise)
    else:
        C_psd = C_noise.copy()
    R = cov_to_corr(C_psd)
    tri = get_upper_triangle(R)
    return fisher_z_mean(tri)


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
    T_np = T_idx.detach().cpu().numpy()
    mask = np.isin(T_np, time_bins)
    idx = torch.from_numpy(np.where(mask)[0]).to(SpikeCounts.device)
    return SpikeCounts[idx], EyeTraj[idx], T_idx[idx]


def compute_window(SpikeCounts, EyeTraj, T_idx, time_bins):
    """Compute fz_U, fz_C, Dz for a subset of time bins."""
    S_sub, Eye_sub, T_sub = filter_to_time_bins(SpikeCounts, EyeTraj, T_idx, time_bins)
    S_np = S_sub.detach().cpu().numpy().astype(np.float64)

    # Total covariance
    S_centered = S_np - S_np.mean(axis=0, keepdims=True)
    C_total = (S_centered.T @ S_centered) / (S_np.shape[0] - 1)

    # PSTH covariance (split-half)
    C_psth, _ = bagged_split_half_psth_covariance(
        S_sub, T_sub, n_boot=20, min_trials_per_time=MIN_TRIALS_PER_TIME,
        weighting='pair_count'
    )

    # Uncorrected noise
    C_noiseU = C_total - C_psth
    fz_U = fz_from_cov(C_noiseU, use_psd=True)

    # FEM-corrected noise
    try:
        Crate, Erate, Ceye, bin_centers, count_e, bin_edges = estimate_rate_covariance(
            S_sub, Eye_sub, T_sub, n_bins=N_BINS, Ctotal=C_total, intercept_mode='linear'
        )
        C_noiseC = C_total - Crate
        fz_C = fz_from_cov(C_noiseC, use_psd=True)
    except Exception as e:
        fz_C = np.nan

    Dz = fz_C - fz_U if not np.isnan(fz_C) else np.nan
    return fz_U, fz_C, Dz


def process_session(session_name, device, dataset_configs):
    """Process one session: returns dict with results or None if skipped."""
    print(f"\n{'='*60}")
    print(f"Processing: {session_name}")
    print(f"{'='*60}")
    t0 = time.time()

    try:
        result = load_session(session_name, dataset_configs)
    except Exception as e:
        print(f"  SKIP: Failed to load session: {e}")
        return None

    if result is None:
        print(f"  SKIP: Session not found or no fixrsvp data")
        return None

    # Check for empty results
    if any(x is None for x in result):
        print(f"  SKIP: Incomplete data returned")
        return None

    robs, eyepos, valid_mask, neuron_mask, meta = result
    n_neurons = meta['n_neurons_used']
    print(f"  Neurons: {n_neurons}, Trials: {meta['n_trials_good']}/{meta['n_trials_total']}")

    if n_neurons < 5:
        print(f"  SKIP: Too few neurons ({n_neurons})")
        return None

    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    valid_times, trial_counts = get_valid_time_bins(T_idx)
    n_valid = len(valid_times)
    print(f"  Valid time bins: {n_valid}")

    if n_valid < MIN_VALID_BINS:
        print(f"  SKIP: Only {n_valid} valid time bins (need >= {MIN_VALID_BINS})")
        return None

    # Sliding window analysis
    n_windows = n_valid - SLIDING_WINDOW + 1
    print(f"  Computing {n_windows} sliding windows...")

    slide_fz_U = []
    slide_fz_C = []
    slide_Dz = []

    for i in range(n_windows):
        bins = valid_times[i:i + SLIDING_WINDOW]
        fz_U, fz_C, Dz = compute_window(SpikeCounts, EyeTraj, T_idx, bins)
        slide_fz_U.append(fz_U)
        slide_fz_C.append(fz_C)
        slide_Dz.append(Dz)

        if (i + 1) % 10 == 0 or i == n_windows - 1:
            print(f"    Window {i+1}/{n_windows}: fz_U={fz_U:.4f}, fz_C={fz_C:.4f}, Dz={Dz:.4f}")

    slide_fz_U = np.array(slide_fz_U)
    slide_fz_C = np.array(slide_fz_C)
    slide_Dz = np.array(slide_Dz)

    # Spearman correlations
    positions = np.arange(n_windows)
    rho_U, p_U = sp_stats.spearmanr(positions, slide_fz_U)
    rho_C, p_C = sp_stats.spearmanr(positions, slide_fz_C)

    # For Dz, handle NaN
    valid_dz = ~np.isnan(slide_Dz)
    if valid_dz.sum() >= 5:
        rho_Dz, p_Dz = sp_stats.spearmanr(positions[valid_dz], slide_Dz[valid_dz])
    else:
        rho_Dz, p_Dz = np.nan, np.nan

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. rho_U={rho_U:.3f}, rho_C={rho_C:.3f}, rho_Dz={rho_Dz:.3f}")

    # Interpolate onto common grid [0, 1]
    x_norm = np.linspace(0, 1, n_windows)
    x_common = np.linspace(0, 1, INTERP_GRID)

    interp_fz_U = interp1d(x_norm, slide_fz_U, kind='linear')(x_common)
    interp_fz_C = interp1d(x_norm, slide_fz_C, kind='linear')(x_common)
    if valid_dz.all():
        interp_Dz = interp1d(x_norm, slide_Dz, kind='linear')(x_common)
    else:
        # Interpolate only valid points
        interp_Dz = np.full(INTERP_GRID, np.nan)
        if valid_dz.sum() >= 2:
            interp_Dz = interp1d(x_norm[valid_dz], slide_Dz[valid_dz],
                                 kind='linear', fill_value='extrapolate')(x_common)

    # Free GPU memory
    del SpikeCounts, EyeTraj, T_idx, idx_tr
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        'session': session_name,
        'n_neurons': n_neurons,
        'n_valid_bins': n_valid,
        'n_windows': n_windows,
        'rho_U': rho_U, 'p_U': p_U,
        'rho_C': rho_C, 'p_C': p_C,
        'rho_Dz': rho_Dz, 'p_Dz': p_Dz,
        'interp_fz_U': interp_fz_U,
        'interp_fz_C': interp_fz_C,
        'interp_Dz': interp_Dz,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    t_start = time.time()

    # Load dataset configs once
    sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs
    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))

    all_sessions = ALLEN_SESSIONS + LOGAN_SESSIONS
    results = {}

    for session_name in all_sessions:
        # Reload configs each time since prepare_data may mutate them
        dataset_configs_fresh = load_dataset_configs(str(DATASET_CONFIGS_PATH))
        r = process_session(session_name, device, dataset_configs_fresh)
        if r is not None:
            results[session_name] = r

    # Split by subject
    allen_results = [results[s] for s in ALLEN_SESSIONS if s in results]
    logan_results = [results[s] for s in LOGAN_SESSIONS if s in results]

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(allen_results)} Allen sessions, {len(logan_results)} Logan sessions")
    print(f"Total time: {(time.time() - t_start)/60:.1f} min")
    print(f"{'='*70}")

    # ===== Figure =====
    x_common = np.linspace(0, 1, INTERP_GRID)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = [
        ('interp_fz_U', 'fz_U (uncorrected)', 'tab:red'),
        ('interp_fz_C', 'fz_C (FEM-corrected)', 'tab:blue'),
        ('interp_Dz', 'Dz = fz_C - fz_U', 'tab:purple'),
    ]

    for row_idx, (subject, sess_results) in enumerate([('Allen', allen_results), ('Logan', logan_results)]):
        for col_idx, (key, label, color) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            traces = []
            for r in sess_results:
                trace = r[key]
                if not np.any(np.isnan(trace)):
                    traces.append(trace)
                    ax.plot(x_common, trace, color=color, alpha=0.25, linewidth=0.8)

            if traces:
                mean_trace = np.mean(traces, axis=0)
                ax.plot(x_common, mean_trace, color=color, linewidth=2.5, label=f'Mean (n={len(traces)})')

            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Fraction of fixation duration')
            if col_idx == 0:
                ax.set_ylabel(subject, fontsize=14, fontweight='bold')
            if row_idx == 0:
                ax.set_title(label, fontsize=11)
            ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, 'h2_multi_session.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    # ===== Report =====
    report_path = os.path.join(SAVE_DIR, 'h2_report.md')
    with open(report_path, 'w') as f:
        f.write("# H2: Multi-Session Replication of Time-Resolved FEM-Corrected Noise Correlations\n\n")

        for subject, sess_list, all_list in [
            ('Allen', allen_results, ALLEN_SESSIONS),
            ('Logan', logan_results, LOGAN_SESSIONS),
        ]:
            f.write(f"## {subject} ({len(sess_list)}/{len(all_list)} sessions)\n\n")

            # Per-session table
            f.write("| Session | n_neurons | n_valid_bins | rho_U | p_U | rho_C | p_C | rho_Dz | p_Dz |\n")
            f.write("|---------|-----------|-------------|-------|-----|-------|-----|--------|------|\n")
            for r in sess_list:
                f.write(f"| {r['session']} | {r['n_neurons']} | {r['n_valid_bins']} | "
                        f"{r['rho_U']:.3f} | {r['p_U']:.3e} | "
                        f"{r['rho_C']:.3f} | {r['p_C']:.3e} | "
                        f"{r['rho_Dz']:.3f} | {r['p_Dz']:.3e} |\n")
            f.write("\n")

            # Per-subject aggregate
            if sess_list:
                rhos_U = [r['rho_U'] for r in sess_list]
                rhos_C = [r['rho_C'] for r in sess_list]
                rhos_Dz = [r['rho_Dz'] for r in sess_list if not np.isnan(r['rho_Dz'])]
                ps_U = [r['p_U'] for r in sess_list]
                ps_C = [r['p_C'] for r in sess_list]
                ps_Dz = [r['p_Dz'] for r in sess_list if not np.isnan(r['p_Dz'])]

                frac_sig_U = sum(1 for p in ps_U if p < 0.05) / len(ps_U)
                frac_sig_C = sum(1 for p in ps_C if p < 0.05) / len(ps_C)
                frac_sig_Dz = sum(1 for p in ps_Dz if p < 0.05) / len(ps_Dz) if ps_Dz else 0

                f.write(f"**{subject} Aggregate:**\n\n")
                f.write(f"| Metric | Median rho | Mean rho | Frac sig (p<0.05) |\n")
                f.write(f"|--------|-----------|---------|-------------------|\n")
                f.write(f"| fz_U | {np.median(rhos_U):.3f} | {np.mean(rhos_U):.3f} | {frac_sig_U:.2f} ({sum(1 for p in ps_U if p < 0.05)}/{len(ps_U)}) |\n")
                f.write(f"| fz_C | {np.median(rhos_C):.3f} | {np.mean(rhos_C):.3f} | {frac_sig_C:.2f} ({sum(1 for p in ps_C if p < 0.05)}/{len(ps_C)}) |\n")
                if rhos_Dz:
                    f.write(f"| Dz | {np.median(rhos_Dz):.3f} | {np.mean(rhos_Dz):.3f} | {frac_sig_Dz:.2f} ({sum(1 for p in ps_Dz if p < 0.05)}/{len(ps_Dz)}) |\n")
                f.write("\n")

        # Overall tests
        f.write("## Overall Statistical Tests\n\n")

        all_results = allen_results + logan_results
        if len(all_results) >= 3:
            rhos_U_all = [r['rho_U'] for r in all_results]
            rhos_C_all = [r['rho_C'] for r in all_results]
            rhos_Dz_all = [r['rho_Dz'] for r in all_results if not np.isnan(r['rho_Dz'])]

            f.write("**Wilcoxon signed-rank test (H0: median rho = 0):**\n\n")
            for name, rhos in [('fz_U', rhos_U_all), ('fz_C', rhos_C_all), ('Dz', rhos_Dz_all)]:
                if len(rhos) >= 3:
                    stat, pval = sp_stats.wilcoxon(rhos, alternative='two-sided')
                    median_rho = np.median(rhos)
                    f.write(f"- **{name}**: median rho = {median_rho:.3f}, "
                            f"W = {stat:.1f}, p = {pval:.4e} "
                            f"{'(significant)' if pval < 0.05 else '(not significant)'}\n")

            f.write("\n**Sign test (fraction of sessions with negative rho for fz_U):**\n\n")
            n_neg_U = sum(1 for r in rhos_U_all if r < 0)
            n_total = len(rhos_U_all)
            sign_p = sp_stats.binomtest(n_neg_U, n_total, 0.5).pvalue
            f.write(f"- fz_U: {n_neg_U}/{n_total} sessions with rho < 0, "
                    f"binomial p = {sign_p:.4e}\n")

            n_neg_C = sum(1 for r in rhos_C_all if r < 0)
            sign_p_C = sp_stats.binomtest(n_neg_C, len(rhos_C_all), 0.5).pvalue
            f.write(f"- fz_C: {n_neg_C}/{len(rhos_C_all)} sessions with rho < 0, "
                    f"binomial p = {sign_p_C:.4e}\n")

            # Paired comparison: is rho_U more negative than rho_C?
            f.write("\n**Paired Wilcoxon: rho_U vs rho_C (is uncorrected trend stronger?):**\n\n")
            paired_results = [(r['rho_U'], r['rho_C']) for r in all_results]
            diff = [u - c for u, c in paired_results]
            stat, pval = sp_stats.wilcoxon(diff, alternative='two-sided')
            f.write(f"- Median difference (rho_U - rho_C) = {np.median(diff):.3f}, "
                    f"W = {stat:.1f}, p = {pval:.4e}\n")

        f.write("\n")

    print(f"Report saved to {report_path}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("QUANTITATIVE SUMMARY")
    print("=" * 70)

    for subject, sess_list in [('Allen', allen_results), ('Logan', logan_results)]:
        print(f"\n--- {subject} ({len(sess_list)} sessions) ---")
        if sess_list:
            rhos_U = [r['rho_U'] for r in sess_list]
            rhos_C = [r['rho_C'] for r in sess_list]
            print(f"  fz_U rho: median={np.median(rhos_U):.3f}, mean={np.mean(rhos_U):.3f}")
            print(f"  fz_C rho: median={np.median(rhos_C):.3f}, mean={np.mean(rhos_C):.3f}")
            print(f"  Sig fz_U: {sum(1 for r in sess_list if r['p_U'] < 0.05)}/{len(sess_list)}")
            print(f"  Sig fz_C: {sum(1 for r in sess_list if r['p_C'] < 0.05)}/{len(sess_list)}")

    all_r = allen_results + logan_results
    if all_r:
        rhos_U_all = [r['rho_U'] for r in all_r]
        rhos_C_all = [r['rho_C'] for r in all_r]
        print(f"\n--- Overall ({len(all_r)} sessions) ---")
        print(f"  fz_U rho: median={np.median(rhos_U_all):.3f}")
        print(f"  fz_C rho: median={np.median(rhos_C_all):.3f}")
        if len(all_r) >= 3:
            stat, pval = sp_stats.wilcoxon(rhos_U_all)
            print(f"  Wilcoxon fz_U: W={stat:.1f}, p={pval:.4e}")
            stat, pval = sp_stats.wilcoxon(rhos_C_all)
            print(f"  Wilcoxon fz_C: W={stat:.1f}, p={pval:.4e}")


if __name__ == "__main__":
    main()
