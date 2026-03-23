"""
Test the pair-count weighting fix for the noise correlation bias.

Root cause: inconsistent weighting between second moment accumulation
(n_pairs_t-weighted) and mean subtraction (trial-count-weighted) in
estimate_rate_covariance.

Fix: use pair-count-weighted mean rate for the mean subtraction, making
both terms use the same n_pairs_t weighting.
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
from tqdm import tqdm

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
SESSION_NAME = "Allen_2022-04-13"
DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
N_SHUFFLES_GLOBAL = 50
N_SHUFFLES_WITHIN = 20


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

report_lines = []


def report(msg=""):
    print(msg)
    report_lines.append(msg)


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
    vals = C[mask]
    return float(np.nanmean(vals))


def fz_from_cov(C_noise, use_psd=True):
    if use_psd:
        C_psd = project_to_psd(C_noise)
    else:
        C_psd = C_noise.copy()
    R = cov_to_corr(C_psd)
    tri = get_upper_triangle(R)
    return fisher_z_mean(tri)


def global_shuffle_eye(EyeTraj, rng):
    N = EyeTraj.shape[0]
    perm = rng.permutation(N)
    return EyeTraj[perm]


def within_time_bin_shuffle(EyeTraj, T_idx, rng):
    unique_times = torch.unique(T_idx).cpu().numpy()
    perm_indices = torch.arange(EyeTraj.shape[0], device=EyeTraj.device)
    for t in unique_times:
        mask = (T_idx == t).cpu().numpy()
        idx_t = np.where(mask)[0]
        if len(idx_t) < 2:
            continue
        shuffled = rng.permutation(idx_t)
        perm_indices[idx_t] = torch.tensor(shuffled, device=EyeTraj.device)
    return EyeTraj[perm_indices]


# ---------------------------------------------------------------------------
# Fixed estimator (Approach A: pair-count-weighted mean subtraction)
# ---------------------------------------------------------------------------

def estimate_rate_covariance_fixed(SpikeCounts, EyeTraj, T_idx, n_bins=25,
                                    Ctotal=None, intercept_mode='linear'):
    """Fixed version: pair-count-weighted mean subtraction."""
    MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
        SpikeCounts, EyeTraj, T_idx, n_bins=n_bins
    )

    # Compute pair-count-weighted mean rate
    unique_times = torch.unique(T_idx)
    total_pair_weight = 0.0
    weighted_mean = torch.zeros(SpikeCounts.shape[1], device=SpikeCounts.device,
                                dtype=torch.float64)

    for t in unique_times:
        mask = (T_idx == t)
        n_t = mask.sum().item()
        if n_t < 2:
            continue
        n_pairs_t = n_t * (n_t - 1) / 2
        mu_t = SpikeCounts[mask].mean(0).to(torch.float64)
        weighted_mean += n_pairs_t * mu_t
        total_pair_weight += n_pairs_t

    Erate_paired = (weighted_mean / total_pair_weight).detach().cpu().numpy()
    Ceye = MM - Erate_paired[:, None] * Erate_paired[None, :]

    # Same intercept fitting as original
    if intercept_mode == 'linear':
        Crate = fit_intercept_linear(Ceye, bin_centers, count_e, eval_at_first_bin=True)
    elif intercept_mode == 'isotonic':
        from VisionCore.covariance import fit_intercept_pava
        Crate = fit_intercept_pava(Ceye, count_e)
    else:
        Crate = Ceye[0].copy()

    if Ctotal is not None:
        bad_mask = np.diag(Crate) > 0.99 * np.diag(Ctotal)
        Crate[bad_mask, :] = np.nan
        Crate[:, bad_mask] = np.nan
        Ceye[:, bad_mask, :] = np.nan
        Ceye[:, :, bad_mask] = np.nan

    return Crate, Erate_paired, Ceye, bin_centers, count_e, bin_edges


# ---------------------------------------------------------------------------
# Part 2: Real data comparison
# ---------------------------------------------------------------------------

def part2_real_data(SpikeCounts, EyeTraj, T_idx, Ctotal, Cpsth, Erate):
    """Compare original vs fixed pipeline on real (unshuffled) data."""
    report("\n" + "=" * 70)
    report("PART 2: Real data — original vs fixed pipeline")
    report("=" * 70)

    # Original pipeline
    Crate_orig, Erate_orig, Ceye_orig, bc_orig, ce_orig, be_orig = \
        estimate_rate_covariance(
            SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS,
            Ctotal=Ctotal, intercept_mode='linear'
        )

    CnoiseC_orig = 0.5 * ((Ctotal - Crate_orig) + (Ctotal - Crate_orig).T)
    CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)

    fz_U = fz_from_cov(CnoiseU, use_psd=True)
    fz_C_orig = fz_from_cov(CnoiseC_orig, use_psd=True)
    dz_orig = fz_C_orig - fz_U

    # Fixed pipeline (Approach A)
    Crate_fixed, Erate_fixed, Ceye_fixed, bc_fix, ce_fix, be_fix = \
        estimate_rate_covariance_fixed(
            SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS,
            Ctotal=Ctotal, intercept_mode='linear'
        )

    CnoiseC_fixed = 0.5 * ((Ctotal - Crate_fixed) + (Ctotal - Crate_fixed).T)
    fz_C_fixed = fz_from_cov(CnoiseC_fixed, use_psd=True)
    dz_fixed = fz_C_fixed - fz_U

    report(f"\n  fz(CnoiseU)                = {fz_U:.6f}")
    report(f"  fz(CnoiseC_orig)           = {fz_C_orig:.6f}")
    report(f"  fz(CnoiseC_fixed)          = {fz_C_fixed:.6f}")
    report(f"  Dz_real (original)         = {dz_orig:.6f}")
    report(f"  Dz_real (fixed)            = {dz_fixed:.6f}")
    report(f"  Change in Dz_real          = {dz_fixed - dz_orig:.6f}")
    report(f"  off_diag_mean(Crate_orig)  = {off_diag_mean(Crate_orig):.8f}")
    report(f"  off_diag_mean(Crate_fixed) = {off_diag_mean(Crate_fixed):.8f}")
    report(f"  off_diag_mean(Cpsth)       = {off_diag_mean(Cpsth):.8f}")

    # Mean rate comparison
    erate_diff = np.abs(Erate_orig - Erate_fixed)
    report(f"\n  Erate difference (max abs)  = {erate_diff.max():.8f}")
    report(f"  Erate difference (mean abs) = {erate_diff.mean():.8f}")

    return {
        'fz_U': fz_U, 'fz_C_orig': fz_C_orig, 'fz_C_fixed': fz_C_fixed,
        'dz_orig': dz_orig, 'dz_fixed': dz_fixed,
        'Ctotal': Ctotal, 'Cpsth': Cpsth, 'CnoiseU': CnoiseU,
        'Crate_orig': Crate_orig, 'Crate_fixed': Crate_fixed,
        'be_orig': be_orig,
    }


# ---------------------------------------------------------------------------
# Part 3: Global shuffle null (THE KEY TEST)
# ---------------------------------------------------------------------------

def part3_global_shuffle(SpikeCounts, EyeTraj, T_idx, Ctotal, fz_U, bin_edges):
    """Run global shuffle with original and fixed pipelines."""
    report("\n" + "=" * 70)
    report(f"PART 3: Global shuffle null ({N_SHUFFLES_GLOBAL} iterations)")
    report("=" * 70)

    rng_orig = np.random.default_rng(42)
    rng_fixed = np.random.default_rng(42)  # same seed for matched comparisons

    dz_shuff_orig = []
    dz_shuff_fixed = []
    fz_shuff_orig = []
    fz_shuff_fixed = []

    for i in tqdm(range(N_SHUFFLES_GLOBAL), desc="Global shuffle"):
        # Same permutation for both
        perm = rng_orig.permutation(EyeTraj.shape[0])
        EyeTraj_shuff = EyeTraj[perm]
        _ = rng_fixed.permutation(EyeTraj.shape[0])  # keep RNG in sync

        # --- Original pipeline ---
        Crate_o, _, Ceye_o, bc_o, ce_o, _ = estimate_rate_covariance(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
            Ctotal=Ctotal, intercept_mode='linear'
        )
        CnoiseC_o = 0.5 * ((Ctotal - Crate_o) + (Ctotal - Crate_o).T)
        fz_o = fz_from_cov(CnoiseC_o, use_psd=True)
        fz_shuff_orig.append(fz_o)
        dz_shuff_orig.append(fz_o - fz_U)

        # --- Fixed pipeline ---
        Crate_f, _, Ceye_f, bc_f, ce_f, _ = estimate_rate_covariance_fixed(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
            Ctotal=Ctotal, intercept_mode='linear'
        )
        CnoiseC_f = 0.5 * ((Ctotal - Crate_f) + (Ctotal - Crate_f).T)
        fz_f = fz_from_cov(CnoiseC_f, use_psd=True)
        fz_shuff_fixed.append(fz_f)
        dz_shuff_fixed.append(fz_f - fz_U)

    dz_shuff_orig = np.array(dz_shuff_orig)
    dz_shuff_fixed = np.array(dz_shuff_fixed)

    n = N_SHUFFLES_GLOBAL
    report(f"\n  ORIGINAL pipeline:")
    report(f"    Mean Dz_shuff  = {np.mean(dz_shuff_orig):.6f} +/- {np.std(dz_shuff_orig)/np.sqrt(n):.6f} (SE)")
    report(f"    Std  Dz_shuff  = {np.std(dz_shuff_orig):.6f}")
    report(f"    Range          = [{np.min(dz_shuff_orig):.6f}, {np.max(dz_shuff_orig):.6f}]")

    report(f"\n  FIXED pipeline (Approach A: pair-weighted mean):")
    report(f"    Mean Dz_shuff  = {np.mean(dz_shuff_fixed):.6f} +/- {np.std(dz_shuff_fixed)/np.sqrt(n):.6f} (SE)")
    report(f"    Std  Dz_shuff  = {np.std(dz_shuff_fixed):.6f}")
    report(f"    Range          = [{np.min(dz_shuff_fixed):.6f}, {np.max(dz_shuff_fixed):.6f}]")

    report(f"\n  Shift reduction  = {np.mean(dz_shuff_orig) - np.mean(dz_shuff_fixed):.6f}")
    report(f"  Fixed null centered at 0? {abs(np.mean(dz_shuff_fixed)) < 2 * np.std(dz_shuff_fixed)/np.sqrt(n)}")

    # t-test against zero
    from scipy import stats as sp_stats
    t_orig, p_orig = sp_stats.ttest_1samp(dz_shuff_orig, 0)
    t_fixed, p_fixed = sp_stats.ttest_1samp(dz_shuff_fixed, 0)
    report(f"\n  t-test vs 0 (original):  t={t_orig:.3f}, p={p_orig:.4f}")
    report(f"  t-test vs 0 (fixed):     t={t_fixed:.3f}, p={p_fixed:.4f}")

    return {
        'dz_orig': dz_shuff_orig, 'dz_fixed': dz_shuff_fixed,
        'dz_orig_mean': float(np.mean(dz_shuff_orig)),
        'dz_orig_se': float(np.std(dz_shuff_orig) / np.sqrt(n)),
        'dz_fixed_mean': float(np.mean(dz_shuff_fixed)),
        'dz_fixed_se': float(np.std(dz_shuff_fixed) / np.sqrt(n)),
        'p_orig': float(p_orig),
        'p_fixed': float(p_fixed),
    }


# ---------------------------------------------------------------------------
# Part 4: Within-time-bin shuffle
# ---------------------------------------------------------------------------

def part4_within_time_bin_shuffle(SpikeCounts, EyeTraj, T_idx, Ctotal, fz_U,
                                    bin_edges):
    """Within-time-bin shuffle with both pipelines."""
    report("\n" + "=" * 70)
    report(f"PART 4: Within-time-bin shuffle ({N_SHUFFLES_WITHIN} iterations)")
    report("=" * 70)

    rng = np.random.default_rng(123)

    dz_orig = []
    dz_fixed = []

    for i in tqdm(range(N_SHUFFLES_WITHIN), desc="Within-bin shuffle"):
        EyeTraj_shuff = within_time_bin_shuffle(EyeTraj, T_idx, rng)

        # Original
        Crate_o, _, _, _, _, _ = estimate_rate_covariance(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
            Ctotal=Ctotal, intercept_mode='linear'
        )
        CnoiseC_o = 0.5 * ((Ctotal - Crate_o) + (Ctotal - Crate_o).T)
        fz_o = fz_from_cov(CnoiseC_o, use_psd=True)
        dz_orig.append(fz_o - fz_U)

        # Fixed
        Crate_f, _, _, _, _, _ = estimate_rate_covariance_fixed(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
            Ctotal=Ctotal, intercept_mode='linear'
        )
        CnoiseC_f = 0.5 * ((Ctotal - Crate_f) + (Ctotal - Crate_f).T)
        fz_f = fz_from_cov(CnoiseC_f, use_psd=True)
        dz_fixed.append(fz_f - fz_U)

    dz_orig = np.array(dz_orig)
    dz_fixed = np.array(dz_fixed)
    n = N_SHUFFLES_WITHIN

    report(f"\n  ORIGINAL pipeline:")
    report(f"    Mean Dz_shuff  = {np.mean(dz_orig):.6f} +/- {np.std(dz_orig)/np.sqrt(n):.6f} (SE)")

    report(f"\n  FIXED pipeline:")
    report(f"    Mean Dz_shuff  = {np.mean(dz_fixed):.6f} +/- {np.std(dz_fixed)/np.sqrt(n):.6f} (SE)")

    report(f"\n  Shift reduction  = {np.mean(dz_orig) - np.mean(dz_fixed):.6f}")

    return {
        'dz_orig_mean': float(np.mean(dz_orig)),
        'dz_orig_se': float(np.std(dz_orig) / np.sqrt(n)),
        'dz_fixed_mean': float(np.mean(dz_fixed)),
        'dz_fixed_se': float(np.std(dz_fixed) / np.sqrt(n)),
    }


# ---------------------------------------------------------------------------
# Part 5: Signal preservation check
# ---------------------------------------------------------------------------

def part5_signal_preservation(real_results):
    """Check that the fix preserves the real-data signal."""
    report("\n" + "=" * 70)
    report("PART 5: Signal preservation check")
    report("=" * 70)

    dz_orig = real_results['dz_orig']
    dz_fixed = real_results['dz_fixed']
    change = dz_fixed - dz_orig

    report(f"\n  Dz_real (original) = {dz_orig:.6f}")
    report(f"  Dz_real (fixed)    = {dz_fixed:.6f}")
    report(f"  Change             = {change:.6f}")
    report(f"  |Change| < 0.005?  {'YES' if abs(change) < 0.005 else 'NO'}")

    if abs(change) < 0.005:
        report("  --> Fix preserves the real-data signal (change < 0.005)")
    elif abs(change) < 0.01:
        report("  --> Small change in real signal (0.005 < change < 0.01)")
    else:
        report("  --> WARNING: Fix substantially changes the real signal!")

    return {'dz_orig': dz_orig, 'dz_fixed': dz_fixed, 'change': change}


# ---------------------------------------------------------------------------
# Part 6: Summary table
# ---------------------------------------------------------------------------

def part6_summary(real_results, shuff_global, shuff_within, signal_check):
    """Print the final summary table."""
    report("\n" + "=" * 70)
    report("PART 6: Summary")
    report("=" * 70)

    dz_real_orig = real_results['dz_orig']
    dz_real_fixed = real_results['dz_fixed']
    dz_shuff_orig = shuff_global['dz_orig_mean']
    dz_shuff_fixed = shuff_global['dz_fixed_mean']
    dz_shuff_orig_se = shuff_global['dz_orig_se']
    dz_shuff_fixed_se = shuff_global['dz_fixed_se']

    # Signal-to-null ratio (use absolute values)
    sn_orig = abs(dz_real_orig / dz_shuff_orig) if abs(dz_shuff_orig) > 1e-6 else float('inf')
    sn_fixed = abs(dz_real_fixed / dz_shuff_fixed) if abs(dz_shuff_fixed) > 1e-6 else float('inf')

    report(f"\n  {'':25s} {'Original':>12s} {'Fixed':>12s} {'Change':>12s}")
    report(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    report(f"  {'Real data Dz':25s} {dz_real_orig:>12.6f} {dz_real_fixed:>12.6f} {dz_real_fixed-dz_real_orig:>+12.6f}")
    report(f"  {'Shuffle null Dz':25s} {dz_shuff_orig:>12.6f} {dz_shuff_fixed:>12.6f} {dz_shuff_fixed-dz_shuff_orig:>+12.6f}")
    report(f"  {'Shuffle null SE':25s} {dz_shuff_orig_se:>12.6f} {dz_shuff_fixed_se:>12.6f} {'':>12s}")
    report(f"  {'|Signal/Null|':25s} {sn_orig:>12.1f}x {sn_fixed:>12.1f}x {'':>12s}")

    report(f"\n  Within-time-bin shuffle:")
    report(f"  {'':25s} {'Original':>12s} {'Fixed':>12s} {'Change':>12s}")
    report(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    report(f"  {'Shuffle null Dz':25s} {shuff_within['dz_orig_mean']:>12.6f} {shuff_within['dz_fixed_mean']:>12.6f} {shuff_within['dz_fixed_mean']-shuff_within['dz_orig_mean']:>+12.6f}")

    report(f"\n  Global shuffle t-test vs 0:")
    report(f"    Original: p = {shuff_global['p_orig']:.4f}")
    report(f"    Fixed:    p = {shuff_global['p_fixed']:.4f}")

    if abs(dz_shuff_fixed) < abs(dz_shuff_orig) * 0.5:
        report(f"\n  CONCLUSION: Fix reduces shuffle null bias by "
               f"{(1 - abs(dz_shuff_fixed)/abs(dz_shuff_orig))*100:.0f}%")
    if shuff_global['p_fixed'] > 0.05:
        report(f"  Fixed shuffle null is NOT significantly different from 0 (p={shuff_global['p_fixed']:.3f})")
    else:
        report(f"  Fixed shuffle null is STILL significantly different from 0 (p={shuff_global['p_fixed']:.4f})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(real_results, shuff_global):
    """Generate diagnostic figures."""

    # Figure 1: Shuffle null distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(shuff_global['dz_orig'], bins=15, alpha=0.6, label='Original', color='C0')
    ax.hist(shuff_global['dz_fixed'], bins=15, alpha=0.6, label='Fixed', color='C2')
    ax.axvline(0, color='k', ls='--', lw=1, label='Zero')
    ax.axvline(np.mean(shuff_global['dz_orig']), color='C0', ls='-', lw=2)
    ax.axvline(np.mean(shuff_global['dz_fixed']), color='C2', ls='-', lw=2)
    ax.set_xlabel('Dz (shuffle null)')
    ax.set_ylabel('Count')
    ax.set_title('Global shuffle: Dz distribution')
    ax.legend(fontsize=9)

    ax = axes[1]
    dz_diff = shuff_global['dz_orig'] - shuff_global['dz_fixed']
    ax.hist(dz_diff, bins=15, alpha=0.7, color='C1')
    ax.axvline(np.mean(dz_diff), color='C1', ls='-', lw=2)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Dz_orig - Dz_fixed (per iteration)')
    ax.set_ylabel('Count')
    ax.set_title('Paired difference (orig - fixed)')

    ax = axes[2]
    labels = ['Real\n(orig)', 'Real\n(fixed)', 'Shuff\n(orig)', 'Shuff\n(fixed)']
    means = [
        real_results['dz_orig'], real_results['dz_fixed'],
        shuff_global['dz_orig_mean'], shuff_global['dz_fixed_mean'],
    ]
    ses = [
        0, 0,
        shuff_global['dz_orig_se'], shuff_global['dz_fixed_se'],
    ]
    colors = ['C0', 'C2', 'C0', 'C2']
    bars = ax.bar(labels, means, yerr=[s * 1.96 for s in ses], capsize=5,
                  color=colors, alpha=0.7, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_ylabel('Dz')
    ax.set_title('Summary: Real vs Shuffle')

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'weighting_fix_results.png'), dpi=150)
    plt.close(fig)
    report(f"\n  Figure saved: {os.path.join(SAVE_DIR, 'weighting_fix_results.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report("=" * 70)
    report("WEIGHTING FIX TEST")
    report("Pair-count-weighted mean subtraction for noise correlation bias")
    report("=" * 70)

    device = get_device()
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Compute baseline covariances
    report("\nComputing baseline covariances...")
    ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()
    Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()

    Cpsth, PSTH_mean = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
        seed=42, global_mean=Erate
    )

    # Part 2: Real data comparison
    real_results = part2_real_data(SpikeCounts, EyeTraj, T_idx, Ctotal, Cpsth, Erate)

    # Part 3: Global shuffle (THE KEY TEST)
    shuff_global = part3_global_shuffle(
        SpikeCounts, EyeTraj, T_idx, Ctotal,
        real_results['fz_U'], real_results['be_orig']
    )

    # Part 4: Within-time-bin shuffle
    shuff_within = part4_within_time_bin_shuffle(
        SpikeCounts, EyeTraj, T_idx, Ctotal,
        real_results['fz_U'], real_results['be_orig']
    )

    # Part 5: Signal preservation
    signal_check = part5_signal_preservation(real_results)

    # Part 6: Summary
    part6_summary(real_results, shuff_global, shuff_within, signal_check)

    # Plots
    make_plots(real_results, shuff_global)

    # Save report
    report_path = os.path.join(SAVE_DIR, 'weighting_fix_report.md')
    with open(report_path, 'w') as f:
        f.write("# Weighting Fix Test Report\n\n")
        f.write(f"Session: {SESSION_NAME}\n")
        f.write(f"Global shuffles: {N_SHUFFLES_GLOBAL}\n")
        f.write(f"Within-bin shuffles: {N_SHUFFLES_WITHIN}\n\n")
        f.write("```\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write("```\n")
    report(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
