"""
Hypothesis 6: Shuffle null bias and Crate > Ctotal excess diagnosis.

Tests whether:
  (a) The shuffle null itself is biased (Crate_shuff off-diag > Ctotal off-diag)
  (b) Negative noise correlations are real biology or a finite-sample artifact
  (c) Within-time-bin shuffling (the correct null for distance-binning bias) shows bias

Parts:
  1. Shuffle null characterization (50 global shuffles)
  2. Direct Crate > Ctotal legitimacy test
  3. Finite-sample regression-to-mean test (within-time-bin shuffle)
  4. Compare shuffle strategies: global vs within-time-bin vs real
  5. Quantify real FEM effect vs bias
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
DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"

N_SHUFFLES = 50  # Number of shuffle iterations


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
    """Load Allen_2022-04-13 session data."""
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
    """Extract windows and return tensors."""
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


def compute_standard_covariances(SpikeCounts, EyeTraj, T_idx):
    """Compute Ctotal, Crate, Cpsth, CnoiseU, CnoiseC on real (unshuffled) data."""
    ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()
    Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()

    Crate, _, Ceye, bin_centers, count_e, bin_edges = estimate_rate_covariance(
        SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS,
        Ctotal=Ctotal, intercept_mode='linear'
    )

    Cpsth, PSTH_mean = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
        seed=42, global_mean=Erate
    )

    CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

    return Ctotal, Crate, Cpsth, CnoiseU, CnoiseC, Erate, bin_edges


def compute_metrics(C_noise, label=""):
    """Compute standard noise correlation metrics from a noise covariance matrix."""
    C_psd = project_to_psd(C_noise)
    R = cov_to_corr(C_psd)
    tri = get_upper_triangle(R)
    fz = fisher_z_mean(tri)
    off_diag_mean = np.nanmean(tri)
    return fz, off_diag_mean, tri


def off_diag_cov_mean(C):
    """Mean of upper-triangular off-diagonal elements of a covariance matrix."""
    n = C.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = C[mask]
    return np.nanmean(vals)


def global_shuffle_eye(EyeTraj, rng):
    """Globally permute eye trajectories across all trials."""
    N = EyeTraj.shape[0]
    perm = rng.permutation(N)
    return EyeTraj[perm]


def within_time_bin_shuffle(SpikeCounts, EyeTraj, T_idx, rng):
    """
    Within-time-bin shuffle: for each time bin, randomly permute which
    spike count vector is associated with which eye trajectory.

    This preserves:
    - Marginal spike count distribution within each time bin
    - Marginal eye trajectory distribution within each time bin
    - Temporal structure (which trials belong to which time bin)

    But breaks:
    - Any coupling between spike counts and eye trajectories within a time bin

    This is the correct null hypothesis for testing whether distance-binning
    creates a spurious Crate > Ctotal effect.
    """
    unique_times = torch.unique(T_idx).cpu().numpy()
    # We permute the spike-to-eye assignment within each time bin
    # by shuffling eye trajectory indices within each time bin
    perm_indices = torch.arange(EyeTraj.shape[0], device=EyeTraj.device)

    for t in unique_times:
        mask = (T_idx == t).cpu().numpy()
        idx_t = np.where(mask)[0]
        if len(idx_t) < 2:
            continue
        shuffled = rng.permutation(idx_t)
        perm_indices[idx_t] = torch.tensor(shuffled, device=EyeTraj.device)

    return EyeTraj[perm_indices]


def run_crate_from_eye(SpikeCounts, EyeTraj_input, T_idx, Ctotal, bin_edges):
    """Compute Crate from given spike counts and eye trajectories using fixed bin edges."""
    MM, bin_centers, count_e, _ = compute_conditional_second_moments(
        SpikeCounts, EyeTraj_input, T_idx, n_bins=bin_edges
    )
    Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
    Ceye = MM - Erate[:, None] * Erate[None, :]

    Crate = fit_intercept_linear(Ceye, bin_centers, count_e, eval_at_first_bin=True)

    # Apply same variance cap as in estimate_rate_covariance
    bad_mask = np.diag(Crate) > 0.99 * np.diag(Ctotal)
    Crate[bad_mask, :] = np.nan
    Crate[:, bad_mask] = np.nan

    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)
    return Crate, CnoiseC


# ============================================================================
# Part 1: Shuffle null characterization
# ============================================================================

def part1_shuffle_null(SpikeCounts, EyeTraj, T_idx, Ctotal, bin_edges):
    """Run N_SHUFFLES global shuffles and characterize the null distribution."""
    print("\n" + "=" * 70)
    print("PART 1: Shuffle null characterization (global shuffle)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    results = {
        'Crate_off_diag': [],
        'CnoiseC_off_diag': [],
        'fz_CnoiseC': [],
    }

    for i in tqdm(range(N_SHUFFLES), desc="Global shuffles"):
        EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)
        Crate_s, CnoiseC_s = run_crate_from_eye(
            SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, bin_edges
        )
        results['Crate_off_diag'].append(off_diag_cov_mean(Crate_s))
        results['CnoiseC_off_diag'].append(off_diag_cov_mean(CnoiseC_s))
        fz, _, _ = compute_metrics(CnoiseC_s)
        results['fz_CnoiseC'].append(fz)

    for k in results:
        results[k] = np.array(results[k])

    ctotal_off = off_diag_cov_mean(Ctotal)

    print(f"\n  Ctotal off-diag mean:        {ctotal_off:.6f}")
    print(f"  Crate_shuff off-diag mean:   {np.nanmean(results['Crate_off_diag']):.6f} "
          f"+/- {np.nanstd(results['Crate_off_diag']):.6f}")
    print(f"  CnoiseC_shuff off-diag mean: {np.nanmean(results['CnoiseC_off_diag']):.6f} "
          f"+/- {np.nanstd(results['CnoiseC_off_diag']):.6f}")
    print(f"  Crate_shuff - Ctotal:        {np.nanmean(results['Crate_off_diag']) - ctotal_off:.6f}")
    print(f"  fz(CnoiseC_shuff):           {np.nanmean(results['fz_CnoiseC']):.4f} "
          f"+/- {np.nanstd(results['fz_CnoiseC']):.4f}")

    # Key test: does Crate_shuff exceed Ctotal?
    excess_frac = np.mean(results['Crate_off_diag'] > ctotal_off)
    print(f"\n  Fraction of shuffles where Crate_shuff off-diag > Ctotal off-diag: {excess_frac:.2f}")

    if excess_frac > 0.9:
        print("  --> BIAS DETECTED: Global shuffle null also shows Crate > Ctotal")
    elif excess_frac < 0.1:
        print("  --> NO BIAS: Under null, Crate_shuff ~ Ctotal (excess is real signal)")
    else:
        print("  --> AMBIGUOUS: Mixed results")

    return results


# ============================================================================
# Part 2: Direct Crate > Ctotal legitimacy test
# ============================================================================

def part2_direct_comparison(Ctotal, Crate_real, results_global):
    """Compare real vs shuffled Crate excess over Ctotal."""
    print("\n" + "=" * 70)
    print("PART 2: Direct test of Crate > Ctotal legitimacy")
    print("=" * 70)

    ctotal_off = off_diag_cov_mean(Ctotal)
    crate_real_off = off_diag_cov_mean(Crate_real)
    crate_shuff_off = np.nanmean(results_global['Crate_off_diag'])

    real_excess = crate_real_off - ctotal_off
    shuff_excess = crate_shuff_off - ctotal_off
    fem_driven = crate_real_off - crate_shuff_off

    print(f"\n  off_diag_mean(Ctotal):       {ctotal_off:.6f}")
    print(f"  off_diag_mean(Crate_real):   {crate_real_off:.6f}")
    print(f"  off_diag_mean(Crate_shuff):  {crate_shuff_off:.6f}")
    print(f"\n  TOTAL excess (Crate_real - Ctotal):    {real_excess:.6f}")
    print(f"  SHUFFLE excess (Crate_shuff - Ctotal): {shuff_excess:.6f}")
    print(f"  FEM-driven excess (Crate_real - Crate_shuff): {fem_driven:.6f}")
    print(f"\n  Decomposition of total excess:")
    if abs(real_excess) > 1e-10:
        print(f"    Bias component:      {shuff_excess:.6f} ({100*shuff_excess/real_excess:.1f}%)")
        print(f"    Real FEM component:  {fem_driven:.6f} ({100*fem_driven/real_excess:.1f}%)")
    else:
        print(f"    No excess to decompose")

    return {
        'ctotal_off': ctotal_off,
        'crate_real_off': crate_real_off,
        'crate_shuff_off': crate_shuff_off,
        'real_excess': real_excess,
        'shuff_excess': shuff_excess,
        'fem_driven': fem_driven,
    }


# ============================================================================
# Part 3: Within-time-bin shuffle (regression-to-mean test)
# ============================================================================

def part3_within_time_bin_shuffle(SpikeCounts, EyeTraj, T_idx, Ctotal, bin_edges):
    """
    The critical test: within-time-bin shuffle breaks spike-eye coupling
    but preserves temporal structure and marginal distributions.

    If this null shows Crate off-diag > Ctotal, it's a finite-sample
    regression-to-mean bias from the distance binning procedure.
    """
    print("\n" + "=" * 70)
    print("PART 3: Within-time-bin shuffle (regression-to-mean test)")
    print("=" * 70)

    rng = np.random.default_rng(123)

    results = {
        'Crate_off_diag': [],
        'CnoiseC_off_diag': [],
        'fz_CnoiseC': [],
    }

    for i in tqdm(range(N_SHUFFLES), desc="Within-time-bin shuffles"):
        EyeTraj_perm = within_time_bin_shuffle(SpikeCounts, EyeTraj, T_idx, rng)
        Crate_p, CnoiseC_p = run_crate_from_eye(
            SpikeCounts, EyeTraj_perm, T_idx, Ctotal, bin_edges
        )
        results['Crate_off_diag'].append(off_diag_cov_mean(Crate_p))
        results['CnoiseC_off_diag'].append(off_diag_cov_mean(CnoiseC_p))
        fz, _, _ = compute_metrics(CnoiseC_p)
        results['fz_CnoiseC'].append(fz)

    for k in results:
        results[k] = np.array(results[k])

    ctotal_off = off_diag_cov_mean(Ctotal)

    print(f"\n  Ctotal off-diag mean:            {ctotal_off:.6f}")
    print(f"  Crate_within_shuff off-diag:     {np.nanmean(results['Crate_off_diag']):.6f} "
          f"+/- {np.nanstd(results['Crate_off_diag']):.6f}")
    print(f"  CnoiseC_within_shuff off-diag:   {np.nanmean(results['CnoiseC_off_diag']):.6f} "
          f"+/- {np.nanstd(results['CnoiseC_off_diag']):.6f}")
    print(f"  Within-shuff excess:             {np.nanmean(results['Crate_off_diag']) - ctotal_off:.6f}")
    print(f"  fz(CnoiseC_within_shuff):        {np.nanmean(results['fz_CnoiseC']):.4f} "
          f"+/- {np.nanstd(results['fz_CnoiseC']):.4f}")

    excess_frac = np.mean(results['Crate_off_diag'] > ctotal_off)
    print(f"\n  Fraction with Crate_within > Ctotal: {excess_frac:.2f}")

    if excess_frac > 0.9:
        print("  --> REGRESSION-TO-MEAN BIAS CONFIRMED: Distance binning itself creates excess")
    elif excess_frac < 0.1:
        print("  --> NO DISTANCE-BINNING BIAS: Crate excess requires real spike-eye coupling")
    else:
        print("  --> PARTIAL BIAS from distance binning")

    return results


# ============================================================================
# Part 4: Compare shuffle strategies
# ============================================================================

def part4_compare_strategies(SpikeCounts, EyeTraj, T_idx, Ctotal, Crate_real,
                             CnoiseC_real, CnoiseU_real, results_global, results_within,
                             fz_real_C, fz_real_U):
    """Compare all three conditions: real, global shuffle, within-time-bin shuffle."""
    print("\n" + "=" * 70)
    print("PART 4: Compare shuffle strategies")
    print("=" * 70)

    ctotal_off = off_diag_cov_mean(Ctotal)
    crate_real_off = off_diag_cov_mean(Crate_real)

    rows = [
        ("Real data", crate_real_off, off_diag_cov_mean(CnoiseC_real), fz_real_C),
        ("Global shuffle",
         np.nanmean(results_global['Crate_off_diag']),
         np.nanmean(results_global['CnoiseC_off_diag']),
         np.nanmean(results_global['fz_CnoiseC'])),
        ("Within-time shuffle",
         np.nanmean(results_within['Crate_off_diag']),
         np.nanmean(results_within['CnoiseC_off_diag']),
         np.nanmean(results_within['fz_CnoiseC'])),
    ]

    print(f"\n  {'Condition':<22s} {'Crate_off':<12s} {'CnoiseC_off':<12s} {'fz(CnoiseC)':<12s} {'Crate-Ctotal':<12s}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for name, cr_off, cn_off, fz_cn in rows:
        excess = cr_off - ctotal_off
        print(f"  {name:<22s} {cr_off:>11.6f} {cn_off:>11.6f} {fz_cn:>11.4f} {excess:>11.6f}")

    print(f"\n  Uncorrected fz(CnoiseU): {fz_real_U:.4f}")
    print(f"  Ctotal off-diag:         {ctotal_off:.6f}")

    # Compute Dz for each condition
    dz_real = fz_real_C - fz_real_U
    dz_global = np.nanmean(results_global['fz_CnoiseC']) - fz_real_U
    dz_within = np.nanmean(results_within['fz_CnoiseC']) - fz_real_U

    print(f"\n  Dz (FEM correction effect = fz_C - fz_U):")
    print(f"    Real data:           {dz_real:+.4f}")
    print(f"    Global shuffle:      {dz_global:+.4f}")
    print(f"    Within-time shuffle: {dz_within:+.4f}")

    return rows


# ============================================================================
# Part 5: Quantify real FEM effect vs bias
# ============================================================================

def part5_quantify(Ctotal, Crate_real, CnoiseC_real, CnoiseU_real,
                   results_global, results_within, fz_real_C, fz_real_U):
    """Final quantification: how much of the negative Dz is bias vs real signal."""
    print("\n" + "=" * 70)
    print("PART 5: Quantify real FEM effect vs bias")
    print("=" * 70)

    ctotal_off = off_diag_cov_mean(Ctotal)
    crate_real_off = off_diag_cov_mean(Crate_real)
    crate_within_off = np.nanmean(results_within['Crate_off_diag'])
    crate_global_off = np.nanmean(results_global['Crate_off_diag'])

    # In covariance space
    total_excess_cov = crate_real_off - ctotal_off
    within_bias_cov = crate_within_off - ctotal_off
    global_bias_cov = crate_global_off - ctotal_off
    real_signal_cov = crate_real_off - crate_within_off

    print(f"\n  --- Covariance-space decomposition ---")
    print(f"  Total Crate excess over Ctotal:     {total_excess_cov:.6f}")
    print(f"  Within-bin shuffle bias:             {within_bias_cov:.6f}")
    print(f"  Global shuffle bias:                 {global_bias_cov:.6f}")
    print(f"  Real FEM signal (real - within):     {real_signal_cov:.6f}")
    if abs(total_excess_cov) > 1e-10:
        print(f"\n  Bias fraction (within-bin): {100*within_bias_cov/total_excess_cov:.1f}%")
        print(f"  Real signal fraction:       {100*real_signal_cov/total_excess_cov:.1f}%")

    # In Fisher-z space
    dz_real = fz_real_C - fz_real_U
    dz_within = np.nanmean(results_within['fz_CnoiseC']) - fz_real_U
    dz_global = np.nanmean(results_global['fz_CnoiseC']) - fz_real_U

    print(f"\n  --- Fisher-z space decomposition ---")
    print(f"  Dz real (fz_C - fz_U):              {dz_real:+.4f}")
    print(f"  Dz within-bin shuffle null:          {dz_within:+.4f}")
    print(f"  Dz global shuffle null:              {dz_global:+.4f}")
    print(f"  Dz attributable to bias (within):    {dz_within:+.4f}")
    print(f"  Dz attributable to real FEM signal:  {dz_real - dz_within:+.4f}")
    if abs(dz_real) > 1e-6:
        print(f"\n  Bias fraction of Dz (within-bin):  {100*dz_within/dz_real:.1f}%")
        print(f"  Real signal fraction of Dz:        {100*(dz_real - dz_within)/dz_real:.1f}%")

    # Interpretation — use Fisher-z space which is the meaningful metric
    print(f"\n  --- INTERPRETATION ---")
    dz_bias_frac = abs(summary['dz_within']) / abs(summary['dz_real']) if abs(summary['dz_real']) > 1e-6 else 0
    if within_bias_cov <= 0:
        print("  Shuffle null shows Crate BELOW Ctotal (no positive bias).")
        print("  The Crate > Ctotal excess is driven entirely by real spike-eye coupling.")
    if dz_bias_frac < 0.10:
        print(f"  In Fisher-z space, shuffle null accounts for only {100*dz_bias_frac:.1f}% of Dz.")
        print("  The negative noise correlations are REAL BIOLOGY.")
        print("  Likely mechanism: lateral inhibition / competitive interactions")
        print("  that produce anti-correlated noise, normally masked by positive")
        print("  FEM-driven correlations.")
    elif dz_bias_frac > 0.90:
        print(f"  In Fisher-z space, shuffle null accounts for {100*dz_bias_frac:.1f}% of Dz.")
        print("  The negative Dz is mostly a METHODOLOGICAL ARTIFACT.")
    else:
        print(f"  In Fisher-z space, shuffle null accounts for {100*dz_bias_frac:.1f}% of Dz.")
        print("  Partial real signal, partial methodological bias.")

    return {
        'total_excess_cov': total_excess_cov,
        'within_bias_cov': within_bias_cov,
        'real_signal_cov': real_signal_cov,
        'dz_real': dz_real,
        'dz_within': dz_within,
        'dz_global': dz_global,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results_global, results_within, Ctotal, Crate_real,
                 fz_real_C, fz_real_U, summary):
    """Generate summary figures."""
    ctotal_off = off_diag_cov_mean(Ctotal)
    crate_real_off = off_diag_cov_mean(Crate_real)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Distribution of Crate off-diag under both shuffles
    ax = axes[0, 0]
    ax.hist(results_global['Crate_off_diag'], bins=20, alpha=0.5,
            label='Global shuffle', color='C0', density=True)
    ax.hist(results_within['Crate_off_diag'], bins=20, alpha=0.5,
            label='Within-time shuffle', color='C1', density=True)
    ax.axvline(ctotal_off, color='k', ls='--', lw=2, label=f'Ctotal off-diag = {ctotal_off:.5f}')
    ax.axvline(crate_real_off, color='red', ls='-', lw=2, label=f'Crate_real off-diag = {crate_real_off:.5f}')
    ax.set_xlabel('Crate off-diagonal mean')
    ax.set_ylabel('Density')
    ax.set_title('A. Shuffle null distributions of Crate off-diag')
    ax.legend(fontsize=8)

    # Panel B: Distribution of fz(CnoiseC) under both shuffles
    ax = axes[0, 1]
    ax.hist(results_global['fz_CnoiseC'], bins=20, alpha=0.5,
            label='Global shuffle', color='C0', density=True)
    ax.hist(results_within['fz_CnoiseC'], bins=20, alpha=0.5,
            label='Within-time shuffle', color='C1', density=True)
    ax.axvline(fz_real_C, color='red', ls='-', lw=2, label=f'Real fz(CnoiseC) = {fz_real_C:.4f}')
    ax.axvline(fz_real_U, color='green', ls='--', lw=2, label=f'fz(CnoiseU) = {fz_real_U:.4f}')
    ax.set_xlabel('Fisher-z mean of noise correlations')
    ax.set_ylabel('Density')
    ax.set_title('B. Shuffle null distributions of fz(CnoiseC)')
    ax.legend(fontsize=8)

    # Panel C: Excess decomposition bar chart
    ax = axes[1, 0]
    labels = ['Total\nexcess', 'Within-bin\nbias', 'Global\nbias', 'Real FEM\nsignal']
    values = [summary['total_excess_cov'], summary['within_bias_cov'],
              summary['dz_global'] * 0,  # placeholder - use cov values
              summary['real_signal_cov']]
    # Actually recompute global bias in cov space
    global_bias_cov = np.nanmean(results_global['Crate_off_diag']) - ctotal_off
    values = [summary['total_excess_cov'], summary['within_bias_cov'],
              global_bias_cov, summary['real_signal_cov']]
    colors = ['gray', 'C1', 'C0', 'red']
    ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='k')
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.set_ylabel('Off-diagonal covariance')
    ax.set_title('C. Crate excess decomposition (covariance space)')

    # Panel D: Dz decomposition
    ax = axes[1, 1]
    labels_dz = ['Dz real', 'Dz within-bin\nnull', 'Dz global\nnull', 'Dz real\nsignal']
    values_dz = [summary['dz_real'], summary['dz_within'],
                 summary['dz_global'], summary['dz_real'] - summary['dz_within']]
    colors_dz = ['gray', 'C1', 'C0', 'red']
    ax.bar(labels_dz, values_dz, color=colors_dz, alpha=0.7, edgecolor='k')
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.set_ylabel('Delta-z')
    ax.set_title('D. Dz decomposition (Fisher-z space)')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "h6_shuffle_and_crate_excess.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    # Additional figure: CnoiseC off-diag distributions
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes2[0]
    ax.hist(results_global['CnoiseC_off_diag'], bins=20, alpha=0.5,
            label='Global shuffle', color='C0', density=True)
    ax.hist(results_within['CnoiseC_off_diag'], bins=20, alpha=0.5,
            label='Within-time shuffle', color='C1', density=True)
    ax.axvline(off_diag_cov_mean(Ctotal - Crate_real), color='red', ls='-', lw=2,
               label='Real CnoiseC off-diag')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('CnoiseC off-diagonal mean')
    ax.set_ylabel('Density')
    ax.set_title('CnoiseC off-diag under shuffle nulls')
    ax.legend(fontsize=8)

    ax = axes2[1]
    # Scatter: Crate off-diag vs fz(CnoiseC) for both shuffles
    ax.scatter(results_global['Crate_off_diag'], results_global['fz_CnoiseC'],
               alpha=0.4, s=20, label='Global shuffle', color='C0')
    ax.scatter(results_within['Crate_off_diag'], results_within['fz_CnoiseC'],
               alpha=0.4, s=20, label='Within-time shuffle', color='C1')
    ax.scatter([crate_real_off], [fz_real_C], s=100, color='red', zorder=5,
               marker='*', label='Real data')
    ax.set_xlabel('Crate off-diagonal mean')
    ax.set_ylabel('fz(CnoiseC)')
    ax.set_title('Crate off-diag vs fz(CnoiseC)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig2_path = os.path.join(SAVE_DIR, "h6_shuffle_scatter.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig2_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("HYPOTHESIS 6: Shuffle null bias & Crate > Ctotal excess")
    print("=" * 70)

    device = get_device()

    # Load data
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Compute real covariances
    print("\nComputing real (unshuffled) covariances...")
    Ctotal, Crate_real, Cpsth, CnoiseU_real, CnoiseC_real, Erate, bin_edges = \
        compute_standard_covariances(SpikeCounts, EyeTraj, T_idx)

    # Compute real metrics
    fz_real_C, _, _ = compute_metrics(CnoiseC_real, "CnoiseC")
    fz_real_U, _, _ = compute_metrics(CnoiseU_real, "CnoiseU")
    dz_real = fz_real_C - fz_real_U

    print(f"\n  Real data baseline:")
    print(f"    fz(CnoiseU) = {fz_real_U:.4f}")
    print(f"    fz(CnoiseC) = {fz_real_C:.4f}")
    print(f"    Dz = {dz_real:+.4f}")
    print(f"    Ctotal off-diag = {off_diag_cov_mean(Ctotal):.6f}")
    print(f"    Crate off-diag  = {off_diag_cov_mean(Crate_real):.6f}")
    print(f"    CnoiseC off-diag = {off_diag_cov_mean(CnoiseC_real):.6f}")

    # Part 1: Global shuffle null
    results_global = part1_shuffle_null(SpikeCounts, EyeTraj, T_idx, Ctotal, bin_edges)

    # Part 2: Direct comparison
    part2_results = part2_direct_comparison(Ctotal, Crate_real, results_global)

    # Part 3: Within-time-bin shuffle
    results_within = part3_within_time_bin_shuffle(SpikeCounts, EyeTraj, T_idx, Ctotal, bin_edges)

    # Part 4: Compare strategies
    part4_compare_strategies(SpikeCounts, EyeTraj, T_idx, Ctotal, Crate_real,
                             CnoiseC_real, CnoiseU_real, results_global, results_within,
                             fz_real_C, fz_real_U)

    # Part 5: Quantify
    summary = part5_quantify(Ctotal, Crate_real, CnoiseC_real, CnoiseU_real,
                             results_global, results_within, fz_real_C, fz_real_U)

    # Plot
    plot_results(results_global, results_within, Ctotal, Crate_real,
                 fz_real_C, fz_real_U, summary)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Dz (real):                {summary['dz_real']:+.4f}")
    print(f"  Dz (within-bin null):     {summary['dz_within']:+.4f}")
    print(f"  Dz (global null):         {summary['dz_global']:+.4f}")
    print(f"  Dz from real FEM signal:  {summary['dz_real'] - summary['dz_within']:+.4f}")
    if abs(summary['dz_real']) > 1e-6:
        bias_pct = 100 * summary['dz_within'] / summary['dz_real']
        print(f"  Bias accounts for:        {bias_pct:.1f}% of total Dz")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
