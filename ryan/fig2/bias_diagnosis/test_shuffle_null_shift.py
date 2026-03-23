"""
Systematic diagnosis of the shuffle null shift (Dz_shuff ~ -0.006 instead of 0).

Tests six hypotheses:
  A: Reused bin edges from real data bias the shuffled distance distribution
  B: Global shuffle changes per-time-bin eye statistics
  C: Estimator asymmetry (trajectory-matching vs split-half)
  D: PSD projection differential
  E: Finite-sample noise amplification in cov-to-corr
  F: Interaction effects — find minimal fix
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
N_SHUFFLES = 50


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


def off_diag_mean(C):
    """Mean of upper-triangular off-diagonal elements."""
    n = C.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = C[mask]
    return np.nanmean(vals)


def fz_from_cov(C_noise, use_psd=True):
    """Compute fisher_z_mean of noise correlations from noise covariance."""
    if use_psd:
        C_psd = project_to_psd(C_noise)
    else:
        C_psd = C_noise.copy()
    R = cov_to_corr(C_psd)
    tri = get_upper_triangle(R)
    return fisher_z_mean(tri)


def count_negative_eigenvalues(C):
    """Count negative eigenvalues of a symmetric matrix."""
    C_clean = np.nan_to_num(C, nan=0.0)
    C_sym = 0.5 * (C_clean + C_clean.T)
    w = np.linalg.eigvalsh(C_sym)
    return int(np.sum(w < -1e-12)), w


def global_shuffle_eye(EyeTraj, rng):
    """Globally permute eye trajectories across all trials."""
    N = EyeTraj.shape[0]
    perm = rng.permutation(N)
    return EyeTraj[perm]


def within_time_bin_shuffle(EyeTraj, T_idx, rng):
    """Permute eye trajectories only within each time bin."""
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


def cyclic_shift_per_time_bin(EyeTraj, T_idx, rng):
    """Apply a random cyclic shift to eye trajectories within each time bin."""
    unique_times = torch.unique(T_idx).cpu().numpy()
    perm_indices = torch.arange(EyeTraj.shape[0], device=EyeTraj.device)

    for t in unique_times:
        mask = (T_idx == t).cpu().numpy()
        idx_t = np.where(mask)[0]
        if len(idx_t) < 2:
            continue
        shift = rng.integers(1, len(idx_t))
        shifted = np.roll(idx_t, shift)
        perm_indices[idx_t] = torch.tensor(shifted, device=EyeTraj.device)

    return EyeTraj[perm_indices]


def compute_crate_from_eye(SpikeCounts, EyeTraj_input, T_idx, Ctotal, n_bins_arg):
    """Compute Crate from given eye trajectories.

    n_bins_arg: int (recompute edges) or array (reuse edges).
    Returns Crate, CnoiseC, bin_edges_used.
    """
    MM, bin_centers, count_e, bin_edges_out = compute_conditional_second_moments(
        SpikeCounts, EyeTraj_input, T_idx, n_bins=n_bins_arg
    )
    Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
    Ceye = MM - Erate[:, None] * Erate[None, :]

    Crate = fit_intercept_linear(Ceye, bin_centers, count_e, eval_at_first_bin=True)

    # Variance cap
    bad_mask = np.diag(Crate) > 0.99 * np.diag(Ctotal)
    Crate[bad_mask, :] = np.nan
    Crate[:, bad_mask] = np.nan

    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)
    return Crate, CnoiseC, Ceye, bin_centers, count_e, bin_edges_out


def compute_crate_mean_bins(Ceye, count_e, Ctotal):
    """Compute Crate as the weighted mean across ALL distance bins (flat curve assumption).

    This is the ML estimator when the curve is truly flat (no eye-distance dependence).
    """
    valid = count_e > 0
    w = count_e[valid].astype(np.float64)
    w /= w.sum()

    # Weighted mean across bins
    Crate_mean = np.zeros_like(Ceye[0])
    for k, vk in enumerate(np.where(valid)[0]):
        Crate_mean += w[k] * Ceye[vk]

    # Variance cap
    bad_mask = np.diag(Crate_mean) > 0.99 * np.diag(Ctotal)
    Crate_mean[bad_mask, :] = np.nan
    Crate_mean[:, bad_mask] = np.nan

    CnoiseC_mean = 0.5 * ((Ctotal - Crate_mean) + (Ctotal - Crate_mean).T)
    return Crate_mean, CnoiseC_mean


# ============================================================================
# EXPERIMENT 1: Bin Edge Effect (Hypothesis A)
# ============================================================================

def experiment_1_bin_edges(SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU,
                           bin_edges_real, fz_U):
    """Compare reused vs fresh bin edges under the shuffle null."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Bin Edge Effect (Hypothesis A)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    results_reused = []
    results_fresh = []

    for i in tqdm(range(N_SHUFFLES), desc="Exp1: bin edge comparison"):
        EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)

        # Reused bin edges (current approach)
        _, CnoiseC_reused, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, bin_edges_real
        )
        fz_reused = fz_from_cov(CnoiseC_reused, use_psd=True)
        results_reused.append(fz_reused)

        # Fresh bin edges (recomputed from shuffled distances)
        _, CnoiseC_fresh, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, N_BINS
        )
        fz_fresh = fz_from_cov(CnoiseC_fresh, use_psd=True)
        results_fresh.append(fz_fresh)

    results_reused = np.array(results_reused)
    results_fresh = np.array(results_fresh)

    dz_reused = results_reused - fz_U
    dz_fresh = results_fresh - fz_U

    print(f"\n  fz(CnoiseU) [fixed]:          {fz_U:.6f}")
    print(f"  fz(CnoiseC_shuff) reused:     {np.mean(results_reused):.6f} +/- {np.std(results_reused):.6f}")
    print(f"  fz(CnoiseC_shuff) fresh:      {np.mean(results_fresh):.6f} +/- {np.std(results_fresh):.6f}")
    print(f"  Dz_shuff (reused edges):      {np.mean(dz_reused):.6f} +/- {np.std(dz_reused)/np.sqrt(N_SHUFFLES):.6f}")
    print(f"  Dz_shuff (fresh edges):       {np.mean(dz_fresh):.6f} +/- {np.std(dz_fresh)/np.sqrt(N_SHUFFLES):.6f}")
    print(f"  Difference (fresh - reused):  {np.mean(dz_fresh) - np.mean(dz_reused):.6f}")

    if abs(np.mean(dz_fresh) - np.mean(dz_reused)) < 0.001:
        print("  --> Bin edges have NEGLIGIBLE effect on the shift")
    else:
        print("  --> Bin edges CONTRIBUTE to the shift")

    return {
        'dz_reused_mean': float(np.mean(dz_reused)),
        'dz_reused_se': float(np.std(dz_reused) / np.sqrt(N_SHUFFLES)),
        'dz_fresh_mean': float(np.mean(dz_fresh)),
        'dz_fresh_se': float(np.std(dz_fresh) / np.sqrt(N_SHUFFLES)),
    }


# ============================================================================
# EXPERIMENT 2: Shuffle Strategy Comparison (Hypothesis B)
# ============================================================================

def experiment_2_shuffle_strategies(SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU,
                                     bin_edges_real, fz_U):
    """Compare global, within-time-bin, and cyclic-shift shuffles."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Shuffle Strategy Comparison (Hypothesis B)")
    print("=" * 70)

    rng_global = np.random.default_rng(42)
    rng_within = np.random.default_rng(42)
    rng_cyclic = np.random.default_rng(42)

    fz_global = []
    fz_within = []
    fz_cyclic = []

    for i in tqdm(range(N_SHUFFLES), desc="Exp2: shuffle strategies"):
        # Global shuffle
        EyeTraj_g = global_shuffle_eye(EyeTraj, rng_global)
        _, CnoiseC_g, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_g, T_idx, Ctotal, bin_edges_real
        )
        fz_global.append(fz_from_cov(CnoiseC_g, use_psd=True))

        # Within-time-bin shuffle
        EyeTraj_w = within_time_bin_shuffle(EyeTraj, T_idx, rng_within)
        _, CnoiseC_w, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_w, T_idx, Ctotal, bin_edges_real
        )
        fz_within.append(fz_from_cov(CnoiseC_w, use_psd=True))

        # Cyclic shift per time bin
        EyeTraj_c = cyclic_shift_per_time_bin(EyeTraj, T_idx, rng_cyclic)
        _, CnoiseC_c, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_c, T_idx, Ctotal, bin_edges_real
        )
        fz_cyclic.append(fz_from_cov(CnoiseC_c, use_psd=True))

    fz_global = np.array(fz_global)
    fz_within = np.array(fz_within)
    fz_cyclic = np.array(fz_cyclic)

    dz_global = fz_global - fz_U
    dz_within = fz_within - fz_U
    dz_cyclic = fz_cyclic - fz_U

    print(f"\n  fz(CnoiseU) [fixed]:          {fz_U:.6f}")
    print(f"\n  {'Strategy':<22s}  {'Dz_mean':>10s}  {'Dz_SE':>10s}  {'fz_mean':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    for name, dz_arr, fz_arr in [
        ("Global shuffle", dz_global, fz_global),
        ("Within-time-bin", dz_within, fz_within),
        ("Cyclic shift", dz_cyclic, fz_cyclic),
    ]:
        print(f"  {name:<22s}  {np.mean(dz_arr):>10.6f}  "
              f"{np.std(dz_arr)/np.sqrt(N_SHUFFLES):>10.6f}  {np.mean(fz_arr):>10.6f}")

    print(f"\n  Global - Within:   {np.mean(dz_global) - np.mean(dz_within):.6f}")
    print(f"  Global - Cyclic:   {np.mean(dz_global) - np.mean(dz_cyclic):.6f}")

    if abs(np.mean(dz_global) - np.mean(dz_within)) < 0.001:
        print("  --> Shuffle strategy has NEGLIGIBLE effect")
    else:
        print("  --> Shuffle strategy MATTERS")

    return {
        'dz_global_mean': float(np.mean(dz_global)),
        'dz_global_se': float(np.std(dz_global) / np.sqrt(N_SHUFFLES)),
        'dz_within_mean': float(np.mean(dz_within)),
        'dz_within_se': float(np.std(dz_within) / np.sqrt(N_SHUFFLES)),
        'dz_cyclic_mean': float(np.mean(dz_cyclic)),
        'dz_cyclic_se': float(np.std(dz_cyclic) / np.sqrt(N_SHUFFLES)),
    }


# ============================================================================
# EXPERIMENT 3: Estimator Comparison (Hypothesis C)
# ============================================================================

def experiment_3_estimator_comparison(SpikeCounts, EyeTraj, T_idx, Ctotal, Cpsth,
                                       bin_edges_real, Erate):
    """Compare off-diag of Crate_shuff (intercept) vs Crate_mean_shuff vs Cpsth."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Estimator Comparison (Hypothesis C)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    crate_intercept_offdiag = []
    crate_mean_offdiag = []
    cpsth_offdiag = off_diag_mean(Cpsth)

    for i in tqdm(range(N_SHUFFLES), desc="Exp3: estimator comparison"):
        EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)

        # Intercept fit (standard approach)
        Crate_int, _, Ceye, bin_centers, count_e, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, bin_edges_real
        )
        crate_intercept_offdiag.append(off_diag_mean(Crate_int))

        # Weighted mean across bins (flat-curve ML estimator)
        Crate_mean, _ = compute_crate_mean_bins(Ceye, count_e, Ctotal)
        crate_mean_offdiag.append(off_diag_mean(Crate_mean))

    crate_intercept_offdiag = np.array(crate_intercept_offdiag)
    crate_mean_offdiag = np.array(crate_mean_offdiag)
    ctotal_offdiag = off_diag_mean(Ctotal)

    print(f"\n  off_diag_mean(Ctotal):                 {ctotal_offdiag:.8f}")
    print(f"  off_diag_mean(Cpsth):                  {cpsth_offdiag:.8f}")
    print(f"  off_diag_mean(Crate_intercept_shuff):  {np.mean(crate_intercept_offdiag):.8f} "
          f"+/- {np.std(crate_intercept_offdiag):.8f}")
    print(f"  off_diag_mean(Crate_mean_shuff):       {np.mean(crate_mean_offdiag):.8f} "
          f"+/- {np.std(crate_mean_offdiag):.8f}")

    print(f"\n  Differences from Cpsth:")
    print(f"    Crate_intercept_shuff - Cpsth:  {np.mean(crate_intercept_offdiag) - cpsth_offdiag:.8f}")
    print(f"    Crate_mean_shuff - Cpsth:       {np.mean(crate_mean_offdiag) - cpsth_offdiag:.8f}")
    print(f"    Ctotal - Cpsth:                 {ctotal_offdiag - cpsth_offdiag:.8f}")

    print(f"\n  Differences from Ctotal:")
    print(f"    Crate_intercept_shuff - Ctotal: {np.mean(crate_intercept_offdiag) - ctotal_offdiag:.8f}")
    print(f"    Crate_mean_shuff - Ctotal:      {np.mean(crate_mean_offdiag) - ctotal_offdiag:.8f}")
    print(f"    Cpsth - Ctotal:                 {cpsth_offdiag - ctotal_offdiag:.8f}")

    # The key question: is Crate_shuff > Cpsth systematically?
    frac_intercept_above_cpsth = np.mean(crate_intercept_offdiag > cpsth_offdiag)
    frac_mean_above_cpsth = np.mean(crate_mean_offdiag > cpsth_offdiag)

    print(f"\n  Fraction Crate_intercept > Cpsth:  {frac_intercept_above_cpsth:.2f}")
    print(f"  Fraction Crate_mean > Cpsth:       {frac_mean_above_cpsth:.2f}")

    if frac_intercept_above_cpsth > 0.9 and frac_mean_above_cpsth < 0.6:
        print("  --> INTERCEPT FIT biases Crate upward; mean-bin estimator does not")
    elif frac_intercept_above_cpsth > 0.9 and frac_mean_above_cpsth > 0.9:
        print("  --> BOTH estimators show Crate > Cpsth (trajectory-matching vs split-half asymmetry)")
    else:
        print("  --> Estimator comparison is inconclusive")

    return {
        'cpsth_offdiag': float(cpsth_offdiag),
        'ctotal_offdiag': float(ctotal_offdiag),
        'crate_intercept_mean': float(np.mean(crate_intercept_offdiag)),
        'crate_intercept_se': float(np.std(crate_intercept_offdiag) / np.sqrt(N_SHUFFLES)),
        'crate_mean_mean': float(np.mean(crate_mean_offdiag)),
        'crate_mean_se': float(np.std(crate_mean_offdiag) / np.sqrt(N_SHUFFLES)),
        'frac_intercept_above_cpsth': float(frac_intercept_above_cpsth),
        'frac_mean_above_cpsth': float(frac_mean_above_cpsth),
    }


# ============================================================================
# EXPERIMENT 4: PSD Projection Effect (Hypothesis D)
# ============================================================================

def experiment_4_psd_projection(SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU,
                                 bin_edges_real, fz_U):
    """Compare Dz with and without PSD projection, count negative eigenvalues."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: PSD Projection Effect (Hypothesis D)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # First compute fz_U with and without PSD
    fz_U_psd = fz_from_cov(CnoiseU, use_psd=True)
    fz_U_nopsd = fz_from_cov(CnoiseU, use_psd=False)

    dz_with_psd = []
    dz_without_psd = []
    neg_eig_CnoiseC = []
    neg_eig_CnoiseU_count, _ = count_negative_eigenvalues(CnoiseU)

    for i in tqdm(range(N_SHUFFLES), desc="Exp4: PSD projection"):
        EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)
        _, CnoiseC_shuff, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, bin_edges_real
        )

        # With PSD on both
        fz_C_psd = fz_from_cov(CnoiseC_shuff, use_psd=True)
        dz_with_psd.append(fz_C_psd - fz_U_psd)

        # Without PSD on either
        fz_C_nopsd = fz_from_cov(CnoiseC_shuff, use_psd=False)
        dz_without_psd.append(fz_C_nopsd - fz_U_nopsd)

        # Count negative eigenvalues
        n_neg, _ = count_negative_eigenvalues(CnoiseC_shuff)
        neg_eig_CnoiseC.append(n_neg)

    dz_with_psd = np.array(dz_with_psd)
    dz_without_psd = np.array(dz_without_psd)
    neg_eig_CnoiseC = np.array(neg_eig_CnoiseC)

    print(f"\n  fz(CnoiseU) with PSD:    {fz_U_psd:.6f}")
    print(f"  fz(CnoiseU) without PSD: {fz_U_nopsd:.6f}")
    print(f"  PSD effect on CnoiseU:   {fz_U_psd - fz_U_nopsd:.6f}")
    print(f"\n  Dz_shuff WITH PSD:       {np.mean(dz_with_psd):.6f} +/- {np.std(dz_with_psd)/np.sqrt(N_SHUFFLES):.6f}")
    print(f"  Dz_shuff WITHOUT PSD:    {np.mean(dz_without_psd):.6f} +/- {np.std(dz_without_psd)/np.sqrt(N_SHUFFLES):.6f}")
    print(f"  PSD contribution to Dz:  {np.mean(dz_with_psd) - np.mean(dz_without_psd):.6f}")
    print(f"\n  Negative eigenvalues in CnoiseU:            {neg_eig_CnoiseU_count}")
    print(f"  Negative eigenvalues in CnoiseC_shuff (mean): {np.mean(neg_eig_CnoiseC):.1f} "
          f"+/- {np.std(neg_eig_CnoiseC):.1f}")
    print(f"  Negative eigenvalues in CnoiseC_shuff (range): [{np.min(neg_eig_CnoiseC)}, {np.max(neg_eig_CnoiseC)}]")

    if abs(np.mean(dz_without_psd)) < 0.001 and abs(np.mean(dz_with_psd)) > 0.003:
        print("  --> PSD projection is the PRIMARY cause of the shift")
    elif abs(np.mean(dz_with_psd) - np.mean(dz_without_psd)) > 0.002:
        print("  --> PSD projection CONTRIBUTES to the shift")
    else:
        print("  --> PSD projection has NEGLIGIBLE effect on the shift")

    return {
        'dz_with_psd_mean': float(np.mean(dz_with_psd)),
        'dz_with_psd_se': float(np.std(dz_with_psd) / np.sqrt(N_SHUFFLES)),
        'dz_without_psd_mean': float(np.mean(dz_without_psd)),
        'dz_without_psd_se': float(np.std(dz_without_psd) / np.sqrt(N_SHUFFLES)),
        'neg_eig_CnoiseU': neg_eig_CnoiseU_count,
        'neg_eig_CnoiseC_mean': float(np.mean(neg_eig_CnoiseC)),
        'fz_U_psd': float(fz_U_psd),
        'fz_U_nopsd': float(fz_U_nopsd),
    }


# ============================================================================
# EXPERIMENT 5: Covariance vs Correlation Space (Hypothesis E)
# ============================================================================

def experiment_5_cov_vs_corr(SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU,
                              bin_edges_real):
    """Test whether the bias exists in covariance space or only in correlation space."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Covariance vs Correlation Space (Hypothesis E)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Reference values
    offdiag_CnoiseU = off_diag_mean(CnoiseU)

    cov_diffs = []   # off_diag(CnoiseC_shuff) - off_diag(CnoiseU) in raw covariance
    corr_diffs_nopsd = []  # same but in correlation space (no PSD)
    corr_diffs_psd = []    # same but in correlation space (with PSD)

    for i in tqdm(range(N_SHUFFLES), desc="Exp5: cov vs corr"):
        EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)
        _, CnoiseC_shuff, _, _, _, _ = compute_crate_from_eye(
            SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, bin_edges_real
        )

        # Covariance space
        offdiag_CnoiseC = off_diag_mean(CnoiseC_shuff)
        cov_diffs.append(offdiag_CnoiseC - offdiag_CnoiseU)

        # Correlation space without PSD
        fz_C_nopsd = fz_from_cov(CnoiseC_shuff, use_psd=False)
        fz_U_nopsd = fz_from_cov(CnoiseU, use_psd=False)
        corr_diffs_nopsd.append(fz_C_nopsd - fz_U_nopsd)

        # Correlation space with PSD
        fz_C_psd = fz_from_cov(CnoiseC_shuff, use_psd=True)
        fz_U_psd = fz_from_cov(CnoiseU, use_psd=True)
        corr_diffs_psd.append(fz_C_psd - fz_U_psd)

    cov_diffs = np.array(cov_diffs)
    corr_diffs_nopsd = np.array(corr_diffs_nopsd)
    corr_diffs_psd = np.array(corr_diffs_psd)

    print(f"\n  off_diag_mean(CnoiseU):  {offdiag_CnoiseU:.8f}")
    print(f"\n  {'Space':<25s}  {'mean(diff)':>12s}  {'SE':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*10}")
    for name, arr in [
        ("Covariance (raw)", cov_diffs),
        ("Correlation (no PSD)", corr_diffs_nopsd),
        ("Correlation (with PSD)", corr_diffs_psd),
    ]:
        print(f"  {name:<25s}  {np.mean(arr):>12.8f}  {np.std(arr)/np.sqrt(N_SHUFFLES):>10.8f}")

    if abs(np.mean(cov_diffs)) < 1e-6 and abs(np.mean(corr_diffs_nopsd)) > 0.002:
        print("\n  --> Bias is ZERO in covariance space but nonzero in correlation space")
        print("  --> The cov-to-corr nonlinearity creates the shift")
    elif abs(np.mean(cov_diffs)) > 1e-6:
        print(f"\n  --> Bias EXISTS in covariance space ({np.mean(cov_diffs):.8f})")
        print("  --> The shift originates before normalization")
    else:
        print("\n  --> Both covariance and correlation space show near-zero bias")

    return {
        'cov_diff_mean': float(np.mean(cov_diffs)),
        'cov_diff_se': float(np.std(cov_diffs) / np.sqrt(N_SHUFFLES)),
        'corr_nopsd_diff_mean': float(np.mean(corr_diffs_nopsd)),
        'corr_nopsd_diff_se': float(np.std(corr_diffs_nopsd) / np.sqrt(N_SHUFFLES)),
        'corr_psd_diff_mean': float(np.mean(corr_diffs_psd)),
        'corr_psd_diff_se': float(np.std(corr_diffs_psd) / np.sqrt(N_SHUFFLES)),
    }


# ============================================================================
# EXPERIMENT 6: Interaction Effects (find minimal fix)
# ============================================================================

def experiment_6_interactions(SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU,
                               bin_edges_real, fz_U):
    """Test combinations to find minimal set of changes that eliminates the shift."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Interaction Effects (finding minimal fix)")
    print("=" * 70)

    rng_seed = 42

    # Define conditions to test
    conditions = [
        # (name, shuffle_type, bin_edges, use_psd)
        ("baseline (global+reused+PSD)", "global", "reused", True),
        ("fresh edges only", "global", "fresh", True),
        ("no PSD only", "global", "reused", False),
        ("within-time only", "within", "reused", True),
        ("fresh + no PSD", "global", "fresh", False),
        ("within + no PSD", "within", "reused", False),
        ("within + fresh", "within", "fresh", True),
        ("within + fresh + no PSD", "within", "fresh", False),
    ]

    # For no-PSD conditions, also compute fz_U without PSD
    fz_U_psd = fz_from_cov(CnoiseU, use_psd=True)
    fz_U_nopsd = fz_from_cov(CnoiseU, use_psd=False)

    results = {}

    for cond_name, shuffle_type, edge_type, use_psd in conditions:
        rng = np.random.default_rng(rng_seed)
        fz_ref = fz_U_psd if use_psd else fz_U_nopsd

        dz_vals = []
        for i in range(N_SHUFFLES):
            if shuffle_type == "global":
                EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)
            elif shuffle_type == "within":
                EyeTraj_shuff = within_time_bin_shuffle(EyeTraj, T_idx, rng)

            n_bins_arg = bin_edges_real if edge_type == "reused" else N_BINS

            _, CnoiseC_shuff, _, _, _, _ = compute_crate_from_eye(
                SpikeCounts, EyeTraj_shuff, T_idx, Ctotal, n_bins_arg
            )
            fz_C = fz_from_cov(CnoiseC_shuff, use_psd=use_psd)
            dz_vals.append(fz_C - fz_ref)

        dz_vals = np.array(dz_vals)
        results[cond_name] = {
            'dz_mean': float(np.mean(dz_vals)),
            'dz_se': float(np.std(dz_vals) / np.sqrt(N_SHUFFLES)),
        }

    print(f"\n  {'Condition':<35s}  {'Dz_mean':>10s}  {'Dz_SE':>10s}  {'|Dz|<0.001?':>12s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*12}")
    for name, res in results.items():
        fixed = "YES" if abs(res['dz_mean']) < 0.001 else "no"
        print(f"  {name:<35s}  {res['dz_mean']:>10.6f}  {res['dz_se']:>10.6f}  {fixed:>12s}")

    # Find minimal fix
    print("\n  --- Minimal fix analysis ---")
    baseline_dz = results["baseline (global+reused+PSD)"]['dz_mean']
    for name, res in results.items():
        reduction = baseline_dz - res['dz_mean']
        if abs(reduction) > 0.001:
            print(f"  {name}: reduces |Dz| by {reduction:.6f}")

    return results


# ============================================================================
# Plotting
# ============================================================================

def make_summary_plot(exp1, exp2, exp3, exp4, exp5, exp6):
    """Generate summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel A: Bin edges (Exp 1)
    ax = axes[0, 0]
    labels = ['Reused edges', 'Fresh edges']
    means = [exp1['dz_reused_mean'], exp1['dz_fresh_mean']]
    ses = [exp1['dz_reused_se'], exp1['dz_fresh_se']]
    bars = ax.bar(labels, means, yerr=[s * 1.96 for s in ses], capsize=5,
                  color=['C0', 'C1'], alpha=0.7, edgecolor='k')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('Dz_shuff')
    ax.set_title('A. Bin Edge Effect (Hyp A)')

    # Panel B: Shuffle strategies (Exp 2)
    ax = axes[0, 1]
    labels = ['Global', 'Within-time', 'Cyclic']
    means = [exp2['dz_global_mean'], exp2['dz_within_mean'], exp2['dz_cyclic_mean']]
    ses = [exp2['dz_global_se'], exp2['dz_within_se'], exp2['dz_cyclic_se']]
    bars = ax.bar(labels, means, yerr=[s * 1.96 for s in ses], capsize=5,
                  color=['C0', 'C1', 'C2'], alpha=0.7, edgecolor='k')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('Dz_shuff')
    ax.set_title('B. Shuffle Strategy (Hyp B)')

    # Panel C: Estimator comparison (Exp 3)
    ax = axes[0, 2]
    labels = ['Cpsth', 'Crate_intercept', 'Crate_mean', 'Ctotal']
    vals = [exp3['cpsth_offdiag'], exp3['crate_intercept_mean'],
            exp3['crate_mean_mean'], exp3['ctotal_offdiag']]
    colors = ['green', 'C0', 'C1', 'gray']
    ax.bar(labels, vals, color=colors, alpha=0.7, edgecolor='k')
    ax.set_ylabel('Off-diagonal mean (cov)')
    ax.set_title('C. Estimator Comparison (Hyp C)')
    ax.tick_params(axis='x', rotation=20)

    # Panel D: PSD effect (Exp 4)
    ax = axes[1, 0]
    labels = ['With PSD', 'Without PSD']
    means = [exp4['dz_with_psd_mean'], exp4['dz_without_psd_mean']]
    ses = [exp4['dz_with_psd_se'], exp4['dz_without_psd_se']]
    bars = ax.bar(labels, means, yerr=[s * 1.96 for s in ses], capsize=5,
                  color=['C0', 'C3'], alpha=0.7, edgecolor='k')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('Dz_shuff')
    ax.set_title('D. PSD Projection (Hyp D)')

    # Panel E: Cov vs Corr space (Exp 5)
    ax = axes[1, 1]
    labels = ['Covariance', 'Corr (no PSD)', 'Corr (PSD)']
    means = [exp5['cov_diff_mean'], exp5['corr_nopsd_diff_mean'], exp5['corr_psd_diff_mean']]
    ses = [exp5['cov_diff_se'], exp5['corr_nopsd_diff_se'], exp5['corr_psd_diff_se']]
    bars = ax.bar(labels, means, yerr=[s * 1.96 for s in ses], capsize=5,
                  color=['C4', 'C3', 'C0'], alpha=0.7, edgecolor='k')
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_ylabel('Mean difference (CnoiseC - CnoiseU)')
    ax.set_title('E. Cov vs Corr Space (Hyp E)')

    # Panel F: Interaction table (Exp 6)
    ax = axes[1, 2]
    ax.axis('off')
    names = list(exp6.keys())
    dz_means = [exp6[n]['dz_mean'] for n in names]
    # Shorten names for display
    short_names = [n.replace("(global+reused+PSD)", "").strip() for n in names]

    table_data = [[f"{dz:.4f}"] for dz in dz_means]
    table = ax.table(cellText=table_data, rowLabels=short_names,
                     colLabels=['Dz_shuff'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Color cells by magnitude
    for i, dz in enumerate(dz_means):
        cell = table[i + 1, 0]
        if abs(dz) < 0.001:
            cell.set_facecolor('#90EE90')  # green
        elif abs(dz) < 0.003:
            cell.set_facecolor('#FFFF90')  # yellow
        else:
            cell.set_facecolor('#FFB0B0')  # red

    ax.set_title('F. Interaction Effects (Hyp F)')

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "shuffle_null_shift_diagnosis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to {fig_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SHUFFLE NULL SHIFT DIAGNOSIS")
    print("Testing why Dz_shuff ~ -0.006 instead of 0")
    print("=" * 70)

    device = get_device()
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Compute baseline values
    print("\nComputing baseline covariances...")
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

    fz_U = fz_from_cov(CnoiseU, use_psd=True)
    fz_C = fz_from_cov(CnoiseC, use_psd=True)
    dz_real = fz_C - fz_U

    print(f"\n  Baseline values:")
    print(f"    fz(CnoiseU) = {fz_U:.6f}")
    print(f"    fz(CnoiseC) = {fz_C:.6f}")
    print(f"    Dz_real     = {dz_real:.6f}")

    # Run all experiments
    exp1 = experiment_1_bin_edges(
        SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU, bin_edges, fz_U
    )

    exp2 = experiment_2_shuffle_strategies(
        SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU, bin_edges, fz_U
    )

    exp3 = experiment_3_estimator_comparison(
        SpikeCounts, EyeTraj, T_idx, Ctotal, Cpsth, bin_edges, Erate
    )

    exp4 = experiment_4_psd_projection(
        SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU, bin_edges, fz_U
    )

    exp5 = experiment_5_cov_vs_corr(
        SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU, bin_edges
    )

    exp6 = experiment_6_interactions(
        SpikeCounts, EyeTraj, T_idx, Ctotal, CnoiseU, bin_edges, fz_U
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)

    print(f"\n  Baseline Dz_real = {dz_real:.6f}")
    print(f"\n  Hypothesis A (bin edges):        Dz shift with reused={exp1['dz_reused_mean']:.6f}, "
          f"fresh={exp1['dz_fresh_mean']:.6f}")
    print(f"  Hypothesis B (shuffle strategy): global={exp2['dz_global_mean']:.6f}, "
          f"within={exp2['dz_within_mean']:.6f}, cyclic={exp2['dz_cyclic_mean']:.6f}")
    print(f"  Hypothesis C (estimator):        Crate_intercept-Cpsth={exp3['crate_intercept_mean']-exp3['cpsth_offdiag']:.8f}, "
          f"Crate_mean-Cpsth={exp3['crate_mean_mean']-exp3['cpsth_offdiag']:.8f}")
    print(f"  Hypothesis D (PSD):              with_PSD={exp4['dz_with_psd_mean']:.6f}, "
          f"without_PSD={exp4['dz_without_psd_mean']:.6f}")
    print(f"  Hypothesis E (cov vs corr):      cov_diff={exp5['cov_diff_mean']:.8f}, "
          f"corr_nopsd={exp5['corr_nopsd_diff_mean']:.6f}, corr_psd={exp5['corr_psd_diff_mean']:.6f}")

    print(f"\n  Interaction analysis (Exp 6):")
    baseline_dz = exp6["baseline (global+reused+PSD)"]['dz_mean']
    for name, res in exp6.items():
        status = "FIXED" if abs(res['dz_mean']) < 0.001 else f"shift={res['dz_mean']:.6f}"
        print(f"    {name:<35s}: {status}")

    # Plot
    make_summary_plot(exp1, exp2, exp3, exp4, exp5, exp6)

    # Return all results for report generation
    return {
        'dz_real': dz_real, 'fz_U': fz_U, 'fz_C': fz_C,
        'exp1': exp1, 'exp2': exp2, 'exp3': exp3,
        'exp4': exp4, 'exp5': exp5, 'exp6': exp6,
    }


if __name__ == "__main__":
    all_results = main()
