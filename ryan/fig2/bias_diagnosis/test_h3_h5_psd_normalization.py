"""
Hypothesis 3 + 5: PSD projection asymmetry and Jensen's inequality in
cov-to-corr normalization.

H3: CnoiseC = Ctotal - Crate is more indefinite than CnoiseU = Ctotal - Cpsth.
PSD projection (clamping negative eigenvalues to 0) may differentially bias
the corrected vs uncorrected noise correlations.

H5: cov_to_corr divides by sqrt(var_i * var_j). If variance estimates are noisy,
E[cov/sqrt(var)] != E[cov]/sqrt(E[var]) (Jensen's inequality). PSD projection
also changes variances (diagonal elements), which changes the normalization.

Parts:
  1. Characterize eigenspectra of CnoiseU vs CnoiseC on real data
  2. Quantify PSD projection effect on correlations for each
  3. Full decomposition with/without PSD
  4. Jensen's inequality test via diagonal stabilization
  5. Monte Carlo with controlled negative eigenvalues
  6. Bottom line: can H3+H5 explain Dz ~ -0.09?
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


def compute_covariances(SpikeCounts, EyeTraj, T_idx):
    """Compute Ctotal, Crate, Cpsth, CnoiseU, CnoiseC."""
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

    return Ctotal, Crate, Cpsth, CnoiseU, CnoiseC, Erate


# ============================================================================
# Part 1: Characterize eigenspectra
# ============================================================================

def part1_eigenspectra(CnoiseU, CnoiseC):
    """Characterize and compare eigenspectra of CnoiseU vs CnoiseC."""
    print("\n" + "="*70)
    print("PART 1: Eigenspectra characterization")
    print("="*70)

    results = {}

    for label, C in [("CnoiseU", CnoiseU), ("CnoiseC", CnoiseC)]:
        # Remove NaN rows/cols for eigendecomposition
        finite_mask = np.all(np.isfinite(C), axis=0) & np.all(np.isfinite(C), axis=1)
        C_clean = C[np.ix_(finite_mask, finite_mask)]
        C_sym = 0.5 * (C_clean + C_clean.T)
        n = C_sym.shape[0]

        eigenvalues = np.linalg.eigvalsh(C_sym)
        eigenvalues_sorted = np.sort(eigenvalues)  # ascending

        n_negative = np.sum(eigenvalues < 0)
        neg_mass = np.sum(eigenvalues[eigenvalues < 0])
        total_mass = np.sum(np.abs(eigenvalues))
        frob_norm = np.sqrt(np.sum(eigenvalues**2))
        neg_frob = np.sqrt(np.sum(eigenvalues[eigenvalues < 0]**2))
        neg_frob_frac = neg_frob / frob_norm if frob_norm > 0 else 0

        results[label] = {
            "n_neurons": n,
            "n_negative": n_negative,
            "neg_mass": neg_mass,
            "total_mass": total_mass,
            "neg_frac_mass": neg_mass / total_mass if total_mass > 0 else 0,
            "neg_frob_frac": neg_frob_frac,
            "eigenvalues": eigenvalues_sorted,
            "min_eigenvalue": eigenvalues_sorted[0],
            "max_eigenvalue": eigenvalues_sorted[-1],
        }

        print(f"\n  {label}:")
        print(f"    Neurons (finite): {n}")
        print(f"    Negative eigenvalues: {n_negative}/{n}")
        print(f"    Total negative eigenvalue mass: {neg_mass:.6f}")
        print(f"    Total absolute eigenvalue mass: {total_mass:.6f}")
        print(f"    Negative mass fraction: {neg_mass/total_mass:.4f}" if total_mass > 0 else "    N/A")
        print(f"    Frobenius norm in neg eigenspace: {neg_frob_frac:.4f}")
        print(f"    Min eigenvalue: {eigenvalues_sorted[0]:.6f}")
        print(f"    Max eigenvalue: {eigenvalues_sorted[-1]:.6f}")
        print(f"    Eigenvalue range (5 smallest): {eigenvalues_sorted[:5]}")

    # Compare
    print(f"\n  Comparison:")
    print(f"    CnoiseC has {results['CnoiseC']['n_negative'] - results['CnoiseU']['n_negative']} "
          f"more negative eigenvalues")
    print(f"    CnoiseC neg mass: {results['CnoiseC']['neg_mass']:.6f} vs "
          f"CnoiseU: {results['CnoiseU']['neg_mass']:.6f}")
    print(f"    CnoiseC neg Frobenius frac: {results['CnoiseC']['neg_frob_frac']:.4f} vs "
          f"CnoiseU: {results['CnoiseU']['neg_frob_frac']:.4f}")

    # Plot eigenspectra
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, r) in zip(axes, results.items()):
        evals = r["eigenvalues"]
        colors = ['red' if e < 0 else 'steelblue' for e in evals]
        ax.bar(range(len(evals)), evals, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f"{label} eigenspectrum\n"
                     f"({r['n_negative']} negative, mass={r['neg_mass']:.4f})")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "h3_h5_eigenspectra.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return results


# ============================================================================
# Part 2: PSD projection effect on correlations
# ============================================================================

def part2_psd_effect(CnoiseU, CnoiseC):
    """Quantify how PSD projection changes mean correlations."""
    print("\n" + "="*70)
    print("PART 2: PSD projection effect on correlations")
    print("="*70)

    results = {}

    for label, C in [("CnoiseU", CnoiseU), ("CnoiseC", CnoiseC)]:
        # Without PSD
        R_nopsd = cov_to_corr(C)
        tri_nopsd = get_upper_triangle(R_nopsd)
        tri_nopsd = tri_nopsd[np.isfinite(tri_nopsd)]

        # With PSD
        C_psd = project_to_psd(C)
        R_psd = cov_to_corr(C_psd)
        tri_psd = get_upper_triangle(R_psd)
        tri_psd = tri_psd[np.isfinite(tri_psd)]

        mean_nopsd = np.mean(tri_nopsd)
        mean_psd = np.mean(tri_psd)
        delta_mean = mean_psd - mean_nopsd

        z_nopsd = fisher_z_mean(tri_nopsd)
        z_psd = fisher_z_mean(tri_psd)
        delta_z = z_psd - z_nopsd

        results[label] = {
            "mean_nopsd": mean_nopsd,
            "mean_psd": mean_psd,
            "delta_mean": delta_mean,
            "z_nopsd": z_nopsd,
            "z_psd": z_psd,
            "delta_z_psd": delta_z,
            "n_pairs": len(tri_nopsd),
            "tri_nopsd": tri_nopsd,
            "tri_psd": tri_psd,
        }

        print(f"\n  {label}:")
        print(f"    N pairs (finite): {len(tri_nopsd)} (no PSD), {len(tri_psd)} (PSD)")
        print(f"    Mean corr (no PSD): {mean_nopsd:.6f}")
        print(f"    Mean corr (PSD):    {mean_psd:.6f}")
        print(f"    Delta mean corr:    {delta_mean:.6f}")
        print(f"    Fisher z (no PSD):  {z_nopsd:.6f}")
        print(f"    Fisher z (PSD):     {z_psd:.6f}")
        print(f"    Delta z (PSD effect): {delta_z:.6f}")

    asymmetric_bias = results["CnoiseC"]["delta_z_psd"] - results["CnoiseU"]["delta_z_psd"]
    print(f"\n  === Asymmetric PSD bias ===")
    print(f"    delta_z_psd(CnoiseC) - delta_z_psd(CnoiseU) = {asymmetric_bias:.6f}")
    print(f"    (positive = PSD projection raises corrected MORE than uncorrected)")
    print(f"    (negative = PSD projection raises uncorrected MORE, contributing to negative Dz)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, r) in zip(axes, results.items()):
        ax.hist(r["tri_nopsd"], bins=50, alpha=0.5, label=f"No PSD (mean={r['mean_nopsd']:.4f})", density=True)
        ax.hist(r["tri_psd"], bins=50, alpha=0.5, label=f"PSD (mean={r['mean_psd']:.4f})", density=True)
        ax.axvline(r["mean_nopsd"], color='C0', linestyle='--')
        ax.axvline(r["mean_psd"], color='C1', linestyle='--')
        ax.set_title(f"{label}: PSD effect on correlations\ndelta_z = {r['delta_z_psd']:.4f}")
        ax.set_xlabel("Correlation")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "h3_h5_psd_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return results, asymmetric_bias


# ============================================================================
# Part 3: Full decomposition with/without PSD
# ============================================================================

def part3_full_decomposition(CnoiseU, CnoiseC):
    """Compute Dz under all 4 PSD on/off combinations."""
    print("\n" + "="*70)
    print("PART 3: Full decomposition with/without PSD")
    print("="*70)

    def get_z(C, use_psd):
        if use_psd:
            C = project_to_psd(C)
        R = cov_to_corr(C)
        tri = get_upper_triangle(R)
        tri = tri[np.isfinite(tri)]
        return fisher_z_mean(tri), np.mean(tri), tri

    # 1. Both PSD (paper pipeline)
    z_U_psd, m_U_psd, tri_U_psd = get_z(CnoiseU, True)
    z_C_psd, m_C_psd, tri_C_psd = get_z(CnoiseC, True)
    dz_psd = z_C_psd - z_U_psd

    # 2. Neither PSD
    z_U_nopsd, m_U_nopsd, tri_U_nopsd = get_z(CnoiseU, False)
    z_C_nopsd, m_C_nopsd, tri_C_nopsd = get_z(CnoiseC, False)
    dz_nopsd = z_C_nopsd - z_U_nopsd

    # 3. PSD on U only
    z_U_psdonly, _, _ = get_z(CnoiseU, True)
    z_C_nopsdonly, _, _ = get_z(CnoiseC, False)
    dz_psd_U_only = z_C_nopsdonly - z_U_psdonly

    # 4. PSD on C only
    z_U_nopsdonly, _, _ = get_z(CnoiseU, False)
    z_C_psdonly, _, _ = get_z(CnoiseC, True)
    dz_psd_C_only = z_C_psdonly - z_U_nopsdonly

    print(f"\n  Fisher z values:")
    print(f"    z_U (PSD): {z_U_psd:.6f}    z_U (no PSD): {z_U_nopsd:.6f}")
    print(f"    z_C (PSD): {z_C_psd:.6f}    z_C (no PSD): {z_C_nopsd:.6f}")

    print(f"\n  Dz = z_corrected - z_uncorrected:")
    print(f"    (1) Both PSD (paper):   Dz_psd       = {dz_psd:.6f}")
    print(f"    (2) Neither PSD:        Dz_nopsd     = {dz_nopsd:.6f}")
    print(f"    (3) PSD on U only:      Dz_psd_U     = {dz_psd_U_only:.6f}")
    print(f"    (4) PSD on C only:      Dz_psd_C     = {dz_psd_C_only:.6f}")

    psd_contribution = dz_psd - dz_nopsd
    print(f"\n  PSD contribution to Dz: {psd_contribution:.6f}")
    print(f"    (Dz_psd - Dz_nopsd)")
    if abs(dz_psd) > 1e-6:
        print(f"    = {psd_contribution/dz_psd*100:.1f}% of total Dz_psd")

    results = {
        "dz_psd": dz_psd,
        "dz_nopsd": dz_nopsd,
        "dz_psd_U_only": dz_psd_U_only,
        "dz_psd_C_only": dz_psd_C_only,
        "psd_contribution": psd_contribution,
        "z_U_psd": z_U_psd, "z_U_nopsd": z_U_nopsd,
        "z_C_psd": z_C_psd, "z_C_nopsd": z_C_nopsd,
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Both PSD\n(paper)", "Neither PSD", "PSD on U only", "PSD on C only"]
    values = [dz_psd, dz_nopsd, dz_psd_U_only, dz_psd_C_only]
    colors = ['C0', 'C1', 'C2', 'C3']
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.4f}", ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    ax.set_ylabel("Dz (corrected - uncorrected)")
    ax.set_title("Effect of PSD projection on noise correlation bias")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "h3_h5_decomposition.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return results


# ============================================================================
# Part 4: Jensen's inequality / normalization test
# ============================================================================

def part4_jensen_normalization(CnoiseU, CnoiseC, SpikeCounts, EyeTraj, T_idx):
    """Test whether diagonal noise in covariance estimates biases normalization."""
    print("\n" + "="*70)
    print("PART 4: Jensen's inequality in normalization")
    print("="*70)

    results = {}

    for label, C in [("CnoiseU", CnoiseU), ("CnoiseC", CnoiseC)]:
        C_psd = project_to_psd(C)

        # Standard normalization
        R_standard = cov_to_corr(C_psd)
        tri_standard = get_upper_triangle(R_standard)
        tri_standard = tri_standard[np.isfinite(tri_standard)]

        # Off-diagonal covariance (raw, before normalization)
        offdiag = get_upper_triangle(C_psd)
        offdiag_finite = offdiag[np.isfinite(offdiag)]
        mean_cov = np.mean(offdiag_finite)

        # Diagonal variances
        variances = np.diag(C_psd)
        valid = variances > 1e-3
        n_valid = np.sum(valid)

        # Method 1: Standard (noisy diagonal)
        z_standard = fisher_z_mean(tri_standard)

        # Method 2: Stabilized diagonal — use median variance for normalization
        median_var = np.median(variances[valid])
        C_stab = C_psd.copy()
        np.fill_diagonal(C_stab, median_var)
        R_stab = cov_to_corr(C_stab)
        tri_stab = get_upper_triangle(R_stab)
        tri_stab = tri_stab[np.isfinite(tri_stab)]
        z_stab_median = fisher_z_mean(tri_stab)

        # Method 3: Smoothed diagonal — use geometric mean of each pair's variances
        # This is what standard correlation does, but let's try with clipped variances
        std_devs = np.sqrt(np.maximum(variances, 1e-3))
        std_clipped = np.clip(std_devs, np.percentile(std_devs[valid], 10),
                              np.percentile(std_devs[valid], 90))
        outer_std_clipped = np.outer(std_clipped, std_clipped)
        R_clipped = C_psd / outer_std_clipped
        R_clipped = np.clip(R_clipped, -1, 1)
        np.fill_diagonal(R_clipped, 0)
        R_clipped[~np.isfinite(R_clipped)] = np.nan
        tri_clipped = get_upper_triangle(R_clipped)
        tri_clipped = tri_clipped[np.isfinite(tri_clipped)]
        z_clipped = fisher_z_mean(tri_clipped)

        # Method 4: Bootstrap diagonal stability test
        # Resample spike counts, recompute covariance, measure diagonal variance
        n_boot = 50
        rng = np.random.default_rng(42)
        n_samples = SpikeCounts.shape[0]
        diag_boots = []
        for _ in range(n_boot):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            S_boot = SpikeCounts[idx]
            ix_fin = np.isfinite(S_boot.sum(1).detach().cpu().numpy())
            C_boot = torch.cov(S_boot[ix_fin].T, correction=1).detach().cpu().numpy()
            diag_boots.append(np.diag(C_boot))
        diag_boots = np.array(diag_boots)
        diag_cv = np.std(diag_boots, axis=0) / (np.mean(diag_boots, axis=0) + 1e-12)
        mean_diag_cv = np.mean(diag_cv[valid])

        results[label] = {
            "mean_cov": mean_cov,
            "z_standard": z_standard,
            "z_stab_median": z_stab_median,
            "z_clipped": z_clipped,
            "mean_diag_cv": mean_diag_cv,
            "n_valid": n_valid,
        }

        print(f"\n  {label}:")
        print(f"    Mean off-diagonal covariance: {mean_cov:.6f}")
        print(f"    N valid neurons: {n_valid}")
        print(f"    Diagonal CV (bootstrap): {mean_diag_cv:.4f}")
        print(f"    z (standard):       {z_standard:.6f}")
        print(f"    z (median diag):    {z_stab_median:.6f}")
        print(f"    z (clipped diag):   {z_clipped:.6f}")
        print(f"    Jensen shift (standard - clipped): {z_standard - z_clipped:.6f}")

    # Compute Dz under different normalization schemes
    dz_standard = results["CnoiseC"]["z_standard"] - results["CnoiseU"]["z_standard"]
    dz_clipped = results["CnoiseC"]["z_clipped"] - results["CnoiseU"]["z_clipped"]
    dz_median = results["CnoiseC"]["z_stab_median"] - results["CnoiseU"]["z_stab_median"]

    print(f"\n  Dz comparison across normalization:")
    print(f"    Dz (standard):  {dz_standard:.6f}")
    print(f"    Dz (clipped):   {dz_clipped:.6f}")
    print(f"    Dz (median):    {dz_median:.6f}")
    print(f"    Jensen contribution to Dz: {dz_standard - dz_clipped:.6f}")

    results["dz_standard"] = dz_standard
    results["dz_clipped"] = dz_clipped
    results["dz_median"] = dz_median
    results["jensen_contribution"] = dz_standard - dz_clipped

    return results


# ============================================================================
# Part 5: Monte Carlo with controlled negative eigenvalues
# ============================================================================

def part5_monte_carlo():
    """Simulate PSD projection effect as function of negative eigenvalue mass."""
    print("\n" + "="*70)
    print("PART 5: Monte Carlo — negative eigenvalue mass vs PSD-induced bias")
    print("="*70)

    rng = np.random.default_rng(42)
    n_neurons = 40  # typical after filtering
    n_reps = 100

    # Range of negative eigenvalue mass fractions to test
    neg_mass_fractions = np.linspace(0, 0.5, 11)

    mc_results = []

    for neg_frac in neg_mass_fractions:
        delta_z_list = []
        delta_mean_list = []

        for rep in range(n_reps):
            # Generate a covariance matrix with controlled eigenstructure
            # Start with random positive eigenvalues
            total_variance = 1.0
            n_pos = max(1, int(n_neurons * (1 - neg_frac * 2)))
            n_neg = n_neurons - n_pos

            # Eigenvalues: positive part sums to (1 - neg_frac) * total
            pos_evals = rng.exponential(1.0, n_pos)
            pos_evals = pos_evals / pos_evals.sum() * (1 - neg_frac) * total_variance

            if n_neg > 0:
                neg_evals = -rng.exponential(1.0, n_neg)
                neg_evals = neg_evals / np.abs(neg_evals.sum()) * neg_frac * total_variance
                all_evals = np.concatenate([pos_evals, neg_evals])
            else:
                all_evals = pos_evals

            # Random orthogonal basis
            Q, _ = np.linalg.qr(rng.standard_normal((n_neurons, n_neurons)))
            C = (Q * all_evals) @ Q.T
            C = 0.5 * (C + C.T)

            # Correlations without PSD
            R_nopsd = cov_to_corr(C)
            tri_nopsd = get_upper_triangle(R_nopsd)
            tri_nopsd = tri_nopsd[np.isfinite(tri_nopsd)]

            # Correlations with PSD
            C_psd = project_to_psd(C)
            R_psd = cov_to_corr(C_psd)
            tri_psd = get_upper_triangle(R_psd)
            tri_psd = tri_psd[np.isfinite(tri_psd)]

            if len(tri_nopsd) > 0 and len(tri_psd) > 0:
                z_nopsd = fisher_z_mean(tri_nopsd)
                z_psd = fisher_z_mean(tri_psd)
                delta_z_list.append(z_psd - z_nopsd)
                delta_mean_list.append(np.mean(tri_psd) - np.mean(tri_nopsd))

        mc_results.append({
            "neg_frac": neg_frac,
            "delta_z_mean": np.mean(delta_z_list),
            "delta_z_std": np.std(delta_z_list),
            "delta_mean_corr": np.mean(delta_mean_list),
            "n_valid": len(delta_z_list),
        })

        print(f"  neg_mass_frac={neg_frac:.2f}: delta_z = {np.mean(delta_z_list):.6f} "
              f"+/- {np.std(delta_z_list):.6f}, delta_mean_corr = {np.mean(delta_mean_list):.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fracs = [r["neg_frac"] for r in mc_results]
    dz_means = [r["delta_z_mean"] for r in mc_results]
    dz_stds = [r["delta_z_std"] for r in mc_results]
    dm_means = [r["delta_mean_corr"] for r in mc_results]

    axes[0].errorbar(fracs, dz_means, yerr=dz_stds, marker='o', capsize=3)
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_xlabel("Negative eigenvalue mass fraction")
    axes[0].set_ylabel("Delta Fisher z (PSD - no PSD)")
    axes[0].set_title("PSD projection shift in Fisher z\n(Monte Carlo, n=40 neurons)")

    axes[1].plot(fracs, dm_means, marker='o')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel("Negative eigenvalue mass fraction")
    axes[1].set_ylabel("Delta mean correlation (PSD - no PSD)")
    axes[1].set_title("PSD projection shift in mean correlation")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "h3_h5_monte_carlo.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return mc_results


def part5b_predict_real_data_bias(mc_results, eigen_results):
    """Use MC curve to predict PSD-induced bias for real data."""
    print("\n  Predicting real-data PSD bias from Monte Carlo curve:")

    fracs = np.array([r["neg_frac"] for r in mc_results])
    dz_means = np.array([r["delta_z_mean"] for r in mc_results])

    for label in ["CnoiseU", "CnoiseC"]:
        real_neg_frac = abs(eigen_results[label]["neg_frac_mass"])
        # Linear interpolation
        predicted_dz = np.interp(real_neg_frac, fracs, dz_means)
        print(f"    {label}: neg_mass_frac = {real_neg_frac:.4f} -> predicted delta_z = {predicted_dz:.6f}")

    real_neg_frac_U = abs(eigen_results["CnoiseU"]["neg_frac_mass"])
    real_neg_frac_C = abs(eigen_results["CnoiseC"]["neg_frac_mass"])
    pred_dz_U = np.interp(real_neg_frac_U, fracs, dz_means)
    pred_dz_C = np.interp(real_neg_frac_C, fracs, dz_means)
    predicted_asymmetric = pred_dz_C - pred_dz_U
    print(f"    Predicted asymmetric PSD bias: {predicted_asymmetric:.6f}")
    return predicted_asymmetric


# ============================================================================
# Part 6: Bottom line
# ============================================================================

def part6_bottom_line(decomp_results, psd_results, asymmetric_bias, jensen_results,
                      mc_results, eigen_results, predicted_mc_bias):
    """Summarize whether H3+H5 explain the observed Dz ~ -0.09."""
    print("\n" + "="*70)
    print("PART 6: BOTTOM LINE")
    print("="*70)

    observed_dz = decomp_results["dz_psd"]
    print(f"\n  Observed Dz (with PSD, paper pipeline): {observed_dz:.6f}")

    # H3: PSD projection asymmetry
    psd_contribution = decomp_results["psd_contribution"]
    psd_pct = psd_contribution / observed_dz * 100 if abs(observed_dz) > 1e-6 else float('nan')
    print(f"\n  H3 — PSD projection asymmetry:")
    print(f"    PSD contribution to Dz: {psd_contribution:.6f}")
    print(f"    = {psd_pct:.1f}% of observed Dz")
    print(f"    Asymmetric PSD bias (C vs U): {asymmetric_bias:.6f}")
    print(f"    MC-predicted asymmetric bias: {predicted_mc_bias:.6f}")

    # H5: Jensen's inequality
    jensen_contribution = jensen_results.get("jensen_contribution", 0)
    jensen_pct = jensen_contribution / observed_dz * 100 if abs(observed_dz) > 1e-6 else float('nan')
    print(f"\n  H5 — Jensen's inequality in normalization:")
    print(f"    Jensen contribution to Dz: {jensen_contribution:.6f}")
    print(f"    = {jensen_pct:.1f}% of observed Dz")

    # Combined
    combined = psd_contribution + jensen_contribution
    combined_pct = combined / observed_dz * 100 if abs(observed_dz) > 1e-6 else float('nan')
    print(f"\n  Combined H3+H5:")
    print(f"    Combined contribution: {combined:.6f}")
    print(f"    = {combined_pct:.1f}% of observed Dz")

    # Dz without PSD (the fundamental gap)
    dz_nopsd = decomp_results["dz_nopsd"]
    print(f"\n  Key insight:")
    print(f"    Dz WITHOUT PSD: {dz_nopsd:.6f}")
    print(f"    Dz WITH PSD:    {observed_dz:.6f}")
    print(f"    The bias exists REGARDLESS of PSD projection.")
    if abs(dz_nopsd) > 0.5 * abs(observed_dz):
        print(f"    => H3 (PSD) is NOT the primary driver. The bias is in the covariance.")
    else:
        print(f"    => H3 (PSD) is a significant contributor.")

    return {
        "observed_dz": observed_dz,
        "psd_contribution": psd_contribution,
        "psd_pct": psd_pct,
        "jensen_contribution": jensen_contribution,
        "jensen_pct": jensen_pct,
        "combined": combined,
        "combined_pct": combined_pct,
        "dz_nopsd": dz_nopsd,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("H3 + H5: PSD Projection Asymmetry & Jensen's Inequality")
    print(f"Session: {SESSION_NAME}")
    print("="*70)

    device = get_device()

    # Load data
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Compute covariances
    print("\nComputing covariance decomposition...")
    Ctotal, Crate, Cpsth, CnoiseU, CnoiseC, Erate = compute_covariances(
        SpikeCounts, EyeTraj, T_idx
    )

    # Part 1: Eigenspectra
    eigen_results = part1_eigenspectra(CnoiseU, CnoiseC)

    # Part 2: PSD effect
    psd_results, asymmetric_bias = part2_psd_effect(CnoiseU, CnoiseC)

    # Part 3: Full decomposition
    decomp_results = part3_full_decomposition(CnoiseU, CnoiseC)

    # Part 4: Jensen's inequality
    jensen_results = part4_jensen_normalization(CnoiseU, CnoiseC, SpikeCounts, EyeTraj, T_idx)

    # Part 5: Monte Carlo
    mc_results = part5_monte_carlo()
    predicted_mc_bias = part5b_predict_real_data_bias(mc_results, eigen_results)

    # Part 6: Bottom line
    summary = part6_bottom_line(
        decomp_results, psd_results, asymmetric_bias, jensen_results,
        mc_results, eigen_results, predicted_mc_bias
    )

    print("\n" + "="*70)
    print("Done. Figures saved to:", SAVE_DIR)
    print("="*70)

    return summary


if __name__ == "__main__":
    summary = main()
