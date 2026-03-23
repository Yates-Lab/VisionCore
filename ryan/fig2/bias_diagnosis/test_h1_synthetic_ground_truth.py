"""
Hypothesis 1: Does the LOTC pipeline itself introduce a negative bias in corrected noise correlations?

Generates synthetic data with KNOWN ground truth (PSTH, FEM gain fields, intrinsic noise correlations),
runs the exact pipeline from VisionCore.covariance, and compares recovered vs true noise correlations.

Cases:
  A: Zero true noise correlations (diagonal Σ_intrinsic) → pipeline should give Δz ≈ 0
  B: Positive true noise correlations (r ≈ 0.05) → pipeline should recover corrected r ≈ 0.05
  C: High-rate neurons (0.5–2.0 sp/bin) matching regime of strongest observed bias
  D: Vary N_trials (200–2000) to check finite-sample effects
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

# Pipeline imports
from VisionCore.covariance import (
    compute_conditional_second_moments,
    estimate_rate_covariance,
    bagged_split_half_psth_covariance,
    cov_to_corr,
    get_upper_triangle,
)
from VisionCore.subspace import project_to_psd
from VisionCore.stats import fisher_z_mean

if torch.cuda.is_available():
    # Use GPU with most free memory
    free_mem = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_mem.append(free)
    best_gpu = int(np.argmax(free_mem))
    DEVICE = f"cuda:{best_gpu}"
    print(f"Selected GPU {best_gpu} with {free_mem[best_gpu]/1e9:.1f} GB free")
else:
    DEVICE = "cpu"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_cells=40,
    n_trials=600,
    n_time_bins=50,
    t_hist=10,
    mean_rate=0.3,
    rate_range=(0.05, 0.8),
    fem_strength=0.15,
    true_noise_corr=0.0,
    eye_drift_std=0.02,
    seed=42,
    device="cpu",
):
    """
    Generate synthetic spike-count data with known PSTH, FEM, and noise components.

    Returns
    -------
    SpikeCounts : (N_windows, n_cells)
    EyeTraj : (N_windows, t_hist, 2)
    T_idx : (N_windows,)
    true_Sigma_intrinsic : (n_cells, n_cells) — the true intrinsic noise covariance
    true_noise_corr_matrix : (n_cells, n_cells) — true noise correlation matrix
    info : dict with generation metadata
    """
    rng = np.random.default_rng(seed)

    # --- 1. PSTH: smooth random firing rate profile per neuron ---
    # Use sinusoidal basis for smoothness
    n_basis = 5
    basis_freqs = rng.uniform(0.5, 3.0, size=(n_cells, n_basis))
    basis_phases = rng.uniform(0, 2 * np.pi, size=(n_cells, n_basis))
    basis_amps = rng.uniform(0.3, 1.0, size=(n_cells, n_basis))
    t_grid = np.linspace(0, 2 * np.pi, n_time_bins)

    psth = np.zeros((n_time_bins, n_cells))
    for c in range(n_cells):
        for b in range(n_basis):
            psth[:, c] += basis_amps[c, b] * np.sin(basis_freqs[c, b] * t_grid + basis_phases[c, b])

    # Scale PSTH to have desired mean rate and range
    psth = psth - psth.min(axis=0, keepdims=True)
    psth = psth / (psth.max(axis=0, keepdims=True) + 1e-8)
    cell_base_rates = rng.uniform(rate_range[0], rate_range[1], size=n_cells)
    psth_modulation = rng.uniform(0.3, 0.8, size=n_cells)  # fraction of rate that is PSTH-modulated
    psth = cell_base_rates[None, :] * (1.0 - psth_modulation[None, :] + psth_modulation[None, :] * psth)

    # --- 2. FEM gain fields ---
    # Each neuron has a preferred eye position (gain field center)
    gain_centers = rng.normal(0, 0.3, size=(n_cells, 2))
    gain_widths = rng.uniform(0.5, 2.0, size=n_cells)

    # --- 3. Eye trajectories: random walk per trial ---
    eye_trajs_all = []
    for trial in range(n_trials):
        eye = np.zeros((t_hist, 2))
        eye[0] = rng.normal(0, 0.1, size=2)
        for tt in range(1, t_hist):
            eye[tt] = eye[tt - 1] + rng.normal(0, eye_drift_std, size=2)
        eye_trajs_all.append(eye)
    eye_trajs_all = np.array(eye_trajs_all)  # (n_trials, t_hist, 2)

    # Mean eye position per trial (for gain field computation)
    mean_eye = eye_trajs_all.mean(axis=1)  # (n_trials, 2)

    # --- 4. Compute FEM gain modulation ---
    # gain(trial, cell) = exp(-||eye - center||^2 / (2 * width^2))
    # Multiplicative gain on firing rate
    fem_gain = np.zeros((n_trials, n_cells))
    for c in range(n_cells):
        d2 = np.sum((mean_eye - gain_centers[c]) ** 2, axis=1)
        fem_gain[:, c] = np.exp(-d2 / (2 * gain_widths[c] ** 2))

    # Normalize to mean 1, then scale by fem_strength
    fem_gain = fem_gain / fem_gain.mean(axis=0, keepdims=True)
    fem_modulation = 1.0 + fem_strength * (fem_gain - 1.0)

    # --- 5. Intrinsic noise correlation structure ---
    # Use a latent Gaussian model: z ~ N(0, Sigma_corr), rate = base_rate * exp(sigma*z - sigma^2/2)
    # The noise_scale sigma controls SNR. For Poisson data, spike-count correlations are:
    #   r_spike ≈ r_latent * Var_signal / (Var_signal + Var_poisson)
    # We need a large sigma to get measurable spike-count correlations at low rates.
    if true_noise_corr > 0:
        Sigma_corr = np.eye(n_cells) * (1 - true_noise_corr) + true_noise_corr * np.ones((n_cells, n_cells))
        L_corr = np.linalg.cholesky(Sigma_corr)
    else:
        L_corr = np.eye(n_cells)
        Sigma_corr = np.eye(n_cells)

    # Choose noise_scale to produce measurable correlations:
    # Higher rates need smaller noise_scale; lower rates need larger.
    # sigma=0.5 gives ~10-20% CV in rate, which with r_latent=0.05 should yield r_spike~0.01-0.03
    noise_scale = 0.5

    # --- 6. Generate spike counts ---
    N_windows = n_trials * n_time_bins
    SpikeCounts_np = np.zeros((N_windows, n_cells))
    T_idx_np = np.zeros(N_windows, dtype=np.int64)
    EyeTraj_np = np.zeros((N_windows, t_hist, 2))

    idx = 0
    for trial in range(n_trials):
        for t in range(n_time_bins):
            lam = psth[t, :] * fem_modulation[trial, :]

            z = L_corr @ rng.standard_normal(n_cells)
            lam_noisy = lam * np.exp(noise_scale * z - 0.5 * noise_scale ** 2)
            lam_noisy = np.maximum(lam_noisy, 1e-6)
            spikes = rng.poisson(lam_noisy)

            SpikeCounts_np[idx] = spikes
            T_idx_np[idx] = t
            EyeTraj_np[idx] = eye_trajs_all[trial]
            idx += 1

    # --- 7. Compute TRUE intrinsic covariance ---
    # The true intrinsic noise cov is the covariance of spikes AFTER removing PSTH and FEM effects.
    # We can compute it analytically or empirically from the generative model.
    # Empirical: compute residuals after removing psth * fem_modulation for each (trial, t)
    residuals = np.zeros((N_windows, n_cells))
    idx = 0
    for trial in range(n_trials):
        for t in range(n_time_bins):
            expected_rate = psth[t, :] * fem_modulation[trial, :]
            residuals[idx] = SpikeCounts_np[idx] - expected_rate
            idx += 1

    true_Sigma_intrinsic = np.cov(residuals.T)
    true_noise_corr_matrix = cov_to_corr(true_Sigma_intrinsic, min_var=1e-6)

    # Convert to tensors
    SpikeCounts = torch.tensor(SpikeCounts_np, dtype=torch.float32, device=device)
    EyeTraj = torch.tensor(EyeTraj_np, dtype=torch.float32, device=device)
    T_idx = torch.tensor(T_idx_np, dtype=torch.long, device=device)

    info = {
        "n_cells": n_cells,
        "n_trials": n_trials,
        "n_time_bins": n_time_bins,
        "n_windows": N_windows,
        "mean_rate_actual": SpikeCounts_np.mean(),
        "true_noise_corr_target": true_noise_corr,
        "true_noise_corr_actual": np.nanmean(get_upper_triangle(true_noise_corr_matrix)),
        "psth": psth,
        "fem_modulation": fem_modulation,
    }

    return SpikeCounts, EyeTraj, T_idx, true_Sigma_intrinsic, true_noise_corr_matrix, info


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(SpikeCounts, EyeTraj, T_idx, n_bins=15):
    """Run the LOTC pipeline and return all covariance estimates."""
    S_np = SpikeCounts.detach().cpu().numpy()

    # 1. Total covariance
    Ctotal = np.cov(S_np.T)

    # 2. Rate covariance (FEM-corrected)
    Crate, Erate, Ceye, bin_centers, count_e, bin_edges = estimate_rate_covariance(
        SpikeCounts, EyeTraj, T_idx, n_bins=n_bins, Ctotal=Ctotal, intercept_mode="linear"
    )

    # 3. PSTH covariance
    Cpsth, PSTH_mean = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=20, global_mean=Erate
    )

    # 4. Noise covariances
    CnoiseU = Ctotal - Cpsth  # uncorrected
    CnoiseC = Ctotal - Crate  # corrected (FEM removed)

    # 5. PSD projection
    CnoiseU_psd = project_to_psd(CnoiseU)
    CnoiseC_psd = project_to_psd(CnoiseC)

    # 6. Correlation matrices
    corrU = cov_to_corr(CnoiseU_psd)
    corrC = cov_to_corr(CnoiseC_psd)

    # 7. Fisher z means
    zU = fisher_z_mean(get_upper_triangle(corrU))
    zC = fisher_z_mean(get_upper_triangle(corrC))
    dz = zC - zU

    return {
        "Ctotal": Ctotal,
        "Crate": Crate,
        "Cpsth": Cpsth,
        "CnoiseU": CnoiseU,
        "CnoiseC": CnoiseC,
        "CnoiseU_psd": CnoiseU_psd,
        "CnoiseC_psd": CnoiseC_psd,
        "corrU": corrU,
        "corrC": corrC,
        "zU": zU,
        "zC": zC,
        "dz": dz,
        "Erate": Erate,
        "Ceye": Ceye,
        "bin_centers": bin_centers,
        "count_e": count_e,
    }


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES = {
    "A: Zero noise corr (diagonal)": dict(
        n_cells=40, n_trials=500, n_time_bins=20, true_noise_corr=0.0, rate_range=(0.05, 0.8),
        fem_strength=0.15, seed=42,
    ),
    "B: Positive noise corr (r≈0.05)": dict(
        n_cells=40, n_trials=500, n_time_bins=20, true_noise_corr=0.05, rate_range=(0.05, 0.8),
        fem_strength=0.15, seed=43,
    ),
    "C: High-rate neurons (0.5-2.0)": dict(
        n_cells=40, n_trials=500, n_time_bins=20, true_noise_corr=0.03, rate_range=(0.5, 2.0),
        fem_strength=0.15, seed=44,
    ),
    "D1: N_trials=200": dict(
        n_cells=40, n_trials=200, n_time_bins=20, true_noise_corr=0.03, rate_range=(0.05, 0.8),
        fem_strength=0.15, seed=45,
    ),
    "D2: N_trials=600": dict(
        n_cells=40, n_trials=600, n_time_bins=20, true_noise_corr=0.03, rate_range=(0.05, 0.8),
        fem_strength=0.15, seed=46,
    ),
    "D3: N_trials=1500": dict(
        n_cells=40, n_trials=1500, n_time_bins=10, true_noise_corr=0.03, rate_range=(0.05, 0.8),
        fem_strength=0.15, seed=47,
    ),
    "E: Strong FEM (0.4)": dict(
        n_cells=40, n_trials=500, n_time_bins=20, true_noise_corr=0.03, rate_range=(0.1, 1.0),
        fem_strength=0.40, seed=48,
    ),
    "F: High rate + strong corr": dict(
        n_cells=40, n_trials=500, n_time_bins=20, true_noise_corr=0.10, rate_range=(0.5, 2.0),
        fem_strength=0.15, seed=49,
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("H1: Synthetic Ground-Truth Test — LOTC Pipeline Bias Diagnosis")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print()

    results_summary = {}

    for case_name, params in CASES.items():
        print("-" * 70)
        print(f"  Case: {case_name}")
        print("-" * 70)

        # Generate data
        SpikeCounts, EyeTraj, T_idx, true_Sigma, true_corr, info = generate_synthetic_data(
            device=DEVICE, **params
        )

        print(f"  N_windows={info['n_windows']}, N_cells={info['n_cells']}, "
              f"N_trials={info['n_trials']}")
        print(f"  Mean rate: {info['mean_rate_actual']:.4f} sp/bin")
        print(f"  True noise corr (target): {info['true_noise_corr_target']:.4f}")
        print(f"  True noise corr (actual empirical): {info['true_noise_corr_actual']:.4f}")

        # Run pipeline
        res = run_pipeline(SpikeCounts, EyeTraj, T_idx, n_bins=15)

        # Compare with true values
        true_z = fisher_z_mean(get_upper_triangle(true_corr))

        # Also compute what the "ideal" corrected noise cov would be
        # CnoiseC_ideal = Ctotal - Crate_true - Cpsth ≈ Sigma_intrinsic
        # But pipeline gives CnoiseC = Ctotal - Crate_pipeline

        # Detailed diagnostics
        corrU_vals = get_upper_triangle(res["corrU"])
        corrC_vals = get_upper_triangle(res["corrC"])
        true_corr_vals = get_upper_triangle(true_corr)

        corrU_finite = corrU_vals[np.isfinite(corrU_vals)]
        corrC_finite = corrC_vals[np.isfinite(corrC_vals)]
        true_corr_finite = true_corr_vals[np.isfinite(true_corr_vals)]

        print(f"\n  --- Pipeline Results ---")
        print(f"  Fisher z (uncorrected):  {res['zU']:+.4f}")
        print(f"  Fisher z (corrected):    {res['zC']:+.4f}")
        print(f"  Δz (corrected - uncorr): {res['dz']:+.4f}")
        print(f"  True noise Fisher z:     {true_z:+.4f}")
        print()
        print(f"  Mean corr (uncorrected): {np.nanmean(corrU_finite):+.4f}")
        print(f"  Mean corr (corrected):   {np.nanmean(corrC_finite):+.4f}")
        print(f"  True mean noise corr:    {np.nanmean(true_corr_finite):+.4f}")
        print()

        # Check for NaN fraction
        n_total = len(corrC_vals)
        n_nan_C = np.sum(~np.isfinite(corrC_vals))
        n_nan_U = np.sum(~np.isfinite(corrU_vals))
        print(f"  NaN fraction: corrU={n_nan_U/n_total:.2%}, corrC={n_nan_C/n_total:.2%}")

        # Bias metrics
        bias_dz = res["dz"]
        bias_corrected_vs_true = res["zC"] - true_z
        print(f"\n  Pipeline bias (Δz):                     {bias_dz:+.4f}")
        print(f"  Corrected z - True z:                   {bias_corrected_vs_true:+.4f}")
        print(f"  Observed real-data Δz for comparison:    -0.11 (pooled), -0.30 (high-rate)")

        # Crate diagnostics
        crate_diag = np.diag(res["Crate"])
        ctotal_diag = np.diag(res["Ctotal"])
        crate_fraction = np.nanmean(crate_diag / ctotal_diag)
        print(f"\n  Crate/Ctotal diag fraction (mean):      {crate_fraction:.4f}")

        cpsth_diag = np.diag(res["Cpsth"])
        cpsth_fraction = np.nanmean(cpsth_diag / ctotal_diag)
        print(f"  Cpsth/Ctotal diag fraction (mean):      {cpsth_fraction:.4f}")

        # Check if Crate off-diagonal is too large (overestimation → negative bias)
        crate_offdiag = np.nanmean(get_upper_triangle(res["Crate"]))
        ctotal_offdiag = np.nanmean(get_upper_triangle(res["Ctotal"]))
        print(f"  Crate off-diag mean:                    {crate_offdiag:.6f}")
        print(f"  Ctotal off-diag mean:                   {ctotal_offdiag:.6f}")
        print()

        results_summary[case_name] = {
            "dz": res["dz"],
            "zU": res["zU"],
            "zC": res["zC"],
            "true_z": true_z,
            "bias_corrected_vs_true": bias_corrected_vs_true,
            "mean_rate": info["mean_rate_actual"],
            "true_noise_corr": info["true_noise_corr_actual"],
            "crate_fraction": crate_fraction,
            "n_trials": info["n_trials"],
        }

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Case':<35s} {'Δz':>7s} {'zC':>7s} {'zU':>7s} {'true_z':>7s} {'bias':>7s} {'rate':>6s}")
    print("-" * 80)
    for case_name, r in results_summary.items():
        print(f"{case_name:<35s} {r['dz']:+.4f} {r['zC']:+.4f} {r['zU']:+.4f} "
              f"{r['true_z']:+.4f} {r['bias_corrected_vs_true']:+.4f} {r['mean_rate']:.3f}")
    print("-" * 80)
    print("bias = zC - true_z (positive = overestimate, negative = underestimate)")
    print(f"Real data reference: Δz ≈ -0.11 (pooled), Δz ≈ -0.30 (high-rate pairs)")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Δz across cases
    case_labels = [k.split(":")[0].strip() for k in results_summary.keys()]
    dzs = [r["dz"] for r in results_summary.values()]
    biases = [r["bias_corrected_vs_true"] for r in results_summary.values()]
    x = np.arange(len(case_labels))

    axes[0].bar(x - 0.15, dzs, 0.3, label="Δz (pipeline)", color="steelblue")
    axes[0].bar(x + 0.15, biases, 0.3, label="zC - true_z (bias)", color="salmon")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].axhline(-0.11, color="red", ls="--", lw=1, label="Real data Δz=-0.11")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(case_labels, rotation=45, ha="right")
    axes[0].set_ylabel("Fisher z")
    axes[0].set_title("Pipeline Δz and Bias")
    axes[0].legend(fontsize=7)

    # Panel 2: N_trials scaling (D cases)
    d_cases = {k: v for k, v in results_summary.items() if k.startswith("D")}
    if d_cases:
        ntr = [v["n_trials"] for v in d_cases.values()]
        dz_d = [v["dz"] for v in d_cases.values()]
        bias_d = [v["bias_corrected_vs_true"] for v in d_cases.values()]
        axes[1].plot(ntr, dz_d, "o-", label="Δz")
        axes[1].plot(ntr, bias_d, "s-", label="zC - true_z")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel("N_trials")
        axes[1].set_ylabel("Fisher z")
        axes[1].set_title("Finite-Sample Scaling")
        axes[1].legend()

    # Panel 3: Crate fraction
    crate_fracs = [r["crate_fraction"] for r in results_summary.values()]
    axes[2].bar(x, crate_fracs, color="mediumpurple")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(case_labels, rotation=45, ha="right")
    axes[2].set_ylabel("Crate diag / Ctotal diag")
    axes[2].set_title("FEM Variance Fraction")

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "h1_pipeline_bias.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()
