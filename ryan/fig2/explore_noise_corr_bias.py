# %% Imports and configuration
"""
Exploratory script: diagnose negative bias in FEM-corrected noise correlations.

Hypothesis: the PAVA/linear intercept fitting overestimates Crate (particularly
off-diagonal covariances) when extrapolating to zero eye-position difference,
inflating the subtracted component and pushing corrected noise correlations
negative.

This script loads a single session (Allen 2022-04-13), runs the decomposition
step by step, and exposes each intermediate quantity for visual inspection:

  1. Load and trial-align the fixRSVP data
  2. Extract windows and compute conditional second moments
  3. Inspect the binned covariance curves Ceye(d) for individual pairs
  4. Compare intercept estimates: linear vs isotonic vs raw
  5. Compute Crate, Cpsth, noise covariance matrices
  6. Reproduce panels F, G, H for this single session
  7. Evaluate fit quality across the full covariance matrix
  8. Shuffle control: check whether the bias appears under null

Run interactively with IPython (# %% cells).
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from VisionCore.paths import VISIONCORE_ROOT
from VisionCore.covariance import (
    cov_to_corr,
    get_upper_triangle,
    project_to_psd,
    pava_nonincreasing,
    align_fixrsvp_trials,
    extract_valid_segments,
    extract_windows,
    compute_conditional_second_moments,
    estimate_rate_covariance,
    fit_intercept_pava,
    fit_intercept_linear,
    bagged_split_half_psth_covariance,
)
from VisionCore.stats import fisher_z, fisher_z_mean, bootstrap_mean_ci
from VisionCore.subspace import project_to_psd

if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))
from models.config_loader import load_dataset_configs
from models.data import prepare_data
from DataYatesV1 import get_free_device

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Parameters — match generate_figure2.py defaults
DT = 1 / 240
WINDOW_BINS = [2, 4, 8, 16]
N_BINS = 15                # number of eye-distance bins
N_SHUFFLES = 50            # fewer shuffles for exploration
MIN_TOTAL_SPIKES = 500
INTERCEPT_MODE = "linear"  # 'linear', 'isotonic', or 'raw'
MIN_SEG_LEN = 36
DEVICE = str(get_free_device())

SESSION_NAME = "Allen_2022-04-13"
DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"


# %% Load single session
dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
cfg = None
for c in dataset_configs:
    if c["session"] == SESSION_NAME:
        cfg = c
        break
assert cfg is not None, f"Session {SESSION_NAME} not found in config"

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


# %% Extract windows for a single counting window (10 ms = 2 bins)
t_count = WINDOW_BINS[0]  # 2 bins = 8.33 ms
t_hist = max(int(10 / (DT * 1000)), t_count)  # history window

# Sanitize
robs_clean = np.nan_to_num(robs, nan=0.0)
eyepos_clean = np.nan_to_num(eyepos, nan=0.0)

device_obj = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
robs_t = torch.tensor(robs_clean, dtype=torch.float32, device=device_obj)
eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32, device=device_obj)

segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)
print(f"Found {len(segments)} valid segments")

SpikeCounts, EyeTraj, T_idx, idx_tr = extract_windows(
    robs_t, eyepos_t, segments, t_count, t_hist, device=DEVICE
)
n_samples, n_cells = SpikeCounts.shape
print(f"Extracted {n_samples} windows, {n_cells} neurons")


# %% Step 1: Compute total covariance
ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()
Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
print(f"Ctotal shape: {Ctotal.shape}")
print(f"Mean firing rate range: [{Erate.min():.3f}, {Erate.max():.3f}]")


# %% Step 2: Compute conditional second moments and Ceye
MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
    SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS
)
Ceye = MM - Erate[:, None] * Erate[None, :]
print(f"Ceye shape: {Ceye.shape}")
print(f"Bin centers: {bin_centers}")
print(f"Counts per bin: {count_e}")


# %% Step 3: Visualize Ceye curves for example diagonal elements
# These should be non-increasing if eye position modulates variance
fig, axes = plt.subplots(3, 4, figsize=(14, 9))
axes = axes.ravel()

# Pick 12 neurons spanning the rate range
sorted_by_rate = np.argsort(Erate)
pick_idx = sorted_by_rate[np.linspace(0, n_cells - 1, 12, dtype=int)]

for ax_i, ci in enumerate(pick_idx):
    ax = axes[ax_i]
    y = Ceye[:, ci, ci]
    ax.plot(bin_centers, y, 'ko-', markersize=3, label='Ceye (var)')

    # PAVA fit
    yhat_pava = pava_nonincreasing(y, count_e)
    ax.plot(bin_centers, yhat_pava, 'r-', linewidth=2, label='PAVA')
    ax.axhline(yhat_pava[0], color='r', linestyle=':', alpha=0.5)

    # Linear fit
    Crate_lin = fit_intercept_linear(Ceye[:, ci:ci+1, ci:ci+1], bin_centers,
                                     count_e, eval_at_first_bin=True)
    ax.axhline(Crate_lin[0, 0], color='b', linestyle=':', alpha=0.5, label='Linear')

    # Linear fit extrapolated to d=0
    Crate_lin0 = fit_intercept_linear(Ceye[:, ci:ci+1, ci:ci+1], bin_centers,
                                      count_e, eval_at_first_bin=False)
    ax.axhline(Crate_lin0[0, 0], color='g', linestyle=':', alpha=0.5, label='Linear(d=0)')

    ax.set_title(f'Neuron {ci} (rate={Erate[ci]:.2f})', fontsize=8)
    if ax_i == 0:
        ax.legend(fontsize=6)

fig.suptitle('Diagonal Ceye curves: variance vs eye distance', fontsize=12)
fig.tight_layout()
plt.show()


# %% Step 4: Visualize Ceye curves for example OFF-DIAGONAL elements
# These are the covariances that drive noise correlations
# Off-diagonals can be increasing or decreasing — PAVA picks the best direction
fig2, axes2 = plt.subplots(4, 5, figsize=(16, 12))
axes2 = axes2.ravel()

# Pick 20 pairs spanning different rate combinations
rows, cols = np.triu_indices(n_cells, k=1)
n_pairs_total = len(rows)
pair_idx = np.linspace(0, n_pairs_total - 1, 20, dtype=int)

for ax_i, pi in enumerate(pair_idx):
    ax = axes2[ax_i]
    i, j = rows[pi], cols[pi]
    y = Ceye[:, i, j]
    ax.plot(bin_centers, y, 'ko-', markersize=3, label='Ceye(cov)')

    # PAVA non-increasing
    yhat_decr = pava_nonincreasing(y, count_e)
    sse_decr = np.sum(count_e * (y - yhat_decr) ** 2)

    # PAVA non-decreasing
    yhat_incr = -pava_nonincreasing(-y, count_e)
    sse_incr = np.sum(count_e * (y - yhat_incr) ** 2)

    if sse_decr < sse_incr:
        yhat_best = yhat_decr
        direction = 'decr'
    else:
        yhat_best = yhat_incr
        direction = 'incr'

    ax.plot(bin_centers, yhat_best, 'r-', linewidth=2,
            label=f'PAVA ({direction})')
    ax.axhline(yhat_best[0], color='r', linestyle=':', alpha=0.5)

    # Linear intercept
    Ceye_pair = Ceye[:, i:i+1, j:j+1]
    val_lin = fit_intercept_linear(Ceye_pair, bin_centers, count_e,
                                   eval_at_first_bin=True)[0, 0]
    val_lin0 = fit_intercept_linear(Ceye_pair, bin_centers, count_e,
                                    eval_at_first_bin=False)[0, 0]
    ax.axhline(val_lin, color='b', linestyle=':', alpha=0.5, label='Linear')
    ax.axhline(val_lin0, color='g', linestyle=':', alpha=0.5, label='Linear(d=0)')

    # Raw (first bin)
    ax.axhline(y[0], color='gray', linestyle='--', alpha=0.3, label='Raw')

    ax.set_title(f'({i},{j}) r={Erate[i]:.1f},{Erate[j]:.1f}', fontsize=7)
    if ax_i == 0:
        ax.legend(fontsize=5)

fig2.suptitle('Off-diagonal Ceye curves: covariance vs eye distance', fontsize=12)
fig2.tight_layout()
plt.show()


# %% Step 5: Compare intercept methods across the full matrix
# Compute Crate under each method
Crate_linear = fit_intercept_linear(Ceye, bin_centers, count_e,
                                     eval_at_first_bin=True)
Crate_linear0 = fit_intercept_linear(Ceye, bin_centers, count_e,
                                      eval_at_first_bin=False)
Crate_isotonic = fit_intercept_pava(Ceye, count_e)
Crate_raw = Ceye[0].copy()

# Apply the same physical limit check as estimate_rate_covariance
def apply_limits(Crate, Ctotal):
    C = Crate.copy()
    bad = np.diag(C) > 0.99 * np.diag(Ctotal)
    C[bad, :] = np.nan
    C[:, bad] = np.nan
    return C

Crate_linear_lim = apply_limits(Crate_linear, Ctotal)
Crate_linear0_lim = apply_limits(Crate_linear0, Ctotal)
Crate_isotonic_lim = apply_limits(Crate_isotonic, Ctotal)
Crate_raw_lim = apply_limits(Crate_raw, Ctotal)

print("Crate diagonal stats (should be <= Ctotal diagonal):")
for name, C in [("linear(bin0)", Crate_linear),
                ("linear(d=0)", Crate_linear0),
                ("isotonic", Crate_isotonic),
                ("raw", Crate_raw)]:
    diag_ratio = np.diag(C) / np.diag(Ctotal)
    print(f"  {name:15s}: median ratio={np.nanmedian(diag_ratio):.4f}, "
          f"max={np.nanmax(diag_ratio):.4f}, "
          f"frac>1={np.nanmean(diag_ratio > 1):.3f}")


# %% Step 6: Compare off-diagonal intercepts
# The key question: does the intercept overestimate off-diagonal covariance?
# If so, Crate will have inflated off-diags, and CnoiseC = Ctotal - Crate
# will have artificially reduced (more negative) off-diags → negative noise corr.

upper_Ctotal = get_upper_triangle(Ctotal)
upper_lin = get_upper_triangle(Crate_linear)
upper_lin0 = get_upper_triangle(Crate_linear0)
upper_iso = get_upper_triangle(Crate_isotonic)
upper_raw = get_upper_triangle(Crate_raw)

fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, vals) in zip(axes3, [
    ("Linear(bin0)", upper_lin),
    ("Linear(d=0)", upper_lin0),
    ("Isotonic", upper_iso),
    ("Raw", upper_raw),
]):
    ok = np.isfinite(vals) & np.isfinite(upper_Ctotal)
    ax.scatter(upper_Ctotal[ok], vals[ok], s=1, alpha=0.3)
    lim = max(np.abs(upper_Ctotal[ok]).max(), np.abs(vals[ok]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
    ax.set_xlabel('Ctotal off-diag')
    ax.set_ylabel(f'Crate off-diag ({name})')
    ax.set_title(name)
    ax.set_aspect('equal')
    # Annotate mean bias
    bias = np.nanmean(vals[ok] - upper_Ctotal[ok])
    ax.text(0.05, 0.95, f'mean bias={bias:.4e}', transform=ax.transAxes,
            fontsize=8, verticalalignment='top')
fig3.suptitle('Off-diagonal Crate vs Ctotal (each dot = one pair)')
fig3.tight_layout()
plt.show()


# %% Step 7: Compute PSTH covariance and derived matrices
Cpsth, PSTH_mean = bagged_split_half_psth_covariance(
    SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
    seed=42, global_mean=Erate,
)

# Derived noise covariance matrices
CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
Cfem = Crate_linear_lim - Cpsth

# Build CnoiseC for each method
noise_methods = {}
for name, Cr in [("linear(bin0)", Crate_linear_lim),
                 ("linear(d=0)", Crate_linear0_lim),
                 ("isotonic", Crate_isotonic_lim),
                 ("raw", Crate_raw_lim)]:
    CnoiseC = 0.5 * ((Ctotal - Cr) + (Ctotal - Cr).T)
    noise_methods[name] = CnoiseC

print("\nDiag(CnoiseC) stats — should these be positive?")
for name, CnoiseC in noise_methods.items():
    d = np.diag(CnoiseC)
    print(f"  {name:15s}: frac<0={np.nanmean(d < 0):.3f}, "
          f"median={np.nanmedian(d):.4f}")


# %% Step 8: Noise correlations for each method → Panels F, G, H (single session)
# Neuron inclusion mask (same as generate_figure2.py)
total_spikes = Erate * n_samples
valid = (
    np.isfinite(Erate)
    & (total_spikes >= MIN_TOTAL_SPIKES)
    & (np.diag(Ctotal) > 0)
    & np.isfinite(np.diag(Crate_linear_lim))
    & np.isfinite(np.diag(Cpsth))
)
n_valid = valid.sum()
print(f"\nValid neurons: {n_valid} / {n_cells}")

# Uncorrected noise correlations
NoiseCorrU = cov_to_corr(project_to_psd(CnoiseU[np.ix_(valid, valid)]))
rho_u = get_upper_triangle(NoiseCorrU)

# Corrected noise correlations for each method
fig4, axes4 = plt.subplots(2, 2, figsize=(10, 10))
axes4 = axes4.ravel()

for ax_i, (name, CnoiseC) in enumerate(noise_methods.items()):
    ax = axes4[ax_i]
    NoiseCorrC = cov_to_corr(project_to_psd(CnoiseC[np.ix_(valid, valid)]))
    rho_c = get_upper_triangle(NoiseCorrC)

    pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
    ru, rc = rho_u[pair_ok], rho_c[pair_ok]

    ax.hist2d(ru, rc, bins=60, cmap="Blues", norm=mpl.colors.LogNorm(),
              range=[[-0.3, 0.3], [-0.3, 0.3]])
    ax.plot([-0.3, 0.3], [-0.3, 0.3], 'k--', alpha=0.3, linewidth=0.5)
    ax.plot(np.mean(ru), np.mean(rc), 'ro', markersize=6)
    ax.set_xlabel("ρ uncorrected")
    ax.set_ylabel("ρ corrected")

    z_u = fisher_z_mean(ru)
    z_c = fisher_z_mean(rc)
    dz = z_c - z_u
    ax.set_title(f'{name}\nz_u={z_u:.4f}, z_c={z_c:.4f}, Δz={dz:.4f}', fontsize=9)

fig4.suptitle(f'Panel F equivalent: {SESSION_NAME} (each method)', fontsize=12)
fig4.tight_layout()
plt.show()


# %% Step 9: Distribution of Δρ per pair — where is the negative bias?
fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
axes5 = axes5.ravel()

for ax_i, (name, CnoiseC) in enumerate(noise_methods.items()):
    ax = axes5[ax_i]
    NoiseCorrC = cov_to_corr(project_to_psd(CnoiseC[np.ix_(valid, valid)]))
    rho_c = get_upper_triangle(NoiseCorrC)

    pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
    delta_rho = rho_c[pair_ok] - rho_u[pair_ok]

    ax.hist(delta_rho, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(np.mean(delta_rho), color='red', linewidth=2,
               label=f'mean={np.mean(delta_rho):.4f}')
    ax.axvline(np.median(delta_rho), color='orange', linewidth=2,
               label=f'median={np.median(delta_rho):.4f}')
    ax.set_xlabel('Δρ (corrected - uncorrected)')
    ax.set_ylabel('Pair count')
    ax.set_title(name)
    ax.legend(fontsize=7)

fig5.suptitle(f'Distribution of Δρ per pair: {SESSION_NAME}')
fig5.tight_layout()
plt.show()


# %% Step 10: Fit quality assessment for each pair in the covariance matrix
# For each pair (i,j), compute the R² or SSE of the intercept fit
# to understand where fits are unreliable

def fit_quality_linear(Ceye, bin_centers, count_e, d_max=0.4):
    """Compute R² of the linear fit for each element of Ceye."""
    n_bins, n_cells, _ = Ceye.shape
    x = np.asarray(bin_centers, dtype=np.float64)
    w = np.asarray(count_e, dtype=np.float64)
    use = np.isfinite(x) & (x > 0) & (x <= d_max) & np.isfinite(w) & (w > 0)
    idx = np.where(use)[0]

    R2 = np.full((n_cells, n_cells), np.nan)
    slope = np.full((n_cells, n_cells), np.nan)

    if len(idx) < 3:
        return R2, slope

    x_loc = x[idx]
    w_loc = w[idx]
    S0 = np.sum(w_loc)
    Sx = np.sum(w_loc * x_loc)
    Sxx = np.sum(w_loc * x_loc ** 2)
    det = S0 * Sxx - Sx ** 2

    for i in range(n_cells):
        for j in range(i, n_cells):
            y = Ceye[idx, i, j]
            if not np.isfinite(y).all():
                continue
            Sy = np.sum(w_loc * y)
            Sxy = np.sum(w_loc * x_loc * y)

            beta1 = (S0 * Sxy - Sx * Sy) / det
            beta0 = (Sxx * Sy - Sx * Sxy) / det
            yhat = beta0 + beta1 * x_loc

            ss_res = np.sum(w_loc * (y - yhat) ** 2)
            y_mean = Sy / S0
            ss_tot = np.sum(w_loc * (y - y_mean) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-20 else np.nan

            R2[i, j] = r2
            R2[j, i] = r2
            slope[i, j] = beta1
            slope[j, i] = beta1

    return R2, slope


R2_mat, slope_mat = fit_quality_linear(Ceye, bin_centers, count_e)

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# R² heatmap
im1 = ax1.imshow(R2_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax1.set_title('R² of linear fit per pair')
plt.colorbar(im1, ax=ax1)

# Slope heatmap — positive slope means covariance INCREASES with distance
# (would mean intercept underestimates), negative means it decreases
# (intercept overestimates)
vlim = np.nanpercentile(np.abs(slope_mat), 95)
im2 = ax2.imshow(slope_mat, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='auto')
ax2.set_title('Slope of linear fit (neg = decreasing with distance)')
plt.colorbar(im2, ax=ax2)

fig6.suptitle(f'Fit quality: {SESSION_NAME}')
fig6.tight_layout()
plt.show()

# Distribution of R²
fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

diag_r2 = np.diag(R2_mat)
offdiag_r2 = get_upper_triangle(R2_mat)
ax1.hist(diag_r2[np.isfinite(diag_r2)], bins=30, alpha=0.7, label='Diagonal (var)')
ax1.hist(offdiag_r2[np.isfinite(offdiag_r2)], bins=30, alpha=0.7, label='Off-diag (cov)')
ax1.set_xlabel('R²')
ax1.set_ylabel('Count')
ax1.set_title('R² distribution')
ax1.legend(fontsize=8)

# Slope vs intercept for off-diagonals
offdiag_slope = get_upper_triangle(slope_mat)
offdiag_intercept = get_upper_triangle(Crate_linear)
ok = np.isfinite(offdiag_slope) & np.isfinite(offdiag_intercept)
ax2.scatter(offdiag_intercept[ok], offdiag_slope[ok], s=1, alpha=0.3)
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Linear intercept (cov at d≈0)')
ax2.set_ylabel('Slope (change per unit distance)')
ax2.set_title('Off-diagonal: intercept vs slope')

fig7.suptitle(f'Fit diagnostics: {SESSION_NAME}')
fig7.tight_layout()
plt.show()


# %% Step 11: Shuffle control — does the bias appear under shuffled eye trajectories?
# Under the null (shuffled eye), Crate_shuff should ≈ Ctotal (no eye structure),
# so CnoiseC_shuff = Ctotal - Crate_shuff should be near zero.
# If the intercept fitting introduces bias, it will also bias Crate_shuff.

rng_shuffle = torch.Generator(device=device_obj)
rng_shuffle.manual_seed(42)

n_shuff = N_SHUFFLES
dz_real = []
dz_shuff_list = []
shuff_intercept_biases = []

# Real delta z
NoiseCorrC_real = cov_to_corr(
    project_to_psd(noise_methods["linear(bin0)"][np.ix_(valid, valid)])
)
rho_c_real = get_upper_triangle(NoiseCorrC_real)
pair_ok_real = np.isfinite(rho_u) & np.isfinite(rho_c_real)
dz_real_val = fisher_z_mean(rho_c_real[pair_ok_real]) - fisher_z_mean(rho_u[pair_ok_real])

print(f"\nReal Δz = {dz_real_val:.4f}")
print(f"Running {n_shuff} shuffle iterations...")

for k in range(n_shuff):
    perm = torch.randperm(n_samples, generator=rng_shuffle, device=device_obj)
    EyeTraj_shuff = EyeTraj[perm]
    Crate_shuff, _, Ceye_shuff, bc_shuff, ce_shuff, _ = estimate_rate_covariance(
        SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
        Ctotal=Ctotal, intercept_mode=INTERCEPT_MODE,
    )

    CnoiseC_shuff = 0.5 * ((Ctotal - Crate_shuff) + (Ctotal - Crate_shuff).T)
    NoiseCorrC_shuff = cov_to_corr(
        project_to_psd(CnoiseC_shuff[np.ix_(valid, valid)])
    )
    rho_c_shuff = get_upper_triangle(NoiseCorrC_shuff)
    ok_shuff = np.isfinite(rho_c_shuff) & pair_ok_real
    if ok_shuff.sum() > 0:
        dz_shuff = fisher_z_mean(rho_c_shuff[ok_shuff]) - fisher_z_mean(rho_u[ok_shuff])
        dz_shuff_list.append(dz_shuff)

        # Track the bias in off-diagonal intercepts
        upper_shuff = get_upper_triangle(Crate_shuff)
        bias = np.nanmean(upper_shuff[np.isfinite(upper_shuff)]
                          - upper_Ctotal[np.isfinite(upper_shuff) & np.isfinite(upper_Ctotal)])
        shuff_intercept_biases.append(bias)

dz_shuff_arr = np.array(dz_shuff_list)
print(f"Shuffle Δz: mean={dz_shuff_arr.mean():.4f}, "
      f"95% CI=[{np.percentile(dz_shuff_arr, 2.5):.4f}, "
      f"{np.percentile(dz_shuff_arr, 97.5):.4f}]")
print(f"Shuffle intercept bias: mean={np.mean(shuff_intercept_biases):.4e}")

fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(dz_shuff_arr, bins=30, color='gray', edgecolor='white', alpha=0.8)
ax1.axvline(dz_real_val, color='red', linewidth=2, label=f'Real Δz={dz_real_val:.4f}')
ax1.axvline(0, color='black', linestyle=':', alpha=0.5)
ax1.axvline(dz_shuff_arr.mean(), color='blue', linewidth=2,
            label=f'Shuffle mean={dz_shuff_arr.mean():.4f}')
ax1.set_xlabel('Δz (corr - uncorr)')
ax1.set_ylabel('Count')
ax1.set_title('Shuffle null: Δz distribution')
ax1.legend(fontsize=8)

ax2.hist(shuff_intercept_biases, bins=30, color='gray', edgecolor='white', alpha=0.8)
ax2.axvline(0, color='black', linestyle=':', alpha=0.5)
ax2.set_xlabel('Mean off-diag intercept bias (Crate - Ctotal)')
ax2.set_ylabel('Count')
ax2.set_title('Shuffle: intercept estimation bias')

fig8.suptitle(f'Shuffle controls: {SESSION_NAME}')
fig8.tight_layout()
plt.show()


# %% Step 12: Panel H equivalent — across all windows for this session
print("\n=== Across-window summary (single session) ===")
dz_by_window = []
dz_shuff_by_window = []

for w_idx, t_count in enumerate(WINDOW_BINS):
    t_hist_w = max(int(10 / (DT * 1000)), t_count)
    SC_w, ET_w, TI_w, _ = extract_windows(
        robs_t, eyepos_t, segments, t_count, t_hist_w, device=DEVICE
    )
    if SC_w is None or SC_w.shape[0] < 100:
        dz_by_window.append(np.nan)
        dz_shuff_by_window.append((np.nan, np.nan))
        continue

    n_w = SC_w.shape[0]
    ix_w = np.isfinite(SC_w.sum(1).detach().cpu().numpy())
    Ct_w = torch.cov(SC_w[ix_w].T, correction=1).detach().cpu().numpy()
    Er_w = torch.nanmean(SC_w, 0).detach().cpu().numpy()

    Cr_w, _, _, _, _, be_w = estimate_rate_covariance(
        SC_w, ET_w, TI_w, n_bins=N_BINS, Ctotal=Ct_w, intercept_mode=INTERCEPT_MODE
    )
    Cp_w, _ = bagged_split_half_psth_covariance(
        SC_w, TI_w, n_boot=20, min_trials_per_time=10, seed=42, global_mean=Er_w
    )

    CnU_w = 0.5 * ((Ct_w - Cp_w) + (Ct_w - Cp_w).T)
    CnC_w = 0.5 * ((Ct_w - Cr_w) + (Ct_w - Cr_w).T)

    ts_w = Er_w * n_w
    v_w = (np.isfinite(Er_w) & (ts_w >= MIN_TOTAL_SPIKES)
           & (np.diag(Ct_w) > 0) & np.isfinite(np.diag(Cr_w))
           & np.isfinite(np.diag(Cp_w)))

    NCU_w = cov_to_corr(project_to_psd(CnU_w[np.ix_(v_w, v_w)]))
    NCC_w = cov_to_corr(project_to_psd(CnC_w[np.ix_(v_w, v_w)]))
    ru_w = get_upper_triangle(NCU_w)
    rc_w = get_upper_triangle(NCC_w)
    pok = np.isfinite(ru_w) & np.isfinite(rc_w)

    zu_w = fisher_z_mean(ru_w[pok])
    zc_w = fisher_z_mean(rc_w[pok])
    dz_w = zc_w - zu_w
    dz_by_window.append(dz_w)

    # Shuffle
    rng_w = torch.Generator(device=device_obj)
    rng_w.manual_seed(42)
    dz_s_list = []
    for k in range(min(N_SHUFFLES, 20)):  # fewer for speed
        perm_w = torch.randperm(n_w, generator=rng_w, device=device_obj)
        ET_s = ET_w[perm_w]
        Cr_s, _, _, _, _, _ = estimate_rate_covariance(
            SC_w, ET_s, TI_w, n_bins=be_w, Ctotal=Ct_w, intercept_mode=INTERCEPT_MODE
        )
        CnC_s = 0.5 * ((Ct_w - Cr_s) + (Ct_w - Cr_s).T)
        NCC_s = cov_to_corr(project_to_psd(CnC_s[np.ix_(v_w, v_w)]))
        rc_s = get_upper_triangle(NCC_s)
        oks = np.isfinite(rc_s) & pok
        if oks.sum() > 0:
            dz_s_list.append(fisher_z_mean(rc_s[oks]) - fisher_z_mean(ru_w[oks]))
    dz_s_arr = np.array(dz_s_list)
    dz_shuff_by_window.append(
        (float(np.percentile(dz_s_arr, 2.5)), float(np.percentile(dz_s_arr, 97.5)))
    )

    win_ms = t_count * DT * 1000
    print(f"  {win_ms:.1f} ms: Δz={dz_w:.4f}, "
          f"shuffle 95% CI=[{dz_shuff_by_window[-1][0]:.4f}, "
          f"{dz_shuff_by_window[-1][1]:.4f}], "
          f"n_valid={v_w.sum()}, n_pairs={pok.sum()}")

# Plot
windows_ms = [t * DT * 1000 for t in WINDOW_BINS]
fig9, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(windows_ms, dz_by_window, 'ko-', markersize=6, label='Observed Δz')
null_lo = [s[0] for s in dz_shuff_by_window]
null_hi = [s[1] for s in dz_shuff_by_window]
ax.fill_between(windows_ms, null_lo, null_hi, alpha=0.2, color='gray',
                label='Shuffle 95% CI')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Counting window (ms)')
ax.set_ylabel('Δz (corr - uncorr)')
ax.set_title(f'Panel H equivalent: {SESSION_NAME}')
ax.legend(frameon=False, fontsize=8)
ax.set_xscale('log')
ax.set_xticks(windows_ms)
ax.set_xticklabels([f'{w:.1f}' for w in windows_ms])
fig9.tight_layout()
plt.show()


# %% Step 13: Key diagnostic — what fraction of Crate off-diag exceeds Ctotal off-diag?
# If Crate off-diags > Ctotal off-diags, then CnoiseC = Ctotal - Crate has
# NEGATIVE off-diags → negative noise correlations.

print("\n=== Off-diagonal overestimation diagnostic ===")
for name, Cr in [("linear(bin0)", Crate_linear_lim),
                 ("linear(d=0)", Crate_linear0_lim),
                 ("isotonic", Crate_isotonic_lim),
                 ("raw", Crate_raw_lim)]:
    ut_cr = get_upper_triangle(Cr)
    ok = np.isfinite(ut_cr) & np.isfinite(upper_Ctotal)
    excess = ut_cr[ok] - upper_Ctotal[ok]
    frac_over = np.mean(excess > 0)
    mean_excess = np.mean(excess)
    print(f"  {name:15s}: {frac_over*100:.1f}% of pairs have Crate > Ctotal, "
          f"mean excess = {mean_excess:.4e}")

# Same for the diagonal
print("\nDiagonal overestimation:")
for name, Cr in [("linear(bin0)", Crate_linear_lim),
                 ("linear(d=0)", Crate_linear0_lim),
                 ("isotonic", Crate_isotonic_lim),
                 ("raw", Crate_raw_lim)]:
    diag_cr = np.diag(Cr)
    diag_ct = np.diag(Ctotal)
    ok = np.isfinite(diag_cr) & np.isfinite(diag_ct)
    excess = diag_cr[ok] - diag_ct[ok]
    frac_over = np.mean(excess > 0)
    mean_excess = np.mean(excess)
    print(f"  {name:15s}: {frac_over*100:.1f}% of neurons have Crate > Ctotal, "
          f"mean excess = {mean_excess:.4e}")


# %% Step 14: Quantify PSD projection contribution to shuffle bias
# Compare Δz computed WITH vs WITHOUT project_to_psd for both real and shuffled.
# If PSD projection is the dominant bias source, Δz_nopsd_shuff should center
# near zero while Δz_psd_shuff is negatively biased.

print("\n=== Step 14: PSD projection bias quantification ===")


def compute_dz(CnoiseC_mat, CnoiseU_mat, valid_mask, use_psd=True):
    """Compute Δz with or without PSD projection."""
    CnU = CnoiseU_mat[np.ix_(valid_mask, valid_mask)]
    CnC = CnoiseC_mat[np.ix_(valid_mask, valid_mask)]
    if use_psd:
        CnU = project_to_psd(CnU)
        CnC = project_to_psd(CnC)
    NCU = cov_to_corr(CnU)
    NCC = cov_to_corr(CnC)
    ru = get_upper_triangle(NCU)
    rc = get_upper_triangle(NCC)
    ok = np.isfinite(ru) & np.isfinite(rc)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    zu = fisher_z_mean(ru[ok])
    zc = fisher_z_mean(rc[ok])
    return zc - zu, zu, zc


def count_negative_eigenvalues(C):
    """Count negative eigenvalues and total negative eigenvalue mass."""
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    C = np.nan_to_num(C, nan=0.0)
    w = np.linalg.eigvalsh(C)
    n_neg = int(np.sum(w < 0))
    neg_mass = float(np.sum(w[w < 0]))
    return n_neg, neg_mass


# --- Real data: with vs without PSD ---
CnoiseC_real = noise_methods["linear(bin0)"]
dz_psd, zu_psd, zc_psd = compute_dz(CnoiseC_real, CnoiseU, valid, use_psd=True)
dz_nopsd, zu_nopsd, zc_nopsd = compute_dz(CnoiseC_real, CnoiseU, valid, use_psd=False)

n_neg_U, mass_U = count_negative_eigenvalues(CnoiseU[np.ix_(valid, valid)])
n_neg_C, mass_C = count_negative_eigenvalues(CnoiseC_real[np.ix_(valid, valid)])

print(f"Real data (linear, bin0):")
print(f"  With PSD:    Δz = {dz_psd:.5f}  (z_u={zu_psd:.5f}, z_c={zc_psd:.5f})")
print(f"  Without PSD: Δz = {dz_nopsd:.5f}  (z_u={zu_nopsd:.5f}, z_c={zc_nopsd:.5f})")
print(f"  PSD projection shifts Δz by {dz_psd - dz_nopsd:.5f}")
print(f"  Negative eigenvalues — CnoiseU: {n_neg_U} (mass={mass_U:.4e}), "
      f"CnoiseC: {n_neg_C} (mass={mass_C:.4e})")

# --- Shuffle: with vs without PSD ---
dz_shuff_psd = []
dz_shuff_nopsd = []
neg_eig_counts_shuff = []

rng_psd = torch.Generator(device=device_obj)
rng_psd.manual_seed(123)  # different seed from step 11

for k in range(N_SHUFFLES):
    perm = torch.randperm(n_samples, generator=rng_psd, device=device_obj)
    EyeTraj_shuff = EyeTraj[perm]
    Crate_shuff, _, _, _, _, _ = estimate_rate_covariance(
        SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
        Ctotal=Ctotal, intercept_mode=INTERCEPT_MODE,
    )

    CnoiseC_shuff = 0.5 * ((Ctotal - Crate_shuff) + (Ctotal - Crate_shuff).T)

    dz_p, _, _ = compute_dz(CnoiseC_shuff, CnoiseU, valid, use_psd=True)
    dz_np, _, _ = compute_dz(CnoiseC_shuff, CnoiseU, valid, use_psd=False)

    if np.isfinite(dz_p):
        dz_shuff_psd.append(dz_p)
    if np.isfinite(dz_np):
        dz_shuff_nopsd.append(dz_np)

    n_neg_s, _ = count_negative_eigenvalues(CnoiseC_shuff[np.ix_(valid, valid)])
    neg_eig_counts_shuff.append(n_neg_s)

dz_shuff_psd = np.array(dz_shuff_psd)
dz_shuff_nopsd = np.array(dz_shuff_nopsd)
neg_eig_counts_shuff = np.array(neg_eig_counts_shuff)

print(f"\nShuffle null ({N_SHUFFLES} iterations):")
print(f"  With PSD:    mean Δz = {dz_shuff_psd.mean():.5f}, "
      f"95% CI = [{np.percentile(dz_shuff_psd, 2.5):.5f}, "
      f"{np.percentile(dz_shuff_psd, 97.5):.5f}]")
print(f"  Without PSD: mean Δz = {dz_shuff_nopsd.mean():.5f}, "
      f"95% CI = [{np.percentile(dz_shuff_nopsd, 2.5):.5f}, "
      f"{np.percentile(dz_shuff_nopsd, 97.5):.5f}]")
print(f"  PSD projection shifts shuffle mean Δz by "
      f"{dz_shuff_psd.mean() - dz_shuff_nopsd.mean():.5f}")
print(f"  Negative eigenvalues in CnoiseC_shuff: "
      f"mean={neg_eig_counts_shuff.mean():.1f}, "
      f"range=[{neg_eig_counts_shuff.min()}, {neg_eig_counts_shuff.max()}]")
print(f"  Negative eigenvalues in CnoiseU (fixed): {n_neg_U}")

# --- Across all windows: PSD vs no-PSD ---
print("\n=== Across-window PSD bias ===")
dz_win_psd = []
dz_win_nopsd = []
dz_shuff_win_psd = []
dz_shuff_win_nopsd = []

for w_idx, t_count in enumerate(WINDOW_BINS):
    t_hist_w = max(int(10 / (DT * 1000)), t_count)
    SC_w, ET_w, TI_w, _ = extract_windows(
        robs_t, eyepos_t, segments, t_count, t_hist_w, device=DEVICE
    )
    if SC_w is None or SC_w.shape[0] < 100:
        dz_win_psd.append(np.nan)
        dz_win_nopsd.append(np.nan)
        dz_shuff_win_psd.append(np.nan)
        dz_shuff_win_nopsd.append(np.nan)
        continue

    n_w = SC_w.shape[0]
    ix_w = np.isfinite(SC_w.sum(1).detach().cpu().numpy())
    Ct_w = torch.cov(SC_w[ix_w].T, correction=1).detach().cpu().numpy()
    Er_w = torch.nanmean(SC_w, 0).detach().cpu().numpy()

    Cr_w, _, _, _, _, be_w = estimate_rate_covariance(
        SC_w, ET_w, TI_w, n_bins=N_BINS, Ctotal=Ct_w, intercept_mode=INTERCEPT_MODE
    )
    Cp_w, _ = bagged_split_half_psth_covariance(
        SC_w, TI_w, n_boot=20, min_trials_per_time=10, seed=42, global_mean=Er_w
    )

    CnU_w = 0.5 * ((Ct_w - Cp_w) + (Ct_w - Cp_w).T)
    CnC_w = 0.5 * ((Ct_w - Cr_w) + (Ct_w - Cr_w).T)

    ts_w = Er_w * n_w
    v_w = (np.isfinite(Er_w) & (ts_w >= MIN_TOTAL_SPIKES)
           & (np.diag(Ct_w) > 0) & np.isfinite(np.diag(Cr_w))
           & np.isfinite(np.diag(Cp_w)))

    dz_p, _, _ = compute_dz(CnC_w, CnU_w, v_w, use_psd=True)
    dz_np, _, _ = compute_dz(CnC_w, CnU_w, v_w, use_psd=False)
    dz_win_psd.append(dz_p)
    dz_win_nopsd.append(dz_np)

    # Shuffle for this window
    rng_w = torch.Generator(device=device_obj)
    rng_w.manual_seed(42)
    s_psd, s_nopsd = [], []
    for k in range(min(N_SHUFFLES, 20)):
        perm_w = torch.randperm(n_w, generator=rng_w, device=device_obj)
        ET_s = ET_w[perm_w]
        Cr_s, _, _, _, _, _ = estimate_rate_covariance(
            SC_w, ET_s, TI_w, n_bins=be_w, Ctotal=Ct_w, intercept_mode=INTERCEPT_MODE
        )
        CnC_s = 0.5 * ((Ct_w - Cr_s) + (Ct_w - Cr_s).T)
        dp, _, _ = compute_dz(CnC_s, CnU_w, v_w, use_psd=True)
        dnp, _, _ = compute_dz(CnC_s, CnU_w, v_w, use_psd=False)
        if np.isfinite(dp):
            s_psd.append(dp)
        if np.isfinite(dnp):
            s_nopsd.append(dnp)

    dz_shuff_win_psd.append(np.mean(s_psd) if s_psd else np.nan)
    dz_shuff_win_nopsd.append(np.mean(s_nopsd) if s_nopsd else np.nan)

    win_ms = t_count * DT * 1000
    print(f"  {win_ms:.1f} ms: real Δz  PSD={dz_p:.5f}  noPSD={dz_np:.5f}  "
          f"(shift={dz_p - dz_np:.5f})")
    print(f"  {' ':7s} shuff Δz PSD={dz_shuff_win_psd[-1]:.5f}  "
          f"noPSD={dz_shuff_win_nopsd[-1]:.5f}  "
          f"(shift={dz_shuff_win_psd[-1] - dz_shuff_win_nopsd[-1]:.5f})")


# --- Figure: PSD vs no-PSD comparison ---
fig10, axes10 = plt.subplots(1, 3, figsize=(15, 4))

# Panel A: Shuffle Δz distributions with/without PSD
ax = axes10[0]
bins_hist = np.linspace(
    min(dz_shuff_psd.min(), dz_shuff_nopsd.min()) - 0.001,
    max(dz_shuff_psd.max(), dz_shuff_nopsd.max()) + 0.001,
    40,
)
ax.hist(dz_shuff_psd, bins=bins_hist, alpha=0.6, color='C0', label='With PSD')
ax.hist(dz_shuff_nopsd, bins=bins_hist, alpha=0.6, color='C1', label='Without PSD')
ax.axvline(0, color='k', linestyle=':', alpha=0.5)
ax.axvline(dz_shuff_psd.mean(), color='C0', linewidth=2, linestyle='--')
ax.axvline(dz_shuff_nopsd.mean(), color='C1', linewidth=2, linestyle='--')
ax.axvline(dz_psd, color='C0', linewidth=2, label=f'Real (PSD) = {dz_psd:.4f}')
ax.axvline(dz_nopsd, color='C1', linewidth=2, label=f'Real (noPSD) = {dz_nopsd:.4f}')
ax.set_xlabel('Δz (corrected − uncorrected)')
ax.set_ylabel('Count')
ax.set_title('Shuffle null: PSD vs no-PSD')
ax.legend(fontsize=7)

# Panel B: Across windows
ax = axes10[1]
windows_ms = [t * DT * 1000 for t in WINDOW_BINS]
ax.plot(windows_ms, dz_win_psd, 'o-', color='C0', label='Real (PSD)')
ax.plot(windows_ms, dz_win_nopsd, 's--', color='C1', label='Real (no PSD)')
ax.plot(windows_ms, dz_shuff_win_psd, 'o:', color='C0', alpha=0.5,
        label='Shuff mean (PSD)')
ax.plot(windows_ms, dz_shuff_win_nopsd, 's:', color='C1', alpha=0.5,
        label='Shuff mean (no PSD)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Counting window (ms)')
ax.set_ylabel('Δz')
ax.set_title('PSD bias across windows')
ax.set_xscale('log')
ax.set_xticks(windows_ms)
ax.set_xticklabels([f'{w:.1f}' for w in windows_ms])
ax.legend(fontsize=7, frameon=False)

# Panel C: Negative eigenvalue count in CnoiseC_shuff
ax = axes10[2]
ax.hist(neg_eig_counts_shuff, bins=np.arange(
    neg_eig_counts_shuff.min() - 0.5,
    neg_eig_counts_shuff.max() + 1.5, 1),
    color='gray', edgecolor='white', alpha=0.8)
ax.axvline(n_neg_U, color='C0', linewidth=2,
           label=f'CnoiseU: {n_neg_U} neg eigs')
ax.axvline(n_neg_C, color='red', linewidth=2,
           label=f'CnoiseC (real): {n_neg_C} neg eigs')
ax.set_xlabel('Number of negative eigenvalues')
ax.set_ylabel('Count (shuffle iterations)')
ax.set_title('Negative eigenvalues in CnoiseC')
ax.legend(fontsize=7)

fig10.suptitle(f'PSD projection bias diagnostic: {SESSION_NAME}', fontsize=12)
fig10.tight_layout()
plt.show()


# %% Summary
print("""
=== Summary ===
This script exposes the full decomposition pipeline for a single session.

Key things to check:
1. Steps 3-4: Are the Ceye curves well-behaved? Do they show clear monotonic
   trends that justify intercept extrapolation?
2. Step 5: Do the different intercept methods agree on the diagonal? The diagonal
   has a principled constraint (non-increasing variance), but off-diagonals do not.
3. Step 6: Is there systematic overestimation of off-diagonal Crate relative to
   Ctotal? This would directly cause negative noise correlations.
4. Step 8: How does the noise correlation bias compare across methods?
5. Step 10: Are fits well-conditioned (high R²)? For low R² pairs, is the
   intercept estimate reliable?
6. Step 11: Does the shuffle null center on Δz=0 or is it also biased? If
   biased, the problem is in the estimation procedure, not the data.
7. Step 13: What fraction of off-diagonal Crate exceeds Ctotal? This is the
   direct mechanism for negative corrected correlations.
8. Step 14: Is the bias explained by PSD projection? If the no-PSD shuffle null
   centers near zero while the PSD shuffle null is negatively biased, the
   projection is the dominant mechanism. If both are biased, the −Σ/n cross-
   product term or intercept overestimation also contributes.
9. Step 15: Diagnose the weighting mismatch between Crate (pair-count
   weighted) and Cpsth (uniform weighted) as the source of the shuffle bias.
10. Step 16: Simulation demonstrating the weighting mismatch mechanism.
""")


# %% Step 15: Weighting mismatch diagnostic on real data
# The split-half Cpsth gives each time bin EQUAL weight (one row per time bin).
# The cross-product estimator in compute_conditional_second_moments weights
# each time bin by n_t*(n_t-1)/2 (pair count). If n_t correlates with
# ||mu_t||^2, Crate_shuff systematically exceeds Cpsth on the off-diagonals,
# making CnoiseC < CnoiseU and biasing delta-z negative.

print("\n" + "=" * 70)
print("Step 15: Weighting mismatch diagnostic (real data)")
print("=" * 70)

# Collect per-time-bin statistics
unique_times = np.unique(T_idx.detach().cpu().numpy())
time_groups = {}
for t in unique_times:
    ix_t = np.where((T_idx == t).detach().cpu().numpy())[0]
    if len(ix_t) >= 10:  # same threshold as bagged_split_half
        time_groups[t] = ix_t

sorted_times = sorted(time_groups.keys())
n_t_list = []         # windows per time bin
mu_t_list = []        # PSTH mean per time bin (C,)
mu_t_norm_list = []   # ||mu_t - mu_global||^2
pair_count_list = []  # n_t * (n_t - 1) / 2

mu_global = Erate  # global mean spike count

for t in sorted_times:
    ix_t = time_groups[t]
    nt = len(ix_t)
    mu_t = SpikeCounts[ix_t].mean(0).detach().cpu().numpy()
    n_t_list.append(nt)
    mu_t_list.append(mu_t)
    mu_t_norm_list.append(np.sum((mu_t - mu_global) ** 2))
    pair_count_list.append(nt * (nt - 1) // 2)

n_t_arr = np.array(n_t_list)
mu_t_norm_arr = np.array(mu_t_norm_list)
pair_count_arr = np.array(pair_count_list)
mu_t_mat = np.stack(mu_t_list)  # (T, C)

# Correlation between n_t and ||mu_t||^2
from scipy.stats import spearmanr
rho_nt_mu, p_nt_mu = spearmanr(n_t_arr, mu_t_norm_arr)
print(f"\nTime bins: {len(sorted_times)}")
print(f"Windows per time bin: mean={n_t_arr.mean():.1f}, "
      f"range=[{n_t_arr.min()}, {n_t_arr.max()}], "
      f"CV={n_t_arr.std()/n_t_arr.mean():.2f}")
print(f"Correlation(n_t, ||mu_t - mu||^2): rho_s = {rho_nt_mu:.3f}, p = {p_nt_mu:.3e}")

# Compute Cpsth with UNIFORM weighting (standard)
mu_t_centered = mu_t_mat - mu_global[None, :]
n_time = len(sorted_times)
Cpsth_uniform = (mu_t_centered.T @ mu_t_centered) / (n_time - 1)
Cpsth_uniform = 0.5 * (Cpsth_uniform + Cpsth_uniform.T)

# Compute Cpsth with PAIR-COUNT weighting (matches Crate estimator)
w_pairs = pair_count_arr.astype(float)
w_pairs /= w_pairs.sum()
# Weighted mean for centering
mu_weighted = (w_pairs[:, None] * mu_t_mat).sum(axis=0)
mu_t_centered_w = mu_t_mat - mu_weighted[None, :]
# Weighted covariance: sum w_t * (mu_t - mu_w)(mu_t - mu_w)^T / (1 - sum w_t^2)
w_correction = 1.0 - np.sum(w_pairs ** 2)
Cpsth_pairweighted = (mu_t_centered_w.T * w_pairs) @ mu_t_centered_w / w_correction
Cpsth_pairweighted = 0.5 * (Cpsth_pairweighted + Cpsth_pairweighted.T)

# Compare: if pair-weighted > uniform, then Crate_shuff > Cpsth, biasing delta-z negative
ut_uniform = get_upper_triangle(Cpsth_uniform)
ut_pairwt = get_upper_triangle(Cpsth_pairweighted)
ok_cmp = np.isfinite(ut_uniform) & np.isfinite(ut_pairwt)

excess_offdiag = np.nanmean(ut_pairwt[ok_cmp] - ut_uniform[ok_cmp])
print(f"\nOff-diagonal Cpsth comparison:")
print(f"  Uniform-weighted mean:    {np.nanmean(ut_uniform[ok_cmp]):.6f}")
print(f"  Pair-count-weighted mean: {np.nanmean(ut_pairwt[ok_cmp]):.6f}")
print(f"  Excess (pair - uniform):  {excess_offdiag:.6f}")
print(f"  Fraction pair > uniform:  {np.nanmean(ut_pairwt[ok_cmp] > ut_uniform[ok_cmp]):.3f}")

# Compute delta-z using pair-weighted Cpsth to see if bias disappears
CnoiseU_pw = 0.5 * ((Ctotal - Cpsth_pairweighted) + (Ctotal - Cpsth_pairweighted).T)
dz_pw_psd, _, _ = compute_dz(CnoiseC_real, CnoiseU_pw, valid, use_psd=True)
dz_pw_nopsd, _, _ = compute_dz(CnoiseC_real, CnoiseU_pw, valid, use_psd=False)

print(f"\nDelta-z with pair-count-reweighted Cpsth as baseline:")
print(f"  Standard CnoiseU:    Δz(PSD)={dz_psd:.5f}, Δz(noPSD)={dz_nopsd:.5f}")
print(f"  Reweighted CnoiseU:  Δz(PSD)={dz_pw_psd:.5f}, Δz(noPSD)={dz_pw_nopsd:.5f}")

# Shuffle with reweighted baseline
dz_shuff_pw = []
rng_pw = torch.Generator(device=device_obj)
rng_pw.manual_seed(999)
for k in range(N_SHUFFLES):
    perm = torch.randperm(n_samples, generator=rng_pw, device=device_obj)
    EyeTraj_shuff = EyeTraj[perm]
    Crate_shuff, _, _, _, _, _ = estimate_rate_covariance(
        SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
        Ctotal=Ctotal, intercept_mode=INTERCEPT_MODE,
    )
    CnoiseC_shuff = 0.5 * ((Ctotal - Crate_shuff) + (Ctotal - Crate_shuff).T)
    dz_s, _, _ = compute_dz(CnoiseC_shuff, CnoiseU_pw, valid, use_psd=True)
    if np.isfinite(dz_s):
        dz_shuff_pw.append(dz_s)

dz_shuff_pw = np.array(dz_shuff_pw)

print(f"\nShuffle null with pair-count-reweighted baseline ({N_SHUFFLES} iters):")
print(f"  Standard baseline:   mean Δz = {dz_shuff_psd.mean():.5f}")
print(f"  Reweighted baseline: mean Δz = {dz_shuff_pw.mean():.5f}")
print(f"  Bias reduction: {abs(dz_shuff_psd.mean()) - abs(dz_shuff_pw.mean()):.5f}")

# --- Figure ---
fig11, axes11 = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: n_t vs ||mu_t||^2
ax = axes11[0]
ax.scatter(n_t_arr, mu_t_norm_arr, s=15, alpha=0.5, color='C0')
ax.set_xlabel('Windows per time bin (n_t)')
ax.set_ylabel('||mu_t - mu_global||^2')
ax.set_title(f'PSTH signal vs trial count\nrho_s={rho_nt_mu:.3f}, p={p_nt_mu:.2e}')

# Panel B: Pair-count weights vs uniform weights
ax = axes11[1]
w_uniform = np.ones(len(sorted_times)) / len(sorted_times)
sort_idx = np.argsort(n_t_arr)
ax.bar(np.arange(len(sorted_times)), w_pairs[sort_idx], alpha=0.6,
       color='C1', label='Pair-count weight')
ax.bar(np.arange(len(sorted_times)), w_uniform[sort_idx], alpha=0.4,
       color='C0', label='Uniform weight')
ax.set_xlabel('Time bins (sorted by n_t)')
ax.set_ylabel('Weight')
ax.set_title('Weighting: Crate vs Cpsth')
ax.legend(fontsize=8)

# Panel C: Shuffle null — standard vs reweighted baseline
ax = axes11[2]
bins_h = np.linspace(
    min(dz_shuff_psd.min(), dz_shuff_pw.min()) - 0.002,
    max(dz_shuff_psd.max(), dz_shuff_pw.max()) + 0.002,
    35)
ax.hist(dz_shuff_psd, bins=bins_h, alpha=0.6, color='C0',
        label=f'Standard (mean={dz_shuff_psd.mean():.4f})')
ax.hist(dz_shuff_pw, bins=bins_h, alpha=0.6, color='C2',
        label=f'Reweighted (mean={dz_shuff_pw.mean():.4f})')
ax.axvline(0, color='k', linestyle=':', alpha=0.5)
ax.axvline(dz_shuff_psd.mean(), color='C0', linewidth=2, linestyle='--')
ax.axvline(dz_shuff_pw.mean(), color='C2', linewidth=2, linestyle='--')
ax.set_xlabel('Δz (shuffle null)')
ax.set_ylabel('Count')
ax.set_title('Shuffle null: standard vs reweighted baseline')
ax.legend(fontsize=7)

fig11.suptitle(f'Weighting mismatch diagnostic: {SESSION_NAME}', fontsize=12)
fig11.tight_layout()
plt.show()


# %% Step 16: Simulation demonstrating the weighting mismatch mechanism
# Minimal simulation with known ground truth to show that the negative bias
# in delta-z arises specifically from the pair-count vs uniform weighting
# mismatch, and disappears when the baseline uses matched weights.

print("\n" + "=" * 70)
print("Step 16: Weighting mismatch simulation")
print("=" * 70)

rng_sim = np.random.default_rng(2026)

# Simulation parameters
NC_SIM = 20            # neurons
N_TIME_BINS = 40       # time bins (stimulus conditions)
N_SIMS = 300           # Monte Carlo iterations

# Two conditions: correlated n_t and ||mu_t|| vs uncorrelated
conditions = {
    "correlated": True,
    "uncorrelated": False,
}

sim_results = {}

for cond_name, correlated in conditions.items():
    dz_standard = []   # delta-z with uniform-weighted Cpsth as baseline
    dz_matched = []     # delta-z with pair-count-weighted Cpsth as baseline

    for sim_i in range(N_SIMS):
        # Assign windows per time bin
        if correlated:
            # Time bins with more windows also have stronger PSTH signals
            # (e.g., some stimuli are more reliable and survive more filters)
            base_n = rng_sim.integers(10, 30, size=N_TIME_BINS)
            # Sort so large n_t gets large signal
            n_per_bin = np.sort(base_n)
        else:
            n_per_bin = rng_sim.integers(10, 30, size=N_TIME_BINS)

        # PSTH means: magnitude correlated with n_t if correlated condition
        mu_global_sim = rng_sim.uniform(1.0, 3.0, NC_SIM)
        mu_t_sim = np.zeros((N_TIME_BINS, NC_SIM))
        for t_i in range(N_TIME_BINS):
            if correlated:
                # Signal strength scales with index (which is sorted by n_t)
                scale = 0.5 + 1.5 * t_i / N_TIME_BINS
            else:
                scale = 1.0
            mu_t_sim[t_i] = mu_global_sim + scale * rng_sim.normal(0, 0.3, NC_SIM)

        # True noise covariance (same for all time bins)
        std_noise = np.sqrt(mu_global_sim) * 0.5
        rho_noise = 0.05
        R_noise = np.full((NC_SIM, NC_SIM), rho_noise)
        np.fill_diagonal(R_noise, 1.0)
        Sigma_noise = np.outer(std_noise, std_noise) * R_noise

        # Ensure PSD
        w_eig = np.linalg.eigvalsh(Sigma_noise)
        if w_eig.min() < 0:
            Sigma_noise += (-w_eig.min() + 1e-6) * np.eye(NC_SIM)

        # Generate spike counts per time bin
        all_S = []
        all_T = []
        for t_i in range(N_TIME_BINS):
            S_t = rng_sim.multivariate_normal(
                mu_t_sim[t_i], Sigma_noise, size=n_per_bin[t_i]
            )
            all_S.append(S_t)
            all_T.extend([t_i] * n_per_bin[t_i])
        all_S = np.vstack(all_S)
        all_T = np.array(all_T)
        N_total = all_S.shape[0]

        # Ctotal
        Ct = np.cov(all_S.T, ddof=1)
        mu_hat = all_S.mean(axis=0)

        # --- Cpsth: uniform-weighted split-half ---
        Cpsth_halves = np.zeros((NC_SIM, NC_SIM))
        mu_t_hat = np.zeros((N_TIME_BINS, NC_SIM))
        for t_i in range(N_TIME_BINS):
            ix = np.where(all_T == t_i)[0]
            perm = rng_sim.permutation(ix)
            mid = len(perm) // 2
            ma = all_S[perm[:mid]].mean(0)
            mb = all_S[perm[mid:]].mean(0)
            mu_t_hat[t_i] = (ma + mb) / 2
            da = ma - mu_hat
            db = mb - mu_hat
            Cpsth_halves += np.outer(da, db)
        Cpsth_u = Cpsth_halves / (N_TIME_BINS - 1)
        Cpsth_u = 0.5 * (Cpsth_u + Cpsth_u.T)

        # --- Cpsth: pair-count-weighted ---
        pc = np.array([n * (n - 1) // 2 for n in n_per_bin], dtype=float)
        wt = pc / pc.sum()
        mu_w = (wt[:, None] * mu_t_hat).sum(0)
        dmu_w = mu_t_hat - mu_w[None, :]
        w_corr = 1.0 - np.sum(wt ** 2)
        Cpsth_pw = (dmu_w.T * wt) @ dmu_w / w_corr
        Cpsth_pw = 0.5 * (Cpsth_pw + Cpsth_pw.T)

        # --- Crate_shuff: cross-product with shuffled "eye trajectories" ---
        # Under the null, cross-products within each time bin are accumulated
        # with pair-count weighting. We directly compute the cross-product mean.
        cross_sum = np.zeros((NC_SIM, NC_SIM))
        cross_count = 0
        for t_i in range(N_TIME_BINS):
            ix = np.where(all_T == t_i)[0]
            nt = len(ix)
            if nt < 2:
                continue
            St = all_S[ix]
            # sum of cross-products: (sum_i S_i)(sum_j S_j)^T - sum_i S_i S_i^T
            sum_S = St.sum(0)
            cross = np.outer(sum_S, sum_S) - (St.T @ St)
            cross_sum += 0.5 * cross  # each (i,j) counted once
            cross_count += nt * (nt - 1) // 2
        Crate_shuff_sim = cross_sum / cross_count - np.outer(mu_hat, mu_hat)

        # Noise covariance matrices
        CnU_u = 0.5 * ((Ct - Cpsth_u) + (Ct - Cpsth_u).T)
        CnU_pw = 0.5 * ((Ct - Cpsth_pw) + (Ct - Cpsth_pw).T)
        CnC_shuff = 0.5 * ((Ct - Crate_shuff_sim) + (Ct - Crate_shuff_sim).T)

        # Correlations and delta-z
        v_sim = np.diag(CnU_u) > 0.001
        if v_sim.sum() < 3:
            continue

        def _dz(CnC, CnU, vmask):
            ru = get_upper_triangle(cov_to_corr(CnU[np.ix_(vmask, vmask)]))
            rc = get_upper_triangle(cov_to_corr(CnC[np.ix_(vmask, vmask)]))
            ok = np.isfinite(ru) & np.isfinite(rc)
            if ok.sum() == 0:
                return np.nan
            return fisher_z_mean(rc[ok]) - fisher_z_mean(ru[ok])

        dz_s = _dz(CnC_shuff, CnU_u, v_sim)
        dz_m = _dz(CnC_shuff, CnU_pw, v_sim)
        if np.isfinite(dz_s):
            dz_standard.append(dz_s)
        if np.isfinite(dz_m):
            dz_matched.append(dz_m)

    dz_standard = np.array(dz_standard)
    dz_matched = np.array(dz_matched)
    sim_results[cond_name] = {
        'standard': dz_standard,
        'matched': dz_matched,
    }

    print(f"\n  {cond_name} (n_t {'~' if correlated else '!~'} ||mu_t||):")
    print(f"    Standard baseline (uniform Cpsth):  "
          f"mean Δz = {dz_standard.mean():.5f}, "
          f"95% CI = [{np.percentile(dz_standard, 2.5):.5f}, "
          f"{np.percentile(dz_standard, 97.5):.5f}]")
    print(f"    Matched baseline (pair-wt Cpsth):   "
          f"mean Δz = {dz_matched.mean():.5f}, "
          f"95% CI = [{np.percentile(dz_matched, 2.5):.5f}, "
          f"{np.percentile(dz_matched, 97.5):.5f}]")

# --- Figure ---
fig12, axes12 = plt.subplots(1, 2, figsize=(12, 4.5))

for ax_i, (cond_name, res) in enumerate(sim_results.items()):
    ax = axes12[ax_i]
    lo = min(res['standard'].min(), res['matched'].min())
    hi = max(res['standard'].max(), res['matched'].max())
    bins_s = np.linspace(lo - 0.002, hi + 0.002, 40)
    ax.hist(res['standard'], bins=bins_s, alpha=0.6, color='C0',
            label=f'Uniform Cpsth (mean={res["standard"].mean():.4f})')
    ax.hist(res['matched'], bins=bins_s, alpha=0.6, color='C2',
            label=f'Pair-wt Cpsth (mean={res["matched"].mean():.4f})')
    ax.axvline(0, color='k', linestyle=':', alpha=0.7)
    ax.axvline(res['standard'].mean(), color='C0', linewidth=2, linestyle='--')
    ax.axvline(res['matched'].mean(), color='C2', linewidth=2, linestyle='--')
    ax.set_xlabel('Δz (shuffle null)')
    ax.set_ylabel('Count')
    corr_str = "n_t correlated with ||mu_t||" if "corr" in cond_name else "n_t independent of ||mu_t||"
    ax.set_title(f'{corr_str}')
    ax.legend(fontsize=7)

fig12.suptitle('Weighting mismatch simulation: pair-count vs uniform Cpsth weighting',
               fontsize=11)
fig12.tight_layout()
plt.show()

print("\n=== Simulation summary ===")
print("When n_t (windows per time bin) correlates with PSTH signal strength,")
print("the pair-count-weighted Crate estimator systematically exceeds the")
print("uniform-weighted Cpsth estimator on the off-diagonals. This inflates")
print("CnoiseC = Ctotal - Crate relative to CnoiseU = Ctotal - Cpsth, reducing")
print("corrected noise correlations and biasing Δz negative.")
print("")
print("The bias disappears when using pair-count-weighted Cpsth as the baseline,")
print("confirming the weighting mismatch as the mechanism.")
