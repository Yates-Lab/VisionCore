"""
Hypothesis 4: Global mean subtraction in Ceye = MM - E[rate] x E[rate]^T
introduces systematic bias that inflates off-diagonal Crate.

The mechanism: compute_conditional_second_moments accumulates cross-trial
outer products per time bin, then normalizes by total pair counts.  The
weight w_t(d) of time bin t in distance bin d generally differs from the
marginal proportion p_t of windows from time bin t.

Ceye_current(d)  = MM(d) - mu_global mu_global^T
Ceye_correct(d)  = MM(d) - mu_weighted(d) mu_weighted(d)^T
   where mu_weighted(d) = sum_t w_t(d) mu_t

bias(d) = Ceye_current(d) - Ceye_correct(d)
        = mu_weighted(d) mu_weighted(d)^T - mu_global mu_global^T

If mu_weighted(d) != mu_global, the off-diagonal bias is nonzero.
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
    compute_conditional_second_moments,
    estimate_rate_covariance,
    bagged_split_half_psth_covariance,
    fit_intercept_linear,
    fit_intercept_pava,
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
            free, total = torch.cuda.mem_get_info(i)
            free_mem.append(free)
        best_gpu = int(np.argmax(free_mem))
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
    """Extract windows and return tensors + metadata."""
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


# ============================================================================
# Part 1: Characterize weight variation w_t(d) vs p_t on real data
# ============================================================================

def compute_weight_variation(SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS):
    """
    For each distance bin d, compute w_t(d) = fraction of pairs from time bin t.
    Compare to marginal p_t = fraction of all windows from time bin t.

    Also compute the effective mean mu_weighted(d) for each distance bin.

    Returns a dict with all diagnostic quantities.
    """
    N_samples, T, _ = EyeTraj.shape
    device = EyeTraj.device
    C = SpikeCounts.shape[1]

    EyeFlat = EyeTraj.reshape(N_samples, -1)
    inv_sqrt_T = 1.0 / torch.sqrt(torch.tensor(float(T), device=device, dtype=EyeTraj.dtype))

    # Compute bin edges from all pairs
    i_up, j_up = torch.triu_indices(N_samples, N_samples, offset=1)
    dist_all = torch.cdist(EyeFlat, EyeFlat)[i_up, j_up] * inv_sqrt_T
    bin_edges = np.percentile(dist_all.cpu().numpy(), np.arange(0, 100, 100 / (n_bins + 1)))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins_actual = len(bin_edges) - 1
    bin_edges_t = torch.as_tensor(bin_edges, device=device, dtype=EyeTraj.dtype)

    unique_times = np.unique(T_idx.detach().cpu().numpy())
    n_times = len(unique_times)
    time_to_idx = {int(t): i for i, t in enumerate(unique_times)}

    # Marginal proportions p_t
    T_idx_np = T_idx.detach().cpu().numpy()
    total_windows = len(T_idx_np)
    p_t = np.zeros(n_times)
    for t in unique_times:
        p_t[time_to_idx[int(t)]] = np.sum(T_idx_np == t) / total_windows

    # Count pairs per (time_bin, distance_bin)
    # Also accumulate spike count sums per (time_bin, distance_bin) for mu_weighted
    pair_counts = np.zeros((n_times, n_bins_actual))
    # We need to track which windows contribute to each distance bin,
    # not just which pairs. For the mean, we need the mean of the
    # *marginal* spike counts of windows that appear in pairs for bin d.
    # A window i contributes to bin d if there exists j such that (i,j) is in bin d.
    # The effective mean is more subtle than just weighting by time bin.

    # For the analytical bias, what matters is:
    # MM(d) = sum_t [n_pairs_t(d) / n_total_pairs(d)] * (1/n_pairs_t(d)) sum_{(i,j) in bin d, time t} S_i S_j^T
    # For cross-trial products within the same time bin, if trials are independent:
    # E[S_i S_j^T | t] = mu_t mu_t^T + Crate_within_t
    # So: MM(d) approx sum_t w_t(d) [mu_t mu_t^T + Crate(d)]
    #            = sum_t w_t(d) mu_t mu_t^T + Crate(d)
    # (assuming Crate doesn't depend on t, only on d)
    #
    # Ceye_current(d) = MM(d) - mu_global mu_global^T
    #                 = sum_t w_t(d) mu_t mu_t^T + Crate(d) - mu_global mu_global^T
    #
    # The "correct" subtraction for Ceye_correct(d) should yield just Crate(d).
    # This requires subtracting: sum_t w_t(d) mu_t mu_t^T
    #
    # But we subtract mu_global mu_global^T = [sum_t p_t mu_t][sum_t p_t mu_t]^T
    #
    # So: bias(d) = Ceye_current(d) - Crate(d)
    #            = sum_t w_t(d) mu_t mu_t^T - mu_global mu_global^T
    #
    # This equals: sum_t w_t(d) mu_t mu_t^T - [sum_t p_t mu_t][sum_t p_t mu_t]^T
    #
    # Note: if w_t(d) = p_t for all d, then:
    # bias = sum_t p_t mu_t mu_t^T - [sum_t p_t mu_t]^2 = Cov_across_time(mu_t)
    # This is the PSTH covariance! It's present at ALL distance bins equally.
    # So the Ceye *curve* has a constant offset = Cpsth, and the intercept
    # correctly captures Crate + Cpsth. The subsequent Cfem = Crate - Cpsth
    # would be correct.
    #
    # BUT if w_t(d) varies with d, the bias varies with d, which distorts
    # the Ceye curve shape and the intercept estimate.

    # Let's compute per-time-bin per-distance-bin pair counts
    S_np = SpikeCounts.detach().cpu().numpy()

    # Per-time-bin mean spike rates
    mu_t_arr = np.zeros((n_times, C))
    n_windows_t = np.zeros(n_times)
    for t in unique_times:
        idx = time_to_idx[int(t)]
        mask = (T_idx_np == t)
        n_windows_t[idx] = mask.sum()
        if mask.sum() > 0:
            mu_t_arr[idx] = S_np[mask].mean(axis=0)

    # Accumulate pair counts per (time_bin, distance_bin)
    for t in unique_times:
        t_idx_local = time_to_idx[int(t)]
        valid = np.where(T_idx_np == t)[0]
        N = len(valid)
        if N < 2:
            continue

        X = EyeTraj[valid]
        Xflat = X.reshape(N, -1)
        D = torch.cdist(Xflat, Xflat) * inv_sqrt_T

        ii, jj = torch.triu_indices(N, N, offset=1, device=device)
        d = D[ii, jj]
        bid = torch.bucketize(d, bin_edges_t, right=False)
        ok = (bid >= 1) & (bid <= n_bins_actual)
        bid_ok = bid[ok].cpu().numpy()

        for k in range(1, n_bins_actual + 1):
            count = (bid_ok == k).sum()
            pair_counts[t_idx_local, k - 1] += count

    # w_t(d) = pair_counts[t, d] / sum_t pair_counts[t, d]
    total_pairs_per_bin = pair_counts.sum(axis=0)
    w_td = np.zeros_like(pair_counts)
    for d in range(n_bins_actual):
        if total_pairs_per_bin[d] > 0:
            w_td[:, d] = pair_counts[:, d] / total_pairs_per_bin[d]

    # Compute analytical bias at each distance bin
    # bias(d) = sum_t w_t(d) mu_t mu_t^T - mu_global mu_global^T
    mu_global = S_np.mean(axis=0)
    mu_global_outer = np.outer(mu_global, mu_global)

    bias_per_bin = np.zeros((n_bins_actual, C, C))
    weighted_mean_per_bin = np.zeros((n_bins_actual, C))
    for d in range(n_bins_actual):
        weighted_outer = np.zeros((C, C))
        mu_w = np.zeros(C)
        for t_idx in range(n_times):
            weighted_outer += w_td[t_idx, d] * np.outer(mu_t_arr[t_idx], mu_t_arr[t_idx])
            mu_w += w_td[t_idx, d] * mu_t_arr[t_idx]
        bias_per_bin[d] = weighted_outer - mu_global_outer
        weighted_mean_per_bin[d] = mu_w

    # Also compute what the "correct" Ceye would use: the w_t(d)-weighted outer product
    # Ceye_correct(d) = MM(d) - sum_t w_t(d) mu_t mu_t^T
    # The bias is: Ceye_current - Ceye_correct = sum_t w_t(d) mu_t mu_t^T - mu_global mu_global^T

    return {
        "unique_times": unique_times,
        "n_times": n_times,
        "p_t": p_t,
        "w_td": w_td,
        "pair_counts": pair_counts,
        "total_pairs_per_bin": total_pairs_per_bin,
        "mu_t_arr": mu_t_arr,
        "mu_global": mu_global,
        "bias_per_bin": bias_per_bin,
        "weighted_mean_per_bin": weighted_mean_per_bin,
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "n_bins_actual": n_bins_actual,
        "n_windows_t": n_windows_t,
    }


def print_weight_variation(wv):
    """Print diagnostics about weight variation."""
    print("\n" + "=" * 80)
    print("PART 1: Weight Variation w_t(d) vs p_t")
    print("=" * 80)

    p_t = wv["p_t"]
    w_td = wv["w_td"]
    n_times = wv["n_times"]
    n_bins = wv["n_bins_actual"]

    print(f"\nNumber of unique time bins: {n_times}")
    print(f"Number of distance bins: {n_bins}")
    print(f"Windows per time bin: {wv['n_windows_t']}")
    print(f"Total pairs per distance bin: {wv['total_pairs_per_bin']}")

    # Max deviation
    max_dev = 0
    for d in range(n_bins):
        for t in range(n_times):
            dev = abs(w_td[t, d] - p_t[t])
            if dev > max_dev:
                max_dev = dev
    print(f"\nMax |w_t(d) - p_t| across all (t, d): {max_dev:.6f}")

    # Average deviation per distance bin
    avg_dev_per_bin = np.zeros(n_bins)
    for d in range(n_bins):
        avg_dev_per_bin[d] = np.mean(np.abs(w_td[:, d] - p_t))
    print(f"Mean |w_t(d) - p_t| per distance bin:")
    for d in range(n_bins):
        print(f"  bin {d:2d} (d={wv['bin_centers'][d]:.4f}): "
              f"avg_dev={avg_dev_per_bin[d]:.6f}, "
              f"pairs={wv['total_pairs_per_bin'][d]:.0f}")

    # Total variation distance per bin
    tvd_per_bin = np.zeros(n_bins)
    for d in range(n_bins):
        tvd_per_bin[d] = 0.5 * np.sum(np.abs(w_td[:, d] - p_t))
    print(f"\nTotal variation distance (TVD) per distance bin:")
    print(f"  Mean TVD: {np.mean(tvd_per_bin):.6f}")
    print(f"  Max TVD:  {np.max(tvd_per_bin):.6f}")
    print(f"  Min TVD:  {np.min(tvd_per_bin):.6f}")

    return avg_dev_per_bin, tvd_per_bin


# ============================================================================
# Part 2: Analytical bias magnitude
# ============================================================================

def analyze_bias_magnitude(wv):
    """Compute and report the analytical bias from global mean subtraction."""
    print("\n" + "=" * 80)
    print("PART 2: Analytical Bias Magnitude")
    print("=" * 80)

    bias_per_bin = wv["bias_per_bin"]
    n_bins = wv["n_bins_actual"]
    C = bias_per_bin.shape[1]

    # Off-diagonal bias per bin
    offdiag_bias_per_bin = np.zeros(n_bins)
    for d in range(n_bins):
        offdiag = get_upper_triangle(bias_per_bin[d])
        offdiag_bias_per_bin[d] = np.mean(offdiag)

    print(f"\nMean off-diagonal bias per distance bin:")
    for d in range(n_bins):
        print(f"  bin {d:2d} (d={wv['bin_centers'][d]:.4f}): "
              f"mean_offdiag_bias = {offdiag_bias_per_bin[d]:+.6e}")

    # How much does this bias VARY across distance bins?
    # This is what matters: if the bias is constant across d,
    # the intercept fitting removes it. Only the d-dependent part matters.
    bias_mean_across_d = np.mean(offdiag_bias_per_bin)
    bias_var_across_d = offdiag_bias_per_bin - bias_mean_across_d
    print(f"\nMean off-diagonal bias (averaged across d): {bias_mean_across_d:+.6e}")
    print(f"Std of off-diagonal bias across d:          {np.std(offdiag_bias_per_bin):.6e}")
    print(f"Range of bias across d: [{np.min(offdiag_bias_per_bin):+.6e}, "
          f"{np.max(offdiag_bias_per_bin):+.6e}]")

    # The constant part of the bias is absorbed by the intercept.
    # The d-varying part distorts the Ceye curve shape.
    print(f"\nBias variation (deviation from mean) per bin:")
    for d in range(n_bins):
        print(f"  bin {d:2d}: delta_bias = {bias_var_across_d[d]:+.6e}")

    # Diagonal bias (affects variance estimates)
    diag_bias_per_bin = np.zeros(n_bins)
    for d in range(n_bins):
        diag_bias_per_bin[d] = np.mean(np.diag(bias_per_bin[d]))
    print(f"\nMean diagonal bias per distance bin:")
    print(f"  Mean across d: {np.mean(diag_bias_per_bin):+.6e}")
    print(f"  Std across d:  {np.std(diag_bias_per_bin):.6e}")

    # Compare weighted mean vs global mean
    mu_global = wv["mu_global"]
    weighted_mean = wv["weighted_mean_per_bin"]
    diff_norm = np.zeros(n_bins)
    for d in range(n_bins):
        diff_norm[d] = np.linalg.norm(weighted_mean[d] - mu_global)
    print(f"\n||mu_weighted(d) - mu_global|| per bin:")
    print(f"  Mean: {np.mean(diff_norm):.6e}")
    print(f"  Max:  {np.max(diff_norm):.6e}")
    print(f"  ||mu_global||: {np.linalg.norm(mu_global):.4f}")
    print(f"  Relative deviation: {np.max(diff_norm) / np.linalg.norm(mu_global):.6e}")

    return offdiag_bias_per_bin, bias_var_across_d


# ============================================================================
# Part 3: Corrected pipeline — bin-specific mean subtraction
# ============================================================================

def run_corrected_pipeline(SpikeCounts, EyeTraj, T_idx, wv, Ctotal):
    """
    Run the covariance decomposition with corrected mean subtraction.

    Instead of: Ceye(d) = MM(d) - mu_global mu_global^T
    Use:        Ceye_corrected(d) = MM(d) - sum_t w_t(d) mu_t mu_t^T
    """
    print("\n" + "=" * 80)
    print("PART 3: Corrected Pipeline (bin-specific mean subtraction)")
    print("=" * 80)

    # Run standard pipeline
    MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
        SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS
    )
    Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
    Ceye_current = MM - Erate[:, None] * Erate[None, :]

    # Corrected: subtract bin-specific weighted outer product
    bias_per_bin = wv["bias_per_bin"]
    Ceye_corrected = Ceye_current - bias_per_bin
    # Equivalently: Ceye_corrected = MM - sum_t w_t(d) mu_t mu_t^T

    # Fit intercepts on both
    Crate_current = fit_intercept_linear(Ceye_current, bin_centers, count_e, eval_at_first_bin=True)
    Crate_corrected = fit_intercept_linear(Ceye_corrected, bin_centers, count_e, eval_at_first_bin=True)

    # Also try PAVA
    Crate_current_pava = fit_intercept_pava(Ceye_current, count_e)
    Crate_corrected_pava = fit_intercept_pava(Ceye_corrected, count_e)

    # Also try raw bin 0
    Crate_current_raw = Ceye_current[0].copy()
    Crate_corrected_raw = Ceye_corrected[0].copy()

    # PSTH covariance
    Cpsth, PSTH_mean = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
        seed=42, global_mean=Erate,
    )

    n_cells = SpikeCounts.shape[1]
    n_samples = SpikeCounts.shape[0]

    # Compute Dz for each
    total_spikes = Erate * n_samples
    valid_base = (
        np.isfinite(Erate) & (total_spikes >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > 0) & np.isfinite(np.diag(Cpsth))
    )
    CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)

    def compute_dz(Crate_in, label, method="linear"):
        C = Crate_in.copy()
        bad = np.diag(C) > 0.99 * np.diag(Ctotal)
        C[bad, :] = np.nan
        C[:, bad] = np.nan
        valid = valid_base & np.isfinite(np.diag(C))
        n_valid = valid.sum()

        CnoiseC = 0.5 * ((Ctotal - C) + (Ctotal - C).T)

        NoiseCorrU = cov_to_corr(project_to_psd(CnoiseU[np.ix_(valid, valid)]))
        NoiseCorrC = cov_to_corr(project_to_psd(CnoiseC[np.ix_(valid, valid)]))

        rho_u = get_upper_triangle(NoiseCorrU)
        rho_c = get_upper_triangle(NoiseCorrC)

        pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
        zU = fisher_z_mean(rho_u[pair_ok])
        zC = fisher_z_mean(rho_c[pair_ok])
        dz = zC - zU

        # High-rate pairs
        rate_sub = Erate[valid]
        high_rate_mask = rate_sub > np.median(rate_sub)
        rows_v, cols_v = np.triu_indices(n_valid, k=1)
        high_pair = high_rate_mask[rows_v] & high_rate_mask[cols_v]
        ok_idx = np.where(pair_ok)[0]
        high_in_ok = high_pair[ok_idx] if len(ok_idx) <= len(high_pair) else np.ones(len(ok_idx), dtype=bool)
        zU_hi = fisher_z_mean(rho_u[pair_ok][high_in_ok])
        zC_hi = fisher_z_mean(rho_c[pair_ok][high_in_ok])
        dz_hi = zC_hi - zU_hi

        offdiag_crate = get_upper_triangle(C[np.ix_(valid, valid)])
        offdiag_ctotal = get_upper_triangle(Ctotal[np.ix_(valid, valid)])

        return {
            "label": label, "method": method,
            "zU": zU, "zC": zC, "dz": dz,
            "zU_hi": zU_hi, "zC_hi": zC_hi, "dz_hi": dz_hi,
            "n_valid": n_valid,
            "mean_offdiag_crate": np.nanmean(offdiag_crate),
            "mean_offdiag_ctotal": np.nanmean(offdiag_ctotal),
        }

    results = {}
    results["current_linear"] = compute_dz(Crate_current, "Current (global mean)", "linear")
    results["corrected_linear"] = compute_dz(Crate_corrected, "Corrected (bin-specific)", "linear")
    results["current_pava"] = compute_dz(Crate_current_pava, "Current (global mean)", "PAVA")
    results["corrected_pava"] = compute_dz(Crate_corrected_pava, "Corrected (bin-specific)", "PAVA")
    results["current_raw"] = compute_dz(Crate_current_raw, "Current (global mean)", "raw")
    results["corrected_raw"] = compute_dz(Crate_corrected_raw, "Corrected (bin-specific)", "raw")

    # Print comparison
    print(f"\n{'Pipeline':>30s} | {'Method':>8s} | {'zU':>7s} {'zC':>7s} {'Dz':>8s} "
          f"{'Dz_hi':>8s} {'n_valid':>7s} {'<Crate_od>':>12s} {'<Ctotal_od>':>12s}")
    print("-" * 115)
    for key in ["current_linear", "corrected_linear",
                "current_pava", "corrected_pava",
                "current_raw", "corrected_raw"]:
        r = results[key]
        print(f"{r['label']:>30s} | {r['method']:>8s} | {r['zU']:+.4f} {r['zC']:+.4f} {r['dz']:+.5f} "
              f"{r['dz_hi']:+.5f} {r['n_valid']:>7d} {r['mean_offdiag_crate']:+.4e} {r['mean_offdiag_ctotal']:+.4e}")

    # Differences
    print(f"\nDifference (corrected - current):")
    for method in ["linear", "pava", "raw"]:
        cur = results[f"current_{method}"]
        cor = results[f"corrected_{method}"]
        print(f"  {method:>8s}: delta_Dz = {cor['dz'] - cur['dz']:+.5f}, "
              f"delta_Dz_hi = {cor['dz_hi'] - cur['dz_hi']:+.5f}")

    return results, Ceye_current, Ceye_corrected, bin_centers, count_e


# ============================================================================
# Part 4: Quantify against observed effect
# ============================================================================

def quantify_against_observed(results):
    """Compare bias magnitude to observed Dz."""
    print("\n" + "=" * 80)
    print("PART 4: Comparison to Observed Effect")
    print("=" * 80)

    observed_dz_pooled = -0.11
    observed_dz_highrate = -0.30

    for method in ["linear", "pava", "raw"]:
        cur = results[f"current_{method}"]
        cor = results[f"corrected_{method}"]
        delta_dz = cor["dz"] - cur["dz"]
        delta_dz_hi = cor["dz_hi"] - cur["dz_hi"]

        pct_pooled = (delta_dz / observed_dz_pooled) * 100 if observed_dz_pooled != 0 else 0
        pct_highrate = (delta_dz_hi / observed_dz_highrate) * 100 if observed_dz_highrate != 0 else 0

        print(f"\n  {method} intercept:")
        print(f"    Current  Dz = {cur['dz']:+.5f}, Dz_hi = {cur['dz_hi']:+.5f}")
        print(f"    Corrected Dz = {cor['dz']:+.5f}, Dz_hi = {cor['dz_hi']:+.5f}")
        print(f"    Delta Dz (pooled):   {delta_dz:+.5f}  ({pct_pooled:+.1f}% of observed -0.11)")
        print(f"    Delta Dz (high-rate): {delta_dz_hi:+.5f}  ({pct_highrate:+.1f}% of observed -0.30)")


# ============================================================================
# Part 5: Edge cases — time bins, PSTH amplitude, FEM strength
# ============================================================================

def test_edge_cases(robs, eyepos, valid_mask, device):
    """
    Test how the bias depends on:
    1. Number of time bins (window size)
    2. PSTH amplitude (scale spike rates)
    """
    print("\n" + "=" * 80)
    print("PART 5: Edge Cases")
    print("=" * 80)

    robs_clean = np.nan_to_num(robs, nan=0.0)
    eyepos_clean = np.nan_to_num(eyepos, nan=0.0)
    device_obj = torch.device(device)
    robs_t = torch.tensor(robs_clean, dtype=torch.float32, device=device_obj)
    eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32, device=device_obj)
    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)

    # Test 1: Varying window sizes (different t_count -> different number of unique time bins)
    print("\n--- Test 1: Effect of window size (number of time bins) ---")
    window_sizes = [1, 2, 4, 8, 16]
    edge_results = []
    for t_count in window_sizes:
        t_hist = max(T_HIST, t_count)
        SpikeCounts, EyeTraj, T_idx, _ = extract_windows(
            robs_t, eyepos_t, segments, t_count, t_hist, device=device
        )
        if SpikeCounts is None or SpikeCounts.shape[0] < 100:
            print(f"  t_count={t_count}: insufficient data")
            continue

        n_samples = SpikeCounts.shape[0]
        n_unique_times = len(np.unique(T_idx.detach().cpu().numpy()))
        Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
        ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
        Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()

        wv = compute_weight_variation(SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS)

        # Bias summary: mean off-diagonal bias variation across distance bins
        bias_offdiag = np.zeros(wv["n_bins_actual"])
        for d in range(wv["n_bins_actual"]):
            bias_offdiag[d] = np.mean(get_upper_triangle(wv["bias_per_bin"][d]))
        bias_std = np.std(bias_offdiag)
        bias_range = np.max(bias_offdiag) - np.min(bias_offdiag)

        # TVD
        tvd_per_bin = np.zeros(wv["n_bins_actual"])
        for d in range(wv["n_bins_actual"]):
            tvd_per_bin[d] = 0.5 * np.sum(np.abs(wv["w_td"][:, d] - wv["p_t"]))
        mean_tvd = np.mean(tvd_per_bin)

        # Quick Dz comparison
        MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
            SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS
        )
        Ceye_cur = MM - Erate[:, None] * Erate[None, :]
        Ceye_cor = Ceye_cur - wv["bias_per_bin"]
        Crate_cur = fit_intercept_linear(Ceye_cur, bin_centers, count_e, eval_at_first_bin=True)
        Crate_cor = fit_intercept_linear(Ceye_cor, bin_centers, count_e, eval_at_first_bin=True)

        Cpsth, _ = bagged_split_half_psth_covariance(
            SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
            seed=42, global_mean=Erate,
        )

        total_spikes = Erate * n_samples
        valid_base = (
            np.isfinite(Erate) & (total_spikes >= MIN_TOTAL_SPIKES)
            & (np.diag(Ctotal) > 0) & np.isfinite(np.diag(Cpsth))
        )
        CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)

        def quick_dz(Crate_in):
            C = Crate_in.copy()
            bad = np.diag(C) > 0.99 * np.diag(Ctotal)
            C[bad, :] = np.nan
            C[:, bad] = np.nan
            valid = valid_base & np.isfinite(np.diag(C))
            if valid.sum() < 3:
                return np.nan
            CnoiseC = 0.5 * ((Ctotal - C) + (Ctotal - C).T)
            NoiseCorrU = cov_to_corr(project_to_psd(CnoiseU[np.ix_(valid, valid)]))
            NoiseCorrC = cov_to_corr(project_to_psd(CnoiseC[np.ix_(valid, valid)]))
            rho_u = get_upper_triangle(NoiseCorrU)
            rho_c = get_upper_triangle(NoiseCorrC)
            pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
            if pair_ok.sum() == 0:
                return np.nan
            return fisher_z_mean(rho_c[pair_ok]) - fisher_z_mean(rho_u[pair_ok])

        dz_cur = quick_dz(Crate_cur)
        dz_cor = quick_dz(Crate_cor)

        print(f"  t_count={t_count:2d} ({t_count/240*1000:.1f}ms): "
              f"n_windows={n_samples}, n_time_bins={n_unique_times}, "
              f"mean_TVD={mean_tvd:.4f}, "
              f"bias_std={bias_std:.2e}, bias_range={bias_range:.2e}, "
              f"Dz_cur={dz_cur:+.4f}, Dz_cor={dz_cor:+.4f}, "
              f"delta_Dz={dz_cor-dz_cur:+.5f}")

        edge_results.append({
            "t_count": t_count,
            "n_windows": n_samples,
            "n_time_bins": n_unique_times,
            "mean_tvd": mean_tvd,
            "bias_std": bias_std,
            "bias_range": bias_range,
            "dz_cur": dz_cur,
            "dz_cor": dz_cor,
            "delta_dz": dz_cor - dz_cur,
        })

    return edge_results


# ============================================================================
# Figures
# ============================================================================

def make_figures(wv, offdiag_bias_per_bin, pipeline_results,
                 Ceye_current, Ceye_corrected, bin_centers, count_e,
                 edge_results):
    """Generate diagnostic figures."""
    print("\n>>> Generating figures...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel A: w_t(d) heatmap
    ax = axes[0, 0]
    im = ax.imshow(wv["w_td"], aspect='auto', cmap='viridis',
                   extent=[0, wv["n_bins_actual"], wv["n_times"], 0])
    ax.set_xlabel("Distance bin")
    ax.set_ylabel("Time bin index")
    ax.set_title("A: w_t(d) — time bin weights per distance bin")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Overlay p_t as horizontal lines
    for t in range(wv["n_times"]):
        ax.axhline(t + 0.5, color='white', lw=0.3, alpha=0.3)

    # Panel B: p_t vs mean w_t across distance
    ax = axes[0, 1]
    mean_w_per_t = wv["w_td"].mean(axis=1)
    std_w_per_t = wv["w_td"].std(axis=1)
    t_indices = np.arange(wv["n_times"])
    ax.errorbar(t_indices, mean_w_per_t, yerr=std_w_per_t, fmt='o', color='steelblue',
                markersize=3, label='mean w_t(d) +/- std', capsize=2)
    ax.plot(t_indices, wv["p_t"], 'rx', markersize=5, label='p_t (marginal)')
    ax.set_xlabel("Time bin index")
    ax.set_ylabel("Weight / proportion")
    ax.set_title("B: w_t(d) variation vs marginal p_t")
    ax.legend(fontsize=7)

    # Panel C: Off-diagonal bias vs distance
    ax = axes[0, 2]
    ax.plot(wv["bin_centers"], offdiag_bias_per_bin, 'o-', color='darkred', markersize=4)
    ax.axhline(np.mean(offdiag_bias_per_bin), color='gray', ls='--',
               label=f'mean={np.mean(offdiag_bias_per_bin):.2e}')
    ax.set_xlabel("Eye trajectory distance")
    ax.set_ylabel("Mean off-diagonal bias")
    ax.set_title("C: Analytical bias in Ceye vs distance")
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # Panel D: Example Ceye curves (current vs corrected)
    ax = axes[1, 0]
    # Pick a representative off-diagonal pair
    n_cells = Ceye_current.shape[1]
    # Average off-diagonal Ceye curve
    offdiag_cur = np.zeros(len(bin_centers))
    offdiag_cor = np.zeros(len(bin_centers))
    for d in range(len(bin_centers)):
        offdiag_cur[d] = np.mean(get_upper_triangle(Ceye_current[d]))
        offdiag_cor[d] = np.mean(get_upper_triangle(Ceye_corrected[d]))
    ax.plot(bin_centers, offdiag_cur, 'o-', color='red', markersize=4, label='Current (global mean)')
    ax.plot(bin_centers, offdiag_cor, 's-', color='blue', markersize=4, label='Corrected (bin-specific)')
    ax.set_xlabel("Eye trajectory distance")
    ax.set_ylabel("Mean off-diagonal Ceye")
    ax.set_title("D: Ceye curves (current vs corrected)")
    ax.legend(fontsize=8)

    # Panel E: Dz comparison bar chart
    ax = axes[1, 1]
    methods = ["linear", "pava", "raw"]
    x_pos = np.arange(len(methods))
    dz_cur = [pipeline_results[f"current_{m}"]["dz"] for m in methods]
    dz_cor = [pipeline_results[f"corrected_{m}"]["dz"] for m in methods]
    width = 0.35
    ax.bar(x_pos - width/2, dz_cur, width, color='red', alpha=0.7, label='Current')
    ax.bar(x_pos + width/2, dz_cor, width, color='blue', alpha=0.7, label='Corrected')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(-0.11, color='gray', ls='--', lw=1, label='Observed Dz=-0.11')
    ax.set_ylabel("Dz (Fisher z)")
    ax.set_title("E: Dz comparison (current vs corrected)")
    ax.legend(fontsize=7)

    # Panel F: Edge case — Dz vs window size
    ax = axes[1, 2]
    if edge_results:
        t_counts = [r["t_count"] for r in edge_results]
        dz_curs = [r["dz_cur"] for r in edge_results]
        dz_cors = [r["dz_cor"] for r in edge_results]
        delta_dzs = [r["delta_dz"] for r in edge_results]
        ax.plot(t_counts, dz_curs, 'o-', color='red', label='Current Dz')
        ax.plot(t_counts, dz_cors, 's-', color='blue', label='Corrected Dz')
        ax.plot(t_counts, delta_dzs, '^-', color='green', label='Delta Dz')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel("Window size (bins)")
        ax.set_ylabel("Dz (Fisher z)")
        ax.set_title("F: Dz vs window size")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No edge case data", ha='center', va='center', transform=ax.transAxes)

    plt.suptitle("H4: Global Mean Subtraction Bias in Ceye", fontsize=14, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "h4_mean_subtraction.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("H4: Global Mean Subtraction Bias in Ceye = MM - E[rate] x E[rate]^T")
    print("=" * 80)

    device = get_device()

    # Load data
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    if robs is None:
        print("ERROR: Could not load session data")
        return

    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Part 1: Weight variation
    print("\n>>> Computing weight variation...")
    wv = compute_weight_variation(SpikeCounts, EyeTraj, T_idx, n_bins=N_BINS)
    avg_dev, tvd = print_weight_variation(wv)

    # Part 2: Analytical bias
    offdiag_bias, bias_var = analyze_bias_magnitude(wv)

    # Part 3: Corrected pipeline
    ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()
    pipeline_results, Ceye_cur, Ceye_cor, bin_centers, count_e = run_corrected_pipeline(
        SpikeCounts, EyeTraj, T_idx, wv, Ctotal
    )

    # Part 4: Quantify
    quantify_against_observed(pipeline_results)

    # Part 5: Edge cases
    edge_results = test_edge_cases(robs, eyepos, valid_mask, device)

    # Figures
    make_figures(wv, offdiag_bias, pipeline_results, Ceye_cur, Ceye_cor,
                 bin_centers, count_e, edge_results)

    # Summary
    print("\n" + "=" * 80)
    print("H4 SUMMARY")
    print("=" * 80)

    delta_dz_linear = pipeline_results["corrected_linear"]["dz"] - pipeline_results["current_linear"]["dz"]
    delta_dz_hi_linear = pipeline_results["corrected_linear"]["dz_hi"] - pipeline_results["current_linear"]["dz_hi"]

    print(f"\n1. Weight variation w_t(d) vs marginal p_t:")
    print(f"   Mean TVD across distance bins: {np.mean(tvd):.6f}")
    print(f"   Max TVD: {np.max(tvd):.6f}")

    print(f"\n2. Analytical bias in off-diagonal Ceye:")
    print(f"   Mean bias (constant offset): {np.mean(offdiag_bias):+.6e}")
    print(f"   Std of bias across d (distortion): {np.std(offdiag_bias):.6e}")
    print(f"   Range of bias across d: {np.max(offdiag_bias) - np.min(offdiag_bias):.6e}")

    print(f"\n3. Effect of correction on Dz:")
    print(f"   Delta Dz (pooled, linear): {delta_dz_linear:+.5f}")
    print(f"   Delta Dz (high-rate, linear): {delta_dz_hi_linear:+.5f}")
    print(f"   Current Dz: {pipeline_results['current_linear']['dz']:+.5f}")
    print(f"   Corrected Dz: {pipeline_results['corrected_linear']['dz']:+.5f}")

    pct_pooled = abs(delta_dz_linear / 0.11) * 100
    pct_hi = abs(delta_dz_hi_linear / 0.30) * 100
    print(f"\n4. Fraction of observed bias explained:")
    print(f"   Pooled Dz=-0.11:   delta={delta_dz_linear:+.5f} ({pct_pooled:.1f}%)")
    print(f"   High-rate Dz=-0.30: delta={delta_dz_hi_linear:+.5f} ({pct_hi:.1f}%)")

    if abs(delta_dz_linear) > 0.01:
        conclusion = "PARTIALLY CONFIRMED"
        detail = (f"Global mean subtraction introduces a bias of {delta_dz_linear:+.5f} Dz "
                  f"({pct_pooled:.1f}% of the observed -0.11).")
    elif abs(delta_dz_linear) > 0.001:
        conclusion = "WEAKLY SUPPORTED"
        detail = (f"A small effect of {delta_dz_linear:+.5f} Dz exists "
                  f"({pct_pooled:.1f}% of -0.11), but is insufficient to explain the observation.")
    else:
        conclusion = "REJECTED"
        detail = (f"The correction changes Dz by only {delta_dz_linear:+.5f}, "
                  f"which is negligible compared to the observed -0.11.")

    print(f"\nCONCLUSION: Hypothesis 4 is {conclusion}.")
    print(f"  {detail}")

    if edge_results:
        print(f"\n5. Edge cases (window size dependence):")
        for r in edge_results:
            print(f"   t_count={r['t_count']:2d}: "
                  f"delta_Dz={r['delta_dz']:+.5f}, "
                  f"mean_TVD={r['mean_tvd']:.4f}")


if __name__ == "__main__":
    main()
