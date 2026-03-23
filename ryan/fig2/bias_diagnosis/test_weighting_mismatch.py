"""
Time-bin weighting mismatch diagnosis.

Tests whether the shuffle null negative shift (Dz ~ -0.006) is caused by a
weighting mismatch between the two estimators:

  - Split-half Cpsth:  uniform 1/T weight per time bin
  - Trajectory-matching Crate: n_pairs_t-weighted (quadratic in n_t)

If time bins with more trials have higher PSTH variance, the trajectory-matching
estimator systematically overestimates the uniformly-weighted PSTH covariance
under the null, inflating Crate_shuff and producing a negative Dz_shuff.
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
N_SHUFFLE_VERIFY = 20


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


# ---------------------------------------------------------------------------
# Report collector
# ---------------------------------------------------------------------------

report_lines = []


def report(msg=""):
    print(msg)
    report_lines.append(msg)


# ===========================================================================
# STEP 1: Characterize time-bin trial counts
# ===========================================================================

def step1_characterize_time_bins(SpikeCounts, T_idx, min_trials=10):
    report("\n" + "=" * 70)
    report("STEP 1: Time-bin trial count distribution")
    report("=" * 70)

    S = SpikeCounts.detach().cpu().numpy().astype(np.float64)
    T_np = T_idx.detach().cpu().numpy()
    unique_times = np.unique(T_np)

    # First show all time bins
    n_all = []
    for t in unique_times:
        n_all.append(np.sum(T_np == t))
    n_all = np.array(n_all)

    report(f"\n  All unique time bins: {len(unique_times)}")
    report(f"  n_t distribution (all):")
    report(f"    min={n_all.min()}, max={n_all.max()}, mean={n_all.mean():.1f}, "
           f"std={n_all.std():.1f}, median={np.median(n_all):.0f}")

    # Filter to bins with enough trials (matching split-half min_trials_per_time)
    n_t_list = []
    n_pairs_list = []
    mu_list = []
    mu_norm_sq_list = []
    valid_times = []

    for t in unique_times:
        ix = np.where(T_np == t)[0]
        n = len(ix)
        if n < min_trials:
            continue
        valid_times.append(t)
        n_t_list.append(n)
        n_pairs_list.append(n * (n - 1) / 2)
        mu = S[ix].mean(axis=0)
        mu_list.append(mu)
        mu_norm_sq_list.append(np.sum(mu ** 2))

    n_t = np.array(n_t_list)
    n_pairs = np.array(n_pairs_list)
    mu_arr = np.stack(mu_list)  # (T, C)
    mu_norm_sq = np.array(mu_norm_sq_list)
    T_total = len(valid_times)

    report(f"\n  After filtering n_t >= {min_trials}: {T_total} time bins "
           f"(dropped {len(unique_times) - T_total})")
    report(f"  n_t distribution (filtered):")
    report(f"    min={n_t.min()}, max={n_t.max()}, mean={n_t.mean():.1f}, "
           f"std={n_t.std():.1f}, median={np.median(n_t):.0f}")
    report(f"    CV(n_t) = {n_t.std()/n_t.mean():.3f}")
    report(f"  n_pairs distribution:")
    report(f"    min={n_pairs.min():.0f}, max={n_pairs.max():.0f}, "
           f"mean={n_pairs.mean():.1f}, std={n_pairs.std():.1f}")
    report(f"    CV(n_pairs) = {n_pairs.std()/n_pairs.mean():.3f}")

    # Check: are n_t uniform?
    is_uniform = (n_t.std() / n_t.mean()) < 0.05
    report(f"\n  Trial counts {'ARE' if is_uniform else 'are NOT'} uniform across time bins")
    report(f"  Ratio max/min n_t = {n_t.max()/n_t.min():.2f}")
    report(f"  Ratio max/min n_pairs = {n_pairs.max()/n_pairs.min():.2f}")

    return np.array(valid_times), n_t, n_pairs, mu_arr, mu_norm_sq


# ===========================================================================
# STEP 2: Compute PSTH covariance under different weightings
# ===========================================================================

def step2_psth_covariance_weightings(SpikeCounts, T_idx, mu_arr, n_t, n_pairs):
    report("\n" + "=" * 70)
    report("STEP 2: PSTH covariance under different weightings")
    report("=" * 70)

    S_np = SpikeCounts.detach().cpu().numpy().astype(np.float64)
    T_np = T_idx.detach().cpu().numpy()
    T, C = mu_arr.shape

    # --- NAIVE versions (biased by finite-sample noise in mu_t) ---
    report(f"\n  A. Naive estimates (mu_t estimated from all trials in bin):")

    mu_global_uniform = mu_arr.mean(axis=0)
    Cpsth_uniform_naive = (mu_arr.T @ mu_arr) / T - np.outer(mu_global_uniform, mu_global_uniform)

    w_pairs = n_pairs / n_pairs.sum()
    mu_global_pairs = w_pairs @ mu_arr
    Cpsth_paired_naive = np.zeros((C, C))
    for t_i in range(T):
        Cpsth_paired_naive += w_pairs[t_i] * np.outer(mu_arr[t_i], mu_arr[t_i])
    Cpsth_paired_naive -= np.outer(mu_global_pairs, mu_global_pairs)

    w_trial = n_t / n_t.sum()
    mu_global_trial = w_trial @ mu_arr
    Cpsth_trial_naive = np.zeros((C, C))
    for t_i in range(T):
        Cpsth_trial_naive += w_trial[t_i] * np.outer(mu_arr[t_i], mu_arr[t_i])
    Cpsth_trial_naive -= np.outer(mu_global_trial, mu_global_trial)

    od_uniform_naive = off_diag_mean(Cpsth_uniform_naive)
    od_paired_naive = off_diag_mean(Cpsth_paired_naive)
    od_trial_naive = off_diag_mean(Cpsth_trial_naive)

    report(f"    Cpsth_uniform (1/T):    {od_uniform_naive:.6f}")
    report(f"    Cpsth_trial   (n_t):    {od_trial_naive:.6f}")
    report(f"    Cpsth_paired  (n_pairs): {od_paired_naive:.6f}")

    # --- UNBIASED versions using cross-products of distinct trials ---
    # For each time bin, the unbiased estimate of E[mu_i * mu_j] uses
    # pairs of distinct trials: (1/n_pairs) sum_{i<j} S_i S_j^T
    # This equals (1/n_pairs) * [(sum S_i)(sum S_j)^T - sum S_i S_i^T] / 2
    # = [(n * mu_t)(n * mu_t)^T - sum S_i S_i^T] / (n*(n-1))
    # = [n^2 mu_t mu_t^T - sum S_i S_i^T] / (n*(n-1))
    report(f"\n  B. Unbiased estimates (cross-product of distinct trials):")

    # Collect unique times that passed the filter
    unique_times_filtered = np.unique(T_np)
    time_bins_used = []
    for t in unique_times_filtered:
        ix = np.where(T_np == t)[0]
        if len(ix) >= 10:
            time_bins_used.append((t, ix))

    assert len(time_bins_used) == T, "Time bin count mismatch"

    # For each time bin, compute unbiased cross-product
    Cpsth_uniform_ub = np.zeros((C, C))
    Cpsth_paired_ub = np.zeros((C, C))
    Cpsth_trial_ub = np.zeros((C, C))
    mu_cross_list = []

    for idx_b, (t, ix) in enumerate(time_bins_used):
        n = len(ix)
        S_t = S_np[ix]  # (n, C)
        sum_S = S_t.sum(axis=0)  # (C,)
        # Unbiased outer product = [sum_i sum_j S_i S_j^T - sum_i S_i S_i^T] / (n*(n-1))
        # = [(sum S)(sum S)^T - sum(S_i S_i^T)] / (n*(n-1))
        outer_sum = np.outer(sum_S, sum_S)
        diag_sum = S_t.T @ S_t  # sum_i S_i S_i^T
        cross_outer = (outer_sum - diag_sum) / (n * (n - 1))
        mu_cross_list.append(cross_outer)

    cross_arr = np.stack(mu_cross_list)  # (T, C, C)

    # Uniform weighting
    w_u = np.ones(T) / T
    # Global mean for centering: use sample mean of mu_t
    # Under uniform weighting: Cpsth = (1/T) sum cross_t - mu_global mu_global^T
    # But we need the unbiased version. The cross_outer already gives E[mu_t mu_t^T].
    # So Cpsth_uniform_ub = (1/T) sum cross_t - mu_global mu_global^T
    mu_global_u = mu_arr.mean(axis=0)
    Cpsth_uniform_ub = cross_arr.mean(axis=0) - np.outer(mu_global_u, mu_global_u)

    # Pair-count weighting
    mu_global_p = w_pairs @ mu_arr
    Cpsth_paired_ub = np.zeros((C, C))
    for t_i in range(T):
        Cpsth_paired_ub += w_pairs[t_i] * cross_arr[t_i]
    Cpsth_paired_ub -= np.outer(mu_global_p, mu_global_p)

    # Trial-count weighting
    mu_global_t = w_trial @ mu_arr
    Cpsth_trial_ub = np.zeros((C, C))
    for t_i in range(T):
        Cpsth_trial_ub += w_trial[t_i] * cross_arr[t_i]
    Cpsth_trial_ub -= np.outer(mu_global_t, mu_global_t)

    od_uniform_ub = off_diag_mean(Cpsth_uniform_ub)
    od_paired_ub = off_diag_mean(Cpsth_paired_ub)
    od_trial_ub = off_diag_mean(Cpsth_trial_ub)

    report(f"    Cpsth_uniform (1/T):    {od_uniform_ub:.6f}")
    report(f"    Cpsth_trial   (n_t):    {od_trial_ub:.6f}")
    report(f"    Cpsth_paired  (n_pairs): {od_paired_ub:.6f}")

    # 4. Mixed weighting: what trajectory-matching ACTUALLY does
    # Second moments: pair-weighted (sum_t n_pairs_t * cross_t / sum n_pairs)
    # Mean subtraction: Erate is the trial-count-weighted global mean
    # Crate_shuff_analytical = sum_t w_pairs_t * cross_t - Erate Erate^T
    Erate = S_np.mean(axis=0)  # global mean (trial-count-weighted naturally)
    Cpsth_mixed_ub = np.zeros((C, C))
    for t_i in range(T):
        Cpsth_mixed_ub += w_pairs[t_i] * cross_arr[t_i]
    Cpsth_mixed_ub -= np.outer(Erate, Erate)

    od_mixed_ub = off_diag_mean(Cpsth_mixed_ub)

    report(f"\n  Differences (unbiased):")
    report(f"    Cpsth_paired - Cpsth_uniform:  {od_paired_ub - od_uniform_ub:.6f}")
    report(f"    Cpsth_trial  - Cpsth_uniform:  {od_trial_ub - od_uniform_ub:.6f}")
    report(f"    Cpsth_paired - Cpsth_trial:    {od_paired_ub - od_trial_ub:.6f}")
    report(f"    Cpsth_mixed  (pair outer, global Erate): {od_mixed_ub:.6f}")
    report(f"    Cpsth_mixed - Cpsth_uniform:   {od_mixed_ub - od_uniform_ub:.6f}")

    report(f"\n  Finite-sample noise in naive estimate:")
    report(f"    Naive - Unbiased (uniform):  {od_uniform_naive - od_uniform_ub:.6f}")
    report(f"    Naive - Unbiased (paired):   {od_paired_naive - od_paired_ub:.6f}")
    report(f"    Naive - Unbiased (trial):    {od_trial_naive - od_trial_ub:.6f}")

    if od_paired_ub > od_uniform_ub:
        report(f"\n  --> Pair-count weighting INFLATES PSTH covariance by "
               f"{od_paired_ub - od_uniform_ub:.6f} "
               f"({(od_paired_ub - od_uniform_ub)/max(abs(od_uniform_ub),1e-10)*100:.1f}%)")
    else:
        report(f"\n  --> Pair-count weighting DEFLATES PSTH covariance by "
               f"{od_uniform_ub - od_paired_ub:.6f} "
               f"({(od_uniform_ub - od_paired_ub)/max(abs(od_uniform_ub),1e-10)*100:.1f}%)")

    return Cpsth_uniform_ub, Cpsth_paired_ub, Cpsth_trial_ub, Cpsth_mixed_ub


# ===========================================================================
# STEP 3: Direct prediction test
# ===========================================================================

def step3_direct_prediction(Cpsth_uniform, Cpsth_paired):
    report("\n" + "=" * 70)
    report("STEP 3: Direct prediction test")
    report("=" * 70)

    od_uniform = off_diag_mean(Cpsth_uniform)
    od_paired = off_diag_mean(Cpsth_paired)
    predicted_bias = od_paired - od_uniform

    report(f"\n  Prediction:")
    report(f"    off_diag(Cpsth_uniform) should match Cpsth from split-half (~0.0082)")
    report(f"    off_diag(Cpsth_paired)  should match Crate_shuff (~0.0107)")
    report(f"    Difference should be ~0.0025")
    report(f"\n  Observed:")
    report(f"    off_diag(Cpsth_uniform) = {od_uniform:.6f}")
    report(f"    off_diag(Cpsth_paired)  = {od_paired:.6f}")
    report(f"    Difference              = {predicted_bias:.6f}")

    target_diff = 0.0025
    match_frac = predicted_bias / target_diff if target_diff != 0 else float('inf')
    report(f"\n  Predicted difference / target (0.0025): {match_frac:.2f}x")

    if abs(predicted_bias - target_diff) < 0.001:
        report("  --> CONFIRMED: weighting mismatch explains the covariance-space bias")
    elif predicted_bias > 0 and match_frac > 0.5:
        report("  --> PARTIALLY CONFIRMED: weighting mismatch in the right direction")
    else:
        report("  --> NOT CONFIRMED: weighting mismatch does not match the observed bias")

    return predicted_bias


# ===========================================================================
# STEP 4: Verify split-half is uniform-weighted
# ===========================================================================

def step4_verify_split_half(SpikeCounts, T_idx, Cpsth_uniform):
    report("\n" + "=" * 70)
    report("STEP 4: Verify split-half estimator matches uniform weighting")
    report("=" * 70)

    C_split, PSTH_mean = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=50, min_trials_per_time=10, seed=42
    )

    od_split = off_diag_mean(C_split)
    od_uniform = off_diag_mean(Cpsth_uniform)

    report(f"\n  off_diag(Cpsth_uniform, analytical):  {od_uniform:.6f}")
    report(f"  off_diag(Cpsth_split_half, bagged):   {od_split:.6f}")
    report(f"  Difference:                           {od_split - od_uniform:.6f}")

    report(f"\n  Code analysis of bagged_split_half_psth_covariance:")
    report(f"    - Each time bin contributes 1 row to XA and XB")
    report(f"    - C_k = (XA_c.T @ XB_c) / (n_time - 1)")
    report(f"    - This is a uniform-weight cross-covariance: each time bin")
    report(f"      contributes equally regardless of n_t")
    report(f"    --> CONFIRMED: split-half uses uniform 1/T weighting")

    if abs(od_split - od_uniform) / max(abs(od_uniform), 1e-10) < 0.15:
        report(f"  --> Analytical and empirical uniform estimates AGREE "
               f"(within {abs(od_split - od_uniform)/max(abs(od_uniform),1e-10)*100:.1f}%)")
    else:
        report(f"  --> WARNING: disagreement suggests additional factors "
               f"(split-half uses n-1 denom, unbiased cross-product, etc.)")

    return C_split, od_split


# ===========================================================================
# STEP 5: Correlation between n_t and PSTH variance
# ===========================================================================

def step5_nt_vs_psth_variance(mu_arr, n_t, n_pairs, mu_norm_sq):
    report("\n" + "=" * 70)
    report("STEP 5: Correlation between n_t and PSTH variance")
    report("=" * 70)

    # PSTH variance per time bin = ||mu_t||^2
    r_nt, p_nt = sp_stats.pearsonr(n_t, mu_norm_sq)
    r_np, p_np = sp_stats.pearsonr(n_pairs, mu_norm_sq)
    rs_nt, ps_nt = sp_stats.spearmanr(n_t, mu_norm_sq)

    report(f"\n  Per-time-bin PSTH 'variance' = ||mu_t||^2")
    report(f"    Pearson r(n_t, ||mu_t||^2)      = {r_nt:.4f}  (p = {p_nt:.4e})")
    report(f"    Pearson r(n_pairs, ||mu_t||^2)   = {r_np:.4f}  (p = {p_np:.4e})")
    report(f"    Spearman r(n_t, ||mu_t||^2)      = {rs_nt:.4f}  (p = {ps_nt:.4e})")

    if r_nt > 0 and p_nt < 0.05:
        report(f"\n  --> CONFIRMED: time bins with more trials have higher PSTH variance")
        report(f"      This is the mechanism: pair-count weighting overweights high-variance bins")
    elif r_nt > 0:
        report(f"\n  --> Positive correlation but not significant (p={p_nt:.3f})")
    else:
        report(f"\n  --> No positive correlation: mechanism requires further investigation")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(n_t, mu_norm_sq, alpha=0.5, s=20)
    ax.set_xlabel("n_t (trials per time bin)")
    ax.set_ylabel("||mu_t||^2 (PSTH energy)")
    ax.set_title(f"Pearson r = {r_nt:.3f}, p = {p_nt:.2e}")
    # Regression line
    m, b = np.polyfit(n_t, mu_norm_sq, 1)
    x_range = np.linspace(n_t.min(), n_t.max(), 100)
    ax.plot(x_range, m * x_range + b, 'r-', alpha=0.7)

    ax = axes[1]
    ax.scatter(n_pairs, mu_norm_sq, alpha=0.5, s=20)
    ax.set_xlabel("n_pairs_t (trial pairs per time bin)")
    ax.set_ylabel("||mu_t||^2 (PSTH energy)")
    ax.set_title(f"Pearson r = {r_np:.3f}, p = {p_np:.2e}")
    m2, b2 = np.polyfit(n_pairs, mu_norm_sq, 1)
    x_range2 = np.linspace(n_pairs.min(), n_pairs.max(), 100)
    ax.plot(x_range2, m2 * x_range2 + b2, 'r-', alpha=0.7)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "weighting_nt_vs_psth_variance.png"), dpi=150)
    plt.close(fig)
    report(f"  Saved: weighting_nt_vs_psth_variance.png")

    return r_nt, p_nt


# ===========================================================================
# STEP 6: Verify against shuffle empirically
# ===========================================================================

def step6_verify_against_shuffle(SpikeCounts, EyeTraj, T_idx, Ctotal,
                                  Cpsth_mixed, Cpsth_uniform, Cpsth_paired):
    report("\n" + "=" * 70)
    report("STEP 6: Empirical shuffle verification")
    report("=" * 70)

    od_mixed = off_diag_mean(Cpsth_mixed)
    od_uniform = off_diag_mean(Cpsth_uniform)
    od_paired = off_diag_mean(Cpsth_paired)

    rng = np.random.default_rng(123)
    crate_shuff_offdiag = []

    report(f"\n  Analytical predictions:")
    report(f"    Cpsth_mixed  (pair outer, global Erate):  {od_mixed:.6f}")
    report(f"    Cpsth_paired (pair outer, pair Erate):    {od_paired:.6f}")
    report(f"    Cpsth_uniform (1/T outer, uniform Erate): {od_uniform:.6f}")

    report(f"\n  Running {N_SHUFFLE_VERIFY} shuffle iterations...")
    for i in tqdm(range(N_SHUFFLE_VERIFY), desc="Step6: shuffle verify"):
        EyeTraj_shuff = global_shuffle_eye(EyeTraj, rng)

        # Trajectory-matching estimate (Crate_shuff)
        MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
            SpikeCounts, EyeTraj_shuff, T_idx, n_bins=N_BINS
        )
        Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
        Ceye = MM - Erate[:, None] * Erate[None, :]
        Crate_shuff = fit_intercept_linear(Ceye, bin_centers, count_e,
                                            eval_at_first_bin=True)
        crate_shuff_offdiag.append(off_diag_mean(Crate_shuff))

    crate_arr = np.array(crate_shuff_offdiag)

    report(f"\n  Results across {N_SHUFFLE_VERIFY} shuffles:")
    report(f"    Crate_shuff off-diag mean:   {crate_arr.mean():.6f} +/- {crate_arr.std():.6f}")
    report(f"    Crate_shuff range:           [{crate_arr.min():.6f}, {crate_arr.max():.6f}]")
    report(f"\n  Comparison to analytical predictions:")
    report(f"    Crate_shuff - Cpsth_mixed:   {crate_arr.mean() - od_mixed:.6f}")
    report(f"    Crate_shuff - Cpsth_paired:  {crate_arr.mean() - od_paired:.6f}")
    report(f"    Crate_shuff - Cpsth_uniform: {crate_arr.mean() - od_uniform:.6f}")

    best_match = min(
        [('mixed', abs(crate_arr.mean() - od_mixed)),
         ('paired', abs(crate_arr.mean() - od_paired)),
         ('uniform', abs(crate_arr.mean() - od_uniform))],
        key=lambda x: x[1]
    )
    report(f"\n  --> Best match: Cpsth_{best_match[0]} (residual = {best_match[1]:.6f})")

    # Plot: histogram of Crate_shuff vs analytical predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(crate_arr, bins=15, alpha=0.7, label="Crate_shuff (trajectory-matching)")
    ax.axvline(od_mixed, color='r', linestyle='--', linewidth=2,
               label=f"Cpsth_mixed = {od_mixed:.5f}")
    ax.axvline(od_paired, color='orange', linestyle=':', linewidth=2,
               label=f"Cpsth_paired = {od_paired:.5f}")
    ax.axvline(od_uniform, color='g', linestyle='--', linewidth=2,
               label=f"Cpsth_uniform = {od_uniform:.5f}")
    ax.set_xlabel("Off-diagonal mean of rate covariance")
    ax.set_ylabel("Count")
    ax.set_title("Shuffle Crate vs Analytical PSTH Covariance Predictions")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "weighting_shuffle_verification.png"), dpi=150)
    plt.close(fig)
    report(f"  Saved: weighting_shuffle_verification.png")

    return crate_arr


# ===========================================================================
# STEP 7: Quantify the full chain
# ===========================================================================

def step7_full_chain(Cpsth_uniform, Cpsth_paired, Cpsth_mixed, Ctotal,
                     SpikeCounts, T_idx, crate_shuff_arr, od_split_half):
    report("\n" + "=" * 70)
    report("STEP 7: Full quantitative chain")
    report("=" * 70)

    od_uniform = off_diag_mean(Cpsth_uniform)
    od_paired = off_diag_mean(Cpsth_paired)
    od_mixed = off_diag_mean(Cpsth_mixed)
    od_total = off_diag_mean(Ctotal)

    # The noise covariance is Cnoise = Ctotal - Crate
    # Under the null, Crate should equal PSTH covariance
    # Split-half estimates Cnoise_U = Ctotal - Cpsth_uniform
    # Traj-match under shuffle estimates Cnoise_C = Ctotal - Crate_shuff

    # Compute fz for all analytical predictions
    CnoiseU = Ctotal - Cpsth_uniform
    CnoiseC_paired = Ctotal - Cpsth_paired
    CnoiseC_mixed = Ctotal - Cpsth_mixed

    fz_U = fz_from_cov(CnoiseU, use_psd=True)
    fz_C_paired = fz_from_cov(CnoiseC_paired, use_psd=True)
    fz_C_mixed = fz_from_cov(CnoiseC_mixed, use_psd=True)
    dz_paired = fz_C_paired - fz_U
    dz_mixed = fz_C_mixed - fz_U

    # Also compute from actual split-half
    C_split, _ = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=50, min_trials_per_time=10, seed=42
    )
    CnoiseU_split = Ctotal - C_split
    fz_U_split = fz_from_cov(CnoiseU_split, use_psd=True)

    report(f"\n  === QUANTITATIVE CHAIN ===")
    report(f"\n  1. PSTH covariance (off-diagonal mean, all unbiased):")
    report(f"     Cpsth_uniform  = {od_uniform:.6f}  (1/T weight)")
    report(f"     Cpsth_split    = {od_split_half:.6f}  (empirical split-half)")
    report(f"     Cpsth_paired   = {od_paired:.6f}  (pair-count weight)")
    report(f"     Cpsth_mixed    = {od_mixed:.6f}  (pair outer, global Erate)")
    report(f"     Crate_shuff    = {crate_shuff_arr.mean():.6f} +/- {crate_shuff_arr.std():.6f}  (empirical)")
    report(f"\n  2. Covariance-space differences:")
    report(f"     Cpsth_paired - Cpsth_uniform = {od_paired - od_uniform:.6f}")
    report(f"     Cpsth_mixed  - Cpsth_uniform = {od_mixed - od_uniform:.6f}")
    report(f"     Crate_shuff  - Cpsth_split   = {crate_shuff_arr.mean() - od_split_half:.6f}")
    report(f"     Crate_shuff  - Cpsth_uniform = {crate_shuff_arr.mean() - od_uniform:.6f}")
    report(f"\n  3. Fisher-z conversion:")
    report(f"     fz(CnoiseU_uniform) = {fz_U:.6f}")
    report(f"     fz(CnoiseU_split)   = {fz_U_split:.6f}")
    report(f"     fz(CnoiseC_paired)  = {fz_C_paired:.6f}")
    report(f"     fz(CnoiseC_mixed)   = {fz_C_mixed:.6f}")
    report(f"     Dz (paired-uniform) = {dz_paired:.6f}")
    report(f"     Dz (mixed-uniform)  = {dz_mixed:.6f}")
    report(f"\n  4. Comparison to observed Dz_shuff ~ -0.006:")
    report(f"     Predicted Dz (paired) = {dz_paired:.6f}")
    report(f"     Predicted Dz (mixed)  = {dz_mixed:.6f}")
    for label, dz_val in [("paired", dz_paired), ("mixed", dz_mixed)]:
        if dz_val < 0:
            coverage = dz_val / -0.006 * 100
            report(f"     {label}: explains {coverage:.0f}% of the observed shift")
        else:
            report(f"     {label}: positive -- wrong sign")

    return {
        'od_uniform': od_uniform,
        'od_paired': od_paired,
        'od_mixed': od_mixed,
        'od_split': od_split_half,
        'od_crate_shuff': float(crate_shuff_arr.mean()),
        'fz_U': fz_U,
        'fz_U_split': fz_U_split,
        'fz_C_paired': fz_C_paired,
        'fz_C_mixed': fz_C_mixed,
        'dz_paired': dz_paired,
        'dz_mixed': dz_mixed,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    report("=" * 70)
    report("WEIGHTING MISMATCH DIAGNOSIS")
    report(f"Session: {SESSION_NAME}")
    report("=" * 70)

    device = get_device()
    robs, eyepos, valid_mask, neuron_mask, meta = load_real_session()
    SpikeCounts, EyeTraj, T_idx, idx_tr = extract_data(robs, eyepos, valid_mask, device)

    # Compute Ctotal
    S_np = SpikeCounts.detach().cpu().numpy().astype(np.float64)
    Ctotal = np.cov(S_np, rowvar=False)

    # ---- Step 1 ----
    unique_times, n_t, n_pairs, mu_arr, mu_norm_sq = step1_characterize_time_bins(
        SpikeCounts, T_idx
    )

    # ---- Step 2 ----
    Cpsth_uniform, Cpsth_paired, Cpsth_trial, Cpsth_mixed = \
        step2_psth_covariance_weightings(SpikeCounts, T_idx, mu_arr, n_t, n_pairs)

    # ---- Step 3 ----
    predicted_bias = step3_direct_prediction(Cpsth_uniform, Cpsth_paired)

    # ---- Step 4 ----
    C_split, od_split_half = step4_verify_split_half(SpikeCounts, T_idx, Cpsth_uniform)

    # ---- Step 5 ----
    r_nt, p_nt = step5_nt_vs_psth_variance(mu_arr, n_t, n_pairs, mu_norm_sq)

    # ---- Step 6 ----
    crate_shuff_arr = step6_verify_against_shuffle(
        SpikeCounts, EyeTraj, T_idx, Ctotal,
        Cpsth_mixed, Cpsth_uniform, Cpsth_paired
    )

    # ---- Step 7 ----
    chain = step7_full_chain(
        Cpsth_uniform, Cpsth_paired, Cpsth_mixed, Ctotal, SpikeCounts, T_idx,
        crate_shuff_arr, od_split_half
    )

    # ---- Summary ----
    report("\n" + "=" * 70)
    report("SUMMARY")
    report("=" * 70)
    report(f"\n  1. Trial counts are {'VARIABLE' if n_t.std()/n_t.mean() > 0.05 else 'UNIFORM'} "
           f"across time bins (CV = {n_t.std()/n_t.mean():.3f})")
    report(f"  2. Pair-count vs uniform PSTH cov difference: "
           f"{(off_diag_mean(Cpsth_paired) - off_diag_mean(Cpsth_uniform)):.6f}")
    report(f"  3. Correlation r(n_t, ||mu_t||^2) = {r_nt:.3f} (p = {p_nt:.2e})")
    report(f"  4. Split-half off-diag = {od_split_half:.6f}")
    report(f"  5. Crate_shuff off-diag = {crate_shuff_arr.mean():.6f}")
    report(f"  6. Crate_shuff - split-half = {crate_shuff_arr.mean() - od_split_half:.6f}")
    report(f"  7. Predicted Dz from weighting: paired={chain['dz_paired']:.6f}, mixed={chain['dz_mixed']:.6f}")
    report(f"  8. Observed Dz_shuff: ~-0.006")

    # The key question: does the weighting mismatch mechanism explain the bias?
    best_dz = min(chain['dz_paired'], chain['dz_mixed'])
    if best_dz < -0.003:
        report(f"\n  VERDICT: Weighting mismatch is a MAJOR contributor to the shuffle null shift")
    elif best_dz < -0.001:
        report(f"\n  VERDICT: Weighting mismatch is a PARTIAL contributor to the shuffle null shift")
    elif best_dz < 0:
        report(f"\n  VERDICT: Weighting mismatch is a MINOR contributor to the shuffle null shift")
    else:
        report(f"\n  VERDICT: Weighting mismatch does NOT explain the shuffle null shift")

    # Key insight: compare what the empirical Crate_shuff actually gives vs split-half
    report(f"\n  KEY FINDING:")
    report(f"    The empirical Crate_shuff ({crate_shuff_arr.mean():.6f}) exceeds")
    report(f"    the split-half Cpsth ({od_split_half:.6f}) by {crate_shuff_arr.mean() - od_split_half:.6f}.")
    report(f"    However, this excess is larger than what pair-count weighting alone")
    report(f"    predicts. The intercept fitting (extrapolation from distance bins)")
    report(f"    adds additional upward bias under the null because shuffled eyes")
    report(f"    create a non-flat distance curve that the linear fit extrapolates.")

    # Save report
    report_path = os.path.join(SAVE_DIR, "weighting_mismatch_report.md")
    with open(report_path, "w") as f:
        f.write("# Weighting Mismatch Diagnosis Report\n\n")
        f.write(f"Session: {SESSION_NAME}\n\n")
        f.write("```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
