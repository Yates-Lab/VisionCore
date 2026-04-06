"""
Covariance decomposition primitives for the Law of Total Covariance (LOTC).

Flat functions refactored from DualWindowAnalysis class. Each function takes
explicit arguments instead of reading from self.
"""
import numpy as np
import torch
from tqdm import tqdm

from VisionCore.subspace import project_to_psd  # re-export for convenience


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cov_to_corr(C, min_var=1e-3):
    """
    Convert covariance matrix to correlation matrix.

    Returns NaN for neurons with variance below min_var.
    Diagonal is set to 0 by convention (for noise correlation analysis).

    Parameters
    ----------
    C : ndarray (N, N)
        Covariance matrix.
    min_var : float
        Minimum variance threshold. Neurons below this get NaN correlations.

    Returns
    -------
    R : ndarray (N, N)
        Correlation matrix with diagonal = 0.
    """
    C = np.asarray(C, dtype=np.float64)
    variances = np.diag(C)

    # Neurons with variance below threshold get NaN std
    valid_mask = variances > min_var
    std_devs = np.full_like(variances, np.nan)
    std_devs[valid_mask] = np.sqrt(variances[valid_mask])

    # Outer product of std devs — NaN propagates correctly
    outer_std = np.outer(std_devs, std_devs)

    R = C / outer_std
    R = np.clip(R, -1.0, 1.0)

    # NaN entries stay NaN after clip (numpy behavior)
    # But we need to restore NaN where outer_std was NaN
    R[~np.isfinite(R)] = np.nan

    # Set diagonal to 0
    np.fill_diagonal(R, 0.0)

    return R


def pava_nonincreasing(y, w, eps=1e-12):
    """
    Weighted isotonic regression (Pool-Adjacent-Violators Algorithm).

    Enforces the fitted sequence to be non-increasing.

    Parameters
    ----------
    y : array-like
        Response values.
    w : array-like
        Weights for each observation.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    yhat : ndarray
        Isotonic fit (non-increasing).
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    means = []
    weights = []
    starts = []
    ends = []
    for i in range(len(y)):
        means.append(y[i])
        weights.append(w[i])
        starts.append(i)
        ends.append(i)
        while len(means) >= 2 and means[-2] < means[-1]:
            w_new = weights[-2] + weights[-1]
            m_new = (weights[-2] * means[-2] + weights[-1] * means[-1]) / (w_new + eps)
            means[-2] = m_new
            weights[-2] = w_new
            ends[-2] = ends[-1]
            means.pop()
            weights.pop()
            starts.pop()
            ends.pop()
    yhat = np.empty_like(y)
    for m, s, e in zip(means, starts, ends):
        yhat[s:e + 1] = m
    return yhat


def get_upper_triangle(C):
    """Extract upper-triangle values (k=1 diagonal offset) from a square matrix."""
    rows, cols = np.triu_indices_from(C, k=1)
    return C[rows, cols]


def extract_valid_segments(valid_mask, min_len_bins=36):
    """
    Find contiguous valid segments in a (n_trials, n_time) boolean mask.

    Parameters
    ----------
    valid_mask : ndarray (n_trials, n_time)
        Boolean mask of valid time bins per trial.
    min_len_bins : int
        Minimum segment length to keep.

    Returns
    -------
    segments : list of (trial, start, stop) tuples
    """
    mask = np.asarray(valid_mask, dtype=bool)
    n_trials = mask.shape[0]
    segments = []
    for tr in range(n_trials):
        padded = np.concatenate(([False], mask[tr], [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        stops = np.where(diffs == -1)[0]
        for s, e in zip(starts, stops):
            if (e - s) >= min_len_bins:
                segments.append((tr, s, e))
    return segments


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_windows(robs, eyepos, segments, t_count, t_hist, device="cuda"):
    """
    Extract spike count windows and eye trajectories from segments.

    Parameters
    ----------
    robs : torch.Tensor (n_trials, n_time, n_cells)
        Spike observations.
    eyepos : torch.Tensor (n_trials, n_time, 2)
        Eye positions.
    segments : list of (trial, start, stop)
        Valid contiguous segments.
    t_count : int
        Number of bins in the counting window.
    t_hist : int
        Number of bins in the history window for trajectory similarity.
    device : str
        Torch device.

    Returns
    -------
    SpikeCounts : torch.Tensor (N, n_cells)
    EyeTraj : torch.Tensor (N, total_len, 2)
    T_idx : torch.Tensor (N,)
    idx_tr : torch.Tensor (N,)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    total_len = t_hist + t_count
    trial_indices, time_indices = [], []

    for (tr, start, stop) in segments:
        if (stop - start) < total_len:
            continue
        t_starts = np.arange(start, stop - total_len + 1, t_count)
        trial_indices.extend([tr] * len(t_starts))
        time_indices.extend(t_starts)

    if len(trial_indices) == 0:
        return None, None, None, None

    idx_tr = torch.tensor(trial_indices, device=device, dtype=torch.long)
    idx_t0 = torch.tensor(time_indices, device=device, dtype=torch.long)

    # Gather eye trajectory (full history + count window)
    offsets = torch.arange(total_len, device=device).unsqueeze(0)
    gather_t = idx_t0.unsqueeze(1) + offsets
    gather_tr = idx_tr.unsqueeze(1).expand(-1, total_len)
    EyeTraj = eyepos[gather_tr, gather_t, :]

    # Gather spikes (count window only)
    spike_offsets = torch.arange(t_hist, total_len, device=device).unsqueeze(0)
    gather_t_spk = idx_t0.unsqueeze(1) + spike_offsets
    gather_tr_spk = idx_tr.unsqueeze(1).expand(-1, t_count)
    S_raw = robs[gather_tr_spk, gather_t_spk, :]
    SpikeCounts = torch.sum(S_raw, dim=1)

    T_idx = idx_t0 + t_hist

    return SpikeCounts, EyeTraj, T_idx, idx_tr


# ---------------------------------------------------------------------------
# Second moment estimation
# ---------------------------------------------------------------------------

def compute_eye_distances(EyeTraj):
    """
    Compute RMS eye trajectory distance matrix.

    Parameters
    ----------
    EyeTraj : torch.Tensor (N, T, 2)

    Returns
    -------
    dist_matrix : torch.Tensor (N, N)
    """
    N, T, _ = EyeTraj.shape
    EyeFlat = EyeTraj.reshape(N, -1)
    return torch.cdist(EyeFlat, EyeFlat) / np.sqrt(T)


def bin_pairs_by_distance(dist_matrix, n_bins):
    """
    Compute percentile-based bin edges from upper triangle of distance matrix.

    Parameters
    ----------
    dist_matrix : torch.Tensor (N, N)
    n_bins : int

    Returns
    -------
    bin_edges : ndarray
    bin_centers : ndarray
    """
    N = dist_matrix.shape[0]
    i, j = torch.triu_indices(N, N, offset=1)
    dist = dist_matrix[i, j]
    bin_edges = np.percentile(dist.cpu().numpy(), np.arange(0, 100, 100 / (n_bins + 1)))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_edges, bin_centers


def compute_conditional_second_moments(SpikeCounts, EyeTraj, T_idx, n_bins=25):
    """
    Compute E[SS^T | distance_bin] using time-matched distinct-trial pairs.

    Parameters
    ----------
    SpikeCounts : torch.Tensor (N, C)
    EyeTraj : torch.Tensor (N, T, 2)
    T_idx : torch.Tensor (N,)
    n_bins : int or array-like
        Number of bins, or pre-computed bin edges.

    Returns
    -------
    MM : ndarray (n_bins, C, C) — conditional second moments
    bin_centers : ndarray (n_bins,)
    count_e : ndarray (n_bins,) — pair counts per bin
    bin_edges : ndarray
    """
    N_samples, T, _ = EyeTraj.shape
    device = EyeTraj.device
    C = SpikeCounts.shape[1]

    # Compute distance matrix
    EyeFlat = EyeTraj.reshape(N_samples, -1)
    inv_sqrt_T = 1.0 / torch.sqrt(torch.tensor(float(T), device=device, dtype=EyeTraj.dtype))

    # Bin edges
    if isinstance(n_bins, int):
        i_up, j_up = torch.triu_indices(N_samples, N_samples, offset=1)
        dist_up = torch.cdist(EyeFlat, EyeFlat)[i_up, j_up] * inv_sqrt_T
        bin_edges = np.percentile(dist_up.cpu().numpy(), np.arange(0, 100, 100 / (n_bins + 1)))
    else:
        bin_edges = np.asarray(n_bins)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins_actual = len(bin_edges) - 1

    unique_times = np.unique(T_idx.detach().cpu().numpy())
    bin_edges_t = torch.as_tensor(bin_edges, device=device, dtype=EyeTraj.dtype)

    # Accumulators on CPU
    SS_e_t = torch.zeros((n_bins_actual, C, C), device='cpu', dtype=torch.float64)
    count_e_t = torch.zeros((n_bins_actual,), device='cpu', dtype=torch.long)

    def accumulate_split(valid_idx):
        N = len(valid_idx)
        if N < 2:
            return

        X = EyeTraj[valid_idx]
        S = SpikeCounts[valid_idx]

        ii, jj = torch.triu_indices(N, N, offset=1, device=device)

        Xflat = X.reshape(N, -1)
        D = torch.cdist(Xflat, Xflat) * inv_sqrt_T
        d = D[ii, jj]

        bid = torch.bucketize(d, bin_edges_t, right=False)

        ok = (bid >= 1) & (bid <= n_bins_actual)
        if not ok.any():
            return
        ii = ii[ok]
        jj = jj[ok]
        bid = bid[ok]

        for k in range(1, n_bins_actual + 1):
            mk = (bid == k)
            if not mk.any():
                continue
            Si = S[ii[mk]]
            Sj = S[jj[mk]]
            M = Si.transpose(0, 1).matmul(Sj)
            SS_e_t[k - 1] += M.detach().cpu().to(torch.float64)
            count_e_t[k - 1] += mk.sum().detach().cpu()

    for t in unique_times:
        valid = np.where((T_idx == t).detach().cpu().numpy())[0]
        if len(valid) < 10:
            continue
        accumulate_split(valid)

    SS_e = SS_e_t.numpy()
    count_e = count_e_t.numpy()

    MM = SS_e / count_e[:, None, None]
    MM = 0.5 * (MM + np.swapaxes(MM, -1, -2))

    return MM, bin_centers, count_e, bin_edges


# ---------------------------------------------------------------------------
# Intercept fitting
# ---------------------------------------------------------------------------

def _fit_best_monotonic(y, w):
    """Fit both non-increasing and non-decreasing PAVA, return intercept of better fit."""
    y_decr = pava_nonincreasing(y, w)
    sse_decr = np.sum(w * (y - y_decr) ** 2)

    y_incr = -pava_nonincreasing(-y, w)
    sse_incr = np.sum(w * (y - y_incr) ** 2)

    return y_decr[0] if sse_decr < sse_incr else y_incr[0]


def fit_intercept_pava(Ceye, count_e):
    """
    PAVA-based intercept fitting for each element of the covariance tensor.

    Parameters
    ----------
    Ceye : ndarray (n_bins, n_cells, n_cells)
    count_e : ndarray (n_bins,)

    Returns
    -------
    C_intercept : ndarray (n_cells, n_cells)
    """
    n_bins, n_cells, _ = Ceye.shape
    C_intercept = np.zeros((n_cells, n_cells), dtype=Ceye.dtype)

    for i in range(n_cells):
        # Diagonal: variance must be non-increasing
        y_diag = Ceye[:, i, i]
        yhat = pava_nonincreasing(y_diag, count_e)
        C_intercept[i, i] = yhat[0]

        # Off-diagonals: can be increasing or decreasing
        for j in range(i + 1, n_cells):
            y = Ceye[:, i, j]
            valid = np.isfinite(y)
            if not valid.any():
                C_intercept[i, j] = np.nan
                C_intercept[j, i] = np.nan
                continue
            val = _fit_best_monotonic(y[valid], count_e[valid])
            C_intercept[i, j] = val
            C_intercept[j, i] = val

    return C_intercept


def fit_intercept_linear(Ceye, bin_centers, count_e, d_max=0.4, min_bins=3,
                         eps=1e-8, eval_at_first_bin=True):
    """
    Weighted local linear regression intercept for each (i,j).

    Parameters
    ----------
    Ceye : ndarray (n_bins, n_cells, n_cells)
    bin_centers : ndarray (n_bins,)
    count_e : ndarray (n_bins,)
    d_max : float
        Maximum distance to include in regression.
    min_bins : int
        Minimum number of bins required.
    eps : float
        Numerical stability threshold.
    eval_at_first_bin : bool
        If True, evaluate at first bin center (conservative).
        If False, extrapolate to d=0.

    Returns
    -------
    C_intercept : ndarray (n_cells, n_cells)
    """
    n_bins, n_cells, _ = Ceye.shape
    C_intercept = np.full((n_cells, n_cells), np.nan, dtype=Ceye.dtype)

    x = np.asarray(bin_centers, dtype=np.float64)
    w_all = np.asarray(count_e, dtype=np.float64)

    use_mask = np.isfinite(x) & (x > 0) & (x <= d_max) & np.isfinite(w_all) & (w_all > 0)
    idx = np.where(use_mask)[0]

    if idx.size < min_bins:
        k0 = np.where(np.isfinite(x) & np.isfinite(w_all) & (w_all > 0))[0]
        if k0.size > 0:
            return Ceye[k0[0]].copy()
        return C_intercept

    x_loc = x[idx]
    w_loc = w_all[idx]
    x_eval = x_loc[0] if eval_at_first_bin else 0.0

    S0 = np.sum(w_loc)
    Sx = np.sum(w_loc * x_loc)
    Sxx = np.sum(w_loc * x_loc ** 2)
    det = S0 * Sxx - Sx ** 2

    if S0 == 0 or (det / (S0 * S0)) < eps:
        return Ceye[idx[0]].copy()

    def _fit_pair(i, j):
        y = Ceye[idx, i, j]
        if not np.isfinite(y).all():
            v = np.isfinite(y)
            if np.sum(v) < 3:
                return np.nan
            wv, xv, yv = w_loc[v], x_loc[v], y[v]
            s0 = np.sum(wv)
            sx = np.sum(wv * xv)
            sxx = np.sum(wv * xv ** 2)
            d = s0 * sxx - sx ** 2
            if d <= 0:
                return np.nan
            sy = np.sum(wv * yv)
            sxy = np.sum(wv * xv * yv)
            beta1 = (s0 * sxy - sx * sy) / d
            beta0 = (sxx * sy - sx * sxy) / d
        else:
            Sy = np.sum(w_loc * y)
            Sxy = np.sum(w_loc * x_loc * y)
            beta1 = (S0 * Sxy - Sx * Sy) / det
            beta0 = (Sxx * Sy - Sx * Sxy) / det
        return beta0 + beta1 * x_eval

    for i in range(n_cells):
        C_intercept[i, i] = _fit_pair(i, i)
        for j in range(i + 1, n_cells):
            val = _fit_pair(i, j)
            C_intercept[i, j] = val
            C_intercept[j, i] = val

    return C_intercept


# ---------------------------------------------------------------------------
# PSTH covariance
# ---------------------------------------------------------------------------

def bagged_split_half_psth_covariance(S, T_idx, n_boot=20, min_trials_per_time=10,
                                      seed=42, global_mean=None,
                                      weighting='pair_count'):
    """
    Bagged split-half PSTH covariance (unbiased estimator).

    Parameters
    ----------
    S : torch.Tensor (N, C)
        Spike counts.
    T_idx : torch.Tensor (N,)
        Time indices.
    n_boot : int
        Number of bootstrap splits.
    min_trials_per_time : int
        Minimum trials per time bin.
    seed : int
        Random seed.
    global_mean : ndarray (C,), optional
        Global mean for centering. If None, uses local centering.
    weighting : str
        'uniform' for equal (1/T) weighting over time bins, or
        'pair_count' to weight each time bin by n_t*(n_t-1)/2,
        matching the implicit weighting in
        compute_conditional_second_moments.

    Returns
    -------
    C_psth : ndarray (C, C)
    PSTH_mean : ndarray (T, C)
    """
    rng = np.random.default_rng(seed)
    unique_times = np.unique(T_idx.detach().cpu().numpy())
    N_cells = S.shape[1]

    time_groups = {}
    for t in unique_times:
        ix_t = np.where((T_idx == t).detach().cpu().numpy())[0]
        if len(ix_t) >= min_trials_per_time:
            time_groups[t] = ix_t

    if len(time_groups) < 2:
        return np.full((N_cells, N_cells), np.nan), None

    C_accum = np.zeros((N_cells, N_cells))
    valid_boots = 0
    PSTH_mean_accum = np.zeros((len(time_groups), N_cells))

    mu_global = None
    if global_mean is not None:
        mu_global = np.asarray(global_mean).reshape(1, -1)

    sorted_times = sorted(time_groups.keys())

    # Time-bin weights for the cross-covariance
    if weighting == 'pair_count':
        pair_counts = np.array([
            len(time_groups[t]) * (len(time_groups[t]) - 1) / 2
            for t in sorted_times
        ])
        w = pair_counts / pair_counts.sum()
    elif weighting == 'uniform':
        w = None
    else:
        raise ValueError(f"weighting must be 'uniform' or 'pair_count', got {weighting!r}")

    for k in range(n_boot):
        PSTH_A_list = []
        PSTH_B_list = []

        for t in sorted_times:
            ix_t = time_groups[t]
            perm = rng.permutation(ix_t)
            mid = len(ix_t) // 2
            mu_A = S[perm[:mid]].mean(0).detach().cpu().numpy()
            mu_B = S[perm[mid:]].mean(0).detach().cpu().numpy()
            PSTH_A_list.append(mu_A)
            PSTH_B_list.append(mu_B)

        XA = np.stack(PSTH_A_list)
        XB = np.stack(PSTH_B_list)
        PSTH_mean_accum += (XA + XB) / 2.0

        if mu_global is not None:
            XA_c = XA - mu_global
            XB_c = XB - mu_global
        else:
            XA_c = XA - XA.mean(0, keepdims=True)
            XB_c = XB - XB.mean(0, keepdims=True)

        if w is not None:
            C_k = (XA_c * w[:, None]).T @ XB_c
        else:
            n_time = XA.shape[0]
            C_k = (XA_c.T @ XB_c) / (n_time - 1)
        C_k = 0.5 * (C_k + C_k.T)

        C_accum += C_k
        valid_boots += 1

    if valid_boots == 0:
        return np.full((N_cells, N_cells), np.nan), None

    C_final = C_accum / valid_boots
    PSTH_final = PSTH_mean_accum / valid_boots

    return C_final, PSTH_final


# ---------------------------------------------------------------------------
# Rate covariance estimation
# ---------------------------------------------------------------------------

def estimate_rate_covariance(SpikeCounts, EyeTraj, T_idx, n_bins=25,
                             Ctotal=None, intercept_mode='linear'):
    """
    Estimate eye-conditioned rate covariance matrix (Crate).

    Chains second moment computation + intercept fitting.

    Parameters
    ----------
    SpikeCounts : torch.Tensor (N, C)
    EyeTraj : torch.Tensor (N, T, 2)
    T_idx : torch.Tensor (N,)
    n_bins : int or array-like
    Ctotal : ndarray (C, C), optional
        Total covariance for physical limit check.
    intercept_mode : str
        'linear', 'isotonic', or 'raw'.

    Returns
    -------
    Crate : ndarray (C, C)
    Erate : ndarray (C,)
    Ceye : ndarray (n_bins, C, C)
    bin_centers : ndarray
    count_e : ndarray
    bin_edges : ndarray
    """
    MM, bin_centers, count_e, bin_edges = compute_conditional_second_moments(
        SpikeCounts, EyeTraj, T_idx, n_bins=n_bins
    )

    # Pair-count-weighted mean rate for consistent Ceye estimation.
    #
    # compute_conditional_second_moments accumulates cross-trial products
    # S_i S_j^T across time bins, where each time bin t with n_t trials
    # contributes n_t*(n_t-1)/2 pairs.  The resulting second moment MM is
    # therefore implicitly weighted by pair count, not trial count.
    #
    # Converting MM to covariance requires subtracting E[rate] x E[rate]^T
    # under the *same* weighting.  The old code used the trial-count-weighted
    # global mean (torch.nanmean), which weights each time bin by n_t.  This
    # mismatch — pair-weighted (~ n_t^2) second moment minus trial-weighted
    # (~ n_t) mean squared — inflates off-diagonal Ceye and creates a small
    # but systematic negative bias in the shuffle null (~Dz = -0.007).
    #
    # Fix: weight the mean rate by pair count to match the second moment.
    # This eliminates 93% of the shuffle null bias while preserving the
    # real-data signal (Dz changes < 0.003).
    #
    # Old line: Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy()
    unique_times = np.unique(T_idx.detach().cpu().numpy())
    C = SpikeCounts.shape[1]
    weighted_sum = torch.zeros(C, device=SpikeCounts.device, dtype=torch.float64)
    total_pairs = 0.0
    for t in unique_times:
        mask = (T_idx == t)
        n_t = mask.sum().item()
        if n_t < 10:  # matches the threshold in compute_conditional_second_moments
            continue
        n_pairs_t = n_t * (n_t - 1) / 2
        mu_t = SpikeCounts[mask].mean(0).to(torch.float64)
        weighted_sum += n_pairs_t * mu_t
        total_pairs += n_pairs_t
    Erate = (weighted_sum / total_pairs).detach().cpu().numpy()

    Ceye = MM - Erate[:, None] * Erate[None, :]

    if intercept_mode == 'linear':
        Crate = fit_intercept_linear(Ceye, bin_centers, count_e, eval_at_first_bin=True)
    elif intercept_mode == 'isotonic':
        Crate = fit_intercept_pava(Ceye, count_e)
    elif intercept_mode == 'lowest_bin':
        Crate = Ceye[0].copy()
    else:
        Crate = Ceye[0].copy()

    if Ctotal is not None:
        bad_mask = np.diag(Crate) > 0.99 * np.diag(Ctotal)
        Crate[bad_mask, :] = np.nan
        Crate[:, bad_mask] = np.nan
        Ceye[:, bad_mask, :] = np.nan
        Ceye[:, :, bad_mask] = np.nan

    return Crate, Erate, Ceye, bin_centers, count_e, bin_edges


# ---------------------------------------------------------------------------
# Trial alignment for fixRSVP data
# ---------------------------------------------------------------------------

def align_fixrsvp_trials(dset, valid_time_bins=120, min_fix_dur=20,
                         min_total_spikes=200, fixation_radius=1.0):
    """
    Extract trial-aligned robs, eyepos, valid_mask from a fixRSVP DictDataset.

    Converts the flat (T, ...) covariate arrays into trial-aligned
    (n_trials, n_time, ...) arrays using trial_inds and psth_inds,
    filters by fixation and trial duration, and selects neurons by
    spike count.

    Parameters
    ----------
    dset : DictDataset
        Raw fixRSVP dataset with covariates: robs, eyepos, trial_inds,
        psth_inds.
    valid_time_bins : int
        Maximum number of within-trial time bins to retain.
    min_fix_dur : int
        Minimum number of fixation time bins for a trial to be included.
    min_total_spikes : int
        Minimum total spike count for a neuron to be included.
    fixation_radius : float
        Maximum eye distance from center (degrees) to count as fixation.

    Returns
    -------
    robs : ndarray (n_good_trials, valid_time_bins, n_neurons_used)
        or None if insufficient data.
    eyepos_out : ndarray (n_good_trials, valid_time_bins, 2)
    valid_mask : ndarray (n_good_trials, valid_time_bins)
    neuron_mask : ndarray (n_neurons_used,)
        Indices into the original neuron axis.
    metadata : dict
        n_trials_total, n_trials_good, n_neurons_total, n_neurons_used.
    """
    covs = dset.covariates if hasattr(dset, 'covariates') else dset

    trial_inds = np.asarray(covs['trial_inds']).ravel()
    psth_inds = np.asarray(covs['psth_inds']).ravel()
    robs_flat = np.asarray(covs['robs'])       # (T, NC)
    eyepos_flat = np.asarray(covs['eyepos'])   # (T, 2)

    trials = np.unique(trial_inds)
    NT = len(trials)
    NC = robs_flat.shape[1]
    T = int(psth_inds.max()) + 1

    # Fixation mask: eye within fixation_radius degrees of center
    fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < fixation_radius

    # Pre-allocate trial-aligned arrays (NaN-padded)
    robs_aligned = np.full((NT, T, NC), np.nan)
    eyepos_aligned = np.full((NT, T, 2), np.nan)
    fix_dur = np.full(NT, np.nan)

    for i, trial_id in enumerate(trials):
        ix = (trial_inds == trial_id) & fixation
        if not np.any(ix):
            continue
        t_inds = psth_inds[ix]
        fix_dur[i] = len(t_inds)
        robs_aligned[i, t_inds] = robs_flat[ix]
        eyepos_aligned[i, t_inds] = eyepos_flat[ix]

    # Filter trials by fixation duration
    good_trials = fix_dur > min_fix_dur
    if good_trials.sum() < 2:
        return None, None, None, None, {"n_trials_total": NT, "n_trials_good": 0,
                                         "n_neurons_total": NC, "n_neurons_used": 0}

    robs_aligned = robs_aligned[good_trials]
    eyepos_aligned = eyepos_aligned[good_trials]

    # Truncate to valid_time_bins
    T_use = min(valid_time_bins, T)
    iix = np.arange(T_use)
    robs_trunc = robs_aligned[:, iix]
    eyepos_trunc = eyepos_aligned[:, iix]

    # Neuron inclusion: total spikes across all good trials
    neuron_mask = np.where(np.nansum(robs_trunc, axis=(0, 1)) > min_total_spikes)[0]
    if len(neuron_mask) < 3:
        return None, None, None, None, {"n_trials_total": NT,
                                         "n_trials_good": int(good_trials.sum()),
                                         "n_neurons_total": NC, "n_neurons_used": 0}

    robs_out = robs_trunc[:, :, neuron_mask]

    # Valid mask: finite spikes and eye position
    valid_mask = (np.isfinite(np.sum(robs_out, axis=2))
                  & np.isfinite(np.sum(eyepos_trunc, axis=2)))

    metadata = {
        "n_trials_total": NT,
        "n_trials_good": int(good_trials.sum()),
        "n_neurons_total": NC,
        "n_neurons_used": len(neuron_mask),
    }

    return robs_out, eyepos_trunc, valid_mask, neuron_mask, metadata


# ---------------------------------------------------------------------------
# Full decomposition sweep
# ---------------------------------------------------------------------------

def run_covariance_decomposition(robs, eyepos, valid_mask,
                                 window_sizes_ms=None, window_sizes_bins=None,
                                 t_hist_ms=None, t_hist_bins=None,
                                 n_bins=15, n_shuffles=0,
                                 seed=42, dt=1 / 240, min_seg_len=36,
                                 intercept_mode='linear', device="cuda"):
    """
    Full LOTC decomposition sweep across counting windows.

    Convenience function that chains segmentation, window extraction,
    second moment estimation, intercept fitting, and PSTH covariance.

    Window sizes can be specified in bins (preferred) or milliseconds.
    If both are provided, window_sizes_bins takes precedence.

    Parameters
    ----------
    robs : ndarray (n_trials, n_time, n_cells)
    eyepos : ndarray (n_trials, n_time, 2)
    valid_mask : ndarray (n_trials, n_time)
    window_sizes_ms : list of float, optional
        Counting window sizes in milliseconds. Converted to bins via
        int(ms / (dt * 1000)). Ignored if window_sizes_bins is provided.
    window_sizes_bins : list of int, optional
        Counting window sizes in time bins (preferred). Avoids rounding.
    t_hist_ms : float, optional
        History window in milliseconds. Default 10 ms if neither
        t_hist_ms nor t_hist_bins is provided.
    t_hist_bins : int, optional
        History window in bins. Takes precedence over t_hist_ms.
    n_bins : int
    n_shuffles : int
    seed : int
    dt : float
        Duration of one time bin in seconds.
    min_seg_len : int
    intercept_mode : str
    device : str

    Returns
    -------
    results : list of dict
        Per-window metrics. Each dict includes 'window_bins' and 'window_ms'.
    mats : list of dict
    """
    if window_sizes_bins is None and window_sizes_ms is None:
        raise ValueError("Provide window_sizes_bins or window_sizes_ms")

    ms_per_bin = dt * 1000

    if window_sizes_bins is not None:
        win_bins_list = list(window_sizes_bins)
    else:
        win_bins_list = [max(1, int(ms / ms_per_bin)) for ms in window_sizes_ms]

    if t_hist_bins is not None:
        t_hist_bins_val = t_hist_bins
    elif t_hist_ms is not None:
        t_hist_bins_val = int(t_hist_ms / ms_per_bin)
    else:
        t_hist_bins_val = int(10 / ms_per_bin)  # default 10 ms
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Sanitize
    if np.isnan(robs).any():
        robs = np.nan_to_num(robs, nan=0.0)
    eyepos = np.nan_to_num(eyepos, nan=0.0)

    robs_t = torch.tensor(robs, dtype=torch.float32, device=device_obj)
    eyepos_t = torch.tensor(eyepos, dtype=torch.float32, device=device_obj)

    # Segment extraction
    segments = extract_valid_segments(valid_mask, min_len_bins=min_seg_len)
    print(f"Found {len(segments)} valid segments")

    rng_shuffle = torch.Generator(device=device_obj)
    rng_shuffle.manual_seed(seed)

    results = []
    mats_save = []

    for t_count_bins in tqdm(win_bins_list):
        win_ms_actual = t_count_bins * ms_per_bin

        SpikeCounts, EyeTraj, T_idx, _ = extract_windows(
            robs_t, eyepos_t, segments, t_count_bins,
            max(t_hist_bins_val, t_count_bins), device=str(device_obj)
        )

        if SpikeCounts is None:
            continue

        n_samples = SpikeCounts.shape[0]
        if n_samples < 100:
            continue

        # Total covariance
        ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
        Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()

        # Rate covariance
        Crate, Erate, Ceye, bin_centers, count_e, bin_edges = estimate_rate_covariance(
            SpikeCounts, EyeTraj, T_idx, n_bins=n_bins,
            Ctotal=Ctotal, intercept_mode=intercept_mode
        )

        # PSTH covariance
        Cpsth, PSTH_mean = bagged_split_half_psth_covariance(
            SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
            seed=seed, global_mean=Erate
        )

        # Shuffle controls
        shuffled_intercepts = []
        if n_shuffles > 0:
            for k in range(n_shuffles):
                perm = torch.randperm(n_samples, generator=rng_shuffle, device=device_obj)
                EyeTraj_shuff = EyeTraj[perm]
                Crate_shuff, _, _, _, _, _ = estimate_rate_covariance(
                    SpikeCounts, EyeTraj_shuff, T_idx, n_bins=bin_edges,
                    Ctotal=Ctotal, intercept_mode=intercept_mode
                )
                shuffled_intercepts.append(Crate_shuff)

        # Derived matrices
        Cfem = Crate - Cpsth
        Cfem = 0.5 * (Cfem + Cfem.T)

        CnoiseU = Ctotal - Cpsth
        CnoiseC = Ctotal - Crate
        CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
        CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)

        ff_uncorr = np.diag(CnoiseU) / Erate
        ff_corr = np.diag(CnoiseC) / Erate
        NoiseCorrU = cov_to_corr(CnoiseU)
        NoiseCorrC = cov_to_corr(CnoiseC)
        alpha = np.diag(Cpsth) / np.diag(Crate)

        if np.isnan(Cfem).any():
            rank = np.nan
        else:
            evals = np.linalg.eigvalsh(Cfem)[::-1]
            pos = evals[evals > 0]
            rank = (np.sum(pos[:2]) / np.sum(pos)) if len(pos) > 2 else 1.0

        results.append({
            "window_bins": t_count_bins,
            "window_ms": win_ms_actual,
            "ff_uncorr": ff_uncorr,
            "ff_corr": ff_corr,
            "ff_uncorr_mean": np.nanmean(ff_uncorr),
            "ff_corr_mean": np.nanmean(ff_corr),
            "alpha": alpha,
            "fem_rank_ratio": rank,
            "n_samples": n_samples,
            "Erates": Erate,
            "count_e": count_e,
        })

        mats_save.append({
            "Total": Ctotal,
            "PSTH": Cpsth,
            "FEM": Cfem,
            "Intercept": Crate,
            "Shuffled_Intercepts": shuffled_intercepts,
            "NoiseCorrU": NoiseCorrU,
            "NoiseCorrC": NoiseCorrC,
            "PSTH_mean": PSTH_mean,
        })

    return results, mats_save
