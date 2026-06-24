r"""Eye-position-distribution-matched Law-of-Total-Covariance (LOTC) decomposition.

This module is the production home of the McFarland-et-al.-(2016) cross-trial
covariance decomposition, extended with the two corrections developed and
validated in ``ryan/methods_eyepos_matching`` (see that folder's ``writeup.md``
and ``note_consistency.md``):

  * **Extension 1** — consistent pair-count time-bin weighting under variable
    trials-per-time-bin n_t (``time_bin_weighting='pair_count'``).
  * **Extension 2** — eye-position-distribution matching: the close-pair rate
    second moment is sampled in proportion to p(e)^2 while the total / PSTH
    covariances are over p(e); importance reweighting toward a target q(e)
    removes the resulting bias under a non-homogeneous stimulus.

      target='naive'   : reproduce McFarland's original (close-pair 2nd moment
                         over p^2, mean over p — an inconsistent mix).
      target='full'    : Direction 1, q = p   (the actual viewing distribution).
      target='central' : Direction 2, q = p^2 (the close-pair distribution).

Production defaults are the validated corrected settings: ``target='full'``,
``time_bin_weighting='pair_count'``, ``cpsth_method='mcfarland'``,
``closepair_density='direct'``.

The functions here are a self-contained port of
``ryan/methods_eyepos_matching/estimators.py`` (the estimator math) plus the
retained utilities and the model-side (digital-twin) decomposition. Production
imports nothing from the methods folder; that folder is frozen as the methods
development record.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal

from VisionCore.subspace import project_to_psd  # re-export for convenience


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cov_to_corr(C, min_var=1e-3):
    """Convert covariance matrix to correlation matrix.

    Returns NaN for neurons with variance below ``min_var``. Diagonal is set to
    0 by convention (for noise-correlation analysis).
    """
    C = np.asarray(C, dtype=np.float64)
    variances = np.diag(C)

    valid_mask = variances > min_var
    std_devs = np.full_like(variances, np.nan)
    std_devs[valid_mask] = np.sqrt(variances[valid_mask])

    outer_std = np.outer(std_devs, std_devs)

    R = C / outer_std
    R = np.clip(R, -1.0, 1.0)
    R[~np.isfinite(R)] = np.nan
    np.fill_diagonal(R, 0.0)
    return R


def get_upper_triangle(C):
    """Extract upper-triangle values (k=1 diagonal offset) from a square matrix."""
    rows, cols = np.triu_indices_from(C, k=1)
    return C[rows, cols]


def extract_valid_segments(valid_mask, min_len_bins=36):
    """Find contiguous valid segments in a (n_trials, n_time) boolean mask.

    Returns a list of ``(trial, start, stop)`` tuples for runs of at least
    ``min_len_bins`` contiguous valid bins.
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


def extract_windows(robs, eyepos, segments, t_count, t_hist):
    """Stride count/trajectory windows within valid segments (numpy).

    Mirrors the legacy torch ``extract_windows`` semantics: the count window
    covers ``[t0 + t_hist : t0 + total_len)`` (summed spike counts), the
    trajectory covers ``[t0 : t0 + total_len)``, stride is ``t_count`` within
    each segment, and ``T_idx = t0 + t_hist`` (the count-window start bin).
    ``total_len = t_hist + t_count``.

    Parameters
    ----------
    robs : ndarray (n_trials, n_time, n_cells)
    eyepos : ndarray (n_trials, n_time, 2)
    segments : list of (trial, start, stop)
    t_count, t_hist : int

    Returns
    -------
    counts (N, n_cells), trajectories (N, total_len, 2), T_idx (N,)
        or (None, None, None) if no windows fit.
    """
    robs = np.asarray(robs)
    eyepos = np.asarray(eyepos)
    total_len = t_hist + t_count
    trial_indices, time_indices = [], []
    for (tr, start, stop) in segments:
        if (stop - start) < total_len:
            continue
        t_starts = np.arange(start, stop - total_len + 1, t_count)
        trial_indices.extend([tr] * len(t_starts))
        time_indices.extend(t_starts.tolist())
    if len(trial_indices) == 0:
        return None, None, None

    idx_tr = np.asarray(trial_indices)
    idx_t0 = np.asarray(time_indices)

    offsets = np.arange(total_len)[None, :]
    gather_t = idx_t0[:, None] + offsets
    gather_tr = np.broadcast_to(idx_tr[:, None], gather_t.shape)
    trajectories = eyepos[gather_tr, gather_t, :]

    spike_offsets = np.arange(t_hist, total_len)[None, :]
    gather_t_spk = idx_t0[:, None] + spike_offsets
    gather_tr_spk = np.broadcast_to(idx_tr[:, None], gather_t_spk.shape)
    counts = robs[gather_tr_spk, gather_t_spk, :].sum(axis=1)

    T_idx = idx_t0 + t_hist
    return counts, trajectories, T_idx


# ---------------------------------------------------------------------------
# Density estimation p_hat(e)
# ---------------------------------------------------------------------------

def _density_fn(E, kind):
    """Return a callable p_hat: (M,2)->(M,) for the eye-position density."""
    if callable(kind):
        return kind
    if kind == "gaussian":
        mu = E.mean(0)
        cov = np.cov(E.T) + 1e-9 * np.eye(2)
        rv = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
        return lambda X: rv.pdf(X)
    if kind == "kde":
        kde = gaussian_kde(E.T)
        return lambda X: kde(X.T)
    raise ValueError(f"unknown density kind: {kind!r}")


# ---------------------------------------------------------------------------
# Weighted moment helpers
# ---------------------------------------------------------------------------

def _weighted_mean(S, w):
    w = w / w.sum()
    return (w[:, None] * S).sum(0)


def _weighted_cov(S, w):
    """Reliability-weights unbiased covariance with normalized weights w."""
    w = w / w.sum()
    mu = (w[:, None] * S).sum(0)
    Sc = S - mu
    C = (Sc * w[:, None]).T @ Sc
    denom = 1.0 - np.sum(w ** 2)
    return C / denom if denom > 0 else C


def _all_pairs_second_moment(S, E, T, phat, target,
                             time_bin_weighting="pair_count", phat_pair=None):
    """McFarland M6/M12 all-distinct-pair second moment, target-reweighted.

    At each time bin t, every distinct trial pair (i, j) with i<j is included
    (no Delta_e restriction -- the close-pair Crate estimator without the
    close-pair filter, i.e. the conditional second moment Ceye(Delta_e) read at
    its eye-distribution-marginalized asymptote). Per-pair weight
    pw_q(i, j) = w_i * w_j with w_i = q(e_i)/p(e_i):

        target in {'naive', 'full'}: w_i = 1               (q = p)
        target == 'central'        : w_i = p_pair/p        (q = p^2)

    Across-bin scheme (same as the close-pair estimator):
        'pair_count' : pool pairs, average per pair (bin ~ n_t(n_t-1)/2);
        'uniform'    : per-bin mean first, uniform across bins (bin ~ 1);
        'trial_count': per-bin mean, then weight by n_t across bins.

    Uses the identity 2 Sum_{i<j} w_i w_j S_i x S_j =
    (Sum w_i S_i) x (Sum w_i S_i) - Sum w_i^2 S_i x S_i to avoid an explicit
    pair tensor. Distinct-trial pairing cancels Poisson and simultaneous
    cross-cell noise in expectation.
    """
    if target == "central":
        if phat_pair is None:
            w_trial = np.clip(phat(E), 1e-12, None)
        else:
            w_trial = (np.clip(phat_pair(E), 1e-12, None)
                       / np.clip(phat(E), 1e-12, None))
    else:                                # 'naive' and 'full' -- q = p marginal
        w_trial = np.ones(len(S))

    return _all_pairs_second_moment_with_weights(
        S, w_trial, T, time_bin_weighting=time_bin_weighting
    )


def _all_pairs_second_moment_with_weights(S, w_trial, T,
                                          time_bin_weighting="pair_count"):
    """McFarland M6 all-distinct-pair second moment with explicit per-trial
    weights ``w_trial``. Same algebraic identity as
    :func:`_all_pairs_second_moment`, factored out so the weighting source
    (single-bin p_hat vs. trajectory representative-point ratio) can vary."""
    C = S.shape[1]
    num = np.zeros((C, C), dtype=np.float64)
    denom = 0.0
    for t in np.unique(T):
        ix = np.where(T == t)[0]
        if len(ix) < 2:
            continue
        w_t = w_trial[ix]
        S_t = S[ix]
        wS_sum = (w_t[:, None] * S_t).sum(0)
        outer_sum = np.outer(wS_sum, wS_sum)
        wS2 = (w_t[:, None] ** 2) * S_t
        diag_term = wS2.T @ S_t
        pair_sum_t = 0.5 * (outer_sum - diag_term)
        w_sum = w_t.sum()
        n_pair_weight_t = 0.5 * (w_sum ** 2 - (w_t ** 2).sum())
        if n_pair_weight_t <= 0:
            continue
        if time_bin_weighting == "pair_count":
            num += pair_sum_t
            denom += n_pair_weight_t
        elif time_bin_weighting == "uniform":
            num += pair_sum_t / n_pair_weight_t
            denom += 1.0
        elif time_bin_weighting == "trial_count":
            nt_for_bin = float(len(ix))
            num += pair_sum_t / n_pair_weight_t * nt_for_bin
            denom += nt_for_bin
        else:
            raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    if denom <= 0:
        return np.full((C, C), np.nan)
    prod = num / denom
    return 0.5 * (prod + prod.T)


def _split_half_psth_cov(S, T, sw, n_boot, seed, min_tpp,
                         time_bin_weighting="pair_count"):
    """Split-half PSTH covariance with per-sample target weights sw.

    Bagged-bootstrap alternative to ``_all_pairs_second_moment`` -- both target
    the same population PSTH covariance via distinct-trial cross-pair products,
    recovering the all-pairs M6 estimator as ``n_boot -> inf``. Retained as a
    fallback; the production default is ``cpsth_method='mcfarland'``.
    """
    rng = np.random.default_rng(seed)
    time_bins = [t for t in np.unique(T) if (T == t).sum() >= min_tpp]
    if len(time_bins) < 2:
        return np.full((S.shape[1], S.shape[1]), np.nan)
    idx_by_t = {t: np.where(T == t)[0] for t in time_bins}
    nt = np.array([len(idx_by_t[t]) for t in time_bins], float)
    if time_bin_weighting == "pair_count":
        wph = nt * (nt - 1) / 2
    elif time_bin_weighting == "uniform":
        wph = np.ones_like(nt)
    elif time_bin_weighting == "trial_count":
        wph = nt
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    wph = wph / wph.sum()

    C = np.zeros((S.shape[1], S.shape[1]))
    for _ in range(n_boot):
        A, B = [], []
        for t in time_bins:
            ix = rng.permutation(idx_by_t[t])
            mid = len(ix) // 2
            ia, ib = ix[:mid], ix[mid:]
            wa, wb = sw[ia], sw[ib]
            A.append((wa[:, None] * S[ia]).sum(0) / wa.sum())
            B.append((wb[:, None] * S[ib]).sum(0) / wb.sum())
        A = np.asarray(A); B = np.asarray(B)
        Ac = A - (wph[:, None] * A).sum(0)
        Bc = B - (wph[:, None] * B).sum(0)
        Ck = (Ac * wph[:, None]).T @ Bc
        C += 0.5 * (Ck + Ck.T)
    return C / n_boot


def _close_pair_midpoints(E, T, threshold):
    """Midpoints of all distinct same-time-bin close pairs (|e_i-e_j|<threshold).

    The realized sample from the close-pair density; fitting a KDE on these
    points gives p_hat_pair directly (``closepair_density='direct'``).
    """
    acc = []
    for t in np.unique(T):
        ix = np.where(T == t)[0]
        if len(ix) < 2:
            continue
        Et = E[ix]
        i, j = np.triu_indices(len(ix), k=1)
        d = np.linalg.norm(Et[i] - Et[j], axis=1)
        close = d < threshold
        if not close.any():
            continue
        gi, gj = ix[i[close]], ix[j[close]]
        acc.append(0.5 * (E[gi] + E[gj]))
    if not acc:
        return np.zeros((0, 2))
    return np.concatenate(acc)


def _close_pair_second_moment(S, E, T, phat, target, threshold, weight_clip,
                              time_bin_weighting="pair_count", phat_pair=None):
    """Importance-reweighted close-pair second moment MM (C, C).

    All distinct same-time-bin trial pairs with |e_i - e_j| < threshold are
    collected. Pair (i,j) at midpoint m gets weight pw combining a target
    importance weight and an across-bin scheme:

      target in {'naive','central'}: pw_q = 1            (p^2 sampling kept)
      target == 'full'             : pw_q = p/p_pair     (reweight p^2 -> p),
                                     clipped to weight_clip * median.

    Under constant n_t the three ``time_bin_weighting`` choices coincide.
    """
    C = S.shape[1]
    Si_acc, Sj_acc, mid_acc, t_acc = [], [], [], []
    for t in np.unique(T):
        ix = np.where(T == t)[0]
        if len(ix) < 2:
            continue
        Et = E[ix]
        i, j = np.triu_indices(len(ix), k=1)
        d = np.linalg.norm(Et[i] - Et[j], axis=1)
        close = d < threshold
        if not close.any():
            continue
        gi, gj = ix[i[close]], ix[j[close]]
        Si_acc.append(gi); Sj_acc.append(gj)
        mid_acc.append(0.5 * (E[gi] + E[gj]))
        t_acc.append(np.full(int(close.sum()), t))
    if not Si_acc:
        return np.full((C, C), np.nan)
    gi = np.concatenate(Si_acc); gj = np.concatenate(Sj_acc)
    mid = np.concatenate(mid_acc)
    tpair = np.concatenate(t_acc)

    if target == "full":
        if phat_pair is None:
            pw_q = 1.0 / np.clip(phat(mid), 1e-12, None)
        else:
            pw_q = (np.clip(phat(mid), 1e-12, None)
                    / np.clip(phat_pair(mid), 1e-12, None))
        pw_q = np.clip(pw_q, None, weight_clip * np.median(pw_q))
    else:
        pw_q = np.ones(len(gi))

    if time_bin_weighting == "pair_count":
        pw_t = np.ones(len(gi))
    elif time_bin_weighting == "uniform":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv)
        pw_t = 1.0 / nP_t[inv]
    elif time_bin_weighting == "trial_count":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv).astype(float)
        nt_by_bin = {int(u): int((T == u).sum()) for u in np.unique(T)}
        nt_per_pair = np.array([nt_by_bin[int(u)] for u in tpair], dtype=float)
        pw_t = nt_per_pair / nP_t[inv]
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")

    pw = pw_q * pw_t
    pw = pw / pw.sum()
    prod = (S[gi].T * pw) @ S[gj]
    return 0.5 * (prod + prod.T)


# ---------------------------------------------------------------------------
# Single-bin decomposition
# ---------------------------------------------------------------------------

def decompose(counts, eye, target="full", valid=None, threshold=0.05,
              density="kde", weight_clip=1e6, n_boot=20, seed=42,
              min_trials_per_time_bin=10, time_bin_weighting="pair_count",
              cpsth_method="mcfarland", closepair_density="direct"):
    """Distribution-matched LOTC decomposition of a multi-cell session.

    Single-eye-position-per-sample estimator (each (trial, time-bin) is one
    sample). For multi-bin eye trajectories use :func:`decompose_trajectory`.

    Parameters
    ----------
    counts : ndarray (n_trials, n_time_bins, n_cells)
        Spike counts (or deterministic rates) per trial and analysis time bin.
    eye : ndarray (n_trials, n_time_bins, 2)
        Per-(trial, time-bin) eye position in degrees.
    target : {'naive', 'full', 'central'}
        Eye distribution to match all terms to (see module docstring).
    valid : ndarray (n_trials, n_time_bins) of bool, optional
        Usable-sample mask; combined with finiteness. Time bins with fewer than
        ``min_trials_per_time_bin`` valid trials are dropped.
    threshold : float
        Close-pair eye-distance threshold (deg).
    density : {'kde', 'gaussian'} or callable
        Eye-position density estimator p_hat used for importance weights.
    weight_clip : float
        Cap on importance weights (multiples of the median) for target='full'.
    time_bin_weighting : {'pair_count', 'uniform', 'trial_count'}
        Across-bin combination used by all terms (see module docstring).
    cpsth_method : {'mcfarland', 'split_half'}
        PSTH-covariance debiasing. 'mcfarland' (default) is the deterministic
        all-distinct-pair second moment; 'split_half' is the bagged bootstrap.
    closepair_density : {'direct', 'squared'}
        How the close-pair midpoint density is estimated for the importance
        weights. 'direct' (default) fits a separate KDE on the realized
        close-pair midpoints; 'squared' assumes p_pair = p_hat^2.

    Returns
    -------
    dict: Ctotal, Cpsth, Crate, CnoiseC (C,C); one_minus_alpha, fano, Erate (C,);
          noise_corr (C,C); n_time_bins, n_samples.
    """
    counts = np.asarray(counts, float)
    eye = np.asarray(eye, float)
    n_trials, n_time_bins, n_cells = counts.shape

    finite = np.isfinite(counts).all(2) & np.isfinite(eye).all(2)
    valid = finite if valid is None else (np.asarray(valid, bool) & finite)
    keep_time_bin = valid.sum(0) >= min_trials_per_time_bin
    mask = valid & keep_time_bin[None, :]

    tr, ph = np.where(mask)
    S = counts[tr, ph, :]            # (N, C)
    E = eye[tr, ph, :]               # (N, 2)
    T = ph                           # time-bin label per sample

    # Degenerate input (e.g. a per-cell-masked model cell with few valid
    # samples): too few points to fit the density / form distinct-trial pairs.
    if S.shape[0] < 3 or len(np.unique(T)) < 2:
        nan_C = np.full((n_cells, n_cells), np.nan)
        nan_v = np.full(n_cells, np.nan)
        return {
            "Ctotal": nan_C, "Cpsth": nan_C, "Crate": nan_C, "CnoiseC": nan_C,
            "noise_corr": nan_C, "fano": nan_v, "Erate": nan_v,
            "one_minus_alpha": nan_v,
            "n_time_bins": int(keep_time_bin.sum()), "n_samples": int(mask.sum()),
        }

    phat = _density_fn(E, density)
    p_samp = np.clip(phat(E), 1e-12, None)

    if closepair_density == "direct":
        mid_cp = _close_pair_midpoints(E, T, threshold)
        phat_pair = _density_fn(mid_cp, density) if len(mid_cp) >= 2 else None
    elif closepair_density == "squared":
        phat_pair = None
    else:
        raise ValueError(f"unknown closepair_density: {closepair_density!r}")

    if target == "central":
        if phat_pair is None:
            tw = p_samp.copy()
        else:
            tw = np.clip(phat_pair(E), 1e-12, None) / p_samp
    else:                            # 'naive' and 'full' keep full-p sampling
        tw = np.ones(len(S))

    nt_by_t = {t: int((T == t).sum()) for t in np.unique(T)}
    if time_bin_weighting == "pair_count":
        pw_t = {t: max(nt_by_t[t] - 1, 0) / 2.0 for t in nt_by_t}
    elif time_bin_weighting == "uniform":
        pw_t = {t: (1.0 / nt_by_t[t]) if nt_by_t[t] > 0 else 0.0 for t in nt_by_t}
    elif time_bin_weighting == "trial_count":
        pw_t = {t: 1.0 for t in nt_by_t}
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    time_bin_w = np.array([pw_t[t] for t in T], dtype=float)
    sw = tw * time_bin_w

    Erate = _weighted_mean(S, sw)
    Ctotal = _weighted_cov(S, sw)
    if cpsth_method == "mcfarland":
        MM_psth = _all_pairs_second_moment(S, E, T, phat, target,
                                           time_bin_weighting=time_bin_weighting,
                                           phat_pair=phat_pair)
        Cpsth = MM_psth - np.outer(Erate, Erate)
    elif cpsth_method == "split_half":
        Cpsth = _split_half_psth_cov(S, T, sw, n_boot, seed,
                                     min_trials_per_time_bin,
                                     time_bin_weighting=time_bin_weighting)
    else:
        raise ValueError(f"unknown cpsth_method: {cpsth_method!r}")
    MM = _close_pair_second_moment(S, E, T, phat, target, threshold, weight_clip,
                                   time_bin_weighting=time_bin_weighting,
                                   phat_pair=phat_pair)
    Crate = MM - np.outer(Erate, Erate)

    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)
    noise_corr = cov_to_corr(CnoiseC)
    with np.errstate(divide="ignore", invalid="ignore"):
        fano = np.diag(CnoiseC) / Erate
        alpha = np.clip(np.diag(Cpsth) / np.diag(Crate), 0.0, 1.0)
    one_minus_alpha = 1.0 - alpha
    one_minus_alpha[~(np.diag(Crate) > 0)] = np.nan

    return {
        "Ctotal": Ctotal, "Cpsth": Cpsth, "Crate": Crate, "CnoiseC": CnoiseC,
        "noise_corr": noise_corr, "fano": fano, "Erate": Erate,
        "one_minus_alpha": one_minus_alpha,
        "n_time_bins": int(keep_time_bin.sum()), "n_samples": int(mask.sum()),
    }


def rate_variance_by_distance(counts, trajectories, T_idx, bin_edges,
                              min_trials_per_time_bin=10,
                              time_bin_weighting="pair_count"):
    """Conditional rate covariance Ceye(Delta_e) binned by close-pair distance.

    Diagnostic used by the figure-2 example panel: it exposes the full
    Ceye-vs-distance curve whose Delta_e -> 0 intercept the close-pair estimator
    reads off. Same-T_idx distinct-trial pairs are binned by RMS trajectory
    distance (the ``decompose_trajectory`` metric); per bin the uncentred second
    moment ``E[S_i (x) S_j]`` is accumulated and the pair-count-weighted
    ``Erate (x) Erate`` is subtracted.

    Parameters
    ----------
    counts : ndarray (N, n_cells)
    trajectories : ndarray (N, t_window, 2)
    T_idx : ndarray (N,)
    bin_edges : array-like
        Distance bin edges (deg, per-bin RMS units).

    Returns
    -------
    dict: bin_centers (n_bins,), count_e (n_bins,), Ceye (n_bins, C, C).
    """
    counts = np.asarray(counts, float)
    trajectories = np.asarray(trajectories, float)
    T_idx = np.asarray(T_idx).astype(int)
    bin_edges = np.asarray(bin_edges, float)
    n_bins = len(bin_edges) - 1
    N, t_window, _ = trajectories.shape
    C = counts.shape[1]
    inv_sqrt_T = 1.0 / float(np.sqrt(float(t_window)))
    flat = trajectories.reshape(N, -1)

    SS = np.zeros((n_bins, C, C), dtype=np.float64)
    count_e = np.zeros(n_bins, dtype=np.int64)
    # pair-count-weighted mean rate (matches the close-pair estimator's Erate)
    wsum = np.zeros(C, dtype=np.float64)
    wtot = 0.0
    for t in np.unique(T_idx):
        ix = np.where(T_idx == t)[0]
        if len(ix) < max(2, min_trials_per_time_bin):
            continue
        F = flat[ix]
        D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1) * inv_sqrt_T
        i, j = np.triu_indices(len(ix), k=1)
        d = D[i, j]
        bid = np.digitize(d, bin_edges) - 1
        Si = counts[ix[i]]
        Sj = counts[ix[j]]
        for k in range(n_bins):
            mk = bid == k
            if not mk.any():
                continue
            SS[k] += Si[mk].T @ Sj[mk]
            count_e[k] += int(mk.sum())
        n_t = len(ix)
        wsum += (n_t - 1) / 2.0 * counts[ix].mean(0) * n_t
        wtot += (n_t - 1) / 2.0 * n_t
    Erate = wsum / wtot if wtot > 0 else np.full(C, np.nan)

    Ceye = np.full((n_bins, C, C), np.nan)
    for k in range(n_bins):
        if count_e[k] > 0:
            MM = SS[k] / count_e[k]
            MM = 0.5 * (MM + MM.T)
            Ceye[k] = MM - np.outer(Erate, Erate)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {"bin_centers": bin_centers, "count_e": count_e, "Ceye": Ceye,
            "Erate": Erate}


# ---------------------------------------------------------------------------
# Trajectory-mode estimator (multi-bin eye trajectories)
# ---------------------------------------------------------------------------

def _rms_traj_close_pairs(trajectories, T_idx, threshold):
    """Enumerate same-T_idx close pairs by RMS trajectory distance.

    Per-pair distance is the L2 distance of flattened trajectory vectors scaled
    by 1/sqrt(t_window), so the threshold has the same per-bin-distance units as
    the single-bin scheme (and the flat-trajectory limit coincides with
    :func:`decompose` at the same threshold).
    """
    N, T, _ = trajectories.shape
    inv_sqrt_T = 1.0 / float(np.sqrt(float(T)))
    flat = trajectories.reshape(N, -1)
    gi_acc, gj_acc, t_acc, mid_acc = [], [], [], []
    for t in np.unique(T_idx):
        ix = np.where(T_idx == t)[0]
        if len(ix) < 2:
            continue
        F = flat[ix]
        D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1) * inv_sqrt_T
        i, j = np.triu_indices(len(ix), k=1)
        d = D[i, j]
        close = d < threshold
        if not close.any():
            continue
        gi = ix[i[close]]; gj = ix[j[close]]
        gi_acc.append(gi); gj_acc.append(gj)
        t_acc.append(np.full(int(close.sum()), t))
        mid_acc.append(0.5 * (trajectories[gi] + trajectories[gj]))
    if not gi_acc:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0, int),
                np.zeros((0, T, 2)))
    return (np.concatenate(gi_acc), np.concatenate(gj_acc),
            np.concatenate(t_acc),
            np.concatenate(mid_acc, axis=0))


def _geometric_median(points, n_iter=128, eps=1e-10, tol=1e-9):
    """Geometric median (L1 multivariate median) by Weiszfeld iteration.

    Reduces over the second-to-last axis: ``points`` (..., M, 2) -> (..., 2).
    Robust to within-window microsaccades (a few bins far from the fixation
    cluster leave it inside the cluster), unlike the centroid. Distances are
    floored at ``eps`` for stability when an iterate lands on a data point.
    """
    P = np.asarray(points, dtype=float)
    x = P.mean(axis=-2)
    for _ in range(n_iter):
        diff = P - x[..., None, :]
        d = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
        w = 1.0 / d
        x_new = (w[..., None] * P).sum(-2) / w.sum(-1)[..., None]
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x


def decompose_trajectory(counts, trajectories, T_idx, target="full",
                         valid=None, threshold=0.05, weight_clip=1e6,
                         n_boot=20, seed=42, min_trials_per_time_bin=10,
                         time_bin_weighting="pair_count",
                         cpsth_method="mcfarland",
                         reduction="geometric_median",
                         closepair_density="direct"):
    """Trajectory-mode distribution-matched LOTC decomposition.

    Multi-bin extension of :func:`decompose`. Each sample is a window of
    ``t_window`` contiguous bins of eye trajectory. The close-pair filter is the
    RMS trajectory distance, but the importance-weight density is built by
    reducing each trajectory to a single representative 2-D point and fitting one
    KDE on those points -- the exact single-bin construction with ``p^2`` implied
    by ``p_hat`` (no separate close-pair-midpoint KDE unless
    ``closepair_density='direct'``).

    Parameters
    ----------
    counts : ndarray (N, n_cells)
        Per-sample window-integrated spike counts (or deterministic rates).
    trajectories : ndarray (N, t_window, 2)
        Per-sample per-bin eye trajectories.
    T_idx : ndarray (N,)
        Analysis time bin index per sample.
    reduction : {'geometric_median', 'centroid'}
        How to reduce each trajectory to its representative 2-D point.

    Other parameters as in :func:`decompose`.

    Returns
    -------
    dict with the same keys as :func:`decompose` plus ``n_close_pairs``.
    """
    counts = np.asarray(counts, dtype=float)
    trajectories = np.asarray(trajectories, dtype=float)
    T_idx = np.asarray(T_idx).astype(int)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape (N, t_window, 2)")
    if counts.ndim != 2:
        raise ValueError("counts must have shape (N, n_cells)")
    N, t_window, _ = trajectories.shape
    if counts.shape[0] != N or T_idx.shape[0] != N:
        raise ValueError("counts, trajectories, T_idx must share N")

    finite = (np.isfinite(counts).all(axis=1)
              & np.isfinite(trajectories).all(axis=(1, 2)))
    valid = finite if valid is None else (np.asarray(valid, bool) & finite)

    keep_mask = np.zeros(N, bool)
    keep_T = []
    for t in np.unique(T_idx[valid]):
        ix = np.where(valid & (T_idx == t))[0]
        if len(ix) >= min_trials_per_time_bin:
            keep_mask[ix] = True
            keep_T.append(t)
    keep_T = np.asarray(keep_T)

    S = counts[keep_mask]
    Tr = trajectories[keep_mask]
    T = T_idx[keep_mask]
    n_cells = S.shape[1]

    if reduction == "geometric_median":
        rho = _geometric_median(Tr)
    elif reduction == "centroid":
        rho = Tr.mean(axis=1)
    else:
        raise ValueError(f"unknown reduction: {reduction!r}")
    phat = _density_fn(rho, "kde")
    p_rho = np.clip(phat(rho), 1e-12, None)

    gi, gj, tpair, _mid_traj = _rms_traj_close_pairs(Tr, T, threshold)
    n_pairs = len(gi)

    if closepair_density == "direct":
        rho_mid_cp = 0.5 * (rho[gi] + rho[gj]) if n_pairs >= 2 else None
        phat_pair = (_density_fn(rho_mid_cp, "kde")
                     if rho_mid_cp is not None else None)
    elif closepair_density == "squared":
        phat_pair = None
    else:
        raise ValueError(f"unknown closepair_density: {closepair_density!r}")

    if target == "central":
        if phat_pair is None:
            tw = p_rho.copy()
        else:
            tw = np.clip(phat_pair(rho), 1e-12, None) / p_rho
    elif target in ("naive", "full"):
        tw = np.ones(len(S))
    else:
        raise ValueError(f"unknown target: {target!r}")

    nt_by_T = {int(t): int((T == t).sum()) for t in np.unique(T)}
    if time_bin_weighting == "pair_count":
        pw_t_by_T = {t: max(nt_by_T[t] - 1, 0) / 2.0 for t in nt_by_T}
    elif time_bin_weighting == "uniform":
        pw_t_by_T = {t: (1.0 / nt_by_T[t]) if nt_by_T[t] > 0 else 0.0
                     for t in nt_by_T}
    elif time_bin_weighting == "trial_count":
        pw_t_by_T = {t: 1.0 for t in nt_by_T}
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    tb_w = np.array([pw_t_by_T[int(t)] for t in T], dtype=float)
    sw = tw * tb_w

    Erate = _weighted_mean(S, sw)
    Ctotal = _weighted_cov(S, sw)

    # Centred S for the close-pair / all-pairs second moments (numerical
    # precision when t_window * mu_0 dominates the O(1) variance signal).
    S_c = S - Erate[None, :]

    if target == "central":
        w_trial = (p_rho.copy() if phat_pair is None
                   else np.clip(phat_pair(rho), 1e-12, None) / p_rho)
    else:
        w_trial = np.ones(len(S))

    if cpsth_method == "mcfarland":
        Cpsth = _all_pairs_second_moment_with_weights(
            S_c, w_trial, T, time_bin_weighting=time_bin_weighting)
    elif cpsth_method == "split_half":
        Cpsth = _split_half_psth_cov(S, T, sw, n_boot, seed,
                                     min_trials_per_time_bin,
                                     time_bin_weighting=time_bin_weighting)
    else:
        raise ValueError(f"unknown cpsth_method: {cpsth_method!r}")

    if n_pairs == 0:
        Crate = np.full((n_cells, n_cells), np.nan)
    else:
        rho_mid = 0.5 * (rho[gi] + rho[gj])
        if target == "full":
            if phat_pair is None:
                pw_q = 1.0 / np.clip(phat(rho_mid), 1e-12, None)
            else:
                pw_q = (np.clip(phat(rho_mid), 1e-12, None)
                        / np.clip(phat_pair(rho_mid), 1e-12, None))
            pw_q = np.clip(pw_q, None, weight_clip * np.median(pw_q))
        else:
            pw_q = np.ones(n_pairs)
        if time_bin_weighting == "pair_count":
            pw_tt = np.ones(n_pairs)
        elif time_bin_weighting == "uniform":
            _, inv = np.unique(tpair, return_inverse=True)
            nP_t = np.bincount(inv)
            pw_tt = 1.0 / nP_t[inv]
        elif time_bin_weighting == "trial_count":
            _, inv = np.unique(tpair, return_inverse=True)
            nP_t = np.bincount(inv).astype(float)
            nt_per_pair = np.array([nt_by_T[int(u)] for u in tpair], dtype=float)
            pw_tt = nt_per_pair / nP_t[inv]
        else:
            raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
        pw = pw_q * pw_tt
        pw = pw / pw.sum()
        prod = (S_c[gi].T * pw) @ S_c[gj]
        Crate = 0.5 * (prod + prod.T)

    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)
    noise_corr = cov_to_corr(CnoiseC)
    with np.errstate(divide="ignore", invalid="ignore"):
        fano = np.diag(CnoiseC) / Erate
        alpha = np.clip(np.diag(Cpsth) / np.diag(Crate), 0.0, 1.0)
    one_minus_alpha = 1.0 - alpha
    one_minus_alpha[~(np.diag(Crate) > 0)] = np.nan

    return {
        "Ctotal": Ctotal, "Cpsth": Cpsth, "Crate": Crate, "CnoiseC": CnoiseC,
        "noise_corr": noise_corr, "fano": fano, "Erate": Erate,
        "one_minus_alpha": one_minus_alpha,
        "n_time_bins": int(len(np.unique(T))),
        "n_samples": int(len(T)),
        "n_close_pairs": int(n_pairs),
    }


# ---------------------------------------------------------------------------
# Deterministic-rate variance decomposition (model digital twin)
# ---------------------------------------------------------------------------

def rate_variance_components(rate, valid=None, min_trials_per_phase=2):
    """Decompose a deterministic rate matrix into PSTH and FEM variance.

    Model-side analogue of the empirical LOTC decomposition. Returns, per
    neuron, the fraction of rate variance driven by fixational eye movements
    (``one_minus_alpha``) via a one-way random-effects ANOVA of the rate grouped
    by stimulus phase (time bin). Because the digital twin is deterministic
    (no Poisson observation noise), the eye-distance close-pair apparatus is
    unnecessary and the decomposition collapses to the ANOVA below.

    Group rates by phase t (n_t valid trials, T kept phases, N total samples);
    with phase mean rhat_bar(t) and grand mean rhat_bar:

        SS_within  = Sum_t Sum_i (rhat[i,t] - rhat_bar(t))^2     df = N - T
        SS_between = Sum_t n_t (rhat_bar(t) - rhat_bar)^2        df = T - 1
        MS_within  = SS_within / (N - T)   ;  MS_between = SS_between / (T - 1)
        n0 = (N - Sum_t n_t^2 / N) / (T - 1)     (= n when balanced)

        sigma2_within  (FEM)  = MS_within
        sigma2_between (PSTH) = max((MS_between - MS_within) / n0, 0)
        one_minus_alpha       = clip(sigma2_within / (sigma2_within + sigma2_between), 0, 1)

    The MS_between debias term (subtracting MS_within / n0) removes the
    finite-n_t inflation of the between-phase variance; skipping it would inflate
    alpha (deflate 1-alpha). 1-alpha is invariant to affine rescaling of rhat.

    Parameters
    ----------
    rate : ndarray (n_trials, n_phases)
        Per-trial deterministic rate for ONE neuron; columns are stimulus phase.
        Non-finite entries are treated as missing.
    valid : ndarray (n_trials, n_phases) of bool, optional
        Mask of usable samples; combined with isfinite(rate).
    min_trials_per_phase : int
        Phases with fewer valid trials than this are dropped (>= 2 needed).

    Returns
    -------
    dict: sigma2_within, sigma2_between, sigma2_total, one_minus_alpha (float),
          n_phases, n_samples (int).
    """
    rate = np.asarray(rate, dtype=np.float64)
    if rate.ndim != 2:
        raise ValueError(f"rate must be 2D (trials, phases), got {rate.shape}")

    finite = np.isfinite(rate)
    if valid is None:
        valid = finite
    else:
        valid = np.asarray(valid, dtype=bool) & finite

    nan_result = {
        "sigma2_within": np.nan, "sigma2_between": np.nan,
        "sigma2_total": np.nan, "one_minus_alpha": np.nan,
        "n_phases": 0, "n_samples": 0,
    }

    r0 = np.where(valid, rate, 0.0)
    n_t = valid.sum(axis=0).astype(np.float64)
    keep = n_t >= min_trials_per_phase
    if keep.sum() < 2:
        return nan_result

    n_t = n_t[keep]
    sum_t = r0.sum(axis=0)[keep]
    sumsq_t = (r0 ** 2).sum(axis=0)[keep]
    mu_t = sum_t / n_t
    ss_within_t = sumsq_t - n_t * mu_t ** 2

    T = int(keep.sum())
    N = float(n_t.sum())
    if N <= T:
        return nan_result

    grand = (n_t * mu_t).sum() / N
    ss_within = float(ss_within_t.sum())
    ss_between = float((n_t * (mu_t - grand) ** 2).sum())

    ms_within = ss_within / (N - T)
    ms_between = ss_between / (T - 1)
    n0 = (N - (n_t ** 2).sum() / N) / (T - 1)

    sigma2_within = ms_within
    sigma2_between = max((ms_between - ms_within) / n0, 0.0)
    sigma2_total = sigma2_within + sigma2_between

    if sigma2_total <= 0:
        one_minus_alpha = np.nan
    else:
        one_minus_alpha = float(np.clip(sigma2_within / sigma2_total, 0.0, 1.0))

    return {
        "sigma2_within": sigma2_within,
        "sigma2_between": sigma2_between,
        "sigma2_total": sigma2_total,
        "one_minus_alpha": one_minus_alpha,
        "n_phases": T,
        "n_samples": int(N),
    }


def psth_variance_splithalf(rate, valid=None, min_trials_per_phase=2,
                            n_boot=50, seed=0):
    """Bagged split-half estimate of the between-phase (PSTH) variance.

    Assumption-light cross-check on the analytic ``sigma2_between`` from
    :func:`rate_variance_components`. At each phase the valid trials are randomly
    split into disjoint halves A and B; the cross-covariance of the two
    half-PSTHs across phases cancels the per-phase finite-trial sampling noise in
    expectation. Averaged ("bagged") over ``n_boot`` random splits.

    Returns the bagged split-half between-phase variance (may be negative for a
    near-zero-PSTH cell), or NaN if fewer than two phases qualify.
    """
    rate = np.asarray(rate, dtype=np.float64)
    finite = np.isfinite(rate)
    if valid is None:
        valid = finite
    else:
        valid = np.asarray(valid, dtype=bool) & finite

    n_trials, n_phases = rate.shape
    phase_trials = []
    for t in range(n_phases):
        idx = np.where(valid[:, t])[0]
        if len(idx) >= min_trials_per_phase:
            phase_trials.append((t, idx))
    if len(phase_trials) < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_boot):
        mu_a, mu_b = [], []
        for t, idx in phase_trials:
            perm = rng.permutation(idx)
            mid = len(perm) // 2
            if mid < 1 or len(perm) - mid < 1:
                continue
            mu_a.append(rate[perm[:mid], t].mean())
            mu_b.append(rate[perm[mid:], t].mean())
        if len(mu_a) < 2:
            continue
        a = np.asarray(mu_a)
        b = np.asarray(mu_b)
        acc += np.mean((a - a.mean()) * (b - b.mean())) * len(a) / (len(a) - 1)
    return acc / n_boot


def pipeline_one_minus_alpha(rate, eyepos, valid=None, threshold=0.05,
                             min_trials_per_phase=10, target="full",
                             cpsth_method="mcfarland", closepair_density="direct",
                             time_bin_weighting="pair_count", density="kde",
                             weight_clip=1e6, n_bins=None, n_boot=20, seed=42,
                             device=None):
    """Per-cell 1-alpha via the empirical close-pair estimator on deterministic
    rates ("estimator B").

    The SAME machinery the empirical pipeline uses on real spikes -- the
    distribution-matched close-pair rate covariance and the PSTH covariance --
    but applied to a DETERMINISTIC multi-neuron rate field rather than noisy
    spike counts, so the model and the neurons can be placed on the same
    estimator (see ``ryan/methods_eyepos_matching/note_anova.md``). Each
    (trial, phase) is one sample; phases are the stimulus time bins; close pairs
    are distinct-trial pairs whose instantaneous eye positions lie within
    ``threshold`` degrees.

    Defaults to the production ``target='full'`` matched estimator. ``n_bins``,
    ``device`` are accepted for backward compatibility and ignored (the
    estimator is single-bin numpy).

    Parameters
    ----------
    rate : ndarray (n_trials, n_phases, n_cells)
    eyepos : ndarray (n_trials, n_phases, 2)
    valid : ndarray (n_trials, n_phases) of bool, optional

    Returns
    -------
    dict: one_minus_alpha, crate_diag, cpsth_diag (n_cells,); n_phases, n_samples.
    """
    rate = np.asarray(rate, dtype=np.float64)
    eyepos = np.asarray(eyepos, dtype=np.float64)
    if rate.ndim != 3:
        raise ValueError(f"rate must be 3D (trials, phases, cells), got {rate.shape}")

    out = decompose(
        rate, eyepos, target=target, valid=valid, threshold=threshold,
        density=density, weight_clip=weight_clip, n_boot=n_boot, seed=seed,
        min_trials_per_time_bin=min_trials_per_phase,
        time_bin_weighting=time_bin_weighting, cpsth_method=cpsth_method,
        closepair_density=closepair_density,
    )
    crate_diag = np.diag(out["Crate"]).astype(np.float64).copy()
    cpsth_diag = np.diag(out["Cpsth"]).astype(np.float64).copy()
    return {
        "one_minus_alpha": out["one_minus_alpha"],
        "crate_diag": crate_diag,
        "cpsth_diag": cpsth_diag,
        "n_phases": out["n_time_bins"],
        "n_samples": out["n_samples"],
    }


# ---------------------------------------------------------------------------
# Trial alignment for fixRSVP data
# ---------------------------------------------------------------------------

def align_fixrsvp_trials(dset, valid_time_bins=120, min_fix_dur=20,
                         min_total_spikes=200, fixation_radius=1.0,
                         fixation_center="origin", require_dpi_valid=False):
    """Extract trial-aligned robs, eyepos, valid_mask from a fixRSVP DictDataset.

    Converts flat (T, ...) covariate arrays into trial-aligned
    (n_trials, n_time, ...) arrays using trial_inds and psth_inds, filters by
    fixation and trial duration, and selects neurons by spike count.

    Parameters
    ----------
    dset : DictDataset
        Raw fixRSVP dataset with covariates: robs, eyepos, trial_inds, psth_inds.
    valid_time_bins : int
        Maximum number of within-trial time bins to retain.
    min_fix_dur : int
        Minimum number of fixation time bins for a trial to be included.
    min_total_spikes : int
        Minimum total spike count for a neuron to be included.
    fixation_radius : float
        Maximum eye distance from the fixation center (degrees) to count as
        fixation.
    fixation_center : {'origin', 'median_valid'}
        Reference point for the fixation radius. 'origin' uses (0, 0);
        'median_valid' uses the per-session median of finite eye positions.
    require_dpi_valid : bool
        If True and the dataset exposes a ``dpi_valid`` / ``eye_valid`` covariate,
        additionally require it to be true for a bin to count as fixation.

    Returns
    -------
    robs, eyepos_out, valid_mask, neuron_mask, metadata
        ``robs`` is None (with metadata) if there is insufficient data.
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

    if fixation_center == "median_valid":
        finite_eye = np.isfinite(eyepos_flat).all(axis=1)
        if finite_eye.any():
            center = np.median(eyepos_flat[finite_eye], axis=0)
        else:
            center = np.zeros(2)
    elif fixation_center == "origin":
        center = np.zeros(2)
    else:
        raise ValueError(f"unknown fixation_center: {fixation_center!r}")

    fixation = np.hypot(eyepos_flat[:, 0] - center[0],
                        eyepos_flat[:, 1] - center[1]) < fixation_radius

    if require_dpi_valid:
        for key in ("dpi_valid", "eye_valid"):
            if key in covs:
                fixation = fixation & np.asarray(covs[key]).ravel().astype(bool)
                break

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

    good_trials = fix_dur > min_fix_dur
    if good_trials.sum() < 2:
        return None, None, None, None, {"n_trials_total": NT, "n_trials_good": 0,
                                        "n_neurons_total": NC, "n_neurons_used": 0}

    robs_aligned = robs_aligned[good_trials]
    eyepos_aligned = eyepos_aligned[good_trials]

    T_use = min(valid_time_bins, T)
    iix = np.arange(T_use)
    robs_trunc = robs_aligned[:, iix]
    eyepos_trunc = eyepos_aligned[:, iix]

    neuron_mask = np.where(np.nansum(robs_trunc, axis=(0, 1)) > min_total_spikes)[0]
    if len(neuron_mask) < 3:
        return None, None, None, None, {"n_trials_total": NT,
                                        "n_trials_good": int(good_trials.sum()),
                                        "n_neurons_total": NC, "n_neurons_used": 0}

    robs_out = robs_trunc[:, :, neuron_mask]
    valid_mask = (np.isfinite(np.sum(robs_out, axis=2))
                  & np.isfinite(np.sum(eyepos_trunc, axis=2)))

    metadata = {
        "n_trials_total": NT,
        "n_trials_good": int(good_trials.sum()),
        "n_neurons_total": NC,
        "n_neurons_used": len(neuron_mask),
    }

    return robs_out, eyepos_trunc, valid_mask, neuron_mask, metadata
