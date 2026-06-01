r"""Eye-position-distribution-matched Law-of-Total-Covariance decomposition.

The McFarland et al. (2016) close-pair estimator conditions on distinct-trial pairs
with eye distance below a threshold to cancel independent Poisson noise. Those close
pairs are sampled in proportion to the SQUARED eye density p(e)^2 (central), whereas
the total covariance Ctotal and the PSTH covariance Cpsth are over the full eye
distribution p(e). For a homogeneous stimulus the distribution is irrelevant; for a
non-homogeneous one the mismatch biases 1-alpha, the noise correlation Ctotal-Crate,
and the Fano factor.

This module computes the decomposition with the target eye distribution as a single
parameter, via importance reweighting toward a target q(e):

    a sample drawn from p contributes weight  q(e)/p(e)          (Ctotal, Cpsth, mean)
    a close PAIR (sampled from p^2) contributes q(e)/p(e)^2      (rate 2nd moment)
    an all-pair (sampled from p×p)  contributes q(e_i)q(e_j)/(p(e_i)p(e_j))   (Cpsth 2nd moment)

  target='naive'   : reproduce the current pipeline -- close-pair 2nd moment over p^2
                     but mean over p (an INCONSISTENT mix; not a variance under any q).
  target='full'    : Direction 1, q = p   -> close-pair weight 1/p, sample weight 1.
  target='central' : Direction 2, q = p^2 -> close-pair weight 1,   sample weight p.

The two consistent targets coincide iff the stimulus is homogeneous; their gap is a
direct measure of non-homogeneity.

The PSTH covariance Cpsth uses McFarland's all-distinct-pair second moment (Eqs. 6
and 12) by default -- the same code-path family as the close-pair Crate estimator
(Eqs. 8 and 16) with the Δe < ε filter removed. This makes Crate and Cpsth two
operating points of one estimator on the Δe axis: Crate is its Δe → 0 intercept,
Cpsth is its eye-distribution-marginalized asymptote. The bagged split-half
alternative (``cpsth_method='split_half'``) is retained for the parallel
implementation; see writeup §A.7 for the comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal

from VisionCore.covariance import cov_to_corr


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
                             time_bin_weighting="pair_count"):
    """McFarland M6/M12 all-distinct-pair second moment, target-reweighted.

    At each time bin t, every distinct trial pair (i, j) with i<j is included
    (no Δe restriction -- this is M8/Crate without the close-pair filter, i.e.
    the same conditional second moment Ceye(Δe) read off at its eye-distribution-
    marginalized asymptote rather than its Δe → 0 intercept). Per-pair weight

        pw_q(i, j) = w_i * w_j,    w_i = q(e_i)/p(e_i)

    where w_i is the per-trial importance weight that reweights the natural
    pair sampling p(e_i)·p(e_j) to q(e_i)·q(e_j):

        target ∈ {'naive', 'full'}: w_i = 1               (q = p)
        target == 'central'       : w_i = p(e_i)          (q = p^2)

    Combined with the across-bin scheme (identical to ``_close_pair_second_moment``):

        'pair_count': pw_t = 1               -- pool pairs, average per pair.
        'uniform'   : pw_t = 1/|P_t|         -- per-bin mean first, uniform across bins.

    Computed efficiently per bin using the identity

        2 * Σ_{i<j} w_i w_j S_i ⊗ S_j
            = (Σ_i w_i S_i) ⊗ (Σ_i w_i S_i) − Σ_i w_i² S_i ⊗ S_i,

    avoiding the (Σ_t n_t(n_t-1)/2) explicit pair tensor.

    Distinct-trial pairing cancels Poisson AND simultaneous cross-cell noise in
    expectation for the same reason as McFarland's M6/M8: trial i and trial j
    have independent noise, so neither single-cell observation noise nor
    same-trial cross-cell noise correlations leak into the average.
    """
    if target == "central":
        w_trial = np.clip(phat(E), 1e-12, None)
    else:                                # 'naive' and 'full' -- q = p marginal
        w_trial = np.ones(len(S))

    C = S.shape[1]
    num = np.zeros((C, C), dtype=np.float64)
    denom = 0.0

    for t in np.unique(T):
        ix = np.where(T == t)[0]
        if len(ix) < 2:
            continue
        w_t = w_trial[ix]
        S_t = S[ix]

        wS_sum = (w_t[:, None] * S_t).sum(0)          # (C,)
        outer_sum = np.outer(wS_sum, wS_sum)          # Σ_{i,j} w_i w_j S_i ⊗ S_j
        wS2 = (w_t[:, None] ** 2) * S_t               # (n_t, C)
        diag_term = wS2.T @ S_t                       # Σ_i w_i² S_i ⊗ S_i
        pair_sum_t = 0.5 * (outer_sum - diag_term)    # Σ_{i<j} w_i w_j S_i ⊗ S_j

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
    but this one uses only the cross-half subset of pairs per split, recovering
    the all-pairs M6 estimator in expectation as ``n_boot → ∞``. Retained so
    we can fall back to a bootstrap-based estimator if its side benefits
    (natural SEM across splits, lower memory at very large C) become decisive;
    not the default path used by ``decompose``. See writeup §A.7 for the
    comparison.

    The per-time-bin mean at bin t is the sw-weighted mean of its trials
    (=> E_q[r|t]). Time bins are combined with one of two weightings:

      * ``'pair_count'`` (default, matched): w_t = n_t(n_t-1)/2, the same weight
        the close-pair rate estimator implicitly uses (its pool ~ n_t pairs per
        time bin). This is the post-fix pipeline (covariance.py).
      * ``'uniform'``: w_t = 1, the pre-fix historical Cpsth weighting (1/T per
        time bin). Provided so the Extension-1 bias under variable n_t can be
        exhibited against the matched case.

    The cross-covariance of disjoint trial halves cancels Poisson regardless of
    the weighting.
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


def _close_pair_second_moment(S, E, T, phat, target, threshold, weight_clip,
                              time_bin_weighting="pair_count"):
    """Importance-reweighted close-pair second moment MM (C, C).

    All distinct same-time-bin trial pairs with |e_i - e_j| < threshold are
    collected. Pair (i,j) at midpoint m gets weight pw combining a target-
    distribution importance weight and an across-bin combination scheme:

      target in {'naive','central'}: pw_q = 1            (p^2 sampling kept)
      target == 'full'             : pw_q = 1/phat(m)    (reweight p^2 -> p), clipped

      time_bin_weighting == 'pair_count': pw_t = 1 -- pool all close pairs and
        average uniformly per pair; bin t contributes ~|P_t| ≈ n_t(n_t-1)/2.
      time_bin_weighting == 'uniform':    pw_t = 1/|P_t| -- per-bin mean of pair
        products first, then uniform across bins; bin t contributes 1/T regardless
        of |P_t|. This is the literal reading of McFarland's nested bracket
        <<.>_{i≠j}>_t in Eq. M8.

    Under constant n_t the two time_bin_weighting choices coincide.
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
        pw_q = 1.0 / np.clip(phat(mid), 1e-12, None)
        pw_q = np.clip(pw_q, None, weight_clip * np.median(pw_q))
    else:
        pw_q = np.ones(len(gi))

    if time_bin_weighting == "pair_count":
        pw_t = np.ones(len(gi))
    elif time_bin_weighting == "uniform":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv)
        pw_t = 1.0 / nP_t[inv]
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")

    pw = pw_q * pw_t
    pw = pw / pw.sum()
    prod = (S[gi].T * pw) @ S[gj]               # (C,C) = sum_k pw_k S_i,k outer S_j,k
    return 0.5 * (prod + prod.T)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose(counts, eye, target="full", valid=None, threshold=0.05,
              density="kde", weight_clip=1e6, n_boot=20, seed=42,
              min_trials_per_time_bin=10, time_bin_weighting="pair_count",
              cpsth_method="mcfarland"):
    """Distribution-matched LOTC decomposition of a multi-cell session.

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
        Effectively off by default (1e6): for an eccentric-sensitive cell the FEM
        variance lives in the periphery where close pairs are sparse and must be
        upweighted by 1/p, so capping trades the resulting tail variance for bias.
        This unbounded-weight cost is precisely why target='central' (bounded
        weights, prop to p) is the more stable choice.
    time_bin_weighting : {'pair_count', 'uniform'}
        Across-bin combination used by ALL three estimators (Ctotal, Cpsth,
        Crate's close-pair MM and Ybar^2 subtractor). Both are consistent
        directions for the LOTC under variable n_t (see writeup §3.2):
          * 'pair_count' (default): bin t weighted ∝ n_t(n_t-1)/2 -- pool close
            pairs and average uniformly per pair; inverse-variance optimal on
            the close-pair pool but concentrates on high-n_t bins.
          * 'uniform': bin t weighted 1/T regardless of n_t -- per-bin mean of
            pair products first, then uniform across bins; matches McFarland's
            literal nested-bracket reading of Eqs. M8/Eq.6 but pays a variance
            penalty under sharply variable n_t.
        Under constant n_t the two coincide.
    cpsth_method : {'mcfarland', 'split_half'}
        How to debias the PSTH covariance against same-time-bin observation
        noise (single-cell Poisson plus simultaneous cross-cell noise
        correlations). Both estimators pair distinct trials at the same bin so
        their noise is independent; the difference is computational form.
          * 'mcfarland' (default): McFarland Eq. 6/12 -- the all-distinct-pair
            second moment, with the close-pair filter of M8/Crate removed.
            Same code-path family as the Crate close-pair estimator, just at
            the eye-distribution-marginalized end of the Ceye(Δe) axis.
            Deterministic; minimum variance for fixed data.
          * 'split_half': bagged split-half (n_boot, seed). Stochastic; targets
            the same population quantity in expectation; converges to
            'mcfarland' as n_boot → ∞.
        See writeup §A.7 for the comparison.
    n_boot, seed, min_trials_per_time_bin : see helpers.

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

    phat = _density_fn(E, density)
    p_samp = np.clip(phat(E), 1e-12, None)

    # per-sample target weight: 'central' reweights p -> p^2 (weight prop to p_hat)
    if target == "central":
        tw = p_samp.copy()
    else:                            # 'naive' and 'full' keep full-p sampling
        tw = np.ones(len(S))

    # per-sample time-bin compensation so total weight at bin t matches the chosen
    # across-bin scheme. 'pair_count' (matched): n_t * pw_t = n_t(n_t-1)/2 (the
    # pair count, identical to MM's intrinsic time-bin weighting). 'uniform': n_t *
    # pw_t = 1 (the pre-fix historical Cpsth weighting). Under constant n_t both
    # collapse to a constant and cancel in normalization, preserving every
    # constant-n_t behavior (including ``test_naive_path_matches_existing_pipeline``).
    nt_by_t = {t: int((T == t).sum()) for t in np.unique(T)}
    if time_bin_weighting == "pair_count":
        pw_t = {t: max(nt_by_t[t] - 1, 0) / 2.0 for t in nt_by_t}
    elif time_bin_weighting == "uniform":
        pw_t = {t: (1.0 / nt_by_t[t]) if nt_by_t[t] > 0 else 0.0 for t in nt_by_t}
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    time_bin_w = np.array([pw_t[t] for t in T], dtype=float)
    sw = tw * time_bin_w

    Erate = _weighted_mean(S, sw)
    Ctotal = _weighted_cov(S, sw)
    if cpsth_method == "mcfarland":
        MM_psth = _all_pairs_second_moment(S, E, T, phat, target,
                                           time_bin_weighting=time_bin_weighting)
        Cpsth = MM_psth - np.outer(Erate, Erate)
    elif cpsth_method == "split_half":
        Cpsth = _split_half_psth_cov(S, T, sw, n_boot, seed,
                                     min_trials_per_time_bin,
                                     time_bin_weighting=time_bin_weighting)
    else:
        raise ValueError(f"unknown cpsth_method: {cpsth_method!r}")
    MM = _close_pair_second_moment(S, E, T, phat, target, threshold, weight_clip,
                                   time_bin_weighting=time_bin_weighting)
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


# ---------------------------------------------------------------------------
# §4.6: trajectory-mode estimator (multi-bin eye trajectories)
# ---------------------------------------------------------------------------

def _rms_traj_close_pairs(trajectories, T_idx, threshold):
    """Enumerate same-T_idx close pairs by RMS trajectory distance.

    Mirrors VisionCore/covariance.py::compute_eye_distances: the per-pair
    distance is the L2 distance of the flattened trajectory vectors, scaled
    by 1/sqrt(t_window), so the threshold has the same per-bin-distance units
    as the single-bin §5.2 scheme (and the flat-trajectory limit of this
    estimator coincides with `decompose` at the same threshold).
    """
    N, T, _ = trajectories.shape
    inv_sqrt_T = 1.0 / float(np.sqrt(float(T)))
    flat = trajectories.reshape(N, -1)                       # (N, T*2)
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


def _pooled_per_bin_kde(per_bin_positions):
    """Fit a 2-D gaussian_kde on pooled per-bin positions.

    Returns a callable p_hat: (M, 2) -> (M,). Bandwidth uses ``scipy``'s
    Scott's-rule default on the *pooled* point count: within-trajectory
    correlation means the effective sample size is < N*T, so this slightly
    under-smooths in dense regions. Practical impact is small at the per-bin
    drift / centroid spread ratios typical for fixational data (writeup §4.6).
    """
    E = np.asarray(per_bin_positions, dtype=float).reshape(-1, 2)
    finite = np.isfinite(E).all(axis=1)
    E = E[finite]
    kde = gaussian_kde(E.T)
    return lambda X: kde(np.asarray(X, dtype=float).reshape(-1, 2).T)


def decompose_trajectory(counts, trajectories, T_idx, target="full",
                         valid=None, threshold=0.05, weight_clip=1e6,
                         n_boot=20, seed=42, min_trials_per_time_bin=10,
                         time_bin_weighting="pair_count",
                         cpsth_method="mcfarland"):
    """Trajectory-mode distribution-matched LOTC decomposition (writeup §4.6).

    The multi-bin extension of :func:`decompose`. Each sample is a window of
    ``t_window`` contiguous bins of eye trajectory rather than a single
    eye-position observation; close pairs are filtered by RMS trajectory
    distance (matching ``VisionCore/covariance.py``); the importance-weight
    density is estimated by two pooled-per-bin 2-D KDEs:

      p_marg     = gaussian_kde over pooled per-bin positions of all samples
      p_cp,marg  = gaussian_kde over pooled per-bin positions of close-pair
                   *midpoint* trajectories (recipe (b) of writeup §4.6).

    In the flat-trajectory limit (zero within-window drift) the ratio
    p_cp,marg(e)/p_marg(e) collapses to the centroid density p_centroid(e), so
    evaluating that ratio (or its inverse) at the trajectory centroid recovers
    §4.4's importance weights exactly. For non-zero within-window drift the
    ratio is a smoothed proxy for p_centroid, biased by an amount controlled
    by sigma_drift / sigma_eye (validated empirically in fig_trajectory.py).

    Parameters
    ----------
    counts : ndarray (N, n_cells)
        Per-sample window-integrated spike counts (or deterministic rates).
    trajectories : ndarray (N, t_window, 2)
        Per-sample per-bin eye trajectories.
    T_idx : ndarray (N,)
        Analysis time bin index per sample.
    target : {'naive', 'full', 'central'}
        Eye distribution to match all terms to:
          * 'naive'  : reproduce the unmatched estimator (no reweight).
          * 'full'   : Direction 1, target = p_marg (the actual per-bin marginal).
                       Per-pair weight at midpoint centroid c_mid:
                       p_marg(c_mid) / p_cp,marg(c_mid)   (the inverse ratio;
                       reduces to 1/p_centroid in the flat limit).
                       Per-sample weight 1.
          * 'central': Direction 2, target = p_cp,marg (the close-pair density).
                       Per-sample weight at centroid c_i:
                       p_cp,marg(c_i) / p_marg(c_i)       (the ratio;
                       reduces to p_centroid in the flat limit).
                       Per-pair weight 1.
    valid : ndarray (N,) of bool, optional
        Sample-level validity mask; combined with finiteness.
    threshold : float
        Close-pair RMS trajectory threshold (per-bin distance units; see
        :func:`_rms_traj_close_pairs`).
    weight_clip, time_bin_weighting, cpsth_method, n_boot, seed,
    min_trials_per_time_bin : see :func:`decompose`.

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

    # keep only T_idx bins with enough valid trials
    keep_mask = np.zeros(N, bool)
    keep_T = []
    for t in np.unique(T_idx[valid]):
        ix = np.where(valid & (T_idx == t))[0]
        if len(ix) >= min_trials_per_time_bin:
            keep_mask[ix] = True
            keep_T.append(t)
    keep_T = np.asarray(keep_T)

    S = counts[keep_mask]                                    # (N', C)
    Tr = trajectories[keep_mask]                             # (N', t_window, 2)
    T = T_idx[keep_mask]                                     # (N',)
    centroids = Tr.mean(axis=1)                              # (N', 2)
    n_cells = S.shape[1]

    # close pairs (RMS trajectory filter)
    gi, gj, tpair, mid_traj = _rms_traj_close_pairs(Tr, T, threshold)
    n_pairs = len(gi)

    # densities
    p_marg = _pooled_per_bin_kde(Tr.reshape(-1, 2))
    if n_pairs > 0:
        p_cp_marg = _pooled_per_bin_kde(mid_traj.reshape(-1, 2))
    else:
        p_cp_marg = None

    # per-sample target weight (centroid-evaluated)
    p_marg_c = np.clip(p_marg(centroids), 1e-12, None)
    if target == "central":
        if p_cp_marg is None:
            return _nan_decompose(n_cells, len(T), N, n_pairs)
        ratio_samp = np.clip(p_cp_marg(centroids), 1e-12, None) / p_marg_c
        tw = ratio_samp
    elif target in ("naive", "full"):
        tw = np.ones(len(S))
    else:
        raise ValueError(f"unknown target: {target!r}")

    # per-sample time-bin compensation (same scheme as decompose)
    nt_by_T = {int(t): int((T == t).sum()) for t in np.unique(T)}
    if time_bin_weighting == "pair_count":
        pw_t_by_T = {t: max(nt_by_T[t] - 1, 0) / 2.0 for t in nt_by_T}
    elif time_bin_weighting == "uniform":
        pw_t_by_T = {t: (1.0 / nt_by_T[t]) if nt_by_T[t] > 0 else 0.0
                     for t in nt_by_T}
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    tb_w = np.array([pw_t_by_T[int(t)] for t in T], dtype=float)
    sw = tw * tb_w

    Erate = _weighted_mean(S, sw)
    Ctotal = _weighted_cov(S, sw)

    # Centred S for the close-pair and all-pairs second moments. The summed-
    # rate count has mean ~ t_window * mu_0 = O(t_window) while the variance
    # signal is O(1), so the literal "second moment minus Erate^2" subtraction
    # in the single-bin formula loses precision badly at the smaller sample
    # sizes typical of the trajectory-mode validation (writeup §4.6). Centring
    # first turns each pair product (Y_i - mu)(Y_j - mu) into a small * small
    # cancellation rather than a (T*mu)^2 - (T*mu)^2 catastrophic one; the
    # identity Σ_{i<j} w_i w_j (Y_i-μ)(Y_j-μ) = ½[(Σw_i(Y_i-μ))² - Σw_i²(Y_i-μ)²]
    # is the same M6 algebraic trick, just evaluated on centred inputs.
    S_c = S - Erate[None, :]

    # Cpsth: McFarland M6 / split-half, with trajectory-centroid w_trial
    if target == "central":
        w_trial = ratio_samp.copy()
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

    # Crate: close-pair (centred) cross-product with target-importance weight
    # at the midpoint trajectory centroid.
    if n_pairs == 0:
        Crate = np.full((n_cells, n_cells), np.nan)
    else:
        mid_centroids = mid_traj.mean(axis=1)                        # (P, 2)
        if target == "full":
            num = np.clip(p_marg(mid_centroids), 1e-12, None)
            den = np.clip(p_cp_marg(mid_centroids), 1e-12, None)
            pw_q = num / den
            pw_q = np.clip(pw_q, None, weight_clip * np.median(pw_q))
        else:                                                        # naive / central
            pw_q = np.ones(n_pairs)
        if time_bin_weighting == "pair_count":
            pw_tt = np.ones(n_pairs)
        else:                                                        # uniform
            _, inv = np.unique(tpair, return_inverse=True)
            nP_t = np.bincount(inv)
            pw_tt = 1.0 / nP_t[inv]
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


def _all_pairs_second_moment_with_weights(S, w_trial, T,
                                          time_bin_weighting="pair_count"):
    """McFarland M6 all-distinct-pair second moment with explicit per-trial
    weights ``w_trial`` (so the trajectory-mode central-target Cpsth uses the
    same p_cp,marg/p_marg ratio as its sample weight). Same algebraic identity
    as :func:`_all_pairs_second_moment` -- factored out so the weighting source
    (single-bin p_hat vs. trajectory pooled-per-bin ratio) can vary."""
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
        else:
            raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")
    if denom <= 0:
        return np.full((C, C), np.nan)
    prod = num / denom
    return 0.5 * (prod + prod.T)


def _nan_decompose(n_cells, n_time_bins, n_samples, n_pairs):
    nan_C = np.full((n_cells, n_cells), np.nan)
    nan_v = np.full(n_cells, np.nan)
    return {
        "Ctotal": nan_C, "Cpsth": nan_C, "Crate": nan_C, "CnoiseC": nan_C,
        "noise_corr": nan_C, "fano": nan_v, "Erate": nan_v,
        "one_minus_alpha": nan_v,
        "n_time_bins": int(n_time_bins),
        "n_samples": int(n_samples),
        "n_close_pairs": int(n_pairs),
    }
