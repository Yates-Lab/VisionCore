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

  target='naive'   : reproduce the current pipeline -- close-pair 2nd moment over p^2
                     but mean over p (an INCONSISTENT mix; not a variance under any q).
  target='full'    : Direction 1, q = p   -> close-pair weight 1/p, sample weight 1.
  target='central' : Direction 2, q = p^2 -> close-pair weight 1,   sample weight p.

The two consistent targets coincide iff the stimulus is homogeneous; their gap is a
direct measure of non-homogeneity.
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


def _split_half_psth_cov(S, T, sw, n_boot, seed, min_tpp):
    """Split-half PSTH covariance with per-sample target weights sw.

    The phase mean at phase t is the sw-weighted mean of its trials (=> E_q[r|t]).
    Phases are combined with pair-count (n_t^2) weighting, matching the empirical
    pipeline, and the cross-covariance of disjoint trial halves cancels Poisson.
    """
    rng = np.random.default_rng(seed)
    phases = [t for t in np.unique(T) if (T == t).sum() >= min_tpp]
    if len(phases) < 2:
        return np.full((S.shape[1], S.shape[1]), np.nan)
    idx_by_t = {t: np.where(T == t)[0] for t in phases}
    nt = np.array([len(idx_by_t[t]) for t in phases], float)
    wph = nt * (nt - 1) / 2
    wph = wph / wph.sum()

    C = np.zeros((S.shape[1], S.shape[1]))
    for _ in range(n_boot):
        A, B = [], []
        for t in phases:
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


def _close_pair_second_moment(S, E, T, phat, target, threshold, weight_clip):
    """Importance-reweighted close-pair second moment MM (C, C).

    All distinct same-phase trial pairs with |e_i - e_j| < threshold are pooled
    globally (matching the empirical ``below_threshold`` intercept). Pair (i,j) at
    midpoint m gets weight pw:
        target in {'naive','central'}: pw = 1            (p^2 sampling kept)
        target == 'full'             : pw = 1/phat(m)    (reweight p^2 -> p), clipped
    """
    C = S.shape[1]
    Si_acc, Sj_acc, mid_acc = [], [], []
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
    if not Si_acc:
        return np.full((C, C), np.nan)
    gi = np.concatenate(Si_acc); gj = np.concatenate(Sj_acc)
    mid = np.concatenate(mid_acc)
    if target == "full":
        pw = 1.0 / np.clip(phat(mid), 1e-12, None)
        pw = np.clip(pw, None, weight_clip * np.median(pw))
    else:
        pw = np.ones(len(gi))
    pw = pw / pw.sum()
    prod = (S[gi].T * pw) @ S[gj]               # (C,C) = sum_k pw_k S_i,k outer S_j,k
    return 0.5 * (prod + prod.T)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose(counts, eye, target="full", valid=None, threshold=0.05,
              density="kde", weight_clip=1e6, n_boot=20, seed=42,
              min_trials_per_phase=10):
    """Distribution-matched LOTC decomposition of a multi-cell session.

    Parameters
    ----------
    counts : ndarray (n_trials, n_phases, n_cells)
        Spike counts (or deterministic rates) per trial and stimulus phase.
    eye : ndarray (n_trials, n_phases, 2)
        Per-(trial, phase) eye position in degrees.
    target : {'naive', 'full', 'central'}
        Eye distribution to match all terms to (see module docstring).
    valid : ndarray (n_trials, n_phases) of bool, optional
        Usable-sample mask; combined with finiteness. Phases with fewer than
        ``min_trials_per_phase`` valid trials are dropped.
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
    n_boot, seed, min_trials_per_phase : see helpers.

    Returns
    -------
    dict: Ctotal, Cpsth, Crate, CnoiseC (C,C); one_minus_alpha, fano, Erate (C,);
          noise_corr (C,C); n_phases, n_samples.
    """
    counts = np.asarray(counts, float)
    eye = np.asarray(eye, float)
    n_trials, n_phases, n_cells = counts.shape

    finite = np.isfinite(counts).all(2) & np.isfinite(eye).all(2)
    valid = finite if valid is None else (np.asarray(valid, bool) & finite)
    keep_phase = valid.sum(0) >= min_trials_per_phase
    mask = valid & keep_phase[None, :]

    tr, ph = np.where(mask)
    S = counts[tr, ph, :]            # (N, C)
    E = eye[tr, ph, :]               # (N, 2)
    T = ph                           # phase label per sample

    phat = _density_fn(E, density)
    p_samp = np.clip(phat(E), 1e-12, None)

    # per-sample target weight: 'central' reweights p -> p^2 (weight prop to p_hat)
    if target == "central":
        sw = p_samp.copy()
    else:                            # 'naive' and 'full' keep full-p sampling
        sw = np.ones(len(S))

    Erate = _weighted_mean(S, sw)
    Ctotal = _weighted_cov(S, sw)
    Cpsth = _split_half_psth_cov(S, T, sw, n_boot, seed, min_trials_per_phase)
    MM = _close_pair_second_moment(S, E, T, phat, target, threshold, weight_clip)
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
        "n_phases": int(keep_phase.sum()), "n_samples": int(mask.sum()),
    }
