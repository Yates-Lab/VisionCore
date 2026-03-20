"""General-purpose statistical utilities for neural data analysis."""
import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests


def geomean(x, axis=None, eps=1e-12):
    """Geometric mean of positive values. Returns NaN if no positive values."""
    x = np.asarray(x, dtype=float)
    if axis is None:
        pos = x[np.isfinite(x) & (x > eps)]
        return np.exp(np.mean(np.log(pos))) if len(pos) > 0 else np.nan
    # axis case: mask non-positive to NaN and use nanmean of log
    masked = np.where((x > eps) & np.isfinite(x), x, np.nan)
    return np.exp(np.nanmean(np.log(masked), axis=axis))


def iqr_25_75(x):
    """Return (q25, q75) of finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


def bootstrap_mean_ci(x, nboot=5000, ci=0.95, seed=0):
    """
    Bootstrap confidence interval for the mean.

    Returns: (mean, (ci_lo, ci_hi))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(nboot, x.size))
    boot_means = x[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return float(x.mean()), (float(lo), float(hi))


def bootstrap_paired_diff_ci(a, b, nboot=5000, ci=0.95, seed=0):
    """
    Bootstrap CI for mean(a - b) using paired resampling.

    Returns: (mean_diff, (ci_lo, ci_hi))
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(nboot, diff.size))
    boot_means = diff[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return float(diff.mean()), (float(lo), float(hi))


def fisher_z(rho, eps=1e-6):
    """Fisher z-transform: arctanh(rho), clipped to avoid infinity."""
    rho = np.asarray(rho, dtype=float)
    return np.arctanh(np.clip(rho, -1 + eps, 1 - eps))


def fisher_z_mean(rho, eps=1e-6):
    """Mean of Fisher z-transformed correlations."""
    z = fisher_z(rho, eps)
    z = z[np.isfinite(z)]
    return float(np.mean(z)) if len(z) > 0 else np.nan


def emp_p_one_sided(null, obs, direction="greater"):
    """
    Empirical p-value with +1 smoothing.

    direction='greater': P(null >= obs)
    direction='less':    P(null <= obs)
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    n = len(null)
    if n == 0:
        return np.nan
    if direction == "greater":
        return float((np.sum(null >= obs) + 1) / (n + 1))
    else:
        return float((np.sum(null <= obs) + 1) / (n + 1))


def wilcoxon_signed_rank(a, b, alternative="two-sided"):
    """
    Wilcoxon signed-rank test on paired samples.

    Returns: (statistic, p_value)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    result = sp_stats.wilcoxon(a[mask], b[mask], alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def fdr_correct(pvals, q=0.05, method="fdr_bh"):
    """
    Benjamini-Hochberg FDR correction.

    Returns: (adjusted_pvals, significant_mask)
    """
    pvals = np.asarray(pvals, dtype=float)
    reject, p_adj, _, _ = multipletests(pvals, alpha=q, method=method)
    return p_adj, reject


def paired_valid(a, b, positive=True):
    """
    Filter to pairs where both values are finite (and optionally positive).

    Returns: (a_valid, b_valid, mask)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if positive:
        mask &= (a > 0) & (b > 0)
    return a[mask], b[mask], mask
