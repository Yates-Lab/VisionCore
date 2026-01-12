# scripts/figure_common.py
"""
Shared utilities for all LOTC figure scripts.
This module consolidates common functions used across figures_alpha.py,
figures_fanofactors.py, figures_noisecorr.py, and figures_subspace.py.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Publication style
# ----------------------------
def set_pub_style():
    """Standard publication styling - call once at start of any figure script."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
    })


# ----------------------------
# Data sanitization
# ----------------------------
def _finite(x):
    """Filter array to finite values only."""
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _safe_mean(x):
    """Mean of finite values, or NaN if empty."""
    x = _finite(x)
    return float(np.mean(x)) if x.size else np.nan


# ----------------------------
# Descriptive statistics
# ----------------------------
def iqr_25_75(x):
    """Return 25th and 75th percentile as tuple."""
    x = _finite(x)
    if x.size == 0:
        return (np.nan, np.nan)
    q25, q75 = np.percentile(x, [25, 75])
    return (float(q25), float(q75))


def median_iqr(x):
    """Return (median, (q25, q75))."""
    x = _finite(x)
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    return med, (float(q25), float(q75))


def geomean(x, axis=None, eps=1e-12):
    """Geometric mean (positive values only, NaNs for non-positive)."""
    x = np.asarray(x, dtype=float)
    x = np.where(x <= 0, np.nan, x)
    return np.exp(np.nanmean(np.log(x + eps), axis=axis))


# ----------------------------
# Bootstrap utilities
# ----------------------------
def bootstrap_mean_ci(x, nboot=5000, ci=0.95, rng=0):
    """
    Bootstrap CI for the mean.
    Returns: (mean, (ci_lo, ci_hi))
    """
    x = _finite(x)
    if x.size < 2:
        return np.nan, (np.nan, np.nan)
    rg = np.random.default_rng(rng)
    boots = np.empty(nboot, dtype=float)
    for b in range(nboot):
        boots[b] = np.mean(rg.choice(x, size=x.size, replace=True))
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(np.mean(x)), (float(lo), float(hi))


# ----------------------------
# Shuffle array utilities
# ----------------------------
def align_shuffle(shuff, n_units=None, n_neurons=None):
    """
    Ensure shuffle array is [S, N] (shuffles x units).
    Transposes if necessary.

    Accepts either n_units or n_neurons as the size parameter.
    """
    # Accept either parameter name for backwards compatibility
    n = n_units if n_units is not None else n_neurons
    if n is None:
        raise ValueError("Must provide n_units or n_neurons")

    shuff = np.asarray(shuff, dtype=float)
    if shuff.ndim != 2:
        raise ValueError(f"shuffle must be 2D, got shape {shuff.shape}")
    if shuff.shape[1] == n:
        return shuff
    if shuff.shape[0] == n:
        return shuff.T
    # heuristic fallback
    if abs(shuff.shape[0] - n) < abs(shuff.shape[1] - n):
        return shuff.T
    return shuff


# ----------------------------
# Empirical p-values
# ----------------------------
def emp_p_one_sided(null, obs, direction="greater"):
    """
    Empirical p-value with +1 smoothing.
      direction='greater': p = P(null >= obs)
      direction='less'   : p = P(null <= obs)
    """
    null = _finite(null)
    if null.size == 0 or not np.isfinite(obs):
        return np.nan
    if direction == "greater":
        return (np.sum(null >= obs) + 1) / (null.size + 1)
    if direction == "less":
        return (np.sum(null <= obs) + 1) / (null.size + 1)
    raise ValueError("direction must be 'greater' or 'less'")


# ----------------------------
# Paired data utilities
# ----------------------------
def paired_valid(a, b, positive=True):
    """
    Filter to pairs where both are finite (and optionally positive).
    Returns: (a_valid, b_valid, mask)
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if positive:
        ok = ok & (a > 0) & (b > 0)
    return a[ok], b[ok], ok


# ----------------------------
# Fisher z transform (for correlations)
# ----------------------------
def fisher_z(rho, eps=1e-6):
    """Fisher z transform of correlation(s), clipped to avoid infinities."""
    rho = np.asarray(rho, dtype=np.float64)
    rho = np.clip(rho, -1 + eps, 1 - eps)
    return np.arctanh(rho)


def fisher_z_mean(rho, eps=1e-6):
    """Mean Fisher z across correlations."""
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    rho = rho[np.isfinite(rho)]
    if rho.size == 0:
        return np.nan
    return float(np.mean(fisher_z(rho, eps=eps)))

