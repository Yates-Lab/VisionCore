import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ----------------------------
# styling
# ----------------------------
def set_pub_style():
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
# helpers
# ----------------------------
def geomean(x, axis=None, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = np.where(x <= 0, np.nan, x)
    return np.exp(np.nanmean(np.log(x + eps), axis=axis))

def iqr_25_75(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([np.nan, np.nan])
    return np.percentile(x, [25, 75])

def paired_valid(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return a[ok], b[ok], ok

def align_shuffle(shuff, n_neurons):
    shuff = np.asarray(shuff, dtype=float)
    if shuff.ndim != 2:
        raise ValueError(f"shuffle must be 2D, got shape {shuff.shape}")
    # want [S, N]
    if shuff.shape[1] == n_neurons:
        return shuff
    if shuff.shape[0] == n_neurons:
        return shuff.T
    # fallback: transpose if it makes N closer
    if abs(shuff.shape[0] - n_neurons) < abs(shuff.shape[1] - n_neurons):
        return shuff.T
    return shuff

def fit_slope_through_origin(x, y):
    """
    Churchland-style slope of variance vs mean.
    Here we do y ~ slope * x with intercept fixed at 0.
    slope = (x^T y) / (x^T x)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x = x[ok]; y = y[ok]
    if x.size < 3:
        return np.nan, ok
    slope = np.dot(x, y) / np.dot(x, x)
    return slope, ok

def bootstrap_slope_diff(x, y_unc, y_cor, nboot=5000, rng=0):
    """
    Paired bootstrap over neurons:
      resample indices with replacement
      fit slope_unc and slope_cor
      store diff = slope_cor - slope_unc
    Returns: (slope_unc, slope_cor, diff_hat, ci_lo, ci_hi, p_one_sided)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y_unc = np.asarray(y_unc, dtype=float).reshape(-1)
    y_cor = np.asarray(y_cor, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y_unc) & np.isfinite(y_cor) & (x > 0) & (y_unc >= 0) & (y_cor >= 0)
    x = x[ok]; y_unc = y_unc[ok]; y_cor = y_cor[ok]
    n = x.size
    if n < 5:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # point estimates
    s_unc = np.dot(x, y_unc) / np.dot(x, x)
    s_cor = np.dot(x, y_cor) / np.dot(x, x)
    diff_hat = s_cor - s_unc

    rg = np.random.default_rng(rng)
    diffs = np.zeros(nboot, dtype=float)

    idx = np.arange(n)
    for b in range(nboot):
        ib = rg.choice(idx, size=n, replace=True)
        xb = x[ib]
        yub = y_unc[ib]
        ycb = y_cor[ib]
        s_u = np.dot(xb, yub) / np.dot(xb, xb)
        s_c = np.dot(xb, ycb) / np.dot(xb, xb)
        diffs[b] = s_c - s_u

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # one-sided p for reduction: diff >= 0 is "no reduction"; want diff < 0
    p_one = (np.sum(diffs >= 0) + 1) / (nboot + 1)
    return s_unc, s_cor, diff_hat, lo, hi, p_one

def bootstrap_slopes_paired(x, y_unc, y_cor, nboot=5000, ci=0.95, rng=0):
    """
    Paired bootstrap over neurons for slope-through-origin fits.
    Returns:
      s_unc_hat, s_cor_hat,
      (unc_lo, unc_hi), (cor_lo, cor_hi),
      diff_hat, (diff_lo, diff_hi),
      p_one_sided  where p = P(diff >= 0)  (evidence against reduction)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y_unc = np.asarray(y_unc, dtype=float).reshape(-1)
    y_cor = np.asarray(y_cor, dtype=float).reshape(-1)

    ok = np.isfinite(x) & np.isfinite(y_unc) & np.isfinite(y_cor) & (x > 0) & (y_unc >= 0) & (y_cor >= 0)
    x = x[ok]; y_unc = y_unc[ok]; y_cor = y_cor[ok]
    n = x.size
    if n < 5:
        return (np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan), np.nan, (np.nan, np.nan), np.nan)

    # point estimates
    s_unc_hat = np.dot(x, y_unc) / np.dot(x, x)
    s_cor_hat = np.dot(x, y_cor) / np.dot(x, x)
    diff_hat = s_cor_hat - s_unc_hat

    rg = np.random.default_rng(rng)
    idx = np.arange(n)

    s_unc_bs = np.zeros(nboot, dtype=float)
    s_cor_bs = np.zeros(nboot, dtype=float)
    diff_bs  = np.zeros(nboot, dtype=float)

    for b in range(nboot):
        ib = rg.choice(idx, size=n, replace=True)
        xb = x[ib]
        yu = y_unc[ib]
        yc = y_cor[ib]

        s_u = np.dot(xb, yu) / np.dot(xb, xb)
        s_c = np.dot(xb, yc) / np.dot(xb, xb)
        s_unc_bs[b] = s_u
        s_cor_bs[b] = s_c
        diff_bs[b]  = s_c - s_u

    alpha = (1.0 - ci) / 2.0
    qlo = 100 * alpha
    qhi = 100 * (1 - alpha)

    unc_lo, unc_hi = np.percentile(s_unc_bs, [qlo, qhi])
    cor_lo, cor_hi = np.percentile(s_cor_bs, [qlo, qhi])
    diff_lo, diff_hi = np.percentile(diff_bs, [qlo, qhi])

    # one-sided p-value for reduction: H0 is diff >= 0, want diff < 0
    p_one = (np.sum(diff_bs >= 0) + 1) / (nboot + 1)

    return (s_unc_hat, s_cor_hat,
            (unc_lo, unc_hi), (cor_lo, cor_hi),
            diff_hat, (diff_lo, diff_hi),
            p_one)


# ----------------------------
# main computation
# ----------------------------
def compute_all_fano_stats(metrics, alternative="less", fdr_q=0.05, nboot_slope=5000):
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    nW = len(metrics)

    # per-neuron summaries
    g_unc = np.full(nW, np.nan)
    g_cor = np.full(nW, np.nan)
    iqr_unc = np.full((nW, 2), np.nan)
    iqr_cor = np.full((nW, 2), np.nan)
    ratio = np.full(nW, np.nan)
    pct_red = np.full(nW, np.nan)
    p_wil = np.full(nW, np.nan)
    n_valid = np.zeros(nW, dtype=int)

    # shuffle null on ratio (geomean-based)
    null_ratio_ci95 = np.full((nW, 2), np.nan)
    p_emp_ratio = np.full(nW, np.nan)

    # slope stats
    slope_unc = np.full(nW, np.nan)
    slope_cor = np.full(nW, np.nan)
    slope_unc_ci = np.full((nW, 2), np.nan)  # NEW
    slope_cor_ci = np.full((nW, 2), np.nan)  # NEW
    slope_diff = np.full(nW, np.nan)
    slope_diff_ci = np.full((nW, 2), np.nan)
    p_slope_one = np.full(nW, np.nan)

    for i, m in enumerate(metrics):
        unc = np.asarray(m["uncorr"]).reshape(-1)
        cor = np.asarray(m["corr"]).reshape(-1)
        unc_v, cor_v, ok = paired_valid(unc, cor)
        n_valid[i] = unc_v.size

        g_unc[i] = geomean(unc_v)
        g_cor[i] = geomean(cor_v)
        iqr_unc[i] = iqr_25_75(unc_v)
        iqr_cor[i] = iqr_25_75(cor_v)

        ratio[i] = g_cor[i] / g_unc[i]
        pct_red[i] = 100.0 * (1.0 - ratio[i])

        if unc_v.size >= 5:
            p_wil[i] = wilcoxon(cor_v - unc_v, alternative=alternative, zero_method="wilcox").pvalue

        # shuffle ratio null
        sh_u = align_shuffle(m["shuff_uncorr"], n_neurons=unc.size)
        sh_c = align_shuffle(m["shuff_corr"], n_neurons=cor.size)
        # match same valid neurons as real paired analysis
        if ok is not None and ok.size == sh_u.shape[1]:
            sh_u = sh_u[:, ok]
            sh_c = sh_c[:, ok]

        # ratio per shuffle (geomean across neurons)
        ru = geomean(sh_u, axis=1)
        rc = geomean(sh_c, axis=1)
        rnull = rc / ru

        null_ratio_ci95[i] = np.percentile(rnull, [2.5, 97.5])
        # empirical one-sided p: how often null ratio <= observed ratio (as strong or stronger reduction)
        p_emp_ratio[i] = (np.sum(rnull <= ratio[i]) + 1) / (rnull.size + 1)

        # slope method: x=mean rate, y=variance
        x = np.asarray(m["erate"]).reshape(-1)
        y_unc = unc * x # variance is FF * mean
        y_cor = cor * x

        (s_u, s_c,
        (u_lo, u_hi), (c_lo, c_hi),
        d_hat, (d_lo, d_hi),
        p_one) = bootstrap_slopes_paired(x, y_unc, y_cor, nboot=nboot_slope, ci=0.95, rng=1234 + i)

        slope_unc[i] = s_u
        slope_cor[i] = s_c
        slope_unc_ci[i] = [u_lo, u_hi]
        slope_cor_ci[i] = [c_lo, c_hi]
        slope_diff[i] = d_hat
        slope_diff_ci[i] = [d_lo, d_hi]
        p_slope_one[i] = p_one

    # multiple comparisons across windows for per-neuron wilcoxon
    okp = np.isfinite(p_wil)
    p_wil_fdr = np.full_like(p_wil, np.nan)
    sig_fdr = np.zeros_like(p_wil, dtype=bool)
    if np.any(okp):
        reject, p_adj, *_ = multipletests(p_wil[okp], alpha=fdr_q, method="fdr_bh")
        p_wil_fdr[okp] = p_adj
        sig_fdr[okp] = reject

    out = dict(
        window_ms=window_ms,
        g_unc=g_unc, g_cor=g_cor,
        iqr_unc=iqr_unc, iqr_cor=iqr_cor,
        ratio=ratio, pct_red=pct_red,
        p_wil=p_wil, p_wil_fdr=p_wil_fdr, sig_fdr=sig_fdr,
        n_valid=n_valid,
        null_ratio_ci95=null_ratio_ci95,
        p_emp_ratio=p_emp_ratio,
        slope_unc=slope_unc, slope_cor=slope_cor,
        slope_unc_ci=slope_unc_ci, slope_cor_ci=slope_cor_ci,  # NEW
        slope_ratio=slope_cor / slope_unc,
        slope_diff=slope_diff, slope_diff_ci=slope_diff_ci, p_slope_one=p_slope_one
    )
    return out

# ----------------------------
# plotting: main 2x2 figure
# ----------------------------
def plot_fano_paper_figure(stats, title=None):
    set_pub_style()
    w = stats["window_ms"]
    order = np.argsort(w)
    w = w[order]

    # reorder
    def r(x): return np.asarray(x)[order]
    g_unc, g_cor = r(stats["g_unc"]), r(stats["g_cor"])
    iqr_unc, iqr_cor = r(stats["iqr_unc"]), r(stats["iqr_cor"])
    ratio = r(stats["ratio"])
    null_ratio_ci95 = r(stats["null_ratio_ci95"])
    slope_u, slope_c = r(stats["slope_unc"]), r(stats["slope_cor"])
    slope_ratio = r(stats["slope_ratio"])
    slope_diff_ci = r(stats["slope_diff_ci"])
    sig = r(stats["sig_fdr"])

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)

    # A) Per-neuron FF (gmean + IQR)
    ax = axs[0, 0]
    ax.plot(w, g_unc, marker="o", linewidth=1.8, label="Uncorrected (gmean)")
    ax.plot(w, g_cor, marker="o", linewidth=1.8, label="Corrected (gmean)")
    ax.fill_between(w, iqr_unc[:, 0], iqr_unc[:, 1], alpha=0.18, label="Uncorrected IQR")
    ax.fill_between(w, iqr_cor[:, 0], iqr_cor[:, 1], alpha=0.18, label="Corrected IQR")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Fano factor")
    ax.set_title("Per-neuron Fano factors")
    ax.legend(frameon=False, loc="best")

    # B) Per-neuron ratio with shuffle null band
    ax = axs[0, 1]
    ax.plot(w, ratio, marker="o", linewidth=1.8, label="Real ratio (corr/uncorr)")
    ax.fill_between(w, null_ratio_ci95[:, 0], null_ratio_ci95[:, 1], alpha=0.20,
                    label="Shuffle null 95% CI (ratio)")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Ratio (corr/uncorr)")
    ax.set_title("Correction effect vs shuffle control")
    ax.legend(frameon=False, loc="best")

    slope_u, slope_c = r(stats["slope_unc"]), r(stats["slope_cor"])
    slope_u_ci = r(stats["slope_unc_ci"])   # NEW
    slope_c_ci = r(stats["slope_cor_ci"])   # NEW
    slope_diff_ci = r(stats["slope_diff_ci"])

    # C) Population FF by slope with 95% bootstrap CIs (per condition)
    ax = axs[1, 0]

    u_yerr = np.vstack([slope_u - slope_u_ci[:, 0], slope_u_ci[:, 1] - slope_u])
    c_yerr = np.vstack([slope_c - slope_c_ci[:, 0], slope_c_ci[:, 1] - slope_c])

    ax.errorbar(w, slope_u, yerr=u_yerr, fmt='o-', capsize=3, linewidth=1.6, label="Uncorrected slope-FF (95% CI)")
    ax.errorbar(w, slope_c, yerr=c_yerr, fmt='o-', capsize=3, linewidth=1.6, label="Corrected slope-FF (95% CI)")

    ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Slope Fano (Var vs Mean)")
    ax.set_title("Population Fano (Churchland slope)")

    # mark windows where CI on (corr - uncorr) is strictly < 0
    ytop = np.nanmax([np.nanmax(slope_u_ci[:,1]), np.nanmax(slope_c_ci[:,1])])
    for wi, (lo, hi) in zip(w, slope_diff_ci):
        if np.isfinite(lo) and np.isfinite(hi) and hi < 0:
            ax.plot([wi], [ytop], marker="v", markersize=6, clip_on=False)

    ax.legend(frameon=False, loc="best")

    # D) Slope ratio (optional) – easier to read than two lines
    ax = axs[1, 1]
    ax.plot(w, slope_ratio, marker="o", linewidth=1.8, label="Slope ratio (corr/uncorr)")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Ratio (corr/uncorr)")
    ax.set_title("Population effect size (slope ratio)")
    ax.legend(frameon=False, loc="best")

    # optional: annotate per-neuron significance (FDR) on panel A
    # (small stars above the corrected curve)
    # keep it subtle:
    ymin, ymax = axs[0, 0].get_ylim()
    ystar = ymax - 0.05 * (ymax - ymin)
    for wi, si in zip(w, sig):
        if si:
            axs[0, 0].text(wi, ystar, "∗", ha="center", va="center", fontsize=12)

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    return fig, axs


def plot_supp_var_mean_panels(metrics, savepath=None, max_panels=None):
    """
    Creates a row of per-window Var vs Mean scatter panels with slope fits (through origin).
    This is the 'raw' population Fano analysis figure.

    metrics[i] requires: 'window_ms', 'erate', 'uncorr', 'corr'
    """
    set_pub_style()

    # order windows
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    order = np.argsort(window_ms)
    mets = [metrics[i] for i in order]
    window_ms = window_ms[order]

    if max_panels is not None:
        mets = mets[:max_panels]
        window_ms = window_ms[:max_panels]

    n = len(mets)
    fig, axs = plt.subplots(1, n, figsize=(3.2*n, 2.8), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for ax, m in zip(axs, mets):
        x = np.asarray(m["erate"]).reshape(-1)
        ff_u = np.asarray(m["uncorr"]).reshape(-1)
        ff_c = np.asarray(m["corr"]).reshape(-1)

        # var = fano * mean (consistent with your code)
        y_u = ff_u * x
        y_c = ff_c * x

        # finite mask shared
        ok = np.isfinite(x) & np.isfinite(y_u) & np.isfinite(y_c) & (x > 0) & (y_u >= 0) & (y_c >= 0)
        x0, yu0, yc0 = x[ok], y_u[ok], y_c[ok]

        # scatter
        ax.scatter(x0, yu0, s=14, alpha=0.55, label=None)  # default color cycle
        ax.scatter(x0, yc0, s=14, alpha=0.55, label=None)

        # slopes through origin
        su = np.dot(x0, yu0) / np.dot(x0, x0)
        sc = np.dot(x0, yc0) / np.dot(x0, x0)

        # fitted lines
        xx = np.linspace(0, np.nanmax(x0)*1.02, 100)
        ax.plot(xx, su*xx, linestyle="--", linewidth=2.0)
        ax.plot(xx, sc*xx, linestyle="--", linewidth=2.0)

        ax.set_title(f"Window {int(m['window_ms'])}ms")
        ax.set_xlabel("Mean Rate (spk/s)")
        ax.set_ylabel("Variance")

        # clean legend with numeric slopes
        # we can't easily force colors without specifying; instead, label by order:
        # first scatter/line = uncorrected; second = corrected.
        ax.legend(
            handles=[
                plt.Line2D([0],[0], linestyle="--", linewidth=2.0,
                           label=f"Uncorrected FF = {su:.2f}"),
                plt.Line2D([0],[0], linestyle="--", linewidth=2.0,
                           label=f"Corrected FF = {sc:.2f}"),
            ],
            frameon=False, loc="upper left"
        )

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs

def plot_supp_fano_hist_panels(metrics, bins=np.linspace(0, 5, 80), savepath=None):
    set_pub_style()
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    order = np.argsort(window_ms)
    mets = [metrics[i] for i in order]

    n = len(mets)
    fig, axs = plt.subplots(1, n, figsize=(3.2*n, 2.8), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for ax, m in zip(axs, mets):
        ax.hist(m["uncorr"], bins=bins, alpha=0.55, label="Uncorrected")
        ax.hist(m["corr"],   bins=bins, alpha=0.55, label="Corrected")
        ax.set_title(f"Window {int(m['window_ms'])}ms")
        ax.set_xlabel("Per-neuron Fano factor")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs

def plot_supp_shuffle_ratio_hist_one_window(stats, metrics, window_index=0, savepath=None):
    set_pub_style()
    # keep consistent with compute_all_fano_stats pairing logic
    m = metrics[window_index]
    unc = np.asarray(m["uncorr"]).reshape(-1)
    cor = np.asarray(m["corr"]).reshape(-1)
    unc_v, cor_v, ok = paired_valid(unc, cor)

    real_ratio = geomean(cor_v) / geomean(unc_v)

    sh_u = align_shuffle(m["shuff_uncorr"], n_neurons=unc.size)
    sh_c = align_shuffle(m["shuff_corr"], n_neurons=cor.size)
    sh_u = sh_u[:, ok]
    sh_c = sh_c[:, ok]

    rnull = (geomean(sh_c, axis=1) / geomean(sh_u, axis=1))

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0), constrained_layout=True)
    ax.hist(rnull, bins=50, alpha=0.65)
    ax.axvline(real_ratio, linestyle="--", linewidth=2.0, label=f"Real ratio = {real_ratio:.3f}")
    ax.set_xlabel("Ratio (corr/uncorr) under shuffle")
    ax.set_ylabel("Count")
    ax.set_title(f"Shuffle null vs real (Window {int(m['window_ms'])}ms)")
    ax.legend(frameon=False)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, ax
