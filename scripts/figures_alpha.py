# scripts/figures_alpha.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


# ----------------------------
# style (match your other modules)
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
def _finite(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]

def iqr_25_75(x):
    x = _finite(x)
    if x.size == 0:
        return (np.nan, np.nan)
    q25, q75 = np.percentile(x, [25, 75])
    return float(q25), float(q75)

def median_iqr(x):
    x = _finite(x)
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    return med, (float(q25), float(q75))

def bootstrap_mean_ci(x, nboot=5000, ci=0.95, rng=0):
    x = _finite(x)
    if x.size < 2:
        return np.nan, (np.nan, np.nan)
    rg = np.random.default_rng(rng)
    boots = np.empty(nboot, dtype=float)
    for b in range(nboot):
        boots[b] = np.mean(rg.choice(x, size=x.size, replace=True))
    lo, hi = np.percentile(boots, [(1-ci)/2*100, (1+(ci))/2*100])
    return float(np.mean(x)), (float(lo), float(hi))

def align_shuffle(shuff, n_units):
    """
    Ensure shuffle array is [S, N] (shuffles x units) or [N, S] depending on input.
    Returns [S, N].
    """
    shuff = np.asarray(shuff, dtype=float)
    if shuff.ndim != 2:
        raise ValueError(f"shuff_alphas must be 2D, got {shuff.shape}")
    if shuff.shape[0] == n_units:
        return shuff.T
    if shuff.shape[1] == n_units:
        return shuff
    # heuristic fallback: pick orientation whose 2nd dim is closer to n_units
    return shuff.T if abs(shuff.shape[0] - n_units) < abs(shuff.shape[1] - n_units) else shuff

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
# main computation
# ----------------------------
def compute_all_alpha_stats(
    metrics,
    fdr_q=0.05,
    nboot=5000,
    corr_method="spearman",
):
    """
    Computes stats for FEM modulation fraction m = 1 - alpha.

    metrics[i] must contain:
      - 'alpha' (N,)
      - 'shuff_alphas' (N, S) or (S, N)
      - 'uncorr' (N,) fano
      - 'corr'   (N,) fano

    Returns dict with per-window summaries + null comparisons + relationships to FF.
    """
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    nW = len(metrics)

    # per-window summaries (real)
    m_mean = np.full(nW, np.nan)
    m_mean_ci = np.full((nW, 2), np.nan)
    m_med = np.full(nW, np.nan)
    m_iqr = np.full((nW, 2), np.nan)
    n_units = np.zeros(nW, dtype=int)

    # shuffle null on mean(m) and median(m)
    null_mean_ci95 = np.full((nW, 2), np.nan)
    null_med_ci95 = np.full((nW, 2), np.nan)
    p_emp_mean = np.full(nW, np.nan)   # P(null >= real) since we expect real m to be larger than shuffle
    p_emp_med = np.full(nW, np.nan)

    # relationships to FF
    # 1) association between m and FF_uncorr
    rho_m_ffu = np.full(nW, np.nan); p_m_ffu = np.full(nW, np.nan)
    # 2) association between m and FF_corr
    rho_m_ffc = np.full(nW, np.nan); p_m_ffc = np.full(nW, np.nan)
    # 3) association between m and ΔFF = uncorr - corr (effect size per neuron)
    rho_m_dff = np.full(nW, np.nan); p_m_dff = np.full(nW, np.nan)

    for i, mw in enumerate(metrics):
        alpha = np.asarray(mw.get("alpha", []), dtype=float).reshape(-1)
        ffu = np.asarray(mw.get("uncorr", []), dtype=float).reshape(-1)
        ffc = np.asarray(mw.get("corr", []), dtype=float).reshape(-1)

        # define modulation fraction m = 1 - alpha
        m = 1.0 - alpha

        # basic validity
        ok = np.isfinite(m) & np.isfinite(ffu) & np.isfinite(ffc)
        m0 = m[ok]; ffu0 = ffu[ok]; ffc0 = ffc[ok]
        n_units[i] = int(m0.size)
        if m0.size < 5:
            continue

        # real summaries
        m_mean[i], (lo, hi) = bootstrap_mean_ci(m0, nboot=nboot, rng=1000 + i)
        m_mean_ci[i] = [lo, hi]
        m_med[i], (q25, q75) = median_iqr(m0)
        m_iqr[i] = [q25, q75]

        # shuffle null
        sh = np.asarray(mw.get("shuff_alphas", []), dtype=float)
        if sh.size:
            sh = align_shuffle(sh, n_units=alpha.size)  # [S, N_all]
            # apply same ok mask (units) used in real association (after ok computed in full-length)
            # but ok was on trimmed arrays; rebuild full ok mask:
            ok_full = np.isfinite(m) & np.isfinite(ffu) & np.isfinite(ffc)
            sh = sh[:, ok_full]  # [S, N_valid]

            # convert to m for shuffle: m_s = 1 - alpha_s
            m_sh = 1.0 - sh

            # null distributions of mean(m) and median(m)
            null_mean = np.nanmean(m_sh, axis=1)
            null_med = np.nanmedian(m_sh, axis=1)

            null_mean_ci95[i] = np.percentile(_finite(null_mean), [2.5, 97.5])
            null_med_ci95[i] = np.percentile(_finite(null_med), [2.5, 97.5])

            # empirical p that null >= observed (because we expect real modulation > shuffle)
            p_emp_mean[i] = emp_p_one_sided(null_mean, m_mean[i], direction="greater")
            p_emp_med[i] = emp_p_one_sided(null_med, m_med[i], direction="greater")

        # relationships (Spearman by default; robust to nonlinearity)
        dff = ffu0 - ffc0

        if corr_method == "spearman":
            rho, p = spearmanr(m0, ffu0, nan_policy="omit")
            rho_m_ffu[i], p_m_ffu[i] = rho, p
            rho, p = spearmanr(m0, ffc0, nan_policy="omit")
            rho_m_ffc[i], p_m_ffc[i] = rho, p
            rho, p = spearmanr(m0, dff,  nan_policy="omit")
            rho_m_dff[i], p_m_dff[i] = rho, p
        else:
            raise ValueError("Only corr_method='spearman' supported in this helper.")

    # multiple comparisons across windows for correlation tests (three families)
    def fdr(pvals):
        pvals = np.asarray(pvals, dtype=float)
        out = np.full_like(pvals, np.nan)
        sig = np.zeros_like(pvals, dtype=bool)
        ok = np.isfinite(pvals)
        if np.any(ok):
            reject, padj, *_ = multipletests(pvals[ok], alpha=fdr_q, method="fdr_bh")
            out[ok] = padj
            sig[ok] = reject
        return out, sig

    p_m_ffu_fdr, sig_m_ffu = fdr(p_m_ffu)
    p_m_ffc_fdr, sig_m_ffc = fdr(p_m_ffc)
    p_m_dff_fdr, sig_m_dff = fdr(p_m_dff)

    return dict(
        window_ms=window_ms,
        n_units=n_units,

        m_mean=m_mean, m_mean_ci=m_mean_ci,
        m_med=m_med, m_iqr=m_iqr,

        null_mean_ci95=null_mean_ci95,
        null_med_ci95=null_med_ci95,
        p_emp_mean=p_emp_mean,
        p_emp_med=p_emp_med,

        rho_m_ffu=rho_m_ffu, p_m_ffu=p_m_ffu, p_m_ffu_fdr=p_m_ffu_fdr, sig_m_ffu=sig_m_ffu,
        rho_m_ffc=rho_m_ffc, p_m_ffc=p_m_ffc, p_m_ffc_fdr=p_m_ffc_fdr, sig_m_ffc=sig_m_ffc,
        rho_m_dff=rho_m_dff, p_m_dff=p_m_dff, p_m_dff_fdr=p_m_dff_fdr, sig_m_dff=sig_m_dff,
    )


# ----------------------------
# plotting
# ----------------------------
def plot_alpha_paper_figure(alpha_stats, title=None):
    """
    Main figure: stability of modulation fraction m=1-alpha across windows,
    plus comparison to shuffle null (band).
    """
    set_pub_style()
    w = np.asarray(alpha_stats["window_ms"], dtype=float)
    order = np.argsort(w)
    w = w[order]

    def r(x): return np.asarray(x)[order]

    m_mean = r(alpha_stats["m_mean"])
    m_ci = r(alpha_stats["m_mean_ci"])
    m_med = r(alpha_stats["m_med"])
    m_iqr = r(alpha_stats["m_iqr"])
    null_mean_ci = r(alpha_stats["null_mean_ci95"])
    p_emp = r(alpha_stats["p_emp_mean"])

    # convert mean CI to yerr
    yerr = np.vstack([m_mean - m_ci[:, 0], m_ci[:, 1] - m_mean])

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

    # A) mean(m) with bootstrap CI + shuffle band
    ax = axs[0]
    ax.errorbar(w, m_mean, yerr=yerr, fmt="o-", capsize=3, label="Real mean(1-α)")
    ax.fill_between(w, null_mean_ci[:, 0], null_mean_ci[:, 1], alpha=0.18, color="gray",
                    label="Shuffle null 95% CI (mean)")
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("FEM modulation fraction (1 − α)")
    ax.set_title("Rate modulation explained by FEMs")
    ax.set_xlim(np.min(w) - 2, np.max(w) + 2)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best")

    # annotate empirical p-values subtly
    ymin, ymax = ax.get_ylim()
    ytxt = ymin + 0.06 * (ymax - ymin)
    for wi, pi in zip(w, p_emp):
        if np.isfinite(pi):
            ax.text(wi, ytxt, f"{pi:.2g}", ha="center", va="bottom", fontsize=8, alpha=0.85)

    # B) median + IQR (more robust than mean)
    ax = axs[1]
    ax.plot(w, m_med, "o-", label="Real median(1-α)")
    ax.fill_between(w, m_iqr[:, 0], m_iqr[:, 1], alpha=0.18, label="Real IQR")
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("FEM modulation fraction (1 − α)")
    ax.set_title("Population distribution")
    ax.set_xlim(np.min(w) - 2, np.max(w) + 2)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best")

    if title:
        fig.suptitle(title, y=1.03, fontsize=12)

    return fig, axs


def plot_supp_alpha_hist_panels(metrics, bins=np.linspace(0, 1, 51), savepath=None):
    """
    Supplemental: histograms of m=1-alpha by window, with mean/median markers.
    """
    set_pub_style()
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    n = len(window_ms)

    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharex=True, sharey=True, constrained_layout=True)

    for i, mw in enumerate(metrics):
        ax = axs[i]
        m = 1.0 - np.asarray(mw.get("alpha", []), dtype=float)
        m = m[np.isfinite(m)]
        ax.hist(m, bins=bins, alpha=0.75)
        if m.size:
            ax.axvline(np.mean(m), color="crimson", linestyle="--", alpha=0.7, label="mean")
            ax.axvline(np.median(m), color="k", linestyle="--", alpha=0.6, label="median")
        ax.set_title(f"Window {int(mw['window_ms'])}ms")
        ax.set_xlabel("1 − α")
        if i == 0:
            ax.set_ylabel("Count")
        if i == n - 1:
            ax.legend(frameon=False)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs


def plot_supp_alpha_vs_metrics(metrics, which="uncorr", savepath=None, alpha=0.10):
    """
    Supplemental: scatter of FF vs m=1-alpha for each window.
      which in {'uncorr','corr','delta'} where delta = uncorr - corr
    """
    set_pub_style()
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    n = len(window_ms)

    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharex=True, constrained_layout=True)

    for i, mw in enumerate(metrics):
        ax = axs[i]
        m = 1.0 - np.asarray(mw.get("alpha", []), dtype=float)
        ffu = np.asarray(mw.get("uncorr", []), dtype=float)
        ffc = np.asarray(mw.get("corr", []), dtype=float)

        if which == "uncorr":
            y = ffu
            ylabel = "Fano factor (uncorrected)"
        elif which == "corr":
            y = ffc
            ylabel = "Fano factor (corrected)"
        elif which == "delta":
            y = ffu - ffc
            ylabel = "ΔFano (uncorr − corr)"
        else:
            raise ValueError("which must be 'uncorr', 'corr', or 'delta'.")

        ok = np.isfinite(m) & np.isfinite(y)
        m0 = m[ok]; y0 = y[ok]

        ax.plot(m0, y0, "o", alpha=alpha, markersize=3)
        ax.set_title(f"{int(mw['window_ms'])}ms")
        ax.set_xlabel("1 − α")
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)

        # Spearman annotation
        if m0.size >= 10:
            rho, p = spearmanr(m0, y0, nan_policy="omit")
            ax.text(0.02, 0.98, f"ρ={rho:.2f}\np={p:.2g}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs


def plot_supp_alpha_shuffle_comparison(metrics, alpha_stats, savepath=None):
    """
    Supplemental: per-window null distributions of mean(m) from shuffles,
    with real mean(m) overlaid.
    """
    set_pub_style()
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharey=True, constrained_layout=True)

    for i, mw in enumerate(metrics):
        ax = axs[i]
        alpha = np.asarray(mw.get("alpha", []), dtype=float).reshape(-1)
        m_real = 1.0 - alpha
        m_real = m_real[np.isfinite(m_real)]

        sh = np.asarray(mw.get("shuff_alphas", []), dtype=float)
        if sh.size == 0:
            ax.set_title(f"{int(mw['window_ms'])}ms\n(no shuffles)")
            continue

        sh = align_shuffle(sh, n_units=alpha.size)  # [S, N]
        ok = np.isfinite(m_real)
        # ok above is already filtered; build full mask:
        ok_full = np.isfinite(1.0 - alpha)
        sh = sh[:, ok_full]
        m_sh = 1.0 - sh
        null = np.nanmean(m_sh, axis=1)
        null = null[np.isfinite(null)]

        ax.hist(null, bins=60, density=True, color="gray", alpha=0.55, label="Shuffle null mean(1-α)")
        ax.axvline(alpha_stats["m_mean"][i], color="crimson", linestyle="--", lw=1.8, label="Real mean(1-α)")

        p = alpha_stats["p_emp_mean"][i]
        ax.set_title(f"{int(mw['window_ms'])}ms\np={p:.3g}" if np.isfinite(p) else f"{int(mw['window_ms'])}ms")
        ax.set_xlabel("Mean(1 − α)")
        if i == 0:
            ax.set_ylabel("Density")
        if i == n - 1:
            ax.legend(frameon=False, loc="upper left")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs


# ----------------------------
# printing
# ----------------------------
def print_alpha_stats(alpha_stats):
    w = np.asarray(alpha_stats["window_ms"], dtype=float)
    order = np.argsort(w)
    w = w[order]

    def r(x): return np.asarray(x)[order]

    n_units = r(alpha_stats["n_units"])
    m_mean = r(alpha_stats["m_mean"])
    m_ci = r(alpha_stats["m_mean_ci"])
    m_med = r(alpha_stats["m_med"])
    m_iqr = r(alpha_stats["m_iqr"])
    null_ci = r(alpha_stats["null_mean_ci95"])
    p_emp = r(alpha_stats["p_emp_mean"])

    rho_u = r(alpha_stats["rho_m_ffu"])
    p_u = r(alpha_stats["p_m_ffu"])
    p_u_fdr = r(alpha_stats["p_m_ffu_fdr"])

    rho_c = r(alpha_stats["rho_m_ffc"])
    p_c = r(alpha_stats["p_m_ffc"])
    p_c_fdr = r(alpha_stats["p_m_ffc_fdr"])

    rho_d = r(alpha_stats["rho_m_dff"])
    p_d = r(alpha_stats["p_m_dff"])
    p_d_fdr = r(alpha_stats["p_m_dff_fdr"])

    for i in range(len(w)):
        print(f"\nWindow {int(w[i])} ms (N valid units = {int(n_units[i])})")
        print(f"  m = 1-α : mean={m_mean[i]:.3f}  bootstrap95%CI=[{m_ci[i,0]:.3f}, {m_ci[i,1]:.3f}]")
        print(f"          median={m_med[i]:.3f}  IQR=[{m_iqr[i,0]:.3f}, {m_iqr[i,1]:.3f}]")
        if np.isfinite(null_ci[i,0]):
            print(f"  Shuffle null mean(m) 95% CI=[{null_ci[i,0]:.3f}, {null_ci[i,1]:.3f}]")
            print(f"  Empirical p(null mean(m) >= real mean(m)) = {p_emp[i]:.3g}")

        print(f"  Spearman(m, FF_uncorr):  ρ={rho_u[i]:.2f}, p={p_u[i]:.3g}, FDR p={p_u_fdr[i]:.3g}")
        print(f"  Spearman(m, FF_corr):    ρ={rho_c[i]:.2f}, p={p_c[i]:.3g}, FDR p={p_c_fdr[i]:.3g}")
        print(f"  Spearman(m, ΔFF=U−C):    ρ={rho_d[i]:.2f}, p={p_d[i]:.3g}, FDR p={p_d_fdr[i]:.3g}")
