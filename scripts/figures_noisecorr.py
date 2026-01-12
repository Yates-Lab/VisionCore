# scripts/figures_noisecorr.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ----------------------------
# style
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
# stats helpers
# ----------------------------
def fisher_z_mean(rho, eps=1e-6):
    """Mean Fisher z across correlations (rho), robust to |rho|~1."""
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    rho = rho[np.isfinite(rho)]
    if rho.size == 0:
        return np.nan
    rho = np.clip(rho, -1 + eps, 1 - eps)
    return float(np.mean(np.arctanh(rho)))

def bootstrap_mean_ci(x, nboot=5000, ci=0.95, rng=0):
    """Bootstrap CI for mean of x (1D)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, (np.nan, np.nan)
    rg = np.random.default_rng(rng)
    boots = np.empty(nboot, dtype=float)
    for b in range(nboot):
        boots[b] = np.mean(rg.choice(x, size=x.size, replace=True))
    lo, hi = np.percentile(boots, [(1-ci)/2*100, (1+(ci))/2*100])
    return float(np.mean(x)), (float(lo), float(hi))

def iqr_25_75(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    q25, q75 = np.percentile(x, [25, 75])
    return float(q25), float(q75)

def _safe_mean(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


# ----------------------------
# computation
# ----------------------------
def compute_all_noisecorr_stats(
    metrics,
    eps_rho=1e-3,
    alternative="less",
    fdr_q=0.05,
    nboot=5000,
):
    """
    Compute publication-ready stats for noise correlations.

    Assumes extract_metrics produced (per window):
      - rho_uncorr, rho_corr              : raw upper-tri rho across pooled pairs
      - rho_u_meanz_by_ds, rho_c_meanz_by_ds : per-dataset mean Fisher z
      - rho_delta_meanz_by_ds             : per-dataset delta z = zC - zU
      - shuff_rho_delta_meanz             : shuffle null on delta (concat across datasets/shuffles)
      - shuff_rho_c_meanz                 : optional, shuffled corrected mean-z

    Returns dict 'stats' similar in spirit to Fano stats.
    """
    window_ms = np.array([m["window_ms"] for m in metrics], dtype=float)
    nW = len(metrics)

    # central tendency on raw pairwise rho (for intuition/2D panels)
    rho_u_mean = np.full(nW, np.nan)
    rho_c_mean = np.full(nW, np.nan)
    rho_u_iqr = np.full((nW, 2), np.nan)
    rho_c_iqr = np.full((nW, 2), np.nan)

    # Fisher z summaries (preferred for inference)
    z_u_mean = np.full(nW, np.nan)
    z_c_mean = np.full(nW, np.nan)
    z_u_ci = np.full((nW, 2), np.nan)
    z_c_ci = np.full((nW, 2), np.nan)

    # delta z (per-dataset)
    dz_mean = np.full(nW, np.nan)
    dz_ci = np.full((nW, 2), np.nan)
    n_ds = np.zeros(nW, dtype=int)

    # hypothesis tests
    p_wil = np.full(nW, np.nan)          # per-dataset paired test on delta z
    p_wil_fdr = np.full(nW, np.nan)
    sig_fdr = np.zeros(nW, dtype=bool)

    # shuffle null on delta z
    null_dz_ci95 = np.full((nW, 2), np.nan)
    p_emp_dz = np.full(nW, np.nan)       # empirical p that null <= observed (for reduction)

    for i, m in enumerate(metrics):
        # pooled-pairs rho summaries (careful: not independent, but ok for descriptive stats)
        ru = np.asarray(m.get("rho_uncorr", []), dtype=float)
        rc = np.asarray(m.get("rho_corr", []), dtype=float)
        rho_u_mean[i] = _safe_mean(ru)
        rho_c_mean[i] = _safe_mean(rc)
        rho_u_iqr[i] = iqr_25_75(ru)
        rho_c_iqr[i] = iqr_25_75(rc)

        # per-dataset mean-z summaries (recommended)
        zu = np.asarray(m.get("rho_u_meanz_by_ds", []), dtype=float)
        zc = np.asarray(m.get("rho_c_meanz_by_ds", []), dtype=float)
        dz = np.asarray(m.get("rho_delta_meanz_by_ds", []), dtype=float)

        zu = zu[np.isfinite(zu)]
        zc = zc[np.isfinite(zc)]
        dz = dz[np.isfinite(dz)]
        n_ds[i] = int(dz.size)

        z_u_mean[i] = float(np.mean(zu)) if zu.size else np.nan
        z_c_mean[i] = float(np.mean(zc)) if zc.size else np.nan
        dz_mean[i]  = float(np.mean(dz)) if dz.size else np.nan

        # bootstrap CIs across datasets (not across pairs)
        _, (lo, hi) = bootstrap_mean_ci(zu, nboot=nboot, rng=10_000 + i) if zu.size else (np.nan, (np.nan, np.nan))
        z_u_ci[i] = [lo, hi]
        _, (lo, hi) = bootstrap_mean_ci(zc, nboot=nboot, rng=20_000 + i) if zc.size else (np.nan, (np.nan, np.nan))
        z_c_ci[i] = [lo, hi]
        _, (lo, hi) = bootstrap_mean_ci(dz, nboot=nboot, rng=30_000 + i) if dz.size else (np.nan, (np.nan, np.nan))
        dz_ci[i] = [lo, hi]

        # Wilcoxon signed-rank test on per-dataset deltas (paired by dataset)
        if dz.size >= 5:
            # we test dz < 0 (correction reduces correlations)
            p_wil[i] = wilcoxon(dz, alternative=alternative, zero_method="wilcox").pvalue

        # shuffle null on delta z (concat across ds & shuffles)
        null = np.asarray(m.get("shuff_rho_delta_meanz", []), dtype=float)
        null = null[np.isfinite(null)]
        if null.size:
            null_dz_ci95[i] = np.percentile(null, [2.5, 97.5])
            # empirical one-sided p: how often null is <= observed mean delta (more negative)
            # (smaller dz = stronger reduction)
            obs = dz_mean[i]
            if np.isfinite(obs):
                p_emp_dz[i] = (np.sum(null <= obs) + 1) / (null.size + 1)

    # multiple comparisons across windows
    okp = np.isfinite(p_wil)
    if np.any(okp):
        reject, p_adj, *_ = multipletests(p_wil[okp], alpha=fdr_q, method="fdr_bh")
        p_wil_fdr[okp] = p_adj
        sig_fdr[okp] = reject

    return dict(
        window_ms=window_ms,

        # descriptive (raw rho)
        rho_u_mean=rho_u_mean,
        rho_c_mean=rho_c_mean,
        rho_u_iqr=rho_u_iqr,
        rho_c_iqr=rho_c_iqr,

        # inferential (per-dataset z)
        z_u_mean=z_u_mean,
        z_c_mean=z_c_mean,
        z_u_ci=z_u_ci,
        z_c_ci=z_c_ci,
        dz_mean=dz_mean,
        dz_ci=dz_ci,
        n_ds=n_ds,

        # tests
        p_wil=p_wil,
        p_wil_fdr=p_wil_fdr,
        sig_fdr=sig_fdr,

        # shuffle null on delta
        null_dz_ci95=null_dz_ci95,
        p_emp_dz=p_emp_dz,
    )


# ----------------------------
# plotting
# ----------------------------
def plot_noisecorr_paper_figure(stats, title=None):
    """
    Main figure: summary in Fisher-z space with bootstrap CI across datasets,
    plus effect size (delta z) and shuffle-null CI band.
    """
    set_pub_style()

    w = np.asarray(stats["window_ms"], dtype=float)
    order = np.argsort(w)
    w = w[order]

    def r(x): return np.asarray(x)[order]

    z_u_mean = r(stats["z_u_mean"])
    z_c_mean = r(stats["z_c_mean"])
    z_u_ci = r(stats["z_u_ci"])
    z_c_ci = r(stats["z_c_ci"])

    dz_mean = r(stats["dz_mean"])
    dz_ci = r(stats["dz_ci"])
    null_ci = r(stats["null_dz_ci95"])
    sig = r(stats["sig_fdr"])

    # convert CI to yerr for matplotlib
    u_yerr = np.vstack([z_u_mean - z_u_ci[:, 0], z_u_ci[:, 1] - z_u_mean])
    c_yerr = np.vstack([z_c_mean - z_c_ci[:, 0], z_c_ci[:, 1] - z_c_mean])

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)

    # A) mean z vs window (with bootstrap CI across datasets)
    ax = axs[0]
    ax.errorbar(w + 0.5, z_u_mean, yerr=u_yerr, fmt="o-", capsize=3, label="Uncorrected (mean z)")
    ax.errorbar(w,       z_c_mean, yerr=c_yerr, fmt="o-", capsize=3, label="Corrected (mean z)")
    ax.axhline(0, color="k", lw=1, alpha=0.35)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Mean Fisher z(noise corr)")
    ax.set_title("Noise correlations")
    ax.legend(frameon=False)

    # mark FDR-significant windows (tiny star near corrected point)
    ymin, ymax = ax.get_ylim()
    ystar = ymax - 0.06 * (ymax - ymin)
    for wi, si in zip(w, sig):
        if si:
            ax.text(wi, ystar, "∗", ha="center", va="center", fontsize=12)

    # B) delta z vs window with shuffle-null band
    ax = axs[1]
    # CI on dz (bootstrap across datasets)
    dz_yerr = np.vstack([dz_mean - dz_ci[:, 0], dz_ci[:, 1] - dz_mean])
    ax.errorbar(w, dz_mean, yerr=dz_yerr, fmt="o-", capsize=3, label="Real Δz = zC - zU")
    ax.fill_between(w, null_ci[:, 0], null_ci[:, 1], alpha=0.18, label="Shuffle null 95% CI (Δz)")
    ax.axhline(0, color="k", lw=1, alpha=0.35)
    ax.set_xlabel("Window (ms)")
    ax.set_ylabel("Δ mean Fisher z")
    ax.set_title("Effect size vs shuffle control")
    ax.legend(frameon=False)

    if title:
        fig.suptitle(title, y=1.03, fontsize=12)

    return fig, axs


# ----------------------------
# supplemental plots
# ----------------------------
def plot_supp_noisecorr_hist_panels(metrics, bins=np.linspace(-0.5, 0.5, 101), savepath=None):
    """
    Supplemental: raw rho histograms by window (uncorr vs corr).
    """
    set_pub_style()
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharex=True, sharey=True, constrained_layout=True)

    for i, m in enumerate(metrics):
        ax = axs[i]
        ax.hist(m["rho_uncorr"], bins=bins, density=True, alpha=0.45, label="Uncorrected")
        ax.hist(m["rho_corr"],   bins=bins, density=True, alpha=0.45, label="Corrected")
        ax.axvline(0, color="k", lw=1, alpha=0.35)
        ax.set_title(f"Window {int(m['window_ms'])}ms")
        ax.set_xlabel("Noise correlation ρ")
        if i == 0:
            ax.set_ylabel("Density")
        if i == n - 1:
            ax.legend(frameon=False, loc="upper left")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs

def plot_supp_noisecorr_shuffle_corrected_panels(metrics, savepath=None):
    """
    Supplemental: distribution of shuffle null rho corrected per window,
    with observed rho corrected marked.
    """

    set_pub_style()
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharey=True, constrained_layout=True)

    for i, m in enumerate(metrics):
        ax = axs[i]
        null = np.asarray(m.get("shuff_rho_c_meanz", []), dtype=float)
        null = null[np.isfinite(null)]
        if null.size:
            ax.hist(null, bins=60, density=True, alpha=0.55, color="gray", label="Shuffle null ρ (corr)")
            ci = np.percentile(null, [2.5, 97.5])
            ax.axvline(ci[0], color="k", lw=1, alpha=0.4)
            ax.axvline(ci[1], color="k", lw=1, alpha=0.4)

        obs = np.nanmean(np.asarray(m.get("rho_c_meanz", []), dtype=float))
        ax.axvline(obs, color="crimson", linestyle="--", lw=1.5, label="Observed ρ (corr)")

        # empirical p: null <= obs (more negative than obs)
        p_emp = (np.sum(null <= obs) + 1) / (null.size + 1) if null.size and np.isfinite(obs) else np.nan
        ax.set_title(f"{int(m['window_ms'])}ms\np={p_emp:.3g}" if np.isfinite(p_emp) else f"{int(m['window_ms'])}ms")
        ax.set_xlabel("Noise correlation (ρ corrected)")
        if i == 0:
            ax.set_ylabel("Density")
        if i == n - 1:
            ax.legend(frameon=False, loc="upper left")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs



def plot_supp_noisecorr_shuffle_delta_panels(metrics, savepath=None):
    """
    Supplemental: distribution of shuffle null Δz per window,
    with observed Δz marked.
    """
    set_pub_style()
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharey=True, constrained_layout=True)

    for i, m in enumerate(metrics):
        ax = axs[i]
        null = np.asarray(m.get("shuff_rho_delta_meanz", []), dtype=float)
        null = null[np.isfinite(null)]
        if null.size:
            ax.hist(null, bins=60, density=True, alpha=0.55, color="gray", label="Shuffle null Δz")
            ci = np.percentile(null, [2.5, 97.5])
            ax.axvline(ci[0], color="k", lw=1, alpha=0.4)
            ax.axvline(ci[1], color="k", lw=1, alpha=0.4)

        obs = np.nanmean(np.asarray(m.get("rho_delta_meanz_by_ds", []), dtype=float))
        ax.axvline(obs, color="crimson", linestyle="--", lw=1.5, label="Observed Δz")

        # empirical p: null <= obs (more negative than obs)
        p_emp = (np.sum(null <= obs) + 1) / (null.size + 1) if null.size and np.isfinite(obs) else np.nan
        ax.set_title(f"{int(m['window_ms'])}ms\np={p_emp:.3g}" if np.isfinite(p_emp) else f"{int(m['window_ms'])}ms")
        ax.set_xlabel("Δ mean Fisher z")
        if i == 0:
            ax.set_ylabel("Density")
        if i == n - 1:
            ax.legend(frameon=False, loc="upper left")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs


def plot_supp_noisecorr_2d_panels(metrics, bins=np.linspace(-0.5, 0.5, 101), cmap="Blues", savepath=None):
    """
    Supplemental: 2D histogram (uncorr vs corr) per window, log1p density.
    """
    set_pub_style()
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharex=True, sharey=True, constrained_layout=True)

    for i, m in enumerate(metrics):
        ax = axs[i]
        x = np.asarray(m["rho_uncorr"], dtype=float)
        y = np.asarray(m["rho_corr"], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]

        cnt, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins])
        cnt = np.log1p(cnt)

        ax.imshow(
            cnt.T, origin="lower", interpolation="none", aspect="auto",
            cmap=cmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], "k--", alpha=0.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.4)
        ax.axvline(0, color="k", linestyle="--", alpha=0.4)

        # red dot at mean in z space (then back-transform)
        zu = fisher_z_mean(x, eps=1e-6)
        zc = fisher_z_mean(y, eps=1e-6)
        ax.plot([np.tanh(zu)], [np.tanh(zc)], "o", color="crimson", markersize=5)

        ax.set_title(f"Window {int(m['window_ms'])}ms")
        ax.set_xlabel("ρ (uncorrected)")
        ax.set_ylabel("ρ (corrected)")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig, axs


# ----------------------------
# printing
# ----------------------------
def print_noisecorr_stats(stats):
    w = np.asarray(stats["window_ms"], dtype=float)
    order = np.argsort(w)
    w = w[order]

    def r(x): return np.asarray(x)[order]

    # descriptive in rho-space
    ru_mean = r(stats["rho_u_mean"])
    rc_mean = r(stats["rho_c_mean"])
    ru_iqr = r(stats["rho_u_iqr"])
    rc_iqr = r(stats["rho_c_iqr"])

    # inferential in z-space
    zu = r(stats["z_u_mean"]); zc = r(stats["z_c_mean"])
    zu_ci = r(stats["z_u_ci"]); zc_ci = r(stats["z_c_ci"])
    dz = r(stats["dz_mean"]); dz_ci = r(stats["dz_ci"])
    null_ci = r(stats["null_dz_ci95"])
    n_ds = r(stats["n_ds"])
    p = r(stats["p_wil"]); p_fdr = r(stats["p_wil_fdr"])
    p_emp = r(stats["p_emp_dz"])

    for i in range(len(w)):
        print(f"\nWindow {int(w[i])} ms (N datasets = {int(n_ds[i])})")
        print(f"  Raw ρ (pooled pairs):")
        print(f"    Uncorr mean={ru_mean[i]:.4f}, IQR=[{ru_iqr[i,0]:.4f}, {ru_iqr[i,1]:.4f}]")
        print(f"    Corr   mean={rc_mean[i]:.4f}, IQR=[{rc_iqr[i,0]:.4f}, {rc_iqr[i,1]:.4f}]")
        print(f"  Mean Fisher z across datasets (bootstrap 95% CI):")
        print(f"    Uncorr z={zu[i]:.4f}  CI=[{zu_ci[i,0]:.4f}, {zu_ci[i,1]:.4f}]")
        print(f"    Corr   z={zc[i]:.4f}  CI=[{zc_ci[i,0]:.4f}, {zc_ci[i,1]:.4f}]")
        print(f"  Effect Δz = zC - zU = {dz[i]:.4f}  CI=[{dz_ci[i,0]:.4f}, {dz_ci[i,1]:.4f}]")
        if np.isfinite(null_ci[i,0]):
            print(f"  Shuffle null Δz 95% CI = [{null_ci[i,0]:.4f}, {null_ci[i,1]:.4f}]")
            print(f"  Empirical p(null <= observed Δz) = {p_emp[i]:.3g}")
        if np.isfinite(p[i]):
            print(f"  Wilcoxon test on per-dataset Δz: p={p[i]:.3g}, FDR p={p_fdr[i]:.3g}")
