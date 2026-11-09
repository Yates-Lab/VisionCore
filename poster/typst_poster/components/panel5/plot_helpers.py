"""Frozen plotting helpers from paper/fig2 for the approved poster.

Copied from _panel_common.py, generate_panel_fano.py and
 generate_panel_noisecorr.py. Drawing operations are preserved; implicit
analysis/data-loading fallbacks are disabled. The caller supplies plot_data.npz.
This module deliberately has no dependency on the analysis pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt

def pstars(p):
    """Significance stars from a p-value (n.s. if not significant or NaN)."""
    import numpy as np
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def nearest_window(windows_ms, target_ms):
    """Return the WINDOWS_MS entry closest to ``target_ms``.

    Returns the original element (not a rounded value) so it can be used as an
    exact key into the per-window stats dicts (fano_stats, nc_stats, ...).
    """
    import numpy as np
    arr = np.asarray(windows_ms, dtype=float)
    return windows_ms[int(np.argmin(np.abs(arr - float(target_ms))))]


def fmt_emp_p(p, n_shuff=None):
    """Format an empirical (shuffle-null) p-value for display, respecting the
    +1-smoothing floor of 1/(n_shuff+1): a p sitting at that floor is rendered
    as ``p < <floor>`` rather than an exact value the resampling can't resolve.
    """
    import numpy as np
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if n_shuff:
        floor = 1.0 / (n_shuff + 1)
        if p <= floor * (1 + 1e-9):
            return f"p < {floor:.2g}"
    if p < 1e-3:
        return "p < 0.001"
    return f"p = {p:.3g}"


def pair_violin(ax, dist_u, dist_c, *, mean_u, mean_c, err_u, err_c,
                null_lo=None, null_hi=None, p=None, n_shuff=None,
                ref=0.0, ref_label=None, ylabel=None, color="tab:blue",
                labels=("uncorrected", "FEM-\ncorrected")):
    """Shared E/F panel grammar: grey violins of the raw distribution (``dist_u``
    uncorrected, ``dist_c`` corrected) with the across-unit mean +/- error
    overlaid as open(uncorrected)->filled(corrected) markers joined by a line,
    an optional grey shuffle-null band at the corrected marker, a dotted
    reference line, and a significance bracket carrying stars + the explicit
    shuffle-null p-value above them.

    ``err_u``/``err_c`` are (low, high) absolute y-coordinates of the whisker
    endpoints (so callers can pass asymmetric whiskers, e.g. an SD computed in
    Fisher-z then back-transformed to rho). ``null_lo``/``null_hi`` are absolute
    y-coordinates of the null band drawn at the corrected position.
    """
    import numpy as np
    x = [0.0, 1.0]
    dist_u = np.asarray(dist_u, dtype=float); dist_u = dist_u[np.isfinite(dist_u)]
    dist_c = np.asarray(dist_c, dtype=float); dist_c = dist_c[np.isfinite(dist_c)]

    viol = ax.violinplot([dist_u, dist_c], positions=x, widths=0.7,
                         showmeans=False, showmedians=False, showextrema=False)
    for body in viol["bodies"]:
        body.set_facecolor("0.8")
        body.set_edgecolor("none")
        body.set_alpha(0.85)
        body.set_zorder(1)

    ax.axhline(ref, color="gray", linestyle=":", alpha=0.8, zorder=0)
    if ref_label is not None:
        ax.text(1.52, ref, ref_label, color="gray", fontsize=7,
                va="center", ha="left")

    # Grey shuffle-null band at the corrected marker: where the corrected
    # estimate would land if the FEM correction were random.
    if null_lo is not None and null_hi is not None and np.isfinite(null_lo) \
            and np.isfinite(null_hi):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (x[1] - 0.2, null_lo), 0.4, null_hi - null_lo,
            facecolor="0.6", alpha=0.30, edgecolor="0.35", lw=0.7,
            linestyle="--", zorder=2))
        ax.text(x[1] + 0.22, 0.5 * (null_lo + null_hi), "shuffle\nnull",
                color="0.4", fontsize=6.5, va="center", ha="left")

    # Connector + mean markers with (possibly asymmetric) whiskers.
    ax.plot(x, [mean_u, mean_c], "-", color=color, lw=2.0, zorder=4)
    yerr = np.array([[mean_u - err_u[0], mean_c - err_c[0]],
                     [err_u[1] - mean_u, err_c[1] - mean_c]])
    ax.errorbar(x, [mean_u, mean_c], yerr=yerr, fmt="none", ecolor=color,
                elinewidth=1.5, capsize=0, zorder=5)
    ax.plot(x[0], mean_u, "o", mfc="white", mec=color, ms=7, mew=1.6, zorder=6)
    ax.plot(x[1], mean_c, "o", mfc=color, mec=color, ms=7, mew=1.6, zorder=6)

    lows = [float(np.percentile(dist_u, 1)), float(np.percentile(dist_c, 1)),
            err_u[0], err_c[0], ref]
    highs = [float(np.percentile(dist_u, 99)), float(np.percentile(dist_c, 99)),
             err_u[1], err_c[1]]
    if null_lo is not None and np.isfinite(null_lo):
        lows.append(null_lo); highs.append(null_hi)
    bot, top = min(lows), max(highs)
    span = top - bot if top > bot else 1.0

    if p is not None:
        ybr = top + 0.06 * span
        h = 0.03 * span
        ax.plot([x[0], x[0], x[1], x[1]], [ybr, ybr + h, ybr + h, ybr],
                color="k", lw=1.0, clip_on=False)
        ax.text(0.5, ybr + h, pstars(p), ha="center", va="bottom",
                color="k", fontsize=9, clip_on=False)
        ax.text(0.5, ybr + h + 0.085 * span, fmt_emp_p(p, n_shuff),
                ha="center", va="bottom", color="k", fontsize=7, clip_on=False)
        ax.set_ylim(bot - 0.05 * span, ybr + h + 0.085 * span + 0.10 * span)
    else:
        ax.set_ylim(bot - 0.05 * span, top + 0.10 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, 1.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return viol


def plot_fano_population(ax=None, refresh=False, data=None, window_ms=25.0):
    """Single-window population (slope-through-origin) Fano factor: one faint
    line per session joining its uncorrected and corrected value, with the
    across-session mean +/- SD overlaid (open uncorrected -> filled corrected,
    joined by a line) and the Poisson (Fano = 1) reference dotted. A grey
    shuffle-null band at the corrected marker shows where the corrected mean
    would fall if the FEM correction were random, and the bracket carries stars
    + the explicit shuffle-null p-value (session-matched to the across-session
    mean statistic)."""
    if data is None:
        raise ValueError("Supply the committed poster result cache")
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 3.0))
    else:
        fig = ax.figure

    w = nearest_window(data["WINDOWS_MS"], window_ms)
    s = data["fano_stats"][w]
    su = np.asarray(s.get("sess_slope_unc", []), dtype=float)
    sc = np.asarray(s.get("sess_slope_cor", []), dtype=float)
    if su.size == 0 or sc.size == 0:
        raise ValueError("Missing saved per-session Fano slopes")
    x = [0.0, 1.0]

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.8, zorder=0)
    # Sit just above the Fano = 1 line, left-justified a touch right of the
    # FEM-corrected tick (x = 1.0).
    ax.text(1.06, 1.0, "Poisson", color="gray", fontsize=7, va="bottom",
            ha="left")

    # one faint line per session
    for a, b in zip(su, sc):
        ax.plot(x, [a, b], color="0.75", lw=0.7, alpha=0.7, zorder=1)

    # Grey shuffle-null band (2.5-97.5%) at the corrected marker: where the
    # corrected across-session mean would fall if the FEM correction were random.
    null_lo, null_hi = s.get("mean_sess_cor_null_ci", (np.nan, np.nan))
    if np.isfinite(null_lo) and np.isfinite(null_hi):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (x[1] - 0.2, null_lo), 0.4, null_hi - null_lo,
            facecolor="0.6", alpha=0.30, edgecolor="0.35", lw=0.7,
            linestyle="--", zorder=2))
        ax.text(x[1] + 0.22, 0.5 * (null_lo + null_hi), "shuffle\nnull",
                color="0.4", fontsize=6.5, va="center", ha="left")

    # across-session mean +/- SD overlay
    mu = (float(su.mean()), float(sc.mean()))
    sd = (float(su.std(ddof=1)), float(sc.std(ddof=1)))
    ax.plot(x, mu, "-", color="tab:blue", lw=2.0, zorder=4)
    ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="tab:blue", elinewidth=1.5,
                capsize=0, zorder=5)
    ax.plot(x[0], mu[0], "o", mfc="white", mec="tab:blue", ms=7, mew=1.6,
            zorder=6)
    ax.plot(x[1], mu[1], "o", mfc="tab:blue", mec="tab:blue", ms=7, mew=1.6,
            zorder=6)

    p = s.get("p_emp_mean_sess", np.nan)
    n_shuff = s.get("n_shuff_sess")

    lo = np.nanmin([float(su.min()), float(sc.min()), 1.0, mu[1] - sd[1],
                    null_lo])
    hi = np.nanmax([float(su.max()), float(sc.max()), 1.0, mu[0] + sd[0],
                    null_hi])
    span = hi - lo if hi > lo else 1.0
    ybr = hi + 0.06 * span
    h = 0.03 * span
    ax.plot([x[0], x[0], x[1], x[1]], [ybr, ybr + h, ybr + h, ybr],
            color="k", lw=1.0, clip_on=False)
    ax.text(0.5, ybr + h, pstars(p), ha="center", va="bottom", color="k",
            fontsize=9, clip_on=False)
    ax.text(0.5, ybr + h + 0.085 * span, fmt_emp_p(p, n_shuff), ha="center",
            va="bottom", color="k", fontsize=7, clip_on=False)
    ax.set_ylim(lo - 0.06 * span, ybr + h + 0.085 * span + 0.12 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(["uncorrected", "FEM-\ncorrected"])
    ax.set_ylabel("Variability (Pop. Fano factor)")
    ax.set_xlim(-0.5, 1.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_nc_violin(ax=None, refresh=False, data=None, window_ms=25.0):
    """Single-window per-pair noise correlation (rho), matching panel E's
    grammar: grey violins of the per-pair distribution (uncorrected vs
    FEM-corrected) with the across-dataset mean noise correlation overlaid as
    open->filled markers +/- SD, joined by
    a line. A grey shuffle-null band at the corrected marker shows where the
    corrected mean would fall under the principled delta-correlation shuffle,
    and the bracket carries stars + the explicit shuffle-null p-value."""
    if data is None:
        raise ValueError("Supply the committed poster result cache")
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 3.0))
    else:
        fig = ax.figure

    w = nearest_window(data["WINDOWS_MS"], window_ms)
    s = data["nc_stats"][w]

    # Markers: across-dataset mean of the per-dataset mean correlation, with
    # +/- SD whiskers across datasets.
    ru, rc = s["r_u_mean"], s["r_c_mean"]
    su, sc = s["r_u_sd"], s["r_c_sd"]
    mean_u, mean_c = float(ru), float(rc)
    err_u = (float(ru - su), float(ru + su))
    err_c = (float(rc - sc), float(rc + sc))

    # Shuffle-null band (2.5-97.5%) for the corrected mean = uncorrected mean +
    # null delta-r. The null is the across-session-mean delta-r under the
    # eye-trajectory shuffle (same aggregation level as the observed mean).
    dr_lo, dr_hi = s["null_dr_ci"]
    null_lo = float(ru + dr_lo)
    null_hi = float(ru + dr_hi)

    pair_violin(
        ax,
        np.asarray(s["rho_u"], dtype=float),
        np.asarray(s["rho_c"], dtype=float),
        mean_u=mean_u, mean_c=mean_c, err_u=err_u, err_c=err_c,
        null_lo=null_lo, null_hi=null_hi,
        p=s["p_emp_dr"], n_shuff=s.get("n_shuff_dr"),
        ref=0.0, ylabel=r"Noise correlation ($r_{\mathrm{sc}}$)",
    )
    return fig, ax
