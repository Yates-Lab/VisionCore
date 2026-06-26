"""Shared helpers and matplotlib defaults for merged fig2 panel scripts."""
import sys
from pathlib import Path

import matplotlib as mpl

from VisionCore.paths import FIGURES_DIR, STATS_DIR


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


FIG_DIR = FIGURES_DIR / "fig2"
STAT_DIR = STATS_DIR / "fig2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Keep local fig2 modules importable for scripts copied in from the old fig3.
_FIG2_DIR = Path(__file__).resolve().parent
if str(_FIG2_DIR) not in sys.path:
    sys.path.insert(0, str(_FIG2_DIR))


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


def sig_bracket(ax, x0, x1, y, text, *, h=None, color="k", lw=1.0, fontsize=9):
    """Draw a significance bracket spanning x0..x1 at height ``y`` with ``text``
    centered above it. ``h`` (tick height) defaults to 3% of the y-range."""
    if h is None:
        y_lo, y_hi = ax.get_ylim()
        h = 0.03 * (y_hi - y_lo)
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y],
            color=color, lw=lw, clip_on=False)
    ax.text(0.5 * (x0 + x1), y + h, text, ha="center", va="bottom",
            color=color, fontsize=fontsize, clip_on=False)


def pair_box(ax, a, b, *, ref=0.0, ref_label=None, p=None, ylabel=None,
             labels=("uncorrected", "FEM-\ncorrected")):
    """Two box-and-whisker columns (5-95 percentile whiskers, no fliers) for a
    paired uncorrected/corrected per-unit distribution, with a dotted reference
    line (Poisson=1, independence=0) and a significance bracket. Returns the
    boxplot dict."""
    import numpy as np
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    x = [0.0, 1.0]

    bp = ax.boxplot(
        [a, b], positions=x, widths=0.55, showfliers=False, whis=(5, 95),
        patch_artist=True, medianprops=dict(color="k", lw=1.3),
        boxprops=dict(linewidth=0.9), whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
    )
    for patch, face in zip(bp["boxes"], ["white", "tab:blue"]):
        patch.set_facecolor(face)
        patch.set_edgecolor("#263746")

    ax.axhline(ref, color="gray", linestyle=":", alpha=0.8, zorder=0)
    if ref_label is not None:
        ax.text(1.52, ref, ref_label, color="gray", fontsize=7,
                va="center", ha="left")

    top = max(float(np.percentile(a, 95)), float(np.percentile(b, 95)))
    bot = min(float(np.percentile(a, 2)), float(np.percentile(b, 2)), ref)
    span = top - bot
    if p is not None:
        sig_bracket(ax, x[0], x[1], top + 0.05 * span, pstars(p))
    ax.set_ylim(bot - 0.05 * span, top + 0.18 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, 1.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return bp


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


def standalone_save(fig, name):
    """Save panel figure as both .pdf and .png under FIG_DIR."""
    out = FIG_DIR / name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out.with_suffix('.pdf')}")
