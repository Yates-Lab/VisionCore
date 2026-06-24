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


def standalone_save(fig, name):
    """Save panel figure as both .pdf and .png under FIG_DIR."""
    out = FIG_DIR / name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out.with_suffix('.pdf')}")
