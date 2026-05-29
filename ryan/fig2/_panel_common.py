"""Shared helpers and matplotlib defaults for fig2 panel scripts."""
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


def standalone_save(fig, name):
    """Save panel figure as both .pdf and .png under FIG_DIR."""
    out = FIG_DIR / name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out.with_suffix('.pdf')}")
