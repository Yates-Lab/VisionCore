"""Shared helpers and matplotlib defaults for fig3 panel scripts."""
import sys
from pathlib import Path

import matplotlib as mpl

from VisionCore.paths import FIGURES_DIR, STATS_DIR


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


FIG_DIR = FIGURES_DIR / "fig3"
STAT_DIR = STATS_DIR / "fig3"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Expose the fig2 compute module on sys.path (data is shared between fig2/fig3)
_FIG2_DIR = Path(__file__).resolve().parent.parent / "fig2"
if str(_FIG2_DIR) not in sys.path:
    sys.path.insert(0, str(_FIG2_DIR))


def standalone_save(fig, name):
    """Save panel figure as both .pdf and .png under FIG_DIR."""
    out = FIG_DIR / name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved {out.with_suffix('.pdf')}")
