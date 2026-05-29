"""Minimal shared plotting style + figure directory for the methods write-up.

Figures land in ./figures (next to the write-up) so the .md compiles to a
self-contained HTML/PDF with relative image paths.
"""
from pathlib import Path
import matplotlib as mpl

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# distribution colors used consistently across all figures
C_FULL = "#1f6fb2"   # p(e)  -- full fixational distribution
C_CLOSE = "#c0392b"  # p(e)^2 -- close-pair / central distribution
C_TRUTH = "#444444"
C_OK = "#2e8b57"


def configure():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
    })


def save(fig, name):
    """Save a figure as PNG (for HTML/preview) under FIG_DIR; return the path."""
    configure()
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    return out
