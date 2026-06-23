"""Standalone PDF/PNG render of the architecture half of Figure 3 panel A.

Useful for iterating on the architecture layout in isolation.

Usage:
    uv run declan/fig3/generate_fig3a_architecture.py [--recompute]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from _fig3_data import FIG_DIR, configure_matplotlib
from _fig3a_data import load_panel_a_assets
from _fig3a_architecture import plot_panel_a_architecture


def main(recompute=False):
    configure_matplotlib()
    assets = load_panel_a_assets(recompute=recompute)

    # Figure size is sacrificial — bbox_inches="tight" crops to the axes box,
    # which set_aspect("equal") inside plot_panel_a_architecture sizes to the
    # true data aspect (no horizontal stretch from tight_layout).
    fig, ax = plt.subplots(figsize=(12, 6.0))
    plot_panel_a_architecture(ax, assets)

    out_pdf = FIG_DIR / "panel_a_architecture.pdf"
    out_png = FIG_DIR / "panel_a_architecture.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render panel A architecture only.")
    p.add_argument("--recompute", action="store_true",
                   help="Force a fresh asset rebuild.")
    args = p.parse_args()
    main(recompute=args.recompute)
