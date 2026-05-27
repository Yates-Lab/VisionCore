"""Standalone PDF/PNG render of the stimulus half of Figure 4 panel A.

Useful for iterating on the stimulus layout in isolation, without paying
the cost of rendering the architecture half each time.

Usage:
    uv run ryan/fig4/generate_fig4a_stimulus.py [--recompute]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib
from _fig4a_data import load_panel_a_assets
from _fig4a_stimulus import plot_panel_a_stimulus, CANVAS_W, CANVAS_H


def main(recompute=False):
    configure_matplotlib()
    assets = load_panel_a_assets(recompute=recompute)

    # Width chosen so the data-coords aspect matches the figure aspect at
    # this height. CANVAS_W : CANVAS_H is the data extent of the axes.
    fig_h = 3.6
    fig_w = fig_h * (CANVAS_W / CANVAS_H)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_panel_a_stimulus(ax, assets)
    fig.tight_layout(pad=0.15)

    out_pdf = FIG_DIR / "panel_a_stimulus.pdf"
    out_png = FIG_DIR / "panel_a_stimulus.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render panel A stimulus only.")
    p.add_argument("--recompute", action="store_true",
                   help="Force a fresh asset rebuild (gratings/fixrsvp/eye trace).")
    args = p.parse_args()
    main(recompute=args.recompute)
