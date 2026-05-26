"""
Compose Figure 2: arrange panels C-K (FEM-modulation histogram, Fano
factor analysis, noise correlations, and subspace alignment) into a
single composite figure.

Each panel is rendered by its own module (generate_fig2c.py … 2k.py) that
loads precomputed derived data from compute_fig2_data.py. This composer
just lays them out with GridSpec and saves the result.

Usage:
    uv run ryan/fig2/generate_figure2.py
    uv run ryan/fig2/generate_figure2.py --refresh    # recompute derived data
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data
from generate_fig2c import plot_panel_c
from generate_fig2d import plot_panel_d
from generate_fig2e import plot_panel_e
from generate_fig2f import plot_panel_f
from generate_fig2g import plot_panel_g
from generate_fig2h import plot_panel_h
from generate_fig2i import plot_panel_i
from generate_fig2j import plot_panel_j
from generate_fig2k import plot_panel_k


def compose(refresh=False):
    data = load_fig2_data(refresh=refresh)

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    gs = GridSpec(3, 12, figure=fig, hspace=0.1, wspace=0.1)

    panels = [
        ("C", plot_panel_c, gs[0, 0:4]),
        ("D", plot_panel_d, gs[0, 4:8]),
        ("E", plot_panel_e, gs[0, 8:12]),
        ("F", plot_panel_f, gs[1, 0:4]),
        ("G", plot_panel_g, gs[1, 4:8]),
        ("H", plot_panel_h, gs[1, 8:12]),
        ("I", plot_panel_i, gs[2, 0:4]),
        ("J", plot_panel_j, gs[2, 4:8]),
        ("K", plot_panel_k, gs[2, 8:12]),
    ]

    for letter, plot_fn, spec in panels:
        ax = fig.add_subplot(spec)
        plot_fn(ax=ax, data=data)
        ax.set_title(letter, loc="left", fontweight="bold")

    out = FIG_DIR / "fig2_composite.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out}")


def _parse_args():
    p = argparse.ArgumentParser(description="Compose figure 2.")
    p.add_argument("-r", "--refresh", action="store_true",
                   help="Force recompute of derived fig2 data.")
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    compose(refresh=args.refresh)
