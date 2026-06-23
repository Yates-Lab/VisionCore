"""
Compose Figure 2:
    Row 1: A (eye + spike traces) + B (Δ-eye histogram + rate variance)
    Row 2: C (FEM modulation) + D (Δ variance vs rate) + E (Fano factor)

Population panels have moved to Figure 3.

Usage:
    uv run ryan/fig2/generate_figure2.py
    uv run ryan/fig2/generate_figure2.py --refresh
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data
from generate_fig2a import plot_panel_a, make_axes as make_axes_a
from generate_fig2c import plot_panel_c
from generate_fig2d import plot_panel_d
from generate_fig2e import plot_panel_e


def compose(refresh=False):
    data = load_fig2_data(refresh=refresh)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, height_ratios=[1.15, 0.85], hspace=0.22, wspace=0.32,
                  figure=fig)

    # --- Row 1: panel A (left half) + panel B (right half), both inside
    # the lead-in 2x2 layout that A's make_axes builds.
    axes_a = make_axes_a(fig, subplot_spec=gs[0, :])
    plot_panel_a(axes=axes_a)

    # --- Row 2: C, D, E ---
    for letter, plot_fn, spec in [
        ("C", plot_panel_c, gs[1, 0]),
        ("D", plot_panel_d, gs[1, 1]),
        ("E", plot_panel_e, gs[1, 2]),
    ]:
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
