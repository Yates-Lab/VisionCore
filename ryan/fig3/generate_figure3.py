"""
Compose Figure 3: covariance decomposition demo (A) on top, population
metrics (B–G: noise correlations, eigenspectra, subspace alignment) below.

Panel A is a full-width row with the equation header + 4 matrix
heatmaps; B–G occupy a 2×3 grid below it.

Usage:
    uv run ryan/fig3/generate_figure3.py
    uv run ryan/fig3/generate_figure3.py --refresh
"""
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data
from generate_fig3a import plot_panel_a, make_axes as make_axes_a
from generate_fig3b import plot_panel_b
from generate_fig3c import plot_panel_c
from generate_fig3d import plot_panel_d
from generate_fig3e import plot_panel_e
from generate_fig3f import plot_panel_f
from generate_fig3g import plot_panel_g


def compose(refresh=False):
    data = load_fig2_data(refresh=refresh)

    # Publication-scale font sizes for an ~8" wide composite.
    with mpl.rc_context({"font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig = plt.figure(figsize=(8, 8.5))
        gs = GridSpec(3, 3, height_ratios=[1.5, 1.0, 1.0],
                      hspace=0.30, wspace=0.45, figure=fig,
                      left=0.07, right=0.98, top=0.97, bottom=0.07)

        axes_a = make_axes_a(fig, subplot_spec=gs[0, :])
        plot_panel_a(axes=axes_a, data=data, font_scale=0.5)

        for letter, plot_fn, spec in [
            ("B", plot_panel_b, gs[1, 0]),
            ("C", plot_panel_c, gs[1, 1]),
            ("D", plot_panel_d, gs[1, 2]),
            ("E", plot_panel_e, gs[2, 0]),
            ("F", plot_panel_f, gs[2, 1]),
            ("G", plot_panel_g, gs[2, 2]),
        ]:
            ax = fig.add_subplot(spec)
            _, primary_ax = plot_fn(ax=ax, data=data)
            primary_ax.set_title(letter, loc="left", fontweight="bold",
                                 fontsize=10)

    out = FIG_DIR / "fig3_composite.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out}")


def _parse_args():
    p = argparse.ArgumentParser(description="Compose figure 3.")
    p.add_argument("-r", "--refresh", action="store_true",
                   help="Force recompute of derived fig2 data.")
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    compose(refresh=args.refresh)
