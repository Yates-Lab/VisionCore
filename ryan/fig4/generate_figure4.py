"""Figure 4: Digital Twin Performance — orchestrator.

Renders each panel's standalone PDF, then assembles the composite figure
with the schematic SVG (Panel A) and panels B-F.

Panels:
  A  Architecture schematic (fig3-schematic.svg, placed in composite)
  B  Example neuron PSTH overlay (observed + twin)            -> generate_fig4b
  C  Single-trial rasters: observed | twin (same neuron as B) -> generate_fig4c
  D  Histogram of normalized correlation (ccnorm)             -> generate_fig4d
  E  Single-trial r^2 scatter: model vs PSTH                  -> generate_fig4e
  F  Improvement over PSTH vs FEM modulation (1-α)            -> generate_fig4f

Usage:
    uv run ryan/fig4/generate_figure4.py [--recompute]
"""
import argparse
import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib, load_fig4_data
from _fig4_helpers import select_example_neuron
from generate_fig4a import plot_panel_a
from generate_fig4b import plot_panel_b
from generate_fig4c import plot_panel_c
from generate_fig4d import plot_panel_d
from generate_fig4e import plot_panel_e
from generate_fig4f import plot_panel_f


def _save_standalone_panels(data, example):
    fig_a, _ = plot_panel_a()
    fig_a.tight_layout(pad=0.2)
    fig_a.savefig(FIG_DIR / "panel_a_schematic.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_a)

    fig_b, _, _ = plot_panel_b(data=data, example=example)
    fig_b.tight_layout()
    fig_b.savefig(FIG_DIR / "panel_b_psth.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_b)

    fig_c, _, _, _ = plot_panel_c(data=data, example=example)
    fig_c.savefig(FIG_DIR / "panel_c_rasters.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_c)

    fig_d, _ = plot_panel_d(data=data)
    fig_d.tight_layout()
    fig_d.savefig(FIG_DIR / "panel_d_ccnorm_hist.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_d)

    fig_e, _ = plot_panel_e(data=data)
    fig_e.tight_layout()
    fig_e.savefig(FIG_DIR / "panel_e_r2_scatter.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_e)

    fig_f, _ = plot_panel_f(data=data)
    fig_f.tight_layout()
    fig_f.savefig(FIG_DIR / "panel_f_improvement_vs_fem.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_f)


def _build_composite(data, example):
    # Wider canvas: panel A (full page width) on top, B-F in one row below.
    fig = plt.figure(figsize=(14, 7.0), constrained_layout=True)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.4])

    # --- Panel A: full-width schematic ---
    ax_a = fig.add_subplot(gs_outer[0])
    plot_panel_a(ax=ax_a)
    ax_a.set_title("A", fontweight="bold", loc="left", x=-0.005)

    # --- B-F in a single row below ---
    gs_bot = gs_outer[1].subgridspec(
        1, 5, width_ratios=[1, 1.2, 1, 1, 1], wspace=0.35
    )

    ax_b = fig.add_subplot(gs_bot[0, 0])
    plot_panel_b(ax=ax_b, data=data, example=example,
                 legend_fontsize=7, show_ccnorm_title=False)
    ax_b.set_title("B", fontweight="bold", loc="left")

    ax_c = fig.add_subplot(gs_bot[0, 1])
    _, _, im_c, _ = plot_panel_c(
        ax=ax_c, data=data, example=example,
        label_fontsize=8, scale_fontsize=7, colorbar=False,
    )
    ax_c.set_title("C", fontweight="bold", loc="left")
    fig.colorbar(im_c, ax=ax_c, shrink=0.8, pad=0.02, label="sp/s")

    ax_d = fig.add_subplot(gs_bot[0, 2])
    plot_panel_d(ax=ax_d, data=data, legend_fontsize=7)
    ax_d.set_title("D", fontweight="bold", loc="left")

    ax_e = fig.add_subplot(gs_bot[0, 3])
    plot_panel_e(ax=ax_e, data=data, legend_fontsize=7, print_stats=False)
    ax_e.set_title("E", fontweight="bold", loc="left")

    ax_f = fig.add_subplot(gs_bot[0, 4])
    plot_panel_f(ax=ax_f, data=data, legend_fontsize=7, print_stats=False)
    ax_f.set_title("F", fontweight="bold", loc="left")

    fig.savefig(FIG_DIR / "fig4_composite.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig4_composite.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main(recompute=False):
    configure_matplotlib()
    data = load_fig4_data(recompute=recompute)
    example = select_example_neuron(data)

    _save_standalone_panels(data, example)
    _build_composite(data, example)

    print(f"\nAll panel figures saved to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compose figure 4.")
    p.add_argument("--recompute", action="store_true",
                   help="Force model re-inference (skip cache).")
    args = p.parse_args()
    main(recompute=args.recompute)
