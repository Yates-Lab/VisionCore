"""Figure 4: Digital Twin Performance — orchestrator.

Renders each panel's standalone PDF, then assembles the composite figure
with the native schematic (Panel A) and result panels B-J.

Panels:
  A  Architecture schematic (stimulus + model, drawn natively)   -> generate_fig4a
  B  Example neuron PSTH overlay (observed + twin)               -> generate_fig4b
  C  Histogram of normalized correlation (ccnorm)                -> generate_fig4c
  D  (reserved) per-cell rate-fluctuation decomposition vs 1-α   -> TODO
  E  Single-trial rasters: observed | twin (same neuron as B)    -> generate_fig4e
  F  Single-trial r^2 scatter: model vs PSTH                     -> generate_fig4f
  G  Improvement over PSTH vs FEM modulation (1-α)               -> generate_fig4g
  H  Ablation rasters: twin | ablated | residual                 -> generate_fig4h
  I  Single-trial r^2: ablated vs intact                         -> generate_fig4i
  J  Ablation cost (Δr^2) vs FEM modulation (1-α)                -> generate_fig4j

Layout (8 x 10.5 in):
  row 0  A           full-width schematic
  row 1  B  C  D     phenomenology (PSTH, ccnorm, rate-fluctuation decomp)
  row 2  E  F  G     single-trial structure (rasters, r^2 scatter, improvement)
  row 3  H  I  J     extraretinal ablation

Usage:
    uv run ryan/fig4/generate_figure4.py [--recompute]
"""
import argparse
import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib, load_fig4_data
from _fig4_ablation_data import load_ablation_data, print_ablation_stats
from _fig4_helpers import select_example_neuron
from _fig4a_data import load_panel_a_assets
from generate_fig4a import render_panel_a, _plot_halves, _fit_two_axes_in_rect
from generate_fig4b import plot_panel_b
from generate_fig4c import plot_panel_c
from generate_fig4d import plot_panel_d, load_panel_d_results, export_panel_d_stats
from generate_fig4e import plot_panel_e
from generate_fig4f import plot_panel_f
from generate_fig4g import plot_panel_g
from generate_fig4h import plot_panel_h
from generate_fig4i import plot_panel_i
from generate_fig4j import plot_panel_j


def _place_schematic(fig, rect, assets):
    """Fit the two-half schematic to fill `rect` (a figure-fraction Bbox).

    `rect` should be the *settled* extent of the schematic's row, read back
    via `ax.get_position()` after constrained_layout has run — the subplotspec
    position uses default margins and leaves the schematic inset.
    """
    ax_stim = fig.add_axes([rect.x0, rect.y0, rect.width / 2, rect.height])
    ax_arch = fig.add_axes([rect.x0 + rect.width / 2, rect.y0,
                            rect.width / 2, rect.height])
    a_stim, a_arch = _plot_halves(fig, ax_stim, ax_arch, assets)
    _fit_two_axes_in_rect(fig, ax_stim, ax_arch, rect, a_stim, a_arch)
    return ax_stim, ax_arch


def _save_standalone_panels(data, example, abl, panel_d):
    fig_a, _ = render_panel_a()
    fig_a.savefig(FIG_DIR / "panel_a_schematic.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_a)

    fig_b, _, _ = plot_panel_b(data=data, example=example)
    fig_b.tight_layout()
    fig_b.savefig(FIG_DIR / "panel_b_psth.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_b)

    fig_c, _ = plot_panel_c(data=data)
    fig_c.tight_layout()
    fig_c.savefig(FIG_DIR / "panel_c_ccnorm_hist.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_c)

    fig_d, _ = plot_panel_d(data=data, results=panel_d)
    fig_d.tight_layout()
    fig_d.savefig(FIG_DIR / "panel_d_one_minus_alpha.pdf", bbox_inches="tight", dpi=300)
    fig_d.savefig(FIG_DIR / "panel_d_one_minus_alpha.png", bbox_inches="tight", dpi=300)
    plt.close(fig_d)

    fig_e, _, _, _ = plot_panel_e(data=data, example=example)
    fig_e.savefig(FIG_DIR / "panel_e_rasters.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_e)

    fig_f, _ = plot_panel_f(data=data)
    fig_f.tight_layout()
    fig_f.savefig(FIG_DIR / "panel_f_r2_scatter.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_f)

    fig_g, _ = plot_panel_g(data=data)
    fig_g.tight_layout()
    fig_g.savefig(FIG_DIR / "panel_g_improvement_vs_fem.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig_g)

    # Bottom-row ablation panels, both conditions (zeroed committed, permuted supp.)
    for cond in ("zeroed", "permuted"):
        fig_h, ax_h, im_rate, im_resid = plot_panel_h(data=abl, cond=cond)
        if im_resid is not None:
            fig_h.colorbar(im_rate, ax=ax_h, shrink=0.7, pad=0.02, label="rate (sp/s)")
            fig_h.colorbar(im_resid, ax=ax_h, shrink=0.7, pad=0.08, label="Δ rate")
        fig_h.savefig(FIG_DIR / f"panel_h_rasters_{cond}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_h)

        fig_i, _ = plot_panel_i(data=abl, cond=cond, print_stats=False)
        fig_i.tight_layout()
        fig_i.savefig(FIG_DIR / f"panel_i_r2_ablated_{cond}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_i)

        fig_j, _ = plot_panel_j(data=abl, cond=cond, print_stats=False)
        fig_j.tight_layout()
        fig_j.savefig(FIG_DIR / f"panel_j_cost_vs_fem_{cond}.pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_j)


def _build_composite(data, example, abl, panel_d):
    assets = load_panel_a_assets()
    fig = plt.figure(figsize=(8, 10.5), constrained_layout=True)
    gs_outer = fig.add_gridspec(4, 1, height_ratios=[2.7, 2.6, 2.6, 2.6])

    # Row 0 reserves the full-width schematic slot. We leave it bare and fit
    # the schematic into its settled extent after layout (see below).
    ax_a = fig.add_subplot(gs_outer[0])
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax_a.spines[side].set_visible(False)

    # --- Row 1 — phenomenology: B (PSTH), C (ccnorm), D (reserved) ---
    gs_r1 = gs_outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.2)

    ax_b = fig.add_subplot(gs_r1[0, 0])
    plot_panel_b(ax=ax_b, data=data, example=example,
                 legend_fontsize=7, show_ccnorm_title=False)
    ax_b.set_title("B", fontweight="bold", loc="left")

    ax_c = fig.add_subplot(gs_r1[0, 1])
    plot_panel_c(ax=ax_c, data=data, legend_fontsize=7)
    ax_c.set_title("C", fontweight="bold", loc="left")

    ax_d = fig.add_subplot(gs_r1[0, 2])
    plot_panel_d(ax=ax_d, data=data, results=panel_d, legend_fontsize=7,
                 print_stats=False)
    ax_d.set_title("D", fontweight="bold", loc="left")

    # --- Row 2 — single-trial structure: E (rasters), F (scatter), G (improvement) ---
    gs_r2 = gs_outer[2].subgridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.2)

    ax_e = fig.add_subplot(gs_r2[0, 0])
    _, _, im_e, _ = plot_panel_e(
        ax=ax_e, data=data, example=example,
        label_fontsize=8, scale_fontsize=7, colorbar=False,
    )
    ax_e.set_title("E", fontweight="bold", loc="left")
    fig.colorbar(im_e, ax=ax_e, shrink=0.8, pad=0.02, label="sp/s")

    ax_f = fig.add_subplot(gs_r2[0, 1])
    plot_panel_f(ax=ax_f, data=data, legend_fontsize=7, print_stats=False)
    ax_f.set_title("F", fontweight="bold", loc="left")

    ax_g = fig.add_subplot(gs_r2[0, 2])
    plot_panel_g(ax=ax_g, data=data, legend_fontsize=7, print_stats=False)
    ax_g.set_title("G", fontweight="bold", loc="left")

    # --- Row 3 — extraretinal ablation: H (rasters), I (r2), J (cost) ---
    gs_r3 = gs_outer[3].subgridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.2)

    ax_h = fig.add_subplot(gs_r3[0, 0])
    _, _, _, im_resid = plot_panel_h(ax=ax_h, data=abl, cond="zeroed",
                                     label_fontsize=8)
    ax_h.set_title("H", fontweight="bold", loc="left")
    if im_resid is not None:
        fig.colorbar(im_resid, ax=ax_h, shrink=0.8, pad=0.02, label="Δ rate")

    ax_i = fig.add_subplot(gs_r3[0, 1])
    plot_panel_i(ax=ax_i, data=abl, cond="zeroed", legend_fontsize=7,
                 print_stats=False)
    ax_i.set_title("I", fontweight="bold", loc="left")

    ax_j = fig.add_subplot(gs_r3[0, 2])
    plot_panel_j(ax=ax_j, data=abl, cond="zeroed", legend_fontsize=7,
                 print_stats=False)
    ax_j.set_title("J", fontweight="bold", loc="left")

    # Let constrained_layout settle the result rows, then fill row 0's actual
    # extent with the schematic (subplotspec position would leave it inset).
    fig.draw_without_rendering()
    rect = ax_a.get_position()
    _place_schematic(fig, rect, assets)
    fig.text(rect.x0, rect.y1, "A", fontweight="bold", ha="left", va="top",
             fontsize=plt.rcParams["axes.titlesize"])

    fig.savefig(FIG_DIR / "fig4_composite.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig4_composite.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _build_supplement_permuted(abl):
    """Supplementary figure: the permuted (adversarial) ablation control."""
    fig = plt.figure(figsize=(11, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    _, _, _, im_resid = plot_panel_h(ax=ax_a, data=abl, cond="permuted")
    ax_a.set_title("A", fontweight="bold", loc="left")
    if im_resid is not None:
        fig.colorbar(im_resid, ax=ax_a, shrink=0.8, pad=0.02, label="Δ rate")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_i(ax=ax_b, data=abl, cond="permuted", legend_fontsize=7,
                 print_stats=False)
    ax_b.set_title("B", fontweight="bold", loc="left")

    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_j(ax=ax_c, data=abl, cond="permuted", legend_fontsize=7,
                 print_stats=False)
    ax_c.set_title("C", fontweight="bold", loc="left")

    fig.suptitle("Fig 4 supplement — extraretinal ablation, permuted control")
    fig.savefig(FIG_DIR / "fig4_supp_permuted.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "fig4_supp_permuted.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main(recompute=False):
    configure_matplotlib()
    data = load_fig4_data(recompute=recompute)
    example = select_example_neuron(data)
    abl = load_ablation_data(recompute=recompute)
    print_ablation_stats(abl)

    panel_d = load_panel_d_results(data=data, recompute=recompute)
    export_panel_d_stats(panel_d)

    _save_standalone_panels(data, example, abl, panel_d)
    _build_composite(data, example, abl, panel_d)
    _build_supplement_permuted(abl)

    print(f"\nAll panel figures saved to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compose figure 4.")
    p.add_argument("--recompute", action="store_true",
                   help="Force model re-inference (skip cache).")
    args = p.parse_args()
    main(recompute=args.recompute)
