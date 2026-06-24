r"""Figure 3 digital-twin row (panels B-E) standalone.

Builds only the digital-twin row of the combined figure 3 -- B (example PSTH),
C (held-out responses / ccnorm), D (eye-state zeroing ablation), E (FEM-linked
model gain) -- reusing the panel functions from ``generate_figure3_combined``.
Panels F-I (reafferent geometry, the declan/ TFTS pipeline) are produced
separately and are intentionally not built here.

Usage:
    uv run python paper/fig3/generate_figure3_digitaltwin_row.py [--recompute]
"""
import argparse

import matplotlib.pyplot as plt

from _fig3_data import FIG_DIR, configure_matplotlib, load_fig3_data
from _fig3_helpers import select_example_neuron
import generate_figure3_combined as C


def main(recompute=False):
    configure_matplotlib()
    data = load_fig3_data(recompute=recompute)
    example = select_example_neuron(data)
    abl = C._load_ablation_cache()

    fig = plt.figure(figsize=(11.0, 2.7))
    gs = fig.add_gridspec(1, 4, wspace=0.42, left=0.05, right=0.985,
                          top=0.84, bottom=0.2)

    ax_b = fig.add_subplot(gs[0, 0])
    C._plot_example_psth_intact_vs_zeroed(
        ax_b, abl, fallback_data=data, fallback_example=example,
    )
    C._standard_panel_heading(ax_b, "B", "Example PSTH")

    ax_c = fig.add_subplot(gs[0, 1])
    C._plot_ccnorm_hist_intact_vs_zeroed(ax_c, data, letter="C")
    C._standard_panel_heading(ax_c, "C", "Held-out responses")

    ax_d = fig.add_subplot(gs[0, 2])
    if abl is not None:
        C._plot_ablation_r2_pooled(ax_d, abl, cond="zeroed", letter="D")
    else:
        C._plot_ablation_placeholder(ax_d)
    C._standard_panel_heading(ax_d, "D", "Eye-state zeroing")

    ax_e = fig.add_subplot(gs[0, 3])
    C._plot_improvement_vs_fem_modulation(ax_e, data)
    C._standard_panel_heading(ax_e, "E", "FEM-linked model gain")

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"figure3_digitaltwin_row.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="Recompute the digital-twin inference cache.")
    args = ap.parse_args()
    main(recompute=args.recompute)
