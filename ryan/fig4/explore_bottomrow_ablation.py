"""Side-by-side viewer for the Fig 4 bottom-row ablation (zeroed vs permuted).

This is a convenience comparison view (zeroed on top, permuted on bottom) that
reuses the committed loader and panel functions — it holds NO inference logic of
its own. The canonical pipeline is `_fig4_ablation_data.py` (data) +
`generate_fig4{h,i,j}.py` (panels), assembled by `generate_figure4.py`.

Usage:
    uv run python explore_bottomrow_ablation.py            # render from cache
    uv run python explore_bottomrow_ablation.py --recompute # rebuild cache (GPU)
"""
import argparse
import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib
from _fig4_ablation_data import load_ablation_data, print_ablation_stats, ABLATIONS
from generate_fig4h import plot_panel_h
from generate_fig4i import plot_panel_i
from generate_fig4j import plot_panel_j


def main(recompute=False):
    configure_matplotlib()
    data = load_ablation_data(recompute=recompute)
    print_ablation_stats(data)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, cond in enumerate(ABLATIONS):
        _, _, _, im_resid = plot_panel_h(ax=axes[row, 0], data=data, cond=cond)
        if im_resid is not None:
            fig.colorbar(im_resid, ax=axes[row, 0], shrink=0.7, pad=0.02,
                         label="Δ rate")
        plot_panel_i(ax=axes[row, 1], data=data, cond=cond, print_stats=False)
        plot_panel_j(ax=axes[row, 2], data=data, cond=cond, print_stats=False)
        axes[row, 0].set_ylabel(f"{cond}\n", fontsize=10, fontweight="bold")

    fig.suptitle("Fig 4 bottom row — extraretinal ablation "
                 "(top: zeroed, bottom: permuted)", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "explore_bottomrow_ablation"
    fig.savefig(f"{out}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=150)
    print(f"\nsaved {out}.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recompute", action="store_true",
                   help="Force re-inference (ignore cache).")
    args = p.parse_args()
    main(recompute=args.recompute)
