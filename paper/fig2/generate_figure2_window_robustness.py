"""
Figure 2 supplemental — robustness across counting windows.

Shows the three headline metrics as a function of counting window, demonstrating
that the FEM-correction result is not specific to the 25 ms window used in the
main figure:

    A: FEM modulation fraction (1 - alpha) vs window (pooled, with shuffle null)
    B: population Fano factor vs window (uncorrected vs FEM-corrected, per subject)
    C: mean noise correlation (Fisher z) vs window (uncorrected vs FEM-corrected,
       per subject)

Usage:
    uv run paper/fig2/generate_figure2_window_robustness.py
    uv run paper/fig2/generate_figure2_window_robustness.py --refresh
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data
from generate_panel_femfraction import plot_alpha_vs_window
from generate_panel_fano import plot_panel_e as plot_fano_vs_window
from generate_panel_noisecorr import plot_panel_c as plot_nc_vs_window


def _label(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold")


def compose(refresh=False):
    data = load_fig2_data(refresh=refresh)

    fig = plt.figure(figsize=(12, 3.8))
    gs = GridSpec(1, 3, wspace=0.34, figure=fig)
    for letter, fn, col in [
        ("A", plot_alpha_vs_window, 0),
        ("B", plot_fano_vs_window, 1),
        ("C", plot_nc_vs_window, 2),
    ]:
        ax = fig.add_subplot(gs[0, col])
        fn(ax=ax, data=data)
        _label(ax, letter)

    stem = FIG_DIR / "figure2_window_robustness"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Figure 2 window-robustness supplemental.")
    p.add_argument("-r", "--refresh", action="store_true",
                   help="Force recompute of derived fig2 data.")
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    compose(refresh=args.refresh)
