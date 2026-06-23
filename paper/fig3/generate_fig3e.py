"""Figure 3 panel E: single-trial rasters (observed | twin) for example neuron.

Usage:
    uv run declan/fig3/generate_fig3e.py
"""
import matplotlib.pyplot as plt

from _fig3_data import FIG_DIR, configure_matplotlib, load_fig3_data
from _fig3_helpers import draw_raster_pair, select_example_neuron


def plot_panel_e(ax=None, data=None, example=None,
                 label_fontsize=9, scale_fontsize=8, colorbar=True):
    """Draw concatenated observed|twin raster on `ax`. Returns (fig, ax, im, example)."""
    if data is None:
        data = load_fig3_data()
    if example is None:
        example = select_example_neuron(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
    else:
        fig = ax.figure

    im = draw_raster_pair(
        ax, example["robs_trials_rate"], example["rhat_trials_rate"],
        window_s=example["window_s"],
        vmin=example["vmin"], vmax=example["vmax"],
        label_fontsize=label_fontsize, scale_fontsize=scale_fontsize,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="sp/s")
    return fig, ax, im, example


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax, im, _ = plot_panel_e()
    out = FIG_DIR / "panel_e_rasters.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
