"""Figure 4 panel B: example neuron PSTH overlay (observed + twin).

Usage:
    uv run ryan/fig4/generate_fig4b.py
"""
import matplotlib.pyplot as plt

from _fig4_data import FIG_DIR, configure_matplotlib, load_fig4_data
from _fig4_helpers import select_example_neuron


def plot_panel_b(ax=None, data=None, example=None,
                 legend_fontsize=8, show_ccnorm_title=True):
    """Draw the PSTH overlay on `ax`. Creates a new figure if `ax` is None."""
    if data is None:
        data = load_fig4_data()
    if example is None:
        example = select_example_neuron(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    t = example["t"]
    w = example["psth_window"]
    ax.plot(t[w], example["robs_trace"][w], 'k', linewidth=1, label="Observed")
    ax.plot(t[w], example["rhat_trace"][w], color='tab:red',
            linewidth=1, label="Twin")
    ax.set_xlim(0, example["window_s"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (sp/s)")
    if show_ccnorm_title:
        ax.set_title(f"ccnorm = {example['ccnorm']:.2f}")
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax, example


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax, _ = plot_panel_b()
    fig.tight_layout()
    out = FIG_DIR / "panel_b_psth.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
