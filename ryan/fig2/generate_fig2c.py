"""
Figure 2 panel C: FEM modulation fraction (1 - alpha) histogram.

Per-subject overlay across the primary counting window.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_c(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]

    m0_full = data["m_by_window"][0]
    labels = data["subject_per_neuron_by_window"][0]

    valid_m0 = m0_full[np.isfinite(m0_full)]
    bins = np.linspace(np.nanmin(valid_m0), np.nanmax(valid_m0), 31)

    present = [s for s in reversed(SUBJECTS) if (labels == s).any()]
    subj_m0 = [m0_full[labels == s] for s in present]
    subj_colors = [SUBJECT_COLORS[s] for s in present]
    ax.hist(subj_m0, bins=bins, color=subj_colors, edgecolor="white",
            stacked=True, alpha=0.75)

    y_marker = ax.get_ylim()[1] * 1.02
    for m0, color in zip(subj_m0, subj_colors):
        ax.plot(np.nanmedian(m0), y_marker, marker="v", color=color,
                markersize=10, clip_on=False)

    ax.set_xlabel("Fraction of rate modulation\ndue to FEM (1-α)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(True, alpha=0.3, zorder=-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_c()
    fig.tight_layout()
    standalone_save(fig, "panel_c_alpha")
