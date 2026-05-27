"""
Figure 3 panel B: noise correlation scatter (FEM-corrected vs uncorrected)
for the primary counting window, colored by subject.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_b(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    WINDOWS_MS = data["WINDOWS_MS"]
    s0 = data["nc_stats"][WINDOWS_MS[0]]
    pair_labels = data["metrics"][0]["subject_per_pair"]

    for subj in SUBJECTS:
        mask = pair_labels == subj
        if not mask.any():
            continue
        color = SUBJECT_COLORS[subj]
        ax.scatter(s0["rho_u"][mask], s0["rho_c"][mask],
                   s=1, alpha=0.05, c=color, rasterized=True)
        ax.plot(np.mean(s0["rho_u"][mask]), np.mean(s0["rho_c"][mask]),
                "o", color=color, markersize=5, markeredgecolor="black",
                markeredgewidth=0.5, label=subj)
    ax.plot([-0.3, 0.3], [-0.3, 0.3], "k--", alpha=0.3, linewidth=0.5)
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel("ρ uncorrected")
    ax.set_ylabel("ρ FEM-corrected")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_b()
    fig.tight_layout()
    standalone_save(fig, "panel_b_noisecorr_scatter")
