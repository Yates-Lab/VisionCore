"""
Figure 3 panel F: participation-ratio scatter (FEM vs PSTH), one point per
session, colored by subject.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_f(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    sub_subjects = data["sub_subjects"]
    pr_fem_list = data["pr_fem_list"]
    pr_psth_list = data["pr_psth_list"]

    for subj in sorted(set(sub_subjects)):
        s_mask = np.array(sub_subjects) == subj
        ax.scatter(np.array(pr_fem_list)[s_mask],
                   np.array(pr_psth_list)[s_mask],
                   c=SUBJECT_COLORS.get(subj, "gray"), s=40,
                   edgecolors="black", linewidths=0.5, label=subj)
    pr_max = max(np.max(pr_psth_list), np.max(pr_fem_list)) * 1.1
    ax.plot([0, pr_max], [0, pr_max], "k--", alpha=0.3)
    ax.set_xlim(0, pr_max)
    ax.set_ylim(0, pr_max)
    ax.set_xlabel("FEM PR")
    ax.set_ylabel("PSTH PR")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_f()
    fig.tight_layout()
    standalone_save(fig, "panel_f_participation_ratio")
