"""
Figure 3 panel G: subspace alignment scatter — PSTH variance captured by
the FEM subspace (X) vs FEM variance captured by the PSTH subspace (Y).
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_g(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    sub_subjects = data["sub_subjects"]
    var_p_given_f = data["var_p_given_f"]
    var_f_given_p = data["var_f_given_p"]

    for subj in sorted(set(sub_subjects)):
        s_mask = [s == subj for s in sub_subjects]
        ax.scatter(np.array(var_p_given_f)[s_mask],
                   np.array(var_f_given_p)[s_mask],
                   c=SUBJECT_COLORS.get(subj, "gray"), s=40,
                   edgecolors="black", linewidths=0.5, label=subj)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("X: PSTH var in FEM subspace")
    ax.set_ylabel("Y: FEM var in PSTH subspace")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_g()
    fig.tight_layout()
    standalone_save(fig, "panel_g_subspace_alignment")
