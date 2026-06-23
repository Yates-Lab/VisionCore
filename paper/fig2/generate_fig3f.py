"""
Figure 3 panel F: participation-ratio scatter (FEM vs PSTH), one point per
session, colored by subject. Per-subject legend entries carry an exact
one-sided sign test (PSTH PR > FEM PR) annotated with stars.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binomtest

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def _stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def plot_panel_f(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    SUBJECT_DISPLAY_NAMES = data.get("SUBJECT_DISPLAY_NAMES", {})
    sub_subjects = np.array(data["sub_subjects"])
    pr_fem = np.array(data["pr_fem_list"])
    pr_psth = np.array(data["pr_psth_list"])

    legend_handles = []
    for subj in sorted(set(sub_subjects)):
        s_mask = sub_subjects == subj
        ax.scatter(pr_fem[s_mask], pr_psth[s_mask],
                   c=SUBJECT_COLORS.get(subj, "gray"), s=40,
                   edgecolors="black", linewidths=0.5)

        # Exact one-sided sign test: PSTH PR > FEM PR per session.
        diffs = pr_psth[s_mask] - pr_fem[s_mask]
        diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
        n = diffs.size
        if n > 0:
            k = int((diffs > 0).sum())
            p = binomtest(k, n, p=0.5, alternative="greater").pvalue
        else:
            p = np.nan
        subj_label = SUBJECT_DISPLAY_NAMES.get(subj, f"Monkey {subj[0]}")
        label = f"{subj_label} {_stars(p)}"
        legend_handles.append(Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=SUBJECT_COLORS.get(subj, "gray"),
            markeredgecolor="black", markeredgewidth=0.5, markersize=7,
            label=label,
        ))

    pr_max = max(np.max(pr_psth), np.max(pr_fem)) * 1.1
    ax.plot([0, pr_max], [0, pr_max], "k--", alpha=0.3)
    ax.set_xlim(0, pr_max)
    ax.set_ylim(0, pr_max)
    ax.set_xlabel("FEM PR")
    ax.set_ylabel("PSTH PR")
    ax.legend(handles=legend_handles, frameon=False, fontsize=7,
              loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_f()
    fig.tight_layout()
    standalone_save(fig, "panel_f_participation_ratio")
