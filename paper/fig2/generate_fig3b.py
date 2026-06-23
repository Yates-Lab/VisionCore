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
    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    SUBJECT_DISPLAY_NAMES = data.get("SUBJECT_DISPLAY_NAMES", {})
    WINDOWS_MS = data["WINDOWS_MS"]
    s0 = data["nc_stats"][WINDOWS_MS[0]]
    pair_labels = data["metrics"][0]["subject_per_pair"]

    subj_counts = {s: int((pair_labels == s).sum()) for s in SUBJECTS}
    draw_order = sorted(
        [s for s in SUBJECTS if subj_counts[s] > 0],
        key=lambda s: subj_counts[s], reverse=True,
    )
    n_panels = max(len(draw_order), 1)
    if ax is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 4),
                                 sharey=True, squeeze=False)
        axes = tuple(axes.ravel())
    else:
        fig = ax.figure
        ss = ax.get_subplotspec()
        gs_sub = ss.subgridspec(1, n_panels, wspace=0.08)
        ax.remove()
        axes = []
        for i in range(n_panels):
            share = axes[0] if axes else None
            axes.append(fig.add_subplot(gs_sub[i], sharey=share))
        axes = tuple(axes)

    xlim = (-0.1, 0.3)
    ylim = (-0.2, 0.3)
    lo = max(xlim[0], ylim[0])
    hi = min(xlim[1], ylim[1])

    for subj, sub_ax in zip(draw_order, axes):
        mask = pair_labels == subj
        color = SUBJECT_COLORS[subj]
        # Per-subject alpha so the smaller cloud reads as dense as the larger.
        alpha = min(0.25, 1500 / max(subj_counts[subj], 1))
        sub_ax.axhline(0, color="black", linewidth=0.6, alpha=0.6, zorder=1)
        sub_ax.axvline(0, color="black", linewidth=0.6, alpha=0.6, zorder=1)
        sub_ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=0.6,
                    zorder=1)
        sub_ax.scatter(s0["rho_u"][mask], s0["rho_c"][mask],
                       s=1, alpha=alpha, c=color, rasterized=True, zorder=2)
        sub_ax.plot(np.mean(s0["rho_u"][mask]), np.mean(s0["rho_c"][mask]),
                    "o", color=color, markersize=6, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=10)
        sub_ax.set_xlim(*xlim)
        sub_ax.set_ylim(*ylim)
        sub_ax.set_xlabel("ρ uncorrected")
        sub_ax.set_title(SUBJECT_DISPLAY_NAMES.get(subj, f"Monkey {subj[0]}"),
                         fontsize=9)
        sub_ax.spines["top"].set_visible(False)
        sub_ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("ρ FEM-corrected")
    for sub_ax in axes[1:]:
        plt.setp(sub_ax.get_yticklabels(), visible=False)
    return fig, axes[0]


if __name__ == "__main__":
    fig, ax = plot_panel_b()
    fig.tight_layout()
    standalone_save(fig, "panel_b_noisecorr_scatter")
