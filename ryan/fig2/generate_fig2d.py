"""
Figure 2 panel D: Δ variance (FEM-corrected - uncorrected) vs mean rate,
scatter colored by subject (Allen + Logan only in current dataset).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_d(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure

    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    WINDOWS_MS = data["WINDOWS_MS"]
    s0 = data["fano_stats"][WINDOWS_MS[0]]
    labels = s0["subject_per_neuron"]

    for subj in ("Allen", "Logan"):
        mask = labels == subj
        if not mask.any():
            continue
        e_sub = s0["erate"][mask]
        vu_sub = s0["var_u"][mask]
        vc_sub = s0["var_c"][mask]
        color = SUBJECT_COLORS[subj]
        ax.scatter(e_sub, vc_sub - vu_sub, s=6, alpha=0.3, c=color, label=subj)

    ax.set_xlabel("Mean rate")
    ax.set_ylabel("$\\Delta$ Variance (Corr - Uncorr)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10 ** -3)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_d()
    fig.tight_layout()
    standalone_save(fig, "panel_d_delta_var")
