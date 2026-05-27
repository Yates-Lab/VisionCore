"""
Figure 3 panel E: PSTH and FEM eigenspectra (median + IQR per subject),
log y-scale.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_e(ax=None, refresh=False, data=None, max_dims=10):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    sub_subjects = data["sub_subjects"]
    spectra_psth = data["spectra_psth"]
    spectra_fem = data["spectra_fem"]

    for subj in SUBJECTS:
        s_mask = np.array(sub_subjects) == subj
        if not s_mask.any():
            continue
        color = SUBJECT_COLORS[subj]
        for spec_list, ls, label_type in [(spectra_psth, "-", "PSTH"),
                                          (spectra_fem, "--", "FEM")]:
            spec_sub = [s for s, m in zip(spec_list, s_mask) if m]
            if not spec_sub:
                continue
            all_spec = np.full((len(spec_sub), max_dims), np.nan)
            for i, s in enumerate(spec_sub):
                L = min(len(s), max_dims)
                all_spec[i, :L] = s[:L]
            median = np.nanmedian(all_spec, axis=0)
            q25 = np.nanpercentile(all_spec, 25, axis=0)
            q75 = np.nanpercentile(all_spec, 75, axis=0)
            dims = np.arange(1, max_dims + 1)
            ax.plot(dims, median, color=color, ls=ls,
                    label=f"{subj} {label_type}", marker="o", markersize=4)
            ax.fill_between(dims, q25, q75, color=color, alpha=0.15)
    ax.set_xlim(1, max_dims)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Frac. total variance")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_e()
    fig.tight_layout()
    standalone_save(fig, "panel_e_eigenspectra")
