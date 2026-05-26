"""
Figure 2 panel E: population Fano factor vs counting window, per subject,
both uncorrected and FEM-corrected.
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_e(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    WINDOWS_MS = data["WINDOWS_MS"]
    metrics = data["metrics"]

    for subj in SUBJECTS:
        slopes_unc_sub, slopes_cor_sub = [], []
        for m_dict in metrics:
            n_mask = m_dict["subject_per_neuron"] == subj
            e_sub = m_dict["erate"][n_mask]
            ff_u_sub = m_dict["uncorr"][n_mask]
            ff_c_sub = m_dict["corr"][n_mask]
            ok = (np.isfinite(ff_u_sub) & np.isfinite(ff_c_sub)
                  & (ff_u_sub > 0) & (ff_c_sub > 0) & (e_sub > 0))
            e_v = e_sub[ok]
            vu = ff_u_sub[ok] * e_v
            vc = ff_c_sub[ok] * e_v
            if len(e_v) > 0:
                slopes_unc_sub.append(float(np.sum(e_v * vu) / np.sum(e_v ** 2)))
                slopes_cor_sub.append(float(np.sum(e_v * vc) / np.sum(e_v ** 2)))
            else:
                slopes_unc_sub.append(np.nan)
                slopes_cor_sub.append(np.nan)
        if not any(np.isfinite(slopes_unc_sub)):
            continue
        color = SUBJECT_COLORS[subj]
        ax.plot(WINDOWS_MS, slopes_unc_sub, "o-", color=color,
                label=f"{subj} Uncorr")
        ax.plot(WINDOWS_MS, slopes_cor_sub, "o--", color=color,
                label=f"{subj} Corr")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Poisson")
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Population Fano factor")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_e()
    fig.tight_layout()
    standalone_save(fig, "panel_e_fano_vs_window")
