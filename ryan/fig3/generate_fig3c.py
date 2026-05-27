"""
Figure 3 panel C: mean Fisher z noise correlation vs counting window,
per subject, both uncorrected and FEM-corrected.
"""
import numpy as np
import matplotlib.pyplot as plt

from VisionCore.stats import bootstrap_mean_ci
from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_c(ax=None, refresh=False, data=None):
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
        for label, key, ls in [("Uncorr", "u", "-"), ("Corr", "c", "--")]:
            means, ci_lo_list, ci_hi_list = [], [], []
            for m_dict in metrics:
                ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
                vals = m_dict[f"rho_{key}_meanz_by_ds"][ds_mask]
                if len(vals) > 0:
                    mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
                    means.append(mn)
                    ci_lo_list.append(ci[0])
                    ci_hi_list.append(ci[1])
                else:
                    means.append(np.nan)
                    ci_lo_list.append(np.nan)
                    ci_hi_list.append(np.nan)
            if not any(np.isfinite(means)):
                continue
            color = SUBJECT_COLORS[subj]
            ax.errorbar(WINDOWS_MS, means,
                        yerr=[np.array(means) - ci_lo_list,
                              np.array(ci_hi_list) - means],
                        fmt=f"o{ls}", color=color, capsize=3,
                        label=f"{subj} {label}")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Mean Fisher z")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_c()
    fig.tight_layout()
    standalone_save(fig, "panel_c_noisecorr_vs_window")
