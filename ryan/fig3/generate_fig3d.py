"""
Figure 3 panel D: Δz (corrected - uncorrected) vs counting window per
subject, with shuffle null 95% CI bands.
"""
import numpy as np
import matplotlib.pyplot as plt

from VisionCore.stats import bootstrap_mean_ci
from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def plot_panel_d(ax=None, refresh=False, data=None):
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
    nc_stats = data["nc_stats"]

    for subj in SUBJECTS:
        dz_means_sub, dz_lo_sub, dz_hi_sub = [], [], []
        for m_dict in metrics:
            ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
            vals = m_dict["rho_delta_meanz_by_ds"][ds_mask]
            if len(vals) > 0:
                mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
                dz_means_sub.append(mn)
                dz_lo_sub.append(ci[0])
                dz_hi_sub.append(ci[1])
            else:
                dz_means_sub.append(np.nan)
                dz_lo_sub.append(np.nan)
                dz_hi_sub.append(np.nan)
        if not any(np.isfinite(dz_means_sub)):
            continue
        color = SUBJECT_COLORS[subj]
        ax.errorbar(WINDOWS_MS, dz_means_sub,
                    yerr=[np.array(dz_means_sub) - dz_lo_sub,
                          np.array(dz_hi_sub) - dz_means_sub],
                    fmt="o-", color=color, capsize=3, label=subj)
        null_lo_sub = [nc_stats[w]["null_dz_ci_by_subject"][subj][0]
                       for w in WINDOWS_MS]
        null_hi_sub = [nc_stats[w]["null_dz_ci_by_subject"][subj][1]
                       for w in WINDOWS_MS]
        ax.fill_between(WINDOWS_MS, null_lo_sub, null_hi_sub, alpha=0.15,
                        color=color, label=f"{subj} shuffle 95% CI")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Δz (corr - uncorr)")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_d()
    fig.tight_layout()
    standalone_save(fig, "panel_d_effect_size")
