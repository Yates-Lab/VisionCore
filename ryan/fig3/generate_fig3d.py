"""
Figure 3 panel D: Δz (corrected - uncorrected) vs counting window per
subject, against the shuffle null 95% CI band. Per-subject empirical
significance is annotated with stars below each window; observed Δz is
shown as filled markers without error bars since the claim is whether the
effect exceeds the shuffle null, not whether it differs from zero.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from VisionCore.stats import emp_p_one_sided
from _panel_common import standalone_save
from compute_fig2_data import load_fig2_data


def pstars(p):
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def plot_panel_d(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    WINDOWS_MS = np.asarray(data["WINDOWS_MS"], dtype=float)
    metrics = data["metrics"]
    nc_stats = data["nc_stats"]

    present = [s for s in SUBJECTS
               if any(s in m["subject_by_ds"] for m in metrics)]

    series = {}
    for subj in present:
        means, p_vals = [], []
        for w_idx, m_dict in enumerate(metrics):
            ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
            vals = m_dict["rho_delta_meanz_by_ds"][ds_mask]
            if len(vals) == 0:
                means.append(np.nan); p_vals.append(np.nan); continue
            means.append(float(np.mean(vals)))
            shuff_mask = m_dict["shuff_rho_subject"] == subj
            shuff = m_dict["shuff_rho_delta_meanz"][shuff_mask]
            if shuff.size > 0:
                p_vals.append(emp_p_one_sided(shuff, means[-1],
                                              direction="less"))
            else:
                p_vals.append(np.nan)
        series[subj] = dict(means=np.array(means), p=np.array(p_vals))

    # Shuffle null 95% bands at undodged x so each band spans the full window range.
    for subj in reversed(present):
        color = SUBJECT_COLORS[subj]
        lo = np.array([nc_stats[w]["null_dz_ci_by_subject"][subj][0]
                       for w in WINDOWS_MS])
        hi = np.array([nc_stats[w]["null_dz_ci_by_subject"][subj][1]
                       for w in WINDOWS_MS])
        ax.fill_between(WINDOWS_MS, lo, hi, color=color, alpha=0.15,
                        linewidth=0, zorder=1)

    # Connecting lines first, then markers on a second pass so that no line
    # ever sits over another subject's points.
    for subj in reversed(present):
        ax.plot(WINDOWS_MS, series[subj]["means"], "-",
                color=SUBJECT_COLORS[subj], lw=1.5, zorder=2)
    for subj in reversed(present):
        ax.plot(WINDOWS_MS, series[subj]["means"], "o", ls="none",
                color=SUBJECT_COLORS[subj], markersize=5, zorder=5)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.grid(True, alpha=0.3, zorder=0)

    # Stacked per-subject stars below the lower null edge: Allen (blue) above
    # Logan (green). Only annotated when significant; nothing drawn for n.s.
    all_lo = [nc_stats[w]["null_dz_ci_by_subject"][s][0]
              for w in WINDOWS_MS for s in present]
    all_hi = [nc_stats[w]["null_dz_ci_by_subject"][s][1]
              for w in WINDOWS_MS for s in present]
    all_means = np.concatenate([series[s]["means"] for s in present])
    span = float(np.nanmax([np.nanmax(all_hi), np.nanmax(all_means)])
                 - np.nanmin([np.nanmin(all_lo), np.nanmin(all_means)]))
    gap = 0.05 * span
    # Anchor stars under the lowest observed point at each window, not under
    # the (much wider) null CI band.
    bottoms = [np.nanmin([series[s]["means"][wi] for s in present])
               for wi, w in enumerate(WINDOWS_MS)]
    star_floor = np.inf
    for wi, w in enumerate(WINDOWS_MS):
        base = bottoms[wi]
        for row, subj in enumerate(present):
            p = series[subj]["p"][wi]
            if not (np.isfinite(p) and p < 5e-2):
                continue
            y = base - gap - row * gap
            ax.text(w, y, pstars(p), ha="center", va="top",
                    color=SUBJECT_COLORS[subj], fontsize=10,
                    fontweight="bold")
            star_floor = min(star_floor, y)

    ax.set_ylim(-0.15, 0.15)

    null_handle = Patch(facecolor="gray", alpha=0.3, edgecolor="none",
                        label="Shuffle null 95% CI")
    ax.legend(handles=[null_handle], frameon=False, fontsize=7,
              loc="upper right")
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Δz (corr − uncorr)")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_d()
    fig.tight_layout()
    standalone_save(fig, "panel_d_effect_size")
