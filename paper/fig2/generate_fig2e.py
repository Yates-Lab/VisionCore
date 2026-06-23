"""
Figure 2 panel E: population (slope-through-origin) Fano factor vs counting
window, per subject, uncorrected (dashed, open markers) and FEM-corrected
(solid, filled markers), with session-clustered bootstrap CIs. Stars below each
window give the corrected-vs-uncorrected significance per subject (blue = Allen,
green = Logan).
"""
import numpy as np
import matplotlib.pyplot as plt

from _panel_common import standalone_save, pstars
from compute_fig2_data import load_fig2_data


def _yerr(slope, ci):
    """Asymmetric yerr columns from a (lo, hi) CI; NaN where slope is missing."""
    lo = np.array([c[0] for c in ci], dtype=float)
    hi = np.array([c[1] for c in ci], dtype=float)
    return np.vstack([slope - lo, hi - slope])


def plot_panel_e(ax=None, refresh=False, data=None):
    if data is None:
        data = load_fig2_data(refresh=refresh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    SUBJECTS = data["SUBJECTS"]
    SUBJECT_COLORS = data["SUBJECT_COLORS"]
    WINDOWS_MS = np.asarray(data["WINDOWS_MS"], dtype=float)
    fano_stats = data["fano_stats"]

    series = {}  # per-subject slopes/CIs/p for plotting + star placement
    for subj in SUBJECTS:
        s_unc, s_cor, ci_unc, ci_cor, p_list = [], [], [], [], []
        for w in WINDOWS_MS:
            ps = fano_stats[w]["per_subject"].get(subj)
            if ps is None:
                s_unc.append(np.nan); s_cor.append(np.nan)
                ci_unc.append((np.nan, np.nan)); ci_cor.append((np.nan, np.nan))
                p_list.append(np.nan)
                continue
            s_unc.append(ps["slope_unc"]); s_cor.append(ps["slope_cor"])
            ci_unc.append(ps["slope_unc_ci"]); ci_cor.append(ps["slope_cor_ci"])
            p_list.append(ps["p_slope"])
        s_unc = np.array(s_unc); s_cor = np.array(s_cor)
        if not np.any(np.isfinite(s_unc)):
            continue
        color = SUBJECT_COLORS[subj]
        # Raw de-emphasized: dashed, open markers. Corrected is the hero: solid,
        # filled markers. Linestyle convention matches panel D.
        ax.errorbar(WINDOWS_MS, s_unc, yerr=_yerr(s_unc, ci_unc), fmt="o--",
                    color=color, lw=1.5, capsize=0, markerfacecolor="white")
        ax.errorbar(WINDOWS_MS, s_cor, yerr=_yerr(s_cor, ci_cor), fmt="o-",
                    color=color, lw=1.5, capsize=0)
        series[subj] = dict(ci_cor=ci_cor, p=p_list)

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)

    # Per-window stars sit just below the lower corrected point (emphasis on the
    # corrected estimate), stacked downward per subject in subject color.
    present = list(series)
    span = max(c[1] for s in present for c in series[s]["ci_cor"]
               if np.isfinite(c[1]))
    gap = 0.035 * span
    overall_bot = np.inf
    for wi, w in enumerate(WINDOWS_MS):
        base = np.nanmin([series[s]["ci_cor"][wi][0] for s in present])
        for row, subj in enumerate(present):
            y = base - gap - row * gap
            ax.text(w, y, pstars(series[subj]["p"][wi]), ha="center",
                    va="top", color=SUBJECT_COLORS[subj], fontsize=9)
        overall_bot = min(overall_bot, base - gap - len(present) * gap)

    ax.set_ylim(bottom=overall_bot - 0.04 * span)
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Variability (Fano factor)")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_e()
    fig.tight_layout()
    standalone_save(fig, "panel_e_fano_vs_window")
