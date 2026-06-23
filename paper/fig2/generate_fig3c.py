"""
Figure 3 panel C: mean Fisher z noise correlation vs counting window,
per subject, uncorrected (dashed, open markers) and FEM-corrected (solid,
filled markers). Subjects are dodged in x to disambiguate overlap.
Styled to match fig2 panel E (population Fano factor).
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
    WINDOWS_MS = np.asarray(data["WINDOWS_MS"], dtype=float)
    metrics = data["metrics"]

    present = [s for s in SUBJECTS
               if any(s in m["subject_by_ds"] for m in metrics)]
    n = max(len(present), 1)
    # Fixed additive dodge so the per-subject offset is the same at every
    # window on a linear axis (calibrated against the 25 ms spacing).
    gap_ms = 1.8
    dodge_ms = {s: (i - (n - 1) / 2) * gap_ms for i, s in enumerate(present)}

    # Lines + error bars first; markers go on top in a second pass so that
    # one subject's line never sits over another's points.
    series = {}
    for subj in present:
        x = WINDOWS_MS + dodge_ms[subj]
        series[subj] = {"x": x}
        for key in ("u", "c"):
            means, lo, hi = [], [], []
            for m_dict in metrics:
                ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
                vals = m_dict[f"rho_{key}_meanz_by_ds"][ds_mask]
                if len(vals) > 0:
                    mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
                    means.append(mn); lo.append(ci[0]); hi.append(ci[1])
                else:
                    means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            series[subj][key] = (
                np.array(means), np.array(lo), np.array(hi),
            )

    for subj in reversed(present):
        color = SUBJECT_COLORS[subj]
        x = series[subj]["x"]
        for key, ls in [("u", "--"), ("c", "-")]:
            means, lo, hi = series[subj][key]
            ax.errorbar(x, means, yerr=[means - lo, hi - means],
                        fmt="none", ecolor=color, lw=1.5, capsize=0,
                        zorder=2)
            ax.plot(x, means, ls=ls, color=color, lw=1.5, zorder=2)

    for subj in reversed(present):
        color = SUBJECT_COLORS[subj]
        x = series[subj]["x"]
        for key, mfc in [("u", "white"), ("c", color)]:
            means, _, _ = series[subj][key]
            ax.plot(x, means, marker="o", ls="none", color=color,
                    markersize=5, markerfacecolor=mfc,
                    markeredgecolor=color, zorder=5)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Counting window (ms)")
    ax.set_ylabel("Noise correlations (mean Fisher z)")
    ax.set_xticks(WINDOWS_MS)
    ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_c()
    fig.tight_layout()
    standalone_save(fig, "panel_c_noisecorr_vs_window")
