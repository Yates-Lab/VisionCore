"""Figure 4 panel F: single-trial r^2 scatter (twin vs leave-one-out PSTH).

Usage:
    uv run ryan/fig4/generate_fig4f.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from _fig4_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS,
    configure_matplotlib, load_fig4_data,
)


def plot_panel_f(ax=None, data=None, legend_fontsize=8, print_stats=True):
    """Draw the r^2 scatter on `ax`. Returns (fig, ax)."""
    if data is None:
        data = load_fig4_data()
    ve_model = data["ve_model"]
    ve_psth = data["ve_psth"]
    subjects = data["subjects"]
    good = data["good"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    for subj in SUBJECTS:
        mask = (subjects == subj) & good
        if not mask.any():
            continue
        ax.scatter(ve_psth[mask], ve_model[mask], s=5, alpha=0.5,
                   color=SUBJECT_COLORS[subj])

    lims = [0, max(0.4, np.nanmax(ve_model[good]) * 1.1)]
    ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Single-trial $r^2$ (PSTH)")
    ax.set_ylabel("Single-trial $r^2$ (Model)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if print_stats:
        for subj in SUBJECTS + ["All"]:
            mask = good.copy()
            if subj != "All":
                mask = mask & (subjects == subj)
            x = ve_model[mask]
            y = ve_psth[mask]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            d = x - y
            stat, p = wilcoxon(d, alternative='greater')
            print(f"Panel F — {subj} (N={len(d)}): "
                  f"median model r^2={np.median(x):.3f}, PSTH r^2={np.median(y):.3f}, "
                  f"Wilcoxon stat={stat:.1f}, p={p:.3g}")

    return fig, ax


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax = plot_panel_f()
    fig.tight_layout()
    out = FIG_DIR / "panel_f_r2_scatter.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
