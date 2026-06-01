"""Figure 4 panel C: histogram of normalized correlation (ccnorm) per subject.

Usage:
    uv run ryan/fig4/generate_fig4c.py
"""
import numpy as np
import matplotlib.pyplot as plt

from _fig4_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS,
    configure_matplotlib, load_fig4_data,
)


def plot_panel_c(ax=None, data=None):
    """Draw the ccnorm histogram on `ax`. Returns (fig, ax).

    Styled to match figure 2 panel C: per-subject stacked histogram with
    median triangle markers above the axes, no legend.
    """
    if data is None:
        data = load_fig4_data()
    ccnorm = data["ccnorm"]
    subjects = data["subjects"]
    good = data["good"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
    else:
        fig = ax.figure

    finite = good & np.isfinite(ccnorm)
    bins = np.linspace(0, 1, 21)

    present = [s for s in reversed(SUBJECTS) if ((subjects == s) & finite).any()]
    subj_vals = [ccnorm[(subjects == s) & finite] for s in present]
    subj_colors = [SUBJECT_COLORS[s] for s in present]
    ax.hist(subj_vals, bins=bins, color=subj_colors, edgecolor="white",
            stacked=True, alpha=0.75)

    y_marker = ax.get_ylim()[1] * 1.02
    for subj, vals, color in zip(present, subj_vals, subj_colors):
        med = np.nanmedian(vals)
        q25, q75 = np.nanpercentile(vals, [25, 75])
        ax.plot(med, y_marker, marker="v", color=color, markersize=10,
                clip_on=False)
        print(f"Panel C — {subj} (N={vals.size}): "
              f"median ccnorm={med:.2f}, IQR=[{q25:.2f}, {q75:.2f}]")

    ax.set_xlabel("Normalized correlation (ccnorm)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, zorder=-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax = plot_panel_c()
    fig.tight_layout()
    out = FIG_DIR / "panel_c_ccnorm_hist.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
