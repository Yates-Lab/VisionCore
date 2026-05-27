"""Figure 4 panel D: histogram of normalized correlation (ccnorm) per subject.

Usage:
    uv run ryan/fig4/generate_fig4d.py
"""
import numpy as np
import matplotlib.pyplot as plt

from _fig4_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS,
    configure_matplotlib, load_fig4_data,
)


def plot_panel_d(ax=None, data=None, legend_fontsize=8):
    """Draw the ccnorm histogram on `ax`. Returns (fig, ax)."""
    if data is None:
        data = load_fig4_data()
    ccnorm = data["ccnorm"]
    subjects = data["subjects"]
    good = data["good"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
    else:
        fig = ax.figure

    bins = np.linspace(0, 1, 21)
    for subj in SUBJECTS:
        mask = (subjects == subj) & good & np.isfinite(ccnorm)
        if not mask.any():
            continue
        vals = ccnorm[mask]
        color = SUBJECT_COLORS[subj]
        med = np.nanmedian(vals)
        q25, q75 = np.nanpercentile(vals, [25, 75])
        ax.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.5)
        ax.axvline(med, color=color, linewidth=2, ls=(0, (1, 1)),
                   label=f"{subj}: {med:.2f} [{q25:.2f}, {q75:.2f}]")
        print(f"Panel D — {subj} (N={mask.sum()}): "
              f"median ccnorm={med:.2f}, IQR=[{q25:.2f}, {q75:.2f}]")

    ax.set_xlabel("Normalized correlation (ccnorm)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax = plot_panel_d()
    fig.tight_layout()
    out = FIG_DIR / "panel_d_ccnorm_hist.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
