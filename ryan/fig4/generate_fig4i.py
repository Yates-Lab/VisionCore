"""Figure 4 panel I: single-trial r^2, behavior-ablated vs intact twin.

Points on the unity line = removing the extraretinal behavior input does not
change the twin's single-trial prediction.

Usage:
    uv run ryan/fig4/generate_fig4i.py [--cond zeroed|permuted]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from _fig4_data import FIG_DIR, SUBJECTS, SUBJECT_COLORS, configure_matplotlib
from _fig4_ablation_data import load_ablation_data, COND_LABEL


def plot_panel_i(ax=None, data=None, cond="zeroed", legend_fontsize=8,
                 print_stats=True):
    """Draw the ablated-vs-intact r^2 scatter on `ax`. Returns (fig, ax)."""
    if data is None:
        data = load_ablation_data()
    x = data["ve"]["intact"]
    y = data["ve"][cond]
    subjects = data["subjects"]
    good = data["good"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    for subj in SUBJECTS:
        m = good & (subjects == subj) & np.isfinite(x) & np.isfinite(y)
        if m.any():
            ax.scatter(x[m], y[m], s=5, alpha=0.5,
                       color=SUBJECT_COLORS[subj], label=subj)

    m = good & np.isfinite(x) & np.isfinite(y)
    hi = max(0.4, np.nanpercentile(np.r_[x[m], y[m]], 99) * 1.1)
    lims = [0, hi]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("Single-trial $r^2$ (intact)")
    ax.set_ylabel(f"Single-trial $r^2$ ({COND_LABEL[cond]})")
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    d = (y[m] - x[m])
    ax.text(0.97, 0.06, f"median Δ$r^2$={np.median(d):+.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

    if print_stats:
        for subj in ["All"] + SUBJECTS:
            mm = good & np.isfinite(x) & np.isfinite(y)
            if subj != "All":
                mm = mm & (subjects == subj)
            d = y[mm] - x[mm]
            if len(d) < 5:
                print(f"Panel I ({cond}) — {subj} (N={len(d)}): too few cells")
                continue
            stat, pval = wilcoxon(d)
            print(f"Panel I ({cond}) — {subj} (N={len(d)}): "
                  f"median Δr²={np.median(d):+.4f}, Wilcoxon p={pval:.3g}")

    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 4 panel I.")
    p.add_argument("--cond", default="zeroed", choices=["zeroed", "permuted"])
    args = p.parse_args()

    configure_matplotlib()
    fig, ax = plot_panel_i(cond=args.cond)
    fig.tight_layout()
    out = FIG_DIR / f"panel_i_r2_ablated_{args.cond}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
