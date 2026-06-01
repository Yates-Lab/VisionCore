"""Figure 4 panel J: ablation cost (Δr^2) vs FEM modulation (1 - α).

The cost of removing the extraretinal behavior input does not grow with how
FEM-driven a cell is — a flat, near-zero trend (and a non-significant Spearman)
shows even the most eye-movement-dominated cells lose nothing.

Usage:
    uv run ryan/fig4/generate_fig4j.py [--cond zeroed|permuted]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from _fig4_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS, configure_matplotlib, annotate_corr,
)
from _fig4_ablation_data import load_ablation_data, COND_LABEL


def plot_panel_j(ax=None, data=None, cond="zeroed", legend_fontsize=8,
                 print_stats=True, n_bins=6):
    """Draw the Δr^2-vs-(1-α) scatter with a binned-median trend. Returns (fig, ax)."""
    if data is None:
        data = load_ablation_data()
    alpha = data["alpha"]
    x = data["ve"]["intact"]
    y = data["ve"][cond]
    subjects = data["subjects"]
    good = data["good"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    base = good & np.isfinite(alpha) & np.isfinite(x) & np.isfinite(y)
    for subj in SUBJECTS:
        m = base & (subjects == subj)
        if m.any():
            ax.scatter(1 - alpha[m], x[m] - y[m], s=5, alpha=0.4,
                       color=SUBJECT_COLORS[subj])

    median_handle = None

    fem = 1 - alpha[base]
    cost = x[base] - y[base]
    if fem.size > n_bins:
        edges = np.quantile(fem, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        idx = np.digitize(fem, edges) - 1
        bx, bm = [], []
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() >= 5:
                bx.append(np.median(fem[sel]))
                bm.append(np.median(cost[sel]))
        if bx:
            (median_handle,) = ax.plot(bx, bm, "-o", color="k", lw=1.2, ms=4,
                                       zorder=5, label="binned median")
        r_s, p_s = sp_stats.spearmanr(fem, cost)
        annotate_corr(ax, r_s, p_s, loc="upper left")
        if print_stats:
            print(f"Panel J ({cond}) — N={fem.size}: Spearman r={r_s:+.3f}, "
                  f"p={p_s:.3g}; median Δr²={np.median(cost):+.4f}")

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("FEM modulation (1 - α)")
    ax.set_ylabel(f"Δ$r^2$ (intact − {COND_LABEL[cond]})")
    if median_handle is not None:
        ax.legend([median_handle], ["binned median"], frameon=False,
                  fontsize=legend_fontsize, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return fig, ax


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figure 4 panel J.")
    p.add_argument("--cond", default="zeroed", choices=["zeroed", "permuted"])
    args = p.parse_args()

    configure_matplotlib()
    fig, ax = plot_panel_j(cond=args.cond)
    fig.tight_layout()
    out = FIG_DIR / f"panel_j_cost_vs_fem_{args.cond}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
