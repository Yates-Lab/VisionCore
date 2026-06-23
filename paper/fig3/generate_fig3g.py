"""Figure 3 panel G: r^2 improvement (model / PSTH) vs FEM modulation (1 - α).

Usage:
    uv run declan/fig3/generate_fig3g.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from _fig3_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS,
    configure_matplotlib, load_fig3_data,
)


def plot_panel_g(ax=None, data=None, legend_fontsize=8, print_stats=True):
    """Draw the improvement-vs-FEM scatter on `ax`. Returns (fig, ax)."""
    if data is None:
        data = load_fig3_data()
    ve_model = data["ve_model"]
    ve_psth = data["ve_psth"]
    alpha = data["alpha"]
    subjects = data["subjects"]
    good = data["good"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
    else:
        fig = ax.figure

    has_alpha = good & np.isfinite(alpha) & (ve_psth > 0)

    for subj in SUBJECTS:
        mask = has_alpha & (subjects == subj)
        if not mask.any():
            continue
        fem_mod = 1 - alpha[mask]
        improvement = ve_model[mask] / ve_psth[mask]
        ax.scatter(fem_mod, improvement, s=5, alpha=0.5,
                   color=SUBJECT_COLORS[subj], label=subj)

    ax.axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("FEM modulation (1 - α)")
    ax.set_ylabel("$r^2$ improvement (Model / PSTH)")
    ax.set_ylim(0, 5)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if print_stats:
        for subj in SUBJECTS + ["All"]:
            mask = has_alpha.copy()
            if subj != "All":
                mask = mask & (subjects == subj)
            fem_mod = 1 - alpha[mask]
            improvement = ve_model[mask] / ve_psth[mask]
            ok = np.isfinite(fem_mod) & np.isfinite(improvement)
            r_s, p_s = sp_stats.spearmanr(fem_mod[ok], improvement[ok])
            print(f"Panel G — {subj} (N={ok.sum()}): "
                  f"Spearman r={r_s:.3f}, p={p_s:.3g}")

    return fig, ax


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax = plot_panel_g()
    fig.tight_layout()
    out = FIG_DIR / "panel_g_improvement_vs_fem.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
