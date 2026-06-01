"""Figure 4 panel G: r^2 improvement (model / PSTH) vs FEM modulation (1 - α).

Usage:
    uv run ryan/fig4/generate_fig4g.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from _fig4_data import (
    FIG_DIR, SUBJECTS, SUBJECT_COLORS,
    configure_matplotlib, load_fig4_data, annotate_corr,
)


def plot_panel_g(ax=None, data=None, legend_fontsize=8, print_stats=True):
    """Draw the improvement-vs-FEM scatter on `ax`. Returns (fig, ax)."""
    if data is None:
        data = load_fig4_data()
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
                   color=SUBJECT_COLORS[subj])

    # Population linear regression: more FEM-modulated cells gain more from the
    # model over the PSTH.
    fem_all = 1 - alpha[has_alpha]
    imp_all = ve_model[has_alpha] / ve_psth[has_alpha]
    ok = np.isfinite(fem_all) & np.isfinite(imp_all)
    reg = sp_stats.linregress(fem_all[ok], imp_all[ok])
    xs = np.array([fem_all[ok].min(), fem_all[ok].max()])
    ax.plot(xs, reg.slope * xs + reg.intercept, color="red", linewidth=1.0)

    rho_all, p_all = sp_stats.spearmanr(fem_all[ok], imp_all[ok])
    annotate_corr(ax, rho_all, p_all, loc="upper left")

    ax.axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("FEM modulation (1 - α)")
    ax.set_ylabel("$r^2$ improvement (Model / PSTH)")
    ax.set_ylim(0, 5)
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
            reg = sp_stats.linregress(fem_mod[ok], improvement[ok])
            print(f"Panel G — {subj} (N={ok.sum()}): "
                  f"Spearman r={r_s:.3f}, p={p_s:.3g}, "
                  f"OLS slope={reg.slope:.3f}, intercept={reg.intercept:.3f}, "
                  f"p={reg.pvalue:.3g}")

    return fig, ax


if __name__ == "__main__":
    configure_matplotlib()
    fig, ax = plot_panel_g()
    fig.tight_layout()
    out = FIG_DIR / "panel_g_improvement_vs_fem.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
