"""
Make a comparison sheet for possible replacements of Figure 2 panel G.

The options all use the current combined-figure data policy: Luke omitted and
the remaining subjects pooled for plotting. This script does not modify the
main figure; it writes a side-by-side mockup sheet under outputs/figures/fig2.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _panel_common import FIG_DIR
from compute_fig2_data import load_fig2_data
from generate_fig3g import EDGE_SIG, SIG_ALPHA, _emp_p_greater, plot_panel_g
from generate_figure2_3_combined import (
    POOLED_COLOR,
    _filter_subjects,
    _pool_subjects_for_plotting,
)


METRIC_LABELS = [
    "Stimulus in\nFEM subspace",
    "FEM in\nstimulus subspace",
]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax, letter, title):
    ax.set_title("", loc="left")
    ax.text(-0.10, 1.07, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=12, ha="left", va="bottom")
    ax.text(0.02, 1.07, title, transform=ax.transAxes,
            fontsize=9, ha="left", va="bottom")


def _session_null_summaries(data):
    x_obs = np.asarray(data["var_p_given_f"], dtype=float)
    y_obs = np.asarray(data["var_f_given_p"], dtype=float)
    null_idx = np.asarray(data.get("null_session_idx", []), dtype=int)
    null_x = np.asarray(data.get("null_var_p_given_f", []), dtype=float)
    null_y = np.asarray(data.get("null_var_f_given_p", []), dtype=float)

    n_sessions = x_obs.size
    mean_x = np.full(n_sessions, np.nan)
    mean_y = np.full(n_sessions, np.nan)
    lo_x = np.full(n_sessions, np.nan)
    lo_y = np.full(n_sessions, np.nan)
    hi_x = np.full(n_sessions, np.nan)
    hi_y = np.full(n_sessions, np.nan)
    px = np.full(n_sessions, np.nan)
    py = np.full(n_sessions, np.nan)

    for i in range(n_sessions):
        m = null_idx == i
        if not np.any(m):
            continue
        nx = null_x[m]
        ny = null_y[m]
        nx = nx[np.isfinite(nx)]
        ny = ny[np.isfinite(ny)]
        if nx.size:
            mean_x[i] = np.mean(nx)
            lo_x[i], hi_x[i] = np.percentile(nx, [2.5, 97.5])
            px[i] = _emp_p_greater(nx, x_obs[i])
        if ny.size:
            mean_y[i] = np.mean(ny)
            lo_y[i], hi_y[i] = np.percentile(ny, [2.5, 97.5])
            py[i] = _emp_p_greater(ny, y_obs[i])

    observed = np.column_stack([x_obs, y_obs])
    null_mean = np.column_stack([mean_x, mean_y])
    null_lo = np.column_stack([lo_x, lo_y])
    null_hi = np.column_stack([hi_x, hi_y])
    pvals = np.column_stack([px, py])
    sig = (px < SIG_ALPHA) & (py < SIG_ALPHA)
    return observed, null_mean, null_lo, null_hi, pvals, sig, null_x, null_y


def _plot_paired_sessions(ax, observed, null_mean, sig):
    rng = np.random.default_rng(7)
    for j in range(2):
        ok = np.isfinite(observed[:, j]) & np.isfinite(null_mean[:, j])
        jitter = rng.normal(0, 0.018, ok.sum())
        x0 = np.full(ok.sum(), j - 0.13) + jitter
        x1 = np.full(ok.sum(), j + 0.13) + jitter
        for a, b, n0, obs, s in zip(x0, x1, null_mean[ok, j],
                                    observed[ok, j], sig[ok]):
            ax.plot([a, b], [n0, obs], color="0.72", lw=0.7, zorder=1)
            ax.scatter(a, n0, s=18, facecolor="white", edgecolor="0.55",
                       linewidth=0.6, zorder=2)
            ax.scatter(b, obs, s=26, color=POOLED_COLOR,
                       edgecolor=EDGE_SIG[0] if s else "black",
                       linewidth=1.1 if s else 0.45, zorder=3)
        ax.plot([j - 0.18, j - 0.08], [np.nanmedian(null_mean[:, j])] * 2,
                color="0.30", lw=1.2, zorder=4)
        ax.plot([j + 0.08, j + 0.18], [np.nanmedian(observed[:, j])] * 2,
                color=POOLED_COLOR, lw=1.8, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Variance captured")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    _despine(ax)
    handles = [
        Line2D([0], [0], marker="o", color="0.55", markerfacecolor="white",
               linestyle="none", markersize=5, label="shuffle mean"),
        Line2D([0], [0], marker="o", color=POOLED_COLOR,
               markeredgecolor="black", linestyle="none", markersize=5,
               label="observed"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")


def _plot_null_violin(ax, observed, sig, null_x, null_y):
    nulls = [
        null_x[np.isfinite(null_x)],
        null_y[np.isfinite(null_y)],
    ]
    viol = ax.violinplot(nulls, positions=[0, 1], widths=0.52,
                         showmeans=False, showmedians=False,
                         showextrema=False)
    for body in viol["bodies"]:
        body.set_facecolor("#d7e6ef")
        body.set_edgecolor("none")
        body.set_alpha(0.9)

    rng = np.random.default_rng(11)
    for j in range(2):
        ok = np.isfinite(observed[:, j])
        jitter = rng.normal(0, 0.055, ok.sum())
        edgecolors = np.where(sig[ok], EDGE_SIG[0], "black")
        linewidths = np.where(sig[ok], 1.2, 0.45)
        ax.scatter(np.full(ok.sum(), j) + jitter, observed[ok, j],
                   s=28, color=POOLED_COLOR, edgecolor=edgecolors,
                   linewidth=linewidths, zorder=3)
        ax.plot([j - 0.22, j + 0.22], [np.nanmedian(observed[:, j])] * 2,
                color=POOLED_COLOR, lw=2.0, zorder=4)
        ax.plot([j - 0.22, j + 0.22], [np.nanmedian(nulls[j])] * 2,
                color="0.35", lw=1.2, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Variance captured")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    _despine(ax)


def _plot_delta_from_shuffle(ax, observed, null_mean, sig):
    delta = observed - null_mean
    rng = np.random.default_rng(19)
    for j in range(2):
        ok = np.isfinite(delta[:, j])
        jitter = rng.normal(0, 0.055, ok.sum())
        edgecolors = np.where(sig[ok], EDGE_SIG[0], "black")
        linewidths = np.where(sig[ok], 1.2, 0.45)
        ax.scatter(np.full(ok.sum(), j) + jitter, delta[ok, j],
                   s=30, color=POOLED_COLOR, edgecolor=edgecolors,
                   linewidth=linewidths, zorder=3)
        med = np.nanmedian(delta[:, j])
        lo, hi = np.nanpercentile(delta[:, j], [25, 75])
        ax.plot([j - 0.23, j + 0.23], [med, med], color=POOLED_COLOR,
                lw=2.2, zorder=4)
        ax.vlines(j, lo, hi, color=POOLED_COLOR, lw=1.3, zorder=4)
    ax.axhline(0, color="0.55", lw=1.0, ls="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Observed - session shuffle mean")
    ymax = np.nanmax(np.abs(delta))
    ax.set_ylim(-0.08, max(0.5, ymax * 1.15))
    ax.grid(axis="y", alpha=0.25)
    _despine(ax)


def make_options(refresh=False):
    data = _pool_subjects_for_plotting(_filter_subjects(load_fig2_data(refresh)))
    observed, null_mean, _, _, _, sig, null_x, null_y = _session_null_summaries(data)

    rc = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(rc):
        fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.4),
                                constrained_layout=True)

        plot_panel_g(ax=axs[0, 0], data=data)
        axs[0, 0].set_xlabel("Stimulus variance in FEM subspace")
        axs[0, 0].set_ylabel("FEM variance in stimulus subspace")
        _panel_label(axs[0, 0], "A", "Current scatter")

        _plot_paired_sessions(axs[0, 1], observed, null_mean, sig)
        _panel_label(axs[0, 1], "B", "Paired session shift")

        _plot_null_violin(axs[1, 0], observed, sig, null_x, null_y)
        _panel_label(axs[1, 0], "C", "Observed against shuffle")

        _plot_delta_from_shuffle(axs[1, 1], observed, null_mean, sig)
        _panel_label(axs[1, 1], "D", "Alignment above chance")

    out = FIG_DIR / "panel_g_alignment_options"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.08, dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight",
                pad_inches=0.08, dpi=220)
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')}")
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    make_options()
