r"""Supplement: how each twin condition recapitulates the fig2 covariance results.

Four rows, five columns. The top row is the empirical result (real spikes through
the fig2 estimator); the next three rows are the three within-model conditions,
each realized as Poisson(twin rate) and pushed through the IDENTICAL fig2
estimator on the same intersection population (see _supp_data.py):

  Full        (intact)     : full retinal stimulus + full behavior.
  Ablated     (zeroed)     : behavior set to 0 (extraretinal route removed).
  Stabilized  (stabilized) : retinal image frozen at one session-global centroid
                             gaze (reafferent route removed), behavior intact.

Columns:
  1-alpha            : per-cell FEM modulation. Row 0 shows the neuron distribution;
                       model rows overlay the twin (raw rates) for that condition on
                       the neuron reference. Units with 1-alpha outside [0,1] are
                       excluded, matching fig2.
  Fano factor        : population slope-through-origin (uncorrected -> FEM-corrected).
  Noise correlation  : per-pair rho (uncorrected -> FEM-corrected).
  Participation ratio: residual / stimulus / FEM.
  Subspace alignment : observed vs eye-shuffle null.

If the twin's FEM-driven rate modulation is the source of fig2's population
structure, then Poisson(full-twin rate) -- Fano=1, noise-corr=0 by construction --
should reproduce the empirical effect after the FEM correction, and the two
ablations should show which route (reafferent vs extraretinal) carries it.

Usage:
    uv run python paper/supp_model_replication/generate_supp_model_replication.py
        [--n-shuffles N] [--seed S] [--refresh]
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr

from VisionCore.paths import VISIONCORE_ROOT

# fig2 panel functions + shared supplement data layer
sys.path.insert(0, str(VISIONCORE_ROOT / "paper"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig2"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_model_replication"))

from _supp_data import (  # noqa: E402
    compute_supp_bundle, compute_panel_c_data, FIG_DIR, configure_matplotlib,
)
from generate_panel_fano import plot_fano_population  # noqa: E402
from generate_panel_noisecorr import plot_nc_violin  # noqa: E402
from generate_figure2 import (  # noqa: E402
    _plot_pr_comparison, _plot_subspace_alignment_vs_shuffle,
)

WINDOW_MS = 25.0

# Condition colors (match fig3): full = blue, ablated = red, stabilized = purple.
INTACT_COLOR = "#1f77b4"
ABLATED_COLOR = "#d62728"
STABILIZED_COLOR = "#9467bd"
EMPIRICAL_COLOR = "0.25"

# (row label, condition key or None for empirical, twin color)
ROWS = [
    ("Empirical\n(real spikes)", None, EMPIRICAL_COLOR),
    ("Full twin\n(intact)", "intact", INTACT_COLOR),
    (u"Ablated\n(behavior → 0)", "zeroed", ABLATED_COLOR),
    ("Stabilized\n(retina frozen)", "stabilized", STABILIZED_COLOR),
]

COL_TITLES = [
    r"FEM modulation ($1-\alpha$)", "Fano factor", "Noise correlation",
    "Participation ratio", "Subspace alignment",
]


def _in01(v):
    """fig2's 1-alpha inclusion: finite and within [0, 1]. Uses the UNCLIPPED
    1-alpha, so out-of-range cells are excluded (not piled onto 0 / 1)."""
    v = np.asarray(v, float)
    return np.isfinite(v) & (v >= 0.0) & (v <= 1.0)


def _plot_alpha_hist_all(ax, neuron, models):
    """Top-row 1-alpha overlay: empirical neuron distribution as grey bars plus
    every twin condition as a colored step histogram. `models` is a list of
    (label, values, color). All series use the unclipped 1-alpha restricted to
    [0, 1] (fig2's exclusion), which removes the clip pile-up at 0."""
    bins = np.linspace(0, 1, 26)
    n = np.asarray(neuron, float)[_in01(neuron)]
    ax.hist(n, bins=bins, color="0.6", alpha=0.55, density=True,
            edgecolor="white", linewidth=0.3, zorder=1,
            label=f"Neuron (n={n.size})")
    ax.axvline(np.median(n), color="0.3", ls="--", lw=1.0, zorder=2)
    for label, vals, color in models:
        v = np.asarray(vals, float)[_in01(vals)]
        ax.hist(v, bins=bins, histtype="step", color=color, lw=1.7,
                density=True, zorder=3,
                label=f"{label} (med {np.median(v):.2f})")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$1-\alpha$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_alpha_scatter(ax, neuron, model, color):
    """Per-cell twin (y) vs neuron (x) 1-alpha with the identity line, an OLS
    regression line, and binned twin means +/- SEM over the neuron axis. Both
    axes use the unclipped 1-alpha restricted to [0, 1] (matched per cell)."""
    x = np.asarray(neuron, float)
    y = np.asarray(model, float)
    ok = _in01(x) & _in01(y)
    x, y = x[ok], y[ok]

    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.6, zorder=0)
    ax.scatter(x, y, s=6, alpha=0.3, color=color, linewidths=0, zorder=1)

    # OLS regression (fig3 panel-E grammar).
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, b * xs + a, color=color, lw=1.8, zorder=4,
            label=f"OLS (slope {b:.2f})")

    # Binned twin mean +/- SEM across the neuron axis.
    edges = np.linspace(0, 1, 9)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    bx, by, be = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() >= 5:
            bx.append(0.5 * (edges[k] + edges[k + 1]))
            by.append(float(np.mean(y[m])))
            be.append(float(np.std(y[m]) / np.sqrt(m.sum())))
    ax.errorbar(bx, by, yerr=be, fmt="o", color="k", ms=3.5, lw=1.0,
                capsize=2, zorder=5, mfc="white", mec="k",
                label="binned mean ± SEM")

    rho = spearmanr(x, y).correlation
    r = pearsonr(x, y)[0]
    ax.text(0.04, 0.96, rf"$\rho$={rho:.2f}  r={r:.2f}  n={x.size}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"Neuron $1-\alpha$")
    ax.set_ylabel(r"Twin $1-\alpha$")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(n_shuffles=200, seed=100, refresh=False):
    # Row bundles: empirical (real spikes, condition-independent) + one
    # Poisson(twin rate) bundle per within-model condition.
    emp = compute_supp_bundle("empirical", n_shuffles=n_shuffles, seed=seed,
                              refresh=refresh)
    poi = {
        cond: compute_supp_bundle("poisson", condition=cond,
                                  n_shuffles=n_shuffles, seed=seed, refresh=refresh)
        for cond in ("intact", "zeroed", "stabilized")
    }
    row_bundle = {None: emp, **poi}

    # Per-cell 1-alpha (raw twin rates) per condition; B_obs (neuron) is shared.
    # Use the UNCLIPPED arrays + a [0,1] exclusion (fig2's convention).
    panel_c = {cond: compute_panel_c_data(condition=cond, refresh=refresh)
               for cond in ("intact", "zeroed", "stabilized")}
    top_models = [
        ("Full", panel_c["intact"]["B_model_uncl"], INTACT_COLOR),
        ("Ablated", panel_c["zeroed"]["B_model_uncl"], ABLATED_COLOR),
        ("Stabilized", panel_c["stabilized"]["B_model_uncl"], STABILIZED_COLOR),
    ]

    fig = plt.figure(figsize=(19.0, 15.0))
    gs = GridSpec(4, 5, figure=fig, hspace=0.5, wspace=0.42,
                  left=0.075, right=0.985, top=0.92, bottom=0.055)

    top_axes = []          # top-row axes per column (for column headers)
    row_left = []          # (col-0 axis, color) per row (for row labels)
    for ri, (label, cond, color) in enumerate(ROWS):
        data = row_bundle[cond]

        ax_a = fig.add_subplot(gs[ri, 0])
        if cond is None:
            # Top row: neuron distribution + all three twin conditions overlaid.
            _plot_alpha_hist_all(ax_a, panel_c["intact"]["B_obs_uncl"], top_models)
        else:
            # Model row: this condition's twin vs the neurons, matched per cell.
            _plot_alpha_scatter(ax_a, panel_c[cond]["B_obs_uncl"],
                                panel_c[cond]["B_model_uncl"], color)

        ax_f = fig.add_subplot(gs[ri, 1])
        plot_fano_population(ax=ax_f, data=data, window_ms=WINDOW_MS)
        ax_n = fig.add_subplot(gs[ri, 2])
        plot_nc_violin(ax=ax_n, data=data, window_ms=WINDOW_MS)
        ax_g = _plot_pr_comparison(fig, gs[ri, 3], data)
        ax_i = _plot_subspace_alignment_vs_shuffle(fig, gs[ri, 4], data)

        if ri == 0:
            top_axes = [ax_a, ax_f, ax_n, ax_g, ax_i]
        row_left.append((ax_a, color))

    # Column headers (centered over each top-row axis) and colored row labels.
    fig.canvas.draw()
    for ax, title in zip(top_axes, COL_TITLES):
        pos = ax.get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.012, title,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    for (ax, color), (label, _, _) in zip(row_left, ROWS):
        pos = ax.get_position()
        fig.text(0.012, 0.5 * (pos.y0 + pos.y1), label, ha="left", va="center",
                 fontsize=10.5, fontweight="bold", rotation=90, color=color)

    fig.suptitle(
        f"Digital-twin conditions vs fig2 covariance results "
        f"(25 ms window, {n_shuffles} shuffles)\n"
        "empirical vs Poisson(twin rate) per within-model condition, "
        "identical estimator + intersection population",
        fontsize=12.5, y=0.975)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"supp_model_replication.{ext}"
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")

    # Numeric summary: Fano / noise-corr per row at the subspace window.
    print("\n=== @25ms Fano (unc->cor) | NC rho (unc->cor) ===")
    for label, cond, _ in ROWS:
        data = row_bundle[cond]
        w = data["WINDOWS_MS"][data["SUBSPACE_WINDOW_IDX"]]
        fs, nc = data["fano_stats"][w], data["nc_stats"][w]
        tag = "empirical" if cond is None else cond
        print(f"[{tag:<10}] Fano: {fs['slope_unc']:.3f} -> {fs['slope_cor']:.3f} "
              f"| NC: {np.tanh(nc['z_u_mean']):.3f} -> {np.tanh(nc['z_c_mean']):.3f}")


if __name__ == "__main__":
    configure_matplotlib()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    make_figure(n_shuffles=args.n_shuffles, seed=args.seed, refresh=args.refresh)
