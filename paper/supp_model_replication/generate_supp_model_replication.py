r"""Supplement: how each twin condition recapitulates the fig2 covariance results.

Four rows, four columns. The top row is the empirical result (real spikes through
the fig2 estimator); the next three rows are the three within-model conditions,
each realized as Poisson(twin rate) and pushed through the IDENTICAL fig2
estimator on the same intersection population (see _supp_data.py):

  Full        (intact)     : full retinal stimulus + full behavior.
  Ablated     (zeroed)     : behavior set to 0 (extraretinal route removed).
  Stabilized  (stabilized) : retinal image frozen at one session-global centroid
                             gaze (reafferent route removed), behavior intact.

Columns:
  Fano factor        : population slope-through-origin (uncorrected -> FEM-corrected).
  Noise correlation  : per-pair rho (uncorrected -> FEM-corrected).
  Participation ratio: residual / stimulus / FEM.
  Subspace alignment : observed vs eye-shuffle null.

The per-cell FEM-modulation (1-alpha) column that used to lead this figure has
been promoted to a primary result -- figure 3 panel E (see paper/fig3/
_fig3_femfraction.py) -- and is no longer repeated here.

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

from VisionCore.paths import VISIONCORE_ROOT

# fig2 panel functions + shared supplement data layer
sys.path.insert(0, str(VISIONCORE_ROOT / "paper"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig2"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_model_replication"))

from _supp_data import (  # noqa: E402
    compute_supp_bundle, FIG_DIR, configure_matplotlib,
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
    "Fano factor", "Noise correlation",
    "Participation ratio", "Subspace alignment",
]


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

    # Four columns (the per-cell 1-alpha column that used to lead this figure is
    # now figure 3 panel E): Fano, noise correlation, participation ratio,
    # subspace alignment.
    fig = plt.figure(figsize=(15.4, 15.0))
    gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.42,
                  left=0.085, right=0.985, top=0.92, bottom=0.055)

    top_axes = []          # top-row axes per column (for column headers)
    row_left = []          # (col-0 axis, color) per row (for row labels)
    for ri, (label, cond, color) in enumerate(ROWS):
        data = row_bundle[cond]

        ax_f = fig.add_subplot(gs[ri, 0])
        plot_fano_population(ax=ax_f, data=data, window_ms=WINDOW_MS)
        ax_n = fig.add_subplot(gs[ri, 1])
        plot_nc_violin(ax=ax_n, data=data, window_ms=WINDOW_MS)
        ax_g = _plot_pr_comparison(fig, gs[ri, 2], data)
        ax_i = _plot_subspace_alignment_vs_shuffle(fig, gs[ri, 3], data)

        if ri == 0:
            top_axes = [ax_f, ax_n, ax_g, ax_i]
        row_left.append((ax_f, color))

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
