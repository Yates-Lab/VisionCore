"""Extended single-trial perturbation figure (fork of figure 3's panel D).

Two rows, one shared set of seven conditions:

  A  single-trial r^2      vs model perturbation
  B  single-trial bits/spike vs model perturbation

Left to right: the leave-one-out PSTH (trial average), the full twin, the
behavior-ablated twin, then the stabilization ladder from the narrowest scope to
the widest (within-window -> within-trial -> across-trial), and finally the
across-trial freeze combined with behavior ablation. The three pure
stabilizations share one colour because they are one manipulation at three
scopes; the combined freeze+ablation is set apart.

Every non-reference box is tested against the FULL twin (paired Wilcoxon over
cells) and annotated with the median difference. The PSTH median is also drawn
as a dashed reference line across each panel.

Usage:
    uv run python ryan/figure3e_extended/generate_figure3e_extended.py [--recompute]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import wilcoxon

from VisionCore.paths import FIGURES_DIR

from _ext_data import (
    BOX_ORDER, COND_LABEL, CACHE_PATH, load_extended_data,
)


OUT_DIR = FIGURES_DIR / "fig3e_extended"

PSTH_COLOR = "0.55"
FULL_COLOR = "#1f77b4"
ABLATED_COLOR = "#d62728"
STABILIZED_COLOR = "#9467bd"
STAB_ABLATED_COLOR = "#2ca02c"

COND_COLOR = {
    "psth": PSTH_COLOR,
    "full": FULL_COLOR,
    "ablated": ABLATED_COLOR,
    "stab_window": STABILIZED_COLOR,
    "stab_trial": STABILIZED_COLOR,
    "stab_global": STABILIZED_COLOR,
    "stab_global_ablated": STAB_ABLATED_COLOR,
}

PANEL_LETTER_SIZE = 10
PANEL_TITLE_SIZE = 8.0
REFERENCE = "full"   # every contrast is against the full twin


def configure_matplotlib():
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _lighten(color, frac=0.5):
    import matplotlib.colors as mcolors
    c = np.asarray(mcolors.to_rgb(color))
    return tuple(1.0 - (1.0 - c) * frac)


def _box_whisker(ax, groups, positions, colors, *, width=0.6):
    """Faint condition-tinted box (25-75 IQR) with a black outline and median,
    whiskers at the 2.5th/97.5th percentiles, fliers hidden -- matching fig3."""
    bp = ax.boxplot(groups, positions=positions, widths=width,
                    patch_artist=True, showfliers=False, whis=(2.5, 97.5),
                    medianprops=dict(lw=2.0, solid_capstyle="round"),
                    boxprops=dict(lw=1.1), whiskerprops=dict(lw=1.0),
                    capprops=dict(lw=1.0), zorder=3)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(_lighten(color))
        box.set_edgecolor("black")
        box.set_zorder(3)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_zorder(4)
    for part in bp["whiskers"] + bp["caps"]:
        part.set_color("black")
    return bp


def _panel_heading(ax, letter, title):
    ax.text(-0.055, 1.12, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", color="#202124",
            clip_on=False)
    ax.text(0.005, 1.12, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=PANEL_TITLE_SIZE, color="#202124", linespacing=1.05,
            clip_on=False)


def plot_metric(ax, data, metric, ylabel, *, show_xticklabels, label):
    """One row: seven condition boxes, PSTH median reference line, and each
    non-reference condition annotated with its paired contrast against the full
    twin."""
    pop = np.asarray(data["population"], dtype=bool)
    vals = {c: np.asarray(data[metric][c], dtype=float) for c in BOX_ORDER}

    m = pop.copy()
    for c in BOX_ORDER:
        m &= np.isfinite(vals[c])
    groups = {c: vals[c][m] for c in BOX_ORDER}
    n = int(m.sum())

    positions = np.arange(len(BOX_ORDER))
    _box_whisker(ax, [groups[c] for c in BOX_ORDER], positions,
                 [COND_COLOR[c] for c in BOX_ORDER])

    stats = boxplot_stats([groups[c] for c in BOX_ORDER], whis=(2.5, 97.5))
    whis_hi = max(s["whishi"] for s in stats)
    whis_lo = min(s["whislo"] for s in stats)
    rng = whis_hi - whis_lo

    # Headroom above the tallest whisker for the per-condition annotations.
    y_bottom = min(0.0, whis_lo - 0.06 * rng)
    y_top = whis_hi + 0.42 * rng
    ax.set_ylim(y_bottom, y_top)

    psth_med = float(np.median(groups["psth"]))
    ax.axhline(psth_med, color=PSTH_COLOR, lw=0.8, ls="--", alpha=0.85, zorder=0)
    ax.axhline(0, color="0.7", lw=0.6, ls=":", zorder=0)

    ref = groups[REFERENCE]
    ref_med = float(np.median(ref))
    y_annot = whis_hi + 0.08 * rng
    printout = {}
    for i, c in enumerate(BOX_ORDER):
        med = float(np.median(groups[c]))
        printout[c] = {"median": med, "n": n}
        if c == REFERENCE:
            continue
        d = float(np.median(groups[c] - ref))
        p = float(wilcoxon(groups[c], ref).pvalue)
        printout[c].update({"delta_vs_full": d, "p_vs_full": p})
        ax.text(i, y_annot, _stars(p), ha="center", va="bottom",
                fontsize=7.0, color="0.15", clip_on=False)
        ax.text(i, y_annot + 0.10 * rng, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=5.6, color="0.35", clip_on=False)

    ax.set_xlim(-0.65, len(BOX_ORDER) - 0.35)
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels([COND_LABEL[c] for c in BOX_ORDER], fontsize=5.4)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference-line label, parked right of the last box.
    ax.text(len(BOX_ORDER) - 0.42, psth_med, "trial avg.\nmedian",
            color="0.5", fontsize=5.2, va="center", ha="left", clip_on=False)

    print(f"\n=== {label} (N={n} cells, fig2 population) ===")
    print(f"{'condition':<22}{'median':>10}{'Δ vs full':>12}{'p (Wilcoxon)':>16}")
    for c in BOX_ORDER:
        s = printout[c]
        if c == REFERENCE:
            print(f"{c:<22}{s['median']:>+10.4f}{'--':>12}{'(reference)':>16}")
        else:
            print(f"{c:<22}{s['median']:>+10.4f}{s['delta_vs_full']:>+12.4f}"
                  f"{s['p_vs_full']:>16.2e}")
    print(f"full median = {ref_med:+.4f}; PSTH median = {psth_med:+.4f}; "
          f"full - PSTH = {float(np.median(ref - groups['psth'])):+.4f}")
    return printout


def compose(*, recompute=False, out_dir=OUT_DIR, dpi=300, cache_path=CACHE_PATH):
    configure_matplotlib()
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    })
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_extended_data(recompute=recompute, cache_path=cache_path)

    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=False)
    gs = GridSpec(2, 1, figure=fig, left=0.10, right=0.965, bottom=0.115,
                  top=0.935, hspace=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    stats_r2 = plot_metric(ax_a, data, "ve", "Single-trial $r^2$",
                           show_xticklabels=False, label="Single-trial r^2")
    stats_bps = plot_metric(ax_b, data, "bps", "Single-trial bits per spike",
                            show_xticklabels=True, label="Single-trial bits/spike")

    _panel_heading(ax_a, "A", "Single-trial variance explained")
    _panel_heading(ax_b, "B", "Single-trial information rate")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"figure3e_extended.{ext}", dpi=dpi)

    manifest = {
        "figure": "figure3e_extended",
        "cache": str(CACHE_PATH),
        "source_script": str(__file__),
        "condition_order": BOX_ORDER,
        "n_sessions": len(data["results"]),
        "n_cells_total": int(len(data["population"])),
        "n_cells_population": int(np.asarray(data["population"]).sum()),
        "stats": {"single_trial_r2": stats_r2, "bits_per_spike": stats_bps},
    }
    with open(out_dir / "figure3e_extended_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return fig, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recompute", action="store_true",
                   help="Force re-running the six-condition inference sweep.")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--cache", type=str, default=None,
                   help="Alternate cache path (e.g. a single-session smoke cache).")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    out_dir = OUT_DIR if args.out_dir is None else Path(args.out_dir)
    fig, _ = compose(recompute=args.recompute, out_dir=out_dir, dpi=args.dpi,
                     cache_path=Path(args.cache) if args.cache else CACHE_PATH)
    plt.close(fig)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
