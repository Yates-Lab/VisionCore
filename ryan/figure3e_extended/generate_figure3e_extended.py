"""Extended single-trial perturbation figure (fork of figure 3's panel D).

Four rows, one shared set of seven conditions:

  A  single-trial r^2                   vs model perturbation
  B  single-trial bits/spike            vs model perturbation
  C  residual rate fraction             (model vs model; no spikes involved)
  D  Poisson self-consistency fraction  observed likelihood gain over the gain
                                        expected under each condition's own
                                        predicted rates

Rows A, B and D are predictive and share the same paired contrasts against the
full twin; row C is within-model. A and B are referenced to the PSTH median, D
to its own expected value of 1.

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
    BOX_ORDER, COND_LABEL, CACHE_PATH, RESID_CONDS, load_extended_data,
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


def plot_metric(ax, data, metric, ylabel, *, show_xticklabels, label,
                ref_value=None, ref_label="trial avg.\nmedian",
                ref_color=PSTH_COLOR, ref_label_dx=0.42):
    """One row: seven condition boxes, a horizontal reference line, and each
    non-reference condition annotated with its paired contrast against the full
    twin.

    `ref_value=None` draws the PSTH median -- the empirical benchmark for the
    predictive rows. A metric with its own natural reference (the
    self-consistency fraction, whose expectation is 1) passes that value
    instead."""
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

    psth_med = float(np.median(groups["psth"]))
    ref_y = psth_med if ref_value is None else float(ref_value)

    # Headroom above the tallest whisker for the per-condition annotations, and
    # enough room that the reference line is never pushed off the axis.
    y_bottom = min(0.0, whis_lo - 0.06 * rng, ref_y - 0.06 * rng)
    y_top = max(whis_hi + 0.42 * rng, ref_y + 0.06 * rng)
    ax.set_ylim(y_bottom, y_top)

    ax.axhline(ref_y, color=ref_color, lw=0.8, ls="--", alpha=0.85, zorder=0)
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

    # Reference-line label, parked in the gutter right of the last box.
    # `ref_label_dx` widens that gutter for labels too long for the figure margin.
    ax.text(len(BOX_ORDER) - ref_label_dx, ref_y, ref_label,
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


def plot_residual_fraction(ax, data, *, show_xticklabels, label):
    """Third row: how much of the twin's own predicted rate modulation each
    perturbation moves, as var(full - perturbed) / var(full) per unit.

    Purely within-model -- the observed spikes play no part -- so it reads as
    "what fraction of the twin's rate variance depends on this input". `full` is
    the reference (identically 0) and the PSTH is not a model, so neither gets a
    box; their slots are kept so the x-axis stays aligned with the rows above.

    Note a residual fraction can exceed 1: the perturbed twin is not a shrunken
    version of the full twin, so the difference can carry more variance than the
    reference itself."""
    pop = np.asarray(data["population"], dtype=bool)
    vals = {c: np.asarray(data["resid_frac"][c], dtype=float) for c in RESID_CONDS}

    m = pop.copy()
    for c in RESID_CONDS:
        m &= np.isfinite(vals[c])
    groups = {c: vals[c][m] for c in RESID_CONDS}
    n = int(m.sum())

    positions = [BOX_ORDER.index(c) for c in RESID_CONDS]
    _box_whisker(ax, [groups[c] for c in RESID_CONDS], positions,
                 [COND_COLOR[c] for c in RESID_CONDS])

    stats = boxplot_stats([groups[c] for c in RESID_CONDS], whis=(2.5, 97.5))
    whis_hi = max(s["whishi"] for s in stats)
    # Non-negative quantity: the axis is floored at 0 rather than keyed to the
    # lowest whisker.
    ax.set_ylim(0, whis_hi * 1.24)

    # The two non-model slots, marked rather than left ambiguously blank.
    ax.plot([BOX_ORDER.index("full")], [0], marker="_", ms=11, color=FULL_COLOR,
            mew=2.0, zorder=4, clip_on=False)
    ax.text(BOX_ORDER.index("full"), whis_hi * 0.045, "0 by\ndefinition",
            ha="center", va="bottom", fontsize=4.9, color="0.45",
            linespacing=1.15)
    ax.text(BOX_ORDER.index("psth"), whis_hi * 0.045, "n/a\n(not a model)",
            ha="center", va="bottom", fontsize=4.9, color="0.55",
            linespacing=1.15)
    ax.axhline(1.0, color="0.7", lw=0.6, ls=":", zorder=0)

    # Median printed just above each box's own whisker rather than on one shared
    # row: these conditions span 0.05 to ~1.0, so a common annotation height
    # would strand the labels far from the small boxes.
    printout = {}
    for c, pos, s in zip(RESID_CONDS, positions, stats):
        med = float(np.median(groups[c]))
        printout[c] = {"median": med, "n": n}
        ax.text(pos, s["whishi"] + 0.02 * whis_hi, f"{med:.2f}", ha="center",
                va="bottom", fontsize=6.2, color="0.25", clip_on=False)

    ax.set_xlim(-0.65, len(BOX_ORDER) - 0.35)
    ax.set_xticks(np.arange(len(BOX_ORDER)))
    if show_xticklabels:
        ax.set_xticklabels([COND_LABEL[c] for c in BOX_ORDER], fontsize=5.4)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("Fraction of twin's rate\nvariance in residual")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"\n=== {label} (N={n} cells, fig2 population) ===")
    print(f"{'condition':<22}{'median':>10}{'IQR':>22}")
    for c in RESID_CONDS:
        q1, q3 = np.percentile(groups[c], [25, 75])
        print(f"{c:<22}{printout[c]['median']:>10.3f}   [{q1:.3f}, {q3:.3f}]")
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
    if "selfcons" not in data:
        raise RuntimeError(
            f"{cache_path} predates the Poisson self-consistency row, and the "
            "cache stores per-neuron summaries rather than the per-bin rate "
            "traces the metric needs, so it cannot be backfilled. Re-run the "
            "sweep:\n    uv run python "
            "ryan/figure3e_extended/generate_figure3e_extended.py --recompute")

    fig = plt.figure(figsize=(7.2, 14.4), constrained_layout=False)
    gs = GridSpec(5, 1, figure=fig, left=0.105, right=0.965, bottom=0.055,
                  top=0.972, hspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[3, 0])
    ax_e = fig.add_subplot(gs[4, 0])

    stats_r2 = plot_metric(ax_a, data, "ve", "Single-trial $r^2$",
                           show_xticklabels=False, label="Single-trial r^2")
    # Reference 1, not the PSTH median: the denominator is fig2's measured
    # explainable variance, so 1 is where a model that had captured all of the
    # stimulus- and gaze-conditional rate modulation would sit.
    stats_norm = plot_metric(
        ax_b, data, "r2_norm",
        "Fraction of explainable\nvariance ($r^2 / R^2_{max}$)",
        show_xticklabels=False, label="Normalized single-trial r^2",
        ref_value=1.0, ref_label="ceiling", ref_color="0.35",
        ref_label_dx=0.42)
    stats_bps = plot_metric(ax_c, data, "bps", "Single-trial bits per spike",
                            show_xticklabels=False, label="Single-trial bits/spike")
    stats_resid = plot_residual_fraction(
        ax_d, data, show_xticklabels=False,
        label="Residual rate variance (model vs model)")
    # Reference 1, not the PSTH median: this row asks whether each condition
    # realizes ITS OWN expected gain, so every box has a different denominator
    # and the PSTH is just one more predictor rather than a benchmark.
    stats_self = plot_metric(
        ax_e, data, "selfcons", "Poisson self-consistency\nfraction",
        show_xticklabels=True, label="Poisson self-consistency fraction",
        ref_value=1.0, ref_label="expected\nvalue 1", ref_color="0.35",
        ref_label_dx=0.60)

    _panel_heading(ax_a, "A", "Single-trial variance explained")
    _panel_heading(ax_b, "B", "Single-trial $r^2$ as a fraction of the "
                              "explainable rate variance measured in Fig. 2\n"
                              "($R^2_{max}$ = rate variance / total variance, "
                              "per unit, at the model's 120 Hz resolution)")
    _panel_heading(ax_c, "C", "Single-trial information rate")
    _panel_heading(ax_d, "D", "Fraction of the twin's rate modulation "
                              "that each input carries")
    _panel_heading(ax_e, "E", "Observed likelihood gain relative to the gain "
                              "expected if each condition's own\npredicted "
                              "rates generated independent Poisson counts "
                              "(expectation 1, not a bound)")

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
        "stats": {"single_trial_r2": stats_r2,
                  "normalized_single_trial_r2": stats_norm,
                  "bits_per_spike": stats_bps,
                  "residual_rate_fraction": stats_resid,
                  "poisson_self_consistency": stats_self},
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
