#!/usr/bin/env python3
"""Panel J (promoted): trace-contour match advantage by local edge coherence.

This is the single-panel distillation of the behavior-model bridge: for each
population, how much more (or less) predicted SSI the real trace-contour
matching gives you than a randomly rotated trajectory, as a function of local
edge coherence. It reuses the same data as the standalone explainer figure's
Panel C (`_fig4_bridge_explainer.py`) and the same population palette, which
now lives in `_fig4_style.py` rather than being reached through the explainer,
rendered at this
slot's real footprint (originally Panel I's; the whole figure's G/H/I shifted
to H/I/J once a new panel G was inserted between F and the old G -- see
_fig4_ssi_common.py's EF_INSET_* constants and draw_contour_components_panel).

This replaced panel_i_edge_alignment.py's descriptive drift-cloud/edge
alignment plot in the main ssi_figure_v2 slot: that panel showed behavior
correlates with coherence, but never showed that the correlation is
model-beneficial relative to chance -- this panel closes that loop. The
original is left in place, unwired, for reference/comparison.

Only three of the five populations from the explainer figure are shown here
(aligned high-SF, all high-SF, all low-SF): oblique and orthogonal high-SF
sit between aligned and all-high-SF and don't add a distinguishable line at
this panel's size -- all-high-SF already carries that "partial or no
alignment" middle ground. The five-population version remains the one to use
in the explainer/option-sheet context where there's room for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import _fig4_panel_header
import _fig4_style as style
from _fig4_style import configure_matplotlib

OUT_DIR = _paths.PANELS_DIR
COHERENCE_SUMMARY_CSV = _paths.BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV
METRIC_FAMILY = "component_rms"
# The real ssi_figure_v2 gs[2, 2] cell (MAIN_GRID_KWARGS at FIGURE_SIZE_IN =
# (8.5, 11.0)) -- panel I's actual footprint, not its ~2.35x2.25 standalone
# preview approximation.
FIGSIZE = (1.955, 2.432)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_values(csv_path: Path = COHERENCE_SUMMARY_CSV) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame[frame["score_type"].astype(str).eq("component_mean_marginal") & frame["metric_family"].astype(str).eq(METRIC_FAMILY)].copy()
    frame["coherence_bin"] = pd.Categorical(frame["coherence_bin"], categories=style.COHERENCE_ORDER, ordered=True)
    return frame


PLOT_POPULATION_ORDER = ("high_sf_aligned", "high_sf_all", "low_sf_all")
SHORT_POPULATION_LABELS = {
    "high_sf_aligned": "Aligned high-SF",
    "high_sf_all": "All high-SF",
    "low_sf_all": "Low-SF",
}


TITLE = "Contour-matched FEMs\nbeat rotations for\naligned high-SF units"


def draw_panel(
    ax: plt.Axes,
    *,
    label: str = "J",
    title: str = TITLE,
    values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Draw the coherence-resolved match-advantage panel on ``ax``."""
    values = load_values() if values is None else values.copy()
    x = np.arange(len(style.COHERENCE_ORDER), dtype=float)

    ax.axhline(0.0, color=style.INK, lw=0.9, ls=":", alpha=0.6)

    # Significance is encoded on the marker itself -- filled when the 95% CI
    # excludes zero, open (white) when it doesn't -- rather than floating
    # asterisks: three overlapping series with per-point jittered text got
    # cluttered fast, whereas filled/open reuses an element already in the
    # plot and reads at a glance without extra ink.
    for population_key in PLOT_POPULATION_ORDER:
        sub = values[values["population_key"].astype(str).eq(population_key)].sort_values("coherence_bin")
        y = sub["observed_minus_rotated"].to_numpy(dtype=float)
        lo = sub["observed_minus_rotated_ci95_low"].to_numpy(dtype=float)
        hi = sub["observed_minus_rotated_ci95_high"].to_numpy(dtype=float)
        is_aligned = population_key == "high_sf_aligned"
        color = style.POPULATION_COLORS[population_key]
        marker = style.POPULATION_MARKERS[population_key]
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            color=color,
            marker="none",
            lw=2.0 if is_aligned else 1.5,
            capsize=0,
            zorder=4 if is_aligned else 3,
        )
        significant = (lo > 0.0) | (hi < 0.0)
        face = np.where(significant, color, "white")
        ax.scatter(
            x,
            y,
            marker=marker,
            s=17.0,
            facecolors=face,
            edgecolors=color,
            linewidths=1.0,
            zorder=5 if is_aligned else 4,
        )

    ax.set_xlim(-0.45, len(style.COHERENCE_ORDER) - 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(style.COHERENCE_ORDER, fontsize=5.6, rotation=0, ha="center")
    ax.set_xlabel("local edge coherence", labelpad=1.5)
    # Single line, not the original 2-line "observed - random rotated\n(pp
    # SSI, RMS excursion)": a rotated ylabel's line-stacking direction
    # becomes horizontal once rotated 90 degrees, so a 2-line label costs
    # roughly double the width of a 1-line one -- real money in J's narrow
    # column. The title already establishes "real vs. randomly rotated" and
    # this whole panel row is RMS-excursion-based, so neither needs repeating
    # here.
    ax.set_ylabel("SSI advantage (pp)", labelpad=2.0)
    ax.grid(axis="y", color=style.PALE_GRID, lw=0.75)
    ax.set_axisbelow(True)
    _fig4_panel_header.draw_bottom_row_header(
        ax,
        label,
        title,
        title_linespacing=_fig4_panel_header.PANEL_TITLE_LINESPACING,
        color=style.INK,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Explicit handles, not the plotted artists' own labels: the errorbar
    # (line only, marker="none") and scatter (varies facecolor per point)
    # calls above aren't set up to hand the legend a clean single swatch per
    # population -- these mirror them, filled to match a "significant" point.
    population_handles = [
        Line2D(
            [0],
            [0],
            color=style.POPULATION_COLORS[key],
            marker=style.POPULATION_MARKERS[key],
            markersize=4.2,
            markerfacecolor=style.POPULATION_COLORS[key],
            markeredgewidth=1.0,
            lw=2.0 if key == "high_sf_aligned" else 1.5,
            label=SHORT_POPULATION_LABELS[key],
        )
        for key in PLOT_POPULATION_ORDER
    ]
    significance_handle = Line2D(
        [0],
        [0],
        color=style.INK,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        lw=0.0,
        label="open: CI includes 0",
    )
    ax.legend(
        handles=[*population_handles, significance_handle],
        frameon=False,
        fontsize=5.4,
        loc="lower left",
        handlelength=1.2,
        labelspacing=0.28,
        borderaxespad=0.2,
        handletextpad=0.4,
    )
    ax.tick_params(axis="y", labelsize=6.8)
    ax.tick_params(axis="x", labelsize=5.6, pad=2.0)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    _fig4_panel_header.align_bottom_row_xlabel(ax)

    return values


def build_panel(
    out_dir: Path = OUT_DIR,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    label: str = "J",
    title: str = TITLE,
) -> dict[str, Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_values()
    values.to_csv(out_dir / "panel_g_match_advantage_values.csv", index=False)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    ax = _fig4_panel_header.add_bottom_row_axes(fig)
    draw_panel(ax, label=label, title=title, values=values)
    paths = {
        "png": out_dir / "panel_g_match_advantage.png",
        "pdf": out_dir / "panel_g_match_advantage.pdf",
        "svg": out_dir / "panel_g_match_advantage.svg",
    }
    fig.savefig(paths["png"], dpi=220, transparent=True)
    fig.savefig(paths["pdf"], transparent=True)
    fig.savefig(paths["svg"], transparent=True)
    plt.close(fig)
    return paths


def main() -> None:
    paths = build_panel()
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
