#!/usr/bin/env python3
"""Panel B with a 15 degree unit-contour orientation-match threshold."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
from VisionCore.paths import VISIONCORE_ROOT as ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import _fig4_geometry_story_cde8bins as story


MATCH_MAX_DEG = 15.0
HIGH_SF_MIN_CPD = 0.75
OUT_STEM = "backimage_real_trace_panel_b_cell_baseline_sf075_coh020_match15"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(story._json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_story() -> None:
    story.MATCH_MAX_DEG = MATCH_MAX_DEG
    story.HIGH_SF_MIN_CPD = HIGH_SF_MIN_CPD
    if story.SF_GROUP_MODE == "table_tertiles":
        low_sf_label = "lower SF tertile"
        high_sf_label = "upper SF tertile"
    else:
        low_sf_label = "SF < 0.50"
        high_sf_label = f"SF >= {HIGH_SF_MIN_CPD:.2f}"
    story.SF_GROUPS["high_ge0p75"]["label"] = high_sf_label
    story.SF_GROUPS["high_ge0p75"]["title"] = high_sf_label
    story.SF_GROUPS["low_lt0p5"]["label"] = low_sf_label
    story.SF_GROUPS["low_lt0p5"]["title"] = low_sf_label
    story.B_PANEL_SPECS = [
        ("low_lt0p5", "strong_contours_no_osi", f"{low_sf_label}\nall contour units"),
        ("low_lt0p5", "contour_matched", f"{low_sf_label}\nunit-contour match <= 15 deg"),
        ("high_ge0p75", "strong_contours_no_osi", f"{high_sf_label}\nall contour units"),
        ("high_ge0p75", "contour_matched", f"{high_sf_label}\nunit-contour match <= 15 deg"),
    ]


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=1.5,
            label="drift-only bins",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="0.25",
            markeredgewidth=1.25,
            linewidth=1.5,
            label="microsaccade bins",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.35,
            linewidth=0.0,
            label="matched stabilized baseline",
        ),
    ]


def _plot_panel_b(panel_b: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.35), sharey=True)
    b_ylim = story._shared_ylim(story._ylim_series(panel_b), pad_low=0.13, pad_high=0.17)
    for idx, (sf_group, relation, title) in enumerate(story.B_PANEL_SPECS):
        ax = axes[idx]
        frame = panel_b[panel_b["sf_group"].eq(sf_group) & panel_b["relation"].eq(relation)].copy()
        color = str(story.SF_GROUPS[sf_group]["color"])
        story._plot_b_series(ax, frame, color=color)
        story._format_broken_axis(
            ax,
            ticks=story.B_TICKS,
            min_pos=story.B_MIN_POS,
            max_pos=story.B_MAX_POS,
            xlabel="trajectory path (arcmin)",
        )
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_ylim(*b_ylim)
        n_units = int(frame["n_selected_units"].iloc[0]) if not frame.empty else 0
        n_pairs = int(frame["n_selected_unit_image_pairs"].iloc[0]) if not frame.empty else 0
        ax.set_title(f"{title}\n{n_units} units, {n_pairs} pairs", fontsize=9.2, pad=6, color=color)
        if idx == 0:
            ax.set_ylabel("SSI residual\n(% vs matched stabilized)")
        else:
            ax.spines["left"].set_visible(False)
    fig.suptitle(
        "Panel B with tighter orientation-aligned unit-image selection",
        fontsize=13.0,
        y=0.98,
    )
    fig.legend(
        handles=_legend_handles(),
        frameon=False,
        fontsize=8.0,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.02),
    )
    sf_description = (
        "SF groups: cycle-valid lower/upper tertiles. "
        if story.SF_GROUP_MODE == "table_tertiles"
        else f"Low SF: {story.SF_METRIC_COL} < {story.LOW_SF_MAX_CPD:.2f}; "
        f"high SF: {story.SF_METRIC_COL} >= {HIGH_SF_MIN_CPD:.2f}. "
    )
    fig.text(
        0.5,
        -0.085,
        (
            f"Contour windows: coherence >= {story.CONTOUR_COHERENCE_MIN:.2f}. "
            f"{sf_description}Only the orientation-aligned columns use the "
            "15 deg unit-contour match threshold."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.25",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.92))
    return fig


def _plot_high_sf_matched(panel_b: pd.DataFrame) -> plt.Figure:
    frame = panel_b[panel_b["sf_group"].eq("high_ge0p75") & panel_b["relation"].eq("contour_matched")].copy()
    fig, ax = plt.subplots(figsize=(5.05, 3.95))
    color = str(story.SF_GROUPS["high_ge0p75"]["color"])
    story._plot_b_series(ax, frame, color=color)
    story._format_broken_axis(
        ax,
        ticks=story.B_TICKS,
        min_pos=story.B_MIN_POS,
        max_pos=story.B_MAX_POS,
        xlabel="trajectory path (arcmin)",
    )
    ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
    ax.set_ylim(*story._shared_ylim(story._ylim_series(frame), pad_low=0.16, pad_high=0.18))
    n_units = int(frame["n_selected_units"].iloc[0]) if not frame.empty else 0
    n_pairs = int(frame["n_selected_unit_image_pairs"].iloc[0]) if not frame.empty else 0
    ax.set_title(
        (
            "Panel B: upper SF tertile, unit-contour match <= 15 deg"
            if story.SF_GROUP_MODE == "table_tertiles"
            else f"Panel B: SF >= {HIGH_SF_MIN_CPD:.2f}, unit-contour match <= 15 deg"
        )
        + f"\n{n_units} units, {n_pairs} pairs",
        fontsize=11.5,
        pad=7,
        color=color,
    )
    ax.set_ylabel("SSI residual\n(% vs matched stabilized)")
    ax.legend(handles=_legend_handles()[:2], frameon=False, fontsize=8.2, loc="lower left")
    fig.tight_layout()
    return fig


def main() -> None:
    _configure_story()
    data = story.load_dataset(story.MATRIX_DIR)
    panel_b, selection = story._compute_panel_b(data)
    story.OUT_DIR.mkdir(parents=True, exist_ok=True)

    values_csv = story.OUT_DIR / f"{OUT_STEM}_values.csv"
    selection_csv = story.OUT_DIR / f"{OUT_STEM}_selection_summary.csv"
    summary_json = story.OUT_DIR / f"{OUT_STEM}_summary.json"
    sheet_png = story.OUT_DIR / f"{OUT_STEM}.png"
    sheet_pdf = story.OUT_DIR / f"{OUT_STEM}.pdf"
    high_png = story.OUT_DIR / f"{OUT_STEM}_high_sf_aligned_only.png"
    high_pdf = story.OUT_DIR / f"{OUT_STEM}_high_sf_aligned_only.pdf"

    panel_b.to_csv(values_csv, index=False)
    selection.to_csv(selection_csv, index=False)

    fig = _plot_panel_b(panel_b)
    fig.savefig(sheet_png, dpi=230, bbox_inches="tight")
    fig.savefig(sheet_pdf, bbox_inches="tight")
    plt.close(fig)

    fig = _plot_high_sf_matched(panel_b)
    fig.savefig(high_png, dpi=230, bbox_inches="tight")
    fig.savefig(high_pdf, bbox_inches="tight")
    plt.close(fig)

    _write_json(
        summary_json,
        {
            "analysis": OUT_STEM,
            "matrix_dir": story.MATRIX_DIR,
            "out_dir": story.OUT_DIR,
            "outputs": {
                "sheet_png": sheet_png,
                "sheet_pdf": sheet_pdf,
                "high_sf_aligned_png": high_png,
                "high_sf_aligned_pdf": high_pdf,
                "values_csv": values_csv,
                "selection_summary_csv": selection_csv,
                "summary_json": summary_json,
            },
            "selection": {
                "sf_group_mode": story.SF_GROUP_MODE,
                "sf_metric_col": story.SF_METRIC_COL,
                "low_sf": (
                    "unit.sf_group == low_sf"
                    if story.SF_GROUP_MODE == "table_tertiles"
                    else f"{story.SF_METRIC_COL} < {story.LOW_SF_MAX_CPD}"
                ),
                "high_sf": (
                    "unit.sf_group == high_sf"
                    if story.SF_GROUP_MODE == "table_tertiles"
                    else f"{story.SF_METRIC_COL} >= {HIGH_SF_MIN_CPD}"
                ),
                "contour_coherence_min": story.CONTOUR_COHERENCE_MIN,
                "min_osi": story.MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": story.ORTHOGONAL_MIN_DEG,
                "panel_b_drift_bins": story.N_DRIFT_BINS,
                "panel_b_microsaccade_bins": story.N_MICROSACCADE_BINS,
            },
            "baseline": "Each plotted nonzero bin is compared with a cell-matched stabilized baseline weighted by that bin's image composition.",
        },
    )

    print(sheet_png)
    print(sheet_pdf)
    print(high_png)
    print(high_pdf)
    print(values_csv)
    print(selection_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
