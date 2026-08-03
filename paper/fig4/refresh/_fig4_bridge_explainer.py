#!/usr/bin/env python3
"""Composite explainer figure for the random-rotation trace-contour bridge.

This is a staging figure, not a final ssi_figure_v2 panel. It exists to
unpack the single strongest claim in the behavior-model bridge compendium
(outputs/fig/ssi_figure_v2/behavior_model_bridge/behavior_model_bridge_diagnostic_compendium.pdf)
into one self-contained, four-panel story:

    A. What the random-rotation null does and does not change.
    B. The headline result: observed trace-contour matching beats random
       rotation, selectively for high-SF populations.
    C. That advantage grows with local edge coherence (dose-response).
    D. The mechanism: it is carried by the contour-normal component, and
       flips sign for low-SF units (double dissociation).

Colors, typography, and the panel-header style all follow
paper/fig4/_fig4_ssi_common.py (BLUE = low-SF, ORANGE =
high-SF, panel titles uncolored when the panel mixes SF groups) so this reads
as part of the same figure family rather than a separately-styled document.

Panel A reuses the real stimulus image, contour axis, and trace-drawing code
from paper/fig4/_fig4_contour_schematic.py (the same assets behind
ssi_figure_v2 Panel D) rather than a synthetic diagram: the same real trace,
rotated to two illustrative orientations relative to the same fixed contour.
Panels B-D are built directly from the random-rotation null CSVs already
computed by _fig4_bridge_rotation_null.py and
_fig4_bridge_by_coherence.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
import _fig4_paths as _paths

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from _fig4_style import (
    GRAY, INK, ORANGE, PALE_GRID, POPULATION_COLORS, POPULATION_MARKERS,
    configure_matplotlib,
)
import _fig4_bridge as bridge
import _fig4_bridge_by_coherence as coh_bridge
import _fig4_contour_schematic as ssi_schematic

OUT_DIR = _paths.FIG_DIR / "behavior_model_bridge"
OUT_STEM = "behavior_model_bridge_explainer_figure"
MATCH_NULL_SUMMARY_CSV = _paths.BRIDGE_MATCH_NULL_SUMMARY_CSV
COHERENCE_SUMMARY_CSV = _paths.BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV


HEADLINE_SUBSET_KEY = "coh_ge_0p2"
HEADLINE_POPULATION_ORDER = (
    "high_sf_aligned",
    "high_sf_oblique",
    "high_sf_orthogonal",
    "high_sf_all",
    "low_sf_all",
)
METRIC_MARKERS = {"component_rms": "o", "component_range": "s"}
MECHANISM_METRIC = "component_rms"
MECHANISM_POPULATIONS = ("high_sf_aligned", "low_sf_all")

# Illustrative rotation angles for Panel A: chosen (from the real trace's
# actual geometry, not cherry-picked per rendered look) so one orientation
# reads as contour-aligned and the other as clearly cross-cutting -- see
# _schematic_rms_by_angle in the module docstring-adjacent comment below.
SCHEMATIC_ALIGNED_ROTATION_DEG = 90.0
SCHEMATIC_RANDOM_ROTATION_DEG = 15.0
CENTER_ZOOM_HALF_DEG = 0.25
CENTER_ZOOM_TRACE_PAD_DEG = 0.04


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Panel A: schematic of the random-rotation null, built from real D assets.
# ---------------------------------------------------------------------------


def _schematic_zoom_half_px(traces: list[np.ndarray], ppd: float) -> float:
    """Same auto-fit convention as _fig4_ssi_common.trace_fit_center_zoom_metadata."""
    extents = [float(np.nanmax(np.abs(t))) for t in traces if t.size]
    min_half_px = CENTER_ZOOM_HALF_DEG * ppd
    if not extents:
        return min_half_px
    return max(min_half_px, max(extents) + CENTER_ZOOM_TRACE_PAD_DEG * ppd)


def plot_schematic(ax_obs: plt.Axes, ax_rot: plt.Axes) -> dict[str, float]:
    payload = ssi_schematic.load_real_payload()
    contour_axis_image_deg = float(payload.get("contour_axis_image_deg", 10.352312))
    synthetic_left = ssi_schematic.make_synthetic_left_side(payload.get("patch"), contour_axis_image_deg)
    motion_eye = synthetic_left["eye"]
    ppd = ssi_schematic.MODEL_PPD
    normal = ssi_schematic.panel_a_grating_normal(contour_axis_image_deg)
    empty_trace = np.zeros((0, 2), dtype=np.float64)

    panel_specs = (
        (ax_obs, SCHEMATIC_ALIGNED_ROTATION_DEG, "Aligned orientation", ORANGE),
        (ax_rot, SCHEMATIC_RANDOM_ROTATION_DEG, "Randomly rotated", GRAY),
    )
    rotated_by_angle = {
        angle: ssi_schematic.rotate_trace_xy_px(motion_eye["large_xy_px"], angle) for _, angle, _, _ in panel_specs
    }
    # Both panels share one zoom window (fit to whichever rotation is larger)
    # so the two normal-RMS bands are visually comparable, not an artifact of
    # independently auto-scaled crops.
    half_px = _schematic_zoom_half_px(list(rotated_by_angle.values()), ppd)

    rms_by_angle: dict[str, float] = {}
    for ax, angle, title, color in panel_specs:
        rotated = rotated_by_angle[angle]
        axis_center = ssi_schematic.add_stimulus(
            ax,
            payload.get("patch"),
            contour_axis_image_deg,
            motion_eye={"small_xy_px": empty_trace, "large_xy_px": empty_trace},
        )
        ssi_schematic.add_panel_a_trace_path(ax, axis_center, rotated, color, lw=2.2, zorder=6)
        cx, cy = axis_center
        ax.set_xlim(cx - half_px, cx + half_px)
        ax.set_ylim(cy + half_px, cy - half_px)

        rms_arcmin = float(np.std(rotated @ normal)) / ppd * 60.0
        rms_by_angle[f"{angle:g}_deg"] = rms_arcmin
        ax.set_title(title, fontsize=8.6, fontweight="bold", color=INK, pad=4)
        ax.text(
            0.97,
            0.045,
            f"normal RMS ≈ {rms_arcmin:.1f}′",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=color, edgecolor="none", alpha=0.92),
            zorder=10,
        )
        ax.text(
            0.03,
            0.96,
            "teal = local\nimage contour",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="white",
            linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#00000090", edgecolor="none"),
        )

    return rms_by_angle


def draw_rotation_arrow(fig: plt.Figure, ax_obs: plt.Axes, ax_rot: plt.Axes) -> None:
    """Connect the two schematic axes with an arrow drawn in figure space.

    Using figure-fraction coordinates (rather than an axes transform with
    coordinates outside [0, 1]) keeps the arrow and its label confined to the
    gap between the two panels regardless of the exact gridspec spacing, so
    nothing gets clipped at the edge of the canvas.
    """
    pos_obs = ax_obs.get_position()
    pos_rot = ax_rot.get_position()
    mid_y = (pos_obs.y0 + pos_obs.y1) / 2.0
    x_start = pos_obs.x1 + 0.012
    x_end = pos_rot.x0 - 0.012
    arrow = FancyArrowPatch(
        (x_start, mid_y),
        (x_end, mid_y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        color=GRAY,
        lw=1.3,
        clip_on=False,
    )
    fig.add_artist(arrow)
    fig.text(
        (x_start + x_end) / 2.0,
        mid_y + 0.018,
        "rotate the same\ntrace by a\nrandom angle",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=GRAY,
    )


# ---------------------------------------------------------------------------
# Panel B: headline forest plot.
# ---------------------------------------------------------------------------


def plot_headline(ax: plt.Axes, summary: pd.DataFrame) -> None:
    metrics = (
        ("component_rms", "RMS excursion", -0.13),
        ("component_range", "Projected range", 0.13),
    )
    frame = summary[
        summary["subset_key"].astype(str).eq(HEADLINE_SUBSET_KEY)
        & summary["score_type"].astype(str).eq("component_mean_marginal")
        & summary["population_key"].astype(str).isin(HEADLINE_POPULATION_ORDER)
        & summary["metric_family"].astype(str).isin([m[0] for m in metrics])
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No headline rows found in {MATCH_NULL_SUMMARY_CSV}")

    y_base = np.arange(len(HEADLINE_POPULATION_ORDER), dtype=float)[::-1]
    y_lookup = dict(zip(HEADLINE_POPULATION_ORDER, y_base, strict=True))
    population_labels = {key: coh_bridge.POPULATION_LABELS[key] for key in HEADLINE_POPULATION_ORDER}

    ax.axvspan(-0.14, 0.0, color=GRAY, alpha=0.10, lw=0)
    ax.axvspan(0.0, 0.20, color=ORANGE, alpha=0.08, lw=0)
    ax.axhspan(
        y_lookup["high_sf_aligned"] - 0.42,
        y_lookup["high_sf_aligned"] + 0.42,
        color=ORANGE,
        alpha=0.14,
        lw=0,
        zorder=0,
    )
    ax.axvline(0.0, color=INK, lw=1.0, ls=":")

    for metric_key, label, offset in metrics:
        sub = frame[frame["metric_family"].astype(str).eq(metric_key)].set_index("population_key")
        xs, xlo, xhi, ys, colors = [], [], [], [], []
        for population in HEADLINE_POPULATION_ORDER:
            row = sub.loc[population]
            x = float(row["observed_minus_rotated_session_mean"])
            lo = float(row["observed_minus_rotated_ci95_low"])
            hi = float(row["observed_minus_rotated_ci95_high"])
            xs.append(x)
            xlo.append(x - lo)
            xhi.append(hi - x)
            ys.append(y_lookup[population] + offset)
            colors.append(POPULATION_COLORS[population])
        for x, xl, xh, y, color in zip(xs, xlo, xhi, ys, colors, strict=True):
            ax.errorbar(
                [x],
                [y],
                xerr=[[xl], [xh]],
                fmt=METRIC_MARKERS[metric_key],
                ms=6.0,
                lw=1.6,
                capsize=3.0,
                color=color,
                markerfacecolor="white",
                markeredgewidth=1.6,
                zorder=3,
            )

    metric_legend_handles = [
        plt.Line2D([0], [0], color=INK, marker=METRIC_MARKERS[key], markerfacecolor="white", lw=1.4, label=label)
        for key, label, _ in metrics
    ]

    ax.set_yticks([y_lookup[p] for p in HEADLINE_POPULATION_ORDER])
    ax.set_yticklabels([population_labels[p] for p in HEADLINE_POPULATION_ORDER])
    for tick, population in zip(ax.get_yticklabels(), HEADLINE_POPULATION_ORDER, strict=True):
        tick.set_color(POPULATION_COLORS[population])
        tick.set_fontweight("bold" if population == "high_sf_aligned" else "normal")
    ax.set_xlabel("Observed − random-rotated model SSI prediction (pp)")
    ax.set_xlim(-0.13, 0.21)
    ax.set_ylim(-0.75, len(HEADLINE_POPULATION_ORDER) - 0.25)
    ax.grid(axis="x", color=PALE_GRID, lw=0.8)
    _clean_axis(ax)
    ax.text(
        -0.128,
        len(HEADLINE_POPULATION_ORDER) - 0.35,
        "random rotation better",
        ha="left",
        va="top",
        fontsize=6.8,
        color=GRAY,
    )
    ax.text(
        0.205,
        len(HEADLINE_POPULATION_ORDER) - 0.35,
        "observed matching better",
        ha="right",
        va="top",
        fontsize=6.8,
        color=ORANGE,
    )
    ax.legend(handles=metric_legend_handles, frameon=False, fontsize=6.6, loc="lower right")


# ---------------------------------------------------------------------------
# Panel C: coherence dose-response.
# ---------------------------------------------------------------------------


def plot_dose_response(axes: list[plt.Axes], coherence_summary: pd.DataFrame) -> None:
    frame = coherence_summary[coherence_summary["score_type"].astype(str).eq("component_mean_marginal")].copy()
    frame["coherence_bin"] = pd.Categorical(frame["coherence_bin"], categories=bridge.COHERENCE_ORDER, ordered=True)
    x = np.arange(len(bridge.COHERENCE_ORDER), dtype=float)

    for ax, metric_key in zip(axes, coh_bridge.PRIMARY_METRICS, strict=True):
        sub_metric = frame[frame["metric_family"].astype(str).eq(metric_key)].copy()
        ax.axhspan(0.0, 0.25, color=ORANGE, alpha=0.07, lw=0)
        ax.axhspan(-0.25, 0.0, color=GRAY, alpha=0.08, lw=0)
        ax.axhline(0.0, color=INK, lw=1.0, ls=":", alpha=0.6)
        for population_key in HEADLINE_POPULATION_ORDER:
            sub = sub_metric[sub_metric["population_key"].astype(str).eq(population_key)].copy()
            sub = sub.sort_values("coherence_bin")
            y = sub["observed_minus_rotated"].to_numpy(dtype=float)
            lo = sub["observed_minus_rotated_ci95_low"].to_numpy(dtype=float)
            hi = sub["observed_minus_rotated_ci95_high"].to_numpy(dtype=float)
            lw = 2.2 if population_key == "high_sf_aligned" else 1.4
            alpha = 1.0 if population_key in ("high_sf_aligned", "low_sf_all") else 0.85
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                color=POPULATION_COLORS[population_key],
                marker=POPULATION_MARKERS[population_key],
                markerfacecolor="white",
                markeredgewidth=1.3,
                lw=lw,
                capsize=2.4,
                alpha=alpha,
                label=coh_bridge.POPULATION_LABELS[population_key],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(bridge.COHERENCE_ORDER)
        ax.set_xlabel("local edge coherence")
        ax.set_title(coh_bridge.METRIC_TITLES[metric_key], loc="left", fontsize=8.6, fontweight="bold", color=INK)
        ax.grid(axis="y", color=PALE_GRID, lw=0.75)
        _clean_axis(ax)

    axes[0].set_ylabel("observed − random rotated\n(pp SSI)")


# ---------------------------------------------------------------------------
# Panel D: mechanism / double dissociation.
# ---------------------------------------------------------------------------


def plot_mechanism(ax: plt.Axes, summary: pd.DataFrame) -> None:
    frame = summary[
        summary["subset_key"].astype(str).eq(HEADLINE_SUBSET_KEY)
        & summary["score_type"].astype(str).eq("component")
        & summary["metric_family"].astype(str).eq(MECHANISM_METRIC)
        & summary["population_key"].astype(str).isin(MECHANISM_POPULATIONS)
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No mechanism rows found in {MATCH_NULL_SUMMARY_CSV}")

    components = ("across", "along")
    component_labels = {"across": "contour-normal", "along": "contour-parallel"}
    component_hatch = {"across": None, "along": "///"}
    group_labels = {key: coh_bridge.POPULATION_LABELS[key] for key in MECHANISM_POPULATIONS}
    group_colors = {key: POPULATION_COLORS[key] for key in MECHANISM_POPULATIONS}

    bar_width = 0.34
    group_x = np.array([0.0, 1.6])
    ax.axhline(0.0, color=INK, lw=1.0, ls=":")

    for offset_idx, component in enumerate(components):
        offset = (offset_idx - 0.5) * bar_width
        sub = frame[frame["component"].astype(str).eq(component)].set_index("population_key")
        ys, ylo, yhi, colors = [], [], [], []
        for population in MECHANISM_POPULATIONS:
            row = sub.loc[population]
            y = float(row["observed_minus_rotated_session_mean"])
            lo = float(row["observed_minus_rotated_ci95_low"])
            hi = float(row["observed_minus_rotated_ci95_high"])
            ys.append(y)
            ylo.append(y - lo)
            yhi.append(hi - y)
            colors.append(group_colors[population])
        ax.bar(
            group_x + offset,
            ys,
            width=bar_width,
            color=colors,
            hatch=component_hatch[component],
            edgecolor="white",
            linewidth=0.6,
            label=component_labels[component],
            zorder=2,
        )
        ax.errorbar(
            group_x + offset,
            ys,
            yerr=np.vstack([ylo, yhi]),
            fmt="none",
            ecolor=INK,
            elinewidth=1.1,
            capsize=3.0,
            zorder=3,
        )

    ax.set_xlim(group_x[0] - 0.85, group_x[-1] + 0.85)
    ax.set_xticks(group_x)
    ax.set_xticklabels([group_labels[p] for p in MECHANISM_POPULATIONS])
    for tick, population in zip(ax.get_xticklabels(), MECHANISM_POPULATIONS, strict=True):
        tick.set_color(group_colors[population])
        tick.set_fontweight("bold")
    ax.set_ylabel("observed − random rotated\nRMS-excursion prediction (pp)")
    ax.grid(axis="y", color=PALE_GRID, lw=0.8)
    _clean_axis(ax)
    component_legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch=component_hatch[c], label=component_labels[c])
        for c in components
    ]
    ax.legend(handles=component_legend_handles, frameon=False, fontsize=6.6, loc="upper right")


# ---------------------------------------------------------------------------
# Figure assembly.
# ---------------------------------------------------------------------------


LEFT_MARGIN = 0.145
RIGHT_MARGIN = 0.975
TOP_MARGIN = 0.905
BOTTOM_MARGIN = 0.075


def build(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    match_null_summary = pd.read_csv(MATCH_NULL_SUMMARY_CSV)
    coherence_summary = pd.read_csv(COHERENCE_SUMMARY_CSV)

    fig = plt.figure(figsize=(10.6, 13.8))
    outer = fig.add_gridspec(
        5,
        1,
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=TOP_MARGIN,
        bottom=BOTTOM_MARGIN,
        height_ratios=[0.42, 1.05, 0.95, 0.16, 1.05],
        hspace=0.58,
    )

    row_a = outer[0].subgridspec(1, 3, width_ratios=[1.0, 0.32, 1.0], wspace=0.08)
    ax_obs = fig.add_subplot(row_a[0, 0])
    ax_rot = fig.add_subplot(row_a[0, 2])
    schematic_rms = plot_schematic(ax_obs, ax_rot)

    ax_headline = fig.add_subplot(outer[1])
    plot_headline(ax_headline, match_null_summary)

    row_c = outer[2].subgridspec(1, 2, wspace=0.30)
    ax_rms = fig.add_subplot(row_c[0, 0])
    ax_range = fig.add_subplot(row_c[0, 1], sharey=ax_rms)
    plot_dose_response([ax_rms, ax_range], coherence_summary)

    ax_legend_row = fig.add_subplot(outer[3])
    ax_legend_row.axis("off")

    ax_mechanism = fig.add_subplot(outer[4])
    plot_mechanism(ax_mechanism, match_null_summary)

    # Draw the schematic's connecting arrow only after both axes have their
    # final (aspect-adjusted) positions.
    draw_rotation_arrow(fig, ax_obs, ax_rot)

    # Shared population legend for panel C, placed in its own slim row so it
    # cannot collide with panel C's x-axis labels or panel D's title.
    handles, labels = ax_rms.get_legend_handles_labels()
    legend_pos = ax_legend_row.get_position()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, (legend_pos.y0 + legend_pos.y1) / 2.0),
        ncol=5,
        frameon=False,
        fontsize=7.2,
        handlelength=1.6,
        columnspacing=1.3,
    )

    panel_specs = (
        ("A", ax_obs, "Random-rotation null: same trace, broken trace-contour relationship"),
        ("B", ax_headline, "Observed matching beats random rotation, selectively for high-SF units"),
        ("C", ax_rms, "The match advantage grows with local edge coherence"),
        ("D", ax_mechanism, "Mechanism: carried by the contour-normal axis; low-SF units flip sign"),
    )
    for letter, anchor_ax, panel_title in panel_specs:
        bbox = anchor_ax.get_position()
        fig.text(LEFT_MARGIN, bbox.y1 + 0.020, letter, fontsize=14.0, fontweight="bold", ha="left", va="bottom", color=INK)
        fig.text(
            LEFT_MARGIN + 0.030,
            bbox.y1 + 0.020,
            panel_title,
            fontsize=10.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK,
        )

    fig.text(
        LEFT_MARGIN,
        0.988,
        "Real trace-contour matching is model-beneficial, selectively for high-SF units",
        fontsize=13.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    fig.text(
        LEFT_MARGIN,
        0.964,
        "Staging figure for ssi_figure_v2 — random-rotation null on 0.325 s BackImage snippets; pp = percentage points of predicted SSI residual",
        fontsize=8.5,
        color=GRAY,
        ha="left",
        va="top",
    )

    mechanism_bbox = ax_mechanism.get_position()
    fig.text(
        (LEFT_MARGIN + RIGHT_MARGIN) / 2.0,
        max(mechanism_bbox.y0 - 0.050, 0.012),
        "Signs flip between populations and between components: normal-axis matching helps aligned\n"
        "high-SF units and hurts low-SF units, so the effect is not a generic feature of rotating a trace.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=GRAY,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    svg = out_dir / f"{OUT_STEM}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)

    provenance = {
        "figure": OUT_STEM,
        "inputs": {
            "match_null_summary_csv": _relative(MATCH_NULL_SUMMARY_CSV),
            "coherence_summary_csv": _relative(COHERENCE_SUMMARY_CSV),
        },
        "panel_a_note": (
            "Real stimulus image, contour axis, and trace shape reused from ssi_figure_v2 Panel D "
            "(paper/fig4/_fig4_contour_schematic.py); the two orientations shown are the same "
            "real trace rotated to illustrative angles, not two different recordings or a claim about "
            "this specific window's true orientation. Population-level statistics are in panels B-D."
        ),
        "panel_a_normal_rms_arcmin_by_rotation": schematic_rms,
        "outputs": {"png": _relative(png), "pdf": _relative(pdf), "svg": _relative(svg)},
    }
    provenance_path = out_dir / f"{OUT_STEM}_provenance.json"
    provenance_path.write_text(json.dumps(_json_ready(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"png": png, "pdf": pdf, "svg": svg, "provenance": provenance_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build(out_dir=args.out_dir)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
