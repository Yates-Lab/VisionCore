#!/usr/bin/env python3
"""Shared machinery for the trajectory-path-bin panels (displayed B and D).

Both panels plot SSI change against path length for a low-SF and a high-SF
group on one shared axis; they differ only in which source groups they draw and
where they sit on the page. Everything common lives here -- the values loader,
the broken x-axis, the two-colour titles, the support notes -- and the two
panel modules (`panel_b_path_bins.py`, `panel_d_path_bins.py`) supply the rest.

The source letters B/C/E/F in `PANEL_SPECS` are the ones measured off the
Illustrator reference: displayed panel B draws source pair (B, C), and
displayed panel D draws source pair (E, F).

Values come precomputed from ``paper/fig4/_fig4_orientation_match_sf05.py``,
whose upstream wrapper configures the BackImage story-panel code to use low
SF < 0.5, high SF >= 0.5, contour coherence >= 0.20, and a 15 degree
unit-contour match threshold for the aligned panels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, DrawingArea

import _fig4_panel_header

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths
from _fig4_broken_axis import (
    B_MAX_POS,
    B_MIN_POS,
    B_TICKS,
    format_broken_axis,
    plot_b_series,
    shared_ylim as _shared_ylim,
    x_broken_log,
    ylim_series,
)
from _fig4_style import BLUE, GRAY, INK, ORANGE, configure_matplotlib

FIG4_DIR = ROOT / "paper" / "fig4"

# The selection this panel's cached CSVs were produced under. These used to be
# read back off the upstream analysis modules at import: this module imported
# `_fig4_orientation_match_sf05`, which mutates globals on
# `_fig4_orientation_match`, which mutates globals on
# `_fig4_geometry_story_cde8bins`, and the resolved values were then read back
# out -- ~4,000 lines of trace-bank analysis imported to obtain nine numbers.
# They are recorded here instead, and the scripts they came from are still
# named in the provenance below.
SELECTION = {
    "sf_metric_col": "sf_split_metric",
    "low_sf_max_cpd": 0.5,
    "high_sf_min_cpd": 0.5,
    "contour_coherence_min": 0.2,
    "min_osi": 0.05,
    "match_max_deg": 15.0,
    "orthogonal_min_deg": 67.5,
    "n_drift_bins": 8,
    "n_microsaccade_bins": 5,
}

OUT_DIR = _paths.PANELS_DIR
VALUES_CSV = _paths.STORY_PANEL_B_VALUES_CSV
SELECTION_CSV = _paths.STORY_PANEL_B_SELECTION_SUMMARY_CSV
SUMMARY_JSON = _paths.STORY_PANEL_B_SUMMARY_JSON
UPSTREAM_WRAPPER = (
    FIG4_DIR / "refresh" / "_fig4_orientation_match_sf05.py"
)
UPSTREAM_CORE = (
    FIG4_DIR / "refresh" / "_fig4_orientation_match.py"
)
UPSTREAM_STORY = (
    FIG4_DIR / "refresh" / "_fig4_geometry_story_cde8bins.py"
)

BROKEN_AXIS_BREAK_CENTER = 0.545
BROKEN_AXIS_TICK_OFFSET = 0.048
BROKEN_AXIS_TICK_HALF_WIDTH = 0.040
BROKEN_AXIS_TICK_HALF_HEIGHT = 0.035
TOP_ROW_PAIR_AXES_BOX = (0.1465, 0.1998, 0.8100, 0.7000)
TOP_ROW_PAIR_LETTER_X = 0.0
TOP_ROW_PAIR_HEADER_Y = 0.9205
TOP_ROW_PAIR_LETTER_Y_OFFSET_PT = -3.15

PANEL_SPECS = {
    "B": {
        "title": "Low-SF units",
        "sf_group": "low_lt0p5",
        "relation": "strong_contours_no_osi",
        "color": BLUE,
        "ylabel": "SSI change (%)",
    },
    "C": {
        "title": "High-SF units",
        "sf_group": "high_ge0p75",
        "relation": "strong_contours_no_osi",
        "color": ORANGE,
        "ylabel": "SSI change (%)",
    },
    "E": {
        "title": "Low-SF aligned units",
        "sf_group": "low_lt0p5",
        "relation": "contour_matched",
        "color": BLUE,
        "ylabel": "SSI change (%)",
    },
    "F": {
        "title": "High-SF aligned units",
        "sf_group": "high_ge0p75",
        "relation": "contour_matched",
        "color": ORANGE,
        "ylabel": "SSI change (%)",
    },
}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_panel_values(values_csv: Path = VALUES_CSV) -> pd.DataFrame:
    if not values_csv.exists():
        raise FileNotFoundError(values_csv)
    values = pd.read_csv(values_csv)
    required = {
        "sf_group",
        "sf_group_label",
        "relation",
        "context",
        "path_median_arcmin",
        "path_bin_order",
        "ssi_percent_vs_cell_baseline",
        # The ssi_percent_ci95_{low,high}_image_boot columns were required here
        # but never read: the panel stopped drawing bootstrap intervals, and the
        # check outlived the plotting code that needed them. Requiring columns
        # nothing consumes rejects otherwise-valid caches.
        "n_selected_units",
        "n_selected_unit_image_pairs",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Missing required columns in {values_csv}: {missing}")
    return values


def shared_ylim(values: pd.DataFrame, *, pad_low: float = 0.12, pad_high: float = 0.14) -> tuple[float, float]:
    if values.empty:
        return (-20.0, 48.0)
    return _shared_ylim(ylim_series(values), pad_low=pad_low, pad_high=pad_high)


def shared_ylim_for(
    values: pd.DataFrame,
    *,
    relations: tuple[str, ...],
    pad_low: float = 0.12,
    pad_high: float = 0.14,
) -> tuple[float, float]:
    if values.empty:
        return (-20.0, 48.0)
    frame = values[values["relation"].isin(relations)].copy()
    if frame.empty:
        return shared_ylim(values, pad_low=pad_low, pad_high=pad_high)
    return _shared_ylim(ylim_series(frame), pad_low=pad_low, pad_high=pad_high)


def path_xlimit_right(values: pd.DataFrame, *, pad: float = 0.22) -> float:
    if values.empty or "path_median_arcmin" not in values:
        return 5.55
    raw = pd.to_numeric(values["path_median_arcmin"], errors="coerce").to_numpy(dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return 5.55
    mapped = x_broken_log(raw, min_pos=B_MIN_POS, max_pos=B_MAX_POS)
    return float(np.nanmax(mapped) + pad)


def _remove_upstream_break_label(ax: plt.Axes) -> None:
    for text in list(ax.texts):
        if text.get_text() == "//":
            text.remove()


def _draw_broken_x_axis(ax: plt.Axes) -> None:
    _remove_upstream_break_label(ax)
    ax.spines["bottom"].set_visible(False)
    trans = ax.get_xaxis_transform()
    x_left, x_right = ax.get_xlim()
    left_slash_x = BROKEN_AXIS_BREAK_CENTER - BROKEN_AXIS_TICK_OFFSET
    right_slash_x = BROKEN_AXIS_BREAK_CENTER + BROKEN_AXIS_TICK_OFFSET
    ax.plot(
        [x_left, left_slash_x],
        [0.0, 0.0],
        transform=trans,
        color="black",
        lw=0.8,
        clip_on=False,
        zorder=10,
    )
    ax.plot(
        [right_slash_x, x_right],
        [0.0, 0.0],
        transform=trans,
        color="black",
        lw=0.8,
        clip_on=False,
        zorder=10,
    )
    for offset in (-BROKEN_AXIS_TICK_OFFSET, BROKEN_AXIS_TICK_OFFSET):
        ax.plot(
            [
                BROKEN_AXIS_BREAK_CENTER + offset - BROKEN_AXIS_TICK_HALF_WIDTH,
                BROKEN_AXIS_BREAK_CENTER + offset + BROKEN_AXIS_TICK_HALF_WIDTH,
            ],
            [-BROKEN_AXIS_TICK_HALF_HEIGHT, BROKEN_AXIS_TICK_HALF_HEIGHT],
            transform=trans,
            color="black",
            lw=1.05,
            clip_on=False,
            solid_capstyle="butt",
            zorder=11,
        )


def _add_split_sf_zero_anchor(ax: plt.Axes) -> None:
    diameter = 7.0
    linewidth = 1.25
    pad = linewidth
    area = DrawingArea(diameter + 2 * pad, diameter + 2 * pad, 0, 0, clip=False)
    center = (diameter / 2 + pad, diameter / 2 + pad)
    area.add_artist(patches.Circle(center, diameter / 2, facecolor="white", edgecolor="none"))
    area.add_artist(patches.Arc(center, diameter, diameter, theta1=90, theta2=270, color=BLUE, lw=linewidth))
    area.add_artist(patches.Arc(center, diameter, diameter, theta1=-90, theta2=90, color=ORANGE, lw=linewidth))
    marker = AnnotationBbox(
        area,
        (0.0, 0.0),
        xycoords="data",
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        annotation_clip=False,
    )
    marker.set_zorder(12)
    marker.set_clip_on(False)
    ax.add_artist(marker)


def format_broken_path_axis(ax: plt.Axes, *, xlim_right: float | None = None) -> None:
    format_broken_axis(
        ax,
        ticks=B_TICKS,
        min_pos=B_MIN_POS,
        max_pos=B_MAX_POS,
        xlabel="path length (arcmin)",
    )
    if xlim_right is not None:
        ax.set_xlim(-0.12, xlim_right)
    _draw_broken_x_axis(ax)


def load_provenance(
    *,
    values_csv: Path = VALUES_CSV,
    selection_csv: Path = SELECTION_CSV,
    summary_json: Path = SUMMARY_JSON,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    return {
        "panels": ["B", "C", "E", "F"],
        "source_wrapper_script": _relative(UPSTREAM_WRAPPER),
        "source_core_script": _relative(UPSTREAM_CORE),
        "source_story_script": _relative(UPSTREAM_STORY),
        "source_values_csv": _relative(values_csv),
        "source_selection_csv": _relative(selection_csv),
        "source_summary_json": _relative(summary_json),
        "selection": summary.get(
            "selection",
            {
                "sf_metric_col": SELECTION["sf_metric_col"],
                "low_sf": f"{SELECTION['sf_metric_col']} < {SELECTION['low_sf_max_cpd']}",
                "high_sf": f"{SELECTION['sf_metric_col']} >= {SELECTION['high_sf_min_cpd']}",
                "contour_coherence_min": SELECTION["contour_coherence_min"],
                "min_osi": SELECTION["min_osi"],
                "match_max_deg": SELECTION["match_max_deg"],
                "orthogonal_min_deg": SELECTION["orthogonal_min_deg"],
                "panel_b_drift_bins": SELECTION["n_drift_bins"],
                "panel_b_microsaccade_bins": SELECTION["n_microsaccade_bins"],
            },
        ),
        "baseline": summary.get(
            "baseline",
            "Each plotted nonzero bin is compared with a cell-matched stabilized baseline weighted by that bin's image composition.",
        ),
        "error_bars": (
            "95% CI from a paired image-level bootstrap (N=10000, seed=47) of the moving-vs-cell-baseline "
            "ratio delta -- see ratio_delta_stats() in "
            "_fig4_trace_schematics.py. Images, not units, are the "
            "resampled cluster: each point is a spike-weighted ratio pooled over every (unit, image, "
            "trajectory-row) triple in the bin, so a naive per-unit SEM would double-count non-independent "
            "unit/trajectory draws nested within each image and would not be paired against the shared "
            "baseline computed from the same images."
        ),
    }


def _add_support_note(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    n_units = int(pd.to_numeric(frame["n_selected_units"], errors="coerce").dropna().iloc[0])
    n_pairs = int(pd.to_numeric(frame["n_selected_unit_image_pairs"], errors="coerce").dropna().iloc[0])
    # Upper-left, not upper-right: these series start near SSI=0 at path=0
    # and only grow larger later, so the top-right corner is exactly where
    # the last (often highest) data point and this note used to collide.
    ax.text(
        0.045,
        0.930,
        f"{n_units} units\n{n_pairs} pairs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=GRAY,
        linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    )


def _support_note_label(label: str) -> str:
    if label in {"B", "E"}:
        return "low-SF"
    if label in {"C", "F"}:
        return "high-SF"
    return label


def _support_note_line(label: str, frame: pd.DataFrame) -> str:
    n_units = int(pd.to_numeric(frame["n_selected_units"], errors="coerce").dropna().iloc[0])
    n_pairs = int(pd.to_numeric(frame["n_selected_unit_image_pairs"], errors="coerce").dropna().iloc[0])
    return f"{_support_note_label(label)}  {n_units} units, {n_pairs} pairs"


def _add_pair_support_note(ax: plt.Axes, frames: list[tuple[str, pd.DataFrame]]) -> None:
    valid_frames = [(label, frame) for label, frame in frames if not frame.empty]
    if not valid_frames:
        return
    for row, (label, frame) in enumerate(valid_frames):
        ax.text(
            0.045,
            0.950 - 0.050 * row,
            _support_note_line(label, frame),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            color=PANEL_SPECS[label]["color"],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.80),
        )


def _draw_two_color_title(ax: plt.Axes, label: str, title: str, *, color: str) -> None:
    """Panel letter in black (matching every other lettered panel in the
    figure) with the descriptive title text in the population color, as two
    separate Text artists -- ax.set_title only takes one color for the
    whole string."""
    _fig4_panel_header.draw_panel_header(ax, label, title, y=1.020, title_color=color)


def _draw_pair_title(
    ax: plt.Axes,
    labels: tuple[str, str],
    *,
    panel_label: str | None = None,
    panel_title: str | None = None,
    panel_subtitle: str | None = None,
    use_middle_header: bool = False,
) -> None:
    if panel_label is not None and panel_title is not None:
        has_subtitle = bool(panel_subtitle)
        if has_subtitle or use_middle_header:
            _fig4_panel_header.draw_middle_row_header(
                ax,
                panel_label,
                panel_title,
                subtitle=panel_subtitle,
                title_linespacing=_fig4_panel_header.MIDDLE_ROW_TITLE_LINESPACING,
                subtitle_linespacing=1.02,
                subtitle_gap=0.022,
            )
        else:
            _fig4_panel_header.draw_panel_header(
                ax,
                panel_label,
                panel_title,
                y=1.045,
                title_linespacing=_fig4_panel_header.PANEL_TITLE_LINESPACING,
                title_y_offset_pt=_fig4_panel_header.TOP_ROW_TITLE_Y_OFFSET_PT,
            )
        return

    y = 1.055
    for row, label in enumerate(labels):
        spec = PANEL_SPECS[label]
        line_y = y - row * 0.075
        _fig4_panel_header.draw_panel_header(ax, label, spec["title"], y=line_y, title_color=spec["color"])


def _add_microsaccade_legend(ax: plt.Axes, *, color: str) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], color=color, marker="o", markerfacecolor="white", markeredgewidth=1.1, lw=1.4, label="drift only"
        ),
        Line2D([0], [0], color=color, marker="o", markerfacecolor=color, markeredgewidth=1.1, lw=1.4, label="microsaccade"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5.6,
        loc="lower right",
        handlelength=1.5,
        labelspacing=0.3,
        borderaxespad=0.2,
        handletextpad=0.4,
    )


def draw_panel(
    ax: plt.Axes,
    *,
    label: str,
    title: str,
    sf_group: str,
    relation: str,
    color: str,
    ylabel: str | None,
    values: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
    show_microsaccade_legend: bool = False,
) -> pd.DataFrame:
    values = load_panel_values() if values is None else values.copy()
    frame = values[values["sf_group"].eq(sf_group) & values["relation"].eq(relation)].copy()
    if frame.empty:
        raise ValueError(f"No values for sf_group={sf_group!r}, relation={relation!r}")

    plot_b_series(ax, frame, color=color)
    format_broken_path_axis(ax, xlim_right=path_xlimit_right(frame))
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*(shared_ylim(values) if ylim is None else ylim))
    _draw_two_color_title(ax, label, title, color=color)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2.0)
    _add_support_note(ax, frame)
    if show_microsaccade_legend:
        _add_microsaccade_legend(ax, color=color)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    return frame


def draw_pair_panel(
    ax: plt.Axes,
    *,
    labels: tuple[str, str],
    values: pd.DataFrame | None = None,
    ylim: tuple[float, float] | None = None,
    show_microsaccade_legend: bool = False,
    panel_label: str | None = None,
    panel_title: str | None = None,
    panel_subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel_x: float | None = None,
    use_middle_header: bool = False,
    draw_title: bool = True,
) -> list[tuple[str, pd.DataFrame]]:
    values = load_panel_values() if values is None else values.copy()
    frames: list[tuple[str, pd.DataFrame]] = []
    for label in labels:
        spec = PANEL_SPECS[label]
        frame = values[values["sf_group"].eq(spec["sf_group"]) & values["relation"].eq(spec["relation"])].copy()
        if frame.empty:
            raise ValueError(
                f"No values for sf_group={spec['sf_group']!r}, relation={spec['relation']!r}, label={label!r}"
            )
        plot_b_series(ax, frame, color=spec["color"])
        frames.append((label, frame))

    combined = pd.concat([frame for _, frame in frames], ignore_index=True)
    format_broken_path_axis(ax, xlim_right=path_xlimit_right(combined))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.set_ylim(*(shared_ylim(values) if ylim is None else ylim))
    ax.set_ylabel("SSI change (%)", labelpad=2.0)
    if ylabel_x is not None:
        ax.yaxis.set_label_coords(ylabel_x, 0.5)
    if draw_title:
        _draw_pair_title(
            ax,
            labels,
            panel_label=panel_label,
            panel_title=panel_title,
            panel_subtitle=panel_subtitle,
            use_middle_header=use_middle_header,
        )
    _add_pair_support_note(ax, frames)
    _add_split_sf_zero_anchor(ax)
    if show_microsaccade_legend:
        _add_microsaccade_legend(ax, color=INK)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)
    return frames


def build_preview_grid(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    """A 2x2 grid of all four source panels, for quick side-by-side
    iteration. Not part of the composed figure -- panels B and D are built
    by their own modules."""
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    values.to_csv(out_dir / "path_bins_preview_values.csv", index=False)
    (out_dir / "path_bins_preview_provenance.json").write_text(
        json.dumps(load_provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(4.85, 4.50), constrained_layout=True, sharey=False)
    top_ylim = shared_ylim_for(values, relations=("strong_contours_no_osi",))
    bottom_ylim = shared_ylim_for(values, relations=("contour_matched",))
    for ax, label in zip(axes.flat, ["B", "C", "E", "F"], strict=True):
        spec = PANEL_SPECS[label]
        ylim = top_ylim if label in {"B", "C"} else bottom_ylim
        draw_panel(ax, label=label, values=values, ylim=ylim, **spec)
    paths = {
        "png": out_dir / "path_bins_preview.png",
        "pdf": out_dir / "path_bins_preview.pdf",
        "svg": out_dir / "path_bins_preview.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    plt.close(fig)
    return paths


def build_pair_panel(
    labels: tuple[str, str],
    *,
    figsize: tuple[float, float],
    out_dir: Path = OUT_DIR,
    panel_label: str | None = None,
    panel_title: str | None = None,
    panel_subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel_x: float | None = None,
    axes_box: tuple[float, float, float, float] | None = None,
    ylim_pad_low: float = 0.12,
    ylim_pad_high: float = 0.14,
    tight_pad: float = 0.55,
    separate_header: bool = False,
    show_microsaccade_legend: bool | None = None,
) -> Path:
    """Render a source pair (B, C) or (E, F) as one shared plotting axis.

    Callers are the two panel modules, which own the layout arguments. The
    legend default reproduces the historical behaviour of keying off the source
    letters; passing it explicitly is preferred, so that which panel carries the
    legend is a stated choice rather than a consequence of a tuple comparison.
    """
    if show_microsaccade_legend is None:
        show_microsaccade_legend = labels == ("B", "C")
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    relations = tuple(dict.fromkeys(PANEL_SPECS[label]["relation"] for label in labels))
    ylim = shared_ylim_for(values, relations=relations, pad_low=ylim_pad_low, pad_high=ylim_pad_high)

    if axes_box is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    else:
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        ax = fig.add_axes(list(axes_box))
    draw_pair_panel(
        ax,
        labels=labels,
        values=values,
        ylim=ylim,
        show_microsaccade_legend=show_microsaccade_legend,
        panel_label=panel_label,
        panel_title=panel_title,
        panel_subtitle=panel_subtitle,
        xlabel=xlabel,
        ylabel_x=ylabel_x,
        use_middle_header=(axes_box is not None),
        draw_title=not separate_header,
    )
    if axes_box is None:
        fig.tight_layout(pad=tight_pad)
    elif separate_header:
        pass
    else:
        _fig4_panel_header.align_middle_row_xlabel(ax)

    if separate_header:
        header_ax = fig.add_axes([0, 0, 1, 1], zorder=20)
        header_ax.set_axis_off()
        _fig4_panel_header.draw_panel_header(
            header_ax,
            panel_label or labels[0],
            panel_title or PANEL_SPECS[labels[0]]["title"],
            y=TOP_ROW_PAIR_HEADER_Y,
            letter_x=TOP_ROW_PAIR_LETTER_X,
            letter_y_offset_pt=TOP_ROW_PAIR_LETTER_Y_OFFSET_PT,
            title_linespacing=_fig4_panel_header.PANEL_TITLE_LINESPACING,
            title_y_offset_pt=_fig4_panel_header.TOP_ROW_TITLE_Y_OFFSET_PT,
        )

    # Name by the DISPLAYED letter, not the source letters this pair was
    # combined from -- "panel_bc.pdf" for the panel a reader sees as "B" sent
    # you to the wrong file.
    out_path = out_dir / f"panel_{(panel_label or labels[0]).lower()}.pdf"
    if axes_box is None:
        fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    elif separate_header:
        fig.savefig(out_path, transparent=True)
    else:
        fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def build_single_panel(letter: str, *, figsize: tuple[float, float], out_dir: Path = OUT_DIR) -> Path:
    """v3 architecture: B/C/E/F each get their own independently-rendered
    figure at their measured size (see compose_ssi_figure_v3.py), instead
    of a shared 2x2 grid -- build_panel() above still makes that grid for
    quick side-by-side preview/iteration, unrelated to the real figure.
    """
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_panel_values()
    spec = PANEL_SPECS[letter]
    relations = ("strong_contours_no_osi",) if letter in {"B", "C"} else ("contour_matched",)
    ylim = shared_ylim_for(values, relations=relations)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    draw_panel(ax, label=letter, values=values, ylim=ylim, show_microsaccade_legend=(letter == "B"), **spec)
    fig.tight_layout(pad=0.55)

    out_path = out_dir / f"panel_{letter.lower()}.pdf"
    fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build_preview_grid(args.out_dir)
    for key in ("png", "pdf", "svg"):
        print(paths[key])


if __name__ == "__main__":
    main()
