#!/usr/bin/env python3
"""Option sheet comparing candidate dose axes for Panel G (aligned high-SF units).

Panel G currently uses unsigned component path as its x-axis. The behavior
bridge work (now _fig4_bridge*.py) found that path
is the axis with the biggest model-side effect but the one real behavior does
not track; RMS excursion and projected range are smaller model effects but are
exactly what the random-rotation null shows the animal's real trace-contour
matching is doing. This script renders all four candidate axes side by side,
in a style close to the real Panel G (_fig4_matched_bins_bracket.py), each
annotated with both pieces of evidence, so the choice of x-axis is a visible
comparison rather than a hidden decision.

This is a decision-support sheet, not a polished figure panel: it reuses the
already-computed panel_g_alternative_x_axes_diagnostic_* CSVs (aligned high-SF
only) plus the random-rotation match-null summary, and does not rerun any
bootstraps.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

from _fig4_geometry_story import (
    _add_bracket,
    _format_p_label,
)
from _fig4_matched_bins_bracket import _add_vertical_bracket

VALUES_CSV = _paths.PATH_BINS_VALUES_CSV
LAST_BIN_CONTRASTS_CSV = _paths.PATH_BINS_LAST_BIN_CONTRASTS_CSV
TRACE_BANK_REFERENCE_CSV = _paths.PATH_BINS_TRACE_BANK_REFERENCE_CSV
POPULATIONS_CSV = _paths.PATH_BINS_POPULATIONS_CSV
MATCH_NULL_SUMMARY_CSV = _paths.BRIDGE_MATCH_NULL_SUMMARY_CSV

OUT_DIR = _paths.PANELS_DIR
OUT_STEM = "_fig4_option_sheet"

POPULATION_KEY = "high_sf_aligned"
BRIDGE_SUBSET_KEY = "coh_ge_0p2"

ORANGE = "#D55E00"
DIRECTION_COLOR = ORANGE
INK = "#111111"
GRAY = "#6B6F75"
GRID = "#E7E7E7"

METRIC_ORDER = ("component_path", "component_rms", "component_range", "path_per_range")
METRIC_UNITS = {
    "component_path": "arcmin",
    "component_rms": "arcmin",
    "component_range": "arcmin",
    "path_per_range": "ratio",
}
METRIC_XLABEL = {
    "component_path": "unsigned component path (arcmin)",
    "component_rms": "component RMS excursion (arcmin)",
    "component_range": "component peak-to-peak range (arcmin)",
    "path_per_range": "path / range (tortuosity proxy)",
}
# Bins cluster tightly at the low end with one long tail bin (the model curves
# are dose-response, not evenly sampled), so a broken log x-axis is what
# actually separates the points instead of stacking them against the left
# edge. This is the same _x_broken_log convention the real Panel G uses (a
# fixed "0" anchor for the artificially stabilized/zero-motion condition,
# then a log-spaced axis for the real-motion bins) -- min_pos/max_pos/ticks
# for component_path are the exact values _fig4_matched_bins_bracket.py
# uses, so Option 1 reproduces the real Panel G axis exactly.
METRIC_AXIS = {
    "component_path": {"min_pos": 45.0, "max_pos": 180.0, "ticks": [0, 50, 65, 90, 120, 160]},
    "component_rms": {"min_pos": 0.9, "max_pos": 8.0, "ticks": [0, 1, 2, 3, 5, 7]},
    "component_range": {"min_pos": 4.0, "max_pos": 28.0, "ticks": [0, 5, 10, 15, 20, 25]},
    "path_per_range": {"min_pos": 5.5, "max_pos": 17.0, "ticks": [0, 6, 8, 10, 12, 16]},
}
COMPONENT_STYLE = {
    "across": {"label": "across (contour-normal)", "linestyle": "-", "marker": "o"},
    "along": {"label": "along (contour-parallel)", "linestyle": (0, (4.2, 2.0)), "marker": "s"},
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _broken_log_map(
    values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float, zero_gap: float = 1.0, span: float = 5.1
) -> np.ndarray:
    """Same convention as the real Panel G's _x_broken_log, but with the

    distance reserved for the zero-anchor/break (`zero_gap`) exposed as a
    parameter. The real G hardcodes this at 1.0; for metrics whose real bins
    all sit close together (e.g. RMS in arcmin vs. path in arcmin), that
    fixed gap eats a disproportionate share of the axis, so it's tunable
    here rather than baked in.
    """
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    mapped[positive] = zero_gap + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _xlim_right(
    ticks: list[float], *, min_pos: float, max_pos: float, zero_gap: float = 1.0, span: float = 5.1, pad: float = 0.50
) -> float:
    tick_x = _broken_log_map(ticks, min_pos=min_pos, max_pos=max_pos, zero_gap=zero_gap, span=span)
    return float(np.nanmax(tick_x) + pad)


def _format_axis_local(ax: plt.Axes, *, ticks: list[float], min_pos: float, max_pos: float, zero_gap: float, span: float) -> None:
    ax.set_xticks(_broken_log_map(ticks, min_pos=min_pos, max_pos=max_pos, zero_gap=zero_gap, span=span))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])


def _draw_broken_axis_break(ax: plt.Axes, *, zero_gap: float) -> None:
    """Diagonal double-tick break mark, fixed visually across broken axes."""
    break_center = 0.545 * zero_gap
    tick_half = 0.040
    tick_offset = 0.048
    tick_half_height = 0.035
    left_slash_x = break_center - tick_offset
    right_slash_x = break_center + tick_offset
    for text in list(ax.texts):
        if text.get_text() == "//":
            text.remove()
    ax.spines["bottom"].set_visible(False)
    trans = ax.get_xaxis_transform()
    x_left, x_right = ax.get_xlim()
    ax.plot([x_left, left_slash_x], [0.0, 0.0], transform=trans, color="black", lw=0.8, clip_on=False, zorder=10)
    ax.plot([right_slash_x, x_right], [0.0, 0.0], transform=trans, color="black", lw=0.8, clip_on=False, zorder=10)
    for offset in (-tick_offset, tick_offset):
        ax.plot(
            [break_center + offset - tick_half, break_center + offset + tick_half],
            [-tick_half_height, tick_half_height],
            transform=trans,
            color="black",
            lw=1.05,
            clip_on=False,
            solid_capstyle="butt",
            zorder=11,
        )


def _draw_dose_panel(
    ax: plt.Axes,
    *,
    metric_family: str,
    values: pd.DataFrame,
    reference: pd.DataFrame,
    populations: pd.DataFrame,
    last_bin_contrasts: pd.DataFrame,
    ylim: tuple[float, float],
    exclude_last_bins: int = 0,
    axis_override: dict | None = None,
    final_bracket_x_offset: float = 0.0,
) -> str | None:
    """Draw one candidate dose-axis panel; returns a note if bins were dropped."""
    axis_spec = axis_override if axis_override is not None else METRIC_AXIS[metric_family]
    min_pos = axis_spec["min_pos"]
    max_pos = axis_spec["max_pos"]
    ticks = axis_spec["ticks"]
    zero_gap = axis_spec.get("zero_gap", 1.0)
    span = axis_spec.get("span", 5.1)

    # Ylim must be fixed before the brackets are drawn: their vertical
    # placement (near-zero bracket height, final-bin bracket text offset)
    # is computed from the axes' final data range, exactly as the real
    # Panel G fixes its ylim before drawing its brackets.
    ax.set_ylim(*ylim)

    frame = values[values["metric_family"].eq(metric_family) & values["population_key"].eq(POPULATION_KEY)].copy()
    dropped_note: str | None = None
    last_visible_bin_order = int(frame["component_bin_order"].max()) if not frame.empty else None
    if exclude_last_bins > 0 and not frame.empty:
        max_bin_order = int(frame["component_bin_order"].max())
        keep_through = max_bin_order - exclude_last_bins
        dropped = frame[frame["component_bin_order"] > keep_through]
        if not dropped.empty:
            dose_lo = float(dropped["component_min"].min())
            unit = METRIC_UNITS[metric_family]
            dropped_note = f"tail omitted (>{dose_lo:.1f} {unit})"
        frame = frame[frame["component_bin_order"] <= keep_through]
        last_visible_bin_order = keep_through

    ref_row = reference[reference["metric_family"].eq(metric_family)]
    if not ref_row.empty:
        q25, q75 = _broken_log_map(
            [float(ref_row["q25"].iloc[0]), float(ref_row["q75"].iloc[0])],
            min_pos=min_pos,
            max_pos=max_pos,
            zero_gap=zero_gap,
            span=span,
        )
        ax.axvspan(float(q25), float(q75), facecolor="#7c7c7c", edgecolor="none", alpha=0.12, zorder=0)

    first_rows: dict[str, pd.Series] = {}
    last_rows: dict[str, pd.Series] = {}
    for component, style in COMPONENT_STYLE.items():
        sub = frame[frame["component"].eq(component)].sort_values("component_bin_order")
        if sub.empty:
            continue
        x = _broken_log_map(sub["plot_median"], min_pos=min_pos, max_pos=max_pos, zero_gap=zero_gap, span=span)
        y = sub["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        ci_lo = sub["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
        ci_hi = sub["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
        yerr = np.vstack([y - ci_lo, ci_hi - y])
        ax.plot(x, y, color=DIRECTION_COLOR, linestyle=style["linestyle"], linewidth=1.6, label=style["label"], zorder=3)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=DIRECTION_COLOR,
            linestyle="none",
            marker=style["marker"],
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.05,
            elinewidth=0.9,
            capsize=0,
            zorder=4,
        )
        # Artificially stabilized / zero-motion anchor: by construction
        # ssi_percent_vs_cell_baseline is 0 at zero real motion.
        ax.scatter(
            [0.0],
            [0.0],
            marker=style["marker"],
            s=26,
            facecolors="white",
            edgecolors=DIRECTION_COLOR,
            linewidths=1.2,
            zorder=5,
        )
        first_rows[component] = sub.iloc[0]
        last_rows[component] = sub.iloc[-1]

    _format_axis_local(ax, ticks=ticks, min_pos=min_pos, max_pos=max_pos, zero_gap=zero_gap, span=span)
    ax.set_xlim(-0.12, _xlim_right(ticks, min_pos=min_pos, max_pos=max_pos, zero_gap=zero_gap, span=span))
    _draw_broken_axis_break(ax, zero_gap=zero_gap)

    ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
    ax.grid(axis="y", color=GRID, lw=0.75)
    _clean_axis(ax)
    ax.set_xlabel(METRIC_XLABEL[metric_family])

    if {"across", "along"}.issubset(first_rows):
        across_first = first_rows["across"]
        along_first = first_rows["along"]
        x1 = float(
            _broken_log_map(
                [max(float(across_first["plot_median"]), float(along_first["plot_median"]))],
                min_pos=min_pos,
                max_pos=max_pos,
                zero_gap=zero_gap,
                span=span,
            )[0]
        )
        y_lo, y_hi = ax.get_ylim()
        y_span = max(y_hi - y_lo, 1.0)
        text = (
            "near 0\n"
            f"across {_format_p_label(float(across_first['population_delta_p_image_bootstrap_sign']))}\n"
            f"along {_format_p_label(float(along_first['population_delta_p_image_bootstrap_sign']))}"
        )
        _add_bracket(
            ax,
            x0=0.0,
            x1=x1,
            y=y_hi - 0.185 * y_span,
            text=text,
            color=DIRECTION_COLOR,
            text_x=x1 + 0.07 * zero_gap,
            text_ha="left",
        )

    # The bracket's pp/p-value comes from last_bin_contrasts, which is
    # computed at both the true final bin and the second-to-last bin (see
    # panel_g_alternative_x_axes_diagnostic.py::_compute_family) -- pick
    # whichever one matches this panel's actual rightmost *visible* bin, so
    # a panel that hides the final bin (EXCLUDE_LAST_BINS>0) still gets a
    # correctly-labeled bracket instead of losing it entirely.
    matching_contrast = last_bin_contrasts.loc[
        (last_bin_contrasts["population_key"] == POPULATION_KEY)
        & (last_bin_contrasts["metric_family"] == metric_family)
        & (last_bin_contrasts["component_bin_order"] == last_visible_bin_order)
    ]
    if not matching_contrast.empty and {"across", "along"}.issubset(last_rows):
        contrast_row = matching_contrast.iloc[0]
        pp = float(contrast_row["across_minus_along_percent_point"])
        p_val = float(contrast_row["contrast_p_image_bootstrap_sign"])
        x_last = float(
            _broken_log_map(
                [float(last_rows["across"]["plot_median"])], min_pos=min_pos, max_pos=max_pos, zero_gap=zero_gap, span=span
            )[0]
        )
        _add_vertical_bracket(
            ax,
            x=x_last + final_bracket_x_offset,
            y0=float(last_rows["across"]["ssi_percent_vs_cell_baseline"]),
            y1=float(last_rows["along"]["ssi_percent_vs_cell_baseline"]),
            label=f"{pp:+.1f} pp\n{_format_p_label(p_val)}",
            color=DIRECTION_COLOR,
        )

    # Top-right, matching the real Panel G, so it never collides with a
    # bottom-left across/along legend when this panel is rendered solo.
    pop_row = populations[populations["population_key"].eq(POPULATION_KEY)].iloc[0]
    ax.text(
        0.97,
        0.93,
        f"{int(pop_row['n_selected_units'])} units\n{int(pop_row['n_selected_unit_image_pairs'])} pairs",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    )

    return dropped_note


def _evidence_chip_text(metric_family: str, contrasts: pd.DataFrame, bridge: pd.DataFrame) -> str:
    # Always the true final bin's contrast, regardless of what any one
    # panel chooses to hide from view -- this chip is a fixed decision-support
    # summary, not tied to a specific panel's visible x-range.
    family_contrasts = contrasts.loc[
        (contrasts["population_key"] == POPULATION_KEY) & (contrasts["metric_family"] == metric_family)
    ]
    model_row = family_contrasts.loc[family_contrasts["component_bin_order"].idxmax()]
    model_pp = float(model_row["across_minus_along_percent_point"])
    model_p = float(model_row["contrast_p_image_bootstrap_sign"])

    bridge_rows = bridge.loc[
        (bridge["population_key"] == POPULATION_KEY)
        & (bridge["subset_key"] == BRIDGE_SUBSET_KEY)
        & (bridge["score_type"] == "component_mean_marginal")
        & (bridge["metric_family"] == metric_family)
    ]
    if bridge_rows.empty:
        bridge_text = "no bridge estimate"
    else:
        bridge_row = bridge_rows.iloc[0]
        bridge_pp = float(bridge_row["observed_minus_rotated_session_mean"])
        bridge_p = float(bridge_row["p_rotation_two_sided"])
        bridge_text = f"{bridge_pp:+.3f} pp vs. random rotation, {_format_p_label(bridge_p)}"

    return f"Model:  {model_pp:+.1f} pp, {_format_p_label(model_p)}\nBridge: {bridge_text}"


def build(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    configure_matplotlib()
    values = pd.read_csv(VALUES_CSV)
    last_bin_contrasts = pd.read_csv(LAST_BIN_CONTRASTS_CSV)
    reference = pd.read_csv(TRACE_BANK_REFERENCE_CSV)
    populations = pd.read_csv(POPULATIONS_CSV)
    bridge = pd.read_csv(MATCH_NULL_SUMMARY_CSV)

    fig = plt.figure(figsize=(13.0, 8.6))
    outer = fig.add_gridspec(1, 4, left=0.045, right=0.955, top=0.845, bottom=0.185, wspace=0.20)

    axes = [fig.add_subplot(outer[0, idx]) for idx in range(4)]
    y_all = values[values["population_key"].eq(POPULATION_KEY)]
    y_lo = float(min(0.0, y_all["population_delta_percent_ci95_low_image_boot"].min()))
    y_hi = float(max(0.0, y_all["population_delta_percent_ci95_high_image_boot"].max()))
    y_span = y_hi - y_lo
    shared_ylim = (y_lo - 0.16 * y_span, y_hi + 0.40 * y_span)

    option_letters = ("Option 1", "Option 2", "Option 3", "Option 4")
    for ax, metric_family, letter in zip(axes, METRIC_ORDER, option_letters, strict=True):
        _draw_dose_panel(
            ax,
            metric_family=metric_family,
            values=values,
            reference=reference,
            populations=populations,
            last_bin_contrasts=last_bin_contrasts,
            ylim=shared_ylim,
        )
        current = " (current G)" if metric_family == "component_path" else ""
        ax.set_title(f"{letter}{current}", loc="left", fontsize=9.4, fontweight="bold", pad=6)

    axes[0].set_ylabel("SSI change (%)\nvs. cell baseline")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper right", handlelength=1.8)

    for ax, metric_family in zip(axes, METRIC_ORDER, strict=True):
        bbox = ax.get_position()
        fig.text(
            bbox.x0,
            bbox.y0 - 0.05,
            _evidence_chip_text(metric_family, last_bin_contrasts, bridge),
            ha="left",
            va="top",
            fontsize=6.9,
            color="#333333",
            linespacing=1.5,
        )

    fig.text(
        0.045,
        0.965,
        "Panel G option sheet: which dose axis for aligned high-SF units?",
        fontsize=15.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.045,
        0.925,
        "Same aligned high-SF units (22 units, 356 pairs) and across/along decomposition as the current Panel G; only the x-axis dose metric changes.\n"
        "Model = across−along model contrast at the final dose bin. Bridge = real trace-contour matching minus random-rotation control,\n"
        "aligned high-SF units, local edge coherence ≥ 0.2 (behavior_model_bridge_random_rotation_match_null_summary.csv).",
        fontsize=8.6,
        color="#4b5563",
        ha="left",
        va="top",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{OUT_STEM}.png"
    pdf = out_dir / f"{OUT_STEM}.pdf"
    svg = out_dir / f"{OUT_STEM}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf, "svg": svg}


GOOD_G_FIGSIZE = (2.45, 2.35)  # matches _fig4_matched_bins_bracket.build_panel


def build_option_at_g_size(
    metric_family: str,
    *,
    option_label: str,
    out_stem: str,
    out_dir: Path = OUT_DIR,
    figsize: tuple[float, float] = GOOD_G_FIGSIZE,
    exclude_last_bins: int = 0,
    axis_override: dict | None = None,
) -> dict[str, Path]:
    """Render one candidate axis at the exact size the real Panel G is drawn at.

    The option sheet stretches each panel out to a full column for
    legibility while comparing; this renders a single option at
    _fig4_matched_bins_bracket.py's actual (2.45in x 2.35in) footprint so
    it can be dropped straight into the main ssi_figure_v2 gridspec slot for
    a like-for-like look.

    exclude_last_bins drops the N highest dose bins from both the plot and
    the y-axis range calculation -- useful when one long tail bin (few
    windows, wide CI) is stretching the y-axis so much that the low-dose
    bins, where most of the data actually is, collapse to a flat line.
    """
    configure_matplotlib()
    values = pd.read_csv(VALUES_CSV)
    last_bin_contrasts = pd.read_csv(LAST_BIN_CONTRASTS_CSV)
    reference = pd.read_csv(TRACE_BANK_REFERENCE_CSV)
    populations = pd.read_csv(POPULATIONS_CSV)

    frame = values[values["metric_family"].eq(metric_family) & values["population_key"].eq(POPULATION_KEY)]
    if exclude_last_bins > 0 and not frame.empty:
        keep_through = int(frame["component_bin_order"].max()) - exclude_last_bins
        frame = frame[frame["component_bin_order"] <= keep_through]
    vals = [0.0]
    for col in [
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ]:
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)
    # Same pad convention as the real Panel G's _set_panel_ylim.
    ylim = (lo - 0.16 * span, hi + 0.36 * span)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    dropped_note = _draw_dose_panel(
        ax,
        metric_family=metric_family,
        values=values,
        reference=reference,
        populations=populations,
        last_bin_contrasts=last_bin_contrasts,
        ylim=ylim,
        exclude_last_bins=exclude_last_bins,
        axis_override=axis_override,
    )
    ax.set_title(f"G  {option_label}", loc="left", fontsize=8.8, fontweight="bold", pad=4, color=INK)
    if dropped_note is not None:
        # Appended to the xlabel (rather than a free-floating fig.text) so
        # tight_layout accounts for it and it can't get clipped at this
        # figure's small, fixed size.
        ax.set_xlabel(f"{ax.get_xlabel()}\n{dropped_note}")
    ax.set_ylabel("SSI change (%)")
    ax.tick_params(labelsize=6.8)
    ax.xaxis.label.set_size(6.9)
    ax.yaxis.label.set_size(7.0)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        short_labels = ["across" if "across" in item else "along" for item in labels]
        ax.legend(
            handles,
            short_labels,
            frameon=False,
            fontsize=5.9,
            loc="lower left",
            handlelength=1.8,
            borderaxespad=0.2,
        )

    fig.tight_layout(pad=0.55)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": out_dir / f"{out_stem}.png",
        "pdf": out_dir / f"{out_stem}.pdf",
        "svg": out_dir / f"{out_stem}.svg",
    }
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"], dpi=300)
    fig.savefig(paths["svg"], dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    paths = build()
    for path in paths.values():
        print(path)

    option2_paths = build_option_at_g_size(
        "component_rms",
        option_label="RMS Excursion (Option 2)",
        out_stem="panel_g_option2_rms_at_g_size",
    )
    for path in option2_paths.values():
        print(path)

    option2_no_tail_paths = build_option_at_g_size(
        "component_rms",
        option_label="RMS Excursion (Option 2)",
        out_stem="panel_g_option2_rms_at_g_size_no_tail",
        exclude_last_bins=1,
        axis_override={"min_pos": 0.9, "max_pos": 3.3, "ticks": [0, 1, 2, 3], "zero_gap": 0.5},
    )
    for path in option2_no_tail_paths.values():
        print(path)


if __name__ == "__main__":
    main()
