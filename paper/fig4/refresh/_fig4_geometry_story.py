#!/usr/bin/env python3
"""Assemble the reordered BackImage real-trace contour-geometry story figure."""

from __future__ import annotations

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
import _fig4_paths as _paths
from _fig4_brackets import add_bracket as _add_bracket, format_p_label as _format_p_label

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import patches, transforms


# These are refresh-path write targets. They were CWD-relative, which silently
# scattered outputs wherever the script happened to be run from; anchoring them
# keeps every refresh run writing to the same place.
SUMMARY_DIR = _paths.FIG_DIR / "schematic_pathlength_summary" / "unit_first_and_population"
MATRIX_DIR = _paths.TRACE_BANK_MERGED_DIR
CONDITION_DIR = _paths.FIG_DIR / "trace_conditioning"
FIG_DIR = SUMMARY_DIR / "figures"
OUT_DIR = _paths.FIG_DIR / "plot_collections"
MOVIE_TABLE = _paths.CACHE_DIR / "fig4_phase1_movie_analysis_table.csv"
TRACE_CONTEXT_REFERENCE = _paths.CACHE_DIR / "fig4_trace_bank_metadata_filtered.csv"

SF_ORDER = ("low_sf", "middle_sf", "high_sf")
SF_COLORS = {"low_sf": "#0072B2", "middle_sf": "#009E73", "high_sf": "#D55E00"}
SF_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
PANEL_B_SF_ORDER = ("low_mid_sf", "high_sf")
PANEL_B_SF_COLORS = {"low_mid_sf": "#0072B2", "high_sf": "#D55E00"}
PANEL_B_SF_LABELS = {"low_mid_sf": "Low SF", "high_sf": "High SF"}
AXIS_CLASS_STYLES = {
    "across_contour_axis": ("across contour", "0.18", "-", "o", 4),
    "along_contour_axis": ("along contour", "0.18", "--", "s", 4),
    "oblique": ("oblique", "0.45", "-.", "D", 3),
    "low_anisotropy": ("weak directional bias", "0.68", ":", "^", 2),
}
AXIS_CLASS_ORDER = ("along_contour_axis", "oblique", "across_contour_axis", "low_anisotropy")
COMPONENT_STYLES = {
    "across_path_arcmin": ("across contour", "-", "o"),
    "along_path_arcmin": ("along contour", (0, (4.2, 2.0)), "s"),
}
LOWER_MIN_POS = 45.0
LOWER_MAX_POS = 180.0
LOWER_TICKS = [0, 50, 65, 90, 120, 160]


def _add_percent_change(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    group_cols = ["relation", "sf_group"]
    if "component_metric" in df.columns:
        group_cols.append("component_metric")
    for _, group in df.groupby(group_cols, sort=False):
        baseline = float(group.loc[group["context"].eq("stabilized"), "population_ssi_bits_per_spike"].iloc[0])
        copy = group.copy()
        copy["population_ssi_percent_vs_stabilized"] = 100.0 * copy["population_ssi_delta_vs_stabilized"] / baseline
        if {
            "population_delta_ci95_low_image_boot",
            "population_delta_ci95_high_image_boot",
        }.issubset(copy.columns):
            copy["population_delta_percent_ci95_low_image_boot"] = (
                100.0 * copy["population_delta_ci95_low_image_boot"] / baseline
            )
            copy["population_delta_percent_ci95_high_image_boot"] = (
                100.0 * copy["population_delta_ci95_high_image_boot"] / baseline
            )
        out.append(copy)
    return pd.concat(out, ignore_index=True)


def _combined_low_mid_panel_b(total: pd.DataFrame) -> pd.DataFrame:
    """Create a figure-only low-SF group by pooling low and middle SF units."""
    low_mid = total[total["sf_group"].isin(["low_sf", "middle_sf"])].copy()
    group_cols = [
        "relation",
        "relation_label",
        "context",
        "context_label",
        "path_bin",
        "path_bin_order",
        "path_median_arcmin",
    ]
    agg = {
        "n_traces": "first",
        "n_units": "sum",
        "n_images_contributing": "max",
        "n_movie_samples": "sum",
        "information_numerator_bits": "sum",
        "expected_spikes": "sum",
    }
    combined = low_mid.groupby(group_cols, dropna=False, sort=False).agg(agg).reset_index()
    combined["sf_group"] = "low_mid_sf"
    combined["population_ssi_bits_per_spike"] = (
        combined["information_numerator_bits"] / combined["expected_spikes"]
    )
    rows = []
    for _, group in combined.groupby("relation", sort=False):
        baseline = float(
            group.loc[group["context"].eq("stabilized"), "population_ssi_bits_per_spike"].iloc[0]
        )
        copy = group.copy()
        copy["population_ssi_delta_vs_stabilized"] = (
            copy["population_ssi_bits_per_spike"] - baseline
        )
        copy["population_ssi_percent_vs_stabilized"] = (
            100.0 * copy["population_ssi_delta_vs_stabilized"] / baseline
        )
        rows.append(copy)
    high = total[total["sf_group"].eq("high_sf")].copy()
    return pd.concat(rows + [high], ignore_index=True, sort=False)


def _sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def _load_axis_summary() -> pd.DataFrame:
    movie = pd.read_csv(
        MOVIE_TABLE,
        usecols=[
            "trace_image_axis_class",
            "trace_path_length_bin",
            "rendered_path_length_arcmin",
            "population_ssi",
        ],
    ).dropna(subset=["trace_image_axis_class", "trace_path_length_bin"])
    rows = []
    for (axis_class, trace_bin), group in movie.groupby(
        ["trace_image_axis_class", "trace_path_length_bin"], sort=True
    ):
        rows.append(
            {
                "trace_image_axis_class": str(axis_class),
                "trace_path_length_bin": str(trace_bin),
                "path_median_arcmin": float(np.nanmedian(group["rendered_path_length_arcmin"])),
                "mean_ssi": float(np.nanmean(group["population_ssi"])),
                "sem_ssi": _sem(group["population_ssi"]),
                "n_movies": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _load_trace_context() -> pd.DataFrame:
    if not TRACE_CONTEXT_REFERENCE.exists():
        return pd.DataFrame()
    reference = pd.read_csv(TRACE_CONTEXT_REFERENCE)
    if "rendered_path_length_arcmin" not in reference.columns:
        return pd.DataFrame()
    if "has_microsaccade" in reference.columns:
        has_ms = reference["has_microsaccade"].fillna(False).astype(bool)
    elif "rendered_n_microsaccade_events" in reference.columns:
        has_ms = pd.to_numeric(reference["rendered_n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    elif "n_microsaccade_events" in reference.columns:
        has_ms = pd.to_numeric(reference["n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    else:
        has_ms = pd.Series(False, index=reference.index)
    work = pd.DataFrame(
        {
            "rendered_path_length_arcmin": pd.to_numeric(
                reference["rendered_path_length_arcmin"], errors="coerce"
            ),
            "has_microsaccade": has_ms,
        }
    ).dropna(subset=["rendered_path_length_arcmin"])
    rows = []
    for has_microsaccade, label, display_label in [
        (False, "no_microsaccade", "drift-only"),
        (True, "microsaccade", "microsaccade"),
    ]:
        values = work.loc[work["has_microsaccade"].eq(has_microsaccade), "rendered_path_length_arcmin"].to_numpy(
            dtype=float
        )
        if values.size == 0:
            continue
        rows.append(
            {
                "trace_path_context": label,
                "display_label": display_label,
                "n_traces": int(values.size),
                "q25_arcmin": float(np.nanpercentile(values, 25.0)),
                "median_arcmin": float(np.nanmedian(values)),
                "q75_arcmin": float(np.nanpercentile(values, 75.0)),
            }
        )
    return pd.DataFrame(rows)


def _load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = _add_percent_change(pd.read_csv(SUMMARY_DIR / "spike_weighted_population_summary.csv"))
    comp = _add_percent_change(pd.read_csv(SUMMARY_DIR / "spike_weighted_population_component_summary.csv"))
    axis_summary = _load_axis_summary()
    trace_context = _load_trace_context()
    return total, comp, axis_summary, trace_context


def _x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _format_axis(
    ax: plt.Axes,
    *,
    ticks: list[float],
    min_pos: float,
    max_pos: float,
    show_xlabel: bool = True,
) -> None:
    ax.set_xlim(-0.12, 5.35)
    ax.set_xticks(_x_broken_log(ticks, min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.8)


def _add_trace_context_bands(ax: plt.Axes, context: pd.DataFrame, *, include_legend: bool = True) -> None:
    if context.empty:
        return
    styles = {
        "no_microsaccade": {
            "color": "#8c8c8c",
            "alpha": 0.24,
            "line_alpha": 0.62,
            "linestyle": "-",
            "y0": 0.940,
            "y1": 0.982,
        },
        "microsaccade": {
            "color": "#5f5f5f",
            "alpha": 0.20,
            "line_alpha": 0.70,
            "linestyle": "--",
            "y0": 0.888,
            "y1": 0.930,
        },
    }
    strip_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    legend_handles: list[patches.Patch] = []
    for row in context.sort_values("median_arcmin").itertuples(index=False):
        key = str(getattr(row, "trace_path_context", "trace_path"))
        style = styles.get(key, styles["no_microsaccade"])
        low = float(getattr(row, "q25_arcmin", np.nan))
        high = float(getattr(row, "q75_arcmin", np.nan))
        median = float(getattr(row, "median_arcmin", np.nan))
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            continue
        y0 = float(style["y0"])
        y1 = float(style["y1"])
        ax.add_patch(
            patches.Rectangle(
                (low, y0),
                high - low,
                y1 - y0,
                transform=strip_transform,
                facecolor=str(style["color"]),
                edgecolor="none",
                alpha=float(style["alpha"]),
                zorder=0,
            )
        )
        if math.isfinite(median):
            ax.plot(
                [median, median],
                [y0, y1],
                transform=strip_transform,
                color=str(style["color"]),
                alpha=float(style["line_alpha"]),
                linestyle=str(style["linestyle"]),
                linewidth=0.9,
                zorder=1,
            )
        if include_legend:
            display = str(getattr(row, "display_label", key)).replace("_", " ")
            n_traces = int(getattr(row, "n_traces", 0))
            legend_handles.append(
                patches.Patch(
                    facecolor=str(style["color"]),
                    alpha=float(style["alpha"]),
                    edgecolor="none",
                    label=f"{display} q25-q75 (n={n_traces})",
                )
            )
    if include_legend and legend_handles:
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=legend_handles + existing_handles,
            labels=[h.get_label() for h in legend_handles] + existing_labels,
            frameon=False,
            fontsize=7.7,
            loc="lower right",
        )


def _plot_path_series(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    x_col: str,
    color: str,
    min_pos: float,
    max_pos: float,
    label: str | None = None,
    linestyle: str = "-",
    marker: str = "o",
    linewidth: float = 1.9,
    include_microsaccade: bool = True,
) -> None:
    zero = rows[rows["context"].eq("stabilized")]
    drift = rows[rows["context"].eq("drift_only")].sort_values(x_col)
    ms = rows[rows["context"].eq("microsaccade")].sort_values(x_col)
    if not zero.empty and not drift.empty:
        joined = pd.concat([zero.iloc[:1], drift], ignore_index=True)
        ax.plot(
            _x_broken_log(joined[x_col], min_pos=min_pos, max_pos=max_pos),
            joined["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    elif not drift.empty:
        ax.plot(
            _x_broken_log(drift[x_col], min_pos=min_pos, max_pos=max_pos),
            drift["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    groups = [(drift, False)]
    if include_microsaccade:
        groups.append((ms, True))
    for plot_rows, filled in groups:
        if plot_rows.empty:
            continue
        ax.plot(
            _x_broken_log(plot_rows[x_col], min_pos=min_pos, max_pos=max_pos),
            plot_rows["population_ssi_percent_vs_stabilized"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.scatter(
            _x_broken_log(plot_rows[x_col], min_pos=min_pos, max_pos=max_pos),
            plot_rows["population_ssi_percent_vs_stabilized"],
            marker=marker,
            s=26,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.35,
            zorder=4,
        )
    if not zero.empty:
        ax.scatter(
            [0.0],
            [0.0],
            marker=marker,
            s=28,
            facecolors="white",
            edgecolors=color,
            linewidths=1.35,
            zorder=5,
        )


def _first_drift_row(rows: pd.DataFrame, x_col: str) -> pd.Series | None:
    drift = rows[rows["context"].eq("drift_only")].sort_values(x_col)
    if drift.empty:
        return None
    return drift.iloc[0]


def _plot_component_series_with_ci(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    color: str,
    min_pos: float,
    max_pos: float,
    label: str,
    linestyle: str | tuple[int, tuple[float, ...]],
    marker: str,
) -> str | None:
    zero = rows[rows["context"].eq("stabilized")]
    drift = rows[rows["context"].eq("drift_only")].sort_values("component_median_arcmin")
    if zero.empty or drift.empty:
        return None
    drift_x = _x_broken_log(drift["component_median_arcmin"], min_pos=min_pos, max_pos=max_pos)
    drift_y = drift["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
    ax.plot(
        drift_x,
        drift_y,
        color=color,
        linestyle=linestyle,
        linewidth=2.1,
        label=label,
        zorder=3,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker=marker,
        s=30,
        facecolors="white",
        edgecolors=color,
        linewidths=1.35,
        zorder=5,
    )
    ci_low = drift["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
    ci_high = drift["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
    yerr = np.vstack([drift_y - ci_low, ci_high - drift_y])
    ax.errorbar(
        drift_x,
        drift_y,
        yerr=yerr,
        color=color,
        linestyle="none",
        marker=marker,
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linewidth=1.6,
        elinewidth=1.2,
        capsize=2.0,
        zorder=4,
    )
    first = drift.iloc[0]
    return f"{label}: {_format_p_label(float(first['population_delta_p_image_bootstrap_sign']))}"


def _plot_total_series_with_ci(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    min_pos: float,
    max_pos: float,
) -> str | None:
    zero = rows[rows["context"].eq("stabilized")]
    drift = rows[rows["context"].eq("drift_only")].sort_values("path_median_arcmin")
    if zero.empty or drift.empty:
        return None
    drift_x = _x_broken_log(drift["path_median_arcmin"], min_pos=min_pos, max_pos=max_pos)
    drift_y = drift["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
    ax.plot(
        drift_x,
        drift_y,
        color="0.38",
        linestyle="-",
        linewidth=2.0,
        marker="D",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.15,
        label="full trajectory",
        zorder=2,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="D",
        s=28,
        facecolors="white",
        edgecolors="0.38",
        linewidths=1.15,
        zorder=5,
    )
    ci_low = drift["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
    ci_high = drift["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
    yerr = np.vstack([drift_y - ci_low, ci_high - drift_y])
    ax.errorbar(
        drift_x,
        drift_y,
        yerr=yerr,
        color="0.38",
        linestyle="none",
        marker="D",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.10,
        linewidth=1.4,
        elinewidth=1.05,
        capsize=2.0,
        zorder=2,
    )
    first = drift.iloc[0]
    return f"full trajectory: {_format_p_label(float(first['population_delta_p_image_bootstrap_sign']))}"


def _panel_a(ax: plt.Axes, axis_summary: pd.DataFrame, trace_context: pd.DataFrame) -> None:
    _add_trace_context_bands(ax, trace_context, include_legend=False)
    for axis_class in AXIS_CLASS_ORDER:
        label, color, linestyle, marker, zorder = AXIS_CLASS_STYLES[axis_class]
        rows = axis_summary[axis_summary["trace_image_axis_class"].eq(axis_class)].sort_values("path_median_arcmin")
        if rows.empty:
            continue
        x = rows["path_median_arcmin"].to_numpy(dtype=float)
        y = rows["mean_ssi"].to_numpy(dtype=float)
        err = rows["sem_ssi"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.25,
            linewidth=1.8,
            capsize=2,
            alpha=0.95,
            zorder=zorder,
            label=label,
        )
    _style_axis(ax)
    ax.set_title("Information gain varies with gaze trajectory relative to local contours", fontsize=11.3, pad=8)
    ax.set_ylabel("population SSI (bits/spike)")
    ax.set_xlabel("trajectory path length bin median (arcmin)")
    ax.set_xlim(86, 171)
    ax.set_xticks([90, 105, 113, 123, 137, 163])
    y = axis_summary["mean_ssi"].to_numpy(dtype=float)
    err = axis_summary["sem_ssi"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(err)
    if np.any(ok):
        lo = float(np.nanmin(y[ok] - err[ok]))
        hi = float(np.nanmax(y[ok] + err[ok]))
        span = max(hi - lo, 0.001)
        ax.set_ylim(lo - 0.10 * span, hi + 0.20 * span)
    ref_handles = [
        patches.Patch(facecolor="#8c8c8c", alpha=0.24, edgecolor="none", label="drift-only q25-q75"),
        patches.Patch(facecolor="#5f5f5f", alpha=0.20, edgecolor="none", label="microsaccade q25-q75"),
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=ref_handles + line_handles,
        labels=[h.get_label() for h in ref_handles] + line_labels,
        title="trajectory relative to contour",
        frameon=False,
        fontsize=7.4,
        title_fontsize=7.8,
        loc="lower right",
    )


def _panel_b(fig: plt.Figure, subspec: object, total: pd.DataFrame) -> list[plt.Axes]:
    panel_total = _combined_low_mid_panel_b(total)
    sub = subspec.subgridspec(1, 4, wspace=0.18)
    axes = [fig.add_subplot(sub[0, idx]) for idx in range(4)]
    panel_specs = [
        ("low_mid_sf", "strong_contours_no_osi", "Low SF\nall units"),
        ("low_mid_sf", "contour_matched", "Low SF\norientation-aligned"),
        ("high_sf", "strong_contours_no_osi", "High SF\nall units"),
        ("high_sf", "contour_matched", "High SF\norientation-aligned"),
    ]
    y = panel_total[
        panel_total["relation"].isin(["strong_contours_no_osi", "contour_matched"])
        & panel_total["sf_group"].isin(PANEL_B_SF_ORDER)
    ]["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
    finite = y[np.isfinite(y)] if y.size else np.asarray([], dtype=float)
    shared_ylim: tuple[float, float] | None = None
    if finite.size:
        lo = min(0.0, float(finite.min()))
        hi = max(0.0, float(finite.max()))
        span = max(hi - lo, 1.0)
        shared_ylim = (lo - 0.10 * span, hi + 0.14 * span)

    for idx, (sf_group, relation, title) in enumerate(panel_specs):
        ax = axes[idx]
        rows = panel_total[
            panel_total["relation"].eq(relation) & panel_total["sf_group"].eq(sf_group)
        ].copy()
        _plot_path_series(
            ax,
            rows,
            x_col="path_median_arcmin",
            color=PANEL_B_SF_COLORS[sf_group],
            min_pos=88.0,
            max_pos=180.0,
            linewidth=1.75,
            include_microsaccade=True,
        )
        ax.axhline(0, color="0.35", lw=0.9, ls=":")
        _format_axis(
            ax,
            ticks=[0, 90, 105, 120, 150, 175],
            min_pos=88.0,
            max_pos=180.0,
            show_xlabel=True,
        )
        _style_axis(ax)
        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)
        ax.set_title(title, fontsize=9.5, pad=6, color=PANEL_B_SF_COLORS[sf_group])
        ax.set_xlabel("path length (arcmin)")
        if idx == 0:
            ax.set_ylabel("SSI change (%)")
        else:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)

    return axes


def _panel_component(
    ax: plt.Axes,
    comp: pd.DataFrame,
    total: pd.DataFrame,
    *,
    relation: str,
    title: str,
    show_ylabel: bool = True,
    autoscale_y: bool = True,
    stat_style: str = "text",
    fixed_ylim: tuple[float, float] | None = None,
) -> None:
    source = comp[(comp["relation"].eq(relation)) & (comp["sf_group"].eq("high_sf"))].copy()
    total_source = total[(total["relation"].eq(relation)) & (total["sf_group"].eq("high_sf"))].copy()
    ax.axhline(0, color="0.35", lw=0.9, ls=":")
    stat_labels = []
    total_stat_label = _plot_total_series_with_ci(
        ax,
        total_source,
        min_pos=LOWER_MIN_POS,
        max_pos=LOWER_MAX_POS,
    )
    if total_stat_label is not None:
        stat_labels.append(total_stat_label)
    for metric, (label, linestyle, marker) in COMPONENT_STYLES.items():
        rows = source[source["component_metric"].eq(metric)]
        stat_label = _plot_component_series_with_ci(
            ax,
            rows,
            color=SF_COLORS["high_sf"],
            min_pos=LOWER_MIN_POS,
            max_pos=LOWER_MAX_POS,
            label=label,
            linestyle=linestyle,
            marker=marker,
        )
        if stat_label is not None:
            stat_labels.append(stat_label)
    _format_axis(ax, ticks=LOWER_TICKS, min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)
    x_sources = [
        total_source.loc[total_source["context"].eq("drift_only"), "path_median_arcmin"],
        source.loc[source["context"].eq("drift_only"), "component_median_arcmin"],
    ]
    max_x = max(
        [
            float(pd.to_numeric(values, errors="coerce").max())
            for values in x_sources
            if not values.empty and pd.notna(pd.to_numeric(values, errors="coerce").max())
        ],
        default=float(max(LOWER_TICKS)),
    )
    ax.set_xlim(
        -0.12,
        _x_broken_log([max(max_x, max(LOWER_TICKS))], min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)[0] + 0.28,
    )
    _style_axis(ax)
    ax.set_title(title, fontsize=11.3, pad=8)
    ax.set_ylabel("SSI change (%)" if show_ylabel else "")
    ax.set_xlabel("path length (arcmin; log scale after break)")
    y = source["population_ssi_percent_vs_stabilized"].to_numpy(dtype=float)
    finite = y[np.isfinite(y)]
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    elif autoscale_y and finite.size:
        lo = min(0.0, float(finite.min()))
        hi = max(0.0, float(finite.max()))
        span = max(hi - lo, 1.0)
        ax.set_ylim(lo - 0.12 * span, hi + 0.14 * span)
    if stat_labels and stat_style == "text":
        ax.text(
            0.03,
            0.94,
            "first bin vs 0\n" + "\n".join(stat_labels),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.1,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.2},
        )
    elif stat_style == "bracket":
        total_first = _first_drift_row(total_source, "path_median_arcmin")
        across_first = _first_drift_row(
            source[source["component_metric"].eq("across_path_arcmin")],
            "component_median_arcmin",
        )
        along_first = _first_drift_row(
            source[source["component_metric"].eq("along_path_arcmin")],
            "component_median_arcmin",
        )
        y_lo, y_top = ax.get_ylim()
        y_span = max(y_top - y_lo, 1.0)
        if total_first is not None:
            x1 = _x_broken_log(
                [float(total_first["path_median_arcmin"])],
                min_pos=LOWER_MIN_POS,
                max_pos=LOWER_MAX_POS,
            )[0]
            _add_bracket(
                ax,
                x0=0.0,
                x1=float(x1),
                y=y_top - 0.035 * y_span,
                text="full " + _format_p_label(
                    float(total_first["population_delta_p_image_bootstrap_sign"])
                ),
                color="0.38",
                linestyle="-",
            )
        if across_first is not None and along_first is not None:
            component_first_x = max(
                float(across_first["component_median_arcmin"]),
                float(along_first["component_median_arcmin"]),
            )
            x1 = _x_broken_log([component_first_x], min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)[0]
            label = (
                "across "
                + _format_p_label(float(across_first["population_delta_p_image_bootstrap_sign"]))
                + "\nalong "
                + _format_p_label(float(along_first["population_delta_p_image_bootstrap_sign"]))
            )
            _add_bracket(
                ax,
                x0=0.0,
                x1=float(x1),
                y=y_top - 0.14 * y_span,
                text=label,
                color=SF_COLORS["high_sf"],
                linestyle="-",
                text_x=float(x1) + 0.10,
                text_ha="left",
            )
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")


def _component_shared_ylim(comp: pd.DataFrame, total: pd.DataFrame, relations: list[str]) -> tuple[float, float]:
    comp_sub = comp[
        comp["relation"].isin(relations)
        & comp["sf_group"].eq("high_sf")
        & comp["context"].isin(["stabilized", "drift_only"])
    ].copy()
    total_sub = total[
        total["relation"].isin(relations)
        & total["sf_group"].eq("high_sf")
        & total["context"].isin(["stabilized", "drift_only"])
    ].copy()
    if comp_sub.empty and total_sub.empty:
        return (-1.0, 1.0)
    vals = [0.0]
    for sub in [comp_sub, total_sub]:
        for col in [
            "population_ssi_percent_vs_stabilized",
            "population_delta_percent_ci95_low_image_boot",
            "population_delta_percent_ci95_high_image_boot",
        ]:
            if col in sub.columns:
                arr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    return lo - 0.12 * span, hi + 0.14 * span


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )


def _save_panel_c_standalone(total: pd.DataFrame, comp: pd.DataFrame) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    ylim = _component_shared_ylim(comp, total, ["contour_matched"])
    span = max(ylim[1] - ylim[0], 1.0)
    ylim = (ylim[0], ylim[1] + 0.10 * span)
    _panel_component(
        ax,
        comp,
        total,
        relation="contour_matched",
        title="Aligned high-SF units",
        show_ylabel=True,
        autoscale_y=False,
        stat_style="bracket",
        fixed_ylim=ylim,
    )
    fig.tight_layout()
    png = OUT_DIR / "backimage_real_trace_panel_c_aligned_high_sf_no_zero_bridge.png"
    pdf = OUT_DIR / "backimage_real_trace_panel_c_aligned_high_sf_no_zero_bridge.pdf"
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    total, comp, axis_summary, trace_context = _load_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 8.4))
    gs = fig.add_gridspec(
        2,
        1,
        left=0.06,
        right=0.985,
        top=0.885,
        bottom=0.13,
        hspace=0.42,
        height_ratios=(0.78, 1.0),
    )
    axes_b = _panel_b(fig, gs[0, 0], total)
    lower = gs[1, :].subgridspec(1, 3, wspace=0.25)
    ax_c = fig.add_subplot(lower[0, 0])
    ax_d = fig.add_subplot(lower[0, 1])
    ax_e = fig.add_subplot(lower[0, 2])

    lower_relations = ["contour_matched", "contour_intermediate", "contour_orthogonal"]
    lower_ylim = _component_shared_ylim(comp, total, lower_relations)
    _panel_component(
        ax_c,
        comp,
        total,
        relation="contour_matched",
        title="Aligned high-SF units",
        show_ylabel=True,
        autoscale_y=False,
    )
    _panel_component(
        ax_d,
        comp,
        total,
        relation="contour_intermediate",
        title="Oblique high-SF units",
        show_ylabel=False,
        autoscale_y=False,
    )
    _panel_component(
        ax_e,
        comp,
        total,
        relation="contour_orthogonal",
        title="Orthogonal high-SF units",
        show_ylabel=False,
        autoscale_y=False,
    )
    for ax in (ax_c, ax_d, ax_e):
        ax.set_ylim(*lower_ylim)

    _panel_label(axes_b[0], "B")
    _panel_label(ax_c, "C")
    _panel_label(ax_d, "D")
    _panel_label(ax_e, "E")

    fig.suptitle("Real fixational motion changes SSI according to local contour geometry", fontsize=15.5, y=0.972)
    fig.text(
        0.5,
        0.045,
        "B uses strong contour image windows and contrasts a figure-only low/middle-SF pool with high-SF units; "
        "C-E fix high-SF unit-contour alignment. "
        "Open points are drift-only snippets; filled points show snippets with detected microsaccades where present.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="0.25",
    )
    png = OUT_DIR / "backimage_real_trace_geometry_reordered_story_figure.png"
    pdf = OUT_DIR / "backimage_real_trace_geometry_reordered_story_figure.pdf"
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    standalone_png, standalone_pdf = _save_panel_c_standalone(total, comp)
    print(png)
    print(pdf)
    print(standalone_png)
    print(standalone_pdf)


if __name__ == "__main__":
    main()
