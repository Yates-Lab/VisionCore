#!/usr/bin/env python3
"""Cell-baselined revision of the BackImage contour-geometry story figure."""

from __future__ import annotations

import json
import math
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
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from _fig4_component_2d_surface import (
    MATRIX_DIR,
    OUT_DIR,
    _assign_bins,
    _compute_component_metrics,
    _json_ready,
    _quantile_edges,
)
from _fig4_component_path_baseline import (
    _cell_matched_baseline,
)
from _fig4_trace_schematics import (
    add_equal_count_trace_bins,
    accumulate_population_movie_rows,
    axis_delta_deg,
    baseline_rows_by_image,
    load_dataset,
    ratio_delta_stats,
)


OUT_STEM = "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075_coh020_cde8bins"
SF_METRIC_COL = "sf_split_metric"
LOW_SF_MAX_CPD = 0.50
HIGH_SF_MIN_CPD = 0.75
CONTOUR_COHERENCE_MIN = 0.20
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
N_DRIFT_BINS = 8
N_MICROSACCADE_BINS = 5
N_COMPONENT_BINS = 8
EPS = 1e-12

# Image-level paired bootstrap of the moving-vs-cell-baseline ratio delta,
# same convention as _fig4_cell_baseline_errorbars.py:
# each bin's point estimate is a spike-weighted ratio pooled over every
# (unit, image, trajectory-row) triple, so units aren't independent replicates
# to resample -- images are the exchangeable, randomly-drawn design axis, and
# per-image sums already fold in whatever units/trajectory-draws landed on
# each image within the bin.
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47

B_MIN_POS = 88.0
B_MAX_POS = 180.0
B_TICKS = (0.0, 90.0, 105.0, 120.0, 150.0, 175.0)
LOWER_MIN_POS = 45.0
LOWER_MAX_POS = 180.0
LOWER_TICKS = (0.0, 50.0, 65.0, 90.0, 120.0, 160.0)

SF_GROUPS = {
    "low_lt0p5": {
        "label": "SF < 0.50",
        "title": "SF < 0.50",
        "color": "#0072B2",
    },
    "high_ge0p75": {
        "label": "SF >= 0.75",
        "title": "SF >= 0.75",
        "color": "#D55E00",
    },
}
B_PANEL_SPECS = [
    ("low_lt0p5", "strong_contours_no_osi", "SF < 0.50\nall contour units"),
    ("low_lt0p5", "contour_matched", "SF < 0.50\norientation-aligned"),
    ("high_ge0p75", "strong_contours_no_osi", "SF >= 0.75\nall contour units"),
    ("high_ge0p75", "contour_matched", "SF >= 0.75\norientation-aligned"),
]
LOWER_PANEL_SPECS = [
    ("contour_matched", "Aligned SF >= 0.75 units"),
    ("contour_intermediate", "Oblique SF >= 0.75 units"),
    ("contour_orthogonal", "Orthogonal SF >= 0.75 units"),
]
COMPONENT_STYLES = {
    "across_path_arcmin": ("across-contour component path", "-", "o"),
    "along_path_arcmin": ("along-contour component path", (0, (4.2, 2.0)), "s"),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > EPS else float("nan")


def _pct_delta(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > EPS):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _population_values(pop: dict[str, Any]) -> dict[str, float]:
    n = float(pop["n_movie_samples"])
    return {
        "population_ssi_bits_per_spike": float(pop["population_ssi_bits_per_spike"]),
        "information_bits_per_sample": _finite_ratio(float(pop["information_numerator_bits"]), n),
        "expected_spikes_per_sample": _finite_ratio(float(pop["expected_spikes"]), n),
    }


def _x_broken_log(values: np.ndarray | pd.Series | list[float], *, min_pos: float, max_pos: float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mapped = np.zeros_like(x, dtype=float)
    positive = x > 0
    span = 5.1
    mapped[positive] = 1.0 + span * np.log(x[positive] / min_pos) / np.log(max_pos / min_pos)
    return mapped


def _format_broken_axis(
    ax: plt.Axes,
    *,
    ticks: tuple[float, ...],
    min_pos: float,
    max_pos: float,
    xlabel: str,
    show_xlabel: bool = True,
) -> None:
    ax.set_xlim(-0.12, 5.35)
    ax.set_xticks(_x_broken_log(list(ticks), min_pos=min_pos, max_pos=max_pos))
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    if show_xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.text(
        0.52,
        -0.075,
        "//",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        rotation=-20,
        clip_on=False,
    )
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _sf_mask(unit: pd.DataFrame, sf_group: str) -> np.ndarray:
    sf = pd.to_numeric(unit[SF_METRIC_COL], errors="coerce").to_numpy(dtype=float)
    if sf_group == "low_lt0p5":
        return np.isfinite(sf) & (sf < LOW_SF_MAX_CPD)
    if sf_group == "high_ge0p75":
        return np.isfinite(sf) & (sf >= HIGH_SF_MIN_CPD)
    raise ValueError(f"unknown SF group {sf_group!r}")


def _selected_unit_images(
    unit: pd.DataFrame,
    image: pd.DataFrame,
    *,
    sf_group: str,
    relation: str,
) -> dict[int, np.ndarray]:
    image_indices = image["image_index"].astype(int).to_numpy()
    image_axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    contour_mask = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= CONTOUR_COHERENCE_MIN)
    selected: dict[int, np.ndarray] = {}
    use_units = unit[_sf_mask(unit, sf_group)].copy()
    for row in use_units.itertuples(index=False):
        unit_index = int(row.unit_index)
        if relation == "strong_contours_no_osi":
            image_subset = image_indices[contour_mask]
        else:
            pref = float(row.prior_preferred_orientation_deg)
            osi = float(row.prior_orientation_selectivity_index)
            if not (math.isfinite(pref) and math.isfinite(osi) and osi >= MIN_OSI):
                continue
            delta = axis_delta_deg(image_axis, pref)
            if relation == "contour_matched":
                keep = contour_mask & np.isfinite(delta) & (delta <= MATCH_MAX_DEG)
            elif relation == "contour_intermediate":
                keep = contour_mask & np.isfinite(delta) & (delta > MATCH_MAX_DEG) & (delta < ORTHOGONAL_MIN_DEG)
            elif relation == "contour_orthogonal":
                keep = contour_mask & np.isfinite(delta) & (delta >= ORTHOGONAL_MIN_DEG)
            else:
                raise ValueError(f"unknown relation {relation!r}")
            image_subset = image_indices[keep]
        if image_subset.size:
            selected[unit_index] = image_subset.astype(int)
    return selected


def _cell_residual_values(
    data: dict[str, Any],
    *,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    moving_pop = accumulate_population_movie_rows(
        ssi=data["ssi"],
        expected=data["expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    cell_pop = _cell_matched_baseline(
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    moving = _population_values(moving_pop)
    cell = _population_values(cell_pop)
    ssi_percent = _pct_delta(
        moving["population_ssi_bits_per_spike"],
        cell["population_ssi_bits_per_spike"],
    )
    delta_stats = ratio_delta_stats(
        moving_pop["per_image_num"],
        moving_pop["per_image_den"],
        cell_pop["per_image_num"],
        cell_pop["per_image_den"],
        n_resamples=n_bootstrap,
        rng=rng,
    )
    delta_low = float(delta_stats["population_delta_ci95_low_image_boot"])
    delta_high = float(delta_stats["population_delta_ci95_high_image_boot"])
    cell_ssi_bps = cell["population_ssi_bits_per_spike"]
    percent_ci_low = (
        100.0 * delta_low / cell_ssi_bps
        if math.isfinite(delta_low) and math.isfinite(cell_ssi_bps) and abs(cell_ssi_bps) > EPS
        else float("nan")
    )
    percent_ci_high = (
        100.0 * delta_high / cell_ssi_bps
        if math.isfinite(delta_high) and math.isfinite(cell_ssi_bps) and abs(cell_ssi_bps) > EPS
        else float("nan")
    )
    return {
        "n_movie_samples": int(moving_pop["n_movie_samples"]),
        "n_images_contributing": int(moving_pop["n_images_contributing"]),
        "moving_population_ssi_bits_per_spike": moving["population_ssi_bits_per_spike"],
        "cell_baseline_population_ssi_bits_per_spike": cell["population_ssi_bits_per_spike"],
        "ssi_percent_vs_cell_baseline": ssi_percent,
        "ssi_percent_ci95_low_image_boot": percent_ci_low,
        "ssi_percent_ci95_high_image_boot": percent_ci_high,
        "ssi_delta_p_image_bootstrap_sign": float(delta_stats["population_delta_p_image_bootstrap_sign"]),
        "moving_information_bits_per_sample": moving["information_bits_per_sample"],
        "cell_baseline_information_bits_per_sample": cell["information_bits_per_sample"],
        "information_percent_vs_cell_baseline": _pct_delta(
            moving["information_bits_per_sample"],
            cell["information_bits_per_sample"],
        ),
        "moving_expected_spikes_per_sample": moving["expected_spikes_per_sample"],
        "cell_baseline_expected_spikes_per_sample": cell["expected_spikes_per_sample"],
        "spikes_percent_vs_cell_baseline": _pct_delta(
            moving["expected_spikes_per_sample"],
            cell["expected_spikes_per_sample"],
        ),
    }


def _movie_with_trace_bins(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace, trace_bins = add_equal_count_trace_bins(
        data["trace"],
        n_drift_bins=N_DRIFT_BINS,
        n_microsaccade_bins=N_MICROSACCADE_BINS,
    )
    movie = data["movie"].copy()
    trace_lookup = trace.set_index("trace_bank_index")
    trace_index = movie["trace_index"].astype(int)
    movie["context"] = trace_lookup.loc[trace_index, "context"].to_numpy()
    movie["context_label"] = trace_lookup.loc[trace_index, "context_label"].to_numpy()
    movie["path_bin"] = trace_lookup.loc[trace_index, "path_bin"].to_numpy()
    movie["path_bin_order"] = trace_lookup.loc[trace_index, "path_bin_order"].to_numpy()
    return movie, trace_bins


def _compute_panel_b(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    movie, trace_bins = _movie_with_trace_bins(data)
    row_image_index = movie["image_index"].astype(int).to_numpy()
    row_context = movie["context"].astype(str).to_numpy()
    row_path_bin = movie["path_bin"].astype(str).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    panel_keys = {(sf_group, relation) for sf_group, relation, _title in B_PANEL_SPECS}

    for sf_group, relation in sorted(panel_keys):
        unit_to_images = _selected_unit_images(data["unit"], data["image"], sf_group=sf_group, relation=relation)
        selection_rows.append(
            {
                "panel": "B",
                "sf_group": sf_group,
                "relation": relation,
                "n_selected_units": int(len(unit_to_images)),
                "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
            }
        )
        for bin_row in trace_bins.sort_values(["context", "path_bin_order"]).itertuples(index=False):
            path_bin = str(bin_row.path_bin)
            context = str(bin_row.context)
            row_mask = (row_context == context) & (row_path_bin == path_bin)
            values = _cell_residual_values(
                data,
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
                rng=rng,
            )
            rows.append(
                {
                    "panel": "B",
                    "sf_group": sf_group,
                    "sf_group_label": SF_GROUPS[sf_group]["label"],
                    "relation": relation,
                    "context": context,
                    "path_bin": path_bin,
                    "path_bin_order": int(bin_row.path_bin_order),
                    "path_median_arcmin": float(bin_row.median_path_arcmin),
                    "n_traces": int(bin_row.n_traces),
                    "n_selected_units": int(len(unit_to_images)),
                    "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
                    **values,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(selection_rows)


def _compute_component_panel(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _compute_component_metrics(data)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for relation, _title in LOWER_PANEL_SPECS:
        unit_to_images = _selected_unit_images(
            data["unit"],
            data["image"],
            sf_group="high_ge0p75",
            relation=relation,
        )
        selection_rows.append(
            {
                "panel": "CDE",
                "sf_group": "high_ge0p75",
                "relation": relation,
                "n_selected_units": int(len(unit_to_images)),
                "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
            }
        )
        for metric_col, (metric_label, _linestyle, _marker) in COMPONENT_STYLES.items():
            edges = _quantile_edges(metrics.loc[drift_mask, metric_col].to_numpy(dtype=float), N_COMPONENT_BINS)
            bins = _assign_bins(metrics[metric_col].to_numpy(dtype=float), edges)
            for bin_index in range(N_COMPONENT_BINS):
                row_mask = drift_mask & (bins == bin_index)
                global_rows = metrics[row_mask]
                values = _cell_residual_values(
                    data,
                    row_image_index=row_image_index,
                    row_mask=row_mask,
                    baseline_lookup=baseline_lookup,
                    unit_to_images=unit_to_images,
                    n_images=n_images,
                    rng=rng,
                )
                rows.append(
                    {
                        "panel": "CDE",
                        "sf_group": "high_ge0p75",
                        "sf_group_label": SF_GROUPS["high_ge0p75"]["label"],
                        "relation": relation,
                        "component_metric": metric_col,
                        "component_metric_label": metric_label,
                        "context": "drift_only",
                        "component_bin": f"drift_only_q{bin_index + 1:02d}",
                        "component_bin_order": int(bin_index + 1),
                        "component_min_arcmin": float(edges[bin_index]),
                        "component_max_arcmin": float(edges[bin_index + 1]),
                        "component_median_arcmin": float(np.nanmedian(global_rows[metric_col])),
                        "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                        "n_selected_units": int(len(unit_to_images)),
                        "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
                        **values,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(selection_rows)


def _plot_b_series(ax: plt.Axes, frame: pd.DataFrame, *, color: str) -> None:
    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=28,
        facecolors="white",
        edgecolors=color,
        linewidths=1.35,
        zorder=5,
    )
    for context, filled in [("drift_only", False), ("microsaccade", True)]:
        sub = frame[frame["context"].eq(context)].sort_values("path_bin_order")
        if sub.empty:
            continue
        x = _x_broken_log(sub["path_median_arcmin"], min_pos=B_MIN_POS, max_pos=B_MAX_POS)
        y = sub["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        if "ssi_percent_ci95_low_image_boot" in sub.columns:
            ci_low = pd.to_numeric(sub["ssi_percent_ci95_low_image_boot"], errors="coerce").to_numpy(dtype=float)
            ci_high = pd.to_numeric(sub["ssi_percent_ci95_high_image_boot"], errors="coerce").to_numpy(dtype=float)
            has_ci = np.isfinite(ci_low) & np.isfinite(ci_high) & np.isfinite(y)
            if np.any(has_ci):
                yerr_low = np.clip(y[has_ci] - ci_low[has_ci], 0.0, None)
                yerr_high = np.clip(ci_high[has_ci] - y[has_ci], 0.0, None)
                ax.errorbar(
                    x[has_ci],
                    y[has_ci],
                    yerr=[yerr_low, yerr_high],
                    color=color,
                    linestyle="none",
                    elinewidth=1.1,
                    capsize=0,
                    zorder=3,
                )
        ax.plot(x, y, color=color, linewidth=1.75, zorder=2)
        ax.scatter(
            x,
            y,
            marker="o",
            s=24,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.25,
            zorder=4,
        )


def _plot_component_series(ax: plt.Axes, frame: pd.DataFrame, *, color: str) -> None:
    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=30,
        facecolors="white",
        edgecolors=color,
        linewidths=1.35,
        zorder=5,
    )
    for metric_col, (label, linestyle, marker) in COMPONENT_STYLES.items():
        sub = frame[frame["component_metric"].eq(metric_col)].sort_values("component_bin_order")
        if sub.empty:
            continue
        x = _x_broken_log(sub["component_median_arcmin"], min_pos=LOWER_MIN_POS, max_pos=LOWER_MAX_POS)
        y = sub["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.95, label=label, zorder=3)
        ax.scatter(
            x,
            y,
            marker=marker,
            s=25,
            facecolors="white",
            edgecolors=color,
            linewidths=1.25,
            zorder=4,
        )


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


def _ylim_series(frame: pd.DataFrame, col: str = "ssi_percent_vs_cell_baseline") -> list[pd.Series]:
    """Point estimate plus bootstrap CI bounds (if present), so shared y-limits
    aren't clipping the error bars this function's callers go on to draw."""
    series = [frame[col]]
    for ci_col in ("ssi_percent_ci95_low_image_boot", "ssi_percent_ci95_high_image_boot"):
        if ci_col in frame.columns:
            series.append(frame[ci_col])
    return series


def _shared_ylim(values: list[pd.Series], *, pad_low: float = 0.12, pad_high: float = 0.14) -> tuple[float, float]:
    arrs = [pd.to_numeric(series, errors="coerce").to_numpy(dtype=float) for series in values if not series.empty]
    vals = [0.0]
    for arr in arrs:
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    return lo - pad_low * span, hi + pad_high * span


def _plot_figure(panel_b: pd.DataFrame, component: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(11.2, 8.3))
    gs = fig.add_gridspec(
        2,
        1,
        left=0.06,
        right=0.985,
        top=0.885,
        bottom=0.18,
        hspace=0.42,
        height_ratios=(0.78, 1.0),
    )
    sub_b = gs[0, 0].subgridspec(1, 4, wspace=0.18)
    axes_b = [fig.add_subplot(sub_b[0, idx]) for idx in range(4)]
    b_ylim = _shared_ylim(_ylim_series(panel_b), pad_low=0.11, pad_high=0.14)
    for idx, (sf_group, relation, title) in enumerate(B_PANEL_SPECS):
        ax = axes_b[idx]
        frame = panel_b[panel_b["sf_group"].eq(sf_group) & panel_b["relation"].eq(relation)].copy()
        color = str(SF_GROUPS[sf_group]["color"])
        _plot_b_series(ax, frame, color=color)
        _format_broken_axis(
            ax,
            ticks=B_TICKS,
            min_pos=B_MIN_POS,
            max_pos=B_MAX_POS,
            xlabel="trajectory path (arcmin)",
        )
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_ylim(*b_ylim)
        n_units = int(frame["n_selected_units"].iloc[0]) if not frame.empty else 0
        n_pairs = int(frame["n_selected_unit_image_pairs"].iloc[0]) if not frame.empty else 0
        ax.set_title(f"{title}\n{n_units} units, {n_pairs} pairs", fontsize=9.0, pad=5, color=color)
        if idx == 0:
            ax.set_ylabel("SSI residual\n(% vs matched stabilized)")
        else:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)

    lower = gs[1, :].subgridspec(1, 3, wspace=0.25)
    axes_cde = [fig.add_subplot(lower[0, idx]) for idx in range(3)]
    cde_ylim = _shared_ylim(_ylim_series(component), pad_low=0.13, pad_high=0.15)
    for idx, (relation, title) in enumerate(LOWER_PANEL_SPECS):
        ax = axes_cde[idx]
        frame = component[component["relation"].eq(relation)].copy()
        color = str(SF_GROUPS["high_ge0p75"]["color"])
        _plot_component_series(ax, frame, color=color)
        _format_broken_axis(
            ax,
            ticks=LOWER_TICKS,
            min_pos=LOWER_MIN_POS,
            max_pos=LOWER_MAX_POS,
            xlabel="component path (arcmin)",
        )
        ax.axhline(0.0, color="0.35", lw=0.9, ls=":")
        ax.set_ylim(*cde_ylim)
        n_units = int(frame["n_selected_units"].iloc[0]) if not frame.empty else 0
        n_pairs = int(frame["n_selected_unit_image_pairs"].iloc[0]) if not frame.empty else 0
        ax.set_title(f"{title}\n{n_units} units, {n_pairs} pairs", fontsize=10.5, pad=7)
        if idx == 0:
            ax.set_ylabel("SSI residual\n(% vs matched stabilized)")
        else:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7.8, loc="lower left")

    _panel_label(axes_b[0], "B")
    _panel_label(axes_cde[0], "C")
    _panel_label(axes_cde[1], "D")
    _panel_label(axes_cde[2], "E")

    marker_handles = [
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
            label="microsaccade bins in B",
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
    fig.legend(handles=marker_handles, frameon=False, fontsize=8.0, ncol=3, loc="lower center", bbox_to_anchor=(0.54, 0.083))
    fig.suptitle(
        "Cell-matched real fixational motion effects separate by contour geometry",
        fontsize=15.2,
        y=0.972,
    )
    fig.text(
        0.5,
        0.035,
        (
            "B uses contour image windows (coherence >= 0.20), SF < 0.50 as low, and SF >= 0.75 as high; "
            "units with 0.50 <= SF < 0.75 are excluded. C-E use SF >= 0.75, 8 wider component bins, and omit the full-trajectory curve."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )
    return fig


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    panel_b, b_selection = _compute_panel_b(data)
    component, cde_selection = _compute_component_panel(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    values_csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    b_csv = OUT_DIR / f"{OUT_STEM}_panel_b_values.csv"
    cde_csv = OUT_DIR / f"{OUT_STEM}_component_values.csv"
    selection_csv = OUT_DIR / f"{OUT_STEM}_selection_summary.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    png = OUT_DIR / f"{OUT_STEM}.png"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"

    values = pd.concat([panel_b, component], ignore_index=True, sort=False)
    values.to_csv(values_csv, index=False)
    panel_b.to_csv(b_csv, index=False)
    component.to_csv(cde_csv, index=False)
    selection = pd.concat([b_selection, cde_selection], ignore_index=True, sort=False)
    selection.to_csv(selection_csv, index=False)

    fig = _plot_figure(panel_b, component)
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    with PdfPages(OUT_DIR / f"{OUT_STEM}_multipage.pdf") as pages:
        pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_geometry_reordered_story_figure_cell_baseline_sf075",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "png": png,
                "pdf": pdf,
                "values_csv": values_csv,
                "panel_b_values_csv": b_csv,
                "component_values_csv": cde_csv,
                "selection_summary_csv": selection_csv,
                "summary_json": json_path,
            },
            "selection": {
                "sf_metric_col": SF_METRIC_COL,
                "low_sf": f"{SF_METRIC_COL} < {LOW_SF_MAX_CPD}",
                "high_sf": f"{SF_METRIC_COL} >= {HIGH_SF_MIN_CPD}",
                "excluded_sf_band": f"{LOW_SF_MAX_CPD} <= {SF_METRIC_COL} < {HIGH_SF_MIN_CPD}",
                "contour_coherence_min": CONTOUR_COHERENCE_MIN,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
            },
            "binning": {
                "panel_b_drift_bins": N_DRIFT_BINS,
                "panel_b_microsaccade_bins": N_MICROSACCADE_BINS,
                "component_drift_bins": N_COMPONENT_BINS,
            },
            "baseline": "Each plotted nonzero bin is compared with a cell-matched stabilized baseline weighted by that bin's image composition.",
            "note": "C-E intentionally omit the full-trajectory curve because component path length and total path length are not commensurate axes.",
        },
    )
    print(png)
    print(pdf)
    print(values_csv)
    print(selection_csv)
    print(json_path)


if __name__ == "__main__":
    main()
