#!/usr/bin/env python3
"""Panel-G analogs with alternative contour-relative dose axes."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
from _fig4_style import GRAY, INK, ORANGE, configure_matplotlib
import _fig4_cell_baseline_errorbars as panel_c
from _fig4_component_2d_surface import _assign_bins
from _fig4_component_path_baseline import (
    _cell_matched_baseline,
)
from make_backimage_panel_c_across_along_tail_contrast import (
    _bootstrap_residual_difference,
)
from _fig4_trace_schematics import (
    accumulate_population_movie_rows,
    axis_delta_deg,
    baseline_rows_by_image,
    finite_ratio,
    ratio_delta_stats,
)
from paper.fig4.upstream.build_robust_unit_tuning import production_sf_mask


OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
OUT_STEM = "panel_g_alternative_x_axes_diagnostic"
SF_MIN_CPD = 0.50
SF_GROUP_MODE = os.environ.get("FIG4_SF_GROUP_MODE", "absolute_cpd")
_HIGH_TITLE = "Upper-SF-Tertile" if SF_GROUP_MODE == "table_tertiles" else "High-SF"
_LOW_TITLE = "Lower-SF-Tertile" if SF_GROUP_MODE == "table_tertiles" else "Low-SF"
_HIGH_SUBTITLE = (
    "cycle-valid upper SF tertile"
    if SF_GROUP_MODE == "table_tertiles"
    else "SF >=0.5"
)
_LOW_SUBTITLE = (
    "cycle-valid lower SF tertile"
    if SF_GROUP_MODE == "table_tertiles"
    else "SF <0.5"
)
CONTOUR_COHERENCE_MIN = 0.20
MIN_OSI = 0.05
MATCH_MAX_DEG = 15.0
ORTHOGONAL_MIN_DEG = 67.5
BODY_QUANTILES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
TAIL_QUANTILE = 0.95
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 47
EPS = 1e-12

FAMILIES = (
    {
        "key": "component_path",
        "title": "Unsigned Component Path",
        "description": "sum(abs(projected sample-to-sample displacement))",
        "across": "across_path_arcmin",
        "along": "along_path_arcmin",
        "xlabel": "component path (arcmin)",
        "unit": "arcmin",
    },
    {
        "key": "component_rms",
        "title": "RMS Excursion",
        "description": "RMS of centered projected trace position",
        "across": "across_rms_arcmin",
        "along": "along_rms_arcmin",
        "xlabel": "component RMS excursion (arcmin)",
        "unit": "arcmin",
    },
    {
        "key": "component_range",
        "title": "Projected Range",
        "description": "peak-to-peak projected trace position",
        "across": "across_range_arcmin",
        "along": "along_range_arcmin",
        "xlabel": "component peak-to-peak range (arcmin)",
        "unit": "arcmin",
    },
    {
        "key": "path_per_range",
        "title": "Tortuosity Proxy",
        "description": "component path divided by component peak-to-peak range",
        "across": "across_path_per_range",
        "along": "along_path_per_range",
        "xlabel": "component path / range",
        "unit": "ratio",
    },
)

COMPONENT_SPECS = (
    ("across", "contour-normal", "-", "o", "across"),
    ("along", "contour-parallel", (0, (4.2, 2.0)), "s", "along"),
)

POPULATION_SPECS = (
    {
        "key": "high_sf_all",
        "title": f"All {_HIGH_TITLE} Units",
        "subtitle": f"{_HIGH_SUBTITLE}; contour coherence >=0.2; no orientation relation filter",
        "sf_group": "high",
        "relation": "all",
        "requires_orientation_tuning": False,
    },
    {
        "key": "high_sf_aligned",
        "title": f"Aligned {_HIGH_TITLE} Units",
        "subtitle": f"{_HIGH_SUBTITLE}; unit-contour <=15 deg",
        "sf_group": "high",
        "relation": "aligned",
        "requires_orientation_tuning": True,
    },
    {
        "key": "high_sf_oblique",
        "title": f"Oblique {_HIGH_TITLE} Units",
        "subtitle": f"{_HIGH_SUBTITLE}; unit-contour 15-67.5 deg",
        "sf_group": "high",
        "relation": "oblique",
        "requires_orientation_tuning": True,
    },
    {
        "key": "high_sf_orthogonal",
        "title": f"Orthogonal {_HIGH_TITLE} Units",
        "subtitle": f"{_HIGH_SUBTITLE}; unit-contour >=67.5 deg",
        "sf_group": "high",
        "relation": "orthogonal",
        "requires_orientation_tuning": True,
    },
    {
        "key": "low_sf_all",
        "title": f"All {_LOW_TITLE} Units",
        "subtitle": f"{_LOW_SUBTITLE}; contour coherence >=0.2; no orientation relation filter",
        "sf_group": "low",
        "relation": "all",
        "requires_orientation_tuning": False,
    },
)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_extended_component_metrics(data: dict[str, Any]) -> pd.DataFrame:
    movie = data["movie"]
    trace_xy = np.asarray(data["trace_xy"], dtype=np.float32)
    trace_index = movie["trace_index"].astype(int).to_numpy()
    axes = pd.to_numeric(movie["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=np.float64)
    theta = np.radians(axes)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xy = trace_xy[trace_index]
    steps = np.diff(trace_xy, axis=1)[trace_index]
    along_step = steps[:, :, 0] * cos_t[:, None] + steps[:, :, 1] * sin_t[:, None]
    across_step = -steps[:, :, 0] * sin_t[:, None] + steps[:, :, 1] * cos_t[:, None]
    centered = xy - np.nanmean(xy, axis=1, keepdims=True)
    along_pos = centered[:, :, 0] * cos_t[:, None] + centered[:, :, 1] * sin_t[:, None]
    across_pos = -centered[:, :, 0] * sin_t[:, None] + centered[:, :, 1] * cos_t[:, None]

    along_path = np.nansum(np.abs(along_step), axis=1) * 60.0
    across_path = np.nansum(np.abs(across_step), axis=1) * 60.0
    along_rms = np.sqrt(np.nanmean(along_pos * along_pos, axis=1)) * 60.0
    across_rms = np.sqrt(np.nanmean(across_pos * across_pos, axis=1)) * 60.0
    along_range = (np.nanmax(along_pos, axis=1) - np.nanmin(along_pos, axis=1)) * 60.0
    across_range = (np.nanmax(across_pos, axis=1) - np.nanmin(across_pos, axis=1)) * 60.0
    along_path_per_range = along_path / np.maximum(along_range, EPS)
    across_path_per_range = across_path / np.maximum(across_range, EPS)

    invalid = ~np.isfinite(axes)
    for arr in (
        along_path,
        across_path,
        along_rms,
        across_rms,
        along_range,
        across_range,
        along_path_per_range,
        across_path_per_range,
    ):
        arr[invalid] = np.nan

    if "rendered_n_microsaccade_events" in movie.columns:
        has_ms = pd.to_numeric(movie["rendered_n_microsaccade_events"], errors="coerce").fillna(0).gt(0)
    else:
        has_ms = pd.Series(False, index=movie.index)

    return pd.DataFrame(
        {
            "movie_index": movie["movie_index"].astype(int).to_numpy(),
            "image_index": movie["image_index"].astype(int).to_numpy(),
            "trace_index": trace_index,
            "context": np.where(has_ms.to_numpy(dtype=bool), "microsaccade", "drift_only"),
            "has_microsaccade": has_ms.to_numpy(dtype=bool),
            "rendered_path_length_arcmin": pd.to_numeric(movie["rendered_path_length_arcmin"], errors="coerce").to_numpy(dtype=float),
            "along_path_arcmin": along_path,
            "across_path_arcmin": across_path,
            "along_rms_arcmin": along_rms,
            "across_rms_arcmin": across_rms,
            "along_range_arcmin": along_range,
            "across_range_arcmin": across_range,
            "along_path_per_range": along_path_per_range,
            "across_path_per_range": across_path_per_range,
        }
    )


def _selected_unit_images_for_population(
    data: dict[str, Any],
    population: dict[str, Any],
) -> dict[int, np.ndarray]:
    unit = data["unit"]
    image = data["image"]
    pref = pd.to_numeric(unit["prior_preferred_orientation_deg"], errors="coerce").to_numpy(dtype=float)
    osi = pd.to_numeric(unit["prior_orientation_selectivity_index"], errors="coerce").to_numpy(dtype=float)
    unit_index = unit["unit_index"].astype(int).to_numpy()

    image_indices = image["image_index"].astype(int).to_numpy()
    image_axis = pd.to_numeric(image["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=float)
    coherence = pd.to_numeric(image["image_orientation_coherence"], errors="coerce").to_numpy(dtype=float)
    contour_mask = np.isfinite(image_axis) & np.isfinite(coherence) & (coherence >= CONTOUR_COHERENCE_MIN)

    sf_group = str(population["sf_group"])
    relation = str(population["relation"])
    requires_orientation_tuning = bool(population["requires_orientation_tuning"])
    selected: dict[int, np.ndarray] = {}
    keep_units = production_sf_mask(
        unit,
        group=sf_group,
        mode=SF_GROUP_MODE,
        metric_column=panel_c.SF_METRIC_COL,
        low_max_cpd=SF_MIN_CPD,
        high_min_cpd=SF_MIN_CPD,
    )
    for idx, unit_id in enumerate(unit_index):
        if not keep_units[idx]:
            continue

        if requires_orientation_tuning:
            if not (math.isfinite(float(pref[idx])) and math.isfinite(float(osi[idx])) and float(osi[idx]) >= MIN_OSI):
                continue
            delta = axis_delta_deg(image_axis, float(pref[idx]))
            if relation == "aligned":
                keep = contour_mask & np.isfinite(delta) & (delta <= MATCH_MAX_DEG)
            elif relation == "oblique":
                keep = contour_mask & np.isfinite(delta) & (delta > MATCH_MAX_DEG) & (delta < ORTHOGONAL_MIN_DEG)
            elif relation == "orthogonal":
                keep = contour_mask & np.isfinite(delta) & (delta >= ORTHOGONAL_MIN_DEG)
            else:
                raise ValueError(f"Orientation-tuned population has unsupported relation {relation!r}")
        elif relation == "all":
            keep = contour_mask
        else:
            raise ValueError(f"Untuned population has unsupported relation {relation!r}")

        images = image_indices[keep]
        if images.size:
            selected[int(unit_id)] = images.astype(int)
    return selected


def _population_for_mask(
    data: dict[str, Any],
    *,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
) -> dict[str, Any]:
    moving = accumulate_population_movie_rows(
        ssi=data["ssi"],
        expected=data["expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    cell = _cell_matched_baseline(
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_image_index=row_image_index,
        row_mask=row_mask,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        n_images=n_images,
    )
    moving_ssi = finite_ratio(float(moving["information_numerator_bits"]), float(moving["expected_spikes"]))
    cell_ssi = finite_ratio(float(cell["information_numerator_bits"]), float(cell["expected_spikes"]))
    return {
        "moving": moving,
        "cell": cell,
        "moving_ssi": moving_ssi,
        "cell_ssi": cell_ssi,
        "ssi_percent_vs_cell_baseline": panel_c._pct_delta(moving_ssi, cell_ssi),
    }


def _shared_edges(metrics: pd.DataFrame, drift_mask: np.ndarray, across_col: str, along_col: str) -> np.ndarray:
    across = pd.to_numeric(metrics.loc[drift_mask, across_col], errors="coerce").to_numpy(dtype=float)
    along = pd.to_numeric(metrics.loc[drift_mask, along_col], errors="coerce").to_numpy(dtype=float)
    across = across[np.isfinite(across) & (across > 0)]
    along = along[np.isfinite(along) & (along > 0)]
    pooled = np.concatenate([across, along])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    if pooled.size == 0 or across.size == 0 or along.size == 0:
        raise ValueError(f"No finite positive values for {across_col}/{along_col}")
    body_edges = np.quantile(pooled, BODY_QUANTILES)
    tail_low = max(float(np.quantile(across, TAIL_QUANTILE)), float(np.quantile(along, TAIL_QUANTILE)))
    tail_high = min(float(np.max(across)), float(np.max(along)))
    edges = np.asarray([*body_edges, tail_low, tail_high], dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    if np.any(np.diff(edges) <= 0):
        unique = np.unique(pooled)
        if unique.size < len(edges):
            raise ValueError(f"Cannot make tail-enriched bins for {across_col}/{along_col}")
        edges = np.quantile(unique, np.linspace(0.0, 1.0, len(edges)))
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges.astype(np.float64)


def _pooled_medians(metrics: pd.DataFrame, drift_mask: np.ndarray, across_col: str, along_col: str, edges: np.ndarray) -> np.ndarray:
    across_bins = _assign_bins(metrics[across_col].to_numpy(dtype=float), edges)
    along_bins = _assign_bins(metrics[along_col].to_numpy(dtype=float), edges)
    pooled = np.concatenate(
        [
            metrics.loc[drift_mask, across_col].to_numpy(dtype=float),
            metrics.loc[drift_mask, along_col].to_numpy(dtype=float),
        ]
    )
    pooled_bins = np.concatenate([across_bins[drift_mask], along_bins[drift_mask]])
    medians: list[float] = []
    for bin_index in range(len(edges) - 1):
        vals = pooled[pooled_bins == bin_index]
        vals = vals[np.isfinite(vals)]
        medians.append(float(np.nanmedian(vals)) if vals.size else float("nan"))
    return np.asarray(medians, dtype=float)


def _reference_context_by_family(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    reference: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        values = np.concatenate(
            [
                pd.to_numeric(metrics.loc[drift_mask, str(family["across"])], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(metrics.loc[drift_mask, str(family["along"])], errors="coerce").to_numpy(dtype=float),
            ]
        )
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            reference[str(family["key"])] = {}
            continue
        reference[str(family["key"])] = {
            "definition": (
                "Drift-only BackImage real-trace-bank component values pooled across contour-normal "
                "and contour-parallel projections, ignoring unit tuning and trace-contour alignment."
            ),
            "unit": str(family["unit"]),
            "n_component_values": int(values.size),
            "q25": float(np.nanpercentile(values, 25.0)),
            "median": float(np.nanmedian(values)),
            "q75": float(np.nanpercentile(values, 75.0)),
        }
    return reference


def _compute_family(
    data: dict[str, Any],
    metrics: pd.DataFrame,
    family: dict[str, str],
    *,
    population: dict[str, Any],
    population_index: int,
    family_index: int,
    unit_to_images: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    n_images = int(data["stabilized_ssi"].shape[0])
    edges = _shared_edges(metrics, drift_mask, str(family["across"]), str(family["along"]))
    medians = _pooled_medians(metrics, drift_mask, str(family["across"]), str(family["along"]), edges)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 10_000 * int(population_index) + 1000 * int(family_index))
    rows: list[dict[str, Any]] = []
    # Contrast populations for the true final bin (used by the untruncated
    # option sheet) and the second-to-last bin (used by any panel, like
    # panel_e_rms_excursion.py, that hides the noisy final bin via
    # EXCLUDE_LAST_BINS=1 -- that panel needs a bracket describing whichever
    # bin is actually its rightmost visible one, not the hidden one).
    contrast_populations: dict[int, dict[str, dict[str, Any]]] = {}

    for component, component_label, linestyle, marker, axis_key in COMPONENT_SPECS:
        metric_col = str(family[axis_key])
        bins = _assign_bins(metrics[metric_col].to_numpy(dtype=float), edges)
        for bin_index in range(len(edges) - 1):
            row_mask = drift_mask & (bins == bin_index)
            pop = _population_for_mask(
                data,
                row_image_index=row_image_index,
                row_mask=row_mask,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                n_images=n_images,
            )
            delta_stats = ratio_delta_stats(
                pop["moving"]["per_image_num"],
                pop["moving"]["per_image_den"],
                pop["cell"]["per_image_num"],
                pop["cell"]["per_image_den"],
                n_resamples=N_BOOTSTRAP,
                rng=rng,
            )
            delta_low = float(delta_stats["population_delta_ci95_low_image_boot"])
            delta_high = float(delta_stats["population_delta_ci95_high_image_boot"])
            cell_ssi = float(pop["cell_ssi"])
            values = pd.to_numeric(metrics.loc[row_mask, metric_col], errors="coerce").to_numpy(dtype=float)
            rows.append(
                {
                    "population_key": str(population["key"]),
                    "population_title": str(population["title"]),
                    "population_subtitle": str(population["subtitle"]),
                    "metric_family": str(family["key"]),
                    "metric_family_title": str(family["title"]),
                    "metric_family_description": str(family["description"]),
                    "component": component,
                    "component_label": component_label,
                    "component_metric": metric_col,
                    "component_bin_order": int(bin_index + 1),
                    "component_min": float(edges[bin_index]),
                    "component_max": float(edges[bin_index + 1]),
                    "plot_median": float(medians[bin_index]),
                    "component_median": float(np.nanmedian(values)) if values.size else float("nan"),
                    "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                    "n_movie_samples": int(pop["moving"]["n_movie_samples"]),
                    "n_images_contributing": int(pop["moving"]["n_images_contributing"]),
                    "moving_population_ssi_bits_per_spike": float(pop["moving_ssi"]),
                    "cell_baseline_population_ssi_bits_per_spike": cell_ssi,
                    "ssi_percent_vs_cell_baseline": float(pop["ssi_percent_vs_cell_baseline"]),
                    "population_delta_percent_ci95_low_image_boot": 100.0 * delta_low / cell_ssi
                    if math.isfinite(delta_low) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS
                    else float("nan"),
                    "population_delta_percent_ci95_high_image_boot": 100.0 * delta_high / cell_ssi
                    if math.isfinite(delta_high) and math.isfinite(cell_ssi) and abs(cell_ssi) > EPS
                    else float("nan"),
                    "population_delta_p_image_bootstrap_sign": float(delta_stats["population_delta_p_image_bootstrap_sign"]),
                    "linestyle": str(linestyle),
                    "marker": str(marker),
                }
            )
            if bin_index >= len(edges) - 3:
                contrast_populations.setdefault(bin_index, {})[component] = pop

    contrast_rng = np.random.default_rng(BOOTSTRAP_SEED + 10_000 * int(population_index) + 1000 * int(family_index) + 7)
    contrasts: list[dict[str, Any]] = []
    for bin_index in sorted(contrast_populations):
        pops = contrast_populations[bin_index]
        if not {"across", "along"}.issubset(pops):
            continue
        contrast = _bootstrap_residual_difference(pops["across"], pops["along"], rng=contrast_rng)
        contrast.update(
            {
                "population_key": str(population["key"]),
                "population_title": str(population["title"]),
                "metric_family": str(family["key"]),
                "metric_family_title": str(family["title"]),
                "component_bin_order": int(bin_index + 1),
                "last_bin_across_metric": str(family["across"]),
                "last_bin_along_metric": str(family["along"]),
            }
        )
        contrasts.append(contrast)

    meta = {
        "metric_family": str(family["key"]),
        "edges": edges,
        "pooled_medians": medians,
        "contrast": contrasts,
    }
    return pd.DataFrame(rows), meta


def _format_p(p: float) -> str:
    if not math.isfinite(float(p)):
        return "p=n/a"
    if float(p) < 0.001:
        return "p<0.001"
    return f"p={float(p):.3f}"


def _set_shared_ylim(axes: np.ndarray, values: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    vals = [0.0]
    for col in (
        "ssi_percent_vs_cell_baseline",
        "population_delta_percent_ci95_low_image_boot",
        "population_delta_percent_ci95_high_image_boot",
    ):
        arr = pd.to_numeric(values[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    for col in ("contrast_ci95_low_image_boot", "contrast_ci95_high_image_boot"):
        arr = pd.to_numeric(contrasts[col], errors="coerce").to_numpy(dtype=float)
        vals.extend(arr[np.isfinite(arr)].tolist())
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1.0)
    for ax in axes.ravel():
        ax.set_ylim(lo - 0.14 * span, hi + 0.34 * span)


def _draw_bracket(ax: plt.Axes, *, x: float, y0: float, y1: float, label: str) -> None:
    low, high = sorted([float(y0), float(y1)])
    x0, x1 = ax.get_xlim()
    tick = 0.018 * (x1 - x0)
    ax.plot([x, x], [low, high], color=ORANGE, lw=1.1, clip_on=False, zorder=7)
    ax.plot([x - tick, x], [low, low], color=ORANGE, lw=1.1, clip_on=False, zorder=7)
    ax.plot([x - tick, x], [high, high], color=ORANGE, lw=1.1, clip_on=False, zorder=7)
    ax.text(x + 0.35 * tick, 0.5 * (low + high), label, ha="left", va="center", fontsize=6.5, color=ORANGE)


def _add_reference_band(ax: plt.Axes, family: dict[str, str], reference: dict[str, dict[str, Any]]) -> None:
    context = reference.get(str(family["key"]), {})
    low = float(context.get("q25", np.nan))
    high = float(context.get("q75", np.nan))
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        return
    ax.axvspan(low, high, color=GRAY, alpha=0.13, lw=0, zorder=0, label="trace-bank q25-q75")


def _plot_sheet(
    values: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    population: dict[str, Any],
    reference: dict[str, dict[str, Any]],
) -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharey=True, constrained_layout=True)
    _set_shared_ylim(axes, values, contrasts)
    for ax, family in zip(axes.ravel(), FAMILIES, strict=True):
        frame = values[values["metric_family"].astype(str).eq(str(family["key"]))].copy()
        last_rows: dict[str, pd.Series] = {}
        _add_reference_band(ax, family, reference)
        for component, label, linestyle, marker, _axis_key in COMPONENT_SPECS:
            sub = frame[frame["component"].astype(str).eq(component)].sort_values("component_bin_order")
            x = sub["plot_median"].to_numpy(dtype=float)
            y = sub["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
            ci_low = sub["population_delta_percent_ci95_low_image_boot"].to_numpy(dtype=float)
            ci_high = sub["population_delta_percent_ci95_high_image_boot"].to_numpy(dtype=float)
            ax.plot(x, y, color=ORANGE, linestyle=linestyle, linewidth=1.8, label=label)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - ci_low, ci_high - y]),
                color=ORANGE,
                linestyle="none",
                marker=marker,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                elinewidth=0.95,
                capsize=2.0,
            )
            last_rows[component] = sub.iloc[-1]
        # This sheet always shows every bin, so it always wants the true
        # final bin's contrast (the largest component_bin_order), unlike
        # _fig4_option_sheet.py's truncated dose panels.
        family_contrasts = contrasts[contrasts["metric_family"].astype(str).eq(str(family["key"]))]
        contrast = family_contrasts.loc[family_contrasts["component_bin_order"].idxmax()]
        x_values = frame["plot_median"].to_numpy(dtype=float)
        finite_x = x_values[np.isfinite(x_values)]
        if finite_x.size:
            ref_context = reference.get(str(family["key"]), {})
            ref_vals = np.asarray([ref_context.get("q25", np.nan), ref_context.get("q75", np.nan)], dtype=float)
            ref_vals = ref_vals[np.isfinite(ref_vals)]
            all_x = np.concatenate([finite_x, ref_vals]) if ref_vals.size else finite_x
            pad = 0.07 * max(float(np.nanmax(all_x) - np.nanmin(all_x)), 1e-6)
            ax.set_xlim(float(np.nanmin(all_x)) - pad, float(np.nanmax(all_x)) + 1.55 * pad)
            _draw_bracket(
                ax,
                x=float(last_rows["across"]["plot_median"]) + 0.45 * pad,
                y0=float(last_rows["across"]["ssi_percent_vs_cell_baseline"]),
                y1=float(last_rows["along"]["ssi_percent_vs_cell_baseline"]),
                label=(
                    f"{float(contrast.across_minus_along_percent_point):+.1f} pp\n"
                    f"{_format_p(float(contrast.contrast_p_image_bootstrap_sign))}"
                ),
            )
        ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
        ax.grid(axis="y", color="#d8dde3", lw=0.75)
        ax.set_title(f"{family['title']}\n{family['description']}", loc="left", fontweight="bold", color=INK)
        ax.set_xlabel(str(family["xlabel"]))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("SSI residual (% vs cell baseline)")
    axes[1, 0].set_ylabel("SSI residual (% vs cell baseline)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        f"Panel-G analogs: {population['title']}\n{population['subtitle']}",
        fontsize=12.5,
        fontweight="bold",
    )
    return fig


def build(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    data = panel_c.load_dataset(panel_c.MATRIX_DIR)
    metrics = _compute_extended_component_metrics(data)
    reference = _reference_context_by_family(metrics)
    frames: list[pd.DataFrame] = []
    population_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for population_index, population in enumerate(POPULATION_SPECS):
        unit_to_images = _selected_unit_images_for_population(data, population)
        population_rows.append(
            {
                "population_key": str(population["key"]),
                "population_title": str(population["title"]),
                "population_subtitle": str(population["subtitle"]),
                "sf_group": str(population["sf_group"]),
                "relation": str(population["relation"]),
                "requires_orientation_tuning": bool(population["requires_orientation_tuning"]),
                "n_selected_units": int(len(unit_to_images)),
                "n_selected_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
            }
        )
        for family_index, family in enumerate(FAMILIES):
            frame, family_meta = _compute_family(
                data,
                metrics,
                family,
                population=population,
                population_index=population_index,
                family_index=family_index,
                unit_to_images=unit_to_images,
            )
            frames.append(frame)
            meta.append(
                {
                    "population_key": str(population["key"]),
                    **{key: value for key, value in family_meta.items() if key != "contrast"},
                }
            )
            contrasts.extend(family_meta["contrast"])
    values = pd.concat(frames, ignore_index=True)
    contrast_df = pd.DataFrame(contrasts)
    population_df = pd.DataFrame(population_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{OUT_STEM}.pdf"
    default_png = out_dir / f"{OUT_STEM}.png"
    default_svg = out_dir / f"{OUT_STEM}.svg"
    values_csv = out_dir / f"{OUT_STEM}_values.csv"
    contrast_csv = out_dir / f"{OUT_STEM}_last_bin_contrasts.csv"
    population_csv = out_dir / f"{OUT_STEM}_populations.csv"
    reference_csv = out_dir / f"{OUT_STEM}_trace_bank_reference.csv"
    provenance_json = out_dir / f"{OUT_STEM}_provenance.json"
    values.to_csv(values_csv, index=False)
    contrast_df.to_csv(contrast_csv, index=False)
    population_df.to_csv(population_csv, index=False)
    pd.DataFrame(
        [
            {
                "metric_family": key,
                **context,
            }
            for key, context in reference.items()
        ]
    ).to_csv(reference_csv, index=False)

    png_paths: dict[str, Path] = {}
    svg_paths: dict[str, Path] = {}
    with PdfPages(pdf) as pages:
        for population in POPULATION_SPECS:
            pop_values = values[values["population_key"].astype(str).eq(str(population["key"]))].copy()
            pop_contrasts = contrast_df[contrast_df["population_key"].astype(str).eq(str(population["key"]))].copy()
            fig = _plot_sheet(pop_values, pop_contrasts, population=population, reference=reference)
            png = out_dir / f"{OUT_STEM}_{population['key']}.png"
            svg = out_dir / f"{OUT_STEM}_{population['key']}.svg"
            fig.savefig(png, dpi=230)
            fig.savefig(svg)
            if str(population["key"]) == "high_sf_aligned":
                fig.savefig(default_png, dpi=230)
                fig.savefig(default_svg)
            pages.savefig(fig)
            png_paths[str(population["key"])] = png
            svg_paths[str(population["key"])] = svg
            plt.close(fig)
    _write_json(
        provenance_json,
        {
            "analysis": OUT_STEM,
            "selection": {
                "sf_group_mode": SF_GROUP_MODE,
                "sf_min_cpd": SF_MIN_CPD if SF_GROUP_MODE == "absolute_cpd" else None,
                "contour_coherence_min": CONTOUR_COHERENCE_MIN,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
                "populations": population_rows,
            },
            "binning": {
                "n_bins": int(len(values["component_bin_order"].unique())),
                "scheme": "shared drift-only bins pooled across the two component axes within each metric family; Panel-G-style body quantiles plus a high-tail bin",
                "body_quantiles": BODY_QUANTILES,
                "tail_quantile": TAIL_QUANTILE,
                "families": meta,
            },
            "x_axis_reference": reference,
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "paired image bootstrap; error bars are moving-vs-cell baseline; brackets are across-minus-along, computed at both the true final bin and the second-to-last bin so panels that hide the final bin (EXCLUDE_LAST_BINS>0) can still show a bracket at their own rightmost visible bin",
            },
            "outputs": {
                "pdf": pdf,
                "default_png": default_png,
                "default_svg": default_svg,
                "pngs": png_paths,
                "svgs": svg_paths,
                "values_csv": values_csv,
                "last_bin_contrasts_csv": contrast_csv,
                "populations_csv": population_csv,
                "trace_bank_reference_csv": reference_csv,
                "provenance_json": provenance_json,
            },
        },
    )
    return {
        "pdf": pdf,
        "default_png": default_png,
        "default_svg": default_svg,
        **{f"png_{key}": path for key, path in png_paths.items()},
        **{f"svg_{key}": path for key, path in svg_paths.items()},
        "values_csv": values_csv,
        "last_bin_contrasts_csv": contrast_csv,
        "populations_csv": population_csv,
        "trace_bank_reference_csv": reference_csv,
        "provenance_json": provenance_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    paths = build(args.out_dir)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
