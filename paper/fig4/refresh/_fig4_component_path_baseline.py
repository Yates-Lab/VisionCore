#!/usr/bin/env python3
"""Decompose component path-length surfaces into composition and motion terms.

For each across/along component path-length cell, this diagnostic splits the
usual moving-vs-global-stabilized surface into:

    moving - global stabilized
      = cell-matched stabilized - global stabilized
      + moving - cell-matched stabilized

The additive plotted units are percent of the global stabilized baseline.
Cell-relative motion percentages are also saved in the CSV.
"""

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from _fig4_component_2d_surface import (
    MATRIX_DIR,
    MIN_OSI,
    N_BINS,
    ORTHOGONAL_MIN_DEG,
    OUT_DIR,
    SF_GROUP,
    _assign_bins,
    _compute_component_metrics,
    _finite_ratio,
    _format_count,
    _json_ready,
    _pct_delta,
    _quantile_edges,
    _symmetric_limits,
)
from _fig4_trace_schematics import (
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    unit_image_selection,
)


OUT_STEM = "backimage_real_trace_component_path_baseline_decomposition_surface"
RELATION = "contour_matched"
RELATION_LABEL = "High-SF units matched to strong-contour axis"
MATCH_MAX_DEG = 22.5
EPS = 1e-12
PATH_FAMILY = {
    "key": "path",
    "title": "Component path length",
    "across": "across_path_arcmin",
    "along": "along_path_arcmin",
    "unit": "arcmin",
    "description": "sum absolute projected frame-to-frame displacement",
}
OUTCOMES = [
    {
        "key": "ssi",
        "title": "SSI bits/spike",
        "moving_col": "population_ssi_bits_per_spike",
        "global_col": "global_baseline_population_ssi_bits_per_spike",
        "cell_col": "cell_baseline_population_ssi_bits_per_spike",
        "unit": "% of global stabilized SSI",
    },
    {
        "key": "information",
        "title": "Information bits/window",
        "moving_col": "information_bits_per_sample",
        "global_col": "global_baseline_information_bits_per_sample",
        "cell_col": "cell_baseline_information_bits_per_sample",
        "unit": "% of global stabilized information",
    },
    {
        "key": "spikes",
        "title": "Expected spikes/window",
        "moving_col": "expected_spikes_per_sample",
        "global_col": "global_baseline_expected_spikes_per_sample",
        "cell_col": "cell_baseline_expected_spikes_per_sample",
        "unit": "% of global stabilized spikes",
    },
]
SURFACES = [
    ("global_effect_percent_of_global", "Moving vs global stabilized"),
    ("composition_effect_percent_of_global", "Cell-stabilized composition"),
    ("motion_effect_percent_of_global", "Moving vs cell-stabilized residual"),
]
HIGHLIGHT_CELLS = [
    (8, 1, "#20262c", "Q8/Q1"),
    (1, 8, "#20262c", "Q1/Q8"),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct_of_reference(value: float, reference: float) -> float:
    if not (math.isfinite(value) and math.isfinite(reference) and abs(reference) > EPS):
        return float("nan")
    return 100.0 * value / reference


def _cell_matched_baseline(
    *,
    stabilized_ssi: np.ndarray,
    stabilized_expected: np.ndarray,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
) -> dict[str, Any]:
    candidate_rows = np.flatnonzero(np.asarray(row_mask, dtype=bool))
    candidate_images = np.asarray(row_image_index[candidate_rows], dtype=int)
    counts_by_image = np.bincount(candidate_images, minlength=n_images) if candidate_images.size else np.zeros(n_images, dtype=int)
    total_num = 0.0
    total_den = 0.0
    per_image_num = np.zeros(n_images, dtype=np.float64)
    per_image_den = np.zeros(n_images, dtype=np.float64)
    n_movie_samples = 0

    for unit_index, image_indices in unit_to_images.items():
        images = np.asarray(image_indices, dtype=int)
        if images.size == 0:
            continue
        counts = counts_by_image[images].astype(np.float64, copy=False)
        keep = counts > 0
        if not np.any(keep):
            continue
        images = images[keep]
        counts = counts[keep]
        baseline_rows = np.asarray([baseline_lookup[int(image_idx)] for image_idx in images], dtype=int)
        value = np.asarray(stabilized_ssi[baseline_rows, int(unit_index)], dtype=np.float64)
        weight = np.asarray(stabilized_expected[baseline_rows, int(unit_index)], dtype=np.float64)
        ok = np.isfinite(value) & np.isfinite(weight) & np.isfinite(counts)
        if not np.any(ok):
            continue
        images = images[ok]
        counts = counts[ok]
        value = value[ok]
        weight = weight[ok]
        numer = value * weight * counts
        denom = weight * counts
        per_image_num[images] += numer
        per_image_den[images] += denom
        total_num += float(np.nansum(numer))
        total_den += float(np.nansum(denom))
        n_movie_samples += int(np.nansum(counts))

    return {
        "population_ssi_bits_per_spike": _finite_ratio(total_num, total_den),
        "information_numerator_bits": total_num,
        "expected_spikes": total_den,
        "per_image_num": per_image_num,
        "per_image_den": per_image_den,
        "n_movie_samples": n_movie_samples,
        "n_images_contributing": int(np.count_nonzero(per_image_den > EPS)),
    }


def _population_values(pop: dict[str, Any]) -> dict[str, float]:
    n = float(pop["n_movie_samples"])
    return {
        "population_ssi_bits_per_spike": float(pop["population_ssi_bits_per_spike"]),
        "information_bits_per_sample": _finite_ratio(float(pop["information_numerator_bits"]), n),
        "expected_spikes_per_sample": _finite_ratio(float(pop["expected_spikes"]), n),
    }


def _metric_rows(
    *,
    moving_values: dict[str, float],
    global_values: dict[str, float],
    cell_values: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for outcome in OUTCOMES:
        key = str(outcome["key"])
        moving = float(moving_values[str(outcome["moving_col"])])
        global_base = float(global_values[str(outcome["moving_col"])])
        cell_base = float(cell_values[str(outcome["moving_col"])])
        global_effect = moving - global_base
        composition = cell_base - global_base
        motion = moving - cell_base
        out[f"{key}_global_effect_percent_of_global"] = _pct_of_reference(global_effect, global_base)
        out[f"{key}_composition_effect_percent_of_global"] = _pct_of_reference(composition, global_base)
        out[f"{key}_motion_effect_percent_of_global"] = _pct_of_reference(motion, global_base)
        out[f"{key}_motion_effect_percent_vs_cell_baseline"] = _pct_delta(moving, cell_base)
        out[f"{key}_moving_vs_global_percent"] = _pct_delta(moving, global_base)
        out[f"{key}_cell_baseline_vs_global_percent"] = _pct_delta(cell_base, global_base)
        out[f"{key}_moving_value"] = moving
        out[f"{key}_global_baseline_value"] = global_base
        out[f"{key}_cell_baseline_value"] = cell_base
    return out


def _compute_surfaces(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    selections = unit_image_selection(
        data["unit"],
        data["image"],
        relation=RELATION,
        sf_groups=[SF_GROUP],
        min_osi=MIN_OSI,
        match_max_deg=MATCH_MAX_DEG,
        orthogonal_min_deg=ORTHOGONAL_MIN_DEG,
        image_axis_col="image_edge_axis_deg",
    )
    unit_to_images = selections[SF_GROUP]
    n_images = int(data["stabilized_ssi"].shape[0])
    row_image_index = metrics["image_index"].astype(int).to_numpy()

    global_baseline = accumulate_population(
        ssi=data["ssi"],
        expected=data["expected"],
        stabilized_ssi=data["stabilized_ssi"],
        stabilized_expected=data["stabilized_expected"],
        row_grid=row_grid,
        baseline_lookup=baseline_lookup,
        unit_to_images=unit_to_images,
        trace_indices=None,
        n_images=n_images,
    )
    global_values = _population_values(global_baseline)

    drift_global = metrics[metrics["context"].astype(str).eq("drift_only")]
    across_col = str(PATH_FAMILY["across"])
    along_col = str(PATH_FAMILY["along"])
    across_edges = _quantile_edges(drift_global[across_col].to_numpy(dtype=float), N_BINS)
    along_edges = _quantile_edges(drift_global[along_col].to_numpy(dtype=float), N_BINS)
    across_bins = _assign_bins(metrics[across_col].to_numpy(dtype=float), across_edges)
    along_bins = _assign_bins(metrics[along_col].to_numpy(dtype=float), along_edges)
    drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    for across_bin in range(N_BINS):
        for along_bin in range(N_BINS):
            row_mask = drift_mask & (across_bins == across_bin) & (along_bins == along_bin)
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
            moving_values = _population_values(moving_pop)
            cell_values = _population_values(cell_pop)
            global_rows = metrics[row_mask]
            rows.append(
                {
                    "metric_family": PATH_FAMILY["key"],
                    "metric_family_title": PATH_FAMILY["title"],
                    "relation": RELATION,
                    "relation_label": RELATION_LABEL,
                    "sf_group": SF_GROUP,
                    "match_max_deg": MATCH_MAX_DEG,
                    "across_bin": int(across_bin + 1),
                    "along_bin": int(along_bin + 1),
                    "across_bin_label": f"Q{across_bin + 1}",
                    "along_bin_label": f"Q{along_bin + 1}",
                    "across_min_arcmin": float(across_edges[across_bin]),
                    "across_max_arcmin": float(across_edges[across_bin + 1]),
                    "along_min_arcmin": float(along_edges[along_bin]),
                    "along_max_arcmin": float(along_edges[along_bin + 1]),
                    "across_median_arcmin": float(np.nanmedian(global_rows[across_col])),
                    "along_median_arcmin": float(np.nanmedian(global_rows[along_col])),
                    "n_movie_rows_global": int(np.count_nonzero(row_mask)),
                    "n_movie_samples": int(moving_pop["n_movie_samples"]),
                    "n_images_contributing": int(moving_pop["n_images_contributing"]),
                    "cell_baseline_n_movie_samples": int(cell_pop["n_movie_samples"]),
                    "cell_baseline_n_images_contributing": int(cell_pop["n_images_contributing"]),
                    "global_baseline_population_ssi_bits_per_spike": global_values["population_ssi_bits_per_spike"],
                    "global_baseline_information_bits_per_sample": global_values["information_bits_per_sample"],
                    "global_baseline_expected_spikes_per_sample": global_values["expected_spikes_per_sample"],
                    "cell_baseline_population_ssi_bits_per_spike": cell_values["population_ssi_bits_per_spike"],
                    "cell_baseline_information_bits_per_sample": cell_values["information_bits_per_sample"],
                    "cell_baseline_expected_spikes_per_sample": cell_values["expected_spikes_per_sample"],
                    "population_ssi_bits_per_spike": moving_values["population_ssi_bits_per_spike"],
                    "information_bits_per_sample": moving_values["information_bits_per_sample"],
                    "expected_spikes_per_sample": moving_values["expected_spikes_per_sample"],
                    **_metric_rows(
                        moving_values=moving_values,
                        global_values=global_values,
                        cell_values=cell_values,
                    ),
                }
            )

    metadata = {
        "n_units": int(len(unit_to_images)),
        "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        "global_baseline": global_values,
        "across_edges_arcmin": across_edges,
        "along_edges_arcmin": along_edges,
    }
    return pd.DataFrame(rows), metadata


def _grid(surface: pd.DataFrame, value_col: str) -> np.ndarray:
    arr = np.full((N_BINS, N_BINS), np.nan, dtype=float)
    for row in surface.itertuples(index=False):
        arr[int(row.across_bin) - 1, int(row.along_bin) - 1] = float(getattr(row, value_col))
    return arr


def _labels(surface: pd.DataFrame, axis: str) -> list[str]:
    labels = []
    for idx in range(1, N_BINS + 1):
        if axis == "across":
            values = surface[surface["across_bin"].eq(idx)]["across_median_arcmin"].to_numpy(dtype=float)
        else:
            values = surface[surface["along_bin"].eq(idx)]["along_median_arcmin"].to_numpy(dtype=float)
        labels.append(f"Q{idx}\n{float(np.nanmedian(values)):.1f}")
    return labels


def _annotate_grid(ax: plt.Axes, values: np.ndarray, *, vmin: float, vmax: float, fmt: str = "{:+.0f}") -> None:
    threshold = 0.52 * max(abs(vmin), abs(vmax))
    for across_bin in range(N_BINS):
        for along_bin in range(N_BINS):
            value = values[across_bin, along_bin]
            if math.isfinite(value):
                ax.text(
                    along_bin,
                    across_bin,
                    fmt.format(float(value)),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if abs(value) >= threshold else "0.16",
                )


def _highlight_cells(ax: plt.Axes) -> None:
    for across_bin, along_bin, color, _label in HIGHLIGHT_CELLS:
        ax.add_patch(
            Rectangle(
                (int(along_bin) - 1.5, int(across_bin) - 1.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=color,
                linewidth=1.6,
            )
        )


def _scaled_colormap(values: np.ndarray) -> tuple[dict[str, Any], float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"cmap": "RdBu_r", "vmin": -1.0, "vmax": 1.0}, -1.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin < 0.0 < vmax:
        return {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)}, vmin, vmax
    if vmax <= 0.0:
        lo = min(vmin, -1e-6)
        return {"cmap": "Blues_r", "vmin": lo, "vmax": 0.0}, lo, 0.0
    hi = max(vmax, 1e-6)
    return {"cmap": "Reds", "vmin": 0.0, "vmax": hi}, 0.0, hi


def _plot_outcome(surface: pd.DataFrame, outcome: dict[str, str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.9), constrained_layout=False)
    fig.suptitle(
        f"{outcome['title']} component path baseline decomposition",
        fontsize=13.2,
        y=0.985,
    )
    xlabels = _labels(surface, "along")
    ylabels = _labels(surface, "across")
    value_cols = [f"{outcome['key']}_{suffix}" for suffix, _title in SURFACES]
    all_values = np.concatenate([surface[col].to_numpy(dtype=float) for col in value_cols])
    vmin, vmax = _symmetric_limits(all_values)

    for idx, (suffix, title) in enumerate(SURFACES):
        ax = axes[idx]
        col = f"{outcome['key']}_{suffix}"
        values = _grid(surface, col)
        image = ax.imshow(values, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9.8)
        ax.set_xticks(np.arange(N_BINS), xlabels)
        ax.set_yticks(np.arange(N_BINS), ylabels)
        ax.set_xlabel("along-contour bin; median arcmin", fontsize=8.3)
        if idx == 0:
            ax.set_ylabel("across-contour bin; median arcmin", fontsize=8.3)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.2)
        _annotate_grid(ax, values, vmin=vmin, vmax=vmax)
        _highlight_cells(ax)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(str(outcome["unit"]), fontsize=6.9)
        cbar.ax.tick_params(labelsize=6.7)

    count_ax = axes[3]
    counts = _grid(surface, "n_movie_samples")
    count_image = count_ax.imshow(np.log10(np.maximum(counts, 1.0)), origin="lower", cmap="viridis", aspect="auto")
    count_ax.set_title("Selected samples", fontsize=9.8)
    count_ax.set_xticks(np.arange(N_BINS), xlabels)
    count_ax.set_yticks(np.arange(N_BINS), [])
    count_ax.set_xlabel("along-contour bin; median arcmin", fontsize=8.3)
    count_ax.tick_params(labelsize=7.2)
    median_count = float(np.nanmedian(counts))
    for across_bin in range(N_BINS):
        for along_bin in range(N_BINS):
            count_ax.text(
                along_bin,
                across_bin,
                _format_count(counts[across_bin, along_bin]),
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if counts[across_bin, along_bin] >= median_count else "0.12",
            )
    _highlight_cells(count_ax)
    cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.046, pad=0.025)
    cbar.set_label("log10 samples", fontsize=6.9)
    cbar.ax.tick_params(labelsize=6.7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.text(
        0.5,
        0.025,
        (
            "Additive identity in plotted units: moving vs global = composition + residual motion. "
            "Outlined cells are Q8 across/Q1 along and Q1 across/Q8 along."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.83, bottom=0.18, wspace=0.34)
    return fig


def _plot_motion_residual_scaled(surface: pd.DataFrame, outcome: dict[str, str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.9), constrained_layout=False)
    fig.suptitle(
        f"{outcome['title']} residual motion surface with local color scaling",
        fontsize=13.0,
        y=0.985,
    )
    xlabels = _labels(surface, "along")
    ylabels = _labels(surface, "across")
    panels = [
        (
            f"{outcome['key']}_motion_effect_percent_of_global",
            "Moving vs cell-stabilized residual\n% of global stabilized",
            str(outcome["unit"]),
        ),
        (
            f"{outcome['key']}_motion_effect_percent_vs_cell_baseline",
            "Moving vs cell-stabilized residual\n% of cell baseline",
            "% vs cell baseline",
        ),
    ]
    for idx, (value_col, title, color_label) in enumerate(panels):
        ax = axes[idx]
        values = _grid(surface, value_col)
        image_kwargs, vmin, vmax = _scaled_colormap(values)
        image = ax.imshow(values, origin="lower", aspect="auto", **image_kwargs)
        ax.set_title(title, fontsize=9.7)
        ax.set_xticks(np.arange(N_BINS), xlabels)
        ax.set_yticks(np.arange(N_BINS), ylabels)
        ax.set_xlabel("along-contour bin; median arcmin", fontsize=8.3)
        if idx == 0:
            ax.set_ylabel("across-contour bin; median arcmin", fontsize=8.3)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.2)
        _annotate_grid(ax, values, vmin=vmin, vmax=vmax)
        _highlight_cells(ax)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(color_label, fontsize=6.9)
        cbar.ax.tick_params(labelsize=6.7)

    count_ax = axes[2]
    counts = _grid(surface, "n_movie_samples")
    count_image = count_ax.imshow(np.log10(np.maximum(counts, 1.0)), origin="lower", cmap="viridis", aspect="auto")
    count_ax.set_title("Selected samples", fontsize=9.7)
    count_ax.set_xticks(np.arange(N_BINS), xlabels)
    count_ax.set_yticks(np.arange(N_BINS), [])
    count_ax.set_xlabel("along-contour bin; median arcmin", fontsize=8.3)
    count_ax.tick_params(labelsize=7.2)
    median_count = float(np.nanmedian(counts))
    for across_bin in range(N_BINS):
        for along_bin in range(N_BINS):
            count_ax.text(
                along_bin,
                across_bin,
                _format_count(counts[across_bin, along_bin]),
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if counts[across_bin, along_bin] >= median_count else "0.12",
            )
    _highlight_cells(count_ax)
    cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.046, pad=0.025)
    cbar.set_label("log10 samples", fontsize=6.9)
    cbar.ax.tick_params(labelsize=6.7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.text(
        0.5,
        0.025,
        (
            "This is the third decomposition column redrawn with color limits from the residual surface only. "
            "Outlined cells are Q8 across/Q1 along and Q1 across/Q8 along."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.82, bottom=0.18, wspace=0.32)
    return fig


def _corner_extract(surface: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (surface["across_bin"].eq(8) & surface["along_bin"].eq(1))
        | (surface["across_bin"].eq(1) & surface["along_bin"].eq(8))
        | (surface["across_bin"].eq(1) & surface["along_bin"].between(6, 8))
        | (surface["across_bin"].eq(8) & surface["along_bin"].between(1, 4))
    )
    cols = [
        "across_bin",
        "along_bin",
        "across_median_arcmin",
        "along_median_arcmin",
        "ssi_global_effect_percent_of_global",
        "ssi_composition_effect_percent_of_global",
        "ssi_motion_effect_percent_of_global",
        "ssi_motion_effect_percent_vs_cell_baseline",
        "information_global_effect_percent_of_global",
        "information_composition_effect_percent_of_global",
        "information_motion_effect_percent_of_global",
        "information_motion_effect_percent_vs_cell_baseline",
        "spikes_global_effect_percent_of_global",
        "spikes_composition_effect_percent_of_global",
        "spikes_motion_effect_percent_of_global",
        "spikes_motion_effect_percent_vs_cell_baseline",
        "n_movie_samples",
        "n_images_contributing",
    ]
    return surface.loc[keep, cols].sort_values(["across_bin", "along_bin"]).reset_index(drop=True)


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    summary, metadata = _compute_surfaces(data, metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    corner_csv = OUT_DIR / f"{OUT_STEM}_corner_extract.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    residual_pdf = OUT_DIR / f"{OUT_STEM}_motion_residual_scaled.pdf"
    pngs: list[Path] = []
    residual_pngs: list[Path] = []

    summary.to_csv(csv, index=False)
    corner = _corner_extract(summary)
    corner.to_csv(corner_csv, index=False)
    with PdfPages(pdf) as pages:
        for outcome in OUTCOMES:
            fig = _plot_outcome(summary, outcome)
            pages.savefig(fig, bbox_inches="tight")
            png = OUT_DIR / f"{OUT_STEM}_{outcome['key']}.png"
            fig.savefig(png, dpi=240, bbox_inches="tight")
            pngs.append(png)
            plt.close(fig)
    with PdfPages(residual_pdf) as pages:
        for outcome in OUTCOMES:
            fig = _plot_motion_residual_scaled(summary, outcome)
            pages.savefig(fig, bbox_inches="tight")
            png = OUT_DIR / f"{OUT_STEM}_motion_residual_scaled_{outcome['key']}.png"
            fig.savefig(png, dpi=240, bbox_inches="tight")
            residual_pngs.append(png)
            plt.close(fig)

    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_path_baseline_decomposition_surface",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf,
                "motion_residual_scaled_pdf": residual_pdf,
                "pngs": pngs,
                "motion_residual_scaled_pngs": residual_pngs,
                "csv": csv,
                "corner_extract_csv": corner_csv,
                "summary_json": json_path,
            },
            "selection": {
                "relation": RELATION,
                "relation_label": RELATION_LABEL,
                "sf_group": SF_GROUP,
                "min_osi": MIN_OSI,
                "match_max_deg": MATCH_MAX_DEG,
                "orthogonal_min_deg": ORTHOGONAL_MIN_DEG,
                **metadata,
            },
            "metric_family": PATH_FAMILY,
            "n_bins": N_BINS,
            "surface_identity": (
                "global_effect_percent_of_global = composition_effect_percent_of_global "
                "+ motion_effect_percent_of_global for each outcome."
            ),
            "note": (
                "Cell-matched stabilized baseline repeats each selected unit-image stabilized response by the "
                "number of movie rows from that image in the component cell."
            ),
        },
    )
    print(pdf)
    for png in pngs:
        print(png)
    print(residual_pdf)
    for png in residual_pngs:
        print(png)
    print(csv)
    print(corner_csv)
    print(json_path)


if __name__ == "__main__":
    main()
