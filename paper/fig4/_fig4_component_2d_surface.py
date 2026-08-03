#!/usr/bin/env python3
"""2D component surfaces for BackImage real-trace SSI.

This is the joint-bin counterpart to the marginal component path/RMS panels.
Rows are across-contour component bins, columns are along-contour component
bins.  The estimand is the spike-weighted population SSI used by the reordered
story figure.
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

import _fig4_paths as _paths

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from _fig4_trace_schematics import (
    accumulate_population,
    accumulate_population_movie_rows,
    baseline_rows_by_image,
    build_movie_row_grid,
    load_dataset,
    unit_image_selection,
)


# Refresh-path write targets; anchored rather than CWD-relative (see the note
# in _fig4_geometry_story.py).
MATRIX_DIR = _paths.TRACE_BANK_MERGED_DIR
CONDITION_DIR = _paths.FIG_DIR / "trace_conditioning"
OUT_DIR = _paths.FIG_DIR / "plot_collections"
OUT_STEM = "backimage_real_trace_component_2d_surface_diagnostic"

SF_GROUP = "high_sf"
RELATION = "contour_matched"
RELATION_LABEL = "Aligned high-SF units on strong contours"
MIN_OSI = 0.05
MATCH_MAX_DEG = 22.5
ORTHOGONAL_MIN_DEG = 67.5
N_BINS = 8
EPS = 1e-12

FAMILIES = [
    {
        "key": "path",
        "title": "Component path length",
        "across": "across_path_arcmin",
        "along": "along_path_arcmin",
        "unit": "arcmin",
        "description": "sum absolute projected frame-to-frame displacement",
    },
    {
        "key": "rms",
        "title": "Component RMS excursion",
        "across": "across_rms_arcmin",
        "along": "along_rms_arcmin",
        "unit": "arcmin",
        "description": "RMS of centered trace position projected onto each contour-relative axis",
    },
]
OUTCOMES = [
    (
        "population_ssi_percent_vs_stabilized",
        "SSI bits/spike\n% vs stabilized",
        "SSI efficiency",
        "RdBu_r",
    ),
    (
        "information_bits_per_sample_percent_vs_stabilized",
        "information bits/window\n% vs stabilized",
        "Information",
        "RdBu_r",
    ),
    (
        "expected_spikes_per_sample_percent_vs_stabilized",
        "expected spikes/window\n% vs stabilized",
        "Expected spikes",
        "RdBu_r",
    ),
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > EPS else float("nan")


def _pct_delta(value: float, baseline: float) -> float:
    if not (math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > EPS):
        return float("nan")
    return 100.0 * (value - baseline) / baseline


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("Cannot bin an empty metric.")
    edges = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1))
    span = max(float(edges[-1] - edges[0]), 1e-6)
    edges[0] -= 1e-6 * span
    edges[-1] += 1e-6 * span
    return edges


def _assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=int)
    ok = np.isfinite(arr)
    out[ok] = np.searchsorted(edges[1:-1], arr[ok], side="right")
    out[(out < 0) | (out >= len(edges) - 1)] = -1
    return out


def _compute_component_metrics(data: dict[str, Any]) -> pd.DataFrame:
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
    invalid = ~np.isfinite(axes)
    for arr in (along_path, across_path, along_rms, across_rms):
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
            "has_microsaccade": has_ms.to_numpy(dtype=bool),
            "context": np.where(has_ms.to_numpy(dtype=bool), "microsaccade", "drift_only"),
            "rendered_path_length_arcmin": pd.to_numeric(
                movie["rendered_path_length_arcmin"],
                errors="coerce",
            ).to_numpy(dtype=float),
            "along_path_arcmin": along_path,
            "across_path_arcmin": across_path,
            "along_rms_arcmin": along_rms,
            "across_rms_arcmin": across_rms,
        }
    )


def _population_metrics(
    pop: dict[str, Any],
    *,
    baseline_ssi: float,
    baseline_information_bits_per_sample: float,
    baseline_expected_spikes_per_sample: float,
) -> dict[str, float]:
    ssi = float(pop["population_ssi_bits_per_spike"])
    info = _finite_ratio(float(pop["information_numerator_bits"]), float(pop["n_movie_samples"]))
    spikes = _finite_ratio(float(pop["expected_spikes"]), float(pop["n_movie_samples"]))
    return {
        "population_ssi_bits_per_spike": ssi,
        "population_ssi_percent_vs_stabilized": _pct_delta(ssi, baseline_ssi),
        "information_bits_per_sample": info,
        "information_bits_per_sample_percent_vs_stabilized": _pct_delta(
            info,
            baseline_information_bits_per_sample,
        ),
        "expected_spikes_per_sample": spikes,
        "expected_spikes_per_sample_percent_vs_stabilized": _pct_delta(
            spikes,
            baseline_expected_spikes_per_sample,
        ),
        "information_numerator_bits": float(pop["information_numerator_bits"]),
        "expected_spikes": float(pop["expected_spikes"]),
    }


def _make_surfaces(data: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    baseline = accumulate_population(
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
    baseline_ssi = float(baseline["population_ssi_bits_per_spike"])
    baseline_info = _finite_ratio(float(baseline["information_numerator_bits"]), float(baseline["n_movie_samples"]))
    baseline_spikes = _finite_ratio(float(baseline["expected_spikes"]), float(baseline["n_movie_samples"]))

    rows: list[dict[str, Any]] = []
    drift_global = metrics[metrics["context"].astype(str).eq("drift_only")]
    row_image_index = metrics["image_index"].astype(int).to_numpy()
    for family in FAMILIES:
        across_col = str(family["across"])
        along_col = str(family["along"])
        across_edges = _quantile_edges(drift_global[across_col].to_numpy(dtype=float), N_BINS)
        along_edges = _quantile_edges(drift_global[along_col].to_numpy(dtype=float), N_BINS)
        across_bins = _assign_bins(metrics[across_col].to_numpy(dtype=float), across_edges)
        along_bins = _assign_bins(metrics[along_col].to_numpy(dtype=float), along_edges)
        drift_mask = metrics["context"].astype(str).eq("drift_only").to_numpy(dtype=bool)
        for across_bin in range(N_BINS):
            for along_bin in range(N_BINS):
                row_mask = drift_mask & (across_bins == across_bin) & (along_bins == along_bin)
                pop = accumulate_population_movie_rows(
                    ssi=data["ssi"],
                    expected=data["expected"],
                    row_image_index=row_image_index,
                    row_mask=row_mask,
                    unit_to_images=unit_to_images,
                    n_images=n_images,
                )
                global_rows = metrics[row_mask]
                rows.append(
                    {
                        "metric_family": str(family["key"]),
                        "metric_family_title": str(family["title"]),
                        "metric_family_description": str(family["description"]),
                        "relation": RELATION,
                        "relation_label": RELATION_LABEL,
                        "sf_group": SF_GROUP,
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
                        "n_movie_samples": int(pop["n_movie_samples"]),
                        "n_images_contributing": int(pop["n_images_contributing"]),
                        "baseline_population_ssi_bits_per_spike": baseline_ssi,
                        "baseline_information_bits_per_sample": baseline_info,
                        "baseline_expected_spikes_per_sample": baseline_spikes,
                        **_population_metrics(
                            pop,
                            baseline_ssi=baseline_ssi,
                            baseline_information_bits_per_sample=baseline_info,
                            baseline_expected_spikes_per_sample=baseline_spikes,
                        ),
                    }
                )
    metadata = {
        "n_units": int(len(unit_to_images)),
        "n_unit_image_pairs": int(sum(len(images) for images in unit_to_images.values())),
        "baseline": {
            "population_ssi_bits_per_spike": baseline_ssi,
            "information_bits_per_sample": baseline_info,
            "expected_spikes_per_sample": baseline_spikes,
        },
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


def _symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    mag = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1.0)
    return -1.05 * mag, 1.05 * mag


def _format_count(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value >= 1000:
        return f"{value / 1000.0:.1f}k"
    return f"{int(round(value))}"


def _plot_family(surface: pd.DataFrame, family: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.75), constrained_layout=False)
    fig.suptitle(
        f"{family['title']} surface: {RELATION_LABEL}",
        fontsize=13.0,
        y=0.985,
    )
    xlabels = _labels(surface, "along")
    ylabels = _labels(surface, "across")
    for idx, (value_col, color_label, title, cmap) in enumerate(OUTCOMES):
        ax = axes[idx]
        values = _grid(surface, value_col)
        vmin, vmax = _symmetric_limits(values)
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10.0)
        ax.set_xticks(np.arange(N_BINS), xlabels)
        ax.set_yticks(np.arange(N_BINS), ylabels)
        ax.set_xlabel("along-contour bin; median arcmin", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("across-contour bin; median arcmin", fontsize=8.5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.4)
        threshold = 0.52 * max(abs(vmin), abs(vmax))
        for across_bin in range(N_BINS):
            for along_bin in range(N_BINS):
                value = values[across_bin, along_bin]
                if math.isfinite(value):
                    ax.text(
                        along_bin,
                        across_bin,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if abs(value) >= threshold else "0.16",
                    )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(color_label, fontsize=7.2)
        cbar.ax.tick_params(labelsize=7.0)

    count_ax = axes[3]
    counts = _grid(surface, "n_movie_samples")
    count_image = count_ax.imshow(np.log10(np.maximum(counts, 1.0)), origin="lower", cmap="viridis", aspect="auto")
    count_ax.set_title("Selected samples", fontsize=10.0)
    count_ax.set_xticks(np.arange(N_BINS), xlabels)
    count_ax.set_yticks(np.arange(N_BINS), ylabels)
    count_ax.set_yticklabels([])
    count_ax.set_xlabel("along-contour bin; median arcmin", fontsize=8.5)
    count_ax.tick_params(labelsize=7.4)
    for across_bin in range(N_BINS):
        for along_bin in range(N_BINS):
            count_ax.text(
                along_bin,
                across_bin,
                _format_count(counts[across_bin, along_bin]),
                ha="center",
                va="center",
                fontsize=6.3,
                color="white" if counts[across_bin, along_bin] >= np.nanmedian(counts) else "0.12",
            )
    cbar = fig.colorbar(count_image, ax=count_ax, fraction=0.046, pad=0.025)
    cbar.set_label("log10 samples", fontsize=7.2)
    cbar.ax.tick_params(labelsize=7.0)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.text(
        0.5,
        0.025,
        "Rows increase across-contour component size; columns increase along-contour component size. "
        "Bins are marginal drift-only quantiles crossed into joint cells; values are spike-weighted population point estimates.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.84, bottom=0.18, wspace=0.36)
    return fig


def _plot_surfaces(summary: pd.DataFrame) -> tuple[Path, list[Path]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    png_paths: list[Path] = []
    with PdfPages(pdf) as pages:
        for family in FAMILIES:
            surface = summary[summary["metric_family"].eq(str(family["key"]))].copy()
            fig = _plot_family(surface, family)
            pages.savefig(fig, bbox_inches="tight")
            png = OUT_DIR / f"{OUT_STEM}_{family['key']}.png"
            fig.savefig(png, dpi=240, bbox_inches="tight")
            png_paths.append(png)
            plt.close(fig)
    return pdf, png_paths


def main() -> None:
    data = load_dataset(MATRIX_DIR)
    metrics = _compute_component_metrics(data)
    summary, metadata = _make_surfaces(data, metrics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"{OUT_STEM}_values.csv"
    json_path = OUT_DIR / f"{OUT_STEM}_summary.json"
    summary.to_csv(csv, index=False)
    pdf, pngs = _plot_surfaces(summary)
    _write_json(
        json_path,
        {
            "analysis": "backimage_real_trace_component_2d_surface_diagnostic",
            "matrix_dir": MATRIX_DIR,
            "out_dir": OUT_DIR,
            "outputs": {
                "pdf": pdf,
                "pngs": pngs,
                "csv": csv,
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
            "n_bins": N_BINS,
            "families": FAMILIES,
            "note": (
                "This plot crosses marginal equal-count drift-only bins for the two component axes. "
                "It is a joint surface for the story-panel spike-weighted population estimand, not a causal manipulation."
            ),
        },
    )
    print(pdf)
    for png in pngs:
        print(png)
    print(csv)
    print(json_path)


if __name__ == "__main__":
    main()
