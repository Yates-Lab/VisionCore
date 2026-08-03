#!/usr/bin/env python3
"""Plot real-trace BackImage SSI schematics with explicit aggregation modes.

The paired outputs make the estimand visible:

- unit_first: each unit is averaged over selected image x trace movies, then
  summarized across units.
- population: unit SSI values are weighted by expected spikes and accumulated
  before converting back to bits/spike.

The matrix row contract is the row order of movie_feature_table.csv.  Do not use
the shard-local matrix_row_index column for indexing merged matrix files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter


from VisionCore.paths import VISIONCORE_ROOT as ROOT
DEFAULT_MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged"
)
SF_ORDER = ["low_sf", "middle_sf", "high_sf"]
SF_LABELS = {"low_sf": "Low SF", "middle_sf": "Middle SF", "high_sf": "High SF"}
SF_COLORS = {"low_sf": "#1f77b4", "middle_sf": "#009E73", "high_sf": "#d62728"}
CONTEXT_LABELS = {
    "stabilized": "counterfactual stabilized",
    "drift_only": "no detected microsaccade",
    "microsaccade": ">=1 detected microsaccade",
}
COMPONENT_PATH_SPECS = [
    ("across_path_arcmin", "Across-contour path length"),
    ("along_path_arcmin", "Along-contour path length"),
]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sf-groups", default="low_sf,middle_sf,high_sf")
    parser.add_argument("--n-drift-bins", type=int, default=8)
    parser.add_argument("--n-microsaccade-bins", type=int, default=5)
    parser.add_argument("--match-max-deg", type=float, default=22.5)
    parser.add_argument("--orthogonal-min-deg", type=float, default=67.5)
    parser.add_argument("--min-osi", type=float, default=0.05)
    parser.add_argument("--image-axis-col", type=str, default="image_edge_axis_deg")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=47)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(val) for val in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sem(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def axis_delta_deg(a_deg: Any, b_deg: Any) -> np.ndarray:
    a = np.asarray(a_deg, dtype=np.float64)
    b = np.asarray(b_deg, dtype=np.float64)
    return np.abs(0.5 * np.degrees(np.angle(np.exp(2j * np.radians(a - b)))))


def microsaccade_count(frame: pd.DataFrame) -> pd.Series:
    for col in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0).clip(lower=0).astype(int)
    return pd.Series(np.zeros(frame.shape[0], dtype=int), index=frame.index)


def bool_column_mask(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Expected boolean column {column!r} in image feature table.")
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"}).to_numpy(dtype=bool)


def compute_component_movie_metrics(
    movie: pd.DataFrame,
    trace: pd.DataFrame,
    trace_xy: np.ndarray,
    *,
    image_axis_col: str,
) -> pd.DataFrame:
    required = {"movie_index", "image_index", "trace_index", image_axis_col}
    missing = sorted(required.difference(movie.columns))
    if missing:
        raise ValueError(f"Movie table is missing required columns for component metrics: {missing}")
    trace_index = movie["trace_index"].astype(int).to_numpy()
    if np.any((trace_index < 0) | (trace_index >= trace_xy.shape[0])):
        raise ValueError("movie.trace_index contains values outside trace_xy.")

    axes = pd.to_numeric(movie[image_axis_col], errors="coerce").to_numpy(dtype=np.float64)
    theta = np.radians(axes)
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    xy = np.asarray(trace_xy, dtype=np.float32)
    steps = np.diff(xy, axis=1)[trace_index]
    along_step = steps[:, :, 0] * cos_t[:, None] + steps[:, :, 1] * sin_t[:, None]
    across_step = -steps[:, :, 0] * sin_t[:, None] + steps[:, :, 1] * cos_t[:, None]
    along_path = np.nansum(np.abs(along_step), axis=1) * 60.0
    across_path = np.nansum(np.abs(across_step), axis=1) * 60.0
    invalid_axis = ~np.isfinite(axes)
    along_path[invalid_axis] = np.nan
    across_path[invalid_axis] = np.nan

    trace_has_ms = (microsaccade_count(trace) > 0).to_numpy(dtype=bool)
    has_ms = trace_has_ms[trace_index]
    context = np.where(has_ms, "microsaccade", "drift_only")
    return pd.DataFrame(
        {
            "movie_index": movie["movie_index"].astype(int).to_numpy(),
            "image_index": movie["image_index"].astype(int).to_numpy(),
            "trace_index": trace_index,
            image_axis_col: axes,
            "has_microsaccade": has_ms,
            "context": context,
            "context_label": [CONTEXT_LABELS[str(key)] for key in context],
            "across_path_arcmin": across_path,
            "along_path_arcmin": along_path,
            "rendered_path_length_arcmin": pd.to_numeric(
                movie["rendered_path_length_arcmin"],
                errors="coerce",
            ).to_numpy(dtype=np.float64),
        },
    )


def load_dataset(matrix_dir: Path) -> dict[str, Any]:
    ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r")
    expected = np.load(matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r")
    stabilized_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy", mmap_mode="r")
    stabilized_expected = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy", mmap_mode="r")
    trace_xy = np.load(matrix_dir / "trace_xy.npy", mmap_mode="r")
    movie = pd.read_csv(matrix_dir / "movie_feature_table.csv")
    image = pd.read_csv(matrix_dir / "image_feature_table.csv")
    trace = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    unit = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    baseline_table = pd.read_csv(matrix_dir / "stabilized_movie_feature_table.csv")

    if ssi.shape != expected.shape:
        raise ValueError(f"ssi_matrix shape {ssi.shape} does not match expected_spikes_matrix shape {expected.shape}.")
    if ssi.shape[0] != movie.shape[0]:
        raise ValueError("ssi_matrix rows must match movie_feature_table row order.")
    if ssi.shape[1] != unit.shape[0]:
        raise ValueError("ssi_matrix columns must match unit_feature_table rows.")
    if stabilized_ssi.shape != stabilized_expected.shape:
        raise ValueError("stabilized SSI and expected-spike arrays must have the same shape.")
    if stabilized_ssi.shape[0] != image.shape[0] or stabilized_ssi.shape[1] != unit.shape[0]:
        raise ValueError("stabilized arrays must be image x unit.")
    if trace_xy.shape[0] != trace.shape[0]:
        raise ValueError("trace_xy rows must match trace_feature_table rows.")
    return {
        "ssi": ssi,
        "expected": expected,
        "stabilized_ssi": stabilized_ssi,
        "stabilized_expected": stabilized_expected,
        "trace_xy": trace_xy,
        "movie": movie,
        "image": image,
        "trace": trace,
        "unit": unit,
        "baseline_table": baseline_table,
    }


def add_equal_count_trace_bins(trace: pd.DataFrame, *, n_drift_bins: int, n_microsaccade_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = trace.copy()
    if "trace_bank_index" not in work.columns:
        work = work.copy()
        work.insert(0, "trace_bank_index", np.arange(work.shape[0], dtype=int))
    work["has_microsaccade"] = microsaccade_count(work) > 0
    work["rendered_path_length_arcmin"] = pd.to_numeric(work["rendered_path_length_arcmin"], errors="coerce")
    work["context"] = pd.NA
    work["context_label"] = pd.NA
    work["path_bin"] = pd.NA
    work["path_bin_order"] = pd.NA

    rows: list[dict[str, Any]] = []
    specs = [
        (False, "drift_only", n_drift_bins),
        (True, "microsaccade", n_microsaccade_bins),
    ]
    for has_ms, context, n_bins in specs:
        sub = work[work["has_microsaccade"].eq(has_ms) & work["rendered_path_length_arcmin"].notna()].copy()
        sub = sub.sort_values(["rendered_path_length_arcmin", "trace_bank_index"], kind="mergesort")
        if sub.empty:
            continue
        chunks = np.array_split(sub.index.to_numpy(dtype=int), int(n_bins))
        for order, chunk in enumerate(chunks, start=1):
            if chunk.size == 0:
                continue
            label = f"{context}_q{order:02d}"
            vals = work.loc[chunk, "rendered_path_length_arcmin"].to_numpy(dtype=float)
            work.loc[chunk, "context"] = context
            work.loc[chunk, "context_label"] = CONTEXT_LABELS[context]
            work.loc[chunk, "path_bin"] = label
            work.loc[chunk, "path_bin_order"] = order
            rows.append(
                {
                    "context": context,
                    "context_label": CONTEXT_LABELS[context],
                    "path_bin": label,
                    "path_bin_order": order,
                    "n_traces": int(chunk.size),
                    "min_path_arcmin": float(np.nanmin(vals)),
                    "q25_path_arcmin": float(np.nanpercentile(vals, 25.0)),
                    "median_path_arcmin": float(np.nanmedian(vals)),
                    "q75_path_arcmin": float(np.nanpercentile(vals, 75.0)),
                    "max_path_arcmin": float(np.nanmax(vals)),
                }
            )
    bins = pd.DataFrame(rows).sort_values(["context", "path_bin_order"]).reset_index(drop=True)
    return work, bins


def add_equal_count_component_bins(
    metrics: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
    n_drift_bins: int,
    n_microsaccade_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = metrics.copy()
    bin_col = f"{metric_col}_bin"
    order_col = f"{metric_col}_bin_order"
    work[bin_col] = pd.NA
    work[order_col] = pd.NA
    values = pd.to_numeric(work[metric_col], errors="coerce")
    rows: list[dict[str, Any]] = []
    specs = [
        (False, "drift_only", n_drift_bins),
        (True, "microsaccade", n_microsaccade_bins),
    ]
    for has_ms, context, n_bins in specs:
        sub = work[
            work["has_microsaccade"].eq(has_ms)
            & values.notna()
            & np.isfinite(values)
            & (values > 0)
        ].copy()
        sub = sub.sort_values([metric_col, "movie_index"], kind="mergesort")
        if sub.empty:
            continue
        chunks = np.array_split(sub.index.to_numpy(dtype=int), int(n_bins))
        for order, chunk in enumerate(chunks, start=1):
            if chunk.size == 0:
                continue
            label = f"{context}_q{order:02d}"
            vals = values.loc[chunk].to_numpy(dtype=float)
            trace_ids = work.loc[chunk, "trace_index"].astype(int).to_numpy()
            work.loc[chunk, bin_col] = label
            work.loc[chunk, order_col] = order
            rows.append(
                {
                    "component_metric": metric_col,
                    "component_metric_label": metric_label,
                    "component_bin_col": bin_col,
                    "context": context,
                    "context_label": CONTEXT_LABELS[context],
                    "component_bin": label,
                    "component_bin_order": int(order),
                    "n_movie_rows_global": int(chunk.size),
                    "n_unique_traces_global": int(np.unique(trace_ids).size),
                    "min_component_arcmin": float(np.nanmin(vals)),
                    "q25_component_arcmin": float(np.nanpercentile(vals, 25.0)),
                    "median_component_arcmin": float(np.nanmedian(vals)),
                    "q75_component_arcmin": float(np.nanpercentile(vals, 75.0)),
                    "max_component_arcmin": float(np.nanmax(vals)),
                }
            )
    bins = pd.DataFrame(rows).sort_values(["component_metric", "context", "component_bin_order"]).reset_index(drop=True)
    return work, bins


def build_movie_row_grid(movie: pd.DataFrame) -> np.ndarray:
    images = movie["image_index"].astype(int).to_numpy()
    traces = movie["trace_index"].astype(int).to_numpy()
    row_grid = np.full((int(images.max()) + 1, int(traces.max()) + 1), -1, dtype=int)
    row_numbers = np.arange(movie.shape[0], dtype=int)
    if pd.DataFrame({"image_index": images, "trace_index": traces}).duplicated().any():
        raise ValueError("movie_feature_table has duplicate image x trace rows.")
    row_grid[images, traces] = row_numbers
    return row_grid


def baseline_rows_by_image(image: pd.DataFrame, baseline_table: pd.DataFrame) -> dict[int, int]:
    if "baseline_row_index" not in baseline_table.columns:
        baseline_table = baseline_table.copy()
        baseline_table["baseline_row_index"] = np.arange(baseline_table.shape[0], dtype=int)
    return {
        int(row.image_index): int(row.baseline_row_index)
        for row in baseline_table[["image_index", "baseline_row_index"]].drop_duplicates("image_index").itertuples(index=False)
    }


def unit_image_selection(
    unit: pd.DataFrame,
    image: pd.DataFrame,
    *,
    relation: str,
    sf_groups: list[str],
    min_osi: float,
    match_max_deg: float,
    orthogonal_min_deg: float,
    image_axis_col: str = "image_edge_axis_deg",
) -> dict[str, dict[int, np.ndarray]]:
    if image_axis_col not in image.columns:
        raise ValueError(f"image table is missing {image_axis_col!r}")
    image_axis = pd.to_numeric(image[image_axis_col], errors="coerce").to_numpy(dtype=float)
    image_indices = image["image_index"].astype(int).to_numpy()
    finite_axis = np.isfinite(image_axis)
    strong_contour = bool_column_mask(image, "image_contour_strong")
    selections: dict[str, dict[int, np.ndarray]] = {sf_group: {} for sf_group in sf_groups}
    for unit_row in unit[unit["sf_group"].astype(str).isin(sf_groups)].itertuples(index=False):
        sf_group = str(unit_row.sf_group)
        unit_index = int(unit_row.unit_index)
        pref = float(unit_row.prior_preferred_orientation_deg)
        osi = float(unit_row.prior_orientation_selectivity_index)
        if relation == "all_images_no_osi":
            selected = image_indices[finite_axis]
        elif relation == "strong_contours_no_osi":
            selected = image_indices[finite_axis & strong_contour]
        else:
            if not (math.isfinite(pref) and math.isfinite(osi) and osi >= float(min_osi)):
                selected = np.asarray([], dtype=int)
            else:
                delta = axis_delta_deg(image_axis, pref)
                if relation == "contour_matched":
                    selected = image_indices[strong_contour & np.isfinite(delta) & (delta <= float(match_max_deg))]
                elif relation == "contour_intermediate":
                    selected = image_indices[
                        strong_contour
                        & np.isfinite(delta)
                        & (delta > float(match_max_deg))
                        & (delta < float(orthogonal_min_deg))
                    ]
                elif relation == "contour_orthogonal":
                    selected = image_indices[strong_contour & np.isfinite(delta) & (delta >= float(orthogonal_min_deg))]
                elif relation == "all_orientation_tuned":
                    selected = image_indices[np.isfinite(delta)]
                else:
                    raise ValueError(f"Unknown relation {relation!r}")
        if selected.size:
            selections[sf_group][unit_index] = np.asarray(selected, dtype=int)
    return selections


def finite_ratio(num: float, den: float) -> float:
    return float(num / den) if den > EPS and math.isfinite(num) and math.isfinite(den) else float("nan")


def bootstrap_ratio_ci(
    per_image_num: np.ndarray,
    per_image_den: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    ok = np.isfinite(per_image_num) & np.isfinite(per_image_den) & (per_image_den > EPS)
    num = np.asarray(per_image_num[ok], dtype=np.float64)
    den = np.asarray(per_image_den[ok], dtype=np.float64)
    n = int(num.size)
    if n <= 1 or int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    sample = rng.integers(0, n, size=(int(n_bootstrap), n))
    values = np.nansum(num[sample], axis=1) / np.maximum(np.nansum(den[sample], axis=1), EPS)
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def ratio_delta_stats(
    condition_num: np.ndarray,
    condition_den: np.ndarray,
    baseline_num: np.ndarray,
    baseline_den: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired image-bootstrap CI and two-sided bootstrap sign p for a ratio delta."""
    cond_num = np.asarray(condition_num, dtype=np.float64)
    cond_den = np.asarray(condition_den, dtype=np.float64)
    base_num = np.asarray(baseline_num, dtype=np.float64)
    base_den = np.asarray(baseline_den, dtype=np.float64)
    ok = (
        np.isfinite(cond_num)
        & np.isfinite(cond_den)
        & np.isfinite(base_num)
        & np.isfinite(base_den)
        & ((cond_den > EPS) | (base_den > EPS))
    )
    cond_num = cond_num[ok]
    cond_den = cond_den[ok]
    base_num = base_num[ok]
    base_den = base_den[ok]
    n = int(cond_num.size)
    observed = finite_ratio(float(np.nansum(cond_num)), float(np.nansum(cond_den))) - finite_ratio(
        float(np.nansum(base_num)),
        float(np.nansum(base_den)),
    )
    if n <= 1 or int(n_resamples) <= 0:
        return {
            "population_delta_ci95_low_image_boot": float("nan"),
            "population_delta_ci95_high_image_boot": float("nan"),
            "population_delta_p_image_bootstrap_sign": float("nan"),
        }

    sample = rng.integers(0, n, size=(int(n_resamples), n))
    boot_values = np.nansum(cond_num[sample], axis=1) / np.maximum(np.nansum(cond_den[sample], axis=1), EPS)
    boot_values -= np.nansum(base_num[sample], axis=1) / np.maximum(np.nansum(base_den[sample], axis=1), EPS)
    lo, hi = np.nanpercentile(boot_values, [2.5, 97.5])
    below = (float(np.count_nonzero(boot_values <= 0.0)) + 1.0) / (float(int(n_resamples)) + 1.0)
    above = (float(np.count_nonzero(boot_values >= 0.0)) + 1.0) / (float(int(n_resamples)) + 1.0)
    p = min(1.0, 2.0 * min(below, above))
    return {
        "population_delta_ci95_low_image_boot": float(lo),
        "population_delta_ci95_high_image_boot": float(hi),
        "population_delta_p_image_bootstrap_sign": float(min(max(p, 0.0), 1.0)),
    }


def holm_adjust(p_values: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    vals = p[valid]
    order = np.argsort(vals)
    ranked = vals[order]
    m = int(ranked.size)
    adjusted_ranked = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, value in enumerate(ranked):
        adjusted = float((m - rank) * value)
        running = max(running, adjusted)
        adjusted_ranked[rank] = min(running, 1.0)
    adjusted_vals = np.empty(m, dtype=np.float64)
    adjusted_vals[order] = adjusted_ranked
    out[valid] = adjusted_vals
    return out


def accumulate_population(
    *,
    ssi: np.ndarray,
    expected: np.ndarray,
    stabilized_ssi: np.ndarray,
    stabilized_expected: np.ndarray,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    unit_to_images: dict[int, np.ndarray],
    trace_indices: np.ndarray | None,
    n_images: int,
) -> dict[str, Any]:
    total_num = 0.0
    total_den = 0.0
    per_image_num = np.zeros(n_images, dtype=np.float64)
    per_image_den = np.zeros(n_images, dtype=np.float64)
    n_movie_samples = 0
    for unit_index, image_indices in unit_to_images.items():
        images = np.asarray(image_indices, dtype=int)
        if images.size == 0:
            continue
        if trace_indices is None:
            baseline_rows = np.asarray([baseline_lookup[int(image_idx)] for image_idx in images], dtype=int)
            value = np.asarray(stabilized_ssi[baseline_rows, int(unit_index)], dtype=np.float64)
            weight = np.asarray(stabilized_expected[baseline_rows, int(unit_index)], dtype=np.float64)
            numer = value * weight
            per_image_num[images] += numer
            per_image_den[images] += weight
            total_num += float(np.nansum(numer))
            total_den += float(np.nansum(weight))
            n_movie_samples += int(value.size)
        else:
            traces = np.asarray(trace_indices, dtype=int)
            rows = row_grid[np.ix_(images, traces)]
            if np.any(rows < 0):
                raise ValueError("Missing image x trace rows in movie_feature_table.")
            value = np.asarray(ssi[rows, int(unit_index)], dtype=np.float64)
            weight = np.asarray(expected[rows, int(unit_index)], dtype=np.float64)
            numer = value * weight
            per_image_num[images] += np.nansum(numer, axis=1)
            per_image_den[images] += np.nansum(weight, axis=1)
            total_num += float(np.nansum(numer))
            total_den += float(np.nansum(weight))
            n_movie_samples += int(value.size)
    return {
        "population_ssi_bits_per_spike": finite_ratio(total_num, total_den),
        "information_numerator_bits": total_num,
        "expected_spikes": total_den,
        "per_image_num": per_image_num,
        "per_image_den": per_image_den,
        "n_movie_samples": n_movie_samples,
        "n_images_contributing": int(np.count_nonzero(per_image_den > EPS)),
    }


def accumulate_population_movie_rows(
    *,
    ssi: np.ndarray,
    expected: np.ndarray,
    row_image_index: np.ndarray,
    row_mask: np.ndarray,
    unit_to_images: dict[int, np.ndarray],
    n_images: int,
) -> dict[str, Any]:
    total_num = 0.0
    total_den = 0.0
    per_image_num = np.zeros(n_images, dtype=np.float64)
    per_image_den = np.zeros(n_images, dtype=np.float64)
    n_movie_samples = 0
    row_mask = np.asarray(row_mask, dtype=bool)
    candidate_rows = np.flatnonzero(row_mask)
    candidate_images = np.asarray(row_image_index[candidate_rows], dtype=int)
    for unit_index, image_indices in unit_to_images.items():
        images = np.asarray(image_indices, dtype=int)
        if images.size == 0 or candidate_rows.size == 0:
            continue
        selected_image_mask = np.zeros(n_images, dtype=bool)
        selected_image_mask[images] = True
        keep = selected_image_mask[candidate_images]
        rows = candidate_rows[keep]
        image_for_row = candidate_images[keep]
        if rows.size == 0:
            continue
        value = np.asarray(ssi[rows, int(unit_index)], dtype=np.float64)
        weight = np.asarray(expected[rows, int(unit_index)], dtype=np.float64)
        numer = value * weight
        ok = np.isfinite(numer) & np.isfinite(weight)
        if not np.any(ok):
            continue
        np.add.at(per_image_num, image_for_row[ok], numer[ok])
        np.add.at(per_image_den, image_for_row[ok], weight[ok])
        total_num += float(np.nansum(numer[ok]))
        total_den += float(np.nansum(weight[ok]))
        n_movie_samples += int(np.count_nonzero(ok))
    return {
        "population_ssi_bits_per_spike": finite_ratio(total_num, total_den),
        "information_numerator_bits": total_num,
        "expected_spikes": total_den,
        "per_image_num": per_image_num,
        "per_image_den": per_image_den,
        "n_movie_samples": n_movie_samples,
        "n_images_contributing": int(np.count_nonzero(per_image_den > EPS)),
    }


def build_curves_for_relation(
    *,
    relation: str,
    relation_label: str,
    selections: dict[str, dict[int, np.ndarray]],
    sf_groups: list[str],
    trace: pd.DataFrame,
    trace_bins: pd.DataFrame,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    ssi: np.ndarray,
    expected: np.ndarray,
    stabilized_ssi: np.ndarray,
    stabilized_expected: np.ndarray,
    unit: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_images = int(stabilized_ssi.shape[0])
    trace_ids_by_bin = {
        str(path_bin): sub["trace_bank_index"].astype(int).to_numpy()
        for path_bin, sub in trace[trace["path_bin"].notna()].groupby("path_bin", sort=False)
    }
    unit_rows: list[dict[str, Any]] = []
    pop_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    unit_meta = unit.set_index("unit_index", drop=False)

    for sf_group in sf_groups:
        unit_to_images = selections.get(sf_group, {})
        for unit_index, image_indices in unit_to_images.items():
            unit_rec = unit_meta.loc[int(unit_index)]
            selection_rows.append(
                {
                    "relation": relation,
                    "relation_label": relation_label,
                    "sf_group": sf_group,
                    "unit_index": int(unit_index),
                    "unit_label": str(unit_rec.get("unit_label", f"u{int(unit_index):03d}")),
                    "prior_preferred_orientation_deg": float(unit_rec.get("prior_preferred_orientation_deg", np.nan)),
                    "prior_orientation_selectivity_index": float(unit_rec.get("prior_orientation_selectivity_index", np.nan)),
                    "n_selected_images": int(len(image_indices)),
                    "selected_image_indices": " ".join(str(int(idx)) for idx in image_indices),
                }
            )

        baseline_pop = accumulate_population(
            ssi=ssi,
            expected=expected,
            stabilized_ssi=stabilized_ssi,
            stabilized_expected=stabilized_expected,
            row_grid=row_grid,
            baseline_lookup=baseline_lookup,
            unit_to_images=unit_to_images,
            trace_indices=None,
            n_images=n_images,
        )
        baseline_ci = bootstrap_ratio_ci(
            baseline_pop["per_image_num"],
            baseline_pop["per_image_den"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        baseline_value = float(baseline_pop["population_ssi_bits_per_spike"])
        pop_rows.append(
            {
                "relation": relation,
                "relation_label": relation_label,
                "sf_group": sf_group,
                "context": "stabilized",
                "context_label": CONTEXT_LABELS["stabilized"],
                "path_bin": "stabilized_zero_motion",
                "path_bin_order": 0,
                "path_median_arcmin": 0.0,
                "n_traces": 0,
                "n_units": int(len(unit_to_images)),
                "n_images_contributing": baseline_pop["n_images_contributing"],
                "n_movie_samples": baseline_pop["n_movie_samples"],
                "population_ssi_bits_per_spike": baseline_value,
                "population_ssi_delta_vs_stabilized": 0.0,
                "population_ci95_low_image_boot": baseline_ci[0],
                "population_ci95_high_image_boot": baseline_ci[1],
                "population_delta_ci95_low_image_boot": float("nan"),
                "population_delta_ci95_high_image_boot": float("nan"),
                "population_delta_p_image_bootstrap_sign": float("nan"),
                "information_numerator_bits": baseline_pop["information_numerator_bits"],
                "expected_spikes": baseline_pop["expected_spikes"],
            }
        )
        for unit_index, image_indices in unit_to_images.items():
            baseline_rows = np.asarray([baseline_lookup[int(image_idx)] for image_idx in image_indices], dtype=int)
            values = np.asarray(stabilized_ssi[baseline_rows, int(unit_index)], dtype=np.float64)
            unit_rows.append(
                {
                    "relation": relation,
                    "relation_label": relation_label,
                    "sf_group": sf_group,
                    "unit_index": int(unit_index),
                    "context": "stabilized",
                    "context_label": CONTEXT_LABELS["stabilized"],
                    "path_bin": "stabilized_zero_motion",
                    "path_bin_order": 0,
                    "path_median_arcmin": 0.0,
                    "n_traces": 0,
                    "n_selected_images": int(len(image_indices)),
                    "n_movie_samples": int(values.size),
                    "unit_ssi_bits_per_spike": float(np.nanmean(values)) if values.size else float("nan"),
                }
            )

        for bin_row in trace_bins.itertuples(index=False):
            path_bin = str(bin_row.path_bin)
            traces = trace_ids_by_bin[path_bin]
            pop = accumulate_population(
                ssi=ssi,
                expected=expected,
                stabilized_ssi=stabilized_ssi,
                stabilized_expected=stabilized_expected,
                row_grid=row_grid,
                baseline_lookup=baseline_lookup,
                unit_to_images=unit_to_images,
                trace_indices=traces,
                n_images=n_images,
            )
            ci = bootstrap_ratio_ci(
                pop["per_image_num"],
                pop["per_image_den"],
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            value = float(pop["population_ssi_bits_per_spike"])
            delta_stats = ratio_delta_stats(
                pop["per_image_num"],
                pop["per_image_den"],
                baseline_pop["per_image_num"],
                baseline_pop["per_image_den"],
                n_resamples=n_bootstrap,
                rng=rng,
            )
            pop_rows.append(
                {
                    "relation": relation,
                    "relation_label": relation_label,
                    "sf_group": sf_group,
                    "context": str(bin_row.context),
                    "context_label": str(bin_row.context_label),
                    "path_bin": path_bin,
                    "path_bin_order": int(bin_row.path_bin_order),
                    "path_median_arcmin": float(bin_row.median_path_arcmin),
                    "n_traces": int(bin_row.n_traces),
                    "n_units": int(len(unit_to_images)),
                    "n_images_contributing": pop["n_images_contributing"],
                    "n_movie_samples": pop["n_movie_samples"],
                    "population_ssi_bits_per_spike": value,
                    "population_ssi_delta_vs_stabilized": value - baseline_value,
                    "population_ci95_low_image_boot": ci[0],
                    "population_ci95_high_image_boot": ci[1],
                    **delta_stats,
                    "information_numerator_bits": pop["information_numerator_bits"],
                    "expected_spikes": pop["expected_spikes"],
                }
            )
            for unit_index, image_indices in unit_to_images.items():
                rows = row_grid[np.ix_(np.asarray(image_indices, dtype=int), traces)]
                if np.any(rows < 0):
                    raise ValueError("Missing image x trace rows in movie_feature_table.")
                values = np.asarray(ssi[rows, int(unit_index)], dtype=np.float64).reshape(-1)
                unit_rows.append(
                    {
                        "relation": relation,
                        "relation_label": relation_label,
                        "sf_group": sf_group,
                        "unit_index": int(unit_index),
                        "context": str(bin_row.context),
                        "context_label": str(bin_row.context_label),
                        "path_bin": path_bin,
                        "path_bin_order": int(bin_row.path_bin_order),
                        "path_median_arcmin": float(bin_row.median_path_arcmin),
                        "n_traces": int(bin_row.n_traces),
                        "n_selected_images": int(len(image_indices)),
                        "n_movie_samples": int(values.size),
                        "unit_ssi_bits_per_spike": float(np.nanmean(values)) if values.size else float("nan"),
                    }
                )

    unit_curves = pd.DataFrame(unit_rows)
    unit_summary_rows: list[dict[str, Any]] = []
    for keys, sub in unit_curves.groupby(["relation", "sf_group", "context", "path_bin"], sort=False):
        relation_key, sf_group, context, path_bin = keys
        values = pd.to_numeric(sub["unit_ssi_bits_per_spike"], errors="coerce")
        base = unit_curves[
            unit_curves["relation"].astype(str).eq(str(relation_key))
            & unit_curves["sf_group"].astype(str).eq(str(sf_group))
            & unit_curves["context"].astype(str).eq("stabilized")
        ][["unit_index", "unit_ssi_bits_per_spike"]].rename(columns={"unit_ssi_bits_per_spike": "unit_baseline_ssi"})
        paired = sub.merge(base, on="unit_index", how="left", validate="one_to_one")
        delta = pd.to_numeric(paired["unit_ssi_bits_per_spike"], errors="coerce") - pd.to_numeric(
            paired["unit_baseline_ssi"],
            errors="coerce",
        )
        unit_summary_rows.append(
            {
                "relation": str(relation_key),
                "relation_label": str(sub["relation_label"].iloc[0]),
                "sf_group": str(sf_group),
                "context": str(context),
                "context_label": str(sub["context_label"].iloc[0]),
                "path_bin": str(path_bin),
                "path_bin_order": int(sub["path_bin_order"].iloc[0]),
                "path_median_arcmin": float(sub["path_median_arcmin"].iloc[0]),
                "n_traces": int(sub["n_traces"].iloc[0]),
                "n_units": int(sub["unit_index"].nunique()),
                "mean_unit_ssi_bits_per_spike": float(np.nanmean(values)),
                "sem_unit_ssi_bits_per_spike": sem(values),
                "median_unit_ssi_bits_per_spike": float(np.nanmedian(values)),
                "mean_unit_ssi_delta_vs_stabilized": float(np.nanmean(delta)),
                "sem_unit_ssi_delta_vs_stabilized": sem(delta),
                "median_unit_ssi_delta_vs_stabilized": float(np.nanmedian(delta)),
                "mean_selected_images_per_unit": float(np.nanmean(sub["n_selected_images"].to_numpy(dtype=float))),
                "mean_movie_samples_per_unit": float(np.nanmean(sub["n_movie_samples"].to_numpy(dtype=float))),
            }
        )
    unit_summary = pd.DataFrame(unit_summary_rows)
    population_summary = pd.DataFrame(pop_rows)
    population_summary["population_delta_q_holm_bootstrap_sign_relation"] = holm_adjust(
        population_summary["population_delta_p_image_bootstrap_sign"],
    )
    return pd.DataFrame(selection_rows), unit_curves, unit_summary, population_summary


def build_component_population_summary_for_relation(
    *,
    relation: str,
    relation_label: str,
    selections: dict[str, dict[int, np.ndarray]],
    sf_groups: list[str],
    component_metrics: pd.DataFrame,
    component_bins_by_metric: dict[str, pd.DataFrame],
    ssi: np.ndarray,
    expected: np.ndarray,
    stabilized_ssi: np.ndarray,
    stabilized_expected: np.ndarray,
    row_grid: np.ndarray,
    baseline_lookup: dict[int, int],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    n_images = int(stabilized_ssi.shape[0])
    row_image_index = component_metrics["image_index"].astype(int).to_numpy()
    pop_rows: list[dict[str, Any]] = []

    for sf_group in sf_groups:
        unit_to_images = selections.get(sf_group, {})
        baseline_pop = accumulate_population(
            ssi=ssi,
            expected=expected,
            stabilized_ssi=stabilized_ssi,
            stabilized_expected=stabilized_expected,
            row_grid=row_grid,
            baseline_lookup=baseline_lookup,
            unit_to_images=unit_to_images,
            trace_indices=None,
            n_images=n_images,
        )
        baseline_ci = bootstrap_ratio_ci(
            baseline_pop["per_image_num"],
            baseline_pop["per_image_den"],
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        baseline_value = float(baseline_pop["population_ssi_bits_per_spike"])
        for metric_col, metric_label in COMPONENT_PATH_SPECS:
            pop_rows.append(
                {
                    "relation": relation,
                    "relation_label": relation_label,
                    "sf_group": sf_group,
                    "component_metric": metric_col,
                    "component_metric_label": metric_label,
                    "context": "stabilized",
                    "context_label": CONTEXT_LABELS["stabilized"],
                    "component_bin": "stabilized_zero_motion",
                    "component_bin_order": 0,
                    "component_median_arcmin": 0.0,
                    "n_movie_rows_global": 0,
                    "n_unique_traces_global": 0,
                    "n_units": int(len(unit_to_images)),
                    "n_images_contributing": baseline_pop["n_images_contributing"],
                    "n_movie_samples": baseline_pop["n_movie_samples"],
                    "population_ssi_bits_per_spike": baseline_value,
                    "population_ssi_delta_vs_stabilized": 0.0,
                    "population_ci95_low_image_boot": baseline_ci[0],
                    "population_ci95_high_image_boot": baseline_ci[1],
                    "population_delta_ci95_low_image_boot": float("nan"),
                    "population_delta_ci95_high_image_boot": float("nan"),
                    "population_delta_p_image_bootstrap_sign": float("nan"),
                    "information_numerator_bits": baseline_pop["information_numerator_bits"],
                    "expected_spikes": baseline_pop["expected_spikes"],
                }
            )
            bin_defs = component_bins_by_metric.get(metric_col, pd.DataFrame())
            bin_col = f"{metric_col}_bin"
            if bin_defs.empty or bin_col not in component_metrics.columns:
                continue
            metric_bin_values = component_metrics[bin_col].astype(object)
            for bin_row in bin_defs.itertuples(index=False):
                component_bin = str(bin_row.component_bin)
                row_mask = metric_bin_values.eq(component_bin).to_numpy(dtype=bool)
                pop = accumulate_population_movie_rows(
                    ssi=ssi,
                    expected=expected,
                    row_image_index=row_image_index,
                    row_mask=row_mask,
                    unit_to_images=unit_to_images,
                    n_images=n_images,
                )
                ci = bootstrap_ratio_ci(
                    pop["per_image_num"],
                    pop["per_image_den"],
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                )
                value = float(pop["population_ssi_bits_per_spike"])
                delta_stats = ratio_delta_stats(
                    pop["per_image_num"],
                    pop["per_image_den"],
                    baseline_pop["per_image_num"],
                    baseline_pop["per_image_den"],
                    n_resamples=n_bootstrap,
                    rng=rng,
                )
                pop_rows.append(
                    {
                        "relation": relation,
                        "relation_label": relation_label,
                        "sf_group": sf_group,
                        "component_metric": metric_col,
                        "component_metric_label": metric_label,
                        "context": str(bin_row.context),
                        "context_label": str(bin_row.context_label),
                        "component_bin": component_bin,
                        "component_bin_order": int(bin_row.component_bin_order),
                        "component_median_arcmin": float(bin_row.median_component_arcmin),
                        "n_movie_rows_global": int(bin_row.n_movie_rows_global),
                        "n_unique_traces_global": int(bin_row.n_unique_traces_global),
                        "n_units": int(len(unit_to_images)),
                        "n_images_contributing": pop["n_images_contributing"],
                        "n_movie_samples": pop["n_movie_samples"],
                        "population_ssi_bits_per_spike": value,
                        "population_ssi_delta_vs_stabilized": value - baseline_value,
                        "population_ci95_low_image_boot": ci[0],
                        "population_ci95_high_image_boot": ci[1],
                        **delta_stats,
                        "information_numerator_bits": pop["information_numerator_bits"],
                        "expected_spikes": pop["expected_spikes"],
                    }
                )

    summary = pd.DataFrame(pop_rows)
    if not summary.empty:
        summary["population_delta_q_holm_bootstrap_sign_relation_metric"] = np.nan
        for (_relation, metric_col), idx in summary.groupby(["relation", "component_metric"], sort=False).groups.items():
            summary.loc[idx, "population_delta_q_holm_bootstrap_sign_relation_metric"] = holm_adjust(
                summary.loc[idx, "population_delta_p_image_bootstrap_sign"],
            )
    return summary


def ordered_plot_rows(summary: pd.DataFrame, sf_group: str, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = summary[summary["sf_group"].astype(str).eq(str(sf_group))].copy()
    zero = sub[sub["context"].astype(str).eq("stabilized")].sort_values("path_bin_order").head(1)
    drift = sub[sub["context"].astype(str).eq("drift_only")].sort_values("path_median_arcmin")
    ms = sub[sub["context"].astype(str).eq("microsaccade")].sort_values("path_median_arcmin")
    for part in (zero, drift, ms):
        if value_col not in part.columns:
            raise ValueError(f"Missing {value_col}")
    return zero, drift, ms


def y_limits_for_plot(
    summary: pd.DataFrame,
    value_col: str,
    low_col: str,
    high_col: str,
    sym_error_col: str = "",
) -> tuple[float, float]:
    vals = []
    for col in [value_col, low_col, high_col]:
        if col in summary.columns:
            arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
            vals.append(arr[np.isfinite(arr)])
    if sym_error_col and sym_error_col in summary.columns and value_col in summary.columns:
        y = pd.to_numeric(summary[value_col], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(summary[sym_error_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(e)
        if np.any(ok):
            vals.append(y[ok] - e[ok])
            vals.append(y[ok] + e[ok])
    merged = np.concatenate([v for v in vals if v.size]) if vals else np.asarray([], dtype=float)
    if merged.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(merged))
    hi = float(np.nanmax(merged))
    span = max(hi - lo, 0.01)
    return lo - 0.08 * span, hi + 0.08 * span


def add_break_marks(ax_left: Any, ax_right: Any) -> None:
    kwargs = dict(marker=[(-1, -0.8), (1, 0.8)], markersize=12, linestyle="none", color="black", mec="black", mew=1.6, clip_on=False)
    ax_left.plot([1.0], [0.0], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0.0], [0.0], transform=ax_right.transAxes, **kwargs)


def plot_broken_schematic(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    title: str,
    subtitle: str,
    sf_groups: list[str],
    value_col: str,
    delta_col: str,
    low_col: str,
    high_col: str,
    y_label: str,
    delta_label: str,
    ci_label: str,
    dpi: int,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.2, 4.7), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.55, 3.4, 0.55, 3.4], wspace=0.06)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    panels = [
        (axes[0], axes[1], value_col, y_label, "Absolute SSI"),
        (axes[2], axes[3], delta_col, delta_label, "Movement modulation"),
    ]
    for ax_left, ax_right, metric_col, ylabel, panel_title in panels:
        err_low = low_col if metric_col == value_col else ""
        err_high = high_col if metric_col == value_col else ""
        ymin, ymax = y_limits_for_plot(summary, metric_col, err_low, err_high)
        for ax in [ax_left, ax_right]:
            ax.set_ylim(ymin, ymax)
            ax.grid(True, color="0.9", linewidth=0.8)
            ax.spines[["top"]].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.yaxis.set_visible(False)
        ax_left.set_xlim(-0.18, 0.18)
        positives = pd.to_numeric(summary.loc[summary["path_median_arcmin"] > 0, "path_median_arcmin"], errors="coerce")
        ax_right.set_xscale("log")
        ax_right.set_xlim(float(positives.min()) * 0.95, float(positives.max()) * 1.06)
        ticks = [90, 105, 120, 140, 170]
        ax_right.xaxis.set_major_locator(FixedLocator(ticks))
        ax_right.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax_right.xaxis.set_minor_formatter(NullFormatter())
        ax_left.set_xticks([0.0])
        ax_left.set_xticklabels(["0"])
        add_break_marks(ax_left, ax_right)
        if metric_col == delta_col:
            ax_left.axhline(0.0, color="0.35", lw=1.0, ls=":")
            ax_right.axhline(0.0, color="0.35", lw=1.0, ls=":")
        ax_left.set_ylabel(ylabel)
        ax_right.set_title(panel_title, fontsize=12, pad=8)

        for sf_group in sf_groups:
            color = SF_COLORS.get(sf_group, "0.2")
            zero, drift, ms = ordered_plot_rows(summary, sf_group, metric_col)
            if not zero.empty:
                z = zero.iloc[0]
                y = float(z[metric_col])
                if metric_col == value_col and err_low and err_high:
                    lo = float(z[err_low])
                    hi = float(z[err_high])
                    if math.isfinite(lo) and math.isfinite(hi):
                        ax_left.errorbar([0.0], [y], yerr=[[y - lo], [hi - y]], color=color, lw=1.8, capsize=0)
                ax_left.plot([0.0], [y], marker="o", markersize=6.5, color=color, markerfacecolor="white", markeredgewidth=1.8, lw=0)

            for rows, marker_face in [(drift, "white"), (ms, color)]:
                if rows.empty:
                    continue
                x = rows["path_median_arcmin"].to_numpy(dtype=float)
                y = rows[metric_col].to_numpy(dtype=float)
                if metric_col == value_col and err_low and err_high:
                    lo = rows[err_low].to_numpy(dtype=float)
                    hi = rows[err_high].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                    ax_right.errorbar(x, y, yerr=yerr, color=color, lw=1.9, capsize=0, alpha=0.90)
                else:
                    err_col = "sem_unit_ssi_delta_vs_stabilized"
                    if err_col in rows.columns:
                        e = rows[err_col].to_numpy(dtype=float)
                        ax_right.errorbar(x, y, yerr=e, color=color, lw=1.9, capsize=0, alpha=0.90)
                ax_right.plot(
                    x,
                    y,
                    color=color,
                    lw=2.0,
                    marker="o",
                    markersize=6.0,
                    markerfacecolor=marker_face,
                    markeredgewidth=1.8,
                    zorder=4,
                )
    handles = [
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white", markeredgewidth=1.6, lw=1.8, label="no detected microsaccade"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", markeredgewidth=1.6, lw=1.8, label=">=1 detected microsaccade"),
    ]
    handles.extend(
        Line2D([0], [0], color=SF_COLORS.get(sf_group, "0.2"), lw=2.3, label=SF_LABELS.get(sf_group, sf_group))
        for sf_group in sf_groups
    )
    if ci_label:
        handles.append(Line2D([0], [0], color="0.45", lw=1.8, label=ci_label))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.01))
    fig.supxlabel("Eye movement size (trajectory path length, arcmin; log scale after break)", y=0.055, fontsize=11)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13.5, y=0.98)
    fig.tight_layout(rect=(0, 0.13, 1, 0.88))
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=int(dpi), bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def plot_single_broken_schematic(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    title: str,
    subtitle: str,
    sf_groups: list[str],
    value_col: str,
    low_col: str,
    high_col: str,
    sym_error_col: str = "",
    y_label: str,
    error_label: str,
    dpi: int,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 5.0), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 5.1], wspace=0.045)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ymin, ymax = y_limits_for_plot(summary, value_col, low_col, high_col, sym_error_col)
    for ax in [ax_left, ax_right]:
        ax.set_ylim(ymin, ymax)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top"]].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.set_visible(False)
    ax_left.set_xlim(-0.18, 0.18)
    positives = pd.to_numeric(summary.loc[summary["path_median_arcmin"] > 0, "path_median_arcmin"], errors="coerce")
    ax_right.set_xscale("log")
    ax_right.set_xlim(float(positives.min()) * 0.96, float(positives.max()) * 1.04)
    ticks = [90, 105, 120, 140, 160, 175]
    ax_right.xaxis.set_major_locator(FixedLocator(ticks))
    ax_right.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax_right.xaxis.set_minor_formatter(NullFormatter())
    ax_left.set_xticks([0.0])
    ax_left.set_xticklabels(["0"])
    add_break_marks(ax_left, ax_right)
    if "delta" in value_col:
        ax_left.axhline(0.0, color="0.35", lw=1.0, ls=":")
        ax_right.axhline(0.0, color="0.35", lw=1.0, ls=":")
    ax_left.set_ylabel(y_label)

    for sf_group in sf_groups:
        color = SF_COLORS.get(sf_group, "0.2")
        zero, drift, ms = ordered_plot_rows(summary, sf_group, value_col)
        if not zero.empty:
            z = zero.iloc[0]
            y = float(z[value_col])
            if low_col and high_col and low_col in zero.columns and high_col in zero.columns:
                lo = float(z[low_col])
                hi = float(z[high_col])
                if math.isfinite(lo) and math.isfinite(hi):
                    ax_left.errorbar([0.0], [y], yerr=[[max(y - lo, 0.0)], [max(hi - y, 0.0)]], color=color, lw=1.8, capsize=0)
            elif sym_error_col and sym_error_col in zero.columns:
                e = float(z[sym_error_col])
                if math.isfinite(e):
                    ax_left.errorbar([0.0], [y], yerr=[[e], [e]], color=color, lw=1.8, capsize=0)
            ax_left.plot([0.0], [y], marker="o", markersize=6.8, color=color, markerfacecolor="white", markeredgewidth=1.9, lw=0)

        for rows, marker_face in [(drift, "white"), (ms, color)]:
            if rows.empty:
                continue
            x = rows["path_median_arcmin"].to_numpy(dtype=float)
            y = rows[value_col].to_numpy(dtype=float)
            if low_col and high_col and low_col in rows.columns and high_col in rows.columns:
                lo = rows[low_col].to_numpy(dtype=float)
                hi = rows[high_col].to_numpy(dtype=float)
                yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                ax_right.errorbar(x, y, yerr=yerr, color=color, lw=1.9, capsize=0, alpha=0.9)
            elif sym_error_col and sym_error_col in rows.columns:
                e = rows[sym_error_col].to_numpy(dtype=float)
                ax_right.errorbar(x, y, yerr=e, color=color, lw=1.9, capsize=0, alpha=0.9)
            ax_right.plot(
                x,
                y,
                color=color,
                lw=2.15,
                marker="o",
                markersize=6.2,
                markerfacecolor=marker_face,
                markeredgewidth=1.9,
                zorder=4,
            )

    handles = [
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white", markeredgewidth=1.7, lw=1.8, label="no detected microsaccade"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", markeredgewidth=1.7, lw=1.8, label=">=1 detected microsaccade"),
    ]
    handles.extend(
        Line2D([0], [0], color=SF_COLORS.get(sf_group, "0.2"), lw=2.5, label=SF_LABELS.get(sf_group, sf_group))
        for sf_group in sf_groups
    )
    if error_label:
        handles.append(Line2D([0], [0], color="0.45", lw=1.8, label=error_label))
    fig.legend(
        handles=handles,
        loc="lower center",
        frameon=False,
        fontsize=8.5,
        ncol=2,
        bbox_to_anchor=(0.56, 0.075),
    )
    fig.supxlabel("Eye movement size (trajectory path length, arcmin; log scale after break)", y=0.025, fontsize=11)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13.2, y=0.965)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.26, top=0.79)
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=int(dpi), bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def format_p_text(value: float) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    if float(value) < 0.001:
        return "<0.001"
    return f"{float(value):.3f}"


def format_p_label(value: float) -> str:
    text = format_p_text(value)
    if text.startswith("<"):
        return f"p{text}"
    return f"p={text}"


def smallest_path_comparison(
    summary: pd.DataFrame,
    metric_col: str,
    *,
    x_col: str = "path_median_arcmin",
) -> dict[str, float | str] | None:
    p_col = "population_delta_p_image_bootstrap_sign"
    if p_col not in summary.columns or metric_col not in summary.columns or x_col not in summary.columns:
        return None
    zero = summary[summary["context"].astype(str).eq("stabilized")]
    drift = summary[summary["context"].astype(str).eq("drift_only")].sort_values(x_col)
    if zero.empty or drift.empty:
        return None
    zero_row = zero.iloc[0]
    row = drift.iloc[0]
    return {
        "zero_y": float(zero_row[metric_col]),
        "first_x": float(row[x_col]),
        "first_y": float(row[metric_col]),
        "label": format_p_label(float(row[p_col])),
    }


def draw_broken_axis_comparison_bracket(
    fig: Any,
    ax_left: Any,
    ax_right: Any,
    comparison: dict[str, float | str],
    *,
    color: str = "0.18",
    fontsize: float = 6.8,
) -> None:
    y_min, y_max = ax_left.get_ylim()
    span = float(y_max - y_min)
    if not math.isfinite(span) or span <= 0:
        return
    zero_y = float(comparison["zero_y"])
    first_x = float(comparison["first_x"])
    first_y = float(comparison["first_y"])
    y = max(max(zero_y, first_y) + 0.07 * span, y_min + 0.62 * span)
    y = min(y, y_max - 0.08 * span)
    cap_y = y - 0.07 * span
    label_y = min(y + 0.025 * span, y_max - 0.025 * span)

    fig_inv = fig.transFigure.inverted()

    def to_fig(ax: Any, x: float, data_y: float) -> tuple[float, float]:
        return tuple(fig_inv.transform(ax.transData.transform((x, data_y))))

    left_top = to_fig(ax_left, 0.0, y)
    left_bottom = to_fig(ax_left, 0.0, cap_y)
    right_top = to_fig(ax_right, first_x, y)
    right_bottom = to_fig(ax_right, first_x, cap_y)
    label_pos = ((left_top[0] + right_top[0]) / 2.0, to_fig(ax_right, first_x, label_y)[1])

    for xs, ys in [
        ([left_bottom[0], left_top[0]], [left_bottom[1], left_top[1]]),
        ([left_top[0], right_top[0]], [left_top[1], right_top[1]]),
        ([right_top[0], right_bottom[0]], [right_top[1], right_bottom[1]]),
    ]:
        fig.add_artist(
            Line2D(xs, ys, transform=fig.transFigure, color=color, lw=0.9, solid_capstyle="round", clip_on=False, zorder=20)
        )
    fig.text(
        label_pos[0],
        label_pos[1],
        str(comparison["label"]),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        zorder=21,
    )


def plot_sf_rows_population_panel(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    title: str,
    subtitle: str,
    sf_groups: list[str],
    dpi: int,
    absolute_summary: pd.DataFrame | None = None,
    modulation_summary: pd.DataFrame | None = None,
    absolute_column_title: str = "Absolute SSI",
    modulation_column_title: str = "Movement modulation",
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    absolute_summary = summary if absolute_summary is None else absolute_summary
    modulation_summary = summary if modulation_summary is None else modulation_summary
    n_rows = len(sf_groups)
    fig = plt.figure(figsize=(11.8, 8.35), constrained_layout=False)
    gs = fig.add_gridspec(
        n_rows,
        5,
        width_ratios=[0.40, 3.0, 0.74, 0.40, 3.0],
        wspace=0.055,
        hspace=0.34,
    )
    axes: dict[tuple[int, int], tuple[Any, Any]] = {}
    bracket_specs: list[tuple[Any, Any, dict[str, float | str]]] = []

    for row_idx, sf_group in enumerate(sf_groups):
        color = SF_COLORS.get(sf_group, "0.2")
        for col_idx, (column_summary, metric_col, low_col, high_col, ylabel) in enumerate(
            [
                (
                    absolute_summary,
                    "population_ssi_bits_per_spike",
                    "population_ci95_low_image_boot",
                    "population_ci95_high_image_boot",
                    "Population SSI\n(bits/spike)",
                ),
                (
                    modulation_summary,
                    "population_ssi_delta_vs_stabilized",
                    "",
                    "",
                    "Population SSI - stabilized\n(bits/spike)",
                ),
            ]
        ):
            grid_col = 0 if col_idx == 0 else 3
            ax_left = fig.add_subplot(gs[row_idx, grid_col])
            ax_right = fig.add_subplot(gs[row_idx, grid_col + 1])
            axes[(row_idx, col_idx)] = (ax_left, ax_right)
            sf_summary = column_summary[column_summary["sf_group"].astype(str).eq(str(sf_group))].copy()

            ymin, ymax = y_limits_for_plot(sf_summary, metric_col, low_col, high_col)
            for ax in (ax_left, ax_right):
                ax.set_ylim(ymin, ymax)
                ax.grid(True, color="0.9", linewidth=0.8)
                ax.spines[["top"]].set_visible(False)
            ax_left.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            ax_right.yaxis.set_visible(False)
            ax_left.set_xlim(-0.18, 0.18)
            positives = pd.to_numeric(
                sf_summary.loc[sf_summary["path_median_arcmin"] > 0, "path_median_arcmin"],
                errors="coerce",
            )
            ax_right.set_xscale("log")
            ax_right.set_xlim(float(positives.min()) * 0.96, float(positives.max()) * 1.04)
            ticks = [90, 105, 120, 140, 160, 175]
            ax_right.xaxis.set_major_locator(FixedLocator(ticks))
            ax_right.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
            ax_right.xaxis.set_minor_formatter(NullFormatter())
            ax_left.set_xticks([0.0])
            ax_left.set_xticklabels(["0"])
            if row_idx < n_rows - 1:
                ax_left.tick_params(axis="x", labelbottom=False)
                ax_right.tick_params(axis="x", labelbottom=False)
            add_break_marks(ax_left, ax_right)

            if metric_col.endswith("delta_vs_stabilized"):
                ax_left.axhline(0.0, color="0.35", lw=1.0, ls=":")
                ax_right.axhline(0.0, color="0.35", lw=1.0, ls=":")

            if row_idx == n_rows // 2:
                ax_left.set_ylabel(ylabel, fontsize=9.5)
            else:
                ax_left.set_ylabel("")
            if col_idx == 0:
                ax_right.text(
                    0.02,
                    0.86,
                    SF_LABELS.get(sf_group, sf_group),
                    color=color,
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    transform=ax_right.transAxes,
                )

            zero, drift, ms = ordered_plot_rows(sf_summary, sf_group, metric_col)
            if not zero.empty:
                z = zero.iloc[0]
                y = float(z[metric_col])
                if low_col and high_col and low_col in zero.columns and high_col in zero.columns:
                    lo = float(z[low_col])
                    hi = float(z[high_col])
                    if math.isfinite(lo) and math.isfinite(hi):
                        ax_left.errorbar(
                            [0.0],
                            [y],
                            yerr=[[max(y - lo, 0.0)], [max(hi - y, 0.0)]],
                            color=color,
                            lw=1.5,
                            capsize=0,
                        )
                ax_left.plot(
                    [0.0],
                    [y],
                    marker="o",
                    markersize=5.6,
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=1.6,
                    lw=0,
                )

            for rows, marker_face in [(drift, "white"), (ms, color)]:
                if rows.empty:
                    continue
                x = rows["path_median_arcmin"].to_numpy(dtype=float)
                y = rows[metric_col].to_numpy(dtype=float)
                if low_col and high_col and low_col in rows.columns and high_col in rows.columns:
                    lo = rows[low_col].to_numpy(dtype=float)
                    hi = rows[high_col].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                    ax_right.errorbar(x, y, yerr=yerr, color=color, lw=1.5, capsize=0, alpha=0.9)
                ax_right.plot(
                    x,
                    y,
                    color=color,
                    lw=1.85,
                    marker="o",
                    markersize=5.2,
                    markerfacecolor=marker_face,
                    markeredgewidth=1.6,
                    zorder=4,
                )
            comparison = smallest_path_comparison(sf_summary, metric_col)
            if comparison is not None:
                bracket_specs.append((ax_left, ax_right, comparison))

    axes[(0, 0)][1].set_title(absolute_column_title, fontsize=12, pad=8)
    axes[(0, 1)][1].set_title(modulation_column_title, fontsize=12, pad=8)
    handles = [
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white", markeredgewidth=1.6, lw=1.8, label="no detected microsaccade"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", markeredgewidth=1.6, lw=1.8, label=">=1 detected microsaccade"),
        Line2D([0], [0], color="0.45", lw=1.6, label="95% image bootstrap CI on absolute SSI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.54, 0.055))
    fig.supxlabel("Eye movement size (trajectory path length, arcmin; log scale after break)", y=0.016, fontsize=11)
    title_text = f"{title}\n{subtitle}" if str(subtitle).strip() else str(title)
    fig.suptitle(title_text, fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.875)
    fig.canvas.draw()
    for ax_left, ax_right, comparison in bracket_specs:
        draw_broken_axis_comparison_bracket(fig, ax_left, ax_right, comparison)
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=int(dpi), bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def component_log_ticks(values: pd.Series | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return [1.0]
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    candidates = np.asarray([20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 220, 260, 320], dtype=float)
    ticks = candidates[(candidates >= lo * 0.95) & (candidates <= hi * 1.05)]
    if ticks.size >= 3:
        return [float(tick) for tick in ticks[:: max(1, int(math.ceil(ticks.size / 5)))]]
    generated = np.geomspace(max(lo, 1e-3), hi, num=min(5, max(2, int(arr.size))))
    return [float(round(tick)) for tick in generated]


def ordered_component_plot_rows(
    summary: pd.DataFrame,
    sf_group: str,
    component_metric: str,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = summary[
        summary["sf_group"].astype(str).eq(str(sf_group))
        & summary["component_metric"].astype(str).eq(str(component_metric))
    ].copy()
    zero = sub[sub["context"].astype(str).eq("stabilized")].sort_values("component_bin_order").head(1)
    drift = sub[sub["context"].astype(str).eq("drift_only")].sort_values("component_median_arcmin")
    ms = sub[sub["context"].astype(str).eq("microsaccade")].sort_values("component_median_arcmin")
    for part in (zero, drift, ms):
        if value_col not in part.columns:
            raise ValueError(f"Missing {value_col}")
    return zero, drift, ms


def plot_component_population_12_panel(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    title: str,
    sf_groups: list[str],
    dpi: int,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.8, 8.35), constrained_layout=False)
    gs = fig.add_gridspec(
        len(sf_groups),
        11,
        width_ratios=[0.32, 2.55, 0.46, 0.32, 2.55, 0.78, 0.32, 2.55, 0.46, 0.32, 2.55],
        wspace=0.045,
        hspace=0.34,
    )
    panel_specs = [
        ("across_path_arcmin", "population_ssi_bits_per_spike", "population_ci95_low_image_boot", "population_ci95_high_image_boot", "Across path\nAbsolute SSI"),
        ("across_path_arcmin", "population_ssi_delta_vs_stabilized", "", "", "Across path\nMovement modulation"),
        ("along_path_arcmin", "population_ssi_bits_per_spike", "population_ci95_low_image_boot", "population_ci95_high_image_boot", "Along path\nAbsolute SSI"),
        ("along_path_arcmin", "population_ssi_delta_vs_stabilized", "", "", "Along path\nMovement modulation"),
    ]
    grid_cols = [0, 3, 6, 9]
    axes: dict[tuple[int, int], tuple[Any, Any]] = {}
    bracket_specs: list[tuple[Any, Any, dict[str, float | str]]] = []

    for row_idx, sf_group in enumerate(sf_groups):
        color = SF_COLORS.get(sf_group, "0.2")
        for col_idx, (component_metric, metric_col, low_col, high_col, column_title) in enumerate(panel_specs):
            grid_col = grid_cols[col_idx]
            ax_left = fig.add_subplot(gs[row_idx, grid_col])
            ax_right = fig.add_subplot(gs[row_idx, grid_col + 1])
            axes[(row_idx, col_idx)] = (ax_left, ax_right)
            panel_summary = summary[
                summary["sf_group"].astype(str).eq(str(sf_group))
                & summary["component_metric"].astype(str).eq(str(component_metric))
            ].copy()
            ymin, ymax = y_limits_for_plot(panel_summary, metric_col, low_col, high_col)
            for ax in (ax_left, ax_right):
                ax.set_ylim(ymin, ymax)
                ax.grid(True, color="0.9", linewidth=0.8)
                ax.spines[["top"]].set_visible(False)
                ax.tick_params(labelsize=8.5)
            ax_left.spines["right"].set_visible(False)
            ax_right.spines["left"].set_visible(False)
            ax_right.yaxis.set_visible(False)
            ax_left.set_xlim(-0.18, 0.18)
            positives = pd.to_numeric(
                panel_summary.loc[panel_summary["component_median_arcmin"] > 0, "component_median_arcmin"],
                errors="coerce",
            )
            ax_right.set_xscale("log")
            if positives.dropna().empty:
                ax_right.set_xlim(1.0, 2.0)
            else:
                ax_right.set_xlim(float(positives.min()) * 0.96, float(positives.max()) * 1.04)
            ticks = component_log_ticks(positives)
            ax_right.xaxis.set_major_locator(FixedLocator(ticks))
            ax_right.xaxis.set_major_formatter(FixedFormatter([str(int(tick)) for tick in ticks]))
            ax_right.xaxis.set_minor_formatter(NullFormatter())
            ax_left.set_xticks([0.0])
            ax_left.set_xticklabels(["0"])
            if row_idx < len(sf_groups) - 1:
                ax_left.tick_params(axis="x", labelbottom=False)
                ax_right.tick_params(axis="x", labelbottom=False)
            add_break_marks(ax_left, ax_right)
            if metric_col.endswith("delta_vs_stabilized"):
                ax_left.axhline(0.0, color="0.35", lw=1.0, ls=":")
                ax_right.axhline(0.0, color="0.35", lw=1.0, ls=":")
            if row_idx == len(sf_groups) // 2 and col_idx == 0:
                ax_left.set_ylabel("Population SSI\n(bits/spike)", fontsize=9.3)
            elif row_idx == len(sf_groups) // 2 and col_idx == 1:
                ax_left.set_ylabel("Population SSI - stabilized\n(bits/spike)", fontsize=9.3)
            else:
                ax_left.set_ylabel("")
            if col_idx == 0:
                ax_right.text(
                    0.02,
                    0.86,
                    SF_LABELS.get(sf_group, sf_group),
                    color=color,
                    fontsize=10.5,
                    fontweight="bold",
                    va="center",
                    ha="left",
                    transform=ax_right.transAxes,
                )

            zero, drift, ms = ordered_component_plot_rows(panel_summary, sf_group, component_metric, metric_col)
            if not zero.empty:
                z = zero.iloc[0]
                y = float(z[metric_col])
                if low_col and high_col and low_col in zero.columns and high_col in zero.columns:
                    lo = float(z[low_col])
                    hi = float(z[high_col])
                    if math.isfinite(lo) and math.isfinite(hi):
                        ax_left.errorbar(
                            [0.0],
                            [y],
                            yerr=[[max(y - lo, 0.0)], [max(hi - y, 0.0)]],
                            color=color,
                            lw=1.35,
                            capsize=0,
                        )
                ax_left.plot(
                    [0.0],
                    [y],
                    marker="o",
                    markersize=5.2,
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    lw=0,
                    zorder=4,
                )
            for rows, marker_face in [(drift, "white"), (ms, color)]:
                if rows.empty:
                    continue
                x = rows["component_median_arcmin"].to_numpy(dtype=float)
                y = rows[metric_col].to_numpy(dtype=float)
                if low_col and high_col and low_col in rows.columns and high_col in rows.columns:
                    lo = rows[low_col].to_numpy(dtype=float)
                    hi = rows[high_col].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                    ax_right.errorbar(x, y, yerr=yerr, color=color, lw=1.35, capsize=0, alpha=0.88)
                ax_right.plot(
                    x,
                    y,
                    color=color,
                    lw=1.75,
                    marker="o",
                    markersize=4.7,
                    markerfacecolor=marker_face,
                    markeredgewidth=1.45,
                    zorder=4,
                )
            comparison = smallest_path_comparison(panel_summary, metric_col, x_col="component_median_arcmin")
            if comparison is not None:
                bracket_specs.append((ax_left, ax_right, comparison))

    for col_idx, (_component_metric, _metric_col, _low_col, _high_col, column_title) in enumerate(panel_specs):
        axes[(0, col_idx)][1].set_title(column_title, fontsize=11.2, pad=8)
    handles = [
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="white", markeredgewidth=1.5, lw=1.8, label="no detected microsaccade"),
        Line2D([0], [0], color="0.25", marker="o", markerfacecolor="0.25", markeredgewidth=1.5, lw=1.8, label=">=1 detected microsaccade"),
        Line2D([0], [0], color="0.45", lw=1.6, label="95% image bootstrap CI on absolute SSI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.54, 0.055))
    fig.supxlabel("Eye movement component size (component path length, arcmin; log scale after break)", y=0.016, fontsize=11)
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.875)
    fig.canvas.draw()
    for ax_left, ax_right, comparison in bracket_specs:
        draw_broken_axis_comparison_bracket(fig, ax_left, ax_right, comparison, fontsize=6.4)
    paths = {"png": out_dir / f"{stem}.png", "pdf": out_dir / f"{stem}.pdf"}
    fig.savefig(paths["png"], dpi=int(dpi), bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def filter_sf(summary: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return summary[summary["sf_group"].astype(str).isin(groups)].copy()


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else matrix_dir / "phase1_phase2_conditioning_v1" / "schematic_pathlength_summary_v1" / "unit_first_and_population_v1"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sf_groups = parse_csv_list(str(args.sf_groups))
    data = load_dataset(matrix_dir)
    trace, trace_bins = add_equal_count_trace_bins(
        data["trace"],
        n_drift_bins=int(args.n_drift_bins),
        n_microsaccade_bins=int(args.n_microsaccade_bins),
    )
    trace_bins.to_csv(out_dir / "trace_path_bin_definitions_8_drift_5_ms.csv", index=False)
    component_metrics = compute_component_movie_metrics(
        data["movie"],
        data["trace"],
        data["trace_xy"],
        image_axis_col=str(args.image_axis_col),
    )
    component_bins_by_metric: dict[str, pd.DataFrame] = {}
    component_bin_tables: list[pd.DataFrame] = []
    for metric_col, metric_label in COMPONENT_PATH_SPECS:
        component_metrics, component_bins = add_equal_count_component_bins(
            component_metrics,
            metric_col=metric_col,
            metric_label=metric_label,
            n_drift_bins=int(args.n_drift_bins),
            n_microsaccade_bins=int(args.n_microsaccade_bins),
        )
        component_bins_by_metric[metric_col] = component_bins
        component_bin_tables.append(component_bins)
    component_bins_table = pd.concat(component_bin_tables, ignore_index=True) if component_bin_tables else pd.DataFrame()
    component_bins_table.to_csv(out_dir / "component_path_bin_definitions_8_drift_5_ms.csv", index=False)
    row_grid = build_movie_row_grid(data["movie"])
    baseline_lookup = baseline_rows_by_image(data["image"], data["baseline_table"])
    rng = np.random.default_rng(int(args.bootstrap_seed))

    relation_specs = [
        ("all_images_no_osi", "all image windows; no OSI gate"),
        ("strong_contours_no_osi", "strong contour image windows; no OSI gate"),
        ("all_orientation_tuned", "all image windows; OSI gated"),
        ("contour_matched", "strong contour image windows; orientation aligned"),
        ("contour_intermediate", "strong contour image windows; orientation intermediate"),
        ("contour_orthogonal", "strong contour image windows; orientation orthogonal"),
    ]
    all_selection = []
    all_unit_curves = []
    all_unit_summary = []
    all_population_summary = []
    all_component_population_summary = []
    population_summary_by_relation: dict[str, pd.DataFrame] = {}
    component_population_summary_by_relation: dict[str, pd.DataFrame] = {}
    figures: dict[str, dict[str, str]] = {}
    for relation, relation_label in relation_specs:
        selections = unit_image_selection(
            data["unit"],
            data["image"],
            relation=relation,
            sf_groups=sf_groups,
            min_osi=float(args.min_osi),
            match_max_deg=float(args.match_max_deg),
            orthogonal_min_deg=float(args.orthogonal_min_deg),
            image_axis_col=str(args.image_axis_col),
        )
        selection, unit_curves, unit_summary, population_summary = build_curves_for_relation(
            relation=relation,
            relation_label=relation_label,
            selections=selections,
            sf_groups=sf_groups,
            trace=trace,
            trace_bins=trace_bins,
            row_grid=row_grid,
            baseline_lookup=baseline_lookup,
            ssi=data["ssi"],
            expected=data["expected"],
            stabilized_ssi=data["stabilized_ssi"],
            stabilized_expected=data["stabilized_expected"],
            unit=data["unit"],
            rng=rng,
            n_bootstrap=int(args.n_bootstrap),
        )
        all_selection.append(selection)
        all_unit_curves.append(unit_curves)
        all_unit_summary.append(unit_summary)
        all_population_summary.append(population_summary)
        population_summary_by_relation[relation] = population_summary
        if relation in {
            "all_images_no_osi",
            "strong_contours_no_osi",
            "contour_matched",
            "contour_intermediate",
            "contour_orthogonal",
        }:
            component_population_summary = build_component_population_summary_for_relation(
                relation=relation,
                relation_label=relation_label,
                selections=selections,
                sf_groups=sf_groups,
                component_metrics=component_metrics,
                component_bins_by_metric=component_bins_by_metric,
                ssi=data["ssi"],
                expected=data["expected"],
                stabilized_ssi=data["stabilized_ssi"],
                stabilized_expected=data["stabilized_expected"],
                row_grid=row_grid,
                baseline_lookup=baseline_lookup,
                rng=rng,
                n_bootstrap=int(args.n_bootstrap),
            )
            all_component_population_summary.append(component_population_summary)
            component_population_summary_by_relation[relation] = component_population_summary

        subtitle = f"{relation_label}; path bins split as {int(args.n_drift_bins)} drift-only / {int(args.n_microsaccade_bins)} microsaccade"
        for groups, suffix in [
            (["low_sf"], "low_sf"),
            (["middle_sf"], "middle_sf"),
            (["high_sf"], "high_sf"),
            (["low_sf", "high_sf"], "low_high_sf"),
            (["low_sf", "middle_sf", "high_sf"], "low_middle_high_sf"),
        ]:
            groups = [group for group in groups if group in sf_groups]
            if not groups:
                continue
            sf_text = ", ".join(SF_LABELS.get(group, group) for group in groups)
            unit_plot_summary = filter_sf(unit_summary, groups)
            pop_plot_summary = filter_sf(population_summary, groups)
            figures[f"{relation}_{suffix}_unit_first"] = plot_broken_schematic(
                unit_plot_summary,
                fig_dir,
                stem=f"{relation}_{suffix}_unit_first_ms_split_broken_log",
                title=f"{sf_text}: unit-first SSI",
                subtitle=subtitle,
                sf_groups=groups,
                value_col="mean_unit_ssi_bits_per_spike",
                delta_col="mean_unit_ssi_delta_vs_stabilized",
                low_col="",
                high_col="",
                y_label="Mean unit SSI (bits/spike)",
                delta_label="Mean unit SSI minus stabilized baseline (bits/spike)",
                ci_label="+/- SEM across units",
                dpi=int(args.dpi),
            )
            figures[f"{relation}_{suffix}_population"] = plot_broken_schematic(
                pop_plot_summary,
                fig_dir,
                stem=f"{relation}_{suffix}_spike_weighted_population_ms_split_broken_log",
                title=f"{sf_text}: spike-weighted population SSI",
                subtitle=subtitle,
                sf_groups=groups,
                value_col="population_ssi_bits_per_spike",
                delta_col="population_ssi_delta_vs_stabilized",
                low_col="population_ci95_low_image_boot",
                high_col="population_ci95_high_image_boot",
                y_label="Spike-weighted population SSI (bits/spike)",
                delta_label="Population SSI minus stabilized baseline (bits/spike)",
                ci_label="95% image bootstrap CI",
                dpi=int(args.dpi),
            )
            figures[f"{relation}_{suffix}_unit_first_absolute_single"] = plot_single_broken_schematic(
                unit_plot_summary,
                fig_dir,
                stem=f"{relation}_{suffix}_unit_first_absolute_ms_split_broken_log",
                title=f"{sf_text}: unit-first SSI",
                subtitle=subtitle,
                sf_groups=groups,
                value_col="mean_unit_ssi_bits_per_spike",
                low_col="",
                high_col="",
                sym_error_col="sem_unit_ssi_bits_per_spike",
                y_label="Mean unit SSI (bits/spike)",
                error_label="+/- SEM across units",
                dpi=int(args.dpi),
            )
            figures[f"{relation}_{suffix}_unit_first_delta_single"] = plot_single_broken_schematic(
                unit_plot_summary,
                fig_dir,
                stem=f"{relation}_{suffix}_unit_first_delta_ms_split_broken_log",
                title=f"{sf_text}: unit-first SSI modulation",
                subtitle=subtitle,
                sf_groups=groups,
                value_col="mean_unit_ssi_delta_vs_stabilized",
                low_col="",
                high_col="",
                sym_error_col="sem_unit_ssi_delta_vs_stabilized",
                y_label="Mean unit SSI minus stabilized baseline (bits/spike)",
                error_label="+/- SEM across units",
                dpi=int(args.dpi),
            )
            figures[f"{relation}_{suffix}_population_absolute_single"] = plot_single_broken_schematic(
                pop_plot_summary,
                fig_dir,
                stem=f"{relation}_{suffix}_spike_weighted_population_absolute_ms_split_broken_log",
                title=f"{sf_text}: spike-weighted population SSI",
                subtitle=subtitle,
                sf_groups=groups,
                value_col="population_ssi_bits_per_spike",
                low_col="population_ci95_low_image_boot",
                high_col="population_ci95_high_image_boot",
                y_label="Spike-weighted population SSI (bits/spike)",
                error_label="95% image bootstrap CI",
                dpi=int(args.dpi),
            )
            figures[f"{relation}_{suffix}_population_delta_single"] = plot_single_broken_schematic(
                pop_plot_summary,
                fig_dir,
                stem=f"{relation}_{suffix}_spike_weighted_population_delta_ms_split_broken_log",
                title=f"{sf_text}: population SSI modulation",
                subtitle=subtitle,
                sf_groups=groups,
                value_col="population_ssi_delta_vs_stabilized",
                low_col="",
                high_col="",
                y_label="Population SSI minus stabilized baseline (bits/spike)",
                error_label="",
                dpi=int(args.dpi),
            )
        panel_groups = [group for group in ["low_sf", "middle_sf", "high_sf"] if group in sf_groups]
        if panel_groups:
            same_selection_titles = {
                "all_images_no_osi": "Spike-weighted population SSI - all image windows, all units",
                "strong_contours_no_osi": "Spike-weighted population SSI - strong contour images, all units",
                "contour_matched": "Spike-weighted population SSI - strong contour images, orientation-aligned units",
                "contour_intermediate": "Spike-weighted population SSI - strong contour images, orientation-intermediate units",
                "contour_orthogonal": "Spike-weighted population SSI - strong contour images, orientation-orthogonal units",
            }
            panel_title = same_selection_titles.get(relation, "Low, Middle, High SF: spike-weighted population SSI")
            panel_subtitle = "" if relation in same_selection_titles else subtitle
            figures[f"{relation}_low_middle_high_sf_population_absolute_delta_panel"] = plot_sf_rows_population_panel(
                filter_sf(population_summary, panel_groups),
                fig_dir,
                stem=f"{relation}_low_middle_high_sf_spike_weighted_population_absolute_delta_six_panel",
                title=panel_title,
                subtitle=panel_subtitle,
                sf_groups=panel_groups,
                dpi=int(args.dpi),
            )
            if relation in component_population_summary_by_relation:
                component_titles = {
                    "all_images_no_osi": "Spike-weighted population SSI - all image windows, all units",
                    "strong_contours_no_osi": "Spike-weighted population SSI - strong contour images, all units",
                    "contour_matched": "Spike-weighted population SSI - strong contour images, orientation-aligned units",
                    "contour_intermediate": "Spike-weighted population SSI - strong contour images, orientation-intermediate units",
                    "contour_orthogonal": "Spike-weighted population SSI - strong contour images, orientation-orthogonal units",
                }
                figures[f"{relation}_low_middle_high_sf_component_path_population_12_panel"] = plot_component_population_12_panel(
                    filter_sf(component_population_summary_by_relation[relation], panel_groups),
                    fig_dir,
                    stem=f"{relation}_low_middle_high_sf_spike_weighted_population_across_along_component_path_12_panel",
                    title=component_titles.get(relation, "Spike-weighted population SSI"),
                    sf_groups=panel_groups,
                    dpi=int(args.dpi),
                )

    panel_groups = [group for group in ["low_sf", "middle_sf", "high_sf"] if group in sf_groups]
    if panel_groups and {"all_images_no_osi", "contour_matched"}.issubset(population_summary_by_relation):
        figures["all_images_absolute_contour_matched_delta_low_middle_high_sf_panel"] = plot_sf_rows_population_panel(
            filter_sf(population_summary_by_relation["contour_matched"], panel_groups),
            fig_dir,
            stem="all_images_absolute_contour_matched_delta_low_middle_high_sf_six_panel",
            title="Low, Middle, High SF: spike-weighted population SSI",
            subtitle=(
                f"left: all image windows, no OSI gate; right: contour-matched image windows; "
                f"path bins split as {int(args.n_drift_bins)} drift-only / {int(args.n_microsaccade_bins)} microsaccade"
            ),
            sf_groups=panel_groups,
            dpi=int(args.dpi),
            absolute_summary=filter_sf(population_summary_by_relation["all_images_no_osi"], panel_groups),
            modulation_summary=filter_sf(population_summary_by_relation["contour_matched"], panel_groups),
            absolute_column_title="Absolute SSI\nall images, no OSI gate",
            modulation_column_title="Movement modulation\ncontour-matched",
        )

    selection_table = pd.concat(all_selection, ignore_index=True)
    unit_curves_table = pd.concat(all_unit_curves, ignore_index=True)
    unit_summary_table = pd.concat(all_unit_summary, ignore_index=True)
    population_summary_table = pd.concat(all_population_summary, ignore_index=True)
    component_population_summary_table = (
        pd.concat(all_component_population_summary, ignore_index=True)
        if all_component_population_summary
        else pd.DataFrame()
    )
    selection_csv = out_dir / "unit_image_selection.csv"
    unit_curves_csv = out_dir / "unit_first_curves.csv"
    unit_summary_csv = out_dir / "unit_first_summary.csv"
    population_summary_csv = out_dir / "spike_weighted_population_summary.csv"
    population_tests_csv = out_dir / "spike_weighted_population_vs_stabilized_tests.csv"
    component_population_summary_csv = out_dir / "spike_weighted_population_component_summary.csv"
    component_population_tests_csv = out_dir / "spike_weighted_population_component_vs_stabilized_tests.csv"
    selection_table.to_csv(selection_csv, index=False)
    unit_curves_table.to_csv(unit_curves_csv, index=False)
    unit_summary_table.to_csv(unit_summary_csv, index=False)
    population_summary_table.to_csv(population_summary_csv, index=False)
    component_population_summary_table.to_csv(component_population_summary_csv, index=False)
    test_cols = [
        "relation",
        "relation_label",
        "sf_group",
        "context",
        "context_label",
        "path_bin",
        "path_bin_order",
        "path_median_arcmin",
        "n_traces",
        "n_units",
        "population_ssi_bits_per_spike",
        "population_ssi_delta_vs_stabilized",
        "population_delta_ci95_low_image_boot",
        "population_delta_ci95_high_image_boot",
        "population_delta_p_image_bootstrap_sign",
        "population_delta_q_holm_bootstrap_sign_relation",
    ]
    population_summary_table[
        population_summary_table["context"].astype(str).ne("stabilized")
    ][[col for col in test_cols if col in population_summary_table.columns]].to_csv(population_tests_csv, index=False)
    component_test_cols = [
        "relation",
        "relation_label",
        "sf_group",
        "component_metric",
        "component_metric_label",
        "context",
        "context_label",
        "component_bin",
        "component_bin_order",
        "component_median_arcmin",
        "n_movie_rows_global",
        "n_unique_traces_global",
        "n_units",
        "population_ssi_bits_per_spike",
        "population_ssi_delta_vs_stabilized",
        "population_delta_ci95_low_image_boot",
        "population_delta_ci95_high_image_boot",
        "population_delta_p_image_bootstrap_sign",
        "population_delta_q_holm_bootstrap_sign_relation_metric",
    ]
    if not component_population_summary_table.empty:
        component_population_summary_table[
            component_population_summary_table["context"].astype(str).ne("stabilized")
        ][[col for col in component_test_cols if col in component_population_summary_table.columns]].to_csv(
            component_population_tests_csv,
            index=False,
        )
    else:
        pd.DataFrame(columns=component_test_cols).to_csv(component_population_tests_csv, index=False)

    write_json(
        out_dir / "summary.json",
        {
            "analysis": "backimage_real_trace_unit_first_and_population_schematics",
            "matrix_dir": matrix_dir,
            "out_dir": out_dir,
            "sf_groups": sf_groups,
            "n_drift_bins": int(args.n_drift_bins),
            "n_microsaccade_bins": int(args.n_microsaccade_bins),
            "match_max_deg": float(args.match_max_deg),
            "orthogonal_min_deg": float(args.orthogonal_min_deg),
            "min_osi": float(args.min_osi),
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "contracts": {
                "matrix_rows": "movie_feature_table.csv row order indexes ssi_matrix.npy and expected_spikes_matrix.npy; matrix_row_index is not used because merged shards can contain shard-local values.",
                "unit_first": "Each unit is averaged over selected image x trace movies within a bin; group means and SEMs are across units.",
                "population": "sum(unit_ssi_bits_per_spike * expected_spikes) / sum(expected_spikes) over selected units, images, and traces.",
                "population_ci": "Image-bootstrap CI over per-image accumulated numerator/denominator contributions, conditioned on the selected trace bin.",
                "population_delta_stats": "Each nonzero path bin is compared with the stabilized zero-motion baseline using paired image-bootstrap ratio deltas; p is the uncorrected two-sided bootstrap sign probability and q is Holm-corrected within the relation for audit only.",
                "figure_stats": "Six-panel annotations report only the uncorrected p for the smallest drift-only path bin versus the stabilized zero-motion baseline.",
                "component_bins": "Across/along component bins are equal-count image x trace movie bins after projecting trace steps onto each image's contour axis.",
                "strong_contour_relations": "strong_contours_no_osi, contour_matched, contour_intermediate, and contour_orthogonal require image_contour_strong == True. contour_matched, contour_intermediate, and contour_orthogonal also require OSI >= min_osi and the configured unit-contour orientation relationship.",
            },
            "outputs": {
                "trace_bins_csv": out_dir / "trace_path_bin_definitions_8_drift_5_ms.csv",
                "component_bins_csv": out_dir / "component_path_bin_definitions_8_drift_5_ms.csv",
                "selection_csv": selection_csv,
                "unit_curves_csv": unit_curves_csv,
                "unit_summary_csv": unit_summary_csv,
                "population_summary_csv": population_summary_csv,
                "population_tests_csv": population_tests_csv,
                "component_population_summary_csv": component_population_summary_csv,
                "component_population_tests_csv": component_population_tests_csv,
                "figures": figures,
            },
        },
    )
    print(f"Wrote {out_dir}")
    print(population_summary_table.head().to_string(index=False))


if __name__ == "__main__":
    main()
