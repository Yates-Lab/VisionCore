#!/usr/bin/env python3
"""Build the Fig. 4 native BackImage trace-bank metadata cache.

This recovers the source path for
``fig4_trace_bank_metadata_filtered.csv`` without importing the divergent
experimental tree. It samples source windows from the reviewed BackImage/FEM
table, center-crops native 40-sample eye traces, writes the full diagnostic
trace-bank metadata schema, and writes the historical path-length-filtered
subset used by the Fig. 4 geometry-story refresh.
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

import numpy as np
import pandas as pd

UPSTREAM_DIR = Path(__file__).resolve().parent
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from real_trace_matrix.core import (
    build_native_snippet_trace_bank,
    json_ready,
    load_backimage_eyepos_by_session,
    load_source_rows,
    trace_bank_metric_payload,
    trace_bank_metric_summary_rows,
    trace_metric_value,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[3]
RUN_STEM = "backimage_trace_bank_diffusion_large_fixation_sample_n5000_n40_v1"
DEFAULT_SOURCE_CSV = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_OUT_DIR = ROOT / "outputs/active_sensing_movie_information" / RUN_STEM

LEGACY_TRACE_BANK_METADATA_COLUMNS = [
    "source_row",
    "session",
    "trial_idx",
    "global_start",
    "global_stop",
    "source_window_global_start",
    "source_window_global_stop",
    "snippet_global_start",
    "snippet_global_stop",
    "snippet_n_samples",
    "snippet_duration_s",
    "source_window_n_samples",
    "source_window_duration_s",
    "mean_x_deg",
    "mean_y_deg",
    "observed_rms_deg",
    "source_trace_observed_rms_deg",
    "path_length_deg",
    "duration_s",
    "lag1_autocorr",
    "covariance_shape",
    "trace_cov_anisotropy",
    "source_trace_cov_anisotropy",
    "source_anisotropy",
    "trace_bank_snippet_policy",
    "source_n_samples",
    "source_duration_s",
    "source_mean_x_deg",
    "source_mean_y_deg",
    "source_abs_mean_radius_deg",
    "source_rms_radius_deg",
    "source_median_radius_deg",
    "source_p05_radius_deg",
    "source_p95_radius_deg",
    "source_max_radius_deg",
    "source_cov_xx_deg2",
    "source_cov_xy_deg2",
    "source_cov_yy_deg2",
    "source_cloud_area_deg2",
    "source_drift_orientation_deg",
    "source_step_mean_deg",
    "source_step_median_deg",
    "source_step_p95_deg",
    "source_speed_mean_deg_s",
    "source_speed_median_deg_s",
    "source_speed_p95_deg_s",
    "source_path_length_deg",
    "source_path_length_deg_s",
    "source_direction_persistence",
    "source_curvature_rad",
    "source_return_to_center_strength",
    "source_position_autocorr_lag1",
    "source_position_autocorr_lag4",
    "source_velocity_autocorr_lag1",
    "source_velocity_autocorr_lag4",
    "source_fraction_within_0p05deg",
    "source_fraction_within_0p10deg",
    "source_fraction_within_0p25deg",
    "source_msd_lag1_deg2",
    "source_msd_lag2_deg2",
    "source_msd_lag4_deg2",
    "source_msd_lag8_deg2",
    "source_msd_lag16_deg2",
    "source_diffusion_constant_deg2_s",
    "source_position_psd_slope_1_30hz",
    "source_position_high_freq_power_fraction_15_60hz",
    "source_diffusion_constant_arcmin2_s",
    "source_rms_radius_arcmin",
    "source_path_length_arcmin",
    "rendered_n_samples",
    "rendered_duration_s",
    "rendered_mean_x_deg",
    "rendered_mean_y_deg",
    "rendered_abs_mean_radius_deg",
    "rendered_rms_radius_deg",
    "rendered_median_radius_deg",
    "rendered_p05_radius_deg",
    "rendered_p95_radius_deg",
    "rendered_max_radius_deg",
    "rendered_cov_xx_deg2",
    "rendered_cov_xy_deg2",
    "rendered_cov_yy_deg2",
    "rendered_cloud_area_deg2",
    "rendered_anisotropy",
    "rendered_drift_orientation_deg",
    "rendered_step_mean_deg",
    "rendered_step_median_deg",
    "rendered_step_p95_deg",
    "rendered_speed_mean_deg_s",
    "rendered_speed_median_deg_s",
    "rendered_speed_p95_deg_s",
    "rendered_path_length_deg",
    "rendered_path_length_deg_s",
    "rendered_direction_persistence",
    "rendered_curvature_rad",
    "rendered_return_to_center_strength",
    "rendered_position_autocorr_lag1",
    "rendered_position_autocorr_lag4",
    "rendered_velocity_autocorr_lag1",
    "rendered_velocity_autocorr_lag4",
    "rendered_fraction_within_0p05deg",
    "rendered_fraction_within_0p10deg",
    "rendered_fraction_within_0p25deg",
    "rendered_msd_lag1_deg2",
    "rendered_msd_lag2_deg2",
    "rendered_msd_lag4_deg2",
    "rendered_msd_lag8_deg2",
    "rendered_msd_lag16_deg2",
    "rendered_diffusion_constant_deg2_s",
    "rendered_position_psd_slope_1_30hz",
    "rendered_position_high_freq_power_fraction_15_60hz",
    "rendered_diffusion_constant_arcmin2_s",
    "rendered_rms_radius_arcmin",
    "rendered_path_length_arcmin",
    "source_microsaccade_threshold_dps",
    "source_n_microsaccade_events",
    "source_fraction_microsaccade_samples",
    "source_peak_microsaccade_speed_dps",
    "rendered_microsaccade_threshold_dps",
    "rendered_n_microsaccade_events",
    "rendered_fraction_microsaccade_samples",
    "rendered_peak_microsaccade_speed_dps",
    "microsaccade_threshold_dps",
    "n_microsaccade_events",
    "fraction_microsaccade_samples",
    "peak_microsaccade_speed_dps",
    "observed_rms_arcmin",
    "path_length_arcmin",
    "source_rendered_diffusion_delta_deg2_s",
    "source_rendered_diffusion_abs_delta_deg2_s",
    "source_cov_major_var_deg2",
    "source_cov_minor_var_deg2",
    "source_cov_major_sd_arcmin",
    "source_cov_minor_sd_arcmin",
    "source_cov_axis_ratio",
    "source_cov_orientation_deg",
    "source_bcea68_deg2",
    "source_bcea68_arcmin2",
    "source_cov_anisotropy",
    "rendered_cov_major_var_deg2",
    "rendered_cov_minor_var_deg2",
    "rendered_cov_major_sd_arcmin",
    "rendered_cov_minor_sd_arcmin",
    "rendered_cov_axis_ratio",
    "rendered_cov_orientation_deg",
    "rendered_bcea68_deg2",
    "rendered_bcea68_arcmin2",
    "rendered_cov_anisotropy",
    "source_speed_mean_arcmin_s",
    "source_speed_median_arcmin_s",
    "source_speed_p95_arcmin_s",
    "source_path_speed_arcmin_s",
    "rendered_speed_mean_arcmin_s",
    "rendered_speed_median_arcmin_s",
    "rendered_speed_p95_arcmin_s",
    "rendered_path_speed_arcmin_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-source-windows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--path-length-threshold-arcmin", type=float, default=350.0)
    parser.add_argument("--summary-bin-metric", type=str, default="rendered_diffusion_constant_deg2_s")
    parser.add_argument("--summary-bins", type=int, default=6)
    parser.add_argument("--microsaccade-speed-threshold-dps", type=float, default=None)
    parser.add_argument("--microsaccade-threshold-z", type=float, default=6.0)
    parser.add_argument("--microsaccade-pad-frames", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sample_source_rows(rows: pd.DataFrame, *, n_source_windows: int, seed: int, n_timepoints: int) -> pd.DataFrame:
    work = rows.copy()
    if "source_row" not in work.columns:
        work["source_row"] = np.arange(work.shape[0], dtype=int)
    if "n_samples" in work.columns:
        keep = pd.to_numeric(work["n_samples"], errors="coerce") >= int(n_timepoints)
        work = work.loc[keep].copy()
    if int(n_source_windows) > 0 and work.shape[0] > int(n_source_windows):
        work = work.sample(n=int(n_source_windows), random_state=int(seed), replace=False)
    return work.sort_values("source_row", kind="mergesort").reset_index(drop=True)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([trace_metric_value(row, key) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def metric_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = _finite_values(rows, key)
    out: dict[str, Any] = {"finite_n": int(values.size)}
    if values.size == 0:
        return out
    out["zero_or_negative_fraction"] = float(np.mean(values <= 0.0))
    out["quantiles"] = {
        str(q): float(np.nanquantile(values, q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    }
    out["mean"] = float(np.nanmean(values))
    return out


def source_rendered_diffusion_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = _finite_values(rows, "source_rendered_diffusion_abs_delta_deg2_s")
    if values.size == 0:
        return {"max": float("nan"), "mean": float("nan")}
    return {"max": float(np.nanmax(values) * 3600.0), "mean": float(np.nanmean(values) * 3600.0)}


def quantile_bin_summary(rows: list[dict[str, Any]], *, metric: str, n_bins: int) -> list[dict[str, Any]]:
    values = np.asarray([trace_metric_value(row, metric) for row in rows], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return []
    order = finite[np.argsort(values[finite], kind="mergesort")]
    chunks = [chunk for chunk in np.array_split(order, max(1, int(n_bins))) if chunk.size]
    out: list[dict[str, Any]] = []
    for bin_index, chunk in enumerate(chunks):
        chunk_values = values[chunk]
        out.append(
            {
                "bin_index": int(bin_index),
                "bin_label": f"q{bin_index + 1:02d}",
                "metric_low": float(np.nanmin(chunk_values)),
                "metric_high": float(np.nanmax(chunk_values)),
                "metric_median": float(np.nanmedian(chunk_values)),
                "n_trace_bank_members": int(chunk.size),
            }
        )
    return out


def path_length_context_windows(rows: list[dict[str, Any]], *, n_bins: int = 6) -> list[dict[str, Any]]:
    values = np.asarray([trace_metric_value(row, "rendered_path_length_arcmin") for row in rows], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return []
    order = finite[np.argsort(values[finite], kind="mergesort")]
    chunks = [chunk for chunk in np.array_split(order, max(1, int(n_bins))) if chunk.size]
    out: list[dict[str, Any]] = []
    for code, chunk in enumerate(chunks):
        path_values = values[chunk]
        out.append(
            {
                "path_length_window_index": int(code),
                "path_length_window_label": f"q{code + 1:02d}",
                "path_length_low_arcmin": float(np.nanmin(path_values)),
                "path_length_q25_arcmin": float(np.nanpercentile(path_values, 25.0)),
                "path_length_median_arcmin": float(np.nanmedian(path_values)),
                "path_length_q75_arcmin": float(np.nanpercentile(path_values, 75.0)),
                "path_length_high_arcmin": float(np.nanmax(path_values)),
                "n_trace_bank_members": int(chunk.size),
            }
        )
    return out


def microsaccade_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = np.asarray([int(row.get("n_microsaccade_events", 0)) for row in rows], dtype=int)
    values = np.asarray([trace_metric_value(row, "rendered_diffusion_constant_arcmin2_s") for row in rows], dtype=np.float64)
    out: dict[str, Any] = {
        "event_counts": {str(value): int(np.sum(events == value)) for value in sorted(set(events.tolist()))},
    }
    for label, mask in (("no_detected_microsaccade", events == 0), ("with_detected_microsaccade", events > 0)):
        group = values[mask]
        group = group[np.isfinite(group)]
        if group.size:
            out[f"{label}_diffusion_arcmin2_s"] = {
                "n": int(group.size),
                "median": float(np.nanmedian(group)),
                "q95": float(np.nanquantile(group, 0.95)),
            }
        else:
            out[f"{label}_diffusion_arcmin2_s"] = {"n": 0, "median": None, "q95": None}
    return out


def legacy_trace_bank_metadata_rows(bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bank:
        item = dict(item)
        item["observed_rms_arcmin"] = float(item["observed_rms_deg"]) * 60.0
        item["path_length_arcmin"] = float(item["path_length_deg"]) * 60.0
        payload = {key: value for key, value in item.items() if key != "trace"}
        payload.update(trace_bank_metric_payload(item))
        rows.append({column: payload.get(column, "") for column in LEGACY_TRACE_BANK_METADATA_COLUMNS})
    return rows


def filter_by_path_length(rows: list[dict[str, Any]], *, threshold_arcmin: float) -> list[dict[str, Any]]:
    threshold = float(threshold_arcmin)
    if threshold <= 0.0:
        return list(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        value = trace_metric_value(row, "path_length_arcmin")
        if math.isfinite(value) and value <= threshold:
            out.append(row)
    return out


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    metadata_path = out_dir / "trace_bank_metadata.csv"
    if metadata_path.exists() and not bool(args.force):
        raise FileExistsError(f"Refusing to overwrite {metadata_path}; pass --force.")

    rows = load_source_rows(Path(args.source_csv))
    sampled = sample_source_rows(
        rows,
        n_source_windows=int(args.n_source_windows),
        seed=int(args.seed),
        n_timepoints=int(args.n_timepoints),
    )
    sessions = sampled["session"].astype(str).dropna().unique().tolist()
    eyepos_by_session = load_backimage_eyepos_by_session(sessions)
    bank, builder_meta = build_native_snippet_trace_bank(
        sampled,
        eyepos_by_session,
        int(args.n_timepoints),
        dt=float(args.bin_seconds),
        microsaccade_speed_threshold_dps=(
            float(args.microsaccade_speed_threshold_dps)
            if args.microsaccade_speed_threshold_dps is not None
            else None
        ),
        microsaccade_threshold_z=float(args.microsaccade_threshold_z),
        microsaccade_pad_frames=int(args.microsaccade_pad_frames),
    )
    trace_rows = legacy_trace_bank_metadata_rows(bank)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metadata_path, trace_rows, fieldnames=LEGACY_TRACE_BANK_METADATA_COLUMNS)
    write_csv(out_dir / "trace_bank_metric_summary.csv", trace_bank_metric_summary_rows(trace_rows))
    write_csv(out_dir / "trace_bank_path_length_context_windows.csv", path_length_context_windows(trace_rows))

    bin_summary = quantile_bin_summary(
        trace_rows,
        metric=str(args.summary_bin_metric),
        n_bins=int(args.summary_bins),
    )
    summary = {
        "source_csv": Path(args.source_csv),
        "out_dir": out_dir,
        "sampling_policy": "pandas random sample without replacement from screened BackImage fixation windows, then native center crop",
        "seed": int(args.seed),
        "n_timepoints": int(args.n_timepoints),
        "dt_s": float(args.bin_seconds),
        "total_source_windows": int(rows.shape[0]),
        "usable_source_windows_ge_n_timepoints": int(
            (pd.to_numeric(rows.get("n_samples"), errors="coerce") >= int(args.n_timepoints)).sum()
            if "n_samples" in rows.columns
            else rows.shape[0]
        ),
        "sampled_source_windows": int(sampled.shape[0]),
        "trace_bank_rows": int(len(trace_rows)),
        **builder_meta,
        "rendered_diffusion_constant_arcmin2_s": metric_distribution(
            trace_rows,
            "rendered_diffusion_constant_arcmin2_s",
        ),
        "source_rendered_diffusion_abs_delta_arcmin2_s": source_rendered_diffusion_delta(trace_rows),
        "quantile_bin_partition_check": {
            "covered_rows": int(sum(int(row["n_trace_bank_members"]) for row in bin_summary)),
            "unique_rows": int(len(trace_rows)),
        },
        "bin_summary": bin_summary,
        "microsaccade_split": microsaccade_distribution(trace_rows),
        "trace_bank_metadata_csv": metadata_path,
        "trace_bank_metric_summary_csv": out_dir / "trace_bank_metric_summary.csv",
        "trace_bank_path_length_context_windows_csv": out_dir / "trace_bank_path_length_context_windows.csv",
    }
    write_json(out_dir / "trace_bank_large_sample_summary.json", summary)

    threshold = float(args.path_length_threshold_arcmin)
    filtered_rows = filter_by_path_length(trace_rows, threshold_arcmin=threshold)
    filtered_dir = out_dir / f"filtered_path_length_le{int(round(threshold))}arcmin"
    filtered_metadata_path = filtered_dir / "trace_bank_metadata_filtered.csv"
    if filtered_metadata_path.exists() and not bool(args.force):
        raise FileExistsError(f"Refusing to overwrite {filtered_metadata_path}; pass --force.")
    filtered_dir.mkdir(parents=True, exist_ok=True)
    write_csv(filtered_metadata_path, filtered_rows, fieldnames=LEGACY_TRACE_BANK_METADATA_COLUMNS)
    write_csv(filtered_dir / "trace_bank_metric_summary.csv", trace_bank_metric_summary_rows(filtered_rows))
    write_csv(filtered_dir / "trace_bank_path_length_context_windows.csv", path_length_context_windows(filtered_rows))
    filtered_bin_summary = quantile_bin_summary(
        filtered_rows,
        metric=str(args.summary_bin_metric),
        n_bins=int(args.summary_bins),
    )
    filtered_summary = {
        "source_dir": out_dir,
        "target_dir": filtered_dir,
        "filter": f"path_length_arcmin <= {threshold:g}",
        "path_length_threshold_arcmin": threshold,
        "input_rows": int(len(trace_rows)),
        "kept_rows": int(len(filtered_rows)),
        "dropped_rows": int(len(trace_rows) - len(filtered_rows)),
        "dropped_fraction": float(1.0 - len(filtered_rows) / max(1, len(trace_rows))),
        "kept_microsaccade_event_counts": microsaccade_distribution(filtered_rows)["event_counts"],
        "rendered_diffusion_constant_arcmin2_s": metric_distribution(
            filtered_rows,
            "rendered_diffusion_constant_arcmin2_s",
        ),
        "source_rendered_diffusion_abs_delta_arcmin2_s": source_rendered_diffusion_delta(filtered_rows),
        "bin_summary": filtered_bin_summary,
        "trace_bank_metadata_filtered_csv": filtered_metadata_path,
        "trace_bank_metric_summary_csv": filtered_dir / "trace_bank_metric_summary.csv",
        "trace_bank_path_length_context_windows_csv": filtered_dir / "trace_bank_path_length_context_windows.csv",
    }
    filtered_summary.update(microsaccade_distribution(filtered_rows))
    write_json(filtered_dir / "trace_bank_filtered_summary.json", filtered_summary)
    print(
        json.dumps(
            json_ready(
                {
                    "trace_bank_metadata_csv": metadata_path,
                    "trace_bank_metadata_filtered_csv": filtered_metadata_path,
                    "trace_bank_rows": len(trace_rows),
                    "filtered_rows": len(filtered_rows),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return out_dir


def main() -> int:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
