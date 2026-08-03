#!/usr/bin/env python3
"""Behavioral BackImage component path distributions by local edge coherence."""

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

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
from _fig4_style import INK, configure_matplotlib
from fixation_stats import plot_backimage_contour_motion_components as contour_motion
import panel_f_unwrapped_edge_coherence as panel_h


OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels"
OUT_STEM = "behavior_component_path_by_coherence"
WINDOWS_CSV = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)
PANEL_G_PROVENANCE_JSON = OUT_DIR / "panel_g_relation_sweep_matched_bins_provenance.json"
CONTOUR_MOTION_WINDOWS_CSV = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
    / "contour_motion_component_windows.csv"
)
DT = 1.0 / 120.0
EQUIVALENT_WINDOW_S = 0.325
EQUIVALENT_WINDOW_LABEL = f"{EQUIVALENT_WINDOW_S:g} s"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 83
DIRECTIONAL_PROFILE_ANGLE_STEP_DEG = 3.75
DIRECTIONAL_PROFILE_RANDOMIZATIONS = 1_000
DIRECTIONAL_PROFILE_SEED = 91

COHERENCE_BANDS = (
    (0.0, 0.2, "0-0.2"),
    (0.2, 0.5, "0.2-0.5"),
    (0.5, 0.8, "0.5-0.8"),
    (0.8, 1.0, "0.8-1"),
)
COMPONENTS = (
    ("across", "contour-normal path", "across_component_path_equiv_arcmin", "#7a3b9a"),
    ("along", "contour-parallel path", "along_component_path_equiv_arcmin", "#1b7f5c"),
)
RMS_COMPONENTS = (
    ("along", "contour-parallel RMS", "rms_along_arcmin", "#1b7f5c"),
    ("across", "contour-normal RMS", "rms_across_arcmin", "#7a3b9a"),
)
ALIGNMENT_COLUMNS = (
    "rms_along_arcmin",
    "rms_across_arcmin",
    "rms_delta_along_minus_across_arcmin",
    "rms_ratio_along_over_across",
    "drift_edge_cos2",
)
WINDOW_JOIN_KEYS = ("session", "trial_idx", "global_start", "global_stop", "local_start", "local_stop")
COHERENCE_COLORS = ("#9aa5b1", "#6c8fb5", "#2c7fb8", "#0b4f83")
PATH_BIN_EDGES_ARCMIN = (0.0, 45.0, 55.0, 62.0, 85.0, 113.0, 130.0, math.inf)
PATH_BIN_LABELS = ("<45", "45-55", "55-62", "62-85", "85-113", "113-130", ">=130")
GRID = "#d8dde3"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _coherence_band(coherence: float) -> tuple[int, float, float, str] | None:
    if not math.isfinite(float(coherence)):
        return None
    for order, (lo, hi, label) in enumerate(COHERENCE_BANDS):
        if hi >= 1.0:
            if coherence >= lo and coherence <= hi:
                return order, lo, hi, label
        elif coherence >= lo and coherence < hi:
            return order, lo, hi, label
    return None


def _finite_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _duration_s(row: pd.Series, trace: np.ndarray) -> float:
    duration = pd.to_numeric(pd.Series([row.get("duration_s", np.nan)]), errors="coerce").iloc[0]
    if math.isfinite(float(duration)) and float(duration) > 0:
        return float(duration)
    n = int(trace.shape[0]) if trace.ndim == 2 else 0
    return max(float(n - 1) * DT, DT)


def _component_path_features(row: pd.Series) -> dict[str, float | int]:
    trace = contour_motion._window_trace(row)
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2:
        return {
            "trace_n_samples_loaded": int(x.shape[0]) if x.ndim >= 1 else 0,
            "trace_valid_step_count": 0,
            "total_path_arcmin": float("nan"),
            "along_component_path_arcmin": float("nan"),
            "across_component_path_arcmin": float("nan"),
            "trace_duration_s_used": float("nan"),
            "total_path_equiv_arcmin": float("nan"),
            "along_component_path_equiv_arcmin": float("nan"),
            "across_component_path_equiv_arcmin": float("nan"),
            "along_minus_across_component_path_equiv_arcmin": float("nan"),
            "along_over_across_component_path": float("nan"),
        }
    steps = np.diff(x, axis=0)
    ok = np.isfinite(steps).all(axis=1)
    steps = steps[ok]
    if steps.size == 0:
        return {
            "trace_n_samples_loaded": int(x.shape[0]),
            "trace_valid_step_count": 0,
            "total_path_arcmin": float("nan"),
            "along_component_path_arcmin": float("nan"),
            "across_component_path_arcmin": float("nan"),
            "trace_duration_s_used": float(_duration_s(row, x)),
            "total_path_equiv_arcmin": float("nan"),
            "along_component_path_equiv_arcmin": float("nan"),
            "across_component_path_equiv_arcmin": float("nan"),
            "along_minus_across_component_path_equiv_arcmin": float("nan"),
            "along_over_across_component_path": float("nan"),
        }
    edge_axis = float(row["image_edge_axis_deg"])
    along_vec, across_vec = contour_motion._axis_vectors(np.asarray([edge_axis], dtype=np.float64))
    along_steps = steps @ along_vec[0]
    across_steps = steps @ across_vec[0]
    total = float(np.sum(np.linalg.norm(steps, axis=1)) * 60.0)
    along = float(np.sum(np.abs(along_steps)) * 60.0)
    across = float(np.sum(np.abs(across_steps)) * 60.0)
    duration = _duration_s(row, x)
    scale = EQUIVALENT_WINDOW_S / duration if duration > 0 else float("nan")
    return {
        "trace_n_samples_loaded": int(x.shape[0]),
        "trace_valid_step_count": int(steps.shape[0]),
        "trace_duration_s_used": float(duration),
        "total_path_arcmin": total,
        "along_component_path_arcmin": along,
        "across_component_path_arcmin": across,
        "total_path_equiv_arcmin": total * scale,
        "along_component_path_equiv_arcmin": along * scale,
        "across_component_path_equiv_arcmin": across * scale,
        "along_minus_across_component_path_equiv_arcmin": (along - across) * scale,
        "along_over_across_component_path": along / across if across > 0 else float("nan"),
    }


def _add_equivalent_path_columns(windows: pd.DataFrame) -> pd.DataFrame:
    """Scale raw behavior paths onto the same duration axis as the Panel G trace bank."""

    out = windows.copy()
    stale_cols = [col for col in out.columns if "_200ms" in str(col)]
    if stale_cols:
        out = out.drop(columns=stale_cols)
    if "trace_duration_s_used" in out.columns:
        duration = pd.to_numeric(out["trace_duration_s_used"], errors="coerce")
    else:
        duration = pd.to_numeric(out.get("duration_s", np.nan), errors="coerce")
    scale = EQUIVALENT_WINDOW_S / duration.where(duration > 0)
    for base in ("total_path", "along_component_path", "across_component_path"):
        raw_col = f"{base}_arcmin"
        if raw_col in out.columns:
            out[f"{base}_equiv_arcmin"] = pd.to_numeric(out[raw_col], errors="coerce") * scale
    if {"along_component_path_equiv_arcmin", "across_component_path_equiv_arcmin"}.issubset(out.columns):
        out["along_minus_across_component_path_equiv_arcmin"] = (
            out["along_component_path_equiv_arcmin"] - out["across_component_path_equiv_arcmin"]
        )
    return out


def _add_alignment_columns(windows: pd.DataFrame) -> pd.DataFrame:
    if all(col in windows.columns for col in ALIGNMENT_COLUMNS):
        return windows
    if not CONTOUR_MOTION_WINDOWS_CSV.exists():
        return windows
    needed = list(WINDOW_JOIN_KEYS) + list(ALIGNMENT_COLUMNS)
    ref = pd.read_csv(CONTOUR_MOTION_WINDOWS_CSV, usecols=lambda col: col in needed)
    missing = [col for col in needed if col not in ref.columns]
    if missing:
        return windows

    out = windows.drop(columns=[col for col in ALIGNMENT_COLUMNS if col in windows.columns], errors="ignore").copy()
    same_rows = len(out) == len(ref)
    if same_rows:
        for key in WINDOW_JOIN_KEYS:
            same_rows = same_rows and out[key].astype(str).equals(ref[key].astype(str))
    if same_rows:
        for col in ALIGNMENT_COLUMNS:
            out[col] = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=float)
        return out

    merged = out.merge(ref, on=list(WINDOW_JOIN_KEYS), how="left", validate="one_to_one")
    return merged


def compute_component_paths(windows_csv: Path = WINDOWS_CSV) -> pd.DataFrame:
    required = [
        "session",
        "stimulus",
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "duration_s",
        "phase",
        "image_feature_ok",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "path_length_deg",
        "rms_radius_deg",
        "drift_edge_cos2",
    ]
    windows = pd.read_csv(windows_csv, usecols=lambda col: col in required)
    missing = [col for col in required if col not in windows.columns]
    if missing:
        raise ValueError(f"Missing required columns in {windows_csv}: {missing}")

    work = windows[windows["stimulus"].astype(str).eq("backimage")].copy()
    work = work[_finite_bool(work["image_feature_ok"])].copy()
    for col in ["image_orientation_coherence", "image_edge_axis_deg", "path_length_deg", "duration_s"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[
        np.isfinite(work["image_orientation_coherence"])
        & np.isfinite(work["image_edge_axis_deg"])
        & np.isfinite(work["path_length_deg"])
        & np.isfinite(work["duration_s"])
        & (work["duration_s"] > 0)
    ].copy()
    bands = [_coherence_band(float(v)) for v in work["image_orientation_coherence"]]
    work["coherence_band_order"] = [band[0] if band is not None else np.nan for band in bands]
    work["coherence_bin_low"] = [band[1] if band is not None else np.nan for band in bands]
    work["coherence_bin_high"] = [band[2] if band is not None else np.nan for band in bands]
    work["coherence_bin"] = [band[3] if band is not None else None for band in bands]
    work = work[work["coherence_bin"].notna()].copy()
    work["subject"] = work["session"].map(contour_motion._subject_from_session)

    rows: list[dict[str, Any]] = []
    total = len(work)
    for idx, row in enumerate(work.itertuples(index=False), start=1):
        series = pd.Series(row._asdict())
        features = _component_path_features(series)
        rows.append(features)
        if idx % 2000 == 0:
            print(f"computed component paths for {idx}/{total} behavior windows", flush=True)
    features = pd.DataFrame(rows, index=work.index)
    out = pd.concat([work.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["table_total_path_arcmin"] = out["path_length_deg"].astype(float) * 60.0
    out["recomputed_minus_table_total_path_arcmin"] = out["total_path_arcmin"] - out["table_total_path_arcmin"]
    out = _add_equivalent_path_columns(out)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _bootstrap_median(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.median(values))
    if values.size == 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        boots[i] = float(np.median(values[rng.integers(0, values.size, size=values.size)]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize(windows: pd.DataFrame, *, n_bootstrap: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band_order, band in enumerate([band[2] for band in COHERENCE_BANDS]):
        sub_band = windows[windows["coherence_bin"].astype(str).eq(band)].copy()
        for component, component_label, metric, _color in COMPONENTS:
            vals = pd.to_numeric(sub_band[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            session_values = (
                sub_band.groupby("session", observed=True)[metric]
                .median(numeric_only=True)
                .to_numpy(dtype=float)
            )
            point, lo, hi = _bootstrap_median(
                session_values,
                n_bootstrap=n_bootstrap,
                seed=seed + 100 * band_order + len(rows),
            )
            rows.append(
                {
                    "coherence_band_order": int(band_order),
                    "coherence_bin": band,
                    "component": component,
                    "component_label": component_label,
                    "metric": metric,
                    "window_median_arcmin_equiv": float(np.nanmedian(vals)) if vals.size else float("nan"),
                    "window_q10_arcmin_equiv": float(np.nanpercentile(vals, 10.0)) if vals.size else float("nan"),
                    "window_q25_arcmin_equiv": float(np.nanpercentile(vals, 25.0)) if vals.size else float("nan"),
                    "window_q75_arcmin_equiv": float(np.nanpercentile(vals, 75.0)) if vals.size else float("nan"),
                    "window_q90_arcmin_equiv": float(np.nanpercentile(vals, 90.0)) if vals.size else float("nan"),
                    "session_median_arcmin_equiv": point,
                    "session_ci95_low_arcmin_equiv": lo,
                    "session_ci95_high_arcmin_equiv": hi,
                    "n_windows": int(vals.size),
                    "n_sessions": int(sub_band["session"].nunique()),
                }
            )

        diff_metric = "along_minus_across_component_path_equiv_arcmin"
        diff_session = (
            sub_band.groupby("session", observed=True)[diff_metric]
            .median(numeric_only=True)
            .to_numpy(dtype=float)
        )
        point, lo, hi = _bootstrap_median(
            diff_session,
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 + band_order,
        )
        diff_vals = pd.to_numeric(sub_band[diff_metric], errors="coerce").to_numpy(dtype=float)
        diff_vals = diff_vals[np.isfinite(diff_vals)]
        rows.append(
            {
                "coherence_band_order": int(band_order),
                "coherence_bin": band,
                "component": "along_minus_across",
                "component_label": "along - across",
                "metric": diff_metric,
                "window_median_arcmin_equiv": float(np.nanmedian(diff_vals)) if diff_vals.size else float("nan"),
                "window_q10_arcmin_equiv": float(np.nanpercentile(diff_vals, 10.0)) if diff_vals.size else float("nan"),
                "window_q25_arcmin_equiv": float(np.nanpercentile(diff_vals, 25.0)) if diff_vals.size else float("nan"),
                "window_q75_arcmin_equiv": float(np.nanpercentile(diff_vals, 75.0)) if diff_vals.size else float("nan"),
                "window_q90_arcmin_equiv": float(np.nanpercentile(diff_vals, 90.0)) if diff_vals.size else float("nan"),
                "session_median_arcmin_equiv": point,
                "session_ci95_low_arcmin_equiv": lo,
                "session_ci95_high_arcmin_equiv": hi,
                "n_windows": int(diff_vals.size),
                "n_sessions": int(sub_band["session"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_alignment(
    windows: pd.DataFrame,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_specs = list(RMS_COMPONENTS) + [
        ("rms_delta", "parallel - normal RMS", "rms_delta_along_minus_across_arcmin", "#284b63"),
        ("cos2", "motion-contour cos2", "drift_edge_cos2", "#111111"),
    ]
    for band_order, band in enumerate([band[2] for band in COHERENCE_BANDS]):
        sub_band = windows[windows["coherence_bin"].astype(str).eq(band)].copy()
        for metric_key, metric_label, metric, _color in metric_specs:
            if metric not in sub_band.columns:
                continue
            vals = pd.to_numeric(sub_band[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            session_values = (
                sub_band.groupby("session", observed=True)[metric]
                .median(numeric_only=True)
                .to_numpy(dtype=float)
            )
            point, lo, hi = _bootstrap_median(
                session_values,
                n_bootstrap=n_bootstrap,
                seed=seed + 5000 + 100 * band_order + len(rows),
            )
            rows.append(
                {
                    "coherence_band_order": int(band_order),
                    "coherence_bin": band,
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "metric": metric,
                    "window_median": float(np.nanmedian(vals)) if vals.size else float("nan"),
                    "window_q25": float(np.nanpercentile(vals, 25.0)) if vals.size else float("nan"),
                    "window_q75": float(np.nanpercentile(vals, 75.0)) if vals.size else float("nan"),
                    "session_median": point,
                    "session_ci95_low": lo,
                    "session_ci95_high": hi,
                    "n_windows": int(vals.size),
                    "n_sessions": int(sub_band["session"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def occupancy_table(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band_order, band in enumerate([band[2] for band in COHERENCE_BANDS]):
        sub_band = windows[windows["coherence_bin"].astype(str).eq(band)]
        for component, component_label, metric, _color in COMPONENTS:
            vals = pd.to_numeric(sub_band[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            total = max(int(vals.size), 1)
            for bin_index, label in enumerate(PATH_BIN_LABELS):
                lo = float(PATH_BIN_EDGES_ARCMIN[bin_index])
                hi = float(PATH_BIN_EDGES_ARCMIN[bin_index + 1])
                if math.isinf(hi):
                    keep = vals >= lo
                else:
                    keep = (vals >= lo) & (vals < hi)
                count = int(np.count_nonzero(keep))
                rows.append(
                    {
                        "coherence_band_order": int(band_order),
                        "coherence_bin": band,
                        "component": component,
                        "component_label": component_label,
                        "path_bin_order": int(bin_index),
                        "path_bin": label,
                        "path_bin_low_arcmin_equiv": lo,
                        "path_bin_high_arcmin_equiv": hi,
                        "n_windows": count,
                        "fraction_windows": float(count / total),
                        "percent_windows": float(100.0 * count / total),
                        "n_component_values": int(vals.size),
                    }
                )
    return pd.DataFrame(rows)


def _directional_profile_angles(angle_step_deg: float = DIRECTIONAL_PROFILE_ANGLE_STEP_DEG) -> np.ndarray:
    return np.arange(0.0, 180.0 + 0.5 * float(angle_step_deg), float(angle_step_deg), dtype=float)


def _component_path_profile_for_row(row: pd.Series, angles_rad: np.ndarray) -> np.ndarray:
    trace = contour_motion._window_trace(row)
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2:
        return np.full_like(angles_rad, np.nan, dtype=np.float64)
    steps = np.diff(x, axis=0)
    ok = np.isfinite(steps).all(axis=1)
    steps = steps[ok]
    if steps.size == 0:
        return np.full_like(angles_rad, np.nan, dtype=np.float64)
    duration = _duration_s(row, x)
    scale = EQUIVALENT_WINDOW_S / duration if duration > 0 else float("nan")
    if not np.isfinite(scale):
        return np.full_like(angles_rad, np.nan, dtype=np.float64)
    edge_axis = np.radians(float(row["image_edge_axis_deg"]))
    theta = edge_axis + np.asarray(angles_rad, dtype=np.float64)
    axes = np.column_stack([np.cos(theta), np.sin(theta)])
    projected_steps = steps @ axes.T
    return np.sum(np.abs(projected_steps), axis=0) * 60.0 * scale


def compute_directional_component_path_profiles(
    windows: pd.DataFrame,
    *,
    angle_step_deg: float = DIRECTIONAL_PROFILE_ANGLE_STEP_DEG,
    n_randomizations: int = DIRECTIONAL_PROFILE_RANDOMIZATIONS,
    seed: int = DIRECTIONAL_PROFILE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute b_position_spread-style unwrapped profiles for unsigned path length.

    The observed profile projects each trace's sample-to-sample steps onto axes
    0..180 degrees from the local contour and sums absolute projected distance.
    The random-orientation reference samples a random relative axis per window,
    preserving each trace's step sequence but breaking its measured contour axis.
    """

    angles_deg = _directional_profile_angles(angle_step_deg)
    angles_rad = np.radians(angles_deg)
    rng = np.random.default_rng(seed)
    profile_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    random_angle_choices = np.arange(max(1, len(angles_deg) - 1), dtype=int)
    for band_order, (_lo, _hi, band) in enumerate(COHERENCE_BANDS):
        sub = windows[windows["coherence_bin"].astype(str).eq(band)].copy()
        profiles: list[np.ndarray] = []
        total = len(sub)
        for idx, row in enumerate(sub.itertuples(index=False), start=1):
            series = pd.Series(row._asdict())
            profiles.append(_component_path_profile_for_row(series, angles_rad))
            if idx % 1000 == 0:
                print(f"computed directional component-path profiles for {band}: {idx}/{total}", flush=True)
        if not profiles:
            continue
        matrix = np.vstack(profiles)
        observed = np.nanmedian(matrix, axis=0)
        n_finite = int(np.count_nonzero(np.isfinite(matrix).any(axis=1)))
        for angle, value in zip(angles_deg, observed, strict=True):
            profile_rows.append(
                {
                    "coherence_band_order": int(band_order),
                    "coherence_bin": band,
                    "relative_angle_deg": float(angle),
                    "component_path_equiv_arcmin": float(value),
                    "n_windows": n_finite,
                }
            )

        valid_rows = np.isfinite(matrix).any(axis=1)
        valid_matrix = matrix[valid_rows]
        null_medians = np.full(int(n_randomizations), np.nan, dtype=np.float64)
        if valid_matrix.size:
            row_idx = np.arange(valid_matrix.shape[0])
            for i in range(int(n_randomizations)):
                angle_idx = rng.choice(random_angle_choices, size=valid_matrix.shape[0], replace=True)
                null_medians[i] = float(np.nanmedian(valid_matrix[row_idx, angle_idx]))
        null_med = float(np.nanmedian(null_medians))
        null_lo, null_hi = np.nanquantile(null_medians, [0.025, 0.975])
        for angle, value in zip(angles_deg, observed, strict=True):
            reference_rows.append(
                {
                    "coherence_band_order": int(band_order),
                    "coherence_bin": band,
                    "relative_angle_deg": float(angle),
                    "n_windows": n_finite,
                    "observed_component_path_equiv_arcmin": float(value),
                    "random_orientation_median_component_path_equiv_arcmin": null_med,
                    "random_orientation_ci95_low_component_path_equiv_arcmin": float(null_lo),
                    "random_orientation_ci95_high_component_path_equiv_arcmin": float(null_hi),
                    "observed_minus_random_median_component_path_equiv_arcmin": float(value - null_med)
                    if np.isfinite(value) and np.isfinite(null_med)
                    else float("nan"),
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(reference_rows)


def _load_reference_band() -> dict[str, float] | None:
    if not PANEL_G_PROVENANCE_JSON.exists():
        return None
    try:
        payload = json.loads(PANEL_G_PROVENANCE_JSON.read_text(encoding="utf-8"))
        context = payload["binning"]["standard_drift_component_path_context"]
        return {
            "q25_arcmin": float(context["q25_arcmin"]),
            "q75_arcmin": float(context["q75_arcmin"]),
        }
    except Exception:
        return None


def _log_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return np.log10(arr)


def _format_log_y_axis(ax: plt.Axes, values: np.ndarray | None = None) -> None:
    ticks = np.asarray([25.0, 35.0, 45.0, 55.0, 70.0, 85.0, 113.0, 150.0], dtype=float)
    if values is not None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size:
            lo = max(10.0, float(np.nanpercentile(finite, 0.5)) * 0.85)
            hi = float(np.nanpercentile(finite, 99.5)) * 1.15
            ticks = ticks[(ticks >= lo * 0.95) & (ticks <= hi * 1.05)]
            if ticks.size < 4:
                ticks = np.asarray([20.0, 40.0, 70.0, 113.0, 160.0])
            ax.set_ylim(math.log10(lo), math.log10(hi))
    ax.set_yticks(np.log10(ticks))
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])


def _coherence_tick_labels(windows: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for _lo, _hi, label in COHERENCE_BANDS:
        n = int(np.count_nonzero(windows["coherence_bin"].astype(str).eq(label)))
        labels.append(f"{label}\nn={n}")
    return labels


def _draw_violin_panel(ax: plt.Axes, windows: pd.DataFrame, reference: dict[str, float] | None) -> None:
    all_values = []
    positions = []
    colors = []
    labels = [band[2] for band in COHERENCE_BANDS]
    offsets = {"across": -0.14, "along": 0.14}
    for band_order, band in enumerate(labels):
        for component, _component_label, metric, color in COMPONENTS:
            vals = pd.to_numeric(
                windows.loc[windows["coherence_bin"].astype(str).eq(band), metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                continue
            all_values.append(np.log10(vals))
            positions.append(float(band_order) + offsets[component])
            colors.append(color)
    parts = ax.violinplot(all_values, positions=positions, widths=0.22, showmeans=False, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(0.8)
    for pos, vals, color in zip(positions, all_values, colors, strict=True):
        qs = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.plot([pos, pos], [qs[0], qs[2]], color=color, lw=1.6, zorder=3)
        ax.scatter([pos], [qs[1]], s=13, color=color, edgecolor="white", linewidth=0.4, zorder=4)
    if reference:
        ax.axhspan(math.log10(reference["q25_arcmin"]), math.log10(reference["q75_arcmin"]), color="#777777", alpha=0.10, zorder=0)
    all_raw = windows[["across_component_path_equiv_arcmin", "along_component_path_equiv_arcmin"]].to_numpy(dtype=float).ravel()
    _format_log_y_axis(ax, all_raw)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(_coherence_tick_labels(windows))
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel(f"component path per {EQUIVALENT_WINDOW_LABEL}\n(arcmin; log scale)")
    ax.set_title(f"A  Window distributions, {EQUIVALENT_WINDOW_LABEL}-equivalent", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    _clean_axis(ax)


def _draw_session_summary_panel(ax: plt.Axes, summary: pd.DataFrame, reference: dict[str, float] | None) -> None:
    x = np.arange(len(COHERENCE_BANDS), dtype=float)
    for component, label, _metric, color in COMPONENTS:
        sub = summary[summary["component"].astype(str).eq(component)].sort_values("coherence_band_order")
        y = sub["session_median_arcmin_equiv"].to_numpy(dtype=float)
        lo = sub["session_ci95_low_arcmin_equiv"].to_numpy(dtype=float)
        hi = sub["session_ci95_high_arcmin_equiv"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker="o", lw=1.8, capsize=2.5, label=label)
    if reference:
        ax.axhspan(reference["q25_arcmin"], reference["q75_arcmin"], color="#777777", alpha=0.10, zorder=0, label="Panel G q25-q75")
    ax.set_xticks(x)
    ax.set_xticklabels([band[2] for band in COHERENCE_BANDS])
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel(f"unsigned path per {EQUIVALENT_WINDOW_LABEL} (arcmin)")
    ax.set_title("C  Unsigned component path", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _clean_axis(ax)


def _draw_rms_alignment_panel(ax: plt.Axes, alignment_summary: pd.DataFrame) -> None:
    x = np.arange(len(COHERENCE_BANDS), dtype=float)
    handles: list[Any] = []
    labels: list[str] = []
    for component, label, _metric, color in RMS_COMPONENTS:
        sub = alignment_summary[alignment_summary["metric_key"].astype(str).eq(component)].sort_values("coherence_band_order")
        if sub.empty:
            continue
        y = sub["session_median"].to_numpy(dtype=float)
        lo = sub["session_ci95_low"].to_numpy(dtype=float)
        hi = sub["session_ci95_high"].to_numpy(dtype=float)
        handle = ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker="o", lw=1.8, capsize=2.5, label=label)
        handles.append(handle)
        labels.append(label)
    ax.set_xticks(x)
    ax.set_xticklabels([band[2] for band in COHERENCE_BANDS])
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel("position RMS (arcmin)")
    ax.set_title("B  Covariance/RMS alignment", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.75)
    _clean_axis(ax)

    sub = alignment_summary[alignment_summary["metric_key"].astype(str).eq("cos2")].sort_values("coherence_band_order")
    if not sub.empty:
        ax2 = ax.twinx()
        y = sub["session_median"].to_numpy(dtype=float)
        lo = sub["session_ci95_low"].to_numpy(dtype=float)
        hi = sub["session_ci95_high"].to_numpy(dtype=float)
        handle = ax2.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color="#222222", marker="s", lw=1.4, capsize=2.0, ls="--", label="cos2")
        handles.append(handle)
        labels.append("cos2")
        ax2.set_ylabel("motion-contour cos2")
        ax2.spines["top"].set_visible(False)
        ax2.set_ylim(-0.05, 0.75)
    ax.legend(handles, labels, frameon=False, fontsize=7, loc="best")


def _draw_delta_panel(ax: plt.Axes, summary: pd.DataFrame, alignment_summary: pd.DataFrame) -> None:
    x = np.arange(len(COHERENCE_BANDS), dtype=float)
    path = summary[summary["component"].astype(str).eq("along_minus_across")].sort_values("coherence_band_order")
    if not path.empty:
        y = path["session_median_arcmin_equiv"].to_numpy(dtype=float)
        lo = path["session_ci95_low_arcmin_equiv"].to_numpy(dtype=float)
        hi = path["session_ci95_high_arcmin_equiv"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color="#5e3c99", marker="o", lw=1.8, capsize=2.5, label="path: parallel - normal")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([band[2] for band in COHERENCE_BANDS])
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel(f"path delta per {EQUIVALENT_WINDOW_LABEL} (arcmin)")
    ax.set_title("D  Same windows, different summaries", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.75)
    _clean_axis(ax)

    rms = alignment_summary[alignment_summary["metric_key"].astype(str).eq("rms_delta")].sort_values("coherence_band_order")
    if not rms.empty:
        ax2 = ax.twinx()
        y = rms["session_median"].to_numpy(dtype=float)
        lo = rms["session_ci95_low"].to_numpy(dtype=float)
        hi = rms["session_ci95_high"].to_numpy(dtype=float)
        ax2.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color="#1b7f5c", marker="s", lw=1.4, capsize=2.0, ls="--", label="RMS: parallel - normal")
        ax2.axhline(0, color="#555555", lw=0.8, alpha=0.45)
        ax2.set_ylabel("RMS delta (arcmin)")
        ax2.spines["top"].set_visible(False)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=7, loc="best")
    else:
        ax.legend(frameon=False, fontsize=7, loc="best")


def _draw_occupancy_heatmap(
    ax: plt.Axes,
    occupancy: pd.DataFrame,
    *,
    component: str,
    title: str,
    vmax: float,
) -> Any:
    sub = occupancy[occupancy["component"].astype(str).eq(component)].copy()
    matrix = (
        sub.pivot(index="path_bin_order", columns="coherence_band_order", values="percent_windows")
        .reindex(index=range(len(PATH_BIN_LABELS)), columns=range(len(COHERENCE_BANDS)))
        .to_numpy(dtype=float)
    )
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=float(vmax))
    ax.set_xticks(range(len(COHERENCE_BANDS)))
    ax.set_xticklabels([band[2] for band in COHERENCE_BANDS])
    ax.set_yticks(range(len(PATH_BIN_LABELS)))
    ax.set_yticklabels(PATH_BIN_LABELS)
    ax.set_xlabel("local edge coherence")
    ax.set_ylabel(f"component path bin\narcmin per {EQUIVALENT_WINDOW_LABEL}")
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            val = matrix[y, x]
            if np.isfinite(val):
                label = "<1" if 0.0 < val < 0.5 else f"{val:.0f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=6.4, color="#17212b" if val < 0.55 * vmax else "white")
    return im


def plot_sheet(
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    occupancy: pd.DataFrame,
    *,
    reference: dict[str, float] | None,
    out_dir: Path,
) -> dict[str, Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), constrained_layout=True)
    _draw_violin_panel(axes[0, 0], windows, reference)
    _draw_rms_alignment_panel(axes[0, 1], alignment_summary)
    _draw_session_summary_panel(axes[0, 2], summary, reference)
    _draw_delta_panel(axes[1, 0], summary, alignment_summary)
    vmax = max(35.0, float(np.nanmax(occupancy["percent_windows"].to_numpy(dtype=float))))
    im = _draw_occupancy_heatmap(
        axes[1, 1],
        occupancy,
        component="across",
        title="E  Normal-path occupancy (%)",
        vmax=vmax,
    )
    _draw_occupancy_heatmap(
        axes[1, 2],
        occupancy,
        component="along",
        title="F  Parallel-path occupancy (%)",
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=axes[1, 1:], shrink=0.92, pad=0.015)
    cbar.set_label("% windows")
    fig.suptitle(
        "Behavioral BackImage component path lengths by local edge coherence",
        fontsize=12.5,
        fontweight="bold",
    )
    paths = {
        "png": out_dir / f"{OUT_STEM}.png",
        "pdf": out_dir / f"{OUT_STEM}.pdf",
        "svg": out_dir / f"{OUT_STEM}.svg",
    }
    fig.savefig(paths["png"], dpi=230)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    plt.close(fig)
    return paths


def _band_color_map() -> dict[str, str]:
    return {band[2]: color for band, color in zip(COHERENCE_BANDS, COHERENCE_COLORS, strict=True)}


def _style_unwrapped_axis(ax: plt.Axes) -> None:
    ax.axvline(90.0, color="#7d858c", lw=0.75, ls=":", zorder=1)
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks([0.0, 90.0, 180.0])
    ax.set_xticklabels(["parallel", "normal", "parallel"])
    ax.grid(axis="y", color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    _clean_axis(ax)


def _path_reference_row(path_reference: pd.DataFrame, band: str) -> pd.Series | None:
    sub = path_reference[path_reference["coherence_bin"].astype(str).eq(band)]
    if sub.empty:
        return None
    return sub.iloc[0]


def plot_directional_component_path_profiles(
    path_profile: pd.DataFrame,
    path_reference: pd.DataFrame,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    colors = _band_color_map()
    fig, axes = plt.subplots(1, len(COHERENCE_BANDS), figsize=(10.6, 2.7), sharey=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    finite = path_reference[
        [
            "observed_component_path_equiv_arcmin",
            "random_orientation_ci95_low_component_path_equiv_arcmin",
            "random_orientation_ci95_high_component_path_equiv_arcmin",
        ]
    ].to_numpy(dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    ylo, yhi = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = max(1.0, 0.08 * (yhi - ylo))
    for ax, (_lo, _hi, band) in zip(axes_arr, COHERENCE_BANDS, strict=True):
        color = colors[band]
        sub = path_reference[path_reference["coherence_bin"].astype(str).eq(band)].sort_values("relative_angle_deg")
        if sub.empty:
            continue
        x = sub["relative_angle_deg"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            sub["random_orientation_ci95_low_component_path_equiv_arcmin"].to_numpy(dtype=float),
            sub["random_orientation_ci95_high_component_path_equiv_arcmin"].to_numpy(dtype=float),
            color="#555555",
            alpha=0.10,
            lw=0,
            zorder=0,
        )
        ax.plot(
            x,
            sub["random_orientation_median_component_path_equiv_arcmin"],
            color="#333333",
            lw=1.0,
            ls=(0, (3.0, 2.0)),
            label="random axis",
            zorder=2,
        )
        ax.plot(x, sub["observed_component_path_equiv_arcmin"], color=color, lw=1.8, label="observed", zorder=3)
        ax.set_ylim(ylo - pad, yhi + pad)
        n_windows = int(sub["n_windows"].dropna().iloc[0])
        ax.set_title(f"coh {band}\nn={n_windows}", fontsize=8.5)
        _style_unwrapped_axis(ax)
    axes_arr[0].set_ylabel(f"unsigned component path\nper {EQUIVALENT_WINDOW_LABEL} (arcmin)")
    for ax in axes_arr:
        ax.set_xlabel("axis from local edge")
    axes_arr[-1].legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle("Directional unsigned component path relative to local edge", fontsize=11.5, fontweight="bold")
    paths = {
        "component_path_profile_png": out_dir / f"{OUT_STEM}_directional_path_profile.png",
        "component_path_profile_pdf": out_dir / f"{OUT_STEM}_directional_path_profile.pdf",
        "component_path_profile_svg": out_dir / f"{OUT_STEM}_directional_path_profile.svg",
    }
    fig.savefig(paths["component_path_profile_png"], dpi=230)
    fig.savefig(paths["component_path_profile_pdf"])
    fig.savefig(paths["component_path_profile_svg"])
    plt.close(fig)
    return paths


def plot_position_spread_vs_component_path_profiles(
    path_reference: pd.DataFrame,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    rms_values = panel_h.load_panel_values()
    rms_reference = panel_h.load_random_orientation_reference()
    colors = _band_color_map()
    fig, axes = plt.subplots(2, len(COHERENCE_BANDS), figsize=(10.8, 5.1), constrained_layout=True)

    rms_finite = rms_values["rms_arcmin"].to_numpy(dtype=float)
    rms_ref_finite = rms_reference[
        [
            "random_orientation_ci95_low_arcmin",
            "random_orientation_ci95_high_arcmin",
        ]
    ].to_numpy(dtype=float).ravel()
    rms_all = np.concatenate([rms_finite, rms_ref_finite])
    rms_all = rms_all[np.isfinite(rms_all)]
    rms_lo, rms_hi = float(np.nanmin(rms_all)), float(np.nanmax(rms_all))
    rms_pad = max(0.04, 0.08 * (rms_hi - rms_lo))

    path_finite = path_reference[
        [
            "observed_component_path_equiv_arcmin",
            "random_orientation_ci95_low_component_path_equiv_arcmin",
            "random_orientation_ci95_high_component_path_equiv_arcmin",
        ]
    ].to_numpy(dtype=float).ravel()
    path_finite = path_finite[np.isfinite(path_finite)]
    path_lo, path_hi = float(np.nanmin(path_finite)), float(np.nanmax(path_finite))
    path_pad = max(1.0, 0.08 * (path_hi - path_lo))

    for col, (_lo, _hi, band) in enumerate(COHERENCE_BANDS):
        color = colors[band]
        ax = axes[0, col]
        rms_sub = rms_values[rms_values["wide_coherence_bin"].astype(str).eq(band)].sort_values("relative_angle_deg")
        rms_ref = rms_reference[rms_reference["wide_coherence_bin"].astype(str).eq(band)]
        x = rms_sub["relative_angle_deg"].to_numpy(dtype=float)
        if not rms_ref.empty:
            ref = rms_ref.iloc[0]
            ax.fill_between(
                x,
                np.full_like(x, float(ref["random_orientation_ci95_low_arcmin"])),
                np.full_like(x, float(ref["random_orientation_ci95_high_arcmin"])),
                color="#555555",
                alpha=0.10,
                lw=0,
            )
            ax.plot(
                x,
                np.full_like(x, float(ref["random_orientation_median_rms_arcmin"])),
                color="#333333",
                lw=1.0,
                ls=(0, (3.0, 2.0)),
            )
        ax.plot(x, rms_sub["rms_arcmin"], color=color, lw=1.8)
        ax.set_ylim(rms_lo - rms_pad, rms_hi + rms_pad)
        ax.set_title(f"coh {band}", fontsize=8.5)
        _style_unwrapped_axis(ax)

        ax = axes[1, col]
        path_sub = path_reference[path_reference["coherence_bin"].astype(str).eq(band)].sort_values("relative_angle_deg")
        x = path_sub["relative_angle_deg"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            path_sub["random_orientation_ci95_low_component_path_equiv_arcmin"].to_numpy(dtype=float),
            path_sub["random_orientation_ci95_high_component_path_equiv_arcmin"].to_numpy(dtype=float),
            color="#555555",
            alpha=0.10,
            lw=0,
        )
        ax.plot(
            x,
            path_sub["random_orientation_median_component_path_equiv_arcmin"],
            color="#333333",
            lw=1.0,
            ls=(0, (3.0, 2.0)),
        )
        ax.plot(x, path_sub["observed_component_path_equiv_arcmin"], color=color, lw=1.8)
        ax.set_ylim(path_lo - path_pad, path_hi + path_pad)
        _style_unwrapped_axis(ax)

    axes[0, 0].set_ylabel("position spread RMS\n(arcmin)")
    axes[1, 0].set_ylabel(f"unsigned component path\nper {EQUIVALENT_WINDOW_LABEL} (arcmin)")
    for ax in axes[1, :]:
        ax.set_xlabel("axis from local edge")
    fig.suptitle(
        "Same windows: position spread vs unsigned component path",
        fontsize=11.5,
        fontweight="bold",
    )
    paths = {
        "spread_vs_path_profile_png": out_dir / f"{OUT_STEM}_spread_vs_directional_path_profile.png",
        "spread_vs_path_profile_pdf": out_dir / f"{OUT_STEM}_spread_vs_directional_path_profile.pdf",
        "spread_vs_path_profile_svg": out_dir / f"{OUT_STEM}_spread_vs_directional_path_profile.svg",
    }
    fig.savefig(paths["spread_vs_path_profile_png"], dpi=230)
    fig.savefig(paths["spread_vs_path_profile_pdf"])
    fig.savefig(paths["spread_vs_path_profile_svg"])
    plt.close(fig)
    return paths


def build(out_dir: Path = OUT_DIR, *, force_recompute: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_out = out_dir / f"{OUT_STEM}_windows.csv"
    summary_out = out_dir / f"{OUT_STEM}_summary.csv"
    alignment_summary_out = out_dir / f"{OUT_STEM}_alignment_summary.csv"
    occupancy_out = out_dir / f"{OUT_STEM}_occupancy.csv"
    directional_profile_out = out_dir / f"{OUT_STEM}_directional_path_profile.csv"
    directional_reference_out = out_dir / f"{OUT_STEM}_directional_path_random_orientation_reference.csv"
    provenance_out = out_dir / f"{OUT_STEM}_provenance.json"
    if windows_out.exists() and not force_recompute:
        windows = _add_alignment_columns(_add_equivalent_path_columns(pd.read_csv(windows_out)))
        windows.to_csv(windows_out, index=False)
    else:
        windows = _add_alignment_columns(compute_component_paths(WINDOWS_CSV))
        windows.to_csv(windows_out, index=False)
    summary = summarize(windows)
    alignment_summary = summarize_alignment(windows)
    occupancy = occupancy_table(windows)
    summary.to_csv(summary_out, index=False)
    alignment_summary.to_csv(alignment_summary_out, index=False)
    occupancy.to_csv(occupancy_out, index=False)
    if directional_profile_out.exists() and directional_reference_out.exists() and not force_recompute:
        directional_profile = pd.read_csv(directional_profile_out)
        directional_reference = pd.read_csv(directional_reference_out)
    else:
        directional_profile, directional_reference = compute_directional_component_path_profiles(windows)
        directional_profile.to_csv(directional_profile_out, index=False)
        directional_reference.to_csv(directional_reference_out, index=False)
    reference = _load_reference_band()
    figure_paths = plot_sheet(windows, summary, alignment_summary, occupancy, reference=reference, out_dir=out_dir)
    directional_figure_paths = plot_directional_component_path_profiles(
        directional_profile,
        directional_reference,
        out_dir=out_dir,
    )
    comparison_figure_paths = plot_position_spread_vs_component_path_profiles(
        directional_reference,
        out_dir=out_dir,
    )
    _write_json(
        provenance_out,
        {
            "analysis": OUT_STEM,
            "input_windows_csv": WINDOWS_CSV,
            "outputs": {
                **figure_paths,
                **directional_figure_paths,
                **comparison_figure_paths,
                "windows_csv": windows_out,
                "summary_csv": summary_out,
                "alignment_summary_csv": alignment_summary_out,
                "occupancy_csv": occupancy_out,
                "directional_path_profile_csv": directional_profile_out,
                "directional_path_random_orientation_reference_csv": directional_reference_out,
                "provenance_json": provenance_out,
            },
            "selection": {
                "stimulus": "backimage",
                "image_feature_ok": True,
                "coherence_bands": COHERENCE_BANDS,
                "n_windows": int(len(windows)),
                "n_sessions": int(windows["session"].nunique()),
            },
            "metric_definition": {
                "raw_component_path_arcmin": "sum(abs(projected sample-to-sample eye displacement)) along/across local image_edge_axis_deg, in arcmin.",
                "plotted_component_path_arcmin": f"raw component path scaled by {EQUIVALENT_WINDOW_S:g} s / window duration, matching the Panel G trace-bank snippet duration.",
                "panel_g_trace_bank_snippet_duration_s": EQUIVALENT_WINDOW_S,
                "path_bin_edges_arcmin_equiv": PATH_BIN_EDGES_ARCMIN,
                "path_bin_labels": PATH_BIN_LABELS,
                "alignment_columns_source": CONTOUR_MOTION_WINDOWS_CSV,
                "directional_component_path_profile": (
                    "For each window and relative axis angle, sum(abs(projected sample-to-sample eye displacement)) "
                    f"and scale by {EQUIVALENT_WINDOW_S:g} s / window duration."
                ),
                "directional_random_orientation_reference": (
                    "For each coherence band, sample a random relative axis per window from the profile angle grid "
                    "and take the median unsigned component path. The reference is flat across plotted relative "
                    "angle because a randomized contour axis removes the meaning of parallel/normal."
                ),
            },
            "bootstrap": {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "unit": "session bootstrap over session medians",
            },
            "directional_profile": {
                "angle_step_deg": DIRECTIONAL_PROFILE_ANGLE_STEP_DEG,
                "n_randomizations": DIRECTIONAL_PROFILE_RANDOMIZATIONS,
                "seed": DIRECTIONAL_PROFILE_SEED,
            },
            "panel_g_reference_band": reference,
        },
    )
    return {
        **figure_paths,
        **directional_figure_paths,
        **comparison_figure_paths,
        "windows_csv": windows_out,
        "summary_csv": summary_out,
        "alignment_summary_csv": alignment_summary_out,
        "occupancy_csv": occupancy_out,
        "directional_path_profile_csv": directional_profile_out,
        "directional_path_random_orientation_reference_csv": directional_reference_out,
        "provenance_json": provenance_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    paths = build(args.out_dir, force_recompute=bool(args.force_recompute))
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
