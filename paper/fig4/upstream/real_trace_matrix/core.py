from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
FIG4_DIR = ROOT / "paper" / "fig4"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FIG4_DIR) not in sys.path:
    sys.path.insert(0, str(FIG4_DIR))

EPS = 1e-8
IMAGE_FEATURE_COLUMNS = [
    "image_patch_rms_contrast",
    "image_patch_std",
    "image_gradient_energy",
    "image_oriented_gradient_energy",
    "image_multi_orientation_energy",
    "image_edge_density",
    "image_orientation_coherence",
    "image_gradient_axis_deg",
    "image_edge_axis_deg",
    "image_spectrum_anisotropy",
    "image_edge_spectrum_contour_axis_agreement",
    "image_oriented_8plus_power_proxy",
    "image_spectrum_orientation_deg",
    "image_high_freq_power_fraction",
    "image_power_8plus_cpd_fraction",
    "image_contour_reliable",
    "image_contour_strong",
]
TRACE_FEATURE_COLUMNS = [
    "rendered_path_length_arcmin",
    "rendered_path_speed_arcmin_s",
    "rendered_rms_radius_arcmin",
    "rendered_bcea68_arcmin2",
    "rendered_cov_anisotropy",
    "rendered_cov_axis_ratio",
    "rendered_cov_orientation_deg",
    "rendered_speed_p95_arcmin_s",
    "rendered_diffusion_constant_arcmin2_s",
    "rendered_position_autocorr_lag1",
    "rendered_velocity_autocorr_lag1",
    "rendered_n_microsaccade_events",
    "rendered_fraction_microsaccade_samples",
    "rendered_peak_microsaccade_speed_dps",
]
TRACE_BANK_METRIC_SUMMARY_SPECS = (
    ("rendered_path_length_arcmin", "path length", "arcmin"),
    ("rendered_path_speed_arcmin_s", "mean speed", "arcmin/s"),
    ("rendered_rms_radius_arcmin", "RMS radius", "arcmin"),
    ("rendered_bcea68_arcmin2", "BCEA68", "arcmin^2"),
    ("rendered_cov_anisotropy", "covariance anisotropy", "fraction"),
    ("rendered_n_microsaccade_events", "microsaccade events", "count"),
    ("rendered_fraction_microsaccade_samples", "microsaccade samples", "fraction"),
)


def progress(message: str) -> None:
    print(f"[fig4-real-trace-matrix] {message}", flush=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "source_row" not in rows.columns:
        rows = rows.copy()
        rows["source_row"] = np.arange(rows.shape[0], dtype=int)
    return rows


def circular_axis_delta_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    return 0.5 * np.degrees(np.angle(np.exp(2j * np.radians(np.asarray(a_deg) - np.asarray(b_deg)))))


def add_derived_image_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if {"image_gradient_energy", "image_orientation_coherence"}.issubset(out.columns):
        gradient = pd.to_numeric(out["image_gradient_energy"], errors="coerce").astype(float)
        coherence = pd.to_numeric(out["image_orientation_coherence"], errors="coerce").astype(float)
        out["image_oriented_gradient_energy"] = gradient * np.maximum(coherence, 0.0)
        out["image_multi_orientation_energy"] = gradient * np.maximum(1.0 - coherence, 0.0)
    required = {
        "image_power_8plus_cpd_fraction",
        "image_patch_std",
        "image_spectrum_anisotropy",
        "image_spectrum_orientation_deg",
        "image_edge_axis_deg",
    }
    if required.issubset(out.columns):
        abs8 = (
            pd.to_numeric(out["image_power_8plus_cpd_fraction"], errors="coerce").astype(float)
            * pd.to_numeric(out["image_patch_std"], errors="coerce").astype(float)
            * pd.to_numeric(out["image_patch_std"], errors="coerce").astype(float)
        )
        spectrum_contour_axis = pd.to_numeric(out["image_spectrum_orientation_deg"], errors="coerce").to_numpy(float) + 90.0
        edge_axis = pd.to_numeric(out["image_edge_axis_deg"], errors="coerce").to_numpy(float)
        agreement = np.cos(2.0 * np.radians(circular_axis_delta_deg(edge_axis, spectrum_contour_axis)))
        out["image_edge_spectrum_contour_axis_agreement"] = agreement
        out["image_oriented_8plus_power_proxy"] = (
            abs8
            * np.maximum(pd.to_numeric(out["image_spectrum_anisotropy"], errors="coerce").astype(float), 0.0)
            * np.maximum(agreement, 0.0)
        )
    return out


def image_candidate_rows(
    rows: pd.DataFrame,
    *,
    contrast_quantile: float,
    n_timepoints: int,
    min_orientation_coherence: float,
    min_drift_anisotropy: float,
) -> pd.DataFrame:
    work = add_derived_image_features(rows)
    if "image_feature_ok" in work.columns:
        work = work[work["image_feature_ok"].astype(bool)].copy()
    if "n_samples" in work.columns:
        work = work[pd.to_numeric(work["n_samples"], errors="coerce") >= int(n_timepoints)].copy()
    if "image_patch_fraction_inside_image" in work.columns:
        work = work[pd.to_numeric(work["image_patch_fraction_inside_image"], errors="coerce") >= 0.99].copy()
    if "image_patch_rms_contrast" in work.columns and work.shape[0]:
        contrast = pd.to_numeric(work["image_patch_rms_contrast"], errors="coerce")
        threshold = float(contrast.quantile(float(contrast_quantile)))
        work = work[contrast >= threshold].copy()
    if float(min_orientation_coherence) > 0.0:
        if "image_orientation_coherence" not in work.columns:
            raise ValueError("--image-min-orientation-coherence requires image_orientation_coherence.")
        coherence = pd.to_numeric(work["image_orientation_coherence"], errors="coerce")
        work = work[coherence >= float(min_orientation_coherence)].copy()
    if float(min_drift_anisotropy) > 0.0:
        if "anisotropy" not in work.columns:
            raise ValueError("--image-min-drift-anisotropy requires anisotropy.")
        anisotropy = pd.to_numeric(work["anisotropy"], errors="coerce")
        work = work[anisotropy >= float(min_drift_anisotropy)].copy()
    return work.drop_duplicates("source_row").reset_index(drop=True)


def sample_rows_random(work: pd.DataFrame, n_rows: int, *, rng: np.random.Generator) -> pd.DataFrame:
    if work.shape[0] < int(n_rows):
        raise ValueError(f"Requested {n_rows} rows, but only {work.shape[0]} are available.")
    indices = rng.choice(work.index.to_numpy(), size=int(n_rows), replace=False)
    return work.loc[indices].copy().reset_index(drop=True)


def sample_image_rows(
    work: pd.DataFrame,
    n_rows: int,
    *,
    rng: np.random.Generator,
    min_strong_contour_images: int,
    strong_contour_orientation_coherence_min: float,
) -> pd.DataFrame:
    min_strong_contour_images = int(min_strong_contour_images)
    if min_strong_contour_images <= 0:
        return sample_rows_random(work, n_rows, rng=rng)
    if min_strong_contour_images > int(n_rows):
        raise ValueError("--min-strong-contour-images cannot exceed --n-images.")
    if "image_orientation_coherence" not in work.columns:
        raise ValueError("--min-strong-contour-images requires image_orientation_coherence.")
    coherence = pd.to_numeric(work["image_orientation_coherence"], errors="coerce")
    strong = work[coherence >= float(strong_contour_orientation_coherence_min)].copy()
    strong_selected = sample_rows_random(strong, min_strong_contour_images, rng=rng)
    selected_source_rows = set(strong_selected["source_row"].astype(int).to_list())
    remainder_pool = work[~work["source_row"].astype(int).isin(selected_source_rows)].copy()
    remainder = sample_rows_random(remainder_pool, int(n_rows) - min_strong_contour_images, rng=rng)
    selected = pd.concat([strong_selected, remainder], ignore_index=True)
    return selected.iloc[rng.permutation(np.arange(selected.shape[0]))].reset_index(drop=True)


def annotate_selected_image_flags(images: pd.DataFrame, *, reliable_min: float, strong_min: float) -> pd.DataFrame:
    out = images.copy()
    if "image_orientation_coherence" in out.columns:
        coherence = pd.to_numeric(out["image_orientation_coherence"], errors="coerce")
        out["image_contour_reliable"] = coherence >= float(reliable_min)
        out["image_contour_strong"] = coherence >= float(strong_min)
    return out


def trace_hash(trace: np.ndarray) -> str:
    arr = np.asarray(trace, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(arr).view(np.uint8)).hexdigest()


def trace_rms(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    centered = arr - np.nanmean(arr, axis=0, keepdims=True)
    return float(np.sqrt(np.nanmean(np.sum(centered * centered, axis=1))))


def path_length(trace: np.ndarray) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.nansum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def lag1_autocorr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 3:
        return float("nan")
    x = arr[:-1].reshape(arr.shape[0] - 1, -1)
    y = arr[1:].reshape(arr.shape[0] - 1, -1)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - np.nanmean(y, axis=0, keepdims=True)
    denom = float(np.sqrt(np.nansum(x * x) * np.nansum(y * y)))
    if denom <= 0.0:
        return float("nan")
    return float(np.nansum(x * y) / denom)


def trace_covariance_shape(trace: np.ndarray) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return {
            "cov_xx_deg2": float("nan"),
            "cov_xy_deg2": float("nan"),
            "cov_yy_deg2": float("nan"),
            "cov_major_var_deg2": float("nan"),
            "cov_minor_var_deg2": float("nan"),
            "cov_axis_ratio": float("nan"),
            "cov_orientation_deg": float("nan"),
        }
    centered = arr - np.nanmean(arr, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    major = max(float(vals[0]), 0.0)
    minor = max(float(vals[1]), 0.0)
    axis_ratio = float(np.sqrt(major / max(minor, EPS))) if major > 0 else float("nan")
    orientation = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return {
        "cov_xx_deg2": float(cov[0, 0]),
        "cov_xy_deg2": float(cov[0, 1]),
        "cov_yy_deg2": float(cov[1, 1]),
        "cov_major_var_deg2": major,
        "cov_minor_var_deg2": minor,
        "cov_axis_ratio": axis_ratio,
        "cov_orientation_deg": orientation,
    }


def trace_covariance_anisotropy(trace: np.ndarray) -> float:
    shape = trace_covariance_shape(trace)
    major = float(shape["cov_major_var_deg2"])
    minor = float(shape["cov_minor_var_deg2"])
    denom = major + minor
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float((major - minor) / denom)


def bcea68_arcmin2(trace: np.ndarray) -> float:
    shape = trace_covariance_shape(trace)
    major = max(float(shape["cov_major_var_deg2"]), 0.0)
    minor = max(float(shape["cov_minor_var_deg2"]), 0.0)
    if not math.isfinite(major) or not math.isfinite(minor):
        return float("nan")
    return float(2.279 * math.pi * math.sqrt(major * minor) * 3600.0)


def speed_threshold_mad(trace: np.ndarray, *, dt: float, z: float) -> float:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] < 2:
        return float("inf")
    speed = np.linalg.norm(np.diff(arr, axis=0, prepend=arr[:1]), axis=1) / float(dt)
    speed = speed[np.isfinite(speed)]
    if speed.size < 3:
        return float("inf")
    med = float(np.median(speed))
    mad = float(np.median(np.abs(speed - med)))
    return med + float(z) * 1.4826 * mad


def microsaccade_stats(
    trace: np.ndarray,
    *,
    dt: float,
    threshold_dps: float | None,
    threshold_z: float,
    pad_frames: int,
) -> dict[str, Any]:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.shape[0] == 0:
        return {
            "microsaccade_threshold_dps": float("nan"),
            "n_microsaccade_events": 0,
            "fraction_microsaccade_samples": 0.0,
            "peak_microsaccade_speed_dps": 0.0,
            "microsaccade_event_mask": np.zeros((0,), dtype=bool),
        }
    threshold = float(threshold_dps) if threshold_dps is not None else speed_threshold_mad(arr, dt=dt, z=threshold_z)
    speed = np.linalg.norm(np.diff(arr, axis=0, prepend=arr[:1]), axis=1) / float(dt)
    mask = np.isfinite(speed) & (speed > threshold)
    if int(pad_frames) > 0 and np.any(mask):
        padded = mask.copy()
        for idx in np.flatnonzero(mask):
            lo = max(0, int(idx) - int(pad_frames))
            hi = min(mask.size, int(idx) + int(pad_frames) + 1)
            padded[lo:hi] = True
        mask = padded
    events = 0
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        events += 1
        while i < mask.size and mask[i]:
            i += 1
    return {
        "microsaccade_threshold_dps": threshold,
        "n_microsaccade_events": int(events),
        "fraction_microsaccade_samples": float(np.mean(mask)),
        "peak_microsaccade_speed_dps": float(np.nanmax(speed)) if speed.size else 0.0,
        "microsaccade_event_mask": mask,
    }


def trace_scale_metrics(trace: np.ndarray, *, dt: float, prefix: str) -> dict[str, float]:
    arr = np.asarray(trace, dtype=np.float64)
    centered = arr - np.nanmean(arr, axis=0, keepdims=True)
    velocity = np.diff(arr, axis=0, prepend=arr[:1]) / float(dt)
    speed = np.linalg.norm(velocity, axis=1)
    shape = trace_covariance_shape(arr)
    path_deg = path_length(arr)
    duration_s = max((arr.shape[0] - 1) * float(dt), float(dt))
    out = {
        f"{prefix}path_length_deg": path_deg,
        f"{prefix}path_length_arcmin": path_deg * 60.0,
        f"{prefix}path_speed_deg_s": path_deg / duration_s,
        f"{prefix}path_speed_arcmin_s": path_deg * 60.0 / duration_s,
        f"{prefix}rms_radius_deg": trace_rms(arr),
        f"{prefix}rms_radius_arcmin": trace_rms(arr) * 60.0,
        f"{prefix}max_radius_deg": float(np.nanmax(np.linalg.norm(centered, axis=1))) if centered.size else 0.0,
        f"{prefix}speed_p95_deg_s": float(np.nanpercentile(speed, 95)) if speed.size else 0.0,
        f"{prefix}speed_p95_arcmin_s": float(np.nanpercentile(speed * 60.0, 95)) if speed.size else 0.0,
        f"{prefix}diffusion_constant_deg2_s": float(np.nanmean(np.sum(np.diff(arr, axis=0) ** 2, axis=1)) / (4.0 * float(dt)))
        if arr.shape[0] > 1
        else 0.0,
        f"{prefix}position_autocorr_lag1": lag1_autocorr(arr),
        f"{prefix}velocity_autocorr_lag1": lag1_autocorr(velocity),
        f"{prefix}bcea68_arcmin2": bcea68_arcmin2(arr),
        f"{prefix}cov_anisotropy": trace_covariance_anisotropy(arr),
        f"{prefix}cov_axis_ratio": float(shape["cov_axis_ratio"]),
        f"{prefix}cov_orientation_deg": float(shape["cov_orientation_deg"]),
    }
    out[f"{prefix}diffusion_constant_arcmin2_s"] = out[f"{prefix}diffusion_constant_deg2_s"] * 3600.0
    return out


def trace_metric_value(item: dict[str, Any], metric: str) -> float:
    candidates = [str(metric)]
    if not str(metric).startswith("rendered_"):
        candidates.append(f"rendered_{metric}")
    if not str(metric).startswith("source_"):
        candidates.append(f"source_{metric}")
    for key in candidates:
        if key not in item:
            continue
        try:
            value = float(item[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def trace_bank_metric_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in item.items():
        if key == "trace":
            continue
        if isinstance(value, (int, np.integer)):
            payload[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            payload[key] = float(value)
    return payload


def trace_bank_metadata_row(item: dict[str, Any], idx: int, *, n_timepoints: int, scale_metric: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trace_bank_index": int(idx),
        "source_row": int(item["source_row"]),
        "session": str(item["session"]),
        "trial_idx": int(item.get("trial_idx", -1)),
        "global_start": int(item["global_start"]),
        "global_stop": int(item["global_stop"]),
        "source_window_global_start": int(item.get("source_window_global_start", item["global_start"])),
        "source_window_global_stop": int(item.get("source_window_global_stop", item["global_stop"])),
        "snippet_global_start": int(item.get("snippet_global_start", item["global_start"])),
        "snippet_global_stop": int(item.get("snippet_global_stop", item["global_stop"])),
        "snippet_n_samples": int(item.get("snippet_n_samples", int(n_timepoints))),
        "snippet_duration_s": float(item.get("snippet_duration_s", np.nan)),
        "trace_hash": trace_hash(item["trace"]),
        "scale_metric": str(scale_metric),
        "scale_metric_value": trace_metric_value(item, str(scale_metric)),
    }
    row.update(trace_bank_metric_payload(item))
    return row


def trace_bank_metric_summary_rows(trace_bank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace_bank_rows:
        return []
    ms = np.asarray([int(row.get("rendered_n_microsaccade_events", 0)) > 0 for row in trace_bank_rows], dtype=bool)
    row_groups = [
        ("all", np.ones(len(trace_bank_rows), dtype=bool)),
        ("no_detected_microsaccade", ~ms),
        ("with_detected_microsaccade", ms),
    ]
    out: list[dict[str, Any]] = []
    for metric_key, label, unit in TRACE_BANK_METRIC_SUMMARY_SPECS:
        all_values = np.asarray([trace_metric_value(row, metric_key) for row in trace_bank_rows], dtype=np.float64)
        for group_label, group_mask in row_groups:
            values = all_values[group_mask]
            values = values[np.isfinite(values)]
            row: dict[str, Any] = {
                "group": group_label,
                "metric": metric_key,
                "label": label,
                "unit": unit,
                "n_rows": int(np.count_nonzero(group_mask)),
                "finite_n": int(values.size),
            }
            if values.size:
                row.update(
                    {
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
                        "min": float(np.nanmin(values)),
                        "q25": float(np.nanquantile(values, 0.25)),
                        "median": float(np.nanmedian(values)),
                        "q75": float(np.nanquantile(values, 0.75)),
                        "max": float(np.nanmax(values)),
                    }
                )
            out.append(row)
    return out


def load_backimage_eyepos_by_session(sessions: list[str]) -> dict[str, np.ndarray]:
    from DataYatesV1 import get_session
    from fixation_stats.extraction import _as_numpy, _load_dict_dataset

    out: dict[str, np.ndarray] = {}
    for name in sorted(set(str(v) for v in sessions)):
        subject, date = name.split("_", 1)
        session = get_session(subject, date)
        dset_path = Path(session.sess_dir) / "datasets" / "backimage.dset"
        dset = _load_dict_dataset(dset_path)
        out[name] = _as_numpy(dset["eyepos"]).astype(np.float64)
    return out


def build_native_snippet_trace_bank(
    rows: pd.DataFrame,
    eyepos_by_session: dict[str, np.ndarray],
    n_timepoints: int,
    *,
    dt: float,
    microsaccade_speed_threshold_dps: float | None,
    microsaccade_threshold_z: float,
    microsaccade_pad_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    n_short = 0
    n_bad = 0
    n_timepoints = int(n_timepoints)
    if n_timepoints < 2:
        raise ValueError("--n-timepoints must be at least 2.")

    for _, row in rows.iterrows():
        session = str(row["session"])
        eyepos = np.asarray(eyepos_by_session[session], dtype=np.float64)
        window_start = max(0, min(int(row["global_start"]), int(eyepos.shape[0])))
        window_stop = max(0, min(int(row["global_stop"]), int(eyepos.shape[0])))
        n_available = int(window_stop - window_start)
        if n_available < n_timepoints:
            n_short += 1
            continue
        snippet_offset = int((n_available - n_timepoints) // 2)
        snippet_start = int(window_start + snippet_offset)
        snippet_stop = int(snippet_start + n_timepoints)
        raw = np.asarray(eyepos[snippet_start:snippet_stop], dtype=np.float64)
        if raw.ndim != 2 or raw.shape != (n_timepoints, 2):
            n_bad += 1
            continue
        trace = raw.copy()
        finite = np.isfinite(trace).all(axis=1)
        if not np.all(finite):
            good = np.flatnonzero(finite)
            if good.size == 0:
                trace = np.zeros_like(trace)
            else:
                bad = np.flatnonzero(~finite)
                for dim in range(2):
                    trace[bad, dim] = np.interp(bad, good, trace[good, dim])
        trace -= np.mean(trace, axis=0, keepdims=True)
        trace = trace.astype(np.float32)
        ms = microsaccade_stats(
            trace,
            dt=float(dt),
            threshold_dps=microsaccade_speed_threshold_dps,
            threshold_z=float(microsaccade_threshold_z),
            pad_frames=int(microsaccade_pad_frames),
        )
        metrics = {
            **trace_scale_metrics(trace, dt=float(dt), prefix="source_"),
            **trace_scale_metrics(trace, dt=float(dt), prefix="rendered_"),
        }
        snippet_duration_s = float((n_timepoints - 1) * float(dt))
        source_window_duration_s = float(row.get("duration_s", np.nan))
        if not math.isfinite(source_window_duration_s):
            source_window_duration_s = float(n_available - 1) * float(dt)
        item: dict[str, Any] = {
            "source_row": int(row["source_row"]),
            "session": session,
            "trial_idx": int(row.get("trial_idx", -1)),
            "global_start": int(snippet_start),
            "global_stop": int(snippet_stop),
            "source_window_global_start": int(window_start),
            "source_window_global_stop": int(window_stop),
            "snippet_global_start": int(snippet_start),
            "snippet_global_stop": int(snippet_stop),
            "snippet_n_samples": int(n_timepoints),
            "snippet_duration_s": snippet_duration_s,
            "source_window_n_samples": int(n_available),
            "source_window_duration_s": source_window_duration_s,
            "mean_x_deg": float(np.nanmean(raw[:, 0])),
            "mean_y_deg": float(np.nanmean(raw[:, 1])),
            "trace": trace,
            "observed_rms_deg": trace_rms(trace),
            "source_trace_observed_rms_deg": trace_rms(trace),
            "path_length_deg": path_length(trace),
            "duration_s": snippet_duration_s,
            "lag1_autocorr": lag1_autocorr(trace),
            "covariance_shape": trace_covariance_shape(trace),
            "trace_cov_anisotropy": trace_covariance_anisotropy(trace),
            "source_trace_cov_anisotropy": trace_covariance_anisotropy(trace),
            "source_anisotropy": trace_covariance_anisotropy(trace),
            "trace_bank_snippet_policy": "center_crop_native_n_timepoints",
        }
        item.update(metrics)
        item.update(
            {
                "source_microsaccade_threshold_dps": float(ms["microsaccade_threshold_dps"]),
                "source_n_microsaccade_events": int(ms["n_microsaccade_events"]),
                "source_fraction_microsaccade_samples": float(ms["fraction_microsaccade_samples"]),
                "source_peak_microsaccade_speed_dps": float(ms["peak_microsaccade_speed_dps"]),
                "rendered_microsaccade_threshold_dps": float(ms["microsaccade_threshold_dps"]),
                "rendered_n_microsaccade_events": int(ms["n_microsaccade_events"]),
                "rendered_fraction_microsaccade_samples": float(ms["fraction_microsaccade_samples"]),
                "rendered_peak_microsaccade_speed_dps": float(ms["peak_microsaccade_speed_dps"]),
                "microsaccade_threshold_dps": float(ms["microsaccade_threshold_dps"]),
                "n_microsaccade_events": int(ms["n_microsaccade_events"]),
                "fraction_microsaccade_samples": float(ms["fraction_microsaccade_samples"]),
                "peak_microsaccade_speed_dps": float(ms["peak_microsaccade_speed_dps"]),
            }
        )
        bank.append(item)

    meta = {
        "trace_bank_snippet_policy": "center_crop_native_n_timepoints",
        "trace_bank_native_dt_s": float(dt),
        "trace_bank_native_snippet_n_timepoints": int(n_timepoints),
        "n_trace_bank_source_windows_skipped_short": int(n_short),
        "n_trace_bank_source_windows_skipped_bad_shape": int(n_bad),
    }
    return bank, meta


def microsaccade_event_count(item: dict[str, Any]) -> int:
    for key in ("rendered_n_microsaccade_events", "n_microsaccade_events", "source_n_microsaccade_events"):
        if key not in item:
            continue
        value = pd.to_numeric(pd.Series([item[key]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return max(0, int(value))
    return 0


def _sample_trace_items_unstratified(
    items: list[dict[str, Any]],
    n_traces: int,
    *,
    metric: str,
    sampling: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if len(items) < int(n_traces):
        raise ValueError(f"Requested {n_traces} traces, but only {len(items)} are available.")
    if str(sampling) == "random":
        indices = rng.choice(np.arange(len(items)), size=int(n_traces), replace=False)
        return [items[int(idx)] for idx in indices]
    values = np.asarray([float(item.get(metric, np.nan)) for item in items], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < int(n_traces):
        raise ValueError(f"Metric {metric!r} has only {finite.size} finite values for quantile sampling.")
    order = finite[np.argsort(values[finite], kind="mergesort")]
    chunks = np.array_split(order, int(n_traces))
    return [items[int(rng.choice(chunk))] for chunk in chunks]


def sample_trace_items(
    items: list[dict[str, Any]],
    n_traces: int,
    *,
    metric: str,
    sampling: str,
    rng: np.random.Generator,
    min_microsaccade_traces: int,
) -> list[dict[str, Any]]:
    min_microsaccade_traces = int(min_microsaccade_traces)
    if min_microsaccade_traces <= 0:
        return _sample_trace_items_unstratified(items, n_traces, metric=metric, sampling=sampling, rng=rng)
    if min_microsaccade_traces > int(n_traces):
        raise ValueError("--min-microsaccade-traces cannot exceed --n-traces.")
    microsaccade_items = [item for item in items if microsaccade_event_count(item) > 0]
    drift_items = [item for item in items if microsaccade_event_count(item) <= 0]
    selected = _sample_trace_items_unstratified(
        microsaccade_items,
        min_microsaccade_traces,
        metric=metric,
        sampling=sampling,
        rng=rng,
    ) + _sample_trace_items_unstratified(
        drift_items,
        int(n_traces) - min_microsaccade_traces,
        metric=metric,
        sampling=sampling,
        rng=rng,
    )
    return sorted(selected, key=lambda item: float(item.get(metric, np.inf)))


def extract_patch(
    row: pd.Series,
    *,
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]],
    patch_size_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from fixation_stats.backimage_canvas import _backimage_canvas, _clip_patch
    from fixation_stats.image_features import gaze_deg_to_screen_px

    key = (str(row["session"]), int(row["trial_idx"]))
    if key not in canvas_cache:
        canvas_cache[key] = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    canvas, ppd, screen_shape = canvas_cache[key]
    center_px = gaze_deg_to_screen_px(
        np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]),
        ppd=ppd,
        screen_shape=screen_shape,
    )
    patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), int(patch_size_px))
    return patch, {
        "patch_center_x_px": float(center_px[0]),
        "patch_center_y_px": float(center_px[1]),
        "patch_ppd": float(ppd),
    }


def image_sampling_summary(images: pd.DataFrame, *, n_candidates: int, args: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_images": int(args.n_images),
        "image_contrast_quantile": float(args.image_contrast_quantile),
        "candidate_rows_after_gates": int(n_candidates),
        "image_min_orientation_coherence": float(args.image_min_orientation_coherence),
        "image_min_drift_anisotropy": float(args.image_min_drift_anisotropy),
        "min_strong_contour_images": int(args.min_strong_contour_images),
        "strong_contour_orientation_coherence_min": float(args.strong_contour_orientation_coherence_min),
    }
    if "image_orientation_coherence" in images.columns:
        coherence = pd.to_numeric(images["image_orientation_coherence"], errors="coerce")
        reliable_min = max(0.2, float(args.image_min_orientation_coherence))
        out.update(
            {
                "selected_reliable_contour_images": int((coherence >= reliable_min).sum()),
                "selected_strong_contour_images": int(
                    (coherence >= float(args.strong_contour_orientation_coherence_min)).sum()
                ),
                "selected_orientation_coherence_min": float(coherence.min()),
                "selected_orientation_coherence_median": float(coherence.median()),
                "selected_orientation_coherence_max": float(coherence.max()),
            }
        )
    return out


def write_unit_feature_table(path: Path, unit_rows: list[dict[str, Any]], unit_tuning_csv: Path, n_units: int) -> None:
    if unit_rows:
        base = pd.DataFrame(unit_rows).copy()
        if "unit_index" not in base.columns:
            base.insert(0, "unit_index", np.arange(base.shape[0], dtype=int))
        if "unit_label" not in base.columns:
            base["unit_label"] = [f"u{idx:03d}" for idx in range(base.shape[0])]
    else:
        base = pd.DataFrame(
            {"unit_index": np.arange(int(n_units), dtype=int), "unit_label": [f"u{idx:03d}" for idx in range(int(n_units))]}
        )
    if unit_tuning_csv.exists():
        tuning = pd.read_csv(unit_tuning_csv)
        if "unit_index" in tuning.columns:
            tuning = tuning.drop_duplicates("unit_index").copy()
            base = base.merge(tuning, on="unit_index", how="left", suffixes=("", "_tuning"))
            if "unit_label_tuning" in base.columns:
                base["unit_label"] = base["unit_label_tuning"].fillna(base["unit_label"])
                base = base.drop(columns=["unit_label_tuning"])
    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path, index=False)


def score_matrix(
    *,
    scorer: Any,
    image_rows: pd.DataFrame,
    trace_items: list[dict[str, Any]],
    frame_batch_size: int,
    trace_batch_size: int,
    n_timepoints: int,
    bin_seconds: float,
    patch_size_px: int,
    write_outputs: bool,
    out_dir: Path | None = None,
    patch_loader: Callable[..., tuple[np.ndarray, dict[str, Any]]] = extract_patch,
) -> dict[str, Any]:
    traces = [np.asarray(item["trace"], dtype=np.float32) for item in trace_items]
    n_images = int(image_rows.shape[0])
    n_traces = int(len(traces))
    n_movies = n_images * n_traces
    n_units = int(scorer.n_units)
    ssi_matrix = np.zeros((n_movies, n_units), dtype=np.float32)
    expected_matrix = np.zeros((n_movies, n_units), dtype=np.float32)
    mean_rate_matrix = np.zeros((n_movies, n_units), dtype=np.float32)
    population_ssi = np.zeros((n_movies,), dtype=np.float32)
    movie_rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    started = time.perf_counter()
    for shard_image_ordinal, (_, image_row) in enumerate(image_rows.iterrows()):
        global_image_index = int(image_row["image_index"]) if "image_index" in image_row.index else int(shard_image_ordinal)
        patch, patch_meta = patch_loader(
            image_row,
            canvas_cache=canvas_cache,
            patch_size_px=int(patch_size_px),
        )
        image_ssi, image_expected, image_mean_rate, image_population = scorer.score_traces_for_patch(
            patch,
            traces,
            trace_batch_size=int(trace_batch_size),
            frame_batch_size=int(frame_batch_size),
            n_timepoints=int(n_timepoints),
            bin_seconds=float(bin_seconds),
        )
        for trace_index in range(n_traces):
            matrix_row_index = shard_image_ordinal * n_traces + trace_index
            movie_index = global_image_index * n_traces + trace_index
            ssi_matrix[matrix_row_index] = image_ssi[trace_index]
            expected_matrix[matrix_row_index] = image_expected[trace_index]
            mean_rate_matrix[matrix_row_index] = image_mean_rate[trace_index]
            population_ssi[matrix_row_index] = image_population[trace_index]
            if write_outputs:
                trace_item = trace_items[trace_index]
                row = {
                    "movie_index": int(movie_index),
                    "matrix_row_index": int(matrix_row_index),
                    "image_index": int(global_image_index),
                    "shard_image_ordinal": int(shard_image_ordinal),
                    "trace_index": int(trace_index),
                    "image_source_row": int(image_row["source_row"]),
                    "trace_source_row": int(trace_item["source_row"]),
                    "image_session": str(image_row["session"]),
                    "image_trial_idx": int(image_row["trial_idx"]),
                    "trace_session": str(trace_item["session"]),
                    "trace_trial_idx": int(trace_item.get("trial_idx", -1)),
                    **patch_meta,
                }
                for key in IMAGE_FEATURE_COLUMNS:
                    if key in image_row.index:
                        row[key] = image_row[key]
                for key in TRACE_FEATURE_COLUMNS:
                    if key in trace_item:
                        row[key] = trace_item[key]
                movie_rows.append(row)
        progress(
            f"scored image {shard_image_ordinal + 1}/{n_images} "
            f"(global_image_index={global_image_index}); movies={min((shard_image_ordinal + 1) * n_traces, n_movies)}"
        )
    elapsed = time.perf_counter() - started
    if write_outputs:
        if out_dir is None:
            raise ValueError("out_dir is required when write_outputs=True.")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "ssi_matrix.npy", ssi_matrix)
        np.save(out_dir / "expected_spikes_matrix.npy", expected_matrix)
        np.save(out_dir / "mean_rate_matrix.npy", mean_rate_matrix)
        np.save(out_dir / "population_ssi.npy", population_ssi)
        np.save(out_dir / "trace_xy.npy", np.stack(traces, axis=0).astype(np.float32))
        write_csv(out_dir / "movie_feature_table.csv", movie_rows)
    return {
        "elapsed_s": float(elapsed),
        "n_images": n_images,
        "n_traces": n_traces,
        "n_movies": n_movies,
        "n_units": n_units,
        "movies_per_s": float(n_movies / elapsed) if elapsed > 0.0 else float("nan"),
        "seconds_per_movie": float(elapsed / n_movies) if n_movies > 0 else float("nan"),
    }


def score_stabilized_images(
    *,
    scorer: Any,
    images: pd.DataFrame,
    frame_batch_size: int,
    n_timepoints: int,
    bin_seconds: float,
    patch_size_px: int,
    patch_loader: Callable[..., tuple[np.ndarray, dict[str, Any]]] = extract_patch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    zero_trace = np.zeros((int(n_timepoints), 2), dtype=np.float32)
    n_images = int(images.shape[0])
    n_units = int(scorer.n_units)
    ssi = np.zeros((n_images, n_units), dtype=np.float32)
    expected = np.zeros((n_images, n_units), dtype=np.float32)
    mean_rate = np.zeros((n_images, n_units), dtype=np.float32)
    population = np.zeros((n_images,), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    started = time.perf_counter()
    for baseline_row_index, (_, image_row) in enumerate(images.iterrows()):
        patch, patch_meta = patch_loader(image_row, canvas_cache=canvas_cache, patch_size_px=int(patch_size_px))
        image_ssi, image_expected, image_mean_rate, image_population = scorer.score_traces_for_patch(
            patch,
            [zero_trace],
            trace_batch_size=1,
            frame_batch_size=int(frame_batch_size),
            n_timepoints=int(n_timepoints),
            bin_seconds=float(bin_seconds),
        )
        ssi[baseline_row_index] = image_ssi[0]
        expected[baseline_row_index] = image_expected[0]
        mean_rate[baseline_row_index] = image_mean_rate[0]
        population[baseline_row_index] = image_population[0]
        rows.append(
            {
                "baseline_row_index": int(baseline_row_index),
                "image_index": int(image_row["image_index"]),
                "condition_id": "counterfactual_stabilized_zero_motion",
                "n_timepoints": int(n_timepoints),
                "bin_seconds": float(bin_seconds),
                "zero_trace_path_length_arcmin": 0.0,
                "stabilized_population_ssi": float(image_population[0]),
                "stabilized_total_expected_spikes": float(np.sum(image_expected[0], dtype=np.float64)),
                **patch_meta,
            }
        )
        progress(f"scored stabilized image {baseline_row_index + 1}/{n_images} (image_index={int(image_row['image_index'])})")
    elapsed = time.perf_counter() - started
    return ssi, expected, mean_rate, population, rows, {"elapsed_s": float(elapsed)}
