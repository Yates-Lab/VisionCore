#!/usr/bin/env python3
"""Build a contour-relative SSI schematic with a fig3-style model-input cube."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches, patheffects

try:
    from scipy.ndimage import shift as ndi_shift
except Exception:
    ndi_shift = None

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths
from _fig4_style import INK

# Panel A's model-input cube is drawn by figure 3's panel-A builder, so this is
# a hard cross-figure dependency, not an optional embellishment. It used to be
# wrapped in `except Exception: HAVE_FIG3_CUBE = False`, and
# `add_visual_model_input_cube` then quietly substituted a flat 2D
# `add_stimulus()` -- the figure still rendered, with a different schematic than
# the one it claimed to show. The import is deliberately unguarded: if fig3
# moves or its private helpers are renamed, that must break the build here
# rather than silently change what panel A depicts.
FIG3_DIR = ROOT / "paper" / "fig3"
if str(FIG3_DIR) not in sys.path:
    sys.path.insert(0, str(FIG3_DIR))

from fixation_stats.backimage_canvas import _backimage_canvas, _clip_patch

from generate_fig3a import (  # noqa: E402
    CYAN as FIG3_CYAN,
    TEXT_COLOR as FIG3_TEXT_COLOR,
    SCR_H as FIG3_SCR_H,
    SCR_W as FIG3_SCR_W,
    _draw_lag_cube as draw_fig3_lag_cube,
    _project_screen as draw_fig3_project_screen,
    box_corners_3d as fig3_box_corners_3d,
    screen_corners_3d as fig3_screen_corners_3d,
)


OUT_BASE = _paths.FIG_DIR / "ssi_contour_schematic_revised"
SOURCE_OVERVIEW_OUT_BASE = _paths.FIG_DIR / "ssi_contour_source_window_overview"
SCHEMATIC_RR100_FINAL_MAP_NPZ = _paths.SCHEMATIC_FINAL_MAPS_NPZ
SCHEMATIC_RR100_FINAL_MAP_METRICS_CSV = _paths.SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV
UNIT_MAPS_NPZ = _paths.UNIT_MAPS_NPZ
UNIT_MAPS_SELECTED_PATCH_NPY = _paths.UNIT_MAPS_SELECTED_PATCH_NPY
UNIT_MAPS_SSI_ALL_UNITS_CSV = _paths.UNIT_MAPS_SSI_ALL_UNITS_CSV
SF_GROUP_CSV = _paths.SF_TUNING_UNIT_GROUPS_CSV
ORIENTATION_GROUP_CSV = _paths.UNIT_MAPS_ORIENTATION_GROUPS_CSV
TRACE_COMPONENT_METRICS_CSV = _paths.TRACE_COMPONENT_MOVIE_METRICS_CSV
TRACE_XY_NPY = _paths.TRACE_XY_NPY
NEW_BANK_IMAGE_TABLE = _paths.IMAGE_FEATURE_TABLE_CSV
SCHEMATIC_REAL_TRACE_CENTER40_CSV = _paths.SCHEMATIC_TRACE_CENTER40_CSV
SCHEMATIC_STIMULUS_PAYLOAD_NPZ = _paths.SCHEMATIC_STIMULUS_PAYLOAD_NPZ


RED = "#c51f27"
BLUE = "#1e4ed8"
GRAY = "#5f6368"
READOUT_GREEN = "#1f5e1f"
READOUT_FILL = "#d9ecd9"
CORE_FILL = "#d7d9d8"
CORE_EDGE = "#b7bbb8"
FILTER_COLORS = ["#d8842a", "#d9b36d", "#45a4a7", "#cc3b1d"]
EPS = 1e-12

SMALL_CONDITION = "along1_across0p125"
REFERENCE_CONDITION = "along1_across1"
LARGE_CONDITION = "along1_across3"
STABILIZED_CONDITION = "along1_across0"
RIGHT_PANEL_UNIT_SF = "auto"
RIGHT_PANEL_EXACT_UNIT_INDEX = 38
GRID_CONDITIONS = [SMALL_CONDITION, REFERENCE_CONDITION, LARGE_CONDITION]
GRID_ROW_LABELS = ["0.125x", "1x", "3x"]
GRID_ROW_COLORS = [RED, GRAY, BLUE]
GRID_FRAMES = [0, 13, 26, 39]
PREFERRED_REAL_UNITS = {"high": 66, "medium": 47, "low": 33}
TARGET_ORIENTATION_GROUP = "contour_biased"
LOW_TRACE_QUANTILE = 0.05
HIGH_TRACE_QUANTILE = 0.95
SCHEMATIC_NEW_BANK_IMAGE_INDEX = 86
SCHEMATIC_PATCH_SIZE_PX = 151
MODEL_SOURCE_PATCH_SIZE_PX = 540
MODEL_PPD = 37.50476617
SCHEMATIC_LARGE_TRACE_INDEX = 946
ACTIVATION_CMAP = plt.get_cmap("bone_r", 1024)
SPATIAL_ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "ssi_spatial_activation",
    ["#effafa", "#c8e2de", "#93acb0", "#5d607d", "#23223a"],
    N=1024,
)
ACTIVATION_INTERPOLATION = "lanczos"
PANEL_B_ACTIVATION_MAP_STYLE = "mean_centered_diverging"
PANEL_B_SHARP_GALLERY_PERCENTILES = (1.0, 99.5)
SHOW_ACTIVATION_MAP_SECTION = False
USE_SYNTHETIC_LEFT_SIDE = True
RASTER_EXPORT_DPI = 220
VECTOR_EXPORT_DPI = 300
MODEL_INPUT_CUBE_W = 1.55
MODEL_INPUT_CUBE_H = 1.18
MODEL_INPUT_CUBE_D = 1.62
MODEL_INPUT_CUBE_YAW_DEG = -40.0
MODEL_INPUT_CUBE_PITCH_DEG = 0.0
MODEL_INPUT_CUBE_ROLL_DEG = 0.0
MODEL_INPUT_N_LAGS = 32
SYNTHETIC_TRACE_FRAMES = 160
SYNTHETIC_TRACE_CYCLES = 2.35
SYNTHETIC_SMALL_ACROSS_PX = 5.5
SYNTHETIC_LARGE_SCALE = 3.0
SYNTHETIC_GABOR_SIZE_PX = 65
PANEL_A_SMALL_TRACE_INDEX = 127
PANEL_A_LARGE_TRACE_INDEX = 833
PANEL_A_SMALL_TRACE_ROTATION_DEG = 0.0
PANEL_A_LARGE_TRACE_ROTATION_DEG = 90.0
PANEL_A_HIGH_SF_CPD = 8.0
PANEL_A_LOW_SF_CPD = 2.0
PANEL_A_FRAME_RATE_HZ = 120.0
PANEL_A_INTEGRATION_FRAMES = 3


def smooth2d(a, n=4):
    for _ in range(n):
        a = (
            a
            + np.roll(a, 1, 0)
            + np.roll(a, -1, 0)
            + np.roll(a, 1, 1)
            + np.roll(a, -1, 1)
        ) / 5.0
    return a


def normalize_image(image):
    arr = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def condition_index(payload, condition_id):
    ids = [str(x) for x in np.asarray(payload["condition_id"]).astype(str)]
    try:
        return ids.index(condition_id)
    except ValueError as exc:
        raise KeyError(f"Missing condition {condition_id!r}; available={ids}") from exc


def orientation_axis_180(angle_deg):
    return float(float(angle_deg) % 180.0)


def axial_angle_distance_deg(angles_deg, target_deg):
    return np.abs((np.asarray(angles_deg, dtype=np.float64) - float(target_deg) + 90.0) % 180.0 - 90.0)


def gaze_axis_deg_to_image_axis_deg(axis_deg):
    """Match Figure 4: gaze-space +y-up angles become image-array +row-down angles."""
    return float(-float(axis_deg))


def axis_vector_image(axis_deg):
    theta = np.deg2rad(float(axis_deg))
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)


def instantaneous_bits(movie):
    y = np.maximum(np.asarray(movie, dtype=np.float64), 0.0)
    if y.ndim != 3:
        raise ValueError(f"Expected a single-unit movie with shape (T,H,W), got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    rbar = np.mean(flat, axis=1)
    gain = flat / (rbar[:, None] + EPS)
    return np.mean(gain * np.log2(gain + EPS), axis=1)


def unit_mean_trace(payload, unit_index, condition_id):
    cond_idx = condition_index(payload, condition_id)
    movie = np.asarray(payload["maps"][cond_idx, :, int(unit_index)], dtype=np.float64)
    return np.mean(np.maximum(movie, 0.0), axis=(-2, -1))


def unit_ssi_trace(payload, unit_index, condition_id):
    cond_idx = condition_index(payload, condition_id)
    movie = np.asarray(payload["maps"][cond_idx, :, int(unit_index)], dtype=np.float64)
    return instantaneous_bits(movie)


def unit_frame_maps(payload, unit_index, condition_ids=GRID_CONDITIONS, frame_indices=GRID_FRAMES):
    rows = []
    for condition_id in condition_ids:
        cond_idx = condition_index(payload, condition_id)
        rows.append([np.asarray(payload["maps"][cond_idx, frame, int(unit_index)], dtype=np.float64) for frame in frame_indices])
    return rows


def unit_map_limits(payload, unit_index):
    maps = []
    for condition_id in GRID_CONDITIONS:
        cond_idx = condition_index(payload, condition_id)
        maps.append(np.asarray(payload["maps"][cond_idx, :, int(unit_index)], dtype=np.float64))
    values = np.maximum(np.concatenate([m.reshape(-1) for m in maps]), 0.0)
    vmax = float(np.nanpercentile(values, 99.2))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(values)) if values.size else 1.0
    return 0.0, max(vmax, EPS)


def example_frame(payload, unit_index):
    ssi = unit_ssi_trace(payload, unit_index, REFERENCE_CONDITION)
    if np.all(~np.isfinite(ssi)):
        return 0
    return int(np.nanargmax(ssi))


def projected_bank_trace_arcmin(trace_xy, contour_axis_deg, component="across"):
    trace = np.asarray(trace_xy, dtype=np.float64)
    centered = trace - np.nanmean(trace, axis=0, keepdims=True)
    theta = np.deg2rad(float(contour_axis_deg))
    along_u = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    across_u = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    unit = across_u if component == "across" else along_u
    return (centered @ unit) * 60.0


def load_schematic_center40_trace():
    if not SCHEMATIC_REAL_TRACE_CENTER40_CSV.exists():
        return None
    try:
        trace_df = pd.read_csv(SCHEMATIC_REAL_TRACE_CENTER40_CSV)
    except Exception as exc:
        raise ValueError(f"Could not read schematic stimulus payload cache {cache_path}: {exc}") from exc
    if trace_df.empty or not {"x_centered_deg", "y_centered_deg"}.issubset(trace_df.columns):
        return None
    if "sample_idx" in trace_df.columns:
        trace_df = trace_df.sort_values("sample_idx")
    trace = trace_df[["x_centered_deg", "y_centered_deg"]].to_numpy(dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 2:
        return None
    if not np.all(np.isfinite(trace)):
        return None
    return trace.astype(np.float32)


def lag_trace(trace_xy, n_lags=MODEL_INPUT_N_LAGS):
    trace = np.asarray(trace_xy, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] == 0:
        return None
    if trace.shape[0] >= int(n_lags):
        out = trace[-int(n_lags) :]
    else:
        pad = np.repeat(trace[:1], int(n_lags) - trace.shape[0], axis=0)
        out = np.vstack([pad, trace])
    return out.astype(np.float32)


def endpoint_stabilized_trace(trace_xy):
    trace = np.asarray(trace_xy, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] == 0:
        return None
    return np.repeat(trace[-1:, :], trace.shape[0], axis=0).astype(np.float32)


def load_schematic_rr100_final_maps():
    if not SCHEMATIC_RR100_FINAL_MAP_NPZ.exists():
        return None
    try:
        with np.load(SCHEMATIC_RR100_FINAL_MAP_NPZ, allow_pickle=False) as data:
            payload = {key: data[key].copy() for key in data.files}
    except Exception:
        return None
    if SCHEMATIC_RR100_FINAL_MAP_METRICS_CSV.exists():
        try:
            payload["unit_metrics"] = pd.read_csv(SCHEMATIC_RR100_FINAL_MAP_METRICS_CSV)
        except Exception:
            payload["unit_metrics"] = None
    else:
        payload["unit_metrics"] = None
    return payload


def load_cached_new_bank_stimulus_patch(cache_path=SCHEMATIC_STIMULUS_PAYLOAD_NPZ):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            row = json.loads(str(data["row_json"].item()))
            center = tuple(float(v) for v in data["crop_center_xy"].astype(float).tolist())
            return {
                "patch": data["patch"].astype(np.float64),
                "model_source_patch": data["model_source_patch"].astype(np.float64),
                "canvas": data["canvas"].astype(np.float64),
                "row": row,
                "image_index": int(data["image_index"].item()),
                "source_row": int(data["source_row"].item()),
                "crop_center_xy": center,
                "crop_size_px": int(data["crop_size_px"].item()),
                "model_source_patch_size_px": int(data["model_source_patch_size_px"].item()),
                "contour_axis_deg": float(data["contour_axis_deg"].item()),
                "contour_axis_image_deg": float(data["contour_axis_image_deg"].item()),
                "real_trace_center40": data["real_trace_center40"].astype(np.float32),
                "real_trace_lag32": data["real_trace_lag32"].astype(np.float32),
                "endpoint_stabilized_trace_lag32": data["endpoint_stabilized_trace_lag32"].astype(np.float32),
            }
    except Exception:
        return None


def build_new_bank_stimulus_patch_from_data(image_index=SCHEMATIC_NEW_BANK_IMAGE_INDEX):
    if not NEW_BANK_IMAGE_TABLE.exists():
        return None
    images = pd.read_csv(NEW_BANK_IMAGE_TABLE)
    if images.empty:
        return None

    image_index = int(image_index)
    selected = images[images["image_index"].eq(image_index)].copy()
    if selected.empty:
        selected = images.sort_values("image_oriented_gradient_energy", ascending=False).head(1).copy()
    row = selected.iloc[0]

    canvas, _, _ = _backimage_canvas(str(row["session"]), int(row["trial_idx"]))
    center = (float(row["image_patch_center_x_px"]), float(row["image_patch_center_y_px"]))
    patch = _clip_patch(canvas, center, SCHEMATIC_PATCH_SIZE_PX)
    source_patch = _clip_patch(canvas, center, MODEL_SOURCE_PATCH_SIZE_PX)
    contour_axis_deg = float(row["image_edge_axis_deg"])
    contour_axis_image_deg = float(row.get("image_edge_axis_array_deg", -contour_axis_deg))
    center40_trace = load_schematic_center40_trace()
    real_trace_lag32 = lag_trace(center40_trace) if center40_trace is not None else None
    stable_trace_lag32 = endpoint_stabilized_trace(real_trace_lag32) if real_trace_lag32 is not None else None
    return {
        "patch": normalize_image(patch),
        "model_source_patch": normalize_image(source_patch),
        "canvas": normalize_image(canvas),
        "row": row.to_dict(),
        "image_index": int(row["image_index"]),
        "source_row": int(row["source_row"]),
        "crop_center_xy": center,
        "crop_size_px": SCHEMATIC_PATCH_SIZE_PX,
        "model_source_patch_size_px": MODEL_SOURCE_PATCH_SIZE_PX,
        "contour_axis_deg": contour_axis_deg,
        "contour_axis_image_deg": orientation_axis_180(contour_axis_image_deg),
        "real_trace_center40": center40_trace,
        "real_trace_lag32": real_trace_lag32,
        "endpoint_stabilized_trace_lag32": stable_trace_lag32,
    }


def write_new_bank_stimulus_cache(
    out_path=SCHEMATIC_STIMULUS_PAYLOAD_NPZ,
    image_index=SCHEMATIC_NEW_BANK_IMAGE_INDEX,
):
    payload = build_new_bank_stimulus_patch_from_data(image_index=image_index)
    if payload is None:
        raise FileNotFoundError(
            f"could not build schematic stimulus payload from {NEW_BANK_IMAGE_TABLE}"
        )
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = _paths.CACHE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patch=np.asarray(payload["patch"], dtype=np.float32),
        model_source_patch=np.asarray(payload["model_source_patch"], dtype=np.float32),
        canvas=np.asarray(payload["canvas"], dtype=np.float32),
        row_json=np.asarray(json.dumps(payload["row"], sort_keys=True, allow_nan=True)),
        image_index=np.asarray(payload["image_index"], dtype=np.int64),
        source_row=np.asarray(payload["source_row"], dtype=np.int64),
        crop_center_xy=np.asarray(payload["crop_center_xy"], dtype=np.float64),
        crop_size_px=np.asarray(payload["crop_size_px"], dtype=np.int64),
        model_source_patch_size_px=np.asarray(payload["model_source_patch_size_px"], dtype=np.int64),
        contour_axis_deg=np.asarray(payload["contour_axis_deg"], dtype=np.float64),
        contour_axis_image_deg=np.asarray(payload["contour_axis_image_deg"], dtype=np.float64),
        real_trace_center40=np.asarray(payload["real_trace_center40"], dtype=np.float32),
        real_trace_lag32=np.asarray(payload["real_trace_lag32"], dtype=np.float32),
        endpoint_stabilized_trace_lag32=np.asarray(payload["endpoint_stabilized_trace_lag32"], dtype=np.float32),
    )
    return out_path


def load_new_bank_stimulus_patch(image_index=SCHEMATIC_NEW_BANK_IMAGE_INDEX):
    cached = load_cached_new_bank_stimulus_patch()
    if cached is not None:
        return cached
    return build_new_bank_stimulus_patch_from_data(image_index=image_index)


def load_component_sorted_eye_traces(target_contour_axis_deg, image_index=None):
    if not TRACE_COMPONENT_METRICS_CSV.exists() or not TRACE_XY_NPY.exists():
        return None

    cols = [
        "movie_index",
        "image_index",
        "trace_index",
        "image_edge_axis_deg",
        "has_microsaccade",
        "rendered_n_microsaccade_events",
        "rendered_path_length_arcmin",
        "along_path_arcmin",
        "across_path_arcmin",
        "along_rms_arcmin",
        "across_rms_arcmin",
        "across_path_bin",
        "along_path_bin",
    ]
    metrics = pd.read_csv(TRACE_COMPONENT_METRICS_CSV, usecols=cols)
    metrics = metrics[np.isfinite(metrics["across_path_arcmin"])].copy()
    if metrics.empty:
        return None

    selection_mode = "nearest_axis"
    if image_index is not None and metrics["image_index"].eq(int(image_index)).any():
        image_index = int(image_index)
        selection_mode = "requested_image"
    else:
        image_axes = metrics[["image_index", "image_edge_axis_deg"]].drop_duplicates().copy()
        image_axes["axis_distance_deg"] = axial_angle_distance_deg(
            image_axes["image_edge_axis_deg"],
            target_contour_axis_deg,
        )
        image_axes = image_axes.sort_values(["axis_distance_deg", "image_index"])
        image_index = int(image_axes.iloc[0]["image_index"])
    sub = metrics[metrics["image_index"].eq(image_index)].copy()
    drift_only = sub[~sub["has_microsaccade"].astype(bool)].copy()
    if len(drift_only) >= 2:
        sub = drift_only
    sub = sub.sort_values("across_path_arcmin").reset_index(drop=True)
    if len(sub) < 2:
        return None

    trace_xy = np.load(TRACE_XY_NPY)

    def pack_trace(row, fallback_index):
        trace_index = int(row["trace_index"])
        if trace_index < 0 or trace_index >= trace_xy.shape[0]:
            row = sub.iloc[int(fallback_index)]
            trace_index = int(row["trace_index"])
        trace = trace_xy[trace_index]
        axis_deg = float(row["image_edge_axis_deg"])
        axis_distance = 0.0 if selection_mode == "requested_image" else float(
            axial_angle_distance_deg([axis_deg], target_contour_axis_deg)[0]
        )
        return {
            "movie_index": int(row["movie_index"]),
            "image_index": int(row["image_index"]),
            "trace_index": trace_index,
            "image_edge_axis_deg": axis_deg,
            "axis_distance_deg": axis_distance,
            "has_microsaccade": bool(row["has_microsaccade"]),
            "rendered_n_microsaccade_events": int(row["rendered_n_microsaccade_events"]),
            "rendered_path_length_arcmin": float(row["rendered_path_length_arcmin"]),
            "along_path_arcmin": float(row["along_path_arcmin"]),
            "across_path_arcmin": float(row["across_path_arcmin"]),
            "along_rms_arcmin": float(row["along_rms_arcmin"]),
            "across_rms_arcmin": float(row["across_rms_arcmin"]),
            "across_path_bin": str(row["across_path_bin"]),
            "along_path_bin": str(row["along_path_bin"]),
            "along_trace_arcmin": projected_bank_trace_arcmin(trace, axis_deg, component="along"),
            "across_trace_arcmin": projected_bank_trace_arcmin(trace, axis_deg, component="across"),
        }

    def pick_trace(quantile, fallback_index, preferred_trace_index=None):
        if preferred_trace_index is not None:
            preferred = sub[sub["trace_index"].eq(int(preferred_trace_index))]
            if not preferred.empty:
                return pack_trace(preferred.iloc[0], fallback_index)
        target = float(sub["across_path_arcmin"].quantile(float(quantile)))
        row = sub.iloc[(sub["across_path_arcmin"] - target).abs().argmin()]
        return pack_trace(row, fallback_index)

    low = pick_trace(LOW_TRACE_QUANTILE, 0)
    preferred_large_trace_index = (
        SCHEMATIC_LARGE_TRACE_INDEX
        if image_index == SCHEMATIC_NEW_BANK_IMAGE_INDEX
        else None
    )
    high = pick_trace(HIGH_TRACE_QUANTILE, len(sub) - 1, preferred_trace_index=preferred_large_trace_index)
    if low["trace_index"] == high["trace_index"]:
        low = pick_trace(0.0, 0)
        high = pick_trace(1.0, len(sub) - 1)

    return {
        "selection_rule": (
            f"{selection_mode}; image {image_index}; axis target {target_contour_axis_deg:.2f} deg; "
            f"drift-only traces near q{LOW_TRACE_QUANTILE:.2f}/q{HIGH_TRACE_QUANTILE:.2f} "
            "of across-contour path length"
        ),
        "low": low,
        "high": high,
    }


def load_real_payload():
    """Load panel A's real unit-map payload.

    Returns None only when degraded rendering is explicitly enabled. The
    four-way `exists()` gate this replaces returned None on any missing input,
    which made panel A fall back to a synthetic schematic without a word in the
    output about which of the four was absent.
    """
    for path, what in (
        (UNIT_MAPS_NPZ, "panel A instantaneous unit maps"),
        (UNIT_MAPS_SELECTED_PATCH_NPY, "panel A selected stimulus patch"),
        (SF_GROUP_CSV, "panel A spatial-frequency tuning groups"),
        (ORIENTATION_GROUP_CSV, "panel A orientation tuning groups"),
    ):
        if _paths.require(path, what) is None:
            return None

    cache_path = UNIT_MAPS_NPZ
    patch_path = UNIT_MAPS_SELECTED_PATCH_NPY

    with np.load(cache_path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    payload["patch"] = normalize_image(np.load(patch_path))
    payload["contour_axis_deg"] = float(np.asarray(payload.get("axis_deg", [-10.352312]), dtype=float).reshape(-1)[0])
    payload["contour_axis_image_deg"] = orientation_axis_180(
        gaze_axis_deg_to_image_axis_deg(payload["contour_axis_deg"])
    )
    new_bank_stimulus = load_new_bank_stimulus_patch()
    if new_bank_stimulus is not None:
        payload["patch"] = new_bank_stimulus["patch"]
        payload["contour_axis_deg"] = new_bank_stimulus["contour_axis_deg"]
        payload["contour_axis_image_deg"] = new_bank_stimulus["contour_axis_image_deg"]
        payload["stimulus_image_index"] = new_bank_stimulus["image_index"]
        payload["stimulus_source_row"] = new_bank_stimulus["source_row"]
        payload["stimulus_row"] = new_bank_stimulus["row"]
        payload["stimulus_canvas"] = new_bank_stimulus["canvas"]
        payload["stimulus_crop_center_xy"] = new_bank_stimulus["crop_center_xy"]
        payload["stimulus_crop_size_px"] = new_bank_stimulus["crop_size_px"]
        payload["stimulus_model_source_patch"] = new_bank_stimulus["model_source_patch"]
        payload["stimulus_model_source_patch_size_px"] = new_bank_stimulus["model_source_patch_size_px"]
        payload["stimulus_real_trace_center40"] = new_bank_stimulus["real_trace_center40"]
        payload["stimulus_real_trace_lag32"] = new_bank_stimulus["real_trace_lag32"]
        payload["stimulus_endpoint_stabilized_trace_lag32"] = new_bank_stimulus["endpoint_stabilized_trace_lag32"]

    schematic_final_maps = load_schematic_rr100_final_maps()
    if schematic_final_maps is not None:
        payload["schematic_rr100_final_maps"] = schematic_final_maps.get("final_maps")
        payload["schematic_rr100_final_condition_id"] = schematic_final_maps.get("condition_id")
        payload["schematic_rr100_final_condition_label"] = schematic_final_maps.get("condition_label")
        payload["schematic_rr100_final_condition_traces"] = schematic_final_maps.get("condition_traces")
        payload["schematic_rr100_final_map_unit_metrics"] = schematic_final_maps.get("unit_metrics")

    sf_groups = pd.read_csv(SF_GROUP_CSV)
    orientation_groups = pd.read_csv(ORIENTATION_GROUP_CSV)
    all_ssi = pd.read_csv(UNIT_MAPS_SSI_ALL_UNITS_CSV)
    joined = all_ssi.merge(
        sf_groups[["unit_index", "unit_label", "sf_group", "sf_split_metric", "sf_rank_low_to_high"]],
        on=["unit_index", "unit_label"],
        how="left",
    )
    joined = joined.merge(
        orientation_groups[
            [
                "unit_index",
                "unit_label",
                "orientation_group",
                "orientation_group_label",
                "preferred_orientation_deg",
                "preferred_delta_from_contour_deg",
                "preferred_delta_from_across_deg",
                "orientation_selectivity_index",
                "orientation_probe_contour_minus_across_norm",
            ]
        ],
        on=["unit_index", "unit_label"],
        how="left",
    )
    reference = joined[
        joined["condition_id"].eq(REFERENCE_CONDITION)
        & joined["axis_mode"].eq("across_sweep")
    ].copy()

    rows = []
    group_lookup = {"high": "high_sf", "medium": "middle_sf", "low": "low_sf"}
    label_lookup = {"high": "High-SF unit", "medium": "Medium-SF unit", "low": "Low-SF unit"}
    score_col = "displayed_movie_time_resolved_ssi_bits_per_spike"
    for sf_key, group_name in group_lookup.items():
        group_rows = reference[
            reference["sf_group"].eq(group_name)
            & reference["orientation_group"].eq(TARGET_ORIENTATION_GROUP)
        ].copy()
        preferred = group_rows[group_rows["unit_index"].eq(PREFERRED_REAL_UNITS[sf_key])]
        if preferred.empty:
            preferred = group_rows.sort_values(score_col, ascending=False).head(1)
        if preferred.empty:
            group_rows = reference[reference["sf_group"].eq(group_name)].copy()
            preferred = group_rows.sort_values(score_col, ascending=False).head(1)
        if preferred.empty:
            continue
        row = preferred.iloc[0]
        unit_idx = int(row["unit_index"])
        orientation_label = str(row.get("orientation_group_label", "orientation matched"))
        delta_from_contour = float(row.get("preferred_delta_from_contour_deg", np.nan))
        rows.append(
            {
                "sf": sf_key,
                "label": f"{label_lookup[sf_key]} {row['unit_label']}\n{orientation_label}",
                "unit_label": str(row["unit_label"]),
                "unit_index": unit_idx,
                "sf_cpd": float(row["sf_split_metric"]),
                "score": float(row[score_col]),
                "preferred_orientation_deg": float(row.get("preferred_orientation_deg", np.nan)),
                "delta_from_contour_deg": delta_from_contour,
                "orientation_group": str(row.get("orientation_group", "")),
                "orientation_label": orientation_label,
                "trace_small": unit_mean_trace(payload, unit_idx, SMALL_CONDITION),
                "trace_large": unit_mean_trace(payload, unit_idx, LARGE_CONDITION),
                "ssi_small": unit_ssi_trace(payload, unit_idx, SMALL_CONDITION),
                "ssi_large": unit_ssi_trace(payload, unit_idx, LARGE_CONDITION),
                "frame_maps": unit_frame_maps(payload, unit_idx),
                "map_vlim": unit_map_limits(payload, unit_idx),
                "example_frame": example_frame(payload, unit_idx),
            }
        )
    payload["unit_rows"] = rows
    payload["component_sorted_eye_traces"] = load_component_sorted_eye_traces(
        payload["contour_axis_deg"],
        image_index=payload.get("stimulus_image_index"),
    )
    return payload


def make_stimulus(seed=7, n=260):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    base = smooth2d(rng.normal(size=(n, n)), n=14)
    base = (base - base.min()) / (base.max() - base.min())

    theta = np.deg2rad(24)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    texture = 0.48 + 0.34 * base + 0.09 * np.sin(u / 4.0) + 0.04 * np.sin(u / 1.8)

    # A clean oblique contour plus a slight luminance step across it.
    y_line = 236 - 1.72 * x
    d = y - y_line
    contour = np.exp(-(d / 3.0) ** 2)
    step = 0.14 * np.tanh(d / 8.5)
    img = texture + step + 0.42 * contour
    img = smooth2d(img, n=2)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def add_panel_header(fig, letter, title, x, y):
    fig.text(x, y, letter, fontsize=16, weight="bold", ha="left", va="center")
    fig.text(x + 0.022, y, title, fontsize=15, weight="bold", ha="left", va="center")


def add_small_arrow(fig, x0, x1, y):
    fig.patches.append(
        patches.FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color=GRAY,
        )
    )


def square_height(fig, width):
    return width * fig.get_figwidth() / fig.get_figheight()


def add_axis_arrows(ax, xlabel=True, ylabel=True):
    ax.annotate(
        "",
        xy=(1.0, -0.09),
        xytext=(0.0, -0.09),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK),
    )
    ax.annotate(
        "",
        xy=(-0.09, 1.0),
        xytext=(-0.09, 0.0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK),
    )
    if xlabel:
        ax.text(0.5, -0.18, "spatial test position", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    if ylabel:
        ax.text(-0.18, 0.5, "spatial test position", transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=8)


def shift_image_with_edge(image, dx, dy):
    image = np.asarray(image, dtype=np.float64)
    if ndi_shift is not None:
        return ndi_shift(
            image,
            shift=(float(dy), float(dx)),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    h, w = image.shape[:2]
    dx = int(round(float(dx)))
    dy = int(round(float(dy)))
    pad = max(abs(dx), abs(dy), 2)
    padded = np.pad(image, pad_width=pad, mode="edge")
    r0 = pad - dy
    c0 = pad - dx
    return padded[r0:r0 + h, c0:c0 + w]


def contour_model_input_cube(
    image,
    contour_axis_image_deg,
    n_lags=MODEL_INPUT_N_LAGS,
    *,
    across_motion_scale=1.0,
    along_motion_scale=1.0,
):
    image = normalize_image(image)
    theta = np.deg2rad(float(contour_axis_image_deg))
    tangent = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent /= np.linalg.norm(tangent)

    span_px = max(7.0, 0.055 * min(image.shape[:2]))
    phases = np.linspace(-1.0, 1.0, int(n_lags))
    wobble = 0.20 * np.sin(np.linspace(0.0, 2.0 * np.pi, int(n_lags)))
    frames = []
    for phase, tangential_phase in zip(phases, wobble):
        offset = (
            normal * (phase * span_px * float(across_motion_scale))
            + tangent * (tangential_phase * span_px * float(along_motion_scale))
        )
        frames.append(shift_image_with_edge(image, offset[0], offset[1]))
    return np.stack(frames, axis=0)


def trace_model_input_cube(
    source_image,
    trace_xy,
    *,
    out_size=SCHEMATIC_PATCH_SIZE_PX,
    ppd=MODEL_PPD,
    n_lags=MODEL_INPUT_N_LAGS,
):
    source = normalize_image(source_image)
    trace = lag_trace(trace_xy, n_lags=n_lags)
    if trace is None:
        return None

    h, w = source.shape[:2]
    center_x = float(w // 2)
    center_y = float(h // 2)
    frames = []
    for x_deg, y_deg in trace:
        # Match Ryan's shared model input helper: x/y are flipped internally, so
        # source crop centers move (-eye_y, +eye_x) in image pixel coordinates.
        frame_center = (
            center_x - float(y_deg) * float(ppd),
            center_y + float(x_deg) * float(ppd),
        )
        frames.append(_clip_patch(source, frame_center, int(out_size)))
    return np.stack(frames, axis=0)


def bilinear_quad_point(quad, x, y):
    q = np.asarray(quad, dtype=np.float64)
    return (
        (1 - x) * (1 - y) * q[0]
        + x * (1 - y) * q[1]
        + x * y * q[2]
        + (1 - x) * y * q[3]
    )


def front_face_vector(front_quad, image_vector):
    q = np.asarray(front_quad, dtype=np.float64)
    right = q[1] - q[0]
    up = q[3] - q[0]
    vec = float(image_vector[0]) * right - float(image_vector[1]) * up
    norm = float(np.hypot(vec[0], vec[1]))
    return vec / norm if norm > 0 else vec


def add_contour_motion_overlays_on_cube(ax, cube_p2, contour_axis_image_deg, *, show_labels=True):
    front_quad = cube_p2[:4]
    theta = np.deg2rad(float(contour_axis_image_deg))
    tangent_img = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    normal_img = np.array([-tangent_img[1], tangent_img[0]], dtype=np.float64)
    tangent = front_face_vector(front_quad, tangent_img)
    normal = front_face_vector(front_quad, normal_img)

    center = bilinear_quad_point(front_quad, 0.50, 0.52)
    line_half = 0.37 * MODEL_INPUT_CUBE_W
    line_start = center - tangent * line_half
    line_end = center + tangent * line_half
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color=FIG3_CYAN,
        lw=2.0,
        alpha=0.90,
        solid_capstyle="round",
        zorder=7.0,
    )
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color="white",
        lw=0.55,
        alpha=0.88,
        solid_capstyle="round",
        zorder=7.1,
    )

    red_center = center + tangent * 0.13
    blue_center = center - tangent * 0.14
    red_half = 0.11
    blue_half = 0.33
    for c, half, color, label, offset in (
        (red_center, red_half, RED, "small across-contour\nmotion", tangent * 0.34 - normal * 0.22),
        (blue_center, blue_half, BLUE, "large across-contour\nmotion", -tangent * 0.26 + normal * 0.32),
    ):
        start = c - normal * half
        end = c + normal * half
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.7, mutation_scale=13),
            zorder=7.4,
        )
        if not show_labels:
            continue
        label_pos = c + offset
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            color=color,
            fontsize=6.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.70),
            zorder=8.0,
        )

    gaze = center + tangent * 0.03
    ax.plot(gaze[0], gaze[1], marker="o", ms=3.8, mfc="white", mec=INK, mew=0.9, zorder=8.2)
    return tuple(gaze)


def add_trace_motion_overlay_on_cube(
    ax,
    cube_p2,
    contour_axis_image_deg,
    trace_xy,
    *,
    path_color=BLUE,
    endpoint_color=RED,
):
    front_quad = cube_p2[:4]
    theta = np.deg2rad(float(contour_axis_image_deg))
    tangent_img = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    normal_img = np.array([-tangent_img[1], tangent_img[0]], dtype=np.float64)
    tangent = front_face_vector(front_quad, tangent_img)
    normal = front_face_vector(front_quad, normal_img)

    center = bilinear_quad_point(front_quad, 0.50, 0.52)
    line_half = 0.37 * MODEL_INPUT_CUBE_W
    line_start = center - tangent * line_half
    line_end = center + tangent * line_half
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color=FIG3_CYAN,
        lw=2.0,
        alpha=0.90,
        solid_capstyle="round",
        zorder=7.0,
    )
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color="white",
        lw=0.55,
        alpha=0.88,
        solid_capstyle="round",
        zorder=7.1,
    )

    trace = lag_trace(trace_xy)
    if trace is None:
        ax.plot(center[0], center[1], marker="D", ms=4.5, mfc=endpoint_color, mec="white", mew=0.65, zorder=8.0)
        return tuple(center)

    rel = trace - trace[-1:, :]
    image_rel = np.column_stack([-rel[:, 1], rel[:, 0]])
    along = image_rel @ tangent_img
    across = image_rel @ normal_img
    extent = float(np.nanpercentile(np.hypot(along, across), 98.0))
    if not np.isfinite(extent) or extent <= EPS:
        pts = np.repeat(center[None, :], trace.shape[0], axis=0)
    else:
        display_scale = 0.33 / extent
        pts = center[None, :] + along[:, None] * display_scale * tangent[None, :] + across[:, None] * display_scale * normal[None, :]

    if float(np.nanmax(np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1]))) > 1e-6:
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=path_color,
            lw=1.7,
            alpha=0.94,
            solid_capstyle="round",
            zorder=7.8,
        )
        ax.plot(
            pts[0, 0],
            pts[0, 1],
            marker="o",
            ms=3.6,
            mfc="white",
            mec=path_color,
            mew=0.85,
            zorder=8.0,
        )
    ax.plot(
        pts[-1, 0],
        pts[-1, 1],
        marker="D",
        ms=4.8,
        mfc=endpoint_color,
        mec="white",
        mew=0.70,
        zorder=8.2,
    )
    return tuple(pts[-1])


def add_visual_model_input_cube(
    ax,
    stimulus_image=None,
    contour_axis_image_deg=10.352312,
    *,
    show_motion_labels=True,
    show_model_labels=True,
    across_motion_scale=1.0,
    along_motion_scale=1.0,
    source_image=None,
    trace_xy=None,
    trace_path_color=BLUE,
    show_motion_overlay=True,
):
    image = normalize_image(stimulus_image) if stimulus_image is not None else make_stimulus()
    if trace_xy is not None:
        cube = trace_model_input_cube(source_image if source_image is not None else image, trace_xy)
        if cube is None:
            cube = contour_model_input_cube(
                image,
                contour_axis_image_deg,
                across_motion_scale=across_motion_scale,
                along_motion_scale=along_motion_scale,
            )
            trace_xy = None
    else:
        cube = contour_model_input_cube(
            image,
            contour_axis_image_deg,
            across_motion_scale=across_motion_scale,
            along_motion_scale=along_motion_scale,
        )
    corners = fig3_box_corners_3d(
        (0.0, 0.0, 0.0),
        (MODEL_INPUT_CUBE_W, MODEL_INPUT_CUBE_H, MODEL_INPUT_CUBE_D),
        yaw_deg=MODEL_INPUT_CUBE_YAW_DEG,
        pitch_deg=MODEL_INPUT_CUBE_PITCH_DEG,
        roll_deg=MODEL_INPUT_CUBE_ROLL_DEG,
    )
    cube_p2 = draw_fig3_lag_cube(
        ax,
        cube[::-1],
        corners,
        outline=FIG3_CYAN,
        edge_width=1.25,
        zorder=3.0,
    )

    top_y = float(cube_p2[:, 1].max())
    bottom_y = float(cube_p2[:, 1].min())
    left_x = float(cube_p2[:, 0].min())
    right_x = float(cube_p2[:, 0].max())
    cube_cx = 0.5 * (left_x + right_x)

    if show_model_labels:
        ax.text(cube_cx, top_y + 0.74, "Model input", ha="center", va="bottom", fontsize=8.6, color=FIG3_TEXT_COLOR, fontweight="bold")
        ax.text(cube_cx, top_y + 0.42, "Visual", ha="center", va="bottom", fontsize=8.0, color=FIG3_TEXT_COLOR)
        ax.text(cube_cx, top_y + 0.20, "space × space × time", ha="center", va="bottom", fontsize=6.5, color="#555", style="italic")

    p_front_bot = cube_p2[0] + np.array([0.0, -0.18])
    p_back_bot = cube_p2[4] + np.array([0.0, -0.18])
    ax.annotate("", xy=p_back_bot, xytext=p_front_bot, arrowprops=dict(arrowstyle="<-", lw=0.8, color=GRAY), zorder=7)
    d = p_back_bot - p_front_bot
    norm = float(np.hypot(d[0], d[1])) or 1.0
    perp = np.array([d[1], -d[0]]) / norm
    if perp[1] > 0:
        perp = -perp
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    label_pos = 0.5 * (p_front_bot + p_back_bot) + perp * 0.16
    ax.text(label_pos[0], label_pos[1], f"{MODEL_INPUT_N_LAGS / 120.0 * 1000.0:.0f} ms", ha="center", va="center", rotation=angle, rotation_mode="anchor", fontsize=6.0, color="#555", style="italic")

    gaze = None
    if bool(show_motion_overlay) and trace_xy is not None:
        gaze = add_trace_motion_overlay_on_cube(
            ax,
            cube_p2,
            contour_axis_image_deg,
            trace_xy,
            path_color=trace_path_color,
        )
    elif bool(show_motion_overlay):
        gaze = add_contour_motion_overlays_on_cube(
            ax,
            cube_p2,
            contour_axis_image_deg,
            show_labels=show_motion_labels,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_xlim(left_x - 0.34, right_x + 0.34)
    top_pad = 1.02 if show_model_labels else 0.28
    ax.set_ylim(bottom_y - 0.58, top_y + top_pad)
    return gaze


def add_stimulus(ax, stimulus_image=None, contour_axis_image_deg=10.352312, motion_eye=None):
    image = normalize_image(stimulus_image) if stimulus_image is not None else make_stimulus()
    n = int(image.shape[0])
    ax.imshow(image, cmap="gray", interpolation="bicubic")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_xlim(0, n - 1)
    ax.set_ylim(n - 1, 0)
    ax.add_patch(patches.Rectangle((0, 0), n - 1, n - 1, fill=False, lw=1.0, ec=INK))

    # Motion overlays are centered on the local contour in the displayed patch.
    if stimulus_image is not None:
        tangent = axis_vector_image(float(contour_axis_image_deg))
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        axis_center = np.array([0.5 * (n - 1), 0.5 * (n - 1)], dtype=np.float64)
        scale = float(n) / 540.0
        axis_half_len = 205.0 * scale
        line_start = axis_center - tangent * axis_half_len
        line_end = axis_center + tangent * axis_half_len
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            color="#18a6b8",
            lw=2.0,
            alpha=0.78,
            linestyle=(0, (5.5, 4.0)),
            solid_capstyle="round",
        )
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            color="white",
            lw=0.45,
            alpha=0.76,
            linestyle=(0, (5.5, 4.0)),
            solid_capstyle="round",
        )

        red_center = axis_center
        blue_center = axis_center
        red_half_len = 34.0 * scale
        blue_half_len = 104.0 * scale
    else:
        contour_slope = -1.72
        tangent = np.array([1.0, contour_slope])
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-contour_slope, 1.0])
        normal = normal / np.linalg.norm(normal)
        red_center = np.array([0.5 * (n - 1), 0.5 * (n - 1)])
        blue_center = red_center.copy()
        axis_center = red_center.copy()
        red_half_len = 22.0
        blue_half_len = 68.0

    normal = normal / np.linalg.norm(normal)

    if motion_eye is not None:
        add_panel_a_trace_path(ax, axis_center, motion_eye["large_xy_px"], BLUE, lw=1.85, zorder=4)
        add_panel_a_trace_path(ax, axis_center, motion_eye["small_xy_px"], RED, lw=1.95, zorder=5)
    else:
        blue_start = blue_center - normal * blue_half_len
        blue_end = blue_center + normal * blue_half_len
        ax.annotate(
            "",
            xy=blue_end,
            xytext=blue_start,
            arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=3.1, mutation_scale=22),
            zorder=4,
        )

        red_start = red_center - normal * red_half_len
        red_end = red_center + normal * red_half_len
        ax.annotate(
            "",
            xy=red_end,
            xytext=red_start,
            arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=3.0, mutation_scale=19),
            zorder=5,
        )

    gaze = axis_center if stimulus_image is not None else red_center
    ax.plot(gaze[0], gaze[1], marker="o", ms=4.2, mfc="white", mec=INK, mew=1.0, zorder=5)
    return tuple(gaze)


def add_source_overview(ax, canvas, crop_center_xy, crop_size_px, *, label=True, skew=False):
    image = normalize_image(canvas)
    h, w = image.shape[:2]
    cx, cy = crop_center_xy
    size = float(crop_size_px)
    left = float(cx) - size / 2.0
    top = float(cy) - size / 2.0

    if (
        skew
        and draw_fig3_project_screen is not None
        and fig3_screen_corners_3d is not None
        and FIG3_SCR_W is not None
        and FIG3_SCR_H is not None
    ):
        roi = np.asarray(
            [
                [top, top + size],
                [left, left + size],
            ],
            dtype=np.float64,
        )
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        dst_quad, roi_quad, _h_fwd, _ = draw_fig3_project_screen(
            ax,
            image_u8,
            fig3_screen_corners_3d((0.0, 0.0, 0.0), FIG3_SCR_W, FIG3_SCR_H),
            roi=roi,
            screen_shape=image.shape[:2],
            edge_color=INK,
            edge_width=0.85,
            roi_color=FIG3_CYAN,
            roi_width=1.25,
            zorder=1,
        )
        ax._source_roi_quad = roi_quad
        corners = np.asarray(dst_quad, dtype=np.float64)
        pad_x = 0.035 * float(np.ptp(corners[:, 0]))
        pad_y = 0.070 * float(np.ptp(corners[:, 1]))
        ax.set_xlim(float(np.min(corners[:, 0]) - pad_x), float(np.max(corners[:, 0]) + pad_x))
        ax.set_ylim(float(np.min(corners[:, 1]) - pad_y), float(np.max(corners[:, 1]) + pad_y))
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("W")
    else:
        ax.imshow(image, cmap="gray", interpolation="bicubic")
        ax.add_patch(
            patches.Rectangle(
                (left, top),
                size,
                size,
                fill=False,
                lw=1.6,
                ec=FIG3_CYAN,
            )
        )
        ax.set_xlim(0, w - 1)
        ax.set_ylim(h - 1, 0)
        ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        if skew:
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.85)
            spine.set_edgecolor(INK)
    if label:
        ax.text(
            0.02,
            0.97,
            "source image",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.4),
        )


def connect_crop_to_source_overview(fig, stim_ax, overview_ax, crop_center_xy, crop_size_px):
    cx, cy = crop_center_xy
    size = float(crop_size_px)
    left = float(cx) - size / 2.0
    right = float(cx) + size / 2.0
    top = float(cy) - size / 2.0
    bottom = float(cy) + size / 2.0
    stim_pos = stim_ax.get_position()
    overview_pos = overview_ax.get_position()
    source_roi_quad = getattr(overview_ax, "_source_roi_quad", None)
    overview_transform = getattr(overview_ax, "_source_plane_transform", overview_ax.transData)
    if source_roi_quad is not None and overview_pos.x0 < stim_pos.x0:
        roi_quad = np.asarray(source_roi_quad, dtype=np.float64)
        starts = [
            fig.transFigure.inverted().transform(overview_ax.transData.transform(roi_quad[2])),
            fig.transFigure.inverted().transform(overview_ax.transData.transform(roi_quad[1])),
        ]
        ends = [
            fig.transFigure.inverted().transform(stim_ax.transAxes.transform((0.0, 1.0))),
            fig.transFigure.inverted().transform(stim_ax.transAxes.transform((0.0, 0.0))),
        ]
    elif overview_pos.x0 < stim_pos.x0:
        starts = [
            fig.transFigure.inverted().transform(overview_transform.transform((right, top))),
            fig.transFigure.inverted().transform(overview_transform.transform((right, bottom))),
        ]
        ends = [
            fig.transFigure.inverted().transform(stim_ax.transAxes.transform((0.0, 1.0))),
            fig.transFigure.inverted().transform(stim_ax.transAxes.transform((0.0, 0.0))),
        ]
    else:
        starts = [
            fig.transFigure.inverted().transform(stim_ax.transAxes.transform((1.0, 1.0))),
            fig.transFigure.inverted().transform(stim_ax.transAxes.transform((1.0, 0.0))),
        ]
        ends = [
            fig.transFigure.inverted().transform(overview_transform.transform((left, top))),
            fig.transFigure.inverted().transform(overview_transform.transform((left, bottom))),
        ]
    for start, end in zip(starts, ends):
        fig.patches.append(
            patches.FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-",
                linestyle=(0, (2.8, 2.8)),
                linewidth=0.65,
                color=FIG3_CYAN,
                alpha=0.46,
            )
        )


def add_eye_icon(ax, sf="high", show_label=True, label=None, orientation_deg=None):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    n = 140
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    bar_angle = 8.0 if orientation_deg is None or not np.isfinite(orientation_deg) else -float(orientation_deg)
    theta = np.deg2rad(bar_angle - 90.0)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    envelope = np.exp(-(xx**2 + yy**2) / (2 * 0.42**2))
    freq_lookup = {"high": 9.0, "medium": 4.8, "low": 2.2}
    freq = freq_lookup.get(sf, 4.8)
    gabor = 0.50 + 0.48 * np.cos(2 * np.pi * freq * xr) * envelope
    aperture = xx**2 + yy**2 <= 0.92**2
    rgba = plt.get_cmap("gray")(np.clip(gabor, 0, 1))
    rgba[..., 3] = aperture.astype(float)
    ax.imshow(rgba, extent=(0.08, 0.92, 0.16, 1.00), origin="lower", interpolation="bicubic")
    ax.add_patch(patches.Circle((0.50, 0.58), 0.40, fill=False, lw=1.05, ec=INK))
    if show_label:
        label_lookup = {"high": "High-SF unit", "medium": "Medium-SF unit", "low": "Low-SF unit"}
        ax.text(0.5, 0.00, label or label_lookup.get(sf, "Unit"), ha="center", va="top", fontsize=8.5)


def scale_trace_pair(red, blue):
    red = np.asarray(red, dtype=np.float64)
    blue = np.asarray(blue, dtype=np.float64)
    both = np.concatenate([red[np.isfinite(red)], blue[np.isfinite(blue)]])
    if both.size == 0:
        return np.zeros_like(red), np.zeros_like(blue)
    if float(np.nanmin(both)) < 0.0 < float(np.nanmax(both)):
        denom = float(np.nanpercentile(np.abs(both), 98.0))
        if not np.isfinite(denom) or denom <= EPS:
            denom = float(np.nanmax(np.abs(both)))
        if denom <= EPS:
            return np.full_like(red, 0.50), np.full_like(blue, 0.50)
        return 0.50 + 0.34 * np.clip(red / denom, -1.0, 1.0), 0.50 + 0.34 * np.clip(blue / denom, -1.0, 1.0)
    lo, hi = np.nanpercentile(both, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
    if hi <= lo:
        return np.full_like(red, 0.50), np.full_like(blue, 0.50)
    red = np.clip((red - lo) / (hi - lo), 0, 1)
    blue = np.clip((blue - lo) / (hi - lo), 0, 1)
    return 0.14 + 0.70 * red, 0.14 + 0.70 * blue


def synthetic_across_contour_trace_pair(n_frames=SYNTHETIC_TRACE_FRAMES):
    t = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
    small = SYNTHETIC_SMALL_ACROSS_PX * np.sin(2.0 * np.pi * (SYNTHETIC_TRACE_CYCLES * t - 0.10))
    large = SYNTHETIC_LARGE_SCALE * small
    return {"t": t, "small_px": small, "large_px": large}


def sf_cpd_to_wavelength_px(sf_cpd):
    return MODEL_PPD / float(sf_cpd)


def panel_a_grating_normal(contour_axis_image_deg):
    theta = np.deg2rad(float(contour_axis_image_deg))
    return np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float64)


def trace_to_image_px(trace_xy_deg):
    trace = np.asarray(trace_xy_deg, dtype=np.float64)
    centered = trace - np.nanmean(trace, axis=0, keepdims=True)
    # Retinal-image displacement convention used by the model-input rendering.
    return np.column_stack([-centered[:, 1], centered[:, 0]]) * MODEL_PPD


def rotate_trace_xy_px(trace_xy_px, angle_deg):
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    theta = np.deg2rad(float(angle_deg))
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )
    return trace @ rotation.T


def trace_bank_metrics(trace_index, image_index=SCHEMATIC_NEW_BANK_IMAGE_INDEX):
    if not TRACE_COMPONENT_METRICS_CSV.exists():
        return {}
    try:
        metrics = pd.read_csv(
            TRACE_COMPONENT_METRICS_CSV,
            usecols=[
                "image_index",
                "trace_index",
                "has_microsaccade",
                "rendered_path_length_arcmin",
                "across_path_arcmin",
                "along_path_arcmin",
            ],
        )
    except Exception:
        return {}
    rows = metrics[
        metrics["trace_index"].eq(int(trace_index))
        & metrics["image_index"].eq(int(image_index))
    ].copy()
    if rows.empty:
        rows = metrics[metrics["trace_index"].eq(int(trace_index))].copy()
    if rows.empty:
        return {}
    drift_rows = rows[~rows["has_microsaccade"].astype(bool)]
    if not drift_rows.empty:
        rows = drift_rows
    row = rows.iloc[0]
    return {
        "rendered_path_length_arcmin": float(row["rendered_path_length_arcmin"]),
        "across_path_arcmin": float(row["across_path_arcmin"]),
        "along_path_arcmin": float(row["along_path_arcmin"]),
    }


def trace_path_length_arcmin(trace_xy_deg):
    trace = np.asarray(trace_xy_deg, dtype=np.float64)
    return float(np.nansum(np.linalg.norm(np.diff(trace, axis=0), axis=1)) * 60.0)


def selected_real_panel_a_trace_pair(contour_axis_image_deg):
    if not TRACE_XY_NPY.exists():
        return None
    try:
        trace_xy = np.load(TRACE_XY_NPY, mmap_mode="r")
    except Exception:
        return None
    if PANEL_A_SMALL_TRACE_INDEX >= trace_xy.shape[0] or PANEL_A_LARGE_TRACE_INDEX >= trace_xy.shape[0]:
        return None

    small_deg = np.asarray(trace_xy[PANEL_A_SMALL_TRACE_INDEX], dtype=np.float64)
    large_deg = np.asarray(trace_xy[PANEL_A_LARGE_TRACE_INDEX], dtype=np.float64)
    small_xy_px = rotate_trace_xy_px(
        trace_to_image_px(small_deg),
        PANEL_A_SMALL_TRACE_ROTATION_DEG,
    )
    large_xy_px = rotate_trace_xy_px(
        trace_to_image_px(large_deg),
        PANEL_A_LARGE_TRACE_ROTATION_DEG,
    )
    normal = panel_a_grating_normal(contour_axis_image_deg)
    small_metrics = trace_bank_metrics(PANEL_A_SMALL_TRACE_INDEX)
    large_metrics = trace_bank_metrics(PANEL_A_LARGE_TRACE_INDEX)
    return {
        "t": np.linspace(0.0, 1.0, small_xy_px.shape[0], dtype=np.float64),
        "small_xy_px": small_xy_px,
        "large_xy_px": large_xy_px,
        "small_px": small_xy_px @ normal,
        "large_px": large_xy_px @ normal,
        "small_trace_index": PANEL_A_SMALL_TRACE_INDEX,
        "large_trace_index": PANEL_A_LARGE_TRACE_INDEX,
        "small_path_arcmin": small_metrics.get(
            "rendered_path_length_arcmin",
            trace_path_length_arcmin(small_deg),
        ),
        "large_path_arcmin": large_metrics.get(
            "rendered_path_length_arcmin",
            trace_path_length_arcmin(large_deg),
        ),
    }


def sampled_luminance_trace(motion_px, wavelength_px):
    motion = np.asarray(motion_px, dtype=np.float64)
    luminance = 0.50 + 0.46 * np.sin(2.0 * np.pi * motion / float(wavelength_px))
    return np.clip(luminance, 0.0, 1.0)


def temporal_integrate_trace(values, n_frames=PANEL_A_INTEGRATION_FRAMES):
    values = np.asarray(values, dtype=np.float64)
    n = max(1, int(n_frames))
    if n <= 1:
        return values.copy()
    kernel = np.ones(n, dtype=np.float64) / float(n)
    padded = np.pad(values, (n - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def schematic_grating_patch(size_px, wavelength_px, *, angle_deg):
    yy, xx = np.mgrid[:size_px, :size_px].astype(np.float64)
    cx = cy = 0.5 * (int(size_px) - 1)
    x = xx - cx
    y = yy - cy
    theta = np.deg2rad(float(angle_deg))
    normal_coord = -np.sin(theta) * x + np.cos(theta) * y
    envelope = np.exp(-0.5 * ((x / (0.58 * size_px)) ** 2 + (y / (0.58 * size_px)) ** 2))
    image = 0.50 + 0.46 * np.sin(2.0 * np.pi * normal_coord / float(wavelength_px)) * envelope
    return np.clip(image, 0.0, 1.0)


def add_panel_a_trace_path(ax, center, trace_xy_px, color, *, lw, zorder):
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 2:
        return
    points = np.asarray(center, dtype=np.float64)[None, :] + trace
    ax.plot(
        points[:, 0],
        points[:, 1],
        color="white",
        lw=float(lw) + 0.85,
        alpha=0.72,
        solid_capstyle="round",
        zorder=zorder - 0.1,
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        lw=float(lw),
        alpha=0.92,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.plot(points[-1, 0], points[-1, 1], marker="o", ms=2.4, mfc=color, mec="white", mew=0.55, zorder=zorder + 0.2)


def add_grating_trace_icon(ax, row, eye, *, contour_axis_image_deg, show_label=True):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    size = 160
    image = schematic_grating_patch(
        size,
        row["wavelength_px"],
        angle_deg=contour_axis_image_deg,
    )
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
    center = np.asarray([0.5 * (size - 1), 0.5 * (size - 1)], dtype=np.float64)
    trace_arrays = [
        np.asarray(eye["small_xy_px"], dtype=np.float64),
        np.asarray(eye["large_xy_px"], dtype=np.float64),
    ]
    trace_extent = max(float(np.nanmax(np.abs(trace))) for trace in trace_arrays)
    view_half = min(0.48 * size, max(18.0, trace_extent + 7.0))
    tangent = axis_vector_image(contour_axis_image_deg)
    line_half = 0.92 * view_half
    ax.plot(
        [center[0] - tangent[0] * line_half, center[0] + tangent[0] * line_half],
        [center[1] - tangent[1] * line_half, center[1] + tangent[1] * line_half],
        color=FIG3_CYAN,
        lw=1.15,
        alpha=0.80,
        linestyle=(0, (4.2, 3.2)),
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(center[0], center[1], marker="o", ms=3.0, mfc="white", mec=INK, mew=0.65, zorder=4)
    add_panel_a_trace_path(ax, center, eye["large_xy_px"], BLUE, lw=1.15, zorder=5)
    add_panel_a_trace_path(ax, center, eye["small_xy_px"], RED, lw=1.25, zorder=6)
    ax.set_xlim(center[0] - view_half, center[0] + view_half)
    ax.set_ylim(center[1] + view_half, center[1] - view_half)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#444")
    if show_label:
        ax.text(
            0.045,
            0.055,
            row.get("short_label", row.get("label", "")),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="white",
            weight="bold",
            linespacing=0.95,
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="black")],
        )


def synthetic_ssi_proxy(trace):
    trace = np.asarray(trace, dtype=np.float64)
    centered = trace - float(np.nanmean(trace))
    scaled = centered**2
    hi = float(np.nanpercentile(scaled, 98.0))
    if not np.isfinite(hi) or hi <= EPS:
        return np.zeros_like(trace)
    return scaled / hi


def make_synthetic_left_side(stimulus_image=None, contour_axis_image_deg=10.352312):
    eye = selected_real_panel_a_trace_pair(contour_axis_image_deg)
    if eye is None:
        eye = synthetic_across_contour_trace_pair()
        normal = panel_a_grating_normal(contour_axis_image_deg)
        eye["small_xy_px"] = np.outer(eye["small_px"], normal)
        eye["large_xy_px"] = np.outer(eye["large_px"], normal)
        eye["small_trace_index"] = -1
        eye["large_trace_index"] = -1
        eye["small_path_arcmin"] = float("nan")
        eye["large_path_arcmin"] = float("nan")
    specs = [
        {
            "sf": "high",
            "label": f"High-SF\n{PANEL_A_HIGH_SF_CPD:g} cpd",
            "short_label": f"High-SF\n{PANEL_A_HIGH_SF_CPD:g} cpd",
            "wavelength_px": sf_cpd_to_wavelength_px(PANEL_A_HIGH_SF_CPD),
        },
        {
            "sf": "low",
            "label": f"Low-SF\n{PANEL_A_LOW_SF_CPD:g} cpd",
            "short_label": f"Low-SF\n{PANEL_A_LOW_SF_CPD:g} cpd",
            "wavelength_px": sf_cpd_to_wavelength_px(PANEL_A_LOW_SF_CPD),
        },
    ]
    rows = []
    for spec in specs:
        small = temporal_integrate_trace(
            sampled_luminance_trace(eye["small_px"], spec["wavelength_px"])
        )
        large = temporal_integrate_trace(
            sampled_luminance_trace(eye["large_px"], spec["wavelength_px"])
        )
        rows.append(
            {
                "sf": spec["sf"],
                "label": spec["label"],
                "short_label": spec["short_label"],
                "wavelength_px": spec["wavelength_px"],
                "preferred_orientation_deg": float(contour_axis_image_deg),
                "trace_small": small,
                "trace_large": large,
                "ssi_small": synthetic_ssi_proxy(small),
                "ssi_large": synthetic_ssi_proxy(large),
            }
        )
    return {"eye": eye, "unit_rows": rows}


def add_synthetic_eye_trace_panel(ax, synthetic_eye):
    fig = ax.figure
    bbox = ax.get_position()
    ax.remove()
    t = np.asarray(synthetic_eye["t"], dtype=np.float64)
    red = np.asarray(synthetic_eye["small_px"], dtype=np.float64)
    blue = np.asarray(synthetic_eye["large_px"], dtype=np.float64)
    ylim = 1.18 * float(np.nanmax(np.abs(blue)))
    if not np.isfinite(ylim) or ylim <= EPS:
        ylim = 1.0

    fig.text(
        bbox.x0,
        bbox.y1 + 0.010,
        "real FEM traces",
        fontsize=8.2,
        color=GRAY,
        ha="left",
        va="bottom",
    )
    lane_gap = 0.009
    lane_h = 0.5 * (bbox.height - lane_gap)
    small_trace_index = int(synthetic_eye.get("small_trace_index", -1))
    large_trace_index = int(synthetic_eye.get("large_trace_index", -1))
    small_path = float(synthetic_eye.get("small_path_arcmin", np.nan))
    large_path = float(synthetic_eye.get("large_path_arcmin", np.nan))
    small_label = "short path"
    large_label = "long path"
    if np.isfinite(small_path):
        small_label = f"{small_label}  {small_path:.0f}'"
    if np.isfinite(large_path):
        large_label = f"{large_label}  {large_path:.0f}'"
    lanes = [
        (bbox.y0 + lane_h + lane_gap, red, RED, small_label),
        (bbox.y0, blue, BLUE, large_label),
    ]
    lane_axes = []
    for y0, values, color, label in lanes:
        lane_ax = fig.add_axes([bbox.x0, y0, bbox.width, lane_h])
        lane_ax.axhline(0.0, color="#cdd1d6", lw=0.8, zorder=0)
        lane_ax.plot(t, values, color=color, lw=1.8)
        lane_ax.text(
            1.01,
            0.50,
            label,
            transform=lane_ax.transAxes,
            fontsize=7.4,
            color=color,
            ha="left",
            va="center",
            clip_on=False,
        )
        lane_ax.set_ylim(-ylim, ylim)
        lane_ax.set_xlim(0.0, 1.0)
        lane_ax.set_xticks([])
        lane_ax.set_yticks([])
        lane_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        lane_axes.append(lane_ax)

    bottom_ax = lane_axes[-1]
    bottom_ax.annotate(
        "",
        xy=(1.02, -0.12),
        xytext=(0.0, -0.12),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK),
        clip_on=False,
    )
    bottom_ax.text(0.50, -0.38, "frame", transform=bottom_ax.transAxes, ha="center", fontsize=7.8, clip_on=False)


def add_split_response_trace_axes(
    fig,
    left,
    bottom,
    width,
    height,
    *,
    sf="high",
    red_trace=None,
    blue_trace=None,
    preserve_luminance=False,
):
    signed_activation = False
    if red_trace is None or blue_trace is None:
        t = np.linspace(0, 1, 240)
        if sf == "high":
            red = 0.55 + 0.27 * np.sin(2 * np.pi * (2.2 * t + 0.05)) * np.exp(-0.8 * t)
            blue = 0.48 + 0.13 * np.sin(2 * np.pi * (1.3 * t + 0.3)) + 0.05 * np.sin(2 * np.pi * 5.1 * t)
        elif sf == "medium":
            red = 0.48 + 0.17 * np.sin(2 * np.pi * (2.0 * t + 0.15)) * np.exp(-0.45 * t)
            blue = 0.47 + 0.18 * np.sin(2 * np.pi * (2.6 * t - 0.04)) + 0.04 * np.sin(2 * np.pi * 5.0 * t)
        else:
            red = 0.44 + 0.10 * np.sin(2 * np.pi * (1.6 * t + 0.2)) * np.exp(-1.0 * t)
            blue = 0.47 + 0.30 * np.sin(2 * np.pi * (2.1 * t - 0.05)) + 0.08 * np.sin(2 * np.pi * 5.2 * t)
    else:
        both = np.concatenate([
            np.asarray(red_trace, dtype=np.float64).ravel(),
            np.asarray(blue_trace, dtype=np.float64).ravel(),
        ])
        signed_activation = float(np.nanmin(both)) < 0.0 < float(np.nanmax(both))
        if preserve_luminance:
            red = np.clip(np.asarray(red_trace, dtype=np.float64).ravel(), 0.0, 1.0)
            blue = np.clip(np.asarray(blue_trace, dtype=np.float64).ravel(), 0.0, 1.0)
            signed_activation = False
        else:
            red, blue = scale_trace_pair(red_trace, blue_trace)
        t = np.linspace(0, 1, len(red))

    lane_gap = 0.010
    lane_h = 0.5 * (float(height) - lane_gap)
    axes = [
        (fig.add_axes([left, bottom + lane_h + lane_gap, width, lane_h]), red, RED, "small"),
        (fig.add_axes([left, bottom, width, lane_h]), blue, BLUE, "large"),
    ]
    for ax, values, color, label in axes:
        if signed_activation or preserve_luminance:
            ax.axhline(0.50, color="#d6d8dc", lw=0.75, zorder=0)
        else:
            ax.axhline(0.10, color="#d6d8dc", lw=0.75, zorder=0)
        ax.plot(t, values, color=color, lw=1.7)
        ax.text(-0.018, 0.52, label, transform=ax.transAxes, color=color, fontsize=7.8, ha="right", va="center", clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.04, 0.96)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    bottom_ax = axes[-1][0]
    bottom_ax.annotate(
        "",
        xy=(1.02, -0.12),
        xytext=(0.0, -0.12),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK),
        clip_on=False,
    )
    bottom_ax.text(0.50, -0.36, "time", transform=bottom_ax.transAxes, ha="center", fontsize=8, clip_on=False)


def make_activation_map(kind="sharp", phase=0.0, n=140):
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    if kind == "sharp":
        x0 = 0.23 * np.sin(phase)
        y0 = 0.20 * np.cos(phase * 0.8)
        z = np.exp(-(((x - x0) / 0.20) ** 2 + ((y - y0) / 0.11) ** 2))
        z += 0.35 * np.exp(-(((x + 0.34) / 0.34) ** 2 + ((y + 0.20) / 0.24) ** 2))
    elif kind == "medium":
        x0 = 0.19 * np.sin(phase)
        y0 = 0.16 * np.cos(phase * 0.8)
        z = np.exp(-(((x - x0) / 0.34) ** 2 + ((y - y0) / 0.22) ** 2))
        z += 0.25 * np.exp(-(((x + 0.30) / 0.48) ** 2 + ((y + 0.22) / 0.34) ** 2))
    else:
        x0 = 0.15 * np.sin(phase)
        y0 = 0.11 * np.cos(phase * 0.8)
        z = np.exp(-(((x - x0) / 0.54) ** 2 + ((y - y0) / 0.38) ** 2))
        z += 0.15 * np.exp(-(((x + 0.35) / 0.70) ** 2 + ((y + 0.25) / 0.52) ** 2))
    z = (z - z.min()) / (z.max() - z.min())
    return z


def add_activation_map(ax, kind="sharp", title="", real_map=None, map_vlim=None):
    if real_map is None:
        z = make_activation_map(kind)
        vmin, vmax = 0.0, 1.0
    else:
        z = np.maximum(np.asarray(real_map, dtype=np.float64), 0.0)
        if map_vlim is None:
            vmin, vmax = 0.0, float(np.nanpercentile(z, 99.0))
        else:
            vmin, vmax = map_vlim
        vmax = max(float(vmax), EPS)
    ax.imshow(z, cmap=ACTIVATION_CMAP, vmin=vmin, vmax=vmax, interpolation=ACTIVATION_INTERPOLATION)
    ax.set_aspect("equal", adjustable="box")
    contour_levels = [vmin + 0.42 * (vmax - vmin), vmin + 0.68 * (vmax - vmin)]
    contour_levels = [level for level in contour_levels if float(np.nanmin(z)) < level < float(np.nanmax(z))]
    if contour_levels:
        ax.contour(z, levels=contour_levels, colors=[INK], linewidths=[0.45, 0.75], alpha=0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor(INK)
    ax.set_title(title, fontsize=10, pad=5)
    add_axis_arrows(ax)


def real_unit_condition_map(payload, unit_row, condition_id, frame_index):
    cond_idx = condition_index(payload, condition_id)
    unit_idx = int(unit_row["unit_index"])
    frame_idx = int(np.clip(int(frame_index), 0, payload["maps"].shape[1] - 1))
    return np.asarray(payload["maps"][cond_idx, frame_idx, unit_idx], dtype=np.float64)


def real_map_pair_limits(maps):
    values = np.concatenate([
        np.maximum(np.asarray(m, dtype=np.float64).ravel(), 0.0)
        for m in maps
        if m is not None
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmax = float(np.nanpercentile(values, 99.2))
    if not np.isfinite(vmax) or vmax <= EPS:
        vmax = float(np.nanmax(values)) if values.size else 1.0
    return 0.0, max(vmax, EPS)


def mean_centered_response_map(map_values):
    rate = np.maximum(np.asarray(map_values, dtype=np.float64), 0.0)
    finite = rate[np.isfinite(rate)]
    if finite.size == 0:
        return np.zeros_like(rate, dtype=np.float64)
    mean_rate = float(np.nanmean(finite))
    if not np.isfinite(mean_rate):
        return np.zeros_like(rate, dtype=np.float64)
    return rate - mean_rate


def mean_centered_map_pair_limits(maps):
    values = np.concatenate([
        mean_centered_response_map(m).ravel()
        for m in maps
        if m is not None
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(values), 98.5))
    if not np.isfinite(vmax) or vmax <= EPS:
        vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, EPS)
    return -vmax, vmax


def panel_b_map_pair_limits(maps):
    if PANEL_B_ACTIVATION_MAP_STYLE == "mean_centered_diverging":
        return mean_centered_map_pair_limits(maps)
    return real_map_pair_limits(maps)


def choose_right_panel_real_unit(payload):
    rows = payload.get("unit_rows", []) if payload is not None else []
    exact_metrics = None if payload is None else payload.get("schematic_rr100_final_map_unit_metrics")
    exact_maps = None if payload is None else payload.get("schematic_rr100_final_maps")
    if exact_maps is not None and isinstance(exact_metrics, pd.DataFrame) and not exact_metrics.empty:
        candidates = exact_metrics.copy()
        candidates["figure_candidate_score"] = pd.to_numeric(
            candidates.get("figure_candidate_score", 0.0),
            errors="coerce",
        ).fillna(0.0)
        candidates["real_minus_stable_map_ssi"] = pd.to_numeric(
            candidates.get("real_minus_stable_map_ssi", 0.0),
            errors="coerce",
        ).fillna(0.0)
        if RIGHT_PANEL_EXACT_UNIT_INDEX is not None:
            requested = candidates[candidates["unit_index"].astype(int).eq(int(RIGHT_PANEL_EXACT_UNIT_INDEX))].copy()
            if not requested.empty:
                candidates = requested
        if RIGHT_PANEL_UNIT_SF != "auto" and "sf_group" in candidates.columns:
            sf_lookup = {"high": "high_sf", "medium": "middle_sf", "low": "low_sf"}
            target_sf = sf_lookup.get(str(RIGHT_PANEL_UNIT_SF), str(RIGHT_PANEL_UNIT_SF))
            sf_candidates = candidates[candidates["sf_group"].astype(str).eq(target_sf)].copy()
            if not sf_candidates.empty:
                candidates = sf_candidates
        sharpening = candidates[candidates["real_minus_stable_map_ssi"] > 0].copy()
        if not sharpening.empty:
            candidates = sharpening
        candidates = candidates.sort_values(
            ["figure_candidate_score", "real_minus_stable_map_ssi"],
            ascending=False,
        )
        if not candidates.empty:
            row = candidates.iloc[0].to_dict()
            sf_group = str(row.get("sf_group", ""))
            sf_lookup = {"high_sf": "high", "middle_sf": "medium", "low_sf": "low"}
            row["sf"] = sf_lookup.get(sf_group, "auto")
            row["unit_label"] = str(row.get("unit_label", f"u{int(row['unit_index']):03d}"))
            row["label"] = f"Exact final-map unit {row['unit_label']}"
            row["right_panel_selection_source"] = "schematic_rr100_endpoint_final_maps"
            return row
    if not rows:
        return None
    if RIGHT_PANEL_UNIT_SF != "auto":
        by_sf = {row["sf"]: row for row in rows}
        return by_sf.get(RIGHT_PANEL_UNIT_SF) or rows[0]

    try:
        ref_idx = condition_index(payload, REFERENCE_CONDITION)
        stable_idx = condition_index(payload, STABILIZED_CONDITION)
    except Exception:
        return rows[0]

    best = None
    for row in rows:
        unit_idx = int(row["unit_index"])
        ref_maps = np.maximum(np.asarray(payload["maps"][ref_idx, :, unit_idx], dtype=np.float64), 0.0)
        stable_maps = np.maximum(np.asarray(payload["maps"][stable_idx, :, unit_idx], dtype=np.float64), 0.0)
        contrast = instantaneous_bits(ref_maps) - instantaneous_bits(stable_maps)
        if np.all(~np.isfinite(contrast)):
            continue
        frame_idx = int(np.nanargmax(contrast))
        score = float(contrast[frame_idx])
        candidate = dict(row)
        candidate["right_panel_frame_index"] = frame_idx
        candidate["right_panel_contrast_score"] = score
        if best is None or score > best["right_panel_contrast_score"]:
            best = candidate
    return best or rows[0]


def oriented_gaussian(x, y, cx, cy, theta_deg, sigma_along, sigma_across):
    theta = np.deg2rad(float(theta_deg))
    dx = x - float(cx)
    dy = y - float(cy)
    along = dx * np.cos(theta) + dy * np.sin(theta)
    across = -dx * np.sin(theta) + dy * np.cos(theta)
    return np.exp(
        -0.5
        * (
            (along / float(sigma_along)) ** 2
            + (across / float(sigma_across)) ** 2
        )
    )


def make_spatial_activation_map(kind="fem", n=180):
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    theta = -16.0
    stripe = 0.08 * (0.5 + 0.5 * np.sin(15.0 * (x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))))
    field = 0.09 + stripe
    centers = [
        (-0.72, 0.60, 0.95),
        (-0.42, 0.25, 0.78),
        (-0.05, 0.04, 0.86),
        (0.38, -0.25, 0.82),
        (0.70, -0.57, 0.74),
        (0.52, 0.44, 0.64),
        (-0.78, -0.52, 0.58),
    ]
    if kind == "fem":
        sigma_along, sigma_across = 0.25, 0.050
        smooth_passes = 2
        contour_levels = [0.43, 0.61]
    else:
        sigma_along, sigma_across = 0.43, 0.155
        smooth_passes = 11
        contour_levels = [0.49, 0.66]

    for cx, cy, amp in centers:
        field += amp * oriented_gaussian(
            x,
            y,
            cx,
            cy,
            theta,
            sigma_along,
            sigma_across,
        )

    field = smooth2d(field, n=smooth_passes)
    field = (field - field.min()) / (field.max() - field.min())
    if kind == "stable":
        field = 0.82 * field**1.35
    return field, contour_levels


def add_spatial_activation_map(ax, kind="fem", real_map=None, map_vlim=None):
    if real_map is None:
        field, contour_levels = make_spatial_activation_map(kind)
        vmin, vmax = 0.0, 1.0
    else:
        raw_field = np.maximum(np.asarray(real_map, dtype=np.float64), 0.0)
        if PANEL_B_ACTIVATION_MAP_STYLE == "mean_centered_diverging":
            field = mean_centered_response_map(raw_field)
            if map_vlim is None:
                vmin, vmax = mean_centered_map_pair_limits([raw_field])
            else:
                vmin, vmax = map_vlim
            contour_levels = []
        else:
            field = raw_field
            if map_vlim is None:
                vmin, vmax = real_map_pair_limits([field])
            else:
                vmin, vmax = map_vlim
            vmax = max(float(vmax), EPS)
            contour_levels = [
                vmin + 0.42 * (vmax - vmin),
                vmin + 0.68 * (vmax - vmin),
            ]
            zmin = float(np.nanmin(field))
            zmax = float(np.nanmax(field))
            contour_levels = [level for level in contour_levels if zmin < level < zmax]

    cmap = ACTIVATION_CMAP
    interpolation = ACTIVATION_INTERPOLATION
    if PANEL_B_ACTIVATION_MAP_STYLE == "mean_centered_diverging":
        cmap = "RdBu_r"
        interpolation = "bilinear"
        contour_levels = []
    elif PANEL_B_ACTIVATION_MAP_STYLE == "shared_gray_smooth":
        cmap = "gray"
        interpolation = "bilinear"
        contour_levels = []
    elif PANEL_B_ACTIVATION_MAP_STYLE == "sharp_gallery_gray":
        finite = field[np.isfinite(field)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, PANEL_B_SHARP_GALLERY_PERCENTILES)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        cmap = "gray"
        interpolation = "nearest"
        contour_levels = []

    ax.imshow(
        field,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    if contour_levels:
        ax.contour(
            field,
            levels=contour_levels,
            colors=["#343047"] * len(contour_levels),
            linewidths=[0.55, 0.85][: len(contour_levels)],
            alpha=0.74,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_edgecolor("#4b4b4b")


def _draw_readout_surface(ax, x0, y0, *, alpha=1.0, zorder=2):
    surface = np.array(
        [
            [x0, y0 - 0.155],
            [x0 + 0.092, y0 - 0.098],
            [x0 + 0.092, y0 + 0.120],
            [x0, y0 + 0.165],
        ]
    )
    ax.add_patch(
        patches.Polygon(
            surface,
            closed=True,
            facecolor=READOUT_FILL,
            edgecolor=READOUT_GREEN,
            linewidth=0.85,
            alpha=alpha,
            zorder=zorder,
        )
    )
    ax.plot(
        [x0 + 0.047, x0 + 0.047],
        [y0 - 0.105, y0 + 0.105],
        color=READOUT_GREEN,
        lw=0.65,
        alpha=alpha,
        zorder=zorder + 0.1,
    )
    yy = np.linspace(-1.0, 1.0, 90)
    curve_y = y0 + 0.105 * yy
    curve_x = x0 + 0.047 + 0.034 * np.exp(-0.5 * (yy / 0.38) ** 2)
    ax.plot(
        curve_x,
        curve_y,
        color=READOUT_GREEN,
        lw=1.0,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder + 0.2,
    )
    ax.plot(
        x0 + 0.047,
        y0,
        marker="o",
        ms=2.0,
        color=READOUT_GREEN,
        alpha=alpha,
        zorder=zorder + 0.3,
    )


def add_model_readout_bridge(ax, *, show_label=False):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    core = patches.FancyBboxPatch(
        (0.035, 0.220),
        0.235,
        0.570,
        boxstyle="round,pad=0.012,rounding_size=0.040",
        facecolor=CORE_FILL,
        edgecolor=CORE_EDGE,
        linewidth=0.65,
    )
    ax.add_patch(core)
    ax.text(0.062, 0.705, "Core", fontsize=6.4, color=INK, ha="left", va="center")
    ax.text(0.062, 0.620, "Conv. filters", fontsize=5.5, color=INK, ha="left", va="center")
    for idx, color in enumerate(FILTER_COLORS):
        y = 0.305 + idx * 0.082
        ax.add_patch(
            patches.Polygon(
                [(0.070, y), (0.160, y + 0.040), (0.160, y + 0.105), (0.070, y + 0.065)],
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=0.30,
            )
        )
        ax.plot([0.172, 0.172], [y + 0.020, y + 0.083], color=color, lw=1.8, solid_capstyle="butt")
        ax.add_patch(
            patches.Arc(
                (0.220, y + 0.052),
                0.090,
                0.082,
                theta1=-55,
                theta2=55,
                color=INK,
                lw=0.55,
                alpha=0.62,
            )
        )

    ax.annotate(
        "",
        xy=(0.352, 0.505),
        xytext=(0.288, 0.505),
        arrowprops=dict(arrowstyle="-|>", lw=0.75, color=GRAY, alpha=0.88),
    )

    for idx in range(5):
        dx = idx * 0.010
        dy = idx * 0.004
        ax.add_patch(
            patches.Rectangle(
                (0.370 + dx, 0.448 + dy),
                0.155,
                0.088,
                facecolor="#f1eadb",
                edgecolor=READOUT_GREEN,
                linewidth=0.42,
                alpha=0.72,
            )
        )
        ax.plot(
            [0.410 + dx, 0.410 + dx],
            [0.454 + dy, 0.528 + dy],
            color=FILTER_COLORS[idx % len(FILTER_COLORS)],
            lw=0.70,
            alpha=0.70,
        )

    ax.annotate(
        "",
        xy=(0.605, 0.505),
        xytext=(0.540, 0.505),
        arrowprops=dict(arrowstyle="-|>", lw=0.75, color=GRAY, alpha=0.88),
    )

    _draw_readout_surface(ax, 0.638, 0.430, alpha=0.22, zorder=1)
    _draw_readout_surface(ax, 0.682, 0.505, alpha=1.00, zorder=3)
    _draw_readout_surface(ax, 0.724, 0.580, alpha=0.22, zorder=1)
    ax.annotate(
        "",
        xy=(0.805, 0.725),
        xytext=(0.646, 0.325),
        arrowprops=dict(arrowstyle="-|>", lw=0.72, color=READOUT_GREEN, alpha=0.78),
    )
    ax.text(0.818, 0.735, "x,y", fontsize=5.3, color=READOUT_GREEN, ha="left", va="center")
    ax.text(
        0.762,
        0.235,
        "one response\nper position",
        fontsize=5.4,
        color=GRAY,
        ha="center",
        va="top",
        linespacing=0.92,
    )
    ax.text(
        0.152,
        0.170,
        "model",
        fontsize=5.9,
        color=GRAY,
        ha="center",
        va="center",
    )
    if show_label:
        ax.text(
            0.610,
            0.985,
            "same spatial kernel, shifted center",
            fontsize=5.8,
            color=GRAY,
            ha="center",
            va="top",
        )


def add_motion_sharpening_panel(
    fig,
    stimulus_image,
    contour_axis_image_deg,
    *,
    real_payload=None,
    real_unit_row=None,
):
    real_maps = {}
    real_map_vlim = None
    exact_maps = None if real_payload is None else real_payload.get("schematic_rr100_final_maps")
    exact_condition_id = None if real_payload is None else real_payload.get("schematic_rr100_final_condition_id")
    if exact_maps is not None and exact_condition_id is not None and real_unit_row is not None:
        ids = [str(x) for x in np.asarray(exact_condition_id).astype(str)]
        try:
            real_idx = ids.index("real_trace_final")
            stable_idx = ids.index("endpoint_stabilized_final")
            unit_idx = int(real_unit_row["unit_index"])
            maps_arr = np.asarray(exact_maps, dtype=np.float32)
            real_maps["fem"] = np.asarray(maps_arr[real_idx, unit_idx], dtype=np.float64)
            real_maps["stable"] = np.asarray(maps_arr[stable_idx, unit_idx], dtype=np.float64)
            real_map_vlim = panel_b_map_pair_limits(real_maps.values())
        except Exception:
            real_maps = {}
            real_map_vlim = None
    if not real_maps and real_payload is not None and real_unit_row is not None:
        frame_index = int(real_unit_row.get("right_panel_frame_index", real_unit_row.get("example_frame", 0)))
        try:
            real_maps["fem"] = real_unit_condition_map(
                real_payload,
                real_unit_row,
                REFERENCE_CONDITION,
                frame_index,
            )
            real_maps["stable"] = real_unit_condition_map(
                real_payload,
                real_unit_row,
                STABILIZED_CONDITION,
                frame_index,
            )
            real_map_vlim = panel_b_map_pair_limits(real_maps.values())
        except Exception:
            real_maps = {}
            real_map_vlim = None

    trace_source_image = None
    real_trace = None
    stable_trace = None
    if real_payload is not None:
        trace_source_image = real_payload.get("stimulus_model_source_patch")
        real_trace = real_payload.get("stimulus_real_trace_lag32")
        stable_trace = real_payload.get("stimulus_endpoint_stabilized_trace_lag32")

    rows = [
        ("FEM jittered movie", "fem", 0.560, 0.610, real_trace, 1.0, 1.0, BLUE),
        ("Stabilized movie", "stable", 0.165, 0.215, stable_trace, 0.0, 0.0, GRAY),
    ]
    map_label = "warm/cool = above/below map mean"
    if PANEL_B_ACTIVATION_MAP_STYLE != "mean_centered_diverging":
        map_label = "brighter = stronger activation"
    fig.text(0.890, 0.895, map_label, fontsize=7.6, color=GRAY, ha="center", va="center")
    for row_i, (label, map_kind, cube_y, map_y, trace_xy, across_scale, along_scale, trace_color) in enumerate(rows):
        fig.text(0.490, cube_y + 0.355, label, fontsize=15, ha="left", va="center")
        cube_ax = fig.add_axes([0.485, cube_y, 0.185, square_height(fig, 0.185)])
        add_visual_model_input_cube(
            cube_ax,
            stimulus_image,
            contour_axis_image_deg,
            show_motion_labels=False,
            show_model_labels=False,
            across_motion_scale=across_scale,
            along_motion_scale=along_scale,
            source_image=trace_source_image,
            trace_xy=trace_xy,
            trace_path_color=trace_color,
            show_motion_overlay=False,
        )
        map_w = 0.145
        map_h = square_height(fig, map_w)
        bridge_w = 0.115
        bridge_h = map_h * 0.76
        bridge_y = map_y + (map_h - bridge_h) / 2
        bridge_ax = fig.add_axes([0.685, bridge_y, bridge_w, bridge_h])
        add_model_readout_bridge(bridge_ax, show_label=(row_i == 0))
        add_small_arrow(fig, 0.804, 0.820, map_y + map_h * 0.52)
        fig.text(
            0.811,
            map_y + map_h * 0.39,
            "map\npixel",
            fontsize=5.8,
            color=GRAY,
            ha="center",
            va="top",
            linespacing=0.90,
        )

        map_ax = fig.add_axes([0.825, map_y, map_w, map_h])
        add_spatial_activation_map(
            map_ax,
            map_kind,
            real_map=real_maps.get(map_kind),
            map_vlim=real_map_vlim,
        )


def add_frame_grid(
    fig,
    left,
    bottom,
    kind="sharp",
    frame_w=0.027,
    gap=0.006,
    n_rows=3,
    n_cols=4,
    frame_maps=None,
    map_vlim=None,
    row_labels=None,
    row_colors=None,
):
    frame_h = square_height(fig, frame_w)
    row_gap = 0.010
    total_w = n_cols * frame_w + (n_cols - 1) * gap
    total_h = n_rows * frame_h + (n_rows - 1) * row_gap
    if frame_maps is not None:
        n_rows = len(frame_maps)
        n_cols = len(frame_maps[0]) if frame_maps else n_cols
        total_w = n_cols * frame_w + (n_cols - 1) * gap
        total_h = n_rows * frame_h + (n_rows - 1) * row_gap
        vmin, vmax = map_vlim if map_vlim is not None else (0.0, None)
    else:
        vmin, vmax = 0.0, 1.0

    for r in range(n_rows):
        for c in range(n_cols):
            ax = fig.add_axes(
                [
                    left + c * (frame_w + gap),
                    bottom + (n_rows - 1 - r) * (frame_h + row_gap),
                    frame_w,
                    frame_h,
                ]
            )
            if frame_maps is None:
                phase = c * 0.75 + r * 0.42
                z = make_activation_map(kind, phase=phase)
                if r == 0:
                    z = z**1.2
                elif r == 2:
                    z = np.roll(z, 10, axis=1)
            else:
                z = np.maximum(np.asarray(frame_maps[r][c], dtype=np.float64), 0.0)
            ax.imshow(z, cmap=ACTIVATION_CMAP, vmin=vmin, vmax=vmax, interpolation=ACTIVATION_INTERPOLATION)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor(INK)

    if row_labels:
        for r, label in enumerate(row_labels):
            y = bottom + (n_rows - 1 - r) * (frame_h + row_gap) + frame_h / 2
            color = row_colors[r] if row_colors and r < len(row_colors) else GRAY
            fig.text(left - 0.006, y, label, fontsize=6.4, ha="right", va="center", color=color)

    fig.text(left + frame_w / 2, bottom - 0.022, "frame 1", fontsize=7.0, ha="center")
    fig.text(left + total_w / 2, bottom - 0.022, "...", fontsize=10.5, ha="center")
    fig.text(left + total_w - frame_w / 2, bottom - 0.022, "frame T", fontsize=7.0, ha="center")
    return total_h


def add_source_overview_if_available(fig, payload, *, stim_ax=None):
    if payload is None:
        return None
    source_image = payload.get("stimulus_canvas")
    center = payload.get("stimulus_crop_center_xy")
    if source_image is None:
        source_image = payload.get("stimulus_model_source_patch")
        if source_image is not None:
            h, w = np.asarray(source_image).shape[:2]
            center = (0.5 * (w - 1), 0.5 * (h - 1))
    crop_size = payload.get("stimulus_crop_size_px")
    if source_image is None or center is None or crop_size is None:
        return None
    overview_ax = fig.add_axes([0.025, 0.645, 0.250, 0.220])
    add_source_overview(overview_ax, source_image, center, crop_size, label=False, skew=True)
    if stim_ax is not None:
        connect_crop_to_source_overview(fig, stim_ax, overview_ax, center, crop_size)
    return overview_ax


def save_source_window_overview(payload):
    if payload is None:
        return
    canvas = payload.get("stimulus_canvas")
    center = payload.get("stimulus_crop_center_xy")
    crop_size = payload.get("stimulus_crop_size_px")
    if canvas is None or center is None or crop_size is None:
        return

    fig = plt.figure(figsize=(8.0, 4.8), facecolor="white")
    ax = fig.add_axes([0.035, 0.045, 0.930, 0.910])
    add_source_overview(ax, canvas, center, crop_size, label=False)
    fig.savefig(f"{SOURCE_OVERVIEW_OUT_BASE}.png", bbox_inches="tight", pad_inches=0.04, dpi=RASTER_EXPORT_DPI)
    fig.savefig(f"{SOURCE_OVERVIEW_OUT_BASE}.svg", bbox_inches="tight", pad_inches=0.04, dpi=VECTOR_EXPORT_DPI)
    fig.savefig(f"{SOURCE_OVERVIEW_OUT_BASE}.pdf", bbox_inches="tight", pad_inches=0.04, dpi=VECTOR_EXPORT_DPI)
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "savefig.dpi": RASTER_EXPORT_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    real_payload = load_real_payload()
    stimulus_image = real_payload["patch"] if real_payload is not None else None
    contour_axis_image_deg = (
        real_payload["contour_axis_image_deg"] if real_payload is not None else 10.352312
    )
    synthetic_left = make_synthetic_left_side(stimulus_image, contour_axis_image_deg)
    right_panel_real_unit = choose_right_panel_real_unit(real_payload)

    fig = plt.figure(figsize=(14.7, 8.0), facecolor="white")

    add_panel_header(fig, "A", "Contour-relative stimulus and unit responses", 0.020, 0.955)
    add_panel_header(fig, "B", "Motion sharpens unit activations across space", 0.455, 0.955)
    if SHOW_ACTIVATION_MAP_SECTION:
        add_panel_header(fig, "C", "Activation maps across movie frames", 0.520, 0.950)

    stim_w = 0.135
    stim_ax = fig.add_axes([0.235, 0.605, stim_w, square_height(fig, stim_w)])
    gaze_xy = add_stimulus(
        stim_ax,
        stimulus_image,
        contour_axis_image_deg,
        motion_eye=synthetic_left["eye"],
    )

    add_source_overview_if_available(fig, real_payload, stim_ax=stim_ax)
    fig.text(0.080, 0.850, "full stimulus", fontsize=7.8, color=GRAY, ha="center", va="center")
    fig.text(0.305, 0.850, "151 px model window", fontsize=7.8, color=GRAY, ha="center", va="center")

    eye_trace_ax = fig.add_axes([0.055, 0.500, 0.200, 0.100])
    add_synthetic_eye_trace_panel(eye_trace_ax, synthetic_left["eye"])
    add_motion_sharpening_panel(
        fig,
        stimulus_image,
        contour_axis_image_deg,
        real_payload=real_payload,
        real_unit_row=right_panel_real_unit,
    )

    fig.text(0.238, 0.458, "sampled luminance at center pixel", fontsize=9, color=GRAY, ha="center")
    real_by_sf = {row["sf"]: row for row in synthetic_left["unit_rows"]}

    unit_rows = [
        ("high", "High-SF unit", "sharp", 0.335, 0.620, real_by_sf.get("high")),
        ("low", "Low-SF unit", "broad", 0.185, 0.240, real_by_sf.get("low")),
    ]

    for sf, label, map_kind, trace_y, row_y, real_row in unit_rows:
        orientation_deg = real_row["preferred_orientation_deg"] if real_row else None
        icon_ax = fig.add_axes([0.035, trace_y - 0.017, 0.064, square_height(fig, 0.064)])
        if real_row:
            add_grating_trace_icon(
                icon_ax,
                real_row,
                synthetic_left["eye"],
                contour_axis_image_deg=contour_axis_image_deg,
            )
        else:
            add_eye_icon(icon_ax, sf, label=label, orientation_deg=orientation_deg)
        add_split_response_trace_axes(
            fig,
            0.150,
            trace_y,
            0.215,
            0.118,
            sf=sf,
            red_trace=real_row["trace_small"] if real_row else None,
            blue_trace=real_row["trace_large"] if real_row else None,
            preserve_luminance=real_row is not None,
        )

        if SHOW_ACTIVATION_MAP_SECTION:
            row_icon_ax = fig.add_axes([0.458, row_y + 0.064, 0.042, square_height(fig, 0.042)])
            add_eye_icon(row_icon_ax, sf, show_label=False, orientation_deg=orientation_deg)
            row_label = real_row["label"] if real_row else label
            fig.text(0.479, row_y + 0.050, row_label, fontsize=8.0, ha="center", va="top", linespacing=1.0)

            map_w = 0.125
            map_ax = fig.add_axes([0.535, row_y, map_w, square_height(fig, map_w)])
            if real_row:
                cond_idx = condition_index(real_payload, REFERENCE_CONDITION)
                real_map = real_payload["maps"][cond_idx, real_row["example_frame"], real_row["unit_index"]]
                add_activation_map(map_ax, map_kind, "", real_map=real_map, map_vlim=real_row["map_vlim"])
            else:
                add_activation_map(map_ax, map_kind, "")

            grid_left = 0.685
            add_small_arrow(fig, 0.663, 0.678, row_y + square_height(fig, map_w) * 0.52)
            add_frame_grid(
                fig,
                grid_left,
                row_y + 0.012,
                map_kind,
                frame_w=0.030,
                gap=0.007,
                n_rows=3,
                n_cols=4,
                frame_maps=real_row["frame_maps"] if real_row else None,
                map_vlim=real_row["map_vlim"] if real_row else None,
                row_labels=GRID_ROW_LABELS if real_row else None,
                row_colors=GRID_ROW_COLORS if real_row else None,
            )

    if SHOW_ACTIVATION_MAP_SECTION:
        fig.text(0.754, 0.895, "activation map frames over trace", fontsize=9, color=GRAY, ha="center")
        fig.text(0.598, 0.895, "example activation map", fontsize=9, color=GRAY, ha="center")

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_BASE}.png", bbox_inches="tight", pad_inches=0.08, dpi=RASTER_EXPORT_DPI)
    fig.savefig(f"{OUT_BASE}.svg", bbox_inches="tight", pad_inches=0.08, dpi=VECTOR_EXPORT_DPI)
    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight", pad_inches=0.08, dpi=VECTOR_EXPORT_DPI)
    plt.close(fig)
    save_source_window_overview(real_payload)


if __name__ == "__main__":
    main()
