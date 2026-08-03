#!/usr/bin/env python3
"""Plot BackImage contour-relative FEM components.

This is a figure sandbox for the Figure 4E/tutorial candidate:

* contour-parallel vs contour-normal position-cloud RMS in arcmin;
* contour-parallel vs contour-normal diffusion constants from projected
  multi-lag MSD slopes;
* a polar density version of drift/edge alignment split by local
  orientation-coherence bracket.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

import sys

from VisionCore.paths import VISIONCORE_ROOT

# `jake` is a repo-root package rather than an installed one, so the root has to
# be importable before it resolves (same approach as paper/fig3/_fig3_data.py).
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from jake.twininfo.eye_controls import detect_microsaccade_events  # noqa: E402

from .extraction import _as_numpy, _contiguous_true_blocks, _load_dict_dataset, _speed_threshold_mad_valid_pairs
from .image_features import image_axis_rad_to_gaze_axis_rad
from .io_utils import parse_csv_list, write_json
from .run_backimage_image_structure_analysis import circular_axis_delta_deg


DEFAULT_INPUT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix_slope_v1"
    / "backimage_image_fem_windows.csv"
)
DEFAULT_CONDITION_INPUT = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "window_features.csv"
)
DEFAULT_OUT_DIR = (
    Path("outputs")
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_contour_motion_component_plots_v1"
)
DEFAULT_MSD_LAGS = (1, 2, 4, 8, 16)
DEFAULT_NO_COHERENCE_MAX = 0.10

COLORS = {
    "along": "#1b7f5c",
    "across": "#7a3b9a",
    "low": "#67768a",
    "mid": "#c58f2d",
    "high": "#1b7f5c",
    "Allen": "#245c8a",
    "Logan": "#b26b22",
    "fixrsvp": "#5b6f95",
    "backimage": "#1b7f5c",
    "no_coherence": "#333333",
}

COMBINED_RMS_RATIO_COLORS = {
    "no_coherence": "#242a2f",
    "low": "#8e9aa6",
    "mid": "#6f7a83",
    "high": "#3366aa",
}


@dataclass(frozen=True)
class ComponentPlotConfig:
    input_windows: str
    condition_window_features: str
    out_dir: str
    dt: float
    msd_lags: list[int]
    condition_phases: list[str]
    condition_discard_initial_s: float
    condition_saccade_pad_s: float
    condition_hard_speed_cutoff_deg_s: float | None
    condition_post_event_exclusion_s: float
    condition_min_segment_s: float
    condition_max_diffusion_lag_s: float
    condition_within_segment_max_diffusion_lag_s: float
    condition_min_segments_per_lag: int
    no_coherence_max: float
    n_bootstrap: int
    seed: int
    recompute_traces: bool


def _subject_from_session(session: Any) -> str:
    return str(session).split("_", 1)[0]


def _axis_vectors(axis_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.radians(np.asarray(axis_deg, dtype=np.float64))
    along = np.column_stack([np.cos(theta), np.sin(theta)])
    across = np.column_stack([-np.sin(theta), np.cos(theta)])
    return along, across


def _patch_orientation_features(patch: np.ndarray) -> dict[str, Any]:
    """Estimate the local contour axis from an already gaze-centered image patch."""
    try:
        arr = np.asarray(patch, dtype=np.float64)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        if arr.ndim != 2 or arr.size < 16 or not np.isfinite(arr).all():
            return {
                "image_feature_ok": False,
                "image_feature_error": "bad_patch",
            }
        gx = ndimage.sobel(arr, axis=1, mode="nearest")
        gy = ndimage.sobel(arr, axis=0, mode="nearest")
        grad_energy = gx * gx + gy * gy
        grad_mag = np.sqrt(grad_energy)
        jxx = float(np.mean(gx * gx))
        jyy = float(np.mean(gy * gy))
        jxy = float(np.mean(gx * gy))
        coherence_den = jxx + jyy
        coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / coherence_den if coherence_den > 0 else np.nan
        gradient_orientation_image = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        edge_orientation_image = gradient_orientation_image + np.pi / 2.0
        gradient_orientation = image_axis_rad_to_gaze_axis_rad(gradient_orientation_image)
        edge_orientation = image_axis_rad_to_gaze_axis_rad(edge_orientation_image)
        return {
            "image_feature_ok": True,
            "image_feature_error": "",
            "image_patch_mean": float(np.mean(arr)),
            "image_patch_std": float(np.std(arr)),
            "image_patch_rms_contrast": float(np.std(arr) / (abs(float(np.mean(arr))) + 1e-6)),
            "image_gradient_energy": float(np.mean(grad_energy)),
            "image_edge_density": float(np.mean(grad_mag > (np.mean(grad_mag) + np.std(grad_mag)))),
            "image_orientation_coherence": float(coherence),
            "image_gradient_axis_deg": float(np.degrees(gradient_orientation)),
            "image_edge_axis_deg": float(np.degrees(edge_orientation)),
            "image_gradient_axis_array_deg": float(np.degrees(gradient_orientation_image)),
            "image_edge_axis_array_deg": float(np.degrees(edge_orientation_image)),
        }
    except Exception as exc:
        return {
            "image_feature_ok": False,
            "image_feature_error": str(exc),
        }


def _project_covariance_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    along, across = _axis_vectors(out["image_edge_axis_deg"].to_numpy(dtype=np.float64))
    cxx = out["cov_xx_deg2"].to_numpy(dtype=np.float64)
    cxy = out["cov_xy_deg2"].to_numpy(dtype=np.float64)
    cyy = out["cov_yy_deg2"].to_numpy(dtype=np.float64)

    def project_var(u: np.ndarray) -> np.ndarray:
        return u[:, 0] * u[:, 0] * cxx + 2.0 * u[:, 0] * u[:, 1] * cxy + u[:, 1] * u[:, 1] * cyy

    out["rms_along_arcmin"] = 60.0 * np.sqrt(np.maximum(project_var(along), 0.0))
    out["rms_across_arcmin"] = 60.0 * np.sqrt(np.maximum(project_var(across), 0.0))
    out["rms_ratio_along_over_across"] = out["rms_along_arcmin"] / out["rms_across_arcmin"]
    out["rms_delta_along_minus_across_arcmin"] = out["rms_along_arcmin"] - out["rms_across_arcmin"]
    drift_orientation = 0.5 * np.degrees(np.arctan2(2.0 * cxy, cxx - cyy))
    out["drift_orientation_deg"] = drift_orientation
    out["drift_edge_delta_deg"] = circular_axis_delta_deg(drift_orientation, out["image_edge_axis_deg"].to_numpy(dtype=np.float64))
    out["drift_edge_cos2"] = np.cos(2.0 * np.radians(out["drift_edge_delta_deg"].to_numpy(dtype=np.float64)))
    return out


def _fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y - pred) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def projected_motion_features(
    trace: np.ndarray,
    *,
    edge_axis_deg: float,
    dt: float,
    msd_lags: Iterable[int] = DEFAULT_MSD_LAGS,
) -> dict[str, float]:
    """Compute contour-relative RMS and projected 1D MSD-slope diffusion."""
    x = np.asarray(trace, dtype=np.float64)
    out: dict[str, float] = {}
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 3 or not np.isfinite(x).all():
        for key in (
            "rms_along_arcmin",
            "rms_across_arcmin",
            "msd_slope_along_arcmin2_s",
            "msd_slope_across_arcmin2_s",
            "diffusion_along_arcmin2_s",
            "diffusion_across_arcmin2_s",
            "diffusion_fit_r2_along",
            "diffusion_fit_r2_across",
        ):
            out[key] = float("nan")
        return out

    along, across = _axis_vectors(np.asarray([edge_axis_deg], dtype=np.float64))
    u = along[0]
    v = across[0]
    centered = x - np.mean(x, axis=0)
    pos_along = centered @ u
    pos_across = centered @ v
    out["rms_along_arcmin"] = float(60.0 * np.sqrt(np.mean(pos_along * pos_along)))
    out["rms_across_arcmin"] = float(60.0 * np.sqrt(np.mean(pos_across * pos_across)))
    out["rms_ratio_along_over_across"] = float(out["rms_along_arcmin"] / out["rms_across_arcmin"]) if out["rms_across_arcmin"] > 0 else float("nan")
    out["rms_delta_along_minus_across_arcmin"] = float(out["rms_along_arcmin"] - out["rms_across_arcmin"])

    times: list[float] = []
    msd_along: list[float] = []
    msd_across: list[float] = []
    for lag in msd_lags:
        lag = int(lag)
        if lag <= 0 or x.shape[0] <= lag:
            out[f"msd_lag{lag}_along_arcmin2"] = float("nan")
            out[f"msd_lag{lag}_across_arcmin2"] = float("nan")
            continue
        disp = x[lag:] - x[:-lag]
        da = disp @ u
        dc = disp @ v
        ma = float(np.mean(da * da) * 3600.0)
        mc = float(np.mean(dc * dc) * 3600.0)
        out[f"msd_lag{lag}_along_arcmin2"] = ma
        out[f"msd_lag{lag}_across_arcmin2"] = mc
        times.append(lag * float(dt))
        msd_along.append(ma)
        msd_across.append(mc)

    slope_a, intercept_a, r2_a = _fit_slope(np.asarray(times), np.asarray(msd_along))
    slope_c, intercept_c, r2_c = _fit_slope(np.asarray(times), np.asarray(msd_across))
    out["msd_slope_along_arcmin2_s"] = slope_a
    out["msd_slope_across_arcmin2_s"] = slope_c
    out["msd_intercept_along_arcmin2"] = intercept_a
    out["msd_intercept_across_arcmin2"] = intercept_c
    out["diffusion_fit_r2_along"] = r2_a
    out["diffusion_fit_r2_across"] = r2_c
    out["diffusion_along_arcmin2_s"] = max(slope_a / 2.0, 0.0) if np.isfinite(slope_a) else float("nan")
    out["diffusion_across_arcmin2_s"] = max(slope_c / 2.0, 0.0) if np.isfinite(slope_c) else float("nan")
    out["diffusion_mean_projected_arcmin2_s"] = (
        max((slope_a + slope_c) / 4.0, 0.0)
        if np.isfinite(slope_a) and np.isfinite(slope_c)
        else float("nan")
    )
    return out


@lru_cache(maxsize=64)
def _session_backimage_trace_index(session_name: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    from DataYatesV1 import get_session

    subject, date = session_name.split("_", 1)
    session = get_session(subject, date)
    dset = _load_dict_dataset(Path(session.sess_dir) / "datasets" / "backimage.dset")
    eyepos = _as_numpy(dset["eyepos"]).astype(np.float64)
    trial_inds = _as_numpy(dset.covariates["trial_inds"]).reshape(-1).astype(int)
    trial_map = {int(trial): np.where(trial_inds == int(trial))[0] for trial in np.unique(trial_inds)}
    return eyepos, trial_map


def _window_trace(row: pd.Series) -> np.ndarray:
    eyepos, trial_map = _session_backimage_trace_index(str(row["session"]))
    trial_idx = int(row["trial_idx"])
    local_start = int(row["local_start"])
    local_stop = int(row["local_stop"])
    idx = trial_map.get(trial_idx)
    if idx is None or idx.size < local_stop:
        global_start = int(row["global_start"])
        global_stop = int(row["global_stop"])
        return eyepos[global_start:global_stop]
    return eyepos[idx[local_start:local_stop]]


def _coherence_brackets(values: pd.Series) -> tuple[pd.Series, dict[str, str]]:
    cats = pd.qcut(values.astype(float), 3, labels=["low", "mid", "high"])
    labels: dict[str, str] = {}
    for name in ["low", "mid", "high"]:
        sub = values[cats == name].astype(float)
        labels[name] = f"{name} ({sub.min():.2f}-{sub.max():.2f})"
    return cats, labels


def _no_coherence_subset(df: pd.DataFrame, no_coherence_max: float | None) -> pd.DataFrame:
    if no_coherence_max is None or not np.isfinite(no_coherence_max) or no_coherence_max <= 0.0:
        return df.iloc[0:0].copy()
    coherence = pd.to_numeric(df["image_orientation_coherence"], errors="coerce")
    return df[coherence <= float(no_coherence_max)].copy()


def _polar_density_and_stats(delta: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return None, {
            "n_windows": 0,
            "mean_cos2_delta": float("nan"),
            "median_abs_delta_deg": float("nan"),
            "fraction_within_15deg_parallel": float("nan"),
            "fraction_within_30deg_parallel": float("nan"),
            "fraction_within_30deg_orthogonal": float("nan"),
        }
    theta_half = np.radians(np.mod(delta, 180.0))
    theta = np.mod(np.concatenate([theta_half, theta_half + np.pi]), 2.0 * np.pi)
    counts, _ = np.histogram(theta, bins=bins, density=False)
    density = counts.astype(float) / max(float(np.sum(counts)) * (bins[1] - bins[0]), 1.0)
    density = _smooth_periodic(density, sigma_bins=1.4)
    stats = {
        "n_windows": int(delta.size),
        "mean_cos2_delta": float(np.nanmean(np.cos(2.0 * np.radians(delta)))),
        "median_abs_delta_deg": float(np.nanmedian(np.abs(delta))),
        "fraction_within_15deg_parallel": float(np.nanmean(np.abs(delta) <= 15.0)),
        "fraction_within_30deg_parallel": float(np.nanmean(np.abs(delta) <= 30.0)),
        "fraction_within_30deg_orthogonal": float(np.nanmean(np.abs(np.abs(delta) - 90.0) <= 30.0)),
    }
    return density, stats


def _draw_polar_density(
    ax: plt.Axes,
    centers: np.ndarray,
    density: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    fill_alpha: float = 0.08,
) -> None:
    theta = np.r_[centers, centers[0]]
    radius = np.r_[density, density[0]]
    ax.plot(theta, radius, color=color, lw=linewidth, ls=linestyle, label=label)
    if fill_alpha > 0.0:
        ax.fill(theta, radius, color=color, alpha=fill_alpha)


def _directional_rms_values(df: pd.DataFrame, rel_angle_rad: float) -> np.ndarray:
    edge = np.radians(pd.to_numeric(df["image_edge_axis_deg"], errors="coerce").to_numpy(dtype=np.float64))
    cxx = pd.to_numeric(df["cov_xx_deg2"], errors="coerce").to_numpy(dtype=np.float64)
    cxy = pd.to_numeric(df["cov_xy_deg2"], errors="coerce").to_numpy(dtype=np.float64)
    cyy = pd.to_numeric(df["cov_yy_deg2"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(edge) & np.isfinite(cxx) & np.isfinite(cxy) & np.isfinite(cyy)
    if not np.any(ok):
        return np.asarray([], dtype=np.float64)
    theta = edge[ok] + float(rel_angle_rad)
    ux = np.cos(theta)
    uy = np.sin(theta)
    var = ux * ux * cxx[ok] + 2.0 * ux * uy * cxy[ok] + uy * uy * cyy[ok]
    return 60.0 * np.sqrt(np.maximum(var, 0.0))


def _directional_rms_profile(df: pd.DataFrame, centers: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    if df.empty:
        return None, {
            "n_windows": 0,
            "median_parallel_rms_arcmin": float("nan"),
            "median_orthogonal_rms_arcmin": float("nan"),
            "parallel_minus_orthogonal_rms_arcmin": float("nan"),
            "parallel_over_orthogonal_rms": float("nan"),
        }

    profile = np.empty_like(centers, dtype=np.float64)
    for i, angle in enumerate(centers):
        vals = _directional_rms_values(df, float(angle))
        vals = vals[np.isfinite(vals)]
        profile[i] = float(np.nanmedian(vals)) if vals.size else float("nan")

    parallel = _directional_rms_values(df, 0.0)
    orthogonal = _directional_rms_values(df, 0.5 * np.pi)
    parallel = parallel[np.isfinite(parallel)]
    orthogonal = orthogonal[np.isfinite(orthogonal)]
    par_med = float(np.nanmedian(parallel)) if parallel.size else float("nan")
    orth_med = float(np.nanmedian(orthogonal)) if orthogonal.size else float("nan")
    stats = {
        "n_windows": int(min(parallel.size, orthogonal.size)),
        "median_parallel_rms_arcmin": par_med,
        "median_orthogonal_rms_arcmin": orth_med,
        "parallel_minus_orthogonal_rms_arcmin": float(par_med - orth_med) if np.isfinite(par_med) and np.isfinite(orth_med) else float("nan"),
        "parallel_over_orthogonal_rms": float(par_med / orth_med) if np.isfinite(par_med) and np.isfinite(orth_med) and orth_med > 0.0 else float("nan"),
    }
    if not np.isfinite(profile).any():
        return None, stats
    return profile, stats


def _draw_polar_rms_profile(
    ax: plt.Axes,
    centers: np.ndarray,
    profile: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    theta = np.r_[centers, centers[0]]
    radius = np.r_[profile, profile[0]]
    ax.plot(theta, radius, color=color, lw=linewidth, ls=linestyle, label=label)


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
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
        boots[i] = np.median(values[rng.integers(0, values.size, size=values.size)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _session_bracket_summary(df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = [
        "rms_along_arcmin",
        "rms_across_arcmin",
        "rms_delta_along_minus_across_arcmin",
        "rms_ratio_along_over_across",
        "diffusion_along_arcmin2_s",
        "diffusion_across_arcmin2_s",
        "msd_slope_along_arcmin2_s",
        "msd_slope_across_arcmin2_s",
        "drift_edge_cos2",
    ]
    session_rows = (
        df.groupby(["session", "coherence_bracket"], observed=True)[metrics + ["image_orientation_coherence"]]
        .median(numeric_only=True)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for bracket in ["low", "mid", "high"]:
        sub = session_rows[session_rows["coherence_bracket"].astype(str) == bracket]
        for metric in metrics + ["image_orientation_coherence"]:
            point, lo, hi = _bootstrap_ci(
                sub[metric].to_numpy(dtype=np.float64),
                n_bootstrap=n_bootstrap,
                seed=seed + 101 * (len(rows) + 1),
            )
            rows.append({
                "coherence_bracket": bracket,
                "metric": metric,
                "session_median": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_sessions": int(sub["session"].nunique()),
                "n_window_session_bins": int(sub.shape[0]),
            })
    return pd.DataFrame(rows)


def _paired_high_low_contrasts(df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = [
        "rms_along_arcmin",
        "rms_across_arcmin",
        "rms_delta_along_minus_across_arcmin",
        "rms_ratio_along_over_across",
        "diffusion_along_arcmin2_s",
        "diffusion_across_arcmin2_s",
        "msd_slope_along_arcmin2_s",
        "msd_slope_across_arcmin2_s",
        "drift_edge_cos2",
    ]
    session_rows = (
        df.groupby(["session", "coherence_bracket"], observed=True)[metrics]
        .median(numeric_only=True)
        .reset_index()
    )
    out: list[dict[str, Any]] = []
    for metric in metrics:
        wide = session_rows.pivot(index="session", columns="coherence_bracket", values=metric)
        if "high" not in wide.columns or "low" not in wide.columns:
            continue
        delta = (wide["high"] - wide["low"]).to_numpy(dtype=np.float64)
        point, lo, hi = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed + 700 + len(out))
        out.append({
            "metric": metric,
            "contrast": "high_minus_low_session_paired",
            "median_delta": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_sessions": int(np.count_nonzero(np.isfinite(delta))),
        })
    return pd.DataFrame(out)


def _summary_lookup(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = summary[summary["metric"].astype(str) == metric].copy()
    sub["coherence_bracket"] = pd.Categorical(sub["coherence_bracket"], categories=["low", "mid", "high"], ordered=True)
    return sub.sort_values("coherence_bracket")


def _plot_component_lines(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metrics: tuple[str, str],
    ylabel: str,
    title: str,
    out_path: Path,
    symlog: bool = False,
    show_session_lines: bool = True,
    component_labels: tuple[str, str] = ("along", "across"),
    tight_ylim: bool = False,
) -> None:
    session_rows = (
        df.groupby(["session", "coherence_bracket"], observed=True)[list(metrics)]
        .median(numeric_only=True)
        .reset_index()
    )
    x_map = {"low": 0, "mid": 1, "high": 2}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    rng = np.random.default_rng(4)
    for component, metric, color in [
        (component_labels[0], metrics[0], COLORS["along"]),
        (component_labels[1], metrics[1], COLORS["across"]),
    ]:
        piv = session_rows.pivot(index="session", columns="coherence_bracket", values=metric)
        if show_session_lines:
            for _, row in piv.iterrows():
                xs = np.asarray([x_map[b] for b in ["low", "mid", "high"] if b in row.index], dtype=float)
                ys = np.asarray([row.get(b, np.nan) for b in ["low", "mid", "high"] if b in row.index], dtype=float)
                ok = np.isfinite(ys)
                if np.count_nonzero(ok) >= 2:
                    ax.plot(xs[ok] + rng.normal(0, 0.015, size=np.count_nonzero(ok)), ys[ok], color=color, alpha=0.12, lw=0.8)
        sub = _summary_lookup(summary, metric)
        xs = np.asarray([x_map[str(b)] for b in sub["coherence_bracket"]], dtype=float)
        y = sub["session_median"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(xs, y, yerr=yerr, color=color, marker="o", lw=2.0, capsize=3, label=component)
    ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
    ax.set_xlabel("local orientation coherence")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    if symlog:
        ax.set_yscale("symlog", linthresh=1.0)
        ax.axhline(0, color="#555555", lw=0.8)
    elif tight_ylim:
        ci_values: list[float] = []
        for metric in metrics:
            sub = _summary_lookup(summary, metric)
            ci_values.extend(sub["ci95_low"].to_numpy(dtype=float).tolist())
            ci_values.extend(sub["ci95_high"].to_numpy(dtype=float).tolist())
        arr = np.asarray(ci_values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            pad = max(0.05, 0.12 * (hi - lo))
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _animal_diagnostic_summaries(df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[pd.DataFrame] = []
    contrast_rows: list[pd.DataFrame] = []
    for subject in sorted(df["subject"].dropna().unique()):
        sub = df[df["subject"].astype(str) == str(subject)].copy()
        if sub.empty:
            continue
        summary = _session_bracket_summary(sub, n_bootstrap=n_bootstrap, seed=seed + 1000 + len(summary_rows))
        summary.insert(0, "subject", str(subject))
        summary_rows.append(summary)
        contrasts = _paired_high_low_contrasts(sub, n_bootstrap=n_bootstrap, seed=seed + 2000 + len(contrast_rows))
        contrasts.insert(0, "subject", str(subject))
        contrast_rows.append(contrasts)
    return (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame(),
        pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(),
    )


def _plot_component_lines_by_subject(
    df: pd.DataFrame,
    summary_by_subject: pd.DataFrame,
    *,
    metrics: tuple[str, str],
    ylabel: str,
    title: str,
    out_path: Path,
    component_labels: tuple[str, str] = ("contour-parallel", "contour-normal"),
    tight_ylim: bool = True,
) -> None:
    subjects = sorted(df["subject"].dropna().unique())
    if not subjects:
        return
    fig, axes = plt.subplots(1, len(subjects), figsize=(4.0 * len(subjects), 3.2), sharey=True)
    axes_arr = np.atleast_1d(axes)
    x_map = {"low": 0, "mid": 1, "high": 2}
    global_ci: list[float] = []
    if tight_ylim:
        for metric in metrics:
            sub = summary_by_subject[summary_by_subject["metric"].astype(str) == metric]
            global_ci.extend(sub["ci95_low"].to_numpy(dtype=float).tolist())
            global_ci.extend(sub["ci95_high"].to_numpy(dtype=float).tolist())
    for ax, subject in zip(axes_arr, subjects, strict=True):
        sub_summary = summary_by_subject[summary_by_subject["subject"].astype(str) == str(subject)]
        for component, metric, color in [
            (component_labels[0], metrics[0], COLORS["along"]),
            (component_labels[1], metrics[1], COLORS["across"]),
        ]:
            sub = _summary_lookup(sub_summary, metric)
            xs = np.asarray([x_map[str(b)] for b in sub["coherence_bracket"]], dtype=float)
            y = sub["session_median"].to_numpy(dtype=float)
            lo = sub["ci95_low"].to_numpy(dtype=float)
            hi = sub["ci95_high"].to_numpy(dtype=float)
            ax.errorbar(xs, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker="o", lw=2.0, capsize=3, label=component)
        n_sessions = df[df["subject"].astype(str) == str(subject)]["session"].nunique()
        ax.set_title(f"{subject} (n={n_sessions})", loc="left", fontsize=10)
        ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
        ax.set_xlabel("orientation coherence")
        ax.grid(axis="y", color="#dddddd", lw=0.8)
    axes_arr[0].set_ylabel(ylabel)
    if tight_ylim:
        arr = np.asarray(global_ci, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            pad = max(0.05, 0.12 * (hi - lo))
            axes_arr[0].set_ylim(max(0.0, lo - pad), hi + pad)
    axes_arr[-1].legend(frameon=False, loc="best")
    fig.suptitle(title, x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_ratio_delta(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    specs = [
        ("rms_delta_along_minus_across_arcmin", "along - across RMS\narcmin"),
        ("rms_ratio_along_over_across", "along / across RMS"),
    ]
    x_map = {"low": 0, "mid": 1, "high": 2}
    for ax, (metric, ylabel) in zip(axes, specs, strict=True):
        sub = _summary_lookup(summary, metric)
        xs = np.asarray([x_map[str(b)] for b in sub["coherence_bracket"]], dtype=float)
        y = sub["session_median"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(xs, y, yerr=np.vstack([y - lo, hi - y]), color="#284b63", marker="o", lw=2, capsize=3)
        ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
        ax.set_xlabel("local orientation coherence")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=0.8)
        if "delta" in metric:
            ax.axhline(0, color="#555555", lw=0.8)
        else:
            ax.axhline(1, color="#555555", lw=0.8)
    fig.suptitle("Contour-parallel allocation", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _smooth_periodic(counts: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma_bins)) ** 2)
    kernel /= np.sum(kernel)
    padded = np.concatenate([counts[-radius:], counts, counts[:radius]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[radius:-radius]


def _plot_polar_alignment(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    bins = np.linspace(0.0, 2.0 * np.pi, 97)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    for bracket in ["low", "mid", "high"]:
        sub = df[df["coherence_bracket"].astype(str) == bracket].copy()
        delta = sub["drift_edge_delta_deg"].to_numpy(dtype=np.float64)
        density, stats = _polar_density_and_stats(delta, bins)
        if density is None:
            continue
        color = COLORS[bracket]
        label = f"{bracket}: cos2={stats['mean_cos2_delta']:.2f}"
        _draw_polar_density(ax, centers, density, color=color, label=label)
        rows.append({
            "coherence_bracket": bracket,
            "reference": False,
            "coherence_max": float("nan"),
            **stats,
        })
    ref = _no_coherence_subset(df, no_coherence_max)
    ref_density, ref_stats = _polar_density_and_stats(ref["drift_edge_delta_deg"].to_numpy(dtype=np.float64), bins)
    if ref_density is not None:
        label = f"no coh <= {float(no_coherence_max):.2f}"
        _draw_polar_density(
            ax,
            centers,
            ref_density,
            color=COLORS["no_coherence"],
            label=label,
            linestyle="--",
            linewidth=1.6,
            fill_alpha=0.0,
        )
        rows.append({
            "coherence_bracket": "no_coherence",
            "reference": True,
            "coherence_max": float(no_coherence_max),
            **ref_stats,
        })
    ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
    ax.set_yticklabels([])
    ax.set_title("Drift axis relative to local contour", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.04, 0.48), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_polar_alignment_by_subject(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    subjects = sorted(df["subject"].dropna().unique())
    if not subjects:
        return pd.DataFrame()
    bins = np.linspace(0.0, 2.0 * np.pi, 97)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=(4.5 * len(subjects), 4.0),
        subplot_kw={"projection": "polar"},
    )
    axes_arr = np.atleast_1d(axes)
    rows: list[dict[str, Any]] = []
    for ax, subject in zip(axes_arr, subjects, strict=True):
        subject_df = df[df["subject"].astype(str) == str(subject)].copy()
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        for bracket in ["low", "mid", "high"]:
            sub = subject_df[subject_df["coherence_bracket"].astype(str) == bracket].copy()
            delta = sub["drift_edge_delta_deg"].to_numpy(dtype=np.float64)
            density, stats = _polar_density_and_stats(delta, bins)
            if density is None:
                continue
            color = COLORS[bracket]
            _draw_polar_density(ax, centers, density, color=color, label=f"{bracket}: {stats['mean_cos2_delta']:.2f}")
            rows.append({
                "subject": str(subject),
                "coherence_bracket": bracket,
                "reference": False,
                "coherence_max": float("nan"),
                "n_sessions": int(sub["session"].nunique()),
                **stats,
            })
        ref = _no_coherence_subset(subject_df, no_coherence_max)
        ref_density, ref_stats = _polar_density_and_stats(ref["drift_edge_delta_deg"].to_numpy(dtype=np.float64), bins)
        if ref_density is not None:
            _draw_polar_density(
                ax,
                centers,
                ref_density,
                color=COLORS["no_coherence"],
                label=f"no coh <= {float(no_coherence_max):.2f}",
                linestyle="--",
                linewidth=1.6,
                fill_alpha=0.0,
            )
            rows.append({
                "subject": str(subject),
                "coherence_bracket": "no_coherence",
                "reference": True,
                "coherence_max": float(no_coherence_max),
                "n_sessions": int(ref["session"].nunique()),
                **ref_stats,
            })
        ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
        ax.set_yticklabels([])
        ax.set_title(f"{subject}: drift vs local contour", va="bottom", fontsize=10)
    axes_arr[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.48))
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_polar_rms_profile(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    centers = np.linspace(0.0, 2.0 * np.pi, 97)[:-1]
    rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    for bracket in ["low", "mid", "high"]:
        sub = df[df["coherence_bracket"].astype(str) == bracket].copy()
        profile, stats = _directional_rms_profile(sub, centers)
        if profile is None:
            continue
        label = f"{bracket}: ratio={stats['parallel_over_orthogonal_rms']:.2f}"
        _draw_polar_rms_profile(ax, centers, profile, color=COLORS[bracket], label=label)
        rows.append({
            "coherence_bracket": bracket,
            "reference": False,
            "coherence_max": float("nan"),
            **stats,
        })
    ref = _no_coherence_subset(df, no_coherence_max)
    ref_profile, ref_stats = _directional_rms_profile(ref, centers)
    if ref_profile is not None:
        _draw_polar_rms_profile(
            ax,
            centers,
            ref_profile,
            color=COLORS["no_coherence"],
            label=f"no coh <= {float(no_coherence_max):.2f}",
            linestyle="--",
            linewidth=1.6,
        )
        rows.append({
            "coherence_bracket": "no_coherence",
            "reference": True,
            "coherence_max": float(no_coherence_max),
            **ref_stats,
        })
    ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
    ax.set_title("Motion scale relative to local contour", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.04, 0.48), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_polar_rms_profile_by_subject(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    subjects = sorted(df["subject"].dropna().unique())
    if not subjects:
        return pd.DataFrame()
    centers = np.linspace(0.0, 2.0 * np.pi, 97)[:-1]
    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=(4.5 * len(subjects), 4.0),
        subplot_kw={"projection": "polar"},
    )
    axes_arr = np.atleast_1d(axes)
    rows: list[dict[str, Any]] = []
    for ax, subject in zip(axes_arr, subjects, strict=True):
        subject_df = df[df["subject"].astype(str) == str(subject)].copy()
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        for bracket in ["low", "mid", "high"]:
            sub = subject_df[subject_df["coherence_bracket"].astype(str) == bracket].copy()
            profile, stats = _directional_rms_profile(sub, centers)
            if profile is None:
                continue
            _draw_polar_rms_profile(
                ax,
                centers,
                profile,
                color=COLORS[bracket],
                label=f"{bracket}: {stats['parallel_over_orthogonal_rms']:.2f}",
            )
            rows.append({
                "subject": str(subject),
                "coherence_bracket": bracket,
                "reference": False,
                "coherence_max": float("nan"),
                "n_sessions": int(sub["session"].nunique()),
                **stats,
            })
        ref = _no_coherence_subset(subject_df, no_coherence_max)
        ref_profile, ref_stats = _directional_rms_profile(ref, centers)
        if ref_profile is not None:
            _draw_polar_rms_profile(
                ax,
                centers,
                ref_profile,
                color=COLORS["no_coherence"],
                label=f"no coh <= {float(no_coherence_max):.2f}",
                linestyle="--",
                linewidth=1.6,
            )
            rows.append({
                "subject": str(subject),
                "coherence_bracket": "no_coherence",
                "reference": True,
                "coherence_max": float(no_coherence_max),
                "n_sessions": int(ref["session"].nunique()),
                **ref_stats,
            })
        ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
        ax.set_title(f"{subject}: RMS vs local contour", va="bottom", fontsize=10)
    axes_arr[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.48))
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _rms_ratio_stats(stats: dict[str, Any], ref_stats: dict[str, Any]) -> dict[str, float]:
    par = float(stats["median_parallel_rms_arcmin"])
    orth = float(stats["median_orthogonal_rms_arcmin"])
    ref_par = float(ref_stats["median_parallel_rms_arcmin"])
    ref_orth = float(ref_stats["median_orthogonal_rms_arcmin"])
    par_ratio = par / ref_par if np.isfinite(par) and np.isfinite(ref_par) and ref_par > 0.0 else float("nan")
    orth_ratio = orth / ref_orth if np.isfinite(orth) and np.isfinite(ref_orth) and ref_orth > 0.0 else float("nan")
    return {
        "parallel_rms_ratio_to_no_coherence": float(par_ratio),
        "orthogonal_rms_ratio_to_no_coherence": float(orth_ratio),
        "orthogonal_minus_parallel_ratio_to_no_coherence": (
            float(orth_ratio - par_ratio) if np.isfinite(par_ratio) and np.isfinite(orth_ratio) else float("nan")
        ),
        "parallel_over_orthogonal_relative_rms": (
            float(par_ratio / orth_ratio) if np.isfinite(par_ratio) and np.isfinite(orth_ratio) and orth_ratio > 0.0 else float("nan")
        ),
    }


def _plot_polar_rms_ratio_profile(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    centers = np.linspace(0.0, 2.0 * np.pi, 97)[:-1]
    ref = _no_coherence_subset(df, no_coherence_max)
    ref_profile, ref_stats = _directional_rms_profile(ref, centers)
    if ref_profile is None:
        return pd.DataFrame()
    ok_ref = np.isfinite(ref_profile) & (ref_profile > 0.0)
    rows: list[dict[str, Any]] = []
    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    _draw_polar_rms_profile(
        ax,
        centers,
        np.ones_like(centers, dtype=np.float64),
        color=COMBINED_RMS_RATIO_COLORS["no_coherence"],
        label=f"no coh <= {float(no_coherence_max):.2f}",
        linestyle="--",
        linewidth=1.6,
    )
    rows.append({
        "coherence_bracket": "no_coherence",
        "reference": True,
        "coherence_max": float(no_coherence_max),
        **ref_stats,
        "parallel_rms_ratio_to_no_coherence": 1.0,
        "orthogonal_rms_ratio_to_no_coherence": 1.0,
        "orthogonal_minus_parallel_ratio_to_no_coherence": 0.0,
        "parallel_over_orthogonal_relative_rms": 1.0,
    })
    for bracket in ["low", "mid", "high"]:
        sub = df[df["coherence_bracket"].astype(str) == bracket].copy()
        profile, stats = _directional_rms_profile(sub, centers)
        if profile is None:
            continue
        ratio = np.full_like(profile, np.nan, dtype=np.float64)
        ok = ok_ref & np.isfinite(profile)
        ratio[ok] = profile[ok] / ref_profile[ok]
        ratio_stats = _rms_ratio_stats(stats, ref_stats)
        label = (
            f"{bracket}: par {ratio_stats['parallel_rms_ratio_to_no_coherence']:.2f}, "
            f"orth {ratio_stats['orthogonal_rms_ratio_to_no_coherence']:.2f}"
        )
        _draw_polar_rms_profile(ax, centers, ratio, color=COMBINED_RMS_RATIO_COLORS[bracket], label=label)
        rows.append({
            "coherence_bracket": bracket,
            "reference": False,
            "coherence_max": float("nan"),
            **stats,
            **ratio_stats,
        })
    ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
    ax.set_title("Motion scale / no-coherence baseline", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.04, 0.48), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_polar_rms_ratio_profile_by_subject(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    subjects = sorted(df["subject"].dropna().unique())
    if not subjects:
        return pd.DataFrame()
    centers = np.linspace(0.0, 2.0 * np.pi, 97)[:-1]
    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=(4.5 * len(subjects), 4.0),
        subplot_kw={"projection": "polar"},
    )
    axes_arr = np.atleast_1d(axes)
    rows: list[dict[str, Any]] = []
    for ax, subject in zip(axes_arr, subjects, strict=True):
        subject_df = df[df["subject"].astype(str) == str(subject)].copy()
        ref = _no_coherence_subset(subject_df, no_coherence_max)
        ref_profile, ref_stats = _directional_rms_profile(ref, centers)
        if ref_profile is None:
            continue
        ok_ref = np.isfinite(ref_profile) & (ref_profile > 0.0)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        _draw_polar_rms_profile(
            ax,
            centers,
            np.ones_like(centers, dtype=np.float64),
            color=COLORS["no_coherence"],
            label=f"no coh <= {float(no_coherence_max):.2f}",
            linestyle="--",
            linewidth=1.6,
        )
        rows.append({
            "subject": str(subject),
            "coherence_bracket": "no_coherence",
            "reference": True,
            "coherence_max": float(no_coherence_max),
            "n_sessions": int(ref["session"].nunique()),
            **ref_stats,
            "parallel_rms_ratio_to_no_coherence": 1.0,
            "orthogonal_rms_ratio_to_no_coherence": 1.0,
            "orthogonal_minus_parallel_ratio_to_no_coherence": 0.0,
            "parallel_over_orthogonal_relative_rms": 1.0,
        })
        for bracket in ["low", "mid", "high"]:
            sub = subject_df[subject_df["coherence_bracket"].astype(str) == bracket].copy()
            profile, stats = _directional_rms_profile(sub, centers)
            if profile is None:
                continue
            ratio = np.full_like(profile, np.nan, dtype=np.float64)
            ok = ok_ref & np.isfinite(profile)
            ratio[ok] = profile[ok] / ref_profile[ok]
            ratio_stats = _rms_ratio_stats(stats, ref_stats)
            label = (
                f"{bracket}: {ratio_stats['parallel_rms_ratio_to_no_coherence']:.2f}/"
                f"{ratio_stats['orthogonal_rms_ratio_to_no_coherence']:.2f}"
            )
            _draw_polar_rms_profile(ax, centers, ratio, color=COLORS[bracket], label=label)
            rows.append({
                "subject": str(subject),
                "coherence_bracket": bracket,
                "reference": False,
                "coherence_max": float("nan"),
                "n_sessions": int(sub["session"].nunique()),
                **stats,
                **ratio_stats,
            })
        ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
        ax.set_title(f"{subject}: RMS / no coherence", va="bottom", fontsize=10)
    axes_arr[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.48))
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_polar_rms_four_panel_by_subject(df: pd.DataFrame, out_path: Path, *, no_coherence_max: float | None) -> pd.DataFrame:
    subjects = sorted(df["subject"].dropna().unique())
    if not subjects:
        return pd.DataFrame()
    centers = np.linspace(0.0, 2.0 * np.pi, 97)[:-1]
    fig, axes = plt.subplots(
        2,
        len(subjects),
        figsize=(4.8 * len(subjects), 7.4),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    rows: list[dict[str, Any]] = []

    for col_idx, subject in enumerate(subjects):
        subject_df = df[df["subject"].astype(str) == str(subject)].copy()
        ref = _no_coherence_subset(subject_df, no_coherence_max)
        ref_profile, ref_stats = _directional_rms_profile(ref, centers)
        if ref_profile is None:
            continue
        ok_ref = np.isfinite(ref_profile) & (ref_profile > 0.0)

        ax_abs = axes[0, col_idx]
        ax_rel = axes[1, col_idx]
        for ax in (ax_abs, ax_rel):
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])

        _draw_polar_rms_profile(
            ax_abs,
            centers,
            ref_profile,
            color=COLORS["no_coherence"],
            label=f"no coh <= {float(no_coherence_max):.2f}",
            linestyle="--",
            linewidth=1.6,
        )
        _draw_polar_rms_profile(
            ax_rel,
            centers,
            np.ones_like(centers, dtype=np.float64),
            color=COLORS["no_coherence"],
            label=f"no coh <= {float(no_coherence_max):.2f}",
            linestyle="--",
            linewidth=1.6,
        )
        rows.append({
            "panel": "absolute_rms",
            "subject": str(subject),
            "coherence_bracket": "no_coherence",
            "reference": True,
            "coherence_max": float(no_coherence_max),
            "n_sessions": int(ref["session"].nunique()),
            **ref_stats,
        })
        rows.append({
            "panel": "ratio_to_no_coherence",
            "subject": str(subject),
            "coherence_bracket": "no_coherence",
            "reference": True,
            "coherence_max": float(no_coherence_max),
            "n_sessions": int(ref["session"].nunique()),
            **ref_stats,
            "parallel_rms_ratio_to_no_coherence": 1.0,
            "orthogonal_rms_ratio_to_no_coherence": 1.0,
            "orthogonal_minus_parallel_ratio_to_no_coherence": 0.0,
            "parallel_over_orthogonal_relative_rms": 1.0,
        })

        for bracket in ["low", "mid", "high"]:
            sub = subject_df[subject_df["coherence_bracket"].astype(str) == bracket].copy()
            profile, stats = _directional_rms_profile(sub, centers)
            if profile is None:
                continue
            ratio = np.full_like(profile, np.nan, dtype=np.float64)
            ok = ok_ref & np.isfinite(profile)
            ratio[ok] = profile[ok] / ref_profile[ok]
            ratio_stats = _rms_ratio_stats(stats, ref_stats)
            _draw_polar_rms_profile(
                ax_abs,
                centers,
                profile,
                color=COLORS[bracket],
                label=f"{bracket}: {stats['parallel_over_orthogonal_rms']:.2f}",
            )
            _draw_polar_rms_profile(
                ax_rel,
                centers,
                ratio,
                color=COLORS[bracket],
                label=(
                    f"{bracket}: {ratio_stats['parallel_rms_ratio_to_no_coherence']:.2f}/"
                    f"{ratio_stats['orthogonal_rms_ratio_to_no_coherence']:.2f}"
                ),
            )
            rows.append({
                "panel": "absolute_rms",
                "subject": str(subject),
                "coherence_bracket": bracket,
                "reference": False,
                "coherence_max": float("nan"),
                "n_sessions": int(sub["session"].nunique()),
                **stats,
            })
            rows.append({
                "panel": "ratio_to_no_coherence",
                "subject": str(subject),
                "coherence_bracket": bracket,
                "reference": False,
                "coherence_max": float("nan"),
                "n_sessions": int(sub["session"].nunique()),
                **stats,
                **ratio_stats,
            })

        ax_abs.set_title(f"{subject}: RMS anisotropy", va="bottom", fontsize=10)
        ax_rel.set_title(f"{subject}: RMS / no coherence", va="bottom", fontsize=10)

    axes[0, -1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.48))
    axes[1, -1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.48))
    fig.suptitle("Contour-relative eye-motion scale", x=0.06, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


CONDITION_METRICS = (
    ("rms_radius_arcmin", "position RMS (arcmin)"),
    ("step_median_arcmin", "median step (arcmin)"),
    ("within_segment_signed_d_median_arcmin2_s", "local MSD slope/4 (arcmin^2/s)"),
    ("ensemble_diffusion_arcmin2_s", "ensemble-onset D (arcmin^2/s)"),
)

CONDITION_CONTOUR_METRICS = (
    ("rms_along_arcmin", "contour-parallel RMS (arcmin)"),
    ("rms_across_arcmin", "contour-normal RMS (arcmin)"),
    ("rms_delta_along_minus_across_arcmin", "parallel - normal RMS (arcmin)"),
    ("rms_ratio_along_over_across", "parallel / normal RMS"),
    ("drift_edge_cos2", "motion-contour alignment cos2"),
)


@lru_cache(maxsize=128)
def _session_stimulus_eye_data(session_name: str, stimulus: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from DataYatesV1 import get_session

    subject, date = session_name.split("_", 1)
    session = get_session(subject, date)
    dset = _load_dict_dataset(Path(session.sess_dir) / "datasets" / f"{stimulus}.dset")
    eyepos = _as_numpy(dset["eyepos"]).astype(np.float64)
    trial_inds = _as_numpy(dset.covariates["trial_inds"]).reshape(-1).astype(int)
    if "dpi_valid" in dset.covariates:
        valid = _as_numpy(dset.covariates["dpi_valid"]).reshape(-1).astype(bool)
    elif "dfs" in dset.covariates:
        dfs = _as_numpy(dset.covariates["dfs"])
        valid = np.asarray(dfs).reshape(dfs.shape[0], -1).any(axis=1)
    else:
        valid = np.ones(eyepos.shape[0], dtype=bool)
    valid &= np.isfinite(eyepos).all(axis=1)
    valid &= (np.abs(eyepos[:, 0]) <= 12.0) & (np.abs(eyepos[:, 1]) <= 12.0)
    if str(stimulus).lower() == "fixrsvp":
        valid &= np.linalg.norm(eyepos, axis=1) <= 1.0
    return eyepos, trial_inds, valid


@lru_cache(maxsize=32)
def _session_stimulus_images(session_name: str, stimulus: str) -> np.ndarray:
    from DataYatesV1 import get_session

    subject, date = session_name.split("_", 1)
    session = get_session(subject, date)
    dset = _load_dict_dataset(Path(session.sess_dir) / "datasets" / f"{stimulus}.dset")
    return _as_numpy(dset["stim"])


def _padded_high_speed_step_mask(
    trace: np.ndarray,
    valid: np.ndarray,
    *,
    dt: float,
    cutoff_deg_s: float | None,
    pad_samples: int,
) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    mask = np.zeros(x.shape[0], dtype=bool)
    if cutoff_deg_s is None or not np.isfinite(cutoff_deg_s) or float(cutoff_deg_s) <= 0.0:
        return mask
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 2 or v.shape[0] != x.shape[0]:
        return mask
    finite = np.isfinite(x).all(axis=1)
    pair_valid = v[:-1] & v[1:] & finite[:-1] & finite[1:]
    if not np.any(pair_valid):
        return mask
    speed = np.linalg.norm(np.diff(x, axis=0), axis=1) / float(dt)
    high = pair_valid & np.isfinite(speed) & (speed > float(cutoff_deg_s))
    if not np.any(high):
        return mask
    mask[:-1] |= high
    mask[1:] |= high
    if pad_samples > 0:
        padded = mask.copy()
        for i in np.where(mask)[0]:
            lo = max(0, int(i) - int(pad_samples))
            hi = min(mask.size, int(i) + int(pad_samples) + 1)
            padded[lo:hi] = True
        mask = padded
    return mask


def _drift_segments_for_condition(
    session_name: str,
    stimulus: str,
    *,
    dt: float,
    discard_initial_s: float,
    saccade_pad_s: float,
    hard_speed_cutoff_deg_s: float | None,
    post_event_exclusion_s: float,
    min_segment_s: float,
) -> list[np.ndarray]:
    eyepos, trial_inds, valid_base = _session_stimulus_eye_data(str(session_name), str(stimulus))
    segments: list[np.ndarray] = []
    discard_samples = max(0, int(round(float(discard_initial_s) / float(dt))))
    pad_samples = max(0, int(round(float(saccade_pad_s) / float(dt))))
    post_event_samples = max(0, int(round(float(post_event_exclusion_s) / float(dt))))
    min_segment_samples = max(3, int(round(float(min_segment_s) / float(dt))))
    for trial_idx in np.unique(trial_inds):
        idx = np.where(trial_inds == int(trial_idx))[0]
        if idx.size < min_segment_samples:
            continue
        trace = eyepos[idx]
        valid = valid_base[idx].copy()
        if discard_samples > 0:
            valid[:min(discard_samples, valid.size)] = False
        if np.count_nonzero(valid) < min_segment_samples:
            continue
        threshold = _speed_threshold_mad_valid_pairs(trace, valid, dt=float(dt), z=6.0)
        _, event_mask, _ = detect_microsaccade_events(
            trace,
            dt=float(dt),
            threshold_deg_s=float(threshold),
            min_samples=1,
            pad_samples=pad_samples,
        )
        hard_speed_mask = _padded_high_speed_step_mask(
            trace,
            valid,
            dt=float(dt),
            cutoff_deg_s=hard_speed_cutoff_deg_s,
            pad_samples=pad_samples,
        )
        removed_event = event_mask | hard_speed_mask
        clean = valid & ~removed_event
        for start, stop in _contiguous_true_blocks(clean):
            trimmed_start = int(start)
            if post_event_samples > 0 and start > 0 and np.any(removed_event[max(0, start - 2):start]):
                trimmed_start = min(int(stop), int(start) + post_event_samples)
            if stop - trimmed_start >= min_segment_samples:
                segments.append(trace[trimmed_start:stop].copy())
    return segments


def _ensemble_displacement_diffusion(
    segments: list[np.ndarray],
    *,
    dt: float,
    max_lag_s: float,
    min_segments_per_lag: int,
) -> dict[str, float]:
    if not segments:
        return {
            "ensemble_diffusion_arcmin2_s": float("nan"),
            "ensemble_diffusion_slope_arcmin2_s": float("nan"),
            "ensemble_diffusion_intercept_arcmin2": float("nan"),
            "ensemble_diffusion_fit_r2": float("nan"),
            "ensemble_diffusion_n_segments": 0.0,
            "ensemble_diffusion_max_lag_s": float("nan"),
            "ensemble_diffusion_n_fit_lags": 0.0,
        }
    max_available = max(seg.shape[0] - 1 for seg in segments)
    max_lag = min(max_available, max(1, int(round(float(max_lag_s) / float(dt)))))
    times: list[float] = []
    variances: list[float] = []
    counts: list[int] = []
    for lag in range(1, max_lag + 1):
        disp = [seg[lag] - seg[0] for seg in segments if seg.shape[0] > lag and np.isfinite(seg[[0, lag]]).all()]
        if len(disp) < int(min_segments_per_lag):
            continue
        arr = np.asarray(disp, dtype=np.float64)
        var2 = float(np.var(arr[:, 0]) + np.var(arr[:, 1]))
        times.append(lag * float(dt))
        variances.append(var2 * 3600.0)
        counts.append(len(disp))
    slope, intercept, r2 = _fit_slope(np.asarray(times), np.asarray(variances))
    return {
        "ensemble_diffusion_arcmin2_s": slope / 4.0 if np.isfinite(slope) else float("nan"),
        "ensemble_diffusion_slope_arcmin2_s": slope,
        "ensemble_diffusion_intercept_arcmin2": intercept,
        "ensemble_diffusion_fit_r2": r2,
        "ensemble_diffusion_n_segments": float(len(segments)),
        "ensemble_diffusion_max_lag_s": float(np.max(times)) if times else float("nan"),
        "ensemble_diffusion_n_fit_lags": float(len(times)),
        "ensemble_diffusion_min_count_per_lag": float(min(counts)) if counts else float("nan"),
    }


def _within_segment_diffusion_summary(
    segments: list[np.ndarray],
    *,
    dt: float,
    max_lag_s: float,
) -> dict[str, float]:
    signed_d: list[float] = []
    fit_r2: list[float] = []
    durations: list[float] = []
    max_lags: list[float] = []
    for seg in segments:
        x = np.asarray(seg, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 4:
            continue
        max_lag = min(x.shape[0] - 1, max(1, int(round(float(max_lag_s) / float(dt)))))
        times: list[float] = []
        msd: list[float] = []
        for lag in range(1, max_lag + 1):
            disp = x[lag:] - x[:-lag]
            if disp.shape[0] < 2 or not np.isfinite(disp).all():
                continue
            times.append(lag * float(dt))
            msd.append(float(np.mean(np.sum(disp * disp, axis=1)) * 3600.0))
        slope, _intercept, r2 = _fit_slope(np.asarray(times), np.asarray(msd))
        if not np.isfinite(slope):
            continue
        signed_d.append(float(slope / 4.0))
        fit_r2.append(float(r2))
        durations.append(float(x.shape[0] * float(dt)))
        max_lags.append(float(max_lag * float(dt)))
    arr = np.asarray(signed_d, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    r2_arr = np.asarray(fit_r2, dtype=np.float64)
    r2_arr = r2_arr[np.isfinite(r2_arr)]
    dur_arr = np.asarray(durations, dtype=np.float64)
    lag_arr = np.asarray(max_lags, dtype=np.float64)
    if arr.size == 0:
        return {
            "within_segment_signed_d_median_arcmin2_s": float("nan"),
            "within_segment_positive_d_median_arcmin2_s": float("nan"),
            "within_segment_signed_d_p75_arcmin2_s": float("nan"),
            "within_segment_signed_d_p90_arcmin2_s": float("nan"),
            "within_segment_diffusion_n_segments": 0.0,
            "within_segment_diffusion_fit_r2_median": float("nan"),
            "within_segment_duration_median_s": float("nan"),
            "within_segment_diffusion_max_lag_median_s": float("nan"),
        }
    return {
        "within_segment_signed_d_median_arcmin2_s": float(np.median(arr)),
        "within_segment_positive_d_median_arcmin2_s": float(np.median(np.maximum(arr, 0.0))),
        "within_segment_signed_d_p75_arcmin2_s": float(np.percentile(arr, 75.0)),
        "within_segment_signed_d_p90_arcmin2_s": float(np.percentile(arr, 90.0)),
        "within_segment_diffusion_n_segments": float(arr.size),
        "within_segment_diffusion_fit_r2_median": float(np.median(r2_arr)) if r2_arr.size else float("nan"),
        "within_segment_duration_median_s": float(np.median(dur_arr)) if dur_arr.size else float("nan"),
        "within_segment_diffusion_max_lag_median_s": float(np.median(lag_arr)) if lag_arr.size else float("nan"),
    }


def _condition_diffusion_table(
    session_rows: pd.DataFrame,
    *,
    dt: float,
    discard_initial_s: float,
    saccade_pad_s: float,
    hard_speed_cutoff_deg_s: float | None,
    post_event_exclusion_s: float,
    min_segment_s: float,
    max_lag_s: float,
    within_segment_max_lag_s: float,
    min_segments_per_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = session_rows[["subject", "session", "stimulus"]].drop_duplicates()
    for rec in keys.to_dict("records"):
        segments = _drift_segments_for_condition(
            str(rec["session"]),
            str(rec["stimulus"]),
            dt=float(dt),
            discard_initial_s=float(discard_initial_s),
            saccade_pad_s=float(saccade_pad_s),
            hard_speed_cutoff_deg_s=hard_speed_cutoff_deg_s,
            post_event_exclusion_s=float(post_event_exclusion_s),
            min_segment_s=float(min_segment_s),
        )
        fit = _ensemble_displacement_diffusion(
            segments,
            dt=float(dt),
            max_lag_s=float(max_lag_s),
            min_segments_per_lag=int(min_segments_per_lag),
        )
        fit.update(_within_segment_diffusion_summary(
            segments,
            dt=float(dt),
            max_lag_s=float(within_segment_max_lag_s),
        ))
        rows.append({**rec, **fit})
    return pd.DataFrame(rows)


def _condition_session_table(
    input_path: Path,
    *,
    phases: list[str],
    dt: float,
    discard_initial_s: float,
    saccade_pad_s: float,
    hard_speed_cutoff_deg_s: float | None,
    post_event_exclusion_s: float,
    min_segment_s: float,
    max_lag_s: float,
    within_segment_max_lag_s: float,
    min_segments_per_lag: int,
) -> pd.DataFrame:
    msd_lags = [1, 2, 4, 8, 16]
    msd_cols = [f"msd_lag{lag}_deg2" for lag in msd_lags]
    needed = [
        "session",
        "stimulus",
        "phase",
        "rms_radius_deg",
        "step_median_deg",
        "diffusion_constant_deg2_s",
    ] + msd_cols
    df = pd.read_csv(input_path, usecols=needed)
    df = df[df["stimulus"].astype(str).isin(["fixrsvp", "backimage"])].copy()
    if phases:
        df = df[df["phase"].astype(str).isin(phases)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["rms_radius_deg", "step_median_deg", "diffusion_constant_deg2_s"])
    df["subject"] = df["session"].map(_subject_from_session)
    df["rms_radius_arcmin"] = df["rms_radius_deg"].astype(float) * 60.0
    df["step_median_arcmin"] = df["step_median_deg"].astype(float) * 60.0
    df["diffusion_constant_arcmin2_s"] = df["diffusion_constant_deg2_s"].astype(float) * 3600.0
    lag_t = np.asarray(msd_lags, dtype=np.float64) / 120.0
    signed_slopes: list[float] = []
    for row in df[msd_cols].itertuples(index=False, name=None):
        y = np.asarray(row, dtype=np.float64)
        slope, _, _ = _fit_slope(lag_t, y)
        signed_slopes.append(slope)
    df["signed_msd_slope_arcmin2_s"] = np.asarray(signed_slopes, dtype=np.float64) * 3600.0
    df["signed_msd_d_eff_arcmin2_s"] = df["signed_msd_slope_arcmin2_s"] / 4.0
    session = (
        df.groupby(["subject", "session", "stimulus"], observed=True)[
            ["rms_radius_arcmin", "step_median_arcmin", "diffusion_constant_arcmin2_s", "signed_msd_d_eff_arcmin2_s", "signed_msd_slope_arcmin2_s"]
        ]
        .median(numeric_only=True)
        .reset_index()
    )
    diffusion = _condition_diffusion_table(
        session,
        dt=float(dt),
        discard_initial_s=float(discard_initial_s),
        saccade_pad_s=float(saccade_pad_s),
        hard_speed_cutoff_deg_s=hard_speed_cutoff_deg_s,
        post_event_exclusion_s=float(post_event_exclusion_s),
        min_segment_s=float(min_segment_s),
        max_lag_s=float(max_lag_s),
        within_segment_max_lag_s=float(within_segment_max_lag_s),
        min_segments_per_lag=int(min_segments_per_lag),
    )
    return session.merge(diffusion, on=["subject", "session", "stimulus"], how="left")


def _condition_summary(session: pd.DataFrame, *, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for subject in sorted(session["subject"].dropna().unique()):
        sub = session[session["subject"].astype(str) == str(subject)]
        for stimulus in ["fixrsvp", "backimage"]:
            stim = sub[sub["stimulus"].astype(str) == stimulus]
            for metric, _ in CONDITION_METRICS:
                point, lo, hi = _bootstrap_ci(
                    stim[metric].to_numpy(dtype=np.float64),
                    n_bootstrap=n_bootstrap,
                    seed=seed + 3000 + len(summary_rows),
                )
                summary_rows.append({
                    "subject": str(subject),
                    "stimulus": stimulus,
                    "metric": metric,
                    "session_median": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_sessions": int(stim["session"].nunique()),
                })
        for metric, _ in CONDITION_METRICS:
            wide = sub.pivot(index="session", columns="stimulus", values=metric)
            if "backimage" not in wide.columns or "fixrsvp" not in wide.columns:
                continue
            delta = (wide["backimage"] - wide["fixrsvp"]).to_numpy(dtype=np.float64)
            point, lo, hi = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed + 4000 + len(contrast_rows))
            contrast_rows.append({
                "subject": str(subject),
                "metric": metric,
                "contrast": "backimage_minus_fixrsvp_session_paired",
                "median_delta": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_sessions": int(np.count_nonzero(np.isfinite(delta))),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def _plot_condition_comparison(
    session: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    *,
    show_session_lines: bool = False,
    tight_ylim: bool = True,
) -> None:
    fig, axes = plt.subplots(1, len(CONDITION_METRICS), figsize=(4.0 * len(CONDITION_METRICS), 3.25))
    axes_arr = np.atleast_1d(axes)
    x_map = {"fixrsvp": 0.0, "backimage": 1.0}
    offsets = {"Allen": -0.045, "Logan": 0.045}
    for ax, (metric, ylabel) in zip(axes_arr, CONDITION_METRICS, strict=True):
        for subject in sorted(session["subject"].dropna().unique()):
            color = COLORS.get(str(subject), "#444444")
            sub = session[session["subject"].astype(str) == str(subject)]
            wide = sub.pivot(index="session", columns="stimulus", values=metric)
            if show_session_lines:
                for _, row in wide.iterrows():
                    ys = np.asarray([row.get("fixrsvp", np.nan), row.get("backimage", np.nan)], dtype=float)
                    if np.isfinite(ys).all():
                        xs = np.asarray([x_map["fixrsvp"], x_map["backimage"]], dtype=float) + offsets.get(str(subject), 0.0)
                        ax.plot(xs, ys, color=color, alpha=0.18, lw=0.8)
            for stimulus in ["fixrsvp", "backimage"]:
                rec = summary[
                    (summary["subject"].astype(str) == str(subject))
                    & (summary["stimulus"].astype(str) == stimulus)
                    & (summary["metric"].astype(str) == metric)
                ]
                if rec.empty:
                    continue
                y = float(rec["session_median"].iloc[0])
                lo = float(rec["ci95_low"].iloc[0])
                hi = float(rec["ci95_high"].iloc[0])
                x = x_map[stimulus] + offsets.get(str(subject), 0.0)
                ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], color=color, marker="o", ms=5, lw=2, capsize=3, label=str(subject) if stimulus == "fixrsvp" else None)
        ax.set_xticks([0, 1], ["FixRSVP", "BackImage"])
        ax.set_ylabel(ylabel)
        if tight_ylim:
            sub_summary = summary[summary["metric"].astype(str) == metric]
            arr = np.asarray(
                sub_summary["ci95_low"].to_numpy(dtype=float).tolist()
                + sub_summary["ci95_high"].to_numpy(dtype=float).tolist(),
                dtype=float,
            )
            arr = arr[np.isfinite(arr)]
            if arr.size:
                lo = float(np.min(arr))
                hi = float(np.max(arr))
                pad = max(0.05, 0.14 * (hi - lo))
                lower = lo - pad if lo < 0 else max(0.0, lo - pad)
                ax.set_ylim(lower, hi + pad)
        if "signed" in str(metric):
            ax.axhline(0.0, color="#555555", lw=0.8)
        ax.grid(axis="y", color="#dddddd", lw=0.8)
    axes_arr[0].legend(frameon=False, loc="best")
    fig.suptitle("Event-gated fixation motion scale by stimulus", x=0.04, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _fixrsvp_contour_windows(input_path: Path, *, phases: list[str]) -> pd.DataFrame:
    needed = [
        "session",
        "stimulus",
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "phase",
        "n_samples",
        "duration_s",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "rms_radius_deg",
        "step_median_deg",
    ]
    df = pd.read_csv(input_path, usecols=needed)
    df = df[df["stimulus"].astype(str) == "fixrsvp"].copy()
    if phases:
        df = df[df["phase"].astype(str).isin(phases)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cov_xx_deg2", "cov_xy_deg2", "cov_yy_deg2"])
    if df.empty:
        return df

    feature_rows: list[dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        rec = row._asdict()
        session_name = str(rec["session"])
        try:
            stim = _session_stimulus_images(session_name, "fixrsvp")
            center_idx = int((int(rec["global_start"]) + int(rec["global_stop"]) - 1) // 2)
            if center_idx < 0 or center_idx >= stim.shape[0]:
                raise IndexError(f"center index {center_idx} outside stimulus array length {stim.shape[0]}")
            feats = _patch_orientation_features(stim[center_idx])
            feats["image_patch_center_global_index"] = center_idx
        except Exception as exc:
            feats = {
                "image_feature_ok": False,
                "image_feature_error": str(exc),
                "image_patch_center_global_index": float("nan"),
            }
        feature_rows.append(feats)
        if i % 1000 == 0:
            print(f"computed FixRSVP local contour axes for {i}/{len(df)} windows")

    features = pd.DataFrame(feature_rows, index=df.index)
    out = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out["subject"] = out["session"].map(_subject_from_session)
    out["contour_axis_source"] = "fixrsvp_center_stim_crop"
    out = out[out["image_feature_ok"].astype(bool)].copy()
    out = out.dropna(subset=["image_orientation_coherence", "image_edge_axis_deg"])
    out = _project_covariance_components(out)
    return out


def _backimage_contour_windows_for_condition(backimage_df: pd.DataFrame, *, phases: list[str]) -> pd.DataFrame:
    cols = [
        "session",
        "stimulus",
        "trial_idx",
        "global_start",
        "global_stop",
        "local_start",
        "local_stop",
        "phase",
        "n_samples",
        "duration_s",
        "cov_xx_deg2",
        "cov_xy_deg2",
        "cov_yy_deg2",
        "rms_radius_deg",
        "step_median_deg",
        "subject",
        "image_feature_ok",
        "image_feature_error",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "image_gradient_energy",
        "image_edge_density",
        "image_patch_mean",
        "image_patch_std",
        "rms_along_arcmin",
        "rms_across_arcmin",
        "rms_ratio_along_over_across",
        "rms_delta_along_minus_across_arcmin",
        "drift_orientation_deg",
        "drift_edge_delta_deg",
        "drift_edge_cos2",
    ]
    keep = [col for col in cols if col in backimage_df.columns]
    out = backimage_df[keep].copy()
    out = out[out["stimulus"].astype(str) == "backimage"].copy()
    if phases:
        out = out[out["phase"].astype(str).isin(phases)].copy()
    out["subject"] = out["session"].map(_subject_from_session)
    out["contour_axis_source"] = "backimage_static_canvas_mean_gaze"
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["cov_xx_deg2", "cov_xy_deg2", "cov_yy_deg2", "image_orientation_coherence", "image_edge_axis_deg"])
    if "rms_along_arcmin" not in out.columns or "rms_across_arcmin" not in out.columns:
        out = _project_covariance_components(out)
    return out


def _assign_stimulus_coherence_brackets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    labels = pd.Series(index=out.index, dtype=object)
    rows: list[dict[str, Any]] = []
    for stimulus in ["fixrsvp", "backimage"]:
        idx = out["stimulus"].astype(str) == stimulus
        vals = out.loc[idx, "image_orientation_coherence"].astype(float)
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        if vals.size < 3:
            cats = pd.Series("mid", index=vals.index, dtype=object)
        else:
            ranks = vals.rank(method="first")
            cats = pd.Series(pd.qcut(ranks, 3, labels=["low", "mid", "high"]).astype(str), index=vals.index)
        labels.loc[cats.index] = cats
        for bracket in ["low", "mid", "high"]:
            sub = vals[cats == bracket]
            if sub.empty:
                continue
            rows.append({
                "stimulus": stimulus,
                "coherence_bracket": bracket,
                "n_windows": int(sub.size),
                "coherence_min": float(sub.min()),
                "coherence_median": float(sub.median()),
                "coherence_max": float(sub.max()),
            })
    out["coherence_bracket"] = pd.Categorical(labels, categories=["low", "mid", "high"], ordered=True)
    out = out.dropna(subset=["coherence_bracket"]).copy()
    return out, pd.DataFrame(rows)


def _condition_contour_window_table(
    backimage_df: pd.DataFrame,
    input_path: Path,
    *,
    phases: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backimage = _backimage_contour_windows_for_condition(backimage_df, phases=phases)
    fixrsvp = _fixrsvp_contour_windows(input_path, phases=phases)
    cols = sorted(set(backimage.columns).union(fixrsvp.columns))
    combined = pd.concat([backimage.reindex(columns=cols), fixrsvp.reindex(columns=cols)], ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=["rms_along_arcmin", "rms_across_arcmin", "image_orientation_coherence", "image_edge_axis_deg"])
    combined, bracket_info = _assign_stimulus_coherence_brackets(combined)
    return combined, bracket_info


def _condition_contour_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric, _ in CONDITION_CONTOUR_METRICS] + ["image_orientation_coherence"]
    return (
        df.groupby(["subject", "session", "stimulus", "coherence_bracket"], observed=True)[metrics]
        .median(numeric_only=True)
        .reset_index()
    )


def _condition_contour_summary(
    df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    by_subject: bool = False,
) -> pd.DataFrame:
    session_rows = _condition_contour_session_rows(df)
    group_cols = ["stimulus", "coherence_bracket"]
    if by_subject:
        group_cols = ["subject"] + group_cols
    metrics = [metric for metric, _ in CONDITION_CONTOUR_METRICS] + ["image_orientation_coherence"]
    rows: list[dict[str, Any]] = []
    for key, sub in session_rows.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec_base = dict(zip(group_cols, key, strict=True))
        for metric in metrics:
            point, lo, hi = _bootstrap_ci(
                sub[metric].to_numpy(dtype=np.float64),
                n_bootstrap=n_bootstrap,
                seed=seed + 5000 + 37 * (len(rows) + 1),
            )
            rows.append({
                **rec_base,
                "metric": metric,
                "session_median": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_sessions": int(sub["session"].nunique()),
                "n_window_session_bins": int(sub.shape[0]),
            })
    return pd.DataFrame(rows)


def _condition_contour_high_low_contrasts(
    df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    by_subject: bool = False,
) -> pd.DataFrame:
    session_rows = _condition_contour_session_rows(df)
    metrics = [metric for metric, _ in CONDITION_CONTOUR_METRICS]
    group_cols = ["stimulus"]
    if by_subject:
        group_cols = ["subject", "stimulus"]
    rows: list[dict[str, Any]] = []
    for key, sub in session_rows.groupby(group_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec_base = dict(zip(group_cols, key, strict=True))
        for metric in metrics:
            wide = sub.pivot(index="session", columns="coherence_bracket", values=metric)
            if "high" not in wide.columns or "low" not in wide.columns:
                continue
            delta = (wide["high"] - wide["low"]).to_numpy(dtype=np.float64)
            point, lo, hi = _bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed + 6000 + len(rows))
            rows.append({
                **rec_base,
                "metric": metric,
                "contrast": "high_minus_low_session_paired",
                "median_delta": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_sessions": int(np.count_nonzero(np.isfinite(delta))),
            })
    return pd.DataFrame(rows)


def _condition_contour_summary_lookup(summary: pd.DataFrame, *, stimulus: str, metric: str, subject: str | None = None) -> pd.DataFrame:
    sub = summary[
        (summary["stimulus"].astype(str) == str(stimulus))
        & (summary["metric"].astype(str) == str(metric))
    ].copy()
    if subject is not None and "subject" in sub.columns:
        sub = sub[sub["subject"].astype(str) == str(subject)].copy()
    sub["coherence_bracket"] = pd.Categorical(sub["coherence_bracket"], categories=["low", "mid", "high"], ordered=True)
    return sub.sort_values("coherence_bracket")


def _plot_condition_contour_component_lines(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    x_map = {"low": 0, "mid": 1, "high": 2}
    for ax, stimulus in zip(np.atleast_1d(axes), ["fixrsvp", "backimage"], strict=True):
        for label, metric, color in [
            ("contour-parallel", "rms_along_arcmin", COLORS["along"]),
            ("contour-normal", "rms_across_arcmin", COLORS["across"]),
        ]:
            sub = _condition_contour_summary_lookup(summary, stimulus=stimulus, metric=metric)
            if sub.empty:
                continue
            xs = np.asarray([x_map[str(b)] for b in sub["coherence_bracket"]], dtype=float)
            y = sub["session_median"].to_numpy(dtype=float)
            lo = sub["ci95_low"].to_numpy(dtype=float)
            hi = sub["ci95_high"].to_numpy(dtype=float)
            ax.errorbar(xs, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker="o", lw=2.0, capsize=3, label=label)
        ax.set_title("FixRSVP" if stimulus == "fixrsvp" else "BackImage", loc="left", fontsize=10)
        ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
        ax.set_xlabel("within-stimulus contour coherence")
        ax.grid(axis="y", color="#dddddd", lw=0.8)
    axes[0].set_ylabel("position-cloud RMS (arcmin)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Contour-relative FEM scale by stimulus", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_condition_contour_component_lines_by_subject(summary: pd.DataFrame, out_path: Path) -> None:
    if "subject" not in summary.columns:
        return
    subjects = sorted(summary["subject"].dropna().astype(str).unique())
    if not subjects:
        return
    fig, axes = plt.subplots(2, len(subjects), figsize=(4.1 * len(subjects), 5.6), sharey=True, squeeze=False)
    x_map = {"low": 0, "mid": 1, "high": 2}
    for row_idx, stimulus in enumerate(["fixrsvp", "backimage"]):
        for col_idx, subject in enumerate(subjects):
            ax = axes[row_idx, col_idx]
            for label, metric, color in [
                ("contour-parallel", "rms_along_arcmin", COLORS["along"]),
                ("contour-normal", "rms_across_arcmin", COLORS["across"]),
            ]:
                sub = _condition_contour_summary_lookup(summary, stimulus=stimulus, metric=metric, subject=subject)
                if sub.empty:
                    continue
                xs = np.asarray([x_map[str(b)] for b in sub["coherence_bracket"]], dtype=float)
                y = sub["session_median"].to_numpy(dtype=float)
                lo = sub["ci95_low"].to_numpy(dtype=float)
                hi = sub["ci95_high"].to_numpy(dtype=float)
                ax.errorbar(xs, y, yerr=np.vstack([y - lo, hi - y]), color=color, marker="o", lw=2.0, capsize=3, label=label)
            ax.set_title(f"{subject} / {'FixRSVP' if stimulus == 'fixrsvp' else 'BackImage'}", loc="left", fontsize=10)
            ax.set_xticks([0, 1, 2], ["low", "mid", "high"])
            ax.grid(axis="y", color="#dddddd", lw=0.8)
            if row_idx == 1:
                ax.set_xlabel("within-stimulus coherence")
            if col_idx == 0:
                ax.set_ylabel("RMS (arcmin)")
    axes[0, -1].legend(frameon=False, loc="best")
    fig.suptitle("Contour-relative FEM scale by animal", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_condition_along_across_density(
    df: pd.DataFrame,
    out_path: Path,
    *,
    limit_arcmin: float | None = 4.0,
) -> None:
    arr = df[["rms_across_arcmin", "rms_along_arcmin"]].to_numpy(dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if limit_arcmin is None:
        limit = float(np.nanpercentile(arr, 98.0)) if arr.size else 4.0
        limit = max(2.5, min(8.0, 1.15 * limit))
    else:
        limit = float(limit_arcmin)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8), sharex=True, sharey=True)
    bins = np.linspace(0.0, limit, 64)
    xcent = 0.5 * (bins[:-1] + bins[1:])
    ycent = 0.5 * (bins[:-1] + bins[1:])
    xx, yy = np.meshgrid(xcent, ycent)
    for row_idx, stimulus in enumerate(["fixrsvp", "backimage"]):
        for col_idx, bracket in enumerate(["low", "mid", "high"]):
            ax = axes[row_idx, col_idx]
            sub = df[(df["stimulus"].astype(str) == stimulus) & (df["coherence_bracket"].astype(str) == bracket)]
            x = sub["rms_across_arcmin"].to_numpy(dtype=float)
            y = sub["rms_along_arcmin"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= limit) & (y <= limit)
            x = x[ok]
            y = y[ok]
            color = COLORS.get(bracket, "#444444")
            if x.size >= 12:
                hist, _, _ = np.histogram2d(x, y, bins=[bins, bins])
                density = ndimage.gaussian_filter(hist.T, sigma=1.0)
                positive = density[density > 0]
                if positive.size >= 4:
                    levels = np.unique(np.quantile(positive, [0.45, 0.68, 0.86, 0.95]))
                    if levels.size >= 2:
                        ax.contourf(xx, yy, density, levels=levels, colors=[color], alpha=0.10)
                        ax.contour(xx, yy, density, levels=levels, colors=[color], linewidths=1.2)
                ax.scatter([np.median(x)], [np.median(y)], s=30, color=color, edgecolor="white", linewidth=0.7, zorder=4)
            ax.plot([0, limit], [0, limit], color="#888888", lw=0.8, ls="--")
            ax.set_title(
                f"{'FixRSVP' if stimulus == 'fixrsvp' else 'BackImage'} / {bracket}\nshown={x.size}/{sub.shape[0]}",
                loc="left",
                fontsize=9,
            )
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.grid(color="#e1e1e1", lw=0.7)
            if row_idx == 1:
                ax.set_xlabel("contour-normal RMS (arcmin)")
            if col_idx == 0:
                ax.set_ylabel("contour-parallel RMS (arcmin)")
    fig.suptitle("Along/across central window-density contours", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_condition_polar_alignment_by_stimulus(
    df: pd.DataFrame,
    out_path: Path,
    *,
    no_coherence_max: float | None,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 2.0 * np.pi, 97)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1), subplot_kw={"projection": "polar"})
    rows: list[dict[str, Any]] = []
    for ax, stimulus in zip(np.atleast_1d(axes), ["fixrsvp", "backimage"], strict=True):
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        for bracket in ["low", "mid", "high"]:
            sub = df[(df["stimulus"].astype(str) == stimulus) & (df["coherence_bracket"].astype(str) == bracket)].copy()
            delta = sub["drift_edge_delta_deg"].to_numpy(dtype=np.float64)
            density, stats = _polar_density_and_stats(delta, bins)
            if density is None:
                continue
            color = COLORS[bracket]
            _draw_polar_density(ax, centers, density, color=color, label=f"{bracket}: {stats['mean_cos2_delta']:.2f}")
            rows.append({
                "stimulus": stimulus,
                "coherence_bracket": bracket,
                "reference": False,
                "coherence_max": float("nan"),
                "n_sessions": int(sub["session"].nunique()),
                **stats,
            })
        ref = _no_coherence_subset(df[df["stimulus"].astype(str) == stimulus], no_coherence_max)
        ref_density, ref_stats = _polar_density_and_stats(ref["drift_edge_delta_deg"].to_numpy(dtype=np.float64), bins)
        if ref_density is not None:
            _draw_polar_density(
                ax,
                centers,
                ref_density,
                color=COLORS["no_coherence"],
                label=f"no coh <= {float(no_coherence_max):.2f}",
                linestyle="--",
                linewidth=1.6,
                fill_alpha=0.0,
            )
            rows.append({
                "stimulus": stimulus,
                "coherence_bracket": "no_coherence",
                "reference": True,
                "coherence_max": float(no_coherence_max),
                "n_sessions": int(ref["session"].nunique()),
                **ref_stats,
            })
        ax.set_thetagrids([0, 90, 180, 270], ["parallel", "orthogonal", "parallel", "orthogonal"])
        ax.set_yticklabels([])
        ax.set_title("FixRSVP" if stimulus == "fixrsvp" else "BackImage", va="bottom", fontsize=10)
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.48))
    fig.suptitle("Motion axis relative to local contour", x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def load_component_table(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input_windows)
    required = [
        "session",
        "trial_idx",
        "local_start",
        "local_stop",
        "global_start",
        "global_stop",
        "image_orientation_coherence",
        "image_edge_axis_deg",
        "drift_orientation_deg",
        "drift_edge_delta_deg",
        "drift_edge_cos2",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input_windows}: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    df["subject"] = df["session"].map(_subject_from_session)
    df["coherence_bracket"], _ = _coherence_brackets(df["image_orientation_coherence"])

    if not bool(args.recompute_traces):
        return _project_covariance_components(df)

    feature_rows: list[dict[str, float]] = []
    lags = [int(v) for v in args.msd_lags]
    for i, row in enumerate(df.itertuples(index=False), start=1):
        series = pd.Series(row._asdict())
        trace = _window_trace(series)
        feats = projected_motion_features(
            trace,
            edge_axis_deg=float(series["image_edge_axis_deg"]),
            dt=float(args.dt),
            msd_lags=lags,
        )
        feature_rows.append(feats)
        if i % 2000 == 0:
            print(f"processed {i}/{len(df)} windows")
    features = pd.DataFrame(feature_rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-windows", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--condition-window-features", type=Path, default=DEFAULT_CONDITION_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--msd-lags", default=",".join(str(v) for v in DEFAULT_MSD_LAGS))
    parser.add_argument("--condition-phases", default="no_recent_event,mid_fixation,late_fixation")
    parser.add_argument("--condition-discard-initial-s", type=float, default=0.20)
    parser.add_argument("--condition-saccade-pad-s", type=float, default=0.050)
    parser.add_argument(
        "--condition-hard-speed-cutoff-deg-s",
        type=float,
        default=10.0,
        help="Additional absolute speed gate for drift diffusion; set <=0 to disable.",
    )
    parser.add_argument(
        "--condition-post-event-exclusion-s",
        type=float,
        default=0.10,
        help="Discard this much clean trace immediately after each detected/gated event before ensemble diffusion.",
    )
    parser.add_argument("--condition-min-segment-s", type=float, default=0.10)
    parser.add_argument("--condition-max-diffusion-lag-s", type=float, default=0.80)
    parser.add_argument("--condition-within-segment-max-diffusion-lag-s", type=float, default=0.20)
    parser.add_argument("--condition-min-segments-per-lag", type=int, default=20)
    parser.add_argument(
        "--no-coherence-max",
        type=float,
        default=DEFAULT_NO_COHERENCE_MAX,
        help=(
            "Absolute image_orientation_coherence cutoff for the dashed no-coherence reference "
            "in polar alignment plots. Set <=0 to disable."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute-traces", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> Path:
    args.msd_lags = [int(v) for v in parse_csv_list(str(args.msd_lags))]
    condition_phases = parse_csv_list(str(args.condition_phases))
    hard_speed_cutoff = float(args.condition_hard_speed_cutoff_deg_s)
    hard_speed_cutoff_or_none = hard_speed_cutoff if np.isfinite(hard_speed_cutoff) and hard_speed_cutoff > 0.0 else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_component_table(args)
    df.to_csv(out_dir / "contour_motion_component_windows.csv", index=False)
    summary = _session_bracket_summary(df, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    contrasts = _paired_high_low_contrasts(df, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    animal_summary, animal_contrasts = _animal_diagnostic_summaries(df, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    summary.to_csv(out_dir / "contour_motion_component_summary.csv", index=False)
    contrasts.to_csv(out_dir / "contour_motion_component_high_low_contrasts.csv", index=False)
    animal_summary.to_csv(out_dir / "contour_motion_component_summary_by_animal.csv", index=False)
    animal_contrasts.to_csv(out_dir / "contour_motion_component_high_low_contrasts_by_animal.csv", index=False)

    _plot_component_lines(
        df,
        summary,
        metrics=("rms_along_arcmin", "rms_across_arcmin"),
        ylabel="position-cloud RMS (arcmin)",
        title="Contour-relative FEM scale",
        out_path=out_dir / "candidate1_along_across_rms_by_coherence",
        show_session_lines=False,
        component_labels=("contour-parallel", "contour-normal"),
        tight_ylim=True,
    )
    _plot_component_lines(
        df,
        summary,
        metrics=("rms_along_arcmin", "rms_across_arcmin"),
        ylabel="position-cloud RMS (arcmin)",
        title="Contour-relative FEM scale",
        out_path=out_dir / "supplement_session_lines_along_across_rms_by_coherence",
        component_labels=("contour-parallel", "contour-normal"),
    )
    _plot_component_lines_by_subject(
        df,
        animal_summary,
        metrics=("rms_along_arcmin", "rms_across_arcmin"),
        ylabel="position-cloud RMS (arcmin)",
        title="Contour-relative FEM scale by animal",
        out_path=out_dir / "diagnostic_animal_along_across_rms_by_coherence",
    )
    _plot_component_lines(
        df,
        summary,
        metrics=("diffusion_along_arcmin2_s", "diffusion_across_arcmin2_s"),
        ylabel="projected diffusion D (arcmin^2/s)",
        title="Projected MSD-slope diffusion",
        out_path=out_dir / "candidate2_along_across_diffusion_by_coherence",
        show_session_lines=False,
    )
    _plot_component_lines(
        df,
        summary,
        metrics=("msd_slope_along_arcmin2_s", "msd_slope_across_arcmin2_s"),
        ylabel="signed projected MSD slope (arcmin^2/s)",
        title="Projected MSD slope diagnostic",
        out_path=out_dir / "candidate3_signed_msd_slope_by_coherence",
        symlog=True,
        show_session_lines=False,
    )
    _plot_ratio_delta(df, summary, out_dir / "candidate4_rms_allocation_ratio_delta")
    no_coherence_max = float(args.no_coherence_max)
    polar = _plot_polar_alignment(
        df,
        out_dir / "candidate5_polar_alignment_by_coherence",
        no_coherence_max=no_coherence_max,
    )
    polar.to_csv(out_dir / "polar_alignment_by_coherence.csv", index=False)
    animal_polar = _plot_polar_alignment_by_subject(
        df,
        out_dir / "diagnostic_animal_polar_alignment_by_coherence",
        no_coherence_max=no_coherence_max,
    )
    animal_polar.to_csv(out_dir / "polar_alignment_by_coherence_by_animal.csv", index=False)
    polar_rms = _plot_polar_rms_profile(
        df,
        out_dir / "candidate5b_polar_rms_profile_by_coherence",
        no_coherence_max=no_coherence_max,
    )
    polar_rms.to_csv(out_dir / "polar_rms_profile_by_coherence.csv", index=False)
    animal_polar_rms = _plot_polar_rms_profile_by_subject(
        df,
        out_dir / "diagnostic_animal_polar_rms_profile_by_coherence",
        no_coherence_max=no_coherence_max,
    )
    animal_polar_rms.to_csv(out_dir / "polar_rms_profile_by_coherence_by_animal.csv", index=False)
    polar_rms_ratio = _plot_polar_rms_ratio_profile(
        df,
        out_dir / "candidate5c_polar_rms_ratio_to_no_coherence",
        no_coherence_max=no_coherence_max,
    )
    polar_rms_ratio.to_csv(out_dir / "polar_rms_ratio_to_no_coherence.csv", index=False)
    animal_polar_rms_ratio = _plot_polar_rms_ratio_profile_by_subject(
        df,
        out_dir / "diagnostic_animal_polar_rms_ratio_to_no_coherence",
        no_coherence_max=no_coherence_max,
    )
    animal_polar_rms_ratio.to_csv(out_dir / "polar_rms_ratio_to_no_coherence_by_animal.csv", index=False)
    animal_polar_rms_four_panel = _plot_polar_rms_four_panel_by_subject(
        df,
        out_dir / "diagnostic_animal_polar_rms_four_panel",
        no_coherence_max=no_coherence_max,
    )
    animal_polar_rms_four_panel.to_csv(out_dir / "polar_rms_four_panel_by_animal.csv", index=False)

    condition_contour, condition_contour_brackets = _condition_contour_window_table(
        df,
        Path(args.condition_window_features),
        phases=condition_phases,
    )
    condition_contour_summary = _condition_contour_summary(
        condition_contour,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    condition_contour_summary_by_animal = _condition_contour_summary(
        condition_contour,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        by_subject=True,
    )
    condition_contour_contrasts = _condition_contour_high_low_contrasts(
        condition_contour,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    condition_contour_contrasts_by_animal = _condition_contour_high_low_contrasts(
        condition_contour,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        by_subject=True,
    )
    condition_contour.to_csv(out_dir / "fixrsvp_backimage_contour_motion_windows.csv", index=False)
    condition_contour_brackets.to_csv(out_dir / "fixrsvp_backimage_contour_coherence_brackets.csv", index=False)
    condition_contour_summary.to_csv(out_dir / "fixrsvp_backimage_contour_motion_summary.csv", index=False)
    condition_contour_summary_by_animal.to_csv(out_dir / "fixrsvp_backimage_contour_motion_summary_by_animal.csv", index=False)
    condition_contour_contrasts.to_csv(out_dir / "fixrsvp_backimage_contour_motion_high_low_contrasts.csv", index=False)
    condition_contour_contrasts_by_animal.to_csv(out_dir / "fixrsvp_backimage_contour_motion_high_low_contrasts_by_animal.csv", index=False)
    _plot_condition_contour_component_lines(
        condition_contour_summary,
        out_dir / "candidate6_fixrsvp_backimage_along_across_rms_by_coherence",
    )
    _plot_condition_contour_component_lines_by_subject(
        condition_contour_summary_by_animal,
        out_dir / "diagnostic_animal_fixrsvp_backimage_along_across_rms_by_coherence",
    )
    _plot_condition_along_across_density(
        condition_contour,
        out_dir / "candidate7_fixrsvp_backimage_along_across_density_contours",
    )
    condition_contour_polar = _plot_condition_polar_alignment_by_stimulus(
        condition_contour,
        out_dir / "candidate8_fixrsvp_backimage_polar_alignment_by_coherence",
        no_coherence_max=no_coherence_max,
    )
    condition_contour_polar.to_csv(out_dir / "fixrsvp_backimage_polar_alignment_by_coherence.csv", index=False)

    condition_session = _condition_session_table(
        Path(args.condition_window_features),
        phases=condition_phases,
        dt=float(args.dt),
        discard_initial_s=float(args.condition_discard_initial_s),
        saccade_pad_s=float(args.condition_saccade_pad_s),
        hard_speed_cutoff_deg_s=hard_speed_cutoff_or_none,
        post_event_exclusion_s=float(args.condition_post_event_exclusion_s),
        min_segment_s=float(args.condition_min_segment_s),
        max_lag_s=float(args.condition_max_diffusion_lag_s),
        within_segment_max_lag_s=float(args.condition_within_segment_max_diffusion_lag_s),
        min_segments_per_lag=int(args.condition_min_segments_per_lag),
    )
    condition_summary, condition_contrasts = _condition_summary(
        condition_session,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    condition_session.to_csv(out_dir / "fixrsvp_backimage_condition_session_medians.csv", index=False)
    condition_summary.to_csv(out_dir / "fixrsvp_backimage_condition_summary_by_animal.csv", index=False)
    condition_contrasts.to_csv(out_dir / "fixrsvp_backimage_condition_contrasts_by_animal.csv", index=False)
    _plot_condition_comparison(
        condition_session,
        condition_summary,
        out_dir / "diagnostic_fixrsvp_vs_backimage_motion_scale_by_animal",
    )
    _plot_condition_comparison(
        condition_session,
        condition_summary,
        out_dir / "supplement_fixrsvp_vs_backimage_motion_scale_by_animal_session_lines",
        show_session_lines=True,
        tight_ylim=False,
    )

    cfg = ComponentPlotConfig(
        input_windows=str(args.input_windows),
        condition_window_features=str(args.condition_window_features),
        out_dir=str(out_dir),
        dt=float(args.dt),
        msd_lags=[int(v) for v in args.msd_lags],
        condition_phases=condition_phases,
        condition_discard_initial_s=float(args.condition_discard_initial_s),
        condition_saccade_pad_s=float(args.condition_saccade_pad_s),
        condition_hard_speed_cutoff_deg_s=hard_speed_cutoff_or_none,
        condition_post_event_exclusion_s=float(args.condition_post_event_exclusion_s),
        condition_min_segment_s=float(args.condition_min_segment_s),
        condition_max_diffusion_lag_s=float(args.condition_max_diffusion_lag_s),
        condition_within_segment_max_diffusion_lag_s=float(args.condition_within_segment_max_diffusion_lag_s),
        condition_min_segments_per_lag=int(args.condition_min_segments_per_lag),
        no_coherence_max=no_coherence_max,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        recompute_traces=bool(args.recompute_traces),
    )
    write_json(out_dir / "run_metadata.json", {
        "config": asdict(cfg),
        "n_windows": int(df.shape[0]),
        "n_sessions": int(df["session"].nunique()),
        "condition_contour_n_windows": int(condition_contour.shape[0]),
        "condition_contour_n_sessions": int(condition_contour["session"].nunique()),
        "condition_comparison_n_session_stimulus_rows": int(condition_session.shape[0]),
        "condition_comparison_phases": condition_phases,
        "condition_diffusion_fit_note": (
            "Condition diffusion follows the Intoy/Rucci-style ensemble estimate: drift segments are aligned "
            "at segment onset; for each elapsed time, the 2D variance of displacement across all available "
            "segments is regressed against elapsed seconds, D = slope / 4. Variable-length FixRSVP trials "
            "contribute to all elapsed times they contain. A second absolute speed gate removes any adjacent "
            f"valid sample pair above {hard_speed_cutoff_or_none} deg/s before segmenting, with the same saccade pad. "
            f"The first {float(args.condition_post_event_exclusion_s):.3g} s after each detected/gated event is "
            "also excluded from diffusion segments."
        ),
        "diffusion_note": (
            "Projected along/across diffusion constants use 1D MSD slopes: "
            "D_axis = max(slope(MSD_axis vs lag_seconds) / 2, 0). "
            "Signed slopes are also saved because bounded FEM windows can have negative multi-lag slopes."
        ),
        "condition_diffusion_note": (
            "FixRSVP/BackImage condition plots include within_segment_signed_d_median_arcmin2_s "
            "as the local drift-scale slope diagnostic and ensemble_diffusion_arcmin2_s as the "
            "fixation-onset fan-out diagnostic. "
            "The original positive-part diffusion_constant_arcmin2_s and previous signed window-lag "
            "signed_msd_d_eff_arcmin2_s are still saved in the session CSV for auditing."
        ),
        "rms_note": "Along is the local image_edge_axis_deg contour direction; across is the orthogonal contour-normal axis.",
        "condition_contour_note": (
            "Combined FixRSVP/BackImage contour plots use within-stimulus terciles of local orientation coherence. "
            "BackImage contour axes come from the reviewed static-canvas local image features at mean gaze. "
            "FixRSVP contour axes are estimated from the center sample's gaze-centered 51x51 stimulus crop "
            "using the same Sobel structure-tensor edge-axis convention. "
            "The along/across density-contour plot is a central-window view capped at 4 arcmin per axis; "
            "the saved window table is untrimmed."
        ),
        "polar_no_coherence_reference_note": (
            "Polar alignment plots show angular density, not eye speed. Dashed no-coherence references use "
            f"windows with image_orientation_coherence <= {no_coherence_max:.3g}; set --no-coherence-max <= 0 to disable."
        ),
        "polar_rms_profile_note": (
            "candidate5b/diagnostic_animal_polar_rms_profile_by_coherence use median position-cloud RMS in arcmin "
            "as the polar radius, with 0/180 deg contour-parallel and 90/270 deg contour-normal. These panels, "
            "not the angular-density panels, are the polar view of orthogonal motion compression."
        ),
        "polar_rms_ratio_note": (
            "candidate5c/diagnostic_animal_polar_rms_ratio_to_no_coherence divide each RMS profile by the matching "
            "no-coherence profile at the same contour-relative angle, so the dashed reference is a unit circle. "
            "Values below 1 at 90/270 deg indicate contour-normal compression relative to no-coherence windows."
        ),
        "polar_rms_four_panel_note": (
            "diagnostic_animal_polar_rms_four_panel combines absolute RMS anisotropy and RMS/no-coherence ratio "
            "for each animal in a 2x2 layout."
        ),
    })
    print(f"Wrote contour-motion component plots to {out_dir}")
    return out_dir


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
