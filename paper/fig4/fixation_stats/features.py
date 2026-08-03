from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import signal


DEFAULT_MSD_LAGS = (1, 2, 4, 8, 16)


def _finite_trace(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {x.shape}")
    return x[np.isfinite(x).all(axis=1)]


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _safe_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _autocorr_rows(x: np.ndarray, lag: int) -> float:
    if x.shape[0] <= lag:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return num / den if den > 0 else float("nan")


def _velocity_autocorr(step: np.ndarray, lag: int) -> float:
    if step.shape[0] <= lag:
        return float("nan")
    a = step[:-lag]
    b = step[lag:]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = den > 1e-12
    return float(np.mean(num[valid] / den[valid])) if np.any(valid) else float("nan")


def _direction_persistence(step: np.ndarray) -> tuple[float, float]:
    if step.shape[0] < 2:
        return float("nan"), float("nan")
    a = step[:-1]
    b = step[1:]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    if not np.any(valid):
        return float("nan"), float("nan")
    cosang = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.arccos(cosang)
    return float(np.mean(cosang)), float(np.mean(np.abs(angles)))


def _power_features(centered: np.ndarray, dt: float) -> dict[str, float]:
    if centered.shape[0] < 8:
        return {
            "position_psd_slope_1_30hz": float("nan"),
            "position_high_freq_power_fraction_15_60hz": float("nan"),
        }
    fs = 1.0 / float(dt)
    nperseg = min(centered.shape[0], 256)
    freqs, pxx = signal.welch(centered[:, 0], fs=fs, nperseg=nperseg, detrend="constant")
    _, pyy = signal.welch(centered[:, 1], fs=fs, nperseg=nperseg, detrend="constant")
    power = np.asarray(pxx + pyy, dtype=np.float64)
    total = float(np.sum(power[freqs > 0]))
    high = float(np.sum(power[(freqs >= 15.0) & (freqs <= min(60.0, fs / 2.0))]))
    slope_mask = (freqs >= 1.0) & (freqs <= min(30.0, fs / 2.0)) & (power > 0)
    if np.count_nonzero(slope_mask) >= 3:
        slope = float(np.polyfit(np.log(freqs[slope_mask]), np.log(power[slope_mask]), 1)[0])
    else:
        slope = float("nan")
    return {
        "position_psd_slope_1_30hz": slope,
        "position_high_freq_power_fraction_15_60hz": high / total if total > 0 else float("nan"),
    }


def fixation_window_features(
    trace: np.ndarray,
    *,
    dt: float,
    msd_lags: Sequence[int] = DEFAULT_MSD_LAGS,
) -> dict[str, float]:
    """Compute compact fixation/drift statistics for one eye-position window."""
    x = _finite_trace(trace)
    out: dict[str, float] = {"n_samples": float(x.shape[0]), "duration_s": float(x.shape[0] * dt)}
    if x.shape[0] < 3:
        return out

    mean = np.mean(x, axis=0)
    centered = x - mean
    radius = np.linalg.norm(centered, axis=1)
    step = np.diff(x, axis=0)
    step_radius = np.linalg.norm(step, axis=1)
    speed = step_radius / float(dt)
    cov = np.cov(centered.T) if x.shape[0] > 1 else np.full((2, 2), np.nan)
    cov = np.asarray(cov, dtype=np.float64)
    if np.isfinite(cov).all():
        evals = np.linalg.eigvalsh(cov)
        evals = np.maximum(evals, 0.0)
        lam_min, lam_max = float(evals[0]), float(evals[1])
        drift_orientation = 0.5 * np.arctan2(2.0 * float(cov[0, 1]), float(cov[0, 0] - cov[1, 1]))
    else:
        lam_min = lam_max = float("nan")
        drift_orientation = float("nan")

    path_length = float(np.sum(step_radius))
    persistence, curvature = _direction_persistence(step)
    dot = np.sum(centered[:-1] * step, axis=1)
    r2 = np.sum(centered[:-1] * centered[:-1], axis=1)
    valid_r = r2 > 1e-12
    return_strength = -float(np.mean(dot[valid_r] / r2[valid_r])) if np.any(valid_r) else float("nan")

    out.update({
        "mean_x_deg": float(mean[0]),
        "mean_y_deg": float(mean[1]),
        "abs_mean_radius_deg": float(np.linalg.norm(mean)),
        "rms_radius_deg": float(np.sqrt(np.mean(radius ** 2))),
        "median_radius_deg": float(np.median(radius)),
        "p05_radius_deg": _safe_quantile(radius, 0.05),
        "p95_radius_deg": _safe_quantile(radius, 0.95),
        "max_radius_deg": float(np.max(radius)),
        "cov_xx_deg2": float(cov[0, 0]),
        "cov_xy_deg2": float(cov[0, 1]),
        "cov_yy_deg2": float(cov[1, 1]),
        "cloud_area_deg2": float(np.pi * np.sqrt(max(lam_min * lam_max, 0.0))) if np.isfinite(lam_min + lam_max) else float("nan"),
        "anisotropy": (lam_max - lam_min) / (lam_max + lam_min) if (lam_max + lam_min) > 0 else float("nan"),
        "drift_orientation_deg": float(np.degrees(drift_orientation)),
        "step_mean_deg": _safe_mean(step_radius),
        "step_median_deg": float(np.median(step_radius)),
        "step_p95_deg": _safe_quantile(step_radius, 0.95),
        "speed_mean_deg_s": _safe_mean(speed),
        "speed_median_deg_s": float(np.median(speed)),
        "speed_p95_deg_s": _safe_quantile(speed, 0.95),
        "path_length_deg": path_length,
        "path_length_deg_s": path_length / ((x.shape[0] - 1) * float(dt)),
        "direction_persistence": persistence,
        "curvature_rad": curvature,
        "return_to_center_strength": return_strength,
        "position_autocorr_lag1": _autocorr_rows(centered, 1),
        "position_autocorr_lag4": _autocorr_rows(centered, 4),
        "velocity_autocorr_lag1": _velocity_autocorr(step, 1),
        "velocity_autocorr_lag4": _velocity_autocorr(step, 4),
        "fraction_within_0p05deg": float(np.mean(radius <= 0.05)),
        "fraction_within_0p10deg": float(np.mean(radius <= 0.10)),
        "fraction_within_0p25deg": float(np.mean(radius <= 0.25)),
    })

    msd_x: list[float] = []
    msd_t: list[float] = []
    for lag in msd_lags:
        lag = int(lag)
        if lag <= 0 or x.shape[0] <= lag:
            out[f"msd_lag{lag}_deg2"] = float("nan")
            continue
        disp = x[lag:] - x[:-lag]
        msd = float(np.mean(np.sum(disp * disp, axis=1)))
        out[f"msd_lag{lag}_deg2"] = msd
        msd_x.append(msd)
        msd_t.append(lag * float(dt))
    if len(msd_x) >= 2:
        slope = float(np.polyfit(np.asarray(msd_t), np.asarray(msd_x), 1)[0])
        out["diffusion_constant_deg2_s"] = max(slope / 4.0, 0.0)
    else:
        out["diffusion_constant_deg2_s"] = float("nan")
    out.update(_power_features(centered, dt))
    return out


def event_feature_rows(
    trace: np.ndarray,
    events: Sequence[dict[str, Any]],
    *,
    dt: float,
) -> list[dict[str, float | int]]:
    x = _finite_trace(trace)
    rows: list[dict[str, float | int]] = []
    for event in events:
        onset = int(event["onset"])
        offset = int(event["offset"])
        if onset < 0 or offset >= x.shape[0] or offset < onset:
            continue
        delta = x[offset] - x[onset]
        amp = float(np.linalg.norm(delta))
        rows.append({
            "event_onset_sample": onset,
            "event_offset_sample": offset,
            "event_onset_s": float(onset * dt),
            "event_offset_s": float(offset * dt),
            "event_duration_s": float((offset - onset + 1) * dt),
            "event_amplitude_deg": amp,
            "event_direction_deg": float(np.degrees(np.arctan2(delta[1], delta[0]))),
            "event_peak_speed_deg_s": float(event.get("peak_speed_deg_s", np.nan)),
        })
    return rows
