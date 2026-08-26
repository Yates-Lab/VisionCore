"""Auditable eye-motion conditions derived from an already filtered trace.

The production fixation bank is reconstructed from raw DDPI samples, filtered
continuously before resampling, and stored at the model input rate.  Functions
in this module never treat finite differences as a substitute for retinal
rendering: they only define matched trajectories that downstream code replays
over the same natural image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator


EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class MotionConditionConfig:
    """Parameters for interpretable perturbations of a filtered eye trace."""

    frame_rate_hz: float = 240.0
    drift_cutoff_hz: float = 35.0
    microsaccade_threshold_z: float = 6.0
    microsaccade_min_speed_deg_s: float = 10.0
    microsaccade_padding_samples: int = 2
    lowpass_order: int = 6


def _validate_trace(trace_xy: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace_xy, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 8:
        raise ValueError(f"trace_xy must have shape [time>=8,2], got {trace.shape}")
    if not np.all(np.isfinite(trace)):
        raise ValueError("trace_xy contains non-finite samples")
    return trace


def eye_speed(trace_xy: np.ndarray, *, frame_rate_hz: float) -> np.ndarray:
    """Return sample-aligned eye speed in degrees/s."""
    trace = _validate_trace(trace_xy)
    if frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be positive")
    velocity = np.diff(trace, axis=0, prepend=trace[:1]) * float(frame_rate_hz)
    return np.linalg.norm(velocity, axis=1)


def lowpass_trace(
    trace_xy: np.ndarray,
    *,
    frame_rate_hz: float,
    cutoff_hz: float,
    order: int = 6,
) -> np.ndarray:
    """Zero-phase low-pass an eye trace without shifting its fixation center."""
    trace = _validate_trace(trace_xy)
    nyquist = 0.5 * float(frame_rate_hz)
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff_hz must lie between 0 and Nyquist ({nyquist:g} Hz)")
    if order < 1:
        raise ValueError("order must be positive")
    center = np.median(trace, axis=0, keepdims=True)
    sos = signal.butter(
        int(order), float(cutoff_hz), btype="lowpass", fs=float(frame_rate_hz), output="sos"
    )
    filtered = signal.sosfiltfilt(sos, trace - center, axis=0)
    return filtered + center


def microsaccade_mask(
    trace_xy: np.ndarray,
    *,
    frame_rate_hz: float,
    threshold_z: float = 6.0,
    minimum_speed_deg_s: float = 10.0,
    padding_samples: int = 2,
) -> tuple[np.ndarray, float]:
    """Detect high-speed events with a robust, explicitly reported threshold."""
    speed = eye_speed(trace_xy, frame_rate_hz=frame_rate_hz)
    median = float(np.median(speed))
    mad = float(np.median(np.abs(speed - median)))
    threshold = max(
        float(minimum_speed_deg_s), median + float(threshold_z) * 1.4826 * mad
    )
    mask = speed > threshold
    if padding_samples > 0 and np.any(mask):
        kernel = np.ones(2 * int(padding_samples) + 1, dtype=np.int8)
        mask = np.convolve(mask.astype(np.int8), kernel, mode="same") > 0
    return mask, threshold


def remove_masked_motion(trace_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bridge detected events with shape-preserving interpolation."""
    trace = _validate_trace(trace_xy)
    event = np.asarray(mask, dtype=bool)
    if event.shape != (len(trace),):
        raise ValueError("mask must have one value per trace sample")
    if not np.any(event):
        return trace.copy()
    keep = np.flatnonzero(~event)
    if len(keep) < 4:
        raise ValueError("too few non-event samples to bridge the trace")
    time = np.arange(len(trace), dtype=float)
    output = np.column_stack(
        [PchipInterpolator(keep, trace[keep, coordinate], extrapolate=True)(time) for coordinate in range(2)]
    )
    # Preserve every measured sample outside the padded event itself.
    output[~event] = trace[~event]
    return output


def phase_scramble_trace(trace_xy: np.ndarray, *, seed: int) -> np.ndarray:
    """Randomize temporal phase while preserving each coordinate's FFT power."""
    trace = _validate_trace(trace_xy)
    center = trace.mean(axis=0, keepdims=True)
    coefficient = np.fft.rfft(trace - center, axis=0)
    rng = np.random.default_rng(int(seed))
    if len(coefficient) > 2:
        stop = len(coefficient) - 1 if len(trace) % 2 == 0 else len(coefficient)
        phase = rng.uniform(-np.pi, np.pi, size=(stop - 1, 2))
        coefficient[1:stop] = np.abs(coefficient[1:stop]) * np.exp(1j * phase)
    coefficient[0] = 0.0
    if len(trace) % 2 == 0:
        coefficient[-1] = np.real(coefficient[-1])
    return np.fft.irfft(coefficient, n=len(trace), axis=0) + center


def build_motion_conditions(
    filtered_trace_xy: np.ndarray,
    *,
    config: MotionConditionConfig = MotionConditionConfig(),
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Create matched replay conditions from one instrument-valid trace.

    ``full_filtered`` is the primary condition. ``drift_only`` and
    ``high_frequency_residual`` are a declared cutoff sensitivity analysis,
    not claims that the cutoff perfectly isolates physiological drift/tremor.
    The two additive pairs reconstruct the primary trace exactly.
    """
    trace = _validate_trace(filtered_trace_xy)
    drift = lowpass_trace(
        trace,
        frame_rate_hz=config.frame_rate_hz,
        cutoff_hz=config.drift_cutoff_hz,
        order=config.lowpass_order,
    )
    event_mask, threshold = microsaccade_mask(
        trace,
        frame_rate_hz=config.frame_rate_hz,
        threshold_z=config.microsaccade_threshold_z,
        minimum_speed_deg_s=config.microsaccade_min_speed_deg_s,
        padding_samples=config.microsaccade_padding_samples,
    )
    without_events = remove_masked_motion(trace, event_mask)
    center = np.median(trace, axis=0, keepdims=True)
    conditions = {
        "stabilized": np.repeat(center, len(trace), axis=0),
        "drift_only": drift,
        "high_frequency_residual": center + (trace - drift),
        "full_filtered": trace.copy(),
        "without_microsaccades": without_events,
        "microsaccade_component": center + (trace - without_events),
        "time_reversed": trace[::-1].copy(),
        "phase_scrambled": phase_scramble_trace(trace, seed=seed),
        "rotated_90deg": center + np.column_stack((-(trace - center)[:, 1], (trace - center)[:, 0])),
    }
    speed = eye_speed(trace, frame_rate_hz=config.frame_rate_hz)
    metadata: dict[str, float | int] = {
        "frame_rate_hz": float(config.frame_rate_hz),
        "drift_cutoff_hz": float(config.drift_cutoff_hz),
        "microsaccade_threshold_deg_s": float(threshold),
        "microsaccade_samples": int(np.sum(event_mask)),
        "microsaccade_fraction": float(np.mean(event_mask)),
        "mean_speed_deg_s": float(np.mean(speed)),
        "peak_speed_deg_s": float(np.max(speed)),
        "path_length_deg": float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()),
    }
    return conditions, metadata
