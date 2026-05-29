"""Microsaccade detection for real eye-trace window selection."""
from __future__ import annotations

import numpy as np

from .common import DT


def speed_threshold_mad(trace: np.ndarray, dt: float = DT, z: float = 6.0) -> float:
    """Robust speed threshold from median absolute deviation."""
    x = np.asarray(trace, dtype=np.float64)
    inc = np.diff(x, axis=0)
    speed = np.linalg.norm(inc, axis=1) / float(dt)
    med = np.median(speed)
    mad = np.median(np.abs(speed - med))
    return float(med + float(z) * 1.4826 * mad)


def detect_microsaccade_events(
    trace: np.ndarray,
    *,
    dt: float = DT,
    threshold_deg_s: float | None = None,
    min_samples: int = 1,
    pad_samples: int = 0,
) -> tuple[list[dict[str, int | float]], np.ndarray, float]:
    """Detect high-speed microsaccade-like events.

    This detector is deliberately operational: it defines windows with no
    high-speed events and windows with exactly one high-speed event so the
    analysis can compare fixation-only and microsaccade-containing retinal
    movies using the same real trace source.
    """
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected trace shape (T, 2), got {x.shape}")
    inc = np.diff(x, axis=0, prepend=x[:1])
    speed = np.linalg.norm(inc, axis=1) / float(dt)
    if threshold_deg_s is None:
        threshold_deg_s = speed_threshold_mad(x, dt=dt)
    event_mask = speed > float(threshold_deg_s)
    if pad_samples > 0 and event_mask.any():
        padded = event_mask.copy()
        for i in np.where(event_mask)[0]:
            lo = max(0, int(i) - int(pad_samples))
            hi = min(event_mask.size, int(i) + int(pad_samples) + 1)
            padded[lo:hi] = True
        event_mask = padded

    events: list[dict[str, int | float]] = []
    i = 0
    while i < event_mask.size:
        if not event_mask[i]:
            i += 1
            continue
        start = i
        while i < event_mask.size and event_mask[i]:
            i += 1
        stop = i
        if stop - start >= int(min_samples):
            events.append({
                "onset": int(start),
                "offset": int(stop - 1),
                "duration_samples": int(stop - start),
                "peak_speed_deg_s": float(np.max(speed[start:stop])),
            })
    return events, event_mask, float(threshold_deg_s)
