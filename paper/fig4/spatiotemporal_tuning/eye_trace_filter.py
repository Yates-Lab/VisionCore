"""Audited DDPI loading and zero-phase anti-alias filtering for Figure 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


def load_ddpi(path: Path) -> pd.DataFrame:
    """Read only the eye-position columns required by the replay pipeline."""
    frame = pd.read_csv(
        path,
        usecols=("t_ephys", "dpi_i", "dpi_j", "valid"),
        dtype={"t_ephys": "float64", "dpi_i": "float64", "dpi_j": "float64"},
    )
    valid_text = frame.valid.astype(str).str.lower()
    frame["valid"] = valid_text.isin(("true", "1", "1.0"))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=("t_ephys", "dpi_i", "dpi_j")
    )
    return frame.sort_values("t_ephys").drop_duplicates("t_ephys")


def anti_alias_eye_position(
    ddpi: pd.DataFrame,
    *,
    start_ephys: float,
    stop_ephys: float,
    target_rate_hz: float,
    passband_hz: float,
    stopband_hz: float,
    padding_seconds: float,
) -> dict[str, np.ndarray | float]:
    """Zero-phase filter raw pixel eye position before native-240 sampling."""
    time = ddpi.t_ephys.to_numpy(dtype=float)
    raw_nyquist = 0.5 / np.median(np.diff(time[: min(len(ddpi), 10000)]))
    if not 0 < passband_hz < stopband_hz < raw_nyquist:
        raise ValueError("eye filter edges are inconsistent with raw DDPI Nyquist")
    left, right = np.searchsorted(
        time, (start_ephys - padding_seconds, stop_ephys + padding_seconds)
    )
    subset = ddpi.iloc[left:right]
    valid = subset.valid.to_numpy(dtype=bool)
    raw_time = subset.t_ephys.to_numpy(dtype=float)[valid]
    # Pixel convention is [row=i, column=j], matching the displayed image.
    raw_position = subset.loc[valid, ["dpi_i", "dpi_j"]].to_numpy(dtype=float)
    if len(raw_time) < 32:
        raise RuntimeError("too few valid DDPI samples for anti-alias filtering")
    source_rate_hz = float(1.0 / np.median(np.diff(raw_time)))
    uniform_time = np.arange(raw_time[0], raw_time[-1], 1.0 / source_rate_hz)
    uniform_position = np.column_stack(
        [
            np.interp(uniform_time, raw_time, raw_position[:, coordinate])
            for coordinate in range(2)
        ]
    )
    sos = signal.iirdesign(
        wp=float(passband_hz),
        ws=float(stopband_hz),
        gpass=0.1,
        gstop=60.0,
        fs=source_rate_hz,
        output="sos",
    )
    fixation_center = np.median(uniform_position, axis=0, keepdims=True)
    filtered_uniform = fixation_center + signal.sosfiltfilt(
        sos, uniform_position - fixation_center, axis=0
    )
    target_time = np.arange(
        start_ephys + 0.5 / target_rate_hz,
        stop_ephys,
        1.0 / target_rate_hz,
    )
    filtered = np.column_stack(
        [
            np.interp(target_time, uniform_time, filtered_uniform[:, coordinate])
            for coordinate in range(2)
        ]
    )
    unfiltered = np.column_stack(
        [
            np.interp(target_time, raw_time, raw_position[:, coordinate])
            for coordinate in range(2)
        ]
    )
    return {
        "target_time": target_time,
        "filtered_position_px": filtered,
        "unfiltered_position_px": unfiltered,
        "source_rate_hz": source_rate_hz,
        "sos": sos,
    }
