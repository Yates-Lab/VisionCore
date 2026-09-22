"""Audited DDPI loading and zero-phase anti-alias filtering for Figure 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal


def gaussian_filter_spec(sigma_seconds: float = 0.006) -> dict:
    """Explicit provenance for the non-ringing Figure-4 analysis filter."""
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    return {
        "kind": "zero-phase positive Gaussian on uniform raw-DDPI grid before 240-Hz sampling",
        "family": "gaussian",
        "sigma_seconds": float(sigma_seconds),
        "truncate_sigma": 5.0,
        "minus_3db_hz": float(np.sqrt(np.log(2)) / (2 * np.pi * sigma_seconds)),
        "selection_basis": "gentle rolloff near the independently measured 20-Hz DDPI PSD break; transient and width-sensitivity controls",
    }


def filter_contract_valid(spec: dict) -> bool:
    """Recognize explicit Gaussian or legacy 20/30-Hz filter provenance."""
    if "zero-phase" not in str(spec.get("kind", "")).lower():
        return False
    if spec.get("family") == "gaussian":
        sigma = float(spec.get("sigma_seconds", np.nan))
        return bool(np.isfinite(sigma) and 0.004 <= sigma <= 0.008
                    and float(spec.get("truncate_sigma", 0)) >= 5)
    return bool(np.isclose(float(spec.get("passband_hz", np.nan)), 20)
                and np.isclose(float(spec.get("stopband_hz", np.nan)), 30))


def filter_qc_passed(provenance: dict) -> bool:
    """Use the declared filter's QC, without relabeling Gaussian as a stopband."""
    spec = provenance.get("filter", {})
    if spec.get("family") == "gaussian":
        qc = provenance.get("filter_validation", {})
        return filter_contract_valid(spec) and all(qc.get(key) is True for key in
            ("positive_kernel", "monotonic_step_response", "continuous_raw_before_resampling", "padding_covers_kernel"))
    return bool(provenance.get("spectral_filter_qc", {}).get("stopband_suppression_gate", False))


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
    filter_family: str = "elliptic",
    gaussian_sigma_seconds: float = 0.006,
) -> dict[str, np.ndarray | float]:
    """Zero-phase filter raw pixel eye position before native-240 sampling."""
    time = ddpi.t_ephys.to_numpy(dtype=float)
    raw_nyquist = 0.5 / np.median(np.diff(time[: min(len(ddpi), 10000)]))
    if filter_family not in ("elliptic", "gaussian"):
        raise ValueError(f"unknown eye filter family: {filter_family}")
    if filter_family == "elliptic" and not 0 < passband_hz < stopband_hz < raw_nyquist:
        raise ValueError("eye filter edges are inconsistent with raw DDPI Nyquist")
    if filter_family == "gaussian":
        gaussian_filter_spec(gaussian_sigma_seconds)
        if padding_seconds < 5 * gaussian_sigma_seconds:
            raise ValueError("padding does not cover the Gaussian kernel")
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
    fixation_center = np.median(uniform_position, axis=0, keepdims=True)
    if filter_family == "gaussian":
        if (start_ephys - uniform_time[0] < 5 * gaussian_sigma_seconds or
                uniform_time[-1] - stop_ephys < 5 * gaussian_sigma_seconds):
            raise ValueError("raw record does not cover Gaussian padding")
        sos = np.empty((0, 6))
        filtered_uniform = ndimage.gaussian_filter1d(
            uniform_position, gaussian_sigma_seconds * source_rate_hz,
            axis=0, mode="reflect", truncate=5.0,
        )
    else:
        sos = signal.iirdesign(wp=float(passband_hz), ws=float(stopband_hz),
            gpass=0.1, gstop=60.0, fs=source_rate_hz, output="sos")
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
