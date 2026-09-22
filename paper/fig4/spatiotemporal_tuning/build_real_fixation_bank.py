#!/usr/bin/env python3
"""Build a native-rate real-fixation bank for retinal causal-chain analyses.

The bank is deliberately independent of the natural-image bank.  It extracts
long intervals between macro-saccades from every requested BackImage session,
retains detected sub-degree events inside those intervals, filters raw
high-rate DDPI position before sampling at 240 Hz, and stores both the filtered
primary trace and the unfiltered-resampled sensitivity control.

By default, every retained trace contains 60 real history samples followed by
240 scored samples.  The counts are explicit command-line parameters so the
same audited construction can produce the 250-ms Figure-4 replay (60+60
samples) without post-hoc cropping.  Coordinates are centered physical
``[x, y]`` degrees: positive x is screen-right and positive y is screen-up,
matching the Figure-4 scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
DATAYATES_ROOT = ROOT.parent / "DataYatesV1"
for candidate in (ROOT, DATAYATES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from paper.fig4.spatiotemporal_tuning.eye_trace_filter import (
    anti_alias_eye_position,
    load_ddpi,
)


DEFAULT_CONFIG = ROOT / "paper/model_selection/configs/multi_240_long_split3_dekel35.yaml"
DEFAULT_PROCESSED = Path("/mnt/ssd/YatesMarmoV1/processed")
TARGET_RATE_HZ = 240.0
HISTORY_SAMPLES = 60
ANALYSIS_SAMPLES = 240
TOTAL_SAMPLES = HISTORY_SAMPLES + ANALYSIS_SAMPLES


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    """Select a deterministic ordered subset spanning the complete bank."""
    if count < 1 or count > length:
        raise ValueError(f"count must be in [1, {length}], got {count}")
    indices = np.rint(np.linspace(0, length - 1, count)).astype(int)
    if len(np.unique(indices)) != count:
        raise RuntimeError("evenly spaced selection produced duplicate rows")
    return indices


def _trapezoid(values: np.ndarray, coordinates: np.ndarray, *, axis: int = -1):
    """NumPy 1.x/2.x-compatible trapezoidal integration."""
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(values, coordinates, axis=axis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-traces", type=int, default=100)
    parser.add_argument("--history-samples", type=int, default=HISTORY_SAMPLES)
    parser.add_argument("--analysis-samples", type=int, default=ANALYSIS_SAMPLES)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--candidates-per-session", type=int, default=24)
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        default=1.0,
        help=(
            "Stride between complete 1.25-s windows inside one guarded "
            "fixation; scored intervals do not overlap at the default."
        ),
    )
    parser.add_argument("--saccade-guard-seconds", type=float, default=0.05)
    parser.add_argument("--eye-filter-padding-seconds", type=float, default=0.6)
    parser.add_argument("--eye-passband-hz", type=float, default=20.0)
    parser.add_argument("--eye-stopband-hz", type=float, default=30.0)
    parser.add_argument(
        "--macro-saccade-min-amplitude-deg",
        type=float,
        default=1.0,
        help=(
            "Only saved events at or above this displacement split/guard an "
            "interval. Smaller detected events remain in the replay bank."
        ),
    )
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--maximum-valid-gap-seconds", type=float, default=0.02)
    parser.add_argument(
        "--maximum-analysis-speed-deg-s",
        type=float,
        default=120.0,
        help=(
            "Reject residual saccade-like jumps whose peak speed exceeds this "
            "threshold after eye-position filtering. This is a fallback guard "
            "for events missed by the saved saccade detector."
        ),
    )
    parser.add_argument("--maximum-scale", type=float, default=2.0)
    parser.add_argument("--source-patch-size", type=int, default=540)
    parser.add_argument("--spectrum-crop-size", type=int, default=255)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_sessions(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sessions = [str(value) for value in config.get("sessions", [])]
    if not sessions:
        raise ValueError(f"dataset config {path} contains no sessions")
    return sessions


def backimage_trial_intervals(path: Path) -> tuple[list[dict[str, Any]], float]:
    """Return contiguous BackImage trial intervals without loading stimulus storage."""
    payload = torch.load(path, weights_only=False, mmap=True, map_location="cpu")
    covariates = payload["covariates"]
    time = covariates["t_bins"].numpy()
    trial = covariates["trial_inds"].numpy().astype(np.int64, copy=False)
    if len(time) != len(trial) or len(time) < 2:
        raise ValueError(f"invalid BackImage time/trial arrays in {path}")
    dt = float(np.median(np.diff(time[np.flatnonzero(np.diff(trial) == 0)[:10000]])))
    if not np.isclose(dt, 1.0 / TARGET_RATE_HZ, rtol=2e-3, atol=1e-6):
        raise ValueError(f"BackImage dataset is not native 240 Hz: dt={dt}")
    boundary = np.r_[0, np.flatnonzero(np.diff(trial) != 0) + 1, len(trial)]
    rows = []
    for left, right in zip(boundary[:-1], boundary[1:]):
        rows.append(
            {
                "trial_idx": int(trial[left]),
                "trial_start_ephys": float(time[left] - 0.5 * dt),
                "trial_stop_ephys": float(time[right - 1] + 0.5 * dt),
            }
        )
    return rows, dt


def valid_saccade_events(saccades_path: Path) -> list[dict[str, float]]:
    """Load finite, physiologically possible saved events with displacement."""
    events = json.loads(saccades_path.read_text(encoding="utf-8"))
    rows: list[dict[str, float]] = []
    for event in events:
        start = float(event.get("start_time", np.nan))
        stop = float(event.get("end_time", np.nan))
        duration = stop - start
        coordinates = np.asarray(
            [
                event.get("start_x", np.nan),
                event.get("start_y", np.nan),
                event.get("end_x", np.nan),
                event.get("end_y", np.nan),
            ],
            dtype=float,
        )
        if not (np.isfinite(start) and np.isfinite(stop) and 0.0 < duration <= 0.25):
            continue
        if np.all(np.isfinite(coordinates)):
            amplitude = float(
                np.hypot(coordinates[2] - coordinates[0], coordinates[3] - coordinates[1])
            )
        else:
            # Unknown displacement remains a guarded event instead of being
            # silently admitted into a fixation.
            amplitude = float("inf")
        rows.append(
            {
                "start_time": start,
                "end_time": stop,
                "duration_seconds": duration,
                "amplitude_deg": amplitude,
                "detector_peak": float(event.get("A", np.nan)),
            }
        )
    return sorted(rows, key=lambda row: row["start_time"])


def guarded_fixations(
    trials: list[dict[str, Any]],
    saccades_path: Path,
    *,
    guard_seconds: float,
    macro_saccade_min_amplitude_deg: float = 1.0,
) -> list[dict[str, Any]]:
    if macro_saccade_min_amplitude_deg <= 0:
        raise ValueError("macro_saccade_min_amplitude_deg must be positive")
    events = valid_saccade_events(saccades_path)
    macro = [
        row
        for row in events
        if float(row["amplitude_deg"]) >= float(macro_saccade_min_amplitude_deg)
    ]
    starts = np.asarray([row["start_time"] for row in macro], dtype=float)
    stops = np.asarray([row["end_time"] for row in macro], dtype=float)
    rows: list[dict[str, Any]] = []
    fixation_index = 0
    for trial in trials:
        trial_start = float(trial["trial_start_ephys"])
        trial_stop = float(trial["trial_stop_ephys"])
        overlap = (starts < trial_stop) & (stops > trial_start)
        cursor = trial_start + float(guard_seconds)
        for event_start, event_stop in zip(starts[overlap], stops[overlap]):
            stop = min(float(event_start) - float(guard_seconds), trial_stop)
            if stop > cursor:
                rows.append(
                    {
                        **trial,
                        "fixation_index": fixation_index,
                        "fixation_start_ephys": float(cursor),
                        "fixation_stop_ephys": float(stop),
                        "fixation_duration_seconds": float(stop - cursor),
                    }
                )
                fixation_index += 1
            cursor = max(cursor, float(event_stop) + float(guard_seconds))
        stop = trial_stop - float(guard_seconds)
        if stop > cursor:
            rows.append(
                {
                    **trial,
                    "fixation_index": fixation_index,
                    "fixation_start_ephys": float(cursor),
                    "fixation_stop_ephys": float(stop),
                    "fixation_duration_seconds": float(stop - cursor),
                }
            )
            fixation_index += 1
    return rows


def within_window_event_summary(
    events: list[dict[str, float]],
    *,
    scored_start: float,
    scored_stop: float,
    macro_saccade_min_amplitude_deg: float,
) -> dict[str, float | int | str]:
    """Summarize saved sub-degree events inside the scored one-second interval."""
    overlap = [
        row
        for row in events
        if row["start_time"] < scored_stop
        and row["end_time"] > scored_start
        and row["amplitude_deg"] < macro_saccade_min_amplitude_deg
    ]
    amplitudes = [row["amplitude_deg"] for row in overlap if np.isfinite(row["amplitude_deg"])]
    peaks = [row["detector_peak"] for row in overlap if np.isfinite(row["detector_peak"])]
    onset = [max(0.0, row["start_time"] - scored_start) for row in overlap]
    return {
        "saved_microsaccade_count": int(len(overlap)),
        "saved_microsaccade_max_amplitude_deg": float(max(amplitudes, default=0.0)),
        "saved_microsaccade_max_detector_peak": float(max(peaks, default=0.0)),
        "saved_microsaccade_onsets_seconds": ";".join(f"{value:.6f}" for value in onset),
    }


def ddpi_quality(
    ddpi: pd.DataFrame,
    start: float,
    stop: float,
) -> tuple[float, float]:
    time = ddpi.t_ephys.to_numpy(dtype=float)
    left, right = np.searchsorted(time, (start, stop))
    subset = ddpi.iloc[left:right]
    if len(subset) < 4:
        return 0.0, float("inf")
    valid = subset.valid.to_numpy(dtype=bool)
    valid_time = subset.t_ephys.to_numpy(dtype=float)[valid]
    gap = float(np.max(np.diff(valid_time))) if len(valid_time) > 1 else float("inf")
    return float(np.mean(valid)), gap


def ij_pixels_to_centered_xy_degrees(
    position_ij: np.ndarray,
    *,
    ppd: float,
    center_from: slice,
) -> np.ndarray:
    """Convert row/column pixels to centered right/up physical degrees."""
    value = np.asarray(position_ij, dtype=np.float64)
    center = np.median(value[center_from], axis=0)
    output = np.empty_like(value)
    output[:, 0] = (value[:, 1] - center[1]) / float(ppd)
    output[:, 1] = -(value[:, 0] - center[0]) / float(ppd)
    return output


def trace_metrics(
    trace_xy: np.ndarray, *, history_samples: int = HISTORY_SAMPLES
) -> dict[str, float]:
    analysis = np.asarray(trace_xy[int(history_samples) :], dtype=np.float64)
    radius = np.linalg.norm(analysis - analysis.mean(axis=0), axis=1)
    step = np.diff(analysis, axis=0)
    speed = np.linalg.norm(step, axis=1) * TARGET_RATE_HZ
    return {
        "analysis_rms_radius_deg": float(np.sqrt(np.mean(np.square(radius)))),
        "analysis_path_length_deg": float(np.sum(np.linalg.norm(step, axis=1))),
        "analysis_speed_mean_deg_s": float(np.mean(speed)),
        "analysis_speed_peak_deg_s": float(np.max(speed)),
        "analysis_max_radius_deg": float(np.max(radius)),
    }


def complete_fixation_windows(
    fixation: dict[str, Any],
    *,
    required_seconds: float,
    stride_seconds: float,
) -> list[dict[str, Any]]:
    """Tile complete history+analysis windows inside one guarded fixation."""
    start = float(fixation["fixation_start_ephys"])
    stop = float(fixation["fixation_stop_ephys"])
    if stop - start + 1e-9 < float(required_seconds):
        return []
    latest = stop - float(required_seconds)
    if latest <= start + 1e-9:
        starts = np.asarray((start,))
    else:
        starts = np.arange(start, latest + 1e-9, float(stride_seconds))
    return [
        {
            **fixation,
            "window_index_within_fixation": int(index),
            "window_start_ephys": float(window_start),
            "window_stop_ephys": float(window_start + required_seconds),
        }
        for index, window_start in enumerate(starts)
    ]


def session_candidates(
    session: str,
    *,
    processed_root: Path,
    candidates_per_session: int,
    guard_seconds: float,
    filter_padding_seconds: float,
    passband_hz: float,
    stopband_hz: float,
    minimum_valid_fraction: float,
    maximum_valid_gap_seconds: float,
    maximum_analysis_speed_deg_s: float,
    maximum_allowed_radius_deg: float,
    window_stride_seconds: float,
    macro_saccade_min_amplitude_deg: float,
    history_samples: int,
    analysis_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    session_dir = processed_root / session
    dataset_path = session_dir / "datasets/backimage.dset"
    ddpi_path = session_dir / "dpi/ddpi.csv"
    saccades_path = session_dir / "saccades/saccades.json"
    for path in (dataset_path, ddpi_path, saccades_path):
        if not path.exists():
            raise FileNotFoundError(path)
    trials, _ = backimage_trial_intervals(dataset_path)
    saved_events = valid_saccade_events(saccades_path)
    fixations = guarded_fixations(
        trials,
        saccades_path,
        guard_seconds=guard_seconds,
        macro_saccade_min_amplitude_deg=macro_saccade_min_amplitude_deg,
    )
    total_samples = int(history_samples) + int(analysis_samples)
    required_seconds = total_samples / TARGET_RATE_HZ
    eligible = [row for row in fixations if row["fixation_duration_seconds"] >= required_seconds]
    eligible.sort(key=lambda row: -float(row["fixation_duration_seconds"]))
    windows = [
        window
        for fixation in eligible
        for window in complete_fixation_windows(
            fixation,
            required_seconds=required_seconds,
            stride_seconds=float(window_stride_seconds),
        )
    ]
    # Sample throughout the recording instead of taking only the longest or
    # earliest fixations.  The stable session-specific seed makes the bank
    # exactly reproducible while preventing temporal ordering bias.
    session_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{session}".encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(session_seed)
    rng.shuffle(windows)
    # Evaluate more than the requested per-session quota so quality gates do
    # not preferentially remove sessions with brief DDPI dropouts.
    windows = windows[: max(int(candidates_per_session) * 12, 96)]
    audit = {
        "guarded_fixations": int(len(fixations)),
        "eligible_fixations": int(len(eligible)),
        "windows_available": int(len(windows)),
        "windows_evaluated": 0,
        "rejected_ddpi_quality": 0,
        "rejected_sample_count": 0,
        "rejected_peak_speed": 0,
        "rejected_crop_radius": 0,
        "accepted": 0,
    }
    if not windows:
        return [], audit
    ddpi = load_ddpi(ddpi_path)
    ppd_payload = torch.load(dataset_path, weights_only=False, mmap=True, map_location="cpu")
    ppd = float(np.asarray(ppd_payload["metadata"]["ppd"]).squeeze())
    output: list[dict[str, Any]] = []
    for fixation in windows:
        audit["windows_evaluated"] += 1
        start = float(fixation["window_start_ephys"])
        stop = float(fixation["window_stop_ephys"])
        valid_fraction, maximum_gap = ddpi_quality(ddpi, start, stop)
        if valid_fraction < minimum_valid_fraction or maximum_gap > maximum_valid_gap_seconds:
            audit["rejected_ddpi_quality"] += 1
            continue
        eye = anti_alias_eye_position(
            ddpi,
            start_ephys=start,
            stop_ephys=stop,
            target_rate_hz=TARGET_RATE_HZ,
            passband_hz=passband_hz,
            stopband_hz=stopband_hz,
            padding_seconds=filter_padding_seconds,
        )
        filtered_ij = np.asarray(eye["filtered_position_px"], dtype=np.float64)
        raw_ij = np.asarray(eye["unfiltered_position_px"], dtype=np.float64)
        if len(filtered_ij) != total_samples or raw_ij.shape != filtered_ij.shape:
            audit["rejected_sample_count"] += 1
            continue
        center_slice = slice(int(history_samples), total_samples)
        filtered_xy = ij_pixels_to_centered_xy_degrees(
            filtered_ij, ppd=ppd, center_from=center_slice
        )
        raw_xy = ij_pixels_to_centered_xy_degrees(
            raw_ij, ppd=ppd, center_from=center_slice
        )
        metrics = trace_metrics(filtered_xy, history_samples=int(history_samples))
        if float(metrics["analysis_speed_peak_deg_s"]) > maximum_analysis_speed_deg_s:
            audit["rejected_peak_speed"] += 1
            continue
        if float(metrics["analysis_max_radius_deg"]) > maximum_allowed_radius_deg:
            audit["rejected_crop_radius"] += 1
            continue
        event_summary = within_window_event_summary(
            saved_events,
            scored_start=start + int(history_samples) / TARGET_RATE_HZ,
            scored_stop=stop,
            macro_saccade_min_amplitude_deg=macro_saccade_min_amplitude_deg,
        )
        output.append(
            {
                **fixation,
                "session": session,
                "window_start_ephys": start,
                "window_stop_ephys": stop,
                "target_rate_hz": TARGET_RATE_HZ,
                "ppd": ppd,
                "valid_fraction": valid_fraction,
                "maximum_valid_gap_seconds": maximum_gap,
                "raw_eye_sample_rate_hz": float(eye["source_rate_hz"]),
                "filtered_vs_raw_rms_difference_deg": float(
                    np.sqrt(np.mean(np.square(filtered_xy - raw_xy)))
                ),
                **metrics,
                **event_summary,
                "filtered_trace_xy_deg": filtered_xy.astype(np.float32),
                "raw_trace_xy_deg": raw_xy.astype(np.float32),
            }
        )
        audit["accepted"] += 1
        if len(output) >= int(candidates_per_session):
            break
    return output, audit


def quantile_session_balanced_selection(
    candidates: list[dict[str, Any]],
    n_traces: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if len(candidates) < n_traces:
        raise RuntimeError(f"only {len(candidates)} fixation traces passed; requested {n_traces}")
    rng = np.random.default_rng(seed)
    values = np.asarray([row["analysis_path_length_deg"] for row in candidates])
    order = np.argsort(values, kind="mergesort")
    quantile_groups = np.array_split(order, int(n_traces))
    session_count: dict[str, int] = {}
    n_sessions = len({str(row["session"]) for row in candidates})
    cap = max(4, int(np.ceil(n_traces / max(n_sessions, 1))) + 2)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for group in quantile_groups:
        choices = list(map(int, group))
        rng.shuffle(choices)
        choices.sort(key=lambda index: session_count.get(str(candidates[index]["session"]), 0))
        pick = next(
            (
                index
                for index in choices
                if index not in used
                and session_count.get(str(candidates[index]["session"]), 0) < cap
            ),
            None,
        )
        if pick is None:
            continue
        used.add(pick)
        selected.append(candidates[pick])
        key = str(candidates[pick]["session"])
        session_count[key] = session_count.get(key, 0) + 1
    if len(selected) < n_traces:
        remaining = [index for index in order if int(index) not in used]
        for index in remaining:
            selected.append(candidates[int(index)])
            if len(selected) == n_traces:
                break
    return sorted(selected[:n_traces], key=lambda row: row["analysis_path_length_deg"])


def spectral_filter_qc(
    filtered: np.ndarray,
    raw: np.ndarray,
    *,
    history_samples: int,
    sample_rate_hz: float,
    stopband_hz: float,
) -> dict[str, float | bool]:
    def spectrum(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        analysis = np.asarray(value[:, int(history_samples) :], dtype=np.float64)
        analysis = analysis - analysis.mean(axis=1, keepdims=True)
        window = np.hanning(analysis.shape[1])[None, :, None]
        power = np.abs(np.fft.rfft(analysis * window, axis=1)) ** 2
        frequency = np.fft.rfftfreq(analysis.shape[1], d=1.0 / float(sample_rate_hz))
        return frequency, power.sum(axis=(0, 2))

    frequency, filtered_power = spectrum(filtered)
    raw_frequency, raw_power = spectrum(raw)
    if not np.array_equal(frequency, raw_frequency):
        raise RuntimeError("filtered/raw spectral grids differ")

    def fraction(power: np.ndarray) -> float:
        total = float(_trapezoid(power, frequency))
        keep = frequency >= float(stopband_hz)
        return float(_trapezoid(power[keep], frequency[keep]) / max(total, 1e-12))

    filtered_high = fraction(filtered_power)
    raw_high = fraction(raw_power)
    return {
        "frequency_threshold_hz": float(stopband_hz),
        "filtered_power_fraction_above_threshold": filtered_high,
        "raw_resampled_power_fraction_above_threshold": raw_high,
        "filtered_to_raw_rms_ratio": float(
            np.sqrt(np.mean(np.square(filtered)))
            / max(float(np.sqrt(np.mean(np.square(raw)))), 1e-12)
        ),
        "stopband_suppression_gate": bool(
            filtered_high <= 0.01 and filtered_high <= 0.1 * raw_high
        ),
    }


def write_bank(
    out_dir: Path,
    selected: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    session_audits: list[dict[str, Any]],
    source_candidate_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = np.stack([row.pop("filtered_trace_xy_deg") for row in selected])
    raw = np.stack([row.pop("raw_trace_xy_deg") for row in selected])
    table = pd.DataFrame(selected)
    table.insert(0, "trace_index", np.arange(len(table), dtype=int))
    np.save(out_dir / "trace_xy_filtered.npy", filtered)
    np.save(out_dir / "trace_xy_raw.npy", raw)
    table.to_csv(out_dir / "trace_table.csv", index=False)
    audit_table = pd.DataFrame(session_audits)
    audit_table.to_csv(out_dir / "session_gate_audit.csv", index=False)
    sessions = table.session.value_counts().sort_index()
    manifest = {
        "analysis": "native-240 filtered real-fixation bank",
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256(args.dataset_config.resolve()),
        "n_traces": int(len(table)),
        "n_sessions": int(table.session.nunique()),
        "session_counts": {str(key): int(value) for key, value in sessions.items()},
        "coordinate_convention": "[x_deg screen-right, y_deg screen-up], centered on analysis interval",
        "target_rate_hz": TARGET_RATE_HZ,
        "history_samples": int(args.history_samples),
        "analysis_samples": int(args.analysis_samples),
        "analysis_slice": [
            int(args.history_samples),
            int(args.history_samples) + int(args.analysis_samples),
        ],
        "selection": {
            "source_candidate_count": int(source_candidate_count),
            "selected_trace_count": int(len(table)),
            "window_sampling": "deterministic random sampling across each complete recording",
            "population_weighting": (
                "path-length quantile coverage with a modest per-session cap; "
                "not fixation-duration weighting"
            ),
            "seed": int(args.seed),
        },
        "filter": {
            "kind": "zero-phase analysis IIR on uniform raw-DDPI grid before 240-Hz sampling",
            "passband_hz": float(args.eye_passband_hz),
            "stopband_hz": float(args.eye_stopband_hz),
            "selection_basis": (
                "population DDPI position-PSD breakpoint audit; separate from "
                "the 120-Hz native-sampling Nyquist"
            ),
        },
        "spectral_filter_qc": spectral_filter_qc(
            filtered,
            raw,
            history_samples=int(args.history_samples),
            sample_rate_hz=TARGET_RATE_HZ,
            stopband_hz=float(args.eye_stopband_hz),
        ),
        "quality": {
            "saccade_guard_seconds": float(args.saccade_guard_seconds),
            "macro_saccade_min_amplitude_deg": float(
                args.macro_saccade_min_amplitude_deg
            ),
            "subthreshold_saved_events_retained": True,
            "minimum_valid_fraction": float(args.minimum_valid_fraction),
            "maximum_valid_gap_seconds": float(args.maximum_valid_gap_seconds),
            "maximum_analysis_speed_deg_s": float(args.maximum_analysis_speed_deg_s),
            "maximum_motion_scale": float(args.maximum_scale),
            "window_stride_seconds": float(args.window_stride_seconds),
            "scored_intervals_overlap_at_default_stride": False,
        },
        "files": {
            "filtered": str((out_dir / "trace_xy_filtered.npy").resolve()),
            "raw": str((out_dir / "trace_xy_raw.npy").resolve()),
            "table": str((out_dir / "trace_table.csv").resolve()),
            "session_gate_audit": str((out_dir / "session_gate_audit.csv").resolve()),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.n_traces < 1:
        raise ValueError("n-traces must be positive")
    if args.history_samples < 1 or args.analysis_samples < 2:
        raise ValueError("history-samples and analysis-samples must be positive")
    if args.window_stride_seconds <= 0:
        raise ValueError("window-stride-seconds must be positive")
    if args.maximum_analysis_speed_deg_s <= 0:
        raise ValueError("maximum-analysis-speed-deg-s must be positive")
    if args.macro_saccade_min_amplitude_deg <= 0:
        raise ValueError("macro-saccade-min-amplitude-deg must be positive")
    sessions = configured_sessions(args.dataset_config.resolve())
    if args.session_limit:
        sessions = sessions[: int(args.session_limit)]
    half_margin_px = 0.5 * (float(args.source_patch_size) - float(args.spectrum_crop_size))
    # Scale applies to displacement around fixation center, so the unscaled
    # trace must remain within this margin divided by the largest scale.
    maximum_allowed_radius_by_ppd = None
    candidates: list[dict[str, Any]] = []
    session_audits: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions, start=1):
        dataset_path = args.processed_root / session / "datasets/backimage.dset"
        payload = torch.load(dataset_path, weights_only=False, mmap=True, map_location="cpu")
        ppd = float(np.asarray(payload["metadata"]["ppd"]).squeeze())
        maximum_allowed_radius_by_ppd = half_margin_px / (ppd * float(args.maximum_scale))
        rows, session_audit = session_candidates(
            session,
            processed_root=args.processed_root,
            candidates_per_session=int(args.candidates_per_session),
            guard_seconds=float(args.saccade_guard_seconds),
            filter_padding_seconds=float(args.eye_filter_padding_seconds),
            passband_hz=float(args.eye_passband_hz),
            stopband_hz=float(args.eye_stopband_hz),
            minimum_valid_fraction=float(args.minimum_valid_fraction),
            maximum_valid_gap_seconds=float(args.maximum_valid_gap_seconds),
            maximum_analysis_speed_deg_s=float(args.maximum_analysis_speed_deg_s),
            maximum_allowed_radius_deg=float(maximum_allowed_radius_by_ppd),
            window_stride_seconds=float(args.window_stride_seconds),
            macro_saccade_min_amplitude_deg=float(
                args.macro_saccade_min_amplitude_deg
            ),
            history_samples=int(args.history_samples),
            analysis_samples=int(args.analysis_samples),
            seed=int(args.seed),
        )
        candidates.extend(rows)
        session_audits.append({"session": session, **session_audit})
        print(
            f"fixation bank session {session_index}/{len(sessions)} {session}: "
            f"{len(rows)} candidates; {len(candidates)} cumulative; "
            f"audit={session_audit}",
            flush=True,
        )
    selected = quantile_session_balanced_selection(
        candidates, int(args.n_traces), seed=int(args.seed)
    )
    if len({row["session"] for row in selected}) < min(10, len(sessions)):
        raise RuntimeError("selected fixation bank spans fewer than the required sessions")
    write_bank(
        args.out_dir.resolve(),
        selected,
        args=args,
        session_audits=session_audits,
        source_candidate_count=len(candidates),
    )
    print(args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
