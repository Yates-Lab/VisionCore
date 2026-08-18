#!/usr/bin/env python3
"""Build the native-240 real-fixation bank for the M77 causal-chain analysis.

The bank is deliberately independent of the natural-image bank.  It extracts
long inter-saccadic intervals from every requested BackImage session, filters
raw high-rate DDPI position before sampling at 240 Hz, and stores both the
filtered primary trace and the unfiltered-resampled sensitivity control.

Every retained trace contains 60 real history samples followed by 240 scored
samples.  Coordinates are centered physical ``[x, y]`` degrees: positive x is
screen-right and positive y is screen-up, matching the Figure-4 scorer.
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

from paper.fig4.spatiotemporal_tuning.run_real_backimage_power import (
    anti_alias_eye_position,
    load_ddpi,
)


DEFAULT_CONFIG = ROOT / "paper/model_selection/configs/multi_240_long_split3_dekel35.yaml"
DEFAULT_PROCESSED = Path("/mnt/ssd/YatesMarmoV1/processed")
TARGET_RATE_HZ = 240.0
HISTORY_SAMPLES = 60
ANALYSIS_SAMPLES = 240
TOTAL_SAMPLES = HISTORY_SAMPLES + ANALYSIS_SAMPLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-traces", type=int, default=100)
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
    parser.add_argument("--eye-passband-hz", type=float, default=100.0)
    parser.add_argument("--eye-stopband-hz", type=float, default=118.0)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--maximum-valid-gap-seconds", type=float, default=0.02)
    parser.add_argument(
        "--maximum-analysis-speed-deg-s",
        type=float,
        default=30.0,
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


def guarded_fixations(
    trials: list[dict[str, Any]],
    saccades_path: Path,
    *,
    guard_seconds: float,
) -> list[dict[str, Any]]:
    events = json.loads(saccades_path.read_text(encoding="utf-8"))
    starts = np.asarray([row["start_time"] for row in events], dtype=float)
    stops = np.asarray([row["end_time"] for row in events], dtype=float)
    # A small number of historical detector rows contain corrupted end times
    # hundreds of days beyond the recording.  They cannot be real saccades and
    # would otherwise erase every subsequent fixation in the session.
    duration = stops - starts
    valid_event = (
        np.isfinite(starts)
        & np.isfinite(stops)
        & (duration > 0.0)
        & (duration <= 0.25)
    )
    starts, stops = starts[valid_event], stops[valid_event]
    order = np.argsort(starts)
    starts, stops = starts[order], stops[order]
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


def trace_metrics(trace_xy: np.ndarray) -> dict[str, float]:
    analysis = np.asarray(trace_xy[HISTORY_SAMPLES:], dtype=np.float64)
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
) -> list[dict[str, Any]]:
    session_dir = processed_root / session
    dataset_path = session_dir / "datasets/backimage.dset"
    ddpi_path = session_dir / "dpi/ddpi.csv"
    saccades_path = session_dir / "saccades/saccades.json"
    for path in (dataset_path, ddpi_path, saccades_path):
        if not path.exists():
            raise FileNotFoundError(path)
    trials, _ = backimage_trial_intervals(dataset_path)
    fixations = guarded_fixations(trials, saccades_path, guard_seconds=guard_seconds)
    required_seconds = TOTAL_SAMPLES / TARGET_RATE_HZ
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
    # Interleave fixations before limiting the audit pool, so one unusually
    # long interval cannot exhaust the per-session quota.
    windows.sort(
        key=lambda row: (
            int(row["window_index_within_fixation"]),
            -float(row["fixation_duration_seconds"]),
        )
    )
    # Evaluate more than the requested per-session quota so quality gates do
    # not preferentially remove sessions with brief DDPI dropouts.
    windows = windows[: max(int(candidates_per_session) * 6, 48)]
    if not windows:
        return []
    ddpi = load_ddpi(ddpi_path)
    ppd_payload = torch.load(dataset_path, weights_only=False, mmap=True, map_location="cpu")
    ppd = float(np.asarray(ppd_payload["metadata"]["ppd"]).squeeze())
    output: list[dict[str, Any]] = []
    for fixation in windows:
        start = float(fixation["window_start_ephys"])
        stop = float(fixation["window_stop_ephys"])
        valid_fraction, maximum_gap = ddpi_quality(ddpi, start, stop)
        if valid_fraction < minimum_valid_fraction or maximum_gap > maximum_valid_gap_seconds:
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
        if len(filtered_ij) != TOTAL_SAMPLES or raw_ij.shape != filtered_ij.shape:
            continue
        center_slice = slice(HISTORY_SAMPLES, TOTAL_SAMPLES)
        filtered_xy = ij_pixels_to_centered_xy_degrees(
            filtered_ij, ppd=ppd, center_from=center_slice
        )
        raw_xy = ij_pixels_to_centered_xy_degrees(
            raw_ij, ppd=ppd, center_from=center_slice
        )
        metrics = trace_metrics(filtered_xy)
        if float(metrics["analysis_speed_peak_deg_s"]) > maximum_analysis_speed_deg_s:
            continue
        if float(metrics["analysis_max_radius_deg"]) > maximum_allowed_radius_deg:
            continue
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
                "filtered_trace_xy_deg": filtered_xy.astype(np.float32),
                "raw_trace_xy_deg": raw_xy.astype(np.float32),
            }
        )
        if len(output) >= int(candidates_per_session):
            break
    return output


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


def write_bank(
    out_dir: Path,
    selected: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered = np.stack([row.pop("filtered_trace_xy_deg") for row in selected])
    raw = np.stack([row.pop("raw_trace_xy_deg") for row in selected])
    table = pd.DataFrame(selected)
    table.insert(0, "trace_index", np.arange(len(table), dtype=int))
    np.save(out_dir / "trace_xy_filtered.npy", filtered)
    np.save(out_dir / "trace_xy_raw.npy", raw)
    table.to_csv(out_dir / "trace_table.csv", index=False)
    sessions = table.session.value_counts().sort_index()
    manifest = {
        "analysis": "M77 native-240 filtered real-fixation bank",
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256(args.dataset_config.resolve()),
        "n_traces": int(len(table)),
        "n_sessions": int(table.session.nunique()),
        "session_counts": {str(key): int(value) for key, value in sessions.items()},
        "coordinate_convention": "[x_deg screen-right, y_deg screen-up], centered on analysis interval",
        "target_rate_hz": TARGET_RATE_HZ,
        "history_samples": HISTORY_SAMPLES,
        "analysis_samples": ANALYSIS_SAMPLES,
        "analysis_slice": [HISTORY_SAMPLES, TOTAL_SAMPLES],
        "filter": {
            "kind": "zero-phase IIR on uniform raw-DDPI grid before 240-Hz sampling",
            "passband_hz": float(args.eye_passband_hz),
            "stopband_hz": float(args.eye_stopband_hz),
        },
        "quality": {
            "saccade_guard_seconds": float(args.saccade_guard_seconds),
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
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.n_traces < 1:
        raise ValueError("n-traces must be positive")
    if args.window_stride_seconds <= 0:
        raise ValueError("window-stride-seconds must be positive")
    if args.maximum_analysis_speed_deg_s <= 0:
        raise ValueError("maximum-analysis-speed-deg-s must be positive")
    sessions = configured_sessions(args.dataset_config.resolve())
    if args.session_limit:
        sessions = sessions[: int(args.session_limit)]
    half_margin_px = 0.5 * (float(args.source_patch_size) - float(args.spectrum_crop_size))
    # Scale applies to displacement around fixation center, so the unscaled
    # trace must remain within this margin divided by the largest scale.
    maximum_allowed_radius_by_ppd = None
    candidates: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions, start=1):
        dataset_path = args.processed_root / session / "datasets/backimage.dset"
        payload = torch.load(dataset_path, weights_only=False, mmap=True, map_location="cpu")
        ppd = float(np.asarray(payload["metadata"]["ppd"]).squeeze())
        maximum_allowed_radius_by_ppd = half_margin_px / (ppd * float(args.maximum_scale))
        rows = session_candidates(
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
        )
        candidates.extend(rows)
        print(
            f"fixation bank session {session_index}/{len(sessions)} {session}: "
            f"{len(rows)} candidates; {len(candidates)} cumulative",
            flush=True,
        )
    selected = quantile_session_balanced_selection(
        candidates, int(args.n_traces), seed=int(args.seed)
    )
    if len({row["session"] for row in selected}) < min(10, len(sessions)):
        raise RuntimeError("selected fixation bank spans fewer than the required sessions")
    write_bank(args.out_dir.resolve(), selected, args=args)
    print(args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
