#!/usr/bin/env python3
"""Estimate retinal SF–TF power from long, real BackImage fixations.

This analysis intentionally uses the full natural image and raw, high-rate
DDPI samples.  Eye position is anti-aliased before evaluation on the native
240-Hz model grid.  Long inter-saccadic fixations are extracted from ten-second
BackImage presentations and rendered in temporal Welch windows from a large,
guard-banded image crop; spectra, rather than movie segments, are then averaged
across fixations.  Scaling is applied to each complete filtered eye trajectory
around its median fixation position.

The script is a real-trace estimator and does not use Brownian motion or the
instantaneous ``TF = k dot velocity`` approximation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[3]
DATAYATES_ROOT = ROOT.parent / "DataYatesV1"
for candidate in (ROOT, DATAYATES_ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-mat", type=Path, required=True)
    parser.add_argument("--ddpi-csv", type=Path, required=True)
    parser.add_argument("--saccades-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-fixations", type=int, default=8)
    parser.add_argument("--min-fixation-seconds", type=float, default=1.0)
    parser.add_argument(
        "--saccade-guard-seconds",
        type=float,
        default=0.05,
        help="Trim this interval from both sides of each inter-saccadic fixation.",
    )
    parser.add_argument("--target-rate-hz", type=float, default=240.0)
    parser.add_argument("--eye-passband-hz", type=float, default=100.0)
    parser.add_argument("--eye-stopband-hz", type=float, default=118.0)
    parser.add_argument("--eye-filter-padding-seconds", type=float, default=0.6)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.98)
    parser.add_argument("--maximum-valid-gap-seconds", type=float, default=0.02)
    parser.add_argument("--crop-size", type=int, default=255)
    parser.add_argument("--welch-seconds", type=float, default=1.0)
    parser.add_argument("--welch-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--min-spatial-cpd", type=float, default=0.2)
    parser.add_argument("--max-spatial-cpd", type=float, default=16.0)
    parser.add_argument("--n-spatial-bins", type=int, default=49)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def load_backimage_trials(path: Path) -> tuple[list[dict[str, Any]], dict]:
    """Load displayed BackImage trials and convert their times to ephys time."""
    import mat73

    from DataYatesV1.exp.backimage import BackImageTrial
    from DataYatesV1.exp.general import get_trial_protocols
    from DataYatesV1.utils.general import get_clock_functions

    experiment = mat73.loadmat(path)["ExpStruct"]
    protocols = np.asarray(get_trial_protocols(experiment), dtype=object)
    ptb_to_ephys, _ = get_clock_functions(experiment)
    indices = np.flatnonzero(protocols == "BackImage")
    rows: list[dict[str, Any]] = []
    for experiment_index in indices:
        trial = BackImageTrial(
            experiment["D"][int(experiment_index)], experiment["S"]
        )
        start = float(ptb_to_ephys(trial.image_onset_ptb))
        stop = float(ptb_to_ephys(trial.image_offset_ptb))
        rows.append(
            {
                "experiment_index": int(experiment_index),
                "trial": trial,
                "start_ephys": start,
                "stop_ephys": stop,
                "duration_seconds": stop - start,
                "image_file": str(trial.image_file),
            }
        )
    settings = experiment["S"]
    metadata = {
        "ppd": float(np.asarray(settings["pixPerDeg"]).squeeze()),
        "screen_rect": np.asarray(settings["screenRect"], dtype=float).tolist(),
        "n_backimage_trials": int(len(rows)),
    }
    return rows, metadata


def load_ddpi(path: Path) -> pd.DataFrame:
    """Read only the four columns needed for reconstruction from the large CSV."""
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


def fixation_segments(
    trials: list[dict[str, Any]],
    saccades_path: Path,
    *,
    guard_seconds: float,
) -> list[dict[str, Any]]:
    """Split each BackImage presentation into guarded inter-saccadic fixations."""
    saccades = json.loads(saccades_path.read_text(encoding="utf-8"))
    starts = np.asarray([row["start_time"] for row in saccades], dtype=float)
    stops = np.asarray([row["end_time"] for row in saccades], dtype=float)
    order = np.argsort(starts)
    starts, stops = starts[order], stops[order]
    output: list[dict[str, Any]] = []
    fixation_index = 0
    for trial in trials:
        trial_start = float(trial["start_ephys"])
        trial_stop = float(trial["stop_ephys"])
        overlap = (starts < trial_stop) & (stops > trial_start)
        local_start = starts[overlap]
        local_stop = stops[overlap]
        cursor = trial_start + float(guard_seconds)
        for saccade_start, saccade_stop in zip(local_start, local_stop):
            stop = min(float(saccade_start) - float(guard_seconds), trial_stop)
            if stop > cursor:
                output.append(
                    {
                        **trial,
                        "fixation_index": int(fixation_index),
                        "start_ephys": float(cursor),
                        "stop_ephys": float(stop),
                        "duration_seconds": float(stop - cursor),
                    }
                )
                fixation_index += 1
            cursor = max(cursor, float(saccade_stop) + float(guard_seconds))
            if cursor >= trial_stop:
                break
        stop = trial_stop - float(guard_seconds)
        if stop > cursor:
            output.append(
                {
                    **trial,
                    "fixation_index": int(fixation_index),
                    "start_ephys": float(cursor),
                    "stop_ephys": float(stop),
                    "duration_seconds": float(stop - cursor),
                }
            )
            fixation_index += 1
    return output


def trial_quality(
    trial: dict[str, Any],
    ddpi: pd.DataFrame,
    *,
    padding_seconds: float,
    crop_size: int,
    maximum_motion_scale: float,
) -> dict[str, Any]:
    start = float(trial["start_ephys"] - padding_seconds)
    stop = float(trial["stop_ephys"] + padding_seconds)
    time = ddpi.t_ephys.to_numpy(dtype=float)
    left, right = np.searchsorted(time, (start, stop))
    subset = ddpi.iloc[left:right]
    if len(subset) < 4:
        return {
            "valid_fraction": 0.0,
            "maximum_valid_gap_seconds": float("inf"),
            "crop_safe_at_maximum_motion_scale": False,
            "minimum_crop_edge_margin_px": float("-inf"),
        }
    valid = subset.valid.to_numpy(dtype=bool)
    valid_time = subset.t_ephys.to_numpy(dtype=float)[valid]
    maximum_gap = (
        float(np.max(np.diff(valid_time))) if len(valid_time) > 1 else float("inf")
    )
    trial_mask = (
        (subset.t_ephys.to_numpy(dtype=float) >= float(trial["start_ephys"]))
        & (subset.t_ephys.to_numpy(dtype=float) < float(trial["stop_ephys"]))
        & valid
    )
    position = subset.loc[trial_mask, ["dpi_i", "dpi_j"]].to_numpy(dtype=float)
    crop_safe = False
    minimum_edge_margin = float("-inf")
    if len(position):
        center = np.median(position, axis=0)
        scaled = center + float(maximum_motion_scale) * (position - center)
        half = 0.5 * float(crop_size - 1)
        destination = np.asarray(trial["trial"].dest_rect, dtype=float)
        height = float(destination[3] - destination[1])
        width = float(destination[2] - destination[0])
        margins = np.asarray(
            [
                np.min(scaled[:, 0]) - half,
                height - 1.0 - half - np.max(scaled[:, 0]),
                np.min(scaled[:, 1]) - half,
                width - 1.0 - half - np.max(scaled[:, 1]),
            ]
        )
        minimum_edge_margin = float(np.min(margins))
        crop_safe = minimum_edge_margin >= 0.0
    return {
        "valid_fraction": float(np.mean(valid)),
        "maximum_valid_gap_seconds": maximum_gap,
        "raw_sample_rate_hz": float(
            1.0 / np.median(np.diff(subset.t_ephys.to_numpy(dtype=float)))
        ),
        "minimum_crop_edge_margin_px": minimum_edge_margin,
        "crop_safe_at_maximum_motion_scale": bool(crop_safe),
    }


def select_fixations(
    fixations: list[dict[str, Any]],
    ddpi: pd.DataFrame,
    *,
    n_fixations: int,
    min_duration_seconds: float,
    minimum_valid_fraction: float,
    maximum_valid_gap_seconds: float,
    padding_seconds: float,
    crop_size: int,
    maximum_motion_scale: float,
) -> list[dict[str, Any]]:
    candidates = []
    gate_counts = {
        "total": int(len(fixations)),
        "duration": 0,
        "valid_fraction": 0,
        "valid_gap": 0,
        "crop_safe": 0,
        "all": 0,
    }
    for fixation in fixations:
        if float(fixation["duration_seconds"]) < min_duration_seconds:
            continue
        gate_counts["duration"] += 1
        quality = trial_quality(
            fixation,
            ddpi,
            padding_seconds=padding_seconds,
            crop_size=crop_size,
            maximum_motion_scale=maximum_motion_scale,
        )
        row = {**fixation, **quality}
        if row["valid_fraction"] >= minimum_valid_fraction:
            gate_counts["valid_fraction"] += 1
        if row["maximum_valid_gap_seconds"] <= maximum_valid_gap_seconds:
            gate_counts["valid_gap"] += 1
        if row["crop_safe_at_maximum_motion_scale"]:
            gate_counts["crop_safe"] += 1
        if (
            row["valid_fraction"] >= minimum_valid_fraction
            and row["maximum_valid_gap_seconds"] <= maximum_valid_gap_seconds
            and row["crop_safe_at_maximum_motion_scale"]
        ):
            candidates.append(row)
            gate_counts["all"] += 1
    candidates.sort(
        key=lambda row: (
            -float(row["valid_fraction"]),
            float(row["maximum_valid_gap_seconds"]),
            -float(row["duration_seconds"]),
        )
    )
    # Favor image diversity before reusing the same displayed photograph.
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for candidate in candidates:
        if candidate["image_file"] in used_images:
            continue
        selected.append(candidate)
        used_images.add(candidate["image_file"])
        if len(selected) >= n_fixations:
            break
    if len(selected) < n_fixations:
        selected_ids = {row["experiment_index"] for row in selected}
        for candidate in candidates:
            if candidate["experiment_index"] in selected_ids:
                continue
            selected.append(candidate)
            if len(selected) >= n_fixations:
                break
    if len(selected) < n_fixations:
        raise RuntimeError(
            f"only {len(selected)} BackImage fixations passed the requested gates; "
            f"gate counts={gate_counts}"
        )
    return selected


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
    """Filter raw pixel-valued eye position before evaluating it at 240 Hz."""
    if not 0 < passband_hz < stopband_hz < 0.5 / np.median(
        np.diff(ddpi.t_ephys.to_numpy(dtype=float)[: min(len(ddpi), 10000)])
    ):
        raise ValueError("eye filter edges are inconsistent with raw DDPI Nyquist")
    time = ddpi.t_ephys.to_numpy(dtype=float)
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
    # Some optimal IIR designs place the edge of the allowed passband ripple at
    # DC.  Filter only deviations from fixation center so that this harmless
    # gain convention cannot translate the absolute gaze position by pixels.
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


def render_crop(
    image: np.ndarray,
    position_ij_px: np.ndarray,
    *,
    crop_size: int,
    device: str,
) -> np.ndarray:
    """Bilinearly sample a gaze-centered crop from the full displayed image."""
    import torch
    import torch.nn.functional as functional

    value = np.asarray(image, dtype=np.float32)
    if value.ndim == 3:
        value = value.mean(axis=2)
    if value.ndim != 2:
        raise ValueError(f"expected a grayscale image, got shape {value.shape}")
    position = np.asarray(position_ij_px, dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must have shape [time, row/column]")
    half = 0.5 * float(crop_size - 1)
    offset = np.arange(crop_size, dtype=np.float32) - half
    yy, xx = np.meshgrid(offset, offset, indexing="ij")
    row = position[:, 0, None, None] + yy[None]
    column = position[:, 1, None, None] + xx[None]
    if (
        np.min(row) < 0
        or np.max(row) > value.shape[0] - 1
        or np.min(column) < 0
        or np.max(column) > value.shape[1] - 1
    ):
        raise RuntimeError("requested retinal crop leaves the displayed natural image")
    grid = np.stack(
        (
            2.0 * column / float(value.shape[1] - 1) - 1.0,
            2.0 * row / float(value.shape[0] - 1) - 1.0,
        ),
        axis=-1,
    )
    source = torch.from_numpy(value / 255.0).to(device)[None, None]
    source = source.expand(len(position), -1, -1, -1)
    grid_tensor = torch.from_numpy(grid).to(device)
    with torch.no_grad():
        movie = functional.grid_sample(
            source,
            grid_tensor,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
    return movie.cpu().numpy().astype(np.float32, copy=False)


def spatial_bin_contract(
    crop_size: int,
    ppd: float,
    *,
    minimum_cpd: float,
    maximum_cpd: float,
    n_bins: int,
) -> dict[str, np.ndarray]:
    edges = np.geomspace(minimum_cpd, maximum_cpd, int(n_bins) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ky = np.fft.fftfreq(crop_size, d=1.0 / ppd)
    kx = np.fft.rfftfreq(crop_size, d=1.0 / ppd)
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    radial = np.hypot(xx, yy).reshape(-1)
    index = np.searchsorted(edges, radial, side="right") - 1
    valid = (index >= 0) & (index < len(centers))
    # Folding temporal +/- frequencies accounts for the omitted negative-kx
    # conjugate.  Modes on the rFFT boundary are half weighted because their
    # spatial conjugates are also present in the retained ky rows.
    mode_weight = np.ones_like(radial)
    boundary = np.isclose(xx.reshape(-1), 0.0)
    if crop_size % 2 == 0:
        boundary |= np.isclose(xx.reshape(-1), 0.5 * ppd)
    mode_weight[boundary] = 0.5
    count = np.bincount(
        index[valid], weights=mode_weight[valid], minlength=len(centers)
    )
    return {
        "edges_cpd": edges,
        "centers_cpd": centers,
        "flat_index": index,
        "valid": valid,
        "mode_weight": mode_weight,
        "mode_count": count,
    }


def folded_segment_power(
    movie: np.ndarray,
    *,
    frame_rate_hz: float,
    binning: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return direction-folded temporal PSD averaged within radial SF bins."""
    value = np.asarray(movie, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("movie must have shape [time, height, width]")
    mean_luminance = float(np.mean(value))
    contrast = value / max(mean_luminance, EPS) - 1.0
    contrast -= contrast.mean(axis=0, keepdims=True)
    spatial_1d = signal.windows.tukey(value.shape[1], alpha=0.15, sym=False)
    spatial_window = np.outer(spatial_1d, spatial_1d).astype(np.float32)
    temporal_window = signal.windows.hann(value.shape[0], sym=False).astype(np.float32)
    coefficient = np.fft.rfft2(
        contrast * spatial_window[None], axes=(-2, -1), norm="ortho"
    )
    spectrum = np.fft.fft(
        coefficient * temporal_window[:, None, None], axis=0
    )
    raw_power = np.square(np.abs(spectrum)) / max(
        frame_rate_hz * float(np.sum(np.square(temporal_window))), EPS
    )
    raw_power /= max(float(np.mean(np.square(spatial_window))), EPS)
    n_time = value.shape[0]
    positive_count = n_time // 2 + 1
    folded = raw_power[:positive_count].copy()
    if n_time % 2 == 0:
        interior = np.arange(1, positive_count - 1)
    else:
        interior = np.arange(1, positive_count)
    folded[interior] += raw_power[-interior]
    temporal_hz = (
        np.arange(positive_count, dtype=np.float64) * frame_rate_hz / float(n_time)
    )

    flat_index = binning["flat_index"]
    valid = binning["valid"]
    weight = binning["mode_weight"]
    count = binning["mode_count"]
    output = np.zeros((positive_count, len(count)), dtype=np.float64)
    for temporal_index in range(positive_count):
        flat_power = folded[temporal_index].reshape(-1)
        summed = np.bincount(
            flat_index[valid],
            weights=flat_power[valid] * weight[valid],
            minlength=len(count),
        )
        output[temporal_index] = np.divide(
            summed, count, out=np.zeros_like(summed), where=count > 0
        )
    return temporal_hz, output


def welch_starts(n_time: int, n_per_segment: int, overlap_fraction: float) -> np.ndarray:
    if not 0 <= overlap_fraction < 1:
        raise ValueError("welch-overlap-fraction must lie in [0, 1)")
    step = max(int(round(n_per_segment * (1.0 - overlap_fraction))), 1)
    if n_time < n_per_segment:
        return np.empty(0, dtype=int)
    return np.arange(0, n_time - n_per_segment + 1, step, dtype=int)


def render_power_figure(
    path: Path,
    *,
    scales: np.ndarray,
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    power: np.ndarray,
    frame_rate_hz: float,
) -> None:
    keep_tf = (temporal_hz > 0) & (
        temporal_hz <= 0.8 * 0.5 * float(frame_rate_hz)
    )
    log_power = np.log10(np.maximum(power[:, keep_tf], EPS))
    finite = log_power[np.isfinite(log_power)]
    lower, upper = np.quantile(finite, (0.03, 0.995))
    levels = np.linspace(lower, upper, 15)
    figure, axes = plt.subplots(
        2,
        len(scales),
        figsize=(3.45 * len(scales), 6.1),
        constrained_layout=True,
        squeeze=False,
    )
    contour = None
    slice_targets = (1.0, 2.0, 4.0, 8.0)
    slice_colors = ("#3CB371", "#25858E", "#3B528B", "#5B2A86")
    for scale_index, scale in enumerate(scales):
        display = gaussian_filter(
            log_power[scale_index], sigma=(0.65, 0.65), mode="nearest"
        )
        contour = axes[0, scale_index].contourf(
            spatial_cpd,
            temporal_hz[keep_tf],
            display,
            levels=levels,
            cmap="magma",
            extend="both",
        )
        axes[0, scale_index].set(
            xscale="log",
            yscale="log",
            xlabel="spatial frequency (cycles/degree)",
            title=f"{scale:g}× measured eye motion",
        )
        if scale_index == 0:
            axes[0, scale_index].set_ylabel("temporal frequency (Hz)")
        axes[0, scale_index].grid(color="white", alpha=0.14, linewidth=0.6)

        for target, color in zip(slice_targets, slice_colors):
            spatial_index = int(np.argmin(np.abs(np.log(spatial_cpd / target))))
            axes[1, scale_index].plot(
                temporal_hz[keep_tf],
                log_power[scale_index, :, spatial_index],
                color=color,
                lw=1.8,
                label=f"{spatial_cpd[spatial_index]:.1f} c/deg",
            )
        axes[1, scale_index].set(
            xscale="log",
            xlabel="temporal frequency (Hz)",
            title="temporal slices",
        )
        if scale_index == 0:
            axes[1, scale_index].set_ylabel("log10 power density")
        axes[1, scale_index].grid(alpha=0.17)
    axes[1, -1].legend(frameon=False, fontsize=8)
    if contour is not None:
        colorbar = figure.colorbar(contour, ax=list(axes[0]), shrink=0.87, pad=0.015)
        colorbar.set_label("log10 retinal contrast power density")
    figure.suptitle(
        "Real long fixations redistribute natural-image power across SF and TF",
        fontsize=13.5,
        fontweight="semibold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.n_fixations < 1:
        raise ValueError("n-fixations must be positive")
    if args.crop_size < 95 or args.crop_size % 2 != 1:
        raise ValueError("crop-size must be odd and at least 95 pixels")
    if not np.all(np.asarray(args.scales) > 0):
        raise ValueError("all motion scales must be positive")
    trials, experiment_metadata = load_backimage_trials(args.experiment_mat.resolve())
    fixations = fixation_segments(
        trials,
        args.saccades_json.resolve(),
        guard_seconds=float(args.saccade_guard_seconds),
    )
    ddpi = load_ddpi(args.ddpi_csv.resolve())
    selected = select_fixations(
        fixations,
        ddpi,
        n_fixations=int(args.n_fixations),
        min_duration_seconds=max(
            float(args.min_fixation_seconds), float(args.welch_seconds)
        ),
        minimum_valid_fraction=float(args.minimum_valid_fraction),
        maximum_valid_gap_seconds=float(args.maximum_valid_gap_seconds),
        padding_seconds=float(args.eye_filter_padding_seconds),
        crop_size=int(args.crop_size),
        maximum_motion_scale=float(np.max(np.asarray(args.scales, dtype=float))),
    )
    ppd = float(experiment_metadata["ppd"])
    n_per_segment = int(round(args.welch_seconds * args.target_rate_hz))
    binning = spatial_bin_contract(
        int(args.crop_size),
        ppd,
        minimum_cpd=float(args.min_spatial_cpd),
        maximum_cpd=float(args.max_spatial_cpd),
        n_bins=int(args.n_spatial_bins),
    )
    scales = np.asarray(args.scales, dtype=float)
    accumulator: np.ndarray | None = None
    temporal_hz: np.ndarray | None = None
    n_windows = np.zeros(len(scales), dtype=int)
    trial_audit = []

    for fixation_number, trial in enumerate(selected, start=1):
        eye = anti_alias_eye_position(
            ddpi,
            start_ephys=float(trial["start_ephys"]),
            stop_ephys=float(trial["stop_ephys"]),
            target_rate_hz=float(args.target_rate_hz),
            passband_hz=float(args.eye_passband_hz),
            stopband_hz=float(args.eye_stopband_hz),
            padding_seconds=float(args.eye_filter_padding_seconds),
        )
        position = np.asarray(eye["filtered_position_px"], dtype=float)
        fixation_center = np.median(position, axis=0)
        starts = welch_starts(
            len(position), n_per_segment, float(args.welch_overlap_fraction)
        )
        image = trial["trial"].get_image()
        for scale_index, scale in enumerate(scales):
            scaled = fixation_center + scale * (position - fixation_center)
            for start in starts:
                segment_position = scaled[start : start + n_per_segment]
                movie = render_crop(
                    image,
                    segment_position,
                    crop_size=int(args.crop_size),
                    device=str(args.device),
                )
                frequencies, segment_power = folded_segment_power(
                    movie,
                    frame_rate_hz=float(args.target_rate_hz),
                    binning=binning,
                )
                if accumulator is None:
                    temporal_hz = frequencies
                    accumulator = np.zeros(
                        (len(scales), len(frequencies), args.n_spatial_bins),
                        dtype=np.float64,
                    )
                elif not np.array_equal(temporal_hz, frequencies):
                    raise RuntimeError("Welch frequency grid changed across segments")
                accumulator[scale_index] += segment_power
                n_windows[scale_index] += 1
        trial_audit.append(
            {
                key: (
                    value
                    if isinstance(value, (str, int, float, bool)) or value is None
                    else str(value)
                )
                for key, value in trial.items()
                if key != "trial"
            }
            | {
                "n_target_samples": int(len(position)),
                "n_welch_windows": int(len(starts)),
                "raw_eye_sample_rate_hz": float(eye["source_rate_hz"]),
                "filtered_vs_unfiltered_rms_difference_px": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.asarray(eye["filtered_position_px"])
                                - np.asarray(eye["unfiltered_position_px"])
                            )
                        )
                    )
                ),
            }
        )
        print(
            f"real BackImage spectrum fixation {fixation_number}/{len(selected)}: "
            f"{trial['image_file']} · {trial['duration_seconds']:.2f} s · "
            f"{len(starts)} Welch windows",
            flush=True,
        )

    if accumulator is None or temporal_hz is None or np.any(n_windows == 0):
        raise RuntimeError("no valid retinal-movie Welch windows were accumulated")
    power = accumulator / n_windows[:, None, None]
    resolved_spatial = np.asarray(binning["mode_count"]) > 0
    if np.count_nonzero(resolved_spatial) < 8:
        raise RuntimeError("fewer than eight radial SF bins contain Fourier modes")
    spatial_cpd = np.asarray(binning["centers_cpd"])[resolved_spatial]
    power = power[:, :, resolved_spatial]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "real_backimage_power.npz"
    np.savez_compressed(
        archive_path,
        scales=scales,
        spatial_cpd=spatial_cpd,
        spatial_edges_cpd=binning["edges_cpd"],
        requested_spatial_centers_cpd=binning["centers_cpd"],
        resolved_spatial_bins=resolved_spatial,
        spatial_mode_count=binning["mode_count"],
        temporal_hz=temporal_hz,
        power=power.astype(np.float32),
        n_welch_windows=n_windows,
    )
    figure_path = args.out_dir / "real_backimage_power.png"
    render_power_figure(
        figure_path,
        scales=scales,
        spatial_cpd=spatial_cpd,
        temporal_hz=temporal_hz,
        power=power,
        frame_rate_hz=float(args.target_rate_hz),
    )
    summary = {
        "analysis": "real long-fixation BackImage retinal SF-TF power",
        "experiment_mat": str(args.experiment_mat.resolve()),
        "ddpi_csv": str(args.ddpi_csv.resolve()),
        "experiment_metadata": experiment_metadata,
        "n_fixations": int(len(selected)),
        "fixations": trial_audit,
        "saccades_json": str(args.saccades_json.resolve()),
        "saccade_guard_seconds": float(args.saccade_guard_seconds),
        "target_rate_hz": float(args.target_rate_hz),
        "eye_filter": {
            "kind": "zero-phase IIR design on a uniform raw-DDPI grid",
            "passband_hz": float(args.eye_passband_hz),
            "stopband_hz": float(args.eye_stopband_hz),
            "passband_ripple_db": 0.1,
            "stopband_attenuation_db": 60.0,
            "filter_before_240hz_sampling": True,
        },
        "rendering": {
            "source": "BackImageTrial.get_image full displayed natural image",
            "subpixel_sampler": "torch grid_sample bilinear align_corners=True",
            "crop_size_px": int(args.crop_size),
            "ppd": ppd,
            "motion_scale_center": "per-trial median filtered gaze position",
        },
        "spectrum": {
            "method": "overlapping Welch windows; 2-D spatial rFFT then folded temporal FFT",
            "window_seconds": float(args.welch_seconds),
            "overlap_fraction": float(args.welch_overlap_fraction),
            "spatial_window": "2-D Tukey alpha=0.15",
            "temporal_window": "periodic Hann",
            "temporal_mean_removed_per_pixel_per_window": True,
            "radial_statistic": "mean power per Fourier mode in logarithmic SF bins",
            "empty_radial_bins_removed_before_display": True,
            "display_smoothing_enters_statistics": False,
        },
        "scales": scales.tolist(),
        "n_welch_windows": n_windows.tolist(),
        "archive": str(archive_path.resolve()),
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
