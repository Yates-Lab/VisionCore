"""Production twininfo analysis pipeline.

The pipeline mirrors the analysis outline in the README and intentionally keeps
slow work behind explicit flags:

1. Eye movement selector and visualization.
2. Image selector, pyramid QC, and crop-hotspot selection.
3. Retinal stimulus movies, if requested.
4. Activation-map movies, if requested.
5. Individual cumulative information traces.
6. Spatial-frequency cumulative information traces.
7. Final information-gain summary figures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .activation_movies import make_activation_movies
from .common import DEFAULT_CCMAX_THRESHOLD, DT, N_LAGS, OUTPUT_DIR, load_digital_twin, write_json
from .image_selection import PYRAMID_HEIGHT, PYRAMID_ORDER, SF_BANDS_4, crop_rows, select_image_crops
from .io_utils import ensure_run_dirs, write_csv
from .lagcube_information import (
    approximate_unit_fisher_scores,
    block_current_samples,
    block_endpoint_lag_cubes,
    cross_shift_grid,
    cumulative_pattern_fisher,
    cumulative_spatial_ssi,
    final_metric_row,
    finite_difference_shift_set,
    run_shifted_lag_cube_rate_maps,
    square_shift_grid,
    unique_shifts,
)
from .retinal_examples import TraceExample, model_lag_cubes_from_image_trace, pyramid_local_image_controls
from .retinal_movies import CONDITION_LABELS, make_representative_stimulus_movies
from .population import (
    GRID_POSITION_MODES,
    PERFORMANCE_METRICS,
    POPULATION_SELECTION_MODES,
    build_analysis_population,
)
from .trace_selection import run_trace_selection_step


PHASE_CONDITIONS = ("real", "stabilized", "pyramid_phase_scrambled")
SF_CONDITIONS = ("real",) + SF_BANDS_4
MAIN_CONDITIONS = ("real", "stabilized", "pyramid_phase_scrambled") + SF_BANDS_4

CONDITION_COLORS = {
    "real": "#1f77b4",
    "stabilized": "#2ca02c",
    "pyramid_phase_scrambled": "#d62728",
    "sf_low": "#9467bd",
    "sf_mid_low": "#8c564b",
    "sf_mid_high": "#ff7f0e",
    "sf_high": "#17becf",
}


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str | None = None
    seed: int = 0
    image_indices: tuple[int, ...] | None = None
    n_crops_per_image: int = 3
    n_examples_per_kind: int = 10
    selected_trace_example_ids: tuple[str, ...] = ()
    t_max: int = 128
    stride: int = 8
    population_size: int = 100
    population_selection: str = "top_performance"
    performance_metric: str = "ccnorm"
    min_performance_score: float | None = None
    population_grid_position_mode: str = "full_grid"
    population_grid_stride: int = 1
    deduplicate_units: bool = True
    dedupe_correlation_threshold: float = 0.95
    dedupe_candidate_multiplier: float = 5.0
    dedupe_battery_frames: int = 96
    ccmax_threshold: float = DEFAULT_CCMAX_THRESHOLD
    batch_size: int = 64
    fisher_step_arcmin: float = 0.5
    shift_grid_max_arcmin: float = 1.0
    shift_grid_step_arcmin: float = 1.0
    shift_grid_mode: str = "square"
    make_stimulus_movies: bool = False
    make_activation_movies: bool = False
    movie_examples_per_kind: int = 1
    movie_fps: int = 30
    recompute: bool = False


def _jsonable_config(config: PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["image_indices"] = None if config.image_indices is None else list(config.image_indices)
    payload["selected_trace_example_ids"] = list(config.selected_trace_example_ids)
    payload["analysis_version"] = "production_pyramid_spatial_ssi_v2"
    payload["conditions"] = list(MAIN_CONDITIONS)
    payload["pyramid_height"] = PYRAMID_HEIGHT
    payload["pyramid_order"] = PYRAMID_ORDER
    return payload


def _folder_safe(value: str) -> str:
    """Return a short filesystem-safe label that remains readable."""
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "analysis"


def run_slug(config: PipelineConfig) -> str:
    """Human-readable analysis folder name.

    Output-generation flags such as MP4 generation are intentionally excluded:
    they change which artifacts are written, but not the analysis definition.
    Use ``--run-name`` when a fixed folder name is preferred.
    """
    if config.run_name:
        return _folder_safe(config.run_name)

    if config.image_indices is None:
        image_label = "all-images"
    else:
        image_label = "images-" + "-".join(str(idx) for idx in config.image_indices)

    trace_label = f"{config.n_examples_per_kind}fix-{config.n_examples_per_kind}ms"
    crop_label = f"{config.n_crops_per_image}crop" if config.n_crops_per_image == 1 else f"{config.n_crops_per_image}crops"
    unit_label = f"{config.population_size}units-per-pixel"
    selection_label = config.performance_metric if config.population_selection == "top_performance" else config.population_selection
    if config.min_performance_score is not None:
        threshold = f"{float(config.min_performance_score):.3g}".replace(".", "p")
        selection_label = f"{selection_label}-ge-{threshold}"
    grid_label = f"{config.shift_grid_mode}-grid"
    pop_grid_label = (
        f"{config.population_grid_position_mode}-s{config.population_grid_stride}"
        if config.population_grid_position_mode == "full_grid"
        else config.population_grid_position_mode
    )
    t_label = f"{config.t_max}frames"
    seed_label = f"seed-{config.seed}"
    return _folder_safe(
        f"{image_label}_{crop_label}_{trace_label}_{unit_label}_{selection_label}_{pop_grid_label}_{grid_label}_{t_label}_{seed_label}"
    )


def _example_seed(seed: int, example_id: str, image_index: int, crop_rank: int) -> int:
    payload = f"{seed}:{example_id}:{image_index}:{crop_rank}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def _filter_trace_examples(config: PipelineConfig, examples: list[TraceExample]) -> list[TraceExample]:
    if not config.selected_trace_example_ids:
        return examples
    by_id = {example.example_id: example for example in examples}
    missing = [example_id for example_id in config.selected_trace_example_ids if example_id not in by_id]
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(f"Missing selected trace IDs {missing}. Available: {available}")
    return [by_id[example_id] for example_id in config.selected_trace_example_ids]


def _condition_blocks(
    *,
    condition: str,
    image: np.ndarray,
    trace: np.ndarray,
    t_max: int,
    crop_center_offset_px: tuple[float, float],
    real_cubes: np.ndarray,
    control_images: dict[str, np.ndarray],
) -> np.ndarray:
    """Return one overlapping model lag cube per aligned movie sample."""
    if condition == "real":
        blocks, _current = block_endpoint_lag_cubes(real_cubes)
        return blocks.astype(np.float32)
    if condition == "stabilized":
        stable_trace = np.repeat(np.mean(trace[:t_max], axis=0, keepdims=True), t_max, axis=0).astype(np.float32)
        cubes = model_lag_cubes_from_image_trace(
            image,
            stable_trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
        blocks, _current = block_endpoint_lag_cubes(cubes)
        return blocks.astype(np.float32)
    if condition in control_images:
        cubes = model_lag_cubes_from_image_trace(
            control_images[condition],
            trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
        blocks, _current = block_endpoint_lag_cubes(cubes)
        return blocks.astype(np.float32)
    raise ValueError(f"Unsupported condition: {condition}")


def _ci95(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    correction = np.sqrt(n / np.maximum(n - 1, 1))
    sem = std * correction / np.sqrt(np.maximum(n, 1))
    sem = np.where(n > 1, sem, 0.0)
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def _write_series_npz(path: Path, records: list[dict[str, Any]], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    for key in ("example_id", "kind", "condition"):
        payload[f"record_{key}"] = np.asarray([str(row[key]) for row in records])
    payload["record_image_index"] = np.asarray([int(row["image_index"]) for row in records], dtype=np.int32)
    payload["record_crop_rank"] = np.asarray([int(row["crop_rank"]) for row in records], dtype=np.int32)
    np.savez_compressed(path, **payload)


def _plot_individual_traces(
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    metric_key: str,
    ylabel: str,
    conditions: tuple[str, ...],
    path: Path,
) -> None:
    time_s = arrays["time_s"]
    fig, axs = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for ax, kind in zip(axs, ("fixation", "microsaccade"), strict=True):
        for condition in conditions:
            color = CONDITION_COLORS[condition]
            ix = [
                i for i, row in enumerate(records)
                if row["kind"] == kind and row["condition"] == condition
            ]
            for i in ix:
                ax.plot(time_s, arrays[metric_key][i], color=color, alpha=0.25, lw=0.8)
            if ix:
                ax.plot([], [], color=color, lw=2.0, label=CONDITION_LABELS[condition])
        ax.set_title(f"{kind}: individual image/crop/trace movies")
        ax.set_ylabel(ylabel)
        ax.grid(color="0.9", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[-1].set_xlabel("time in movie (s)")
    axs[0].legend(frameon=False, loc="upper left", ncols=min(4, len(conditions)))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_average_overview(
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    conditions: tuple[str, ...],
    title: str,
    path: Path,
) -> None:
    time_s = arrays["time_s"]
    metrics = (
        ("cumulative_fisher_pattern", "cumulative pattern FI"),
        ("cumulative_fisher_pattern_per_spike", "pattern FI / expected spike"),
        ("cumulative_spatial_ssi_bits", "cumulative spatial SSI (bits)"),
    )
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 7.6), sharex=True)
    for r, kind in enumerate(("fixation", "microsaccade")):
        for c, (metric_key, ylabel) in enumerate(metrics):
            ax = axs[r, c]
            for condition in conditions:
                ix = [
                    i for i, row in enumerate(records)
                    if row["kind"] == kind and row["condition"] == condition
                ]
                if not ix:
                    continue
                mean, lo, hi = _ci95(arrays[metric_key][ix])
                color = CONDITION_COLORS[condition]
                ax.plot(time_s, mean, color=color, lw=2.0, label=CONDITION_LABELS[condition])
                ax.fill_between(time_s, lo, hi, color=color, alpha=0.16, linewidth=0)
            if r == 0:
                ax.set_title(ylabel)
            if c == 0:
                ax.set_ylabel(kind)
            if r == 1:
                ax.set_xlabel("time in movie (s)")
            ax.grid(color="0.9", lw=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axs[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _final_metric_lookup(rows: list[dict[str, Any]], metric: str) -> dict[tuple[str, int, int, str], float]:
    return {
        (str(row["example_id"]), int(row["image_index"]), int(row["crop_rank"]), str(row["condition"])): float(row[metric])
        for row in rows
    }


def _plot_gain_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot condition-minus-stabilized final information gains."""
    metrics = (
        ("final_cumulative_fisher_pattern", "final pattern FI gain"),
        ("final_cumulative_spatial_ssi_bits", "final cumulative spatial SSI gain (bits)"),
    )
    gain_conditions = ("real", "pyramid_phase_scrambled") + SF_BANDS_4
    fig, axs = plt.subplots(2, 2, figsize=(14.0, 8.4), sharex=True)
    x = np.arange(len(gain_conditions), dtype=np.float64)
    for r, kind in enumerate(("fixation", "microsaccade")):
        for c, (metric, title) in enumerate(metrics):
            ax = axs[r, c]
            values = _final_metric_lookup(rows, metric)
            means = []
            errs = []
            for condition in gain_conditions:
                deltas = []
                for row in rows:
                    if row["kind"] != kind or row["condition"] != condition:
                        continue
                    key = (str(row["example_id"]), int(row["image_index"]), int(row["crop_rank"]), "stabilized")
                    if key in values:
                        deltas.append(float(row[metric]) - values[key])
                arr = np.asarray(deltas, dtype=np.float64)
                if arr.size:
                    mean, lo, hi = _ci95(arr[:, None])
                    means.append(float(mean[0]))
                    errs.append(float(hi[0] - mean[0]))
                else:
                    means.append(float("nan"))
                    errs.append(0.0)
            colors = [CONDITION_COLORS[condition] for condition in gain_conditions]
            ax.bar(x, means, yerr=errs, capsize=3, color=colors, alpha=0.86)
            ax.axhline(0.0, color="0.15", lw=1.0)
            ax.set_title(f"{kind}: {title}")
            ax.set_xticks(x)
            ax.set_xticklabels([CONDITION_LABELS[cnd] for cnd in gain_conditions], rotation=25, ha="right")
            ax.grid(axis="y", color="0.9", lw=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Average information gain relative to stabilized movies; error bars are approximate 95% CI", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_final_metric_summary(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = (
        ("final_cumulative_fisher_pattern", "final pattern FI"),
        ("final_cumulative_fisher_pattern_per_spike", "final pattern FI/spike"),
        ("final_cumulative_spatial_ssi_bits", "final cumulative spatial SSI (bits)"),
    )
    x = np.arange(len(MAIN_CONDITIONS), dtype=np.float64)
    width = 0.34
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, (metric, title) in zip(axs, metrics, strict=True):
        for k_i, kind in enumerate(("fixation", "microsaccade")):
            means = []
            errs = []
            for condition in MAIN_CONDITIONS:
                vals = np.asarray([
                    float(row[metric]) for row in rows
                    if row["kind"] == kind and row["condition"] == condition
                ], dtype=np.float64)
                mean, lo, hi = _ci95(vals[:, None])
                means.append(float(mean[0]))
                errs.append(float(hi[0] - mean[0]))
            ax.bar(x + (k_i - 0.5) * width, means, width=width, yerr=errs, capsize=3, alpha=0.86, label=kind)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[cnd] for cnd in MAIN_CONDITIONS], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_information_step(
    *,
    config: PipelineConfig,
    dirs: dict[str, Path],
    model: Any,
    population: Any,
    device: Any,
    examples: list[TraceExample],
    image_by_index: dict[int, np.ndarray],
    crops: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, int, str], np.ndarray]]:
    """Run model responses and cumulative information on all image/crop/trace pairs."""
    shift_grid = (
        square_shift_grid(config.shift_grid_max_arcmin, config.shift_grid_step_arcmin)
        if config.shift_grid_mode == "square"
        else cross_shift_grid(config.shift_grid_max_arcmin, config.shift_grid_step_arcmin)
    )
    fisher_shifts = finite_difference_shift_set(config.fisher_step_arcmin)
    all_shifts = unique_shifts(fisher_shifts)

    series_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pyramid_audit_rows: list[dict[str, Any]] = []
    unit_scores: dict[tuple[str, int, int, str], np.ndarray] = {}
    unit_score_rows: list[dict[str, Any]] = []
    series_arrays: dict[str, list[np.ndarray]] = {
        "cumulative_fisher_pattern": [],
        "cumulative_fisher_pattern_per_spike": [],
        "cumulative_expected_spikes": [],
        "cumulative_ssi_total_bits_per_window": [],
        "cumulative_ssi_pattern_bits_per_window": [],
        "prefix_ssi_total_bits_per_spike": [],
        "prefix_ssi_pattern_bits_per_spike": [],
        "cumulative_ssi_bits_per_second": [],
        "spatial_ssi_bits_per_spike": [],
        "spatial_ssi_bits_per_second": [],
        "cumulative_spatial_ssi_bits": [],
        "cumulative_spatial_ssi_bits_per_spike": [],
        "cumulative_spatial_ssi_bits_per_second": [],
        "cumulative_spatial_ssi_expected_spikes": [],
    }

    pair_count = len(examples) * len(crops)
    pair_i = 0
    for crop in crops:
        image_index = int(crop["image_index"])
        crop_rank = int(crop["crop_rank"])
        crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
        image = image_by_index[image_index]
        for example in examples:
            pair_i += 1
            print(
                f"Pair {pair_i}/{pair_count}: {example.example_id} image={image_index} crop={crop_rank}",
                flush=True,
            )
            seed = _example_seed(config.seed, example.example_id, image_index, crop_rank)
            real_cubes = model_lag_cubes_from_image_trace(
                image,
                example.trace,
                t_max=config.t_max,
                crop_center_offset_px=crop_offset,
            )
            control_images, audits = pyramid_local_image_controls(
                image,
                example.trace,
                np.random.default_rng(seed),
                crop_center_offset_px=crop_offset,
                height=PYRAMID_HEIGHT,
                order=PYRAMID_ORDER,
                sf_bands=SF_BANDS_4,
            )
            for audit in audits:
                row = dict(audit)
                row.update({
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "crop_center_offset_x_px": float(crop_offset[0]),
                    "crop_center_offset_y_px": float(crop_offset[1]),
                })
                pyramid_audit_rows.append(row)

            for condition in MAIN_CONDITIONS:
                print(f"  condition={condition}", flush=True)
                cubes = _condition_blocks(
                    condition=condition,
                    image=image,
                    trace=example.trace,
                    t_max=config.t_max,
                    crop_center_offset_px=crop_offset,
                    real_cubes=real_cubes,
                    control_images=control_images,
                )
                rate_maps_by_shift = run_shifted_lag_cube_rate_maps(
                    model,
                    population,
                    device,
                    cubes,
                    all_shifts,
                    batch_size=config.batch_size,
                )
                fisher = cumulative_pattern_fisher(
                    rate_maps_by_shift,
                    fisher_step_arcmin=config.fisher_step_arcmin,
                    dt=DT,
                )
                center_rate_map = rate_maps_by_shift[(0.0, 0.0)]
                ssi = cumulative_spatial_ssi(center_rate_map, dt=DT)
                row = final_metric_row(
                    example_id=example.example_id,
                    kind=example.kind,
                    image_index=image_index,
                    condition=condition,
                    fisher=fisher,
                    ssi=ssi,
                    psd_errors=None,
                )
                row.update({
                    "crop_rank": crop_rank,
                    "crop_center_offset_x_px": float(crop_offset[0]),
                    "crop_center_offset_y_px": float(crop_offset[1]),
                    "n_events_in_window": int(example.n_events_in_window),
                    "event_onset": example.event_onset,
                    "fisher_step_arcmin": float(config.fisher_step_arcmin),
                    "shift_grid_max_arcmin": float(config.shift_grid_max_arcmin),
                    "shift_grid_step_arcmin": float(config.shift_grid_step_arcmin),
                    "shift_grid_mode": config.shift_grid_mode,
                    "n_shift_grid_states": int(shift_grid.shape[0]),
                    "spatial_ssi_uses_shift_grid": False,
                    "rate_map_time_samples": int(center_rate_map.shape[0]),
                    "rate_map_units_per_pixel": int(center_rate_map.shape[1]),
                    "rate_map_spatial_bins": int(center_rate_map.shape[2] * center_rate_map.shape[3]),
                })
                summary_rows.append(row)
                series_records.append({
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                })
                for key in series_arrays:
                    source = fisher if key in fisher else ssi
                    series_arrays[key].append(np.asarray(source[key], dtype=np.float32))
                score = approximate_unit_fisher_scores(fisher["mu0"], fisher["dmu"])
                unit_scores[(example.example_id, image_index, crop_rank, condition)] = score
                top = np.argsort(score)[-min(12, score.size):][::-1]
                unit_score_rows.append({
                    "example_id": example.example_id,
                    "kind": example.kind,
                    "image_index": image_index,
                    "crop_rank": crop_rank,
                    "condition": condition,
                    "top_unit_indices": top.tolist(),
                    "top_unit_scores": score[top].tolist(),
                })

    arrays_np = {key: np.stack(vals, axis=0).astype(np.float32) for key, vals in series_arrays.items()}
    arrays_np["analysis_sample_index"] = block_current_samples(config.t_max, n_lags=N_LAGS).astype(np.int32)
    arrays_np["time_s"] = (arrays_np["analysis_sample_index"].astype(np.float32) * DT).astype(np.float32)
    _write_series_npz(dirs["cache"] / "cumulative_information_series.npz", series_records, arrays_np)
    write_csv(summary_rows, dirs["metadata"] / "05_lagcube_information_summary.csv")
    write_csv(series_records, dirs["metadata"] / "05_information_series_records.csv")
    write_csv(unit_score_rows, dirs["metadata"] / "04_unit_score_summary.csv")
    write_csv(pyramid_audit_rows, dirs["metadata"] / "02_pyramid_image_control_audit.csv")

    _plot_individual_traces(
        series_records,
        arrays_np,
        metric_key="cumulative_fisher_pattern",
        ylabel="cumulative pattern FI",
        conditions=PHASE_CONDITIONS,
        path=dirs["figures"] / "05_individual_cumulative_pattern_fi_phase.pdf",
    )
    _plot_individual_traces(
        series_records,
        arrays_np,
        metric_key="cumulative_spatial_ssi_bits",
        ylabel="cumulative spatial SSI (bits)",
        conditions=PHASE_CONDITIONS,
        path=dirs["figures"] / "05_individual_cumulative_spatial_ssi_phase.pdf",
    )
    _plot_average_overview(
        series_records,
        arrays_np,
        conditions=PHASE_CONDITIONS,
        title="Phase-control averages; bands are approximate 95% CI",
        path=dirs["figures"] / "05_phase_information_average_overview.pdf",
    )
    _plot_individual_traces(
        series_records,
        arrays_np,
        metric_key="cumulative_fisher_pattern",
        ylabel="cumulative pattern FI",
        conditions=SF_CONDITIONS,
        path=dirs["figures"] / "06_individual_cumulative_pattern_fi_sf.pdf",
    )
    _plot_average_overview(
        series_records,
        arrays_np,
        conditions=SF_CONDITIONS,
        title="Spatial-frequency averages; bands preserve natural pyramid energy",
        path=dirs["figures"] / "06_sf_information_average_overview.pdf",
    )
    _plot_final_metric_summary(summary_rows, dirs["figures"] / "07_final_metric_summary.pdf")
    _plot_gain_summary(summary_rows, dirs["figures"] / "07_information_gain_vs_stabilized.pdf")

    write_json(dirs["metadata"] / "05_information_grid.json", {
        "n_trace_examples": len(examples),
        "n_image_crops": len(crops),
        "n_pairs": len(examples) * len(crops),
        "conditions": list(MAIN_CONDITIONS),
        "phase_conditions": list(PHASE_CONDITIONS),
        "sf_conditions": list(SF_CONDITIONS),
        "shift_grid_deg": shift_grid.tolist(),
        "all_shifts_deg": all_shifts.tolist(),
        "analysis_sample_index": arrays_np["analysis_sample_index"].tolist(),
    })
    return summary_rows, series_records, unit_scores


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    run_dir = OUTPUT_DIR / run_slug(config)
    dirs = ensure_run_dirs(run_dir)
    summary_path = dirs["metadata"] / "run_summary.json"
    if summary_path.exists() and not config.recompute:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    write_json(dirs["metadata"] / "run_config.json", _jsonable_config(config))

    model, _model_info, device = load_digital_twin()
    rng = np.random.default_rng(config.seed)
    population, population_rows = build_analysis_population(
        model,
        N=config.population_size,
        rng=rng,
        selection=config.population_selection,
        performance_metric=config.performance_metric,
        min_performance_score=config.min_performance_score,
        ccmax_threshold=config.ccmax_threshold,
        grid_position_mode=config.population_grid_position_mode,
        grid_stride=config.population_grid_stride,
        deduplicate_units=config.deduplicate_units,
        dedupe_correlation_threshold=config.dedupe_correlation_threshold,
        dedupe_candidate_multiplier=config.dedupe_candidate_multiplier,
        dedupe_battery_frames=config.dedupe_battery_frames,
        dedupe_battery_seed=config.seed,
        dedupe_batch_size=config.batch_size,
    )
    write_csv(population_rows, dirs["metadata"] / "00_population_units.csv")
    n_biological_twins = len({int(row["global_unit_idx"]) for row in population_rows}) if population_rows else 0

    examples = run_trace_selection_step(
        figure_dir=dirs["figures"],
        metadata_dir=dirs["metadata"],
        seed=config.seed,
        n_examples_per_kind=config.n_examples_per_kind,
        t_max=config.t_max,
        stride=config.stride,
        model=model,
    )
    examples = _filter_trace_examples(config, examples)
    write_csv(
        [
            {
                "example_id": example.example_id,
                "kind": example.kind,
                "source_trace_index": example.source_trace_index,
                "window_start": example.window_start,
                "window_stop": example.window_stop,
                "n_events_in_window": example.n_events_in_window,
                "event_onset": example.event_onset,
            }
            for example in examples
        ],
        dirs["metadata"] / "01_trace_examples_used.csv",
    )

    image_by_index, image_crops, image_figures = select_image_crops(
        image_indices=config.image_indices,
        n_crops_per_image=config.n_crops_per_image,
        figure_dir=dirs["figures"],
        metadata_path=dirs["metadata"] / "02_image_crop_hotspots.csv",
        seed=config.seed,
        t_max=config.t_max,
        trace_arrays=[example.trace for example in examples],
    )
    crop_table = crop_rows(image_crops)

    stimulus_movies: list[str] = []
    if config.make_stimulus_movies:
        stimulus_movies = make_representative_stimulus_movies(
            examples=examples,
            image_by_index=image_by_index,
            crop_rows=crop_table,
            movie_dir=dirs["movies"],
            figure_dir=dirs["figures"],
            seed=config.seed,
            t_max=config.t_max,
            fps=config.movie_fps,
            max_examples_per_kind=config.movie_examples_per_kind,
        )

    summary_rows, _series_records, unit_scores = run_information_step(
        config=config,
        dirs=dirs,
        model=model,
        population=population,
        device=device,
        examples=examples,
        image_by_index=image_by_index,
        crops=crop_table,
    )

    activation_movies: list[str] = []
    if config.make_activation_movies:
        activation_movies = make_activation_movies(
            model=model,
            population=population,
            device=device,
            examples=examples,
            image_by_index=image_by_index,
            crop_rows=crop_table,
            unit_scores=unit_scores,
            movie_dir=dirs["movies"],
            figure_dir=dirs["figures"],
            seed=config.seed,
            t_max=config.t_max,
            batch_size=config.batch_size,
            fps=config.movie_fps,
            max_examples_per_kind=config.movie_examples_per_kind,
        )

    summary = {
        "run_dir": str(run_dir),
        "n_trace_examples": len(examples),
        "n_images": len(image_by_index),
        "n_image_crops": len(crop_table),
        "n_biological_twins": int(n_biological_twins),
        "n_simulated_neurons": int(population.N),
        "n_summary_rows": len(summary_rows),
        "conditions": list(MAIN_CONDITIONS),
        "phase_conditions": list(PHASE_CONDITIONS),
        "sf_conditions": list(SF_CONDITIONS),
        "figures": sorted(str(path) for path in dirs["figures"].glob("*.pdf")),
        "image_selection_figures": image_figures,
        "stimulus_movies": stimulus_movies,
        "activation_movies": activation_movies,
        "movies": stimulus_movies + activation_movies,
        "summary_csv": str(dirs["metadata"] / "05_lagcube_information_summary.csv"),
        "series_npz": str(dirs["cache"] / "cumulative_information_series.npz"),
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None,
                        help="Optional readable folder name under outputs/twininfo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-indices", nargs="+", type=int, default=None,
                        help="Natural image indices. Omit to use every available natural image.")
    parser.add_argument("--n-crops-per-image", type=int, default=3)
    parser.add_argument("--n-examples-per-kind", type=int, default=10)
    parser.add_argument("--selected-trace-example-ids", nargs="+", default=())
    parser.add_argument("--t-max", type=int, default=128)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--population-selection", choices=POPULATION_SELECTION_MODES, default="top_performance",
                        help="How to choose simulated neurons.")
    parser.add_argument("--performance-metric", choices=PERFORMANCE_METRICS, default="ccnorm",
                        help="Fig. 3 metric used when --population-selection=top_performance.")
    parser.add_argument("--min-performance-score", type=float, default=None,
                        help="Optional minimum score for --performance-metric, e.g. ccnorm >= 0.5.")
    parser.add_argument("--population-grid-position-mode", choices=GRID_POSITION_MODES, default="full_grid",
                        help="How to assign one retinotopic grid position to each selected unit.")
    parser.add_argument("--population-grid-stride", type=int, default=1,
                        help="Stride through the model's existing spatial rate map when using full_grid.")
    parser.add_argument("--deduplicate-units", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop top-ranked biological twins whose battery responses are too correlated.")
    parser.add_argument("--dedupe-correlation-threshold", type=float, default=0.95)
    parser.add_argument("--dedupe-candidate-multiplier", type=float, default=5.0)
    parser.add_argument("--dedupe-battery-frames", type=int, default=96)
    parser.add_argument("--ccmax-threshold", type=float, default=DEFAULT_CCMAX_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fisher-step-arcmin", type=float, default=0.5)
    parser.add_argument("--shift-grid-max-arcmin", type=float, default=1.0)
    parser.add_argument("--shift-grid-step-arcmin", type=float, default=1.0)
    parser.add_argument("--shift-grid-mode", choices=("square", "cross"), default="square")
    parser.add_argument("--make-stimulus-movies", action="store_true")
    parser.add_argument("--make-activation-movies", action="store_true")
    parser.add_argument("--movie-examples-per-kind", type=int, default=1)
    parser.add_argument("--movie-fps", type=int, default=30)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(PipelineConfig(
        run_name=args.run_name,
        seed=args.seed,
        image_indices=None if args.image_indices is None else tuple(int(v) for v in args.image_indices),
        n_crops_per_image=args.n_crops_per_image,
        n_examples_per_kind=args.n_examples_per_kind,
        selected_trace_example_ids=tuple(str(v) for v in args.selected_trace_example_ids),
        t_max=args.t_max,
        stride=args.stride,
        population_size=args.population_size,
        population_selection=args.population_selection,
        performance_metric=args.performance_metric,
        min_performance_score=args.min_performance_score,
        population_grid_position_mode=args.population_grid_position_mode,
        population_grid_stride=args.population_grid_stride,
        deduplicate_units=bool(args.deduplicate_units),
        dedupe_correlation_threshold=args.dedupe_correlation_threshold,
        dedupe_candidate_multiplier=args.dedupe_candidate_multiplier,
        dedupe_battery_frames=args.dedupe_battery_frames,
        ccmax_threshold=args.ccmax_threshold,
        batch_size=args.batch_size,
        fisher_step_arcmin=args.fisher_step_arcmin,
        shift_grid_max_arcmin=args.shift_grid_max_arcmin,
        shift_grid_step_arcmin=args.shift_grid_step_arcmin,
        shift_grid_mode=args.shift_grid_mode,
        make_stimulus_movies=bool(args.make_stimulus_movies),
        make_activation_movies=bool(args.make_activation_movies),
        movie_examples_per_kind=args.movie_examples_per_kind,
        movie_fps=args.movie_fps,
        recompute=bool(args.recompute),
    ))
    print("Twininfo production pipeline complete")
    print(f"  run: {summary['run_dir']}")
    print(f"  summary: {summary['summary_csv']}")
    print(f"  figures: {len(summary['figures'])}")
    print(f"  movies: {len(summary['movies'])}")


if __name__ == "__main__":
    main()
