#!/usr/bin/env python3
"""Local-linear audit of motion-dependent sharpening through M77.

For matched natural-image and real-fixation pairs, this command compares the
actual measured-motion replay with the first-order tangent prediction around
the stabilized movie.  A single forward-mode JVP returns every visual stage
and the RR100 output, with behavior clamped to the identical all-zero vector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import render_movies
from paper.fig4.spatiotemporal_tuning.run_m77_retinal_causal_chain import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    DEFAULT_MCFARLAND,
    DEFAULT_POPULATION,
    DEFAULT_MATRIX,
    DEFAULT_CHAIN,
    lagged_movie_view,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.score_real_trace_matrix import RR100_VERSION


EPS = 1e-12
STAGES = (
    "signed temporal convolution",
    "GroupNorm/LRN/split-ReLU stem",
    "spatial stage 1",
    "spatial stage 2",
    "spatial stage 3",
    "RR100 output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_MATRIX / "image_feature_table.csv")
    parser.add_argument("--trace-bank", type=Path, default=DEFAULT_CHAIN / "fixation_bank")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-kind", choices=("filtered", "raw"), default="filtered")
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--n-traces", type=int, default=20)
    parser.add_argument("--n-timepoints", type=int, default=64)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument(
        "--fd-epsilon",
        type=float,
        default=0.10,
        help="Central finite-difference step as a fraction of the full stabilized-to-moving displacement.",
    )
    parser.add_argument(
        "--fd-check-epsilon",
        type=float,
        default=0.05,
        help="Second central step used only to audit local-derivative convergence.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--save-example-maps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def evenly_spaced_rows(length: int, requested: int) -> np.ndarray:
    if length < 1 or requested < 1:
        raise ValueError("bank and requested sample size must be positive")
    return np.unique(np.round(np.linspace(0, length - 1, min(length, requested))).astype(int))


def hierarchy_function(scorer: RealTraceMatrixScorer):
    core = scorer.model.model.convnet

    def forward(stimulus: torch.Tensor):
        pre = core.temporal_conv(stimulus)
        if pre.shape[2] != 1:
            raise RuntimeError("60-frame M77 histories must collapse time")
        pre = pre.squeeze(2)
        stem = core.temporal_nonlinearity(pre)
        stage1 = core.stage1_nonlinearity(core.stage1_conv(stem))
        stage2 = core.stage2_nonlinearity(core.stage2_conv(core._downsample(stage1)))
        stage3 = core.stage3_nonlinearity(core.stage3_conv(core._downsample(stage2)))
        output = scorer.apply_population_view(
            scorer._compute_rate_map(stimulus), scorer.population_view
        )
        return pre, stem, stage1, stage2, stage3, output

    return forward


def activation_metrics(value: torch.Tensor, *, signed: bool) -> dict[str, float]:
    """Summarize an activation movie without conflating channels as samples."""
    raw = value.float()
    positive = raw.square() if signed else raw.clamp_min(0.0)
    flat = positive.flatten(start_dim=2)
    mean = flat.mean(dim=2)
    gain = flat / mean[..., None].clamp_min(EPS)
    bits = (gain * gain.clamp_min(EPS).log2()).mean(dim=2)
    weighted_information = (bits * mean).sum() / mean.sum().clamp_min(EPS)
    spatial_std = flat.std(dim=2, unbiased=False)
    spatial_modulation = (spatial_std / mean.clamp_min(EPS)).mean()
    temporal_variance = positive.var(dim=0, unbiased=False).mean()
    clipped = torch.mean((raw < 0).float()) if not signed else torch.zeros((), device=raw.device)
    return {
        "mean_activation": float(positive.mean().detach().cpu()),
        "temporal_drive": float(temporal_variance.detach().cpu()),
        "spatial_modulation": float(spatial_modulation.detach().cpu()),
        "spatial_information": float(weighted_information.detach().cpu()),
        "negative_fraction_before_clip": float(clipped.detach().cpu()),
    }


def _positive_for_information(value: torch.Tensor, *, signed: bool) -> torch.Tensor:
    return value.float().square() if signed else value.float().clamp_min(0.0)


def score_pair(
    forward,
    stabilized: torch.Tensor,
    moving: torch.Tensor,
    *,
    batch_size: int,
    retain_maps: bool,
    fd_epsilon: float,
    fd_check_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    metric_names = tuple(activation_metrics(torch.ones(2, 1, 2, 2), signed=False))
    baseline_accumulator = {name: [] for name in metric_names}
    actual_accumulator = {name: [] for name in metric_names}
    tangent_accumulator = {name: [] for name in metric_names}
    baseline_moments: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(STAGES)
    actual_moments: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(STAGES)
    tangent_moments: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(STAGES)
    convergence_accumulator: list[list[tuple[float, int]]] = [[] for _ in STAGES]
    n_timepoints = 0
    examples: dict[str, np.ndarray] = {}
    for start in range(0, len(stabilized), int(batch_size)):
        baseline = stabilized[start : start + int(batch_size)]
        target = moving[start : start + int(batch_size)]
        delta = target - baseline
        with torch.no_grad():
            baseline_outputs = forward(baseline)
            actual_outputs = forward(target)
            plus = forward(baseline + float(fd_epsilon) * delta)
            minus = forward(baseline - float(fd_epsilon) * delta)
            check_plus = forward(baseline + float(fd_check_epsilon) * delta)
            check_minus = forward(baseline - float(fd_check_epsilon) * delta)
        directional = tuple(
            (high - low) / (2.0 * float(fd_epsilon)) for high, low in zip(plus, minus)
        )
        check_directional = tuple(
            (high - low) / (2.0 * float(fd_check_epsilon))
            for high, low in zip(check_plus, check_minus)
        )
        tangent_outputs = tuple(base + change for base, change in zip(baseline_outputs, directional))
        for stage, (baseline_output, actual, tangent) in enumerate(
            zip(baseline_outputs, actual_outputs, tangent_outputs)
        ):
            signed = stage == 0
            baseline_metric = activation_metrics(baseline_output, signed=signed)
            actual_metric = activation_metrics(actual, signed=signed)
            tangent_metric = activation_metrics(tangent, signed=signed)
            weight = len(actual)
            for name in metric_names:
                baseline_accumulator[name].append((baseline_metric[name], weight))
                actual_accumulator[name].append((actual_metric[name], weight))
                tangent_accumulator[name].append((tangent_metric[name], weight))
            for source, moments in (
                (_positive_for_information(baseline_output, signed=signed), baseline_moments),
                (_positive_for_information(actual, signed=signed), actual_moments),
                (_positive_for_information(tangent, signed=signed), tangent_moments),
            ):
                batch_sum = source.sum(dim=0).detach().float().cpu()
                batch_square = source.square().sum(dim=0).detach().float().cpu()
                previous = moments[stage]
                moments[stage] = (
                    batch_sum if previous is None else previous[0] + batch_sum,
                    batch_square if previous is None else previous[1] + batch_square,
                )
            difference = directional[stage] - check_directional[stage]
            relative = float(
                difference.square().mean().sqrt().div(
                    check_directional[stage].square().mean().sqrt().clamp_min(EPS)
                ).detach().cpu()
            )
            convergence_accumulator[stage].append((relative, weight))
            if retain_maps and start == 0:
                examples[f"baseline_stage_{stage}"] = baseline_output[0].detach().float().cpu().numpy()
                examples[f"actual_stage_{stage}"] = actual[0].detach().float().cpu().numpy()
                examples[f"tangent_stage_{stage}"] = tangent[0].detach().float().cpu().numpy()
        n_timepoints += len(actual_outputs[0])
    baseline_summary = np.zeros((len(STAGES), len(metric_names)), dtype=np.float64)
    actual_summary = np.zeros((len(STAGES), len(metric_names)), dtype=np.float64)
    tangent_summary = np.zeros_like(actual_summary)
    # Each metric accumulator was appended once per stage in stage-major order
    # within every temporal batch. Recover that ordering explicitly.
    n_batches = len(actual_accumulator[metric_names[0]]) // len(STAGES)
    for metric_index, name in enumerate(metric_names):
        for stage in range(len(STAGES)):
            baseline_values = [baseline_accumulator[name][batch * len(STAGES) + stage] for batch in range(n_batches)]
            actual_values = [actual_accumulator[name][batch * len(STAGES) + stage] for batch in range(n_batches)]
            tangent_values = [tangent_accumulator[name][batch * len(STAGES) + stage] for batch in range(n_batches)]
            baseline_summary[stage, metric_index] = np.average(
                [item[0] for item in baseline_values], weights=[item[1] for item in baseline_values]
            )
            actual_summary[stage, metric_index] = np.average(
                [item[0] for item in actual_values], weights=[item[1] for item in actual_values]
            )
            tangent_summary[stage, metric_index] = np.average(
                [item[0] for item in tangent_values], weights=[item[1] for item in tangent_values]
            )
    temporal_index = metric_names.index("temporal_drive")
    for stage in range(len(STAGES)):
        for summary, moments in (
            (baseline_summary, baseline_moments),
            (actual_summary, actual_moments),
            (tangent_summary, tangent_moments),
        ):
            if moments[stage] is None:
                raise RuntimeError("activation temporal moments were not accumulated")
            total, square = moments[stage]
            variance = square / n_timepoints - (total / n_timepoints).square()
            summary[stage, temporal_index] = float(variance.clamp_min(0).mean())
    convergence = np.asarray(
        [
            np.average([item[0] for item in values], weights=[item[1] for item in values])
            for values in convergence_accumulator
        ],
        dtype=np.float32,
    )
    return baseline_summary, actual_summary, tangent_summary, {
        "metric_names": np.asarray(metric_names),
        "fd_relative_rms_by_stage": convergence,
        **examples,
    }


def render_summary(
    path: Path,
    baseline: np.ndarray,
    actual: np.ndarray,
    tangent: np.ndarray,
    metric_names: np.ndarray,
) -> None:
    metric_index = {str(name): index for index, name in enumerate(metric_names)}
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    x = np.arange(len(STAGES))
    for axis, metric, title, ylabel in (
        (axes[0], "mean_activation", "A  Motion-driven activation", "mean activation (arbitrary units)"),
        (axes[1], "spatial_modulation", "B  Spatial modulation", "spatial coefficient of variation"),
        (axes[2], "spatial_information", "C  Nonlinear spatial sharpening", "spatial information (bits/event)"),
    ):
        index = metric_index[metric]
        actual_center = np.nanmedian(actual[..., index] - baseline[..., index], axis=(0, 1))
        tangent_center = np.nanmedian(tangent[..., index] - baseline[..., index], axis=(0, 1))
        axis.plot(x, actual_center, "o-", lw=2.0, color="#C94C4C", label="full nonlinear replay")
        axis.plot(x, tangent_center, "o--", lw=1.8, color="#3178B5", label="local tangent replay")
        axis.set_xticks(x, ["signed\nstem", "GN/LRN/\nsplit-ReLU", "spatial\n1", "spatial\n2", "spatial\n3", "RR100\noutput"])
        axis.set_ylabel(f"motion − stabilized {ylabel}")
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    figure.suptitle("M77 nonlinear sharpening audit · measured motion versus first-order response at stabilization", fontsize=14, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def crossed_bootstrap_median(
    value: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError("crossed bootstrap expects [image,trace]")
    draws = np.empty(int(n_bootstrap), dtype=float)
    for draw in range(int(n_bootstrap)):
        images = rng.integers(0, array.shape[0], array.shape[0])
        traces = rng.integers(0, array.shape[1], array.shape[1])
        draws[draw] = np.nanmedian(array[np.ix_(images, traces)])
    return float(np.nanmedian(array)), *map(float, np.quantile(draws, (0.025, 0.975)))


def render_example_maps(path: Path, examples: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(
        len(STAGES), 4, figsize=(12.2, 2.55 * len(STAGES)), constrained_layout=True
    )
    for stage, stage_name in enumerate(STAGES):
        maps = []
        for condition in ("baseline", "actual", "tangent"):
            value = np.asarray(examples[f"{condition}_stage_{stage}"], dtype=float)
            value = np.square(value).mean(axis=0) if stage == 0 else np.clip(value, 0, None).mean(axis=0)
            maps.append(value)
        difference = maps[1] - maps[2]
        low = min(float(np.min(item)) for item in maps)
        high = max(float(np.max(item)) for item in maps)
        sequential = None
        for column, (value, title) in enumerate(
            zip(maps, ("stabilized", "full moving", "local tangent"))
        ):
            sequential = axes[stage, column].imshow(value, cmap="viridis", vmin=low, vmax=high)
            axes[stage, column].set_title(title if stage == 0 else "", fontsize=9)
            axes[stage, column].set_xticks([])
            axes[stage, column].set_yticks([])
        limit = max(float(np.max(np.abs(difference))), EPS)
        diverging = axes[stage, 3].imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[stage, 3].set_title("full − tangent" if stage == 0 else "", fontsize=9)
        axes[stage, 3].set_xticks([])
        axes[stage, 3].set_yticks([])
        axes[stage, 0].set_ylabel(stage_name, fontsize=9)
        if sequential is not None:
            figure.colorbar(sequential, ax=axes[stage, :3], shrink=0.68, pad=0.006)
        figure.colorbar(diverging, ax=axes[stage, 3], shrink=0.68, pad=0.015)
    figure.suptitle(
        "Representative M77 activation maps · every color scale is explicit",
        fontsize=14,
        fontweight="semibold",
    )
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.motion_scale <= 0:
        raise ValueError("motion-scale must be positive")
    if not 0 < args.fd_check_epsilon < args.fd_epsilon <= 0.25:
        raise ValueError("require 0 < fd-check-epsilon < fd-epsilon <= 0.25")
    images = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    trace_table = pd.read_csv(args.trace_bank / "trace_table.csv").sort_values("trace_index").reset_index(drop=True)
    traces = np.load(args.trace_bank / f"trace_xy_{args.trace_kind}.npy", mmap_mode="r")
    image_rows = evenly_spaced_rows(len(images), args.n_images)
    trace_rows = evenly_spaced_rows(len(trace_table), args.n_traces)
    endpoints = np.unique(np.round(np.linspace(1, 240, min(args.n_timepoints, 240))).astype(int))
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_config.resolve(),
        population_spec_dir=args.population_spec_dir.resolve(),
        rr100_version=RR100_VERSION,
        device=args.device,
        strict=True,
        mcfarland_outputs=args.mcfarland_outputs.resolve(),
    )
    scorer.model.model.eval()
    scorer.readout.eval()
    forward = hierarchy_function(scorer)
    metric_names = None
    actual = None
    baseline = None
    tangent = None
    fd_convergence = None
    examples: dict[str, np.ndarray] = {}
    canvas_cache: dict = {}
    for image_position, image_row in enumerate(image_rows):
        patch, _ = extract_patch(images.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=args.patch_size_px)
        for trace_position, trace_row in enumerate(trace_rows):
            trace = np.asarray(traces[int(trace_row)], dtype=np.float32)
            movies = render_movies(
                patch,
                np.stack((np.zeros_like(trace), trace * float(args.motion_scale))),
                device=args.device,
            )
            stabilized_history = lagged_movie_view(
                torch.from_numpy(movies[0]).to(args.device), 60
            )[endpoints]
            moving_history = lagged_movie_view(
                torch.from_numpy(movies[1]).to(args.device), 60
            )[endpoints]
            stabilized_history = (stabilized_history - 127.0) / 255.0
            moving_history = (moving_history - 127.0) / 255.0
            pair_baseline, pair_actual, pair_tangent, pair_examples = score_pair(
                forward,
                stabilized_history,
                moving_history,
                batch_size=args.batch_size,
                retain_maps=bool(args.save_example_maps and image_position == 0 and trace_position == 0),
                fd_epsilon=float(args.fd_epsilon),
                fd_check_epsilon=float(args.fd_check_epsilon),
            )
            pair_convergence = pair_examples.pop("fd_relative_rms_by_stage")
            if actual is None:
                metric_names = pair_examples.pop("metric_names")
                shape = (len(image_rows), len(trace_rows), len(STAGES), len(metric_names))
                baseline = np.zeros(shape, dtype=np.float32)
                actual = np.zeros(shape, dtype=np.float32)
                tangent = np.zeros_like(actual)
                fd_convergence = np.zeros((len(image_rows), len(trace_rows), len(STAGES)), dtype=np.float32)
            baseline[image_position, trace_position] = pair_baseline
            actual[image_position, trace_position] = pair_actual
            tangent[image_position, trace_position] = pair_tangent
            fd_convergence[image_position, trace_position] = pair_convergence
            if pair_examples and not examples:
                examples = pair_examples
        print(f"nonlinear tangent image {image_position + 1}/{len(image_rows)}", flush=True)
    if baseline is None or actual is None or tangent is None or metric_names is None or fd_convergence is None:
        raise RuntimeError("no nonlinear tangent pairs were scored")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "nonlinear_sharpening.npz"
    np.savez_compressed(
        archive_path,
        baseline=baseline,
        actual=actual,
        tangent=tangent,
        actual_motion_effect=actual - baseline,
        tangent_motion_effect=tangent - baseline,
        difference=actual - tangent,
        metric_names=metric_names,
        stage_names=np.asarray(STAGES),
        image_rows=image_rows,
        trace_rows=trace_rows,
        endpoint_indices=endpoints,
        motion_scale=np.asarray(args.motion_scale),
        fd_relative_rms_by_stage=fd_convergence,
        **examples,
    )
    figure_path = args.out_dir / "m77_nonlinear_sharpening.png"
    render_summary(figure_path, baseline, actual, tangent, metric_names)
    example_figure = None
    if examples:
        example_figure = args.out_dir / "m77_nonlinear_sharpening_example_maps.png"
        render_example_maps(example_figure, examples)
    info_index = list(metric_names).index("spatial_information")
    actual_effect = actual[..., info_index] - baseline[..., info_index]
    tangent_effect = tangent[..., info_index] - baseline[..., info_index]
    median_gap = np.nanmedian(actual_effect - tangent_effect, axis=(0, 1))
    rng = np.random.default_rng(args.seed)
    inference_rows = []
    for stage_index, stage in enumerate(STAGES):
        for contrast, value in (
            ("full_motion_minus_stabilized", actual_effect[..., stage_index]),
            ("full_minus_tangent_motion_effect", actual_effect[..., stage_index] - tangent_effect[..., stage_index]),
        ):
            center, low, high = crossed_bootstrap_median(
                value, n_bootstrap=args.n_bootstrap, rng=rng
            )
            inference_rows.append(
                {
                    "stage": stage,
                    "contrast": contrast,
                    "median": center,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(args.out_dir / "nonlinear_sharpening_inference.csv", index=False)
    output_full = inference.loc[
        inference.stage.eq("RR100 output")
        & inference.contrast.eq("full_motion_minus_stabilized")
    ].iloc[0]
    output_gap = inference.loc[
        inference.stage.eq("RR100 output")
        & inference.contrast.eq("full_minus_tangent_motion_effect")
    ].iloc[0]
    summary = {
        "analysis": "M77 layerwise full nonlinear versus local tangent replay",
        "behavior": "identical all-zero 42-dimensional vector",
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_image_trace_pairs": int(len(image_rows) * len(trace_rows)),
        "n_scored_timepoints_per_pair": int(len(endpoints)),
        "motion_scale": float(args.motion_scale),
        "linearization_point": "matched stabilized movie for the same natural image",
        "tangent_estimator": "symmetric central finite directional derivative; robust to the undefined JVP of Lp pooling on exact zero split-ReLU patches",
        "fd_epsilon": float(args.fd_epsilon),
        "fd_check_epsilon": float(args.fd_check_epsilon),
        "median_fd_relative_rms_difference": {
            stage: float(value)
            for stage, value in zip(STAGES, np.nanmedian(fd_convergence, axis=(0, 1)))
        },
        "signed_stem_information_definition": "spatial information of squared signed temporal-convolution drive",
        "tangent_negative_values": "clipped to zero only for nonnegative-stage information metrics; negative fraction saved as a metric",
        "median_actual_minus_tangent_spatial_information": {
            stage: float(value) for stage, value in zip(STAGES, median_gap)
        },
        "claim_gates": {
            "full_replay_increases_rr100_spatial_information": bool(float(output_full.ci_low) > 0),
            "tangent_underestimates_rr100_spatial_sharpening": bool(float(output_gap.ci_low) > 0),
        },
        "inference": "crossed image and trace bootstrap of the median",
        "archive": str(archive_path.resolve()),
        "figure": str(figure_path.resolve()),
        "example_map_figure": str(example_figure.resolve()) if example_figure else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
