#!/usr/bin/env python3
"""Run exact causal phase/nonlinearity tests for the Figure 4 mechanism.

The experiment uses the validated corrected-history subset (8 images, 24
drift-only trajectories, five movement scales).  It compares the intact twin
to a first-order stabilized-tangent twin and performs a 2 x 2 intervention at
the first SplitReLU, independently selecting the local polarity route and
response magnitude from the stabilized or moving movie.  A switch-count-
matched shuffled-route control tests whether organized phase structure, rather
than the number of polarity transitions alone, matters downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    LEGACY_MATRIX_DIR,
    N_PRECEDING,
    write_json,
)
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    DirectPopulationReadout,
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.nonlinear_phase_causal.common import (
    CONDITIONS,
    CONDITION_DEFINITIONS,
    EXAMPLE_IMAGE,
    EXAMPLE_OUTPUT,
    EXAMPLE_TRACE,
    N_SCORED,
    N_TRACES,
    OUT,
    SCALES,
    json_ready,
    matched_switch_shuffled_route,
    sha256,
    splitrelu_from_route_magnitude,
    splitrelu_from_route_mask,
    unit_rate_metrics,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


PHASE_SOURCE = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
SOURCE = ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
BASE_SEED = 20260813


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument(
        "--image-ids",
        default="",
        help="optional comma-separated selected image IDs (for independent workers)",
    )
    parser.add_argument("--scales", default=",".join(f"{value:g}" for value in SCALES))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tangent", action="store_true")
    parser.add_argument(
        "--parts-only",
        action="store_true",
        help="write resumable per-image parts without merging or replacing the run manifest",
    )
    return parser.parse_args()


def parse_scales(value: str) -> np.ndarray:
    scales = np.asarray([float(item.strip()) for item in value.split(",") if item.strip()], dtype=np.float32)
    if len(scales) == 0 or np.any(scales < 0) or len(np.unique(scales)) != len(scales):
        raise ValueError(value)
    return scales


def historical_groups() -> dict[str, np.ndarray]:
    table = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    sf = pd.to_numeric(table.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"low": np.flatnonzero(sf < 0.5), "high": np.flatnonzero(sf >= 0.5)}
    if {key: len(value) for key, value in groups.items()} != {"low": 71, "high": 29}:
        raise ValueError({key: len(value) for key, value in groups.items()})
    return groups


class StemCapture(AbstractContextManager["StemCapture"]):
    """Capture the signed tensor entering the first SplitReLU and its output."""

    def __init__(self, module: torch.nn.Module):
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.inputs.append(args[0].detach().clone())
        self.outputs.append(output.detach().clone())

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.handle.remove()


class ReplaceOutput(AbstractContextManager["ReplaceOutput"]):
    """Temporarily replace one module's output with an exact cached tensor."""

    def __init__(self, module: torch.nn.Module, replacement: torch.Tensor):
        self.replacement = replacement
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if output.shape != self.replacement.shape:
            raise ValueError((tuple(output.shape), tuple(self.replacement.shape)))
        return self.replacement.to(device=output.device, dtype=output.dtype)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.handle.remove()


def preactivation(
    model: Any,
    readout: DirectPopulationReadout,
    x: torch.Tensor,
    behavior: torch.Tensor | None,
) -> torch.Tensor:
    core = model.core_forward(x, behavior)
    return readout(core[:, :, -1])


def stabilized_tangent_preactivation(
    function: Callable[[torch.Tensor], torch.Tensor],
    stable_x: torch.Tensor,
    moving_x: torch.Tensor,
    *,
    mode: str = "reverse",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``z(x0)`` and ``z(x0) + J_z(x0)(x1-x0)`` exactly via a JVP."""

    x0 = stable_x.detach()
    direction = (moving_x - stable_x).detach()
    if mode == "forward":
        anchor, directional = torch.func.jvp(function, (x0,), (direction,))
    elif mode == "reverse":
        anchor, directional = torch.autograd.functional.jvp(
            function,
            x0.requires_grad_(True),
            direction,
            create_graph=False,
            strict=False,
        )
    else:
        raise ValueError(mode)
    return anchor.detach(), (anchor + directional).detach()


def _append_metrics(
    accumulators: dict[str, dict[str, np.ndarray]],
    condition: str,
    rate_map: torch.Tensor,
) -> None:
    metrics = unit_rate_metrics(rate_map)
    expected = metrics["expected_spikes"].detach().cpu().numpy()
    accumulators[condition]["expected"] += expected.sum(axis=0)
    accumulators[condition]["mean_rate"] += metrics["mean_rate_hz"].detach().cpu().numpy().sum(axis=0)
    for metric in ("ssi", "cv2", "quadratic_bits"):
        value = metrics[metric].detach().cpu().numpy()
        accumulators[condition][metric] += (value * expected).sum(axis=0)


def _finish_metrics(accumulators: dict[str, dict[str, np.ndarray]], n_frames: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for metric in ("ssi", "cv2", "quadratic_bits"):
        result[metric] = np.stack(
            [
                accumulators[condition][metric]
                / np.maximum(accumulators[condition]["expected"], 1e-12)
                for condition in CONDITIONS
            ]
        ).astype(np.float32)
    result["expected_spikes"] = np.stack(
        [accumulators[condition]["expected"] for condition in CONDITIONS]
    ).astype(np.float32)
    result["mean_rate_hz"] = np.stack(
        [accumulators[condition]["mean_rate"] / n_frames for condition in CONDITIONS]
    ).astype(np.float32)
    return result


def _new_accumulators(n_units: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        condition: {
            key: np.zeros(n_units, dtype=np.float64)
            for key in ("ssi", "cv2", "quadratic_bits", "expected", "mean_rate")
        }
        for condition in CONDITIONS
    }


def _population_maps(rate_map: torch.Tensor, groups: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            rate_map[:, groups["low"]].mean(dim=1).detach().cpu().numpy(),
            rate_map[:, groups["high"]].mean(dim=1).detach().cpu().numpy(),
        ],
        axis=1,
    )


def score_trace_scale(
    *,
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    patch: np.ndarray,
    base_history: np.ndarray,
    scale: float,
    frame_batch_size: int,
    groups: dict[str, np.ndarray],
    seed: int,
    capture_example: bool,
    skip_tangent: bool,
) -> dict[str, np.ndarray]:
    """Score one matched stabilized/moving movie pair over all 40 outputs."""

    model = scorer.model.model
    stem_act = model.convnet.stem.components["act"]
    positive_gain = stem_act.ongain.detach()
    dtype = next(model.parameters()).dtype
    image = _standardize_uint_like(patch)
    pair_histories = scaled_histories(base_history[None], np.asarray([0.0, scale], dtype=np.float32))
    stims = (make_corrected_causal_stims(image, pair_histories, torch=scorer.torch) - 127.0) / 255.0
    stable_stims = stims[:N_SCORED]
    moving_stims = stims[N_SCORED:]
    accumulators = _new_accumulators(readout.n_units)
    switch_count = np.zeros(8, dtype=np.float64)
    switch_total = np.zeros(8, dtype=np.float64)
    magnitude_difference_sq = np.zeros(8, dtype=np.float64)
    stable_magnitude_sq = np.zeros(8, dtype=np.float64)
    diagnostics = {
        "splitrelu_self_max_abs_error": 0.0,
        "stable_preactivation_replay_max_abs_error": 0.0,
        "tangent_anchor_max_abs_error": 0.0,
        "shuffled_switch_count_max_abs_error": 0.0,
    }
    example_payload: dict[str, np.ndarray] = {}

    for frame_start in range(0, N_SCORED, frame_batch_size):
        frame_stop = min(frame_start + frame_batch_size, N_SCORED)
        x0 = stable_stims[frame_start:frame_stop].to(scorer.device)
        x1 = moving_stims[frame_start:frame_stop].to(scorer.device)
        batch_size = len(x0)
        behavior = scorer._zero_behavior(len(x0), dtype)

        paired_x = torch.cat((x0, x1), dim=0)
        paired_behavior = scorer._zero_behavior(len(paired_x), dtype)
        with torch.no_grad(), StemCapture(stem_act) as paired_capture:
            paired_z = preactivation(model, readout, paired_x, paired_behavior)
        if len(paired_capture.inputs) != 1:
            raise RuntimeError(len(paired_capture.inputs))
        z0, z1 = paired_z.split(batch_size, dim=0)
        a0, a1 = paired_capture.inputs[0].split(batch_size, dim=0)
        split0, split1 = paired_capture.outputs[0].split(batch_size, dim=0)
        expected_stable_act = splitrelu_from_route_magnitude(a0, a0, positive_gain=positive_gain)
        expected_moving_act = splitrelu_from_route_magnitude(a1, a1, positive_gain=positive_gain)
        diagnostics["splitrelu_self_max_abs_error"] = max(
            diagnostics["splitrelu_self_max_abs_error"],
            float((expected_stable_act - split0).abs().max().cpu()),
            float((expected_moving_act - split1).abs().max().cpu()),
        )

        rate_maps: dict[str, torch.Tensor] = {
            "stable": model.activation(z0),
            "full": model.activation(z1),
        }
        if skip_tangent:
            rate_maps["tangent"] = torch.full_like(rate_maps["stable"], torch.nan)
        elif np.isclose(scale, 0.0):
            # The direction is identically zero, so the JVP is exactly zero.
            rate_maps["tangent"] = rate_maps["stable"]
        else:
            tangent_anchor, tangent_z = stabilized_tangent_preactivation(
                lambda value: preactivation(model, readout, value, behavior), x0, x1
            )
            diagnostics["tangent_anchor_max_abs_error"] = max(
                diagnostics["tangent_anchor_max_abs_error"],
                float((tangent_anchor - z0).abs().max().cpu()),
            )
            rate_maps["tangent"] = model.activation(tangent_z)

        replacements = {
            "magnitude_only": splitrelu_from_route_magnitude(a0, a1, positive_gain=positive_gain),
            "route_only": splitrelu_from_route_magnitude(a1, a0, positive_gain=positive_gain),
        }
        # Seed each scored output independently so the null is invariant to
        # frame batching and worker scheduling.
        shuffled_route = torch.cat(
            [
                matched_switch_shuffled_route(
                    a0[sample : sample + 1],
                    a1[sample : sample + 1],
                    seed=seed + frame_start + sample,
                )
                for sample in range(batch_size)
            ],
            dim=0,
        )
        replacements["shuffled_route"] = splitrelu_from_route_mask(
            shuffled_route, a1, positive_gain=positive_gain
        )
        true_switch = torch.logical_xor(a0 > 0, a1 > 0).flatten(start_dim=2).sum(dim=2)
        shuffled_switch = torch.logical_xor(a0 > 0, shuffled_route).flatten(start_dim=2).sum(dim=2)
        diagnostics["shuffled_switch_count_max_abs_error"] = max(
            diagnostics["shuffled_switch_count_max_abs_error"],
            float((true_switch - shuffled_switch).abs().max().cpu()),
        )
        if np.isclose(scale, 0.0):
            for condition in replacements:
                rate_maps[condition] = rate_maps["full"]
        else:
            hybrid_conditions = tuple(replacements)
            hybrid_replacement = torch.cat([replacements[name] for name in hybrid_conditions], dim=0)
            hybrid_x = torch.cat([x1] * len(hybrid_conditions), dim=0)
            hybrid_behavior = scorer._zero_behavior(len(hybrid_x), dtype)
            with torch.no_grad(), ReplaceOutput(stem_act, hybrid_replacement):
                hybrid_z = preactivation(model, readout, hybrid_x, hybrid_behavior)
            for condition, z in zip(
                hybrid_conditions, hybrid_z.split(batch_size, dim=0), strict=True
            ):
                rate_maps[condition] = model.activation(z)

        if np.isclose(scale, 0.0):
            diagnostics["stable_preactivation_replay_max_abs_error"] = max(
                diagnostics["stable_preactivation_replay_max_abs_error"],
                float((z0 - z1).abs().max().cpu()),
            )
        for condition in CONDITIONS:
            _append_metrics(accumulators, condition, rate_maps[condition])

        switch = torch.logical_xor(a0 > 0, a1 > 0)
        reduce_dims = tuple(dim for dim in range(switch.ndim) if dim != 1)
        switch_count += switch.sum(dim=reduce_dims).double().cpu().numpy()
        switch_total += np.prod([switch.shape[dim] for dim in reduce_dims])
        magnitude_difference_sq += (
            (a1.abs() - a0.abs()).square().sum(dim=reduce_dims).double().cpu().numpy()
        )
        stable_magnitude_sq += a0.square().sum(dim=reduce_dims).double().cpu().numpy()

        if capture_example and frame_start <= EXAMPLE_OUTPUT < frame_stop:
            local = EXAMPLE_OUTPUT - frame_start
            example_payload = {
                "rate_maps": np.stack(
                    [_population_maps(rate_maps[condition][local : local + 1], groups)[0] for condition in CONDITIONS]
                ).astype(np.float32),
                "stable_stem_signed": a0[local, :, -1].detach().cpu().numpy().astype(np.float32),
                "moving_stem_signed": a1[local, :, -1].detach().cpu().numpy().astype(np.float32),
                "shuffled_route": shuffled_route[local, :, -1].detach().cpu().numpy().astype(np.uint8),
                "retinal_last_frame": x1[local, 0, -1].detach().cpu().numpy().astype(np.float32),
                "trajectory_xy": pair_histories[1, N_PRECEDING:].astype(np.float32),
            }

        del a0, a1, x0, x1, rate_maps, replacements
        if str(scorer.device).startswith("cuda"):
            torch.cuda.empty_cache()

    result = _finish_metrics(accumulators, N_SCORED)
    result.update(
        {
            "switch_fraction_by_stem_channel": (switch_count / np.maximum(switch_total, 1)).astype(np.float32),
            "magnitude_rms_change_by_stem_channel": np.sqrt(
                magnitude_difference_sq / np.maximum(stable_magnitude_sq, 1e-12)
            ).astype(np.float32),
            **{key: np.asarray(value, dtype=np.float64) for key, value in diagnostics.items()},
        }
    )
    for key, value in example_payload.items():
        result[f"example_{key}"] = value
    return result


def _selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    images = pd.read_csv(PHASE_SOURCE / "selection/selected_images.csv")
    traces = pd.read_csv(PHASE_SOURCE / "selection/selected_traces.csv")
    if len(images) != 8 or len(traces) != N_TRACES:
        raise ValueError((len(images), len(traces)))
    return images, traces


def _part_signature(
    *, image_id: int, trace_ids: np.ndarray, scales: np.ndarray, checkpoint_hash: str, skip_tangent: bool
) -> str:
    payload = json.dumps(
        {
            "analysis": "fig4_nonlinear_phase_causal_v1",
            "image_id": int(image_id),
            "trace_ids": trace_ids.tolist(),
            "scales": scales.tolist(),
            "checkpoint_sha256": checkpoint_hash,
            "skip_tangent": bool(skip_tangent),
        },
        sort_keys=True,
    )
    return __import__("hashlib").sha256(payload.encode()).hexdigest()


def score_image_part(
    *,
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    patch: np.ndarray,
    image_id: int,
    trace_ids: np.ndarray,
    base_histories: np.ndarray,
    scales: np.ndarray,
    groups: dict[str, np.ndarray],
    frame_batch_size: int,
    skip_tangent: bool,
) -> dict[str, np.ndarray]:
    shape = (len(trace_ids), len(scales), len(CONDITIONS), readout.n_units)
    arrays = {
        metric: np.full(shape, np.nan, dtype=np.float32)
        for metric in ("ssi", "cv2", "quadratic_bits", "expected_spikes", "mean_rate_hz")
    }
    arrays["switch_fraction_by_stem_channel"] = np.full(
        (len(trace_ids), len(scales), 8), np.nan, dtype=np.float32
    )
    arrays["magnitude_rms_change_by_stem_channel"] = np.full_like(
        arrays["switch_fraction_by_stem_channel"], np.nan
    )
    diagnostic_names = (
        "splitrelu_self_max_abs_error",
        "stable_preactivation_replay_max_abs_error",
        "tangent_anchor_max_abs_error",
        "shuffled_switch_count_max_abs_error",
    )
    for name in diagnostic_names:
        arrays[name] = np.full((len(trace_ids), len(scales)), np.nan, dtype=np.float64)
    example_by_scale: dict[int, dict[str, np.ndarray]] = {}

    for trace_ordinal, (trace_id, history) in enumerate(zip(trace_ids, base_histories, strict=True)):
        for scale_ordinal, scale in enumerate(scales):
            capture_example = int(image_id) == EXAMPLE_IMAGE and int(trace_id) == EXAMPLE_TRACE
            result = score_trace_scale(
                scorer=scorer,
                readout=readout,
                patch=patch,
                base_history=history,
                scale=float(scale),
                frame_batch_size=frame_batch_size,
                groups=groups,
                seed=BASE_SEED + int(image_id) * 100_003 + int(trace_id) * 101 + scale_ordinal,
                capture_example=capture_example,
                skip_tangent=skip_tangent,
            )
            for key in arrays:
                if key in result:
                    arrays[key][trace_ordinal, scale_ordinal] = result[key]
            if capture_example:
                example_by_scale[scale_ordinal] = {
                    key.removeprefix("example_"): value
                    for key, value in result.items()
                    if key.startswith("example_")
                }
            print(
                f"image={image_id} trace {trace_ordinal + 1}/{len(trace_ids)} id={trace_id} "
                f"scale={float(scale):g}",
                flush=True,
            )
    arrays.update(
        {
            "image_index": np.asarray(image_id, dtype=np.int64),
            "trace_index": trace_ids.astype(np.int64),
            "scales": scales.astype(np.float32),
            "conditions": np.asarray(CONDITIONS),
        }
    )
    if len(example_by_scale) == len(scales):
        for key in next(iter(example_by_scale.values())):
            arrays[f"example_{key}"] = np.stack(
                [example_by_scale[index][key] for index in range(len(scales))]
            )
    return arrays


def merge_parts(
    part_paths: list[Path],
    *,
    output_path: Path,
    groups: dict[str, np.ndarray],
) -> None:
    payloads = []
    for path in part_paths:
        with np.load(path) as archive:
            payloads.append({key: np.asarray(archive[key]) for key in archive.files})
    stack_keys = (
        "ssi",
        "cv2",
        "quadratic_bits",
        "expected_spikes",
        "mean_rate_hz",
        "switch_fraction_by_stem_channel",
        "magnitude_rms_change_by_stem_channel",
        "splitrelu_self_max_abs_error",
        "stable_preactivation_replay_max_abs_error",
        "tangent_anchor_max_abs_error",
        "shuffled_switch_count_max_abs_error",
    )
    merged = {key: np.stack([payload[key] for payload in payloads]) for key in stack_keys}
    merged.update(
        {
            "selected_image_index": np.asarray([int(payload["image_index"]) for payload in payloads]),
            "selected_trace_index": payloads[0]["trace_index"],
            "scales": payloads[0]["scales"],
            "conditions": payloads[0]["conditions"],
            "low_unit_indices": groups["low"],
            "high_unit_indices": groups["high"],
        }
    )
    example = [payload for payload in payloads if "example_rate_maps" in payload]
    if len(example) == 1:
        for key, value in example[0].items():
            if key.startswith("example_"):
                merged[key] = value
        merged["example_image_index"] = np.asarray(EXAMPLE_IMAGE)
        merged["example_trace_index"] = np.asarray(EXAMPLE_TRACE)
        merged["example_output_index"] = np.asarray(EXAMPLE_OUTPUT)
    np.savez_compressed(output_path, **merged)


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "exact_arrays"
    parts_dir = raw_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    selected_images, selected_traces = _selection()
    if args.image_ids:
        requested = np.asarray([int(item) for item in args.image_ids.split(",") if item.strip()])
        missing = sorted(set(requested.tolist()) - set(selected_images.image_index.astype(int).tolist()))
        if missing:
            raise ValueError(f"image IDs are not in the predeclared selection: {missing}")
        selected_images = selected_images.set_index("image_index").loc[requested].reset_index()
    if args.max_images:
        selected_images = selected_images.iloc[: args.max_images].copy()
    if args.max_traces:
        selected_traces = selected_traces.iloc[: args.max_traces].copy()
    image_ids = selected_images.image_index.to_numpy(int)
    trace_ids = selected_traces.trace_bank_index.to_numpy(int)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        base_histories = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=str(args.device),
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    scorer.model.model.eval()
    readout = build_direct_readout(scorer).eval()
    groups = historical_groups()
    checkpoint_hash = sha256(MODEL_CHECKPOINT_PATH)
    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    canvas_cache: dict[Any, Any] = {}
    part_paths: list[Path] = []
    for image_ordinal, image_id in enumerate(image_ids):
        part_path = parts_dir / f"image_{image_id:03d}.npz"
        signature = _part_signature(
            image_id=int(image_id),
            trace_ids=trace_ids,
            scales=scales,
            checkpoint_hash=checkpoint_hash,
            skip_tangent=bool(args.skip_tangent),
        )
        if part_path.is_file() and not args.overwrite:
            with np.load(part_path) as archive:
                saved_signature = str(archive["part_signature"].item())
            if saved_signature != signature:
                raise RuntimeError(f"Existing part has a different signature: {part_path}")
            print(f"using completed part {part_path}", flush=True)
            part_paths.append(part_path)
            continue
        patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
        payload = score_image_part(
            scorer=scorer,
            readout=readout,
            patch=patch,
            image_id=int(image_id),
            trace_ids=trace_ids,
            base_histories=base_histories,
            scales=scales,
            groups=groups,
            frame_batch_size=int(args.frame_batch_size),
            skip_tangent=bool(args.skip_tangent),
        )
        payload["part_signature"] = np.asarray(signature)
        np.savez_compressed(part_path, **payload)
        part_paths.append(part_path)
        print(f"completed image {image_ordinal + 1}/{len(image_ids)} id={image_id}", flush=True)

    if args.parts_only:
        print(f"completed {len(part_paths)} resumable image part(s)", flush=True)
        return 0

    is_canonical = (
        len(image_ids) == 8
        and len(trace_ids) == N_TRACES
        and np.array_equal(scales, SCALES)
        and not args.skip_tangent
    )
    output_path = raw_dir / (
        "exact_nonlinear_phase_metrics.npz" if is_canonical else "exact_nonlinear_phase_metrics_partial.npz"
    )
    merge_parts(part_paths, output_path=output_path, groups=groups)
    manifest = {
        "analysis": "fig4_nonlinear_phase_causal_v1",
        "canonical_complete": is_canonical,
        "n_images": len(image_ids),
        "n_traces": len(trace_ids),
        "n_scales": len(scales),
        "scales": scales,
        "conditions": CONDITION_DEFINITIONS,
        "tangent_definition": (
            "z_tangent = z(x_stable) + J_z(x_stable)(x_moving-x_stable), where z is the "
            "complete twin output before its final softplus; the usual softplus and exact 51x51 SSI follow"
        ),
        "jvp_implementation": "torch.autograd.functional.jvp (reverse-over-reverse automatic differentiation)",
        "factorial_definition": (
            "At the first post-normalization SplitReLU, independently source the binary positive/negative "
            "branch route from stable or moving signed activation and source magnitude from stable or moving"
        ),
        "shuffled_control": (
            "shuffle the moving-vs-stable route-switch mask over internal-time and space independently "
            "within sample and stem channel, preserving exact switch counts"
        ),
        "selection": "the predeclared validated 8-image x 24 drift-only corrected-history subset",
        "selected_image_index": image_ids,
        "selected_trace_index": trace_ids,
        "frame_batch_size": int(args.frame_batch_size),
        "checkpoint": MODEL_CHECKPOINT_PATH,
        "checkpoint_sha256": checkpoint_hash,
        "source_files_sha256": {
            "run_experiment.py": sha256(Path(__file__)),
            "common.py": sha256(Path(__file__).with_name("common.py")),
            "analyze_and_plot.py": sha256(Path(__file__).with_name("analyze_and_plot.py")),
        },
        "output": output_path,
        "output_sha256": sha256(output_path),
        "merge_elapsed_minutes": (time.time() - start) / 60.0,
        "device": str(args.device),
    }
    write_json(output_dir / "run_manifest.json", json_ready(manifest))
    print(f"wrote {output_path} in {(time.time() - start) / 60.0:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
