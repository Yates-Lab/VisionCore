#!/usr/bin/env python3
"""Evaluate a twin linearized at measured (1x) FEM amplitude.

For each image/trajectory/output, the complete pre-softplus twin is expanded
about the matched 1x FEM movie and evaluated at every declared movement scale:

    T_1(s) = z(x_1) + J_z(x_1) [x_s - x_1].

The five scale directions are evaluated together as independent batch items so
that one exact JVP covers all scales in each frame batch.  The 1x prediction is
an exact anchor by construction; behavior away from 1x diagnoses curvature of
the fitted twin along the FEM-amplitude path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, LEGACY_MATRIX_DIR, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.nonlinear_phase_causal.common import N_SCORED, N_TRACES, OUT, json_ready, sha256
from paper.fig4.nonlinear_phase_causal.run_experiment import (
    SOURCE,
    _selection,
    historical_groups,
    preactivation,
    stabilized_tangent_preactivation,
)
from paper.fig4.nonlinear_phase_causal.common import unit_rate_metrics
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


ANCHOR_SCALE = 1.0
DEFAULT_OUT = OUT / "one_x_tangent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=2)
    parser.add_argument("--jvp-mode", choices=("reverse", "forward"), default="reverse")
    parser.add_argument(
        "--scales",
        default="0,1",
        help="comma-separated target amplitudes; must include the 1x anchor",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--image-ids", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parts-only", action="store_true")
    return parser.parse_args()


def parse_scales(value: str) -> np.ndarray:
    scales = np.asarray([float(item) for item in value.split(",") if item.strip()], dtype=np.float32)
    if len(scales) == 0 or np.any(scales < 0) or len(np.unique(scales)) != len(scales):
        raise ValueError(value)
    if not np.any(np.isclose(scales, ANCHOR_SCALE)):
        raise ValueError("Target scales must include the 1x anchor")
    return scales


def part_signature(
    *,
    image_id: int,
    trace_ids: np.ndarray,
    scales: np.ndarray,
    checkpoint_hash: str,
    frame_batch_size: int,
    jvp_mode: str,
) -> str:
    encoded = json.dumps(
        {
            "analysis": "fig4_one_x_tangent_v1",
            "image_id": int(image_id),
            "trace_ids": trace_ids.tolist(),
            "scales": scales.tolist(),
            "anchor_scale": ANCHOR_SCALE,
            "checkpoint_sha256": checkpoint_hash,
            "frame_batch_size": int(frame_batch_size),
            "jvp_mode": str(jvp_mode),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def score_trace(
    *,
    scorer: RealTraceMatrixScorer,
    readout: torch.nn.Module,
    patch: np.ndarray,
    base_history: np.ndarray,
    scales: np.ndarray,
    frame_batch_size: int,
    jvp_mode: str,
) -> dict[str, np.ndarray]:
    model = scorer.model.model
    dtype = next(model.parameters()).dtype
    image = _standardize_uint_like(patch)
    histories = scaled_histories(base_history[None], scales)
    stims = (make_corrected_causal_stims(image, histories, torch=scorer.torch) - 127.0) / 255.0
    stims = stims.reshape(len(scales), N_SCORED, *stims.shape[1:])
    anchor_index = int(np.flatnonzero(np.isclose(scales, ANCHOR_SCALE))[0])
    numerator = np.zeros((len(scales), readout.n_units), dtype=np.float64)
    expected = np.zeros_like(numerator)
    rate_sum = np.zeros_like(numerator)
    anchor_replicate_max_error = 0.0
    zero_direction_max_error = 0.0

    for frame_start in range(0, N_SCORED, int(frame_batch_size)):
        frame_stop = min(frame_start + int(frame_batch_size), N_SCORED)
        anchor = stims[anchor_index, frame_start:frame_stop].to(scorer.device)
        targets = torch.cat(
            [stims[index, frame_start:frame_stop] for index in range(len(scales))], dim=0
        ).to(scorer.device)
        anchors = torch.cat([anchor] * len(scales), dim=0)
        behavior = scorer._zero_behavior(len(anchors), dtype)
        anchor_z, tangent_z = stabilized_tangent_preactivation(
            lambda value: preactivation(model, readout, value, behavior),
            anchors,
            targets,
            mode=jvp_mode,
        )
        batch = frame_stop - frame_start
        anchor_blocks = anchor_z.reshape(len(scales), batch, *anchor_z.shape[1:])
        tangent_blocks = tangent_z.reshape(len(scales), batch, *tangent_z.shape[1:])
        anchor_replicate_max_error = max(
            anchor_replicate_max_error,
            float((anchor_blocks - anchor_blocks[anchor_index : anchor_index + 1]).abs().max().cpu()),
        )
        zero_direction_max_error = max(
            zero_direction_max_error,
            float((tangent_blocks[anchor_index] - anchor_blocks[anchor_index]).abs().max().cpu()),
        )
        for scale_index in range(len(scales)):
            rate = model.activation(tangent_blocks[scale_index])
            metrics = unit_rate_metrics(rate)
            weight = metrics["expected_spikes"].detach().cpu().numpy()
            expected[scale_index] += weight.sum(axis=0)
            numerator[scale_index] += (metrics["ssi"].detach().cpu().numpy() * weight).sum(axis=0)
            rate_sum[scale_index] += metrics["mean_rate_hz"].detach().cpu().numpy().sum(axis=0)
        del anchor, anchors, targets, anchor_z, tangent_z, anchor_blocks, tangent_blocks
        if str(scorer.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "ssi": (numerator / np.maximum(expected, 1e-12)).astype(np.float32),
        "expected_spikes": expected.astype(np.float32),
        "mean_rate_hz": (rate_sum / N_SCORED).astype(np.float32),
        "anchor_replicate_max_abs_error": np.asarray(anchor_replicate_max_error, dtype=np.float64),
        "zero_direction_max_abs_error": np.asarray(zero_direction_max_error, dtype=np.float64),
    }


def score_image(
    *,
    scorer: RealTraceMatrixScorer,
    readout: torch.nn.Module,
    patch: np.ndarray,
    image_id: int,
    trace_ids: np.ndarray,
    histories: np.ndarray,
    scales: np.ndarray,
    frame_batch_size: int,
    jvp_mode: str,
) -> dict[str, np.ndarray]:
    arrays = {
        "ssi": np.full((len(trace_ids), len(scales), readout.n_units), np.nan, dtype=np.float32),
        "expected_spikes": np.full(
            (len(trace_ids), len(scales), readout.n_units), np.nan, dtype=np.float32
        ),
        "mean_rate_hz": np.full(
            (len(trace_ids), len(scales), readout.n_units), np.nan, dtype=np.float32
        ),
        "anchor_replicate_max_abs_error": np.full(len(trace_ids), np.nan, dtype=np.float64),
        "zero_direction_max_abs_error": np.full(len(trace_ids), np.nan, dtype=np.float64),
    }
    for ordinal, (trace_id, history) in enumerate(zip(trace_ids, histories, strict=True)):
        result = score_trace(
            scorer=scorer,
            readout=readout,
            patch=patch,
            base_history=history,
            scales=scales,
            frame_batch_size=frame_batch_size,
            jvp_mode=jvp_mode,
        )
        for key in arrays:
            arrays[key][ordinal] = result[key]
        print(f"image={image_id} trace {ordinal + 1}/{len(trace_ids)} id={trace_id}", flush=True)
    arrays.update(
        {
            "image_index": np.asarray(image_id, dtype=np.int64),
            "trace_index": trace_ids.astype(np.int64),
            "scales": scales.astype(np.float32),
            "anchor_scale": np.asarray(ANCHOR_SCALE, dtype=np.float32),
        }
    )
    return arrays


def merge_parts(parts: list[Path], output_path: Path, groups: dict[str, np.ndarray]) -> None:
    payloads = []
    for path in parts:
        with np.load(path) as archive:
            payloads.append({key: np.asarray(archive[key]) for key in archive.files})
    merged = {
        key: np.stack([payload[key] for payload in payloads])
        for key in (
            "ssi",
            "expected_spikes",
            "mean_rate_hz",
            "anchor_replicate_max_abs_error",
            "zero_direction_max_abs_error",
        )
    }
    merged.update(
        {
            "selected_image_index": np.asarray([int(payload["image_index"]) for payload in payloads]),
            "selected_trace_index": payloads[0]["trace_index"],
            "scales": payloads[0]["scales"],
            "anchor_scale": payloads[0]["anchor_scale"],
            "low_unit_indices": groups["low"],
            "high_unit_indices": groups["high"],
        }
    )
    np.savez_compressed(output_path, **merged)


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    start = time.time()
    output_dir = args.output_dir.resolve()
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    selected_images, selected_traces = _selection()
    if args.image_ids:
        requested = np.asarray([int(value) for value in args.image_ids.split(",") if value.strip()])
        selected_images = selected_images.set_index("image_index").loc[requested].reset_index()
    if args.max_images:
        selected_images = selected_images.iloc[: args.max_images]
    if args.max_traces:
        selected_traces = selected_traces.iloc[: args.max_traces]
    image_ids = selected_images.image_index.to_numpy(int)
    trace_ids = selected_traces.trace_bank_index.to_numpy(int)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        histories = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)

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
    checkpoint_hash = sha256(MODEL_CHECKPOINT_PATH)
    images = pd.read_csv(SOURCE / "image_feature_table.csv").set_index("image_index")
    canvas_cache: dict[Any, Any] = {}
    parts: list[Path] = []
    for ordinal, image_id in enumerate(image_ids):
        part = parts_dir / f"image_{image_id:03d}.npz"
        signature = part_signature(
            image_id=int(image_id),
            trace_ids=trace_ids,
            scales=scales,
            checkpoint_hash=checkpoint_hash,
            frame_batch_size=int(args.frame_batch_size),
            jvp_mode=str(args.jvp_mode),
        )
        if part.is_file() and not args.overwrite:
            with np.load(part) as archive:
                saved = str(archive["part_signature"].item())
            if saved != signature:
                raise RuntimeError(f"Existing part signature differs: {part}")
            print(f"using completed part {part}", flush=True)
            parts.append(part)
            continue
        patch, _ = extract_patch(images.loc[image_id], canvas_cache=canvas_cache, patch_size_px=540)
        payload = score_image(
            scorer=scorer,
            readout=readout,
            patch=patch,
            image_id=int(image_id),
            trace_ids=trace_ids,
            histories=histories,
            scales=scales,
            frame_batch_size=int(args.frame_batch_size),
            jvp_mode=str(args.jvp_mode),
        )
        payload["part_signature"] = np.asarray(signature)
        np.savez_compressed(part, **payload)
        parts.append(part)
        print(f"completed image {ordinal + 1}/{len(image_ids)}", flush=True)
    if args.parts_only:
        return 0

    groups = historical_groups()
    canonical = len(image_ids) == 8 and len(trace_ids) == N_TRACES
    endpoint_only = np.array_equal(scales, np.asarray([0.0, 1.0], dtype=np.float32))
    stem = "one_x_endpoint_tangent_metrics" if endpoint_only else "one_x_tangent_metrics"
    output_path = output_dir / (f"{stem}.npz" if canonical else f"{stem}_partial.npz")
    merge_parts(parts, output_path, groups)
    manifest = {
        "analysis": "fig4_one_x_tangent_v1",
        "canonical_complete": canonical,
        "definition": "T1(s)=z(x1)+J_z(x1)(x_s-x1), followed by unchanged softplus and exact SSI",
        "anchor_scale": ANCHOR_SCALE,
        "n_images": len(image_ids),
        "n_traces": len(trace_ids),
        "scales": scales,
        "frame_batch_size": int(args.frame_batch_size),
        "jvp_mode": str(args.jvp_mode),
        "checkpoint": MODEL_CHECKPOINT_PATH,
        "checkpoint_sha256": checkpoint_hash,
        "output": output_path,
        "output_sha256": sha256(output_path),
        "merge_elapsed_minutes": (time.time() - start) / 60.0,
    }
    write_json(output_dir / "run_manifest.json", json_ready(manifest))
    print(json.dumps(json_ready(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
