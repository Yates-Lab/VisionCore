#!/usr/bin/env python3
"""Score compact step-and-hold and constant-velocity causal-history pilots."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import LEGACY_MATRIX_DIR, OUT_DIR, sha256_file, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import build_direct_rr100_readout
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    N_LAGS,
    OUT_SIZE,
    PPD,
    RealTraceMatrixScorer,
    _embed_time_lags,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = OUT_DIR / "first_pass_v1" / "synthetic_history_pilot"
AMPLITUDES_DEG = np.asarray([0.0, 0.02, 0.05, 0.10, 0.20], np.float32)
STEP_LAG_FRAMES = np.asarray([0, 4, 8, 12, 20, 30], np.int32)
VELOCITIES_DEG_S = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], np.float32)
DIRECTIONS_DEG = np.asarray([0.0, 45.0, 90.0, 135.0], np.float32)


def step_histories() -> np.ndarray:
    rows = []
    for amplitude in AMPLITUDES_DEG:
        for lag in STEP_LAG_FRAMES:
            for direction in DIRECTIONS_DEG:
                theta = math.radians(float(direction))
                displacement = float(amplitude) * np.asarray([math.cos(theta), math.sin(theta)], np.float32)
                history = np.zeros((N_LAGS, 2), np.float32)
                jump_index = N_LAGS - 1 - int(lag)
                history[jump_index:] = displacement
                rows.append(history)
    return np.stack(rows)


def velocity_histories() -> np.ndarray:
    rows = []
    relative_time = (np.arange(N_LAGS, dtype=np.float32) - (N_LAGS - 1)) / 120.0
    for velocity in VELOCITIES_DEG_S:
        for direction in DIRECTIONS_DEG:
            theta = math.radians(float(direction))
            vector = float(velocity) * np.asarray([math.cos(theta), math.sin(theta)], np.float32)
            rows.append(relative_time[:, None] * vector[None])
    return np.stack(rows)


def make_stims(image: np.ndarray, histories: np.ndarray, torch) -> object:
    image = _standardize_uint_like(image)
    batch = len(histories)
    repeated = np.broadcast_to(image[None], (batch * N_LAGS, *image.shape)).copy()
    eye = torch.from_numpy(histories.reshape(-1, 2))
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    shifted = _shift_movie_with_eye(
        torch.from_numpy(repeated), eye_norm, out_size=OUT_SIZE, scale_factor=1.0, torch=torch
    ).reshape(batch, N_LAGS, *OUT_SIZE)
    return torch.cat([_embed_time_lags(shifted[i], n_lags=N_LAGS, torch=torch) for i in range(batch)], dim=0)


def score_histories(scorer, readout, patch: np.ndarray, histories: np.ndarray, history_batch: int, frame_batch: int):
    from scripts.spatial_info import compute_rate_map

    unit_ssi = np.full((len(histories), 100), np.nan, np.float32)
    expected = np.full_like(unit_ssi, np.nan)
    dtype = next(scorer.model.model.parameters()).dtype
    scorer.model.model.eval()
    readout.eval()
    with scorer.torch.no_grad():
        for start in range(0, len(histories), history_batch):
            stop = min(start + history_batch, len(histories))
            stims = (make_stims(patch, histories[start:stop], scorer.torch) - 127.0) / 255.0
            outputs = []
            for frame_start in range(0, len(stims), frame_batch):
                x = stims[frame_start : frame_start + frame_batch].to(scorer.device)
                behavior = scorer._zero_behavior(len(x), dtype)
                rate_map = compute_rate_map(scorer.model, readout, x, behavior=behavior).clamp_min(0).to(scorer.torch.float64)
                flat = rate_map.reshape(len(x), rate_map.shape[1], -1)
                rbar = flat.mean(dim=2)
                gain = flat / (rbar[..., None] + 1e-8)
                bits = (gain * (gain + 1e-8).log() / math.log(2.0)).mean(dim=2)
                outputs.append((bits.cpu().numpy().astype(np.float32), (rbar / 120.0).cpu().numpy().astype(np.float32)))
            unit_ssi[start:stop] = np.concatenate([x[0] for x in outputs])
            expected[start:stop] = np.concatenate([x[1] for x in outputs])
    return unit_ssi, expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-batch-size", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "synthetic_history_response.npz"
    if output.is_file():
        print(f"Using existing {output}")
        return 0
    selected_images = pd.read_csv(
        ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling/selected_images.csv"
    )
    image_ids = selected_images.image_index.to_numpy(int)
    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    steps = step_histories()
    velocities = velocity_histories()
    step_shape = (len(image_ids), len(AMPLITUDES_DEG), len(STEP_LAG_FRAMES), len(DIRECTIONS_DEG), 100)
    velocity_shape = (len(image_ids), len(VELOCITIES_DEG_S), len(DIRECTIONS_DEG), 100)
    step_ssi = np.full(step_shape, np.nan, np.float32)
    step_expected = np.full_like(step_ssi, np.nan)
    velocity_ssi = np.full(velocity_shape, np.nan, np.float32)
    velocity_expected = np.full_like(velocity_ssi, np.nan)
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH, dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR, rr100_version=RR100_VERSION,
        device=str(args.device), strict=True, mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    readout = build_direct_rr100_readout(scorer)
    canvas_cache = {}
    for image_ordinal, image_id in enumerate(image_ids):
        patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
        values = score_histories(
            scorer, readout, patch, steps,
            history_batch=int(args.history_batch_size), frame_batch=int(args.frame_batch_size),
        )
        step_ssi[image_ordinal] = values[0].reshape(step_shape[1:])
        step_expected[image_ordinal] = values[1].reshape(step_shape[1:])
        values = score_histories(
            scorer, readout, patch, velocities,
            history_batch=int(args.history_batch_size), frame_batch=int(args.frame_batch_size),
        )
        velocity_ssi[image_ordinal] = values[0].reshape(velocity_shape[1:])
        velocity_expected[image_ordinal] = values[1].reshape(velocity_shape[1:])
        print(f"synthetic history image {image_ordinal + 1}/{len(image_ids)} id={image_id}", flush=True)
    np.savez_compressed(
        output, step_ssi=step_ssi, step_expected_spikes=step_expected,
        velocity_ssi=velocity_ssi, velocity_expected_spikes=velocity_expected,
        amplitudes_deg=AMPLITUDES_DEG, step_lag_frames=STEP_LAG_FRAMES,
        velocities_deg_s=VELOCITIES_DEG_S, directions_deg=DIRECTIONS_DEG,
        selected_image_index=image_ids,
    )
    write_json(
        OUT / "manifest.json",
        {
            "analysis": "single_output_step_hold_and_constant_velocity_pilot",
            "status": "pilot",
            "n_images": len(image_ids), "image_ids": image_ids,
            "amplitudes_deg": AMPLITUDES_DEG, "step_lag_frames": STEP_LAG_FRAMES,
            "velocities_deg_s": VELOCITIES_DEG_S, "directions_deg": DIRECTIONS_DEG,
            "endpoint": "one output after an exact 32-frame causal input",
            "checkpoint": MODEL_CHECKPOINT_PATH, "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
            "output": output, "output_sha256": sha256_file(output),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
