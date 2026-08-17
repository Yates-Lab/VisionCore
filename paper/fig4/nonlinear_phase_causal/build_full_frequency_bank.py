#!/usr/bin/env python3
"""Rebuild the response bank with the complete 32 x 25 x 25 DCT volume."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.nonlinear_phase_causal.response_subspace import (
    N_LAGS,
    OUTPUT_SIZE,
    PATCH_SIZE,
    extract_local_patches,
    generate_movie_batch,
    orthonormal_dct,
)
from paper.fig4.nonlinear_phase_causal.run_experiment import preactivation
from paper.fig4.nonlinear_phase_causal.run_response_subspace_pilot import load_teacher


SOURCE = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot"
DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_full_frequency"
N_FULL_FEATURES = N_LAGS * PATCH_SIZE * PATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--movie-batch-size", type=int, default=8)
    parser.add_argument("--gradient-movies", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patches_to_full_dct(patches: torch.Tensor) -> torch.Tensor:
    if patches.ndim != 5 or tuple(patches.shape[-3:]) != (N_LAGS, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(tuple(patches.shape))
    batch, grid = patches.shape[:2]
    temporal = orthonormal_dct(N_LAGS, dtype=patches.dtype).to(patches.device)
    spatial = orthonormal_dct(PATCH_SIZE, dtype=patches.dtype).to(patches.device)
    value = patches.reshape(batch * grid, N_LAGS, PATCH_SIZE * PATCH_SIZE)
    value = torch.matmul(temporal, value).reshape(-1, N_LAGS, PATCH_SIZE, PATCH_SIZE)
    value = torch.matmul(spatial, value)
    value = torch.matmul(value, spatial.T)
    return value.reshape(batch, grid, N_FULL_FEATURES)


def copy_targets(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "unit_selection.csv", output / "unit_selection.csv")
    for name in (
        "train_dense_z.npy", "val_dense_z.npy", "test_dense_z.npy",
        "train_dense_offsets.npy", "val_dense_offsets.npy", "test_dense_offsets.npy",
        "train_z_full.npy", "val_z_full.npy", "test_z_full.npy",
    ):
        shutil.copy2(source / "bank" / name, output / "bank" / name)


def build_features(args, manifest):
    bank = args.output_dir / "bank"
    seed = int(manifest["seed"])
    for split_number, split in enumerate(("train", "val", "test")):
        offsets = np.load(bank / f"{split}_dense_offsets.npy", mmap_mode="r")
        n_movies, location_count = offsets.shape[:2]
        path = bank / f"{split}_dense_features.npy"
        expected = (n_movies * location_count, N_FULL_FEATURES)
        if path.exists() and not args.overwrite and np.load(path, mmap_mode="r").shape == expected:
            print(f"full-frequency {split}: reusing {path}", flush=True)
            continue
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=expected)
        for start in range(0, n_movies, args.movie_batch_size):
            stop = min(start + args.movie_batch_size, n_movies)
            movie, _ = generate_movie_batch(
                stop - start,
                seed=seed + split_number * 1_000_000 + start,
                device=args.device,
            )
            chunks = []
            for movie_index in range(stop - start):
                movie_offsets = np.asarray(offsets[start + movie_index], dtype=int)
                chunks.append(
                    patches_to_full_dct(
                        extract_local_patches(movie[movie_index : movie_index + 1], movie_offsets)
                    )[0]
                )
            coefficients = torch.stack(chunks)
            output[start * location_count : stop * location_count] = (
                coefficients.reshape(-1, N_FULL_FEATURES).cpu().numpy().astype(np.float16)
            )
            if start == 0 or stop == n_movies or stop % 128 == 0:
                print(f"full-frequency {split}: {stop}/{n_movies} movies", flush=True)
        output.flush()


def measure_gradients(args, manifest):
    bank = args.output_dir / "bank"
    selection = pd.read_csv(args.output_dir / "unit_selection.csv").unit_index.to_numpy(int)
    n_movies = min(int(args.gradient_movies), int(manifest["movie_counts"]["train"]))
    path = bank / "train_response_gradient_dct.npy"
    expected = (n_movies, len(selection), N_FULL_FEATURES)
    if path.exists() and not args.overwrite and np.load(path, mmap_mode="r").shape == expected:
        print(f"full-frequency gradients: reusing {path}", flush=True)
        return
    scorer, readout = load_teacher(args.device)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=expected)
    seed = int(manifest["seed"])
    radius = PATCH_SIZE // 2
    for start in range(0, n_movies, args.movie_batch_size):
        stop = min(start + args.movie_batch_size, n_movies)
        movie, _ = generate_movie_batch(stop - start, seed=seed + start, device=args.device)
        movie.requires_grad_(True)
        behavior = scorer._zero_behavior(len(movie), movie.dtype)
        z = preactivation(scorer.model.model, readout, movie, behavior)[:, selection]
        for subset_index in range(len(selection)):
            gradient = torch.autograd.grad(
                z[:, subset_index, OUTPUT_SIZE // 2, OUTPUT_SIZE // 2].sum(),
                movie,
                retain_graph=subset_index + 1 < len(selection),
            )[0]
            local = gradient[:, 0, :, 75 - radius : 76 + radius, 75 - radius : 76 + radius]
            coefficients = patches_to_full_dct(local[:, None])[:, 0]
            output[start:stop, subset_index] = coefficients.detach().cpu().numpy().astype(np.float16)
        if start == 0 or stop == n_movies or stop % 32 == 0:
            print(f"full-frequency gradients: {stop}/{n_movies}", flush=True)
    output.flush()


def compute_scaling(bank: Path):
    train = np.load(bank / "train_dense_features.npy", mmap_mode="r")
    sums = np.zeros(train.shape[1], dtype=np.float64)
    squares = np.zeros(train.shape[1], dtype=np.float64)
    for start in range(0, len(train), 1024):
        value = np.asarray(train[start : start + 1024], dtype=np.float32)
        sums += value.sum(axis=0, dtype=np.float64)
        squares += np.square(value, dtype=np.float64).sum(axis=0)
    mean = sums / len(train)
    variance = squares / len(train) - np.square(mean)
    np.savez(
        bank / "feature_scaling.npz",
        mean=mean.astype(np.float32),
        std=np.sqrt(np.maximum(variance, 1e-10)).astype(np.float32),
    )


def main():
    args = parse_args()
    started = time.time()
    (args.output_dir / "bank").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source_dir / "bank/manifest.json").read_text())
    copy_targets(args.source_dir, args.output_dir)
    build_features(args, manifest)
    compute_scaling(args.output_dir / "bank")
    measure_gradients(args, manifest)
    manifest.update(
        {
            "analysis": "RR100 response bank using complete local space-time volume",
            "feature_basis": "all 32 temporal DCT modes crossed with all 25 x 25 spatial DCT modes",
            "n_features": N_FULL_FEATURES,
            "source_target_bank": str(args.source_dir.resolve()),
            "elapsed_seconds": time.time() - started,
        }
    )
    (args.output_dir / "bank/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
