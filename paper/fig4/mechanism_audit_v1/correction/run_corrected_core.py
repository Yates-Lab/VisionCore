#!/usr/bin/env python3
"""Score one resumable corrected-history Figure 4 image shard."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    CORE_DIR,
    LEGACY_MATRIX_DIR,
    OUT_DIR,
    sha256_file,
    write_json,
)
from paper.fig4.mechanism_audit_v1.correction.scoring import (
    build_direct_rr100_readout,
    score_corrected_histories_for_patch,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


BANK_KEYS = {
    "true_history": ("real_trace_true_history_v1", "true_history_xy"),
    "held_initial": ("real_trace_held_initial_history_v1", "held_initial_history_xy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=sorted(BANK_KEYS), required=True)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-stop", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_CHECKPOINT_PATH)
    parser.add_argument("--dataset-configs", type=Path, default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument("--mcfarland-outputs", type=Path, default=ROOT / "scripts/mcfarland_outputs_mono.pkl")
    return parser.parse_args()


def open_array(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        arr = np.lib.format.open_memmap(path, mode="r+")
        if arr.shape != shape:
            raise ValueError(f"{path} has shape {arr.shape}, expected {shape}")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    arr[:] = np.nan
    arr.flush()
    return arr


def main() -> int:
    args = parse_args()
    bank_name, bank_key = BANK_KEYS[args.bank]
    image_start, image_stop = int(args.image_start), int(args.image_stop)
    if not (0 <= image_start < image_stop <= 100):
        raise ValueError("Image shard must satisfy 0 <= start < stop <= 100")
    shard_dir = CORE_DIR / bank_name / f"images_{image_start:03d}_{image_stop:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = OUT_DIR / "logs/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / f"{bank_name}_images_{image_start:03d}_{image_stop:03d}.lock").open(
        "a+", encoding="utf-8"
    )
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        histories = np.asarray(archive[bank_key], dtype=np.float32)
    n_images, n_traces, n_units = image_stop - image_start, len(histories), 100
    ssi = open_array(shard_dir / "ssi_matrix.npy", (n_images, n_traces, n_units))
    expected = open_array(shard_dir / "expected_spikes_matrix.npy", (n_images, n_traces, n_units))
    rate = open_array(shard_dir / "mean_rate_matrix.npy", (n_images, n_traces, n_units))
    population = open_array(shard_dir / "population_ssi.npy", (n_images, n_traces))
    completed_path = shard_dir / "completed_images.npy"
    if completed_path.exists():
        completed = np.load(completed_path).astype(bool)
    else:
        completed = np.zeros(n_images, dtype=bool)
        np.save(completed_path, completed)

    pending = np.flatnonzero(~completed)
    if int(args.max_images) > 0:
        pending = pending[: int(args.max_images)]
    manifest_path = shard_dir / "manifest.json"
    hashes_path = shard_dir / "hashes.json"
    if len(pending) == 0 and manifest_path.is_file() and hashes_path.is_file():
        print(f"Using complete locked shard {shard_dir}", flush=True)
        return 0

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=Path(args.checkpoint),
        dataset_configs=Path(args.dataset_configs),
        population_spec_dir=Path(args.population_spec_dir),
        rr100_version=RR100_VERSION,
        device=str(args.device),
        strict=True,
        mcfarland_outputs=Path(args.mcfarland_outputs),
    )
    direct_rr_readout = build_direct_rr100_readout(scorer)
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    start_time = time.perf_counter()
    done_now = 0
    for local_index in pending:
        global_index = image_start + int(local_index)
        patch, _ = extract_patch(images.iloc[global_index], canvas_cache=canvas_cache, patch_size_px=540)
        unit_ssi, unit_expected, unit_rate, pop_ssi = score_corrected_histories_for_patch(
            scorer,
            patch,
            histories,
            trace_batch_size=int(args.trace_batch_size),
            frame_batch_size=int(args.frame_batch_size),
            direct_rr_readout=direct_rr_readout,
        )
        ssi[local_index] = unit_ssi
        expected[local_index] = unit_expected
        rate[local_index] = unit_rate
        population[local_index] = pop_ssi
        for arr in (ssi, expected, rate, population):
            arr.flush()
        completed[local_index] = True
        np.save(completed_path, completed)
        done_now += 1
        elapsed = time.perf_counter() - start_time
        print(
            f"{bank_name} image {global_index + 1}/100; shard completed {completed.sum()}/{n_images}; "
            f"this run {elapsed / done_now:.1f} s/image",
            flush=True,
        )
    manifest = {
        "analysis": "corrected_history_core_ssi_shard",
        "bank": bank_name,
        "bank_key": bank_key,
        "legacy_name": "legacy_wrapped_prefix",
        "image_start": image_start,
        "image_stop": image_stop,
        "completed_images": int(completed.sum()),
        "n_images": n_images,
        "n_traces": n_traces,
        "n_units": n_units,
        "trace_batch_size": int(args.trace_batch_size),
        "frame_batch_size": int(args.frame_batch_size),
        "device": str(args.device),
        "readout_optimization": "exact direct slice of positive one-hot RR100 medoid channels",
        "bank_npz": BANK_DIR / "corrected_history_trajectory_banks.npz",
        "bank_npz_sha256": sha256_file(BANK_DIR / "corrected_history_trajectory_banks.npz"),
        "model": scorer.provenance,
        "outputs": {
            "ssi": shard_dir / "ssi_matrix.npy",
            "expected": shard_dir / "expected_spikes_matrix.npy",
            "mean_rate": shard_dir / "mean_rate_matrix.npy",
            "population_ssi": shard_dir / "population_ssi.npy",
            "completed": completed_path,
        },
    }
    write_json(shard_dir / "manifest.json", manifest)
    if bool(np.all(completed)):
        write_json(
            shard_dir / "hashes.json",
            {name: sha256_file(path) for name, path in manifest["outputs"].items()},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
