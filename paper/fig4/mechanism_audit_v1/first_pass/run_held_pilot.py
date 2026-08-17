#!/usr/bin/env python3
"""Score a resumable, objectively selected held-prefix image pilot.

The pilot keeps all 1,000 trajectories and all 100 frozen RR units. Images are
chosen without looking at the new responses: contour-qualified images are
ordered by orientation coherence, divided into equal strata, and the median
image in each stratum is selected.  The resulting pilot is an early scientific
checkpoint, not a substitute for the later 100-image held-prefix bank.
"""

from __future__ import annotations

import argparse
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


PILOT_DIR = OUT_DIR / "first_pass_v1" / "held_pilot"


def choose_quantile_indices(values: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    order = candidates[np.argsort(values[candidates], kind="mergesort")]
    chunks = np.array_split(order, int(count))
    return np.asarray([chunk[len(chunk) // 2] for chunk in chunks if len(chunk)], dtype=int)


def select_images(images: pd.DataFrame, count: int) -> np.ndarray:
    coherence = pd.to_numeric(images["image_orientation_coherence"], errors="coerce").to_numpy(float)
    candidates = np.flatnonzero(np.isfinite(coherence) & (coherence >= 0.2))
    if candidates.size < count:
        raise ValueError(f"Only {candidates.size} contour-qualified images for count={count}")
    return choose_quantile_indices(coherence, candidates, count)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    selected_rows = select_images(images, int(args.n_images))
    selected = images.iloc[selected_rows].copy().reset_index(drop=True)
    selected["pilot_ordinal"] = np.arange(len(selected), dtype=int)
    selected.to_csv(PILOT_DIR / "selected_images.csv", index=False)

    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        histories = np.asarray(archive["held_initial_history_xy"], dtype=np.float32)
    shape = (len(selected), len(histories), 100)
    ssi = open_array(PILOT_DIR / "ssi_matrix.npy", shape)
    expected = open_array(PILOT_DIR / "expected_spikes_matrix.npy", shape)
    rate = open_array(PILOT_DIR / "mean_rate_matrix.npy", shape)
    population = open_array(PILOT_DIR / "population_ssi.npy", shape[:-1])
    completed_path = PILOT_DIR / "completed_images.npy"
    completed = np.load(completed_path).astype(bool) if completed_path.exists() else np.zeros(len(selected), bool)
    np.save(completed_path, completed)
    pending = np.flatnonzero(~completed)
    if int(args.max_images) > 0:
        pending = pending[: int(args.max_images)]
    if len(pending) == 0:
        print(f"No pending pilot images in {PILOT_DIR}", flush=True)
        return 0

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=str(args.device),
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    direct = build_direct_rr100_readout(scorer)
    canvas_cache: dict = {}
    started = time.perf_counter()
    for run_ordinal, pilot_ordinal in enumerate(pending, start=1):
        image_row = selected.iloc[int(pilot_ordinal)]
        image_id = int(image_row["image_index"])
        patch, _ = extract_patch(image_row, canvas_cache=canvas_cache, patch_size_px=540)
        values = score_corrected_histories_for_patch(
            scorer,
            patch,
            histories,
            trace_batch_size=int(args.trace_batch_size),
            frame_batch_size=int(args.frame_batch_size),
            direct_rr_readout=direct,
        )
        ssi[pilot_ordinal], expected[pilot_ordinal], rate[pilot_ordinal], population[pilot_ordinal] = values
        for arr in (ssi, expected, rate, population):
            arr.flush()
        completed[pilot_ordinal] = True
        np.save(completed_path, completed)
        elapsed = time.perf_counter() - started
        print(
            f"held pilot image {run_ordinal}/{len(pending)}: image_id={image_id}; "
            f"complete={completed.sum()}/{len(completed)}; {elapsed / run_ordinal:.1f} s/image",
            flush=True,
        )

    manifest = {
        "analysis": "held_initial_history_objective_image_pilot",
        "status": "pilot_not_full_correction_gate",
        "selection_rule": (
            "image_orientation_coherence >= 0.2; sort eligible images by coherence; "
            "split into equal strata; choose median row in each stratum"
        ),
        "n_selected_images": len(selected),
        "n_completed_images": int(completed.sum()),
        "n_trajectories": len(histories),
        "n_units": 100,
        "image_ids": selected["image_index"].astype(int).tolist(),
        "device": str(args.device),
        "trace_batch_size": int(args.trace_batch_size),
        "frame_batch_size": int(args.frame_batch_size),
        "bank": "real_trace_held_initial_history_v1",
        "bank_sha256": sha256_file(BANK_DIR / "corrected_history_trajectory_banks.npz"),
        "model": scorer.provenance,
    }
    write_json(PILOT_DIR / "manifest.json", manifest)
    if bool(np.all(completed)):
        write_json(
            PILOT_DIR / "hashes.json",
            {
                name: sha256_file(PILOT_DIR / name)
                for name in (
                    "ssi_matrix.npy",
                    "expected_spikes_matrix.npy",
                    "mean_rate_matrix.npy",
                    "population_ssi.npy",
                    "completed_images.npy",
                    "selected_images.csv",
                )
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
