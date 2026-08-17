#!/usr/bin/env python3
"""Generate the one permitted exact ConvGRU/RR100 cache for the low-rank test.

The cache is deliberately pair-resumable.  A completed image--trajectory pair
contains all five movement scales and all forty scored output frames, so no
partially written pair is ever treated as valid by downstream analyses.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    CACHE,
    MAP_CACHE,
    MAP_SIZE,
    N_CHANNELS,
    N_FRAMES,
    N_IMAGES,
    N_SCALES,
    N_TRAJECTORIES,
    N_UNITS,
    READOUT_CACHE,
    SCALES,
    SOURCE_EXACT,
    SOURCE_SELECTION,
    STATE_CACHE,
    STATE_SIZE,
    ensure_output_dirs,
    json_ready,
    rate_map_components,
    readout_preactivation,
    sha256_file,
    write_json,
)
from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    MODEL_CHECKPOINT_SHA256,
    RR100_VERSION,
)


GENERATOR_VERSION = "fig4-causal-low-rank-cache-v1"
MANIFEST = CACHE / "cache_generation_manifest.json"
LOCK_PATH = CACHE / "cache_generation.lock"
VALIDATION_NAMES = (
    "state_h_max_abs",
    "state_z_relative_rmse",
    "state_rate_relative_rmse",
    "state_gain_relative_rmse",
    "state_ssi_max_abs",
    "stored_z_relative_rmse",
    "stored_rate_relative_rmse",
    "stored_gain_relative_rmse",
    "stored_rate_ssi_max_abs",
    "source_aggregate_ssi_max_abs",
    "source_aggregate_expected_max_abs",
    "source_aggregate_expected_relative_rmse",
)


class AcceleratorBudgetReached(RuntimeError):
    """Raised only after persisting the work and timing for the last safe batch."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--gpu-budget-hours", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=0, help="Development/resume limit; zero means all pairs.")
    parser.add_argument("--max-state-gain-relative-rmse", type=float, default=1e-3)
    parser.add_argument("--max-state-ssi-error", type=float, default=5e-4)
    parser.add_argument("--max-stored-gain-relative-rmse", type=float, default=1e-3)
    parser.add_argument("--max-stored-rate-ssi-error", type=float, default=5e-4)
    parser.add_argument("--max-source-ssi-error", type=float, default=5e-5)
    parser.add_argument("--max-source-expected-relative-rmse", type=float, default=1e-4)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_generation_lock() -> Iterator[None]:
    CACHE.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _dataset(
    handle: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    chunks: tuple[int, ...],
    *,
    fillvalue: Any,
) -> h5py.Dataset:
    return handle.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression="lzf",
        shuffle=True,
        fillvalue=fillvalue,
    )


def _initialize_state_cache(path: Path, image_ids: np.ndarray, trajectory_ids: np.ndarray) -> None:
    with h5py.File(path, "x", libver="latest") as handle:
        _dataset(
            handle,
            "h",
            (N_IMAGES, N_TRAJECTORIES, N_SCALES, N_FRAMES, N_CHANNELS, STATE_SIZE, STATE_SIZE),
            "f2",
            (1, 1, 1, 1, N_CHANNELS, STATE_SIZE, STATE_SIZE),
            fillvalue=np.float16(np.nan),
        )
        handle.create_dataset("completed_pairs", shape=(N_IMAGES, N_TRAJECTORIES), dtype="?", fillvalue=False)
        handle.create_dataset("pair_elapsed_seconds", shape=(N_IMAGES, N_TRAJECTORIES), dtype="f8", fillvalue=np.nan)
        handle.create_dataset("pair_accelerator_seconds", shape=(N_IMAGES, N_TRAJECTORIES), dtype="f8", fillvalue=np.nan)
        handle.create_dataset(
            "float16_validation",
            shape=(N_IMAGES, N_TRAJECTORIES, len(VALIDATION_NAMES)),
            dtype="f8",
            fillvalue=np.nan,
        )
        handle.create_dataset("image_ids", data=image_ids.astype(np.int64))
        handle.create_dataset("trajectory_ids", data=trajectory_ids.astype(np.int64))
        handle.create_dataset("scales", data=SCALES)
        handle.attrs.update(
            {
                "generator_version": GENERATOR_VERSION,
                "created_utc": utc_now(),
                "state_definition": "last internal ConvGRU time point supplied to the exact frozen RR100 readout",
                "dtype": "float16",
                "compression": "lzf+shuffle",
                "validation_names_json": json.dumps(VALIDATION_NAMES),
                "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
                "total_accelerator_seconds_consumed": 0.0,
                "complete": False,
            }
        )


def _initialize_map_cache(path: Path, image_ids: np.ndarray, trajectory_ids: np.ndarray) -> None:
    map_shape = (N_IMAGES, N_TRAJECTORIES, N_SCALES, N_FRAMES, N_UNITS, MAP_SIZE, MAP_SIZE)
    map_chunks = (1, 1, 1, 1, N_UNITS, MAP_SIZE, MAP_SIZE)
    metric_shape = (N_IMAGES, N_TRAJECTORIES, N_SCALES, N_FRAMES, N_UNITS)
    metric_chunks = (1, 1, 1, N_FRAMES, N_UNITS)
    with h5py.File(path, "x", libver="latest") as handle:
        for name in ("preactivation", "rate", "gain"):
            _dataset(handle, name, map_shape, "f2", map_chunks, fillvalue=np.float16(np.nan))
        for name in ("ssi", "expected_spikes", "mean_rate"):
            _dataset(handle, name, metric_shape, "f4", metric_chunks, fillvalue=np.float32(np.nan))
        handle.create_dataset("completed_pairs", shape=(N_IMAGES, N_TRAJECTORIES), dtype="?", fillvalue=False)
        handle.create_dataset("image_ids", data=image_ids.astype(np.int64))
        handle.create_dataset("trajectory_ids", data=trajectory_ids.astype(np.int64))
        handle.create_dataset("scales", data=SCALES)
        handle.attrs.update(
            {
                "generator_version": GENERATOR_VERSION,
                "created_utc": utc_now(),
                "map_definition": "exact tiled RR100 preactivation/rate/mean-normalized rate map",
                "map_dtype": "float16",
                "metric_dtype": "float32",
                "compression": "lzf+shuffle",
                "normalization": "rate / mean_xy(rate), independently per frame and RR100 unit",
                "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
                "complete": False,
            }
        )


def _validate_cache_schema(
    state: h5py.File,
    maps: h5py.File,
    image_ids: np.ndarray,
    trajectory_ids: np.ndarray,
) -> np.ndarray:
    expected_h = (N_IMAGES, N_TRAJECTORIES, N_SCALES, N_FRAMES, N_CHANNELS, STATE_SIZE, STATE_SIZE)
    expected_map = (N_IMAGES, N_TRAJECTORIES, N_SCALES, N_FRAMES, N_UNITS, MAP_SIZE, MAP_SIZE)
    expected_metric = expected_map[:-2]
    if state["h"].shape != expected_h:
        raise RuntimeError(f"State-cache shape mismatch: {state['h'].shape} != {expected_h}")
    for name in ("preactivation", "rate", "gain"):
        if maps[name].shape != expected_map:
            raise RuntimeError(f"Map-cache {name} shape mismatch: {maps[name].shape} != {expected_map}")
    for name in ("ssi", "expected_spikes", "mean_rate"):
        if maps[name].shape != expected_metric:
            raise RuntimeError(f"Map-cache {name} shape mismatch: {maps[name].shape} != {expected_metric}")
    for handle in (state, maps):
        if not np.array_equal(handle["image_ids"][:], image_ids):
            raise RuntimeError("Image identifiers differ from the authoritative exact subset")
        if not np.array_equal(handle["trajectory_ids"][:], trajectory_ids):
            raise RuntimeError("Trajectory identifiers differ from the authoritative exact subset")
        if not np.array_equal(handle["scales"][:], SCALES):
            raise RuntimeError("Movement-scale ordering differs from the authoritative exact subset")
        state_done = state["completed_pairs"][:]
    map_done = maps["completed_pairs"][:]
    if not np.array_equal(state_done, map_done):
        # A crash can occur after one completion bit is flushed.  The pair is
        # conservatively invalidated in both files and replayed in full.
        repaired = state_done & map_done
        state["completed_pairs"][:] = repaired
        maps["completed_pairs"][:] = repaired
        state.flush()
        maps.flush()
        state_done = repaired
    return state_done


def _relative_rmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (estimate.double() - reference.double()).square().mean().sqrt()
    denominator = reference.double().square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).detach().cpu())


def _max_abs(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((estimate.double() - reference.double()).abs().max().detach().cpu())


def _float16_roundtrip(value: torch.Tensor, label: str) -> torch.Tensor:
    quantized = value.to(torch.float16)
    if not bool(torch.isfinite(quantized).all()):
        raise FloatingPointError(f"Float16 overflow/non-finite value in {label}")
    return quantized.to(torch.float32)


def _batch_validation(
    h: torch.Tensor,
    z: torch.Tensor,
    rate: torch.Tensor,
    gain: torch.Tensor,
    ssi: torch.Tensor,
    readout: Any,
) -> np.ndarray:
    h_q = _float16_roundtrip(h, "h")
    z_from_h_q = readout(h_q)
    rate_from_h_q = torch.nn.functional.softplus(z_from_h_q)
    comp_from_h_q = rate_map_components(rate_from_h_q.double())
    z_q = _float16_roundtrip(z, "preactivation")
    rate_q = _float16_roundtrip(rate, "rate")
    gain_q = _float16_roundtrip(gain, "gain")
    stored_rate_ssi = rate_map_components(rate_q.double())["ssi"]
    return np.asarray(
        [
            _max_abs(h_q, h),
            _relative_rmse(z_from_h_q, z),
            _relative_rmse(rate_from_h_q, rate),
            _relative_rmse(comp_from_h_q["gain"], gain),
            _max_abs(comp_from_h_q["ssi"], ssi),
            _relative_rmse(z_q, z),
            _relative_rmse(rate_q, rate),
            _relative_rmse(gain_q, gain),
            _max_abs(stored_rate_ssi, ssi),
            np.nan,
            np.nan,
            np.nan,
        ],
        dtype=np.float64,
    )


def _source_validation(
    frame_ssi: np.ndarray,
    frame_expected: np.ndarray,
    source_ssi: np.ndarray,
    source_expected: np.ndarray,
) -> tuple[float, float, float]:
    aggregate_expected = np.sum(frame_expected, axis=1)
    aggregate_ssi = np.sum(frame_ssi * frame_expected, axis=1) / np.maximum(aggregate_expected, 1e-12)
    ssi_abs = float(np.max(np.abs(aggregate_ssi.astype(np.float64) - source_ssi.astype(np.float64))))
    expected_delta = aggregate_expected.astype(np.float64) - source_expected.astype(np.float64)
    expected_abs = float(np.max(np.abs(expected_delta)))
    expected_rel = float(
        np.sqrt(np.mean(expected_delta**2))
        / max(float(np.sqrt(np.mean(source_expected.astype(np.float64) ** 2))), 1e-30)
    )
    return ssi_abs, expected_abs, expected_rel


def _check_validation(value: np.ndarray, args: argparse.Namespace, image_id: int, trajectory_id: int) -> None:
    named = dict(zip(VALIDATION_NAMES, value.tolist()))
    limits = {
        "state_gain_relative_rmse": (named["state_gain_relative_rmse"], args.max_state_gain_relative_rmse),
        "state_ssi_max_abs": (named["state_ssi_max_abs"], args.max_state_ssi_error),
        "stored_gain_relative_rmse": (named["stored_gain_relative_rmse"], args.max_stored_gain_relative_rmse),
        "stored_rate_ssi_max_abs": (named["stored_rate_ssi_max_abs"], args.max_stored_rate_ssi_error),
        "source_aggregate_ssi_max_abs": (named["source_aggregate_ssi_max_abs"], args.max_source_ssi_error),
        "source_aggregate_expected_relative_rmse": (
            named["source_aggregate_expected_relative_rmse"],
            args.max_source_expected_relative_rmse,
        ),
    }
    failed = {key: {"value": value, "limit": limit} for key, (value, limit) in limits.items() if value > limit}
    if failed:
        raise RuntimeError(
            f"Float16/source validation failed for image={image_id}, trajectory={trajectory_id}: {failed}"
        )


def _save_or_validate_readout(
    readout: Any,
    image_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    low_unit_indices: np.ndarray,
    high_unit_indices: np.ndarray,
) -> None:
    payload = {
        "feature_weights": readout.features.weight.detach().cpu().numpy()[:, :, 0, 0].astype(np.float32),
        "bias": readout.bias.detach().cpu().numpy().astype(np.float32),
        "space_weights": readout.space_weights.detach().cpu().numpy()[:, 0].astype(np.float32),
        "image_ids": image_ids.astype(np.int64),
        "trajectory_ids": trajectory_ids.astype(np.int64),
        "scales": SCALES,
        "low_unit_indices": low_unit_indices.astype(np.int64),
        "high_unit_indices": high_unit_indices.astype(np.int64),
    }
    if READOUT_CACHE.exists():
        with np.load(READOUT_CACHE) as existing:
            existing_payload = {key: np.asarray(existing[key]) for key in existing.files}
        core_keys = {"feature_weights", "bias", "space_weights", "image_ids", "trajectory_ids", "scales"}
        if not core_keys.issubset(existing_payload):
            raise RuntimeError(f"Existing readout cache is missing frozen-readout keys: {sorted(core_keys)}")
        for key in core_keys:
            if not np.array_equal(existing_payload[key], payload[key]):
                raise RuntimeError(f"Existing frozen readout cache differs at {key}")
        missing = set(payload) - set(existing_payload)
        if missing:
            existing_payload.update({key: payload[key] for key in missing})
            np.savez_compressed(READOUT_CACHE, **existing_payload)
        elif set(existing_payload) - set(payload):
            raise RuntimeError(f"Existing readout cache has unexpected keys: {sorted(set(existing_payload) - set(payload))}")
        return
    np.savez_compressed(READOUT_CACHE, **payload)


def _load_authoritative_subset() -> dict[str, np.ndarray]:
    with np.load(SOURCE_EXACT) as archive:
        result = {
            "image_ids": np.asarray(archive["selected_image_index"], dtype=np.int64),
            "trajectory_ids": np.asarray(archive["selected_trace_index"], dtype=np.int64),
            "scales": np.asarray(archive["scales"], dtype=np.float32),
            "ssi": np.asarray(archive["ssi"], dtype=np.float32),
            "expected_spikes": np.asarray(archive["expected_spikes"], dtype=np.float32),
            "low_unit_indices": np.asarray(archive["low_unit_indices"], dtype=np.int64),
            "high_unit_indices": np.asarray(archive["high_unit_indices"], dtype=np.int64),
        }
    if result["image_ids"].shape != (N_IMAGES,) or result["trajectory_ids"].shape != (N_TRAJECTORIES,):
        raise RuntimeError("Authoritative subset is not the required 8-image x 24-trajectory design")
    if not np.array_equal(result["scales"], SCALES):
        raise RuntimeError("Authoritative movement-scale order changed")
    selected_images = pd.read_csv(SOURCE_SELECTION / "selected_images.csv").image_index.to_numpy(np.int64)
    selected_trajectories = pd.read_csv(SOURCE_SELECTION / "selected_traces.csv").trace_bank_index.to_numpy(np.int64)
    if not np.array_equal(selected_images, result["image_ids"]):
        raise RuntimeError("Selected-image table and exact cache disagree")
    if not np.array_equal(selected_trajectories, result["trajectory_ids"]):
        raise RuntimeError("Selected-trajectory table and exact cache disagree")
    return result


def _manifest_payload(
    *,
    status: str,
    args: argparse.Namespace,
    completed: np.ndarray,
    state: h5py.File,
    started_utc: str,
    model_provenance: dict[str, Any] | None,
    message: str | None = None,
) -> dict[str, Any]:
    timing = state["pair_accelerator_seconds"][:]
    validation = state["float16_validation"][:]
    valid_rows = validation[np.asarray(completed, dtype=bool)]
    maxima = (
        {name: float(np.nanmax(valid_rows[:, index])) for index, name in enumerate(VALIDATION_NAMES)}
        if len(valid_rows)
        else {}
    )
    payload = {
        "analysis": "causal_low_rank_convgru_state_cache",
        "generator_version": GENERATOR_VERSION,
        "status": status,
        "message": message,
        "started_utc": started_utc,
        "updated_utc": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": args.device,
        "gpu_budget_hours": args.gpu_budget_hours,
        "accelerator_forward_hours": float(
            state.attrs.get("total_accelerator_seconds_consumed", np.nansum(timing))
        )
        / 3600.0,
        "completed_pairs": int(np.count_nonzero(completed)),
        "total_pairs": int(completed.size),
        "complete_fraction": float(np.mean(completed)),
        "state_cache": STATE_CACHE,
        "map_cache": MAP_CACHE,
        "readout_cache": READOUT_CACHE,
        "source_exact": SOURCE_EXACT,
        "source_exact_sha256": sha256_file(SOURCE_EXACT),
        "checkpoint": MODEL_CHECKPOINT_PATH,
        "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "selection": "fixed authoritative 8 images x 24 drift trajectories x 5 scales x 40 scored frames",
        "storage": "float16 LZF+shuffle for h/preactivation/rate/gain; float32 for per-frame SSI/expected/mean rate",
        "float16_validation_names": VALIDATION_NAMES,
        "float16_validation_maxima": maxima,
        "arguments": vars(args),
        "model_provenance": model_provenance,
    }
    return json_ready(payload)


def _prepare_files(args: argparse.Namespace, image_ids: np.ndarray, trajectory_ids: np.ndarray) -> None:
    targets = (STATE_CACHE, MAP_CACHE, READOUT_CACHE, MANIFEST)
    if args.overwrite:
        for path in targets:
            if path.exists():
                path.unlink()
    state_exists = STATE_CACHE.exists()
    map_exists = MAP_CACHE.exists()
    if state_exists != map_exists:
        raise RuntimeError(
            "Only one of the two cache files exists. Refusing to guess provenance; rerun with --overwrite."
        )
    if not state_exists:
        _initialize_state_cache(STATE_CACHE, image_ids, trajectory_ids)
        _initialize_map_cache(MAP_CACHE, image_ids, trajectory_ids)


def run(args: argparse.Namespace) -> int:
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be positive")
    if not (0.0 < args.gpu_budget_hours <= 4.0):
        raise ValueError("The authorized GPU budget must be in (0, 4] hours")
    ensure_output_dirs()
    authoritative = _load_authoritative_subset()
    image_ids = authoritative["image_ids"]
    trajectory_ids = authoritative["trajectory_ids"]
    _prepare_files(args, image_ids, trajectory_ids)
    started_utc = utc_now()

    with h5py.File(STATE_CACHE, "r+", libver="latest") as state, h5py.File(
        MAP_CACHE, "r+", libver="latest"
    ) as maps:
        completed = _validate_cache_schema(state, maps, image_ids, trajectory_ids)
        if bool(completed.all()):
            state.attrs["complete"] = True
            maps.attrs["complete"] = True
            state.flush()
            maps.flush()
            print(f"Exact caches already complete; refusing to overwrite without --overwrite: {STATE_CACHE}")
            return 0
        prior_accelerator_seconds = float(
            state.attrs.get(
                "total_accelerator_seconds_consumed",
                np.nansum(state["pair_accelerator_seconds"][:]),
            )
        )
        state.attrs["total_accelerator_seconds_consumed"] = prior_accelerator_seconds
        if prior_accelerator_seconds >= args.gpu_budget_hours * 3600.0:
            write_json(
                MANIFEST,
                _manifest_payload(
                    status="gpu_budget_reached",
                    args=args,
                    completed=completed,
                    state=state,
                    started_utc=started_utc,
                    model_provenance=None,
                    message="Budget was already exhausted before resumption.",
                ),
            )
            return 2

        scorer = RealTraceMatrixScorer.load(
            checkpoint_path=MODEL_CHECKPOINT_PATH,
            dataset_configs=DEFAULT_DATASET_CONFIGS,
            population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
            rr100_version=RR100_VERSION,
            device=str(args.device),
            strict=True,
            mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
        )
        model = scorer.model.model.eval()
        readout = build_direct_readout(scorer).eval()
        _save_or_validate_readout(
            readout,
            image_ids,
            trajectory_ids,
            authoritative["low_unit_indices"],
            authoritative["high_unit_indices"],
        )
        if not isinstance(model.activation, torch.nn.Softplus):
            raise RuntimeError(f"Expected exact nn.Softplus output, found {type(model.activation).__name__}")
        dtype = next(model.parameters()).dtype
        images = pd.read_csv(
            ROOT
            / "outputs/active_sensing_movie_information/"
            "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/"
            "image_feature_table.csv"
        ).sort_values("image_index").reset_index(drop=True)
        if not all(int(images.iloc[int(image_id)].image_index) == int(image_id) for image_id in image_ids):
            raise RuntimeError("Image table is not position-indexed by stable image identifier")
        with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
            base_histories = np.asarray(archive["true_history_xy"][trajectory_ids], dtype=np.float32)

        canvas_cache: dict[Any, Any] = {}
        generated_this_call = 0
        try:
            for image_position, image_id in enumerate(image_ids):
                patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
                standardized_patch = _standardize_uint_like(patch)
                for trajectory_position, trajectory_id in enumerate(trajectory_ids):
                    if completed[image_position, trajectory_position]:
                        continue
                    if args.max_pairs > 0 and generated_this_call >= args.max_pairs:
                        write_json(
                            MANIFEST,
                            _manifest_payload(
                                status="partial_requested",
                                args=args,
                                completed=completed,
                                state=state,
                                started_utc=started_utc,
                                model_provenance=scorer.provenance,
                                message="Stopped at the explicit --max-pairs development limit.",
                            ),
                        )
                        return 0
                    used = float(state.attrs["total_accelerator_seconds_consumed"])
                    if used >= args.gpu_budget_hours * 3600.0:
                        write_json(
                            MANIFEST,
                            _manifest_payload(
                                status="gpu_budget_reached",
                                args=args,
                                completed=completed,
                                state=state,
                                started_utc=started_utc,
                                model_provenance=scorer.provenance,
                                message="Stopped before beginning another complete image--trajectory pair.",
                            ),
                        )
                        return 2

                    pair_started = time.perf_counter()
                    pair_accelerator = 0.0
                    pair_validation = np.full(len(VALIDATION_NAMES), -np.inf, dtype=np.float64)
                    frame_ssi = np.full((N_SCALES, N_FRAMES, N_UNITS), np.nan, dtype=np.float32)
                    frame_expected = np.full_like(frame_ssi, np.nan)
                    histories = scaled_histories(base_histories[trajectory_position : trajectory_position + 1])
                    stims = (
                        make_corrected_causal_stims(standardized_patch, histories, torch=scorer.torch) - 127.0
                    ) / 255.0
                    movies = stims.reshape(N_SCALES, N_FRAMES, *stims.shape[1:])

                    with torch.no_grad():
                        for scale_position in range(N_SCALES):
                            for frame_start in range(0, N_FRAMES, args.frame_batch_size):
                                frame_stop = min(frame_start + args.frame_batch_size, N_FRAMES)
                                x = movies[scale_position, frame_start:frame_stop].to(scorer.device)
                                if str(args.device).startswith("cuda"):
                                    torch.cuda.synchronize(device=args.device)
                                accelerator_started = time.perf_counter()
                                recurrent = model.core_forward(x, scorer._zero_behavior(len(x), dtype))
                                h = recurrent[:, :, -1].float()
                                z = readout(h).float()
                                rate = model.activation(z).float()
                                if h.shape[1:] != (N_CHANNELS, STATE_SIZE, STATE_SIZE):
                                    raise RuntimeError(f"Unexpected last ConvGRU state shape: {tuple(h.shape)}")
                                if z.shape[1:] != (N_UNITS, MAP_SIZE, MAP_SIZE):
                                    raise RuntimeError(f"Unexpected RR100 map shape: {tuple(z.shape)}")
                                direct_z = readout_preactivation(
                                    h,
                                    readout.features.weight[:, :, 0, 0],
                                    readout.bias,
                                    readout.space_weights[:, 0],
                                )
                                if not torch.allclose(direct_z, z, atol=2e-6, rtol=2e-6):
                                    maximum = float((direct_z - z).abs().max().detach().cpu())
                                    raise RuntimeError(
                                        "Cached direct-readout tensors do not reproduce the frozen readout "
                                        f"within the preregistered tolerance; max_abs={maximum:.8g}"
                                    )
                                components = rate_map_components(rate.double())
                                validation = _batch_validation(
                                    h, z, rate, components["gain"], components["ssi"], readout
                                )
                                if str(args.device).startswith("cuda"):
                                    torch.cuda.synchronize(device=args.device)
                                batch_accelerator = time.perf_counter() - accelerator_started
                                pair_accelerator += batch_accelerator
                                state.attrs["total_accelerator_seconds_consumed"] = float(
                                    state.attrs["total_accelerator_seconds_consumed"]
                                ) + batch_accelerator
                                pair_validation = np.maximum(pair_validation, validation)
                                destination = (
                                    image_position,
                                    trajectory_position,
                                    scale_position,
                                    slice(frame_start, frame_stop),
                                )
                                state["h"][destination] = h.detach().cpu().numpy().astype(np.float16)
                                maps["preactivation"][destination] = z.detach().cpu().numpy().astype(np.float16)
                                maps["rate"][destination] = rate.detach().cpu().numpy().astype(np.float16)
                                maps["gain"][destination] = components["gain"].cpu().numpy().astype(np.float16)
                                for name in ("ssi", "expected_spikes", "mean_rate"):
                                    value = components[name].cpu().numpy().astype(np.float32)
                                    maps[name][destination] = value
                                    if name == "ssi":
                                        frame_ssi[scale_position, frame_start:frame_stop] = value
                                    elif name == "expected_spikes":
                                        frame_expected[scale_position, frame_start:frame_stop] = value
                                state.flush()
                                if (
                                    float(state.attrs["total_accelerator_seconds_consumed"])
                                    >= args.gpu_budget_hours * 3600.0
                                ):
                                    raise AcceleratorBudgetReached(
                                        "GPU budget reached during an incomplete pair; its completion bit remains false"
                                    )
                                del x, recurrent, h, z, rate, direct_z, components

                    source_values = _source_validation(
                        frame_ssi,
                        frame_expected,
                        authoritative["ssi"][image_position, trajectory_position],
                        authoritative["expected_spikes"][image_position, trajectory_position],
                    )
                    pair_validation[-3:] = source_values
                    _check_validation(pair_validation, args, int(image_id), int(trajectory_id))
                    state["float16_validation"][image_position, trajectory_position] = pair_validation
                    state["pair_elapsed_seconds"][image_position, trajectory_position] = (
                        time.perf_counter() - pair_started
                    )
                    state["pair_accelerator_seconds"][image_position, trajectory_position] = pair_accelerator
                    state.flush()
                    maps.flush()
                    state["completed_pairs"][image_position, trajectory_position] = True
                    maps["completed_pairs"][image_position, trajectory_position] = True
                    state.flush()
                    maps.flush()
                    completed[image_position, trajectory_position] = True
                    generated_this_call += 1
                    write_json(
                        MANIFEST,
                        _manifest_payload(
                            status="running",
                            args=args,
                            completed=completed,
                            state=state,
                            started_utc=started_utc,
                            model_provenance=scorer.provenance,
                        ),
                    )
                    print(
                        f"cached pair {int(np.count_nonzero(completed))}/{completed.size}: "
                        f"image={int(image_id)} trajectory={int(trajectory_id)} "
                        f"accelerator={pair_accelerator:.2f}s",
                        flush=True,
                    )
                    del stims, movies
        except AcceleratorBudgetReached as error:
            write_json(
                MANIFEST,
                _manifest_payload(
                    status="gpu_budget_reached",
                    args=args,
                    completed=completed,
                    state=state,
                    started_utc=started_utc,
                    model_provenance=scorer.provenance,
                    message=str(error),
                ),
            )
            return 2
        except Exception as error:
            write_json(
                MANIFEST,
                _manifest_payload(
                    status="failed",
                    args=args,
                    completed=completed,
                    state=state,
                    started_utc=started_utc,
                    model_provenance=scorer.provenance,
                    message=f"{type(error).__name__}: {error}",
                ),
            )
            raise

        state.attrs["complete"] = bool(completed.all())
        maps.attrs["complete"] = bool(completed.all())
        state.flush()
        maps.flush()
        write_json(
            MANIFEST,
            _manifest_payload(
                status="complete",
                args=args,
                completed=completed,
                state=state,
                started_utc=started_utc,
                model_provenance=scorer.provenance,
            ),
        )
    return 0


def main() -> int:
    args = parse_args()
    try:
        with exclusive_generation_lock():
            return run(args)
    except BlockingIOError as error:
        raise RuntimeError(f"Another cache generator holds {LOCK_PATH}") from error


if __name__ == "__main__":
    raise SystemExit(main())
