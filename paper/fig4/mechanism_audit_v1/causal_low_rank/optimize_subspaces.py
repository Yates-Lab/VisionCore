#!/usr/bin/env python3
"""Fit shared causal ConvGRU channel subspaces on training/validation identities only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    ANALYSIS_SEED,
    CONFIG,
    CONTRASTS,
    FITS,
    MAP_CACHE,
    N_CHANNELS,
    N_FRAMES,
    READOUT_CACHE,
    SCALES,
    STATE_CACHE,
    OUT,
    ensure_output_dirs,
    exclusive_gpu_analysis_lock,
    normalize_rate,
    intervention_preactivations,
    load_global_gpu_budget,
    projector,
    qr_basis,
    record_global_gpu_time,
    scale_index,
    softplus_rate,
    write_json,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold, load_readout, target_units


SCREENING_RANKS = (1, 2, 4, 8, 16, 32)


class GlobalGPUBudgetReached(RuntimeError):
    """Raised between minibatches before the shared four-hour cap is exceeded."""


@dataclass
class Denominators:
    sufficiency: float
    necessity: float
    n_records: int


_ENDPOINT_DENOMINATOR_CACHE: dict[tuple, Denominators] = {}


@dataclass
class ResidentTrainingData:
    """One complete training split held in CPU memory as exact float32 arrays."""

    pairs: tuple[tuple[int, int], ...]
    h_a: np.ndarray
    h_b: np.ndarray
    gain_a: np.ndarray
    gain_b: np.ndarray
    expected_spikes_a: np.ndarray
    expected_spikes_b: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for value in (
                    self.h_a,
                    self.h_b,
                    self.gain_a,
                    self.gain_b,
                    self.expected_spikes_a,
                    self.expected_spikes_b,
                )
            )
        )


_TRAIN_RESIDENT_CACHE: dict[tuple, ResidentTrainingData] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screening", "crossval", "final", "shuffle"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contrasts", nargs="*", default=[value.key for value in CONTRASTS])
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--ranks", nargs="*", type=int)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--patience-evaluations", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_configuration() -> dict:
    path = CONFIG / "optimization_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare_analysis.py first: {path}")
    return json.loads(path.read_text())


def fit_dir(stage: str, contrast: str, fold: int, rank: int) -> Path:
    return FITS / stage / contrast / f"fold_{fold}" / f"rank_{rank:03d}"


def _resolved_optimizer_settings(config: dict, args: argparse.Namespace) -> dict:
    settings = dict(config["optimizer"])
    settings["max_steps"] = settings.pop("maximum_steps")
    settings["validation_interval"] = settings.pop("validation_every_steps")
    settings["patience_evaluations"] = settings.pop(
        "early_stopping_patience_validation_evaluations"
    )
    settings["gradient_clip"] = settings.pop("gradient_clip_norm")
    settings["minimum_improvement"] = settings.pop("minimum_validation_improvement")
    for key, value in (
        ("max_steps", args.max_steps),
        ("frame_batch_size", args.frame_batch_size),
        ("validation_interval", args.validation_interval),
        ("patience_evaluations", args.patience_evaluations),
        ("learning_rate", args.learning_rate),
    ):
        if value is not None:
            settings[key] = value
    return settings


def _fit_is_complete(destination: Path) -> bool:
    return all(
        (destination / name).is_file()
        for name in ("U.npy", "P.npy", "training_curve.csv", "fit_metadata.json")
    )


def _reuse_screening_fit_for_crossval(
    *,
    stage: str,
    contrast_key: str,
    fold_index: int,
    rank: int,
    config: dict,
    args: argparse.Namespace,
) -> dict | None:
    """Materialize the identical screening-fold fit as a crossval product.

    Fold 0 uses the same training and validation identities, objective,
    initialization seeds, and optimizer configuration in both stages.  Reuse
    is allowed only after checking every one of those equivalences and that
    the source fit postdates the currently passing integrity gate.
    """
    screening_fold = int(config["rank_sweep"]["screening_fold"])
    if stage != "crossval" or fold_index != screening_fold or args.overwrite:
        return None
    source = fit_dir("screening", contrast_key, fold_index, rank)
    if not _fit_is_complete(source):
        return None
    integrity_path = CONFIG.parent / "integrity_tests.json"
    if not integrity_path.is_file():
        return None
    source_files = [
        source / name
        for name in ("U.npy", "P.npy", "training_curve.csv", "fit_metadata.json")
    ]
    if min(path.stat().st_mtime_ns for path in source_files) < integrity_path.stat().st_mtime_ns:
        return None
    try:
        metadata = json.loads((source / "fit_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        metadata.get("stage") != "screening"
        or metadata.get("contrast") != contrast_key
        or int(metadata.get("fold", -1)) != fold_index
        or int(metadata.get("rank", -1)) != rank
        or metadata.get("settings") != _resolved_optimizer_settings(config, args)
    ):
        return None
    u = np.asarray(np.load(source / "U.npy"), dtype=np.float32)
    p = np.asarray(np.load(source / "P.npy"), dtype=np.float32)
    if u.shape != (N_CHANNELS, rank) or p.shape != (N_CHANNELS, N_CHANNELS):
        return None
    destination = fit_dir("crossval", contrast_key, fold_index, rank)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "P.npy", destination / "P.npy")
    shutil.copy2(source / "training_curve.csv", destination / "training_curve.csv")
    reused_metadata = dict(metadata)
    reused_metadata.update(
        {
            "stage": "crossval",
            "reused_from_screening": True,
            "reused_source": str(source),
            "reuse_reason": (
                "same predeclared fold, train/validation identities, objective, "
                "optimizer settings, and deterministic restart seeds"
            ),
            "reuse_source_postdates_integrity_gate": True,
        }
    )
    write_json(destination / "fit_metadata.json", reused_metadata)
    # U.npy is the last artifact written so incomplete reuse is never mistaken
    # for a complete fit on resume.
    shutil.copy2(source / "U.npy", destination / "U.npy")
    return {
        "status": "reused_screening",
        "path": str(destination),
        "source": str(source),
    }


def _pair_arrays(dataset: h5py.Dataset, pairs: list[tuple[int, int]], scale_i: int, frames: np.ndarray) -> np.ndarray:
    chunks = []
    frame_ids = np.sort(np.asarray(frames, dtype=np.int64))
    for image_i, trajectory_i in pairs:
        chunks.append(np.asarray(dataset[image_i, trajectory_i, scale_i, frame_ids], dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def batch_from_cache(
    state: h5py.File,
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    frames: np.ndarray,
    units: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    a_i, b_i = scale_index(scale_a), scale_index(scale_b)
    result: dict[str, torch.Tensor] = {}
    for name, source, index in (
        ("h_a", state["h"], a_i),
        ("h_b", state["h"], b_i),
    ):
        result[name] = torch.as_tensor(_pair_arrays(source, pairs, index, frames), device=device)
    # Optimization depends only on endpoint normalized maps and spike weights.
    # Pre-softplus endpoint maps are needed later by the evaluator, but loading
    # them into every training/validation batch wastes HDF5 bandwidth and (for
    # resident validation) hundreds of MB of accelerator memory.
    for name in ("gain", "expected_spikes"):
        for suffix, index in (("a", a_i), ("b", b_i)):
            raw = _pair_arrays(maps[name], pairs, index, frames)
            raw = raw[:, units]
            result[f"{name}_{suffix}"] = torch.as_tensor(raw, device=device)
    return result


def _hdf5_fingerprint(handle: h5py.File, datasets: tuple[str, ...]) -> tuple:
    """Fingerprint immutable cache content sufficiently for process-local reuse."""
    path = Path(str(handle.filename)).resolve()
    stat = path.stat()
    return (
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        tuple(
            (name, tuple(int(value) for value in handle[name].shape), str(handle[name].dtype))
            for name in datasets
        ),
    )


def _training_resident_cache_key(
    state: h5py.File,
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    units: np.ndarray,
) -> tuple:
    """Key a resident split by both source caches and its complete data slice."""
    return (
        "fig4-training-resident-v1",
        _hdf5_fingerprint(state, ("h",)),
        _hdf5_fingerprint(maps, ("gain", "expected_spikes")),
        tuple((int(image), int(trajectory)) for image, trajectory in pairs),
        scale_index(scale_a),
        scale_index(scale_b),
        tuple(int(value) for value in np.asarray(units, dtype=np.int64)),
    )


def load_training_resident(
    state: h5py.File,
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    units: np.ndarray,
    deadline: float | None = None,
) -> ResidentTrainingData:
    """Load a complete training slice once and reuse it across ranks/restarts.

    Source cache values are converted exactly as in :func:`batch_from_cache`:
    the HDF5 values are first materialized as NumPy float32 and unit selection
    is then applied to the endpoint maps.  Keeping pair and frame axes explicit
    lets every optimization step reproduce the original one-pair/eight-frame
    sample without another HDF5 access.
    """
    key = _training_resident_cache_key(
        state, maps, pairs, scale_a, scale_b, units
    )
    cached = _TRAIN_RESIDENT_CACHE.get(key)
    if cached is not None:
        return cached

    pair_tuple = tuple((int(image), int(trajectory)) for image, trajectory in pairs)
    unit_ids = np.asarray(units, dtype=np.int64)
    a_i, b_i = scale_index(scale_a), scale_index(scale_b)
    n_pairs = len(pair_tuple)
    state_tail = tuple(int(value) for value in state["h"].shape[-3:])
    map_tail = tuple(int(value) for value in maps["gain"].shape[-2:])
    arrays = {
        "h_a": np.empty((n_pairs, N_FRAMES, *state_tail), dtype=np.float32),
        "h_b": np.empty((n_pairs, N_FRAMES, *state_tail), dtype=np.float32),
        "gain_a": np.empty((n_pairs, N_FRAMES, len(unit_ids), *map_tail), dtype=np.float32),
        "gain_b": np.empty((n_pairs, N_FRAMES, len(unit_ids), *map_tail), dtype=np.float32),
        "expected_spikes_a": np.empty((n_pairs, N_FRAMES, len(unit_ids)), dtype=np.float32),
        "expected_spikes_b": np.empty((n_pairs, N_FRAMES, len(unit_ids)), dtype=np.float32),
    }
    for pair_i, (image_i, trajectory_i) in enumerate(pair_tuple):
        if deadline is not None and time.monotonic() >= deadline:
            raise GlobalGPUBudgetReached(
                "Global GPU budget reached while loading the resident training split"
            )
        arrays["h_a"][pair_i] = np.asarray(
            state["h"][image_i, trajectory_i, a_i], dtype=np.float32
        )
        arrays["h_b"][pair_i] = np.asarray(
            state["h"][image_i, trajectory_i, b_i], dtype=np.float32
        )
        for name in ("gain", "expected_spikes"):
            for suffix, scale_i in (("a", a_i), ("b", b_i)):
                raw = np.asarray(
                    maps[name][image_i, trajectory_i, scale_i], dtype=np.float32
                )
                arrays[f"{name}_{suffix}"][pair_i] = raw[:, unit_ids]

    resident = ResidentTrainingData(pairs=pair_tuple, **arrays)
    _TRAIN_RESIDENT_CACHE[key] = resident
    return resident


def batch_from_training_resident(
    resident: ResidentTrainingData,
    pair_index: int,
    frames: np.ndarray,
    device: str,
) -> dict[str, torch.Tensor]:
    """Materialize the original selected pair/frames from CPU-resident data."""
    pair_i = int(pair_index)
    if pair_i < 0 or pair_i >= len(resident.pairs):
        raise IndexError(f"Training pair index {pair_i} is outside [0, {len(resident.pairs)})")
    frame_ids = np.sort(np.asarray(frames, dtype=np.int64))
    if np.any(frame_ids < 0) or np.any(frame_ids >= N_FRAMES):
        raise IndexError(f"Training frame ids must be inside [0, {N_FRAMES})")
    return {
        name: torch.as_tensor(getattr(resident, name)[pair_i, frame_ids], device=device)
        for name in (
            "h_a",
            "h_b",
            "gain_a",
            "gain_b",
            "expected_spikes_a",
            "expected_spikes_b",
        )
    }


def select_training_minibatch_indices(
    rng: np.random.Generator,
    n_pairs: int,
    frame_batch: int,
) -> tuple[int, np.ndarray]:
    """Draw exactly one training pair followed by sorted frames."""
    if int(n_pairs) <= 0:
        raise ValueError("At least one training pair is required")
    pair_index = int(rng.integers(int(n_pairs)))
    frames = np.sort(
        rng.choice(N_FRAMES, size=min(int(frame_batch), N_FRAMES), replace=False)
    )
    return pair_index, frames


def _endpoint_denominator_cache_key(
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    units: np.ndarray,
) -> tuple:
    path = Path(str(maps.filename)).resolve()
    stat = path.stat()
    return (
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        tuple((int(image), int(trajectory)) for image, trajectory in pairs),
        scale_index(scale_a),
        scale_index(scale_b),
        tuple(np.asarray(units, dtype=np.int64).tolist()),
    )


def _compute_endpoint_denominators(
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    units: np.ndarray,
    deadline: float | None = None,
) -> Denominators:
    a_i, b_i = scale_index(scale_a), scale_index(scale_b)
    total = 0.0
    for image_i, trajectory_i in pairs:
        if deadline is not None and time.monotonic() >= deadline:
            raise GlobalGPUBudgetReached(
                "Global GPU budget reached while computing endpoint denominators"
            )
        g_a = np.asarray(maps["gain"][image_i, trajectory_i, a_i, :, units], dtype=np.float64)
        g_b = np.asarray(maps["gain"][image_i, trajectory_i, b_i, :, units], dtype=np.float64)
        e_a = np.asarray(maps["expected_spikes"][image_i, trajectory_i, a_i, :, units], dtype=np.float64)
        e_b = np.asarray(maps["expected_spikes"][image_i, trajectory_i, b_i, :, units], dtype=np.float64)
        weight = 0.5 * (e_a + e_b)
        total += float(np.sum((g_b - g_a) ** 2 * weight[..., None, None]))
    return Denominators(total, total, len(pairs) * N_FRAMES)


def endpoint_denominators(
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    units: np.ndarray,
    deadline: float | None = None,
) -> Denominators:
    """Return the exact global map-effect denominator with safe memoization.

    The key includes the immutable cache file fingerprint and the complete
    analysis slice, so ranks/restarts can reuse the expensive HDF5 reduction
    without allowing one fold, contrast, or population to contaminate another.
    """
    key = _endpoint_denominator_cache_key(
        maps, pairs, scale_a, scale_b, units
    )
    if key not in _ENDPOINT_DENOMINATOR_CACHE:
        _ENDPOINT_DENOMINATOR_CACHE[key] = _compute_endpoint_denominators(
            maps, pairs, scale_a, scale_b, units, deadline
        )
    return _ENDPOINT_DENOMINATOR_CACHE[key]


def movement_pca_cache_path(contrast_key: str, fold_index: int) -> Path:
    """Canonical training-only PCA cache shared by fits and evaluation."""
    return (
        FITS
        / "baselines"
        / str(contrast_key)
        / f"fold_{int(fold_index)}"
        / "movement_pca_full.npy"
    )


def _movement_pca_signature(
    state: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
) -> dict:
    state_path = Path(str(state.filename)).resolve()
    stat = state_path.stat()
    return {
        "schema_version": "fig4-movement-pca-cache-v1",
        "state_cache": str(state_path),
        "state_cache_size": int(stat.st_size),
        "state_cache_mtime_ns": int(stat.st_mtime_ns),
        "pairs": [[int(image), int(trajectory)] for image, trajectory in pairs],
        "scale_a": float(scale_a),
        "scale_b": float(scale_b),
        "channels": N_CHANNELS,
        "definition": "eigenvectors of uncentered training delta-h channel second moment",
    }


def _valid_full_pca(value: np.ndarray) -> bool:
    if value.shape != (N_CHANNELS, N_CHANNELS) or not np.isfinite(value).all():
        return False
    gram = np.asarray(value, dtype=np.float64).T @ np.asarray(value, dtype=np.float64)
    return bool(np.allclose(gram, np.eye(N_CHANNELS), atol=2e-4, rtol=2e-4))


def load_or_compute_movement_pca(
    state: h5py.File,
    pairs: list[tuple[int, int]],
    contrast_key: str,
    fold_index: int,
    scale_a: float,
    scale_b: float,
    device: str,
    deadline: float | None = None,
) -> torch.Tensor:
    """Load one provenance-checked full PCA basis or compute it once."""
    destination = movement_pca_cache_path(contrast_key, fold_index)
    metadata_path = destination.with_suffix(".json")
    signature = _movement_pca_signature(state, pairs, scale_a, scale_b)
    value: np.ndarray | None = None
    adopted_legacy = False
    if destination.is_file():
        candidate = np.asarray(np.load(destination), dtype=np.float32)
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if metadata.get("signature") == signature and _valid_full_pca(candidate):
                value = candidate
        else:
            # The first optimized fit created this cache before the sidecar was
            # introduced.  It can be adopted only if it is a valid full basis
            # produced after the currently passing intervention gate.
            integrity_path = CONFIG.parent / "integrity_tests.json"
            if (
                integrity_path.is_file()
                and destination.stat().st_mtime_ns >= integrity_path.stat().st_mtime_ns
                and _valid_full_pca(candidate)
            ):
                value = candidate
                adopted_legacy = True
    if value is None:
        computed = movement_pca_basis(
            state,
            pairs,
            scale_a,
            scale_b,
            N_CHANNELS,
            device,
            deadline,
        )
        value = computed.detach().cpu().numpy().astype(np.float32)
        if not _valid_full_pca(value):
            raise RuntimeError("Computed movement-PCA basis is not finite and orthonormal")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npy")
        np.save(temporary, value)
        os.replace(temporary, destination)
    write_json(
        metadata_path,
        {
            "signature": signature,
            "basis_path": destination,
            "adopted_legacy_without_prior_sidecar": adopted_legacy,
        },
    )
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def loss_for_basis(
    batch: dict[str, torch.Tensor],
    basis: torch.Tensor,
    readout: dict[str, torch.Tensor],
    denominators: Denominators,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, sufficiency, and necessity global-loss estimates."""
    scale = float(denominators.n_records) / float(len(batch["h_a"]))
    weight = 0.5 * (batch["expected_spikes_a"] + batch["expected_spikes_b"])
    z_s, z_n = intervention_preactivations(
        batch["h_a"],
        batch["h_b"],
        basis,
        readout["feature"],
        readout["bias"],
        readout["space"],
    )
    gain_s = normalize_rate(softplus_rate(z_s))
    gain_n = normalize_rate(softplus_rate(z_n))
    num_s = ((gain_s - batch["gain_b"]).square() * weight[..., None, None]).sum()
    num_n = ((gain_n - batch["gain_a"]).square() * weight[..., None, None]).sum()
    sufficiency = scale * num_s / max(denominators.sufficiency, 1e-30)
    necessity = scale * num_n / max(denominators.necessity, 1e-30)
    return 0.5 * (sufficiency + necessity), sufficiency, necessity


@torch.no_grad()
def evaluate_validation_resident(
    batch: dict[str, torch.Tensor],
    bases: torch.Tensor,
    readout: dict[str, torch.Tensor],
    denominators: Denominators,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate every restart on one exact, GPU-resident validation split.

    The six validation image--trajectory pairs are fixed for a fold and fit
    comfortably in accelerator memory.  Loading them once removes repeated
    HDF5 decompression at every early-stopping check without changing a single
    sample, weight, denominator, or readout operation.
    """
    total = np.empty(len(bases), dtype=np.float64)
    sufficiency = np.empty(len(bases), dtype=np.float64)
    necessity = np.empty(len(bases), dtype=np.float64)
    for restart, basis in enumerate(bases):
        value, value_s, value_n = loss_for_basis(batch, basis, readout, denominators)
        total[restart] = float(value.cpu())
        sufficiency[restart] = float(value_s.cpu())
        necessity[restart] = float(value_n.cpu())
    return total, sufficiency, necessity


@torch.no_grad()
def evaluate_validation(
    state: h5py.File,
    maps: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    units: np.ndarray,
    bases: torch.Tensor,
    readout: dict[str, torch.Tensor],
    device: str,
    frame_batch: int,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominators = endpoint_denominators(maps, pairs, scale_a, scale_b, units)
    numerator_s = np.zeros(len(bases), dtype=np.float64)
    numerator_n = np.zeros(len(bases), dtype=np.float64)
    for pair in pairs:
        if deadline is not None and time.monotonic() >= deadline:
            raise GlobalGPUBudgetReached("Global authorized GPU budget reached during validation")
        for start in range(0, N_FRAMES, frame_batch):
            frames = np.arange(start, min(start + frame_batch, N_FRAMES))
            batch = batch_from_cache(state, maps, [pair], scale_a, scale_b, frames, units, device)
            weight = 0.5 * (batch["expected_spikes_a"] + batch["expected_spikes_b"])
            for restart, basis in enumerate(bases):
                z_s, z_n = intervention_preactivations(
                    batch["h_a"],
                    batch["h_b"],
                    basis,
                    readout["feature"],
                    readout["bias"],
                    readout["space"],
                )
                gain_s = normalize_rate(softplus_rate(z_s))
                gain_n = normalize_rate(softplus_rate(z_n))
                numerator_s[restart] += float(
                    ((gain_s - batch["gain_b"]).square() * weight[..., None, None]).sum().cpu()
                )
                numerator_n[restart] += float(
                    ((gain_n - batch["gain_a"]).square() * weight[..., None, None]).sum().cpu()
                )
            del batch, weight
    loss_s = numerator_s / max(denominators.sufficiency, 1e-30)
    loss_n = numerator_n / max(denominators.necessity, 1e-30)
    return 0.5 * (loss_s + loss_n), loss_s, loss_n


@torch.no_grad()
def movement_pca_basis(
    state: h5py.File,
    pairs: list[tuple[int, int]],
    scale_a: float,
    scale_b: float,
    rank: int,
    device: str,
    deadline: float | None = None,
) -> torch.Tensor:
    covariance = torch.zeros((N_CHANNELS, N_CHANNELS), dtype=torch.float64, device=device)
    a_i, b_i = scale_index(scale_a), scale_index(scale_b)
    n_observations = 0
    for image_i, trajectory_i in pairs:
        if deadline is not None and time.monotonic() >= deadline:
            raise GlobalGPUBudgetReached(
                "Global GPU budget reached while computing movement PCA"
            )
        h_a = torch.as_tensor(np.asarray(state["h"][image_i, trajectory_i, a_i], dtype=np.float32), device=device)
        h_b = torch.as_tensor(np.asarray(state["h"][image_i, trajectory_i, b_i], dtype=np.float32), device=device)
        delta = (h_b - h_a).permute(1, 0, 2, 3).reshape(N_CHANNELS, -1).double()
        covariance += delta @ delta.T
        n_observations += delta.shape[1]
    covariance /= max(n_observations, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors[:, -int(rank) :].flip(1).float()


def initial_parameters(pca: torch.Tensor, rank: int, fold: int, contrast_ordinal: int, device: str) -> torch.Tensor:
    values = []
    for restart in range(3):
        if restart == 1:
            values.append(pca)
            continue
        seed = (
            ANALYSIS_SEED
            + 100_000 * contrast_ordinal
            + 10_000 * int(fold)
            + 100 * int(rank)
            + restart
            + 1
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        random = torch.randn((N_CHANNELS, int(rank)), generator=generator, device=device)
        values.append(qr_basis(random))
    return torch.stack(values)


def train_one(
    *,
    stage: str,
    contrast,
    contrast_ordinal: int,
    fold_index: int,
    rank: int,
    config: dict,
    args: argparse.Namespace,
) -> dict:
    destination = fit_dir(stage, contrast.key, fold_index, rank)
    if _fit_is_complete(destination) and not args.overwrite:
        return {"status": "exists", "path": str(destination)}
    if args.dry_run:
        return {"status": "dry_run", "path": str(destination)}
    reused = _reuse_screening_fit_for_crossval(
        stage=stage,
        contrast_key=contrast.key,
        fold_index=fold_index,
        rank=rank,
        config=config,
        args=args,
    )
    if reused is not None:
        return reused
    destination.mkdir(parents=True, exist_ok=True)
    device = str(args.device)
    units = target_units(contrast.group)
    fold = load_fold(fold_index)
    train_pairs = fold.train.pairs
    validation_pairs = fold.validation.pairs
    settings = _resolved_optimizer_settings(config, args)
    max_steps = int(settings["max_steps"])
    frame_batch = int(settings["frame_batch_size"])
    validation_frame_batch = int(settings["validation_frame_batch_size"])
    validation_interval = int(settings["validation_interval"])
    patience = int(settings["patience_evaluations"])
    rng = np.random.default_rng(ANALYSIS_SEED + fold_index * 10_000 + contrast_ordinal * 1_000 + rank)
    readout = load_readout(device, units)
    curve_rows: list[dict] = []
    started = time.time()
    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        deadline = getattr(args, "_gpu_deadline_monotonic", None)
        denominators = endpoint_denominators(
            maps,
            train_pairs,
            contrast.scale_a,
            contrast.scale_b,
            units,
            deadline,
        )
        pca_full = load_or_compute_movement_pca(
            state,
            train_pairs,
            contrast.key,
            fold_index,
            contrast.scale_a,
            contrast.scale_b,
            device,
            deadline,
        )
        pca = pca_full[:, :rank]
        validation_denominators = endpoint_denominators(
            maps,
            validation_pairs,
            contrast.scale_a,
            contrast.scale_b,
            units,
            deadline,
        )
        validation_batch = batch_from_cache(
            state,
            maps,
            validation_pairs,
            contrast.scale_a,
            contrast.scale_b,
            np.arange(N_FRAMES),
            units,
            device,
        )
        print(
            f"loading CPU-resident training split ({len(train_pairs)} pairs) for "
            f"{contrast.key} fold={fold_index}",
            flush=True,
        )
        training_resident = load_training_resident(
            state,
            maps,
            train_pairs,
            contrast.scale_a,
            contrast.scale_b,
            units,
            deadline,
        )
        print(
            f"resident training split ready ({training_resident.nbytes / 2**30:.2f} GiB)",
            flush=True,
        )
        initial = initial_parameters(pca, rank, fold_index, contrast_ordinal, device)
        parameters = torch.nn.ParameterList(
            [torch.nn.Parameter(initial[restart].clone()) for restart in range(3)]
        )
        optimizers = [
            torch.optim.AdamW(
                [parameters[restart]],
                lr=float(settings["learning_rate"]),
                weight_decay=float(settings.get("weight_decay", 0.0)),
            )
            for restart in range(3)
        ]
        best_loss = np.full(3, np.inf)
        best_step = np.zeros(3, dtype=int)
        best_basis = np.zeros((3, N_CHANNELS, rank), dtype=np.float32)
        stale = np.zeros(3, dtype=int)
        converged = np.zeros(3, dtype=bool)
        last_train_values = [(torch.tensor(float("nan")),) * 3 for _ in range(3)]
        last_gradient_norms = [float("nan")] * 3
        for step in range(max_steps):
            deadline = getattr(args, "_gpu_deadline_monotonic", None)
            if deadline is not None and time.monotonic() >= deadline:
                raise GlobalGPUBudgetReached(
                    "Global authorized GPU budget reached between optimization minibatches"
                )
            pair_index, frames = select_training_minibatch_indices(
                rng, len(train_pairs), frame_batch
            )
            batch = batch_from_training_resident(
                training_resident, pair_index, frames, device
            )
            train_values = list(last_train_values)
            gradient_norms = list(last_gradient_norms)
            for restart in range(3):
                if converged[restart]:
                    continue
                optimizers[restart].zero_grad(set_to_none=True)
                basis = qr_basis(parameters[restart])
                loss, loss_s, loss_n = loss_for_basis(batch, basis, readout, denominators)
                loss.backward()
                gradient_norms[restart] = float(
                    torch.nn.utils.clip_grad_norm_([parameters[restart]], float(settings["gradient_clip"]))
                )
                optimizers[restart].step()
                train_values[restart] = (loss.detach(), loss_s.detach(), loss_n.detach())
            last_train_values = train_values
            last_gradient_norms = gradient_norms
            should_validate = step == 0 or (step + 1) % validation_interval == 0 or step + 1 == max_steps
            if should_validate:
                with torch.no_grad():
                    bases = torch.stack([qr_basis(value) for value in parameters])
                val, val_s, val_n = evaluate_validation_resident(
                    validation_batch,
                    bases,
                    readout,
                    validation_denominators,
                )
                for restart in range(3):
                    improved = bool(val[restart] < best_loss[restart] - float(settings["minimum_improvement"]))
                    if improved:
                        best_loss[restart] = val[restart]
                        best_step[restart] = step + 1
                        best_basis[restart] = bases[restart].detach().cpu().numpy()
                        stale[restart] = 0
                    else:
                        stale[restart] += 1
                    converged[restart] = stale[restart] >= patience
                    curve_rows.append(
                        {
                            "step": step + 1,
                            "restart": restart,
                            "initialization": "movement_pca" if restart == 1 else f"random_{restart}",
                            "train_total_loss_estimate": float(train_values[restart][0].cpu()),
                            "train_sufficiency_loss_estimate": float(train_values[restart][1].cpu()),
                            "train_necessity_loss_estimate": float(train_values[restart][2].cpu()),
                            "validation_total_loss": float(val[restart]),
                            "validation_sufficiency_loss": float(val_s[restart]),
                            "validation_necessity_loss": float(val_n[restart]),
                            "gradient_norm_before_clip": gradient_norms[restart],
                            "best_validation_loss": float(best_loss[restart]),
                            "stale_validation_evaluations": int(stale[restart]),
                            "converged": bool(converged[restart]),
                        }
                    )
                if np.all(converged):
                    break
        selected = int(np.argmin(best_loss))
    u = best_basis[selected]
    np.save(destination / "U.npy", u)
    np.save(destination / "P.npy", projector(u))
    with (destination / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    metadata = {
        "stage": stage,
        "contrast": contrast.key,
        "fold": fold_index,
        "rank": rank,
        "target_group": contrast.group,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "validation_execution": (
            "fixed complete validation split loaded once on-device; exact samples, weights, "
            "global denominator, literal state intervention, and frozen tiled readout unchanged"
        ),
        "training_execution": (
            "complete training split loaded once as CPU-resident float32 arrays and reused "
            "across ranks in this process; each step retains the original seeded draw of one "
            "pair followed by sorted without-replacement frames"
        ),
        "settings": settings,
        "restart_seeds": [
            ANALYSIS_SEED
            + 100_000 * contrast_ordinal
            + 10_000 * fold_index
            + 100 * rank
            + restart
            + 1
            for restart in range(3)
        ],
        "selected_restart": selected,
        "selected_initialization": "movement_pca" if selected == 1 else f"random_{selected}",
        "best_validation_loss": float(best_loss[selected]),
        "best_step": int(best_step[selected]),
        "all_best_validation_losses": best_loss,
        "all_best_steps": best_step,
        "elapsed_seconds": time.time() - started,
        "objective": "equal average of global paired-expected-spike-weighted normalized-map sufficiency and necessity MSE ratios",
    }
    write_json(destination / "fit_metadata.json", metadata)
    return metadata


def _run(args: argparse.Namespace) -> int:
    ensure_output_dirs()
    integrity_path = CONFIG.parent / "integrity_tests.json"
    if not integrity_path.exists():
        raise RuntimeError("Run validate_interventions.py successfully before optimization")
    integrity = json.loads(integrity_path.read_text())
    if not bool(integrity.get("optimization_allowed", False)):
        raise RuntimeError(f"Mandatory intervention integrity gate failed: {integrity_path}")
    config = load_configuration()
    hard_limit_hours = float(config["gpu_budget"]["hard_limit_hours"])
    contrast_lookup = {value.key: (ordinal, value) for ordinal, value in enumerate(CONTRASTS)}
    unknown = sorted(set(args.contrasts) - set(contrast_lookup))
    if unknown:
        raise ValueError(f"Unknown contrasts: {unknown}")
    if args.folds is not None:
        folds = args.folds
    elif args.stage == "screening":
        folds = [int(config["rank_sweep"]["screening_fold"])]
    elif args.stage == "final":
        folds = [0]
    else:
        folds = [0, 1, 2, 3]
    if args.ranks is not None:
        ranks = args.ranks
    elif args.stage == "screening":
        ranks = list(config["rank_sweep"]["screening_ranks"])
    else:
        stage2 = CONFIG / "stage2_ranks.json"
        if not stage2.exists():
            raise FileNotFoundError("Run evaluate_subspaces.py --stage screening to predeclare Stage 2 ranks")
        ranks = json.loads(stage2.read_text())["ranks"]
    summaries = []
    for key in args.contrasts:
        ordinal, contrast = contrast_lookup[key]
        for fold in folds:
            for rank in ranks:
                if int(rank) <= 0 or int(rank) >= N_CHANNELS:
                    continue
                print(f"fit {args.stage} {key} fold={fold} rank={rank}", flush=True)
                budget = load_global_gpu_budget(hard_limit_hours)
                consumed = float(budget["total_conservative_gpu_hours"])
                if consumed >= hard_limit_hours:
                    raise RuntimeError(
                        f"Authorized analysis GPU budget reached ({consumed:.3f} h); stopping before another fit"
                    )
                fit_started = time.monotonic()
                args._gpu_deadline_monotonic = fit_started + (hard_limit_hours - consumed) * 3600.0
                summary = None
                try:
                    summary = train_one(
                        stage=args.stage,
                        contrast=contrast,
                        contrast_ordinal=ordinal,
                        fold_index=int(fold),
                        rank=int(rank),
                        config=config,
                        args=args,
                    )
                finally:
                    elapsed_seconds = time.monotonic() - fit_started
                    if summary is None or summary.get("status") not in (
                        "exists",
                        "dry_run",
                        "reused_screening",
                    ):
                        record_global_gpu_time(
                            f"optimization:{args.stage}",
                            elapsed_seconds,
                            hard_limit_hours=hard_limit_hours,
                            details={
                                "contrast": key,
                                "fold": int(fold),
                                "rank": int(rank),
                                "completed": summary is not None,
                            },
                        )
                assert summary is not None
                summaries.append(summary)
    write_json(FITS / f"{args.stage}_last_run.json", {"fits": summaries, "completed_at_unix": time.time()})
    return 0


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    with exclusive_gpu_analysis_lock():
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
