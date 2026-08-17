"""Shared paths, provenance checks, and compute accounting for rank-8 validation."""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
CAUSAL_OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1"
OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
RANK8_OUT = OUT / "rank8_validation"
CROSS_TRANSFER_OUT = RANK8_OUT / "cross_transfer"

RANK = 8
FOLDS = (0, 1, 2, 3)
CONTINUATION_FOLDS = (1, 2, 3)
CONTRASTS = ("low_0_to_2", "high_0_to_1", "high_1_to_3")
HIGH_CONTRASTS = ("high_0_to_1", "high_1_to_3")

INTEGRITY = CAUSAL_OUT / "integrity_tests.json"
STAGE2 = CAUSAL_OUT / "config/stage2_ranks.json"
STATE_CACHE = CAUSAL_OUT / "cache/convgru_states.h5"
MAP_CACHE = CAUSAL_OUT / "cache/rr100_maps.h5"
READOUT_CACHE = CAUSAL_OUT / "cache/readout_weights.npz"
SOURCE_GPU_BUDGET = CAUSAL_OUT / "gpu_budget.json"
# The causal-low-rank ledger is the one authoritative cumulative ledger.  The
# overnight entry points may raise its reporting boundary to ten hours, but
# must never maintain a second counter that can diverge from it.
GPU_BUDGET = SOURCE_GPU_BUDGET
GPU_LOCK = CAUSAL_OUT / "gpu_budget.run.lock"
HARD_LIMIT_HOURS = 10.0


def ensure_dirs() -> None:
    for path in (OUT, RANK8_OUT, CROSS_TRANSFER_OUT):
        path.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fit_directory(contrast: str, fold: int) -> Path:
    return (
        CAUSAL_OUT
        / "fits/crossval"
        / str(contrast)
        / f"fold_{int(fold)}"
        / f"rank_{RANK:03d}"
    )


def evaluation_directory(contrast: str, fold: int) -> Path:
    return (
        CAUSAL_OUT
        / "evaluation/results/crossval"
        / str(contrast)
        / f"fold_{int(fold)}"
        / f"rank_{RANK:03d}"
    )


def cross_transfer_path(source: str, target: str, fold: int) -> Path:
    return (
        CROSS_TRANSFER_OUT
        / f"source_{source}__target_{target}__fold_{int(fold)}.json"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read required JSON product: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def validate_fixed_rank_provenance() -> dict[str, Any]:
    """Fail closed unless the prior validation-only screen selected rank eight."""
    if not INTEGRITY.is_file():
        raise FileNotFoundError(INTEGRITY)
    integrity = _read_json(INTEGRITY)
    if not bool(integrity.get("optimization_allowed", False)):
        raise RuntimeError(f"The mandatory intervention gate is not passing: {INTEGRITY}")
    if not STAGE2.is_file():
        raise FileNotFoundError(STAGE2)
    stage2 = _read_json(STAGE2)
    elbow = stage2.get("elbow", {})
    if bool(elbow.get("ambiguous", True)) or int(elbow.get("elbow_rank", -1)) != RANK:
        raise RuntimeError(
            "Rank 8 is not a unique validation-selected elbow in the saved screening product"
        )
    if bool(stage2.get("selection_used_test_metrics", True)):
        raise RuntimeError("The saved rank selection reports using test metrics")
    return {"integrity": integrity, "stage2": stage2}


def validate_exact_cache() -> None:
    """Verify that all exact saved-cache pairs exist without loading their arrays."""
    for path in (STATE_CACHE, MAP_CACHE, READOUT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    with h5py.File(STATE_CACHE, "r") as states, h5py.File(MAP_CACHE, "r") as maps:
        state_done = np.asarray(states["completed_pairs"][:], dtype=bool)
        map_done = np.asarray(maps["completed_pairs"][:], dtype=bool)
        if not state_done.all() or not map_done.all():
            raise RuntimeError(
                "The exact cache is incomplete: "
                f"states={int(state_done.sum())}/{state_done.size}, "
                f"maps={int(map_done.sum())}/{map_done.size}"
            )
        if not np.array_equal(state_done, map_done):
            raise RuntimeError("State and RR100 map completion masks differ")


def validate_basis(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.shape != (128, RANK):
        raise RuntimeError(f"Rank-8 basis has the wrong shape at {path}: {value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"Rank-8 basis contains nonfinite values: {path}")
    if not np.allclose(value.T @ value, np.eye(RANK), atol=2e-4, rtol=2e-4):
        raise RuntimeError(f"Rank-8 basis is not orthonormal: {path}")
    return value.astype(np.float32)


def validate_fold0_reuse() -> None:
    """Require the already materialized screening-fold rank-8 products."""
    for contrast in CONTRASTS:
        directory = fit_directory(contrast, 0)
        validate_basis(directory / "U.npy")
        for name in ("P.npy", "training_curve.csv", "fit_metadata.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        metadata = _read_json(directory / "fit_metadata.json")
        if int(metadata.get("rank", -1)) != RANK or int(metadata.get("fold", -1)) != 0:
            raise RuntimeError(f"Fold-0 metadata does not describe the required rank-8 fit: {directory}")


def preflight() -> dict[str, Any]:
    ensure_dirs()
    provenance = validate_fixed_rank_provenance()
    validate_exact_cache()
    validate_fold0_reuse()
    return provenance


def _source_hours() -> float:
    if not GPU_BUDGET.is_file():
        return 0.0
    source = _read_json(GPU_BUDGET)
    return float(source.get("total_conservative_gpu_hours", 0.0))


def load_budget() -> dict[str, Any]:
    """Load the single authoritative cumulative Figure 4 GPU ledger."""
    ensure_dirs()
    if GPU_BUDGET.is_file():
        value = _read_json(GPU_BUDGET)
    else:
        value = {
            "schema_version": "fig4-global-gpu-budget-v1",
            "hard_limit_hours": HARD_LIMIT_HOURS,
            "cache_accelerator_hours": 0.0,
            "gpu_stage_wall_hours": 0.0,
            "events": [],
        }
    if float(value.get("hard_limit_hours", 0.0)) > HARD_LIMIT_HOURS:
        raise RuntimeError(f"Unexpected rank-8 budget limit in {GPU_BUDGET}")
    value["hard_limit_hours"] = HARD_LIMIT_HOURS
    cache_hours = float(value.get("cache_accelerator_hours", 0.0))
    stage_hours = float(
        value.get("gpu_stage_wall_hours", value.get("optimization_wall_hours", 0.0))
    )
    value["total_conservative_gpu_hours"] = float(
        max(value.get("total_conservative_gpu_hours", 0.0), cache_hours + stage_hours)
    )
    value["remaining_gpu_hours"] = max(
        HARD_LIMIT_HOURS - value["total_conservative_gpu_hours"], 0.0
    )
    value["status"] = (
        "interim_report_required"
        if value["total_conservative_gpu_hours"] >= HARD_LIMIT_HOURS
        else "within_budget"
    )
    return value


def record_gpu_time(stage: str, elapsed_seconds: float, details: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs()
    lock_path = GPU_BUDGET.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        value = load_budget()
        elapsed_hours = max(float(elapsed_seconds), 0.0) / 3600.0
        value["gpu_stage_wall_hours"] = float(
            value.get("gpu_stage_wall_hours", 0.0)
        ) + elapsed_hours
        value["events"].append(
            {
                "stage": str(stage),
                "wall_hours": elapsed_hours,
                "completed_at_unix": time.time(),
                "details": details,
            }
        )
        value["total_conservative_gpu_hours"] = float(
            value.get("cache_accelerator_hours", 0.0)
        ) + float(value["gpu_stage_wall_hours"])
        value["remaining_gpu_hours"] = max(
            HARD_LIMIT_HOURS - value["total_conservative_gpu_hours"], 0.0
        )
        value["status"] = (
            "interim_report_required"
            if value["total_conservative_gpu_hours"] >= HARD_LIMIT_HOURS
            else "within_budget"
        )
        write_json(GPU_BUDGET, value)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return value


def budget_deadline() -> float:
    value = load_budget()
    remaining = float(value["remaining_gpu_hours"])
    if remaining <= 0.0:
        raise RuntimeError(
            "The cumulative ten-GPU-hour limit has been reached; write the required interim report"
        )
    return time.monotonic() + remaining * 3600.0


@contextmanager
def exclusive_gpu_lock() -> Iterator[None]:
    """Share the existing analysis lock so old and overnight jobs cannot overlap."""
    ensure_dirs()
    with GPU_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another Figure 4 GPU stage holds {GPU_LOCK}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
