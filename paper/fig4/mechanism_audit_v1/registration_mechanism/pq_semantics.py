#!/usr/bin/env python3
"""Saved-cache semantic decomposition of rank-8 ConvGRU projectors.

This module implements sections 4--6 of the ConvGRU registration-mechanism
audit.  It deliberately has no model import and no model-forward path.  Every
quantity is computed from the validated ConvGRU-state cache, the frozen tiled
RR100 readout cache, the crossed fold assignments, and each fold's rank-8
basis.

The expensive stages are resumable at the fold/contrast level::

    python -m paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
        --stage variance --folds 0 1 2 3
    python -m paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
        --stage readout --folds 0 1 2 3 --device cuda:0
    python -m paper.fig4.mechanism_audit_v1.registration_mechanism.pq_semantics \
        --stage consolidate

The default device is CPU so importing or smoke-testing this module cannot
consume accelerator budget accidentally.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (  # noqa: E402
    CONTRASTS,
    MAP_CACHE,
    N_CHANNELS,
    N_FRAMES,
    N_UNITS,
    OUT as LOW_RANK_OUT,
    READOUT_CACHE,
    SCALES,
    STATE_SIZE,
    STATE_CACHE,
    project_channel_delta,
    rate_map_components,
    readout_preactivation,
    scale_index,
    sha256_file,
    softplus_rate,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import (  # noqa: E402
    load_fold,
    load_readout,
    target_units,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.common import (  # noqa: E402
    budget_deadline,
    exclusive_gpu_lock,
    load_budget,
    record_gpu_time,
)


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
INTERMEDIATE = OUT / "pq_intermediate"
ARRAYS = OUT / "pq_supporting_arrays"
MANIFEST = OUT / "pq_semantics_manifest.json"

LEVERAGE_CSV = OUT / "native_channel_leverage.csv"
LEVERAGE_OVERLAP_CSV = OUT / "native_channel_leverage_overlap.csv"
LEVERAGE_TOPK_CSV = OUT / "native_channel_leverage_topk_overlap.csv"
VARIANCE_CSV = OUT / "pq_variance_decomposition.csv"
READOUT_CSV = OUT / "pq_readout_decomposition.csv"
PER_UNIT_CSV = OUT / "per_unit_pq_reliance.csv"
PROBES_CSV = OUT / "pq_descriptive_probes.csv"

UNIT_FEATURE_TABLE = (
    ROOT
    / "outputs/active_sensing_movie_information"
    / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
    / "merged/unit_feature_table.csv"
)
TRAJECTORY_BANK = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1/banks"
    / "corrected_history_trajectory_banks.npz"
)
TRAJECTORY_TABLE = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1"
    / "true_history_trajectory_table.csv"
)

SCHEMA_VERSION = "fig4-registration-pq-semantics-v1"
RANK = 8
Q_RANK = N_CHANNELS - RANK
EPS64 = 1e-30
_GPU_DEADLINE_MONOTONIC = math.inf


def _gpu_budget_checkpoint() -> None:
    if time.monotonic() >= _GPU_DEADLINE_MONOTONIC:
        raise RuntimeError(
            "The cumulative ten-GPU-hour limit was reached between saved-cache P/Q batches; "
            "write the required interim report before more accelerator work"
        )


@dataclass
class ScalarMoments:
    """Streaming scalar moments over an arbitrary number of tensor entries."""

    count: int = 0
    total: float = 0.0
    square: float = 0.0

    def add(self, value: np.ndarray | torch.Tensor) -> None:
        array = _numpy(value).astype(np.float64, copy=False)
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.square += float(np.square(array).sum(dtype=np.float64))

    @property
    def mean(self) -> float:
        return _safe_divide(self.total, self.count)

    @property
    def variance(self) -> float:
        if self.count == 0:
            return math.nan
        return max(self.square / self.count - (self.total / self.count) ** 2, 0.0)


@dataclass
class VectorMoments:
    """Streaming independent moments for the unit axis of ``[B,U,...]``."""

    n_units: int
    count_per_unit: int = 0
    total: np.ndarray = field(init=False)
    square: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.total = np.zeros(self.n_units, dtype=np.float64)
        self.square = np.zeros(self.n_units, dtype=np.float64)

    def add(self, value: np.ndarray | torch.Tensor) -> None:
        array = _numpy(value).astype(np.float64, copy=False)
        if array.ndim < 2 or array.shape[1] != self.n_units:
            raise ValueError(f"Expected [B,{self.n_units},...], got {array.shape}")
        axes = (0,) + tuple(range(2, array.ndim))
        per_unit_count = int(np.prod([array.shape[axis] for axis in axes], dtype=np.int64))
        self.count_per_unit += per_unit_count
        self.total += array.sum(axis=axes, dtype=np.float64)
        self.square += np.square(array).sum(axis=axes, dtype=np.float64)

    @property
    def variance(self) -> np.ndarray:
        if self.count_per_unit == 0:
            return np.full(self.n_units, np.nan)
        mean = self.total / self.count_per_unit
        return np.maximum(self.square / self.count_per_unit - np.square(mean), 0.0)


@dataclass
class CrossMoments:
    """Per-unit centered variances and covariance for paired readout terms."""

    n_units: int
    count_per_unit: int = 0
    sum_x: np.ndarray = field(init=False)
    sum_y: np.ndarray = field(init=False)
    sum_x2: np.ndarray = field(init=False)
    sum_y2: np.ndarray = field(init=False)
    sum_xy: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sum_x = np.zeros(self.n_units, dtype=np.float64)
        self.sum_y = np.zeros(self.n_units, dtype=np.float64)
        self.sum_x2 = np.zeros(self.n_units, dtype=np.float64)
        self.sum_y2 = np.zeros(self.n_units, dtype=np.float64)
        self.sum_xy = np.zeros(self.n_units, dtype=np.float64)

    def add(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        left = _numpy(x).astype(np.float64, copy=False)
        right = _numpy(y).astype(np.float64, copy=False)
        if left.shape != right.shape or left.ndim < 2 or left.shape[1] != self.n_units:
            raise ValueError(f"Expected matched [B,{self.n_units},...], got {left.shape}, {right.shape}")
        axes = (0,) + tuple(range(2, left.ndim))
        self.count_per_unit += int(np.prod([left.shape[axis] for axis in axes], dtype=np.int64))
        self.sum_x += left.sum(axis=axes, dtype=np.float64)
        self.sum_y += right.sum(axis=axes, dtype=np.float64)
        self.sum_x2 += np.square(left).sum(axis=axes, dtype=np.float64)
        self.sum_y2 += np.square(right).sum(axis=axes, dtype=np.float64)
        self.sum_xy += np.multiply(left, right).sum(axis=axes, dtype=np.float64)

    def values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.count_per_unit == 0:
            missing = np.full(self.n_units, np.nan)
            return missing.copy(), missing.copy(), missing.copy()
        n = self.count_per_unit
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x = np.maximum(self.sum_x2 / n - np.square(mean_x), 0.0)
        var_y = np.maximum(self.sum_y2 / n - np.square(mean_y), 0.0)
        covariance = self.sum_xy / n - mean_x * mean_y
        return var_x, var_y, covariance


@dataclass
class MapRecoveryAccumulator:
    """Per-unit residual and reference-effect SSE for complete maps."""

    n_units: int = N_UNITS
    residual: np.ndarray = field(init=False)
    reference_effect: np.ndarray = field(init=False)
    residual_equal: np.ndarray = field(init=False)
    reference_effect_equal: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.residual = np.zeros(self.n_units, dtype=np.float64)
        self.reference_effect = np.zeros(self.n_units, dtype=np.float64)
        self.residual_equal = np.zeros(self.n_units, dtype=np.float64)
        self.reference_effect_equal = np.zeros(self.n_units, dtype=np.float64)

    def add(
        self,
        prediction: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
        reference: np.ndarray | torch.Tensor,
        weight: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        pred = _numpy(prediction).astype(np.float64, copy=False)
        truth = _numpy(target).astype(np.float64, copy=False)
        base = _numpy(reference).astype(np.float64, copy=False)
        if pred.shape != truth.shape or base.shape not in (truth.shape, truth.shape[1:]):
            raise ValueError(f"Map shapes do not align: {pred.shape}, {truth.shape}, {base.shape}")
        axes = (0,) + tuple(range(2, truth.ndim))
        residual = np.square(pred - truth)
        effect = np.square(truth - base)
        self.residual_equal += residual.sum(axis=axes, dtype=np.float64)
        self.reference_effect_equal += effect.sum(axis=axes, dtype=np.float64)
        if weight is None:
            weight_array = np.ones(truth.shape[:2], dtype=np.float64)
        else:
            weight_array = _numpy(weight).astype(np.float64, copy=False)
            if weight_array.shape != truth.shape[:2]:
                raise ValueError(f"Expected map weight {truth.shape[:2]}, got {weight_array.shape}")
        expand = weight_array[(...,) + (None,) * (truth.ndim - 2)]
        self.residual += (residual * expand).sum(axis=axes, dtype=np.float64)
        self.reference_effect += (effect * expand).sum(axis=axes, dtype=np.float64)

    def recovery(self) -> np.ndarray:
        return 1.0 - _safe_array_divide(self.residual, self.reference_effect)

    def recovery_equal_unit(self) -> np.ndarray:
        return 1.0 - _safe_array_divide(self.residual_equal, self.reference_effect_equal)


def _numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > EPS64 else math.nan


def _safe_array_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > EPS64)
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_basis(value: np.ndarray, rank: int = RANK, atol: float = 2e-5) -> np.ndarray:
    """Validate, but never rotate, a fitted basis."""
    basis = np.asarray(value, dtype=np.float32)
    if basis.shape != (N_CHANNELS, int(rank)):
        raise ValueError(f"Expected basis {(N_CHANNELS, rank)}, got {basis.shape}")
    gram = np.asarray(basis, dtype=np.float64).T @ np.asarray(basis, dtype=np.float64)
    error = float(np.max(np.abs(gram - np.eye(rank))))
    if not np.isfinite(error) or error > float(atol):
        raise ValueError(f"Basis is not orthonormal: max |U^T U-I|={error:.3g}")
    return basis


def basis_path(contrast_key: str, fold: int) -> Path:
    return LOW_RANK_OUT / "fits/crossval" / contrast_key / f"fold_{int(fold)}" / "rank_008/U.npy"


def available_folds(contrast_keys: Sequence[str] | None = None) -> list[int]:
    keys = list(contrast_keys or [contrast.key for contrast in CONTRASTS])
    return [fold for fold in range(4) if all(basis_path(key, fold).is_file() for key in keys)]


def projection_energy(
    value: np.ndarray | torch.Tensor,
    basis: np.ndarray | torch.Tensor,
) -> tuple[float, float, float, int]:
    """Return total, P, Q squared energy and number of spatial/sample sites.

    ``value`` is ``[B,C,Y,X]``.  Energies sum over channels and are averaged
    over ``B*Y*X`` only.  Thus total=P+Q is a per-location channel-vector
    energy, while division by 8 or 120 is explicitly deferred to reporting.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    u = basis if isinstance(basis, torch.Tensor) else torch.as_tensor(basis, device=tensor.device)
    tensor = tensor.to(dtype=torch.float64)
    u = u.to(device=tensor.device, dtype=torch.float64)
    if tensor.ndim != 4 or tensor.shape[1] != u.shape[0]:
        raise ValueError(f"Expected [B,C,Y,X] compatible with U, got {tuple(tensor.shape)}, {tuple(u.shape)}")
    coordinates = torch.einsum("ck,bcyx->bkyx", u, tensor)
    total = float(torch.square(tensor).sum().item())
    p_energy = float(torch.square(coordinates).sum().item())
    q_energy = max(total - p_energy, 0.0)
    sites = int(tensor.shape[0] * tensor.shape[2] * tensor.shape[3])
    return total, p_energy, q_energy, sites


def energy_row(
    *,
    fold: int,
    contrast: str,
    analysis: str,
    scale: float,
    total_sum: float,
    p_sum: float,
    q_sum: float,
    sites: int,
    definition: str,
) -> dict[str, Any]:
    total = _safe_divide(total_sum, sites)
    p_energy = _safe_divide(p_sum, sites)
    q_energy = _safe_divide(q_sum, sites)
    p_per_dimension = _safe_divide(p_energy, RANK)
    q_per_dimension = _safe_divide(q_energy, Q_RANK)
    return {
        "schema_version": SCHEMA_VERSION,
        "fold": int(fold),
        "contrast": contrast,
        "analysis": analysis,
        "scale": float(scale),
        "definition": definition,
        "n_candidate_dimensions": RANK,
        "n_complementary_dimensions": Q_RANK,
        "n_sample_spatial_sites": int(sites),
        "total_energy": total,
        "candidate_p_energy": p_energy,
        "complementary_q_energy": q_energy,
        "candidate_p_total_fraction": _safe_divide(p_energy, total),
        "complementary_q_total_fraction": _safe_divide(q_energy, total),
        "candidate_p_energy_per_dimension": p_per_dimension,
        "complementary_q_energy_per_dimension": q_per_dimension,
        "p_to_q_per_dimension_energy_ratio": _safe_divide(p_per_dimension, q_per_dimension),
    }


def leverage_scores(basis: np.ndarray) -> np.ndarray:
    return np.square(validate_basis(basis).astype(np.float64)).sum(axis=1)


def effective_participating_channels(leverage: np.ndarray) -> float:
    values = np.asarray(leverage, dtype=np.float64)
    return _safe_divide(float(values.sum()) ** 2, float(np.square(values).sum()))


def exact_channel_readout_strength(
    feature_weights: np.ndarray,
    space_weights: np.ndarray,
    units: np.ndarray,
) -> np.ndarray:
    """Frobenius norm of the complete separable tiled readout per channel."""
    feature = np.asarray(feature_weights, dtype=np.float64)[units]
    space = np.asarray(space_weights, dtype=np.float64)[units]
    spatial_energy = np.square(space).sum(axis=(1, 2))
    return np.sqrt((np.square(feature) * spatial_energy[:, None]).sum(axis=0))


def weight_reliance(feature_weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Exact channel-weight P reliance; spatial factors cancel per RR100 unit."""
    weights = np.asarray(feature_weights, dtype=np.float64)
    u = np.asarray(validate_basis(basis), dtype=np.float64)
    p_energy = np.square(weights @ u).sum(axis=1)
    total = np.square(weights).sum(axis=1)
    return _safe_array_divide(p_energy, total)


def variance_decomposition(var_p: np.ndarray, var_q: np.ndarray, covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose ``var(a_P+a_Q)`` without dropping covariance/cancellation."""
    p = np.asarray(var_p, dtype=np.float64)
    q = np.asarray(var_q, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    full = p + q + 2.0 * cov
    return {
        "variance_full": full,
        "variance_p": p,
        "variance_q": q,
        "twice_covariance_pq": 2.0 * cov,
        "p_fraction_of_full_variance": _safe_array_divide(p, full),
        "q_fraction_of_full_variance": _safe_array_divide(q, full),
        "covariance_fraction_of_full_variance": _safe_array_divide(2.0 * cov, full),
        "activity_weighted_p_reliance_excluding_covariance": _safe_array_divide(p, p + q),
    }


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.ptp(left[finite]) <= 0 or np.ptp(right[finite]) <= 0:
        return math.nan
    function = pearsonr if method == "pearson" else spearmanr
    return float(function(left[finite], right[finite]).statistic)


def _population_indices() -> dict[str, np.ndarray]:
    return {
        "all": np.arange(N_UNITS, dtype=np.int64),
        "lower_sf_71": target_units("low"),
        "higher_sf_29": target_units("high"),
    }


def _fit_units(contrast_key: str) -> np.ndarray:
    contrast = next(value for value in CONTRASTS if value.key == contrast_key)
    return target_units(contrast.group)


def _read_state(handle: h5py.File, pair: tuple[int, int], scale: float) -> np.ndarray:
    image, trajectory = pair
    return np.asarray(handle["h"][image, trajectory, scale_index(scale)], dtype=np.float32)


def _project(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return project_channel_delta(value, basis)


def _readout_state(value: torch.Tensor, readout: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    z = readout_preactivation(value, readout["feature"], readout["bias"], readout["space"])
    rate = softplus_rate(z)
    components = rate_map_components(rate)
    return {"z": z, "rate": rate, **components}


def _readout_linear(value: torch.Tensor, readout: dict[str, torch.Tensor]) -> torch.Tensor:
    return readout_preactivation(
        value,
        readout["feature"],
        torch.zeros_like(readout["bias"]),
        readout["space"],
    )


def _relative_rmse(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    pred = _numpy(prediction).astype(np.float64, copy=False)
    truth = _numpy(target).astype(np.float64, copy=False)
    return float(np.sqrt(np.square(pred - truth).mean()) / max(np.sqrt(np.square(truth).mean()), EPS64))


def _fit_dir(fold: int, contrast: str) -> Path:
    return INTERMEDIATE / f"fold_{int(fold)}" / contrast


def _stage_identity(path: Path) -> tuple[int, str]:
    relative = path.resolve().relative_to(INTERMEDIATE.resolve())
    if len(relative.parts) < 3 or not relative.parts[0].startswith("fold_"):
        raise ValueError(f"Stage directory does not encode fold/contrast identity: {path}")
    return int(relative.parts[0].split("_", 1)[1]), str(relative.parts[1])


def _input_fingerprint() -> dict[str, Any]:
    def stat(path: Path) -> dict[str, Any]:
        value = path.stat()
        return {"path": str(path), "size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns}

    return {
        "pq_semantics_code_sha256": sha256_file(Path(__file__).resolve()),
        "causal_common_code_sha256": sha256_file(
            ROOT / "paper/fig4/mechanism_audit_v1/causal_low_rank/common.py"
        ),
        "causal_data_code_sha256": sha256_file(
            ROOT / "paper/fig4/mechanism_audit_v1/causal_low_rank/data.py"
        ),
        "state_cache": stat(STATE_CACHE),
        "map_cache": stat(MAP_CACHE),
        "readout_cache": {**stat(READOUT_CACHE), "sha256": sha256_file(READOUT_CACHE)},
        "fold_assignments_sha256": sha256_file(LOW_RANK_OUT / "config/fold_assignments.json"),
    }


def _stage_done(path: Path, expected_basis_sha256: str) -> bool:
    marker = path / "complete.json"
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        expected_fold, expected_contrast = _stage_identity(path)
    except (ValueError, IndexError):
        return False
    if not (
        bool(value.get("complete"))
        and value.get("schema_version") == SCHEMA_VERSION
        and int(value.get("fold", -1)) == expected_fold
        and str(value.get("contrast", "")) == expected_contrast
        and value.get("basis_sha256") == expected_basis_sha256
        and value.get("input_fingerprint") == _input_fingerprint()
    ):
        return False
    for product in value.get("products", []):
        product_path = Path(product["path"])
        if not product_path.is_file() or sha256_file(product_path) != product.get("sha256"):
            return False
    return bool(value.get("products"))


def _write_stage_complete(path: Path, *, basis_file: Path, stage: str, products: Sequence[Path]) -> None:
    fold, contrast = _stage_identity(path)
    _atomic_json(
        path / "complete.json",
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "stage": stage,
            "fold": fold,
            "contrast": contrast,
            "basis_path": basis_file,
            "basis_sha256": sha256_file(basis_file),
            "input_fingerprint": _input_fingerprint(),
            "products": [
                {"path": str(product), "sha256": sha256_file(product), "size_bytes": product.stat().st_size}
                for product in products
            ],
            "completed_unix": time.time(),
        },
    )


def _assert_cache_gate() -> None:
    for path in (STATE_CACHE, MAP_CACHE, READOUT_CACHE):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved cache is missing: {path}")
    integrity = LOW_RANK_OUT / "integrity_tests.json"
    if not integrity.is_file():
        integrity = LOW_RANK_OUT / "cache/integrity_tests.json"
    if not integrity.is_file():
        raise FileNotFoundError("The mandatory saved-cache integrity gate is missing")
    result = json.loads(integrity.read_text(encoding="utf-8"))
    if not bool(result.get("optimization_allowed", False)):
        raise RuntimeError("The saved-cache P=0/P=I integrity gate did not pass")
    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        if not bool(state.attrs.get("complete", False)) or not bool(maps.attrs.get("complete", False)):
            raise RuntimeError("Saved state/map cache is not marked complete")
        if not np.asarray(state["completed_pairs"][:], dtype=bool).all():
            raise RuntimeError("Saved state cache contains incomplete pairs")
        if not np.asarray(maps["completed_pairs"][:], dtype=bool).all():
            raise RuntimeError("Saved map cache contains incomplete pairs")
        for name in ("image_ids", "trajectory_ids", "scales"):
            if not np.array_equal(np.asarray(state[name]), np.asarray(maps[name])):
                raise RuntimeError(f"State/map cache mismatch for {name}")


def _accumulate_energy(
    accumulator: dict[float, dict[str, float | int]],
    key: float,
    value: tuple[float, float, float, int],
) -> None:
    row = accumulator.setdefault(
        float(key), {"total": 0.0, "p": 0.0, "q": 0.0, "sites": 0}
    )
    total, p_energy, q_energy, sites = value
    row["total"] = float(row["total"]) + total
    row["p"] = float(row["p"]) + p_energy
    row["q"] = float(row["q"]) + q_energy
    row["sites"] = int(row["sites"]) + sites


def _difference_energy_from_raw_moments(
    raw_total_sum: float,
    raw_p_sum: float,
    mean_value: np.ndarray,
    basis: np.ndarray,
    n_replicates: int,
) -> tuple[float, float, float]:
    """Variance energy using E||x||^2-||E x||^2, retaining spatial maps."""
    mean_array = np.asarray(mean_value)
    if mean_array.ndim == 3:
        mean_array = mean_array[None]
    mean_total, mean_p, _, _ = projection_energy(mean_array, basis)
    total = max(float(raw_total_sum) / n_replicates - mean_total, 0.0)
    p_energy = max(float(raw_p_sum) / n_replicates - mean_p, 0.0)
    return total, p_energy, max(total - p_energy, 0.0)


def _state_semantic_decomposition(
    *,
    state: h5py.File,
    fold: int,
    contrast_key: str,
    basis: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute held-out motion, content, image-mean, and trajectory energies."""
    split = load_fold(fold).test
    pairs = split.pairs
    images = tuple(split.image_positions)
    trajectories = tuple(split.trajectory_positions)
    u = torch.as_tensor(basis, dtype=torch.float64)

    movement: dict[float, dict[str, float | int]] = {}
    channel_sum = np.zeros(N_CHANNELS, dtype=np.float64)
    channel_square = np.zeros(N_CHANNELS, dtype=np.float64)
    channel_sites = 0
    target = next(value for value in CONTRASTS if value.key == contrast_key)

    # Motion is paired against stabilization for every nonzero scale.  The
    # native-channel association is separately tied to the projector's own
    # defining contrast (1x->3x for the reversal projector).
    for pair in pairs:
        h0 = _read_state(state, pair, 0.0)
        for scale in SCALES[1:]:
            hs = _read_state(state, pair, float(scale))
            _accumulate_energy(movement, float(scale), projection_energy(hs - h0, u))
        h_a = h0 if np.isclose(target.scale_a, 0.0) else _read_state(state, pair, target.scale_a)
        h_b = _read_state(state, pair, target.scale_b)
        delta = np.asarray(h_b - h_a, dtype=np.float64)
        reduce_axes = (0, 2, 3)
        channel_sum += delta.sum(axis=reduce_axes, dtype=np.float64)
        channel_square += np.square(delta).sum(axis=reduce_axes, dtype=np.float64)
        channel_sites += int(delta.shape[0] * delta.shape[2] * delta.shape[3])

    rows: list[dict[str, Any]] = []
    for scale, values in movement.items():
        rows.append(
            energy_row(
                fold=fold,
                contrast=contrast_key,
                analysis="movement_change_from_stabilization",
                scale=scale,
                total_sum=float(values["total"]),
                p_sum=float(values["p"]),
                q_sum=float(values["q"]),
                sites=int(values["sites"]),
                definition="h(scale)-h(0), paired image/trajectory/frame; mean over held-out records and spatial positions",
            )
        )

    # Raw stabilization moments retain a CxYx spatial grand-mean map.  Image
    # means first average trajectories and frames, exactly as specified.
    content_raw_total = 0.0
    content_raw_p = 0.0
    content_n_records = 0
    content_sum = np.zeros((N_CHANNELS, state["h"].shape[-2], state["h"].shape[-1]), dtype=np.float64)
    image_means: list[np.ndarray] = []
    trajectory_accumulator = {
        float(scale): {"total": 0.0, "p": 0.0, "q": 0.0, "sites": 0}
        for scale in SCALES
    }

    for image in images:
        for scale in SCALES:
            trajectory_sum: np.ndarray | None = None
            raw_total = 0.0
            raw_p = 0.0
            for trajectory in trajectories:
                value = _read_state(state, (int(image), int(trajectory)), float(scale))
                if trajectory_sum is None:
                    trajectory_sum = np.zeros_like(value, dtype=np.float64)
                trajectory_sum += value
                total, p_energy, _, _ = projection_energy(value, u)
                raw_total += total
                raw_p += p_energy
                if np.isclose(scale, 0.0):
                    content_raw_total += total
                    content_raw_p += p_energy
                    content_sum += value.sum(axis=0, dtype=np.float64)
                    content_n_records += int(value.shape[0])
            assert trajectory_sum is not None
            mean_trajectory = trajectory_sum / len(trajectories)
            total_var, p_var, q_var = _difference_energy_from_raw_moments(
                raw_total,
                raw_p,
                mean_trajectory,
                basis,
                len(trajectories),
            )
            trajectory_accumulator[float(scale)]["total"] += total_var
            trajectory_accumulator[float(scale)]["p"] += p_var
            trajectory_accumulator[float(scale)]["q"] += q_var
            trajectory_accumulator[float(scale)]["sites"] += int(
                mean_trajectory.shape[0] * mean_trajectory.shape[2] * mean_trajectory.shape[3]
            )
            if np.isclose(scale, 0.0):
                image_means.append(mean_trajectory.mean(axis=0, dtype=np.float64))

    grand_mean = content_sum / content_n_records
    content_total, content_p, content_q = _difference_energy_from_raw_moments(
        content_raw_total,
        content_raw_p,
        grand_mean,
        basis,
        content_n_records,
    )
    spatial_sites = int(grand_mean.shape[-2] * grand_mean.shape[-1])
    rows.append(
        energy_row(
            fold=fold,
            contrast=contrast_key,
            analysis="stabilized_visual_content_all_records",
            scale=0.0,
            total_sum=content_total,
            p_sum=content_p,
            q_sum=content_q,
            sites=spatial_sites,
            definition="h(i,j,0,t)-grand mean over held-out i,j,t; grand mean retains channel and spatial position",
        )
    )

    image_stack = np.stack(image_means, axis=0)
    grand_image_mean = image_stack.mean(axis=0, dtype=np.float64)
    image_delta = image_stack - grand_image_mean[None]
    image_total, image_p, image_q, image_sites = projection_energy(image_delta, basis)
    rows.append(
        energy_row(
            fold=fold,
            contrast=contrast_key,
            analysis="stabilized_visual_content_image_means",
            scale=0.0,
            total_sum=image_total,
            p_sum=image_p,
            q_sum=image_q,
            sites=image_sites,
            definition="image means over held-out trajectories and frames, centered across the held-out images",
        )
    )

    for scale, values in trajectory_accumulator.items():
        rows.append(
            energy_row(
                fold=fold,
                contrast=contrast_key,
                analysis="trajectory_specific_at_fixed_image_scale_frame",
                scale=scale,
                total_sum=float(values["total"]),
                p_sum=float(values["p"]),
                q_sum=float(values["q"]),
                sites=int(values["sites"]),
                definition="variance across held-out trajectories at fixed image, scale, frame and spatial position",
            )
        )

    channel_mean = channel_sum / max(channel_sites, 1)
    channel_mean_square = channel_square / max(channel_sites, 1)
    channel_variance = np.maximum(channel_mean_square - np.square(channel_mean), 0.0)
    return pd.DataFrame(rows), channel_mean_square, channel_variance


def _leverage_table(
    *,
    fold: int,
    contrast_key: str,
    basis: np.ndarray,
    motion_mean_square: np.ndarray,
    motion_variance: np.ndarray,
) -> pd.DataFrame:
    with np.load(READOUT_CACHE, allow_pickle=False) as archive:
        feature = np.asarray(archive["feature_weights"], dtype=np.float64)
        space = np.asarray(archive["space_weights"], dtype=np.float64)
    leverage = leverage_scores(basis)
    units = _fit_units(contrast_key)
    strength = exact_channel_readout_strength(feature, space, units)
    strength_all = exact_channel_readout_strength(feature, space, np.arange(N_UNITS))
    order = np.argsort(-leverage, kind="stable")
    rank = np.empty(N_CHANNELS, dtype=np.int64)
    rank[order] = np.arange(1, N_CHANNELS + 1)
    cumulative_by_rank = np.cumsum(leverage[order])
    cumulative = cumulative_by_rank[rank - 1]
    summary = {
        "effective_participating_channel_count": effective_participating_channels(leverage),
        "leverage_sum": float(leverage.sum()),
        "leverage_pearson_readout_strength": _correlation(leverage, strength, "pearson"),
        "leverage_spearman_readout_strength": _correlation(leverage, strength, "spearman"),
        "leverage_pearson_motion_delta_mean_square": _correlation(leverage, motion_mean_square, "pearson"),
        "leverage_spearman_motion_delta_mean_square": _correlation(leverage, motion_mean_square, "spearman"),
        "leverage_pearson_centered_motion_variance": _correlation(leverage, motion_variance, "pearson"),
        "leverage_spearman_centered_motion_variance": _correlation(leverage, motion_variance, "spearman"),
    }
    table = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "fold": int(fold),
            "contrast": contrast_key,
            "native_channel": np.arange(N_CHANNELS, dtype=np.int64),
            "leverage_score_p_cc": leverage,
            "leverage_rank_descending": rank,
            "cumulative_leverage_at_channel_rank": cumulative,
            "cumulative_leverage_fraction_at_channel_rank": cumulative / RANK,
            "exact_target_population_readout_frobenius_strength": strength,
            "exact_all_rr100_readout_frobenius_strength": strength_all,
            "contrast_motion_delta_mean_square": motion_mean_square,
            "contrast_motion_delta_centered_variance": motion_variance,
        }
    )
    for key, value in summary.items():
        table[key] = value
    return table


def run_variance_stage(fold: int, contrast_key: str, *, overwrite: bool = False) -> dict[str, Any]:
    basis_file = basis_path(contrast_key, fold)
    if not basis_file.is_file():
        raise FileNotFoundError(f"Fold-specific rank-8 basis is missing: {basis_file}")
    stage_dir = _fit_dir(fold, contrast_key) / "variance"
    basis_hash = sha256_file(basis_file)
    if not overwrite and _stage_done(stage_dir, basis_hash):
        return {"status": "reused", "fold": fold, "contrast": contrast_key, "path": str(stage_dir)}
    stage_dir.mkdir(parents=True, exist_ok=True)
    stale_marker = stage_dir / "complete.json"
    if stale_marker.exists():
        stale_marker.unlink()
    basis = validate_basis(np.load(basis_file, allow_pickle=False))
    with h5py.File(STATE_CACHE, "r") as state:
        variance, motion_mean_square, motion_variance = _state_semantic_decomposition(
            state=state,
            fold=fold,
            contrast_key=contrast_key,
            basis=basis,
        )
    leverage = _leverage_table(
        fold=fold,
        contrast_key=contrast_key,
        basis=basis,
        motion_mean_square=motion_mean_square,
        motion_variance=motion_variance,
    )
    variance_path = stage_dir / "pq_variance_decomposition.csv"
    leverage_path = stage_dir / "native_channel_leverage.csv"
    arrays_path = stage_dir / "variance_supporting_arrays.npz"
    _atomic_csv(variance_path, variance)
    _atomic_csv(leverage_path, leverage)
    np.savez_compressed(
        arrays_path,
        basis=basis,
        projector=(basis @ basis.T).astype(np.float32),
        leverage=leverage.leverage_score_p_cc.to_numpy(np.float32),
        native_channel_motion_delta_mean_square=np.asarray(motion_mean_square, dtype=np.float32),
        native_channel_motion_delta_centered_variance=np.asarray(motion_variance, dtype=np.float32),
    )
    _write_stage_complete(
        stage_dir,
        basis_file=basis_file,
        stage="variance",
        products=(variance_path, leverage_path, arrays_path),
    )
    return {"status": "completed", "fold": fold, "contrast": contrast_key, "path": str(stage_dir)}


def _training_pairs(fold: int) -> list[tuple[int, int]]:
    """Exact outer-training pairs: fit-training union validation, no test pair."""
    value = load_fold(fold)
    pairs = sorted(set(value.train.pairs) | set(value.validation.pairs))
    if set(pairs) & set(value.test.pairs):
        raise RuntimeError(f"Fold {fold}: training-mean pairs overlap the held-out test block")
    return pairs


def _training_mean_state(fold: int, *, overwrite: bool = False) -> tuple[np.ndarray, Path]:
    destination = INTERMEDIATE / f"fold_{fold}" / "outer_training_mean_scale0.npy"
    metadata = destination.with_suffix(".json")
    pairs = _training_pairs(fold)
    if destination.is_file() and metadata.is_file() and not overwrite:
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            mean = np.load(destination, allow_pickle=False)
            reusable = (
                value.get("schema_version") == SCHEMA_VERSION
                and value.get("pairs") == [list(pair) for pair in pairs]
                and value.get("state_cache_fingerprint") == _input_fingerprint()["state_cache"]
                and value.get("fold_assignments_sha256")
                == _input_fingerprint()["fold_assignments_sha256"]
                and value.get("mean_state_sha256") == sha256_file(destination)
                and mean.shape == (N_CHANNELS, STATE_SIZE, STATE_SIZE)
                and np.isfinite(mean).all()
            )
            if reusable:
                return np.asarray(mean, dtype=np.float32), destination
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    total: np.ndarray | None = None
    n_records = 0
    with h5py.File(STATE_CACHE, "r") as state:
        for pair in pairs:
            value = _read_state(state, pair, 0.0)
            if total is None:
                total = np.zeros(value.shape[1:], dtype=np.float64)
            total += value.sum(axis=0, dtype=np.float64)
            n_records += int(value.shape[0])
    if total is None or n_records == 0:
        raise RuntimeError(f"Fold {fold}: cannot form an outer-training stabilization mean")
    mean = (total / n_records).astype(np.float32)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, mean, allow_pickle=False)
    os.replace(temporary, destination)
    _atomic_json(
        metadata,
        {
            "schema_version": SCHEMA_VERSION,
            "definition": "mean stabilized ConvGRU state over exact fold train+validation pairs and all 40 frames",
            "fold": fold,
            "pairs": [list(pair) for pair in pairs],
            "n_records": n_records,
            "test_pairs_excluded": True,
            "state_cache": str(STATE_CACHE),
            "state_cache_fingerprint": _input_fingerprint()["state_cache"],
            "fold_assignments_sha256": _input_fingerprint()["fold_assignments_sha256"],
            "mean_state_sha256": sha256_file(destination),
            "mean_state_shape": list(mean.shape),
        },
    )
    return mean, destination


def _cached_endpoint(
    maps: h5py.File,
    pair: tuple[int, int],
    scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image, trajectory = pair
    index = (image, trajectory, scale_index(scale))
    return {
        "z": torch.as_tensor(np.asarray(maps["preactivation"][index], dtype=np.float32), device=device),
        "rate": torch.as_tensor(np.asarray(maps["rate"][index], dtype=np.float32), device=device),
        "gain": torch.as_tensor(np.asarray(maps["gain"][index], dtype=np.float32), device=device),
        "ssi": torch.as_tensor(np.asarray(maps["ssi"][index], dtype=np.float32), device=device),
        "expected_spikes": torch.as_tensor(
            np.asarray(maps["expected_spikes"][index], dtype=np.float32), device=device
        ),
        "mean_rate": torch.as_tensor(np.asarray(maps["mean_rate"][index], dtype=np.float32), device=device),
    }


def _new_recovery_bundle() -> dict[str, MapRecoveryAccumulator]:
    return {name: MapRecoveryAccumulator() for name in ("z", "rate", "gain")}


def _add_recovery_bundle(
    bundle: dict[str, MapRecoveryAccumulator],
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> None:
    target_weight = target.get("expected_spikes")
    reference_weight = reference.get("expected_spikes")
    if target_weight is None or reference_weight is None:
        raise ValueError("Recovery bundles require target and reference expected-spike weights")
    weight = 0.5 * (target_weight + reference_weight)
    for name in bundle:
        bundle[name].add(prediction[name], target[name], reference[name], weight=weight)


def _population_recovery(
    accumulator: MapRecoveryAccumulator,
    units: np.ndarray,
) -> float:
    return 1.0 - _safe_divide(
        float(accumulator.residual[units].sum()),
        float(accumulator.reference_effect[units].sum()),
    )


def _population_recovery_equal(accumulator: MapRecoveryAccumulator, units: np.ndarray) -> float:
    values = accumulator.recovery_equal_unit()[units]
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def _new_dose_summary() -> dict[str, np.ndarray]:
    return {
        "ssi_numerator": np.zeros(N_UNITS, dtype=np.float64),
        "ssi_weight": np.zeros(N_UNITS, dtype=np.float64),
        "mean_rate_sum": np.zeros(N_UNITS, dtype=np.float64),
        "n_maps": np.zeros(N_UNITS, dtype=np.int64),
    }


def _add_dose_summary(summary: dict[str, np.ndarray], value: dict[str, torch.Tensor]) -> None:
    ssi = _numpy(value["ssi"]).astype(np.float64, copy=False)
    expected = _numpy(value["expected_spikes"]).astype(np.float64, copy=False)
    mean_rate = _numpy(value["mean_rate"]).astype(np.float64, copy=False)
    summary["ssi_numerator"] += (ssi * expected).sum(axis=0, dtype=np.float64)
    summary["ssi_weight"] += expected.sum(axis=0, dtype=np.float64)
    summary["mean_rate_sum"] += mean_rate.sum(axis=0, dtype=np.float64)
    summary["n_maps"] += mean_rate.shape[0]


def _population_dose(summary: dict[str, np.ndarray], units: np.ndarray) -> tuple[float, float]:
    ssi = _safe_divide(
        float(summary["ssi_numerator"][units].sum()),
        float(summary["ssi_weight"][units].sum()),
    )
    rate = _safe_divide(
        float(summary["mean_rate_sum"][units].sum()),
        float(summary["n_maps"][units].sum()),
    )
    return ssi, rate


def _new_population_map_sums() -> dict[str, dict[str, np.ndarray | float]]:
    return {
        name: {"numerator": np.zeros((51, 51), dtype=np.float64), "weight": 0.0}
        for name in _population_indices()
    }


def _add_population_maps(
    sums: dict[str, dict[str, np.ndarray | float]], value: dict[str, torch.Tensor]
) -> None:
    gain = _numpy(value["gain"]).astype(np.float64, copy=False)
    weight = _numpy(value["expected_spikes"]).astype(np.float64, copy=False)
    for name, units in _population_indices().items():
        selected_gain = gain[:, units]
        selected_weight = weight[:, units]
        sums[name]["numerator"] += (
            selected_gain * selected_weight[..., None, None]
        ).sum(axis=(0, 1), dtype=np.float64)
        sums[name]["weight"] = float(sums[name]["weight"]) + float(selected_weight.sum())


def _final_population_maps(sums: dict[str, dict[str, np.ndarray | float]]) -> np.ndarray:
    result = []
    for name in _population_indices():
        numerator = np.asarray(sums[name]["numerator"], dtype=np.float64)
        result.append(numerator / max(float(sums[name]["weight"]), EPS64))
    return np.asarray(result, dtype=np.float32)


def _standard_gain_recovery(
    residual_sse: np.ndarray,
    target_variation_sse: np.ndarray,
    units: np.ndarray,
) -> float:
    return 1.0 - _safe_divide(
        float(residual_sse[units].sum()), float(target_variation_sse[units].sum())
    )


def _unit_observational_metadata(fold: int) -> pd.DataFrame:
    """Historical SF plus SSI covariates from this fold's crossed test block."""
    pairs = load_fold(fold).test.pairs
    numerator = np.zeros((len(SCALES), N_UNITS), dtype=np.float64)
    denominator = np.zeros((len(SCALES), N_UNITS), dtype=np.float64)
    with h5py.File(MAP_CACHE, "r") as maps:
        for image, trajectory in pairs:
            ssi = np.asarray(maps["ssi"][image, trajectory], dtype=np.float64)
            expected = np.asarray(maps["expected_spikes"][image, trajectory], dtype=np.float64)
            numerator += (ssi * expected).sum(axis=1, dtype=np.float64)
            denominator += expected.sum(axis=1, dtype=np.float64)
    scales = np.asarray(SCALES, dtype=np.float64)
    curve = _safe_array_divide(numerator, denominator)
    optimum = scales[np.nanargmax(curve, axis=0)]
    index = {float(scale): int(np.flatnonzero(np.isclose(scales, scale))[0]) for scale in scales}
    features = pd.read_csv(UNIT_FEATURE_TABLE).sort_values("unit_index")
    if not np.array_equal(features.unit_index.to_numpy(int), np.arange(N_UNITS)):
        raise RuntimeError("RR100 unit-feature table is not indexed 0..99")
    groups = np.full(N_UNITS, "lower_sf_71", dtype=object)
    groups[target_units("high")] = "higher_sf_29"
    return pd.DataFrame(
        {
            "unit_index": np.arange(N_UNITS),
            "historical_sf_population": groups,
            "sf_split_metric": pd.to_numeric(features.sf_split_metric, errors="coerce").to_numpy(float),
            "optimal_movement_scale": optimum,
            "observed_ssi_0x_bits": curve[index[0.0]],
            "observed_ssi_0_to_1_change_bits": curve[index[1.0]] - curve[index[0.0]],
            "observed_ssi_0_to_2_change_bits": curve[index[2.0]] - curve[index[0.0]],
            "observed_ssi_1_to_3_change_bits": curve[index[3.0]] - curve[index[1.0]],
            "high_motion_reversal": curve[index[3.0]] < curve[index[1.0]],
            "heldout_neural_prediction_quality": np.nan,
            "heldout_neural_prediction_quality_status": "not available in the validated saved products",
            "observational_ssi_scope": f"fold_{fold}_crossed_test_block_only_2_images_x_6_trajectories_x_40_frames",
        }
    )


def _contrast_observed_benefit(metadata: pd.DataFrame, contrast_key: str) -> np.ndarray:
    column = {
        "low_0_to_2": "observed_ssi_0_to_2_change_bits",
        "high_0_to_1": "observed_ssi_0_to_1_change_bits",
        "high_1_to_3": "observed_ssi_1_to_3_change_bits",
    }[contrast_key]
    return metadata[column].to_numpy(float)


def run_readout_stage(
    fold: int,
    contrast_key: str,
    *,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the exact tiled RR100 P/Q decomposition on one held-out fold."""
    basis_file = basis_path(contrast_key, fold)
    if not basis_file.is_file():
        raise FileNotFoundError(f"Fold-specific rank-8 basis is missing: {basis_file}")
    stage_dir = _fit_dir(fold, contrast_key) / "readout"
    basis_hash = sha256_file(basis_file)
    if not overwrite and _stage_done(stage_dir, basis_hash):
        return {"status": "reused", "fold": fold, "contrast": contrast_key, "path": str(stage_dir)}
    stage_dir.mkdir(parents=True, exist_ok=True)
    stale_marker = stage_dir / "complete.json"
    if stale_marker.exists():
        stale_marker.unlink()

    torch_device = torch.device(device)
    basis = validate_basis(np.load(basis_file, allow_pickle=False))
    u = torch.as_tensor(basis, dtype=torch.float32, device=torch_device)
    readout = load_readout(device=torch_device)
    h_mean_numpy, h_mean_path = _training_mean_state(fold, overwrite=False)
    h_mean = torch.as_tensor(h_mean_numpy, dtype=torch.float32, device=torch_device)
    mean_output = _readout_state(h_mean[None], readout)
    bias_map = readout["bias"][None, :, None, None]

    baseline_recovery = {
        component: _new_recovery_bundle() for component in ("candidate_p_content", "complementary_q_content")
    }
    baseline_dose = {component: _new_dose_summary() for component in ("full", "training_mean", "candidate_p_content", "complementary_q_content")}
    baseline_maps = {component: _new_population_map_sums() for component in baseline_dose}
    baseline_image_means: dict[str, dict[str, list[np.ndarray]]] = {
        component: {metric: [] for metric in ("z", "rate", "gain")}
        for component in baseline_dose
    }

    moving_scales = [float(value) for value in SCALES[1:]]
    movement_recovery = {
        scale: {
            component: _new_recovery_bundle()
            for component in ("candidate_p_only_movement", "complementary_q_only_movement")
        }
        for scale in moving_scales
    }
    contrast_recovery = {
        component: _new_recovery_bundle()
        for component in ("candidate_p_only_contrast", "complementary_q_only_contrast")
    }
    dose = {
        float(scale): {
            component: _new_dose_summary()
            for component in ("full", "candidate_p_only_movement", "complementary_q_only_movement")
        }
        for scale in SCALES
    }
    movement_maps = {
        float(scale): {
            component: _new_population_map_sums()
            for component in ("full", "candidate_p_only_movement", "complementary_q_only_movement")
        }
        for scale in SCALES
    }

    activity = CrossMoments(N_UNITS)
    absolute_gain_residual = {
        "candidate_p_absolute": np.zeros((len(SCALES), N_UNITS), dtype=np.float64),
        "complementary_q_absolute": np.zeros((len(SCALES), N_UNITS), dtype=np.float64),
    }
    absolute_gain_target = np.zeros((len(SCALES), N_UNITS), dtype=np.float64)
    absolute_rate_sum = {
        "full": np.zeros((len(SCALES), N_UNITS), dtype=np.float64),
        "candidate_p_absolute": np.zeros((len(SCALES), N_UNITS), dtype=np.float64),
        "complementary_q_absolute": np.zeros((len(SCALES), N_UNITS), dtype=np.float64),
    }
    absolute_rate_count = np.zeros(len(SCALES), dtype=np.int64)
    image_contribution_means: dict[float, dict[str, list[np.ndarray]]] = {
        float(scale): {"p": [], "q": []} for scale in SCALES
    }
    decomposition_errors: list[float] = []
    cache_z_errors: list[float] = []
    cache_rate_errors: list[float] = []

    test = load_fold(fold).test
    trajectories = tuple(test.trajectory_positions)

    def absolute_terms(
        h: torch.Tensor,
        cached: dict[str, torch.Tensor],
        scale_i: int,
        image_sums: dict[str, np.ndarray],
    ) -> None:
        h_p = _project(h, u)
        h_q = h - h_p
        a_p = _readout_linear(h_p, readout)
        a_q = _readout_linear(h_q, readout)
        z_literal = readout_preactivation(h, readout["feature"], readout["bias"], readout["space"])
        decomposition_errors.append(_relative_rmse(bias_map + a_p + a_q, z_literal))
        cache_z_errors.append(_relative_rmse(z_literal, cached["z"]))
        literal_rate = softplus_rate(z_literal)
        cache_rate_errors.append(_relative_rmse(literal_rate, cached["rate"]))
        activity.add(a_p, a_q)
        image_sums["p"] += _numpy(a_p.sum(dim=0)).astype(np.float64)
        image_sums["q"] += _numpy(a_q.sum(dim=0)).astype(np.float64)

        full_components = rate_map_components(literal_rate)
        absolute_rate_sum["full"][scale_i] += _numpy(full_components["mean_rate"]).sum(axis=0)
        absolute_rate_count[scale_i] += int(h.shape[0])
        target_gain = full_components["gain"]
        absolute_gain_target[scale_i] += np.square(_numpy(target_gain) - 1.0).sum(axis=(0, 2, 3))
        for component, z_component in (
            ("candidate_p_absolute", bias_map + a_p),
            ("complementary_q_absolute", bias_map + a_q),
        ):
            component_output = rate_map_components(softplus_rate(z_component))
            absolute_rate_sum[component][scale_i] += _numpy(component_output["mean_rate"]).sum(axis=0)
            absolute_gain_residual[component][scale_i] += np.square(
                _numpy(component_output["gain"] - target_gain)
            ).sum(axis=(0, 2, 3))

    with h5py.File(STATE_CACHE, "r") as state, h5py.File(MAP_CACHE, "r") as maps:
        # The training-mean reference is a single map; its descriptive values
        # do not depend on the number of test observations.
        _add_dose_summary(baseline_dose["training_mean"], mean_output)
        _add_population_maps(baseline_maps["training_mean"], mean_output)

        for image in test.image_positions:
            _gpu_budget_checkpoint()
            baseline_image_sums = {
                component: {
                    metric: np.zeros((N_UNITS, 51, 51), dtype=np.float64)
                    for metric in ("z", "rate", "gain")
                }
                for component in baseline_dose
            }
            per_image = {
                float(scale): {
                    "p": np.zeros((N_UNITS, 51, 51), dtype=np.float64),
                    "q": np.zeros((N_UNITS, 51, 51), dtype=np.float64),
                }
                for scale in SCALES
            }
            for trajectory in trajectories:
                _gpu_budget_checkpoint()
                pair = (int(image), int(trajectory))
                h0 = torch.as_tensor(_read_state(state, pair, 0.0), device=torch_device)
                endpoint0 = _cached_endpoint(maps, pair, 0.0, torch_device)
                delta_content = h0 - h_mean[None]
                p_content_state = h_mean[None] + _project(delta_content, u)
                q_content_state = h_mean[None] + (delta_content - _project(delta_content, u))
                p_content = _readout_state(p_content_state, readout)
                q_content = _readout_state(q_content_state, readout)

                _add_recovery_bundle(
                    baseline_recovery["candidate_p_content"],
                    p_content,
                    endpoint0,
                    {
                        name: mean_output[name][0]
                        for name in ("z", "rate", "gain", "expected_spikes")
                    },
                )
                _add_recovery_bundle(
                    baseline_recovery["complementary_q_content"],
                    q_content,
                    endpoint0,
                    {
                        name: mean_output[name][0]
                        for name in ("z", "rate", "gain", "expected_spikes")
                    },
                )
                for component, value in (
                    ("full", endpoint0),
                    ("candidate_p_content", p_content),
                    ("complementary_q_content", q_content),
                ):
                    _add_dose_summary(baseline_dose[component], value)
                    _add_population_maps(baseline_maps[component], value)
                    for metric in ("z", "rate", "gain"):
                        baseline_image_sums[component][metric] += _numpy(value[metric]).sum(
                            axis=0, dtype=np.float64
                        )
                for metric in ("z", "rate", "gain"):
                    baseline_image_sums["training_mean"][metric] += (
                        _numpy(mean_output[metric][0]).astype(np.float64) * N_FRAMES
                    )

                # At scale zero, both movement-only constructions are the
                # intact stabilized state by definition.
                for component in dose[0.0]:
                    _add_dose_summary(dose[0.0][component], endpoint0)
                    _add_population_maps(movement_maps[0.0][component], endpoint0)
                absolute_terms(h0, endpoint0, 0, per_image[0.0])

                contrast_definition = next(value for value in CONTRASTS if value.key == contrast_key)
                contrast_states: dict[float, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {
                    0.0: (h0, endpoint0)
                }

                for scale_i, scale in enumerate(SCALES[1:], start=1):
                    scale_value = float(scale)
                    hs = torch.as_tensor(_read_state(state, pair, scale_value), device=torch_device)
                    endpoint = _cached_endpoint(maps, pair, scale_value, torch_device)
                    contrast_states[scale_value] = (hs, endpoint)
                    delta = hs - h0
                    p_delta = _project(delta, u)
                    p_movement = _readout_state(h0 + p_delta, readout)
                    q_movement = _readout_state(h0 + (delta - p_delta), readout)
                    for component, value in (
                        ("candidate_p_only_movement", p_movement),
                        ("complementary_q_only_movement", q_movement),
                    ):
                        _add_recovery_bundle(movement_recovery[scale_value][component], value, endpoint, endpoint0)
                        _add_dose_summary(dose[scale_value][component], value)
                        _add_population_maps(movement_maps[scale_value][component], value)
                    _add_dose_summary(dose[scale_value]["full"], endpoint)
                    _add_population_maps(movement_maps[scale_value]["full"], endpoint)
                    absolute_terms(hs, endpoint, scale_i, per_image[scale_value])

                h_a, endpoint_a = contrast_states[float(contrast_definition.scale_a)]
                h_b, endpoint_b = contrast_states[float(contrast_definition.scale_b)]
                contrast_delta = h_b - h_a
                p_contrast_delta = _project(contrast_delta, u)
                p_contrast = _readout_state(h_a + p_contrast_delta, readout)
                q_contrast = _readout_state(h_a + (contrast_delta - p_contrast_delta), readout)
                _add_recovery_bundle(
                    contrast_recovery["candidate_p_only_contrast"],
                    p_contrast,
                    endpoint_b,
                    endpoint_a,
                )
                _add_recovery_bundle(
                    contrast_recovery["complementary_q_only_contrast"],
                    q_contrast,
                    endpoint_b,
                    endpoint_a,
                )

            divisor = len(trajectories) * N_FRAMES
            for component in baseline_dose:
                for metric in ("z", "rate", "gain"):
                    baseline_image_means[component][metric].append(
                        baseline_image_sums[component][metric] / divisor
                    )
            for scale in SCALES:
                image_contribution_means[float(scale)]["p"].append(per_image[float(scale)]["p"] / divisor)
                image_contribution_means[float(scale)]["q"].append(per_image[float(scale)]["q"] / divisor)

    rows: list[dict[str, Any]] = []
    populations = _population_indices()
    baseline_full_by_population: dict[str, tuple[float, float]] = {}
    for population, units in populations.items():
        baseline_full_by_population[population] = _population_dose(baseline_dose["full"], units)
        full_ssi, full_rate = baseline_full_by_population[population]
        mean_ssi, mean_rate = _population_dose(baseline_dose["training_mean"], units)
        for component in baseline_dose:
            component_ssi, component_rate = _population_dose(baseline_dose[component], units)
            if component in baseline_recovery:
                recovery = baseline_recovery[component]
                z_r2 = _population_recovery(recovery["z"], units)
                rate_r2 = _population_recovery(recovery["rate"], units)
                gain_r2 = _population_recovery(recovery["gain"], units)
                z_r2_equal = _population_recovery_equal(recovery["z"], units)
                rate_r2_equal = _population_recovery_equal(recovery["rate"], units)
                gain_r2_equal = _population_recovery_equal(recovery["gain"], units)
            elif component == "full":
                z_r2 = rate_r2 = gain_r2 = 1.0
                z_r2_equal = rate_r2_equal = gain_r2_equal = 1.0
            else:
                z_r2 = rate_r2 = gain_r2 = 0.0
                z_r2_equal = rate_r2_equal = gain_r2_equal = 0.0
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "fold": fold,
                    "contrast": contrast_key,
                    "analysis": "baseline_visual_reconstruction",
                    "scale": 0.0,
                    "population": population,
                    "n_units": len(units),
                    "component": component,
                    "preactivation_map_recovery_vs_training_mean_r2": z_r2,
                    "rate_map_recovery_vs_training_mean_r2": rate_r2,
                    "normalized_map_recovery_vs_training_mean_r2": gain_r2,
                    "preactivation_map_recovery_vs_training_mean_r2_equal_unit_unweighted": z_r2_equal,
                    "rate_map_recovery_vs_training_mean_r2_equal_unit_unweighted": rate_r2_equal,
                    "normalized_map_recovery_vs_training_mean_r2_equal_unit_unweighted": gain_r2_equal,
                    "primary_map_recovery_weighting": "paired expected spikes",
                    "mean_rate_hz": component_rate,
                    "mean_rate_ratio_to_full": _safe_divide(component_rate, full_rate),
                    "exact_ssi_bits": component_ssi,
                    "ssi_difference_from_full_bits": component_ssi - full_ssi,
                    "reference_training_mean_ssi_bits": mean_ssi,
                    "reference_training_mean_rate_hz": mean_rate,
                }
            )

    baseline_image_metrics: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        component: {} for component in baseline_dose
    }
    for metric in ("z", "rate", "gain"):
        target_stack = np.stack(baseline_image_means["full"][metric], axis=0)
        target_delta = target_stack - target_stack.mean(axis=0, keepdims=True)
        target_energy = np.square(target_delta).mean(axis=(0, 2, 3))
        for component in baseline_dose:
            prediction_stack = np.stack(baseline_image_means[component][metric], axis=0)
            prediction_delta = prediction_stack - prediction_stack.mean(axis=0, keepdims=True)
            prediction_energy = np.square(prediction_delta).mean(axis=(0, 2, 3))
            residual = np.square(prediction_delta - target_delta).mean(axis=(0, 2, 3))
            recovery = 1.0 - _safe_array_divide(residual, target_energy)
            ratio = _safe_array_divide(prediction_energy, target_energy)
            baseline_image_metrics[component][metric] = (target_energy, ratio, recovery)
    for population, units in populations.items():
        for component in baseline_dose:
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "fold": fold,
                "contrast": contrast_key,
                "analysis": "baseline_image_to_image_response_variance",
                "scale": 0.0,
                "population": population,
                "n_units": len(units),
                "component": component,
                "variance_definition": "variance across held-out image means at each RR100 unit and map position",
            }
            for metric in ("z", "rate", "gain"):
                target_energy, ratio, recovery = baseline_image_metrics[component][metric]
                row[f"{metric}_target_image_variance"] = float(np.nanmean(target_energy[units]))
                row[f"{metric}_predicted_to_target_image_variance_ratio"] = float(
                    np.nanmean(ratio[units])
                )
                row[f"{metric}_centered_image_map_recovery_r2"] = float(
                    np.nanmean(recovery[units])
                )
            rows.append(row)

    for scale in SCALES:
        scale_value = float(scale)
        for population, units in populations.items():
            base_ssi, _ = baseline_full_by_population[population]
            full_ssi, full_rate = _population_dose(dose[scale_value]["full"], units)
            for component in dose[scale_value]:
                ssi, rate = _population_dose(dose[scale_value][component], units)
                if np.isclose(scale_value, 0.0):
                    z_r2 = rate_r2 = gain_r2 = math.nan
                    z_r2_equal = rate_r2_equal = gain_r2_equal = math.nan
                elif component == "full":
                    z_r2 = rate_r2 = gain_r2 = 1.0
                    z_r2_equal = rate_r2_equal = gain_r2_equal = 1.0
                else:
                    recovery = movement_recovery[scale_value][component]
                    z_r2 = _population_recovery(recovery["z"], units)
                    rate_r2 = _population_recovery(recovery["rate"], units)
                    gain_r2 = _population_recovery(recovery["gain"], units)
                    z_r2_equal = _population_recovery_equal(recovery["z"], units)
                    rate_r2_equal = _population_recovery_equal(recovery["rate"], units)
                    gain_r2_equal = _population_recovery_equal(recovery["gain"], units)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "fold": fold,
                        "contrast": contrast_key,
                        "analysis": "movement_effect_decomposition",
                        "scale": scale_value,
                        "population": population,
                        "n_units": len(units),
                        "component": component,
                        "preactivation_movement_effect_recovery_r2": z_r2,
                        "rate_movement_effect_recovery_r2": rate_r2,
                        "normalized_map_movement_effect_recovery_r2": gain_r2,
                        "preactivation_movement_effect_recovery_r2_equal_unit_unweighted": z_r2_equal,
                        "rate_movement_effect_recovery_r2_equal_unit_unweighted": rate_r2_equal,
                        "normalized_map_movement_effect_recovery_r2_equal_unit_unweighted": gain_r2_equal,
                        "primary_map_recovery_weighting": "paired expected spikes",
                        "mean_rate_hz": rate,
                        "mean_rate_ratio_to_full": _safe_divide(rate, full_rate),
                        "exact_ssi_bits": ssi,
                        "ssi_change_from_stabilization_bits": ssi - base_ssi,
                        "ssi_difference_from_full_bits": ssi - full_ssi,
                    }
                )

    defining_contrast = next(value for value in CONTRASTS if value.key == contrast_key)
    for population, units in populations.items():
        for component, recovery in contrast_recovery.items():
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "fold": fold,
                    "contrast": contrast_key,
                    "analysis": "defining_contrast_movement_effect_decomposition",
                    "scale": float(defining_contrast.scale_b),
                    "scale_a": float(defining_contrast.scale_a),
                    "scale_b": float(defining_contrast.scale_b),
                    "population": population,
                    "n_units": len(units),
                    "component": component,
                    "preactivation_movement_effect_recovery_r2": _population_recovery(
                        recovery["z"], units
                    ),
                    "rate_movement_effect_recovery_r2": _population_recovery(
                        recovery["rate"], units
                    ),
                    "normalized_map_movement_effect_recovery_r2": _population_recovery(
                        recovery["gain"], units
                    ),
                    "preactivation_movement_effect_recovery_r2_equal_unit_unweighted": _population_recovery_equal(
                        recovery["z"], units
                    ),
                    "rate_movement_effect_recovery_r2_equal_unit_unweighted": _population_recovery_equal(
                        recovery["rate"], units
                    ),
                    "normalized_map_movement_effect_recovery_r2_equal_unit_unweighted": _population_recovery_equal(
                        recovery["gain"], units
                    ),
                    "primary_map_recovery_weighting": "paired expected spikes",
                    "contrast_definition": (
                        f"literal h({defining_contrast.scale_a:g}x) + P/Q "
                        f"[h({defining_contrast.scale_b:g}x)-h({defining_contrast.scale_a:g}x)]"
                    ),
                }
            )

    # Absolute linear decomposition across held-out image means.  Centering is
    # across images at every unit and spatial position, so within-map spatial
    # structure is not mistaken for image-to-image variance.
    image_variance_by_scale: dict[float, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        scale_value = float(scale)
        p_stack = np.stack(image_contribution_means[scale_value]["p"], axis=0)
        q_stack = np.stack(image_contribution_means[scale_value]["q"], axis=0)
        p_delta = p_stack - p_stack.mean(axis=0, keepdims=True)
        q_delta = q_stack - q_stack.mean(axis=0, keepdims=True)
        var_p = np.square(p_delta).mean(axis=(0, 2, 3))
        var_q = np.square(q_delta).mean(axis=(0, 2, 3))
        covariance = (p_delta * q_delta).mean(axis=(0, 2, 3))
        decomposition = variance_decomposition(var_p, var_q, covariance)
        image_variance_by_scale[scale_value] = decomposition
        for population, units in populations.items():
            aggregate_var_p = float(np.nanmean(var_p[units]))
            aggregate_var_q = float(np.nanmean(var_q[units]))
            aggregate_covariance = float(np.nanmean(covariance[units]))
            aggregate = variance_decomposition(
                np.asarray([aggregate_var_p]),
                np.asarray([aggregate_var_q]),
                np.asarray([aggregate_covariance]),
            )
            aggregated = {key: float(value[0]) for key, value in aggregate.items()}
            robustness = {
                f"equal_unit_mean_of_ratios_{key}": float(np.nanmean(value[units]))
                for key, value in decomposition.items()
                if "fraction" in key or "reliance" in key
            }
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "fold": fold,
                    "contrast": contrast_key,
                    "analysis": "absolute_preactivation_variance_across_heldout_image_means",
                    "scale": scale_value,
                    "population": population,
                    "n_units": len(units),
                    "component": "bias_once_plus_a_p_plus_a_q",
                    **aggregated,
                    **robustness,
                    "primary_population_fraction_rule": "aggregate component variances/covariance across selected units, then form ratios",
                    "variance_definition": "variance across held-out image means at each unit and map position; averaged across positions and units",
                }
            )

    for scale_i, scale in enumerate(SCALES):
        scale_value = float(scale)
        for population, units in populations.items():
            full_rate = float(np.nanmean(absolute_rate_sum["full"][scale_i, units] / max(absolute_rate_count[scale_i], 1)))
            for component in ("candidate_p_absolute", "complementary_q_absolute"):
                rate = float(np.nanmean(absolute_rate_sum[component][scale_i, units] / max(absolute_rate_count[scale_i], 1)))
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "fold": fold,
                        "contrast": contrast_key,
                        "analysis": "absolute_component_readout",
                        "scale": scale_value,
                        "population": population,
                        "n_units": len(units),
                        "component": component,
                        "normalized_map_recovery_from_flat_r2": _standard_gain_recovery(
                            absolute_gain_residual[component][scale_i], absolute_gain_target[scale_i], units
                        ),
                        "mean_rate_hz": rate,
                        "mean_rate_ratio_to_full": _safe_divide(rate, full_rate),
                        "bias_rule": "readout bias included once in each absolute component ablation; excluded from a_P/a_Q variance",
                    }
                )

    var_p_activity, var_q_activity, cov_activity = activity.values()
    activity_decomposition = variance_decomposition(var_p_activity, var_q_activity, cov_activity)
    metadata = _unit_observational_metadata(fold)
    per_unit = metadata.copy()
    per_unit.insert(0, "schema_version", SCHEMA_VERSION)
    per_unit.insert(1, "fold", fold)
    per_unit.insert(2, "contrast", contrast_key)
    with np.load(READOUT_CACHE, allow_pickle=False) as archive:
        feature_weights = np.asarray(archive["feature_weights"], dtype=np.float64)
    per_unit["weight_based_candidate_p_reliance"] = weight_reliance(feature_weights, basis)
    for key, value in activity_decomposition.items():
        per_unit[f"activity_{key}"] = value
    per_unit["observed_ssi_benefit_for_projector_contrast_bits"] = _contrast_observed_benefit(
        metadata, contrast_key
    )
    for component, recovery in baseline_recovery.items():
        for metric, accumulator in recovery.items():
            per_unit[f"baseline_{component}_{metric}_recovery_r2"] = accumulator.recovery()
    for component in baseline_dose:
        for metric in ("z", "rate", "gain"):
            _, ratio, recovery = baseline_image_metrics[component][metric]
            per_unit[f"baseline_{component}_{metric}_image_variance_ratio"] = ratio
            per_unit[f"baseline_{component}_{metric}_centered_image_recovery_r2"] = recovery
    contrast_definition = next(value for value in CONTRASTS if value.key == contrast_key)
    per_unit["projector_defining_scale_a"] = float(contrast_definition.scale_a)
    per_unit["projector_defining_scale_b"] = float(contrast_definition.scale_b)
    for component, recovery in contrast_recovery.items():
        for metric, accumulator in recovery.items():
            per_unit[f"direct_a_to_b_{component}_{metric}_effect_recovery_r2"] = accumulator.recovery()
            per_unit[
                f"direct_a_to_b_{component}_{metric}_effect_recovery_r2_equal_unit_unweighted"
            ] = accumulator.recovery_equal_unit()

    baseline_component_axis = np.asarray(list(baseline_maps), dtype="U40")
    movement_component_axis = np.asarray(list(next(iter(movement_maps.values()))), dtype="U40")
    population_axis = np.asarray(list(populations), dtype="U20")
    baseline_map_array = np.stack(
        [_final_population_maps(baseline_maps[name]) for name in baseline_component_axis], axis=0
    )
    movement_map_array = np.stack(
        [
            np.stack(
                [_final_population_maps(movement_maps[float(scale)][name]) for name in movement_component_axis],
                axis=0,
            )
            for scale in SCALES
        ],
        axis=0,
    )
    readout_path = stage_dir / "pq_readout_decomposition.csv"
    unit_path = stage_dir / "per_unit_pq_reliance.csv"
    arrays_path = stage_dir / "pq_readout_supporting_arrays.npz"
    diagnostics_path = stage_dir / "readout_numerical_diagnostics.json"
    diagnostic_limits = {
        "p_plus_q_preactivation_relative_rmse_max": 5e-5,
        "state_replay_vs_cached_preactivation_relative_rmse_max": 1e-3,
        "state_replay_vs_cached_rate_relative_rmse_max": 1e-3,
    }
    diagnostics = {
        "p_plus_q_preactivation_relative_rmse_max": max(decomposition_errors, default=math.nan),
        "state_replay_vs_cached_preactivation_relative_rmse_max": max(cache_z_errors, default=math.nan),
        "state_replay_vs_cached_rate_relative_rmse_max": max(cache_rate_errors, default=math.nan),
    }
    failed_diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if not np.isfinite(value) or value > diagnostic_limits[key]
    }
    _atomic_json(
        diagnostics_path,
        {
            "schema_version": SCHEMA_VERSION,
            "fold": fold,
            "contrast": contrast_key,
            "no_core_replay": True,
            "readout": "exact cached 1x1 channel weights followed by per-unit 14x14 tiled spatial convolution and softplus",
            **diagnostics,
            "limits": diagnostic_limits,
            "passed": not failed_diagnostics,
            "failures": failed_diagnostics,
            "training_mean_state": str(h_mean_path),
            "training_mean_excludes_test": True,
            "n_test_pairs": len(test.pairs),
            "n_frames_per_pair_scale": N_FRAMES,
        },
    )
    if failed_diagnostics:
        raise RuntimeError(
            f"Fold {fold} {contrast_key}: exact readout diagnostic gate failed: {failed_diagnostics}"
        )
    _atomic_csv(readout_path, pd.DataFrame(rows))
    _atomic_csv(unit_path, per_unit)
    np.savez_compressed(
        arrays_path,
        scales=np.asarray(SCALES, dtype=np.float32),
        populations=population_axis,
        baseline_components=baseline_component_axis,
        baseline_expected_spike_weighted_mean_gain_maps=baseline_map_array,
        movement_components=movement_component_axis,
        movement_expected_spike_weighted_mean_gain_maps=movement_map_array,
        basis=basis,
        activity_variance_p=var_p_activity.astype(np.float32),
        activity_variance_q=var_q_activity.astype(np.float32),
        activity_covariance_pq=cov_activity.astype(np.float32),
    )
    _write_stage_complete(
        stage_dir,
        basis_file=basis_file,
        stage="readout",
        products=(readout_path, unit_path, arrays_path, diagnostics_path),
    )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        torch.cuda.empty_cache()
    return {"status": "completed", "fold": fold, "contrast": contrast_key, "path": str(stage_dir)}


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fixed-ridge probe with train-only standardization and an intercept."""
    x_train = np.asarray(train_x, dtype=np.float64)
    x_test = np.asarray(test_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Ridge feature matrices must be two-dimensional with matching columns")
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    mean_x = x_train.mean(axis=0)
    scale_x = x_train.std(axis=0)
    keep = scale_x > 1e-10
    if not np.any(keep):
        return np.broadcast_to(y_train.mean(axis=0), (len(x_test), y_train.shape[1])).copy()
    train = (x_train[:, keep] - mean_x[keep]) / scale_x[keep]
    test = (x_test[:, keep] - mean_x[keep]) / scale_x[keep]
    mean_y = y_train.mean(axis=0)
    centered_y = y_train - mean_y
    gram = train.T @ train
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64),
        train.T @ centered_y,
    )
    return test @ coefficients + mean_y


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.square(truth - truth.mean()).sum())
    return 1.0 - _safe_divide(float(np.square(pred - truth).sum()), denominator)


def _spatially_pooled_features(
    state: h5py.File,
    pairs: Sequence[tuple[int, int]],
    scales: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    image_position: list[np.ndarray] = []
    trajectory_position: list[np.ndarray] = []
    scale_values: list[np.ndarray] = []
    for image, trajectory in pairs:
        for scale in scales:
            value = _read_state(state, (image, trajectory), scale)
            features.append(value.mean(axis=(2, 3), dtype=np.float64).astype(np.float32))
            image_position.append(np.full(value.shape[0], image, dtype=np.int16))
            trajectory_position.append(np.full(value.shape[0], trajectory, dtype=np.int16))
            scale_values.append(np.full(value.shape[0], scale, dtype=np.float32))
    return (
        np.concatenate(features, axis=0),
        np.concatenate(image_position),
        np.concatenate(trajectory_position),
        np.concatenate(scale_values),
    )


def _split_pq_features(channel_features: np.ndarray, basis: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(channel_features, dtype=np.float64)
    u = np.asarray(validate_basis(basis), dtype=np.float64)
    coordinates = x @ u
    return {
        "candidate_p_coordinates_rank8": coordinates,
        "complementary_q_reconstructed_channels_rank120": x - coordinates @ u.T,
    }


def _movement_targets(
    trajectory_positions: np.ndarray,
    scales: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(READOUT_CACHE, allow_pickle=False) as archive:
        trajectory_ids = np.asarray(archive["trajectory_ids"], dtype=np.int64)
    with np.load(TRAJECTORY_BANK, allow_pickle=False) as archive:
        trace_xy = np.asarray(archive["stored_scored_trace_xy"], dtype=np.float64)
    table = pd.read_csv(TRAJECTORY_TABLE).sort_values("trajectory_id")
    path_lookup = table.set_index("trajectory_id")["rendered_path_length_arcmin"]
    frames = np.tile(np.arange(N_FRAMES, dtype=np.int64), len(scales) // N_FRAMES)
    source_ids = trajectory_ids[np.asarray(trajectory_positions, dtype=np.int64)]
    eye_xy = trace_xy[source_ids, frames]
    scale = np.asarray(scales, dtype=np.float64)
    # Retinal image motion has the opposite sign to eye position.  The cached
    # trajectory coordinates are in degrees and already centered exactly as
    # used by the corrected renderer.
    retinal_xy = -eye_xy * scale[:, None]
    paths = np.asarray([float(path_lookup.loc[int(value)]) for value in source_ids]) * scale
    return {
        "movement_scale": scale,
        "trajectory_path_length_arcmin": paths,
        "retinal_x_displacement_deg": retinal_xy[:, 0],
        "retinal_y_displacement_deg": retinal_xy[:, 1],
    }


def _probe_feature_cache(fold: int, *, overwrite: bool = False) -> tuple[dict[str, np.ndarray], Path]:
    """Cache spatially pooled state features once per fold, never per projector."""
    destination = INTERMEDIATE / f"fold_{fold}" / "probe_spatially_pooled_features.npz"
    metadata = destination.with_suffix(".json")
    required = {
        "train_x",
        "train_image",
        "train_trajectory",
        "train_scale",
        "test_x",
        "test_image",
        "test_trajectory",
        "test_scale",
    }

    def load_valid() -> dict[str, np.ndarray] | None:
        if not destination.is_file() or not metadata.is_file() or overwrite:
            return None
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
            if not (
                value.get("schema_version") == SCHEMA_VERSION
                and value.get("state_cache_fingerprint") == _input_fingerprint()["state_cache"]
                and value.get("fold_assignments_sha256")
                == _input_fingerprint()["fold_assignments_sha256"]
                and value.get("product_sha256") == sha256_file(destination)
            ):
                return None
            with np.load(destination, allow_pickle=False) as archive:
                if required != set(archive.files):
                    return None
                result = {name: np.asarray(archive[name]) for name in archive.files}
            if result["train_x"].ndim != 2 or result["test_x"].ndim != 2:
                return None
            if result["train_x"].shape[1] != N_CHANNELS or result["test_x"].shape[1] != N_CHANNELS:
                return None
            if not np.isfinite(result["train_x"]).all() or not np.isfinite(result["test_x"]).all():
                return None
            return result
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    cached = load_valid()
    if cached is not None:
        return cached, destination

    train_pairs = _training_pairs(fold)
    test_pairs = load_fold(fold).test.pairs
    with h5py.File(STATE_CACHE, "r") as state:
        train = _spatially_pooled_features(state, train_pairs, [float(value) for value in SCALES])
        test = _spatially_pooled_features(state, test_pairs, [float(value) for value in SCALES])
    result = {
        "train_x": train[0],
        "train_image": train[1],
        "train_trajectory": train[2],
        "train_scale": train[3],
        "test_x": test[0],
        "test_image": test[1],
        "test_trajectory": test[2],
        "test_scale": test[3],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result)
    os.replace(temporary, destination)
    _atomic_json(
        metadata,
        {
            "schema_version": SCHEMA_VERSION,
            "fold": fold,
            "definition": "spatial mean of saved ConvGRU state for fixed crossed probe splits",
            "state_cache_fingerprint": _input_fingerprint()["state_cache"],
            "fold_assignments_sha256": _input_fingerprint()["fold_assignments_sha256"],
            "product_sha256": sha256_file(destination),
            "train_pairs": [list(pair) for pair in train_pairs],
            "test_pairs": [list(pair) for pair in test_pairs],
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in result.items()},
        },
    )
    return result, destination


def run_probe_stage(
    fold: int,
    contrast_key: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Optional descriptive ridge probes with scientifically valid splits."""
    basis_file = basis_path(contrast_key, fold)
    if not basis_file.is_file():
        raise FileNotFoundError(f"Fold-specific rank-8 basis is missing: {basis_file}")
    stage_dir = _fit_dir(fold, contrast_key) / "probes"
    basis_hash = sha256_file(basis_file)
    if not overwrite and _stage_done(stage_dir, basis_hash):
        return {"status": "reused", "fold": fold, "contrast": contrast_key, "path": str(stage_dir)}
    stage_dir.mkdir(parents=True, exist_ok=True)
    stale_marker = stage_dir / "complete.json"
    if stale_marker.exists():
        stale_marker.unlink()
    basis = validate_basis(np.load(basis_file, allow_pickle=False))
    fold_value = load_fold(fold)
    rows: list[dict[str, Any]] = []
    cached, feature_cache_path = _probe_feature_cache(fold, overwrite=False)
    train_x = cached["train_x"]
    test_x = cached["test_x"]
    train_targets = _movement_targets(cached["train_trajectory"], cached["train_scale"])
    test_targets = _movement_targets(cached["test_trajectory"], cached["test_scale"])
    for representation, x_train in _split_pq_features(train_x, basis).items():
        x_test = _split_pq_features(test_x, basis)[representation]
        train_y = np.column_stack([train_targets[name] for name in train_targets])
        prediction = ridge_predict(x_train, train_y, x_test, alpha=1.0)
        for target_i, target_name in enumerate(train_targets):
            truth = test_targets[target_name]
            pred = prediction[:, target_i]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "fold": fold,
                    "contrast": contrast_key,
                    "probe": target_name,
                    "representation": representation,
                    "split": "crossed_outer_train_to_unseen_test_images_and_trajectories",
                    "n_train_records": len(train_x),
                    "n_test_records": len(test_x),
                    "ridge_alpha_fixed": 1.0,
                    "metric": "r2",
                    "value": _r2(truth, pred),
                    "pearson_r": _correlation(truth, pred, "pearson"),
                    "causal_interpretation_allowed": False,
                }
            )

    # Unseen-class image identification is undefined. Use only the projector-
    # held-out images and disjoint held-out trajectory halves at 1x.
    held_trajectories = sorted(fold_value.test.trajectory_positions)
    identity_train_trajectories = set(held_trajectories[::2])
    identity_test_trajectories = set(held_trajectories[1::2])
    at_one = np.isclose(cached["test_scale"], 1.0)
    identity_train = at_one & np.isin(cached["test_trajectory"], list(identity_train_trajectories))
    identity_test = at_one & np.isin(cached["test_trajectory"], list(identity_test_trajectories))
    identity_train_x = test_x[identity_train]
    identity_test_x = test_x[identity_test]
    identity_train_image = cached["test_image"][identity_train]
    identity_test_image = cached["test_image"][identity_test]
    image_axis = np.asarray(sorted(fold_value.test.image_positions), dtype=np.int64)
    train_labels = np.column_stack([identity_train_image == value for value in image_axis]).astype(float)
    test_class = np.searchsorted(image_axis, identity_test_image)
    for representation, x_train in _split_pq_features(identity_train_x, basis).items():
        x_test = _split_pq_features(identity_test_x, basis)[representation]
        prediction = ridge_predict(x_train, train_labels, x_test, alpha=1.0)
        accuracy = float(np.mean(np.argmax(prediction, axis=1) == test_class))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "fold": fold,
                "contrast": contrast_key,
                "probe": "image_identity_within_heldout_image_set",
                "representation": representation,
                "split": "projector-heldout images; probe train/test use disjoint heldout trajectories at 1x",
                "n_train_records": len(identity_train_x),
                "n_test_records": len(identity_test_x),
                "n_image_classes": len(image_axis),
                "ridge_alpha_fixed": 1.0,
                "metric": "classification_accuracy",
                "value": accuracy,
                "unseen_image_identity_generalization_claimed": False,
                "causal_interpretation_allowed": False,
            }
        )

    path = stage_dir / "pq_descriptive_probes.csv"
    _atomic_csv(path, pd.DataFrame(rows))
    feature_reference = stage_dir / "probe_feature_cache_reference.json"
    _atomic_json(
        feature_reference,
        {
            "schema_version": SCHEMA_VERSION,
            "path": str(feature_cache_path),
            "sha256": sha256_file(feature_cache_path),
            "shared_across_projectors_within_fold": True,
        },
    )
    _write_stage_complete(
        stage_dir, basis_file=basis_file, stage="probes", products=(path, feature_reference)
    )
    return {"status": "completed", "fold": fold, "contrast": contrast_key, "path": str(stage_dir)}


def run_readout_requests(
    folds: Sequence[int],
    contrast_keys: Sequence[str],
    *,
    device: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Run readout requests with shared locking/accounting for CUDA only."""
    global _GPU_DEADLINE_MONOTONIC
    requests = [(int(fold), str(contrast)) for fold in folds for contrast in contrast_keys]
    results: list[dict[str, Any]] = []

    def execute() -> list[dict[str, Any]]:
        for fold, contrast in requests:
            results.append(
                run_readout_stage(fold, contrast, device=device, overwrite=overwrite)
            )
        return results

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return execute()

    before = load_budget()
    if float(before["remaining_gpu_hours"]) <= 0.0:
        raise RuntimeError(
            "The cumulative ten-GPU-hour limit has been reached; write the required interim report"
        )
    started = time.monotonic()
    caught: BaseException | None = None
    with exclusive_gpu_lock():
        _GPU_DEADLINE_MONOTONIC = budget_deadline()
        try:
            execute()
        except BaseException as error:  # debit attempted CUDA time before re-raising
            caught = error
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize(torch_device)
            elapsed = time.monotonic() - started
            record_gpu_time(
                "registration_pq_saved_cache_readout",
                elapsed,
                {
                    "device": str(torch_device),
                    "requests": [[fold, contrast] for fold, contrast in requests],
                    "completed_requests": [
                        [int(value["fold"]), str(value["contrast"])] for value in results
                    ],
                    "no_core_replay": True,
                    "error": None if caught is None else repr(caught),
                },
            )
            _GPU_DEADLINE_MONOTONIC = math.inf
    if caught is not None:
        raise caught
    return results


def _basis_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(4):
        for contrast in CONTRASTS:
            path = basis_path(contrast.key, fold)
            if path.is_file():
                basis = validate_basis(np.load(path, allow_pickle=False))
                rows.append(
                    {
                        "fold": fold,
                        "contrast": contrast.key,
                        "path": path,
                        "sha256": sha256_file(path),
                        "basis": basis,
                        "leverage": leverage_scores(basis),
                    }
                )
    return rows


def _leverage_overlap_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = _basis_inventory()
    summary_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []

    def add_pair(left: dict[str, Any], right: dict[str, Any], scope: str) -> None:
        u_left = np.asarray(left["basis"], dtype=np.float64)
        u_right = np.asarray(right["basis"], dtype=np.float64)
        l_left = np.asarray(left["leverage"], dtype=np.float64)
        l_right = np.asarray(right["leverage"], dtype=np.float64)
        overlap = float(np.square(u_left.T @ u_right).sum() / RANK)
        cosine = _safe_divide(
            float(l_left @ l_right),
            float(np.linalg.norm(l_left) * np.linalg.norm(l_right)),
        )
        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "comparison_scope": scope,
                "fold_a": left["fold"],
                "contrast_a": left["contrast"],
                "fold_b": right["fold"],
                "contrast_b": right["contrast"],
                "projector_overlap_trace_papb_over_rank": overlap,
                "leverage_cosine_similarity": cosine,
                "leverage_spearman_correlation": _correlation(l_left, l_right, "spearman"),
                "leverage_mass_intersection_fraction": float(np.minimum(l_left, l_right).sum() / RANK),
                "threshold_free_note": "No channel threshold is used; see the complete top-k overlap curve for k=1..128",
            }
        )
        order_left = np.argsort(-l_left, kind="stable")
        order_right = np.argsort(-l_right, kind="stable")
        for k in range(1, N_CHANNELS + 1):
            shared = len(set(order_left[:k].tolist()) & set(order_right[:k].tolist()))
            topk_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "comparison_scope": scope,
                    "fold_a": left["fold"],
                    "contrast_a": left["contrast"],
                    "fold_b": right["fold"],
                    "contrast_b": right["contrast"],
                    "top_k": k,
                    "shared_native_channels": shared,
                    "shared_fraction_of_top_k": shared / k,
                    "random_expected_shared_channels": k * k / N_CHANNELS,
                }
            )

    for fold in range(4):
        values = [row for row in inventory if row["fold"] == fold]
        for left_i in range(len(values)):
            for right_i in range(left_i + 1, len(values)):
                add_pair(values[left_i], values[right_i], "within_fold_across_contrasts")
    for contrast in [value.key for value in CONTRASTS]:
        values = [row for row in inventory if row["contrast"] == contrast]
        for left_i in range(len(values)):
            for right_i in range(left_i + 1, len(values)):
                add_pair(values[left_i], values[right_i], "within_contrast_across_folds")
    return pd.DataFrame(summary_rows), pd.DataFrame(topk_rows)


def _valid_stage_marker(stage_dir: Path, *, stage: str, expected_products: Sequence[str]) -> dict[str, Any] | None:
    marker = stage_dir / "complete.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    parts = stage_dir.parts
    fold = int(parts[-3].split("_")[-1])
    contrast = parts[-2]
    basis_file = basis_path(contrast, fold)
    if not basis_file.is_file() or not _stage_done(stage_dir, sha256_file(basis_file)):
        return None
    if value.get("stage") != stage:
        return None
    recorded = {Path(item["path"]).name for item in value["products"]}
    if not set(expected_products).issubset(recorded):
        return None
    return value


def _collect_partials(
    filename: str, stage: str, *, expected_products: Sequence[str]
) -> tuple[pd.DataFrame, list[Path], set[tuple[int, str]]]:
    paths: list[Path] = []
    cells: set[tuple[int, str]] = set()
    for stage_dir in sorted(INTERMEDIATE.glob(f"fold_*/*/{stage}")):
        marker = _valid_stage_marker(stage_dir, stage=stage, expected_products=expected_products)
        if marker is None:
            continue
        path = stage_dir / filename
        if path.is_file():
            paths.append(path)
            cells.add((int(stage_dir.parts[-3].split("_")[-1]), stage_dir.parts[-2]))
    tables = [pd.read_csv(path) for path in paths]
    return (
        pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame(),
        paths,
        cells,
    )


def _compact_array_manifest(array_paths: Sequence[Path]) -> dict[str, Any]:
    staging = OUT / f".pq_supporting_arrays.tmp.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    products: list[dict[str, Any]] = []
    for source in array_paths:
        relative = source.relative_to(INTERMEDIATE)
        name = "__".join(relative.parts[:-1]) + "__" + source.name
        destination = staging / name
        try:
            os.link(source, destination)
            copy_mode = "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            copy_mode = "copy"
        with np.load(destination, allow_pickle=False) as archive:
            arrays = {
                key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
                for key in archive.files
            }
        products.append(
            {
                "path": str(destination),
                "source": str(source),
                "copy_mode": copy_mode,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "arrays": arrays,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": "Compact bases, leverage vectors, per-unit activity moments, and population-mean P/Q dose maps; no individual condition is independently normalized",
        "products": products,
    }
    _atomic_json(staging / "manifest.json", manifest)
    if ARRAYS.exists():
        backup = OUT / f".pq_supporting_arrays.old.{os.getpid()}"
        os.replace(ARRAYS, backup)
        os.replace(staging, ARRAYS)
        shutil.rmtree(backup)
    else:
        os.replace(staging, ARRAYS)
    for product in manifest["products"]:
        product["path"] = str(ARRAYS / Path(product["path"]).name)
    _atomic_json(ARRAYS / "manifest.json", manifest)
    return manifest


def consolidate(*, require_all_folds: bool = False) -> dict[str, Any]:
    """Create the four canonical CSVs and a provenance/completeness manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    variance_expected = (
        "native_channel_leverage.csv",
        "pq_variance_decomposition.csv",
        "variance_supporting_arrays.npz",
    )
    readout_expected = (
        "pq_readout_decomposition.csv",
        "per_unit_pq_reliance.csv",
        "pq_readout_supporting_arrays.npz",
        "readout_numerical_diagnostics.json",
    )
    probe_expected = ("pq_descriptive_probes.csv", "probe_feature_cache_reference.json")
    leverage, leverage_paths, leverage_cells = _collect_partials(
        "native_channel_leverage.csv", "variance", expected_products=variance_expected
    )
    variance, variance_paths, variance_cells = _collect_partials(
        "pq_variance_decomposition.csv", "variance", expected_products=variance_expected
    )
    readout, readout_paths, readout_cells = _collect_partials(
        "pq_readout_decomposition.csv", "readout", expected_products=readout_expected
    )
    per_unit, per_unit_paths, per_unit_cells = _collect_partials(
        "per_unit_pq_reliance.csv", "readout", expected_products=readout_expected
    )
    probes, probe_paths, probe_cells = _collect_partials(
        "pq_descriptive_probes.csv", "probes", expected_products=probe_expected
    )
    expected = {(fold, contrast.key) for fold in range(4) for contrast in CONTRASTS}
    inventory = _basis_inventory()
    available = {(int(row["fold"]), str(row["contrast"])) for row in inventory}
    missing_basis = sorted(expected - available)
    missing_variance = sorted(expected - (leverage_cells & variance_cells))
    missing_readout = sorted(expected - (readout_cells & per_unit_cells))
    missing_probes = sorted(expected - probe_cells)
    complete = not missing_basis and not missing_variance and not missing_readout and not missing_probes
    if require_all_folds and not complete:
        raise RuntimeError(
            "P/Q semantic analysis is incomplete; no canonical products were published: "
            f"missing_basis={missing_basis}, missing_variance={missing_variance}, "
            f"missing_readout={missing_readout}, missing_probes={missing_probes}"
        )

    products = (
        (LEVERAGE_CSV, leverage),
        (VARIANCE_CSV, variance),
        (READOUT_CSV, readout),
        (PER_UNIT_CSV, per_unit),
    )
    for path, table in products:
        if not table.empty:
            sort_columns = [
                name
                for name in ("fold", "contrast", "analysis", "scale", "population", "component", "unit_index", "native_channel")
                if name in table.columns
            ]
            if sort_columns:
                table = table.sort_values(sort_columns, kind="stable")
            _atomic_csv(path, table)
        elif path.exists():
            path.unlink()
    if not probes.empty:
        _atomic_csv(
            PROBES_CSV,
            probes.sort_values(["fold", "contrast", "probe", "representation"], kind="stable"),
        )
    elif PROBES_CSV.exists():
        PROBES_CSV.unlink()

    overlap, topk = _leverage_overlap_tables()
    if not overlap.empty:
        _atomic_csv(LEVERAGE_OVERLAP_CSV, overlap)
    elif LEVERAGE_OVERLAP_CSV.exists():
        LEVERAGE_OVERLAP_CSV.unlink()
    if not topk.empty:
        _atomic_csv(LEVERAGE_TOPK_CSV, topk)
    elif LEVERAGE_TOPK_CSV.exists():
        LEVERAGE_TOPK_CSV.unlink()

    # CSV paths were admitted only through fully validated, hash-checked stage
    # markers.  Derive array siblings from those exact validated stages rather
    # than raw-globbing the filesystem, which could publish a failed/stale NPZ.
    array_paths = sorted({path.parent / "variance_supporting_arrays.npz" for path in variance_paths})
    array_paths += sorted({path.parent / "pq_readout_supporting_arrays.npz" for path in readout_paths})
    array_manifest = _compact_array_manifest(array_paths)

    def cache_info(path: Path) -> dict[str, Any]:
        return {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "fold-wise held-out P/Q semantics and exact tiled RR100 readout decomposition",
        "complete_all_four_folds": complete,
        "status": "complete" if complete else "partial_waiting_for_rank8_folds",
        "no_core_replay": True,
        "model_imported": False,
        "candidate_subspace": "P=UU^T, rank 8, same channel projector at every spatial position",
        "complementary_subspace": "Q=I-P, rank 120; never described as other channels",
        "fold_semantics": "every reported semantic/readout metric uses only that projector fold's 2-image x 6-trajectory crossed test block",
        "baseline_mean_semantics": "h_mean uses fold train+validation stabilization records and excludes every test pair",
        "readout_semantics": "cached frozen feature weights, per-unit 14x14 tiled spatial kernels, one bias, softplus, and exact SSI",
        "energy_semantics": {
            "total_fraction": "candidate energy divided by candidate+complementary energy",
            "per_dimension": "candidate energy/8 and complementary energy/120; never conflated with total fraction",
            "content": "held-out stabilization deviations from a channel-and-position-resolved grand mean",
            "image_means": "held-out image means over trajectory and frame before across-image variance",
            "trajectory": "across-trajectory variance at fixed held-out image, scale, frame, and spatial position",
        },
        "absolute_readout_bias_rule": "a_P and a_Q exclude bias; complete preactivation adds bias exactly once",
        "available_projectors": [
            {key: _json_ready(value) for key, value in row.items() if key not in ("basis", "leverage")}
            for row in inventory
        ],
        "missing_basis_fold_contrasts": missing_basis,
        "missing_variance_fold_contrasts": missing_variance,
        "missing_readout_fold_contrasts": missing_readout,
        "missing_probe_fold_contrasts": missing_probes,
        "inputs": {
            "states": cache_info(STATE_CACHE),
            "maps": cache_info(MAP_CACHE),
            "readout": {**cache_info(READOUT_CACHE), "sha256": sha256_file(READOUT_CACHE)},
            "fold_assignments": str(LOW_RANK_OUT / "config/fold_assignments.json"),
        },
        "canonical_products": {
            str(path.name): {
                "path": str(path),
                "rows": int(len(table)),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for path, table in products
        },
        "descriptive_probe_product": {
            "path": str(PROBES_CSV),
            "exists": PROBES_CSV.is_file(),
            "rows": int(len(probes)),
            "status": "required descriptive output; never interpreted causally",
        },
        "supporting_array_manifest": str(ARRAYS / "manifest.json"),
        "n_supporting_array_products": len(array_manifest["products"]),
        "partial_sources": [
            str(path)
            for path in leverage_paths + variance_paths + readout_paths + per_unit_paths + probe_paths
        ],
        "scientific_boundaries": [
            "P is a channel subspace, not eight native channels.",
            "Readout relevance is not movement specificity.",
            "The descriptive variance and reliance results do not establish recurrent registration.",
            "No unique native-channel circuit is selected by leverage.",
        ],
    }
    _atomic_json(MANIFEST, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("variance", "readout", "probes", "consolidate", "all"), required=True
    )
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--contrasts", nargs="*", default=[value.key for value in CONTRASTS])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-all-folds", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    contrast_keys = list(args.contrasts)
    valid_keys = {value.key for value in CONTRASTS}
    invalid = sorted(set(contrast_keys) - valid_keys)
    if invalid:
        raise ValueError(f"Unknown contrasts: {invalid}")
    if args.stage != "consolidate":
        _assert_cache_gate()
        folds = list(args.folds) if args.folds is not None else available_folds(contrast_keys)
        if not folds:
            raise RuntimeError("No requested fold has all requested canonical rank-8 bases")
        if args.require_all_folds and folds != [0, 1, 2, 3]:
            raise RuntimeError(f"All four folds were required, but resolved folds are {folds}")
        for fold in folds:
            for contrast_key in contrast_keys:
                if args.stage in ("variance", "all"):
                    print(json.dumps(run_variance_stage(fold, contrast_key, overwrite=args.overwrite)))
                if args.stage in ("probes", "all"):
                    print(json.dumps(run_probe_stage(fold, contrast_key, overwrite=args.overwrite)))
        if args.stage in ("readout", "all"):
            for result in run_readout_requests(
                folds,
                contrast_keys,
                device=args.device,
                overwrite=args.overwrite,
            ):
                print(json.dumps(result))
    if args.stage in ("consolidate", "all"):
        print(json.dumps(_json_ready(consolidate(require_all_folds=args.require_all_folds)), indent=2))


if __name__ == "__main__":
    main()
