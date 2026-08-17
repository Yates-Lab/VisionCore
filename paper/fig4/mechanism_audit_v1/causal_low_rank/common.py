from __future__ import annotations

import hashlib
import json
import math
import os
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1"
CACHE = OUT / "cache"
CONFIG = OUT / "config"
FITS = OUT / "fits"
EVALUATION = OUT / "evaluation"
FIGURES = OUT / "figures"
FIGURE_DATA = OUT / "figure_data"
GLOBAL_GPU_BUDGET = OUT / "gpu_budget.json"
GLOBAL_GPU_RUN_LOCK = OUT / "gpu_budget.run.lock"
MAX_AUTHORIZED_GPU_HOURS = 10.0

SOURCE_EXACT = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
    / "exact_arrays/exact_phase_spatial_metrics.npz"
)
SOURCE_SELECTION = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
    / "selection"
)
STATE_CACHE = CACHE / "convgru_states.h5"
MAP_CACHE = CACHE / "rr100_maps.h5"
READOUT_CACHE = CACHE / "readout_weights.npz"

SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
N_IMAGES = 8
N_TRAJECTORIES = 24
N_SCALES = 5
N_FRAMES = 40
N_CHANNELS = 128
STATE_SIZE = 64
N_UNITS = 100
MAP_SIZE = 51
BIN_SECONDS = 1.0 / 120.0
EPS = 1e-8
ANALYSIS_SEED = 20260812


@dataclass(frozen=True)
class Contrast:
    key: str
    label: str
    group: str
    scale_a: float
    scale_b: float


CONTRASTS: tuple[Contrast, ...] = (
    Contrast("low_0_to_2", "Lower-SF sharpening", "low", 0.0, 2.0),
    Contrast("high_0_to_1", "Higher-SF sharpening", "high", 0.0, 1.0),
    Contrast("high_1_to_3", "Higher-SF reversal", "high", 1.0, 3.0),
)
CONTRAST_BY_KEY = {value.key: value for value in CONTRASTS}


def ensure_output_dirs() -> None:
    for path in (OUT, CACHE, CONFIG, FITS, EVALUATION, FIGURES, FIGURE_DATA):
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
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def _cache_accelerator_hours() -> float:
    manifest = CACHE / "cache_generation_manifest.json"
    if not manifest.is_file():
        return 0.0
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        return float(value.get("accelerator_forward_hours", 0.0))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        # Failing open here would erase already-spent cache-generation time
        # from the analysis-wide cap.  A damaged provenance file therefore
        # requires explicit repair rather than silently granting more GPU time.
        raise RuntimeError(
            f"Cannot read accelerator time from cache manifest: {manifest}"
        ) from error


def load_global_gpu_budget(hard_limit_hours: float = 4.0) -> dict[str, Any]:
    """Load the one analysis-wide accelerator ledger, including legacy fits."""
    cache_hours = _cache_accelerator_hours()
    if GLOBAL_GPU_BUDGET.is_file():
        try:
            value = json.loads(GLOBAL_GPU_BUDGET.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            # A resume-safe hard cap must fail closed: resetting a corrupt
            # ledger to zero could authorize more than four GPU-hours.
            raise RuntimeError(
                f"Cannot read global GPU budget ledger: {GLOBAL_GPU_BUDGET}"
            ) from error
    else:
        value = {}
    # Before the global ledger existed, optimizer wall time was stored under
    # this narrower field.  Promote it exactly once without losing completed
    # work when an analysis resumes across code versions.
    stage_hours = float(
        value.get("gpu_stage_wall_hours", value.get("optimization_wall_hours", 0.0))
    )
    result = dict(value)
    result.update(
        {
            "schema_version": "fig4-global-gpu-budget-v1",
            "hard_limit_hours": min(float(hard_limit_hours), MAX_AUTHORIZED_GPU_HOURS),
            "cache_accelerator_hours": max(
                cache_hours, float(value.get("cache_accelerator_hours", 0.0))
            ),
            "gpu_stage_wall_hours": stage_hours,
        }
    )
    result.setdefault("events", [])
    result["total_conservative_gpu_hours"] = (
        result["cache_accelerator_hours"] + result["gpu_stage_wall_hours"]
    )
    # Recompute this derived field after every legacy or overnight event.
    # Keeping a value written by an earlier event would overstate the budget
    # remaining once another process appends accelerator time.
    result["remaining_gpu_hours"] = max(
        result["hard_limit_hours"] - result["total_conservative_gpu_hours"], 0.0
    )
    return result


def record_global_gpu_time(
    stage: str,
    elapsed_seconds: float,
    *,
    hard_limit_hours: float = 4.0,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically debit completed/attempted GPU wall time from the shared cap."""
    ensure_output_dirs()
    lock_path = GLOBAL_GPU_BUDGET.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = load_global_gpu_budget(hard_limit_hours)
        elapsed_hours = max(float(elapsed_seconds), 0.0) / 3600.0
        value["gpu_stage_wall_hours"] += elapsed_hours
        value["events"].append(
            {
                "stage": str(stage),
                "wall_hours": elapsed_hours,
                "details": details or {},
            }
        )
        if str(stage).startswith("optimization"):
            value["optimization_wall_hours"] = float(
                value.get("optimization_wall_hours", 0.0)
            ) + elapsed_hours
        value["total_conservative_gpu_hours"] = (
            value["cache_accelerator_hours"] + value["gpu_stage_wall_hours"]
        )
        value["remaining_gpu_hours"] = max(
            value["hard_limit_hours"] - value["total_conservative_gpu_hours"], 0.0
        )
        value["status"] = (
            "stopped_at_budget"
            if value["total_conservative_gpu_hours"] >= value["hard_limit_hours"]
            else "within_budget"
        )
        temporary = GLOBAL_GPU_BUDGET.with_name(
            f"{GLOBAL_GPU_BUDGET.name}.tmp.{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, GLOBAL_GPU_BUDGET)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return value


@contextmanager
def exclusive_gpu_analysis_lock() -> Iterator[None]:
    """Serialize all post-cache GPU stages that share the four-hour ledger."""
    ensure_output_dirs()
    with GLOBAL_GPU_RUN_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another causal-low-rank GPU stage holds {GLOBAL_GPU_RUN_LOCK}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def scale_index(scale: float) -> int:
    matches = np.flatnonzero(np.isclose(SCALES, float(scale)))
    if len(matches) != 1:
        raise KeyError(scale)
    return int(matches[0])


def qr_basis(a: torch.Tensor) -> torch.Tensor:
    """Return the reduced, sign-canonical QR basis used by every learned fit."""
    q, r = torch.linalg.qr(a, mode="reduced")
    sign = torch.where(torch.diagonal(r) < 0, -torch.ones_like(torch.diagonal(r)), torch.ones_like(torch.diagonal(r)))
    return q * sign.unsqueeze(0)


def project_channel_delta(delta_h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Apply P=UU^T identically at every sample and spatial position."""
    coordinates = torch.einsum("ck,bcyx->bkyx", u, delta_h)
    return torch.einsum("ck,bkyx->bcyx", u, coordinates)


def readout_preactivation(
    h: torch.Tensor,
    feature_weights: torch.Tensor,
    bias: torch.Tensor,
    space_weights: torch.Tensor,
) -> torch.Tensor:
    feature_map = F.conv2d(h, feature_weights[:, :, None, None])
    spatial = F.conv2d(feature_map, space_weights[:, None], groups=len(feature_weights), padding="valid")
    return spatial + bias[None, :, None, None]


def projected_delta_preactivation(
    delta_h: torch.Tensor,
    u: torch.Tensor,
    feature_weights: torch.Tensor,
    space_weights: torch.Tensor,
) -> torch.Tensor:
    """Algebraically split readout of ``P delta_h``, excluding readout bias.

    This is useful as a floating-point diagnostic, but it is not the
    production intervention path.  On CUDA, evaluating ``R(h) + R(P delta)``
    can differ measurably from evaluating ``R(h + P delta)`` because the two
    expressions use different convolution and accumulation orderings.  The
    latter is the literal state intervention specified by the analysis.
    """
    projected_delta = project_channel_delta(delta_h, u)
    feature_delta = F.conv2d(projected_delta, feature_weights[:, :, None, None])
    return F.conv2d(feature_delta, space_weights[:, None], groups=len(feature_weights), padding="valid")


def intervention_preactivations(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    u: torch.Tensor,
    feature_weights: torch.Tensor,
    bias: torch.Tensor,
    space_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact frozen readout on literal sufficient/necessary states.

    The shared channel projector is constructed once, then the complete
    patched states are passed through the same tiled readout used for intact
    states.  Keeping the state addition inside the readout call is important:
    an algebraically split preactivation path has a different floating-point
    operation order on CUDA and is therefore not numerically exact.
    """
    return intervention_preactivations_from_delta(
        h_a, h_b, h_b - h_a, u, feature_weights, bias, space_weights
    )


def intervention_preactivations_from_delta(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    delta_h: torch.Tensor,
    u: torch.Tensor,
    feature_weights: torch.Tensor,
    bias: torch.Tensor,
    space_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal state intervention with an explicitly supplied paired delta.

    The ordinary causal fit supplies ``h_b-h_a``.  The shuffled-target null
    instead supplies another image--trajectory pair's delta while retaining
    the recipient and donor anchors of the target pair.
    """
    projected_delta = project_channel_delta(delta_h, u)
    z_sufficiency = readout_preactivation(
        h_a + projected_delta,
        feature_weights,
        bias,
        space_weights,
    )
    z_necessity = readout_preactivation(
        h_b - projected_delta,
        feature_weights,
        bias,
        space_weights,
    )
    return z_sufficiency, z_necessity


def softplus_rate(preactivation: torch.Tensor) -> torch.Tensor:
    return F.softplus(preactivation)


def normalize_rate(rate: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return rate / rate.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))


def rate_map_components(rate: torch.Tensor, eps: float = EPS) -> dict[str, torch.Tensor]:
    """Exact Figure 4 per-map mean rate, expected spikes, normalized map, and SSI."""
    value = rate.clamp_min(0)
    mean_rate = value.mean(dim=(-2, -1))
    gain = value / mean_rate[..., None, None].clamp_min(float(eps))
    ssi = (gain * torch.log2(gain + float(eps))).mean(dim=(-2, -1))
    return {
        "mean_rate": mean_rate,
        "expected_spikes": mean_rate * BIN_SECONDS,
        "gain": gain,
        "ssi": ssi,
    }


def paired_expected_weight(expected_a: torch.Tensor, expected_b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (expected_a + expected_b)


def weighted_sse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    unit_weight: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target).square() * unit_weight[..., None, None]).sum()


def aggregate_ssi(ssi: np.ndarray, expected: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    numer = np.sum(np.asarray(ssi, dtype=np.float64) * np.asarray(expected, dtype=np.float64), axis=axis)
    denom = np.sum(np.asarray(expected, dtype=np.float64), axis=axis)
    return np.divide(numer, np.maximum(denom, 1e-12))


def orthonormalize_numpy(value: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(value, dtype=np.float64), mode="reduced")
    sign = np.where(np.diag(r) < 0, -1.0, 1.0)
    return (q * sign[None]).astype(np.float32)


def haar_basis(rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return orthonormalize_numpy(rng.standard_normal((N_CHANNELS, int(rank))))


def projector(u: np.ndarray) -> np.ndarray:
    basis = np.asarray(u, dtype=np.float64)
    return (basis @ basis.T).astype(np.float32)


def subspace_overlap(u_a: np.ndarray, u_b: np.ndarray) -> float:
    a = np.asarray(u_a, dtype=np.float64)
    b = np.asarray(u_b, dtype=np.float64)
    denominator = min(a.shape[1], b.shape[1])
    return float(np.square(a.T @ b).sum() / max(denominator, 1))


def principal_angles_deg(u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(np.asarray(u_a).T @ np.asarray(u_b), compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def iter_pair_indices(image_positions: Iterable[int], trajectory_positions: Iterable[int]) -> list[tuple[int, int]]:
    return [(int(i), int(j)) for i in image_positions for j in trajectory_positions]
