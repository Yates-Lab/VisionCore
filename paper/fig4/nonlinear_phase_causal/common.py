"""Shared algebra and exact spatial metrics for the nonlinear phase audit."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1"
RAW = OUT / "exact_arrays"
PARTS = RAW / "parts"
DATA = OUT / "plot_data"
DIAGNOSTICS = OUT / "diagnostics"

SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
N_SCORED = 40
N_TRACES = 24
EXAMPLE_IMAGE = 15
EXAMPLE_TRACE = 423
EXAMPLE_OUTPUT = N_SCORED // 2

CONDITIONS = (
    "stable",
    "full",
    "tangent",
    "magnitude_only",
    "route_only",
    "shuffled_route",
)

CONDITION_DEFINITIONS = {
    "stable": "stable phase/polarity route and stable magnitude (SS)",
    "full": "moving phase/polarity route and moving magnitude; intact model (MM)",
    "tangent": "first-order JVP of the full pre-softplus twin about the matched stabilized movie",
    "magnitude_only": "stable phase/polarity route with moving magnitude (SM)",
    "route_only": "moving phase/polarity route with stable magnitude (MS)",
    "shuffled_route": "moving magnitude with phase/polarity switch locations shuffled within sample and stem channel",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def splitrelu_from_route_magnitude(
    route_source: torch.Tensor,
    magnitude_source: torch.Tensor,
    *,
    positive_gain: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Reconstruct SplitReLU output from independently chosen route and magnitude.

    ``route_source > 0`` chooses the positive or negative output branch while
    ``abs(magnitude_source)`` supplies its non-negative amplitude.  Supplying
    the same tensor for both sources exactly reproduces SplitReLU.
    """

    if route_source.shape != magnitude_source.shape:
        raise ValueError((tuple(route_source.shape), tuple(magnitude_source.shape)))
    route = route_source > 0
    magnitude = magnitude_source.abs()
    gain = torch.as_tensor(positive_gain, dtype=magnitude.dtype, device=magnitude.device).abs()
    positive = torch.where(route, magnitude * gain, torch.zeros_like(magnitude))
    negative = torch.where(route, torch.zeros_like(magnitude), magnitude)
    return torch.cat((positive, negative), dim=1)


def matched_switch_shuffled_route(
    stable_source: torch.Tensor,
    moving_source: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Shuffle moving-induced polarity switches while preserving their count.

    The binary switch mask relative to the stable movie is independently
    permuted within every sample and stem channel over all remaining axes.
    Thus each sample/channel retains the exact number of route switches but
    loses their organized spatiotemporal placement.
    """

    if stable_source.shape != moving_source.shape:
        raise ValueError((tuple(stable_source.shape), tuple(moving_source.shape)))
    if stable_source.ndim < 3:
        raise ValueError(tuple(stable_source.shape))
    stable_route = stable_source > 0
    switch = torch.logical_xor(stable_route, moving_source > 0)
    shuffled = torch.empty_like(switch)
    generator = torch.Generator(device=switch.device)
    generator.manual_seed(int(seed))
    for sample in range(switch.shape[0]):
        for channel in range(switch.shape[1]):
            flat = switch[sample, channel].reshape(-1)
            order = torch.randperm(flat.numel(), generator=generator, device=flat.device)
            shuffled[sample, channel] = flat[order].reshape_as(switch[sample, channel])
    return torch.logical_xor(stable_route, shuffled)


def splitrelu_from_route_mask(
    route_mask: torch.Tensor,
    magnitude_source: torch.Tensor,
    *,
    positive_gain: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Reconstruct SplitReLU output from a binary route mask and magnitude."""

    if route_mask.shape != magnitude_source.shape:
        raise ValueError((tuple(route_mask.shape), tuple(magnitude_source.shape)))
    if route_mask.dtype != torch.bool:
        raise TypeError(route_mask.dtype)
    magnitude = magnitude_source.abs()
    gain = torch.as_tensor(positive_gain, dtype=magnitude.dtype, device=magnitude.device).abs()
    positive = torch.where(route_mask, magnitude * gain, torch.zeros_like(magnitude))
    negative = torch.where(route_mask, torch.zeros_like(magnitude), magnitude)
    return torch.cat((positive, negative), dim=1)


def unit_rate_metrics(rate_map: torch.Tensor) -> dict[str, torch.Tensor]:
    """Exact SSI, spatial contrast, and expected spikes for each map and unit."""

    flat = rate_map.clamp_min(0).double().flatten(start_dim=2)
    mean_rate = flat.mean(dim=2)
    gain = flat / (mean_rate[..., None] + 1e-8)
    ssi = (gain * torch.log2(gain + 1e-8)).mean(dim=2)
    cv2 = ((gain - 1.0) ** 2).mean(dim=2)
    return {
        "ssi": ssi,
        "cv2": cv2,
        "quadratic_bits": cv2 / (2.0 * math.log(2.0)),
        "mean_rate_hz": mean_rate,
        "expected_spikes": mean_rate / 120.0,
    }


def weighted_time_average(
    metric: np.ndarray,
    expected_spikes: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Expected-spike-weight a per-frame spatial metric over ``axis``."""

    metric = np.asarray(metric, dtype=np.float64)
    weight = np.asarray(expected_spikes, dtype=np.float64)
    return np.sum(metric * weight, axis=axis) / np.maximum(np.sum(weight, axis=axis), 1e-12)
