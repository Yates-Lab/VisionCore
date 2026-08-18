"""Response-first low-rank spatiotemporal models for the RR100 twin.

The fitted object is a response model, not a Jacobian decomposition.  Each
unit gets a private rank-k projection of a complete 32-frame local movie and
a full linear-quadratic decoder inside that projection.  The same model can
be evaluated convolutionally to predict an entire RR100 response map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


FRAME_RATE_HZ = 120.0
PPD = 37.50476617
N_LAGS = 32
INPUT_SIZE = 151
OUTPUT_SIZE = 51
OUTPUT_STRIDE = 2
PATCH_SIZE = 25
PATCH_RADIUS = PATCH_SIZE // 2
OUTPUT_INPUT_ORIGIN = 25  # input center corresponding to response-map index 0
DCT_MAX_SPATIAL_CPD = 14.0


def orthonormal_dct(n: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix with shape ``(n, n)``."""
    positions = torch.arange(n, dtype=dtype) + 0.5
    frequencies = torch.arange(n, dtype=dtype)[:, None]
    basis = torch.cos(math.pi * frequencies * positions[None] / float(n))
    basis[0] *= math.sqrt(1.0 / n)
    if n > 1:
        basis[1:] *= math.sqrt(2.0 / n)
    return basis


def deterministic_unit_subset(table: pd.DataFrame, n_per_group: int = 6) -> pd.DataFrame:
    """Choose units before fitting, evenly across OSI ranks in each SF group."""
    required = {
        "unit_index",
        "sf_split_metric",
        "prior_orientation_selectivity_index",
        "dynamic_peak_response_amp",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing unit-selection columns: {sorted(missing)}")
    work = table.copy()
    work["response_subspace_group"] = np.where(
        pd.to_numeric(work.sf_split_metric, errors="coerce") < 0.5,
        "lower-SF",
        "higher-SF",
    )
    chosen = []
    quantiles = np.linspace(0.05, 0.95, int(n_per_group))
    for group in ("lower-SF", "higher-SF"):
        part = work.loc[work.response_subspace_group.eq(group)].copy()
        response_threshold = float(part.dynamic_peak_response_amp.median())
        part = part.loc[part.dynamic_peak_response_amp.ge(response_threshold)].copy()
        part = part.sort_values(
            ["prior_orientation_selectivity_index", "unit_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        if len(part) < n_per_group:
            raise ValueError(f"Only {len(part)} units in {group}")
        targets = quantiles * (len(part) - 1)
        indices = np.rint(targets).astype(int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError((group, indices.tolist()))
        selected = part.iloc[indices].copy()
        selected["selection_osi_quantile"] = quantiles
        selected["selection_rule"] = (
            "nearest rank to evenly spaced 0.05..0.95 prior-OSI quantiles among units "
            "at or above their SF-group median prior grating response amplitude"
        )
        selected["selection_min_dynamic_peak_response_amp"] = response_threshold
        chosen.append(selected)
    out = pd.concat(chosen, ignore_index=True)
    out.insert(0, "subset_index", np.arange(len(out), dtype=int))
    return out


def response_grid_offsets(radius: int = 8, count: int = 5) -> np.ndarray:
    values = np.rint(np.linspace(-int(radius), int(radius), int(count))).astype(int)
    yy, xx = np.meshgrid(values, values, indexing="ij")
    return np.stack((yy.ravel(), xx.ravel()), axis=1)


def input_centers_for_output_offsets(offset_yx: np.ndarray) -> np.ndarray:
    offset = np.asarray(offset_yx, dtype=int)
    if offset.ndim != 2 or offset.shape[1] != 2:
        raise ValueError(offset.shape)
    return INPUT_SIZE // 2 + OUTPUT_STRIDE * offset


def generate_movie_batch(
    batch_size: int,
    *,
    seed: int,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Generate a reproducible fixed-covariance, fixed-contrast Gaussian ensemble.

    A single covariance and RMS are essential here: otherwise global contrast
    and spectrum changes create response variance distributed across the whole
    stimulus space and confound the question of response-subspace rank.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    white = torch.randn(
        int(batch_size), 1, N_LAGS, INPUT_SIZE, INPUT_SIZE,
        generator=generator, device=device, dtype=dtype,
    )
    smooth3 = F.avg_pool3d(white, kernel_size=3, stride=1, padding=1)
    smooth7 = F.avg_pool3d(white, kernel_size=7, stride=1, padding=3)
    mix = torch.tensor([0.15, 0.85, 0.50], device=device, dtype=dtype)
    mix = mix / mix.square().sum().sqrt()
    mix = mix[None].expand(int(batch_size), -1)
    movie = (
        mix[:, 0, None, None, None, None] * white
        + mix[:, 1, None, None, None, None] * smooth3
        + mix[:, 2, None, None, None, None] * smooth7
    )
    movie = movie - movie.mean(dim=(2, 3, 4), keepdim=True)
    movie = movie / movie.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    rms = torch.full((int(batch_size),), 0.16, device=device, dtype=dtype)
    movie = (movie * rms[:, None, None, None, None]).clamp(-0.5, 0.5)
    metadata = {
        "rms": rms.detach().cpu().numpy().astype(np.float32),
        "mix_weights": mix.detach().cpu().numpy().astype(np.float32),
        "clipped_fraction": np.asarray(
            [(movie.abs() >= 0.5).float().mean().item()], dtype=np.float32
        ),
    }
    return movie, metadata


def extract_local_patches(movie: torch.Tensor, output_offset_yx: np.ndarray) -> torch.Tensor:
    """Extract local movies centered on selected output-map positions."""
    if tuple(movie.shape[1:]) != (1, N_LAGS, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(tuple(movie.shape))
    centers = input_centers_for_output_offsets(output_offset_yx)
    patches = []
    for center_y, center_x in centers:
        y0, x0 = int(center_y - PATCH_RADIUS), int(center_x - PATCH_RADIUS)
        patches.append(movie[:, 0, :, y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE])
    return torch.stack(patches, dim=1)


def dct_feature_mask() -> torch.Tensor:
    """All temporal DCT modes crossed with spatial modes below 14 c/deg."""
    spatial_cpd = torch.arange(PATCH_SIZE, dtype=torch.float32) * PPD / (2 * PATCH_SIZE)
    radial = torch.sqrt(spatial_cpd[:, None].square() + spatial_cpd[None, :].square())
    spatial = radial <= DCT_MAX_SPATIAL_CPD
    return spatial[None].expand(N_LAGS, -1, -1).reshape(-1)


N_DCT_FEATURES = int(dct_feature_mask().sum())


def patches_to_dct(patches: torch.Tensor) -> torch.Tensor:
    """Represent local movies with all 32 temporal modes and <=14-c/deg spatial modes."""
    if patches.ndim != 5 or tuple(patches.shape[-3:]) != (N_LAGS, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(tuple(patches.shape))
    batch, grid = patches.shape[:2]
    temporal = orthonormal_dct(N_LAGS, dtype=patches.dtype).to(patches.device)
    spatial = orthonormal_dct(PATCH_SIZE, dtype=patches.dtype).to(patches.device)
    value = patches.reshape(batch * grid, N_LAGS, PATCH_SIZE * PATCH_SIZE)
    value = torch.matmul(temporal, value).reshape(-1, N_LAGS, PATCH_SIZE, PATCH_SIZE)
    value = torch.matmul(spatial, value)
    value = torch.matmul(value, spatial.T)
    value = value.reshape(batch, grid, -1)
    return value[..., dct_feature_mask().to(value.device)]


def dct_filters_to_movies(coefficients: torch.Tensor) -> torch.Tensor:
    """Invert flattened full-volume DCT coefficients into movie filters."""
    if coefficients.shape[-1] not in (N_DCT_FEATURES, N_LAGS * PATCH_SIZE * PATCH_SIZE):
        raise ValueError(tuple(coefficients.shape))
    leading = coefficients.shape[:-1]
    if coefficients.shape[-1] == N_DCT_FEATURES:
        full = coefficients.new_zeros(*leading, N_LAGS * PATCH_SIZE * PATCH_SIZE)
        full[..., dct_feature_mask().to(coefficients.device)] = coefficients
        coefficients = full
    temporal = orthonormal_dct(N_LAGS, dtype=coefficients.dtype).to(coefficients.device)
    spatial = orthonormal_dct(PATCH_SIZE, dtype=coefficients.dtype).to(coefficients.device)
    value = coefficients.reshape(-1, N_LAGS, PATCH_SIZE, PATCH_SIZE)
    value = torch.matmul(spatial.T, value)
    value = torch.matmul(value, spatial)
    value = torch.matmul(temporal.T, value.reshape(-1, N_LAGS, PATCH_SIZE * PATCH_SIZE))
    return value.reshape(*leading, N_LAGS, PATCH_SIZE, PATCH_SIZE)


def dct_frequency_penalty() -> torch.Tensor:
    temporal_hz = torch.arange(N_LAGS, dtype=torch.float32) * FRAME_RATE_HZ / (2 * N_LAGS)
    spatial_cpd = torch.arange(PATCH_SIZE, dtype=torch.float32) * PPD / (2 * PATCH_SIZE)
    temporal = (temporal_hz / 30.0).square()[:, None, None]
    spatial = (
        spatial_cpd[:, None].square() + spatial_cpd[None, :].square()
    )[None] / (10.0**2)
    value = (temporal + spatial).reshape(-1)
    return value[dct_feature_mask()]


@dataclass
class ResponseSubspaceState:
    rank: int
    weights: torch.Tensor
    linear: torch.Tensor
    quadratic: torch.Tensor
    bias: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


class LowRankLQ(nn.Module):
    """Private per-unit low-rank projections with a full L+Q decoder."""

    def __init__(self, n_units: int, rank: int, n_features: int):
        super().__init__()
        self.n_units = int(n_units)
        self.rank = int(rank)
        self.n_features = int(n_features)
        weights = torch.randn(self.n_units, self.rank, self.n_features)
        weights = torch.linalg.qr(weights.transpose(1, 2), mode="reduced").Q.transpose(1, 2)
        self.weights = nn.Parameter(weights)
        self.linear = nn.Parameter(torch.zeros(self.n_units, self.rank))
        quadratic = torch.eye(self.rank)[None].repeat(self.n_units, 1, 1) * 0.03
        self.quadratic = nn.Parameter(quadratic)
        self.bias = nn.Parameter(torch.zeros(self.n_units))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat_weights = self.weights.reshape(self.n_units * self.rank, self.n_features)
        projected = torch.matmul(features, flat_weights.T).reshape(-1, self.n_units, self.rank)
        symmetric_q = 0.5 * (self.quadratic + self.quadratic.transpose(1, 2))
        linear = torch.einsum("nur,ur->nu", projected, self.linear)
        quadratic = torch.einsum("nur,urs,nus->nu", projected, symmetric_q, projected)
        return self.bias[None] + linear + quadratic

    @torch.no_grad()
    def retract(self) -> None:
        """Orthonormalize every subspace while preserving its predictions."""
        for unit in range(self.n_units):
            q, r = torch.linalg.qr(self.weights[unit].T, mode="reduced")
            old_linear = self.linear[unit].clone()
            old_quadratic = self.quadratic[unit].clone()
            self.weights[unit].copy_(q.T)
            self.linear[unit].copy_(r @ old_linear)
            self.quadratic[unit].copy_(r @ old_quadratic @ r.T)

    def smoothness_loss(self, penalty: torch.Tensor) -> torch.Tensor:
        return (self.weights.square() * penalty[None, None]).sum(dim=2).mean()

    def orthogonality_loss(self) -> torch.Tensor:
        gram = self.weights @ self.weights.transpose(1, 2)
        identity = torch.eye(self.rank, device=gram.device, dtype=gram.dtype)[None]
        return (gram - identity).square().mean()


def convolutional_predict(
    movie: torch.Tensor,
    state: ResponseSubspaceState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a fitted center-response model to every exact map location."""
    weights = state.weights.to(movie.device, movie.dtype)
    feature_mean = state.feature_mean.to(movie.device, movie.dtype)
    feature_std = state.feature_std.to(movie.device, movie.dtype)
    physical_dct = weights / feature_std[None, None]
    filters = dct_filters_to_movies(physical_dct).reshape(
        state.weights.shape[0] * state.rank, 1, N_LAGS, PATCH_SIZE, PATCH_SIZE
    )
    # Output index 0 has input center 25, hence the odd 13-pixel crop origin.
    local_movie = movie[..., 13:138, 13:138]
    projected = F.conv3d(local_movie, filters, stride=(1, OUTPUT_STRIDE, OUTPUT_STRIDE))[:, :, 0]
    offsets = -(weights * feature_mean[None, None] / feature_std[None, None]).sum(dim=2)
    projected = projected.reshape(len(movie), weights.shape[0], state.rank, OUTPUT_SIZE, OUTPUT_SIZE)
    projected = projected + offsets[None, :, :, None, None]
    linear = state.linear.to(movie.device, movie.dtype)
    quadratic = state.quadratic.to(movie.device, movie.dtype)
    quadratic = 0.5 * (quadratic + quadratic.transpose(1, 2))
    normalized = state.bias.to(movie.device, movie.dtype)[None, :, None, None]
    normalized = normalized + torch.einsum("burhw,ur->buhw", projected, linear)
    normalized = normalized + torch.einsum("burhw,urs,bushw->buhw", projected, quadratic, projected)
    z = normalized * state.target_std.to(movie.device, movie.dtype)[None, :, None, None]
    z = z + state.target_mean.to(movie.device, movie.dtype)[None, :, None, None]
    return z, F.softplus(z)


def r2_score(target: np.ndarray, prediction: np.ndarray, axis: int = 0) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.sum((target - prediction) ** 2, axis=axis)
    total = np.sum((target - np.mean(target, axis=axis, keepdims=True)) ** 2, axis=axis)
    return 1.0 - residual / np.maximum(total, 1e-12)
