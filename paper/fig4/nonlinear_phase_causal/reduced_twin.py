"""A deployable shared-subspace surrogate for the RR100 digital twin.

This follows the Fixational Transients subspace workflow: shared convolutional
filters generate low-dimensional signals, those signals are normalized, and a
flexible learned nonlinearity maps them to population responses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from paper.fig4.nonlinear_phase_causal.response_subspace import (
    N_LAGS,
    OUTPUT_SIZE,
    OUTPUT_STRIDE,
    PATCH_SIZE,
    dct_filters_to_movies,
)


class PopulationNonlinearity(nn.Module):
    """Flexible positive response function in shared generator coordinates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        dimensions = (int(input_dim), *[int(value) for value in hidden_dims])
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(left, right))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dimensions[-1], int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, generators: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(5.0 * torch.tanh(generators / 5.0)))


@dataclass
class ReducedTwinState:
    rank: int
    unit_indices: torch.Tensor
    basis: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    generator_mean: torch.Tensor
    generator_std: torch.Tensor
    hidden_dims: tuple[int, ...]
    dropout: float
    decoder_state: dict[str, torch.Tensor]


class ReducedTwin(nn.Module):
    """Shared linear spatiotemporal front end plus learned population response."""

    def __init__(self, state: ReducedTwinState):
        super().__init__()
        self.rank = int(state.rank)
        self.register_buffer("unit_indices", state.unit_indices.clone().long())
        self.register_buffer("basis", state.basis.clone().float())
        self.register_buffer("feature_mean", state.feature_mean.clone().float())
        self.register_buffer("feature_std", state.feature_std.clone().float())
        self.register_buffer("generator_mean", state.generator_mean.clone().float())
        self.register_buffer("generator_std", state.generator_std.clone().float())
        self.decoder = PopulationNonlinearity(
            self.rank,
            len(self.unit_indices),
            hidden_dims=tuple(state.hidden_dims),
            dropout=float(state.dropout),
        )
        self.decoder.load_state_dict(state.decoder_state)

    def normalize_generators(self, value: torch.Tensor) -> torch.Tensor:
        shape = [1] * value.ndim
        shape[-1] = self.rank
        return (value - self.generator_mean.view(*shape)) / self.generator_std.view(*shape)

    def decode_generators(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1]
        prediction = self.decoder(self.normalize_generators(value).reshape(-1, self.rank))
        return prediction.reshape(*shape, len(self.unit_indices))

    def movie_generators(self, movie: torch.Tensor) -> torch.Tensor:
        """Return shared generator maps as B x K x 51 x 51."""
        dtype, device = movie.dtype, movie.device
        standardized_basis = self.basis.to(device, dtype)
        feature_std = self.feature_std.to(device, dtype)
        feature_mean = self.feature_mean.to(device, dtype)
        physical_dct = standardized_basis / feature_std[None]
        filters = dct_filters_to_movies(physical_dct).reshape(
            self.rank, 1, N_LAGS, PATCH_SIZE, PATCH_SIZE
        )
        local_movie = movie[..., 13:138, 13:138]
        generators = F.conv3d(
            local_movie,
            filters,
            stride=(1, OUTPUT_STRIDE, OUTPUT_STRIDE),
        )[:, :, 0]
        offset = -(standardized_basis * feature_mean[None] / feature_std[None]).sum(dim=1)
        return generators + offset[None, :, None, None]

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        generators = self.movie_generators(movie)
        flattened = generators.permute(0, 2, 3, 1)
        rate = self.decode_generators(flattened)
        return rate.permute(0, 3, 1, 2)


def normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target) / target_std).square().mean()
