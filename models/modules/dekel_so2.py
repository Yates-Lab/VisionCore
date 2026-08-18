"""M77 topology with an ordinary spatiotemporal stem and an SO(2) spatial suffix.

The first layer intentionally remains the production M77 ``Conv3d``: it learns
eight unconstrained 60 x 7 x 7 filters and removes time.  Its sign-split output
is then interpreted as scalar fields and passed through a continuous-rotation
equivariant spatial stack.  Consequently, the *spatial suffix* is SO(2)
equivariant; the complete movie-to-feature mapping is not claimed to be
equivariant because the ordinary first layer can be orientation selective.

This module is imported lazily by the convnet factory so that ``escnn`` remains
an optional dependency for all existing VisionCore models.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from escnn import gspaces
    from escnn import nn as enn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "SO2DekelCore requires escnn. Install VisionCore[equivariant] or "
        "escnn==1.0.11 in the active environment."
    ) from exc

from .dekel import DekelCore, HammingConv3d, SignPairNormReLU


class SO2FieldNormLRNSignReLU(nn.Module):
    """Equivariant analogue of M77's pre-split GroupNorm -> LRN -> ReLU.

    Each field contains the real Fourier coefficients of an orientation
    function.  Group normalization uses a vector-valued mean and one invariant
    variance per group of fields.  Affine gains and LRN divisors are shared by
    all coefficients within a field, so they commute with SO(2).  ReLU is
    applied after sampling the orientation function and projecting it back to
    the configured Fourier band.  Positive and negative branches are retained
    as separate fields, matching M77's sign-preserving split.
    """

    def __init__(
        self,
        space,
        *,
        num_fields: int,
        irreps: Sequence[tuple],
        orientation_samples: int,
        num_groups: int,
        eps: float = 1.0e-5,
        lrn_size: int = 5,
        lrn_alpha: float = 0.1,
        lrn_beta: float = 0.75,
        lrn_k: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_fields = int(num_fields)
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.lrn_size = int(lrn_size)
        self.lrn_alpha = float(lrn_alpha)
        self.lrn_beta = float(lrn_beta)
        self.lrn_k = float(lrn_k)
        if self.num_fields < 1:
            raise ValueError("num_fields must be positive")
        if self.num_groups < 1 or self.num_fields % self.num_groups:
            raise ValueError(
                f"num_groups ({self.num_groups}) must divide num_fields "
                f"({self.num_fields})"
            )
        if self.lrn_size < 1 or self.lrn_size % 2 != 1:
            raise ValueError("lrn_size must be a positive odd integer")

        self.pointwise = enn.FourierPointwise(
            space,
            channels=self.num_fields,
            irreps=list(irreps),
            N=int(orientation_samples),
            function="p_relu",
            inplace=False,
        )
        self.in_type = self.pointwise.in_type
        self.field_size = self.in_type.size // self.num_fields
        self.out_type = enn.FieldType(
            space,
            list(self.in_type.representations) * 2,
        )

        # A scalar gain commutes with every representation matrix.  The bias is
        # added only along the invariant (frequency-zero) Fourier coefficient.
        self.weight = nn.Parameter(torch.ones(self.num_fields))
        self.bias = nn.Parameter(torch.zeros(self.num_fields))
        invariant = torch.zeros(self.field_size)
        invariant[0] = 1.0
        self.register_buffer("invariant_direction", invariant)

    def _group_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, fields, field_size, height, width = tensor.shape
        if fields != self.num_fields or field_size != self.field_size:
            raise ValueError(
                f"Expected reshaped fields (*, {self.num_fields}, "
                f"{self.field_size}, H, W), got {tuple(tensor.shape)}"
            )
        fields_per_group = self.num_fields // self.num_groups
        value = tensor.reshape(
            batch,
            self.num_groups,
            fields_per_group,
            self.field_size,
            height,
            width,
        )
        # Component means transform as a representation vector.  A single
        # squared norm over complete fields is invariant under SO(2).
        mean = value.mean(dim=(2, 4, 5), keepdim=True)
        centered = value - mean
        variance = centered.square().mean(dim=(2, 3, 4, 5), keepdim=True)
        value = centered * torch.rsqrt(variance + self.eps)
        value = value.reshape(
            batch, self.num_fields, self.field_size, height, width
        )
        value = value * self.weight[None, :, None, None, None]
        value = value + (
            self.bias[None, :, None, None, None]
            * self.invariant_direction[None, None, :, None, None]
        )
        return value

    def _local_response_norm(self, value: torch.Tensor) -> torch.Tensor:
        # Field energy is invariant to rotations within the Fourier fiber.
        energy = value.square().mean(dim=2)
        batch, _, height, width = energy.shape
        local = energy.permute(0, 2, 3, 1).reshape(-1, 1, self.num_fields)
        local = F.avg_pool1d(
            local,
            kernel_size=self.lrn_size,
            stride=1,
            padding=self.lrn_size // 2,
            count_include_pad=True,
        )
        local = local.reshape(batch, height, width, self.num_fields).permute(
            0, 3, 1, 2
        )
        divisor = (self.lrn_k + self.lrn_alpha * local).pow(-self.lrn_beta)
        return value * divisor[:, :, None]

    def forward(self, input: enn.GeometricTensor) -> enn.GeometricTensor:
        if input.type != self.in_type:
            raise ValueError(f"Expected field type {self.in_type}, got {input.type}")
        batch, _, height, width = input.tensor.shape
        value = input.tensor.reshape(
            batch, self.num_fields, self.field_size, height, width
        )
        value = self._local_response_norm(self._group_norm(value))
        value = value.reshape(batch, self.in_type.size, height, width)
        normalized = enn.GeometricTensor(value, self.in_type, input.coords)
        positive = self.pointwise(normalized)
        negative = self.pointwise(
            enn.GeometricTensor(-value, self.in_type, input.coords)
        )
        return enn.tensor_directsum([positive, negative])


class SO2DekelCore(nn.Module):
    """M77's feed-forward topology with a continuous-SO(2) spatial suffix."""

    def __init__(self, config) -> None:
        super().__init__()
        self.initial_channels = int(config["initial_channels"])
        self.temporal_support = int(config.get("temporal_support", 60))
        self.temporal_channels = int(config.get("temporal_channels", 8))
        if self.temporal_channels != 8:
            raise ValueError(
                "The first SO(2) M77 pilot is defined with exactly 8 ordinary "
                f"spatiotemporal filters, got {self.temporal_channels}"
            )

        spatial_fields = config.get("spatial_fields", [8, 8, 8])
        spatial_kernels = config.get("spatial_kernels", [15, 11, 9])
        norm_groups = config.get("norm_groups", [8, 4, 4, 4])
        if len(spatial_fields) != 3 or len(spatial_kernels) != 3:
            raise ValueError("SO2DekelCore requires exactly three spatial stages")
        if len(norm_groups) != 4:
            raise ValueError("norm_groups must specify the stem and three stages")
        self.spatial_fields = tuple(int(value) for value in spatial_fields)
        self.spatial_kernels = tuple(int(value) for value in spatial_kernels)

        self.maximum_frequency = int(config.get("maximum_frequency", 3))
        self.orientation_samples = int(config.get("orientation_samples", 16))
        if self.maximum_frequency < 1:
            raise ValueError("maximum_frequency must be positive")
        if self.orientation_samples < 2 * self.maximum_frequency + 1:
            raise ValueError(
                "orientation_samples must exceed the SO(2) Fourier Nyquist rate"
            )
        self.space = gspaces.rot2dOnR2(
            N=-1,
            maximum_frequency=self.maximum_frequency,
        )
        self.irreps = self.space.fibergroup.bl_irreps(self.maximum_frequency)

        self.scaffold_size = int(config.get("scaffold_size", 9))
        self.scaffold_mode = str(config.get("scaffold_mode", "nearest"))
        self.strict_input_size = bool(config.get("strict_input_size", True))
        input_size = config.get("input_size", [35, 35])
        self.input_size = tuple(int(value) for value in input_size)

        normalization = config.get("normalization", {}) or {}
        normalization_type = str(
            normalization.get("type", "groupnorm_lrn_presplit")
        )
        if normalization_type != "groupnorm_lrn_presplit":
            raise ValueError(
                "SO2DekelCore currently preserves only M77's "
                "groupnorm_lrn_presplit ordering"
            )
        normalization_kwargs = {
            "lrn_size": int(normalization.get("lrn_size", 5)),
            "lrn_alpha": float(normalization.get("lrn_alpha", 0.1)),
            "lrn_beta": float(normalization.get("lrn_beta", 0.75)),
            "lrn_k": float(normalization.get("lrn_k", 1.0)),
        }

        frequency_mask = config.get("frequency_mask", {}) or {}
        if bool(frequency_mask.get("hidden_spatial", False)):
            raise ValueError(
                "A Cartesian hidden-layer frequency mask would break continuous "
                "rotation equivariance; use the SO(2) kernel basis instead"
            )
        stem_frequency_axes = []
        if bool(frequency_mask.get("temporal", False)):
            stem_frequency_axes.append(-3)
        if bool(frequency_mask.get("stem_spatial", False)):
            stem_frequency_axes.extend((-2, -1))
        frequency_window = str(frequency_mask.get("window", "hann"))
        frequency_fft_pad = int(frequency_mask.get("fft_pad", 2))

        # This is deliberately the ordinary M77 spatiotemporal layer.
        self.temporal_conv = HammingConv3d(
            self.initial_channels,
            self.temporal_channels,
            (self.temporal_support, 7, 7),
            bias=False,
            frequency_mask_axes=stem_frequency_axes,
            frequency_window=frequency_window,
            frequency_fft_pad=frequency_fft_pad,
        )
        self.temporal_nonlinearity = SignPairNormReLU(
            self.temporal_channels,
            int(norm_groups[0]),
            normalization=normalization_type,
            **normalization_kwargs,
        )
        self.stem_type = enn.FieldType(
            self.space,
            [self.space.trivial_repr] * (2 * self.temporal_channels),
        )

        f1, f2, f3 = self.spatial_fields
        k1, k2, k3 = self.spatial_kernels
        self.stage1_nonlinearity = SO2FieldNormLRNSignReLU(
            self.space,
            num_fields=f1,
            irreps=self.irreps,
            orientation_samples=self.orientation_samples,
            num_groups=int(norm_groups[1]),
            **normalization_kwargs,
        )
        self.stage1_conv = enn.R2Conv(
            self.stem_type,
            self.stage1_nonlinearity.in_type,
            kernel_size=k1,
            padding=k1 // 2,
            bias=False,
        )
        self.pool1 = enn.PointwiseAvgPoolAntialiased2D(
            self.stage1_nonlinearity.out_type,
            sigma=float(config.get("pool_sigma", 0.6)),
            stride=2,
        )

        self.stage2_nonlinearity = SO2FieldNormLRNSignReLU(
            self.space,
            num_fields=f2,
            irreps=self.irreps,
            orientation_samples=self.orientation_samples,
            num_groups=int(norm_groups[2]),
            **normalization_kwargs,
        )
        self.stage2_conv = enn.R2Conv(
            self.stage1_nonlinearity.out_type,
            self.stage2_nonlinearity.in_type,
            kernel_size=k2,
            padding=k2 // 2,
            bias=False,
        )
        self.pool2 = enn.PointwiseAvgPoolAntialiased2D(
            self.stage2_nonlinearity.out_type,
            sigma=float(config.get("pool_sigma", 0.6)),
            stride=2,
        )

        self.stage3_nonlinearity = SO2FieldNormLRNSignReLU(
            self.space,
            num_fields=f3,
            irreps=self.irreps,
            orientation_samples=self.orientation_samples,
            num_groups=int(norm_groups[3]),
            **normalization_kwargs,
        )
        self.stage3_conv = enn.R2Conv(
            self.stage2_nonlinearity.out_type,
            self.stage3_nonlinearity.in_type,
            kernel_size=k3,
            padding=k3 // 2,
            bias=False,
        )

        field_sizes = {
            module.field_size
            for module in (
                self.stage1_nonlinearity,
                self.stage2_nonlinearity,
                self.stage3_nonlinearity,
            )
        }
        if len(field_sizes) != 1:
            raise RuntimeError(f"Inconsistent SO(2) field sizes: {field_sizes}")
        self.modulation_field_size = field_sizes.pop()
        self.modulation_num_fields = 2 * sum(self.spatial_fields)
        self._final_out_channels = (
            self.modulation_field_size * self.modulation_num_fields
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.temporal_conv.conv.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.temporal_conv.conv.bias is not None:
            nn.init.zeros_(self.temporal_conv.conv.bias)

    def _forward_geometric_stages(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(
                f"SO2DekelCore expects NCTHW input, got shape {tuple(x.shape)}"
            )
        if x.shape[2] != self.temporal_support:
            raise ValueError(
                f"SO2DekelCore requires exactly {self.temporal_support} stimulus "
                f"frames; got {x.shape[2]}"
            )
        stem = self.temporal_conv(x)
        if stem.shape[2] != 1:
            raise RuntimeError(
                "The full-history convolution must collapse time to a singleton"
            )
        stem = self.temporal_nonlinearity(stem.squeeze(2))
        stage1 = self.stage1_nonlinearity(
            self.stage1_conv(enn.GeometricTensor(stem, self.stem_type))
        )
        stage2 = self.stage2_nonlinearity(self.stage2_conv(self.pool1(stage1)))
        stage3 = self.stage3_nonlinearity(self.stage3_conv(self.pool2(stage2)))
        return stage1, stage2, stage3

    def forward_stages(self, x: torch.Tensor, *, strict_spatial: bool = True):
        if (
            strict_spatial
            and self.strict_input_size
            and tuple(x.shape[-2:]) != self.input_size
        ):
            raise ValueError(
                f"SO2DekelCore requires {self.input_size[0]}x{self.input_size[1]} "
                f"stimuli; got {tuple(x.shape[-2:])}"
            )
        return tuple(stage.tensor for stage in self._forward_geometric_stages(x))

    def _resize(self, value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tuple(value.shape[-2:]) == tuple(size):
            return value
        kwargs = {}
        if self.scaffold_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(
            value,
            size=size,
            mode=self.scaffold_mode,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stages = self.forward_stages(x)
        size = (self.scaffold_size, self.scaffold_size)
        scaffold = torch.cat([self._resize(stage, size) for stage in stages], dim=1)
        return scaffold.unsqueeze(2)

    def forward_spatial_map(self, x: torch.Tensor) -> torch.Tensor:
        stages = self.forward_stages(x, strict_spatial=False)
        target_size = tuple(stages[-1].shape[-2:])
        return torch.cat(
            [self._resize(stage, target_size) for stage in stages], dim=1
        ).unsqueeze(2)

    def get_output_channels(self) -> int:
        return self._final_out_channels

    def effective_temporal_weight(self) -> torch.Tensor:
        return self.temporal_conv.weight

    def first_layer_separable_components(self):
        return DekelCore.first_layer_separable_components(self)

    def plot_temporal_filters(self, sampling_rate: float = 240.0):
        return DekelCore.plot_temporal_filters(self, sampling_rate=sampling_rate)

    def plot_first_layer_spatial_filters(self):
        return DekelCore.plot_first_layer_spatial_filters(self)


__all__ = ["SO2DekelCore", "SO2FieldNormLRNSignReLU"]
