"""Dekel-style feed-forward spatiotemporal vision core.

This module ports the inductive biases of the 2022 single-session model used
for the successful response-subspace analysis:

* one full-history 3-D convolution that removes the temporal axis;
* interleaved positive/negative channel pairs followed by GroupNorm and ReLU;
* three large, Hamming-windowed spatial convolutions;
* smooth :math:`L_5` pooling rather than hard max pooling; and
* a multiscale scaffold at a common 9 x 9 resolution.

The regularization operators live in ``training.regularizers`` so their
strength and schedules remain experiment-configurable.  Parameter names in
this module are intentionally stable (``temporal_conv`` and ``stage*_conv``)
because the YAML regularizers target those names.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_sign_split(x: torch.Tensor) -> torch.Tensor:
    """Return interleaved ``[z0, -z0, z1, -z1, ...]`` channels."""
    return torch.stack((x, -x), dim=2).flatten(1, 2)


def _spatial_hamming_mask(height: int, width: int) -> torch.Tensor:
    """Dekel's square-root separable spatial Hamming envelope."""
    wy = torch.hamming_window(height, periodic=False)
    wx = torch.hamming_window(width, periodic=False)
    return torch.sqrt(wy[:, None] * wx[None, :]).view(1, 1, 1, height, width)


def _frequency_window_weight(
    weight: torch.Tensor,
    axes: Sequence[int],
    *,
    window: str = "hann",
    fft_pad: int = 2,
) -> torch.Tensor:
    """Band-limit selected kernel axes with a separable frequency window.

    This restores the useful ``aa_freq`` behavior that existed in an older
    VisionCore convolution implementation.  Kernels are centered in a padded
    array before the FFT, multiplied by low-pass windows that reach zero at
    Nyquist, transformed back, and center-cropped to their original support.
    The operation is differentiable and therefore constrains the effective
    weight without overwriting the optimizer's raw parameter.
    """
    if not axes:
        return weight
    if fft_pad < 1:
        raise ValueError("frequency_fft_pad must be at least 1")

    axes = tuple(ax if ax >= 0 else weight.ndim + ax for ax in axes)
    if len(set(axes)) != len(axes) or any(ax < 2 or ax >= weight.ndim for ax in axes):
        raise ValueError(f"Invalid kernel frequency axes {axes} for shape {weight.shape}")

    original_dtype = weight.dtype
    work = weight.float() if weight.dtype in {torch.float16, torch.bfloat16} else weight
    kernel_sizes = [work.shape[ax] for ax in axes]
    fft_sizes = [
        1 << int(math.ceil(math.log2(max(1, size * fft_pad))))
        for size in kernel_sizes
    ]

    padded_shape = list(work.shape)
    for ax, size in zip(axes, fft_sizes):
        padded_shape[ax] = size
    padded = work.new_zeros(padded_shape)
    insert = [slice(None)] * work.ndim
    for ax, fft_size, kernel_size in zip(axes, fft_sizes, kernel_sizes):
        start = (fft_size - kernel_size) // 2
        insert[ax] = slice(start, start + kernel_size)
    padded[tuple(insert)] = work

    spectrum = torch.fft.fftn(torch.fft.ifftshift(padded, dim=axes), dim=axes)
    spectrum = torch.fft.fftshift(spectrum, dim=axes)
    mask = work.new_ones([1] * work.ndim)
    for ax, fft_size in zip(axes, fft_sizes):
        if not hasattr(torch.signal.windows, window):
            raise ValueError(f"Unknown frequency window {window!r}")
        vector = getattr(torch.signal.windows, window)(
            fft_size, device=work.device, dtype=work.dtype
        )
        vector = vector / vector.max().clamp_min(1e-12)
        shape = [1] * work.ndim
        shape[ax] = fft_size
        mask = mask * vector.view(shape)
    spectrum = spectrum * mask

    filtered = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(spectrum, dim=axes), dim=axes).real,
        dim=axes,
    )
    crop = [slice(None)] * work.ndim
    for ax, fft_size, kernel_size in zip(axes, fft_sizes, kernel_sizes):
        start = (fft_size - kernel_size) // 2
        crop[ax] = slice(start, start + kernel_size)
    return filtered[tuple(crop)].to(dtype=original_dtype)


class HammingConv3d(nn.Module):
    """Conv3d whose effective weights have a fixed spatial Hamming envelope."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        *,
        bias: bool = False,
        frequency_mask_axes: Sequence[int] = (),
        frequency_window: str = "hann",
        frequency_fft_pad: int = 2,
    ) -> None:
        super().__init__()
        kt, kh, kw = kernel_size
        self.kernel_size = tuple(kernel_size)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(0, kh // 2, kw // 2),
            bias=bias,
        )
        self.register_buffer("spatial_window", _spatial_hamming_mask(kh, kw))
        self.frequency_mask_axes = tuple(frequency_mask_axes)
        self.frequency_window = str(frequency_window)
        self.frequency_fft_pad = int(frequency_fft_pad)

    @property
    def weight(self) -> torch.Tensor:
        weight = self.conv.weight * self.spatial_window
        return _frequency_window_weight(
            weight,
            self.frequency_mask_axes,
            window=self.frequency_window,
            fft_pad=self.frequency_fft_pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(
            x,
            self.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class HammingConv2d(nn.Module):
    """Conv2d whose effective weights have a fixed spatial Hamming envelope."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = False,
        frequency_mask_axes: Sequence[int] = (),
        frequency_window: str = "hann",
        frequency_fft_pad: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        # Reuse the 3-D-shaped helper and remove its temporal singleton.
        self.register_buffer(
            "spatial_window",
            _spatial_hamming_mask(kernel_size, kernel_size).squeeze(2),
        )
        self.frequency_mask_axes = tuple(frequency_mask_axes)
        self.frequency_window = str(frequency_window)
        self.frequency_fft_pad = int(frequency_fft_pad)

    @property
    def weight(self) -> torch.Tensor:
        weight = self.conv.weight * self.spatial_window
        return _frequency_window_weight(
            weight,
            self.frequency_mask_axes,
            window=self.frequency_window,
            fft_pad=self.frequency_fft_pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class SignPairNormReLU(nn.Module):
    """Configurable sign-paired divisive normalization followed by ReLU.

    "groupnorm" is the exact historical single-session operation.  "lrn" and
    "groupnorm_lrn" provide matched post-split controls.  The
    "groupnorm_lrn_presplit" mode reproduces the ordering in Dekel's
    historical LocalResponseBatchNorm experiments.
    """

    def __init__(
        self,
        in_channels: int,
        num_groups: int,
        eps: float = 1e-5,
        *,
        normalization: str = "groupnorm",
        lrn_size: int = 5,
        lrn_alpha: float = 1e-4,
        lrn_beta: float = 0.75,
        lrn_k: float = 1.0,
    ):
        super().__init__()
        out_channels = 2 * int(in_channels)
        normalization = str(normalization).lower().replace("+", "_")
        aliases = {
            "group_norm": "groupnorm",
            "local_response_norm": "lrn",
            "localresponsenorm": "lrn",
            "groupnormlocalresponsenorm": "groupnorm_lrn",
            "groupnorm_local_response_norm": "groupnorm_lrn",
            "historical_groupnorm_lrn": "groupnorm_lrn_presplit",
        }
        normalization = aliases.get(normalization, normalization)
        if normalization not in {
            "groupnorm",
            "lrn",
            "groupnorm_lrn",
            "groupnorm_lrn_presplit",
        }:
            raise ValueError(
                "normalization must be one of groupnorm, lrn, or "
                "groupnorm_lrn[_presplit]; "
                f"got {normalization!r}"
            )
        if lrn_size < 1:
            raise ValueError("lrn_size must be positive")
        post_split_groupnorm = normalization in {"groupnorm", "groupnorm_lrn"}
        pre_split_groupnorm = normalization == "groupnorm_lrn_presplit"
        norm_channels = int(in_channels) if pre_split_groupnorm else out_channels
        if (
            (post_split_groupnorm or pre_split_groupnorm)
            and norm_channels % num_groups
        ):
            raise ValueError(
                f"normalization channels ({norm_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        channels_per_group = norm_channels // num_groups
        if post_split_groupnorm and channels_per_group % 2:
            raise ValueError(
                "Each GroupNorm group must contain complete +/- channel pairs; "
                f"got {channels_per_group} channels per group"
            )
        self.normalization = normalization
        self.norm = (
            nn.GroupNorm(num_groups, norm_channels, eps=eps, affine=True)
            if post_split_groupnorm or pre_split_groupnorm
            else nn.Identity()
        )
        self.lrn = (
            nn.LocalResponseNorm(
                size=int(lrn_size),
                alpha=float(lrn_alpha),
                beta=float(lrn_beta),
                k=float(lrn_k),
            )
            if normalization in {
                "lrn",
                "groupnorm_lrn",
                "groupnorm_lrn_presplit",
            }
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == "groupnorm_lrn_presplit":
            return F.relu(_pairwise_sign_split(self.lrn(self.norm(x))))
        paired = _pairwise_sign_split(x)
        return F.relu(self.lrn(self.norm(paired)))


class DekelCore(nn.Module):
    """Feed-forward full-history CNN with a multiscale feature scaffold.

    Parameters mirror the historical single-session model by default. Doubling
    ``temporal_channels`` and each ``spatial_channels`` value reproduces the
    width of Dekel's later multisession variant without changing its topology.

    Input and output follow the VisionCore convention:

    ``(batch, channels, 60, 35, 35) -> (batch, scaffold_channels, 1, 9, 9)``.
    """

    def __init__(self, config):
        super().__init__()
        self.initial_channels = int(config["initial_channels"])
        self.temporal_support = int(config.get("temporal_support", 60))
        self.temporal_channels = int(config.get("temporal_channels", 4))

        spatial_channels = config.get("spatial_channels", [24, 24, 24])
        spatial_kernels = config.get("spatial_kernels", [15, 11, 9])
        norm_groups = config.get("norm_groups", [4, 8, 6, 4])
        if len(spatial_channels) != 3 or len(spatial_kernels) != 3:
            raise ValueError("DekelCore requires exactly three spatial stages")
        if len(norm_groups) != 4:
            raise ValueError("norm_groups must specify the stem and three stages")

        self.spatial_channels = tuple(int(v) for v in spatial_channels)
        self.spatial_kernels = tuple(int(v) for v in spatial_kernels)
        self.scaffold_size = int(config.get("scaffold_size", 9))
        self.scaffold_mode = str(config.get("scaffold_mode", "nearest"))
        self.pool_norm = float(config.get("pool_norm", 5.0))
        self.strict_input_size = bool(config.get("strict_input_size", True))
        input_size = config.get("input_size", [35, 35])
        self.input_size = tuple(int(v) for v in input_size)

        normalization = config.get("normalization", {}) or {}
        normalization_type = str(normalization.get("type", "groupnorm"))
        normalization_kwargs = {
            "normalization": normalization_type,
            "lrn_size": int(normalization.get("lrn_size", 5)),
            "lrn_alpha": float(normalization.get("lrn_alpha", 1e-4)),
            "lrn_beta": float(normalization.get("lrn_beta", 0.75)),
            "lrn_k": float(normalization.get("lrn_k", 1.0)),
        }
        self.normalization_config = dict(normalization_kwargs)

        frequency_mask = config.get("frequency_mask", {}) or {}
        temporal_frequency_mask = bool(frequency_mask.get("temporal", False))
        stem_spatial_frequency_mask = bool(
            frequency_mask.get("stem_spatial", frequency_mask.get("spatial", False))
        )
        hidden_spatial_frequency_mask = bool(
            frequency_mask.get("hidden_spatial", frequency_mask.get("spatial", False))
        )
        frequency_window = str(frequency_mask.get("window", "hann"))
        frequency_fft_pad = int(frequency_mask.get("fft_pad", 2))
        stem_frequency_axes = []
        if temporal_frequency_mask:
            stem_frequency_axes.append(-3)
        if stem_spatial_frequency_mask:
            stem_frequency_axes.extend((-2, -1))
        hidden_frequency_axes = (-2, -1) if hidden_spatial_frequency_mask else ()

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
            **normalization_kwargs,
        )

        c1, c2, c3 = self.spatial_channels
        k1, k2, k3 = self.spatial_kernels
        self.stage1_conv = HammingConv2d(
            2 * self.temporal_channels,
            c1,
            k1,
            frequency_mask_axes=hidden_frequency_axes,
            frequency_window=frequency_window,
            frequency_fft_pad=frequency_fft_pad,
        )
        self.stage1_nonlinearity = SignPairNormReLU(
            c1, int(norm_groups[1]), **normalization_kwargs
        )
        self.stage2_conv = HammingConv2d(
            2 * c1,
            c2,
            k2,
            frequency_mask_axes=hidden_frequency_axes,
            frequency_window=frequency_window,
            frequency_fft_pad=frequency_fft_pad,
        )
        self.stage2_nonlinearity = SignPairNormReLU(
            c2, int(norm_groups[2]), **normalization_kwargs
        )
        self.stage3_conv = HammingConv2d(
            2 * c2,
            c3,
            k3,
            frequency_mask_axes=hidden_frequency_axes,
            frequency_window=frequency_window,
            frequency_fft_pad=frequency_fft_pad,
        )
        self.stage3_nonlinearity = SignPairNormReLU(
            c3, int(norm_groups[3]), **normalization_kwargs
        )

        self.pool = nn.LPPool2d(
            norm_type=self.pool_norm, kernel_size=3, stride=2, ceil_mode=False
        )
        self._final_out_channels = 2 * (c1 + c2 + c3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit zero padding is part of the original implementation and
        # produces 35 -> 18 -> 9 spatial sizes.
        return self.pool(F.pad(x, (1, 1, 1, 1)))

    def _to_scaffold_size(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (self.scaffold_size, self.scaffold_size):
            return x
        kwargs = {}
        if self.scaffold_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return F.interpolate(
            x,
            size=(self.scaffold_size, self.scaffold_size),
            mode=self.scaffold_mode,
            **kwargs,
        )

    def forward_stages(
        self,
        x: torch.Tensor,
        *,
        strict_spatial: bool = True,
    ):
        if x.ndim != 5:
            raise ValueError(f"DekelCore expects NCTHW input, got shape {tuple(x.shape)}")
        if x.shape[2] != self.temporal_support:
            raise ValueError(
                f"DekelCore requires exactly {self.temporal_support} stimulus frames "
                f"(~250 ms at 240 Hz); got {x.shape[2]}"
            )
        if (
            strict_spatial
            and self.strict_input_size
            and tuple(x.shape[-2:]) != self.input_size
        ):
            raise ValueError(
                f"DekelCore requires {self.input_size[0]}x{self.input_size[1]} "
                f"stimuli; got {tuple(x.shape[-2:])}"
            )

        stem = self.temporal_conv(x)
        if stem.shape[2] != 1:
            raise RuntimeError(
                "The full-history convolution must collapse time to a singleton; "
                f"got temporal size {stem.shape[2]}"
            )
        stem = self.temporal_nonlinearity(stem.squeeze(2))

        stage1 = self.stage1_nonlinearity(self.stage1_conv(stem))
        stage2 = self.stage2_nonlinearity(self.stage2_conv(self._downsample(stage1)))
        stage3 = self.stage3_nonlinearity(self.stage3_conv(self._downsample(stage2)))
        return stage1, stage2, stage3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stages = self.forward_stages(x)
        scaffold = torch.cat([self._to_scaffold_size(v) for v in stages], dim=1)
        return scaffold.unsqueeze(2)

    def forward_spatial_map(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned core convolutionally over a larger field.

        Normal training collapses every scale to a fixed 9x9 (or 13x13)
        scaffold because the Gaussian readout predicts one neural response per
        movie. Figure 4 instead translates that learned readout over a large
        retinal image to obtain a spatial rate map. For that use, align the
        shallower stages to the deepest stage's *native* spatial lattice rather
        than resizing the complete field back to the training scaffold.

        On the configured training aperture this is exactly the ordinary
        forward path: 35 -> 18 -> 9 and 51 -> 26 -> 13. On a 151-pixel field it
        yields a 38x38 feature scaffold, over which the learned 9x9/13x13
        Gaussian readout can be translated without retraining.
        """
        stages = self.forward_stages(x, strict_spatial=False)
        target_size = tuple(stages[-1].shape[-2:])
        kwargs = {}
        if self.scaffold_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        aligned = [
            value
            if tuple(value.shape[-2:]) == target_size
            else F.interpolate(
                value,
                size=target_size,
                mode=self.scaffold_mode,
                **kwargs,
            )
            for value in stages
        ]
        return torch.cat(aligned, dim=1).unsqueeze(2)

    def get_output_channels(self) -> int:
        return self._final_out_channels

    def effective_temporal_weight(self) -> torch.Tensor:
        """Return the windowed first-layer weights used by the forward pass."""
        return self.temporal_conv.weight

    def first_layer_separable_components(self):
        """Return the leading temporal/spatial SVD component of each stem filter.

        A spatial mean can cancel almost perfectly for oriented or opponent
        filters.  The leading separable component is therefore a much more
        faithful diagnostic of the learned temporal waveform.  Its sign is
        fixed deterministically at the waveform's largest-magnitude sample.
        """
        weight = self.effective_temporal_weight().detach().float().cpu()
        if weight.shape[1] != 1:
            raise ValueError("First-layer diagnostics currently require one input channel")
        flattened = weight[:, 0].flatten(start_dim=2)
        temporal, singular_values, spatial_t = torch.linalg.svd(
            flattened, full_matrices=False
        )
        temporal = temporal[:, :, 0] * singular_values[:, :1]
        spatial = spatial_t[:, 0].reshape(
            weight.shape[0], weight.shape[-2], weight.shape[-1]
        )
        anchors = temporal.abs().argmax(dim=1, keepdim=True)
        signs = temporal.gather(1, anchors).sign().clamp_min(0).mul(2).sub(1)
        temporal = temporal * signs
        spatial = spatial * signs[:, :, None]
        separability = singular_values[:, 0].square() / singular_values.square().sum(dim=1)
        return temporal, spatial, separability

    def plot_temporal_filters(self, sampling_rate: float = 240.0):
        """Plot temporal marginals and power spectra of the first-layer bank."""
        import matplotlib.pyplot as plt

        temporal, _, separability = self.first_layer_separable_components()
        t_ms = torch.arange(temporal.shape[1]) * (1000.0 / sampling_rate)
        freq = torch.fft.rfftfreq(temporal.shape[1], d=1.0 / sampling_rate)
        power = torch.fft.rfft(temporal, dim=1).abs().square()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for idx in range(temporal.shape[0]):
            label = f"f{idx} ({100 * separability[idx]:.0f}% rank-1)"
            axes[0].plot(t_ms.numpy(), temporal[idx].numpy(), label=label)
            axes[1].plot(freq.numpy(), power[idx].numpy(), label=f"f{idx}")
        axes[0].axhline(0, color="0.6", linewidth=0.8)
        axes[0].set(xlabel="history position (ms)", ylabel="leading SVD component", title="First-layer temporal profiles")
        axes[1].set(xlabel="temporal frequency (Hz)", ylabel="power", title="Temporal spectra")
        axes[1].set_xlim(0, sampling_rate / 2)
        axes[0].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    def plot_first_layer_spatial_filters(self):
        """Plot the peak-energy spatial slice of each first-layer filter."""
        import matplotlib.pyplot as plt

        _, spatial, separability = self.first_layer_separable_components()
        n = spatial.shape[0]
        fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 2.7), squeeze=False)
        vmax = spatial.abs().max().item()
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(spatial[idx], cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"f{idx}, {100 * separability[idx]:.0f}% rank-1")
            ax.axis("off")
        fig.suptitle("First-layer leading spatial components")
        fig.tight_layout()
        return fig


__all__ = [
    "DekelCore",
    "HammingConv2d",
    "HammingConv3d",
    "SignPairNormReLU",
]
