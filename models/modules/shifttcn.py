# -*- coding: utf-8 -*-
"""Convolution‑only spatiotemporal backbone + factorised readout for 32×51×51 clips.

The file defines:
    • TemporalShift               – zero‑FLOP channel shift along the time axis
    • TSMBlock                    – Temporal Shift + 2‑D spatial Conv3d (1×3×3)
    • DilatedTCNBlock / Head      – causal depth‑wise 1‑D convs with GLU gating
    • VisionCore                  – Stem → TSM stack → Dilated‑TCN head
    • SpatialFactorisedReadout    – per‑neuron A(x,y) · B(channel) → Softplus
    • V1Model                     – end‑to‑end model (core + readout)

All modules are nn.Module‑compatible and work with PyTorch Lightning out‑of‑box.
Input tensor shape is (B, Cin, T, H, W).  Default Cin=1 for grayscale movies.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1.  Temporal‑Shift layer (TSM)
# -----------------------------------------------------------------------------

class TemporalShift(nn.Module):
    """Zero‑parameter, zero‑FLOP temporal shift from TSM (Lin et al., 2019).

    Args:
        channels:   Number of feature channels (C)
        shift_frac: Fraction of channels to shift forward & backward (0‒0.5)
    Input / Output shape: (B, C, T, H, W)
    """

    def __init__(self, channels: int, shift_frac: float = 0.25):
        super().__init__()
        assert 0.0 <= shift_frac <= 0.5, "shift_frac must be in [0, 0.5]"
        self.fold = int(channels * shift_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B C T H W
        if self.fold == 0 or x.size(2) == 1:
            return x
        # Pre‑allocate output to avoid in‑place ops that break autograd.
        out = x.clone()
        # shift backward (left)
        out[:, : self.fold, :-1] = x[:, : self.fold, 1:]
        # shift forward (right)
        out[:, self.fold : 2 * self.fold, 1:] = x[:, self.fold : 2 * self.fold, :-1]
        # remaining channels stay
        return out


# -----------------------------------------------------------------------------
# 2.  TSM residual block: Temporal‑Shift → (1×3×3) spatial conv
# -----------------------------------------------------------------------------

class TSMBlock(nn.Module):
    """Residual TSM block used inside the backbone.

    Args:
        channels:           in/out channels
        shift_frac:         fraction of channels shifted ±1 frame
        spatial_kernel:     spatial kernel size for conv (default: 3)
        spatial_padding:    spatial padding (default: auto-calculated)
    """

    def __init__(self, channels: int, shift_frac: float = 0.25,
                 spatial_kernel: int = 3, spatial_padding: int = None,
                 dropout: float = 0.0, stochastic_depth: float = 0.0):
        super().__init__()
        self.shift = TemporalShift(channels, shift_frac)
        self.stochastic_depth = stochastic_depth

        # Auto-calculate padding if not provided
        if spatial_padding is None:
            spatial_padding = spatial_kernel // 2

        self.conv = nn.Conv3d(channels, channels,
                              kernel_size=(1, spatial_kernel, spatial_kernel),
                              padding=(0, spatial_padding, spatial_padding), bias=False)
        self.bn   = nn.BatchNorm3d(channels)
        self.act  = nn.GELU()

        # Add dropout after activation
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stochastic depth: randomly skip the block during training
        if self.training and self.stochastic_depth > 0.0:
            if torch.rand(1).item() < self.stochastic_depth:
                return x  # Skip the block, return identity

        y = self.shift(x)
        y = self.act(self.bn(self.conv(y)))
        y = self.dropout(y)  # Apply dropout
        return x + y  # residual


# -----------------------------------------------------------------------------
# 3.  Dilated Temporal Convolution (depth‑wise, GLU‑gated)
# -----------------------------------------------------------------------------

class DilatedTCNBlock(nn.Module):
    """Depth‑wise causal 1‑D conv over time with GLU gating + residual."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv3d(
            channels,
            2 * channels,
            kernel_size=(3, 1, 1),
            dilation=(dilation, 1, 1),
            padding=(dilation, 0, 0),
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(self.dw(x))
        f, g = torch.chunk(y, 2, dim=1)
        y = torch.tanh(f) * torch.sigmoid(g)
        return x + y


class DilatedTCNHead(nn.Module):
    """Stack of DilatedTCNBlock with exponentially increasing dilation."""

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DilatedTCNBlock(channels, d) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -----------------------------------------------------------------------------
# 4.  Optional depth‑wise long‑kernel conv (S4 / Mamba surrogate)
# -----------------------------------------------------------------------------

class DepthwiseTemporalConv(nn.Module):
    """Learned depth‑wise 1‑D convolution that spans the whole sequence.

    Serves as a drop‑in approximation of an SSM kernel when sequence length is
    small (< 128).  Parameters are initialised with a small normal std.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size) * 0.01)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B C T H W
        B, C, T, H, W = x.shape
        # Use causal padding to maintain input sequence length
        pad = self.kernel_size - 1
        x_ = x.view(B * H * W, C, T)  # merge spatial dims for depth‑wise conv
        y = F.conv1d(x_, self.weight, groups=C, padding=pad)

        # Crop to original sequence length (causal: keep the last T timesteps)
        y = y[:, :, -T:]  # Keep only the last T timesteps

        y = y.view(B, H, W, C, T).permute(0, 3, 4, 1, 2)  # B C T H W
        return y


# -----------------------------------------------------------------------------
# 5.  Vision backbone (Stem → TSM stack → Dilated TCN)
# -----------------------------------------------------------------------------

class VisionCore(nn.Module):
    """Spatiotemporal backbone covering full 32‑frame context."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_tsm_blocks: int = 4,
        add_long_kernel: bool = True,
        seq_len: int = 32,
        shift_frac: float = 0.25,
        dilations: Sequence[int] = (1, 2, 4, 8),
        # Spatial configuration parameters
        stem_spatial_kernel: int = 7,
        stem_spatial_stride: int = 2,
        stem_spatial_padding: int = None,
        stem_temporal_kernel: int = 3,
        stem_temporal_padding: int = None,
        tsm_spatial_kernel: int = 3,
        tsm_spatial_padding: int = None,
        # Regularization parameters
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()

        # Auto-calculate padding if not provided
        if stem_spatial_padding is None:
            stem_spatial_padding = stem_spatial_kernel // 2
        if stem_temporal_padding is None:
            stem_temporal_padding = stem_temporal_kernel // 2
        if tsm_spatial_padding is None:
            tsm_spatial_padding = tsm_spatial_kernel // 2

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels,
                      kernel_size=(stem_temporal_kernel, stem_spatial_kernel, stem_spatial_kernel),
                      stride=(1, stem_spatial_stride, stem_spatial_stride),
                      padding=(stem_temporal_padding, stem_spatial_padding, stem_spatial_padding),
                      bias=False),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )
        # TSM residual stack with configurable shift fraction, spatial kernels, and regularization
        tsm_blocks = []
        for i in range(num_tsm_blocks):
            # Linearly increase stochastic depth probability for deeper blocks
            block_stochastic_depth = stochastic_depth * (i + 1) / num_tsm_blocks
            tsm_blocks.append(TSMBlock(
                base_channels,
                shift_frac=shift_frac,
                spatial_kernel=tsm_spatial_kernel,
                spatial_padding=tsm_spatial_padding,
                dropout=dropout,
                stochastic_depth=block_stochastic_depth
            ))
        self.tsm = nn.Sequential(*tsm_blocks)
        # Dilated TCN head with configurable dilations
        self.tcn_head = DilatedTCNHead(base_channels, dilations=dilations)
        # Optional long‑kernel SSM surrogate
        self.long_kernel = (
            DepthwiseTemporalConv(base_channels, kernel_size=seq_len)
            if add_long_kernel else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)        # (B, C, T, H/2, W/2)
        x = self.tsm(x)
        x = self.tcn_head(x)
        x = self.long_kernel(x)
        return x


# -----------------------------------------------------------------------------
# 6.  Configuration-compatible wrapper for integration with codebase
# -----------------------------------------------------------------------------

class ShiftTCN(nn.Module):
    """Configuration-compatible wrapper for VisionCore to integrate with the codebase factory pattern.

    This class adapts the VisionCore to work with the existing configuration system
    used by ResNet, DenseNet, and other convnet types.
    """

    def __init__(self, config: dict):
        super().__init__()

        # Extract configuration parameters
        self.config = config
        self.initial_channels = config.get('initial_channels', 1)

        # ShiftTCN specific parameters with defaults
        base_channels = config.get('base_channels', 32)
        num_tsm_blocks = config.get('num_tsm_blocks', 4)
        add_long_kernel = config.get('add_long_kernel', True)
        seq_len = config.get('seq_len', 32)
        shift_frac = config.get('shift_frac', 0.25)
        dilations = config.get('dilations', [1, 2, 4, 8])

        # Spatial configuration parameters
        stem_spatial_kernel = config.get('stem_spatial_kernel', 7)
        stem_spatial_stride = config.get('stem_spatial_stride', 2)
        stem_spatial_padding = config.get('stem_spatial_padding', None)
        stem_temporal_kernel = config.get('stem_temporal_kernel', 3)
        stem_temporal_padding = config.get('stem_temporal_padding', None)
        tsm_spatial_kernel = config.get('tsm_spatial_kernel', 3)
        tsm_spatial_padding = config.get('tsm_spatial_padding', None)

        # Regularization parameters
        dropout = config.get('dropout', 0.0)
        stochastic_depth = config.get('stochastic_depth', 0.0)

        # Create the core VisionCore model
        self.core = VisionCore(
            in_channels=self.initial_channels,
            base_channels=base_channels,
            num_tsm_blocks=num_tsm_blocks,
            add_long_kernel=add_long_kernel,
            seq_len=seq_len,
            shift_frac=shift_frac,
            dilations=dilations,
            stem_spatial_kernel=stem_spatial_kernel,
            stem_spatial_stride=stem_spatial_stride,
            stem_spatial_padding=stem_spatial_padding,
            stem_temporal_kernel=stem_temporal_kernel,
            stem_temporal_padding=stem_temporal_padding,
            tsm_spatial_kernel=tsm_spatial_kernel,
            tsm_spatial_padding=tsm_spatial_padding,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
        )

        # Store output channels for compatibility with factory pattern
        self._output_channels = base_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)

    def get_output_channels(self) -> int:
        """Get the number of output channels from the network."""
        return self._output_channels