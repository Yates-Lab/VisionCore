from typing import Dict, List

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .common import chomp
from .conv_blocks import ConvBlock
from .norm_act_pool import get_activation_layer

__all__ = [
    "X3DUnit",
    "X3DNet",
]


class X3DUnit(nn.Module):
    """
    One separable X3D block:
      depthwise (tx1x1)  →
      depthwise (1xkxk)  →
      pointwise expand   →
      SiLU + GRN         →
      pointwise project  →
      residual
    """
    def __init__(self, C_in, C_out, t_kernel=5, s_kernel=3,
                 exp_ratio=4, norm='grn', act='silu', dim=3,
                 dropout=0.0, stochastic_depth=0.0, t_stride=1, s_stride=1):
        super().__init__()
        self.stochastic_depth = stochastic_depth

        conv = lambda ks, **kw: ConvBlock(
            dim=dim, in_channels=kw.pop('in_ch'), out_channels=kw.pop('out_ch'),
            conv_params=dict(type='depthwise', kernel_size=ks, stride=kw.pop('stride', 1),
                             padding=[k//2 for k in ks], bias=False),
            norm_type=norm, act_type='none', causal=False, **kw)

        self.pre_temporal = conv((t_kernel, 1, 1), in_ch=C_in, out_ch=C_in, stride=(t_stride, 1, 1))
        self.pre_spatial  = conv((1, s_kernel, s_kernel), in_ch=C_in, out_ch=C_in, stride=(1, s_stride, s_stride))

        # point-wise expand / project (standard conv) with dropout after expansion
        self.expand = ConvBlock(dim=dim, in_channels=C_in,
                                out_channels=C_in * exp_ratio,
                                conv_params=dict(type='standard', kernel_size=1, bias=False),
                                norm_type=norm, act_type=act, causal=False,
                                dropout=dropout)  # Add dropout after expansion
        self.project = ConvBlock(dim=dim, in_channels=C_in * exp_ratio,
                                 out_channels=C_out,
                                 conv_params=dict(type='standard', kernel_size=1, bias=False),
                                 norm_type=norm, act_type='none', causal=False)

        # residual projection if needed
        self.need_proj = C_in != C_out or t_stride > 1 or s_stride > 1
        if self.need_proj:
            self.proj = ConvBlock(dim=dim, in_channels=C_in, out_channels=C_out,
                                  conv_params=dict(type='standard', kernel_size=1,
                                                   stride=(t_stride, s_stride, s_stride), bias=False),
                                  norm_type=norm, act_type='none', causal=False)

        self.act = get_activation_layer(act)

    def forward(self, x):
        identity = self.proj(x) if self.need_proj else x

        # Stochastic depth: randomly skip the block during training
        if self.training and self.stochastic_depth > 0.0:
            if torch.rand(1).item() < self.stochastic_depth:
                # Skip the block, return identity
                return identity

        y = self.pre_temporal(x)
        y = self.pre_spatial(y)
        y = self.project(self.expand(y))

        # Handle temporal dimension mismatch due to stride
        if y.shape != identity.shape:
            # Crop identity to match y's temporal dimension
            if y.dim() == 5 and identity.dim() == 5 and y.shape[2] != identity.shape[2]:
                min_time = min(y.shape[2], identity.shape[2])
                y = y[:, :, :min_time]
                identity = identity[:, :, :min_time]

        return self.act(y + identity)


class SEBlock(nn.Module):
    """Squeeze‑and‑Excitation block — light‑weight channel‑attention.

    Args:
        channels: in/out channels.
        rd_ratio: reduction ratio (≈ 16).  If 0 disables SE.
    """

    def __init__(self, channels: int, rd_ratio: int = 16, dim: int = 3):
        super().__init__()
        if rd_ratio == 0:
            self.attn = nn.Identity()
            return

        rd = max(channels // rd_ratio, 1)
        if dim == 3:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # B,C,1,1,1
                nn.Conv3d(channels, rd, 1, bias=True),
                nn.SiLU(),
                nn.Conv3d(rd, channels, 1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, rd, 1, bias=True),
                nn.SiLU(),
                nn.Conv2d(rd, channels, 1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        w = self.attn(x)
        return x * w


class X3DNet(nn.Module):
    """Flexible X3D backbone with temporal & spatial stride scheduling, optional SE attention,
    and stochastic depth.

    **Config keys** (all optional unless stated):
    ```python
    {
        # === required ===
        "dim": 3,
        "initial_channels": 3,
        "channels": [96, 192, 384],  # out‑channels per stage

        # === architecture ===
        "depth": [2, 5, 5],          # blocks per stage (int → broadcast)
        "t_kernel": 5,               # default temporal kernel (first stage)
        "s_kernel": 3,               # spatial kernel size (odd)
        "exp_ratio": 4,              # bottleneck expansion ratio in X3DUnit
        "stride_stages": [1, 2, 2],  # temporal stride per stage
        "spatial_stride_stages": [2, 2, 2],  # spatial stride (H,W) per stage
        "lite_lk": False, "lk_every": 2,    # large‑kernel trick (optional)

        # === attention ===
        "attention": "se",           # "se" | None
        "attention_stage": -1,       # stage index (‑1 → last)
        "se_reduction": 16,          # rd_ratio for SE

        # === regularisation ===
        "dropout": 0.1,              # dropout rate (default 0.1 for good regularization)
        "stochastic_depth": 0.1,     # stochastic depth rate (default 0.1, important for deep models)

        # === normalisation / activation ===
        "norm_type": "grn",
        "act_type": "silu",

        # === misc ===
        "checkpointing": False,
    }
    ```
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.dim = config.get("dim", 3)
        self.initial_channels = config["initial_channels"]
        self.use_checkpointing = config.get("checkpointing", False)
        self.layers = nn.ModuleList()
        self._build_network()  # sets self._final_out_channels

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def get_output_channels(self) -> int:
        return self._final_out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        for layer in self.layers:
            if (
                self.use_checkpointing and self.training and isinstance(layer, X3DUnit)
            ):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    # ---------------------------------------------------------------------
    # internal: builder
    # ---------------------------------------------------------------------
    def _build_network(self) -> None:  # noqa: C901  — long but readable
        cfg = self.config
        C_in = self.initial_channels

        width: List[int] = cfg["channels"]
        depth: List[int] = cfg.get("depth", [1] * len(width))
        if isinstance(depth, int):
            depth = [depth] * len(width)
        if len(depth) == 1:
            depth *= len(width)
        assert len(width) == len(depth), "depth length must match channels"

        t_stride = cfg.get("stride_stages", [1] * len(width))
        if len(t_stride) == 1:
            t_stride *= len(width)
        spatial_stride = cfg.get("spatial_stride_stages", [1] * len(width))
        if len(spatial_stride) == 1:
            spatial_stride *= len(width)
        assert len(t_stride) == len(width) == len(spatial_stride)

        # convenience aliases
        t_kernel_def = cfg.get("t_kernel", 5)
        s_kernel_base = cfg.get("s_kernel", 3)
        exp = cfg.get("exp_ratio", 4)
        norm = cfg.get("norm_type", "grn")
        act = cfg.get("act_type", "silu")
        dropout = cfg.get("dropout", 0.1)  # Default to 0.1 for good regularization
        sd = cfg.get("stochastic_depth", 0.1)  # Default to 0.1, important for deep models

        # attention settings
        attn_type = cfg.get("attention", None)
        attn_stage = cfg.get("attention_stage", -1)
        se_rd_ratio = cfg.get("se_reduction", 16)

        lite_lk = cfg.get("lite_lk", False)
        lk_every = cfg.get("lk_every", 2)

        for stage, (blocks, C_out, t_s, s_s) in enumerate(
            zip(depth, width, t_stride, spatial_stride)
        ):
            # —— kernel schedule per stage ——
            largek = lite_lk and stage % lk_every == 1
            t_k = 3 if largek else t_kernel_def
            s_k = 7 if largek else s_kernel_base

            # —— build first block (may downsample) ——
            first = X3DUnit(
                C_in,
                C_out,
                t_k,
                s_k,
                exp,
                norm,
                act,
                dim=self.dim,
                t_stride=t_s,
                s_stride=s_s,
                dropout=dropout,
                stochastic_depth=sd,
            )
            self.layers.append(first)

            # —— remaining blocks ——
            for _ in range(blocks - 1):
                blk = X3DUnit(
                    C_out,
                    C_out,
                    t_k,
                    s_k,
                    exp,
                    norm,
                    act,
                    dim=self.dim,
                    dropout=dropout,
                    stochastic_depth=sd,
                )
                self.layers.append(blk)

            # —— optional stage‑level attention ——
            if attn_type and (attn_stage == stage or (attn_stage == -1 and stage == len(width) - 1)):
                if attn_type.lower() == "se":
                    self.layers.append(SEBlock(C_out, se_rd_ratio, dim=self.dim))
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            C_in = C_out  # for next stage

        self._final_out_channels = C_in