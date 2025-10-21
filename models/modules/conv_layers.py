"""
Convolutional Layers.

Custom layers include
- StandardConv: standard 3D convolution
- DepthwiseConv: depthwise separable 3D convolution

We always use 3D convolutions, even for 2D data. The first dimension is always time.

Anti-aliasing is supported for both signal and frequency domains.
Weight norm is supported for all layers.


"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from torch.nn.utils import parametrize

__all__ = ['StandardConv', 'DepthwiseConv']

# -------------
# Windowing and WeightNorm
# -------------

def _get_window(window_type: str, window_size: int, power: float = 1.0, **window_kwargs):
    """Get a window function from torch.signal.windows."""
    w = getattr(torch.signal.windows, window_type)(window_size, **window_kwargs).pow(power)
    w = w / w.max().clamp_min(1e-12)
    return w

def _separable_mask_from_sizes(
    kt: int, kh: int, kw: int, *,
    window: str, window_kwargs: dict,
    device, dtype
) -> torch.Tensor:
    """
    Build a separable 3D window mask M of shape (1, 1, kt, kh, kw).
    Apply a 1D window on axes with size >= 5; otherwise ones.
    Each 1D window is peak-normalized so M.max() == 1.
    """
    def win_or_ones(n: int):
        if n >= 5:
            return _get_window(window, n, **window_kwargs).to(device, dtype)
        return torch.ones(n, device=device, dtype=dtype)

    wt = win_or_ones(kt).view(kt, 1, 1)  # (T,1,1)
    wh = win_or_ones(kh).view(1, kh, 1)  # (1,H,1)
    ww = win_or_ones(kw).view(1, 1, kw)  # (1,1,W)

    M = wt * wh * ww                      # (T,H,W), peak ≤ 1 and =1 if all axes got windows with peak 1
    return M.view(1, 1, kt, kh, kw)       # broadcast over (Cout, Cin/groups)


class _WindowParam(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, w):
        return w * self.mask

class _WeightNorm(nn.Module):
    def __init__(self, dim=0, keep_unit_norm=False, eps=1e-12):
        super().__init__()
        self.dim, self.eps, self._initd = dim, eps, False
        self.keep_unit_norm = keep_unit_norm

    def _lazy_init(self, w):
        if self._initd: return
        self.v = nn.Parameter(w.detach().clone())
        red = tuple(d for d in range(w.ndim) if d != self.dim)
        if self.keep_unit_norm:
            g0 = torch.ones(w.size(self.dim), device=w.device, dtype=w.dtype)
            self.g = nn.Parameter(g0, requires_grad=False)   # unit norm
        else:
            g0 = torch.linalg.vector_norm(w, dim=red)        # scale learnable
            self.g = nn.Parameter(g0, requires_grad=True)

        self.register_parameter("v", self.v)
        self.register_parameter("g", self.g)
        self._initd = True

    def forward(self, w):
        self._lazy_init(w)
        red = tuple(d for d in range(self.v.ndim) if d != self.dim)
        n = torch.linalg.vector_norm(self.v, dim=red, keepdim=True).clamp_min(self.eps)
        shape = [1]*self.v.ndim; shape[self.dim] = self.g.shape[0]
        return self.g.view(*shape) * (self.v / n)

    
# -------------
# Base + plotting
# -------------
class ConvBase(nn.Module):
    """
    Base class: exposes effective weight and a plotting helper.
    Subclasses must populate a conv module with .weight (after parametrizations).
    """
    def __init__(self):
        super().__init__()

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    def plot_weights(self, scale_globally: bool = True):
        """
        Visualize effective weights (after AA + WN) per input channel.
        1D: [Cout, Cin, K]   → grids of [Cout,1,1,K]
        2D: [Cout, Cin, H,W] → grids of [Cout,1,H,W]
        3D: [Cout, Cin, D,H,W] → grids of [Cout*D,1,H,W] (depth tiled along batch)
        """
        try:
            import math
            import matplotlib.pyplot as plt
            import torchvision.utils as vutils
        except Exception:
            return None, None

        w = self.weight.detach().cpu()
        assert w.ndim == 5, f"Expected 5D weight, got {w.shape}"

        Cout, Cin, D, H, W = w.shape
        per_in = [w[:, i].reshape(Cout * D, 1, H, W) for i in range(Cin)]
        if D == 1:
            label = "2D"
            nrow = int(math.ceil(Cout**0.5))
        elif (H==1) & (W==1):
            label = "1D"
            nrow = D
        else:
            label = "3D"
            nrow = D

        gmin = w.min().item() if scale_globally else None
        gmax = w.max().item() if scale_globally else None
        ncols = min(4, len(per_in))
        nrows = (len(per_in) + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for i, ax in enumerate(axs.flatten()):
            if i >= len(per_in):
                ax.axis("off"); continue
            K = per_in[i]

            grid = vutils.make_grid(
                K, nrow=nrow, normalize=True, scale_each=not scale_globally,
                padding=1, pad_value=0.5,
                value_range=(gmin, gmax) if scale_globally else None
            )
            ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(f"Input {i}", fontsize=10)
        fig.suptitle(f"{self.__class__.__name__} Weights ({label})", fontsize=12)
        fig.tight_layout()
        return fig, axs

# -------------
# Standard conv
# -------------  
class StandardConv(ConvBase):
    """
    Standard convolution implemented with Conv3d, always expecting input (B, C, T, H, W).

    - kernel_size, stride, padding, dilation MUST be 3-tuples (T,H,W).
    - Windowing: separable per-axis window applied only on axes with size >= 5.
    - WeightNorm: optional, applied AFTER windowing so it controls the effective weights.
    - Causal/asymmetric padding should be handled by the caller (e.g., ConvBlock with F.pad).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int],
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (0, 0, 0),
                 dilation: Tuple[int, int, int] = (1, 1, 1),
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "replicate",
                 *,
                 # windowing
                 aa_signal: bool = False,
                 aa_window: str = "hann",
                 aa_window_kwargs: Optional[dict] = None,
                 # weight norm
                 use_weight_norm: bool = False,
                 keep_unit_norm: bool = False,
                 weight_norm_dim: int = 0):
        super().__init__()
        assert len(kernel_size) == 3 and len(stride) == 3 and len(padding) == 3 and len(dilation) == 3, \
            "All size/step args must be 3-tuples (T,H,W)."

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode
        )

        # Register WeightNorm
        if use_weight_norm:
            parametrize.register_parametrization(self.conv, "weight", _WeightNorm(dim=0, keep_unit_norm=keep_unit_norm))          # index 1

        # Register anti-alias windowing as a parametrization (if enabled).
        if aa_signal:
            kt, kh, kw = kernel_size
            mask = _separable_mask_from_sizes(
                kt, kh, kw,
                window=aa_window, window_kwargs=aa_window_kwargs or {},
                device=self.conv.weight.device, dtype=self.conv.weight.dtype
            )
            parametrize.register_parametrization(self.conv, "weight", _WindowParam(mask))

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (B, C, T, H, W); causal/external padding handled by caller.
        return self.conv(x)

    @property
    def weight(self) -> torch.Tensor:
        # Effective (windowed + weight-normed) weight as seen by the conv.
        return self.conv.weight
    
# -------------
# Depthwise conv
# -------------  
class DepthwiseConv(ConvBase):
    """
    Depth-wise separable conv using Conv3d:
      depthwise: groups = in_channels, kernel=(kt,kh,kw)
      pointwise: 1x1x1
    Expects input (B, C, T, H, W). Causal/asymmetric padding stays external.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int],
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (0, 0, 0),
                 dilation: Tuple[int, int, int] = (1, 1, 1),
                 *,
                 bias: bool = True,
                 padding_mode: str = "replicate",
                 # windowing
                 aa_signal: bool = False,
                 aa_window: str = "hann",
                 aa_window_kwargs: Optional[dict] = None,
                 # weight norm
                 use_weight_norm: bool = False,
                 keep_unit_norm: bool = False,
                 weight_norm_dim: int = 0):
        super().__init__()
        assert len(kernel_size) == 3 and len(stride) == 3 and len(padding) == 3 and len(dilation) == 3, \
            "All size/step args must be 3-tuples (T,H,W)."

        # Depthwise conv (groups = Cin)
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False, padding_mode=padding_mode
        )

        # Pointwise 1×1×1
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1, bias=bias
        )

        if use_weight_norm:
            parametrize.register_parametrization(self.depthwise, "weight", _WeightNorm(dim=weight_norm_dim, keep_unit_norm=keep_unit_norm))
            parametrize.register_parametrization(self.pointwise, "weight", _WeightNorm(dim=0, keep_unit_norm=keep_unit_norm))

        # Anti-alias window on depthwise only (computed once)
        if aa_signal:
            kt, kh, kw = kernel_size
            mask = _separable_mask_from_sizes(
                kt, kh, kw,
                window=aa_window, window_kwargs=aa_window_kwargs or {},
                device=self.depthwise.weight.device, dtype=self.depthwise.weight.dtype
            )
            parametrize.register_parametrization(self.depthwise, "weight", _WindowParam(mask))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,H,W); any causal/asymmetric padding already applied by caller
        y = self.depthwise(x)
        y = self.pointwise(y)
        return y

    @property
    def weight(self) -> torch.Tensor:
        """
        Effective fused kernel for plotting:
          returns [C_out, C_in, kt, kh, kw]
        """
        pw = self.pointwise.weight         # [C_out, C_in, 1, 1, 1]
        dw = self.depthwise.weight         # [C_in, 1, kt, kh, kw] (already AA’d / WN’d if enabled)
        # Broadcast depthwise as (1, C_in, kt, kh, kw)
        dw_bc = dw.view(dw.shape[0], 1, *dw.shape[2:]).permute(1, 0, 2, 3, 4)
        return pw * dw_bc                  # [C_out, C_in, kt, kh, kw]


CONV_LAYER_MAP = {
    'standard': StandardConv,
    'depthwise': DepthwiseConv,
}