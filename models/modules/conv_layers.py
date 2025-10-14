"""
Convolutional network modules for DataYatesV1.

This module contains convolutional network components for feature extraction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Sequence, Tuple, Optional
from torch.nn.utils import weight_norm, remove_weight_norm

__all__ = ['StandardConv', 'DepthwiseConv']

'''
Helpers for anti-aliasing convolutions
'''

def _get_window_fn(window_type, window_size, **window_kwargs):
    """
    Get a window function from torch.signal.windows.

    Args:
        window_type (str): Name of the window function in torch.signal.windows
        window_size (int): Size of the window
        **window_kwargs: Additional keyword arguments to pass to the window function

    Returns:
        torch.Tensor: Window function values
    """
    # Check if the window type exists in torch.signal.windows
    assert hasattr(torch.signal.windows, window_type), f'Window type {window_type} not found in torch.signal.windows'

    # Get the window function
    window_function = getattr(torch.signal.windows, window_type)

    # Create the window
    window = window_function(window_size, **window_kwargs)

    return window

class AntiAliasedMixin:
    # expects: self.aa_time, self.aa_freq, self.aa_window, self.aa_window_kwargs,
    #          self.aa_fft_pad, self.aa_double_time

    def _win1d(self, size, device, dtype):
        w = _get_window_fn(self.aa_window, size, **(self.aa_window_kwargs or {})).to(device=device, dtype=dtype)
        return w / w.max().clamp_min(torch.finfo(w.dtype).eps)

    def _kernel_axes(self, dim):  # which axes are kernel dims in weight
        return {1: (-1,), 2: (-2, -1), 3: (-3, -2, -1)}[dim]

    def _apply_spacetime_window(self, w: torch.Tensor, dim: int) -> torch.Tensor:
        axes = self._kernel_axes(dim)
        win = torch.ones(1, device=w.device, dtype=w.dtype)

        for ax in axes:
            vec = self._win1d(w.shape[ax], w.device, w.dtype)
            shape = [1]*w.ndim; shape[ax] = w.shape[ax]
            win = win * vec.view(*shape)
        return w * win

    # 1D: rFFT + half-window (fast)
    def _freq_aa_1d(self, w: torch.Tensor) -> torch.Tensor:
        k = w.shape[-1]
        Nfft = int(np.fft.next_fast_len(k * self.aa_fft_pad))
        W = torch.fft.rfft(w, n=Nfft, dim=-1)
        bins = W.shape[-1]
        wf = self._win1d(2*bins - 2, w.device, w.dtype)  # full
        half = wf[bins - 1:]                              # descending half (1→0 to Nyquist)
        W = W * half.view(1, 1, -1)
        out = torch.fft.irfft(W, n=Nfft, dim=-1)[..., :k]
        if self.aa_time and self.aa_double_time:
            out = self._apply_spacetime_window(out, dim=1)
        return out

    # 2D/3D: FFTN + fftshift + separable per-axis windows
    def _freq_aa_nd(self, w: torch.Tensor, dim: int) -> torch.Tensor:
        axes = self._kernel_axes(dim)
        kshape = [w.shape[a] for a in axes]
        s_fft = [int(np.fft.next_fast_len(k * self.aa_fft_pad)) for k in kshape]

        W = torch.fft.fftn(w, s=s_fft, dim=axes)
        W = torch.fft.fftshift(W, dim=axes)

        fw = 1.0
        for ax, N in zip(axes, s_fft):
            v = self._win1d(N, w.device, w.dtype)
            shape = [1]*W.ndim; shape[ax] = N
            fw = fw * v.view(*shape)

        W = W * fw
        W = torch.fft.ifftshift(W, dim=axes)
        wf = torch.fft.ifftn(W, s=s_fft, dim=axes).real

        # center-crop back to original kernel support
        slices = [slice(None), slice(None)]
        for N, k in zip(s_fft, kshape):
            start = (N - k) // 2
            slices.append(slice(start, start + k))
        out = wf[tuple(slices)]

        if self.aa_time and self.aa_double_time:
            out = self._apply_spacetime_window(out, dim=dim)
        return out

    def _aa_weight(self, w: torch.Tensor, dim: int) -> torch.Tensor:
        if self.aa_time:
            w = self._apply_spacetime_window(w, dim)
        if self.aa_freq:
            w = self._freq_aa_1d(w) if dim == 1 else self._freq_aa_nd(w, dim)
        return w


class ConvBase(AntiAliasedMixin, nn.Module):
    """Base class for convolution modules with weight plotting."""
    def __init__(self,
            aa_time: bool = False,
            aa_freq: bool = False,
            aa_window: str = 'hann',
            aa_window_kwargs: Optional[dict] = None,
            aa_fft_pad: int = 2,
            aa_double_time: bool = False):
        
        super().__init__()
        self.aa_time = aa_time
        self.aa_freq = aa_freq
        self.aa_window = aa_window
        self.aa_window_kwargs = aa_window_kwargs
        self.aa_fft_pad = aa_fft_pad
        self.aa_double_time = aa_double_time


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    def plot_weights(self, scale_globally=True) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
        """
        Plot convolution weights using torchvision.utils.make_grid for better visualization.

        Args:
            scale_globally (bool): If True, normalize all kernels with one global min/max.
                                  If False, scale each kernel independently.

        Returns:
            Tuple of (figure, axes) or (None, None) if plotting is not possible
        """
        try:
            import torchvision.utils as vutils
            weights = self.weight.detach().cpu()
        except (NotImplementedError, AttributeError, ImportError):
            # print(f"Weight property not available for {self.__class__.__name__} or torchvision not installed. Cannot plot.")
            return None, None # Simplified error handling

        if weights.ndim not in [3, 4, 5]:
            # print(f"Weight plotting not implemented for ndim {weights.ndim}.")
            return None, None

        if weights.ndim == 5: # 3D conv

            # Handle 3D convolution weights: [out_channels, in_channels, depth, height, width]
            out_channels, in_channels, depth, height, width = weights.shape

            # Create a figure with subplots for each input channel
            n_in_cols = min(4, in_channels)  # Limit to 4 columns for readability
            n_in_rows = int(np.ceil(in_channels / n_in_cols))

            fig, axs = plt.subplots(n_in_rows, n_in_cols,
                                    figsize=(n_in_cols * 2, 2 * out_channels),
                                    squeeze=False)
            axs_flat = axs.flatten()

            # Global min/max for consistent scaling if requested
            global_min = weights.min().item() if scale_globally else None
            global_max = weights.max().item() if scale_globally else None

            for i_in in range(in_channels):
                if i_in < len(axs_flat):
                    ax = axs_flat[i_in]
                    ax.axis('off')

                    # Extract weights for this input channel: [out_channels, depth, height, width]
                    channel_weights = weights[:, i_in]

                    # Reshape to [out_channels * depth, 1, height, width] for make_grid
                    reshaped_weights = channel_weights.reshape(-1, 1, height, width)

                    # Create grid with depth kernels per row (each row is one output channel)
                    grid = vutils.make_grid(
                        reshaped_weights,
                        nrow=depth,
                        normalize=True,
                        scale_each=not scale_globally,
                        padding=1,
                        pad_value=0.5,  # Light gray padding
                        value_range=(global_min, global_max) if scale_globally else None
                    )

                    # Display the grid
                    ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
                    ax.set_title(f'Input Channel {i_in}', fontsize=10)

            # Hide any unused subplots
            for i in range(in_channels, len(axs_flat)):
                axs_flat[i].axis('off')

            fig.suptitle(f'{self.__class__.__name__} 3D Weights: {out_channels} Out-Channels, {in_channels} In-Channels, {depth} Depth',
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        elif weights.ndim == 4: # 2D conv
            
            # Handle 2D convolution weights: [out_channels, in_channels, height, width]
            out_channels, in_channels, height, width = weights.shape

            # Create a figure with subplots for each input channel
            n_in_cols = min(4, in_channels)  # Limit to 4 columns for readability
            n_in_rows = int(np.ceil(in_channels / n_in_cols))

            fig, axs = plt.subplots(n_in_rows, n_in_cols,
                                    figsize=(n_in_cols * 3, n_in_rows * 2.5),
                                    squeeze=False)
            axs_flat = axs.flatten()

            # Global min/max for consistent scaling if requested
            global_min = weights.min().item() if scale_globally else None
            global_max = weights.max().item() if scale_globally else None

            for i_in in range(in_channels):
                if i_in < len(axs_flat):
                    ax = axs_flat[i_in]
                    ax.axis('off')

                    # Extract weights for this input channel: [out_channels, height, width]
                    channel_weights = weights[:, i_in]

                    # Reshape to [out_channels, 1, height, width] for make_grid
                    reshaped_weights = channel_weights.unsqueeze(1)

                    # Create grid with sqrt(out_channels) kernels per row
                    nrow = int(np.ceil(np.sqrt(out_channels)))
                    grid = vutils.make_grid(
                        reshaped_weights,
                        nrow=nrow,
                        normalize=True,
                        scale_each=not scale_globally,
                        padding=1,
                        pad_value=0.5,  # Light gray padding
                        value_range=(global_min, global_max) if scale_globally else None
                    )

                    # Display the grid
                    ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
                    ax.set_title(f'Input Channel {i_in}', fontsize=10)

            # Hide any unused subplots
            for i in range(in_channels, len(axs_flat)):
                axs_flat[i].axis('off')

            fig.suptitle(f'{self.__class__.__name__} 2D Weights: {out_channels} Out-Channels, {in_channels} In-Channels',
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        elif weights.ndim == 3:  # 1D conv: [Cout, Cin, W]
            out_channels, in_channels, width = weights.shape
            weights_img = weights.unsqueeze(-2)  # -> [Cout, Cin, 1, W]
            n_in_cols = min(4, in_channels)
            n_in_rows = int(np.ceil(in_channels / n_in_cols))

            fig, axs = plt.subplots(n_in_rows, n_in_cols,
                                    figsize=(n_in_cols * 3, n_in_rows * 2.0),
                                    squeeze=False)
            axs_flat = axs.flatten()
            global_min = weights.min().item() if scale_globally else None
            global_max = weights.max().item() if scale_globally else None

            import torchvision.utils as vutils
            for i_in in range(in_channels):
                if i_in >= len(axs_flat): break
                ax = axs_flat[i_in]
                ax.axis('off')
                # [Cout, 1, 1, W] grid
                reshaped = weights_img[:, i_in].unsqueeze(1)  # [Cout, 1, 1, W]
                nrow = int(np.ceil(np.sqrt(out_channels)))
                grid = vutils.make_grid(
                    reshaped,
                    nrow=nrow,
                    normalize=True,
                    scale_each=not scale_globally,
                    padding=1,
                    pad_value=0.5,
                    value_range=(global_min, global_max) if scale_globally else None
                )
                ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
                ax.set_title(f'Input Channel {i_in}', fontsize=10)

            for j in range(in_channels, len(axs_flat)):
                axs_flat[j].axis('off')

            fig.suptitle(f'{self.__class__.__name__} 1D Weights: {out_channels} Out-Channels, {in_channels} In-Channels',
                        fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, axs

ConvNds = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}

class StandardConv(ConvBase):
    """Standard N-Dimensional Convolution Wrapper (2D or 3D)."""
    def __init__(self, dim: int, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'replicate', **kwargs):
        super().__init__(**kwargs)
        if dim not in [1, 2, 3]: raise ValueError(f"Unsupported dim for StandardConv: {dim}.")
        self.dim = dim
        ConvNd = ConvNds[dim]

        self.conv = ConvNd(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups,
                           bias=bias, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv._conv_forward(x, self.weight, self.conv.bias)

    def apply_weight_norm(self, dim: int = 0):
        self.conv = weight_norm(self.conv, name='weight', dim=dim)

    def remove_weight_norm(self):
        self.conv = remove_weight_norm(self.conv)
        
    @property
    def weight(self) -> torch.Tensor:
        w = self.conv.weight
        if self.aa_time or self.aa_freq:
            w = self._aa_weight(w, self.dim)
        return w

class DepthwiseConv(ConvBase):
    """Depth-wise-separable conv (1-D, 2-D, or 3-D). Returns fused kernel for plotting."""
    def __init__(self, dim: int, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 bias: bool = True, padding_mode: str = "replicate", **kwargs):

        super().__init__(**kwargs)
        if dim not in (1, 2, 3):
            raise ValueError(f"Unsupported dim {dim} for DepthwiseConv")

        self.dim = dim
        Conv = ConvNds[dim]

        # depth-wise (groups = Cin)
        self.depthwise = Conv(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False, padding_mode=padding_mode
        )
        # point-wise 1×1(×1)
        self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Anti-alias only the spatial/temporal kernel (depthwise)
        dw_weight = self.depthwise.weight
        if self.aa_time or self.aa_freq:
            dw_weight = self._aa_weight(dw_weight, self.dim)

        # Use AA'd depthwise kernels; pointwise is not windowed
        x = self.depthwise._conv_forward(x, dw_weight, self.depthwise.bias)
        return self.pointwise(x)

    # -------- weight property for plotting (fused kernel) --------
    def apply_weight_norm(self, dim: int = 0):
        self.pointwise = weight_norm(self.pointwise, name='weight', dim=dim)
        self.depthwise = weight_norm(self.depthwise, name='weight', dim=dim)
    
    def remove_weight_norm(self):
        self.pointwise = remove_weight_norm(self.pointwise)
        self.depthwise = remove_weight_norm(self.depthwise)

    @property
    def weight(self) -> torch.Tensor:
        """
        Effective fused kernel of shape:
          - 1D: (C_out, C_in, K)
          - 2D: (C_out, C_in, H, W)
          - 3D: (C_out, C_in, D, H, W)
        """
        dw = self.depthwise.weight
        if self.aa_time or self.aa_freq:
            dw = self._aa_weight(dw, self.dim)

        # Reorder depthwise to put Cin in the second dim for broadcasting with pointwise
        if dw.dim() == 3:       # 1D: [Cin, 1, K] -> [1, Cin, K]
            dw = dw.permute(1, 0, 2)
        elif dw.dim() == 4:     # 2D: [Cin, 1, H, W] -> [1, Cin, H, W]
            dw = dw.permute(1, 0, 2, 3)
        elif dw.dim() == 5:     # 3D: [Cin, 1, D, H, W] -> [1, Cin, D, H, W]
            dw = dw.permute(1, 0, 2, 3, 4)
        else:
            raise RuntimeError(f"Unexpected depthwise weight ndim: {dw.dim()}")

        pw = self.pointwise.weight  # shapes:
        # 1D: (C_out, C_in, 1)
        # 2D: (C_out, C_in, 1, 1)
        # 3D: (C_out, C_in, 1, 1, 1)

        # Broadcasting multiply to get fused (C_out, C_in, K/HW/DHW)
        eff = pw * dw
        return eff
    


CONV_LAYER_MAP = {
    'standard': StandardConv,
    'depthwise': DepthwiseConv,
}