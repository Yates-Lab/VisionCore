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

__all__ = ['ConvBase', 'StandardConv', 'StackedConv2d', 'WindowedConv2d', 'SeparableWindowedConv2D', 'AntiAliasedConv1d']

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
        win = 1.0
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

        elif weights.ndim == 3: # 1D conv

            # Handle 1D convolution weights: [out_channels, in_channels, width]
            out_channels, in_channels, width = weights.shape
            weights = weights.unsqueeze(-1)  # Add a dummy dimension for height

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
                    
                    ax.plot(reshaped_weights.squeeze().numpy())
            
            # Hide any unused subplots
            for i in range(in_channels, len(axs_flat)):
                axs_flat[i].axis('off')
                axs_flat[i].set_title(f'Input Channel {i_in}', fontsize=10)
                axs_flat[i].set_ylabel('Weight Value')
                axs_flat[i].set_xlabel('Kernel Index')
                axs_flat[i].grid(True, alpha=0.3)
                axs_flat[i].tick_params(axis='both', labelsize=8)
                axs_flat[i].set_xticks(np.arange(0, width, max(1, width//10)))
                axs_flat[i].set_yticks(np.arange(global_min, global_max, (global_max - global_min)/10))
        
            fig.suptitle(f'{self.__class__.__name__} 1D Weights: {out_channels} Out-Channels, {in_channels} In-Channels',
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        return fig, axs

ConvNds = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}

class StandardConv(ConvBase):
    """Standard N-Dimensional Convolution Wrapper (2D or 3D)."""
    def __init__(self, dim: int, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'replicate'):
        super().__init__()
        if dim not in [1, 2, 3]: raise ValueError(f"Unsupported dim for StandardConv: {dim}.")
        ConvNd = ConvNds[dim]

        self.conv = ConvNd(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups,
                           bias=bias, padding_mode=padding_mode)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)
    @property
    def weight(self) -> torch.Tensor: return self.conv.weight

class DepthwiseConv(ConvBase):
    """Depth-wise-separable conv (2-D or 3-D).  Returns full fused kernel for plotting."""
    def __init__(self, dim: int, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 bias: bool = True, padding_mode: str = "replicate"):

        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"Unsupported dim {dim} for DepthwiseConv")

        Conv = nn.Conv2d if dim == 2 else nn.Conv3d

        # depth-wise (groups = Cin)
        self.depthwise = Conv(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False, padding_mode=padding_mode
        )
        # point-wise 1×1(×1)
        self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    # -------- weight property for plotting --------
    @property
    def weight(self) -> torch.Tensor:
        """
        Effective fused kernel of shape  
        (C_out, C_in, kT, kH, kW)  ─ exactly what plot_weights() expects.
        """

        # permute dim 0 and 1 for depthwise (handle 2D and 3D)
        dw = self.depthwise.weight
        if dw.dim() == 4: dw = dw.permute(1, 0, 2, 3)  # 2D
        else: dw = dw.permute(1, 0, 2, 3, 4)  # 3D

        pw = self.pointwise.weight

        eff = pw * dw

        return eff
    


CONV_LAYER_MAP = {
    'standard': StandardConv,
    'depthwise': DepthwiseConv,
}