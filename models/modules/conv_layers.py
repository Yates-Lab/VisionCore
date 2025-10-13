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

def _hann(L, device, dtype):
    if L == 1:
        return torch.ones(1, device=device, dtype=dtype)
    n = torch.arange(L, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (L - 1))

def _kaiser(L, beta, device, dtype):
    # torch doesn’t have i0; use torch.special.i0 (PyTorch >= 1.8)
    n = torch.arange(L, device=device, dtype=dtype)
    x = (2.0*n)/(L-1) - 1.0
    denom = torch.special.i0(torch.tensor(beta, device=device, dtype=dtype))
    return torch.special.i0(beta * torch.sqrt((1 - x**2).clamp(min=0))) / denom

def _raised_cosine_lowpass(freqs, cutoff, transition):
    """
    freqs: [Nfft//2+1] normalized to Nyquist=0.5 (i.e., cycles/sample in [0, 0.5])
    cutoff, transition in [0, 0.5]; passband=[0, cutoff], stopband >= cutoff+transition
    """
    h = torch.zeros_like(freqs)
    passband = freqs <= cutoff
    stopband  = freqs >= (cutoff + transition)
    trans = (~passband) & (~stopband)

    h[passband] = 1.0
    if trans.any():
        # cosine ramp from 1→0 across the transition bandwidth
        x = (freqs[trans] - cutoff) / transition  # in (0,1)
        h[trans] = 0.5 * (1 + torch.cos(math.pi * x))  # smooth taper
    # stopband stays 0
    return h

class AntiAliasedConv1d(nn.Conv1d):
    """
    Drop-in replacement for nn.Conv1d that windows the learned kernel
    in both time and frequency domains to reduce aliasing/ringing.

    Same signature as nn.Conv1d plus optional anti-aliasing kwargs:
      - aa_time_window:  'none' | 'hann' | 'kaiser' (default 'hann')
      - aa_kaiser_beta:  float (only if aa_time_window='kaiser', default 8.6)
      - aa_cutoff:       normalized cutoff (0..0.5, Nyquist=0.5). Default 0.45
      - aa_transition:   transition width (0..0.5-cutoff). Default 0.05
      - aa_fft_pad:      pad factor for FFT length (>=1). Default 2
      - aa_enable:       bool to toggle all anti-aliasing. Default True
      - aa_enable_time:  bool to toggle time-domain windowing. Default True
      - aa_enable_freq:  bool to toggle frequency-domain filtering. Default True
      - aa_double_window: bool to apply time window after IFFT to reduce ripple. Default True
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None,
        *,
        aa_time_window: str = 'hann',
        aa_kaiser_beta: float = 8.6,
        aa_cutoff: float = 0.45,
        aa_transition: float = 0.05,
        aa_fft_pad: int = 2,
        aa_enable: bool = True,
        aa_enable_time: bool = True,
        aa_enable_freq: bool = False,
        aa_double_window: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, device=device, dtype=dtype)

        # Store AA config
        self.aa_enable = aa_enable
        self.aa_enable_time = aa_enable_time
        self.aa_enable_freq = aa_enable_freq
        self.aa_double_window = aa_double_window
        self.aa_time_window = aa_time_window.lower()
        self.aa_kaiser_beta = float(aa_kaiser_beta)
        self.aa_cutoff = float(aa_cutoff)
        self.aa_transition = float(aa_transition)
        self.aa_fft_pad = int(aa_fft_pad)

        # Validate
        k = self.kernel_size[0]
        if self.aa_transition < 0 or self.aa_cutoff < 0 or self.aa_cutoff > 0.5:
            raise ValueError("aa_cutoff must be in [0,0.5] and aa_transition >= 0")
        if self.aa_cutoff + self.aa_transition > 0.5:
            # clamp internally to avoid weirdness
            self.aa_transition = max(0.0, 0.5 - self.aa_cutoff)

        # Pre-build (lazy) buffers on first forward when device/dtype are known
        self.register_buffer("_aa_time_win", None, persistent=False)
        self.register_buffer("_aa_freq_mask", None, persistent=False)
        self.register_buffer("_aa_freqs", None, persistent=False)
        # Track device/dtype for lazy rebuild
        self._aa_last_device = None
        self._aa_last_dtype = None

    def _maybe_build_windows(self, device, dtype):
        # Rebuild if device/dtype changed or not built yet
        if (self._aa_time_win is not None and
            self._aa_last_device == device and
            self._aa_last_dtype == dtype):
            return

        k = self.kernel_size[0]

        # Time window
        if self.aa_time_window == 'hann':
            w = _hann(k, device, dtype)
        elif self.aa_time_window == 'kaiser':
            w = _kaiser(k, self.aa_kaiser_beta, device, dtype)
        elif self.aa_time_window in ('none', 'off', 'disable'):
            w = torch.ones(k, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown aa_time_window: {self.aa_time_window}")

        # Frequency mask
        Nfft = int(1 << (int(math.ceil(math.log2(max(1, k*self.aa_fft_pad))))))
        freqs = torch.fft.rfftfreq(Nfft, d=1.0)  # cycles/sample, Nyquist=0.5
        mask = _raised_cosine_lowpass(freqs.to(device=device, dtype=dtype),
                                      cutoff=self.aa_cutoff,
                                      transition=self.aa_transition)

        self._aa_time_win = w  # [k]
        self._aa_freqs = freqs.to(device=device, dtype=dtype)            # [Nfft//2+1]
        self._aa_freq_mask = mask                                       # [Nfft//2+1]
        # Fix: Actually track device and dtype (not dummy tensors)
        self._aa_last_device = device
        self._aa_last_dtype = dtype

    def _window_weight(self):
        """
        Apply time window and frequency-domain lowpass to self.weight.
        Returns weight_eff with same shape as self.weight.
        """
        if not self.aa_enable:
            return self.weight

        device = self.weight.device
        dtype = self.weight.dtype

        # Shapes:
        # weight: [out_ch, in_ch/groups, k]
        k = self.kernel_size[0]
        w = self.weight

        # If neither time nor freq windowing is enabled, return original weight
        if not self.aa_enable_time and not self.aa_enable_freq:
            return w

        # Build windows if needed
        self._maybe_build_windows(device, dtype)
        w_time = self._aa_time_win  # [k]

        # Time-domain windowing (if enabled)
        if self.aa_enable_time:
            w = w * w_time.view(1, 1, k)

        # Frequency-domain filtering (if enabled)
        if self.aa_enable_freq:
            mask = self._aa_freq_mask   # [Nfft//2+1]
            Nfft = (mask.numel() - 1) * 2

            # FFT along kernel axis
            Wf = torch.fft.rfft(w, n=Nfft, dim=2)  # [out, in, Nfft//2+1]

            # Apply smooth low-pass mask in frequency domain
            Wf_lp = Wf * mask.view(1, 1, -1)

            # Back to time
            w = torch.fft.irfft(Wf_lp, n=Nfft, dim=2)[..., :k]  # trim to original length

            # (Optional) a second gentle time-window to reduce edge ripple post-ifft
            if self.aa_enable_time and self.aa_double_window:
                w = w * w_time.view(1, 1, k)

        return w

    def forward(self, input):
        # identical to nn.Conv1d.forward but with windowed weight
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self._window_weight(), self.bias, self.stride,
                            (0,), self.dilation, self.groups)
        return F.conv1d(input, self._window_weight(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvBase(nn.Module):
    """Base class for convolution modules with weight plotting."""
    def __init__(self):
        super().__init__()

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

        if weights.ndim not in [4, 5]:
            # print(f"Weight plotting not implemented for ndim {weights.ndim}.")
            return None, None

        is_3d = weights.ndim == 5

        if is_3d:
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

        else:
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

        return fig, axs

class StandardConv(ConvBase):
    """Standard N-Dimensional Convolution Wrapper (2D or 3D)."""
    def __init__(self, dim: int, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'replicate'):
        super().__init__()
        if dim not in [2, 3]: raise ValueError(f"Unsupported dim for StandardConv: {dim}.")
        ConvNd = nn.Conv2d if dim == 2 else nn.Conv3d
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
        

class StackedConv2d(ConvBase):
    """Approximates a large 2D kernel with stacked smaller kernels (2D only)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 sub_kernel_size: Union[int, Tuple[int, int]] = 3, stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0, padding_mode: str = 'replicate',
                 groups: int = 1, bias: bool = False, dropout_percentage: float = 0.0):
        super().__init__()
        _k_eff = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        _k_sub = sub_kernel_size[0] if isinstance(sub_kernel_size, tuple) else sub_kernel_size
        if _k_sub <= 1: raise ValueError("sub_kernel_size must be > 1")
        self.num_layers = max(1, (_k_eff - _k_sub) // (_k_sub - 1) + 1)
        self.kernel_size_eff = ((self.num_layers - 1) * (_k_sub - 1) + _k_sub,) * 2
        self.sub_kernel_size = (_k_sub,) * 2
        self.in_channels, self.out_channels, self._groups = in_channels, out_channels, groups

        convs = []
        current_in = in_channels
        for i in range(self.num_layers):
            is_last = (i == self.num_layers - 1)
            convs.append(nn.Conv2d(current_in, out_channels, self.sub_kernel_size,
                                   stride=stride if i == 0 else 1, padding=padding,
                                   padding_mode=padding_mode, groups=groups, bias=bias if is_last else False))
            current_in = out_channels
            if dropout_percentage > 0 and not is_last: convs.append(nn.Dropout2d(dropout_percentage))
        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.convs(x)
    @property
    def weight(self) -> torch.Tensor: # Approximate effective kernel
        device = next(self.parameters()).device
        H_eff, W_eff = self.kernel_size_eff
        # Simplified impulse and weight calculation for brevity, may need refinement for groups
        if self._groups != 1:
            # print("Warning: Effective weight for grouped StackedConv2d is highly approximate.")
            return self.convs[0].weight # Fallback for grouped
        impulse = torch.zeros(self.in_channels, self.in_channels, H_eff, W_eff, device=device)
        for i in range(self.in_channels): impulse[i, i, H_eff//2, W_eff//2] = 1.0
        with torch.no_grad(): effective_weight = self.convs(impulse)
        return effective_weight.permute(1, 0, 2, 3)

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

class WindowedConv2d(ConvBase):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]],
                    stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int]] = 0,
                    dilation: Union[int, Sequence[int]] = 1, groups: int = 1, bias: bool = True,
                    padding_mode: str = 'zeros', window_type: str = 'hann', window_kwargs: dict = {}):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        _ks = (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        if len(_ks) != 2: raise ValueError("WindowedConv2d only supports 2D kernels.")
        win_h = _get_window_fn(window_type=window_type, window_size=_ks[0], **window_kwargs) # type: ignore
        win_w = _get_window_fn(window_type=window_type, window_size=_ks[1], **window_kwargs) # type: ignore
        window = torch.outer(torch.as_tensor(win_h), torch.as_tensor(win_w)).float()
        self.register_buffer('window', (window / window.max()).unsqueeze(0).unsqueeze(0))
        self._window_type = window_type
    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight * self.window.to(self.conv.weight.device, self.conv.weight.dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv._conv_forward(x, self.weight, self.conv.bias)

class SeparableWindowedConv2D(ConvBase):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]],
                    stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int]] = 0,
                    dilation: Union[int, Sequence[int]] = 1, bias: bool = True,
                    window_type: str = 'hann', window_kwargs: dict = {}):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.depthwise = WindowedConv2d(out_channels, out_channels, kernel_size, stride=stride, # type: ignore
                                    padding=padding, dilation=dilation, groups=out_channels,
                                    bias=bias, window_type=window_type, window_kwargs=window_kwargs)
        self._window_type = window_type # For plot_weights
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.depthwise(self.pointwise(x))
    @property
    def weight(self) -> torch.Tensor: raise NotImplementedError("Use plot_weights() for SeparableWindowedConv2D.")
    def plot_weights(self, scale_globally=True) -> Tuple[Optional[plt.Figure], Optional[np.ndarray]]:
        """
        Plot separable convolution weights using torchvision.utils.make_grid.

        Args:
            scale_globally (bool): If True, normalize all kernels with one global min/max.
                                  If False, scale each kernel independently.

        Returns:
            Tuple of (figure, axes) or (None, None) if plotting is not possible
        """
        try:
            import torchvision.utils as vutils
            pw_weights = self.pointwise.weight.detach().cpu()
            dw_weights = self.depthwise.weight.detach().cpu()  # Effective weight with window
        except (NotImplementedError, AttributeError, ImportError):
            return None, None  # Simplified error handling

        # Create figure with two subplots: one for pointwise, one for depthwise
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
        ax_pw, ax_dw = axs[0, 0], axs[0, 1]

        # 1. Pointwise weights visualization
        c_out, c_in = pw_weights.shape[:2]

        # For pointwise, just show as a 2D heatmap
        ax_pw.imshow(pw_weights.squeeze().numpy(), cmap='coolwarm', aspect='auto', interpolation='nearest')
        ax_pw.set_title(f'Pointwise Weights ({c_out}×{c_in})', fontsize=10)
        ax_pw.set_xlabel('Input Channels')
        ax_pw.set_ylabel('Output Channels')

        # 2. Depthwise weights visualization using make_grid
        # Reshape to [out_channels, 1, height, width] for make_grid
        reshaped_weights = dw_weights[:, 0].unsqueeze(1)  # Depthwise has 1 input channel per output

        # Create grid with sqrt(out_channels) kernels per row
        nrow = int(np.ceil(np.sqrt(c_out)))

        # Global min/max for consistent scaling if requested
        global_min = dw_weights.min().item() if scale_globally else None
        global_max = dw_weights.max().item() if scale_globally else None

        grid = vutils.make_grid(
            reshaped_weights,
            nrow=nrow,
            normalize=True,
            scale_each=not scale_globally,
            padding=2,
            pad_value=0.5,  # Light gray padding
            value_range=(global_min, global_max) if scale_globally else None
        )

        # Display the grid
        ax_dw.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
        ax_dw.axis('off')
        ax_dw.set_title(f'Depthwise Kernels (Window: {self._window_type})', fontsize=10)

        fig.suptitle(f'Separable Windowed Conv2D: {c_out} Output Channels', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        return fig, axs

def visualize_3d_weights_general_concise(weights_tensor, scale_globally=False):
    """
    Visualizes 3D conv weights [N, Cin, D, H, W] concisely, handling multiple input channels.
    Each row in the output will correspond to an (output_filter, input_channel) pair,
    showing its D temporal slices.

    Args:
        weights_tensor (torch.Tensor): Tensor of shape [N, Cin, D, H, W]
        scale_globally (bool): If True, normalize all kernels with one global min/max.
                               If False, scale each kernel independently.

    Returns:
        plt.Figure: The matplotlib figure object
    """
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure weights_tensor is a torch tensor
    if not isinstance(weights_tensor, torch.Tensor):
        weights_tensor = torch.tensor(weights_tensor)

    # Get tensor dimensions
    num_filters, in_channels, depth, height, width = weights_tensor.shape

    # Reshape to treat each spatial kernel (across N, Cin, D) as a separate image for the grid:
    # Target shape for make_grid: (num_total_kernels, 1, H, W)
    # num_total_kernels = num_filters * in_channels * depth
    reshaped_weights = weights_tensor.reshape(num_filters * in_channels * depth, 1, height, width)

    # Create a grid.
    # nrow = depth: will show the D temporal slices for each (N_idx, Cin_idx) pair in a row.
    # This means the grid will have (num_filters * in_channels) rows.
    # Get global min/max for consistent scaling if requested
    global_min = weights_tensor.min().item() if scale_globally else None
    global_max = weights_tensor.max().item() if scale_globally else None

    grid = vutils.make_grid(
        reshaped_weights,
        nrow=depth,
        normalize=True,
        scale_each=not scale_globally, # if scale_globally is True, scale_each becomes False
        padding=1,
        value_range=(global_min, global_max) if scale_globally else None
    )

    # Display the grid
    # Adjust figsize: width is prop to depth, height is prop to (num_filters * in_channels)
    fig_width = depth * 1.2  # Proportional to number of temporal slices in a row
    fig_height = (num_filters * in_channels) * 1.2 # Proportional to number of (filter, in_channel) rows
    # Cap max figure dimensions to avoid excessively large plots
    max_dim = 20
    fig_width = min(fig_width, max_dim)
    fig_height = min(fig_height, max_dim * (num_filters * in_channels / depth if depth > 0 else 1))

    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    title = (f'{num_filters} Out-Filters, {in_channels} In-Channels, {depth} Lags/Kernel\n'
             f'Scaling: {"Global" if scale_globally else "Individual Kernels"}')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle

    return fig


CONV_LAYER_MAP = {
    'standard': StandardConv,
    'stacked2d': StackedConv2d,
    'windowed2d': WindowedConv2d,
    'depthwise': DepthwiseConv,
    'separable_windowed2d': SeparableWindowedConv2D,
}