"""
Data transformation utilities for neural data analysis.

This module provides a registry-based system for data transformations that can be
applied to neural datasets. Transformations are registered by name and can be
chained together in pipelines.

The module supports:
- Registry-based transform functions
- Pipeline composition of multiple transforms
- Configurable transforms with parameters
"""

import torch
import torch.nn.functional as F
import math
from typing import Callable, Dict, List, Any



# ──────────────────────────────────────────────────────────────────────────────
# Helpers
class SpatiotemporalWhitening(object):
    """
    Spatiotemporal whitening filter for video data (T x H x W).
    The filter is defined in the 3D frequency domain.
    R(f) = f_st * exp(-(f_st / f_0)^n)
    where f_st is the radial spatiotemporal frequency.
    """
    def __init__(self, data_shape, device, f_0=0.4, n=4):
        """
        Args:
            data_shape (tuple): The shape of the data (T, H, W).
            device: The torch device to use.
            f_0 (float): Cutoff frequency.
            n (int): Steepness of the filter roll-off.
        """
        self.f_0 = f_0
        self.n = n
        dim_t, dim_h, dim_w = data_shape

        # Create 3D frequency grid
        f_t = torch.fft.fftfreq(dim_t)
        f_h = torch.fft.fftfreq(dim_h)
        f_w = torch.fft.fftfreq(dim_w)
        
        # Use meshgrid to create 3D coordinates
        ft_grid, fh_grid, fw_grid = torch.meshgrid(f_t, f_h, f_w, indexing='ij')

        # Calculate radial spatiotemporal frequency
        # f_st = sqrt(f_t^2 + f_h^2 + f_w^2)
        f_st = torch.sqrt(ft_grid**2 + fh_grid**2 + fw_grid**2)
        
        # Build the filter
        self.filter = f_st * torch.exp(-(f_st / self.f_0).pow(self.n))
        
        # Reshape for broadcasting and move to device
        # Shape becomes (T, 1, H, W) to match input
        self.filter = self.filter.unsqueeze(1).to(device=device)


    def __call__(self, video_data):
        """
        Apply the spatiotemporal whitening filter.
        Args:
            video_data (torch.Tensor): Input video of shape (T, H, W) or (T, 1, H, W).
        Returns:
            torch.Tensor: Whitened video of the same shape.
        """
        original_shape = video_data.shape

        # Handle both 3D (T, H, W) and 4D (T, 1, H, W) inputs
        if video_data.dim() == 3:
            # Add channel dimension: (T, H, W) -> (T, 1, H, W)
            video_data = video_data.unsqueeze(1)
        elif video_data.dim() == 4 and video_data.shape[1] == 1:
            # Already correct shape
            pass
        else:
            raise ValueError("Input must be of shape (T, H, W) or (T, 1, H, W)")

        # FFT expects complex input or real input without channel dim
        # We apply the 3D FFT over dims 0, 2, and 3
        video_f = torch.fft.fftn(video_data, dim=(0, 2, 3))

        # Apply the filter via element-wise multiplication
        video_f_filtered = video_f * self.filter

        # Inverse FFT to go back to the spatiotemporal domain
        whitened_video = torch.fft.ifftn(video_f_filtered, dim=(0, 2, 3)).real

        # Return in original shape
        if len(original_shape) == 3:
            whitened_video = whitened_video.squeeze(1)

        return whitened_video
    
# ──────────────────────────────────────────────────────────────────────────────
# 1.  Transform registry
# ──────────────────────────────────────────────────────────────────────────────
class TransformFn(Callable[[torch.Tensor], torch.Tensor]): ...
TRANSFORM_REGISTRY: Dict[str, Callable[[Dict[str, Any]], TransformFn]] = {}

def _register(name):
    def wrap(fn):
        TRANSFORM_REGISTRY[name] = fn
        return fn
    return wrap

@_register("fftwhitening")
def _make_fftwhitening(cfg, dataset_config=None):
    """
    Create a spatiotemporal whitening transform.

    Args:
        cfg: Dict with optional parameters:
            - f_0 (float): Cutoff frequency (default: 0.4)
            - n (int): Steepness of filter roll-off (default: 4)

    Example:
        fftwhitening: {}  # Use defaults
        fftwhitening: {f_0: 0.3, n: 6}  # Custom parameters
    """
    f_0 = cfg.get("f_0", 0.4)
    n = cfg.get("n", 4)

    def fftwhitening(x: torch.Tensor):
        # Determine data shape for filter creation
        if x.dim() == 3:
            data_shape = x.shape  # (T, H, W)
        elif x.dim() == 4 and x.shape[1] == 1:
            data_shape = (x.shape[0], x.shape[2], x.shape[3])  # (T, H, W)
        else:
            raise ValueError("Input must be of shape (T, H, W) or (T, 1, H, W)")

        # Create whitening filter
        whitening_filter = SpatiotemporalWhitening(
            data_shape=data_shape,
            device=x.device,
            f_0=f_0,
            n=n
        )

        return whitening_filter(x)

    return fftwhitening

@_register("pixelnorm")
def _make_pixelnorm(cfg, dataset_config=None):
    def pixelnorm(x: torch.Tensor):
        return (x.float() - 127) / 255
    return pixelnorm

@_register("diff")
def _make_diff(cfg, dataset_config=None):
    axis = cfg.get("axis", 0)
    def diff(x: torch.Tensor):
        # prepend first slice to keep length constant
        prepend = x.index_select(axis, torch.tensor([0], device=x.device))
        return torch.diff(x, dim=axis, prepend=prepend)
    return diff

@_register("mul")
def _make_mul(cfg, dataset_config=None):
    factor = cfg if isinstance(cfg, (int,float)) else cfg.get("factor", 1.0)
    def mul(x): return x * factor
    return mul

@_register("temporal_basis")
def _make_basis(cfg, dataset_config=None):
    from DataYatesV1.models.modules import TemporalBasis
    basis = TemporalBasis(**cfg)
    def tb(x):                         # x (T, …)  or (B,T, …)
        # TemporalBasis expects (B,C,T); reshape accordingly
        orig_shape = x.shape

        if x.ndim == 2:                # (T, C) → (1,C,T)
            x = x.transpose(0,1).unsqueeze(0)
        elif x.ndim == 3:              # (T, H, W) → (1, 1, T, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Unsupported tensor rank for temporal_basis")

        y = basis(x)                    # (B,C',T)
        if len(orig_shape) == 2:                # (1,C',T) → (T,C')
            y = y.permute(0,2,1).squeeze(0) # back to (T,C')
        elif len(orig_shape) == 3:  
            # (1, C, T, H, W) → (T, C, H, W)
            y = y.permute(2,0,1,3,4).reshape((orig_shape[0], -1, orig_shape[1], orig_shape[2])) # to (T, Cnew, H, W)

        return y.view(*y.shape)         # torchscript friendliness
    return tb

@_register("splitrelu")
def _make_splitrelu(cfg, dataset_config=None):
    from DataYatesV1.models.modules import SplitRelu
    return SplitRelu(**cfg)

@_register("symlog")
def _make_symlog(cfg, dataset_config=None):
    def symlog(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    return symlog

@_register("maxnorm")
def _make_maxnorm(cfg, dataset_config=None):
    def maxnorm(x):
        return x / torch.max(torch.abs(x))
    return maxnorm

@_register("dacones")
def _make_dacones(cfg, dataset_config=None):
    from DataYatesV1.models.modules import DAModel
    def dacones(x):
        cones = DAModel(**cfg)
        # permute (B,H,W) to (1, B, H, W) and squeeze back to (B,H,W)
        dtype = x.dtype
        y = cones(x.unsqueeze(0).float()).squeeze(0).permute(1,0,2,3)
        if dtype == torch.uint8:
            y /= 2.5 
            y *= 255
            y = y.clamp(0, 255).to(torch.uint8)
        else:
            y = y.to(dtype)
        return y

    return dacones

@_register("unsqueeze")
def _make_unsqueeze(cfg, dataset_config=None):
    axis = cfg if isinstance(cfg, int) else cfg.get("axis", 0)
    def unsqueeze(x):
        return x.unsqueeze(axis)
    return unsqueeze

@_register("cast")
def _make_cast(cfg, dataset_config=None):
    """
    Transform to convert tensor to a different dtype.

    Args:
        cfg: Either a string (dtype name) or dict with 'dtype' key
             Supported dtypes: 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', etc.
    """
    if isinstance(cfg, str):
        dtype_str = cfg
    else:
        dtype_str = cfg.get("dtype", "float32")

    # Map string names to torch dtypes
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': torch.bool,
        'uint8': torch.uint8,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Supported: {list(dtype_map.keys())}")

    target_dtype = dtype_map[dtype_str]

    def to_dtype(x: torch.Tensor):
        return x.to(dtype=target_dtype)

    return to_dtype


@_register("select_units")
def _make_select_units(cfg, dataset_config=None):
    """
    Create a transform that selects a subset of units along the last dimension.

    This is useful for filtering neural data (e.g., robs, history) to a specific
    subset of units (cids) after other transforms have been applied.

    Args:
        cfg: Dict with parameters:
            - indices: Can be either:
                - 'cids': Use the cids from dataset_config
                - list: Explicit list of unit indices to select
        dataset_config: Optional dataset config dict (needed if indices='cids')

    Example:
        select_units: {indices: 'cids'}  # Use cids from dataset config
        select_units: {indices: [0, 1, 2, 5, 10]}  # Explicit indices

    Returns:
        A function that selects the specified units along the last dimension.
    """
    indices = cfg.get("indices")
    if indices is None:
        raise ValueError("select_units requires 'indices' parameter")

    # Handle 'cids' special case
    if indices == 'cids':
        if dataset_config is None:
            raise ValueError("select_units with indices='cids' requires dataset_config to be passed")
        indices = dataset_config.get('cids')
        if indices is None:
            raise ValueError("dataset_config does not contain 'cids'")

    # Convert to tensor for indexing
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    def select_units(x: torch.Tensor) -> torch.Tensor:
        """Select units along the last dimension."""
        return x[..., indices_tensor]

    return select_units


@_register("smooth")
def _make_smooth(cfg, dataset_config=None):
    """
    Transform to apply smoothing along the temporal (0th) dimension using F.conv1d.

    Args:
        cfg: Dict with 'type' and 'params' keys, or just params value
             - type: smoothing type (default: 'gaussian')
             - params: parameters for smoothing (for gaussian: standard deviation)

    Example:
        smooth: {type: 'gaussian', params: 1.5}
        smooth: 1.5  # shorthand for gaussian with sigma=1.5
    """
    # Handle shorthand notation (just the sigma value)
    if isinstance(cfg, (int, float)):
        smooth_type = 'gaussian'
        params = cfg
    else:
        smooth_type = cfg.get('type', 'gaussian')
        params = cfg.get('params', 1.0)

    if smooth_type != 'gaussian':
        raise ValueError(f"Unsupported smoothing type '{smooth_type}'. Only 'gaussian' is implemented.")

    sigma = params

    def _build_gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build a 1D Gaussian kernel for convolution."""
        if sigma <= 0:
            return torch.tensor([1.0], device=device, dtype=dtype)

        # Kernel size: 6*sigma + 1 (ensures we capture ~99.7% of the distribution)
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1

        # Create 1D Gaussian kernel
        center = kernel_size // 2
        x = torch.arange(kernel_size, device=device, dtype=dtype) - center
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize

        return kernel

    def smooth_gaussian(x: torch.Tensor):
        """Apply Gaussian smoothing along the 0th dimension using F.conv1d."""
        if sigma <= 0:
            return x.detach()  # No smoothing, but detach gradients

        original_shape = x.shape
        original_dtype = x.dtype
        device = x.device

        # Build Gaussian kernel
        kernel = _build_gaussian_kernel(sigma, device, torch.float32)
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2

        # Reshape kernel for conv1d: (out_channels, in_channels, kernel_length)
        kernel = kernel.view(1, 1, -1)

        # Handle different tensor dimensions
        if x.ndim == 1:
            # 1D: (T,) -> (1, 1, T) for conv1d
            x_conv = x.float().unsqueeze(0).unsqueeze(0)
            # Use reflect padding to avoid boundary artifacts (like scipy default)
            x_padded = F.pad(x_conv, (padding, padding), mode='reflect')
            smoothed = F.conv1d(x_padded, kernel, padding=0)
            result = smoothed.squeeze(0).squeeze(0)

        elif x.ndim == 2:
            # 2D: (T, C) -> (C, 1, T) for conv1d, then back to (T, C)
            T, C = x.shape
            x_conv = x.float().transpose(0, 1).unsqueeze(1)  # (C, 1, T)
            # Use reflect padding to avoid boundary artifacts (like scipy default)
            x_padded = F.pad(x_conv, (padding, padding), mode='reflect')
            smoothed = F.conv1d(x_padded, kernel, padding=0)  # (C, 1, T)
            result = smoothed.squeeze(1).transpose(0, 1)  # (T, C)

        else:
            # Higher dimensions: flatten non-temporal dims, apply conv1d, reshape back
            T = original_shape[0]
            non_temporal_shape = original_shape[1:]
            non_temporal_size = math.prod(non_temporal_shape)

            # Reshape to (T, flattened_features) then to (flattened_features, 1, T)
            x_flat = x.float().view(T, non_temporal_size)
            x_conv = x_flat.transpose(0, 1).unsqueeze(1)  # (features, 1, T)
            # Use reflect padding to avoid boundary artifacts (like scipy default)
            x_padded = F.pad(x_conv, (padding, padding), mode='reflect')
            smoothed = F.conv1d(x_padded, kernel, padding=0)  # (features, 1, T)
            smoothed_flat = smoothed.squeeze(1).transpose(0, 1)  # (T, features)
            result = smoothed_flat.view(T, *non_temporal_shape)

        # Convert back to original dtype and detach gradients
        return result.to(dtype=original_dtype).detach()

    return smooth_gaussian


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Build a composite transform pipeline
# ──────────────────────────────────────────────────────────────────────────────
def make_pipeline(op_list: List[Dict[str, Any]], dataset_config: Dict[str, Any] = None) -> TransformFn:
    """
    Build a composite transform pipeline from a list of operations.

    Args:
        op_list: List of operation dictionaries
        dataset_config: Optional dataset configuration (for transforms that need access to config like cids)

    Returns:
        A function that applies all transforms in sequence
    """
    fns: List[TransformFn] = []
    for op_dict in op_list:
        name, cfg = next(iter(op_dict.items()))
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"Unknown transform '{name}'")
        # Pass dataset_config to the transform factory if it needs it
        fns.append(TRANSFORM_REGISTRY[name](cfg, dataset_config))

    def pipeline(x):
        for fn in fns:
            x = fn(x)
        return x
    return pipeline

