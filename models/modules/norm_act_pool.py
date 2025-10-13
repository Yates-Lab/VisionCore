# norm_act_pool.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Tuple

from .common import SplitRelu # Assuming common.py is in the same directory level

__all__ = ['get_norm_layer', 'get_activation_layer', 'get_pooling_layer', 'RMSNorm', 'LayerNorm', 'BSoftplus', 'AABlur_SE_SoftPool', 'LocalChannelSpatialAttn']


class LocalChannelSpatialAttn(nn.Module):
    """
    Local channel & spatial attention for (B, C, T, H, W) tensors.

    1) Sliding-window channel attention via Conv1d over channels.
    2) Depthwise spatial attention per-frame via Conv2d.

    Args:
        C (int): Number of input channels (must be divisible by chan_window)
        chan_window (int): Odd integer, size of sliding window over channel axis
        spatial_kernel (int): Odd integer, kernel size for depthwise spatial attention
    """

    def __init__(self, C: int, chan_window: int = 3, spatial_kernel: int = 3):
        super().__init__()
        assert chan_window % 2 == 1, "chan_window must be odd"
        assert C % chan_window == 0, "C must be divisible by chan_window"

        # (1) Channel attention: treat channels as length for Conv1d
        # Groups = C // chan_window ensures non-overlapping blocks of size chan_window
        self.chan_conv1d = nn.Conv1d(
            in_channels=C,
            out_channels=C,
            kernel_size=chan_window,
            padding=chan_window//2,
            groups=C // chan_window,
            bias=True
        )

        # (2) Spatial attention per-frame: depthwise 2D conv
        self.spatial_conv2d = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=spatial_kernel,
            padding=spatial_kernel//2,
            groups=C,  # one filter per channel
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # --- 1) Channel sliding-window attention ---
        # reshape so channel axis is the "length" for Conv1d:
        # (B, C, T, H, W) -> (B*T*H*W, C, 1)
        x_ch = x.permute(0, 2, 3, 4, 1).reshape(-1, C, 1)
        # convolve along channels in sliding windows of size chan_window
        g_ch = self.chan_conv1d(x_ch)            # (B*T*H*W, C, 1)
        # reshape back to (B, C, T, H, W)
        g_ch = g_ch.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).sigmoid()

        # --- 2) Spatial depthwise attention per-frame ---
        # fold batch & time so each frame is independent
        x_sp = x.reshape(-1, C, H, W)            # (B*T, C, H, W)
        g_sp = self.spatial_conv2d(x_sp).sigmoid()  # (B*T, C, H, W)
        # unfold back to (B, C, T, H, W)
        g_sp = g_sp.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        # --- 3) Combine and apply ---
        return x * g_ch * g_sp


class AABlur_SE_SoftPool(nn.Module):
    """
    Anti-aliased stride-2 downsample with optional SoftPool and SE gate.
    Works on (B, C, T, H, W) tensors from a spatiotemporal stem.

    Args:
        C (int): Number of input channels
        stride (int): Stride for pooling (default: 2)
        r (int): Reduction ratio for SE block (default: 8)
        use_soft (bool): Whether to use SoftPool instead of AvgPool (default: True)
    """
    def __init__(self, C: int, stride: int = 2, r: int = 8, use_soft: bool = True):
        super().__init__()
        # fixed blur kernel (depth-wise, so no extra params)
        k = torch.tensor([1., 2., 1.])
        blur = (k[:, None] * k[None, :]).div(16)          # 3×3 low-pass
        self.register_buffer("blur", blur[None, None])
        self.stride = stride
        self.use_soft = use_soft

        self.se = nn.Sequential(                       # squeeze-and-excite
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(C, C // r, 1, bias=True),
            nn.SiLU(),
            nn.Conv3d(C // r, C, 1, bias=True),
            nn.Sigmoid()
        )

    def _softpool2d(self, x: torch.Tensor, k: int = 2, s: int = 2) -> torch.Tensor:
        """SoftPool implementation for 2D tensors."""
        w = torch.exp(x)
        return F.avg_pool2d(x * w, k, s) / (F.avg_pool2d(w, k, s) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (B, C, T, H, W)"""
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t * c, 1, h, w)
        x = F.conv2d(x, self.blur, padding=1, groups=1)   # anti-alias
        if self.use_soft:
            x = self._softpool2d(x, 2, self.stride)
        else:
            x = F.avg_pool2d(x, 2, self.stride)
        h2, w2 = x.shape[-2:]
        x = x.view(b, t, c, h2, w2).permute(0, 2, 1, 3, 4)

        # channel-wise gating (SE)
        return x * self.se(x)


class BSoftplus(nn.Module):
    """
    Learnable beta parameter softplus activation: (1/beta) * softplus(beta * x)

    Implements: log(1 + exp(beta * x)) / beta

    Args:
        beta_init (float): Initial value for beta parameter. Default: 1.0
        beta_min (float): Minimum value for beta (prevents vanishing). Default: 0.1
        beta_max (float): Maximum value for beta (prevents explosion). Default: 10.0
    """

    def __init__(self, beta_init: float = 5.0, beta_min: float = 0.1, beta_max: float = 10.0):
        super().__init__()
        # Use log(beta) as the actual parameter to ensure beta > 0
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta_init)))
        self.beta_min = beta_min
        self.beta_max = beta_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp log_beta to prevent beta from vanishing or exploding
        beta = torch.exp(self.log_beta).clamp(min=self.beta_min, max=self.beta_max)

        # Compute (1/beta) * softplus(beta * x) = log(1 + exp(beta * x)) / beta
        return torch.nn.functional.softplus(beta * x) / beta

    def extra_repr(self) -> str:
        beta = torch.exp(torch.clamp(self.log_beta,
                                     min=torch.log(torch.tensor(self.beta_min, dtype=self.log_beta.dtype)),
                                     max=torch.log(torch.tensor(self.beta_max, dtype=self.log_beta.dtype))))
        return f"beta={beta.item():.4f}, beta_min={self.beta_min}, beta_max={self.beta_max}"


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, norm_dims: tuple = (1,), eps: float = 1e-4, affine: bool = True):
        """
        Root Mean Square Normalization with optional learnable affine parameters.

        Args:
            num_features (int): Number of features (channels) for affine parameters.
                               Only used if affine=True.
            norm_dims (tuple): A tuple of dimension indices over which the mean
                               of squares will be computed. These are the
                               dimensions that get normalized.
                               E.g., `(-1,)` for the last dimension.
                               Defaults to the channel dimension: (1,)
            eps (float): A small value added to the denominator for
                         numerical stability.
            affine (bool): If True, adds learnable scale (gamma) and shift (beta) parameters.
                          Default: True (recommended for better performance).
        """
        super().__init__()
        if not isinstance(norm_dims, tuple):
            raise TypeError("norm_dims must be a tuple of dimension indices.")
        self.norm_dims = norm_dims
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if affine:
            # Learnable scale and shift parameters
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure computation is in FP32 for numerical stability
        input_dtype = x.dtype
        x_fp32 = x.float()

        mean_of_squares = torch.mean(x_fp32.pow(2), dim=self.norm_dims, keepdim=True)
        normalized_x = x_fp32 * torch.rsqrt(mean_of_squares + self.eps)

        # Apply affine transformation if enabled
        if self.affine:
            # Reshape gamma and beta to broadcast correctly
            # For (B, C, ...) tensors, we want (1, C, 1, 1, ...) shape
            shape = [1] * x.ndim
            shape[1] = self.num_features  # Channel dimension
            gamma = self.gamma.view(*shape).float()
            beta = self.beta.view(*shape).float()
            normalized_x = normalized_x * gamma + beta

        # Convert back to input dtype
        return normalized_x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, norm_dims={self.norm_dims}, eps={self.eps}, affine={self.affine}"
    
class LayerNorm(nn.Module):
    def __init__(self, norm_dims: tuple = (1,), eps: float = 1e-4):
        """
        Custom Layer Norm (no learnable parameters).

        Args:
            norm_dims (tuple): A tuple of dimension indices over which the mean
                               and variance will be computed. These are the
                               dimensions that get normalized.
                               E.g., `(-1,)` for the last dimension.
                               Defaults to the channel dimension: (1,)
            eps (float): A small value added to the denominator for
                         numerical stability.
        """
        super().__init__()
        if not isinstance(norm_dims, tuple):
            raise TypeError("norm_dims must be a tuple of dimension indices.")
        self.norm_dims = norm_dims
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure computation is in FP32 for numerical stability
        input_dtype = x.dtype
        x_fp32 = x.float()

        # Calculate mean and variance over the specified self.norm_dims
        mean = torch.mean(x_fp32, dim=self.norm_dims, keepdim=True)
        variance = torch.mean(x_fp32.pow(2), dim=self.norm_dims, keepdim=True) - mean.pow(2)
        normalized_x = (x_fp32 - mean) * torch.rsqrt(variance + self.eps)

        # Convert back to input dtype
        return normalized_x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"norm_dims={self.norm_dims}, eps={self.eps}"
    
class SafeGRN(nn.Module):
    """Global‑Response‑Norm variant with a *learnable* additive offset.

    Standard GRN divides by the per‑sample RMS of the feature map. If the RMS is
    extremely small (sparse input) the scale factor explodes. We fix this by
    adding a small positive constant **offset** *after* the square‑root.  The
    denominator becomes

        denom = sqrt(mean(y**2) + eps) + offset

    which guarantees that the gain is bounded by ``1/offset`` regardless of
    sparsity.

    Args:
        channels (int): #feature channels.
        eps (float): numerical epsilon *inside* the square‑root.
        offset (float): additive safety term.  0.2–0.5 is robust for Poisson-
            scale data.  Make it learnable by setting ``learnable_offset=True``.
        dim (int): 3 ⇒ input has shape (B,C,T,H,W); 2 ⇒ (B,C,H,W).
        learnable_offset (bool): if True the offset is a learnable parameter.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        offset: float = 0.5,
        dim: int = 3,
        learnable_offset: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

        if learnable_offset:
            self.offset = nn.Parameter(torch.full((1,), offset))
        else:
            self.register_buffer("offset", torch.tensor(offset))

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        reduce_dims = (2, 3, 4) if self.dim == 3 else (2, 3)
        rms = torch.sqrt(x.pow(2).mean(dim=reduce_dims, keepdim=True) + self.eps)
        denom = rms + self.offset.clamp(min=.1, max=1.0)
        x_hat = x / denom
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return (
            self.gamma.view(*shape) * x_hat + self.beta.view(*shape)
        )

    
class GlobalResponseNorm(nn.Module):
    def __init__(self, C, eps=1e-4, gamma_init=0.1, clamp_ratio=50.):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, C, 1, 1, 1), gamma_init))
        self.beta  = nn.Parameter(torch.zeros(1, C, 1, 1, 1))
        self.eps   = eps
        self.clamp = clamp_ratio          # ≥ 1  (set None to disable)

    def forward(self, x):
        # Ensure computation is in FP32 for numerical stability
        input_dtype = x.dtype
        x_fp32 = x.float()
        gamma_fp32 = self.gamma.float()
        beta_fp32 = self.beta.float()

        g   = torch.norm(x_fp32, p=2, dim=1, keepdim=True)
        mu  = g.mean((2, 3, 4), keepdim=True)
        r   = g / (mu + self.eps)
        if self.clamp is not None:                       # guard FP16 range
            r = torch.clamp(r, max=self.clamp)
        result = x_fp32 + gamma_fp32 * (x_fp32 * r) + beta_fp32

        # Convert back to input dtype
        return result.to(input_dtype)

def get_norm_layer(norm_type: Optional[str],
                   num_features: int,
                   # `dim` here refers to the type of operation (1D, 2D, 3D for BatchNorm/InstanceNorm)
                   # For LayerNorm/RMSNorm, 'op_dim' is the targeted dimension(s) int or Tuple[int]
                   op_dim: Union[int, Tuple[int, ...]], # 1, 2, or 3 for BatchNorm/InstanceNorm type
                   norm_params: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Factory for normalization layers.
    Can be:
    - 'batch', 'instance', 'layer', 'rms', 'group', or 'none'
    """
    if norm_type is None or norm_type.lower() == 'none': return nn.Identity()
    if norm_params is None: norm_params = {}

    norm_type_lower = norm_type.lower()
    eps = norm_params.get('eps', 1e-5)
    target_dim = norm_params.get('target_dim', 1) # Default to channel dimension
    affine = norm_params.get('affine', True) # Default affine=True for most, False for InstanceNorm by default

    if norm_type_lower == 'batch':
        if op_dim == 1: return nn.BatchNorm1d(num_features, eps=eps, affine=affine)
        if op_dim == 2: return nn.BatchNorm2d(num_features, eps=eps, affine=affine)
        if op_dim == 3: return nn.BatchNorm3d(num_features, eps=eps, affine=affine)
        raise ValueError(f"Unsupported op_dim {op_dim} for BatchNorm.")

    elif norm_type_lower == 'instance':
        # PyTorch InstanceNorm default affine is False
        instance_affine = norm_params.get('affine', False)
        if op_dim == 1: return nn.InstanceNorm1d(num_features, eps=eps, affine=instance_affine)
        if op_dim == 2: return nn.InstanceNorm2d(num_features, eps=eps, affine=instance_affine)
        if op_dim == 3: return nn.InstanceNorm3d(num_features, eps=eps, affine=instance_affine)
        raise ValueError(f"Unsupported op_dim {op_dim} for InstanceNorm.")

    elif norm_type_lower == 'layer':
        if isinstance(target_dim, int): target_dim = (target_dim,)
        return LayerNorm(target_dim, eps=eps) # custom Layer norm with no learned parameters

    elif norm_type_lower == 'rms':
        if isinstance(target_dim, int): target_dim = (target_dim,)
        return RMSNorm(num_features, target_dim, eps=eps, affine=affine) # custom RMS norm with optional affine parameters

    elif norm_type_lower == 'group':
        num_groups = norm_params.get('num_groups', 1) # Default to 1 (LayerNorm-like over spatial)
        if num_features > 0 and num_features % num_groups != 0:
             # Fallback if not divisible, or user should ensure it is.
             print(f"Warning: GroupNorm num_features ({num_features}) not divisible by num_groups ({num_groups}). Using num_groups=1.")
             num_groups = 1
        elif num_features == 0 and num_groups > 1 : # Edge case
             num_groups = 1

        return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)
    
    elif norm_type_lower == 'grn':
        return SafeGRN(num_features, learnable_offset=True)
    
    else:
        raise ValueError(f"Unknown normalization type: '{norm_type}'.")


def get_activation_layer(act_type: Optional[str],
                         act_params: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Factory for activation layers.
    Can be:
    - 'relu', 'leakyrelu', 'gelu', 'sigmoid', 'tanh', 'silu', 'swish', 'mish', 'splitrelu'
    - 'softplus', 'bsoftplus' (learnable beta parameter softplus)
    - Any other activation in torch.nn can be specified by name
    - If None or 'none', returns nn.Identity()

    For 'bsoftplus', act_params can include:
    - beta_init (float): Initial beta value (default: 1.0)
    - beta_min (float): Minimum beta value (default: 0.1)
    - beta_max (float): Maximum beta value (default: 10.0)
    """
    
    if act_type is None or act_type.lower() == 'none': return nn.Identity()
    if act_params is None: act_params = {}

    act_type_lower = act_type.lower()
    inplace = act_params.get('inplace', False) # Common param

    if act_type_lower == 'relu': return nn.ReLU(inplace=inplace)
    if act_type_lower == 'leakyrelu': return nn.LeakyReLU(act_params.get('negative_slope', 0.01), inplace=inplace)
    if act_type_lower == 'gelu': return nn.GELU(approximate=act_params.get('approximate', 'none'))
    if act_type_lower == 'sigmoid': return nn.Sigmoid()
    if act_type_lower == 'tanh': return nn.Tanh()
    if act_type_lower in ['silu', 'swish']: return nn.SiLU(inplace=inplace)
    if act_type_lower == 'mish': return nn.Mish(inplace=inplace) # Assumes PyTorch >= 2.0
    if act_type_lower == 'softplus': return nn.Softplus()
    if act_type_lower == 'bsoftplus':
        return BSoftplus(beta_init=act_params.get('beta_init', 1.0),
                        beta_min=act_params.get('beta_min', 0.1),
                        beta_max=act_params.get('beta_max', 10.0))
    if act_type_lower == 'none': return nn.Identity()
    if act_type_lower == 'identity': return nn.Identity()
    if act_type_lower == 'square': return lambda x: x**2
    if act_type_lower == 'splitrelu':
        return SplitRelu(split_dim=act_params.get('split_dim', 1),
                         trainable_gain=act_params.get('trainable_gain', False))

    # Attempt to get from torch.nn for other activations
    try:
        ActivationClass = getattr(nn, act_type) # Case-sensitive
        # Simple instantiation, pass inplace if it's a known arg for that class
        # More complex arg passing would require inspect, but keep it simple
        if 'inplace' in ActivationClass.__init__.__code__.co_varnames:
            return ActivationClass(inplace=inplace)
        return ActivationClass()
    except AttributeError:
        raise ValueError(f"Unknown activation type: '{act_type}'.")


def get_pooling_layer(pool_params: Optional[Dict[str, Any]] = None, op_dim: int = 2) -> nn.Module:
    """Factory for pooling layers.
    Can be:
    - 'max', 'avg', 'adaptivemax', 'adaptiveavg', 'learned', 'aablur', 'locatt', or 'none'

    For 'learned' pooling, pool_params should include:
    - 'in_channels': Number of input channels for the ConvBlock
    - 'out_channels': Number of output channels for the ConvBlock (optional, defaults to in_channels)
    - 'conv_params': Parameters for the convolution (kernel_size, stride, etc.)
    - 'norm_type': Normalization type for the ConvBlock (optional)
    - 'act_type': Activation type for the ConvBlock (optional)
    - Other ConvBlock parameters as needed

    For 'aablur' pooling (3D only), pool_params should include:
    - 'channels': Number of input channels (required)
    - 'stride': Stride for pooling (optional, default: 2)
    - 'r': Reduction ratio for SE block (optional, default: 8)
    - 'use_soft': Whether to use SoftPool instead of AvgPool (optional, default: True)

    For 'locatt' pooling (3D only), pool_params should include:
    - 'channels': Number of input channels (required, must be divisible by chan_window)
    - 'chan_window': Sliding window size over channel axis (optional, default: 3)
    - 'spatial_kernel': Kernel size for spatial attention (optional, default: 3)
    """
    if pool_params is None or pool_params.get('type', 'none').lower() == 'none':
        return nn.Identity()

    pool_type = pool_params['type'].lower()

    if pool_type == 'aablur':
        # AABlur pooling only works with 3D tensors
        if op_dim != 3:
            raise ValueError("AABlur pooling only supports 3D tensors (op_dim=3)")

        # Extract required parameters
        channels = pool_params.get('channels')
        if channels is None:
            raise ValueError("'aablur' pooling requires 'channels' parameter")

        stride = pool_params.get('stride', 2)
        r = pool_params.get('r', 8)
        use_soft = pool_params.get('use_soft', True)

        return AABlur_SE_SoftPool(C=channels, stride=stride, r=r, use_soft=use_soft)

    elif pool_type == 'locatt':
        # LocalChannelSpatialAttn only works with 3D tensors
        if op_dim != 3:
            raise ValueError("LocalChannelSpatialAttn only supports 3D tensors (op_dim=3)")

        # Extract required parameters
        channels = pool_params.get('channels')
        if channels is None:
            raise ValueError("'locatt' pooling requires 'channels' parameter")

        chan_window = pool_params.get('chan_window', 3)
        spatial_kernel = pool_params.get('spatial_kernel', 3)

        return LocalChannelSpatialAttn(C=channels, chan_window=chan_window, spatial_kernel=spatial_kernel)

    elif pool_type == 'learned':
        # Import ConvBlock here to avoid circular imports
        from .conv_blocks import ConvBlock

        # Extract required parameters
        in_channels = pool_params.get('in_channels')
        if in_channels is None:
            raise ValueError("'learned' pooling requires 'in_channels' parameter")

        out_channels = pool_params.get('out_channels', in_channels)
        conv_params = pool_params.get('conv_params', {})

        # Set default conv_params if not provided
        if 'kernel_size' not in conv_params:
            conv_params['kernel_size'] = pool_params.get('kernel_size', 3)
        if 'stride' not in conv_params:
            conv_params['stride'] = pool_params.get('stride', 2)
        if 'padding' not in conv_params:
            conv_params['padding'] = pool_params.get('padding', 1)

        # Extract other ConvBlock parameters
        norm_type = pool_params.get('norm_type', 'batch')
        norm_params = pool_params.get('norm_params', None)
        act_type = pool_params.get('act_type', 'relu')
        act_params = pool_params.get('act_params', None)
        dropout = pool_params.get('dropout', 0.0)
        use_weight_norm = pool_params.get('use_weight_norm', False)
        causal = pool_params.get('causal', True)
        order = pool_params.get('order', ('conv', 'norm', 'act', 'dropout', 'pool'))

        return ConvBlock(
            dim=op_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            conv_params=conv_params,
            norm_type=norm_type,
            norm_params=norm_params,
            act_type=act_type,
            act_params=act_params,
            pool_params=None,  # No nested pooling
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            causal=causal,
            order=order
        )

    # Handle traditional pooling types
    kernel_size = pool_params['kernel_size']
    stride = pool_params.get('stride', kernel_size) # Default stride to kernel_size
    padding = pool_params.get('padding', 0)

    PoolNd = getattr(nn, f"MaxPool{op_dim}d", None) if pool_type == 'max' else \
             getattr(nn, f"AvgPool{op_dim}d", None) if pool_type == 'avg' else \
             getattr(nn, f"AdaptiveMaxPool{op_dim}d", None) if pool_type == 'adaptivemax' else \
             getattr(nn, f"AdaptiveAvgPool{op_dim}d", None) if pool_type == 'adaptiveavg' else None

    if PoolNd is None: raise ValueError(f"Unsupported pool type '{pool_type}' or op_dim {op_dim}.")

    if 'adaptive' in pool_type:
        return PoolNd(output_size=kernel_size) # kernel_size is output_size for adaptive
    return PoolNd(kernel_size=kernel_size, stride=stride, padding=padding)

