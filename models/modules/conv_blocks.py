# blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import Sequence, Optional, Dict, Any

from .common import get_padding, SplitRelu, chomp
from .conv_layers import CONV_LAYER_MAP, ConvBase
from .norm_act_pool import get_norm_layer, get_activation_layer, get_pooling_layer

__all__ = ['ConvBlock', 'ResBlock']

class ConvBlock(nn.Module):
    """
    Configurable convolutional block (2D or 3D).
    Order: [Pad] -> Conv -> [WeightNorm] -> Norm -> Activation -> [Dropout] -> [Pool]
    """
    def __init__(self,
                 dim: int, # Dimensionality of the convolution (2 or 3)
                 in_channels: int,
                 out_channels: int, # Output channels for the nn.ConvNd layer
                 conv_params: Dict[str, Any],
                 norm_type: Optional[str] = 'batch', # 'batch', 'layer', 'rms', or None
                 norm_params: Optional[Dict[str, Any]] = None, # Params for batch/layer/rms norm
                 act_type: Optional[str] = 'relu', # 'relu', 'leakyrelu', 'prelu', 'gelu', 'mish', 'splitrelu', 'swish', 'none'
                 act_params: Optional[Dict[str, Any]] = None, # Params for activation
                 pool_params: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.0,
                 use_weight_norm: bool = False,
                 causal: bool = True, # If True and dim=3, applies causal temporal padding
                 order: Sequence[str] = ('pad', 'conv', 'norm', 'act', 'dropout', 'pool')
                 ):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self._out_channels_conv = out_channels # Channels directly out of nn.ConvNd
        self.causal = causal
        self.order = order

        # --- Prepare Params (with defaults) ---
        # This is annoying. We need to fix the config handling.
        if norm_type in [None, 'none']: norm_params = {}
        if act_type in [None, 'none']: act_params = {}
        if pool_params is None or (isinstance(pool_params, str) and pool_params.lower() == 'none'):
            pool_params = {}
        
        _conv_params = conv_params.copy()
        _norm_params = (norm_params or {}).copy()
        _act_params = (act_params or {}).copy()
        _pool_params = (pool_params or {}).copy()


        # --- Padding Handling ---
        # This padding is applied via F.pad if `causal` is True or 'pad' in order.
        self.padding_amount = (0,) * (2 * self.dim)
        # The conv layer's own padding is used if not causal and 'pad' not in order.
        # If causal, conv layer padding is set to 0.
        conv_native_padding = _conv_params.get('padding', 0)

        if self.causal and self.dim == 3: # Only apply get_padding for 3D causal
            dilation = _conv_params.get('dilation', 1)
            kernel_size_for_pad = _conv_params['kernel_size']
            self.padding_amount = get_padding(kernel_size_for_pad, dilation, self.dim, self.causal, conv_native_padding)
            _conv_params['padding'] = 0 # External padding will be used
        elif 'pad' in self.order: # General external padding (rarely used with current get_padding)
            dilation = _conv_params.get('dilation', 1)
            kernel_size_for_pad = _conv_params['kernel_size']
            self.padding_amount = get_padding(kernel_size_for_pad, dilation, self.dim, self.causal)
            _conv_params['padding'] = 0
        else: # Use nn.ConvNd's own padding parameter
            _conv_params['padding'] = conv_native_padding


        # --- Build Components ---
        self.components = nn.ModuleDict()

        # 1. Convolution
        conv_type = _conv_params.pop('type', 'standard')
        ConvClass = CONV_LAYER_MAP.get(conv_type)
        if ConvClass is None: raise ValueError(f"Unknown conv type: '{conv_type}'.")

        _conv_params.setdefault('bias', norm_type in ['none', None]) # Bias if no norm

        # Add dim parameter for conv types that need it
        _conv_params['dim'] = self.dim

        conv_layer = ConvClass(in_channels=self.in_channels, out_channels=self._out_channels_conv, **_conv_params)

        # Apply weight normalization only to layers that have a direct 'weight' parameter
        # DepthwiseConv and other composite layers don't have a direct 'weight' attribute
        if use_weight_norm:
            if hasattr(conv_layer, 'apply_weight_norm'):
                conv_layer.apply_weight_norm(_conv_params.get('weight_norm_dim', 0))
            else:
                # For composite layers like DepthwiseConv, skip weight norm or apply to sub-layers
                # For now, we'll just skip it with a warning
                import warnings
                warnings.warn(f"Weight normalization requested but {conv_type} conv doesn't have a direct 'apply_weight_norm' method. Skipping.")

        self.components['conv'] = conv_layer

        current_channels_for_norm_act = self._out_channels_conv

        self.components['norm'] = get_norm_layer(norm_type, current_channels_for_norm_act, self.dim, _norm_params)

        # 3. Activation
        self.components['act'] = get_activation_layer(act_type, _act_params)
        self._is_split_relu = isinstance(self.components['act'], SplitRelu)

        # 4. Dropout
        self.components['dropout'] = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 5. Pooling
        # For AABlur and SE pooling, we need to pass the conv output channel count
        if _pool_params and _pool_params.get('type', '').lower() in ('aablur', 'se_pool'):
            _pool_params = _pool_params.copy()  # Don't modify the original
            _pool_params['channels'] = self._out_channels_conv
        self.components['pool'] = get_pooling_layer(_pool_params, self.dim)

    def _apply_pad(self, x: torch.Tensor) -> torch.Tensor:
        if any(p > 0 for p in self.padding_amount):
            # causal-time replicate on 3D; zeros elsewhere would also be fine,
            # but we keep replicate since that's your default to avoid edge effects.
            return F.pad(x, self.padding_amount, mode='replicate')
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name in self.order:
            if layer_name == 'pad':
                x = self._apply_pad(x)
            else:
                layer = self.components[layer_name]
                x = layer(x)
        
        return x

    @property
    def output_channels(self) -> int:
        """Returns the number of channels after the Conv layer and SplitReLU (if used)."""
        # Your SplitReLU doubles channels, so account for that.
        return self._out_channels_conv * 2 if self._is_split_relu else self._out_channels_conv

    def get_conv_layer(self) -> ConvBase:
         """Utility to get the underlying ConvBase layer, handling weight_norm."""
         # Use safe dictionary-like access for ModuleDict
         conv = self.components['conv'] if 'conv' in self.components else None
         # Unwrap weight_norm if applied
         if hasattr(torch.nn.utils.parametrize, 'ParametrizedModule'):
             if isinstance(conv, torch.nn.utils.parametrize.ParametrizedModule):
                 if hasattr(conv, 'module'): return conv.module # type: ignore
         # Should be a ConvBase instance or None
         return conv # type: ignore

# --- ResBlock Helper ---

def _minimal_crop_like(t: torch.Tensor, ref: torch.Tensor, causal_time: bool) -> torch.Tensor:
    """Crop t to ref on time/space if needed (never pad). For causal time, keep last frames."""
    if t.shape == ref.shape:
        return t
    if t.dim() != ref.dim():
        raise RuntimeError(f"Shape rank mismatch: {t.shape} vs {ref.shape}")

    # time dim index (for 3D tensors): (B,C,T,H,W)
    if t.dim() == 5 and t.shape[2] != ref.shape[2]:
        if causal_time:
            t = t[:, :, -ref.shape[2]:]  # keep most recent frames
        else:
            # center-crop time if ever needed non-causally
            dt = t.shape[2] - ref.shape[2]
            start = max(dt // 2, 0)
            t = t[:, :, start:start + ref.shape[2]]

    # spatial center-crop to match
    if t.shape[-2:] != ref.shape[-2:]:
        t = chomp(t, (ref.shape[-2], ref.shape[-1]))
    return t


class ResBlock(nn.Module):
    """
    ResNet-style block wrapper.

    Expectation: the `shortcut` should already project to the same channel/stride
    as `main_block`. We keep a tiny crop guard for odd/even padding quirks.
    No zero-padding or truncation of channels at runtime.
    """
    def __init__(self,
                 main_block: ConvBlock,
                 shortcut: nn.Module,
                 post_add_activation: nn.Module = nn.Identity(),
                 causal_time: bool = True,
                 allow_crop: bool = True):
        super().__init__()
        self.main_block = main_block
        self.shortcut = shortcut
        self.post_add_activation = post_add_activation
        self.causal_time = bool(causal_time)
        self.allow_crop = bool(allow_crop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main_block(x)
        s = self.shortcut(x)

        if y.shape != s.shape:
            if not self.allow_crop:
                raise RuntimeError(
                    f"ResBlock mismatch: main {tuple(y.shape)} vs shortcut {tuple(s.shape)}. "
                    f"Use a projection shortcut with matching stride/channels."
                )
            s = _minimal_crop_like(s, y, causal_time=self.causal_time)

        out = y + s
        return self.post_add_activation(out)

    @property
    def output_channels(self) -> int:
        return self.main_block.output_channels
