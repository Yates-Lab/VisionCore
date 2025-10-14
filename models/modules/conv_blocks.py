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
                 order: Sequence[str] = ('conv', 'norm', 'act', 'dropout', 'pool')
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
        # `get_padding` now implements the specific logic:
        # causal temporal for 3D, zero spatial, zero for 2D.
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply F.pad if self.padding_amount is non-zero (i.e., for 3D causal)
        if any(p > 0 for p in self.padding_amount):
            # Mode for F.pad, 'replicate' is a common default for causal
            # Try to get padding_mode from the underlying conv layer if possible
            pad_mode = 'replicate' # Default
            # Use safe dictionary-like access for ModuleDict
            conv_comp = self.components['conv'] if 'conv' in self.components else None
            if conv_comp is not None:
                if hasattr(conv_comp, 'conv') and hasattr(conv_comp.conv, 'padding_mode'): # StandardConv case
                    pad_mode = conv_comp.conv.padding_mode
                elif hasattr(conv_comp, 'padding_mode'): # Direct attribute (e.g., StackedConv2d)
                    pad_mode = conv_comp.padding_mode

            if pad_mode == 'zeros': pad_mode = 'constant' # F.pad uses 'constant' for zero padding
            x = F.pad(x, self.padding_amount, mode=pad_mode)

        # Apply components in specified order
        for layer_name in self.order:
            # Use safe dictionary-like access for ModuleDict
            layer = self.components[layer_name] if layer_name in self.components else None
            if layer is not None: x = layer(x) # Apply if layer exists and is not None/Identity
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
class ResBlock(nn.Module):
     """
     Helper module for a single ResNet block with shortcut connection.

     This implementation handles both identity shortcuts and projection shortcuts,
     and ensures proper spatial dimension matching between the main path and shortcut.
     """
     def __init__(self, main_block: ConvBlock, shortcut: nn.Module, post_add_activation: nn.Module = nn.Identity()):
          super().__init__()
          self.main_block = main_block
          self.shortcut = shortcut
          self.post_add_activation = post_add_activation

     def forward(self, x: torch.Tensor) -> torch.Tensor:
          # Process shortcut path
          identity = self.shortcut(x)

          # Process main path
          out = self.main_block(x)

          # Handle dimension mismatches (both spatial and temporal for 3D)
          # First check if we need to handle temporal dimension mismatch (for 3D convolutions)
          if out.dim() == 5 and identity.dim() == 5 and out.shape[2] != identity.shape[2]:
               # Get the minimum temporal dimension
               min_time = min(out.shape[2], identity.shape[2])

               # Crop both tensors to the minimum temporal size
               # For temporal dimension, we take the last 'min_time' frames
               # This is important for causal convolutions where later frames have more context
               out = out[:, :, -min_time:]
               identity = identity[:, :, -min_time:]

          # Handle spatial dimension mismatches (height and width)
          if out.shape[-2:] != identity.shape[-2:]:
               # Get the minimum spatial dimensions
               min_height = min(out.shape[-2], identity.shape[-2])
               min_width = min(out.shape[-1], identity.shape[-1])

               # Crop both tensors to the minimum size
               out = chomp(out, (min_height, min_width))
               identity = chomp(identity, (min_height, min_width))

          # Handle channel dimension mismatch
          if out.shape[1] != identity.shape[1]:
               # If enable_shortcut_projection is False, we need to handle this case
               # by either padding the smaller tensor or truncating the larger one
               if out.shape[1] > identity.shape[1]:
                    # Pad identity with zeros to match out's channel dimension
                    if out.dim() == 5:  # 3D case
                         pad_size = out.shape[1] - identity.shape[1]
                         zero_pad = torch.zeros(
                              identity.shape[0], pad_size, identity.shape[2],
                              identity.shape[3], identity.shape[4],
                              device=identity.device
                         )
                         identity = torch.cat([identity, zero_pad], dim=1)
                    else:  # 4D case
                         pad_size = out.shape[1] - identity.shape[1]
                         zero_pad = torch.zeros(
                              identity.shape[0], pad_size, identity.shape[2],
                              identity.shape[3], device=identity.device
                         )
                         identity = torch.cat([identity, zero_pad], dim=1)
               else:
                    # Truncate out to match identity's channel dimension
                    out = out[:, :identity.shape[1]]

          # Add residual and apply post-activation
          return self.post_add_activation(out + identity)

     @property
     def output_channels(self) -> int:
          # Output channels are determined by the main block's output
          return self.main_block.output_channels

