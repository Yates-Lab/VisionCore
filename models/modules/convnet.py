# networks.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Dict, Any, Tuple, Union, Optional

from .common import chomp
# Avoid circular import - ViViT will be imported lazily when needed

def chomp_causal_spatial(tensor_to_crop: torch.Tensor, reference_tensor: torch.Tensor) -> torch.Tensor:
    """
    Crops the input tensor causally in time and centrally in space to match the
    shape of the reference tensor.

    Args:
        tensor_to_crop: Tensor to be cropped
        reference_tensor: Tensor whose shape we want to match

    Returns:
        Cropped tensor with same shape as reference_tensor (except channels)
    """
    # 1. Causal temporal cropping (if 5D tensor)
    if tensor_to_crop.dim() == 5 and tensor_to_crop.shape[2] > reference_tensor.shape[2]:
        target_temporal_dim = reference_tensor.shape[2]
        tensor_to_crop = tensor_to_crop[:, :, -target_temporal_dim:]

    # 2. Spatial center-cropping using existing chomp function
    target_spatial_shape = reference_tensor.shape[-2:]
    if tensor_to_crop.shape[-2:] != target_spatial_shape:
        tensor_to_crop = chomp(tensor_to_crop, target_spatial_shape)

    return tensor_to_crop
from .conv_blocks import ConvBlock, ResBlock
from .norm_act_pool import get_activation_layer
from .shifttcn import ShiftTCN

__all__ = ['BaseConvNet', 'VanillaCNN', 'ResNet', 'DenseNet', 'ShiftTCN']

class BaseConvNet(nn.Module):
    """Base class for configurable convolutional networks."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim: int = config['dim']
        self.initial_channels: int = config['initial_channels']
        self.base_channels: int = config.get('base_channels', self.initial_channels)
        self.use_checkpointing: bool = config.get('checkpointing', False)
        self.layers = nn.ModuleList()
        self._build_network()
        self.final_activation = get_activation_layer(config.get('final_activation', 'none'))

    def _build_network(self): raise NotImplementedError
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpointing and self.training and isinstance(layer, (ConvBlock, ResBlock)):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.final_activation(x)
    
    def get_output_channels(self) -> int:
        if not self.layers: return self.initial_channels
        last_layer_with_channels = self.initial_channels
        for layer in reversed(self.layers):
            if hasattr(layer, 'output_channels'):
                last_layer_with_channels = layer.output_channels
                break
        return last_layer_with_channels


class VanillaCNN(BaseConvNet):
    """Sequential CNN built from ConvBlocks."""
    def _build_network(self):
        current_channels = self.initial_channels

        # Unified format only: channels list + block_config
        if 'channels' not in self.config or 'block_config' not in self.config:
            raise ValueError("VanillaCNN requires 'channels' list and 'block_config' in config. Legacy format is no longer supported.")

        channels = self.config['channels']
        block_cfg_base = self.config['block_config'].copy()
        block_cfg_base['dim'] = self.dim

        for i, out_channels in enumerate(channels):
            block = ConvBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                **block_cfg_base
            )
            self.layers.append(block)
            current_channels = block.output_channels

# ResBlock helper moved to blocks.py for better organization if preferred,
# but keeping it here is also fine if it's only used by ResNet.
# Let's assume it's defined in blocks.py as per the previous update.
# from .blocks import ResBlock

class ResNet(BaseConvNet):
    """ResNet built from ConvBlocks."""
    def _build_network(self):
        current_channels = self.initial_channels

        # Optional Stem
        if 'stem_config' in self.config:
            stem_cfg = self.config['stem_config'].copy()
            stem_cfg['dim'] = self.dim
            out_ch_stem = stem_cfg.pop('out_channels')
            # No special handling needed for normalization parameters
            # ConvBlock will handle normalization internally
            stem = ConvBlock(in_channels=current_channels, out_channels=out_ch_stem, **stem_cfg)
            self.layers.append(stem)
            current_channels = stem.output_channels

        # Support both single block_config (backwards compatible) and list of block_configs
        if 'channels' not in self.config:
            raise ValueError("ResNet requires 'channels' list in config.")

        # Check for block configuration - support both old and new formats
        if 'block_config' in self.config and 'block_configs' in self.config:
            raise ValueError("ResNet config cannot have both 'block_config' and 'block_configs'. Use one or the other.")

        if 'block_config' not in self.config and 'block_configs' not in self.config:
            raise ValueError("ResNet requires either 'block_config' (single config for all layers) or 'block_configs' (list of configs per layer).")

        channels = self.config['channels']

        # Handle both single block_config and list of block_configs
        if 'block_config' in self.config:
            # Backwards compatible: single block config for all layers
            block_cfg_base = self.config['block_config'].copy()
            block_cfg_base['dim'] = self.dim
            block_configs = [block_cfg_base.copy() for _ in channels]
        else:
            # New format: list of block configs, one per layer
            block_configs = self.config['block_configs']
            if len(block_configs) != len(channels):
                raise ValueError(f"Number of block_configs ({len(block_configs)}) must match number of channel stages ({len(channels)})")

            # Add dim to each config
            block_configs = [cfg.copy() for cfg in block_configs]
            for cfg in block_configs:
                cfg['dim'] = self.dim

        for i, (out_channels, block_cfg) in enumerate(zip(channels, block_configs)):
            # Remove out_channels from block config if present to avoid conflicts
            cfg = block_cfg.copy()
            cfg.pop('out_channels', None)
            cfg.pop('channel_multiplier', None)

            main_block = ConvBlock(in_channels=current_channels, out_channels=out_channels, **cfg)
            main_block_out_ch = main_block.output_channels

            self._add_resnet_block(main_block, current_channels, main_block_out_ch, cfg)
            current_channels = main_block_out_ch

    def _add_resnet_block(self, main_block, current_channels, main_block_out_ch, block_cfg=None):
        """Add a ResNet block with shortcut connection."""
        # Enhanced shortcut connection handling
        needs_projection = False

        # Infer stride from main_block's conv_params and pool_params
        if block_cfg:
            block_conv_params = block_cfg.get('conv_params', {})
            block_pool_params = block_cfg.get('pool_params', {})
        else:
            # For new unified format, get from main_block config
            block_conv_params = getattr(main_block, 'config', {}).get('conv_params', {})
            block_pool_params = getattr(main_block, 'config', {}).get('pool_params', {})

        conv_stride_cfg = block_conv_params.get('stride', 1)
        pool_stride_cfg = block_pool_params.get('stride', 1) if block_pool_params else 1

        # Calculate total effective stride (conv_stride * pool_stride)
        if isinstance(conv_stride_cfg, tuple) and isinstance(pool_stride_cfg, tuple):
            total_stride_cfg = tuple(c * p for c, p in zip(conv_stride_cfg, pool_stride_cfg))
        elif isinstance(conv_stride_cfg, tuple):
            total_stride_cfg = tuple(c * pool_stride_cfg for c in conv_stride_cfg)
        elif isinstance(pool_stride_cfg, tuple):
            total_stride_cfg = tuple(conv_stride_cfg * p for p in pool_stride_cfg)
        else:
            total_stride_cfg = conv_stride_cfg * pool_stride_cfg

        # Check if total stride requires projection
        if isinstance(total_stride_cfg, tuple):
            needs_projection = any(s > 1 for s in total_stride_cfg)
        else:
            needs_projection = total_stride_cfg > 1

        # Check if channel dimensions require projection
        if current_channels != main_block_out_ch:
            needs_projection = True

        # Check if projection is enabled in config
        enable_projection = self.config.get('resnet_shortcut_projection', True)

        # Default to identity shortcut
        shortcut: nn.Module = nn.Identity()

        # Create projection shortcut if needed and enabled
        if needs_projection and enable_projection:
            ShortcutConv = nn.Conv2d if self.dim == 2 else nn.Conv3d

            # Get shortcut parameters from config
            sc_params = self.config.get('resnet_shortcut_params', {})
            kernel_size = sc_params.get('kernel_size', 1)
            bias = sc_params.get('bias', False)

            # Shortcut norm can be configured
            sc_norm_type = self.config.get('resnet_shortcut_norm_type')

            # Get shortcut normalization parameters
            sc_norm_params = self.config.get('resnet_shortcut_norm_params', {})

            from .norm_act_pool import get_norm_layer
            sc_norm = get_norm_layer(sc_norm_type, main_block_out_ch, self.dim, sc_norm_params)

            # Create the projection shortcut
            shortcut = nn.Sequential(
                ShortcutConv(
                    current_channels,
                    main_block_out_ch,
                    kernel_size=kernel_size,
                    stride=total_stride_cfg,
                    bias=bias
                ),
                sc_norm
            )

        post_add_act = get_activation_layer(self.config.get('resnet_post_add_activation'))
        self.layers.append(ResBlock(main_block, shortcut, post_add_act))


class DenseNet(BaseConvNet):
    """
    DenseNet built from ConvBlocks.

    In a DenseNet, each layer receives the feature maps from all preceding layers.
    The number of input channels for each layer increases as we go deeper in the network.
    """
    def _build_network(self, verbose=False):
        # Initialize with the input channels
        current_channels = self.initial_channels
        if verbose:
            print(f"\nBuilding DenseNet with initial channels: {current_channels}")

        # Unified format only: explicit channel list
        if 'channels' not in self.config or 'block_config' not in self.config:
            raise ValueError("DenseNet requires 'channels' list and 'block_config' in config. Legacy format is no longer supported.")

        channels = self.config['channels']
        num_blocks = len(channels)
        if verbose:
            print(f"Using channels format: {channels}")

        block_cfg_base = self.config['block_config'].copy()
        if verbose:
            print(f"Base block config: {block_cfg_base}")
        block_cfg_base['dim'] = self.dim

        # Check if we're using SplitReLU which would double the output channels
        uses_split_relu = block_cfg_base.get('act_type', '') == 'splitrelu'
        if uses_split_relu and verbose:
            print(f"DenseNet using SplitReLU - output channels will be doubled")

        # Store channel counts for each layer for debugging and validation
        self.input_channels_per_block = [current_channels]

        # Create each block with the correct number of input channels
        for i in range(num_blocks):
            if verbose:
                print(f"Creating DenseNet block {i} with input channels: {current_channels}")

            # Create a block that takes all previous feature maps as input
            block = ConvBlock(
                in_channels=current_channels,
                out_channels=channels[i],  # Use channels[i] instead of growth_rate
                **block_cfg_base
            )
            self.layers.append(block)

            # Check if this block uses SplitReLU
            block_uses_split_relu = hasattr(block, '_is_split_relu') and block._is_split_relu

            # Get the actual output channels from the block
            block_output_channels = block.output_channels
            

            # Update the channel count for the next block
            # The next block will receive all previous feature maps plus this block's output
            current_channels += block_output_channels
            if verbose:
                print(f"  Block {i} output channels: {block_output_channels}")
                print(f"  Block {i} uses SplitReLU: {block_uses_split_relu}")
                print(f"  Total channels after block {i}: {current_channels}")

            # Store the input channel count for the next block
            self.input_channels_per_block.append(current_channels)

        # The final output channels is the sum of initial channels and all growth
        self._final_out_channels = current_channels
        if verbose:
            print(f"DenseNet final output channels: {self._final_out_channels}")

    # We've moved the functionality of _run_one_block directly into the forward method
    # to avoid issues with checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient forward pass through the DenseNet with incremental cropping and concatenation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with all feature maps concatenated
        """
        # Start with the input as our initial concatenated feature map
        x_cat = x

        # Process through each block
        for block in self.layers:
            # Run the block on the current concatenated features
            if self.use_checkpointing and self.training:
                block_out = checkpoint(
                    block,
                    x_cat,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
            else:
                block_out = block(x_cat)

            # Crop the previous concatenated tensor to match the new output's shape
            x_cat_chomped = chomp_causal_spatial(x_cat, block_out)

            # Create the new concatenated tensor for the next iteration
            x_cat = torch.cat([x_cat_chomped, block_out], dim=1)

        return self.final_activation(x_cat)

    def get_output_channels(self) -> int:
        """Get the number of output channels from the network."""
        return self._final_out_channels


CONVNETS = {'vanilla': VanillaCNN,
            'cnn': VanillaCNN,
            'resnet': ResNet,
            'densenet': DenseNet,
            'shifttcn': ShiftTCN,
            'vivit': None}  # ViViT handled separately in factory due to lazy import

def build_model(config: Dict[str, Any]) -> nn.Module:
    """Builds a CNN model based on config."""
    model_type = config.get('model_type', 'vanilla').lower()
    if model_type in ['vanilla', 'cnn']: return VanillaCNN(config)
    if model_type == 'resnet': return ResNet(config)
    if model_type == 'densenet': return DenseNet(config)
    if model_type == 'shifttcn': return ShiftTCN(config)
    if model_type in ['x3d', 'x3dnet']:
        # Import X3DNet from x3d module
        from .x3d import X3DNet
        return X3DNet(config)
    if model_type == 'vivit':
        # Import ViViT from viT module (lazy import to avoid circular dependency)
        from .viT import ViViT
        return ViViT(config)
    raise ValueError(f"Unknown model type: '{model_type}'.")