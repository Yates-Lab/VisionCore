# networks.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, Tuple
from .common import chomp
from .conv_blocks import ConvBlock, ResBlock
from .norm_act_pool import get_activation_layer

__all__ = ['BaseConvNet', 'VanillaCNN', 'ResNet', 'DenseNet']

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



class BaseConvNet(nn.Module):
    """Base class for configurable convolutional networks."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.initial_channels: int = config['initial_channels']
        self.base_channels: int = config.get('base_channels', self.initial_channels)
        self.layers = nn.ModuleList()
        self._build_network()
        self.final_activation = get_activation_layer(config.get('final_activation', 'none'))

    def _build_network(self): raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'stem'): x = self.stem(x)
        for layer in self.layers:
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
    """Sequential CNN built from ConvBlocks (stem → stack of ConvBlocks)."""
    def _build_network(self, verbose: bool = False):
        current_channels = self.initial_channels
        self.layers = nn.ModuleList()

        # Optional Stem (same behavior as ResNet/DenseNet)
        if 'stem_config' in self.config:
            self.stem, current_channels = _build_stem(
                self.config['stem_config'], self.initial_channels
            )

        # Unified format: channels + block_configs (one per block)
        assert 'channels' in self.config, "VanillaCNN requires 'channels' list in config."
        assert 'block_configs' in self.config, "VanillaCNN requires 'block_configs' list in config."

        channels = self.config['channels']
        block_configs = self.config['block_configs']
        assert len(block_configs) == len(channels), (
            f"Number of block_configs ({len(block_configs)}) must match number of channels ({len(channels)})"
        )

        # Prepare per-block configs (no need to add dim - inferred from kernel_size)
        block_configs = [cfg.copy() for cfg in block_configs]

        # Build blocks
        for i, (out_channels, cfg) in enumerate(zip(channels, block_configs)):
            block = ConvBlock(in_channels=current_channels,
                              out_channels=out_channels,
                              **cfg)
            self.layers.append(block)
            current_channels = block.output_channels

            if verbose:
                print(f"[VanillaCNN] Block {i}: out={out_channels} (eff={block.output_channels}), "
                      f"next_in={current_channels}")

        self._final_out_channels = current_channels

    def get_output_channels(self) -> int:
        return getattr(self, "_final_out_channels", super().get_output_channels())

# ResBlock helper moved to blocks.py for better organization if preferred,
# but keeping it here is also fine if it's only used by ResNet.
# Let's assume it's defined in blocks.py as per the previous update.
# from .blocks import ResBlock

def _build_stem(stem_config: Dict[str, Any], current_channels: int) -> Tuple[nn.Module, int]:
    """Build stem module for convnets."""
    stem_cfg = stem_config.copy()
    out_ch_stem = stem_cfg.pop('out_channels', None)

    if out_ch_stem is None:
        raise ValueError("ConvBlock stem requires 'out_channels' in stem_config")

    # ConvBlock infers dimensionality from kernel_size
    stem = ConvBlock(in_channels=current_channels, out_channels=out_ch_stem, **stem_cfg)

    return stem, stem.output_channels

class ResNet(BaseConvNet):
    """ResNet built from ConvBlocks."""
    def _build_network(self):
        current_channels = self.initial_channels

        # Optional Stem
        if 'stem_config' in self.config:
            self.stem, current_channels = _build_stem(self.config['stem_config'], self.initial_channels)

        assert 'channels' in self.config, "ResNet requires 'channels' list in config."
        assert 'block_configs' in self.config, "ResNet requires 'block_configs' list in config."

        channels = self.config['channels']

        block_configs = self.config['block_configs']
        assert len(block_configs) == len(channels), f"Number of block_configs ({len(block_configs)}) must match number of channel stages ({len(channels)})"

        # Prepare per-block configs (no need to add dim - inferred from kernel_size)
        block_configs = [cfg.copy() for cfg in block_configs]

        for i, (out_channels, block_cfg) in enumerate(zip(channels, block_configs)):
            # Remove out_channels from block config if present to avoid conflicts
            cfg = block_cfg.copy()
            cfg.pop('out_channels', None)
            cfg.pop('channel_multiplier', None)

            main_block = ConvBlock(in_channels=current_channels, out_channels=out_channels, **cfg)
            main_block_out_ch = main_block.output_channels

            self._add_resnet_block(main_block, current_channels, main_block_out_ch, cfg)
            current_channels = main_block_out_ch

    def _get_total_stride(self, block_cfg):
        """Calculate total stride from conv and pool params."""
        if not block_cfg:
            return 1

        conv_params = block_cfg.get('conv_params', {})
        pool_params = block_cfg.get('pool_params', {})

        conv_stride = conv_params.get('stride', 1)
        pool_stride = pool_params.get('stride', 1) if pool_params else 1

        # Multiply strides
        if isinstance(conv_stride, tuple) and isinstance(pool_stride, tuple):
            return tuple(c * p for c, p in zip(conv_stride, pool_stride))
        elif isinstance(conv_stride, tuple):
            return tuple(c * pool_stride for c in conv_stride)
        elif isinstance(pool_stride, tuple):
            return tuple(conv_stride * p for p in pool_stride)
        else:
            return conv_stride * pool_stride

    def _add_resnet_block(self, main_block, current_channels, main_block_out_ch, block_cfg=None):
        """Add a ResNet block with shortcut connection."""

        # Determine if we need projection
        needs_projection = (current_channels != main_block_out_ch)

        # Infer stride from block config
        total_stride = self._get_total_stride(block_cfg)

        # Check if stride requires projection
        if isinstance(total_stride, tuple):
            needs_stride_projection = any(s > 1 for s in total_stride)
        else:
            needs_stride_projection = total_stride > 1

        if needs_projection or needs_stride_projection:
            # Simple 1x1x1 conv projection (linear, no normalization or activation)
            shortcut = nn.Conv3d(
                current_channels,
                main_block_out_ch,
                kernel_size=1,
                stride=total_stride,
                bias=False
            )
        else:
            shortcut = nn.Identity()

        post_add_act = get_activation_layer(self.config.get('resnet_post_add_activation'))
        self.layers.append(ResBlock(main_block, shortcut, post_add_act))


class DenseNet(BaseConvNet):
    """
    DenseNet built from ConvBlocks.

    Each block takes the concatenation of all previous features and
    contributes `growth = block.output_channels` new channels.
    """
    def _build_network(self, verbose: bool = False):
        current_channels = self.initial_channels
        self.layers = nn.ModuleList()
        self.input_channels_per_block = []

        # Optional Stem
        if 'stem_config' in self.config:
            self.stem, current_channels = _build_stem(self.config['stem_config'], self.initial_channels)

        assert 'channels' in self.config, "DenseNet requires a 'channels' list (growth per block)."
        assert 'block_configs' in self.config, "DenseNet requires a 'block_configs' list."

        channels = self.config['channels']              # list of growth values (one per block)
        block_configs = self.config['block_configs']
        assert len(block_configs) == len(channels), (
            f"Number of block_configs ({len(block_configs)}) must match number of channels ({len(channels)})"
        )

        # Prepare per-block configs (no need to add dim - inferred from kernel_size)
        block_configs = [cfg.copy() for cfg in block_configs]
        for cfg in block_configs:
            cfg.pop('out_channels', None)       # we'll set it explicitly from `channels`
            cfg.pop('channel_multiplier', None) # guard against stray keys

        # Build blocks
        for i, (growth, cfg) in enumerate(zip(channels, block_configs)):
            main_block = ConvBlock(in_channels=current_channels,
                                   out_channels=growth,
                                   **cfg)
            self.layers.append(main_block)

            self.input_channels_per_block.append(current_channels)

            growth_out = main_block.output_channels  # accounts for SplitRelu doubling, etc.
            current_channels += growth_out           # dense concat grows channels

            if verbose:
                print(f"[DenseNet] Block {i}: growth={growth_out}, uses SplitReLU={main_block._is_split_relu}, "
                      f"next_input_channels={current_channels}")

        self._final_out_channels = current_channels
        if verbose:
            print(f"[DenseNet] final output channels: {self._final_out_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature map (stem or raw input)
        x_cat = self.stem(x) if hasattr(self, 'stem') else x

        # Dense concatenation
        for block in self.layers:
            if x_cat.requires_grad and block.training:
                out = checkpoint(block, x_cat)
            else:
                out = block(x_cat)

            x_cat = torch.cat([chomp_causal_spatial(x_cat, out), out], dim=1)

        return self.final_activation(x_cat)

    def get_output_channels(self) -> int:
        return self._final_out_channels


CONVNETS = {'vanilla': VanillaCNN,
            'cnn': VanillaCNN,
            'resnet': ResNet,
            'densenet': DenseNet}  # ViViT handled separately in factory due to lazy import

def build_model(config: Dict[str, Any]) -> nn.Module:
    """Builds a CNN model based on config."""
    model_type = config.get('model_type', 'vanilla').lower()
    if model_type in ['vanilla', 'cnn']: return VanillaCNN(config)
    if model_type == 'resnet': return ResNet(config)
    if model_type == 'densenet': return DenseNet(config)
    raise ValueError(f"Unknown model type: '{model_type}'.")