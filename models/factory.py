"""
Factory module for creating model components.

This module provides factory functions for creating model components based on
configuration dictionaries. It centralizes the logic for component creation
and makes it easier to add new component types.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from .modules import (
    DAModel, ConvBlock, ConvLSTM, ConvGRU,
    DynamicGaussianReadout, TemporalBasis, DynamicGaussianReadoutEI, DynamicGaussianSN, AffineAdapter,
    LearnableTemporalConv
)
from .config import build_component_config

# Type aliases for clarity
ConfigDict = Dict[str, Any]

def create_frontend(
    frontend_type: str,
    in_channels: int,
    sampling_rate: int = 240,
    **kwargs
) -> Tuple[nn.Module, int]:
    """
    Create a frontend component.

    Args:
        frontend_type: Type of frontend ('da', 'conv', 'temporal_basis', 'learnable_temporal', 'adapter', 'none')
        in_channels: Number of input channels
        height: Height of input (required for 'da' frontend)
        width: Width of input (required for 'da' frontend)
        sampling_rate: Sampling rate in Hz (required for 'da' frontend)
        **kwargs: Additional parameters for the frontend

    Returns:
        Tuple of (frontend module, output channels)
    """
    if frontend_type == 'none':
        return nn.Identity(), in_channels

    elif frontend_type == 'da':
        assert in_channels == 1, "DA frontend only supports 1 input channel right now."

        # Create the DA model
        frontend = DAModel(sampling_rate=sampling_rate, **kwargs)
        return frontend, 1  # DAModel outputs 1 channel

    elif frontend_type == 'temporal_basis':

        # Create the TemporalBasis model
        frontend = TemporalBasis(in_channels=in_channels, sampling_rate=sampling_rate, **kwargs)

        # Calculate output channels: input_channels * (num_delta_funcs + num_cosine_funcs)
        output_channels = (frontend.num_cosine_funcs + frontend.num_delta_funcs) * in_channels
        return frontend, output_channels

    elif frontend_type == 'conv':

        # Create the ConvBlock
        frontend = ConvBlock(dim=3, in_channels=in_channels, **kwargs)

        out_channels = frontend.output_channels

        return frontend, out_channels

    elif frontend_type == 'learnable_temporal':

        # Create the LearnableTemporalConv
        frontend = LearnableTemporalConv(**kwargs)

        # Calculate output channels: input_channels * num_channels
        output_channels = frontend.get_output_channels(in_channels)
        return frontend, output_channels

    elif frontend_type == 'adapter':

        # Create the AffineAdapter
        frontend = AffineAdapter(**kwargs)

        # AffineAdapter preserves input channels
        return frontend, in_channels

    else:
        raise ValueError(f"Unknown frontend type: {frontend_type}")


def create_convnet(
    convnet_type: str,
    in_channels: int,
    **kwargs: Any,
) -> Tuple[nn.Module, int]:
    """
    kwargs are exactly the YAML “params:” subtree for this convnet.
    Example call:
        create_convnet("densenet", 1,
                       dim=3, growth_rate=4, num_blocks=3,
                       block_config={ "conv_params": {"type": "depthwise"} })
    """

    # 'none' is still allowed
    if convnet_type.lower() == "none":
        return nn.Identity(), in_channels

    # aliases
    if convnet_type.lower() in {"conv", "cnn"}:
        convnet_type = "vanilla"

    # build the config expected by build_convnet()
    cfg = dict(model_type=convnet_type.lower(),
               initial_channels=in_channels,
               **kwargs)                # kwargs come straight from YAML

    # Handle X3D separately since it's in its own module
    if convnet_type.lower() in ['x3d', 'x3dnet']:
        from .modules.x3d import X3DNet
        core = X3DNet(cfg)
        return core, core.get_output_channels()

    # Handle other convnets
    from .modules.convnet import CONVNETS
    # build the network
    core = CONVNETS[convnet_type.lower()](cfg)

    # keep your wrapper for shape-handling (you can thin it later)
    # wrapped = ConvNetWrapper(core, is_3d=cfg.get("dim", 3) == 3)

    return core, core.get_output_channels()


def create_recurrent(
    recurrent_type: str,
    input_dim: int,
    hidden_dim: int = 64,
    kernel_size: int = 3,
    **kwargs
) -> Tuple[nn.Module, int]:
    """
    Returns (module, output_channels)
    `input_dim` is interpreted as **channel-dim** when the type is convolutional.
    """
    r = recurrent_type.lower()
    if r == "none":
        return nn.Identity(), input_dim

    if r == "convgru":
        # Filter kwargs to only include parameters that ConvGRU accepts
        convgru_kwargs = {k: v for k, v in kwargs.items() if k in ['fast_phase']}
        layer = ConvGRU(input_dim, hidden_dim, kernel_size, **convgru_kwargs)
        # Calculate output channels: hidden_dim + input_dim if fast_phase=True
        fast_phase = kwargs.get('fast_phase', False)
        output_channels = (input_dim + hidden_dim) if fast_phase else hidden_dim
        return layer, output_channels

    if r == "convlstm":
        # Filter kwargs to only include parameters that ConvLSTM accepts
        convlstm_kwargs = {k: v for k, v in kwargs.items() if k in ['fast_phase']}
        layer = ConvLSTM(input_dim, hidden_dim, kernel_size, **convlstm_kwargs)
        # Calculate output channels: hidden_dim + input_dim if fast_phase=True
        fast_phase = kwargs.get('fast_phase', False)
        output_channels = (input_dim + hidden_dim) if fast_phase else hidden_dim
        return layer, output_channels

    if r == "gru":
        layer = nn.GRU(input_dim, hidden_dim, batch_first=True, **kwargs)
        return layer, hidden_dim
    
    if r == "lstm":
        layer = nn.LSTM(input_dim, hidden_dim, batch_first=True, **kwargs)
        return layer, hidden_dim

    raise ValueError(f"Unknown recurrent_type '{recurrent_type}'")


def create_modulator(
    modulator_type: str,
    **kwargs
) -> Tuple[Optional[nn.Module], int]:
    """
    Create a modulator component.

    Args:
        modulator_type: Type of modulator ('concat', 'film', 'stn', 'pc', 'none')
        **kwargs: Additional parameters for the modulator

    Returns:
        Tuple of (modulator module, output dimension)
    """
    if modulator_type.lower() == 'none':
        return None, 0
    
    # Import the MODULATORS dictionary
    from .modules.modulator import MODULATORS
    
    # Check if modulator type exists
    modulator_type = modulator_type.lower()
    if modulator_type not in MODULATORS:
        raise ValueError(f"Unknown modulator type: {modulator_type}")
    
    # Create config dictionary
    config = {
        'type': modulator_type,
        'behavior_dim': kwargs.pop('behavior_dim', kwargs.pop('n_vars', 2)),
    }
    
    # Add remaining kwargs to config
    config.update(kwargs)
    
    # Build modulator using the appropriate class from MODULATORS
    modulator = MODULATORS[modulator_type](config)
    
    # Return modulator and its output dimension
    return modulator, modulator.out_dim

def create_readout(
    readout_type: str,
    in_channels: int,
    **kwargs
) -> nn.Module:
    """
    Create a readout component.

    Args:
        readout_type: Type of readout ('gaussian', 'linear')
        in_channels: Number of input channels
        **kwargs: Additional parameters for the readout

    Returns:
        Readout module
    """
    if readout_type == 'gaussian':
        # Get default parameters and merge with provided parameters
        config = build_component_config('readout', 'gaussian',
                                       in_channels=in_channels,
                                       **kwargs)

        # Create the DynamicGaussianReadout
        return DynamicGaussianReadout(**config)

    elif readout_type == 'gaussianei':
        # Get default parameters and merge with provided parameters
        config = build_component_config('readout', 'gaussian_ei',
                                       in_channels=in_channels,
                                       **kwargs)

        # Create the DynamicGaussianReadoutEI
        return DynamicGaussianReadoutEI(**config)

    elif readout_type == 'gaussiansn':
        # Get default parameters and merge with provided parameters
        config = build_component_config('readout', 'gaussian_sn',
                                       in_channels=in_channels,
                                       **kwargs)

        # Create the DynamicGaussianSN
        return DynamicGaussianSN(**config)

    elif readout_type == 'linear':
        # Get default parameters and merge with provided parameters
        config = build_component_config('readout', 'linear',
                                       in_channels=in_channels,
                                       **kwargs)

        # Create a custom linear readout that handles spatial dimensions
        class FlattenedLinearReadout(nn.Module):
            def __init__(self, in_channels, n_units, bias):
                super().__init__()
                self.in_channels = in_channels
                self.n_units = n_units
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
                self.fc = nn.Linear(in_channels, n_units, bias=bias)
                self.bias = self.fc.bias if bias else None

            def forward(self, x):
                # Handle 5D input (N, C, S, H, W)
                if x.dim() == 5:
                    x = x[:, :, -1]  # Take last time step -> (N, C, H, W)

                # Only pool if spatial dimensions are > 1x1
                if x.shape[-2:] != (1, 1):
                    x = self.adaptive_pool(x)  # -> (N, C, 1, 1)

                x = torch.flatten(x, 1)    # -> (N, C)
                return self.fc(x)

        return FlattenedLinearReadout(
            in_channels=config['in_channels'],
            n_units=config['n_units'],
            bias=config['bias']
        )

    else:
        raise ValueError(f"Unknown readout type: {readout_type}")

def create_model_from_config(config: ConfigDict) -> nn.Module:
    """
    Create a model from a configuration dictionary.

    This function delegates to the build_model function in the build module.

    Args:
        config: Configuration dictionary for the model

    Returns:
        Constructed model
    """
    from .build import build_model
    return build_model(config)
