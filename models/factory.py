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
    DAModel, ConvBlock, ConvGRU, RecurrentWrapper,
    TemporalBasis, AffineAdapter,
    LearnableTemporalConv
)

from .modules import readout as readout_modules

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
        return lambda x: x, in_channels

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

        # Create the ConvBlock (dim inferred from kernel_size in kwargs)
        frontend = ConvBlock(in_channels=in_channels, **kwargs)

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

    All recurrent modules follow VisionCore interface:
    - Input: (B, C, T, H, W)
    - Output: (B, C_hidden, T, H, W) for ConvGRU or (B, C_hidden, T, 1, 1) for GRU/LSTM

    `input_dim` is interpreted as **channel-dim** for ConvGRU.
    For GRU/LSTM, it will be flattened to C*H*W internally by RecurrentWrapper.
    """
    r = recurrent_type.lower()
    if r == "none":
        return nn.Identity(), input_dim

    if r == "convgru":
        # ConvGRU handles (B, C, T, H, W) natively
        layer = ConvGRU(input_dim, hidden_dim, kernel_size, **kwargs)
        return layer, hidden_dim

    elif r == "gru":
        # Wrap nn.GRU to handle (B, C, T, H, W) → (B, hidden_size, T, 1, 1)
        # RNN will be created lazily on first forward pass
        layer = RecurrentWrapper('gru', hidden_dim, input_channels=input_dim, **kwargs)
        return layer, hidden_dim

    elif r == "lstm":
        # Wrap nn.LSTM to handle (B, C, T, H, W) → (B, hidden_size, T, 1, 1)
        # RNN will be created lazily on first forward pass
        layer = RecurrentWrapper('lstm', hidden_dim, input_channels=input_dim, **kwargs)
        return layer, hidden_dim

    else:
        raise ValueError(f"Unknown recurrent type: {recurrent_type}")



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
    
    # Handle polar modulator separately since it has different interface
    modulator_type = modulator_type.lower()
    if modulator_type == 'polar':
        from .modules.polar_modulator import PolarModulator
        # feature_dim should be (n_pairs, n_levels) for Polar
        feature_dim = kwargs.pop('feature_dim', None)
        if feature_dim is None:
            raise ValueError("Polar modulator requires 'feature_dim' as (n_pairs, n_levels)")

        config = {
            'n_pairs': feature_dim[0],
            'n_levels': feature_dim[1],
            'behavior_dim': kwargs.pop('behavior_dim', kwargs.pop('n_vars', 2)),
        }
        config.update(kwargs)

        modulator = PolarModulator(config)
        return modulator, modulator.out_dim

    # Import the MODULATORS dictionary
    from .modules.modulator import MODULATORS

    # Check if modulator type exists
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

READOUTS = {
    'gaussian': readout_modules.DynamicGaussianReadout,
    'linear': readout_modules.FlattenedLinearReadout,
}

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

    return READOUTS[readout_type](in_channels=in_channels, **kwargs)