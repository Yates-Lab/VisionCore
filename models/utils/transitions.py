"""
Utility functions for transitions between different layers in neural networks.

This module provides functions to handle transitions between different types of layers,
such as convolutional to recurrent, recurrent to readout, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten_conv_for_recurrent(x):
    """
    Flatten convolutional output for recurrent input.
    
    Converts from (N, C, S, H, W) to (N, C*H*W, S) format.
    
    Args:
        x (torch.Tensor): Convolutional output tensor with shape (N, C, S, H, W)
            where N is batch size, C is channels, S is sequence length,
            H is height, and W is width.
    
    Returns:
        torch.Tensor: Flattened tensor with shape (N, C*H*W, S)
    """
    # x shape: (N, C, S, H, W)
    N, C, S, H, W = x.shape
    
    # First, permute to (N, S, C, H, W)
    x_permuted = x.permute(0, 2, 1, 3, 4)
    
    # Then flatten C, H, W for each time step
    # Result: (N, S, C*H*W)
    x_flattened = x_permuted.reshape(N, S, -1)
    
    # Finally, permute to (N, C*H*W, S) for recurrent input
    x_final = x_flattened.permute(0, 2, 1)
    
    return x_final

def expand_modulator_for_sequence(modulator, sequence_length):
    """
    Expand modulator output to match sequence dimension.
    
    Converts from (N, C_mod) to (N, C_mod, S) format.
    
    Args:
        modulator (torch.Tensor): Modulator output tensor with shape (N, C_mod)
            where N is batch size and C_mod is modulator channels.
        sequence_length (int): Length of the sequence to expand to.
    
    Returns:
        torch.Tensor: Expanded tensor with shape (N, C_mod, S)
    """
    # modulator shape: (N, C_mod)
    if modulator.dim() == 2:
        # Expand to match sequence dimension
        return modulator.unsqueeze(-1).expand(-1, -1, sequence_length)
    
    # If already has sequence dimension, return as is
    return modulator

def concat_features(x, modulator):
    """
    Concatenate features along the channel dimension.
    
    Args:
        x (torch.Tensor): Main features tensor with shape (N, C, S)
            where N is batch size, C is channels, and S is sequence length.
        modulator (torch.Tensor): Modulator tensor with shape (N, C_mod, S)
            where C_mod is modulator channels.
    
    Returns:
        torch.Tensor: Concatenated tensor with shape (N, C+C_mod, S)
    """
    # Ensure modulator has the right shape
    if modulator.dim() == 2:
        # Expand to match sequence dimension
        modulator = expand_modulator_for_sequence(modulator, x.shape[2])
    
    # Concatenate along feature dimension
    return torch.cat([x, modulator], dim=1)

def expand_recurrent_to_spatial(x, height, width):
    """
    Expand recurrent output to spatial dimensions.
    
    Converts from (N, C) to (N, C, H, W) format for non-sequence output,
    or from (N, C, S) to (N, C, S, H, W) for sequence output.
    
    Args:
        x (torch.Tensor): Recurrent output tensor with shape (N, C) or (N, C, S)
            where N is batch size, C is channels, and S is sequence length.
        height (int): Height of the spatial dimensions.
        width (int): Width of the spatial dimensions.
    
    Returns:
        torch.Tensor: Expanded tensor with shape (N, C, H, W) or (N, C, S, H, W)
    """
    if x.dim() == 2:
        # Non-sequence output: (N, C) -> (N, C, H, W)
        return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
    elif x.dim() == 3:
        # Sequence output: (N, C, S) -> (N, C, S, H, W)
        return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}. Expected 2D or 3D tensor.")

def apply_modulation(x, modulator, mode='concatenate'):
    """
    Apply modulation to features.
    
    Args:
        x (torch.Tensor): Main features tensor.
        modulator (torch.Tensor): Modulator tensor.
        mode (str): Modulation mode ('concatenate', 'multiply', 'add').
    
    Returns:
        torch.Tensor: Modulated tensor.
    """
    if mode == 'concatenate':
        return concat_features(x, modulator)
    elif mode == 'multiply':
        # Ensure modulator has the right shape
        if modulator.dim() == 2:
            modulator = expand_modulator_for_sequence(modulator, x.shape[2])
        return x * modulator
    elif mode == 'add':
        # Ensure modulator has the right shape
        if modulator.dim() == 2:
            modulator = expand_modulator_for_sequence(modulator, x.shape[2])
        return x + modulator
    else:
        raise ValueError(f"Unknown modulation mode: {mode}")
