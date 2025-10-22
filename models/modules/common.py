"""
Common neural network modules for DataYatesV1.

This module contains common components used across different model architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Sequence

# list which modules/functions are available to import
__all__ = ['Noise', 'SplitRelu', 'chomp', 'get_padding']

class SplitRelu(nn.Module):
    """
    SplitRelu activation. Output channels are doubled along split_dim.
    
    Parameters:
    -----------
    split_dim : int
        Dimension to split along (default: 1, which is the channel dimension for NCHW format)
    trainable_gain : bool
        Whether to make the gain parameter trainable (default: False)
    """
    def __init__(self, split_dim=1, trainable_gain=False):
        super().__init__()
        self.split_dim = split_dim
        self.ongain = nn.Parameter(torch.ones(1), requires_grad=trainable_gain)

    def forward(self, x):
        # Output channels are doubled along the channel dimension (dim=1 for NCHW, NCSHW)
        return torch.cat([F.relu(x) * self.ongain.abs(), F.relu(-x)], dim=self.split_dim)


class Noise(nn.Module):
    '''
    Add Gaussian noise to the input tensor. This acts as a regularizer during training.

    Parameters:
    -----------
    sigma : float
        Standard deviation of the Gaussian noise.
    '''
    def __init__(self, sigma=0.1) -> None:
        super().__init__()
        self.sigma = sigma
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.sigma
        return x
    
def chomp(tensor: torch.Tensor, target_spatial_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Crops spatial dimensions (H, W) of a 4D (NCHW) or 5D (NCDHW) tensor
    to match the target_spatial_shape (H_target, W_target).
    """
    if tensor.dim() not in [4, 5]:
        raise ValueError(f"Unsupported tensor dim for chomp: {tensor.dim()}. Expects 4 or 5.")
    if len(target_spatial_shape) != 2:
        raise ValueError(f"target_spatial_shape must be a tuple of length 2 (H, W), got {len(target_spatial_shape)}")

    current_H, current_W = tensor.shape[-2:]
    target_H, target_W = target_spatial_shape

    diff_H = current_H - target_H
    diff_W = current_W - target_W

    if diff_H < 0 or diff_W < 0:
        raise ValueError(f"Target shape ({target_H},{target_W}) is larger than tensor shape ({current_H},{current_W}) in spatial dimensions.")

    start_H = diff_H // 2
    end_H = start_H + target_H
    start_W = diff_W // 2
    end_W = start_W + target_W

    if tensor.dim() == 5: # NCDHW
        return tensor[:, :, :, start_H:end_H, start_W:end_W]
    else: # NCHW
        return tensor[:, :, start_H:end_H, start_W:end_W]

def get_padding(kernel_size: Union[int, Sequence[int]],
                dilation: Union[int, Sequence[int]] = 1,
                causal: bool = True,
                padding_config: Union[int, Sequence[int], None] = None) -> Tuple[int, ...]:
    """
    Calculates padding for 3D convolutions (always NCTHW format).

    Rules:
    - Always assumes 3D convolutions with dimensions (T, H, W)
    - Any dimension with kernel_size=1 gets padding=0 (overrides padding_config)
    - For causal mode: applies causal padding on T dimension (unless kernel_size[0]=1)
    - For non-causal: applies symmetric padding from padding_config

    Args:
        kernel_size: Kernel size as int or 3-tuple (T, H, W)
        dilation: Dilation as int or 3-tuple (T, H, W)
        causal: If True, apply causal padding on temporal dimension
        padding_config: Padding configuration. Can be:
                       - None: Use zero padding
                       - int: Same padding for all dimensions
                       - sequence: Per-dimension padding (T, H, W)

    Returns:
        Padding tuple in F.pad format: (W_left, W_right, H_left, H_right, T_left, T_right)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    if len(kernel_size) != 3 or len(dilation) != 3:
        raise ValueError(f"kernel_size and dilation must have length 3, got {len(kernel_size)} and {len(dilation)}")

    # Parse padding_config into 3D padding (T, H, W)
    if padding_config is None:
        config_padding = (0, 0, 0)
    elif isinstance(padding_config, int):
        config_padding = (padding_config, padding_config, padding_config)
    elif isinstance(padding_config, (list, tuple)):
        if len(padding_config) == 3:
            config_padding = tuple(padding_config)
        elif len(padding_config) == 2:
            # Assume spatial-only (H, W), set T=0
            config_padding = (0, padding_config[0], padding_config[1])
        else:
            raise ValueError(f"padding_config must be int, 2-tuple (H,W), or 3-tuple (T,H,W), got length {len(padding_config)}")
    else:
        config_padding = (0, 0, 0)

    # Build padding in F.pad format: (W_left, W_right, H_left, H_right, T_left, T_right)
    padding_list = []

    # Iterate dimensions in reverse order (W, H, T) for F.pad format
    for i in range(2, -1, -1):  # i = 2 (W), 1 (H), 0 (T)
        # Rule: if kernel_size[i] == 1, padding must be 0
        if kernel_size[i] == 1:
            padding_list.extend([0, 0])
        elif causal and i == 0:  # Temporal dimension with causal padding
            # Causal padding: (kernel_size - 1) * dilation on the left/past, 0 on right/future
            total_padding = dilation[i] * (kernel_size[i] - 1)
            padding_list.extend([total_padding, 0])
        else:
            # Symmetric padding from config
            pad_val = config_padding[i]
            padding_list.extend([pad_val, pad_val])

    return tuple(padding_list)
