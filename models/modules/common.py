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
                dim: int = 2,
                causal: bool = False,
                padding_config: Union[int, Sequence[int], None] = None) -> Tuple[int, ...]:
    """
    Calculates padding:
    - For 3D causal: Causal padding for the first dimension (Time/Depth), config padding for spatial dims.
    - Otherwise: All zero padding.

    Args:
        padding_config: Padding configuration from conv_params. Can be:
                       - None: Use zero padding
                       - int: Same padding for all dimensions
                       - sequence: Per-dimension padding (for 3D causal, spatial part is extracted)
    """
    if isinstance(kernel_size, int): kernel_size = (kernel_size,) * dim
    if isinstance(dilation, int): dilation = (dilation,) * dim

    if not all(len(param) == dim for param in [kernel_size, dilation]): # type: ignore
         raise ValueError(f"kernel_size and dilation must have length {dim}")

    # Parse spatial padding from padding_config
    if padding_config is None:
        spatial_padding = (0,) * (dim - 1) if dim > 1 else ()
    elif isinstance(padding_config, int):
        spatial_padding = (padding_config,) * (dim - 1) if dim > 1 else ()
    elif isinstance(padding_config, (list, tuple)):
        if dim == 3 and len(padding_config) == 3:
            # For 3D: extract spatial part [temporal, height, width] -> [height, width]
            spatial_padding = padding_config[1:]
        elif len(padding_config) == dim - 1:
            # Already spatial-only padding
            spatial_padding = padding_config
        elif len(padding_config) == dim:
            # Full padding provided, extract spatial part
            spatial_padding = padding_config[1:] if dim == 3 else padding_config
        else:
            raise ValueError(f"padding_config length {len(padding_config)} doesn't match expected dimensions")
    else:
        spatial_padding = (0,) * (dim - 1) if dim > 1 else ()

    padding_list = []
    # Iterate dims in reverse order (W, H, [D]) for F.pad format
    spatial_idx = 0
    for i in range(dim - 1, -1, -1):
        if dim == 3 and causal and i == 0: # First dim (Depth/Time) of a 3D causal convolution
            # Causal padding: (kernel_size - 1) * dilation on the left/past
            total_padding = dilation[i] * (kernel_size[i] - 1) # type: ignore
            padding_list.extend([total_padding, 0])
        else:
            # Use spatial_padding for spatial dimensions
            if spatial_padding and spatial_idx < len(spatial_padding):
                pad_val = spatial_padding[-(spatial_idx + 1)]  # Reverse order for F.pad
                padding_list.extend([pad_val, pad_val])
                spatial_idx += 1
            else:
                padding_list.extend([0, 0])
    return tuple(padding_list)
