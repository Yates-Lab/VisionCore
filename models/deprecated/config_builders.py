"""
Configuration builder functions for DataYatesV1 models.

This module provides functions for building configuration dictionaries for
different model components. These functions convert simple parameter dictionaries
into the more complex nested structure expected by the modular system.
"""

from typing import Dict, Any, Optional, Union, Tuple, List

def build_convblock_config(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]] = (3, 3, 3),
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 1,
    dim: int = 3,
    conv_type: str = 'standard',
    norm_type: str = 'batch',
    act_type: str = 'relu',
    use_layernorm: bool = False,
    causal: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a configuration dictionary for a ConvBlock.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolution
        stride: Stride for convolution
        padding: Padding for convolution
        dim: Dimensionality of convolution (2 or 3)
        conv_type: Type of convolution ('standard', 'stacked2d', etc.)
        norm_type: Type of normalization ('batch', 'layer', etc.)
        act_type: Type of activation ('relu', 'leakyrelu', etc.)
        use_layernorm: Whether to use layer normalization (overrides norm_type)
        causal: Whether to use causal padding for 3D convolutions
        **kwargs: Additional parameters to include in the configuration

    Returns:
        Dictionary containing the configuration for a ConvBlock
    """
    # Override norm_type if use_layernorm is specified
    if use_layernorm:
        norm_type = 'layer'

    # Build the configuration dictionary
    config = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'dim': dim,
        'conv_params': {
            'type': conv_type,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        },
        'norm_type': norm_type,
        'act_type': act_type,
        'causal': causal
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        if key == 'conv_params' and isinstance(value, dict):
            # Merge nested conv_params dictionary
            config['conv_params'].update(value)
        elif key == 'norm_params' and isinstance(value, dict):
            # Add norm_params dictionary
            config['norm_params'] = value
        elif key == 'act_params' and isinstance(value, dict):
            # Add act_params dictionary
            config['act_params'] = value
        elif key == 'pool_params' and isinstance(value, dict):
            # Add pool_params dictionary
            config['pool_params'] = value
        else:
            # Regular parameter
            config[key] = value

    return config

def build_densenet_config(
    initial_channels: int,
    growth_rate: int = 4,
    num_blocks: int = 3,
    kernel_size: Union[int, Tuple[int, ...]] = (3, 3, 3),
    dim: int = 3,
    use_layernorm: bool = False,
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a configuration dictionary for a DenseNet.

    Args:
        initial_channels: Number of input channels
        growth_rate: Growth rate for DenseNet
        num_blocks: Number of dense blocks
        kernel_size: Kernel size for convolutions
        dim: Dimensionality of convolution (2 or 3)
        use_layernorm: Whether to use layer normalization
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters to include in the configuration

    Returns:
        Dictionary containing the configuration for a DenseNet
    """
    # Build the configuration dictionary
    config = {
        'model_type': 'densenet',
        'dim': dim,
        'initial_channels': initial_channels,
        'growth_rate': growth_rate,
        'num_blocks': num_blocks,
        'checkpointing': use_checkpointing,
        'block_config': {
            'conv_params': {
                'type': 'standard',
                'kernel_size': kernel_size,
                'padding': 1
            },
            'norm_type': 'layer' if use_layernorm else 'batch',
            'act_type': 'relu'
        }
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        if key == 'block_config' and isinstance(value, dict):
            # Merge nested block_config dictionary
            config['block_config'].update(value)
        else:
            # Regular parameter
            config[key] = value

    return config

def build_resnet_config(
    in_channels: int,
    out_channels: int = 32,
    kernel_size: Union[int, Tuple[int, ...]] = (3, 3, 3),
    stride: Union[int, Tuple[int, ...]] = 1,
    dim: int = 3,
    use_layernorm: bool = False,
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a configuration dictionary for a ResNet.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolutions
        stride: Stride for convolutions
        dim: Dimensionality of convolution (2 or 3)
        use_layernorm: Whether to use layer normalization
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters to include in the configuration

    Returns:
        Dictionary containing the configuration for a ResNet
    """
    # Determine normalization type
    norm_type = 'layer' if use_layernorm else 'batch'

    # Build the configuration dictionary
    config = {
        'model_type': 'resnet',
        'dim': dim,
        'initial_channels': in_channels,
        'base_channels': out_channels,
        'checkpointing': use_checkpointing,
        'resnet_shortcut_norm_type': norm_type,
        'resnet_post_add_activation': 'none',
        'stem_config': {
            'out_channels': out_channels,
            'conv_params': {
                'type': 'standard',
                'kernel_size': 3,
                'stride': 1,
                'padding': 1
            },
            'norm_type': norm_type,
            'act_type': 'relu'
        },
        'layer_configs': [
            {
                'channel_multiplier': 1,
                'conv_params': {
                    'type': 'standard',
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': 1
                },
                'norm_type': norm_type,
                'act_type': 'relu'
            }
        ]
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        if key == 'stem_config' and isinstance(value, dict):
            # Merge nested stem_config dictionary
            config['stem_config'].update(value)
        elif key == 'layer_configs' and isinstance(value, list):
            # Replace layer_configs list
            config['layer_configs'] = value
        else:
            # Regular parameter
            config[key] = value

    return config

def build_model_config(
    model_type: str = 'v1',
    frontend_type: str = 'da',
    convnet_type: str = 'densenet',
    recurrent_type: str = 'none',
    modulator_type: str = 'none',
    readout_type: str = 'gaussian',
    frontend_params: Optional[Dict[str, Any]] = None,
    convnet_params: Optional[Dict[str, Any]] = None,
    recurrent_params: Optional[Dict[str, Any]] = None,
    modulator_params: Optional[Dict[str, Any]] = None,
    readout_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a complete model configuration dictionary.

    Args:
        model_type: Type of model to build ('v1')
        frontend_type: Type of frontend ('da', 'conv', 'temporal_basis', 'adapter', 'none')
        convnet_type: Type of convnet ('densenet', 'conv', 'none')
        recurrent_type: Type of recurrent layer ('convlstm', 'convgru', 'none')
        modulator_type: Type of modulator ('lstm', 'linear', 'none')
        readout_type: Type of readout ('gaussian', 'linear')
        frontend_params: Parameters for frontend
        convnet_params: Parameters for convnet
        recurrent_params: Parameters for recurrent layer
        modulator_params: Parameters for modulator
        readout_params: Parameters for readout
        **kwargs: Additional parameters to include in the configuration

    Returns:
        Dictionary containing the complete model configuration
    """
    # Initialize parameter dictionaries if not provided
    frontend_params = frontend_params or {}
    convnet_params = convnet_params or {}
    recurrent_params = recurrent_params or {}
    modulator_params = modulator_params or {}
    readout_params = readout_params or {}

    # Build the configuration dictionary
    config = {
        'model_type': model_type,
        'frontend_type': frontend_type,
        'convnet_type': convnet_type,
        'recurrent_type': recurrent_type,
        'modulator_type': modulator_type,
        'readout_type': readout_type,
        'frontend_params': frontend_params,
        'convnet_params': convnet_params,
        'recurrent_params': recurrent_params,
        'modulator_params': modulator_params,
        'readout_params': readout_params
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        config[key] = value

    return config


# New unified config builders for the streamlined convnet approach

def build_unified_convnet_config(
    convnet_type: str,
    initial_channels: int,
    channels: List[int],
    dim: int = 3,
    conv_type: str = 'standard',
    kernel_size: Union[int, Tuple[int, ...]] = (3, 3, 3),
    padding: Union[int, Tuple[int, ...]] = 1,
    stride: Union[int, Tuple[int, ...]] = 1,
    norm_type: str = 'batch',
    act_type: str = 'relu',
    pool_params: Optional[Dict[str, Any]] = None,
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a unified configuration dictionary for any convnet type (DenseNet, ResNet, VanillaCNN, X3DNet).

    Args:
        convnet_type: Type of convnet ('densenet', 'resnet', 'vanilla', 'x3d')
        initial_channels: Number of input channels
        channels: List of output channels for each layer
        dim: Dimensionality of convolution (2 or 3)
        conv_type: Type of convolution ('standard', 'depthwise', etc.)
        kernel_size: Kernel size for convolutions
        padding: Padding for convolutions
        stride: Stride for convolutions
        norm_type: Type of normalization ('batch', 'layer', 'rms', 'grn', 'none')
        act_type: Type of activation ('relu', 'gelu', 'mish', 'silu', etc.)
        pool_params: Pooling parameters (None for no pooling)
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters to include in the configuration

    Returns:
        Dictionary containing the unified convnet configuration
    """
    convnet_type = convnet_type.lower()

    # Handle X3DNet specially since it has a different structure
    if convnet_type in ['x3d', 'x3dnet']:
        return build_x3dnet_config(
            initial_channels=initial_channels,
            channels=channels,
            dim=dim,
            norm_type=norm_type,
            act_type=act_type,
            use_checkpointing=use_checkpointing,
            **kwargs
        )

    # Build the base configuration for other convnet types
    config = {
        'model_type': convnet_type,
        'dim': dim,
        'initial_channels': initial_channels,
        'channels': channels,
        'checkpointing': use_checkpointing,
        'block_config': {
            'conv_params': {
                'type': conv_type,
                'kernel_size': kernel_size,
                'padding': padding,
                'stride': stride
            },
            'norm_type': norm_type,
            'act_type': act_type,
            'pool_params': pool_params or {}
        }
    }

    # Add ResNet-specific parameters if needed
    if convnet_type == 'resnet':
        config.update({
            'resnet_shortcut_norm_type': norm_type,
            'resnet_post_add_activation': act_type,
            'resnet_shortcut_projection': True
        })

    # Add any additional parameters
    for key, value in kwargs.items():
        if key == 'block_config' and isinstance(value, dict):
            # Merge nested block_config dictionary
            config['block_config'].update(value)
        else:
            # Regular parameter
            config[key] = value

    return config


def build_densenet_unified_config(
    initial_channels: int,
    channels: List[int],
    dim: int = 3,
    conv_type: str = 'depthwise',
    kernel_size: Union[int, Tuple[int, ...]] = (3, 5, 5),
    padding: Union[int, Tuple[int, ...]] = (1, 2, 2),
    norm_type: str = 'rms',
    act_type: str = 'mish',
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a DenseNet configuration using the new unified format.

    Args:
        initial_channels: Number of input channels
        channels: List of output channels for each dense block
        dim: Dimensionality of convolution (2 or 3)
        conv_type: Type of convolution ('depthwise' recommended for DenseNet)
        kernel_size: Kernel size for convolutions
        padding: Padding for convolutions
        norm_type: Type of normalization ('rms' recommended for DenseNet)
        act_type: Type of activation ('mish' recommended for DenseNet)
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the DenseNet configuration
    """
    return build_unified_convnet_config(
        convnet_type='densenet',
        initial_channels=initial_channels,
        channels=channels,
        dim=dim,
        conv_type=conv_type,
        kernel_size=kernel_size,
        padding=padding,
        norm_type=norm_type,
        act_type=act_type,
        use_checkpointing=use_checkpointing,
        **kwargs
    )


def build_resnet_unified_config(
    initial_channels: int,
    channels: List[int],
    dim: int = 3,
    conv_type: str = 'standard',
    kernel_size: Union[int, Tuple[int, ...]] = (3, 3, 3),
    padding: Union[int, Tuple[int, ...]] = (1, 1, 1),
    norm_type: str = 'batch',
    act_type: str = 'relu',
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a ResNet configuration using the new unified format.

    Args:
        initial_channels: Number of input channels
        channels: List of output channels for each residual block
        dim: Dimensionality of convolution (2 or 3)
        conv_type: Type of convolution ('standard' recommended for ResNet)
        kernel_size: Kernel size for convolutions
        padding: Padding for convolutions
        norm_type: Type of normalization ('batch' recommended for ResNet)
        act_type: Type of activation ('relu' recommended for ResNet)
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the ResNet configuration
    """
    return build_unified_convnet_config(
        convnet_type='resnet',
        initial_channels=initial_channels,
        channels=channels,
        dim=dim,
        conv_type=conv_type,
        kernel_size=kernel_size,
        padding=padding,
        norm_type=norm_type,
        act_type=act_type,
        use_checkpointing=use_checkpointing,
        **kwargs
    )


def build_vanilla_cnn_unified_config(
    initial_channels: int,
    channels: List[int],
    dim: int = 3,
    conv_type: str = 'standard',
    kernel_size: Union[int, Tuple[int, ...]] = (3, 3, 3),
    padding: Union[int, Tuple[int, ...]] = (1, 1, 1),
    norm_type: str = 'batch',
    act_type: str = 'relu',
    use_checkpointing: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a VanillaCNN configuration using the new unified format.

    Args:
        initial_channels: Number of input channels
        channels: List of output channels for each convolutional layer
        dim: Dimensionality of convolution (2 or 3)
        conv_type: Type of convolution ('standard' recommended for VanillaCNN)
        kernel_size: Kernel size for convolutions
        padding: Padding for convolutions
        norm_type: Type of normalization ('batch' recommended for VanillaCNN)
        act_type: Type of activation ('relu' recommended for VanillaCNN)
        use_checkpointing: Whether to use checkpointing (usually False for VanillaCNN)
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the VanillaCNN configuration
    """
    return build_unified_convnet_config(
        convnet_type='vanilla',
        initial_channels=initial_channels,
        channels=channels,
        dim=dim,
        conv_type=conv_type,
        kernel_size=kernel_size,
        padding=padding,
        norm_type=norm_type,
        act_type=act_type,
        use_checkpointing=use_checkpointing,
        **kwargs
    )


def build_x3dnet_config(
    initial_channels: int,
    channels: List[int],
    depth: Optional[List[int]] = None,
    dim: int = 3,
    t_kernel: int = 5,
    s_kernel: int = 3,
    exp_ratio: int = 4,
    norm_type: str = 'grn',
    act_type: str = 'silu',
    stride_stages: Optional[List[int]] = None,
    lite_lk: bool = False,
    lk_every: int = 2,
    dropout: float = 0.0,
    stochastic_depth: float = 0.0,
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build an X3DNet configuration using the new unified format.

    Args:
        initial_channels: Number of input channels
        channels: List of output channels for each stage (width)
        depth: List of number of blocks per stage (defaults to [1] * len(channels))
        dim: Dimensionality of convolution (should be 3 for X3D)
        t_kernel: Temporal kernel size for the first stage
        s_kernel: Spatial kernel size
        exp_ratio: MLP expansion ratio
        norm_type: Type of normalization ('grn' recommended for X3D)
        act_type: Type of activation ('silu' recommended for X3D)
        stride_stages: Temporal strides per stage (defaults to [1, 2, 2, ...])
        lite_lk: Whether to use lite large kernel
        lk_every: Large kernel every N stages
        dropout: Dropout rate after expansion (0.1-0.3 recommended for large models)
        stochastic_depth: Stochastic depth rate (0.1-0.2 recommended for deep models)
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the X3DNet configuration
    """
    if depth is None:
        depth = [1] * len(channels)

    if stride_stages is None:
        stride_stages = [1] + [2] * (len(channels) - 1)

    # Ensure lists are the same length
    if len(depth) != len(channels):
        raise ValueError(f"depth ({len(depth)}) and channels ({len(channels)}) must have same length")

    if len(stride_stages) != len(channels):
        stride_stages = stride_stages[:len(channels)] + [2] * max(0, len(channels) - len(stride_stages))

    config = {
        'model_type': 'x3d',
        'dim': dim,
        'initial_channels': initial_channels,
        'channels': channels,  # New unified format
        'depth': depth,        # Legacy format (for X3D-specific logic)
        'width': channels,     # Legacy format (for X3D-specific logic)
        't_kernel': t_kernel,
        's_kernel': s_kernel,
        'exp_ratio': exp_ratio,
        'norm_type': norm_type,
        'act_type': act_type,
        'stride_stages': stride_stages,
        'lite_lk': lite_lk,
        'lk_every': lk_every,
        'dropout': dropout,
        'stochastic_depth': stochastic_depth,
        'checkpointing': use_checkpointing
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        config[key] = value

    return config


def build_x3dnet_legacy_config(
    initial_channels: int,
    depth: List[int],
    width: List[int],
    dim: int = 3,
    t_kernel: int = 5,
    s_kernel: int = 3,
    exp_ratio: int = 4,
    norm_type: str = 'grn',
    act_type: str = 'silu',
    stride_stages: Optional[List[int]] = None,
    lite_lk: bool = False,
    lk_every: int = 2,
    use_checkpointing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build an X3DNet configuration using the legacy format.

    Args:
        initial_channels: Number of input channels
        depth: List of number of blocks per stage
        width: List of output channels for each stage
        dim: Dimensionality of convolution (should be 3 for X3D)
        t_kernel: Temporal kernel size for the first stage
        s_kernel: Spatial kernel size
        exp_ratio: MLP expansion ratio
        norm_type: Type of normalization ('grn' recommended for X3D)
        act_type: Type of activation ('silu' recommended for X3D)
        stride_stages: Temporal strides per stage (defaults to [1, 2, 2, ...])
        lite_lk: Whether to use lite large kernel
        lk_every: Large kernel every N stages
        use_checkpointing: Whether to use checkpointing
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the X3DNet configuration
    """
    if stride_stages is None:
        stride_stages = [1] + [2] * (len(width) - 1)

    # Ensure lists are the same length
    if len(depth) != len(width):
        raise ValueError(f"depth ({len(depth)}) and width ({len(width)}) must have same length")

    if len(stride_stages) != len(width):
        stride_stages = stride_stages[:len(width)] + [2] * max(0, len(width) - len(stride_stages))

    config = {
        'model_type': 'x3d',
        'dim': dim,
        'initial_channels': initial_channels,
        'depth': depth,
        'width': width,
        't_kernel': t_kernel,
        's_kernel': s_kernel,
        'exp_ratio': exp_ratio,
        'norm_type': norm_type,
        'act_type': act_type,
        'stride_stages': stride_stages,
        'lite_lk': lite_lk,
        'lk_every': lk_every,
        'checkpointing': use_checkpointing
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        config[key] = value

    return config
