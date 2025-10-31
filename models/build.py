"""
Model factory for DataYatesV1.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .utils.general import ensure_tensor

from .modules.models import ModularV1Model, MultiDatasetV1Model, MultiDatasetV1ModelSpikeHistory

__all__ = ['build_model', 'initialize_model_components', 'get_name_from_config']

# Type aliases for clarity
ConfigDict = Dict[str, Any]

def build_model(config: ConfigDict, dataset_configs: List[ConfigDict] = None) -> nn.Module:
    """
    Build a model based on the provided configuration.

    This is the main entry point for creating models. It accepts a configuration
    dictionary that specifies the model architecture and parameters.

    Returns:
        nn.Module: Constructed model
    """
    model_type = config.get('model_type', 'v1')

    if model_type == 'v1':
        return ModularV1Model(config)
    elif model_type == 'v1multi':
        if dataset_configs is None:
            raise ValueError("dataset_configs is required for v1multi model type")
        return MultiDatasetV1Model(config, dataset_configs)
    elif model_type == 'v1multi_history':
        if dataset_configs is None:
            raise ValueError("dataset_configs is required for v1multi_history model type")
        return MultiDatasetV1ModelSpikeHistory(config, dataset_configs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def initialize_model_components(model: nn.Module, init_bias=None) -> None:
    """
    Initialize model components with appropriate weight initialization.

    Args:
        model: Model to initialize
        init_bias: Bias initialization values. For multidataset models, should be a list of arrays.
    """

    # Initialize bias
    if init_bias is not None:
        if hasattr(model, 'readouts'):  # MultiDatasetV1Model
            if not isinstance(init_bias, list):
                raise ValueError("init_bias must be a list for multidataset models")
            if len(init_bias) != len(model.readouts):
                raise ValueError(f"init_bias list length ({len(init_bias)}) must match number of readouts ({len(model.readouts)})")

            for i, (readout, bias) in enumerate(zip(model.readouts, init_bias)):
                if hasattr(readout, 'bias') and readout.bias is not None:
                    assert len(bias) == readout.n_units, f'init_bias[{i}] must have the same length as readout {i} n_units'
                    readout.bias.data = ensure_tensor(bias, dtype=readout.bias.dtype, device=readout.bias.device)
        elif hasattr(model, 'readout'):  # ModularV1Model
            if hasattr(model.readout, 'bias') and model.readout.bias is not None:
                assert len(init_bias) == model.readout.n_units, 'init_bias must have the same length as n_units'
                model.readout.bias.data = ensure_tensor(init_bias, dtype=model.readout.bias.dtype, device=model.readout.bias.device)
        else:
            raise ValueError("Model must have a 'readout' or 'readouts' attribute for bias initialization")

    # Zero-init the final normalization layer in each ResNet block for better gradient flow
    # This is a modern ResNet best practice (used in ResNet-v2, etc.)
    from .modules.conv_blocks import ResBlock
    from .modules.norm_act_pool import RMSNorm

    for module in model.modules():
        if isinstance(module, ResBlock):
            # Zero-init the last norm in the main_block
            # This makes the residual branch initially contribute nothing,
            # so the network starts as identity mappings
            if hasattr(module.main_block, 'components') and 'norm' in module.main_block.components:
                norm_layer = module.main_block.components['norm']
                # Zero-init gamma (scale) for RMSNorm, BatchNorm, LayerNorm, etc.
                if hasattr(norm_layer, 'weight') and norm_layer.weight is not None:
                    nn.init.zeros_(norm_layer.weight)
                # Also zero-init gamma for RMSNorm if it has affine parameters
                if isinstance(norm_layer, RMSNorm) and norm_layer.affine:
                    nn.init.zeros_(norm_layer.gamma)

    for name, module in model.named_modules():
        # ConvNet initialization (DenseNet, ResNet, etc.)
        if 'densenet' in name.lower() or 'convnet' in name.lower():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                # Use leaky_relu approximation for SiLU/Swish activations
                # SiLU has an effective gain closer to leaky_relu with small negative slope
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Recurrent initialization
        elif 'convgru' in name.lower() or 'convlstm' in name.lower() or 'recurrent' in name.lower():
            if isinstance(module, nn.Conv2d):
                # Use Xavier/Glorot for recurrent connections
                nn.init.xavier_uniform_(module.weight)


def get_name_from_config(config):
    name = f"{config['model_type']}"
    for key in config.keys():
        try:
            if isinstance(config[key], dict):
                # and hasattr(config[key], 'type'):
                component = config[key]['type']
                name += f"_{component}"      
        except Exception as e:
            print(f"Failed to get name from {key}: {e}")          
    
    return name
        
