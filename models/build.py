"""
Model factory for DataYatesV1.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from DataYatesV1.utils.general import ensure_tensor

from .factory import (
    create_frontend, create_convnet, create_recurrent,
    create_modulator, create_readout
)
from .modules.models import ModularV1Model, MultiDatasetV1Model

__all__ = ['build_model', 'initialize_model_components', 'get_name_from_config']

# Type aliases for clarity
ConfigDict = Dict[str, Any]

def build_model(config: ConfigDict, dataset_configs: List[ConfigDict] = None) -> nn.Module:
    """
    Build a model based on the provided configuration.

    This is the main entry point for creating models. It accepts a configuration
    dictionary that specifies the model architecture and parameters.

    Args:
        config: Dictionary containing model configuration
            - model_type: Type of model to build ('v1' or 'v1multi')
            - frontend_type: Type of frontend ('da', 'conv', 'adapter', 'none')
            - convnet_type: Type of convnet ('densenet', 'conv', 'none')
            - recurrent_type: Type of recurrent layer ('convlstm', 'convgru', 'none')
            - modulator_type: Type of modulator ('lstm', 'linear', 'none')
            - readout_type: Type of readout ('gaussian', 'linear')
            - frontend_params: Parameters for frontend
            - convnet_params: Parameters for convnet
            - recurrent_params: Parameters for recurrent layer
            - modulator_params: Parameters for modulator
            - readout_params: Parameters for readout
            - output_activation: Activation function to apply to output
            - init_rates: Initial firing rates for readout bias initialization
        dataset_configs: List of dataset configurations (required for v1multi)

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

    for name, module in model.named_modules():
        # DenseNet initialization
        if 'densenet' in name.lower() or 'convnet' in name.lower():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
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
        
