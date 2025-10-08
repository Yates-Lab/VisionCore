"""
Configuration system for DataYatesV1 models.

This module provides a standardized configuration system for model components,
including validation, default values, and utility functions for building
configurations.
"""

from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import torch
import torch.nn as nn
import warnings

# Type aliases for clarity
ConfigDict = Dict[str, Any]

# Default configurations for each component type
DEFAULT_CONFIGS = {
    # Frontend defaults
    'frontend': {
        'da': {
            'alpha': 1.0,
            'beta': 0.00008,
            'gamma': 0.5,
            'tau_y_ms': 5.0,
            'tau_z_ms': 60.0,
            'n_y': 5.0,
            'n_z': 2.0,
            'filter_length': 200,
            'learnable_params': False
        },
        'temporal_basis': {
            'num_delta_funcs': 0,
            'num_cosine_funcs': 10,
            'history_bins': 32,
            'log_spacing': True,
            'peak_range_ms': (5, 100),
            'bin_size_ms': 1,
            'normalize': True,
            'orthogonalize': False,
            'causal': True
        },
        'conv': {
            'out_channels': 3,
            'kernel_size': (15, 3, 3),
            'stride': 1,
            'dim': 3,
            'causal': True
        },
        'adapter': {
            'init_sigma': 1.0,
            'grid_size': 25,
            'transform': 'scale'
        }
    },

    # ConvNet defaults
    'convnet': {
        'densenet': {
            'growth_rate': 4,
            'num_blocks': 3,
            'kernel_size': (3, 3, 3),
            'dim': 3,
            'norm': 'none',
            'use_checkpointing': True,
            'act': 'relu',
            'pool': 'none'
        },
        'conv': {
            'out_channels': 32,
            'kernel_size': (3, 3, 3),
            'stride': 1,
            'dim': 3,
            'norm': 'none',
            'act': 'relu',
            'pool': 'none',
            'causal': True
        },
        'resnet': {
            'out_channels': 32,
            'kernel_size': (3, 3, 3),
            'stride': 1,
            'dim': 3,
            'norm': 'none',
            'act': 'relu',
            'pool': 'none',
            'use_checkpointing': True
        },
        'shifttcn': {
            'base_channels': 32,
            'num_tsm_blocks': 4,
            'add_long_kernel': True,
            'seq_len': 32,
            'shift_frac': 0.25,
            'dilations': [1, 2, 4, 8],
            # Spatial configuration
            'stem_spatial_kernel': 7,
            'stem_spatial_stride': 2,
            'stem_temporal_kernel': 3,
            'tsm_spatial_kernel': 3,
            # Regularization
            'dropout': 0.1,
            'stochastic_depth': 0.1
        }
    },

    # Recurrent defaults
    'recurrent': {
        'convlstm': {
            'hidden_dim': 64,
            'kernel_size': 3,
            'dropout': 0.5,
            'bias': True,
            'forget_bias_init': 1.0,
            'start_time': 10,
            'fast_phase': False,
            'with_modulator': False,
            'modulation_mode': 'concatenate'
        },
        'convgru': {
            'hidden_dim': 64,
            'kernel_size': 3,
            'dropout': 0.5,
            'bias': True,
            'start_time': 10,
            'fast_phase': False,
            'with_modulator': False,
            'modulation_mode': 'concatenate'
        },
        'lstm': {
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.5,
            'bias': True,
            'start_time': 10,
            'bidirectional': False
        },
        'gru': {
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.5,
            'bias': True,
            'start_time': 10,
            'bidirectional': False
        }
    },

    # Modulator defaults
    'modulator': {
        'lstm': {
            'n_vars': 2,
            'K': 32,
            'lstm_layers': 1,
            'bidirectional': False,
            'dropout': 0.0,
            'mode': 'concatenate'
        },
        'gru': {
            'n_vars': 2,
            'K': 32,
            'gru_layers': 1,
            'bidirectional': False,
            'dropout': 0.0,
            'mode': 'concatenate'
        },
        'linear': {
            'n_vars': 2,
            'K': 32
        },
        'mlp': {
            'n_vars': 2,
            'K': 32,
            'hidden_layers': [64, 128],
            'activation': 'tanh',
            'dropout': 0.0
        }

    },

    # Readout defaults
    'readout': {
        'gaussian': {
            'n_units': 16,
            'bias': True,
            'initial_std': 5.0
        },
        'gaussian_ei': {
            'n_units': 16,
            'bias': True,
            'initial_std_ex': 3.0,
            'initial_std_inh': 6.0,
            'weight_constraint_fn': None
        },
        'gaussian_sn': {
            'n_units': 16,
            'bias': True,
            'initial_std_ex': 3.0,
            'initial_std_inh': 6.0,
            'weight_constraint_fn': None,
            'initial_beta': 0.1,
            'initial_sigma': 0.05,
            'min_beta': 1e-4,
            'min_sigma': 1e-5
        },
        'linear': {
            'n_units': 16,
            'bias': True
        }
    },

    # Baseline defaults
    'baseline': {
        'enabled': False,
        'activation': 'relu',
        'init_value': 0.001
    }
}

# Required parameters for each component type
REQUIRED_PARAMS = {
    'frontend': {
        'da': ['sampling_rate', 'height', 'width'],
        'temporal_basis': ['sampling_rate'],
        'conv': ['in_channels'],
        'adapter': []
    },
    'convnet': {
        'densenet': ['initial_channels'],
        'conv': ['in_channels'],
        'resnet': ['in_channels']
    },
    'recurrent': {
        'convlstm': ['input_dim'],
        'convgru': ['input_dim'],
        'lstm': ['input_dim'],
        'gru': ['input_dim']
    },
    'modulator': {
        'lstm': [],
        'linear': []
    },
    'readout': {
        'gaussian': ['in_channels'],
        'gaussian_ei': ['in_channels'],
        'gaussian_sn': ['in_channels'],
        'linear': ['in_channels']
    }
}

def validate_component_config(component_type: str, component_name: str, config: ConfigDict) -> None:
    """
    Validate that a component configuration has all required parameters.

    Args:
        component_type: Type of component ('frontend', 'convnet', etc.)
        component_name: Name of the component ('da', 'densenet', etc.)
        config: Configuration dictionary for the component

    Raises:
        ValueError: If a required parameter is missing
    """
    if component_type not in REQUIRED_PARAMS:
        warnings.warn(f"Unknown component type: {component_type}")
        return

    if component_name not in REQUIRED_PARAMS[component_type]:
        warnings.warn(f"Unknown component name: {component_name} for type {component_type}")
        return

    required = REQUIRED_PARAMS[component_type][component_name]
    missing = [param for param in required if param not in config]

    if missing:
        raise ValueError(f"Missing required parameters for {component_type}.{component_name}: {missing}")

def get_component_defaults(component_type: str, component_name: str) -> ConfigDict:
    """
    Get default configuration for a component.

    Args:
        component_type: Type of component ('frontend', 'convnet', etc.)
        component_name: Name of the component ('da', 'densenet', etc.)

    Returns:
        Default configuration dictionary for the component
    """
    if component_type not in DEFAULT_CONFIGS:
        warnings.warn(f"No default configuration for component type: {component_type}")
        return {}

    if component_name not in DEFAULT_CONFIGS[component_type]:
        warnings.warn(f"No default configuration for component: {component_name} of type {component_type}")
        return {}

    return DEFAULT_CONFIGS[component_type][component_name].copy()

def merge_configs(default_config: ConfigDict, user_config: ConfigDict) -> ConfigDict:
    """
    Merge default configuration with user-provided configuration.

    Args:
        default_config: Default configuration dictionary
        user_config: User-provided configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = default_config.copy()

    for key, value in user_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add the value
            result[key] = value

    return result

def build_component_config(component_type: str, component_name: str, **kwargs) -> ConfigDict:
    """
    Build a configuration dictionary for a component.

    Args:
        component_type: Type of component ('frontend', 'convnet', etc.)
        component_name: Name of the component ('da', 'densenet', etc.)
        **kwargs: Additional parameters to include in the configuration

    Returns:
        Configuration dictionary for the component
    """
    # Get default configuration
    config = get_component_defaults(component_type, component_name)

    # Merge with user-provided parameters
    config = merge_configs(config, kwargs)

    # Validate the configuration
    validate_component_config(component_type, component_name, config)

    return config

def build_model_config(
    model_type: str = 'v1',
    frontend_type: str = 'da',
    convnet_type: str = 'densenet',
    recurrent_type: str = 'none',
    modulator_type: str = 'none',
    readout_type: str = 'gaussian',
    transition_type: str = 'concat',
    frontend_params: Optional[Dict[str, Any]] = None,
    convnet_params: Optional[Dict[str, Any]] = None,
    recurrent_params: Optional[Dict[str, Any]] = None,
    modulator_params: Optional[Dict[str, Any]] = None,
    readout_params: Optional[Dict[str, Any]] = None,
    transition_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ConfigDict:
    """
    Build a complete model configuration dictionary.

    Args:
        model_type: Type of model to build ('v1')
        frontend_type: Type of frontend ('da', 'conv', 'temporal_basis', 'adapter', 'none')
        convnet_type: Type of convnet ('densenet', 'conv', 'resnet', 'none')
        transition_type: Type of transition ('concat', 'add', 'linear', 'none')
        recurrent_type: Type of recurrent layer ('convlstm', 'convgru', 'lstm', 'gru', 'none')
        modulator_type: Type of modulator ('lstm', 'linear', 'mlp', 'none')
        readout_type: Type of readout ('gaussian', 'gaussian_ei', 'gaussian_sn', 'linear')
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
    transition_params = transition_params or {}

    # Build the configuration dictionary
    config = {
        'model_type': model_type,
        'frontend_type': frontend_type,
        'convnet_type': convnet_type,
        'transition_type': transition_type,
        'recurrent_type': recurrent_type,
        'modulator_type': modulator_type,
        'readout_type': readout_type,
        'frontend_params': frontend_params,
        'convnet_params': convnet_params,
        'recurrent_params': recurrent_params,
        'modulator_params': modulator_params,
        'readout_params': readout_params,
        'transition_params': transition_params
    }

    # Add any additional parameters
    for key, value in kwargs.items():
        config[key] = value

    return config
