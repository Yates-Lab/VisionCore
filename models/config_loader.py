"""
Configuration loader for DataYatesV1 models.

This module provides functions for loading and saving model configurations
from YAML files, with validation and schema checking.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

# Type aliases
ConfigDict = Dict[str, Any]

def load_dataset_configs(dataset_config_paths: List[Union[str, Path]], base_dir: Union[str, Path] = None) -> List[ConfigDict]:
    """
    Load dataset configurations from a list of paths.

    Args:
        dataset_config_paths: List of paths to dataset configuration files
        base_dir: Base directory for relative paths

    Returns:
        List of dataset configurations

    Raises:
        FileNotFoundError: If any config file doesn't exist
        ValueError: If any config is invalid
    """
    dataset_configs = []
    for dataset_info in dataset_config_paths:
        if isinstance(dataset_info, str):
            # Simple path string
            dataset_path = dataset_info
            weight = 1.0
        elif isinstance(dataset_info, dict):
            # Dict with path and optional weight
            dataset_path = dataset_info['path']
            weight = dataset_info.get('weight', 1.0)
        else:
            raise ValueError(f"Invalid dataset config format: {dataset_info}")

        # Load dataset config
        dataset_config_path = Path(dataset_path)
        if not dataset_config_path.is_absolute():
            # Make relative paths relative to the base directory
            dataset_config_path = Path(base_dir) / dataset_config_path

        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")

        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        # Add weight to dataset config
        dataset_config['_weight'] = weight
        dataset_config['_config_path'] = str(dataset_config_path)

        # Validate dataset config for multidataset training
        validate_dataset_config_for_multidataset(dataset_config)

        dataset_configs.append(dataset_config)

    return dataset_configs

def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load a model configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the model configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")
            
    # Validate the loaded config
    # validate_config(config)
    
    return config

def save_config(config: ConfigDict, config_path: Union[str, Path], overwrite: bool = False) -> None:
    """
    Save a model configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the YAML file
        overwrite: Whether to overwrite if file exists
        
    Raises:
        FileExistsError: If the file exists and overwrite=False
        yaml.YAMLError: If the config can't be serialized to YAML
    """
    config_path = Path(config_path)
    
    # Check if file exists and overwrite flag
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Config file already exists: {config_path}")
        
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate before saving
    validate_config(config)
    
    with open(config_path, 'w') as f:
        try:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error saving config to {config_path}: {e}")

def validate_config(config: ConfigDict) -> None:
    """
    Validate a model configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Required top-level keys
    required_keys = {'model_type'}
    
    # Check for required keys
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Validate model type
    if config['model_type'] not in {'v1', 'v1multi'}:  # Add more model types as needed
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Component-specific validation
    for component in ['frontend', 'convnet', 'recurrent', 'readout']:
        if component in config:
            validate_component_config(component, config[component])

def validate_component_config(component_name: str, component_config: ConfigDict) -> None:
    """
    Validate a component configuration dictionary.
    
    Args:
        component_name: Name of the component ('frontend', 'convnet', etc.)
        component_config: Configuration dictionary for the component
        
    Raises:
        ValueError: If the component configuration is invalid
    """
    # Required keys for each component
    required_keys = {'type', 'params'}
    
    # Check for required keys
    missing_keys = required_keys - set(component_config.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys for {component_name}: {missing_keys}")
    
    # Validate component types
    valid_types = {
        'frontend': {'da', 'conv', 'temporal_basis', 'adapter', 'none'},
        'convnet': {'densenet', 'conv', 'resnet', 'none'},
        'recurrent': {'convlstm', 'convgru', 'none'},
        'readout': {'gaussian', 'gaussianei', 'linear'}
    }
    
    if component_name in valid_types:
        if component_config['type'] not in valid_types[component_name]:
            raise ValueError(
                f"Invalid {component_name} type: {component_config['type']}. "
                f"Must be one of: {valid_types[component_name]}"
            )


def load_multidataset_config(train_config_path: Union[str, Path]) -> Tuple[ConfigDict, List[ConfigDict]]:
    """
    Load a multidataset training configuration.

    Args:
        train_config_path: Path to the training configuration file

    Returns:
        Tuple of (model_config, list_of_dataset_configs)

    Raises:
        FileNotFoundError: If any config file doesn't exist
        ValueError: If the configuration is invalid
    """
    train_config_path = Path(train_config_path)

    # Load the training config
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # Validate training config structure
    if 'training_type' not in train_config:
        raise ValueError("Missing 'training_type' in training config")
    if train_config['training_type'] != 'v1multi':
        raise ValueError(f"Expected training_type 'v1multi', got '{train_config['training_type']}'")
    if 'model_config' not in train_config:
        raise ValueError("Missing 'model_config' path in training config")
    if 'datasets' not in train_config:
        raise ValueError("Missing 'datasets' list in training config")

    # Load model config
    model_config_path = Path(train_config['model_config'])
    if not model_config_path.is_absolute():
        # Make relative paths relative to the training config directory
        model_config_path = train_config_path.parent / model_config_path

    model_config = load_config(model_config_path)

    # Validate that model config is v1multi type
    if model_config.get('model_type') != 'v1multi':
        raise ValueError(f"Model config must have model_type 'v1multi', got '{model_config.get('model_type')}'")

    dataset_configs = load_dataset_configs(train_config['datasets'], train_config_path.parent)

    return model_config, dataset_configs


def validate_dataset_config_for_multidataset(dataset_config: ConfigDict) -> None:
    """
    Validate that a dataset config is compatible with multidataset training.

    Args:
        dataset_config: Dataset configuration to validate

    Raises:
        ValueError: If the dataset config is incompatible
    """

    # Check that cids exist
    if 'cids' not in dataset_config:
        raise ValueError("Dataset config must include 'cids' for multidataset training")

    # Check that sampling_rate matches if specified
    # (This will be checked later when comparing across datasets)