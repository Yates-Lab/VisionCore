"""
Model management utilities for DataYatesV1.

This module provides functions for managing trained models, including:
- Finding models by configuration
- Comparing models
- Exporting models for deployment
"""

import os
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime

from .config_loader import load_config
from .checkpoint import load_model_from_checkpoint, find_best_checkpoint
from .lightning import PLCoreVisionModel

# Type aliases
ConfigDict = Dict[str, Any]


class ModelRegistry:
    """
    Registry for managing trained models.

    This class provides methods for:
    - Registering trained models
    - Finding models by configuration
    - Loading models
    """

    def __init__(self, registry_dir: Union[str, Path]):
        """
        Initialize the model registry.

        Args:
            registry_dir: Directory to store the registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / 'model_registry.json'
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Load the registry if it exists
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': []}
            self._save_registry()

    def _save_registry(self):
        """Save the registry to disk."""
        # Create a JSON-serializable copy of the registry
        registry_copy = self._make_json_serializable(self.registry)

        with open(self.registry_file, 'w') as f:
            json.dump(registry_copy, f, indent=2)

    def _make_json_serializable(self, obj):
        """
        Convert an object to a JSON-serializable format.

        This handles:
        - NumPy arrays and scalars
        - Lists and dictionaries containing NumPy objects
        - Other basic Python types

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def register_model(
        self,
        model_id: str,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        dataset_config_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Register a trained model.

        Args:
            model_id: Unique identifier for the model
            checkpoint_path: Path to the model checkpoint
            config_path: Path to the model configuration
            metrics: Performance metrics for the model
            metadata: Additional metadata for the model
            dataset_info: Information about the dataset used for training (deprecated, use dataset_config_path)
            dataset_config_path: Path to the dataset configuration file

        Returns:
            Dictionary with the model registration information
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load configuration if provided
        config = None
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            config = load_config(config_path)

        # Handle dataset information - prefer dataset_config_path over dataset_info
        if dataset_config_path is not None:
            dataset_config_path = Path(dataset_config_path)
            if not dataset_config_path.exists():
                raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")
            dataset_info_entry = str(dataset_config_path.absolute())
        elif dataset_info is not None:
            # Backward compatibility: still accept dataset_info but warn
            print("Warning: dataset_info parameter is deprecated. Use dataset_config_path instead.")
            dataset_info_entry = dataset_info
        else:
            dataset_info_entry = {}

        # Create model entry
        model_entry = {
            'model_id': model_id,
            'checkpoint_path': str(checkpoint_path.absolute()),
            'config_path': str(config_path.absolute()) if config_path is not None else None,
            'metrics': metrics or {},
            'metadata': metadata or {},
            'dataset_info': dataset_info_entry,
            'registered_at': datetime.now().isoformat()
        }

        # Add to registry
        self.registry['models'].append(model_entry)
        self._save_registry()

        return model_entry

    def find_models(
        self,
        model_id: Optional[str] = None,
        config_match: Optional[Dict[str, Any]] = None,
        metric_threshold: Optional[Dict[str, Tuple[str, float]]] = None,
        dataset_match: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find models in the registry.

        Args:
            model_id: Filter by model ID (can be a partial match)
            config_match: Filter by configuration parameters
            metric_threshold: Filter by metric thresholds, e.g., {'val_loss': ('max', 0.1)}
            dataset_match: Filter by dataset information, e.g., {'session': 'Allen_2022-04-13'}

        Returns:
            List of matching model entries
        """
        models = self.registry['models']

        # Filter by model ID
        if model_id is not None:
            models = [m for m in models if model_id in m['model_id']]

        # Filter by configuration
        if config_match is not None and len(models) > 0:
            filtered_models = []
            for model in models:
                if model['config_path'] is None:
                    continue

                try:
                    config = load_config(model['config_path'])
                    match = True
                    for key, value in config_match.items():
                        if key not in config or config[key] != value:
                            match = False
                            break
                    if match:
                        filtered_models.append(model)
                except Exception as e:
                    print(f"Error loading config for model {model['model_id']}: {e}")
            models = filtered_models

        # Filter by dataset information
        if dataset_match is not None and len(models) > 0:
            filtered_models = []
            for model in models:
                if 'dataset_info' not in model:
                    continue

                # Handle both old format (dict) and new format (path string)
                dataset_info = model['dataset_info']
                if isinstance(dataset_info, str):
                    # New format: dataset_info is a path to config file
                    try:
                        with open(dataset_info, 'r') as f:
                            dataset_info = yaml.safe_load(f)
                    except Exception as e:
                        print(f"Warning: Could not load dataset config from {dataset_info}: {e}")
                        continue
                elif isinstance(dataset_info, dict):
                    # Old format: dataset_info is the actual config dict
                    pass
                else:
                    # Empty or invalid dataset_info
                    continue

                match = True
                for key, value in dataset_match.items():
                    if key not in dataset_info or dataset_info[key] != value:
                        match = False
                        break

                if match:
                    filtered_models.append(model)
            models = filtered_models

        # Filter by metrics
        if metric_threshold is not None and len(models) > 0:
            filtered_models = []
            for model in models:
                if 'metrics' not in model:
                    continue

                match = True
                for metric, (op, threshold) in metric_threshold.items():
                    if metric not in model['metrics']:
                        match = False
                        break

                    value = model['metrics'][metric]
                    if op == 'min' and value > threshold:
                        match = False
                        break
                    elif op == 'max' and value < threshold:
                        match = False
                        break

                if match:
                    filtered_models.append(model)
            models = filtered_models

        return models

    def load_model(
        self,
        model_id: str,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> PLCoreVisionModel:
        """
        Load a model from the registry.

        Args:
            model_id: ID of the model to load
            map_location: Device to load the model onto

        Returns:
            Loaded model

        Raises:
            ValueError: If the model is not found
        """
        models = self.find_models(model_id=model_id)
        if not models:
            raise ValueError(f"Model not found: {model_id}")

        model_entry = models[0]
        checkpoint_path = model_entry['checkpoint_path']
        config_path = model_entry['config_path']

        # Load configuration if available
        config = None
        if config_path is not None:
            try:
                config = load_config(config_path)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")

        # Load the model
        model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            map_location=map_location
        )

        return model

    def get_best_model(
        self,
        metric: str = 'val_loss',
        mode: str = 'min',
        config_match: Optional[Dict[str, Any]] = None,
        dataset_match: Optional[Dict[str, Any]] = None,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Tuple[PLCoreVisionModel, Dict[str, Any]]:
        """
        Get the best model based on a metric.

        Args:
            metric: Metric to use for comparison
            mode: 'min' or 'max' depending on whether lower or higher is better
            config_match: Filter by configuration parameters
            dataset_match: Filter by dataset information
            map_location: Device to load the model onto

        Returns:
            Tuple of (loaded model, model entry)

        Raises:
            ValueError: If no models are found
        """
        models = self.find_models(config_match=config_match, dataset_match=dataset_match)
        if not models:
            raise ValueError("No models found matching the criteria")

        # Filter models that have the metric
        models = [m for m in models if 'metrics' in m and metric in m['metrics']]
        if not models:
            raise ValueError(f"No models found with metric: {metric}")

        # Sort by metric
        if mode == 'min':
            best_model = min(models, key=lambda m: m['metrics'][metric])
        else:
            best_model = max(models, key=lambda m: m['metrics'][metric])

        # Load the model
        model = self.load_model(best_model['model_id'], map_location=map_location)

        return model, best_model
