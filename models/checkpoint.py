"""
Model checkpointing and loading utilities for DataYatesV1.

This module provides functions for saving, loading, and managing model checkpoints.
It extends PyTorch Lightning's checkpointing capabilities with additional features
for model consistency checking and configuration management.
"""

import os
import yaml
import torch
import lightning as pl
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from .config_loader import load_config, save_config
from .build import build_model
from .lightning import PLCoreVisionModel

# Type aliases
ConfigDict = Dict[str, Any]


def find_best_checkpoint(checkpoint_dir: Union[str, Path], metric: str = 'val_loss', mode: str = 'min') -> Optional[Path]:
    """
    Find the best checkpoint in a directory based on a metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to use for comparison ('val_loss', 'val_bps', etc.)
        mode: 'min' or 'max' depending on whether lower or higher is better
        
    Returns:
        Path to the best checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
        
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoint_files:
        return None
        
    # Load metrics from each checkpoint
    checkpoint_metrics = []
    for ckpt_file in checkpoint_files:
        try:
            # Load just the metadata without loading the full model
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            if 'hyper_parameters' in checkpoint and metric in checkpoint.get('metrics', {}):
                checkpoint_metrics.append((ckpt_file, checkpoint['metrics'][metric]))
        except Exception as e:
            print(f"Error loading checkpoint {ckpt_file}: {e}")
            
    if not checkpoint_metrics:
        # If no metrics found, return the latest checkpoint
        return max(checkpoint_files, key=os.path.getmtime)
        
    # Sort by metric
    if mode == 'min':
        best_checkpoint = min(checkpoint_metrics, key=lambda x: x[1])[0]
    else:
        best_checkpoint = max(checkpoint_metrics, key=lambda x: x[1])[0]
        
    return best_checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[ConfigDict] = None,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None
) -> PLCoreVisionModel:
    """
    Load a model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Optional configuration to use instead of the one in the checkpoint
        strict: Whether to strictly enforce that the keys in checkpoint match the keys in model
        map_location: Device to load the model onto
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If the checkpoint is invalid or incompatible
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # If no config provided, use the one from the checkpoint
    if config is None:
        if 'hyper_parameters' in checkpoint and 'model_config' in checkpoint['hyper_parameters']:
            config = checkpoint['hyper_parameters']['model_config']
        else:
            raise RuntimeError(f"No model configuration found in checkpoint: {checkpoint_path}")
    
    # Create the model
    if 'hyper_parameters' in checkpoint and 'model_class' in checkpoint['hyper_parameters']:
        # Use the model class from the checkpoint if available
        model = PLCoreVisionModel.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            strict=strict,
            map_location=map_location
        )
    else:
        # Create a new model and load the state dict
        model = PLCoreVisionModel(
            model_class=build_model,
            model_config=config
        )
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        
    return model


def test_model_consistency(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    test_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Tuple[bool, float]:
    """
    Test if two models produce the same outputs.
    
    Args:
        model1: First model
        model2: Second model
        test_input: Input tensor or batch dictionary
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        Tuple of (is_consistent, max_difference)
    """
    model1.eval()
    model2.eval()
    
    # Handle different input types
    if isinstance(test_input, torch.Tensor):
        batch = {'stim': test_input}
    else:
        batch = test_input
        
    # Move to the same device as model1
    batch = {k: v.to(next(model1.parameters()).device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    
    # Compare outputs
    with torch.no_grad():
        out1 = model1(batch)['rhat']
        out2 = model2(batch)['rhat']
        
        max_diff = (out1 - out2).abs().max().item()
        is_consistent = torch.allclose(out1, out2, rtol=rtol, atol=atol)
        
    return is_consistent, max_diff


def save_model_with_metadata(
    model: PLCoreVisionModel,
    save_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    save_config: bool = True
) -> Path:
    """
    Save a model with additional metadata.
    
    Args:
        model: Model to save
        save_dir: Directory to save the model in
        metadata: Additional metadata to save with the model
        save_config: Whether to save the model configuration separately
        
    Returns:
        Path to the saved checkpoint
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(save_dir),
        filename='model',
        save_last=True
    )
    
    # Save the model
    checkpoint_path = checkpoint_callback.format_checkpoint_name(model)
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.strategy.connect(model)
    trainer.save_checkpoint(checkpoint_path)
    
    # Add metadata to the checkpoint
    if metadata:
        checkpoint = torch.load(checkpoint_path)
        checkpoint['metadata'] = metadata
        torch.save(checkpoint, checkpoint_path)
    
    # Save the configuration separately if requested
    if save_config and hasattr(model, 'hparams') and hasattr(model.hparams, 'model_config'):
        config_path = save_dir / 'model_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(model.hparams.model_config, f, default_flow_style=False)
    
    return Path(checkpoint_path)
