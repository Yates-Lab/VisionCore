#!/usr/bin/env python3
"""
Evaluation script for trained multidataset models.

This script loads a trained model from checkpoints created by run_all_models.sh
and evaluates its performance on a single dataset.

Usage:
    python evaluate_trained_model.py --checkpoint_path checkpoints/x3d_ddp_bs32_ds10_lr1e-4_20250613_160101/epoch=00-val_loss_total=0.1869.ckpt --dataset_idx 0
"""

import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import contextlib

# Import the training module to access the model class
from jake.multidataset_ddp.train_ddp_multidataset import MultiDatasetDDPModule

# Import data loading utilities
from DataYatesV1.utils.data.loading import remove_pixel_norm
from DataYatesV1.utils.data import prepare_data
from DataYatesV1.models.losses import MaskedLoss, PoissonBPSAggregator


class ModelEvaluator:
    """Class for evaluating trained multidataset models."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the evaluator with a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run evaluation on
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {self.checkpoint_path}")
        print(f"Using device: {self.device}")
        
        # Load the model from checkpoint
        self.model = self._load_model()
        self.model.eval()
        
        # Set up loss function (same as training)
        if isinstance(self.model.model.activation, nn.Identity):
            log_input = True
            print("Using log_input=True for PoissonNLLLoss due to Identity activation")
        else:
            log_input = False
            print("Using log_input=False for PoissonNLLLoss due to non-Identity activation")
        
        self.loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=log_input, full=False, reduction='none'))
        
    def _load_model(self) -> MultiDatasetDDPModule:
        """Load the model from checkpoint."""
        try:
            # Load the model using PyTorch Lightning's checkpoint loading
            model = MultiDatasetDDPModule.load_from_checkpoint(
                str(self.checkpoint_path),
                map_location=self.device
            )
            model.to(self.device)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def load_single_dataset(self, dataset_idx: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict]:
        """
        Load a single dataset for evaluation.
        
        Args:
            dataset_idx: Index of the dataset to load (0-based)
            
        Returns:
            Tuple of (train_dataset, val_dataset, dataset_config)
        """
        if dataset_idx >= len(self.model.dataset_configs):
            raise ValueError(f"Dataset index {dataset_idx} out of range. Model has {len(self.model.dataset_configs)} datasets.")
        
        dataset_config = self.model.dataset_configs[dataset_idx].copy()
        dataset_name = self.model.dataset_names[dataset_idx]
        
        print(f"\nLoading dataset {dataset_idx}: {dataset_name}")
        
        # Remove pixel normalization if present
        dataset_config, pixel_norm_removed = remove_pixel_norm(dataset_config)
        
        # Load data with suppressed output
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            train_dset, val_dset, dataset_config = prepare_data(dataset_config)
        
        # Apply Float32View wrapper (same as training)
        class Float32View(torch.utils.data.Dataset):
            def __init__(self, base, pixel_norm_removed=True):
                self.base = base
                self.pixel_norm_removed = pixel_norm_removed
            
            def __len__(self):
                return len(self.base)
            
            def __getitem__(self, idx):
                item = self.base[idx]
                item["stim"] = item["stim"].float()
                if self.pixel_norm_removed:
                    item["stim"] = (item["stim"] - 127) / 255
                item["robs"] = item["robs"].float()
                if "behavior" in item:
                    item["behavior"] = item["behavior"].float()
                return item
        
        train_dset = Float32View(train_dset, pixel_norm_removed)
        val_dset = Float32View(val_dset, pixel_norm_removed)
        
        print(f"✓ Dataset loaded: {len(train_dset)} train, {len(val_dset)} val samples")
        print(f"  Dataset config: {len(dataset_config.get('cids', []))} units")
        
        return train_dset, val_dset, dataset_config
    
    def evaluate_dataset(
        self, 
        dataset: torch.utils.data.Dataset, 
        dataset_idx: int,
        batch_size: int = 64,
        max_batches: Optional[int] = None,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_idx: Index of the dataset (for model routing)
            batch_size: Batch size for evaluation
            max_batches: Maximum number of batches to evaluate (None for all)
            return_predictions: Whether to return predictions and targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating on dataset {dataset_idx}...")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize metrics
        total_loss = 0.0
        bps_aggregator = PoissonBPSAggregator()
        n_batches = 0
        n_samples = 0
        
        # Store predictions if requested
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Forward pass
                rhat = self.model.model(batch['stim'], dataset_idx, batch.get('behavior'))
                
                # Create batch for loss computation
                loss_batch = {
                    'rhat': rhat,
                    'robs': batch['robs']
                }
                if 'dfs' in batch:
                    loss_batch['dfs'] = batch['dfs']
                
                # Compute loss in FP32 for stability
                with torch.autocast(device_type='cuda', enabled=False):
                    batch_fp32 = {
                        'rhat': loss_batch['rhat'].float().clamp_min(1e-4),
                        'robs': loss_batch['robs'].float()
                    }
                    if 'dfs' in loss_batch:
                        batch_fp32['dfs'] = loss_batch['dfs'].float()
                    
                    loss = self.loss_fn(batch_fp32)
                
                # Update metrics
                total_loss += loss.item()
                bps_aggregator(loss_batch)
                n_batches += 1
                n_samples += batch['robs'].shape[0]
                
                # Store predictions if requested
                if return_predictions:
                    all_predictions.append(rhat.cpu())
                    all_targets.append(batch['robs'].cpu())
                
                # Print progress
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        # Compute final metrics
        avg_loss = total_loss / n_batches
        bps = bps_aggregator.closure()
        avg_bps = bps.mean().item()
        
        results = {
            'loss': avg_loss,
            'bps': avg_bps,
            'bps_per_unit': bps.cpu().numpy(),
            'n_samples': n_samples,
            'n_batches': n_batches
        }
        
        if return_predictions:
            results['predictions'] = torch.cat(all_predictions, dim=0)
            results['targets'] = torch.cat(all_targets, dim=0)
        
        print(f"✓ Evaluation complete:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Average BPS: {avg_bps:.6f}")
        print(f"  Samples: {n_samples}")
        
        return results

    def compare_train_val_performance(self, dataset_idx: int, batch_size: int = 64, max_batches: int = 100) -> Dict[str, Dict]:
        """
        Compare model performance on training vs validation sets.

        Args:
            dataset_idx: Index of the dataset to evaluate
            batch_size: Batch size for evaluation
            max_batches: Maximum number of batches to evaluate per split

        Returns:
            Dictionary with 'train' and 'val' results
        """
        train_dset, val_dset, dataset_config = self.load_single_dataset(dataset_idx)

        print(f"\n{'='*60}")
        print(f"COMPARING TRAIN/VAL PERFORMANCE - Dataset {dataset_idx}")
        print(f"Dataset: {self.model.dataset_names[dataset_idx]}")
        print(f"{'='*60}")

        # Evaluate on training set
        print("\n--- TRAINING SET ---")
        train_results = self.evaluate_dataset(
            train_dset, dataset_idx, batch_size=batch_size, max_batches=max_batches
        )

        # Evaluate on validation set
        print("\n--- VALIDATION SET ---")
        val_results = self.evaluate_dataset(
            val_dset, dataset_idx, batch_size=batch_size, max_batches=max_batches
        )

        # Print comparison
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<15} {'Train':<12} {'Val':<12} {'Difference':<12}")
        print("-" * 60)
        print(f"{'Loss':<15} {train_results['loss']:<12.6f} {val_results['loss']:<12.6f} {val_results['loss'] - train_results['loss']:<12.6f}")
        print(f"{'BPS':<15} {train_results['bps']:<12.6f} {val_results['bps']:<12.6f} {val_results['bps'] - train_results['bps']:<12.6f}")
        print(f"{'Samples':<15} {train_results['n_samples']:<12} {val_results['n_samples']:<12} {val_results['n_samples'] - train_results['n_samples']:<12}")

        return {
            'train': train_results,
            'val': val_results,
            'dataset_config': dataset_config
        }

    def plot_bps_distribution(self, results: Dict[str, Dict], save_path: Optional[str] = None):
        """
        Plot BPS distribution across units.

        Args:
            results: Results from compare_train_val_performance
            save_path: Optional path to save the plot
        """
        train_bps = results['train']['bps_per_unit']
        val_bps = results['val']['bps_per_unit']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of BPS values
        ax1.hist(train_bps, bins=30, alpha=0.7, label='Train', density=True)
        ax1.hist(val_bps, bins=30, alpha=0.7, label='Val', density=True)
        ax1.set_xlabel('BPS (bits per spike)')
        ax1.set_ylabel('Density')
        ax1.set_title('BPS Distribution Across Units')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot: train vs val BPS
        ax2.scatter(train_bps, val_bps, alpha=0.6, s=20)
        ax2.plot([min(train_bps.min(), val_bps.min()), max(train_bps.max(), val_bps.max())],
                 [min(train_bps.min(), val_bps.min()), max(train_bps.max(), val_bps.max())],
                 'r--', alpha=0.8, label='y=x')
        ax2.set_xlabel('Train BPS')
        ax2.set_ylabel('Val BPS')
        ax2.set_title('Train vs Val BPS per Unit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(train_bps, val_bps)[0, 1]
        ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        info = {
            'checkpoint_path': str(self.checkpoint_path),
            'model_type': type(self.model).__name__,
            'num_datasets': len(self.model.dataset_configs),
            'dataset_names': self.model.dataset_names,
            'hyperparameters': dict(self.model.hparams),
            'device': str(self.device)
        }

        # Get model architecture info
        if hasattr(self.model, 'model'):
            model = self.model.model
            info['activation'] = type(model.activation).__name__

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            info['total_parameters'] = total_params
            info['trainable_parameters'] = trainable_params

        return info


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained multidataset model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--dataset_idx", type=int, default=0,
                       help="Index of dataset to evaluate (0-based)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum number of batches to evaluate (None for all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots to files")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MULTIDATASET MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset index: {args.dataset_idx}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("="*80)

    # Initialize evaluator
    evaluator = ModelEvaluator(args.checkpoint_path, device=args.device)

    # Print model info
    model_info = evaluator.get_model_info()
    print("\nMODEL INFORMATION:")
    print("-" * 40)
    for key, value in model_info.items():
        if key == 'hyperparameters':
            print(f"{key}:")
            for hkey, hvalue in value.items():
                print(f"  {hkey}: {hvalue}")
        elif key == 'dataset_names':
            print(f"{key}: {len(value)} datasets")
            for i, name in enumerate(value):
                print(f"  {i}: {name}")
        else:
            print(f"{key}: {value}")

    # Evaluate the specified dataset
    try:
        results = evaluator.compare_train_val_performance(
            args.dataset_idx,
            batch_size=args.batch_size,
            max_batches=args.max_batches
        )

        # Plot results
        if args.save_plots:
            plot_path = output_dir / f"bps_distribution_dataset_{args.dataset_idx}.png"
            evaluator.plot_bps_distribution(results, save_path=str(plot_path))
        else:
            evaluator.plot_bps_distribution(results)

        # Save results to file
        results_file = output_dir / f"evaluation_results_dataset_{args.dataset_idx}.npz"
        np.savez(
            results_file,
            train_loss=results['train']['loss'],
            train_bps=results['train']['bps'],
            train_bps_per_unit=results['train']['bps_per_unit'],
            val_loss=results['val']['loss'],
            val_bps=results['val']['bps'],
            val_bps_per_unit=results['val']['bps_per_unit'],
            dataset_idx=args.dataset_idx,
            checkpoint_path=args.checkpoint_path
        )
        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✓ Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
