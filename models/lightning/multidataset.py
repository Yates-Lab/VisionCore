"""
PyTorch Lightning module for multidataset training.

This module contains the Lightning module for training models on multiple datasets
simultaneously with gradient accumulation across datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from typing import Dict, List, Any, Union

from .core import PLCoreVisionModel
from ..losses import MaskedLoss, PoissonBPSAggregator


class MultiDatasetPLCore(pl.LightningModule):
    """
    PyTorch Lightning module for multidataset training.
    
    This module handles training on multiple datasets simultaneously by:
    - Routing data through appropriate dataset-specific components
    - Accumulating gradients across all datasets in each training step
    - Tracking per-dataset metrics separately
    - Validating on all datasets when validation is triggered
    """
    
    def __init__(self, model_class=None, model_config=None, dataset_configs=None, model=None,
                 optimizer='AdamW', optim_kwargs=None, eval_modules=None, viz_n_epochs=5,
                 verbose=1, accumulate_grad_batches=1, dataset_info=None):
        """
        Initialize the multidataset Lightning module.
        
        Args:
            model_class: The model class to instantiate (used with model_config + dataset_configs)
            model_config: Main model configuration dictionary
            dataset_configs: List of dataset configuration dictionaries
            model: Pre-instantiated model (alternative to model_class + configs)
            optimizer: Name of the optimizer to use
            optim_kwargs: Optimizer parameters
            eval_modules: List of evaluation modules
            viz_n_epochs: Frequency of visualization in epochs
            verbose: Verbosity level
            accumulate_grad_batches: Number of batches to accumulate gradients (per dataset)
            dataset_info: Information about the datasets used for training
        """
        super().__init__()
        
        # Initialize default values
        if eval_modules is None:
            eval_modules = []
        if optim_kwargs is None:
            optim_kwargs = {'lr': 1e-3, 'weight_decay': 1e-4}
        
        # Store dataset information
        self.dataset_configs = dataset_configs
        self.dataset_info = dataset_info
        self.num_datasets = len(dataset_configs) if dataset_configs else 0
        
        # Handle model instantiation
        if model is not None:
            # Use pre-instantiated model
            self.model = model
            # Save hyperparameters, excluding the model
            self.save_hyperparameters(ignore=['model'])
        elif model_class is not None and model_config is not None and dataset_configs is not None:
            # Instantiate model from class and configs
            self.model = model_class(model_config, dataset_configs)
            # Save all hyperparameters
            self.save_hyperparameters()
        else:
            raise ValueError("Either provide a pre-instantiated model or model_class with model_config and dataset_configs")
        
        # Store other parameters
        self.optimizer = optimizer
        self.optim_kwargs = optim_kwargs
        self.accumulate_grad_batches = accumulate_grad_batches
        
        if isinstance(self.model.activation, nn.Identity):
            log_input = True
            print("Using log_input=True for PoissonNLLLoss due to Identity activation")
        else:
            log_input = False

        # Loss and Metrics - separate for each dataset
        self.loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=log_input, full=False, reduction='none'))
        self.verbose = verbose
        
        # Per-dataset BPS aggregators for training
        self.train_bps_aggregators = [PoissonBPSAggregator() for _ in range(self.num_datasets)]
        # Per-dataset BPS aggregators for validation
        self.val_bps_aggregators = [PoissonBPSAggregator() for _ in range(self.num_datasets)]
        
        self.eval_modules = eval_modules
        self.viz_n_epochs = viz_n_epochs

        # Initialize regularization system
        self.model_config = model_config  # Store for regularization
        self.reg_terms = []  # Will be initialized after model parameters are available
    
    def forward(self, batch_dict: Dict[str, Dict[str, torch.Tensor]]):
        """
        Forward pass through the model for all datasets.

        Args:
            batch_dict: Dictionary with dataset names as keys and batch dictionaries as values

        Returns:
            Dict with dataset names as keys and minimal batch dictionaries containing only
            essential data (rhat, robs, behavior, dfs) - stimulus is discarded to save memory
        """
        output_dict = {}

        for dataset_idx, (dataset_name, batch) in enumerate(batch_dict.items()):
            # Extract inputs
            stimulus = batch['stim']
            behavior = batch.get('behavior', None)

            # Forward pass through model for this dataset
            rhat = self.model(stimulus, dataset_idx, behavior)

            # Create minimal output batch - only keep essential data for loss computation
            # Discard large stimulus tensor to save memory
            output_batch = {
                'rhat': rhat,
                'robs': batch['robs']  # Keep targets for loss computation
            }

            # Keep other small tensors if they exist
            if 'behavior' in batch:
                output_batch['behavior'] = batch['behavior']
            if 'dfs' in batch:
                output_batch['dfs'] = batch['dfs']

            output_dict[dataset_name] = output_batch

        return output_dict

    def _compute_auxiliary_loss(self):
        """
        Compute auxiliary loss for PC modulator prediction error.

        Returns:
            torch.Tensor or None: Auxiliary loss if PC modulator is present and has prediction error
        """
        # Check if model has a PC modulator with prediction error
        if hasattr(self.model, 'modulator') and self.model.modulator is not None:
            if hasattr(self.model.modulator, 'pred_err') and self.model.modulator.pred_err is not None:
                # Get lambda weight from model config or use default
                lambda_pred = getattr(self, 'lambda_pred', 0.1)
                if hasattr(self, 'hparams') and hasattr(self.hparams, 'model_config'):
                    lambda_pred = self.hparams.model_config.get('lambda_pred', lambda_pred)

                # Compute L2 loss on prediction error
                pred_err = self.model.modulator.pred_err
                aux_loss = lambda_pred * (pred_err ** 2).mean()
                return aux_loss

        return None

    def training_step(self, batch_dict: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        """
        Training step with gradient accumulation across datasets.
        
        Args:
            batch_dict: Dictionary with dataset names as keys and batch dictionaries as values
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Combined loss across all datasets
        """
        if 'stim' in batch_dict:
            batch_dict = {'dataset_0': batch_dict}
        
        # Forward pass for all datasets
        output_dict = self(batch_dict)
        
        total_loss = 0.0
        dataset_losses = {}
        
        # Compute loss for each dataset and accumulate
        for dataset_idx, (dataset_name, batch) in enumerate(output_dict.items()):
            # Ensure loss computation is in FP32 for stability
            with torch.autocast(device_type='cuda', enabled=False):
                # Cast predictions and targets to FP32 for loss computation
                if 'rhat' in batch:
                    batch_fp32 = {'rhat': batch['rhat'].float(), 'robs': batch['robs'].float()}
                    if 'dfs' in batch:
                        batch_fp32['dfs'] = batch['dfs'].float()
                    loss = self.loss_fn(batch_fp32)
                else:
                    loss = self.loss_fn(batch)

            # Scale loss by number of datasets for proper averaging
            scaled_loss = loss / self.num_datasets
            total_loss += scaled_loss

            # Store unscaled loss for logging
            dataset_losses[dataset_name] = loss.detach()

            # Update BPS aggregator for this dataset
            self.train_bps_aggregators[dataset_idx](batch)

        # Add auxiliary loss for PC modulator if present
        aux_loss = self._compute_auxiliary_loss()
        if aux_loss is not None:
            total_loss += aux_loss
            self.log('aux_loss', aux_loss.item(), batch_size=sum(len(b) for b in batch_dict.values()), prog_bar=False)

        # Log per-dataset losses
        for dataset_name, loss in dataset_losses.items():
            self.log(f'train_loss_{dataset_name}', loss.item(),
                    batch_size=len(batch_dict[dataset_name]), prog_bar=False)
        
        # Log total loss
        self.log('train_loss_total', total_loss.item(), 
                batch_size=sum(len(batch) for batch in batch_dict.values()), prog_bar=True)
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Set up for training epoch."""
        super().on_train_epoch_start()
        for aggregator in self.train_bps_aggregators:
            aggregator.reset()
    
    def on_train_epoch_end(self):
        """End of training epoch."""
        super().on_train_epoch_end()
        
        # Compute and log BPS for each dataset
        total_bps = 0.0
        total_units = 0
        
        for dataset_idx, (aggregator, dataset_config) in enumerate(zip(self.train_bps_aggregators, self.dataset_configs)):
            bps = aggregator.closure()
            dataset_name = f"dataset_{dataset_idx}"
            
            # Log per-dataset BPS
            self.log(f'train_bps_{dataset_name}', bps.mean().item(), prog_bar=False)
            
            # Accumulate for overall BPS
            n_units = len(dataset_config.get('cids', []))
            total_bps += bps.sum().item()
            total_units += n_units
            
            aggregator.reset()
        
        # Log overall BPS
        if total_units > 0:
            overall_bps = total_bps / total_units
            self.log('train_bps_overall', overall_bps, prog_bar=True)
    
    def validation_step(self, batch_dict: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        """
        Validation step.
        
        Args:
            batch_dict: Dictionary with dataset names as keys and batch dictionaries as values
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Combined validation loss
        """
        # Forward pass for all datasets
        output_dict = self(batch_dict)
        
        total_loss = 0.0
        dataset_losses = {}
        
        # Compute loss for each dataset
        for dataset_idx, (dataset_name, batch) in enumerate(output_dict.items()):
            # Ensure loss computation is in FP32 for stability
            with torch.autocast(device_type='cuda', enabled=False):
                # Cast predictions and targets to FP32 for loss computation
                if 'rhat' in batch:
                    batch_fp32 = batch.copy()
                    batch_fp32['rhat'] = batch['rhat'].float().clamp_min(1e-4)
                    batch_fp32['robs'] = batch['robs'].float()
                    if 'dfs' in batch:
                        batch_fp32['dfs'] = batch['dfs'].float()
                    loss = self.loss_fn(batch_fp32)
                else:
                    loss = self.loss_fn(batch)

            # Scale loss by number of datasets for proper averaging
            scaled_loss = loss / self.num_datasets
            total_loss += scaled_loss

            # Store unscaled loss for logging
            dataset_losses[dataset_name] = loss.detach()

            # Update BPS aggregator for this dataset
            self.val_bps_aggregators[dataset_idx](batch)
        
        # Log per-dataset validation losses
        for dataset_name, loss in dataset_losses.items():
            self.log(f'val_loss_{dataset_name}', loss.item(), 
                    batch_size=len(batch_dict[dataset_name]), prog_bar=False)
        
        # Log total validation loss
        self.log('val_loss_total', total_loss.item(), 
                batch_size=sum(len(batch) for batch in batch_dict.values()), prog_bar=True)
        
        return total_loss
    
    def on_validation_epoch_start(self):
        """Set up for validation epoch."""
        super().on_validation_epoch_start()
        for aggregator in self.val_bps_aggregators:
            aggregator.reset()
    
    def on_validation_epoch_end(self):
        """End of validation epoch."""
        super().on_validation_epoch_end()
        
        try:
            # Compute and log BPS for each dataset
            total_bps = 0.0
            total_units = 0
            dataset_bps = []
            
            for dataset_idx, (aggregator, dataset_config) in enumerate(zip(self.val_bps_aggregators, self.dataset_configs)):
                bps = aggregator.closure()
                dataset_name = f"dataset_{dataset_idx}"
                
                # Log per-dataset BPS
                self.log(f'val_bps_{dataset_name}', bps.mean().item(), prog_bar=False)
                
                # Store for visualization
                dataset_bps.append(bps)
                
                # Accumulate for overall BPS
                n_units = len(dataset_config.get('cids', []))
                total_bps += bps.sum().item()
                total_units += n_units
                
                aggregator.reset()
            
            # Log overall BPS
            if total_units > 0:
                overall_bps = total_bps / total_units
                self.log('val_bps_overall', overall_bps, prog_bar=True)
            
            # Visualization
            if not hasattr(self, 'val_epoch'):
                self.val_epoch = 0
            if self.val_epoch % self.viz_n_epochs == 0:
                try:
                    self.log_state(dataset_bps)
                except Exception as e:
                    print(f"Warning: Error in log_state: {e}")
            self.val_epoch += 1
            
        except Exception as e:
            print(f"Warning: Error in validation_epoch_end: {e}")
    
    def log_state(self, dataset_bps=None):
        """
        Log the state of training to wandb.
        
        Args:
            dataset_bps: List of BPS tensors for each dataset
        """
        if hasattr(self, 'trainer'):
            checkpoint_dir = Path(self.trainer.checkpoint_callback.dirpath)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch = self.current_epoch
            save_fig = True
        else:
            save_fig = False
        
        log_dict = {}
        
        if dataset_bps is not None:
            # Create subplot for each dataset
            n_datasets = len(dataset_bps)
            fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4))
            if n_datasets == 1:
                axes = [axes]
            
            for i, (bps, ax) in enumerate(zip(dataset_bps, axes)):
                ax.stem(bps.cpu())
                ax.set_xlabel('Units')
                ax.set_ylabel('Bits per spike')
                ax.set_title(f'Dataset {i} Validation BPS')
            
            plt.tight_layout()
            
            if save_fig:
                fig_path = checkpoint_dir / f'val_bps_multidataset_{epoch}.png'
                fig.savefig(fig_path)
                log_dict['val_bps_multidataset'] = wandb.Image(str(fig_path))
            else:
                log_dict['val_bps_multidataset'] = wandb.Image(fig)
                if self.verbose <= 1:
                    plt.close()
        
        # Run eval modules
        for eval_module in self.eval_modules:
            try:
                module_logs = eval_module.wandb_logs(self)
                log_dict.update(module_logs)
            except Exception as e:
                print(f"Warning: Error in eval_module {eval_module.__class__.__name__}: {e}")
        
        plt.close('all')
        if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log(log_dict)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        if self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), **self.optim_kwargs)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.optim_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        return optimizer
