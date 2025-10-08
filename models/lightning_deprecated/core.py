"""
PyTorch Lightning modules for DataYatesV1.

This module contains PyTorch Lightning modules for training neural network models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

from ..losses import MaskedLoss, PoissonBPSAggregator

class PLCoreVisionModel(pl.LightningModule):
    """
    PyTorch Lightning module for training vision models.

    This implementation follows the training procedure from the paper:
    - Poisson negative-log likelihood loss
    - Nesterov momentum
    - Learning rate schedule with linear warmup and cosine decay
    - Small batch size with gradient accumulation
    """
    def __init__(self, model_class=None, model_config=None, model=None, optimizer='AdamW',
                 optim_kwargs=None, eval_modules=None, viz_n_epochs=5, verbose=1,
                 accumulate_grad_batches=1, cids=None, dataset_info=None):
        """
        Initialize the Lightning module.

        Parameters:
        -----------
        model_class : type, optional
            The model class to instantiate (used with model_config)
        model_config : dict, optional
            Configuration dictionary for the model
        model : nn.Module, optional
            Pre-instantiated model (alternative to model_class + model_config)
        optimizer : str
            Name of the optimizer to use
        optim_kwargs : dict
            Optimizer parameters
        eval_modules : list
            List of evaluation modules
        viz_n_epochs : int
            Frequency of visualization in epochs
        verbose : int
            Verbosity level
        accumulate_grad_batches : int
            Number of batches to accumulate gradients
        cids : list or None
            List of cell IDs
        dataset_info : dict or None
            Information about the dataset used for training
        """
        super().__init__()

        # Initialize default values
        if eval_modules is None:
            eval_modules = []
        if optim_kwargs is None:
            optim_kwargs = {'lr': 1e-3, 'weight_decay': 1e-4}

        # Store dataset information
        self.dataset_info = dataset_info

        # Handle model instantiation
        if model is not None:
            # Use pre-instantiated model
            self.model = model
            # Save hyperparameters, excluding the model
            self.save_hyperparameters(ignore=['model'])
        elif model_class is not None and model_config is not None:
            # Instantiate model from class and config
            self.model = model_class(model_config)
            # Save all hyperparameters
            self.save_hyperparameters()
        else:
            raise ValueError("Either provide a pre-instantiated model or a model_class with model_config")

        # Store other parameters
        self.optimizer = optimizer
        self.optim_kwargs = optim_kwargs
        self.accumulate_grad_batches = accumulate_grad_batches

        # Loss and Metrics
        if isinstance(self.model.activation, nn.Identity):
            log_input = True
            print("Using log_input=True for PoissonNLLLoss due to Identity activation")
        else:
            log_input = False

        self.cids = cids
        self.loss = MaskedLoss(nn.PoissonNLLLoss(log_input=log_input, full=False, reduction='none'))
        self.verbose = verbose
        self.train_bps_aggregator = PoissonBPSAggregator()
        self.val_bps_aggregator = PoissonBPSAggregator()
        self.eval_modules = eval_modules
        self.viz_n_epochs = viz_n_epochs

    def forward(self, batch):
        """
        Forward pass through the model.

        Parameters:
        -----------
        batch : dict
            Batch dictionary containing input data

        Returns:
        --------
        dict
            Batch dictionary with predictions added
        """
        try:
            
            # Check for NaN values in stim
            if torch.isnan(batch['stim']).any():
                print("Warning: NaN values in stimulus. Replacing with zeros.")
                stim = torch.nan_to_num(batch['stim'].clone(), nan=0.0)
            else:
                stim = batch['stim']

            # Forward pass through the model
            if hasattr(self.model, 'modulator') and self.model.modulator is not None and 'behavior' in batch:
                batch['rhat'] = self.model(stim, batch['behavior'])
            else:
                batch['rhat'] = self.model(stim)

            return batch

        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return the original batch with a dummy rhat if possible
            if 'robs' in batch:
                batch['rhat'] = torch.ones_like(batch['robs'], dtype=torch.float32) * 1e-6
            return batch

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
        --------
        torch.optim.Optimizer
            Configured optimizer
        """
        # Setup optimizer according to the specified type
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                **self.optim_kwargs
            )
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), **self.optim_kwargs)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not recognized")

        return optimizer

    def on_train_epoch_start(self):
        """Set up for training epoch."""
        super().on_train_epoch_start()
        self.train_bps_aggregator.reset()

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

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Parameters:
        -----------
        batch : dict
            Batch dictionary containing input data
        batch_idx : int
            Batch index

        Returns:
        --------
        torch.Tensor
            Loss value
        """
        # Forward pass with safety checks
        batch = self(batch)

        # Ensure loss computation is in FP32 for stability
        with torch.autocast(device_type='cuda', enabled=False):
            # Cast predictions and targets to FP32 for loss computation
            if 'rhat' in batch and batch['rhat'].dtype != torch.float32:
                batch_fp32 = batch.copy()
                batch_fp32['rhat'] = batch['rhat'].float()
                batch_fp32['robs'] = batch['robs'].float()
                if 'dfs' in batch:
                    batch_fp32['dfs'] = batch['dfs'].float()
                loss = self.loss(batch_fp32)
            else:
                loss = self.loss(batch)

            # Add auxiliary loss for PC modulator if present
            aux_loss = self._compute_auxiliary_loss()
            if aux_loss is not None:
                loss = loss + aux_loss
                self.log('aux_loss', aux_loss.item(), batch_size=len(batch), prog_bar=False)

        if torch.isnan(loss).any():
            # graceful exit by simulating a KeyboardInterrupt
            print("Loss is NaN, exiting training.")
            raise KeyboardInterrupt
    
        # Log and return the loss
        self.log('train_loss', loss.item(), batch_size=len(batch), prog_bar=True)
        self.train_bps_aggregator(batch)
        return loss

    def on_train_epoch_end(self):
        """End of training epoch."""
        super().on_train_epoch_end()

        bps = self.train_bps_aggregator.closure()
        self.log('train_bps', bps.mean().item(), prog_bar=True)
        self.train_bps_aggregator.reset()

    def on_train_end(self):
        """End of training."""
        super().on_train_end()
        self.log_state()

    def on_validation_epoch_start(self):
        """Set up for validation epoch."""
        super().on_validation_epoch_start()
        self.val_bps_aggregator.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Parameters:
        -----------
        batch : dict
            Batch dictionary containing input data
        batch_idx : int
            Batch index

        Returns:
        --------
        torch.Tensor
            Loss value
        """
        # Forward pass
        # with torch.no_grad():
        batch = self(batch) # adds rhat to the batch

        # Ensure loss computation is in FP32 for stability
        with torch.autocast(device_type='cuda', enabled=False):
            # Cast predictions and targets to FP32 for loss computation
            if 'rhat' in batch:
                batch_fp32 = batch.copy()
                batch_fp32['rhat'] = batch['rhat'].float()
                batch_fp32['robs'] = batch['robs'].float()
                if 'dfs' in batch:
                    batch_fp32['dfs'] = batch['dfs'].float()
                loss = self.loss(batch_fp32)
            else:
                loss = self.loss(batch)

        # Log validation metrics
        self.log('val_loss', loss.item(), batch_size=len(batch), prog_bar=True)
        self.val_bps_aggregator(batch)
        return loss

    def on_validation_epoch_end(self):
        """
        End of validation epoch.

        This is where all the logging and visualization code goes.
        """
        super().on_validation_epoch_end()
        try:
            bps = self.val_bps_aggregator.closure()
            self.log('val_bps', max(0, bps.mean().item()), prog_bar=True)
            self.val_bps_aggregator.reset()
            if not hasattr(self, 'val_epoch'):
                self.val_epoch = 0
            if self.val_epoch % self.viz_n_epochs == 0:
                try:
                    self.log_state(bps)
                except Exception as e:
                    print(f"Warning: Error in log_state: {e}")
                    # Continue training even if visualization fails
            self.val_epoch += 1
        except Exception as e:
            print(f"Warning: Error in validation_epoch_end: {e}")
            # Continue training even if validation metrics fail

    def log_state(self, val_bps=None):
        """
        Log the state of training to wandb.

        Parameters:
        -----------
        val_bps : torch.Tensor or None
            Validation bits per spike
        """
        if hasattr(self, 'trainer'):
            checkpoint_dir = Path(self.trainer.checkpoint_callback.dirpath)
            # make directory if it doesn't exist
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            epoch = self.current_epoch
            save_fig = True
        else:
            save_fig = False

        log_dict = {}

        if val_bps is not None:
            fig = plt.figure()
            plt.stem(val_bps.cpu())
            plt.xlabel('Units')
            plt.ylabel('Bits per spike')
            plt.title('Validation BPS')
            log_dict['val_bps_units'] = wandb.Image(fig)
            if save_fig:
                fig_path = checkpoint_dir / f'val_bps_units_{epoch}.png'
                fig.savefig(fig_path)
                log_dict['val_bps_units'] = wandb.Image(str(fig_path), caption='Validation BPS')
            else:
                if self.verbose > 1:
                    plt.show()
                else:
                    plt.close()

        for eval_module in self.eval_modules:
            try:
                module_logs = eval_module.wandb_logs(self)
                log_dict.update(module_logs)
            except Exception as e:
                print(f"Warning: Error in eval_module {eval_module.__class__.__name__}: {e}")
                # Continue with other eval modules

        plt.close('all')
        self.logger.experiment.log(log_dict) # log the images to wandb
