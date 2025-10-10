"""
PyTorch Lightning module for multi-dataset neural encoding models.
"""

import os
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import MaskedLoss, PoissonBPSAggregator, MaskedZIPNLLLoss
from training.regularizers import create_regularizers, get_excluded_params_for_weight_decay
from training.schedulers import LinearWarmupCosineAnnealingLR


class MultiDatasetModel(pl.LightningModule):
    """
    Lightning module for training neural encoding models on multiple datasets.
    
    This module:
    - Supports multi-dataset training with separate readouts per dataset
    - Implements curriculum learning with contrast-weighted sampling
    - Supports pretrained vision components with optional freezing
    - Includes regularization (L1, L2, group lasso)
    - Supports predictive coding modulators with auxiliary loss
    - Handles modulator-only models (no vision processing)
    - Computes bits-per-spike (BPS) metrics for validation
    
    Parameters
    ----------
    model_cfg : str
        Path to model configuration YAML file
    cfg_dir : str
        Directory containing dataset configuration files
    lr : float
        Learning rate for readout heads
    wd : float
        Weight decay coefficient
    max_ds : int
        Maximum number of datasets to load
    pretrained_checkpoint : str, optional
        Path to checkpoint with pretrained vision components
    freeze_vision : bool, optional
        Whether to freeze pretrained vision components (default: False)
    compile_model : bool, optional
        Whether to use torch.compile for model (default: False)
        
    Example
    -------
    >>> model = MultiDatasetModel(
    ...     model_cfg='configs/model.yaml',
    ...     cfg_dir='configs/datasets',
    ...     lr=1e-4,
    ...     wd=1e-5,
    ...     max_ds=20,
    ...     pretrained_checkpoint='checkpoints/pretrained.ckpt',
    ...     freeze_vision=True
    ... )
    >>> trainer = pl.Trainer(...)
    >>> trainer.fit(model, datamodule=dm)
    
    Attributes
    ----------
    model : nn.Module
        The neural encoding model
    loss_fn : MaskedLoss
        Poisson NLL loss with masking support
    bps_aggs : list of PoissonBPSAggregator
        BPS aggregators for each dataset
    reg_terms : list
        Regularization terms
    is_modulator_only : bool
        Whether this is a modulator-only model (no vision processing)
    """
    
    def __init__(self, model_cfg: str, cfg_dir: str, lr: float, wd: float,
                 max_ds: int, pretrained_checkpoint: str = None,
                 freeze_vision: bool = False, compile_model: bool = False):
        super().__init__()
        self.save_hyperparameters()

        from models.config_loader import load_dataset_configs, load_config
        from models import build_model

        # Load dataset configurations from parent config
        # cfg_dir should now point to a parent config file (e.g., multi_basic_120_backimage_all.yaml)
        self.cfgs = load_dataset_configs(cfg_dir)

        # Limit to max_ds datasets
        self.cfgs = self.cfgs[:max_ds]

        # Extract dataset names from session names
        self.names = [cfg['session'] for cfg in self.cfgs]
        for c, n in zip(self.cfgs, self.names):
            c["_dataset_name"] = n

        # Load model config and build model
        self.model_config = load_config(model_cfg)
        base_model = build_model(self.model_config, self.cfgs)

        # Detect modulator-only models (no vision processing)
        self.is_modulator_only = (
            self.model_config.get('adapter', {}).get('type') == 'none' and
            self.model_config.get('frontend', {}).get('type') == 'none' and
            self.model_config.get('convnet', {}).get('type') == 'none'
        )

        if self.is_modulator_only:
            print("Modulator-only model detected - will skip stimulus processing")
        else:
            print("Vision model detected - will process stimulus data")

        # Apply torch.compile if requested
        if compile_model:
            try:
                self.model = torch.compile(base_model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"torch.compile failed: {e}")
                print("Falling back to uncompiled model")
                self.model = base_model
        else:
            print("torch.compile disabled")
            self.model = base_model

        # Load pretrained vision components if specified
        if pretrained_checkpoint is not None:
            self._load_pretrained_components(pretrained_checkpoint, freeze_vision)

        # Initialize regularization system
        named_params = list(self.model.named_parameters())
        self.reg_terms = create_regularizers(self.model_config, named_params)

        self.core_lr_scale = lr * self.hparams.get("core_lr_scale", 1.0)
        self.head_lr = lr  # unchanged for dataset heads
        self.log_input = isinstance(self.model.activation, nn.Identity)

        # Initialize loss function based on config
        loss_type = self.model_config.get('loss_type', 'poisson')
        if loss_type == 'zip' or loss_type == 'zero_inflated_poisson':
            self.loss_fn = MaskedZIPNLLLoss(log_input=self.log_input)
            print(f"Using Zero-Inflated Poisson loss (log_input={self.log_input})")
        elif loss_type == 'poisson':
            self.loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=self.log_input, reduction="none"))
            print(f"Using standard Poisson loss (log_input={self.log_input})")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'poisson' or 'zip'")

        self.bps_aggs = [PoissonBPSAggregator() for _ in self.names]
        self.val_losses = []

        # Check for PC modulator and log configuration
        if hasattr(self.model, 'modulator') and self.model.modulator is not None:
            modulator_type = type(self.model.modulator).__name__
            print(f"Detected modulator: {modulator_type}")
            if 'PredictiveCoding' in modulator_type:
                lambda_pred = self.model_config.get('lambda_pred', 0.1)
                print(f"  PC modulator detected with lambda_pred={lambda_pred}")
                print(f"  Auxiliary loss will be computed and logged to wandb as 'aux_loss'")
            else:
                print(f"  Non-PC modulator detected: {modulator_type}")
        else:
            print("  No modulator detected in model")

    def _load_pretrained_components(self, pretrained_checkpoint: str, freeze_vision: bool = False):
        """
        Load pretrained vision components from a checkpoint.
        
        Parameters
        ----------
        pretrained_checkpoint : str
            Path to checkpoint file
        freeze_vision : bool, optional
            Whether to freeze loaded parameters (default: False)
            
        Returns
        -------
        int
            Number of parameters loaded
        """
        print(f"Loading pretrained components from: {pretrained_checkpoint}")

        # Load the pretrained checkpoint
        checkpoint = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint

        # Check for torch.compile key mismatch and fix if needed
        state_dict_keys = list(pretrained_state_dict.keys())
        has_orig_mod_prefix = any(key.startswith('model._orig_mod.') for key in state_dict_keys)

        if has_orig_mod_prefix:
            print("   Detected torch.compile checkpoint - fixing key mismatch...")
            # Fix the state dict keys by removing model._orig_mod. prefix
            fixed_state_dict = {}
            for key, value in pretrained_state_dict.items():
                if key.startswith('model._orig_mod.'):
                    # Remove the model._orig_mod. prefix
                    new_key = key[len('model._orig_mod.'):]
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value
            pretrained_state_dict = fixed_state_dict

        # Filter to vision components (everything except modulator and readouts)
        vision_prefixes = ['model.adapters', 'model.frontend', 'model.convnet']
        vision_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if any(key.startswith(prefix) for prefix in vision_prefixes):
                vision_state_dict[key] = value

        # Load the vision components
        missing_keys, unexpected_keys = self.model.load_state_dict(vision_state_dict, strict=False)

        # Filter missing keys to only show vision components that should have been loaded
        relevant_missing = [k for k in missing_keys if any(k.startswith(prefix) for prefix in vision_prefixes)]

        print(f"✓ Loaded {len(vision_state_dict)} pretrained vision parameters")
        if relevant_missing:
            print(f"⚠ Missing {len(relevant_missing)} expected vision parameters")
            print(f"   Missing keys: {relevant_missing[:5]}...")  # Show first 5 missing keys

        # Show breakdown by component
        component_counts = {}
        for key in vision_state_dict.keys():
            for prefix in vision_prefixes:
                if key.startswith(prefix):
                    component_counts[prefix] = component_counts.get(prefix, 0) + 1
                    break
        print(f"   Breakdown: {component_counts}")

        # Optionally freeze vision components
        if freeze_vision:
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if any(name.startswith(prefix.replace('model.', '')) for prefix in vision_prefixes):
                    param.requires_grad = False
                    frozen_count += 1
            print(f"✓ Froze {frozen_count} vision parameters")

        return len(vision_state_dict)

    def _compute_auxiliary_loss(self):
        """
        Compute auxiliary loss for PC modulator if present.

        Returns
        -------
        torch.Tensor or None
            Auxiliary loss if PC modulator is present and has prediction error
        """
        # Check if model has a PC modulator with prediction error
        if hasattr(self.model, 'modulator') and self.model.modulator is not None:
            if hasattr(self.model.modulator, 'pred_err') and self.model.modulator.pred_err is not None:
                # Get lambda weight from model config or use default
                lambda_pred = getattr(self, 'lambda_pred', 0.1)
                if hasattr(self, 'hparams') and hasattr(self.hparams, 'model_config'):
                    lambda_pred = self.hparams.model_config.get('lambda_pred', lambda_pred)
                elif hasattr(self, 'model_config'):
                    lambda_pred = self.model_config.get('lambda_pred', lambda_pred)

                # Compute L2 loss on prediction error
                pred_err = self.model.modulator.pred_err
                aux_loss = lambda_pred * (pred_err ** 2).mean()
                return aux_loss

        return None

    def forward(self, stim, ds_idx, beh=None):
        """
        Forward pass through the model.

        Parameters
        ----------
        stim : torch.Tensor or None
            Stimulus tensor (None for modulator-only models)
        ds_idx : int
            Dataset index
        beh : torch.Tensor, optional
            Behavior tensor

        Returns
        -------
        torch.Tensor
            Predicted neural responses
        """
        # Check if this is a modulator-only model (no vision processing)
        if self.is_modulator_only:
            y = self.model(stimulus=None, dataset_idx=ds_idx, behavior=beh)
        else:
            y = self.model(stim, ds_idx, beh)
        return torch.clamp(y, min=-20 if self.log_input else 1e-8)

    def _step(self, batch_list, tag: str):
        """
        Process a batch of data (training or validation).

        Parameters
        ----------
        batch_list : list of dict
            List of batches (one per dataset in the batch)
        tag : str
            'train' or 'val'

        Returns
        -------
        torch.Tensor
            Mean loss across all batches
        """
        losses = []
        for b in batch_list:
            # For modulator-only models, skip moving stimulus to device to save memory
            if self.is_modulator_only:
                # Only move non-stimulus data to device
                b = {k: v.to(self.device) if isinstance(v, torch.Tensor) and k != "stim" else v
                     for k, v in b.items()}
            else:
                # Normal models: move all data to device
                b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}

            name = self.names[b["dataset_idx"][0]]

            # If val, no gradients
            with torch.no_grad() if tag == "val" else contextlib.nullcontext():
                # Pass stimulus or None based on model type
                stimulus = None if self.is_modulator_only else b["stim"]
                rhat = self(stimulus, b["dataset_idx"][0], b.get("behavior"))

            batch_loss = {
                'rhat': rhat.float(),
                'robs': b["robs"].float(),
                'dfs': b.get("dfs").float()
            }

            loss = self.loss_fn(batch_loss)

            if torch.isfinite(loss):
                if self.global_rank == 0:  # only log from rank 0 during steps
                    # Get batch size from behavior or robs instead of stim for modulator-only models
                    if self.is_modulator_only:
                        batch_size = b["behavior"].shape[0] if "behavior" in b else b["robs"].shape[0]
                    else:
                        batch_size = b["stim"].shape[0]

                    self.log(f"{tag}_loss/{name}", loss,
                            on_step=(tag == "train"),
                            on_epoch=True,
                            sync_dist=False,
                            batch_size=batch_size)
                losses.append(loss)

                if tag == "val":
                    # check if activation is identity (i.e., training with log_input=True, and we need to torch.exp(rhat))
                    if isinstance(self.model.activation, nn.Identity):
                        batch_loss['rhat'] = torch.exp(batch_loss['rhat'])

                    # update BPS
                    self.bps_aggs[b["dataset_idx"][0]](batch_loss)
                    self.val_losses.append(loss.detach())
            else:
                self.log(f"{tag}_nan_skip/{name}", 1, on_step=True, sync_dist=False)

        if not losses:  # all skipped → return dummy tensor that flows grad
            return torch.zeros([], device=self.device, requires_grad=True)

        return torch.stack(losses).mean()

    def training_step(self, bl, _):
        """
        Training step.

        Parameters
        ----------
        bl : list of dict
            Batch list
        _ : int
            Batch index (unused)

        Returns
        -------
        torch.Tensor
            Total loss (base + auxiliary + regularization)
        """
        # Get base loss from datasets
        base_loss = self._step(bl, "train")

        # Add auxiliary loss for PC modulator if present
        aux_loss = self._compute_auxiliary_loss()
        total_loss = base_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            # Log auxiliary loss to wandb
            if self.global_rank == 0:
                self.log('aux_loss', aux_loss.item(),
                        batch_size=sum(len(b) for b in bl),
                        on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
                # Debug output for first few steps
                if self.global_step < 5:
                    print(f"🎯 Step {self.global_step}: aux_loss={aux_loss.item():.6f}, base_loss={base_loss.item():.6f}")

        # Add regularization penalties
        epoch = self.current_epoch

        for reg in self.reg_terms:
            reg_loss = reg.loss(epoch)
            if reg_loss != 0.0:
                total_loss = total_loss + reg_loss
                # Log individual regularization losses
                if self.global_rank == 0:
                    self.log(f"reg_loss/{reg.name}", reg_loss.item(),
                            on_step=False, on_epoch=True, sync_dist=False)

        return total_loss

    def validation_step(self, bl, _):
        """Validation step."""
        self._step(bl, "val")

    def on_validation_epoch_start(self):
        """Reset BPS aggregators at the start of validation."""
        for agg in self.bps_aggs:
            agg.reset()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch."""
        # 1. average val-loss
        if self.val_losses:
            self.log("val_loss",
                     torch.stack(self.val_losses).mean(),
                     prog_bar=True, sync_dist=True)
        self.val_losses.clear()

        # 2. BPS per-dataset & overall
        per_ds = []
        for name, agg in zip(self.names, self.bps_aggs):

            # a) build local SUM & COUNT tensors on *every* rank
            if len(agg.robs) == 0:  # aggregator is empty
                local_sum = torch.tensor(0.0, device=self.device)
                local_count = torch.tensor(0, device=self.device)
            else:
                bps = agg.closure()  # (units,) - may contain NaN for cells with no samples
                if bps is not None:
                    # Filter out NaN values (cells with no valid samples on this rank)
                    valid_mask = ~torch.isnan(bps)
                    if valid_mask.any():
                        valid_bps = bps[valid_mask].clamp_min(0.0)
                        local_sum = valid_bps.sum().to(self.device)
                        local_count = torch.tensor(valid_bps.numel(), device=self.device)
                    else:
                        # All values are NaN on this rank
                        local_sum = torch.tensor(0.0, device=self.device)
                        local_count = torch.tensor(0, device=self.device)
                else:
                    # No data from aggregator
                    local_sum = torch.tensor(0.0, device=self.device)
                    local_count = torch.tensor(0, device=self.device)

            # b) global reduction (all-reduce)
            global_sum = self.trainer.strategy.reduce(local_sum, reduce_op="sum")
            global_count = self.trainer.strategy.reduce(local_count, reduce_op="sum")

            # Compute BPS mean on all ranks (needed for overall calculation)
            if global_count > 0:
                bps_mean = global_sum / global_count
                per_ds.append(bps_mean)

                # Log per-dataset BPS only on rank-0 to avoid duplication
                if self.global_rank == 0:
                    self.log(f"val_bps/{name}", bps_mean, sync_dist=False)

        # Compute overall BPS and log it (ensure it's available on all ranks)
        if per_ds:
            overall = torch.stack(per_ds).mean()
            overall = torch.clamp_min(overall, 0.0)
        else:
            # Ensure metric is always available, even if no data processed yet
            overall = torch.tensor(0.0, device=self.device)

        # Log with sync_dist=True so all ranks have access to the metric
        self.log("val_bps_overall", overall, prog_bar=True, sync_dist=True, rank_zero_only=False)

        torch.cuda.empty_cache()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns
        -------
        optimizer or dict
            Optimizer (and optionally scheduler configuration)
        """
        # Get parameters that should be excluded from weight decay due to regularization conflicts
        excluded_param_names = set(get_excluded_params_for_weight_decay(self.reg_terms))

        # Split params into "core" vs "head" with different learning rates
        # Also separate params with/without weight decay
        core_params_wd, core_params_no_wd = [], []
        head_params_wd, head_params_no_wd = [], []

        for n, p in self.named_parameters():
            # Determine if this param should have weight decay
            apply_wd = n not in excluded_param_names

            # heuristics – adjust to your model structure if needed
            if any(key in n for key in ["frontend", "convnet", "modulator"]):
                if apply_wd:
                    core_params_wd.append(p)
                else:
                    core_params_no_wd.append(p)
            else:
                if apply_wd:
                    head_params_wd.append(p)
                else:
                    head_params_no_wd.append(p)

        # Create parameter groups with appropriate weight decay settings
        param_groups = []
        if core_params_wd:
            param_groups.append({"params": core_params_wd, "lr": self.core_lr_scale, "weight_decay": self.hparams.wd})
        if core_params_no_wd:
            param_groups.append({"params": core_params_no_wd, "lr": self.core_lr_scale, "weight_decay": 0.0})
        if head_params_wd:
            param_groups.append({"params": head_params_wd, "lr": self.head_lr, "weight_decay": self.hparams.wd})
        if head_params_no_wd:
            param_groups.append({"params": head_params_no_wd, "lr": self.head_lr, "weight_decay": 0.0})

        optim = torch.optim.AdamW(param_groups)

        # Log regularization info
        if excluded_param_names and self.global_rank == 0:
            print(f"[regularization] Excluded {len(excluded_param_names)} parameters from weight decay: {excluded_param_names}")

        # Learning rate scheduler
        sched_type = self.hparams.get("lr_scheduler", "none")
        if sched_type == "none":
            return optim

        if sched_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, step_size=30, gamma=0.5)
        elif sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="min", factor=0.5, patience=3, verbose=True)
        elif sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.trainer.max_epochs)
        elif sched_type == "cosine_warmup":
            # Use a simple linear warmup followed by cosine annealing
            warmup_epochs = self.hparams.get("warmup_epochs", 5)
            scheduler = LinearWarmupCosineAnnealingLR(
                optim,
                warmup_epochs=warmup_epochs,
                max_epochs=self.trainer.max_epochs,
                warmup_start_lr=0.0,
                eta_min=0.0
            )
        else:
            raise ValueError(f"Unknown scheduler {sched_type}")

        # Lightning expects a dict if the scheduler needs a monitored metric
        if sched_type == "plateau":
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return [optim], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override optimizer step to apply proximal updates after gradient step.

        Parameters
        ----------
        epoch : int
            Current epoch
        batch_idx : int
            Current batch index
        optimizer : torch.optim.Optimizer
            Optimizer
        optimizer_closure : callable
            Closure that computes loss

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Standard optimizer step
        loss = optimizer_closure()
        optimizer.step()
        optimizer.zero_grad()

        # Apply proximal updates for regularization
        for group in optimizer.param_groups:
            lr = group["lr"]
            for reg in self.reg_terms:
                reg.prox(epoch, lr)

        return loss

