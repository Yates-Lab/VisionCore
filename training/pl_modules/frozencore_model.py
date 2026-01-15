"""
PyTorch Lightning module for training new adapters/readouts on frozen core components.

This module loads a pretrained model, freezes the shared components (frontend, convnet,
modulator, recurrent), and rebuilds new adapters and readouts for new dataset configs.
"""

import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import MaskedLoss, PoissonBPSAggregator
from models.factory import create_frontend, create_readout
from training.regularizers import create_regularizers, get_excluded_params_for_weight_decay
from training.schedulers import LinearWarmupCosineAnnealingLR, LinearWarmupCosineAnnealingWarmRestartsLR


class FrozenCoreModel(pl.LightningModule):
    """
    Lightning module for training new adapters/readouts on frozen core components.
    
    This module:
    1. Loads a pretrained model from checkpoint
    2. Freezes core components (frontend, convnet, modulator, recurrent)
    3. Rebuilds new adapters and readouts for new dataset configs
    4. Trains only the new adapters and readouts
    
    Parameters
    ----------
    pretrained_checkpoint : str
        Path to pretrained checkpoint file or checkpoint directory
    model_type : str, optional
        Model type to select from checkpoint directory
    model_index : int, optional
        Model index within model_type (None = best model)
    cfg_dir : str
        Path to NEW dataset configuration file
    lr : float
        Learning rate for adapters and readouts
    wd : float
        Weight decay coefficient
    max_ds : int
        Maximum number of datasets to load
    """
    
    def __init__(
        self,
        pretrained_checkpoint: str,
        cfg_dir: str,
        lr: float,
        wd: float,
        max_ds: int,
        model_type: str = None,
        model_index: int = None,
    ):
        super().__init__()
        
        from models.config_loader import load_dataset_configs, load_config
        from eval.eval_stack_multidataset import load_model
        from eval.eval_stack_utils import scan_checkpoints
        
        self.save_hyperparameters()
        
        # -------------------------------------------------------------------------
        # Step 1: Load the pretrained model
        # -------------------------------------------------------------------------
        pretrained_path = Path(pretrained_checkpoint)
        
        if pretrained_path.is_dir():
            # It's a directory - use model_type to find best checkpoint
            if model_type is None:
                raise ValueError("model_type required when pretrained_checkpoint is a directory")
            
            print(f"Loading pretrained model from directory: {pretrained_path}")
            pretrained_pl_model, model_info = load_model(
                model_type=model_type,
                model_index=model_index,
                checkpoint_dir=str(pretrained_path),
                device='cpu',
                verbose=True
            )
        else:
            # It's a file - load directly
            print(f"Loading pretrained model from checkpoint: {pretrained_path}")
            pretrained_pl_model, model_info = load_model(
                checkpoint_path=str(pretrained_path),
                device='cpu',
                verbose=True
            )
        
        # Get the underlying model and config
        pretrained_model = pretrained_pl_model.model
        self.model_config = pretrained_pl_model.model_config
        
        print(f"Loaded model with config: {self.model_config.get('model_type', 'unknown')}")
        
        # -------------------------------------------------------------------------
        # Step 2: Load NEW dataset configurations
        # -------------------------------------------------------------------------
        self.cfgs = load_dataset_configs(cfg_dir)[:max_ds]
        self.names = [cfg['session'] for cfg in self.cfgs]
        for c, n in zip(self.cfgs, self.names):
            c["_dataset_name"] = n
        
        print(f"New datasets: {self.names}")
        
        # -------------------------------------------------------------------------
        # Step 3: Build new model with frozen core
        # -------------------------------------------------------------------------
        self.model = self._build_frozen_core_model(pretrained_model)
        
        # Count parameters
        self.frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        self.trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Frozen parameters: {self.frozen_count:,}")
        print(f"Trainable parameters: {self.trainable_count:,}")
        
        # -------------------------------------------------------------------------
        # Step 4: Setup training components
        # -------------------------------------------------------------------------
        # Initialize regularization for new components only
        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.reg_terms = create_regularizers(self.model_config, named_params)
        
        self.log_input = isinstance(self.model.activation, nn.Identity)
        self.loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=self.log_input, reduction="none"))
        print(f"Using Poisson loss (log_input={self.log_input})")

        self.bps_aggs = [PoissonBPSAggregator() for _ in self.names]
        self.val_losses = []  # Track validation losses for epoch averaging

    def _build_frozen_core_model(self, pretrained_model):
        """
        Build a new model with frozen core and fresh adapters/readouts.

        This method:
        1. Copies the shared components (frontend, convnet, modulator, recurrent) from pretrained
        2. Freezes those components
        3. Creates new adapters and readouts for the new dataset configs
        """
        from models.modules.models import MultiDatasetV1Model

        # Create a new model with the same config but new dataset configs
        new_model = MultiDatasetV1Model(self.model_config, self.cfgs)

        # Copy shared components from pretrained model
        shared_components = ['frontend', 'convnet', 'modulator', 'recurrent']

        for component_name in shared_components:
            if hasattr(pretrained_model, component_name) and hasattr(new_model, component_name):
                pretrained_component = getattr(pretrained_model, component_name)
                new_component = getattr(new_model, component_name)

                # Skip if either component is None or Identity
                if pretrained_component is None or new_component is None:
                    print(f"  - {component_name} is None (skipping)")
                    continue
                if isinstance(pretrained_component, nn.Identity):
                    print(f"  - {component_name} is Identity (no parameters)")
                    continue

                # Load state dict
                try:
                    new_component.load_state_dict(pretrained_component.state_dict())
                    print(f"  ✓ Loaded {component_name} from pretrained model")
                except Exception as e:
                    print(f"  ⚠ Could not load {component_name}: {e}")

        # Freeze shared components
        frozen_count = 0
        for component_name in shared_components:
            if hasattr(new_model, component_name):
                component = getattr(new_model, component_name)
                # Skip None or Identity components
                if component is None or isinstance(component, nn.Identity):
                    continue
                for param in component.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()

        print(f"  ✓ Froze {frozen_count:,} parameters in shared components")

        # The adapters and readouts are freshly initialized by MultiDatasetV1Model
        # They remain trainable (requires_grad=True by default)

        # Print info about trainable components
        adapter_params = sum(p.numel() for a in new_model.adapters for p in a.parameters())
        readout_params = sum(p.numel() for r in new_model.readouts for p in r.parameters())
        print(f"  New adapters: {len(new_model.adapters)} with {adapter_params:,} parameters")
        print(f"  New readouts: {len(new_model.readouts)} with {readout_params:,} parameters")

        return new_model

    def forward(self, stimulus, dataset_idx: int = 0, behavior=None):
        """Forward pass through the model."""
        return self.model(stimulus, dataset_idx=dataset_idx, behavior=behavior)

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
            # Move all data to device
            b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}

            name = self.names[b["dataset_idx"][0]]

            # If val, no gradients
            with torch.no_grad() if tag == "val" else contextlib.nullcontext():
                rhat = self(b["stim"], b["dataset_idx"][0], b.get("behavior"))

            batch_loss = {
                'rhat': rhat.float(),
                'robs': b["robs"].float(),
                'dfs': b.get("dfs").float()
            }

            loss = self.loss_fn(batch_loss)

            if torch.isfinite(loss):
                if self.global_rank == 0:  # only log from rank 0 during steps
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
        """Training step."""
        bl_list = [bl] if isinstance(bl, dict) else bl
        base_loss = self._step(bl_list, "train")

        # Add regularization
        total_loss = base_loss
        epoch = self.current_epoch

        for reg in self.reg_terms:
            reg_loss = reg.loss(epoch)
            if torch.isfinite(reg_loss) and reg_loss.abs() > 0:
                total_loss = total_loss + reg_loss
                if self.global_rank == 0:
                    self.log(f"reg_loss/{reg.name}", reg_loss.item(),
                            on_step=False, on_epoch=True, sync_dist=False)

        return total_loss

    def validation_step(self, bl, _):
        """Validation step."""
        bl_list = [bl] if isinstance(bl, dict) else bl
        self._step(bl_list, "val")

    def on_validation_epoch_start(self):
        """Reset BPS aggregators at start of validation."""
        for agg in self.bps_aggs:
            agg.reset()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch.

        Follows the same distributed-safe pattern as MultiDatasetModel.
        """
        # 1. Average val-loss
        if self.val_losses:
            self.log("val_loss",
                     torch.stack(self.val_losses).mean(),
                     prog_bar=True, sync_dist=True)
        self.val_losses.clear()

        # 2. BPS per-dataset & overall
        per_ds = []
        for name, agg in zip(self.names, self.bps_aggs):
            # Build local SUM & COUNT tensors on every rank
            if len(agg.robs) == 0:
                local_sum = torch.tensor(0.0, device=self.device)
                local_count = torch.tensor(0, device=self.device)
            else:
                bps = agg.closure()
                if bps is not None:
                    valid_mask = ~torch.isnan(bps)
                    if valid_mask.any():
                        valid_bps = bps[valid_mask].clamp_min(0.0)
                        local_sum = valid_bps.sum().to(self.device)
                        local_count = torch.tensor(valid_bps.numel(), device=self.device)
                    else:
                        local_sum = torch.tensor(0.0, device=self.device)
                        local_count = torch.tensor(0, device=self.device)
                else:
                    local_sum = torch.tensor(0.0, device=self.device)
                    local_count = torch.tensor(0, device=self.device)

            # Global reduction (all-reduce)
            global_sum = self.trainer.strategy.reduce(local_sum, reduce_op="sum")
            global_count = self.trainer.strategy.reduce(local_count, reduce_op="sum")

            if global_count > 0:
                bps_mean = global_sum / global_count
                per_ds.append(bps_mean)

                if self.global_rank == 0:
                    self.log(f"val_bps/{name}", bps_mean, sync_dist=False)

        # Compute overall BPS
        if per_ds:
            overall = torch.stack(per_ds).mean()
            overall = torch.clamp_min(overall, 0.0)
        else:
            overall = torch.tensor(0.0, device=self.device)

        self.log("val_bps_overall", overall, prog_bar=True, sync_dist=True, rank_zero_only=False)

        torch.cuda.empty_cache()

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler for trainable parameters only.

        Uses the same weight decay exclusion logic as MultiDatasetModel.
        """
        # Get excluded parameters from regularizers
        excluded_names = set(get_excluded_params_for_weight_decay(self.reg_terms))

        # Build param groups with proper weight decay handling
        # For frozen core, all trainable params are "head" (adapters + readouts)
        wd_params, no_wd_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Only decay true "weights": tensor dims > 1 AND name endswith(".weight"),
            # and not in custom excluded set from regularizers
            apply_wd = (n not in excluded_names) and n.endswith(".weight") and (p.ndim > 1)
            (wd_params if apply_wd else no_wd_params).append(p)

        param_groups = []
        if wd_params:
            param_groups.append({"params": wd_params, "lr": self.hparams.lr, "weight_decay": self.hparams.wd})
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "lr": self.hparams.lr, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        # Log regularization info
        if excluded_names and self.global_rank == 0:
            print(f"[regularization] Excluded {len(excluded_names)} parameters from weight decay: {excluded_names}")

        # Configure LR scheduler
        lr_scheduler_type = self.hparams.get("lr_scheduler", "none")

        if lr_scheduler_type == "none":
            return optimizer

        total_epochs = self.trainer.max_epochs
        warmup_epochs = self.hparams.get("warmup_epochs", 5)

        if lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5)
        elif lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, verbose=True)
        elif lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs)
        elif lr_scheduler_type == "cosine_warmup":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=total_epochs,
                warmup_start_lr=0.0,
                eta_min=0.0
            )
        elif lr_scheduler_type == "cosine_warmup_restart":
            restart_period = self.hparams.get("restart_period", None)
            scheduler = LinearWarmupCosineAnnealingWarmRestartsLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=total_epochs,
                restart_period=restart_period,
                warmup_start_lr=0.0,
                eta_min=0.0
            )
        else:
            raise ValueError(f"Unknown scheduler {lr_scheduler_type}")

        # Lightning expects a dict if the scheduler needs a monitored metric
        if lr_scheduler_type == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return [optimizer], [scheduler]
