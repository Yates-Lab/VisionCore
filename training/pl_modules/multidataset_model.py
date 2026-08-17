"""
PyTorch Lightning module for multi-dataset neural encoding models.
"""

import os
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import MaskedLoss, PoissonBPSAggregator
from training.regularizers import create_regularizers, get_excluded_params_for_weight_decay
from training.schedulers import LinearWarmupCosineAnnealingLR, LinearWarmupCosineAnnealingWarmRestartsLR
# from schedulefree import AdamWScheduleFree

def _adamw_param_groups_named(named_params, wd, excluded_names, core_keys=("frontend","convnet","modulator"),
                              core_lr=None, head_lr=1e-3):

    core_lr = core_lr if core_lr is not None else head_lr

    core_wd, core_no, head_wd, head_no = [], [], [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue

        # Only decay true "weights": tensor dims > 1 AND name endswith(".weight"),
        # and not in your custom excluded set from YAML regs
        apply_wd = (n not in excluded_names) and n.endswith(".weight") and (p.ndim > 1)

        # The post-readout behavior residual is a dataset head even though its
        # name contains "modulator".  Treating it as visual core would silently
        # give it the reduced core LR during restricted joint refinement.
        is_output_head = "output_modulator" in n
        is_core = (not is_output_head) and any(k in n for k in core_keys)

        if is_core:
            (core_wd if apply_wd else core_no).append(p)
        else:
            (head_wd if apply_wd else head_no).append(p)

    param_groups = []
    if core_wd: param_groups.append({"params": core_wd, "lr": core_lr, "weight_decay": wd})
    if core_no: param_groups.append({"params": core_no, "lr": core_lr, "weight_decay": 0.0})
    if head_wd: param_groups.append({"params": head_wd, "lr": head_lr,             "weight_decay": wd})
    if head_no: param_groups.append({"params": head_no, "lr": head_lr,             "weight_decay": 0.0})
    return param_groups

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
    pretrained_scope : {"vision", "compatible", "complete"}, optional
        ``vision`` preserves the legacy adapter/frontend/convnet warm start.
        ``compatible`` loads every matching model tensor except a newly added
        output modulator. ``complete`` also loads an existing output modulator
        for joint-refinement branches. Both full-model modes verify the saved
        neuron identities (default: ``vision``).
    freeze_pretrained : bool, optional
        Freeze every parameter loaded from the selected pretrained scope. This
        is useful for identity-preserving residual fits (default: False).
    trainable_parameter_patterns : str, optional
        Comma-separated substrings. When provided, freeze every model parameter
        except names matching at least one substring. This persists selective
        unfreezing across checkpoint reconstruction.
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
                 freeze_vision: bool = False, compile_model: bool = False,
                 model_config_dict: dict = None,
                 pretrained_scope: str = "vision",
                 freeze_pretrained: bool = False,
                 trainable_parameter_patterns: str = None,
                 pretrained_exclude_prefixes: str = None,
                 pretrained_shape_adaptation: str = "none"):
        super().__init__()

        from models.config_loader import load_dataset_configs, load_config
        from models import build_model

        # Load model config
        # If model_config_dict is provided (from checkpoint), use it
        # Otherwise load from model_cfg path
        if model_config_dict is not None:
            self.model_config = model_config_dict
            print(f"Loading model from saved config dict (checkpoint is self-contained)")
        else:
            self.model_config = load_config(model_cfg)
            print(f"Loading model config from: {model_cfg}")

        # Save hyperparameters - this will save all __init__ arguments
        self.save_hyperparameters()

        # Override model_config_dict in hparams with the actual config
        # This ensures checkpoints are self-contained
        self.hparams.model_config_dict = self.model_config

        # Load dataset configurations from parent config
        # cfg_dir should now point to a parent config file (e.g., multi_basic_120_backimage_all.yaml)
        self.cfgs = load_dataset_configs(cfg_dir)

        # Limit to max_ds datasets
        self.cfgs = self.cfgs[:max_ds]

        # Extract dataset names from session names
        self.names = [cfg['session'] for cfg in self.cfgs]
        for c, n in zip(self.cfgs, self.names):
            c["_dataset_name"] = n

        # Snapshot the resolved cids into the checkpoint (TWIN_IMPROVEMENTS 1).
        # Readout sizes are otherwise recoverable only by re-reading the
        # session YAMLs at load time, so editing those YAMLs later silently
        # breaks every checkpoint trained against them. With this, a checkpoint
        # carries its own population and `eval.load_twin` can name a drifting
        # session instead of dumping tensor-size mismatches.
        self.hparams.dataset_cids = {
            n: list(c.get('cids', [])) for c, n in zip(self.cfgs, self.names)
        }

        # Build model using the loaded config
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
                base_model.core_forward = torch.compile(
                base_model.core_forward,
                backend="inductor",
                dynamic=False,      # shapes are fixed; fewer guards, better cudagraph capture
                fullgraph=True      # try to keep it one fused graph (falls back if it can’t)
            )
                print("Model compiled successfully")
            except Exception as e:
                print(f"torch.compile failed: {e}")

        self.model = base_model

        # Populated by `_load_pretrained_components`.  Readout biases loaded
        # from a checkpoint must not be overwritten in `on_fit_start`.
        self._pretrained_parameter_names = set()
        self._pretrained_readout_indices = set()

        # Load pretrained components if specified.  The default scope is the
        # historical vision-only path, so existing commands retain their old
        # semantics.
        if pretrained_checkpoint is not None:
            self._load_pretrained_components(
                pretrained_checkpoint,
                freeze_vision=freeze_vision,
                pretrained_scope=pretrained_scope,
                freeze_pretrained=freeze_pretrained,
                exclude_prefixes=pretrained_exclude_prefixes,
                shape_adaptation=pretrained_shape_adaptation,
            )

        if trainable_parameter_patterns:
            self._set_trainable_parameter_patterns(trainable_parameter_patterns)

        # Initialize regularization system
        named_params = list(self.model.named_parameters())
        self.reg_terms = create_regularizers(self.model_config, named_params)

        self.core_lr_scaled = lr * self.hparams.get("core_lr_scale", 1.0)
        self.head_lr = lr  # unchanged for dataset heads
        self.log_input = isinstance(self.model.activation, nn.Identity)

        # Initialize loss function
        self.loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=self.log_input, reduction="none"))
        print(f"Using Poisson loss (log_input={self.log_input})")

        self.bps_aggs = [PoissonBPSAggregator() for _ in self.names]
        self.val_losses = []
        self.val_losses_by_ds = {i: [] for i in range(len(self.names))}

        # Build subject -> dataset index mapping (subject = name before first '_')
        self._subject_ds = {}
        for i, name in enumerate(self.names):
            subj = name.split("_")[0]
            self._subject_ds.setdefault(subj, []).append(i)

    def _load_pretrained_components(
        self,
        pretrained_checkpoint: str,
        freeze_vision: bool = False,
        pretrained_scope: str = "vision",
        freeze_pretrained: bool = False,
        exclude_prefixes: str = None,
        shape_adaptation: str = "none",
    ):
        """
        Load pretrained vision components from a checkpoint.
        
        Parameters
        ----------
        pretrained_checkpoint : str
            Path to checkpoint file
        freeze_vision : bool, optional
            Whether to freeze loaded parameters (default: False)
        pretrained_scope : {"vision", "compatible"}, optional
            Components to load. ``compatible`` loads the complete existing
            model while leaving a new output modulator at its initialization.
        freeze_pretrained : bool, optional
            Freeze all parameters loaded from the selected scope.
            
        Returns
        -------
        int
            Number of parameters loaded
        """
        if pretrained_scope not in {"vision", "compatible", "complete"}:
            raise ValueError(
                "pretrained_scope must be 'vision', 'compatible', or "
                "'complete', got "
                f"{pretrained_scope!r}"
            )
        if shape_adaptation not in {
            "none",
            "trim_readout_features",
            "scale_gaussian_readout",
        }:
            raise ValueError(
                "shape_adaptation must be 'none', 'trim_readout_features', or "
                f"'scale_gaussian_readout', got {shape_adaptation!r}"
            )
        if isinstance(exclude_prefixes, str):
            exclude_prefixes = tuple(
                part.strip() for part in exclude_prefixes.split(',')
                if part.strip()
            )
        else:
            exclude_prefixes = tuple(exclude_prefixes or ())
        print(
            f"Loading pretrained components from: {pretrained_checkpoint} "
            f"(scope={pretrained_scope})"
        )

        # Load the pretrained checkpoint
        checkpoint = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        else:
            pretrained_state_dict = checkpoint

        # Lightning prefixes the wrapped model with `model.`; torch.compile
        # inserts `_orig_mod.` after it.  Normalize both variants to the keys
        # used by `self.model.state_dict()`.
        normalized_state = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('model._orig_mod.'):
                key = key[len('model._orig_mod.'):]
            elif key.startswith('model.'):
                key = key[len('model.'):]
            normalized_state[key] = value

        target_state = self.model.state_dict()
        vision_prefixes = ('adapters', 'frontend', 'convnet')
        output_prefix = 'output_modulator'

        if pretrained_scope in {"compatible", "complete"}:
            # A full warm start is only scientifically valid if every dataset
            # still refers to the same neurons in the same order.
            saved_hparams = checkpoint.get('hyper_parameters', {}) or {}
            saved_cids = saved_hparams.get('dataset_cids')
            if saved_cids is not None:
                current_cids = {
                    name: list(cfg.get('cids', []))
                    for name, cfg in zip(self.names, self.cfgs)
                }
                cid_errors = [
                    name for name, cids in current_cids.items()
                    if name not in saved_cids or list(saved_cids[name]) != cids
                ]
                if cid_errors:
                    raise ValueError(
                        "Compatible checkpoint load refused because dataset "
                        "neuron identities/order changed for: "
                        + ", ".join(cid_errors[:5])
                    )
            else:
                print(
                    "⚠ Checkpoint predates saved dataset_cids; compatible load "
                    "can verify tensor shapes but not neuron identities"
                )

            expected_keys = set(target_state)
            if pretrained_scope == "compatible":
                expected_keys = {
                    key for key in expected_keys
                    if not key.startswith(output_prefix)
                }
        else:
            expected_keys = {
                key for key in target_state
                if key.startswith(vision_prefixes)
            }
        if exclude_prefixes:
            expected_keys = {
                key for key in expected_keys
                if not key.startswith(exclude_prefixes)
            }

        selected_state = {}
        shape_errors = []
        adapted_shapes = []
        for key in expected_keys:
            if key not in normalized_state:
                continue
            source = normalized_state[key]
            target = target_state[key]
            if tuple(source.shape) != tuple(target.shape):
                can_trim_readout = (
                    shape_adaptation == "trim_readout_features"
                    and key.startswith("readouts.")
                    and key.endswith(".features.weight")
                    and source.ndim == target.ndim == 4
                    and source.shape[0] == target.shape[0]
                    and source.shape[1] >= target.shape[1]
                    and tuple(source.shape[2:]) == tuple(target.shape[2:])
                )
                if can_trim_readout:
                    selected_state[key] = source[:, : target.shape[1]].clone()
                    adapted_shapes.append(
                        (key, tuple(source.shape), tuple(target.shape))
                    )
                else:
                    shape_errors.append((key, tuple(source.shape), tuple(target.shape)))
            else:
                selected_state[key] = source

        if shape_adaptation == "scale_gaussian_readout":
            saved_hparams = checkpoint.get('hyper_parameters', {}) or {}
            source_model_config = (
                saved_hparams.get('model_config_dict')
                or saved_hparams.get('model_config')
                or {}
            )
            target_model_config = getattr(self, 'model_config', {}) or {}

            def scaffold_hw(config):
                value = (
                    (config.get('convnet') or {})
                    .get('params', {})
                    .get('scaffold_size')
                )
                if isinstance(value, int):
                    return (value, value)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return tuple(int(v) for v in value)
                return None

            source_hw = scaffold_hw(source_model_config)
            target_hw = scaffold_hw(target_model_config)
            if source_hw is None or target_hw is None:
                raise ValueError(
                    "scale_gaussian_readout requires source and target "
                    "convnet.params.scaffold_size metadata"
                )
            if any(value <= 1 for value in (*source_hw, *target_hw)):
                raise ValueError(
                    "scale_gaussian_readout requires scaffold dimensions > 1"
                )
            scale_yx = torch.tensor(
                [
                    (target_hw[0] - 1) / (source_hw[0] - 1),
                    (target_hw[1] - 1) / (source_hw[1] - 1),
                ],
                dtype=torch.float32,
            )
            scaled_readout_tensors = []
            for key, value in tuple(selected_state.items()):
                if (
                    key.startswith('readouts.')
                    and (key.endswith('.mean') or key.endswith('.std'))
                    and value.ndim == 2
                    and value.shape[-1] == 2
                ):
                    selected_state[key] = value.clone() * scale_yx.to(
                        dtype=value.dtype
                    )
                    scaled_readout_tensors.append(key)
            if not scaled_readout_tensors:
                raise ValueError(
                    "scale_gaussian_readout found no Gaussian mean/std tensors"
                )
            adapted_shapes.extend(
                (key, source_hw, target_hw) for key in scaled_readout_tensors
            )

        missing_expected = sorted(expected_keys.difference(selected_state))
        if shape_errors or missing_expected:
            details = []
            if shape_errors:
                details.append(f"shape mismatches: {shape_errors[:5]}")
            if missing_expected:
                details.append(f"missing keys: {missing_expected[:5]}")
            raise RuntimeError(
                f"Cannot perform {pretrained_scope} checkpoint load; "
                + "; ".join(details)
            )

        self.model.load_state_dict(selected_state, strict=False)
        self._pretrained_parameter_names = {
            name for name, _ in self.model.named_parameters()
            if name in selected_state
        }
        self._pretrained_readout_indices = {
            int(name.split('.')[1])
            for name in selected_state
            if name.startswith('readouts.') and name.split('.')[1].isdigit()
        }

        component_counts = {}
        for key in selected_state:
            component = key.split('.', 1)[0]
            component_counts[component] = component_counts.get(component, 0) + 1
        print(f"✓ Loaded {len(selected_state)} pretrained tensors")
        print(f"   Breakdown: {component_counts}")
        if exclude_prefixes:
            print(f"   Excluded prefixes: {exclude_prefixes}")
        if adapted_shapes:
            print(
                "   Applied explicit readout shape/coordinate adaptations: "
                f"{adapted_shapes[:5]}"
            )

        frozen_names = set()
        if freeze_pretrained:
            frozen_names.update(self._pretrained_parameter_names)
        if freeze_vision:
            frozen_names.update(
                name for name in self._pretrained_parameter_names
                if name.startswith(vision_prefixes)
            )
        for name, param in self.model.named_parameters():
            if name in frozen_names:
                param.requires_grad = False
        if frozen_names:
            print(f"✓ Froze {len(frozen_names)} loaded parameter tensors")

        return len(selected_state)

    def _set_trainable_parameter_patterns(self, patterns):
        """Freeze everything except explicitly named parameter families."""
        if isinstance(patterns, str):
            patterns = [part.strip() for part in patterns.split(',')]
        patterns = tuple(part for part in patterns if part)
        if not patterns:
            raise ValueError("trainable_parameter_patterns contained no patterns")

        trainable_names = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = any(pattern in name for pattern in patterns)
            if parameter.requires_grad:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "trainable_parameter_patterns matched no model parameters: "
                f"{patterns}"
            )
        print(
            f"✓ Selectively unfroze {len(trainable_names)} parameter tensors "
            f"matching {patterns}"
        )
        return trainable_names

    def load_distilled_output_modulator(self, artifact_path: str):
        """Load a standalone behavior-residual artifact with CID checks.

        The artifact is used only to initialize the optional second output
        head.  Subsequent Lightning checkpoints contain those tensors in their
        ordinary state_dict and are therefore self-contained.
        """
        module = getattr(self.model, 'distilled_output_modulator', None)
        if module is None:
            raise ValueError(
                "A distilled behavior artifact was supplied, but the model "
                "config has no distilled_output_modulator"
            )
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
        saved_names = list(artifact.get('dataset_names') or ())
        if saved_names != self.names:
            raise ValueError(
                "Distilled behavior dataset order differs from the student: "
                f"artifact={saved_names[:3]}, student={self.names[:3]}"
            )
        saved_cids = artifact.get('cids_by_session') or {}
        cid_errors = [
            name for name, cfg in zip(self.names, self.cfgs)
            if list(saved_cids.get(name, ())) != list(cfg.get('cids', ()))
        ]
        if cid_errors:
            raise ValueError(
                "Distilled behavior neuron identities/order changed for: "
                + ", ".join(cid_errors[:5])
            )
        configured = dict(
            (self.model_config.get('distilled_output_modulator') or {})
            .get('params') or {}
        )
        if dict(artifact.get('config') or {}) != configured:
            raise ValueError(
                "Distilled behavior artifact config does not match the "
                "student distilled_output_modulator config"
            )
        module.load_state_dict(artifact['state_dict'], strict=True)
        print(
            f"✓ Loaded distilled output behavior head from {artifact_path} "
            f"for {len(self.names)} datasets"
        )
        return len(artifact['state_dict'])

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

    def on_fit_start(self):
        """Initialize readout biases from empirical firing rates."""
        super().on_fit_start()
        dm = self.trainer.datamodule
        if not hasattr(dm, 'train_dsets') or not dm.train_dsets:
            return

        if self.global_rank == 0:
            print("Initializing readout biases from empirical firing rates...")
        for idx, name in enumerate(self.names):
            if idx in self._pretrained_readout_indices:
                if self.global_rank == 0:
                    print(f"  {name}: preserving pretrained readout bias")
                continue
            if name not in dm.train_dsets:
                continue

            dset = dm.train_dsets[name]
            # Unwrap Float32View if present
            combined = dset.base if hasattr(dset, 'base') else dset

            # Accumulate weighted mean firing rate across all sub-datasets
            total_rate = None
            total_weight = None
            for sub_dset in combined.dsets:
                robs = sub_dset['robs'].float()   # (T, N)
                dfs = sub_dset['dfs'].float() if 'dfs' in sub_dset else torch.ones_like(robs)
                # sum(robs * dfs, dim=0) / sum(dfs, dim=0) = weighted mean per neuron
                weighted_sum = (robs * dfs).sum(dim=0)
                weight_sum = dfs.sum(dim=0)
                if total_rate is None:
                    total_rate = weighted_sum
                    total_weight = weight_sum
                else:
                    total_rate += weighted_sum
                    total_weight += weight_sum

            if total_rate is None:
                continue

            mean_rate = total_rate / total_weight.clamp(min=1.0)

            # Compute inverse of output activation to get bias
            if self.log_input:
                # Identity activation → PoissonNLL with log_input=True → bias = log(rate)
                bias = torch.log(mean_rate.clamp(min=1e-8))
            else:
                # Softplus activation → inverse softplus: log(exp(x) - 1)
                # Numerically stable: for large x, inv_softplus ≈ x
                clamped = mean_rate.clamp(min=1e-6)
                bias = torch.where(
                    clamped > 20.0,
                    clamped,
                    torch.log(torch.expm1(clamped))
                )

            readout = self.model.readouts[idx]
            if hasattr(readout, 'bias') and readout.bias is not None:
                readout.bias.data = bias.to(dtype=readout.bias.dtype, device=readout.bias.device)
                if self.global_rank == 0:
                    print(f"  {name}: bias range [{bias.min():.3f}, {bias.max():.3f}], "
                          f"mean_rate range [{mean_rate.min():.4f}, {mean_rate.max():.4f}]")

    @staticmethod
    def _sync_loader_sampler_epoch(loader, epoch):
        """Set epoch on custom samplers, including lists of validation loaders."""
        if isinstance(loader, (list, tuple)):
            for child in loader:
                MultiDatasetModel._sync_loader_sampler_epoch(child, epoch)
            return
        for sampler in (
            getattr(loader, "sampler", None),
            getattr(loader, "batch_sampler", None),
        ):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def on_train_epoch_start(self):
        """Set up for training epoch."""
        super().on_train_epoch_start()
        # Custom homogeneous batch samplers are recreated when a checkpoint is
        # resumed, so their internal counter is not restored by Lightning.
        # Pinning it to the restored trainer epoch prevents epochs 0..N from
        # being replayed after an epoch-N checkpoint.  This is a no-op for
        # ordinary samplers and for a fresh epoch-zero run.
        self._sync_loader_sampler_epoch(
            self.trainer.train_dataloader, self.current_epoch
        )
        optimizer = self.optimizers()
        # if isinstance(optimizer, AdamWScheduleFree):
        #     optimizer.train()

    def forward(
        self,
        stim,
        ds_idx,
        beh=None,
        history=None,
        output_beh=None,
    ):
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
        history : torch.Tensor, optional
            Spike history tensor (for spike history models)

        Returns
        -------
        torch.Tensor
            Predicted neural responses
        """
        # Check if this is a modulator-only model (no vision processing)
        if self.is_modulator_only:
            y = self.model(
                stimulus=None,
                dataset_idx=ds_idx,
                behavior=beh,
                history=history,
                output_behavior=output_beh,
            )
        else:
            y = self.model(
                stimulus=stim,
                dataset_idx=ds_idx,
                behavior=beh,
                history=history,
                output_behavior=output_beh,
            )
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
                rhat = self(
                    stimulus,
                    b["dataset_idx"][0],
                    b.get("behavior"),
                    b.get("history"),
                    b.get("output_behavior"),
                )

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
                    ds_idx = b["dataset_idx"][0].item()
                    self.bps_aggs[ds_idx](batch_loss)
                    self.val_losses.append(loss.detach())
                    self.val_losses_by_ds[ds_idx].append(loss.detach())
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
        bl : dict or list of dict
            Batch (single dataset dict when homogeneous batching with default_collate,
            otherwise list of dicts when using group_collate)
        _ : int
            Batch index (unused)

        Returns
        -------
        torch.Tensor
            Total loss (base + auxiliary + regularization)
        """
        # Normalize batch input to a list of dicts for internal processing
        bl_list = [bl] if isinstance(bl, dict) else bl

        # Get base loss from datasets
        base_loss = self._step(bl_list, "train")

        # Log overall train loss
        if self.global_rank == 0:
            bs = sum(b["robs"].shape[0] for b in bl_list)
            self.log("train_loss", base_loss,
                     batch_size=bs,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)

        # Add auxiliary loss for PC modulator if present
        aux_loss = self._compute_auxiliary_loss()
        total_loss = base_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            # Log auxiliary loss to wandb
            if self.global_rank == 0:
                bs = sum(b["robs"].shape[0] for b in bl_list)
                self.log('aux_loss', aux_loss.item(),
                        batch_size=bs,
                        on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)

        # Add regularization penalties
        epoch = self.current_epoch

        for reg in self.reg_terms:
            reg_loss = reg.loss(epoch)
            if torch.isfinite(reg_loss) and reg_loss.abs() > 0:
                total_loss = total_loss + reg_loss
                # Log individual regularization losses
                if self.global_rank == 0:
                    self.log(f"reg_loss/{reg.name}", reg_loss.item(),
                            on_step=False, on_epoch=True, sync_dist=False)

        return total_loss

    def validation_step(self, bl, _):
        """Validation step supporting dict or list batches."""
        bl_list = [bl] if isinstance(bl, dict) else bl
        self._step(bl_list, "val")

    def on_validation_epoch_start(self):
        """Synchronize validation sampling and reset validation accumulators."""
        super().on_validation_epoch_start()
        # Keep shuffled partial-validation draws tied to the actual trainer
        # epoch across checkpoint resumes.  This belongs in the same hook as
        # the aggregator reset; a second definition would silently override it.
        self._sync_loader_sampler_epoch(
            self.trainer.val_dataloaders, self.current_epoch
        )
        for agg in self.bps_aggs:
            agg.reset()
        for v in self.val_losses_by_ds.values():
            v.clear()

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
        per_ds_bps = {}  # dataset_idx -> bps_mean, for per-subject aggregation
        for ds_i, (name, agg) in enumerate(zip(self.names, self.bps_aggs)):

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
                per_ds_bps[ds_i] = bps_mean

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

        # 3. Per-subject aggregated val_loss and val_bps
        if self.global_rank == 0:
            for subj, ds_idxs in self._subject_ds.items():
                # Aggregate val_loss across sessions for this subject
                subj_losses = []
                for i in ds_idxs:
                    subj_losses.extend(self.val_losses_by_ds[i])
                if subj_losses:
                    self.log(f"val_loss/{subj}",
                             torch.stack(subj_losses).mean(),
                             sync_dist=False)

                # Aggregate val_bps across sessions for this subject
                subj_bps = [per_ds_bps[i] for i in ds_idxs if i in per_ds_bps]
                if subj_bps:
                    self.log(f"val_bps/{subj}",
                             torch.stack(subj_bps).mean(),
                             sync_dist=False)

        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns
        -------
        optimizer or dict
            Optimizer (and optionally scheduler configuration)
        """
        # ----- exclusions from your YAML regularizers -----
        excluded_names = set(get_excluded_params_for_weight_decay(self.reg_terms))  # already in your code
        head_lr = self.head_lr
        core_lr = self.core_lr_scaled

        # ----- build groups: WD only on true weights -----
        pg = _adamw_param_groups_named(
            list(self.named_parameters()),
            wd=self.hparams.wd,
            excluded_names=excluded_names,
            core_lr=core_lr,
            head_lr=head_lr,
        )

        optim = torch.optim.AdamW(pg, betas=(0.9, 0.999), eps=1e-8)

        # Log regularization info
        if excluded_names and self.global_rank == 0:
            print(f"[regularization] Excluded {len(excluded_names)} parameters from weight decay: {excluded_names}")

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
        elif sched_type == "cosine_warmup_restart":
            # Use linear warmup followed by cosine annealing with warm restarts
            warmup_epochs = self.hparams.get("warmup_epochs", 5)
            restart_period = self.hparams.get("restart_period", None)
            scheduler = LinearWarmupCosineAnnealingWarmRestartsLR(
                optim,
                warmup_epochs=warmup_epochs,
                max_epochs=self.trainer.max_epochs,
                restart_period=restart_period,
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

        # Apply each proximal operator exactly once.  Parameter-specific
        # learning rates are resolved inside Regularizer.prox from the
        # optimizer groups; the previous nested loop repeated a proximal
        # update once per optimizer group (clamps happened to be idempotent).
        # This deliberately happens before zero_grad: multisession proximal
        # sparsity must be able to identify the readout used by this batch and
        # leave inactive sessions untouched.
        fallback_lr = max(float(group["lr"]) for group in optimizer.param_groups)
        for reg in self.reg_terms:
            reg.prox(epoch, fallback_lr, optimizer=optimizer)

        optimizer.zero_grad()

        return loss
