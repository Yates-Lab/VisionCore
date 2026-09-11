"""
PyTorch Lightning module for multi-dataset neural encoding models.
"""

import os
import contextlib
import re
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import MaskedLoss, PoissonBPSAggregator
from training.regularizers import create_regularizers, get_excluded_params_for_weight_decay
from training.schedulers import LinearWarmupCosineAnnealingLR, LinearWarmupCosineAnnealingWarmRestartsLR
# from schedulefree import AdamWScheduleFree


_DATASET_COMPONENT_KEY = re.compile(
    r"^(adapters|readouts|phase_readouts)\.(\d+)(\..+)$"
)

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

        is_core = any(k in n for k in core_keys)

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


def set_trainable_model_components(
    model: nn.Module, component_names
) -> tuple[int, int]:
    """Make only the named top-level model components trainable.

    This is intentionally explicit for calibration fits.  In particular, a
    joint ``readouts`` + ``phase_readouts`` fit can let the established deep
    head compensate for a stronger phase-preserving branch without changing
    the visual core or behavior modulator.
    """
    if isinstance(component_names, str):
        component_names = [component_names]
    requested = tuple(dict.fromkeys(str(value) for value in component_names))
    allowed = {
        "adapters",
        "frontend",
        "convnet",
        "modulator",
        "recurrent",
        "readouts",
        "phase_readouts",
    }
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown trainable model components: {unknown}")
    if not requested:
        raise ValueError("trainable_components must not be empty")
    for component in requested:
        if not hasattr(model, component) or getattr(model, component) is None:
            raise ValueError(
                f"Requested trainable component {component!r} is absent"
            )

    trainable_count = 0
    frozen_count = 0
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        trainable = root in requested
        parameter.requires_grad = trainable
        trainable_count += int(trainable)
        frozen_count += int(not trainable)
    return trainable_count, frozen_count

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
                 freeze_vision: bool = False, compile_model: bool = False,
                 model_config_dict: dict = None,
                 pretrained_load_heads: bool = False,
                 core_lr_scale: float = 1.0,
                 selected_sessions: list[str] = None):
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

        # The warm-start checkpoint is an initialization input, not a runtime
        # dependency of the resulting checkpoint.  The complete trained state
        # is saved below, so do not make future loads re-open the parent file.
        self.save_hyperparameters(ignore=["pretrained_checkpoint"])

        # Override model_config_dict in hparams with the actual config
        # This ensures checkpoints are self-contained
        self.hparams.model_config_dict = self.model_config

        # Load dataset configurations from parent config
        # cfg_dir should now point to a parent config file (e.g., multi_basic_120_backimage_all.yaml)
        self.cfgs = load_dataset_configs(cfg_dir)

        if selected_sessions is not None:
            requested = list(dict.fromkeys(selected_sessions))
            available = {str(cfg["session"]) for cfg in self.cfgs}
            missing = sorted(set(requested) - available)
            if missing:
                raise ValueError(f"Requested sessions are absent: {missing}")
            by_name = {str(cfg["session"]): cfg for cfg in self.cfgs}
            self.cfgs = [by_name[name] for name in requested]

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

        # Load pretrained vision components if specified
        self._loaded_pretrained_heads = False
        self._restored_from_checkpoint = False
        if pretrained_checkpoint is not None:
            self._load_pretrained_components(
                pretrained_checkpoint,
                freeze_vision,
                load_heads=pretrained_load_heads,
            )

        trainable_components = self.model_config.get("trainable_components")
        if trainable_components is not None:
            trainable_count, frozen_count = set_trainable_model_components(
                self.model, trainable_components
            )
            label = (
                "Selected-component fit "
                f"({', '.join(str(v) for v in trainable_components)})"
            )
            print(
                f"✓ {label}: {trainable_count} trainable and "
                f"{frozen_count} frozen tensors"
            )

        # Initialize regularization system
        named_params = list(self.model.named_parameters())
        self.reg_terms = create_regularizers(self.model_config, named_params)

        self.core_lr_scaled = lr * core_lr_scale
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

    @staticmethod
    def _canonical_pretrained_state_dict(state_dict):
        """Map Lightning/compiled checkpoint keys to inner-model key names."""
        canonical = {}
        for key, value in state_dict.items():
            if key.startswith("model._orig_mod."):
                key = key[len("model._orig_mod."):]
            elif key.startswith("model."):
                key = key[len("model."):]
            canonical[key] = value
        return canonical

    @staticmethod
    def _remap_pretrained_dataset_components(
        source,
        source_sessions=None,
        target_sessions=None,
    ):
        """Map indexed per-session tensors by session identity, not list index.

        A selected-session calibration constructs a one-head target model even
        when its source checkpoint contains all sessions.  Numeric index 0 in
        those two models generally names different sessions.  Silently using
        that index can load another animal's readout when unit counts happen to
        agree, or produce a tensor-size failure when they do not.
        """
        if source_sessions is None or target_sessions is None:
            return dict(source)
        source_sessions = [str(value) for value in source_sessions]
        target_sessions = [str(value) for value in target_sessions]
        if len(source_sessions) != len(set(source_sessions)):
            raise ValueError("pretrained checkpoint contains duplicate session names")
        if len(target_sessions) != len(set(target_sessions)):
            raise ValueError("target model contains duplicate session names")
        source_index = {session: index for index, session in enumerate(source_sessions)}
        missing = [session for session in target_sessions if session not in source_index]
        if missing:
            raise ValueError(
                "target sessions are absent from the pretrained checkpoint: "
                f"{missing}"
            )
        target_index = {session: index for index, session in enumerate(target_sessions)}

        remapped = {}
        for key, value in source.items():
            match = _DATASET_COMPONENT_KEY.match(key)
            if match is None:
                remapped[key] = value
                continue
            old_index = int(match.group(2))
            if old_index >= len(source_sessions):
                raise IndexError(
                    f"pretrained key {key!r} exceeds its session table"
                )
            session = source_sessions[old_index]
            if session not in target_index:
                continue
            new_key = (
                f"{match.group(1)}.{target_index[session]}{match.group(3)}"
            )
            if new_key in remapped:
                raise ValueError(f"pretrained remapping produced duplicate key {new_key}")
            remapped[new_key] = value
        return remapped

    def _compatible_pretrained_state(
        self,
        state_dict,
        load_heads=False,
        source_sessions=None,
        target_sessions=None,
    ):
        """Select compatible tensors and migrate Gaussian readouts exactly."""
        source = MultiDatasetModel._canonical_pretrained_state_dict(state_dict)
        source = MultiDatasetModel._remap_pretrained_dataset_components(
            source,
            source_sessions=source_sessions,
            target_sessions=target_sessions,
        )
        target = self.model.state_dict()
        prefixes = ["adapters", "frontend", "convnet"]
        if load_heads:
            # ``phase_readouts`` is included so a converged shallow branch can
            # be refined under a new optimizer/sampling schedule.  When the
            # source predates that branch, the migration below still creates
            # the exact zero-output initialization from the deep RF location.
            prefixes.extend([
                "modulator",
                "readouts",
                "phase_readouts",
            ])

        selected = {}
        migrated_readouts = set()
        migrated_low_rank_readouts = set()
        for key, value in source.items():
            if not any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
                continue
            if key not in target or target[key].shape != value.shape:
                continue

            migrated = value
            if load_heads and key.startswith("readouts.") and key.endswith("features.weight"):
                readout_prefix = key.rsplit(".features.weight", 1)[0]
                target_spatial_key = readout_prefix + ".spatial_weights"
                source_spatial_key = readout_prefix + ".spatial_weights"
                if target_spatial_key in target and source_spatial_key not in source:
                    migrated_readouts.add(readout_prefix)
            selected[key] = migrated

        # A lower-rank sparse Gaussian is exactly the first factors of a
        # higher-rank readout. Populate those factors and zero the new channel
        # factors so an architectural expansion starts with equivalent logits.
        if load_heads:
            from models.modules.readout import SparseGaussianLowRankReadout

            for component_name in ("readouts", "phase_readouts"):
                modules = getattr(self.model, component_name, None)
                if modules is None:
                    continue
                for readout_index, readout in enumerate(modules):
                    if not isinstance(readout, SparseGaussianLowRankReadout):
                        continue
                    prefix = f"{component_name}.{readout_index}"
                    feature_key = prefix + ".features.weight"
                    spatial_key = prefix + ".spatial_weights"
                    if feature_key in selected and spatial_key in selected:
                        continue
                    if feature_key not in source:
                        continue
                    source_feature = source[feature_key]
                    if spatial_key in source:
                        source_spatial = source[spatial_key]
                    elif component_name == "readouts" and source_feature.shape[0] == readout.n_units:
                        # Embed an ordinary Gaussian as one sparse factor before
                        # expanding its rank. Broaden its envelope, compensating
                        # exactly in the spatial factor and channel weights.
                        height, width = readout.spatial_shape
                        std_key = prefix + ".std"
                        old_std = source[std_key]
                        new_std = old_std.clamp_min(readout.migration_std_floor)
                        masks = [
                            MultiDatasetModel._gaussian_mask_from_parameters(
                                source[prefix + ".mean"], std,
                                source[prefix + ".theta"], height, width,
                            )
                            for std in (old_std, new_std)
                        ]
                        ratio = masks[0] / masks[1].clamp_min(torch.finfo(new_std.dtype).tiny)
                        norm = ratio.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
                        source_spatial = (ratio / norm).unsqueeze(1)
                        source_feature = source_feature * norm.reshape(-1, 1, 1, 1) / readout.output_scale
                        selected[std_key] = new_std
                    else:
                        continue
                    target_feature = target[feature_key]
                    target_spatial = target[spatial_key]
                    if (
                        source_feature.shape[0] % readout.n_units != 0
                        or target_feature.shape[0]
                        != readout.n_units * readout.rank
                        or source_feature.shape[1:] != target_feature.shape[1:]
                        or target_spatial.shape
                        != (readout.n_units, readout.rank, *readout.spatial_shape)
                    ):
                        continue
                    source_rank = source_feature.shape[0] // readout.n_units
                    if (
                        source_rank >= readout.rank
                        or source_spatial.shape
                        != (readout.n_units, source_rank, *readout.spatial_shape)
                    ):
                        continue
                    embedded_feature = torch.zeros_like(target_feature).reshape(
                        readout.n_units,
                        readout.rank,
                        *target_feature.shape[1:],
                    )
                    embedded_spatial = torch.zeros_like(target_spatial)
                    # Leave a small, distinct spatial seed in dormant factors.
                    # Their zero channel weights preserve the old logits while
                    # the nonzero maps give them a gradient on the first update.
                    # Rescale every retained factor inversely so its product and
                    # the joint spatial unit norm are both preserved.
                    source_feature = source_feature.reshape(
                        readout.n_units,
                        source_rank,
                        *source_feature.shape[1:],
                    )
                    seed_fraction = 0.05
                    extras = target_spatial[:, source_rank:].clone()
                    extras = extras / torch.linalg.vector_norm(
                        extras, dim=(1, 2, 3), keepdim=True
                    ).clamp_min(1.0e-12)
                    extras.mul_(seed_fraction)
                    source_norm = torch.linalg.vector_norm(
                        source_spatial, dim=(1, 2, 3), keepdim=True
                    ).clamp_min(1.0e-12)
                    retained_norm = (1.0 - seed_fraction ** 2) ** 0.5
                    scale = retained_norm / source_norm
                    embedded_spatial[:, :source_rank].copy_(
                        source_spatial * scale
                    )
                    embedded_spatial[:, source_rank:].copy_(extras)
                    embedded_feature[:, :source_rank].copy_(
                        source_feature / scale.unsqueeze(-1)
                    )
                    selected[feature_key] = embedded_feature.reshape_as(
                        target_feature
                    )
                    selected[spatial_key] = embedded_spatial
                    migrated_low_rank_readouts.add(prefix)

        # Reparameterize every ordinary Gaussian readout as A(y,x) * G(y,x).
        # With no requested floor, a unit-norm constant A reproduces the old
        # implementation.  With a broader target Gaussian G_new, initialize
        # A proportional to G_old/G_new and compensate its norm in the channel
        # weights.  In either case C_new*A*G_new == C_old*G_old at step zero.
        for readout_prefix in migrated_readouts:
            readout_index = int(readout_prefix.split(".")[1])
            readout = self.model.readouts[readout_index]
            spatial_key = readout_prefix + ".spatial_weights"
            feature_key = readout_prefix + ".features.weight"
            mean_key = readout_prefix + ".mean"
            std_key = readout_prefix + ".std"
            theta_key = readout_prefix + ".theta"
            height, width = target[spatial_key].shape[-2:]
            floor = float(getattr(readout, "migration_std_floor", 0.0))

            if floor <= 0:
                scale = float(height * width) ** 0.5
                selected[spatial_key] = torch.full_like(
                    target[spatial_key], 1.0 / scale
                )
                selected[feature_key] = source[feature_key] * scale
                continue

            old_std = source[std_key]
            new_std = old_std.clamp_min(floor)
            old_mask = MultiDatasetModel._gaussian_mask_from_parameters(
                source[mean_key], old_std, source[theta_key], height, width
            )
            new_mask = MultiDatasetModel._gaussian_mask_from_parameters(
                source[mean_key], new_std, source[theta_key], height, width
            )
            ratio = old_mask / new_mask.clamp_min(torch.finfo(new_mask.dtype).tiny)
            ratio_norm = ratio.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
            selected[spatial_key] = (ratio / ratio_norm).unsqueeze(1)
            selected[feature_key] = source[feature_key] * ratio_norm.squeeze(-1).squeeze(-1)[:, None, None, None]
            selected[std_key] = new_std

        # A newly added shallow branch must contribute exactly zero, but its
        # learned location can inherit the corresponding deep neuron's RF.
        # Readout coordinates are in pixels relative to map center, so scale
        # means/stds from the 9x9 scaffold to the 35x35 signed stage-1 map.
        initialized_phase_readouts = set()
        for component_name in ("phase_readouts",):
            phase_readouts = getattr(self.model, component_name, None)
            if not load_heads or phase_readouts is None:
                continue
            for readout_index, phase_readout in enumerate(phase_readouts):
                phase_prefix = f"{component_name}.{readout_index}"
                deep_prefix = f"readouts.{readout_index}"
                phase_mean_key = phase_prefix + ".mean"
                deep_mean_key = deep_prefix + ".mean"
                deep_std_key = deep_prefix + ".std"
                deep_theta_key = deep_prefix + ".theta"
                # A source model that already owns a compatible shallow
                # branch is authoritative.  Do not replace its learned
                # location/width (or zero its learned feature weights) with
                # the deep-readout migration intended only for old models.
                if phase_mean_key in selected:
                    continue
                if deep_mean_key not in source or phase_mean_key not in target:
                    continue
                deep_readout = self.model.readouts[readout_index]
                deep_shape = tuple(getattr(deep_readout, "spatial_shape", (9, 9)))
                phase_shape = tuple(
                    getattr(phase_readout, "spatial_shape", deep_shape)
                )
                scale = source[deep_mean_key].new_tensor(
                    [
                        (phase_shape[0] - 1) / max(deep_shape[0] - 1, 1),
                        (phase_shape[1] - 1) / max(deep_shape[1] - 1, 1),
                    ]
                )
                selected[phase_mean_key] = source[deep_mean_key] * scale
                phase_std_key = phase_prefix + ".std"
                if deep_std_key in source and phase_std_key in target:
                    selected[phase_std_key] = source[deep_std_key] * scale
                phase_theta_key = phase_prefix + ".theta"
                if deep_theta_key in source and phase_theta_key in target:
                    selected[phase_theta_key] = source[deep_theta_key]
                initialized_phase_readouts.add(phase_prefix)
        self._low_rank_migrated_readouts = migrated_low_rank_readouts
        self._phase_position_initialized_readouts = initialized_phase_readouts
        return selected, migrated_readouts

    @staticmethod
    def _gaussian_mask_from_parameters(mean, std, theta, height, width):
        """Pure-state equivalent of DynamicGaussianReadout's normalized mask."""
        device, dtype = mean.device, mean.dtype
        y = torch.linspace(-(height - 1) / 2.0, (height - 1) / 2.0, height, device=device, dtype=dtype)
        x = torch.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0)
        centered = grid - mean[:, None, None]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        rotation = torch.stack(
            (
                torch.stack((cos_theta, -sin_theta), dim=-1),
                torch.stack((sin_theta, cos_theta), dim=-1),
            ),
            dim=-2,
        )
        rotated = torch.einsum("nhwi,nij->nhwj", centered, rotation)
        exponent = -0.5 * ((rotated / std.clamp_min(1e-3)[:, None, None]) ** 2).sum(-1)
        mask = torch.exp(exponent)
        return mask / (mask.sum((-2, -1), keepdim=True) + 1e-8)

    def _load_pretrained_components(
        self,
        pretrained_checkpoint: str,
        freeze_vision: bool = False,
        load_heads: bool = False,
    ):
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

        source_cids = checkpoint.get("hyper_parameters", {}).get("dataset_cids")
        source_sessions = list(source_cids) if isinstance(source_cids, dict) else None
        if load_heads and isinstance(source_cids, dict):
            target_cids = self.hparams.dataset_cids
            for session in self.names:
                if session not in source_cids:
                    raise ValueError(
                        f"Pretrained checkpoint has no head for session {session!r}"
                    )
                if list(source_cids[session]) != list(target_cids[session]):
                    raise ValueError(
                        f"Pretrained and target cids differ for session {session!r}"
                    )

        selected_state, migrated_readouts = self._compatible_pretrained_state(
            pretrained_state_dict,
            load_heads=load_heads,
            source_sessions=source_sessions,
            target_sessions=self.names if source_sessions is not None else None,
        )
        self.model.load_state_dict(selected_state, strict=False)
        self._loaded_pretrained_heads = bool(load_heads)

        print(f"✓ Loaded {len(selected_state)} compatible pretrained parameters")
        if migrated_readouts:
            print(
                f"✓ Exactly embedded {len(migrated_readouts)} Gaussian readouts "
                "inside factorized sparse spatial maps"
            )
        if getattr(self, "_low_rank_migrated_readouts", None):
            print(
                f"✓ Exactly embedded {len(self._low_rank_migrated_readouts)} "
                "lower-rank sparse readouts in expanded low-rank heads"
            )
        if getattr(self, "_phase_position_initialized_readouts", None):
            print(
                f"✓ Initialized {len(self._phase_position_initialized_readouts)} "
                "zero-output shallow readouts at the pretrained RF locations"
            )

        prefixes = ["adapters", "frontend", "convnet"]
        if load_heads:
            prefixes.extend([
                "modulator",
                "readouts",
                "phase_readouts",
            ])
        component_counts = {}
        for key in selected_state:
            for prefix in prefixes:
                if key == prefix or key.startswith(prefix + "."):
                    component_counts[prefix] = component_counts.get(prefix, 0) + 1
                    break
        print(f"   Breakdown: {component_counts}")

        # Optionally freeze vision components
        if freeze_vision:
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in ["adapters", "frontend", "convnet"]
                ):
                    param.requires_grad = False
                    frozen_count += 1
            print(f"✓ Froze {frozen_count} vision parameters")

        return len(selected_state)

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

    def on_load_checkpoint(self, checkpoint):
        """Remember that learned heads, including their biases, were restored."""
        self._restored_from_checkpoint = True

    def on_fit_start(self):
        """Initialize readout biases from empirical firing rates."""
        super().on_fit_start()
        if self._loaded_pretrained_heads or self._restored_from_checkpoint:
            if self.global_rank == 0:
                print("Keeping restored readout biases")
            return
        dm = self.trainer.datamodule
        if not hasattr(dm, 'train_dsets') or not dm.train_dsets:
            return

        if self.global_rank == 0:
            print("Initializing readout biases from empirical firing rates...")
        for idx, name in enumerate(self.names):
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

    def on_train_epoch_start(self):
        """Set up for training epoch."""
        super().on_train_epoch_start()
        optimizer = self.optimizers()
        # if isinstance(optimizer, AdamWScheduleFree):
        #     optimizer.train()

    def forward(self, stim, ds_idx, beh=None, history=None):
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
            y = self.model(stimulus=None, dataset_idx=ds_idx, behavior=beh, history=history)
        else:
            y = self.model(stimulus=stim, dataset_idx=ds_idx, behavior=beh, history=history)
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
                rhat = self(stimulus, b["dataset_idx"][0], b.get("behavior"), b.get("history"))

            dfs = b.get("dfs").float()
            batch_loss = {
                'rhat': rhat.float(),
                'robs': b["robs"].float(),
                'dfs': dfs,
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
        """Reset BPS aggregators at the start of validation."""
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

        # Apply every proximal operator exactly once, after the optimizer step
        # and before gradients are cleared.  The gradient-presence check is how
        # homogeneous multisession training identifies the one active readout;
        # clearing gradients first silently disabled proximal sparsity.
        for reg in self.reg_terms:
            reg.prox(epoch, optimizer=optimizer)

        optimizer.zero_grad()

        return loss
