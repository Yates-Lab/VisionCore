#!/usr/bin/env python3
"""
Multi-dataset V1 training

  python train_ddp_multidataset_auto.py \
      --model_config configs_multi/resnet_modulator.yaml \
      --dataset_configs_path /mnt/.../dataset_basic_multi \
      --max_datasets 20 \
      --batch_size 256 \
      --num_gpus 2 \
      --steps_per_epoch 1000
"""

# ---------------------------------------------------------------------#
#  Imports
# ---------------------------------------------------------------------#
import os, time, argparse, math, contextlib
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict

import torch, torch.nn as nn
import pytorch_lightning as pl

# Fix for NumPy 2.0 compatibility
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (ModelCheckpoint,
                                         LearningRateMonitor,
                                         EarlyStopping)
from pytorch_lightning.loggers import WandbLogger

# Custom warmup scheduler implementation (no external dependencies needed)
class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.warmup_start_lr = warmup_start_lr
            self.eta_min = eta_min
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                # Linear warmup
                return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                        for base_lr in self.base_lrs]
            else:
                # Cosine annealing
                return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                        for base_lr in self.base_lrs]
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from DataYatesV1.models.losses import MaskedLoss, PoissonBPSAggregator

# Import regularization functions (will be used in MultiDatasetModel)
try:
    from regularizers import create_regularizers, get_excluded_params_for_weight_decay
except ImportError as e:
    print(f"Warning: Failed to import regularizers: {e}")
    # Fallback for when regularizers.py doesn't exist yet
    def create_regularizers(*args, **kwargs):
        return []
    def get_excluded_params_for_weight_decay(*args, **kwargs):
        return []

torch.set_float32_matmul_precision('medium') # 'high'

# ---------------------------------------------------------------------#
#  Timing callbacks
#  These are here to keep the terminal active and let us have some sense
#  of how things are running because progress bar is disabled (for speed)
# ---------------------------------------------------------------------#
class Heartbeat(pl.Callback):
    """Emit a heartbeat at every major Lightning hook."""
    def __init__(self): self.rank = int(os.environ.get("LOCAL_RANK", 0))

    def _log(self, hook):                   # only rank-0 prints
        if self.rank == 0:
            t = time.strftime("%H:%M:%S")
            print(f"[{t}] {hook}", flush=True)

    def on_fit_start          (self, *a): self._log("fit-start")
    def on_train_start        (self, *a): self._log("train-start")
    def on_train_batch_start  (self, *a): self._log("train-batch-start")
    def on_after_backward     (self, *a): self._log("after-backward")
    def on_train_batch_end    (self, *a): self._log("train-batch-end")
    def on_validation_start   (self, *a): self._log("val-start")
    def on_validation_end     (self, *a): self._log("val-end")
    def on_validation_batch_start(self, *a): self._log("val-batch-start")
    def on_validation_batch_end  (self, *a): self._log("val-batch-end")
    def on_validation_epoch_start(self, *a): self._log("val-epoch-start")
    def on_validation_epoch_end  (self, *a): self._log("val-epoch-end")

class EpochHeartbeat(pl.Callback):
    """
    Rank-0 console heartbeat.
    Prints:
        • training begins        (once)
        • “… running validation” (each val loop)
        • “validation done”      (each val loop)
        • “epoch N — metric: x”  (after every train epoch)
    """
    def __init__(self, metric_key: str = "train_loss"):
        super().__init__()
        self.metric_key = metric_key
        self.rank = int(os.environ.get("LOCAL_RANK", 0))

    # ---------- helpers ------------------------------------------------
    def _print(self, msg: str):
        if self.rank == 0:
            stamp = time.strftime("[%H:%M:%S] ")
            print(stamp + msg, flush=True)

    # ---------- Lightning hooks ----------------------------------------
    def on_fit_start        (self, *_): self._print("training begins")
    def on_validation_start (self, *_): self._print("… running validation")
    def on_validation_end   (self, *_): self._print("validation done")

    def on_train_epoch_end(self, trainer, *_):
        metric = trainer.callback_metrics.get(self.metric_key)
        if metric is not None:
            val = metric.item() if torch.is_tensor(metric) else float(metric)
            self._print(f"epoch {trainer.current_epoch} — {self.metric_key}: {val:.4f}")
        else:
            self._print(f"epoch {trainer.current_epoch} finished")


# class PrintTimings(pl.Callback):
#     """
#     Prints a heartbeat every N steps during training.
#     """
#     def __init__(self, metric_name: str, every: int = 100):
#         self.metric_name = metric_name
#         self.every = every
#         self.prev = None

#     def _stamp(self, trainer, msg: str):
#         if trainer.global_rank == 0:
#             now = time.time()
#             if self.prev is None: self.prev = now
#             delta, self.prev = now - self.prev, now
#             print(f"[E{trainer.current_epoch}|GS{trainer.global_step}] {msg:<25} Δ{delta:4.2f}s",
#                   flush=True)

#     def on_train_batch_end(self, tr, *_):
#         if (tr.global_step + 1) % self.every == 0:
#             loss = tr.logged_metrics.get("train_loss", torch.tensor(float("nan")))
#             self._stamp(tr, f"step {tr.global_step+1}")

#     def on_validation_start(self, tr, *_): self._stamp(tr, "> validation")
#     def on_validation_end  (self, tr, *_): self._stamp(tr, "< validation")
#     def on_train_epoch_end (self, tr, *_): self._stamp(tr, "> epoch end")


class CurriculumCallback(pl.Callback):
    """
    Updates the contrast-weighted sampler with current training step.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Update sampler with current step for curriculum
        if hasattr(trainer.datamodule, 'train_dataloader'):
            train_loader = trainer.train_dataloader
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_step'):
                train_loader.sampler.set_step(trainer.global_step)

# ---------------------------------------------------------------------#
#  Data helpers
# ---------------------------------------------------------------------#
def cast_stim(x: torch.Tensor, norm_removed: bool, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = x.to(dtype)
    return (x - 127) / 255.0 if norm_removed else x

class Float32View(Dataset):
    def __init__(self, base, norm_removed: bool, float16: bool = False):
        self.base, self.norm_removed = base, norm_removed
        self.float16 = float16
    def __len__(self):  return len(self.base)
    def __getitem__(self, idx):
        it = self.base[idx]
        it["stim"] = cast_stim(it["stim"], self.norm_removed, dtype=torch.bfloat16 if self.float16 else torch.float32)
        it["robs"] = it["robs"].to(torch.bfloat16 if self.float16 else torch.float32)
        if "behavior" in it: it["behavior"] = it["behavior"].to(torch.bfloat16 if self.float16 else torch.float32)
        return it

def group_collate(batch):
    import torch.utils.data._utils.collate as _dc
    groups = defaultdict(list)
    for s in batch: groups[s["dataset_idx"]].append(s)
    return [_dc.default_collate(v) for v in groups.values()]


# ---------------------------------------------------------------------#
#  Contrast-Weighted Curriculum Sampler
# ---------------------------------------------------------------------#
class ContrastWeightedSampler(Sampler):
    """
    Distributed sampler with contrast-weighted curriculum learning.

    Early training: emphasizes high-contrast samples
    Late training: returns to unbiased sampling
    """

    def __init__(self, dataset, contrast_scores_dict, dataset_name_to_idx,
                 num_replicas=None, rank=None, shuffle=True, seed=0,
                 drop_last=False, warmup_steps=8000):
        """
        Args:
            dataset: ConcatDataset of tagged datasets
            contrast_scores_dict: {dataset_name: tensor of frame contrasts}
            dataset_name_to_idx: {dataset_name: dataset_idx}
            warmup_steps: Number of steps for curriculum (default: 8000)
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.warmup_steps = warmup_steps

        # Store pre-computed normalized contrast scores (already normalized!)
        self.normalized_contrasts = contrast_scores_dict

        # Map dataset names to indices
        self.name_to_idx = dataset_name_to_idx

        # Use pre-computed contrast mapping (no computation here!)
        self.sample_contrasts = self._build_contrast_mapping_from_cache()

        # Calculate dataset sizes for distributed sampling
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _build_contrast_mapping_from_cache(self):
        """Build mapping from ConcatDataset indices to cached contrast scores (FAST!)"""
        sample_contrasts = torch.zeros(len(self.dataset))

        # Get dataset cumulative sizes from ConcatDataset
        cumulative_sizes = self.dataset.cumulative_sizes

        for dataset_idx, cumulative_size in enumerate(cumulative_sizes):
            # Calculate start and end indices for this dataset
            dataset_start_idx = cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else 0
            dataset_end_idx = cumulative_size

            # Get dataset name from the Tag wrapper
            tag_dataset = self.dataset.datasets[dataset_idx]
            dataset_name = None
            for name, idx in self.name_to_idx.items():
                if idx == tag_dataset.idx:
                    dataset_name = name
                    break

            if dataset_name and dataset_name in self.normalized_contrasts:
                contrasts = self.normalized_contrasts[dataset_name]
                dataset_size = dataset_end_idx - dataset_start_idx

                # Vectorized assignment (FAST!)
                if len(contrasts) >= dataset_size:
                    sample_contrasts[dataset_start_idx:dataset_end_idx] = contrasts[:dataset_size]
                else:
                    # Handle size mismatch
                    sample_contrasts[dataset_start_idx:dataset_start_idx + len(contrasts)] = contrasts
                    sample_contrasts[dataset_start_idx + len(contrasts):dataset_end_idx] = 1.0
            else:
                # Fallback for missing dataset
                sample_contrasts[dataset_start_idx:dataset_end_idx] = 1.0

        return sample_contrasts

    def _compute_weights_vectorized(self, step):
        """Compute curriculum weights using vectorized operations (FAST!)"""
        if step >= self.warmup_steps:
            return torch.ones_like(self.sample_contrasts)  # Uniform after warmup
        else:
            α = 0.5 + 0.5 * (step / self.warmup_steps)
            return torch.clamp(α * self.sample_contrasts, max=1.0)  # Vectorized!

    def set_epoch(self, epoch):
        """Set epoch for distributed training"""
        self.epoch = epoch

    def set_step(self, step):
        """Set current training step for curriculum"""
        self.current_step = step

    def __iter__(self):
        # Get current step from trainer if available
        current_step = getattr(self, 'current_step', 0)

        if self.shuffle:
            # Generate deterministic random state
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            if current_step >= self.warmup_steps:
                # Unbiased sampling after warmup (fast!)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                # Contrast-weighted sampling with vectorized computation (FAST!)
                weights = self._compute_weights_vectorized(current_step)

                # Sample with replacement using computed weights
                indices = torch.multinomial(
                    weights, len(self.dataset), replacement=True, generator=g
                ).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples

# ---------------------------------------------------------------------#
#  LightningDataModule
# ---------------------------------------------------------------------#
class MultiDatasetDM(pl.LightningDataModule):
    def __init__(self, cfg_dir: str, max_ds: int, batch: int,
                 workers: int, steps_per_epoch: int, enable_curriculum: bool = False):
        super().__init__()
        self.cfg_dir, self.max_ds = cfg_dir, max_ds
        self.batch, self.workers, self.spe = batch, workers, steps_per_epoch
        self.enable_curriculum = enable_curriculum
        self.contrast_scores = None
        self.name2idx = None

    def setup(self, stage: Optional[str] = None):
        from DataYatesV1.models.config_loader import load_dataset_configs
        from DataYatesV1.utils.data.loading   import remove_pixel_norm
        from DataYatesV1.utils.data           import prepare_data

        yaml_files = sorted([f for f in os.listdir(self.cfg_dir)
                             if f.endswith(".yaml") and "base" not in f])[: self.max_ds]
        self.names = [Path(f).stem for f in yaml_files]
        self.cfgs  = load_dataset_configs(yaml_files, self.cfg_dir)

        self.train_dsets, self.val_dsets, self.name2idx = {}, {}, {}
        for idx, (cfg, name) in enumerate(zip(self.cfgs, self.names)):
            cfg["_dataset_name"] = name
            cfg, norm_removed = remove_pixel_norm(cfg)
            with open(os.devnull, "w") as _null, \
                 contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
                tr, va, _ = prepare_data(cfg, strict=False)
            self.train_dsets[name] = Float32View(tr, norm_removed, float16=False)
            self.val_dsets[name]   = Float32View(va, norm_removed, float16=False)
            self.name2idx[name]    = idx
        print(f"✓ loaded {len(self.train_dsets)} datasets")

        # Precompute contrast scores for curriculum learning
        if self.enable_curriculum:
            self._precompute_contrast_scores()

    def _precompute_contrast_scores(self):
        """Precompute and cache contrast scores for all datasets (ONCE!)"""
        print("Computing contrast scores for curriculum learning...")
        start_time = time.time()

        self.contrast_scores = {}
        all_contrasts_for_median = []
        total_frames = 0

        # Step 1: Compute raw contrast scores for all datasets
        for dataset_name, train_dataset in self.train_dsets.items():
            try:
                # Access the CombinedEmbeddedDataset
                combined_dataset = train_dataset.base

                # Compute contrast for all dsets within this dataset
                dataset_contrasts = []

                for dset_idx, dset in enumerate(combined_dataset.dsets):
                    raw_stim = dset['stim']

                    # Fast per-frame contrast computation (vectorized!)
                    frame_contrasts = raw_stim.to(torch.float16).std(dim=(1,2,3))
                    dataset_contrasts.append(frame_contrasts)
                    total_frames += len(frame_contrasts)

                # Concatenate all contrasts for this dataset
                if dataset_contrasts:
                    dataset_contrasts = torch.cat(dataset_contrasts)
                    self.contrast_scores[dataset_name] = dataset_contrasts
                    all_contrasts_for_median.append(dataset_contrasts)

                    print(f"  {dataset_name}: {len(dataset_contrasts)} frames, "
                          f"range [{dataset_contrasts.min():.3f}, {dataset_contrasts.max():.3f}]")
                else:
                    # Fallback if no dsets found
                    dataset_size = len(train_dataset)
                    fallback_contrasts = torch.ones(dataset_size)
                    self.contrast_scores[dataset_name] = fallback_contrasts
                    all_contrasts_for_median.append(fallback_contrasts)

            except Exception as e:
                print(f"  Warning: Failed to compute contrast for {dataset_name}: {e}")
                # Fallback to uniform contrasts
                dataset_size = len(train_dataset)
                fallback_contrasts = torch.ones(dataset_size)
                self.contrast_scores[dataset_name] = fallback_contrasts
                all_contrasts_for_median.append(fallback_contrasts)

        # Step 2: Compute global median and normalize (as you specified!)
        all_contrasts = torch.cat(all_contrasts_for_median)
        global_median = all_contrasts.median()

        # Normalize so median = 1.0 (your specification!)
        for dataset_name in self.contrast_scores:
            self.contrast_scores[dataset_name] = self.contrast_scores[dataset_name] / global_median

        elapsed = time.time() - start_time
        print(f"✓ Computed contrast for {total_frames:,} frames in {elapsed:.2f} seconds")
        print(f"✓ Global median contrast: {global_median:.3f}, normalized to 1.0")

    # -----------------------------------------------------------------
    def _mk_loader(self, dsets: Dict[str, Dataset], shuffle: bool):
        class Tag(Dataset):
            def __init__(self, ds, idx): self.ds, self.idx = ds, idx
            def __len__(self):  return len(self.ds)
            def __getitem__(self, i):
                it = self.ds[i]; it["dataset_idx"] = self.idx; return it

        tagd = [Tag(ds, self.name2idx[n]) for n, ds in dsets.items()]
        cat  = torch.utils.data.ConcatDataset(tagd)
    
        # -------- rank-aware sampler for *both* train & val ------------
        if torch.distributed.is_initialized():
            if shuffle and self.enable_curriculum and self.contrast_scores is not None:
                # Use contrast-weighted sampler for training
                sampler = ContrastWeightedSampler(
                    cat,
                    self.contrast_scores,
                    self.name2idx,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True,
                    drop_last=True,
                    warmup_steps=8000
                )
            else:
                # Use standard distributed sampler (same as old code)
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(
                    cat,
                    shuffle=shuffle,
                    drop_last=True,
                )
        else:
            sampler = None # single-GPU debug run

        return DataLoader(cat, batch_size=self.batch, sampler=sampler,
                          num_workers=self.workers, pin_memory=True,
                          drop_last=True,
                          persistent_workers=False, collate_fn=group_collate)


    def train_dataloader(self): return self._mk_loader(self.train_dsets, True)
    def val_dataloader  (self): return self._mk_loader(self.val_dsets,   True)

# ---------------------------------------------------------------------#
#  LightningModule
# ---------------------------------------------------------------------#
class MultiDatasetModel(pl.LightningModule):
    def __init__(self, model_cfg: str, cfg_dir: str, lr: float, wd: float,
                 max_ds: int, pretrained_checkpoint: str = None,
                 freeze_vision: bool = False, compile_model: bool = False):
        super().__init__()
        self.save_hyperparameters()

        from DataYatesV1.models.config_loader import load_dataset_configs, load_config
        from DataYatesV1.models                import build_model

        yaml = sorted([f for f in os.listdir(cfg_dir)
                       if f.endswith(".yaml") and "base" not in f])[: max_ds]
        self.names = [Path(f).stem for f in yaml]
        self.cfgs = load_dataset_configs(yaml, cfg_dir)
        for c, n in zip(self.cfgs, self.names): c["_dataset_name"] = n
        # self.model = torch.compile(build_model(load_config(model_cfg), self.cfgs).half())
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
        self.head_lr       = lr  # unchanged for dataset heads
        self.log_input = isinstance(self.model.activation, nn.Identity)
        self.loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=self.log_input, reduction="none"))
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
                print(f" Non-PC modulator detected: {modulator_type}")
        else:
            print(" No modulator detected in model")

    def _load_pretrained_components(self, pretrained_checkpoint: str, freeze_vision: bool = False):
        """Load pretrained vision components from a checkpoint."""
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
            print(f"Froze {frozen_count} vision parameters")

        return len(vision_state_dict)

    def _compute_auxiliary_loss(self):
        """
        Compute auxiliary loss for PC modulator if present.

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
                elif hasattr(self, 'model_config'):
                    lambda_pred = self.model_config.get('lambda_pred', lambda_pred)

                # Compute L2 loss on prediction error
                pred_err = self.model.modulator.pred_err
                aux_loss = lambda_pred * (pred_err ** 2).mean()
                return aux_loss

        return None

    # Forward
    def forward(self, stim, ds_idx, beh=None):
        # Check if this is a modulator-only model (no vision processing)
        if self.is_modulator_only:
            y = self.model(stimulus=None, dataset_idx=ds_idx, behavior=beh)
        else:
            y = self.model(stim, ds_idx, beh)
        return torch.clamp(y, min=-20 if self.log_input else 1e-8)

    # Step
    # def _step(self, batch_list, tag: str):
    #     import time
    #     step_start = time.time()
        
    #     losses = []
    #     for i, b in enumerate(batch_list):
    #         batch_start = time.time()
    #         b = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
    #         data_move_time = time.time() - batch_start
            
    #         forward_start = time.time()
    #         batch_loss = {'rhat': self(b["stim"], b["dataset_idx"][0], b.get("behavior")),
    #                     'robs': b["robs"].float(), 'dfs': b.get("dfs")}
    #         forward_time = time.time() - forward_start
            
    #         loss_start = time.time()
    #         loss = self.loss_fn(batch_loss)
    #         loss_time = time.time() - loss_start
            
    #         if self.global_rank == 0 and i == 0:  # Only log first batch
    #             print(f"Batch {i}: data_move={data_move_time:.3f}s, forward={forward_time:.3f}s, loss={loss_time:.3f}s")

    #         losses.append(loss)
        
    #     step_time = time.time() - step_start
    #     if self.global_rank == 0:
    #         print(f"Total step time: {step_time:.3f}s")
        
    #     return torch.stack(losses).mean()
    
    def _step(self, batch_list, tag: str):
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
            # with torch.autocast(device_type='cuda', enabled=False):
            # if val, no gradients
            with torch.no_grad() if tag == "val" else contextlib.nullcontext():
                # Pass stimulus or None based on model type
                stimulus = None if self.is_modulator_only else b["stim"]
                rhat = self(stimulus, b["dataset_idx"][0], b.get("behavior"))
                
            batch_loss = {'rhat': rhat.float(),
                                'robs': b["robs"].float(), 'dfs': b.get("dfs").float()}
            
            loss = self.loss_fn(batch_loss)
            # ----- inside _step -------------------------------------------------
            if torch.isfinite(loss):
                if self.global_rank == 0: # only log from rank 0 during steps
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
        self._step(bl, "val")

    # --------- reset aggregators at each val loop --------------------
    def on_validation_epoch_start(self):
        for agg in self.bps_aggs: agg.reset()

    # ------------- compute & log at epoch end ------------------------
    def on_validation_epoch_end(self):
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
            if len(agg.robs) == 0: # aggregator is empty
                local_sum   = torch.tensor(0.0, device=self.device)
                local_count = torch.tensor(0,   device=self.device)
            else:
                bps = agg.closure().clamp_min(0.0)  # (units,)
                local_sum   = bps.sum().to(self.device)
                local_count = torch.tensor(bps.numel(), device=self.device)

             # b) global reduction (all-reduce)
            global_sum   = self.trainer.strategy.reduce(local_sum,   reduce_op="sum")
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
        # Get parameters that should be excluded from weight decay due to regularization conflicts
        excluded_param_names = set(get_excluded_params_for_weight_decay(self.reg_terms))

        # -------- split params into "core" vs "head" with different learning rates ------------------------
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

        # -------- learning rate scheduler ------------------------
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
        """Override optimizer step to apply proximal updates after gradient step."""
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

# ---------------------------------------------------------------------#
#  Main
# ---------------------------------------------------------------------#
def main():
    p = argparse.ArgumentParser()
    # CLI identical to your shell script
    p.add_argument("--model_config",        type=str, required=True)
    p.add_argument("--dataset_configs_path",type=str, required=True)
    p.add_argument("--max_datasets",        type=int, default=30)
    p.add_argument("--batch_size",          type=int, default=64)
    p.add_argument("--learning_rate",       type=float, default=1e-4)
    p.add_argument("--core_lr_scale",       type=float, default=1.0,
                    help="core-lr = learning_rate * core_lr_scale (default=1.0)")
    p.add_argument("--weight_decay",        type=float, default=1e-5)
    p.add_argument("--max_epochs",          type=int, default=100)
    p.add_argument("--precision",           type=str, default="bf16",
                    choices=["16","bf16","32","16-mixed","bf16-mixed"])
    p.add_argument("--num_gpus",            type=int, default=2)
    p.add_argument("--num_workers",         type=int, default=16)
    p.add_argument("--steps_per_epoch",     type=int, default=1000)
    p.add_argument("--project_name",        type=str, default="multidataset")
    p.add_argument("--experiment_name",     type=str, default=None)
    p.add_argument("--checkpoint_dir",      type=str, default="./checkpoints")
    p.add_argument("--lr_scheduler",        type=str, default="none",
                    choices=["none", "step", "plateau", "cosine", "cosine_warmup"],
                    help="Learning rate scheduler type")
    p.add_argument("--warmup_epochs",       type=int, default=5,
                    help="Number of warmup epochs for learning rate (only used with cosine_warmup scheduler)")
    p.add_argument("--enable_curriculum",      action="store_true", default=False,
                    help="Enable contrast-weighted curriculum learning")
    # Pretraining options
    p.add_argument("--pretrained_checkpoint",  type=str, default=None,
                    help="Path to pretrained checkpoint for vision components")
    p.add_argument("--freeze_vision",          action="store_true", default=False,
                    help="Freeze pretrained vision components (adapter, frontend, convnet)")
    # Compilation options
    p.add_argument("--compile",                action="store_true", default=False,
                    help="Enable torch.compile for model compilation")
    # passthroughs (ignored but kept for compatibility)
    p.add_argument("--gradient_clip_val",      type=float, default=1.0)
    p.add_argument("--accumulate_grad_batches",type=int,   default=1)
    p.add_argument("--early_stopping_patience",type=int,   default=10)
    p.add_argument("--early_stopping_min_delta",type=float,default=0.0)
    args = p.parse_args()

    # experiment name
    if args.experiment_name is None:
        args.experiment_name = (f"{Path(args.model_config).stem}"
                                f"_ddp_bs{args.batch_size}_ds{args.max_datasets}"
                                f"_lr{args.learning_rate}_wd{args.weight_decay}")

    # datamodule & model
    dm = MultiDatasetDM(args.dataset_configs_path, args.max_datasets,
                        args.batch_size, args.num_workers, args.steps_per_epoch,
                        enable_curriculum=args.enable_curriculum)
    model = MultiDatasetModel(args.model_config, args.dataset_configs_path,
                              args.learning_rate, args.weight_decay,
                              args.max_datasets,
                              pretrained_checkpoint=args.pretrained_checkpoint,
                              freeze_vision=args.freeze_vision,
                              compile_model=args.compile)
    
    # Pass core_lr_scale to the model's hparams
    model.hparams.core_lr_scale = args.core_lr_scale
    model.hparams.lr_scheduler = args.lr_scheduler
    model.hparams.warmup_epochs = args.warmup_epochs

    # callbacks / logger
    ckpt_dir = Path(args.checkpoint_dir) / args.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cbs = [ModelCheckpoint(dirpath=str(ckpt_dir),
                           filename="{epoch:02d}-{val_bps_overall:.4f}",
                           monitor="val_bps_overall", mode="max", save_top_k=3, save_last=True),
           LearningRateMonitor("epoch"),
           EpochHeartbeat("train_loss"),
            # Heartbeat(),
           EarlyStopping(monitor="val_bps_overall", mode="max",
                         patience=args.early_stopping_patience,
                         min_delta=args.early_stopping_min_delta,
                         verbose=True,
                         check_on_train_epoch_end=False)]  # Only check after validation

    # Add curriculum callback if enabled
    if args.enable_curriculum:
        cbs.append(CurriculumCallback())
    logger = (WandbLogger(project=args.project_name, name=args.experiment_name,
                          save_dir="./logs")
              if int(os.environ.get("LOCAL_RANK", 0)) == 0 else None)

    # trainer with sanity-tested DDP flags
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=args.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True,
                             gradient_as_bucket_view=True),
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=0.05,
        precision=args.precision, log_every_n_steps=50,
        gradient_clip_val=args.gradient_clip_val,  # Actually use the gradient clipping
        accumulate_grad_batches=args.accumulate_grad_batches,  # Use gradient accumulation
        callbacks=cbs, logger=logger, val_check_interval=1.0,
        enable_progress_bar=False)

    if trainer.global_rank == 0:
        print("="*60)
        print("Starting:", args.experiment_name)
        print("="*60, flush=True)
        # print accumualte grad number
        print("accumulate_grad_batches =", trainer.accumulate_grad_batches)
        print("num_training_batches  =", trainer.num_training_batches)
    
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
