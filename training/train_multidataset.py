#!/usr/bin/env python3
"""
Single‑GPU training script for multi‑dataset neural encoding models.

Optimized for single GPU throughput (no DDP). Lets you pin a specific GPU via --gpu.

Features:
- Single‑GPU training (no distributed init, minimal overhead)
- Multi‑dataset training with separate readouts per dataset
- Optional curriculum learning (same callback; no special sampler on 1 GPU)
- Optional torch.compile for core path
- WandB logging and checkpointing
- Early stopping and LR monitoring

Example:
    python training/train_multidataset.py \
        --model_config configs/model.yaml \
        --dataset_configs_path configs/datasets \
        --batch_size 64 --gpu 0 --precision bf16
"""

import os
import sys
import argparse
import copy
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from modular training package
from training.pl_modules import MultiDatasetModel, MultiDatasetDM
from training.callbacks import EpochHeartbeat, CurriculumCallback, ModelLoggingCallback
from training.utils import group_collate

# Set PyTorch matmul precision
torch.set_float32_matmul_precision("medium")
# Enable cudnn autotuner for faster convs on fixed-size inputs
torch.backends.cudnn.benchmark = True


def apply_stimulus_sampling_weight_overrides(model_config, overrides):
    """Return a copied config with ``NAME=WEIGHT`` sampling overrides.

    Sampling and loss weighting are deliberately separate controls: sampling
    changes how often distinct examples from a stimulus bank enter optimizer
    updates, whereas loss weighting only rescales an example already drawn.
    """
    resolved = copy.deepcopy(model_config)
    weights = resolved.setdefault("stimulus_sampling_weights", {})
    for specification in overrides or ():
        if "=" not in specification:
            raise ValueError(
                "--stimulus_sampling_weight must use NAME=WEIGHT, got "
                f"{specification!r}"
            )
        name, raw_weight = specification.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("stimulus sampling-weight name cannot be empty")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid stimulus sampling weight {raw_weight!r}"
            ) from exc
        if weight <= 0:
            raise ValueError("stimulus sampling weights must be positive")
        weights[name] = weight
    return resolved





def main():
    """Main training function for single‑GPU runs."""
    # ---------------------------------------------------------------------
    # Argument parsing
    # ---------------------------------------------------------------------
    p = argparse.ArgumentParser(
        description="Train multi‑dataset neural encoding models on a single GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and data
    p.add_argument("--model_config", type=str, required=True,
                   help="Path to model configuration YAML file")
    p.add_argument("--dataset_configs_path", type=str, required=True,
                   help="Path to parent dataset configuration YAML file (specifies sessions)")
    p.add_argument("--max_datasets", type=int, default=30,
                   help="Maximum number of datasets/sessions to load")
    p.add_argument(
        "--session",
        action="append",
        default=None,
        help="Use only this exact session (repeatable).",
    )
    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Learning rate for readout heads")
    p.add_argument("--core_lr_scale", type=float, default=1.0,
                   help="Learning rate scale for core (frontend/convnet/modulator)")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="Weight decay coefficient")
    p.add_argument(
        "--stimulus_sampling_weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help=(
            "Override how often a named stimulus bank is sampled during "
            "training. May be repeated; resolved values are saved in the "
            "checkpoint and do not alter validation sampling."
        ),
    )
    p.add_argument("--max_epochs", type=int, default=100,
                   help="Maximum number of training epochs")
    p.add_argument("--gradient_clip_val", type=float, default=1.0,
                   help="Gradient clipping value")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Number of batches to accumulate gradients")

    # Learning rate scheduler
    p.add_argument("--lr_scheduler", type=str, default="none",
                   choices=["none", "step", "plateau", "cosine", "cosine_warmup", "cosine_warmup_restart"],
                   help="Learning rate scheduler type")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Number of warmup epochs (for cosine_warmup and cosine_warmup_restart schedulers)")
    p.add_argument("--restart_period", type=int, default=None,
                   help="Restart period in epochs for cosine_warmup_restart scheduler (default: half of remaining epochs after warmup)")

    # Curriculum learning and batching
    p.add_argument("--enable_curriculum", action="store_true", default=False,
                   help="Enable curriculum learning (uses callback; works with ByDatasetBatchSampler on 1 GPU)")
    # BooleanOptionalAction, not store_true: with store_true and default=True
    # there was no way to switch this off, so the cross-session batching path
    # (the one the paper model was trained under, via the DDP script) was
    # unreachable from this script. Use --no-homogeneous_batches to disable.
    p.add_argument("--homogeneous_batches", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Yield one dataset per batch via ByDatasetBatchSampler (faster on 1 GPU)")

    # Pretrained models
    p.add_argument("--pretrained_checkpoint", type=str, default=None,
                   help="Path to pretrained checkpoint for vision components")
    p.add_argument("--freeze_vision", action="store_true", default=False,
                   help="Freeze pretrained vision components")
    p.add_argument(
        "--pretrained_load_heads",
        action="store_true",
        default=False,
        help=(
            "Also warm-start compatible behavior/readout parameters. An ordinary "
            "Gaussian readout is exactly embedded in a sparse-Gaussian readout."
        ),
    )

    # Model compilation
    p.add_argument("--compile", action="store_true", default=False,
                   help="Enable torch.compile for model compilation")

    # Hardware and performance
    p.add_argument("--precision", type=str, default="bf16",
                   choices=["16", "bf16", "32", "16-mixed", "bf16-mixed"],
                   help="Training precision")
    p.add_argument("--dset_dtype", type=str, default="uint8",
                   choices=["uint8", "bfloat16", "float32"],
                   help="Dataset storage dtype in CPU RAM (uint8=1x, bfloat16=2x, float32=4x memory)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use (e.g., 0 or 1)")
    p.add_argument("--num_workers", type=int, default=16,
                   help="Number of dataloader workers")
    # Validation cost control. This script previously hard-coded
    # limit_val_batches=1.0, so one validation pass ran the entire validation
    # set -- ~5900 batches, ~22 min, every epoch. That dominates any short run.
    # Defaults here reproduce the old behaviour exactly.
    p.add_argument("--limit_val_batches", type=float, default=1.0,
                   help="Fraction of validation batches per validation pass")
    p.add_argument("--check_val_every_n_epoch", type=int, default=1,
                   help="Run validation every N epochs")
    p.add_argument("--steps_per_epoch", type=int, default=1000,
                   help="Number of training steps per epoch")

    # Logging and checkpointing
    p.add_argument("--project_name", type=str, default="multidataset",
                   help="WandB project name")
    p.add_argument("--experiment_name", type=str, default=None,
                   help="Experiment name (auto-generated if not provided)")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                   help="Directory for saving checkpoints")

    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Checkpoint to resume from (restores optimizer, "
                        "scheduler and epoch). Default None starts fresh.")
    # Default None leaves the run unseeded, which is the historical behaviour.
    # Note what this does and does not control: the trial splits re-seed the
    # global RNG themselves inside `split_inds_by_trial*` (splitting.py), so
    # they are fixed at SPLIT_SEED regardless of this flag, and so is the batch
    # order that follows data preparation. Two runs differing only in --seed
    # therefore differ in weight initialisation and GPU nondeterminism, not in
    # which trials they see or the order they see them in.
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for weight init (see note in source). None = unseeded.")

    # Early stopping
    # Patience counts *validation calls*, not epochs, so with
    # --check_val_every_n_epoch N the effective horizon is patience * N epochs.
    # Sample-budgeted comparison runs want no early stopping at all: stopping
    # mid-cosine leaves the learning rate un-annealed and the arms no longer
    # compute-matched. Use --no-early_stopping for those.
    p.add_argument("--early_stopping", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Enable early stopping on val_bps_overall")
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Early stopping patience (validation calls)")
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                   help="Minimum change to qualify as improvement")

    # Model logging
    p.add_argument("--enable_logging", action="store_true", default=False,
                   help="Enable periodic model logging (kernels + eval stack)")
    p.add_argument("--fast_log_interval", type=int, default=5,
                   help="Interval for fast logging (kernel visualizations)")
    p.add_argument("--slow_log_interval", type=int, default=10,
                   help="Interval for slow logging (evaluation stack)")
    p.add_argument("--log_dataset_idx", type=int, default=7,
                   help="Dataset index to evaluate during slow logging")

    args = p.parse_args()

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # ---------------------------------------------------------------------
    # Experiment name
    # ---------------------------------------------------------------------
    if args.experiment_name is None:
        args.experiment_name = (
            f"{Path(args.model_config).stem}"
            f"_sgpu_bs{args.batch_size}_ds{args.max_datasets}"
            f"_lr{args.learning_rate}_wd{args.weight_decay}"
        )

    # ---------------------------------------------------------------------
    # Create DataModule and Model
    # ---------------------------------------------------------------------
    from models.config_loader import load_config
    model_config_dict = load_config(args.model_config)
    if args.stimulus_sampling_weight:
        model_config_dict = apply_stimulus_sampling_weight_overrides(
            model_config_dict, args.stimulus_sampling_weight
        )

    # Single-GPU DataModule
    dm = MultiDatasetDM(
        cfg_dir=args.dataset_configs_path,
        max_ds=args.max_datasets,
        batch=args.batch_size,
        workers=args.num_workers,
        steps_per_epoch=args.steps_per_epoch,
        enable_curriculum=args.enable_curriculum,
        dset_dtype=args.dset_dtype,
        homogeneous_batches=args.homogeneous_batches,
        stimulus_sampling_weights=model_config_dict.get(
            "stimulus_sampling_weights", {}
        ),
        dataset_sampling=model_config_dict.get(
            "dataset_sampling", "proportional"
        ),
        selected_sessions=args.session,
    )

    model = MultiDatasetModel(
        model_cfg=args.model_config,
        cfg_dir=args.dataset_configs_path,
        lr=args.learning_rate,
        wd=args.weight_decay,
        max_ds=args.max_datasets,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_vision=args.freeze_vision,
        compile_model=args.compile,
        pretrained_load_heads=args.pretrained_load_heads,
        core_lr_scale=args.core_lr_scale,
        model_config_dict=model_config_dict,
        selected_sessions=args.session,
    )

    # Pass additional hyperparameters to model
    model.hparams.core_lr_scale = args.core_lr_scale
    model.hparams.lr_scheduler = args.lr_scheduler
    model.hparams.warmup_epochs = args.warmup_epochs
    model.hparams.restart_period = args.restart_period

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir) / args.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Checkpoint saving
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:02d}-{val_bps_overall:.4f}",
            monitor="val_bps_overall",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
        # Epoch heartbeat
        EpochHeartbeat(metric_key="train_loss"),
    ]

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_bps_overall",
                mode="max",
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                verbose=True,
                check_on_train_epoch_end=False,
            )
        )

    # Add curriculum callback if enabled (no‑op unless sampler supports set_step)
    if args.enable_curriculum:
        callbacks.append(CurriculumCallback())

    # Add model logging callback if enabled
    if args.enable_logging:
        callbacks.append(
            ModelLoggingCallback(
                fast_interval=args.fast_log_interval,
                slow_interval=args.slow_log_interval,
                eval_dataset_idx=args.log_dataset_idx,
                batch_size=64,
                rescale=True
            )
        )

    # ---------------------------------------------------------------------
    # Logger
    # ---------------------------------------------------------------------
    logger = WandbLogger(
        project=args.project_name,
        name=args.experiment_name,
        save_dir="./logs",
    )

    # ---------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------
    # Use Single‑device strategy automatically; bind to the requested GPU index.
    trainer = pl.Trainer(
        # Training duration
        max_epochs=args.max_epochs,
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=0,

        # Hardware
        accelerator="gpu",
        devices=[args.gpu],  # pin to a specific GPU index
        precision=args.precision,
        strategy="auto",   # no DDP overhead for single‑GPU

        # Optimization
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,

        # Logging and callbacks
        log_every_n_steps=50,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=logger,

        # UI
        enable_progress_bar=True,
    )

    # ---------------------------------------------------------------------
    # Print training info
    # ---------------------------------------------------------------------
    print("=" * 60)
    print(f"Starting: {args.experiment_name}")
    print("=" * 60)
    print(f"GPU index: {args.gpu}")
    print(f"Accumulate grad batches: {trainer.accumulate_grad_batches}")
    print(f"Num training batches: {trainer.num_training_batches}")
    print(f"Curriculum learning: {args.enable_curriculum}")
    print(f"Pretrained checkpoint: {args.pretrained_checkpoint or 'None'}")
    print(f"Pretrained compatible heads: {args.pretrained_load_heads}")
    print(f"Freeze vision: {args.freeze_vision}")
    print("=" * 60, flush=True)

    # ---------------------------------------------------------------------
    # Train!
    # ---------------------------------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
