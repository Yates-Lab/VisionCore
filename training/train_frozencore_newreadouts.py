#!/usr/bin/env python3
"""
Training script for learning new readouts on frozen core components.

This script:
1. Loads a pretrained model from a checkpoint
2. Freezes core components (frontend, convnet, modulator, recurrent)
3. Rebuilds new adapters and readouts for new dataset configs
4. Trains only the new adapters and readouts

This is useful for transfer learning - keeping the learned visual representations
frozen while training new per-dataset components for new recording sessions.

Example:
    python training/train_frozencore_newreadouts.py \
        --pretrained_checkpoint /path/to/checkpoint.ckpt \
        --dataset_configs_path configs/new_datasets.yaml \
        --max_datasets 20 \
        --batch_size 64 \
        --learning_rate 1e-3 \
        --max_epochs 100 \
        --num_gpus 2
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping
)
from pytorch_lightning.loggers import WandbLogger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pl_modules import MultiDatasetDM
from training.pl_modules.frozencore_model import FrozenCoreModel
from training.callbacks import EpochHeartbeat, CurriculumCallback, ModelLoggingCallback

# Set PyTorch matmul precision
torch.set_float32_matmul_precision('medium')


def main():
    """Main training function."""
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    p = argparse.ArgumentParser(
        description='Train new readouts on frozen pretrained core',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required: pretrained checkpoint
    p.add_argument("--pretrained_checkpoint", type=str, required=True,
                   help="Path to pretrained checkpoint (or checkpoint directory)")
    p.add_argument("--model_type", type=str, default=None,
                   help="Model type to select from checkpoint directory (e.g., 'resnet_none_convgru')")
    p.add_argument("--model_index", type=int, default=None,
                   help="Model index within model_type (None = best model)")
    
    # Dataset configuration
    p.add_argument("--dataset_configs_path", type=str, required=True,
                   help="Path to NEW dataset configuration YAML file")
    p.add_argument("--max_datasets", type=int, default=30,
                   help="Maximum number of datasets/sessions to load")
    
    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size per GPU")
    p.add_argument("--learning_rate", type=float, default=1e-3,
                   help="Learning rate for adapters and readouts")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="Weight decay coefficient")
    p.add_argument("--max_epochs", type=int, default=100,
                   help="Maximum number of training epochs")
    p.add_argument("--gradient_clip_val", type=float, default=1.0,
                   help="Gradient clipping value")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Number of batches to accumulate gradients")
    
    # Learning rate scheduler
    p.add_argument("--lr_scheduler", type=str, default="cosine_warmup",
                   choices=["none", "step", "plateau", "cosine", "cosine_warmup", "cosine_warmup_restart"],
                   help="Learning rate scheduler type")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Number of warmup epochs")
    
    # Curriculum learning
    p.add_argument("--enable_curriculum", action="store_true", default=False,
                   help="Enable contrast-weighted curriculum learning")
    
    # Hardware and performance
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["16", "bf16", "32", "16-mixed", "bf16-mixed"],
                   help="Training precision")
    p.add_argument("--dset_dtype", type=str, default="bfloat16",
                   choices=["uint8", "bfloat16", "float32"],
                   help="Dataset storage dtype")
    p.add_argument("--num_gpus", type=int, default=2,
                   help="Number of GPUs to use")
    p.add_argument("--num_workers", type=int, default=16,
                   help="Number of dataloader workers")
    p.add_argument("--steps_per_epoch", type=int, default=1000,
                   help="Number of training steps per epoch")
    
    # Logging and checkpointing
    p.add_argument("--project_name", type=str, default="frozencore_readouts",
                   help="WandB project name")
    p.add_argument("--experiment_name", type=str, default=None,
                   help="Experiment name (auto-generated if not provided)")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                   help="Directory for saving checkpoints")
    
    # Early stopping
    p.add_argument("--early_stopping_patience", type=int, default=20,
                   help="Early stopping patience (epochs)")
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                   help="Minimum change to qualify as improvement")

    # Model logging
    p.add_argument("--enable_logging", action="store_true", default=False,
                   help="Enable periodic model logging")
    p.add_argument("--fast_log_interval", type=int, default=5,
                   help="Interval for fast logging")
    p.add_argument("--slow_log_interval", type=int, default=10,
                   help="Interval for slow logging")
    p.add_argument("--log_dataset_idx", type=int, default=0,
                   help="Dataset index for slow logging")

    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Experiment name
    # -------------------------------------------------------------------------
    if args.experiment_name is None:
        model_stem = args.model_type or Path(args.pretrained_checkpoint).stem
        args.experiment_name = (
            f"frozencore_{model_stem}"
            f"_bs{args.batch_size}_ds{args.max_datasets}"
            f"_lr{args.learning_rate}_wd{args.weight_decay}"
        )

    # -------------------------------------------------------------------------
    # Create DataModule
    # -------------------------------------------------------------------------
    dm = MultiDatasetDM(
        cfg_dir=args.dataset_configs_path,
        max_ds=args.max_datasets,
        batch=args.batch_size,
        workers=args.num_workers,
        steps_per_epoch=args.steps_per_epoch,
        enable_curriculum=args.enable_curriculum,
        dset_dtype=args.dset_dtype
    )

    # -------------------------------------------------------------------------
    # Create Model with frozen core and new readouts
    # -------------------------------------------------------------------------
    model = FrozenCoreModel(
        pretrained_checkpoint=args.pretrained_checkpoint,
        model_type=args.model_type,
        model_index=args.model_index,
        cfg_dir=args.dataset_configs_path,
        lr=args.learning_rate,
        wd=args.weight_decay,
        max_ds=args.max_datasets,
    )

    # Pass additional hyperparameters
    model.hparams.lr_scheduler = args.lr_scheduler
    model.hparams.warmup_epochs = args.warmup_epochs

    # -------------------------------------------------------------------------
    # Prepare data before initializing wandb (so failed data prep doesn't
    # create orphaned wandb runs)
    # -------------------------------------------------------------------------
    print("Preparing datasets...")
    dm.setup("fit")
    print(f"✓ Data preparation complete: {len(dm.train_dsets)} datasets loaded")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir) / args.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:02d}-{val_bps_overall:.4f}",
            monitor="val_bps_overall",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EpochHeartbeat(metric_key="train_loss"),
        EarlyStopping(
            monitor="val_bps_overall",
            mode="max",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            verbose=True,
            check_on_train_epoch_end=False
        )
    ]

    if args.enable_curriculum:
        callbacks.append(CurriculumCallback())

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

    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------
    logger = None
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger = WandbLogger(
            project=args.project_name,
            name=args.experiment_name,
            save_dir="./logs"
        )

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=0.05,
        num_sanity_val_steps=0,

        accelerator="gpu",
        devices=args.num_gpus,
        precision=args.precision,

        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        ),

        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,

        log_every_n_steps=50,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=logger,

        enable_progress_bar=False
    )

    # -------------------------------------------------------------------------
    # Print training info
    # -------------------------------------------------------------------------
    if trainer.global_rank == 0:
        print("=" * 60)
        print(f"FROZEN CORE READOUT TRAINING")
        print("=" * 60)
        print(f"Experiment: {args.experiment_name}")
        print(f"Pretrained checkpoint: {args.pretrained_checkpoint}")
        print(f"Model type: {args.model_type}")
        print(f"New dataset configs: {args.dataset_configs_path}")
        print(f"Max datasets: {args.max_datasets}")
        print(f"Frozen parameters: {model.frozen_count:,}")
        print(f"Trainable parameters: {model.trainable_count:,}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"LR scheduler: {args.lr_scheduler}")
        print(f"Curriculum learning: {args.enable_curriculum}")
        print("=" * 60, flush=True)

    # -------------------------------------------------------------------------
    # Train!
    # -------------------------------------------------------------------------
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

