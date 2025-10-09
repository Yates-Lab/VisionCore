#!/usr/bin/env python3
"""
Multi-dataset distributed training script for neural encoding models.

This script trains neural encoding models on multiple datasets using
PyTorch Lightning with distributed data parallel (DDP) training.

Features:
- Multi-dataset training with separate readouts per dataset
- Distributed training across multiple GPUs
- Curriculum learning with contrast-weighted sampling
- Pretrained vision components with optional freezing
- Regularization (L1, L2, group lasso)
- Predictive coding modulators with auxiliary loss
- Bits-per-spike (BPS) validation metrics
- WandB logging and checkpointing

Example:
    python training/train_ddp_multidataset.py \
        --model_config configs/model.yaml \
        --dataset_configs_path configs/datasets \
        --max_datasets 20 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --weight_decay 1e-5 \
        --max_epochs 100 \
        --num_gpus 2 \
        --enable_curriculum
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

# Import from modular training package
from training.pl_modules import MultiDatasetModel, MultiDatasetDM
from training.callbacks import EpochHeartbeat, CurriculumCallback

# Set PyTorch matmul precision
torch.set_float32_matmul_precision('medium')


def main():
    """Main training function."""
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    p = argparse.ArgumentParser(
        description='Train multi-dataset neural encoding models with DDP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    p.add_argument("--model_config", type=str, required=True,
                   help="Path to model configuration YAML file")
    p.add_argument("--dataset_configs_path", type=str, required=True,
                   help="Directory containing dataset configuration YAML files")
    p.add_argument("--max_datasets", type=int, default=30,
                   help="Maximum number of datasets to load")
    
    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size per GPU")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Learning rate for readout heads")
    p.add_argument("--core_lr_scale", type=float, default=1.0,
                   help="Learning rate scale for core (frontend/convnet/modulator)")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="Weight decay coefficient")
    p.add_argument("--max_epochs", type=int, default=100,
                   help="Maximum number of training epochs")
    p.add_argument("--gradient_clip_val", type=float, default=1.0,
                   help="Gradient clipping value")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Number of batches to accumulate gradients")
    
    # Learning rate scheduler
    p.add_argument("--lr_scheduler", type=str, default="none",
                   choices=["none", "step", "plateau", "cosine", "cosine_warmup"],
                   help="Learning rate scheduler type")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Number of warmup epochs (for cosine_warmup scheduler)")
    
    # Curriculum learning
    p.add_argument("--enable_curriculum", action="store_true", default=False,
                   help="Enable contrast-weighted curriculum learning")
    
    # Pretrained models
    p.add_argument("--pretrained_checkpoint", type=str, default=None,
                   help="Path to pretrained checkpoint for vision components")
    p.add_argument("--freeze_vision", action="store_true", default=False,
                   help="Freeze pretrained vision components")
    
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
    p.add_argument("--num_gpus", type=int, default=2,
                   help="Number of GPUs to use")
    p.add_argument("--num_workers", type=int, default=16,
                   help="Number of dataloader workers")
    p.add_argument("--steps_per_epoch", type=int, default=1000,
                   help="Number of training steps per epoch")
    
    # Logging and checkpointing
    p.add_argument("--project_name", type=str, default="multidataset",
                   help="WandB project name")
    p.add_argument("--experiment_name", type=str, default=None,
                   help="Experiment name (auto-generated if not provided)")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                   help="Directory for saving checkpoints")
    
    # Early stopping
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                   help="Minimum change to qualify as improvement")
    
    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Experiment name
    # -------------------------------------------------------------------------
    if args.experiment_name is None:
        args.experiment_name = (
            f"{Path(args.model_config).stem}"
            f"_ddp_bs{args.batch_size}_ds{args.max_datasets}"
            f"_lr{args.learning_rate}_wd{args.weight_decay}"
        )

    # -------------------------------------------------------------------------
    # Create DataModule and Model
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
    
    model = MultiDatasetModel(
        model_cfg=args.model_config,
        cfg_dir=args.dataset_configs_path,
        lr=args.learning_rate,
        wd=args.weight_decay,
        max_ds=args.max_datasets,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_vision=args.freeze_vision,
        compile_model=args.compile
    )
    
    # Pass additional hyperparameters to model
    model.hparams.core_lr_scale = args.core_lr_scale
    model.hparams.lr_scheduler = args.lr_scheduler
    model.hparams.warmup_epochs = args.warmup_epochs

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
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
            save_last=True
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
        # Epoch heartbeat
        EpochHeartbeat(metric_key="train_loss"),
        # Early stopping
        EarlyStopping(
            monitor="val_bps_overall",
            mode="max",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            verbose=True,
            check_on_train_epoch_end=False
        )
    ]
    
    # Add curriculum callback if enabled
    if args.enable_curriculum:
        callbacks.append(CurriculumCallback())

    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------
    # Only create logger on rank 0
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
        # Training duration
        max_epochs=args.max_epochs,
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=0.05,  # Use 5% of validation data
        
        # Hardware
        accelerator="gpu",
        devices=args.num_gpus,
        precision=args.precision,
        
        # Distributed strategy
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        ),
        
        # Optimization
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        
        # Logging and callbacks
        log_every_n_steps=50,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=logger,
        
        # UI
        enable_progress_bar=False
    )

    # -------------------------------------------------------------------------
    # Print training info
    # -------------------------------------------------------------------------
    if trainer.global_rank == 0:
        print("=" * 60)
        print(f"Starting: {args.experiment_name}")
        print("=" * 60)
        print(f"Accumulate grad batches: {trainer.accumulate_grad_batches}")
        print(f"Num training batches: {trainer.num_training_batches}")
        print(f"Curriculum learning: {args.enable_curriculum}")
        print(f"Pretrained checkpoint: {args.pretrained_checkpoint or 'None'}")
        print(f"Freeze vision: {args.freeze_vision}")
        print("=" * 60, flush=True)

    # -------------------------------------------------------------------------
    # Train!
    # -------------------------------------------------------------------------
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

