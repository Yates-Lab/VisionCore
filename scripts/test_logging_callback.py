#!/usr/bin/env python3
"""
Test script for ModelLoggingCallback.

This script:
1. Creates a minimal training setup with 1 dataset
2. Runs 1 epoch with only 10 steps
3. Triggers slow logging after epoch 1 to test the callback
4. Logs to a test WandB project

Usage:
    python scripts/test_logging_callback.py --gpu 0
    python scripts/test_logging_callback.py --gpu 0 --test_eval_only  # Test eval stack only
"""

import sys
import os
from pathlib import Path
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pl_modules import MultiDatasetModel, MultiDatasetDM
from training.callbacks import ModelLoggingCallback

# Suppress dynamo errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def test_eval_stack_only(model, dataset_idx=0):
    """Test just the eval stack without full training."""
    print("\n" + "=" * 80)
    print("Testing eval_stack_single_dataset directly...")
    print("=" * 80)

    from eval.eval_stack_multidataset import eval_stack_single_dataset

    try:
        results = eval_stack_single_dataset(
            model=model,
            dataset_idx=dataset_idx,
            analyses=['bps', 'ccnorm', 'saccade', 'sta', 'qc'],
            batch_size=64,
            rescale=True
        )
        print("\n✓ Eval stack completed successfully!")
        print(f"  Results keys: {list(results.keys())}")
        return True
    except Exception as e:
        print("\n✗ Eval stack failed:")
        import traceback
        traceback.print_exc()
        return False


def main():
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--test_eval_only", action="store_true",
                       help="Only test eval stack, don't run training")
    parser.add_argument("--dataset_idx", type=int, default=0,
                       help="Dataset index to evaluate (for test_eval_only)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    model_config = "experiments/model_configs/learned_dense_film_none_gaussian.yaml"
    dataset_configs_path = "experiments/dataset_configs/multi_basic_120_backimage_history.yaml"

    # Test parameters
    max_datasets = 1  # Only load 1 dataset for quick testing
    batch_size = 32
    steps_per_epoch = 10  # Only 10 steps per epoch
    max_epochs = 1  # Only 1 epoch
    num_workers = 4
    gpu = args.gpu

    # Logging parameters
    fast_log_interval = 1  # Log kernels after epoch 1
    slow_log_interval = 1  # Log eval stack after epoch 1
    log_dataset_idx = args.dataset_idx  # Use specified dataset index
    
    print("=" * 80)
    print("Testing ModelLoggingCallback")
    print("=" * 80)
    print(f"Model config: {model_config}")
    print(f"Dataset configs: {dataset_configs_path}")
    print(f"Max datasets: {max_datasets}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Max epochs: {max_epochs}")
    print(f"Log dataset idx: {log_dataset_idx}")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Create DataModule
    # -------------------------------------------------------------------------
    print("\nCreating DataModule...")
    dm = MultiDatasetDM(
        cfg_dir=dataset_configs_path,
        max_ds=max_datasets,
        batch=batch_size,
        workers=num_workers,
        steps_per_epoch=steps_per_epoch,
        enable_curriculum=False,
        dset_dtype='bfloat16',
        homogeneous_batches=True,
    )
    
    # Setup to load datasets
    dm.setup('fit')
    print(f"✓ Loaded {len(dm.names)} dataset(s):")
    for i, name in enumerate(dm.names):
        print(f"  [{i}] {name}")
    
    # -------------------------------------------------------------------------
    # Create Model
    # -------------------------------------------------------------------------
    print("\nCreating Model...")
    model = MultiDatasetModel(
        model_cfg=model_config,
        cfg_dir=dataset_configs_path,
        lr=1e-3,
        wd=1e-4,
        max_ds=max_datasets,
        pretrained_checkpoint=None,
        freeze_vision=False,
        compile_model=False
    )
    print(f"✓ Model created with {len(model.names)} readout(s)")

    # -------------------------------------------------------------------------
    # Test eval stack only if requested
    # -------------------------------------------------------------------------
    if args.test_eval_only:
        print("\n" + "=" * 80)
        print("Running eval stack test only (no training)")
        print("=" * 80)
        success = test_eval_stack_only(model, dataset_idx=log_dataset_idx)
        if success:
            print("\n✓ Eval stack test passed!")
            return
        else:
            print("\n✗ Eval stack test failed!")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Create Logging Callback
    # -------------------------------------------------------------------------
    print("\nCreating ModelLoggingCallback...")
    logging_callback = ModelLoggingCallback(
        fast_interval=fast_log_interval,
        slow_interval=slow_log_interval,
        eval_dataset_idx=log_dataset_idx,
        batch_size=64,
        rescale=True
    )
    print(f"✓ Callback created:")
    print(f"  - Fast logging interval: {fast_log_interval}")
    print(f"  - Slow logging interval: {slow_log_interval}")
    print(f"  - Eval dataset idx: {log_dataset_idx}")
    print(f"  - Eval analyses: {logging_callback.eval_analyses}")
    
    # -------------------------------------------------------------------------
    # Create WandB Logger
    # -------------------------------------------------------------------------
    print("\nCreating WandB logger...")
    logger = WandbLogger(
        project="visioncore-logging-test",
        name="test_logging_callback",
        save_dir="./wandb_test",
        log_model=False,
    )
    print("✓ WandB logger created")
    
    # -------------------------------------------------------------------------
    # Create Trainer
    # -------------------------------------------------------------------------
    print("\nCreating Trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[gpu],
        logger=logger,
        callbacks=[logging_callback],
        log_every_n_steps=5,
        enable_checkpointing=False,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )
    print("✓ Trainer created")
    
    # -------------------------------------------------------------------------
    # Run Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Starting training (this will trigger logging after epoch 1)...")
    print("=" * 80)
    
    try:
        trainer.fit(model, datamodule=dm)
        print("\n" + "=" * 80)
        print("✓ Training completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ Training failed with error:")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if logger.experiment is not None:
            logger.experiment.finish()
        print("\n✓ WandB run finished")

if __name__ == "__main__":
    main()

