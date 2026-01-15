#!/usr/bin/env python3
"""
Test script for the frozen core readout training pipeline.

This script tests each component independently to identify issues:
1. Loading pretrained model from checkpoint
2. Loading new dataset configs
3. Building new model with frozen core
4. Data loading and batching
5. Forward pass through model
6. Training step
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Configuration - modify these paths as needed
CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
MODEL_TYPE = "resnet_none_convgru"
NEW_DATASET_CONFIG = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_long_rowley.yaml"
MAX_DATASETS = 2


def test_load_pretrained_model():
    """Test 1: Load pretrained model from checkpoint."""
    print("\n" + "="*60)
    print("TEST 1: Loading pretrained model")
    print("="*60)
    
    from eval.eval_stack_multidataset import load_model
    
    try:
        pl_model, model_info = load_model(
            model_type=MODEL_TYPE,
            model_index=None,
            checkpoint_dir=CHECKPOINT_DIR,
            device='cpu',
            verbose=True
        )
        print(f"✓ Loaded model successfully")
        print(f"  Model type: {pl_model.model_config.get('model_type', 'unknown')}")
        print(f"  Datasets: {len(pl_model.names)}")
        print(f"  Total params: {sum(p.numel() for p in pl_model.model.parameters()):,}")
        return pl_model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_new_dataset_configs():
    """Test 2: Load new dataset configurations."""
    print("\n" + "="*60)
    print("TEST 2: Loading new dataset configs")
    print("="*60)
    
    from models.config_loader import load_dataset_configs
    
    try:
        cfgs = load_dataset_configs(NEW_DATASET_CONFIG)[:MAX_DATASETS]
        names = [cfg['session'] for cfg in cfgs]
        print(f"✓ Loaded {len(cfgs)} dataset configs")
        print(f"  Sessions: {names}")
        return cfgs, names
    except Exception as e:
        print(f"✗ Failed to load dataset configs: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_build_new_model(pretrained_model, model_config, cfgs):
    """Test 3: Build new model with frozen core."""
    print("\n" + "="*60)
    print("TEST 3: Building new model with frozen core")
    print("="*60)
    
    from models.modules.models import MultiDatasetV1Model
    
    try:
        # Add dataset names
        for c in cfgs:
            c["_dataset_name"] = c['session']
        
        # Create new model
        new_model = MultiDatasetV1Model(model_config, cfgs)
        print(f"✓ Created new model")
        print(f"  Adapters: {len(new_model.adapters)}")
        print(f"  Readouts: {len(new_model.readouts)}")
        
        # Copy shared components
        shared_components = ['frontend', 'convnet', 'modulator', 'recurrent']
        for name in shared_components:
            pretrained_comp = getattr(pretrained_model, name, None)
            new_comp = getattr(new_model, name, None)
            
            if pretrained_comp is None or new_comp is None:
                print(f"  - {name}: None (skipping)")
                continue
            if isinstance(pretrained_comp, nn.Identity):
                print(f"  - {name}: Identity (no params)")
                continue
                
            try:
                new_comp.load_state_dict(pretrained_comp.state_dict())
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
        
        # Freeze shared components
        frozen = 0
        for name in shared_components:
            comp = getattr(new_model, name, None)
            if comp is None or isinstance(comp, nn.Identity):
                continue
            for p in comp.parameters():
                p.requires_grad = False
                frozen += p.numel()
        
        trainable = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
        print(f"✓ Frozen: {frozen:,}, Trainable: {trainable:,}")
        
        return new_model
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading():
    """Test 4: Load data and create batches."""
    print("\n" + "="*60)
    print("TEST 4: Data loading")
    print("="*60)
    
    from training.pl_modules import MultiDatasetDM
    
    try:
        dm = MultiDatasetDM(
            cfg_dir=NEW_DATASET_CONFIG,
            max_ds=MAX_DATASETS,
            batch=4,  # Small batch for testing
            workers=0,  # No multiprocessing for debugging
            steps_per_epoch=10,
            enable_curriculum=False,
            dset_dtype='bfloat16'
        )
        dm.setup('fit')
        print(f"✓ Created DataModule")
        print(f"  Train datasets: {list(dm.train_dsets.keys())}")
        
        # Get a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Handle both list and dict batch formats
        if isinstance(batch, list):
            print(f"✓ Got batch (list of {len(batch)} items)")
            batch = batch[0]  # Take first item for testing

        print(f"✓ Got batch")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

        return dm, batch
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, batch):
    """Test 5: Forward pass through model (using autocast like PyTorch Lightning)."""
    print("\n" + "="*60)
    print("TEST 5: Forward pass")
    print("="*60)

    try:
        model.eval()

        with torch.no_grad():
            stim = batch['stim']
            dataset_idx = batch['dataset_idx'][0].item() if isinstance(batch['dataset_idx'], torch.Tensor) else batch['dataset_idx']
            behavior = batch.get('behavior', None)

            print(f"  Input stim: shape={stim.shape}, dtype={stim.dtype}")
            if behavior is not None:
                print(f"  Input behavior: shape={behavior.shape}, dtype={behavior.dtype}")
            print(f"  Dataset idx: {dataset_idx}")

            # Use autocast like PyTorch Lightning does with precision="bf16-mixed"
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                output = model(stim, dataset_idx=dataset_idx, behavior=behavior)

            print(f"✓ Forward pass successful")
            print(f"  Output: shape={output.shape}, dtype={output.dtype}")
            return model, output
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_training_step(model, batch):
    """Test 6: Training step with loss computation."""
    print("\n" + "="*60)
    print("TEST 6: Training step")
    print("="*60)

    from models.losses import MaskedLoss

    try:
        model.train()

        stim = batch['stim']
        robs = batch['robs']
        dfs = batch['dfs']
        dataset_idx = batch['dataset_idx'][0].item() if isinstance(batch['dataset_idx'], torch.Tensor) else batch['dataset_idx']
        behavior = batch.get('behavior', None)

        # Forward pass with autocast (like PyTorch Lightning)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = model(stim, dataset_idx=dataset_idx, behavior=behavior)

        # Compute loss (convert to float like MultiDatasetModel does)
        log_input = isinstance(model.activation, nn.Identity)
        loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=log_input, reduction="none"))

        batch_loss = {
            'rhat': output.float(),
            'robs': robs.float(),
            'dfs': dfs.float()
        }
        loss = loss_fn(batch_loss)
        print(f"✓ Loss computed: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        print(f"✓ Backward pass successful")

        # Check gradients
        trainable_with_grad = 0
        trainable_no_grad = 0
        frozen_with_grad = 0

        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    trainable_with_grad += 1
                else:
                    trainable_no_grad += 1
            else:
                if p.grad is not None:
                    frozen_with_grad += 1

        print(f"  Trainable params with gradients: {trainable_with_grad}")
        print(f"  Trainable params without gradients: {trainable_no_grad}")
        print(f"  Frozen params with gradients (should be 0): {frozen_with_grad}")

        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FROZEN CORE PIPELINE TESTS")
    print("="*60)
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"New dataset config: {NEW_DATASET_CONFIG}")
    print(f"Max datasets: {MAX_DATASETS}")

    # Test 1: Load pretrained model
    pl_model = test_load_pretrained_model()
    if pl_model is None:
        print("\n❌ STOPPING: Cannot proceed without pretrained model")
        return 1

    # Test 2: Load new dataset configs
    cfgs, names = test_load_new_dataset_configs()
    if cfgs is None:
        print("\n❌ STOPPING: Cannot proceed without dataset configs")
        return 1

    # Test 3: Build new model
    new_model = test_build_new_model(
        pl_model.model,
        pl_model.model_config,
        cfgs
    )
    if new_model is None:
        print("\n❌ STOPPING: Cannot proceed without new model")
        return 1

    # Test 4: Data loading
    dm, batch = test_data_loading()
    if batch is None:
        print("\n❌ STOPPING: Cannot proceed without data")
        return 1

    # Test 5: Forward pass
    new_model, output = test_forward_pass(new_model, batch)
    if output is None:
        print("\n❌ STOPPING: Cannot proceed without forward pass")
        return 1

    # Test 6: Training step (use the bfloat16 model from test 5)
    success = test_training_step(new_model, batch)
    if not success:
        print("\n❌ STOPPING: Training step failed")
        return 1

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

