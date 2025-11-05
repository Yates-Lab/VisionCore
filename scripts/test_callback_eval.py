#!/usr/bin/env python3
"""
Test script to debug the callback logging issue.

This mimics eval_stack_devel.py but uses a model from training instead of a checkpoint.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np
import matplotlib.pyplot as plt

# %%
print("=" * 80)
print("Step 1: Create a model from training (like the callback receives)")
print("=" * 80)

from training.pl_modules import MultiDatasetModel

model_config = "experiments/model_configs/learned_dense_film_none_gaussian.yaml"
dataset_configs_path = "experiments/dataset_configs/multi_basic_120_backimage_history.yaml"

# Create model exactly as training does
model = MultiDatasetModel(
    model_cfg=model_config,
    cfg_dir=dataset_configs_path,
    lr=1e-3,
    wd=1e-4,
    max_ds=1,  # Just load 1 dataset for testing
    pretrained_checkpoint=None,
    freeze_vision=False,
    compile_model=False
)

print(f"✓ Model created")
print(f"  Model type: {type(model)}")
print(f"  Has 'names': {hasattr(model, 'names')}")
print(f"  Has 'dataset_configs': {hasattr(model, 'dataset_configs')}")
print(f"  Has 'model': {hasattr(model, 'model')}")
if hasattr(model, 'names'):
    print(f"  Dataset names: {model.names}")
if hasattr(model, 'dataset_configs'):
    print(f"  Number of dataset configs: {len(model.dataset_configs)}")

# Move to GPU
device = torch.device('cuda:0')
model = model.to(device)
model.eval()

# %%
print("\n" + "=" * 80)
print("Step 2: Test load_single_dataset (what eval stack does first)")
print("=" * 80)

from eval.eval_stack_utils import load_single_dataset

dataset_idx = 0

try:
    train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
    print(f"✓ Dataset loaded successfully")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Number of sub-datasets: {len(train_data.dsets)}")
    print(f"  Sub-dataset types: {[d.metadata['name'] for d in train_data.dsets]}")
    
    # Check stimulus shapes
    print("\n  Checking stimulus shapes:")
    for i, dset in enumerate(train_data.dsets):
        stim_type = dset.metadata['name']
        sample = dset[0]
        stim_shape = sample['stim'].shape
        print(f"    {stim_type}: {stim_shape}")

    # Also check the dataset config
    print(f"\n  Dataset config keys_lags:")
    print(f"    stim: {dataset_config['keys_lags']['stim']}")
    print(f"  Dataset config types: {dataset_config['types']}")
        
except Exception as e:
    print(f"✗ Failed to load dataset:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# %%
print("\n" + "=" * 80)
print("Step 3: Test running model on a batch from CombinedEmbeddedDataset")
print("=" * 80)

from eval.eval_stack_utils import run_model, get_stim_inds

# Test with each stimulus type using get_stim_inds
stim_types = ['backimage', 'gaborium', 'gratings', 'fixrsvp']

for stim_type in stim_types:
    print(f"\nTesting {stim_type}...")

    try:
        # Get indices for this stimulus type
        stim_inds = get_stim_inds(stim_type, train_data, val_data)
        print(f"  Found {len(stim_inds)} {stim_type} samples")

        # Get a small batch using the CombinedEmbeddedDataset
        batch_size = 4
        batch_inds = stim_inds[:min(batch_size, len(stim_inds))]
        batch = val_data[batch_inds]

        print(f"  Batch stim shape: {batch['stim'].shape}")
        print(f"  Batch robs shape: {batch['robs'].shape}")

        # Run model
        batch_with_pred = run_model(model, batch, dataset_idx)
        print(f"  ✓ Model ran successfully")
        print(f"    Output shape: {batch_with_pred['rhat'].shape}")

    except Exception as e:
        print(f"  ✗ Failed:")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

# %%
print("\n" + "=" * 80)
print("Step 4: Test eval_stack_single_dataset (what the callback calls)")
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
    print(f"\n✓ eval_stack_single_dataset completed successfully!")
    print(f"  Result keys: {list(results.keys())}")
    
    if 'bps' in results:
        print(f"  BPS stimulus types: {list(results['bps'].keys())}")
    if 'ccnorm' in results:
        print(f"  CCNORM keys: {list(results['ccnorm'].keys())}")
    if 'saccade' in results:
        print(f"  Saccade stimulus types: {list(results['saccade'].keys())}")
    if 'sta' in results:
        print(f"  STA keys: {list(results['sta'].keys())}")
        
except Exception as e:
    print(f"\n✗ eval_stack_single_dataset failed:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# %%
print("\n" + "=" * 80)
print("Step 5: Test plotting (what the callback does after eval)")
print("=" * 80)

from torchvision.utils import make_grid

# Test STA plotting
if 'sta' in results:
    print("\nTesting STA plotting...")
    sta_dict = results['sta']
    N = len(sta_dict['peak_lag'])
    print(f"  Number of cells: {N}")
    
    try:
        num_lags = sta_dict['Z_STA_robs'].shape[0]
        rf_pairs = []
        
        for cc in range(min(N, 3)):  # Just test first 3 cells
            this_lag = sta_dict['peak_lag'][cc]
            sta_robs = sta_dict['Z_STA_robs'][this_lag,:,:,cc]
            sta_rhat = sta_dict['Z_STA_rhat'][this_lag,:,:,cc]
            
            # zscore each
            sta_robs = (sta_robs - sta_robs.mean()) / sta_robs.std()
            sta_rhat = (sta_rhat - sta_rhat.mean()) / sta_rhat.std()
            
            grid = torch.stack([sta_robs, sta_rhat], 0).unsqueeze(1)
            grid = make_grid(grid, nrow=2, normalize=True, scale_each=False, padding=2, pad_value=1)
            grid = 0.2989 * grid[0:1,:,:] + 0.5870 * grid[1:2,:,:] + 0.1140 * grid[2:3,:,:]
            rf_pairs.append(grid)
        
        log_grid = make_grid(torch.stack(rf_pairs), nrow=int(np.sqrt(3)), 
                            normalize=True, scale_each=True, padding=2, pad_value=1)
        
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(log_grid.detach().cpu().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title('STA Test (first 3 cells)')
        
        print(f"  ✓ STA plotting successful")
        plt.close(fig)
        
    except Exception as e:
        print(f"  ✗ STA plotting failed:")
        import traceback
        traceback.print_exc()

# Test BPS plotting
if 'bps' in results:
    print("\nTesting BPS plotting...")
    try:
        fig = plt.figure(figsize=(12, 6))
        for k in results['bps']:
            if k in ['val', 'cids']:
                continue
            plt.plot(np.arange(len(results['bps'][k]['bps'])), 
                    np.maximum(results['bps'][k]['bps'], -.1), 
                    label=k, alpha=0.5)
        plt.legend()
        plt.title('BPS Test')
        
        print(f"  ✓ BPS plotting successful")
        plt.close(fig)
        
    except Exception as e:
        print(f"  ✗ BPS plotting failed:")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)

