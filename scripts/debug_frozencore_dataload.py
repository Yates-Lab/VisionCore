#%%
"""
Debug script to reproduce the torch.cat empty list error in frozencore training.

The error occurs in CombinedEmbeddedDataset.__init__ at models/data/datasets.py:250
when torch.cat() receives an empty list of tensors.

This script checks:
1. Whether all dataset configs load successfully
2. Whether the dfs (data filters) produce any valid samples
3. Whether the train/val split produces empty indices
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from models.config_loader import load_dataset_configs
from models.data import prepare_data
from models.data.loading import get_embedded_datasets
from models.data.splitting import split_inds_by_trial

#%% Load dataset configs
dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_long_rowley.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

print(f"Loaded {len(dataset_configs)} dataset configs:")
for i, cfg in enumerate(dataset_configs):
    print(f"  [{i}] {cfg['session']} (lab: {cfg.get('lab', 'yates')})")

#%% Test each dataset config individually with strict=True to see actual errors
print("\n" + "="*60)
print("Testing each dataset config with strict=True")
print("="*60)

for i, cfg in enumerate(dataset_configs):
    print(f"\n--- Dataset {i}: {cfg['session']} ---")
    try:
        train_data, val_data, dset_cfg = prepare_data(cfg, strict=True)
        print(f"  SUCCESS: train={len(train_data)}, val={len(val_data)}")

        # Check dfs values
        for j, dset in enumerate(train_data.dsets):
            dfs = dset['dfs']
            n_valid = dfs.any(dim=1).sum().item() if dfs.ndim > 1 else dfs.sum().item()
            print(f"    dset[{j}] ({dset.metadata.get('name', 'unknown')}): "
                  f"{n_valid}/{len(dfs)} valid samples ({100*n_valid/len(dfs):.1f}%)")

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

#%% Test with strict=False (mimics training behavior)
print("\n" + "="*60)
print("Testing each dataset config with strict=False")
print("="*60)

for i, cfg_orig in enumerate(dataset_configs):
    # Deep copy to avoid mutating original config
    import copy
    cfg = copy.deepcopy(cfg_orig)

    print(f"\n--- Dataset {i}: {cfg['session']} ---")
    try:
        train_data, val_data, dset_cfg = prepare_data(cfg, strict=False)

        # Check if we got any data
        if len(train_data) == 0:
            print(f"  WARNING: No training samples!")
        else:
            print(f"  train={len(train_data)}, val={len(val_data)}")

        # Check each underlying dataset
        for j, dset in enumerate(train_data.dsets):
            dfs = dset['dfs']
            n_valid = dfs.any(dim=1).sum().item() if dfs.ndim > 1 else dfs.sum().item()
            print(f"    dset[{j}]: {n_valid}/{len(dfs)} valid samples")

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

#%% Manually trace through data loading for the failing session
print("\n" + "="*60)
print("Detailed trace for each session")
print("="*60)

for i, cfg in enumerate(dataset_configs):
    print(f"\n{'='*60}")
    print(f"Session: {cfg['session']}")
    print(f"{'='*60}")

    lab = cfg.get("lab", "yates")
    sess_name = cfg["session"]
    dset_types = cfg["types"]

    # Get the appropriate session loader
    if lab.lower() == "yates":
        from DataYatesV1.utils.io import get_session
    elif lab.lower() == "rowley":
        from DataRowleyV1V2.data.registry import get_session
    else:
        print(f"  Unknown lab: {lab}")
        continue

    try:
        sess = get_session(*sess_name.split("_"))
        print(f"  Session loaded: {sess}")
    except Exception as e:
        print(f"  Failed to load session: {e}")
        continue

    # Try loading each dataset type
    for dt in dset_types:
        print(f"\n  --- Dataset type: {dt} ---")
        try:
            dset = sess.get_dataset(dt, config=cfg)
            print(f"    Loaded: {len(dset)} samples")

            # Check what keys are available
            print(f"    Keys: {list(dset.covariates.keys())[:10]}...")

            # Check dpi_valid
            if 'dpi_valid' in dset:
                dpi_valid = dset['dpi_valid']
                n_dpi_valid = (dpi_valid > 0).sum().item()
                print(f"    dpi_valid: {n_dpi_valid}/{len(dpi_valid)} valid ({100*n_dpi_valid/len(dpi_valid):.1f}%)")
            else:
                print(f"    WARNING: 'dpi_valid' not in dataset!")

            # Check trial_inds
            if 'trial_inds' in dset:
                n_trials = dset['trial_inds'].unique().numel()
                print(f"    trial_inds: {n_trials} unique trials")
            else:
                print(f"    WARNING: 'trial_inds' not in dataset!")

        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Debug complete")
print("="*60)

# %%
