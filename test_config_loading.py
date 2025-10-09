#!/usr/bin/env python3
"""
Test script to verify the new config loading system works correctly.
"""

from pathlib import Path
from models.config_loader import load_dataset_configs

# Test loading the parent config
parent_config_path = Path("experiments/dataset_configs/multi_basic_120_backimage_all.yaml")

print(f"Loading parent config: {parent_config_path}")
print("=" * 80)

try:
    dataset_configs = load_dataset_configs(parent_config_path)
    
    print(f"\nSuccessfully loaded {len(dataset_configs)} dataset configs!")
    print("\nDataset names:")
    for i, cfg in enumerate(dataset_configs):
        print(f"  {i+1}. {cfg['session']} (lab: {cfg.get('lab', 'N/A')}, {len(cfg['cids'])} cells)")
    
    # Show details of first config
    print("\n" + "=" * 80)
    print("First dataset config details:")
    print("=" * 80)
    first_cfg = dataset_configs[0]
    
    print(f"\nSession: {first_cfg['session']}")
    print(f"Lab: {first_cfg.get('lab', 'N/A')}")
    print(f"Number of cells (cids): {len(first_cfg['cids'])}")
    print(f"Types: {first_cfg.get('types', 'N/A')}")
    print(f"Sampling: {first_cfg.get('sampling', 'N/A')}")
    print(f"\nKeys/lags:")
    for key, val in first_cfg.get('keys_lags', {}).items():
        print(f"  {key}: {val}")
    
    print(f"\nTransforms:")
    for key, val in first_cfg.get('transforms', {}).items():
        print(f"  {key}:")
        print(f"    source: {val.get('source', 'N/A')}")
        print(f"    expose_as: {val.get('expose_as', 'N/A')}")
        if 'ops' in val:
            print(f"    ops: {len(val['ops'])} operations")
    
    print(f"\nDatafilters:")
    for key, val in first_cfg.get('datafilters', {}).items():
        print(f"  {key}: {val}")
    
    print(f"\nTrain/val split: {first_cfg.get('train_val_split', 'N/A')}")
    print(f"Seed: {first_cfg.get('seed', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✓ Config loading test PASSED!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ Config loading test FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

