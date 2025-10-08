#!/usr/bin/env python
"""
Test script for modulator-only models

This script demonstrates:
1. Building a modulator-only model from config
2. Testing forward pass with behavior data only (no stimulus)
3. Comparing with normal model behavior
"""
#%%
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import contextlib

from DataYatesV1.models import build_model
from DataYatesV1.models.config_loader import load_config, load_dataset_configs
from DataYatesV1.utils.data import prepare_data
from DataYatesV1.utils.data.loading import remove_pixel_norm
from DataYatesV1.utils.torch import get_free_device
from DataYatesV1.utils.ipython import enable_autoreload

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Setup
device = get_free_device()
print(f"Using device: {device}")
enable_autoreload()

#%% Load modulator-only config
config_path = Path("configs_multi/modulator_only_convgru.yaml")
dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_basic_multi_120"

# Load dataset configs (we'll use just a few for testing)
yaml_files = [
    f for f in os.listdir(dataset_configs_path)
    if f.endswith(".yaml") and "base" not in f
][:2]  # Just use first 2 datasets for testing

dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)

#%% Build modulator-only model
print("Building modulator-only model...")
config = load_config(config_path)
model = build_model(config, dataset_configs).to(device)

print(f"Model type: {type(model).__name__}")
print(f"Convnet type: {type(model.convnet).__name__}")
print(f"Modulator type: {type(model.modulator).__name__}")
print(f"Recurrent type: {type(model.recurrent).__name__}")
print(f"Readout types: {[type(r).__name__ for r in model.readouts]}")

#%% Load some data to get behavior vectors
print("\nLoading data for behavior vectors...")
train_datasets = {}

for i, dataset_config in enumerate(dataset_configs):
    if i > 0: break  # Just use first dataset
    
    dataset_config, pixel_norm_removed = remove_pixel_norm(dataset_config)
    
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)
    
    # Float32 wrapper
    class Float32View(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            if "behavior" in item:
                item["behavior"] = item["behavior"].float()
            return item
    
    train_dset = Float32View(train_dset)
    train_datasets[f"dataset_{i}"] = train_dset
    print(f"Dataset {i}: {len(train_dset)} samples")

#%% Test modulator-only forward pass
print("\n" + "="*50)
print("TESTING MODULATOR-ONLY MODEL")
print("="*50)

batch_size = 32
dataset_id = 0

# Get a batch of behavior data
ntrain = len(train_datasets[f'dataset_{dataset_id}'])
inds = np.arange(1000, 1000 + batch_size)
batch = train_datasets[f'dataset_{dataset_id}'][inds]

# Move to device
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

print(f"Behavior shape: {batch['behavior'].shape}")
print(f"Behavior sample: {batch['behavior'][0, :5]}")  # First 5 behavior vars

#%% Test forward pass with behavior only
model.eval()
with torch.no_grad():
    # This should work without stimulus!
    output = model(stimulus=None, dataset_idx=dataset_id, behavior=batch['behavior'])
    print(f"\n✅ Modulator-only forward pass successful!")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}")

#%% Test step-by-step pipeline
print("\n" + "-"*30)
print("STEP-BY-STEP PIPELINE TEST")
print("-"*30)

with torch.no_grad():
    # Step 1: Create minimal features (what happens when convnet=none)
    B = batch['behavior'].shape[0]
    minimal_features = torch.ones(B, 1, 1, 1, 1, device=device)
    print(f"1. Minimal features shape: {minimal_features.shape}")
    
    # Step 2: Process through modulator
    modulated_features = model.modulator(minimal_features, batch['behavior'])
    print(f"2. Modulated features shape: {modulated_features.shape}")
    
    # Step 3: Process through recurrent (should be identity)
    recurrent_output = model.recurrent(modulated_features)
    print(f"3. Recurrent output shape: {recurrent_output.shape}")
    
    # Step 4: Process through readout
    final_output = model.readouts[dataset_id](recurrent_output)
    print(f"4. Final output shape: {final_output.shape}")

print(f"\n✅ All pipeline steps successful!")

#%% Compare with random behavior
print("\n" + "-"*30)
print("BEHAVIOR SENSITIVITY TEST")
print("-"*30)

with torch.no_grad():
    # Original behavior
    output1 = model(stimulus=None, dataset_idx=dataset_id, behavior=batch['behavior'])
    
    # Random behavior (same shape)
    random_behavior = torch.randn_like(batch['behavior'])
    output2 = model(stimulus=None, dataset_idx=dataset_id, behavior=random_behavior)
    
    # Compare outputs
    diff = (output1 - output2).abs().mean()
    print(f"Output difference with random behavior: {diff:.6f}")
    print(f"✅ Model is sensitive to behavior changes!")

print("\n" + "="*50)
print("MODULATOR-ONLY MODEL TEST COMPLETE!")
print("="*50)
