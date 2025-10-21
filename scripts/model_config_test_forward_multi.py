#!/usr/bin/env python
"""
Example script for checking whether a model config builds

This script demonstrates:
1. Builds a model from a config
2. Prepares data
3. Tests the forward and backward on one batch
4. Tests the jacobian
"""
#%%
import sys
import torch
from torch.utils.data import DataLoader

import lightning as pl

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from models import build_model, initialize_model_components
from models.config_loader import load_config
from models.data import prepare_data
from models.utils.general import ensure_tensor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda:1')
# device = torch.device('cpu')
print(f"Using device: {device}")

from DataYatesV1.utils.ipython import enable_autoreload
enable_autoreload()

from contextlib import nullcontext
AMP_BF16 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)


#%%
from models.config_loader import load_dataset_configs
import os
# 
config_path = Path("/home/jake/repos/VisionCore/experiments/model_configs/learned_res_split_gru.yaml")

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_backimage_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

#%% Initialize model
config = load_config(config_path)
model = build_model(config, dataset_configs).to(device)

# # run model readout forward with dummy input
# with torch.no_grad():
#     output = model.readouts[0](torch.randn(1, 256, 1, 9, 9).to(device))

# model.readouts[0].plot_weights()


#%% Load Data
import contextlib

train_datasets = {}
val_datasets = {}
updated_configs = []

for i, dataset_config in enumerate(dataset_configs):
    if i > 1: break

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):          # ← optional
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)

    # cast to bfloat16
    train_dset.cast(torch.bfloat16, target_keys=['stim', 'robs', 'dfs'])
    val_dset.cast(torch.bfloat16, target_keys=['stim', 'robs', 'dfs'])
    
    dataset_name = f"dataset_{i}"
    train_datasets[dataset_name] = train_dset
    val_datasets[dataset_name] = val_dset
    updated_configs.append(dataset_config)

    print(f"Dataset {i}: {len(train_dset)} train, {len(val_dset)} val samples")

#%%
plt.plot(train_datasets['dataset_0'].dsets[0]['stim'][:1000,0,25,25].float().cpu().numpy())

#%% prepare dataloaders
# train_loader, val_loader = create_multidataset_loaders(train_datasets, val_datasets, batch_size=2, num_workers=os.cpu_count()//2)

#%% test one dataset
batch_size = 256
dataset_id = 0

ntrain = len(train_datasets[f'dataset_{dataset_id}'])
# inds = np.random.randint(0, ntrain - batch_size, batch_size)
inds = np.arange(1000, 1000+batch_size)
batch = train_datasets[f'dataset_{dataset_id}'][inds]

batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
# batch["stim"] = batch["stim"].to(torch.bfloat16)          # ! reduce mem

#%%
                        # convert to rates


dtype = torch.bfloat16
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# Test forward pass
model.eval()
with AMP_BF16():
    output = model(batch['stim'], dataset_id, batch['behavior'])
    print(f"Output shape: {output.shape}")

# import torch.nn as nn
# loss = nn.functional.poisson_nll_loss(output, batch['robs'], log_input=False, reduction='mean')



#%%
with AMP_BF16():    
    x = model.adapters[dataset_id](batch['stim'])
    y = model.frontend(x)
    z = model.convnet(y)
    # w = model.recurrent(z)
    # print(f"Convnet output shape: {z.shape}")
    # w = model.modulator(z, batch['behavior'])
print(z.shape)


#%% Run full model
with AMP_BF16():                                          # <-- new
    output = model(batch["stim"], dataset_id, batch["behavior"])         # predict log-rates

output = torch.exp(output)    

#%%
_ = plt.plot(output.detach().cpu().numpy())
# %% test all datasets
import torch.nn as nn
batch_size = 32

for dataset_id in range(len(train_datasets)):
    ntrain = len(train_datasets[f'dataset_{dataset_id}'])
    inds = np.random.randint(0, ntrain - batch_size, batch_size)

    batch = train_datasets[f'dataset_{dataset_id}'][inds]

    # dtype = torch.float32
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch['stim'], dataset_id)
        print(f"Output shape: {output.shape}")
        loss = nn.functional.poisson_nll_loss(output, batch['robs'], log_input=False, reduction='mean')
        print(f"Loss: {loss.item()}")
    
# %%
