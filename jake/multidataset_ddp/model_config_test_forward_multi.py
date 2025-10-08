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
import torch
from torch.utils.data import DataLoader

import lightning as pl

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from DataYatesV1.models import build_model, initialize_model_components
from DataYatesV1.models.config_loader import load_config
from DataYatesV1.models.lightning import PLCoreVisionModel
from DataYatesV1.models.checkpoint import load_model_from_checkpoint, test_model_consistency, find_best_checkpoint
from DataYatesV1.models.model_manager import ModelRegistry
from DataYatesV1.utils.data import prepare_data
from DataYatesV1.utils.general import ensure_tensor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

from DataYatesV1.utils.torch import get_free_device
# Check if CUDA is available
device = get_free_device()
# device = torch.device('cpu')
print(f"Using device: {device}")

from DataYatesV1.utils.ipython import enable_autoreload
enable_autoreload()

from contextlib import nullcontext
AMP_BF16 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)


#%%
import yaml
import os
config_path = Path("/home/jake/repos/DataYatesV1/jake/multidataset_ddp/configs_multi/learned_res2d_small.yaml")
# config_path = Path("/home/tejas/Documents/fixational-transients/DataYatesV1/jake/multidataset_ddp/configs_multi/learned_res2d_small.yaml")
# Alternative config: Path("/home/tejas/Documents/fixational-transients/DataYatesV1/jake/multidataset_ddp/configs_multi/learned_dense_small_gru.yaml")
# config_path = Path("/home/jake/repos/DataYatesV1/jake/multidataset_ddp/configs_multi/core_res_modulator.yaml")
dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_basic_multi_120"
# dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_cones_multi"
# List full paths to *.yaml files that do not contain "base" in the name
yaml_files = [
    f for f in os.listdir(dataset_configs_path)
    if f.endswith(".yaml") and "base" not in f
]

from DataYatesV1.models.config_loader import load_dataset_configs
dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)

#%% Initialize model
config = load_config(config_path)
model = build_model(config, dataset_configs).to(device)

# run model readout forward with dummy input
# with torch.no_grad():
#     output = model.readouts[0](torch.randn(1, 256, 1, 9, 9).to(device))

# model.readouts[0].plot_weights()



#%%
# for n, p in model.named_parameters():
#     print(n, p.shape)

#%% Load Data
from DataYatesV1.utils.data.loading import remove_pixel_norm
import contextlib

train_datasets = {}
val_datasets = {}
updated_configs = []

for i, dataset_config in enumerate(dataset_configs):
    if i > 1: break
    dataset_config, pixel_norm_removed = remove_pixel_norm(dataset_config)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):          # ← optional
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)

    
    # Ensure the model sees fp32
    class Float32View(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            item["stim"] = item["stim"].float()  # cast back to fp32
            if pixel_norm_removed:
                item["stim"] = (item["stim"] - 127) / 255
            item["robs"] = item["robs"].float()
            if "behavior" in item:
                item["behavior"] = item["behavior"].float()
            return item

    train_dset = Float32View(train_dset)
    val_dset   = Float32View(val_dset)

    dataset_name = f"dataset_{i}"
    train_datasets[dataset_name] = train_dset
    val_datasets[dataset_name] = val_dset
    dataset_config["pixel_norm_removed"] = pixel_norm_removed
    updated_configs.append(dataset_config)

    print(f"Dataset {i}: {len(train_dset)} train, {len(val_dset)} val samples")

#%%

# train_dset, val_dset, dataset_config = prepare_data(dataset_config)

plt.plot(train_datasets['dataset_0'].base.dsets[0]['stim'][:1000,0,25,25])

#%%
# #%% test cones
# from DataYatesV1.models.modules import DAModel

# cfg = {'alpha': 0.01,
#       'beta': 0.00008,
#       'gamma': 0.5,
#       'tau_y_ms': 5.0,
#       'tau_z_ms': 60.0,
#       'n_y': 5.0,
#       'n_z': 2.0,
#       'filter_length': 32,
#       'learnable_params': False}

# cones = DAModel(**cfg)

# x = train_datasets['dataset_0'].base.dsets[0]['stim'].clone()

#         # permute (B,H,W) to (1, B, H, W) and squeeze back to (B,H,W)
# dtype = x.dtype
# print(dtype)

# y = cones(x.unsqueeze(0).float()).squeeze(0).permute(1,0,2,3)
# # rerun after pixelnorm on floats

# def pixelnorm(x):
#     return (x - 127) / 255.0

# y1 = cones(pixelnorm(x.float()).unsqueeze(0).float()).squeeze(0).permute(1,0,2,3)
# if dtype == torch.uint8:
#     y /= 2.5 
#     y *= 255
#     y = y.clamp(0, 255).to(torch.uint8)
#     # y = (y - y.mean())*
#     # y = (y * 127).to(torch.uint8)


# #%%
# inds = np.arange(10,1000) #+ np.random.randint(0, len(x)-1000)

# plt.plot(pixelnorm(y[inds,0,25,25].float()))
# # plt.gca().twinx()
# # plt.plot(y1[inds,0,25,25], 'r')

#         # return cones(x.unsqueeze(0).float()).squeeze(0).permute(1,0,2,3).to(dtype)

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
batch["stim"] = batch["stim"].to(torch.bfloat16)          # ! reduce mem

#%%
                        # convert to rates


# dtype = torch.float32
# batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# # Test forward pass
# model.eval()
# with torch.no_grad():
#     output = model(batch['stim'], dataset_id)
#     print(f"Output shape: {output.shape}")

# import torch.nn as nn
# loss = nn.functional.poisson_nll_loss(output, batch['robs'], log_input=False, reduction='mean')



#%%
with AMP_BF16():    
    x = model.adapters[dataset_id](batch['stim'])
    y = model.frontend(x)
    z = model.convnet(y)
    w = model.recurrent(z)
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
    