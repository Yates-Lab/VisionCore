#%%
"""
Simple script to load a trained FrozenCoreModel from checkpoint.
"""

import sys
sys.path.insert(0, '..')

import torch
from pathlib import Path

from training.pl_modules import FrozenCoreModel

#%% Configuration
# Path to the trained checkpoint
checkpoint_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/frozencore_readouts_120/checkpoints/frozencore_resnet_none_convgru_bs256_ds30_lr1e-3_wd1.0e-5_warmup5/last.ckpt"

# Device to load on
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% Load the model
print(f"Loading FrozenCoreModel from: {checkpoint_path}")

model = FrozenCoreModel.load_from_checkpoint(
    checkpoint_path,
    map_location='cpu',
    strict=False
)

model.to(device)
model.eval()

print("✓ Model loaded successfully!")

#%% Print model info
print(f"\nModel Information:")
print(f"  Datasets: {len(model.names)}")
print(f"  Dataset names: {model.names}")
print(f"  Activation: {type(model.model.activation).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.model.parameters())
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {frozen_params:,}")

#%% Example: Run inference on a dataset
import matplotlib.pyplot as plt
plt.imshow(model.model.readouts[1].features.weight.detach().cpu().squeeze(), interpolation='none')
# Uncomment and modify to run inference

# from eval.eval_stack_utils import load_single_dataset
# 
# dataset_idx = 0
# train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
# 
# print(f"\nDataset: {model.names[dataset_idx]}")
# print(f"  Train samples: {len(train_data)}")
# print(f"  Val samples: {len(val_data)}")
# print(f"  Units: {len(dataset_config['cids'])}")

# %%

