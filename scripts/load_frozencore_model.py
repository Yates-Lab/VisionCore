#%%
"""
Simple script to load a trained FrozenCoreModel from checkpoint.
"""

import sys
#sys.path.insert(0, '..')

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

from eval.eval_stack_utils import load_single_dataset, evaluate_dataset

datasets = model.names
eval_results = {}
for iD in range(len(datasets)):
    # Load dataset
    train_data, val_data, dataset_config = load_single_dataset(model, iD)
    eval_results[model.names[iD]] = {}
    dset_types = dataset_config['types']
    for t in dset_types:
        print(f"  Evaluating on {t} set")
        eval_results[model.names[iD]][t] = {}
        trian_res = evaluate_dataset(model, train_data, indices=train_data.get_dataset_inds(t), dataset_idx=iD, batch_size=256, desc=f"Eval (train) {t}")
        eval_results[model.names[iD]][t]['train'] = trian_res
        val_res = evaluate_dataset(model, val_data, indices=val_data.get_dataset_inds(t), dataset_idx=iD, batch_size=256, desc=f"Eval (val) {t}")
        eval_results[model.names[iD]][t]['val'] = val_res

#%%
n_dsets = len(datasets)
fig, axs = plt.subplots(1, n_dsets, figsize=(6*n_dsets, 5))
for iD in range(n_dsets):
    dset_name = datasets[iD]
    axs[iD].set_title(f"Dataset: {dset_name}")
    axs[iD].set_xlabel("Unit")
    axs[iD].set_ylabel("Bits per spike")
    for t, res in eval_results[dset_name].items():
        axs[iD].scatter(range(len(res['val']['bps'])), res['val']['bps'], label=t, alpha=0.7)
    axs[iD].legend()
    axs[iD].set_ylim(-.5, 5)
plt.tight_layout()
plt.show()
#%%
# Plot train vs test bps for each dataset

for iD in range(n_dsets):
    dset_name = datasets[iD]
    n_types = len(eval_results[dset_name])
    fig, axs = plt.subplots(1, n_types, figsize=(6*n_types, 5))
    if n_types == 1:
        axs = [axs]
    for it, (t, res) in enumerate(eval_results[dset_name].items()):
        axs[it].set_title(f"Dataset: {dset_name} - Type: {t}")
        axs[it].set_xlabel("Train Bits per spike")
        axs[it].set_ylabel("Val Bits per spike")
        axs[it].scatter(res['train']['bps'], res['val']['bps'], alpha=0.7)
        axs[it].plot([0, 5], [0, 5], 'k--', alpha=0.5)

        max_bps = max(max(res['train']['bps']), max(res['val']['bps']))
        axs[it].set_xlim(-.1, max_bps + 0.5)
        axs[it].set_ylim(-.1, max_bps + 0.5)
    plt.tight_layout()
    plt.show()

#%%


# 
# print(f"\nDataset: {model.names[dataset_idx]}")
# print(f"  Train samples: {len(train_data)}")
# print(f"  Val samples: {len(val_data)}")
# print(f"  Units: {len(dataset_config['cids'])}")

# %%

