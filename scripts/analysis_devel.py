#!/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import torch._dynamo 
torch._dynamo.config.suppress_errors = True # suppress dynamo errors

import sys
sys.path.append('..')

import numpy as np
from DataYatesV1 import enable_autoreload, get_free_device, prepare_data

import matplotlib.pyplot as plt
from eval.eval_stack_utils import load_single_dataset, get_stim_inds, evaluate_dataset, argmin_subpixel, argmax_subpixel
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()

device = get_free_device()


#%%
import torch

def autocov_fft(x, max_lag=None, unbiased=True):
    """
    x: [T, N] (each column a neuron), already demeaned
    returns: [L+1, N] autocovariance from lag 0..L
    """
    T, N = x.shape
    if max_lag is None:
        max_lag = T-1
    nfft = 1 << (2*T-1).bit_length()          # next pow2 >= 2T-1
    Xf = torch.fft.rfft(torch.nn.functional.pad(x, (0,0,0,nfft-T)), n=nfft, dim=0)
    S = (Xf.conj()*Xf).real                   # periodogram in time domain after IFFT → autocov (unnormalized)
    r = torch.fft.irfft(S, n=nfft, dim=0).real[:max_lag+1]  # [L+1, N], this equals sum_{t} x[t]x[t+k]
    if unbiased:
        denom = torch.arange(T, T-max_lag-1, -1, device=x.device).unsqueeze(1)  # T, T-1, ..., T-L
    else:
        denom = torch.full((max_lag+1,1), T, device=x.device)
    return r / denom  # autocovariance

def psd_periodogram(x, Δt):
    """
    x: [T, N], demeaned (counts or rates)
    returns f: [F], Sxx: [F, N] one-sided PSD with units per Hz
    """
    T, N = x.shape
    Xf = torch.fft.rfft(x, dim=0)                     # [F, N]
    S = (Δt / T) * (Xf.conj()*Xf).real               # two-sided PSD per Hz
    # make one-sided
    F = S.shape[0]
    if T % 2 == 0:  # even T includes Nyquist
        S[1:F-1] *= 2.0
    else:
        S[1:F] *= 2.0
    f = torch.fft.rfftfreq(T, d=Δt)                  # Hz
    return f, S

#%%
from models.config_loader import load_dataset_configs
import os

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_backimage_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

#%%

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
# %%

train_datasets['dataset_1'].dsets[0].metadata['sess']

robs = train_datasets['dataset_1'].dsets[0]['robs'].to(torch.float32)
r_true = autocov_fft(robs, max_lag=40)

# --- shuffle correction ---
# Shuffle time bins independently for each neuron
robs_shuff = robs[torch.randperm(robs.shape[0])]
r_shuffle = autocov_fft(robs_shuff, max_lag=40)

# --- normalize to get correlation (unitless) ---
r_diff = r_true - r_shuffle
r_norm = r_diff / r_true[0]

# --- plot ---
Δt = 1/240
lags = torch.arange(41) * Δt * 1000  # ms
plt.figure(figsize=(6,4))
plt.plot(lags, (r_true / r_true[0]).mean(1).cpu(), label='True', color='C0')
plt.plot(lags, (r_shuffle / r_true[0]).mean(1).cpu(), label='Shuffle', color='C1', linestyle='--')
plt.plot(lags, (r_diff / r_true[0]).mean(1).cpu(), label='Shuffle-corrected', color='C2')
plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.legend()
plt.tight_layout()
plt.show()

#%%
# plt.figure(figsize=(2, 20))
_ = plt.plot(r_diff/r_diff[0][None,:] + .5*torch.arange(r_diff.shape[1])[None,:])

# %%
Δt = 1/240
R = robs / Δt
Rc = R - R.mean(dim=0, keepdim=True)
f, Sxx = psd_periodogram(Rc, Δt)   # Sxx units: (spikes/s)^2 / Hz
# %%

plt.plot(f, Sxx)

# %%
cc += 1
plt.plot(np.arange(r_diff.shape[0])/240*1000, r_diff[:,cc])
plt.plot(np.arange(r_diff.shape[0])/240*1000, r_true[:,cc])

# %%
batch_size = 1256
inds = np.random.randint(0, robs.shape[0]-batch_size, size=1)
inds = inds + np.arange(batch_size)
r_true = autocov_fft(robs[inds], max_lag=40)


plt.plot(robs[inds,:].mean(1))

#%%

plt.subplot(1,2,1)
plt.imshow((robs[inds,:]-robs[inds,:].mean(1, keepdim=True)).T, aspect='auto', cmap='gray_r', interpolation='none')
plt.subplot(1,2,2)
_ = plt.plot(r_true[1:])

#%%



# %%
