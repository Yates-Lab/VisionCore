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

#%% Discover Available Models
print("Discovering available models...")
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_6/checkpoints"
# checkpoint_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_240_backimage/checkpoints"
# checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'
models_by_type = scan_checkpoints(checkpoint_dir, verbose=False)

print(f"Found {len(models_by_type)} model types:")
for model_type, models in models_by_type.items():
    if models:
        best_model = models[0]
        if best_model.get('metric_type') == 'bps' and best_model.get('val_bps') is not None:
            best_metric = f"best BPS: {best_model['val_bps']:.4f}"
        else:
            best_metric = f"best loss: {best_model['val_loss']:.4f}"
        print(f"  {model_type}: {len(models)} models ({best_metric})")
    else:
        print(f"  {model_type}: 0 models")

#%% LOAD A MODEL
import os
checkpoint_path = None
# checkpoint_path = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage/checkpoints/learned_res_small_gru_optimized_aa_ddp_bs256_ds30_lr1e-3_wd1e-3_corelrscale1.0_warmup5/last.ckpt'
# checkpoint_path = os.path.join(checkpoint_dir, 'learned_res_small_film_ddp_bs256_ds30_lr1e-3_wd1e-5_corelrscale1.0_warmup10_zip/last.ckpt')
# model_type = 'resnet'
model_type = 'dense_concat_convgru'
model, model_info = load_model(
        model_type=model_type,
        model_index=3, # none for best model
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.model.eval()
model.model.convnet.use_checkpointing = False 

model = model.to(device)

plt.plot(model.model.frontend.temporal_conv.weight.squeeze().detach().cpu().T)
model.model.convnet.stem.components.conv.plot_weights()

#%%

from eval.eval_stack_multidataset import eval_stack_single_dataset

# During training - simple and clean!
results = eval_stack_single_dataset(
    model=model,
    dataset_idx=0,
    analyses=['bps', 'ccnorm', 'saccade']
)

#%%
cc = 0
s = 0
#%%
cc += 1
r = results['saccade']['backimage']['robs'][:,:,cc]
win = results['saccade']['backimage']['win']
dt = 1/240
plt.imshow(r, cmap='gray_r', interpolation='none', aspect='auto', extent=[win[0]*dt, win[1]*dt, 0, r.shape[0]])
plt.xlim(0, .25)

#%%
s += 1 
robs = results['saccade']['backimage']['robs'][s].T
rhat = results['saccade']['backimage']['rhat'][s].T
plt.imshow(np.concatenate([robs, rhat], 0), cmap='gray_r', interpolation='none', aspect='auto')

#%%
# Access results
val_bps = results['bps']['val']
ccnorm = results['ccnorm']['ccnorm']
#%%

ccnorm

#%%
train_data, val_data, dataset_config = load_single_dataset(model, 0)
stim_types_in_dataset = [d.metadata['name'] for d in train_data.dsets]

#%%

plt.imshow(results['bps']['fixrsvp']['rhat'].T, cmap='gray_r', interpolation='none', aspect='auto')

stim_indices = get_stim_inds('fixrsvp', train_data, val_data)
data = val_data.shallow_copy()
data.inds = stim_indices

dset_idx = np.unique(stim_indices[:,0]).item()
time_inds = data.dsets[dset_idx]['psth_inds'].numpy()
trial_inds = data.dsets[dset_idx]['trial_inds'].numpy()

[plt.axvline(x, color='r', linestyle='--') for x in np.where(time_inds==0)[0]]
plt.xlim(0, 1000)


# %%
from eval.eval_stack_utils import get_fixrsvp_trials
robs_trial, rhat_trial, dfs_trial = get_fixrsvp_trials(
            model, results['bps'], 0, train_data, val_data)
        
# %%
cc += 1
if cc >= robs_trial.shape[2]:
    cc = 0
_ = plt.plot(np.nanmean(robs_trial, (0))[:,cc])
_ = plt.plot(np.nanmean(rhat_trial, (0))[:,cc])
# %%
robs = results['bps']['fixrsvp']['robs']
rhat = results['bps']['fixrsvp']['rhat']
stim_indices = get_stim_inds('fixrsvp', train_data, val_data)

#%%
data = val_data.shallow_copy()
data.inds = stim_indices

dset_idx = np.unique(stim_indices[:,0]).item()
time_inds = data.dsets[dset_idx]['psth_inds'].numpy()
trial_inds = data.dsets[dset_idx]['trial_inds'].numpy()
unique_trials = np.unique(trial_inds)

#%%
n_trials = len(unique_trials)
n_time = np.max(time_inds).item()+1
n_units = data.dsets[dset_idx]['robs'].shape[1]
robs_trial = np.nan*np.zeros((n_trials, n_time, n_units))
rhat_trial = np.nan*np.zeros((n_trials, n_time, n_units))
dfs_trial = np.nan*np.zeros((n_trials, n_time, n_units))

for itrial in range(n_trials):
    print(f"Trial {itrial}/{n_trials}")

    trial_idx = np.where(trial_inds == unique_trials[itrial])[0]
    eval_inds = np.where(np.isin(stim_indices[:,1], trial_idx))[0]
    data_inds = trial_idx[np.where(np.isin(trial_idx, stim_indices[:,1]))[0]]

    assert torch.all(robs[eval_inds] == data.dsets[dset_idx]['robs'][data_inds]).item(), 'robs mismatch'

    robs_trial[itrial, time_inds[data_inds]] = robs[eval_inds]
    rhat_trial[itrial, time_inds[data_inds]] = rhat[eval_inds]
    dfs_trial[itrial, time_inds[data_inds]] = data.dsets[dset_idx]['dfs'][data_inds]

    # return robs_trial, rhat_trial, dfs_trial
# %%
# cc += 1
np.isfinite(robs_trial[:,:,0])
plt.imshow(robs_trial[:,:150,cc], cmap='gray_r', interpolation='none')
# %%
m = np.nanmean(np.isfinite(robs_trial), axis=2)>0
last = m.shape[1]-1 - np.argmax(m[:, ::-1], axis=1)
last[~m.any(1)] = -1   # rows with no finite values -> -1

# %%
plt.imshow(robs_trial[last,:150,cc], cmap='gray_r', interpolation='none')
# %%
