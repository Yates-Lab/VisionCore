#!/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.


TODO:
1. make comparison to modulator only explicit and quantitative
2. are neurons not much better because both models are shit or both models are good?

"""

#%% Setup and Imports
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import sys
from pathlib import Path
sys.path.append('.')

import numpy as np
from eval_stack_multidataset import evaluate_model_multidataset
from DataYatesV1 import enable_autoreload

import matplotlib.pyplot as plt

from eval_stack_utils import argmin_subpixel, argmax_subpixel

enable_autoreload()

#%% Discover Available Models
from eval_stack_utils import scan_checkpoints
from gratings_analysis import run_gratings_analysis

def get_checkpoint_dir(experiment):
    """
    Get the checkpoint and cache directories for a given experiment.

    Parameters
    ----------
    experiment : str
        Name of the experiment ('backimage_only_120', 'gaborium_only_120', 'full_120')

    Returns
    -------
    checkpoint_dir : str
        Path to the checkpoint directory
    cache_dir : str
        Path to the cache directory
    """
    if experiment == 'backimage_only_120':
        checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage/checkpoints'
    elif experiment == 'gaborium_only_120':
        checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_gaborium/checkpoints'
    else:
        checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'

    cache_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_caches/" + experiment
    return checkpoint_dir, cache_dir
    
def get_all_results_by_experiment(experiment, models_to_compare):
    """
    Load all results for a given experiment.

    Parameters
    ----------
    experiment : str
        Name of the experiment ('backimage_only_120', 'gaborium_only_120', 'full_120')
    models_to_compare : list
        List of models to compare

    Returns
    -------
    all_results : dict
        Dictionary of all results
    """
    checkpoint_dir, cache_dir = get_checkpoint_dir(experiment)
    models_by_type = scan_checkpoints(checkpoint_dir)
    available_models = [m for m in models_to_compare if m in models_by_type]

    all_results = {}
    for model_type in available_models:
        print(f"\nLoading {model_type}...")
        
        results = evaluate_model_multidataset(
            model_type=model_type,
            analyses=['bps', 'ccnorm', 'saccade', 'qc'],
            checkpoint_dir=checkpoint_dir,
            save_dir=cache_dir,
            recalc=False,
            rescale=True,
            batch_size=64
        )
        all_results.update(results)
    
    # all_results = run_gratings_analysis(all_results, checkpoint_dir, cache_dir, recalc=True, batch_size=64)

    return all_results

#%% run bps eval for all models

# Define models to compare
models_to_compare = [ 'modulator_only_convgru', 'learned_res_small_gru', 'learned_res_small_none_gru']
experiments_to_run = ['backimage_only_120', 'gaborium_only_120', 'full_120']

# models_to_compare = ['learned_res_small_none_gru']
# experiments_to_run = ['backimage_only_120']

# A dictionary to hold the final results
all_results = {}

# Loop and run the complete analysis for each experiment with a single function call
for experiment in experiments_to_run:
    print(f"--- Running Full Analysis for: {experiment} ---")
    all_results[experiment] = get_all_results_by_experiment(experiment, models_to_compare)

# run gratings analysis
from gaborium_analysis import run_gaborium_analysis
# run gratings analysis
for experiment in experiments_to_run:
    checkpoint_dir, cache_dir = get_checkpoint_dir(experiment)
    all_results[experiment] = run_gratings_analysis(all_results[experiment], checkpoint_dir, cache_dir, recalc=False, batch_size=64)
    
    all_results[experiment] = run_gaborium_analysis(all_results[experiment], checkpoint_dir, cache_dir, recalc=False, batch_size=64, test_mode=False)

print("\nAll analyses complete.")

#%% Check BPS
from eval_stack_utils import bits_per_spike

model_name = 'learned_res_small_gru'
stim_type = 'gaborium'

plt.figure(figsize=(10,5))
for dataset_ix in range(0,20):
    experiment0 = 'backimage_only_120'

    robs = all_results[experiment0][model_name]['bps'][stim_type]['robs'][dataset_ix]
    rhat = all_results[experiment0][model_name]['bps'][stim_type]['rhat'][dataset_ix]
    dfs = all_results[experiment0][model_name]['bps'][stim_type]['dfs'][dataset_ix]
    bps_orig_0 = all_results[experiment0][model_name]['bps'][stim_type]['bps'][dataset_ix]

    bps_0 = bits_per_spike(rhat, robs, dfs)

    experiment1 = 'full_120'
    robs = all_results[experiment1][model_name]['bps'][stim_type]['robs'][dataset_ix]
    rhat = all_results[experiment1][model_name]['bps'][stim_type]['rhat'][dataset_ix]
    dfs = all_results[experiment1][model_name]['bps'][stim_type]['dfs'][dataset_ix]
    bps_orig_1 = all_results[experiment1][model_name]['bps'][stim_type]['bps'][dataset_ix]

    bps_1 = bits_per_spike(rhat, robs, dfs)

    plt.subplot(1,2,1)
    plt.plot(bps_0, bps_1, 'k.', alpha=.25)
    plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.xlabel(f'{experiment0} BPS')
    plt.ylabel(f'{experiment1} BPS')
    plt.title(f'{stim_type}')
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
             
    plt.subplot(1,2,2)
    plt.plot(bps_orig_0/bps_orig_1, 'k.', alpha=.25)
    # plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.xlabel(f'{experiment0} BPS')
    plt.ylabel(f'{experiment1} BPS')
    plt.title("Original BPS")
    # plt.xlim(-1, 3)
    plt.ylim(-1, 3)





#%% save all results as pickle 
# import pickle
# import os

# fname = 'all_results_120_analysis_w_datanames.pkl'
# with open(fname, 'wb') as f:
#     pickle.dump(all_results, f)

#%% Get depths using the manual labels

from classify_layer import get_depths, layer_splits, plot_laminar_boundaries, plot_laminar_data_across_sessions
depths = get_depths(all_results, 'standard', concatenate=True, map_linear=False, layer4c_only=False)
splits = layer_splits['standard']

#%% Plot all laminar boundaries on each session
plot_laminar_data_across_sessions(all_results, collapse_shanks=False, plot_latency_overlay=True)

#%% Extract QC data
from extract_functions import extract_bps_saccade, extract_ccnorm, extract_qc_spatial

experiments = list(all_results.keys())
print(f"Found {len(experiments)} experiments: {experiments}")
for experiment in experiments:
    print(f"Found {len(all_results[experiment].keys())} models for {experiment}")

experiment = experiments[0]
model_types = list(all_results[experiment].keys())
model_name = model_types[1]

_, waveforms, cids, dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results[experiment])


#%% Extract modulation indices from the gabor and grating stimuli
# data_ix = np.isin(dids , 'Allen_2022-02-16')
data_ix = np.ones_like(dids, dtype=bool)
def rescale(x):
    return 2*((x + 1) / (.3 + 1)) - 1

MI_gabors_robs = rescale(-np.concatenate(all_results[experiment][model_name]['sta']['modulation_index_robs']))
MI_gabors_rhat = rescale(-np.concatenate(all_results[experiment][model_name]['sta']['modulation_index_rhat']))

peak_lag_robs = np.concatenate(all_results[experiment][model_name]['sta']['peak_lag_subpixel_robs'])
peak_lag_rhat = np.concatenate(all_results[experiment][model_name]['sta']['peak_lag_subpixel_rhat'])

dt = 1000/120 # TODO: this switches a lot throughout the codebase and can create bugs between cells
iix = (contamination < 101) & data_ix
print(f"Number of valid indices: {np.sum(iix)}")

if isinstance(depths[0], list):
    # plot all deptsh but plot each experiment with a different color looping over depths
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(depths[0])))
    for i in [1]: #range(len(depths[0])):
        peak_lag = all_results[experiment][model_name]['sta']['peak_lag_subpixel_robs'][i]
        # ax.scatter(dt*peak_lag, depths[1][i], color=colors[i], alpha=.5, s=5)
        ax.scatter(dt*peak_lag, depths[1][i], color='k', alpha=.5, s=10)
    
    plot_laminar_boundaries(ax, splits)
    plt.title('Aligned Depth')
    plt.ylim(-400, 2000)
    plt.xlim(10, 80)
    plt.xlabel('Latency (ms)')

    depth = np.concatenate(depths[1])

else:
    plt.subplot(1,2,1)
    plt.plot(dt*peak_lag_robs[iix], depths[0][iix], 'k.', alpha=.5)
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.plot(dt*peak_lag_robs[iix], depths[1][iix], 'k.', alpha=.5)
    plot_laminar_boundaries(plt.gca(), splits)
    plt.title('Aligned Depth')
    # plt.ylim(0, 1250)
    plt.xlim(10, 80)
    plt.xlabel('Latency (ms)')

    depth = depths[1]


# depth[depth > 1100 ] = np.nan

#%%

# data_ix = np.ones_like(dids, dtype=bool)

from eval_stack_utils import scatter_kde_horizontal
MI_gratings_robs = 2*np.concatenate(all_results[experiment][model_name]['gratings']['modulation_index_robs'])-1
MI_gratings_rhat = 2*np.concatenate(all_results[experiment][model_name]['gratings']['modulation_index_rhat'])-1


MI_robs = (MI_gabors_robs + MI_gratings_robs) / 2
MI_rhat = (MI_gabors_rhat + MI_gratings_rhat) / 2

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(MI_gabors_robs, MI_gratings_robs, 'k.', alpha=.1)
plt.plot([-1, 1], [-1, 1], 'k--')

plt.xlabel('Gaborium MI (robs)')
plt.ylabel('Gratings MI (robs)')

plt.subplot(1,2,2)
plt.plot(MI_robs, MI_rhat, 'k.', alpha=.1)
plt.plot([-1, 1], [-1, 1], 'k--')
plt.xlabel('MI (robs)')
plt.ylabel('MI (rhat)')

plt.show()

iix = ~(np.isnan(MI_robs)) # | np.isnan(MI_rhat))
# su_ix = (contamination < 50) & (missing_pct < 25)
su_ix = contamination < 101
print(f"Number of valid indices: {np.sum(iix)}")
print(f"Number of single unit indices: {np.sum(su_ix)}")

iix = iix & su_ix & data_ix
# get correlation coefficient for valid indices and test for significance
from scipy.stats import pearsonr
r, p = pearsonr(MI_gabors_robs[iix], MI_gratings_robs[iix])
print(f"robs: r={r:.2f}, p={p:.2f}")

r, p = pearsonr(MI_robs[iix], MI_rhat[iix])
print(f"rhat: r={r:.2f}, p={p:.2f}")


plt.figure(figsize=(5, 5))
plt.plot(MI_robs[iix], depth[iix], 'k.', alpha=.5)
plot_laminar_boundaries(plt.gca(), splits)
plt.xlabel('MI (robs)')
plt.ylabel('Depth (um)')
plt.show()

plt.hist(MI_gratings_robs[iix], bins=np.linspace(-1, 1, 30), alpha=.5)
plt.hist(MI_gabors_robs[iix], bins=np.linspace(-1, 1, 30), alpha=.5)
plt.xlabel('Modulation Index (robs)')
plt.show()
 

#%%

print(f"\nQC data: {len(depth)} cells, contamination range: {np.nanmin(contamination):.1f}-{np.nanmax(contamination):.1f}%")

# get waveform data and use to find narrow, broad, and axonal units
wfs = []
for i in range(len(waveforms)):
    ch = np.argmax(np.var(waveforms[i], 0))
    wfs.append(waveforms[i].T[ch])

wfs = np.stack(wfs).T
wfs = wfs / np.max(np.abs(wfs), 0, keepdims=True)

start = 39 # start finding the peak and trough after the trigger
end = 70
tau_trough, val_trough = argmin_subpixel(wfs[start:end], 0)
tau_peak, val_peak = argmax_subpixel(wfs[start:end], 0)

#% plot a few examples to check
n_examples = 10
sx = int(np.ceil(np.sqrt(n_examples)))
sy = int(np.ceil(n_examples / sx))
fig, axes = plt.subplots(sx, sy, figsize=(16, 16))
for i, ax in enumerate(axes.flatten()):
    if i >= n_examples:
        ax.axis('off')
        continue
    j = np.random.randint(wfs.shape[1])
    ax.plot(wfs[:,j])
    ax.axvline(tau_trough[j]+start, color='k')
    ax.axvline(tau_peak[j]+start, color='r')
    ax.set_title(f'Cell {j}')
    ax.axis('off')


#%% get waveform stats
wf_dur= tau_peak - tau_trough

iix = (contamination<=25) & data_ix

plt.hist(wf_dur[iix], 20)

wf_separation_bins = [-np.inf, -12,  0, 12, 19, np.inf]

nclasses = len(wf_separation_bins) - 1
fig, ax = plt.subplots(1,nclasses, figsize=(10, 3), sharey=True, sharex=True)
wf_class = np.digitize(wf_dur, wf_separation_bins)

for i in np.unique(wf_class):
    ix = (wf_class==i) & iix
    ax[i-1].plot(wfs[:,ix], 'k', alpha = .1)
    ax[i-1].set_title(f'Class {i} n = {np.sum(ix)}')

plt.figure()
for i in np.unique(wf_class):
    ix = (wf_class==i) & iix
    cnt, bins = np.histogram(depth[ix], np.linspace(np.nanmin(depth), np.nanmax(depth), 50), density=True)
    plt.fill_betweenx(bins[:-1], cnt*400 + i, np.ones_like(cnt)*i, alpha=.5, color='k')
    
plot_laminar_boundaries(plt.gca(), splits)

depth_bins = [-np.inf, -150, 150, np.inf]
depth_class = np.digitize(depth, depth_bins)

waveforms_dict = {'wf': wfs, 'wf_dur': wf_dur, 'wf_class': wf_class, 'depth': depth, 'depth_class': depth_class, 'contamination': contamination}


wf_dt = 1000/30000

#%%

iix = (contamination < 50) & (wf_dur*wf_dt > 0) & ~np.isnan(MI_robs)

# iix = (contamination < 50) & ~np.isnan(MI_robs)
print(f"Number of valid indices: {np.sum(iix)}")

fig, axs = plt.subplots(1,2, figsize=(5, 8))
scatter_kde_horizontal(wf_dur[iix]*wf_dt, depth[iix], slice_spacing=100, bandwidth=.2, ax=axs[0])
axs[0].axvline(.4, color='k', linestyle='--')

axs[0].set_ylim(0, 1250)
axs[0].set_xlim(0.01, 1)
axs[0].set_xlabel('Peak - Trough (ms)')
axs[0].set_ylabel('Cortical Depth (um, 0 is deep)')
plot_laminar_boundaries(axs[0], splits)


plt.show()


#%% MI by depth for different waveform durations

iix = (contamination < 50) & (wf_dur*wf_dt < .4) & (wf_dur*wf_dt > 0) & ~np.isnan(MI_robs)
print(f"Number of valid indices: {np.sum(iix)}")

fig, ax = plt.subplots(1,3, figsize=(5, 5), sharey=True)
# scatter_kde_horizontal(MI_robs[iix], depth[iix], slice_spacing=100, ax=ax[0])
# scatter_kde_horizontal(peak_lag_robs[iix], depth[iix], slice_spacing=100, ax=ax[1])

ax[0].scatter(MI_robs[iix], depth[iix], s=10, facecolor='r', alpha = .5, edgecolor=None)
ax[1].scatter(peak_lag_robs[iix]*dt, depth[iix], s=10, facecolor='r', alpha = .5, edgecolor=None)
ax[2].scatter(wf_dur[iix]*wf_dt, depth[iix], s=10, facecolor='r', alpha = .5, edgecolor=None)
ax[2].axvline(.4, color='k', linestyle='--')
iix = (contamination < 50) & (wf_dur*wf_dt > .4) & ~np.isnan(MI_robs)
print(f"Number of valid indices: {np.sum(iix)}")

ax[0].scatter(MI_robs[iix], depth[iix], s=10, facecolor='b', alpha = .5, edgecolor=None)
ax[1].scatter(peak_lag_robs[iix]*dt, depth[iix], s=10, facecolor='b', alpha = .5, edgecolor=None)
ax[2].scatter(wf_dur[iix]*wf_dt, depth[iix], s=10, facecolor='b', alpha = .5, edgecolor=None)

plot_laminar_boundaries(ax[0], splits)
plot_laminar_boundaries(ax[1], splits)

ax[0].set_xlabel('MI (robs)')
ax[1].set_xlabel('Latency (ms)')
ax[2].set_xlabel('Peak - Trough (ms)')
ax[1].set_xlim(0, 80)
ax[1].set_ylim(0, 1250)
ax[0].set_ylim(0, 1250)
ax[0].axvline(0, color='k', linestyle='--')
plt.show()


#%% scatter colored by peak - trough
# Define colormap mapping for wf_dur*wf_dt
wf_dur_ms = wf_dur * wf_dt

sortby = 'wf_dur'

if sortby == 'wf_dur':
    vmin, vmid, vmax = 0.2, 0.4, 0.7
    wf_dur_norm = np.clip((wf_dur_ms - vmin) / (vmax - vmin), 0, 1)
    cmap = 'turbo'
elif sortby == 'MI':
    vmin, vmid, vmax = -.5, 0, .5
    wf_dur_norm = np.clip((MI_robs - vmin) / (vmax - vmin), 0, 1)
    cmap = 'coolwarm'
elif sortby == 'latency':
    vmin, vmid, vmax = 20, 35, 60
    wf_dur_norm = np.clip((peak_lag_robs*dt - vmin) / (vmax - vmin), 0, 1)
    cmap = 'turbo'

iix = (contamination < 50) & ~np.isnan(MI_robs) & (wf_dur >0)
print(f"Number of valid indices: {np.sum(iix)}")

fig, ax = plt.subplots(1,3, figsize=(6, 5), sharey=True)

# Create scatter plots colored by waveform duration
im0 = ax[0].scatter(MI_robs[iix], depth[iix], c=wf_dur_norm[iix], s=10, alpha=0.7, cmap=cmap, vmin=0, vmax=1)
im1 = ax[1].scatter(peak_lag_robs[iix]*dt, depth[iix], c=wf_dur_norm[iix], s=10, alpha=0.7, cmap=cmap, vmin=0, vmax=1)
im2 = ax[2].scatter(wf_dur_ms[iix], depth[iix], c=wf_dur_norm[iix], s=10, alpha=0.7, cmap=cmap, vmin=0, vmax=1)

# Add vertical lines
ax[0].axvline(0, color='k', linestyle='--')


# Add laminar boundaries
plot_laminar_boundaries(ax[0], splits)
plot_laminar_boundaries(ax[1], splits)

# Set labels and limits
ax[0].set_xlabel('MI (robs)')
ax[1].set_xlabel('Latency (ms)')
ax[2].set_xlabel('Peak - Trough (ms)')
ax[1].set_xlim(0, 80)
# ax[0].set_ylim(0, 1250)

# Add colorbar
# plt.colorbar(im2, ax=ax, label='Waveform Duration (ms)')
plt.show()

#%% saccade fixation cycle


import torch.nn as nn
from eval_stack_utils import shift_tensor

def shift_robs(batch, saccade_info):
    dt = 1/120
    sac_dur_bins = []
    for isac in range(len(saccade_info)):
        sac_dur = saccade_info[isac]['end_time']-saccade_info[isac]['start_time']
        sac_dur_bins.append(-np.round(sac_dur/dt).astype(int))

    dtype = batch['robs'].dtype
    robs_shifted = shift_tensor(batch['robs'], torch.tensor(sac_dur_bins), shift_dim=1, batch_dim=0, fill_value=np.nan)
    rhat_shifted = shift_tensor(batch['rhat'], torch.tensor(sac_dur_bins), shift_dim=1, batch_dim=0, fill_value=np.nan)
    batch['robs'] = robs_shifted.to(dtype)
    batch['rhat'] = rhat_shifted.to(dtype)
    nans = torch.isnan(batch['robs'])
    batch['dfs'][nans] = 0
    batch['rhat'][nans] = 0
    batch['robs'][nans] = 0
    
    return batch

def get_rbar(results, dataset_idx, offset_align=False):
    labels = ['robs', 'dfs', 'rhat']
    batch = {labels[i]: torch.tensor(results[labels[i]][dataset_idx]) for i in range(len(labels))}
    if offset_align:
        batch = shift_robs(batch, results['saccade_info'][dataset_idx])

    rbar = (batch['robs']*batch['dfs']).sum(0) / batch['dfs'].sum(0)
    return rbar.detach().numpy()

def get_ll(results, dataset_idx, offset_align=False):
    labels = ['robs', 'dfs', 'rhat']
    batch = {labels[i]: torch.tensor(results[labels[i]][dataset_idx]) for i in range(len(labels))}
    
    if offset_align:
        batch = shift_robs(batch, results['saccade_info'][dataset_idx])

    loss = nn.PoissonNLLLoss(log_input=False, full=False, reduction='none')
    ll = ((loss(batch['rhat'], batch['robs']) * batch['dfs']).sum(0) / batch['dfs'].sum(0)).detach().numpy()
    return -ll

def get_ll_null(results, dataset_idx, offset_align=False):
    labels = ['robs', 'dfs', 'rhat']
    batch = {labels[i]: torch.tensor(results[labels[i]][dataset_idx]) for i in range(len(labels))}
    
    if offset_align:
        batch = shift_robs(batch, results['saccade_info'][dataset_idx])

    loss = nn.PoissonNLLLoss(log_input=False, full=False, reduction='none')
    base_firing_rate = torch.ones_like(batch['rhat'])*batch['robs'].mean((0,1), keepdim=True)
    ll = ((loss(base_firing_rate, batch['robs']) * batch['dfs']).sum(0) / batch['dfs'].sum(0)).detach().numpy()
    return -ll


def get_Rnorm(R, tbins):
    from scipy.ndimage import gaussian_filter1d

    R = gaussian_filter1d(R, 1, axis=0)
    R0 = R[tbins < 0].mean(0)
    R = (R - R0)/R.max(0)

    return R

# Meshgrid of x (time) and y (depths)
def plot_pcolor(x, y, I, ax=None, vmin=None, vmax=None, cmap='coolwarm'):
    X, Y = np.meshgrid(x, y)

    if ax is None:
        ax = plt.gca()
    pc = ax.pcolormesh(X, Y, I,    
                    vmin=vmin, vmax=vmax,
                    shading='auto', cmap=cmap)

    return pc

saccades_dict = {}

results = all_results[experiment]

for model_id in range(len(results.keys())):
    model_name = list(results.keys())[model_id]
    
    saccades_dict[model_name] = {}

    for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
        saccades_dict[model_name][stim_type]={}
        # Get BPS and saccade data for gaborium
        bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, results)
        
        Rdata = get_Rnorm(saccade_robs, saccade_time_bins)
        Rmodel = get_Rnorm(saccade_rhat, saccade_time_bins)

        time_ix = (saccade_time_bins > 0) & (saccade_time_bins < 22)
        tau_trough_robs, val_trough_robs = argmin_subpixel(Rdata[time_ix], 0)
        tau_peak_robs, val_peak_robs = argmax_subpixel(Rdata[time_ix], 0)
        tau_trough_rhat, val_trough_rhat = argmin_subpixel(Rmodel[time_ix], 0)
        tau_peak_rhat, val_peak_rhat = argmax_subpixel(Rmodel[time_ix], 0)

        saccades_dict[model_name][stim_type] = {'robs': saccade_robs, 'rhat': saccade_rhat, 
                'time_bins': saccade_time_bins, 'cids': cids, 'dids': dids,
                'robs_normed': Rdata, 'rhat_normed': Rmodel,
                'tau_trough_robs': tau_trough_robs, 'val_trough_robs': val_trough_robs,
                'tau_peak_robs': tau_peak_robs, 'val_peak_robs': val_peak_robs,
                'tau_trough_rhat': tau_trough_rhat, 'val_trough_rhat': val_trough_rhat,
                'tau_peak_rhat': tau_peak_rhat, 'val_peak_rhat': val_peak_rhat}

#%% get LLR 
stim_type = 'backimage'

num_datasets = len(all_results[experiment][model_name]['saccade'][stim_type]['robs'])
ll_onset = {}
for experiment in all_results.keys():
    ll_onset[experiment] = {}
    for model_name in all_results[experiment].keys():
        ll_onset[experiment][model_name] = [get_ll(all_results[experiment][model_name]['saccade'][stim_type], dataset_idx) for dataset_idx in range(num_datasets)]
    
    ll_onset[experiment]['null'] = [get_ll_null(all_results[experiment][model_name]['saccade'][stim_type], dataset_idx) for dataset_idx in range(num_datasets)]
#%%
ll_offset = {}
for experiment in all_results.keys():
    ll_offset[experiment] = {}
    for model_name in all_results[experiment].keys():
        ll_offset[experiment][model_name] = [get_ll(all_results[experiment][model_name]['saccade'][stim_type], dataset_idx, offset_align=True) for dataset_idx in range(num_datasets)]
    
    ll_offset[experiment]['null'] = [get_ll_null(all_results[experiment][model_name]['saccade'][stim_type], dataset_idx, offset_align=True) for dataset_idx in range(num_datasets)]

#%%
rbar_onset = [get_rbar(all_results[experiment][model_name]['saccade'][stim_type], dataset_idx) for dataset_idx in range(num_datasets)]
rbar_offset = [get_rbar(all_results[experiment][model_name]['saccade'][stim_type], dataset_idx, offset_align=True) for dataset_idx in range(num_datasets)]

#%%

def get_llr_analyses(model_name, alignment, sortby, layer, cell_type, wf_type):

    if alignment == 'onset':
        ll = ll_onset.copy()
        rbar_raw = np.concatenate(rbar_onset,1)
    elif alignment == 'offset':
        ll = ll_offset.copy()
        rbar_raw = np.concatenate(rbar_offset,1)

    t_axis = np.arange(*all_results[experiment][model_name]['saccade'][stim_type]['win'][0])*(1000/120)
    ll_mod = np.concatenate(ll['full_120']['modulator_only_convgru'], 1)
    ll_vis = np.concatenate(ll['full_120']['learned_res_small_none_gru'], 1)
    ll_full = np.concatenate(ll['full_120']['learned_res_small_gru'], 1)
    ll_null = np.concatenate(ll['full_120']['null'], 1)

    # _ = plt.plot( (ll_vis-ll_mod) / np.log(2), 'k', alpha=.1)
    llr = (ll_vis-ll_mod) / np.log(2)
    llr_null = (ll_vis-ll_null) / np.log(2)
    llr_bps = llr / np.nanmean(rbar_raw, 0, keepdims=True)
    llr_null_ps = llr_null / np.nanmean(rbar_raw, 0, keepdims=True)
    

    mask = (t_axis > -50) & (t_axis < 200)

    llr = llr[mask,:]
    llr_bps = llr_bps[mask,:]
    llr_null = llr_null[mask,:]

    rbar = saccades_dict[model_name][stim_type]['robs_normed']
    peak = saccades_dict[model_name][stim_type]['tau_peak_rhat']
    dt = 1000/120
    rbar = rbar[mask,:]
    rbar_raw = rbar_raw[mask,:]

    iix = (contamination < 25) & ~np.isnan(depth) # & ~np.isnan(MI_robs) 

    bounds = [0, 1250]
    class_iix = np.ones_like(depth, dtype=bool)

    if layer == '6':
        bounds = [splits['exclusion_bottom'], splits['5/6']]
    elif layer == '5':
        bounds = [splits['5/6'], splits['4/5']]
    elif layer == '4':
        bounds = [splits['4/5'], splits['3/2']]
    elif layer == '4C':
        bounds = [splits['4/5'], splits['4C']]
    elif layer == '4AB':
        bounds = [splits['4C'], splits['3/2']]
    elif layer == '23':
        bounds = [splits['3/2'], splits['exclusion_top']]

    class_iix = (depth > bounds[0]) & (depth < bounds[1])
    iix = iix & class_iix

    if cell_type == 'simple':
        iix = iix & (MI_robs > 0)
    elif cell_type == 'complex':
        iix = iix & (MI_robs < 0)

    if wf_type == 'fast':
        iix = iix & (wf_dur*wf_dt < .4)
    elif wf_type == 'slow':
        iix = iix & (wf_dur*wf_dt > .4)


    if sortby == 'depth':
        ind = np.argsort(depth[iix])
        sorter = depth[iix][ind]
    elif sortby == 'latency':
        ind = np.argsort(peak[iix]*dt)
        sorter = peak[iix][ind]*dt

    fig, axs = plt.subplots(1,5,figsize=(15, 7), sharey=True, sharex=True)

    use_pcolor = True
    if use_pcolor:
        cmap = 'coolwarm'
        pc = plot_pcolor(t_axis[mask], sorter, rbar_raw[:,iix][:,ind].T, ax=axs[0], cmap=cmap)
        plt.colorbar(pc, fraction=0.03, pad=0.01)

        pc = plot_pcolor(t_axis[mask], sorter, rbar[:,iix][:,ind].T, ax=axs[1], vmin=-.7, vmax=.7, cmap=cmap)
        plt.colorbar(pc,fraction=0.03, pad=0.01)

        pc = plot_pcolor(t_axis[mask], sorter, llr[:,iix][:,ind].T*120, ax=axs[2], vmin=-12, vmax=12, cmap=cmap)
        plt.colorbar(pc,fraction=0.03, pad=0.01)
        pc = plot_pcolor(t_axis[mask], sorter, llr_bps[:,iix][:,ind].T, ax=axs[3], vmin=-.5, vmax=.5, cmap=cmap)
        plt.colorbar(pc, fraction=0.03, pad=0.01)

        pc = plot_pcolor(t_axis[mask], sorter, llr_null[:,iix][:,ind].T, ax=axs[4], vmin=-.1, vmax=.5, cmap=cmap)
        plt.colorbar(pc, fraction=0.03, pad=0.01)

        if sortby == 'depth' and layer == 'all':
            for ax in axs:
                plot_laminar_boundaries(ax, splits)

        if sortby == 'depth':
            axs[0].set_ylim(bounds[0], bounds[1])
        elif sortby == 'latency':
            axs[0].set_ylim(35, 160)
    else:
        axs[0].imshow(rbar_raw[:,iix][:,ind].T, aspect='auto', cmap='coolwarm', interpolation='none')
        axs[1].imshow(rbar[:,iix][:,ind].T, aspect='auto', cmap='coolwarm', interpolation='none', vmin=-.7, vmax=.7)
        axs[2].imshow(llr[:,iix][:,ind].T*120, aspect='auto', cmap='coolwarm', interpolation='none', vmin=-12, vmax=12)
        axs[3].imshow(llr_bps[:,iix][:,ind].T, aspect='auto', cmap='coolwarm', interpolation='none', vmin=-.5, vmax=.5)
        axs[4].imshow(llr_null[:,iix][:,ind].T, aspect='auto', cmap='coolwarm', interpolation='none')
        

    axs[0].set_title('Firing Rate (raw)')
    axs[1].set_title('Firing Rate (normed)')
    axs[2].set_title('LLR (visual - modulator)')
    axs[3].set_title('LLR (bps)')
    axs[4].set_title('BPS (vis)')
    axs[0].set_ylabel('Sorted by {}'.format(sortby))
    axs[2].set_xlabel(f'Time from saccade {alignment} (ms)')

    r_onset = np.concatenate(rbar_onset,1)
    r_offset = np.concatenate(rbar_offset,1)
    rdiff = r_onset - r_offset
    rdiff = rdiff[mask,:]
    rdiff = rdiff[:,iix][:,ind]

    r_onset = r_onset[mask,:]
    r_onset = r_onset[:,iix][:,ind]
    r_offset = r_offset[mask,:]
    r_offset = r_offset[:,iix][:,ind]

    # set white space between axes
    plt.subplots_adjust(wspace=.3)
    plt.suptitle(f'{stim_type} layer {layer}')
    plt.savefig(f'saccade_figures/saccade_{alignment}_{layer}_{cell_type}_{wf_type}_llr_{stim_type}_{sortby}.png')

    # store everything in out dict
    out_dict = {'r_onset': r_onset, 'r_offset': r_offset,
                'rdiff': rdiff, 't_axis': t_axis[mask], 'sorter': sorter,
                'llr': llr[:,iix][:,ind]*120, 'llr_bps': llr_bps[:,iix][:,ind], 'llr_null': llr_null[:,iix][:,ind]}

    return out_dict

#%% plot main LLR analyses for all models
_ = get_llr_analyses(model_name, 'onset', 'latency', 'all', 'all', 'all')
_ = get_llr_analyses(model_name, 'onset', 'depth', 'all', 'all', 'all')
_ = get_llr_analyses(model_name, 'offset', 'latency', 'all', 'all', 'all')
_ = get_llr_analyses(model_name, 'offset', 'depth', 'all', 'all', 'all')
#%%
model_name = 'learned_res_small_gru'
alignment = 'onset'
sortby = 'latency'

out_dict = {}
options = []
for layer in ['6', '5', '4', '23']:
    for cell_type in ['simple', 'complex']:
        for wf_type in ['fast', 'slow']:
            option = f'{layer}_{cell_type}_{wf_type}'
            print(f'Processing {option}')
            try:
                out_dict[option] = get_llr_analyses(model_name, alignment, sortby, layer, cell_type, wf_type)
                print(f'✅ {option}')
                options.append(option)
            except Exception as e:
                print(f'❌ {option}: {e}')

#%%
def plot_error_shade(x, y, se, ax=None, color='k', alpha=.2, label=None):
    if ax is None:
        ax = plt.gca()
    h = ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y-se*2, y+se*2, color=color, alpha=alpha)
    return h

plot_field = 'llr'
N = len(options)
sx = int(np.ceil(np.sqrt(N)))
sy = int(np.ceil(N/sx))
fig, axs = plt.subplots(sy, sx, figsize=(16, 16), sharex=True, sharey=True)
for i, option in enumerate(options):
    ax = axs.flatten()[i]
    r_onset = out_dict[option][plot_field]
    if plot_field == 'r_onset':
        r_onset = get_Rnorm(r_onset, out_dict[option]['t_axis'])

    N = r_onset.shape[1]
    _ = ax.plot(out_dict[option]['t_axis'], r_onset, 'k', alpha=.25)
    _ = plot_error_shade(out_dict[option]['t_axis'], r_onset.mean(1), r_onset.std(1)/np.sqrt(N), ax=ax, color='k')
    ax.axvline(0, color='k', linestyle='--')
    ax.axhline(0, color='k', linestyle='--')
    # display N = in corner of plot
    ax.text(0.05, 0.95, f'N = {N}', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(option.replace('_', ' '))

ax.set_ylim(-10, 20)
plt.savefig(f'saccade_figures/saccade_{alignment}_layers_and_cell_types.png')
#%% compare saccade onset aligned and offset aligned
cmap = plt.cm.tab10

plot_field = 'r_onset'
layers = ['6', '5', '4', '23']
N = len(layers)
sx = int(np.ceil(np.sqrt(N)))
sy = int(np.ceil(N/sx))
fig, axs = plt.subplots(sy, sx, figsize=(16, 16), sharex=True, sharey=True)

for ilayer, layer in enumerate(layers):
    ax = axs.flatten()[ilayer]
    ax.set_title(f'Layer {layer}')

    for i, option in enumerate(options):
        if layer in option:
            r_onset = out_dict[option][plot_field]
            if plot_field == 'r_onset':
                r_onset = get_Rnorm(r_onset, out_dict[option]['t_axis'])
            N = r_onset.shape[1]
            # _ = ax.plot(out_dict[option]['t_axis'], r_onset, 'k', alpha=.25)
            if 'complex' in option:
                color = 'r'
                linewidth = 1
            elif 'simple' in option:
                color = 'b'
                linewidth = 2
            _ = ax.plot(out_dict[option]['t_axis'], r_onset, color=color, alpha=.5, linewidth=linewidth)
            # plot_error_shade(out_dict[option]['t_axis'], r_onset.mean(1), r_onset.std(1)/np.sqrt(N), ax=ax, color=color, label=option.replace('_', ' '))
    
    
    ax.axhline(0, color='k', linestyle='--')
    ax.plot(0, 0, 'r.', label='complex')
    ax.plot(0, 0, 'b.', label='simple')
    ax.legend()

if plot_field == 'llr':
    ax.set_ylim(-10, 50)
# ax.set_ylim(-1, 1)

plt.axvline(0, color='k', linestyle='--')

    # display N = in corner of plot
    # ax.text(0.05, 0.95, f'N = {N}', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # ax.set_title(option.replace('_', ' '))

#remove hardcoding and make name based on extract_model_type
#%% STE Latency and examples
'''
Compare models trained on backimage only and full dataset


'''

experiment1 = 'backimage_only_120'#'full_120'
ste_robs = torch.from_numpy(np.concatenate(all_results[experiment1][model_name]['sta']['Z_STE_robs'], -1))
sta_robs = torch.from_numpy(np.concatenate(all_results[experiment1][model_name]['sta']['Z_STA_robs'], -1))

model_name = 'learned_res_small_gru'
ste_rhat1 = torch.from_numpy(np.concatenate(all_results[experiment1][model_name]['sta']['Z_STE_rhat'], -1))
sta_rhat1 = torch.from_numpy(np.concatenate(all_results[experiment1][model_name]['sta']['Z_STA_rhat'], -1))

experiment0 = 'full_120'
ste_rhat0 = torch.from_numpy(np.concatenate(all_results[experiment0][model_name]['sta']['Z_STE_rhat'], -1))
sta_rhat0 = torch.from_numpy(np.concatenate(all_results[experiment0][model_name]['sta']['Z_STA_rhat'], -1))

bps_full = np.concatenate(all_results[experiment0][model_name]['bps']['gaborium']['bps'])
bps_backimage = np.concatenate(all_results[experiment1][model_name]['bps']['gaborium']['bps'])

lag_robs, _ = argmax_subpixel(ste_robs.var((1,2)), 0)
lag_rhat0, _ = argmax_subpixel(ste_rhat0.var((1,2)), 0)
lag_rhat1, _ = argmax_subpixel(ste_rhat1.var((1,2)), 0)

lag_robs_sta, _ = argmax_subpixel(sta_robs.var((1,2)), 0)
lag_rhat0_sta, _ = argmax_subpixel(sta_rhat0.var((1,2)), 0)
lag_rhat1_sta, _ = argmax_subpixel(sta_rhat1.var((1,2)), 0)

dt = 1000/120
lag_robs = lag_robs*dt
lag_rhat0 = lag_rhat0*dt
lag_rhat1 = lag_rhat1*dt
lag_robs_sta = lag_robs_sta*dt
lag_rhat0_sta = lag_rhat0_sta*dt
lag_rhat1_sta = lag_rhat1_sta*dt

iix = (contamination < 25) & (bps_backimage > .1)

plot_examples = True

if plot_examples:

    N = 100
    fig, axs = plt.subplots(N, 1, figsize=(8, N), sharex=True, sharey=True)

    for i, cc in enumerate(np.where(iix)[0][:N]):

        ste_robs_plotting = ste_robs.reshape(16*51, 51, -1)[:,:,cc].T
        ste_rhat0_plotting = ste_rhat0.reshape(16*51, 51, -1)[:,:,cc].T
        ste_rhat1_plotting = ste_rhat1.reshape(16*51, 51, -1)[:,:,cc].T

        def minmax(x):
            # return x
            return (x - x.min()) / (x.max() - x.min())

        ste_robs_plotting = minmax(ste_robs_plotting)
        ste_rhat0_plotting = minmax(ste_rhat0_plotting)
        ste_rhat1_plotting = minmax(ste_rhat1_plotting)

        # concatenate
        ste_plotting = torch.concat([ste_robs_plotting, ste_rhat0_plotting, ste_rhat1_plotting], 0)

        axs[i].imshow(ste_plotting, aspect='auto', cmap='gray')
        # axis off
        axs[i].axis('off')

# tight layout collapse whitespace
plt.tight_layout()




plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(lag_robs[iix], lag_rhat0[iix], 'k.', alpha=.25)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Data')
plt.ylabel('Full Training')

plt.title('STE Latency')

plt.subplot(1,2,2)
plt.plot(lag_robs[iix], lag_rhat1[iix], 'k.', alpha=.25)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Data')
plt.ylabel('Backimage Only')

#%% Plot simple cells
iix = (contamination < 50) & (bps_backimage > .1) & (MI_robs > -0.2)

print(f"Number of simple cells: {np.sum(iix)}")

plot_examples = True
lags = np.array([0, 15])*dt

if plot_examples:

    N = np.minimum(np.sum(iix), 100)
    fig, axs = plt.subplots(N, 1, figsize=(8, N), sharex=True, sharey=True)

    for i, cc in enumerate(np.where(iix)[0][:N]):

        sta_robs_plotting = sta_robs.reshape(16*51, 51, -1)[:,:,cc].T
        sta_rhat0_plotting = sta_rhat0.reshape(16*51, 51, -1)[:,:,cc].T
        sta_rhat1_plotting = sta_rhat1.reshape(16*51, 51, -1)[:,:,cc].T

        def minmax(x):
            # return x
            return (x - x.min()) / (x.max() - x.min())

        sta_robs_plotting = minmax(sta_robs_plotting)
        sta_rhat0_plotting = minmax(sta_rhat0_plotting)
        sta_rhat1_plotting = minmax(sta_rhat1_plotting)

        # concatenate
        sta_plotting = torch.concat([sta_robs_plotting, sta_rhat0_plotting, sta_rhat1_plotting], 0)

        axs[i].imshow(sta_plotting, aspect='auto', cmap='gray', extent=[lags[0], lags[1], 0, sta_plotting.shape[1]])
        axs[i].axvline(lag_robs[cc], color='r', linestyle='--')
        axs[i].axvline(lag_robs_sta[cc], color='g', linestyle='--')
        # axis off
        axs[i].axis('off')

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(lag_robs_sta[iix], lag_robs[iix], 'k.', alpha=.25)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('STA Latency')
plt.ylabel('STE Latency')

plt.subplot(1,2,2)
plt.plot(lag_robs_sta[iix], lag_rhat0_sta[iix], 'k.', alpha=.25)
plt.plot(lag_robs_sta[iix], lag_rhat1_sta[iix], 'g.', alpha=.25)
plt.plot(plt.xlim(), plt.xlim(), 'k--')

#%%
iix = (contamination < 101) & (bps_backimage > .1) & (MI_robs > -0.5)

N = np.minimum(np.sum(iix), 100)
sx = int(np.ceil(np.sqrt(N)))
sy = int(np.ceil(N/sx))

fig, axs = plt.subplots(sx, sy, figsize=(26, 16))

for i, cc in enumerate(np.where(iix)[0][:N]):
    ax = axs.flatten()[i]
    lag = np.argmax(ste_robs[:,:,:,cc].std((1,2)))
    sta_robs_plotting = sta_robs[lag, :,:, cc]
    lag = np.argmax(ste_rhat1[:,:,:,cc].std((1,2)))
    sta_rhat1_plotting = sta_rhat1[lag, :,:, cc]

    def minmax(x):
        # return x
        return (x - x.min()) / (x.max() - x.min())

    sta_robs_plotting = minmax(sta_robs_plotting)
    sta_rhat1_plotting = minmax(sta_rhat1_plotting)

    # concatenate
    sta_plotting = torch.concat([sta_robs_plotting, torch.zeros((51,2)),sta_rhat1_plotting], dim=1)

    ax.imshow(sta_plotting, aspect='auto', cmap='gray')
    # axis off
    ax.axis('off')

    
# plt.plot(lag_robs, lag_rhat1)

#%%
fig, ax = plt.subplots(1,3, figsize=(10, 7), sharey=True, sharex=True)

dt = 1000/120
iix = contamination < 50
ax[0].scatter(lag_robs[iix], depth[iix], s=20, facecolor='k', alpha = .25, edgecolor='w')
ax[1].scatter(lag_rhat0[iix], depth[iix], s=20, facecolor='k', alpha = .25, edgecolor='w')
ax[2].scatter(lag_rhat1[iix], depth[iix], s=20, facecolor='k', alpha = .25, edgecolor='w')
ax[1].set_xlim(0, 80)

for i in range(3):
    plot_laminar_boundaries(ax[i], splits)

ax[0].set_title('Data')
ax[1].set_title(experiment0)
ax[2].set_title(experiment1)

ax[0].set_xlabel('Stimulus Latency')
ax[0].set_ylabel('Depth (um)')

# plt.savefig(fig_dir / 'sta_latency.pdf')

# #%%
# ste_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['Z_STE_robs'], -1))
# ste_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[0]]['sta']['Z_STE_rhat'], -1))
# ste_rhat0 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['Z_STE_rhat'], -1))
# sta_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['Z_STA_robs'], -1))
# sta_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[0]]['sta']['Z_STA_rhat'], -1))
# sta_rhat0 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['Z_STA_rhat'], -1))

# peak_lag = np.argmax(ste_robs.var((1,2)),0)
# # sta_rhat1[,...].var((0,1))

# inds = np.argsort(sta_robs.var((0,1,2)))
# inds = np.array(inds)[::-1].tolist()
# inds = inds[200:]

# n_cells = sta_robs.shape[-1]

# max_plots_per_fig = 20**2

# n_cells = np.minimum(n_cells, max_plots_per_fig)

# sx = np.floor(np.sqrt(n_cells)).astype(int)
# sy = np.ceil(n_cells / sx).astype(int)
# fig, axs = plt.subplots(sy, sx, figsize=(16*2, 16))

# lag = 4

# H = sta_robs.shape[1]

# for i in range(n_cells):
#     ax = axs.flatten()[i]
#     # v = sta_robs[lag,:,:,i].abs().max()
#     # I = torch.concat([sta_robs[lag,:,:,i], torch.ones(H,1), sta_rhat[lag,:,:,i]], 1)
#     # ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)
#     j = inds[i]
#     I = torch.concat([sta_robs[lag,:,:,j], torch.zeros(H,1), sta_rhat1[lag,:,:,j]], 1)
#     ax.imshow(I, cmap='gray_r', interpolation='none')
#     # ax.set_title(f'Cell {i}')
#     ax.axis('off')

# plt.tight_layout()

# plt.show()


# #%% plot STA Latency vs. Saccade Peak and tough latency



# dt = 1000/120
# from scipy.stats import pearsonr

# wf_class_breakout = False

# for stim_type in ['backimage']: # @, 'gaborium', 'fixrsvp', 'gratings']:

#     fig, ax = plt.subplots(1, figsize=(6, 6))
#     model_name = list(saccades_dict.keys())[1]

#     try:
        
            
#         if wf_class_breakout:
#             n_loop = np.unique(waveforms_dict['wf_class'])
#         else:
#             n_loop = [1]

#         for iclass in n_loop:
#             if wf_class_breakout:
#                 iix = (waveforms_dict['wf_class']==iclass) & (waveforms_dict['contamination'] < 100)
#             else:
#                 iix = (waveforms_dict['contamination'] < 100)
            
#             print(f"n = {iix.sum()}")
        
#             saccade_tau_peak = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
#             saccade_tau_trough = saccades_dict[model_name][stim_type]['tau_trough_robs'][iix]*dt
            
#             # throw out outliers
#             id = (saccade_tau_peak < 150) & (saccade_tau_peak > 20)
#             saccade_tau_peak = saccade_tau_peak[id]
#             saccade_tau_trough = saccade_tau_trough[id]

#             sta_tau = lag_robs[iix]*dt
#             sta_tau = sta_tau[id]

#             ax.scatter(saccade_tau_peak, sta_tau, s=20, edgecolors='w', alpha=.5)
            

            
#             ax.set_title('Saccade vs. STA Latency')
#             ax.set_xlabel('Saccade Latency (ms)')
#             ax.set_ylabel('STA Latency (ms)')
#             ax.set_xlim(0, 150)
#             ax.set_ylim(0, 100)
#             ax.plot([0, 150], [0, 150], 'k--')

#             ax.scatter(saccade_tau_trough, sta_tau, s=20, edgecolors='w', alpha=.5)
            
#             ax.set_title('Saccade vs. STA Latency')
#             ax.set_xlabel('Saccade Latency (ms)')
#             ax.set_ylabel('STA Latency (ms)')
#             # ax[0].set_xlim(0, 150)
#             # ax[0].set_ylim(0, 100)
#             # ax[0].plot(plt.xlim(), plt.xlim(), 'k--')
#             dsim = ( (saccade_tau_peak + .1) / (sta_tau + .1))
#             ix = (dsim < 1) & (dsim > .5)
#             ix = ix & (saccade_tau_peak < 100) & (saccade_tau_trough > 0)
#             ax.scatter(saccade_tau_trough[ix], sta_tau[ix], s=20, edgecolors='w', alpha=.5)
#             ax.scatter(saccade_tau_peak[ix], sta_tau[ix], s=20, edgecolors='w', alpha=1)
#             the_list = np.where(iix)[0][id][ix]
#             # ax.plot(np.stack([saccade_tau_trough[ix], saccade_tau_peak[ix]]), np.stack([sta_tau[ix], sta_tau[ix]]), 'k')
#         plt.savefig(fig_dir / f'sta_saccade_stim_latency_comparison_{stim_type}.pdf')
#     except Exception as e:
#         print(f"❌ STA vs. Saccade latency comparison failed: {e}")
#         import traceback
#         traceback.print_exc()


# #%%
# # the_list = the_list[waveforms_dict['wf_class'][the_list]>2]

# saccade_tau_peak = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
# saccade_tau_trough = saccades_dict[model_name][stim_type]['tau_trough_robs'][iix]*dt

# # throw out outliers
# id = (saccade_tau_peak < 150) & (saccade_tau_peak > 20)
# saccade_tau_peak = saccade_tau_peak[id]
# saccade_tau_trough = saccade_tau_trough[id]

# sta_tau = lag_robs[iix]*dt
# sta_tau = sta_tau[id]

# dep = waveforms_dict['depth'][iix][id]
# plt.subplot(1,2,1)
# plt.plot(sta_tau, -dep, '.')
# plt.plot(lag_robs[the_list]*dt, -waveforms_dict['depth'][the_list], '.')
# plt.xlim(0, 100)
# plt.axhline(150, color='k', linestyle='--')
# plt.axhline(-150, color='k', linestyle='--')
# plt.subplot(1,2,2)
# plt.plot(saccade_tau_peak, -dep, '.')
# plt.plot(saccade_tau_trough, -dep, '.')
# plt.plot(sta_tau, -dep, '.')
# plt.xlim(0, 100)
# plt.axhline(150, color='k', linestyle='--')
# plt.axhline(-150, color='k', linestyle='--')

# #%%
# # the_list = the_list[waveforms_dict['wf_class'][the_list]>2]
# plt.plot(waveforms_dict['wf'][:,the_list])

# #%%

# print(f"Depth n = {len(depth)}")
# print(f"Contamination n = {len(contamination)}")
# print(f"Waveforms Class n = {len(wf_class)}")
# print(f"Depth Class n = {len(depth_class)}")


# #%% Does not run. Needs gratings_dict
# dt = 1000/120

# model_name = list(saccades_dict.keys())[0]
# print(f"Model name: {model_name}")
# stim_type = 'backimage'
# iix = (waveforms_dict['contamination'] < 20) & (waveforms_dict['wf_class'] > 2)
# iix = iix & (gratings_dict['ori_snr'] > 1) & (gratings_dict['sf_snr'] > 1)
# saccade_tau = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
# plt.plot(saccade_tau, gratings_dict['sf_tuning_data'][iix], '.')
# plt.ylim(2, 5)

# #%%
# dt = 1000/120
# from scipy.stats import pearsonr

# wf_class_breakout = False

# for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:

#     fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=True, sharex=True)
#     model_name = list(saccades_dict.keys())[1]

#     try:
#         for i, model_name in enumerate(saccades_dict.keys()):
            
#             if wf_class_breakout:
#                 n_loop = np.unique(waveforms_dict['wf_class'])
#             else:
#                 n_loop = [1]

#             for iclass in n_loop:
#                 if wf_class_breakout:
#                     iix = (waveforms_dict['wf_class']==iclass) & (waveforms_dict['contamination'] < 100)
#                 else:
#                     iix = (waveforms_dict['contamination'] < 100)
                
#                 print(f"n = {iix.sum()}")
            
#                 saccade_tau_data = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
#                 saccade_tau_model = saccades_dict[model_name][stim_type]['tau_peak_rhat'][iix]*dt
#                 # throw out outliers
#                 id = (saccade_tau_data < 150) & (saccade_tau_data > 20) & (saccade_tau_model < 150) & (saccade_tau_model > 20)
#                 saccade_tau_data = saccade_tau_data[id]
#                 saccade_tau_model = saccade_tau_model[id]
                
#                 r2 = pearsonr(saccade_tau_data, saccade_tau_model)[0]**2
#                 print(f"R2 for {model_name} = {r2}")
#                 # show text for r^2 in figure 
#                 ax[i].text(0.05, 0.95, f'r² = {r2:.3f}', transform=ax[i].transAxes,
#                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#                 ax[i].scatter(saccade_tau_data, saccade_tau_model, s=20, edgecolors='w', alpha=.5)
#                 ax[i].plot(plt.xlim(), plt.xlim(), 'k--')
#                 ax[i].set_title(get_model_name(model_name))
#                 ax[i].set_xlabel('Data (ms)')
#                 ax[i].set_ylabel('Model (ms)')
#                 ax[i].set_xlim(20, 150)
#                 ax[i].set_ylim(20, 150)

#         plt.savefig(fig_dir / f'saccade_latency_comparison_{stim_type}.pdf')
#     except Exception as e:
#         print(f"❌ Saccade latency comparison failed: {e}")

# #%% Compute saccade metrics
# dt = 1000/120
# win = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['win'][dset]
# sac_time_bins = np.arange(win[0], win[1])*dt

# sortby = 'latency'
# for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
#     fig, ax = plt.subplots(1,3, figsize=(10, 5), sharey=True, sharex=True)

#     iix = (waveforms_dict['contamination'] < 100)

#     model_name = list(saccades_dict.keys())[1]
#     Rdata = saccades_dict[model_name][stim_type]['robs_normed']
#     Rmodel = saccades_dict[model_name][stim_type]['rhat_normed']
#     iix = np.where(iix & ~np.isnan(np.sum(Rdata,0)))[0]
#     Rdata = Rdata[:,iix]
#     Rmodel = Rmodel[:,iix]

#     if sortby == 'latency':
#         ind = np.argsort(np.argmax(Rmodel, 0))
#     elif sortby == 'depth':
#         ind = np.argsort(waveforms_dict['depth'][iix])

#     ax[0].imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[sac_time_bins[0], sac_time_bins[-1], 0, len(ind)])
#     ax[0].set_title(f'Data ({stim_type})')
#     ax[0].set_xlim(-50, 250)
#     ax[0].axvline(0, color='k', linestyle='--')
#     ax[0].set_ylabel('Neuron (sorted by peak latency)')
#     ax[0].set_xlabel('Time from Saccade Onset (ms)')

#     for i, model_name in enumerate(saccades_dict.keys()):
#         Rmodel = saccades_dict[model_name][stim_type]['rhat_normed'][:,iix]
#         ax[i+1].imshow(Rmodel[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[sac_time_bins[0], sac_time_bins[-1], 0, len(ind)])
#         ax[i+1].set_title(get_model_name(model_name))
#         ax[i+1].set_xlim(-50, 250)
#         ax[i+1].axvline(0, color='k', linestyle='--')

#     plt.savefig(fig_dir / f'saccade_sorted_{sortby}_{stim_type}.pdf')

# #%% Get some summary stats on saccades

# stim_type = 'backimage'
# model_id = 1
# dset = 0
# dt = 1/120
# win = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['win'][dset]
# sac_time_bins = np.arange(win[0], win[1])*dt

# prev = []
# next = []
# amp = []
# dur = []
# for dset in range(len(all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['robs'])):
#     prev.append([s['time_previous'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])
#     next.append([s['time_next'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])
#     amp.append([s['A'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])
#     dur.append([s['end_time']-s['start_time'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])

# prev = np.concatenate(prev)
# next = np.concatenate(next)
# amp = np.concatenate(amp)
# dur = np.concatenate(dur)

# plt.figure(figsize=(3,1))
# plt.hist(next+np.random.rand(len(next))/100, np.linspace(0, 1, 150), color='gray')
# plt.hist(dur+np.random.rand(len(dur))/100, np.linspace(0, .1, 150), color='k', alpha=.8)

# # turn off the spikes
# plt.box(False)
# plt.xlim(0, .5)
# plt.xlabel('Saccade Duration (s)')


# plt.figure(figsize=(3,1))
# plt.hist(next+np.random.rand(len(next))/100, np.linspace(0, 1, 150), color='gray')
# plt.hist(dur+np.random.rand(len(dur))/100, np.linspace(0, .1, 150), color='k', alpha=.8)

# plt.savefig(fig_dir / 'saccade_duration.pdf')

# # turn off the spikes
# plt.box(False)
# plt.xlim(-0.05, .25)
# plt.xlabel('Saccade Duration (s)')

# plt.savefig(fig_dir / 'saccade_duration_clipped.pdf')

# #%%
# iix = (gratings_dict['ori_snr'] > 1)# & (contamination < 20)
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plt.plot(gratings_dict['ori_tuning_data'][iix], gratings_dict['ori_tuning_model'][iix], '.', alpha=.1)
# plt.xlim(15, 160)
# plt.ylim(15, 160)
# plt.xlabel('Data (deg)')
# plt.ylabel('Model (deg)')
# plt.title('Orientation Tuning')

# plt.subplot(1,2,2)
# plt.plot(gratings_dict['sf_tuning_data'][iix], gratings_dict['sf_tuning_model'][iix], '.', alpha=.1)
# plt.xlim(1, 8)
# plt.ylim(1, 8)
# plt.xlabel('Data (cyc/deg)')
# plt.ylabel('Model (cyc/deg)')
# plt.title('Spatial Frequency Tuning')

# plt.savefig(fig_dir / 'gratings_tuning.pdf')

# #%% What is the modulator doing?

# fig1, ax1 = plt.subplots(1,3,figsize=(10, 3), sharey=True, sharex=True)
# fig2, ax2 = plt.subplots(1,3,figsize=(10, 3), sharey=True, sharex=True)

# wf_class_breakout = False

# for istim, stim_type in enumerate(['backimage', 'gaborium', 'gratings']):
    
#     Rmodel1 = saccades_dict[list(saccades_dict.keys())[1]][stim_type]['rhat']
#     Rmodel0 = saccades_dict[list(saccades_dict.keys())[0]][stim_type]['rhat']

#     iix = (waveforms_dict['contamination'] < 100) & ~np.isnan(Rmodel1.sum(0))
#     Rdelta = Rmodel1[:,iix] / Rmodel0[:,iix]
#     ind = np.argsort(np.argmin(Rdelta, 0))
#     ax1[istim].imshow(Rdelta[:,ind].T, aspect='auto', interpolation='none', cmap='coolwarm', vmin=.5, vmax=1.5, extent=[sac_time_bins[0], sac_time_bins[-1], 0, len(ind)])
#     ax1[istim].set_xlim(-.05, .25)

#     wf_class = waveforms_dict['wf_class'][iix]
#     depth_class = waveforms_dict['depth_class'][iix]

    
#     if wf_class_breakout:
#         n_loop = [3, 4, 5] #[1,2]# [2, 3, 4] #np.unique(wf_class)
#     else:
#         n_loop = [1]

#     cmap = plt.cm.get_cmap("tab10", max(n_loop))
#     linestyles = ['--', '-', ':']

#     for iclass in n_loop:
#         for idepth in [1,2,3]: # 4 is nan
#             if wf_class_breakout:
#                 ix = (wf_class==iclass) & (waveforms_dict['depth_class'][iix]==idepth)
#             else:
#                 ix = (waveforms_dict['depth_class'][iix]==idepth)
#             # ix = (wf_class==iclass) & (waveforms_dict['depth_class'][iix]==idepth)
#             mu = Rdelta[:, ix].mean(1)
#             se = Rdelta[:, ix].std(1)/np.sqrt(ix.sum())
#             ax2[istim].plot(sac_time_bins, mu, color=cmap(iclass-1), linestyle=linestyles[idepth-1], label=f'Depth {idepth}')
#             # plot errorbars as SE mean
#             ax2[istim].fill_between(sac_time_bins, mu-se*2, mu+se*2, color=cmap(iclass-1), alpha=.2)
#     ax2[istim].axhline(1, color='k', linestyle='--')
#     ax2[istim].set_title(stim_type)
#     ax1[istim].set_title(stim_type)
#     ax2[istim].set_xlim(-.05, .25)
#     ax2[istim].set_xlabel('Time from Saccade Onset (s)')
    
#     for iax in range(len(ax2)):
#         ax2[iax].set_xlim(-.05, .25)
#            # plt.plot(sac_time_bins, Rdelta[:, wf_class==iclass].mean(1), label=iclass)
    
# fig1.savefig(fig_dir / f'modulator_effect_{wf_class_breakout}.pdf')
# fig2.savefig(fig_dir / f'modulator_effect_by_depth_{wf_class_breakout}.pdf')


# #%% Recompute CCnorm Values



# i = 0
# #%%
# i += 5
# plt.fill_between(np.arange(ccnorm_1['rbarhat'].shape[0]), ccnorm_1['rbar'][:,ind[i]], facecolor='k', alpha=.5, edgecolor='k')
# plt.plot(ccnorm_1['rbarhat'][:,ind[i]], 'r')
# plt.plot(ccnorm_1['rbarhat_range'][0,:,ind[i]], 'r--')
# plt.plot(ccnorm_1['rbarhat_range'][1,:,ind[i]], 'b--')


# #%% Same plot but with histogram difference
# for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
#     bps = []
#     model_names = []
#     plt.figure(figsize=(5,5))
#     for imodel in range(len(all_results.keys())):
#         model_name = list(all_results.keys())[imodel]
#         bps_ = np.concatenate(all_results[model_name]['bps'][stim_type]['bps'])
#         # bps_ = bps_[contamination < 20]
#         bps.append(np.maximum(bps_, 0))
#         model_names.append(get_model_name(model_name))

#     for iclass in np.unique(wf_class):
#         ix = (wf_class==iclass) & iix
#         cnt, bins = np.histogram(bps[1][ix]-bps[0][ix], np.linspace(-1, 1, 50), density=True)
#         cnt = cnt / np.max(cnt)
#         bin_centers = (bins[:-1] + bins[1:]) / 2
#         plt.fill_between(bin_centers, cnt + iclass, np.ones_like(cnt)*iclass, alpha=.5, color='k')
#     plt.axvline(0, color='k', linestyle='--')
#     # ix = (wf_class==i) & iix
#     # cnt, bins = np.histogram(depth[ix], np.linspace(np.nanmin(depth), np.nanmax(depth), 50), density=True)
#     # plt.fill_betweenx(bins[:-1], cnt*400 + i, np.ones_like(cnt)*i, alpha=.5, color='k')
#     # plt.axhline(-150, color='k', linestyle='--')
#     # plt.axhline(150, color='k', linestyle='--')

# #%% plot by depth

# experiment = 'full_120'
# base_model = 'learned_res_small_none_gru'
# test_experiments = ['full_120', 'backimage_only_120', 'gaborium_only_120']
# test_model = ['learned_res_small_gru', 'learned_res_small_gru', 'learned_res_small_gru']
# stim_types = ['backimage', 'gaborium', 'fixrsvp', 'gratings']
# fig, axs = plt.subplots(len(test_experiments),len(stim_types),figsize=(12,3*len(test_experiments)), sharey=True)


# for i, stim_type in enumerate(stim_types):
#     base_bps = np.concatenate(all_results[experiment][base_model]['bps'][stim_type]['bps'])
#     base_bps = np.maximum(base_bps, -.1)
#     n_models = 1
#     # n_models = len(all_results.keys())
    
#     for imodel in range(len(test_experiments)):
#         ax = axs[imodel, i]
#         experiment = test_experiments[imodel]
#         model_name = test_model[imodel]
#         bps_ = np.concatenate(all_results[experiment][model_name]['bps'][stim_type]['bps'])
#         bps_ = np.maximum(bps_, -.1)
#         # bps_ = bps_[contamination < 20]
#         bps = (bps_ - base_bps) #/ np.abs(base_bps)
#         ax.scatter(bps, depth, alpha=.25, color='k', edgecolors='w')
#         ax.set_ylabel(f'{experiment}')
    
#     ax.set_title(stim_type)
#     ax.set_xlim(-.5, 1)
#     ax.set_ylim(0, 1250)

# depth_bins = np.linspace(np.nanmin(depth), np.nanmax(depth), 10)

# # %%


# n_models = 2
# stim_type = 'gaborium'
# bps_comparison = {}
# for model_id in range(len(all_results.keys())):
#     model_name = list(all_results.keys())[model_id]
#     # Get BPS and saccade data for gaborium
#     bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
#     bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}


# good_ix = np.where(~np.isnan(np.sum(bps_comparison[0]['robs'],0)))[0]
# # good_ix = np.where((contamination < 20) & (wf_class>2))[0]
# Rdata = get_Rnorm(bps_comparison[0]['robs'][:,good_ix], saccade_time_bins)
# Rmodel = []
# for model_id in range(len(all_results.keys())):
#     Rmodel.append(get_Rnorm(bps_comparison[model_id]['rhat'][:,good_ix], saccade_time_bins))

# ind = np.argmax(Rmodel[0], 0)
# ind = np.argsort(ind)
# # ind = np.argsort(depth[good_ix])

# dt = 1000/120
# tbins = saccade_time_bins*dt
# plt.figure(figsize=(10, 5))
# plt.subplot(1,n_models+1,1)
# vmin = np.nanmin(Rdata)
# vmax = np.nanmax(Rdata)
# plt.imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)], vmin=vmin, vmax=vmax)
# plt.title(f'Data {stim_type}')
# plt.xlim(-50, 250)
# plt.axvline(0, color='k', linestyle='--')

# for i in range(n_models):
    
#     plt.subplot(1,n_models+1,i+2)
#     plt.imshow(Rmodel[i][:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)], vmin=vmin, vmax=vmax)
#     # plt.set_yticks([])
#     plt.title(get_model_name(bps_comparison[i]['name']))
#     if i==0:
#         plt.xlabel('Time from saccade onset (ms)')
#     plt.axvline(0, color='k', linestyle='--')
#     plt.xlim(-50, 250)
# plt.show()

# _ = plt.plot(saccade_time_bins*dt, Rmodel[1]-Rmodel[0], 'k', alpha=.1)
# plt.xlim(-50, 250)
# plt.axhline(0)

# # %%

# stim_type = 'gratings'

# llrs = []
# for dset in range(len(all_results[list(all_results.keys())[0]]['saccade'][stim_type]['rhat'])):
#     rhat1 = all_results[list(all_results.keys())[1]]['saccade'][stim_type]['rhat'][dset]
#     rhat0 = all_results[list(all_results.keys())[0]]['saccade'][stim_type]['rhat'][dset]
#     robs = all_results[list(all_results.keys())[0]]['saccade'][stim_type]['robs'][dset]
#     dfs = torch.tensor(all_results[list(all_results.keys())[0]]['saccade'][stim_type]['dfs'][dset])


#     loss_1 = poisson_loss(torch.tensor(rhat1), torch.tensor(robs), reduction='none')  # shape [trials, neurons]
#     loss_0 = poisson_loss(torch.tensor(rhat0), torch.tensor(robs), reduction='none')

#     llr = (dfs *(loss_1 - loss_0)).sum(0) / dfs.sum(0)

#     llrs.append(llr.numpy())
#     _ = plt.plot(-llr, 'k', alpha=.1)


# llr = np.concatenate(llrs, 1)
# plt.plot(-llr, 'k', alpha=.1)
# plt.show()

# # %%
# stim_type = 'backimage'
# model_id = 1
# dset = 0
# dt = 1/120
# win = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['win'][dset]
# sac_time_bins = np.arange(win[0], win[1])*dt

# prev = [s['time_previous'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]
# next = [s['time_next'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]
# amp = [s['A'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]
# dur = [s['end_time']-s['start_time'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]

# robs = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['robs'][dset]
# rhat_base = all_results[list(all_results.keys())[0]]['saccade'][stim_type]['rhat'][dset]
# rhat_gru = all_results[list(all_results.keys())[1]]['saccade'][stim_type]['rhat'][dset]

# cid = 0
# #%%
# sorter = prev
# bin_edges = np.percentile(sorter, [0, 25, 50, 75, 100])

# cid += 1
# fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

# cmap = plt.cm.get_cmap("coolwarm", len(bin_edges)-1)
# for ibin in range(len(bin_edges)-1):
    
#     ind = np.where((sorter > bin_edges[ibin]) & (sorter < bin_edges[ibin+1]))[0]
    
#     ax[0].plot(sac_time_bins, np.mean(robs[ind,:,cid], 0), color=cmap(ibin))     
#     ax[1].plot(sac_time_bins, np.mean(rhat_base[ind,:,cid], 0), color=cmap(ibin))     
#     ax[2].plot(sac_time_bins, np.mean(rhat_gru[ind,:,cid], 0), color=cmap(ibin)) 

# ax[0].set_xlim(-.2, .5)
#     # plt.imshow(robs[ind,:,10], aspect='auto', interpolation='none', cmap='gray_r')
# # plt.hist(prev, np.arange(0, 60)*dt)

# # %%
# ind = np.argsort(next)
# plt.imshow(robs[ind,:,10], aspect='auto', interpolation='none', cmap='gray_r')
# # %%


# # %%

# ste_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_robs'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:])
# ste_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:])
# ste_rhat0 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:])

# lag_robs, _ = argmax_subpixel(ste_robs.var((1,2)), 0)
# lag_rhat0, _ = argmax_subpixel(ste_rhat0.var((1,2)), 0)
# lag_rhat1, _ = argmax_subpixel(ste_rhat1.var((1,2)), 0)

# fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

# dt = 1000/120
# iix = contamination < 20
# ax[0].scatter(lag_robs[iix]*dt, -depth[iix], s=10, facecolor='k', alpha = .25, edgecolor='w')
# ax[1].scatter(lag_rhat0[iix]*dt, -depth[iix], s=10, facecolor='k', alpha = .25, edgecolor='w')
# ax[2].scatter(lag_rhat1[iix]*dt, -depth[iix], s=10, facecolor='k', alpha = .25, edgecolor='w')
# ax[1].set_xlim(10, 60)
# # ax[1].set_xlim(0.01, 0.07)


# Rdata
# # %%
# Rmodel = []
# for model_id in range(2):
#     model_name = list(all_results.keys())[model_id]

#     bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
#     bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}

#     Rdata = get_Rnorm(bps_comparison[0]['robs'], saccade_time_bins)
#     rmodel = get_Rnorm(bps_comparison[model_id]['rhat'], saccade_time_bins)
#     print(rmodel.shape)
#     Rmodel.append(rmodel)

# #%%

# time_ix = (saccade_time_bins > 0) & (saccade_time_bins < 17)
# tau_trough_data, val_trough_data = argmin_subpixel(Rdata[time_ix], 0)
# tau_peak_data, val_peak_data = argmax_subpixel(Rdata[time_ix], 0)
# tau_trough_base, val_trough_base = argmin_subpixel(Rmodel[0][time_ix], 0)
# tau_peak_base, val_peak_base = argmax_subpixel(Rmodel[0][time_ix], 0)
# tau_trough_gru, val_trough_gru = argmin_subpixel(Rmodel[1][time_ix], 0)
# tau_peak_gru, val_peak_gru = argmax_subpixel(Rmodel[1][time_ix], 0)

# fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)
# ax[0].plot(tau_trough_data*dt, -val_trough_data, 'r.', alpha=.1)
# ax[0].plot(tau_peak_data*dt, val_peak_data, 'b.', alpha=.1)
# ax[0].set_title('Data')

# ax[1].plot(tau_trough_base*dt, -val_trough_base, 'r.', alpha=.1)
# ax[1].plot(tau_peak_base*dt, val_peak_base, 'b.', alpha=.1)
# ax[1].set_title('Base')

# ax[2].plot(tau_trough_gru*dt, -val_trough_gru, 'r.', alpha=.1)
# ax[2].plot(tau_peak_gru*dt, val_peak_gru, 'b.', alpha=.1)
# ax[2].set_title('GRU')

# ax[1].set_xlim(0, 150)

# # %%

# fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

# iix = (contamination < 20)
# iix = iix & (tau_trough_data > 1)
# iix = iix & (tau_peak_data < 15)
# ax[0].scatter(tau_trough_data[iix]*dt, -depth[iix], alpha=.1, facecolor='r', edgecolor='w')
# ax[0].scatter(tau_peak_data[iix]*dt, -depth[iix], alpha=.1, facecolor='b', edgecolor='w')
# ax[0].set_title('Data')

# ax[1].scatter(tau_trough_base[iix]*dt, -depth[iix], alpha=.1, facecolor='r', edgecolor='w')
# ax[1].scatter(tau_peak_base[iix]*dt, -depth[iix], alpha=.1, facecolor='b', edgecolor='w')
# ax[1].set_title('Base')

# ax[2].scatter(tau_trough_gru[iix]*dt, -depth[iix], alpha=.1, facecolor='r', edgecolor='w')
# ax[2].scatter(tau_peak_gru[iix]*dt, -depth[iix], alpha=.1, facecolor='b', edgecolor='w')
# ax[2].set_title('GRU')

# # ax[1].set_xlim(0, 150)


# # plt.plot(tau_trough_data, lag_robs, '.')
# # plt.xlim(0, 20)
# # plt.ylim(0, 10)
# # %%
# plt.plot(tau_trough_data[iix]*dt, tau_peak_data[iix]*dt, '.')
# plt.plot(tau_trough_base[iix]*dt, tau_peak_base[iix]*dt, '.')
# plt.plot(tau_trough_gru[iix]*dt, tau_peak_gru[iix]*dt, '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k--')
# plt.xlabel('Trough (ms)')
# plt.ylabel('Peak (ms)')
# plt.title('Latency (ms)')
# plt.show()
# # %%

# r = Rdata[:,iix]
# plt.plot(r[:,tau_trough_data[iix]>tau_peak_data[iix]])

# # %%

# #%% track all STAs

# sta_robs = np.concatenate(all_results[model_name]['sta']['sta_robs'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]
# sta_rhat = np.concatenate(all_results[model_name]['sta']['sta_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]

# ste_robs = np.concatenate(all_results[model_name]['sta']['ste_robs'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]
# ste_rhat = np.concatenate(all_results[model_name]['sta']['ste_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]

# sta_robs = torch.from_numpy(sta_robs)
# sta_rhat = torch.from_numpy(sta_rhat)
# ste_robs = torch.from_numpy(ste_robs)
# ste_rhat = torch.from_numpy(ste_rhat)

# n_cells = sta_robs.shape[-1]

# max_plots_per_fig = 10**2

# n_cells = np.minimum(n_cells, max_plots_per_fig)

# sx = np.floor(np.sqrt(n_cells)).astype(int)
# sy = np.ceil(n_cells / sx).astype(int)
# fig, axs = plt.subplots(sy, sx, figsize=(16*2, 16))

# lag = 4

# H = sta_robs.shape[1]

# for i in range(n_cells):
#     ax = axs.flatten()[i]
#     # v = sta_robs[lag,:,:,i].abs().max()
#     # I = torch.concat([sta_robs[lag,:,:,i], torch.ones(H,1), sta_rhat[lag,:,:,i]], 1)
#     # ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)

#     I = torch.concat([ste_robs[lag,:,:,i], torch.zeros(H,1), ste_rhat[lag,:,:,i]], 1)
#     ax.imshow(I, cmap='gray_r', interpolation='none')
#     # ax.set_title(f'Cell {i}')
#     ax.axis('off')

# plt.tight_layout()

# plt.show()







# ccnorms = []
# model_names = []
# plt.figure(figsize=(6,3))
# for imodel in range(len(all_results.keys())):
#     model_name = list(all_results.keys())[imodel]
#     ccnorm = np.concatenate(all_results[model_name]['ccnorm']['fixrsvp']['ccnorm'])
#     iix = (contamination < 20) & (depth > 100) & (depth < 300)
#     ccnorm = ccnorm[contamination < 20]
#     ccnorms.append(ccnorm)
#     model_names.append(get_model_name(model_name))
    

# fig, ax = custom_boxplot(
#         ccnorms,
#         labels=model_names,
#         title="CCNORM (fixrsvp)",
#         xlabel="Model",
#         ylabel="CC NORM",
#         sig_spacing_factor=1.5
#     )
# # plt.ylim(.5,.75)
# plt.show()
# %%
