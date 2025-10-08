#!/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

#%% Setup and Imports
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import sys
from pathlib import Path
sys.path.append('.')

import numpy as np
from eval_stack_multidataset import evaluate_model_multidataset
from DataYatesV1 import get_session
from DataYatesV1 import enable_autoreload

import matplotlib.pyplot as plt

enable_autoreload()

#%% Discover Available Models
print("🔍 Discovering available models...")
from eval_stack_utils import scan_checkpoints
models_by_type = scan_checkpoints('/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints')

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

#%%
from eval_stack_multidataset import load_model

model, model_info = load_model(
        model_type='learned_res_modulator_small',
        model_index=None,
        checkpoint_path=None,
        checkpoint_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints",
        device='cpu'
    )

#%% Load Multiple Models for Comparison
print("\n📊 Loading models for comparison...")

# Define models to compare
models_to_compare = ['learned_res_modulator_small', 'learned_res']
available_models = [m for m in models_to_compare if m in models_by_type]

print(f"Comparing models: {available_models}")

# Load results for each model
all_results = {}
for model_type in available_models:
    print(f"\nLoading {model_type}...")
    
    results = evaluate_model_multidataset(
        model_type=model_type,
        analyses=['bps', 'ccnorm', 'saccade'],  # Start with BPS and CCNORM for speed
        recalc=False,
        batch_size=64
    )
    all_results.update(results)
    model_name = list(results.keys())[0]
    n_cells = len(results[model_name]['qc']['all_cids'])
    print(f"  ✅ {model_name}: {n_cells} cells")

print(f"\n✅ Loaded {len(all_results)} models for comparison")


#%%
import matplotlib.pyplot as plt
from extract_functions import extract_bps_saccade, extract_ccnorm, extract_qc_spatial

stim_type = 'backimage'
bps_comparison = {}
for model_id in [0, 1]:
    model_name = list(all_results.keys())[model_id]
    # Get BPS and saccade data for gaborium
    bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
    bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}


print(f"Len of bps: {len(bps)}, Len of cids: {len(cids)}")
cid = 0

#%% 

feat_ex = torch.concatenate([r.features_ex_weight.detach().cpu().squeeze() for r in model.model.readouts])
feat_inh = torch.concatenate([r.features_inh_weight.detach().cpu().squeeze() for r in model.model.readouts])
feat = torch.concatenate([feat_ex, feat_inh], 1)
feat /= torch.linalg.norm(feat, dim=1)[:, None]
D = torch.cov(feat)
u, s, v = torch.linalg.svd(D)
s = s.detach().cpu().numpy()
plt.plot(np.cumsum(s)/np.sum(s))
plt.show()


#%%

cid += 2
plt.plot(feat[cid])

#%%
# D = torch.mm(feat, feat.T)
plt.imshow(D.detach().cpu(), cmap='viridis', interpolation='none')
plt.show()

#%% Plot STAs for every cell
from DataYatesV1 import prepare_data
eval_dicts = all_results['learned_res_ddp_bs256_ds30_lr1e-3_wd1e-4_corelrscale0.5_warmup5']['bps']
from eval_stack_multidataset import load_single_dataset, get_stim_inds
from tqdm import tqdm

def get_sta_ste(model, eval_dicts, didx=0, lags=list(range(16))):

    robs = eval_dicts['gaborium']['robs'][didx]
    rhat = eval_dicts['gaborium']['rhat'][didx]

    # overwrite stimulus transforms
    dataset_config = model.model.dataset_configs[didx].copy()
    dataset_config['transforms']['stim'] = {'source': 'stim',
        'ops': [{'pixelnorm': {}}],
        'expose_as': 'stim'}

    dataset_config['keys_lags']['stim'] = list(range(25))
    dataset_config['types'] = ['gaborium']
    
    train_data, val_data, dataset_config = prepare_data(dataset_config)
    stim_indices = get_stim_inds( 'gaborium', train_data, val_data)

    # shallow copy the dataset not to mess it up
    data = val_data.shallow_copy()
    data.inds = stim_indices

    dset_idx = np.unique(stim_indices[:,0]).item()

    # confirm inds match
    assert torch.all(robs == data.dsets[dset_idx]['robs'][data.inds[:,1]]), 'robs mismatch'
    dfs = data.dsets[dset_idx]['dfs'][data.inds[:,1]]
    norm_dfs = dfs.sum(0) # if forward
    norm_robs = (robs * dfs).sum(0) # if reverse
    norm_rhat = (rhat * dfs).sum(0) # if reverse

    n_cells = robs.shape[1]
    n_lags = len(lags)
    H, W  = data.dsets[dset_idx]['stim'].shape[1:3]
    sta_robs = torch.zeros((n_lags, H, W, n_cells))
    ste_robs = torch.zeros((n_lags, H, W, n_cells))
    sta_rhat = torch.zeros((n_lags, H, W, n_cells))
    ste_rhat = torch.zeros((n_lags, H, W, n_cells))
    for lag in tqdm(lags):
        stim = data.dsets[dset_idx]['stim'][data.inds[:,1]-lag]    
        sta_robs[lag] = torch.einsum('thw, tc->hwc', stim, robs*dfs)
        ste_robs[lag] = torch.einsum('thw, tc->hwc', stim.pow(2), robs*dfs)
        sta_rhat[lag] = torch.einsum('thw, tc->hwc', stim, rhat*dfs)
        ste_rhat[lag] = torch.einsum('thw, tc->hwc', stim.pow(2), rhat*dfs)
    
    return {'sta_robs': sta_robs, 'ste_robs': ste_robs, 'sta_rhat': sta_rhat, 'ste_rhat': ste_rhat, 'norm_dfs': norm_dfs, 'norm_robs': norm_robs, 'norm_rhat': norm_rhat}

for didx in range(10, len(model.names)):
    sta = get_sta_ste(model, eval_dicts, didx=didx, lags=list(range(16)))
    n_cells = sta['sta_robs'].shape[-1]
    sx = np.floor(np.sqrt(n_cells)).astype(int)
    sy = np.ceil(n_cells / sx).astype(int)
    fig, axs = plt.subplots(sy, sx, figsize=(16, 16))
    lag = 8
    sta_robs = sta['sta_robs'] / sta['norm_dfs'][None,None,None,:]
    sta_rhat = sta['sta_rhat'] / sta['norm_dfs'][None,None,None,:]
    H = sta_robs.shape[1]

    for i in range(n_cells):
        ax = axs.flatten()[i]
        v = sta_robs[lag,:,:,i].abs().max()
        I = torch.concat([sta_robs[lag,:,:,i], torch.ones(H,1), sta_rhat[lag,:,:,i]], 1)
        
        ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)
        ax.set_title(f'Cell {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'sta_{model.names[didx]}.png')
    plt.show()

#%%
    
    

#%%
from DataYatesV1.utils.rf import calc_sta
calc_sta(stim, robs, lags, dfs=None, inds=None, stim_modifier=lambda x: x, reverse_correlate=True, batch_size=None, device=None, progress=False):
#%%
def get_fixrsvp_trials(model, eval_dicts, didx):

    robs = eval_dicts[didx]['fixrsvp'][0]
    rhat = eval_dicts[didx]['fixrsvp'][1]
    train_data, val_data, dataset_config = load_single_dataset(model, didx)
    stim_indices = get_stim_inds('fixrsvp', train_data, val_data)
    data = val_data.shallow_copy()
    data.inds = stim_indices


    dset_idx = np.unique(stim_indices[:,0]).item()
    time_inds = data.dsets[dset_idx]['psth_inds'].numpy()
    trial_inds = data.dsets[dset_idx]['trial_inds'].numpy()
    unique_trials = np.unique(trial_inds)

    n_trials = len(unique_trials)
    n_time = np.max(time_inds).item()+1
    n_units = data.dsets[dset_idx]['robs'].shape[1]
    robs_trial = np.nan*np.zeros((n_trials, n_time, n_units))
    rhat_trial = np.nan*np.zeros((n_trials, n_time, n_units))
    dfs_trial = np.nan*np.zeros((n_trials, n_time, n_units))

    for itrial in range(n_trials):
        trial_idx = np.where(trial_inds == unique_trials[itrial])[0]
        eval_inds = np.where(np.isin(stim_indices[:,1], trial_idx))[0]
        data_inds = trial_idx[np.where(np.isin(trial_idx, stim_indices[:,1]))[0]]

        # print(f'Trial {itrial} has {len(eval_inds)} eval inds and {len(data_inds)} data inds')
        assert torch.all(robs[eval_inds] == data.dsets[dset_idx]['robs'][data_inds]).item(), 'robs mismatch'

        robs_trial[itrial, time_inds[data_inds]] = robs[eval_inds]
        rhat_trial[itrial, time_inds[data_inds]] = rhat[eval_inds]
        dfs_trial[itrial, time_inds[data_inds]] = data.dsets[dset_idx]['dfs'][data_inds]

    return robs_trial, rhat_trial, dfs_trial
#%%
cid += 1
plt.plot(saccade_time_bins, bps_comparison[0]['robs'][:, cid], 'k')
plt.plot(saccade_time_bins, bps_comparison[0]['rhat'][:, cid], label=bps_comparison[0]['name'])
plt.plot(saccade_time_bins, bps_comparison[1]['rhat'][:, cid], label=bps_comparison[1]['name'])
plt.legend()
plt.show()

#%%

bps_ni_1 = bps_comparison[0]['bps']
bps_ni_2 = bps_comparison[1]['bps']


depth, waveforms, cids, dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results)
print(f"Len of depth: {len(depth)}, Len of cids: {len(cids)}, Len of contamination: {len(contamination)}")

#%% CCNORM
ccnorm = np.concatenate(all_results['learned_res_ddp_bs256_ds30_lr1e-3_wd1e-4_corelrscale0.5_warmup5']['ccnorm']['fixrsvp']['ccnorm'])
plt.hist(ccnorm, density=True, alpha=.5)
plt.hist(ccnorm[contamination < 20], density=True, alpha=.5)
plt.show()

#%% PLOT BY DEPTH!!
iix = contamination < 20
plt.plot(bps_ni_1, bps_ni_2, 'k.')
plt.plot(bps_ni_1[iix], bps_ni_2[iix], 'r.')
plt.plot([-1, 3], [-1, 3], 'k--')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.show()

# plt.plot(bps_ni_2-bps_ni_1, -depth, 'k.', alpha=.1)
plt.plot(bps_ni_2[iix]-bps_ni_1[iix], -depth[iix], 'r.', alpha=.5)
plt.axvline(0, color='k', linestyle='--')
plt.axhline(0, color='k', linestyle='--')
plt.xlim(-.25, .25)
plt.xlabel('LLR (modulator - none)')
plt.ylabel('Depth (um)')
plt.show()


#%% Do it for all stimulus types

depth, waveforms, cids, dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results)
iix = (contamination < 20) & (bps_ni_1 > .1)

print(f"Len of depth: {len(depth)}, Len of cids: {len(cids)}, Len of contamination: {len(contamination)}")
plt.figure(figsize=(15, 4))
stim_types = ['backimage', 'gaborium', 'fixrsvp', 'gratings']
for stim_type in stim_types:
    bps_comparison = {}
    for model_id in [0, 1]:
        model_name = list(all_results.keys())[model_id]
        # Get BPS and saccade data for gaborium
        bps, saccade_robs, saccade_rhat, tbins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
        bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}

    plt.subplot(1, len(stim_types), stim_types.index(stim_type)+1)
    bps1 = bps_comparison[0]['bps']
    bps2 = bps_comparison[1]['bps']
    plt.plot(bps2[iix]-bps1[iix], -depth[iix], 'r.', alpha=.5)
    plt.axvline(0, color='k', linestyle='--')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlim(-.25, .25)
    plt.xlabel('LLR (modulator - none)')
    plt.ylabel('Depth (um)')
    plt.title(stim_type)
plt.show()



#%% plot saccade-triggered averages as function of depth

stim_type = 'backimage'
bps_comparison = {}
for model_id in [0, 1]:
    model_name = list(all_results.keys())[model_id]
    # Get BPS and saccade data for gaborium
    bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
    bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}

def get_Rnorm(R, tbins):
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d

    R = gaussian_filter1d(R, 1, axis=0)
    # R = savgol_filter(R, 21, 3, axis=0)
    R0 = R[tbins < 0].mean(0)
    R = (R - R0)/R.max(0)

    return R

good_ix = np.where(~np.isnan(np.sum(bps_comparison[0]['robs'],0)))[0]
Rdata = get_Rnorm(bps_comparison[0]['robs'][:,good_ix], saccade_time_bins)
Rmodel = []
for model_id in [0, 1]:
    Rmodel.append(get_Rnorm(bps_comparison[model_id]['rhat'][:,good_ix], saccade_time_bins))

ind = np.argmax(Rmodel[0], 0)
ind = np.argsort(ind)

plt.figure(figsize=(10, 5))
plt.subplot(1,3,1)
plt.imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)])
plt.title(f'Data {stim_type}')
plt.xlim(0, 50)
for i in range(2):
    plt.subplot(1,3,i+2)
    plt.imshow(Rmodel[i][:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)])
    plt.title(bps_comparison[i]['name'])
    plt.xlim(0, 50)
plt.show()

#%%
dt = 1/240
def get_trough_peak(R, saccade_time_bins, temperature=1.0):
    """
    Find trough and peak positions using softmax for sub-integer precision.

    Args:
        R: Response array [time, neurons]
        saccade_time_bins: Time bins for saccade analysis
        temperature: Softmax temperature (lower = sharper, higher = smoother)

    Returns:
        lag_min, val_min, lag_max, val_max: Sub-integer lag positions and values
    """
    post_saccade = np.where((dt*saccade_time_bins > 0) & (dt*saccade_time_bins < .15))[0]
    R_post = R[post_saccade]  # [time_post, neurons]

    # For maximum: use softmax on the values to get sub-integer positions
    max_probs = np.exp(R_post / temperature) / np.sum(np.exp(R_post / temperature), axis=0, keepdims=True)
    # Convert back to original post_saccade indices (fractional)
    lag_max_fractional = np.sum(max_probs * np.arange(len(post_saccade))[:, None], axis=0)
    lag_max = post_saccade[0] + lag_max_fractional  # Map back to original time indices
    val_max = np.sum(max_probs * R_post, axis=0)  # Expected value

    # For minimum: use softmax on negative values
    min_probs = np.exp(-R_post / temperature) / np.sum(np.exp(-R_post / temperature), axis=0, keepdims=True)
    lag_min_fractional = np.sum(min_probs * np.arange(len(post_saccade))[:, None], axis=0)
    lag_min = post_saccade[0] + lag_min_fractional  # Map back to original time indices
    val_min = np.sum(min_probs * R_post, axis=0)  # Expected value

    return lag_min, val_min, lag_max, val_max

def get_trough_peak_original(R, saccade_time_bins):
    """Original hard argmax/argmin implementation for comparison."""
    post_saccade = np.where((dt*saccade_time_bins > 0) & (dt*saccade_time_bins < .15))[0]
    lag_max = np.argmax(R[post_saccade], axis=0)
    lag_min = np.argmin(R[post_saccade], axis=0)
    val_max = R[post_saccade[lag_max], np.arange(len(lag_max))]
    val_min = R[post_saccade[lag_min], np.arange(len(lag_min))]

    return lag_min, val_min, lag_max, val_max


lag_min_data, val_min_data, lag_max_data, val_max_data = get_trough_peak_original(Rdata, saccade_time_bins)

# Debug comparison with original method
lag_min_orig, val_min_orig, lag_max_orig, val_max_orig = get_trough_peak_original(Rdata, saccade_time_bins)
print(f"Softmax vs Original comparison (first 5 neurons):")
print(f"Max lag diff: {(lag_max_data - lag_max_orig)[:5]}")
print(f"Min lag diff: {(lag_min_data - lag_min_orig)[:5]}")
print(f"Max val diff: {(val_max_data - val_max_orig)[:5]}")
print(f"Min val diff: {(val_min_data - val_min_orig)[:5]}")


fig, axs = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

axs[0].plot(1e3*dt*lag_max_data, val_max_data, 'b.', alpha=.1, label='Max')
axs[0].plot(1e3*dt*lag_min_data, np.abs(val_min_data), 'r.', alpha = .1, label='Min')
axs[0].set_title(f"Data {stim_type}")

for i in range(2):
    lag_min_model, val_min_model, lag_max_model, val_max_model = get_trough_peak_original(Rmodel[i], saccade_time_bins)
    axs[i+1].plot(1e3*dt*lag_max_model, val_max_model, 'b.', alpha=.1, label='Max')
    axs[i+1].plot(1e3*dt*lag_min_model, np.abs(val_min_model), 'r.', alpha = .1, label='Min')
    axs[i+1].set_title(bps_comparison[i]['name'])
    axs[i+1].set_xlabel('Latency (ms)')
    axs[i+1].set_ylabel('Modulation')

plt.legend()

#%%

lag_min_model, val_min_model, lag_max_model, val_max_model = get_trough_peak_original(Rmodel[0], saccade_time_bins)
jitter1 = 0*np.random.normal(0, 1, len(lag_max_data))
jitter2 = 0*np.random.normal(0, 1, len(lag_max_model))
plt.plot(1e3*dt*lag_max_data+jitter1, 1e3*dt*lag_max_model+jitter2, 'k.', alpha=.1)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.title('Peak Latency (ms)')
plt.xlabel('Data')
plt.ylabel('Model')
plt.show()

#%%
iix = (contamination < 20) & (bps_ni_1 > .1) & (bps_ni_2 - bps_ni_1 > .01) & (np.var(R[tbins<0], 0) < .7*np.var(R[tbins>0], 0))

plt.figure(figsize=(5, 10))
_ = plt.plot(R[:,iix] - depth[iix]/50,  alpha=.5)

#%%
datasets = np.unique(dids)
inclusion = (contamination < 20) & (bps_ni_1 > .1) & (bps_ni_2 - bps_ni_1 > .01) & (np.var(R[tbins<0], 0) < .7*np.var(R[tbins>0], 0))


for dataset in datasets:
    iix = np.isin(dids, dataset) & inclusion
    if np.sum(iix) == 0:
        continue
    plt.figure(figsize=(5, 10))
    _ = plt.plot(R[:,iix] - depth[iix]/50,  alpha=.5)
    plt.title(dataset)
    plt.show()


#%% plot bps by monkey

#%% Helper function to debug one dataset
dataset_idx = 0
"""
Debug a single dataset to track sizes of everything.
"""
from eval_stack_utils import load_single_dataset, get_stim_inds, evaluate_dataset

print(f"=== DEBUGGING DATASET {dataset_idx} ===")

# 1. Check model readout size
model_units = model.model.readouts[dataset_idx].n_units
print(f"Model readout[{dataset_idx}].n_units: {model_units}")

# 2. Load dataset and check config
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
dataset_cids = dataset_config.get('cids', [])
print(f"Dataset config CIDs length: {len(dataset_cids)}")
print(f"Match model units? {len(dataset_cids) == model_units}")

#%%
model = model.to('cuda:0')
# 3. Check stimulus indices for gaborium
gab_indices = get_stim_inds('gaborium', train_data, val_data)
print(f"Gaborium stimulus indices shape: {gab_indices.shape}")

# 4. Run evaluation and check BPS size
gab_robs, gab_rhat, gab_bps = evaluate_dataset(
    model, train_data, gab_indices, dataset_idx, 64, "Debug Gaborium"
)

print(f"Gaborium results:")
print(f"  robs shape: {gab_robs.shape}")
print(f"  rhat shape: {gab_rhat.shape}")
print(f"  bps shape: {gab_bps.shape}")
print(f"  bps length matches model units? {len(gab_bps) == model_units}")
print(f"  bps length matches config CIDs? {len(gab_bps) == len(dataset_cids)}")


#%% Test with one dataset
# Load model

from eval_stack_multidataset import load_model
model, model_info = load_model(model_type='learned_res', device='cpu')

# Debug first dataset
debug_info = debug_single_dataset(model, 0)
print(f"\nSummary: {debug_info}")

#%%
#%%

stim_type = 'gaborium'
model_id = 0
model_name = list(results.keys())[model_id]
bps, saccade_robs, saccade_rhat, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)

#%%

# Get CCNORM data
ccnorm, rbar, rhatbar, ccnorm_cids, ccnorm_dids = extract_ccnorm(model_name, all_results)

# Get QC and spatial data  
depth, waveforms, qc_cids, qc_dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results)

# For one cell (index 0):
print(f"Cell 0: BPS={bps[0]:.4f}, Depth={depth[0]:.1f}μm, Contamination={contamination[0]:.1f}%")


#%% Examine Cell Tracking and QC Data
print("\n🔬 Examining cell tracking and QC data...")

# Get first model for detailed examination
first_model = list(all_results.keys())[0]
bps1 = np.concatenate(all_results[first_model]['bps']['gaborium']['bps'])
second_model = list(all_results.keys())[1]
bps2 = np.concatenate(all_results[second_model]['bps']['gaborium']['bps'])

plt.scatter(bps1, bps2, s=10, alpha=1, color='k', edgecolors='w', linewidth=0.5)
plt.plot([-1, 3], [-1, 3], 'k--')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.xlabel(f'{first_model} BPS')
plt.ylabel(f'{second_model} BPS')
plt.show()

#%% DEBUGGING
# Check what's in the BPS results structure
model_name = list(results.keys())[0]
bps_data = results[model_name]['bps']['gaborium']

print("BPS arrays info:")
for i, bps_array in enumerate(bps_data['bps']):
    print(f"  Dataset {i}: BPS shape {bps_array.shape} Model N Units {model.model.readouts[i].n_units}")

print(f"\nTotal BPS length when concatenated: {sum(len(arr) for arr in bps_data['bps'])}")
print(f"CIDs length: {len(bps_data['cids'])}")
print(f"Datasets length: {len(bps_data['datasets'])}")
# %%
