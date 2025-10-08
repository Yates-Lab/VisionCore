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
checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'
models_by_type = scan_checkpoints(checkpoint_dir)

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
        model_type='learned_res_small_gru',
        model_index=None,
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.eval()

#%% Load Multiple Models for Comparison
print("\n📊 Loading models for comparison...")

# Define models to compare
models_to_compare = [ 'learned_res_small_gru', 'learned_res_small_none_gru', 'learned_res_small_none_gru_none_pool']#, 'learned_res_small', 'learned_res_small_pc', 'learned_res_small_stn', 'learned_res_small_film']
available_models = [m for m in models_to_compare if m in models_by_type]

print(f"Comparing models: {available_models}")

# Load results for each model
all_results = {}
for model_type in available_models:
    print(f"\nLoading {model_type}...")
    
    results = evaluate_model_multidataset(
        model_type=model_type,
        analyses=['bps', 'ccnorm', 'saccade', 'sta'],  # Include all analyses including STA
        checkpoint_dir=checkpoint_dir,
        save_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack_smooth_120",
        recalc=False,
        batch_size=64
    )
    all_results.update(results)
    model_name = list(results.keys())[0]
    n_cells = len(results[model_name]['qc']['all_cids'])
    print(f"  ✅ {model_name}: {n_cells} cells")

    # Save incrementally after each model to prevent data loss
    import pickle
    from pathlib import Path
    save_path = Path('all_results_120_analysis_incremental.pkl')
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"  💾 Incremental save: {len(all_results)} models ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  ⚠️ Incremental save failed: {e}")

print(f"\n✅ Loaded {len(all_results)} models for comparison")


#%% Save all_results for plotting

import pickle
import numpy as np
from pathlib import Path

# Create a timestamped filename for safety
save_path = Path('all_results_120_analysis.pkl')
print(f"Saving results to: {save_path.absolute()}")

# Save with error handling
try:
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"✅ Successfully saved all_results ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
except Exception as e:
    print(f"❌ Failed to save: {e}")

# Verify the saved file loads correctly

print("🔍 Verifying saved file...")
try:
    with open(save_path, 'rb') as f:
        all_results_loaded = pickle.load(f)
    print("✅ File loaded successfully")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    all_results_loaded = None

# Compare structure and key data to verify integrity

if all_results_loaded is not None:
    print("\n📊 Comparing original vs loaded data:")

    # Check top-level structure
    orig_models = set(all_results.keys())
    loaded_models = set(all_results_loaded.keys())
    print(f"Models - Original: {len(orig_models)}, Loaded: {len(loaded_models)}")
    print(f"Models match: {orig_models == loaded_models}")

    # Check a few key metrics for each model
    for model_name in orig_models:
        if model_name in all_results_loaded:
            orig_model = all_results[model_name]
            loaded_model = all_results_loaded[model_name]

            print(f"\n🔍 Model: {model_name}")

            # Check BPS data
            if 'bps' in orig_model and 'bps' in loaded_model:
                for stim_type in ['gaborium', 'backimage', 'fixrsvp']:
                    if stim_type in orig_model['bps'] and stim_type in loaded_model['bps']:
                        orig_bps = orig_model['bps'][stim_type]['bps']
                        loaded_bps = loaded_model['bps'][stim_type]['bps']

                        if len(orig_bps) > 0 and len(loaded_bps) > 0:
                            # Compare first array
                            orig_arr = np.array(orig_bps[0]) if isinstance(orig_bps[0], list) else orig_bps[0]
                            loaded_arr = np.array(loaded_bps[0]) if isinstance(loaded_bps[0], list) else loaded_bps[0]

                            arrays_equal = np.allclose(orig_arr, loaded_arr, equal_nan=True)
                            print(f"  {stim_type} BPS arrays equal: {arrays_equal}")
                            if not arrays_equal:
                                print(f"    Original shape: {orig_arr.shape}, mean: {np.nanmean(orig_arr):.4f}")
                                print(f"    Loaded shape: {loaded_arr.shape}, mean: {np.nanmean(loaded_arr):.4f}")

            # Check QC data
            if 'qc' in orig_model and 'qc' in loaded_model:
                orig_cids = orig_model['qc']['all_cids']
                loaded_cids = loaded_model['qc']['all_cids']
                cids_equal = len(orig_cids) == len(loaded_cids) and all(o == l for o, l in zip(orig_cids, loaded_cids))
                print(f"  QC CIDs equal: {cids_equal} (count: {len(orig_cids)})")

    print(f"\n✅ Verification complete. Use 'all_results_loaded' for analysis or rename to 'all_results'")
    print(f"💾 Saved file: {save_path.absolute()}")
else:
    print("❌ Cannot verify - file failed to load")

#%%
sys.exit()

#%% Plot STAs for every cell
from DataYatesV1 import prepare_data
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


#%%
model_names = list(all_results.keys())
model_name = model_names[0]
eval_dicts = all_results[model_name]['bps']

sta_datasets = []
for didx in range(len(model.names)):
    sta = get_sta_ste(model, eval_dicts, didx=didx, lags=list(range(16)))
    n_cells = sta['sta_robs'].shape[-1]
    sx = np.floor(np.sqrt(n_cells)).astype(int)
    sy = np.ceil(n_cells / sx).astype(int)
    fig, axs = plt.subplots(sy, sx, figsize=(16, 16))
    lag = 4
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
    sta_datasets.append(sta)

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

iix = contamination < 20
plt.plot(bps_ni_1, bps_ni_2, 'k.')
plt.plot(bps_ni_1[iix], bps_ni_2[iix], 'r.')
plt.plot([-1, 3], [-1, 3], 'k--')
plt.xlabel(bps_comparison[0]['name'])
plt.ylabel(bps_comparison[1]['name'])
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.show()

#%% CCNORM
ccnorms = []
model_names = []
plt.figure(figsize=(6,3))
for imodel in range(2):
    model_name = list(all_results.keys())[imodel]
    ccnorm = np.concatenate(all_results[model_name]['ccnorm']['fixrsvp']['ccnorm'])
    ccnorms.append(ccnorm)
    plt.subplot(1,2,imodel+1)
    plt.hist(ccnorm, density=True, alpha=.5)
    plt.hist(ccnorm[contamination < 20], density=True, alpha=.5)
    plt.axvline(np.nanmedian(ccnorm), color='b', linestyle='--')
    plt.axvline(.5, color='k', linestyle='--')
    model_names.append(model_name.split('_ddp')[0])
    plt.title(model_name.split('_ddp')[0])

plt.figure(figsize=(3,3))
plt.plot(ccnorms[0], ccnorms[1], 'k.')
plt.plot(ccnorms[0][contamination < 20], ccnorms[1][contamination < 20], 'r.')
plt.xlabel(model_names[0])
plt.ylabel(model_names[1])
plt.title('CCNORM')
plt.plot(plt.xlim(), plt.xlim(), 'k--')

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

stim_type = 'gaborium'
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

dt = 1000/120
tbins = saccade_time_bins*dt
plt.figure(figsize=(10, 5))
plt.subplot(1,3,1)
plt.imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)])
plt.title(f'Data {stim_type}')
plt.xlim(-50, 250)
for i in range(2):
    
    plt.subplot(1,3,i+2)
    plt.imshow(Rmodel[i][:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)])
    plt.title(bps_comparison[i]['name'])
    if i==0:
        plt.xlabel('Time from saccade onset (ms)')
    plt.xlim(-50, 250)
plt.show()

#%% sort by cortical depth

depths = depth[good_ix]
ind = np.argsort(depths)
plt.figure(figsize=(10, 5))
plt.subplot(1,3,1)
plt.imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)])
plt.title(f'Data {stim_type}')
plt.xlim(-50, 250)
plt.ylabel('Neuron (sorted by depth)')
for i in range(2):
    plt.subplot(1,3,i+2)
    plt.imshow(Rmodel[i][:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)])
    plt.title(bps_comparison[i]['name'])
    if i==0:
        plt.xlabel('Time from saccade onset (ms)')
    plt.xlim(-50, 250)
plt.show()

#%%

dt = 1/220
def get_trough_peak(R, saccade_time_bins, dt, temperature=0.02):
    """
    Fractional trough/peak latency and amplitude via soft-argmax/min.
    R is already normalised to [-1, 1].
    """
    # post-saccadic window:   0 < t < 150 ms
    post_idx = np.where((dt * saccade_time_bins > 0) &
                        (dt * saccade_time_bins < 0.15))[0]
    R_post   = R[post_idx]                        # [T, N]
    t_bins   = saccade_time_bins[post_idx][:, None].astype(float)
    t_inds   = post_idx[:, None].astype(float)

    # ---- soft-argmax ----
    logits_max = (R_post - R_post.max(0, keepdims=True)) / temperature
    p_max      = np.exp(logits_max)
    p_max     /= p_max.sum(0, keepdims=True)

    lag_max_t = (p_max * t_bins).sum(0)             # fractional bin
    lag_max_i = (p_max * t_inds).sum(0)             # fractional index

    # linear interpolation for amplitude
    lo, hi  = np.floor(lag_max_i).astype(int), np.ceil(lag_max_i).astype(int)
    w_hi    = lag_max_i - lo
    val_max = (1-w_hi) * R[lo, np.arange(R.shape[1])] + \
               w_hi  * R[hi, np.arange(R.shape[1])]

    # ---- soft-argmin ----
    logits_min = (-R_post - (-R_post).max(0, keepdims=True)) / temperature
    p_min      = np.exp(logits_min)
    p_min     /= p_min.sum(0, keepdims=True)

    lag_min_t = (p_min * t_bins).sum(0)
    lag_min_i = (p_min * t_inds).sum(0)

    lo, hi  = np.floor(lag_min_i).astype(int), np.ceil(lag_min_i).astype(int)
    w_hi    = lag_min_i - lo
    val_min = (1-w_hi) * R[lo, np.arange(R.shape[1])] + \
               w_hi  * R[hi, np.arange(R.shape[1])]

    return lag_min_t, val_min, lag_max_t, val_max

def get_trough_peak_original(R, saccade_time_bins):
    """Original hard argmax/argmin implementation for comparison."""
    post_saccade = np.where((dt*saccade_time_bins > 0) & (dt*saccade_time_bins < .15))[0]
    lag_max = np.argmax(R[post_saccade], axis=0)
    lag_min = np.argmin(R[post_saccade], axis=0)
    val_max = R[post_saccade[lag_max], np.arange(len(lag_max))]
    val_min = R[post_saccade[lag_min], np.arange(len(lag_min))]

    return lag_min, val_min, lag_max, val_max


lag_min_data, val_min_data, lag_max_data, val_max_data = get_trough_peak(Rdata, saccade_time_bins, dt)

# Debug comparison with original method
lag_min_orig, val_min_orig, lag_max_orig, val_max_orig = get_trough_peak(Rdata, saccade_time_bins, dt)
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
    lag_min_model, val_min_model, lag_max_model, val_max_model = get_trough_peak(Rmodel[i], saccade_time_bins, dt)
    axs[i+1].plot(1e3*dt*lag_max_model, val_max_model, 'b.', alpha=.1, label='Max')
    axs[i+1].plot(1e3*dt*lag_min_model, np.abs(val_min_model), 'r.', alpha = .1, label='Min')
    axs[i+1].set_title(bps_comparison[i]['name'])
    axs[i+1].set_xlabel('Latency (ms)')
    axs[i+1].set_ylabel('Modulation')

plt.legend()
plt.show()

lag_min_model, val_min_model, lag_max_model, val_max_model = get_trough_peak(Rmodel[0], saccade_time_bins, dt)
jitter1 = 0*np.random.normal(0, 1, len(lag_max_data))
jitter2 = 0*np.random.normal(0, 1, len(lag_max_model))
plt.figure(figsize=(5, 5))
plt.plot(1e3*dt*lag_max_data+jitter1, 1e3*dt*lag_max_model+jitter2, 'k.', alpha=.1)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.title('Peak Latency (ms)')
plt.xlabel('Data (ms)')
plt.ylabel('Model (ms)')

plt.show()

#%%
iix = (contamination < 20) & (bps_ni_1 > .1) & (bps_ni_2 - bps_ni_1 > .01)

plt.figure(figsize=(5, 10))
_ = plt.plot(Rdata[:,iix] - depth[iix]/50,  alpha=.5)
#%% 

try:
    feat_ex = torch.concatenate([r.features_ex_weight.detach().cpu().squeeze() for r in model.model.readouts])
    feat_inh = torch.concatenate([r.features_inh_weight.detach().cpu().squeeze() for r in model.model.readouts])
    feat = torch.concatenate([feat_ex, feat_inh], 1)
except Exception as e:
    feat = torch.concatenate([r.features.weight.detach().cpu().squeeze() for r in model.model.readouts])

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




# %%
