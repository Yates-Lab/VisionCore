#%%

import sys

sys.path.append('./scripts')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

import matplotlib as mpl

enable_autoreload()
device = get_free_device()

from mcfarland_sim import run_mcfarland_on_dataset, extract_metrics, DualWindowAnalysis
from utils import get_model_and_dataset_configs

#%%

from models.config_loader import load_dataset_configs

dataset_configs_path = "/home/declan/VisionCore/experiments/dataset_configs/single_basic_120_long_rowley.yaml"
    
dataset_configs = load_dataset_configs(dataset_configs_path)

print(dataset_configs)


# %%

from models.data import prepare_data

train_data, val_data, dataset_config = prepare_data(dataset_configs[0], strict=False)

#%%
#%%#%%
dataset = train_data.shallow_copy()
dataset.inds = inds

dset = dataset.dsets[ dataset.inds[0,0].item() ]  # fixrsvp sub-dataset
robs = dset['robs'].numpy()
eyepos = dset['eyepos'].numpy()
dfs = dset['dfs'].numpy()
cids = np.array(dset.metadata['cids'])

#%% === Build trials, good_trials, neuron_mask, valid_mask, dt =================

trial_inds = dset['trial_inds'].numpy()
trials = np.unique(trial_inds)
NC = robs.shape[1]
NT = len(trials)

# trial-aligned arrays
trial_lengths = [np.sum(trial_inds == tr) for tr in trials]
max_T = int(max(trial_lengths))

robs_trial = np.full((NT, max_T, NC), np.nan, dtype=np.float32)
eyepos_trial = np.full((NT, max_T, 2), np.nan, dtype=np.float32)
dfs_trial = np.full((NT, max_T, NC), np.nan, dtype=np.float32)
fix_dur = np.zeros(NT, dtype=int)

# simple fixation definition (center < 1 deg)
fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < 1

for i, tr in enumerate(trials):
    ix = (trial_inds == tr) & fixation
    T_trial = ix.sum()
    if T_trial == 0:
        continue
    robs_trial[i, :T_trial] = robs[ix]
    eyepos_trial[i, :T_trial] = eyepos[ix]
    dfs_trial[i, :T_trial] = dfs[ix]
    fix_dur[i] = T_trial

# McFarland-style trial and neuron selection
windows_ms = [5, 10, 20, 40, 80]
total_spikes_threshold = 200

good_trials = fix_dur > 20  # require at least 20 fixation+dfs bins
robs_mc = robs_trial[good_trials]
eyepos_mc = eyepos_trial[good_trials]
dfs_mc = dfs_trial[good_trials]
fix_dur_mc = fix_dur[good_trials]  # Use a new variable name for masked fix_dur

print(f"McFarland prep: {robs_mc.shape[0]} trials, {robs_mc.shape[2]} neurons, {robs_mc.shape[1]} time bins")

# sampling after prepare_data: 120 Hz → dt = 1/120
dt = 1 / 120.0
valid_time_bins = min(240, robs_mc.shape[1])

# neuron mask: intersect spike-count QC with high-ccmax cids (if available)

# try to infer session name from metadata to locate high-ccmax file
sess_name = None
meta = getattr(dset, 'metadata', None)
sess_obj = None
if hasattr(meta, 'get'):
    try:
        sess_obj = meta.get('sess', None)
    except Exception:
        sess_obj = None
if sess_obj is not None and hasattr(sess_obj, 'name'):
    sess_name = sess_obj.name
if sess_name is None:
    # fallback for rowley Luke session
    sess_name = 'Luke_2025-08-04'

figures_dir = Path('../figures')
high_cc_path = figures_dir / f"{sess_name}_high_ccmax_cids.npy"

if high_cc_path.exists():
    high_cc_cids = np.load(high_cc_path)
    high_cc_mask = np.isin(cids, high_cc_cids)
    print(f"Loaded high-ccmax cids from {high_cc_path} (N={high_cc_cids.size})")
else:
    high_cc_mask = np.ones_like(cids, dtype=bool)
    print(f"High-ccmax cids file not found at {high_cc_path}; using spike-count mask only.")

spike_ok = np.nansum(robs_mc, axis=(0, 1)) > total_spikes_threshold
combined_mask_bool = spike_ok & high_cc_mask
neuron_mask = np.where(combined_mask_bool)[0]
print(f"Using {len(neuron_mask)} neurons / {robs_mc.shape[2]} total (spikes>{total_spikes_threshold} & high-ccmax)")

# valid mask: dfs + finite spikes and eye positions
# dfs is (T, NC) in [0,1] or {0,1}; treat >0.5 as "valid"
dfs_valid = np.nanmean(dfs_mc[:, :, neuron_mask], axis=2) > 0.5

valid_mask = (
    dfs_valid &
    np.isfinite(np.sum(robs_mc[:, :, neuron_mask], axis=2)) &
    np.isfinite(np.sum(eyepos_mc, axis=2))
)

eyepos_used = eyepos_mc[:, iix]
analyzer_luke = DualWindowAnalysis(robs_used, eyepos_used, valid_used, dt=dt)
eyepos_used = eyepos_mc[:, iix]

# time index window
iix = np.arange(valid_time_bins)
robs_used = robs_mc[:, iix][:, :, neuron_mask]
eyepos_used = eyepos_mc[:, iix]
valid_used = valid_mask[:, iix]

print(f"robs_used shape: {robs_used.shape}")
print(f"eyepos_used shape: {eyepos_used.shape}")
print(f"valid_used shape: {valid_used.shape}")

analyzer_luke = DualWindowAnalysis(robs_used, eyepos_used, valid_used, dt=dt)
print("DualWindowAnalysis initialized for Luke_2025-08-04 fixrsvp")

# %% Run McFarland sweep on Luke fixrsvp
windows_ms = [5, 10, 20, 40, 80]
results_luke, last_mats_luke = analyzer_luke.run_sweep(windows_ms, t_hist_ms=50, n_bins=15)

#%% Save Luke McFarland stats in a figure_fixrsvp-compatible format
"""Package and save Luke_2025-08-04 McFarland stats.

We mimic the `output` dict produced by
figure_fixrsvp_mcfarland_covariance_binned.run_mcfarland_on_dataset so that
figure scripts can later do:

    import pickle
    with open(path, 'rb') as f:
        output_luke = pickle.load(f)
    outputs = [output_luke]

and reuse the existing plotting code with minimal changes.
"""

import os
import pickle

subject = 'Luke'
date = '2025-08-04'

# Try to get cids from metadata if present, otherwise fall back to a simple index
try:
    cids_luke = np.array(dset.metadata.get('cids', np.arange(dset['robs'].shape[1])))
except Exception:
    cids_luke = np.arange(dset['robs'].shape[1])

output_luke = {
    'sess': f'{subject}_{date}',
    'cids': cids_luke,
    'neuron_mask': neuron_mask,
    'windows': windows_ms,
    'cids_used': cids_luke[neuron_mask],
    'results': results_luke,
    'last_mats': last_mats_luke,
}

save_dir = Path('../figures')
save_dir.mkdir(exist_ok=True)
save_path = save_dir / f'mcfarland_fixrsvp_Luke_{date}.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(output_luke, f)

print(f"Saved Luke McFarland stats to {save_path}")

#%% Example: look at one window (e.g-----------===============================================. 20 ms)
i = windows_ms.index(20)

FF_uncorr = results_luke[i]['ff_uncorr']
FF_corr   = results_luke[i]['ff_corr']
Erates    = results_luke[i]['Erates']

print("Window:", results_luke[i]['window_ms'], "ms")
print("Mean FF uncorr:", FF_uncorr.mean())
print("Mean FF corr:",   FF_corr.mean())

CnoiseU = last_mats_luke[i]['NoiseCorrU']
CnoiseC = last_mats_luke[i]['NoiseCorrC']

plt.figure()
plt.imshow(CnoiseU, vmin=-0.2, vmax=0.2); plt.title('Noise corr (uncorr)'); plt.colorbar()

plt.figure()
plt.imshow(CnoiseC, vmin=-0.2, vmax=0.2); plt.title('Noise corr (FEM-corrected)'); plt.colorbar()
plt.show()


# Example visualization using masked arrays
good_trials_40 = fix_dur_mc > 40
robs_40 = robs_mc[good_trials_40]
eyepos_40 = eyepos_mc[good_trials_40]
fix_dur_40 = fix_dur_mc[good_trials_40]

ind = np.argsort(fix_dur_40)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos_40[ind,:,0])
plt.xlim(0, 60)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs_40,2)[ind])
plt.xlim(0, 60)




#%%


#%%
cc = 0
ind = np.argsort(fix_dur)

NC = robs.shape[-1]
sx = int(np.sqrt(NC))
sy = int(np.ceil(NC / sx))
fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)
for cc in range(NC):
    ax = axs.flatten()[cc]
    ax.imshow(robs[ind][:,:240,cc], aspect='auto', cmap='gray_r', interpolation='none')
    ax.set_title(f'Cell {cc}')
    ax.axis('off')

#%%
fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)
for cc in range(NC):
    ax = axs.flatten()[cc]
    ax.plot(np.nanmean(robs[:,:120,cc],0), 'k')
    ax.set_title(f'Cell {cc}')
    ax.axis('off')


#%%

itrial += 1
if itrial >= len(robs):
    itrial = 0
fig, axs = plt.subplots(2,1, figsize=(10,5), sharex=True, sharey=False)
axs[1].plot(eyepos[itrial][:,:])
xd = axs[1].get_xlim()
axs[0].imshow(robs[itrial][:,:].T>0, aspect='auto', cmap='gray_r', interpolation='none')
axs[0].set_xlim(xd)
axs[1].set_ylim(-1, 1)


# # %%

#%% 
cc += 1
fig, ax = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=False)
ax.imshow(robs[ind][:,:240,cc], aspect='auto', cmap='gray_r', interpolation='none')

# fun = train_data.dsets[0].metadata['sess'].get_missing_pct_interp(train_data.dsets[0].metadata['cids'])
# # %%
# missing_pct = fun(train_data.dsets[0]['t_bins'])

# _ = plt.plot(train_data.dsets[0]['t_bins'], missing_pct)
# # %%

# threshold = 45

# mask = missing_pct < threshold
# mask[:,np.where(np.median(missing_pct, 0) < threshold)[0]] = True

# plt.imshow(mask[:1000], interpolation='none')


# # %%

# %%
