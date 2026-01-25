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


#%% Directly load Luke_2025-08-04 fixrsvp.dset (bypass YAML / prepare_data)

from pathlib import Path
from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset  # same DictDataset class used elsewhere

subject = "Luke"
date = "2025-08-04"
eye_calibration = "left_eye_x-0.5_y-0.3" # or the one you prefer

sess = get_session(subject, date)
print(f"Session: {sess.name}")
print(f"Processed path: {sess.processed_path}")

fix_path = Path(sess.processed_path) / "datasets" / eye_calibration / "fixrsvp.dset"
print(f"Loading fixrsvp from: {fix_path}")
assert fix_path.exists(), "fixrsvp.dset not found at expected path"

dset_fix = DictDataset.load(fix_path)

print("Loaded fixrsvp DictDataset:")
print(f"  robs:       {dset_fix['robs'].shape}")
print(f"  trial_inds: {dset_fix['trial_inds'].shape}")
print(f"  eyepos:     {dset_fix['eyepos'].shape}")

#%%
"""Trial-align Luke fixrsvp data directly from dset_fix.

We bypass prepare_data and psth_inds; instead we:
- Use trial_inds to group samples into trials
- Apply a simple fixation mask (eye radius < 1 deg)
- Pad trials to the maximum length for plotting convenience
"""

# Extract arrays from DictDataset
trial_inds = dset_fix['trial_inds'].numpy()
trials = np.unique(trial_inds)
NC = dset_fix['robs'].shape[1]
NT = len(trials)

# Determine maximum trial length (in samples)
trial_lengths = [np.sum(trial_inds == tr) for tr in trials]
max_T = int(max(trial_lengths))
print(f"Number of trials: {NT}")
print(f"Number of neurons: {NC}")
print(f"Max trial length: {max_T}")

# Allocate trial-aligned arrays
robs = np.nan * np.zeros((NT, max_T, NC))
eyepos = np.nan * np.zeros((NT, max_T, 2))
fix_dur = np.zeros(NT)

# Define fixation criterion (eye position < 1 degree from center)
eyepos_raw = dset_fix['eyepos'].numpy()
fixation = np.hypot(eyepos_raw[:, 0], eyepos_raw[:, 1]) < 1

for itrial, tr in enumerate(trials):
    ix = (trial_inds == tr) & fixation
    if np.sum(ix) == 0:
        continue

    trial_robs = dset_fix['robs'][ix].numpy()
    trial_eye = eyepos_raw[ix]

    T_trial = trial_robs.shape[0]
    robs[itrial, :T_trial, :] = trial_robs
    eyepos[itrial, :T_trial, :] = trial_eye
    fix_dur[itrial] = T_trial

print(f"Filtered to {np.sum(fix_dur > 0)} trials with at least one fixation sample")

#%% Prepare inputs for McFarland DualWindowAnalysis
"""Prepare Luke fixrsvp spikes/eye traces for McFarland covariance analysis.

We mirror figure_fixrsvp_mcfarland_covariance_binned.run_mcfarland_on_dataset, but
operate directly on dset_fix instead of a CombinedEmbeddedDataset.
"""

# Basic parameters (match original analysis where possible)
windows_ms = [5, 10, 20, 40, 80]
total_spikes_threshold = 200
valid_time_bins = min(240, robs.shape[1])  # clamp to available time
dt = 1 / 240.0  # Rowley fixrsvp is 240 Hz

# Trial selection: require at least 20 valid bins (like original code)

good_trials = fix_dur > 20
robs_mc = robs[good_trials]
eyepos_mc = eyepos[good_trials]

print(f"McFarland prep: {robs_mc.shape[0]} trials, {robs_mc.shape[2]} neurons, {robs_mc.shape[1]} time bins")

# Neuron mask: keep neurons with enough spikes across all good trials
neuron_mask = np.where(np.nansum(robs_mc, axis=(0, 1)) > total_spikes_threshold)[0]
print(f"Using {len(neuron_mask)} neurons / {robs_mc.shape[2]} total (threshold {total_spikes_threshold} spikes)")

# Valid mask: finite spikes and eye positions
valid_mask = (
    np.isfinite(np.sum(robs_mc[:, :, neuron_mask], axis=2)) &
    np.isfinite(np.sum(eyepos_mc, axis=2))
)

# Time index window
iix = np.arange(valid_time_bins)
robs_used = robs_mc[:, iix][:, :, neuron_mask]
eyepos_used = eyepos_mc[:, iix]
valid_used = valid_mask[:, iix]

print(f"robs_used shape: {robs_used.shape}")
print(f"eyepos_used shape: {eyepos_used.shape}")
print(f"valid_used shape: {valid_used.shape}")

from mcfarland_sim import DualWindowAnalysis
# Initialize analyzer (do not run sweep yet)
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

# Try to get cids from metadata if present, otherwise fall back to a simple index

try:
    cids_luke = np.array(dset_fix.metadata.get('cids', np.arange(dset_fix['robs'].shape[1])))
except Exception:
    cids_luke = np.arange(dset_fix['robs'].shape[1])

output_luke = {
    'sess': f'{subject}_{date}',
    'cids': cids_luke,
    'neuron_mask': neuron_mask,
    'windows': windows_ms,
    'cids_used': cids_luke[neuron_mask],
    'results': results_luke,
    'last_mats': last_mats_luke,
}

save_dir = Path('./figures')
save_dir.mkdir(exist_ok=True)
save_path = save_dir / f'mcfarland_fixrsvp_Luke_{date}.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(output_luke, f)

print(f"Saved Luke McFarland stats to {save_path}")

#%% Example: look at one window (e.g. 20 ms)
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

#%%
good_trials = fix_dur > 40
robs = robs[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]

ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 60)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 60)




#%%


#%%
# cc = 0
# ind = np.argsort(fix_dur)

# NC = robs.shape[-1]
# sx = int(np.sqrt(NC))
# sy = int(np.ceil(NC / sx))
# fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)
# for cc in range(NC):
#     ax = axs.flatten()[cc]
#     ax.imshow(robs[ind][:,:240,cc], aspect='auto', cmap='gray_r', interpolation='none')
#     ax.set_title(f'Cell {cc}')
#     ax.axis('off')

# #%%
# fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)
# for cc in range(NC):
#     ax = axs.flatten()[cc]
#     ax.plot(np.nanmean(robs[:,:120,cc],0), 'k')
#     ax.set_title(f'Cell {cc}')
#     ax.axis('off')


# #%%

# itrial += 1
# if itrial >= len(robs):
#     itrial = 0
# fig, axs = plt.subplots(2,1, figsize=(10,5), sharex=True, sharey=False)
# axs[1].plot(eyepos[itrial][:,:])
# xd = axs[1].get_xlim()
# axs[0].imshow(robs[itrial][:,:].T>0, aspect='auto', cmap='gray_r', interpolation='none')
# axs[0].set_xlim(xd)
# axs[1].set_ylim(-1, 1)


# # # %%

# #%% 
# cc += 1
# fig, ax = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=False)
# ax.imshow(robs[ind][:,:240,cc], aspect='auto', cmap='gray_r', interpolation='none')

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

from pathlib import Path
from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset  # same DictDataset class used elsewhere



#%% Check against an old DataYates session
# The session directory is '/mnt/ssd/YatesMarmoV1/processed/Allen_2022-03-04/'
# Extract subject and date from the path

from DataYatesV1 import get_session
subject = 'Allen'
date = '2022-03-04'

# Create session object
sess = get_session(subject, date)

#%% Load a DataYatesV1 session for comparison
"""
Load a DataYatesV1 (old) session to compare ccmax with Rowley data.
Note: DataYatesV1 sessions have different structure than Rowley sessions.
"""

from DataYatesV1 import get_session as get_yates_session
from DataYatesV1 import DictDataset

# Load YatesV1 session
subject = 'Allen'
date = '2022-03-04'

sess_yates = get_yates_session(subject, date)
print(f"YatesV1 Session loaded: {sess_yates.name}")
print(f"Session directory: {sess_yates.sess_dir}")

# List available methods
print(f"\nKey attributes:")
print(f"  sess_dir: {sess_yates.sess_dir}")
print(f"  exp: Experiment metadata")
print(f"  get_dataset(): Load preprocessed datasets")
print(f"  get_spike_times(): Load spike data")

# Look for datasets in this session
print(f"\n" + "="*60)
print("Looking for available datasets in YatesV1 session...")

# DataYatesV1 sessions typically have datasets in a similar structure
datasets_dir = Path(sess_yates.sess_dir) / 'datasets'

if datasets_dir.exists():
    print(f"Found datasets directory: {datasets_dir}")
    yates_available_datasets = {}
    
    for dataset_path in sorted(datasets_dir.glob('**/*.dset')):
        dataset_type = dataset_path.stem
        dataset_dir = dataset_path.parent.name
        key = f"{dataset_type} ({dataset_dir})"
        yates_available_datasets[key] = {
            'type': dataset_type,
            'path': dataset_path
        }
        print(f"  {key}")
    
    # Load fixRSVP if available
    fixrsvp_key = None
    for key in yates_available_datasets.keys():
        if 'fixrsvp' in key.lower():
            fixrsvp_key = key
            break
    
    if fixrsvp_key:
        print(f"\n" + "="*60)
        print(f"Loading YatesV1 fixRSVP: {fixrsvp_key}")
        
        try:
            yates_path = yates_available_datasets[fixrsvp_key]['path']
            dset_yates = DictDataset.load(yates_path)
            
            print(f"✓ Loaded successfully!")
            print(f"  Dataset length: {len(dset_yates)}")
            print(f"  Response shape: {dset_yates['robs'].shape}")
            print(f"  Trial count: {len(np.unique(dset_yates['trial_inds'].numpy()))}")
            
        except Exception as e:
            print(f"✗ Error loading: {e}")
            dset_yates = None
    else:
        print("No fixRSVP dataset found in YatesV1 session")
        dset_yates = None
else:
    print(f"No datasets directory found at {datasets_dir}")
    dset_yates = None

print(f"\n💡 Key differences between DataYatesV1 and DataRowleyV1V2:")
print(f"  - YatesV1 uses 'sess_dir' instead of 'processed_path'")
print(f"  - YatesV1 has 'get_spike_times()' method")
print(f"  - Both use similar '.dset' file format")
print(f"  - DataFramework is compatible across both")


dset_fix = dset_yates

print("Loaded fixrsvp DictDataset:")
print(f"  robs:       {dset_fix['robs'].shape}")
print(f"  trial_inds: {dset_fix['trial_inds'].shape}")
print(f"  eyepos:     {dset_fix['eyepos'].shape}")

#%%
"""Trial-align Allen fixrsvp data directly from dset_fix.

We bypass prepare_data and psth_inds; instead we:
- Use trial_inds to group samples into trials
- Apply a simple fixation mask (eye radius < 1 deg)
- Pad trials to the maximum length for plotting convenience
"""

# Extract arrays from DictDataset
trial_inds = dset_fix['trial_inds'].numpy()
trials = np.unique(trial_inds)
NC = dset_fix['robs'].shape[1]
NT = len(trials)

# Determine maximum trial length (in samples)
trial_lengths = [np.sum(trial_inds == tr) for tr in trials]
max_T = int(max(trial_lengths))
print(f"Number of trials: {NT}")
print(f"Number of neurons: {NC}")
print(f"Max trial length: {max_T}")

# Allocate trial-aligned arrays
robs = np.nan * np.zeros((NT, max_T, NC))
eyepos = np.nan * np.zeros((NT, max_T, 2))
fix_dur = np.zeros(NT)

# Define fixation criterion (eye position < 1 degree from center)
eyepos_raw = dset_fix['eyepos'].numpy()
fixation = np.hypot(eyepos_raw[:, 0], eyepos_raw[:, 1]) < 1

for itrial, tr in enumerate(trials):
    ix = (trial_inds == tr) & fixation
    if np.sum(ix) == 0:
        continue

    trial_robs = dset_fix['robs'][ix].numpy()
    trial_eye = eyepos_raw[ix]

    T_trial = trial_robs.shape[0]
    robs[itrial, :T_trial, :] = trial_robs
    eyepos[itrial, :T_trial, :] = trial_eye
    fix_dur[itrial] = T_trial

print(f"Filtered to {np.sum(fix_dur > 0)} trials with at least one fixation sample")

#%% Prepare inputs for McFarland DualWindowAnalysis
"""Prepare Allen fixrsvp spikes/eye traces for McFarland covariance analysis.

We mirror figure_fixrsvp_mcfarland_covariance_binned.run_mcfarland_on_dataset, but
operate directly on dset_fix instead of a CombinedEmbeddedDataset.
"""

# Basic parameters (match original analysis where possible)
windows_ms = [5, 10, 20, 40, 80]
total_spikes_threshold = 200
valid_time_bins = min(120, robs.shape[1])  # clamp to available time
dt = 1 / 120.0  # Rowley fixrsvp is 240 Hz

# Trial selection: require at least 20 valid bins (like original code)

good_trials = fix_dur > 20
robs_mc = robs[good_trials]
eyepos_mc = eyepos[good_trials]

print(f"McFarland prep: {robs_mc.shape[0]} trials, {robs_mc.shape[2]} neurons, {robs_mc.shape[1]} time bins")

# Neuron mask: keep neurons with enough spikes across all good trials
neuron_mask = np.where(np.nansum(robs_mc, axis=(0, 1)) > total_spikes_threshold)[0]
print(f"Using {len(neuron_mask)} neurons / {robs_mc.shape[2]} total (threshold {total_spikes_threshold} spikes)")

# Valid mask: finite spikes and eye positions
valid_mask = (
    np.isfinite(np.sum(robs_mc[:, :, neuron_mask], axis=2)) &
    np.isfinite(np.sum(eyepos_mc, axis=2))
)

# Time index window
iix = np.arange(valid_time_bins)
robs_used = robs_mc[:, iix][:, :, neuron_mask]
eyepos_used = eyepos_mc[:, iix]
valid_used = valid_mask[:, iix]

print(f"robs_used shape: {robs_used.shape}")
print(f"eyepos_used shape: {eyepos_used.shape}")
print(f"valid_used shape: {valid_used.shape}")

from mcfarland_sim import DualWindowAnalysis
# Initialize analyzer (do not run sweep yet)
analyzer_Allen = DualWindowAnalysis(robs_used, eyepos_used, valid_used, dt=dt)
print("DualWindowAnalysis initialized for Allen_2025-08-04 fixrsvp")

# %% Run McFarland sweep on Allen fixrsvp
windows_ms = [5, 10, 20, 40, 80]
results_Allen, last_mats_Allen = analyzer_Allen.run_sweep(windows_ms, t_hist_ms=50, n_bins=15)

#%% Save Allen McFarland stats in a figure_fixrsvp-compatible format
"""Package and save Allen_2025-08-04 McFarland stats.

We mimic the `output` dict produced by
figure_fixrsvp_mcfarland_covariance_binned.run_mcfarland_on_dataset so that
figure scripts can later do:

    import pickle
    with open(path, 'rb') as f:
        output_Allen = pickle.load(f)
    outputs = [output_Allen]

and reuse the existing plotting code with minimal changes.
"""

import os
import pickle

# Try to get cids from metadata if present, otherwise fall back to a simple index

try:
    cids_Allen = np.array(dset_fix.metadata.get('cids', np.arange(dset_fix['robs'].shape[1])))
except Exception:
    cids_Allen = np.arange(dset_fix['robs'].shape[1])

output_Allen = {
    'sess': f'{subject}_{date}',
    'cids': cids_Allen,
    'neuron_mask': neuron_mask,
    'windows': windows_ms,
    'cids_used': cids_Allen[neuron_mask],
    'results': results_Allen,
    'last_mats': last_mats_Allen,
}

save_dir = Path('../figures')
save_dir.mkdir(exist_ok=True)
save_path = save_dir / f'mcfarland_fixrsvp_Allen_{date}.pkl'

with open(save_path, 'wb') as f:
    pickle.dump(output_Allen, f)

print(f"Saved Allen McFarland stats to {save_path}")

#%% Example: look at one window (e.g. 20 ms)
i = windows_ms.index(20)

FF_uncorr = results_Allen[i]['ff_uncorr']
FF_corr   = results_Allen[i]['ff_corr']
Erates    = results_Allen[i]['Erates']

print("Window:", results_Allen[i]['window_ms'], "ms")
print("Mean FF uncorr:", FF_uncorr.mean())
print("Mean FF corr:",   FF_corr.mean())

CnoiseU = last_mats_Allen[i]['NoiseCorrU']
CnoiseC = last_mats_Allen[i]['NoiseCorrC']

plt.figure()
plt.imshow(CnoiseU, vmin=-0.2, vmax=0.2); plt.title('Noise corr (uncorr)'); plt.colorbar()

plt.figure()
plt.imshow(CnoiseC, vmin=-0.2, vmax=0.2); plt.title('Noise corr (FEM-corrected)'); plt.colorbar()
plt.show()

#%%
good_trials = fix_dur > 40
robs = robs[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]

ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 60)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 60)