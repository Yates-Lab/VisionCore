## !/usr/bin/env python3

#%%


# print(dg.__file__)
# print(dg.generate_fixrsvp_dataset.__code__.co_firstlineno)

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch

import os
import contextlib


enable_autoreload()
device = get_free_device()


# %% Try loading a model and predicting
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

# %%
from eval.eval_stack_utils import load_single_dataset

# dataset_idx = 10   
# train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
subject = 'Allen'
date = '2022-03-04'
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp_all_cells.yaml'
dataset_configs = load_dataset_configs(dataset_configs_path)

dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
    train_data, val_data, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)


# Combine all indices (train + validation) for maximum data (we don't tend to train on gratings so this should be okay)
inds = torch.concatenate([
    train_data.get_dataset_inds('fixrsvp'),
    val_data.get_dataset_inds('fixrsvp')
], dim=0)

dataset = train_data.shallow_copy()
dataset.inds = inds
# %%

dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

robs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))
for itrial in range(NT):
    ix = trials[itrial] == trial_inds
    ix = ix & fixation

    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)

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

sx = int(np.sqrt(NC))
sy = int(np.ceil(NC / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= NC:
        axs.flatten()[i].axis('off')
        continue
    # axs.flatten()[i].imshow(robs[:, :, i][ind], aspect='auto', interpolation='none', cmap='gray_r')
    axs.flatten()[i].plot(np.nanmean(robs[:, :, i][ind], 0), 'r')
    axs.flatten()[i].set_title(f'{i}')
    axs.flatten()[i].axis('off')
axs.flatten()[i].set_xlim(0, 60)
#%%
sess = dataset.dsets[dset_idx].metadata['sess']
ks_results = sess.ks_results
st = ks_results.spike_times
clu = ks_results.spike_clusters

#%%

plt.imshow(dataset.dsets[dset_idx].covariates['robs'][ix].T, aspect='auto', interpolation='none')
from DataYatesV1.exp.dataset_generation import generate_fixrsvp_dataset
from scipy.interpolate import interp1d

roi_src = dataset.dsets[dset_idx].metadata['roi_src']
roi_src[0] = [-150, 150]
roi_src[1] = [-150, 150]

pix_interp = interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['dpi_pix'], kind='linear', fill_value='extrapolate', axis=0)
 
interps = {
    'dpi_raw': interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['dpi_raw'], kind='linear', fill_value='extrapolate', axis=0),
    'dpi_valid': interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['dpi_valid'], kind='nearest', fill_value='extrapolate'),
    'eyepos': interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['eyepos'], kind='linear', fill_value='extrapolate', axis=0)
}

#%%
exp = sess.exp
from DataYatesV1.exp.general import get_trial_protocols
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1.utils.general import get_clock_functions
from tqdm import tqdm
protocols = get_trial_protocols(exp)
ptb2ephys, _ = get_clock_functions(exp)

st = ks_results.spike_times
clu = ks_results.spike_clusters
cids = np.unique(clu)

fixrsvp_trials = [(iT, FixRsvpTrial(exp['D'][iT], exp['S'])) for iT in range(len(exp['D'])) if protocols[iT] == 'FixRsvpStim']

fixrsvp_dict = {
    't_bins': [],
    'trial_inds': [],
    'psth_inds': [],
    'stim': [],
    'robs': [],
    'dpi_pix': [],
    'roi': [],
}
for k in interps:
    fixrsvp_dict[k] = []
for iT, trial in tqdm(fixrsvp_trials, 'FixRsvp trials'):
     # The first image is different on every trial so we skip it
    image_ids = trial.image_ids
    if len(np.unique(trial.image_ids)) < 2:
            continue
    print(trial, image_ids)
    start_idx = np.where(image_ids == 2)[0][0]

#%%
# import importlib
# import DataYatesV1.exp.dataset_generation as dg
# importlib.reload(dg)
fixrsvp_dset = generate_fixrsvp_dataset(sess.exp, sess.ks_results, roi_src, pix_interp, interps=interps, min_duration=.5, dt=1/240, metadata=dataset.dsets[dset_idx].metadata)

# %% Apply transforms
from models.data.transforms import make_pipeline
from models.data.datafilters import make_datafilter_pipeline
from models.data.loading import apply_downsampling
# import ensure_tensor
from models.utils.general import ensure_tensor
from models.data.filtering import get_valid_dfs
n_lags = 32

# -- unpack ----------------------------------------------------------------
sess_name  = dataset_config["session"]
dset_types = dataset_config["types"]
transforms  = dataset_config.get("transforms", {})
datafilters = dataset_config.get("datafilters", {})
keys_lags  = dataset_config["keys_lags"]
sampling_config = dataset_config.get("sampling", None)


# -------------------------------------------------------------------------
# Calculate downsampling factor if sampling config is present
# -------------------------------------------------------------------------
downsample_factor = 1
if sampling_config:
    source_rate = sampling_config["source_rate"]
    target_rate = sampling_config["target_rate"]
    downsample_factor = source_rate // target_rate
    print(f"Downsampling from {source_rate}Hz to {target_rate}Hz (factor: {downsample_factor})")

# -------------------------------------------------------------------------
# Build transform specs once
# -------------------------------------------------------------------------
transform_specs = {}
for var_name, spec in transforms.items():
    pipeline = make_pipeline(spec.get("ops", []), dataset_config)
    transform_specs[var_name] = dict(
        source      = spec.get("source", var_name),
        pipeline    = pipeline,
        expose_as   = spec.get("expose_as", var_name),
        concatenate = spec.get("concatenate", False),  # Default to overwrite behavior
        )

# Merge any per-variable keys_lags into the master dict
if "keys_lags" in spec:
    keys_lags[spec["expose_as"]] = spec["keys_lags"]

# -------------------------------------------------------------------------
# Build datafilter specs once
# -------------------------------------------------------------------------
datafilter_specs = {}
for var_name, spec in datafilters.items():
    pipeline = make_datafilter_pipeline(spec.get("ops", []))
    datafilter_specs[var_name] = dict(
        pipeline   = pipeline,
        expose_as  = spec.get("expose_as", var_name),
    )

    # Merge any per-variable keys_lags into the master dict
    if "keys_lags" in spec:
        keys_lags[spec["expose_as"]] = spec["keys_lags"]

# -------------------------------------------------------------------------
# Apply transforms and datafilters to fixrsvp dataset
fixrsvp_dset.metadata['sess'] = sess

# -------------------------------------------------------------
# Apply downsampling if specified
# -------------------------------------------------------------
if downsample_factor > 1:
    print(f"Applying downsampling to FixRSVP dataset:")
    fixrsvp_dset = apply_downsampling(fixrsvp_dset, downsample_factor)

# -------------------------------------------------------------
# Apply datafilter pipelines
# -------------------------------------------------------------
if datafilter_specs:
    for var_name, spec in datafilter_specs.items():
        expose_as = spec["expose_as"]
        # print(f"Applying datafilter → {expose_as}")
        mask_tensor = spec["pipeline"](fixrsvp_dset)
        fixrsvp_dset[expose_as] = mask_tensor
else:
    # Fallback to old behavior if no datafilters specified
    fixrsvp_dset['dfs'] = get_valid_dfs(fixrsvp_dset, n_lags)

# -------------------------------------------------------------
# Apply transform pipelines
# -------------------------------------------------------------
# Collect transformed variables by expose_as name for potential concatenation
transformed_vars = {}
concatenate_vars = {}  # Track which variables should be concatenated

for var_name, spec in transform_specs.items():
    
    src_key     = spec["source"]
    expose_as   = spec["expose_as"]
    concatenate = spec["concatenate"]
    # print(f"Transforming {src_key} → {expose_as}")
    data_tensor = ensure_tensor(fixrsvp_dset[src_key], dtype=torch.float32)   # → torch.Tensor
    data_tensor = spec["pipeline"](data_tensor)
    # print(f"{expose_as} shape: {data_tensor.shape}")

    if concatenate:
        # Collect variables marked for concatenation
        if expose_as not in transformed_vars:
            transformed_vars[expose_as] = []
            concatenate_vars[expose_as] = True
        transformed_vars[expose_as].append(data_tensor)
    else:
        # Overwrite behavior (default) - assign directly
        fixrsvp_dset[expose_as] = data_tensor

# Concatenate variables that were marked for concatenation
for expose_as, var_list in transformed_vars.items():
    if concatenate_vars.get(expose_as, False):
        if len(var_list) == 1:
            # Single variable, no concatenation needed
            fixrsvp_dset[expose_as] = var_list[0]
        else:
            # Multiple variables, concatenate along last dimension

            concatenated = torch.cat(var_list, dim=-1)
            fixrsvp_dset[expose_as] = concatenated
            # print(f"Concatenated {len(var_list)} variables for {expose_as}, final shape: {concatenated.shape}")


#%%
from models.data.loading import get_embedded_datasets
train_dset, val_dset = get_embedded_datasets(
        sess,
        types            = [fixrsvp_dset],           # pass in the preprocessed datasets
        keys_lags        = keys_lags,
        train_val_split  = dataset_config["train_val_split"],
        cids             = dataset_config.get("cids", None),
        seed             = dataset_config.get("seed", 1002),
        pre_func         = lambda x: x,          # preprocessing already done
)

# combine indicies for train and test
inds = torch.concatenate([train_dset.inds, val_dset.inds], dim=0)
train_dset.inds = inds

# %%
num_frames_original = dataset.dsets[dset_idx].covariates['stim'].shape[0]
num_frames_new = train_dset.dsets[0]['stim'].shape[0]

print(f"Original dataset has {num_frames_original} frames, new dataset has {num_frames_new} frames")

#%%
iframe = 10
H = dataset.dsets[dset_idx].covariates['stim'][iframe].shape[1]
jj,ii = np.meshgrid(np.arange(H), np.arange(H))

def crop(x, cx, cy):
    return x[ii + cy, jj + cx]

#%%


iframe += 1
cy = 75
cx = 55
plt.subplot(1,2,1)
plt.imshow(dataset.dsets[dset_idx].covariates['stim'][iframe][0], vmin=-1, vmax=1)

plt.subplot(1,2,2)
plt.imshow(fixrsvp_dset['stim'][iframe][0], vmin=-1, vmax=1)


#%%
from tqdm import tqdm
batch_size = 128
N = len(train_dset)

def get_loss(model, dataset, dataset_idx, batch_size, stimfun=lambda x: x):
    N = len(dataset)
    losses = []
    b = 0
    for iB in range(0, N, batch_size):
        batch = dataset[iB:iB+batch_size]
        batch['stim'] = stimfun(batch['stim'])
        batch['dataset_idx'] = [dataset_idx]
        loss = model._step([batch], tag='val').item()  # batch_list, tag: str)
        losses.append(loss) 
        b += 1
        # if b > 5:
        #     break
    
    return np.mean(losses)

#%%
loss = get_loss(model, dataset, dataset_idx, batch_size)
print(f"Loss: {loss}")

#%%
batch = train_dset[0:batch_size]
batch['stim'].shape

def stimfun(x):
    return x[:, :, :, ii + cy, jj + cx]

loss2 = get_loss(model, train_dset, dataset_idx, batch_size, stimfun)
# batch['dataset_idx'] = [dataset_idx]

#%%
mid = fixrsvp_dset['stim'].shape[-1]//2
max_shift = mid - H

#%%
cxs = np.arange(-max_shift, max_shift, 2)
cys = np.arange(-max_shift, max_shift, 2)
cxx, cyy = np.meshgrid(cxs, cys)

Nshift = len(cxs)
losses = np.zeros((Nshift, Nshift))
for i, cx in enumerate(tqdm(cxs, desc='Shifting')):
    for j, cy in enumerate(cys):
        # print(f"Shifting {cx}, {cy}")

        def stimfun(x):
            return x[:, :, :, ii + cy + mid, jj + cx + mid]
        losses[i,j] = get_loss(model, train_dset, dataset_idx, batch_size, stimfun)

#%%
plt.imshow(losses-loss, extent=[cxs[0], cxs[-1], cys[0], cys[-1]], origin='lower', cmap='jet')
i, j = np.unravel_index(np.argmin(losses), losses.shape)
plt.title(f"Best shift: {cxs[i]}, {cys[j]}, min loss: {np.min(losses-loss)}")
plt.colorbar()
plt.savefig(f"../figures/shift_search_{dataset.dsets[dset_idx].metadata['sess'].name}.png")
np.save(f"../figures/shift_search_{dataset.dsets[dset_idx].metadata['sess'].name}.npy", losses)
# out = model._step([batch], tag='val').item()  # batch_list, tag: str)

#%%
batch1 = dataset[0:batch_size]
batch2 = train_dset[0:batch_size]
cx = cxs[i]
cy = cys[j]
def stimfun(x):
    return x[:, :, :, ii + cy + mid, jj + cx + mid]
batch2['stim'] = stimfun(batch2['stim'])

plt.figure()
plt.subplot(1,2,1)
plt.imshow(batch1['stim'][0,0,0].detach().cpu().numpy(), vmin=-1, vmax=1)
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(batch2['stim'][0,0,0].detach().cpu().numpy(), vmin=-1, vmax=1)
plt.title('Best shift')
plt.savefig(f"../figures/best_shift_{dataset.dsets[dset_idx].metadata['sess'].name}.png")
# out
# batch = run_model(model, batch, dataset_idx=dataset_idx)