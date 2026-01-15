#%%

import sys

sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

import matplotlib as mpl


# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()
device = get_free_device()

from mcfarland_sim import run_mcfarland_on_dataset, extract_metrics
from utils import get_model_and_dataset_configs

#%%

from models.config_loader import load_dataset_configs

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_rowley.yaml"
# dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
    
dataset_configs = load_dataset_configs(dataset_configs_path)



# %%

from models.data import prepare_data

train_data, val_data, dataset_config = prepare_data(dataset_configs[0])


#%%

inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

dataset = train_data.shallow_copy()
dataset.inds = inds

# Some warm up plotting and getting key variables
dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

#%%
robs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))
for itrial in range(NT):
    ix = trials[itrial] == trial_inds
    ix = ix & fixation

    stim_inds = np.where(ix)[0]
    stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
    stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
    behavior = dataset.dsets[dset_idx]['behavior'][ix]

    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)

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
