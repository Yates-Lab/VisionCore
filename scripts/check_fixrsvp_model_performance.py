## !/usr/bin/env python3


#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs
import torch
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1 import get_clock_functions
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import torch
from eval.eval_stack_utils import run_model

import matplotlib.pyplot as plt
import matplotlib as mpl


# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()

device = get_free_device()

#%% Load an example model (this will be provided in the logging)
print("Discovering available models...")
# checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_history/checkpoints"
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
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

# LOAD A MODEL
model_type = 'resnet_none_convgru'
model, model_info = load_model(
        model_type=model_type,
        model_index=0, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.model.eval()
model.model.convnet.use_checkpointing = False 

model = model.to(device)

#%% Fast Logging code
if hasattr(model.model, 'frontend') and hasattr(model.model.frontend, 'temporal_conv'):
    fig_frontend = plt.plot(model.model.frontend.temporal_conv.weight.squeeze().detach().cpu().T)
    plt.title('Frontend Kernels')

if hasattr(model.model.convnet, 'stem'):
    fig_stem = model.model.convnet.stem.components.conv.plot_weights()
    plt.title('Stem Kernels')

#%%

dataset_idx = 10
print(f"Dataset {dataset_idx}: {model.names[dataset_idx]}")
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

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


robs = np.nan*np.zeros((NT, T, NC))
rhat = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))
for itrial in range(NT):
    ix = trials[itrial] == trial_inds
    ix = ix & fixation

    stim_inds = np.where(ix)[0]
    stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
    stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
    # behavior = dataset.dsets[dset_idx]['behavior'][ix]
    # behavior = None
    # , 'behavior': behavior

    out = run_model(model, {'stim': stim}, dataset_idx=dataset_idx)

    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    rhat[itrial][psth_inds] = out['rhat'].detach().cpu().numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)

good_trials = fix_dur > 40
robs = robs[good_trials]
rhat = rhat[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]
 
ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 60)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 60)



#%% plot all unit PSTHs
sx = int(np.sqrt(NC))
sy = int(np.ceil(NC / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
rhos = []
ve_rate = []
ve_trial = []
for i in range(sx*sy):
    if i >= NC:
        axs.flatten()[i].axis('off')
        continue
    # axs.flatten()[i].imshow(robs[:, :, i][ind], aspect='auto', interpolation='none', cmap='gray_r')
    rbar = np.nanmean(robs[:, :, i][ind], 0)[:120]
    rhatbar = np.nanmean(rhat[:, :, i][ind], 0)[:120]
    iix = np.isfinite(rbar) & np.isfinite(rhatbar)
    rbar = rbar[iix]
    rhatbar = rhatbar[iix]
    resid_rbar = robs[:,:80,i] - rbar[None,:80].repeat(robs.shape[0], axis=0)
    # zscore
    rbar = (rbar - rbar.mean()) / rbar.std()
    rhatbar = (rhatbar - rhatbar.mean()) / rhatbar.std()
    
    ve = 1 - np.nanvar(resid_rbar)/np.nanvar(robs[:,:80,i])
    ve_rate.append(ve)

    resid_trial = robs[:,:80,i] - rhat[:,:80,i]
    ve = 1 - np.nanvar(resid_trial)/np.nanvar(robs[:,:80,i])
    ve_trial.append(ve)

    ax = axs.flatten()[i]
    ax.plot(rbar, 'k')
    ax.plot(rhatbar, 'r')
    rho = np.corrcoef(rbar, rhatbar)[0,1]
    rhos.append(rho)
    ax.set_title(f'{i} rho={rho:.2f}')
    ax.axis('off')

ax.set_xlim(0, 60)
rhos = np.array(rhos)
ve_rate = np.array(ve_rate)
ve_trial = np.array(ve_trial)

plt.savefig(f"../figures/fixrsvp_{dataset.dsets[dset_idx].metadata['sess'].name}_{dataset_idx}_PSTH.pdf")


#%%


plt.plot(np.maximum(ve_rate, -.2), np.maximum(ve_trial, -.2),'.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.axis('equal')
plt.xlabel("Variance Explained by PSTH")
plt.ylabel("Variance Explained by Model")
plt.xlim(-.05, .2)
plt.ylim(-.05, .2)
# plt.plot(np.sort(ve_rate), '.')


#%%
i = -1
#%%

unit_ind = np.argsort(rhos)[::-1]
i += 1
cc = unit_ind[i]
x = robs[:, :, cc][ind]
y = rhat[:, :, cc][ind]
# affine transform y to match x
y = (y - np.nanmean(y)) / np.nanstd(y) * np.nanstd(x) + np.nanmean(x)

fig, axs = plt.subplots(1,2, figsize=(10,3), sharex=True, sharey=False)
axs[0].imshow(x, aspect='auto', cmap='gray_r', interpolation='none')
axs[1].imshow(y, aspect='auto', cmap='gray_r', interpolation='none')
axs[0].set_title(f'Observed {cc}')
axs[1].set_title(f'Predicted {cc} rho={rhos[cc]:.2f}')

plt.show()


n_to_plot = 10
rbar = np.nanmean(x,0)[:80]
rhatbar = np.nanmean(y, 0)[:80]
rhatstd = np.nanstd(y, 0)[:80]
iix = np.isfinite(rbar) & np.isfinite(rhatbar)
rbar = rbar[iix]
rhatbar = rhatbar[iix]

dt = 1/120
rbar = rbar/dt
rhatbar = rhatbar/dt
rhatstd = rhatstd/dt

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.plot(rbar, 'k')
ax.plot(np.maximum(rhatbar, 0), 'r')
ax.fill_between(np.arange(len(rhatbar)), 0, rhatbar + rhatstd, color='r', alpha=0.2)
ax.set_xlabel('Time (bins of 10ms)')
ax.set_ylabel('Firing Rate (Hz)')
ax.set_title(f'Observed vs. Predicted {cc} rho={rhos[cc]:.2f}')
rho = np.corrcoef(rbar, rhatbar)[0,1]

# %%
