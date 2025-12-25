
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


# from mcfarland_sim import _savgol_1d_nan, savgol_nan_numpy, savgol_nan_torch
from mcfarland_sim import DualWindowAnalysis
    
#%% Load Dataset
from tqdm import tqdm

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

dataset_idx = 2
include_time_lags = False
dataset_configs[dataset_idx]['types'] = ['fixrsvp']
train_data, val_data, dataset_config = prepare_data(dataset_configs[dataset_idx])
sess = train_data.dsets[0].metadata['sess']
ppd = train_data.dsets[0].metadata['ppd']
cids = dataset_config['cids']

# get fixrsvp inds and make one dataaset object
inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

dataset = train_data.shallow_copy()
dataset.inds = inds

# Getting key variables
dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

# Loop over trials and align responses
robs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))

for itrial in tqdm(range(NT)):
    ix = trials[itrial] == trial_inds
    ix = ix & fixation
    if np.sum(ix) == 0:
        continue
    
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    

good_trials = fix_dur > 20
robs = robs[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]

ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 160)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 160)

#%% Run the analysis

# 1. Setup
# Assuming 'robs', 'eyepos', 'valid_mask' are already loaded from your dataset code
# valid_mask should be True where data is good (no fix breaks)
valid_mask = np.isfinite(np.sum(robs, axis=2)) & np.isfinite(np.sum(eyepos, axis=2))
neuron_mask = np.where(np.nansum(robs, (0,1))>500)[0]
iix = np.arange(250)
robs_used = robs[:,iix][:,:,neuron_mask]
analyzer = DualWindowAnalysis(robs_used, eyepos[:,iix], valid_mask[:,iix], dt=1/240)

# 2. Run Sweep
windows = [5, 10, 20, 40]
results, last_mats = analyzer.run_sweep(windows, t_hist_ms=1)

cc = 0
#%%
cc += 1
if cc >= len(neuron_mask):
    cc = 0

analyzer.inspect_neuron_pair(cc, cc, 5, ax=None, show=True)

#%%


window_idx = 0
Ctotal = last_mats[window_idx]['Total']
Cfem = last_mats[window_idx]['FEM']
Crate = last_mats[window_idx]['Intercept']
Cpsth = last_mats[window_idx]['PSTH']
CnoiseU = last_mats[window_idx]['NoiseCorrU']
CnoiseC = last_mats[window_idx]['NoiseCorrC']
FF_uncorr = results[window_idx]['ff_uncorr']
FF_corr = results[window_idx]['ff_corr']



v = np.max(Cfem.flatten())
plt.subplot(1,3,1)
plt.imshow(Ctotal, vmin=-v, vmax=v)
plt.title('Total')
plt.subplot(1,3,2)
plt.imshow(Cfem, vmin=-v, vmax=v)
plt.title('Eye')
plt.subplot(1,3,3)
plt.imshow(Cpsth, vmin=-v, vmax=v)
plt.title('PSTH')

plt.figure()
plt.subplot(1,2,1)
v = .2
plt.imshow(CnoiseU, vmin=-v, vmax=v)
plt.colorbar()
plt.title('Noise (Uncorrected))')
plt.subplot(1,2,2)
plt.imshow(CnoiseC, vmin=-v, vmax=v)
plt.colorbar()
plt.title('Noise (Corrected) ')


plt.figure()
plt.plot(FF_uncorr, FF_corr, '.')
plt.axhline(1, color='k', linestyle='--')
plt.axvline(1, color='k', linestyle='--')
plt.plot(np.mean(FF_uncorr), np.mean(FF_corr), 'ro')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')
plt.title(f"FF Window Size ({windows[window_idx]}ms)")

#%%
def get_upper_triangle(C):
    rows, cols = np.triu_indices_from(C, k=1)
    v = C[rows, cols]
    return v

rho_uncorr = get_upper_triangle(CnoiseU)
rho_corr = get_upper_triangle(CnoiseC)

plt.figure()
plt.plot(rho_uncorr, rho_corr, '.', alpha=0.1)
# plot mean
plt.plot(rho_uncorr.mean(), rho_corr.mean(), 'ro')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Correlation (Uncorrected)')
plt.ylabel('Correlation (Corrected)')
plt.title('Correlation vs Window Size')


#%% 3. Plot Fano Factor Scaling
window_ms = [results[i]['window_ms'] for i in range(len(results))]
ff_uncorr = np.zeros_like(window_ms, dtype=np.float64)
ff_uncorr_std = np.zeros_like(window_ms, dtype=np.float64)
ff_uncorr_se = np.zeros_like(window_ms, dtype=np.float64)
ff_corr = np.zeros_like(window_ms, dtype=np.float64)
ff_corr_std = np.zeros_like(window_ms, dtype=np.float64)
ff_corr_se = np.zeros_like(window_ms, dtype=np.float64)

for iwindow in range(len(window_ms)):
    ff_uncorr[iwindow] = np.nanmean(results[iwindow]['ff_uncorr'])
    ff_corr[iwindow] = np.nanmean(results[iwindow]['ff_corr'])
    ff_uncorr_std[iwindow] = np.nanstd(results[iwindow]['ff_uncorr'])
    ff_corr_std[iwindow] = np.nanstd(results[iwindow]['ff_corr'])
    ff_uncorr_se[iwindow] = ff_uncorr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_uncorr']))
    ff_corr_se[iwindow] = ff_corr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_corr']))

plt.figure(figsize=(8, 6))
plt.plot(window_ms, ff_uncorr, 'o-', label='Standard (Uncorrected)')
plt.plot(window_ms, ff_corr, 'o-', label='FEM-Corrected')
# plot error bars
plt.fill_between(window_ms, ff_uncorr - ff_uncorr_se, ff_uncorr + ff_uncorr_se, alpha=0.2)
plt.fill_between(window_ms, ff_corr - ff_corr_se, ff_corr + ff_corr_se, alpha=0.2)

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Count Window (ms)')
plt.ylabel('Mean Fano Factor')
plt.title('Integration of Noise: FEM Correction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
window_idx = 4
Sigma_FEM = last_mats[window_idx]['FEM']
u, s, vh = np.linalg.svd(Sigma_FEM)
plt.figure()
plt.plot(s, 'o-', label='FEM')

Sigma_PSTH = last_mats[window_idx]['PSTH']
u, s, vh = np.linalg.svd(Sigma_PSTH)
plt.plot(s, 'o-', label='PSTH')

# same for total covariance
Sigma_Total = last_mats[window_idx]['Total']
u, s, vh = np.linalg.svd(Sigma_Total)
plt.plot(s, 'o-', label='Total')


# # now noise cov
# Sigma_Noise = last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH']
# u, s, vh = np.linalg.svd(Sigma_Noise)
# plt.plot(s, 'o-', label='Noise Uncorrected')

# Sigma_Noise = last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'] - last_mats[window_idx]['PSTH']
# u, s, vh = np.linalg.svd(Sigma_Noise)
# plt.plot(s, 'o-', label='Noise Corrected')
plt.title(f"Singular Values ({windows[window_idx]}ms)")
plt.legend()
# plt.yscale('log')
plt.show()
# %%
i = 3
plt.plot(results[i]['ff_uncorr'], results[i]['ff_corr'], 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')


#%%
results[0]
# %%
# show the total covariance matrix subtracting the diagonal
window_idx = 1
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(last_mats[window_idx]['Total'] - np.diag(np.diag(last_mats[window_idx]['Total'])))
plt.title(f"Total Covariance ({windows[window_idx]}ms)")

# show FEM
plt.subplot(1,3,2)
plt.imshow(last_mats[window_idx]['FEM'] - np.diag(np.diag(last_mats[window_idx]['FEM'])))
plt.title(f"FEM Covariance ({windows[window_idx]}ms)")

# show Noise_Corr
plt.subplot(1,3,3)
plt.imshow(last_mats[window_idx]['PSTH'] - np.diag(np.diag(last_mats[window_idx]['PSTH'])))
plt.title(f"PSTH Covariance ({windows[window_idx]}ms)")



# %%
plt.subplot(1,2,1)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH'])
plt.subplot(1,2,2)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'])

# %%
