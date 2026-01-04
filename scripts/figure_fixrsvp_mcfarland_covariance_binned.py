
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

dataset_idx = 7

#%%

def run_mcfarland_on_dataset(dataset_configs, dataset_idx, windows = [5, 10, 20, 40, 80],
        plot=False, total_spikes_threshold=200, valid_time_bins=240, dt=1/240):
    '''
    Run the Covariance Decomposition on a dataset.

    Inputs:
    -------
    dataset_configs : list
        List of dataset configuration dictionaries
    dataset_idx : int
        Index of the dataset to run on
    windows : list
        List of window sizes to run on (in ms)
    total_spikes_threshold : int
        Minimum number of spikes for a neuron to be included
    valid_time_bins : int
        Maximum number of time bins from each trial included
    dt : float
        Time bin size (in seconds)
    
    Returns:
    --------
    output : dict
        Dictionary containing the results of the analysis

    '''

    dataset_config = dataset_configs[dataset_idx].copy()
    dataset_config['types'] = ['fixrsvp']
    train_data, val_data, dataset_config = prepare_data(dataset_config)
    sess = train_data.dsets[0].metadata['sess']
    ppd = train_data.dsets[0].metadata['ppd']
    cids = dataset_config['cids']
    print(f"Running on {sess.name}")

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
    dfs = np.nan*np.zeros((NT, T, NC))
    eyepos = np.nan*np.zeros((NT, T, 2))
    fix_dur =np.nan*np.zeros((NT,))

    for itrial in tqdm(range(NT)):
        # print(f"Trial {itrial}/{NT}")
        ix = trials[itrial] == trial_inds
        ix = ix & fixation
        if np.sum(ix) == 0:
            continue
        
        psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
        fix_dur[itrial] = len(psth_inds)
        robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
        dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
        eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    

    good_trials = fix_dur > 20
    robs = robs[good_trials]
    dfs = dfs[good_trials]
    # robs[dfs!=True]=np.nan
    eyepos = eyepos[good_trials]
    fix_dur = fix_dur[good_trials]

    if plot:
        ind = np.argsort(fix_dur)[::-1]
        plt.subplot(1,2,1)
        plt.imshow(eyepos[ind,:,0])
        plt.xlim(0, 160)
        plt.subplot(1,2,2)
        plt.imshow(np.nanmean(robs,2)[ind])
        plt.xlim(0, 160)

    # Run the analysis
    output = {}
    output['sess'] = sess.name
    output['cids'] = np.array(cids)


    # 1. Setup
    
    # valid_mask should be True where data is good (no fix breaks)
    neuron_mask = np.where(np.nansum(robs, (0,1))>total_spikes_threshold)[0]
    valid_mask = np.isfinite(np.sum(robs[:,:,neuron_mask], axis=2)) & np.isfinite(np.sum(eyepos, axis=2))
    
    NC = robs.shape[2]
    
    print(f"Using {len(neuron_mask)} neurons / {NC} total")
    iix = np.arange(valid_time_bins)

    robs_used = robs[:,iix][:,:,neuron_mask]
    analyzer = DualWindowAnalysis(robs_used, eyepos[:,iix], valid_mask[:,iix], dt=dt)

    # 2. Run Sweep
    results, last_mats = analyzer.run_sweep(windows, t_hist_ms=50, n_bins=25)
    
    output['neuron_mask'] = neuron_mask
    output['windows'] = windows
    output['cids_used'] = output['cids'][neuron_mask]
    output['results'] = results
    output['last_mats'] = last_mats

    if plot:
        window_idx = 1
        Ctotal = last_mats[window_idx]['Total']
        Cfem = last_mats[window_idx]['FEM']
        Crate = last_mats[window_idx]['Intercept']
        Cpsth = last_mats[window_idx]['PSTH']
        CnoiseU = last_mats[window_idx]['NoiseCorrU']
        CnoiseC = last_mats[window_idx]['NoiseCorrC']
        FF_uncorr = results[window_idx]['ff_uncorr']
        FF_corr = results[window_idx]['ff_corr']
        Erates = results[window_idx]['Erates']


        v = np.max(Cfem.flatten())
        plt.figure()
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
        plt.show()

        # 3. Plot Fano Factor Scaling
        window_ms = [results[i]['window_ms'] for i in range(len(results))]
        ff_uncorr = np.zeros_like(window_ms, dtype=np.float64)
        ff_uncorr_std = np.zeros_like(window_ms, dtype=np.float64)
        ff_uncorr_se = np.zeros_like(window_ms, dtype=np.float64)
        ff_corr = np.zeros_like(window_ms, dtype=np.float64)
        ff_corr_std = np.zeros_like(window_ms, dtype=np.float64)
        ff_corr_se = np.zeros_like(window_ms, dtype=np.float64)

        for iwindow in range(len(window_ms)):
            Erates = results[iwindow]['Erates']
            good = Erates > 0.4
            ff_uncorr[iwindow] = np.nanmedian(results[iwindow]['ff_uncorr'][good])
            ff_corr[iwindow] = np.nanmedian(results[iwindow]['ff_corr'][good])
            ff_uncorr_std[iwindow] = np.nanstd(results[iwindow]['ff_uncorr'][good])
            ff_corr_std[iwindow] = np.nanstd(results[iwindow]['ff_corr'][good])
            ff_uncorr_se[iwindow] = ff_uncorr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_uncorr'][good]))
            ff_corr_se[iwindow] = ff_corr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_corr'][good]))

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

    return output, analyzer

#%%

outputs = []
analyzers = []

for dataset_idx in range(len(dataset_configs)):
    print(f"Running on dataset {dataset_idx}")
    try:
        output, analyzer = run_mcfarland_on_dataset(dataset_configs, dataset_idx, plot=False)
        outputs.append(output)
        analyzers.append(analyzer)
    except Exception as e:
        print(f"Failed to run on dataset {dataset_idx}: {e}")


#%%

from mcfarland_sim import compute_robust_fano_statistics

n = len(outputs[0]['results'])
fig, axs = plt.subplots(1,n, figsize=(3*n, 6), sharex=True, sharey=True)
ffs = []
for i in range(n):
    
    ff_uncorrs = []
    ff_corrs = []
    erates = []
    for j in range(len(outputs)):
        window_ms = outputs[j]['results'][i]['window_ms']
        ff_uncorr = outputs[j]['results'][i]['ff_uncorr']
        ff_corr = outputs[j]['results'][i]['ff_corr']
        Erates = outputs[j]['results'][i]['Erates']
        valid = Erates > 0.1
        ff_uncorrs.append(ff_uncorr[valid])
        ff_corrs.append(ff_corr[valid])
        erates.append(Erates[valid])

        axs[i].plot(Erates, ff_uncorr*Erates, 'r.', alpha=0.1)
        axs[i].plot(Erates, ff_corr*Erates, 'b.', alpha=0.1)
        axs[i].set_xlim(0, 5)
        axs[i].set_ylim(0, 15)
    
    ffs.append({'window_ms': window_ms, 'uncorr': np.concatenate(ff_uncorrs), 'corr': np.concatenate(ff_corrs), 'erate': np.concatenate(erates)})
    
    # axs[i].axhline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Mean Rate (spikes/sec)')
    plt.ylabel('Fano Factor')
    

plt.show()

#%%
ff_stat = compute_robust_fano_statistics(ffs[0]['uncorr']*ffs[0]['erate'], ffs[0]['erate'])

import matplotlib.pyplot as plt
from scipy import stats

def plot_slope_estimation(ax, means, variances, title, color):
    """
    Plots the raw data and the robust slope regression line.
    """
    # 1. Filter robustly
    valid = (means > 0.1) & np.isfinite(variances) & np.isfinite(means)
    x = means[valid]
    y = variances[valid]
    
    # 2. Scatter raw data
    ax.scatter(x, y, s=15, alpha=0.6, c=color, label='Neurons')
    
    # 3. Fit Robust Slope (Fix intercept to 0 or allow float?)
    # Generally allowing float is safer to account for additive noise floor,
    # but theoretically Var = F*Mean implies intercept 0.
    res = stats.linregress(x, y)
    
    # 4. Plot the Regression Line
    x_line = np.linspace(0, x.max(), 100)
    y_line = res.slope * x_line + res.intercept
    
    ax.plot(x_line, y_line, 'k--', linewidth=2, label=f'Slope (Fano) = {res.slope:.2f}')
    
    ax.set_title(title)
    ax.set_xlabel("Mean Rate (spk/s)")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return res


#%%
for i in range(n):
    fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    res = plot_slope_estimation(axs[0], ffs[i]['erate'], ffs[i]['uncorr']*ffs[i]['erate'], "Uncorrected", "tab:blue")
    plot_slope_estimation(axs[1], ffs[i]['erate'], ffs[i]['corr']*ffs[i]['erate'], "Corrected", "tab:orange")
    plt.show()

#%%

def geomean(x):
    return np.exp(np.mean(np.log(x)))

i += 1
x = ffs[i]['uncorr']
y = ffs[i]['corr']

ix = (x > 0.1) & (y > 0.1) & (x < 3)

plt.plot(x[ix], y[ix], '.')
plt.plot(geomean(x[ix]), geomean(y[ix]), 'ro')
plt.axhline(1.0, color='r')
plt.axvline(1.0, color='r')
#%%
np.array(output['cids'])
#%%
cc = 0

#%%
cc += 1
if cc >= len(neuron_mask):
    cc = 0
# cc = 9
analyzer.inspect_neuron_pair(cc, cc, 40, ax=None, show=True)

#%%






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

#
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


# 3. Plot Fano Factor Scaling
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
window_idx = 0
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
for i in range(len(results)):
    plt.plot(results[i]['ff_uncorr'], results[i]['ff_corr'], 'o')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(1.0, color='k', linestyle='--', alpha=0.5)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')

#%%



fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
for i in range(len(results)):
    
    CvarU = np.diag(last_mats[i]['Total']-last_mats[i]['PSTH'])
    CvarC = np.diag(last_mats[i]['Total']-last_mats[i]['Intercept'])
    mu = results[i]['Erates']
    plot_slope_estimation(axs[0], mu, CvarU, "Uncorrected", "tab:blue")
    plot_slope_estimation(axs[1], mu, CvarC, "Corrected", "tab:red")

#%%
fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
for i in range(len(results)):
    CvarU = np.diag(last_mats[i]['Total']-last_mats[i]['PSTH'])
    CvarC = np.diag(last_mats[i]['Total']-last_mats[i]['Intercept'])
    mu = results[i]['Erates']
    axs[0].plot(mu, CvarU, '.')
    axs[1].plot(mu, CvarC, '.')
    
    # plt.plot(results[i]['Erates'],results[i]['ff_corr'], '.' )
axs[0].set_title('Uncorrected')
axs[1].set_title('Corrected')
axs[0].set_xlabel('Mean Rate (spikes/sec)')
axs[1].set_xlabel('Mean Rate (spikes/sec)')
axs[0].set_ylabel('Variance (spikes^2/sec)')   
# plot line of unity
for ax in axs:
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', alpha=0.5)


#%%
for thresh in [0.05, 0.1, 0.2, 0.5]:
    for i in range(len(results)):
        ff = results[i]['ff_corr'][results[i]['Erates'] > thresh]
        mu = np.mean(ff)
        std = np.std(ff)
        plt.errorbar(i+thresh, mu, yerr=std/np.sqrt(len(ff)), fmt='o')

plt.axvline(.2, color='k', linestyle='--', alpha=0.5)
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
