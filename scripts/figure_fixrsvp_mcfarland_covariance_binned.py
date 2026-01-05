
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
    results, last_mats = analyzer.run_sweep(windows, t_hist_ms=50, n_bins=15)
    
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

#%% check a pair
# analyzers[0].inspect_neuron_pair(20,20, 20, ax=None, show=True)

#%%

def get_upper_triangle(C):
    rows, cols = np.triu_indices_from(C, k=1)
    v = C[rows, cols]
    return v


n = len(outputs[0]['results'])
fig, axs = plt.subplots(1,n, figsize=(3*n, 3), sharex=False, sharey=False)
ffs = []
for i in range(n):
    
    ff_uncorrs = []
    ff_corrs = []
    erates = []
    rhos_uncorr = []
    rhos_corr = []
    alphas = []
    for j in range(len(outputs)):
        window_ms = outputs[j]['results'][i]['window_ms']
        ff_uncorr = outputs[j]['results'][i]['ff_uncorr']
        ff_corr = outputs[j]['results'][i]['ff_corr']
        Erates = outputs[j]['results'][i]['Erates']
        alpha = outputs[j]['results'][i]['alpha']
        
        CnoiseU = outputs[j]['last_mats'][i]['NoiseCorrU']
        CnoiseC = outputs[j]['last_mats'][i]['NoiseCorrC']
        rho_uncorr = get_upper_triangle(CnoiseU)
        rho_corr = get_upper_triangle(CnoiseC)

        valid = Erates > 0.1
        ff_uncorrs.append(ff_uncorr[valid])
        ff_corrs.append(ff_corr[valid])
        erates.append(Erates[valid])
        rhos_uncorr.append(rho_uncorr)
        rhos_corr.append(rho_corr)
        alphas.append(alpha[valid])

        axs[i].plot(Erates, ff_uncorr*Erates, 'r.', alpha=0.1)
        axs[i].plot(Erates, ff_corr*Erates, 'b.', alpha=0.1)
        xd = [0, np.percentile(Erates[valid], 99)]
        axs[i].plot(xd, xd, 'k--', alpha=0.5)
        axs[i].set_xlim(xd)
        axs[i].set_ylim(xd[0], xd[1]*2)
        
    
    ffs.append({'window_ms': window_ms,
                'uncorr': np.concatenate(ff_uncorrs),
                'corr': np.concatenate(ff_corrs),
                'erate': np.concatenate(erates),
                'alpha': np.concatenate(alphas),
                'rho_uncorr': np.concatenate(rhos_uncorr),
                'rho_corr': np.concatenate(rhos_corr),
                })
    
    # axs[i].axhline(1.0, color='k', linestyle='--', alpha=0.5)
    axs[i].set_xlabel('Mean')
axs[0].set_ylabel('Variance')
    

plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def slope_ci_t(res, n, ci=0.95):
    """Parametric CI using linregress stderr and t critical value."""
    df = n - 2
    tcrit = stats.t.ppf(0.5 + ci/2, df)
    lo = res.slope - tcrit * res.stderr
    hi = res.slope + tcrit * res.stderr
    return lo, hi

def bootstrap_mean_ci(x, n_boot=5000, ci=95, seed=0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return x.mean(), (lo, hi)

def fisherz_mean_ci(r, n_boot=5000, ci=95, seed=0, eps=1e-6):
    """
    Returns:
      mean_r: tanh(mean(arctanh(r)))
      (lo_r, hi_r): bootstrap CI in r-space (computed by bootstrapping z-means then tanh)
    """
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, (np.nan, np.nan)

    # avoid inf at |r|=1
    r = np.clip(r, -1 + eps, 1 - eps)
    z = np.arctanh(r)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, z.size, size=(n_boot, z.size))
    boot_zmeans = z[idx].mean(axis=1)

    alpha = (100 - ci) / 2
    lo_z, hi_z = np.percentile(boot_zmeans, [alpha, 100 - alpha])

    mean_r = np.tanh(z.mean())
    lo_r, hi_r = np.tanh(lo_z), np.tanh(hi_z)
    return mean_r, (lo_r, hi_r)


def bootstrap_slope_ci(x, y, nboot=5000, ci=0.95, rng=0):
    """
    Nonparametric bootstrap: resample (x_i, y_i) pairs.
    Returns (slope_hat, lo, hi, slopes_boot).
    """
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    rng = np.random.default_rng(rng)

    slopes = np.empty(nboot, dtype=float)
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        slopes[b] = stats.linregress(x[idx], y[idx]).slope

    alpha = 1 - ci
    lo, hi = np.quantile(slopes, [alpha/2, 1 - alpha/2])
    slope_hat = stats.linregress(x, y).slope
    return slope_hat, lo, hi, slopes

def plot_slope_estimation(ax, means, variances, title, color, label=''):
    # Filter
    valid = (means > 0.1) & np.isfinite(variances) & np.isfinite(means)
    x = np.asarray(means[valid])
    y = np.asarray(variances[valid])

    

    res = stats.linregress(x, y)
    
    ax.scatter(x, y, s=15, alpha=0.6, c=color, label=f'{label} FF = {res.slope:.2f}')

    x_line = np.linspace(0, x.max(), 100)
    y_line = res.slope * x_line + res.intercept
    ax.plot(x_line, y_line, 'k--', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Mean Rate (spk/s)")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return res, x, y  # return x,y too so we can bootstrap outside


window_ms = np.array([outputs[0]['results'][i]['window_ms'] for i in range(len(outputs[0]['results']))])
n = len(window_ms)

ff_u_slope = np.zeros(n)
ff_c_slope = np.zeros(n)

# bootstrap CI storage (lo, hi)
ff_u_ci = np.zeros((2, n))
ff_c_ci = np.zeros((2, n))

# optional "aleatoric" variability: IQR of per-neuron fano = var/mean
ff_u_iqr = np.zeros((2, n))
ff_c_iqr = np.zeros((2, n))

fig, axs = plt.subplots(1, n, figsize=(3*n, 3))

for i in range(n):    

    # Uncorrected
    res_u, x_u, y_u = plot_slope_estimation(
        axs[i],
        ffs[i]['erate'],
        ffs[i]['uncorr'] * ffs[i]['erate'],
        "",
        "tab:blue",
        label='Uncorrected'
    )
    ff_u_slope[i] = res_u.slope

    # Bootstrap CI for slope (epistemic, fewer assumptions)
    slope_hat, lo, hi, _ = bootstrap_slope_ci(x_u, y_u, nboot=5000, ci=0.95, rng=123 + i)
    ff_u_ci[:, i] = [lo, hi]

    # "Aleatoric" / population variability proxy: spread of per-neuron fano
    fano_u = y_u / x_u
    ff_u_iqr[:, i] = np.quantile(fano_u, [0.25, 0.75])

    # Corrected
    res_c, x_c, y_c = plot_slope_estimation(
        axs[i],
        ffs[i]['erate'],
        ffs[i]['corr'] * ffs[i]['erate'],
        f"Window {ffs[i]['window_ms']}ms",
        "tab:orange",
        label='Corrected'
    )
    ff_c_slope[i] = res_c.slope

    slope_hat, lo, hi, _ = bootstrap_slope_ci(x_c, y_c, nboot=5000, ci=0.95, rng=999 + i)
    ff_c_ci[:, i] = [lo, hi]

    fano_c = y_c / x_c
    ff_c_iqr[:, i] = np.quantile(fano_c, [0.25, 0.75])

# save figure
fig.savefig('../figures/mcfarland/population_fano_window.pdf', bbox_inches='tight', dpi=300) 

# Sort by window_ms so lines don’t zig-zag
order = np.argsort(window_ms)
window_ms = window_ms[order]

ff_u_slope = ff_u_slope[order]
ff_c_slope = ff_c_slope[order]
ff_u_ci = ff_u_ci[:, order]
ff_c_ci = ff_c_ci[:, order]
ff_u_iqr = ff_u_iqr[:, order]
ff_c_iqr = ff_c_iqr[:, order]

#%% plot population summary
# Convert (lo,hi) to asymmetric yerr for matplotlib: (2, n) = [lower_err; upper_err]
u_yerr = np.vstack([ff_u_slope - ff_u_ci[0], ff_u_ci[1] - ff_u_slope])
c_yerr = np.vstack([ff_c_slope - ff_c_ci[0], ff_c_ci[1] - ff_c_slope])

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
axs[0].errorbar(window_ms, ff_u_slope, yerr=u_yerr, fmt='o-', capsize=3, label='Uncorrected slope (95% bootstrap CI)')
axs[0].errorbar(window_ms, ff_c_slope, yerr=c_yerr, fmt='o-', capsize=3, label='Corrected slope (95% bootstrap CI)')
axs[0].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axs[0].set_xlabel("Window (ms)")
axs[0].set_ylabel("Fano / slope")
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_title("Population Fano")



def geomean(x):
    return np.exp(np.mean(np.log(x)))

ff_u_geomean = np.zeros(n)
ff_c_geomean = np.zeros(n)
for i in range(n):

    x = ffs[i]['uncorr']
    y = ffs[i]['corr']
    ix = np.isfinite(x) & np.isfinite(y)
    ff_u_geomean[i] = geomean(x[ix])
    ff_c_geomean[i] = geomean(y[ix])


axs[1].fill_between(window_ms, ff_u_iqr[0], ff_u_iqr[1], alpha=0.12, label='Uncorr per-neuron Fano IQR')
axs[1].fill_between(window_ms, ff_c_iqr[0], ff_c_iqr[1], alpha=0.12, label='Corr per-neuron Fano IQR')
axs[1].plot(window_ms, ff_u_geomean, 'o-', label='Uncorrected geomean')
axs[1].plot(window_ms, ff_c_geomean, 'o-', label='Corrected geomean')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[1].set_xlabel("Window (ms)")
axs[1].set_ylabel("Fano / geomean")
axs[1].set_title("Per-neuron Fano")
axs[1].axhline(1.0, color='k', linestyle='--', alpha=0.5)

# save fig
fig.savefig('../figures/mcfarland/fano_scaling_summary.pdf', bbox_inches='tight', dpi=300) 


#%% plot noise correlations

fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharex=True, sharey=True)
bins = np.linspace(-.5, .5, 100)

mu_u, lo_u, hi_u = np.empty(n), np.empty(n), np.empty(n)  # uncorrected
mu_c, lo_c, hi_c = np.empty(n), np.empty(n), np.empty(n)  # corrected

for i in range(n):
    # 2D hist
    cnt, xedges, yedges = np.histogram2d(
        ffs[i]['rho_uncorr'], ffs[i]['rho_corr'],
        bins=[bins, bins]
    )
    cnt = np.log1p(cnt)

    axs[i].imshow(
        cnt.T,
        origin='lower',
        aspect='auto',
        interpolation='none',
        cmap='Blues',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    axs[i].plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)
    axs[i].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[i].axvline(0, color='k', linestyle='--', alpha=0.5)
    axs[i].set_title(f"Window {ffs[i]['window_ms']}ms")
    axs[i].set_xlabel("Correlation (Uncorrected)")
    axs[i].set_ylabel("Correlation (Corrected)")

    # means + bootstrap CIs
    mu_u[i] = np.nanmean(ffs[i]['rho_uncorr'])
    sd = np.nanstd(ffs[i]['rho_uncorr'])
    lo_u[i] = mu_u[i] - sd
    hi_u[i] = mu_u[i] + sd  
    mu_c[i] = np.nanmean(ffs[i]['rho_corr'])
    sd = np.nanstd(ffs[i]['rho_corr'])
    lo_c[i] = mu_c[i] - sd
    hi_c[i] = mu_c[i] + sd
    # mu_u[i], (lo_u[i], hi_u[i]) = bootstrap_mean_ci(ffs[i]['rho_uncorr'], seed=10_000 + i)
    # mu_c[i], (lo_c[i], hi_c[i]) = bootstrap_mean_ci(ffs[i]['rho_corr'],   seed=20_000 + i)
    axs[i].plot([mu_u[i]], [mu_c[i]], 'ro')

fig.savefig('../figures/mcfarland/noise_correlations.pdf', bbox_inches='tight', dpi=300) 

#%%
fig2, ax = plt.subplots(figsize=(5, 3))

ax.errorbar(
    window_ms, mu_u,
    yerr=np.vstack([mu_u - lo_u, hi_u - mu_u]),
    fmt='o-', capsize=3, label='Uncorrected'
)
ax.errorbar(
    window_ms, mu_c,
    yerr=np.vstack([mu_c - lo_c, hi_c - mu_c]),
    fmt='o-', capsize=3, label='Corrected'
)

ax.axhline(0, color='k', lw=1, alpha=0.3)
ax.set_xlabel("Window (ms)")
ax.set_ylabel("Mean noise correlation")
ax.legend(frameon=False)
plt.tight_layout()

fig2.savefig('../figures/mcfarland/noise_correlations_mean.pdf', bbox_inches='tight', dpi=300) 

#%%
i = 3
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.hist2d(ffs[i]['rho_uncorr'], ffs[i]['rho_corr'], bins=np.linspace(-.2,.2,100)) #, '.', alpha=0.1)
ax.plot(plt.ylim(), plt.ylim(), 'k--', alpha=0.5)
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.set_title(f"Window {ffs[i]['window_ms']}ms")
ax.set_xlabel("Correlation (Uncorrected)")
ax.set_ylabel("Correlation (Corrected)")


#%% Plot histogram of alpha
fig, ax = plt.subplots(1, n, figsize=(3*n, 3))

for i in range(n):
    alpha = ffs[i]['alpha']
    ax[i].hist(1 - alpha, bins=np.linspace(0, 1, 50))
    ax[i].axvline(np.nanmean(1-alpha), color='r', linestyle='--', alpha=0.5)
    ax[i].set_xlabel("1 - alpha")
    ax[i].set_ylabel("Count")
    ax[i].set_title(f"Window {ffs[i]['window_ms']}ms")

fig.savefig('../figures/mcfarland/alpha.pdf', bbox_inches='tight', dpi=300) 

# plot fano factor vs 1-alpha

for field in ['corr', 'uncorr']:
    fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
    for i in range(n):
        alpha = ffs[i]['alpha']
        ff = ffs[i][field]
        ax[i].plot(1 - alpha, ff, 'o', alpha=0.1)
        ax[i].set_xlim(0, 1)
        ax[i].set_xlabel("1 - alpha")
        ax[i].set_ylabel("Fano Factor")
        ax[i].set_title(f"Window {ffs[i]['window_ms']}ms")
    fig.savefig(f'../figures/mcfarland/ff_vs_alpha_{field}.pdf', bbox_inches='tight', dpi=300) 

#%%
j = 0
i = 1
Ctotal = outputs[j]['last_mats'][i]['Total']
Cfem = outputs[j]['last_mats'][i]['FEM']
Cint = outputs[j]['last_mats'][i]['Intercept']
Cpsth = outputs[j]['last_mats'][i]['PSTH']

CnoiseU = Ctotal - Cpsth
CnoiseC = CnoiseU - Cfem
plt.subplot(1,2,1)
plt.imshow(CnoiseU)
plt.subplot(1,2,2)
plt.imshow(CnoiseC)


#%%
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.diag(Ctotal), np.diag(Cint), '.', label='Cintercept')
plt.plot(np.diag(Ctotal), np.diag(Cpsth), '.', label='Cpsth')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Total Variance (diagonal of Ctotal)')
plt.ylabel('Rate Variance (diagonal of Cint, Cpsth)')
plt.legend()
plt.subplot(1,2,2)
plt.plot(get_upper_triangle(Ctotal), get_upper_triangle(Cint), '.')
plt.plot(get_upper_triangle(Ctotal), get_upper_triangle(Cpsth), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Total Covariance (upper triangle of Ctotal)')
plt.ylabel('Rate Covariance (upper triangle of Cint)')

#%%

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
