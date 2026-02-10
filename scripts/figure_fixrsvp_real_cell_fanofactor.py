
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

#%% Utility function for smoothing eye position
import numpy as np
import torch

from scipy.signal import savgol_filter

def _savgol_1d_nan(y, window_length=15, polyorder=3):
    """
    Apply Savitzky–Golay to a 1D array with NaNs.
    NaNs are interpolated for filtering and then restored.
    """
    y = np.asarray(y, float)
    mask = np.isfinite(y)

    # If too few valid points, just return original
    if mask.sum() < polyorder + 2:
        return y

    yy = y.copy()
    idx_valid = np.where(mask)[0]
    idx_nan   = np.where(~mask)[0]

    # Linear interp over NaNs so savgol_filter has no gaps
    yy[idx_nan] = np.interp(idx_nan, idx_valid, yy[idx_valid])

    # Apply SG filter
    ys = savgol_filter(
        yy,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )

    # Restore original NaNs
    ys[~mask] = np.nan
    return ys


def savgol_nan_numpy(x, axis=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along a given axis for a NumPy array.
    """
    return np.apply_along_axis(
        _savgol_1d_nan,
        axis=axis,
        arr=x,
        window_length=window_length,
        polyorder=polyorder,
    )


def savgol_nan_torch(x, dim=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along dim for a torch.Tensor.
    - x: (..., T, ...) tensor
    - dim: time dimension (default 1)
    """
    # Move target dim to last for easier NumPy apply
    x_np = x.detach().cpu().numpy()
    x_np = np.moveaxis(x_np, dim, -1)

    y_np = savgol_nan_numpy(
        x_np,
        axis=-1,
        window_length=window_length,
        polyorder=polyorder,
    )

    # Move axis back and convert to torch
    y_np = np.moveaxis(y_np, -1, dim)
    y = torch.from_numpy(y_np).to(x.device).type_as(x)
    return y

#%% Utility function for fitting exponential
import numpy as np
# from scipy.optimize import curve_fit

# def fit_exponential_decay(x, y, weighted=False):
#     """
#     Fit y(x) = A * exp(-tau * x) + plateau
#     """
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
    
#     # 1. Estimate initial guesses
#     # Plateau: mean of tail (last few points) or min
#     plateau_guess = np.mean(np.partition(y, 2)[:2]) 
    
#     # Amplitude: Range of data
#     A_guess = np.max(y) - plateau_guess
    
#     # Tau: Guess based on reaching halfway point
#     # Find x where y drops halfway between max and plateau
#     half_y = plateau_guess + A_guess / 2
#     idx_half = np.argmin(np.abs(y - half_y))
#     x_half = x[idx_half] if x[idx_half] > 0 else x[1] if len(x) > 1 else 1.0
#     tau_guess = np.log(2) / x_half

#     # 2. Define Model
#     def model(x_val, A_val, tau_val, plat_val):
#         return A_val * np.exp(-tau_val * x_val) + plat_val

#     # 3. Fit
#     try:
#         # Bounds: A > 0, tau > 0, plateau can be anywhere (usually > 0)
#         p0 = [A_guess, tau_guess, plateau_guess]
#         bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        
#         # Sigma for weighting (optional)
#         # weight points closer to 0 more? Or weight by variance?
#         # For now, uniform weighting is usually safer unless we have error bars.
#         sigma = None 
        
#         popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, sigma=sigma, maxfev=10000)
#         A_fit, tau_fit, plat_fit = popt
        
#     except RuntimeError:
#         # Fallback if fit fails
#         A_fit, tau_fit, plat_fit = (0, 0, np.mean(y))

#     # 4. Generate fit curve
#     y_fit = model(x, A_fit, tau_fit, plat_fit)
#     mse = np.mean((y - y_fit)**2)

#     return {
#         'A': A_fit,
#         'tau': tau_fit,
#         'plateau': plat_fit,
#         'y_fit': y_fit,
#         'mse': mse,
#         # Evaluate at 0 for the McFarland correction
#         'intercept_at_0': A_fit + plat_fit 
#     }

from scipy.optimize import curve_fit

def fit_exponential_decay(x, y, total_var_limit=None, weighted=True):
    """
    Robust fit using non-linear least squares with bounds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Heuristic Initial Guesses
    plateau_guess = np.min(y)
    amplitude_guess = np.max(y) - plateau_guess
    
    # Estimate tau (find where y drops by half)
    half_val = plateau_guess + amplitude_guess/2
    idx = (np.abs(y - half_val)).argmin()
    tau_guess = 1.0 / (x[idx] + 1e-6)

    # 2. Define the Model (Exponential)
    def model_exp(x_val, A, tau, plat):
        return A * np.exp(-tau * x_val) + plat

    # 3. Fit with Bounds
    # A > 0, tau > 0, plateau > 0
    # We use 'soft_l1' loss to ignore outliers in the bins
    try:
        p0 = [amplitude_guess, tau_guess, plateau_guess]
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        
        # Weighted by value (higher variance bins count more)
        sigma = 1.0 / (y + 1e-6) if weighted else None

        popt, _ = curve_fit(
            model_exp, x, y, 
            p0=p0, 
            bounds=bounds, 
            sigma=sigma, 
            loss='soft_l1', 
            maxfev=5000
        )
        A, tau, plat = popt
    except RuntimeError:
        return {'A': 0, 'tau': 0, 'plateau': np.mean(y), 'intercept_at_0': np.mean(y), 'y_fit': y}

    # 4. Calculate Extrapolated Variance
    extrapolated_var = A + plat

    # --- CRITICAL FIX: LOGICAL CLAMP ---
    # The signal variance (extrapolated_var) generally shouldn't exceed Total Variance.
    # However, for Poisson, Var_Total = Var_Signal + Mean.
    # So Var_Signal should be <= Var_Total - Mean.
    if total_var_limit is not None:
        if extrapolated_var > total_var_limit:
            # If the fit overshoots, clamp A so that A + plat = limit
            # This forces the curve to respect the physics of the cell
            extrapolated_var = total_var_limit
            A = extrapolated_var - plat

    y_fit = model_exp(x, A, tau, plat)

    return {
        'A': A,
        'tau': tau,
        'plateau': plat,
        'y_fit': y_fit,
        'intercept_at_0': extrapolated_var
    }

#%% Load Dataset
from tqdm import tqdm

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

dataset_idx = 7
include_time_lags = False
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

# Loop over trials and compute the response from the pyramid
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

# %% Plot eye position and example trials
dt = 1/240
t_bins = np.arange(T)*dt
eyepos_centered = eyepos - np.nanmedian(eyepos, (0,1), keepdims=True)
eyepos_centered = savgol_nan_torch(torch.from_numpy(eyepos_centered), dim=1, window_length=9, polyorder=3)
iix = np.hypot(eyepos_centered[:,:,0], eyepos_centered[:,:,1]) < .5
eyepos_centered[~iix] = np.nan
plt.figure(figsize=(5, 3))

# _ = plt.plot(t_bins, eyepos_centered[:,:,0].T, 'k', alpha=0.2)
plot_colors = ['g', 'b', 'r', 'c', 'm', 'y']
n_to_plot = 5
trials_to_plot = np.random.choice(np.where(fix_dur > 1/dt)[0], size=n_to_plot, replace=False)
plot_colors = plot_colors[:n_to_plot]

def plot_trial(trial_idx, color):
    sorted_trial = np.where(np.isin(ind, np.array([trial_idx])))[0].item()
    _ = plt.plot(t_bins, eyepos_centered[trial_idx,:,0].T, color, alpha=1.0, label=f"Trial {sorted_trial}")
for i, color in zip(trials_to_plot, plot_colors):
    plot_trial(i, color)

plt.axhline(-.5, color='r', linestyle='--')
plt.axhline(.5, color='r', linestyle='--')
plt.xlim(0, 1.0)
plt.ylim(-.75, .75)
plt.xlabel('Time (s)')
plt.ylabel('Eye Position (deg)')
plt.legend()
plt.savefig("../figures/tejas_poster/eye_pos_time.pdf", bbox_inches='tight', dpi=300)

#%% Make plots for the example units

dt = 1/240

cids_to_plot = [31, 33, 49, 53, 96, 115, 126]
    
for iunit, cid in enumerate(cids_to_plot):
    id = np.where(np.isin(cids, cid))[0]
    Y = robs[:,:,id][:,:,0]
        
    fig = plt.figure(figsize=(6,4))
    GridSpec = fig.add_gridspec(2, 1, height_ratios=[1, .5], hspace=0.15)
    ax1 = fig.add_subplot(GridSpec[0, 0])

    ax1.imshow(Y, aspect='auto', interpolation='none', cmap='gray_r')
    ax1.set_xlim(0, 240)
    ax1.set_title(f'Unit {cid}')

    ax2 = fig.add_subplot(GridSpec[1, 0])
    ax2.plot(np.nanmean(Y, 0)/dt, 'r')
    ax2.plot(np.nanstd(Y, 0)/dt, 'r', alpha=.5)
    ax2.set_xlim(0, 240)

    for itrial, color in zip(trials_to_plot, plot_colors):
        ax1.axhline(itrial, color=color, linestyle='-', linewidth=2, alpha=.5)

    # save figure as pdf
    fig.savefig(f'../figures/tejas_poster/pyramid_simulation_{sess.name}_{cid}.pdf', bbox_inches='tight', dpi=300)


#%%


# plot rates for those trials
for iunit, cid in enumerate(cids_to_plot):
    plt.figure(figsize=(5, 3))
    id = np.where(np.isin(cids, cid))[0]
    Y = robs[:,:,id][:,:,0]

    for itrial, color in zip(trials_to_plot, plot_colors):
        sorted_trial = np.where(np.isin(ind, np.array([itrial])))[0].item()
        _ = plt.plot(t_bins, Y[itrial], color, alpha=1.0, label=f"Trial {sorted_trial}", linewidth=1)
    
    plt.xlim(0, 1.0)
    plt.xlabel('Time (s)')
    plt.ylabel('Binned Spikes')
    plt.title(f'Unit {cid}')
    plt.legend()

    plt.savefig(f"../figures/tejas_poster/rates_time_{sess.name}_{cid}.pdf", bbox_inches='tight', dpi=300)

#%% Plot all cells
NC = robs.shape[-1]
sx = int(np.sqrt(NC))
sy = int(np.ceil(NC / sx))
fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)

for cc in range(NC):
    ax = axs.flatten()[cc]
    # set axis off by turning off tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    r = robs[ind,:,cc]
    ax.imshow(r, aspect='auto', interpolation='none', cmap='gray_r')
    cid = train_data.dsets[0].metadata['cids'][cc]
    ax.set_title(f'Cell {cid}')
    ax_overlay = ax.twinx()
    rbar = np.nanmean(r, 0)/dt
    ax_overlay.plot(rbar, 'r')
    # ax_overlay.plot(np.nanstd(r, 0)/dt, 'g')
    ax_overlay.set_xlim(0, 240)
    ax_overlay.set_ylim(0, np.nanmax(rbar[:120])*2)
    
fig.savefig(f'../figures/tejas_poster/data_rasters_{sess.name}.pdf', bbox_inches='tight', dpi=300)

#%% Mcfarland Analysis


# def mcfarland2016(robs, eyepos, n_bins=10, plot=False):
#     """
#     Partition the variance due to fixational eye movements (NaN-robust).
#     Inputs:
#         robs:   [n_trials, n_bins] binned responses for a single unit (NaNs allowed)
#         eyepos: [n_trials, n_bins, n_lags, 2] eye position (NaNs allowed)
#         n_bins: number of percentile bins for eye-position distance
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     assert (robs.shape[0] == eyepos.shape[0]) and (robs.shape[1] == eyepos.shape[1]), \
#         'robs and eyepos must have the same number of trials and bins'

#     # Basic stats (NaN-aware)
#     total_var = np.nanvar(robs)
#     mean_rate = np.nanmean(robs)

#     # Cross-trial products, masking pairs where either side is NaN at a given time bin
#     trial_outer = robs[:, None, :] * robs[None, :, :]                     # [T, T, B]
#     valid_robs  = np.isfinite(robs)
#     valid_pairs = valid_robs[:, None, :] & valid_robs[None, :, :]         # [T, T, B]
#     trial_outer = np.where(valid_pairs, trial_outer, np.nan)

#     # Upper-triangular mask (i<j) over trial pairs
#     upper_mask_2d = np.triu(np.ones(trial_outer.shape[:2], bool), k=1)
#     # Broadcast to time
#     upper_mask = np.broadcast_to(upper_mask_2d[..., None], trial_outer.shape)
#     trial_outer = np.where(upper_mask, trial_outer, np.nan)

#     # Rate variance: E[ri*rj] - (E[r])^2 over unequal trial pairs (NaN-aware)
#     rate_var = np.nanmean(trial_outer) - mean_rate**2

#     # Pairwise eye-position distance per (i,j,bin), averaging over lags (NaN-aware)
#     # Subtraction will propagate NaNs if either eyepos is NaN; use nanmean over lags.
#     ep_diff = eyepos[:, None, ...] - eyepos[None, :, ...]                 # [T, T, B, L, 2]
#     ep_dist_lag = np.hypot(ep_diff[..., 0], ep_diff[..., 1])              # [T, T, B, L]
#     ep_dist = np.nanmean(ep_dist_lag, axis=-1)                            # [T, T, B]
#     ep_dist = np.where(upper_mask, ep_dist, np.nan)                       # keep i<j only

#     # Flatten valid distances for percentile binning
#     ep_dist_flat = ep_dist[np.isfinite(ep_dist)]
#     if ep_dist_flat.size == 0:
#         # No valid pairs: return NaNs but keep structure
#         out = dict(alpha=np.nan, total_var=total_var, rate_var=rate_var,
#                    em_corrected_var=np.nan, ep_bins=np.array([]),
#                    bin_rate_vars=np.array([]), cum_ep_thresholds=np.array([]),
#                    cum_rate_vars=np.array([]))
#         return out

#     # Percentile thresholds
#     if isinstance(n_bins, int):
#         ep_dist_thresholds = np.percentile(ep_dist_flat, np.linspace(0, 100, n_bins))
#     else:
#         ep_dist_thresholds = n_bins
#     cum_ep_thresholds = ep_dist_thresholds[1:]

#     # Cumulative rate variances (distance < threshold)
#     cum_rate_vars = []
#     for thresh in cum_ep_thresholds:
#         mask = (ep_dist < thresh) & np.isfinite(trial_outer)
#         # mean over selected pair-time elements, NaN-safe
#         m = np.nanmean(np.where(mask, trial_outer, np.nan))
#         cum_rate_vars.append(m - mean_rate**2)
#     cum_rate_vars = np.array(cum_rate_vars)

#     # Non-cumulative binned values (t0 <= d < t1)
#     ep_bins = []
#     ep_cnt = []
#     bin_rate_vars = []
#     for t0, t1 in zip(ep_dist_thresholds[:-1], ep_dist_thresholds[1:]):
#         mask = (ep_dist >= t0) & (ep_dist < t1) & np.isfinite(trial_outer)
#         # median eye distance in bin (NaN-safe)
#         md = np.nanmedian(np.where(mask, ep_dist, np.nan))
#         ep_bins.append(md)
#         m = np.nanmean(np.where(mask, trial_outer, np.nan))
#         bin_rate_vars.append(m - mean_rate**2)
#         ep_cnt.append(mask.sum())
#     ep_bins = np.array(ep_bins)
#     bin_rate_vars = np.array(bin_rate_vars)
#     ep_cnt = np.array(ep_cnt)

#     # Fit exponential decay
#     fit = fit_exponential_decay(ep_bins, bin_rate_vars)
    
#     # USE THE FIT INTERCEPT (Extrapolation to 0), not the first bin
#     em_corrected_var = fit['intercept_at_0']
    
#     # Calculate Alpha
#     # Protect against divide by zero or negative variance artifacts
#     if (np.isfinite(rate_var) and np.isfinite(em_corrected_var) and em_corrected_var > 1e-9):
#         alpha = rate_var / em_corrected_var
#     else:
#         alpha = np.nan

#     # fit_cumulative = fit_exponential_decay(cum_ep_thresholds, cum_rate_vars, weighted=True)

#     if plot:
#         fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
#         axs[0].axhline(cum_rate_vars[0]/dt, color='g', linestyle='--', label='EM-Corrected Rate Variance')
#         axs[0].axhline(rate_var/dt, color='r', linestyle='--', label='Raw Rate Variance')
#         axs[0].plot(ep_bins, bin_rate_vars/dt, 'o-', label='Binned Rate Variance')
#         axs[0].plot(ep_bins, fit['y_fit']/dt, 'r-', label='Exponential Fit')
#         # axs[0].plot(cum_ep_thresholds, fit_cumulative['y_fit']/dt, 'b-', label='Cumulative Exponential Fit')
#         axs[0].plot(cum_ep_thresholds, cum_rate_vars/dt, 'o-', label='Cumulative Rate Variance')
#         axs[0].legend()
#         axs[0].set_xlim(0, np.max(ep_bins))
#         axs[0].set_ylabel('Variance (spikes^2/sec)')
#         axs[0].set_xlabel('Eye Position Distance (degrees)')
#         axs[0].set_title(f'Variance decomposition.\n$\\alpha$ = {alpha:.2f}  Total variance = {total_var:.2f}, $\\tau$ = {fit["tau"]:.2f}')

        
#         axs[1].hist(ep_dist_flat, bins=50)
#         axs[1].set_xlabel('Eye Position Distance (degrees)')
#         axs[1].set_ylabel('Count')
#         axs[1].set_title('Eye Position Distance Distribution')
#         # axs[1].set_xlim(0, np.max(ep_bins))
#     else:
#         fig = None



#     out = {
#         'alpha': alpha,
#         'total_var': total_var,
#         'rate_var': rate_var,
#         'em_corrected_var': em_corrected_var,
#         'ep_bins': ep_bins,
#         'ep_cnt': ep_cnt,
#         'bin_rate_vars': bin_rate_vars,
#         'cum_ep_thresholds': cum_ep_thresholds,
#         'cum_rate_vars': cum_rate_vars,
#         'exp_fit': fit,
#         # 'exp_fit_cumulative': fit_cumulative,
#     }
#     return out, fig

def mcfarland2016(robs, eyepos, n_bins=10, plot=False):
    """
    Partition variance due to FEMs. 
    Uses exponential fit extrapolation to find var at delta_e = 0.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- [Standard Setup: Mean, Var, Covariance] ---
    total_var = np.nanvar(robs)
    mean_rate = np.nanmean(robs)

    # Cross-trial products (masking NaNs)
    trial_outer = robs[:, None, :] * robs[None, :, :]
    valid_robs  = np.isfinite(robs)
    valid_pairs = valid_robs[:, None, :] & valid_robs[None, :, :]
    trial_outer = np.where(valid_pairs, trial_outer, np.nan)

    upper_mask = np.triu(np.ones(trial_outer.shape[:2], bool), k=1)
    upper_mask = np.broadcast_to(upper_mask[..., None], trial_outer.shape)
    trial_outer = np.where(upper_mask, trial_outer, np.nan)

    # Rate variance (PSTH var)
    rate_var = np.nanmean(trial_outer) - mean_rate**2

    # Eye Distances
    ep_diff = eyepos[:, None, ...] - eyepos[None, :, ...]
    ep_dist_lag = np.hypot(ep_diff[..., 0], ep_diff[..., 1])
    ep_dist = np.nanmean(ep_dist_lag, axis=-1)
    ep_dist = np.where(upper_mask, ep_dist, np.nan)

    # --- [Binning Logic] ---
    ep_dist_flat = ep_dist[np.isfinite(ep_dist)]
    if ep_dist_flat.size == 0:
        return {'alpha': np.nan}, None

    if isinstance(n_bins, int):
        ep_dist_thresholds = np.percentile(ep_dist_flat, np.linspace(0, 100, n_bins))
    else:
        ep_dist_thresholds = n_bins

    # Binning
    ep_bins = []
    bin_rate_vars = []
    
    for t0, t1 in zip(ep_dist_thresholds[:-1], ep_dist_thresholds[1:]):
        mask = (ep_dist >= t0) & (ep_dist < t1) & np.isfinite(trial_outer)
        if mask.sum() > 5: # Min pairs count
            ep_bins.append(np.nanmedian(ep_dist[mask]))
            bin_rate_vars.append(np.nanmean(trial_outer[mask]) - mean_rate**2)

    ep_bins = np.array(ep_bins)
    bin_rate_vars = np.array(bin_rate_vars)

    if len(ep_bins) < 3: # Not enough points to fit
        return {'alpha': np.nan}, None

    # --- [Fitting & Alpha Calculation] ---
    # Fit decay to get true variance at 0 offset
    # Define the physical limit of signal variance
    # For a sub-Poisson process, SignalVar < TotalVar
    # For a super-Poisson process, SignalVar < TotalVar - Mean (approx)
    # A safe hard limit is Total Variance.
    var_limit = total_var 
    
    # Fit with the limit
    fit = fit_exponential_decay(ep_bins, bin_rate_vars, total_var_limit=var_limit, weighted=True)
    
    em_corrected_var = fit['intercept_at_0']

    # [cite_start]Calculate Alpha [cite: 480]
    if (np.isfinite(rate_var) and np.isfinite(em_corrected_var) and em_corrected_var > 1e-9):
        alpha = rate_var / em_corrected_var
    else:
        alpha = np.nan

    # --- [Plotting] ---
    if plot:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        axs.plot(ep_bins, bin_rate_vars, 'o', label='Binned Data')
        axs.plot(ep_bins, fit['y_fit'], 'r-', label=f'Fit (tau={fit["tau"]:.2f})')
        # Show extrapolation line
        x_extrap = np.linspace(0, ep_bins[0], 20)
        y_extrap = fit['A']*np.exp(-fit['tau']*x_extrap) + fit['plateau']
        axs.plot(x_extrap, y_extrap, 'r:', alpha=0.5)
        axs.plot(0, em_corrected_var, 'rx', markersize=10, label='Extrapolated Max')
        # axs.axhline(total_var, color='b', linestyle='-', label='Total Variance')
        axs.axhline(rate_var, color='k', linestyle='--', label='PSTH Variance')
        axs.set_title(f'Alpha: {alpha:.2f}')
        axs.legend()
    else:
        fig = None

    out = {
        'alpha': alpha,
        'total_var': total_var,
        'rate_var': rate_var,
        'em_corrected_var': em_corrected_var,
        'ep_bins': ep_bins,
        'bin_rate_vars': bin_rate_vars,
        'exp_fit': fit,
    }
    return out, fig




#%%

for iunit, cid in enumerate(cids_to_plot):
    id = np.where(np.isin(cids, cid))[0]
    x = robs[:,:,id][:,:,0]
    # out, fig = mcfarland2016(x, eyepos[:,:,None,:], n_bins=np.linspace(0, 1, int(ppd)), plot = True)
    out, fig = mcfarland2016(x, eyepos[:,:,None,:], n_bins=20, plot = True)
    fig.savefig(f'../figures/tejas_poster/mcfarland_{sess.name}_{cid}.pdf', bbox_inches='tight', dpi=300)
    


# %%

out['em_corrected_var']/out['total_var']
# %%
x.shape
# %%
win_size = 10

m = np.nanmean(np.nansum(robs[:,:10], 1), 0)
v = np.nanvar(np.nansum(robs[:,:10], 1), 0)

plt.plot(m, v, 'o')
# plot line of unity
plt.plot([0, 5], [0, 5], 'k--')
plt.xlabel('Mean rate (spikes/sec)')
plt.ylabel('Variance (spikes^2/sec)')
# plt.xlim(0, 20)
# plt.ylim(0, 20)



# %%
from numpy.lib.stride_tricks import sliding_window_view

def get_lagged_trajectories(eyepos, history_bins=12, future_bins=0):
    """
    Unfold eye position into trajectories [Trials, Time, Lags, 2].
    
    Parameters:
    -----------
    eyepos : np.ndarray [Trials, Time, 2]
    history_bins : int
        Number of bins BEFORE the current time t to include.
    future_bins : int
        Number of bins AFTER the current time t to include (usually 0 for causality,
        but useful if defining a window centered on t).
        
    Returns:
    --------
    eyepos_lagged : np.ndarray [Trials, Time_Adjusted, Lags, 2]
        The time dimension will be shorter by (history_bins + future_bins).
    """
    # eyepos is [N, T, 2]
    # We want to slide over the T dimension (axis 1)
    # The window size is history_bins + 1 + future_bins
    window_size = history_bins + 1 + future_bins
    
    # sliding_window_view puts the window dim last
    # Input: [N, T, 2] -> Output: [N, T_adj, 2, Window]
    padded_view = sliding_window_view(eyepos, window_size, axis=1)
    
    # Transpose to get [N, T_adj, Window, 2] corresponding to your mcfarland function expcctation
    # Dimensions of padded_view: [N, T_adj, 2, Win]
    # We want: [N, T_adj, Win, 2]
    eyepos_lagged = np.moveaxis(padded_view, -1, -2)
    
    return eyepos_lagged

from numpy.lib.stride_tricks import sliding_window_view

def analyze_fano_over_windows(robs, eyepos, window_sizes_ms, dt=1/240, 
                              pre_window_ms=80, post_window_cutoff_ms=30, n_bins=20):
    """
    Calculates Fano Factor over variable window sizes.
    Strictly handles NaNs to prevent 'negative variance' artifacts.
    """
    results = {'windows': [], 'ff_uncorrected': [], 'ff_corrected': [], 'alpha': []}
    
    # [cite_start]Paper parameters: 80ms before window start, 30ms before window end [cite: 324]
    offset_bins = int(pre_window_ms / (dt * 1000))
    cutoff_bins = int(post_window_cutoff_ms / (dt * 1000)) 
    
    for win_ms in window_sizes_ms:
        win_bins = int(win_ms / (dt * 1000))
        if win_bins < 1: continue
        
        # --- 1. PREPARE SPIKES (Strict Validity) ---
        spikes_view = sliding_window_view(robs, win_bins, axis=1)
        
        # Identify valid windows: MUST be finite for the ENTIRE window
        window_is_valid = np.all(np.isfinite(spikes_view), axis=-1)
        
        # Sum spikes, but strictly mask invalid windows to NaN
        # (Using 'sum' implies 0 for NaNs, so we must manually mask afterwards)
        robs_binned = np.sum(spikes_view, axis=-1)
        robs_binned[~window_is_valid] = np.nan 
        
        # --- 2. PREPARE EYES (Trajectory Matching) ---
        # Trajectory covers: [t_start - 80ms] to [t_end - 30ms]
        # Total duration = (t_end - t_start) + 80 - 30 = win_bins + 50
        traj_len = offset_bins + win_bins - cutoff_bins
        if traj_len < 1: traj_len = 1
        
        eyepos_view = sliding_window_view(eyepos, traj_len, axis=1)
        eyepos_lagged = np.moveaxis(eyepos_view, -1, -2) 
        
        # --- 3. ALIGNMENT ---
        # Valid spike window 'i' starts at time i.
        # Required eye trajectory starts at time i - offset_bins.
        # We cannot have negative indices, so we discard the first 'offset_bins' of spikes.
        
        # Spike indices: [offset_bins ... N]
        # Eye indices:   [0 ... N - offset_bins]
        
        r_slice = robs_binned[:, offset_bins:]
        e_slice = eyepos_lagged
        
        # Crop to common length
        L = min(r_slice.shape[1], e_slice.shape[1])
        r_slice = r_slice[:, :L]
        e_slice = e_slice[:, :L]

        # --- 4. RUN ANALYSIS ---
        # Data Check: Do we have enough valid windows left?
        valid_count = np.sum(np.isfinite(r_slice))
        if valid_count < 100:
            print(f"Skipping {win_ms}ms: Only {valid_count} valid windows found.")
            continue

        # Run McFarland (it handles NaNs internally for covariance)
        out, _ = mcfarland2016(r_slice, e_slice, n_bins=n_bins, plot=True)
        
        if np.isnan(out['alpha']):
            continue

        # --- 5. COMPUTE FANO FACTORS ---
        # Calculate mean ONLY on the valid slice (consistent with total_var)
        mean_count = np.nanmean(r_slice)
        
        # Uncorrected (PSTH)
        # Var_noise = Total_Var - PSTH_Var
        var_noise_raw = out['total_var'] - out['rate_var']
        ff_raw = var_noise_raw / mean_count
        
        # Corrected (FEM)
        # Var_noise = Total_Var - FEM_Driven_Var
        var_noise_cor = out['total_var'] - out['em_corrected_var']
        ff_cor = var_noise_cor / mean_count
        
        results['windows'].append(win_ms)
        results['ff_uncorrected'].append(ff_raw)
        results['ff_corrected'].append(ff_cor)
        results['alpha'].append(out['alpha'])
        
        print(f"Win {win_ms}ms | FF_raw: {ff_raw:.2f} | FF_cor: {ff_cor:.2f} | Alpha: {out['alpha']:.2f} | Total Var: {out['total_var']:.2f} | PSTH Var: {out['rate_var']:.2f} | FEM Var: {out['em_corrected_var']:.2f}")

    return results

#%% Run Fano Factor Analysis

# Define window sizes (e.g., 4ms to 200ms)
window_sizes = [4, 10, 20, 40, 80, 100, 150, 200]

# Pick a unit
cid = cids_to_plot[6] 
id = np.where(np.isin(cids, cid))[0]
robs_unit = robs[:,:,id][:,:,0] # [N, T]

# Run Analysis
ff_results = analyze_fano_over_windows(
    robs_unit, 
    eyepos, 
    window_sizes, 
    dt=1/240, 
    pre_window_ms=50, # Standard from paper
    post_window_cutoff_ms=30   # Standard from paper
)

# Plot Results
plt.figure(figsize=(6, 5))
plt.plot(ff_results['windows'], ff_results['ff_uncorrected'], 'o-', label='Standard (PSTH-based)')
plt.plot(ff_results['windows'], ff_results['ff_corrected'], 'o-', label='FEM-Corrected')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)

plt.xlabel('Count Window (ms)')
plt.ylabel('Fano Factor')
plt.title(f'Fano Factor vs Window Size (Unit {cid})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'../figures/tejas_poster/ff_scaling_{sess.name}_{cid}.pdf', bbox_inches='tight', dpi=300)
# %%
