
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

def fit_exponential_decay(x, y, weighted=False):
    '''
    Fit exponential decay with a fixed plateau:
        y(x) = plateau + (y[0] - plateau) * exp(-(x - x[0]) * tau)

    Plateau is estimated as the average of the two smallest y-values.
    Uses a closed-form solution for tau in log space (no optimization).

    Parameters
    ----------
    x : array-like
        Independent variable (e.g., ep_bins)
    y : array-like
        Dependent variable (e.g., normalized bin_rate_vars)
    weighted : bool, optional
        If True, use weighted least squares (weight by y - plateau). Default False.

    Returns
    -------
    dict
        'tau'     : float, decay constant
        'plateau' : float, estimated plateau (from data)
        'y_fit'   : array, fitted values
        'mse'     : float, mean squared error
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Estimate plateau as mean of the two smallest y-values
    if y.size < 2:
        raise ValueError("Need at least two points to estimate plateau.")
    smallest_idx = np.argpartition(y, 2)[:2]
    plateau = float(np.mean(y[smallest_idx]))

    x0 = x[0]
    y0 = y[0]

    # Difference above plateau should follow an exponential:
    # z(x) = y(x) - plateau = (y0 - plateau) * exp(-tau * (x - x0))
    z = y - plateau
    z0 = z[0]

    # Degenerate case: if initial value is not above plateau, can't fit a decay
    if z0 <= 0:
        tau = np.inf
        y_fit = np.full_like(y, y0)  # just flat at first point
        mse = float(np.mean((y - y_fit) ** 2))
        return {'tau': tau, 'plateau': plateau, 'y_fit': y_fit, 'mse': mse}

    # Normalize so that z_norm[0] = 1 and asymptote is 0
    z_norm = z / z0

    # Closed-form solution in log space:
    # log(z_norm) = -(x - x0) * tau
    # Minimize sum(w * (log(z_norm) + tau * (x - x0))^2)
    # => tau = -sum(w * log(z_norm) * (x - x0)) / sum(w * (x - x0)^2)

    # Use only points where z_norm > 0 and not too close to 1
    valid_mask = (z_norm > 1e-6) & (np.abs(z_norm - 1) > 1e-6)

    if not np.any(valid_mask):
        tau = np.inf
        y_fit = np.full_like(y, y0)
    else:
        log_z = np.log(z_norm[valid_mask])
        x_diff = x[valid_mask] - x0

        if weighted:
            weights = z_norm[valid_mask]  # weight by amplitude above plateau
        else:
            weights = np.ones_like(x_diff)

        denom = np.sum(weights * x_diff**2)
        if denom <= 0:
            tau = np.inf
            y_fit = np.full_like(y, y0)
        else:
            tau = -np.sum(weights * log_z * x_diff) / denom

            # Ensure tau is positive and finite
            if tau <= 0 or not np.isfinite(tau):
                tau = np.inf
                y_fit = np.full_like(y, y0)
            else:
                # Build fitted curve
                z_norm_fit = np.exp(-(x - x0) * tau)
                z_fit = z0 * z_norm_fit
                y_fit = plateau + z_fit

    mse = float(np.mean((y - y_fit) ** 2))

    return {
        'tau': float(tau),
        'plateau': float(plateau),
        'y_fit': y_fit,
        'mse': mse,
    }

import numpy as np
from scipy.optimize import curve_fit

def fit_sigmoid_decay(x, y, weighted=False):
    """
    Fit a decreasing sigmoid with plateau:
        y(x) = plateau + (max_val - plateau) * 1 / (1 + exp(a * (x - b)))

    - plateau is estimated as the mean of the two smallest y-values
    - max_val is the maximum y
    - Data are normalized to [0, 1] between max_val and plateau
    - Parameters a (> 0) and b are fit in normalized space via non-linear least squares

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.
    weighted : bool, optional
        If True, use weights based on y_norm (down-weight near plateau). Default False.

    Returns
    -------
    dict
        'a'       : float, slope parameter (positive; larger = steeper falloff)
        'b'       : float, midpoint (x at which y is halfway between max and plateau)
        'plateau' : float, estimated lower asymptote
        'max_val' : float, estimated upper asymptote
        'y_fit'   : array, fitted values in original units
        'mse'     : float, mean squared error
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Handle NaNs: keep only finite entries
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit_data = y[mask]

    if x_fit.size < 3:
        raise ValueError("Need at least 3 finite points to fit sigmoid.")

    # Estimate plateau as mean of two smallest y-values
    if y_fit_data.size < 2:
        raise ValueError("Need at least two points to estimate plateau.")
    smallest_idx = np.argpartition(y_fit_data, 2)[:2]
    plateau = float(np.mean(y_fit_data[smallest_idx]))

    # Upper asymptote
    max_val = float(np.max(y_fit_data))

    # Guard against degenerate range
    rng = max_val - plateau
    if rng <= 0:
        # Flat or inverted data: just return constant fit
        y_const = np.full_like(y, max_val)
        mse = float(np.mean((y - y_const)**2))
        return {
            'a': 0.0,
            'b': x_fit[0],
            'plateau': plateau,
            'max_val': max_val,
            'y_fit': y_const,
            'mse': mse,
        }

    # Normalize to [0, 1] between max and plateau
    # y_norm = 1 at max, 0 at plateau
    y_norm = (y_fit_data - plateau) / rng
    # Clip for numerical stability
    eps = 1e-6
    y_norm = np.clip(y_norm, eps, 1.0 - eps)

    # Decreasing logistic: 1 / (1 + exp(a * (x - b))) with a > 0
    def logistic(x_, a, b):
        return 1.0 / (1.0 + np.exp(a * (x_ - b)))

    # Initial guesses:
    # midpoint ~ x where y_norm closest to 0.5
    mid_idx = np.argmin(np.abs(y_norm - 0.5))
    b_init = float(x_fit[mid_idx])

    # Rough slope estimate around midpoint
    if 1 <= mid_idx < len(x_fit) - 1:
        dy = y_norm[mid_idx + 1] - y_norm[mid_idx - 1]
        dx = x_fit[mid_idx + 1] - x_fit[mid_idx - 1]
        slope = dy / dx if dx != 0 else 0.0
        # For logistic, derivative at midpoint is -a/4
        a_init = max(1e-3, -4.0 * slope)
    else:
        a_init = 1.0  # fallback

    # Weights for curve_fit
    if weighted:
        # Higher weight away from plateau (where y_norm is larger)
        w = y_norm
        w[w <= 0] = eps
        sigma = 1.0 / np.sqrt(w)
    else:
        sigma = None

    # Fit a >= 0, b free
    try:
        popt, _ = curve_fit(
            logistic,
            x_fit,
            y_norm,
            p0=[a_init, b_init],
            sigma=sigma,
            absolute_sigma=False,
            bounds=([0.0, -np.inf], [np.inf, np.inf]),
            maxfev=10000,
        )
        a, b = popt
    except Exception:
        # Fallback: use initial guesses
        a, b = a_init, b_init

    # Build fitted curve on full x grid (including any NaNs backfilled)
    y_norm_pred = logistic(x, a, b)
    y_pred = plateau + rng * y_norm_pred

    mse = float(np.mean((y - y_pred)**2))

    return {
        'a': float(a),
        'b': float(b),
        'plateau': float(plateau),
        'max_val': float(max_val),
        'y_fit': y_pred,
        'mse': mse,
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


def mcfarland2016(robs, eyepos, n_bins=10, plot=False):
    """
    Partition the variance due to fixational eye movements (NaN-robust).
    Inputs:
        robs:   [n_trials, n_bins] binned responses for a single unit (NaNs allowed)
        eyepos: [n_trials, n_bins, n_lags, 2] eye position (NaNs allowed)
        n_bins: number of percentile bins for eye-position distance
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert (robs.shape[0] == eyepos.shape[0]) and (robs.shape[1] == eyepos.shape[1]), \
        'robs and eyepos must have the same number of trials and bins'

    # Basic stats (NaN-aware)
    total_var = np.nanvar(robs)
    mean_rate = np.nanmean(robs)

    # Cross-trial products, masking pairs where either side is NaN at a given time bin
    trial_outer = robs[:, None, :] * robs[None, :, :]                     # [T, T, B]
    valid_robs  = np.isfinite(robs)
    valid_pairs = valid_robs[:, None, :] & valid_robs[None, :, :]         # [T, T, B]
    trial_outer = np.where(valid_pairs, trial_outer, np.nan)

    # Upper-triangular mask (i<j) over trial pairs
    upper_mask_2d = np.triu(np.ones(trial_outer.shape[:2], bool), k=1)
    # Broadcast to time
    upper_mask = np.broadcast_to(upper_mask_2d[..., None], trial_outer.shape)
    trial_outer = np.where(upper_mask, trial_outer, np.nan)

    # Rate variance: E[ri*rj] - (E[r])^2 over unequal trial pairs (NaN-aware)
    rate_var = np.nanmean(trial_outer) - mean_rate**2

    # Pairwise eye-position distance per (i,j,bin), averaging over lags (NaN-aware)
    # Subtraction will propagate NaNs if either eyepos is NaN; use nanmean over lags.
    ep_diff = eyepos[:, None, ...] - eyepos[None, :, ...]                 # [T, T, B, L, 2]
    ep_dist_lag = np.hypot(ep_diff[..., 0], ep_diff[..., 1])              # [T, T, B, L]
    ep_dist = np.nanmean(ep_dist_lag, axis=-1)                            # [T, T, B]
    ep_dist = np.where(upper_mask, ep_dist, np.nan)                       # keep i<j only

    # Flatten valid distances for percentile binning
    ep_dist_flat = ep_dist[np.isfinite(ep_dist)]
    if ep_dist_flat.size == 0:
        # No valid pairs: return NaNs but keep structure
        out = dict(alpha=np.nan, total_var=total_var, rate_var=rate_var,
                   em_corrected_var=np.nan, ep_bins=np.array([]),
                   bin_rate_vars=np.array([]), cum_ep_thresholds=np.array([]),
                   cum_rate_vars=np.array([]))
        return out

    # Percentile thresholds
    if isinstance(n_bins, int):
        ep_dist_thresholds = np.percentile(ep_dist_flat, np.linspace(0, 100, n_bins))
    else:
        ep_dist_thresholds = n_bins
    cum_ep_thresholds = ep_dist_thresholds[1:]

    # Cumulative rate variances (distance < threshold)
    cum_rate_vars = []
    for thresh in cum_ep_thresholds:
        mask = (ep_dist < thresh) & np.isfinite(trial_outer)
        # mean over selected pair-time elements, NaN-safe
        m = np.nanmean(np.where(mask, trial_outer, np.nan))
        cum_rate_vars.append(m - mean_rate**2)
    cum_rate_vars = np.array(cum_rate_vars)

    # Non-cumulative binned values (t0 <= d < t1)
    ep_bins = []
    ep_cnt = []
    bin_rate_vars = []
    for t0, t1 in zip(ep_dist_thresholds[:-1], ep_dist_thresholds[1:]):
        mask = (ep_dist >= t0) & (ep_dist < t1) & np.isfinite(trial_outer)
        # median eye distance in bin (NaN-safe)
        md = np.nanmedian(np.where(mask, ep_dist, np.nan))
        ep_bins.append(md)
        m = np.nanmean(np.where(mask, trial_outer, np.nan))
        bin_rate_vars.append(m - mean_rate**2)
        ep_cnt.append(mask.sum())
    ep_bins = np.array(ep_bins)
    bin_rate_vars = np.array(bin_rate_vars)
    ep_cnt = np.array(ep_cnt)

    # EM-corrected variance: smallest cumulative bin
    em_corrected_var = cum_rate_vars[0] if cum_rate_vars.size else np.nan

    # Alpha can be undefined/unstable if em_corrected_var <= 0 or NaN
    alpha = rate_var / em_corrected_var if (np.isfinite(rate_var) and np.isfinite(em_corrected_var) and em_corrected_var != 0) else np.nan

    # fit sigmoid decay
    fit = fit_exponential_decay(ep_bins, bin_rate_vars, weighted=True)
    fit_cumulative = fit_exponential_decay(cum_ep_thresholds, cum_rate_vars, weighted=True)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        axs[0].axhline(cum_rate_vars[0]/dt, color='g', linestyle='--', label='EM-Corrected Rate Variance')
        axs[0].axhline(rate_var/dt, color='r', linestyle='--', label='Raw Rate Variance')
        axs[0].plot(ep_bins, bin_rate_vars/dt, 'o-', label='Binned Rate Variance')
        axs[0].plot(ep_bins, fit['y_fit']/dt, 'r-', label='Exponential Fit')
        axs[0].plot(cum_ep_thresholds, fit_cumulative['y_fit']/dt, 'b-', label='Cumulative Exponential Fit')
        axs[0].plot(cum_ep_thresholds, cum_rate_vars/dt, 'o-', label='Cumulative Rate Variance')
        axs[0].legend()
        axs[0].set_xlim(0, np.max(ep_bins))
        axs[0].set_ylabel('Variance (spikes^2/sec)')
        axs[0].set_xlabel('Eye Position Distance (degrees)')
        axs[0].set_title(f'Variance decomposition.\n$\\alpha$ = {alpha:.2f}  Total variance = {total_var:.2f}, $\\tau$ = {fit["tau"]:.2f}')

        
        axs[1].hist(ep_dist_flat, bins=50)
        axs[1].set_xlabel('Eye Position Distance (degrees)')
        axs[1].set_ylabel('Count')
        axs[1].set_title('Eye Position Distance Distribution')
        # axs[1].set_xlim(0, np.max(ep_bins))
    else:
        fig = None



    out = {
        'alpha': alpha,
        'total_var': total_var,
        'rate_var': rate_var,
        'em_corrected_var': em_corrected_var,
        'ep_bins': ep_bins,
        'ep_cnt': ep_cnt,
        'bin_rate_vars': bin_rate_vars,
        'cum_ep_thresholds': cum_ep_thresholds,
        'cum_rate_vars': cum_rate_vars,
        'exp_fit': fit,
        'exp_fit_cumulative': fit_cumulative,
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
