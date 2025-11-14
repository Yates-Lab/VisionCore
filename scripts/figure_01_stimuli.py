## !/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

from DataYatesV1.exp.general import get_trial_protocols
from DataYatesV1.exp.backimage import BackImageTrial

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

#%% Get all datasets
dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)


#%% Helper functions

# ------------------------------------------------------------------------ 
# Plot saccades
# ------------------------------------------------------------------------ 

def plot_saccades_by_protocol(train_data, val_data, stim_type, fig=None):

    stim_indices = torch.concatenate([train_data.get_dataset_inds(stim_type), val_data.get_dataset_inds(stim_type)], dim=0)
    dataset = val_data.shallow_copy()
    dataset.inds = stim_indices

    dset_idx = np.unique(stim_indices[:,0]).item()
    t_bins = train_data.dsets[dset_idx]['t_bins'].numpy()
    trial_inds = train_data.dsets[dset_idx]['trial_inds'].numpy()

    # get saccade times
    saccade_onsets = torch.tensor([s.start_time for s in sess.saccades])
    saccade_offsets = torch.tensor([s.end_time for s in sess.saccades])
    dx = torch.tensor([s.end_x - s.start_x for s in sess.saccades])
    dy = torch.tensor([s.end_y - s.start_y for s in sess.saccades])
    amp = torch.hypot(dx, dy)
    vel = torch.tensor([s.velocity for s in sess.saccades])
    duration = saccade_offsets - saccade_onsets

    valid_epochs = []
    valid_saccades = np.zeros_like(saccade_onsets, dtype=bool)
    trials = np.unique(trial_inds)
    for trial in trials:
        ix = trial_inds == trial
        start = t_bins[ix][0]
        end = t_bins[ix][-1]
        valid_epochs.append((start, end))
        valid_saccades |= (saccade_onsets.numpy() >= start) & (saccade_offsets.numpy() <= end)

    saccade_onsets = saccade_onsets[valid_saccades]
    saccade_offsets = saccade_offsets[valid_saccades]
    dx = dx[valid_saccades]
    dy = dy[valid_saccades]
    amp = amp[valid_saccades]
    vel = vel[valid_saccades]
    duration = duration[valid_saccades]
    fixation_duration = saccade_onsets.numpy()[1:] - saccade_offsets.numpy()[:-1]

    if fig is None:
        fig = plt.figure(figsize=(10, 10))

    plt.subplot(2,2,1)
    plt.scatter(dx, dy, alpha=.1, s=5)
    
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel('Saccade dX (deg)')
    plt.ylabel('Saccade dY (deg)')

    plt.subplot(2,2,2)
    plt.scatter(amp, vel, alpha=.1, s=5)
    plt.ylim(0, 1500)
    plt.xlim(0, 15)
    plt.xlabel('Saccade Amplitude (deg)')
    plt.ylabel('Saccade Velocity (deg/s)')

    plt.subplot(2,2,3)
    plt.hist(fixation_duration, bins=np.linspace(0, 1, 50), density=True, alpha=.5)
    plt.xlabel('Fixation Duration (s)')

    plt.subplot(2,2,4)
    plt.hist(duration.numpy(), bins=np.linspace(0, .25, 50), density=True, alpha=.5)
    plt.xlabel('Saccade Duration (s)')

    plt.tight_layout()

# ------------------------------------------------------------------------ 
# Helper functions for fixation detection
# ------------------------------------------------------------------------ 
import numpy as np

def runs_from_bool(b):
    b = np.asarray(b, bool)
    x = b.astype(int)
    d = np.diff(x)
    starts = np.flatnonzero(d == 1) + 1
    ends   = np.flatnonzero(d == -1) + 1
    if b[0]:  starts = np.r_[0, starts]
    if b[-1]: ends   = np.r_[ends, b.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]  # [start, end)

def medfilt1d(x, k=3):
    if k <= 1: return x.copy()
    if k % 2 == 0: k += 1
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    n = x.size
    s = xp.strides[0]
    windows = np.lib.stride_tricks.as_strided(xp, shape=(n, k), strides=(s, s))
    return np.median(windows, axis=1)

def merge_short_gaps(mask, fs, max_gap_sec):
    if max_gap_sec <= 0: return mask
    max_gap = max(1, int(round(max_gap_sec * fs)))
    out = mask.copy()
    runs = runs_from_bool(out)
    for (s1, e1), (s2, e2) in zip(runs[:-1], runs[1:]):
        if 0 < (s2 - e1) <= max_gap:
            out[e1:s2] = True
    return out

def apply_min_duration(mask, fs, min_sec):
    if min_sec <= 0: return mask
    out = mask.copy()
    min_len = max(1, int(round(min_sec * fs)))
    for s, e in runs_from_bool(mask):
        if (e - s) < min_len:
            out[s:e] = False
    return out

def wrap_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    # Handle both scalar and array cases
    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0):
        if a == -180.0:
            a = 180.0
    else:
        a[a == -180.0] = 180.0
    return a

def detect_fixations_and_saccades(
    v, pos, t, fs,
    v_thresh=10.0,
    min_fix_dur_sec=0.06,
    max_fix_internal_gap_sec=0.012,
    min_sacc_gap_merge_sec=0.008,
    median_k=3,
    valid_mask=None
):
    v   = np.asarray(v, float)
    pos = np.asarray(pos, float)     # (T,2)
    t   = np.asarray(t, float)
    if valid_mask is None:
        valid_mask = np.isfinite(v) & np.isfinite(pos).all(axis=1)
    else:
        valid_mask = np.asarray(valid_mask, bool) & np.isfinite(v) & np.isfinite(pos).all(axis=1)

    v_abs = np.abs(v)
    v_f   = medfilt1d(v_abs, k=median_k) if median_k and median_k > 1 else v_abs

    fix = (v_f < v_thresh) & valid_mask
    fix = merge_short_gaps(fix, fs, max_fix_internal_gap_sec)
    fix = apply_min_duration(fix, fs, min_fix_dur_sec)
    fix = merge_short_gaps(fix, fs, min_sacc_gap_merge_sec)

    fix_intervals = runs_from_bool(fix)
    sac_mask = valid_mask & (~fix)
    sac_intervals = runs_from_bool(sac_mask)

    labels = np.zeros(v.size, np.int8)  # 0=invalid
    labels[sac_mask] = 2
    labels[fix] = 1

    # saccade table: [start, end, dur_s, amp_deg, dir_deg, peak_v]
    sac_rows = []
    for s, e in sac_intervals:
        dur = t[e-1] - t[s]
        dx, dy = pos[e-1] - pos[s]
        amp = np.hypot(dx, dy)
        ang = wrap_deg(np.degrees(np.arctan2(dy, dx)))
        peak_v = float(np.max(v_f[s:e])) if e > s else 0.0
        sac_rows.append([s, e, dur, amp, ang, peak_v])
    sac_table = np.array(sac_rows, float) if sac_rows else np.zeros((0,6), float)

    # fixation table with neighboring saccades:
    # [start, end, dur_s, prev_sac_idx, next_sac_idx,
    #  prev_amp, prev_dir, prev_dur, next_amp, next_dir, next_dur]
    fix_rows = []
    for s, e in fix_intervals:
        dur = t[e-1] - t[s]
        prev_idx = -1
        next_idx = -1
        for k, (ss, ee) in enumerate(sac_intervals):
            if ee <= s: prev_idx = k
            if ss >= e and next_idx < 0:
                next_idx = k
                break

        def sv(idx):
            if idx < 0 or idx >= len(sac_intervals): return (np.nan, np.nan, np.nan)
            row = sac_table[idx]  # [s,e,dur,amp,dir,peak]
            return (row[3], row[4], row[2])

        p_amp, p_dir, p_dur = sv(prev_idx)
        n_amp, n_dir, n_dur = sv(next_idx)
        fix_rows.append([s, e, dur, prev_idx, next_idx, p_amp, p_dir, p_dur, n_amp, n_dir, n_dur])
    fix_table = np.array(fix_rows, float) if fix_rows else np.zeros((0,11), float)

    return {
        "labels": labels,               # 0 invalid, 1 fix, 2 sacc
        "fix_intervals": fix_intervals, # list of (start, end)
        "sac_intervals": sac_intervals,
        "fix_table": fix_table,
        "sac_table": sac_table,
        "fix_mask": fix,
        "valid_mask": valid_mask,
        "fix_cols": ["start","end","dur_s","prev_sac_idx","next_sac_idx",
                     "prev_amp_deg","prev_dir_deg","prev_dur_s",
                     "next_amp_deg","next_dir_deg","next_dur_s"],
        "sac_cols": ["start","end","dur_s","amp_deg","dir_deg","peak_v"]
    }

def fixations_min_samples(fix_table, min_samples):
    if fix_table.size == 0: return np.array([], int)
    starts = fix_table[:,0].astype(int)
    ends   = fix_table[:,1].astype(int)
    return np.flatnonzero((ends - starts) >= int(min_samples))

def filter_physiological_events(out, max_vel=1200.0, max_amp=15.0, max_dur=0.1):
    """
    Filter out unphysiological saccades and keep only fixations preceded and followed by valid saccades.

    Parameters:
    -----------
    out : dict
        Output from detect_fixations_and_saccades
    max_vel : float
        Maximum physiological saccade velocity (deg/s)
    max_amp : float
        Maximum physiological saccade amplitude (deg)
    max_dur : float
        Maximum physiological saccade duration (s)

    Returns:
    --------
    dict : Filtered output with same structure as input
    """
    sac_table = out['sac_table']
    fix_table = out['fix_table']

    if sac_table.size == 0:
        return out

    # sac_table columns: [start, end, dur_s, amp_deg, dir_deg, peak_v]
    # Filter valid saccades
    valid_sac = (
        (sac_table[:, 5] <= max_vel) &  # peak_v <= max_vel
        (sac_table[:, 3] <= max_amp) &  # amp_deg <= max_amp
        (sac_table[:, 2] <= max_dur)    # dur_s <= max_dur
    )

    valid_sac_indices = np.flatnonzero(valid_sac)

    # Filter saccade table and intervals
    filtered_sac_table = sac_table[valid_sac]
    filtered_sac_intervals = [out['sac_intervals'][i] for i in valid_sac_indices]

    # fix_table columns: [start, end, dur_s, prev_sac_idx, next_sac_idx, ...]
    # Keep only fixations with valid prev AND next saccades
    if fix_table.size > 0:
        prev_valid = np.isin(fix_table[:, 3].astype(int), valid_sac_indices)
        next_valid = np.isin(fix_table[:, 4].astype(int), valid_sac_indices)
        valid_fix = prev_valid & next_valid

        valid_fix_indices = np.flatnonzero(valid_fix)
        filtered_fix_table = fix_table[valid_fix]
        filtered_fix_intervals = [out['fix_intervals'][i] for i in valid_fix_indices]

        # Update saccade indices in fix_table to reflect new saccade numbering
        old_to_new_sac_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_sac_indices)}
        for i in range(len(filtered_fix_table)):
            prev_idx = int(filtered_fix_table[i, 3])
            next_idx = int(filtered_fix_table[i, 4])
            filtered_fix_table[i, 3] = old_to_new_sac_idx.get(prev_idx, -1)
            filtered_fix_table[i, 4] = old_to_new_sac_idx.get(next_idx, -1)
    else:
        filtered_fix_table = fix_table
        filtered_fix_intervals = []

    # Update labels array
    labels = np.zeros_like(out['labels'])
    for s, e in filtered_fix_intervals:
        labels[s:e] = 1
    for s, e in filtered_sac_intervals:
        labels[s:e] = 2

    # Create filtered output
    filtered_out = {
        "labels": labels,
        "fix_intervals": filtered_fix_intervals,
        "sac_intervals": filtered_sac_intervals,
        "fix_table": filtered_fix_table,
        "sac_table": filtered_sac_table,
        "fix_mask": labels == 1,
        "valid_mask": out['valid_mask'],
        "fix_cols": out['fix_cols'],
        "sac_cols": out['sac_cols']
    }

    return filtered_out

# ------------------------------------------------------------------------ 
# Extract trial and plot
# ------------------------------------------------------------------------ 

def get_fixation_data(dataset, dset_idx, out_filtered, ifix, include_saccades=True):
    fix_s, fix_e = out_filtered['fix_intervals'][ifix]

    if include_saccades:
        # Get previous and next saccade indices from the fixation table
        prev_sac_idx = int(out_filtered['fix_table'][ifix, 3])
        next_sac_idx = int(out_filtered['fix_table'][ifix, 4])
        # Get saccade intervals
        prev_sac_s, prev_sac_e = out_filtered['sac_intervals'][prev_sac_idx]
        next_sac_s, next_sac_e = out_filtered['sac_intervals'][next_sac_idx]
        # Get the full interval from start of prev saccade to end of next saccade
        full_s = prev_sac_s
        full_e = next_sac_e
    else:
        full_s = fix_s
        full_e = fix_e

    # Extract data
    stim = dataset.dsets[dset_idx]['stim'][full_s:full_e]
    robs = dataset.dsets[dset_idx]['robs'][full_s:full_e]
    dfs = dataset.dsets[dset_idx]['dfs'][full_s:full_e]
    eyepos = dataset.dsets[dset_idx]['eyepos'][full_s:full_e]
    dpi_pix = dataset.dsets[dset_idx]['dpi_pix'][full_s:full_e]
    dpi_pix_fix = dataset.dsets[dset_idx]['dpi_pix'][fix_s:fix_e]
    t_bins = dataset.dsets[dset_idx]['t_bins'][full_s:full_e].numpy()
    t_bins = t_bins - t_bins[0]
    trial_id = dataset.dsets[dset_idx].covariates['trial_inds'][full_s:full_e].unique().item()
    trial = BackImageTrial(sess.exp['D'][trial_id], sess.exp['S'])

    return stim, robs, dfs, eyepos, dpi_pix, dpi_pix_fix, t_bins, trial

def plot_fixation_trial(dataset, dset_idx, out_filtered, ifix, sess):

    # Extract data
    stim, robs, dfs, eyepos, dpi_pix, dpi_pix_fix, t_bins, trial = get_fixation_data(dataset, dset_idx, out_filtered, ifix)
    
    # Plot with markers showing the boundaries
    fig = plt.figure(figsize=(12, 12))
    layout = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=.2)
    # plot image with dpi_pix overlaid
    ax1 = fig.add_subplot(layout[0])
    I = trial.get_image()**.8
    ax1.imshow(I, cmap='gray')
    ax1.plot(dpi_pix[:,1], dpi_pix[:,0], 'r-')
    ax1.plot(dpi_pix_fix[:,1], dpi_pix_fix[:,0], '-o')

    # plot movie sequence being viewed
    ax2 = fig.add_subplot(layout[1])
    grid = make_grid(stim, nrow=stim.shape[0]//3+1, normalize=True, scale_each=False, padding=2, pad_value=1)
    ax2.imshow(grid.detach().cpu().permute(1, 2, 0).numpy(), aspect='auto', interpolation='none')
    ax2.set_axis_off()

    # plot eyepos
    ax3 = fig.add_subplot(layout[2])
    ax3.plot(t_bins, eyepos)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Eye position (deg)')
    ax3.set_title(f'Fixation {ifix} with surrounding saccades')
    ax3.set_xlim(0, t_bins[-1])
    # plot spikes
    ax4 = fig.add_subplot(layout[3])
    ax4.imshow(robs.T, aspect='auto', cmap='gray_r', interpolation='none', extent=[0, t_bins[-1], 0, robs.shape[1]])
    # ax4.imshow(torch.concat([robs, dfs], 1).T, aspect='auto', cmap='gray_r', interpolation='none', extent=[0, t_bins[-1], 0, robs.shape[1]*2])
    axoverlay = ax4.twinx()
    rbar = robs.mean(1)
    axoverlay.plot(t_bins, rbar, 'r-')
    axoverlay.plot(t_bins, dfs.mean(1)*rbar.max().item(), 'g-')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Neurons')

from scipy import signal

def compute_fixation_power_spectra(rbar, fs, min_samples=128, nperseg=256, noverlap=None):
    """
    Compute power spectra for each fixation using Welch's method.

    Parameters:
    -----------
    rbar : np.ndarray
        Array of shape (n_fixations, max_duration) with NaNs for missing data
    fs : float
        Sampling frequency (Hz)
    min_samples : int
        Minimum number of samples required to compute spectrum
    nperseg : int
        Length of each segment for Welch's method
    noverlap : int or None
        Number of points to overlap between segments (default: nperseg // 2)

    Returns:
    --------
    freqs : np.ndarray
        Frequency bins
    psd_mean : np.ndarray
        Mean power spectral density across all valid fixations
    psd_all : list
        List of individual PSDs for each valid fixation
    valid_indices : np.ndarray
        Indices of fixations that were long enough to analyze
    """
    if noverlap is None:
        noverlap = nperseg // 2

    psd_all = []
    valid_indices = []
    freqs = None

    for ifix in range(rbar.shape[0]):
        # Get valid (non-NaN) samples for this fixation
        row = rbar[ifix, :]
        valid_mask = ~np.isnan(row)
        valid_samples = row[valid_mask]

        # Skip if too short
        if len(valid_samples) < min_samples:
            continue

        # Demean the signal
        valid_samples = valid_samples - np.mean(valid_samples)

        # Adjust nperseg and noverlap for short segments
        nperseg_adj = min(nperseg, len(valid_samples))
        noverlap_adj = min(noverlap, nperseg_adj // 2)  # Ensure noverlap < nperseg

        # Compute power spectral density using Welch's method
        f, psd = signal.welch(
            valid_samples,
            fs=fs,
            nperseg=nperseg_adj,
            noverlap=noverlap_adj,
            scaling='density',
            window='hann'
        )

        psd_all.append(psd)
        valid_indices.append(ifix)

        if freqs is None:
            freqs = f

    valid_indices = np.array(valid_indices)

    # Average across all valid fixations
    if len(psd_all) > 0:
        # Stack and compute mean (handling different lengths if necessary)
        min_len = min(len(p) for p in psd_all)
        psd_all = [p[:min_len] for p in psd_all]  # Truncate all to same length
        psd_stack = np.array(psd_all)
        psd_mean = np.mean(psd_stack, axis=0)
        freqs = freqs[:min_len]
    else:
        psd_mean = np.array([])
        freqs = np.array([])

    return freqs, psd_mean, psd_all, valid_indices

#%% Load a dataset 
dataset_idx = 2
train_data, val_data, dataset_config = prepare_data(dataset_configs[dataset_idx])
sess = train_data.dsets[0].metadata['sess']


#%%
fig = plt.figure(figsize=(10, 10))
for stim_type in ['backimage', 'gaborium']:
    plot_saccades_by_protocol(train_data, val_data, stim_type, fig=fig)

#%% Detect fixtions
stim_type = 'backimage'
stim_indices = torch.concatenate([train_data.get_dataset_inds(stim_type), val_data.get_dataset_inds(stim_type)], dim=0)
dataset = val_data.shallow_copy()
dataset.inds = stim_indices
dset_idx = np.unique(stim_indices[:,0]).item()

dt = np.diff(dataset.dsets[dset_idx]['t_bins'].numpy()).min().item()
eyepos = dataset.dsets[dset_idx]['eyepos']
dpi_valid = dataset.dsets[dset_idx]['dpi_valid'].numpy() > 0
vxy = np.gradient(eyepos.numpy(), axis=0) / dt
v = np.hypot(vxy[:,0], vxy[:,1])

t_bins = dataset.dsets[dset_idx]['t_bins'].numpy()
fs = int(np.round(1/dt))

out = detect_fixations_and_saccades(
    v=v, pos=eyepos, t=t_bins, fs=fs,
    v_thresh=10.0,
    min_fix_dur_sec=0.06,
    max_fix_internal_gap_sec=0.012,
    min_sacc_gap_merge_sec=0.008,
    median_k=3,
    valid_mask=dpi_valid
)

# Filter out unphysiological saccades
out_filtered = filter_physiological_events(
    out,
    max_vel=1200.0,   # deg/s
    max_amp=15.0,     # deg
    max_dur=0.1       # sec
)

print(f"Original: {len(out['sac_intervals'])} saccades, {len(out['fix_intervals'])} fixations")
print(f"Filtered: {len(out_filtered['sac_intervals'])} saccades, {len(out_filtered['fix_intervals'])} fixations")

# %%
# Plot before and after filtering
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before filtering
axes[0].plot(out['sac_table'][:,3], out['sac_table'][:,5], '.', alpha=.1)
axes[0].axhline(1200, color='r', linestyle='--', label='max_vel=1200')
axes[0].axvline(15, color='r', linestyle='--', label='max_amp=15')
axes[0].set_xlim(0, 20)
axes[0].set_ylim(0, 2000)
axes[0].set_xlabel('Saccade Amplitude (deg)')
axes[0].set_ylabel('Peak Velocity (deg/s)')
axes[0].set_title(f'Before Filtering (n={len(out["sac_intervals"])})')
axes[0].legend()

# After filtering
axes[1].plot(out_filtered['sac_table'][:,3], out_filtered['sac_table'][:,5], '.', alpha=.1)
axes[1].axhline(1200, color='r', linestyle='--', label='max_vel=1200')
axes[1].axvline(15, color='r', linestyle='--', label='max_amp=15')
axes[1].set_xlim(0, 20)
axes[1].set_ylim(0, 2000)
axes[1].set_xlabel('Saccade Amplitude (deg)')
axes[1].set_ylabel('Peak Velocity (deg/s)')
axes[1].set_title(f'After Filtering (n={len(out_filtered["sac_intervals"])})')
axes[1].legend()

plt.tight_layout()
plt.show()
# %%
out_filtered.keys()
# %%
num_fix = len(out_filtered['fix_intervals'])

ifix = 1
s, e = out_filtered['fix_intervals'][ifix]

stim = dataset.dsets[dset_idx]['stim'][s:e]
eyepos = dataset.dsets[dset_idx]['eyepos'][s:e]

plt.plot(eyepos)
# %%
ifix +=1
stim, robs, dfs, eyepos, dpi_pix, dpi_pix_fix, t_bins, trial = get_fixation_data(dataset, dset_idx, out_filtered, ifix)
dfs.shape

#%%
num_fix = len(out_filtered['fix_intervals'])
fix_dur = np.array([e-s for s, e in out_filtered['fix_intervals']])
zombies = np.where(fix_dur*dt > 1.0)[0]

for ifix in zombies:
    try:
        plot_fixation_trial(dataset, dset_idx, out_filtered, ifix, sess)
        plt.show()
    except Exception as e:
        print(f"Failed to plot fixation {ifix}: {e}")
# %%
ifix = np.where(fix_dur*dt > .35)[0][20]
plot_fixation_trial(dataset, dset_idx, out_filtered, ifix, sess)
ifix = 15
plot_fixation_trial(dataset, dset_idx, out_filtered, ifix, sess)


#%%
stim, robs, dfs, eyepos, dpi_pix, dpi_pix_fix, t_bins, trial = get_fixation_data(dataset, dset_idx, out_filtered, ifix, include_saccades=False)

x = eyepos[:,0]
f = np.fft.rfft(x-x.mean())
F = np.fft.rfftfreq(len(x), dt)

x2 = eyepos[:,1]
f2 = np.fft.rfft(x2-x2.mean())
F2 = np.fft.rfftfreq(len(x2), dt)
# kill DC. plot
plt.figure()
plt.plot(F[1:], np.abs(f[1:]))
plt.plot(F2[1:], np.abs(f2[1:]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# %% Plot PdfPages of representative trials
from matplotlib.backends.backend_pdf import PdfPages

def create_fixation_pdf(dataset, dset_idx, out_filtered, sess, output_dir='figures'):
    """
    Create a PDF with plots of the 100 longest, 100 median, and 100 shortest fixations.
    Plots 3 fixations per page.

    Parameters:
    -----------
    dataset : Dataset object
    dset_idx : int
        Dataset index
    out_filtered : dict
        Filtered output from detect_fixations_and_saccades
    sess : Session object
    output_dir : str
        Directory to save the PDF
    """
    from pathlib import Path

    # Calculate fixation durations
    num_fix = len(out_filtered['fix_intervals'])
    fix_dur = np.array([e-s for s, e in out_filtered['fix_intervals']])

    # Get indices for longest, median, and shortest fixations
    sorted_indices = np.argsort(fix_dur)

    # Select 100 longest, 100 median, 100 shortest
    n_per_group = min(100, num_fix // 3)  # Ensure we don't exceed available fixations

    shortest_indices = sorted_indices[:n_per_group]
    median_start = (num_fix - n_per_group) // 2
    median_indices = sorted_indices[median_start:median_start + n_per_group]
    longest_indices = sorted_indices[-n_per_group:]

    # Combine all indices
    all_indices = np.concatenate([longest_indices, median_indices, shortest_indices])
    labels = (['longest'] * len(longest_indices) +
              ['median'] * len(median_indices) +
              ['shortest'] * len(shortest_indices))

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create PDF filename
    session_name = f"{sess.name}"
    pdf_filename = output_path / f"backimage_fixations_{session_name}.pdf"

    print(f"Creating PDF with {len(all_indices)} fixations...")
    print(f"  - {len(longest_indices)} longest fixations")
    print(f"  - {len(median_indices)} median fixations")
    print(f"  - {len(shortest_indices)} shortest fixations")
    print(f"Saving to: {pdf_filename}")

    with PdfPages(pdf_filename) as pdf:
        for page_idx in range(0, len(all_indices), 3):
            # Create figure with 3 subplots (one per fixation)
            fig = plt.figure(figsize=(18, 24))

            # Plot up to 3 fixations on this page
            for subplot_idx in range(3):
                fix_idx_in_list = page_idx + subplot_idx
                if fix_idx_in_list >= len(all_indices):
                    break

                ifix = all_indices[fix_idx_in_list]
                label = labels[fix_idx_in_list]

                try:
                    # Get fixation data
                    fix_s, fix_e = out_filtered['fix_intervals'][ifix]

                    # Use a fixed 200ms window centered on the fixation
                    window_dur = 0.2  # 200ms
                    fs = int(np.round(1/dt))
                    window_samples = int(window_dur * fs)

                    # Center the window on the fixation
                    fix_center = (fix_s + fix_e) // 2
                    full_s = max(0, fix_center - window_samples // 2)
                    full_e = min(len(dataset.dsets[dset_idx]['stim']), fix_center + window_samples // 2)

                    # Extract data
                    stim = dataset.dsets[dset_idx]['stim'][full_s:full_e]
                    robs = dataset.dsets[dset_idx]['robs'][full_s:full_e]
                    eyepos = dataset.dsets[dset_idx]['eyepos'][full_s:full_e]
                    dpi_pix = dataset.dsets[dset_idx]['dpi_pix'][full_s:full_e]
                    dpi_pix_fix = dataset.dsets[dset_idx]['dpi_pix'][fix_s:fix_e]
                    t_bins = dataset.dsets[dset_idx]['t_bins'][full_s:full_e].numpy()
                    t_bins = t_bins - t_bins[0]

                    trial_id = dataset.dsets[dset_idx].covariates['trial_inds'][full_s:full_e].unique().item()
                    trial = BackImageTrial(sess.exp['D'][trial_id], sess.exp['S'])

                    # Create subplot layout for this fixation
                    base_row = subplot_idx * 4
                    layout = fig.add_gridspec(12, 1, height_ratios=[3,1,1,1]*3, hspace=.3)

                    # Plot image with eye position overlay
                    ax1 = fig.add_subplot(layout[base_row])
                    I = trial.get_image()**.8
                    ax1.imshow(I, cmap='gray')
                    ax1.plot(dpi_pix[:,1], dpi_pix[:,0], 'r-', linewidth=1)
                    ax1.plot(dpi_pix_fix[:,1], dpi_pix_fix[:,0], '-o', markersize=2)
                    ax1.set_title(f'Fixation {ifix} ({label}, dur={fix_dur[ifix]*dt:.3f}s)', fontsize=10)
                    ax1.set_axis_off()

                    # Plot movie sequence
                    ax2 = fig.add_subplot(layout[base_row + 1])
                    grid = make_grid(stim, nrow=stim.shape[0]//3+1, normalize=True,
                                   scale_each=False, padding=2, pad_value=1)
                    ax2.imshow(grid.detach().cpu().permute(1, 2, 0).numpy(),
                             aspect='auto', interpolation='none')
                    ax2.set_axis_off()

                    # Plot eye position
                    ax3 = fig.add_subplot(layout[base_row + 2])
                    ax3.plot(t_bins, eyepos)
                    ax3.axvline(t_bins[fix_s - full_s], color='g', linestyle='--', linewidth=1)
                    ax3.axvline(t_bins[fix_e - full_s], color='g', linestyle='--', linewidth=1)
                    ax3.set_ylabel('Eye pos (deg)', fontsize=8)
                    ax3.set_xlim(0, t_bins[-1])
                    ax3.tick_params(labelsize=8)

                    # Plot spikes
                    ax4 = fig.add_subplot(layout[base_row + 3])
                    ax4.imshow(robs.T, aspect='auto', cmap='gray_r', interpolation='none',
                             extent=[0, t_bins[-1], 0, robs.shape[1]])
                    axoverlay = ax4.twinx()
                    axoverlay.plot(t_bins, robs.mean(1), 'r-', linewidth=1)
                    axoverlay.tick_params(labelsize=8)
                    ax4.set_xlabel('Time (s)', fontsize=8)
                    ax4.set_ylabel('Neurons', fontsize=8)
                    ax4.tick_params(labelsize=8)

                except Exception as e:
                    print(f"Failed to plot fixation {ifix}: {e}")
                    continue

            # Save this page to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Progress update
            if (page_idx // 3) % 10 == 0:
                print(f"  Processed {min(page_idx + 3, len(all_indices))}/{len(all_indices)} fixations...")

    print(f"PDF saved successfully to: {pdf_filename}")
    return pdf_filename

# Generate the PDF
pdf_file = create_fixation_pdf(dataset, dset_idx, out_filtered, sess)
# %%
fix_dur = np.array([e-s for s, e in out_filtered['fix_intervals']])
rbar = np.nan*np.zeros((num_fix, np.max(fix_dur).item()))
eyepos = np.nan*np.zeros((num_fix, np.max(fix_dur).item(), 2))

for ifix in range(len(out_filtered['fix_intervals'])):
    # Get fixation data
    fix_s, fix_e = out_filtered['fix_intervals'][ifix]

    # Extract data
    stim = dataset.dsets[dset_idx]['stim'][fix_s:fix_e]
    robs = dataset.dsets[dset_idx]['robs'][fix_s:fix_e]
    eyepos_ = dataset.dsets[dset_idx]['eyepos'][fix_s:fix_e]

    rbar[ifix, :robs.shape[0]] = robs.mean(1)
    eyepos[ifix, :eyepos_.shape[0]] = eyepos_

# %% loop over, window, compute the power and average

# Compute power spectrum
# Use min_samples=nperseg to ensure all fixations produce the same frequency resolution
freqs, psd_mean, psd_all, valid_indices = compute_fixation_power_spectra(
    eyepos[:,:,0],
    fs=fs,
    min_samples=128,
    nperseg=128,
    noverlap=128//2
)

print(f"Computed power spectra for {len(valid_indices)}/{rbar.shape[0]} fixations")
print(f"Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz")

# Plot average power spectrum
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogy(freqs, psd_mean)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (power/Hz)')
plt.title(f'Average Power Spectrum (n={len(valid_indices)} fixations)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 120)  # Show up to Nyquist frequency (120 Hz at 240 Hz sampling)

# Plot individual spectra (semi-transparent)
plt.subplot(2, 1, 2)
for psd in psd_all[:100]:  # Plot first 100 to avoid clutter
    plt.semilogy(freqs[:len(psd)], psd, alpha=0.1, color='gray')
plt.semilogy(freqs, psd_mean, 'r-', linewidth=2, label='Mean')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (power/Hz)')
plt.title('Individual Power Spectra (first 100)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 120)  # Show up to Nyquist frequency
plt.legend()
plt.tight_layout()
plt.show()

# %%

plt.figure()
for i in range(eyepos.shape[0]):
    x = eyepos[i, :-10, 0]
    valid = np.isfinite(x)
    x = x[valid]
    x = x-x.mean()
    f = np.fft.rfft(x)
    F = np.fft.rfftfreq(len(x), dt)
    plt.subplot(1,2,1)
    plt.plot(x)
    plt.subplot(1,2,2)
    plt.plot(F, np.abs(f), alpha=0.1, color='gray')

# plt.show()

# %%
