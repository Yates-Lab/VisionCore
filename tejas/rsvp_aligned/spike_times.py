#%%
import os
from pathlib import Path
from tkinter.constants import TRUE
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
import matplotlib.patheffects as pe 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False
import contextlib

#jake plot the line instead of the points
#declan don't raster until after going into illustrator


def extract_spike_times_per_trial(dataset, dset_idx, sess, cids, trial_inds, trials, fixation, dt=1/240):
    """
    Extract spike times for each trial, aligned with the trial structure used in robs.
    
    This function mirrors the logic from bin_spikes but returns actual spike times
    instead of binned counts. Spike times are organized to match the robs structure.
    
    Parameters
    ----------
    dataset : DictDataset
        The dataset containing trial information
    dset_idx : int
        Index of the dataset to use
    sess : Session object
        Session object containing ks_results
    cids : np.ndarray
        Cluster IDs in the order they appear in robs columns
    trial_inds : np.ndarray
        Trial indices for each data point
    trials : np.ndarray
        Unique trial indices
    fixation : np.ndarray
        Boolean array indicating fixation periods
    dt : float
        Time bin size in seconds (default 1/240)
    
    Returns
    -------
    spike_times_trials : list of lists
        spike_times_trials[itrial][cell_idx] = np.array of spike times in seconds
        Shape matches robs: (NT, NC) where each element is a variable-length array
    trial_time_windows : list of tuples
        trial_time_windows[itrial] = (t_start, t_end) in seconds
    trial_t_bins : list of np.ndarray
        trial_t_bins[itrial] = time bin centers for that trial
    """
    # Get raw spike data
    spike_times = sess.ks_results.spike_times
    spike_clusters = sess.ks_results.spike_clusters
    
    # Get t_bins from dataset
    t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'].numpy()
    
    # Create cluster ID to column index mapping (same as bin_spikes)
    cids = np.asarray(cids)
    n_cids = len(cids)
    cids2inds = np.zeros(np.max(cids) + 1, dtype=int)
    cids2inds[cids] = np.arange(n_cids)
    
    # Ensure spike times are sorted (same as bin_spikes)
    if not np.all(np.diff(spike_times) >= 0):
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]
    
    # Filter spikes to only include clusters in cids
    cids_mask = np.isin(spike_clusters, cids)
    spike_times_filtered = spike_times[cids_mask]
    spike_clusters_filtered = spike_clusters[cids_mask]
    
    # Map cluster IDs to column indices
    spike_inds = cids2inds[spike_clusters_filtered]
    
    NT = len(trials)
    NC = len(cids)
    
    # Initialize output structures
    spike_times_trials = [[np.array([]) for _ in range(NC)] for _ in range(NT)]
    trial_time_windows = [(np.nan, np.nan) for _ in range(NT)]
    trial_t_bins = [np.array([]) for _ in range(NT)]
    
    # Loop over trials
    for itrial in tqdm(range(NT), desc="Extracting spike times"):
        # Find data points for this trial with fixation
        ix = trials[itrial] == trial_inds
        ix = ix & fixation
        if np.sum(ix) == 0:
            continue
        
        # Get time bins for this trial
        trial_t_bins_centers = t_bins[ix]
        trial_psth_inds = psth_inds[ix]
        
        if len(trial_t_bins_centers) == 0:
            continue
        
        # Store trial t_bins
        trial_t_bins[itrial] = trial_t_bins_centers
        
        # Compute time window edges (same logic as dataset_generation.py)
        # t_bins are centers, so edges are: center - dt/2 to center + dt/2
        t_start = trial_t_bins_centers[0] - dt/2
        t_end = trial_t_bins_centers[-1] + dt/2
        trial_time_windows[itrial] = (t_start, t_end)
        
        # Extract spikes in this time window (same logic as bin_spikes)
        i0 = np.searchsorted(spike_times_filtered, t_start)
        i1 = np.searchsorted(spike_times_filtered, t_end)
        
        if i0 >= i1:
            # No spikes in this window
            continue
        
        # Get spikes in this time window
        trial_spike_times = spike_times_filtered[i0:i1]
        trial_spike_inds = spike_inds[i0:i1]
        
        # Organize spikes by cell index
        for cell_idx in range(NC):
            cell_mask = trial_spike_inds == cell_idx
            cell_spike_times = trial_spike_times[cell_mask]
            spike_times_trials[itrial][cell_idx] = cell_spike_times
    
    return spike_times_trials, trial_time_windows, trial_t_bins





if __name__ == "__main__":
    subject = 'Allen'
    date = '2022-03-02'

    dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'
    dataset_configs = load_dataset_configs(dataset_configs_path)

    # date = "2022-03-04"
    # subject = "Allen"
    dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)



    sess = train_dset.dsets[0].metadata['sess']
    # ppd = train_data.dsets[0].metadata['ppd']
    cids = dataset_config['cids']
    print(f"Running on {sess.name}")

    # get fixrsvp inds and make one dataaset object
    inds = torch.concatenate([
            train_dset.get_dataset_inds('fixrsvp'),
            val_dset.get_dataset_inds('fixrsvp')
        ], dim=0)

    dataset = train_dset.shallow_copy()
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

    for itrial in tqdm(range(NT), desc="Aligning robs"):
        # print(f"Trial {itrial}/{NT}")
        ix = trials[itrial] == trial_inds
        ix = ix & fixation
        if np.sum(ix) == 0:
            continue
        
        stim_inds = np.where(ix)[0]
        # stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]


        psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
        fix_dur[itrial] = len(psth_inds)
        robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
        dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
        eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()


    # Extract spike times for all trials (before good_trials filter)
    print("Extracting spike times...")
    spike_times_trials, trial_time_windows, trial_t_bins = extract_spike_times_per_trial(
        dataset, dset_idx, sess, cids, trial_inds, trials, fixation, dt=1/240
    )
    
    # Filter to good trials (same as robs filtering)
    good_trials = fix_dur > 20
    robs = robs[good_trials]
    dfs = dfs[good_trials]
    eyepos = eyepos[good_trials]
    fix_dur = fix_dur[good_trials]
    
    # Filter spike times to match good_trials
    spike_times_trials = [spike_times_trials[i] for i in range(NT) if good_trials[i]]
    trial_time_windows = [trial_time_windows[i] for i in range(NT) if good_trials[i]]
    trial_t_bins = [trial_t_bins[i] for i in range(NT) if good_trials[i]]
    
    print(f"Extracted spike times for {len(spike_times_trials)} trials")
    print(f"Shape: {len(spike_times_trials)} trials x {NC} cells")
    
    # Example: Check spike times for first trial, first cell
    if len(spike_times_trials) > 0 and len(spike_times_trials[0]) > 0:
        print(f"\nExample: Trial 0, Cell 0 has {len(spike_times_trials[0][0])} spikes")
        if len(spike_times_trials[0][0]) > 0:
            print(f"  First spike time: {spike_times_trials[0][0][0]:.4f} s")
            print(f"  Trial time window: {trial_time_windows[0][0]:.4f} to {trial_time_windows[0][1]:.4f} s")


    ind = np.argsort(fix_dur)[::-1]
    plt.subplot(1,2,1)
    plt.imshow(eyepos[ind,:,0])
    plt.xlim(0, 160)
    plt.subplot(1,2,2)
    plt.imshow(np.nanmean(robs,2)[ind])
    plt.xlim(0, 160)

#%%
def plot_spikes_as_lines(ax, spike_x, spike_y, spike_vals=None, height=1.0, color="k", linewidth=0.5, alpha=1.0):
    """
    Plot spikes as vertical line segments with optional alpha variation based on spike values.
    
    Parameters:
    - ax: matplotlib axis
    - spike_x: array of x (time) positions
    - spike_y: array of y (row) positions
    - spike_vals: optional array of spike values for alpha variation
    - height: height of each line segment
    - color: line color
    - linewidth: line width
    - alpha: base alpha (modulated by spike_vals if provided)
    """
    spike_x = np.asarray(spike_x)
    spike_y = np.asarray(spike_y)
    
    if spike_x.size == 0:
        return None
    
    if spike_vals is None:
        # Simple case: all spikes same alpha
        x_lines = np.vstack([spike_x, spike_x, np.full(len(spike_x), np.nan)])
        y_lines = np.vstack([spike_y, spike_y + height, np.full(len(spike_y), np.nan)])
        return ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'), 
                       color=color, linewidth=linewidth, alpha=alpha, rasterized=True)[0]
    
    # Alpha varies by spike value
    spike_vals = np.asarray(spike_vals)
    unique_vals = np.unique(spike_vals)
    vmin, vmax = unique_vals[0], unique_vals[-1]
    handles = []
    
    for val in unique_vals:
        sel = spike_vals == val
        if not np.any(sel):
            continue
        if vmax > vmin:
            norm = (val - vmin) / (vmax - vmin)
            alpha_val = np.clip(0.5 + 0.9 * norm, 0.0, 1.0) * alpha
        else:
            alpha_val = alpha
        
        x_sel, y_sel = spike_x[sel], spike_y[sel]
        x_lines = np.vstack([x_sel, x_sel, np.full(sel.sum(), np.nan)])
        y_lines = np.vstack([y_sel, y_sel + height, np.full(sel.sum(), np.nan)])
        handles.append(
            ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'),
                    color=color, linewidth=linewidth, alpha=alpha_val, rasterized=True)[0]
        )
    return handles

def convert_binned_to_spike_arrays(raster_data, time_bins):
    """Convert binned spike data to (spike_x, spike_y, spike_vals) format."""
    mask = np.isfinite(raster_data) & (raster_data > 0)
    row_idx, col_idx = np.where(mask)
    if row_idx.size == 0:
        return np.array([]), np.array([]), np.array([])
    spike_x = time_bins[col_idx]
    spike_y = row_idx
    spike_vals = raster_data[row_idx, col_idx]
    return spike_x, spike_y, spike_vals

def convert_spike_times_to_arrays(spike_times_list, t_start, t_end):
    """Convert list of spike time arrays to (spike_x, spike_y) format."""
    spike_x_list = []
    spike_y_list = []
    for cell_idx, spike_times in enumerate(spike_times_list):
        if len(spike_times) == 0:
            continue
        mask = (spike_times >= t_start) & (spike_times <= t_end)
        cell_spikes = spike_times[mask]
        if len(cell_spikes) > 0:
            spike_x_list.append(cell_spikes)
            spike_y_list.append(np.full(len(cell_spikes), cell_idx))
    if len(spike_x_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(spike_x_list), np.concatenate(spike_y_list)


start_time_bin = 75
end_time_bin = 100
trial_ind = 11

# Plot binned spikes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Convert to relative time for both plots
t_start_abs = trial_time_windows[trial_ind][0]

# Binned version - convert to relative time
robs_slice = robs[trial_ind, start_time_bin:end_time_bin, :].T  # Transpose: cells x time
time_bins_abs = trial_t_bins[trial_ind][start_time_bin:end_time_bin]
time_bins_slice = time_bins_abs - t_start_abs  # Convert to relative time
spike_x_binned, spike_y_binned, spike_vals_binned = convert_binned_to_spike_arrays(robs_slice, time_bins_slice)
spike_vals_binned = None
plot_spikes_as_lines(ax1, spike_x_binned, spike_y_binned, spike_vals_binned, height=3, linewidth=1.5)
ax1.set_xlabel('Time (s, relative to trial start)')
ax1.set_ylabel('Cell')
ax1.set_title(f'Binned Spikes - Trial {trial_ind}')
ax1.invert_yaxis()
# Set xlim to start at 0 (or first bin) and remove margins
if len(time_bins_slice) > 0:
    ax1.set_xlim(left=0 if start_time_bin == 0 else time_bins_slice[0], right=time_bins_slice[-1])
ax1.margins(x=0)

# Spike times version
spike_times_relative = [np.array(cell_spikes) - t_start_abs 
                        for cell_spikes in spike_times_trials[trial_ind]]
t_start_rel = trial_t_bins[trial_ind][start_time_bin] - t_start_abs
t_end_rel = trial_t_bins[trial_ind][end_time_bin-1] - t_start_abs
spike_x_times, spike_y_times = convert_spike_times_to_arrays(spike_times_relative, t_start_rel, t_end_rel)
plot_spikes_as_lines(ax2, spike_x_times, spike_y_times, spike_vals=None, height=3, linewidth=1.5)
ax2.set_xlabel('Time (s, relative to trial start)')
ax2.set_ylabel('Cell')
ax2.set_title(f'Spike Times - Trial {trial_ind}')
ax2.invert_yaxis()
# Set xlim to start at 0 (or first time) and remove margins
ax2.set_xlim(left=0 if start_time_bin == 0 else t_start_rel, right=t_end_rel)
ax2.margins(x=0)

plt.tight_layout()
plt.show()


# %%
