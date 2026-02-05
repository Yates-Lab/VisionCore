#%%

from models.data import DictDataset
from VisionCore.dual_window import DualWindowAnalysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm
from pathlib import Path
#
dset = DictDataset.load('/mnt/ssd/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/gaborium.dset')
#rsvp_dset = DictDataset.load('/mnt/ssd/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/fixrsvp.dset')

#dset = DictDataset.load('/mnt/ssd/YatesMarmoV1/processed/Allen_2022-04-13/datasets/gaborium.dset')
n_lags = 20
valid_eyepos_radius = 7.5  # degrees
snr_threshold = 5.0

print(dset)
#%%
if 'region' in dset.metadata:
    regions = dset.metadata['region']

    region = 'V1'
    region_mask = regions == region
    print(f'Selecting units from region: {region} ({np.sum(region_mask)}/{len(regions)})')

    dset['robs'] = dset['robs'][:, region_mask]
else:
    print('No region information found in metadata; using all units.')

#%%
# Computation parameters
batch_size = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
# Preprocess stimulus and create valid mask

print('PREPROCESSING')


def normalize_stimulus(stim_tensor):
    """
    Normalizes a stimulus tensor to have zero mean and unit standard deviation.
    
    Args:
        stim_tensor (torch.Tensor): The input stimulus tensor.
        
    Returns:
        torch.Tensor: The normalized stimulus tensor.
    """
    stim_float = stim_tensor.float()
    stim_mean = stim_float.mean()
    stim_std = stim_float.std()
    stim_float -= stim_mean
    stim_float /= stim_std
    return stim_float

def create_valid_eyepos_mask(eyepos, dpi_valid, valid_radius):
    """
    Creates a binary mask for valid eye positions within a specified radius.
    
    Args:
        eyepos (torch.Tensor): Tensor of eye positions (x, y).
        dpi_valid (torch.Tensor): Tensor of DPI validity flags.
        valid_radius (float): The radius within which eye positions are considered valid.
        
    Returns:
        torch.Tensor: A binary mask of valid time points.
    """
    return torch.from_numpy(np.logical_and.reduce([
        np.abs(eyepos[:, 0]) < valid_radius,
        np.abs(eyepos[:, 1]) < valid_radius,
        dpi_valid
    ]))
# Normalize stimulus
print('Normalizing stimulus...')
stim = normalize_stimulus(dset['stim'])

# Create valid eye position mask
print(f'Creating valid eye position mask (radius={valid_eyepos_radius} deg)...')
dfs = create_valid_eyepos_mask(dset['eyepos'], dset['dpi_valid'], valid_eyepos_radius)
dfs = dfs.float()

n_valid_frames = dfs.sum().item()
n_frames = stim.shape[0]
print(f'Valid frames: {int(n_valid_frames)}/{n_frames} ({100*n_valid_frames/n_frames:.1f}%)')

#%%
# Calculate STAs (Spike-Triggered Averages)

print('CALCULATING STAs')


def ensure_tensor(x, device=None, dtype=None):
    """
    Ensures that the input is a torch.Tensor. If it is a numpy array, it is converted to a tensor.
    If device is provided, the tensor is moved to the device.

    Parameters:
    ----------
    x : numpy.ndarray, torch.Tensor, int, float, list, or tuple
        The input array or tensor.
    device : torch.device
        The device to move the tensor to.
    dtype : torch.dtype
        The data type to convert the tensor to.

    Returns:
    -------
    torch.Tensor
        The input converted to a tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, int) or isinstance(x, float):
        x = torch.tensor([x])
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x

def calc_sta(stim, robs, lags, dfs=None, inds=None, stim_modifier=lambda x: x, reverse_correlate=True, batch_size=None, device=None, progress=False):
    '''
    Calculates the spike-triggered average (STA) for a given stimulus and response.

    Parameters:
    -----------
    stim : numpy.ndarray or torch.Tensor
        The stimulus data. Shape: (n_frames, n_channels, n_y, n_x) or (n_frames, n_y, n_x)
    robs : numpy.ndarray or torch.Tensor
        Observed rate for the neural data. Shape: (n_frames, n_units)
    lags : int or list
        The number of lags to calculate the STA for. If an int, the STA will be calculated for lags 0 to lags-1.
        If a list, the STA will be calculated for the specified lags.
    dfs : numpy.ndarray or torch.Tensor
        Datafilters for the neural data. Shape: (n_frames, n_units). Used to weight the response data.
        If None, the response data will not be weighted.
        If a 1D array, the datafilter will be repeated for all units.
    inds : numpy.ndarray or torch.Tensor
        Indices to calculate the STA for. Default: None (all indices).
    stim_modifier : function
        Function to modify the stimulus data before calculating the STA. Default: lambda x: x
    reverse_correlate : bool
        If True, the STA will be normalized by the number of spikes. If False, the STA will be normalized by the number of frames.
    batch_size : int
        The batch size to use for calculating the STA. Default: None (all frames).
    device : torch.device
        The device to use for the calculations. Default: None (CPU).
    progress : bool
        If True, a progress bar will be displayed. Default: False
        
    Returns:
    --------
    torch.Tensor
        The spike-triggered average. Shape: (n_units, n_lags, n_channels, n_y, n_x)

    author: RKR 2/7/2024 (largely copied from Dekel)
    '''
    stim = ensure_tensor(stim, dtype=torch.float32)
    # If missing channel dimension, add it
    if stim.dim() == 3:
        stim = stim.unsqueeze(1)

    robs = ensure_tensor(robs, dtype=torch.float32)
    dfs = ensure_tensor(dfs, dtype=torch.float32) if dfs is not None else None
    inds = ensure_tensor(inds, dtype=torch.long) if inds is not None else None

    if robs.dim() == 1:
        robs = robs[:,None]

    if dfs is not None and dfs.dim() == 1:
        # repeat dfs over all units
        dfs = dfs[:,None].repeat(1, robs.shape[1])
    
    n_frames, n_c, n_y, n_x = stim.shape
    n_frames_robs, n_units  = robs.shape
    
    assert n_frames == n_frames_robs, f"Number of frames in stim ({n_frames}) does not match number of frames in robs ({n_frames_robs})"

    if dfs is not None:
        n_frames_dfs, n_units_dfs = dfs.shape
        assert n_units == n_units_dfs, f"Number of units in robs ({n_units}) does not match number of units in dfs ({n_units_dfs})"
        assert n_frames == n_frames_dfs, f"Number of frames in dfs ({n_frames_dfs}) does not match number of frames in stim ({n_frames})"

    if device is None:
        device = stim.device

    if inds is None:
        inds = torch.arange(stim.shape[0])

    if isinstance(lags, int):
        lags = [l for l in range(lags)]
    
    n_lags = len(lags)
    
    sts = torch.zeros((n_units, n_lags, n_c, n_y, n_x), dtype=torch.float64).to(device)
    div = torch.zeros((n_units, n_lags, n_c, n_y, n_x), dtype=torch.float64).to(device)
    if batch_size is None:
        batch_size = n_frames
    n_batches = int(np.ceil(len(inds) / batch_size))

    pbar = tqdm(total=n_lags*n_batches, desc='Calculating STA') if progress else Mock()
    try:
        for iL,lag in enumerate(lags):  
            ix = inds[inds < n_frames-lag] # prevent indexing out of bounds
            for iB in range(0, len(ix), batch_size):
                bs = slice(iB, iB+batch_size)
                inds_batch = ix[bs]
                stim_batch = stim_modifier(stim[inds_batch,...].to(device))
                robs_batch = robs[inds_batch+lag,:].to(device)
                if dfs is not None:
                    dfs_batch = dfs[inds_batch+lag,:].to(device)
                    robs_batch = robs_batch * dfs_batch
                    if reverse_correlate:
                        div += robs_batch.sum(dim=0)[:,None,None,None,None]
                    else:
                        # Optimized: avoid per-unit loop by leveraging stationarity
                        # stim_batch: (batch_size, n_channels, height, width)
                        # dfs_batch: (batch_size, n_units)
                        # Sum stimulus across batch, then scale by valid frames per unit
                        stim_sum = stim_batch.sum(dim=0)  # (n_channels, height, width)
                        dfs_sum = dfs_batch.sum(dim=0)    # (n_units,)
                        div[:,iL] += stim_sum[None,...] * dfs_sum[:,None,None,None]
                else:
                    if reverse_correlate:
                        div += robs_batch.sum(dim=0)[:,None,None,None,None]
                    else:
                        div[:,iL] += stim_batch.sum(dim=0)[None,...]
                sts[:,iL] += torch.einsum('bcij, bn->ncij',stim_batch, robs_batch)
                torch.cuda.empty_cache()
                pbar.update(1)
    finally:
        pbar.close()

    # Handle zeros in div that would cause NaN
    zero_mask = (div == 0)
    if zero_mask.any():
        # Replace zeros with a small value to avoid NaN
        div = torch.where(div == 0, torch.tensor(1e-10, device=div.device, dtype=div.dtype), div)

    out = torch.squeeze(sts / div, dim=2)
    return out

stas = calc_sta(
    stim,
    dset['robs'],
    n_lags,
    dfs,
    device=device,
    batch_size=batch_size,
    stim_modifier=lambda x: x,  # No modification for STA
    progress=True
).cpu().numpy()

print(f'STA shape: {stas.shape}')  # (n_units, n_lags, n_y, n_x)

#%% ============================================================================
# Calculate STEs (Spike-Triggered Ensembles) and SNR
# ==============================================================================

print(f'\n{"="*80}')
print('CALCULATING STEs AND SNR')
print(f'{"="*80}\n')

print(f'Computing STEs with {n_lags} temporal lags...')

stes = calc_sta(
    stim,
    dset['robs'],
    n_lags,
    dfs,
    device=device,
    batch_size=batch_size,
    stim_modifier=lambda x: x**2,  # Square for STE
    progress=True
).cpu().numpy()

print(f'STE shape: {stes.shape}')  # (n_units, n_lags, n_y, n_x)

# Calculate SNR from STEs
print('\nCalculating SNR...')
signal = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
signal = gaussian_filter(signal, sigma=[0, 0, 4, 4])
noise = np.median(signal[:, 0], axis=(1, 2))
snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
max_snr = np.max(snr_per_lag, axis=1)
peak_lag = snr_per_lag.argmax(axis=1)

print(f'SNR range: [{max_snr.min():.2f}, {max_snr.max():.2f}]')
print(f'Mean SNR: {max_snr.mean():.2f}')
print(f'Median SNR: {np.median(max_snr):.2f}')

#%% ============================================================================
# Report units above SNR threshold
# ==============================================================================

print(f'\n{"="*80}')
print('UNIT SELECTION')
print(f'{"="*80}\n')

rf_mask = max_snr >= snr_threshold
n_rfs = np.sum(rf_mask)
rf_cids = np.where(rf_mask)[0]
rf_snr = max_snr[rf_mask]
n_units = dset['robs'].shape[1]

print(f'SNR threshold: {snr_threshold}')
print(f'All units above threshold: {np.sum(max_snr >= snr_threshold)}/{n_units} ({100*n_rfs/n_units:.1f}%)')
print(f'Unit indices: {rf_cids}')

#%% ============================================================================
# Plot STAs and STEs at peak lag for high-SNR units
# ==============================================================================

print(f'\n{"="*80}')
print('PLOTTING STAs AND STEs AT PEAK LAG')
print(f'{"="*80}\n')

# Sort all high-SNR units by SNR (descending)
sorted_indices = rf_cids[np.argsort(-rf_snr)]
n_total = len(sorted_indices)

# Pagination settings
units_per_page = 25
n_pages = int(np.ceil(n_total / units_per_page))
n_cols = 5  # 5 units per row, each unit has STA and STE side by side

print(f'Plotting {n_total} units across {n_pages} page(s) ({units_per_page} units per page)')
print(f'Each unit shows STA (left) and STE (right) at peak lag')
# Save SNR histogram as first page
fig_hist, ax = plt.subplots(figsize=(10, 6))
ax.hist(max_snr, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(snr_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {snr_threshold}')
ax.set_xlabel('Max SNR', fontsize=12)
ax.set_ylabel('Number of Units', fontsize=12)
ax.set_title(f'SNR Distribution\n{n_total} Units above Threshold', fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Plot STA/STE pages
for page in range(n_pages):
    start_idx = page * units_per_page
    end_idx = min(start_idx + units_per_page, n_total)
    page_indices = sorted_indices[start_idx:end_idx]
    n_units_page = len(page_indices)

    # Calculate grid dimensions
    n_rows = int(np.ceil(n_units_page / n_cols))

    # Create figure: 2 columns per unit (STA + STE)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(2.5 * n_cols * 2, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, unit_idx in enumerate(page_indices):
        row = idx // n_cols
        col = (idx % n_cols) * 2  # Each unit takes 2 columns

        lag = peak_lag[unit_idx]

        # Plot STA
        ax_sta = axes[row, col]
        sta_img = stas[unit_idx, lag]
        vmax_sta = np.abs(sta_img).max()
        ax_sta.imshow(sta_img, cmap='RdBu_r', vmin=-vmax_sta, vmax=vmax_sta, origin='upper')
        ax_sta.set_title(f'Unit {unit_idx} STA\nSNR={max_snr[unit_idx]:.1f}, lag={lag}', fontsize=9)
        ax_sta.axis('off')

        # Plot STE
        ax_ste = axes[row, col + 1]
        ste_img = stes[unit_idx, lag]
        ste_centered = ste_img - np.median(ste_img)
        vmax_ste = np.abs(ste_centered).max()
        ax_ste.imshow(ste_centered, cmap='hot', vmin=0, vmax=vmax_ste, origin='upper')
        ax_ste.set_title(f'Unit {unit_idx} STE\nSNR={max_snr[unit_idx]:.1f}, lag={lag}', fontsize=9)
        ax_ste.axis('off')

    # Hide unused axes
    for idx in range(n_units_page, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    fig.suptitle(f'STAs and STEs at Peak Lag (Page {page + 1}/{n_pages})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

#%%
def plot_stas(sta, row_labels:list = None, col_labels:list = None, share_scale=False, ax=None):
    """
    Plots STAs across lags.

    Parameters
    ----------
    sta : np.ndarray
        STA with shape (n_rows, n_lags, n_channels, n_y, n_x) or (n_lags, n_channels, n_y, n_x)
    """
    if isinstance(sta, torch.Tensor):
        sta = sta.detach().cpu().numpy()

    if sta.ndim == 4:
        sta = sta[np.newaxis, ...]

    n_rows, n_lags, n_c, n_y, n_x= sta.shape
    if row_labels is not None:
        assert len(row_labels) == n_rows, 'Number of row labels must match number of rows in sta'
    
    scale = 1 / (np.max(np.abs(sta)) * 2)
    aspect = n_x / n_y
    imshow_kwargs = dict(aspect='equal')
    if n_c == 1:
        # imshow_kwargs['cmap'] = 'gray'
        imshow_kwargs['cmap'] = 'coolwarm'
        imshow_kwargs['vmin'] = 0
        imshow_kwargs['vmax'] = 1

    # Plot sta
    if ax is None:
        fig = plt.figure(figsize=(n_lags*aspect, n_rows))
        ax = fig.subplots(1, 1)
    else:
        fig = ax.figure
    for iR in range(n_rows):
        if not share_scale:
            scale = 1 / (np.max(np.abs(sta[iR])) * 2)

        for iL in range(n_lags):
            x0, x1 = iL*aspect, (iL+1)*aspect
            y0, y1 = -iR-1, -iR
            ax.imshow(sta[iR,iL].transpose(1,2,0) * scale + .5, 
                       extent=[x0, x1, y0, y1], 
                       **imshow_kwargs)
            ax.plot([x0, x1, x1, x0, x0], [y1, y1, y0, y0, y1], 'k-')
        
        ax.set_ylim([-n_rows-.1, .1])
        ax.set_xlim([-.1, n_lags*aspect+.1])
        # turn off 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('equal')

        ax.set_yticks([-iR-.5 for iR in range(n_rows)])
        if row_labels is None: 
            row_labels = [f'{iR}' for iR in range(n_rows)]
        ax.set_yticklabels(row_labels)
        
        ax.set_xticks([(iL+.5)*aspect for iL in range(n_lags)])
        if col_labels is None:
            col_labels = [f'{iL}' for iL in range(n_lags)]
        ax.set_xticklabels(col_labels)
    return fig, ax

# intercalate STAs and STEs from high SNR units (10 per plot)
n_per_plot = 10
n_plots = int(np.ceil(len(rf_cids) / n_per_plot))
for iP in range(n_plots):
    start_idx = iP * n_per_plot
    end_idx = min((iP + 1) * n_per_plot, len(rf_cids))
    plot_cids = rf_cids[start_idx:end_idx]
    n_units_plot = len(plot_cids)

    sta_ste_list = []
    for cid in plot_cids:
        sta = stas[[cid],:,None]  # Shape (1, n_channels, n_y, n_x)
        ste = stes[[cid],:,None]  # Shape (1, n_channels, n_y, n_x)
        ste -= np.median(ste)
        sta_ste_list.append(sta)
        sta_ste_list.append(ste)
    
    sta_ste_array = np.concatenate(sta_ste_list, axis=0)  # Shape (2*n_units_plot, n_channels, n_y, n_x)

    row_labels = []
    for cid in plot_cids:
        row_labels.append(f'Unit {cid} STA')
        row_labels.append(f'Unit {cid} STE')

    fig, ax = plot_stas(
        sta_ste_array,
        row_labels=row_labels,
        col_labels=[f'Lag {iL}' for iL in range(n_lags)],
        share_scale=False
    )
    fig.suptitle(f'STAs and STEs for Units {start_idx} to {end_idx-1}', fontsize=16)
    plt.show()


#%%
temp_stes = stes[rf_mask].mean(axis=(2,3))
temp_stes -= np.median(temp_stes, axis=1, keepdims=True)
avg_ste = temp_stes.mean(axis=0)
t = np.arange(n_lags)  / 240
peak_lag = np.argmax(avg_ste)
print(f'Peak lag: {peak_lag} ({t[peak_lag]*1000:.1f} ms)')

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(t,temp_stes.T, alpha=0.2, color='gray')
plt.plot(t,avg_ste, color='red', linewidth=2)
plt.axvline(t[peak_lag], color='blue', linestyle='--')
plt.title('Temporal STA across high-SNR units\n max at {:.1f} ms'.format(t[peak_lag]*1000))
plt.show()
    


#%%
rf_v1_cids = np.where(region_mask)[0][rf
#%%
# Combine serial data into trial aligned data
robs = dset['robs'].numpy()
eyepos = dset['eyepos'].numpy()
time_inds = dset['psth_inds'].numpy()
trial_inds = dset['trial_inds'].numpy()
dfs = dset['dpi_valid'].numpy()
unique_trials = np.unique(trial_inds)
n_trials = len(unique_trials)
n_time = np.max(time_inds).item()+1
n_units = dset['robs'].shape[1]
robs_trial = np.nan*np.zeros((n_trials, n_time, n_units))
eyepos_trial = np.nan*np.zeros((n_trials, n_time, 2))
dfs_trial = np.nan*np.zeros((n_trials, n_time, 1))
dur_trial = np.zeros(n_trials)

for itrial in range(n_trials):
    trial_idx = np.where(trial_inds == unique_trials[itrial])[0]
    robs_trial[itrial, time_inds[trial_idx]] = robs[trial_idx]      
    eyepos_trial[itrial, time_inds[trial_idx]] = eyepos[trial_idx]
    dfs_trial[itrial, time_inds[trial_idx]] = dfs[trial_idx, None]
    dur_trial[itrial] = len(trial_idx)

dt = 1/240
analyzer2 = DualWindowAnalysis(robs_trial, eyepos_trial, dfs_trial[:,:,0]>0, dt=dt)
analyzer2.run_sweep([10, 20, 40], t_hist_ms=5, n_bins=35)

#%%

t_bins = np.arange(robs_trial.shape[1])*dt
with PdfPages(f'compare_unit_rasters_{sess.name}_v1.pdf') as pdf:
    for cc in cids:
        fig, ax = plt.subplots(2, 2, figsize=(10,10))
        ax[0,0].set_title(f"Neuron {cc} - Jake")
        ind = np.argsort(fix_dur)
        jj, ii = np.where(binned_spikes[:,:,cc][good_trials][ind])
        plot_raster(time_bins[ii], jj, height=1, ax=ax[0,0])
        ax[0,0].axvline(0, color='r', linestyle='--')
        ax[0,0].set_xlim(-0.1, 1.0)
        analyzer.inspect_neuron_pair(cc, cc, 10, ax=ax[0,1], show=False)


        ax[1,0].set_title(f"Neuron {cc} - Ryan")
        ind = np.argsort(dur_trial)
        jj, ii = np.where(robs_trial[:,:,cc][ind])
        plot_raster(t_bins[ii], jj, height=1, ax=ax[1,0])
        ax[1,0].axvline(0, color='r', linestyle='--')
        ax[1,0].set_xlim(-0.1, 1.0)
        analyzer2.inspect_neuron_pair(cc, cc, 10, ax=ax[1,1], show=True)
        plt.show()
        pdf.savefig(fig)
        plt.close(fig)

