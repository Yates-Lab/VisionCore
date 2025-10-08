#%%

# %%
def get_contiguous_block_lengths(dfs):
    """
    Get lengths of contiguous blocks where any cell is valid.
    
    Args:
        dfs: torch.Tensor of shape [time_points, n_cells]
    
    Returns:
        list: Lengths of contiguous valid blocks
    """
    # Check if any cell is valid at each time point
    valid_mask = torch.any(dfs != 0, dim=1)  # Shape: [time_points]
    
    # Find transitions between valid and invalid
    diff = torch.diff(valid_mask.int())
    starts = torch.where(diff == 1)[0] + 1  # Start of valid blocks
    ends = torch.where(diff == -1)[0] + 1   # End of valid blocks
    
    # Handle edge cases
    if valid_mask[0]:  # If sequence starts with valid data
        starts = torch.cat([torch.tensor([0]), starts])
    if valid_mask[-1]:  # If sequence ends with valid data
        ends = torch.cat([ends, torch.tensor([len(valid_mask)])])
    
    # Calculate block lengths
    block_lengths = (ends - starts).tolist()
    
    return block_lengths


# %%


# %%
import numpy as np  
def visualize_dfs_and_position(dset, dataset_config, dset_idx=0, window_start=8000, window_length=1000, dfs=None):
    #visualize dfs and position
    # Show only a specified window
    window_end = window_start + window_length
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 8))

    # Plot data filter (dfs) and velocity
    plt.subplot(2, 1, 1)

    # Calculate velocity for the window
    dt = 1/dataset_config['sampling']['target_rate']
    eyedt = torch.gradient(dset.dsets[dset_idx]['eyepos'][train_dset.inds[:,1]], dim=0)[0]
    eyevel = torch.hypot(eyedt[:,0], eyedt[:,1])/dt

    # Plot data filter - use provided dfs or fall back to train_dset
    ax1 = plt.gca()
    if dfs is None:
        # Use dfs from dset
        dfs_data = dset[:]['dfs'][window_start:window_end]
    else:
        # Validate shape matches dset dfs
        expected_shape = dset[:]['dfs'].shape
        if dfs.shape != expected_shape:
            raise ValueError(f"dfs shape {dfs.shape} does not match expected shape {expected_shape}")
        dfs_data = dfs[window_start:window_end]
    
    # Check if any cell is valid (value of 1) at each time point
    if dfs_data.ndim > 1:
        dfs_data = (dfs_data == 1).any(dim=1).float()  # 1 if any cell is valid, 0 otherwise
    line1 = ax1.plot(dfs_data, 'b-', label='Data Filter', alpha=0.7)
    ax1.set_ylabel('Valid (1.0) / Invalid (0.0)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot velocity on secondary y-axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(eyevel[window_start:window_end], 'r-', label='Eye Speed', alpha=0.7)
    ax2.set_ylabel('Speed (pixels/frame)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Data Filter and Eye Speed')
    plt.xlabel('Frame number')

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Plot eye position
    plt.subplot(2, 1, 2)
    plt.plot(dset[:]['eyepos'][window_start:window_end, 0], label='X position')
    plt.plot(dset[:]['eyepos'][window_start:window_end, 1], label='Y position')
    plt.title('Eye Position')
    plt.ylabel('Position (pixels)')
    plt.xlabel('Frame number')
    plt.legend()

    plt.tight_layout()
    plt.show()
import torch
def get_dfs(data, dset_idx, dataset_config, fixations_only=True, speed_thresh=20, smooth_eyepos=False, smooth_window=5):
    dfs = data.dsets[dset_idx]['dfs'][data.inds[:,1]]
    if fixations_only:
        dt = 1/dataset_config['sampling']['target_rate']
        
        eyepos = data.dsets[dset_idx]['eyepos'][data.inds[:,1]]
        
        if smooth_eyepos:
            # Apply Gaussian smoothing to eyepos
            from scipy.ndimage import gaussian_filter1d
            import numpy as np
            # Convert to numpy for scipy smoothing
            eyepos_np = eyepos.detach().cpu().numpy()
            eyepos_smooth = torch.zeros_like(eyepos)
            # Apply Gaussian filter to each dimension
            eyepos_smooth[:, 0] = torch.from_numpy(gaussian_filter1d(eyepos_np[:, 0], sigma=smooth_window/3.0))
            eyepos_smooth[:, 1] = torch.from_numpy(gaussian_filter1d(eyepos_np[:, 1], sigma=smooth_window/3.0))
            
            eyepos = eyepos_smooth
        
        eyedt = torch.gradient(eyepos, dim=0)[0]
        eyevel = torch.hypot(eyedt[:,0], eyedt[:,1])/dt
        dfs = dfs * (eyevel[:,None] < speed_thresh)

    return dfs


# %%
def compute_power_spectrum(dset, dfs, window_size, overlap_fraction=0.5, dset_idx=0):
    """
    Compute power spectrum using sliding windows over valid contiguous blocks.
    
    Args:
        dset: Dataset containing the data
        dfs: Data filter tensor of shape [time_points, n_cells]
        window_size: Size of the window for FFT computation
        overlap_fraction: Fraction of overlap between windows (0.0 to 0.9)
        dset_idx: Dataset index to use
    
    Returns:
        frequencies: Array of frequency bins
        power_spectrum: Average power spectrum across all windows
    """
    import numpy as np
    from scipy import signal
    
    # Find contiguous valid blocks directly
    valid_mask = torch.any(dfs != 0, dim=1)  # Shape: [time_points]
    
    # Find transitions between valid and invalid
    diff = torch.diff(valid_mask.int())
    starts = torch.where(diff == 1)[0] + 1  # Start of valid blocks
    ends = torch.where(diff == -1)[0] + 1   # End of valid blocks
    
    # Handle edge cases
    if valid_mask[0]:  # If sequence starts with valid data
        starts = torch.cat([torch.tensor([0]), starts])
    if valid_mask[-1]:  # If sequence ends with valid data
        ends = torch.cat([ends, torch.tensor([len(valid_mask)])])
    
    # Calculate step size based on overlap
    step_size = int(window_size * (1 - overlap_fraction))
    if step_size < 1:
        step_size = 1
    
    # Get the data we want to analyze (e.g., eye position or neural data)
    data = dset.dsets[dset_idx]['eyepos'][dset.inds[:,1]]  # Using eyepos as example
    #calculate velocity along 0th dimension

    # data = torch.gradient(data[:,0], dim=0)[0]
    # data = data[:, None]
    speed = np.hypot(np.gradient(data[:,0], axis=0), np.gradient(data[:,1], axis=0))
    data = speed[:, None]

    all_power_spectra = []
    
    # Process each contiguous block
    for start, end in zip(starts, ends):
        block_length = end - start
        if block_length >= window_size:
            # This block is large enough for our window
            block_data = data[start:end]
            
            # Apply sliding window within this block
            for start_idx in range(0, block_length - window_size + 1, step_size):
                end_idx = start_idx + window_size
                
                # Get the absolute indices for this window
                abs_start = start + start_idx
                abs_end = start + end_idx
                
                # Assert that dfs is valid for this entire window (should always be true)
                window_dfs = dfs[abs_start:abs_end]
                assert torch.any(window_dfs != 0, dim=1).all(), f"Window at {abs_start}:{abs_end} contains invalid data points"
                
                window_data = block_data[start_idx:end_idx] - np.mean(block_data[start_idx:end_idx], axis=0)
                
                # Compute FFT for each dimension separately
                for dim in range(window_data.shape[1]):  # X and Y position
                    # Apply window function (Hanning window)
                    windowed_data = window_data[:, dim] * signal.windows.hann(window_size)
                    
                    # Compute FFT
                    fft_data = np.fft.fft(windowed_data)
                    
                    # Compute power spectrum (only positive frequencies)
                    power = np.abs(fft_data[:window_size//2 + 1])**2
                    
                    all_power_spectra.append(power)
    
    if not all_power_spectra:
        print("Warning: No valid windows found for power spectrum computation")
        return np.array([]), np.array([])
    
    # Average all power spectra
    power_spectrum = np.mean(all_power_spectra, axis=0)
    
    # Create frequency array for positive frequencies only
    # Assuming sampling rate from the dataset config
    sampling_rate = 240  # Hz - you might want to get this from dataset_config
    frequencies = np.linspace(0, sampling_rate/2, window_size//2 + 1)
    
    return frequencies, power_spectrum, all_power_spectra


# %%
import yaml
import os

dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_gaborium_for_eye_movements/"

# List full paths to *.yaml files that do not contain "base" in the name
yaml_files = [
    f for f in os.listdir(dataset_configs_path)
    if f.endswith(".yaml") and "base" not in f
]

from DataYatesV1.models.config_loader import load_dataset_configs
dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)

dataset_configs = [dataset_config for dataset_config in dataset_configs if dataset_config['session'] == 'Allen_2022-04-13' ]

#%% Load Data
import contextlib
from DataYatesV1.utils.data import prepare_data
all_power_spectra_for_all_datasets = []
fixations_only = True

for i, dataset_config in enumerate(dataset_configs):

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):          # ← optional
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)

    if fixations_only:  
        dfs = get_dfs(train_dset, 0, dataset_config, smooth_eyepos=True, speed_thresh=5, fixations_only=fixations_only)
    else:
        dfs = train_dset[:]['dfs']

    visualize_dfs_and_position(train_dset, dataset_config, window_start=8000, window_length=1000, dfs=dfs)
    contiguous_block_lengths = get_contiguous_block_lengths(dfs)
    import matplotlib.pyplot as plt
    plt.hist(contiguous_block_lengths, bins=100)
    plt.xlabel('Contiguous Block Length')
    plt.ylabel('Frequency')
    plt.title('Contiguous Block Lengths')
    plt.show()

    # frequencies, power_spectrum, all_power_spectra = compute_power_spectrum(train_dset, dfs, window_size=580, overlap_fraction=0.5, dset_idx=0)
    frequencies, power_spectrum, all_power_spectra = compute_power_spectrum(train_dset, dfs, window_size=60, overlap_fraction=0.5, dset_idx=0)
    # plt.semilogy(frequencies, power_spectrum)
    plt.scatter(frequencies, 10 * np.log(power_spectrum))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    # plt.xlim(0, 20)
    plt.title('Power Spectrum {} \n {}'.format("fixations only" if fixations_only else "all eye movements", dataset_config['session']))
    plt.show()
    all_power_spectra_for_all_datasets.extend(all_power_spectra)
#%%
# frequencies, power_spectrum, all_power_spectra = compute_power_spectrum(train_dset, dfs, window_size=580, overlap_fraction=0.5, dset_idx=0)
dset = train_dset
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
for block in range(0, 10000, 1000):
    data = dset.dsets[0]['eyepos'][dset.inds[:,1]][block:block+1000]  # Using eyepos as example
    #calculate velocity along 0th dimension
    speed = np.hypot(np.gradient(data[:,0], axis=0), np.gradient(data[:,1], axis=0))
    data = speed[:, None]
    axes[0].plot(data)
    f = np.fft.fft(data)
    p = np.abs(f)**2

    axes[1].semilogy(np.fft.fftfreq(len(data), 1/240), p)
    axes[1].set_xlim(0, 20)
plt.tight_layout()
plt.show()
# data = torch.gradient(data[:,0], dim=0)[0]
# data = data[:, None]


# %%
power_spectrum_for_all_datasets = np.mean(all_power_spectra_for_all_datasets, axis=0)

plt.semilogy(frequencies, power_spectrum_for_all_datasets)  
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Fixation Power Spectrum {}'.format("fixations only" if fixations_only else "all eye movements"))
plt.show()
# %%