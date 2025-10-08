#%%
import numpy as np
import torch
def visualize_dfs_and_position_old(gaborium_dset):
    #visualize dfs and position
    # Show only a 10-second window (2400 frames at 240 Hz)
    window_start = 8000  # Start from beginning, or you can change this
    window_end = window_start + 1000  # 10 seconds * 240 Hz
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 8))

    # Plot data filter (dfs) and velocity
    plt.subplot(2, 1, 1)

    # Calculate velocity for the window
    dt = np.diff(gaborium_dset['t_bins'][window_start:window_end+1])
    velocity = np.diff(gaborium_dset['eyepos'][window_start:window_end+1], axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)

    # Plot data filter
    ax1 = plt.gca()
    line1 = ax1.plot(gaborium_dset['dfs'].squeeze()[window_start:window_end], 'b-', label='Data Filter', alpha=0.7)
    ax1.set_ylabel('Valid (1.0) / Invalid (0.0)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot velocity on secondary y-axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(speed, 'r-', label='Eye Speed', alpha=0.7)
    ax2.set_ylabel('Speed (pixels/frame)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Data Filter and Eye Speed - 10 second window')
    plt.xlabel('Frame number')

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Plot eye position
    plt.subplot(2, 1, 2)
    plt.plot(gaborium_dset['eyepos'][window_start:window_end, 0], label='X position')
    plt.plot(gaborium_dset['eyepos'][window_start:window_end, 1], label='Y position')
    plt.title('Eye Position - 10 second window')
    plt.ylabel('Position (pixels)')
    plt.xlabel('Frame number')
    plt.legend()

    plt.tight_layout()
    plt.show()
def visualize_dfs_and_position(dset, dataset_config = None, dset_idx=0, window_start=8000, window_length=1000, dfs=None):
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
#%%
#%%
import yaml
import os

dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_gaborium_for_lnenergy_fits/"

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
        # dfs = get_dfs(train_dset, 0, dataset_config, smooth_eyepos=False, speed_thresh=40, fixations_only=fixations_only)
    else:
        dfs = train_dset[:]['dfs']

    visualize_dfs_and_position(train_dset, dataset_config, window_start=8000, window_length=1000, dfs=dfs)

#%%
def transform_to_gaborium_format(dataset, cell_to_fit = None):
    # Check if dataset is already a dictionary (like gaborium) or a CombinedEmbeddedDataset
    if isinstance(dataset, dict):
        # Already in the right format, just need to fix temporal structure
        robs = dataset['robs']
        # Fix temporal structure: [samples, 1, lags, height, width] -> [samples, lags, height, width]
        stim = dataset['stim'].squeeze(1) if dataset['stim'].dim() == 5 else dataset['stim']
        # Fix dfs: [samples, n_cells] -> [samples, 1] (per-sample validity)
        # dfs = dataset['dfs'].mean(dim=1, keepdim=True) if dataset['dfs'].dim() == 2 else dataset['dfs']
        dfs = dataset['dfs']
        # Include eyepos if it exists
        eyepos = dataset.get('eyepos', None)
        result = {'robs': robs, 'stim': stim, 'dfs': dfs}
        if eyepos is not None:
            result['eyepos'] = eyepos
        # result['inds'] = dataset['inds']
        # result['t_bins'] = dataset['underlying_dset']['t_bins'][result['inds'][:,1]]
        # result['underlying_dset'] = dataset['underlying_dset']
        if cell_to_fit is not None:
            result['stim'] = result['stim'][result['dfs'][:, [cell_to_fit]].squeeze().int().bool()]
            result['robs'] = result['robs'][result['dfs'][:, [cell_to_fit]].squeeze().int().bool()]
            # result['dfs'] = result['dfs'][result['dfs'][:, [cell_to_fit]].squeeze().int().bool(), [cell_to_fit]]
            # result['dfs'] = result['dfs'][:, None]
            result['dfs'] = result['dfs'][result['dfs'][:, [cell_to_fit]].squeeze().int().bool(), :]
        return result
    else:
        raise ValueError("Dataset not supported")
def split_config_dataset_into_train_val(train_dset, val_split=0.2, seed=42, verbose=False):
    """
    Split a config-based train dataset into train and validation sets based on trial indices.
    
    Parameters:
    -----------
    train_dset : CombinedEmbeddedDataset
        The training dataset to split
    val_split : float
        Fraction of trials to use for validation (default: 0.2)
    seed : int
        Random seed for reproducible splits (default: 42)
    verbose : bool
        Whether to print detailed information about the split (default: True)
        
    Returns:
    --------
    tuple : (new_train_dict, new_val_dict)
        Two dictionaries that can be used with transform_to_gaborium_format
    """
    import torch
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get trial indices for the dataset
    trial_inds = train_dset.dsets[0]['trial_inds'][train_dset.dset_inds[0]]
    
    # Find unique trials
    unique_trials = torch.unique(trial_inds)
    n_trials = len(unique_trials)
    
    if verbose:
        print(f"Total unique trials: {n_trials}")
        print(f"Splitting {val_split*100:.1f}% of trials for validation")
    
    # Randomly shuffle trials and split
    shuffled_trials = unique_trials[torch.randperm(n_trials)]
    n_val_trials = int(n_trials * val_split)
    
    val_trials = shuffled_trials[:n_val_trials]
    train_trials = shuffled_trials[n_val_trials:]
    
    if verbose:
        print(f"Train trials: {len(train_trials)}, Val trials: {len(val_trials)}")
    
    # Create boolean masks for train and val samples
    train_mask = torch.isin(trial_inds, train_trials)
    val_mask = torch.isin(trial_inds, val_trials)
    
    if verbose:
        print(f"Train samples: {train_mask.sum()}, Val samples: {val_mask.sum()}")
    
    # Load the full dataset
    full_data = train_dset[:]
    
    # Split each key based on the masks
    new_train_dict = {}
    new_val_dict = {}
    
    for key, value in full_data.items():
        if isinstance(value, torch.Tensor) and len(value) == len(train_mask):
            # Split tensors that match the sample dimension
            new_train_dict[key] = value[train_mask]
            new_val_dict[key] = value[val_mask]
            if verbose:
                print(f"Split {key}: train {new_train_dict[key].shape}, val {new_val_dict[key].shape}")
        else:
            # Keep other data as-is (like metadata)
            new_train_dict[key] = value
            new_val_dict[key] = value
            if verbose:
                print(f"Kept {key} unchanged: {type(value)}")
    
    # Verify no trial overlap using the original trial indices and masks
    train_trial_inds_split = trial_inds[train_mask]
    val_trial_inds_split = trial_inds[val_mask]
    
    train_trial_set = set(train_trial_inds_split.unique().tolist())
    val_trial_set = set(val_trial_inds_split.unique().tolist())
    overlap = train_trial_set.intersection(val_trial_set)
    
    if len(overlap) == 0:
        if verbose:
            print("✅ No trial overlap between train and val splits")
            print(f"Train trials: {sorted(list(train_trial_set))[:5]}... ({len(train_trial_set)} total)")
            print(f"Val trials: {sorted(list(val_trial_set))[:5]}... ({len(val_trial_set)} total)")
    else:
        if verbose:
            print(f"❌ WARNING: {len(overlap)} trials overlap between train and val: {overlap}")
        raise ValueError("Trial overlap between train and val")
    
    return new_train_dict, new_val_dict

new_train_dict, new_val_dict = split_config_dataset_into_train_val(
    train_dset, 
    val_split=0.2,  # 20% of trials for validation
    seed=42
)

train_dset_new_loaded = transform_to_gaborium_format(new_train_dict)
val_dset_new_loaded = transform_to_gaborium_format(new_val_dict)
test_dset_new_loaded = transform_to_gaborium_format(val_dset[:])

#%%

#%% Utility: Save stimulus movie with colorbar
def save_stim_movie(loaded_dset, start_idx, stop_idx, out_path, lag = 0, fps = 240):
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    assert 'stim' in loaded_dset, "Expected key 'stim' in loaded dataset"
    stim = loaded_dset['stim']
    if isinstance(stim, torch.Tensor):
        stim = stim.detach().cpu()

    # Determine shape and extract frames consistently as [T, H, W]
    if stim.dim() == 5:
        # [T, C=1, L, H, W] or [T, 1, L, H, W]
        frames = stim[start_idx:stop_idx, 0, lag]
    elif stim.dim() == 4:
        # Either [T, 1, H, W] or [T, L, H, W]
        if stim.shape[1] == 1:
            frames = stim[start_idx:stop_idx, 0]
        else:
            frames = stim[start_idx:stop_idx, lag]
    elif stim.dim() == 3:
        # [T, H, W]
        frames = stim[start_idx:stop_idx]
    else:
        raise ValueError(f"Unsupported stim shape: {tuple(stim.shape)}")

    frames_np = frames.numpy()
    vmin = float(np.min(frames_np))
    vmax = float(np.max(frames_np))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames_np[0], cmap='gray', vmin=vmin, vmax=vmax, animated=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Stimulus value')
    ax.set_title(f'Stim frames {start_idx}:{stop_idx} (lag={lag})')
    ax.axis('off')

    def init():
        im.set_data(frames_np[0])
        return (im,)

    def update(i):
        im.set_data(frames_np[i])
        return (im,)

    interval_ms = 1000.0 / float(fps)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=frames_np.shape[0], interval=interval_ms, blit=True
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Prefer ffmpeg if available, else fall back to pillow
    try:
        ffmpeg_writer = animation.FFMpegWriter(fps=fps, metadata={'artist': 'DataYatesV1'}, bitrate=1800)
        anim.save(out_path, writer=ffmpeg_writer)
    except Exception:
        from matplotlib.animation import PillowWriter
        pillow_writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=pillow_writer)

    plt.close(fig)

# Example usage (uncomment to generate a short preview)
# save_stim_movie(gaborium_dset_train_new_combined_loaded, start_idx=500, stop_idx=1500, out_path='stim_movie_train.mp4', lag=0, fps=240)

#%% Reproduce gaborium dataset from lnp_time_simple.py
import torch
import numpy as np
from DataYatesV1 import DictDataset, get_session, CombinedEmbeddedDataset, split_inds_by_trial_train_val_test

# Get the same session as used in lnp_time_simple.py
sess = get_session('Allen', '2022-04-13')

# Load the gaborium dataset exactly as in lnp_time_simple.py
gaborium_dset = DictDataset.load(sess.sess_dir / 'shifter' / 'gaborium_shifted.dset')
gaborium_dset['stim'] = gaborium_dset['stim'].float()
gaborium_dset['stim'] = (gaborium_dset['stim'] - gaborium_dset['stim'].mean()) / 255

# Define utilities (same as lnp_time_simple.py)
n_lags = 32  # Match config dataset
use_speed_filter = True  # Set to False to include all eye movements (saccades + fixations)
#%%
def get_inds(dset, n_lags, speed_thresh = 40, use_speed_filter = False):
    dpi_valid = dset['dpi_valid']
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
    dfs = ~new_trials
    dfs &= (dpi_valid > 0)

    if use_speed_filter:
        dt = np.diff(gaborium_dset['t_bins'])
        velocity = np.diff(gaborium_dset['eyepos'], axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)

        dfs_speed = speed < speed_thresh
        dfs_speed = np.concatenate([np.zeros(1, dtype = bool), dfs_speed])
        dfs &= dfs_speed
    
    for iL in range(n_lags):
        dfs &= torch.roll(dfs, 1)
    
    dfs = dfs.float()
    # Create per-cell data filters to match config dataset behavior
    n_cells = dset['robs'].shape[1]
    dfs = dfs[:, None]  # Shape: [n_frames, 1]
    return dfs
n_units = gaborium_dset['robs'].shape[1]
n_y, n_x = gaborium_dset['stim'].shape[1:3]
gaborium_dset['dfs'] = get_inds(gaborium_dset, n_lags, use_speed_filter=use_speed_filter)
gaborium_inds = gaborium_dset['dfs'].any(dim=1).nonzero(as_tuple=True)[0]
visualize_dfs_and_position_old(gaborium_dset)




#%% Calculate and visualize STA for cell 31
print("\n" + "="*50)
print("STA CALCULATION AND VISUALIZATION FOR CELL 31")
print("="*50)

# Calculate STA for cell 31 using the same method as lnp_time_simple.py
cell_to_fit = 31
n_lags_sta = 18  # Use 18 lags for STA calculation
from DataYatesV1 import calc_sta
from DataYatesV1.utils.basic_shifter import plot_sta_images

# Calculate STA using calc_sta function
sta_cell_31 = calc_sta(
    gaborium_dset['stim'].detach().cpu(), 
    gaborium_dset['robs'].cpu()[:, [cell_to_fit]],  # Select only cell 31
    range(n_lags_sta), 
    dfs=gaborium_dset['dfs'].cpu(),  # Use the data filter we created
    progress=True
).cpu().squeeze().numpy()

print(f"STA shape for cell {cell_to_fit}: {sta_cell_31.shape}")
print(f"STA min/max: {sta_cell_31.min():.4f} / {sta_cell_31.max():.4f}")

# Visualize STA using the same method as save_sta_visualization
sta_norm = sta_cell_31 / np.max(np.abs(sta_cell_31))
vmin, vmax = np.min(sta_norm), np.max(sta_norm)

# Ensure symmetric color scale
if abs(vmin) > abs(vmax):
    vmax = -vmin
else:
    vmin = -vmax
#%%
from DataYatesV1.utils.basic_shifter import plot_sta_images
from matplotlib import pyplot as plt
plt.figure(figsize=(12, 10))
fig, axs = plot_sta_images(sta_norm, {'cmap': 'coolwarm', 'vmin': vmin, 'vmax': vmax}, title_prefix='lag')
plt.suptitle(f"STA for Cell {cell_to_fit} (Saccade Data)", fontsize=16)
plt.tight_layout()
plt.show()

# Also show a single lag for comparison
peak_lag = 7  # Middle lag
plt.figure(figsize=(8, 6))
plt.imshow(sta_norm[peak_lag], cmap='coolwarm', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title(f"Cell {cell_to_fit} STA - Lag {peak_lag}")
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()

#%%
# Define which keys and lags to use (same as lnp_time_simple.py)
keys_lags = {
    'robs': 0,
    'stim': np.arange(18),  # Keep 18 lags for stimulus (as in original)
    'dfs': 0,
}

# Split data into training and validation sets (same as lnp_time_simple.py)
gaborium_train_inds, gaborium_val_inds, gaborium_test_inds = split_inds_by_trial_train_val_test(
    gaborium_dset, gaborium_inds, train_split = 0.8, val_split = 0.2, seed=1002
)
print(f'Gaborium sample split: {len(gaborium_train_inds) / len(gaborium_inds):.3f} train, {len(gaborium_val_inds) / len(gaborium_inds):.3f} val')

# Create the datasets (same as lnp_time_simple.py)
gaborium_train_dset = CombinedEmbeddedDataset([gaborium_dset], [gaborium_train_inds], keys_lags)
gaborium_val_dset = CombinedEmbeddedDataset([gaborium_dset], [gaborium_val_inds], keys_lags)
gaborium_test_dset = CombinedEmbeddedDataset([gaborium_dset], [gaborium_test_inds], keys_lags)

# Load datasets
gaborium_train_dset_loaded = gaborium_train_dset[:]
gaborium_val_dset_loaded = gaborium_val_dset[:]
gaborium_test_dset_loaded = gaborium_test_dset[:]

print("Gaborium dataset from lnp_time_simple.py reproduced successfully!")
print(f"Train dataset shape: {gaborium_train_dset_loaded['stim'].shape}")
print(f"Val dataset shape: {gaborium_val_dset_loaded['stim'].shape}")
print(f"Test dataset shape: {gaborium_test_dset_loaded['stim'].shape}")

# Now you can compare gaborium_train_dset_loaded, gaborium_val_dset_loaded, gaborium_test_dset_loaded
# with the train_dset and val_dset from the config-based loading method

# %%

save_stim_movie(gaborium_train_dset_loaded, start_idx=500, stop_idx=1500, out_path='stim_movie_train_incorrect.mp4', lag=0, fps=240)


#%% Compare datasets - detailed analysis
print("="*80)
print("DATASET COMPARISON ANALYSIS")
print("="*80)

# Compare the config-based datasets (from the loop) with gaborium datasets
print("\n1. DATASET STRUCTURE COMPARISON")
print("-" * 50)

# Get the last processed dataset from the loop (assuming we want to compare with the last one)
if 'train_dset_new_loaded' in locals() and 'val_dset_new_loaded' in locals():
    print("Config-based datasets (from prepare_data):")
    print(f"  Train dataset type: {type(train_dset_new_loaded)}")
    print(f"  Val dataset type: {type(val_dset_new_loaded)}")
    print(f"  Train dataset length: {len(train_dset_new_loaded)}")
    print(f"  Val dataset length: {len(val_dset_new_loaded)}")
    
    # Get a sample from config-based datasets
    print(f"  Train sample keys: {list(train_dset_new_loaded.keys())}")
    for key, value in train_dset_new_loaded.items():
        if hasattr(value, 'shape'):
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"    {key}: type={type(value)}")
else:
    print("Config-based datasets not found in current scope")

print("\nGaborium datasets (from lnp_time_simple.py method):")
print(f"  Train dataset type: {type(gaborium_train_dset_loaded)}")
print(f"  Val dataset type: {type(gaborium_val_dset_loaded)}")
print(f"  Test dataset type: {type(gaborium_test_dset_loaded)}")
print(f"  Train dataset length: {len(gaborium_train_dset_loaded)}")
print(f"  Val dataset length: {len(gaborium_val_dset_loaded)}")
print(f"  Test dataset length: {len(gaborium_test_dset_loaded)}")

# Get a sample from gaborium datasets
print(f"  Gaborium sample keys: {list(gaborium_train_dset_loaded.keys())}")
for key, value in gaborium_train_dset_loaded.items():
    if hasattr(value, 'shape'):
        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"    {key}: type={type(value)}")

print("\n2. STIMULUS DATA COMPARISON")
print("-" * 50)

if 'train_dset_new_loaded' in locals():
    # Config-based stimulus analysis
    config_stim = train_dset_new_loaded['stim']
    print("Config-based stimulus:")
    print(f"  Shape: {config_stim.shape}")
    print(f"  Dtype: {config_stim.dtype}")
    print(f"  Min: {config_stim.min().item():.6f}")
    print(f"  Max: {config_stim.max().item():.6f}")
    print(f"  Mean: {config_stim.mean().item():.6f}")
    print(f"  Std: {config_stim.std().item():.6f}")
    print(f"  Range: {config_stim.max().item() - config_stim.min().item():.6f}")

# Gaborium stimulus analysis
gaborium_stim = gaborium_train_dset_loaded['stim']
print("\nGaborium stimulus:")
print(f"  Shape: {gaborium_stim.shape}")
print(f"  Dtype: {gaborium_stim.dtype}")
print(f"  Min: {gaborium_stim.min().item():.6f}")
print(f"  Max: {gaborium_stim.max().item():.6f}")
print(f"  Mean: {gaborium_stim.mean().item():.6f}")
print(f"  Std: {gaborium_stim.std().item():.6f}")
print(f"  Range: {gaborium_stim.max().item() - gaborium_stim.min().item():.6f}")

print("\n3. RESPONSE DATA COMPARISON")
print("-" * 50)

if 'train_dset_new_loaded' in locals():
    # Config-based response analysis
    config_robs = train_dset_new_loaded['robs']
    print("Config-based responses:")
    print(f"  Shape: {config_robs.shape}")
    print(f"  Dtype: {config_robs.dtype}")
    print(f"  Min: {config_robs.min().item():.6f}")
    print(f"  Max: {config_robs.max().item():.6f}")
    print(f"  Mean: {config_robs.mean().item():.6f}")
    print(f"  Std: {config_robs.std().item():.6f}")
    print(f"  Non-zero fraction: {(config_robs > 0).float().mean().item():.6f}")

# Gaborium response analysis
gaborium_robs = gaborium_train_dset_loaded['robs']
print("\nGaborium responses:")
print(f"  Shape: {gaborium_robs.shape}")
print(f"  Dtype: {gaborium_robs.dtype}")
print(f"  Min: {gaborium_robs.min().item():.6f}")
print(f"  Max: {gaborium_robs.max().item():.6f}")
print(f"  Mean: {gaborium_robs.mean().item():.6f}")
print(f"  Std: {gaborium_robs.std().item():.6f}")
print(f"  Non-zero fraction: {(gaborium_robs > 0).float().mean().item():.6f}")

print("\n4. DATA FILTER COMPARISON")
print("-" * 50)

if 'train_dset_new_loaded' in locals():
    # Config-based dfs analysis
    config_dfs = train_dset_new_loaded['dfs']
    print("Config-based data filters (dfs):")
    print(f"  Shape: {config_dfs.shape}")
    print(f"  Dtype: {config_dfs.dtype}")
    print(f"  Valid fraction: {config_dfs.float().mean().item():.6f}")
    print(f"  Min: {config_dfs.min().item():.6f}")
    print(f"  Max: {config_dfs.max().item():.6f}")

# Gaborium dfs analysis
gaborium_dfs = gaborium_train_dset_loaded['dfs']
print("\nGaborium data filters (dfs):")
print(f"  Shape: {gaborium_dfs.shape}")
print(f"  Dtype: {gaborium_dfs.dtype}")
print(f"  Valid fraction: {gaborium_dfs.float().mean().item():.6f}")
print(f"  Min: {gaborium_dfs.min().item():.6f}")
print(f"  Max: {gaborium_dfs.max().item():.6f}")

print("\n5. TEMPORAL STRUCTURE COMPARISON")
print("-" * 50)

if 'train_dset_new_loaded' in locals():
    print("Config-based temporal structure:")
    config_stim_sample = train_dset_new_loaded['stim']
    print(f"  Stimulus lags: {config_stim_sample.shape[0] if len(config_stim_sample.shape) > 1 else 1}")
    if len(config_stim_sample.shape) > 1:
        print(f"  Spatial dimensions: {config_stim_sample.shape[1:]}")

print("Gaborium temporal structure:")
gaborium_stim_sample = gaborium_train_dset_loaded['stim']
print(f"  Stimulus lags: {gaborium_stim_sample.shape[0] if len(gaborium_stim_sample.shape) > 1 else 1}")
if len(gaborium_stim_sample.shape) > 1:
    print(f"  Spatial dimensions: {gaborium_stim_sample.shape[1:]}")

print("\n6. CELL ID COMPARISON")
print("-" * 50)

if 'train_dset_new_loaded' in locals():
    config_robs_sample = train_dset_new_loaded['robs']
    print(f"Config-based number of cells: {config_robs_sample.shape[-1] if len(config_robs_sample.shape) > 0 else 1}")

gaborium_robs_sample = gaborium_train_dset_loaded['robs']
print(f"Gaborium number of cells: {gaborium_robs_sample.shape[-1] if len(gaborium_robs_sample.shape) > 0 else 1}")

print("\n7. MEMORY USAGE ESTIMATION")
print("-" * 50)

if 'train_dset_new_loaded' in locals():
    config_memory = (train_dset_new_loaded['stim'].numel() * train_dset_new_loaded['stim'].element_size() + 
                     train_dset_new_loaded['robs'].numel() * train_dset_new_loaded['robs'].element_size() + 
                     train_dset_new_loaded['dfs'].numel() * train_dset_new_loaded['dfs'].element_size())
    print(f"Config-based dataset memory (approx): {config_memory / 1024**3:.2f} GB")

gaborium_memory = (gaborium_train_dset_loaded['stim'].numel() * gaborium_train_dset_loaded['stim'].element_size() + 
                   gaborium_train_dset_loaded['robs'].numel() * gaborium_train_dset_loaded['robs'].element_size() + 
                   gaborium_train_dset_loaded['dfs'].numel() * gaborium_train_dset_loaded['dfs'].element_size())
print(f"Gaborium dataset memory (approx): {gaborium_memory / 1024**3:.2f} GB")

print("\n8. KEY DIFFERENCES SUMMARY")
print("-" * 50)
print("Key things to check before using gaborium dataset for ln/energy models:")
print("1. Data types match (especially float32 vs float64)")
print("2. Stimulus normalization ranges are consistent")
print("3. Number of temporal lags matches model expectations")
print("4. Cell IDs and counts are appropriate")
print("5. Data filter (dfs) logic is equivalent")
print("6. Memory requirements are manageable")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)

# %%
