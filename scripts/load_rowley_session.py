#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
import torch
from pathlib import Path

#%% Import the session loading utility
from DataRowleyV1V2.data.registry import get_session
from DataYatesV1 import DictDataset

#%% Load the session
# The session directory is '/mnt/ssd/RowleyMarmoV1V2/processed/Luke_2025-08-04'
# Extract subject and date from the path
subject = 'Luke'
date = '2025-08-04'

# Create session object
sess = get_session(subject, date)
print(f"Session loaded: {sess.name}")
print(f"Session directory: {sess.processed_path}")
print(f"\nAvailable methods on session object:")
print(f"  get_dataset() - Load a dataset")
print(f"  load_spikes() - Load spike data")
print(f"  load_dpi() - Load DPI (depth probe info)")
print(f"  load_exp() - Load experiment metadata")

#%% Load the gaborium dataset (shifted version)
# Use the get_dataset method to load the shifted gaborium dataset
print(f"\nLoading gaborium dataset (shifted)...")
dset = None  # Initialize as None

# First, let's explore what's available in the session directory
print(f"\nExploring session directory structure:")
import os
for root, dirs, files in os.walk(sess.processed_path):
    level = root.replace(str(sess.processed_path), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Limit to first 5 files per directory
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')
    if level > 2:  # Limit depth to avoid too much output
        break

print(f"\n" + "="*60)
print("Looking for available datasets...")

# Look for any .dset files and extract dataset info
datasets_dir = sess.processed_path / 'datasets'
available_datasets = {}

if datasets_dir.exists():
    print(f"\nAvailable datasets:")
    for dataset_path in sorted(datasets_dir.glob('**/*.dset')):
        dataset_type = dataset_path.stem  # 'backimage', 'gaborium', etc.
        dataset_dir = dataset_path.parent.name  # 'left_eye_x-0.5_y-0.3', etc.
        key = f"{dataset_type} ({dataset_dir})"
        available_datasets[key] = {
            'type': dataset_type,
            'directory': str(dataset_path.parent.relative_to(datasets_dir)),
            'path': dataset_path
        }
        print(f"  {key}")

# Load the first available dataset (or backimage if available)
dset = None
dataset_to_load = None

# Prefer backimage if available
for key, info in available_datasets.items():
    if 'backimage' in key:
        dataset_to_load = key
        break

# Otherwise just use the first one
if dataset_to_load is None and available_datasets:
    dataset_to_load = list(available_datasets.keys())[0]

if dataset_to_load:
    print(f"\n" + "="*60)
    print(f"Loading: {dataset_to_load}")
    info = available_datasets[dataset_to_load]
    
    try:
        # Load the .dset file directly
        print(f"Loading from: {info['path']}")
        dset = DictDataset.load(info['path'])
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"Dataset length: {len(dset)}")
        print(f"Available keys in data: {list(dset.keys())}")
        print(f"Available keys in covariates: {list(dset.covariates.keys())}")
        
        if 'stim' in dset:
            print(f"Stimulus shape: {dset['stim'].shape}")
        if 'robs' in dset:
            print(f"Response shape: {dset['robs'].shape}")
        if 'eyepos' in dset:
            print(f"Eye position shape: {dset['eyepos'].shape}")
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No datasets found!")

#%% Inspect the data
if dset is None:
    print("Dataset not loaded. Please run the previous cell to load data first.")
else:
    try:
        # Get a sample
        sample_idx = 0
        if 'stim' in dset:
            stim = dset['stim'][sample_idx]
            print(f"\nSample {sample_idx}:")
            print(f"  Stimulus shape: {stim.shape}")
        
        if 'robs' in dset:
            robs = dset['robs'][sample_idx]
            print(f"  Response shape: {robs.shape}")
            print(f"  Number of neurons: {robs.shape[0] if len(robs.shape) > 1 else 1}")
        
        # Plot first few frames of stimulus if available
        if 'stim' in dset and len(stim.shape) >= 3:
            n_frames = min(6, stim.shape[0])
            fig, axes = plt.subplots(1, n_frames, figsize=(n_frames*2, 2))
            for i in range(n_frames):
                if n_frames == 1:
                    ax = axes
                else:
                    ax = axes[i]
                ax.imshow(stim[i].cpu().numpy() if hasattr(stim[i], 'cpu') else stim[i].numpy(), cmap='gray')
                ax.set_title(f'Frame {i}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Plot neural responses if available
        if 'robs' in dset:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            robs_np = robs.cpu().numpy() if hasattr(robs, 'cpu') else robs.numpy()
            ax.plot(robs_np)
            ax.set_xlabel('Neuron index')
            ax.set_ylabel('Response')
            ax.set_title('Neural responses for sample 0')
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error inspecting data: {e}")

#%% Load different dataset types
# This cell allows you to easily switch between different dataset types and eye calibrations
# Edit the variables below to load different datasets

# Available options:
# dataset_type: 'backimage', 'fixrsvp', or 'gaborium'
# eye_calibration: 'left_eye_x-0.5_y-0.3' or 'right_eye_x-0.5_y-0.1'

dataset_type_to_load = 'fixrsvp'  # Change to 'fixrsvp', 'gaborium', etc.
eye_calibration_to_load = 'left_eye_x-0.5_y-0.3'  # or 'right_eye_x-0.5_y-0.1'

# Find the matching dataset
dataset_key = f"{dataset_type_to_load} ({eye_calibration_to_load})"

print(f"\nAttempting to load: {dataset_key}")
print(f"Available datasets:")
for key in sorted(available_datasets.keys()):
    print(f"  {key}")

if dataset_key in available_datasets:
    info = available_datasets[dataset_key]
    print(f"\n" + "="*60)
    print(f"Loading: {dataset_key}")
    
    try:
        print(f"Loading from: {info['path']}")
        dset_new = DictDataset.load(info['path'])
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"Dataset length: {len(dset_new)}")
        print(f"Available keys in data: {list(dset_new.keys())}")
        
        if 'stim' in dset_new:
            print(f"Stimulus shape: {dset_new['stim'].shape}")
        if 'robs' in dset_new:
            print(f"Response shape: {dset_new['robs'].shape}")
        if 'eyepos' in dset_new:
            print(f"Eye position shape: {dset_new['eyepos'].shape}")
        
        # Update dset to the new one for inspection in next cell
        dset = dset_new
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n✗ Dataset '{dataset_key}' not found!")
    print(f"Available options:")
    for key in sorted(available_datasets.keys()):
        print(f"  {key}")


#%% Load fixRSVP dataset for reliability analysis
# Load the fixRSVP dataset instead of backimage
fixrsvp_key = None
for key in sorted(available_datasets.keys()):
    if 'fixrsvp' in key.lower() and 'left_eye' in key:
        fixrsvp_key = key
        break

if fixrsvp_key:
    print(f"\n" + "="*60)
    print(f"Loading fixRSVP dataset: {fixrsvp_key}")
    info = available_datasets[fixrsvp_key]
    
    try:
        print(f"Loading from: {info['path']}")
        dset_fixrsvp = DictDataset.load(info['path'])
        
        print(f"\n✓ fixRSVP dataset loaded successfully!")
        print(f"Dataset length: {len(dset_fixrsvp)}")
        print(f"Available keys in data: {list(dset_fixrsvp.keys())}")
        
        if 'stim' in dset_fixrsvp:
            print(f"Stimulus shape: {dset_fixrsvp['stim'].shape}")
        if 'robs' in dset_fixrsvp:
            print(f"Response shape: {dset_fixrsvp['robs'].shape}")
        
        # Update dset to fixRSVP for next analysis
        dset = dset_fixrsvp
        print(f"\n✓ dset updated to fixRSVP for analysis")
        
    except Exception as e:
        print(f"✗ Error loading fixRSVP dataset: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Could not find fixRSVP dataset!")


#%% Compute per-neuron split-half reliability (fixRSVP data quality assessment)
"""
Compute split-half reliability by:
1. Split trials into first and second half of fixRSVP session
2. Correlate firing rates across all timepoints between halves
3. Get per-neuron reliability scores (ccmax ceiling)
"""

if dset is not None:
    print("\n" + "="*60)
    print("Computing per-neuron split-half reliability (fixRSVP)")
    print("="*60)
    
    # Extract data
    robs = dset['robs'].numpy()  # (samples, neurons)
    trial_inds = dset['trial_inds'].numpy()  # (samples,)
    num_neurons = robs.shape[1]
    unique_trials = np.unique(trial_inds)
    
    print(f"\nData shape: {robs.shape}")
    print(f"Number of trials: {len(unique_trials)}")
    print(f"Computing reliability for {num_neurons} neurons...")
    
    # For each neuron, compute split-half reliability across trials
    # For each trial, split timepoints into two halves and store their means
    # Then correlate the means across trials
    split_half_cc_per_neuron = np.zeros(num_neurons)
    
    np.random.seed(42)  # For reproducibility
    
    for cc in range(num_neurons):
        half0_means = []  # Mean firing rate in first half for each trial
        half1_means = []  # Mean firing rate in second half for each trial
        
        for trial in unique_trials:
            trial_mask = trial_inds == trial
            trial_data = robs[trial_mask, cc]  # (T,)
            
            # Split timepoints randomly
            n_timepoints = len(trial_data)
            if n_timepoints < 4:
                continue
            
            split_idx = np.random.choice([0, 1], size=n_timepoints, p=[0.5, 0.5])
            
            half0_mean = trial_data[split_idx == 0].mean()
            half1_mean = trial_data[split_idx == 1].mean()
            
            half0_means.append(half0_mean)
            half1_means.append(half1_mean)
        
        # Correlate the two halves across trials
        if len(half0_means) > 2:  # Need at least 3 data points to correlate
            cc_trial = np.corrcoef(half0_means, half1_means)[0, 1]
            if not np.isnan(cc_trial):
                split_half_cc_per_neuron[cc] = cc_trial
            else:
                split_half_cc_per_neuron[cc] = np.nan
        else:
            split_half_cc_per_neuron[cc] = np.nan
    
    # Correct for attenuation using Spearman-Brown formula
    split_half_cc_per_neuron_corrected = 2 * split_half_cc_per_neuron / (1 + split_half_cc_per_neuron)
    
    # Remove NaNs for stats
    valid_mask = ~np.isnan(split_half_cc_per_neuron)
    valid_cc = split_half_cc_per_neuron[valid_mask]
    valid_cc_corrected = split_half_cc_per_neuron_corrected[valid_mask]
    
    print(f"\nSplit-half reliability (raw):")
    print(f"  Mean: {np.mean(valid_cc):.3f}")
    print(f"  Median: {np.median(valid_cc):.3f}")
    print(f"  Std: {np.std(valid_cc):.3f}")
    print(f"  Range: [{np.min(valid_cc):.3f}, {np.max(valid_cc):.3f}]")
    
    print(f"\nSplit-half reliability (Spearman-Brown corrected):")
    print(f"  Mean: {np.mean(valid_cc_corrected):.3f} ← Overall ccmax ceiling")
    print(f"  Median: {np.median(valid_cc_corrected):.3f}")
    print(f"  Std: {np.std(valid_cc_corrected):.3f}")
    print(f"  Range: [{np.min(valid_cc_corrected):.3f}, {np.max(valid_cc_corrected):.3f}]")
    print(f"  Valid neurons: {valid_mask.sum()} / {num_neurons}")
    
    # Identify neuron tiers (on valid neurons only)
    q75 = np.percentile(valid_cc_corrected, 75)
    q25 = np.percentile(valid_cc_corrected, 25)
    
    high_quality_mask = valid_cc_corrected >= q75
    low_quality_mask = valid_cc_corrected <= q25
    
    print(f"\nNeuron quality tiers:")
    print(f"  Top 25% (high quality): {high_quality_mask.sum()} neurons (CC >= {q75:.3f})")
    print(f"  Bottom 25% (low quality): {low_quality_mask.sum()} neurons (CC <= {q25:.3f})")
    
    # Create comprehensive plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of raw correlations
    axs[0, 0].hist(valid_cc, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axs[0, 0].axvline(np.mean(valid_cc), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_cc):.3f}')
    axs[0, 0].axvline(np.median(valid_cc), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_cc):.3f}')
    axs[0, 0].set_xlabel('Split-half correlation (raw)')
    axs[0, 0].set_ylabel('Number of neurons')
    axs[0, 0].set_title('Raw split-half reliability distribution')
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # Plot 2: Histogram of corrected correlations
    axs[0, 1].hist(valid_cc_corrected, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axs[0, 1].axvline(np.mean(valid_cc_corrected), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_cc_corrected):.3f}')
    axs[0, 1].axvline(np.median(valid_cc_corrected), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_cc_corrected):.3f}')
    axs[0, 1].set_xlabel('Split-half correlation (Spearman-Brown corrected)')
    axs[0, 1].set_ylabel('Number of neurons')
    axs[0, 1].set_title('Corrected split-half reliability (ccmax per neuron)')
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)
    
    # Plot 3: Sorted reliability per neuron
    sorted_idx = np.argsort(valid_cc_corrected)[::-1]
    sorted_cc_corrected = valid_cc_corrected[sorted_idx]
    axs[1, 0].plot(sorted_cc_corrected, linewidth=0.5, alpha=0.8)
    axs[1, 0].axhline(np.mean(valid_cc_corrected), color='r', linestyle='--', linewidth=2, label=f'Mean')
    axs[1, 0].fill_between(range(len(sorted_cc_corrected)), np.percentile(valid_cc_corrected, 25), np.percentile(valid_cc_corrected, 75), alpha=0.2, label='IQR')
    axs[1, 0].set_xlabel('Neuron index (sorted by reliability)')
    axs[1, 0].set_ylabel('Split-half correlation (corrected)')
    axs[1, 0].set_title('Per-neuron reliability (sorted)')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Plot 4: Raw vs corrected scatter
    axs[1, 1].scatter(valid_cc, valid_cc_corrected, alpha=0.5, s=20, color='gray', label='All neurons')
    axs[1, 1].scatter(valid_cc[high_quality_mask], valid_cc_corrected[high_quality_mask], alpha=0.8, s=30, color='green', label='Top 25%')
    axs[1, 1].scatter(valid_cc[low_quality_mask], valid_cc_corrected[low_quality_mask], alpha=0.8, s=30, color='red', label='Bottom 25%')
    axs[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1, 1].axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1, 1].set_xlabel('Raw split-half correlation')
    axs[1, 1].set_ylabel('Spearman-Brown corrected')
    axs[1, 1].set_title('Raw vs. corrected correlations')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Identify high vs low quality neurons (duplicate - already done above)
    # Note: q75, q25, high_quality_mask, low_quality_mask already defined above
    
    print("\n💡 Interpretation:")
    print("  - Each point represents one neuron's reliability (ccmax ceiling)")
    print("  - Higher values = more reliable neurons (less noise)")
    print("  - Use high-quality neurons for downstream modeling")
    print("  - Low-quality neurons may be noisy or have low SNR")
    print(f"  - You can achieve up to {np.mean(valid_cc_corrected):.1%} correlation with a perfect model")
else:
    print("No dataset loaded - cannot compute reliability metrics")


#%%

#%% Compute per-neuron split-half reliability (session halves with trial matching)
"""
Compute split-half reliability by:
1. Split trials into first and second half of session
2. Match trials by length between halves
3. Crop longer trial to match shorter one
4. Correlate firing rates across all timepoints between halves
"""

if dset is not None:
    print("\n" + "="*60)
    print("Computing per-neuron split-half reliability (session halves)")
    print("="*60)
    
    # Extract data
    robs = dset['robs'].numpy()  # (samples, neurons)
    trial_inds = dset['trial_inds'].numpy()  # (samples,)
    
    unique_trials = np.unique(trial_inds)
    num_neurons = robs.shape[1]
    n_trials = len(unique_trials)
    
    # Split trials into first and second half of session
    split_point = n_trials // 2
    trials_half0 = unique_trials[:split_point]
    trials_half1 = unique_trials[split_point:]
    
    print(f"Total trials: {n_trials}")
    print(f"Half 0 trials: {len(trials_half0)} (trials {unique_trials[0]}-{trials_half0[-1]})")
    print(f"Half 1 trials: {len(trials_half1)} (trials {trials_half1[0]}-{unique_trials[-1]})")
    
    # Get trial lengths for matching
    trial_lengths_half0 = {}
    trial_lengths_half1 = {}
    
    for trial in trials_half0:
        trial_lengths_half0[trial] = np.sum(trial_inds == trial)
    
    for trial in trials_half1:
        trial_lengths_half1[trial] = np.sum(trial_inds == trial)
    
    # Match trials by length (nearest neighbor in terms of duration)
    matched_pairs = []
    for trial0 in trials_half0:
        len0 = trial_lengths_half0[trial0]
        # Find trial in half1 with closest length
        best_trial1 = min(trials_half1, key=lambda t: abs(trial_lengths_half1[t] - len0))
        matched_pairs.append((trial0, best_trial1))
    
    print(f"\nMatched {len(matched_pairs)} trial pairs between halves")
    
    # Collect matched data with individual cropping per pair
    robs_half0_matched = []
    robs_half1_matched = []
    
    for trial0, trial1 in matched_pairs:
        # Get data for each trial
        data0 = robs[trial_inds == trial0]  # (T0, NC)
        data1 = robs[trial_inds == trial1]  # (T1, NC)
        
        # Crop each pair to the shorter length
        min_len_pair = min(len(data0), len(data1))
        
        robs_half0_matched.append(data0[:min_len_pair])
        robs_half1_matched.append(data1[:min_len_pair])
    
    # Concatenate all matched data
    robs_half0 = np.concatenate(robs_half0_matched, axis=0)  # (T_total, NC)
    robs_half1 = np.concatenate(robs_half1_matched, axis=0)  # (T_total, NC)
    
    print(f"Half 0 total timepoints after matching: {len(robs_half0)}")
    print(f"Half 1 total timepoints after matching: {len(robs_half1)}")
    
    # Compute correlation per neuron across all timepoints
    split_half_cc = np.array([
        np.corrcoef(robs_half0[:, cc], robs_half1[:, cc])[0, 1]
        for cc in range(num_neurons)
    ])
    
    # Correct for attenuation (split-half correlation is conservative)
    # Using Spearman-Brown formula: r_full = 2*r_half / (1 + r_half)
    split_half_cc_corrected = 2 * split_half_cc / (1 + split_half_cc)
    
    # Summary statistics
    valid_cc = split_half_cc[~np.isnan(split_half_cc)]
    valid_cc_corrected = split_half_cc_corrected[~np.isnan(split_half_cc_corrected)]
    
    print(f"\n{'='*60}")
    print(f"Split-half reliability (raw):")
    print(f"  Mean: {np.mean(valid_cc):.3f}")
    print(f"  Median: {np.median(valid_cc):.3f}")
    print(f"  Std: {np.std(valid_cc):.3f}")
    print(f"  Range: [{np.min(valid_cc):.3f}, {np.max(valid_cc):.3f}]")
    
    print(f"\nSplit-half reliability (Spearman-Brown corrected):")
    print(f"  Mean: {np.mean(valid_cc_corrected):.3f}")
    print(f"  Median: {np.median(valid_cc_corrected):.3f}")
    print(f"  Std: {np.std(valid_cc_corrected):.3f}")
    print(f"  Range: [{np.min(valid_cc_corrected):.3f}, {np.max(valid_cc_corrected):.3f}]")
    print(f"  (This is your per-neuron 'ccmax' ceiling)")
    
    # Identify neuron tiers (using only valid neurons)
    q75 = np.percentile(valid_cc_corrected, 75)
    q25 = np.percentile(valid_cc_corrected, 25)
    
    high_quality_mask = valid_cc_corrected >= q75
    low_quality_mask = valid_cc_corrected <= q25
    
    print(f"\nNeuron quality tiers:")
    print(f"  Top 25% (high quality): {high_quality_mask.sum()} neurons (CC >= {q75:.3f})")
    print(f"  Bottom 25% (low quality): {low_quality_mask.sum()} neurons (CC <= {q25:.3f})")
    
    # Create 4-panel plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Histogram of raw correlations
    axs[0, 0].hist(valid_cc, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axs[0, 0].axvline(np.mean(valid_cc), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_cc):.3f}')
    axs[0, 0].axvline(np.median(valid_cc), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_cc):.3f}')
    axs[0, 0].set_xlabel('Split-half correlation (raw)')
    axs[0, 0].set_ylabel('Number of neurons')
    axs[0, 0].set_title('Raw split-half reliability across neurons')
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # Panel 2: Histogram of corrected correlations
    axs[0, 1].hist(valid_cc_corrected, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axs[0, 1].axvline(np.mean(valid_cc_corrected), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_cc_corrected):.3f}')
    axs[0, 1].axvline(np.median(valid_cc_corrected), color='darkred', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_cc_corrected):.3f}')
    axs[0, 1].axvline(q75, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Q75: {q75:.3f}')
    axs[0, 1].axvline(q25, color='purple', linestyle=':', linewidth=2, alpha=0.7, label=f'Q25: {q25:.3f}')
    axs[0, 1].set_xlabel('Split-half correlation (Spearman-Brown corrected)')
    axs[0, 1].set_ylabel('Number of neurons')
    axs[0, 1].set_title('Corrected split-half reliability (≈ per-neuron ccmax)')
    axs[0, 1].legend(fontsize=9)
    axs[0, 1].grid(alpha=0.3)
    
    # Panel 3: Sorted reliability
    sorted_idx = np.argsort(split_half_cc_corrected)[::-1]
    sorted_cc = split_half_cc_corrected[sorted_idx]
    axs[1, 0].plot(sorted_cc, linewidth=1, color='steelblue')
    axs[1, 0].axhline(np.mean(valid_cc_corrected), color='r', linestyle='--', linewidth=2, label='Mean')
    axs[1, 0].axhline(q75, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Q75')
    axs[1, 0].axhline(q25, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='Q25')
    axs[1, 0].set_xlabel('Neuron (sorted by reliability)')
    axs[1, 0].set_ylabel('Corrected split-half CC')
    axs[1, 0].set_title('Per-neuron reliability ranking')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Panel 4: Raw vs. Corrected scatter
    axs[1, 1].scatter(valid_cc, valid_cc_corrected, alpha=0.5, s=20, color='gray', label='All neurons')
    axs[1, 1].scatter(valid_cc[high_quality_mask], valid_cc_corrected[high_quality_mask], alpha=0.8, s=30, color='green', label='Top 25%')
    axs[1, 1].scatter(valid_cc[low_quality_mask], valid_cc_corrected[low_quality_mask], alpha=0.8, s=30, color='red', label='Bottom 25%')
    axs[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1, 1].axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1, 1].set_xlabel('Raw split-half correlation')
    axs[1, 1].set_ylabel('Corrected split-half correlation')
    axs[1, 1].set_title('Raw vs. Corrected reliability')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Interpretation:")
    print("  - Each point represents one neuron's reliability (ccmax ceiling)")
    print("  - Higher values = more reliable neurons (less noise)")
    print("  - Use high-quality neurons for downstream modeling")
    print("  - Low-quality neurons may be noisy or have low SNR")
    print(f"  - You can achieve up to {np.mean(valid_cc_corrected):.1%} correlation with a perfect model")
    
else:
    print("No dataset loaded - cannot compute reliability metrics")

#%% Plot spike rasters for best-quality neurons (sorted by trial length)
"""
Visualize spike rasters for high-quality neurons across all trials.
Trials are sorted by duration to see trial-to-trial variability.
"""

if dset is not None and 'high_quality_mask' in locals():
    print("\n" + "="*60)
    print("Plotting spike rasters for high-quality neurons")
    print("="*60)
    
    # Extract data
    robs = dset['robs'].numpy()  # (samples, neurons)
    trial_inds = dset['trial_inds'].numpy()  # (samples,)
    
    unique_trials = np.unique(trial_inds)
    
    # Get best neurons (top 25%)
    best_neurons = np.where(high_quality_mask)[0]
    n_best = min(5, len(best_neurons))  # Show up to 5 best neurons
    best_neurons = best_neurons[:n_best]
    
    print(f"Plotting {len(best_neurons)} best-quality neurons")
    print(f"Neuron IDs: {best_neurons}")
    
    # Compute trial lengths and sort
    trial_lengths = np.array([np.sum(trial_inds == trial) for trial in unique_trials])
    sorted_trial_idx = np.argsort(trial_lengths)[::-1]  # Sort descending
    sorted_trials = unique_trials[sorted_trial_idx]
    sorted_lengths = trial_lengths[sorted_trial_idx]
    
    print(f"Trial lengths: min={sorted_lengths.min()}, max={sorted_lengths.max()}, mean={sorted_lengths.mean():.0f}")
    
    # Create figure with subplots for each neuron
    n_rows = len(best_neurons)
    fig, axs = plt.subplots(n_rows, 1, figsize=(16, 3*n_rows))
    if n_rows == 1:
        axs = [axs]
    
    for idx, neuron_id in enumerate(best_neurons):
        ax = axs[idx]
        
        # Extract spikes for this neuron across all trials
        y_pos = 0
        trial_starts = []
        
        for trial_num, trial in enumerate(sorted_trials):
            trial_mask = trial_inds == trial
            spikes = robs[trial_mask, neuron_id]
            
            # Find spike times (above threshold, e.g., > 0.1 for spike count)
            spike_times = np.where(spikes > 0.1)[0]
            
            if len(spike_times) > 0:
                # Plot spikes as vertical lines
                for spike_time in spike_times:
                    ax.vlines(spike_time, y_pos - 0.4, y_pos + 0.4, colors='black', linewidth=2.0, alpha=0.7)
            
            # Mark trial boundaries
            trial_starts.append(y_pos)
            y_pos += 1
        
        ax.set_ylim(-0.5, len(sorted_trials) - 0.5)
        ax.set_ylabel('Trial (sorted by length)', fontsize=10)
        ax.set_title(f'Neuron {neuron_id} (CC = {valid_cc_corrected[best_neurons[idx]]:.3f})', fontsize=11, fontweight='bold')
        ax.set_yticks([0, len(sorted_trials)//2, len(sorted_trials)-1])
        ax.set_yticklabels([f'{sorted_lengths[0]:.0f}', f'{sorted_lengths[len(sorted_trials)//2]:.0f}', f'{sorted_lengths[-1]:.0f}'])
        ax.grid(alpha=0.2, axis='x')
        
        if idx == n_rows - 1:
            ax.set_xlabel('Time within trial (samples)', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ Raster plot complete")
    print(f"  Trials sorted by duration (top to bottom: long → short)")
    print(f"  Black marks indicate spike times")
    print(f"  Y-axis shows trial duration in samples")

else:
    print("Cannot plot rasters - dataset or quality metrics not available")

#%% Compute ccmax via PSTH-based split-half (like mcfarland_sim)
"""
Compute ccmax using PSTH-based split-half resampling:
1. Bin spike times into discrete time bins
2. Create PSTH: average firing rate per time bin across trials
3. Randomly split trials into two halves
4. Correlate the two PSTHs to estimate ccmax
"""

if dset is not None:
    print("\n" + "="*60)
    print("Computing ccmax via PSTH-based split-half")
    print("="*60)
    
    # Extract data
    robs = dset['robs'].numpy()  # (samples, neurons)
    trial_inds = dset['trial_inds'].numpy()  # (samples,)
    
    unique_trials = np.unique(trial_inds)
    num_neurons = robs.shape[1]
    n_trials = len(unique_trials)
    
    # Bin time parameter (in samples; adjust based on your sampling rate)
    time_bin_size = 10  # Group 10 samples per bin
    
    print(f"\nData shape: {robs.shape}")
    print(f"Number of trials: {n_trials}")
    print(f"Time bin size: {time_bin_size} samples")
    
    # Build PSTH matrix: (trials, time_bins, neurons)
    psth_list = []
    trial_lengths = []
    
    for trial in unique_trials:
        trial_mask = trial_inds == trial
        trial_data = robs[trial_mask]  # (T_trial, NC)
        trial_lengths.append(len(trial_data))
        
        # Bin the trial data
        n_bins = (len(trial_data) + time_bin_size - 1) // time_bin_size
        trial_psth = np.zeros((n_bins, num_neurons))
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * time_bin_size
            end_idx = min((bin_idx + 1) * time_bin_size, len(trial_data))
            trial_psth[bin_idx] = trial_data[start_idx:end_idx].mean(axis=0)
        
        psth_list.append(trial_psth)
    
    print(f"Trial lengths: min={min(trial_lengths)}, max={max(trial_lengths)}, mean={np.mean(trial_lengths):.0f}")
    
    # Pad all PSTHs to same length (maximum trial length)
    max_bins = max([p.shape[0] for p in psth_list])
    psth_array = np.full((n_trials, max_bins, num_neurons), np.nan)
    
    for i, psth in enumerate(psth_list):
        psth_array[i, :psth.shape[0], :] = psth
    
    print(f"PSTH array shape: {psth_array.shape} (trials × time_bins × neurons)")
    
    # Create valid-sample mask (not NaN)
    valid_mask = ~np.isnan(psth_array)
    
    # Randomly split trials into two halves (multiple splits for averaging)
    n_splits = 100
    np.random.seed(42)
    
    ccmax_per_neuron = np.zeros(num_neurons)
    ccmax_per_neuron_raw = np.zeros(num_neurons)
    
    for neuron_idx in range(num_neurons):
        split_correlations = []
        
        for split_num in range(n_splits):
            # Random trial split
            perm = np.random.permutation(n_trials)
            split_A = perm[:n_trials // 2]
            split_B = perm[n_trials // 2:]
            
            # Get PSTH for each half
            psth_A = psth_array[split_A, :, neuron_idx]  # (NA, T)
            psth_B = psth_array[split_B, :, neuron_idx]  # (NB, T)
            valid_A = valid_mask[split_A, :, neuron_idx]
            valid_B = valid_mask[split_B, :, neuron_idx]
            
            # Average across trials per time bin
            mean_A = np.nanmean(np.where(valid_A, psth_A, np.nan), axis=0)  # (T,)
            mean_B = np.nanmean(np.where(valid_B, psth_B, np.nan), axis=0)  # (T,)
            
            # Find time bins with valid data in both halves
            valid_t = np.isfinite(mean_A) & np.isfinite(mean_B)
            
            if valid_t.sum() >= 10:  # Need at least 10 time bins
                # Correlate the two PSTHs
                cc = np.corrcoef(mean_A[valid_t], mean_B[valid_t])[0, 1]
                if np.isfinite(cc):
                    split_correlations.append(cc)
        
        if len(split_correlations) > 0:
            # Average split-half correlation
            cchalf = np.mean(split_correlations)
            
            # Convert to ccmax using Spearman-Brown formula
            # ccmax = sqrt(2 * cchalf / (1 + cchalf))
            if cchalf > 0:
                ccmax_per_neuron_raw[neuron_idx] = cchalf
                ccmax_per_neuron[neuron_idx] = np.sqrt((2.0 * cchalf) / (1.0 + cchalf))
            else:
                ccmax_per_neuron[neuron_idx] = np.nan
        else:
            ccmax_per_neuron[neuron_idx] = np.nan
    
    # Summary statistics
    valid_ccmax = ccmax_per_neuron[~np.isnan(ccmax_per_neuron)]
    valid_cchalf = ccmax_per_neuron_raw[~np.isnan(ccmax_per_neuron_raw)]
    
    print(f"\n{'='*60}")
    print(f"PSTH-based split-half ccmax (raw):")
    print(f"  Mean: {np.mean(valid_cchalf):.3f}")
    print(f"  Median: {np.median(valid_cchalf):.3f}")
    print(f"  Range: [{np.min(valid_cchalf):.3f}, {np.max(valid_cchalf):.3f}]")
    
    print(f"\nPSTH-based ccmax (Spearman-Brown corrected):")
    print(f"  Mean: {np.mean(valid_ccmax):.3f}")
    print(f"  Median: {np.median(valid_ccmax):.3f}")
    print(f"  Std: {np.std(valid_ccmax):.3f}")
    print(f"  Range: [{np.min(valid_ccmax):.3f}, {np.max(valid_ccmax):.3f}]")
    print(f"  Valid neurons: {valid_mask.shape[2]}")
    
    # Plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axs[0].hist(valid_ccmax, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axs[0].axvline(np.mean(valid_ccmax), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_ccmax):.3f}')
    axs[0].set_xlabel('ccmax (Spearman-Brown corrected)')
    axs[0].set_ylabel('Number of neurons')
    axs[0].set_title(f'PSTH-based ccmax distribution ({n_splits} splits)')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    # Sorted values
    sorted_ccmax = np.sort(valid_ccmax)[::-1]
    axs[1].plot(sorted_ccmax, linewidth=1.5)
    axs[1].axhline(np.mean(valid_ccmax), color='r', linestyle='--', linewidth=2, label='Mean')
    axs[1].fill_between(range(len(sorted_ccmax)), np.percentile(valid_ccmax, 25), np.percentile(valid_ccmax, 75), alpha=0.2, label='IQR')
    axs[1].set_xlabel('Neuron (sorted by ccmax)')
    axs[1].set_ylabel('ccmax')
    axs[1].set_title('Per-neuron ccmax ranking')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ PSTH-based ccmax computation complete")
    print(f"  Time bin size: {time_bin_size} samples")
    print(f"  Number of random splits: {n_splits}")
    print(f"  This follows the mcfarland_sim approach more closely")

    # save results for later comparison (figure)
    # save to eg: ./figures/rowleydatatest-ccmax-PSTHLuke0804
    # insert session name into filename
    figures_dir = Path("../figures")
    figure_save_path = figures_dir / f"rowleydatatest-ccmax-PSTH-{sess.name}.png"
    fig.savefig(figure_save_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {figure_save_path}")

    # save high-ccmax cids for downstream McFarland analysis
    try:
        # align ccmax values to dataset neuron IDs (cids)
        cids = np.array(dset.metadata.get('cids', np.arange(num_neurons)))
    except Exception:
        cids = np.arange(num_neurons)

    # threshold on full ccmax_per_neuron (NaNs ignored) to get top 25%
    ccmax_all = ccmax_per_neuron.copy()
    high_thresh = .85#np.nanpercentile(ccmax_all, 75)
    high_cc_mask_all = ccmax_all >= high_thresh
    high_cc_cids = cids[high_cc_mask_all]

    high_cc_path = figures_dir / f"{sess.name}_high_ccmax_cids.npy"
    np.save(high_cc_path, high_cc_cids)
    print(f"  Saved high-ccmax cids (N={high_cc_cids.size}) to: {high_cc_path}")

else:
    print("No dataset loaded - cannot compute PSTH-based ccmax")




#%% Check against an old DataYates session
# The session directory is '/mnt/ssd/YatesMarmoV1/processed/Allen_2022-03-04/'
# Extract subject and date from the path

from DataYatesV1 import get_session
subject = 'Allen'
date = '2022-03-04'

# Create session object
sess = get_session(subject, date)

#%% Load a DataYatesV1 session for comparison
"""
Load a DataYatesV1 (old) session to compare ccmax with Rowley data.
Note: DataYatesV1 sessions have different structure than Rowley sessions.
"""

from DataYatesV1 import get_session as get_yates_session
from DataYatesV1 import DictDataset

# Load YatesV1 session
subject = 'Allen'
date = '2022-03-04'

sess_yates = get_yates_session(subject, date)
print(f"YatesV1 Session loaded: {sess_yates.name}")
print(f"Session directory: {sess_yates.sess_dir}")

# List available methods
print(f"\nKey attributes:")
print(f"  sess_dir: {sess_yates.sess_dir}")
print(f"  exp: Experiment metadata")
print(f"  get_dataset(): Load preprocessed datasets")
print(f"  get_spike_times(): Load spike data")

# Look for datasets in this session
print(f"\n" + "="*60)
print("Looking for available datasets in YatesV1 session...")

# DataYatesV1 sessions typically have datasets in a similar structure
datasets_dir = Path(sess_yates.sess_dir) / 'datasets'

if datasets_dir.exists():
    print(f"Found datasets directory: {datasets_dir}")
    yates_available_datasets = {}
    
    for dataset_path in sorted(datasets_dir.glob('**/*.dset')):
        dataset_type = dataset_path.stem
        dataset_dir = dataset_path.parent.name
        key = f"{dataset_type} ({dataset_dir})"
        yates_available_datasets[key] = {
            'type': dataset_type,
            'path': dataset_path
        }
        print(f"  {key}")
    
    # Load fixRSVP if available
    fixrsvp_key = None
    for key in yates_available_datasets.keys():
        if 'fixrsvp' in key.lower():
            fixrsvp_key = key
            break
    
    if fixrsvp_key:
        print(f"\n" + "="*60)
        print(f"Loading YatesV1 fixRSVP: {fixrsvp_key}")
        
        try:
            yates_path = yates_available_datasets[fixrsvp_key]['path']
            dset_yates = DictDataset.load(yates_path)
            
            print(f"✓ Loaded successfully!")
            print(f"  Dataset length: {len(dset_yates)}")
            print(f"  Response shape: {dset_yates['robs'].shape}")
            print(f"  Trial count: {len(np.unique(dset_yates['trial_inds'].numpy()))}")
            
        except Exception as e:
            print(f"✗ Error loading: {e}")
            dset_yates = None
    else:
        print("No fixRSVP dataset found in YatesV1 session")
        dset_yates = None
else:
    print(f"No datasets directory found at {datasets_dir}")
    dset_yates = None

print(f"\n💡 Key differences between DataYatesV1 and DataRowleyV1V2:")
print(f"  - YatesV1 uses 'sess_dir' instead of 'processed_path'")
print(f"  - YatesV1 has 'get_spike_times()' method")
print(f"  - Both use similar '.dset' file format")
print(f"  - DataFramework is compatible across both")



#%% Compute ccmax via PSTH-based split-half (like mcfarland_sim)
# run on YatesV1 dataset for comparison, same as above, but on dset_yates
# slightly different colormapping for plots
"""
Compute ccmax using PSTH-based split-half resampling:
1. Bin spike times into discrete time bins
2. Create PSTH: average firing rate per time bin across trials
3. Randomly split trials into two halves
4. Correlate the two PSTHs to estimate ccmax
"""

if dset_yates is not None:
    print("\n" + "="*60)
    print("Computing ccmax via PSTH-based split-half")
    print("="*60)
    
    # Extract data
    robs = dset_yates['robs'].numpy()  # (samples, neurons)
    trial_inds = dset_yates['trial_inds'].numpy()  # (samples,)
    
    unique_trials = np.unique(trial_inds)
    num_neurons = robs.shape[1]
    n_trials = len(unique_trials)
    
    # Bin time parameter (in samples; adjust based on your sampling rate)
    time_bin_size = 10  # Group 10 samples per bin
    
    print(f"\nData shape: {robs.shape}")
    print(f"Number of trials: {n_trials}")
    print(f"Time bin size: {time_bin_size} samples")
    
    # Build PSTH matrix: (trials, time_bins, neurons)
    psth_list = []
    trial_lengths = []
    
    for trial in unique_trials:
        trial_mask = trial_inds == trial
        trial_data = robs[trial_mask]  # (T_trial, NC)
        trial_lengths.append(len(trial_data))
        
        # Bin the trial data
        n_bins = (len(trial_data) + time_bin_size - 1) // time_bin_size
        trial_psth = np.zeros((n_bins, num_neurons))
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * time_bin_size
            end_idx = min((bin_idx + 1) * time_bin_size, len(trial_data))
            trial_psth[bin_idx] = trial_data[start_idx:end_idx].mean(axis=0)
        
        psth_list.append(trial_psth)
    
    print(f"Trial lengths: min={min(trial_lengths)}, max={max(trial_lengths)}, mean={np.mean(trial_lengths):.0f}")
    
    # Pad all PSTHs to same length (maximum trial length)
    max_bins = max([p.shape[0] for p in psth_list])
    psth_array = np.full((n_trials, max_bins, num_neurons), np.nan)
    
    for i, psth in enumerate(psth_list):
        psth_array[i, :psth.shape[0], :] = psth
    
    print(f"PSTH array shape: {psth_array.shape} (trials × time_bins × neurons)")
    
    # Create valid-sample mask (not NaN)
    valid_mask = ~np.isnan(psth_array)
    
    # Randomly split trials into two halves (multiple splits for averaging)
    n_splits = 100
    np.random.seed(42)
    
    ccmax_per_neuron = np.zeros(num_neurons)
    ccmax_per_neuron_raw = np.zeros(num_neurons)
    
    for neuron_idx in range(num_neurons):
        split_correlations = []
        
        for split_num in range(n_splits):
            # Random trial split
            perm = np.random.permutation(n_trials)
            split_A = perm[:n_trials // 2]
            split_B = perm[n_trials // 2:]
            
            # Get PSTH for each half
            psth_A = psth_array[split_A, :, neuron_idx]  # (NA, T)
            psth_B = psth_array[split_B, :, neuron_idx]  # (NB, T)
            valid_A = valid_mask[split_A, :, neuron_idx]
            valid_B = valid_mask[split_B, :, neuron_idx]
            
            # Average across trials per time bin
            mean_A = np.nanmean(np.where(valid_A, psth_A, np.nan), axis=0)  # (T,)
            mean_B = np.nanmean(np.where(valid_B, psth_B, np.nan), axis=0)  # (T,)
            
            # Find time bins with valid data in both halves
            valid_t = np.isfinite(mean_A) & np.isfinite(mean_B)
            
            if valid_t.sum() >= 10:  # Need at least 10 time bins
                # Correlate the two PSTHs
                cc = np.corrcoef(mean_A[valid_t], mean_B[valid_t])[0, 1]
                if np.isfinite(cc):
                    split_correlations.append(cc)
        
        if len(split_correlations) > 0:
            # Average split-half correlation
            cchalf = np.mean(split_correlations)
            
            # Convert to ccmax using Spearman-Brown formula
            # ccmax = sqrt(2 * cchalf / (1 + cchalf))
            if cchalf > 0:
                ccmax_per_neuron_raw[neuron_idx] = cchalf
                ccmax_per_neuron[neuron_idx] = np.sqrt((2.0 * cchalf) / (1.0 + cchalf))
            else:
                ccmax_per_neuron[neuron_idx] = np.nan
        else:
            ccmax_per_neuron[neuron_idx] = np.nan
    
    # Summary statistics
    valid_ccmax = ccmax_per_neuron[~np.isnan(ccmax_per_neuron)]
    valid_cchalf = ccmax_per_neuron_raw[~np.isnan(ccmax_per_neuron_raw)]
    
    print(f"\n{'='*60}")
    print(f"PSTH-based split-half ccmax (raw):")
    print(f"  Mean: {np.mean(valid_cchalf):.3f}")
    print(f"  Median: {np.median(valid_cchalf):.3f}")
    print(f"  Range: [{np.min(valid_cchalf):.3f}, {np.max(valid_cchalf):.3f}]")
    
    print(f"\nPSTH-based ccmax (Spearman-Brown corrected):")
    print(f"  Mean: {np.mean(valid_ccmax):.3f}")
    print(f"  Median: {np.median(valid_ccmax):.3f}")
    print(f"  Std: {np.std(valid_ccmax):.3f}")
    print(f"  Range: [{np.min(valid_ccmax):.3f}, {np.max(valid_ccmax):.3f}]")
    print(f"  Valid neurons: {valid_mask.shape[2]}")
    
    # Plot comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axs[0].hist(valid_ccmax, bins=30, alpha=0.7, edgecolor='black', color='darkorange')
    axs[0].axvline(np.mean(valid_ccmax), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_ccmax):.3f}')
    axs[0].set_xlabel('ccmax (Spearman-Brown corrected)')
    axs[0].set_ylabel('Number of neurons')
    axs[0].set_title(f'PSTH-based ccmax distribution ({n_splits} splits)')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    # Sorted values
    sorted_ccmax = np.sort(valid_ccmax)[::-1]
    axs[1].plot(sorted_ccmax, linewidth=1.5)
    axs[1].axhline(np.mean(valid_ccmax), color='r', linestyle='--', linewidth=2, label='Mean')
    axs[1].fill_between(range(len(sorted_ccmax)), np.percentile(valid_ccmax, 25), np.percentile(valid_ccmax, 75), alpha=0.2, label='IQR')
    axs[1].set_xlabel('Neuron (sorted by ccmax)')
    axs[1].set_ylabel('ccmax')
    axs[1].set_title('Per-neuron ccmax ranking')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ PSTH-based ccmax computation complete")
    print(f"  Time bin size: {time_bin_size} samples")
    print(f"  Number of random splits: {n_splits}")
    print(f"  This follows the mcfarland_sim approach more closely")

    # save results for later comparison
    # save to eg: ./figures/rowleydatatest-ccmaxPSTHLuke0804
    # insert session name into filename
    figures_dir = Path("../figures")
    figure_save_path =  figures_dir / f"rowleydatatest-ccmax-PSTH-{sess.name}.png"
    fig.savefig(figure_save_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {figure_save_path}")

else:
    print("No dataset loaded - cannot compute PSTH-based ccmax")
