"""
Multidataset Model Evaluation Stack Script

This script loads a trained multidataset neural network model and evaluates its performance.
It provides an interactive interface to:
- Browse and select models by type (resnet, x3d_modulator, etc.)
- Load the best model or select a specific one
- Evaluate performance on individual datasets
- Visualize model weights and architecture
- Analyze predictions vs observations

Based on model_load_eval_stack_jake.py but adapted for multidataset models.

Author: Jake (adapted from Ryan Ressmeyer's original)
"""

#%% Import libraries
# Standard libraries

# suppress compilation
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import os
import re
from pathlib import Path
from pprint import pprint
from collections import defaultdict

# Data processing libraries
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# DataYatesV1 package imports
from DataYatesV1.utils.data import prepare_data
from DataYatesV1 import get_free_device, enable_autoreload
from DataYatesV1.utils.data.loading import remove_pixel_norm
from DataYatesV1.models.losses import MaskedLoss, PoissonBPSAggregator

# Import the training module to access the model class
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_ddp_multidataset import MultiDatasetModel

# Enable auto-reloading of modules for interactive development
enable_autoreload()

#%% Set up environment

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available and set device
device = get_free_device()
print(f"Using device: {device}")

#%% Model Browser Functions

# Import scan_checkpoints from eval_stack_utils
from eval_stack_utils import scan_checkpoints

# Import helper functions from eval_stack_utils
from eval_stack_utils import extract_model_type, extract_val_loss, extract_val_bps, extract_epoch

def list_models_by_type(models_by_type, model_type=None):
    """List models, optionally filtered by type."""
    if model_type and model_type not in models_by_type:
        print(f"❌ Model type '{model_type}' not found.")
        print(f"Available types: {list(models_by_type.keys())}")
        return
    
    types_to_show = [model_type] if model_type else list(models_by_type.keys())
    
    for mtype in types_to_show:
        models = models_by_type[mtype]
        print(f"\n{'='*60}")
        print(f"MODEL TYPE: {mtype.upper()} ({len(models)} models)")
        print(f"{'='*60}")
        print(f"{'#':<3} {'Val Loss':<10} {'Epoch':<6} {'Experiment':<40}")
        print("-" * 60)
        
        for i, model in enumerate(models):
            print(f"{i+1:<3} {model['val_loss']:<10.4f} {model['epoch']:<6} {model['experiment'][:39]}")



#%% Model selection - EDIT THIS SECTION

# Specify the model type and optionally the index (1-based)
model_type = 'learned_res'  # Change this to: 'resnet', 'x3d', 'x3d_modulator', etc.
model_index = None  # None for best model, or 1, 2, 3... for specific model

# Checkpoint directory (relative to current working directory)
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints"  # Adjust if your checkpoints are elsewhere

#%% Load the specified model

print("🔍 Scanning for trained models...")
models_by_type = scan_checkpoints(checkpoint_dir)

list_models_by_type(models_by_type)
model = None
#%%

if not models_by_type:
    print(f"❌ No models found in {checkpoint_dir}")
    print("Please check your checkpoint directory path.")
    model = None
else:
    print(f"✓ Found {sum(len(models) for models in models_by_type.values())} valid checkpoints")
    print(f"  Model types: {list(models_by_type.keys())}")

    # Show available models of the specified type
    if model_type not in models_by_type:
        print(f"\n❌ Model type '{model_type}' not found.")
        print(f"Available types: {list(models_by_type.keys())}")
        print("\nAll available models:")
        list_models_by_type(models_by_type)
        model = None
    else:
        models = models_by_type[model_type]
        print(f"\nAvailable {model_type} models:")
        list_models_by_type(models_by_type, model_type)

        # Select the model
        if model_index is None:
            selected_model = models[0]  # Best model
            print(f"\n🎯 Loading BEST {model_type} model...")
        else:
            if model_index < 1 or model_index > len(models):
                print(f"❌ Model index {model_index} out of range (1-{len(models)}).")
                model = None
                selected_model = None
            else:
                selected_model = models[model_index - 1]
                print(f"\n🎯 Loading {model_type} model #{model_index}...")

        if selected_model is not None:
            checkpoint_path = selected_model['path']
            print(f"   Checkpoint: {checkpoint_path}")
            print(f"   Val Loss: {selected_model['val_loss']:.4f}")
            print(f"   Epoch: {selected_model['epoch']}")

            # Load the model
            try:
                model = MultiDatasetModel.load_from_checkpoint(
                    str(checkpoint_path),
                    strict=False,
                    map_location='cpu'
                )
                model.to(device)
                model.eval()

                print("✓ Model loaded successfully!")

                # Get model info
                print(f"\nModel Information:")
                
                print(f"  Datasets: {len(model.names)}")
                print(f"  Dataset names: {model.names}")
                print(f"  Activation: {type(model.model.activation).__name__}")

                # Count parameters
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                print(f"  Total parameters: {total_params:,}")
                print(f"  Trainable parameters: {trainable_params:,}")

            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                model = None
        else:
            model = None

#%% Helper functions for evaluation

def load_single_dataset(model, dataset_idx):
    """Load a single dataset for evaluation."""
    if dataset_idx >= len(model.names):
        raise ValueError(f"Dataset index {dataset_idx} out of range. Model has {len(model.name)} datasets.")
    
    if hasattr(model, 'dataset_configs'):
        dataset_config = model.dataset_configs[dataset_idx].copy()
    else:
        # combine datasaet config path with dataset name + yaml extension
        config_path = model.hparams.cfg_dir
        dataset_name = model.names[dataset_idx]
        dataset_config_path = Path(config_path) / f"{dataset_name}.yaml"
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

    dataset_name = model.names[dataset_idx]
    
    print(f"\nLoading dataset {dataset_idx}: {dataset_name}")
    # add datafilters if missing
    dataset_config['datafilters'] = {'dfs': {'ops': [{'valid_nlags': {'n_lags': 32}}, {'missing_pct': {'theshold': 45}}], 'expose_as': 'dfs'}}
    
    # # Remove pixel normalization if present
    # dataset_config, pixel_norm_removed = remove_pixel_norm(dataset_config)
    dataset_config['types'] += ['fixrsvp', 'gratings']
    dataset_config['keys_lags']['eyepos'] = 0
    # Load data with suppressed output
    import contextlib
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)
    
    
    print(f"✓ Dataset loaded: {len(train_dset)} train, {len(val_dset)} val samples")
    print(f"  Dataset config: {len(dataset_config.get('cids', []))} units")
    
    return train_dset, val_dset, dataset_config


# Evaluate model on different datasets
import torch.nn as nn

def get_stim_inds(stim_type, train_data, val_data):
    if stim_type == 'gaborium':
        return train_data.get_dataset_inds('gaborium')
    elif stim_type == 'backimage':
        return val_data.get_dataset_inds('backimage')
    elif stim_type == 'fixrsvp':
        return torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)
    elif stim_type == 'gratings':
        return torch.concatenate([
            train_data.get_dataset_inds('gratings'),
            val_data.get_dataset_inds('gratings')
        ], dim=0)
    else:
        raise ValueError(f"Unknown stim type: {stim_type}")


def eval(model, train_data, val_data, didx, recalc = False):

    def run_model(model, batch, dataset_idx):
        """Run the model on a batch of data."""
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        with torch.no_grad():
            output = model.model(batch['stim'], dataset_idx, batch.get('behavior'))
        batch['rhat'] = output
        return batch


    def evaluate_dataset(model, dataset, indices, batch_size=256, desc="Dataset"):
        """
        Evaluate model on a dataset and calculate bits per spike.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        dataset : CombinedEmbeddedDataset
            The dataset to evaluate on.
        indices : torch.Tensor
            The indices to use for evaluation.
        batch_size : int, optional
            Batch size for evaluation, by default 256.
        desc : str, optional
            Description for progress bar, by default "Dataset".

        Returns
        -------
        tuple
            (model predictions, bits per spike)
        """
        bps_aggregator = PoissonBPSAggregator()
        dataset = dataset.shallow_copy()
        dataset.inds = indices
        robs = []
        rhat = []

        with torch.no_grad():
            for iB in tqdm(range(0, len(dataset), batch_size), desc=desc):
                batch = dataset[iB:iB+batch_size]
                batch = run_model(model, batch, didx)
                if model.log_input:
                    batch['rhat'] = torch.exp(batch['rhat'])

                robs.append(batch['robs'].detach().cpu())  # Move to CPU immediately
                rhat.append(batch['rhat'].detach().cpu())  # Move to CPU immediately
                bps_aggregator(batch)

                # Clean up batch tensors to free GPU memory
                del batch
                torch.cuda.empty_cache()

        robs = torch.cat(robs, dim=0)
        rhat = torch.cat(rhat, dim=0)

        bps = bps_aggregator.closure().cpu().numpy()
        bps_aggregator.reset()

        return robs, rhat, bps

    cache_file = save_dir / f'{model_name}_dataset{didx}_eval_cache.pt'
    if recalc and cache_file.exists():
        cache_file.unlink()
    if not cache_file.exists():
        print(f'No evaluation cache found at {cache_file}. Evaluating from scratch...')

        # Gaborium dataset
        gaborium_inds = get_stim_inds('gaborium', train_data, val_data)
        gaborium_robs, gaborium_rhat, gaborium_bps = evaluate_dataset(
            model, train_data, gaborium_inds, batch_size, "Gaborium"
        )

        # Backimage dataset
        backimage_inds = get_stim_inds('backimage', train_data, val_data)
        backimage_robs, backimage_rhat, backimage_bps = evaluate_dataset(
            model, train_data, backimage_inds, batch_size, "Backimage"
        )

        # FixRSVP dataset
        fixrsvp_inds = get_stim_inds('fixrsvp', train_data, val_data)
        fixrsvp_robs, fixrsvp_rhat, fixrsvp_bps = evaluate_dataset(
            model, train_data, fixrsvp_inds, batch_size, "FixRSVP"
        )

        # Gratings dataset
        gratings_inds = get_stim_inds('gratings', train_data, val_data)
        gratings_robs, gratings_rhat, gratings_bps = evaluate_dataset(
            model, train_data, gratings_inds, batch_size, "Gratings"
        )

        # Validation set BPS
        val_bps_aggregator = PoissonBPSAggregator()
        val_bps_aggregator({'robs': gaborium_robs, 'rhat': gaborium_rhat})
        val_bps_aggregator({'robs': backimage_robs, 'rhat': backimage_rhat})
        val_bps = val_bps_aggregator.closure().cpu().numpy()
        val_bps_aggregator.reset()

        # Save evaluation results to cache
        cache = {
            'gaborium_inds': gaborium_inds,
            'gaborium_eval': (gaborium_robs, gaborium_rhat, gaborium_bps),
            'backimage_inds': backimage_inds,
            'backimage_eval': (backimage_robs, backimage_rhat, backimage_bps),
            'fixrsvp_inds': fixrsvp_inds,
            'fixrsvp_eval': (fixrsvp_robs, fixrsvp_rhat, fixrsvp_bps),
            'gratings_inds': gratings_inds,
            'gratings_eval': (gratings_robs, gratings_rhat, gratings_bps),
            'val_bps': val_bps
        }
        torch.save(cache, cache_file)
        print(f'Evaluation cache saved to {cache_file}')
    else:
        print(f'Loading evaluation cache from {cache_file}')
        cache = torch.load(cache_file, weights_only=False)
        gaborium_inds = cache['gaborium_inds']
        gaborium_robs, gaborium_rhat, gaborium_bps = cache['gaborium_eval']
        backimage_inds = cache['backimage_inds']
        backimage_robs, backimage_rhat, backimage_bps = cache['backimage_eval']
        fixrsvp_inds = cache['fixrsvp_inds']
        fixrsvp_robs, fixrsvp_rhat, fixrsvp_bps = cache['fixrsvp_eval']
        gratings_inds = cache['gratings_inds']
        gratings_robs, gratings_rhat, gratings_bps = cache['gratings_eval']
        val_bps = cache['val_bps']

    return {
        'gaborium': (gaborium_robs, gaborium_rhat, gaborium_bps),
        'backimage': (backimage_robs, backimage_rhat, backimage_bps),
        'fixrsvp': (fixrsvp_robs, fixrsvp_rhat, fixrsvp_bps),
        'gratings': (gratings_robs, gratings_rhat, gratings_bps),
        'val': val_bps
    }

#%%


n_total_units = sum([m.n_units for m in model.model.readouts])
print(f"Model trained using {len(model.model.readouts)} datasets with {n_total_units} units total.")

batch_size = 64
recalc = False

print("Evaluating model on each datasets...")
eval_dicts = []
dataset_ids = []
for didx in range(len(model.names)):
    print(f"\nDataset {didx}: {model.names[didx]}")
    
    try:
        train_data, val_data, dataset_config = load_single_dataset(model, didx)

        model_name = selected_model['experiment']
        save_dir = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack") / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        # Evaluate model on different datasets
        eval_dict = eval(model, train_data, val_data, didx, recalc=recalc)
        eval_dicts.append(eval_dict)
        dataset_ids.append(didx)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        # import traceback
        # traceback.print_exc()


#%%

bps = {}
for stim_type in ['gaborium', 'backimage', 'fixrsvp', 'gratings']:
    bps[stim_type] = np.concatenate([b[stim_type][2] for b in eval_dicts])

bins = np.linspace(-.1, 2, 50)
# 2D histogram of bps
plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
cnt, _, _ = np.histogram2d(bps['gaborium'], bps['backimage'], bins=bins)
plt.imshow(np.log1p(cnt.T), origin='lower', extent=[bins[0], bins[-1], bins[0], bins[-1]], interpolation='none', cmap='gray_r')
plt.plot([-.1, 2], [-.1, 2], 'k--')
plt.xlabel('Gaborium BPS')
plt.ylabel('Backimage BPS')
plt.colorbar()

plt.subplot(1,2,2)
cnt, _, _ = np.histogram2d(bps['gratings'], bps['fixrsvp'], bins=bins)
plt.imshow(np.log1p(cnt.T), origin='lower', extent=[bins[0], bins[-1], bins[0], bins[-1]], interpolation='none', cmap='gray_r')
plt.plot([-.1, 2], [-.1, 2], 'k--')
plt.xlabel('Gratings BPS')
plt.ylabel('FixRSVP BPS')
plt.colorbar()

# set wspacing
plt.subplots_adjust(wspace=.25)

#%%
f = lambda x: np.maximum(x, -.1)
plt.figure(figsize=(8, 3.5))
plt.subplot(1,2,1)
plt.scatter(f(bps['gaborium']), f(bps['backimage']), alpha=.5, color='k', edgecolors='w')
plt.plot([-.1, 2], [-.1, 2], 'k--')
plt.xlabel('Gaborium BPS')
plt.ylabel('Backimage BPS')
# plt.xlim(-.1, 2)
# plt.ylim(-.1, 2)

plt.subplot(1,2,2)
plt.scatter(f(bps['gratings']), f(bps['fixrsvp']), alpha=.5, color='k', edgecolors='w')
plt.plot([-.1, 2], [-.1, 2], 'k--')
plt.xlabel('Gratings BPS')
plt.ylabel('FixRSVP BPS')
# plt.xlim(-.1, 2)
# plt.ylim(-.1, 2)
plt.show()

plt.subplots_adjust(wspace=.25)

#%% Try to get the fixrsvp stim psth
def get_fixrsvp_trials(model, eval_dicts, didx):

    robs = eval_dicts[didx]['fixrsvp'][0]
    rhat = eval_dicts[didx]['fixrsvp'][1]
    train_data, val_data, dataset_config = load_single_dataset(model, didx)
    stim_indices = get_stim_inds('fixrsvp', train_data, val_data)
    data = val_data.shallow_copy()
    data.inds = stim_indices


    dset_idx = np.unique(stim_indices[:,0]).item()
    time_inds = data.dsets[dset_idx]['psth_inds'].numpy()
    trial_inds = data.dsets[dset_idx]['trial_inds'].numpy()
    unique_trials = np.unique(trial_inds)

    n_trials = len(unique_trials)
    n_time = np.max(time_inds).item()+1
    n_units = data.dsets[dset_idx]['robs'].shape[1]
    robs_trial = np.nan*np.zeros((n_trials, n_time, n_units))
    rhat_trial = np.nan*np.zeros((n_trials, n_time, n_units))
    dfs_trial = np.nan*np.zeros((n_trials, n_time, n_units))

    for itrial in range(n_trials):
        trial_idx = np.where(trial_inds == unique_trials[itrial])[0]
        eval_inds = np.where(np.isin(stim_indices[:,1], trial_idx))[0]
        data_inds = trial_idx[np.where(np.isin(trial_idx, stim_indices[:,1]))[0]]

        # print(f'Trial {itrial} has {len(eval_inds)} eval inds and {len(data_inds)} data inds')
        assert torch.all(robs[eval_inds] == data.dsets[dset_idx]['robs'][data_inds]).item(), 'robs mismatch'

        robs_trial[itrial, time_inds[data_inds]] = robs[eval_inds]
        rhat_trial[itrial, time_inds[data_inds]] = rhat[eval_inds]
        dfs_trial[itrial, time_inds[data_inds]] = data.dsets[dset_idx]['dfs'][data_inds]

    return robs_trial, rhat_trial, dfs_trial


def ccnorm_variable_trials(R, P, D=None, *,
                           ddof=0,
                           min_trials_per_bin=20,
                           min_time_bins=20):
    """
    Noise-corrected correlation (CC_norm) that allows N_t (trial count)
    to vary across time bins.

    Parameters
    ----------
    R : (N, T, K) float
        Single-trial responses (NaNs permitted).
    P : (N, T, K) float
        Model predictions (same shape as R); NaNs ignored via mask too.
    D : (N, T, K) bool or None
        Valid-sample mask.  If None, everything not-NaN in R is valid.
    ddof : int
        Passed to variance/covariance calls (0=poulation, 1=sample).
    min_trials_per_bin : int
        Ignore time bins with < this many valid trials.
    min_time_bins : int
        Require at least this many bins after masking, else return NaN.

    Returns
    -------
    cc : (K,) float
        CC_norm per neuron (NaN when SP ≤ 0 or not enough data).
    """
    R = np.asarray(R, float)
    P = np.asarray(P, float)
    if D is None:
        D = ~np.isnan(R)
    else:
        D = np.asarray(D, bool)

    # Apply mask: invalid entries → NaN
    R = np.where(D, R, np.nan)
    P = np.where(D, P, np.nan)

    N, T, K = R.shape
    if N < 2:
        raise ValueError("Need at least two trials.")

    cc = np.full(K, np.nan)

    for k in range(K):
        # 1.  Trial counts per time bin
        n_valid = np.sum(~np.isnan(R[:, :, k]), axis=0)      # (T,)
        good_t  = n_valid >= min_trials_per_bin

        if np.count_nonzero(good_t) < min_time_bins:
            continue                                        # leave as NaN

        r   = R[:, good_t, k]                               # (N, T_good)
        p   = P[:, good_t, k]

        # 2.  PSTH and per-bin noise variance
        y_t     = np.nanmean(r, axis=0)                     # (T_good,)
        n_t     = n_valid[good_t]
        s2_t    = np.nanvar(r, axis=0, ddof=ddof)           # across trials

        # -- explainable signal power (★)
        var_y   = np.nanvar(y_t, ddof=ddof)
        noise_correction = np.nanmean(s2_t / n_t)
        SP      = var_y - noise_correction
        if SP <= 0 or np.isnan(SP):
            continue

        # 3.  Prediction statistics
        p_mean  = np.nanmean(p, axis=0)
        cov     = np.nanmean((y_t - y_t.mean()) *
                             (p_mean - p_mean.mean()))
        var_p   = np.nanvar(p_mean, ddof=ddof)
        if var_p == 0:
            continue

        cc[k] = cov / np.sqrt(var_p * SP)

    return cc


def ccnorm_masked(R, P, D, *, ddof=0, min_n_trials=30, min_t_samples=100):
    """
    Noise-corrected correlation (CC_norm) for data with missing samples.

    Parameters
    ----------
    Robs : ndarray (N, T, K)
        Single-trial responses.
    Pred : ndarray (N, T, K)
        Model predictions.
    Mask : ndarray (N, T, K), bool
        Valid-sample mask (True = keep, False = ignore).
    ddof : int, optional
        Delta-degrees-of-freedom for Var/Cov (default 0).

    Returns
    -------
    ccn : ndarray (K,)
        CC_norm per neuron; NaN if too noisy / no valid data.

    Notes
    -----
    •  For each neuron k we keep only time bins where *all* trials are valid:
         valid_t = D[:, :, k].all(axis=0)
       If < 2 such bins remain the score is set to NaN.

    •  With a full mask of True the function reproduces the unmasked
       closed-form from Schoppe et al. (2016).
    """
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)
    if D is None:
        D = np.ones_like(R, dtype=bool)
    else:
        D = np.asarray(D, dtype=bool)

    R[~D] = np.nan
    P[~D] = np.nan

    if R.shape != P.shape or R.shape != D.shape or R.ndim != 3:
        raise ValueError("R, P, D must have identical shape (N, M, K).")

    N, Tmax, K = R.shape
    
    if N < 2:
        raise ValueError("Need ≥2 trials to estimate signal power.")

    ccn = np.full(K, np.nan)          # initialise result

    valid_samples = np.sum(~np.isnan(R), axis=0)
    valid_samples >= min_n_trials
    
    cids = np.where(np.sum(valid_samples >= min_n_trials, 0)>min_t_samples)[0]

    for k in cids:
        # -------- 1. restrict to time bins with valid data in *all* trials
        # keep = np.where(valid_samples[:,k]>min_n_trials)[0]
        keep = np.where(valid_samples[:,k]==np.max(valid_samples[:,k]))[0]

        r = R[:, keep, k]                   # (N, T)
        p = P[:, keep, k]                   # (N, T)

        # optional: collapse identical predictions across trials
        # (here we assume predictions may differ per trial; average them)
        p_mean = np.nanmean(p, axis=0)             # (T,)
        r_mean = np.nanmean(r, axis=0)             # PSTH y(t)

        # -------- 2. signal power  (same formula, now on the reduced set)
        var_sum = np.nanvar(np.nansum(r, axis=0), ddof=ddof)          # Var(Σ_n R_n)
        var_each = np.nansum(np.nanvar(r, axis=1, ddof=ddof))       # Σ_n Var(R_n)
        N = valid_samples[keep,k]
        SP = (var_sum - var_each) / (N * (N - 1))
        if SP <= 0:
            continue                                        # too noisy → NaN

        # -------- 3. covariance & variance of prediction
        cov = np.cov(r_mean, p_mean, ddof=ddof)[0, 1]
        var_p = np.var(p_mean, ddof=ddof)
        if var_p == 0:
            continue

        # -------- 4. CC_norm
        ccn[k] = cov / np.sqrt(var_p * SP)

    return ccn

#%%
dt = 1/240
ccns = []

for didx in range(len(model.names)):
    try:
        robs_trial, rhat_trial, dfs_trial = get_fixrsvp_trials(model, eval_dicts, didx)
        rbar = np.nansum(robs_trial*dfs_trial, axis=0) / np.nansum(dfs_trial, axis=0)/dt
        rbarhat = np.nansum(rhat_trial*dfs_trial, axis=0) / np.nansum(dfs_trial, axis=0)/dt

        ccn = ccnorm_variable_trials(robs_trial, rhat_trial, dfs_trial, min_trials_per_bin=30, min_time_bins=60)
        ccn = np.minimum(np.maximum(ccn, 0), 1)
        ccns.append(ccn[~np.isnan(ccn)])
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
ccns = np.concatenate(ccns)
plt.hist(ccns, bins=30)
plt.xlabel("Normalized Correlation Coefficient")

#%%
from scipy.ndimage import gaussian_filter1d
f = lambda x: gaussian_filter1d(x, 1)
cids = np.where(ccn > 0.2)[0]

for i, cid in enumerate(cids):
    plt.figure()
    plt.plot(f(rbar[33:200,cid]))
    plt.plot(rbarhat[33:200,cid])
    plt.title(f'Unit {cid} - CCN: {ccn[cid]:.2f}')
    plt.show()

#%%

f = lambda x: gaussian_filter1d(x, 1)
plot_trials = np.where(~np.all(np.isnan(robs_trial[:,:,cid]), 1))[0]

for cid in cids:
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(f(robs_trial[plot_trials, 33:300, cid]), aspect='auto', interpolation='none', cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Time (5ms bins)')
    plt.ylabel('Trial')
    plt.title('Observed Spikes')

    plt.subplot(1,2,2)
    plt.imshow((rhat_trial[plot_trials, 33:300, cid]), aspect='auto', interpolation='none', cmap='viridis')
    plt.xlabel('Time (5ms bins)')
    plt.ylabel('Trial')
    plt.title('Predicted Spikes')

    plt.show()
# for i, (dset_idx, time_idx) in enumerate(stim_indices):
    # robs[trial_inds[i], time_idx] += data.dsets[dset_idx]['robs'][time_idx]

# for i, (dset_idx, time_idx) in enumerate(stim_indices):
#     robs[trial_inds[i]] += data.dsets[dset_idx]['robs'][time_idx]

#%%

#%%
# x = torch.randn(1, 256, 1, 9, 9).to(device)
# model.model.readouts[0](x)
# model.model.readouts[1].plot_weights(ellipse=True)
std_ex = []
std_inh = []
for didx in range(len(model.names)):
    std_ex.append(model.model.readouts[didx].std_ex.data.detach().cpu().numpy().mean(1))
    std_inh.append(model.model.readouts[didx].std_inh.data.detach().cpu().numpy().mean(1))

std_ex= np.concatenate(std_ex)
std_inh = np.concatenate(std_inh)
plt.plot(std_ex, std_inh, 'o', alpha=.15)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
# plt.bar(np.arange(len(model.names)), std_ex, width=0.4, label='Excitatory')

#%%
feats_ex = []
feats_inh = []
for didx in range(len(model.names)):
    feats_ex.append(model.model.readouts[didx]._features_ex_weight.data.detach().cpu().numpy().squeeze())
    feats_inh.append(model.model.readouts[didx]._features_inh_weight.data.detach().cpu().numpy().squeeze())

feats_ex = np.concatenate(feats_ex, axis=0)
feats_inh = np.concatenate(feats_inh, axis=0)
feats = np.concatenate([feats_ex, feats_inh], axis=1)
# feats = feats_ex
feats /= np.linalg.norm(feats, axis=1)[:, None]
D = np.dot(feats, feats.T)

plt.imshow(D, cmap='viridis', interpolation='none')
plt.show()

#%%

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(eval_dict['gaborium'][2])
ax.set_ylim(-.1, 1.1*np.nanmax(eval_dict['gaborium'][2]))

# show grid
ax.grid(True)
plt.show()
plt.close(fig)  # Clean up figure



#%% Visualize BPS comparisons across datasets

def plot_bps_comparison(x_bps, y_bps, x_label, y_label, title=None):
    """
    Create a scatter plot comparing bits per spike between two datasets.

    Parameters
    ----------
    x_bps : numpy.ndarray
        BPS values for x-axis.
    y_bps : numpy.ndarray
        BPS values for y-axis.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    title : str, optional
        Plot title, by default None.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_bps, y_bps)
    ax.plot([-1, 3], [-1, 3], color='k', linestyle='--', alpha=.5)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_xlabel(f'{x_label} BPS')
    ax.set_ylabel(f'{y_label} BPS')
    if title:
        ax.set_title(title)
    plt.show()
    plt.close(fig)  # Clean up figure

# Compare Gaborium vs Backimage BPS
plot_bps_comparison(
    eval_dict['gaborium'][2], eval_dict['backimage'][2],
    'Gaborium', 'Backimage',
    'Comparison of Bits Per Spike: Gaborium vs Backimage'
)

# Compare Gaborium vs Gratings BPS
plot_bps_comparison(
    eval_dict['gaborium'][2], eval_dict['gratings'][2],
    'Gaborium', 'Gratings',
    'Comparison of Bits Per Spike: Gaborium vs Gratings'
)

# Compare Gaborium vs FixRSVP BPS
plot_bps_comparison(
    eval_dict['gaborium'][2], eval_dict['fixrsvp'][2],
    'Gaborium', 'FixRSVP',
    'Comparison of Bits Per Spike: Gaborium vs FixRSVP'
)



#%%
stim = 'gratings'
robs = eval_dict[stim][0]
rhat = eval_dict[stim][1]


inds = np.arange(512) + np.random.randint(0, 10000)

plt.figure(figsize=(10, 5))
plt.subplot(2,1,1)
plt.imshow(robs[inds,:].T, aspect='auto', interpolation='none', cmap='gray_r', vmin=0, vmax=1)
plt.subplot(2,1,2)
plt.imshow(rhat[inds,:].T, aspect='auto', interpolation='none', cmap='gray_r', vmin=0, vmax=.5)

def correlation_matrix(data):
    """Compute correlation matrix for a given dataset."""
    C = torch.cov(data.T)
    # set diagonal to 0
    C.fill_diagonal_(0)
    return C

plt.figure()
plt.subplot(1,2,1)
plt.imshow(correlation_matrix(robs))
plt.subplot(1,2,2)
plt.imshow(correlation_matrix(rhat))

#%% Animation function for predictions
import matplotlib.animation as animation

def animate_predictions_with_cleanup(val_data, model, didx, device,
                                   start_ind=0, stop_ind=None, window_size=260,
                                   step=10, output_path='predictions_animation.mp4',
                                   fps=10, figsize=(10, 5)):
    """
    Create an animation of model predictions vs observations over time windows.

    Parameters
    ----------
    val_data : dataset
        Validation dataset
    model : torch.nn.Module
        Trained model
    didx : int
        Dataset index
    device : torch.device
        Device to run model on
    start_ind : int, default=0
        Starting index for animation
    stop_ind : int, optional
        Stopping index for animation. If None, uses len(val_data) - window_size
    window_size : int, default=260
        Size of time window to display
    step : int, default=10
        Step size between frames
    output_path : str, default='predictions_animation.mp4'
        Path to save animation
    fps : int, default=10
        Frames per second for animation
    figsize : tuple, default=(10, 5)
        Figure size

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object
    """
    if stop_ind is None:
        stop_ind = len(val_data) - window_size

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Initialize empty plots
    im1 = ax1.imshow(np.zeros((1, window_size)), aspect='auto', interpolation='none', cmap='gray_r', vmax=1)
    im2 = ax2.imshow(np.zeros((1, window_size)), aspect='auto', interpolation='none', cmap='gray_r', vmax=0.5)

    ax1.set_ylabel('Units')
    ax1.set_title('Observed Spikes')
    ax2.set_xlabel('Time (5ms bins)')
    ax2.set_ylabel('Units')
    ax2.set_title('Predicted Rates')

    # Text for frame info
    frame_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    def animate(frame):
        """Animation function for each frame."""
        current_ind = start_ind + frame * step
        inds = np.arange(window_size) + current_ind

        # Ensure indices are within bounds
        if inds[-1] >= len(val_data):
            return im1, im2, frame_text

        try:
            # Get batch data
            batch = val_data[inds]
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                rhat = model.model(batch['stim'], didx, batch.get('behavior'))

            # Move to CPU for plotting
            robs_cpu = batch['robs'].detach().cpu().numpy()
            if model.log_input:
                rhat = torch.exp(rhat)
            rhat_cpu = rhat.detach().cpu().numpy()

            # multiply by dfs
            robs_cpu = robs_cpu * batch['dfs'].detach().cpu().numpy()
            rhat_cpu = rhat_cpu * batch['dfs'].detach().cpu().numpy()

            # Update images
            im1.set_array(robs_cpu.T)
            # im1.set_clim(vmin=robs_cpu.min(), vmax=1) #robs_cpu.max())

            im2.set_array(rhat_cpu.T)
            # im2.set_clim(vmin=rhat_cpu.min(), vmax=.25) #rhat_cpu.max())

            # Update frame info
            frame_text.set_text(f'Frame {frame+1}, Time bins {current_ind}-{current_ind+window_size-1}')

            # Clean up batch tensors
            del batch, rhat
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in frame {frame}: {e}")

        return im1, im2, frame_text

    # Calculate number of frames
    n_frames = (stop_ind - start_ind) // step

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                 interval=1000//fps, blit=True, repeat=True)

    # Save animation
    print(f"Saving animation to {output_path}...")
    try:
        # Use ffmpeg writer for MP4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='DataYatesV1'), bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"✅ Animation saved successfully to {output_path}")
    except Exception as e:
        print(f"❌ Error saving animation: {e}")
        print("💡 Make sure ffmpeg is installed: conda install ffmpeg")

    return anim


#%% Example usage of animation function
# Uncomment and modify these lines to create an animation:
start = 69891 + 25000
# Create animation with custom parameters
# anim = animate_predictions_with_cleanup(
#     val_data=val_data,
#     model=model,
#     didx=didx,
#     device=device,
#     start_ind=start,           # Start from beginning
#     stop_ind=start + 5000,         # Stop at index 1000 (or None for end of data)
#     window_size=512,       # Show 260 time bins at once
#     step=20,               # Advance by 20 bins each frame
#     output_path='neural_predictions.mp4',
#     fps=5,                 # 5 frames per second
#     figsize=(12, 6)        # Larger figure
# )

# # Display the animation in notebook (optional)
# # from IPython.display import HTML
# # HTML(anim.to_jshtml())

# %%

cidslist = np.argsort(eval_dict['backimage'][2])[::-1]


# cid = 0  # Initialize cid
# cid += 1
cid = cidslist[10]
plt.figure(figsize=(10, 2))
_ = plt.plot(robs[inds,cid] > 0, 'k')
ax = plt.gca().twinx()
_ = ax.plot(rhat[inds,cid], 'r')

plt.show()
# plt.close()

# %%
# model.model.readouts[didx].plot_weights()
# %%
# model.model.readouts[didx].get_spatial_weights().shape
# %%

stim_type = 'gaborium'
robs = eval_dict[stim_type][0]
pred = eval_dict[stim_type][1]
bps = eval_dict[stim_type][2]

plt.subplot(1,2,1)
plt.imshow(correlation_matrix(robs))
plt.title('Observed')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.subplot(1,2,2)
plt.imshow(correlation_matrix(eval_dict[stim_type][1]))
plt.title('Predicted')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.show()

corr_coefs = torch.zeros(pred.shape[1])
for i in range(pred.shape[1]):
    corr_coefs[i] = torch.corrcoef(torch.stack([pred[:, i], robs[:, i]]))[0, 1]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(bps, '-o')
axs[0].set_ylabel('Bits per spike')
axs[0].set_xlabel('Neuron')
# ax = axs[0].twinx()
# ax.plot(corr_coefs, '-o', color='r')
# ax.set_ylabel('Correlation coefficient')
# ax.set_xlabel('Neuron')

axs[1].plot(corr_coefs, '-o')
axs[1].set_ylabel('Correlation coefficient')
axs[1].set_xlabel('Neuron')

plt.tight_layout()

plt.figure()
plt.plot(corr_coefs, bps, 'o')
plt.xlabel('Correlation coefficient')
plt.ylabel('Bits per spike')

good_units = [int(i) for i in torch.where((corr_coefs > 0.15) & (bps > 0.25))[0]]

print(f'Found {len(good_units)} good units')
# %% load session to get saccade times
from DataYatesV1 import get_session
from jake.detect_saccades import detect_saccades


sess_name = model.names[didx]
sess = get_session(*sess_name.split('_'))
saccades = detect_saccades(sess)
print(f"Detected {len(saccades)} saccades")

saccade_times = torch.sort(torch.tensor([s.start_time for s in saccades])).values.numpy()

valid = np.diff(saccade_times, prepend=0) > 0.1
saccade_times = saccade_times[valid]

plt.figure(figsize=(6, 3))
_ = plt.hist(np.diff(saccade_times, prepend=0), np.linspace(0, 1, 100))
plt.xlabel('Time (s)')
plt.ylabel('Count')
plt.title('Saccade ISI distribution')

plt.show()

#%%
saccade_inds = train_data.get_inds_from_times(saccade_times)



#%%
stim_type = 'backimage'

def get_sac_eval(stim_type, dataset, win = (-10, 100)):
    
    stim_inds = get_stim_inds(stim_type, train_data, dataset)

    dataset = dataset.shallow_copy()
    dataset.inds = stim_inds

    dset = stim_inds[0,0]
    print(f'Dataset {dset}')

    robs = eval_dict[stim_type][0]
    pred = eval_dict[stim_type][1]

    print(f"Number of robs bins: {robs.shape[0]} Number of stim bins: {stim_inds.shape[0]}")

    nbins = win[1]-win[0]

    sac_indices = saccade_inds[saccade_inds[:,0]==dset, 1]
    n_sac = len(sac_indices)
    robs_sac = np.nan*np.zeros((n_sac, nbins, robs.shape[1]))
    pred_sac = np.nan*np.zeros((n_sac, nbins, pred.shape[1]))
    dfs_sac = np.nan*np.zeros((n_sac, nbins, robs.shape[1]))

    for i,isac in enumerate(sac_indices):
        j = np.where(stim_inds[:,1] == isac)[0]
        
        if len(j) == 0:
            continue

        j = j.item()
        
        dataset_idx = np.where(torch.all(dataset.inds == stim_inds[j], 1))[0]
        if len(dataset_idx) == 0:
            continue
        dataset_idx = dataset_idx.item()
        
        if (j + win[0] >= 0) & (j + win[1] < robs.shape[0]):
            robs_ = robs[(j+win[0]):(j+win[1])]
            pred_ = pred[(j+win[0]):(j+win[1])]
            batch= dataset[dataset_idx+win[0]:dataset_idx+win[1]]
            assert torch.all(batch['robs'] == robs_), 'robs mismatch'
            dfs_ = batch['dfs']

            robs_sac[i] = robs_
            pred_sac[i] = pred_
            dfs_sac[i] = dfs_

    good = np.where(np.sum(np.isnan(robs_sac), axis=(1,2)) == 0)[0]

    robs_sac = robs_sac[good]
    pred_sac = pred_sac[good]
    dfs_sac = dfs_sac[good]

    rbar = np.nansum(robs_sac*dfs_sac, axis=0) / np.nansum(dfs_sac, axis=0)
    rbarhat = np.nansum(pred_sac*dfs_sac, axis=0) / np.nansum(dfs_sac, axis=0)

    return {'robs': robs_sac, 'pred': pred_sac, 'dfs': dfs_sac, 'rbar': rbar, 'rbarhat': rbarhat}

sac_eval = {}
stim_types = ['backimage', 'gaborium', 'gratings', 'fixrsvp']
for stim_type in stim_types:
    sac_eval[stim_type] = get_sac_eval(stim_type, val_data, win=(-10, 100))

#%%
for stim_type in stim_types:
    
    rbar = sac_eval[stim_type]['rbar']
    rbarhat = sac_eval[stim_type]['rbarhat']
    plt.figure()    
    plt.subplot(2,1,1)
    _ = plt.plot(rbar)
    plt.title(stim_type)
    yd = plt.ylim()
    plt.subplot(2,1,2)
    _ = plt.plot(rbarhat)
    plt.ylim(yd)
    
    plt.show()

#%%
def model_pred(batch, stage='pred', include_modulator=True):

    if stage=='pred':
        if include_modulator:
            output = model.model(batch['stim'], didx, batch.get('behavior'))
        else:
            output = model.model(batch['stim'], didx, torch.zeros_like(batch.get('behavior')))

        if model.log_input:
            output = torch.exp(output)
        return output
    
    x = model.model.adapters[didx](batch['stim'])
    if stage == 'adapter':
        return x
    x = model.model.frontend(x)
    if stage == 'frontend':
        return x
    x = model.model.convnet(x)
    if stage == 'convnet':
        return x
    
    if include_modulator:
        x = model.model.modulator(x, batch.get('behavior'))

    if stage == 'modulator':
        return x
    
    if stage == 'readout':
        if 'DynamicGaussianReadoutEI' in str(type(model.model.readouts[didx])):
            x = x[:, :, -1, :, :]  # (N, C_in, H, W)
            N, C_in, H, W = x.shape
            device = x.device

            readout = model.model.readouts[didx]
            # Apply positive-constrained feature weights using functional conv2d
            feat_ex = torch.nn.functional.conv2d(x, readout.features_ex_weight, bias=None)  # (N, n_units, H, W)
            feat_inh = torch.nn.functional.conv2d(x, readout.features_inh_weight, bias=None)  # (N, n_units, H, W)

            # Compute Gaussian masks for both pathways
            gaussian_mask_ex = readout.compute_gaussian_mask(H, W, device, pathway='ex')  # (n_units, H, W)
            gaussian_mask_inh = readout.compute_gaussian_mask(H, W, device, pathway='inh')  # (n_units, H, W)

            # Apply masks and sum over spatial dimensions
            out_ex = (feat_ex * gaussian_mask_ex.unsqueeze(0)).sum(dim=(-2, -1))  # (N, n_units)
            out_inh = (feat_inh * gaussian_mask_inh.unsqueeze(0)).sum(dim=(-2, -1))  # (N, n_units)
            return out_ex, out_inh        
    else:
        x = model.model.readouts[didx](x)
        return x

#%%
batch_size=64
bind = np.arange(1000, 1000+batch_size)
batch = val_data[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
# stim = batch['stim'].to(device)

#%% plot e and i of readout
ex, inh = model_pred(batch, stage='readout')
fun = lambda x: torch.exp(x)
cid = 20
plt.plot(fun(ex[:,cid]).detach().cpu(), 'r')
# plt.plot(1/fun(-inh[:,cid]).detach().cpu(), 'b')
plt.plot(fun(inh[:,cid]).detach().cpu(), 'b')
ax = plt.gca().twinx()
ax.plot(torch.exp(ex[:,cid].detach().cpu() - inh[:,cid].detach().cpu()), 'g')
ax.plot(batch['robs'][:,cid].detach().cpu(), 'k')

del ex, inh
torch.cuda.empty_cache()

#%%
conv_out = model_pred(batch, stage='convnet')
_ = plt.plot(conv_out[:,:,-1,5,5].detach().cpu())
del conv_out
torch.cuda.empty_cache()

#%%
pred = model_pred(batch, stage='pred', include_modulator=True)
pred_no_eye = model_pred(batch, stage='pred', include_modulator=False)

n_plots = 20
fig, axs = plt.subplots(n_plots, 1, figsize=(5, n_plots), sharex=True)
plot_cids = np.random.choice(good_units, n_plots, replace=False)
for i, cid in enumerate(plot_cids):
    axs[i].twinx().plot(batch['robs'][:,cid].detach().cpu(), 'k', label='Observed')
    axs[i].plot(pred[:,cid].detach().cpu(), 'r', label='With Eye')
    axs[i].plot(pred_no_eye[:,cid].detach().cpu(), 'b', label='No Eye')
    
    axs[i].set_ylabel(f'Unit {cid}')
    axs[i].set_ylim(0, 1)
axs[0].legend(loc='upper right')


plt.figure()

plt.subplot(2,1,1)
_ = plt.imshow(batch['robs'][:,good_units].detach().cpu().T, aspect='auto', interpolation='none', cmap='gray_r')
plt.ylabel('Units')

plt.subplot(2,1,2)
# spikefn = lambda x: torch.poisson(x)

_ = plt.imshow(pred[:,good_units].detach().cpu().T, aspect='auto', interpolation='none', cmap='gray_r')
plt.xlabel('Time (5ms bins)')
plt.ylabel('Units')

del pred, pred_no_eye
torch.cuda.empty_cache()

# %%

# enc = 

# gain = model.model.modulator.scale_layer(enc)
# offset = model.model.modulator.shift_layer(enc)
# # %%
# plt.figure(figsize=(10, 5))
# plt.subplot(2,1,1)
# _ = plt.plot(gain.detach().cpu())
# plt.ylabel('Gain')
# plt.title('Gain')

# # plot eye pos overlaid
# ax = plt.gca().twinx()
# ax.plot(batch['eyepos'].detach().cpu(), color='k', alpha=1)
# ax.set_ylabel('Eye Position', color='k')    


# plt.subplot(2,1,2)
# _ = plt.plot(offset.detach().cpu())
# plt.xlabel('Time (5ms bins)')
# plt.ylabel('Offset')
# plt.title('Offset')

# # plot eye pos overlaid
# ax = plt.gca().twinx()
# ax.plot(batch['eyepos'].detach().cpu(), color='k', alpha=1)
# ax.set_ylabel('Eye Position', color='k')



# %%
# --------------------------------------------------------------------------
# JACOBIAN
# stim = batch['stim'].to(device)
model.model.eval()
model.model.convnet.use_checkpointing = False 

# Ensure the model is not using torch.compile
if hasattr(model.model, "_orig_mod"):
    # If the model was compiled with torch.compile(), access the original module
    orig_model = model.model._orig_mod
    # Temporarily replace the compiled model with the original
    temp_model = model.model
    model.model = orig_model
else:
    temp_model = None
# %% try using jacrev
from scipy.ndimage import gaussian_filter
# import jacrev
from torch.func import jacrev, vmap
lag = 8
T, C, S, H, W  = batch['stim'].shape
n_units      = len(good_units)
unit_ids     = torch.arange(n_units, device=device)
smooth_sigma = .5

# --------------------------------------------------------------------------
# 2. helper – Jacobian → energy CoM for *every* unit in one call
grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device),
                                indexing='ij')
grid_x = grid_x.expand(n_units, H, W)   # each unit gets the same grids
grid_y = grid_y.expand_as(grid_x)

def irf_J(frame_stim, behavior, unit_idx):
    """
    frame_stim : (C,H,W) tensor with grad
    returns     : (n_units, 2)   (cx, cy) per unit, NaN if IRF==0
    """
    def f(s):
        out = model.model(s.unsqueeze(0), didx, behavior)[0]
        return out[unit_idx]

    return jacrev(f)(frame_stim)

def irf_com(frame_stim, behavior, unit_ids):
    """
    frame_stim : (C,H,W) tensor with grad
    returns     : (n_units, 2)   (cx, cy) per unit, NaN if IRF==0
    """
    J = irf_J(frame_stim, behavior, unit_ids)[:,lag]
    E = J.pow(2)

    if smooth_sigma:
        E = gaussian_filter(E.detach().cpu().numpy(),           # (n_units,H,W)
                            sigma=(0, smooth_sigma, smooth_sigma))
        E = torch.as_tensor(E, device=device)

    tot   = E.flatten(1).sum(-1)                              # (n_units,)
    mask  = tot > 0
    cx    = (E*grid_x).flatten(1).sum(-1) / tot
    cy    = (E*grid_y).flatten(1).sum(-1) / tot
    cx[~mask] = torch.nan
    cy[~mask] = torch.nan
    return torch.stack([cx, cy], 1)           # (n_units,2)

       # (n_units, C, H, W)

#%% --------------------------------------------------------------------------
# 3. pre-compute CoM for all frames *once*
# output = model.model(batch['stim'], didx, batch.get('behavior'))
i = 0
J = irf_J(batch['stim'][i], batch['behavior'][i], good_units)
J.shape


#%% MEI analysis
from mei import mei_synthesis, Jitter, LpNorm, TotalVariation, Combine, GaussianGradientBlur, ClipRange

def net(x):
    return torch.exp(model.model(x, didx, batch.get('behavior')[0])[0])

rwa = 0

for i in tqdm(range(1000)):
    s = torch.randn(1, 1, 25, 51, 51)
    s[s.abs()<.5] = 0
    r = net(s.to(model.device)).detach().cpu().numpy()
    s = s.detach().cpu().numpy()

    rwa += r[:,None,None,None]*s[0]

cid = 0
#%%
unit_list = good_units  

cid += 1
if cid >= len(unit_list):
    cid = 0

mei_img = torch.tensor(rwa[[unit_list[cid]]])

plt.figure()
plt.subplot(1,2,1)
spatial_peak = np.argmax(mei_img.std(dim=(0,1)).detach().cpu())
ii,jj = np.unravel_index(spatial_peak, mei_img.shape[2:])
plt.imshow(mei_img[0,:,ii,:].detach().cpu().numpy(), aspect='auto', cmap='gray')

plt.subplot(1,2,2)
temporal_peak = np.argmax(mei_img.std(dim=(0,2,3)).detach().cpu())
plt.imshow(mei_img[0,temporal_peak,:,:].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title(unit_list[cid])
plt.show()

#%%
#%%

transform = Jitter([4, 4, 4])                       # time, H, W
regulariser = Combine([LpNorm(p=2, weight=0.1), TotalVariation(weight=0.01)])
precond = GaussianGradientBlur(sigma=1.0, order=3)
# post = ClipRange(-1.0, 1.0)

# regulariser = None
# transform = None
# precond = None
post = None
meis = []
for cid in range(len(good_units)):
    mei = mei_synthesis(
        model=net,
            initial_image=torch.randn(1, 1, 25, 51, 51)*.2,
            unit=unit_list[cid],
            n_iter=1000,
            optimizer_fn=torch.optim.SGD,
            optimizer_kwargs={"lr": 10},
            transform=transform,
            regulariser=regulariser,
            preconditioner=precond,
            postprocessor=post,
            device=model.device
        )

    mei_img = mei[0].detach()
    meis.append(mei_img)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    spatial_peak = np.argmax(mei_img.std(dim=(0,1)).detach().cpu())
    ii,jj = np.unravel_index(spatial_peak, mei_img.shape[2:])
    plt.imshow(mei_img[0,:,ii,:].detach().cpu().numpy(), aspect='auto', cmap='gray')
    plt.xlabel('Space')
    plt.ylabel('Time')

    plt.subplot(1,2,2)
    temporal_peak = np.argmax(mei_img.std(dim=(0,2,3)).detach().cpu())
    plt.imshow(mei_img[0,temporal_peak,:,:].detach().cpu().numpy(), aspect='auto', cmap='gray')
    plt.xlabel('Space')
    plt.ylabel('Time')
    plt.title(f'Unit {unit_list[cid]}')
    plt.show()
#%% ---------- example call ----------
i = 40
meis = []
for cid in range(len(good_units)):
    # if cid > 1:
    #     break
    unit_list = good_units         
    # To run with L2 and Total Variation regularization:
    mei_img = mei_synthesis(
        model, didx, batch['behavior'][i], batch['stim'][i], unit_list[cid],
        n_iter=5000,
        optimizer_fn=torch.optim.SGD,
        optimizer_kwargs={'lr': 1.0},
        precondition_sigma=0.0,
        tv_weight=0,       # Encourages smoothness
        lp_weight=0.0,       # Controls image magnitude/contrast
        lp_p=2,                # Use the L2 norm
        device='cuda:1' if torch.cuda.is_available() else 'cpu'
    )

                      # list/array of unit indices
    # mei_img   = mei_synthesis(model, didx, batch['behavior'][i],
    #     ,
    #     unit_list[cid],
    #     n_iter=1000,
    #     sigma=.5,
    #     use_stim=False,
    #     lr=.5)

    meis.append(mei_img.detach().cpu())

    plt.figure()
    plt.subplot(1,2,1)
    spatial_peak = np.argmax(mei_img.std(dim=(0,1)).detach().cpu())
    ii,jj = np.unravel_index(spatial_peak, mei_img.shape[2:])
    plt.imshow(mei_img[0,:,ii,:].detach().cpu().numpy(), aspect='auto', cmap='gray')

    plt.subplot(1,2,2)
    temporal_peak = np.argmax(mei_img.std(dim=(0,2,3)).detach().cpu())
    plt.imshow(mei_img[0,temporal_peak,:,:].detach().cpu().numpy(), aspect='auto', cmap='gray')
    plt.title(unit_list[cid])
    plt.show()

    break

#%%


#%%

#%%

com = torch.empty((T, n_units, 2), device=device)
with torch.set_grad_enabled(True):
    for t in tqdm(range(T)):
        com[t] = irf_com(stim[t], good_units)

com = com.detach().cpu().numpy()

from torch.func import jacrev, vmap
n_units = batch['robs'].shape[1]

#%%
# %matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # registers the 3-D projection

# --- pull out the data ---
T, n_units, _ = com.shape
xs = com[:, :, 0].copy()
ys = com[:, :, 1].copy()
# smooth xs and ys along axis 0
# xs = gaussian_filter(xs, sigma=(2.5,0))
# ys = gaussian_filter(ys, sigma=(2.5,0))
eyetraj = batch['eyepos'].clone().cpu().numpy()
eyetraj -= eyetraj[prewin*2]
ppd = 37
eyetraj *= ppd
eyetraj = -eyetraj
eyetraj[:,0] += xs.mean()
eyetraj[:,1] += ys.mean()
eyetraj = gaussian_filter(eyetraj, sigma=(.5,0))


ts = np.arange(T)                                # 0 … T-1
spikes = batch['robs'].cpu().numpy() > 0

fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, projection='3d')

# one colour per neuron
colors = plt.cm.hsv(np.linspace(0, 1, n_units, endpoint=False))

for i in range(n_units):
    # m = np.ones_like(spikes[:, i])
    m = spikes[:, i]                    # spike mask for this neuron
    if m.any():                         # skip silent cells
        ax.plot(ts[m], xs[m, i], ys[m, i], '-o',
                color=colors[i], lw=1, alpha=.25)  # line connects spike-only points
        
ax.plot(ts, eyetraj[:,0], eyetraj[:,1], color='k', lw=1, alpha=1)        
ax.plot(ts, xs.mean()*np.ones_like(ts), ys.mean()*np.ones_like(ts), color='k', lw=1, alpha=1) 


        # ax.scatter(ts[m], xs[m, i], ys[m, i],   # optional spike markers
                #    color=colors[i], s=6)
        
        # ax.plot(xs[m, i], ys[m, i], ts[m],
        #         color=colors[i], lw=1)  # line connects spike-only points
        # ax.scatter(xs[m, i], ys[m, i], ts[m],   # optional spike markers
        #            color=colors[i], s=6)

# axis labels (flip y if you want image-style origin at top-left)
ax.set_zlabel('X (px)')
ax.set_ylabel('Y (px)')
ax.set_xlabel('Frame')
ax.set_zlim(20, 40)
ax.set_ylim(20, 40)
ax.set_xlim(0, len(ts))
ax.invert_yaxis()

plt.tight_layout()
plt.show()