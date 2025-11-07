"""
Evaluation Stack Utilities - Reusable Functions for Neural Model Evaluation

This module contains reusable functions extracted from model evaluation scripts
to support unified evaluation pipelines across different models and datasets.

Functions include:
- Dataset loading and preparation
- Model evaluation and BPS calculation
- Stimulus-specific analyses (FixRSVP, saccades)
- QC data loading and processing
- Noise-corrected correlation calculations

Author: Extracted from model_load_eval_stack_multidataset.py
"""

import sys
import os
from pathlib import Path

# Add VisionCore root to Python path (go up 1 level from eval/)
_visioncore_root = Path(__file__).parent.parent
sys.path.insert(0, str(_visioncore_root))
import json
import re
import contextlib
from pathlib import Path
from pprint import pprint
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.data import prepare_data
from models.losses import PoissonBPSAggregator

import torch
from torch import nn
import torch.nn.functional as F

def scan_checkpoints(checkpoint_dir="../../checkpoints", verbose=False):
    """Scan checkpoint directory and organize models by type.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing checkpoint subdirectories
    verbose : bool, default False
        If True, print detailed scanning information

    Returns
    -------
    dict
        Dictionary mapping model types to lists of checkpoint info
    """
    checkpoint_dir = Path(checkpoint_dir)

    if verbose:
        print(f"Scanning directory: {checkpoint_dir.absolute()}")

    if not checkpoint_dir.exists():
        if verbose:
            print(f"❌ Checkpoint directory not found: {checkpoint_dir.absolute()}")
        return {}

    models_by_type = defaultdict(list)

    if verbose:
        print(f"Found subdirectories:")
    for exp_dir in checkpoint_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        if verbose:
            print(f"  {exp_dir.name}")

        # Extract model type from experiment name
        exp_name = exp_dir.name
        model_type = extract_model_type(exp_name)
        if verbose:
            print(f"    -> Model type: {model_type}")

        # Find valid checkpoints
        ckpt_files = []
        all_ckpt_files = list(exp_dir.glob("*.ckpt"))
        if verbose:
            print(f"    -> Found {len(all_ckpt_files)} .ckpt files")

        for ckpt_file in all_ckpt_files:
            if verbose:
                print(f"      {ckpt_file.name}")
            # Skip 'last' checkpoints and NaN loss checkpoints
            if "last" in ckpt_file.name.lower() or "nan" in ckpt_file.name.lower():
                if verbose:
                    print(f"        -> Skipped (last or nan)")
                continue

            # Extract validation metrics and epoch
            val_loss = extract_val_loss(ckpt_file.name)
            val_bps = extract_val_bps(ckpt_file.name)
            epoch = extract_epoch(ckpt_file.name)
            if verbose:
                print(f"        -> Val loss: {val_loss}, Val BPS: {val_bps}, Epoch: {epoch}")

            # Prefer BPS over loss when available
            if val_bps is not None and not np.isnan(val_bps):
                ckpt_files.append({
                    'path': ckpt_file,
                    'val_loss': val_loss,  # Keep for backward compatibility
                    'val_bps': val_bps,
                    'epoch': epoch if epoch is not None else 0,
                    'experiment': exp_name,
                    'metric_type': 'bps'
                })
                if verbose:
                    print(f"        -> Added to {model_type} (using BPS)")
            elif val_loss is not None and not np.isnan(val_loss):
                ckpt_files.append({
                    'path': ckpt_file,
                    'val_loss': val_loss,
                    'val_bps': None,
                    'epoch': epoch if epoch is not None else 0,
                    'experiment': exp_name,
                    'metric_type': 'loss'
                })
                if verbose:
                    print(f"        -> Added to {model_type} (using loss)")
            else:
                if verbose:
                    print(f"        -> Skipped (no valid metric)")

        # Add to models by type
        if ckpt_files:
            models_by_type[model_type].extend(ckpt_files)

    # Sort each model type by best available metric (BPS preferred, then loss)
    for model_type in models_by_type:
        def sort_key(x):
            # If BPS is available, use it (higher is better, so negate for ascending sort)
            if x.get('metric_type') == 'bps' and x.get('val_bps') is not None:
                return -x['val_bps']  # Negative because higher BPS is better
            # Otherwise use loss (lower is better)
            else:
                return x['val_loss']

        models_by_type[model_type].sort(key=sort_key)

    if verbose:
        print(f"\nFinal model counts by type:")
        for model_type, models in models_by_type.items():
            print(f"  {model_type}: {len(models)} models")

    return dict(models_by_type)


def extract_model_type(exp_name):
    """Extract model type from experiment name."""
    exp_name = exp_name.lower()
    # Order matters - check more specific patterns first
    if 'resnet_modulator' in exp_name:
        return 'resnet_modulator'
    elif 'learned_dense_film_none_gaussian' in exp_name:
        return 'learned_dense_film_none_gaussian'
    elif 'x3d_modulator' in exp_name:
        return 'x3d_modulator'
    elif 'dense_concat_convgru_gaussian_history' in exp_name:
        return 'dense_concat_convgru_history'
    elif 'dense_concat_convgru_gaussian' in exp_name:
        return 'dense_concat_convgru'
    elif 'dense_none_convgru_gaussian' in exp_name:
        return 'dense_none_convgru'
    elif 'densenet' in exp_name:
        return 'densenet'
    elif 'core_res_modulator' in exp_name:
        return 'core_res_modulator'
    elif 'core_res' in exp_name:
        return 'core_res'
    elif 'learned_resnet_concat_convgru_gaussian' in exp_name:
        return 'resnet_concat_convgru'
    elif 'learned_resnet_none_convgru_gaussian' in exp_name:
        return 'resnet_none_convgru'
    elif 'learned_resnet_film_none_gaussian' in exp_name:
        return 'resnet_film_none_gaussian'
    elif 'resnet' in exp_name:
        return 'resnet'
    elif 'vivit_small' in exp_name:
        return 'vivit_small'
    elif 'modulator_only_convgru' in exp_name:
        return 'modulator_only_convgru'
    else:
        return exp_name  # Return the actual name if no pattern matches


def extract_val_loss(filename):
    """Extract validation loss from checkpoint filename."""
    # Try different patterns
    patterns = [
        r'val_loss_total=([0-9]+\.?[0-9]*)',  # Fixed pattern for decimal numbers
        r'val_loss=([0-9]+\.?[0-9]*)',
        r'loss=([0-9]+\.?[0-9]*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def extract_val_bps(filename):
    """Extract validation BPS from checkpoint filename."""
    # Try different BPS patterns
    patterns = [
        r'val_bps_overall=([0-9]+\.?[0-9]*)',
        r'val_bps=([0-9]+\.?[0-9]*)',
        r'bps=([0-9]+\.?[0-9]*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def extract_epoch(filename):
    """Extract epoch from checkpoint filename."""
    match = re.search(r'epoch=([0-9]+)', filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def load_single_dataset(model, dataset_idx):
    """
    Load a single dataset for evaluation.

    Parameters
    ----------
    model : MultiDatasetModel
        The trained model containing dataset configurations
    dataset_idx : int
        Index of the dataset to load

    Returns
    -------
    tuple
        (train_dset, val_dset, dataset_config)
    """
    import copy

    if dataset_idx >= len(model.names):
        raise ValueError(f"Dataset index {dataset_idx} out of range. Model has {len(model.names)} datasets.")

    # Try to get dataset config from various possible locations
    # Use deepcopy to avoid modifying the original config stored in the model
    if hasattr(model, 'dataset_configs'):
        dataset_config = copy.deepcopy(model.dataset_configs[dataset_idx])
    elif hasattr(model.model, 'dataset_configs'):
        dataset_config = copy.deepcopy(model.model.dataset_configs[dataset_idx])
    elif hasattr(model, 'cfgs'):
        # During training, MultiDatasetModel stores configs in self.cfgs
        dataset_config = copy.deepcopy(model.cfgs[dataset_idx])
    else:
        # Fallback: try to load from file
        # This assumes cfg_dir points to a directory with individual dataset configs
        config_path = model.hparams.cfg_dir
        dataset_name = model.names[dataset_idx]
        dataset_config_path = Path(config_path) / f"{dataset_name}.yaml"
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

    dataset_name = model.names[dataset_idx]
    
    print(f"\nLoading dataset {dataset_idx}: {dataset_name}")
    # add datafilters if missing
    # dataset_config['datafilters'] = {'dfs': {'ops': [{'valid_nlags': {'n_lags': 32}}, {'missing_pct': {'theshold': 45}}], 'expose_as': 'dfs'}}
    
    # Add additional dataset types for evaluation
    if 'backimage' not in dataset_config['types']:
        dataset_config['types'] += ['backimage']
    if 'gaborium' not in dataset_config['types']:
        dataset_config['types'] += ['gaborium']
    if 'gratings' not in dataset_config['types']:
        dataset_config['types'] += ['gratings']
    if 'fixrsvp' not in dataset_config['types']:
        dataset_config['types'] += ['fixrsvp']
    dataset_config['keys_lags']['eyepos'] = 0
    
    # Load data with suppressed output
    import contextlib
    import os
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        train_dset, val_dset, dataset_config = prepare_data(dataset_config, strict=False)
    
    print(f"✓ Dataset loaded: {len(train_dset)} train, {len(val_dset)} val samples")
    print(f"  Dataset config: {len(dataset_config.get('cids', []))} units")
    
    return train_dset, val_dset, dataset_config


def get_stim_inds(stim_type, train_data, val_data):
    """
    Get stimulus indices for different stimulus types.
    
    Parameters
    ----------
    stim_type : str
        Type of stimulus ('gaborium', 'backimage', 'fixrsvp', 'gratings')
    train_data : CombinedEmbeddedDataset
        Training dataset
    val_data : CombinedEmbeddedDataset  
        Validation dataset
        
    Returns
    -------
    torch.Tensor
        Indices for the specified stimulus type
    """
    if stim_type == 'gaborium':
        return val_data.get_dataset_inds('gaborium')
    elif stim_type == 'backimage':
        return val_data.get_dataset_inds('backimage')
    elif stim_type == 'fixrsvp':
        return torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)
    elif stim_type == 'gratings':
        return val_data.get_dataset_inds('gratings') # moved gratings to the validation set
    else:
        raise ValueError(f"Unknown stim type: {stim_type}")


def run_model(model, batch, dataset_idx):
    """
    Run the model on a batch of data.
    
    Parameters
    ----------
    model : MultiDatasetModel
        The trained model
    batch : dict
        Batch of data with 'stim', 'behavior', etc.
    dataset_idx : int
        Index of the dataset
        
    Returns
    -------
    dict
        Batch with added 'rhat' predictions
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    
    with torch.no_grad():
        if hasattr(model, 'is_modulator_only') and model.is_modulator_only:
            output = model.model(None, dataset_idx, batch.get('behavior'))
        elif hasattr(model.model, 'spike_history'):
            output = model.model(batch['stim'], dataset_idx, batch.get('behavior', None), batch.get('history', None))
        else:
            output = model.model(batch['stim'], dataset_idx, batch.get('behavior'))
        # output = model.model(batch['stim'], dataset_idx, batch.get('behavior'))
    batch['rhat'] = output
    
    if model.log_input:
        batch['rhat'] = torch.exp(batch['rhat'])

    return batch


def evaluate_dataset(model, dataset, indices, dataset_idx, batch_size=256, desc="Dataset"):
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
    dataset_idx : int
        Index of the dataset for the model
    batch_size : int, optional
        Batch size for evaluation, by default 256.
    desc : str, optional
        Description for progress bar, by default "Dataset".

    Returns
    -------
    dict
        Dictionary with keys 'robs', 'rhat', 'dfs', 'bps' containing observed responses,
        predicted responses, data flags, and bits per spike
    """
    bps_aggregator = PoissonBPSAggregator()
    dataset = dataset.shallow_copy()
    dataset.inds = indices
    robs = []
    rhat = []
    dfs = []

    with torch.no_grad():
        for iB in tqdm(range(0, len(dataset), batch_size), desc=desc):
            batch = dataset[iB:iB+batch_size]
            batch = run_model(model, batch, dataset_idx)

            robs.append(batch['robs'].detach().cpu())  # Move to CPU immediately
            rhat.append(batch['rhat'].detach().cpu())  # Move to CPU immediately
            dfs.append(batch['dfs'].detach().cpu())  # Move to CPU immediately
            bps_aggregator(batch)

            # Clean up batch tensors to free GPU memory
            del batch
            torch.cuda.empty_cache()

    robs = torch.cat(robs, dim=0)
    rhat = torch.cat(rhat, dim=0)
    dfs = torch.cat(dfs, dim=0)

    bps = bps_aggregator.closure().cpu().numpy()
    bps_aggregator.reset()

    return {'robs': robs, 'rhat': rhat, 'dfs': dfs, 'bps': bps}


def load_qc_data(sess, cids):
    """
    Load quality control data for specified cell IDs.
    
    Parameters
    ----------
    sess : YatesV1Session
        Session object
    cids : array-like
        Cell IDs to load QC data for
        
    Returns
    -------
    dict
        Dictionary containing QC metrics for each cell
    """
    qc_data = {}
    
    try:
        # Load refractory period violation metrics
        refractory = np.load(sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz')
        min_contam_proportions = refractory['min_contam_props'][cids]
        
        # Calculate contamination percentage for each unit
        contam_pct = np.array([
            np.min(min_contam_proportions[iU]) * 100
            for iU in range(len(cids))
        ])
        qc_data['contamination'] = contam_pct
        
    except Exception as e:
        print(f"Warning: Could not load refractory QC data: {e}")
        qc_data['contamination'] = np.full(len(cids), np.nan)
    
    try:
        # Load amplitude truncation metrics
        truncation = np.load(sess.sess_dir / 'qc' / 'amp_truncation' / 'truncation.npz')
        med_missing_pct = np.array([
            np.median(truncation['mpcts'][truncation['cid'] == iC])
            for iC in cids
        ])
        qc_data['truncation'] = med_missing_pct
        
    except Exception as e:
        print(f"Warning: Could not load truncation QC data: {e}")
        qc_data['truncation'] = np.full(len(cids), np.nan)
    
    try:
        # Load waveform data
        waves_full = np.load(sess.sess_dir / 'qc' / 'waveforms' / 'waveforms.npz')
        waveforms = waves_full['waveforms'][cids]
        qc_data['waveforms'] = waveforms
        qc_data['wave_times'] = waves_full['times']
        
    except Exception as e:
        print(f"Warning: Could not load waveform data: {e}")
        qc_data['waveforms'] = np.full((len(cids), 82, 384), np.nan)  # Default shape
        qc_data['wave_times'] = np.arange(82)
    
    try:
        # Load probe geometry and depth information
        ephys_meta = sess.ephys_metadata
        probe_geom = ephys_meta['probe_geometry_um']
        
        # Load laminar boundary information
        laminar_results = np.load(sess.sess_dir / 'laminar' / 'laminar.npz')
        l4_depths = laminar_results['l4_depths']
        
        qc_data['probe_geometry'] = probe_geom
        qc_data['l4_depths'] = l4_depths
        
    except Exception as e:
        print(f"Warning: Could not load probe/laminar data: {e}")
        qc_data['probe_geometry'] = None
        qc_data['l4_depths'] = None
    
    return qc_data


def get_fixrsvp_trials(model, eval_dict, dataset_idx, train_data, val_data):
    """
    Extract trial-aligned data for FixRSVP stimuli.

    Parameters
    ----------
    model : MultiDatasetModel
        The trained model
    eval_dict : dict
        Dictionary containing evaluation results with 'fixrsvp' key
    dataset_idx : int
        Index of the dataset
    train_data : CombinedEmbeddedDataset
        Training dataset
    val_data : CombinedEmbeddedDataset
        Validation dataset

    Returns
    -------
    tuple
        (robs_trial, rhat_trial, dfs_trial) - trial-aligned responses and data flags
    """
    robs = eval_dict['fixrsvp']['robs']
    rhat = eval_dict['fixrsvp']['rhat']
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

        assert torch.all(robs[eval_inds] == data.dsets[dset_idx]['robs'][data_inds]).item(), 'robs mismatch'

        robs_trial[itrial, time_inds[data_inds]] = robs[eval_inds]
        rhat_trial[itrial, time_inds[data_inds]] = rhat[eval_inds]
        dfs_trial[itrial, time_inds[data_inds]] = data.dsets[dset_idx]['dfs'][data_inds]

    return robs_trial, rhat_trial, dfs_trial


import numpy as np

def ccnorm_variable_trials(R, P, D=None, *,
                           ddof=0,
                           min_trials_per_bin=20,
                           min_time_bins=20):
    """
    Noise-corrected correlation allowing trial count N_t to vary over time,
    using an explicit mask instead of NaNs.

    Parameters
    ----------
    R : array, shape (N, T, K)
        Single-trial responses.
    P : array, same shape
        Model predictions.
    D : bool array, same shape, optional
        Valid-sample mask (True=keep). If None, samples where R is NaN are invalid.
    ddof : int
        Passed to variance/covariance (0=pop, 1=sample).
    min_trials_per_bin : int
        Require ≥ this many valid trials in a bin to use it.
    min_time_bins : int
        Require ≥ this many bins after masking, else return NaN for that neuron.

    Returns
    -------
    cc : array, shape (K,)
        CC_norm per neuron.
    """
    R = np.asarray(R, float)
    P = np.asarray(P, float)
    N, T, K = R.shape

    # Build explicit mask D
    if D is None:
        D = ~np.isnan(R)
    else:
        D = np.asarray(D, bool)

    # Prepare output
    cc = np.full(K, np.nan)

    for k in range(K):
        # slice out neuron k
        r_k = R[..., k]       # (N, T)
        p_k = P[..., k]       # (N, T)
        m_k = D[..., k]       # (N, T) mask

        # count valid trials per time-bin
        n_t = m_k.sum(axis=0)               # (T,)
        good_t = n_t >= min_trials_per_bin
        if good_t.sum() < min_time_bins:
            continue

        # masked arrays over the “good” time bins
        r_ma = np.ma.masked_array(r_k[:, good_t], mask=~m_k[:, good_t])
        p_ma = np.ma.masked_array(p_k[:, good_t], mask=~m_k[:, good_t])

        # PSTH and sample-noise variance per bin
        y_t   = r_ma.mean(axis=0).data     # shape (T_good,)
        s2_t  = r_ma.var(axis=0, ddof=ddof).data

        # explainable variance (eq ★)
        var_y = y_t.var(ddof=ddof)
        noise_corr = np.mean(s2_t / n_t[good_t])
        SP = var_y - noise_corr
        if SP <= 0 or np.isnan(SP):
            continue

        # model stats
        p_mean = p_ma.mean(axis=0).data
        cov = np.mean((y_t - y_t.mean()) * (p_mean - p_mean.mean()))
        var_p = p_mean.var(ddof=ddof)
        if var_p == 0:
            continue

        cc[k] = cov / np.sqrt(var_p * SP)

    return cc



def get_saccade_eval(stim_type, train_data, val_data, eval_dict, saccades, win=(-10, 100)):
    """
    Perform saccade-triggered analysis for a given stimulus type.

    Parameters
    ----------
    stim_type : str
        Type of stimulus ('gaborium', 'backimage', 'fixrsvp', 'gratings')
    train_data : CombinedEmbeddedDataset
        Training dataset
    val_data : CombinedEmbeddedDataset
        Validation dataset
    eval_dict : dict
        Dictionary containing evaluation results
    saccades : list of saccade dictionaries
        Output of detect saccades
    win : tuple, optional
        Time window around saccades (start, end) in bins

    Returns
    -------
    dict
        Dictionary with individual saccade responses, averages, and saccade info
    """

    # get saccade times
    saccade_times = torch.tensor([s['start_time'] for s in saccades])

    # Convert saccade times to dataset indices
    saccade_inds = train_data.get_inds_from_times(saccade_times)

    # Get stimulus indices using the helper function
    stim_inds = get_stim_inds(stim_type, train_data, val_data)

    dataset = val_data.shallow_copy()
    dataset.inds = stim_inds

    dset = stim_inds[0,0]
    print(f'Dataset {dset}')

    robs = eval_dict[stim_type]['robs']
    pred = eval_dict[stim_type]['rhat']

    nbins = win[1]-win[0]

    valid_saccades = np.where(saccade_inds[:,0]==dset)[0]
    sac_indices = saccade_inds[valid_saccades, 1]
    n_sac = len(sac_indices)

    # Initialize arrays for all saccades
    robs_sac = np.nan*np.zeros((n_sac, nbins, robs.shape[1]))
    pred_sac = np.nan*np.zeros((n_sac, nbins, pred.shape[1]))
    dfs_sac = np.nan*np.zeros((n_sac, nbins, robs.shape[1]))
    eyevel_sac = np.nan*np.zeros((n_sac, nbins, 2))  # Keep x,y separate

    saccade_info = [saccades[i] for i in valid_saccades]

    # Track inter-saccade intervals
    time_previous = np.nan*np.zeros(n_sac)
    time_next = np.nan*np.zeros(n_sac)

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

            # Extract eye velocity (keep x,y separate)
            eyepos = batch['eyepos'].detach().cpu().numpy()
            eyevel = np.gradient(eyepos, axis=0)  # Shape: [time, 2]

            robs_sac[i] = robs_
            pred_sac[i] = pred_
            dfs_sac[i] = dfs_
            eyevel_sac[i] = eyevel

        # Calculate inter-saccade intervals (do this for all saccades, not just valid data ones)
        if i > 0:
            prev_sac_bin = isac - sac_indices[i-1]
            time_previous[i] = prev_sac_bin

        if i < len(sac_indices)-1:
            next_sac_bin = sac_indices[i+1] - isac
            time_next[i] = next_sac_bin

    # Filter for good saccades (no NaN values)
    good = np.where(np.sum(np.isnan(robs_sac), axis=(1,2)) == 0)[0]

    # Apply inter-saccade interval filtering (same logic as model_analysis_devel)
    dt = 1/120  # Sampling rate
    validix = np.ones(len(good), dtype=bool)

    for idx, g in enumerate(good):
        # Check if eye velocity has NaN values
        if np.isnan(eyevel_sac[g]).any():
            validix[idx] = False
            continue

        # Check inter-saccade intervals
        tn = time_next[g] * dt
        tp = time_previous[g] * dt

        # Apply same filtering logic as model_analysis_devel
        if not np.isnan(tn) and not (0.1 < tn < 0.5):
            validix[idx] = False
        if not np.isnan(tp) and not (0.1 < tp < 0.5):
            validix[idx] = False

    # Get final valid saccades
    valid_final = good[validix]

    robs_sac = robs_sac[valid_final]
    pred_sac = pred_sac[valid_final]
    dfs_sac = dfs_sac[valid_final]
    eyevel_sac = eyevel_sac[valid_final]

    # Update saccade_info with time intervals
    saccade_info_final = []
    for idx, g in enumerate(valid_final):
        info = saccade_info[g].copy()
        info['time_previous'] = time_previous[g]
        info['time_next'] = time_next[g]
        saccade_info_final.append(info)

    print(f"Number of valid saccades after filtering: {len(valid_final)}")

    # Compute averages
    rbar = np.nansum(robs_sac*dfs_sac, axis=0) / np.nansum(dfs_sac, axis=0)
    rbarhat = np.nansum(pred_sac*dfs_sac, axis=0) / np.nansum(dfs_sac, axis=0)

    return {
        'robs': robs_sac,
        'rhat': pred_sac,  # Changed from 'pred' to 'rhat' for consistency
        'dfs': dfs_sac,
        'rbar': rbar,
        'rbarhat': rbarhat,
        'eyevel': eyevel_sac,
        'saccade_info': saccade_info_final,
        'win': win
    }


def detect_saccades_from_session(sess):
    """
    Load or detect saccades from a session.

    Parameters
    ----------
    sess : YatesV1Session
        Session object

    Returns
    -------
    torch.Tensor
        Saccade times
    """
    try:
        # Try to load from JSON first
        saccades = json.load(open(sess.sess_dir / 'saccades' / 'saccades.json'))
        saccade_times = torch.tensor([s['start_time'] for s in saccades])
    except:
        # Fall back to detect_saccades if available
        try:
            from jake.detect_saccades import detect_saccades
            saccades = detect_saccades(sess)
            saccade_times = torch.sort(torch.tensor([s.start_time for s in saccades])).values
        except ImportError:
            print("Warning: Could not load or detect saccades")
            return torch.tensor([])

    # Filter saccades with minimum ISI
    valid = np.diff(saccade_times.numpy(), prepend=0) > 0.06
    
    saccades = [saccades[i] for i in np.where(valid)[0]]

    # clean up saccades based on the main sequence
    vel = np.array([s['A'] for s in saccades])
    amp = np.array([np.hypot(s['end_x']-s['start_x'], s['end_y']-s['start_y']) for s in saccades])

    rat = vel/amp
    med = np.median(rat)
    mad = np.median(np.abs(rat-med))
    inliers = np.where(np.abs(rat-med) < 3*mad)[0]
    saccades = [saccades[i] for i in inliers]

    return saccades

def _argext_subpixel(a, axis, mode):
    """
    Core implementation. Assumes `axis` is an int.
    Returns (idx_sub, val_sub) for the 1D slices along `axis`.
    """
    # For minima, flip the sign
    arr = -a if mode == 'min' else a
    # find integer peak
    idx = np.argmax(arr, axis=axis)
    # gather peak values
    idx_exp = np.expand_dims(idx, axis)
    v0 = np.take_along_axis(arr, idx_exp, axis=axis).squeeze(axis)

    # neighbor integer indices
    N = a.shape[axis]
    idx_m = np.clip(idx - 1, 0, N - 1)
    idx_p = np.clip(idx + 1, 0, N - 1)
    v_m = np.take_along_axis(arr, np.expand_dims(idx_m, axis), axis=axis).squeeze(axis)
    v_p = np.take_along_axis(arr, np.expand_dims(idx_p, axis), axis=axis).squeeze(axis)

    # parabolic interpolation formula:
    #   offset = (v_- - v_+) / (2*(v_- - 2*v0 + v_+))
    #   v_sub = v0 - (v_- - v_+)² / [8*(v_- - 2*v0 + v_+)]
    denom = (v_m - 2*v0 + v_p)
    num   = (v_m - v_p)

    # compute offset safely
    with np.errstate(divide='ignore', invalid='ignore'):
        offset = num / (2 * denom)
    # clamp offsets at boundaries or flat peaks
    at_edge = (idx == 0) | (idx == N-1) | (denom == 0)
    offset = np.where(at_edge, 0.0, offset)

    # subpixel indices & values
    idx_sub = idx.astype(np.float64) + offset
    v_sub = v0 - (num**2) / (8 * denom)
    v_sub = np.where(at_edge, v0, v_sub)

    # for minima, flip back the sign
    if mode == 'min':
        v_sub = -v_sub

    return idx_sub, v_sub


def _argext_dispatch(a, axis, mode):
    a = np.asanyarray(a)
    # axis=None: flatten
    if axis is None:
        flat = a.ravel()
        idx_sub, v_sub = _argext_subpixel(flat, 0, mode)
        return idx_sub, v_sub

    # normalize negative axes
    axis = axis if axis >= 0 else a.ndim + axis
    return _argext_subpixel(a, axis, mode)


def argmax_subpixel(a, axis=None):
    """
    Like np.argmax, but returns sub-pixel refined peak and its value.
    
    Parameters
    ----------
    a : array_like
    axis : int or None, optional
        Axis along which to find the maximum. Default is None (flattened).

    Returns
    -------
    idx_sub : float or ndarray of floats
        Sub-pixel estimate of the index of the maximum.
    val_sub : float or ndarray of floats
        Interpolated maximum value.
    """
    return _argext_dispatch(a, axis, mode='max')


def argmin_subpixel(a, axis=None):
    """
    Like np.argmin, but returns sub-pixel refined trough and its value.
    
    Parameters
    ----------
    a : array_like
    axis : int or None, optional
        Axis along which to find the minimum. Default is None (flattened).

    Returns
    -------
    idx_sub : float or ndarray of floats
        Sub-pixel estimate of the index of the minimum.
    val_sub : float or ndarray of floats
        Interpolated minimum value.
    """
    return _argext_dispatch(a, axis, mode='min')

def get_model_name(model_name):

    if 'learned_res_small_gru' in model_name:
        return 'Full'
    elif 'learned_res_small_pc' in model_name:
        return 'PC'
    elif 'learned_res_small_film' in model_name:
        return 'FiLM'
    elif 'learned_res_small_stn' in model_name:
        return 'STN'
    elif 'learned_res_small' in model_name:
        return 'Vision'
    else:
        return model_name
    


def _poisson_nll_from_rates_indexed(rate_v, robs_v, eps=1e-12):
    rate_v = torch.clamp(rate_v, min=eps)
    robs_v = torch.nan_to_num(robs_v, nan=0.0, posinf=0.0, neginf=0.0)
    # mean over valid entries only
    return (rate_v - robs_v * torch.log(rate_v)).mean()

class _RescaleModelExp(nn.Module):
    def __init__(self, N, mode='globalgain', init_g=None, init_b=None, device=None, dtype=None):
        super().__init__()
        self.mode = mode
        kw = dict(device=device, dtype=dtype)
        to_param = lambda v: nn.Parameter(torch.as_tensor(v, **kw))
        if mode == 'globalgain':
            self.g = to_param(0.0 if init_g is None else init_g)
        elif mode == 'gain':
            self.g = to_param(torch.zeros(N) if init_g is None else init_g)
        elif mode == 'globalaffine':
            self.g = to_param(0.0 if init_g is None else init_g)
            self.b = to_param(0.0 if init_b is None else init_b)
        elif mode == 'affine':
            self.g = to_param(torch.zeros(N) if init_g is None else init_g)
            self.b = to_param(torch.zeros(N) if init_b is None else init_b)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, rhat):
        eg = torch.exp(self.g)
        if self.mode == 'globalgain':
            out = eg * rhat
        elif self.mode == 'gain':
            out = eg.unsqueeze(0) * rhat
        elif self.mode == 'globalaffine':
            out = eg * rhat + torch.exp(self.b)
        elif self.mode == 'affine':
            out = eg.unsqueeze(0) * rhat + torch.exp(self.b).unsqueeze(0)
        return out  # rates; positivity enforced via exp-params

def rescale_rhat(
    robs: torch.Tensor,
    rhat: torch.Tensor,
    dfs: torch.Tensor,
    mode: str = 'globalgain',
    max_iter: int = 200,
    tol: float = 1e-9,
    history_size: int = 100,
    line_search_fn: str = "strong_wolfe",
    eps: float = 1e-12,
):
    assert robs.shape == rhat.shape == dfs.shape, "robs, rhat, dfs must be T x N"
    device, dtype = rhat.device, rhat.dtype
    T, N = rhat.shape

    # 1) Build a strict boolean mask and sanitize BEFORE any math.
    valid = (dfs > 0.5)
    rhat_clean = torch.where(valid, torch.nan_to_num(rhat, nan=0.0, posinf=0.0, neginf=0.0),
                             torch.zeros_like(rhat))
    robs_clean = torch.where(valid, torch.nan_to_num(robs, nan=0.0, posinf=0.0, neginf=0.0),
                             torch.zeros_like(robs))

    # 2) Analytic-ish initialization using only valid entries (no NaN*0 anywhere).
    # sums per column over valid entries
    sum_rhat = rhat_clean.sum(dim=0) + eps
    sum_robs = robs_clean.sum(dim=0)

    if mode in ('globalgain', 'globalaffine'):
        g0 = torch.log((sum_robs.sum() / sum_rhat.sum()).clamp(min=eps)).to(dtype=dtype, device=device)
        if mode == 'globalaffine':
            # mean over valid entries
            n_valid = valid.sum().clamp_min(1).to(dtype)
            mean_r = rhat_clean.sum() / n_valid
            mean_y = robs_clean.sum() / n_valid
            b0 = torch.log(torch.clamp(mean_y - torch.exp(g0) * torch.clamp(mean_r, min=0.0), min=eps))
        else:
            b0 = None
    elif mode in ('gain', 'affine'):
        g0_vec = torch.log((sum_robs / sum_rhat).clamp(min=eps)).to(dtype=dtype, device=device)
        if mode == 'affine':
            count_valid = valid.sum(dim=0).clamp_min(1).to(dtype)
            mean_r = rhat_clean.sum(dim=0) / count_valid
            mean_y = robs_clean.sum(dim=0) / count_valid
            b0_vec = torch.log(torch.clamp(mean_y - torch.exp(g0_vec) * torch.clamp(mean_r, min=0.0), min=eps))
        else:
            b0_vec = None
        g0, b0 = g0_vec, b0_vec
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = _RescaleModelExp(N, mode=mode, init_g=g0, init_b=b0, device=device, dtype=dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )

    # 3) Loss uses boolean indexing; only valid entries participate.
    v = valid  # alias
    def closure():
        optim.zero_grad(set_to_none=True)
        pred = model(rhat_clean)
        loss = _poisson_nll_from_rates_indexed(pred[v], robs_clean[v], eps=eps)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        return loss

    optim.step(closure)

    with torch.no_grad():
        rhat_rescaled = model(rhat_clean)

    return rhat_rescaled, model

import math
def bits_per_spike(r_pred, r_obs, dfs=None):
    ''' 
    Calculate the Poisson log likelihood of the observed data given the predicted rates

    Parameters
    ----------
    r_pred : torch.Tensor (n_samples, n_units) or (n_samples,)
        Predicted spike rates
    r_obs : torch.Tensor (n_samples, n_units) or (n_samples,)
        Observed spike rates
    dfs : torch.Tensor (n_samples, n_units) or (n_samples,), optional
        Data filters for each unit
    
    Returns
    -------
    Iss : torch.Tensor (n_units)
        Information per spike for each unit
    '''

    if dfs is None:
        # dfs not is nan
        dfs = ~(torch.isnan(r_obs) | torch.isnan(r_pred)).to(r_obs.dtype)
    
    # safe handle nans in all tensors
    r_pred = torch.where(torch.isnan(r_pred), torch.zeros_like(r_pred), r_pred)
    r_obs = torch.where(torch.isnan(r_obs), torch.zeros_like(r_obs), r_obs)

    with torch.no_grad():
        T = dfs.sum(dim=0).clamp(1)
        N = (dfs * r_obs).sum(dim=0).clamp(1)
        r_bar = N / T
        # this assumes that the sum of the model predictions = the total number of spikes
        # Iss = (r_obs * dfs / r_bar * torch.log2(r_pred * dfs / r_bar)).nansum(dim=0) / T
        
        # separate each term explicitly
        ll_pred = r_obs * torch.log(r_pred + 1e-8) - r_pred
        ll_null = r_obs * torch.log(r_bar + 1e-8) - r_bar
        
        Iss = (ll_pred - ll_null) * dfs
        Iss = Iss.sum(dim=0) / N / math.log(2) # log-likelihood ratio per spike (in bits)

    return Iss

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

def scatter_kde_horizontal(x, y, slice_spacing=200, bandwidth=None,
                           color="gray", cmap=None, ax=None):
    if ax is None: fig, ax = plt.subplots()
    ax.scatter(x, y, s=10, c="red", alpha=0.6)
    ybins = np.arange(np.nanmin(y), np.nanmax(y), slice_spacing)
    for i, yc in enumerate(ybins):
        mask = (y >= yc - slice_spacing/2) & (y < yc + slice_spacing/2)
        if mask.sum() < 5: continue
        kde = gaussian_kde(x[mask], bw_method=bandwidth)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        dens = kde(xx); dens /= np.nanmax(dens)
        offset = yc
        c = color if cmap is None else plt.cm.get_cmap(cmap)(i/len(ybins))
        ax.fill_between(xx, offset, offset + dens*slice_spacing*0.8,
                        color=c, alpha=0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    return ax


def get_checkpoint_dir(experiment):
    """
    Get the checkpoint and cache directories for a given experiment.

    Parameters
    ----------
    experiment : str
        Name of the experiment ('backimage_only_120', 'gaborium_only_120', 'full_120')

    Returns
    -------
    checkpoint_dir : str
        Path to the checkpoint directory
    cache_dir : str
        Path to the cache directory
    """
    if experiment == 'backimage_only_120':
        checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage/checkpoints'
    elif experiment == 'gaborium_only_120':
        checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_gaborium/checkpoints'
    else:
        checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'

    cache_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_caches/" + experiment
    return checkpoint_dir, cache_dir
    
def get_all_results_by_experiment(experiment, models_to_compare):
    """
    Load all results for a given experiment.

    Parameters
    ----------
    experiment : str
        Name of the experiment ('backimage_only_120', 'gaborium_only_120', 'full_120')
    models_to_compare : list
        List of models to compare

    Returns
    -------
    all_results : dict
        Dictionary of all results
    """
    checkpoint_dir, cache_dir = get_checkpoint_dir(experiment)
    models_by_type = scan_checkpoints(checkpoint_dir)
    available_models = [m for m in models_to_compare if m in models_by_type]

    all_results = {}
    for model_type in available_models:
        print(f"\nLoading {model_type}...")
        
        results = evaluate_model_multidataset(
            model_type=model_type,
            analyses=['bps', 'ccnorm', 'saccade', 'qc'],
            checkpoint_dir=checkpoint_dir,
            save_dir=cache_dir,
            recalc=False,
            rescale=True,
            batch_size=64
        )
        all_results.update(results)
    
    # all_results = run_gratings_analysis(all_results, checkpoint_dir, cache_dir, recalc=True, batch_size=64)

    return all_results


def shift_tensor(x: torch.Tensor,
                    shifts: torch.Tensor,
                    shift_dim: int,
                    batch_dim: int = 0,
                    mode: str = "constant",
                    fill_value: float = 0.0):
    """
    Shift each slice indexed by `batch_dim` along `shift_dim` by `shifts[i]`.
    Same shift applies across the remaining dims of the slice.

    Example: x.shape = (N, H, W), shifts.shape = (N,), shift_dim=1 shifts rows for all columns.
    """
    assert x.size(batch_dim) == shifts.numel(), "shifts must match size of batch_dim"

    # Reorder so (batch_dim, shift_dim) -> (0, -1): (N, ..., L)
    t = x.movedim((batch_dim, shift_dim), (0, -1))
    N, *mid, L = t.shape
    M = int(torch.tensor(mid).prod()) if mid else 1
    t2 = t.reshape(N, M, L)                        # (N, M, L)

    shifts = shifts.to(device=x.device, dtype=torch.long).view(N)
    shifts_full = shifts[:, None].expand(N, M).reshape(-1)   # (N*M,)
    flat = t2.reshape(N * M, L)
    idx = torch.arange(L, device=x.device)                   # (L,)

    if mode == "wrap":
        src = (idx[None, :] - shifts_full[:, None]) % L      # (N*M, L)
        out = flat.gather(1, src)
    else:
        src = idx[None, :] - shifts_full[:, None]
        valid = (src >= 0) & (src < L)
        src_clamped = src.clamp(0, L - 1)
        gathered = flat.gather(1, src_clamped)
        out = torch.full_like(flat, fill_value)
        out[valid] = gathered[valid]

    out = out.view(N, M, L).reshape(N, *mid, L)
    return out.movedim((0, -1), (batch_dim, shift_dim))
