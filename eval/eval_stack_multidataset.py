"""
Unified Evaluation Stack for Multidataset Models

This module provides a unified evaluation pipeline that can compare outputs across
multiple models and datasets. It builds upon the existing model_load_eval_stack_multidataset.py
but provides a cleaner interface for cross-model comparisons.

Key features:
- Automatic model discovery (best model by type) or specific checkpoint loading
- Caching per model/dataset/analysis type
- Cell tracking with CID mapping across datasets
- QC data integration
- Multiple analysis types: BPS, CCNORM, saccade-triggered responses
- Graceful error handling with NaN population for failed analyses

Author: Built from model_load_eval_stack_multidataset.py
"""

import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# DataYatesV1 package imports
from DataYatesV1 import get_session
from models.losses import PoissonBPSAggregator

# Import the training module to access the model class
from training import MultiDatasetModel

# Import our utility functions
from .eval_stack_utils import (
    load_single_dataset, get_stim_inds, evaluate_dataset, load_qc_data,
    get_fixrsvp_trials, get_saccade_eval, ccnorm_split_half_variable_trials,
    detect_saccades_from_session, scan_checkpoints, extract_val_loss, extract_epoch
)


def load_model(model_type=None, model_index=None, checkpoint_path=None,
               checkpoint_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints",
               device='cuda',
               verbose=True):
    """
    Load a model either by type (with automatic best selection) or by specific checkpoint path.
    
    Parameters
    ----------
    model_type : str, optional
        Type of model to load ('learned_res', 'resnet', etc.)
    model_index : int, optional
        Index of model to load (1-based), None for best model
    checkpoint_path : str, optional
        Specific checkpoint path to load
    checkpoint_dir : str
        Directory containing checkpoints
    device : str
        Device to load model on
        
    Returns
    -------
    tuple
        (model, model_info) where model_info contains metadata
    """
    if checkpoint_path is not None:
        # Load specific checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model_info = {
            'path': checkpoint_path,
            'experiment': checkpoint_path.parent.name,
            'val_loss': extract_val_loss(checkpoint_path.name),
            'epoch': extract_epoch(checkpoint_path.name)
        }
        
    else:
        # Discover models by type
        if model_type is None:
            raise ValueError("Must specify either model_type or checkpoint_path")
        
        models_by_type = scan_checkpoints(checkpoint_dir)
        
        if model_type not in models_by_type:
            raise ValueError(f"Model type '{model_type}' not found. Available: {list(models_by_type.keys())}")
        
        models = models_by_type[model_type]
        
        if model_index is None:
            selected_model = models[0]  # Best model
            print(f"Loading BEST {model_type} model...")
        else:
            if model_index < 0 or model_index > len(models)-1:
                raise ValueError(f"Model index {model_index} out of range (0-{len(models)-1}).")
            selected_model = models[model_index]
            print(f"Loading {model_type} model #{model_index}...")
        
        model_info = selected_model
        checkpoint_path = selected_model['path']

    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Val Loss: {model_info.get('val_loss', 'Unknown')}")
    print(f"   Epoch: {model_info.get('epoch', 'Unknown')}")

    # Load the model - need to change to the correct directory for config loading
    original_cwd = os.getcwd()

    try:
        # Change to the multidataset_ddp directory where configs are located
        multidataset_dir = Path(__file__).parent.parent
        os.chdir(multidataset_dir)

        # Load model with proper checkpoint handling
        if verbose:
            print(f"Loading model with checkpoint compatibility...")

        # First, examine the checkpoint to detect key mismatch issues
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict_keys = list(state_dict.keys())

            # Check for torch.compile key mismatch
            has_orig_mod_prefix = any(key.startswith('model._orig_mod.') for key in state_dict_keys)

            if has_orig_mod_prefix:
                if verbose:
                    print(f"   Detected torch.compile checkpoint - fixing key mismatch...")

                # Create model first
                model = MultiDatasetModel.load_from_checkpoint(
                    str(checkpoint_path),
                    strict=False,
                    map_location='cpu'
                )

                # Fix the state dict keys
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model._orig_mod.'):
                        # Remove the model._orig_mod. prefix
                        new_key = key[len('model._orig_mod.'):]
                        fixed_state_dict[new_key] = value
                    else:
                        fixed_state_dict[key] = value

                # Manually load the fixed state dict
                missing_keys, unexpected_keys = model.model.load_state_dict(fixed_state_dict, strict=False)

                if verbose:
                    print(f"   Fixed {len([k for k in state_dict_keys if k.startswith('model._orig_mod.')])} parameter keys")
                    if missing_keys:
                        print(f"   {len(missing_keys)} missing keys (this may be normal)")
                    if unexpected_keys:
                        print(f"   {len(unexpected_keys)} unexpected keys")

            else:
                if verbose:
                    print(f"   Standard checkpoint - loading normally...")
                model = MultiDatasetModel.load_from_checkpoint(
                    str(checkpoint_path),
                    strict=False,
                    map_location='cpu'
                )
        else:
            if verbose:
                print(f"   No state_dict found in checkpoint - loading as-is...")
            model = MultiDatasetModel.load_from_checkpoint(
                str(checkpoint_path),
                strict=False,
                map_location='cpu'
            )

        model.to(device)
        model.eval()

        print("✓ Model loaded successfully!")

        # Get model info
        if verbose:
            print(f"\nModel Information:")
            print(f"  Datasets: {len(model.names)}")
            print(f"  Dataset names: {model.names}")
            print(f"  Activation: {type(model.model.activation).__name__}")

            # Count parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

        return model, model_info

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)


def check_existing_cache_files(model_name, save_dir, num_datasets, analyses, rescale=False):
    """
    Check which analysis cache files already exist for a model.

    Parameters
    ----------
    model_name : str
        Name of the model
    save_dir : Path
        Directory where cache files are stored
    num_datasets : int
        Number of datasets in the model
    analyses : list
        List of analyses to check
    rescale : bool, optional
        Whether to check for rescaled cache files (default: False)

    Returns
    -------
    dict
        Dictionary with analysis names as keys and lists of missing dataset indices as values
    """
    save_dir = Path(save_dir)
    missing_analyses = {analysis: [] for analysis in analyses}

    for dataset_idx in range(num_datasets):
        for analysis in analyses:
            # QC analysis never uses rescale suffix (it's independent of model predictions)
            rescale_suffix = '_rescaled' if rescale and analysis != 'qc' else ''
            cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_{analysis}{rescale_suffix}_cache.pt'
            if not cache_file.exists():
                missing_analyses[analysis].append(dataset_idx)

    return missing_analyses


def evaluate_model_multidataset(model_type='learned_res',
                model_index=None, checkpoint_path=None,
                checkpoint_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/checkpoints",
                save_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack",
                analyses=['bps', 'ccnorm', 'saccade'],
                recalc=False, batch_size=64, device='cuda', rescale=False):
    """
    Unified evaluation pipeline for multidataset models.

    Parameters
    ----------
    model_type : str, optional
        Type of model to load ('learned_res', 'resnet', etc.)
    model_index : int, optional
        Index of model to load (1-based), None for best model
    checkpoint_path : str, optional
        Specific checkpoint path to load
    checkpoint_dir : str
        Directory containing checkpoints
    analyses : list
        List of analyses to run: ['bps', 'ccnorm', 'saccade']
    recalc : bool
        Whether to recalculate cached results
    batch_size : int
        Batch size for evaluation
    device : str
        Device to run evaluation on
    rescale : bool, optional
        Whether to apply affine rescaling to rhat after BPS analysis (default: False)

    Returns
    -------
    dict
        Unified evaluation results with structure:
        {
            'model_name': {
                'bps': {...},
                'ccnorm': {...},
                'saccade': {...},
                'qc': {...}
            }
        }
    """
    # Load model
    model, model_info = load_model(
        model_type=model_type,
        model_index=model_index,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    model_name = model_info['experiment']
    print(f"Model name: {model_name}")
    save_dir = Path(save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check existing cache files if not recalculating
    if not recalc:
        missing_analyses = check_existing_cache_files(model_name, save_dir, len(model.names), analyses, rescale)

        # Report cache status
        total_cache_files = len(model.names) * len(analyses)
        existing_cache_files = total_cache_files - sum(len(missing) for missing in missing_analyses.values())

        print(f"Cache status: {existing_cache_files}/{total_cache_files} cache files exist")
        for analysis, missing_datasets in missing_analyses.items():
            if missing_datasets:
                print(f"   {analysis}: missing {len(missing_datasets)} datasets {missing_datasets}")
            else:
                print(f"   {analysis}: all datasets cached")

    else:
        missing_analyses = {analysis: list(range(len(model.names))) for analysis in analyses}

    print(f"Evaluating model on {len(model.names)} datasets...")

    # Initialize results structure using clean model_type name
    results = {
        model_type: {
            'dataset_names': model.names,
            'bps': {},
            'ccnorm': {},
            'saccade': {},
            'qc': {}
        }
    }

    # Define saccade window, only needed if 'saccade' requested
    sac_win = (-50, 100)

    # Track all cells across datasets
    all_cids = []
    all_datasets = []
    all_dataset_indices = []

    # Determine which datasets need processing
    datasets_to_process = set()
    for analysis, missing_datasets in missing_analyses.items():
        datasets_to_process.update(missing_datasets)

    if not datasets_to_process:
        print("All analyses are cached! Loading results from cache...")
    else:
        print(f"Processing {len(datasets_to_process)} datasets with missing analyses: {sorted(datasets_to_process)}")

    # Process each dataset
    for dataset_idx in range(len(model.names)):
        dataset_name = model.names[dataset_idx]

        # Check if this dataset needs any processing
        needs_processing = dataset_idx in datasets_to_process

        print(f"\n{'='*60}")
        print(f"Processing dataset {dataset_idx}: {dataset_name}")
        if not needs_processing:
            print("📁 All analyses cached - loading from cache only")
        print(f"{'='*60}")

        # Only load dataset if we need to run analyses (not just load from cache)
        if needs_processing:
            # Load dataset
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

            # Get CIDs for this dataset
            dataset_cids = dataset_config.get('cids', [])
            all_cids.extend(dataset_cids)
            all_datasets.extend([dataset_name] * len(dataset_cids))
            all_dataset_indices.extend([dataset_idx] * len(dataset_cids))
        else:
            # For cache-only loading, get CIDs from cache files
            dataset_cids = None
            train_data, val_data, dataset_config = None, None, None

            # Extract CIDs from existing cache files (should be in new format with 'cids' key)
            for analysis in analyses:
                cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_{analysis}_cache.pt'
                if cache_file.exists():
                    cache_data = torch.load(cache_file, weights_only=False)    
                    dataset_cids = cache_data['cids']
                    break

            # Assert that we successfully extracted CIDs from cache
            assert dataset_cids is not None, f"Could not extract CIDs from cache files for dataset {dataset_idx}. Run with recalc=True to regenerate cache files with proper format."

            all_cids.extend(dataset_cids)
            all_datasets.extend([dataset_name] * len(dataset_cids))
            all_dataset_indices.extend([dataset_idx] * len(dataset_cids))

        # Run BPS analysis if requested and missing for this dataset
        bps_results = None
        if 'bps' in analyses:
            if dataset_idx in missing_analyses['bps'] or recalc:
                print("Running BPS analysis...")
                bps_results = run_bps_analysis(
                    model, train_data, val_data, dataset_idx,
                    model_name=model_name, save_dir=save_dir, recalc=recalc,
                    batch_size=batch_size, rescale=rescale
                )
            else:
                print("Loading BPS analysis from cache...")
                rescale_suffix = '_rescaled' if rescale else ''
                cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_bps{rescale_suffix}_cache.pt'
                bps_results = torch.load(cache_file, weights_only=False)

            # Store results for each stimulus type
            if bps_results is not None:
                for stim_type, result in bps_results.items():
                    if stim_type in ['val', 'cids']:  # Skip validation BPS and cids metadata
                        continue

                    if stim_type not in results[model_type]['bps']:
                        results[model_type]['bps'][stim_type] = {
                            'robs': [], 'rhat': [], 'dfs': [], 'bps': [],
                            'cids': [], 'datasets': [], 'dataset_indices': []
                        }

                    results[model_type]['bps'][stim_type]['robs'].append(result['robs'])
                    results[model_type]['bps'][stim_type]['rhat'].append(result['rhat'])
                    results[model_type]['bps'][stim_type]['dfs'].append(result['dfs'])
                    results[model_type]['bps'][stim_type]['bps'].append(result['bps'])
                    results[model_type]['bps'][stim_type]['cids'].extend(dataset_cids)
                    results[model_type]['bps'][stim_type]['datasets'].extend([dataset_name] * len(dataset_cids))
                    results[model_type]['bps'][stim_type]['dataset_indices'].extend([dataset_idx] * len(dataset_cids))

        # Run CCNORM analysis if requested and missing for this dataset
        ccnorm_results = None
        if 'ccnorm' in analyses:
            if dataset_idx in missing_analyses['ccnorm'] or recalc:
                if train_data is None or val_data is None:
                    print("Cannot run CCNORM analysis: dataset not loaded, skipping analysis")
                else:
                    print("Running CCNORM analysis...")
                    ccnorm_results = run_ccnorm_analysis(
                        model, train_data, val_data, dataset_idx,
                        bps_results=bps_results if 'bps' in analyses else None,
                        model_name=model_name, save_dir=save_dir, recalc=recalc, rescale=rescale
                    )
            else:
                print("Loading CCNORM analysis from cache...")
                rescale_suffix = '_rescaled' if rescale else ''
                cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_ccnorm{rescale_suffix}_cache.pt'
                ccnorm_results = torch.load(cache_file, weights_only=False)

            if ccnorm_results is not None:
                if 'fixrsvp' not in results[model_type]['ccnorm']:
                    results[model_type]['ccnorm']['fixrsvp'] = {
                        'ccnorm': [], 'rbar': [], 'rbarhat': [],
                        'robs_trial': [], 'rhat_trial': [], 'dfs_trial': [],
                        'cids': [], 'datasets': [], 'dataset_indices': []
                    }

                results[model_type]['ccnorm']['fixrsvp']['ccnorm'].append(ccnorm_results['ccnorm'])
                results[model_type]['ccnorm']['fixrsvp']['rbar'].append(ccnorm_results['rbar'])
                results[model_type]['ccnorm']['fixrsvp']['rbarhat'].append(ccnorm_results['rbarhat'])
                results[model_type]['ccnorm']['fixrsvp']['robs_trial'].append(ccnorm_results['robs_trial'])
                results[model_type]['ccnorm']['fixrsvp']['rhat_trial'].append(ccnorm_results['rhat_trial'])
                results[model_type]['ccnorm']['fixrsvp']['dfs_trial'].append(ccnorm_results['dfs_trial'])
                results[model_type]['ccnorm']['fixrsvp']['cids'].extend(dataset_cids)
                results[model_type]['ccnorm']['fixrsvp']['datasets'].extend([dataset_name] * len(dataset_cids))
                results[model_type]['ccnorm']['fixrsvp']['dataset_indices'].extend([dataset_idx] * len(dataset_cids))

        # Run saccade analysis if requested and missing for this dataset
        saccade_results = None
        if 'saccade' in analyses:
            if dataset_idx in missing_analyses['saccade'] or recalc:
                if train_data is None or val_data is None:
                    print("Cannot run Saccade analysis: dataset not loaded, skipping analysis")
                else:
                    print("Running Saccade analysis...")
                    saccade_results = run_saccade_analysis(
                        model, train_data, val_data, dataset_idx,
                        bps_results=bps_results if 'bps' in analyses else None,
                        model_name=model_name, save_dir=save_dir, recalc=recalc,
                        rescale=rescale, sac_win=sac_win
                    )
            else:
                print("Loading Saccade analysis from cache...")
                rescale_suffix = '_rescaled' if rescale else ''
                cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_saccade{rescale_suffix}_cache.pt'
                saccade_results = torch.load(cache_file, weights_only=False)
            
            if saccade_results is not None:
                # cleanup

                for stim_type, sac_data in saccade_results.items():

                    if stim_type == 'cids':  # Skip cids metadata
                        continue

                    if stim_type not in results[model_type]['saccade']:
                        results[model_type]['saccade'][stim_type] = {
                            'robs': [], 'rhat': [], 'dfs': [],
                            'rbar': [], 'rbarhat': [],
                            'eyevel': [], 'saccade_info': [], 'win': [],
                            'cids': [], 'datasets': [], 'dataset_indices': []
                        }

                    # Extract all fields from saccade_results
                    results[model_type]['saccade'][stim_type]['robs'].append(sac_data['robs'])
                    results[model_type]['saccade'][stim_type]['rhat'].append(sac_data['rhat'])
                    results[model_type]['saccade'][stim_type]['dfs'].append(sac_data['dfs'])
                    results[model_type]['saccade'][stim_type]['rbar'].append(sac_data['rbar'])
                    results[model_type]['saccade'][stim_type]['rbarhat'].append(sac_data['rbarhat'])
                    results[model_type]['saccade'][stim_type]['eyevel'].append(sac_data['eyevel'])
                    results[model_type]['saccade'][stim_type]['saccade_info'].append(sac_data['saccade_info'])
                    results[model_type]['saccade'][stim_type]['win'].append(sac_data['win'])
                    results[model_type]['saccade'][stim_type]['cids'].extend(dataset_cids)
                    results[model_type]['saccade'][stim_type]['datasets'].extend([dataset_name] * len(dataset_cids))
                    results[model_type]['saccade'][stim_type]['dataset_indices'].extend([dataset_idx] * len(dataset_cids))

        # Run STA analysis if requested and missing for this dataset
        sta_results = None
        if 'sta' in analyses:
            if dataset_idx in missing_analyses.get('sta', []) or recalc:
                print("🔍 Running STA analysis...")
                sta_results = run_sta_analysis(
                    model, train_data, val_data, dataset_idx,
                    bps_results=bps_results if 'bps' in analyses else None,
                    model_name=model_name, save_dir=save_dir, recalc=recalc, rescale=rescale
                )
            else:
                print("📁 Loading STA analysis from cache...")
                rescale_suffix = '_rescaled' if rescale else ''
                cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_sta{rescale_suffix}_cache.pt'
                sta_results = torch.load(cache_file, weights_only=False)

            if sta_results is not None:
                if 'sta' not in results[model_type]:
                    results[model_type]['sta'] = {
                        'sta_robs': [], 'ste_robs': [], 'sta_rhat': [], 'ste_rhat': [],
                        'norm_dfs': [], 'norm_robs': [], 'norm_rhat': [],
                        'cids': [], 'datasets': [], 'dataset_indices': []
                    }

                results[model_type]['sta']['sta_robs'].append(sta_results['sta_robs'])
                results[model_type]['sta']['ste_robs'].append(sta_results['ste_robs'])
                results[model_type]['sta']['sta_rhat'].append(sta_results['sta_rhat'])
                results[model_type]['sta']['ste_rhat'].append(sta_results['ste_rhat'])
                results[model_type]['sta']['norm_dfs'].append(sta_results['norm_dfs'])
                results[model_type]['sta']['norm_robs'].append(sta_results['norm_robs'])
                results[model_type]['sta']['norm_rhat'].append(sta_results['norm_rhat'])
                results[model_type]['sta']['cids'].extend(dataset_cids)
                results[model_type]['sta']['datasets'].extend([dataset_name] * len(dataset_cids))
                results[model_type]['sta']['dataset_indices'].extend([dataset_idx] * len(dataset_cids))

        # Run QC analysis if requested and missing for this dataset
        qc_results = None
        if 'qc' in analyses:
            if dataset_idx in missing_analyses['qc'] or recalc:
                print("🔍 Running QC analysis...")
                qc_results = run_qc_analysis(
                    dataset_name, dataset_cids, dataset_idx,
                    model_name=model_name, save_dir=save_dir, recalc=recalc
                )
            else:
                print("📁 Loading QC analysis from cache...")
                cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_qc_cache.pt'
                qc_results = torch.load(cache_file, weights_only=False)

            if qc_results is not None:
                # Store QC data
                for qc_type, qc_values in qc_results.items():
                    if qc_type == 'cids':  # Skip cids metadata
                        continue

                    if qc_type not in results[model_type]['qc']:
                        results[model_type]['qc'][qc_type] = []

                    if qc_type in ['contamination', 'truncation']:
                        results[model_type]['qc'][qc_type].extend(qc_values)
                    elif qc_type == 'waveforms':
                        results[model_type]['qc'][qc_type].append(qc_values)
                    else:
                        # For probe_geometry, l4_depths, wave_times - store once
                        if len(results[model_type]['qc'][qc_type]) == 0:
                            results[model_type]['qc'][qc_type] = qc_values

    

    # Store overall cell metadata
    results[model_type]['qc']['all_cids'] = all_cids
    results[model_type]['qc']['all_datasets'] = all_datasets
    results[model_type]['qc']['all_dataset_indices'] = all_dataset_indices

    print(f"\n✅ Evaluation complete for model: {model_type} (checkpoint: {model_name})")
    print(f"   Total cells processed: {len(all_cids)}")
    print(f"   Analyses completed: {analyses}")

    return results


def run_bps_analysis(model, train_data, val_data, dataset_idx, model_name=None, save_dir=None, recalc=False, batch_size=64, rescale=False):
    """
    Run BPS analysis for a single dataset, using existing cache if available.

    Parameters
    ----------
    model : MultiDatasetModel
        The trained model
    train_data, val_data : CombinedEmbeddedDataset
        Training and validation datasets
    dataset_idx : int
        Index of the dataset
    model_name : str, optional
        Name of the model for caching (default: None, no caching)
    save_dir : Path, optional
        Directory to save caches (default: None, no caching)
    recalc : bool, optional
        Whether to recalculate (default: False)
    batch_size : int, optional
        Batch size for evaluation (default: 64)
    rescale : bool, optional
        Whether to apply affine rescaling (affects cache naming) (default: False)

    Returns
    -------
    dict
        BPS results for all stimulus types
    """
    # Only use caching if both model_name and save_dir are provided
    use_cache = (model_name is not None) and (save_dir is not None)

    if use_cache:
        rescale_suffix = '_rescaled' if rescale else ''
        cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_bps{rescale_suffix}_cache.pt'

        # Try to load from cache
        if not recalc and cache_file.exists():
            print(f'Loading BPS cache from {cache_file}')
            return torch.load(cache_file, weights_only=False)



    # Calculate from scratch
    print(f'Not using BPS cache. Evaluating from scratch...')

    # Check which stimulus types are available in this dataset
    stim_types_in_dataset = [d.metadata['name'] for d in train_data.dsets]
    n_cids = len(train_data.dsets[0].metadata['cids'])
    print(f"Available stimulus types: {stim_types_in_dataset}")
    print(f"Number of units: {n_cids}")

    # Expected stimulus types
    expected_stim_types = ['gaborium', 'backimage', 'fixrsvp', 'gratings']

    # Initialize results dictionary
    stim_results = {}

    for stim_type in expected_stim_types:
        if stim_type in stim_types_in_dataset:
            # Run actual analysis
            print(f"Running BPS analysis for {stim_type}")
            if stim_type == 'gaborium':
                stim_inds = get_stim_inds(stim_type, train_data, val_data)
                result = evaluate_dataset(
                    model, train_data, stim_inds, dataset_idx, batch_size, stim_type.capitalize()
                )
            else:
                stim_inds = get_stim_inds(stim_type, train_data, val_data)
                result = evaluate_dataset(
                    model, val_data, stim_inds, dataset_idx, batch_size, stim_type.capitalize()
                )

            # Apply rescaling if requested
            if rescale:
                from .eval_stack_utils import rescale_rhat, bits_per_spike
                rhat_rescaled, _ = rescale_rhat(result['robs'], result['rhat'], result['dfs'], mode='affine')
                bps_rescaled = bits_per_spike(rhat_rescaled, result['robs'], result['dfs'])
                result['rhat'] = rhat_rescaled
                result['bps'] = bps_rescaled

            stim_results[stim_type] = result
        else:
            # Create NaN placeholders
            print(f"Creating NaN placeholders for missing {stim_type}")
            # Use a reference dataset to get the right shapes
            ref_stim_type = stim_types_in_dataset[0]
            ref_inds = get_stim_inds(ref_stim_type, train_data, val_data)
            n_samples = len(ref_inds)

            # Create NaN arrays with correct shapes
            nan_robs = torch.full((n_samples, n_cids), float('nan'), dtype=torch.float32)
            nan_rhat = torch.full((n_samples, n_cids), float('nan'), dtype=torch.float32)
            nan_dfs = torch.full((n_samples, n_cids), float('nan'), dtype=torch.float32)
            nan_bps = torch.full((n_cids,), float('nan'), dtype=torch.float32)

            stim_results[stim_type] = {'robs': nan_robs, 'rhat': nan_rhat, 'dfs': nan_dfs, 'bps': nan_bps}

    # Validation set BPS
    val_bps_aggregator = PoissonBPSAggregator()
    val_bps_aggregator({'robs': stim_results['gaborium']['robs'], 'rhat': stim_results['gaborium']['rhat']})
    val_bps_aggregator({'robs': stim_results['backimage']['robs'], 'rhat': stim_results['backimage']['rhat']})
    val_bps = val_bps_aggregator.closure().cpu().numpy()
    val_bps_aggregator.reset()

    # Save evaluation results to cache
    bps_results = {
        'gaborium': stim_results['gaborium'],
        'backimage': stim_results['backimage'],
        'fixrsvp': stim_results['fixrsvp'],
        'gratings': stim_results['gratings'],
        'val': val_bps,
        'cids': train_data.dsets[0].metadata['cids']
    }

    # Only save if caching is enabled
    if use_cache:
        torch.save(bps_results, cache_file)
        print(f'BPS cache saved to {cache_file}')

    return bps_results


def run_ccnorm_analysis(model, train_data, val_data, dataset_idx, bps_results=None, model_name=None, save_dir=None, recalc=False, rescale=False):
    """
    Run CCNORM analysis for FixRSVP stimuli.

    Parameters
    ----------
    model : MultiDatasetModel
        The trained model
    train_data, val_data : CombinedEmbeddedDataset
        Training and validation datasets
    dataset_idx : int
        Index of the dataset
    bps_results : dict or None, optional
        BPS results if available, otherwise will load/calculate (default: None)
    model_name : str, optional
        Name of the model for caching (default: None, no caching)
    save_dir : Path, optional
        Directory to save caches (default: None, no caching)
    recalc : bool, optional
        Whether to recalculate (default: False)
    rescale : bool, optional
        Whether to use rescaled BPS results (affects cache naming) (default: False)

    Returns
    -------
    dict or None
        CCNORM results or None if failed
    """
    # Only use caching if both model_name and save_dir are provided
    use_cache = (model_name is not None) and (save_dir is not None)

    if use_cache:
        rescale_suffix = '_rescaled' if rescale else ''
        cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_ccnorm{rescale_suffix}_cache.pt'

        if not recalc and cache_file.exists():
            print(f'Loading CCNORM cache from {cache_file}')
            return torch.load(cache_file, weights_only=False)

    try:
        print(f'Calculating CCNORM for dataset {dataset_idx}...')

        # Check if fixrsvp is available in this dataset
        stim_types_in_dataset = [d.metadata['name'] for d in train_data.dsets]
        n_cids = len(train_data.dsets[0].metadata['cids'])

        if 'fixrsvp' not in stim_types_in_dataset:
            print(f"FixRSVP not available for dataset {dataset_idx}, creating NaN placeholders")

            # Create NaN placeholders with appropriate shapes
            # Based on the data structure analysis: ccnorm is (n_cids,), rbar/rbarhat are (n_time_bins, n_cids)
            # Use typical time bins (382 based on the analysis)
            n_time_bins = 382  # Typical for fixrsvp trials

            ccnorm_results = {
                'ccnorm': np.full(n_cids, np.nan),
                'rbar': np.full((n_time_bins, n_cids), np.nan),
                'rbarhat': np.full((n_time_bins, n_cids), np.nan),
                'robs_trial': np.full((1, n_time_bins, n_cids), np.nan),  # Minimal trial structure
                'rhat_trial': np.full((1, n_time_bins, n_cids), np.nan),
                'dfs_trial': np.full((1, n_time_bins, n_cids), np.nan),
                'cids': train_data.dsets[0].metadata['cids']
            }

            # Save to cache only if caching is enabled
            if use_cache:
                torch.save(ccnorm_results, cache_file)
                print(f'CCNORM NaN placeholders saved to {cache_file}')

            return ccnorm_results

        # Get BPS results if not provided
        if bps_results is None:
            bps_results = run_bps_analysis(model, train_data, val_data, dataset_idx, model_name=model_name, save_dir=save_dir, recalc=False, batch_size=64, rescale=rescale)

        # Get trial-aligned FixRSVP data
        robs_trial, rhat_trial, dfs_trial = get_fixrsvp_trials(
            model, bps_results, dataset_idx, train_data, val_data
        )

        # Calculate mean responses
        
        
        # display warning that HARDOCDED time bin size
        dt = 1/240  # 240 Hz sampling 
        print(f"WARNING: HARDCODED TIME BIN SIZE OF {dt} SECONDS")
        rbar = np.nansum(robs_trial*dfs_trial, axis=0) / np.nansum(dfs_trial, axis=0)/dt
        rbarhat = np.nansum(rhat_trial*dfs_trial, axis=0) / np.nansum(dfs_trial, axis=0)/dt

        # Calculate CCNORM
        from eval.eval_stack_utils import ccnorm_split_half_variable_trials
        ccn = ccnorm_split_half_variable_trials(robs_trial, rhat_trial, dfs_trial, return_components=False, n_splits=500)
        ccn = np.minimum(np.maximum(ccn, 0), 1)  # Clip to [0, 1]

        ccnorm_results = {
            'ccnorm': ccn,
            'rbar': rbar,
            'rbarhat': rbarhat,
            'robs_trial': robs_trial,
            'rhat_trial': rhat_trial,
            'dfs_trial': dfs_trial,
            'cids': train_data.dsets[0].metadata['cids']
        }

        # Save to cache only if caching is enabled
        if use_cache:
            torch.save(ccnorm_results, cache_file)
            print(f'CCNORM cache saved to {cache_file}')

        return ccnorm_results

    except Exception as e:
        print(f"❌ CCNORM analysis failed for dataset {dataset_idx}: {e}")
        return None


def run_saccade_analysis(model, train_data, val_data, dataset_idx, bps_results=None, model_name=None, save_dir=None, recalc=False, rescale=False, sac_win=(-10, 100)):
    """
    Run saccade-triggered analysis for all stimulus types.

    Parameters
    ----------
    model : MultiDatasetModel
        The trained model
    train_data, val_data : CombinedEmbeddedDataset
        Training and validation datasets
    dataset_idx : int
        Index of the dataset
    bps_results : dict or None, optional
        BPS results if available, otherwise will load/calculate (default: None)
    model_name : str, optional
        Name of the model for caching (default: None, no caching)
    save_dir : Path, optional
        Directory to save caches (default: None, no caching)
    recalc : bool, optional
        Whether to recalculate (default: False)
    rescale : bool, optional
        Whether to use rescaled BPS results (affects cache naming) (default: False)
    sac_win : tuple, optional
        Saccade window in milliseconds (default: (-10, 100))

    Returns
    -------
    dict or None
        Saccade analysis results or None if failed
    """
    # Only use caching if both model_name and save_dir are provided
    use_cache = (model_name is not None) and (save_dir is not None)

    if use_cache:
        rescale_suffix = '_rescaled' if rescale else ''
        cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_saccade{rescale_suffix}_cache.pt'

        if not recalc and cache_file.exists():
            print(f'Loading saccade cache from {cache_file}')
            return torch.load(cache_file, weights_only=False)

    try:
        print(f'Calculating saccade analysis for dataset {dataset_idx}...')

        # Get BPS results if not provided
        if bps_results is None:
            bps_results = run_bps_analysis(model, train_data, val_data, dataset_idx, model_name=model_name, save_dir=save_dir, recalc=False, batch_size=64, rescale=rescale)

        # Get session and detect saccades
        dataset_name = model.names[dataset_idx]
        sess = get_session(*dataset_name.split('_'))
        saccades = detect_saccades_from_session(sess)
        
        if len(saccades) == 0:
            print(f"No saccades found for dataset {dataset_idx}")
            return None

        # Check which stimulus types are available in this dataset
        stim_types_in_dataset = [d.metadata['name'] for d in train_data.dsets]
        n_cids = len(train_data.dsets[0].metadata['cids'])

        # Run saccade analysis for each stimulus type
        saccade_results = {}
        stim_types = ['backimage', 'gaborium', 'gratings', 'fixrsvp']
        n_time_bins = sac_win[1] - sac_win[0]  # Time bins in saccade window

        for stim_type in stim_types:
            if stim_type in stim_types_in_dataset:
                try:
                    sac_eval = get_saccade_eval(stim_type, train_data, val_data, bps_results, saccades, win=sac_win)
                    saccade_results[stim_type] = sac_eval
                except Exception as e:
                    print(f"Warning: Saccade analysis failed for {stim_type}: {e}")
                    # Create NaN placeholders for failed analysis
                    saccade_results[stim_type] = {
                        'robs': np.full((0, n_time_bins, n_cids), np.nan),
                        'rhat': np.full((0, n_time_bins, n_cids), np.nan),
                        'dfs': np.full((0, n_time_bins, n_cids), np.nan),
                        'rbar': np.full((n_time_bins, n_cids), np.nan),
                        'rbarhat': np.full((n_time_bins, n_cids), np.nan),
                        'eyevel': np.full((0, n_time_bins, 2), np.nan),
                        'saccade_info': [],
                        'win': sac_win
                    }
            else:
                print(f"Creating NaN placeholders for missing stimulus type: {stim_type}")
                # Create NaN placeholders for missing stimulus type
                saccade_results[stim_type] = {
                    'robs': np.full((0, n_time_bins, n_cids), np.nan),
                    'rhat': np.full((0, n_time_bins, n_cids), np.nan),
                    'dfs': np.full((0, n_time_bins, n_cids), np.nan),
                    'rbar': np.full((n_time_bins, n_cids), np.nan),
                    'rbarhat': np.full((n_time_bins, n_cids), np.nan),
                    'eyevel': np.full((0, n_time_bins, 2), np.nan),
                    'saccade_info': [],
                    'win': sac_win
                }

        if not saccade_results:
            print(f"All saccade analyses failed for dataset {dataset_idx}")
            return None

        # Add CIDs to results
        saccade_results['cids'] = train_data.dsets[0].metadata['cids']

        # Save to cache only if caching is enabled
        if use_cache:
            torch.save(saccade_results, cache_file)
            print(f'Saccade cache saved to {cache_file}')

        return saccade_results

    except Exception as e:
        print(f"❌ Saccade analysis failed for dataset {dataset_idx}: {e}")
        return None


def run_sta_analysis(model, train_data, val_data, dataset_idx, bps_results=None, model_name=None, save_dir=None, recalc=False, rescale=False, lags=list(range(16))):
    """
    Run STA (Spike-Triggered Average) and STE (Spike-Triggered Ensemble) analysis.

    Parameters
    ----------
    model : MultiDatasetModel
        The trained model
    train_data, val_data : CombinedEmbeddedDataset
        Training and validation datasets
    dataset_idx : int
        Index of the dataset to analyze
    bps_results : dict or None, optional
        BPS analysis results containing robs and rhat (default: None)
    model_name : str, optional
        Name of the model for caching (default: None, no caching)
    save_dir : Path, optional
        Directory to save cache files (default: None, no caching)
    recalc : bool, optional
        Whether to recalculate even if cache exists (default: False)
    rescale : bool, optional
        Whether to use rescaled BPS results (affects cache naming) (default: False)
    lags : list, optional
        List of lag values to compute STA for (default: list(range(16)))

    Returns
    -------
    dict
        STA analysis results with keys: sta_robs, ste_robs, sta_rhat, ste_rhat, norm_dfs, norm_robs, norm_rhat
    """

    from eval.gaborium_analysis import get_sta_ste

    # Only use caching if both model_name and save_dir are provided
    use_cache = (model_name is not None) and (save_dir is not None)

    if use_cache:
        rescale_suffix = '_rescaled' if rescale else ''
        cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_sta{rescale_suffix}_cache.pt'

        if not recalc and cache_file.exists():
            print(f'Loading STA cache from {cache_file}')
            return torch.load(cache_file, weights_only=False)

    try:
        print(f'Calculating STA analysis for dataset {dataset_idx}...')

        # Recompute BPS results on train/test combined
        gaborium_inds = torch.concatenate([
            train_data.get_dataset_inds('gaborium'),
            val_data.get_dataset_inds('gaborium')
        ], dim=0)

        dataset = train_data.shallow_copy()
        # set indices to be the gaborium inds
        dataset.inds = gaborium_inds

        gaborium_eval = evaluate_dataset(
            model, dataset, gaborium_inds, dataset_idx, 64, "Gaborium"
        )

        dataset_config = model.model.dataset_configs[dataset_idx].copy()

        sta_results = get_sta_ste(dataset_config, gaborium_eval['robs'], gaborium_eval['rhat'], lags=lags, fixations_only=True, combine_train_test=True, whiten=True, device=model.device)
        
        # Save to cache only if caching is enabled
        if use_cache:
            torch.save(sta_results, cache_file)
            print(f'STA cache saved to {cache_file}')

        return sta_results

    except Exception as e:
        print(f"❌ STA analysis failed for dataset {dataset_idx}: {e}")
        return None


def run_qc_analysis(dataset_name, dataset_cids, dataset_idx, model_name=None, save_dir=None, recalc=False):
    """
    Run QC analysis for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'Allen_2022-02-16')
    dataset_cids : list
        List of cell IDs for this dataset
    dataset_idx : int
        Index of the dataset
    model_name : str, optional
        Name of the model for caching (default: None, no caching)
    save_dir : Path, optional
        Directory to save caches (default: None, no caching)
    recalc : bool, optional
        Whether to recalculate (default: False)

    Returns
    -------
    dict or None
        QC results or None if failed
    """
    # Only use caching if both model_name and save_dir are provided
    use_cache = (model_name is not None) and (save_dir is not None)

    if use_cache:
        cache_file = save_dir / f'{model_name}_dataset{dataset_idx}_qc_cache.pt'

        if not recalc and cache_file.exists():
            print(f'Loading QC cache from {cache_file}')
            return torch.load(cache_file, weights_only=False)

    try:
        print(f'Loading QC data for {dataset_name}...')

        # Load QC data using existing utility
        sess = get_session(*dataset_name.split('_'))
        qc_data = load_qc_data(sess, dataset_cids)

        # Add cids to the cache
        qc_data['cids'] = dataset_cids

        # Save to cache only if caching is enabled
        if use_cache:
            torch.save(qc_data, cache_file)
            print(f'QC cache saved to {cache_file}')

        return qc_data

    except Exception as e:
        print(f"❌ QC loading failed for {dataset_name}: {e}")
        # Return NaN-filled results
        n_units = len(dataset_cids)
        qc_data = {
            'contamination': [np.nan] * n_units,
            'truncation': [np.nan] * n_units,
            'waveforms': np.full((n_units, 82, 384), np.nan),
            'wave_times': np.arange(82),
            'probe_geometry': None,
            'l4_depths': None,
            'cids': dataset_cids
        }

        # Save failed result to cache to avoid repeated failures (only if caching is enabled)
        if use_cache:
            torch.save(qc_data, cache_file)
            print(f'QC cache (with NaNs) saved to {cache_file}')

        return qc_data


def eval_stack_single_dataset(model, dataset_idx, analyses=['bps'], batch_size=64, rescale=False):
    """
    Evaluate a single dataset during training (no caching, no model loading).

    This function is designed to be called during training to evaluate model performance
    on a specific dataset. It loads the dataset, runs the specified analyses, and returns
    the results without saving to cache.

    Parameters
    ----------
    model : MultiDatasetModel
        The already-loaded trained model
    dataset_idx : int
        Index of the dataset to evaluate
    analyses : list, optional
        List of analyses to run: ['bps', 'ccnorm', 'saccade', 'sta', 'qc'] (default: ['bps'])
    batch_size : int, optional
        Batch size for evaluation (default: 64)
    rescale : bool, optional
        Whether to apply affine rescaling to rhat after BPS analysis (default: False)

    Returns
    -------
    dict
        Evaluation results with structure:
        {
            'bps': {...},
            'ccnorm': {...},
            'saccade': {...},
            'sta': {...},
            'qc': {...}
        }

    Example
    -------
    >>> # During training
    >>> results = eval_stack_single_dataset(model, dataset_idx=0, analyses=['bps', 'ccnorm'])
    >>> bps_val = results['bps']['val']  # Validation BPS
    >>> ccnorm = results['ccnorm']['ccnorm']  # CCNORM values
    """
    dataset_name = model.names[dataset_idx]
    print(f"\n{'='*60}")
    print(f"Evaluating dataset {dataset_idx}: {dataset_name}")
    print(f"Analyses: {analyses}")
    print(f"{'='*60}")

    # Load dataset
    train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
    dataset_cids = dataset_config.get('cids', [])

    # Initialize results structure
    results = {
        'dataset_name': dataset_name,
        'dataset_idx': dataset_idx,
        'cids': dataset_cids
    }

    # Run BPS analysis if requested
    bps_results = None
    if 'bps' in analyses:
        print("Running BPS analysis...")
        bps_results = run_bps_analysis(
            model, train_data, val_data, dataset_idx,
            batch_size=batch_size, rescale=rescale
        )
        results['bps'] = bps_results

    # Run CCNORM analysis if requested
    if 'ccnorm' in analyses:
        print("Running CCNORM analysis...")
        ccnorm_results = run_ccnorm_analysis(
            model, train_data, val_data, dataset_idx,
            bps_results=bps_results, rescale=rescale
        )
        results['ccnorm'] = ccnorm_results

    # Run saccade analysis if requested
    if 'saccade' in analyses:
        print("Running Saccade analysis...")
        saccade_results = run_saccade_analysis(
            model, train_data, val_data, dataset_idx,
            bps_results=bps_results, rescale=rescale
        )
        results['saccade'] = saccade_results

    # Run STA analysis if requested
    if 'sta' in analyses:
        print("Running STA analysis...")
        sta_results = run_sta_analysis(
            model, train_data, val_data, dataset_idx,
            bps_results=bps_results, rescale=rescale
        )
        results['sta'] = sta_results

    # Run QC analysis if requested
    if 'qc' in analyses:
        print("Running QC analysis...")
        qc_results = run_qc_analysis(
            dataset_name, dataset_cids, dataset_idx
        )
        results['qc'] = qc_results

    print(f"\n✅ Evaluation complete for dataset {dataset_idx}: {dataset_name}")
    print(f"   Total cells: {len(dataset_cids)}")
    print(f"   Analyses completed: {analyses}")

    return results


# Example usage
if __name__ == "__main__":
    # Example: Evaluate the best learned_res model
    results = evaluate_model_multidataset(
        model_type='learned_res',
        analyses=['bps', 'ccnorm', 'saccade'],
        recalc=False,
        batch_size=64
    )

    # Print summary
    for model_name, model_results in results.items():
        print(f"\nResults for {model_name}:")
        print(f"  Total cells: {len(model_results['qc']['all_cids'])}")

        if 'bps' in model_results:
            for stim_type in model_results['bps']:
                n_cells = len(model_results['bps'][stim_type]['cids'])
                print(f"  BPS {stim_type}: {n_cells} cells")

        if 'ccnorm' in model_results and 'fixrsvp' in model_results['ccnorm']:
            n_cells = len(model_results['ccnorm']['fixrsvp']['cids'])
            print(f"  CCNORM fixrsvp: {n_cells} cells")

        if 'saccade' in model_results:
            for stim_type in model_results['saccade']:
                n_cells = len(model_results['saccade'][stim_type]['cids'])
                print(f"  Saccade {stim_type}: {n_cells} cells")
