"""
Data loading utilities for neural data analysis.

This module provides functions for loading, preprocessing, and organizing neural datasets
for model training and analysis. It handles dataset loading, time embedding, train/validation
splitting, and calculation of spike-triggered statistics.

The module supports:
- Loading multiple dataset types and combining them
- Creating reproducible train/validation splits based on trials
- Calculating and caching spike-triggered averages and second moments
- Preprocessing datasets with custom functions
"""
import torch
import torch.nn.functional as F
import numpy as np
from .datasets import DictDataset, CombinedEmbeddedDataset
from .filtering import get_valid_dfs
from .splitting import split_inds_by_trial
from .transforms import make_pipeline
from .datafilters import make_datafilter_pipeline
from ..utils.general import ensure_tensor
from DataYatesV1.utils.rf import calc_sta
from typing import Dict, Any, List, Tuple
import yaml
import copy

def get_embedded_datasets(sess, types=None, keys_lags=None, train_val_split=None, cids=None, seed=1002, pre_func=None, **kwargs):
    """
    Create train and validation datasets from multiple dataset types with time embedding.

    This function loads multiple datasets, applies preprocessing, filters valid frames,
    splits them into training and validation sets based on trials, and combines them
    into embedded datasets ready for model training.

    Parameters
    ----------
    sess : Session object
        Session object containing path information
    types : list of str or dict datasets
        List of dataset types to load (e.g., ['gaborium', 'backimage'])
        or a list of DictDataset objects
    keys_lags : dict
        Dictionary mapping dataset keys to lag values for time embedding
        Example: {'robs': 0, 'stim': np.arange(10)}
    train_val_split : float
        Fraction of data to use for training (between 0 and 1)
    cids : array-like, optional
        Cell IDs to include. If None, all cells are included
    seed : int, optional
        Random seed for reproducible train/validation splits, default=1002
    pre_func : callable, optional
        Function to apply to each dataset after loading

    Returns
    -------
    train_dset : CombinedEmbeddedDataset
        Combined dataset for training
    val_dset : CombinedEmbeddedDataset
        Combined dataset for validation
    """
    # Determine maximum number of lags needed based on keys_lags
    n_lags = np.max([np.max(keys_lags[k]) for k in keys_lags])

    # Default preprocessing function if none provided
    if pre_func is None:
        def pre_func(x):
            # Normalize stimulus to [-0.5, 0.5] range
            x['stim'] = (x['stim'].float() - 127) / 255
            # Generate valid frame mask based on trial boundaries and DPI validity
            x['dfs'] = get_valid_dfs(x, n_lags)
            return x

    # Load and preprocess each dataset type
    dsets = []
    for dset_type in types:
        
        dset = dset_type
        
        # Apply preprocessing
        dset = pre_func(dset)

        # Filter by cell IDs if specified
        if cids is not None:
            dset.metadata['cids'] = cids
            dset['robs'] = dset['robs'][:,cids]
            if 'dfs' in dset and dset['dfs'].ndim == 2:
                if (dset['dfs'].shape[1] > 1) and (dset['dfs'].shape[1] != len(cids)):
                    dset['dfs'] = dset['dfs'][:,cids]

        dsets.append(dset)

    # Get indices of valid frames for each dataset
    dset_inds = [dset['dfs'].any(dim=1).nonzero(as_tuple=True)[0] for dset in dsets]

    # Print dataset statistics
    for iD, dset in enumerate(dsets):
        print(f'{types[iD]} dataset size: {len(dset)} / {len(dset_inds[iD])} ({len(dset_inds[iD])/len(dset)*100:.2f}%)')

    # Split indices into training and validation sets by trial
    train_inds, val_inds = [], []
    for iD, dset in enumerate(dsets):
        train_inds_, val_inds_ = split_inds_by_trial(dset, dset_inds[iD], train_val_split, seed)
        train_inds.append(train_inds_)
        val_inds.append(val_inds_)

    # Create combined embedded datasets for training and validation
    train_dset = CombinedEmbeddedDataset(dsets, train_inds, keys_lags)
    val_dset = CombinedEmbeddedDataset(dsets, val_inds, keys_lags)

    return train_dset, val_dset

def get_gaborium_sta_ste(sess, n_lags, cids=None):
    """
    Calculate or load cached spike-triggered averages (STAs) and spike-triggered
    second moments (STEs) for gaborium stimulus data.

    This function first checks if cached STAs/STEs exist and have sufficient lags.
    If so, it loads them from cache. Otherwise, it calculates them from the raw data
    and saves them to cache for future use.

    Parameters
    ----------
    sess : Session object
        Session object containing path information
    n_lags : int
        Number of time lags to calculate STAs/STEs for
    cids : array-like, optional
        Cell IDs to include. If None, all cells are included

    Returns
    -------
    stas : numpy.ndarray
        Spike-triggered averages with shape (n_cells, n_lags, n_y, n_x)
    stes : numpy.ndarray
        Spike-triggered second moments with shape (n_cells, n_lags, n_y, n_x)
    """
    # Verify that the dataset exists
    assert (sess.sess_dir / 'shifter' / 'gaborium_shifted.dset').exists()

    # Define cache file path
    cache_dir = sess.sess_dir / 'shifter' / 'gaborium_sta_ste.npy'

    # Try to load from cache if it exists
    if cache_dir.exists():
        stas, stes = np.load(cache_dir, allow_pickle=True)
        n_lags_cached = stas.shape[1]

        # If cached data has enough lags, use it
        if n_lags_cached >= n_lags:
            if cids is None:
                cids = np.arange(stas.shape[0])

            # Return requested subset of lags and cells
            return stas[cids][:,:n_lags], stes[cids][:,:n_lags]
        else:
            print(f'Cached STAs/STEs have {n_lags_cached} lags, but {n_lags} were requested. Recalculating...')
    else:
        print('Cached STAs/STEs not found. Calculating...')

    # Load and preprocess the dataset
    dset = DictDataset.load(sess.sess_dir / 'shifter' / 'gaborium_shifted.dset')
    dset['stim'] = dset['stim'].float()
    # Normalize stimulus (mean-centered)
    dset['stim'] = (dset['stim'] - dset['stim'].mean()) / 255
    # Generate valid frame mask
    dset['dfs'] = get_valid_dfs(dset, n_lags)

    # Calculate spike-triggered averages (STAs)
    stas = calc_sta(dset['stim'].detach().cpu(),
                   dset['robs'].cpu(),
                   range(n_lags),
                   dfs=dset['dfs'].cpu().squeeze(),
                   progress=True).cpu().squeeze().numpy()

    # Calculate spike-triggered second moments (STEs)
    # Uses squared stimulus values via stim_modifier
    stes = calc_sta(dset['stim'].detach().cpu(),
                   dset['robs'].cpu(),
                   range(n_lags),
                   dfs=dset['dfs'].cpu().squeeze(),
                   stim_modifier=lambda x: x**2,
                   progress=True).cpu().squeeze().numpy()

    # Save results to cache for future use
    try:
        np.save(cache_dir, [stas, stes])
        print(f'STAs/STEs saved to cache: {cache_dir}')
    except Exception as e:
        print(f'Failed to save STAs/STEs to cache: {e}')

    # Filter by cell IDs if specified
    if cids is not None:
        stas = stas[cids]
        stes = stes[cids]

    return stas, stes


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Downsampling functions
# ──────────────────────────────────────────────────────────────────────────────

def downsample_counts(counts: torch.Tensor, factor: int) -> torch.Tensor:
    """Re-bin spike counts by summing over `factor` adjacent bins."""
    T, N = counts.shape
    T_new = T // factor
    return counts[:T_new*factor].reshape(T_new, factor, N).sum(1)


def downsample_stimulus(stim: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample stimulus by simple decimation (every nth frame)."""
    T = stim.shape[0]
    T_new = T // factor
    # Only take the first T_new*factor frames to ensure consistent length
    return stim[:T_new*factor:factor]


def downsample_continuous(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample continuous signals using average pooling (true average, no scaling)."""
    original_shape = x.shape
    T = original_shape[0]

    # Handle different tensor shapes
    if x.ndim == 1:
        # 1D tensor: (T,) -> (1, 1, T) for avg_pool1d
        x_reshaped = x.unsqueeze(0).unsqueeze(0)
        x_pooled = F.avg_pool1d(x_reshaped, kernel_size=factor, stride=factor)
        return x_pooled.squeeze(0).squeeze(0)

    elif x.ndim == 2:
        # 2D tensor: (T, C) -> (1, C, T) for avg_pool1d
        x_reshaped = x.transpose(0, 1).unsqueeze(0)  # (1, C, T)
        x_pooled = F.avg_pool1d(x_reshaped, kernel_size=factor, stride=factor)
        return x_pooled.squeeze(0).transpose(0, 1)  # Back to (T_new, C)

    else:
        # For higher-dimensional tensors: flatten all non-temporal dimensions,
        # apply pooling, then reshape back
        non_temporal_shape = original_shape[1:]
        non_temporal_size = np.prod(non_temporal_shape)

        # Reshape to (T, flattened_features)
        x_flat = x.view(T, non_temporal_size)

        # Apply 2D pooling logic
        x_reshaped = x_flat.transpose(0, 1).unsqueeze(0)  # (1, features, T)
        x_pooled = F.avg_pool1d(x_reshaped, kernel_size=factor, stride=factor)
        x_pooled_flat = x_pooled.squeeze(0).transpose(0, 1)  # (T_new, features)

        # Reshape back to original structure
        T_new = x_pooled_flat.shape[0]
        return x_pooled_flat.view(T_new, *non_temporal_shape)


def apply_downsampling(dset: DictDataset, factor: int) -> DictDataset:
    """Apply appropriate downsampling to each covariate in the dataset."""
    if factor == 1:
        return dset  # No downsampling needed

    print(f"  Downsampling by factor {factor}...")

    # Create a new dictionary to store downsampled data
    downsampled_data = {}

    # Downsample each covariate appropriately
    for key in dset.covariates.keys():
        print(f"    {key}: {dset[key].shape} -> ", end="")

        if key == 'robs':
            # Spike counts: sum adjacent bins
            downsampled_data[key] = downsample_counts(dset[key], factor)
        elif key == 'stim' or key == 'stim_phase' or key == 'ori' or key == 'sf':
            # Stimulus: simple decimation
            downsampled_data[key] = downsample_stimulus(dset[key], factor)
        elif key == 'psth_inds':
            # PSTH indices: divide by downsample factor and round (for PSTH alignment)
            downsampled_data[key] = downsample_stimulus(dset[key], factor) // factor
        else:
            # All other covariates: average pooling
            downsampled_data[key] = downsample_continuous(dset[key], factor)

        print(f"{downsampled_data[key].shape}")

    # Create a new DictDataset with downsampled data
    downsampled_dset = DictDataset(downsampled_data, metadata=dset.metadata.copy())

    return downsampled_dset


def _create_nan_placeholder_dataset(reference_dset, dataset_name, sess, cids=None):
    """
    Create a placeholder dataset filled with NaNs that matches the structure of a reference dataset.

    Parameters
    ----------
    reference_dset : DictDataset
        Reference dataset to copy structure from
    dataset_name : str
        Name of the missing dataset type
    sess : Session
        Session object
    cids : list, optional
        Cell IDs to include

    Returns
    -------
    DictDataset
        Placeholder dataset with NaN values
    """
    from copy import deepcopy

    # Create a new dataset with same structure but NaN values
    placeholder_data = {}

    # Copy basic structure from reference
    for key, value in reference_dset.covariates.items():
        if key == 'robs':
            # Neural responses: create NaN array with same shape
            if cids is not None:
                n_units = len(cids)
            else:
                n_units = value.shape[-1] if value.ndim > 1 else len(value)
            placeholder_data[key] = torch.full((len(value), n_units), float('nan'), dtype=torch.float32)
        elif key == 'stim':
            # Stimulus: create NaN array with same shape
            placeholder_data[key] = torch.full_like(value, float('nan'), dtype=value.dtype)
        else:
            # Other covariates: copy structure but fill with NaNs
            if value.dtype in [torch.float32, torch.float64]:
                placeholder_data[key] = torch.full_like(value, float('nan'))
            else:
                # For integer types, use -1 as placeholder (since NaN doesn't work)
                placeholder_data[key] = torch.full_like(value, -1)

    # Create the placeholder dataset
    placeholder_dset = DictDataset(placeholder_data)

    # Copy metadata and update
    placeholder_dset.metadata = deepcopy(reference_dset.metadata)
    placeholder_dset.metadata['name'] = dataset_name
    placeholder_dset.metadata['sess'] = sess
    placeholder_dset.metadata['is_placeholder'] = True
    if cids is not None:
        placeholder_dset.metadata['cids'] = cids

    return placeholder_dset


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Prepare_data
# ──────────────────────────────────────────────────────────────────────────────
def prepare_data(dataset_config: Dict[str, Any], strict: bool = True):
    """
    Extended prepare_data that supports `transforms:` and `datafilters:` blocks with preprocessing.

    Parameters
    ----------
    dataset_config : dict
        Parsed YAML config (already loaded via yaml.safe_load).
    strict : bool, optional
        If True (default), raises an error if any dataset type is missing.
        If False, skips missing dataset types and continues with available ones.
    Returns
    -------
    train_dset, val_dset, dataset_config  (unchanged downstream interface)
    """
    # strict = True
    print("\nPreparing data (with preprocessing)…")

    # check if dataset_config is a path
    if isinstance(dataset_config, str):
        with open(dataset_config, 'r') as f:
            dataset_config = yaml.safe_load(f)

    # -- unpack ----------------------------------------------------------------
    sess_name  = dataset_config["session"]
    dset_types = dataset_config["types"]
    transforms  = dataset_config.get("transforms", {})
    datafilters = dataset_config.get("datafilters", {})
    keys_lags  = dataset_config["keys_lags"]
    sampling_config = dataset_config.get("sampling", None)
    lab = dataset_config.get("lab", "yates")  # Default to yates for backward compatibility

    # Handle different session naming conventions
    if lab.lower() == "yates":
        from DataYatesV1.utils.io import get_session
    elif lab.lower() == "rowley":
        from DataRowleyV1V2.data.registry import get_session
    else:
        raise ValueError(f"Unknown lab: {lab}")
    
    sess = get_session(*sess_name.split("_"))

    # -------------------------------------------------------------------------
    # Calculate downsampling factor if sampling config is present
    # -------------------------------------------------------------------------
    downsample_factor = 1
    if sampling_config:
        source_rate = sampling_config["source_rate"]
        target_rate = sampling_config["target_rate"]
        downsample_factor = source_rate // target_rate
        print(f"Downsampling from {source_rate}Hz to {target_rate}Hz (factor: {downsample_factor})")

    # -------------------------------------------------------------------------
    # Build transform specs once
    # -------------------------------------------------------------------------
    transform_specs = {}
    for var_name, spec in transforms.items():
        pipeline = make_pipeline(spec.get("ops", []), dataset_config)
        transform_specs[var_name] = dict(
            source      = spec.get("source", var_name),
            pipeline    = pipeline,
            expose_as   = spec.get("expose_as", var_name),
            concatenate = spec.get("concatenate", False),  # Default to overwrite behavior
        )

        # Merge any per-variable keys_lags into the master dict
        if "keys_lags" in spec:
            keys_lags[spec["expose_as"]] = spec["keys_lags"]

    # -------------------------------------------------------------------------
    # Build datafilter specs once
    # -------------------------------------------------------------------------
    datafilter_specs = {}
    for var_name, spec in datafilters.items():
        pipeline = make_datafilter_pipeline(spec.get("ops", []))
        datafilter_specs[var_name] = dict(
            pipeline   = pipeline,
            expose_as  = spec.get("expose_as", var_name),
        )

        # Merge any per-variable keys_lags into the master dict
        if "keys_lags" in spec:
            keys_lags[spec["expose_as"]] = spec["keys_lags"]

    # Check if datafilters are specified, warn if not
    if not datafilters:
        print("WARNING: No datafilters specified in config. This may lead to invalid samples being included in training.")

    # -------------------------------------------------------------------------
    # Load each DictDataset, run transforms and datafilters in-place, and stash
    # -------------------------------------------------------------------------
    n_lags = dataset_config.get("n_lags", np.max([np.max(keys_lags[k]) for k in keys_lags]))

    preprocessed_dsets = []

    for dt in dset_types.copy():
        try:
            dset = sess.get_dataset(dt, config=dataset_config)

            # -------------------------------------------------------------
            # Apply downsampling if specified
            # -------------------------------------------------------------
            if downsample_factor > 1:
                print(f"Applying downsampling to {dt} dataset:")
                dset = apply_downsampling(dset, downsample_factor)

            # -------------------------------------------------------------
            # Apply datafilter pipelines
            # -------------------------------------------------------------
            if datafilter_specs:
                for var_name, spec in datafilter_specs.items():
                    expose_as = spec["expose_as"]
                    # print(f"Applying datafilter → {expose_as}")
                    mask_tensor = spec["pipeline"](dset)
                    dset[expose_as] = mask_tensor
            else:
                # Fallback to old behavior if no datafilters specified
                dset['dfs'] = get_valid_dfs(dset, n_lags)

            # -------------------------------------------------------------
            # Apply transform pipelines
            # -------------------------------------------------------------
            # Collect transformed variables by expose_as name for potential concatenation
            transformed_vars = {}
            concatenate_vars = {}  # Track which variables should be concatenated

            for var_name, spec in transform_specs.items():
                
                src_key     = spec["source"]
                expose_as   = spec["expose_as"]
                concatenate = spec["concatenate"]
                # print(f"Transforming {src_key} → {expose_as}")
                data_tensor = ensure_tensor(dset[src_key], dtype=torch.float32)   # → torch.Tensor
                data_tensor = spec["pipeline"](data_tensor)
                # print(f"{expose_as} shape: {data_tensor.shape}")

                if concatenate:
                    # Collect variables marked for concatenation
                    if expose_as not in transformed_vars:
                        transformed_vars[expose_as] = []
                        concatenate_vars[expose_as] = True
                    transformed_vars[expose_as].append(data_tensor)
                else:
                    # Overwrite behavior (default) - assign directly
                    dset[expose_as] = data_tensor

            # Concatenate variables that were marked for concatenation
            for expose_as, var_list in transformed_vars.items():
                if concatenate_vars.get(expose_as, False):
                    if len(var_list) == 1:
                        # Single variable, no concatenation needed
                        dset[expose_as] = var_list[0]
                    else:
                        # Multiple variables, concatenate along last dimension

                        concatenated = torch.cat(var_list, dim=-1)
                        dset[expose_as] = concatenated
                        # print(f"Concatenated {len(var_list)} variables for {expose_as}, final shape: {concatenated.shape}")

            print(f"stim shape: {dset.covariates['stim'].shape}")
            preprocessed_dsets.append(dset)

        except Exception as e:
            if strict:
                # Re-raise the exception if strict mode is enabled
                raise e
            else:
                # remove this dt from the dataset_config
                print(f"WARNING: Skipping missing dataset '{dt}' due to error: {e}")
                dataset_config['types'].remove(dt)
                continue

    # -------------------------------------------------------------------------
    # Combine datasets
    # -------------------------------------------------------------------------
    train_dset, val_dset = get_embedded_datasets(
        sess,
        types            = preprocessed_dsets,           # pass in the preprocessed datasets
        keys_lags        = keys_lags,
        train_val_split  = dataset_config["train_val_split"],
        cids             = dataset_config.get("cids", None),
        seed             = dataset_config.get("seed", 1002),
        pre_func         = lambda x: x,          # preprocessing already done
    )

    print(f"Train size: {len(train_dset)} samples | "
          f"Val size: {len(val_dset)} samples")

    # IMPORTANT: pass behaviour feature dim back to model yaml --------------
    beh_keys = [v["expose_as"] for v in transform_specs.values()
                if v["expose_as"] == "behavior"]
    if beh_keys:
        # assume they were concatenated along last dim already
        sample = train_dset[0]["behavior"]
        dataset_config["behavior_dim"] = sample.shape[-1]

    return train_dset, val_dset, dataset_config





def remove_pixel_norm(dataset_config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Remove pixelnorm operation from transforms with source "stim" if present.

    Parameters
    ----------
    dataset_config : dict
        Dataset configuration dictionary

    Returns
    -------
    new_config : dict
        Modified dataset configuration with pixelnorm removed
    removed : bool
        True if pixelnorm was found and removed, False otherwise
    """
    # Create a deep copy to avoid modifying the original config
    new_config = copy.deepcopy(dataset_config)
    removed = False

    # Check if transforms exist in the config
    if 'transforms' not in new_config:
        return new_config, removed

    transforms = new_config['transforms']

    # Look for transforms with source "stim"
    for transform_key, transform_spec in transforms.items():
        if transform_spec.get('source') == 'stim':
            # Check if ops exist
            if 'ops' in transform_spec:
                ops = transform_spec['ops']
                new_ops = []

                # Filter out pixelnorm operations
                for op in ops:
                    if isinstance(op, dict) and 'pixelnorm' in op:
                        removed = True
                        # Skip this operation (don't add to new_ops)
                        continue
                    else:
                        new_ops.append(op)

                # Update the ops list
                transform_spec['ops'] = new_ops

    return new_config, removed
