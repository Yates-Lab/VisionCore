"""
Data filtering utilities for neural data analysis.

This module provides a registry-based system for data filters that can be
applied to neural datasets. Filters output boolean masks and can be chained
together in pipelines where they are combined with AND operations.

The module supports:
- Registry-based filter functions
- Pipeline composition of multiple filters with AND combination
- Configurable filters with parameters
- Broadcasting support for combining different shaped masks
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Any
from .datasets import DictDataset

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def mask_valid_timestamps(windows, valid, timestamps):
    """
    windows    : (N, 2) [start, stop) per row
    valid      : (N,)   bool  mask marking usable windows
    timestamps : (T, 1) or (T,) array_like

    Returns
    --------
    mask : (T, 1) boolean ndarray
           True where the timestamp lies inside any valid window.
    """
    windows   = np.asarray(windows)
    valid     = np.asarray(valid, dtype=bool)

    # Keep only valid windows
    win = windows[valid]         # shape (M, 2), no overlaps by contract
    if win.size == 0:
        return np.zeros_like(timestamps, dtype=bool)

    # Sort once by start
    order       = np.argsort(win[:, 0])
    starts      = win[order, 0]  # (M,)
    ends        = win[order, 1]  # (M,)

    # Flatten timestamps for vectorised search
    t_shape = timestamps.shape          # keep for reshape at the end
    ts      = np.asarray(timestamps).ravel()

    # Binary-search: position of rightmost end > ts
    idx  = np.searchsorted(ends, ts, side="right") - 1  # (T,)
    mask = (idx >= 0) & (ts >= starts[idx])

    return mask.reshape(t_shape)


# ──────────────────────────────────────────────────────────────────────────────
# Datafilter registry
# ──────────────────────────────────────────────────────────────────────────────
class DataFilterFn(Callable[[DictDataset], torch.Tensor]): ...
DATAFILTER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], DataFilterFn]] = {}

def _register(name):
    def wrap(fn):
        DATAFILTER_REGISTRY[name] = fn
        return fn
    return wrap

@_register("valid_nlags")
def _make_valid_nlags(cfg):
    """
    Create a datafilter that validates frames based on trial boundaries and temporal continuity.
    
    This is equivalent to the original get_valid_dfs function.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary with 'n_lags' parameter
        
    Returns
    -------
    DataFilterFn
        Function that takes a dataset and returns boolean mask
    """
    n_lags = cfg if isinstance(cfg, int) else cfg.get("n_lags", 1)
    
    def valid_nlags(dset: DictDataset) -> torch.Tensor:
        """
        Generate a binary mask for valid data frames based on trial boundaries and DPI validity.
        
        This function creates a mask that identifies valid frames for analysis by:
        1. Identifying trial boundaries
        2. Excluding the first frame of each trial
        3. Ensuring DPI (eye tracking) data is valid
        4. Ensuring temporal continuity for the specified number of lags
        
        Parameters
        ----------
        dset : DictDataset
            Dataset containing trial indices and DPI validity information
            
        Returns
        -------
        torch.Tensor
            Binary mask tensor of shape [n_frames, 1] where 1 indicates valid frames
        """
        dpi_valid = dset['dpi_valid']
        new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
        dfs = ~new_trials
        dfs &= (dpi_valid > 0)

        for _ in range(n_lags-1):
            dfs &= torch.roll(dfs, 1)

        # Convert to float for compatibility with original get_valid_dfs
        dfs = dfs.float()
        dfs = dfs[:, None]
        return dfs.bool()
    
    return valid_nlags

@_register("missing_pct")
def _make_missing_pct(cfg):
    """
    Create a datafilter that excludes frames based on missing percentage.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary with 'threshold' and 'cids' parameters
        
    Returns
    -------
    DataFilterFn
        Function that takes a dataset and returns boolean mask
    """
    if isinstance(cfg, (int, float)):
        threshold = float(cfg)
        source = None
        allow_high_median_units = True
    else:
        # ``theshold`` appears in several historical configs.  Continue to
        # accept it, but prefer the correctly spelled key for new configs.
        threshold = float(cfg.get("threshold", cfg.get("theshold", 45)))
        source = cfg.get("source")
        allow_high_median_units = bool(
            cfg.get("allow_high_median_units", True)
        )
    
    def missing_pct(dset: DictDataset) -> torch.Tensor:
        """
        Generate a binary mask for valid data frames based on missing percentage.
        
        This function creates a mask that excludes frames where the missing
        percentage exceeds the specified threshold.
        
        Parameters
        ----------
        dset : DictDataset
            Dataset containing missing percentage information
            
        Returns
        -------
        torch.Tensor
            Binary mask tensor of shape [n_frames, n_units] where 1 indicates valid frames
        """
        if source is not None:
            if source not in dset:
                raise KeyError(
                    f"missing_pct datafilter requires exported covariate {source!r}"
                )
            missing_pct = torch.as_tensor(dset[source], dtype=torch.float32)
        else:
            # Backward-compatible path for older artifacts that retain their
            # live Session object in metadata.  New reproducible artifacts
            # should export missing_pct and name it with ``source``.
            sess = dset.metadata.get("sess")
            if sess is None:
                raise KeyError(
                    "missing_pct is not exported and dataset metadata has no 'sess'"
                )
            cids = dset.metadata.get("cluster_ids")
            if cids is None:
                cids = sess.get_cluster_ids()
            times = dset["t_bins"]
            missing_pct = torch.as_tensor(
                sess.get_missing_pct_interp(cids)(times), dtype=torch.float32
            )

        if missing_pct.ndim != 2:
            raise ValueError(
                "missing_pct must have shape [n_samples, n_units], got "
                f"{tuple(missing_pct.shape)}"
            )
        if missing_pct.shape[0] != len(dset):
            raise ValueError(
                "missing_pct sample dimension does not match dataset: "
                f"{missing_pct.shape[0]} != {len(dset)}"
            )
        if "robs" in dset and missing_pct.shape[1] != dset["robs"].shape[1]:
            raise ValueError(
                "missing_pct unit dimension does not match robs: "
                f"{missing_pct.shape[1]} != {dset['robs'].shape[1]}"
            )

        mask = torch.isfinite(missing_pct) & (missing_pct < threshold)
        if allow_high_median_units:
            # Historical behavior: units whose median exceeds the threshold
            # are treated as multi-units and not time-masked.  The Rowley
            # canary disables this override so missing coverage fails closed.
            median = torch.from_numpy(
                np.nanmedian(missing_pct.cpu().numpy(), axis=0)
            ).to(mask.device)
            mask[:, median >= threshold] = True

        return mask
    
    return missing_pct

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Build a composite datafilter pipeline
# ──────────────────────────────────────────────────────────────────────────────
def make_datafilter_pipeline(op_list: List[Dict[str, Any]]) -> DataFilterFn:
    """
    Build a composite datafilter pipeline that combines multiple filters with AND operations.
    
    Parameters
    ----------
    op_list : List[Dict[str, Any]]
        List of operation dictionaries, each containing a single key-value pair
        where the key is the filter name and value is the configuration
        
    Returns
    -------
    DataFilterFn
        Function that takes a dataset and returns combined boolean mask
    """
    fns: List[DataFilterFn] = []
    for op_dict in op_list:
        name, cfg = next(iter(op_dict.items()))
        if name not in DATAFILTER_REGISTRY:
            raise ValueError(f"Unknown datafilter '{name}'")
        fns.append(DATAFILTER_REGISTRY[name](cfg))

    def pipeline(dset: DictDataset) -> torch.Tensor:
        if not fns:
            # If no filters specified, return all True mask
            n_frames = len(dset)
            return torch.ones((n_frames, 1), dtype=torch.bool)
        
        # Apply first filter
        result = fns[0](dset)
        if result.ndim == 1:
            result = result[:, None]
            
        
        # Combine subsequent filters with AND operation
        for fn in fns[1:]:
            mask = fn(dset)
            # Broadcasting handles different shapes (Tx1 with TxN)
            result = result & mask
            
        return result.float()
    
    return pipeline
