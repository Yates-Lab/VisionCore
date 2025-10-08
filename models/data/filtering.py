"""
Data validation utilities for neural data analysis.

This module provides functions for validating and filtering neural data,
including finding valid blocks of data and generating masks for valid frames.
"""

import torch
import numpy as np

def find_valid_blocks(valid, min_block_len):
    """
    Find contiguous blocks where the valid mask is True,
    with a minimum block length.
    
    Parameters:
    -----------
    valid : torch.Tensor
        Boolean tensor of length N
    min_block_len : int
        Minimum length of a valid block
        
    Returns:
    --------
    torch.Tensor
        Tensor of shape (M, 2) containing start and end indices of valid blocks
    """
    # Find transitions (0->1 and 1->0)
    # Add boundary conditions for blocks that start at 0 or end at len-1
    transitions = torch.diff(torch.cat([torch.tensor([0]), valid.float(), torch.tensor([0])]))
    
    # Get indices where transitions occur
    block_start = np.where(transitions==1)[0]
    block_end = np.where(transitions==-1)[0]
    assert len(block_start) == len(block_end), 'Number of block starts and ends must match'

    # Process transitions to get start and end points
    blocks = []
    for i in range(0, len(block_start) - 1, 2):
        start_idx = block_start[i]
        end_idx = block_end[i]
        
        assert all(valid[start_idx:end_idx]), 'Invalid block found'
        if end_idx - start_idx >= min_block_len:
            blocks.append([start_idx, end_idx])
    
    # Convert to tensor if blocks were found
    if blocks:
        return torch.tensor(blocks)
    else:
        return torch.zeros((0, 2), dtype=torch.int64)  # Empty tensor with proper shape

def get_valid_dfs(dset, n_lags):
    """
    Generate a binary mask for valid data frames based on trial boundaries and DPI validity.
    
    This function creates a mask that identifies valid frames for analysis by:
    1. Identifying trial boundaries
    2. Excluding the first frame of each trial
    3. Ensuring DPI (eye tracking) data is valid
    4. Ensuring temporal continuity for the specified number of lags
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset containing trial indices and DPI validity information
    n_lags : int
        Number of time lags to ensure continuity for

    Returns:
    --------
    dfs : torch.Tensor
        Binary mask tensor of shape [n_frames, 1] where 1 indicates valid frames
    """
    dpi_valid = dset['dpi_valid']
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
    dfs = ~new_trials
    dfs &= (dpi_valid > 0)

    for _ in range(n_lags-1):
        dfs &= torch.roll(dfs, 1)
    
    dfs = dfs.float()
    dfs = dfs[:, None]
    return dfs
