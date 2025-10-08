"""
Dataset splitting utilities for neural data analysis.

This module provides functions for splitting datasets into training and validation sets
using various strategies, including blockwise splitting, section-based splitting,
and trial-based splitting.
"""

import torch
import numpy as np
from ..torch import set_seeds
from .filtering import find_valid_blocks

def get_blockwise_train_indices(dset, min_block_len=240, frame_rate=240, frac_train=0.8, seed=None):
    '''
    Get blockwise indices for training and validation.

    Training and test sets come from different valid blocks.
    This ensures no train-test bleed through.
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset containing time bins and DPI validity information
    min_block_len : int, optional
        Minimum length of a valid block. Default is 240.
    frame_rate : int, optional
        Frame rate of the data in Hz. Default is 240.
    frac_train : float, optional
        Fraction of blocks to use for training. Default is 0.8.
    seed : int, optional
        Random seed for shuffling blocks. Default is None.
        
    Returns:
    --------
    train_idx : torch.Tensor
        Indices for training
    val_idx : torch.Tensor
        Indices for validation
    '''
    t_bins = dset['t_bins']
    dt = torch.from_numpy(np.gradient(t_bins))
    valid_time = dt < 1.5 / frame_rate # less than 2 frames @ 240 Hz (hard coded here)
    valid_dpi = dset['dpi_valid'] > 0

    blocks = find_valid_blocks(valid_time & valid_dpi, min_block_len)
    num_blocks = blocks.shape[0]

    if seed is not None:
        # set seed and shuffle block order
        torch.manual_seed(seed)
        blocks = blocks[torch.randperm(num_blocks)]

    print(f'Found {num_blocks} valid blocks')
    print(f'{torch.sum(blocks[:,1]-blocks[:,0])} valid samples')

    # splot blocks into train and validation
    train_blocks = blocks[:int(num_blocks*frac_train)]
    val_blocks = blocks[int(num_blocks*frac_train):]

    train_idx = torch.cat([torch.arange(start, end) for start, end in train_blocks])
    val_idx = torch.cat([torch.arange(start, end) for start, end in val_blocks])
    return train_idx, val_idx

def get_train_val_with_sections(dset, num_sections, val_fraction=0.2):
    '''
    Splits the dataset into training and validation sets by sections.
    
    One way to 'fairly' split the dataset is into a number of sections and use some 
    of those sections for training and the rest for validation. That way we can
    have a separation of fixations in some sense.
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset to split
    num_sections : int
        Number of sections to split the dataset into
    val_fraction : float, optional
        Fraction of the dataset to use for validation. Default is 0.2.
        
    Returns:
    --------
    train_set : DictDataset
        Training dataset
    val_set : DictDataset
        Validation dataset
    '''
    n = len(dset)
    section_size = n // num_sections
    remainder = n % num_sections

    # Create indices for each section
    sections = []
    start = 0
    for i in range(num_sections):
        # For the last section, include the remainder
        if i == num_sections - 1:
            end = start + section_size + remainder
        else:
            end = start + section_size
        sections.append(torch.arange(start, end))
        start = end

    # Convert list of tensors to a list of indices
    # Randomly permute sections
    section_indices = torch.randperm(num_sections)
    train_section_count = int(num_sections * (1 - val_fraction))
    train_sections = [sections[i] for i in section_indices[:train_section_count]]
    val_sections = [sections[i] for i in section_indices[train_section_count:]]

    # Flatten the list of indices
    train_indices = torch.cat(train_sections)
    val_indices = torch.cat(val_sections)

    dset_replicates_state = dset.replicates
    dset.replicates = True
    train_set = dset[train_indices.tolist()]
    val_set = dset[val_indices.tolist()]
    dset.replicates = dset_replicates_state
    train_set.replicates = False
    val_set.replicates = False
    
    # Debug assertions
    assert len(set(train_indices.tolist()).intersection(set(val_indices.tolist()))) == 0, "Train and val indices are not unique"
    assert len(set(train_indices.tolist()).union(set(val_indices.tolist()))) == len(dset), "Train and val indices do not cover the entire dataset"
    
    return train_set, val_set

def split_inds_by_trial_train_val_test(dset, inds, train_split, val_split, seed=1002):
    '''
    Split indices by trial into training, validation, and test sets.

    Parameters
    ----------
    dset : DictDataset
        The dataset containing trial indices.
    inds : torch.Tensor
        The indices to split.
    train_split : float
        The fraction of trials to use for training.
    val_split : float
        The fraction of trials to use for validation.
        The remaining trials (1 - train_split - val_split) will be used for testing.
    seed : int, optional
        The random seed. The default is 1002.

    Returns
    -------
    train_inds : torch.Tensor
        The indices for training.
    val_inds : torch.Tensor
        The indices for validation.
    test_inds : torch.Tensor
        The indices for testing.
    '''

    set_seeds(seed)
    trials = dset['trial_inds'].unique()
    rand_trials = torch.randperm(len(trials))
    
    # Calculate split points
    train_end = int(len(trials) * train_split)
    val_end = int(len(trials) * (train_split + val_split))
    
    # Split trials into three groups
    train_trials = trials[rand_trials[:train_end]]
    val_trials = trials[rand_trials[train_end:val_end]]
    test_trials = trials[rand_trials[val_end:]]
    
    # Map trials back to indices
    train_inds = inds[torch.isin(dset['trial_inds'][inds], train_trials)]
    val_inds = inds[torch.isin(dset['trial_inds'][inds], val_trials)]
    test_inds = inds[torch.isin(dset['trial_inds'][inds], test_trials)]
    
    return train_inds, val_inds, test_inds

def split_inds_by_trial(dset, inds, train_val_split, seed=1002):
    '''
    Split indices by trial into training and validation sets.

    Parameters
    ----------
    dset : DictDataset
        The dataset containing trial indices.
    inds : torch.Tensor
        The indices to split.
    train_val_split : float
        The fraction of indices to use for training.
    seed : int, optional
        The random seed. The default is 1002.

    Returns
    -------
    train_inds : torch.Tensor
        The indices for training.
    val_inds : torch.Tensor
        The indices for validation.
    '''

    set_seeds(seed)
    trials = dset['trial_inds'].unique()
    rand_trials = torch.randperm(len(trials))
    train_trials = trials[rand_trials[:int(len(trials) * train_val_split)]]
    train_inds = inds[torch.isin(dset['trial_inds'][inds], train_trials)]
    val_trials = trials[rand_trials[int(len(trials) * train_val_split):]]
    val_inds = inds[torch.isin(dset['trial_inds'][inds], val_trials)]
    return train_inds, val_inds
