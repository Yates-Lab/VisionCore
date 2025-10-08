"""
Time embedding utilities for neural data analysis.

This module provides functions for time embedding tensors, batches, and datasets.
Time embedding is a technique used to incorporate temporal context into neural data
by stacking multiple time points together.
"""

import torch
from .datasets import DictDataset

def time_embed_tensor(x, num_lags):
    '''
    Time embed a tensor.

    Inputs:
    -------
    x : torch.Tensor
        tensor to time embed (time, ...)
        Assumes time is the first dimension and there is a channel dimension

    num_lags : int
        number of lags to embed

    Returns:
    --------
    out : torch.Tensor
        time embedded tensor (time - num_lags + 1, num_lags, ...)
    '''
    # x is (time, ...)
    # output is (time - num_lags + 1, num_lags, ...)
    out = torch.stack([x[i:i+num_lags] for i in range(x.shape[0] - num_lags + 1)], dim=0)
    # permute dim 1 and 2
    perm = [0, 2, 1] + list(range(3, len(out.shape)))
    return out.permute(*perm)

def time_embed_batch(batch, num_lags, target='stim'):
    '''
    Time embed a target tensor in a batch.

    Inputs:
    -------
    batch : dict
        tensors in the batch (e.g., stim, robs, ...)
    num_lags : int 
        number of lags to embed
    target : str
        key for the target tensor that will be time embedded

    Returns:
    --------
    batch_embed : dict
        batch with the target tensor time embedded 
        and the other tensors unchanged but now with
        num_lags-1 fewer time points
    '''

    T = batch[target].shape[0]
    valid = range(num_lags-1, T)
    batch_embed = {k: v[valid] for k, v in batch.items() if k != target}
    batch_embed[target] = time_embed_tensor(batch[target], num_lags)
    return batch_embed
    
def time_embed_dataset(dset, blocks, num_lags, target='stim'):
    '''
    Time embed a dataset in blocks.

    Intputs:
    --------
    dset : DictDataset
        dataset to embed
    blocks : list of tuples or tensor shape (N, 2)
        list of blocks to embed (start, stop)
    num_lags : int
        number of lags to embed
    target : str
        key for the target tensor that will be time embedded
    
    Returns:
    --------
    DictDataset
        A new dataset with time-embedded data
    '''
    dset_embed = []
    for block in blocks:
        batch = dset[range(block[0], block[1])]
        batch_embed = time_embed_batch(batch, num_lags, target=target)
        dset_embed.append(batch_embed)

    # convert list of dicts to dict of tensors concatenated along dim 0
    dset_embed = {k: torch.cat([batch[k] for batch in dset_embed], dim=0) for k in dset[0].keys()}
    meta = dset.metadata
    meta['num_lags'] = num_lags
    return DictDataset(dset_embed, metadata=meta)
