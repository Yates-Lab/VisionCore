"""
Poisson loss functions for DataYatesV1.

This module contains loss functions and related utilities for Poisson distributed data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..utils.general import ensure_tensor

def calc_poisson_bits_per_spike(r_pred, r_obs, dfs=None):
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
    r_pred = ensure_tensor(r_pred)
    r_obs = ensure_tensor(r_obs)
    if dfs is None:
        dfs = torch.ones_like(r_obs)
        
    if r_pred.ndim == 1:
        r_pred = r_pred.unsqueeze(1)
    if r_obs.ndim == 1: 
        r_obs = r_obs.unsqueeze(1)
    if dfs.ndim == 1:
        dfs = dfs.unsqueeze(1)
    
    assert r_pred.shape == r_obs.shape, 'r_pred and r_obs must have the same shape'
    assert len(r_pred.shape) == len(dfs.shape), 'r_pred and dfs must have the same shape'
    
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


class PoissonBPSAggregator(nn.Module):
    """
    Module to calculate bits per spike by aggregating 
    log likelihoods across batches.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.reset()
        self.device=device
    def reset(self):
        self.robs = []
        self.rhat = []
        self.dfs = None
        self.init = True
    def __call__(self, batch):
        assert 'robs' in batch.keys(), "'robs' must be a key in the batch"
        assert 'rhat' in batch.keys(), "'rhat' must be a key in the batch"
        if self.init == True:
            if 'dfs' in batch.keys():
                self.dfs = []
            self.init = False
        
        if self.dfs is not None:
            assert 'dfs' in batch.keys(), "'dfs' must be a key in the batch"
            self.dfs.append(batch['dfs'].detach().to(self.device))
        self.robs.append(batch['robs'].detach().to(self.device))
        self.rhat.append(batch['rhat'].detach().to(self.device))
    def closure(self):
        self.robs = torch.cat(self.robs, dim=0)
        self.rhat = torch.cat(self.rhat, dim=0)
        if self.dfs is not None:
            self.dfs = torch.cat(self.dfs, dim=0)
        # print('CHECK',self.rhat.shape, self.robs.shape, self.dfs.shape)
        bps = calc_poisson_bits_per_spike(self.rhat, self.robs, self.dfs)
        self.reset()
        return bps
    
class MaskedLoss(nn.Module):
    '''
    A wrapper class for a loss function that applies a mask to the loss output.

    Parameters
    ----------

    loss_fn : torch.nn.Module
        must have a reduction attribute set to 'none'
    pred_key : str (default='rhat')
        Key for the predicted values in the input dictionary
    target_key : str (default='robs')
        Key for the target values in the input dictionary
    mask_key : str (default='dfs')
        Key for the mask values in the input dictionary
    '''
    def __init__(self, loss_fn, pred_key='rhat', target_key='robs', mask_key='dfs'):
        super().__init__()
        self.loss_fn = loss_fn
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key

        # assert the reduction is 'none' # TODO: this will work for pytorch losses. custom losses need a reduction attribute
        assert self.loss_fn.reduction == 'none', 'Loss function must have reduction set to none'
    
    def forward(self, batch):
        pred = batch[self.pred_key]
        target = batch[self.target_key]

        loss = self.loss_fn(pred, target)

        if self.mask_key in batch:
            mask = batch[self.mask_key]
            if mask.ndim == 1:
                mask = mask[:, None]
                div = mask.sum() * target.shape[1]
            else:
                div = mask.sum()

            loss = loss * mask
            loss = loss.sum() / div
        else:
            loss = loss.mean()

        return loss

class MaskedPoissonNLLLoss(MaskedLoss):
    """
    Masked Poisson negative log-likelihood loss function.

    This is a specialized version of the MaskedLoss class for Poisson NLL.
    """
    def __init__(self, pred_key='rhat', target_key='robs', mask_key='dfs'):
        super().__init__(nn.PoissonNLLLoss(log_input=False, full=False, reduction='none'), pred_key, target_key, mask_key)

