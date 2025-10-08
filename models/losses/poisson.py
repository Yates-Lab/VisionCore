"""
Poisson loss functions for DataYatesV1.

This module contains loss functions and related utilities for Poisson distributed data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calc_poisson_bits_per_spike(robs, rhat, eps=1e-9):
    """
    Calculate bits per spike for Poisson distributed data.
    
    Parameters:
    -----------
    robs : torch.Tensor
        Observed spike counts
    rhat : torch.Tensor
        Predicted firing rates
    eps : float
        Small constant to avoid numerical issues
        
    Returns:
    --------
    torch.Tensor
        Bits per spike for each unit
    """
    # Ensure inputs are tensors
    if not isinstance(robs, torch.Tensor):
        robs = torch.tensor(robs, dtype=torch.float32)
    if not isinstance(rhat, torch.Tensor):
        rhat = torch.tensor(rhat, dtype=torch.float32)
    
    # Calculate log-likelihood
    # log p(y|λ) = y log(λ) - λ - log(y!)
    # For Poisson, we can ignore the log(y!) term when comparing models
    
    # Calculate mean firing rates
    mean_obs = robs.mean(dim=0)
    
    # Calculate log-likelihood for model predictions
    ll_model = (robs * torch.log(rhat + eps) - rhat).mean(dim=0)
    
    # Calculate log-likelihood for mean firing rate model (baseline)
    ll_mean = (robs * torch.log(mean_obs + eps) - mean_obs).mean(dim=0)
    
    # Calculate bits per spike
    # BPS = (ll_model - ll_mean) / mean_obs / log(2)
    bps = (ll_model - ll_mean) / (mean_obs + eps) / math.log(2)
    
    return bps

class PoissonBPSAggregator(nn.Module):
    """
    Module to calculate bits per spike by aggregating log likelihoods across batches.
    
    This module accumulates observed and predicted spike counts across batches
    and then calculates bits per spike at the end.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.reset()
        self.device = device
        
    def reset(self):
        """Reset the aggregator."""
        self.robs = []
        self.rhat = []
        self.dfs = None
        self.init = True
        
    def __call__(self, batch):
        """
        Accumulate observed and predicted spike counts from a batch.
        
        Parameters:
        -----------
        batch : dict
            Batch dictionary containing 'robs' and 'rhat' keys
        """
        assert 'robs' in batch.keys(), "'robs' must be a key in the batch"
        assert 'rhat' in batch.keys(), "'rhat' must be a key in the batch"
        
        if self.init == True:
            if 'dfs' in batch.keys():
                self.dfs = []
            self.init = False
            
        self.robs.append(batch['robs'].detach().cpu())
        self.rhat.append(batch['rhat'].detach().cpu())
        
        if self.dfs is not None and 'dfs' in batch.keys():
            self.dfs.append(batch['dfs'].detach().cpu())
    
    def has_data(self) -> bool:          # helper
        return len(self.robs) > 0
            
    def closure(self):
        """
        Calculate bits per spike from accumulated data.
        
        Returns:
        --------
        torch.Tensor
            Bits per spike for each unit
        """
        if not self.has_data():          # guard against empty list (can happen in mult-gpu training)
            return None
        
        robs = torch.cat(self.robs, dim=0)
        rhat = torch.cat(self.rhat, dim=0)
        
        if self.dfs is not None:
            dfs = torch.cat(self.dfs, dim=0)
            if dfs.ndim == 1:
                dfs = dfs[:, None]
            # Apply mask if available

            # if dfs.shape[1] == 1:
            #     dfs = dfs.expand(-1, robs.shape[1])
            # robs = robs * dfs
            # rhat = rhat * dfs
            if dfs.shape[1] == 1:
                # Single mask for all cells
                dfs = dfs.expand(-1, robs.shape[1])
                mask = dfs.squeeze().bool()
                robs = robs[mask]
                rhat = rhat[mask]
            else:
                # Calculate BPS per cell with cell-specific masking
                bps_list = []
                for cell_idx in range(robs.shape[1]):
                    cell_mask = dfs[:, cell_idx].bool()
                    if cell_mask.sum() > 0:  # Only if cell has valid samples
                        cell_robs = robs[cell_mask, cell_idx][:, None]  # Shape: [valid_samples, 1]
                        cell_rhat = rhat[cell_mask, cell_idx][:, None]  # Shape: [valid_samples, 1]
                        cell_bps = calc_poisson_bits_per_spike(cell_robs, cell_rhat)
                        bps_list.append(cell_bps)
                    else:
                        # No valid samples for this cell in this batch/rank
                        # Return NaN so it gets filtered out in distributed reduction
                        # (The other rank may have valid samples for this cell)
                        bps_list.append(torch.tensor([float('nan')]))
        
                # Return per-cell BPS with shape [num_cells]
                return torch.cat(bps_list)
            
        return calc_poisson_bits_per_spike(robs, rhat)

class MaskedLoss(nn.Module):
    """
    Loss function that applies a mask to the input.
    
    This is useful for ignoring certain samples in the loss calculation,
    such as padding or invalid samples.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function to apply
    pred_key : str
        Key for predictions in the batch dictionary
    target_key : str
        Key for targets in the batch dictionary
    mask_key : str
        Key for mask in the batch dictionary
    reduction : None or dimension
        Dimension to reduce the loss over. If None, reduction is across all dims
    """
    def __init__(self, loss_fn, pred_key='rhat', target_key='robs', mask_key='dfs', reduction=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key
        
    def forward(self, batch):
        """
        Calculate masked loss.
        
        Parameters:
        -----------
        batch : dict
            Batch dictionary containing prediction, target, and optional mask keys
            
        Returns:
        --------
        torch.Tensor
            Masked loss value
        """
        pred = batch[self.pred_key]
        target = batch[self.target_key]

        loss = self.loss_fn(pred, target)

        if self.mask_key in batch:
            mask = batch[self.mask_key]
            if mask.ndim == 1:
                mask = mask[:, None]
                div = mask.sum()
            else:
                div = mask.sum()
            
            if mask.shape[1] == 1:
                div = div * target.shape[1]
                # no expansion needed because broadcasting will take care of it
                
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
