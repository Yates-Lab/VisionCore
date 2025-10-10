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


class ZeroInflatedPoissonNLLLoss(nn.Module):
    """
    Zero-Inflated Poisson (ZIP) negative log-likelihood loss.

    The ZIP distribution models data that has excess zeros beyond what a standard
    Poisson distribution would predict. It's a mixture of:
    - A point mass at zero (with probability pi)
    - A Poisson distribution (with probability 1-pi and rate lambda)

    The model outputs two parameters:
    - lambda (lam): The Poisson rate parameter (must be > 0)
    - pi: The probability of the zero-inflation component (must be in [0, 1])

    Parameters
    ----------
    log_input : bool, default=False
        If True, the loss expects log(lambda) as input instead of lambda.
        This can improve numerical stability during training.
    eps : float, default=1e-8
        Small constant for numerical stability in log computations
    reduction : str, default='none'
        Specifies the reduction to apply to the output:
        'none': no reduction will be applied
        'mean': the sum of the output will be divided by the number of elements
        'sum': the output will be summed
    pi_key : str, default='pi'
        Key for the zero-inflation probability in the batch dictionary

    Notes
    -----
    The negative log-likelihood is computed as (omitting constant factorial term):
    - For y = 0: -log(pi + (1-pi) * exp(-lambda))
    - For y > 0: -log(1-pi) - y*log(lambda) + lambda

    The factorial term log(y!) is omitted as it doesn't affect gradients.
    The model should output both 'rhat' (lambda or log(lambda)) and 'pi'.
    """

    def __init__(self, log_input=False, eps=1.0e-8, reduction='none', pi_key='pi'):
        super().__init__()
        self.log_input = log_input
        self.eps = eps
        self.reduction = reduction
        self.pi_key = pi_key

    def forward(self, input, target):
        """
        Compute ZIP negative log-likelihood.

        Parameters
        ----------
        input : torch.Tensor or tuple of torch.Tensor
            If tuple: (lam, pi) where lam is the Poisson rate and pi is zero-inflation prob
            If tensor: assumes this is lam, and pi must be provided separately via batch dict
        target : torch.Tensor
            Observed counts (non-negative integers)

        Returns
        -------
        torch.Tensor
            Negative log-likelihood loss
        """
        # Handle input format
        if isinstance(input, tuple):
            lam, pi = input
        else:
            # This case is for compatibility with MaskedLoss wrapper
            # pi should be extracted from batch in the wrapper
            lam = input
            pi = None  # Will be handled by wrapper

        # Convert from log space if needed
        if self.log_input:
            lam = torch.exp(lam)

        # Ensure lam is positive
        lam = torch.clamp(lam, min=self.eps)

        # If pi is provided as part of input tuple
        if pi is not None:
            # Ensure pi is in [0, 1]
            pi = torch.clamp(pi, min=self.eps, max=1.0 - self.eps)

            # Compute negative log-likelihood (without factorial term - doesn't affect gradients)
            # For y = 0: -log(pi + (1-pi) * exp(-lam))
            # For y > 0: -log(1-pi) - y*log(lam) + lam
            log_p = torch.where(
                target == 0,
                torch.log(pi + (1 - pi) * torch.exp(-lam) + self.eps),
                torch.log(1 - pi + self.eps)
                + target * torch.log(lam + self.eps)
                - lam
            )
            nll = -log_p
        else:
            # Fallback to standard Poisson if pi not provided
            # This maintains compatibility (without factorial term)
            nll = lam - target * torch.log(lam + self.eps)

        # Apply reduction
        if self.reduction == 'none':
            return nll
        elif self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class MaskedZIPNLLLoss(nn.Module):
    """
    Masked Zero-Inflated Poisson negative log-likelihood loss.

    This wrapper applies masking to the ZIP loss, similar to MaskedLoss but
    specifically designed for ZIP which requires both lambda and pi parameters.

    Parameters
    ----------
    log_input : bool, default=False
        If True, expects log(lambda) as input
    eps : float, default=1e-8
        Small constant for numerical stability
    pred_key : str, default='rhat'
        Key for the predicted lambda values in the input dictionary
    pi_key : str, default='pi'
        Key for the predicted pi values in the input dictionary
    target_key : str, default='robs'
        Key for the target values in the input dictionary
    mask_key : str, default='dfs'
        Key for the mask values in the input dictionary

    Example
    -------
    >>> loss_fn = MaskedZIPNLLLoss(log_input=False)
    >>> batch = {
    ...     'rhat': predicted_lambda,  # (N, n_units)
    ...     'pi': predicted_pi,         # (N, n_units)
    ...     'robs': observed_counts,    # (N, n_units)
    ...     'dfs': data_filter_mask     # (N, n_units)
    ... }
    >>> loss = loss_fn(batch)
    """

    def __init__(self, log_input=False, eps=1.0e-8,
                 pred_key='rhat', pi_key='pi', target_key='robs', mask_key='dfs'):
        super().__init__()
        self.loss_fn = ZeroInflatedPoissonNLLLoss(log_input=log_input, eps=eps, reduction='none')
        self.pred_key = pred_key
        self.pi_key = pi_key
        self.target_key = target_key
        self.mask_key = mask_key
        # For compatibility with MaskedLoss interface
        self.reduction = 'none'

    def forward(self, batch):
        """
        Compute masked ZIP loss.

        Parameters
        ----------
        batch : dict
            Dictionary containing predictions, targets, and optional mask

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        lam = batch[self.pred_key]
        target = batch[self.target_key]

        # Get pi if available, otherwise use a default (no zero-inflation)
        if self.pi_key in batch:
            pi = batch[self.pi_key]
            loss = self.loss_fn((lam, pi), target)
        else:
            # Fallback: no zero-inflation (pi=0)
            # This makes it equivalent to standard Poisson
            loss = self.loss_fn(lam, target)

        # Apply mask if present
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
