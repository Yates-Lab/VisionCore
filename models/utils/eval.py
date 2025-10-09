import torch
import torch.nn as nn
from .general import ensure_tensor
from DataYatesV1.utils.data.datasets import CombinedEmbeddedDataset

import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

import os

__all__ = ['calc_poisson_bits_per_spike', 'PoissonBPSAggregator', 'MaskedLoss', 
           'EvalModule', 'FixRSVPPSTHEvalModule', 'PETHEvalModule']

# utilities ---------------------------------------------------------
def _apply_index(stim, indexer):
    """
    Return a view of `stim` according to `indexer`.
    Accepts:
      • None          → stim[:, None, ...]              (legacy default)
      • callable      → indexer(stim)
      • str           → eval(f"stim{indexer}")          (e.g. '[:,None,:,0,...]')
    """
    if indexer is None:                 # default legacy path
        return stim[:, None, ...]
    if callable(indexer):               # user-supplied function
        return indexer(stim)

    if isinstance(indexer, str):        # string: use restricted `eval`
        safe_globals = {}               # keep it minimal
        safe_locals  = {"stim": stim}
        try:
            return eval(f"stim{indexer}", safe_globals, safe_locals)
        except SyntaxError as e:
            raise ValueError(f"Bad indexing expression: {indexer}") from e

    raise TypeError("`stim_indexer` must be None, str, or callable.")


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

class PoissonMaskedLoss(MaskedLoss):
    '''
    A class for a Poisson loss function that applies a mask to the loss output.

    Parameters
    ----------

    pred_key : str (default='rhat')
        Key for the predicted values in the input dictionary
    target_key : str (default='robs')
        Key for the target values in the input dictionary
    mask_key : str (default='dfs')
        Key for the mask values in the input dictionary
    '''
    def __init__(self, pred_key='rhat', target_key='robs', mask_key='dfs'):
        super().__init__(nn.PoissonNLLLoss(log_input=False, full=False, reduction='none'), pred_key, target_key, mask_key)

class EvalModule:
    def wandb_logs(self, model):
        '''
        Logging function for the model.
        '''
        raise NotImplementedError
    
class FixRSVPPSTHEvalModule(EvalModule):
    """
    An evaluation module for FixRSVP datasets.

    Calculates R² for the predicted and observed spike rates across multiple repeats
    of the same stimulus in the FixRSVP paradigm.

    TODO: Implement explainable variance correction
    """
    def __init__(self, dset: CombinedEmbeddedDataset, cids=None, verbose=0):
        """
        Initialize the FixRSVP PSTH evaluation module.

        Parameters
        ----------
        dset : CombinedEmbeddedDataset
            The combined dataset containing a FixRSVP dataset.
        cids : array-like, optional
            Cell IDs to include in the analysis. If None, all cells are included.
        verbose : int, optional
            Verbosity level, by default 0.
        """
        self.verbose = verbose
        self.dset = dset.shallow_copy()

        # Extract the FixRSVP dataset
        self.fixrsvp_dset = [d for d in self.dset.dsets if d.metadata['name'] == 'fixrsvp']
        assert len(self.fixrsvp_dset) == 1, 'There should exactly be one fixrsvp dataset'
        self.fixrsvp_dset = self.fixrsvp_dset[0]

        # Determine which cells to analyze
        n_units = dset.dsets[0]['robs'].shape[1]
        if cids is None:
            cids = torch.arange(n_units)
        else:
            cids = ensure_tensor(cids)
        self.cids = cids

    def evaluate(self, inds, rhat=None, model=None, batch_size=256, dtype=torch.float32, device='cpu'):
        """
        Evaluate the model on FixRSVP data.

        Parameters
        ----------
        inds : torch.Tensor
            Indices to use for evaluation.
        rhat : torch.Tensor, optional
            Model predictions. If None, model must be provided.
        model : torch.nn.Module, optional
            Model to use for predictions. If None, rhat must be provided.
        batch_size : int, optional
            Batch size for evaluation, by default 256.
        dtype : torch.dtype, optional
            Data type for calculations, by default torch.float32.
        device : str, optional
            Device to use for calculations, by default 'cpu'.

        Returns
        -------
        tuple
            (observed rates, predicted rates, PSTH indices, trial indices)
        """
        assert (rhat is not None) or (model is not None), 'Either model predictions (rhat) or model must be provided'

        # Set dataset indices
        self.dset.inds = inds

        # Get model predictions if not provided
        if rhat is None:
            is_training = model.training
            init_device = model.device
            model.eval()
            model.to(device)
            with torch.no_grad():
                rhat = []
                for iB in tqdm(range(0, len(inds), batch_size), desc='Calculating rhat'):
                    batch = self.dset[iB:iB+batch_size]
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch = model(batch)
                    rhat.append(batch['rhat'].detach())
                rhat = torch.cat(rhat, dim=0)
            model.to(init_device)
            if is_training:
                model.train()
        else:
            rhat = ensure_tensor(rhat, dtype=dtype, device=device)

        # Filter predictions to selected cells
        rhat = rhat[:, self.cids]

        # Get observed spike rates
        old_keys_lags = self.dset.keys_lags
        self.dset.set_keys_lags({'robs': 0})
        data = self.dset[:]
        robs = data['robs'][:, self.cids]
        self.dset.set_keys_lags(old_keys_lags)

        # Get PSTH and trial indices from the FixRSVP dataset
        fixrsvp_dset = self.dset.dsets[self.dset.get_dataset_index('fixrsvp')]
        psth_inds = fixrsvp_dset['psth_inds'][inds[:, 1]]
        trial_inds = fixrsvp_dset['trial_inds'][inds[:, 1]]

        return robs, rhat, psth_inds, trial_inds

    @staticmethod
    def calculate_psth(robs, rhat, psth_inds, min_obs=20):
        """
        Calculate the PSTH from observed and predicted spike rates.

        Parameters
        ----------
        robs : torch.Tensor
            Observed spike rates.
        rhat : torch.Tensor
            Predicted spike rates.
        psth_inds : torch.Tensor
            PSTH indices for each time bin.
        min_obs : int, optional
            Minimum number of observations required for a PSTH bin, by default 20.

        Returns
        -------
        tuple
            (observed PSTH, predicted PSTH)
        """
        # Get unique PSTH indices and their counts
        _, inverse, counts = torch.unique(psth_inds, return_counts=True, return_inverse=True)

        # Find range of PSTH indices with sufficient observations
        i0, i1 = np.where(counts > min_obs)[0][[0, -1]]
        psth_ix = torch.arange(i0, i1+1)
        n_psth_bins = i1 - i0 + 1
        n_units = robs.shape[1]

        # Initialize PSTH arrays
        psth = torch.zeros(n_psth_bins, n_units)
        psth_hat = torch.zeros(n_psth_bins, n_units)

        # Calculate PSTH for each bin
        for i in range(n_psth_bins):
            psth[i] = robs[inverse == psth_ix[i]].mean(dim=0)
            psth_hat[i] = rhat[inverse == psth_ix[i]].mean(dim=0)

        # Convert to numpy for compatibility with plotting functions
        psth = psth.cpu().numpy()
        psth_hat = psth_hat.cpu().numpy()

        return psth, psth_hat

    @staticmethod
    def plot_psth_r2(psth, psth_hat):
        """
        Plot R² values for PSTH predictions across units.

        Parameters
        ----------
        psth : numpy.ndarray
            Observed PSTH values.
        psth_hat : numpy.ndarray
            Predicted PSTH values.

        Returns
        -------
        tuple
            (figure, axes, R² values)
        """
        assert psth.shape == psth_hat.shape, 'psth and psth_hat must have the same shape'

        # Calculate R² for each unit
        r2 = np.zeros(psth.shape[1])
        for iU in range(psth.shape[1]):
            r2[iU] = pearsonr(psth[:, iU], psth_hat[:, iU])[0]**2

        # Create stem plot of R² values
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.stem(r2)
        axs.set_xlabel('Unit')
        axs.set_ylabel('$R^2$')

        return fig, axs, r2

    def wandb_logs(self, model):
        """
        Generate logs for Weights & Biases.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.

        Returns
        -------
        dict
            Dictionary of metrics for logging.
        """
        # Evaluate model and calculate PSTH
        robs, rhat, psth_inds, _ = self.evaluate(model)
        psth, psth_hat = self.calculate_psth(robs, rhat, psth_inds)

        # Generate R² plot
        fig, _, r2 = self.plot_psth_r2(psth, psth_hat)

        # Calculate mean R²
        mean_r2 = np.nanmean(r2)

        # Return metrics for logging
        return {
            'mean_fixrsvp_r2': mean_r2,
            'fixrsvp_r2': wandb.Image(fig),
        }

class PETHEvalModule(EvalModule):
    """
    An evaluation module for calculating the peri-event time histogram (PETH) for a set of events.

    The PETH shows the average neural response around specific events like saccades or flashes.
    It helps characterize how neurons respond to these events over time.

    Note: Currently does not take dfs (data filtering status) into account. This needs to be
    implemented in the future if different cells are valid at different times.
    """
    def __init__(self, dset: CombinedEmbeddedDataset, event_inds,
                 n_pre, n_post, cids=None,
                 name='peth_eval', verbose=0):
        """
        Initialize the PETH evaluation module.

        Parameters
        ----------
        dset : CombinedEmbeddedDataset
            A dataset to calculate the PETH over.
        event_inds : torch.Tensor
            Indices of events to calculate the PETH around.
        n_pre : int
            Number of time bins before the event to include.
        n_post : int
            Number of time bins after the event to include.
        cids : array-like, optional
            Cell IDs to include in the analysis. If None, all cells are included.
        name : str, optional
            Name for this PETH evaluation, by default 'peth_eval'.
        verbose : int, optional
            Verbosity level, by default 0.
        """
        self.dset = dset.shallow_copy()
        self.event_inds = event_inds
        self.lags = np.arange(-n_pre, n_post)
        self.name = name

        # Determine which cells to analyze
        n_units = dset.dsets[0]['robs'].shape[1]
        if cids is None:
            cids = torch.arange(n_units)
        else:
            cids = ensure_tensor(cids)
        self.cids = cids

    def evaluate(self, inds, rhat=None, model=None, batch_size=256, dtype=torch.float32, device='cpu'):
        """
        Calculate the peri-event time histogram for observed and predicted spike rates.

        Parameters
        ----------
        inds : torch.Tensor
            Indices to use for evaluation.
        rhat : torch.Tensor, optional
            Model predictions. If None, model must be provided.
        model : torch.nn.Module, optional
            Model to use for predictions. If None, rhat must be provided.
        batch_size : int, optional
            Batch size for evaluation, by default 256.
        dtype : torch.dtype, optional
            Data type for calculations, by default torch.float32.
        device : str, optional
            Device to use for calculations, by default 'cpu'.

        Returns
        -------
        tuple
            (observed PETH, predicted PETH)
        """
        assert (rhat is not None) or (model is not None), 'Either model predictions (rhat) or model must be provided'

        # Set dataset indices
        self.dset.inds = inds

        # Get model predictions if not provided
        if rhat is None:
            is_training = model.training
            init_device = model.device
            model.eval()
            model.to(device)
            with torch.no_grad():
                rhat = []
                for iB in tqdm(range(0, len(inds), batch_size), desc='Calculating rhat'):
                    batch = self.dset[iB:iB+batch_size]
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch = model(batch)
                    rhat.append(batch['rhat'].detach())
                rhat = torch.cat(rhat, dim=0)
            model.to(init_device)
            if is_training:
                model.train()
        else:
            rhat = ensure_tensor(rhat, dtype=dtype, device=device)

        # Filter predictions to selected cells
        rhat = rhat[:, self.cids]

        # Get observed spike rates
        old_keys_lags = self.dset.keys_lags
        self.dset.set_keys_lags({'robs': 0})
        self.dset.inds = inds
        robs = self.dset[:]['robs'][:, self.cids]
        self.dset.set_keys_lags(old_keys_lags)

        # Initialize PETH arrays
        peth_robs = torch.zeros(len(self.lags), len(self.cids))
        peth_rhat = torch.zeros(len(self.lags), len(self.cids))

        # Calculate PETH for each lag
        for i, lag in tqdm(enumerate(self.lags), desc='Calculating PETH'):
            # Shift event indices by current lag
            lagged_inds = self.event_inds.clone()
            lagged_inds[:, 1] += lag

            # Find indices that match the lagged event indices
            # NOTE: not checking for cross-trial continuity
            inds_inds = []
            for iD in range(len(self.dset.dsets)):
                # Get lagged indices for this dataset
                dset_lagged_inds = lagged_inds[lagged_inds[:, 0] == iD, 1]

                # Find matching indices in the evaluation set
                inds_mask = torch.nonzero(inds[:, 0] == iD)
                dset_inds = inds[inds_mask, 1]
                dset_inds = torch.isin(dset_inds, dset_lagged_inds)
                inds_inds.append(inds_mask[dset_inds])

            # Combine indices from all datasets
            inds_inds = torch.cat(inds_inds, dim=0)

            # Calculate mean response at this lag
            peth_robs[i] = torch.mean(robs[inds_inds], dim=0)
            peth_rhat[i] = torch.mean(rhat[inds_inds], dim=0)

        return peth_robs, peth_rhat

    @staticmethod
    def plot_peth_r2(peth_robs, peth_rhat):
        """
        Plot R² values for PETH predictions across units.

        Parameters
        ----------
        peth_robs : torch.Tensor
            Observed PETH values.
        peth_rhat : torch.Tensor
            Predicted PETH values.

        Returns
        -------
        tuple
            (figure, axes, R² values)
        """
        assert peth_robs.shape == peth_rhat.shape, 'peth_robs and peth_rhat must have the same shape'

        # Calculate R² for each unit
        r2 = np.zeros(peth_robs.shape[1])
        for iU in range(peth_robs.shape[1]):
            r2[iU] = pearsonr(peth_robs[:, iU], peth_rhat[:, iU])[0]**2

        # Create stem plot of R² values
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.stem(r2)
        ax.set_xlabel('Unit')
        ax.set_ylabel('$R^2$')

        return fig, ax, r2

    def wandb_logs(self, model):
        """
        Generate logs for Weights & Biases.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.

        Returns
        -------
        dict
            Dictionary of metrics for logging.
        """
        # Evaluate model
        peth_robs, peth_rhat = self.evaluate(model)

        # Generate R² plot
        fig, ax, r2 = self.plot_peth_r2(peth_robs, peth_rhat)
        ax.set_title(f'{self.name} $R^2$')

        # Calculate mean R²
        mean_r2 = r2.mean()

        # Return metrics for logging
        return {
            f'mean_{self.name}_r2': mean_r2,
            f'{self.name}_r2': wandb.Image(fig),
        }

class STAEvalModule(EvalModule):
    """
    An evaluation module for calculating the spike-triggered average (STA) over a dataset.

    The STA is a measure of the average stimulus that precedes a spike, which helps
    characterize the receptive field properties of neurons.
    """
    def __init__(self, dset: CombinedEmbeddedDataset, cids=None, stim_indexer=None, verbose=0):
        """
        Initialize the STA evaluation module.

        Parameters
        ----------
        dset : CombinedEmbeddedDataset
            A dataset to calculate the STA over (typically a Gaborium dataset)
        cids : array-like, optional
            Cell IDs to include in the analysis. If None, all cells are included.
        stim_indexer : callable, optional
            Function to index the stimulus data. If None, the default indexing is used.
            Pass in a string for how to access stim when calculating the STA (e.g. '[:,None,:,0,...]' )
        verbose : int, optional
            Verbosity level, by default 0
        """
        self.verbose = verbose
        self.dset = dset.shallow_copy()
        self.stim_indexer = stim_indexer

        # Determine which cells to analyze
        if cids is None:
            n_units = dset.dsets[0]['robs'].shape[1]
            cids = torch.arange(n_units)
        else:
            cids = ensure_tensor(cids)
        self.cids = cids

    def evaluate(self, inds, rhat=None, model=None, batch_size=256, dtype=torch.float32, device='cpu'):
        """
        Calculate the spike-triggered average (STA) and spike-triggered covariance (STE).

        Parameters
        ----------
        inds : torch.Tensor
            Indices to use for evaluation.
        rhat : torch.Tensor, optional
            Model predictions. If None, model must be provided.
        model : torch.nn.Module, optional
            Model to use for predictions. If None, rhat must be provided.
        batch_size : int, optional
            Batch size for evaluation, by default 256.
        dtype : torch.dtype, optional
            Data type for calculations, by default torch.float32.
        device : str, optional
            Device to use for calculations, by default 'cpu'.

        Returns
        -------
        tuple
            (spike-triggered average, spike-triggered covariance)
        """
        assert (rhat is not None) or (model is not None), 'Either model predictions (rhat) or model must be provided'

        # Set dataset indices
        self.dset.inds = inds

        # Get model predictions if not provided
        if rhat is None:
            is_training = model.training
            init_device = model.device
            model.eval()
            model.to(device)
            with torch.no_grad():
                rhat = []
                for iB in tqdm(range(0, len(inds), batch_size), desc='Calculating rhat'):
                    batch = self.dset[iB:iB+batch_size]
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch = model(batch)
                    rhat.append(batch['rhat'].detach())
                rhat = torch.cat(rhat, dim=0)
            model.to(init_device)
            if is_training:
                model.train()
        else:
            rhat = ensure_tensor(rhat, dtype=dtype, device=device)

        # Filter predictions to selected cells and calculate total spikes
        rhat = rhat[:, self.cids]
        n_spikes = rhat.sum(0)
        sta = None
        ste = None

        # Calculate STA and STE in batches
        for iB in tqdm(range(0, len(self.dset), batch_size), desc='Calculating STA'):
            batch_start = iB
            batch_end = min(iB + batch_size, len(self.dset))
            batch = self.dset[batch_start:batch_end]
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_rhat = rhat[batch_start:batch_end]

            # access stim using indexer
            stim_view  = _apply_index(batch['stim'], self.stim_indexer)

            # Initialize STA and STE tensors on first batch
            if sta is None:
                sta = torch.zeros((len(self.cids), ) + stim_view.shape[2:], dtype=dtype, device=device)
                ste = torch.zeros_like(sta)

            # Update STA and STE with batch contribution
            # STA: weighted average of stimuli, weighted by spike rate
            sta += (stim_view * batch_rhat[:,:,None,None,None]).sum(dim=0) / n_spikes[:,None,None,None]
            # STE: weighted average of squared stimuli, weighted by spike rate
            ste += (stim_view**2 * batch_rhat[:,:,None,None,None]).sum(dim=0) / n_spikes[:,None,None,None]

        # Move results to CPU for further processing
        sta = sta.to('cpu')
        ste = ste.to('cpu')

        return sta, ste