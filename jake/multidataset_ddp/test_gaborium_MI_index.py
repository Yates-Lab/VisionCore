
#%%
import sys
sys.path.append('.')

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from DataYatesV1 import enable_autoreload, get_free_device, prepare_data

#%%
dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_basic_multi_120"
# dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_cones_multi"
# List full paths to *.yaml files that do not contain "base" in the name
yaml_files = [
    f for f in os.listdir(dataset_configs_path)
    if f.endswith(".yaml") and "base" not in f
]

from DataYatesV1.models.config_loader import load_dataset_configs
dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)
# %%

@torch.no_grad()
def estimate_spatial_whitener(stim, dfs, ridge=1e-2, subsample=1):
    N,H,W = stim.shape
    d = H*W
    w = dfs.float().mean(dim=1)
    idx = torch.arange(N, device=stim.device)[::subsample]
    X = stim[idx].reshape(-1, d)
    w = w[idx].clamp_min(0)

    ws = (w.sum() + 1e-8)
    mu = (w[:,None] * X).sum(dim=0) / ws
    X0 = X - mu
    C = (X0.T @ (w[:,None] * X0)) / ws
    # do in double for stability
    evals, evecs = torch.linalg.eigh(C.double())
    evals = evals.clamp_min(0).float()
    evecs = evecs.float()

    scales = (evals + ridge).rsqrt()  # 1/sqrt(λ+ridge)

    def Wx_flat(x_flat):
        y = (x_flat @ evecs) * scales
        return y @ evecs.T

    # per-pixel baseline energy (diag of S = W C W^T)
    # diag(V diag(λ/(λ+ridge)) V^T)  == (V ⊙ V) @ (λ/(λ+ridge))
    shrink = evals / (evals + ridge)                    # [d]
    mu2_vec = (evecs**2) @ shrink                       # [d]
    mu2_hw  = mu2_vec.view(H, W)                        # [H,W]

    return mu.view(H,W), Wx_flat, evecs, evals, ridge, mu2_hw

@torch.no_grad()
def whiten_frames(stim, mu_hw, Wx_flat, chunk=4096):
    N,H,W = stim.shape
    d = H*W
    out = torch.empty_like(stim)
    for start in range(0, N, chunk):
        end = min(N, start+chunk)
        X = (stim[start:end] - mu_hw).reshape(end-start, d)
        out[start:end] = Wx_flat(X).reshape(end-start, H, W)
    return out

def create_gabor_filter(H, W, center_x, center_y, orientation, frequency, sigma, phase=0):
    """
    Create a 2D Gabor filter

    Args:
        H, W: height and width of filter
        center_x, center_y: center position (in pixels)
        orientation: orientation in radians
        frequency: spatial frequency (cycles per pixel)
        sigma: standard deviation of Gaussian envelope
        phase: phase offset in radians

    Returns:
        gabor: [H, W] tensor
    """
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                         torch.arange(W, dtype=torch.float32), indexing='ij')

    # Center coordinates
    x = x - center_x
    y = y - center_y

    # Rotate coordinates
    orientation = torch.tensor(orientation)
    x_rot = x * torch.cos(orientation) + y * torch.sin(orientation)
    y_rot = -x * torch.sin(orientation) + y * torch.cos(orientation)

    # Gaussian envelope
    gaussian = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))

    # Sinusoidal component
    sinusoid = torch.cos(2 * torch.pi * frequency * x_rot + phase)

    # Combine
    gabor = gaussian * sinusoid

    # normalize
    gabor = gabor / gabor.norm()
    gabor = gabor

    return gabor

def create_LN_model(H, W, center_x=25, center_y=25, orientation=0, frequency=0.1, sigma=5):
    """
    Create LN model: single gabor filter + relu

    Returns:
        filter: [H, W] gabor filter
        model_fn: function that takes stimulus and returns rate
    """
    gabor_filter = create_gabor_filter(H, W, center_x, center_y, orientation, frequency, sigma)

    def model_fn(stim):
        # stim: [N, H, W]
        # Apply filter and relu
        response = torch.einsum('nhw,hw->n', stim, gabor_filter)
        rate = F.relu(response)**2
        return rate

    return gabor_filter, model_fn

def create_Energy_model(H, W, center_x=25, center_y=25, orientation=0, frequency=0.1, sigma=5):
    """
    Create Energy model: quadrature pair of gabor filters + square + sum + sqrt

    Returns:
        filters: [2, H, W] quadrature pair
        model_fn: function that takes stimulus and returns rate
    """
    # Quadrature pair (0 and 90 degree phase)
    gabor1 = create_gabor_filter(H, W, center_x, center_y, orientation, frequency, sigma, phase=0)
    gabor2 = create_gabor_filter(H, W, center_x, center_y, orientation, frequency, sigma, phase=torch.pi/2)

    filters = torch.stack([gabor1, gabor2], dim=0)

    def model_fn(stim):
        # stim: [N, H, W]
        # Apply both filters
        response1 = torch.einsum('nhw,hw->n', stim, gabor1)
        response2 = torch.einsum('nhw,hw->n', stim, gabor2)

        # Energy: sqrt(response1^2 + response2^2)
        energy = torch.sqrt(response1**2 + response2**2 + 1e-8)  # small epsilon for stability
        return energy

    return filters, model_fn

def generate_simulated_responses(dataset_config, model_type='LN', n_cells=10, noise_level=0.1):
    """
    Generate simulated responses using LN or Energy models

    Args:
        dataset_config: dataset configuration
        model_type: 'LN' or 'Energy'
        n_cells: number of simulated cells
        noise_level: Poisson noise level

    Returns:
        robs, rhat, dfs: simulated observed rates, predicted rates, and data flags
        stim_data: stimulus data for analysis
    """
    # Load stimulus data
    dataset_config['transforms']['stim'] = {'source': 'stim',
        'ops': [{'pixelnorm': {}}],
        'expose_as': 'stim'}
    dataset_config['keys_lags']['stim'] = list(range(25))
    dataset_config['types'] = ['gaborium']

    train_data, val_data, dataset_config = prepare_data(dataset_config)
    stim_indices = torch.concatenate([train_data.get_dataset_inds('gaborium'), val_data.get_dataset_inds('gaborium')], dim=0)

    # shallow copy the dataset not to mess it up
    data = val_data.shallow_copy()
    data.inds = stim_indices
    dset_idx = np.unique(stim_indices[:,0]).item()

    # Get stimulus at lag 0 (current frame)
    stim = data.dsets[dset_idx]['stim'][data.inds[:,1]]  # [N, H, W]
    N, H, W = stim.shape

    # Create models for each cell
    rhat_list = []
    modlist = []
    for cell_idx in range(n_cells):
        # Random parameters for each cell
        center_x = torch.randint(10, W-10, (1,)).item()
        center_y = torch.randint(10, H-10, (1,)).item()
        orientation = torch.rand(1).item() * torch.pi
        frequency = 0.05 + torch.rand(1).item() * 0.1  # 0.05 to 0.15
        sigma = 3 + torch.rand(1).item() * 4  # 3 to 7

        if model_type == 'LN':
            true_filters, model_fn = create_LN_model(H, W, center_x, center_y, orientation, frequency, sigma)
        elif model_type == 'Energy':
            true_filters, model_fn = create_Energy_model(H, W, center_x, center_y, orientation, frequency, sigma)
        else:
            raise ValueError("model_type must be 'LN' or 'Energy'")

        # Generate responses
        cell_rhat = model_fn(stim)  # [N]
        rhat_list.append(cell_rhat)
        modlist.append(true_filters)

    # Stack responses
    rhat = torch.stack(rhat_list, dim=1)  # [N, n_cells]

    # Add Poisson noise to create robs
    rhat_scaled = rhat * 1  # scale up for reasonable spike counts
    robs = torch.poisson(rhat_scaled + noise_level)

    # All simulated data is valid
    dfs = torch.ones_like(robs)

    return robs, rhat_scaled, dfs, data, dset_idx, modlist

def get_stsums(robs, rhat, dfs, data, dset_idx, lags=[-12, 0], whiten=True):
    """
    Compute spike-triggered sums for STA/STE analysis

    Args:
        robs: observed responses [N, n_cells]
        rhat: predicted responses [N, n_cells]
        dfs: data flags [N, n_cells] - all ones since we simulated clean data
        data: dataset object
        dset_idx: dataset index
        lags: list of lags to analyze
        whiten: whether to whiten stimulus
    """
    if whiten:
        # Get stimulus frames corresponding to our indices
        stim_subset = data.dsets[dset_idx]['stim'][data.inds[:,1]]
        mu_hw, Wx_flat, V, evals, ridge, mu2_hw = estimate_spatial_whitener(stim_subset, dfs)

    norm_dfs = dfs.sum(0) # if forward
    norm_robs = (robs * dfs).sum(0) # if reverse
    norm_rhat = (rhat * dfs).sum(0) # if reverse

    n_cells = robs.shape[1]
    n_lags = len(lags)
    H, W  = data.dsets[dset_idx]['stim'].shape[1:3]
    stsum_stim_robs = torch.zeros((n_lags, H, W, n_cells))
    stsum_energy_robs = torch.zeros((n_lags, H, W, n_cells))
    stsum_stim_rhat = torch.zeros((n_lags, H, W, n_cells))
    stsum_energy_rhat = torch.zeros((n_lags, H, W, n_cells))

    fullN = data.dsets[dset_idx]['stim'].shape[0]
    for i, lag in enumerate(tqdm(lags)):
        idx = (data.inds[:,1] - lag).cpu()
        assert idx.min().item() >= 0 and idx.max().item() < fullN, f"out-of-bounds for lag {lag}"

        stim = data.dsets[dset_idx]['stim'][data.inds[:,1]-lag]
        if whiten:
            stim = whiten_frames(stim, mu_hw, Wx_flat)
        stsum_stim_robs[i] = torch.einsum('thw, tc->hwc', stim, robs*dfs)
        stsum_energy_robs[i] = torch.einsum('thw, tc->hwc', stim.pow(2), robs*dfs)
        stsum_stim_rhat[i] = torch.einsum('thw, tc->hwc', stim, rhat*dfs)
        stsum_energy_rhat[i] = torch.einsum('thw, tc->hwc', stim.pow(2), rhat*dfs)

    return {'stsum_stim_robs': stsum_stim_robs, 'stsum_energy_robs': stsum_energy_robs, 'stsum_stim_rhat': stsum_stim_rhat, 'stsum_energy_rhat': stsum_energy_rhat, 'norm_dfs': norm_dfs, 'norm_robs': norm_robs, 'norm_rhat': norm_rhat, 'mu2_hw': mu2_hw}

@torch.no_grad()
def compute_neff(w):  # w = robs*dfs or rhat*dfs, shape [T,C]
    num = w.sum(0).pow(2)
    den = w.pow(2).sum(0).clamp_min(1e-8)
    return num / den  # [C]

@torch.no_grad()
def sta_ste_zmaps_from_whitened_sums(out, mu2_hw, robs, dfs, rhat=None):
    # reverse-corr means
    norm_robs = out['norm_robs']
    STA_robs = out['stsum_stim_robs']   / norm_robs.view(1,1,1,-1).clamp_min(1e-8)
    STE_robs = out['stsum_energy_robs'] / norm_robs.view(1,1,1,-1).clamp_min(1e-8)

    Neff_robs = compute_neff(robs*dfs).view(1,1,1,-1)

    Z_STA_robs = STA_robs * Neff_robs.sqrt()
    # --- corrected STE baseline & scale (per pixel) ---
    mu2 = mu2_hw.view(1, *mu2_hw.shape, 1)             # [1,H,W,1]
    Z_STE_robs = (STE_robs / mu2 - 1.0) * (Neff_robs/2.0).sqrt()

    Z_STA_rhat = Z_STE_rhat = None
    if rhat is not None:
        norm_rhat = out['norm_rhat']
        STA_rhat = out['stsum_stim_rhat']   / norm_rhat.view(1,1,1,-1).clamp_min(1e-8)
        STE_rhat = out['stsum_energy_rhat'] / norm_rhat.view(1,1,1,-1).clamp_min(1e-8)
        Neff_rhat = compute_neff(rhat*dfs).view(1,1,1,-1)
        Z_STA_rhat = STA_rhat * Neff_rhat.sqrt()
        Z_STE_rhat = (STE_rhat / mu2 - 1.0) * (Neff_rhat/2.0).sqrt()

    return Z_STA_robs, Z_STE_robs, Z_STA_rhat, Z_STE_rhat

#%%

#%%

def run_simulation_analysis(dataset_config, model_type='LN', n_cells=10, lags=[-2, 0], whiten=True):
    """
    Run complete simulation analysis for LN or Energy model

    Args:
        dataset_config: dataset configuration
        model_type: 'LN' or 'Energy'
        n_cells: number of simulated cells
        lags: lags to analyze

    Returns:
        results: dictionary with analysis results
    """
    print(f"Running {model_type} model simulation with {n_cells} cells...")

    # Generate simulated responses
    robs, rhat, dfs, data, dset_idx, modlist = generate_simulated_responses(
        dataset_config, model_type=model_type, n_cells=n_cells
    )

    # Compute spike-triggered sums
    print("Computing spike-triggered sums...")
    stsums = get_stsums(robs, rhat, dfs, data, dset_idx, lags=lags, whiten=whiten)

    # Compute Z-scored STA/STE
    print("Computing Z-scored STA/STE...")
    mu2_hw = stsums['mu2_hw']
    Z_STA_robs, Z_STE_robs, Z_STA_rhat, Z_STE_rhat = sta_ste_zmaps_from_whitened_sums(stsums, mu2_hw, robs, dfs, rhat)

    results = {
        'model_type': model_type,
        'models': modlist,
        'n_cells': n_cells,
        'Z_STA_robs': Z_STA_robs, #[:,1:-1,1:-1,:],
        'Z_STE_robs': Z_STE_robs, #[:,1:-1,1:-1,:],
        'Z_STA_rhat': Z_STA_rhat, #[:,1:-1,1:-1,:],
        'Z_STE_rhat': Z_STE_rhat, #[:,1:-1,1:-1,:],
        'robs': robs,
        'rhat': rhat,
        'dfs': dfs
    }

    return results

# %%

# Example usage
dataset_config = dataset_configs[0]  # Use first available config

# Run LN model simulation
ln_results = run_simulation_analysis(dataset_config, model_type='LN', n_cells=20, lags=[-2, 0], whiten=True)

# Run Energy model simulation
energy_results = run_simulation_analysis(dataset_config, model_type='Energy', n_cells=20, lags=[-2, 0], whiten=True)


# %%
results = ln_results
cc = 0
noise_mask = 0
print("STA null mean, SD:", results['Z_STA_robs'][noise_mask,:,:,cc].mean().item(),
                            results['Z_STA_robs'][noise_mask,:,:,cc].std().item())
print("STE null mean, SD:", results['Z_STE_robs'][noise_mask,:,:,cc].mean().item(),
                            results['Z_STE_robs'][noise_mask,:,:,cc].std().item())



plt.subplot(1,3,1)
if results['model_type'] == 'LN':
    plt.imshow(results['models'][cc])
else:
    plt.imshow(results['models'][cc][0] * results['models'][cc][1])

plt.subplot(1,3,2)
plt.imshow(results['Z_STA_robs'][1,:,:,cc])
plt.subplot(1,3,3)
plt.imshow(results['Z_STE_robs'][1,:,:,cc])
# %%

sd_sta = results['Z_STA_robs'][0,:,:,cc].std().item()
sd_ste = results['Z_STE_robs'][0,:,:,cc].std().item()
print(sd_sta, sd_ste)

snr_sta = results['Z_STA_robs'][1,:,:,cc].abs().max().item() / sd_sta
snr_ste = results['Z_STE_robs'][1,:,:,cc].abs().max().item() / sd_ste

print(snr_sta, snr_ste)

index = (snr_ste - snr_sta) / (snr_ste + snr_sta)
print(index)
# %%
