

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.data import prepare_data
from .eval_stack_utils import get_stim_inds
from pathlib import Path
from .eval_stack_utils import argmin_subpixel, argmax_subpixel

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

def get_spike_triggered_sums(dataset_config_or_data, robs, rhat, lags=[-2] + list(range(16)), fixations_only=True, combine_train_test=False, whiten=True, device=None):

    if device is None:
        from DataYatesV1.utils.torch import get_free_device
        device = get_free_device()

    if isinstance(dataset_config_or_data, dict):
        # It's a config - modify and load the dataset
        dataset_config = dataset_config_or_data.copy()
        dataset_config['transforms']['stim'] = {'source': 'stim',
            'ops': [{'pixelnorm': {}}],
            'expose_as': 'stim'}

        dataset_config['keys_lags']['stim'] = list(range(25))
        dataset_config['types'] = ['gaborium']

        train_data, val_data, dataset_config = prepare_data(dataset_config)
    else:
        # It's pre-loaded data - extract train_data, val_data, dataset_config
        train_data, val_data, dataset_config = dataset_config_or_data
    
    if combine_train_test:
        stim_indices = torch.concatenate([train_data.get_dataset_inds('gaborium'), val_data.get_dataset_inds('gaborium')], dim=0)
    else:
        stim_indices = get_stim_inds( 'gaborium', train_data, val_data)

    # shallow copy the dataset not to mess it up
    data = val_data.shallow_copy()
    data.inds = stim_indices

    dset_idx = np.unique(stim_indices[:,0]).item()
    print(f"  dset_idx: {dset_idx}")
    print(f"  stim_indices: {len(stim_indices)}")
    print(f"  stim shape: {data.dsets[dset_idx]['stim'].shape}")

    if whiten:
        mu_hw, Wx_flat, V, evals, ridge, mu2_hw = estimate_spatial_whitener(data.dsets[dset_idx]['stim'].to(device), data.dsets[dset_idx]['dfs'].to(device))
    else:
        mu2_hw = np.nan

    # confirm inds match
    assert torch.all(robs == data.dsets[dset_idx]['robs'][data.inds[:,1]]), 'robs mismatch'
    dfs = data.dsets[dset_idx]['dfs'][data.inds[:,1]]
    if fixations_only:
        dt = 1/dataset_config['sampling']['target_rate']
        # TODO: make sure all lags are on fixation
        eyedt = torch.gradient(data.dsets[dset_idx]['eyepos'][data.inds[:,1]], dim=0)[0]
        eyevel = torch.hypot(eyedt[:,0], eyedt[:,1])/dt
        dfs = dfs * (eyevel[:,None] < 5)

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
    
    
    robs = robs.to(device)
    rhat = rhat.to(device)
    dfs = dfs.to(device)

    for lag in tqdm(lags):
        stim = data.dsets[dset_idx]['stim'][data.inds[:,1]-lag].to(device)

        if whiten:
            stim = whiten_frames(stim, mu_hw, Wx_flat)
        stsum_stim_robs[lag] = torch.einsum('thw, tc->hwc', stim, robs*dfs).cpu()
        stsum_energy_robs[lag] = torch.einsum('thw, tc->hwc', stim.pow(2), robs*dfs).cpu()
        stsum_stim_rhat[lag] = torch.einsum('thw, tc->hwc', stim, rhat*dfs).cpu()
        stsum_energy_rhat[lag] = torch.einsum('thw, tc->hwc', stim.pow(2), rhat*dfs).cpu()
    
    
    mu2_hw = mu2_hw.cpu()
    dfs = dfs.cpu()

    return {'stsum_stim_robs': stsum_stim_robs, 'stsum_energy_robs': stsum_energy_robs, 'stsum_stim_rhat': stsum_stim_rhat, 'stsum_energy_rhat': stsum_energy_rhat, 'norm_dfs': norm_dfs, 'norm_robs': norm_robs, 'norm_rhat': norm_rhat, 'mu2_hw': mu2_hw}, dfs

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

def sta_at_peak_lag(sta, peak_lag):
    return np.stack([sta[peak_lag[cc],:,:,cc] for cc in range(len(peak_lag))], 0)

def get_sta_ste(dataset_config_or_data, robs, rhat, lags=[-2] + list(range(16)), fixations_only=True, combine_train_test=False, whiten=True, device=None):

    stsums, dfs = get_spike_triggered_sums(dataset_config_or_data, robs, rhat, lags, fixations_only, combine_train_test, whiten, device)
    
    mu2_hw = stsums['mu2_hw']
    Z_STA_robs, Z_STE_robs, Z_STA_rhat, Z_STE_rhat = sta_ste_zmaps_from_whitened_sums(stsums, mu2_hw, robs, dfs, rhat)

    results = {
        'Z_STA_robs': Z_STA_robs,
        'Z_STE_robs': Z_STE_robs,
        'Z_STA_rhat': Z_STA_rhat,
        'Z_STE_rhat': Z_STE_rhat,
    }

    sd_sta_robs = results['Z_STA_robs'][0].std((0,1))
    sd_ste_robs = results['Z_STE_robs'][0].std((0,1))

    sd_sta_rhat = results['Z_STA_rhat'][0].std((0,1)) if rhat is not None else None
    sd_ste_rhat = results['Z_STE_rhat'][0].std((0,1)) if rhat is not None else None

    peak_lag = np.array([results['Z_STE_robs'][:,:,:,cc].std((1,2)).argmax() for cc in range(robs.shape[1])])
    snr_sta_robs = np.array([results['Z_STA_robs'][peak_lag[cc],:,:,cc].abs().max().item() / sd_sta_robs[cc] for cc in range(robs.shape[1])])
    snr_ste_robs = np.array([results['Z_STE_robs'][peak_lag[cc],:,:,cc].abs().max().item() / sd_ste_robs[cc] for cc in range(robs.shape[1])])

    snr_sta_rhat = np.array([results['Z_STA_rhat'][peak_lag[cc],:,:,cc].abs().max().item() / sd_sta_rhat[cc] for cc in range(robs.shape[1])]) if rhat is not None else None
    snr_ste_rhat = np.array([results['Z_STE_rhat'][peak_lag[cc],:,:,cc].abs().max().item() / sd_ste_rhat[cc] for cc in range(robs.shape[1])]) if rhat is not None else None

    # get sub-pixel peak lag
    lag_robs, _ = argmax_subpixel(Z_STE_robs.var((1,2)), 0)
    lag_rhat, _ = argmax_subpixel(Z_STE_rhat.var((1,2)), 0) if rhat is not None else None

    # include contour metrics
    from DataYatesV1.utils.rf import get_contour_metrics

    # for robs
    sta = sta_at_peak_lag(results['Z_STA_robs'], peak_lag)
    ste = sta_at_peak_lag(results['Z_STE_robs'], peak_lag)

    contour_metrics = get_contour_metrics(sta, ste, sort=False, return_snr_list=True)

    snr_value_robs = np.array([x[0] for x in contour_metrics])
    contour_robs = [x[2] for x in contour_metrics]
    area_robs = np.array([x[3] for x in contour_metrics])
    center_robs = np.stack([x[4] for x in contour_metrics], 0)

    # for rhat
    sta = sta_at_peak_lag(results['Z_STA_rhat'], peak_lag)
    ste = sta_at_peak_lag(results['Z_STE_rhat'], peak_lag)
    contour_metrics = get_contour_metrics(sta, ste, sort=False, return_snr_list=True)
    snr_value_rhat = np.array([x[0] for x in contour_metrics])
    contour_rhat = [x[2] for x in contour_metrics]
    area_rhat = np.array([x[3] for x in contour_metrics])
    center_rhat = np.stack([x[4] for x in contour_metrics], 0)

    # add to results
    results['snr_sta_robs'] = snr_sta_robs
    results['snr_ste_robs'] = snr_ste_robs
    results['modulation_index_robs'] = (snr_ste_robs - snr_sta_robs) / (snr_ste_robs + snr_sta_robs)
    results['peak_lag'] = peak_lag
    results['snr_sta_rhat'] = snr_sta_rhat
    results['snr_ste_rhat'] = snr_ste_rhat
    results['modulation_index_rhat'] = (snr_ste_rhat - snr_sta_rhat) / (snr_ste_rhat + snr_sta_rhat) if rhat is not None else None
    results['peak_lag_subpixel_robs'] = lag_robs
    results['peak_lag_subpixel_rhat'] = lag_rhat

    # contour metrics
    results['snr_contour_robs'] = snr_value_robs
    results['contour_robs'] = contour_robs
    results['area_robs'] = area_robs
    results['center_robs'] = center_robs

    results['snr_contour_rhat'] = snr_value_rhat
    results['contour_rhat'] = contour_rhat
    results['area_rhat'] = area_rhat
    results['center_rhat'] = center_rhat

    return results


def run_gaborium_analysis(all_results, checkpoint_dir, save_dir, recalc=False, batch_size=64, device='cuda', test_mode=False):
    """
    Run comprehensive gaborium analysis for all models and datasets.

    Get the STA / STE / modulation index and latency

    Parameters
    ----------
    all_results : dict
        Existing results dictionary from BPS/CCNORM analysis (must include 'bps' results)
    checkpoint_dir : str
        Directory containing model checkpoints
    save_dir : str
        Directory to save gaborium analysis caches
    recalc : bool, optional
        Whether to recalculate cached results (default: False)
    batch_size : int, optional
        Batch size for evaluation (default: 64)
    device : str, optional
        Device to run evaluation on (default: 'cuda')

    Returns
    -------
    dict
        Modified all_results dictionary with comprehensive sta analysis added:
        - all_results[model_type]['sta']['comparison_results'][dataset_idx]: gratings_comparison results
        - all_results[model_type]['sta']['modulation_indices']: extracted modulation indices
        - all_results[model_type]['sta']['datasets']: dataset names
    """

    # Import evaluation utilities
    from eval_stack_multidataset import load_model, load_single_dataset, evaluate_dataset

    print("Running gaborium analysis...")

    # Extract model types from existing results
    model_types = list(all_results.keys())
    print(f"Found {len(model_types)} models: {model_types}")

    # Verify that BPS results exist (required for gratings analysis)
    for model_type in model_types:
        if 'bps' not in all_results[model_type] or 'gaborium' not in all_results[model_type]['bps']:
            raise ValueError(f"Model {model_type} missing BPS gaborium results. Run BPS analysis first.")

    # Load all models once (on CPU to save GPU memory)
    models = {}
    for model_type in model_types:
        print(f"  Loading {model_type}...")
        model, model_info = load_model(
            model_type=model_type,
            checkpoint_dir=checkpoint_dir,
            device='cpu'  # Load on CPU first
        )
        models[model_type] = {
            'model': model,
            'model_info': model_info,
            'experiment': model_info['experiment']
        }

    # Get number of datasets from first model
    first_model = list(models.values())[0]['model']
    num_datasets = len(first_model.names)

    # Initialize gratings results structure for all models
    # Structure matches other analyses: [analysis_name][dataset_idx] = padded_array
    gaborium_analyses = ['Z_STA_robs', 'Z_STE_robs', 'Z_STA_rhat', 'Z_STE_rhat',
                        'snr_sta_robs', 'snr_ste_robs', 'modulation_index_robs', 'peak_lag',
                        'snr_sta_rhat', 'snr_ste_rhat', 'modulation_index_rhat',
                        'peak_lag_subpixel_robs', 'peak_lag_subpixel_rhat',
                        'snr_contour_robs', 'contour_robs', 'area_robs', 'center_robs',
                        'snr_contour_rhat', 'contour_rhat', 'area_rhat', 'center_rhat']

    for model_type in model_types:
        if 'sta' not in all_results[model_type]:
            all_results[model_type]['sta'] = {}

        # Initialize lists for each analysis (both robs and rhat)
        for analysis in gaborium_analyses:
            all_results[model_type]['sta'][analysis] = []

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loop over datasets to find those with gratings stimuli
    for dataset_idx in tqdm(range(num_datasets), desc="Processing datasets"):

        # Check if cache exists for all models BEFORE loading the data
        all_caches_exist = True
        for model_type in model_types:
            # Check cache first
            model_ = models[model_type]['model']
            dataset_config = model_.model.dataset_configs[dataset_idx].copy()
            session = dataset_config['session']
            cache_file = save_dir / f'{model_type}_dataset{dataset_idx}_{session}_gaborium_analysis_cache.pt'
            if not cache_file.exists():
                all_caches_exist = False
                break
        
        if test_mode and dataset_idx > 1:
            break
        
        dataset_name = first_model.names[dataset_idx]
        print(f"\nProcessing dataset {dataset_idx}: {dataset_name}")

        # Save to cache
                    # torch.save(analysis_result, cache_file)
                    # print(f"    Gratings analysis cache saved to {cache_file}")
        if not all_caches_exist or recalc:
            # Load dataset to check for gratings stimuli
            train_data, val_data, dataset_config = load_single_dataset(first_model, dataset_idx)
            
        
            # Get dataset CIDs for proper padding
            dataset_cids = dataset_config.get('cids', [])
            n_total_units = len(dataset_cids)
            print(f"Dataset has {n_total_units} total units")

            # Combine all indices (train + validation) for maximum data (we don't tend to train on gratings so this should be okay)
            gaborium_inds = torch.concatenate([
                train_data.get_dataset_inds('gaborium'),
                val_data.get_dataset_inds('gaborium')
            ], dim=0)

            dataset = train_data.shallow_copy()
            dataset.inds = gaborium_inds

            # Now load train / val data for the STA analysis (has overloaded config)
            dataset_config['transforms']['stim'] = {'source': 'stim',
                'ops': [{'pixelnorm': {}}],
                'expose_as': 'stim'}

            dataset_config['keys_lags']['stim'] = list(range(25))
            dataset_config['types'] = ['gaborium']

            train_data, val_data, dataset_config = prepare_data(dataset_config)
               
        # Loop over models (inner loop - data already loaded or failed)
        for model_type in model_types:

            cache_file = save_dir / f'{model_type}_dataset{dataset_idx}_{session}_gaborium_analysis_cache.pt'
            if cache_file.exists():
                print("Loading from cache...")
                analysis_result = torch.load(cache_file, weights_only=False)
                print(analysis_result['Z_STA_robs'].shape)
            
            else:
                
                print(f"  Running gaborium analysis for {model_type}...")

                # Check cache first
                session = dataset_config['session']
                cache_file = save_dir / f'{model_type}_dataset{dataset_idx}_{session}_gaborium_analysis_cache.pt'

                if not recalc and cache_file.exists():
                    print(f"    Loading gaborium analysis cache from {cache_file}")
                    analysis_result = torch.load(cache_file, weights_only=False)
                else:
                    # Get model and move to device for evaluation
                    model = models[model_type]['model'].to(device)

                    eval_result = evaluate_dataset(
                        model, dataset, gaborium_inds, dataset_idx, batch_size, 'Gaborium'
                    )
                    
                    # run STA / STE analysis
                    analysis_result = get_sta_ste((train_data, val_data, dataset_config), eval_result['robs'], eval_result['rhat'], lags=list(range(16)), fixations_only=False, combine_train_test=True, whiten=True, device=model.device)

                    # Move model back to CPU to save GPU memory
                    models[model_type]['model'] = model.to('cpu')

                    # Save to cache
                    torch.save(analysis_result, cache_file)
                    print(f"    Gaborium analysis cache saved to {cache_file}")

                # # TODO: potentially do error handling and padd with dummy data. currently all datasaet have gaborium, so this won't change
                # n_units = len(train_data.dsets[0].metadata['cids'])
                # dummy_analysis_result = {'n_units': n_units,
                #     'Z_STA_robs': np.full([16, 51, 51, n_units], np.nan),
                #     'Z_STE_robs': np.full([16, 51, 51, n_units], np.nan),
                #     'Z_STA_rhat': np.full([16, 51, 51, n_units], np.nan),
                #     'Z_STE_rhat': np.full([16, 51, 51, n_units], np.nan),
                #     'snr_sta_robs': np.full((n_units,), np.nan),
                #     'snr_ste_robs': np.full((n_units,), np.nan),
                #     'modulation_index_robs': np.full((n_units,), np.nan),
                #     'peak_lag': np.full((n_units,), np.nan),
                #     'snr_sta_rhat': np.full((n_units,), np.nan),
                #     'snr_ste_rhat': np.full((n_units,), np.nan),
                #     'modulation_index_rhat': np.full((n_units,), np.nan),
                #     'peak_lag_subpixel_robs': np.full((n_units,), np.nan),
                #     'peak_lag_subpixel_rhat': np.full((n_units,), np.nan),
                #     'snr_contour_robs': np.full((n_units,), np.nan),
                #     'contour_robs': [None] * n_units,
                #     'area_robs': np.full((n_units,), np.nan),
                #     'center_robs': np.full((n_units, 2), np.nan),
                #     'snr_contour_rhat': np.full((n_units,), np.nan),
                #     'contour_rhat': [None] * n_units,
                #     'area_rhat': np.full((n_units,), np.nan),
                #     'center_rhat': np.full((n_units, 2), np.nan)}

                    # save to cache to avoid having to load the whole dataset again
                    torch.save(analysis_result, cache_file)

        
            for analysis in gaborium_analyses:
                all_results[model_type]['sta'][analysis].append(analysis_result[analysis])

    return all_results

def plot_stas(sta, lag = None, normalize=True, sort_by=None):

    n_cells = sta['Z_STA_robs'].shape[-1]
    sx = np.floor(np.sqrt(n_cells)).astype(int)
    sy = np.ceil(n_cells / sx).astype(int)
    fig, axs = plt.subplots(sy, sx, figsize=(16, 16))
    
    
    # H = sta['Z_STA_robs'].shape[1]

    if sort_by is not None:
        if sort_by == 'modulation_index':
            order = np.argsort(sta['modulation_index_robs'])
            order = order[:n_cells]
        elif sort_by == 'modulation_index_rhat':
            order = np.argsort(sta['modulation_index_rhat'])
            order = order[:n_cells]
        else:
            order = np.arange(n_cells)
    else:
        order = np.arange(n_cells)
    
    for i, cc in enumerate(order):
        if lag is None:
            this_lag = sta['peak_lag'][cc]
        else:
            this_lag = lag

        ax = axs.flatten()[i]
        v = sta['Z_STA_robs'][this_lag,:,:,cc].abs().max()
        Irhat = sta['Z_STA_rhat'][this_lag,:,:,cc]
        if normalize:
            vrhat = Irhat.abs().max()
            Irhat = Irhat / vrhat * v
            
        I = torch.concat([sta['Z_STA_robs'][this_lag,:,:,cc], torch.ones(H,1), Irhat], 1)
        
        ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)
        ax.set_title(f'{cc}: {sta['modulation_index_robs'][cc]:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()