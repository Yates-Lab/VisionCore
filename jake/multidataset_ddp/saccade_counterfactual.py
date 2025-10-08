#!/usr/bin/env python3
"""
Clean saccade counterfactual analysis script.

Extracts saccade-aligned predictions for different prediction types and stimulus conditions.
"""

#%%
import sys
sys.path.append('.')

import numpy as np
import torch
from tqdm import tqdm

from DataYatesV1 import get_session
from eval_stack_utils import detect_saccades_from_session, get_stim_inds
from eval_stack_multidataset import load_model, load_single_dataset


#%%
def model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=False):
    """
    Get model predictions at different stages.
    
    Parameters
    ----------
    batch : dict
        Batch of data with 'stim', 'behavior', etc.
    model : torch model
        Loaded model
    dataset_idx : int
        Dataset index
    stage : str
        Stage to extract ('pred', 'conv.0', etc.)
    include_modulator : bool
        Whether to include modulator
    zero_modulator : bool
        Whether to zero out modulator input
        
    Returns
    -------
    torch.Tensor
        Model output at specified stage
    """
    if include_modulator:
        behavior = batch.get('behavior')
        if zero_modulator:
            behavior = torch.zeros_like(behavior)
    else:
        behavior = None

    # Handle hooks for intermediate stages
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    if stage == 'conv.0.conv':
        main_ref = model.model.convnet.layers[0].main_block.components.conv
        handle = main_ref.register_forward_hook(make_hook("layer"))
        _ = model.model(batch['stim'], dataset_idx, behavior)
        handle.remove()
        return activations['layer']
    
    elif stage == 'conv.0.norm':
        main_ref = model.model.convnet.layers[0].main_block.components.norm
        handle = main_ref.register_forward_hook(make_hook("layer"))
        _ = model.model(batch['stim'], dataset_idx, behavior)
        handle.remove()
        return activations['layer']
    
    elif stage == 'conv.0':
        main_ref = model.model.convnet.layers[0].main_block
        handle = main_ref.register_forward_hook(make_hook("layer"))
        _ = model.model(batch['stim'], dataset_idx, behavior)
        handle.remove()
        return activations['layer']

    elif stage == 'conv.1':
        main_ref = model.model.convnet.layers[1].main_block
        handle = main_ref.register_forward_hook(make_hook("layer"))
        _ = model.model(batch['stim'], dataset_idx, behavior)
        handle.remove()
        return activations['layer']

    elif stage == 'conv.2':
        main_ref = model.model.convnet.layers[2].main_block
        handle = main_ref.register_forward_hook(make_hook("layer"))
        _ = model.model(batch['stim'], dataset_idx, behavior)
        handle.remove()
        return activations['layer']
    
    elif stage == 'modulator.encoder':
        main_ref = model.model.modulator.encoder
        handle = main_ref.register_forward_hook(make_hook("encoder"))
        _ = model.model(batch['stim'], dataset_idx, behavior)
        handle.remove()
        return activations['encoder']
    
    elif stage == 'pred':
        output = model.model(batch['stim'], dataset_idx, behavior)
        if model.log_input:
            output = torch.exp(output)
        return output
    
    # For other stages, use the standard pipeline
    x = model.model.adapters[dataset_idx](batch['stim'])
    if stage == 'adapter':
        return x
    x = model.model.frontend(x)
    if stage == 'frontend':
        return x
    x = model.model.convnet(x)
    if stage == 'convnet':
        return x
    
    if include_modulator and model.model.modulator is not None:
        x = model.model.modulator(x, behavior)
    else:
        print('not using modulator')

    if stage == 'modulator':
        return x
    
    if stage == 'readout':
        if 'DynamicGaussianReadoutEI' in str(type(model.model.readouts[dataset_idx])):
            x = x[:, :, -1, :, :]  # (N, C_in, H, W)
            N, C_in, H, W = x.shape
            device = x.device

            readout = model.model.readouts[dataset_idx]
            # Apply positive-constrained feature weights using functional conv2d
            feat_ex = torch.nn.functional.conv2d(x, readout.features_ex_weight, bias=None)  # (N, n_units, H, W)
            feat_inh = torch.nn.functional.conv2d(x, readout.features_inh_weight, bias=None)  # (N, n_units, H, W)

            # Compute Gaussian masks for both pathways
            gaussian_mask_ex = readout.compute_gaussian_mask(H, W, device, pathway='ex')  # (n_units, H, W)
            gaussian_mask_inh = readout.compute_gaussian_mask(H, W, device, pathway='inh')  # (n_units, H, W)

            # Apply masks and sum over spatial dimensions
            out_ex = (feat_ex * gaussian_mask_ex.unsqueeze(0)).sum(dim=(-2, -1))  # (N, n_units)
            out_inh = (feat_inh * gaussian_mask_inh.unsqueeze(0)).sum(dim=(-2, -1))  # (N, n_units)
            return out_ex, out_inh        
        else:
            x = model.model.readouts[dataset_idx](x)
            return x
    else:
        x = model.model.readouts[dataset_idx](x)
        return x


def extract_saccade_predictions(model, dataset_idx, stim_type, prediction_type='pred', 
                              win=(-50, 100), device='cuda:0'):
    """
    Extract saccade-aligned predictions for a specific dataset and stimulus type.
    
    Parameters
    ----------
    model : torch model
        Loaded model
    dataset_idx : int
        Dataset index (0-based)
    stim_type : str
        Stimulus type ('backimage', 'gaborium', 'fixrsvp', 'gratings')
    prediction_type : str
        Type of prediction ('pred', 'pred.zero', or other model stages)
    win : tuple
        Time window around saccades (start, end) in bins
    device : str
        Device to use for computation
        
    Returns
    -------
    dict
        Dictionary containing:
        - pred_sac: saccade-aligned predictions [n_saccades, n_bins, n_units]
        - robs_sac: saccade-aligned observations [n_saccades, n_bins, n_units] 
        - dfs_sac: saccade-aligned data flags [n_saccades, n_bins, n_units]
        - saccade_info: list of saccade dictionaries
        - valid_indices: indices of valid saccades
        - win: time window used
        - dt: sampling rate (1/120)
    """
    print(f"🔄 Extracting saccade predictions for dataset {dataset_idx}, stim_type '{stim_type}', prediction_type '{prediction_type}'")
    
    # Load dataset
    train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
    
    # Get session and saccades
    dataset_name = model.names[dataset_idx]
    sess = get_session(*dataset_name.split('_'))
    saccades = detect_saccades_from_session(sess)
    
    # Define prediction function based on type
    if prediction_type == 'pred.zero':
        def net(batch):
            return model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=True)
    elif prediction_type == 'pred':
        def net(batch):
            return model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=False)
    else:
        def net(batch):
            out = model_pred(batch, model, dataset_idx, stage=prediction_type)
            sz = out.shape
            return out[:,:,-1, sz[-2]//2, sz[-1]//2]
    
    # Get saccade times and convert to dataset indices
    saccade_times = torch.tensor([s['start_time'] for s in saccades])
    saccade_inds = train_data.get_inds_from_times(saccade_times)
    
    # Get stimulus indices
    stim_inds = get_stim_inds(stim_type, train_data, val_data)
    
    # Create dataset copy with stimulus indices
    dataset = val_data.shallow_copy()
    dataset.inds = stim_inds
    
    dset = stim_inds[0,0]
    print(f'Dataset {dset}')
    
    # Setup arrays
    nbins = win[1] - win[0]
    valid_saccades = np.where(saccade_inds[:,0] == dset)[0]
    sac_indices = saccade_inds[valid_saccades, 1]
    n_sac = len(sac_indices)
    
    print(f"Found {n_sac} saccades in dataset {dset}")
    
    # Get a sample batch to determine dimensions
    sample_batch = dataset[0:1]
    sample_batch = {k: v.to(device) for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}
    sample_pred = net(sample_batch)
    
    n_cells = sample_batch['robs'].shape[1]
    n_pred_units = sample_pred.shape[1]
    
    # Initialize arrays
    robs_sac = np.nan * np.zeros((n_sac, nbins, n_cells))
    pred_sac = np.nan * np.zeros((n_sac, nbins, n_pred_units))
    dfs_sac = np.nan * np.zeros((n_sac, nbins, n_cells))
    
    saccade_info = [saccades[i] for i in valid_saccades]
    
    # Extract saccade-aligned data
    for i, isac in enumerate(tqdm(sac_indices, desc="Processing saccades")):
        # Find matching stimulus index
        j = np.where(stim_inds[:,1] == isac)[0]
        
        if len(j) == 0:
            continue
            
        j = j.item()
        
        # Find dataset index
        dataset_indices = np.where(torch.all(dataset.inds == stim_inds[j], 1))[0]
        if len(dataset_indices) == 0:
            continue
            
        dataset_indices = dataset_indices.item()
        
        # Extract window around saccade
        if (j + win[0] >= 0):
            batch = dataset[dataset_indices + win[0]:dataset_indices + win[1]]
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            if batch['robs'].shape[0] != nbins:
                continue
                
            pred = net(batch)
            
            # Store data
            robs_sac[i] = batch['robs'].detach().cpu().numpy()
            pred_sac[i] = pred.detach().cpu().numpy()
            dfs_sac[i] = batch['dfs'].detach().cpu().numpy()
    
    # Clean up
    del sample_pred, sample_batch
    if 'pred' in locals():
        del pred
    torch.cuda.empty_cache()
    
    # Find valid saccades (no NaNs)
    good = np.where(np.sum(np.isnan(robs_sac), axis=(1,2)) == 0)[0]
    
    print(f"✓ Extracted {len(good)} valid saccades out of {n_sac} total")
    
    return {
        'pred_sac': pred_sac[good],
        'robs_sac': robs_sac[good],
        'dfs_sac': dfs_sac[good],
        'saccade_info': [saccade_info[i] for i in good],
        'valid_indices': good,
        'win': win,
        'dt': 1/120,
        'dataset_idx': dataset_idx,
        'stim_type': stim_type,
        'prediction_type': prediction_type
    }

#%%


# Example usage
from eval_stack_utils import scan_checkpoints

# Load a model
checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'
models_by_type = scan_checkpoints(checkpoint_dir)

model_type = 'learned_res_small_gru'
model, model_info = load_model(
    model_type=model_type,
    model_index=None,  # None for best model
    checkpoint_path=None,
    checkpoint_dir=checkpoint_dir,
    device='cpu'
)

model.model.eval()
model.model.convnet.use_checkpointing = False
model = model.to('cuda:0')


#%%
# Extract saccade predictions
results = extract_saccade_predictions(
    model=model,
    dataset_idx=8,
    stim_type='backimage',
    prediction_type='pred.zero',
    win=(-50, 100),
    device='cuda:0'
)

print(f"Results shape - pred_sac: {results['pred_sac'].shape}")
print(f"Results shape - robs_sac: {results['robs_sac'].shape}")
print(f"Results shape - dfs_sac: {results['dfs_sac'].shape}")

# Example: Compare pred vs pred.zero
results_pred = extract_saccade_predictions(
    model=model,
    dataset_idx=8,
    stim_type='gratings',
    prediction_type='pred',
    win=(-50, 100),
    device='cuda:0'
)

print(f"\nComparison:")
print(f"pred.zero: {results['pred_sac'].shape}")
print(f"pred: {results_pred['pred_sac'].shape}")

    # You can now analyze the difference between modulated and unmodulated predictions
    # pred_diff = results_pred['pred_sac'] - results['pred_sac']

# %%
import matplotlib.pyplot as plt
Rdelta = results_pred['pred_sac'] - results['pred_sac'] 

cid = 1
plt.imshow(Rdelta[:,:,cid], aspect='auto', interpolation='none', cmap='coolwarm')
plt.colorbar()
# %%
cid +=1

ind = np.argsort(Rdelta[:,40:70,cid].sum(1))
plt.figure(figsize=(10,5))
plt.subplot(1,4,1)
plt.imshow(results['robs_sac'][ind,:,cid], aspect='auto', interpolation='none', cmap='coolwarm')
plt.subplot(1,4,2)
plt.imshow(results_pred['pred_sac'][ind,:,cid], aspect='auto', interpolation='none', cmap='coolwarm')
plt.subplot(1,4,3)
plt.imshow(results['pred_sac'][ind,:,cid], aspect='auto', interpolation='none', cmap='coolwarm')
plt.subplot(1,4,4)
plt.imshow(Rdelta[ind,:,cid], aspect='auto', interpolation='none', cmap='coolwarm')
# _ = plt.plot(Rdelta[:,40:70,cid].T)

# %%
# Rdelta[:,40:70,:].sum(1)

plt.imshow(np.mean(Rdelta, 0).T, extent=[-50, 100, 0, 100])
plt.xlim(-10, 30)
# %%
