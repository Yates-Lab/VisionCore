#!/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import torch._dynamo 
torch._dynamo.config.suppress_errors = True # suppress dynamo errors

import sys
sys.path.append('.')

import numpy as np
from DataYatesV1 import enable_autoreload, get_free_device, prepare_data

import matplotlib.pyplot as plt
from eval_stack_utils import load_single_dataset, get_stim_inds, evaluate_dataset
from eval_stack_multidataset import load_model, load_single_dataset


enable_autoreload()

device = get_free_device()


#%% Discover Available Models
print("🔍 Discovering available models...")
from eval_stack_utils import scan_checkpoints
checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'
models_by_type = scan_checkpoints(checkpoint_dir)

print(f"Found {len(models_by_type)} model types:")
for model_type, models in models_by_type.items():
    best_loss = models[0]['val_loss'] if models else 'N/A'
    print(f"  {model_type}: {len(models)} models (best loss: {best_loss:.4f})")

#%%#%% LOAD A MODEL
model_type = 'learned_res_small_none_gru_none_pool'
models_by_type = scan_checkpoints(checkpoint_dir)


model, model_info = load_model(
        model_type=model_type,
        model_index=None, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.model.eval()
model.model.convnet.use_checkpointing = False 

# Ensure the model is not using torch.compile
if hasattr(model.model, "_orig_mod"):
    # If the model was compiled with torch.compile(), access the original module
    orig_model = model.model._orig_mod
    # Temporarily replace the compiled model with the original
    temp_model = model.model
    model.model = orig_model
else:
    temp_model = None

model = model.to(device)
model_name = model_info['experiment']

#%%
dataset_idx = 0
batch_size = 64 # keep small because things blow up fast!

train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
dataset_cids = dataset_config.get('cids', [])

#%%
from eval_stack_multidataset import run_bps_analysis, run_ccnorm_analysis, run_saccade_analysis
from pathlib import Path
import os
save_dir = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack_120") / model_name
os.makedirs(save_dir, exist_ok=True)

# Get CIDs for this dataset
dataset_cids = dataset_config.get('cids', [])
            
bps_results = run_bps_analysis(
    model, train_data, val_data, dataset_idx, model_name,
    save_dir, False, batch_size
)

#%%
ccnorm_results = run_ccnorm_analysis(
                    model, train_data, val_data, dataset_idx, bps_results,
                    model_name, save_dir, False
                )

sac_win = (-50, 100)
                # return model, train_data, val_data, dataset_idx, bps_results, model_name, save_dir, recalc
saccade_results = run_saccade_analysis(
                    model, train_data, val_data, dataset_idx, bps_results,
                    model_name, save_dir, False, sac_win
                )

#%%
from DataYatesV1 import get_session, DictDataset
sess = get_session(*dataset_config['session'].split('_'))


#%% Utilities for evaluation
from tqdm import tqdm

def model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True):

    if stage=='pred':

        behavior = batch.get('behavior')
        if model.model.modulator is not None:
            if not include_modulator:
                behavior = torch.zeros_like(batch.get('behavior'))
        else:
            behavior = None
        
        output = model.model(batch['stim'], dataset_idx, behavior)

        if model.log_input:
            output = torch.exp(output)
        return output
    
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
        x = model.model.modulator(x, batch.get('behavior'))

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
    
def get_sta_ste(model, robs, rhat, didx=0, lags=list(range(16))):

    # overwrite stimulus transforms
    dataset_config = model.model.dataset_configs[didx].copy()
    dataset_config['transforms']['stim'] = {'source': 'stim',
        'ops': [{'pixelnorm': {}}],
        'expose_as': 'stim'}

    dataset_config['keys_lags']['stim'] = list(range(25))
    dataset_config['types'] = ['gaborium']
    
    train_data, val_data, dataset_config = prepare_data(dataset_config)
    stim_indices = get_stim_inds( 'gaborium', train_data, val_data)

    # shallow copy the dataset not to mess it up
    data = val_data.shallow_copy()
    data.inds = stim_indices

    dset_idx = np.unique(stim_indices[:,0]).item()

    # confirm inds match
    assert torch.all(robs == data.dsets[dset_idx]['robs'][data.inds[:,1]]), 'robs mismatch'
    dfs = data.dsets[dset_idx]['dfs'][data.inds[:,1]]
    norm_dfs = dfs.sum(0) # if forward
    norm_robs = (robs * dfs).sum(0) # if reverse
    norm_rhat = (rhat * dfs).sum(0) # if reverse

    n_cells = robs.shape[1]
    n_lags = len(lags)
    H, W  = data.dsets[dset_idx]['stim'].shape[1:3]
    sta_robs = torch.zeros((n_lags, H, W, n_cells))
    ste_robs = torch.zeros((n_lags, H, W, n_cells))
    sta_rhat = torch.zeros((n_lags, H, W, n_cells))
    ste_rhat = torch.zeros((n_lags, H, W, n_cells))
    for lag in tqdm(lags):
        stim = data.dsets[dset_idx]['stim'][data.inds[:,1]-lag]    
        sta_robs[lag] = torch.einsum('thw, tc->hwc', stim, robs*dfs)
        ste_robs[lag] = torch.einsum('thw, tc->hwc', stim.pow(2), robs*dfs)
        sta_rhat[lag] = torch.einsum('thw, tc->hwc', stim, rhat*dfs)
        ste_rhat[lag] = torch.einsum('thw, tc->hwc', stim.pow(2), rhat*dfs)
    
    return {'sta_robs': sta_robs, 'ste_robs': ste_robs, 'sta_rhat': sta_rhat, 'ste_rhat': ste_rhat, 'norm_dfs': norm_dfs, 'norm_robs': norm_robs, 'norm_rhat': norm_rhat}

def plot_stas(sta):
    n_cells = sta['sta_robs'].shape[-1]
    sx = np.floor(np.sqrt(n_cells)).astype(int)
    sy = np.ceil(n_cells / sx).astype(int)
    fig, axs = plt.subplots(sy, sx, figsize=(16, 16))
    lag = 8
    sta_robs = sta['sta_robs'] / sta['norm_dfs'][None,None,None,:]
    sta_rhat = sta['sta_rhat'] / sta['norm_dfs'][None,None,None,:]
    H = sta_robs.shape[1]

    for i in range(n_cells):
        ax = axs.flatten()[i]
        v = sta_robs[lag,:,:,i].abs().max()
        I = torch.concat([sta_robs[lag,:,:,i], torch.ones(H,1), sta_rhat[lag,:,:,i]], 1)
        
        ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)
        ax.set_title(f'Cell {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()



#%%


#%%

dataset_idx = 0
batch_size = 64 # keep small because things blow up fast!

train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
dataset_cids = dataset_config.get('cids', [])

#%%
from DataYatesV1.utils.data.transforms import make_pipeline
pipeline = make_pipeline(dataset_config['transforms']['eye_vel']['ops'])

#%%

stim = train_data.dsets[0]['stim'][::2]

#%% Visualize raw and downsampled stimulus frames
# Pick a starter frame
starter_frame = 100  # You can change this to any frame index

# Get the raw stimulus (shape: N x 1 x 51 x 51)
raw_stim = train_data.dsets[0]['stim']
print(f"Raw stimulus shape: {raw_stim.shape}")

# Create downsampled stimulus (every 2nd frame)
downsampled_stim = raw_stim[::2]
print(f"Downsampled stimulus shape: {downsampled_stim.shape}")

# Display next 10 frames of raw stimulus
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle(f'Raw Stimulus - Next 10 frames starting from frame {starter_frame}')

for i in range(10):
    row = i // 5
    col = i % 5
    frame_idx = starter_frame + i

    if frame_idx < raw_stim.shape[0]:
        # Remove channel dimension (1) and display the 51x51 frame
        frame = raw_stim[frame_idx, 0, :, :]
        axes[row, col].imshow(frame, cmap='gray')
        axes[row, col].set_title(f'Frame {frame_idx}')
        axes[row, col].axis('off')
    else:
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Display next 5 frames of downsampled stimulus
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle(f'Downsampled Stimulus - Next 5 frames starting from frame {starter_frame//2}')

for i in range(5):
    frame_idx = starter_frame//2 + i  # Adjust for downsampling

    if frame_idx < downsampled_stim.shape[0]:
        # Remove channel dimension (1) and display the 51x51 frame
        frame = downsampled_stim[frame_idx, 0, :, :]
        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f'Frame {frame_idx} (orig: {frame_idx*2})')
        axes[i].axis('off')
    else:
        axes[i].axis('off')

plt.tight_layout()
plt.show()

#%%
# one second
T = 240
speed = 1
directions = [0, 45, 90, 135, 180, 225, 270, 315]
cmap = plt.cm.get_cmap("hsv", len(directions))
for direc in directions:
    direction = torch.tensor(direc * np.pi / 180)
    eyepos = torch.concat([torch.cos(direction) * speed * torch.linspace(0, 1, T)[:,None], torch.sin(direction) * speed * torch.linspace(0, 1, T)[:,None]], 1)

    eyevel = pipeline(eyepos)

    enc = model.model.modulator.encoder(eyevel[None].to(model.device))
    scale = model.model.modulator.scale_layer(enc)
    shift = model.model.modulator.shift_layer(enc)

    _ = plt.plot(scale[0].detach().cpu(), color=cmap(directions.index(direc)), label=f'{direc} deg')
plt.show()
#%%

plt.imshow(scale.detach().cpu().numpy().T, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('Conv Dim')
plt.title('Gain')

#%%


#%%
# plot eye pos overlaid
ax = plt.gca().twinx()
ax.plot(eyepos[:,0].detach().cpu(), color='r', alpha=1)
ax.set_ylabel('Eye Position', color='r')
ax.tick_params(axis='y', labelcolor='r')


#%%




model.model.modulator


#%% Run bps analysis to find good cells / get STA



gaborium_inds = get_stim_inds('gaborium', train_data, val_data)
gaborium_robs, gaborium_rhat, gaborium_bps = evaluate_dataset(
    model, train_data, gaborium_inds, dataset_idx, batch_size, "Gaborium"
    )

#%% get stas 
sta_dict = get_sta_ste(model, gaborium_robs, gaborium_rhat, didx=dataset_idx, lags=list(range(16)))

plot_stas(sta_dict)

#%% try plotting the response of one neuron to one batch
cid = 63 # pick from the STA figure
device = model.device # double check in case you ran cells out of order

# randomly sample a batch
start = np.random.randint(0, len(val_data) - batch_size)
bind = np.arange(start, start+batch_size)
batch = val_data[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# plot e and i of readout and the prediction
ex, inh = model_pred(batch, model, dataset_idx, stage='readout')
fun = lambda x: torch.exp(x)
plt.plot(fun(ex[:,cid]).detach().cpu(), 'r')
# plt.plot(1/fun(-inh[:,cid]).detach().cpu(), 'b')
plt.plot(fun(inh[:,cid]).detach().cpu(), 'b')
ax = plt.gca().twinx()
ax.plot(torch.exp(ex[:,cid].detach().cpu() - inh[:,cid].detach().cpu()), 'g')
ax.plot(batch['robs'][:,cid].detach().cpu(), 'k')

del ex, inh
torch.cuda.empty_cache()
#%% Try getting MEI
from mei import mei_synthesis, Jitter, LpNorm, TotalVariation, Combine, GaussianGradientBlur, ClipRange

# define network as function of the stimulus only
def net(x):
    return torch.exp(model.model(x, dataset_idx, batch.get('behavior')[0])[0])

# define MEI parameters
transform = Jitter([4, 4, 4])  # preconditioner for gradients of MEI analysis
regulariser = Combine([LpNorm(p=1, weight=1), TotalVariation(weight=.001)])
precond = GaussianGradientBlur(sigma=[3, 1, 1], order=3)

# (optional: turn off regularizer and preconditioner)
# regulariser = None
# transform = None
# precond = None

# mu = val_data.dsets[0]['stim'].mean().item()
sd = val_data.dsets[0]['stim'].std().item()

# init_image = torch.nn.functional.interpolate(torch.randn(1, 1, 3, 51, 51)*sd, size=(25, 51, 51), mode='nearest')
init_image = torch.randn(1, 1, 25, 51, 51)*sd*2
init_image = precond(init_image.to(device))

# init_image = batch['stim'][0:1].clone()
cid = 63
mei = mei_synthesis(
        model=net,
            initial_image=init_image,
            unit=cid,
            n_iter=1000,
            optimizer_fn=torch.optim.SGD,
            optimizer_kwargs={"lr": 10},
            transform=transform,
            regulariser=regulariser,
            preconditioner=precond,
            postprocessor=None,
            device=model.device
        )

mei_img = mei[0].detach()

_,t,h,w = torch.where(mei_img == torch.max(mei_img))
t = t.item()
h = h.item()
w = w.item()
# plt.plot(mei_img[0,:,:,w].detach().cpu())

plt.figure(figsize=(6,6))
plt.subplot(2,2,1)

plt.imshow(mei_img[0,:,h,:].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.xlabel('Space')
plt.ylabel('Time')
plt.title(f'MEI - Unit {cid}')
temporal_peak = 8# 

plt.axhline(t, color='r', linestyle='--')
plt.axhline(25-temporal_peak, color='b', linestyle='--')

plt.subplot(2,2,2)

I = mei_img[0,-temporal_peak,:,:].detach().cpu().numpy()
# time runs bacwards... does space? 
plt.imshow(I, aspect='auto', cmap='gray')
plt.plot(w,h, 'ro')
plt.xlabel('Space')
plt.ylabel('Space')
plt.title(f'MEI - Unit {cid}')


plt.subplot(2,2,3)
# plot STA at temporal_peak
plt.imshow(sta_dict['sta_robs'][temporal_peak,:,:,cid].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.plot(w,h, 'ro')
plt.xlabel('Space')
plt.ylabel('Time')
plt.title(f'STA - Data {cid}')

plt.subplot(2,2,4)
# plot STA model at temporal_peak
plt.imshow(sta_dict['sta_rhat'][temporal_peak,:,:,cid].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.plot(w,h, 'ro')
plt.xlabel('Space')
plt.ylabel('Time')
plt.title(f'STA - Model {cid}')

plt.tight_layout()
plt.show()


#%% try IRF analysis on gaborium
#  try using jacrev
from scipy.ndimage import gaussian_filter
# import jacrev
from torch.func import jacrev, vmap
lag = 8
T, C, S, H, W  = batch['stim'].shape
n_units      = model.model.readouts[dataset_idx].n_units
unit_ids     = torch.arange(n_units, device=device)
smooth_sigma = .5

# --------------------------------------------------------------------------
# 2. helper – Jacobian → energy CoM for *every* unit in one call
grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device),
                                indexing='ij')
grid_x = grid_x.expand(n_units, H, W)   # each unit gets the same grids
grid_y = grid_y.expand_as(grid_x)

def irf_J(frame_stim, behavior, unit_idx):
    """
    frame_stim : (C,H,W) tensor with grad
    returns     : (n_units, 2)   (cx, cy) per unit, NaN if IRF==0
    """
    def f(s):
        out = model.model(s.unsqueeze(0), dataset_idx, behavior)[0]
        return out[unit_idx]

    return jacrev(f)(frame_stim)

def irf_com(frame_stim, behavior, unit_ids):
    """
    frame_stim : (C,H,W) tensor with grad
    returns     : (n_units, 2)   (cx, cy) per unit, NaN if IRF==0
    """
    J = irf_J(frame_stim, behavior, unit_ids)[:,lag]
    E = J.pow(2)

    if smooth_sigma:
        E = gaussian_filter(E.detach().cpu().numpy(),           # (n_units,H,W)
                            sigma=(0, smooth_sigma, smooth_sigma))
        E = torch.as_tensor(E, device=device)

    tot   = E.flatten(1).sum(-1)                              # (n_units,)
    mask  = tot > 0
    cx    = (E*grid_x).flatten(1).sum(-1) / tot
    cy    = (E*grid_y).flatten(1).sum(-1) / tot
    cx[~mask] = torch.nan
    cy[~mask] = torch.nan
    return torch.stack([cx, cy], 1)           # (n_units,2)

       # (n_units, C, H, W)

#%% --------------------------------------------------------------------------
# 


# find the indices into the frames with the most spikes for the target unit
dset_id = np.where(np.isin(dataset_config['types'], 'gaborium'))[0].item()

data = train_data.shallow_copy()
data.inds = train_data.inds[train_data.inds[:,0]==dset_id]

# indices with most spikes
inds = np.argsort(train_data.dsets[dset_id]['robs'][data.inds[:,1],cid]).numpy()[::-1]

n = 5000
irfs = []
for i in tqdm(range(n)):
    batch = data[inds[i]]
    J = irf_J(batch['stim'].to(device), batch['behavior'].to(device), list(range(n_units)))
    irfs.append(J.detach().cpu().numpy())
    del batch,J
    torch.cuda.empty_cache()



#%% try to recover a subspace
if isinstance(irfs, list):
    irfs = np.concatenate(irfs, 1)
    
cid = 63
N,T,H,W = irfs[cid].shape
k = 5
# pca
u,s,v = torch.svd_lowrank(torch.as_tensor(irfs[cid].reshape(N, T*H*W)).to(device).T, k)

plt.figure(figsize=(20, 10))
for i in range(k):
    pc = u[:,i].reshape(T,H,W).detach().cpu().numpy()
    # find max and take spatiotemporal slice and spatial plot at peak temporal
    t, h, w = np.where(np.abs(pc) == np.abs(pc).max())
    t = t.item()
    h = h.item()
    w = w.item()

    plt.subplot(2,k,i+1+k)
    plt.imshow(pc[t,:,:], aspect='auto', cmap='gray')
    plt.xlabel('Space')
    plt.ylabel('Space')
    
    
    plt.subplot(2,k,i+1)
    plt.imshow(pc[:,h,:], aspect='auto', cmap='gray')
    plt.xlabel('Space')
    plt.ylabel('Time')

plt.tight_layout()
plt.show()

# irf = np.mean(irfs[cid], axis=0)
# r = train_data.dsets[dset_id]['robs'][:,cid].numpy()[inds[:1000]]


# %%
