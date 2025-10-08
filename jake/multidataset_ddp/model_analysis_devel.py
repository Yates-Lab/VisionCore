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
from eval_stack_utils import argmin_subpixel, argmax_subpixel

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()

device = get_free_device()

#%% Discover Available Models
print("🔍 Discovering available models...")
from eval_stack_multidataset import scan_checkpoints
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage/checkpoints"
# checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'
models_by_type = scan_checkpoints(checkpoint_dir)

print(f"Found {len(models_by_type)} model types:")
for model_type, models in models_by_type.items():
    if models:
        best_model = models[0]
        if best_model.get('metric_type') == 'bps' and best_model.get('val_bps') is not None:
            best_metric = f"best BPS: {best_model['val_bps']:.4f}"
        else:
            best_metric = f"best loss: {best_model['val_loss']:.4f}"
        print(f"  {model_type}: {len(models)} models ({best_metric})")
    else:
        print(f"  {model_type}: 0 models")


#%% Utilities for evaluation
from tqdm import tqdm

def model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=False):

    if include_modulator:
        behavior = batch.get('behavior')
        if zero_modulator:
            behavior = torch.zeros_like(behavior)
    else:
        behavior = None

    # set for hooks
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
    
    elif stage=='pred':
        
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
    

#%% LOAD A MODEL


model_type = 'learned_res_small_gru'
model, model_info = load_model(
        model_type=model_type,
        model_index=None, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )


model.model.eval()
model.model.convnet.use_checkpointing = False 

model = model.to(device)

plt.plot(model.model.frontend.temporal_conv.weight.squeeze().detach().cpu().T)
#%% Run bps analysis to find good cells / get STA
dataset_idx = 8
batch_size = 64 # keep small because things blow up fast!

train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

#%%
dataset_cids = dataset_config.get('cids', [])

stim_type = 'backimage'
inds = get_stim_inds(stim_type, train_data, val_data)

dataset = val_data.shallow_copy()
dataset.inds = inds



# #%% DEBUGGING FIXRSVP CCNORM
# from eval_stack_multidataset import run_bps_analysis, run_ccnorm_analysis, run_saccade_analysis
# from pathlib import Path
# from eval_stack_utils import get_fixrsvp_trials

# # Get trial-aligned FixRSVP data
# model_name = model_info['experiment']
# save_dir = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack_120") / model_name

# stim_type = 'fixrsvp'
# stim_inds = get_stim_inds(stim_type, train_data, val_data)
# result = evaluate_dataset(
#                     model, train_data, stim_inds, dataset_idx, batch_size, stim_type.capitalize()
#                 )
# robs, rhat, bps = result['robs'], result['rhat'], result['bps']

# bps_results = {'fixrsvp': (robs, rhat, bps)}

# #%%

# stim_indices = get_stim_inds('fixrsvp', train_data, val_data)
# data = val_data.shallow_copy()
# data.inds = stim_indices

# dset_idx = np.unique(stim_indices[:,0]).item()
# time_inds = data.dsets[dset_idx]['psth_inds'].numpy()
# trial_inds = data.dsets[dset_idx]['trial_inds'].numpy()
# unique_trials = np.unique(trial_inds)

# n_trials = len(unique_trials)
# n_time = np.max(time_inds).item()+1
# n_units = data.dsets[dset_idx]['robs'].shape[1]
# robs_trial = np.nan*np.zeros((n_trials, n_time, n_units))
# rhat_trial = np.nan*np.zeros((n_trials, n_time, n_units))
# dfs_trial = np.nan*np.zeros((n_trials, n_time, n_units))

# for itrial in range(n_trials):
#     trial_idx = np.where(trial_inds == unique_trials[itrial])[0]
#     eval_inds = np.where(np.isin(stim_indices[:,1], trial_idx))[0]
#     data_inds = trial_idx[np.where(np.isin(trial_idx, stim_indices[:,1]))[0]]

#     assert torch.all(robs[eval_inds] == data.dsets[dset_idx]['robs'][data_inds]).item(), 'robs mismatch'

#     robs_trial[itrial, time_inds[data_inds]] = robs[eval_inds]
#     rhat_trial[itrial, time_inds[data_inds]] = rhat[eval_inds]
#     dfs_trial[itrial, time_inds[data_inds]] = data.dsets[dset_idx]['dfs'][data_inds]

# cid = 10 
# #%%
# cid +=1

# plt.subplot(2,1,1)
# plt.imshow(robs_trial[:,:,cid], aspect='auto', interpolation='none', cmap='gray_r')
# plt.xlim(0, 200)
# plt.subplot(2,1,2)
# plt.imshow(rhat_trial[:,:,cid], aspect='auto', interpolation='none', cmap='gray_r')
# plt.xlim(0, 200)
# ax = plt.gca().twinx()
# ax.plot(np.nanmean(robs_trial[:,:,cid], 0), 'k')
# ax.plot(np.nanmean(rhat_trial[:,:,cid], 0), 'r')
# plt.show()

# #%%
# robs_trial, rhat_trial, dfs_trial = get_fixrsvp_trials(
#     model, bps_results, dataset_idx, train_data, val_data
# )


#%% STA utilities



#%% Plot utilities
from gaborium_analysis import get_sta_ste

def plot_stas(sta, lag = None, normalize=True, sort_by=None):

    n_cells = sta['Z_STA_robs'].shape[-1]
    sx = np.floor(np.sqrt(n_cells)).astype(int)
    sy = np.ceil(n_cells / sx).astype(int)
    fig, axs = plt.subplots(sy, sx, figsize=(16, 16))
    
    
    H = sta['Z_STA_robs'].shape[1]

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
            lag = sta['peak_lag'][cc]

        ax = axs.flatten()[i]
        v = sta['Z_STA_robs'][lag,:,:,cc].abs().max()
        Irhat = sta['Z_STA_rhat'][lag,:,:,cc]
        if normalize:
            vrhat = Irhat.abs().max()
            Irhat = Irhat / vrhat * v
            
        I = torch.concat([sta['Z_STA_robs'][lag,:,:,cc], torch.ones(H,1), Irhat], 1)
        
        ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)
        ax.set_title(f'{cc}: {sta['modulation_index_robs'][cc]:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_output(output, use_imshow=True):
    sz = list(output.shape)

    n = np.minimum(sz[1], 30)
    plt.figure(figsize=(10,n))

    clrs = plt.cm.get_cmap("tab10", n)
    if len(output.shape) == 2:
        use_imshow = False

    for i in range(n):
        plt.subplot(n,1,i+1)
        if use_imshow:
                _ = plt.imshow(output.detach().cpu()[:,i,0,sz[-2]//2,:].T, interpolation='none', aspect='auto', cmap='viridis')
        else:
            if len(output.shape)==2:
                _ = plt.plot(output.detach().cpu()[:,i], color=clrs(i))
            else:
                _ = plt.plot(output.detach().cpu()[:,i,0,sz[-2]//2,2:], color=clrs(i))
        ax = plt.gca().twinx()
        ax.plot(batch['eyepos'].cpu().numpy())

#%%
gaborium_inds = torch.concatenate([
            train_data.get_dataset_inds('gaborium'),
            val_data.get_dataset_inds('gaborium')
        ], dim=0)

dataset = train_data.shallow_copy()
# set indices to be the gaborium inds
dataset.inds = gaborium_inds

gaborium_eval = evaluate_dataset(
    model, dataset, gaborium_inds, dataset_idx, batch_size, "Gaborium"
    )


#%% rescale rhat
from eval_stack_utils import rescale_rhat, bits_per_spike
gaborium_robs = gaborium_eval['robs']
gaborium_rhat = gaborium_eval['rhat']
gaborium_dfs = gaborium_eval['dfs']
gaborium_rhat_rescaled, _ = rescale_rhat(gaborium_robs, gaborium_rhat, gaborium_dfs, mode='affine')

# recalculate bps
gaborium_bps = bits_per_spike(gaborium_rhat, gaborium_robs, gaborium_dfs)
gaborium_bps_rescaled = bits_per_spike(gaborium_rhat_rescaled, gaborium_robs, gaborium_dfs)

plt.plot(gaborium_bps, gaborium_bps_rescaled, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')

gaborium_rhat = gaborium_rhat_rescaled

#%%
plt.plot(gaborium_bps, '.')
plt.plot(gaborium_bps_rescaled, '.')
plt.axhline(0, color='k', linestyle='--')

#%% fuxations only

dataset_config = model.model.dataset_configs[dataset_idx].copy()
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
# eyevel = data.dsets[dset_idx]['behavior'][data.inds[:,1]][:,18:22].sum(1)
#%% get stas 

sta_dict = get_sta_ste((train_data, val_data, dataset_config), gaborium_robs, gaborium_rhat, lags=list(range(16)), fixations_only=False, combine_train_test=True, whiten=True, device=model.device)

#%%
for key, val in sta_dict.items():
    if hasattr(val, 'shape'):
        print(key, val.shape)
    else:
        print(key)

#%%
plot_stas(sta_dict, lag=None, normalize=True, sort_by='modulation_index_rhat')

#%%


                    
#%%

plt.plot(sta_dict['modulation_index_rhat'], sta_dict['modulation_index_robs'], '.')
plt.plot([-1,1], [-1,1], 'k--')
plt.xlabel('modulation index rhat')
plt.ylabel('modulation index robs')

for i, (x, y) in enumerate(zip(sta_dict['modulation_index_rhat'], sta_dict['modulation_index_robs'])):
    plt.text(x, y + 0.02, str(i), fontsize=8, ha='center', va='bottom')
#%%

device = model.device # double check in case you ran cells out of order

batch_size = 32
# randomly sample a batch
start = np.random.randint(0, len(dataset) - batch_size)
# start = 30341 - batch_size
# start = start + batch_size - 200
bind = np.arange(start, start+batch_size)


batch = dataset[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

plt.figure()
plt.subplot(1,2,1)
plt.imshow(batch['stim'][0,0,-1].detach().cpu())
plt.subplot(1,2,2)
plt.imshow(batch['stim'][-1,0,-1].detach().cpu())

print(start)
#%%

output = model_pred(batch, model, dataset_idx, stage='conv.0')
plot_output(output, use_imshow=True)

del output
torch.cuda.empty_cache()


#%% Make animation of the 8 conv.0 channels
import matplotlib.animation as animation

# Get the conv.0 output and stimulus data
output = model_pred(batch, model, dataset_idx, stage='conv.0')
stim_data = batch['stim']

# Extract the data we need for animation
# Stimulus: [256, 1, 25, 51, 51] -> take [:, 0, -1, :, :]
# Conv output: [256, 8, 5, 24, 24] -> take [:, i, -1, :, :] for each channel
stim_frames = stim_data[:, 0, 0, :, :].detach().cpu().numpy()  # [256, 51, 51]
conv_frames = output[:, :, -1, :, :].detach().cpu().numpy()     # [256, 8, 24, 24]

# Create 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

# Remove axes decorations for clean look
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# Adjust spacing so subplots touch each other
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

# Initialize image objects for each subplot
ims = []

# Stimulus in top-left (index 0)
im_stim = axes[0].imshow(stim_frames[0], cmap='gray', animated=True)
ims.append(im_stim)

# Conv outputs in remaining 8 positions
for i in range(8):
    im_conv = axes[i+1].imshow(conv_frames[0, i], cmap='viridis', animated=True)
    ims.append(im_conv)

def animate(frame):
    """Update function for animation"""
    # Update stimulus
    ims[0].set_array(stim_frames[frame])

    # Update conv outputs
    for i in range(8):
        ims[i+1].set_array(conv_frames[frame, i])

    return ims

# Create animation
n_frames = stim_frames.shape[0]  # 256 frames
anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                              interval=100, blit=True, repeat=True)

# plt.show()

# Optionally save the animation
anim.save('conv_filters_animation.mp4', writer='ffmpeg', fps=10)

#%% get STAs from first specific layer
from mei import mei_synthesis, Jitter, LpNorm, TotalVariation, Combine, GaussianGradientBlur, ClipRange

batch_size = 1
# randomly sample a batch
start = np.random.randint(0, len(val_data) - batch_size)
bind = np.arange(start, start+batch_size)

batch = val_data[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

model.model.convnet.layers[0].main_block.components.conv.conv.padding_mode = 'zeros'

# define network as function of the stimulus only
target = 'conv.0'
if target == 'pred':
    def net(x):
        batch['stim'] = x
        return model_pred(batch, model, dataset_idx, stage=target)[0,:]
else:

    def net(x):
        batch['stim'] = x
        output = model_pred(batch, model, dataset_idx, stage=target)
        return output[0,:,-1, output.shape[-2]//2, output.shape[-1]//2]

y = net(torch.randn_like(batch['stim']))


# define MEI parameters
transform = Jitter([4, 4, 4])  # preconditioner for gradients of MEI analysis
regulariser = Combine([LpNorm(p=1, weight=1), TotalVariation(weight=.001)])
precond = GaussianGradientBlur(sigma=[1, 1, 1], order=3)

# (optional: turn off regularizer and preconditioner)
regulariser = None
transform = None
# precond = None

# mu = val_data.dsets[0]['stim'].mean().item()
sd = val_data.dsets[0]['stim'].std().item()

# init_image = torch.nn.functional.interpolate(torch.randn(1, 1, 3, 51, 51)*sd, size=(25, 51, 51), mode='nearest')
init_image = torch.randn(1, 1, 25, 51, 51)*sd*2
if precond is not None:
    init_image = precond(init_image.to(device))

# init_image = batch['stim'][0:1].clone()
meis = []
for cid in range(len(y)):
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
    meis.append(mei_img)

    _,t,h,w = torch.where(mei_img.abs() == torch.max(mei_img.abs()))
    t = t.item()
    h = h.item()
    w = w.item()
    # plt.plot(mei_img[0,:,:,w].detach().cpu())

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    v = mei_img.abs().max()

    plt.imshow(mei_img[0,:,h,:].detach().cpu().numpy(), aspect='auto', cmap='gray', vmin=-v, vmax=v)
    plt.xlabel('Space')
    plt.ylabel('Time')
    plt.title(f'MEI - Unit {cid}')
    temporal_peak = 7#

    plt.axhline(t, color='r', linestyle='--')
    # plt.axhline(25-temporal_peak, color='b', linestyle='--')

    plt.subplot(1,2,2)


    I = mei_img[0,t,:,:].detach().cpu().numpy()
    # time runs bacwards... does space? 
    plt.imshow(I, aspect='auto', cmap='gray', vmin=-v, vmax=v)
    plt.axhline(h, color='r', linestyle='--')
    # plt.plot(w,h, 'ro')
    plt.xlabel('Space')
    plt.ylabel('Space')
    plt.title(f'MEI - Unit {cid}')

    n_cells = batch['robs'].shape[1]
    stim_dims = batch['stim'].shape[2:]
    plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_mei_slices(meis, figsize=None, cmap='gray'):
    """
    meis : list of torch.Tensor shape (1, T, H, W)
    Produces an N×2 figure: left=space-time slice, right=spatial slice.
    """
    # convert to numpy and squeeze channel
    arrs = [mei.detach().cpu().squeeze(0).numpy() for mei in meis]  # each is (T,H,W)
    N = len(arrs)
    if figsize is None:
        figsize = (6, 2.5 * N)

    # global colormap limits
    

    fig, axes = plt.subplots(N, 2, figsize=figsize, squeeze=False)
    for i, arr in enumerate(arrs):
        absmax = max(arr.max(), -arr.min())
        vmin, vmax = -absmax, absmax
        T, H, W = arr.shape

        # find global peak (t0, h0, w0)
        t0, h0, w0 = np.unravel_index(np.abs(arr).argmax(), arr.shape)

        # left: plot arr[:, h0, :] → shape (T, W)
        ax = axes[i,0]
        im0 = ax.imshow(arr[:, h0, :],
                        aspect='auto',
                        cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_ylabel('Time')
        ax.set_xlabel('Space (x)')
        if i==0:
            ax.set_title('space–time slice')

        # right: plot arr[t0, :, :] → shape (H, W)
        ax = axes[i,1]
        im1 = ax.imshow(arr[t0, :, :],
                        aspect='auto',
                        cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_ylabel('Space (y)')
        ax.set_xlabel('Space (x)')
        if i==0:
            ax.set_title('spatial slice')

    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def animate_meis(meis, save_path=None, fps=10, cmap='gray', dpi=150):
    """
    meis     : list of 8 torch.Tensor shape (1, T, H, W)
    save_path: if given, e.g. 'meis.mp4', will write an MP4 via ffmpeg
    returns  : a matplotlib.animation.FuncAnimation
    Each subplot uses its own vmin/vmax based on that unit’s full time series.
    """
    # convert and squeeze
    arrs = [mei.detach().cpu().squeeze(0).numpy() for mei in meis]  # each (T,H,W)
    T, H, W = arrs[0].shape
    assert all(a.shape[0]==T for a in arrs), "all MEIs must have same T"
    N = len(arrs)
    sx = int(np.ceil(np.sqrt(N)))
    sy = int(np.ceil(N / sx))

    fig, axes = plt.subplots(sx, sy, figsize=(4*sy, 4*sx))
    
    axes = axes.flatten()
    # set all axes off
    for ax in axes:
        ax.axis('off')
    ims = []

    for i in range(N):
        ax = axes[i]
        arr = arrs[i]
        # per-unit vmin/vmax
        unit_max = np.abs(arr).max()
        vmin, vmax = -unit_max, unit_max

        im = ax.imshow(np.zeros((H,W)),
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       animated=True, aspect='auto')
        ax.axis('off')
        ax.set_title(f'Unit {i}')
        ims.append(im)

    def update(frame):
        for im, arr in zip(ims, arrs):
            im.set_array(arr[frame, :, :])
        return ims

    anim = animation.FuncAnimation(fig, update, frames=T, blit=True)
    if save_path:
        anim.save(save_path, fps=fps, dpi=dpi)
    return anim

#%%
plot_mei_slices(meis)

anim = animate_meis(meis, save_path=f'meis_{target}_{model_type}.mp4', fps=10)

#%%

#%%
device = model.device # double check in case you ran cells out of order

batch_size = 256
# randomly sample a batch
start = np.random.randint(0, len(val_data) - batch_size)
bind = np.arange(start, start+batch_size)


batch = val_data[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

output = model_pred(batch, model, dataset_idx, stage='conv.0')
plot_output(output, use_imshow=False)

del output
torch.cuda.empty_cache()

#%%

output = model_pred(batch, model, dataset_idx, stage='pred')
plot_output(output, use_imshow=False)
n = np.minimum(output.shape[1], 30)
for i in range(n):
    plt.subplot(n,1,i+1)
    ax = plt.gca().twinx()
    ax.plot(batch['robs'][:,i].detach().cpu(), 'k')

del output
torch.cuda.empty_cache()

#%%
mod = model_pred(batch, model, dataset_idx, stage='modulator', zero_modulator=True)
n = np.minimum(mod.shape[1], 100)
plt.figure(figsize=(10,n))
for i in range(n):
    plt.subplot(n,1,i+1)
    _ = plt.plot(mod[:,-i,-1,3,:].detach().cpu(), 'b')

del mod
torch.cuda.empty_cache()

mod = model_pred(batch, model, dataset_idx, stage='modulator')

for i in range(n):
    plt.subplot(n,1,i+1)
    _ = plt.plot(mod[:,-i,-1,3,:].detach().cpu(), 'r')

del mod
torch.cuda.empty_cache()
# ax = plt.gca().twinx()

#%%

model, model_info = load_model(
        model_type=model_type,
        model_index=None, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )


model.model.eval()
model.to(device)
model.model.convnet.use_checkpointing = False 


mod = model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=False)

# get correlation between mod and robs
cc = []
for i in range(mod.shape[1]):
    cc.append(np.corrcoef(mod[:,i].detach().cpu().numpy(), batch['robs'][:,i].detach().cpu().numpy())[0,1].item())

cc = np.array(cc)
orderinds = np.argsort(cc)[::-1]
orderinds = orderinds[np.where(~np.isnan(cc[orderinds]))[0]]


n = np.minimum(mod.shape[1], 50)
plt.figure(figsize=(10,n))
# for i in range(n):
#     plt.subplot(n,1,i+1)
#     _ = plt.plot(mod[:,-i].detach().cpu(), 'k')

del mod
torch.cuda.empty_cache()

model_type = 'learned_res_small_gru'
model, model_info = load_model(
        model_type=model_type,
        model_index=None, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.model.eval()
model.to(device)
model.model.convnet.use_checkpointing = False 

for i in range(n):
    plt.subplot(n,1,i+1)
    
    _ = plt.fill_between(np.arange(batch_size), batch['robs'][:,orderinds[i]].detach().cpu().numpy(), color='gray')

mod = model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=False)
for i in range(n):
    plt.subplot(n,1,i+1)
    _ = plt.plot(mod[:,orderinds[i]].detach().cpu(), 'b')

del mod
torch.cuda.empty_cache()

mod = model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=True)
for i in range(n):
    plt.subplot(n,1,i+1)
    _ = plt.plot(mod[:,orderinds[i]].detach().cpu(), 'r')


del mod
torch.cuda.empty_cache()
plt.savefig('modulator_trial_compare.pdf')
# ax.plot(batch['eyepos'].cpu().numpy())

#%% Run saccade analysis
from DataYatesV1 import get_session
from eval_stack_utils import detect_saccades_from_session

dataset_name = model.names[dataset_idx]
sess = get_session(*dataset_name.split('_'))
saccades = detect_saccades_from_session(sess)

#%%
target = 'pred.zero'

if target == 'pred.zero':
    def net(batch):
        return model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=True)
elif target == 'pred':
    def net(batch):
        return model_pred(batch, model, dataset_idx, stage='pred', include_modulator=True, zero_modulator=False)
else:
    def net(batch):
        out = model_pred(batch, model, dataset_idx, stage=target)
        sz = out.shape
        return out[:,:,-1, sz[-2]//2, sz[-1]//2]

pred = net(batch)
sz = pred.shape
print(sz)

#%% prepare data
win = (-50, 100)
# get saccade times
saccade_times = torch.tensor([s['start_time'] for s in saccades])
    
# Convert saccade times to dataset indices
saccade_inds = train_data.get_inds_from_times(saccade_times)

# Get stimulus indices using the helper function
stim_inds = get_stim_inds(stim_type, train_data, val_data)

dataset = val_data.shallow_copy()

dataset.inds = stim_inds

dset = stim_inds[0,0]
print(f'Dataset {dset}')

# print(f"Number of robs bins: {robs.shape[0]} Number of stim bins: {stim_inds.shape[0]}")
nbins = win[1]-win[0]

valid_saccades = np.where(saccade_inds[:,0]==dset)[0]

sac_indices = saccade_inds[valid_saccades, 1]
n_sac = len(sac_indices)

n_cells = batch['robs'].shape[1]
robs_sac = np.nan*np.zeros((n_sac, nbins, n_cells))
eye_vel_sac = np.nan*np.zeros((n_sac, nbins, 1))
pred_sac = np.nan*np.zeros((n_sac, nbins, sz[1]))
dfs_sac = np.nan*np.zeros((n_sac, nbins, n_cells))

saccade_info = [saccades[i] for i in valid_saccades]

time_previous = np.nan*np.zeros(n_sac)
time_next = np.nan*np.zeros(n_sac)

for i,isac in enumerate(sac_indices):
    print(f"i: {i}/{n_sac}, isac: {isac}")
    # if i > 2:
    #     break

    j = np.where(stim_inds[:,1] == isac)[0]

    if len(j) == 0:
        continue

    j = j.item()

    dataset_indices = np.where(torch.all(dataset.inds == stim_inds[j], 1))[0]
    if len(dataset_indices) == 0:
        continue

    dataset_indices = dataset_indices.item()

    if (j + win[0] >= 0):
        batch= dataset[dataset_indices+win[0]:dataset_indices+win[1]]
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        if batch['robs'].shape[0] != nbins:
            continue
        pred = net(batch)

        if i > 0:
            prev_sac_bin = isac - sac_indices[i-1]
            time_previous[i] = prev_sac_bin

        if i < len(sac_indices)-1:
            next_sac_bin = sac_indices[i+1] - isac
            time_next[i] = next_sac_bin
        
        robs_sac[i] = batch['robs'].detach().cpu().numpy()
        eye_vel = np.sqrt(np.sum(np.gradient(batch['eyepos'].detach().cpu().numpy(), axis=0)**2, 1))
        eye_vel_sac[i] = eye_vel[:,None]
        pred_sac[i] = pred.detach().cpu().numpy()
        dfs_sac[i] = batch['dfs'].detach().cpu().numpy()


del pred
torch.cuda.empty_cache()      


#%%
# eye_vel_sac has shape [n_saccades, nbins, …]
# Build a single “valid” mask of shape (n_saccades,)
dt = 1/120
validix = ~np.isnan(eye_vel_sac).any(axis=1).flatten()
validix = validix & (time_next*dt > 0.1) & (time_previous*dt > 0.1)
validix = validix & (time_next*dt < 0.5) & (time_previous*dt < 0.5)
valid = np.where(validix)[0]

tn0 = time_next[valid]
tp0 = time_previous[valid]
ev0 = eye_vel_sac[valid, :, 0]/dt  # pick the first velocity channel
#%%
# Now sort
plt.subplot(1,2,1)
order = np.argsort(tp0)
tn  = tn0[order]
ev  = ev0[order, :]

# And plot with the correct extent in seconds
plt.imshow(ev,
           aspect='auto',
           interpolation='none',
           cmap='gray_r',
           vmin=0, vmax=100,
           extent=[win[0]*dt, win[1]*dt, 0, ev.shape[0]],
           origin='lower')

plt.subplot(1,2,2)
order = np.argsort(tn0)
tn  = tn0[order]
ev  = ev0[order, :]

# And plot with the correct extent in seconds
plt.imshow(ev,
           aspect='auto',
           interpolation='none',
           cmap='gray_r',
           vmin=0, vmax=100,
           extent=[win[0]*dt, win[1]*dt, 0, ev.shape[0]],
           origin='lower')           
# plt.plot(tn, np.arange(len(tn)), 'r.')
# plt.xlim(-.1, .6)

c = 0

#%%
c += 1
r0 = robs_sac[valid, :, c]/dt  # pick the first velocity channel

plt.subplot(1,2,1)
order = np.argsort(tp0)
tn  = tn0[order]
r  = r0[order, :]

# And plot with the correct extent in seconds
plt.imshow(r,
           aspect='auto',
           interpolation='none',
           cmap='gray_r',
        #    vmin=0, vmax=100,
           extent=[win[0]*dt, win[1]*dt, 0, r.shape[0]],
           origin='lower')

plt.title('Sorted by Previous Saccade')

plt.subplot(1,2,2)
order = np.argsort(tn0)
tn  = tn0[order]
r  = r0[order, :]

# And plot with the correct extent in seconds
plt.imshow(r,
           aspect='auto',
           interpolation='none',
           cmap='gray_r',
        #    vmin=0, vmax=100,
           extent=[win[0]*dt, win[1]*dt, 0, r.shape[0]],
           origin='lower')  

plt.title('Sorted by Next Saccade')

#%%

#%%
good = np.where(np.sum(np.isnan(robs_sac), axis=(1,2)) == 0)[0]

robs_sac = robs_sac[good]
pred_sac = pred_sac[good]
dfs_sac = dfs_sac[good]

saccade_info = [saccade_info[i] for i in good]

print(f"Number of good saccades: {len(good)}")
rbar = np.nansum(robs_sac*dfs_sac, axis=0) / np.nansum(dfs_sac, axis=0)

#%%
predbar = np.nanmean(pred_sac, axis=0)
rbar = np.nanmean(robs_sac, axis=0)
predvar = np.nanvar(pred_sac, axis=0)



def get_Rnorm(R, tbins):
    from scipy.ndimage import gaussian_filter1d

    R = gaussian_filter1d(R, 1, axis=0)
    # R = savgol_filter(R, 21, 3, axis=0)
    R0 = R[(tbins < 0) & (tbins > -0.1)].mean(0)
    R = (R - R0)
    R = R / R.max(0)

    return R


#%% plot model predictions mean and variance
dt = 1/120
tbins = np.arange(win[0], win[1])*dt


Rlayer_mu = get_Rnorm(predbar, tbins)/2
Rlayer_var = get_Rnorm(predvar, tbins)/2

plt.figure(figsize=(3, 45))
for i in range(Rlayer_mu.shape[1]):
    h = plt.plot(tbins, Rlayer_mu[:,i] - i,  alpha=1)
    _ = plt.plot(tbins, Rlayer_var[:,i] - i, color=h[0].get_color(), linestyle='--')
    _ = plt.axhline(-i, color='k', linestyle='--', alpha=.5)
    _ = plt.axvline(0, color='k', linestyle='--', alpha=.5)

plt.xlim(-.1, .35)
# ytick labels to be absolute value of what they were
ax = plt.gca()
yticks = ax.get_yticks()
ax.set_yticklabels(np.abs(yticks).astype(int))
plt.xlabel('Time from saccade onset (s)')
plt.ylabel(f'{target} id')
plt.title('Unit Response')

#%% plot model vs. data
dt = 1/120
tbins = np.arange(win[0], win[1])*dt


Rlayer_mu = get_Rnorm(predbar, tbins)/2
Rlayer_pred = get_Rnorm(rbar, tbins)/2

plt.figure(figsize=(3, 45))
for i in range(Rlayer_mu.shape[1]):
    h = plt.plot(tbins, Rlayer_mu[:,i] - i,  alpha=1)
    _ = plt.plot(tbins, Rlayer_pred[:,i] - i, color=h[0].get_color(), linestyle='--')
    _ = plt.axhline(-i, color='k', linestyle='--', alpha=.5)
    _ = plt.axvline(0, color='k', linestyle='--', alpha=.5)

plt.xlim(-.1, .35)
# ytick labels to be absolute value of what they were
ax = plt.gca()
yticks = ax.get_yticks()
ax.set_yticklabels(np.abs(yticks).astype(int))
plt.xlabel('Time from saccade onset (s)')
plt.ylabel(f'{target} id')
plt.title('Unit Response')
cc = 0
#%% one by one raw
cc += 1
if cc >= rbar.shape[1]:
    cc = 0
plt.plot(rbar[:,cc], 'k', label='Data')
plt.plot(predbar[:,cc], 'r', label='Model')
plt.legend()


#%%



#%%



#%%
cc = 91
sta = Z_STA_robs[:,:,:,cc]
ste = Z_STE_robs[:,:,:,cc]
peak_lag = np.argmax(ste.std((1,2))).item()
snr_sta = sta[peak_lag].abs().max().item()
snr_ste = ste[peak_lag].abs().max().item()

plt.subplot(2,2,1)
plt.imshow(sta[0].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title(f'STA - Data {cc}')
plt.subplot(2,2,2)
plt.imshow(sta[peak_lag].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title(f'STA - SNR {snr_sta:.2f}')
plt.subplot(2,2,3)
plt.imshow(ste[0].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title(f'STE - Data {cc}')
plt.subplot(2,2,4)
plt.imshow(ste[peak_lag].detach().cpu().numpy(), aspect='auto', cmap='gray')
plt.title(f'STE - SNR {snr_ste:.2f}')


#%%
snr = np.array(snrs)
plt.plot((snr[:,0] - snr[:,1]) / (snr[:,0] + snr[:,1]), '.')


#%%
from eval_stack_multidataset import run_saccade_analysis

saccade_results = run_saccade_analysis(
                        model, train_data, val_data, dataset_idx, bps_results if 'bps' in analyses else None,
                        model_name, save_dir, recalc, sac_win
                    )

#%% Try getting MEI
from mei import mei_synthesis, Jitter, LpNorm, TotalVariation, Combine, GaussianGradientBlur, ClipRange

# define network as function of the stimulus only
def net(x):
    return torch.exp(model.model(x, dataset_idx, batch.get('behavior')[0:1])[0])

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
if precond is not None:
    init_image = precond(init_image.to(device))

# init_image = batch['stim'][0:1].clone()
cid = 4
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
# temporal_peak = 7#

plt.axhline(t, color='r', linestyle='--')
# plt.axhline(25-temporal_peak, color='b', linestyle='--')

plt.subplot(2,2,2)

I = mei_img[0,t,:,:].detach().cpu().numpy()
# time runs bacwards... does space? 
plt.imshow(I, aspect='auto', cmap='gray')
plt.plot(w,h, 'ro')
plt.xlabel('Space')
plt.ylabel('Space')
plt.title(f'MEI - Unit {cid}')


#%%
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
    
cid = 10
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

feat = torch.concat([r.features.weight.squeeze().detach().cpu() for r in model.model.readouts])
C = np.cov(feat.numpy())
u,s,v = np.linalg.svd(C)
# subtract diagonal
C -= np.diag(np.diag(C))


C = torch.mm(feat, feat.T)
C.fill_diagonal_(0)
plt.imshow(C, aspect='auto', interpolation='none')
# %%
plt.plot(np.cumsum(s)/np.sum(s))
plt.xlim(0, 256)
# %%

plt.imshow(feat, aspect='auto', interpolation='none', cmap='coolwarm', vmin=-.5, vmax=.5)
# plt.axvline(128, color='k', linestyle='--')
# y axis is neuron id
# x axis is feature < 128 is vision only, >128 is Prediction Error
# plt.xlabel('Feature < 128: Vision, > 128: Prediction Error')
plt.ylabel('Neuron')

plt.colorbar()
# %%

plt.plot(feat[:,:128].abs().sum(1), feat[:,128:].abs().sum(1), '.')
plt.plot(plt.ylim(), plt.ylim(), 'k--')
plt.xlabel('Sum of Error weights')
plt.ylabel('Sum of Vision weights')
# %%
# Modified function to visualize the receptive field
def visualize_receptive_field():
    # First approach: Use a simpler method with hooks to capture the receptive field
    # Create a hook to capture gradients
    gradients = []
    
    def save_grad(grad):
        gradients.append(grad.detach().clone())
    
    # Create an input that requires gradients
    input_tensor = torch.zeros_like(batch['stim']).to(device).requires_grad_(True)
    
    # Place a single point in the middle of the input
    middle_t = input_tensor.shape[2] // 2
    middle_h = input_tensor.shape[3] // 2
    middle_w = input_tensor.shape[4] // 2
    
    # Register hook to save gradients
    input_tensor.register_hook(save_grad)
    
    # Forward pass
    batch['stim'] = input_tensor
    output = model_pred(batch, model, dataset_idx, stage='conv.2')
    
    # Select a specific location in the output to backpropagate from
    target_h = output.shape[3] // 2
    target_w = output.shape[4] // 2
    
    # Create a one-hot tensor at the target location
    target = torch.zeros_like(output)
    target[0, 0, -1, target_h, target_w] = 1.0
    
    # Compute loss and backpropagate
    loss = (output * target).sum()
    
    # Check if output requires grad
    if not output.requires_grad:
        print("Output doesn't require gradients. Let's try a different approach.")
        
        # Let's try a different approach using direct visualization
        # Create a delta function input
        delta = torch.zeros_like(batch['stim']).to(device)
        delta[0, 0, middle_t, middle_h, middle_w] = 1.0
        
        # Run it through the model
        batch['stim'] = delta
        output = model_pred(batch, model, dataset_idx, stage='conv.2')
        
        # Visualize the output
        plt.figure(figsize=(15, 5))
        
        # Show the input
        plt.subplot(1, 2, 1)
        plt.imshow(delta[0, 0, middle_t].detach().cpu().numpy(), cmap='gray')
        plt.title("Delta Input")
        
        # Show the output for the first channel
        plt.subplot(1, 2, 2)
        plt.imshow(output[0, 0, -1].detach().cpu().numpy(), cmap='viridis')
        plt.title("Output Channel 0")
        
        plt.tight_layout()
        plt.show()
        
        return output
    
    # If we get here, we can backpropagate
    loss.backward()
    
    # Get the gradient
    receptive_field = gradients[0]
    
    # Visualize the receptive field
    plt.figure(figsize=(15, 5))
    
    # Show the receptive field at the middle time step
    plt.subplot(1, 2, 1)
    plt.imshow(receptive_field[0, 0, middle_t].detach().cpu().numpy(), cmap='coolwarm')
    plt.title(f"Receptive Field (t={middle_t})")
    
    # Show the receptive field across time at the middle spatial location
    plt.subplot(1, 2, 2)
    plt.imshow(receptive_field[0, 0, :, middle_h, middle_w].detach().cpu().numpy().reshape(-1, 1), 
               aspect='auto', cmap='coolwarm')
    plt.title(f"Temporal RF at center")
    
    plt.tight_layout()
    plt.show()
    
    return receptive_field

# Try the modified function
rf = visualize_receptive_field()

#%%

batch_size = 1
# randomly sample a batch
start = np.random.randint(0, len(val_data) - batch_size)
bind = np.arange(start, start+batch_size)

batch = val_data[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

#%%

