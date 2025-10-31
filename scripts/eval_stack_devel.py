#!/usr/bin/env python3
"""
This script is used to develop the new logging code for model fits.

There are two logging functions that need to result: fast logging and slow logging
1) fast logging (every 5 epochs):
- tracks the kernels in the model. the frontend, stem, convolutional kernels, and readout

2) slow logging (every 10 epochs):
- Use the eval stack to evaluate the model on the datasets
for BPS, CCNORM, Saccade, STA
then plot the results
"""

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import torch._dynamo 
torch._dynamo.config.suppress_errors = True # suppress dynamo errors

import sys
sys.path.append('..')

import numpy as np
from DataYatesV1 import enable_autoreload, get_free_device, prepare_data

import matplotlib.pyplot as plt
from eval.eval_stack_utils import load_single_dataset, get_stim_inds, evaluate_dataset, argmin_subpixel, argmax_subpixel
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()

device = get_free_device()

#%% Load an example model (this will be provided in the logging)
print("Discovering available models...")
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_6/checkpoints"
models_by_type = scan_checkpoints(checkpoint_dir, verbose=False)

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

# LOAD A MODEL
model_type = 'dense_concat_convgru'
model, model_info = load_model(
        model_type=model_type,
        model_index=0, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.model.eval()
model.model.convnet.use_checkpointing = False 

model = model.to(device)

#%% Fast Logging code
if hasattr(model.model, 'frontend') and hasattr(model.model.frontend, 'temporal_conv'):
    fig_frontend = plt.plot(model.model.frontend.temporal_conv.weight.squeeze().detach().cpu().T)
    plt.title('Frontend Kernels')

if hasattr(model.model.convnet, 'stem'):
    fig_stem = model.model.convnet.stem.components.conv.plot_weights()
    plt.title('Stem Kernels')

# plot the kernels in each layer of the convnet
layer_figs = []
for layer in model.model.convnet.layers:
    if hasattr(layer, 'components'):
        layer_figs.append(layer.components.conv.plot_weights(nrow=20))
    else:
        layer_figs.append(layer.main_block.components.conv.plot_weights(nrow=20))

# plot readouts
for readout in model.model.readouts[:1]:
    # run dummy input through readout
    readout(torch.randn(1, readout.features.in_channels, 1, 15, 15).to(device))
    fig_readout = readout.plot_weights(ellipse=False)

#%% SLOW LOGGING (takes substantially longer)

from eval.eval_stack_multidataset import eval_stack_single_dataset

# During training - simple and clean!
results = eval_stack_single_dataset(
    model=model,
    dataset_idx=7,
    rescale=True,
    analyses=['bps', 'ccnorm', 'saccade', 'sta', 'qc']
)



#%% PLOT STAS
from torchvision.utils import make_grid
sta_dict = results['sta']
N = len(sta_dict['peak_lag'])
num_lags = sta_dict['Z_STA_robs'].shape[0]
H = sta_dict['Z_STA_robs'].shape[1]
rf_pairs_full = []
rf_pairs = []
for cc in range(N):
    this_lag = sta_dict['peak_lag'][cc]
    sta_robs = sta_dict['Z_STA_robs'][:,:,:,cc]
    sta_rhat = sta_dict['Z_STA_rhat'][:,:,:,cc]
    # zscore each
    sta_robs = (sta_robs - sta_robs.mean((0,1))) / sta_robs.std((0,1))
    sta_rhat = (sta_rhat - sta_rhat.mean((0,1))) / sta_rhat.std((0,1))
    grid = make_grid(torch.concat([sta_robs, sta_rhat], 0).unsqueeze(1), nrow=num_lags, normalize=True, scale_each=False, padding=2, pad_value=1)
    grid = 0.2989 * grid[0:1,:,:] + 0.5870 * grid[1:2,:,:] + 0.1140 * grid[2:3,:,:] # convert to grayscale
    rf_pairs_full.append(grid)

    # do the same for the peak lag
    sta_robs = sta_dict['Z_STA_robs'][this_lag,:,:,cc]
    sta_rhat = sta_dict['Z_STA_rhat'][this_lag,:,:,cc]
    # zscore each
    sta_robs = (sta_robs - sta_robs.mean()) / sta_robs.std()
    sta_rhat = (sta_rhat - sta_rhat.mean()) / sta_rhat.std()
    grid = torch.stack([sta_robs, sta_rhat], 0).unsqueeze(1)
    grid = make_grid(grid, nrow=2, normalize=True, scale_each=False, padding=2, pad_value=1)
    grid = 0.2989 * grid[0:1,:,:] + 0.5870 * grid[1:2,:,:] + 0.1140 * grid[2:3,:,:] # convert to grayscale
    rf_pairs.append(grid)


# log the full spatio-temporal STAs for each Cell and model    
log_grid_full = make_grid(torch.stack(rf_pairs_full), nrow=3, normalize=True, scale_each=True, padding=2, pad_value=1)
plt.figure(figsize=(20, 20))
plt.imshow(log_grid_full.detach().cpu().permute(1, 2, 0).numpy())

# log the peak lag STAs for each Cell and model
log_grid_peak_lag = make_grid(torch.stack(rf_pairs), nrow=int(np.sqrt(N)), normalize=True, scale_each=True, padding=2, pad_value=1)
plt.figure(figsize=(20, 20))
plt.imshow(log_grid_peak_lag.detach().cpu().permute(1, 2, 0).numpy())

# these can be logged as images because they use make_grid

#%% Plot BPS per stimulus
bps = []
for k in results['bps']:
    if k in ['val', 'cids']:
        continue
    # plt.bar(np.arange(len(results['bps'][k]['bps'])), np.maximum(results['bps'][k]['bps'], -.1), label=k, alpha=0.5)
    plt.plot(np.arange(len(results['bps'][k]['bps'])), np.maximum(results['bps'][k]['bps'], -.1), label=k, alpha=0.5)

plt.legend()

cc = 0



#%% plot CCNORM
rbar = results['ccnorm']['rbar']
rhat_bar = results['ccnorm']['rbarhat']

good_samples = np.sum(~np.isnan(rbar), 0)
good_units = np.where(good_samples == good_samples.max())[0]

NC = len(good_units)
sx = int(np.sqrt(NC))
sy = int(np.ceil(NC / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= NC:
        axs.flatten()[i].axis('off')
        continue
    cc = good_units[i]
    s = np.where(np.isnan(rbar[:,cc]))[0][-1]
    axs.flatten()[i].plot(rbar[:,cc], 'k')
    axs.flatten()[i].plot(rhat_bar[:,cc], 'r')
    axs.flatten()[i].set_title(f'{cc}')
    # turn off
    axs.flatten()[i].axis('off')
    axs.flatten()[i].set_title(f'{cc}')
axs.flatten()[i].set_xlim(32, 150)


#%% Plot saccade triggered averages on BackImage
rbar = results['saccade']['backimage']['rbar']
rhat = results['saccade']['backimage']['rbarhat']
win = results['saccade']['backimage']['win']

N = rbar.shape[1]
sx = int(np.sqrt(N))
sy = int(np.ceil(N / sx))
fig_saccade, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
time_axis = np.arange(win[0], win[1])
for i in range(sx*sy):
    if i >= N:
        axs.flatten()[i].axis('off')
        continue
    axs.flatten()[i].plot(time_axis, rbar[:,i], 'k')
    m = rbar[:,i].mean()
    axs.flatten()[i].axhline(m, linestyle='--', color='k')
    axs.flatten()[i].axvline(0, linestyle='--', color='k')
    axs.flatten()[i].plot(time_axis, rhat[:,i], 'r')
    axs.flatten()[i].plot([0, 10], [m, m], 'k', linewidth=2)
    axs.flatten()[i].set_xlim(win[0], win[1]//2)
    axs.flatten()[i].set_title(f'{i}')
    # turn off
    axs.flatten()[i].axis('off')
    axs.flatten()[i].set_title(f'{i}')
# %%
