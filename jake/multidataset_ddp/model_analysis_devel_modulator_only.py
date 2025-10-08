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
from tqdm import tqdm
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
from eval_stack_utils import scan_checkpoints
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






#%% LOAD A MODEL

def load_model_by_type(model_type, checkpoint_dir):
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
    return model, model_info

model, model_info = load_model_by_type('modulator_only_convgru', checkpoint_dir)
model_full, model_vis_info = load_model_by_type('learned_res_small_gru', checkpoint_dir)
model_vis, model_vis_info = load_model_by_type('learned_res_small_none_gru', checkpoint_dir)


#%% Run bps analysis to find good cells / get STA
dataset_idx = 8
batch_size = 64 # keep small because things blow up fast!

train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
dataset_cids = dataset_config.get('cids', [])

stim_type = 'backimage'
inds = get_stim_inds(stim_type, train_data, val_data)

dataset = val_data.shallow_copy()
dataset.inds = inds

result = evaluate_dataset(
    model, dataset, inds, dataset_idx, batch_size, "backimage"
    )
robs, rhat, bps = result['robs'], result['rhat'], result['bps']

result_vis = evaluate_dataset(
    model_vis, dataset, inds, dataset_idx, batch_size, "backimage"
    )
robs_vis, rhat_vis, bps_vis = result_vis['robs'], result_vis['rhat'], result_vis['bps']

result_full = evaluate_dataset(
    model_full, dataset, inds, dataset_idx, batch_size, "backimage"
    )
robs_full, rhat_full, bps_full = result_full['robs'], result_full['rhat'], result_full['bps']

#%%
plt.figure()
plt.subplot(2,2,1)
plt.plot(bps, bps_vis, 'k.')
plt.plot([-1, 3], [-1, 3], 'k--')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.xlabel('Modulator Only BPS')
plt.ylabel('Vision Only BPS')
plt.box(False)

plt.subplot(2,2,2)
plt.plot(bps_full, bps_vis, 'k.')
plt.plot([-1, 3], [-1, 3], 'k--')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.ylabel('Vision Only BPS')
plt.xlabel('Full Model BPS')
plt.box(False)

plt.subplot(2,2,4)
plt.plot(bps_full, bps, 'k.')
plt.plot([-1, 3], [-1, 3], 'k--')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.ylabel('Modulator Only BPS')
plt.xlabel('Full Model BPS')
plt.box(False)
plt.show()

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

output = model.model(stimulus=None, dataset_idx=dataset_idx, behavior=batch['behavior'])

del output
torch.cuda.empty_cache()

#%%
device = model.device # double check in case you ran cells out of order

batch_size = 256
# randomly sample a batch
start = np.random.randint(0, len(val_data) - batch_size)
bind = np.arange(start, start+batch_size)


batch = val_data[bind]
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

output = model.model(stimulus=None, dataset_idx=dataset_idx, behavior=batch['behavior'])
_ = plt.plot(output.detach().cpu().numpy())

del output
torch.cuda.empty_cache()


#%% Run saccade analysis
from DataYatesV1 import get_session
from eval_stack_utils import detect_saccades_from_session

dataset_name = model.names[dataset_idx]
sess = get_session(*dataset_name.split('_'))
saccades = detect_saccades_from_session(sess)

#%% get saccade robs function

from eval_stack_utils import run_model

def get_sac_eval(model, train_data, val_data, stim_type='backimage', win = (-50, 100)):
    
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

    n_cells = len(model.model.readouts[dataset_idx].bias)
    robs_sac = np.nan*np.zeros((n_sac, nbins, n_cells))
    eye_vel_sac = np.nan*np.zeros((n_sac, nbins, 1))
    pred_sac = np.nan*np.zeros((n_sac, nbins, n_cells))
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

            output = run_model(model, batch, dataset_idx)
            pred = output['rhat']

            pred = torch.exp(pred)

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


    del pred, output
    torch.cuda.empty_cache()
    return {'robs': robs_sac, 'pred': pred_sac, 'dfs': dfs_sac, 'eyevel': eye_vel_sac, 'saccade_info': saccade_info, 'time_previous': time_previous, 'time_next': time_next}

#%%
win = (-20, 100)
sac_eval = get_sac_eval(model, train_data, val_data, stim_type='backimage',win = win)
sac_eval_vis = get_sac_eval(model_vis, train_data, val_data, stim_type='backimage',win = win)
sac_eval_full = get_sac_eval(model_full, train_data, val_data, stim_type='backimage',win = win)

#%% confirm alignment
time_next = sac_eval['time_next']
time_previous = sac_eval['time_previous']
eye_vel_sac = sac_eval['eyevel']

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
           cmap='viridis',
           vmin=0, vmax=100,
           extent=[win[0]*dt, win[1]*dt, 0, ev.shape[0]],
           origin='lower')           
# plt.plot(tn, np.arange(len(tn)), 'r.')
# plt.xlim(-.1, .6)

cc = 0

#%%
%matplotlib inline
cc += 1
r0 = sac_eval['robs'][valid, :, cc]/dt  # pick the first velocity channel
r_mod0 = sac_eval['pred'][valid, :, cc]/dt  # pick the first velocity channel
r_vis0 = sac_eval_vis['pred'][valid, :, cc]/dt  # pick the first velocity channel
r_full0 = sac_eval_full['pred'][valid, :, cc]/dt  # pick the first velocity channel

order = np.argsort(tn0)
vmin = np.nanmin(r0)
vmax = np.nanmax(r0)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, r in enumerate([r0, r_mod0, r_vis0, r_full0]):
    ax = axs.flatten()[i]
    ax.imshow(r[order, :],
        aspect='auto',
        interpolation='none',
        cmap='gray_r',
        vmin=vmin, vmax=vmax,
        extent=[win[0]*dt, win[1]*dt, 0, r.shape[0]],
        origin='lower')
    ax.set_xlim(-.1, .35)

axs[0].set_ylabel('Saccade Index')
axs[0].set_xlabel('Time from saccade onset (s)')
axs[0].set_title(f'cc {cc}')
axs[1].set_title('Modulator Only')
axs[2].set_title('Vision Only')
axs[3].set_title('Full Model')


saccade_info = sac_eval['saccade_info']
sac_dur = np.array([s['end_time']-s['start_time'] for s in saccade_info])
# plt.hist(sac_dur, np.linspace(0, .1, 100))
# plt.xlabel('Saccade Duration (time)')

order = np.argsort(sac_dur[valid])
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, r in enumerate([r0, r_mod0, r_vis0, r_full0]):
    ax = axs.flatten()[i]
    ax.imshow(r[order, :],
        aspect='auto',
        interpolation='none',
        cmap='gray_r',
        vmin=vmin, vmax=vmax,
        extent=[win[0]*dt, win[1]*dt, 0, r.shape[0]],
        origin='lower')
    ax.set_xlim(-.1, .15)

axs[0].set_ylabel('Saccade Index')
axs[0].set_xlabel('Time from saccade onset (s)')
axs[0].set_title(f'cc {cc}')
axs[1].set_title('Modulator Only')
axs[2].set_title('Vision Only')
axs[3].set_title('Full Model')

#%%

edges = np.percentile(sac_dur, [0, 25, 50, 75, 100])

iix = (sac_dur[valid] > edges[0]) & (sac_dur[valid] < edges[1])
plt.figure()
plt.plot(np.nanmean(r0[iix,:], 0))

iix = (sac_dur[valid] > edges[1]) & (sac_dur[valid] < edges[2])
plt.plot(np.nanmean(r0[iix,:], 0))

iix = (sac_dur[valid] > edges[2]) & (sac_dur[valid] < edges[3])
plt.plot(np.nanmean(r0[iix,:], 0))

iix = (sac_dur[valid] > edges[3]) & (sac_dur[valid] < edges[4])
plt.plot(np.nanmean(r0[iix,:], 0))
plt.xlim(10, 60)

#%%

#%%
velocity = [s['A'] for s in saccade_info]
amplitude = [np.hypot(s['end_x']-s['start_x'], s['end_y']-s['start_y']) for s in saccade_info]
velocity = np.array(velocity)
amplitude = np.array(amplitude)

plt.plot(amplitude, velocity, 'k.', alpha=.1)
plt.xlabel('Saccade Amplitude (deg)')
plt.ylabel('Saccade Velocity (deg/s)')

#%%
%matplotlib inline
time = np.array([s['start_time'] for s in saccade_info])
plt.figure()
plt.plot(time, amplitude/velocity, '.', alpha=.1)
plt.show()

#%%

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
plt.title('Unit Response')
cc = 0
#%% one by one raw
sx = int(np.sqrt(rbar.shape[1]))
sy = int(np.ceil(rbar.shape[1]/sx))
fig, axs = plt.subplots(sy, sx, figsize=(26, 26))

for i in range(rbar.shape[1]):
    ax = axs.flatten()[i]
    ax.plot(rbar[:,i], 'k', label='Data')
    ax.plot(predbar[:,i], 'r', label='Model')
    ax.set_title(f'Unit {i}')
# plt.plot(rbar[:,cc], 'k', label='Data')
# plt.plot(predbar[:,cc], 'r', label='Model')
# plt.legend()


#%%
gaborium_inds = torch.concatenate([
            train_data.get_dataset_inds('gaborium'),
            val_data.get_dataset_inds('gaborium')
        ], dim=0)

dataset = train_data.shallow_copy()
# set indices to be the gaborium inds
dataset.inds = gaborium_inds

gaborium_result = evaluate_dataset(
    model, dataset, gaborium_inds, dataset_idx, batch_size, "Gaborium"
    )
gaborium_robs, gaborium_rhat, gaborium_bps = gaborium_result['robs'], gaborium_result['rhat'], gaborium_result['bps']

#%% get stas 

sta_dict = get_sta_ste(model, gaborium_robs, gaborium_rhat, didx=dataset_idx, lags=list(range(16)), combine_train_test=True)
#%%
plot_stas(sta_dict, lag=4, normalize=False)


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

