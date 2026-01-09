## !/usr/bin/env python3


#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs
import torch
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1 import get_clock_functions
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import torch
from eval.eval_stack_utils import run_model

import matplotlib.pyplot as plt
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
# checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_history/checkpoints"
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
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
model_type = 'resnet_none_convgru'
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

#%%

dataset_idx = 10
print(f"Dataset {dataset_idx}: {model.names[dataset_idx]}")
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

dataset = train_data.shallow_copy()
dataset.inds = inds

# Some warm up plotting and getting key variables
dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1


robs = np.nan*np.zeros((NT, T, NC))
rhat = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))
for itrial in range(NT):
    ix = trials[itrial] == trial_inds
    ix = ix & fixation

    stim_inds = np.where(ix)[0]
    stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
    stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
    # behavior = dataset.dsets[dset_idx]['behavior'][ix]
    # behavior = None
    # , 'behavior': behavior

    out = run_model(model, {'stim': stim}, dataset_idx=dataset_idx)

    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    rhat[itrial][psth_inds] = out['rhat'].detach().cpu().numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)

good_trials = fix_dur > 40
robs = robs[good_trials]
rhat = rhat[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]
 
ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 60)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 60)



#%% plot all unit PSTHs
sx = int(np.sqrt(NC))
sy = int(np.ceil(NC / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
rhos = []
ve_rate = []
ve_trial = []
for i in range(sx*sy):
    if i >= NC:
        axs.flatten()[i].axis('off')
        continue
    # axs.flatten()[i].imshow(robs[:, :, i][ind], aspect='auto', interpolation='none', cmap='gray_r')
    rbar = np.nanmean(robs[:, :, i][ind], 0)[:120]
    rhatbar = np.nanmean(rhat[:, :, i][ind], 0)[:120]
    iix = np.isfinite(rbar) & np.isfinite(rhatbar)
    rbar = rbar[iix]
    rhatbar = rhatbar[iix]
    resid_rbar = robs[:,:80,i] - rbar[None,:80].repeat(robs.shape[0], axis=0)
    # zscore
    rbar = (rbar - rbar.mean()) / rbar.std()
    rhatbar = (rhatbar - rhatbar.mean()) / rhatbar.std()
    
    ve = 1 - np.nanvar(resid_rbar)/np.nanvar(robs[:,:80,i])
    ve_rate.append(ve)

    resid_trial = robs[:,:80,i] - rhat[:,:80,i]
    ve = 1 - np.nanvar(resid_trial)/np.nanvar(robs[:,:80,i])
    ve_trial.append(ve)

    ax = axs.flatten()[i]
    ax.plot(rbar, 'k')
    ax.plot(rhatbar, 'r')
    rho = np.corrcoef(rbar, rhatbar)[0,1]
    rhos.append(rho)
    ax.set_title(f'{i} rho={rho:.2f}')
    ax.axis('off')

ax.set_xlim(0, 60)
rhos = np.array(rhos)
ve_rate = np.array(ve_rate)
ve_trial = np.array(ve_trial)

plt.savefig(f"../figures/fixrsvp_{dataset.dsets[dset_idx].metadata['sess'].name}_{dataset_idx}_PSTH.pdf")


#%%
plt.plot(np.maximum(ve_rate, -.2), np.maximum(ve_trial, -.2),'.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.axis('equal')
# plt.xlim(-.2, 1)
# plt.ylim(-.2, 1)
# plt.plot(np.sort(ve_rate), '.')
#%%


x = robs[:, :, cc][ind]

rmu = np.nanmean(rhat, (0,1))

y = rhat[:, :, cc][ind]

y = (y - np.nanmean(y)) / np.nanstd(y) * np.nanstd(x) + np.nanmean(x)

#%%
i = -1
#%%

unit_ind = np.argsort(rhos)[::-1]
i += 1
cc = unit_ind[i]
x = robs[:, :, cc][ind]
y = rhat[:, :, cc][ind]
# affine transform y to match x
y = (y - np.nanmean(y)) / np.nanstd(y) * np.nanstd(x) + np.nanmean(x)

fig, axs = plt.subplots(1,2, figsize=(10,3), sharex=True, sharey=False)
axs[0].imshow(x, aspect='auto', cmap='gray_r', interpolation='none')
axs[1].imshow(y, aspect='auto', cmap='gray_r', interpolation='none')
axs[0].set_title(f'Observed {cc}')
axs[1].set_title(f'Predicted {cc} rho={rhos[cc]:.2f}')
plt.show()


n_to_plot = 10
rbar = np.nanmean(x,0)[:80]
rhatbar = np.nanmean(y, 0)[:80]
rhatstd = np.nanstd(y, 0)[:80]
iix = np.isfinite(rbar) & np.isfinite(rhatbar)
rbar = rbar[iix]
rhatbar = rhatbar[iix]

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.plot(rbar, 'k')
ax.plot(rhatbar, 'r')
ax.fill_between(np.arange(len(rhatbar)), rhatbar - rhatstd, rhatbar + rhatstd, color='r', alpha=0.2)
rho = np.corrcoef(rbar, rhatbar)[0,1]


# %%

#%%

def run_core(model, stimulus=None, behavior=None, history=None):
        """
        Forward pass with spike history processing.

        Args:
            stimulus: Visual stimulus tensor with shape (N, C, T, H, W) or None
            dataset_idx: Index of the dataset (determines adapter/readout)
            behavior: Optional behavioral data with shape (N, n_vars)
            history: Spike history tensor with shape (N, num_lags, n_units) or (N, num_lags * n_units)

        Returns:
            Tensor: Model predictions with shape (N, n_units_for_dataset)
        """
        x = model.adapters[0](stimulus)
        x = stimulus

        if x is None:
            # Modulator-only mode: create minimal features for modulator
            # Use 0 channels so concat modulator only returns behavior embedding
            B = behavior.shape[0]
            device = next(model.parameters()).device
            x = torch.ones(B, 0, 1, 1, 1, device=device, dtype=behavior.dtype)

        x = model.core_forward(x, behavior)

        return x


import torch

def spatial_ssi_population(y, dt=1.0, eps=1e-8, log_base=2.0):
    """
    Spatial single-spike information from a rate map.

    Args:
        y: rates, shape [T, N, H, W]. Must be >= 0. Interpreted as spikes/sec unless dt=1 and you treat as spikes/bin.
        dt: bin width in seconds (used to convert rates -> expected spikes per bin).
        eps: numerical stability.
        log_base: 2.0 for bits, torch.e for nats.

    Returns:
        ispikepop: population bits/spike (avg info of a random spike from the whole trial)
        iratepop:  population bits/sec (info rate over the whole trial)
    """
    T, N, H, W = y.shape
    P = H * W

    r = y.reshape(T, N, P)                      # [T, N, P]
    rbar = r.mean(dim=2)                        # [T, N]     mean rate over space

    g = r / (rbar[..., None] + eps)             # [T, N, P]
    if log_base == 2.0:
        logg = torch.log2(g + eps)
    else:
        logg = torch.log(g + eps)               # nats if log_base is e

    I_tn = (g * logg).mean(dim=2)               # [T, N] bits/spike at each time

    # Expected spikes per bin for each (t,n)
    spikes_tn = rbar * dt                       # [T, N]

    total_bits   = torch.sum(spikes_tn * I_tn)  # scalar, bits over the whole trial
    total_spikes = torch.sum(spikes_tn)         # scalar, expected spikes over the whole trial

    ispikepop = total_bits / (total_spikes + eps)        # bits/spike
    iratepop  = total_bits / (T * dt)                    # bits/sec

    return ispikepop, iratepop, I_tn


#%%
itrial +=1
if itrial >= NT:
    itrial = 0
ix = trials[itrial] == trial_inds
ix = ix & fixation

stim_inds = np.where(ix)[0]
stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
behavior = dataset.dsets[dset_idx]['behavior'][ix]

x = run_core(model.model, stim.to(device), behavior.to(device))

units = []
for readout in model.model.readouts:
    y = readout.features(x[:,:,-1])
    units.append(y)

y = torch.cat(units, dim=1)
y = model.model.activation(y)

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(y[:,i,7,:].detach().cpu().T, aspect='auto', interpolation='none')
    plt.axis('off')
    
plt.show()


# Example:
# ispikepop, iratepop = spatial_ssi_population(y, dt=0.01)  # if 10 ms bins

ispikepop, iratepop, I_t = spatial_ssi_population(y, dt=1/240)     # [T, N]  SSI per time

#%%
# sess = dataset.dsets[dset_idx].metadata['sess']

# trials = dataset.dsets[dset_idx].covariates['trial_inds'].unique().numpy().astype(int)
# fixrsvp_trials = [FixRsvpTrial(sess.exp['D'][iT], sess.exp['S']) for iT in trials]

# images = get_rsvp_fix_stim()
# ims = []
# for i in range(n_frames):
#     im_id = self.image_ids[idx[i]]
#     im = images[f'im{im_id:02d}'].mean(axis=2).astype(np.uint8)
#     pos = self.positions[idx[i]]


#     im_tex, alpha_tex = gen_gauss_image_texture(im, self.bkgnd)
#     im, _ = place_gauss_image_texture(im_tex, alpha_tex, 
#                                         pos, self.radius, self.center_pix,
#                                         self.bkgnd, self.ppd, roi=roi[i], binSize=stride)

#     im = (im + .5 + self.bkgnd).astype(np.uint8)
#     ims.append(im)

# ims = np.stack(ims, axis=0)
#%%
plt.figure()
_ = plt.plot(I_t.sum(1).detach().cpu())
plt.show()
#%%
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye

dt = 1/120
frate = 30 # stimulus frames per second
ppd = 37.50476617
frames_per_im = 6#int(1/dt/frate)
# im_tex, alpha_tex = get_fixrsvp_stack(frames_per_im=frames_per_im)
full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im)
print(full_stack.shape)

#%%
itrial +=1
if itrial >= NT:
    itrial = 0
ix = trials[itrial] == trial_inds
ix = ix & fixation
stim_inds = np.where(ix)[0]
stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
eyepos = dataset.dsets[dset_idx]['eyepos'][ix]


eye_norm = eye_deg_to_norm(eyepos, ppd, full_stack.shape[1:3])



#%%
eye_norm = eye_deg_to_norm(eyepos, ppd, full_stack.shape[1:3])
eye_movie = shift_movie_with_eye(
    torch.from_numpy(full_stack[:stim.shape[0]]).float(),
    eye_norm,
    out_size=(101, 101),          # (outH,outW)
    center=(0.0, 0.0),            # (cx,cy) in [-1,1]
    scale_factor=1.0,
    mode="bilinear")

#%%
def save_sidebyside_movie(movie1, movie2, save_path, fps=30,
                          title1='Dataset Stim', title2='Reconstructed',
                          offset1=0, offset2=0):
    """
    Save a side-by-side comparison video of two movies.

    Parameters
    ----------
    movie1, movie2 : torch.Tensor or np.ndarray
        Movies to compare. Can be (T, H, W) or (T, C, H, W) or (T, C, L, H, W).
        Will extract the first frame if multiple lags/channels.
    save_path : str
        Output path for the video (e.g., 'comparison.mp4')
    fps : int
        Frames per second
    title1, title2 : str
        Titles for each panel
    offset1, offset2 : int
        Frame offsets for each movie (to align them)
    """
    from matplotlib.animation import FFMpegWriter

    # Convert to numpy and extract single frames if needed
    def to_2d_movie(m):
        if hasattr(m, 'detach'):
            m = m.detach().cpu().numpy()
        while m.ndim > 3:
            m = m[:, 0]  # take first channel/lag
        return m

    m1 = to_2d_movie(movie1)
    m2 = to_2d_movie(movie2)

    # Apply offsets and find common length
    m1 = m1[offset1:]
    m2 = m2[offset2:]
    T = min(len(m1), len(m2))
    m1, m2 = m1[:T], m2[:T]

    # Get vmin/vmax for consistent scaling
    vmin1, vmax1 = np.percentile(m1, [1, 99])
    vmin2, vmax2 = np.percentile(m2, [1, 99])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=2)

    writer = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'],
                          bitrate=8000)

    with writer.saving(fig, save_path, dpi=100):
        for t in range(T):
            ax1.clear()
            ax2.clear()

            ax1.imshow(m1[t], cmap='gray', vmin=vmin1, vmax=vmax1)
            ax1.set_title(f'{title1}\nFrame {t}/{T}')
            ax1.axis('off')

            ax2.imshow(m2[t], cmap='gray', vmin=vmin2, vmax=vmax2)
            ax2.set_title(f'{title2}\nFrame {t}/{T}')
            ax2.axis('off')

            writer.grab_frame()

    plt.close(fig)
    print(f"Saved comparison movie to {save_path}")

# Save the comparison
# stim is (T, C, L, H, W), eye_movie is (T, H, W)
save_sidebyside_movie(
    stim[:, 0, 0],  # extract first channel/lag -> (T, H, W)
    eye_movie,
    save_path='../figures/fixrsvp_stim_comparison.mp4',
    fps=10,  # slow enough to see details
    title1='Dataset Stim',
    title2='Reconstructed',
    offset1=0,
    offset2=6  # your offset to align the movies
)

#%%
# #%%
# model.model.readouts[0].compute_gaussian_mask()

# # Route through appropriate readout
# output = self.readouts[dataset_idx](x)

# # Process history through MLP
# # Flatten history if needed: (B, num_lags, n_units) -> (B, num_lags * n_units)
# if history is not None:
#     if history.dim() == 3:
#         B, num_lags, n_units = history.shape
#         history = history.reshape(B, num_lags * n_units)
#     history_output = self.spike_history[dataset_idx](history)
#     # Combine with readout output
#     output = output + history_output

# # Apply activation function
# output = self.activation(output)

# # Add baseline if enabled
# if self.baseline_enabled:
#     baseline_output = self.baseline_activation(self.baselines[dataset_idx])
#     output = output + baseline_output

# return output
#%%
for dataset_idx in range(len(model.names)):
    print(f"Dataset {dataset_idx}: {model.names[dataset_idx]}")
    
    try:    
        train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
        
        # Combine all indices (train + validation) for maximum data (we don't tend to train on gratings so this should be okay)
    
        inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

        dataset = train_data.shallow_copy()
        dataset.inds = inds

        # Some warm up plotting and getting key variables
        dset_idx = inds[:,0].unique().item()
        trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
        trials = np.unique(trial_inds)

        NC = dataset.dsets[dset_idx]['robs'].shape[1]
        T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
        NT = len(trials)

        fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

        #%%
        robs = np.nan*np.zeros((NT, T, NC))
        rhat = np.nan*np.zeros((NT, T, NC))
        eyepos = np.nan*np.zeros((NT, T, 2))
        fix_dur =np.nan*np.zeros((NT,))
        for itrial in range(NT):
            ix = trials[itrial] == trial_inds
            ix = ix & fixation

            stim_inds = np.where(ix)[0]
            stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
            stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
            behavior = dataset.dsets[dset_idx]['behavior'][ix]

            out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)

            psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
            robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
            rhat[itrial][psth_inds] = out['rhat'].detach().cpu().numpy()
            eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
            fix_dur[itrial] = len(psth_inds)

        #%%
        good_trials = fix_dur > 40
        robs = robs[good_trials]
        rhat = rhat[good_trials]
        eyepos = eyepos[good_trials]
        fix_dur = fix_dur[good_trials]

        ind = np.argsort(fix_dur)[::-1]
        plt.subplot(1,2,1)
        plt.imshow(eyepos[ind,:,0])
        plt.xlim(0, 60)
        plt.subplot(1,2,2)
        plt.imshow(np.nanmean(robs,2)[ind])
        plt.xlim(0, 60)

        sx = int(np.sqrt(NC))
        sy = int(np.ceil(NC / sx))
        fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
        for i in range(sx*sy):
            if i >= NC:
                axs.flatten()[i].axis('off')
                continue
            # axs.flatten()[i].imshow(robs[:, :, i][ind], aspect='auto', interpolation='none', cmap='gray_r')
            rbar = np.nanmean(robs[:, :, i][ind], 0)[:80]
            rhatbar = np.nanmean(rhat[:, :, i][ind], 0)[:80]
            iix = np.isfinite(rbar) & np.isfinite(rhatbar)
            rbar = rbar[iix]
            rhatbar = rhatbar[iix]
            # zscore
            rbar = (rbar - rbar.mean()) / rbar.std()
            rhatbar = (rhatbar - rhatbar.mean()) / rhatbar.std()
            ax = axs.flatten()[i]
            ax.plot(rbar, 'k')
            ax.plot(rhatbar, 'r')
            rho = np.corrcoef(rbar, rhatbar)[0,1]
            ax.set_title(f'{i} rho={rho:.2f}')
            ax.axis('off')

        ax.set_xlim(0, 60)

        plt.savefig(f"../figures/fixrsvp_{dataset.dsets[dset_idx].metadata['sess'].name}_{dataset_idx}_PSTH.pdf")


        # load session
        sess = dataset.dsets[dset_idx].metadata['sess']
        ks_results = sess.ks_results
        st = ks_results.spike_times
        clu = ks_results.spike_clusters

        trials = dataset.dsets[dset_idx].covariates['trial_inds'].unique().numpy().astype(int)
        fixrsvp_trials = [FixRsvpTrial(sess.exp['D'][iT], sess.exp['S']) for iT in trials]
        ptb2ephys, _ = get_clock_functions(sess.exp)
        
        # DEBUGGING
        # i_trial = 0
        # #%%
        # i_trial += 1
        # if i_trial >= len(trials):
        #     i_trial = 0
        # this_trial = trials[i_trial]

        # ephys_start = sess.exp['D'][this_trial]['START_EPHYS']
        # ephys_end = sess.exp['D'][this_trial]['END_EPHYS']

        # ix = this_trial == trial_inds
        # print(f"Trial {this_trial} has {np.sum(ix)} frames")

        # eyepos = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
        # stim = dataset.dsets[dset_idx]['stim'][ix]
        # robs = dataset.dsets[dset_idx]['robs'][ix].numpy()


        # plt.figure(figsize=(10,5))
        # plt.subplot(3,1,1)
        # grid = make_grid(stim, nrow=20, normalize=True, scale_each=True, padding=2, pad_value=1)
        # plt.imshow(grid.detach().cpu().permute(1, 2, 0).numpy(), aspect='auto', interpolation='none')
        # plt.axis('off')
        # plt.title(f"Trial {this_trial}")

        # plt.subplot(3,1,2)
        # plt.imshow(robs.T, aspect='auto',cmap='gray_r', interpolation='none', extent=[dataset.dsets[dset_idx]['t_bins'][ix][0], dataset.dsets[dset_idx]['t_bins'][ix][-1], 0, robs.shape[1]], origin='lower')
        # yd = plt.gca().get_ylim()
        # plt.gca().twinx
        # st_ix = (st >= ephys_start) & (st <= ephys_end)

        # for i, cid in enumerate(dataset.dsets[dset_idx].metadata['cids']):
        #     iix = st_ix * (clu == cid)
        #     plt.plot(st[iix], i*np.ones(np.sum(iix)), 'g.', markersize=1)

        # plt.ylim(yd)

        # plt.gca().twinx()
        # plt.plot(ephys_start + sess.exp['D'][this_trial]['eyeSmo'][:,0], sess.exp['D'][this_trial]['eyeSmo'][:,1])
        # plt.plot(dataset.dsets[dset_idx]['t_bins'][ix], eyepos[:,0], 'r')
        # plt.ylim(-2, 2)
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][0], color='k', linestyle='--')
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][-1], color='k', linestyle='--')

        # t0 = ptb2ephys(sess.exp['D'][this_trial]['PR']['NoiseHistory'][0][0])
        # plt.axvline(t0, color='y', linestyle='--')

        # plt.gca().twinx()
        # plt.plot(ptb2ephys(fixrsvp_trials[i_trial].flip_times), fixrsvp_trials[i_trial].image_ids, 'm.')
        # xd = plt.xlim()

        # plt.subplot(3,1,3)
        # stim_inds = np.where(ix)[0]
        # stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
        # stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
        # behavior = dataset.dsets[dset_idx]['behavior'][ix]

        # from eval.eval_stack_utils import run_model
        # out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)
        # # out['rhat']

        # plt.imshow(out['rhat'].detach().cpu().numpy().T, aspect='auto',cmap='gray_r', interpolation='none', extent=[dataset.dsets[dset_idx]['t_bins'][ix][0], dataset.dsets[dset_idx]['t_bins'][ix][-1], 0, robs.shape[1]], origin='lower')
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][0], color='k', linestyle='--')
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][-1], color='k', linestyle='--')
        # plt.xlim(xd)

        # #%%

        # # [plt.axvline(t, color='gray', alpha=.2) for t in dataset.dsets[dset_idx]['t_bins'][ix]]

        # --- helper: draw ONE trial into two axes on the current figure ---
        def draw_trial(ax_top, ax_middle, ax_bottom, this_trial, dset_idx, model, dataset_idx):
            
            ix = (trial_inds == this_trial)
            eyepos = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
            stim   = dataset.dsets[dset_idx]['stim'][ix]
            robs   = dataset.dsets[dset_idx]['robs'][ix].numpy()
            t_bins = dataset.dsets[dset_idx]['t_bins'][ix]
            ephys_start = sess.exp['D'][this_trial]['START_EPHYS']
            ephys_end   = sess.exp['D'][this_trial]['END_EPHYS']

            # --- top panel: stimulus grid ---
            grid = make_grid(stim, nrow=20, normalize=True, scale_each=True, padding=2, pad_value=1)
            ax_top.imshow(grid.detach().cpu().permute(1, 2, 0).numpy(),
                        aspect='auto', interpolation='none')
            ax_top.set_axis_off()
            ax_top.set_title(f"Trial {this_trial}  |  {np.sum(ix)} frames", fontsize=10)

            # --- bottom panel: spikes + eye traces ---
            im = ax_middle.imshow(
                robs.T, aspect='auto', cmap='gray_r', interpolation='none',
                extent=[t_bins[0], t_bins[-1], 0, robs.shape[1]], origin='lower'
            )
            yd = ax_middle.get_ylim()

            # overlay spikes per cluster (assumes st, clu, and st_ix logic in scope)
            st_ix = (st >= ephys_start) & (st <= ephys_end)
            for i, cid in enumerate(dataset.dsets[dset_idx].metadata['cids']):
                iix = st_ix & (clu == cid)
                # tiny points for raster
                ax_middle.plot(st[iix], i*np.ones(np.sum(iix)), 'g.', markersize=1)

            ax_middle.set_ylim(yd)
            ax_middle.set_ylabel("Neuron", fontsize=8)
            ax_middle.set_xlabel("Time (s)", fontsize=8)

            # second y-axis for eye traces
            ax_eye = ax_middle.twinx()
            ax_eye.plot(ephys_start + sess.exp['D'][this_trial]['eyeSmo'][:,0],
                        sess.exp['D'][this_trial]['eyeSmo'][:,1])
            ax_eye.plot(t_bins, eyepos[:,0], 'r')
            ax_eye.set_ylim(-2, 2)
            ax_eye.set_yticks([])
            ax_eye.set_xlim(t_bins[0]-.4, t_bins[0]+1)

            # trial window & markers
            ax_middle.axvline(t_bins[0], color='k', linestyle='--', linewidth=0.8)
            ax_middle.axvline(t_bins[-1], color='k', linestyle='--', linewidth=0.8)

            t0 = ptb2ephys(sess.exp['D'][this_trial]['PR']['NoiseHistory'][0][0])
            ax_middle.axvline(t0, color='y', linestyle='--', linewidth=0.8)

            ax_frames = ax_middle.twinx()
            fixrsvp_trial = FixRsvpTrial(sess.exp['D'][this_trial], sess.exp['S'])
            ax_frames.plot(ptb2ephys(fixrsvp_trial.flip_times), fixrsvp_trial.image_ids, 'm.')
            ax_frames.set_yticks([])
            ax_frames.set_ylim(0, 25)

            stim_inds = np.where(ix)[0]
            stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
            stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
            behavior = dataset.dsets[dset_idx]['behavior'][ix]

            out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)
            print(out['rhat'].shape)
            
            ax_bottom.imshow(out['rhat'].detach().cpu().numpy().T, aspect='auto',cmap='gray_r', interpolation='none',
                            extent=[t_bins[0], t_bins[-1], 0, robs.shape[1]], origin='lower'
            )
            ax_bottom.axvline(t_bins[0], color='k', linestyle='--', linewidth=0.8)
            ax_bottom.axvline(t_bins[-1], color='k', linestyle='--', linewidth=0.8)
            ax_bottom.set_xlim(t_bins[0]-.4, t_bins[0]+1)

        # --- PDF builder: pack multiple trials per page ---
        pdf_path = f"../figures/FixRsvp_{dataset.dsets[dset_idx].metadata['sess'].name}_{model_type}.pdf"
        trials_per_page = 3
        page_size = (8.5, 11)   # inches (Letter)
        test_mode = False
        if test_mode:
            num_trials = 5
        else:
            num_trials = len(trials)
        with PdfPages(pdf_path) as pdf:
            # loop in pages
            for start in range(0, num_trials, trials_per_page):
                end = min(start + trials_per_page, len(trials))
                n_on_page = end - start

                fig = plt.figure(figsize=page_size)
                outer = gridspec.GridSpec(n_on_page, 1, hspace=0.35, top=0.96, bottom=0.06, left=0.06, right=0.97)

                for row, this_trial in enumerate(trials[start:end]):
                    # each trial block gets a 2x1 inner grid (top: stim, bottom: spikes+eye)
                    inner = gridspec.GridSpecFromSubplotSpec(
                        3, 1, subplot_spec=outer[row],
                        height_ratios=[1, 1.2, 1.2], hspace=0.15
                    )
                    ax_top = fig.add_subplot(inner[0])
                    ax_mid = fig.add_subplot(inner[1])
                    ax_bot = fig.add_subplot(inner[2], sharex=ax_mid)
                    try:
                        draw_trial(ax_top, ax_mid, ax_bot, this_trial, dset_idx, model, dataset_idx)
                        print(f"  ✓ Trial {this_trial} drawn successfully")
                    except Exception as e:
                        print(f"  ✗ Trial {this_trial} failed:")
                        print(f"    Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                pdf.savefig(fig)
                plt.close(fig)

        print(f"Wrote {pdf_path}")
    
    except Exception as e:
        print(f"✗ Failed to load dataset:")
        import traceback
        traceback.print_exc()

# %%
