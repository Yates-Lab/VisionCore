## !/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch

enable_autoreload()
device = get_free_device()

#%%

sessions = get_complete_sessions()

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)


#%%

dataset_idx = 0
batch_size = 64 # keep small because things blow up fast!

sess = sessions[dataset_idx]
# get saccade times
saccade_onsets = torch.tensor([s.start_time for s in sess.saccades])
saccade_offsets = torch.tensor([s.end_time for s in sess.saccades])
dx = torch.tensor([s.end_x - s.start_x for s in sess.saccades])
dy = torch.tensor([s.end_y - s.start_y for s in sess.saccades])
amp = torch.hypot(dx, dy)
vel = torch.tensor([s.velocity for s in sess.saccades])
duration = saccade_offsets - saccade_onsets

plt.subplot(2,2,1)
plt.scatter(duration, amp, alpha=.1, s=1)
plt.xlim(0, .1)
plt.ylim(0, 10)
plt.xlabel('Saccade Duration (s)')
plt.ylabel('Saccade Amplitude (deg)')

plt.subplot(2,2,2)
plt.scatter(amp, vel, alpha=.1, s=1)
plt.ylim(0, 1000)
plt.xlim(0, 10)
plt.xlabel('Saccade Amplitude (deg)')
plt.ylabel('Saccade Velocity (deg/s)')

plt.subplot(2,2,3)
plt.scatter(dx, dy, alpha=.1, s=1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('Saccade dX (deg)')
plt.ylabel('Saccade dY (deg)')

plt.subplot(2,2,4)
plt.scatter(amp, duration, alpha=.1, s=1)
plt.xlim(0, 10)
plt.ylim(0, .1)
plt.xlabel('Saccade Amplitude (deg)')
plt.ylabel('Saccade Duration (s)')

plt.tight_layout()
plt.show()
#%%

train_data, val_data, dataset_config = prepare_data(dataset_configs[dataset_idx])
dataset_cids = dataset_config.get('cids', [])

stim_type = 'backimage'
protocal_types = {'backimage': 'BackImage'}
stim_indices = torch.concatenate([train_data.get_dataset_inds(stim_type), val_data.get_dataset_inds(stim_type)], dim=0)
dataset = val_data.shallow_copy()
dataset.inds = stim_indices

# # Convert saccade times to dataset indices
# sacc_onset_inds = dataset.get_inds_from_times(saccade_onsets)
# sacc_offset_inds = dataset.get_inds_from_times(saccade_offsets)




# # sacc_onset_bool = (dataset.inds[:, None, :] == sacc_onset_inds).all(-1).any(1)
# # sacc_offset_bool = (dataset.inds[:, None, :] == sacc_offset_inds).all(-1).any(1)

#%%
dset_idx = stim_indices[:,0].unique().item()
trials = dataset.dsets[dset_idx].covariates['trial_inds'].unique().numpy()
num_trials = len(trials)
T = len(dataset.dsets[dset_idx]['trial_inds'])
sac_onset_bins = np.unique(np.digitize(saccade_onsets, dataset.dsets[dset_idx]['t_bins']))[1:-1]
sac_onset_bool = np.zeros(T, dtype=bool)
sac_onset_bool[sac_onset_bins.flatten()] = True
sac_offset_bins = np.unique(np.digitize(saccade_offsets, dataset.dsets[dset_idx]['t_bins']))[1:-1]
sac_offset_bool = np.zeros(T, dtype=bool)
sac_offset_bool[sac_offset_bins.flatten()] = True

itrial = 0
#%%
itrial += 1

this_trial = trials[itrial]

ImTrial = BackImageTrial(sess.exp['D'][this_trial], sess.exp['S'])
Im = ImTrial.get_image()**.8

ix = dataset.dsets[dset_idx]['trial_inds']==this_trial

print(f"Trial {itrial} has {len(dataset.dsets[dset_idx]['stim'][dataset.dsets[dset_idx]['trial_inds']==this_trial])} frames")

stim = dataset.dsets[dset_idx]['stim'][ix]
eyepos_rhat = dataset.dsets[dset_idx]['dpi_pix'][ix]
dpi_valid = dataset.dsets[dset_idx]['dpi_valid'][ix]
dpi_valid = dpi_valid.numpy() > 0
masked_eyepos = np.ma.array(eyepos_rhat.numpy(), mask=~np.tile(dpi_valid[:, None], (1, 2)))
dfs = dataset.dsets[dset_idx]['dfs'][ix]
sac_on = np.where(sac_onset_bool[ix]*dpi_valid)[0]
sac_off = np.where(sac_offset_bool[ix]*dpi_valid)[0]

if len(sac_on) < len(sac_off):
    if sac_off[0] < sac_on[0]:
        sac_off = sac_off[1:]
    else:
        sac_on = sac_on[:-1]

if len(sac_on) > len(sac_off):
    if sac_off[0] < sac_on[0]:
        sac_on = sac_on[1:]
    else:
        sac_off = sac_off[:-1]

sac_start_stop = np.concatenate([sac_on[:, None], sac_off[:, None]], axis=1)
ds = sac_start_stop[:,1] - sac_start_stop[:,0]
sac_start_stop = sac_start_stop[ds>0]

Ttrial = eyepos_rhat.shape[0]
fix_starts = np.concatenate([[0], sac_start_stop[:,1]+1])
fix_ends = np.concatenate([sac_start_stop[:,0], [Ttrial-1]])
fix_start_stop = np.concatenate([fix_starts[:, None], fix_ends[:, None]], axis=1)

mask_fix = np.zeros(Ttrial, dtype=bool)
for i in range(fix_start_stop.shape[0]):
    mask_fix[fix_start_stop[i,0]:fix_start_stop[i,1]] = True
mask_fix[~dpi_valid] = False
masked_fixations = np.ma.array(eyepos_rhat.numpy(), mask=~np.tile(mask_fix[:, None], (1, 2)))

mask_sac = np.zeros(Ttrial, dtype=bool)
for i in range(sac_start_stop.shape[0]):
    mask_sac[sac_start_stop[i,0]:sac_start_stop[i,1]] = True
mask_sac[~dpi_valid] = False
masked_saccades = np.ma.array(eyepos_rhat.numpy(), mask=~np.tile(mask_sac[:, None], (1, 2)))


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(Im, cmap='gray', origin='upper')
# ax.plot(masked_eyepos[:,1], masked_eyepos[:,0], 'r')
ax.plot(masked_saccades[:,1], masked_saccades[:,0], 'b')
ax.plot(masked_fixations[:,1], masked_fixations[:,0], 'r')


#%%

plt.plot(masked_saccades[:,1], 'b')
plt.plot(masked_fixations[:,1],'r')

# plt.plot(stim[:,0,25,25].numpy())
# _ = [plt.axvline(i, color='r') for i in sac_on]

#%%
exp = sess.exp
dpi = sess.dpi
from DataYatesV1.exp.general import get_trial_protocols
protocols = get_trial_protocols(exp)

protocol_type = 'BackImage'
trial_inds = np.where(np.array(protocols) == protocol_type)[0]

from DataYatesV1.exp.backimage import BackImageTrial
trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in trial_inds]

plt.figure()
plt.scatter(np.arange(len(trial_inds)), trial_inds)
plt.xlabel('Trial index')
plt.ylabel('Trial number')
plt.title(f'Found {len(trial_inds)} trials of type {protocol_type}')
plt.show()

#%%
from DataYatesV1.utils.general import get_clock_functions
ptb2ephys, vpx2ephys = get_clock_functions(exp)

t_bins_rhat = train_data.dsets[stim_inds[0,0]]['t_bins'][stim_inds[:,1]]
trial_inds_rhat = train_data.dsets[stim_inds[0,0]]['trial_inds'][stim_inds[:,1]]

plt.plot(t_bins_rhat, '.')

time_overlap = np.zeros(len(trials))
for i in range(len(trials)):

    # trial start and stops
    t0 = exp['D'][trial_inds[i]]['START_EPHYS']
    t1 = exp['D'][trial_inds[i]]['END_EPHYS']
    plt.plot([i, i],[t0, t1], 'r')
    iix = (t_bins_rhat.numpy() > t0) & (t_bins_rhat.numpy() < t1)
    time_overlap[i] = np.sum(iix)
    print(f"trial {i} has {np.sum(iix)} overlapping time bins")

good_trials = np.where(time_overlap>0)[0]

#%% extract relevant covariates from the model datasets
stim = train_data.dsets[stim_inds[0,0]]['stim'][stim_inds[:,1]]
eyepos_rhat = train_data.dsets[stim_inds[0,0]]['dpi_pix'][stim_inds[:,1]]
dpi_valid = train_data.dsets[stim_inds[0,0]]['dpi_valid'][stim_inds[:,1]]
dpi_valid = dpi_valid.numpy() > 0
dfs = train_data.dsets[stim_inds[0,0]]['dfs'][stim_inds[:,1]]
robs = bps_results['backimage']['robs']
rhat = bps_results['backimage']['rhat']
bps = bps_results['backimage']['bps']

#%%
plt.plot(stim[:100,0,25,25])
plt.gca().twinx()
plt.plot(eyepos_rhat[:100])

#%%


#%% confirm matchup
i += 1
if i >= len(good_trials):
    i = 0

this_trial = good_trials[i]

t0 = exp['D'][trial_inds[this_trial]]['START_EPHYS']
t1 = exp['D'][trial_inds[this_trial]]['END_EPHYS']

iix = (t_bins_rhat.numpy() > t0) & (t_bins_rhat.numpy() < t1)
iid = iix & dpi_valid

# Get DPI data in the time window
t0_dpi = np.searchsorted(dpi['t_ephys'], t0)
t1_dpi = np.searchsorted(dpi['t_ephys'], t1)
trial_dpi = dpi.iloc[t0_dpi:t1_dpi]

plt.figure()
I = trials[this_trial].get_image()**.8
plt.imshow(I, cmap='gray')
iivalid = trial_dpi['valid'].values
plt.plot(trial_dpi['dpi_j'][iivalid], trial_dpi['dpi_i'][iivalid], 'r')
plt.plot(eyepos_rhat[iix,1], eyepos_rhat[iix,0], 'b')

plt.title(i)
# plt.xlim(0, I.shape[1])
# plt.ylim(0, I.shape[0])

plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

axs[0].imshow(robs[iix].T, aspect='auto', interpolation='none', cmap='gray_r', extent=[t_bins_rhat.numpy()[iix][0], t_bins_rhat.numpy()[iix][-1], 0, robs.shape[1]])
axs[1].imshow(rhat[iix].T, aspect='auto', interpolation='none', cmap='gray_r', extent=[t_bins_rhat.numpy()[iix][0], t_bins_rhat.numpy()[iix][-1], 0, rhat.shape[1]])

axs[2].plot(trial_dpi['t_ephys'], trial_dpi['dpi_j'], 'k')
axs[2].plot(trial_dpi['t_ephys'], trial_dpi['dpi_i'], 'gray')
axs[2].plot(t_bins_rhat.numpy()[iix], eyepos_rhat[iix,1], 'b')
axs[2].plot(t_bins_rhat.numpy()[iix], eyepos_rhat[iix,0], 'r')
axs[2].set_ylim(0, I.shape[1])
plt.show()


#%%
from skimage.feature import match_template
from matplotlib.patches import Rectangle

i =10
if i >= len(good_trials):
    i = 0


big_roi = train_data.dsets[0].metadata['roi_src']
big_roi_w = big_roi[1,1] - big_roi[1,0]
big_roi_h = big_roi[0,1] - big_roi[0,0]

this_trial = good_trials[i]

t0 = exp['D'][trial_inds[this_trial]]['START_EPHYS']
t1 = exp['D'][trial_inds[this_trial]]['END_EPHYS']

iix = (t_bins_rhat.numpy() > t0) & (t_bins_rhat.numpy() < t1)
iid = iix & dpi_valid

# Get DPI data in the time window
t0_dpi = np.searchsorted(dpi['t_ephys'], t0)
t1_dpi = np.searchsorted(dpi['t_ephys'], t1)
trial_dpi = dpi.iloc[t0_dpi:t1_dpi]

iframe = 910
history = 100
plt.figure()
I = trials[this_trial].get_image()**.8
P = stim[iix][iframe].squeeze().detach().cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(I, cmap='gray', origin='upper')
iivalid = trial_dpi['valid'].values
x = trial_dpi['dpi_j'].values[iivalid]
y = trial_dpi['dpi_i'].values[iivalid]
x = I.shape[1] - x
y = I.shape[0] - y

ncc = match_template(I, P)               # zero-mean, unit-variance NCC via FFT
yroi, xroi = np.unravel_index(np.argmax(ncc), ncc.shape)   # top-left corner of best match
h, w = P.shape          # patch size

ax.plot(x[iframe-history:iframe], y[iframe-history:iframe], 'r')
ax.add_patch(Rectangle((xroi, yroi), w, h, edgecolor='r', facecolor='none', linewidth=2))
ax.add_patch(Rectangle((big_roi[1,0]+x[iframe], big_roi[0,0]+y[iframe]), big_roi_w, big_roi_h, edgecolor='b', facecolor='none', linewidth=2))
# plt.plot(eyepos_rhat[iix,1], eyepos_rhat[iix,0], 'b')
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(stim[iix][iframe].squeeze().detach().cpu().numpy(), cmap='gray')
plt.axis("off")
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

axs[0].imshow(robs[iix].T, aspect='auto', interpolation='none', cmap='gray_r', extent=[t_bins_rhat.numpy()[iix][0], t_bins_rhat.numpy()[iix][-1], 0, robs.shape[1]])
axs[0].axvline(t_bins_rhat.numpy()[iix][iframe], color='k', linestyle='--')
axs[1].imshow(rhat[iix].T, aspect='auto', interpolation='none', cmap='gray_r', extent=[t_bins_rhat.numpy()[iix][0], t_bins_rhat.numpy()[iix][-1], 0, rhat.shape[1]])
axs[1].axvline(t_bins_rhat.numpy()[iix][iframe], color='k', linestyle='--')
axs[2].plot(trial_dpi['t_ephys'], trial_dpi['dpi_j'], 'k')
axs[2].plot(trial_dpi['t_ephys'], trial_dpi['dpi_i'], 'gray')
axs[2].plot(t_bins_rhat.numpy()[iix], eyepos_rhat[iix,1], 'b')
axs[2].plot(t_bins_rhat.numpy()[iix], eyepos_rhat[iix,0], 'r')
axs[2].axvline(t_bins_rhat.numpy()[iix][iframe], color='k', linestyle='--')
axs[2].set_ylim(0, I.shape[1])
plt.show()

#%% ANIMATION
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from skimage.feature import match_template
import numpy as np

# ------------------------------------------------------------
# 1.  Convenience: gather everything that varies with frame
# ------------------------------------------------------------
big_roi = train_data.dsets[0].metadata['roi_src']
big_roi_w = big_roi[1,1] - big_roi[1,0]
big_roi_h = big_roi[0,1] - big_roi[0,0]

frames      = np.arange(len(iix))          # indices into the valid-time window
I_full      = trials[this_trial].get_image() ** .8            # static cage photo
stim_clip   = stim[iix].squeeze().cpu().numpy()              # [T, h, w] movie
t_clip      = t_bins_rhat.numpy()[iix]                       # [T] time stamps
eye_x, eye_y = eyepos_rhat[iix,1], eyepos_rhat[iix,0]        # model eye
dpi_x, dpi_y = trial_dpi['dpi_j'].values, trial_dpi['dpi_i'].values
dpi_x = I_full.shape[1] - dpi_x
dpi_y = I_full.shape[0] - dpi_y

dpi_t = trial_dpi['t_ephys'].values
tail_len = 10                     # how many past samples to keep visible

# (optional) pre-compute template matches for speed
xy_roi = []
for k, P in enumerate(stim_clip):
    ncc  = match_template(I_full, P)
    y0,x0 = np.unravel_index(np.argmax(ncc), ncc.shape)
    xy_roi.append((x0,y0))
xy_roi = np.asarray(xy_roi)        # shape [T, 2]

# ------------------------------------------------------------
# 2.  Static layout builder
# ------------------------------------------------------------
def init_layout():
    fig = plt.figure(figsize=(15, 6), dpi=200, layout="constrained")
    gs  = GridSpec(1, 2, width_ratios=[2.8, 2.8], figure=fig)

    # LEFT: full image + rectangle + inset
    ax_img   = fig.add_subplot(gs[0])
    ax_img.imshow(I_full, cmap='gray', origin='upper')
    ax_img.axis("off")
    rect     = Rectangle((0,0), 1,1, ec='red', fc='none', lw=2)
    rect_big = Rectangle((0,0), 1,1, ec='cyan', fc='none', lw=2)
    ax_img.add_patch(rect)
    ax_img.add_patch(rect_big)


    # ------------------------------------------------
    # build inset for model input
    ax_inset = inset_axes(ax_img, width="35%", height="35%", loc="lower left",
                          bbox_to_anchor=(0.01,0.01,1,1),
                          bbox_transform=ax_img.transAxes)
    im_inset = ax_inset.imshow(stim_clip[0], cmap='gray', origin='upper')

    ax_inset.set_xticks([])              # no tick marks
    ax_inset.set_yticks([])
    ax_inset.set_xticklabels([])         # no numbers
    ax_inset.set_yticklabels([])

    ax_inset.set_title("Input to Model", fontsize=10, fontweight='bold', pad=3, color='red')
    for s in ax_inset.spines.values():
        s.set_edgecolor('red'); s.set_linewidth(2)
    
    
    ln_path, = ax_img.plot([], [], color='cyan',
                       lw=1.2, alpha=.8, solid_capstyle='round')
    ln_dot,  = ax_img.plot([], [], 'o', color='cyan', ms=4)

    # RIGHT: three stacked panels
    gs_r     = gs[1].subgridspec(3,1, height_ratios=[1,1,.5], hspace=0.05)
    ax_r     = fig.add_subplot(gs_r[0]); ax_p = fig.add_subplot(gs_r[1], sharex=ax_r)
    ax_e     = fig.add_subplot(gs_r[2], sharex=ax_r)

    ax_r.set_title("Observed Spikes",  fontsize=14, fontweight='bold', pad=10)
    ax_p.set_title("Model Predictions", fontsize=14, fontweight='bold', pad=10)

    im_robs  = ax_r.imshow((dfs[iix]*robs[iix]).T, aspect='auto', cmap='gray_r',
                           extent=[t_clip[0], t_clip[-1], 0, robs.shape[1]])
    im_rhat  = ax_p.imshow((dfs[iix]*rhat[iix]).T, aspect='auto', cmap='gray_r',
                           extent=im_robs.get_extent())

    vline_r  = ax_r.axvline(t_clip[0], color='r', ls='--', lw=2)
    vline_p  = ax_p.axvline(t_clip[0], color='r', ls='--', lw=2)
    vline_e  = ax_e.axvline(t_clip[0], color='r', ls='--', lw=2)

    ax_r.set_ylabel('Neuron #'); ax_p.set_ylabel('Neuron #')
    # ax_r.set_xticks([]); ax_p.set_xticks([])
    for a in (ax_r,ax_p,ax_e): a.spines[['top','right']].set_visible(False)

    # eye / dpi traces (static, only grow with frame index)
    ln_dpix, = ax_e.plot(dpi_t, dpi_x,  color='k',   lw=.7, label='DPI x')
    ln_dpiy, = ax_e.plot(dpi_t, dpi_y,  color='0.4', lw=.7, label='DPI y')
    # ln_ex,   = ax_e.plot([], [], color='royalblue',  lw=.9, label='model x')
    # ln_ey,   = ax_e.plot([], [], color='firebrick',  lw=.9, label='model y')
    ax_e.set_ylim(0, I_full.shape[1])
    ax_e.set_xlim(t_clip[0], t_clip[-1])
    ax_e.set_xlabel("Time (s)"); ax_e.set_ylabel("Pixels")
    ax_e.legend(frameon=False, fontsize=7, ncol=2)
    
    
    vlines = [vline_r, vline_p, vline_e]

    # store everything you need later:
    artists = dict(rect=rect,
                   rect_big=rect_big,
               im_inset=im_inset,
               vlines=vlines,
               ln_path=ln_path, ln_dot=ln_dot,
               ln_ex=ln_dpix, ln_ey=ln_dpiy)
    
    return fig, artists

fig, art = init_layout()

# ------------------------------------------------------------
# 3.  Animation callbacks
# ------------------------------------------------------------
def init():
    art['ln_ex'].set_data([], [])
    art['ln_ey'].set_data([], [])

    art['ln_ex'].set_data(t_clip, eye_x)
    art['ln_ey'].set_data(t_clip, eye_y)
    # flatten vlines so we don't hand FuncAnimation a tuple-of-tuples
    return (art['rect'], art['rect_big'], art['im_inset'],
            *art['vlines'],
            art['ln_ex'], art['ln_ey'])


def update(k):
    if k >= len(t_clip):
        return []

    # ----------------------------------------------------------
    # scanpath
    start = max(0, k-tail_len)
    xs = eye_x[start:k+1]
    ys = eye_y[start:k+1]
    art['ln_path'].set_data(xs, ys)
    art['ln_dot'].set_data(xs[-1], ys[-1])   # current gaze dot

    # ----------------------------------------------------------
    # ROI & inset
    x0, y0 = xy_roi[k]
    art['rect'].set_xy((x0, y0))
    art['rect'].set_width(stim_clip[k].shape[1])
    art['rect'].set_height(stim_clip[k].shape[0])
    
    art['rect_big'].set_xy((big_roi[1,0]+xs[-1], big_roi[0,0]+ys[-1]))
    art['rect_big'].set_width(big_roi_w)
    art['rect_big'].set_height(big_roi_h)
    art['im_inset'].set_array(stim_clip[k])

    # ----------------------------------------------------------
    # dashed time-bar
    t_now = t_clip[k]
    for vl in art['vlines']:
        vl.set_xdata([t_now, t_now])

    # ----------------------------------------------------------
    # eye traces in bottom panel
    # art['ln_ex'].set_data(t_clip[:k+1], eye_x[:k+1])
    # art['ln_ey'].set_data(t_clip[:k+1], eye_y[:k+1])

    return (art['rect'], art['rect_big'], art['im_inset'],
            *art['vlines'],
            art['ln_path'], art['ln_dot'],   # <–
            art['ln_ex'],  art['ln_ey'])

# ------------------------------------------------------------
# 4.  Render & save
# ------------------------------------------------------------
fps       = 30
writer    = FFMpegWriter(fps=fps, codec='libx264',
                         extra_args=['-pix_fmt', 'yuv420p'],
                         bitrate=16000)   # increase for lossless-ish

n_frames  = len(t_clip)                 # or len(stim_clip), they are equal
frame_ids = range(n_frames)             # 0 … n_frames-1  (no +1 surprise)


ani = FuncAnimation(fig, update, frames=frame_ids,
                    init_func=init, blit=True)

# ani = FuncAnimation(fig, update, frames=len(frames),
#                     init_func=init, blit=True)

ani.save(f"trial_demo_{model_name}.mp4", writer=writer)


# t_trial = np.arange(start, end, 5)
# t_trial -= t_trial[0]
# t_trial /= 1000 # convert to seconds
#%%


#%%


print(len(t_bins_rhat), bps_results['backimage']['robs'].shape[0])



#%% plot a snippet of robs and rhat

start = np.random.randint(0, robs.shape[0] - 1000)
inds = np.arange(start, start+1000)
plt.figure(figsize=(10, 5))
plt.subplot(2,1,1)
plt.imshow(robs[inds].T, aspect='auto', interpolation='none', cmap='gray_r')
plt.xlabel('Time (5ms bins)')
plt.ylabel('Cell')
plt.title('Observed Spikes')

plt.subplot(2,1,2)
plt.imshow(rhat[inds].T, aspect='auto', interpolation='none', cmap='gray_r')
plt.xlabel('Time (5ms bins)')
plt.ylabel('Cell')
plt.title('Predicted Spikes')

plt.tight_layout()
plt.show()

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
