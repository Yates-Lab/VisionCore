"""
Compute spatial information from the model using reconstructed stimuli.
Allows counterfactual analysis with real vs fake eye traces.
"""
#%% Imports
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

from DataYatesV1 import enable_autoreload, get_free_device
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye
from spatial_info import make_stimulus_stack, make_counterfactual_stim
from spatial_info import get_spatial_readout
from spatial_info import compute_rate_map, compute_rate_map_batched
from spatial_info import spatial_ssi_population, make_movie

enable_autoreload()
device = get_free_device()

from utils import get_model_and_dataset_configs

#%% Get model and data
model, dataset_configs = get_model_and_dataset_configs()
model = model.to(device)

import dill
with open('/home/jake/repos/VisionCore/scripts/mcfarland_outputs_mono.pkl', 'rb') as f:
    outputs = dill.load(f)

readout = get_spatial_readout(model, outputs).to(device)

sessions = [outputs[i]['sess'] for i in range(len(outputs))]
#%% Helper functions



"""
Plotting code for making a nice figure with the spatial information over time on an image
"""
def plot_spatial_info_figure(full_stack, iframe, f, Pr, eyepos, itrial, I_t_null, I_t,
                             crop=(slice(250, 350), slice(250, 350)),
                             outpath=None, dpi=300):
    # -------------------------
    # Style (publication-ish)
    # -------------------------
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,   # nicer embedded fonts
        "ps.fonttype": 42,
    })

    # consistent colors
    c_fem  = "#1f77b4"  # matplotlib default blue
    c_null = "#ff7f0e"  # matplotlib default orange
    c_diag = "0.15"

    fig = plt.figure(figsize=(11.2, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.05, 1.0, 1.25, 1.25],
        hspace=0.45, wspace=0.38
    )

    # Helpers
    def prettify(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.15, linewidth=0.6)
        ax.set_axisbelow(True)

    def panel_label(ax, s):
        ax.text(0.0, 1.08, s, transform=ax.transAxes,
                ha="left", va="bottom", fontweight="bold")

    # -------------------------
    # A) Stimulus
    # -------------------------
    axA = fig.add_subplot(gs[0, 0])
    stim = full_stack[iframe][crop[0], crop[1]]
    axA.imshow(stim, cmap="gray", interpolation="nearest")
    axA.set_title("Stimulus")
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_visible(False)
    panel_label(axA, "A")

    # -------------------------
    # B) Power spectrum
    # -------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(f, Pr[iframe], color=c_fem, lw=1.8)
    axB.set_xlabel("Spatial Frequency (c/deg)")
    axB.set_ylabel("Power")
    axB.set_title("Power Spectrum")
    prettify(axB)
    # optional: focus x-range / log scales if you want
    # axB.set_xscale("log"); axB.set_yscale("log")
    panel_label(axB, "B")

    # -------------------------
    # C) Eye position (wide)
    # -------------------------
    axC = fig.add_subplot(gs[0, 2:4])
    ep = eyepos[itrial]  # expected shape: (T,2) or (T,) ; your screenshot shows 2 traces
    if ep.ndim == 1:
        axC.plot(ep, color=c_fem, lw=1.5)
    else:
        axC.plot(ep[:, 0], color=c_fem,  lw=1.5, label=None)
        axC.plot(ep[:, 1], color=c_null, lw=1.5, label=None)
    axC.set_xlabel("Time (frames)")
    axC.set_ylabel("Eye Position (deg)")
    axC.set_title("Eye Position")
    prettify(axC)
    panel_label(axC, "C")

    # -------------------------
    # D) Spatial info scatter (wide)
    # -------------------------
    axD = fig.add_subplot(gs[1, 0:2])
    x = I_t_null.mean(0)
    y = I_t.mean(0)

    axD.scatter(x, y, s=14, alpha=0.18, color=c_fem, edgecolors="none", rasterized=True)
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    pad = 0.03 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    axD.plot([lo, hi], [lo, hi], color=c_diag, lw=1.6)
    axD.set_xlim(lo, hi)
    axD.set_ylim(lo, hi)

    axD.set_xlabel("Bits (No FEM)")
    axD.set_ylabel("Bits (With FEM)")
    axD.set_title("Spatial Info (Units)")
    prettify(axD)
    panel_label(axD, "D")

    # -------------------------
    # E) Spatial info timecourse (wide)
    # -------------------------
    axE = fig.add_subplot(gs[1, 2:4])
    axE.plot(I_t.mean(1),      color=c_fem,  lw=1.6, label="FEM")
    axE.plot(I_t_null.mean(1), color=c_null, lw=1.6, label="Null")
    axE.set_xlabel("Time (frames)")
    axE.set_ylabel("Spatial Info (bits)")
    axE.legend(frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.85",
               loc="upper left")
    axE.set_title("")  # your original doesn’t title this panel; keep clean
    prettify(axE)
    panel_label(axE, "E")

    # Margins so labels breathe
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.12)

    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    return fig



def radial_power_spectra_np(imgs, ppd, nbins=None, window=True, return_2d=False, eps=0.0):
    """
    imgs: (N,H,W) float/uint, any range
    ppd: pixels per degree (float)
    nbins: number of radial bins (default ~ min(H,W)//2)
    window: apply 2D Hann window (recommended)
    return_2d: also return per-image 2D power spectra (fftshifted)
    Returns:
      f_centers: (B,) cycles/degree
      P_radial:  (N,B) mean power in annuli
      (optional) P2d: (N,H,W) 2D power spectra (fftshifted)
    """
    imgs = np.asarray(imgs, dtype=np.float32)
    N, H, W = imgs.shape
    if nbins is None:
        nbins = min(H, W) // 2

    # --- precompute frequency grid in cycles/degree ---
    fy = np.fft.fftfreq(H, d=1.0) * ppd  # cycles/deg
    fx = np.fft.fftfreq(W, d=1.0) * ppd
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R = np.sqrt(FX**2 + FY**2)
    R = np.fft.fftshift(R)

    # radial bins (0 .. max radius)
    r_max = R.max()
    edges = np.linspace(0.0, r_max + 1e-12, nbins + 1)
    bin_idx = np.digitize(R.ravel(), edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)
    counts = np.bincount(bin_idx, minlength=nbins).astype(np.float32)

    # optional window
    if window:
        wy = np.hanning(H).astype(np.float32)
        wx = np.hanning(W).astype(np.float32)
        win = wy[:, None] * wx[None, :]
    else:
        win = None

    P_radial = np.empty((N, nbins), dtype=np.float32)
    P2d_out = np.empty((N, H, W), dtype=np.float32) if return_2d else None

    for i in range(N):
        x = imgs[i]
        x = x - x.mean()
        if win is not None:
            x = x * win

        F = np.fft.fft2(x, norm="ortho")
        P = (F.real * F.real + F.imag * F.imag)  # |F|^2
        P = np.fft.fftshift(P)

        # radial mean via bincount
        sums = np.bincount(bin_idx, weights=P.ravel(), minlength=nbins).astype(np.float32)
        P_radial[i] = sums / (counts + eps)

        if return_2d:
            P2d_out[i] = P

    f_centers = 0.5 * (edges[:-1] + edges[1:])
    return (f_centers, P_radial, P2d_out) if return_2d else (f_centers, P_radial)

"""
This is the key simulation
Inputs:
    eyepos: (T,2) eye positions in degrees
    full_stack: (N,H,W) stimulus stack (N frames)
    out_size: (H_out, W_out) size of output stimulus
    n_lags: number of time lags to use
    scale: scale factor for stimulus
    plot: whether to plot the eyeposition and stimulus frame
"""
def get_trial_stim_and_rates(eyepos, full_stack,
                             out_size=(151, 151), n_lags=32, scale=1.0, plot=False):

    T = np.where(np.isnan(eyepos[:,0]))[0][0]
    eyepos = eyepos[:T]
    eyepos = torch.from_numpy(eyepos).float()

    null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0)

    eye_stim = make_counterfactual_stim(full_stack, eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale)
    eye_stim_null = make_counterfactual_stim(full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale)
    # print(f"Reconstructed stim shape: {eye_stim.shape}")

    v = out_size[0]/ppd
    if plot:
        plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
        plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
        plt.show()

    # Compute rates on normalized stimulus
    # TODO: This assumes pixelnorm was called (which it almost certainly was, but we should do this better...)
    y = compute_rate_map_batched(model, readout, (eye_stim - 127.0)/255.0)
    y_null = compute_rate_map_batched(model, readout, (eye_stim_null - 127.0)/255.0)

    return y, y_null, eye_stim, eye_stim_null

"""
This is the key simulation now with multiple scales for the eyetrace
Inputs:
    eyepos: (T,2) eye positions in degrees, single trial of fixed max length (540) with NaNs to indicate end
    full_stack: (N,H,W) stimulus stack (N frames)
    out_size: (H_out, W_out) size of output stimulus
    n_lags: number of time lags to use
    stim_scale: scale factor for stimulus
    eye_scale: vecotr of scale factor for eye position (e.g. 1.0 for real, 0.0 for null), default to [1 0]
    plot: whether to plot the eyeposition and stimulus frame
    # now packages the outputs into (T, C, H_out, W_out, n_eyescales) 
    # for the stimulus and rates, where n_eyescales is the number of different eye scales we want to test
    # rather than two separate outputs for real and null, we can just have one output with the different scales
    returns: y: (T, C, H_out, W_out, n_eyescales) rate map for real eye trace
    eye_stim: (T, C, H_out, W_out, n_eyescales) stimulus for real eye trace
"""
def get_trial_stim_and_rates_eyescaled(eyepos, full_stack,
                             out_size=(151, 151), n_lags=32, stim_scale=1.0, eye_scale=[0, 0.5, 1.0, 2.0], plot=False):
    T = np.where(np.isnan(eyepos[:,0]))[0][0]
    #For very fast frame rates, we may have fewer frames our eyetrack supports
    if T > (full_stack.shape[0]-n_lags-1):
        #print(f"Warning: eyepos length {T} greater than full_stack length {full_stack.shape[0]}, truncating eyepos")
        T = full_stack.shape[0]-n_lags-1 # ensure we have enough frames for lags full_stack[:eyepos.shape[0] + n_lags])
    eyepos = eyepos[:T]
    print(f"Using eyepos length {T}")
    eyepos = torch.from_numpy(eyepos).float()

    #scaled_eyepos= torch.zeros_like(eyepos) + eyepos.mean(0) * (1 - eye_scale) + eyepos * eye_scale
    #null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0) * (1 - eye_scale) + eyepos * eye_scale

    for i in range(eye_scale.shape[0]):
        scaled_eyepos = torch.zeros_like(eyepos) + eyepos * eye_scale[i] + (1 - eye_scale[i]) * eyepos.mean(0)
        
        eye_stim_loop = make_counterfactual_stim(full_stack, scaled_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=stim_scale)
        # append to the output tensor
        eye_stim = eye_stim_loop if i == 0 else torch.cat((eye_stim, eye_stim_loop), dim=4)
   
        #eye_stim_null = make_counterfactual_stim(full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=stim_scale)
        # print(f"Reconstructed stim shape: {eye_stim.shape}")

        v = out_size[0]/ppd
        if plot:
            plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
            plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
            plt.show()

        # Compute rates on normalized stimulus
        # TODO: This assumes pixelnorm was called (which it almost certainly was, but we should do this better...)
        #for i in range(eye_scale.shape[0]):
        y_loop = compute_rate_map_batched(model, readout, (eye_stim_loop - 127.0)/255.0)
        # y_loop should be (T, C, H_out, W_out), append to output tensor on new axis
        # y = y_loop if i == 0 else torch.cat((y, y_loop), dim=4)
        # but this errors "IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)"
        # so we need to unsqueeze y_loop first
        y_loop = y_loop.unsqueeze(4)
        y = y_loop if i == 0 else torch.cat((y, y_loop), dim=4)
        
    #y_null = compute_rate_map_batched(model, readout, (eye_stim_null - 127.0)/255.0)

    return y, eye_stim

#%% This cell just loops over datasets and extracts all the fixation eye traces
eyetraces = []
max_T = 540

for name in sessions:
    dataset_idx = model.names.index(name)
    
    try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

            # Get fixrsvp trial indices
            inds = torch.concatenate([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp')
            ], dim=0)

            dataset = train_data.shallow_copy()
            dataset.inds = inds

            dset_idx = inds[:,0].unique().item()
            trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
            trials = np.unique(trial_inds)
            NT = len(trials)

            fixation = np.hypot(
                dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), 
                dataset.dsets[dset_idx]['eyepos'][:,1].numpy()
            ) < 1

            for itrial in range(NT):
                ix = (trials[itrial] == trial_inds) & fixation
                ix = (trials[itrial] == trial_inds) & fixation
                eyepos = dataset.dsets[dset_idx]['eyepos'][ix]
                eyetrace = np.zeros((max_T, 2))*np.nan
                eyetrace[:len(eyepos)] = eyepos.numpy()
                eyetraces.append(eyetrace)

    except Exception as e:
        print(f"Failed to load dataset {name}: {e}")

#%% Organize the eye traces for later use
eyepos = np.stack(eyetraces)
fix_dur = [np.where(np.isnan(e).any(axis=1))[0][0] for e in eyepos]

_ = plt.hist(fix_dur, bins=np.arange(0, 540, 10))

fix_dur = np.array(fix_dur)
good_trials = fix_dur > 60
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]

plt.figure()
_ = plt.plot(eyepos[:,:,0].T, alpha=0.1)

#%% Generate stimulus stack
ppd = 37.50476617
frames_per_im = 6
full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im)
print(f"Full stimulus stack shape: {full_stack.shape}")

#%% Counterfactual eye trace
n_lags = 32
out_size = (151, 151)
dt = 1/120
scale = 1.0

#%% First batch (real stim with static frames)
frame = None
type = 'fixrsvp'
frames_per_im = 1

full_stack = make_stimulus_stack(type=type,
        frame=frame, frames_per_im=frames_per_im)


#%% Plot all images
N = full_stack.shape[0]
sx = int(np.sqrt(N))
sy = int(np.ceil(N / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= N:
        axs.flatten()[i].axis('off')
        continue
    im = full_stack[i][250:350][:, 250:350]
    axs.flatten()[i].imshow(im, cmap='gray')
    axs.flatten()[i].axis('off')
    axs.flatten()[i].set_title(f'{i}')
plt.show()

# calcualte the power spectrum for each image
f, Pr = radial_power_spectra_np(full_stack, ppd=ppd, window=True)       # (B,), (N,B)
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= N:
        axs.flatten()[i].axis('off')
        continue
    axs.flatten()[i].plot(f, Pr[i])
    axs.flatten()[i].set_title(f'{i}')
    axs.flatten()[i].set_xscale('log')
    axs.flatten()[i].set_yscale('log')
    axs.flatten()[i].set_xlabel('Spatial Frequency (c/deg)')
    axs.flatten()[i].set_ylabel('Power')

plt.show()


#%% Find a long fixation to use
trial_list = np.argsort(fix_dur)[::-1]
itrial = trial_list[1]

plt.figure()
plt.plot(eyepos[itrial])
plt.show()

#%% run one image to get a sense
iframe = 43
y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack[[iframe]].repeat(fix_dur[itrial]+n_lags+1, axis=0), out_size=out_size, n_lags=n_lags, scale=scale)

#%% Compute information from rate maps
ispike, irate, I_t = spatial_ssi_population(y)
ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

#%% make a movie of this trial
n_units = 25 # number of units to show
units_to_show = np.argsort(I_t.mean(0)-I_t_null.mean(0)).numpy()[::-1][:n_units] # the ones with the most gain in spatial info
make_movie(y, save_path=f'spatial_info_fixrsvpstatic_{iframe}_{itrial}_activations', n_units_to_show=units_to_show)


#%% make a movie of the stimulus itself
# import imageio.v2 as imageio

frames = eye_stim[:,0,-1,:,:]
# normalize from 0 to 1
frames = (255*(frames - frames.min()) / (frames.max() - frames.min())).numpy().astype(np.uint8)
# imageio.mimsave(f'../figures/spatial_info_fixrsvpstatic_{iframe}_{itrial}_stimulus.mp4', frames, fps=30, format="FFMPEG")
# # make_movie(eye_stim, save_path=, n_units_to_show=units_to_show)

#%% plot rates for some of the units that were shown in the movie
for cc in units_to_show:
    plt.plot(y[:,cc,25,[15, 25, 35]], 'b')
    plt.plot(y_null[:,cc,25,[15, 25, 35]], 'r')
    plt.title(f'Unit {cc}')
    plt.show()

# y2 = compute_rate_map_batched(model, readout, )

# Rather than rates, we can also compute the spatial information directly from the rate maps, which is what we want to do for the main analysis. This will be a bit slower, but it will give us the spatial information for each unit and each time point, 
# and plot as a function of time. Cumulative spatial information should grow over time, and we can compare the real vs null conditions to see how much the eye movements are contributing to the spatial information.


#%% Loop over all frames and run the analysis for a single eye trace
rerun=False # this is slow
if rerun:
    for iframe in range(full_stack.shape[0]):
        print(f"Frame {iframe}")
        y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack[[iframe]].repeat(fix_dur[itrial]+n_lags+1, axis=0), out_size=out_size, n_lags=n_lags, scale=scale)
        ispike, irate, I_t = spatial_ssi_population(y)
        ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

        fig = plot_spatial_info_figure(
            full_stack, iframe, f, Pr,
            eyepos, itrial,
            I_t_null, I_t,
            outpath=f"../figures/spatial_info/spatial_info_{iframe}_{itrial}.png"
        )
        fig.show()


#%% Try again on natural images
frames_per_im = 1
full_stack = make_stimulus_stack(type='nat',
        frame=None, frames_per_im=frames_per_im)


#%% Plot all images and power spectra
N = full_stack.shape[0]
sx = int(np.sqrt(N))
sy = int(np.ceil(N / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= N:
        axs.flatten()[i].axis('off')
        continue
    im = full_stack[i]
    axs.flatten()[i].imshow(im, cmap='gray')
    axs.flatten()[i].axis('off')
    axs.flatten()[i].set_title(f'{i}')
plt.show()

# calcualte the power spectrum for each image
f, Pr = radial_power_spectra_np(full_stack, ppd=ppd, window=True)       # (B,), (N,B)
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= N:
        axs.flatten()[i].axis('off')
        continue
    axs.flatten()[i].plot(f, Pr[i])
    axs.flatten()[i].set_title(f'{i}')
    axs.flatten()[i].set_xscale('log')
    axs.flatten()[i].set_yscale('log')
    axs.flatten()[i].set_xlabel('Spatial Frequency (c/deg)')
    axs.flatten()[i].set_ylabel('Power')

plt.show()



#%% run one image to get a sense
iframe = 24
y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack[[iframe]].repeat(fix_dur[itrial]+n_lags+1, axis=0), out_size=out_size, n_lags=n_lags, scale=scale)

#%% Compute information from rate maps
ispike, irate, I_t = spatial_ssi_population(y)
ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

#%% make a movie of this trial
n_units = 25 # number of units to show
units_to_show = np.argsort(I_t.mean(0)-I_t_null.mean(0)).numpy()[::-1][:n_units] # the ones with the most gain in spatial info
make_movie(y, save_path=f'spatial_info_natstatic_{iframe}_{itrial}_activations', n_units_to_show=units_to_show)


#%% make a movie of the stimulus itself
# import imageio.v2 as imageio

frames = eye_stim[:,0,-1,:,:]
# normalize from 0 to 1
frames = (255*(frames - frames.min()) / (frames.max() - frames.min())).numpy().astype(np.uint8)
# imageio.mimsave(f'../figures/spatial_info_natstatic_{iframe}_{itrial}_stimulus.mp4', frames, fps=30, format="FFMPEG")
# # make_movie(eye_stim, save_path=, n_units_to_show=units_to_show)

#%% plot rates for some of the units that were shown in the movie
for cc in units_to_show:
    plt.plot(y[:,cc,25,[15, 25, 35]], 'b')
    plt.plot(y_null[:,cc,25,[15, 25, 35]], 'r')
    plt.title(f'Unit {cc}')
    plt.show()



#%% Compute power spectrum of all stimuli
f, Pr, P2d = radial_power_spectra_np(full_stack, ppd, return_2d=True)   # plus (N,H,W)


#%% Second batch (real stim with different framerates)

frame = None
type = 'fixrsvp'
frames_per_im = 60

full_stack = make_stimulus_stack(type=type,
        frame=frame, frames_per_im=frames_per_im)


#%% sample with eye positions
from tqdm import tqdm
'''
This analysis should loop over the frames per frame,
because that's effectively how long the images are stable,
and it should calculate the information gain.

But it's probably a good idea to store the I_tand the I_t_null
instead of just the summary rate, because then we can ask how
this accumulates over time. So basically, the longer the trial,
this grows as a function of how many samples you get. 

So the key thing that we expect to find here is that when the
stimulus is flashed, at four hertz, it starts to switch. 
Two hertz, it is totally switched. One hertz, real eye movements are dominant,
and zero hertz, basically, the eye movements add everything.
So it's all consistent with the story that fixational eye movements are really
part about reformatting spatial information into temporal modulations,
and those temporal modulations increase spatial information.

That's what it's saying, and if you don't have flashing stimuli,
you need eye movements to do that. 

'''
i_spikes = []
i_rates = []
i_spikes_null = []
i_rates_null = []

# this is currently looping over all frames, we also need to loop over 
# 'frames per image' to get the different conditions, but this is a start
for itrial in tqdm(range(eyepos.shape[0])):
    y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack, out_size=out_size, n_lags=n_lags, scale=scale)
    ispike, irate, I_t = spatial_ssi_population(y)
    ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)
    i_spikes.append(ispike)
    i_rates.append(irate)
    i_spikes_null.append(ispike_null)
    i_rates_null.append(irate_null)


#%%

itrial +=1
if itrial > len(i_spikes):
    itrial = 0
plt.figure()
plt.subplot(1,2,1)
plt.plot(eyepos[itrial])
plt.subplot(1,2,2)
plt.plot(i_spikes[itrial])
plt.plot(i_spikes_null[itrial])

#%% This doesn't work because of different sizes
# i_spikes = np.stack(i_spikes)
# i_rates = np.stack(i_rates)
# i_spikes_null = np.stack(i_spikes_null)
# i_rates_null = np.stack(i_rates_null)

#%% concatenate the outputs
i_spikes_full = np.empty((0,))
i_rates_full = np.empty((0,))
i_spikes_null_full = np.empty((0,))
i_rates_null_full = np.empty((0,))
for i in range(len(i_spikes)):
    i_spikes_full = i_spikes_full if i == 0 else np.concatenate((i_spikes_full, i_spikes[i]), axis=0)
    i_rates_full = i_rates_full if i == 0 else np.concatenate((i_rates_full, i_rates[i]), axis=0)
    i_spikes_null_full = i_spikes_null_full if i == 0 else np.concatenate((i_spikes_null_full, i_spikes_null[i]), axis=0)
    i_rates_null_full = i_rates_null_full if i == 0 else np.concatenate((i_rates_null_full, i_rates_null[i]), axis=0)


    
#%% need to loop through as there are different sizes of the outputs
plt.figure()
plt.subplot(1,2,1)
plt.plot(i_spikes_null_full, i_spikes_full, '.', alpha=0.1)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Bits/Spike (Null stim)')
plt.ylabel('Bits/Spike (Real stim)')
plt.title('Spatial Info (Spikes)')

plt.subplot(1,2,2)
plt.plot(i_rates_null_full, i_rates_full, '.', alpha=0.1)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Bits/Sec (Null stim)')
plt.ylabel('Bits/Sec (Real stim)')
plt.title('Spatial Info (Time)')


#%%

itrial = 1
plt.plot(eyepos[itrial])

frames_per_im = 1
full_stack = make_stimulus_stack(type='nat',
        frame=None, frames_per_im=frames_per_im)


#%%
y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack, out_size, n_lags, scale)

# compute spatial info
ispike, irate, I_t = spatial_ssi_population(y)
ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

plt.plot(eyepos.numpy())
plt.show()

inds = np.argsort(I_t.mean(0).numpy()-I_t_null.mean(0).numpy())[::-1]

for cc in inds[:10]:
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(y[:,cc,0,::5]/dt, 'b')
    plt.plot(y_null[:,cc,0,::5]/dt, 'r')
    plt.title(f'Unit {cc}')
    plt.ylabel('Rate (spikes/bin)')
    plt.subplot(3,1,2)
    plt.plot(I_t[:,cc])
    plt.plot(I_t_null[:,cc])
    plt.xlabel('Frame')
    plt.ylabel('Spatial Info (bits)')
    plt.subplot(3,1,3) # plot variance across space
    plt.plot(y[:,cc].var((1,2)), 'b--')
    plt.plot(y_null[:,cc].var((1,2)), 'r--')
    plt.plot(y[:,cc].mean((1,2)), 'b')
    plt.plot(y_null[:,cc].mean((1,2)), 'r')
    plt.xlabel('Frame')
    plt.ylabel('Variance across space')
    plt.show()


#%%
'''
This method now loops over different frames per image and different eye scales,
to get a full matrix of spatial information as a function of these two variables.
Null condition is eye scale = 0.0, but note this is now handled differently.

We want to also loop over trials and get mean, but for now just do one trial (itrial).
'''
itrial = 340
i_spikes_fpi = []
i_rates_fpi  = []
I_t_fpi      = []

frames_per_im_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
#frames_per_im_list = [1.88, 3.75, 7.5, 15, 30, 60]
# eye_scale_list = [0.0,0.25, 0.5, 1.0, 2.0, 4.0]
eye_scale_list = [0.0] + list(np.exp(np.linspace(-2.75, np.log(2), 11))) # odd to include 1.0 in the middle
for frames_per_im in frames_per_im_list:
    print(f"Frames per image: {frames_per_im}")
    full_stack = make_stimulus_stack(type='fixrsvp',
            frame=None, frames_per_im=frames_per_im, num_frames=540)
    print(f"full_stack shape: {full_stack.shape}")
    y_scales, eye_stim_scales = get_trial_stim_and_rates_eyescaled(eyepos[itrial], 
            full_stack, out_size=out_size, n_lags=n_lags, stim_scale=scale, eye_scale=np.array(eye_scale_list), plot=False)

    print(f"y_scales shape: {y_scales.shape}")

    ispike_list = []
    irate_list = []
    I_t_list = []

    for i in range(y_scales.shape[4]):
        y = y_scales[...,i] # (T, C, H_out, W_out)
        
        # this returns ispike(T), irate(T), I_t(T,C)
        ispike_loop, irate_loop, I_t_loop = spatial_ssi_population(y) 
        
        ispike_list.append(ispike_loop)   # (T,)
        irate_list.append(irate_loop)     # (T,)
        I_t_list.append(I_t_loop)          # (T, C)

    i_spikes = torch.stack(ispike_list, dim=1)   # (T, n_eyescales)
    i_rates  = torch.stack(irate_list, dim=1)    # (T, n_eyescales)
    I_t      = torch.stack(I_t_list, dim=2)      # (T, C, n_eyescales)
        
    # collect per frames_per_im
    i_spikes_fpi.append(i_spikes)
    i_rates_fpi.append(i_rates)
    I_t_fpi.append(I_t)

# --- stack frames_per_im dimension ---
i_spikes = torch.stack(i_spikes_fpi, dim=2)   # (T, n_eyescales, n_frames_per_im)
i_rates  = torch.stack(i_rates_fpi,  dim=2)   # (T, n_eyescales, n_frames_per_im)
I_t      = torch.stack(I_t_fpi,      dim=3)   # (T, C, n_eyescales, n_frames_per_im)

#%%
# Now plot the results
n_eyescales = i_spikes.shape[1]
n_fpi = i_spikes.shape[2]

# We can make a 2D plot with n_fpi on xaxis and n_eyescales on yaxis,
# and color is the spatial information (averaged over time)
mean_ispike = i_spikes.mean(0).numpy()  # (n_eyescales, n_fpi)
mean_irate  = i_rates.mean(0).numpy()   # (n_eyescales, n_fpi)
# To make the plot coherence we should normalize by the null condition (eye scale = 0)
mean_ispike_norm = mean_ispike / mean_ispike[0:1,:]
mean_irate_norm  = mean_irate / mean_irate[0:1,:]
# We can leave out the first row now that it's normalized
mean_ispike_norm = mean_ispike_norm[1:,:]
mean_irate_norm  = mean_irate_norm[1:,:]

# Plot and save to figures folder
import seaborn as sns
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
#round to 2 decimal places
xpoints=120/np.array(frames_per_im_list[1:])
ypoints=eye_scale_list[1:]
sns.heatmap(mean_ispike_norm, xticklabels=np.round(xpoints,2), 
            yticklabels=np.round(ypoints,2), annot=True, fmt=".2f")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Frames per second')
plt.ylabel('Eye scale factor')
plt.title('Normalized Mean Spatial Information (bits/spike)')  
plt.subplot(1,2,2)
sns.heatmap(mean_irate_norm, xticklabels=np.round(xpoints,2), 
            yticklabels=np.round(ypoints,2), annot=True, fmt=".2f")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Frames per second')
plt.ylabel('Eye scale factor')
plt.title('Normalized Mean Spatial Information Rate (bits/sec)')  
plt.show()
plt.savefig(f'../figures/normalized_mean_spatial_information_rate_{itrial}.png')


#%%
'''
Full loop over trials!
This method now loops over different frames per image and different eye scales,
to get a full matrix of spatial information as a function of these two variables.
Null condition is eye scale = 0.0, but note this is now handled differently.

We want to also loop over trials and get mean, but for now just do one trial (itrial).
'''
import pickle
rerun = False
if rerun:
    i_spikes_trial = []
    i_rates_trial  = []
    I_t_trial      = []
    # breaks on longest fixations,but really shouldn't
    #trial_list = np.argsort(fix_dur)[::-1](trial_list[:70]):
    # pick 50 random trials
    import random
    random.seed(41)
    trial_list_rand = random.sample(range(eyepos.shape[0]), 50)
    for itrial in tqdm(trial_list_rand):
        i_spikes_fpi = []
        i_rates_fpi  = []
        I_t_fpi      = []

        frames_per_im_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        #frames_per_im_list = [1.88, 3.75, 7.5, 15, 30, 60]
        # eye_scale_list = [0.0,0.25, 0.5, 1.0, 2.0, 4.0]
        eye_scale_list = [0.0] + list(np.exp(np.linspace(-2.75, np.log(2), 11))) # odd to include 1.0 in the middle
        for frames_per_im in frames_per_im_list:
            print(f"Frames per image: {frames_per_im}")
            full_stack = make_stimulus_stack(type='fixrsvp',
                    frame=None, frames_per_im=frames_per_im, num_frames=540)
            print(f"full_stack shape: {full_stack.shape}")
            y_scales, eye_stim_scales = get_trial_stim_and_rates_eyescaled(eyepos[itrial], 
                    full_stack, out_size=out_size, n_lags=n_lags, stim_scale=scale, eye_scale=np.array(eye_scale_list), plot=False)

            print(f"y_scales shape: {y_scales.shape}")

            ispike_list = []
            irate_list = []
            I_t_list = []

            for i in range(y_scales.shape[4]):
                y = y_scales[...,i] # (T, C, H_out, W_out)
                
                # this returns ispike(T), irate(T), I_t(T,C)
                ispike_loop, irate_loop, I_t_loop = spatial_ssi_population(y) 
                
                ispike_list.append(ispike_loop)   # (T,)
                irate_list.append(irate_loop)     # (T,)
                I_t_list.append(I_t_loop)          # (T, C)

            # we can only stack if they are the same size, so we need to pad them
            # to the max size
            T_max = max([x.shape[0] for x in ispike_list])
            for j in range(len(ispike_list)):
                T_curr = ispike_list[j].shape[0]
                if T_curr < T_max:
                    pad_size = T_max - T_curr
                    ispike_list[j] = torch.cat([ispike_list[j], torch.zeros(pad_size)], dim=0)
                    irate_list[j] = torch.cat([irate_list[j], torch.zeros(pad_size)], dim=0)
                    I_t_list[j] = torch.cat([I_t_list[j], torch.zeros(pad_size, I_t_list[j].shape[1])], dim=0)
            i_spikes = torch.stack(ispike_list, dim=1)   # (T, n_eyescales)
            i_rates  = torch.stack(irate_list, dim=1)    # (T, n_eyescales)
            I_t      = torch.stack(I_t_list, dim=2)      # (T, C, n_eyescales)
                
            # collect per frames_per_im
            i_spikes_fpi.append(i_spikes)
            i_rates_fpi.append(i_rates)
            I_t_fpi.append(I_t)

        # --- stack frames_per_im dimension ---
        T_max = max([x.shape[0] for x in i_spikes_fpi])
        for j in range(len(i_spikes_fpi)):
            T_curr = i_spikes_fpi[j].shape[0]
            if T_curr < T_max:
                pad_size = T_max - T_curr
                i_spikes_fpi[j] = torch.cat([i_spikes_fpi[j], torch.zeros(pad_size, i_spikes_fpi[j].shape[1])], dim=0)
                i_rates_fpi[j] = torch.cat([i_rates_fpi[j], torch.zeros(pad_size, i_rates_fpi[j].shape[1])], dim=0)
                I_t_fpi[j] = torch.cat([I_t_fpi[j], torch.zeros(pad_size, I_t_fpi[j].shape[1], I_t_fpi[j].shape[2])], dim=0)
        i_spikes = torch.stack(i_spikes_fpi, dim=2)   # (T, n_eyescales, n_frames_per_im)
        i_rates  = torch.stack(i_rates_fpi,  dim=2)   # (T, n_eyescales, n_frames_per_im)
        I_t      = torch.stack(I_t_fpi,      dim=3)   # (T, C, n_eyescales, n_frames_per_im)
        
        # collect per trial
        i_spikes_trial.append(i_spikes)
        i_rates_trial.append(i_rates)
        I_t_trial.append(I_t)

    # --- stack trial dimension ---
    T_max = max([x.shape[0] for x in i_spikes_trial])
    for j in range(len(i_spikes_trial)):
        T_curr = i_spikes_trial[j].shape[0]
        if T_curr < T_max:
            pad_size = T_max - T_curr
            i_spikes_trial[j] = torch.cat([i_spikes_trial[j], torch.zeros(pad_size, i_spikes_trial[j].shape[1], i_spikes_trial[j].shape[2])], dim=0)
            i_rates_trial[j] = torch.cat([i_rates_trial[j], torch.zeros(pad_size, i_rates_trial[j].shape[1], i_rates_trial[j].shape[2])], dim=0)
            I_t_trial[j] = torch.cat([I_t_trial[j], torch.zeros(pad_size, I_t_trial[j].shape[1], I_t_trial[j].shape[2], I_t_trial[j].shape[3])], dim=0)
    i_spikes = torch.stack(i_spikes_trial, dim=3)   # (T, n_eyescales, n_frames_per_im, n_trials)
    i_rates  = torch.stack(i_rates_trial,  dim=3)   # (T, n_eyescales, n_frames_per_im, n_trials)
    I_t      = torch.stack(I_t_trial,      dim=4)   # (T, C, n_eyescales, n_frames_per_im, n_trials)
    
    # Save the outputs

    output_data = {
        'i_spikes': i_spikes,
        'i_rates': i_rates,
        'I_t': I_t,
        'frames_per_im_list': frames_per_im_list,
        'eye_scale_list': eye_scale_list,
        'eyepos': eyepos,
    }
    with open('../declan/spatial_info_fixrsvp_eye_scales_frames_per_im.pkl', 'wb') as f:
        pickle.dump(output_data, f)
else:
    with open('../declan/spatial_info_fixrsvp_eye_scales_frames_per_im.pkl', 'rb') as f:
        data = pickle.load(f)

    # Access the data
    i_spikes = data['i_spikes']
    i_rates = data['i_rates']
    I_t = data['I_t']
    frames_per_im_list = data['frames_per_im_list']
    eye_scale_list = data['eye_scale_list']
    eyepos = data['eyepos']
#%%
# Now plot the results across trials, first we normalize against the null condition
n_eyescales = i_spikes.shape[1]
n_fpi = i_spikes.shape[2]
n_trials = i_spikes.shape[3]

# We can make a 2D plot with n_fpi on xaxis and n_eyescales on yaxis,
# and color is the spatial information (averaged over time). Mean over time first?
mean_ispike = i_spikes.mean(0).numpy()  # (n_eyescales, n_fpi)
mean_irate  = i_rates.mean(0).numpy()   # (n_eyescales, n_fpi)
# To make the plot coherence we should normalize by the null condition (eye scale = 0)
mean_ispike_norm = mean_ispike / mean_ispike[0:1,:]
mean_irate_norm  = mean_irate / mean_irate[0:1,:]
# We can leave out the first row now that it's normalized
mean_ispike_norm = mean_ispike_norm[1:,:]
mean_irate_norm  = mean_irate_norm[1:,:]
# Mean over trials
mean_ispike_norm = mean_ispike_norm.mean(-1)
mean_irate_norm = mean_irate_norm.mean(-1)

# Plot and save to figures folder
import seaborn as sns
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
#round to 2 decimal places
xpoints=120/np.array(frames_per_im_list[1:])
ypoints=eye_scale_list[1:]
sns.heatmap(mean_ispike_norm, xticklabels=np.round(xpoints,2), 
            yticklabels=np.round(ypoints,2), annot=True, fmt=".2f")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Frames per second')
plt.ylabel('Eye scale factor')
plt.title('Normalized Mean Spatial Information (bits/spike)')  
plt.subplot(1,2,2)
sns.heatmap(mean_irate_norm, xticklabels=np.round(xpoints,2), 
            yticklabels=np.round(ypoints,2), annot=True, fmt=".2f")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Frames per second')
plt.ylabel('Eye scale factor')
plt.title('Normalized Mean Spatial Information Rate (bits/sec)')  
plt.show()
# save figure, mean over trials. Making sure it saves correctly
plt.savefig(f'../figures/normalized_mean_spatial_information_rate_50trials.png')

#%%
# check without normalization against null condition
# Now plot the results across trials, first we normalize against the null condition
n_eyescales = i_spikes.shape[1]
n_fpi = i_spikes.shape[2]
n_trials = i_spikes.shape[3]

# We can make a 2D plot with n_fpi on xaxis and n_eyescales on yaxis,
# and color is the spatial information (averaged over time). Mean over time first?
mean_ispike = i_spikes.mean(0).numpy()  # (n_eyescales, n_fpi)
mean_irate  = i_rates.mean(0).numpy()   # (n_eyescales, n_fpi)
# To make the plot coherence we should normalize by the null condition (eye scale = 0)
# mean_ispike_norm = mean_ispike / mean_ispike[0:1,:]
# mean_irate_norm  = mean_irate / mean_irate[0:1,:]
# # We can leave out the first row now that it's normalized
# mean_ispike_norm = mean_ispike_norm[1:,:]
# mean_irate_norm  = mean_irate_norm[1:,:]
# Mean over trials
mean_ispike_norm = mean_ispike.mean(-1)
mean_irate_norm = mean_irate.mean(-1)
# Plot and save to figures folder
import seaborn as sns
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
#round to 2 decimal places
xpoints=120/np.array(frames_per_im_list)
ypoints=eye_scale_list
sns.heatmap(mean_ispike_norm, xticklabels=np.round(xpoints,2), 
            yticklabels=np.round(ypoints,2), annot=True, fmt=".2f")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Frames per second')
plt.ylabel('Eye scale factor')
plt.title('Normalized Mean Spatial Information (bits/spike)')  
plt.subplot(1,2,2)
sns.heatmap(mean_irate_norm, xticklabels=np.round(xpoints,2), 
            yticklabels=np.round(ypoints,2), annot=True, fmt=".2f")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.xlabel('Frames per second')
plt.ylabel('Eye scale factor')
plt.title('Normalized Mean Spatial Information Rate (bits/sec)')  
plt.show()
# save figure, mean over trials. Making sure it saves correctly
#plt.savefig(f'../figures/normalized_mean_spatial_information_rate_50trials.png')


#%% Now try for a single natural image to add in real saccades from that image, 
# interspersed with fixational eye movements of different length from our RSVP stim
# in order to manipulate saccade frequency. Here we just treat saccades as 
# instant jumps in eye position, but we use real positions of fixations from backimage
# stimuli viewed by the monkeys.
# Extract eye positions for a single natural image frame across all trials
from DataYatesV1.exp import BackImageTrial, get_trial_protocols
from DataYatesV1.utils.detect_saccades import detect_saccades
def get_fixations_for_backimage_across_sessions(
    model, sessions, image_file=None, n_images=6
):
    """
    Get fixation eye positions for multiple backimages across sessions.
    Aggregates fixation data (ignoring saccades for now).
    
    Parameters
    ----------
    model : VisionCore model
        Model with .names attribute
    sessions : list
        List of session names to process
    image_file : str, optional
        Specific image to analyze. If None, uses most common images.
    n_images : int
        Number of most-common images to return data for
    
    Returns
    -------
    results : dict
        Keys are image filenames, values are dicts with:
        - 'eyepos': (N_samples, 2) fixation eye positions
        - 'n_trials': number of trials
        - 'n_sessions': number of sessions with this image
    """
    
    from collections import Counter
    
    # First pass: count images across all sessions
    print("Scanning sessions for backimage data...")
    all_image_counts = Counter()
    
    for sess_idx, name in enumerate(sessions):
        try:
            dataset_idx = model.names.index(name)
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)
            
            inds = torch.concatenate([
                train_data.get_dataset_inds('backimage'),
                val_data.get_dataset_inds('backimage')
            ], dim=0)
            
            if len(inds) == 0:
                continue
            
            dataset = train_data.shallow_copy()
            dataset.inds = inds
            dset_idx = inds[:,0].unique().item()
            
            sess_obj = dataset.dsets[dset_idx].metadata['sess']
            exp = sess_obj.exp
            protocols = get_trial_protocols(exp)
            trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
            unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])
            
            backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
            backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
            
            if len(backimage_trial_inds) == 0:
                continue
            
            backimage_trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds]
            
            for t in backimage_trials:
                all_image_counts[t.image_file] += 1
                
        except Exception as e:
            print(f"  Warning: Failed to scan session {name}: {e}")
            continue
    
    if len(all_image_counts) == 0:
        raise ValueError("No backimage trials found across any session")
    
    # Get top N images
    top_images = [img for img, count in all_image_counts.most_common(n_images)]
    print(f"\nTop {n_images} images:")
    for img, count in all_image_counts.most_common(n_images):
        print(f"  {img}: {count} trials")
    
    # Second pass: collect fixation data for top images
    results = {}
    
    for image_file in top_images:
        print(f"\nCollecting fixations for '{image_file}'...")
        
        all_eyepos = []
        n_trials_total = 0
        n_sessions_with_image = 0
        
        for sess_idx, name in enumerate(sessions):
            try:
                dataset_idx = model.names.index(name)
                train_data, val_data, _ = load_single_dataset(model, dataset_idx)
                
                inds = torch.concatenate([
                    train_data.get_dataset_inds('backimage'),
                    val_data.get_dataset_inds('backimage')
                ], dim=0)
                
                if len(inds) == 0:
                    continue
                
                dataset = train_data.shallow_copy()
                dataset.inds = inds
                dset_idx = inds[:,0].unique().item()
                
                sess_obj = dataset.dsets[dset_idx].metadata['sess']
                exp = sess_obj.exp
                protocols = get_trial_protocols(exp)
                trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
                unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])
                
                backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
                backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
                
                if len(backimage_trial_inds) == 0:
                    continue
                
                backimage_trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds]
                
                # Find trials with this image
                trials_with_image = [(idx, t) for idx, t in zip(backimage_trial_inds, backimage_trials)
                                     if t.image_file == image_file]
                
                if len(trials_with_image) == 0:
                    continue
                
                # Get eye position data
                eyepos = dataset.dsets[dset_idx]['eyepos'][:].numpy()
                t_dpi = sess_obj.dpi['t_ephys'].values
                
                # Pre-compute saccade mask (from detect_saccades)
                saccades = detect_saccades(sess_obj)
                is_saccade_mask = np.zeros(len(t_dpi), dtype=bool)
                for sacc in saccades:
                    sacc_samples = (t_dpi >= sacc.start_time) & (t_dpi <= sacc.end_time)
                    is_saccade_mask |= sacc_samples
                
                # Collect fixation eye positions
                session_eyepos = []
                for global_trial_idx, trial_obj in trials_with_image:
                    trial_mask = (trial_inds == global_trial_idx)
                    trial_sample_inds = np.where(trial_mask)[0]
                    
                    if len(trial_sample_inds) == 0:
                        continue
                    
                    # Use saccade mask to exclude saccade periods
                    trial_is_saccade = is_saccade_mask[trial_sample_inds]
                    fixation_samples = trial_sample_inds[~trial_is_saccade]
                    
                    if len(fixation_samples) > 0:
                        session_eyepos.append(eyepos[fixation_samples])
                
                if session_eyepos:
                    all_eyepos.append(np.vstack(session_eyepos))
                    n_sessions_with_image += 1
                    n_trials_total += len(trials_with_image)
                    
                    print(f"  Session {sess_idx} ({name}): {len(trials_with_image)} trials, "
                          f"{sum(len(x) for x in session_eyepos)} fixation samples")
                
            except Exception as e:
                print(f"  Warning: Failed to process session {name}: {e}")
                continue
        
        if all_eyepos:
            eyepos_aggregated = np.vstack(all_eyepos)
            results[image_file] = {
                'eyepos': eyepos_aggregated,
                'n_trials': n_trials_total,
                'n_sessions': n_sessions_with_image,
            }
            print(f"  Total: {n_sessions_with_image} sessions, {n_trials_total} trials, "
                  f"{len(eyepos_aggregated)} fixation samples")
    
    return results


# %% Example usage - multiple images, multiple sessions (default to 3)
print("=" * 60)
print("BACKIMAGE FIXATION ANALYSIS")
print("=" * 60)
if rerun:
    results = get_fixations_for_backimage_across_sessions(
        model=model,
        sessions=sessions[:3],  # First 3 sessions
        n_images=6              # Top 6 most common images
    )

    # % Save results to pickle
    output_path = '../declan/backimage_fixation_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"✓ Saved results to {output_path}")
else:
    # Load results from pickle
    input_path = '../declan/backimage_fixation_results.pkl'
    with open(input_path, 'rb') as f:
        results = pickle.load(f)

# %% Visualize results
n_images = len(results)
sx = int(np.sqrt(n_images))
sy = int(np.ceil(n_images / sx))

fig, axes = plt.subplots(sy, sx, figsize=(4*sx, 5*sy))
axes = axes.flatten()

for idx, (image_file, data) in enumerate(sorted(results.items(), key=lambda x: -x[1]['n_trials'])):
    ax = axes[idx]
    eyepos = data['eyepos']
    n_trials = data['n_trials']
    n_sessions = data['n_sessions']
    
    # Try to get the image (from first available session)
    try:
        # Find first session with this image
        for name in sessions[:3]:
            dataset_idx = model.names.index(name)
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)
            
            inds = torch.concatenate([
                train_data.get_dataset_inds('backimage'),
                val_data.get_dataset_inds('backimage')
            ], dim=0)
            
            if len(inds) == 0:
                continue
            
            dataset = train_data.shallow_copy()
            dataset.inds = inds
            dset_idx = inds[:,0].unique().item()
            
            sess_obj = dataset.dsets[dset_idx].metadata['sess']
            exp = sess_obj.exp
            protocols = get_trial_protocols(exp)
            trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
            unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])
            
            backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
            backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
            
            backimage_trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds]
            
            for trial_idx, trial_obj in zip(backimage_trial_inds, backimage_trials):
                if trial_obj.image_file == image_file:
                    image = trial_obj.get_image()
                    break
            else:
                continue
            break
        
        # Display image with fixation heatmap
        img_height, img_width = image.shape
        img_height_deg = img_height / 37.5  # ppd
        img_width_deg = img_width / 37.5
        
        # Use actual image aspect ratio
        extent_h = img_height_deg / 2
        extent_w = img_width_deg / 2
        
        ax.imshow(image, cmap='gray', origin='upper',
                  extent=[-extent_w, extent_w, -extent_h, extent_h])
        
        # Fixation heatmap
        h, xedges, yedges = np.histogram2d(
            eyepos[:, 0], eyepos[:, 1],
            bins=50,
            range=[[-extent_w, extent_w], [-extent_h, extent_h]]
        )
        
        im = ax.imshow(h.T, extent=[-extent_w, extent_w, -extent_h, extent_h],
                       origin='lower', cmap='hot', aspect='auto', alpha=0.6)
        
        ax.set_title(f'{image_file}\n{n_trials} trials, {n_sessions} sessions, {len(eyepos)} samples',
                     fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading\n{image_file}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{image_file} - {n_trials} trials', fontsize=10)

# Hide unused subplots
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Fixation Heatmaps - {len(results)} Images Across {len(sessions[:3])} Sessions', 
             fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('../figures/backimage_fixation_heatmaps_multi_session.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Saved figure to ../figures/backimage_fixation_heatmaps_multi_session.png")
# %% Example usage - multiple sessions (optional)
# saccades_all, eyepos_fix_all, eyepos_sacc_all, n_trials_all, n_sess, image_file = \
#     get_saccades_and_fixations_across_sessions(model, sessions, image_file='Waterfall.jpg')

# %% TESTING SINGLE IMAGE HYBRID EYE TRACE GENERATION










#


# %% -----------------------------------------------------------------------------------
 # SINGLE NATURAL IMAGE HYBRID EYE TRACE GENERATION
 # methods to create hybrid eye traces with real fixational eye movements
 # from RSVP data and real saccade targets from backimage fixation data
 # to manipulate saccade frequency while keeping realistic eye movement statistics
 # -----------------------------------------------------------------------------------
# %% Load backimage fixation data
print("=" * 60)
print("HYBRID EYE TRACE ANALYSIS")
print("=" * 60)

# Load the precomputed backimage fixation results
import pickle
with open('../declan/backimage_fixation_results.pkl', 'rb') as f:
    backimage_results = pickle.load(f)

# Pick the image with most trials
image_file = max(backimage_results.items(), key=lambda x: x[1]['n_trials'])[0]
fixation_eyepos = backimage_results[image_file]['eyepos']

print(f"\nUsing image: {image_file}")
print(f"  Fixation samples available: {len(fixation_eyepos)}")





# %% -----------------------------------------------------------------------------------
# Workhouse function to create hybrid eye trace
#-----------------------------------------------------------------------------------
def create_hybrid_eye_trace(
    fixation_pool,
    saccade_targets,
    n_saccades=5,
    saccade_duration=6,
    total_duration=540,
    eye_scale=1.0,
    max_trim_frames=20,
    max_attempts=50,
    start_center=None,
    saccade_targets_seq=None,
    micro_bridge_frames=6    # short smooth bridge between stitched fixation bouts
):
    """
    Create a hybrid eye trace:
    - Fixations: sample real RSVP fixation bouts, zero-mean them, scale jitter by eye_scale,
      and re-center on the current fixation location.
    - Saccades: ballistic jumps to given targets (not scaled).
    - If a single bout cannot meet the target fixation length, stitch multiple bouts
      end-to-end with a short smooth bridge (microsaccade) between them, staying centered.

    Returns (hybrid_trace, saccade_mask, saccade_times, plan).
    """
    rng = np.random.default_rng()

    # Initial center
    if start_center is None:
        current_center = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)
    else:
        current_center = np.asarray(start_center, dtype=np.float32)
    initial_center = current_center.copy()

    # Precompute bout lengths
    bout_lengths = np.array([len(b) for b in fixation_pool], dtype=np.int32)

    def trim_bout(bout: np.ndarray, target_len: int) -> np.ndarray | None:
        """Allow small trimming from a longer bout to match target_len."""
        L = len(bout)
        if L == target_len:
            return bout
        if L > target_len and (L - target_len) <= max_trim_frames:
            start = rng.integers(0, L - target_len + 1)
            return bout[start:start + target_len]
        return None

    def compose_fixation_sequence(target_len: int) -> np.ndarray:
        """
        Build a fixation sequence of exact target_len by stitching multiple bouts.
        Each bout is zero-meaned, scaled by eye_scale, and re-centered at current_center.
        Insert a short smooth bridge between consecutive bouts to avoid jumps.
        """
        residual = int(target_len)
        parts = []
        sacc_mask_parts = []
        prev_last = None

        while residual > 0:
            # Reserve bridge frames if we already placed a segment
            avail_for_seg = residual if prev_last is None else max(residual - micro_bridge_frames, 1)

            # Rank candidates by |length - avail_for_seg|
            length_diff = np.abs(bout_lengths - avail_for_seg)
            ranked = np.argsort(length_diff)
            seg = None

            # Try top-K near candidates with trimming if needed
            k = min(50, len(ranked))
            for idx in rng.permutation(ranked[:k]):
                bout = fixation_pool[idx]
                if len(bout) == avail_for_seg:
                    seg = bout
                    break
                seg_try = trim_bout(bout, avail_for_seg)
                if seg_try is not None:
                    seg = seg_try
                    break

            # Fallback: random bouts + trimming
            if seg is None:
                for _ in range(max_attempts):
                    bout = fixation_pool[rng.integers(0, len(fixation_pool))]
                    if len(bout) == avail_for_seg:
                        seg = bout; break
                    seg_try = trim_bout(bout, avail_for_seg)
                    if seg_try is not None:
                        seg = seg_try; break

            # Last resort: take any bout and clamp to avail_for_seg
            if seg is None:
                bout = fixation_pool[rng.integers(0, len(fixation_pool))]
                if len(bout) >= avail_for_seg:
                    start = rng.integers(0, len(bout) - avail_for_seg + 1)
                    seg = bout[start:start + avail_for_seg]
                else:
                    # If even the shortest is too short, just take it and we’ll continue stitching
                    seg = bout

            # Zero-mean jitter -> scale -> re-center
            seg = seg.astype(np.float32)
            jitter = seg - seg.mean(axis=0, keepdims=True)
            seg_centered = current_center[None, :] + jitter * float(eye_scale)

            # If there is a previous segment, insert a short smooth bridge
            if prev_last is not None and residual > 0:
                bridge_len = min(micro_bridge_frames, residual)
                bridge = np.linspace(prev_last, seg_centered[0], num=bridge_len, dtype=np.float32)
                parts.append(bridge)
                sacc_mask_parts.append(np.zeros(bridge_len, dtype=bool))
                residual -= bridge_len

            # Append segment, truncating if necessary to respect residual
            take = min(len(seg_centered), residual)
            parts.append(seg_centered[:take])
            sacc_mask_parts.append(np.zeros(take, dtype=bool))
            residual -= take
            prev_last = parts[-1][-1]

        return np.vstack(parts), np.concatenate(sacc_mask_parts)

    hybrid_parts = []
    saccade_mask_parts = []
    saccade_times = []
    frame_idx = 0

    # Prepare saccade plan (for null matching)
    used_targets = []
    if saccade_targets_seq is not None:
        saccade_targets_seq = np.asarray(saccade_targets_seq, dtype=np.float32)
        assert len(saccade_targets_seq) == n_saccades, "saccade_targets_seq length must match n_saccades"

    # Compute per-fixation target length
    saccade_frames_total = int(n_saccades) * int(saccade_duration)
    fixation_frames_total = int(total_duration) - saccade_frames_total
    frames_per_fixation = max(10, fixation_frames_total // (n_saccades + 1))

    for i_fix in range(n_saccades + 1):
        remaining = total_duration - frame_idx
        reserve = (saccade_duration if i_fix < n_saccades else 0)
        target_len = int(min(frames_per_fixation, max(0, remaining - reserve)))
        if target_len <= 0:
            break

        # Compose fixation from one or more bouts with bridges (prefer single bout; stitch if needed)
        fix_seg, fix_mask = compose_fixation_sequence(target_len)
        hybrid_parts.append(fix_seg)
        saccade_mask_parts.append(fix_mask)
        frame_idx += len(fix_seg)

        # Insert saccade (not scaled) and update center
        if i_fix < n_saccades and frame_idx < total_duration:
            n_sacc_frames = int(min(saccade_duration, total_duration - frame_idx))
            if saccade_targets_seq is not None:
                sacc_target = saccade_targets_seq[i_fix].astype(np.float32)
            else:
                sacc_target = saccade_targets[rng.integers(0, len(saccade_targets))].astype(np.float32)
                used_targets.append(sacc_target)

            start_pos = hybrid_parts[-1][-1]
            end_pos = sacc_target
            saccade_path = np.linspace(start_pos, end_pos, num=n_sacc_frames, dtype=np.float32)

            hybrid_parts.append(saccade_path)
            saccade_mask_parts.append(np.ones(n_sacc_frames, dtype=bool))
            saccade_times.append((frame_idx, frame_idx + n_sacc_frames))
            frame_idx += n_sacc_frames
            current_center = end_pos

    # Concatenate and clamp to total_duration
    hybrid_trace = np.vstack(hybrid_parts)
    saccade_mask = np.concatenate(saccade_mask_parts)
    if len(hybrid_trace) > total_duration:
        hybrid_trace = hybrid_trace[:total_duration]
        saccade_mask = saccade_mask[:total_duration]

    plan = {
        'start_center': initial_center,
        'saccade_targets_seq': (saccade_targets_seq if saccade_targets_seq is not None
                                else (np.stack(used_targets) if used_targets else np.empty((0,2), dtype=np.float32)))
    }
    return hybrid_trace.astype(np.float32), saccade_mask, saccade_times, plan

# %% Extract all fixation periods from FIXRSVP data into a pool -----
# %% Helper: build fixation_pool from FIXRSVP with correct units (deg) and contiguous bouts
def build_fixation_pool_from_fixrsvp(model, sessions, ppd=37.50476617, min_fix_frames=20, amp_thresh_deg=1.0):
    """
    Returns a list of contiguous fixation bouts (np.array[T,2] in degrees),
    extracted from FIXRSVP data without time compression.
    Detects pixel units and converts to deg by dividing by ppd.
    """
    fixation_pool = []

    for name in sessions:
        try:
            dataset_idx = model.names.index(name)
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)

            inds = torch.concatenate([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp')
            ], dim=0)
            if len(inds) == 0:
                continue

            dataset = train_data.shallow_copy()
            dataset.inds = inds
            dset_idx = inds[:,0].unique().item()

            # Original timeline eyepos for this slice
            eyepos_all = dataset.dsets[dset_idx]['eyepos'][:].numpy()  # (N_samples, 2)
            trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
            trials = np.unique(trial_inds[~np.isnan(trial_inds)])

            # Detect unit: pixels vs degrees. If typical magnitude >> 1, treat as pixels.
            median_amp = np.nanmedian(np.hypot(eyepos_all[:,0], eyepos_all[:,1]))
            if median_amp > 5.0:
                print(f"[{name}] eyepos appears in pixels (median={median_amp:.2f}); converting to degrees (/ppd={ppd:.2f})")
                eyepos_all = eyepos_all / ppd

            # Fixation mask in degrees
            fixation_mask_all = (np.hypot(eyepos_all[:,0], eyepos_all[:,1]) < amp_thresh_deg)

            # Extract contiguous fixation bouts per trial
            for t in trials:
                trial_mask = (trial_inds == t)
                trial_trace = eyepos_all[trial_mask]               # (T_trial, 2)
                trial_fix = fixation_mask_all[trial_mask].astype(bool)

                idx_true = np.where(trial_fix)[0]
                if idx_true.size == 0:
                    continue
                split_pts = np.where(np.diff(idx_true) != 1)[0] + 1
                runs = np.split(idx_true, split_pts)

                for run in runs:
                    if run.size >= min_fix_frames:
                        bout = trial_trace[run]                    # (T_bout, 2) deg
                        fixation_pool.append(bout)

        except Exception as e:
            print(f"Failed to load FIXRSVP from session {name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("FIXATION POOL (contiguous bouts, degrees) SUMMARY")
    print("="*60)
    if len(fixation_pool):
        lengths = [len(b) for b in fixation_pool]
        amps = [np.linalg.norm(b - b.mean(0), axis=1).max() for b in fixation_pool]
        print(f"  N bouts: {len(fixation_pool)}")
        print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, med={np.median(lengths):.1f}")
        print(f"  Max within-bout amp (deg): p50={np.percentile(amps,50):.2f}, p90={np.percentile(amps,90):.2f}, p99={np.percentile(amps,99):.2f}")
    else:
        print("  WARNING: fixation_pool is empty")
    return fixation_pool



# %% Extract all fixation periods from FIXRSVP data into a pool -----
print("\n" + "=" * 60)
print("BUILDING FIXATION POOL FROM FIXRSVP DATA (contiguous bouts, degrees)")
print("=" * 60)

if rerun:
    # Replace the old slicing-by-fix_dur logic with proper bout extraction
    fixation_pool = build_fixation_pool_from_fixrsvp(model, sessions, ppd=ppd, min_fix_frames=20, amp_thresh_deg=1.0)
    # % Save fixation pool to pickle
    output_path = '../declan/fixrsvp_fixation_pool.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(fixation_pool, f)
else:
    # Load fixation pool from pickle
    input_path = '../declan/fixrsvp_fixation_pool.pkl'
    with open(input_path, 'rb') as f:
        fixation_pool = pickle.load(f)


# %% Load backimage fixation data and ensure degrees
print("=" * 60)
print("HYBRID EYE TRACE ANALYSIS")
print("=" * 60)

import pickle
with open('../declan/backimage_fixation_results.pkl', 'rb') as f:
    backimage_results = pickle.load(f)

# Pick the image with most trials
image_file = max(backimage_results.items(), key=lambda x: x[1]['n_trials'])[0]
fixation_eyepos = backimage_results[image_file]['eyepos'].astype(np.float32)

# Heuristic unit check for backimage eyepos; convert to degrees if needed
median_amp_bi = np.nanmedian(np.hypot(fixation_eyepos[:,0], fixation_eyepos[:,1]))
if median_amp_bi > 5.0:
    print(f"[BackImage:{image_file}] eyepos appears in pixels (median={median_amp_bi:.2f}); converting to degrees (/ppd={ppd:.2f})")
    fixation_eyepos = fixation_eyepos / ppd

print(f"\nUsing image: {image_file}")
print(f"  Fixation samples available: {len(fixation_eyepos)}")



# %% Test hybrid eye trace generation with the fixation pool
print("\n" + "=" * 60)
print("TESTING HYBRID EYE TRACE GENERATION")
print("=" * 60)
n_frames = 540
example_saccade_rates = [0, 2, 4]
example_eye_scales = [0.0, 1.0]  # visualize null vs real

# Add one extra column for a zoomed-in time series (column 4)
fig, axes = plt.subplots(len(example_saccade_rates), len(example_eye_scales) + 2,
                         figsize=(18, 10))

def get_fixation_windows(saccade_mask: np.ndarray):
    idx = np.where(~saccade_mask)[0]  # fixation frames
    if idx.size == 0:
        return []
    split_pts = np.where(np.diff(idx) != 1)[0] + 1
    runs = np.split(idx, split_pts)
    return [(r[0], r[-1] + 1) for r in runs]  # [start, end)

y_zoom = 0.5  # deg for zoomed time series

for i_sacc, sacc_rate in enumerate(example_saccade_rates):
    n_sacc = int(sacc_rate * (n_frames / 120))

    # Column indices
    col_scatter_0 = 0
    col_scatter_1 = 1
    col_timeseries = len(example_eye_scales)        # 3rd column
    col_zoom_timeseries = len(example_eye_scales) + 1  # 4th column

    # Scatter columns: real vs null for each eye_scale
    for i_eye, eye_scale in enumerate(example_eye_scales):
        # Real trace: scaled FEMs and re-centered at saccade landings
        real_trace, real_mask, real_sacc_times, plan = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=eye_scale
        )

        # Null trace: same saccade plan, no FEM jitter
        null_trace, null_mask, null_sacc_times, _ = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=0.0,
            start_center=plan['start_center'],
            saccade_targets_seq=plan['saccade_targets_seq']
        )

        # Plot 2D trajectory (overlay real vs null) in columns 1 and 2
        ax = axes[i_sacc, i_eye]  # i_eye = 0 or 1
        ax.scatter(real_trace[:, 0], real_trace[:, 1],
                   c=np.arange(len(real_trace)), cmap='viridis', s=2, alpha=0.6, label='real')
        ax.scatter(null_trace[:, 0], null_trace[:, 1],
                   c='gray', s=2, alpha=0.4, label='null')

        # Mark saccades (same for real and null)
        for s0, s1 in real_sacc_times:
            ax.plot(real_trace[s0:s1, 0], real_trace[s0:s1, 1], 'r-', lw=2, alpha=0.9)

        ax.set_title(f'Sacc={sacc_rate}Hz, Scale={eye_scale:.1f}x\n({len(real_sacc_times)} saccades)',
                     fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)

    # Column 3: full time series (real vs null) for both eye scales
    ax_ts = axes[i_sacc, col_timeseries]
    for eye_scale in example_eye_scales:
        real_trace, real_mask, real_sacc_times, plan = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=eye_scale
        )
        null_trace, null_mask, null_sacc_times, _ = create_hybrid_eye_trace(
            fixation_pool=fixation_pool,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames,
            eye_scale=0.0,
            start_center=plan['start_center'],
            saccade_targets_seq=plan['saccade_targets_seq']
        )

        ax_ts.plot(real_trace[:, 0], label=f'Real X ({eye_scale:.1f}x)', alpha=0.8, lw=1.2)
        ax_ts.plot(real_trace[:, 1], label=f'Real Y ({eye_scale:.1f}x)', alpha=0.8, lw=1.2, linestyle='-')
        ax_ts.plot(null_trace[:, 0], label='Null X', alpha=0.6, lw=1.0, linestyle='--', color='gray')
        ax_ts.plot(null_trace[:, 1], label='Null Y', alpha=0.6, lw=1.0, linestyle='--', color='gray')

    ax_ts.set_title(f'Time Series (Sacc={sacc_rate}Hz)', fontsize=10)
    ax_ts.set_xlabel('Frame')
    ax_ts.set_ylabel('Eye Position (deg)')
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    # Column 4: zoomed time series around longest fixation window (X only), axis limits only
    ax_zoomts = axes[i_sacc, col_zoom_timeseries]
    # Generate a single real/null pair at eye_scale=1.0 for zoom
    real_trace, real_mask, real_sacc_times, plan = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=1.0
    )
    null_trace, null_mask, _, _ = create_hybrid_eye_trace(
        fixation_pool=fixation_pool,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames,
        eye_scale=0.0,
        start_center=plan['start_center'],
        saccade_targets_seq=plan['saccade_targets_seq']
    )
    fix_windows = get_fixation_windows(real_mask)
    if len(fix_windows) > 0:
        lengths = [e - s for (s, e) in fix_windows]
        s, e = fix_windows[int(np.argmax(lengths))]
        center_x = real_trace[s:e, 0].mean()
        ax_zoomts.plot(np.arange(s, e), real_trace[s:e, 0], label='Real X', color='#1f77b4', lw=1.2)
        ax_zoomts.plot(np.arange(s, e), null_trace[s:e, 0], label='Null X', color='gray', lw=1.0, linestyle='--', alpha=0.8)
        ax_zoomts.set_xlim(s, e)
        ax_zoomts.set_ylim(center_x - y_zoom, center_x + y_zoom)
        ax_zoomts.set_title(f'Zoomed Time Series X (±{y_zoom} deg)\nSacc={sacc_rate}Hz', fontsize=10)
    else:
        ax_zoomts.plot(real_trace[:, 0], label='Real X', color='#1f77b4', lw=1.2)
        ax_zoomts.plot(null_trace[:, 0], label='Null X', color='gray', lw=1.0, linestyle='--', alpha=0.8)
        cx = real_trace[:, 0].mean()
        ax_zoomts.set_ylim(cx - y_zoom, cx + y_zoom)
        ax_zoomts.set_title(f'Zoomed Time Series X (±{y_zoom} deg)\n(no saccades)', fontsize=10)

    ax_zoomts.set_xlabel('Frame')
    ax_zoomts.set_ylabel('X (deg)')
    ax_zoomts.grid(True, alpha=0.3)
    ax_zoomts.legend(fontsize=8)

plt.suptitle('Hybrid Eye Trace Examples (RSVP fixations centered at saccade landings)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_examples_real_fixations.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved hybrid eye trace examples")


# %% Hybrid eye trace now looks ok, next step is to generate reconstructed stimulus
# from these eye traces and analyze the effect of saccade rate and eye scale
# on the reconstructed stimulus quality (compared to ground truth static image).
# We want to do this for natural images, and take into account that the saccades
# will move the eye to the interesting parts of the image based on the backimage fixation data.
# Its therefore very important to use the same image for reconstruction as was used
# in the backimage fixation data to get realistic results.

# -----------------------------------------------------------------------------------













# %% DEBUG: Natural image eye traces & reconstructed stimulus (NO VIDEO)

print("\n" + "=" * 60)
print("DEBUG: NATURAL IMAGE (STATIC) EYE TRACE VISUALIZATION")
print("=" * 60)

# Pick the first FIXRSVP natural image (frame=None will grab the default)
nat_frames_per_im = 540  # static for entire trial
nat_full_stack = make_stimulus_stack(
    type='fixrsvp',
    frame=None,           # first natural image
    frames_per_im=nat_frames_per_im,
    num_frames=540
)
print(f"Natural image stack shape: {nat_full_stack.shape}")

# Trajectory plots
n_frames = 540
example_saccade_rates = [0, 2, 4]  # Hz
example_eye_scales = [0.0, 1.0]

fig, axes = plt.subplots(len(example_saccade_rates), len(example_eye_scales) + 1,
                         figsize=(15, 10))

for i_sacc, sacc_rate in enumerate(example_saccade_rates):
    n_sacc = int(sacc_rate * (n_frames / 120))

    for i_eye, eye_scale in enumerate(example_eye_scales):
        hybrid_trace, sacc_times = create_hybrid_eye_trace(
            fixation_eyepos=fixation_eyepos,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames
        )
        mean_pos = hybrid_trace.mean(axis=0)
        hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale

        ax = axes[i_sacc, i_eye]
        ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
                   c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
        for sacc_start, sacc_end in sacc_times:
            ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
                    hybrid_trace_scaled[sacc_start:sacc_end, 1],
                    'r-', lw=2, alpha=0.8)
        ax.set_title(f'Sacc={sacc_rate}Hz, Scale={eye_scale:.1f}x\n({len(sacc_times)} saccades)',
                     fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)

    # time series for this saccade rate
    ax = axes[i_sacc, -1]
    for eye_scale in example_eye_scales:
        hybrid_trace, _ = create_hybrid_eye_trace(
            fixation_eyepos=fixation_eyepos,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames
        )
        mean_pos = hybrid_trace.mean(axis=0)
        hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
        ax.plot(hybrid_trace_scaled[:, 0], label=f'X (scale={eye_scale:.1f}x)', alpha=0.7, lw=1)
        ax.plot(hybrid_trace_scaled[:, 1] + 10, label=f'Y (scale={eye_scale:.1f}x)',
                alpha=0.7, lw=1, linestyle='--')

    ax.set_title(f'Time Series (Sacc={sacc_rate}Hz)', fontsize=10)
    ax.set_xlabel('Frame') 
    ax.set_ylabel('Eye Position (deg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hybrid Eye Trace Examples (Natural Image)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/nat_eye_trace_debug_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved trajectory visualization (natural image)")

# Reconstructed stimulus analysis (natural image, NO VIDEO)
print("\n" + "=" * 60)
print("DEBUG: NATURAL IMAGE RECONSTRUCTED STIMULUS ANALYSIS")
print("=" * 60)

conditions = [
    {'sacc_rate': 0, 'eye_scale': 0.0, 'label': 'nat_null_no_saccades_no_eye'},
    {'sacc_rate': 0, 'eye_scale': 1.0, 'label': 'nat_real_eye_no_saccades'},
    {'sacc_rate': 4, 'eye_scale': 1.0, 'label': 'nat_real_eye_4hz_saccades'},
]

max_T = 600
n_lags = 32

for cond in conditions:
    sacc_rate = cond['sacc_rate']
    eye_scale = cond['eye_scale']
    label = cond['label']

    print(f"\nProcessing: {label}")
    n_sacc = int(sacc_rate * (n_frames / 120))

    hybrid_trace, sacc_times = create_hybrid_eye_trace(
        fixation_eyepos=fixation_eyepos,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames
    )

    mean_pos = hybrid_trace.mean(axis=0)
    hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
    hybrid_trace_scaled = hybrid_trace_scaled[:n_frames]

    hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
    hybrid_padded[:len(hybrid_trace_scaled)] = hybrid_trace_scaled

    print(f"  Reconstructing stimulus...")
    y_real, _, eye_stim, _ = get_trial_stim_and_rates(
        hybrid_padded,
        nat_full_stack,          # use natural image stack
        out_size=(151, 151),
        n_lags=n_lags,
        scale=1.0,
        plot=False
    )

    frames = eye_stim[:, 0, -1, :, :].detach().cpu().numpy()
    print(f"  Frames shape: {frames.shape}")
    print(f"  Frame range: [{frames.min():.1f}, {frames.max():.1f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Mean reconstructed stimulus
    ax = axes[0]
    mean_stim = frames.mean(axis=0)
    im = ax.imshow(mean_stim, cmap='gray')
    ax.set_title(f'Mean Reconstructed Stimulus\n{label}', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Eye trajectory
    ax = axes[1]
    ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
               c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
    for sacc_start, sacc_end in sacc_times:
        ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
                hybrid_trace_scaled[sacc_start:sacc_end, 1],
                'r-', lw=2, alpha=0.8)
    ax.set_title('Eye Trajectory', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (deg)')
    ax.set_ylabel('Y (deg)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Sample frames (spatial profile)
    ax = axes[2]
    frame_indices = np.linspace(0, len(frames)-1, 5, dtype=int)
    for i, frame_idx in enumerate(frame_indices):
        ax.plot(frames[frame_idx], label=f'Frame {frame_idx}', alpha=0.7)
    ax.set_title('Sample Frames (Spatial Profile)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Space (pixels)')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Stimulus Analysis: {label}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/reconstructed_stim_analysis_{label}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Completed: {label}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE (Natural Image)")
print("=" * 60)
print("Generated artifacts:")
print("  - Trajectory visualization: nat_eye_trace_debug_trajectories.png")
print("  - Analysis plots: reconstructed_stim_analysis_nat_*.png")


#%%

# %% DEBUG: Zoomed-in eye trace examples (individual trials)

print("\n" + "=" * 60)
print("DEBUG: ZOOMED-IN EYE TRACE EXAMPLES (1-2 TRIALS)")
print("=" * 60)

n_frames = 540
example_conditions = [
    {'sacc_rate': 0, 'eye_scale': 1.0, 'label': 'real_eye_no_saccades'},
    {'sacc_rate': 4, 'eye_scale': 1.0, 'label': 'real_eye_4hz_saccades'},
]

for cond in example_conditions:
    sacc_rate = cond['sacc_rate']
    eye_scale = cond['eye_scale']
    label = cond['label']
    
    n_sacc = int(sacc_rate * (n_frames / 120))
    
    # Create ONE example trace
    hybrid_trace, sacc_times = create_hybrid_eye_trace(
        fixation_eyepos=fixation_eyepos,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames
    )
    
    mean_pos = hybrid_trace.mean(axis=0)
    hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
    
    print(f"\n{label}:")
    print(f"  Total duration: {len(hybrid_trace_scaled)} frames ({len(hybrid_trace_scaled)/120:.2f}s)")
    print(f"  Saccades: {len(sacc_times)}")
    print(f"  Eye position range: X=[{hybrid_trace_scaled[:,0].min():.2f}, {hybrid_trace_scaled[:,0].max():.2f}] deg")
    print(f"                      Y=[{hybrid_trace_scaled[:,1].min():.2f}, {hybrid_trace_scaled[:,1].max():.2f}] deg")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Full trial trajectory
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
               c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=2, alpha=0.6)
    for sacc_start, sacc_end in sacc_times:
        ax1.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
                hybrid_trace_scaled[sacc_start:sacc_end, 1],
                'r-', lw=2, alpha=0.8)
    ax1.set_title('Full Trial Trajectory', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X (deg)')
    ax1.set_ylabel('Y (deg)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Full trial time series
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(hybrid_trace_scaled[:, 0], label='X', alpha=0.7, lw=1)
    ax2.plot(hybrid_trace_scaled[:, 1], label='Y', alpha=0.7, lw=1)
    
    # Mark saccades
    for sacc_start, sacc_end in sacc_times:
        ax2.axvspan(sacc_start, sacc_end, alpha=0.2, color='red')
    
    ax2.set_title('Full Trial Time Series', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Eye Position (deg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Velocity profile
    ax3 = plt.subplot(3, 3, 3)
    velocity = np.sqrt(np.sum(np.diff(hybrid_trace_scaled, axis=0)**2, axis=1))
    ax3.plot(velocity, alpha=0.7, lw=1, color='purple')
    
    # Mark saccades
    for sacc_start, sacc_end in sacc_times:
        ax3.axvspan(sacc_start, sacc_end, alpha=0.2, color='red')
    
    ax3.set_title('Eye Velocity (deg/frame)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Velocity (deg/frame)')
    ax3.grid(True, alpha=0.3)
    
    # Panels 4-6: Zoom into first 3 saccades (or fixation period if no saccades)
    zoom_windows = []
    
    if len(sacc_times) > 0:
        # Create windows around first 3 saccades
        for i in range(min(3, len(sacc_times))):
            sacc_start, sacc_end = sacc_times[i]
            # Expand window by 50 frames before and after
            window_start = max(0, sacc_start - 50)
            window_end = min(len(hybrid_trace_scaled), sacc_end + 50)
            zoom_windows.append((window_start, window_end, f'Saccade {i+1}'))
    else:
        # No saccades: zoom into 3 equal fixation periods
        window_size = len(hybrid_trace_scaled) // 3
        for i in range(3):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            zoom_windows.append((window_start, window_end, f'Fixation {i+1}'))
    
    for panel_idx, (window_start, window_end, title) in enumerate(zoom_windows):
        ax = plt.subplot(3, 3, 4 + panel_idx)
        
        window_trace = hybrid_trace_scaled[window_start:window_end]
        ax.scatter(window_trace[:, 0], window_trace[:, 1],
                  c=np.arange(len(window_trace)), cmap='cool', s=10, alpha=0.7)
        
        # Mark saccade if in this window
        for sacc_start, sacc_end in sacc_times:
            if sacc_start >= window_start and sacc_end <= window_end:
                sacc_start_local = sacc_start - window_start
                sacc_end_local = sacc_end - window_start
                ax.plot(window_trace[sacc_start_local:sacc_end_local, 0],
                       window_trace[sacc_start_local:sacc_end_local, 1],
                       'r-', lw=2.5, alpha=0.9)
        
        ax.set_title(f'{title} (frames {window_start}-{window_end})', fontsize=10, fontweight='bold')
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)
    
    # Panels 7-9: Zoomed time series
    for panel_idx, (window_start, window_end, title) in enumerate(zoom_windows):
        ax = plt.subplot(3, 3, 7 + panel_idx)
        
        window_frames = np.arange(window_start, window_end)
        window_trace = hybrid_trace_scaled[window_start:window_end]
        
        ax.plot(window_frames - window_start, window_trace[:, 0], label='X', alpha=0.7, lw=1.5)
        ax.plot(window_frames - window_start, window_trace[:, 1], label='Y', alpha=0.7, lw=1.5)
        
        # Mark saccade
        for sacc_start, sacc_end in sacc_times:
            if sacc_start >= window_start and sacc_end <= window_end:
                ax.axvspan(sacc_start - window_start, sacc_end - window_start, 
                          alpha=0.2, color='red', label='Saccade')
        
        ax.set_title(f'{title} Time Series', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frame (within window)')
        ax.set_ylabel('Eye Position (deg)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Zoomed Eye Trace Analysis: {label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/eye_trace_zoomed_{label}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved: eye_trace_zoomed_{label}.png")

print("\n" + "=" * 60)
print("DEBUG COMPLETE (Zoomed Eye Traces)")
print("=" * 60)
print("Generated artifacts:")
print("  - eye_trace_zoomed_real_eye_no_saccades.png")
print("  - eye_trace_zoomed_real_eye_4hz_saccades.png")


# %% DEBUG: Visualize hybrid eye traces and reconstructed stimulus

print("\n" + "=" * 60)
print("DEBUG: HYBRID EYE TRACE VISUALIZATION")
print("=" * 60)

# Create a few example hybrid traces with different saccade rates
n_frames = 540
example_saccade_rates = [0, 2, 4]  # Hz
example_eye_scales = [0.0, 1.0]

fig, axes = plt.subplots(len(example_saccade_rates), len(example_eye_scales) + 1, 
                         figsize=(15, 10))

for i_sacc, sacc_rate in enumerate(example_saccade_rates):
    n_sacc = int(sacc_rate * (n_frames / 120))
    
    for i_eye, eye_scale in enumerate(example_eye_scales):
        # Create hybrid trace
        hybrid_trace, sacc_times = create_hybrid_eye_trace(
            fixation_eyepos=fixation_eyepos,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames
        )
        
        # Apply eye scale
        mean_pos = hybrid_trace.mean(axis=0)
        hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
        
        # Plot 2D trajectory
        ax = axes[i_sacc, i_eye]
        ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1], 
                  c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
        
        # Mark saccades in red
        for sacc_start, sacc_end in sacc_times:
            ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0], 
                   hybrid_trace_scaled[sacc_start:sacc_end, 1], 
                   'r-', lw=2, alpha=0.8)
        
        ax.set_title(f'Sacc={sacc_rate}Hz, Scale={eye_scale:.1f}x\n({len(sacc_times)} saccades)', 
                    fontsize=10)
        ax.set_xlabel('X (deg)')
        ax.set_ylabel('Y (deg)')
        ax.grid(True, alpha=0.3)
    
    # Plot time series for this saccade rate
    ax = axes[i_sacc, -1]
    
    for i_eye, eye_scale in enumerate(example_eye_scales):
        hybrid_trace, _ = create_hybrid_eye_trace(
            fixation_eyepos=fixation_eyepos,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames
        )
        
        mean_pos = hybrid_trace.mean(axis=0)
        hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
        
        # Plot X and Y separately
        ax.plot(hybrid_trace_scaled[:, 0], label=f'X (scale={eye_scale:.1f}x)', 
               alpha=0.7, lw=1)
        ax.plot(hybrid_trace_scaled[:, 1] + 10, label=f'Y (scale={eye_scale:.1f}x)', 
               alpha=0.7, lw=1, linestyle='--')
    
    ax.set_title(f'Time Series (Sacc={sacc_rate}Hz)', fontsize=10)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Eye Position (deg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Hybrid Eye Trace Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_debug_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved trajectory visualization")

# %% DEBUG: Create reconstructed stimulus videos for different conditions

print("\n" + "=" * 60)
print("DEBUG: RECONSTRUCTED STIMULUS VIDEOS")
print("=" * 60)

import imageio.v2 as imageio

# Create 3 example conditions
conditions = [
    {'sacc_rate': 0, 'eye_scale': 0.0, 'label': 'null_no_saccades_no_eye'},
    {'sacc_rate': 0, 'eye_scale': 1.0, 'label': 'real_eye_no_saccades'},
    {'sacc_rate': 4, 'eye_scale': 1.0, 'label': 'real_eye_4hz_saccades'},
]

max_T = 600
n_lags = 32

for cond in conditions:
    sacc_rate = cond['sacc_rate']
    eye_scale = cond['eye_scale']
    label = cond['label']
    
    print(f"\nProcessing: {label}")
    
    n_sacc = int(sacc_rate * (n_frames / 120))
    
    # Create hybrid trace
    hybrid_trace, sacc_times = create_hybrid_eye_trace(
        fixation_eyepos=fixation_eyepos,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,
        total_duration=n_frames
    )
    
    # Apply eye scale
    mean_pos = hybrid_trace.mean(axis=0)
    hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
    hybrid_trace_scaled = hybrid_trace_scaled[:n_frames]
    
    # Pad with NaNs
    hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
    hybrid_padded[:len(hybrid_trace_scaled)] = hybrid_trace_scaled
    
    # Get reconstructed stimulus
    print(f"  Reconstructing stimulus...")
    y_real, _, eye_stim, _ = get_trial_stim_and_rates(
        hybrid_padded,
        full_stack,
        out_size=(151, 151),
        n_lags=n_lags,
        scale=1.0,
        plot=False
    )
    
    # Extract the reconstructed stimulus movie
    # eye_stim shape: (T, 1, n_lags, H, W)
    # We'll use the last lag (most recent) for visualization
    frames = eye_stim[:, 0, -1, :, :].detach().cpu().numpy()
    
    print(f"  Frames shape: {frames.shape}")
    print(f"  Frame range: [{frames.min():.1f}, {frames.max():.1f}]")
    
    # Normalize to 0-255
    frames_norm = ((frames - frames.min()) / (frames.max() - frames.min() + 1e-6) * 255).astype(np.uint8)
    
    # Save video
    output_video = f'../figures/reconstructed_stim_{label}.mp4'
    print(f"  Saving video to {output_video}...")
    imageio.mimsave(output_video, frames_norm, fps=30, format='FFMPEG')
    print(f"  ✓ Saved")
    
    # Also create a static frame showing eye position overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Mean reconstructed stimulus
    ax = axes[0]
    mean_stim = frames.mean(axis=0)
    im = ax.imshow(mean_stim, cmap='gray')
    ax.set_title(f'Mean Reconstructed Stimulus\n{label}', fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Panel 2: Eye position trajectory
    ax = axes[1]
    ax.scatter(hybrid_trace_scaled[:, 0], hybrid_trace_scaled[:, 1],
              c=np.arange(len(hybrid_trace_scaled)), cmap='viridis', s=1, alpha=0.6)
    
    # Mark saccades
    for sacc_start, sacc_end in sacc_times:
        ax.plot(hybrid_trace_scaled[sacc_start:sacc_end, 0],
               hybrid_trace_scaled[sacc_start:sacc_end, 1],
               'r-', lw=2, alpha=0.8)
    
    ax.set_title('Eye Trajectory', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (deg)')
    ax.set_ylabel('Y (deg)')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Sample frames over time
    ax = axes[2]
    frame_indices = np.linspace(0, len(frames)-1, 5, dtype=int)
    for i, frame_idx in enumerate(frame_indices):
        ax.plot(frames[frame_idx], label=f'Frame {frame_idx}', alpha=0.7)
    
    ax.set_title('Sample Frames (Spatial Profile)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Space (pixels)')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Stimulus Analysis: {label}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/reconstructed_stim_analysis_{label}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Completed: {label}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
print("Generated videos and analysis plots:")
print("  - Trajectory visualization: hybrid_eye_trace_debug_trajectories.png")
print("  - Stimulus videos: reconstructed_stim_*.mp4")
print("  - Analysis plots: reconstructed_stim_analysis_*.png")
























# %% Create stimulus - STATIC single image (no flashing)
# For a single static image, use frames_per_im = total_frames
# This way the same image is shown for the entire duration
frames_per_im = 540  # Show single image for entire trial (no flash)
full_stack = make_stimulus_stack(type='fixrsvp',
                                 frame=None, 
                                 frames_per_im=frames_per_im, 
                                 num_frames=540)

print(f"Stimulus stack shape: {full_stack.shape}")
print(f"(Single static image for entire trial)")

# %% Minimal example: Single image, eye_scale=1.0, varying saccade frequencies
saccade_rates = [0, 2, 4, 8]  # Number of saccades per second (over 4.5s trial)
n_frames = 540
max_T = 600  # Pad to this length (like in your earlier code)

results_hybrid = {
    'saccade_rates': saccade_rates,
    'i_spikes': [],
    'i_rates': [],
    'I_t': [],
}

print("\n" + "=" * 60)
print("ANALYSIS: Saccade Frequency vs Spatial Information")
print("(Single static image, eye_scale=1.0)")
print("=" * 60)

for sacc_rate in saccade_rates:
    
    n_sacc = int(sacc_rate * (n_frames / 120))  # Convert to number of saccades per trial
    print(f"\nSaccade Rate: {sacc_rate} Hz, approximated to ({120*n_sacc/n_frames:.2f} Hz)")
    # Create hybrid trace using backimage saccade targets + backimage fixations
    hybrid_trace, sacc_times = create_hybrid_eye_trace(
        fixation_eyepos=fixation_eyepos,
        saccade_targets=fixation_eyepos,
        n_saccades=n_sacc,
        saccade_duration=6,  # ~50ms at 120 Hz
        total_duration=n_frames
    )
    
    # Pad to max_T with NaNs (required by get_trial_stim_and_rates)
    hybrid_padded = np.full((max_T, 2), np.nan)
    hybrid_padded[:len(hybrid_trace)] = hybrid_trace
    
    print(f"  Hybrid trace: {len(hybrid_trace)} valid frames, padded to {max_T}")
    print(f"  Saccades inserted: {len(sacc_times)}")
    
    # Compute rates (real eye trace with saccades)
    y_real, _, _, _ = get_trial_stim_and_rates(
        hybrid_padded,
        full_stack,
        out_size=(151, 151),
        n_lags=32,
        scale=1.0,
        plot=False
    )
    
    # Compute rates (null eye trace - central fixation only)
    null_trace = np.full((max_T, 2), np.nan)
    null_trace[:len(hybrid_trace), :] = 0.0  # Central fixation at (0, 0)
    
    y_null, _, _, _ = get_trial_stim_and_rates(
        null_trace,
        full_stack,
        out_size=(151, 151),
        n_lags=32,
        scale=1.0,
        plot=False
    )
    
    # Compute spatial info
    ispike_real, irate_real, I_t_real = spatial_ssi_population(y_real)
    ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)
    
    # Average over time to get scalar values
    ispike_real_val = ispike_real.mean().item() if isinstance(ispike_real, torch.Tensor) else ispike_real.mean()
    irate_real_val = irate_real.mean().item() if isinstance(irate_real, torch.Tensor) else irate_real.mean()
    ispike_null_val = ispike_null.mean().item() if isinstance(ispike_null, torch.Tensor) else ispike_null.mean()
    irate_null_val = irate_null.mean().item() if isinstance(irate_null, torch.Tensor) else irate_null.mean()
    
    results_hybrid['i_spikes'].append(ispike_real_val)
    results_hybrid['i_rates'].append(irate_real_val)
    results_hybrid['I_t'].append(I_t_real.mean().item() if isinstance(I_t_real, torch.Tensor) else I_t_real.mean())
    
    gain_spike = ispike_real_val / ispike_null_val if ispike_null_val > 0 else 0
    gain_rate = irate_real_val / irate_null_val if irate_null_val > 0 else 0
    
    print(f"  Spatial Info (bits/spike): real={ispike_real_val:.4f}, null={ispike_null_val:.4f}, gain={gain_spike:.2f}x")
    print(f"  Spatial Info (bits/sec): real={irate_real_val:.4f}, null={irate_null_val:.4f}, gain={gain_rate:.2f}x")
# %% Visualize results
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Bits per spike
ax = axes[0]
ax.plot(saccade_rates, results_hybrid['i_spikes'], 'o-', lw=2, ms=10, color='steelblue', label='Real eye trace')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Null (central fixation)')
ax.set_xlabel('Saccade Rate (Hz)', fontsize=12)
ax.set_ylabel('Spatial Information (bits/spike)', fontsize=12)
ax.set_title('Spatial Information vs Saccade Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Bits per second
ax = axes[1]
ax.plot(saccade_rates, results_hybrid['i_rates'], 's-', lw=2, ms=10, color='coral', label='Real eye trace')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Null (central fixation)')
ax.set_xlabel('Saccade Rate (Hz)', fontsize=12)
ax.set_ylabel('Spatial Information (bits/sec)', fontsize=12)
ax.set_title('Information Rate vs Saccade Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_saccade_frequency.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Saved figure to ../figures/hybrid_eye_trace_saccade_frequency.png")

# %% Save results
output_path = '../declan/hybrid_eye_trace_results.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results_hybrid, f)

print(f"✓ Saved results to {output_path}")

# %% -----------------------------------------------------------------------------------
#Full loop over eye scales and saccade rates
# % -----------------------------------------------------------------------------------
# % Full loop over eye scales and saccade rates
# %% Load backimage fixation data and get the actual image
print("=" * 60)
print("HYBRID EYE TRACE ANALYSIS - FULL PARAMETER SWEEP")
print("=" * 60)

# Load the precomputed backimage fixation results
with open('../declan/backimage_fixation_results.pkl', 'rb') as f:
    backimage_results = pickle.load(f)

# Pick the image with most trials
image_file = max(backimage_results.items(), key=lambda x: x[1]['n_trials'])[0]
fixation_eyepos = backimage_results[image_file]['eyepos']

print(f"\nUsing image: {image_file}")
print(f"  Fixation samples available: {len(fixation_eyepos)}")

# %% Load the actual backimage to use as stimulus
# Find the first session that has this image
backimage_image = None
for name in sessions[:3]:
    try:
        dataset_idx = model.names.index(name)
        train_data, val_data, _ = load_single_dataset(model, dataset_idx)
        
        inds = torch.concatenate([
            train_data.get_dataset_inds('backimage'),
            val_data.get_dataset_inds('backimage')
        ], dim=0)
        
        if len(inds) == 0:
            continue
        
        dataset = train_data.shallow_copy()
        dataset.inds = inds
        dset_idx = inds[:,0].unique().item()
        
        sess_obj = dataset.dsets[dset_idx].metadata['sess']
        exp = sess_obj.exp
        protocols = get_trial_protocols(exp)
        trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
        unique_trial_inds = np.unique(trial_inds[~np.isnan(trial_inds)])
        
        backimage_trial_inds = np.where(np.array(protocols) == 'BackImage')[0]
        backimage_trial_inds = backimage_trial_inds[np.isin(backimage_trial_inds, unique_trial_inds)]
        
        backimage_trials = [BackImageTrial(exp['D'][iT], exp['S']) for iT in backimage_trial_inds]
        
        for trial_obj in backimage_trials:
            if trial_obj.image_file == image_file:
                backimage_image = trial_obj.get_image()
                print(f"Loaded image from session: {name}")
                print(f"  Image shape: {backimage_image.shape}")
                break
        
        if backimage_image is not None:
            break
            
    except Exception as e:
        print(f"  Warning: Failed to load from session {name}: {e}")
        continue

if backimage_image is None:
    raise ValueError(f"Could not load image {image_file} from any session")


# %% Create stimulus stack from the actual backimage (NumPy, not torch)
# backimage_image is already a NumPy array
n_frames = 540
n_lags = 32  # must match the call to get_trial_stim_and_rates

# Create stimulus stack from the actual backimage (NumPy), long enough for T+n_lags
full_stack = np.repeat(
    backimage_image[np.newaxis, np.newaxis, ...].astype(np.float32),
    n_frames + n_lags,  # 540 + 32 = 572
    axis=0
)

print(f"Full stimulus stack shape: {full_stack.shape}")
print(f"(Single actual backimage for entire trial)")

# %% Parameter ranges
saccade_rates = [0, 1, 2, 4, 8]  # Hz
eye_scale_list = [0.0, 0.25, 0.5, 1.0, 2.0]  # Scale factor for fixational eye movements
n_frames = 540
max_T = 600

# Storage
results_full = {
    'saccade_rates': saccade_rates,
    'eye_scale_list': eye_scale_list,
    'i_spikes': [],  # Will be (n_saccade_rates, n_eye_scales)
    'i_rates': [],   # Will be (n_saccade_rates, n_eye_scales)
    'I_t': [],       # Will be (n_saccade_rates, n_eye_scales)
    'image_file': image_file,
}

print("\n" + "=" * 60)
print("PARAMETER SWEEP: Saccade Rate × Eye Scale")
print(f"Using actual backimage: {image_file}")
print("=" * 60)

#%%
# Add debugging right after get_trial_stim_and_rates in the loop:
for i_sacc, sacc_rate in enumerate(tqdm(saccade_rates, desc="Saccade Rates")):
    i_spikes_eye_scale = []
    i_rates_eye_scale = []
    I_t_eye_scale = []
    
    n_sacc = int(sacc_rate * (n_frames / 120))
    
    for i_eye, eye_scale in enumerate(eye_scale_list):
        
        # Create hybrid trace
        hybrid_trace, sacc_times = create_hybrid_eye_trace(
            fixation_eyepos=fixation_eyepos,
            saccade_targets=fixation_eyepos,
            n_saccades=n_sacc,
            saccade_duration=6,
            total_duration=n_frames
        )
        
        # Apply eye scale to fixational movements
        mean_pos = hybrid_trace.mean(axis=0)
        hybrid_trace_scaled = mean_pos + (hybrid_trace - mean_pos) * eye_scale
        
        # --- safety: enforce max length to match stimulus ---
        hybrid_trace_scaled = hybrid_trace_scaled[:n_frames]
        
        # Pad with NaNs
        hybrid_padded = np.full((max_T, 2), np.nan, dtype=np.float32)
        hybrid_padded[:len(hybrid_trace_scaled)] = hybrid_trace_scaled
        
        # DEBUG: Check input stimulus and eye trace
        if i_sacc == 0 and i_eye == 0:
            print(f"\n=== DEBUGGING INPUTS ===")
            print(f"full_stack shape: {full_stack.shape}")
            print(f"full_stack dtype: {full_stack.dtype}")
            print(f"full_stack min: {full_stack.min():.4f}, max: {full_stack.max():.4f}, mean: {full_stack.mean():.4f}")
            print(f"full_stack contains NaN: {np.isnan(full_stack).any()}")
            print(f"hybrid_padded shape: {hybrid_padded.shape}")
            print(f"hybrid_padded valid samples: {np.sum(~np.isnan(hybrid_padded[:,0]))}")
            print(f"Eye position range: X=[{hybrid_trace_scaled[:,0].min():.2f}, {hybrid_trace_scaled[:,0].max():.2f}], "
                  f"Y=[{hybrid_trace_scaled[:,1].min():.2f}, {hybrid_trace_scaled[:,1].max():.2f}]")
            print(f"Image dimensions (deg): {img_height_deg:.2f} x {img_width_deg:.2f}")
        
        # Compute rates (real eye trace with saccades)
        y_real, _, eye_stim, _ = get_trial_stim_and_rates(
            hybrid_padded,
            full_stack,
            out_size=(151, 151),
            n_lags=32,
            scale=1.0,
            plot=False
        )
        
        # DEBUG: Check reconstructed stimulus
        if i_sacc == 0 and i_eye == 0:
            print(f"\n=== DEBUGGING RECONSTRUCTED STIMULUS ===")
            print(f"eye_stim shape: {eye_stim.shape}")
            print(f"eye_stim min: {eye_stim.min().item():.4f}, max: {eye_stim.max().item():.4f}, mean: {eye_stim.mean().item():.4f}")
            print(f"eye_stim contains NaN: {torch.isnan(eye_stim).any().item()}")
            
            # Check if stimulus normalization is correct (should be ~[-1, 1] or [0, 1])
            print(f"\nStimulus normalization check:")
            print(f"  Expected range for model: [-1, 1] or [0, 1]")
            print(f"  Actual range: [{eye_stim.min().item():.4f}, {eye_stim.max().item():.4f}]")
        
        # DEBUG: Check y_real statistics
        if i_sacc == 0 and i_eye == 0:
            print(f"\n=== DEBUGGING y_real ===")
            print(f"y_real shape: {y_real.shape}")
            print(f"y_real min: {y_real.min().item() if not torch.isnan(y_real).all() else 'all NaN'}")
            print(f"y_real max: {y_real.max().item() if not torch.isnan(y_real).all() else 'all NaN'}")
            print(f"y_real contains NaN: {torch.isnan(y_real).any().item()}")
            
            # Try to isolate where NaNs come from
            with torch.no_grad():
                test_stim = eye_stim[:10]  # First 10 frames
                print(f"\nTesting model with first 10 frames...")
                print(f"  test_stim contains NaN: {torch.isnan(test_stim).any().item()}")
                
                # Normalize stimulus (model expects normalized input)
                test_stim_norm = (test_stim - 127.0) / 255.0
                print(f"  test_stim_norm range: [{test_stim_norm.min().item():.4f}, {test_stim_norm.max().item():.4f}]")
                print(f"  test_stim_norm contains NaN: {torch.isnan(test_stim_norm).any().item()}")
        
        # Compute spatial info
        ispike_real, irate_real, I_t_real = spatial_ssi_population(y_real)
        
        # Average over time to get scalar values
        ispike_real_val = ispike_real.mean().item()
        irate_real_val = irate_real.mean().item()
        I_t_real_val = I_t_real.mean().item()
        
        i_spikes_eye_scale.append(ispike_real_val)
        i_rates_eye_scale.append(irate_real_val)
        I_t_eye_scale.append(I_t_real_val)
        
        if i_eye == 0 or i_eye == len(eye_scale_list) - 1:
            print(f"  Sacc={sacc_rate}Hz, EyeScale={eye_scale:.2f}: "
                  f"bits/spike={ispike_real_val:.4f}, bits/sec={irate_real_val:.4f}")

# %% Save results
output_path = '../declan/hybrid_eye_trace_full_sweep_backimage.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(results_full, f)

print(f"\n✓ Saved results to {output_path}")

# %% Visualize as heatmaps with the actual image in the title
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Show the image in the top panels
img_extent = 5.0  # degrees
ax_img_left = axes[0, 0]
ax_img_right = axes[0, 1]

img_height, img_width = backimage_image.shape
img_height_deg = img_height / 37.5
img_width_deg = img_width / 37.5
extent_h = img_height_deg / 2
extent_w = img_width_deg / 2

ax_img_left.imshow(backimage_image, cmap='gray', origin='upper',
                   extent=[-extent_w, extent_w, -extent_h, extent_h])
ax_img_left.set_title(f'Stimulus: {image_file}', fontsize=12, fontweight='bold')
ax_img_left.set_xlabel('X (deg)')
ax_img_left.set_ylabel('Y (deg)')

# Fixation heatmap overlay (no normalization)
h, xedges, yedges = np.histogram2d(
    fixation_eyepos[:, 0], fixation_eyepos[:, 1],
    bins=50,
    range=[[-extent_w, extent_w], [-extent_h, extent_h]]
)
ax_img_right.imshow(backimage_image, cmap='gray', origin='upper',
                    extent=[-extent_w, extent_w, -extent_h, extent_h])

if h.max() > 0:
    # Use log scale to make sparse counts visible; remove norm=LogNorm for linear
    im_h = ax_img_right.imshow(
        h.T,
        extent=[-extent_w, extent_w, -extent_h, extent_h],
        origin='lower',
        cmap='hot',
        aspect='auto',
        alpha=0.5,
        norm=mpl.colors.LogNorm(vmin=1, vmax=h.max())
    )
    plt.colorbar(im_h, ax=ax_img_right, fraction=0.046, pad=0.04, label='Fixation count')
else:
    ax_img_right.text(0.5, 0.5, 'No fixation samples', ha='center', va='center',
                      transform=ax_img_right.transAxes, color='r')

ax_img_right.set_title(f'Fixation Heatmap ({len(fixation_eyepos)} samples)', 
                       fontsize=12, fontweight='bold')
ax_img_right.set_xlabel('X (deg)')
ax_img_right.set_ylabel('Y (deg)')

# Heatmaps (raw, not normalized)
ispike = results_full['i_spikes']
irate  = results_full['i_rates']

# Panel 3: Bits per spike (raw)
ax = axes[1, 0]
sns.heatmap(
    ispike,
    xticklabels=[f"{x:.2f}" for x in eye_scale_list],
    yticklabels=[f"{x}" for x in saccade_rates],
    annot=True, fmt=".3f", cmap='viridis', ax=ax
)
ax.set_xlabel('Fixational Eye Scale', fontsize=12)
ax.set_ylabel('Saccade Rate (Hz)', fontsize=12)
ax.set_title('Spatial Information (bits/spike)', fontsize=13, fontweight='bold')

# Panel 4: Bits per second (raw)
ax = axes[1, 1]
sns.heatmap(
    irate,
    xticklabels=[f"{x:.2f}" for x in eye_scale_list],
    yticklabels=[f"{x}" for x in saccade_rates],
    annot=True, fmt=".3f", cmap='viridis', ax=ax
)
ax.set_xlabel('Fixational Eye Scale', fontsize=12)
ax.set_ylabel('Saccade Rate (Hz)', fontsize=12)
ax.set_title('Information Rate (bits/sec)', fontsize=13, fontweight='bold')

plt.suptitle(f'Hybrid Eye Traces on Real Backimage: {image_file}', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_heatmaps_backimage.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved figure to ../figures/hybrid_eye_trace_heatmaps_backimage.png")

# %% Additional visualization: Line plots showing interaction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Use raw (not normalized) data
ispike = results_full['i_spikes']
irate = results_full['i_rates']

# Panel 1: Effect of eye scale at different saccade rates
ax = axes[0]
for i_sacc, sacc_rate in enumerate(saccade_rates):
    ax.plot(eye_scale_list, ispike[i_sacc], 
            'o-', lw=2, ms=8, label=f'{sacc_rate} Hz')
ax.set_xlabel('Fixational Eye Scale', fontsize=12)
ax.set_ylabel('Spatial Info (bits/spike)', fontsize=12)
ax.set_title('Effect of Fixational Eye Movements', fontsize=13, fontweight='bold')
ax.legend(title='Saccade Rate')
ax.grid(True, alpha=0.3)

# Panel 2: Effect of saccade rate at different eye scales
ax = axes[1]
for i_eye, eye_scale in enumerate(eye_scale_list):
    ax.plot(saccade_rates, irate[:, i_eye],
            's-', lw=2, ms=8, label=f'{eye_scale:.2f}x')
ax.set_xlabel('Saccade Rate (Hz)', fontsize=12)
ax.set_ylabel('Info Rate (bits/sec)', fontsize=12)
ax.set_title('Effect of Saccade Frequency', fontsize=13, fontweight='bold')
ax.legend(title='Eye Scale')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/hybrid_eye_trace_line_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved figure to ../figures/hybrid_eye_trace_line_plots.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Results shape: {results_full['i_spikes'].shape}")
print(f"  Saccade rates: {saccade_rates}")
print(f"  Eye scales: {eye_scale_list}")
# Find null and max conditions
null_idx = (0, 0)  # 0 Hz saccades, 0x eye scale
null_ispike = results_full['i_spikes'][null_idx]
null_irate = results_full['i_rates'][null_idx]
print(f"  Null condition (0 Hz, 0x): {null_ispike:.4f} bits/spike, {null_irate:.4f} bits/sec")

max_idx = np.unravel_index(results_full['i_spikes'].argmax(), results_full['i_spikes'].shape)
print(f"  Max condition: {results_full['i_spikes'].max():.4f} bits/spike")
print(f"    at ({saccade_rates[max_idx[0]]} Hz, {eye_scale_list[max_idx[1]]}x)")


# %% -----------------------------------------------------------------------------------

# %%
# units_to_show = np.argsort(I_t.mean(0)-I_t_null.mean(0)).numpy()[::-1][:25]

# if frame is None:
#     make_movie(y, save_path='counterfactual1', n_units_to_show=units_to_show)
#     make_movie(y_null, save_path='counterfactualnull', n_units_to_show=units_to_show)
# else:
#     make_movie(y, save_path=f'counterfactual1_frame{frame}', n_units_to_show=units_to_show)
#     make_movie(y_null, save_path=f'counterfactualnull_frame{frame}', n_units_to_show=units_to_show)

# %% 
# unit = -1
# # %%
# unit +=1

# ispike, irate, I_t = spatial_ssi_population(y[:,[unit]], dt=dt)
# ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null[:,[unit]], dt=dt)
# plt.figure()
# plt.plot(I_t)
# plt.plot(I_t_null)
# plt.show()

# H = y.shape[2]
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# vmin = y[:,unit,H//2,:].amin()
# vmax = y[:,unit,H//2,:].amax()
# plt.imshow(y[:,unit,H//2,:].detach().cpu(), vmin=vmin, vmax=vmax)
# plt.title('Real stim')

# plt.subplot(1,2,2)
# plt.imshow(y_null[:,unit,H//2,:].detach().cpu(), vmin=vmin, vmax=vmax)
# plt.title('Null stim')
# plt.colorbar()



# # %% Loop over trials and compute spatial information on the real stimulus
# from tqdm import tqdm
# frame = None # flashed
# type = 'fixrsvp' # fixrsvp stim
# scale = 1.0 # normal scale

# ispikes = []
# irates = []
# ispikes_null = []
# irates_null = []
# I_t_list = []
# I_t_null_list = []

# for itrial in tqdm(trial_list[:70]):
#     ix = (trials[itrial] == trial_inds) & fixation
#     if np.sum(ix) < 64:
#         continue
#     stim_inds_orig = np.where(ix)[0]

#     eyepos = dataset.dsets[dset_idx]['eyepos'][ix]
#     null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0)
#     eye_stim = make_counterfactual_stim(eyepos, type=type,
#         frame=frame, out_size=out_size, n_lags=n_lags, scale_factor=scale)
#     eye_stim_null = make_counterfactual_stim(null_eyepos, type=type,
#         frame=frame, out_size=out_size, n_lags=n_lags, scale_factor=scale)

#     y = run_model(model, eye_stim)
#     y_null = run_model(model, eye_stim_null)

#     # compute spatial info
#     ispike, irate, I_t = spatial_ssi_population(y)
#     ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

#     ispikes.append(ispike)
#     irates.append(irate)
#     ispikes_null.append(ispike_null)
#     irates_null.append(irate_null)
#     I_t_list.append(I_t)
#     I_t_null_list.append(I_t_null)

#     # v = out_size[0]/ppd
#     # plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
#     # plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
#     # plt.show()

# #%%

# plt.subplot(1,2,1)
# plt.plot(np.array(ispikes_null), np.array(ispikes), '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.xlabel('Bits/Spike(Null stim)')
# plt.ylabel('Bits/Spike (Real stim)')
# plt.title('Spatial Info (Units)')

# plt.subplot(1,2,2)
# plt.plot(np.array(irates_null), np.array(irates), '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.xlabel('Spatial Info Rate (Null stim)')
# plt.ylabel('Spatial Info Rate (Real stim)')
# plt.title('Spatial Info (Units)')


# #%%

# type = 'nat' # natural images
# scale = 1.0 # normal scale

# ispikes = []
# irates = []
# ispikes_null = []
# irates_null = []
# I_t_list = []
# I_t_null_list = []

# for itrial in tqdm(trial_list[:70]):
#     ix = (trials[itrial] == trial_inds) & fixation
#     if np.sum(ix) < 64:
#         continue
#     stim_inds_orig = np.where(ix)[0]

#     eyepos = dataset.dsets[dset_idx]['eyepos'][ix]
#     null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0)

#     for frame in range(32):
#         eye_stim = make_counterfactual_stim(eyepos, type=type,
#             frame=frame, out_size=out_size, n_lags=n_lags, scale_factor=scale)
#         eye_stim_null = make_counterfactual_stim(null_eyepos, type=type,
#             frame=frame, out_size=out_size, n_lags=n_lags, scale_factor=scale)

#         y = run_model(model, eye_stim)
#         y_null = run_model(model, eye_stim_null)

#         # compute spatial info
#         ispike, irate, I_t = spatial_ssi_population(y)
#         ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

#         ispikes.append(ispike)
#         irates.append(irate)
#         ispikes_null.append(ispike_null)
#         irates_null.append(irate_null)
#         I_t_list.append(I_t)
#         I_t_null_list.append(I_t_null)

#     # v = out_size[0]/ppd
#     # plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
#     # plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
#     # plt.show()

# #%%

# plt.subplot(1,2,1)
# plt.plot(np.array(ispikes_null), np.array(ispikes), '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.xlabel('Bits/Spike(Null stim)')
# plt.ylabel('Bits/Spike (Real stim)')
# plt.title('Spatial Info (Units)')

# plt.subplot(1,2,2)
# plt.plot(np.array(irates_null), np.array(irates), '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.xlabel('Spatial Info Rate (Null stim)')
# plt.ylabel('Spatial Info Rate (Real stim)')
# plt.title('Spatial Info (Units)')





# #%% Now do it over all sessions...

# for sess in sessions:
#     dataset_idx = model.names.index(sess)
#     print(f"Loading dataset {dataset_idx}: {model.names[dataset_idx]}")
#     train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

#     # Get fixrsvp trial indices
#     inds = torch.concatenate([
#         train_data.get_dataset_inds('fixrsvp'),
#         val_data.get_dataset_inds('fixrsvp')
#     ], dim=0)

#     dataset = train_data.shallow_copy()
#     dataset.inds = inds

#     dset_idx = inds[:,0].unique().item()
#     trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
#     trials = np.unique(trial_inds)
#     NT = len(trials)

#     fixation = np.hypot(
#         dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), 
#         dataset.dsets[dset_idx]['eyepos'][:,1].numpy()
#     ) < 1

#%%



# #%% Fisher information
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt

# def differentiable_grid_sample(image, grid):
#     """
#     A pure PyTorch implementation of grid_sample that supports Forward AD.
#     Assumes align_corners=False and padding_mode='zeros' (zeros everywhere outside).
    
#     Args:
#         image: (B, C, H, W)
#         grid:  (B, H_out, W_out, 2) in range [-1, 1]
#     """
#     B, C, H, W = image.shape
#     _, H_out, W_out, _ = grid.shape
    
#     # 1. Map grid coordinates [-1, 1] to pixel coordinates [0, H-1]
#     # Formula for align_corners=False: x_pix = (x_norm + 1) * W / 2 - 0.5
#     x = grid[..., 0]
#     y = grid[..., 1]
    
#     x_pix = (x + 1) * W * 0.5 - 0.5
#     y_pix = (y + 1) * H * 0.5 - 0.5
    
#     # 2. Get corner pixel coordinates
#     x0 = torch.floor(x_pix).long()
#     x1 = x0 + 1
#     y0 = torch.floor(y_pix).long()
#     y1 = y0 + 1
    
#     # 3. Clamp coords to be inside image for gathering (we will mask zeros later)
#     x0_clamped = torch.clamp(x0, 0, W - 1)
#     x1_clamped = torch.clamp(x1, 0, W - 1)
#     y0_clamped = torch.clamp(y0, 0, H - 1)
#     y1_clamped = torch.clamp(y1, 0, H - 1)
    
#     # 4. Gather pixel values
#     # Flatten image to (B, C, H*W) to use gather efficiently
#     image_flat = image.view(B, C, -1)
    
#     # Helper to calculate linear indices
#     def get_pixel_value(idx_x, idx_y):
#         # Linear index: y * W + x
#         # dimensions: (B, H_out, W_out)
#         lin_idx = idx_y * W + idx_x
#         # Expand for channels: (B, C, H_out, W_out)
#         lin_idx_expanded = lin_idx.unsqueeze(1).expand(-1, C, -1, -1)
#         # Flatten spatial for gather: (B, C, H_out*W_out)
#         lin_idx_flat = lin_idx_expanded.reshape(B, C, -1)
        
#         gathered = torch.gather(image_flat, 2, lin_idx_flat)
#         return gathered.reshape(B, C, H_out, W_out)

#     Ia = get_pixel_value(x0_clamped, y0_clamped) # Top-Left
#     Ib = get_pixel_value(x0_clamped, y1_clamped) # Bottom-Left
#     Ic = get_pixel_value(x1_clamped, y0_clamped) # Top-Right
#     Id = get_pixel_value(x1_clamped, y1_clamped) # Bottom-Right
    
#     # 5. Calculate interpolation weights
#     # wa = (x1 - x) * (y1 - y)
#     wa = (x1 - x_pix) * (y1 - y_pix)
#     wb = (x1 - x_pix) * (y_pix - y0)
#     wc = (x_pix - x0) * (y1 - y_pix)
#     wd = (x_pix - x0) * (y_pix - y0)
    
#     # Expand weights for channels
#     wa = wa.unsqueeze(1)
#     wb = wb.unsqueeze(1)
#     wc = wc.unsqueeze(1)
#     wd = wd.unsqueeze(1)
    
#     # 6. Compute interpolated value
#     out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
#     # 7. Apply Zero Padding (mask out values that were outside boundaries)
#     mask = (x_pix >= 0) & (x_pix < W - 1) & (y_pix >= 0) & (y_pix < H - 1)
#     mask = mask.unsqueeze(1) # (B, 1, H_out, W_out)
    
#     return out * mask.float()

# class DifferentiableStimulus(nn.Module):
#     """
#     Stage 1: Generates a high-resolution static 'world' image.
#     Parameterized by position, orientation, and size (LogMAR).
#     """
#     def __init__(self, 
#                  stim_type='E', 
#                  ppd=120, 
#                  canvas_size=(256, 256), 
#                  template_res=1024, 
#                  device='cuda'):
#         super().__init__()
#         self.stim_type = stim_type
#         self.ppd = ppd
#         self.canvas_size = canvas_size
#         self.device = device
        
#         sz_h = canvas_size[0] / ppd
#         sz_w = canvas_size[1] / ppd
#         self.extent = [-sz_w/2, sz_w/2, -sz_h/2, sz_h/2]
        
#         self.register_buffer('template', self._make_template(stim_type, template_res))
        
#     def _make_template(self, type, res):
#         xx = torch.linspace(-1, 1, res)
#         yy = torch.linspace(-1, 1, res)
#         y, x = torch.meshgrid(yy, xx, indexing='ij')
#         k = 200.0 
        
#         if type == 'E':
#             def box(x0, x1, y0, y1):
#                 return (torch.sigmoid(k * (x - x0)) * torch.sigmoid(k * (x1 - x)) *
#                         torch.sigmoid(k * (y - y0)) * torch.sigmoid(k * (y1 - y)))
#             shape = (box(-1.0, -0.6, -1.0, 1.0) +  
#                      box(-0.6, 1.0, 0.6, 1.0) +    
#                      box(-0.6, 1.0, -0.2, 0.2) +   
#                      box(-0.6, 1.0, -1.0, -0.6))   
#             shape = torch.clamp(shape, 0, 1)
#         else: 
#             shape = torch.sigmoid(k * (1 - x.abs())) * torch.sigmoid(k * (0.2 - y.abs()))

#         return shape.unsqueeze(0).unsqueeze(0).to(self.device)

#     def get_affine_matrix(self, theta, logmar):
#         B = theta.shape[0]
#         x_deg, y_deg, ori_deg = theta[:, 0], theta[:, 1], theta[:, 2]
#         H, W = self.canvas_size
        
#         # World Normalize
#         tx = x_deg * self.ppd / (W / 2.0)
#         ty = -y_deg * self.ppd / (H / 2.0)
#         T_vec = torch.stack([tx, ty], dim=1).unsqueeze(2)

#         angle = ori_deg * (np.pi / 180.0)
#         c, s = torch.cos(angle), torch.sin(angle)
#         row1 = torch.stack([c, s], dim=1)
#         row2 = torch.stack([-s, c], dim=1)
#         R_inv = torch.stack([row1, row2], dim=1) 

#         if isinstance(logmar, float): logmar = torch.full((B,), logmar, device=self.device)
#         size_pix = (5 * (10**logmar / 60.0)) * self.ppd
#         sx_inv = W / (size_pix + 1e-8)
#         sy_inv = H / (size_pix + 1e-8)
#         S_inv = torch.zeros_like(R_inv)
#         S_inv[:, 0, 0] = sx_inv
#         S_inv[:, 1, 1] = sy_inv

#         A = torch.bmm(S_inv, R_inv)
#         b = -torch.bmm(A, T_vec)
#         return torch.cat([A, b], dim=2)

#     def forward(self, theta, logmar=0.0):
#         B = theta.shape[0]
#         affine = self.get_affine_matrix(theta, logmar)
        
#         # NOTE: align_corners=False is critical to match the manual sampler logic
#         grid = F.affine_grid(affine, (B, 1, *self.canvas_size), align_corners=False)
        
#         # Replaced F.grid_sample with differentiable_grid_sample
#         return differentiable_grid_sample(self.template.expand(B,-1,-1,-1), grid)

# class DifferentiableRetina(nn.Module):
#     """
#     Optimized Stage 2: Samples the high-res world image along a trajectory.
#     Uses (Space, Time) grid trick to avoid expanding the source image.
#     """
#     def __init__(self, ppd, world_canvas_size, retina_size=(32, 32)):
#         super().__init__()
#         self.ppd = ppd
#         self.world_h, self.world_w = world_canvas_size
#         self.retina_h, self.retina_w = retina_size
#         self.n_pixels = self.retina_h * self.retina_w
        
#         # 1. Pre-compute flattened Base Grid (centered at 0)
#         xs_pix = torch.linspace(-self.retina_w/2 + 0.5, self.retina_w/2 - 0.5, self.retina_w)
#         ys_pix = torch.linspace(-self.retina_h/2 + 0.5, self.retina_h/2 - 0.5, self.retina_h)
        
#         # Scale to World Norm coords [-1, 1]
#         xs_norm = xs_pix * (2.0 / self.world_w)
#         ys_norm = ys_pix * (2.0 / self.world_h)
        
#         grid_y, grid_x = torch.meshgrid(ys_norm, xs_norm, indexing='ij')
        
#         # Flatten to (P, 2)
#         # We use P = H*W as the "Height" dimension for grid_sample
#         self.register_buffer('base_grid_flat', torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1))

#     def forward(self, images, trajectories):
#         """
#         Args:
#             images: [B, 1, H_world, W_world] (High res world)
#             trajectories: [B, T, 2] (Eye positions in degrees)
            
#         Returns:
#             retinal_movie: [B, C, T, H_ret, W_ret]
#         """
#         B, T, _ = trajectories.shape
#         P = self.n_pixels
        
#         # 1. Convert Eye Position (deg) -> World Normalized Shift
#         x_deg = trajectories[:, :, 0]
#         y_deg = trajectories[:, :, 1]
        
#         shift_x = x_deg * self.ppd * (2.0 / self.world_w)
#         shift_y = -y_deg * self.ppd * (2.0 / self.world_h)
#         shifts = torch.stack([shift_x, shift_y], dim=-1) # [B, T, 2]
        
#         # 2. Construct Spatiotemporal Grid [B, P, T, 2]
#         # P (pixels) acts as "Height", T (time) acts as "Width" for grid_sample
        
#         # Base: [1, P, 1, 2]
#         base = self.base_grid_flat.unsqueeze(0).unsqueeze(2)
        
#         # Shifts: [B, 1, T, 2]
#         shifts = shifts.unsqueeze(1)
        
#         # Broadcast Sum: [B, P, T, 2]
#         # Every pixel P gets shifted by the eye position at time T
#         grid = base + shifts
        
#         # 3. Sample
#         # Replaced F.grid_sample
#         output = differentiable_grid_sample(images, grid)
        
#         # 4. Unflatten Space and Permute
#         output = output.view(B, 1, self.retina_h, self.retina_w, T)
#         output = output.permute(0, 1, 4, 2, 3) 
        
#         return output

# # ==========================================
# # Run Optimization Demo
# # ==========================================

# device = model.device

# # 1. Init
# # Canvas size needs to be large enough to contain the E and the full eye trace
# world_gen = DifferentiableStimulus(ppd=120, canvas_size=(512, 512), device=device)
# retina = DifferentiableRetina(ppd=37.50476617, world_canvas_size=(512, 512), retina_size=(151, 151))
# retina.to(device)

# # 2. Params
# theta = torch.tensor([[-0.0, 0.0, 30.0]], device=device, requires_grad=True) 

# # Random Walk Trace
# T_len = 50
# rw = torch.cumsum(torch.randn(1, T_len, 2, device=device)*0.05, dim=1)
# eye_trace = rw.clone().detach().requires_grad_(True)

# # 3. Forward (Fast!)
# # Stage 1: One World Image
# high_res_world = world_gen(theta, logmar=0.6) 

# # Stage 2: One Grid Sample call for the whole video
# movie = retina(high_res_world, eye_trace)

# # 4. Viz
# plt.figure(figsize=(10, 4))

# plt.subplot(131)
# world_np = high_res_world[0,0].detach().cpu().numpy()
# trace_np = eye_trace[0].detach().cpu().numpy()
# plt.imshow(world_np, extent=world_gen.extent, cmap='gray', origin='lower')
# plt.plot(trace_np[:,0], trace_np[:,1], 'r-', alpha=0.6)
# plt.title("World + Eye Trace")

# plt.subplot(132)
# # Show mean retinal activation over time
# mean_retina = movie[0,0].mean(dim=0).detach().cpu().numpy()
# plt.imshow(mean_retina, cmap='gray', origin='lower')
# plt.title("Average Retinal Input")

# # 5. Optimize Eye Trace to Maximize Energy
# loss = -torch.sum(movie**2)
# loss.backward()

# plt.subplot(133)
# # Visualize gradient on the eye trace itself
# grad_trace = eye_trace.grad[0].cpu().numpy()
# # Magnitude of gradient per time point
# grad_mag = np.linalg.norm(grad_trace, axis=1)
# plt.plot(grad_mag)
# plt.title("Gradient Magnitude on Eye Trace")
# plt.xlabel("Time")

# plt.tight_layout()
# plt.show()

# print(f"Movie Shape: {movie.shape} (B, C, T, H, W)")

# #%%
# import torch.autograd.forward_ad as fwAD

# def optimize_trajectory_for_fisher(
#     model, 
#     readout, 
#     stim_gen, 
#     retina, 
#     initial_eye_trace, 
#     base_theta, 
#     param_idx_to_maximize=2, # 2 = Orientation
#     n_steps=1000,
#     n_lags=32,
#     lr=1e-3,
# ):
#     """
#     Optimizes the eye trace to maximize the Fisher Information of the population 
#     with respect to a specific stimulus parameter (e.g., orientation).
    
#     Uses Forward-Mode AD to compute the Jacobian (dr/dtheta) efficiently,
#     and Reverse-Mode AD to optimize the trajectory (dFisher/dTrajectory).
#     """
    
#     # 1. Setup Optimization
#     # Clone trace and ensure it requires grad for the outer optimization loop
#     eye_trace = initial_eye_trace.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.AdamW([eye_trace], lr=lr, weight_decay=0.0)
#     T = eye_trace.shape[1]
#     indx = np.arange(T)[:,None] + np.arange(n_lags)[None,:]
#     indx = indx[:-n_lags]
#     # # SCHEDULER: Restarts every 100 steps
#     # # T_0=100: The first cycle is 100 steps
#     # # T_mult=2: Each subsequent cycle is 2x longer (allows deeper refining)
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     #     optimizer, T_0=100, T_mult=2, eta_min=lr_min
#     # )
    
#     # Ensure model parts are in eval mode (batch norm, dropout) but require grad
#     model.eval() 
#     readout.eval()
    
#     # History for plotting
#     fisher_history = []
#     trace_history = []
    
#     print(f"Optimizing Fisher Info for Theta index {param_idx_to_maximize}...")
    
#     for step in range(n_steps):
#         optimizer.zero_grad()
        
#         # ==========================================================
#         # Forward Mode AD Context: Compute dr/dtheta
#         # ==========================================================
#         with fwAD.dual_level():
#             # A. Create Dual Input for Stimulus Parameters
#             # We want the derivative w.r.t ONE parameter (e.g. orientation)
#             # Tangent vector is 1.0 for that parameter, 0.0 for others.
#             tangent = torch.zeros_like(base_theta)
#             tangent[:, param_idx_to_maximize] = 1.0
            
#             # dual_theta is the pair (value, derivative_seed)
#             dual_theta = fwAD.make_dual(base_theta, tangent)
            
#             # B. Forward Pass (Stimulus Generation)
#             # Everything here handles "Dual Tensors" automatically in PyTorch > 1.11
            
#             # 1. Generate High Res World (Dual)
#             world = stim_gen(dual_theta, logmar=0.6)
            
#             # 2. Retinal Sampling (Regular Tensor for trace, Dual for image)
#             # Note: eye_trace is NOT dual here because we are not differentiating 
#             # w.r.t eye_trace in the *inner* loop. We are diffing w.r.t theta.
#             movie = retina(world, eye_trace)

#             retinal_movie = movie[0,0,indx].unsqueeze(1)
            
#             # 3. Compute Rate Map
#             rates = compute_rate_map(model, readout, retinal_movie)
            
#             # C. Extract Jacobian
#             # The 'tangent' of the output is exactly \nabla_\theta rates
#             # shape: (Batch, N_neurons)
#             d_rates_d_theta = fwAD.unpack_dual(rates).tangent

#         # ==========================================================
#         # Outer Optimization: Maximize Fisher Information
#         # ==========================================================
        
#         # Fisher Information = Sum of squared gradients across population
#         # I(theta) = \sum_i (dr_i / dtheta)^2
#         # fisher_info = (d_rates_d_theta ** 2).sum() # gaussian 1 sd
#         epsilon = 1e-6
#         fisher_info = ((d_rates_d_theta ** 2) / (rates + epsilon)).sum() # poisson
        
#         # regularization (penalize the L2 norm of the 2nd derivative of the eye trace)
#         reg_lambda = 1e-3
#         reg_loss = reg_lambda * (torch.diff(eye_trace, n=2, dim=1) ** 2).sum()
        
#         # We minimize negative Fisher Info
#         loss = -fisher_info + reg_loss
        
#         # Check for NaN (Ruthless check)
#         if torch.isnan(loss):
#             raise ValueError("Optimization failed: Fisher Information is NaN. Check for exploding gradients or invalid masks.")
            
#         # Backpropagate through the entire chain (including the Forward AD steps)
#         loss.backward()
        
#         optimizer.step()
#         # scheduler.step()
        
#         fisher_history.append(fisher_info.item())
#         trace_history.append(eye_trace.detach().clone())
#         if step % 10 == 0:
#             print(f"Step {step:03d} | Fisher Info: {fisher_info.item():.4f}")

#     return eye_trace.detach(), fisher_history, trace_history

# # ==========================================
# # Usage Example
# # ==========================================
# # Assuming 'model.model.core' is your core module
# # Assuming 'readout' is your new PopulationReadout
# # theta = torch.tensor([[0.0, 0.0, 45.0]], device=model.device) # Batch size 1

# # eye_trace = torch.zeros(1, 33, 2, device=model.device)
# # # eye_trace = torch.cumsum(torch.randn(1, 50, 2, device=model.device)*0.05, dim=1)
# # optimized_trace, history, trace_history = optimize_trajectory_for_fisher(
# #     model=model,
# #     readout=readout,
# #     stim_gen=world_gen,
# #     retina=retina,
# #     initial_eye_trace=eye_trace, # From your snippet
# #     base_theta=theta
# # )


# #%%

# import torch
# import torch.nn as nn
# import torch.autograd.forward_ad as fwAD
# import numpy as np

# def optimize_trajectory_chunked(
#     model, 
#     readout, 
#     stim_gen, 
#     retina, 
#     initial_eye_trace, 
#     base_theta, 
#     param_idx_to_maximize=2,
#     n_steps=100,
#     n_lags=32,
#     lr=1e-3,
#     chunk_size=2,
#     reg_lambda=1e-3,
#     noise_model='poisson'
# ):
    
#     # 1. Setup
#     eye_trace = initial_eye_trace.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.Adam([eye_trace], lr=lr)
    
#     T = eye_trace.shape[1]
#     total_windows = T - n_lags
    
#     model.eval()
#     readout.eval()
    
#     loss_history = []
    
#     print(f"Optimizing with Chunk Size {chunk_size}...")

#     for step in range(n_steps):
#         optimizer.zero_grad()
#         total_fisher = 0
        
#         # ==========================================================
#         # Forward Mode AD
#         # ==========================================================
#         with fwAD.dual_level():
#             tangent = torch.zeros_like(base_theta)
#             tangent[:, param_idx_to_maximize] = 1.0
#             dual_theta = fwAD.make_dual(base_theta, tangent)
            
#             # 1. Generate World (Global)
#             # We assume the static world image fits in memory. 
#             # If 'world' is massive, move this inside the chunk loop too.
#             world = stim_gen(dual_theta, logmar=0.6)
            
#             # ==========================================================
#             # Chunked Processing
#             # ==========================================================
#             # Iterate through valid window starting points
#             for start_i in range(0, total_windows, chunk_size):
#                 # Defines the range of *windows* we want to compute
#                 end_i = min(start_i + chunk_size, total_windows)
                
#                 # We need movie data from 'start_i' up to the end of the last window
#                 # The last window starts at 'end_i - 1' and has length 'n_lags'
#                 # So we need indices up to (end_i - 1) + n_lags
#                 slice_end = (end_i - 1) + n_lags 
                
#                 # 2. Slice the Eye Trace (Preserves Gradients)
#                 # We grab just enough trace to generate the movie for these windows
#                 # shape: (1, 2, required_time)
#                 trace_chunk = eye_trace[:, :, start_i : slice_end]
                
#                 # 3. Lazy Retina Call
#                 # Only generates the movie for this specific time slice.
#                 # The graph created here is small and transient.
#                 # shape: (1, 1, required_time)
#                 movie_chunk = retina(world, trace_chunk)
                
#                 # 4. Prepare Model Input
#                 # We need to turn this continuous movie chunk into (Batch, 1, n_lags)
#                 # We use unfold on the time dimension
#                 # movie_chunk squeezed: (required_time)
#                 # unfolded: (current_batch_size, n_lags)
#                 movie_squeezed = movie_chunk.squeeze()
                
#                 # Note: unfold(dimension, size, step)
#                 input_windows = movie_squeezed.unfold(0, n_lags, 1).unsqueeze(1)
                
#                 # 5. Compute Rates & Gradients
#                 # Input shape: (Batch=chunk_size, 1, n_lags)
#                 rates = compute_rate_map(model, readout, input_windows)
                
#                 rates_primal = rates.primal
#                 d_rates_d_theta = rates.tangent
                
#                 # 6. Compute Fisher (Poisson)
#                 epsilon = 1e-6
#                 if noise_model == 'gaussian':
#                     fisher_per_element = d_rates_d_theta ** 2
#                 elif noise_model == 'poisson':
#                     fisher_per_element = (d_rates_d_theta ** 2) / (rates_primal + epsilon)
                
#                 # Sum over neurons and time (within chunk)
#                 chunk_fisher = fisher_per_element.sum()
                
#                 # 7. Backward Pass (Accumulate Gradients)
#                 # This frees the graph for 'movie_chunk', 'rates', and 'input_windows'
#                 (-chunk_fisher).backward()
                
#                 total_fisher += chunk_fisher.item()
                
#                 # Explicit cleanup to be safe
#                 del movie_chunk, input_windows, rates, d_rates_d_theta
        
#         # ==========================================================
#         # Regularization & Step
#         # ==========================================================
#         accel = torch.diff(eye_trace, n=2, dim=1)
#         reg_loss = reg_lambda * (accel ** 2).sum()
#         reg_loss.backward()
        
#         optimizer.step()
        
#         loss_history.append(total_fisher)
#         if step % 10 == 0:
#             print(f"Step {step:03d} | Total Fisher: {total_fisher:.4f}")

#     return eye_trace.detach(), loss_history

# theta = torch.tensor([[0.0, 0.0, 45.0]], device=model.device) # Batch size 1

# eye_trace = torch.zeros(1, 151, 2, device=model.device)
# # eye_trace = torch.cumsum(torch.randn(1, 50, 2, device=model.device)*0.05, dim=1)
# optimized_trace, history = optimize_trajectory_chunked(
#     model=model,
#     readout=readout,
#     stim_gen=world_gen,
#     retina=retina,
#     initial_eye_trace=eye_trace, # From your snippet
#     base_theta=theta
# )

# # %%

# plt.subplot(1,2,1)
# plt.plot(eye_trace[0].detach().cpu().numpy(), 'b-', label='Initial')
# plt.plot(optimized_trace[0].detach().cpu().numpy(), 'r--', label='Optimized')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Eye Position (deg)')

# # plot fourier power of the initial and optimized traces



# #%%
# plt.plot(history)
# plt.ylabel('Fisher Information')
# plt.xlabel('Optimization Step')


# # %%

# def visualize_results(stim_gen, retina, theta, initial_trace, optimized_trace, fisher_history):
#     # 1. Generate the High-Res World
#     # We detach everything to move to numpy
#     with torch.no_grad():
#         world = stim_gen(theta, logmar=0.6)
#         world_np = world[0, 0].cpu().numpy()
    
#     # 2. Process Traces
#     # Convert trace from degrees to pixels for plotting over the image
#     # We use the same logic as the DifferentiableStimulus to map deg -> pixels
#     def trace_to_pixels(trace, canvas_size, ppd):
#         trace_np = trace[0].detach().cpu().numpy()
#         H, W = canvas_size
        
#         # Invert the normalization done in get_affine_matrix
#         # trace x (deg) * ppd = pixels from center
#         x_pix = trace_np[:, 0] * ppd + (W / 2.0)
#         y_pix = -trace_np[:, 1] * ppd + (H / 2.0) # Note the negative for y-flip
#         return x_pix, y_pix

#     x_init, y_init = trace_to_pixels(initial_trace, stim_gen.canvas_size, stim_gen.ppd)
#     x_opt, y_opt = trace_to_pixels(optimized_trace, stim_gen.canvas_size, stim_gen.ppd)

#     # 3. Plotting
#     plt.figure(figsize=(12, 5))

#     # Panel A: The Optimization Landscape
#     plt.subplot(1, 2, 1)
#     plt.plot(fisher_history, 'k-', lw=1.5)
#     plt.title("Fisher Information Optimization")
#     plt.xlabel("Step")
#     plt.ylabel("Fisher Info (Dr/Dtheta)^2")
#     plt.grid(True, alpha=0.3)

#     # Panel B: The Trajectory
#     plt.subplot(1, 2, 2)
#     # Plot the world
#     plt.imshow(world_np, cmap='gray', origin='upper', 
#                extent=[0, stim_gen.canvas_size[1], stim_gen.canvas_size[0], 0])
    
#     # Plot Initial Trace (faint red)
#     plt.plot(x_init, y_init, 'r-', alpha=0.3, label='Initial (Random Walk)')
#     plt.plot(x_init[0], y_init[0], 'ro', alpha=0.3) # Start point
    
#     # Plot Optimized Trace (Green/Blue)
#     # We use a scatter to show velocity (points closer together = slower speed)
#     plt.plot(x_opt, y_opt, 'c-', lw=2, label='Optimized (Max Info)')
#     # plt.scatter(x_opt, y_opt, c=np.arange(len(x_opt)), cmap='viridis', s=20, zorder=3)
    
#     plt.legend(loc='upper right')
#     plt.title("Eye Trajectory on Stimulus")
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# # Run it
# visualize_results(
#     stim_gen=world_gen, 
#     retina=retina, 
#     theta=theta, 
#     initial_trace=eye_trace, 
#     optimized_trace=optimized_trace, 
#     fisher_history=history
# )

# # %%
# def visualize_rate_maps(model, readout, stim_gen, retina, theta, initial_trace, optimized_trace):
    
#     # 1. Compute Rates for both traces
#     model.eval()
#     readout.eval()
    
#     with torch.no_grad():
#         # Generate Inputs
#         world = stim_gen(theta, logmar=0.6)
        
#         # A. Initial
#         movie_init = retina(world, initial_trace)
#         rates_init = compute_rate_map(model, readout, movie_init) # Shape (1, N_units)
        
#         # B. Optimized
#         movie_opt = retina(world, optimized_trace)
#         rates_opt = compute_rate_map(model, readout, movie_opt)   # Shape (1, N_units)
        
#         # Move to CPU
#         r_init = rates_init[0].cpu().numpy()
#         r_opt = rates_opt[0].cpu().numpy()


#     # --- Plotting ---
#     plt.figure(figsize=(14, 6))
    
#     # Panel 1: Scatter Comparison
#     plt.subplot(1, 3, 1)
#     plt.scatter(r_init, r_opt, alpha=0.5, s=10, c='k')
#     plt.plot([0, r_opt.max()], [0, r_opt.max()], 'r--', alpha=0.5) # Identity line
#     plt.title("Firing Rate Comparison")
#     plt.xlabel("Initial Trace Rates")
#     plt.ylabel("Optimized Trace Rates")
#     plt.grid(True, alpha=0.3)
    
#     # Panel 2: Spatial Map (Initial)
#     plt.subplot(1, 3, 2)
#     plt.imshow(r_init.mean(0), cmap='inferno', origin='lower')
#     plt.title("Population Activity (Initial)")
#     plt.axis('off')
#     plt.colorbar(fraction=0.046, pad=0.04)

#     # Panel 3: Spatial Map (Optimized)
#     plt.subplot(1, 3, 3)
#     plt.imshow(r_opt.mean(0), cmap='inferno', origin='lower')
#     plt.title("Population Activity (Optimized)")
#     plt.axis('off')
#     plt.colorbar(fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.show()

# # Run it
# visualize_rate_maps(model, readout, world_gen, retina, theta, eye_trace, optimized_trace)
# # %%

# torch.cuda.empty_cache()
# #%%
# import torch.autograd.forward_ad as fwAD

# def optimize_batch_trajectories(
#     model, 
#     readout, 
#     stim_gen, 
#     retina, 
#     base_theta, 
#     batch_size=32, 
#     chunk_size=4,        # <--- Process only 4 walkers at a time to save memory
#     n_steps=150, 
#     lr=1e-2,
#     param_idx=2
# ):
#     """
#     Runs batch_size independent trajectory optimizations using Gradient Accumulation.
#     """
    
#     # 1. Initialize Batch of Random Walks (Same as before)
#     start_pos = (torch.rand(batch_size, 1, 2, device=base_theta.device) - 0.5) * 0.6
#     T_len = 50
#     walks = torch.cumsum(torch.randn(batch_size, T_len, 2, device=base_theta.device) * 0.05, dim=1)
#     eye_traces = (start_pos + walks).detach().requires_grad_(True)
    
#     optimizer = torch.optim.Adam([eye_traces], lr=lr)
#     model.eval()
#     readout.eval()
    
#     best_history = []
    
#     print(f"Launching {batch_size} optimizers (Chunk size {chunk_size})...")
    
#     for step in range(n_steps):
#         optimizer.zero_grad()
        
#         step_best_val = -float('inf')
        
#         # --- LOOP OVER CHUNKS ---
#         # We process walkers [0:4], then [4:8], etc.
#         for i in range(0, batch_size, chunk_size):
            
#             # Slice the current chunk of trajectories
#             # Gradients computed on 'trace_chunk' will flow back to 'eye_traces'
#             trace_chunk = eye_traces[i : i + chunk_size]
#             current_batch_size = trace_chunk.shape[0]
            
#             # Forward AD Context (Fresh for each chunk to free memory after)
#             with fwAD.dual_level():
#                 tangent = torch.zeros_like(base_theta)
#                 tangent[:, param_idx] = 1.0
#                 dual_theta = fwAD.make_dual(base_theta, tangent)
                
#                 # Generate World (1, 1, H, W) -> Expand to Chunk Size
#                 world_dual = stim_gen(dual_theta, logmar=0.6) 
#                 world_chunk = world_dual.expand(current_batch_size, -1, -1, -1)
                
#                 # Retina & Model (Process only chunk_size items)
#                 movie = retina(world_chunk, trace_chunk)
#                 rates = compute_rate_map(model, readout, movie)
                
#                 # Jacobian
#                 d_rates = fwAD.unpack_dual(rates).tangent
            
#             # Compute Loss for this chunk
#             # Fisher Info: Sum squares over (Time, Units, H, W)
#             fisher_info_chunk = (d_rates ** 2).sum(dim=(1, 2, 3))
            
#             # Accumulate Gradients
#             # We sum losses so that gradients add up in eye_traces.grad
#             loss = -fisher_info_chunk.sum()
#             loss.backward() 
            
#             # Track stats
#             chunk_max = fisher_info_chunk.max().item()
#             if chunk_max > step_best_val:
#                 step_best_val = chunk_max

#         # --- UPDATE ---
#         # After processing all chunks, eye_traces.grad contains gradients for all 32 walkers
#         optimizer.step()
        
#         best_history.append(step_best_val)
#         if step % 20 == 0:
#             print(f"Step {step:03d} | Best Fisher Info: {step_best_val:.4f}")

#     # --- Select Winner ---
#     # We need one final pass (chunked) to get the final scores without grad
#     final_scores = []
#     with torch.no_grad():
#         for i in range(0, batch_size, chunk_size):
#             trace_chunk = eye_traces[i : i + chunk_size]
#             current_batch_size = trace_chunk.shape[0]
            
#             # We can't use ForwardAD in no_grad mode easily for evaluation,
#             # but we just need to know who won. 
#             # Ideally, we track the winner index during the last optimization loop.
#             # For simplicity here, we assume the last step's scores are close enough.
#             pass

#     # Note: To be perfectly accurate, we should have stored the indices in the loop.
#     # But returning the trace at the index of the max *gradient magnitude* or just re-running
#     # the Forward AD one last time is fine.
#     # Let's just return the whole batch and let the visualizer pick the best?
#     # Or simplified: We just return the last calculated best from the loop.
    
#     # Quick fix to get exact winner index: Re-run Forward AD one last time (chunked)
#     print("Selecting final winner...")
#     all_scores = []
#     for i in range(0, batch_size, chunk_size):
#         with fwAD.dual_level():
#             tangent = torch.zeros_like(base_theta); tangent[:, param_idx] = 1.0
#             dual_theta = fwAD.make_dual(base_theta, tangent)
#             world_chunk = stim_gen(dual_theta, logmar=0.6).expand(eye_traces[i:i+chunk_size].shape[0],-1,-1,-1)
#             rates = compute_rate_map(model, readout, retina(world_chunk, eye_traces[i:i+chunk_size]))
#             d_rates = fwAD.unpack_dual(rates).tangent
#             scores = (d_rates ** 2).sum(dim=(1,2,3))
#             all_scores.append(scores.detach())
    
#     all_scores = torch.cat(all_scores)
#     final_idx = torch.argmax(all_scores)
#     winner_trace = eye_traces[final_idx].detach()
    
#     print(f"Winner: Walker {final_idx} | Info: {all_scores[final_idx].item():.4f}")
#     return winner_trace, eye_traces.detach(), best_history

# def visualize_batch_results(stim_gen, theta, winner_trace, all_traces, history):
#     # 1. Generate Static World for Background
#     with torch.no_grad():
#         world = stim_gen(theta, logmar=0.6)
#         world_np = world[0, 0].cpu().numpy()

#     # 2. Helper: Convert Trace to Pixels
#     def to_pix(trace):
#         # trace shape (T, 2)
#         trace = trace.cpu().numpy()
#         H, W = stim_gen.canvas_size
#         x = trace[:, 0] * stim_gen.ppd + (W / 2.0)
#         y = -trace[:, 1] * stim_gen.ppd + (H / 2.0)
#         return x, y

#     # 3. Plot
#     plt.figure(figsize=(12, 5))

#     # Panel A: Optimization History
#     plt.subplot(1, 2, 1)
#     plt.plot(history, 'k-', lw=2)
#     plt.title(f"Optimization (Best of {len(all_traces)})")
#     plt.xlabel("Step")
#     plt.ylabel("Max Fisher Info")
#     plt.grid(True, alpha=0.3)

#     # Panel B: Trajectory Cloud
#     plt.subplot(1, 2, 2)
#     plt.imshow(world_np, cmap='gray', origin='upper', 
#                extent=[0, stim_gen.canvas_size[1], stim_gen.canvas_size[0], 0])
    
#     # A. Draw Losers (Faintly)
#     for i in range(len(all_traces)):
#         lx, ly = to_pix(all_traces[i])
#         plt.plot(lx, ly, 'r-', alpha=0.1, lw=0.5)
        
#     # B. Draw Winner (Bold Cyan)
#     wx, wy = to_pix(winner_trace)
#     plt.plot(wx, wy, 'c-', lw=2.5, label='Winning Strategy')
#     plt.scatter(wx, wy, c=np.arange(len(wx)), cmap='cool', s=20, zorder=5)

#     plt.legend(loc='upper right')
#     plt.title("Batch Search Results")
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# # Assuming model, readout, world_gen, retina are already defined and on device

# # 1. Run Batch Optimization
# winner, all_traces, history = optimize_batch_trajectories(
#     model=model,
#     readout=readout,
#     stim_gen=world_gen,
#     retina=retina,
#     base_theta=theta, # (1, 3) tensor
#     batch_size=32,    # Increased search space
#     n_steps=150       # Fewer steps needed since we search parallel
# )

# # 2. Visualize
# visualize_batch_results(world_gen, theta, winner, all_traces, history)
# # %%

# plt.plot(winner.detach().cpu().numpy())
# # %%
