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

from DataYatesV1 import enable_autoreload, get_free_device
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye
from spatial_info import make_stimulus_stack, make_counterfactual_stim
from spatial_info import get_spatial_readout
from spatial_info import compute_rate_map, compute_rate_map_batched
from spatial_info import spatial_ssi_population, make_movie

enable_autoreload()
device = get_free_device()

#%% Load model and dataset
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
models_by_type = scan_checkpoints(checkpoint_dir, verbose=False)

model_type = 'resnet_none_convgru'
model, model_info = load_model(
    model_type=model_type,
    model_index=0,
    checkpoint_path=None,
    checkpoint_dir=checkpoint_dir,
    device='cpu'
)
model.model.eval()
model.model.convnet.use_checkpointing = True  # Enable checkpointing to save GPU memory
model = model.to(device)



import dill
with open('mcfarland_outputs.pkl', 'rb') as f:
    outputs = dill.load(f)

readout = get_spatial_readout(model, outputs).to(device)

sessions = [outputs[i]['sess'] for i in range(len(outputs))]
#%% Helper functions


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

    y = compute_rate_map_batched(model, readout, eye_stim)
    y_null = compute_rate_map_batched(model, readout, eye_stim_null)

    return y, y_null, eye_stim, eye_stim_null

#%%

eyetraces = []
max_T = 540

for name in sessions:
    dataset_idx = model.names.index(name)
    
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

#%%
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

#%%
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

#%%
trial_list = np.argsort(fix_dur)[::-1]
itrial = trial_list[0]

plt.figure()
plt.plot(eyepos[itrial])
plt.show()


for iframe in range(full_stack.shape[0]):
    print(f"Frame {iframe}")
    y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack[[iframe]].repeat(fix_dur[itrial]+n_lags+1, axis=0), out_size=out_size, n_lags=n_lags, scale=scale)
    ispike, irate, I_t = spatial_ssi_population(y)
    ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

    fig = plot_spatial_info_figure(
        full_stack, iframe, f, Pr,
        eyepos, itrial,
        I_t_null, I_t,
        outpath=f"../figures/spatial_info/spatial_info_{iframe}.png"
    )
    fig.show()
    break



#%%


#%%


#%%
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

#%%
# i_spikes = np.stack(i_spikes)
# i_rates = np.stack(i_rates)
# i_spikes_null = np.stack(i_spikes_null)
# i_rates_null = np.stack(i_rates_null)

#%%
plt.figure()
plt.subplot(1,2,1)
plt.plot(i_spikes_null, i_spikes, '.', alpha=0.1)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Bits/Spike (Null stim)')
plt.ylabel('Bits/Spike (Real stim)')
plt.title('Spatial Info (Spikes)')

plt.subplot(1,2,2)
plt.plot(i_rates_null, i_rates, '.', alpha=0.1)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Bits/Sec (Null stim)')
plt.ylabel('Bits/Sec (Real stim)')
plt.title('Spatial Info (Time)')


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

print(f"\nSpatial Information (Real stim):   {ispike:.3f} bits/spike, {irate:.3f} bits/sec")
print(f"Spatial Information (Null stim): {ispike_null:.3f} bits/spike, {irate_null:.3f} bits/sec")

plt.figure()
_ = plt.plot(I_t.mean(0), I_t_null.mean(0), '.', alpha=0.1)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Spatial Info (Real FEMs)')
plt.ylabel('Spatial Info (No FEMs)')
plt.title('Spatial Info (Units)')
plt.show()

plt.plot(np.cumsum(I_t.mean(1)))
plt.plot(np.cumsum(I_t_null.mean(1)))
plt.ylabel('Cumulative Spatial Info (bits)')
plt.xlabel('Time (frames)')
plt.legend(['Real stim', 'Null stim'])
plt.title(f'Cumulative Spatial Info (population) {frame}')
# %%
frame = 7
type = 'nat'

full_stack = make_stimulus_stack(type=type,
        frame=frame, frames_per_im=1)

f, Pr = radial_power_spectra_np(full_stack, ppd=ppd, window=True)       # (B,), (N,B)

plt.subplot(1,2,1)
plt.imshow(full_stack[0])
plt.subplot(1,2,2)
plt.plot(f, Pr[0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Spatial Frequency (c/deg)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.show()

#%%


y, y_null, eye_stim, eye_stim_null = get_trial_stim_and_rates(eyepos[itrial], full_stack, out_size, n_lags, scale)
ispike, irate, I_t = spatial_ssi_population(y)
ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null)

units_to_show = np.argsort(I_t.mean(0)-I_t_null.mean(0)).numpy()[::-1][:25]

fig = plot_spatial_info_figure(
        full_stack, iframe, f, Pr,
        eyepos, itrial,
        I_t_null, I_t,
        outpath=f"../figures/spatial_info/nat_spatial_info_{iframe}.png"
    )
fig.show()

#%%

make_movie(y, save_path='counterfactual1', n_units_to_show=units_to_show)

#%%
# if frame is None:
    make_movie(y, save_path='counterfactual1', n_units_to_show=units_to_show)
#     make_movie(y_null, save_path='counterfactualnull', n_units_to_show=units_to_show)
# else:
#     make_movie(y, save_path=f'counterfactual1_frame{frame}', n_units_to_show=units_to_show)
#     make_movie(y_null, save_path=f'counterfactualnull_frame{frame}', n_units_to_show=units_to_show)

#%%
# #%% 
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
