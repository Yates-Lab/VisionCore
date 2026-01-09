"""
Compute spatial information from the model using reconstructed stimuli.
Allows counterfactual analysis with real vs fake eye traces.
"""
#%% Imports
import sys
sys.path.append('..')
import numpy as np
import torch
import matplotlib.pyplot as plt

from DataYatesV1 import enable_autoreload, get_free_device
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye

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

dataset_idx = 10
print(f"Loading dataset {dataset_idx}: {model.names[dataset_idx]}")
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

#%% Get fixrsvp trial indices
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

#%% Generate stimulus stack
ppd = 37.50476617
frames_per_im = 6
full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im)
print(f"Full stimulus stack shape: {full_stack.shape}")

#%% Helper functions
def embed_time_lags(movie, n_lags=32):
    """
    Embed time lags into a movie tensor.
    
    Input: movie (T, H, W) or (T, 1, H, W)
    Output: (T - n_lags + 1, 1, n_lags, H, W)
    """
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)  # (T, 1, H, W)
    
    T, C, H, W = movie.shape
    # Create lagged indices: for each output frame t, we want frames [t, t+1, ..., t+n_lags-1]
    # But stim uses negative lags (past frames), so we want [t-n_lags+1, ..., t]
    out_frames = T - n_lags + 1
    
    # Build lagged tensor
    lagged = torch.zeros(out_frames, C, n_lags, H, W, dtype=movie.dtype, device=movie.device)
    for lag in range(n_lags):
        # lag 0 = current frame, lag 1 = 1 frame ago, etc.
        lagged[:, :, lag] = movie[n_lags - 1 - lag : T - lag]
    
    return lagged

def spatial_ssi_population(y, dt=1.0, eps=1e-8, log_base=2.0):
    """
    Spatial single-spike information from a rate map.
    y: rates, shape [T, N, H, W]. Must be >= 0.
    Returns: ispikepop (bits/spike), iratepop (bits/sec), I_tn (T, N)
    """
    T, N, H, W = y.shape
    # T = time, N = units, H = height, W = width
    P = H * W # number of spatial bins
    r = y.reshape(T, N, P)
    rbar = r.mean(dim=2) # mean across space
    g = r / (rbar[..., None] + eps) # r/rbar
    logg = torch.log2(g + eps) if log_base == 2.0 else torch.log(g + eps) # log(r/rbar)
    
    I_tn = (g * logg).mean(dim=2) # expectation over space 
    
    # rescale to get bits per spike and per bin
    spikes_tn = rbar * dt
    total_bits = torch.sum(spikes_tn * I_tn)
    total_spikes = torch.sum(spikes_tn)
    ispikepop = total_bits / (total_spikes + eps) # bits/spike
    iratepop = total_bits / (T * dt) # bits/sec
    return ispikepop, iratepop, I_tn

def run_core(model, stimulus, behavior):
    """Run model core to get spatial feature maps."""
    x = model.core_forward(stimulus, behavior)
    return x

def run_model(model, stim, batch_size=32):
    """Run model with stim/behavior on CPU, move to GPU only during forward pass.

    Processes data in temporal batches to reduce peak GPU memory usage.
    """
    T = stim.shape[0]
    y_chunks = []

    with torch.no_grad():
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)

            # Move batch to GPU
            stim_gpu = stim[t_start:t_end].to(device)

            x = run_core(model.model, stim_gpu, None)
            del stim_gpu

            units = []
            for readout in model.model.readouts:
                readout.eval()
                units.append(readout.features(x[:, :, -1]))

            del x
            y_batch = model.model.activation(torch.cat(units, dim=1))
            del units

            # Move to CPU immediately
            y_chunks.append(y_batch.cpu())
            del y_batch
            torch.cuda.empty_cache()

    return torch.cat(y_chunks, dim=0)

def make_movie(y, save_path='', n_units_to_show=100):
    from torchvision.utils import make_grid
    from matplotlib.animation import FFMpegWriter

    # if n_units_to_show is list or array, use it as index
    if isinstance(n_units_to_show, (list, np.ndarray)):
        units_to_show = np.array(n_units_to_show)
        n_units_to_show = len(units_to_show)
    else:
        units_to_show = np.arange(n_units_to_show)
    
    y_subset = y[:, units_to_show].detach().cpu()  # (T, N, H, W)
    
    # normalize each unit to [0,1]
    miny = torch.tensor(np.array([y_subset[:,i].min() for i in range(n_units_to_show)]))
    maxy = torch.tensor(np.array([y_subset[:,i].max() for i in range(n_units_to_show)]))
    std = y_subset.std(dim=(0, 2, 3), keepdim=True)
    mu = y_subset.mean(dim=(0, 2, 3), keepdim=True)
    y_subset = (y_subset - mu) / (std + 1e-8)
    # y_subset = (y_subset - miny[None,:,None,None]) / (maxy[None,:,None,None] - miny[None,:,None,None] + 1e-8)

    # Global max for fixed color range
    vmax = None #y_subset.max().item()

    T = y_subset.shape[0]
    nrow = int(np.ceil(np.sqrt(n_units_to_show)))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')

    writer = FFMpegWriter(fps=15, codec='libx264', bitrate=8000)

    save_path = f'../figures/fixrsvp_spatial_activations_{save_path}.mp4'
    with writer.saving(fig, save_path, dpi=100):
        for t in range(T):
            ax.clear()
            # make_grid expects (N, C, H, W), add channel dim
            frames = y_subset[t].unsqueeze(1)  # (N, 1, H, W)
            grid = make_grid(frames, nrow=nrow, normalize=False, padding=1, pad_value=0.0)
            ax.imshow(grid[0].numpy(), cmap='gray', vmin=-6, vmax=6)
            ax.set_title(f'Spatial Activations - Frame {t}/{T}', fontsize=14)
            ax.axis('off')
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved spatial activations movie to {save_path}")

# Reconstruct stimulus
def make_counterfactual_stim(eyepos, type='fixrsvp', frame = None, scale_factor = 1.0, n_lags = 32, out_size = (101, 101)):
    ppd = 37.50476617
    frames_per_im = 6
    
    if type == 'fixrsvp':
        frames_per_im = 6
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='im')
    elif type == 'face':
        frames_per_im = 6
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='face')
    elif type == 'nat':
        frames_per_im = 1
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='nat')

    print(f"Full stimulus stack shape: {full_stack.shape}")
    if frame is not None:
        full_stack = full_stack[[frame]].repeat(eyepos.shape[0]+n_lags*2, axis=0)
    print(f"Full stimulus stack shape: {full_stack.shape}")
    print(f"fullstack data type: {full_stack.dtype}")

    eye_norm = eye_deg_to_norm(torch.fliplr(eyepos), ppd, full_stack.shape[1:3])

    eye_movie = shift_movie_with_eye(
        torch.from_numpy(full_stack[:eyepos.shape[0] + n_lags]).float(),
        torch.cat([eye_norm[:n_lags], eye_norm], dim=0),  # pad beginning
        out_size=out_size,
        center=(0.0, 0.0),
        scale_factor=scale_factor,
        mode="bilinear"
    )

    # Embed time lags to match stim shape
    eye_stim = embed_time_lags(eye_movie, n_lags=n_lags)

    return eye_stim

torch.cuda.empty_cache()

#%% flashed stimulus

n_lags = 64
out_size = (151, 151)
dt = 1/120

fix_dur = np.zeros(NT)
for itrial in range(NT):
    ix = (trials[itrial] == trial_inds) & fixation
    fix_dur[itrial] = np.sum(ix)

trial_list = np.argsort(fix_dur)[::-1]
#%%
itrial = trial_list[0]
ix = (trials[itrial] == trial_inds) & fixation
stim_inds_orig = np.where(ix)[0]

# frame += 6
frame = 4
type = 'nat'
scale = 1.0

eyepos = dataset.dsets[dset_idx]['eyepos'][ix]
null_eyepos = torch.zeros_like(eyepos) + eyepos.mean(0)
eye_stim = make_counterfactual_stim(eyepos, type=type,
    frame=frame, out_size=out_size, n_lags=n_lags, scale_factor=scale)
eye_stim_null = make_counterfactual_stim(null_eyepos, type=type, frame=frame, out_size=out_size, n_lags=n_lags, scale_factor=scale)
print(f"Reconstructed stim shape: {eye_stim.shape}")

v = out_size[0]/ppd
plt.imshow(eye_stim[0,0,0].numpy(), cmap='gray', extent=[-v/2, v/2, -v/2, v/2])
plt.plot(eyepos[:,0].numpy(), eyepos[:,1].numpy(), 'r')
plt.show()

#%%

y = run_model(model, eye_stim)
y_null = run_model(model, eye_stim_null)

# compute spatial info
ispike, irate, I_t = spatial_ssi_population(y, dt=dt)
ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null, dt=dt)

plt.plot(eyepos.numpy())
plt.show()

for cc in [0, 5, 10, 20, 100, 500]:
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y[:,cc,0,::5])
    plt.subplot(2,1,2)
    plt.plot(I_t[:,cc])
    plt.plot(I_t_null[:,cc])


#%%

print(f"\nSpatial Information (Real stim):   {ispike:.3f} bits/spike, {irate:.3f} bits/sec")
print(f"Spatial Information (Null stim): {ispike_null:.3f} bits/spike, {irate_null:.3f} bits/sec")

plt.figure()
_ = plt.plot(I_t.mean(0), I_t_null.mean(0), '.', alpha=0.1)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Spatial Info (Real FEMs)')
plt.ylabel('Spatial Info (No stim)')
plt.title('Spatial Info (Units)')
plt.show()

plt.plot(np.cumsum(I_t.mean(1)))
plt.plot(np.cumsum(I_t_null.mean(1)))
plt.ylabel('Cumulative Spatial Info (bits)')
plt.xlabel('Time (frames)')
plt.legend(['Real stim', 'Null stim'])
plt.title(f'Cumulative Spatial Info (population) {frame}')
# %%
units_to_show = np.argsort(I_t.mean(0)-I_t_null.mean(0)).numpy()[::-1][0:1000:50]

if frame is None:
    make_movie(y, save_path='counterfactual1', n_units_to_show=units_to_show)
    make_movie(y_null, save_path='counterfactualnull', n_units_to_show=units_to_show)
else:
    make_movie(y, save_path=f'counterfactual1_frame{frame}', n_units_to_show=units_to_show)
    make_movie(y_null, save_path=f'counterfactualnull_frame{frame}', n_units_to_show=units_to_show)

#%% 
unit = -1
# %%
unit +=1

ispike, irate, I_t = spatial_ssi_population(y[:,[unit]], dt=dt)
ispike_null, irate_null, I_t_null = spatial_ssi_population(y_null[:,[unit]], dt=dt)
plt.figure()
plt.plot(I_t)
plt.plot(I_t_null)
plt.show()

H = y.shape[2]
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
vmin = y[:,unit,H//2,:].amin()
vmax = y[:,unit,H//2,:].amax()
plt.imshow(y[:,unit,H//2,:].detach().cpu(), vmin=vmin, vmax=vmax)
plt.title('Real stim')

plt.subplot(1,2,2)
plt.imshow(y_null[:,unit,H//2,:].detach().cpu(), vmin=vmin, vmax=vmax)
plt.title('Null stim')
plt.colorbar()



# %%
