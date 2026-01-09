"""
Minimal script to verify that reconstructed stimulus matches the dataset stimulus.
This validates the counterfactual stimulus generation pipeline.
"""
#%% Imports
import sys
sys.path.append('..')
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from DataYatesV1 import enable_autoreload, get_free_device
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye

enable_autoreload()

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

# Fixation mask
fixation = np.hypot(
    dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), 
    dataset.dsets[dset_idx]['eyepos'][:,1].numpy()
) < 1

#%% Generate stimulus stack (all 60 images, each repeated for frames_per_im)
dt = 1/120
frate = 30  # stimulus frames per second
ppd = 37.50476617
frames_per_im = 6  # frames per image at 120Hz viewing 30Hz stim

full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im)
print(f"Full stimulus stack shape: {full_stack.shape}")

#%% Select a trial and extract data
itrial = 0  # change this to test different trials
ix = (trials[itrial] == trial_inds) & fixation
stim_inds = np.where(ix)[0]
stim_inds = stim_inds[:, None] - np.array(dataset_config['keys_lags']['stim'])[None, :]
stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0, 2, 1, 3, 4)
eyepos = dataset.dsets[dset_idx]['eyepos'][ix]

print(f"Trial {itrial}: {stim.shape[0]} frames")

#%% Reconstruct stimulus using eye position
eye_norm = eye_deg_to_norm(torch.fliplr(eyepos), ppd, full_stack.shape[1:3])
eye_movie = shift_movie_with_eye(
    torch.from_numpy(full_stack[:stim.shape[0]]).float(),
    eye_norm,
    out_size=(101, 101),
    center=(0.0, 0.0),
    scale_factor=1.0,
    mode="bilinear"
)

#%% Save side-by-side comparison movie
def save_sidebyside_movie(movie1, movie2, save_path, fps=30,
                          title1='Dataset Stim', title2='Reconstructed',
                          offset1=0, offset2=0):
    """Save a side-by-side comparison video of two movies."""
    
    def to_2d_movie(m):
        if hasattr(m, 'detach'):
            m = m.detach().cpu().numpy()
        while m.ndim > 3:
            m = m[:, 0]
        return m

    m1 = to_2d_movie(movie1)[offset1:]
    m2 = to_2d_movie(movie2)[offset2:]
    T = min(len(m1), len(m2))
    m1, m2 = m1[:T], m2[:T]

    vmin1, vmax1 = np.percentile(m1, [1, 99])
    vmin2, vmax2 = np.percentile(m2, [1, 99])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=2)

    writer = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'], bitrate=8000)

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

#%% Save the comparison
save_sidebyside_movie(
    stim[:, 0, 1],  # extract first channel/lag -> (T, H, W)
    eye_movie,
    save_path='../figures/fixrsvp_stim_comparison.mp4',
    fps=10,
    title1='Dataset Stim',
    title2='Reconstructed',
    offset1=0,
    offset2=6  # offset to align the movies
)


#%%