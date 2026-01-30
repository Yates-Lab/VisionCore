#%%
# #things to try:
# 1. try using all cells instead of just the visual responsive ones
# 2. look at the image content in fixrsvp and find the highest frequency images and see if decoding is better for those images
# 3. about if inference is being done correctly right now...
# 4. NEED TO FIX np.nan_to_num(X_batch, nan=0.0), the fix is probably having attention mask for both time and cells
# 5. fix the issue of the trials not matching
# 6. setup raytune for hyperparameter tuning
# 7. Try seeing if training on all time bins helps or hurts?
# 8. for mixing augmention, try doing based on eyepos and not just random
# 9. shuffle analysis for eyepos for baseline
# 10. try feeding in image itself to model
# 11. use dfs
# 12. see if decoder can predict the patch of image
# 13. see which parts of image are most decodeable
import os
from pathlib import Path
# Device options
use_gpu = True
gpu_index = 0
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
import matplotlib.patheffects as pe 
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1.utils.general import get_clock_functions
import torch
import torch.nn as nn
import numpy as np
import plenoptic as po
from plenoptic.simulate import SteerablePyramidFreq
from tqdm import tqdm

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False
import contextlib
import schedulefree

#%%
def get_spatial_pyramid_features(image_stack, grid_size=6, device='cuda'):
    """
    Extracts Steerable Pyramid features (Real+Imag) and pools them into a 
    fixed spatial grid (grid_size x grid_size).
    
    Args:
        image_stack: (N_images, H, W) or (N, 1, H, W)
        grid_size: Spatial resolution for attention (e.g. 6 -> 36 patches)
    Returns:
        Tensor of shape (N_images, Grid*Grid, Feature_Dim)
    """
    # 1. Format Input
    if isinstance(image_stack, np.ndarray):
        images = torch.from_numpy(image_stack).float()
    else:
        images = image_stack.float()
        
    if images.ndim == 3:
        images = images.unsqueeze(1) # Ensure (N, 1, H, W)
    
    images = images.to(device)
    
    # 2. Initialize Pyramid
    # downsample=True is efficient, we will resize outputs later anyway
    pyr = SteerablePyramidFreq(
        image_shape=images.shape[-2:],
        height='auto',
        order=3,        # 4 orientations
        is_complex=True,
        downsample=True 
    ).to(device)

    batch_size = 20 # Smaller batch size to save GPU memory
    all_grid_features = []
    
    print(f"Extracting Spatial Pyramid Features (Grid {grid_size}x{grid_size})...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i : i + batch_size]
            coeffs = pyr(batch)
            
            # We will stack channels here
            batch_channels = []
            
            # Sort keys for deterministic order
            sorted_keys = sorted(coeffs.keys(), key=lambda x: str(x))
            
            for key in sorted_keys:
                c = coeffs[key]
                
                # Handle Complex (Real/Imag) vs Real
                if torch.is_complex(c):
                    # Separate components -> (B, 2, H_scale, W_scale)
                    components = torch.cat([c.real, c.imag], dim=1) 
                else:
                    components = c 
                
                # FORCE all scales to the same spatial grid size
                # This aligns high-freq (fine) and low-freq (coarse) maps
                pooled = nn.functional.adaptive_avg_pool2d(components, (grid_size, grid_size))
                
                # Flatten spatial grid? No, we flatten spatial grid later.
                # Right now we want: (B, Channels, Grid, Grid)
                batch_channels.append(pooled)
            
            # Concatenate all feature channels: (B, Total_Channels, Grid, Grid)
            features_spatial = torch.cat(batch_channels, dim=1)
            
            # Reshape to: (B, Grid*Grid, Total_Channels)
            # This makes "Patches" the sequence for Attention
            B, C, H, W = features_spatial.shape
            features_flat_grid = features_spatial.view(B, C, H*W).permute(0, 2, 1)
            
            all_grid_features.append(features_flat_grid.cpu())

    # Stack: (N_Images, 36, Feature_Dim)
    full_tensor = torch.cat(all_grid_features, dim=0)
    print(f"Final Feature Shape: {full_tensor.shape}")
    return full_tensor

from scripts.mcfarland_sim import get_fixrsvp_stack
from DataYatesV1.exp.support import get_rsvp_fix_stim

# support_images = get_rsvp_fix_stim()
# stack_images = get_fixrsvp_stack()
#%%
subject = 'Allen'
date = '2022-03-04'

#04-08, 03-02, 04-13, 2-18 all stimuli are not timed right

#03-04, 03-30, 03-02, 04-08, 04-13 (15 epochs), 04-01, 2-18
#4-06 is okay too



dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp_all_cells.yaml'
dataset_configs = load_dataset_configs(dataset_configs_path)

# date = "2022-03-04"
# subject = "Allen"
dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
    train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)



sess = train_dset.dsets[0].metadata['sess']
# ppd = train_data.dsets[0].metadata['ppd']
cids = dataset_config['cids']
print(f"Running on {sess.name}")

# get fixrsvp inds and make one dataaset object
inds = torch.concatenate([
        train_dset.get_dataset_inds('fixrsvp'),
        val_dset.get_dataset_inds('fixrsvp')
    ], dim=0)

dataset = train_dset.shallow_copy()
dataset.inds = inds

# Getting key variables
dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

rsvp_images = torch.from_numpy(get_fixrsvp_stack(frames_per_im=1))
ppd = 37.50476617
#get the central 2.5 degrees of the image
window_size_pixels = 3 * ppd
start_x = int(rsvp_images.shape[1] // 2 - window_size_pixels // 2)
end_x = int(rsvp_images.shape[1] // 2 + window_size_pixels // 2)
start_y = int(rsvp_images.shape[2] // 2 - window_size_pixels // 2)
end_y = int(rsvp_images.shape[2] // 2 + window_size_pixels // 2)
rsvp_images_cropped = rsvp_images[:, start_x:end_x, start_y:end_y]
ptb2ephys, _ = get_clock_functions(sess.exp)
image_ids = np.full((NT, T), -1, dtype=np.int64)
# Loop over trials and align responses
robs = np.nan*np.zeros((NT, T, NC))
dfs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))

for itrial in tqdm(range(NT)):
    # print(f"Trial {itrial}/{NT}")
    trial_mask = trials[itrial] == trial_inds
    if np.sum(trial_mask) == 0:
        continue
    
    trial_id = int(trials[itrial])
    trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
    trial_image_ids = trial.image_ids
    if len(np.unique(trial_image_ids)) < 2:
        continue
    start_idx = np.where(trial_image_ids == 2)[0][0]
    flip_times = ptb2ephys(trial.flip_times[start_idx:])

    psth_inds_all = dataset.dsets[dset_idx].covariates['psth_inds'][trial_mask].numpy()
    trial_bins_all = t_bins[trial_mask]
    hist_idx_all = np.searchsorted(flip_times, trial_bins_all, side='right') - 1 + start_idx
    image_ids[itrial][psth_inds_all] = trial_image_ids[hist_idx_all] - 1

    ix = trial_mask & fixation
    if np.sum(ix) == 0:
        continue

    stim_inds = np.where(ix)[0]
    # stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()

    

# check for if image_ids is correct.
# # pick a trial
# trial_id = int(trials[0])
# trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])
# start_idx = np.where(trial.image_ids == 2)[0][0]
# flip_times = ptb2ephys(trial.flip_times[start_idx:])
# trial_bins = t_bins[trial_inds == trial_id]
# hist_idx = np.searchsorted(flip_times, trial_bins, side='right') - 1 + start_idx

# # This should be identical to the assigned row (before -1 shift)
# np.all(trial.image_ids[hist_idx] - 1 == image_ids[0][dataset.dsets[dset_idx].covariates['psth_inds'][trial_inds == trial_id]])

# time_window_start = 75
# time_window_end =100
time_window_start = 0
time_window_end =200
good_trials = fix_dur > 20
robs = robs[good_trials][:,time_window_start:time_window_end,:]
dfs = dfs[good_trials][:,time_window_start:time_window_end,:]
eyepos = eyepos[good_trials][:,time_window_start:time_window_end,:]
fix_dur = fix_dur[good_trials]
image_ids = image_ids[good_trials][:, time_window_start:time_window_end]


ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
# plt.xlim(0, 160)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
# plt.xlim(0, 160)
plt.show()

plt.plot(np.nanstd(robs, (2,0)))
plt.show()
robs.shape #(79, 335, 133) [trials, time, cells]
eyepos.shape #(79, 335, 2) [trials, time, xycoords]

salvageable_mismatch_time_threshold = 20
reference_trial_ind = None
image_ids_reference = None
for i in range(len(image_ids)):
    if (image_ids[i, time_window_start:time_window_end] != -1).all():
        image_ids_reference = image_ids[i]

        reference_trial_ind = i
        break

unmatched_trials_and_start_time_ind_of_mismatch = {}

for trial_ind, row in enumerate(image_ids):
    start_time_ind_of_mismatch = None
    for time_ind in range(len(row)):
        trial_matches = True
        if row[time_ind] != -1 and image_ids_reference[time_ind] != -1:
            if image_ids_reference[time_ind] != row[time_ind]:
                trial_matches = False
                start_time_ind_of_mismatch = time_ind
                
        if not trial_matches:
            print(f'trial {trial_ind} does not match')
            unmatched_trials_and_start_time_ind_of_mismatch[trial_ind] = start_time_ind_of_mismatch
            break

trials_to_remove = []
for trial_ind, start_time_ind_of_mismatch in unmatched_trials_and_start_time_ind_of_mismatch.items():
    first_trial_ind = reference_trial_ind
    second_trial_ind = trial_ind
    plt.plot(image_ids[first_trial_ind])
    plt.plot(image_ids[second_trial_ind])
    plt.xlim(0, 200)
    plt.xlabel('Time (bins)')
    plt.ylabel('Image ID')
    plt.title(f'Image IDs for trial {first_trial_ind} and {second_trial_ind}')
    plt.legend([f'Trial {first_trial_ind}', f'Trial {second_trial_ind}'])

    plt.show()
    print(f'start time ind of mismatch for trial {trial_ind} is {start_time_ind_of_mismatch}')
    
    if start_time_ind_of_mismatch > salvageable_mismatch_time_threshold:
        robs[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
        eyepos[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
        fix_dur[trial_ind] = start_time_ind_of_mismatch
        dfs[trial_ind, start_time_ind_of_mismatch:, :] = np.nan
        image_ids[trial_ind, start_time_ind_of_mismatch:] = -1
    else:
        trials_to_remove.append(trial_ind)

robs = robs[~np.isin(np.arange(len(robs)), trials_to_remove)]
eyepos = eyepos[~np.isin(np.arange(len(eyepos)), trials_to_remove)]
fix_dur = fix_dur[~np.isin(np.arange(len(fix_dur)), trials_to_remove)]
dfs = dfs[~np.isin(np.arange(len(dfs)), trials_to_remove)]
image_ids = image_ids[~np.isin(np.arange(len(image_ids)), trials_to_remove)]

for trial_ind, row in enumerate(image_ids):
    for time_ind in range(len(row)):

        if row[time_ind] != -1 and image_ids_reference[time_ind] != -1:
            if image_ids_reference[time_ind] != row[time_ind]:
                raise ValueError(f'trial {trial_ind} does not match at time {time_ind}')

#%%
# sess = train_dset.dsets[0].metadata['sess']
# trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
# trials = np.unique(trial_inds)

# trial_id = int(trials[20])  # trial 20 for example
# trial = FixRsvpTrial(sess.exp['D'][trial_id], sess.exp['S'])

# image_ids_trial = trial.image_ids
# flip_times_trial = trial.flip_times

# # indices where image ID changes
# change_idx = np.where(np.diff(image_ids_trial) != 0)[0] + 1
# change_times = flip_times_trial[change_idx]

# dt_change = np.median(np.diff(change_times))
# print("Stim change interval (s):", dt_change) #0.050002098083496094
# print("Stim change rate (Hz):", 1.0 / dt_change) #19.99916080181572



#%%

# # Load Rowley session
# subject = 'Luke'
# date = '2025-08-04'
# from DataRowleyV1V2.data.registry import get_session as get_rowley_session


# print(f"Loading Rowley session: {subject}_{date}")
# sess = get_rowley_session(subject, date)
# print(f"Session loaded: {sess.name}")
# print(f"Session directory: {sess.processed_path}")

# # Load fixRSVP dataset
# eye_calibration = 'left_eye_x-0.5_y-0.3'
# dataset_type = 'fixrsvp'
# dset_path = Path(sess.processed_path) / 'datasets' / eye_calibration / f'{dataset_type}.dset'

# print(f"Loading dataset from: {dset_path}")
# if not dset_path.exists():
#     raise FileNotFoundError(f"Dataset not found: {dset_path}")

# # Load using DictDataset
# from DataYatesV1 import DictDataset
# rowley_dset = DictDataset.load(dset_path)

# print(f"Dataset loaded: {len(rowley_dset)} samples")
# print(f"Response shape: {rowley_dset['robs'].shape}")

# # Extract data
# trial_inds = rowley_dset['trial_inds'].numpy()
# trials = np.unique(trial_inds)
# NC = rowley_dset['robs'].shape[1]
# NT = len(trials)

# # Determine max trial length
# max_T = 0
# for trial in trials:
#     trial_len = np.sum(trial_inds == trial)
#     max_T = max(max_T, trial_len)

# print(f"Number of trials: {NT}")
# print(f"Number of neurons: {NC}")
# print(f"Max trial length: {max_T}")

# # Create trial-aligned arrays
# robs = np.nan * np.zeros((NT, max_T, NC))
# eyepos = np.nan * np.zeros((NT, max_T, 2))
# fix_dur = np.zeros(NT)

# # Define fixation criterion (eye position < 1 degree from center)
# eyepos_raw = rowley_dset['eyepos'].numpy()
# fixation = np.hypot(eyepos_raw[:, 0], eyepos_raw[:, 1]) < 1

# print("Aligning trials...")
# for itrial in tqdm(range(NT)):
#     trial_mask = (trial_inds == trials[itrial]) & fixation
#     if np.sum(trial_mask) == 0:
#         continue
    
#     trial_data = rowley_dset['robs'][trial_mask].numpy()
#     trial_eye = rowley_dset['eyepos'][trial_mask].numpy()
    
#     trial_len = trial_data.shape[0]
#     robs[itrial, :trial_len] = trial_data
#     eyepos[itrial, :trial_len] = trial_eye
#     fix_dur[itrial] = trial_len

# r_flat = np.nan_to_num(robs, nan=0.0).reshape(NT, -1)
# e_flat = np.nan_to_num(eyepos, nan=0.0).reshape(NT, -1)
# sig = np.concatenate([r_flat, e_flat], axis=1)

# _, keep = np.unique(sig, axis=0, return_index=True)
# keep = np.sort(keep)

# robs = robs[keep]
# eyepos = eyepos[keep]
# fix_dur = fix_dur[keep]
# NT = len(keep)
# #search for duplicate trials
# for itrial in range(NT):
#     for jtrial in range(itrial+1, NT):
#         if np.allclose(robs[itrial], robs[jtrial], equal_nan=True):
#             print(f"Duplicate trial found: {itrial} and {jtrial}")
#             raise ValueError("Duplicate trial found")
#             assert np.allclose(eyepos[itrial], eyepos[jtrial], equal_nan=True)

# time_window_start = 0
# time_window_end =200

# # Filter for trials with sufficient duration
# good_trials = fix_dur > 20
# robs = robs[good_trials][:,time_window_start:time_window_end,:]
# eyepos = eyepos[good_trials][:,time_window_start:time_window_end,:]
# fix_dur = fix_dur[good_trials]

# print(f"\nFiltered to {len(fix_dur)} trials with >20 bins")
# print(f"Final robs shape: {robs.shape} (trials × time × neurons)")
# print(f"Final eyepos shape: {eyepos.shape} (trials × time × XY)")

# # Sort by fixation duration for visualization
# ind = np.argsort(fix_dur)[::-1]
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].imshow(eyepos[ind, :, 0])
# axes[0].set_title('Eye position X (sorted by trial length)')
# axes[0].set_xlabel('Time (bins)')
# axes[0].set_ylabel('Trial')
# axes[1].imshow(np.nanmean(robs, 2)[ind])
# axes[1].set_title('Population mean response')
# axes[1].set_xlabel('Time (bins)')
# plt.tight_layout()
# plt.show()
#%%
# Decoder setup
rng = np.random.default_rng(0)
train_frac = 0.8
use_time_encoding = True
time_enc_dim = 8
time_enc_scale = 1.0
ridge_alpha = 10.0

ridge_window_len_input = 10

window_len_input = 50 #70
# window_len_output = 50 #10
window_len_output = 50 #70
window_stride = 1
min_valid_fraction = 0.8
num_epochs = 75 # 75
batch_size = 64 #64
learning_rate = 1e-3
lag_bins = 0
center_per_trial = True
standardize_inputs = False
cache_data_on_gpu = False
dataloader_num_workers = 4
dataloader_pin_memory = True
sample_poisson = False
augmentation_neuron_dropout = 0.1  # Probability of dropping an entire neuron's activity for a window
augmentation_turn_off_percentage = 0.2 #0.2
augmentation_turn_on_percentage = 0.02 #0.02 #0.05
augmentation_mixup_alpha = 0.2  # Mixup interpolation alpha (0.0 to disable)
augmentation_mixup_same_time = False  # If True, only mixup windows with same start index
augment_encodings = False  # If True, augmentations affect ALL features (neural + time + image_id encodings)
transformer_dim = 64 #64 best 16
transformer_heads = 4 #4 #best 2
transformer_layers = 2
transformer_dropout = 0.1 #0.1 #best 0.3
weight_decay = 1e-4 #best 1e-2
loss_on_center = False
require_odd_window = True
use_trajectory_loss = True
use_2d_attention = False  # 2d_attention flag (axial time x neuron)
use_image_id_encoding = False
include_images = True  # If True, use cross-attention with image features
num_unique_images = int(np.max(image_ids_reference)) + 1

# Validation: cannot use 2d attention with image cross-attention
if include_images and use_2d_attention:
    raise ValueError("Cannot use use_2d_attention=True when include_images=True")

lambda_pos = 1.0
lambda_vel = 0.4 #4
lambda_accel = 0 #0.1
velocity_event_thresh = 0.02 #0.02
velocity_event_weight = 0

input_nan_fill_value = 0

# augmentation_turn_off_percentage = 0
# augmentation_turn_on_percentage = 0
# transformer_dim = 16 #64 best 16
# transformer_heads = 2 #4 #best 2
# transformer_layers = 2
# transformer_dropout = 0.3 #0.1 #best 0.3
# weight_decay = 1e-2 #best 1e-2
# loss_on_center = False
# require_odd_window = True
# use_trajectory_loss = True
# lambda_pos = 1.0
# lambda_vel = 0 #4
# lambda_accel = 0 #0.1
# velocity_event_thresh = 0.02 #0.02
# velocity_event_weight = 0

num_trials = robs.shape[0]
trial_indices = np.arange(num_trials)
rng.shuffle(trial_indices)
num_train = int(train_frac * num_trials)
train_trials = trial_indices[:num_train]
val_trials = trial_indices[num_train:]

rsvp_image_features = get_spatial_pyramid_features(rsvp_images_cropped)

# Prepare feature tensor with background for cross-attention model
if include_images:
    # rsvp_image_features shape: (num_images, grid_len, feat_dim)
    num_imgs, grid_len, feat_dim = rsvp_image_features.shape
    
    # Add background/Gray Screen Vector
    # If your dataset uses -1 for "no image", we map it to the last index
    background_vec = torch.zeros((1, grid_len, feat_dim))  # Zero energy for gray screen
    final_feature_table = torch.cat([rsvp_image_features, background_vec], dim=0)
    
    # The index for background is now 'num_imgs' (if you had 0-64 images, background is 65)
    BG_INDEX = num_imgs
else:
    final_feature_table = None
    BG_INDEX = None


def build_time_encoding(num_timepoints, dim):
    if dim % 2 != 0:
        raise ValueError("time_enc_dim must be even.")
    positions = np.arange(num_timepoints)[:, None]
    div_term = np.exp(
        np.arange(0, dim, 2) * (-np.log(10000.0) / dim)
    )
    encoding = np.zeros((num_timepoints, dim))
    encoding[:, 0::2] = np.sin(positions * div_term)
    encoding[:, 1::2] = np.cos(positions * div_term)
    return encoding


def apply_lag(robs_in, eyepos_in, lag):
    if lag <= 0:
        return robs_in, eyepos_in
    if lag >= robs_in.shape[1]:
        raise ValueError("lag_bins must be smaller than time dimension.")
    robs_out = robs_in[:, :-lag, :]
    eyepos_out = eyepos_in[:, lag:, :]
    return robs_out, eyepos_out


robs_aligned, eyepos_aligned = apply_lag(robs, eyepos, lag_bins)
time_len = robs_aligned.shape[1]
if window_len_input is None:
    window_len_input = time_len
if window_len_output is None:
    window_len_output = window_len_input
if window_len_output > window_len_input:
    raise ValueError("window_len_output must be <= window_len_input.")
output_offset = (window_len_input - window_len_output) // 2
if output_offset < 0:
    raise ValueError("window_len_input must be >= window_len_output.")
if use_trajectory_loss and loss_on_center:
    raise ValueError("loss_on_center must be False when use_trajectory_loss=True.")
if loss_on_center and require_odd_window and window_len_output % 2 == 0:
    raise ValueError("window_len must be odd when loss_on_center=True.")

if center_per_trial:
    global_mean = np.nanmean(eyepos_aligned[train_trials], axis=(0, 1))
    eyepos_mean = np.tile(global_mean[None, :], (num_trials, 1))
    eyepos_centered = eyepos_aligned - eyepos_mean[:, None, :]
else:
    eyepos_mean = np.zeros((num_trials, 2))
    eyepos_centered = eyepos_aligned

y_target = eyepos_centered

time_encoding = None
if use_time_encoding and time_enc_dim > 0:
    time_encoding = build_time_encoding(time_len, time_enc_dim)
    time_encoding = np.tile(time_encoding[None, :, :], (num_trials, 1, 1))

if time_encoding is not None:
    robs_feat = np.concatenate(
        [robs_aligned, time_enc_scale * time_encoding], axis=2
    )
else:
    robs_feat = robs_aligned


def build_ridge_samples(X, Y, trials, window_len):
    num_trials, time_len, num_feats = X.shape
    half = window_len // 2
    X_list = []
    Y_list = []
    for trial in trials:
        for t in range(time_len):
            start = t - half
            end = start + window_len
            if start < 0 or end > time_len:
                continue
            X_win = X[trial, start:end, :]
            y_t = Y[trial, t, :]
            if np.isnan(X_win).any() or np.isnan(y_t).any():
                continue
            X_list.append(X_win.reshape(-1))
            Y_list.append(y_t)
    if len(X_list) == 0:
        return np.empty((0, window_len * num_feats)), np.empty((0, 2))
    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


def predict_ridge_trial_global(
    robs_in,
    ridge_w,
    ridge_mean,
    ridge_std,
    window_len,
    trial_idx,
):
    time_len = robs_in.shape[1]
    pred = np.full((time_len, 2), np.nan)
    half = window_len // 2
    X_trial = robs_in[trial_idx]
    for t in range(time_len):
        start = t - half
        end = start + window_len
        if start < 0 or end > time_len:
            continue
        X_win = X_trial[start:end]
        if np.isnan(X_win).any():
            continue
        x_flat = X_win.reshape(-1)
        x_std = (x_flat - ridge_mean) / ridge_std
        pred[t] = x_std @ ridge_w
    return pred


def standardize_train(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def standardize_robs(robs_in, train_trials):
    robs_train = robs_in[train_trials]
    mean = np.nanmean(robs_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(robs_train, axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    return (robs_in - mean) / std, mean, std


def fit_ridge(X, Y, alpha):
    XtX = X.T @ X
    XtY = X.T @ Y
    reg = alpha * np.eye(X.shape[1])
    return np.linalg.solve(XtX + reg, XtY)


def r2_score(y_true, y_pred):
    y_true_mean = y_true.mean(axis=0, keepdims=True)
    ss_tot = np.sum((y_true - y_true_mean) ** 2, axis=0)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    return 1.0 - (ss_res / ss_tot)


# Ridge baseline (global model with context window)
X_train, y_train = build_ridge_samples(
    robs_feat, y_target, train_trials, ridge_window_len_input
)
X_val, y_val = build_ridge_samples(
    robs_feat, y_target, val_trials, ridge_window_len_input
)
X_train, X_val, X_mean, X_std = standardize_train(X_train, X_val)

ridge_w = fit_ridge(X_train, y_train, ridge_alpha)
y_val_pred = X_val @ ridge_w
ridge_mse = np.mean((y_val - y_val_pred) ** 2, axis=0)
ridge_r2 = r2_score(y_val, y_val_pred)

print("Ridge baseline:")
print(f"  MSE (x, y): {ridge_mse}")
print(f"  R2  (x, y): {ridge_r2}")


if standardize_inputs:
    robs_z, robs_mean, robs_std = standardize_robs(
        robs_aligned, train_trials
    )
else:
    robs_z = robs_aligned

if use_2d_attention:
    # Use raw neuron features; time encoding handled via embeddings in the model.
    robs_feat_model = robs_z
else:
    if time_encoding is not None:
        robs_feat_model = np.concatenate(
            [robs_z, time_enc_scale * time_encoding], axis=2
        )
    else:
        robs_feat_model = robs_z

# Only add image ID one-hot encoding if use_image_id_encoding=True AND include_images=False
# When include_images=True, image information comes via cross-attention, not one-hot encoding
if use_image_id_encoding and not include_images:
    # ids is (Time,)
    # Slice to match the time window used for robs/eyepos
    ids = image_ids_reference[time_window_start:time_window_end].astype(int)
    
    # We only create one-hot for valid IDs (1-20). 
    # -1 will result in a row of all ZEROS (a "null" encoding).
    valid_mask = ids >= 0
    one_hot = np.zeros((len(ids), num_unique_images))
    # Fill one-hot only for valid indices
    one_hot[valid_mask, ids[valid_mask]] = 1.0
    
    # Tile for all trials: (Trials, Time, Num_Images)
    image_feat = np.tile(one_hot[None, :, :], (num_trials, 1, 1))
    
    # Concatenate to the model features
    robs_feat_model = np.concatenate([robs_feat_model, image_feat], axis=2)

# Set num_neural_features based on augment_encodings flag
if augment_encodings:
    num_neural_features = robs_feat_model.shape[2]  # All features including encodings
else:
    num_neural_features = robs_z.shape[2]           # Only neural features


class WindowedEyeposDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        robs,           # Pass pure Neural data (Trials, Time, Neurons)
        dfs,            # Pass pure Time Basis (Time, Basis_Funcs)
        image_ids,      # Pass pure Image IDs (Trials, Time)
        eyepos,
        indices,
        window_len_input,
        bg_index=None,       # NEW: The index to use for -1 (gray screen)
        transform=None,
    ):
        self.robs = robs
        self.dfs = dfs
        self.image_ids = image_ids
        self.eyepos = eyepos
        self.indices = indices
        self.window_len_input = window_len_input
        self.bg_index = bg_index 
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial, start = self.indices[idx]
        end = start + self.window_len_input

        # 1. Neural + Time Input (The Query)
        spikes = self.robs[trial, start:end]
        time_enc = self.dfs[start:end] 
        # Note: Concatenate pure spikes and time encoding.
        # DO NOT include old one-hot image vectors.
        x_neurons = np.concatenate([spikes, time_enc], axis=1)

        # 2. Image ID Input (The Key/Value)
        img_seq = self.image_ids[trial, start:end].copy()
        
        # Handle -1 (no image) by mapping to Background Index
        if self.bg_index is not None:
             img_seq[img_seq == -1] = self.bg_index
        
        x_ids = img_seq.astype(np.int64)

        # 3. Targets
        y = self.eyepos[trial, start:end]

        if self.transform:
            x_neurons = self.transform(x_neurons)

        # Return THREE items
        return (
            torch.tensor(x_neurons, dtype=torch.float32),
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

# Create dataset indices
train_indices = []
for trial in train_trials:
    for start in range(0, robs_feat.shape[1] - window_len_input + 1, window_stride):
        # We need check if valid (no nans in eyepos)
        # Using y_target for nan check
        if not np.isnan(y_target[trial, start:start+window_len_input]).any():
             train_indices.append((trial, start))

val_indices = []
for trial in val_trials:
    for start in range(0, robs_feat.shape[1] - window_len_input + 1, window_stride):
         if not np.isnan(y_target[trial, start:start+window_len_input]).any():
             val_indices.append((trial, start))

# NOTE: robs_feat_model contains time encoding already if time_enc was used. 
# But in the new Dataset class we are manually concatenating robs + time_enc inside __getitem__.
# So we should pass robs_z (raw spikes) and time_encoding (raw basis) separately.
# Or just pass the pre-concatenated version if simpler, but let's stick to the clean separation plan.

# Extract raw time encoding matrix (Trials, Time, Dim) -> (Time, Dim) since it's tiled
if time_encoding is not None:
    time_basis_funcs = time_encoding[0] 
else:
    # If no time encoding, pass empty or zero-dim array, or handle inside dataset.
    # For simplicity, if no time encoding, create dummy zeros (Time, 0)
    time_basis_funcs = np.zeros((robs_z.shape[1], 0))

# We also need to map the "robs_z" (standardized) back to the dataset
# robs_z is (Trials, Time, Neurons)

dataset_train = WindowedEyeposDataset(
    robs=robs_z,
    dfs=time_basis_funcs * time_enc_scale, # Scale applied here
    image_ids=image_ids,
    eyepos=y_target,
    indices=train_indices,
    window_len_input=window_len_input,
    bg_index=BG_INDEX
)

dataset_val = WindowedEyeposDataset(
    robs=robs_z,
    dfs=time_basis_funcs * time_enc_scale,
    image_ids=image_ids,
    eyepos=y_target,
    indices=val_indices,
    window_len_input=window_len_input,
    bg_index=BG_INDEX
)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=dataloader_num_workers,
    pin_memory=dataloader_pin_memory,
)

val_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=dataloader_num_workers,
    pin_memory=dataloader_pin_memory,
)


class TransformerEyeposCrossAttn(nn.Module):
    def __init__(self, num_neurons, feature_tensor, model_dim=128, num_heads=4, num_layers=3):
        """
        feature_tensor: Pre-computed tensor (Num_Images+1, Grid_Len, Feat_Dim)
        """
        super().__init__()
        
        # 1. Image Memory (Embedding)
        # We use a trick: Embedding usually returns 1D vectors. 
        # We will flatten the grid (Grid*Feat) into the embedding dim, then reshape in forward.
        self.num_imgs, self.grid_len, self.feat_dim = feature_tensor.shape
        
        # Initialize embedding with your pre-computed features
        # Flatten last two dims for storage: (N, Grid_Len * Feat_Dim)
        flat_tensor = feature_tensor.reshape(self.num_imgs, -1)
        self.image_embed = nn.Embedding.from_pretrained(flat_tensor, freeze=True)
        
        # 2. Projections
        self.neuron_proj = nn.Linear(num_neurons, model_dim)
        self.img_proj = nn.Linear(self.feat_dim, model_dim) # Project pyramid feats to model dim
        
        # 3. Cross Attention (The Core Logic)
        # Query = Neurons, Key = Image Patches, Value = Image Patches
        self.cross_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        
        # 4. Temporal Transformer (Process time dynamics)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Head
        self.head = nn.Linear(model_dim, 2) # X, Y

    def forward(self, x_neurons, x_img_ids):
        """
        x_neurons: (Batch, Time, Neurons)
        x_img_ids: (Batch, Time) -> Integers
        """
        B, T, _ = x_neurons.shape
        
        # --- A. Prepare Queries (Neurons) ---
        q_neurons = self.neuron_proj(x_neurons) # (B, T, Model_Dim)
        
        # --- B. Prepare Keys/Values (Images) ---
        # 1. Lookup Features
        # View as flat list of indices first: (B*T)
        flat_ids = x_img_ids.view(-1)
        
        # Retrieve: (B*T, Grid_Len * Feat_Dim)
        img_raw = self.image_embed(flat_ids)
        
        # Reshape back to grid: (B*T, Grid_Len, Feat_Dim)
        kv_grid = img_raw.view(B*T, self.grid_len, self.feat_dim)
        
        # Project to Model Dimension: (B*T, Grid_Len, Model_Dim)
        kv_proj = self.img_proj(kv_grid)
        
        # --- C. Cross Attention ---
        # We treat every timepoint as an independent "search" first.
        # Query Shape must be: (B*T, 1, Model_Dim)
        q_flat = q_neurons.view(B*T, 1, -1)
        
        # Attend: "For these spikes, which of the 36 patches are relevant?"
        # attn_out: (B*T, 1, Model_Dim)
        # weights:  (B*T, 1, Grid_Len) -> Keep this if you want to visualize!
        attn_out, attn_weights = self.cross_attn(query=q_flat, key=kv_proj, value=kv_proj)
        
        # Reshape context back to (Batch, Time, Model_Dim)
        context = attn_out.view(B, T, -1)
        
        # --- D. Temporal Processing ---
        # Combine Neural info + Visual Context (Residual connection)
        combined = q_neurons + context 
        
        # Now run standard temporal transformer
        out = self.temporal_encoder(combined)
        
        return self.head(out)


# Initialize Model
device = torch.device(f"cuda:{gpu_index}" if use_gpu and torch.cuda.is_available() else "cpu")

if include_images:
    # input_dim = neurons + time_funcs
    input_dim = robs_z.shape[2] + time_basis_funcs.shape[1]
    
    model = TransformerEyeposCrossAttn(
        num_neurons=input_dim,
        feature_tensor=final_feature_table,
        model_dim=transformer_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers
    ).to(device)
else:
    # Fallback to old transformer logic if needed (omitted here since you requested the update)
    pass 

criterion = nn.MSELoss()
optimizer = schedulefree.AdamWScheduleFree(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.train()
    for batch_idx, (x_neurons, x_ids, y_gt) in enumerate(train_loader):
        x_neurons = x_neurons.to(device)
        x_ids = x_ids.to(device)
        y_gt = y_gt.to(device)
        
        optimizer.zero_grad()
        
        # NEW Forward Pass
        y_pred = model(x_neurons, x_ids)
        
        loss = criterion(y_pred, y_gt)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    optimizer.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_neurons, x_ids, y_gt in val_loader:
            x_neurons = x_neurons.to(device)
            x_ids = x_ids.to(device)
            y_gt = y_gt.to(device)
            
            y_pred = model(x_neurons, x_ids)
            loss = criterion(y_pred, y_gt)
            val_loss += loss.item()
            
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")


# -------------------------------------------------------------------------
# Visualization / Prediction Plotting
# -------------------------------------------------------------------------

def predict_trial(model, dataset, trial_idx):
    # Manually retrieve the items for one trial-segment or loop over the whole trial
    # Simplified: grabbing one sample from the validation set
    model.eval()
    with torch.no_grad():
        x_neurons, x_ids, y_gt = dataset[trial_idx]
        x_neurons = x_neurons.unsqueeze(0).to(device)
        x_ids = x_ids.unsqueeze(0).to(device)
        
        y_pred = model(x_neurons, x_ids)
        
        return y_pred.cpu().numpy()[0], y_gt.numpy()
#%%
# Example Plot
# pick a random index from val dataset
idx = 900
pred, gt = predict_trial(model, dataset_val, idx)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(gt[:, 0], label='GT X')
plt.plot(pred[:, 0], label='Pred X')
plt.legend()
plt.title("X Position")

plt.subplot(1, 2, 2)
plt.plot(gt[:, 1], label='GT Y')
plt.plot(pred[:, 1], label='Pred Y')
plt.legend()
plt.title("Y Position")
plt.show()
#%%