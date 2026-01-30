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
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
import matplotlib.patheffects as pe 
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1.utils.general import get_clock_functions
import plenoptic as po
from plenoptic.simulate import SteerablePyramidFreq

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
window_size_pixels = 2 * ppd
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
num_epochs = 100 # 75
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
augment_encodings = True  # If True, augmentations affect ALL features (neural + time + image_id encodings)
transformer_dim = 64 #64 best 16
transformer_heads = 4 #4 #best 2
transformer_layers = 2
transformer_dropout = 0.1 #0.1 #best 0.3
weight_decay = 1e-4 #best 1e-2
loss_on_center = False
require_odd_window = True
use_trajectory_loss = True
use_2d_attention = False  # 2d_attention flag (axial time x neuron)
use_image_id_encoding = True
use_images = True  # If True, use global image features instead of one-hot encoding
image_feat_scale = 0.1  # Scale factor for image features (try 0.1, 0.5, 1.0)
num_unique_images = int(np.max(image_ids_reference)) + 1
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

if use_images:
    # Compute spatial pyramid features with cross-attention (keep spatial structure)
    print("Computing spatial pyramid features for cross-attention...")
    image_grid_size = 20  # 20x20 = 400 patches
    image_patch_bank = get_spatial_pyramid_features(rsvp_images_cropped, grid_size=image_grid_size, device='cuda')
    image_patch_bank = image_patch_bank.numpy()  # (20, num_patches, patch_dim)
    num_patches = image_patch_bank.shape[1]
    patch_dim = image_patch_bank.shape[2]
    print(f"Image patch bank shape: {image_patch_bank.shape} (images, patches, features)")
    
    # Normalize patch features (global normalization across all)
    img_mean = image_patch_bank.mean()
    img_std = image_patch_bank.std()
    image_patch_bank = (image_patch_bank - img_mean) / (img_std + 1e-8)
    image_patch_bank = image_feat_scale * image_patch_bank
    print(f"Patch features normalized: mean~{image_patch_bank.mean():.3f}, std~{image_patch_bank.std():.3f}")
    
    # Create time-indexed patch sequence: (time, num_patches, patch_dim)
    ids = image_ids_reference[time_window_start:time_window_end].astype(int)
    valid_mask = ids >= 0
    image_patch_sequence = np.zeros((len(ids), num_patches, patch_dim), dtype=np.float32)
    image_patch_sequence[valid_mask] = image_patch_bank[ids[valid_mask]]
    print(f"Image patch sequence shape: {image_patch_sequence.shape} (time, patches, features)")
    
    # Don't concatenate to features - will use cross-attention instead

elif use_image_id_encoding:
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
# If augment_encodings=True: augmentations affect ALL features (replicates original "bug" behavior)
# If augment_encodings=False: augmentations only affect neural features (correct behavior)
if augment_encodings:
    num_neural_features = robs_feat_model.shape[2]  # All features including encodings
else:
    num_neural_features = robs_z.shape[2]  # Only neural features


class WindowedEyeposDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X,
        Y,
        trials,
        window_len_input,
        window_len_output,
        stride,
        min_valid_fraction,
        device,
        cache_on_gpu,
        augment=False,
        turn_off_percentage=0.0,
        turn_on_percentage=0.0,
        sample_poisson=False,
        neuron_dropout=0.0,
        mixup_alpha=0.0,
        mixup_same_time=False,
        image_ids_condensed=None,
        num_neural_features=None,
        image_patch_sequence=None,  # (time, num_patches, patch_dim) for cross-attention
    ):
        self.device = device
        self.cache_on_gpu = cache_on_gpu
        self.augment = augment
        self.turn_off_percentage = turn_off_percentage
        self.turn_on_percentage = turn_on_percentage
        self.neuron_dropout = neuron_dropout
        self.sample_poisson = sample_poisson
        self.mixup_alpha = mixup_alpha
        self.mixup_same_time = mixup_same_time
        self.image_ids_condensed = image_ids_condensed
        # Number of neural features (excluding time encoding and image_id encoding)
        # If not provided, assume all features are neural (backward compatibility)
        self.num_neural_features = num_neural_features if num_neural_features is not None else X.shape[2]
        X_raw = torch.from_numpy(X)
        Y_raw = torch.from_numpy(Y)
        valid_y = ~torch.isnan(Y_raw).any(axis=-1)
        valid_x = ~torch.isnan(X_raw).any(axis=-1)
        self.valid_mask = valid_y & valid_x
        self.X = torch.nan_to_num(X_raw, nan=input_nan_fill_value).float()
        self.Y = torch.nan_to_num(Y_raw, nan=0.0).float()
        
        # Store image patches if provided
        if image_patch_sequence is not None:
            self.image_patches = torch.from_numpy(image_patch_sequence).float()
            if self.cache_on_gpu:
                self.image_patches = self.image_patches.to(self.device, non_blocking=True)
        else:
            self.image_patches = None
        
        if self.cache_on_gpu:
            self.X = self.X.to(self.device, non_blocking=True)
            self.Y = self.Y.to(self.device, non_blocking=True)
            self.valid_mask = self.valid_mask.to(self.device, non_blocking=True)
        self.window_len_input = window_len_input
        self.window_len_output = window_len_output
        self.output_offset = (window_len_input - window_len_output) // 2
        self.indices = []
        for trial in trials:
            for start in range(0, X.shape[1] - window_len_input + 1, stride):
                out_start = start + self.output_offset
                out_end = out_start + window_len_output
                window_mask = self.valid_mask[trial, out_start:out_end]
                if window_mask.float().mean() >= min_valid_fraction:
                    self.indices.append((trial, start))
        
        # Build index for same-time mixup
        if self.mixup_same_time and self.mixup_alpha > 0.0:
            self.start_to_indices = {}
            for idx, (_, start) in enumerate(self.indices):
                if start not in self.start_to_indices:
                    self.start_to_indices[start] = []
                self.start_to_indices[start].append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial, start = self.indices[idx]
        if self.image_ids_condensed is not None:
            window_ids = self.image_ids_condensed[start : start + self.window_len_input]
            if (window_ids == -1).any():
                raise ValueError(f"CRITICAL: Model attempted to train on a window containing -1 image IDs at trial {trial}, start {start}!")

        X_win, Y_win, mask, out_start, out_end = self._get_base_item(idx)

        # All augmentations only affect neural features (first num_neural_features columns)
        # Time encoding and image_id encoding columns are left unchanged
        nf = self.num_neural_features

        if self.augment and self.mixup_alpha > 0.0:
            if self.mixup_same_time:
                candidates = [i for i in self.start_to_indices[start] if i != idx]
                if candidates:
                    mix_idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                else:
                    mix_idx = idx  # No other same-start window, skip mixup
            else:
                mix_idx = torch.randint(0, len(self.indices), (1,)).item()
            
            if mix_idx != idx:
                X_mix, Y_mix, mask_mix, _, _ = self._get_base_item(mix_idx)
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                X_win = X_win.clone()
                X_win[:, :nf] = lam * X_win[:, :nf] + (1 - lam) * X_mix[:, :nf]
                Y_win = lam * Y_win + (1 - lam) * Y_mix
                mask = mask * mask_mix
        if self.augment and (self.turn_off_percentage > 0.0 or self.turn_on_percentage > 0.0):
            if self.mixup_alpha <= 0.0:
                X_win = X_win.clone()
            X_neural = X_win[:, :nf]
            if self.turn_off_percentage > 0.0:
                on_mask = X_neural > 0
                off_draw = torch.rand_like(X_neural, dtype=torch.float32)
                X_neural = torch.where(on_mask & (off_draw < self.turn_off_percentage), torch.zeros_like(X_neural), X_neural)
            if self.turn_on_percentage > 0.0:
                off_mask = X_neural == 0
                on_draw = torch.rand_like(X_neural, dtype=torch.float32)
                X_neural = torch.where(off_mask & (on_draw < self.turn_on_percentage), torch.ones_like(X_neural), X_neural)
            X_win[:, :nf] = X_neural

        if self.augment and self.sample_poisson:
            if self.mixup_alpha <= 0.0 and self.turn_off_percentage <= 0.0 and self.turn_on_percentage <= 0.0:
                X_win = X_win.clone()
            X_neural = X_win[:, :nf]
            X_neural[X_neural > 0] = torch.poisson(X_neural[X_neural > 0])
            X_win[:, :nf] = X_neural
        
        if self.augment and self.neuron_dropout > 0.0:
            # Create dropout mask only for neural features
            neuron_mask = (torch.rand((1, nf), device=X_win.device) > self.neuron_dropout).float()
            
            if not (self.turn_off_percentage > 0.0 or self.turn_on_percentage > 0.0 or self.mixup_alpha > 0.0 or self.sample_poisson):
                X_win = X_win.clone()
            X_win[:, :nf] *= neuron_mask

        time_idx = torch.arange(
            out_start,
            out_end,
            dtype=torch.int64,
            device=self.device if self.cache_on_gpu else None,
        )
        
        # Return image patches if available
        if self.image_patches is not None:
            trial, start = self.indices[idx]
            patches_win = self.image_patches[start:start + self.window_len_input]
            return X_win, Y_win, mask, time_idx, patches_win
        else:
            return X_win, Y_win, mask, time_idx, None

    def _get_base_item(self, idx):
        trial, start = self.indices[idx]
        X_win = self.X[trial, start:start + self.window_len_input, :]
        out_start = start + self.output_offset
        out_end = out_start + self.window_len_output
        Y_win = self.Y[trial, out_start:out_end, :]
        mask = self.valid_mask[trial, out_start:out_end].float()
        return X_win, Y_win, mask, out_start, out_end


class TransformerEyepos(torch.nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, model_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.head = torch.nn.Linear(model_dim, 2)

    def forward(self, x, image_patches=None):
        # image_patches ignored for backward compatibility
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.head(x)


class TransformerEyeposWithCrossAttn(torch.nn.Module):
    """Transformer with cross-attention to image patches."""
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout_rate, patch_dim):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, model_dim)
        
        # Project image patches to model_dim
        self.patch_proj = torch.nn.Linear(patch_dim, model_dim)
        
        # Cross-attention: spikes attend to image patches
        self.cross_attn = torch.nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.cross_norm = torch.nn.LayerNorm(model_dim)
        
        # Main transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.head = torch.nn.Linear(model_dim, 2)

    def forward(self, x, image_patches):
        """
        x: (batch, time, input_dim) - spike features + time encoding
        image_patches: (batch, time, num_patches, patch_dim) - patches per timestep
        """
        batch_size, time_len, num_patches, patch_dim_in = image_patches.shape
        
        # Project spikes to model_dim
        x = self.input_proj(x)  # (batch, time, model_dim)
        
        # Project patches to model_dim
        patches = self.patch_proj(image_patches)  # (batch, time, num_patches, model_dim)
        
        # Cross-attention: each timestep's spike repr attends to its image's patches
        # Reshape for efficient batched attention: (batch * time, 1, model_dim) queries (batch * time, num_patches, model_dim)
        x_flat = x.reshape(batch_size * time_len, 1, -1)  # (B*T, 1, model_dim)
        patches_flat = patches.reshape(batch_size * time_len, num_patches, -1)  # (B*T, num_patches, model_dim)
        
        attn_out, _ = self.cross_attn(x_flat, patches_flat, patches_flat)
        attn_out = attn_out.reshape(batch_size, time_len, -1)  # (batch, time, model_dim)
        
        # Residual + norm
        x = self.cross_norm(x + attn_out)
        
        # Self-attention across time
        x = self.encoder(x)
        
        return self.head(x)


class AxialTransformerLayer(torch.nn.Module):
    def __init__(self, model_dim, num_heads, dropout_rate):
        super().__init__()
        self.time_attn = torch.nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.neuron_attn = torch.nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm_time = torch.nn.LayerNorm(model_dim)
        self.norm_neuron = torch.nn.LayerNorm(model_dim)
        self.norm_ff = torch.nn.LayerNorm(model_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(model_dim * 4, model_dim),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        bsz, time_len, num_neurons, dim = x.shape
        x_t = self.norm_time(x)
        x_t = x_t.permute(0, 2, 1, 3).reshape(bsz * num_neurons, time_len, dim)
        attn_t, _ = self.time_attn(x_t, x_t, x_t, need_weights=False)
        attn_t = attn_t.reshape(bsz, num_neurons, time_len, dim).permute(0, 2, 1, 3)
        x = x + attn_t

        x_n = self.norm_neuron(x)
        x_n = x_n.reshape(bsz * time_len, num_neurons, dim)
        attn_n, _ = self.neuron_attn(x_n, x_n, x_n, need_weights=False)
        attn_n = attn_n.reshape(bsz, time_len, num_neurons, dim)
        x = x + attn_n

        x_ff = self.norm_ff(x)
        x = x + self.ff(x_ff)
        return x


class AxialTransformerEyepos(torch.nn.Module):
    def __init__(
        self,
        num_neurons,
        model_dim,
        num_heads,
        num_layers,
        dropout_rate,
        window_len_input,
    ):
        super().__init__()
        self.input_proj = torch.nn.Linear(1, model_dim)
        self.time_embed = torch.nn.Embedding(window_len_input, model_dim)
        self.neuron_embed = torch.nn.Embedding(num_neurons, model_dim)
        self.layers = torch.nn.ModuleList(
            [AxialTransformerLayer(model_dim, num_heads, dropout_rate) for _ in range(num_layers)]
        )
        self.head = torch.nn.Linear(model_dim, 2)

    def forward(self, x):
        bsz, time_len, num_neurons = x.shape
        x = self.input_proj(x.unsqueeze(-1))
        time_idx = torch.arange(time_len, device=x.device)
        neuron_idx = torch.arange(num_neurons, device=x.device)
        time_emb = self.time_embed(time_idx)[None, :, None, :]
        neuron_emb = self.neuron_embed(neuron_idx)[None, None, :, :]
        x = x + time_emb + neuron_emb
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=2)
        return self.head(x)


def masked_mse(pred, target, mask):
    if pred.shape[1] != target.shape[1]:
        raise ValueError("pred and target must have same time dimension.")
    err = (pred - target) ** 2
    # err = torch.sqrt(err.sum(dim=-1))
    err = err.sum(dim=-1)
    masked = err * mask
    return masked.sum() / (mask.sum() + 1e-8)


def masked_mse_center(pred, target, mask, center_idx):
    pred_c = pred[:, center_idx, :]
    target_c = target[:, center_idx, :]
    mask_c = mask[:, center_idx]
    err = (pred_c - target_c) ** 2
    err = err.sum(dim=-1)
    masked = err * mask_c
    return masked.sum() / (mask_c.sum() + 1e-8)


def trajectory_loss(
    pred,
    target,
    mask,
    lambda_pos,
    lambda_vel,
    lambda_accel,
    vel_thresh,
    event_weight,
):
    if pred.shape[1] != target.shape[1]:
        raise ValueError("pred and target must have same time dimension.")
    pos_loss = masked_mse(pred, target, mask)

    if lambda_vel <= 0 and lambda_accel <= 0:
        return lambda_pos * pos_loss

    vel_pred = pred[:, 1:, :] - pred[:, :-1, :]
    vel_tgt = target[:, 1:, :] - target[:, :-1, :]
    mask_vel = mask[:, 1:] * mask[:, :-1]
    vel_err = ((vel_pred - vel_tgt) ** 2).sum(dim=-1)
    vel_mag = torch.sqrt((vel_tgt ** 2).sum(dim=-1) + 1e-8)
    event_mask = (vel_mag > vel_thresh).float()
    vel_weight = 1.0 + event_weight * event_mask
    vel_loss = (vel_err * mask_vel * vel_weight).sum() / (
        (mask_vel * vel_weight).sum() + 1e-8
    )

    if lambda_accel <= 0:
        return lambda_pos * pos_loss + lambda_vel * vel_loss

    accel_pred = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
    accel_tgt = vel_tgt[:, 1:, :] - vel_tgt[:, :-1, :]
    mask_acc = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    accel_err = ((accel_pred - accel_tgt) ** 2).sum(dim=-1)
    smooth_mask = (vel_mag[:, 1:] < vel_thresh).float()
    accel_weight = mask_acc * smooth_mask
    accel_loss = (accel_err * accel_weight).sum() / (
        accel_weight.sum() + 1e-8
    )

    return (
        lambda_pos * pos_loss
        + lambda_vel * vel_loss
        + lambda_accel * accel_loss
    )


def slice_pred_to_output(pred, window_len_input, window_len_output):
    if window_len_output > window_len_input:
        raise ValueError("window_len_output must be <= window_len_input.")
    if pred.shape[1] == window_len_output:
        return pred
    output_offset = (window_len_input - window_len_output) // 2
    return pred[:, output_offset:output_offset + window_len_output, :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = WindowedEyeposDataset(
    robs_feat_model,
    y_target,
    train_trials,
    window_len_input,
    window_len_output,
    window_stride,
    min_valid_fraction,
    device,
    cache_data_on_gpu,
    augment=True,
    turn_off_percentage=augmentation_turn_off_percentage,
    turn_on_percentage=augmentation_turn_on_percentage,
    sample_poisson=sample_poisson,
    neuron_dropout=augmentation_neuron_dropout,
    mixup_alpha=augmentation_mixup_alpha,
    mixup_same_time=augmentation_mixup_same_time,
    image_ids_condensed=image_ids_reference[time_window_start:time_window_end] if use_image_id_encoding and not use_images else None,
    num_neural_features=num_neural_features,
    image_patch_sequence=image_patch_sequence if use_images else None,
)
val_dataset = WindowedEyeposDataset(
    robs_feat_model,
    y_target,
    val_trials,
    window_len_input,
    window_len_output,
    window_stride,
    min_valid_fraction,
    device,
    cache_data_on_gpu,
    augment=False,
    turn_off_percentage=0.0,
    turn_on_percentage=0.0,
    num_neural_features=num_neural_features,
    image_patch_sequence=image_patch_sequence if use_images else None,
)

loader_num_workers = 0 if cache_data_on_gpu else dataloader_num_workers
loader_pin_memory = False if cache_data_on_gpu else dataloader_pin_memory

def custom_collate_fn(batch):
    """Custom collate that handles optional image patches (5th element may be None)."""
    X = torch.stack([item[0] for item in batch])
    Y = torch.stack([item[1] for item in batch])
    mask = torch.stack([item[2] for item in batch])
    time_idx = torch.stack([item[3] for item in batch])
    # Handle patches - stack if not None, otherwise return None
    if batch[0][4] is not None:
        patches = torch.stack([item[4] for item in batch])
    else:
        patches = None
    return X, Y, mask, time_idx, patches

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=loader_num_workers,
    pin_memory=loader_pin_memory,
    persistent_workers=loader_num_workers > 0,
    collate_fn=custom_collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=loader_num_workers,
    pin_memory=loader_pin_memory,
    persistent_workers=loader_num_workers > 0,
    collate_fn=custom_collate_fn,
)

if use_2d_attention:
    model = AxialTransformerEyepos(
        robs_feat_model.shape[2],
        model_dim=transformer_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        dropout_rate=transformer_dropout,
        window_len_input=window_len_input,
    ).to(device)
elif use_images:
    model = TransformerEyeposWithCrossAttn(
        robs_feat_model.shape[2],
        model_dim=transformer_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        dropout_rate=transformer_dropout,
        patch_dim=patch_dim,
    ).to(device)
    print(f"Using TransformerEyeposWithCrossAttn with {num_patches} patches of dim {patch_dim}")
else:
    model = TransformerEyepos(
        robs_feat_model.shape[2],
        model_dim=transformer_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        dropout_rate=transformer_dropout,
    ).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
optimizer = schedulefree.RAdamScheduleFree(model.parameters())
pred_all = None

for epoch in range(num_epochs):
    model.train()
    optimizer.train() if optimizer.__class__.__module__.startswith("schedulefree") else None
    train_losses = []
    center_idx = window_len_output // 2
    for X_batch, y_batch, mask_batch, time_idx, patches_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        time_idx = time_idx.to(device)
        if patches_batch is not None:
            patches_batch = patches_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch, patches_batch)
        pred = slice_pred_to_output(pred, window_len_input, window_len_output)
        if use_trajectory_loss:
            loss = trajectory_loss(
                pred,
                y_batch,
                mask_batch,
                lambda_pos=lambda_pos,
                lambda_vel=lambda_vel,
                lambda_accel=lambda_accel,
                vel_thresh=velocity_event_thresh,
                event_weight=velocity_event_weight,
            )
        elif loss_on_center:
            loss = masked_mse_center(pred, y_batch, mask_batch, center_idx)
        else:
            loss = masked_mse(pred, y_batch, mask_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    optimizer.eval() if optimizer.__class__.__module__.startswith("schedulefree") else None
    val_losses = []
    center_idx = window_len_output // 2
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, time_idx, patches_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            time_idx = time_idx.to(device)
            if patches_batch is not None:
                patches_batch = patches_batch.to(device)
            pred = model(X_batch, patches_batch)
            pred = slice_pred_to_output(pred, window_len_input, window_len_output)
            if use_trajectory_loss:
                loss = trajectory_loss(
                    pred,
                    y_batch,
                    mask_batch,
                    lambda_pos=lambda_pos,
                    lambda_vel=lambda_vel,
                    lambda_accel=lambda_accel,
                    vel_thresh=velocity_event_thresh,
                    event_weight=velocity_event_weight,
                )
            elif loss_on_center:
                loss = masked_mse_center(pred, y_batch, mask_batch, center_idx)
            else:
                loss = masked_mse(pred, y_batch, mask_batch)
            val_losses.append(loss.item())

    print(
        f"Epoch {epoch + 1:02d} | "
        f"train loss: {np.mean(train_losses):.4f} | "
        f"val loss: {np.mean(val_losses):.4f}"
    )
#%%
def run_inference(
    model,
    robs_feat_input,
    trials,
    window_len_input,
    window_len_output,
    device=None,
    batch_size=512,
    use_overlap=False,
    overlap_stride=None,
    center_crop=None,
    edge_align=False,
    image_patch_sequence=None,  # (time, num_patches, patch_dim) for cross-attention
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    pred_all = np.full((len(trials), robs_feat_input.shape[1], 2), np.nan)
    half_input = window_len_input // 2
    output_offset = (window_len_input - window_len_output) // 2
    with torch.no_grad():
        for i, trial in enumerate(trials):
            X = robs_feat_input[trial]
            time_len = X.shape[0]
            if time_len < window_len_input:
                continue
            max_start = max(0, time_len - window_len_input)
            if use_overlap:
                stride = overlap_stride
                if stride is None:
                    stride = max(1, window_len_output // 2)
                crop_len = window_len_output
                if center_crop is not None:
                    if isinstance(center_crop, float):
                        crop_len = int(round(window_len_output * center_crop))
                    else:
                        crop_len = int(center_crop)
                    crop_len = max(1, min(window_len_output, crop_len))
                crop_offset = (window_len_output - crop_len) // 2
                starts = np.arange(0, max_start + 1, stride)
                sum_pred = np.zeros((time_len, 2), dtype=np.float32)
                count = np.zeros(time_len, dtype=np.float32)
                if edge_align:
                    extra_sum = np.zeros((time_len, 2), dtype=np.float32)
                    extra_count = np.zeros(time_len, dtype=np.float32)
                for b in range(0, len(starts), batch_size):
                    starts_batch = starts[b:b + batch_size]
                    idx = starts_batch[:, None] + np.arange(window_len_input)[None, :]
                    X_win = X[idx]
                    X_t = torch.from_numpy(np.nan_to_num(X_win, nan=input_nan_fill_value)).float().to(device)
                    # Get patches for this batch if available
                    patches_t = None
                    if image_patch_sequence is not None:
                        patches_win = image_patch_sequence[idx]  # (batch, window_len, num_patches, patch_dim)
                        patches_t = torch.from_numpy(patches_win).float().to(device)
                    pred_batch_full = model(X_t, patches_t).cpu().numpy()
                    pred_batch = pred_batch_full
                    if crop_len != window_len_output:
                        pred_batch = pred_batch_full[:, crop_offset:crop_offset + crop_len]
                    out_start = starts_batch + output_offset + crop_offset
                    out_idx = out_start[:, None] + np.arange(crop_len)[None, :]
                    valid = (out_idx >= 0) & (out_idx < time_len)
                    if np.any(valid):
                        idx_flat = out_idx[valid].astype(int)
                        pred_flat = pred_batch[valid]
                        np.add.at(sum_pred, idx_flat, pred_flat)
                        np.add.at(count, idx_flat, 1.0)

                    if edge_align:
                        left_threshold = output_offset + crop_offset
                        right_threshold = max_start - left_threshold
                        right_crop_offset = window_len_output - crop_len
                        right_output_offset = window_len_input - window_len_output
                        for j, start in enumerate(starts_batch):
                            if start < left_threshold:
                                crop_offset_j = 0
                                output_offset_j = 0
                            elif start > right_threshold:
                                crop_offset_j = right_crop_offset
                                output_offset_j = right_output_offset
                            else:
                                continue
                            if crop_len != window_len_output:
                                pred_win = pred_batch_full[j, crop_offset_j:crop_offset_j + crop_len]
                            else:
                                pred_win = pred_batch_full[j]
                            out_start = start + output_offset_j + crop_offset_j
                            out_idx = out_start + np.arange(crop_len)
                            valid = (out_idx >= 0) & (out_idx < time_len)
                            if np.any(valid):
                                idx_flat = out_idx[valid].astype(int)
                                pred_flat = pred_win[valid]
                                np.add.at(extra_sum, idx_flat, pred_flat)
                                np.add.at(extra_count, idx_flat, 1.0)
                np.divide(
                    sum_pred,
                    count[:, None],
                    out=pred_all[i, :time_len],
                    where=count[:, None] > 0,
                )
                if edge_align:
                    np.divide(
                        extra_sum,
                        extra_count[:, None],
                        out=pred_all[i, :time_len],
                        where=(count[:, None] == 0) & (extra_count[:, None] > 0),
                    )
            else:
                starts = np.clip(np.arange(time_len) - half_input, 0, max_start)
                idx = starts[:, None] + np.arange(window_len_input)[None, :]
                X_win = X[idx]
                center_idx = (np.arange(time_len) - starts) - output_offset
                for b in range(0, time_len, batch_size):
                    X_batch = X_win[b:b + batch_size]
                    X_t = torch.from_numpy(np.nan_to_num(X_batch, nan=input_nan_fill_value)).float().to(device)
                    # Get patches for this batch if available
                    patches_t = None
                    if image_patch_sequence is not None:
                        batch_idx = idx[b:b + batch_size]
                        patches_win = image_patch_sequence[batch_idx]
                        patches_t = torch.from_numpy(patches_win).float().to(device)
                    pred_batch = model(X_t, patches_t).cpu().numpy()
                    centers = center_idx[b:b + batch_size]
                    valid = (centers >= 0) & (centers < window_len_output)
                    if np.any(valid):
                        rows = np.where(valid)[0]
                        pred_all[i, b:b + batch_size][rows] = pred_batch[rows, centers[valid]]
    return pred_all


trials_all = np.concatenate([train_trials, val_trials])
pred_all = run_inference(
    model,
    robs_feat_model,
    trials_all,
    window_len_input,
    window_len_output,
    device=device,
    use_overlap=True,
    overlap_stride=1,
    center_crop=0.8,
    edge_align=True,
    image_patch_sequence=image_patch_sequence if use_images else None,
)


#%%
def plot_trial_trace(
    model,
    robs_feat_input,
    robs_feat_ridge,
    ridge_w,
    ridge_mean,
    ridge_std,
    ridge_window_len_input,
    eyepos_actual,
    eyepos_mean,
    train_trials,
    val_trials,
    trial_idx=0,
    split="val",
    device=None,
    center_per_trial=False,
    show_ridge=False,
    show_pred=False,
    show_all_train_traces = False,
):
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    trials = train_trials if split == "train" else val_trials
    if trial_idx < 0 or trial_idx >= len(trials):
        raise IndexError("trial_idx out of range for selected split.")

    trial = trials[trial_idx]
    X = robs_feat_input[trial]
    y = eyepos_actual[trial]
    valid = (~np.isnan(X).any(axis=-1)) & (~np.isnan(y).any(axis=-1))

    if device is None:
        device = next(model.parameters()).device

    # model.eval()
    # pred_pos = np.full_like(y, np.nan)
    # half_input = window_len_input // 2
    # max_start = max(0, X.shape[0] - window_len_input)
    # output_offset = (window_len_input - window_len_output) // 2
    # with torch.no_grad():
    #     for t in range(X.shape[0]):
    #         start = min(max(t - half_input, 0), max_start)
    #         end = start + window_len_input
    #         X_win = X[start:end]
    #         if X_win.shape[0] != window_len_input:
    #             continue
    #         X_t = torch.from_numpy(np.nan_to_num(X_win, nan=0.0)).float()[None, ...]
    #         X_t = X_t.to(device)
    #         pred_win = model(X_t).squeeze(0).cpu().numpy()
    #         center_idx = (t - start) - output_offset
    #         if 0 <= center_idx < window_len_output:
    #             pred_pos[t] = pred_win[center_idx]

    
    # if center_per_trial:
    #     pred_pos = pred_pos + eyepos_mean[trial]

    # if "pred_all" in globals() and "trials_all" in globals():
    #     trial_match = np.where(trials_all == trial)[0]
    #     if trial_match.size > 0:
    #         pred_ref = pred_all[int(trial_match[0])]
    #         if center_per_trial:
    #             pred_ref = pred_ref + eyepos_mean[trial]
    #         ref_t = torch.from_numpy(pred_ref)
    #         pos_t = torch.from_numpy(pred_pos)
    #         is_close = torch.allclose(ref_t, pos_t, rtol=1e-4, atol=1e-6, equal_nan=True)
    #         if not is_close:
    #             # max_diff = torch.nanmax(torch.abs(ref_t - pos_t)).item()
    #             print(f"pred_all mismatch for trial {trial}")
    #             # print(f"max abs diff: {max_diff}")
    #             print(f"pred_ref: {pred_ref}")
    #             print(f"pred_pos: {pred_pos}")
    #             raise ValueError("pred_all mismatch")
    # else:
    #     raise ValueError("pred_all and trials_all not defined")

    trial_match = np.where(trials_all == trial)[0]
    pred_pos = pred_all[int(trial_match[0])]
    if center_per_trial:
        pred_pos = pred_pos + eyepos_mean[trial]

    pred_plot = pred_pos.copy()
    pred_plot[~valid] = np.nan

    ridge_plot = None
    if show_ridge:
        ridge_pred = predict_ridge_trial_global(
            robs_feat_ridge,
            ridge_w,
            ridge_mean,
            ridge_std,
            ridge_window_len_input,
            trial,
        )
        valid_ridge = ~np.isnan(y).any(axis=-1)
        if center_per_trial:
            ridge_pred = ridge_pred + eyepos_mean[trial]
        ridge_pred[~valid_ridge] = np.nan
        ridge_plot = ridge_pred

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1, 1, 1, 1.6]}
    )
    t = np.arange(y.shape[0])
    valid_xy = ~np.isnan(y).any(axis=-1)
    axes[0].plot(t[valid_xy], y[valid_xy, 0], color="black", label="actual")
    if show_pred:
        axes[0].plot(t, pred_plot[:, 0], color="tab:blue", alpha=1, label="pred")
    if ridge_plot is not None:
        axes[0].plot(
            t,
            ridge_plot[:, 0],
            color="tab:green",
            alpha=0.8,
            linestyle="--",
            label="ridge",
        )
    if show_all_train_traces:
        for trial_i in train_trials:
            y_i = eyepos_actual[trial_i]
            valid_i = ~np.isnan(y_i).any(axis=-1)
            axes[0].plot(t[valid_i], y_i[valid_i, 0], color="gray", alpha=0.15)
            axes[1].plot(t[valid_i], y_i[valid_i, 1], color="gray", alpha=0.15)
    axes[0].set_ylabel("eye x")
    #set ylim to 0-1
    axes[0].set_ylim(-1, 1)
    axes[0].set_xlim(time_window_start,time_window_end)
    axes[0].legend(frameon=False)

    axes[1].plot(t[valid_xy], y[valid_xy, 1], color="black", label="actual")
    if show_pred:
        axes[1].plot(t, pred_plot[:, 1], color="tab:orange", alpha=1, label="pred")
    
    if ridge_plot is not None:
        axes[1].plot(
            t,
            ridge_plot[:, 1],
            color="tab:green",
            alpha=0.8,
            linestyle="--",
            label="ridge",
        )
    axes[1].set_ylabel("eye y")
    axes[1].set_xlabel("time bin")
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlim(time_window_start,time_window_end)
    axes[1].legend(frameon=False)
    axes[1].sharex(axes[0])

    if show_pred:
        pred_err = np.sqrt(((pred_plot - y) ** 2).sum(axis=-1))
        # pred_err_mse = ((pred_plot - y) ** 2).sum(axis=-1) 
        axes[2].plot(t[valid_xy], pred_err[valid_xy], color="tab:purple", alpha=0.8, 
        label=f"pred err, {np.nanmean(pred_err[valid_xy]):.4f}")
        # label=f"pred MSE, {np.nanmean(pred_err_mse[valid_xy]):.4f}")
    if ridge_plot is not None:
        ridge_err = np.sqrt(((ridge_plot - y) ** 2).sum(axis=-1))
        axes[2].plot(t[valid_xy], ridge_err[valid_xy], color="tab:green", alpha=0.8, linestyle="--", label="ridge err")
    axes[2].set_ylabel("euclid err")
    #plot mean error as red dotted line
    # print(f"mean error: {np.mean(pred_err[valid_xy])}")
    # axes[2].axhline(y=np.nanmean(pred_err[valid_xy]), color="red", linestyle="--", label="mean err")
    # if np.nanmax(pred_err[valid_xy]) < 0.8:
    #     axes[2].set_ylim(bottom=0, top=0.8)
    axes[2].set_xlim(time_window_start,time_window_end)
    mean_pred_err = np.sqrt(((eyepos_mean[trial] - y) ** 2).sum(axis=-1))
    axes[2].plot(t[valid_xy],mean_pred_err[valid_xy], color="tab:red", alpha=0.8, 
    label=f"mean predictor error, {np.nanmean(mean_pred_err[valid_xy]):.4f}")
    axes[2].legend(frameon=False)
    axes[2].sharex(axes[0])



    
    #plot cross at eyepos_mean
    # axes[3].plot(eyepos_mean[trial, 0], eyepos_mean[trial, 1], color="red", marker="x", label="mean")
    axes[3].plot(y[valid_xy, 0], y[valid_xy, 1], color="black", label="actual")
    if show_pred:
        axes[3].plot(
            pred_plot[valid_xy, 0],
            pred_plot[valid_xy, 1],
            color="tab:purple",
            alpha=0.8,
            label="pred",
        )
    if ridge_plot is not None:
        axes[3].plot(
            ridge_plot[valid_xy, 0],
            ridge_plot[valid_xy, 1],
            color="tab:green",
            alpha=0.8,
            linestyle="--",
            label="ridge",
        )
    axes[3].set_xlabel("eye x")
    axes[3].set_ylabel("eye y")
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].set_xlim(-1, 1)
    axes[3].set_ylim(-1, 1)
    axes[3].legend(frameon=False)

    fig.suptitle(f"{split} trial {trial} (idx {trial_idx})")
    plt.tight_layout()
    return fig, axes



#%%
# trial_idx = 0
trial_idx -= 1
fig, axes = plot_trial_trace(
    model,
    robs_feat_model,
    robs_feat,
    ridge_w,
    X_mean,
    X_std,
    ridge_window_len_input,
    eyepos_aligned,
    eyepos_mean,
    train_trials,
    val_trials,
    trial_idx=trial_idx,
    split="train",
    center_per_trial=center_per_trial,
    show_ridge=False,
    show_pred=True,
)

#%%
trial_idx = -1
#%%
trial_idx += 1
fig, axes = plot_trial_trace(
    model,
    robs_feat_model,
    robs_feat,
    ridge_w,
    X_mean,
    X_std,
    ridge_window_len_input,
    eyepos_aligned,
    eyepos_mean,
    train_trials,
    val_trials,
    trial_idx=trial_idx,
    split="val",
    center_per_trial=center_per_trial,
    show_ridge=False,
    show_pred=True,
    show_all_train_traces=True,
)

#%%