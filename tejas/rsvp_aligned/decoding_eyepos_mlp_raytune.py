#!/usr/bin/env python3
"""
Ray Tune hyperparameter optimization for MLP eyepos decoding model.
Runs 100 trials with 15 concurrent trials on GPU 0.
"""
import os
from pathlib import Path
import torch
import numpy as np
import contextlib
import schedulefree
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import optuna

# Set GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Disable strict metric checking (workaround for Ray Tune metric reporting)
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

from models.config_loader import load_dataset_configs
from models.data import prepare_data
from tqdm import tqdm
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1.utils.general import get_clock_functions
from scripts.mcfarland_sim import get_fixrsvp_stack

# Load data (same as original script)
subject = 'Allen'
date = '2022-03-04'
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp_all_cells.yaml'
dataset_configs = load_dataset_configs(dataset_configs_path)
dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
    train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)

sess = train_dset.dsets[0].metadata['sess']
cids = dataset_config['cids']

inds = torch.concatenate([
    train_dset.get_dataset_inds('fixrsvp'),
    val_dset.get_dataset_inds('fixrsvp')
], dim=0)

dataset = train_dset.shallow_copy()
dataset.inds = inds

dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
t_bins = dataset.dsets[dset_idx].covariates['t_bins'].numpy()
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

rsvp_images = get_fixrsvp_stack(frames_per_im=1)
ptb2ephys, _ = get_clock_functions(sess.exp)
image_ids = np.full((NT, T), -1, dtype=np.int64)
robs = np.nan*np.zeros((NT, T, NC))
dfs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))

for itrial in tqdm(range(NT), desc="Loading data"):
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
    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()

time_window_start = 0
time_window_end = 200
good_trials = fix_dur > 20
robs = robs[good_trials][:,time_window_start:time_window_end,:]
dfs = dfs[good_trials][:,time_window_start:time_window_end,:]
eyepos = eyepos[good_trials][:,time_window_start:time_window_end,:]
fix_dur = fix_dur[good_trials]
image_ids = image_ids[good_trials][:, time_window_start:time_window_end]

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
            unmatched_trials_and_start_time_ind_of_mismatch[trial_ind] = start_time_ind_of_mismatch
            break

trials_to_remove = []
for trial_ind, start_time_ind_of_mismatch in unmatched_trials_and_start_time_ind_of_mismatch.items():
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

# Fixed hyperparameters (not tuned)
FIXED_HYPERPARAMS = {
    'window_len_input': 50,
    'window_len_output': 50,
    'lambda_pos': 1.0,
    'lambda_vel': 0.4,
    'lambda_accel': 0,
    'velocity_event_thresh': 0.02,
    'velocity_event_weight': 0,
    'train_frac': 0.8,
    'rng_seed': 0,
    'use_time_encoding': True,
    'time_enc_dim': 8,
    'time_enc_scale': 1.0,
    'ridge_alpha': 10.0,
    'ridge_window_len_input': 10,
}

# Import model classes and functions (copy from original)
class MLPEyepos(torch.nn.Module):
    def __init__(self, input_dim, window_len_input, hidden_dim=64, num_layers=2, dropout_rate=0.1):
        super().__init__()
        flattened_dim = window_len_input * input_dim
        layers = []
        layers.append(torch.nn.Linear(flattened_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        self.layers = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(hidden_dim, window_len_input * 2)
        self.window_len_input = window_len_input

    def forward(self, x):
        bsz = x.shape[0]
        x = x.reshape(bsz, -1)
        x = self.layers(x)
        x = self.head(x)
        x = x.reshape(bsz, self.window_len_input, 2)
        return x

def masked_mse(pred, target, mask):
    if pred.shape[1] != target.shape[1]:
        raise ValueError("pred and target must have same time dimension.")
    err = (pred - target) ** 2
    err = err.sum(dim=-1)
    masked = err * mask
    return masked.sum() / (mask.sum() + 1e-8)

def trajectory_loss(pred, target, mask, lambda_pos, lambda_vel, lambda_accel, vel_thresh, event_weight):
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
    vel_loss = (vel_err * mask_vel * vel_weight).sum() / ((mask_vel * vel_weight).sum() + 1e-8)

    if lambda_accel <= 0:
        return lambda_pos * pos_loss + lambda_vel * vel_loss

    accel_pred = vel_pred[:, 1:, :] - vel_pred[:, :-1, :]
    accel_tgt = vel_tgt[:, 1:, :] - vel_tgt[:, :-1, :]
    mask_acc = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    accel_err = ((accel_pred - accel_tgt) ** 2).sum(dim=-1)
    smooth_mask = (vel_mag[:, 1:] < vel_thresh).float()
    accel_weight = mask_acc * smooth_mask
    accel_loss = (accel_err * accel_weight).sum() / (accel_weight.sum() + 1e-8)

    return lambda_pos * pos_loss + lambda_vel * vel_loss + lambda_accel * accel_loss

def slice_pred_to_output(pred, window_len_input, window_len_output):
    if window_len_output > window_len_input:
        raise ValueError("window_len_output must be <= window_len_input.")
    if pred.shape[1] == window_len_output:
        return pred
    output_offset = (window_len_input - window_len_output) // 2
    return pred[:, output_offset:output_offset + window_len_output, :]

def build_time_encoding(num_timepoints, dim):
    if dim % 2 != 0:
        raise ValueError("time_enc_dim must be even.")
    positions = np.arange(num_timepoints)[:, None]
    div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
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

# WindowedEyeposDataset class (copy from original)
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
        input_nan_fill_value=0,
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
        self.num_neural_features = num_neural_features if num_neural_features is not None else X.shape[2]
        X_raw = torch.from_numpy(X)
        Y_raw = torch.from_numpy(Y)
        valid_y = ~torch.isnan(Y_raw).any(axis=-1)
        valid_x = ~torch.isnan(X_raw).any(axis=-1)
        self.valid_mask = valid_y & valid_x
        self.X = torch.nan_to_num(X_raw, nan=input_nan_fill_value).float()
        self.Y = torch.nan_to_num(Y_raw, nan=0.0).float()
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
        nf = self.num_neural_features

        if self.augment and self.mixup_alpha > 0.0:
            if self.mixup_same_time:
                candidates = [i for i in self.start_to_indices[start] if i != idx]
                if candidates:
                    mix_idx = candidates[torch.randint(0, len(candidates), (1,)).item()]
                else:
                    mix_idx = idx
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
            neuron_mask = (torch.rand((1, nf), device=X_win.device) > self.neuron_dropout).float()
            if not (self.turn_off_percentage > 0.0 or self.turn_on_percentage > 0.0 or self.mixup_alpha > 0.0 or self.sample_poisson):
                X_win = X_win.clone()
            X_win[:, :nf] *= neuron_mask

        time_idx = torch.arange(out_start, out_end, dtype=torch.int64, device=self.device if self.cache_on_gpu else None)
        return X_win, Y_win, mask, time_idx

    def _get_base_item(self, idx):
        trial, start = self.indices[idx]
        X_win = self.X[trial, start:start + self.window_len_input, :]
        out_start = start + self.output_offset
        out_end = out_start + self.window_len_output
        Y_win = self.Y[trial, out_start:out_end, :]
        mask = self.valid_mask[trial, out_start:out_end].float()
        return X_win, Y_win, mask, out_start, out_end

# Prepare data once (shared across all trials)
rng = np.random.default_rng(FIXED_HYPERPARAMS['rng_seed'])
robs_aligned, eyepos_aligned = apply_lag(robs, eyepos, 0)  # Will use config lag_bins
time_len = robs_aligned.shape[1]
num_trials = robs.shape[0]
trial_indices = np.arange(num_trials)
rng.shuffle(trial_indices)
num_train = int(FIXED_HYPERPARAMS['train_frac'] * num_trials)
train_trials = trial_indices[:num_train]
val_trials = trial_indices[num_train:]

# Store preprocessed data in a dict to pass to trainable
PREPROCESSED_DATA = {
    'robs': robs,
    'eyepos': eyepos,
    'image_ids_reference': image_ids_reference,
    'time_window_start': time_window_start,
    'time_window_end': time_window_end,
    'train_trials': train_trials,
    'val_trials': val_trials,
    'num_trials': num_trials,
    'time_len': time_len,
}

def train_mlp(config):
    """Ray Tune trainable function"""
    # Set random seeds for reproducibility
    torch.manual_seed(FIXED_HYPERPARAMS['rng_seed'])
    np.random.seed(FIXED_HYPERPARAMS['rng_seed'])
    
    # Extract config values
    window_len_input = FIXED_HYPERPARAMS['window_len_input']
    window_len_output = FIXED_HYPERPARAMS['window_len_output']
    
    # Process data with config hyperparameters
    lag_bins = int(config['lag_bins'])
    robs_aligned, eyepos_aligned = apply_lag(
        PREPROCESSED_DATA['robs'], 
        PREPROCESSED_DATA['eyepos'], 
        lag_bins
    )
    time_len = robs_aligned.shape[1]
    
    # Center eyepos
    if config['center_per_trial']:
        global_mean = np.nanmean(eyepos_aligned[PREPROCESSED_DATA['train_trials']], axis=(0, 1))
        eyepos_mean = np.tile(global_mean[None, :], (PREPROCESSED_DATA['num_trials'], 1))
        eyepos_centered = eyepos_aligned - eyepos_mean[:, None, :]
    else:
        eyepos_mean = np.zeros((PREPROCESSED_DATA['num_trials'], 2))
        eyepos_centered = eyepos_aligned
    
    y_target = eyepos_centered
    
    # Time encoding
    time_encoding = None
    if FIXED_HYPERPARAMS['use_time_encoding'] and FIXED_HYPERPARAMS['time_enc_dim'] > 0:
        time_encoding = build_time_encoding(time_len, FIXED_HYPERPARAMS['time_enc_dim'])
        time_encoding = np.tile(time_encoding[None, :, :], (PREPROCESSED_DATA['num_trials'], 1, 1))
    
    # Standardize inputs
    if config['standardize_inputs']:
        robs_train = robs_aligned[PREPROCESSED_DATA['train_trials']]
        mean = np.nanmean(robs_train, axis=(0, 1), keepdims=True)
        std = np.nanstd(robs_train, axis=(0, 1), keepdims=True)
        std[std < 1e-8] = 1.0
        robs_z = (robs_aligned - mean) / std
    else:
        robs_z = robs_aligned
    
    # Build feature model
    if time_encoding is not None:
        robs_feat_model = np.concatenate(
            [robs_z, FIXED_HYPERPARAMS['time_enc_scale'] * time_encoding], axis=2
        )
    else:
        robs_feat_model = robs_z
    
    # Image ID encoding
    if config['use_image_id_encoding']:
        num_unique_images = int(np.max(PREPROCESSED_DATA['image_ids_reference'])) + 1
        # Adjust for lag_bins - image_ids should match the aligned time dimension
        time_start = PREPROCESSED_DATA['time_window_start']
        time_end = PREPROCESSED_DATA['time_window_end'] - lag_bins  # Adjust for lag
        ids = PREPROCESSED_DATA['image_ids_reference'][time_start:time_end].astype(int)
        valid_mask = ids >= 0
        one_hot = np.zeros((len(ids), num_unique_images))
        one_hot[valid_mask, ids[valid_mask]] = 1.0
        image_feat = np.tile(one_hot[None, :, :], (PREPROCESSED_DATA['num_trials'], 1, 1))
        robs_feat_model = np.concatenate([robs_feat_model, image_feat], axis=2)
    
    # Set num_neural_features
    if config['augment_encodings']:
        num_neural_features = robs_feat_model.shape[2]
    else:
        num_neural_features = robs_z.shape[2]
    
    # Create datasets
    # Force GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = int(config['batch_size'])
    
    train_dataset = WindowedEyeposDataset(
        robs_feat_model, y_target, PREPROCESSED_DATA['train_trials'],
        window_len_input, window_len_output, 
        int(config['window_stride']), config['min_valid_fraction'],
        device, config['cache_data_on_gpu'],
        augment=True,
        turn_off_percentage=config['augmentation_turn_off_percentage'],
        turn_on_percentage=config['augmentation_turn_on_percentage'],
        sample_poisson=config['sample_poisson'],
        neuron_dropout=config['augmentation_neuron_dropout'],
        mixup_alpha=config['augmentation_mixup_alpha'],
        mixup_same_time=config['augmentation_mixup_same_time'],
        image_ids_condensed=PREPROCESSED_DATA['image_ids_reference'][
            PREPROCESSED_DATA['time_window_start']:PREPROCESSED_DATA['time_window_end'] - lag_bins
        ] if config['use_image_id_encoding'] else None,
        num_neural_features=num_neural_features,
        input_nan_fill_value=config['input_nan_fill_value'],
    )
    
    val_dataset = WindowedEyeposDataset(
        robs_feat_model, y_target, PREPROCESSED_DATA['val_trials'],
        window_len_input, window_len_output,
        int(config['window_stride']), config['min_valid_fraction'],
        device, config['cache_data_on_gpu'],
        augment=False,
        turn_off_percentage=0.0,
        turn_on_percentage=0.0,
        num_neural_features=num_neural_features,
        input_nan_fill_value=config['input_nan_fill_value'],
    )
    
    loader_num_workers = 0 if config['cache_data_on_gpu'] else int(config['dataloader_num_workers'])
    loader_pin_memory = False if config['cache_data_on_gpu'] else config['dataloader_pin_memory']
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=loader_num_workers, pin_memory=loader_pin_memory,
        persistent_workers=loader_num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=loader_num_workers, pin_memory=loader_pin_memory,
        persistent_workers=loader_num_workers > 0,
    )
    
    # Create model
    model = MLPEyepos(
        input_dim=robs_feat_model.shape[2],
        window_len_input=window_len_input,
        hidden_dim=int(config['mlp_hidden_dim']),
        num_layers=int(config['mlp_num_layers']),
        dropout_rate=config['mlp_dropout_rate'],
    ).to(device)
    
    # Optimizer (fixed to schedulefree.RAdamScheduleFree)
    optimizer = schedulefree.RAdamScheduleFree(model.parameters())
    
    # Training loop
    num_epochs = int(config['num_epochs'])
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        if hasattr(optimizer, 'train'):
            optimizer.train()
        
        for X_batch, y_batch, mask_batch, time_idx in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            pred = slice_pred_to_output(pred, window_len_input, window_len_output)
            
            if config['use_trajectory_loss']:
                loss = trajectory_loss(
                    pred, y_batch, mask_batch,
                    lambda_pos=FIXED_HYPERPARAMS['lambda_pos'],
                    lambda_vel=FIXED_HYPERPARAMS['lambda_vel'],
                    lambda_accel=FIXED_HYPERPARAMS['lambda_accel'],
                    vel_thresh=FIXED_HYPERPARAMS['velocity_event_thresh'],
                    event_weight=FIXED_HYPERPARAMS['velocity_event_weight'],
                )
            elif config['loss_on_center']:
                center_idx = window_len_output // 2
                pred_c = pred[:, center_idx, :]
                target_c = y_batch[:, center_idx, :]
                mask_c = mask_batch[:, center_idx]
                err = (pred_c - target_c) ** 2
                err = err.sum(dim=-1)
                masked = err * mask_c
                loss = masked.sum() / (mask_c.sum() + 1e-8)
            else:
                loss = masked_mse(pred, y_batch, mask_batch)
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        if hasattr(optimizer, 'eval'):
            optimizer.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, mask_batch, time_idx in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                pred = model(X_batch)
                pred = slice_pred_to_output(pred, window_len_input, window_len_output)
                
                if config['use_trajectory_loss']:
                    loss = trajectory_loss(
                        pred, y_batch, mask_batch,
                        lambda_pos=FIXED_HYPERPARAMS['lambda_pos'],
                        lambda_vel=FIXED_HYPERPARAMS['lambda_vel'],
                        lambda_accel=FIXED_HYPERPARAMS['lambda_accel'],
                        vel_thresh=FIXED_HYPERPARAMS['velocity_event_thresh'],
                        event_weight=FIXED_HYPERPARAMS['velocity_event_weight'],
                    )
                elif config['loss_on_center']:
                    center_idx = window_len_output // 2
                    pred_c = pred[:, center_idx, :]
                    target_c = y_batch[:, center_idx, :]
                    mask_c = mask_batch[:, center_idx]
                    err = (pred_c - target_c) ** 2
                    err = err.sum(dim=-1)
                    masked = err * mask_c
                    loss = masked.sum() / (mask_c.sum() + 1e-8)
                else:
                    loss = masked_mse(pred, y_batch, mask_batch)
                
                val_losses.append(loss.item())
        
        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
        
        # Report to Ray Tune (report every epoch for early stopping)
        # val_loss must be the primary metric for the scheduler
        # Use tune.report with dict (Ray 2.53.0)
        tune.report({
            "val_loss": float(mean_val_loss),
            "best_val_loss": float(best_val_loss),
            "epoch": int(epoch+1)
        })
    
    # Don't return - Ray Tune uses tune.report() instead

# Define search space
search_space = {
    'window_stride': tune.choice([1, 2, 5, 10]),
    'min_valid_fraction': tune.uniform(0.5, 1.0),
    'num_epochs': tune.choice([50, 75, 100, 125]),
    'batch_size': tune.choice([32, 64, 128, 256]),
    'learning_rate': tune.loguniform(1e-4, 1e-2),
    'lag_bins': tune.choice([0, 1, 2, 3, 4, 5]),
    'center_per_trial': tune.choice([True, False]),
    'standardize_inputs': tune.choice([True, False]),
    'cache_data_on_gpu': tune.choice([True, False]),
    'dataloader_num_workers': tune.choice([0, 2, 4, 8]),
    'dataloader_pin_memory': tune.choice([True, False]),
    'sample_poisson': tune.choice([True, False]),
    'augmentation_neuron_dropout': tune.uniform(0.0, 0.3),
    'augmentation_turn_off_percentage': tune.uniform(0.0, 0.5),
    'augmentation_turn_on_percentage': tune.uniform(0.0, 0.1),
    'augmentation_mixup_alpha': tune.uniform(0.0, 0.5),
    'augmentation_mixup_same_time': tune.choice([True, False]),
    'augment_encodings': tune.choice([True, False]),
    'mlp_hidden_dim': tune.choice([32, 64, 128, 256, 512]),
    'mlp_num_layers': tune.choice([1, 2, 3, 4, 5, 6]),
    'mlp_dropout_rate': tune.uniform(0.0, 0.5),
    'weight_decay': tune.loguniform(1e-6, 1e-2),
    'loss_on_center': tune.choice([True, False]),
    'require_odd_window': tune.choice([True, False]),
    'use_trajectory_loss': tune.choice([True, False]),
    'use_image_id_encoding': tune.choice([True, False]),
    'input_nan_fill_value': tune.choice([-1.0, 0.0]),
}

# Configure Ray Tune
scheduler = ASHAScheduler(metric="val_loss", mode="min", max_t=125, grace_period=10)

# Configure resources - limit to 15 concurrent trials on GPU 0
# Note: max_concurrent_trials is set in TuneConfig
tuner = tune.Tuner(
    train_mlp,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=100,
        scheduler=scheduler,
        search_alg=OptunaSearch(metric="val_loss", mode="min"),
        max_concurrent_trials=15,
    ),
    run_config=tune.RunConfig(
        name="mlp_eyepos_tune",
        storage_path=os.path.abspath("./ray_results"),
        stop={"training_iteration": 125},  # Max epochs
    ),
)

if __name__ == "__main__":
    # Set environment variable before Ray init
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    
    # Initialize Ray with proper resource configuration
    import ray
    try:
        ray.init(
            num_gpus=1,  # Only use 1 GPU (GPU 0)
            num_cpus=60,  # Allow enough CPUs for 15 concurrent trials (4 CPUs per trial)
            ignore_reinit_error=True,
            runtime_env={
                'excludes': [
                    '**/.git/**',
                    '**/*.deb',
                    '**/DataRowleyV1V2/**',
                    '**/*.pyc',
                    '**/__pycache__/**',
                    '**/.venv/**',
                    '**/ray_results/**',
                ],
                'working_dir': None,  # Don't package working directory
            }
        )
    except Exception as e:
        print(f"Warning: Ray init error (may already be initialized): {e}")
    
    # Run tuning
    print("Starting Ray Tune hyperparameter optimization...")
    print(f"Total trials: 100, Concurrent trials: 15")
    print(f"Using GPU 0 only")
    print("="*80)
    
    try:
        results = tuner.fit()
        
        # Print best result
        best_result = results.get_best_result("val_loss", "min")
        print("\n" + "="*80)
        print("BEST HYPERPARAMETERS:")
        print("="*80)
        print(f"Best validation loss: {best_result.metrics.get('best_val_loss', best_result.metrics.get('val_loss', 'N/A')):.4f}")
        print("\nBest config:")
        for key, value in sorted(best_result.config.items()):
            print(f"  {key}: {value}")
        print("="*80)
    except Exception as e:
        print(f"Error during tuning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Shutdown Ray
        try:
            ray.shutdown()
        except:
            pass
