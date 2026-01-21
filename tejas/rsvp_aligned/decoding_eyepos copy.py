#%%
import os
from pathlib import Path
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

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False
import contextlib

#%%
subject = 'Allen'
date = '2022-03-04'

dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'
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
trials = np.unique(trial_inds)

NC = dataset.dsets[dset_idx]['robs'].shape[1]
T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
NT = len(trials)

fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

# Loop over trials and align responses
robs = np.nan*np.zeros((NT, T, NC))
dfs = np.nan*np.zeros((NT, T, NC))
eyepos = np.nan*np.zeros((NT, T, 2))
fix_dur =np.nan*np.zeros((NT,))

for itrial in tqdm(range(NT)):
    # print(f"Trial {itrial}/{NT}")
    ix = trials[itrial] == trial_inds
    ix = ix & fixation
    if np.sum(ix) == 0:
        continue
    
    stim_inds = np.where(ix)[0]
    # stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]


    psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
    fix_dur[itrial] = len(psth_inds)
    robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
    dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
    eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()

time_window_cutoff = 150
good_trials = fix_dur > 20
robs = robs[good_trials][:,:time_window_cutoff,:]
# dfs = dfs[good_trials]
eyepos = eyepos[good_trials][:,:time_window_cutoff,:]
fix_dur = fix_dur[good_trials]


ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 160)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 160)

robs.shape #(79, 335, 133) [trials, time, cells]
eyepos.shape #(79, 335, 2) [trials, time, xycoords]

#%%
# Decoder setup
rng = np.random.default_rng(0)
train_frac = 0.8
use_time_encoding = True
time_enc_dim = 8
time_enc_scale = 1.0
ridge_alpha = 10.0
window_len = 67
window_stride = 16
min_valid_fraction = 0.8
num_epochs = 60
batch_size = 64
learning_rate = 1e-3
dropout_rate = 0
hidden_dim = 64
lag_bins = 2
predict_velocity = False
center_per_trial = True
standardize_gru = True
bidirectional_gru = True
use_time_bias = False
time_bias_dim = 16
model_type = "transformer"
transformer_dim = 64
transformer_heads = 4
transformer_layers = 2
transformer_dropout = 0.1
loss_on_center = True
require_odd_window = True

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


def compute_velocity(eyepos_in):
    vel = np.full_like(eyepos_in, np.nan)
    vel[:, 1:, :] = np.diff(eyepos_in, axis=1)
    return vel


robs_aligned, eyepos_aligned = apply_lag(robs, eyepos, lag_bins)
time_len = robs_aligned.shape[1]
if window_len is None:
    window_len = time_len
if loss_on_center and require_odd_window and window_len % 2 == 0:
    raise ValueError("window_len must be odd when loss_on_center=True.")

if center_per_trial:
    eyepos_mean = np.nanmean(eyepos_aligned, axis=1)
    eyepos_centered = eyepos_aligned - eyepos_mean[:, None, :]
else:
    eyepos_mean = np.zeros((num_trials, 2))
    eyepos_centered = eyepos_aligned

if predict_velocity:
    y_target = compute_velocity(eyepos_centered)
else:
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


def build_timepoint_samples(X, Y, trials):
    X_t = X[trials]
    Y_t = Y[trials]
    valid_y = ~np.isnan(Y_t).any(axis=-1)
    valid_x = ~np.isnan(X_t).any(axis=-1)
    mask = valid_y & valid_x
    X_flat = X_t.reshape(-1, X_t.shape[-1])
    Y_flat = Y_t.reshape(-1, Y_t.shape[-1])
    mask_flat = mask.reshape(-1)
    return X_flat[mask_flat], Y_flat[mask_flat]


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


# Ridge baseline
X_train, y_train = build_timepoint_samples(robs_feat, y_target, train_trials)
X_val, y_val = build_timepoint_samples(robs_feat, y_target, val_trials)
X_train, X_val, X_mean, X_std = standardize_train(X_train, X_val)

ridge_w = fit_ridge(X_train, y_train, ridge_alpha)
y_val_pred = X_val @ ridge_w
ridge_mse = np.mean((y_val - y_val_pred) ** 2, axis=0)
ridge_r2 = r2_score(y_val, y_val_pred)

print("Ridge baseline:")
print(f"  MSE (x, y): {ridge_mse}")
print(f"  R2  (x, y): {ridge_r2}")


if standardize_gru:
    robs_z, robs_mean, robs_std = standardize_robs(
        robs_aligned, train_trials
    )
else:
    robs_z = robs_aligned

if time_encoding is not None:
    robs_feat_gru = np.concatenate(
        [robs_z, time_enc_scale * time_encoding], axis=2
    )
else:
    robs_feat_gru = robs_z


# GRU model with windowed training
class WindowedEyeposDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, trials, window_len, stride, min_valid_fraction):
        self.X = X
        self.Y = Y
        self.window_len = window_len
        self.indices = []
        valid_y = ~np.isnan(Y).any(axis=-1)
        valid_x = ~np.isnan(X).any(axis=-1)
        valid_mask = valid_y & valid_x
        for trial in trials:
            for start in range(0, X.shape[1] - window_len + 1, stride):
                window_mask = valid_mask[trial, start:start + window_len]
                if window_mask.mean() >= min_valid_fraction:
                    self.indices.append((trial, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial, start = self.indices[idx]
        X_win = self.X[trial, start:start + self.window_len, :]
        Y_win = self.Y[trial, start:start + self.window_len, :]
        time_idx = np.arange(start, start + self.window_len, dtype=np.int64)
        valid_y = ~np.isnan(Y_win).any(axis=-1)
        valid_x = ~np.isnan(X_win).any(axis=-1)
        mask = valid_y & valid_x
        X_win = torch.from_numpy(np.nan_to_num(X_win, nan=0.0)).float()
        Y_win = torch.from_numpy(np.nan_to_num(Y_win, nan=0.0)).float()
        mask = torch.from_numpy(mask.astype(np.float32))
        time_idx = torch.from_numpy(time_idx)
        return X_win, Y_win, mask, time_idx


class GRUEyepos(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional_gru,
        )
        head_in = hidden_dim * 2 if bidirectional_gru else hidden_dim
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.head = torch.nn.Linear(head_in, 2)
        if use_time_bias:
            self.time_embed = torch.nn.Embedding(time_len, time_bias_dim)
            self.time_head = torch.nn.Linear(time_bias_dim, 2)

    def forward(self, x, time_idx=None):
        out, _ = self.gru(x)
        out = self.dropout(out)
        pred = self.head(out)
        if use_time_bias:
            if time_idx is None:
                raise ValueError("time_idx is required when use_time_bias is True.")
            time_feat = self.time_embed(time_idx)
            pred = pred + self.time_head(time_feat)
        return pred


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

    def forward(self, x, time_idx=None):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.head(x)


def masked_mse(pred, target, mask):
    err = (pred - target) ** 2
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


train_dataset = WindowedEyeposDataset(
    robs_feat_gru,
    y_target,
    train_trials,
    window_len,
    window_stride,
    min_valid_fraction,
)
val_dataset = WindowedEyeposDataset(
    robs_feat_gru,
    y_target,
    val_trials,
    window_len,
    window_stride,
    min_valid_fraction,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_type == "gru":
    model = GRUEyepos(
        robs_feat_gru.shape[2], hidden_dim=hidden_dim, dropout_rate=dropout_rate
    ).to(device)
elif model_type == "transformer":
    model = TransformerEyepos(
        robs_feat_gru.shape[2],
        model_dim=transformer_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        dropout_rate=transformer_dropout,
    ).to(device)
else:
    raise ValueError("model_type must be 'gru' or 'transformer'.")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    center_idx = window_len // 2
    for X_batch, y_batch, mask_batch, time_idx in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        time_idx = time_idx.to(device)
        optimizer.zero_grad()
        pred = model(X_batch, time_idx=time_idx)
        if loss_on_center:
            loss = masked_mse_center(pred, y_batch, mask_batch, center_idx)
        else:
            loss = masked_mse(pred, y_batch, mask_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    center_idx = window_len // 2
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, time_idx in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            time_idx = time_idx.to(device)
            pred = model(X_batch, time_idx=time_idx)
            if loss_on_center:
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
def plot_trial_trace(
    model,
    robs_feat_input,
    eyepos_actual,
    eyepos_mean,
    train_trials,
    val_trials,
    trial_idx=0,
    split="val",
    device=None,
    predict_velocity=False,
    center_per_trial=False,
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

    X_t = torch.from_numpy(np.nan_to_num(X, nan=0.0)).float()[None, ...]
    time_idx = torch.from_numpy(np.arange(X.shape[0], dtype=np.int64))[None, ...]
    if device is None:
        device = next(model.parameters()).device
    X_t = X_t.to(device)
    time_idx = time_idx.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X_t, time_idx=time_idx).squeeze(0).cpu().numpy()

    if predict_velocity:
        if center_per_trial:
            y_centered = y - eyepos_mean[trial]
        else:
            y_centered = y
        pred_centered = np.cumsum(pred, axis=0)
        valid_idx = np.where(~np.isnan(y_centered).any(axis=-1))[0]
        if len(valid_idx) > 0:
            pred_centered += y_centered[valid_idx[0]]
        else:
            pred_centered[:] = np.nan
        pred_pos = pred_centered
    else:
        pred_pos = pred

    if center_per_trial:
        pred_pos = pred_pos + eyepos_mean[trial]

    pred_plot = pred_pos.copy()
    pred_plot[~valid] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(y[:, 0], color="black", label="actual")
    axes[0].plot(pred_plot[:, 0], color="tab:blue", alpha=0.8, label="pred")
    axes[0].set_ylabel("eye x")
    #set ylim to 0-1
    axes[0].set_ylim(-1, 1)


    axes[0].legend(frameon=False)

    axes[1].plot(y[:, 1], color="black", label="actual")
    axes[1].plot(pred_plot[:, 1], color="tab:orange", alpha=0.8, label="pred")
    axes[1].set_ylabel("eye y")
    axes[1].set_xlabel("time bin")
    axes[1].set_ylim(-1, 1)
    axes[1].legend(frameon=False)

    fig.suptitle(f"{split} trial {trial} (idx {trial_idx})")
    plt.tight_layout()
    return fig, axes

#%%
trial_idx += 1
plot_trial_trace(model, robs_feat_gru, eyepos_aligned, eyepos_mean, train_trials, val_trials, trial_idx=trial_idx, split="train", predict_velocity=predict_velocity, center_per_trial=center_per_trial)

#%%
def select_best_trial(
    model,
    robs_feat_input,
    eyepos_actual,
    eyepos_mean,
    trials,
    device=None,
    predict_velocity=False,
    center_per_trial=False,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    best_idx = 0
    best_mse = np.inf
    with torch.no_grad():
        for i, trial in enumerate(trials):
            X = robs_feat_input[trial]
            y = eyepos_actual[trial]
            valid = (~np.isnan(X).any(axis=-1)) & (~np.isnan(y).any(axis=-1))
            if not np.any(valid):
                continue
            X_t = torch.from_numpy(np.nan_to_num(X, nan=0.0)).float()[None, ...]
            time_idx = torch.from_numpy(np.arange(X.shape[0], dtype=np.int64))[None, ...]
            X_t = X_t.to(device)
            time_idx = time_idx.to(device)
            pred = model(X_t, time_idx=time_idx).squeeze(0).cpu().numpy()

            if predict_velocity:
                if center_per_trial:
                    y_centered = y - eyepos_mean[trial]
                else:
                    y_centered = y
                pred_centered = np.cumsum(pred, axis=0)
                valid_idx = np.where(~np.isnan(y_centered).any(axis=-1))[0]
                if len(valid_idx) > 0:
                    pred_centered += y_centered[valid_idx[0]]
                else:
                    continue
                pred_pos = pred_centered
            else:
                pred_pos = pred

            if center_per_trial:
                pred_pos = pred_pos + eyepos_mean[trial]

            err = (pred_pos - y) ** 2
            mse = np.nanmean(err[valid])
            if mse < best_mse:
                best_mse = mse
                best_idx = i
    return best_idx
# trial_idx = 0
# Plot and save a single trial trace
output_dir = Path("/home/tejas/VisionCore/tejas/rsvp_aligned/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

for plot_split in ["train", "val"]:
    plot_trials = val_trials if plot_split == "val" else train_trials
    plot_trial_idx = select_best_trial(
        model,
        robs_feat_gru,
        eyepos_aligned,
        eyepos_mean,
        plot_trials,
        predict_velocity=predict_velocity,
        center_per_trial=center_per_trial,
    )
    fig, _ = plot_trial_trace(
        model,
        robs_feat_gru,
        eyepos_aligned,
        eyepos_mean,
        train_trials,
        val_trials,
        trial_idx=plot_trial_idx,
        split=plot_split,
        predict_velocity=predict_velocity,
        center_per_trial=center_per_trial,
    )
    plot_path = output_dir / f"eyepos_trace_{plot_split}_{plot_trial_idx}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
# %%
