#!/usr/bin/env python
"""
Example script for checking whether a model config builds

This script demonstrates:
1. Builds a model from a config
2. Prepares data
3. Tests the forward and backward on one batch
4. Tests the jacobian
"""
#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision, math, os

import lightning as pl

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from DataYatesV1.utils.data import prepare_data

import wandb as wdb
from lightning.pytorch.loggers import wandb
from typing import Tuple

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
import os
# only reveal cuda:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from DataYatesV1.utils.torch import get_free_device
# Check if CUDA is available
device = get_free_device()
# device = torch.device('cuda:0')
# device = torch.device('cpu')
print(f"Using device: {device}")

from DataYatesV1.utils.ipython import enable_autoreload
enable_autoreload()

AMP_BF16 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)


#%%
import yaml
import os

dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_basic_multi_120"
# dataset_configs_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset/dataset_cones_multi"
# List full paths to *.yaml files that do not contain "base" in the name
yaml_files = [
    f for f in os.listdir(dataset_configs_path)
    if f.endswith(".yaml") and "base" not in f
]

from DataYatesV1.models.config_loader import load_dataset_configs
dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)

#%%
dataset_config = dataset_configs[0]
dataset_config['types'] = ['backimage']
dataset_config['transforms']['stim'] = {'source': 'stim', 
        'ops': [{'fftwhitening': {}}, {'pixelnorm': {}}, {'unsqueeze': 1}]}

dataset_config['transforms']['shift'] = {'source': 'eyepos',
 'ops': [{'diff': {'axis': 0}}],
 'expose_as': 'shift'}

dataset_config['keys_lags']['shift'] = list(range(len(dataset_config['keys_lags']['stim'])))
#%%
train_dset, val_dset, dataset_config = prepare_data(dataset_config)

# --- Example Usage ---
#%%
from scipy.ndimage import uniform_filter1d

def fixations_only(dset):
    good_inds = dset.dsets[0].covariates['dfs'].sum(1)==dset.dsets[0].covariates['dfs'].shape[1]
    spd = np.hypot(dset.dsets[0].covariates['shift'][:,0], dset.dsets[0].covariates['shift'][:,1])*120

    good_inds = np.where(good_inds)[0]

    # boxcar filter
    spd_thresh = uniform_filter1d((spd<10).float(), size=25)


    good_inds = good_inds[spd_thresh[good_inds] > .99]
    dset.inds = dset.inds[np.isin(dset.inds[:,1], good_inds),:]

    return dset

train_dset = fixations_only(train_dset)
val_dset = fixations_only(val_dset)

train_dset.dsets[0].covariates['shift'] = train_dset.dsets[0].covariates['shift'] * 37/5
val_dset.dsets[0].covariates['shift'] = val_dset.dsets[0].covariates['shift'] * 37/5

train_dl = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=os.cpu_count()//2)
val_dl = DataLoader(val_dset, batch_size=256, shuffle=True, num_workers=os.cpu_count()//2)

sampler = iter(train_dl)
#%%
batch = next(sampler)
stim = batch['stim']

from torchvision.utils import make_grid

plt.figure(figsize=(20,8))

im = stim[:10].reshape(-1, 1, *stim.shape[3:]).cpu()
plt.imshow(make_grid(im, nrow=25, normalize=True, scale_each=True).permute(1, 2, 0).numpy(), extent=[0, 25, 0, 10], origin='lower')
# for i in range(10):
#     x = batch['shift'][i,:]
#     x = x - torch.mean(x, 0, keepdim=True)
#     x = x
#     plt.plot(x/25+i-.5, label=f'shift {i}')

plt.legend()
#%%

batch = next(iter(train_dl))
from torchvision.utils import make_grid
im = batch['stim'][:10].reshape(-1, 1, *batch['stim'].shape[3:])
plt.imshow(make_grid(im, nrow=25, normalize=True, scale_each=True).permute(1, 2, 0).numpy())


import torch, torch.nn as nn, torch.nn.functional as F, math
import lightning as pl
import torchvision

#%%
# --------------------- Poisson latent ---------------------
class Poisson:
    def __init__(self, log_rate, t=0.0):
        # log_rate: any shape
        self.log_rate = log_rate
        self.rate = torch.exp(self.log_rate.clamp(max=5.0)) + 1e-6
        # n_trials chosen from max rate (no grad path)
        with torch.no_grad():
            rmax = max(self.rate.max().item(), 1.0)
            self.n_trials = int(math.ceil(rmax * 5))
        self.t = float(t)

    def rsample(self, hard: bool = False):
        # Reparameterized via exponential inter-arrival; soft indicator for differentiability
        x = torch.distributions.Exponential(self.rate).rsample((self.n_trials,))  # [n_trials, ...]
        times = torch.cumsum(x, dim=0)
        indicator = (times < 1.0).to(self.log_rate.dtype)
        if not (hard or self.t == 0):
            indicator = torch.sigmoid((1.0 - times) / self.t)
        return indicator.sum(0)  # same shape as log_rate

    @staticmethod
    def kl_from_du(prior_log_rate, du):
        """
        KL between Poisson(r) and Poisson(r') with r' = r * exp(du).
        KL = r - r' + r' * log(r'/r) = r - r' + r' * du
        """
        r  = torch.exp(prior_log_rate.clamp(max=5.0)) + 1e-6
        r_ = torch.exp((prior_log_rate + du).clamp(max=5.0)) + 1e-6
        return r - r_ + r_ * du  # same shape as prior



# --------------------- Small helpers ---------------------
def softplus_min(x, minval=1e-6):
    return F.softplus(x) + minval

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, pad=1):
        super().__init__()
        self.in_ch = in_ch; self.hid_ch = hid_ch
        self.conv_gates = nn.Conv2d(in_ch + hid_ch, 2*hid_ch, k, padding=pad)
        self.conv_cand  = nn.Conv2d(in_ch + hid_ch, hid_ch,   k, padding=pad)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.hid_ch, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        gates = self.conv_gates(torch.cat([x, h], dim=1))
        z_gate, r_gate = gates.chunk(2, dim=1)
        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)

        cand = torch.tanh(self.conv_cand(torch.cat([x, r*h], dim=1)))
        h_new = (1 - z) * h + z * cand
        return h_new

# --------------------- Sequential PVAE ---------------------
class SeqPVAE(nn.Module):
    def __init__(self, H=28, W=28, ksize=15, stride=5, pad=5, num_latents=256, hid_ch=128):
        super().__init__()
        self.num_latents = num_latents
        self.ksize, self.stride, self.pad = ksize, stride, pad
        self.H, self.W = H, W

        # Decoder z -> x
        self.dec = nn.ConvTranspose2d(num_latents, 1, ksize, stride=stride, padding=pad, bias=True)

        # Feedforward encoder feature extractor on x_t
        self.enc_x = nn.Sequential(
            nn.Conv2d(1, num_latents, ksize, stride=stride, padding=pad, bias=True),
            nn.SiLU(),
            nn.Conv2d(num_latents, num_latents, 1, stride=1, padding=0, bias=True),
        )

        # Infer latent spatial size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, H, W)
            zshape = self.enc_x(dummy).shape
        _, C, H_lat, W_lat = zshape
        self.H_lat, self.W_lat = H_lat, W_lat
        self.latent_dim = num_latents * H_lat * W_lat

        # Prior base log-rate (learned, broadcastable to batch)
        self.prior0 = nn.Parameter(torch.zeros(1, num_latents, H_lat, W_lat))

        # Transition prior: log_rate_t = prior0 + trans(z_{t-1})
        # Use a light conv stack on previous z (counts) as input
        self.trans_prior = nn.Sequential(
            nn.Conv2d(num_latents, num_latents, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(num_latents, num_latents, 3, padding=1),
        )

        # Posterior RNN (ConvGRU) input: concat(enc_x(x_t), optional z_{t-1})
        self.use_zprev_in_posterior = True
        in_ch = num_latents + (num_latents if self.use_zprev_in_posterior else 0)
        self.rnn = ConvGRUCell(in_ch=in_ch, hid_ch=hid_ch, k=3, pad=1)

        # Posterior head: GRU hidden -> du (log-rate modulation)
        self.post_head = nn.Sequential(
            nn.Conv2d(hid_ch, num_latents, 3, padding=1),
        )

        # temperature for Poisson relax
        self.t = 1.0

        # persistent hidden state for streaming (optional)
        self._h_stream = None
        self._zprev_stream = None

    def decode(self, z):
        x = self.dec(z, output_size=(z.size(0), 1, self.H, self.W))
        return torch.sigmoid(x)

    def _prior_log_rate_t(self, z_prev, t_idx):
        if t_idx == 0 or (z_prev is None):
            return self.prior0
        # transition on z_{t-1}; detach if you want a looser bound
        drift = self.trans_prior(z_prev)
        return (self.prior0 + drift)

    def begin_stream(self, B: int, device, dtype):
        self._h_stream = None
        self._zprev_stream = None

    def end_stream(self):
        self._h_stream = None
        self._zprev_stream = None

    def forward_step(self, x_t, h_prev=None, z_prev=None, t_idx=0, validation=False):
        """
        One filtering step. Shapes:
          x_t: [B,1,H,W]
          z_prev: [B,C,H_lat,W_lat] or None
          h_prev: [B,Hhid,H_lat,W_lat] or None
        Returns dict with z_t, y_t, kl_t, h_t, etc.
        """
        # Encode current frame to latent grid space
        feat_x = self.enc_x(x_t)  # [B,C,H_lat,W_lat]

        # Posterior GRU input
        if self.use_zprev_in_posterior and z_prev is not None:
            rnn_in = torch.cat([feat_x, z_prev], dim=1)
        else:
            rnn_in = torch.cat([feat_x, torch.zeros_like(feat_x)], dim=1)

        h_t = self.rnn(rnn_in, h_prev)

        # Posterior modulation du_t
        du_t = self.post_head(h_t).clamp(max=5.0)

        # Prior log-rate at t
        prior_log_rate_t = self._prior_log_rate_t(z_prev, t_idx).clamp(max=5.0)

        # Posterior log-rate and sample z_t
        log_rate_post_t = (prior_log_rate_t + du_t).clamp(max=5.0)
        dist_t = Poisson(log_rate_post_t, t=self.t)
        z_t = dist_t.rsample(hard=validation)  # [B,C,H_lat,W_lat]

        # Decode to observation
        y_t = self.decode(z_t)

        # Per-step KL (mean over spatial/latent dims later)
        kl_map = Poisson.kl_from_du(prior_log_rate_t, du_t)

        return {
            "h_t": h_t,
            "z_t": z_t,
            "y_t": y_t,
            "du_t": du_t,
            "prior_log_rate_t": prior_log_rate_t,
            "kl_map": kl_map
        }

    def forward(self, x_seq, carry_state=False):
        """
        x_seq: [B, T, 1, H, W]
        carry_state:
           - False: reset hidden each call (typical inside a batch)
           - True:  carry ConvGRU/z across calls (streaming)
        """

        validation = not torch.is_grad_enabled()
        B, T, _, _, _ = x_seq.shape

        h = self._h_stream if (carry_state) else None
        z_prev = self._zprev_stream if (carry_state) else None

        ys, zs, dus, kls = [], [], [], []
        for t in range(T):
            out = self.forward_step(x_seq[:, t], h_prev=h, z_prev=z_prev, t_idx=t, validation=validation)
            h = out["h_t"]
            z_prev = out["z_t"]
            ys.append(out["y_t"])
            zs.append(out["z_t"])
            dus.append(out["du_t"])
            kls.append(out["kl_map"])

        if carry_state:
            # Persist for next call (detach to avoid backward through time across batches)
            self._h_stream = h.detach()
            self._zprev_stream = z_prev.detach()

        Y = torch.stack(ys, dim=1)    # [B,T,1,H,W]
        Z = torch.stack(zs, dim=1)    # [B,T,C,H_lat,W_lat]
        DU = torch.stack(dus, dim=1)  # [B,T,C,H_lat,W_lat]
        KL = torch.stack(kls, dim=1)  # [B,T,C,H_lat,W_lat]
        return Y, Z, DU, KL


class PL_SeqPVAE(pl.LightningModule):
    def __init__(self, image_size=(51, 51), num_latents=256):
        super().__init__()
        self.model = SeqPVAE(H=image_size[0], W=image_size[1], num_latents=num_latents)
        self.opt_ctor = torch.optim.Adam
        self.opt_params = dict(lr=1e-3)
        self.beta = 1.0

    def configure_optimizers(self):
        return self.opt_ctor(self.parameters(), **self.opt_params)

    def forward(self, x_seq, carry_state=False):
        return self.model.forward(x_seq, carry_state=carry_state)


    def training_step(self, batch, batch_idx):
        x_seq = batch['stim'].permute(0, 2, 1, 3, 4).contiguous()                  # [B,T,1,H,W]
        T = x_seq.size(1)
        # Schedules (like your original)
        self.beta = min(5.0, 5.0 * self.current_epoch / 250.0)
        self.model.t = max(1.0 - 0.95 * self.current_epoch / 250.0, 0.05)
        self.log('beta', self.beta, prog_bar=True)
        self.log('t', self.model.t, prog_bar=True)

        y_seq, z_seq, du_seq, kl_map = self(x_seq, carry_state=False)

        # Recon MSE
        mse_t = ((y_seq - x_seq) ** 2).flatten(2).sum(-1)   # [B,T]
        mse = mse_t.mean()

        # KL per-step: sum over latent grid, mean over (B,T)
        kl_t = kl_map.flatten(2).sum(-1)                    # [B,T]
        kl = kl_t.mean()

        loss = self.beta * kl + mse
        self.log_dict({
            'train_loss': loss,
            'train_mse': mse,
            'train_kl': kl,
            'train_elbo': (kl + mse)
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x_seq = batch['stim'].permute(0, 2, 1, 3, 4).contiguous()  # [B,T,1,H,W]
        y_seq, z_seq, du_seq, kl_map = self(x_seq, carry_state=False)
        mse = ((y_seq - x_seq) ** 2).flatten(2).sum(-1).mean()
        kl  = kl_map.flatten(2).sum(-1).mean()
        self.log('val_mse', mse.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_kl', kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo', (kl + mse).item(), on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            # visualize first time slice
            x0 = x_seq[:, 0]
            y0 = y_seq[:, 0]
            fig_y = torchvision.utils.make_grid(y0.reshape(-1, 1, x0.size(-2), x0.size(-1)), nrow=10)
            fig_x = torchvision.utils.make_grid(x0.reshape(-1, 1, x0.size(-2), x0.size(-1)), nrow=10)
            fig_w = torchvision.utils.make_grid(
                self.model.dec.weight.reshape(-1, 1, self.model.ksize, self.model.ksize),
                nrow=int(math.sqrt(self.model.num_latents)), normalize=True
            )
            self.logger.experiment.log({
                'weights': wdb.Image(fig_w, caption="weights"),
                'recons_t0': wdb.Image(fig_y, caption="recons_t0"),
                'inputs_t0': wdb.Image(fig_x, caption="inputs_t0"),
            })

        return (self.beta * kl + mse)


#%%
num_latents = int(6**2)
max_iters = 11
beta_max = 10.0
model = PL_SeqPVAE(image_size=(51,51), num_latents=num_latents)

# print(f'prior shape: {model.model.prior.shape}')
# print(f'viewed prior shape: {model.model._view_latent_map(model.model.prior).shape}')

#%%
k = model.model.dec.weight.clone()
h = k.shape[-1]
k = k.reshape(-1, 1, h, h)
fig_w = torchvision.utils.make_grid(k, nrow=int(math.sqrt(num_latents)), normalize=True)
plt.imshow(fig_w.permute(1,2,0).detach().cpu().numpy())

#%%
name = f'PVAE_SSM_latents{num_latents}_iters{max_iters}_beta{beta_max}' #change to your run name

project_name = "fixSSM" #change to your wandb project name
root_dir = "data"
bsize = 256
train_device = "0"
device = train_device + "," #lightning device formatting
checkpoint_dir = os.path.join(root_dir, name)
os.makedirs(checkpoint_dir, exist_ok=True)


#%% if load from checkpoint
load_from_checkpoint = False
if load_from_checkpoint:
    from pathlib import Path
    ckpts = sorted(Path("/home/jake/repos/predictive-model-benchmark/data/seqGVAE__latents4_iters11_beta10.0").glob("*.ckpt"))
    print(ckpts)  # choose one, e.g., the last one
    model = PL_SeqPVAE.load_from_checkpoint(str(ckpts[-1]), strict=False, device='cpu')
    model = model.to('cpu')
    model.model.nonneg_decode = False
# model = PL_ampSSM.load_from_checkpoint(ckpt_path)

#%%
batch = next(iter(train_dl))
stim = batch['stim'].permute(0, 2, 1, 3, 4).contiguous()

out = model.model(stim)

#%%
def show_recons(batch, out, T):
    
    fig, axs = plt.subplots(2, T, figsize=(T*2, 5))
    for t in range(T):
        axs[0, t].set_xticks([])
        axs[0, t].set_yticks([])
        axs[1, t].set_xticks([])
        axs[1, t].set_yticks([])
        axs[0, t].imshow(batch[0][0][t].detach().cpu(), cmap='gray')
        axs[1, t].imshow(out[0][0][t].detach().cpu(), cmap='gray')
    return fig

fig = show_recons(stim.permute(0, 2, 1, 3, 4).contiguous(), out[0].permute(0, 2, 1, 3, 4), max_iters)

#%%
train = True
if train:
    # turn off progress bar
    trainer_args = {
        "callbacks": [pl.pytorch.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, monitor='val_elbo', save_top_k=1, mode='min', verbose=True)],
        "accelerator": "cpu" if device == 'cpu' else "gpu",
        "logger": wandb.WandbLogger(project=project_name, name=name, save_code=True),
        "gradient_clip_val": 1.0, 
        "enable_progress_bar": True
    }
    trainer_args["logger"].watch(model, log="all")

    if device != 'cpu':
        trainer_args["devices"] = device
    trainer = pl.Trainer(**trainer_args, default_root_dir=checkpoint_dir,
            max_epochs=500, num_sanity_val_steps=0)
    trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)

# %%

# %% predict

u_hist, z_hist, y_seq, y_pred_seq, klu_hist = model(batch) 

# %%

z_hist.shape

# %%

_ = plt.plot(z_hist[0].detach().cpu())
# %% get transition matrix for pose
u_prev = u_hist[:,-1]
z_prev = z_hist[:,-1]
B = u_prev.shape[0]
u = model.model._unpack(u_prev)  # [B,D,G,H,W]
z = model.model._unpack(z_prev)

# L2 amplitudes (consistent for u and z)
u_amp = torch.linalg.vector_norm(u, dim=2)                       # [B,D,H,W]
z_amp = torch.linalg.vector_norm(z, dim=2)                       # [B,D,H,W]
u_pose = u / (u_amp.unsqueeze(2) + model.model.trans_eps)


T = 20
fig, axs = plt.subplots(1, T, figsize=(T*2, 5))
for t in range(T):
    # u_next = model.model._transition_mean(u_prev, z_prev)

    
    


    z_pose = z / (z_amp.unsqueeze(2) + model.model.trans_eps)

    p_tilde = torch.einsum('dgo,bdghw->bdohw', model.model.W_u, u_pose) \
            + torch.einsum('dgo,bdghw->bdohw', model.model.W_z, z_pose) \
                + model.model.b.view(1, model.model.D, model.model.G, 1, 1)
    u_pose_new = p_tilde / (torch.linalg.vector_norm(p_tilde, dim=2, keepdim=True) + model.model.trans_eps)

        
    u_next = u_amp.unsqueeze(2) * u_pose_new
    u_next = model.model._pack(u_next)

    z_next = model.model._sample_z(u_next)
    y_next = model.model._decode_from_vec(z_next)
    axs[t].imshow(y_next[0][0].detach().cpu(), cmap='gray')
    u_prev, z_prev = u_next, z_next


# %%
