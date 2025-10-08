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

#%%
train_dset, val_dset, dataset_config = prepare_data(dataset_config)

# --- Example Usage ---
#%%
good_inds = train_dset.dsets[0].covariates['dfs'].sum(1)==train_dset.dsets[0].covariates['dfs'].shape[1]
good_inds = np.where(good_inds)[0]
stim = train_dset.dsets[0].covariates['stim']
plt.imshow(stim[1000,0].detach().cpu().numpy())

#%%
from torchvision.utils import make_grid



plt.figure(figsize=(20,8))

im = stim[good_inds[:100]].reshape(-1, 1, *stim.shape[2:]).cpu()
plt.imshow(make_grid(im, nrow=25, normalize=True, scale_each=True).permute(1, 2, 0).numpy())

#%%

train_dl = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=os.cpu_count()//2)
val_dl = DataLoader(val_dset, batch_size=256, shuffle=True, num_workers=os.cpu_count()//2)
#%%

batch = next(iter(train_dl))
from torchvision.utils import make_grid
im = batch['stim'][:10].reshape(-1, 1, *batch['stim'].shape[3:])
plt.imshow(make_grid(im, nrow=25, normalize=True, scale_each=True).permute(1, 2, 0).numpy())


#%% Model Classes
# --------------------- 4D Pooling Utility ---------------------
def _l2pool4d(x_c: torch.Tensor, k_xy: int, k_feat: int, pad_xy: int, pad_feat: int, eps=1e-6) -> torch.Tensor:
    """
    Local L2 pooling (RMS) over a 4D neighborhood.
    1. Pools over the 2D spatial dimensions (H, W).
    2. Pools over the 2D feature dimensions, which are derived by
       reshaping the channel dimension C into (sqrt(C), sqrt(C)).

    Args:
        x_c (Tensor): Input tensor of shape [B, C, H, W]. C must be a perfect square.
        k_xy (int): Kernel size for spatial pooling.
        k_feat (int): Kernel size for feature pooling.
        pad_xy (int): Padding for spatial pooling.
        pad_feat (int): Padding for feature pooling.
        eps (float): Small epsilon for numerical stability in sqrt.
    """
    B, C, H, W = x_c.shape
    F_dim = int(math.sqrt(C))
    if F_dim * F_dim != C:
        raise ValueError(f"Number of channels {C} must be a perfect square.")

    # Step 1: Pool over spatial dimensions (computes mean of squares)
    x_squared = x_c * x_c
    spatially_pooled = F.avg_pool2d(
        x_squared, kernel_size=k_xy, stride=1, padding=pad_xy
    )

    # Step 2: Pool over feature dimensions
    # Permute to bring feature dimension to the end for processing
    # [B, C, H, W] -> [B, H, W, C]
    permuted = spatially_pooled.permute(0, 2, 3, 1)

    # Reshape for pooling: [B*H*W, 1, F, F]
    # Each spatial location (h,w) for each batch item is now an independent
    # item with a 1-channel, FxF "image" of features.
    feature_grid = permuted.reshape(B * H * W, 1, F_dim, F_dim)

    # Perform 2D pooling over the feature grid
    feature_pooled = F.avg_pool2d(
        feature_grid, kernel_size=k_feat, stride=1, padding=pad_feat
    )

    # Step 3: Reshape back to original tensor layout
    # [B*H*W, 1, F, F] -> [B, H, W, C]
    unpacked_features = feature_pooled.reshape(B, H, W, C)

    # [B, H, W, C] -> [B, C, H, W]
    restored_order = unpacked_features.permute(0, 3, 1, 2)

    return torch.sqrt(restored_order + eps)


class AmpPoseSSM(nn.Module):
    def __init__(self,
                 H: int = 51,
                 W: int = 51,
                 num_latents: int = 64, # Replaces D and G
                 ksize: int = 15,
                 stride: int = 5,
                 pad: int = 5,
                 output_nl: str = 'identity',
                 nonneg_decode: bool = False,
                 trans_eps: float = 1e-6,
                 inner_lr: float = 0.1,
                 pred_loss_w: float = 0.1,
                 init_log_sigma2_x: float = 0.0,
                 lp_k_xy: int = 3,     # spatial window for L2 amplitude
                 lp_k_feat: int = 3,   # 2D feature window for L2 amplitude
                 pose_k_ch: int = 3,   # depth window for pose mixing
                 pose_k_xy: int = 3):  # spatial window for pose mixing
        """
        Constructor with updated factorization scheme.
          - num_latents: Total number of latent channels (must be a perfect square).
          - lp_k_xy, lp_k_feat: Kernel sizes for 4D amplitude/pose factorization.
        """
        super().__init__()
        self.H, self.W = H, W
        self.num_latents = num_latents
        self.output_nl = output_nl
        self.nonneg_decode = nonneg_decode
        self.trans_eps = trans_eps
        self.inner_lr = inner_lr
        self.pred_loss_w = pred_loss_w

        # ---- Decoder (weight-tying geometry with encoder to get H_lat,W_lat) ----
        self.dec = nn.ConvTranspose2d(num_latents, 1, ksize, stride=stride, padding=pad, bias=False)
        nn.init.kaiming_normal_(self.dec.weight, a=0.1)

        self.enc = nn.Conv2d(1, num_latents, ksize, stride=stride, padding=pad, bias=False)
        self.enc.weight = self.dec.weight

        with torch.no_grad():
            dummy = torch.zeros(1, 1, H, W)
            z_probe = self.enc(dummy)
            self.H_lat, self.W_lat = z_probe.shape[-2:]

        self.latent_dim = num_latents * self.H_lat * self.W_lat
        self.prior = nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False)

        # ---- 4D neighborhood settings ----
        self.lp_k_xy = lp_k_xy
        self.lp_pad_xy = lp_k_xy // 2
        self.lp_k_feat = lp_k_feat
        self.lp_pad_feat = lp_k_feat // 2

        self.pose_k = (pose_k_ch, pose_k_xy, pose_k_xy)
        self.pose_pad = (pose_k_ch // 2, pose_k_xy // 2, pose_k_xy // 2)

        # ---- Pose/Amplitude mixing convs (3D with circular channel padding) ----
        # Padding is set to 0 as it will be handled manually.
        self.W_u = nn.Conv3d(1, 1, kernel_size=self.pose_k, stride=1, padding=0, bias=False)
        self.W_z = nn.Conv3d(1, 1, kernel_size=self.pose_k, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.W_u.weight, a=0.1)
        nn.init.kaiming_normal_(self.W_z.weight, a=0.1)

        self.K_u = nn.Conv3d(1, 1, kernel_size=self.pose_k, stride=1, padding=0, bias=True)
        self.K_z = nn.Conv3d(1, 1, kernel_size=self.pose_k, stride=1, padding=0, bias=True)
        nn.init.zeros_(self.K_u.weight); nn.init.zeros_(self.K_z.weight)
        nn.init.zeros_(self.K_u.bias);   nn.init.zeros_(self.K_z.bias)

        # ---- Learned variances ----
        self.log_sigma2_z = nn.Parameter(torch.full((1, self.latent_dim), -2.0))
        self.log_sigma2_u = nn.Parameter(torch.full((1, self.latent_dim), -2.0))
        
        self.raw_log_sigma2_x = nn.Parameter(torch.tensor(init_log_sigma2_x))
        self.min_sigma_x = 1e-3

    @property
    def log_sigma2_x(self):
        # STABILITY: Use softplus to ensure sigma_x^2 is always > min_sigma_x^2
        # This prevents the natural gradient preconditioner from exploding.
        return torch.log(F.softplus(self.raw_log_sigma2_x) + self.min_sigma_x**2)

    # -------------------- small helpers (shape management) --------------------
    def _view_latent_map(self, x_vec: torch.Tensor) -> torch.Tensor:
        """ [B, latent_dim] -> [B, C, H_lat, W_lat] """
        B = x_vec.shape[0]
        return x_vec.view(B, self.num_latents, self.H_lat, self.W_lat)

    def _flatten(self, ten: torch.Tensor) -> torch.Tensor:
        """ [B, C, H, W] -> [B, latent] """
        return ten.reshape(ten.size(0), -1)

    # -------------------- decode --------------------
    def _decode_from_vec(self, z: torch.Tensor) -> torch.Tensor:
        """ z: [B, latent_dim] -> x_mean: [B, 1, H, W] """
        zc = self._view_latent_map(z)
        if self.nonneg_decode:
            zc = F.relu(zc)
        x = self.dec(zc, output_size=(zc.size(0), 1, self.H, self.W))
        if self.output_nl == 'sigmoid':
            x = torch.sigmoid(x)
        return x

    # -------------------- circular convolution helper --------------------
    def _circular_channel_conv3d(self, x: torch.Tensor, conv_layer: nn.Module,
                                 pad_c: int, pad_h: int, pad_w: int) -> torch.Tensor:
        """
        Applies a 3D convolution with circular padding on the channel dimension
        and zero-padding on the spatial dimensions.
        """
        # x has shape [B, 1, C, H, W]
        # Pad spatial dimensions with zeros
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        # Pad channel dimension with wrap-around
        x_padded = F.pad(x_padded, (0, 0, 0, 0, pad_c, pad_c), mode='circular')
        return conv_layer(x_padded)

    # -------------------- transition mean --------------------
    def _transition_mean(self, u_prev: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        Fully-convolutional amplitude/pose transition with 4D factorization
        and circular channel convolutions.
        """
        uc = self._view_latent_map(u_prev)
        zc = self._view_latent_map(z_prev)

        # --- Amplitude fields via local 4D L2 pooling ---
        u_amp = _l2pool4d(uc, self.lp_k_xy, self.lp_k_feat, self.lp_pad_xy, self.lp_pad_feat, self.trans_eps)
        z_amp = _l2pool4d(zc, self.lp_k_xy, self.lp_k_feat, self.lp_pad_xy, self.lp_pad_feat, self.trans_eps)

        # --- Pose fields (unit vectors under the same local metric) ---
        u_pose = uc / (u_amp + self.trans_eps)
        z_pose = zc / (z_amp + self.trans_eps)

        # --- Pose update by 3D conv with circular channel padding ---
        pad_c, pad_h, pad_w = self.pose_pad
        pose_u_in = u_pose.unsqueeze(1) # [B, 1, C, Hl, Wl]
        pose_z_in = z_pose.unsqueeze(1)
        pose_tilde_u = self._circular_channel_conv3d(pose_u_in, self.W_u, pad_c, pad_h, pad_w)
        pose_tilde_z = self._circular_channel_conv3d(pose_z_in, self.W_z, pad_c, pad_h, pad_w)
        pose_tilde = (pose_tilde_u + pose_tilde_z).squeeze(1)

        # Renormalize to local unit "sphere"
        pose_norm = _l2pool4d(pose_tilde, self.lp_k_xy, self.lp_k_feat, self.lp_pad_xy, self.lp_pad_feat, self.trans_eps)
        pose_hat = pose_tilde / (pose_norm + self.trans_eps)

        # --- Amplitude update (multiplicative, nonnegative) ---
        amp_u_in = u_amp.unsqueeze(1)
        amp_z_in = z_amp.unsqueeze(1)
        Ku = self._circular_channel_conv3d(amp_u_in, self.K_u, pad_c, pad_h, pad_w).squeeze(1)
        Kz = self._circular_channel_conv3d(amp_z_in, self.K_z, pad_c, pad_h, pad_w).squeeze(1)
        gain = F.relu(1.0 + Ku + Kz).clamp(min=0.5, max=1.5)
        amp_hat = u_amp * gain

        # --- Recompose and pack ---
        mu_c = amp_hat * pose_hat
        return self._flatten(mu_c)

    # -------------------- other core model functions --------------------
    def _sample_z(self, u_vec: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(u_vec)
        sigma_z = torch.exp(0.5 * self.log_sigma2_z)
        return u_vec + eps * sigma_z

    def _prior_kl(self, u_t: torch.Tensor, mu_prior: torch.Tensor) -> torch.Tensor:
        diff = u_t - mu_prior
        inv_sigma2_u = torch.exp(-self.log_sigma2_u)
        return 0.5 * (diff * inv_sigma2_u * diff).sum(dim=-1)

    def _ng_step(self, mu_prior, z_vec, err_img, training: bool):
        with torch.enable_grad():
            z_vec = z_vec.requires_grad_(True)
            y = self._decode_from_vec(z_vec)
            (g_z,) = torch.autograd.grad(
                outputs=y, inputs=z_vec, grad_outputs=err_img,
                retain_graph=training, create_graph=training, allow_unused=False
            )
        sigma2_z = torch.exp(self.log_sigma2_z)
        sigma2_x = torch.exp(self.log_sigma2_x)
        precond = sigma2_z / sigma2_x
        return mu_prior + self.inner_lr * g_z * precond

    # -------------------- Inference over a sequence --------------------
    def infer(self, x_seq: torch.Tensor, training: bool) -> Tuple[torch.Tensor, ...]:
        B, _, T, _, _ = x_seq.shape
        u_prev = self.prior.expand(B, -1)
        z_prev = self._sample_z(u_prev)
        u_list, z_list, y_list, ypred_list, klu_list = [], [], [], [], []

        for t in range(T):
            mu_prior = self._transition_mean(u_prev, z_prev)
            z_pred = self._sample_z(mu_prior)
            y_pred = self._decode_from_vec(z_pred)
            z_t = self._sample_z(mu_prior)
            y_t = self._decode_from_vec(z_t)
            err = x_seq[:, :, t, :, :] - y_t
            u_t = self._ng_step(mu_prior, z_t, err, training=training)
            kl_u = self._prior_kl(u_t, mu_prior)

            u_list.append(u_t); z_list.append(z_t); y_list.append(y_t)
            ypred_list.append(y_pred); klu_list.append(kl_u)
            u_prev, z_prev = u_t, z_t

        u_hist = torch.stack(u_list, dim=1)
        z_hist = torch.stack(z_list, dim=1)
        y_seq = torch.stack(y_list, dim=2)
        y_pred_seq = torch.stack(ypred_list, dim=2)
        klu_hist = torch.stack(klu_list, dim=1)
        return u_hist, z_hist, y_seq, y_pred_seq, klu_hist

    def forward(self, x_seq: torch.Tensor, eval_mode: bool = False):
        return self.infer(x_seq, training=not eval_mode)


# --------------------- Visualization Utilities ---------------------
def _kernels_grid_from_deconv(dec_weight: torch.Tensor, nrow: int = 8):
    """ dec_weight: [num_latents, 1, k, k] """
    k = dec_weight.detach().clone()
    grid = torchvision.utils.make_grid(
        k, nrow=nrow, normalize=True, scale_each=True, value_range=None
    )
    return grid

def _to_np(grid_tensor):
    return grid_tensor.detach().cpu().permute(1, 2, 0).numpy()


# --------------------- Lightning Wrapper ---------------------
class PL_AmpPoseSSM(pl.LightningModule):
    def __init__(self,
                 image_size=(51, 51),
                 num_latents: int = 64,
                 inner_lr: float = 0.1,
                 pred_loss_w: float = 0.1,
                 beta_max: float = 10.0,
                 nonneg: bool = True,
                 learn_sigma_x: bool = True,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = AmpPoseSSM(H=image_size[0], W=image_size[1],
                                 num_latents=num_latents,
                                 inner_lr=inner_lr,
                                 pred_loss_w=pred_loss_w,
                                 nonneg_decode=nonneg,
                                 init_log_sigma2_x=(0.0 if learn_sigma_x else math.log(1.0)))
        self.beta = 0.0

    def forward(self, x):
        return self.model(x, eval_mode=not torch.is_grad_enabled())

    def _compute_losses(self, x, eval_mode: bool):
        u_hist, z_hist, y_seq, y_pred_seq, klu_hist = self.model(x, eval_mode=eval_mode)
        # Using MSE for simplicity, but log-likelihood version is also valid.
        recon = (y_seq - x).pow(2).sum((1, 2, 3, 4)).mean()
        pred_recon = (y_pred_seq - x).pow(2).sum((1, 2, 3, 4)).mean()
        kl_u = klu_hist.mean()
        loss = recon + self.hparams.pred_loss_w * pred_recon + self.beta * kl_u
        return loss, recon, pred_recon, kl_u, y_seq, y_pred_seq

    def training_step(self, batch, batch_idx):
        self.beta = min(self.hparams.beta_max, 0.5 * self.current_epoch)
        x = batch['stim']
        loss, recon, pred_recon, kl_u, _, _ = self._compute_losses(x, eval_mode=False)
        self.log_dict({
            'train_elbo': loss,
            'train_mse': recon,
            'train_pred_mse': pred_recon,
            'train_kl': kl_u,
            'beta': self.beta
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['stim']
        loss, recon, pred_recon, kl_u, y_seq, y_pred_seq = self._compute_losses(x, eval_mode=True)
        self.log_dict({
            'val_elbo': loss,
            'val_mse': recon,
            'val_pred_mse': pred_recon,
            'val_kl': kl_u
        }, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0 and wdb and self.logger and hasattr(self.logger.experiment, 'log'):
            B, _, T, H, W = x.shape
            def _mkgrid(seq):
                imgs = seq[:min(4, seq.shape[0])].reshape(-1, 1, H, W)
                return torchvision.utils.make_grid(imgs, nrow=T, normalize=True)
            
            fig_x = _mkgrid(x)
            fig_y = _mkgrid(y_seq)
            fig_yp = _mkgrid(y_pred_seq)
            kgrid = _kernels_grid_from_deconv(self.model.dec.weight, nrow=int(math.sqrt(self.hparams.num_latents)))

            self.logger.experiment.log({
                'weights': wdb.Image(_to_np(kgrid), caption='Decoder Weights'),
                'val_inputs': wdb.Image(_to_np(fig_x), caption='Inputs'),
                'val_recons': wdb.Image(_to_np(fig_y), caption='Reconstructions'),
                'val_pred_only': wdb.Image(_to_np(fig_yp), caption='Predictions'),
            })
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



#%%
num_latents = int(6**2)
max_iters = 11
beta_max = 10.0
nonneg = False
model = PL_AmpPoseSSM(image_size=(51,51), nonneg=nonneg, num_latents=num_latents, beta_max=beta_max)

print(f'prior shape: {model.model.prior.shape}')
print(f'viewed prior shape: {model.model._view_latent_map(model.model.prior).shape}')

#%%
k = model.model.dec.weight.clone()
h = k.shape[-1]
k = k.reshape(-1, 1, h, h)
fig_w = torchvision.utils.make_grid(k, nrow=int(math.sqrt(num_latents)), normalize=True)
plt.imshow(fig_w.permute(1,2,0).detach().cpu().numpy())

#%%
name = f'AmpPoseSSM_{"nonneg" if nonneg else ""}_latents{num_latents}_iters{max_iters}_beta{beta_max}' #change to your run name

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
    model = PL_AmpPoseSSM.load_from_checkpoint(str(ckpts[-1]), strict=False, device='cpu')
    model = model.to('cpu')
    model.model.nonneg_decode = False
# model = PL_ampSSM.load_from_checkpoint(ckpt_path)

#%%
batch = next(iter(train_dl))
stim = batch['stim']

out = model.model(stim, True)

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

fig = show_recons(stim, out[2], max_iters)

#%%
train = True
if train:
    # turn off progress bar
    trainer_args = {
        "callbacks": [pl.pytorch.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, monitor='val_elbo', save_top_k=1, mode='min', verbose=True)],
        "accelerator": "cpu" if device == 'cpu' else "gpu",
        "logger": wandb.WandbLogger(project=project_name, name=name, save_code=True),
        "gradient_clip_val": 100.0, 
        "enable_progress_bar": True
    }
    trainer_args["logger"].watch(model, log="all")

    if device != 'cpu':
        trainer_args["devices"] = device
    trainer = pl.Trainer(**trainer_args, default_root_dir=checkpoint_dir,
            max_epochs=1000, num_sanity_val_steps=0)
    trainer.fit(model, val_dataloaders=val_dl, train_dataloaders=train_dl)

# %%


fig = _kernels_grid_from_deconv(model.model.dec.weight, model.model.D, model.model.G, nrow=model.model.D)
plt.imshow(_to_np(fig))
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
