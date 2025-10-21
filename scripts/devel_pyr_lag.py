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
import sys
import torch
from torch.utils.data import DataLoader

import lightning as pl

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from models import build_model, initialize_model_components
from models.config_loader import load_config
from models.data import prepare_data
from models.utils.general import ensure_tensor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda:1')
# device = torch.device('cpu')
print(f"Using device: {device}")

from DataYatesV1.utils.ipython import enable_autoreload
enable_autoreload()

from contextlib import nullcontext
AMP_BF16 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)


#%%
from models.config_loader import load_dataset_configs
import os
config_path = Path("/home/jake/repos/VisionCore/experiments/model_configs/res_small_gru.yaml")

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_backimage_all.yaml"
dataset_configs = load_dataset_configs(dataset_configs_path)

#%% Initialize model
config = load_config(config_path)
model = build_model(config, dataset_configs).to(device)

# run model readout forward with dummy input
# with torch.no_grad():
#     output = model.readouts[0](torch.randn(1, 256, 1, 9, 9).to(device))

# model.readouts[0].plot_weights()


#%% Load Data
import contextlib

train_datasets = {}
val_datasets = {}
updated_configs = []

for i, dataset_config in enumerate(dataset_configs):
    if i > 1: break

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):          # ← optional
        train_dset, val_dset, dataset_config = prepare_data(dataset_config)

    # cast to bfloat16
    train_dset.cast(torch.bfloat16, target_keys=['stim', 'robs', 'dfs'])
    val_dset.cast(torch.bfloat16, target_keys=['stim', 'robs', 'dfs'])
    
    dataset_name = f"dataset_{i}"
    train_datasets[dataset_name] = train_dset
    val_datasets[dataset_name] = val_dset
    updated_configs.append(dataset_config)

    print(f"Dataset {i}: {len(train_dset)} train, {len(val_dset)} val samples")

#%%
plt.plot(train_datasets['dataset_0'].dsets[0]['stim'][:1000,0,25,25].float().cpu().numpy())

#%% prepare dataloaders

from torchvision.utils import make_grid
def plot_frames(frames, nrow=10, normalize=True):
    fig_w = make_grid(frames, nrow=nrow, normalize=normalize)
    plt.imshow(fig_w.permute(1, 2, 0).numpy()
               )
plot_frames(train_datasets['dataset_0'].dsets[0]['stim'][:150,[0],:,:].float())

from plenoptic.simulate import LaplacianPyramid
from einops import rearrange




# train_loader, val_loader = create_multidataset_loaders(train_datasets, val_datasets, batch_size=2, num_workers=os.cpu_count()//2)

#%% test one dataset
batch_size = 512
dataset_id = 0

ntrain = len(train_datasets[f'dataset_{dataset_id}'])
# inds = np.random.randint(0, ntrain - batch_size, batch_size)
inds = np.arange(1000, 1000+batch_size)
batch = train_datasets[f'dataset_{dataset_id}'][inds]

batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


#%%
from plenoptic.simulate import LaplacianPyramid
from plenoptic.tools.conv import upsample_blur, blur_downsample

from einops import rearrange
sys.path.append('..')
from models.modules.conv_layers import StandardConv
from models.modules.common import chomp
import torch.nn as nn
class PyramidStem(nn.Module):
    def __init__(self,
                 Nlevels = 3,
                 in_channels=1,
                 out_channels=64,
                 kernel_size=9,
                 stride=1,
                 padding='same', bias=False,
                 amplitude_cat=False,
                 aa_signal=True, aa_freq=True):

        super().__init__()
        self.lpyr = LaplacianPyramid(Nlevels)
        self.amplitude_cat = amplitude_cat

        self.conv = StandardConv(dim=2, in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias,
            aa_signal=aa_signal, aa_freq=aa_freq)

        # self.conv.apply_weight_norm(0)

        self.Nlevels = Nlevels
    
    def forward(self, x):
        x = rearrange(x, "B C T H W -> (B T) C H W")
        y = self.lpyr(x)
        y = [self.conv(L) for L in y]
        y = [rearrange(L, "(B T) C H W -> B C T H W", B=batch_size) for L in y]
        min_size = min([L.shape[-1] for L in y])
        y = [chomp(L, (min_size, min_size)) for L in y]
        y = torch.concat(y, dim=1)
        if self.amplitude_cat:
            A = (y[:,::2]**2 + y[:,1::2,:,:,]**2).pow(.5)
            y[:,::2] /= A
            y[:,1::2] /=A
            y = torch.concat([A, y], dim=1)
        return y

pyrstem = PyramidStem(Nlevels=3, out_channels=32, amplitude_cat=True).to(device)

with AMP_BF16():
    x = batch['stim']
    y = pyrstem(x)

print(y.shape)
#     x = rearrange(x, "B C T H W -> (B T) C H W")
#     y = lpyr(x)
#     y = [conv(L) for L in y]
#     y = [rearrange(L, "(B T) C H W -> B C T H W", B=batch_size) for L in y]
#     min_size = min([L.shape[-1] for L in y])
#     y = [chomp(L, (min_size, min_size)) for L in y]
#     y = torch.concat(y, dim=1)
#%%
from plenoptic.tools.conv import upsample_blur, blur_downsample
n_scales = 2
x = rearrange(batch['stim'], "B C T H W -> (B T) C H W")
y = []
for scale in range(n_scales - 1):
    odd = torch.as_tensor(x.shape)[2:4] % 2
    x_down = blur_downsample(x, scale_filter=False)
    x_up = upsample_blur(x_down, odd, scale_filter=False)
    y.append(x - x_up)
    x = x_down
y.append(x)

# Create animation over frames
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Calculate number of frames for animation
n_frames = batch_size - n_scales

fig, axes = plt.subplots(2, n_scales, figsize=(20, 10))
if n_scales == 1:
    axes = axes.reshape(2, 1)

# Initialize plots
images = []
for i in range(n_scales):
    # Top row: scale images
    im = axes[0, i].imshow(y[i][i, 0].detach().cpu().float(), animated=True)
    axes[0, i].set_title(f"Scale {i}")
    axes[0, i].axis('off')
    images.append(im)

    # Bottom row: differences (if applicable)
    if i < n_scales - 1:
        odd = torch.as_tensor(y[i].shape)[2:4] % 2
        P = upsample_blur(y[i+1], odd)[i + 1, 0].detach().cpu().float()
        I = y[i][i, 0].detach().cpu().float()
        im_diff = axes[1, i].imshow(I - P, animated=True)
        axes[1, i].set_title(f"Diff {i}")
        axes[1, i].axis('off')
        images.append(im_diff)
    else:
        axes[1, i].axis('off')

def update(frame):
    """Update function for animation"""
    t = frame
    img_idx = 0

    for i in range(n_scales):
        # Update top row: scale images
        I = y[i][t + i, 0].detach().cpu().float()
        images[img_idx].set_array(I)
        img_idx += 1

        # Update bottom row: differences
        if i < n_scales - 1:
            odd = torch.as_tensor(y[i].shape)[2:4] % 2
            P = upsample_blur(y[i+1], odd)[t + i + 1, 0].detach().cpu().float()
            images[img_idx].set_array(I - P)
            img_idx += 1

    return images

# Create animation
anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)

# Save as MP4
writer = FFMpegWriter(fps=20, metadata=dict(artist='VisionCore'), bitrate=1800)
anim.save('pyramid_animation.mp4', writer=writer)
print(f"Animation saved to pyramid_animation.mp4 ({n_frames} frames)")

plt.close(fig)
    

#%%
lpyr = LaplacianPyramid(3)
x = rearrange(batch['stim'], "B C T H W -> (B T) C H W")
y = lpyr(x)
y = [rearrange(L, "(B T) C H W -> B C T H W", B=batch_size) for L in y]

plt.figure(figsize=(20,20))
plot_frames(y[1].detach().cpu().float()[:100,0,[0]], nrow=20)

#%%

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

# --- helpers: upcast before view_as_complex; downcast on return ---

def _pair_to_complex(y, work_dtype=torch.float32):
    # y: [B, 2K, T, H, W] interleaved (Re, Im)
    # upcast to a supported real dtype for complex packing
    y = y.to(work_dtype)
    B, C2, T, H, W = y.shape
    assert C2 % 2 == 0, "Channel count must be even (Re/Im pairs)."
    y2 = y.view(B, C2//2, 2, T, H, W)                     # [B, K, 2, T, H, W]
    z  = torch.view_as_complex(y2.movedim(2, -1).contiguous())  # [B, K, T, H, W] complex64
    return z

def _complex_to_pair(z, out_dtype):
    # z: [B, K, T, H, W] complex
    y2 = torch.view_as_real(z)                             # [B, K, T, H, W, 2], real dtype matches z.real
    y2 = y2.movedim(-1, 2).contiguous()                    # [B, K, 2, T, H, W]
    y  = y2.view(z.size(0), z.size(1)*2, z.size(2), z.size(3), z.size(4))  # [B, 2K, T, H, W]
    return y.to(out_dtype)

def _normalize_complex(z, eps):
    return z / (torch.abs(z) + eps)


class PP(nn.Module):
    def __init__(
        self, f=17, k=16, i=1, d=1, c=0, mode='valid', branch='phase',
        epsilon=1e-10, activation="amplitude", init_scale=1, tied=True,
    ):
        super().__init__()
        W = torch.randn(k*2, i, 1, f, f) / f**2
        self.W = nn.Parameter(W * init_scale)
        self.tied = tied
        if not tied:
            V = torch.randn(k*2, i, 1, f, f) / f**2
            self.V = nn.Parameter(V * init_scale)

        self.stride = (1, d, d)
        self.space_crop = c
        if mode == 'valid':
            self.padding = (0, 0, 0)
            self.out_padding = (0, 0, 0)
        elif mode == 'same':
            self.padding = (0, f//2, f//2)
            self.out_padding = (0, d-1, d-1)

        self.branch   = branch
        self.epsilon  = float(epsilon)  # keep as float; we’ll create tensors on-the-fly with the right dtype
        self.activation = activation

    # ----- Analysis / synthesis -----
    def analysis(self, x):
        # x: [B, i, T, H, W]
        return F.conv3d(x, self.W, stride=self.stride, padding=self.padding)

    def synthesis(self, y):
        W = self.W if self.tied else self.V
        return F.conv_transpose3d(
            y, W, stride=self.stride, padding=self.padding, output_padding=self.out_padding
        )

    
    @torch.no_grad()
    def advance(self, y):
        """
        y: [B, 2K, T, H, W] (interleaved real/imag pairs)
        returns y_hat same shape; indices >=2 predicted from t-1, t
        """
        B, C, T, H, W = y.shape
        if T < 3:
            return torch.zeros_like(y)

        y_hat = torch.zeros_like(y)

        # previous & current frames for all t (keep contiguous for view ops)
        p = y[:, :, :-2].contiguous()   # [B, 2K, T-2, H, W]
        c = y[:, :, 1:-1].contiguous()  # [B, 2K, T-2, H, W]

        out_dtype = y.dtype  # likely bfloat16 under AMP

        # Do complex math in float32 regardless of AMP to satisfy view_as_complex
        with torch.autocast(device_type='cuda', enabled=False):
            if self.branch in ("phase", "phase_cmplx"):
                zp = _pair_to_complex(p, work_dtype=torch.float32)  # complex64
                zc = _pair_to_complex(c, work_dtype=torch.float32)

                eps = torch.tensor(self.epsilon, dtype=zc.real.dtype, device=zc.device)

                # delta = normalize(c) * conj(normalize(p))
                nzc   = _normalize_complex(zc, eps)
                nzp   = _normalize_complex(zp, eps)
                delta = nzc * torch.conj(nzp)

                zf = delta * zc  # predicted next complex coefficient

                y_hat[:, :, 2:] = _complex_to_pair(zf, out_dtype=out_dtype)
            else:
                raise ValueError(f"Unknown branch: {self.branch}")

        return y_hat

    # ----- The rest stays the same -----
    def forward(self, x):
        y = self.analysis(x)
        return self.nonlin(y)

    def predict(self, x):
        y     = self.analysis(x)
        y_hat = self.advance(y)
        x_hat = self.synthesis(y_hat)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def nonlin(self, y):
        # y is real-valued feature map if called here; most activations are used post-predict on complex
        return y

    def crop(self, x, x_hat, tau=2):
        H, W = x.shape[-2:]
        c = self.space_crop
        target = x[:, :, tau:, c:H-c, c:W-c]
        pred   = x_hat[:, :, tau:, c:H-c, c:W-c]
        return target, pred

# ---------- Multiscale wrapper with Laplacian pyramid ----------
class mPP(PP):
    def __init__(self, *args, J=4, **kwargs):
        super().__init__(*args, **kwargs)
        from plenoptic.simulate import LaplacianPyramid
        self.lpyr = LaplacianPyramid(J)

    def analysis_pyr(self, x):
        # x: [B, C=1, T, H, W]  -> laplacian per frame -> per-level conv3d
        B = x.size(0)
        x_btchw = rearrange(x, "B C T H W -> (B T) C H W")
        levels = self.lpyr(x_btchw)                             # list of [(B*T), C_l, H_l, W_l]
        levels_5d = [rearrange(L, "(B T) C H W -> B C T H W", B=B, T=x.size(2)) for L in levels]
        # same learned filters per scale:
        feats = [self.analysis(L) for L in levels_5d]           # each: [B, 2K, T, H_l', W_l']
        return feats

    def synthesis_pyr(self, feats):
        # feats: list of [B, 2K, T, H_l', W_l']
        recon_levels = [self.synthesis(F) for F in feats]       # list of [B, 1, T, H_l, W_l]
        B, _, T, _, _ = recon_levels[0].shape
        recon_btchw = [rearrange(R, "B C T H W -> (B T) C H W") for R in recon_levels]
        x_btchw = self.lpyr.recon_pyr(recon_btchw)              # [(B*T), 1, H, W]
        x = rearrange(x_btchw, "(B T) C H W -> B C T H W", B=B, T=T)
        return x

    def predict(self, x):
        levels = self.analysis_pyr(x)
        # vectorized advance per scale (no loops over time)
        levels_pred = [self.advance(L) for L in levels]
        x_hat = self.synthesis_pyr(levels_pred)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def to(self, device):
        new_self = super().to(device)
        new_self.lpyr = new_self.lpyr.to(device)
        return new_self


#%%
                        # convert to rates

batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

Nlevels = 3
pyr = mPP(J=Nlevels, mode='same', branch='phase', k=16).to(device)


with AMP_BF16():
    y_levels = pyr.analysis_pyr(batch['stim'])    # list of [B, 2K, T, H_l, W_l]
    y_advanced = [pyr.advance(L) for L in y_levels]
    # visualize one level
    # frames = y_levels[2].detach().cpu().float()[:10, [0], 0]  # as you did
    x_tgt, x_pred = pyr.predict(batch['stim'])


#%%

with AMP_BF16():
    y_levels = pyr.analysis_pyr(batch['stim'])    # list of [B, 2K, T, H_l, W_l]
    
    y = y_levels[0]
    # previous & current frames for all t (keep contiguous for view ops)
    p = y[:, :, :-2].contiguous()   # [B, 2K, T-2, H, W]
    c = y[:, :, 1:-1].contiguous()  # [B, 2K, T-2, H, W]

    out_dtype = y.dtype  # likely bfloat16 under AMP

    # Do complex math in float32 regardless of AMP to satisfy view_as_complex
    with torch.autocast(device_type='cuda', enabled=False):
        zp = _pair_to_complex(p, work_dtype=torch.float32)  # complex64
        zc = _pair_to_complex(c, work_dtype=torch.float32)

        eps = torch.tensor(1e-5, dtype=zc.real.dtype, device=zc.device)

        # delta = normalize(c) * conj(normalize(p))
        nzc   = _normalize_complex(zc, eps)
        nzp   = _normalize_complex(zp, eps)
        delta = nzc * torch.conj(nzp)

        zf = delta * zc  # predict
    

#%%

delta_pair = _complex_to_pair(delta, out_dtype=out_dtype)
#%%

fig_w = make_grid(y_levels[2].detach().cpu().float()[0, 0, :].unsqueeze(1), nrow=10, normalize=True)
plt.subplot(2,1,1)
plt.imshow(fig_w.permute(1, 2, 0).numpy())

plt.subplot(2,1,2)
fig_w = make_grid(delta_pair.detach().cpu().float()[0, 0, :].unsqueeze(1), nrow=10, normalize=True)
plt.imshow(fig_w.permute(1, 2, 0).numpy())

#%%
l = 0
c = y[l][:, :, 1]
p = y[l][:, :, 0]

delta = pyrConv.mult(pyrConv.norm(c), pyrConv.conj(pyrConv.norm(p)))


plt.imshow(delta[0,0].detach().cpu().float())
plt.colorbar()

#%%
plot_frames(y[2].detach().cpu().float()[:10, [0], 0], nrow=10)


# Test forward pass
model.eval()
with AMP_BF16():
    output = model(batch['stim'], dataset_id)
    print(f"Output shape: {output.shape}")


#%%
plt.imshow(y[0][0,:,:,5,5].detach().cpu().float())
# import torch.nn as nn
# loss = nn.functional.poisson_nll_loss(output, batch['robs'], log_input=False, reduction='mean')



#%%
with AMP_BF16():    
    x = model.adapters[dataset_id](batch['stim'])
    y = model.frontend(x)
    z = model.convnet(y)
    w = model.recurrent(z)
    # print(f"Convnet output shape: {z.shape}")
    # w = model.modulator(z, batch['behavior'])
print(z.shape)


#%% Run full model
with AMP_BF16():                                          # <-- new
    output = model(batch["stim"], dataset_id, batch["behavior"])         # predict log-rates

output = torch.exp(output)    

#%%
_ = plt.plot(output.detach().cpu().numpy())
# %% test all datasets
import torch.nn as nn
batch_size = 32

for dataset_id in range(len(train_datasets)):
    ntrain = len(train_datasets[f'dataset_{dataset_id}'])
    inds = np.random.randint(0, ntrain - batch_size, batch_size)

    batch = train_datasets[f'dataset_{dataset_id}'][inds]

    # dtype = torch.float32
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch['stim'], dataset_id)
        print(f"Output shape: {output.shape}")
        loss = nn.functional.poisson_nll_loss(output, batch['robs'], log_input=False, reduction='mean')
        print(f"Loss: {loss.item()}")
    
# %%