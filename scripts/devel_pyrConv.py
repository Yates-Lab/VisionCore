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

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_cones_120_backimage_all.yaml"
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
batch_size = 256
dataset_id = 0

ntrain = len(train_datasets[f'dataset_{dataset_id}'])
# inds = np.random.randint(0, ntrain - batch_size, batch_size)
inds = np.arange(1000, 1000+batch_size)
batch = train_datasets[f'dataset_{dataset_id}'][inds]

batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

#%%
import torch
import torch.nn.functional as F
import torch.nn as nn

class PP(nn.Module):
    """ Polar Prediction Model

    Predict by extrapolating local phases of learned convolutional tight frame

    f: int
        spatial size of filter
    k: int
        number of pairs of channels
    i: int
        number of channels in
    d: int
        downsampling factor (stride of the convolution)
    c: int
        size of spatial crop
    mode: str in ['valid', 'same']
        spatial zero padding
    branch: str in ['phase', 'phase_cmplx']
        choice of real or complex valued implementation of polar prediction,
        which are identical up to machine precision
    epsilon: float
        stability of division
    activation: str in ['linear', 'amplitude', 'amplitude_linear']
        choice of nonlinearity
    tied: boolean
        sharing analysis and synthesis weights
    """
    def __init__(
            self, f=17, k=16, i=1, d=1, c=17, mode='valid', branch='phase',
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

        if branch == 'phase':
            self.tick = self.tick_real
        elif branch == 'phase_cmplx':
            self.tick = self.tick_cmplx

        self.epsilon = torch.tensor(epsilon)
        self.activation = activation

    def forward(self, x):
        y = self.analysis(x)
        x = self.nonlin(y)
        return x

    def predict(self, x):
        y = self.analysis(x)
        y_hat = self.advance(y)
        x_hat = self.synthesis(y_hat)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def analysis(self, x):
        return F.conv3d(x, self.W, stride=self.stride, padding=self.padding)

    def synthesis(self, y):
        if self.tied:
            W = self.W
        else:
            W = self.V
        return F.conv_transpose3d(
            y, W, stride=self.stride, padding=self.padding,
            output_padding=self.out_padding
        )

    def advance(self, y):
        T = y.shape[2]
        y_hat = torch.zeros_like(y)
        for t in range(1, T-1):
            y_hat[:, :, t+1] = self.tick(y[:, :, t-1], y[:, :, t])
        return y_hat

    def tick_real(self, p, c):
        delta = self.mult(self.norm(c), self.conj(self.norm(p)))
        f = self.mult(delta, c)
        return f

    def mult(self, a, b):
        c = torch.empty_like(a, device=a.device)
        # plain
        # c[:,  ::2] = a[:,  ::2] * b[:, ::2] - a[:, 1::2] * b[:, 1::2]
        # c[:, 1::2] = a[:, 1::2] * b[:, ::2] + a[:,  ::2] * b[:, 1::2]
        # Gauss's trick
        one = a[:,  ::2] * b[:, ::2]
        two = a[:, 1::2] * b[:, 1::2]
        three = (a[:,  ::2] + a[:, 1::2]) * (b[:, ::2] + b[:, 1::2])
        c[:,  ::2] = one - two
        c[:, 1::2] = three - one - two
        return c

    def norm(self, x):
        x_ = torch.empty_like(x, device=x.device)
        n = (x[:, ::2] ** 2 + x[:, 1::2] ** 2 + self.epsilon) ** .5
        x_[:, ::2], x_[:, 1::2] = x[:, ::2] / n, x[:, 1::2] / n
        return x_

    def conj(self, x):
        x[:, 1::2] = -x[:, 1::2]
        return x

    def tick_cmplx(self, p, c):
        p = self.rect2pol(p)
        c = self.rect2pol(c)
        delta = c * p.conj() / torch.maximum(
            torch.abs(c) * torch.abs(p), self.epsilon
        )
        f = delta * c
        f = self.pol2rect(f)
        return f

    def rect2pol(self, y):
        return torch.complex(y[:, ::2], y[:, 1::2])

    def pol2rect(self, z):
        return rearrange(
            torch.stack((z.real, z.imag), dim=1), 'b c k h w-> b (k c) h w'
        )

    def nonlin(self, y):
        if self.activation is None:
            x = y
        elif self.activation == 'linear':
            x = y.real
        elif self.activation == 'amplitude':
            x = torch.abs(y)
        elif self.activation == 'amplitude_linear':
            x = torch.cat((y.real, torch.abs(y)), dim=1)
        return x

    def autoencode(self, x):
        return self.synthesis(self.analysis(x))

    def crop(self, x, x_hat, tau=2):
        """
        prediction only valid for center picture and after a warmup period
        """
        H, W = x.shape[-2:]
        c = self.space_crop
        target = x[:, :, tau:, c:H-c, c:W-c]
        pred = x_hat[:, :, tau:, c:H-c, c:W-c]
        assert target.shape == pred.shape, f"{target.shape} {pred.shape}"
        assert target.shape[-1] > 0, f"target shape {target.shape}"
        return target, pred


class mPP(PP):
    """multiscale Polar Prediction Model

    spatial filtering and temporal processing of fixed Laplacian pyramid
    coefficients, same learned filters applied at each scale
    
    J: int
        number of scales
    see documentation of PP for other arguments

    NOTE
    - explicit downsampling for speed
    """
    def __init__(
            self, f=17, k=16, i=1, d=1, c=17, mode='valid', branch='phase',
            epsilon=1e-10, activation="amplitude", init_scale=1, tied=True, J=4
        ):
        super().__init__(
            f=f, k=k, i=i, d=d, c=c, mode=mode, branch=branch, epsilon=epsilon,
            activation=activation, init_scale=init_scale, tied=tied
        )

        self.lpyr = LaplacianPyramid(J)

    def predict(self, x):
        y = self.analysis_pyr(x)
        y_hat = [self.advance(y) for y in y]
        x_hat = self.synthesis_pyr(y_hat)
        x, x_hat = self.crop(x, x_hat)
        return x, x_hat

    def analysis_pyr(self, x):
        # NOTE: the pyramid expects 4 dimensional input tensors
        B = len(x)
        x = rearrange(x, "B C T H W -> (B T) C H W")
        y = self.lpyr(x)
        y = [rearrange(y, "(B T) C H W -> B C T H W", B=B) for y in y]
        y = [self.analysis(y) for y in y]
        return y

    def synthesis_pyr(self, y):
        y = [self.synthesis(y) for y in y]
        B = len(y[0])
        y = [rearrange(y, "B C T H W -> (B T) C H W") for y in y]
        x = self.lpyr.recon_pyr(y)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def to(self, device):
        new_self = super(mPP, self).to(device)
        new_self.lpyr = new_self.lpyr.to(device)
        return new_self

# F.conv3d(y[2], W, stride=stride, padding=f//2)

#%%
                        # convert to rates

batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


Nlevels = 3
pyrConv = mPP(J=Nlevels, mode='same').to(device)

with AMP_BF16():
    y = pyrConv.analysis_pyr(batch['stim'])


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
