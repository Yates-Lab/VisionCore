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
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

from models import build_model

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda:1')
# device = torch.device('cpu')
print(f"Using device: {device}")

from DataYatesV1.utils.ipython import enable_autoreload
enable_autoreload()


#%%

config = {
    'model_type': 'v1',
    'initial_input_channels': 1,
    'adapter': {
        'type': 'adapter',
        'params': {'init_sigma': 1.0, 'grid_size': 51, 'transform': 'scale'}
    },
    'frontend': {
        'type': 'learnable_temporal',
        'params': {'kernel_size': 16, 'num_channels': 4, 'init_type': 'gaussian_derivatives',
                'split_MP': True,
                'causal': True, 'bias': False}
    },
    'convnet': {
        'type': 'densenet',
        'params': {
            'channels': [8, 16],
            'dim': 3,
            'checkpointing': False,
            'block_configs': [
                {
                    'conv_params': {
                        'type': 'standard',
                        'kernel_size': [3, 9, 9],
                        'padding': [1, 1, 1]
                    },
                    'norm_type': 'grn',
                    'act_type': 'silu',
                    'dropout': 0.0,
                    'pool_params': {
                        'type': 'max',
                        'kernel_size': 2,
                        'stride': 2
                    }
                },
                {
                    'conv_params': {
                        'type': 'standard',
                        'kernel_size': [3, 5, 5],
                        'padding': [1, 1, 1]
                    },
                    'norm_type': 'rms',
                    'act_type': 'silu',
                    'dropout': 0.0,
                    'pool_params': None
                }
            ]
        }       
    },
    'modulator': {
        'type': 'none',
        'params': {}
    },
    'recurrent': {
        'type': 'none',
        'params': {}
    },

    'readout': {
        'type': 'gaussian',
        'params': {
            'n_units': 8,
            'bias': True,
            'initial_std': 0.5,
            'initial_mean_scale': 0.1
        }
    },
    'output_activation': 'softplus' 
}



model = build_model(config).to(device)

#%% make a fake batch and run forward

batch_size = 128
H, W = 51, 51
T = 25
stim = torch.randn(batch_size, 1, T, H, W).to(device)
behavior = torch.randn(batch_size, T, 2).to(device)

with torch.no_grad():
    x = model.adapter(stim)
    x = model.frontend(x)
    x = model.convnet(x)
    print(x.shape)

with torch.no_grad():
    output = model(stim, behavior)
    print(output.shape)

# %%

_ = plt.plot(output.detach().cpu().numpy())

# %% 2D


config = {
    'model_type': 'v1',
    'initial_input_channels': 1,
    'adapter': {
        'type': 'adapter',
        'params': {'init_sigma': 1.0, 'grid_size': 51, 'transform': 'scale'}
    },
    'frontend': {
        'type': 'learnable_temporal',
        'params': {'kernel_size': 16, 'num_channels': 4, 'init_type': 'gaussian_derivatives',
                'split_MP': True,
                'causal': True, 'bias': False}
    },
    'convnet': {
        'type': 'densenet',
        'params': {
            'channels': [8, 16],
            'dim': 2,
            'checkpointing': False,
            'block_configs': [
                {
                    'conv_params': {
                        'type': 'standard',
                        'kernel_size': 9,
                        'padding': 1
                    },
                    'norm_type': 'grn',
                    'act_type': 'silu',
                    'dropout': 0.0,
                    'pool_params': {
                        'type': 'max',
                        'kernel_size': 2,
                        'stride': 2
                    }
                },
                {
                    'conv_params': {
                        'type': 'standard',
                        'kernel_size': 5,
                        'padding': 1
                    },
                    'norm_type': 'rms',
                    'act_type': 'silu',
                    'dropout': 0.0,
                    'pool_params': None
                }
            ]
        }       
    },
    'modulator': {
        'type': 'none',
        'params': {}
    },
    'recurrent': {
        'type': 'none',
        'params': {}
    },

    'readout': {
        'type': 'gaussian',
        'params': {
            'n_units': 8,
            'bias': True,
            'initial_std': 0.5,
            'initial_mean_scale': 0.1
        }
    },
    'output_activation': 'softplus' 
}



model = build_model(config).to(device)


# %%
with torch.no_grad():
    x = model.adapter(stim)
    x = model.frontend(x)
    x = model.convnet(x)
    print(x.shape)

with torch.no_grad():
    output = model(stim, behavior)
    print(output.shape)
# %%
import torch, time
torch.backends.cudnn.benchmark = True

B,T,Cin,Cout,H,W = 8, 16, 64, 128, 64, 64
kh,kw = 3,3
x = torch.randn(B, Cin, T, H, W, device='cuda', dtype=torch.float16)

w3 = torch.randn(Cout, Cin, 1, kh, kw, device='cuda', dtype=torch.float16)
w2 = w3.squeeze(2)

# Warmup
for _ in range(10):
    _ = torch.nn.functional.conv3d(x, w3, padding=(0,kh//2,kw//2))

torch.cuda.synchronize(); t0=time.time()
for _ in range(50):
    _ = torch.nn.functional.conv3d(x, w3, padding=(0,kh//2,kw//2))
torch.cuda.synchronize(); t3d=time.time()-t0

x2 = x.transpose(1,2).reshape(B*T, Cin, H, W)
for _ in range(10):
    _ = torch.nn.functional.conv2d(x2, w2, padding=(kh//2,kw//2))

torch.cuda.synchronize(); t0=time.time()
for _ in range(50):
    _ = torch.nn.functional.conv2d(x2, w2, padding=(kh//2,kw//2))
torch.cuda.synchronize(); t2d=time.time()-t0

print(f"3D: {t3d:.3f}s, 2D: {t2d:.3f}s, speedup: {t3d/t2d:.2f}x")

# %%
