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
behavior_dim = 42
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
                'aa_signal': False,
                'causal': True, 'bias': False}
    },
    'convnet': {
        'type': 'densenet',
        'params': {
            'channels': [16, 32],
            'checkpointing': False,
            'stem_config': {
                'out_channels': 8,
                'conv_params':
                    {'type': 'standard',
                     'kernel_size': [1, 5, 5],
                     'padding': 0,
                     'stride': 1,
                     'dilation': 1,
                     'aa_signal': True,
                     'weight_norm_dim': 0,
                     'use_weight_norm': True,
                     'keep_unit_norm': True},
                 'norm_type': 'rms',
                 'act_type': 'splitrelu',
                 'pool_params': None,
                 'dropout': 0.0,
                 'order': ('pad', 'conv', 'norm', 'act')
                
            },
            'block_configs': [
                {
                    'conv_params': {
                        'type': 'standard',
                        'kernel_size': [3, 9, 9],
                        'aa_signal': True,
                        'aa_window': 'hann',
                        'aa_window_kwargs': {'power': 1.0},
                        'use_weight_norm': True,
                        'keep_unit_norm': True,
                        'padding': [0,0,0]
                    },
                    'norm_type': 'grn',
                    'act_type': 'splitrelu',
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
                        'use_weight_norm': True,
                        'keep_unit_norm': True,
                        'aa_signal': True,
                        'padding': [0,0,0]
                    },
                    'norm_type': 'grn',
                    'act_type': 'splitrelu',
                    'dropout': 0.0,
                    'pool_params': None
                }
            ]
        }       
    },
    'modulator': {
        'type': 'concat',
        'params': {'behavior_dim': behavior_dim, 'feature_dim': 16}
    },

    'recurrent': {
        'type': 'convgru',
        'params': {'n_layers': 1, 'hidden_dim': 128, 'kernel_size': 3}
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
behavior = torch.randn(batch_size, behavior_dim).to(device)

with torch.no_grad():
    x = model.adapter(stim)
    print(f"Adapter output shape: {x.shape}")
    x = model.frontend(x)
    print(f"Frontend output shape: {x.shape}")
    x = model.convnet(x)
    print(f"Convnet output shape: {x.shape}")
    x = model.modulator(x, behavior)
    print(f"Modulator output shape: {x.shape}")
    x = model.recurrent(x)
    print(f"Recurrent output shape: {x.shape}")
    x = model.readout(x)
    print(f"Readout output shape: {x.shape}")

with torch.no_grad():
    output = model(stim, behavior)
    print(output.shape)

# %%

_ = plt.plot(output.detach().cpu().numpy())

# %%
