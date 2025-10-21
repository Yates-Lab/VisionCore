

#%%
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

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
from plenoptic.simulate import SteerablePyramidFreq

#%%

from models.modules.conv_layers import StandardConv, DepthwiseConv

conv = StandardConv(
    in_channels=1, out_channels=16,
    kernel_size=(1, 9, 9), padding=(0, 0, 0),
    aa_signal=True,
    aa_window='hann',
    aa_window_kwargs={'power': 1.0},
    use_weight_norm=True,
    keep_unit_norm=True,
    weight_norm_dim=0,
    bias=False)


conv.plot_weights()

#%%
from models.modules.presets import make_steerable_conv

conv = make_steerable_conv(
    in_channels=2,
    kernel_hw=17, kt=1,
    sigmas=(1.6,2.8), n_orient=8, orders=(0,1, 2),
    aa=True, wn=True, unit_norm=True,
    padding_hw=0, temporal="repeat",
    bias=False
)

conv.plot_weights()


# %%
