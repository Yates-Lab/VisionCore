"""
Example script demonstrating the PyramidSimulator class.

This script shows how to:
1. Create a PyramidSimulator with temporal filtering
2. Simulate responses from a movie
3. Visualize filters and RFs
4. Query unit properties
"""
#%%
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import torch

from mcfarland_sim import PyramidSimulator, build_temporal_kernel, get_fixrsvp_stack

#%% Create PyramidSimulator with temporal filtering

# Build temporal kernel
temporal_kernel = build_temporal_kernel(
    kernel_size=16,
    dt=1/240,
    tau_fast=0.008,
    tau_slow=0.022,
    a=0.9,
    n=2
)

# Create simulator
simulator = PyramidSimulator(
    image_shape=(51, 51),
    num_ori=8,
    num_scales=3,
    temporal_kernel=temporal_kernel
)

print(f"Created PyramidSimulator with {simulator.num_scales} scales and {simulator.num_ori} orientations")

#%% Visualize filters

fig, axes = simulator.plot_filters(
    scales=[0, 1, 2],
    orientations=[0, 2, 4, 6],
    figsize=(12, 9)
)
plt.suptitle('Pyramid Filters', fontsize=16)
plt.show()

#%% Visualize RFs with properties

fig, axes = simulator.plot_rfs(
    scales=[0, 1, 2],
    orientations=[0, 2, 4, 6],
    figsize=(12, 9)
)
plt.suptitle('Receptive Fields', fontsize=16)
plt.show()

#%% Query properties for specific units

for scale in range(3):
    for ori in [0, 4]:
        props = simulator.get_unit_properties(scale, ori)
        print(f"Scale {scale}, Ori {ori}:")
        print(f"  RF size: {props['rf_size']:.2f} pixels")
        print(f"  Preferred SF: {props['freq_rad']:.4f} cycles/pixel")
        print(f"  RF center: {props['rf_center']}")

#%% Simulate responses from a movie

# Get stimulus
full_stack = get_fixrsvp_stack(full_size=600, frames_per_im=8)
full_stack = torch.from_numpy(full_stack).float()

# Take a short segment
movie_segment = full_stack[:100, 275:326, 275:326]  # 100 frames, 51x51 pixels

# Simulate responses
responses = simulator.simulate(
    movie_segment)

print(f"\nSimulated responses shape: {responses.shape}")

#%%
# Define units to simulate (all scales and orientations at center)
mid_y, mid_x = 25, 25
units = [(scale, ori, mid_y, mid_x) 
         for scale in range(3) 
         for ori in range(8)]

#%% Plot responses for a few units

fun = lambda x: np.maximum(x, 0)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, unit in enumerate(units[:3]):
    
    ax = axes[i]
    ax.plot(fun(responses[:, unit[0], unit[1], unit[2], unit[3]]))
    ax.set_ylabel(f'Scale {scale}, Ori {ori}')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (frames)')
plt.suptitle('Simulated Responses', fontsize=16)
plt.tight_layout()
plt.show()
# %%
