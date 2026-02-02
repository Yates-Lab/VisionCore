#%%
import os
from pathlib import Path
from tkinter.constants import TRUE
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
from tejas.rsvp_util import get_fixrsvp_data
#%%
# subject = "Allen"
# date = "2022-03-02"
# dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'

# data = get_fixrsvp_data(subject, date, dataset_configs_path, 
# use_cached_data=True, 
# salvageable_mismatch_time_threshold=25, verbose=False)

# robs = data['robs']
# dfs = data['dfs']
# eyepos = data['eyepos']
# fix_dur = data['fix_dur']
# image_ids = data['image_ids']
# cids = data['cids']
    

# ind = np.argsort(fix_dur)[::-1]
# plt.subplot(1,2,1)
# plt.imshow(eyepos[ind,:,0])
# plt.xlim(0, 160)
# plt.subplot(1,2,2)
# plt.imshow(np.nanmean(robs,2)[ind])
# plt.xlim(0, 160)
all_eyepos = []
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'
for session in get_complete_sessions():
    print(session.name)
    subject = session.name.split('_')[0]
    date = session.name.split('_')[1]
    try:
        data = get_fixrsvp_data(subject, date, dataset_configs_path, 
        use_cached_data=True, 
        salvageable_mismatch_time_threshold=25, verbose=False)
        all_eyepos.append(data['eyepos'])
    except ValueError as e:
        print(f"{e}")
        continue

all_eyepos = [eyepos.reshape(-1, 2) for eyepos in all_eyepos ]
eyepos = np.concatenate(all_eyepos, axis=0)

#%% Eye position density plot (contours + crosshair + fixation circle)
# Flatten eye positions across trials/time, dropping NaNs
eyepos_flat = eyepos.reshape(-1, 2)
valid = np.all(np.isfinite(eyepos_flat), axis=1)
eyepos_flat = eyepos_flat[valid]

# Convert to arcmin (eyepos is in degrees)
eyepos_flat = eyepos_flat * 60.0

# 2D histogram in arcmin space
extent = 60  # arcmin
nbins = 61
x_edges = np.linspace(-extent, extent, nbins)
y_edges = np.linspace(-extent, extent, nbins)
if eyepos_flat.size == 0:
    H = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
else:
    H, x_edges, y_edges = np.histogram2d(
        eyepos_flat[:, 0],
        eyepos_flat[:, 1],
        bins=[x_edges, y_edges],
    )

# Smooth for contour-like appearance if scipy is available
try:
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, sigma=1.0)
except Exception:
    pass

# Normalize to max for 0-1 colorbar scale
H = H.T
H_norm = H / np.nanmax(H) if np.nanmax(H) > 0 else H

fig, ax = plt.subplots(figsize=(5, 5))
cmap = mpl.cm.get_cmap("Blues")
levels = np.linspace(0.1, 0.9, 9)

# Smooth, continuous colorbar via imshow, with contour lines on top
cf = ax.imshow(
    H_norm,
    origin="lower",
    extent=[-extent, extent, -extent, extent],
    cmap=cmap,
    vmin=0,
    vmax=0.9,
    interpolation="nearest",
)

if np.nanmax(H_norm) > 0:
    ax.contour(
        np.linspace(-extent, extent, H_norm.shape[1]),
        np.linspace(-extent, extent, H_norm.shape[0]),
        H_norm,
        levels=levels,
        colors="white",
        linewidths=1.0,
    )

# Crosshair
ax.axhline(0, color="black", linestyle=(0, (10, 8)), linewidth=2)
ax.axvline(0, color="black", linestyle=(0, (10, 8)), linewidth=2)

# Fixation circle
circle = mpl.patches.Circle(
    (0, 0), extent, fill=False, edgecolor="red",
    linestyle=(0, (14, 10)), linewidth=2
)
ax.add_patch(circle)

ax.set_xlim(-extent, extent)
ax.set_ylim(-extent, extent)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Horizontal (arcmin)")
ax.set_ylabel("Vertical position (arcmin)")

cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.06)
cb.set_label("Count (normalized)", rotation=270, labelpad=18)
cb.set_ticks(np.linspace(0, 0.8, 9))

#save figure
fig.savefig("eyepos_rsvp.pdf", dpi=1200, bbox_inches="tight")
# %%
