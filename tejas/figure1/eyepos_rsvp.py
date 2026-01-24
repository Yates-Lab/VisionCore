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
#%%
subject = "Allen"
date = "2022-03-02"
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


good_trials = fix_dur > 20
robs = robs[good_trials]
dfs = dfs[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]


ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 160)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 160)
#%%

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
