## !/usr/bin/env python3
"""
Clean extraction functions for cross-model analysis.

Functions to extract BPS, saccade, CCNORM, and QC data from evaluation results.
"""

#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

from DataYatesV1.exp.general import get_trial_protocols
from DataYatesV1.exp.backimage import BackImageTrial

import torch
from torchvision.utils import make_grid

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()
device = get_free_device()

#%% Load session
sessions = get_complete_sessions()
sess = sessions[0]
#%%
trial_protocols = get_trial_protocols(sess.exp)
backimage_trials = np.where(np.isin(trial_protocols, 'BackImage'))[0]

#%%
itrial = 6
trial_id = backimage_trials[itrial]
trial = BackImageTrial(sess.exp['D'][trial_id], sess.exp['S'])
I = trial.get_image()**.8
I = I[100:400, 400:700]
plt.imshow(I, cmap='gray')

#%% OLD Generation code
# #%% Get all datasets
# dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
# dataset_configs = load_dataset_configs(dataset_configs_path)

# #%% Load a dataset 
# dataset_idx = 2
# train_data, val_data, dataset_config = prepare_data(dataset_configs[dataset_idx])
# sess = train_data.dsets[0].metadata['sess']

# #%% Detect fixtions
# stim_type = 'backimage'
# stim_indices = torch.concatenate([train_data.get_dataset_inds(stim_type), val_data.get_dataset_inds(stim_type)], dim=0)
# dataset = val_data.shallow_copy()
# dataset.inds = stim_indices
# dset_idx = np.unique(stim_indices[:,0]).item()

# plt.imshow(dataset[100]['stim'][0,0], cmap='gray')
# # %%


# # %% Try loading a model and predicting
# from DataYatesV1.exp.dataset_generation import generate_fixrsvp_dataset, generate_backimage_dataset
# from scipy.interpolate import interp1d

# roi_src = dataset.dsets[dset_idx].metadata['roi_src']
# roi_src[0] = [-150, 150]
# roi_src[1] = [-150, 150]

# pix_interp = interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['dpi_pix'], kind='linear', fill_value='extrapolate', axis=0)
 
# interps = {
#     'dpi_raw': interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['dpi_raw'], kind='linear', fill_value='extrapolate', axis=0),
#     'dpi_valid': interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['dpi_valid'], kind='nearest', fill_value='extrapolate'),
#     'eyepos': interp1d(dataset.dsets[dset_idx].covariates['t_bins'], dataset.dsets[dset_idx].covariates['eyepos'], kind='linear', fill_value='extrapolate', axis=0)
# }

# #%%
# bi_dset = generate_backimage_dataset(sess.exp, sess.ks_results, roi_src, pix_interp, interps=interps, dt=1/240, metadata=dataset.dsets[dset_idx].metadata)

# #%%

# idx = 5000
# I = bi_dset[idx]['stim']

#%%
I = trial.get_image()**.8
# I = I[100:400, 400:700]
I = I[200:300, 500:600]
I = torch.from_numpy(I)
I = I - I.mean()
I = I / I.std()
plt.imshow(I, cmap='gray')
image_shape = I.shape
num_ori = 8
num_scales = 4

#%%
from plenoptic.simulate import SteerablePyramidFreq
order = num_ori-1
pyr = SteerablePyramidFreq(
                image_shape, order=order, height=num_scales, is_complex=True, 
                downsample=False, tight_frame=False)

plt.imshow(I, cmap='gray')
# %%


# %%
import itertools
import plenoptic as po
empty_image = torch.zeros((1, 1, image_shape[0], image_shape[1]), dtype=torch.float32)
pyr_coeffs = pyr.forward(empty_image)

# insert a 1 in the center of each coefficient...
for k, v in pyr.pyr_size.items():
    mid = (v[0] // 2, v[1] // 2)
    pyr_coeffs[k][..., mid[0], mid[1]] = 1

# ... and then reconstruct this dummy image to visualize the filter.
reconList = []
for scale, ori in itertools.product(range(pyr.num_scales), range(pyr.num_orientations)):
    reconList.append(pyr.recon_pyr(pyr_coeffs, [scale], [ori]))

po.imshow(reconList, col_wrap=order + 1, vrange="indep1", zoom=2);

len(reconList)
# %%
im_batch = I.unsqueeze(0).unsqueeze(0)
pyr_coeffs = pyr(im_batch)

print(pyr_coeffs.keys())
po.pyrshow(pyr_coeffs, zoom=0.5, batch_idx=0)
# %%
for i in range(num_scales):
    filter_im = reconList[num_ori*i].squeeze()
    plt.subplot(1,2,1)
    plt.imshow(filter_im, cmap='gray')
    plt.subplot(1,2,2)
    F = np.fft.fft2(filter_im)
    plt.imshow(np.fft.fftshift(np.abs(F)**2), cmap='gray')
    plt.title(i)
    plt.show()

# %%

level = 1
# make 2D hanning window
window = torch.hann_window(image_shape[0])[:, None] * torch.hann_window(image_shape[1])[None, :]


filter_im = reconList[num_ori*level].squeeze()

plt.figure()
plt.imshow(I, cmap='gray')
plt.axis("off")
plt.savefig("../figures/pyr_image.png")
plt.figure()
plt.imshow(filter_im, cmap='gray')
plt.axis("off")
plt.savefig("../figures/pyr_filter.png")

#%%

# Normalize mask to 0–1
mask = filter_im.clone()
# mask = (mask - mask.min()) / (mask.max() - mask.min())
mask = mask.abs() / mask.abs().max()
# Optional: threshold tiny values
mask[mask.abs() < 0.05] = 0

# Convert to numpy if needed
mask = mask.cpu().numpy()
f_im = filter_im.cpu().numpy()

plt.figure(figsize=(3,3))
plt.imshow(I*.5, cmap='gray', vmin=-2, vmax=2)
plt.imshow(f_im, cmap='gray', alpha=mask)
plt.axis('off')
plt.savefig("../figures/pyr_filter.pdf")

plt.figure(figsize=(3,3))
for ilevel in range(num_scales):
    r = torch.real(pyr_coeffs[(ilevel,0)].squeeze())
    i = torch.imag(pyr_coeffs[(ilevel,0)].squeeze())
    A = torch.abs(pyr_coeffs[(ilevel,0)].squeeze())
    r *= window
    i *= window
    A *= window
    mid = A.shape[0]//2
    plt.subplot(num_scales, 1, ilevel+1)
    fun = lambda x: np.maximum(x, 0)
    plt.plot(fun(r[mid,:]))
    # plt.plot(fun(i[mid,:]))
    plt.plot(A[mid,:])
plt.savefig("../figures/pyr_coeffs.pdf")

#%%


plt.subplot(1,4,1)
plt.imshow(I, cmap='gray')
plt.subplot(1,4,2)
plt.imshow(r, cmap='gray')
plt.subplot(1,4,3)
plt.imshow(i, cmap='gray')
plt.subplot(1,4,4)
plt.imshow(A, cmap='gray')
plt.show()

plt.figure()
fun = lambda x: np.maximum(x, 0)
mid = A.shape[0]//2 + 100
plt.plot(fun(r[mid,:]))
# plt.plot(fun(i[mid,:]))
plt.plot(A[mid,:])


# %%
