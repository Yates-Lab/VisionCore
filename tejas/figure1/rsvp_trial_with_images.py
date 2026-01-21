#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


from models.config_loader import load_dataset_configs
from models.data import prepare_data
import contextlib
import torch
from tqdm import tqdm

import os

import matplotlib as mpl
# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']



from scripts.mcfarland_sim import get_fixrsvp_stack
TIME_WINDOW = 240

#%%
ims = get_fixrsvp_stack(frames_per_im=1)
#%%
ims.shape
plt.imshow(ims[0], cmap='gray')
plt.show()
#%%
# def show_perspective_stacked_images(
#     images,
#     skip_every_n_images=5,
#     spacing=1.2,
#     elev=12,
#     azim=-70,
# ):
#     '''
#     images: list of images to stack (num_images, height, width)
#     '''
#     images = np.asarray(images)
#     if images.ndim < 3:
#         raise ValueError('images should be (num_images, height, width)')

#     num_images = images.shape[0]
#     height = images.shape[-2]
#     width = images.shape[-1]

#     fig_width = 10
#     fig_height = 6
#     fig_dpi = 100
#     fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
#     ax = fig.add_subplot(111, projection='3d')

#     yy, xx = np.mgrid[0:height, 0:width]
#     x = xx / float(width)
#     y = yy / float(height)

#     cmap = plt.get_cmap('gray')

#     z_vals = []
#     for idx, img in enumerate(images):
#         if idx % skip_every_n_images != 0:
#             continue
#         z_val = (idx / max(1, num_images - 1)) * spacing
#         z_vals.append(z_val)
#         z = np.full_like(x, z_val, dtype=float)
#         img_min = np.nanmin(img)
#         img_max = np.nanmax(img)
#         denom = max(img_max - img_min, 1e-6)
#         img_norm = (img - img_min) / denom
#         img_norm = np.nan_to_num(img_norm, nan=0.5, posinf=1.0, neginf=0.0)
#         img_norm = np.clip(img_norm, 0.0, 1.0)
#         # plot_surface expects facecolors for each quad (H-1, W-1, 4)
#         facecolors = cmap(img_norm[:-1, :-1])
#         ax.plot_surface(
#             x,
#             y,
#             z,
#             rstride=1,
#             cstride=1,
#             facecolors=facecolors,
#             shade=False,
#             linewidth=0.0,
#             edgecolor='none',
#             antialiased=False,
#         )

#     ax.view_init(elev=elev, azim=azim)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     max_z = max(z_vals) if z_vals else spacing
#     ax.set_zlim(0, max_z)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.set_xlabel('Time')
#     ax.grid(False)
#     ax.set_box_aspect((2.5, 1, 0.2))

#     for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
#         axis.pane.fill = False
#         axis.pane.set_edgecolor('w')

#     return fig, ax
#%%
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize


def imshow3d(ax, array, value_direction='z', pos=0, norm=None, cmap=None):
    """
    Display a 2D array as a  color-coded 2D image embedded in 3d.

    The image will be in a plane perpendicular to the coordinate axis *value_direction*.

    Parameters
    ----------
    ax : Axes3D
        The 3D Axes to plot into.
    array : 2D numpy array
        The image values.
    value_direction : {'x', 'y', 'z'}
        The axis normal to the image plane.
    pos : float
        The numeric value on the *value_direction* axis at which the image plane is
        located.
    norm : `~matplotlib.colors.Normalize`, default: Normalize
        The normalization method used to scale scalar data. See `imshow()`.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar data
        to colors.
    """
    if norm is None:
        norm = Normalize()
    colors = plt.get_cmap(cmap)(norm(array))

    if value_direction == 'x':
        nz, ny = array.shape
        zi, yi = np.mgrid[0:nz + 1, 0:ny + 1]
        xi = np.full_like(yi, pos)
    elif value_direction == 'y':
        nx, nz = array.shape
        xi, zi = np.mgrid[0:nx + 1, 0:nz + 1]
        yi = np.full_like(zi, pos)
    elif value_direction == 'z':
        ny, nx = array.shape
        yi, xi = np.mgrid[0:ny + 1, 0:nx + 1]
        zi = np.full_like(xi, pos)
    else:
        raise ValueError(f"Invalid value_direction: {value_direction!r}")
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False)

def show_perspective_stacked_images(
    images,
    skip_every_n_images=5,
    spacing=18,
    show_ellipsis=False,
    ellipsis_size=16,
    ellipsis_offset=0.2,
    border_size=2,
):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #set aspect ratio to 1
    ax.set_aspect('auto')
    #set fig size
    fig.set_size_inches(20, 6)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    y_positions = []
    shown_indices = []
    for i in range(len(images)):
        if i % skip_every_n_images != 0:
            continue
        pos = i * spacing
        y_positions.append(pos)
        shown_indices.append(i)
        img = images[i].copy()
        if border_size > 0:
            border_value = img.min()
            bs = int(border_size)
            img[:bs, :] = border_value
            img[-bs:, :] = border_value
            img[:, :bs] = border_value
            img[:, -bs:] = border_value
        imshow3d(ax, img.T[:, ::-1], value_direction='y', pos=pos, cmap='gray')
    if show_ellipsis and skip_every_n_images > 1 and len(shown_indices) > 1:
        x_center = images.shape[2] * 0.5
        z_center = images.shape[1] * 0.5
        for left, right in zip(shown_indices[:-1], shown_indices[1:]):
            mid_pos = ((left + right) / 2.0) * spacing
            text_pos = mid_pos - (spacing * ellipsis_offset)
            ax.text(
                x_center,
                text_pos,
                z_center,
                '...',
                color='black',
                ha='center',
                va='center',
                fontsize=ellipsis_size,
                zorder=10,
            )
    if y_positions:
        max_pos = max(y_positions)
        ax.set_ylim(0, max_pos)
        img_height = images.shape[1]
        ax.set_box_aspect((images.shape[2], max_pos if max_pos > 0 else 1, img_height))
    ax.view_init(elev=0, azim=-45, roll=0)

    #remove all axes, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    #remove all grid lines
    ax.grid(False)
    # remove 3d box/frame lines
    ax.set_axis_off()

    return fig, ax


center_crop_percent = 0.4
y_crop_start = int(ims.shape[1] * center_crop_percent)
y_crop_end = int(ims.shape[1] * (1 - center_crop_percent))
x_crop_start = int(ims.shape[2] * center_crop_percent)
x_crop_end = int(ims.shape[2] * (1 - center_crop_percent))
im_cropped = ims[:, y_crop_start:y_crop_end, x_crop_start:x_crop_end]

num_images_to_show = int(TIME_WINDOW * (1/240)/(1/30))

fig, ax = show_perspective_stacked_images(im_cropped[:num_images_to_show], skip_every_n_images=5, show_ellipsis=False)

fig.savefig("perspective_stack.pdf", dpi=500, bbox_inches="tight")
#%%
#%%

center_crop_percent = 0.4
y_crop_start = int(ims.shape[1] * center_crop_percent)
y_crop_end = int(ims.shape[1] * (1 - center_crop_percent))
x_crop_start = int(ims.shape[2] * center_crop_percent)
x_crop_end = int(ims.shape[2] * (1 - center_crop_percent))
im_cropped = ims[:, y_crop_start:y_crop_end, x_crop_start:x_crop_end]
num_images_to_show = int(TIME_WINDOW * (1/240)/(1/30))
fig, _ = show_perspective_stacked_images(im_cropped[:num_images_to_show], skip_every_n_images=5)
fig.savefig("perspective_stack.pdf", dpi=500, bbox_inches="tight")
plt.show()
#%%

dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'
dataset_configs = load_dataset_configs(dataset_configs_path)

date = "2022-03-04"
subject = "Allen"
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


#%%
trial_idx = 37
fig_width = 20
fig_height = 10
fig_dpi = 500
plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)
plt.imshow(robs[trial_idx].T,  interpolation="nearest",
            cmap="gray_r",
            aspect="auto",)
# plt.xticks([])
# plt.yticks([])
plt.xlim(0, TIME_WINDOW)
plt.savefig(f"rsvp_trial.pdf", dpi=fig_dpi, bbox_inches="tight")


plt.show()