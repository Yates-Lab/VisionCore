#%%
import matplotlib as mpl
# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False

# (optional) pick a clean sans‐serif
# mpl.rcParams['font.familyx'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import os
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
import warnings
from DataYatesV1 import  get_complete_sessions
import matplotlib.patheffects as pe
import contextlib


#%%
#%%
# dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'
# dataset_configs = load_dataset_configs(dataset_configs_path)

# date = "2022-03-04"
# subject = "Allen"
# dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

# with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
#     train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)


# #%%
# sess = train_dset.dsets[0].metadata['sess']
# # ppd = train_data.dsets[0].metadata['ppd']
# cids = dataset_config['cids']
# print(f"Running on {sess.name}")

# # get fixrsvp inds and make one dataaset object
# inds = torch.concatenate([
#         train_dset.get_dataset_inds('fixrsvp'),
#         val_dset.get_dataset_inds('fixrsvp')
#     ], dim=0)

# dataset = train_dset.shallow_copy()
# dataset.inds = inds

# # Getting key variables
# dset_idx = inds[:,0].unique().item()
# trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
# trials = np.unique(trial_inds)

# NC = dataset.dsets[dset_idx]['robs'].shape[1]
# T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
# NT = len(trials)

# fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

# # Loop over trials and align responses
# robs = np.nan*np.zeros((NT, T, NC))
# dfs = np.nan*np.zeros((NT, T, NC))
# eyepos = np.nan*np.zeros((NT, T, 2))
# fix_dur =np.nan*np.zeros((NT,))

# for itrial in tqdm(range(NT)):
#     # print(f"Trial {itrial}/{NT}")
#     ix = trials[itrial] == trial_inds
#     ix = ix & fixation
#     if np.sum(ix) == 0:
#         continue
    

#     psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
#     fix_dur[itrial] = len(psth_inds)
#     robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
#     dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
#     eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()


# good_trials = fix_dur > 20
# robs = robs[good_trials]
# dfs = dfs[good_trials]
# eyepos = eyepos[good_trials]
# fix_dur = fix_dur[good_trials]


# ind = np.argsort(fix_dur)[::-1]
# plt.subplot(1,2,1)
# plt.imshow(eyepos[ind,:,0])
# plt.xlim(0, 160)
# plt.subplot(1,2,2)
# plt.imshow(np.nanmean(robs,2)[ind])
# plt.xlim(0, 160)

#%%
# Load Rowley session
from DataRowleyV1V2.data.registry import get_session as get_rowley_session
from pathlib import Path
USE_ROWLEY_DATA= True
subject = 'Luke'
date = '2025-08-04'

print(f"Loading Rowley session: {subject}_{date}")
sess = get_rowley_session(subject, date)
print(f"Session loaded: {sess.name}")
print(f"Session directory: {sess.processed_path}")

# Load fixRSVP dataset
eye_calibration = 'left_eye_x-0.5_y-0.3'
dataset_type = 'fixrsvp'
dset_path = Path(sess.processed_path) / 'datasets' / eye_calibration / f'{dataset_type}.dset'

print(f"Loading dataset from: {dset_path}")
if not dset_path.exists():
    raise FileNotFoundError(f"Dataset not found: {dset_path}")

# Load using DictDataset
from DataYatesV1 import DictDataset
rowley_dset = DictDataset.load(dset_path)

print(f"Dataset loaded: {len(rowley_dset)} samples")
print(f"Response shape: {rowley_dset['robs'].shape}")

# Extract data
trial_inds = rowley_dset['trial_inds'].numpy()
trials = np.unique(trial_inds)
NC = rowley_dset['robs'].shape[1]
NT = len(trials)
cids = range(NC)

# Determine max trial length
max_T = 0
for trial in trials:
    trial_len = np.sum(trial_inds == trial)
    max_T = max(max_T, trial_len)

print(f"Number of trials: {NT}")
print(f"Number of neurons: {NC}")
print(f"Max trial length: {max_T}")

# Create trial-aligned arrays
robs = np.nan * np.zeros((NT, max_T, NC))
eyepos = np.nan * np.zeros((NT, max_T, 2))
fix_dur = np.zeros(NT)

# Define fixation criterion (eye position < 1 degree from center)
eyepos_raw = rowley_dset['eyepos'].numpy()
fixation = np.hypot(eyepos_raw[:, 0], eyepos_raw[:, 1]) < 1

print("Aligning trials...")
for itrial in tqdm(range(NT)):
    trial_mask = (trial_inds == trials[itrial]) & fixation
    if np.sum(trial_mask) == 0:
        continue
    
    trial_data = rowley_dset['robs'][trial_mask].numpy()
    trial_eye = rowley_dset['eyepos'][trial_mask].numpy()
    
    trial_len = trial_data.shape[0]
    robs[itrial, :trial_len] = trial_data
    eyepos[itrial, :trial_len] = trial_eye
    fix_dur[itrial] = trial_len

r_flat = np.nan_to_num(robs, nan=0.0).reshape(NT, -1)
e_flat = np.nan_to_num(eyepos, nan=0.0).reshape(NT, -1)
sig = np.concatenate([r_flat, e_flat], axis=1)

_, keep = np.unique(sig, axis=0, return_index=True)
keep = np.sort(keep)

robs = robs[keep]
eyepos = eyepos[keep]
fix_dur = fix_dur[keep]
NT = len(keep)
#search for duplicate trials
for itrial in range(NT):
    for jtrial in range(itrial+1, NT):
        if np.allclose(robs[itrial], robs[jtrial], equal_nan=True):
            print(f"Duplicate trial found: {itrial} and {jtrial}")
            raise ValueError("Duplicate trial found")
            assert np.allclose(eyepos[itrial], eyepos[jtrial], equal_nan=True)

# Filter for trials with sufficient duration
good_trials = fix_dur > 20
robs = robs[good_trials]
eyepos = eyepos[good_trials]
fix_dur = fix_dur[good_trials]

print(f"\nFiltered to {len(fix_dur)} trials with >20 bins")
print(f"Final robs shape: {robs.shape} (trials × time × neurons)")
print(f"Final eyepos shape: {eyepos.shape} (trials × time × XY)")

# Sort by fixation duration for visualization
ind = np.argsort(fix_dur)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(eyepos[ind, :, 0])
axes[0].set_title('Eye position X (sorted by trial length)')
axes[0].set_xlabel('Time (bins)')
axes[0].set_ylabel('Trial')
axes[1].imshow(np.nanmean(robs, 2)[ind])
axes[1].set_title('Population mean response')
axes[1].set_xlabel('Time (bins)')
plt.tight_layout()
plt.show()

#%%
rbar = np.nanmean(robs, 0)
v = np.nanvar(rbar, axis=0)

plt.plot(np.sort(v), '.')
plt.ylim(0, .01)
thresh = np.percentile(v, 90)
inds = np.where(v > thresh)[0]
n2plot = len(inds)
sx = int(np.ceil(np.sqrt(n2plot)))
sy = int(np.ceil(n2plot / sx))
fig, axes = plt.subplots(sx, sy, figsize=(12, 12))
for i in range(n2plot):
    ax = axes[i // sx, i % sx]
    ax.plot(rbar[:,inds[i]])
    ax.set_title(f'{inds[i]}')
    ax.set_xlim(0, 80)
    # axes off
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


#%%
#%%
# from tejas.metrics.gaborium import plot_unit_sta_ste
# from tejas.metrics.main_unit_panel import get_unit_info_panel_dict
# unit_info_panel_dict = get_unit_info_panel_dict(date, subject, cache = True)
# unit_sta_ste_dict = unit_info_panel_dict['unit_sta_ste_dict']
# contour_metrics = unit_info_panel_dict['rf_contour_metrics']
# gaussian_fit_metrics = unit_info_panel_dict['rf_gaussian_fit_metrics']
# #%%
# from tejas.metrics.gaborium import get_rf_contour_metrics
# rf_contour_metrics = get_rf_contour_metrics(date, subject)
# #%%

# from tejas.metrics.gratings import get_gratings_for_dataset, plot_ori_tuning
# gratings_info = get_gratings_for_dataset(date, subject, cache = True)
# #%%
# from tejas.metrics.qc import get_qc_units_for_session
# units_qc = get_qc_units_for_session(date, subject, cache = True)
#%%
def get_iix_distance_from_median_eyepos(eyepos, start_time, end_time):
    centroid_pos0 = np.nanmedian(eyepos[:, start_time:end_time, 0])
    centroid_pos1 = np.nanmedian(eyepos[:, start_time:end_time, 1])
    dist_form_centroid = np.nanmedian(np.hypot(eyepos[:, start_time:end_time, 0] - centroid_pos0, 
                    eyepos[:, start_time:end_time, 1] - centroid_pos1),1)
    iix = np.argsort(dist_form_centroid)
    return iix

def microsaccade_exists(eyepos, threshold = 0.3):
    '''
    helper function for get_iix_projection_on_orthogonal_line.

    eyepos will be of form eyepos[idx, start_time_shifted:end_time_shifted, :]
    '''
    median_eyepos = np.nanmedian(eyepos, axis=0)

    #check distance of all points from median_eyepos
    distances = np.hypot(eyepos[:, 0] - median_eyepos[0], eyepos[:, 1] - median_eyepos[1])

    return np.any(distances > threshold) 
def get_iix_projection_on_orthogonal_line(eyepos, start_time, end_time, max_orientation, distance_from_line_threshold, cc, psth, universal_eyepos = False):
    # if psth is not None:
    #     max_psth_idx = np.argmax(psth[start_time:end_time]) - rf_contour_metrics[cc]['ste_peak_lag']
    if not USE_ROWLEY_DATA:
        peak_lag = rf_contour_metrics[cc]['ste_peak_lag']
    else:
        peak_lag = 7
    if universal_eyepos:
        if not USE_ROWLEY_DATA:
            all_lags = []
            for i in range(len(rf_contour_metrics)):
                if i in cids:
                    all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
            peak_lag = np.median(all_lags).astype(int)
        else:
            peak_lag = 7

    time_window_len = end_time - start_time
    start_time_shifted = max(start_time - peak_lag, 0)
    # start_time_shifted = start_time
    end_time_shifted = start_time_shifted + time_window_len

    

    centroid_pos0 = np.nanmedian(eyepos[:, start_time_shifted:end_time_shifted, 0])
    centroid_pos1 = np.nanmedian(eyepos[:, start_time_shifted:end_time_shifted, 1])
    
    # Calculate orthogonal line parameters
    orthogonal_angle = max_orientation + 90
    slope = np.tan(np.deg2rad(orthogonal_angle))
    intercept = centroid_pos1 - slope * centroid_pos0
    
    # Get median eyepos for each trial and filter by distance to line
    valid_indices = []
    projections = []

    
    
    for idx in range(len(eyepos)):
        if np.isnan(eyepos[idx, start_time_shifted:end_time_shifted, :]).all() or np.isnan(eyepos[idx, start_time:end_time, :]).all():
            continue
        if microsaccade_exists(eyepos[idx, start_time_shifted:end_time_shifted, :]):
            continue
        median_eyepos = np.nanmedian(eyepos[idx, start_time_shifted:end_time_shifted, :], axis=0)
        if psth is not None and not universal_eyepos:
            max_psth_idx = np.argmax(psth[start_time:end_time]) - peak_lag
            
            median_eyepos = eyepos[idx, start_time:end_time, :][max_psth_idx]
        distance = np.abs(slope * median_eyepos[0] - median_eyepos[1] + intercept) / np.sqrt(1 + slope**2)
        
        if distance < distance_from_line_threshold:
            # Project point onto line: x_proj = (x0 + m*(y0 - b)) / (1 + m²)
            x_proj = (median_eyepos[0] + slope * (median_eyepos[1] - intercept)) / (1 + slope**2)
            valid_indices.append(idx)
            projections.append(x_proj)

    
    # Sort by projection x-coordinate (left to right)
    sort_order = np.argsort(projections)
    iix = np.array(valid_indices)[sort_order]
    sorted_projections = np.array(projections)[sort_order]


   
    if len(iix)==0:
        return iix, iix
    # Distance along the line from the 0th index (accounting for slope)
    distances_along_line = (sorted_projections - sorted_projections[0]) * np.sqrt(1 + slope**2)
    return iix, distances_along_line
    
def plot_eyepos(iix, start_time, end_time, max_orientation, cc, universal_eyepos = False, use_bins = True, show_all_eyepos = False):
        
        time_window_len = end_time - start_time

        if not USE_ROWLEY_DATA:
            peak_lag = rf_contour_metrics[cc]['ste_peak_lag']
        else:
            peak_lag = 7
        if universal_eyepos:

            if not USE_ROWLEY_DATA:
                all_lags = []
                for i in range(len(rf_contour_metrics)):
                    if i in cids:
                        all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
                peak_lag = np.median(all_lags).astype(int)
            else:
                peak_lag = 7

        start_time = max(start_time - peak_lag, 0)
        end_time = start_time + time_window_len

        centroid_pos0 = np.nanmedian(eyepos[:,start_time:end_time,0])
        centroid_pos1 = np.nanmedian(eyepos[:,start_time:end_time,1])

        
        #plot that is centered at centroid_pos0, centroid_pos1 and is orthogonal to max_orientation
        # Convert orientation angle to slope of perpendicular line (orthogonal = +90 degrees)
        orthogonal_angle = max_orientation + 90
        slope = np.tan(np.deg2rad(orthogonal_angle))
        length = 10

        #set figure size and aspect to equal
        # get figure and ax

        fig, axes = plt.subplots(figsize=(5, 5), dpi=500)


        plt.plot(
            [centroid_pos0 - length/2, centroid_pos0 + length/2],
            [centroid_pos1 - length/2*slope, centroid_pos1 + length/2*slope],
            'k'
        )
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(iix)))
        if show_all_eyepos:
            iix_set = set(iix.tolist() if isinstance(iix, np.ndarray) else list(iix))
            for idx in range(len(eyepos)):
                if idx in iix_set:
                    continue
                plt.plot(
                    eyepos[idx, start_time:end_time, 0],
                    eyepos[idx, start_time:end_time, 1],
                    color=(0.6, 0.6, 0.6, 0.4),
                    alpha=1,
                    linewidth=0.6
                )
        total_count = 0
        for idx in range(len(iix)):
            assert not microsaccade_exists(eyepos[iix[idx], start_time:end_time, :])
            median_eyepos = np.nanmedian(eyepos[iix[idx], start_time:end_time, :], axis=0)
            plt.plot(
                eyepos[iix[idx], start_time:end_time, 0],
                eyepos[iix[idx], start_time:end_time, 1],
                color=colors[idx],
                alpha=1,
                linewidth=0.7
            )
            # marker at median position
            plt.scatter(median_eyepos[0], median_eyepos[1], color=colors[idx], s=20, edgecolor='k', linewidth=0.7, zorder=3)
            if total_count % 5 == 0:
                
                plt.text(median_eyepos[0], median_eyepos[1], total_count, color='k', fontsize=12, ha='center', va='bottom',
                         path_effects=[pe.withStroke(linewidth=4, foreground='white')], zorder=10)
           
            total_count += 1

        # plt.xlim(np.nanmin(eyepos[iix, start_time:end_time, 0]), np.nanmax(eyepos[iix, start_time:end_time, 0]))
        # plt.ylim(np.nanmin(eyepos[iix, start_time:end_time, 1]), np.nanmax(eyepos[iix, start_time:end_time, 1]))

        plt.xlabel('X (degrees)')
        plt.ylabel('Y (degrees)')
        if show_all_eyepos:
            max_x = np.nanmax(np.abs(eyepos[:, start_time:end_time, 0]))
            max_y = np.nanmax(np.abs(eyepos[:, start_time:end_time, 1]))
        else:
            max_x =  np.nanmax(np.abs(eyepos[iix, start_time:end_time, 0]))
            max_y =  np.nanmax(np.abs(eyepos[iix, start_time:end_time, 1]))
        max_for_plot = max(max_x, max_y)
        plt.xlim(-max_for_plot - 0.05, max_for_plot + 0.05)
        plt.ylim(-max_for_plot - 0.05, max_for_plot + 0.05)
        
        
        if use_bins:
            plt.title(f'{start_time} to {end_time} bins')
        else:
            plt.title(f'{int(start_time * 1/240 *1000)} to {int(end_time * 1/240 *1000)} ms')
        
        return fig, axes

def plot_eyepos_colormap(eyepos, iix, start_time, end_time):
    plt.imshow(eyepos[iix,start_time:end_time,0])
    plt.colorbar()
    plt.show()
    plt.imshow(eyepos[iix,start_time:end_time,1])
    plt.colorbar()
    plt.show()
def plot_robs(robs, iix, cc, num_psth = None, distances_along_line = None, alpha_raster =1, bins_x_axis = True, render="line"):
        def _time_axis_params(n_time_bins, bins_x_axis):
            max_ms = (n_time_bins - 1) / 240 * 1000
            if bins_x_axis:
                time_bins = np.arange(n_time_bins)
                tick_step = 20 if n_time_bins <= 100 else 50
                tick_positions = np.arange(0, n_time_bins + 1e-9, tick_step)
                tick_labels = [f'{tick:.0f}' for tick in tick_positions]
                x_max = n_time_bins - 1
            else:
                time_bins = (np.arange(n_time_bins) / 240) * 1000
                if max_ms <= 120:
                    tick_step = 25
                elif max_ms <= 250:
                    tick_step = 50
                else:
                    tick_step = 100
                tick_positions = np.arange(0, max_ms + 1e-9, tick_step)
                tick_labels = [f'{tick:.0f}' for tick in tick_positions]
                x_max = max_ms
            return time_bins, tick_positions, tick_labels, x_max
        tick_height = 0.2
        tick_linewidth =4

        def plot_raster_as_line(ax, raster_data, time_bins, height=1.0, color="k", linewidth=0.5, alpha=1.0):
            mask = np.isfinite(raster_data) & (raster_data > 0)
            row_idx, col_idx = np.where(mask)
            if row_idx.size == 0:
                return None
            values = raster_data[row_idx, col_idx]
            unique_vals = np.unique(values)
            vmin = unique_vals[0]
            vmax = unique_vals[-1]
            handles = []
            for val in unique_vals:
                sel = values == val
                if not np.any(sel):
                    continue
                if vmax > vmin:
                    norm = (val - vmin) / (vmax - vmin)
                    alpha_val = (0.2 + 0.8 * norm) * alpha
                else:
                    alpha_val = alpha
                x_vals = time_bins[col_idx[sel]]
                x = np.vstack([x_vals, x_vals, np.full(sel.sum(), np.nan)])
                y = np.vstack([row_idx[sel], row_idx[sel] + height, np.full(sel.sum(), np.nan)])
                handles.append(
                    ax.plot(
                        x.ravel(order="F"),
                        y.ravel(order="F"),
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha_val,
                        rasterized=True,
                    )[0]
                )
            return handles[-1] if handles else None
        # Handle lists - stitch together along time axis with padding
        if isinstance(robs, list):
            max_len = max([len(iix_segment) for iix_segment in iix])
            # Pad each segment to max_len trials, then concatenate along time axis
            robs_segments_padded = []
            robs_segments_padded_original = []
            for r_seg, iix_seg in zip(robs, iix):
                if len(iix_seg) == 0:
                    robs_segments_padded.append(np.full((max_len, r_seg.shape[1], r_seg.shape[2]), np.nan))
                    robs_segments_padded_original.append(np.full((max_len, r_seg.shape[1], r_seg.shape[2]), np.nan))
                    continue
                r_selected_original = r_seg[np.sort(iix_seg)]
                r_selected = r_seg[iix_seg]
                pad_len = max_len - len(iix_seg)
                if pad_len > 0:
                    r_padded = np.concatenate([r_selected, np.full((pad_len, r_seg.shape[1], r_seg.shape[2]), np.nan)], axis=0)
                    r_padded_original = np.concatenate([r_selected_original, np.full((pad_len, r_seg.shape[1], r_seg.shape[2]), np.nan)], axis=0)
                else:
                    r_padded = r_selected
                    r_padded_original = r_selected_original
                robs_segments_padded.append(r_padded)
                robs_segments_padded_original.append(r_padded_original)
            # Concatenate along time axis (axis=1)
            robs = np.concatenate(robs_segments_padded, axis=1)
            robs_original = np.concatenate(robs_segments_padded_original, axis=1)
            iix = np.arange(max_len)
        else:
            robs_original = robs
        
        # Compute global psth scale for consistent scaling
        if num_psth is not None:
            num_indices_for_each_psth = np.ceil(len(iix) / num_psth).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                psths_all = [np.nanmean(robs[iix[i*num_indices_for_each_psth:(i+1)*num_indices_for_each_psth],:,cids.index(cc)], axis=0) for i in range(num_psth)]
                psth_scale = np.nanmax(psths_all) + 1e-10
        
        # plt.subplot(1,2,1)
        # ax1 = plt.gca()
        show_psth = num_psth is not None and num_psth > 0
        if show_psth:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=500)
            ax_before = axes[0][0]
            ax_after = axes[0][1]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=500)
            ax_before = axes[0]
            ax_after = axes[1]

        n_time_bins = robs_original.shape[1]
        time_bins, tick_positions, tick_labels, x_max = _time_axis_params(n_time_bins, bins_x_axis)
        ax_before.set_rasterization_zorder(1)
        if render == "img":
            raster_extent = None
            if not bins_x_axis:
                raster_extent = (0, x_max, len(iix), 0)
            ax_before.imshow(
                robs_original[np.sort(iix), :, cids.index(cc)],
                alpha=alpha_raster,
                aspect='auto',
                # cmap="gray_r",
                extent=raster_extent,
                interpolation='none',
                rasterized=True,
                zorder=0,
            )
        else:
            plot_raster_as_line(
                ax_before,
                robs_original[np.sort(iix), :, cids.index(cc)],
                time_bins,
                height=tick_height,
                color="k",
                linewidth=tick_linewidth,
                alpha=alpha_raster,
            )
            ax_before.set_ylim(len(iix), 0)
        ax_before.set_title(f'{cc} before')
        ax_before.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
        ax_before.set_ylabel('Trial')
        ax_before.set_xlim(0, x_max)
        ax_before.set_xticks(tick_positions)
        ax_before.set_xticklabels(tick_labels)

        if show_psth:
            # ax2 = ax1.twinx()
            ax2 = axes[1][0]
            ax2.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
            ax2.set_ylabel('Firing Rate')

            ax2.set_yticks([])
            time_bins = time_bins
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                psths = [np.nanmean(robs_original[iix,:,cids.index(cc)], axis=0) for i in range(num_psth)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                psths_ste = [np.nanstd(robs_original[iix,:,cids.index(cc)], axis=0) / np.sqrt(num_indices_for_each_psth) for i in range(num_psth)]
            psths = np.array(psths)
            for i in range(num_psth):
                y_center = (num_psth - i - 0.5) * num_indices_for_each_psth
                y_pos = y_center + psths[i] / psth_scale * num_indices_for_each_psth * 0.8
                y_pos_ste = psths_ste[i] / psth_scale * num_indices_for_each_psth * 0.8
                if i == num_psth//2:

                    ax2.plot(time_bins, y_pos, 'r-', linewidth=1.5)
                    ax2.fill_between(time_bins, y_pos - y_pos_ste, y_pos + y_pos_ste, alpha=0.5, color='r')
                else:
                    ax2.plot(time_bins, y_pos, 'b-', alpha=0, linewidth=1.5)
                    ax2.fill_between(time_bins, y_pos - y_pos_ste, y_pos + y_pos_ste, alpha=0, color='b')
            ax2.set_xlim(0, x_max)
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)
            

        ax1 = ax_after
        n_time_bins = robs.shape[1]
        time_bins, tick_positions, tick_labels, x_max = _time_axis_params(n_time_bins, bins_x_axis)
        ax1.set_rasterization_zorder(1)
        if render == "img":
            raster_extent = None
            if not bins_x_axis:
                raster_extent = (0, x_max, len(iix), 0)
            ax1.imshow(
                robs[iix, :, cids.index(cc)],
                alpha=alpha_raster,
                aspect='auto',
                # cmap="gray_r",
                extent=raster_extent,
                interpolation='none',
                rasterized=True,
                zorder=0,
            )
        else:
            plot_raster_as_line(
                ax1,
                robs[iix, :, cids.index(cc)],
                time_bins,
                height=tick_height,
                color="k",
                linewidth=tick_linewidth,
                alpha=alpha_raster,
            )
            ax1.set_ylim(len(iix), 0)
        ax1.set_title(f'{cc} after')
        ax1.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
        ax1.set_ylabel('Trial (ordered)')
        ax1.set_xlim(0, x_max)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)

        # Add secondary y-axis with distance labels (from peak psth segment)
        if distances_along_line is not None:
            ax_dist = ax1.twinx()
            ax_dist.set_ylim(ax1.get_ylim())
            # Select ~5 equally spaced indices based on distances array length
            n_ticks = 5
            n_distances = len(distances_along_line)
            tick_indices = np.linspace(0, n_distances - 1, n_ticks, dtype=int)
            ax_dist.set_yticks(tick_indices)
            ax_dist.set_yticklabels([f'{distances_along_line[i]:.2f}' for i in tick_indices])
            ax_dist.set_ylabel('Distance along line (degrees)')

        if show_psth:
            # ax2 = ax1.twinx()
            ax2 = axes[1][1]
            ax2.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
            ax2.set_ylabel('Firing Rate')
            ax2.set_yticks([])
            time_bins = time_bins
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                psths = [np.nanmean(robs[iix[i*num_indices_for_each_psth:(i+1)*num_indices_for_each_psth],:,cids.index(cc)], axis=0) for i in range(num_psth)]
                psths_ste = [np.nanstd(robs[iix[i*num_indices_for_each_psth:(i+1)*num_indices_for_each_psth],:,cids.index(cc)], axis=0) / np.sqrt(len(iix[i*num_indices_for_each_psth:(i+1)*num_indices_for_each_psth])) for i in range(num_psth)]
            psths = np.array(psths)[::-1]
            psths_ste = np.array(psths_ste)[::-1]

            
            for i in range(num_psth):
                y_center = i * num_indices_for_each_psth
                y_pos = y_center + psths[i] / psth_scale * num_indices_for_each_psth * 0.8
                y_pos_ste = psths_ste[i] / psth_scale * num_indices_for_each_psth * 0.8
                ax2.plot(time_bins, y_pos, 'r-', linewidth=1.5)
                ax2.fill_between(time_bins, y_pos - y_pos_ste, y_pos + y_pos_ste, alpha=0.5, color='r')
            ax2.set_xlim(0, x_max)
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(tick_labels)


        return fig, axes

#%%

#plot spike raster for cell 115
plt.imshow(robs[:, :, 115])
plt.show()
#%%
cells_to_orientations = {
    62: 0,
    84: 45,
    98: 0, 
    138: 0,
    184: 0,
    229: 0, 
    229: 0,
    226: 45,
    248: 0,
    241: 0,
    252: 0,
    255: 0,
    264: 0,
    269: 45,
    280: 0,
    274: 0,
    329: 135,
    314: 0, 
    300: 45,
    373: 45,
    365: 0,
    362: 45,
}

universal_eyepos = False
trial_stitching = True
# for cc in [154, 122, 115, 92, 29]:
# for cc in cells_to_orientations.keys():
for cc in [115]:

    # cc = cids.index(cc)
#nothing from Allen_2022-02-18
# 61 from Allen_2022-02-24
# 142, 51 from Allen_2022-03-02 
#154, 122, 115, 92 from Allen_2022-03-04 time range 0,250
# 132, 76, 30 from Allen_2022-03-30 time range 0,250
# 112 from Allen_2022-04-01 time range 0,250
# 179, 174, 158, 91, 9 from Allen_2022-04-06  0, 250 
#99, 77, 61 from Allen_2022-04-08 are okay (not great) 0, 250
# 49, 122 from Allen_2022-04-13
# 122, 154 from Allen_2022-04-15 time range 0,150
# 70 from Allen_2022-04-15 time range 100,200

# 154 from Allen_2022-04-15 and 158 from Allen_2022-04-06 are best

#okay:
#for line at 0 degrees, cell 62 time 0
#for line at 45 degrees, cell 84 time 0
#for line at 0 degrees, cell 98 time 0
#for line at 0 degrees, cell 138 time 0
#for line at 0 degrees, cell 184 time 0
#for line at 0 degrees, cell 229 time 0
#for line at 0 degrees, cell 228 time 0
#for line at 45 degrees, cell 226 time 0
#for line at 0 degrees, cell 248 time 0
#for line at 0 degrees, cell 241 time 0
#for line at 0 degrees, cell 252 time 0
#for line at 0 degrees, cell 255 time 0
#for line at 0 degrees, cell 264 time 0
#for line at 45 degrees, cell 269 time 0
#for line at 0 degrees, cell 280 time 0
#for line at 0 degrees, cell 274 time 0
# for line at 135 degrees, cell 329 time 0
#for line at 0 degrees, cell 314 time 0
#for line at 45 degrees, cell 300 time 0
# for line at 45 degrees, cell 373 time 0
#for line at 0 degrees, cell 365 time 0
#for line at 45 degrees, cell 362 time 0






#maybe:
#for line at 135 degrees, cell 1
#for line at 45 degrees, cell 19 time 0
#for line at 45 degrees, cell 30 time 0
#for line at 45 degrees, cell 32 time 0
#for line at 45 degrees, cell 37 time 0
#for line at 45 degrees, cell 50 time 0
#for line at 45 degrees, cell 52 time 0
# for line at 45 degrees, cell 70 time 0
#for line at 90 degree, cell 78 time 0
#for line at 45 degrees, cell 85 time 0
#for line at 45 degrees, cell 106 time 0
#for line at 90 degrees, cell 104 time 0
#for line at 45 degrees, cell 118 time 0
#for line at 45 degrees, cell 128 time 0
#for line at 90 degrees, cell 140 time 0
#for line at 0 degrees, cell 145 time 0
#for line at 0 degrees, cell 192 time 0
#for line at 0 degrees, cell 218 time 0
#for line at 0 degrees, cell 242 time 0
#for line at 0 degrees, cell 234 time 0
#for line at 0 degrees, cell 250 time 0
#for line at 0 degrees, cell 299 time 0
#for line at 0 degrees, cell 289 time 0
#for line at 0 degrees, cell 307 time 0
#for line at 45 degrees, cell 306 time 0
#for line at 0 degrees, cell 381 time 0
#for line at 0 degrees, cell 380 time 0
#for line at 0 degrees, cell 369 time 0
#for line at 0 degrees, cell 368 time 0





    def display(max_orientation):
        # plot_ori_tuning(gratings_info, cc)
        # plt.show()
        # print(np.var(gratings_info['ori_tuning'][cc]))
        # max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
        # max_orientation = 90
        # max_orientation = max_orientation - 90
        len_of_each_segment = 25
        iix_list = []
        robs_list = []
        total_start_time = 0
        total_end_time = 250
        # total_end_time = 100
        distances_to_use = None
        for i in range(total_start_time, total_end_time, len_of_each_segment):
            start_time = i
            end_time = start_time + len_of_each_segment
            #eyepos is shape [num_trial, time, 2]
            # iix = get_iix_distance_from_median_eyepos(eyepos, start_time, end_time)
            distance_from_line_threshold = 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                psth = np.nanmean(robs[:, :, cids.index(cc)], axis=0)
            iix, distances_along_line = get_iix_projection_on_orthogonal_line(eyepos, 
                            start_time, end_time, 
                            max_orientation, distance_from_line_threshold, cc,
                            psth = psth, universal_eyepos = universal_eyepos)

            if trial_stitching:
                robs_list.append(robs[:, start_time:end_time, :])
                iix_list.append(iix)
            


            if np.isclose(psth[start_time:end_time].max(), psth[total_start_time:total_end_time].max(), atol=1e-10) and not universal_eyepos:
                
                plot_eyepos(iix, start_time, end_time, max_orientation, cc, show_all_eyepos = True)
                plt.show()
                # print(f'start time {start_time} end time {end_time}')

                distances_to_use = distances_along_line

                if not trial_stitching:
                    robs_list = robs[:, total_start_time:total_end_time, :]
                    iix_list = iix

        
        # plot_robs(robs[:, start_time:end_time, :], iix, cc)
        # plot_robs(robs[:, start_time:end_time, :], iix, cc, num_psth = 4)
        plot_robs(robs_list, iix_list, cc, distances_along_line = distances_to_use, num_psth = None, render="img")
        plt.show()
        # plot_robs(robs_list, iix_list, cc, num_psth = 4)

    
    # max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
    # print(f'for max orientation {max_orientation}')
    # display(max_orientation)
    # print(f'for line at 90 degrees, cell {cc}')
    # display(0)

    # print(f'for line at 135 degrees, cell {cc}')
    # display(45)
    
    # print(f'for line at 0 degrees, cell {cc}')
    # display(90)

    # print(f'for line at 45 degrees, cell {cc}')
    # display(135)

    # display(cells_to_orientations[cc]-90)
    display(45)






#%%
# universal_eyepos = True
# trial_stitching = False
# for cc in [115, 92]:
# # for cc in [115]:

#     def display(max_orientation):
#         # plot_ori_tuning(gratings_info, cc)
#         # plt.show()
#         # print(np.var(gratings_info['ori_tuning'][cc]))
#         # max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
#         # max_orientation = 90
#         # max_orientation = max_orientation - 90
#         len_of_each_segment = 25
#         iix_list = []
#         robs_list = []
#         total_start_time = 0
#         total_end_time = 100
#         # total_end_time = 100
#         distances_to_use = None
#         for i in range(total_start_time, total_end_time, len_of_each_segment):
#             start_time = i
#             end_time = start_time + len_of_each_segment
#             #eyepos is shape [num_trial, time, 2]
#             # iix = get_iix_distance_from_median_eyepos(eyepos, start_time, end_time)
#             distance_from_line_threshold = 0.3
#             psth = np.nanmean(robs[:, :, cids.index(cc)], axis=0)
#             iix, distances_along_line = get_iix_projection_on_orthogonal_line(eyepos, 
#                             start_time, end_time, 
#                             max_orientation, distance_from_line_threshold, cc,
#                             psth = psth, universal_eyepos = universal_eyepos)

#             if trial_stitching:
            
#                 robs_list.append(robs[:, start_time:end_time, :])
#                 iix_list.append(iix)


#             if np.isclose(psth[start_time:end_time].max(), psth[total_start_time:total_end_time].max(), atol=1e-10):
                
#                 fig, axes = plot_eyepos(iix, start_time, end_time, max_orientation, cc, universal_eyepos = universal_eyepos, use_bins = False)
#                 fig.savefig(f'eyepos_single_cell_aligned_{cc}.pdf', dpi=500, bbox_inches='tight')
                
#                 # plot_eyepos_quiver(iix, start_time, end_time, max_orientation)
#                 print(f'start time {start_time} end time {end_time}')

#                 distances_to_use = distances_along_line

#                 if not trial_stitching:
#                     robs_list = robs[:, total_start_time:total_end_time, :]
#                     iix_list = iix

        
#         # plot_robs(robs[:, start_time:end_time, :], iix, cc)
#         # plot_robs(robs[:, start_time:end_time, :], iix, cc, num_psth = 4)
#         fig, axes = plot_robs(robs_list, iix_list, cc, distances_along_line = distances_to_use, num_psth=2, bins_x_axis = False)
#         fig.savefig(f'raster_single_cell_aligned_{cc}.pdf', dpi=1200, bbox_inches='tight')
#         plt.show()
#         # plot_robs(robs_list, iix_list, cc, num_psth = 4)

#         fig, ax = plot_unit_sta_ste(subject, date, 
#                     cc, 
#                     unit_sta_ste_dict,
#                     contour_metrics = None, 
#                     gaussian_fit_metrics = None, 
#                     sampling_rate = None, 
#                     ax = None, 
#                     show_ln_energy_fit = False)
#         fig.savefig(f'sta_ste_single_cell_aligned_{cc}.pdf', dpi=1200, bbox_inches='tight')
#         plt.show()

#         return robs_list, iix_list, distances_to_use

    
#     max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
#     print(f'for max orientation {max_orientation}')
#     robs_list, iix_list, distances_to_use = display(max_orientation)

#     # print(f'for max orientation 90')
#     # display(90)

#     # print(f'for max orientation max_orientation-90')
#     # display(max_orientation-90)


#%%

