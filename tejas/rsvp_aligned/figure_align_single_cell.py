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

from tejas.rsvp_util import get_fixrsvp_data
subject = 'Allen'
date = '2022-03-04'
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'

data = get_fixrsvp_data(subject, date, dataset_configs_path, 
use_cached_data=True, 
salvageable_mismatch_time_threshold=25, verbose=True)
robs = data['robs']
dfs = data['dfs']
eyepos = data['eyepos']
fix_dur = data['fix_dur']
image_ids = data['image_ids']
cids = data['cids']
spike_times_trials = data['spike_times_trials']
trial_t_bins = data['trial_t_bins']
trial_time_windows = data['trial_time_windows']
rsvp_images = data['rsvp_images']
dataset = data['dataset']

# good_trials = fix_dur > 20
# robs = robs[good_trials]
# dfs = dfs[good_trials]
# eyepos = eyepos[good_trials]
# fix_dur = fix_dur[good_trials]


ind = np.argsort(fix_dur)[::-1]
plt.subplot(1,2,1)
plt.imshow(eyepos[ind,:,0])
plt.xlim(0, 160)
plt.subplot(1,2,2)
plt.imshow(np.nanmean(robs,2)[ind])
plt.xlim(0, 160)

#%%


#%%
from tejas.metrics.gaborium import plot_unit_sta_ste
from tejas.metrics.main_unit_panel import get_unit_info_panel_dict
unit_info_panel_dict = get_unit_info_panel_dict(date, subject, cache = True)
unit_sta_ste_dict = unit_info_panel_dict['unit_sta_ste_dict']
contour_metrics = unit_info_panel_dict['rf_contour_metrics']
gaussian_fit_metrics = unit_info_panel_dict['rf_gaussian_fit_metrics']
#%%
from tejas.metrics.gaborium import get_rf_contour_metrics
rf_contour_metrics = get_rf_contour_metrics(date, subject)
#%%

from tejas.metrics.gratings import get_gratings_for_dataset, plot_ori_tuning
gratings_info = get_gratings_for_dataset(date, subject, cache = True)
#%%
from tejas.metrics.qc import get_qc_units_for_session
units_qc = get_qc_units_for_session(date, subject, cache = True)
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
    
    peak_lag = rf_contour_metrics[cc]['ste_peak_lag']
    if universal_eyepos:
        all_lags = []
        for i in range(len(rf_contour_metrics)):
            if i in cids:
                all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
        peak_lag = np.median(all_lags).astype(int)

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
    
def plot_eyepos(iix, start_time, end_time, max_orientation, cc, universal_eyepos = False, use_bins = True):
        
        time_window_len = end_time - start_time

        peak_lag = rf_contour_metrics[cc]['ste_peak_lag']
        if universal_eyepos:
            all_lags = []
            for i in range(len(rf_contour_metrics)):
                if i in cids:
                    all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
            peak_lag = np.median(all_lags).astype(int)

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
def plot_robs(robs, iix, cc, num_psth = None, distances_along_line = None, alpha_raster =1, bins_x_axis = True, render="line", linear_distance=False, empty_row_style=None, empty_row_margin=0.1, use_spike_times=False, spike_times=None, trial_t_bins=None, dt=1/240, t_start=None, t_end=None, tick_height = 0.2, tick_linewidth = 4, debug=False):
        def _time_axis_params(n_time_bins, bins_x_axis, dt=1/240):
            max_ms = (n_time_bins - 1) * dt * 1000
            if bins_x_axis:
                time_bins = np.arange(n_time_bins)
                tick_step = 20 if n_time_bins <= 100 else 50
                tick_positions = np.arange(0, n_time_bins + 1e-9, tick_step)
                tick_labels = [f'{tick:.0f}' for tick in tick_positions]
                x_max = n_time_bins - 1
            else:
                time_bins = np.arange(n_time_bins) * dt * 1000
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
        

        def plot_raster_spike_times(ax, spike_times_list, trial_indices, height=1.0, color="k", linewidth=0.5, 
                                    y_positions=None, bins_x_axis=True, dt=1/240, trial_t_bins=None, debug=False):
            """Plot raster from spike times (no alpha - all spikes are binary).
            Uses trial_t_bins to get per-trial time windows (spike times are in absolute time)."""
            x_list, y_list = [], []
            total_spikes_before_filter = 0
            total_spikes_after_filter = 0
            for i, trial_idx in enumerate(trial_indices):
                spikes = np.atleast_1d(np.asarray(spike_times_list[trial_idx]))
                total_spikes_before_filter += spikes.size
                if spikes.size == 0:
                    continue
                
                # Get per-trial time window from trial_t_bins
                if trial_t_bins is not None:
                    t_bins = trial_t_bins[trial_idx]
                    valid_mask = ~np.isnan(t_bins)
                    if not np.any(valid_mask):
                        continue
                    valid_t_bins = t_bins[valid_mask]
                    t_start = valid_t_bins[0] - dt/2
                    t_end = valid_t_bins[-1] + dt/2
                    
                    # Filter to time window
                    mask = (spikes >= t_start) & (spikes < t_end)
                    spikes = spikes[mask] - t_start  # make relative to window start
                # else: use spikes as-is (assumes already relative times)
                
                total_spikes_after_filter += spikes.size
                if spikes.size == 0:
                    continue
                x_vals = spikes / dt if bins_x_axis else spikes * 1000  # bins or ms
                y_base = y_positions[i] if y_positions is not None else i
                for x in x_vals:
                    x_list.extend([x, x, np.nan])
                    y_list.extend([y_base, y_base + height, np.nan])
            if debug:
                print(f"[DEBUG spike_times] trials: {len(trial_indices)}, spikes before filter: {total_spikes_before_filter}, after filter: {total_spikes_after_filter}")
                if len(trial_indices) > 0 and trial_t_bins is not None:
                    sample_trial = trial_indices[0]
                    sample = np.atleast_1d(np.asarray(spike_times_list[sample_trial]))
                    t_bins = trial_t_bins[sample_trial]
                    valid_t_bins = t_bins[~np.isnan(t_bins)]
                    print(f"[DEBUG spike_times] sample trial {sample_trial}: {sample.size} spikes")
                    if sample.size > 0:
                        print(f"[DEBUG spike_times]   spike range: [{sample.min():.6f}, {sample.max():.6f}]")
                    if len(valid_t_bins) > 0:
                        print(f"[DEBUG spike_times]   t_bins range: [{valid_t_bins[0]:.6f}, {valid_t_bins[-1]:.6f}]")
            if x_list:
                ax.plot(x_list, y_list, color=color, linewidth=linewidth, rasterized=True)

        def plot_raster_as_line(ax, raster_data, time_bins, height=1.0, color="k", linewidth=0.5, alpha=1.0, y_positions=None):
            mask = np.isfinite(raster_data) & (raster_data > 0)
            row_idx, col_idx = np.where(mask)
            if row_idx.size == 0:
                return None
            values = raster_data[row_idx, col_idx]
            unique_vals = np.unique(values)
            vmin, vmax = unique_vals[0], unique_vals[-1]
            handles = []
            for val in unique_vals:
                sel = values == val
                if not np.any(sel):
                    continue
                alpha_val = min(1.0, (0.2 + 0.8 * (val - vmin) / (vmax - vmin)) * alpha) if vmax > vmin else alpha
                x_vals = time_bins[col_idx[sel]]
                y_vals = y_positions[row_idx[sel]] if y_positions is not None else row_idx[sel]
                x = np.vstack([x_vals, x_vals, np.full(sel.sum(), np.nan)])
                y = np.vstack([y_vals, y_vals + height, np.full(sel.sum(), np.nan)])
                handles.append(
                    ax.plot(x.ravel(order="F"), y.ravel(order="F"), color=color,
                            linewidth=linewidth, alpha=alpha_val, rasterized=True)[0]
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
        if num_psth is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=500)
            ax1 = axes[0][0]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=500)
            ax1 = axes[0]

        n_time_bins = robs_original.shape[1]
        time_bins, tick_positions, tick_labels, x_max = _time_axis_params(n_time_bins, bins_x_axis)
        ax1.set_rasterization_zorder(1)
        if render == "img":
            raster_extent = None
            if not bins_x_axis:
                raster_extent = (0, x_max, len(iix), 0)
            ax1.imshow(
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
            if use_spike_times and spike_times is not None:
                plot_raster_spike_times(ax1, spike_times, np.sort(iix), height=tick_height, color="k",
                                       linewidth=tick_linewidth, bins_x_axis=bins_x_axis, dt=dt, trial_t_bins=trial_t_bins, debug=debug)
            else:
                plot_raster_as_line(ax1, robs_original[np.sort(iix), :, cids.index(cc)], time_bins,
                                   height=tick_height, color="k", linewidth=tick_linewidth, alpha=alpha_raster)
            ax1.set_ylim(len(iix), 0)
        ax1.set_title(f'{cc} before')
        ax1.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
        ax1.set_ylabel('Trial')
        ax1.set_xlim(0, x_max)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)

        if num_psth is not None:
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
            

        ax1 = axes[0][1] if num_psth is not None else axes[1]
        n_time_bins = robs.shape[1]
        time_bins, tick_positions, tick_labels, x_max = _time_axis_params(n_time_bins, bins_x_axis)
        ax1.set_rasterization_zorder(1)
        
        # Linear distance mode: map trials to their actual distance positions
        use_linear = linear_distance and distances_along_line is not None
        if use_linear:
            n_rows = len(distances_along_line)
            dist_min, dist_max = distances_along_line.min(), distances_along_line.max()
            # Map distances to row indices in linear space
            y_positions = (distances_along_line - dist_min) / (dist_max - dist_min) * (n_rows - 1)
            raster_data = robs[iix, :, cids.index(cc)]
        
        if render == "img":
            raster_extent = None
            if not bins_x_axis:
                raster_extent = (0, x_max, len(iix), 0)
            ax1.imshow(
                robs[iix, :, cids.index(cc)],
                alpha=alpha_raster,
                aspect='auto',
                extent=raster_extent,
                interpolation='none',
                rasterized=True,
                zorder=0,
            )
        else:
            if use_linear:
                # Plot using appropriate method with custom y_positions
                if use_spike_times and spike_times is not None:
                    plot_raster_spike_times(ax1, spike_times, iix, height=tick_height, color="k",
                                           linewidth=tick_linewidth, y_positions=y_positions, bins_x_axis=bins_x_axis, dt=dt, trial_t_bins=trial_t_bins, debug=debug)
                else:
                    plot_raster_as_line(ax1, raster_data, time_bins, height=tick_height, color="k",
                                       linewidth=tick_linewidth, alpha=alpha_raster, y_positions=y_positions)
                ax1.set_ylim(n_rows, 0)
                
                # Visualize empty regions if requested
                if empty_row_style is not None:
                    sorted_y = np.sort(y_positions)
                    gap_threshold = 1.5  # gaps larger than this are considered empty
                    for j in range(len(sorted_y) - 1):
                        gap = sorted_y[j + 1] - sorted_y[j]
                        if gap > gap_threshold:
                            # Start after current row's spikes end, end before next row starts
                            y_start = sorted_y[j] + tick_height + empty_row_margin
                            y_end = sorted_y[j + 1] - empty_row_margin
                            if empty_row_style == "shade":
                                ax1.axhspan(y_start, y_end, color='lightblue', alpha=0.3, zorder=-1)
                            elif empty_row_style == "hatch":
                                ax1.axhspan(y_start, y_end, facecolor='none', edgecolor='gray', 
                                           hatch='///', alpha=0.5, zorder=-1)
                            elif empty_row_style == "lines":
                                ax1.axhspan(y_start, y_end, facecolor='none', edgecolor='lightblue',
                                           linewidth=0.5, zorder=-1)
                                ax1.axhline(y=(y_start + y_end) / 2, color='lightblue', 
                                           linestyle='--', linewidth=1, alpha=0.7)
            else:
                if use_spike_times and spike_times is not None:
                    plot_raster_spike_times(ax1, spike_times, iix, height=tick_height, color="k",
                                           linewidth=tick_linewidth, bins_x_axis=bins_x_axis, dt=dt, trial_t_bins=trial_t_bins, debug=debug)
                else:
                    plot_raster_as_line(ax1, robs[iix, :, cids.index(cc)], time_bins, height=tick_height,
                                       color="k", linewidth=tick_linewidth, alpha=alpha_raster)
                ax1.set_ylim(len(iix), 0)
        ax1.set_title(f'{cc} after')
        ax1.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
        ax1.set_ylabel('Trial (ordered)' if not use_linear else 'Distance along line (degrees)')
        ax1.set_xlim(0, x_max)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)

        # Add secondary y-axis with distance labels (from peak psth segment)
        if distances_along_line is not None and not use_linear:
            ax_dist = ax1.twinx()
            ax_dist.set_ylim(ax1.get_ylim())
            n_ticks = 5
            n_distances = len(distances_along_line)
            tick_indices = np.linspace(0, n_distances - 1, n_ticks, dtype=int)
            ax_dist.set_yticks(tick_indices)
            ax_dist.set_yticklabels([f'{distances_along_line[i]:.2f}' for i in tick_indices])
            ax_dist.set_ylabel('Distance along line (degrees)')
        elif use_linear:
            # Set y-axis ticks to show actual distance values
            n_ticks = 5
            tick_positions_y = np.linspace(0, len(iix) - 1, n_ticks)
            tick_labels_y = np.linspace(dist_min, dist_max, n_ticks)
            ax1.set_yticks(tick_positions_y)
            ax1.set_yticklabels([f'{v:.2f}' for v in tick_labels_y])

        if num_psth is not None:
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


cids_and_peak_times = [(14, [150]), (23, [75]), (25, [50]), (29, [75]), (36, [25]), 
(37, [75]), (42, [75]), (46, [75]), (60, [25]), (61, [25]),
(82, [75]), (92, [75]), (102, [25]), (110, [75]), (115, [75]),
(122, [100]), (128, [75]), (147, [25]), (149, [25, 50, 75, 100, 125]),
(154, [100, 200]), (158, [100, 125]), (159, [100]), (160, [100]),
(166, [50]), (169, [50, 100]), (170, [100, 175]), (173, [50, 75]), (174, [125])]
universal_eyepos = False
trial_stitching = False
for cc in [154, 122, 115, 92, 29]:
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
            distance_from_line_threshold = 0.3
            psth = np.nanmean(robs[:, :, cids.index(cc)], axis=0)
            iix, distances_along_line = get_iix_projection_on_orthogonal_line(eyepos, 
                            start_time, end_time, 
                            max_orientation, distance_from_line_threshold, cc,
                            psth = psth, universal_eyepos = universal_eyepos)

            if trial_stitching:
                robs_list.append(robs[:, start_time:end_time, :])
                iix_list.append(iix)
            


            if np.isclose(psth[start_time:end_time].max(), psth[total_start_time:total_end_time].max(), atol=1e-10) and not universal_eyepos:
                
                plot_eyepos(iix, start_time, end_time, max_orientation, cc)
                plt.show()
                # print(f'start time {start_time} end time {end_time}')

                distances_to_use = distances_along_line

                if not trial_stitching:
                    robs_list = robs[:, total_start_time:total_end_time, :]
                    iix_list = iix

        
        # plot_robs(robs[:, start_time:end_time, :], iix, cc)
        # plot_robs(robs[:, start_time:end_time, :], iix, cc, num_psth = 4)
        plot_robs(robs_list, iix_list, cc, distances_along_line = distances_to_use, num_psth = 2, render="img")
        plt.show()
        # plot_robs(robs_list, iix_list, cc, num_psth = 4)

    
    max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
    print(f'for max orientation {max_orientation}')
    display(max_orientation)

    # print(f'for max orientation 90')
    # display(90)

    # print(f'for max orientation max_orientation-90')
    # display(max_orientation-90)



#%%
universal_eyepos = True
trial_stitching = False
for cc in [115, 92]:
# for cc in [115]:

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
        total_end_time = 100
        # total_end_time = 100
        distances_to_use = None
        for i in range(total_start_time, total_end_time, len_of_each_segment):
            start_time = i
            end_time = start_time + len_of_each_segment
            #eyepos is shape [num_trial, time, 2]
            # iix = get_iix_distance_from_median_eyepos(eyepos, start_time, end_time)
            distance_from_line_threshold = 0.3
            psth = np.nanmean(robs[:, :, cids.index(cc)], axis=0)
            iix, distances_along_line = get_iix_projection_on_orthogonal_line(eyepos, 
                            start_time, end_time, 
                            max_orientation, distance_from_line_threshold, cc,
                            psth = psth, universal_eyepos = universal_eyepos)

            if trial_stitching:
            
                robs_list.append(robs[:, start_time:end_time, :])
                iix_list.append(iix)


            if np.isclose(psth[start_time:end_time].max(), psth[total_start_time:total_end_time].max(), atol=1e-10):
                
                fig, axes = plot_eyepos(iix, start_time, end_time, max_orientation, cc, universal_eyepos = universal_eyepos, use_bins = False)
                fig.savefig(f'eyepos_single_cell_aligned_{cc}.pdf', dpi=500, bbox_inches='tight')
                
                # plot_eyepos_quiver(iix, start_time, end_time, max_orientation)
                print(f'start time {start_time} end time {end_time}')

                distances_to_use = distances_along_line

                if not trial_stitching:
                    robs_list = robs[:, total_start_time:total_end_time, :]
                    iix_list = iix

        
        # plot_robs(robs[:, start_time:end_time, :], iix, cc)
        # plot_robs(robs[:, start_time:end_time, :], iix, cc, num_psth = 4)
        spike_times_cc = [spike_times_trials[t][cids.index(cc)] for t in range(len(spike_times_trials))]
        t_start = 0.0
        t_end = total_end_time / 240

        fig, axes = plot_robs(robs_list, iix_list[2:], cc, distances_along_line = distances_to_use[2:], 
                        num_psth=None, bins_x_axis = False, linear_distance = True,
                        # empty_row_style="hatch",
                        empty_row_style=None,
                        use_spike_times=True,
                        spike_times=spike_times_cc,
                        dt=1/240,
                        # tick_height = 1.5,
                        # tick_linewidth = 1,
                        tick_height = 0.6,
                        tick_linewidth = 1,
                        trial_t_bins=trial_t_bins,  # ADD THIS
                        debug=False,
                        empty_row_margin=0.0
                       )
        fig.savefig(f'raster_single_cell_aligned_{cc}.pdf', dpi=1200, bbox_inches='tight')
        plt.show()
        # plot_robs(robs_list, iix_list, cc, num_psth = 4)

        fig, ax = plot_unit_sta_ste(subject, date, 
                    cc, 
                    unit_sta_ste_dict,
                    contour_metrics = None, 
                    gaussian_fit_metrics = None, 
                    sampling_rate = None, 
                    ax = None, 
                    show_ln_energy_fit = False)
        fig.savefig(f'sta_ste_single_cell_aligned_{cc}.pdf', dpi=1200, bbox_inches='tight')
        plt.show()

        return robs_list, iix_list, distances_to_use

    
    max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
    print(f'for max orientation {max_orientation}')
    robs_list, iix_list, distances_to_use = display(max_orientation)

    # print(f'for max orientation 90')
    # display(90)

    # print(f'for max orientation max_orientation-90')
    # display(max_orientation-90)


#%%

