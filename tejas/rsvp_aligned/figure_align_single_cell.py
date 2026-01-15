#%%
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
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml'
dataset_configs = load_dataset_configs(dataset_configs_path)

date = "2022-03-04"
subject = "Allen"
dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")

with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
    train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)


#%%
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
    stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]


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
# import os
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import warnings
# from DataYatesV1 import  get_complete_sessions
# import matplotlib.patheffects as pe

# date = "2022-03-04"
# subject = "Allen"

# dataset_configs_path = "/mnt/sata/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_for_metrics_rsvp"
# yaml_files = [
#     f for f in os.listdir(dataset_configs_path) if f.endswith(".yaml") and "base" not in f and date in f and subject in f
# ]
# dataset_configs = load_dataset_configs(yaml_files, dataset_configs_path)
# from DataYatesV1.utils.data import prepare_data
# train_dset, val_dset, dataset_config = prepare_data(dataset_configs[0])


# inds = train_dset.get_dataset_inds('fixrsvp')
# dataset = train_dset.shallow_copy()
# dataset.inds = inds

# dset_idx = inds[:,0].unique().item()
# trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
# trials = np.unique(trial_inds)

# NC = dataset.dsets[dset_idx]['robs'].shape[1]
# T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
# NT = len(trials)

# fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

# robs = np.nan*np.zeros((NT, T, NC))
# eyepos = np.nan*np.zeros((NT, T, 2))
# fix_dur =np.nan*np.zeros((NT,))

# for itrial in tqdm(range(NT)):
#     ix = trials[itrial] == trial_inds
#     ix = ix & fixation
#     if np.sum(ix) == 0:
#         continue
    
#     psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
#     fix_dur[itrial] = len(psth_inds)
#     robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
#     eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    

# good_trials = fix_dur > 20
# robs = robs[good_trials]
# eyepos = eyepos[good_trials]
# fix_dur = fix_dur[good_trials]

# ind = np.argsort(fix_dur)[::-1]
# plt.subplot(1,2,1)
# plt.imshow(eyepos[ind,:,0])
# # plt.xlim(0, 160)
# # plt.subplot(1,2,2)
# # plt.imshow(np.nanmean(robs,2)[ind])
# # plt.xlim(0, 160)
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
    
def plot_eyepos(iix, start_time, end_time, max_orientation, cc, universal_eyepos = False):
        
        time_window_len = end_time - start_time

        peak_lag = rf_contour_metrics[cc]['ste_peak_lag']
        if universal_eyepos:
            all_lags = []
            for i in range(len(rf_contour_metrics)):
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
                         path_effects=[pe.withStroke(linewidth=4, foreground='white')])
           
            total_count += 1

        plt.xlim(np.nanmin(eyepos[iix, start_time:end_time, 0]), np.nanmax(eyepos[iix, start_time:end_time, 0]))
        plt.ylim(np.nanmin(eyepos[iix, start_time:end_time, 1]), np.nanmax(eyepos[iix, start_time:end_time, 1]))

        plt.xlabel('X (degrees)')
        plt.ylabel('Y (degrees)')
        # max_x =  np.nanmax(np.abs(eyepos[iix[idx], start_time:end_time, 0]))
        # max_y =  np.nanmax(np.abs(eyepos[iix[idx], start_time:end_time, 1]))
        # max_for_plot = max(max_x, max_y)
        # plt.xlim(-max_for_plot - 0.05, max_for_plot + 0.05)
        # plt.ylim(-max_for_plot - 0.05, max_for_plot + 0.05)
        
        

        plt.title(f'{start_time} to {end_time} bins')
        plt.show()

def plot_eyepos_colormap(eyepos, iix, start_time, end_time):
    plt.imshow(eyepos[iix,start_time:end_time,0])
    plt.colorbar()
    plt.show()
    plt.imshow(eyepos[iix,start_time:end_time,1])
    plt.colorbar()
    plt.show()
def plot_robs(robs, iix, cc, num_psth = None, distances_along_line = None, alpha_raster =1):
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
        fig, axes = plt.subplots(2,2, figsize=(15, 10))
        ax1 = axes[0][0]

        ax1.imshow(robs_original[np.sort(iix),:,cids.index(cc)], alpha=alpha_raster, aspect='auto', cmap = "gray_r")
        ax1.set_title(f'{cc} before')
        ax1.set_xlabel('Time (bins)')
        ax1.set_ylabel('Trial')

        if num_psth is not None:
            # ax2 = ax1.twinx()
            ax2 = axes[1][0]
            ax2.set_xlabel('Time (bins)')
            ax2.set_ylabel('Firing Rate')

            ax2.set_yticks([])
            n_time_bins = len(robs_original[0])
            time_bins = np.arange(n_time_bins)
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
            

        ax1 = axes[0][1]
        ax1.imshow(robs[iix,:,cids.index(cc)], alpha=alpha_raster, aspect='auto', cmap = "gray_r")
        ax1.set_title(f'{cc} after')
        ax1.set_xlabel('Time (bins)')
        ax1.set_ylabel('Trial (ordered)')

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

        if num_psth is not None:
            # ax2 = ax1.twinx()
            ax2 = axes[1][1]
            ax2.set_xlabel('Time (bins)')
            ax2.set_ylabel('Firing Rate')
            ax2.set_yticks([])
            n_time_bins = len(robs[0])
            time_bins = np.arange(n_time_bins)
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
                ax2.fill_between(time_bins, y_pos - y_pos_ste, y_pos + y_pos_ste, color='r')


        plt.show()


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
        total_end_time = 200
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
                # print(f'start time {start_time} end time {end_time}')

                distances_to_use = distances_along_line

                if not trial_stitching:
                    robs_list = robs[:, total_start_time:total_end_time, :]
                    iix_list = iix

        
        # plot_robs(robs[:, start_time:end_time, :], iix, cc)
        # plot_robs(robs[:, start_time:end_time, :], iix, cc, num_psth = 4)
        plot_robs(robs_list, iix_list, cc, distances_along_line = distances_to_use, num_psth = 2)
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
                
                plot_eyepos(iix, start_time, end_time, max_orientation, cc)
                # plot_eyepos_quiver(iix, start_time, end_time, max_orientation)
                print(f'start time {start_time} end time {end_time}')

                distances_to_use = distances_along_line

                if not trial_stitching:
                    robs_list = robs[:, total_start_time:total_end_time, :]
                    iix_list = iix

        
        # plot_robs(robs[:, start_time:end_time, :], iix, cc)
        # plot_robs(robs[:, start_time:end_time, :], iix, cc, num_psth = 4)
        plot_robs(robs_list, iix_list, cc, distances_along_line = distances_to_use, num_psth=2)
        # plot_robs(robs_list, iix_list, cc, num_psth = 4)

    
    max_orientation = gratings_info['oris'][np.argmax(gratings_info['ori_tuning'][cc])]
    print(f'for max orientation {max_orientation}')
    display(max_orientation)

    # print(f'for max orientation 90')
    # display(90)

    # print(f'for max orientation max_orientation-90')
    # display(max_orientation-90)


#%%
