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

#jake plot the line instead of the points
#declan don't raster until after going into illustrator




#%%
def microsaccade_exists(eyepos, threshold = 0.3):
    '''
    helper function for get_iix_projection_on_orthogonal_line.

    eyepos will be of form eyepos[idx, start_time_shifted:end_time_shifted, :]
    '''
    median_eyepos = np.nanmedian(eyepos, axis=0)

    #check distance of all points from median_eyepos
    distances = np.hypot(eyepos[:, 0] - median_eyepos[0], eyepos[:, 1] - median_eyepos[1])

    return np.any(distances > threshold) 



def get_eyepos_clusters(eyepos, start_time, end_time, robs, sort_by_cluster_psth = False, 
    max_distance_from_centroid = 0.1, num_clusters = 2, min_cluster_size = 3, cluster_size =  None,
    distance_between_centroids = (-np.inf, np.inf), min_distance_between_inter_cluster_points = 0,
    return_top_k_combos = 1, dedupe= True):
        
    # assert cluster_size is not None or min_cluster_size is not None
    # assert not (cluster_size is not None and min_cluster_size is not None), "Only one of cluster_size or min_cluster_size can be provided"
    valid_indices = []

    
    robs_start_end = robs[:, start_time:end_time, :]

    if cluster_size is not None: assert cluster_size == min_cluster_size, "cluster_size must be equal to min_cluster_size"

    all_lags = []
    for i in range(len(rf_contour_metrics)):
        all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
    peak_lag = np.median(all_lags).astype(int)

    time_window_len = end_time - start_time
    start_time = max(start_time - peak_lag, 0)
    # start_time_shifted = start_time
    end_time = start_time + time_window_len
    
    for idx in range(len(eyepos)):
        if np.isnan(eyepos[idx, start_time:end_time, :]).all():
            continue
        if np.isnan(robs_start_end[idx]).sum() > len(robs_start_end[idx])//2:
            continue
        if microsaccade_exists(eyepos[idx, start_time:end_time, :], threshold = 0.1):
            continue
        valid_indices.append(idx)
    iix = np.array(valid_indices)
    
    if len(iix) < num_clusters * min_cluster_size:
        # Always return lists for consistency
        return [iix], [np.full(len(iix), -1)]  # Not enough points

    # Precompute per-trial summaries for fast PSTH scoring (exact NaN-aware mean over trials*cells)
    # robs_start_end[iix] has shape: [n_trials, time, cells]
    robs_iix = robs_start_end[iix]
    trial_sum_tc = np.nansum(robs_iix, axis=2)                 # [n_trials, time]
    trial_count_tc = np.sum(~np.isnan(robs_iix), axis=2)       # [n_trials, time]
    trial_sum = np.nansum(trial_sum_tc, axis=1)                # [n_trials]
    
    # Compute median positions for each valid trial
    medians = np.array([np.nanmedian(eyepos[i, start_time:end_time, :], axis=0) for i in iix])
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    pairwise_dist = cdist(medians, medians)

    # Precompute candidate pools (within max_distance_from_centroid) for each potential centroid.
    # This avoids repeatedly scanning all points in Python inside the centroid-combo loop.
    within_radius = [np.flatnonzero(pairwise_dist[i] <= max_distance_from_centroid) for i in range(len(medians))]
    
    # 1. Find all valid clusters (each point as potential centroid)
    valid_clusters = []
    for i in range(len(medians)):
        cluster_members = [j for j in range(len(medians)) if pairwise_dist[i, j] <= max_distance_from_centroid]
        if len(cluster_members) >= min_cluster_size:
            valid_clusters.append((i, cluster_members))  # (centroid_idx, member_indices)
    
    if len(valid_clusters) < num_clusters:
        return [iix], [np.full(len(iix), -1)]  # Not enough valid clusters
    
    # 2. Find best combination of num_clusters clusters (maximize sum of pairwise distances)
    from itertools import combinations
    # Track top-K solutions as (score, combo, members)
    top_solutions = []
    best_by_key = {}  # partition_key -> (score, combo, members)
    
    # Cache for population response differences to avoid recomputation
    pop_diff_cache = {}

    # Helper function to compute population response difference
    def get_population_response_difference(c1_members, c2_members, method='psth_diff'):
        # Canonicalize cache key (order-independent)
        # Sort only if needed (combinations() already yields sorted tuples)
        def _members_key(members):
            # members can be np.array, list, or tuple
            if isinstance(members, np.ndarray):
                m_list = members.tolist()
            else:
                m_list = list(members)
            # Ensure ints (stable key)
            m_list = [int(x) for x in m_list]
            # Fast path: already sorted (common case)
            if len(m_list) < 2 or all(m_list[i] <= m_list[i + 1] for i in range(len(m_list) - 1)):
                return tuple(m_list)
            # Fallback: sort to make permutation-invariant
            return tuple(sorted(m_list))

        c1_key = _members_key(c1_members)
        c2_key = _members_key(c2_members)
        if c2_key < c1_key:
            c1_key, c2_key = c2_key, c1_key
        cache_key = (method, c1_key, c2_key)
        if cache_key in pop_diff_cache:
            return pop_diff_cache[cache_key]

        def _psth_from_members(members):
            # Exact equivalent of: np.nanmean(robs_start_end[iix[members]], axis=(0, 2))
            members = np.asarray(members, dtype=int)
            sum_tc = np.nansum(trial_sum_tc[members], axis=0)
            cnt_tc = np.nansum(trial_count_tc[members], axis=0)
            return np.where(cnt_tc > 0, sum_tc / cnt_tc, np.nan)

        psth1 = _psth_from_members(c1_members)
        psth2 = _psth_from_members(c2_members)

        if np.isnan(psth1).sum() > len(psth1)//2 or np.isnan(psth2).sum() > len(psth2)//2:
            return 0
        
        if method == 'psth_diff':
            val = np.linalg.norm(psth1 - psth2)
            # return np.linalg.norm(np.nanmean(cluster1_data, axis=(0)) - np.nanmean(cluster2_data, axis=(0)))
        elif method == 'cell_diff':
            # Fall back to slower path (uses full time×cell means)
            c1_members = np.asarray(c1_members, dtype=int)
            c2_members = np.asarray(c2_members, dtype=int)
            cluster1_data = robs_start_end[iix[c1_members]]
            cluster2_data = robs_start_end[iix[c2_members]]
            # Option B: per-cell unit-norm of each cluster's mean timecourse, then Frobenius norm.
            # cluster*_data: [trials, time, cells] -> mean_tc_*: [time, cells]
            mean_tc_1 = np.nanmean(cluster1_data, axis=0)
            mean_tc_2 = np.nanmean(cluster2_data, axis=0)
            # Normalize each cell's timecourse to unit L2 norm (over time)
            norm1 = np.sqrt(np.nansum(mean_tc_1**2, axis=0)) + 1e-10
            norm2 = np.sqrt(np.nansum(mean_tc_2**2, axis=0)) + 1e-10
            mean_tc_1 = mean_tc_1 / norm1
            mean_tc_2 = mean_tc_2 / norm2
            val = np.linalg.norm(mean_tc_1 - mean_tc_2)
        elif method == 'psth_diff_normed':
            val = np.linalg.norm(psth1 - psth2) / (np.linalg.norm(psth1) + np.linalg.norm(psth2) + 1e-10)
        elif method == 'sum':
            # Exact and fast: sum over trials and cells (NaNs already removed in trial_sum)
            c1_members = np.asarray(c1_members, dtype=int)
            c2_members = np.asarray(c2_members, dtype=int)
            val = np.abs(np.nansum(trial_sum[c1_members]) - np.nansum(trial_sum[c2_members]))
        elif method == 'variance_weighted':
            # Per-trial population mean over cells (NaN-aware)
            c1_members = np.asarray(c1_members, dtype=int)
            c2_members = np.asarray(c2_members, dtype=int)
            pop_mean_tc = np.where(trial_count_tc > 0, trial_sum_tc / trial_count_tc, np.nan)
            std1 = np.nanstd(pop_mean_tc[c1_members], axis=0)
            std2 = np.nanstd(pop_mean_tc[c2_members], axis=0)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2) + 1e-10
            val = np.nansum(np.abs(psth1 - psth2) / pooled_std)
        elif method == 'f_ratio':
            grand_mean = (psth1 + psth2) / 2
            between_var = np.nansum((psth1 - grand_mean)**2 + (psth2 - grand_mean)**2)
            c1_members = np.asarray(c1_members, dtype=int)
            c2_members = np.asarray(c2_members, dtype=int)
            pop_mean_tc = np.where(trial_count_tc > 0, trial_sum_tc / trial_count_tc, np.nan)
            var1 = np.nanvar(pop_mean_tc[c1_members], axis=0)
            var2 = np.nanvar(pop_mean_tc[c2_members], axis=0)
            within_var = np.nansum(var1 + var2) + 1e-10
            val = between_var / within_var
        elif method == 'temporal_decorr':
            valid = ~(np.isnan(psth1) | np.isnan(psth2))
            if valid.sum() < 3:
                pop_diff_cache[cache_key] = 0
                return 0
            corr = np.corrcoef(psth1[valid], psth2[valid])[0, 1]
            val = 1 - corr if not np.isnan(corr) else 0
        elif method == 'peak_diff':
            val = np.nanmax(np.abs(psth1 - psth2))
        else:
            raise ValueError(f'Invalid method: {method}')
            val = 0

        pop_diff_cache[cache_key] = val
        return val

    from tqdm import tqdm
    from math import comb
    n_outer = comb(len(valid_clusters), num_clusters)

    def _partition_key(members):
        """Canonical, order-invariant key for a partition (dedupe (A,B) vs (B,A) and permutations)."""
        member_keys = []
        for mem in members:
            if isinstance(mem, np.ndarray):
                mem_list = mem.tolist()
            else:
                mem_list = list(mem)
            member_keys.append(tuple(sorted(int(x) for x in mem_list)))
        member_keys.sort()
        return tuple(member_keys)

    def _maybe_add_solution(score, combo, members):
        # members: list-like of length num_clusters, each element is iterable of indices into `medians`
        if not dedupe: 
            top_solutions.append((score, combo, members))
        else:
            k = _partition_key(members)
            prev = best_by_key.get(k)
            if prev is None or score > prev[0]:
                best_by_key[k] = (score, combo, members)

            # Rebuild top_solutions (K is small; simple sort is fine)
            top_solutions.clear()
            top_solutions.extend(best_by_key.values())
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        if len(top_solutions) > return_top_k_combos:
            del top_solutions[return_top_k_combos:]
        if dedupe:
            # Trim best_by_key to only keep keys present in the current top-K (prevents unbounded growth)
            keep_keys = set(_partition_key(m) for _, __, m in top_solutions)
            for kk in list(best_by_key.keys()):
                if kk not in keep_keys:
                    del best_by_key[kk]

    for combo in tqdm(combinations(range(len(valid_clusters)), num_clusters), total=n_outer, desc="Centroid combos", leave=False, position=0):
        centroid_indices = [valid_clusters[c][0] for c in combo]
        
        # Check distance between centroids constraint first
        valid_distance = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                dist = pairwise_dist[centroid_indices[i], centroid_indices[j]]
                if dist > distance_between_centroids[1] or dist < distance_between_centroids[0]:
                    valid_distance = False
                    break
            if not valid_distance:
                break
        if not valid_distance:
            continue

        # Get candidate pools for each centroid (points within max_distance_from_centroid)
        candidate_pools = []
        for c in combo:
            centroid_idx = valid_clusters[c][0]
            # Previous (kept for reference):
            # candidates = np.array([idx for idx in range(len(medians))
            #                        if pairwise_dist[centroid_idx, idx] <= max_distance_from_centroid])
            candidates = within_radius[centroid_idx]
            candidate_pools.append(candidates)
        
        # Check if we have enough candidates
        min_required = cluster_size if cluster_size is not None else min_cluster_size
        if any(len(pool) < min_required for pool in candidate_pools):
            continue

        if cluster_size is not None and sort_by_cluster_psth:
            # Exhaustive search: enumerate all combinations of cluster_size from each pool
            from tqdm import tqdm
            from math import comb
            combo_best_score = -np.inf
            combo_best_members = None
            
            # Calculate total combinations for progress bar
            n_c1 = comb(len(candidate_pools[0]), cluster_size)
            n_c2 = comb(len(candidate_pools[1]), cluster_size)
            total_combos = n_c1 * n_c2
            
            combo_iter = ((c1, c2) for c1 in combinations(candidate_pools[0], cluster_size) 
                                   for c2 in combinations(candidate_pools[1], cluster_size))
            
            for c1_members, c2_members in tqdm(combo_iter, total=total_combos, desc="Member combos", leave=False, position=1):
                # Check for overlap (O(k) set check; k = cluster_size)
                c1_set = set(c1_members)
                if any(x in c1_set for x in c2_members):
                    continue
                # Check minimum inter-cluster distance between any pair of points (using pairwise_dist over medians)
                if min_distance_between_inter_cluster_points is not None:
                    if pairwise_dist[np.ix_(c1_members, c2_members)].min() < min_distance_between_inter_cluster_points:
                        continue
                
                # Compute PSTH score
                score = get_population_response_difference(c1_members, c2_members)
                
                if score > combo_best_score:
                    combo_best_score = score
                    combo_best_members = [c1_members, c2_members]
            
            if combo_best_members is not None:
                _maybe_add_solution(combo_best_score, combo, combo_best_members)
        else:
            # Original behavior: use closest points or all points within distance
            score = 0
            current_members = []
            
            for i in range(len(combo)):
                if cluster_size is not None:
                    c_centroid = valid_clusters[combo[i]][0]
                    members = np.argsort(pairwise_dist[c_centroid])[:cluster_size]
                else:
                    members = np.array(valid_clusters[combo[i]][1])
                current_members.append(members)
            
            # Check for overlap between all pairs
            has_overlap = False
            member_sets = [set(m.tolist()) for m in current_members]
            for i in range(len(member_sets)):
                for j in range(i + 1, len(member_sets)):
                    if member_sets[i].intersection(member_sets[j]):
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            if has_overlap:
                continue

            # Check minimum inter-cluster distance between any pair of points (using pairwise_dist over medians)
            if min_distance_between_inter_cluster_points is not None:
                too_close = False
                for i in range(len(current_members)):
                    for j in range(i + 1, len(current_members)):
                        if pairwise_dist[np.ix_(current_members[i], current_members[j])].min() < min_distance_between_inter_cluster_points:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    continue
            
            # Compute score
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if not sort_by_cluster_psth:
                        score += pairwise_dist[centroid_indices[i], centroid_indices[j]]
                    else:
                        score += get_population_response_difference(current_members[i], current_members[j])
            
            _maybe_add_solution(score, combo, current_members)
    
    # 3. Build clusters for top-K solutions
    if len(top_solutions) == 0:
        return [iix], [np.full(len(iix), -1)]

    # Sort descending (already maintained), but ensure deterministic ordering
    top_solutions.sort(key=lambda x: x[0], reverse=True)

    iix_out = []
    clusters_out = []

    for score, combo, members in top_solutions:
        clusters = np.full(len(medians), -1)
        for c_idx, mem in enumerate(members):
            mem = np.asarray(mem, dtype=int)
            clusters[mem] = c_idx
   
        if (np.unique_counts(clusters).counts < min_cluster_size).any():
            raise ValueError(f'Not enough clusters with size {min_cluster_size}')
            return [iix], [np.full(len(iix), -1)]
        
        # Reorder clusters by total spike sum (cluster 0 = smallest sum)
        cluster_sums = []
        for c_idx in range(num_clusters):
            cluster_trial_indices = iix[clusters == c_idx]
            cluster_sums.append(np.nansum(robs_start_end[cluster_trial_indices]))
        
        sorted_order = np.argsort(cluster_sums)  # Indices that would sort by sum
        cluster_mapping = {old: new for new, old in enumerate(sorted_order)}
        clusters = np.array([cluster_mapping.get(c, -1) for c in clusters])

        iix_out.append(iix)
        clusters_out.append(clusters)

    # Always return lists for consistency
    return iix_out, clusters_out
#%%
def plot_eyepos_clusters(
    eyepos,
    iix,
    start_time,
    end_time,
    clusters=None,
    show=True,
    show_unclustered_points=True,
    plot_time_traces=False,
    bins_x_axis=False,
    use_peak_lag=True
):
   
    all_lags = []
    for i in range(len(rf_contour_metrics)):
        all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
    if use_peak_lag:
        peak_lag = np.median(all_lags).astype(int)
    else:
        peak_lag = 0

    # Handle list input (treat len-1 list like single plot)
    is_list_input = isinstance(iix, list)
    if is_list_input and len(iix) == 1:
        iix = iix[0]
        start_time = start_time[0]
        end_time = end_time[0]
        clusters = clusters[0] if clusters is not None else None
        is_list_input = False
    
    if is_list_input:
        iix_list = iix
        start_time_list = start_time
        end_time_list = end_time
        clusters_list = clusters if clusters is not None else [None] * len(iix_list)
    else:
        # Single plot
        time_window_len = end_time - start_time
        start_time = max(start_time - peak_lag, 0)# - 30
        end_time = start_time + time_window_len #+ 100
        
        if plot_time_traces:
            fig, axes = plt.subplots(2, 1, sharex=True)
            ax_x, ax_y = axes
        else:
            fig, ax = plt.subplots()
        if clusters is None:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(iix)))
        else:
            num_clusters = len(set(clusters[clusters >= 0]))
            cluster_colors = plt.cm.coolwarm(np.linspace(0, 1, max(num_clusters, 1)))
            colors = [cluster_colors[c] if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters]
        
        for idx in range(len(iix)):
            if clusters is not None and not show_unclustered_points and clusters[idx] < 0:
                continue
            # assert not microsaccade_exists(eyepos[iix[idx], start_time:end_time, :], threshold=0.1)
            if plot_time_traces:
                if bins_x_axis:
                    t_vals = np.arange(start_time, end_time)
                else:
                    t_vals = (np.arange(start_time, end_time) / 240) * 1000
                ax_x.plot(
                    t_vals,
                    eyepos[iix[idx], start_time:end_time, 0],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
                ax_y.plot(
                    t_vals,
                    eyepos[iix[idx], start_time:end_time, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
            else:
                median_eyepos = np.nanmedian(eyepos[iix[idx], start_time:end_time, :], axis=0)
                ax.plot(
                    eyepos[iix[idx], start_time:end_time, 0],
                    eyepos[iix[idx], start_time:end_time, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7
                )
                ax.scatter(median_eyepos[0], median_eyepos[1], color=colors[idx], s=20, edgecolor='k', linewidth=0.7, zorder=3)

        if plot_time_traces:
            ax_x.set_ylabel("X (degrees)")
            ax_y.set_ylabel("Y (degrees)")
            ax_y.set_xlabel("Time (bins)" if bins_x_axis else "Time (ms)")
            if bins_x_axis:
                ax_x.set_title(f'{start_time} to {end_time} bins')
            else:
                ax_x.set_title(f'{round(start_time * 1/240 * 1000):.0f} to {round(end_time * 1/240 * 1000):.0f} ms')

            ax_x.set_ylim(-1, 1)
            ax_y.set_ylim(-1, 1)
        else:
            ax.set_xlim(np.nanmin(eyepos[iix, start_time:end_time, 0]), np.nanmax(eyepos[iix, start_time:end_time, 0]))
            ax.set_ylim(np.nanmin(eyepos[iix, start_time:end_time, 1]), np.nanmax(eyepos[iix, start_time:end_time, 1]))
            ax.set_xlabel('X (degrees)')
            ax.set_ylabel('Y (degrees)')
            if bins_x_axis:
                ax.set_title(f'{start_time} to {end_time} bins')
            else:
                ax.set_title(f'{round(start_time * 1/240 * 1000):.0f} to {round(end_time * 1/240 * 1000):.0f} ms')
        if show:
            plt.show()
        return (fig, axes) if plot_time_traces else (fig, ax)
    
    # List input - side by side plots
    n_plots = len(iix_list)
    
    # Pre-compute adjusted times for all plots
    adjusted_times = []
    for st, et in zip(start_time_list, end_time_list):
        time_window_len = et - st
        st_adj = max(st - peak_lag, 0)
        et_adj = st_adj + time_window_len
        adjusted_times.append((st_adj, et_adj))
    
    # Find global x and y limits, then make them equal for proper aspect ratio
    global_xmin, global_xmax = np.inf, -np.inf
    global_ymin, global_ymax = np.inf, -np.inf
    for iix_single, (st_adj, et_adj) in zip(iix_list, adjusted_times):
        xmin = np.nanmin(eyepos[iix_single, st_adj:et_adj, 0])
        xmax = np.nanmax(eyepos[iix_single, st_adj:et_adj, 0])
        ymin = np.nanmin(eyepos[iix_single, st_adj:et_adj, 1])
        ymax = np.nanmax(eyepos[iix_single, st_adj:et_adj, 1])
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)
    
    # Make x and y ranges equal for proper aspect ratio
    x_range = global_xmax - global_xmin
    y_range = global_ymax - global_ymin
    max_range = max(x_range, y_range)
    x_center = (global_xmin + global_xmax) / 2
    y_center = (global_ymin + global_ymax) / 2
    global_xmin, global_xmax = x_center - max_range / 2, x_center + max_range / 2
    global_ymin, global_ymax = y_center - max_range / 2, y_center + max_range / 2
    
    # Create subplots (2 rows for time traces, 1 row for XY)
    if plot_time_traces:
        fig, axes = plt.subplots(2, n_plots, figsize=(3 * n_plots, 4), sharex='col', squeeze=False)
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3), sharey=True, squeeze=False)
        axes = axes[0]
    
    for plot_idx, (iix_single, clusters_single, (st_adj, et_adj)) in enumerate(zip(iix_list, clusters_list, adjusted_times)):
        ax = axes[0, plot_idx] if plot_time_traces else axes[plot_idx]
        
        if clusters_single is None:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(iix_single)))
        else:
            num_clusters = len(set(clusters_single[clusters_single >= 0]))
            cluster_colors = plt.cm.coolwarm(np.linspace(0, 1, max(num_clusters, 1)))
            colors = [cluster_colors[c] if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters_single]
        
        for idx in range(len(iix_single)):
            if clusters_single is not None and not show_unclustered_points and clusters_single[idx] < 0:
                continue
            assert not microsaccade_exists(eyepos[iix_single[idx], st_adj:et_adj, :], threshold=0.1)
            if plot_time_traces:
                if bins_x_axis:
                    t_vals = np.arange(st_adj, et_adj)
                else:
                    t_vals = (np.arange(st_adj, et_adj) / 240) * 1000
                ax.plot(
                    t_vals,
                    eyepos[iix_single[idx], st_adj:et_adj, 0],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
                ax_y = axes[1, plot_idx]
                ax_y.plot(
                    t_vals,
                    eyepos[iix_single[idx], st_adj:et_adj, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
            else:
                median_eyepos = np.nanmedian(eyepos[iix_single[idx], st_adj:et_adj, :], axis=0)
                ax.plot(
                    eyepos[iix_single[idx], st_adj:et_adj, 0],
                    eyepos[iix_single[idx], st_adj:et_adj, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7
                )
                ax.scatter(median_eyepos[0], median_eyepos[1], color=colors[idx], s=20, edgecolor='k', linewidth=0.7, zorder=3)
        
        if plot_time_traces:
            if bins_x_axis:
                ax.set_title(f'{st_adj} to {et_adj} bins')
            else:
                ax.set_title(f'{round(st_adj * 1/240 * 1000):.0f} to {round(et_adj * 1/240 * 1000):.0f} ms')
            if plot_idx == 0:
                ax.set_ylabel('X (degrees)')
                axes[1, plot_idx].set_ylabel('Y (degrees)')
            axes[1, plot_idx].set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
            axes[1, plot_idx].set_ylim(-0.6, 0.6)
            axes[0, plot_idx].set_ylim(-0.6, 0.6)
        else:
            ax.set_xlim(global_xmin, global_xmax)
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_aspect('equal')
            ax.set_xlabel('X (degrees)')
            ax.set_title(f'{st_adj} to {et_adj} bins')
        
        # Only show y-axis label on leftmost plot
        if plot_idx == 0 and not plot_time_traces:
            ax.set_ylabel('Y (degrees)')
    
    plt.subplots_adjust(wspace=0)
    if show:
        plt.show()
    return fig, axes

def plot_population_raster(
    robs,
    iix,
    clusters,
    start_time,
    end_time,
    gap=100,
    show_psth=False,
    show_difference_psth=False,
    smooth_psth_sigma=0,
    show=True,
    render="scatter",
    fig_width=None,
    fig_height=12,
    fig_dpi=400,
    bins_x_axis=True,
):
    # robs shape: [trials, time, cells]
    
    # Track trial info and collect spike positions
    trial_info = []
    spike_x = []  # time positions
    spike_y = []  # row positions
    

    if isinstance(robs, list):
        assert len(robs) == len(iix) == len(clusters)
    else:
        robs = [robs]
        iix = [iix]
        clusters = [clusters]

    num_cells = robs[0].shape[2]
    
    robs_list = robs
    iix_list = iix
    clusters_list = clusters
    total_time = int(np.sum([r.shape[1] for r in robs_list]))

    prev_total_time = 0
    psth_height = num_cells * 1.2 # Adjust multiplier to change PSTH height
    psth_segments = {0: [], 1: []}  # Collect mean PSTH per segment
    
    # Pre-compute row positions based on max cluster trials across all segments
    max_cluster0_trials = max(np.sum(np.array(c) == 0) for c in clusters_list)
    max_cluster1_trials = max(np.sum(np.array(c) == 1) for c in clusters_list)
    psth_row_start = max_cluster0_trials * (num_cells + gap)
    # Pre-compute total_rows for consistent y-axis limits
    if show_difference_psth and not show_psth:
        raise ValueError("show_difference_psth=True requires show_psth=True")
    if show_psth:
        n_psth_rows = 2 + (1 if show_difference_psth else 0)
        psth_space = n_psth_rows * (psth_height + gap)
    else:
        psth_space = 0
    total_rows = psth_row_start + psth_space + max_cluster1_trials * (num_cells + gap) - gap

    img = None
    if render == "img":
        img = np.zeros((int(total_rows) + 1, total_time), dtype=np.uint8)

    for robs, iix, clusters in zip(robs_list, iix_list, clusters_list):
        current_row = 0
        trial_number = 1
        
        # Collect segment PSTHs
        seg_psth = {0: [], 1: []}
        
        # Cluster 0 trials first (top)

        for i, trial_idx in enumerate(iix):
            if clusters[i] == 0:
                spikes = robs[trial_idx, :]  # [time, cells]
                seg_psth[0].append(np.nansum(spikes, axis=1))  # sum over cells
                # Find all spike positions
                times, cells = np.where(spikes > 0)
                times += prev_total_time
                spike_x.extend(times)
                spike_y.extend(current_row + cells)
                if img is not None and times.size:
                    rows = (current_row + cells).astype(int)
                    img[rows, times] = 1
                trial_info.append((current_row + num_cells / 2, trial_number, 0))
                current_row += num_cells + gap
                trial_number += 1
        
        # Set cluster 1 start position (consistent across all segments)
        current_row = psth_row_start + psth_space
        
        # Cluster 1 trials (bottom)
        for i, trial_idx in enumerate(iix):
            if clusters[i] == 1:
                spikes = robs[trial_idx, :]  # [time, cells]
                seg_psth[1].append(np.nansum(spikes, axis=1))  # sum over cells
                times, cells = np.where(spikes > 0)
                times += prev_total_time
                spike_x.extend(times)
                spike_y.extend(current_row + cells)
                if img is not None and times.size:
                    rows = (current_row + cells).astype(int)
                    img[rows, times] = 1
                trial_info.append((current_row + num_cells / 2, trial_number, 1))
                current_row += num_cells + gap
                trial_number += 1
        
        # Average this segment's trials and store
        for c in [0, 1]:
            if seg_psth[c]:
                psth_segments[c].append(np.nanmean(seg_psth[c], axis=0))
            else:
                psth_segments[c].append(np.zeros(robs.shape[1]))
        
        prev_total_time += robs.shape[1]
    
    print(f"Found {len(spike_x)} spikes")
    if len(spike_x) == 0:
        print("No spikes to plot")
        return None
    if fig_width is None:
        fig_width = len(robs_list)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)

    if bins_x_axis:
        if render == "img":
            ax.set_rasterization_zorder(1)
            ax.imshow(
                img,
                interpolation="none",
                cmap="gray_r",
                aspect="auto",
                vmin=0,
                vmax=1,
                extent=(0, total_time, total_rows, 0),
                rasterized=True,
                zorder=0,
            )
        else:
            # Plot spikes as vertical ticks - linewidths controls horizontal thickness
            ax.scatter(spike_x, spike_y, s=0.6, c='black', marker='|', linewidths=2)
    else:
        x_scale = 1000 / 240    
        if render == "img":
            ax.set_rasterization_zorder(1)
            ax.imshow(
                img,
                interpolation="none",
                cmap="gray_r",
                aspect="auto",
                vmin=0,
                vmax=1,
                extent=(0, total_time * x_scale, total_rows, 0),
                rasterized=True,
                zorder=0,
            )
        else:
            # Plot spikes as vertical ticks - linewidths controls horizontal thickness
            spike_x_plot = np.asarray(spike_x) * x_scale
            ax.scatter(spike_x_plot, spike_y, s=0.6, c='black', marker='|', linewidths=2)
    
    # Plot PSTHs if enabled
    if show_psth and psth_row_start is not None:
        # Concatenate all segments
        psth0 = np.concatenate(psth_segments[0]) if psth_segments[0] else np.zeros(prev_total_time)
        psth1 = np.concatenate(psth_segments[1]) if psth_segments[1] else np.zeros(prev_total_time)
        if smooth_psth_sigma and smooth_psth_sigma > 0:
            radius = int(np.ceil(3 * smooth_psth_sigma))
            x = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (x / smooth_psth_sigma) ** 2)
            kernel /= np.sum(kernel)
            psth0 = np.convolve(psth0, kernel, mode='same')
            psth1 = np.convolve(psth1, kernel, mode='same')
        max_psth = max(np.nanmax(psth0), np.nanmax(psth1)) + 1e-10
        
        # Normalize and scale to fit in psth_height (inverted y-axis)
        offset0 = psth_row_start
        offset_diff = psth_row_start + (psth_height + gap)
        offset1 = psth_row_start + (2 * (psth_height + gap) if show_difference_psth else (psth_height + gap))

        psth0_scaled = offset0 + psth_height - (psth0 / max_psth) * psth_height
        psth1_scaled = offset1 + psth_height - (psth1 / max_psth) * psth_height
        
        if bins_x_axis:
            x_vals = np.arange(len(psth0))
        else:
            x_vals = np.arange(len(psth0)) * x_scale
        ax.fill_between(x_vals, offset0 + psth_height, psth0_scaled, color='blue', alpha=0.5)

        if show_difference_psth:
            diff_psth = np.abs(psth0 - psth1)
            diff_scaled = offset_diff + psth_height - (diff_psth / max_psth) * psth_height
            ax.fill_between(x_vals, offset_diff + psth_height, diff_scaled, color='k', alpha=0.25)

        ax.fill_between(x_vals, offset1 + psth_height, psth1_scaled, color='red', alpha=0.5)
    
    # Set axis limits
    # ax.set_xlim(0, end_time - start_time)
    if bins_x_axis:
        ax.set_xlim(0, total_time)
    else:
        x_max = total_time * x_scale
        ax.set_xlim(0, x_max)
        max_ms = x_max
        if max_ms <= 120:
            tick_step = 25
        elif max_ms <= 250:
            tick_step = 50
        else:
            tick_step = 100
        start_time_val = np.asarray(start_time).ravel()
        end_time_val = np.asarray(end_time).ravel()
        start_time_val = start_time_val[0] if start_time_val.size else start_time
        end_time_val = end_time_val[-1] if end_time_val.size else end_time
        start_ms = start_time_val / 240 * 1000
        end_ms = end_time_val / 240 * 1000
        n_intervals = max(1, int(round(max_ms / tick_step)))
        tick_positions = np.linspace(0, max_ms, n_intervals + 1)
        tick_labels = np.linspace(start_ms, end_ms, n_intervals + 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{tick:.0f}" for tick in tick_labels])
    ax.set_ylim(total_rows, 0)  # Inverted so trial 1 at top
    
    # Y-axis ticks with colored labels
    tick_positions = [info[0] for info in trial_info]
    tick_labels = [str(info[1]) for info in trial_info]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis='y', length=5, direction='out')
    
    # Color the tick labels by cluster
    cluster_colors = {0: 'blue', 1: 'red'}
    for tick_label, info in zip(ax.get_yticklabels(), trial_info):
        tick_label.set_color(cluster_colors[info[2]])

    if len(robs_list) > 1:
        t = 0
        section_boundaries = [0]
        for seg_robs in robs_list:
            if bins_x_axis:
                ax.axvline(x=t, color='red', linestyle='--')
            else:
                ax.axvline(x=t * x_scale, color='red', linestyle='--')
            t += seg_robs.shape[1]
            section_boundaries.append(t)
        if bins_x_axis:
            start_time_val = np.asarray(start_time).ravel()
            start_time_val = start_time_val[0] if start_time_val.size else start_time
            ax.set_xticks(section_boundaries)
            ax.set_xticklabels([f"{start_time_val + val:g}" for val in section_boundaries])
        else:
            start_time_val = np.asarray(start_time).ravel()
            start_time_val = start_time_val[0] if start_time_val.size else start_time
            start_ms = start_time_val / 240 * 1000
            section_positions = [val * x_scale for val in section_boundaries]
            section_labels = [start_ms + (val / 240 * 1000) for val in section_boundaries]
            ax.set_xticks(section_positions)
            ax.set_xticklabels([f"{val:.0f}" for val in section_labels])
    elif bins_x_axis:
        start_time_val = np.asarray(start_time).ravel()
        start_time_val = start_time_val[0] if start_time_val.size else start_time
        tick_positions = np.array(ax.get_xticks())
        tick_positions = tick_positions[(tick_positions >= 0) & (tick_positions <= total_time)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{start_time_val + val:g}" for val in tick_positions])
    
    ax.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
    ax.set_ylabel('Trial number')
    plt.title(f'session {subject}_{date}')
    if show:
        plt.show()

    return fig, ax, spike_x, spike_y


#%%
from tejas.rsvp_util import get_fixrsvp_data
subject = 'Allen'
date = '2022-03-02'

# date = "2022-03-04"
# date = "2022-04-08"
# date = "2022-04-13" #best
# date = "2022-04-15" #best
# date = "2022-04-06" #decent
# date = "2022-03-02" #best
# date = "2022-04-01" #not great
# date = "2022-03-30" #best
# subject = "Allen"

#jake likes 3-02, 4-06, 4-13

dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp.yaml'

data = get_fixrsvp_data(subject, date, dataset_configs_path, 
use_cached_data=False, 
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
# assert good_trials.sum() == len(robs)
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

from tejas.metrics.gaborium import get_rf_contour_metrics
rf_contour_metrics = get_rf_contour_metrics(date, subject)


# len_of_each_segment = 25
# len_of_each_segment = 40
len_of_each_segment = 32
# total_start_time = 0
total_start_time = 46
# total_end_time = 175
total_end_time = 78
max_distance_from_centroid = 0.10
num_clusters = 2
min_cluster_size = 5 #4
cluster_size = 5 #4
sort_by_cluster_psth = True
# distance_between_centroids = (0.3, 0.4)
distance_between_centroids = (0.02, 0.3)
min_distance_between_inter_cluster_points = 0#0.02
return_top_k_combos = 10


robs_list = []
iix_list = [[] for _ in range(return_top_k_combos)]
clusters_list = [[] for _ in range(return_top_k_combos)]
start_time_list = []
end_time_list = []

for i in range(total_start_time, total_end_time, len_of_each_segment):
    start_time = i
    end_time = start_time + len_of_each_segment
    #eyepos is shape [num_trial, time, 2]

    iix, clusters = get_eyepos_clusters(eyepos, start_time, end_time,
    robs, sort_by_cluster_psth = sort_by_cluster_psth,
    max_distance_from_centroid = max_distance_from_centroid, num_clusters = num_clusters, 
    min_cluster_size = min_cluster_size, cluster_size = cluster_size,
    distance_between_centroids = distance_between_centroids,
    min_distance_between_inter_cluster_points = min_distance_between_inter_cluster_points,
    return_top_k_combos = return_top_k_combos, dedupe = True)


    plot_eyepos_clusters(eyepos, iix[0], start_time, end_time, clusters=clusters[0])
    # plot_population_raster(robs[:, start_time:end_time, :], iix, clusters, show_psth = True, show_difference_psth = True)

    # print(i, clusters)

    for j in range(return_top_k_combos):
        if j < len(iix):
            iix_list[j].append(iix[j])
            clusters_list[j].append(clusters[j])
        else:
            first_iix = iix[0]
            iix_list[j].append(first_iix)
            clusters_list[j].append(np.full(len(first_iix), -1))
                

    # iix_list.append(iix)
    # clusters_list.append(clusters)

    robs_list.append(robs[:, start_time:end_time, :])
    start_time_list.append(start_time)
    end_time_list.append(end_time)



show = True
for j in range(return_top_k_combos)[:2]:
    fig1, ax1 = plot_eyepos_clusters(eyepos, iix_list[j], start_time_list, end_time_list, clusters=clusters_list[j], show=show)
#     fig1.savefig(f"population_figures/eyepos_{subject}_{date}_{j}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig1)
    fig2, ax2, spike_x, spike_y = plot_population_raster(robs_list, iix_list[j], clusters_list[j], start_time_list, end_time_list, 
    show_psth = True, show_difference_psth = False, show=show, render = "scatter", fig_width = 1, fig_height =20, fig_dpi = 400, gap= 50,
    bins_x_axis=True)
#     fig2.savefig(f"population_figures/population_raster_{subject}_{date}_{j}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig2)

start_time_list = np.array(start_time_list)
end_time_list = np.array(end_time_list)
    
#%%
def plot_eyepos_clusters_NEW(
    eyepos,
    iix,
    start_time,
    end_time,
    clusters=None,
    show=True,
    show_unclustered_points=True,
    plot_time_traces=False,
    bins_x_axis=False,
    use_peak_lag=True,
    plot_all_traces = False,
    vertical_line=None,  # Time bin index for vertical marker line
    vertical_linewidth=1.5,  # Line width for vertical marker
):
   
    all_lags = []
    for i in range(len(rf_contour_metrics)):
        all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
    if use_peak_lag:
        peak_lag = np.median(all_lags).astype(int)
    else:
        peak_lag = 0

    # Handle list input (treat len-1 list like single plot)
    is_list_input = isinstance(iix, list)
    if is_list_input and len(iix) == 1:
        iix = iix[0]
        start_time = start_time[0]
        end_time = end_time[0]
        clusters = clusters[0] if clusters is not None else None
        is_list_input = False
    
    if is_list_input:
        iix_list = iix
        start_time_list = start_time
        end_time_list = end_time
        clusters_list = clusters if clusters is not None else [None] * len(iix_list)
    else:
        # Single plot
        time_window_len = end_time - start_time
        start_time = max(start_time - peak_lag, 0)# - 30
        end_time = start_time + time_window_len #+ 100
        
        if plot_time_traces:
            fig, axes = plt.subplots(2, 1, sharex=True)
            ax_x, ax_y = axes
        else:
            fig, ax = plt.subplots()
        if clusters is None:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(iix)))
        else:
            num_clusters = len(set(clusters[clusters >= 0]))
            cluster_colors = plt.cm.coolwarm(np.linspace(0, 1, max(num_clusters, 1)))
            colors = [cluster_colors[c] if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters]
        
        # Plot ALL traces in gray first (background)
        if plot_all_traces and plot_time_traces:
            t_vals_bg = np.arange(start_time, end_time) if bins_x_axis else (np.arange(start_time, end_time) / 240) * 1000
            for trial_i in range(eyepos.shape[0]):
                y_i = eyepos[trial_i, start_time:end_time, :]
                valid_i = ~np.isnan(y_i).any(axis=-1)
                if valid_i.sum() > 0:
                    ax_x.plot(t_vals_bg[valid_i], y_i[valid_i, 0], color="gray", alpha=0.2)
                    ax_y.plot(t_vals_bg[valid_i], y_i[valid_i, 1], color="gray", alpha=0.2)

        for idx in range(len(iix)):
            if clusters is not None and not show_unclustered_points and clusters[idx] < 0:
                continue
            # assert not microsaccade_exists(eyepos[iix[idx], start_time:end_time, :], threshold=0.1)
            if plot_time_traces:
                if bins_x_axis:
                    t_vals = np.arange(start_time, end_time)
                else:
                    t_vals = (np.arange(start_time, end_time) / 240) * 1000
                ax_x.plot(
                    t_vals,
                    eyepos[iix[idx], start_time:end_time, 0],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
                ax_y.plot(
                    t_vals,
                    eyepos[iix[idx], start_time:end_time, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
            else:
                median_eyepos = np.nanmedian(eyepos[iix[idx], start_time:end_time, :], axis=0)
                ax.plot(
                    eyepos[iix[idx], start_time:end_time, 0],
                    eyepos[iix[idx], start_time:end_time, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7
                )
                ax.scatter(median_eyepos[0], median_eyepos[1], color=colors[idx], s=20, edgecolor='k', linewidth=0.7, zorder=3)

        if plot_time_traces:
            ax_x.set_ylabel("X (degrees)")
            ax_y.set_ylabel("Y (degrees)")
            ax_y.set_xlabel("Time (bins)" if bins_x_axis else "Time (ms)")
            if bins_x_axis:
                ax_x.set_title(f'{start_time} to {end_time} bins')
            else:
                ax_x.set_title(f'{round(start_time * 1/240 * 1000):.0f} to {round(end_time * 1/240 * 1000):.0f} ms')

            ax_x.set_ylim(-1, 1)
            ax_y.set_ylim(-1, 1)
        else:
            ax.set_xlim(np.nanmin(eyepos[iix, start_time:end_time, 0]), np.nanmax(eyepos[iix, start_time:end_time, 0]))
            ax.set_ylim(np.nanmin(eyepos[iix, start_time:end_time, 1]), np.nanmax(eyepos[iix, start_time:end_time, 1]))
            ax.set_xlabel('X (degrees)')
            ax.set_ylabel('Y (degrees)')
            if bins_x_axis:
                ax.set_title(f'{start_time} to {end_time} bins')
            else:
                ax.set_title(f'{round(start_time * 1/240 * 1000):.0f} to {round(end_time * 1/240 * 1000):.0f} ms')
        
        # Draw vertical marker line if specified (single-plot case)
        if vertical_line is not None:
            assert start_time <= vertical_line <= end_time, \
                f"vertical_line ({vertical_line}) must be between start_time ({start_time}) and end_time ({end_time})"
            if bins_x_axis:
                vline_x = vertical_line  # Absolute bin value
            else:
                vline_x = vertical_line * (1000 / 240)  # Convert absolute bin to absolute ms
            if plot_time_traces:
                for ax_single in axes:
                    ax_single.axvline(x=vline_x, color='red', linestyle='--', linewidth=vertical_linewidth, zorder=10)
            else:
                ax.axvline(x=vline_x, color='red', linestyle='--', linewidth=vertical_linewidth, zorder=10)
        
        if show:
            plt.show()
        return (fig, axes) if plot_time_traces else (fig, ax)
    
    # List input - side by side plots
    n_plots = len(iix_list)
    
    # Pre-compute adjusted times for all plots
    adjusted_times = []
    for st, et in zip(start_time_list, end_time_list):
        time_window_len = et - st
        st_adj = max(st - peak_lag, 0)
        et_adj = st_adj + time_window_len
        adjusted_times.append((st_adj, et_adj))
    
    # Find global x and y limits, then make them equal for proper aspect ratio
    global_xmin, global_xmax = np.inf, -np.inf
    global_ymin, global_ymax = np.inf, -np.inf
    for iix_single, (st_adj, et_adj) in zip(iix_list, adjusted_times):
        xmin = np.nanmin(eyepos[iix_single, st_adj:et_adj, 0])
        xmax = np.nanmax(eyepos[iix_single, st_adj:et_adj, 0])
        ymin = np.nanmin(eyepos[iix_single, st_adj:et_adj, 1])
        ymax = np.nanmax(eyepos[iix_single, st_adj:et_adj, 1])
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)
    
    # Make x and y ranges equal for proper aspect ratio
    x_range = global_xmax - global_xmin
    y_range = global_ymax - global_ymin
    max_range = max(x_range, y_range)
    x_center = (global_xmin + global_xmax) / 2
    y_center = (global_ymin + global_ymax) / 2
    global_xmin, global_xmax = x_center - max_range / 2, x_center + max_range / 2
    global_ymin, global_ymax = y_center - max_range / 2, y_center + max_range / 2
    
    # Create subplots (2 rows for time traces, 1 row for XY)
    if plot_time_traces:
        fig, axes = plt.subplots(2, n_plots, figsize=(3 * n_plots, 4), sharex='col', squeeze=False)
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3), sharey=True, squeeze=False)
        axes = axes[0]
    
    for plot_idx, (iix_single, clusters_single, (st_adj, et_adj)) in enumerate(zip(iix_list, clusters_list, adjusted_times)):
        ax = axes[0, plot_idx] if plot_time_traces else axes[plot_idx]
        
        if clusters_single is None:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(iix_single)))
        else:
            num_clusters = len(set(clusters_single[clusters_single >= 0]))
            cluster_colors = plt.cm.coolwarm(np.linspace(0, 1, max(num_clusters, 1)))
            colors = [cluster_colors[c] if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters_single]
        
        if plot_all_traces and plot_time_traces:
            t_vals = np.arange(st_adj, et_adj) if bins_x_axis else (np.arange(st_adj, et_adj) / 240) * 1000
            for trial_i in set(range(eyepos.shape[0])) - set(iix_single):
                y_i = eyepos[trial_i, st_adj:et_adj, :]
                valid_i = ~np.isnan(y_i).any(axis=-1)
                ax.plot(t_vals[valid_i], y_i[valid_i, 0], color="gray", alpha=0.2)
                axes[1, plot_idx].plot(t_vals[valid_i], y_i[valid_i, 1], color="gray", alpha=0.2)

        for idx in range(len(iix_single)):
            if clusters_single is not None and not show_unclustered_points and clusters_single[idx] < 0:
                continue
            assert not microsaccade_exists(eyepos[iix_single[idx], st_adj:et_adj, :], threshold=0.1)
            if plot_time_traces:
                if bins_x_axis:
                    t_vals = np.arange(st_adj, et_adj)
                else:
                    t_vals = (np.arange(st_adj, et_adj) / 240) * 1000
                ax.plot(
                    t_vals,
                    eyepos[iix_single[idx], st_adj:et_adj, 0],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
                ax_y = axes[1, plot_idx]
                ax_y.plot(
                    t_vals,
                    eyepos[iix_single[idx], st_adj:et_adj, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7,
                )
            else:
                median_eyepos = np.nanmedian(eyepos[iix_single[idx], st_adj:et_adj, :], axis=0)
                ax.plot(
                    eyepos[iix_single[idx], st_adj:et_adj, 0],
                    eyepos[iix_single[idx], st_adj:et_adj, 1],
                    color=colors[idx],
                    alpha=1,
                    linewidth=0.7
                )
                ax.scatter(median_eyepos[0], median_eyepos[1], color=colors[idx], s=20, edgecolor='k', linewidth=0.7, zorder=3)
        
        if plot_time_traces:
            if bins_x_axis:
                ax.set_title(f'{st_adj} to {et_adj} bins')
            else:
                ax.set_title(f'{round(st_adj * 1/240 * 1000):.0f} to {round(et_adj * 1/240 * 1000):.0f} ms')
            if plot_idx == 0:
                ax.set_ylabel('X (degrees)')
                axes[1, plot_idx].set_ylabel('Y (degrees)')
            axes[1, plot_idx].set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
            axes[1, plot_idx].set_ylim(-0.6, 0.6)
            axes[0, plot_idx].set_ylim(-0.6, 0.6)
        else:
            ax.set_xlim(global_xmin, global_xmax)
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_aspect('equal')
            ax.set_xlabel('X (degrees)')
            ax.set_title(f'{st_adj} to {et_adj} bins')
        
        # Only show y-axis label on leftmost plot
        if plot_idx == 0 and not plot_time_traces:
            ax.set_ylabel('Y (degrees)')
    
    # Draw vertical marker line if specified
    if vertical_line is not None:
        # Validate that vertical_line is within bounds
        st_vals = np.atleast_1d(start_time)
        et_vals = np.atleast_1d(end_time)
        min_start = st_vals.min()
        max_end = et_vals.max()
        assert min_start <= vertical_line <= max_end, \
            f"vertical_line ({vertical_line}) must be between start_time ({min_start}) and end_time ({max_end})"
        
        # Convert to x-axis units (bins or ms)
        # Note: eyepos plots use ABSOLUTE time values on x-axis, not relative
        if bins_x_axis:
            vline_x = vertical_line  # Absolute bin value
        else:
            vline_x = vertical_line * (1000 / 240)  # Convert absolute bin to absolute ms
        
        # Draw on all axes
        for ax in axes.ravel():
            ax.axvline(x=vline_x, color='red', linestyle='--', linewidth=vertical_linewidth, zorder=10)
    
    plt.subplots_adjust(wspace=0)
    if show:
        plt.show()
    return fig, axes

def plot_spikes_as_lines(ax, spike_x, spike_y, spike_vals=None, height=1.0, color="k", linewidth=0.5, alpha=1.0):
    """
    Plot spikes as vertical line segments with optional alpha variation based on spike values.
    
    Parameters:
    - ax: matplotlib axis
    - spike_x: array of x (time) positions
    - spike_y: array of y (row) positions
    - spike_vals: optional array of spike values for alpha variation
    - height: height of each line segment
    - color: line color
    - linewidth: line width
    - alpha: base alpha (modulated by spike_vals if provided)
    """
    spike_x = np.asarray(spike_x)
    spike_y = np.asarray(spike_y)
    
    if spike_x.size == 0:
        return None
    
    if spike_vals is None:
        # Simple case: all spikes same alpha
        x_lines = np.vstack([spike_x, spike_x, np.full(len(spike_x), np.nan)])
        y_lines = np.vstack([spike_y, spike_y + height, np.full(len(spike_y), np.nan)])
        y_lines = np.vstack([spike_y, spike_y + height, np.full(len(spike_y), np.nan)])
        return ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'), 
                       color=color, linewidth=linewidth, alpha=alpha, rasterized=True)[0]
    
    # Alpha varies by spike value
    spike_vals = np.asarray(spike_vals)
    unique_vals = np.unique(spike_vals)
    # print(np.unique_counts(spike_vals))
    vmin, vmax = unique_vals[0], unique_vals[-1]
    handles = []
    
    for val in unique_vals:
        sel = spike_vals == val
        if not np.any(sel):
            continue
        if vmax > vmin:
            norm = (val - vmin) / (vmax - vmin)
            alpha_val = np.clip(0.5 + 0.9 * norm, 0.0, 1.0) * alpha
            # print(alpha_val)
        else:
            alpha_val = alpha
        
        x_sel, y_sel = spike_x[sel], spike_y[sel]
        x_lines = np.vstack([x_sel, x_sel, np.full(sel.sum(), np.nan)])
        y_lines = np.vstack([y_sel, y_sel + height, np.full(sel.sum(), np.nan)])
        handles.append(
            ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'),
                    color=color, linewidth=linewidth, alpha=alpha_val, rasterized=True)[0]
        )
    return handles

def _get_trial_spikes(trial_idx, robs, use_spike_times, spike_times_trials, trial_t_bins, dt, num_cells):
    """
    Extract spike times and cells for a single trial.
    
    Returns:
        times: array of time positions (bin indices for robs, fractional for spike_times)
        cells: array of cell indices
        vals: array of spike values (for alpha) or None if using spike_times
    """
    if use_spike_times and spike_times_trials is not None:
        # Get time bounds from trial_t_bins (handle sparse NaN-padded arrays)
        t_bins = trial_t_bins[trial_idx]
        valid_mask = ~np.isnan(t_bins)
        if not np.any(valid_mask):
            # No valid time bins for this trial
            return np.array([]), np.array([]), None
        
        valid_t_bins = t_bins[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        first_valid_idx = valid_indices[0]  # Index offset for bin coordinates
        
        t_start = valid_t_bins[0] - dt/2   # left edge of first valid bin
        t_end = valid_t_bins[-1] + dt/2    # right edge of last valid bin
        
        times_list = []
        cells_list = []
        for cell_idx in range(num_cells):
            cell_spikes = np.atleast_1d(np.asarray(spike_times_trials[trial_idx][cell_idx]))
            if cell_spikes.size == 0:
                continue
            # Filter to time window
            mask = (cell_spikes >= t_start) & (cell_spikes < t_end)
            filtered = cell_spikes[mask]
            if filtered.size > 0:
                # Convert to fractional bin coordinates
                # Offset by first_valid_idx so coordinates align with slice position, not first valid bin
                bin_coords = (filtered - t_start) / dt + first_valid_idx
                times_list.append(bin_coords)
                cells_list.append(np.full(filtered.size, cell_idx))
        
        if times_list:
            times = np.concatenate(times_list)
            cells = np.concatenate(cells_list).astype(int)
        else:
            times = np.array([])
            cells = np.array([])
        
        vals = None  # No alpha variation for spike times
        
        # Verification: compare with robs
        # Only count robs bins that correspond to valid (non-NaN) time bins
        # When window is expanded beyond fixation, only compare the valid region
        n_valid_bins = len(valid_t_bins)
        # Sum robs only at valid bin positions
        robs_sum = int(np.nansum(robs[trial_idx, valid_indices, :]))
        
        if robs_sum != len(times):
            # Debug: check the time ranges
            all_spikes = []
            for cell_idx in range(num_cells):
                cell_spikes = np.atleast_1d(np.asarray(spike_times_trials[trial_idx][cell_idx]))
                if cell_spikes.size > 0:
                    all_spikes.extend(cell_spikes)
            all_spikes = np.array(all_spikes)
            
            print(f"MISMATCH trial {trial_idx}:")
            print(f"  valid t_bins range: [{valid_t_bins[0]:.6f}, {valid_t_bins[-1]:.6f}]")
            print(f"  filter window: [{t_start:.6f}, {t_end:.6f})")
            print(f"  valid_indices: {first_valid_idx} to {valid_indices[-1]} ({n_valid_bins} bins)")
            print(f"  spike_times_trials has {len(all_spikes)} total spikes for this trial")
            if len(all_spikes) > 0:
                print(f"  spike times range: [{all_spikes.min():.6f}, {all_spikes.max():.6f}]")
                in_window = ((all_spikes >= t_start) & (all_spikes < t_end)).sum()
                print(f"  spikes in filter window: {in_window}")
            print(f"  robs_sum={robs_sum}, spike_times_in_window={len(times)}")
        
        return times, cells, vals
    else:
        # Use binned robs data
        spikes = robs[trial_idx, :]  # [time, cells]
        times, cells = np.where(spikes > 0)
        vals = spikes[times, cells]
        return times, cells, vals


def _compute_psth_from_spike_times(spike_times_trial, t_bins, psth_bin_size, dt):
    """
    Compute PSTH from spike times by binning at specified resolution.
    
    Parameters
    ----------
    spike_times_trial : list of np.ndarray
        spike_times_trial[cell_idx] = array of spike times for that cell
    t_bins : np.ndarray
        Time bin centers for this trial (sparse indexing - may contain NaN)
    psth_bin_size : float
        Bin size for PSTH in seconds (e.g., 0.001 for 1ms)
    dt : float
        Original data bin size (for determining time window)
    
    Returns
    -------
    psth : np.ndarray
        Spike counts per PSTH bin (summed across all cells)
    psth_time_edges : np.ndarray
        Bin edges for the PSTH
    """
    # Get valid (non-NaN) time bins
    valid_mask = ~np.isnan(t_bins)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    valid_t_bins = t_bins[valid_mask]
    
    # Determine time window from t_bins
    t_start = valid_t_bins[0] - dt/2
    t_end = valid_t_bins[-1] + dt/2
    
    # Create PSTH bin edges at specified resolution
    psth_edges = np.arange(t_start, t_end + psth_bin_size/2, psth_bin_size)
    n_psth_bins = len(psth_edges) - 1
    
    if n_psth_bins <= 0:
        return np.array([]), np.array([])
    
    # Bin all spikes from all cells
    all_spike_times = []
    for cell_spikes in spike_times_trial:
        cell_spikes = np.atleast_1d(np.asarray(cell_spikes))
        if cell_spikes.size > 0:
            # Filter to time window
            mask = (cell_spikes >= t_start) & (cell_spikes < t_end)
            all_spike_times.extend(cell_spikes[mask])
    
    if len(all_spike_times) == 0:
        return np.zeros(n_psth_bins), psth_edges
    
    all_spike_times = np.array(all_spike_times)
    
    # Bin the spikes
    psth, _ = np.histogram(all_spike_times, bins=psth_edges)
    
    return psth, psth_edges


def plot_population_raster_NEW(
    robs,
    iix,
    clusters,
    start_time,
    end_time,
    gap=20,
    show_psth=False,
    show_difference_psth=False,
    smooth_psth_sigma=0,
    show=True,
    render="scatter",
    fig_width=5,
    fig_height=12,
    fig_dpi=800,
    bins_x_axis=True,
    # Line render parameters
    line_height=0.1,
    line_color="k",
    line_linewidth=2.8,
    line_alpha=1.0,
    use_line_alpha=True,
    # Spike times parameters
    use_spike_times=False,
    spike_times_trials=None,
    trial_t_bins=None,
    dt=1/240,
    psth_bin_size=0.001,  # PSTH bin size in seconds (default 1ms) when using spike times
    vertical_line=None,  # Time bin index for vertical marker line
    vertical_linewidth=1.5,  # Line width for vertical marker
):
    # robs shape: [trials, time, cells]
    
    # Track trial info and collect spike positions
    trial_info = []
    spike_x = []  # time positions
    spike_y = []  # row positions
    spike_vals = []  # spike values for alpha variation
    

    if isinstance(robs, list):
        assert len(robs) == len(iix) == len(clusters)
    else:
        robs = [robs]
        iix = [iix]
        clusters = [clusters]
    
    # Validate spike times parameters
    n_segments = len(robs)
    if use_spike_times:
        assert spike_times_trials is not None, "spike_times_trials required when use_spike_times=True"
        assert trial_t_bins is not None, "trial_t_bins required when use_spike_times=True"
        if render == "img":
            print("Warning: img rendering with spike_times uses rounded bin indices, losing sub-bin precision")
        # spike_times_trials is the same for all segments (contains all trials)
        # Replicate it for each segment
        spike_times_list = [spike_times_trials] * n_segments
        # trial_t_bins should be [segment][trial] structure
        # Check if it's already structured that way (len matches segments and first element is a list of trials)
        if isinstance(trial_t_bins, list) and len(trial_t_bins) == n_segments and isinstance(trial_t_bins[0], list):
            trial_t_bins_list = trial_t_bins  # Already [segment][trial]
        else:
            trial_t_bins_list = [trial_t_bins]  # Wrap for single segment
    else:
        spike_times_list = [None] * n_segments
        trial_t_bins_list = [None] * n_segments

    num_cells = robs[0].shape[2]
    
    robs_list = robs
    iix_list = iix
    clusters_list = clusters
    total_time = int(np.sum([r.shape[1] for r in robs_list]))

    prev_total_time = 0
    psth_height = num_cells * 1 # Adjust multiplier to change PSTH height
    psth_segments = {0: [], 1: []}  # Collect mean PSTH per segment
    
    # Pre-compute row positions based on max cluster trials across all segments
    max_cluster0_trials = max(np.sum(np.array(c) == 0) for c in clusters_list)
    max_cluster1_trials = max(np.sum(np.array(c) == 1) for c in clusters_list)
    psth_row_start = max_cluster0_trials * (num_cells + gap)
    # Pre-compute total_rows for consistent y-axis limits
    if show_difference_psth and not show_psth:
        raise ValueError("show_difference_psth=True requires show_psth=True")
    if show_psth:
        n_psth_rows = 2 + (1 if show_difference_psth else 0)
        psth_space = n_psth_rows * (psth_height + gap)
    else:
        psth_space = 0
    total_rows = psth_row_start + psth_space + max_cluster1_trials * (num_cells + gap) - gap

    img = None
    if render == "img":
        img = np.zeros((int(total_rows) + 1, total_time), dtype=np.uint8)

    # Collect trial data by cluster for output
    cluster0_trials = []
    cluster1_trials = []

    for robs, iix, clusters, st_trials, t_bins in zip(robs_list, iix_list, clusters_list, spike_times_list, trial_t_bins_list):
        current_row = 0
        trial_number = 1
        
        # Collect segment PSTHs
        seg_psth = {0: [], 1: []}
        seg_psth_n_bins = None  # Track PSTH bin count for this segment
        
        # Cluster 0 trials first (top)

        for i, trial_idx in enumerate(iix):
            if clusters[i] == 0:
                spikes = robs[trial_idx, :]  # [time, cells]
                cluster0_trials.append(spikes)  # collect for output
                
                # Compute PSTH - from spike times if use_spike_times, else from robs
                if use_spike_times and st_trials is not None:
                    trial_psth, _ = _compute_psth_from_spike_times(
                        st_trials[trial_idx], t_bins[trial_idx], psth_bin_size, dt
                    )
                    if len(trial_psth) > 0:
                        seg_psth[0].append(trial_psth)
                        if seg_psth_n_bins is None:
                            seg_psth_n_bins = len(trial_psth)
                else:
                    seg_psth[0].append(np.nansum(spikes, axis=1))  # sum over cells
                
                # Get spike positions (uses helper for spike_times or robs)
                print('on trial', trial_number)
                times, cells, vals = _get_trial_spikes(
                    trial_idx, robs, use_spike_times, st_trials, t_bins, dt, num_cells
                )
                times = times + prev_total_time
                spike_x.extend(times)
                spike_y.extend(current_row + cells)
                if vals is not None:
                    spike_vals.extend(vals)
                if img is not None and len(times) > 0:
                    rows = (current_row + cells).astype(int)
                    img[rows, times.astype(int)] = 1
                trial_info.append((current_row + num_cells / 2, trial_number, 0))
                current_row += num_cells + gap
                trial_number += 1
        
        # Set cluster 1 start position (consistent across all segments)
        current_row = psth_row_start + psth_space
        
        # Cluster 1 trials (bottom)
        for i, trial_idx in enumerate(iix):
            if clusters[i] == 1:
                spikes = robs[trial_idx, :]  # [time, cells]
                cluster1_trials.append(spikes)  # collect for output
                
                # Compute PSTH - from spike times if use_spike_times, else from robs
                if use_spike_times and st_trials is not None:
                    trial_psth, _ = _compute_psth_from_spike_times(
                        st_trials[trial_idx], t_bins[trial_idx], psth_bin_size, dt
                    )
                    if len(trial_psth) > 0:
                        seg_psth[1].append(trial_psth)
                        if seg_psth_n_bins is None:
                            seg_psth_n_bins = len(trial_psth)
                else:
                    seg_psth[1].append(np.nansum(spikes, axis=1))  # sum over cells
                
                # Get spike positions (uses helper for spike_times or robs)
                print('on trial', trial_number)
                times, cells, vals = _get_trial_spikes(
                    trial_idx, robs, use_spike_times, st_trials, t_bins, dt, num_cells
                )
                times = times + prev_total_time
                spike_x.extend(times)
                spike_y.extend(current_row + cells)
                if vals is not None:
                    spike_vals.extend(vals)
                if img is not None and len(times) > 0:
                    rows = (current_row + cells).astype(int)
                    img[rows, times.astype(int)] = 1
                trial_info.append((current_row + num_cells / 2, trial_number, 1))
                current_row += num_cells + gap
                trial_number += 1
        
        # Average this segment's trials and store
        # Determine expected PSTH size for this segment
        if use_spike_times and seg_psth_n_bins is not None:
            expected_psth_size = seg_psth_n_bins
        else:
            expected_psth_size = robs.shape[1]
            
        for c in [0, 1]:
            if seg_psth[c]:
                # Pad/truncate to consistent size if needed
                padded = []
                for p in seg_psth[c]:
                    if len(p) < expected_psth_size:
                        p = np.pad(p, (0, expected_psth_size - len(p)), constant_values=0)
                    elif len(p) > expected_psth_size:
                        p = p[:expected_psth_size]
                    padded.append(p)
                psth_segments[c].append(np.nanmean(padded, axis=0))
            else:
                psth_segments[c].append(np.zeros(expected_psth_size))
        
        prev_total_time += robs.shape[1]
    
    # Stack cluster trials: cluster0 first, then cluster1
    robs_clustered = np.stack(cluster0_trials + cluster1_trials, axis=0)  # shape: (2*trials_per_cluster, T, N)
    
    print(f"Found {len(spike_x)} spikes")
    if len(spike_x) == 0:
        print("No spikes to plot")
        return None
    if fig_width is None:
        fig_width = len(robs_list)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)

    if bins_x_axis:
        if render == "img":
            ax.set_rasterization_zorder(1)
            ax.imshow(
                img,
                interpolation="none",
                cmap="gray_r",
                aspect="auto",
                vmin=0,
                vmax=1,
                extent=(0, total_time, total_rows, 0),
                rasterized=True,
                zorder=0,
            )
        elif render == "line":
            plot_spikes_as_lines(ax, spike_x, spike_y, spike_vals if spike_vals and use_line_alpha else None,
                                 height=line_height, color=line_color, linewidth=line_linewidth, alpha=line_alpha)
        else:
            # Plot spikes as vertical ticks - linewidths controls horizontal thickness
            ax.scatter(spike_x, spike_y, s=0.6, c='black', marker='|', linewidths=2)
    else:
        x_scale = 1000 / 240    
        if render == "img":
            ax.set_rasterization_zorder(1)
            ax.imshow(
                img,
                interpolation="none",
                cmap="gray_r",
                aspect="auto",
                vmin=0,
                vmax=1,
                extent=(0, total_time * x_scale, total_rows, 0),
                rasterized=True,
                zorder=0,
            )
        elif render == "line":
            spike_x_scaled = np.asarray(spike_x) * x_scale
            plot_spikes_as_lines(ax, spike_x_scaled, spike_y, spike_vals if spike_vals and use_line_alpha else None,
                                 height=line_height, color=line_color, linewidth=line_linewidth, alpha=line_alpha)
        else:
            # Plot spikes as vertical ticks - linewidths controls horizontal thickness
            spike_x_plot = np.asarray(spike_x) * x_scale
            ax.scatter(spike_x_plot, spike_y, s=0.6, c='black', marker='|', linewidths=2)
    
    # Plot PSTHs if enabled
    if show_psth and psth_row_start is not None:
        # Concatenate all segments
        psth0 = np.concatenate(psth_segments[0]) if psth_segments[0] else np.zeros(prev_total_time)
        psth1 = np.concatenate(psth_segments[1]) if psth_segments[1] else np.zeros(prev_total_time)
        if smooth_psth_sigma and smooth_psth_sigma > 0:
            radius = int(np.ceil(3 * smooth_psth_sigma))
            x = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (x / smooth_psth_sigma) ** 2)
            kernel /= np.sum(kernel)
            psth0 = np.convolve(psth0, kernel, mode='same')
            psth1 = np.convolve(psth1, kernel, mode='same')
        max_psth = max(np.nanmax(psth0), np.nanmax(psth1)) + 1e-10
        
        # Normalize and scale to fit in psth_height (inverted y-axis)
        offset0 = psth_row_start
        offset_diff = psth_row_start + (psth_height + gap)
        offset1 = psth_row_start + (2 * (psth_height + gap) if show_difference_psth else (psth_height + gap))

        psth0_scaled = offset0 + psth_height - (psth0 / max_psth) * psth_height
        psth1_scaled = offset1 + psth_height - (psth1 / max_psth) * psth_height
        
        # Create x values for PSTH that span the full raster range
        # When using spike times, PSTH may have different number of bins than robs
        if bins_x_axis:
            x_vals = np.linspace(0, total_time, len(psth0))
        else:
            x_vals = np.linspace(0, total_time * x_scale, len(psth0))
        ax.fill_between(x_vals, offset0 + psth_height, psth0_scaled, color='blue', alpha=0.5)

        if show_difference_psth:
            diff_psth = np.abs(psth0 - psth1)
            diff_scaled = offset_diff + psth_height - (diff_psth / max_psth) * psth_height
            ax.fill_between(x_vals, offset_diff + psth_height, diff_scaled, color='k', alpha=0.25)

        ax.fill_between(x_vals, offset1 + psth_height, psth1_scaled, color='red', alpha=0.5)
    
    # Set axis limits
    # ax.set_xlim(0, end_time - start_time)
    if bins_x_axis:
        ax.set_xlim(0, total_time)
    else:
        x_max = total_time * x_scale
        ax.set_xlim(0, x_max)
        max_ms = x_max
        if max_ms <= 120:
            tick_step = 25
        elif max_ms <= 250:
            tick_step = 50
        else:
            tick_step = 100
        start_time_val = np.asarray(start_time).ravel()
        end_time_val = np.asarray(end_time).ravel()
        start_time_val = start_time_val[0] if start_time_val.size else start_time
        end_time_val = end_time_val[-1] if end_time_val.size else end_time
        start_ms = start_time_val / 240 * 1000
        end_ms = end_time_val / 240 * 1000
        n_intervals = max(1, int(round(max_ms / tick_step)))
        tick_positions = np.linspace(0, max_ms, n_intervals + 1)
        tick_labels = np.linspace(start_ms, end_ms, n_intervals + 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{tick:.0f}" for tick in tick_labels])
    ax.set_ylim(total_rows, 0)  # Inverted so trial 1 at top
    
    # Y-axis ticks with colored labels
    tick_positions = [info[0] for info in trial_info]
    tick_labels = [str(info[1]) for info in trial_info]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(axis='y', length=5, direction='out')
    
    # Color the tick labels by cluster
    cluster_colors = {0: 'blue', 1: 'red'}
    for tick_label, info in zip(ax.get_yticklabels(), trial_info):
        tick_label.set_color(cluster_colors[info[2]])

    if len(robs_list) > 1:
        t = 0
        section_boundaries = [0]
        for seg_robs in robs_list:
            if bins_x_axis:
                ax.axvline(x=t, color='red', linestyle='--')
            else:
                ax.axvline(x=t * x_scale, color='red', linestyle='--')
            t += seg_robs.shape[1]
            section_boundaries.append(t)
        if bins_x_axis:
            start_time_val = np.asarray(start_time).ravel()
            start_time_val = start_time_val[0] if start_time_val.size else start_time
            ax.set_xticks(section_boundaries)
            ax.set_xticklabels([f"{start_time_val + val:g}" for val in section_boundaries])
        else:
            start_time_val = np.asarray(start_time).ravel()
            start_time_val = start_time_val[0] if start_time_val.size else start_time
            start_ms = start_time_val / 240 * 1000
            section_positions = [val * x_scale for val in section_boundaries]
            section_labels = [start_ms + (val / 240 * 1000) for val in section_boundaries]
            ax.set_xticks(section_positions)
            ax.set_xticklabels([f"{val:.0f}" for val in section_labels])
    elif bins_x_axis:
        start_time_val = np.asarray(start_time).ravel()
        start_time_val = start_time_val[0] if start_time_val.size else start_time
        tick_positions = np.array(ax.get_xticks())
        tick_positions = tick_positions[(tick_positions >= 0) & (tick_positions <= total_time)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{start_time_val + val:g}" for val in tick_positions])
    
    # Draw vertical marker line if specified
    if vertical_line is not None:
        # Validate that vertical_line is within bounds
        st_vals = np.atleast_1d(start_time)
        et_vals = np.atleast_1d(end_time)
        min_start = st_vals.min()
        max_end = et_vals.max()
        assert min_start <= vertical_line <= max_end, \
            f"vertical_line ({vertical_line}) must be between start_time ({min_start}) and end_time ({max_end})"
        
        # Convert to x-axis units (bins or ms)
        if bins_x_axis:
            vline_x = vertical_line - min_start  # Relative to start
        else:
            vline_x = (vertical_line - min_start) * (1000 / 240)  # Convert to ms
        
        # Draw vertical line spanning full height
        ax.axvline(x=vline_x, color='red', linestyle='--', linewidth=vertical_linewidth, zorder=10)
    
    ax.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
    ax.set_ylabel('Trial number')
    plt.title(f'session {subject}_{date}')
    if show:
        plt.show()

    return fig, ax, spike_x, spike_y, robs_clustered


def population_plot_movie(
    robs, eyepos, iix, clusters, start_time, end_time,
    image_ids, rsvp_images, ppd,
    # Optional parameters
    trail_length=3,  # Number of bins of eye trajectory history
    fps=10,
    save_path=None,
    show=True,
    # Eye position parameters
    show_unclustered_points=False,  # Whether to show unclustered trials (cluster < 0)
    plot_all_traces=True,  # Whether to plot all trials as gray background traces
    image_extent=1.0,  # Extent of image/eyepos display in degrees (radius from center)
    # Pass-through parameters for existing functions
    gap=20,
    render="line",
    dt=1/240,
    use_spike_times=False,
    spike_times_trials=None,
    trial_t_bins=None,
    show_psth=True,
    bins_x_axis=True,
    fig_width=16,
    fig_height=20,
    fig_dpi=150,
    **kwargs
):
    """
    Create an animated movie showing synchronized eye position trajectories,
    RSVP images, eye traces, and raster plots with a moving time marker.
    
    Parameters
    ----------
    robs : np.ndarray
        Spike data, shape [trials, time, cells]
    eyepos : np.ndarray
        Eye position data, shape [trials, time, 2] (X, Y in degrees)
    iix : np.ndarray
        Trial indices to include
    clusters : np.ndarray
        Cluster assignments for each trial in iix
    start_time : int
        Start time bin
    end_time : int
        End time bin
    image_ids : np.ndarray
        Image IDs per bin, shape [trials, time]. Values index into rsvp_images.
    rsvp_images : np.ndarray
        RSVP images, shape [num_images, H, W]
    ppd : float
        Pixels per degree for cropping images
    trail_length : int
        Number of bins of eye trajectory history to show
    fps : int
        Frames per second for animation
    save_path : str, optional
        Path to save the animation (e.g., 'movie.mp4')
    show : bool
        Whether to display the animation
    show_unclustered_points : bool
        Whether to show unclustered trials (cluster < 0). Default False.
    plot_all_traces : bool
        Whether to plot all trials as gray background traces. Default True.
    image_extent : float
        Extent of image/eyepos display in degrees (radius from center). Default 1.0.
        Controls both the image cropping and the axis limits for the left panel.
    gap : int
        Gap between trials in raster plot
    render : str
        Render mode for raster ('line', 'scatter', 'img')
    dt : float
        Time bin size in seconds
    use_spike_times : bool
        Whether to use spike times for raster
    spike_times_trials : list, optional
        Spike times per trial
    trial_t_bins : list, optional
        Time bins per trial
    show_psth : bool
        Whether to show PSTH in raster plot
    bins_x_axis : bool
        Whether to use bins (True) or ms (False) for x-axis
    
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec
    
    # =========================================================================
    # Setup Phase: Create figure and static elements
    # =========================================================================
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.5], height_ratios=[1, 1, 2],
                  hspace=0.3, wspace=0.3)
    
    # Left: image + eye trajectories (spans all rows)
    ax_image = fig.add_subplot(gs[:, 0])
    # Right top: eye X trace
    ax_eye_x = fig.add_subplot(gs[0, 1])
    # Right middle: eye Y trace  
    ax_eye_y = fig.add_subplot(gs[1, 1], sharex=ax_eye_x)
    # Right bottom: raster plot
    ax_raster = fig.add_subplot(gs[2, 1])
    
    # =========================================================================
    # Prepare data
    # =========================================================================
    
    # Handle single values for start_time/end_time
    start_time_val = int(np.atleast_1d(start_time).ravel()[0])
    end_time_val = int(np.atleast_1d(end_time).ravel()[-1])
    n_frames = end_time_val - start_time_val
    
    # Get cluster colors (same as plot_eyepos_clusters_NEW)
    if clusters is None:
        num_trials = len(iix)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_trials))
    else:
        clusters_arr = np.asarray(clusters)
        num_clusters = len(set(clusters_arr[clusters_arr >= 0]))
        cluster_colors = plt.cm.coolwarm(np.linspace(0, 1, max(num_clusters, 1)))
        colors = [cluster_colors[c] if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters_arr]
    
    # =========================================================================
    # Draw static raster plot on ax_raster
    # =========================================================================
    
    # Slice robs for the time window
    robs_sliced = robs[:, start_time_val:end_time_val, :]
    num_cells = robs_sliced.shape[2]
    
    # Collect spikes for plotting
    spike_x_raster = []
    spike_y_raster = []
    current_row = 0
    trial_info_raster = []
    
    # Compute row positions
    max_cluster0_trials = np.sum(np.asarray(clusters) == 0) if clusters is not None else 0
    max_cluster1_trials = np.sum(np.asarray(clusters) == 1) if clusters is not None else 0
    
    psth_height = num_cells * 0.8
    psth_space = 2 * (psth_height + gap) if show_psth else 0
    psth_row_start = max_cluster0_trials * (num_cells + gap)
    
    # X scale for converting bins to ms
    x_scale = 1000 / 240 if not bins_x_axis else 1
    total_time = end_time_val - start_time_val
    
    # Cluster 0 trials first
    trial_number = 1
    psth_data = {0: [], 1: []}
    
    for i, trial_idx in enumerate(iix):
        cluster_id = clusters[i] if clusters is not None else 0
        if cluster_id == 0:
            spikes = robs_sliced[trial_idx, :, :]  # [time, cells]
            psth_data[0].append(np.nansum(spikes, axis=1))
            
            if use_spike_times and spike_times_trials is not None and trial_t_bins is not None:
                # Use spike times for more precise plotting
                t_bins = trial_t_bins[trial_idx]
                valid_mask = ~np.isnan(t_bins)
                if np.any(valid_mask):
                    valid_t_bins = t_bins[valid_mask]
                    t_start = valid_t_bins[0] - dt/2
                    t_end = valid_t_bins[-1] + dt/2
                    
                    for cell_idx in range(num_cells):
                        cell_spikes = np.atleast_1d(np.asarray(spike_times_trials[trial_idx][cell_idx]))
                        if cell_spikes.size > 0:
                            mask = (cell_spikes >= t_start) & (cell_spikes < t_end)
                            filtered = cell_spikes[mask]
                            if filtered.size > 0:
                                # Convert to bin coordinates relative to start
                                bin_coords = (filtered - t_start) / dt
                                spike_x_raster.extend(bin_coords * x_scale)
                                spike_y_raster.extend([current_row + cell_idx] * len(bin_coords))
            else:
                # Use robs for spike positions
                times, cells = np.where(spikes > 0)
                spike_x_raster.extend(times * x_scale)
                spike_y_raster.extend(current_row + cells)
            
            trial_info_raster.append((current_row + num_cells / 2, trial_number, 0))
            current_row += num_cells + gap
            trial_number += 1
    
    # Set cluster 1 start position
    current_row = psth_row_start + psth_space
    
    # Cluster 1 trials
    for i, trial_idx in enumerate(iix):
        cluster_id = clusters[i] if clusters is not None else 1
        if cluster_id == 1:
            spikes = robs_sliced[trial_idx, :, :]
            psth_data[1].append(np.nansum(spikes, axis=1))
            
            if use_spike_times and spike_times_trials is not None and trial_t_bins is not None:
                # Use spike times for more precise plotting
                t_bins = trial_t_bins[trial_idx]
                valid_mask = ~np.isnan(t_bins)
                if np.any(valid_mask):
                    valid_t_bins = t_bins[valid_mask]
                    t_start = valid_t_bins[0] - dt/2
                    t_end = valid_t_bins[-1] + dt/2
                    
                    for cell_idx in range(num_cells):
                        cell_spikes = np.atleast_1d(np.asarray(spike_times_trials[trial_idx][cell_idx]))
                        if cell_spikes.size > 0:
                            mask = (cell_spikes >= t_start) & (cell_spikes < t_end)
                            filtered = cell_spikes[mask]
                            if filtered.size > 0:
                                # Convert to bin coordinates relative to start
                                bin_coords = (filtered - t_start) / dt
                                spike_x_raster.extend(bin_coords * x_scale)
                                spike_y_raster.extend([current_row + cell_idx] * len(bin_coords))
            else:
                # Use robs for spike positions
                times, cells = np.where(spikes > 0)
                spike_x_raster.extend(times * x_scale)
                spike_y_raster.extend(current_row + cells)
            
            trial_info_raster.append((current_row + num_cells / 2, trial_number, 1))
            current_row += num_cells + gap
            trial_number += 1
    
    total_rows = psth_row_start + psth_space + max_cluster1_trials * (num_cells + gap) - gap
    
    # Plot spikes on raster
    if render == "line":
        spike_x_arr = np.asarray(spike_x_raster)
        plot_spikes_as_lines(ax_raster, spike_x_arr, spike_y_raster, height=0.1, 
                            color='k', linewidth=2.8, alpha=1.0)
    else:
        spike_x_plot = np.asarray(spike_x_raster)
        ax_raster.scatter(spike_x_plot, spike_y_raster, s=0.6, c='black', marker='|', linewidths=2)
    
    # Plot PSTHs if enabled
    if show_psth and psth_row_start is not None:
        psth0 = np.nanmean(psth_data[0], axis=0) if psth_data[0] else np.zeros(total_time)
        psth1 = np.nanmean(psth_data[1], axis=0) if psth_data[1] else np.zeros(total_time)
        max_psth = max(np.nanmax(psth0), np.nanmax(psth1)) + 1e-10
        
        offset0 = psth_row_start
        offset1 = psth_row_start + (psth_height + gap)
        
        psth0_scaled = offset0 + psth_height - (psth0 / max_psth) * psth_height
        psth1_scaled = offset1 + psth_height - (psth1 / max_psth) * psth_height
        
        x_vals = np.linspace(0, total_time * x_scale, len(psth0))
        ax_raster.fill_between(x_vals, offset0 + psth_height, psth0_scaled, color='blue', alpha=0.5)
        ax_raster.fill_between(x_vals, offset1 + psth_height, psth1_scaled, color='red', alpha=0.5)
    
    # Set raster axis limits and labels
    ax_raster.set_xlim(0, total_time * x_scale)
    ax_raster.set_ylim(total_rows, 0)
    ax_raster.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
    ax_raster.set_ylabel('Trial number')
    
    # Y-axis ticks
    tick_positions = [info[0] for info in trial_info_raster]
    tick_labels = [str(info[1]) for info in trial_info_raster]
    ax_raster.set_yticks(tick_positions)
    ax_raster.set_yticklabels(tick_labels)
    
    # Color tick labels by cluster
    cluster_colors_map = {0: 'blue', 1: 'red'}
    for tick_label, info in zip(ax_raster.get_yticklabels(), trial_info_raster):
        tick_label.set_color(cluster_colors_map[info[2]])
    
    # =========================================================================
    # Draw static eye trace plots on ax_eye_x and ax_eye_y
    # =========================================================================
    
    # Time values for x-axis
    if bins_x_axis:
        t_vals = np.arange(start_time_val, end_time_val)
    else:
        t_vals = (np.arange(start_time_val, end_time_val) / 240) * 1000
    
    # Plot ALL traces in gray first (background) if enabled
    if plot_all_traces:
        for trial_i in range(eyepos.shape[0]):
            if trial_i in iix:
                continue  # Skip trials that will be plotted in color
            eye_x_bg = eyepos[trial_i, start_time_val:end_time_val, 0]
            eye_y_bg = eyepos[trial_i, start_time_val:end_time_val, 1]
            valid_mask = ~(np.isnan(eye_x_bg) | np.isnan(eye_y_bg))
            if valid_mask.sum() > 0:
                ax_eye_x.plot(t_vals[valid_mask], eye_x_bg[valid_mask], color="gray", alpha=0.2, linewidth=0.5)
                ax_eye_y.plot(t_vals[valid_mask], eye_y_bg[valid_mask], color="gray", alpha=0.2, linewidth=0.5)
    
    # Plot eye traces for each trial (clustered only if show_unclustered_points=False)
    for i, trial_idx in enumerate(iix):
        # Skip unclustered points if show_unclustered_points=False
        if clusters is not None and not show_unclustered_points and clusters_arr[i] < 0:
            continue
        color = colors[i]
        eye_x = eyepos[trial_idx, start_time_val:end_time_val, 0]
        eye_y = eyepos[trial_idx, start_time_val:end_time_val, 1]
        ax_eye_x.plot(t_vals, eye_x, color=color, alpha=0.7, linewidth=0.7)
        ax_eye_y.plot(t_vals, eye_y, color=color, alpha=0.7, linewidth=0.7)
    
    # Set eye trace axis labels and limits
    ax_eye_x.set_ylabel('X (degrees)')
    ax_eye_y.set_ylabel('Y (degrees)')
    ax_eye_y.set_xlabel('Time (bins)' if bins_x_axis else 'Time (ms)')
    ax_eye_x.set_ylim(-1, 1)
    ax_eye_y.set_ylim(-1, 1)
    ax_eye_x.set_xlim(t_vals[0], t_vals[-1])
    plt.setp(ax_eye_x.get_xticklabels(), visible=False)
    
    # =========================================================================
    # Crop images to central 1 degree radius (2 degree diameter)
    # =========================================================================
    
    window_size_pixels = int(2 * image_extent * ppd)  # image_extent radius = 2*image_extent diameter
    img_h, img_w = rsvp_images.shape[1], rsvp_images.shape[2]
    cx, cy = img_h // 2, img_w // 2
    half = window_size_pixels // 2
    
    # Crop all images
    rsvp_images_cropped = rsvp_images[:, cx-half:cx+half, cy-half:cy+half]
    
    # Create a blank/gray image for when no image is shown (image_id = -1)
    blank_image = np.full((window_size_pixels, window_size_pixels), 127, dtype=np.float32)
    
    # =========================================================================
    # Initialize movable elements for animation
    # =========================================================================
    
    # Vertical lines for time marker
    vline_x_pos = t_vals[0]  # Initial position
    vline_raster = ax_raster.axvline(x=0, color='red', linestyle='--', linewidth=1.5, zorder=10)
    vline_eye_x = ax_eye_x.axvline(x=vline_x_pos, color='red', linestyle='--', linewidth=1.5, zorder=10)
    vline_eye_y = ax_eye_y.axvline(x=vline_x_pos, color='red', linestyle='--', linewidth=1.5, zorder=10)
    
    # Image display on ax_image
    # Use first valid image or blank as initial
    initial_img_id = int(image_ids[iix[0], start_time_val]) if image_ids[iix[0], start_time_val] >= 0 else -1
    img_extent = [-image_extent, image_extent, -image_extent, image_extent]
    if initial_img_id >= 0:
        im_display = ax_image.imshow(rsvp_images_cropped[initial_img_id], cmap='gray', 
                                      extent=img_extent, aspect='equal')
    else:
        im_display = ax_image.imshow(blank_image, cmap='gray', 
                                      extent=img_extent, aspect='equal')
    
    ax_image.set_xlabel('X (degrees)')
    ax_image.set_ylabel('Y (degrees)')
    ax_image.set_title('Eye Position on Image')
    
    # Eye trajectory trails on image (one line per trial + current position marker)
    # Only create trails for clustered trials if show_unclustered_points=False
    eye_trails = []
    eye_markers = []
    active_trial_indices = []  # Track which trials have active trails
    for i, trial_idx in enumerate(iix):
        # Skip unclustered points if show_unclustered_points=False
        if clusters is not None and not show_unclustered_points and clusters_arr[i] < 0:
            continue
        active_trial_indices.append((i, trial_idx))
        color = colors[i]
        # Trail line
        trail_line, = ax_image.plot([], [], color=color, linewidth=1.0, alpha=0.8)
        eye_trails.append(trail_line)
        # Current position marker
        marker, = ax_image.plot([], [], 'o', color=color, markersize=6, 
                                markeredgecolor='white', markeredgewidth=0.5)
        eye_markers.append(marker)
    
    # Set image axis limits to match image_extent
    ax_image.set_xlim(-image_extent, image_extent)
    ax_image.set_ylim(-image_extent, image_extent)
    
    # =========================================================================
    # Animation update function
    # =========================================================================
    
    # Reference trial for image_ids (use first trial)
    reference_trial = iix[0]
    
    def update(frame):
        """Update function for animation - called for each frame."""
        current_bin = start_time_val + frame
        
        # Convert current bin to x-axis position
        if bins_x_axis:
            x_pos_eyetrace = current_bin
            x_pos_raster = frame  # Raster uses relative position
        else:
            x_pos_eyetrace = (current_bin / 240) * 1000
            x_pos_raster = frame * x_scale
        
        # Update vertical lines
        vline_raster.set_xdata([x_pos_raster, x_pos_raster])
        vline_eye_x.set_xdata([x_pos_eyetrace, x_pos_eyetrace])
        vline_eye_y.set_xdata([x_pos_eyetrace, x_pos_eyetrace])
        
        # Update image based on image_ids
        img_id = int(image_ids[reference_trial, current_bin])
        if img_id >= 0 and img_id < len(rsvp_images_cropped):
            im_display.set_array(rsvp_images_cropped[img_id])
        else:
            im_display.set_array(blank_image)
        
        # Update eye trajectories (trailing history)
        trail_start = max(start_time_val, current_bin - trail_length)
        
        artists_to_return = [vline_raster, vline_eye_x, vline_eye_y, im_display]
        
        # Only update trails for active (clustered) trials
        for trail_idx, (i, trial_idx) in enumerate(active_trial_indices):
            # Get trail data
            trail_x = eyepos[trial_idx, trail_start:current_bin+1, 0]
            trail_y = eyepos[trial_idx, trail_start:current_bin+1, 1]
            
            # Filter out NaN values for valid trail
            valid_mask = ~(np.isnan(trail_x) | np.isnan(trail_y))
            trail_x_valid = trail_x[valid_mask]
            trail_y_valid = trail_y[valid_mask]
            
            # Update trail line
            eye_trails[trail_idx].set_data(trail_x_valid, trail_y_valid)
            
            # Update current position marker (last valid point)
            if len(trail_x_valid) > 0:
                eye_markers[trail_idx].set_data([trail_x_valid[-1]], [trail_y_valid[-1]])
            else:
                eye_markers[trail_idx].set_data([], [])
            
            artists_to_return.extend([eye_trails[trail_idx], eye_markers[trail_idx]])
        
        return artists_to_return
    
    # =========================================================================
    # Create and run animation
    # =========================================================================
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        # Create progress bar for saving
        pbar = tqdm(total=n_frames, desc="Rendering frames", unit="frame")
        
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=fig_dpi, 
                  progress_callback=progress_callback)
        pbar.close()
        print("Done saving.")
    
    if show:
        plt.show()
    
    return anim, fig


for j in range(return_top_k_combos)[1:2]:
    splice_start = 0
    splice_end = 1
    splice = slice(splice_start, splice_end)
    new_start_time_list = start_time_list[splice] - 30 #+6
    new_end_time_list = end_time_list[splice] + 30 #-2
    use_bins_x_axis = True
    new_robs_list = []
    vertical_line_pos = 40
    for i in range(splice_start, splice_end):
        new_robs_list.append(robs[:, new_start_time_list[i - splice_start]:new_end_time_list[i - splice_start], :])
    # fig1, ax1 = plot_eyepos_clusters(eyepos, iix_list[j][splice], start_time_list[splice], end_time_list[splice], clusters=clusters_list[j][splice], show=show, show_unclustered_points=False, plot_time_traces=True, bins_x_axis=False, use_peak_lag=False)
    fig1, ax1 = plot_eyepos_clusters_NEW(eyepos, iix_list[j][splice], new_start_time_list, new_end_time_list, clusters=clusters_list[j][splice], 
    show=show, show_unclustered_points=False, 
    plot_time_traces=True, bins_x_axis=use_bins_x_axis, use_peak_lag=False, plot_all_traces=True, vertical_line=vertical_line_pos, vertical_linewidth=3)
    # fig1.savefig(f"population_eyepos.pdf", dpi=1200, bbox_inches="tight")
    plt.show()
    plt.close(fig1)

    # fig1, ax1 = plot_eyepos_clusters_NEW(eyepos, iix_list[j][splice], new_start_time_list, new_end_time_list, clusters=clusters_list[j][splice], show=show, show_unclustered_points=False, plot_time_traces=False, bins_x_axis=use_bins_x_axis, use_peak_lag=False)
    # plt.show()
    # plt.close(fig1)

    cluster_list_new = [clusters_list[j][splice][0].copy()]
    counter0 = 0
    counter1 = 0
    for i in range(len(cluster_list_new[0])):
        if cluster_list_new[0][i] == 0:
            counter0 += 1
            if counter0 > 5:
                cluster_list_new[0][i] = -1
        if cluster_list_new[0][i] == 1:
            counter1 += 1
            if counter1 > 5:
                cluster_list_new[0][i] = -1
    
    # # Create time-sliced trial_t_bins to match robs_list
    # trial_t_bins_sliced = []
    # for i in range(splice_start, splice_end):
    #     start = start_time_list[i]
    #     end = end_time_list[i]
    #     # Slice trial_t_bins for each trial to match the time window
    #     sliced_t_bins = [trial_t_bins[trial_idx][start:end] for trial_idx in range(len(trial_t_bins))]
    #     trial_t_bins_sliced.append(sliced_t_bins)
    
    # # fig2, ax2, spike_x, spike_y, robs_clustered = plot_population_raster_NEW(robs_list[splice], iix_list[j][splice], cluster_list_new, start_time_list[splice], end_time_list[splice], 
    # # show_psth = True, show_difference_psth = False, show=show, render = "line",
    # # bins_x_axis=False)
    # fig2, ax2, spike_x, spike_y, robs_clustered = plot_population_raster_NEW(
    #     robs_list[splice], 
    #     iix_list[j][splice], 
    #     cluster_list_new, 
    #     start_time_list[splice], 
    #     end_time_list[splice], 
    #     show_psth=True, 
    #     show_difference_psth=False, 
    #     show=show, 
    #     render="line",
    #     bins_x_axis=False,
    #     # Spike times parameters
    #     use_spike_times=True,
    #     spike_times_trials=spike_times_trials,  # Full spike times (same trial indexing as robs)
    #     trial_t_bins=trial_t_bins_sliced,       # Time-sliced to match robs_list
    #     dt=1/240,
    # )

    trial_t_bins_sliced = []
    for i in range(splice_start, splice_end):
        start = new_start_time_list[i]
        end = new_end_time_list[i]
        # Slice trial_t_bins for each trial to match the time window
        sliced_t_bins = [trial_t_bins[trial_idx][start:end] for trial_idx in range(len(trial_t_bins))]
        trial_t_bins_sliced.append(sliced_t_bins)

    fig2, ax2, spike_x, spike_y, robs_clustered = plot_population_raster_NEW(
        new_robs_list[splice], 
        iix_list[j][splice], 
        cluster_list_new, 
        new_start_time_list, 
        new_end_time_list, 
        show_psth=True, 
        show_difference_psth=False, 
        show=show, 
        render="line",
        bins_x_axis=use_bins_x_axis,
        # Spike times parameters
        use_spike_times=True,
        spike_times_trials=spike_times_trials,  # Full spike times (same trial indexing as robs)
        trial_t_bins=trial_t_bins_sliced,       # Time-sliced to match robs_list
        dt=1/240,
        fig_width=10,
        fig_height=10,
        vertical_line=vertical_line_pos,
        vertical_linewidth=3,
    )
    # fig2, ax2, spike_x, spike_y = plot_population_raster(new_robs_list[splice], iix_list[j][splice], clusters_list[j][splice], new_start_time_list, new_end_time_list, 
    # show_psth = True, show_difference_psth = False, show=show, render = "img", fig_width = 5, fig_height = 40, fig_dpi = 400, gap= 50,
    # bins_x_axis=use_bins_x_axis, smooth_psth_sigma=0)
    fig2.savefig(f"population_raster2.pdf", dpi=400, bbox_inches="tight")
    plt.show()
    plt.close(fig2)
    
    # Example: Create animated movie
    # Get ppd from dataset metadata: ppd = dataset.dsets[0].metadata['ppd']
    ppd = dataset.dsets[0].metadata['ppd']  # pixels per degree (adjust based on your dataset)
    anim, fig_movie = population_plot_movie(
        robs=robs,
        eyepos=eyepos,
        iix=iix_list[j][splice][0],
        clusters=cluster_list_new[0],
        start_time=new_start_time_list[0],
        end_time=new_end_time_list[0],
        image_ids=image_ids,
        rsvp_images=rsvp_images,
        ppd=ppd,
        trail_length=5,  # Number of bins of eye trajectory history
        fps=10,
        save_path='population_movie.mp4',
        show=False,
        # Eye position parameters
        show_unclustered_points=False,  # Only show clustered trials
        plot_all_traces=True,  # Show all trials as gray background
        image_extent=0.8,  # Extent in degrees (radius from center)
        # Spike times parameters (same as plot_population_raster_NEW)
        use_spike_times=True,
        spike_times_trials=spike_times_trials,
        trial_t_bins=trial_t_bins_sliced[0],  # Use sliced t_bins for the time window
        dt=1/240,
        show_psth=True,
        bins_x_axis=use_bins_x_axis,
        render="line",
    )
    plt.close(fig_movie)
#%%
def plot_spikes_as_lines(ax, spike_x, spike_y, spike_vals=None, height=1.0, color="k", linewidth=0.5, alpha=1.0):
    """
    Plot spikes as vertical line segments with optional alpha variation based on spike values.
    
    Parameters:
    - ax: matplotlib axis
    - spike_x: array of x (time) positions
    - spike_y: array of y (row) positions
    - spike_vals: optional array of spike values for alpha variation
    - height: height of each line segment
    - color: line color
    - linewidth: line width
    - alpha: base alpha (modulated by spike_vals if provided)
    """
    spike_x = np.asarray(spike_x)
    spike_y = np.asarray(spike_y)
    
    if spike_x.size == 0:
        return None
    
    if spike_vals is None:
        # Simple case: all spikes same alpha
        x_lines = np.vstack([spike_x, spike_x, np.full(len(spike_x), np.nan)])
        y_lines = np.vstack([spike_y, spike_y + height, np.full(len(spike_y), np.nan)])
        return ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'), 
                       color=color, linewidth=linewidth, alpha=alpha, rasterized=True)[0]
    
    # Alpha varies by spike value
    spike_vals = np.asarray(spike_vals)
    unique_vals = np.unique(spike_vals)
    vmin, vmax = unique_vals[0], unique_vals[-1]
    handles = []
    
    for val in unique_vals:
        sel = spike_vals == val
        if not np.any(sel):
            continue
        if vmax > vmin:
            norm = (val - vmin) / (vmax - vmin)
            alpha_val = np.clip(0.5 + 0.9 * norm, 0.0, 1.0) * alpha
        else:
            alpha_val = alpha
        
        x_sel, y_sel = spike_x[sel], spike_y[sel]
        x_lines = np.vstack([x_sel, x_sel, np.full(sel.sum(), np.nan)])
        y_lines = np.vstack([y_sel, y_sel + height, np.full(sel.sum(), np.nan)])
        handles.append(
            ax.plot(x_lines.ravel(order='F'), y_lines.ravel(order='F'),
                    color=color, linewidth=linewidth, alpha=alpha_val, rasterized=True)[0]
        )
    return handles

def convert_binned_to_spike_arrays(raster_data, time_bins):
    """Convert binned spike data to (spike_x, spike_y, spike_vals) format."""
    mask = np.isfinite(raster_data) & (raster_data > 0)
    row_idx, col_idx = np.where(mask)
    if row_idx.size == 0:
        return np.array([]), np.array([]), np.array([])
    spike_x = time_bins[col_idx]
    spike_y = row_idx
    spike_vals = raster_data[row_idx, col_idx]
    return spike_x, spike_y, spike_vals

def convert_spike_times_to_arrays(spike_times_list, t_start, t_end):
    """Convert list of spike time arrays to (spike_x, spike_y) format."""
    spike_x_list = []
    spike_y_list = []
    for cell_idx, spike_times in enumerate(spike_times_list):
        if len(spike_times) == 0:
            continue
        mask = (spike_times >= t_start) & (spike_times <= t_end)
        cell_spikes = spike_times[mask]
        if len(cell_spikes) > 0:
            spike_x_list.append(cell_spikes)
            spike_y_list.append(np.full(len(cell_spikes), cell_idx))
    if len(spike_x_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(spike_x_list), np.concatenate(spike_y_list)


start_time_bin = 75
end_time_bin = 100
trial_ind = 11

# Plot binned spikes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Convert to relative time for both plots
t_start_abs = trial_time_windows[trial_ind][0]

# Binned version - convert to relative time
robs_slice = robs[trial_ind, start_time_bin:end_time_bin, :].T  # Transpose: cells x time
time_bins_abs = trial_t_bins[trial_ind][start_time_bin:end_time_bin]
time_bins_slice = time_bins_abs - t_start_abs  # Convert to relative time
spike_x_binned, spike_y_binned, spike_vals_binned = convert_binned_to_spike_arrays(robs_slice, time_bins_slice)
spike_vals_binned = None
plot_spikes_as_lines(ax1, spike_x_binned, spike_y_binned, spike_vals_binned, height=3, linewidth=1.5)
ax1.set_xlabel('Time (s, relative to trial start)')
ax1.set_ylabel('Cell')
ax1.set_title(f'Binned Spikes - Trial {trial_ind}')
ax1.invert_yaxis()
# Set xlim to start at 0 (or first bin) and remove margins
if len(time_bins_slice) > 0:
    ax1.set_xlim(left=0 if start_time_bin == 0 else time_bins_slice[0], right=time_bins_slice[-1])
ax1.margins(x=0)

# Spike times version
spike_times_relative = [np.array(cell_spikes) - t_start_abs 
                        for cell_spikes in spike_times_trials[trial_ind]]
dt = 1/240
t_start_rel = trial_t_bins[trial_ind][start_time_bin] - t_start_abs - dt/2  # left edge of first bin
t_end_rel = trial_t_bins[trial_ind][end_time_bin-1] - t_start_abs + dt/2    # right edge of last bin
spike_x_times, spike_y_times = convert_spike_times_to_arrays(spike_times_relative, t_start_rel, t_end_rel)
plot_spikes_as_lines(ax2, spike_x_times, spike_y_times, spike_vals=None, height=3, linewidth=1.5)
ax2.set_xlabel('Time (s, relative to trial start)')
ax2.set_ylabel('Cell')
ax2.set_title(f'Spike Times - Trial {trial_ind}')
ax2.invert_yaxis()
# Set xlim to start at 0 (or first time) and remove margins
ax2.set_xlim(left=0 if start_time_bin == 0 else t_start_rel, right=t_end_rel)
ax2.margins(x=0)

plt.tight_layout()
plt.show()
# %%

print(np.nansum(robs_slice), len(spike_x_times))

# %%
