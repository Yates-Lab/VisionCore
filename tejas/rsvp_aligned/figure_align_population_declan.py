#%%
import os
import sys
sys.path.append('..')
from pathlib import Path
from tkinter.constants import TRUE
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
from DataRowleyV1V2.data.registry import get_session as get_rowley_session
from DataYatesV1 import DictDataset
import matplotlib.patheffects as pe 
import contextlib

# Global flag for whether we have RF contour metrics
USE_RF_METRICS = False
DEFAULT_PEAK_LAG = 0  # Default peak lag when RF metrics unavailable



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

    # Use RF metrics if available, otherwise use default
    if USE_RF_METRICS and 'rf_contour_metrics' in globals():
        all_lags = []
        for i in range(len(rf_contour_metrics)):
            all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
        peak_lag = np.median(all_lags).astype(int)
    else:
        peak_lag = DEFAULT_PEAK_LAG

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
):
   
    # Use RF metrics if available, otherwise use default
    if USE_RF_METRICS and 'rf_contour_metrics' in globals():
        all_lags = []
        for i in range(len(rf_contour_metrics)):
            all_lags.append(rf_contour_metrics[i]['ste_peak_lag'])
        peak_lag = np.median(all_lags).astype(int)
    else:
        peak_lag = DEFAULT_PEAK_LAG

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
                t_vals = np.arange(start_time, end_time)
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
            ax_y.set_xlabel("Time (bins)")
            ax_x.set_title(f'{start_time} to {end_time} bins')

            ax_x.set_ylim(-0.6, 0.6)
            ax_y.set_ylim(-0.6, 0.6)
        else:
            ax.set_xlim(np.nanmin(eyepos[iix, start_time:end_time, 0]), np.nanmax(eyepos[iix, start_time:end_time, 0]))
            ax.set_ylim(np.nanmin(eyepos[iix, start_time:end_time, 1]), np.nanmax(eyepos[iix, start_time:end_time, 1]))
            ax.set_xlabel('X (degrees)')
            ax.set_ylabel('Y (degrees)')
            ax.set_title(f'{start_time} to {end_time} bins')
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
                t_vals = np.arange(st_adj, et_adj)
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
            ax.set_title(f'{st_adj} to {et_adj} bins')
            if plot_idx == 0:
                ax.set_ylabel('X (degrees)')
                axes[1, plot_idx].set_ylabel('Y (degrees)')
            axes[1, plot_idx].set_xlabel('Time (bins)')
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
    gap=100,
    show_psth=False,
    show_difference_psth=False,
    show=True,
    render="scatter",
    fig_width=None,
    fig_height=12,
    fig_dpi=400,
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
    psth_height = num_cells * 1.5  # Adjust multiplier to change PSTH height
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
        
        # Check if we have valid clusters
        has_valid_clusters = np.any(clusters >= 0)
        if not has_valid_clusters:
            print(f"Warning: No valid clusters found in this segment, assigning all trials to cluster 0")
            # Assign half to each cluster for visualization
            clusters = np.array([0 if i < len(iix)//2 else 1 for i in range(len(iix))])
        
        # Cluster 0 trials first (top)

        for i, trial_idx in enumerate(iix):
            if clusters[i] == 0:
                spikes = robs[trial_idx, :]  # [time, cells]
                seg_psth[0].append(np.nansum(spikes, axis=1))  # sum over cells
                # Find all spike positions - handle both continuous and discrete data
                if np.nanmax(spikes) < 0.1:  # Likely continuous firing rates
                    # Threshold for visualization
                    times, cells = np.where(spikes > 0.001)
                else:  # Discrete spike counts
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
                # Find all spike positions - handle both continuous and discrete data
                if np.nanmax(spikes) < 0.1:  # Likely continuous firing rates
                    times, cells = np.where(spikes > 0.001)
                else:  # Discrete spike counts
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
        return None, None, [], []
    if fig_width is None:
        fig_width = len(robs_list)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)

    if render == "img":
        ax.imshow(
            img,
            interpolation="nearest",
            cmap="gray_r",
            aspect="auto",
            vmin=0,
            vmax=1,
            extent=(0, total_time, total_rows, 0),
        )
    else:
        # Plot spikes as vertical ticks - linewidths controls horizontal thickness
        ax.scatter(spike_x, spike_y, s=0.6, c='black', marker='|', linewidths=2)
    
    # Plot PSTHs if enabled
    if show_psth and psth_row_start is not None:
        # Concatenate all segments
        psth0 = np.concatenate(psth_segments[0]) if psth_segments[0] else np.zeros(prev_total_time)
        psth1 = np.concatenate(psth_segments[1]) if psth_segments[1] else np.zeros(prev_total_time)
        max_psth = max(np.nanmax(psth0), np.nanmax(psth1)) + 1e-10
        
        # Normalize and scale to fit in psth_height (inverted y-axis)
        offset0 = psth_row_start
        offset_diff = psth_row_start + (psth_height + gap)
        offset1 = psth_row_start + (2 * (psth_height + gap) if show_difference_psth else (psth_height + gap))

        psth0_scaled = offset0 + psth_height - (psth0 / max_psth) * psth_height
        psth1_scaled = offset1 + psth_height - (psth1 / max_psth) * psth_height
        
        x_vals = np.arange(len(psth0))
        ax.fill_between(x_vals, offset0 + psth_height, psth0_scaled, color='blue', alpha=0.5)

        if show_difference_psth:
            diff_psth = np.abs(psth0 - psth1)
            diff_scaled = offset_diff + psth_height - (diff_psth / max_psth) * psth_height
            ax.fill_between(x_vals, offset_diff + psth_height, diff_scaled, color='k', alpha=0.25)

        ax.fill_between(x_vals, offset1 + psth_height, psth1_scaled, color='red', alpha=0.5)
    
    # Set axis limits
    # ax.set_xlim(0, end_time - start_time)
    ax.set_xlim(0, total_time)
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
        for seg_robs in robs_list:
            ax.axvline(x=t, color='red', linestyle='--')
            t += seg_robs.shape[1]
    
    ax.set_xlabel('Time (bins)')
    ax.set_ylabel('Trial number')
    plt.title(f'session {subject}_{date}')
    if show:
        plt.show()

    return fig, ax, spike_x, spike_y

#%%
# Load Rowley session instead of YatesV1
USE_ROWLEY_DATA = True  # Set to False to use original YatesV1 data

if USE_ROWLEY_DATA:
    # Load Rowley session
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
    
    # No RF contour metrics available for Rowley data
    USE_RF_METRICS = False
    rf_contour_metrics = None

# Clustering parameters (define before running analysis)
if USE_ROWLEY_DATA:
    # Relaxed constraints for Rowley data (fewer trials, shorter sessions)
    len_of_each_segment = 40
    total_start_time = 0
    total_end_time = 160  # Shorter to fit available data
    max_distance_from_centroid = 0.15  # More lenient
    num_clusters = 2
    min_cluster_size = 3  # Fewer trials required
    cluster_size = 3
    sort_by_cluster_psth = True
    distance_between_centroids = (0.05, 0.5)  # Wider range
    min_distance_between_inter_cluster_points = 0.01  # More lenient
    return_top_k_combos = 5  # Fewer combos to search
else:
    # Original constraints for YatesV1 data
    len_of_each_segment = 40
    total_start_time = 0
    total_end_time = 175
    max_distance_from_centroid = 0.1
    num_clusters = 2
    min_cluster_size = 4
    cluster_size = 4
    sort_by_cluster_psth = True
    distance_between_centroids = (0.02, 0.3)
    min_distance_between_inter_cluster_points = 0.02
    return_top_k_combos = 10

if not USE_ROWLEY_DATA:
    # Original YatesV1 loading code
    for session in get_complete_sessions():

        #['2022-03-02', '2022-04-06', '2022-04-13']
        if session.name.split('_')[0] != 'Allen'or session.name.split('_')[1] not in ['2022-03-02']:
            continue
        subject = session.name.split('_')[0]
        date = session.name.split('_')[1]
    
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

    from tejas.metrics.gaborium import get_rf_contour_metrics
    rf_contour_metrics = get_rf_contour_metrics(date, subject)
    USE_RF_METRICS = True

# Run clustering analysis (works for both Rowley and YatesV1 data)
robs_list = []
iix_list = [[] for _ in range(return_top_k_combos)]
clusters_list = [[] for _ in range(return_top_k_combos)]
start_time_list = []
end_time_list = []

print(f"\n{'='*60}")
print(f"Running clustering analysis")
print(f"  Time window: {total_start_time} to {total_end_time}")
print(f"  Segment length: {len_of_each_segment} bins")
print(f"  Max distance from centroid: {max_distance_from_centroid}")
print(f"  Distance between centroids: {distance_between_centroids}")
print(f"  Cluster size: {cluster_size}")
print(f"  Sort by PSTH: {sort_by_cluster_psth}")
print(f"{'='*60}\n")

for i in range(total_start_time, total_end_time, len_of_each_segment):
    start_time = i
    end_time = start_time + len_of_each_segment
    print(f"Processing time segment {start_time}-{end_time}...")
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


print(f"\n{'='*60}")
print(f"Clustering complete! Generating visualizations...")
print(f"{'='*60}\n")

show = False
for j in range(return_top_k_combos)[:2]:
    print(f"Generating plots for combo {j+1}...")
    fig1, ax1 = plot_eyepos_clusters(eyepos, iix_list[j], start_time_list, end_time_list, clusters=clusters_list[j], show=show)
#     fig1.savefig(f"population_figures/eyepos_{subject}_{date}_{j}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig1)
    fig2, ax2, spike_x, spike_y = plot_population_raster(robs_list, iix_list[j], clusters_list[j], show_psth = True, show_difference_psth = True, show=show)
    if fig2 is not None:
        pass  # fig2.savefig(...) would go here
#     fig2.savefig(f"population_figures/population_raster_{subject}_{date}_{j}.png", dpi=300, bbox_inches="tight")
#     plt.close(fig2)
    else:
        print(f"  Skipping raster plot for combo {j+1} (no spikes found)")

print(f"\n✓ Analysis complete!")
print(f"  Session: {subject}_{date}")
print(f"  Found {len(robs_list)} time segments")
print(f"  Generated {return_top_k_combos} clustering solutions")
    
#%%


for j in range(return_top_k_combos)[0:2]:
    splice = slice(1, 2)
    fig1, ax1 = plot_eyepos_clusters(eyepos, iix_list[j][splice], start_time_list[splice], end_time_list[splice], clusters=clusters_list[j][splice], show=show, show_unclustered_points=False, plot_time_traces=True)
    # fig1.savefig(f"population_figures/eyepos_{subject}_{date}_{j}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig1)
    
    
    fig2, ax2, spike_x, spike_y = plot_population_raster(robs_list[splice], iix_list[j][splice], clusters_list[j][splice], 
    show_psth = False, show_difference_psth = False, show=show, render = "img", fig_width = 5, fig_height = 40, fig_dpi = 400, gap= 50)
    # fig2.savefig(f"population_figures/population_raster_{subject}_{date}_{j}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig2)
#%%