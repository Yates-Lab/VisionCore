
"""
Plot Results from Saved Evaluation Data

This script loads the saved all_results dictionary and performs:
1. STA computation and visualization
2. Saccade modulation analysis and plotting

Based on example_cross_model_analysis_120.py but without running model evaluation.
"""

#%%
import sys
from pathlib import Path
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import zoom

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

import pickle
import torch
from tqdm import tqdm

# Import required functions
from DataYatesV1 import prepare_data
from eval_stack_multidataset import load_single_dataset, get_stim_inds
from extract_functions import extract_bps_saccade, extract_ccnorm, extract_qc_spatial

from DataYatesV1 import enable_autoreload

enable_autoreload()

# fig_dir = Path('/home/jake/repos/DataYatesV1/jake/multidataset_ddp/talk_figures')
fig_dir = Path('/home/tejas/Documents/fixational-transients/DataYatesV1/jake/multidataset_ddp/talk_figures_verify')
#%% utility for finding subpixel argmax / argmin

def _argext_subpixel(a, axis, mode):
    """
    Core implementation. Assumes `axis` is an int.
    Returns (idx_sub, val_sub) for the 1D slices along `axis`.
    """
    # For minima, flip the sign
    arr = -a if mode == 'min' else a
    # find integer peak
    idx = np.argmax(arr, axis=axis)
    # gather peak values
    idx_exp = np.expand_dims(idx, axis)
    v0 = np.take_along_axis(arr, idx_exp, axis=axis).squeeze(axis)

    # neighbor integer indices
    N = a.shape[axis]
    idx_m = np.clip(idx - 1, 0, N - 1)
    idx_p = np.clip(idx + 1, 0, N - 1)
    v_m = np.take_along_axis(arr, np.expand_dims(idx_m, axis), axis=axis).squeeze(axis)
    v_p = np.take_along_axis(arr, np.expand_dims(idx_p, axis), axis=axis).squeeze(axis)

    # parabolic interpolation formula:
    #   offset = (v_- - v_+) / (2*(v_- - 2*v0 + v_+))
    #   v_sub = v0 - (v_- - v_+)² / [8*(v_- - 2*v0 + v_+)]
    denom = (v_m - 2*v0 + v_p)
    num   = (v_m - v_p)

    # compute offset safely
    with np.errstate(divide='ignore', invalid='ignore'):
        offset = num / (2 * denom)
    # clamp offsets at boundaries or flat peaks
    at_edge = (idx == 0) | (idx == N-1) | (denom == 0)
    offset = np.where(at_edge, 0.0, offset)

    # subpixel indices & values
    idx_sub = idx.astype(np.float64) + offset
    v_sub = v0 - (num**2) / (8 * denom)
    v_sub = np.where(at_edge, v0, v_sub)

    # for minima, flip back the sign
    if mode == 'min':
        v_sub = -v_sub

    return idx_sub, v_sub


def _argext_dispatch(a, axis, mode):
    a = np.asanyarray(a)
    # axis=None: flatten
    if axis is None:
        flat = a.ravel()
        idx_sub, v_sub = _argext_subpixel(flat, 0, mode)
        return idx_sub, v_sub

    # normalize negative axes
    axis = axis if axis >= 0 else a.ndim + axis
    return _argext_subpixel(a, axis, mode)


def argmax_subpixel(a, axis=None):
    """
    Like np.argmax, but returns sub-pixel refined peak and its value.
    
    Parameters
    ----------
    a : array_like
    axis : int or None, optional
        Axis along which to find the maximum. Default is None (flattened).

    Returns
    -------
    idx_sub : float or ndarray of floats
        Sub-pixel estimate of the index of the maximum.
    val_sub : float or ndarray of floats
        Interpolated maximum value.
    """
    return _argext_dispatch(a, axis, mode='max')


def argmin_subpixel(a, axis=None):
    """
    Like np.argmin, but returns sub-pixel refined trough and its value.
    
    Parameters
    ----------
    a : array_like
    axis : int or None, optional
        Axis along which to find the minimum. Default is None (flattened).

    Returns
    -------
    idx_sub : float or ndarray of floats
        Sub-pixel estimate of the index of the minimum.
    val_sub : float or ndarray of floats
        Interpolated minimum value.
    """
    return _argext_dispatch(a, axis, mode='min')

'''
Other utilities
'''

from scipy.stats import mannwhitneyu
from itertools import combinations

def custom_boxplot(
    data_list,
    labels=None,
    colors=None,  # list or colormap name
    width=0.6,
    median_color='#D62728',
    shade_low='#D3D3D3',     # min→Q1 & Q3→max
    shade_mid='#A9A9A9',     # Q1→median
    shade_high='#808080',    # median→Q3
    test='mannwhitney',
    alpha=0.05,
    figsize=(8,5),
    title=None,
    xlabel=None,
    ylabel=None,
    sig_spacing_factor=1.2  # increase spacing between significance bars
):
    """
    Draws a boxplot matching the exact style from the provided figure,
    without stems (whiskers/caps) since shaded regions mark the range.
    """
    n = len(data_list)
    if labels is None:
        labels = [f'Group {i+1}' for i in range(n)]
    # handle colors if colormap name passed
    if isinstance(colors, str) and colors in plt.colormaps():
        cmap = plt.get_cmap(colors)
        colors = [cmap(i/(n-1)) for i in range(n)]
    
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(1, n+1)
    
    # Collect stats
    stats = []
    for data in data_list:
        clean = data[~np.isnan(data)]
        mn, q1, med, q3, mx = np.percentile(clean, [10, 25, 50, 75, 90])
        # mn, mx = clean.min(), clean.max()
        stats.append((mn, q1, med, q3, mx))
    
    # Draw each box by shading quartiles
    for i, (mn, q1, med, q3, mx) in enumerate(stats):
        x = positions[i]
        left, right = x - width/2, x + width/2
        
        # # min to Q1
        ax.fill_between([left, right], [mn, mn], [q1, q1], color=shade_low)
        # Q1 to median
        ax.fill_between([left, right], [q1, q1], [med, med], color=shade_mid)
        # median to Q3
        ax.fill_between([left, right], [med, med], [q3, q3], color=shade_high)
        # # Q3 to max
        ax.fill_between([left, right], [q3, q3], [mx, mx], color=shade_low)
        
        # median line
        ax.hlines(med, left, right, color=median_color, linewidth=2)
    
    # X-axis
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    if xlabel: ax.set_xlabel(xlabel, fontsize=14)
    if ylabel: ax.set_ylabel(ylabel, fontsize=14)
    if title: ax.set_title(title, fontsize=16)
    
    # Optional statistical comparisons
    ymax = max(mx for mn,q1,med,q3,mx in stats)
    ymin = min(mn for mn,q1,med,q3,mx in stats)
    yrange = ymax - ymin
    step = yrange * 0.05 * sig_spacing_factor  # apply spacing factor
    pairs = list(combinations(range(n),2))
    for idx, (i,j) in enumerate(pairs):
        a = data_list[i][~np.isnan(data_list[i])]
        b = data_list[j][~np.isnan(data_list[j])]
        if test=='mannwhitney':
            stat, p = mannwhitneyu(a,b,alternative='two-sided')
        else:
            raise ValueError("Unsupported test")
        if p < alpha:
            x1, x2 = positions[i], positions[j]
            y = ymax + step*(idx+1)
            ax.plot([x1, x1, x2, x2], [y-step*0.2, y, y, y-step*0.2], c='k', lw=1.2)
            stars = '*' if p>0.01 else '**' if p>0.001 else '***'
            ax.text((x1+x2)/2, y+step*0.02, stars, ha='center', va='bottom', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig, ax

def get_model_name(model_name):

    if 'learned_res_small_gru' in model_name:
        return 'Modulator'
    elif 'learned_res_small_pc' in model_name:
        return 'PC'
    elif 'learned_res_small_film' in model_name:
        return 'FiLM'
    elif 'learned_res_small_stn' in model_name:
        return 'STN'
    elif 'learned_res_small' in model_name:
        return 'Vision'
    else:
        return model_name

from eval_stack_multidataset import ccnorm_variable_trials

def get_ccnorm(all_results, model_name):
    # bins 
    bin_inds = np.arange(40, 148)
    ccnorm = {'rbar': [], 'rbarhat': [], 'rbarhat_range': [], 'ccnorm': [], 'ccnorm_orig': []}

    for dset in range(len(all_results[model_name]['ccnorm']['fixrsvp']['robs_trial'])):
        
        rbar_ = all_results[model_name]['ccnorm']['fixrsvp']['rbar'][dset]

        robs = all_results[model_name]['ccnorm']['fixrsvp']['robs_trial'][dset][:,bin_inds,:]
        rhat = all_results[model_name]['ccnorm']['fixrsvp']['rhat_trial'][dset][:,bin_inds,:]
        dfs  = all_results[model_name]['ccnorm']['fixrsvp']['dfs_trial'][dset][:,bin_inds,:]
        dfs = ((dfs == 1) & ~np.isnan(robs)).astype(np.float32)

        robs[np.isnan(robs)] = 0
        rhat[np.isnan(rhat)] = 0

        # good_trials = np.where(np.sum(dfs==1, 1) > 80)[0]
        # print(f"Number of good trials: {len(good_trials)}")

        # robs = robs[good_trials]
        # rhat = rhat[good_trials]
        # dfs = dfs[good_trials]

        rbar = np.sum(robs * dfs, 0) / np.sum(dfs, 0)
        rbarhat = np.sum(rhat * dfs, 0) / np.sum(dfs, 0)
        rbarhat_range = np.zeros((2, robs.shape[1], robs.shape[2]))
        for cid in range(robs.shape[-1]):
            good_trials = np.where(np.sum(dfs[:, :, cid], 1) > 10)[0]
            if len(good_trials) < 10:
                continue
            rbarhat_range[:, :, cid] = np.percentile(rhat[good_trials, :, cid], [10, 90], axis=0)

        ccn = ccnorm_variable_trials(robs, rhat, dfs, min_trials_per_bin=10, min_time_bins=10)

        ccnorm['rbar'].append(rbar)
        ccnorm['rbarhat'].append(rbarhat)
        ccnorm['rbarhat_range'].append(rbarhat_range)
        ccnorm['ccnorm'].append(ccn)
        ccnorm['ccnorm_orig'].append(all_results[model_name]['ccnorm']['fixrsvp']['ccnorm'][dset])
    
    rbar = np.concatenate(ccnorm['rbar'], 1)
    rhat = np.concatenate(ccnorm['rbarhat'], 1)
    rbar_hat_range = np.concatenate(ccnorm['rbarhat_range'], 2)
    ccn = np.minimum(np.maximum(np.concatenate(ccnorm['ccnorm'], 0), -.1), 1.0)
    ccn_orig = np.concatenate(ccnorm['ccnorm_orig'], 0)

    ccnorm = {'rbar': rbar, 'rbarhat': rhat, 'rbarhat_range': rbar_hat_range, 'ccnorm': ccn, 'ccnorm_orig': ccn_orig}
    return ccnorm

def get_Rnorm(R, tbins):
    from scipy.ndimage import gaussian_filter1d

    R = gaussian_filter1d(R, 1, axis=0)
    # R = savgol_filter(R, 21, 3, axis=0)
    R0 = R[tbins < 0].mean(0)
    R = (R - R0)/R.max(0)

    return R

def poisson_loss(pred_rate, target_counts, eps=1e-8, reduction='mean'):
    """
    pred_rate: tensor of shape [batch, ...], predicted λ >= 0
    target_counts: same shape, integer spike counts r
    """
    # ensure numerical stability
    rate = pred_rate.clamp(min=eps)
    # loss = λ - r * log λ
    loss_per_entry = rate - target_counts * torch.log(rate)
    if reduction == 'mean':
        return loss_per_entry.mean()
    elif reduction == 'sum':
        return loss_per_entry.sum()
    else:
        return loss_per_entry  # no reduction

#%% Load saved results

print("🔍 Loading saved evaluation results...")
# save_path = Path('all_results_120_analysis.pkl')
save_path = Path('/home/tejas/Documents/fixational-transients/DataYatesV1/jake/multidataset_ddp/all_results_120_analysis_incremental.pkl')

if not save_path.exists():
    print(f"❌ Results file not found: {save_path.absolute()}")
    print("Please run example_cross_model_analysis_120.py first to generate the results.")
    sys.exit(1)

try:
    with open(save_path, 'rb') as f:
        all_results = pickle.load(f)
    print(f"✅ Successfully loaded results for {len(all_results)} models")
    for model_name in all_results.keys():
        n_cells = len(all_results[model_name]['qc']['all_cids'])
        print(f"  - {model_name}: {n_cells} cells")
except Exception as e:
    print(f"❌ Failed to load results: {e}")
    sys.exit(1)

gratings_save_path = Path('gratings_results_120.pkl')
if gratings_save_path.exists():
    with open(gratings_save_path, 'rb') as f:
        gratings_results = pickle.load(f)
    print(f"✅ Successfully loaded gratings results for {len(gratings_results)} models")
else:
    print(f"❌ Gratings results file not found: {gratings_save_path.absolute()}")

#%% Load model
from eval_stack_multidataset import load_model

model_name = list(all_results.keys())[1] # model 1

model_type = model_name.split('_pretrained_')[0]
print(f"Loading model {model_type}...")

model, model_info = load_model(
        model_type=model_type,
        model_index=None,
        checkpoint_path=None,
        checkpoint_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints",
        device='cpu'
    )

model.eval()


#%%
plt.plot(model.model.frontend.temporal_conv.weight.squeeze().detach().cpu().T)
plt.title(model_name.split('_ddp')[0])
plt.box(False)
plt.xlabel('Time lag')
plt.ylabel('Temporal Layer Weights')
# plt.savefig(fig_dir / 'temporal_weights.pdf')




#%% Get QC Data for filtering

# Get the first two models for comparison
model_names = list(all_results.keys())
if len(model_names) < 2:
    print("⚠️ Need at least 2 models for comparison. Adding duplicate for demo.")
    model_names = [model_names[0], model_names[0]]

print(f"Comparing models:")

for i, name in enumerate(model_names):
    print(f"  {i}: {name.split('_ddp_')[0]}")
    
model_name = model_names[0]  # Use first model for QC data
_, waveforms, cids, dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results)
from classify_layer import get_depths, layer_splits, plot_laminar_boundaries
depths = get_depths({'dummy':all_results}, 'Allen_2022-02-16', concatenate=True)
splits = layer_splits['Allen_2022-02-16']
depth = depths[1]
# print(f"\nQC data: {len(depth)} cells, contamination range: {np.nanmin(contamination):.1f}-{np.nanmax(contamination):.1f}%")

# get waveform data and use to find narrow, broad, and axonal units
wfs = []
for i in range(len(waveforms)):
    ch = np.argmax(np.var(waveforms[i], 0))
    wfs.append(waveforms[i].T[ch])

wfs = np.stack(wfs).T
wfs = wfs / np.max(np.abs(wfs), 0, keepdims=True)

start = 39 # start finding the peak and trough after the trigger
end = 70
tau_trough, val_trough = argmin_subpixel(wfs[start:end], 0)
tau_peak, val_peak = argmax_subpixel(wfs[start:end], 0)

#% plot a few examples to check
n_examples = 10
sx = int(np.ceil(np.sqrt(n_examples)))
sy = int(np.ceil(n_examples / sx))
fig, axes = plt.subplots(sx, sy, figsize=(16, 16))
for i, ax in enumerate(axes.flatten()):
    if i >= n_examples:
        ax.axis('off')
        continue
    j = np.random.randint(wfs.shape[1])
    ax.plot(wfs[:,j])
    ax.axvline(tau_trough[j]+start, color='k')
    ax.axvline(tau_peak[j]+start, color='r')
    ax.set_title(f'Cell {j}')
    ax.axis('off')


#%% STA Latency

ste_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_robs'], -1) / np.concatenate(all_results[list(all_results.keys())[1]]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[0]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[list(all_results.keys())[0]]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat0 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[list(all_results.keys())[1]]['sta']['norm_dfs'], -1)[None,None,None,:])

lag_robs, _ = argmax_subpixel(ste_robs.var((1,2)), 0)
lag_rhat0, _ = argmax_subpixel(ste_rhat0.var((1,2)), 0)
lag_rhat1, _ = argmax_subpixel(ste_rhat1.var((1,2)), 0)

fig, ax = plt.subplots(1,3, figsize=(10, 7), sharey=True, sharex=True)

dt = 1000/120
iix = contamination < 100
ax[0].scatter(lag_robs[iix]*dt, -depth[iix], s=20, facecolor='k', alpha = .25, edgecolor='w')
ax[1].scatter(lag_rhat0[iix]*dt, -depth[iix], s=20, facecolor='k', alpha = .25, edgecolor='w')
ax[2].scatter(lag_rhat1[iix]*dt, -depth[iix], s=20, facecolor='k', alpha = .25, edgecolor='w')
ax[1].set_xlim(10, 60)

ax[0].axhline(150, color='k', linestyle='--')
ax[1].axhline(150, color='k', linestyle='--')
ax[2].axhline(150, color='k', linestyle='--')
ax[0].axhline(-150, color='k', linestyle='--')
ax[1].axhline(-150, color='k', linestyle='--')
ax[2].axhline(-150, color='k', linestyle='--')

ax[0].set_title('Data')
ax[1].set_title('Vision')
ax[2].set_title('Modulator')

ax[0].set_xlabel('Stimulus Latency')
ax[0].set_ylabel('Depth (um)')

# plt.savefig(fig_dir / 'sta_latency.pdf')

#%% STA/STE Latency (single dataset)

# Choose dataset(s) by index (int) or name (str) to plot STA/STE latency. Can be single value or list.
# Examples:
#   selected_dataset = 0                              # single dataset by index
#   selected_dataset = "yates_2021_07_15"            # single dataset by name  
#   selected_dataset = [0, 1, 2]                     # multiple datasets by index
#   selected_dataset = ["yates_2021_07_15", "Allen_2022-03-04"]  # multiple datasets by name
#   selected_dataset = [0, "yates_2021_07_15"]       # mixed types
# selected_dataset = [2, 4, 12]
# selected_dataset = [2, 4, 6, 8, 10, 12, 13]
selected_dataset =  list(range(20))
target_size = 40
size = (14, 25)
sta_or_ste = 'sta'
# Choose SNR measure for filtering STAs/STEs
snr_measure = 'peak_to_noise'  # Options: 'variance', 'peak_to_noise', 'energy', 'temporal_consistency'
# Toggle between 'sta' and 'ste' by changing the value above

# SNR visualization options
show_dataset_colors = True  # Set to False to show all data in one color

# SNR filtering threshold
snr_threshold = 8  # Set to a number (e.g., 5.0) to only show cells above this SNR threshold
# Examples: snr_threshold = 5.0, snr_threshold = 10.0, snr_threshold = None (show all)

# SNR measure descriptions:
# 'variance': spatial variance / total variance (simple, interpretable)
# 'peak_to_noise': max(abs(STA)) / std(STA) (robust, captures strong responses)
# 'energy': L2 norm of STA (overall response strength)
# 'temporal_consistency': variance at peak lag / variance across lags
# sta_or_ste = 'ste'  # Uncomment this line to use STE instead of STA

# Note: STA/STE data is now properly normalized by subtracting the stimulus mean
# This fixes the reddish tint issue by centering the data around zero
#2, 4, 6, 8, 10, 12, 13

# Use the same model as above (model index 1) for consistency
model_key_sta = list(all_results.keys())[1]

# Access STA/STE structures
sta_struct = all_results[model_key_sta]['sta']
ste_robs_list = sta_struct['ste_robs']          # list over datasets: [lags, H, W, N_d]
sta_robs_list = sta_struct['sta_robs']          # list over datasets: [lags, H, W, N_d]
norm_dfs_list = sta_struct['norm_dfs']          # list over datasets: [N_d]
global_cids = np.array(sta_struct.get('cids', []))           # length sum(N_d)
global_datasets_cells = np.array(sta_struct.get('datasets', []))  # length sum(N_d)

# Select data based on sta_or_ste setting
if sta_or_ste == 'sta':
    robs_list = sta_robs_list
    print(f"✓ Using STA data")
else:
    robs_list = ste_robs_list
    print(f"✓ Using STE data")

# Build per-dataset block sizes and offsets to reconstruct cids per dataset
block_sizes = [arr.shape[-1] for arr in robs_list]
offsets = np.cumsum([0] + block_sizes[:-1]) if len(block_sizes) > 0 else np.array([0])

# Infer block names from global datasets-per-cell if available
block_names = []
if len(global_datasets_cells) == sum(block_sizes):
    for i, start in enumerate(offsets):
        block_names.append(global_datasets_cells[start])
else:
    # Fallback: use unique values from QC dids in the order they appear
    block_names = list(dict.fromkeys(dids))[:len(block_sizes)]

# Normalize selected_dataset to a list
if not isinstance(selected_dataset, (list, tuple)):
    selected_datasets = [selected_dataset]
else:
    selected_datasets = list(selected_dataset)

# Resolve selected dataset indices
dset_indices = []
dataset_names = []
for sel_dset in selected_datasets:
    if isinstance(sel_dset, str):
        if sel_dset in block_names:
            dset_idx = block_names.index(sel_dset)
            dset_indices.append(dset_idx)
            dataset_names.append(sel_dset)
        else:
            print(f"⚠️ Dataset '{sel_dset}' not found. Available blocks: {block_names}. Skipping.")
    else:
        dset_idx = int(sel_dset)
        # Safety clamp
        if dset_idx < 0 or dset_idx >= len(robs_list):
            print(f"⚠️ Dataset index {dset_idx} out of range [0, {len(robs_list)-1}]. Skipping.")
        else:
            dset_indices.append(dset_idx)
            dataset_names.append(block_names[dset_idx] if len(block_names) > dset_idx else f"dataset_{dset_idx}")

if not dset_indices:
    print("⚠️ No valid datasets found. Exiting.")
else:
    print(f"✓ Processing {len(dset_indices)} dataset(s): {dataset_names}")

    # Build QC mapping (did, cid) -> (depth, contamination)
    qc_map = {}
    for di, cid in enumerate(cids):
        qc_map[(dids[di], cid)] = (depth[di], contamination[di])

    # Combine data from all selected datasets
    all_lag_robs = []
    all_depth = []
    all_contam = []
    all_robs = []
    all_dataset_labels = []

    for i, dset_idx in enumerate(dset_indices):
        # Per-dataset arrays normalized by per-dataset norm_dfs
        norm_dfs_raw = norm_dfs_list[dset_idx]
        norm_dfs_safe = np.where(norm_dfs_raw == 0, np.nan, norm_dfs_raw)
        norm_dfs_d = norm_dfs_safe[None, None, None, :]
        
        # Get the raw STA/STE data
        raw_robs = robs_list[dset_idx]
        
        # Calculate the mean across all spatial dimensions and lags for this dataset
        stim_mean = raw_robs.mean(axis=(0, 1, 2), keepdims=True)
        
        # Center the data by subtracting the mean (proper STA normalization)
        centered_robs = raw_robs - stim_mean
        
        # Then normalize by spike counts
        robs_d = torch.from_numpy(centered_robs / norm_dfs_d)

        # Compute lags (per-cell) for data only
        lag_robs_d, _ = argmax_subpixel(robs_d.var((1, 2)), 0)

        # Reconstruct this block's CIDs and dataset name
        start = offsets[dset_idx]
        end = start + block_sizes[dset_idx]
        block_cids = global_cids[start:end] if len(global_cids) == sum(block_sizes) else None
        block_name = dataset_names[i]

        # Derive depth/contamination for this dataset block in the correct order
        depth_d_list = []
        contam_d_list = []
        if block_cids is not None and block_name is not None:
            for cid in block_cids:
                depth_val, contam_val = qc_map.get((block_name, cid), (np.nan, np.nan))
                depth_d_list.append(depth_val)
                contam_d_list.append(contam_val)
        else:
            # Fallback: use all QC units from this dataset (order may not match exactly)
            mask_d = (np.array(dids) == block_name) if block_name is not None else np.zeros_like(depth, dtype=bool)
            depth_d_list = list(depth[mask_d])[:block_sizes[dset_idx]]
            contam_d_list = list(contamination[mask_d])[:block_sizes[dset_idx]]

        depth_d = np.asarray(depth_d_list)
        contam_d = np.asarray(contam_d_list)

        # Store data for this dataset
        n_cells = min(len(depth_d), lag_robs_d.shape[0])
        if n_cells > 0:
            # Ensure lag_robs_d is a tensor before appending
            if isinstance(lag_robs_d, np.ndarray):
                lag_robs_d = torch.from_numpy(lag_robs_d)
            all_lag_robs.append(lag_robs_d[:n_cells])
            all_depth.append(depth_d[:n_cells])
            all_contam.append(contam_d[:n_cells])
            all_robs.append(robs_d[:, :, :, :n_cells])
            all_dataset_labels.extend([block_name] * n_cells)

    # Concatenate all data
    if all_lag_robs:
        lag_robs_combined = torch.cat(all_lag_robs, dim=0)
        depth_combined = np.concatenate(all_depth)
        contam_combined = np.concatenate(all_contam)
        robs_combined = torch.cat(all_robs, dim=-1)
        
        # Calculate SNR for each cell based on selected measure
        print(f"✓ Calculating SNR using measure: {snr_measure}")
        if snr_measure == 'variance':
            # Spatial variance / total variance
            spatial_var = robs_combined.var(dim=(1, 2))  # [n_lags, n_cells]
            total_var = robs_combined.var(dim=(0, 1, 2))  # [n_cells]
            # Add small epsilon to avoid division by zero
            snr_values = spatial_var.mean(dim=0) / (total_var + 1e-8)  # [n_cells]
            
        elif snr_measure == 'peak_to_noise':
            # Peak response / noise estimate
            # Use torch.amax which accepts tuple dimensions
            peak_responses = robs_combined.abs().amax(dim=(0, 1, 2))  # [n_cells]
            
            # Debug: check for NaN/inf in peak responses
            if torch.isnan(peak_responses).any() or torch.isinf(peak_responses).any():
                print(f"⚠️ Found NaN/inf in peak responses: {torch.isnan(peak_responses).sum()} NaNs, {torch.isinf(peak_responses).sum()} infs")
            
            # Use median absolute deviation as robust noise estimate
            # Flatten spatial and temporal dimensions for median operations
            robs_flat = robs_combined.view(-1, robs_combined.shape[-1])  # [n_lags*H*W, n_cells]
            median_vals = torch.median(robs_flat, dim=0, keepdim=True)[0]  # [n_cells]
            mad = torch.median(torch.abs(robs_flat - median_vals), dim=0)[0]  # [n_cells]
            
            # Debug: check for NaN/inf in MAD
            if torch.isnan(mad).any() or torch.isinf(mad).any():
                print(f"⚠️ Found NaN/inf in MAD: {torch.isnan(mad).sum()} NaNs, {torch.isinf(mad).sum()} infs")
            
            # Add small epsilon to avoid division by zero and handle edge cases
            snr_values = peak_responses / (mad + 1e-8)  # [n_cells]
            
            # Debug: check for NaN/inf in final SNR
            if torch.isnan(snr_values).any() or torch.isinf(snr_values).any():
                print(f"⚠️ Found NaN/inf in SNR: {torch.isnan(snr_values).sum()} NaNs, {torch.isinf(snr_values).sum()} infs")
            
        elif snr_measure == 'energy':
            # L2 norm of STA
            # torch.norm doesn't accept tuple dimensions, need to flatten first
            robs_flat = robs_combined.view(-1, robs_combined.shape[-1])  # [n_lags*H*W, n_cells]
            snr_values = torch.norm(robs_flat, dim=0)  # [n_cells]
            
        elif snr_measure == 'temporal_consistency':
            # Variance at peak lag / variance across lags
            spatial_var = robs_combined.var(dim=(1, 2))  # [n_lags, n_cells]
            peak_lag_var = spatial_var.max(dim=0)[0]  # [n_cells]
            across_lag_var = spatial_var.var(dim=0)  # [n_cells]
            # Add small epsilon to avoid division by zero
            snr_values = peak_lag_var / (across_lag_var + 1e-8)  # [n_cells]
            
        else:
            print(f"⚠️ Unknown SNR measure '{snr_measure}', using peak_to_noise")
            # Default to peak_to_noise
            peak_responses = robs_combined.abs().amax(dim=(0, 1, 2))  # [n_cells]
            # Flatten spatial and temporal dimensions for median operations
            robs_flat = robs_combined.view(-1, robs_combined.shape[-1])  # [n_lags*H*W, n_cells]
            median_vals = torch.median(robs_flat, dim=0, keepdim=True)[0]  # [n_cells]
            mad = torch.median(torch.abs(robs_flat - median_vals), dim=0)[0]  # [n_cells]
            # Add small epsilon to avoid division by zero and handle edge cases
            snr_values = peak_responses / (mad + 1e-8)
        
        # Convert to numpy and handle any remaining NaNs
        snr_values = snr_values.detach().cpu().numpy()
        
        # Check for and handle NaN values
        nan_mask = np.isnan(snr_values)
        if nan_mask.any():
            print(f"⚠️ Found {nan_mask.sum()} NaN values in SNR calculations")
            # Replace NaNs with a small value or median of non-NaN values
            non_nan_values = snr_values[~nan_mask]
            if len(non_nan_values) > 0:
                replacement_value = np.median(non_nan_values)
                snr_values[nan_mask] = replacement_value
                print(f"✓ Replaced NaNs with median value: {replacement_value:.3f}")
            else:
                print("⚠️ All SNR values are NaN, setting to 1.0")
                snr_values = np.ones_like(snr_values)
        
        # Check for infinite values
        inf_mask = np.isinf(snr_values)
        if inf_mask.any():
            print(f"⚠️ Found {inf_mask.sum()} infinite values in SNR calculations")
            # Replace infinities with a large finite value
            finite_values = snr_values[~inf_mask]
            if len(finite_values) > 0:
                replacement_value = np.percentile(finite_values, 95)
                snr_values[inf_mask] = replacement_value
                print(f"✓ Replaced infinities with 95th percentile: {replacement_value:.3f}")
            else:
                print("⚠️ All SNR values are infinite, setting to 1.0")
                snr_values = np.ones_like(snr_values)
        
        print(f"✓ SNR range: {snr_values.min():.3f} to {snr_values.max():.3f}")
        print(f"✓ SNR mean: {np.mean(snr_values):.3f}, std: {np.std(snr_values):.3f}")
        
        # Calculate threshold statistics and filter data for other plots
        if snr_threshold is not None:
            # Create mask for cells above threshold
            above_threshold_mask = snr_values >= snr_threshold
            n_above_threshold = above_threshold_mask.sum()
            n_total = len(snr_values)
            
            print(f"✓ SNR threshold: {snr_threshold:.3f}")
            print(f"✓ Cells above threshold: {n_above_threshold}/{n_total} ({100*n_above_threshold/n_total:.1f}%)")
            
            # Filter data for other plots (latency vs depth, spatial STA images)
            lag_robs_filtered = lag_robs_combined[above_threshold_mask]
            depth_filtered = depth_combined[above_threshold_mask]
            contam_filtered = contam_combined[above_threshold_mask]
            robs_filtered = robs_combined[:, :, :, above_threshold_mask]
            
            # Update arrays for other plots (but keep original snr_values for histogram)
            lag_robs_combined = lag_robs_filtered
            depth_combined = depth_filtered
            contam_combined = contam_filtered
            robs_combined = robs_filtered
            
            print(f"✓ Filtered other plots: {len(lag_robs_combined)} cells above threshold")
        else:
            print("✓ No SNR threshold applied - showing all cells")
        
        # Prepare dataset labels for colored histogram if requested
        if show_dataset_colors:
            # Create dataset labels array that matches the cell order
            # all_dataset_labels is a list of strings, not arrays, so we need to handle it differently
            dataset_labels_combined = np.array(all_dataset_labels)
            
            # Apply threshold filtering to dataset labels for other plots
            if snr_threshold is not None:
                dataset_labels_combined = dataset_labels_combined[above_threshold_mask]
            
            unique_datasets, counts = np.unique(dataset_labels_combined, return_counts=True)
            dataset_breakdown = dict(zip(unique_datasets, counts))
            print(f"✓ Dataset breakdown: {dataset_breakdown}")
            
            # Create unfiltered version for SNR histogram (to show full distribution)
            dataset_labels_unfiltered = np.array(all_dataset_labels)
        
    else:
        print("⚠️ No valid cells found in selected datasets.")
        lag_robs_combined = torch.empty(0)
        depth_combined = np.array([])
        contam_combined = np.array([])
        robs_combined = torch.empty(0, 0, 0, 0)
        snr_values = np.array([])

    # Plot combined data
    if len(lag_robs_combined) == 0:
        print("⚠️ No cells available for selected datasets; skipping plots.")
    else:
        dt = 1000/120
        iix_combined = contam_combined < 100

        # First figure: scatter plot of latency vs depth
        fig_d, ax_d = plt.subplots(1, 1, figsize=(10, 8))
        ax_d.scatter(lag_robs_combined[iix_combined]*dt, -depth_combined[iix_combined], 
                    s=20, facecolor='k', alpha=.25, edgecolor='w')
        ax_d.set_xlim(10, 60)
        ax_d.axhline(150, color='k', linestyle='--')
        ax_d.axhline(-150, color='k', linestyle='--')

        title_combined = f"Data ({', '.join(dataset_names)})"
        ax_d.set_title(title_combined)
        ax_d.set_xlabel('Stimulus Latency (ms)')
        ax_d.set_ylabel('Depth (um)')
        plt.tight_layout()
        # To save, uncomment:
        # plt.savefig(fig_dir / f"{sta_or_ste}_latency_combined_datasets.pdf")

        # Second figure: plot spatial STA/STE images at (latency, depth)
        fig_i, ax_i = plt.subplots(1, 1, figsize=size)  # Fixed large size for consistent appearance
        inds_plot = np.where(iix_combined)[0]
        # Match the scatter limits exactly for consistency
        ax_i.set_xlim(ax_d.get_xlim())
        ax_i.set_ylim(ax_d.get_ylim())

        ax_i.axhline(150, color='k', linestyle='--')
        ax_i.axhline(-150, color='k', linestyle='--')
        ax_i.set_xlabel('Stimulus Latency (ms)')
        ax_i.set_ylabel('Depth (um)')
        ax_i.set_title(f"Peak spatial {sta_or_ste.upper()} at latency ({title_combined})")

        if inds_plot.size > 0:
            n_lags = robs_combined.shape[0]
            xs = (lag_robs_combined[inds_plot] * dt)
            ys = (-depth_combined[inds_plot])
            # transparent scatter to expand data limits reliably
            ax_i.scatter(xs, ys, s=0.1, alpha=0)
            for idx, j in enumerate(inds_plot):
                lag_idx = int(np.clip(np.rint(lag_robs_combined[j]).item(), 0, n_lags-1))
                img = robs_combined[lag_idx, :, :, j].detach().cpu().numpy()
                img = np.nan_to_num(img, nan=0.0)
                
                # Normalize to [-1, 1] by dividing by max absolute value (consistent with repo patterns)
                max_abs_img = np.max(np.abs(img))
                if not np.isfinite(max_abs_img) or max_abs_img == 0:
                    continue
                img_norm = img / max_abs_img
                
                # Resize to square for consistent aspect ratio (interpolate to fixed size)
                h, w = img_norm.shape
                # target_size = 20  # Larger fixed square size for better visibility
                zoom_h = target_size / h
                zoom_w = target_size / w
                img_square = zoom(img_norm, (zoom_h, zoom_w), order=1)  # bilinear interpolation
                
                # Use Normalize object for proper color scaling with OffsetImage
                norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                oi = OffsetImage(img_square, cmap='coolwarm', norm=norm, zoom=1.2)  # Larger zoom for bigger, more visible squares
                ab = AnnotationBbox(oi, (xs[idx], ys[idx]), frameon=False, box_alignment=(0.5, 0.5))
                ax_i.add_artist(ab)

        plt.tight_layout()
        # To save, uncomment:
        # plt.savefig(fig_dir / f"{sta_or_ste}_latency_combined_datasets.pdf")

        # Third figure: SNR distribution
        fig_snr, ax_snr = plt.subplots(1, 1, figsize=(10, 6))
        
        if len(snr_values) > 0:
            if show_dataset_colors and 'dataset_labels_unfiltered' in locals():
                # Plot colored histogram by dataset using unfiltered labels (to show full distribution)
                unique_datasets = np.unique(dataset_labels_unfiltered)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_datasets)))  # Use Set3 colormap for distinct colors
                
                # Plot histogram for each dataset with different colors
                for i, dataset in enumerate(unique_datasets):
                    mask = dataset_labels_unfiltered == dataset
                    if mask.any():
                        dataset_snr = snr_values[mask]
                        ax_snr.hist(dataset_snr, bins=50, alpha=0.6, color=colors[i], 
                                   edgecolor='black', linewidth=0.5, label=f'{dataset} (n={mask.sum()})')
                
                ax_snr.legend(title='Datasets', loc='upper right')
                print(f"✓ Plotted SNR histogram with {len(unique_datasets)} dataset colors")
                
            else:
                # Plot single-color histogram
                ax_snr.hist(snr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                print("✓ Plotted SNR histogram in single color")
            
            # Add percentile lines
            ax_snr.axvline(np.median(snr_values), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(snr_values):.3f}')
            ax_snr.axvline(np.percentile(snr_values, 90), color='orange', linestyle='--', linewidth=2, label=f'90th percentile: {np.percentile(snr_values, 90):.3f}')
            ax_snr.axvline(np.percentile(snr_values, 95), color='green', linestyle='--', linewidth=2, label=f'95th percentile: {np.percentile(snr_values, 95):.3f}')
            
            # Add threshold line if threshold is set
            if snr_threshold is not None:
                ax_snr.axvline(snr_threshold, color='purple', linestyle='-', linewidth=3, alpha=0.8, label=f'Threshold: {snr_threshold:.3f}')
            
            ax_snr.set_xlabel(f'SNR ({snr_measure})')
            ax_snr.set_ylabel('Number of cells')
            
            # Update title to show threshold information
            if snr_threshold is not None:
                ax_snr.set_title(f'SNR Distribution for {sta_or_ste.upper()} ({title_combined})\nThreshold: {snr_threshold:.3f} ({n_above_threshold}/{n_total} cells above threshold)')
            else:
                ax_snr.set_title(f'SNR Distribution for {sta_or_ste.upper()} ({title_combined})')
            ax_snr.grid(True, alpha=0.3)
            
            # Add text box with statistics
            if snr_threshold is not None:
                stats_text = f'Threshold: {snr_threshold:.3f}\nCells above: {n_above_threshold}/{n_total}\nMean SNR: {np.mean(snr_values):.3f}\nStd SNR: {np.std(snr_values):.3f}\nMin SNR: {np.min(snr_values):.3f}\nMax SNR: {np.max(snr_values):.3f}'
            else:
                stats_text = f'Total cells: {len(snr_values)}\nMean SNR: {np.mean(snr_values):.3f}\nStd SNR: {np.std(snr_values):.3f}\nMin SNR: {np.min(snr_values):.3f}\nMax SNR: {np.max(snr_values):.3f}'
            
            ax_snr.text(0.02, 0.98, stats_text, transform=ax_snr.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        # To save, uncomment:
        # plt.savefig(fig_dir / f"{sta_or_ste}_snr_distribution.pdf")


# %%

#just view a single sta for a single dataset
model_name = list(all_results.keys())[1]
sta_struct = all_results[model_name]['sta']
sta_robs_list = sta_struct['sta_robs']
sta_robs = sta_robs_list[0]

plt.imshow(sta_robs[0,:,:,0], aspect='auto', cmap='gray')
plt.show()
