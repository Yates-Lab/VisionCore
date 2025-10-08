#%%
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
from eval_stack_multidataset import ccnorm_variable_trials

from DataYatesV1 import enable_autoreload

enable_autoreload()

from eval_stack_utils import argmin_subpixel, argmax_subpixel

# fig_dir = Path('/home/jake/repos/DataYatesV1/jake/multidataset_ddp/talk_figures')
fig_dir = Path('/home/tejas/Documents/fixational-transients/DataYatesV1/jake/multidataset_ddp/talk_figures_verify')
#%% utility for finding subpixel argmax / argmin


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

#%% reorganize the gratings results, which are currently organize by dataset
datasets = all_results[list(all_results.keys())[0]]['saccade']['backimage']['datasets'] 


gratings_dict = {'ori_tuning_data': np.nan*np.zeros(len(datasets)), 'ori_tuning_model': np.nan*np.zeros(len(datasets)),
            'sf_tuning_data': np.nan*np.zeros(len(datasets)), 'sf_tuning_model': np.nan*np.zeros(len(datasets)), 'ori_snr': np.nan*np.zeros(len(datasets)), 'sf_snr': np.nan*np.zeros(len(datasets))}            

for item, val in gratings_results.items():
    dataaset_idx = np.isin(datasets, item)
    
    # get sf tuning
    sf_idx, _ = argmax_subpixel(val['robs']['sf_tuning'], 1)
    sf_idx_model, _ = argmax_subpixel(val['rhat']['sf_tuning'], 1)
    
    alpha = sf_idx - np.floor(sf_idx)
    alpha_mod = sf_idx_model - np.floor(sf_idx_model)
    sfs = val['robs']['sfs']
    sf_tuning_data =  alpha*sfs[np.ceil(sf_idx).astype(int)] + (1-alpha)*sfs[np.floor(sf_idx).astype(int)]
    sf_tuning_model =  alpha_mod*sfs[np.ceil(sf_idx_model).astype(int)] + (1-alpha_mod)*sfs[np.floor(sf_idx_model).astype(int)]

    # get ori tuning
    ori_idx, _ = argmax_subpixel(val['robs']['ori_tuning'], 1)
    ori_idx_model, _ = argmax_subpixel(val['rhat']['ori_tuning'], 1)
    alpha = ori_idx - np.floor(ori_idx)
    alpha_mod = ori_idx_model - np.floor(ori_idx_model)
    oris = val['robs']['oris']
    ori_tuning_data =  alpha*oris[np.ceil(ori_idx).astype(int)] + (1-alpha)*oris[np.floor(ori_idx).astype(int)]
    ori_tuning_model =  alpha_mod*oris[np.ceil(ori_idx_model).astype(int)] + (1-alpha_mod)*oris[np.floor(ori_idx_model).astype(int)]

    gratings_dict['ori_tuning_data'][dataaset_idx] = ori_tuning_data
    gratings_dict['ori_tuning_model'][dataaset_idx] = ori_tuning_model
    gratings_dict['sf_tuning_data'][dataaset_idx] = sf_tuning_data
    gratings_dict['sf_tuning_model'][dataaset_idx] = sf_tuning_model
    gratings_dict['ori_snr'][dataaset_idx] = val['robs']['ori_snr']
    gratings_dict['sf_snr'][dataaset_idx] = val['robs']['sf_snr']

#%% Load model
from eval_stack_multidataset import load_model

model_name = list(all_results.keys())[0] # model 1

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
plt.savefig(fig_dir / 'temporal_weights.pdf')




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
depth, waveforms, cids, dids, contamination, missing_pct = extract_qc_spatial(model_name, all_results)
print(f"\nQC data: {len(depth)} cells, contamination range: {np.nanmin(contamination):.1f}-{np.nanmax(contamination):.1f}%")

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

#%% recompute CCNORM
ccnorm_dict = {}
for model_name in all_results.keys():
    ccnorm_dict[get_model_name(model_name)] = get_ccnorm(all_results, model_name)

ccnorm_0 = get_ccnorm(all_results, list(all_results.keys())[0])
ccnorm_1 = get_ccnorm(all_results, list(all_results.keys())[1])
ccnorm_1['ccnorm'][ccnorm_1['ccnorm']==1] -=np.random.rand(sum(ccnorm_1['ccnorm']==1))*1e-1

plt.hist(ccnorm_1['ccnorm'], np.linspace(0, 1,50), density=True)
plt.xlabel('CCNORM')
plt.ylabel('Density')
plt.savefig(fig_dir / 'ccnorm_distribution.pdf')
# plt.hist(ccnorm_0['ccnorm'], np.linspace(0, 1,100))

ind = np.argsort(ccnorm_1['ccnorm'])[::-1]
ind = ind[np.where(~np.isnan(ccnorm_1['ccnorm'][ind]))[0]]

i = 0
#%%
i += 5
plt.fill_between(np.arange(ccnorm_1['rbarhat'].shape[0]), ccnorm_1['rbar'][:,ind[i]], facecolor='k', alpha=.5, edgecolor='k')
plt.plot(ccnorm_1['rbarhat'][:,ind[i]], 'r')
plt.plot(ccnorm_1['rbarhat_range'][0,:,ind[i]], 'r--')
plt.plot(ccnorm_1['rbarhat_range'][1,:,ind[i]], 'b--')

ccnorm_1['ccnorm'][ind[i]]

#%% get waveofmr stats
wf_dur= tau_peak - tau_trough

iix = (contamination<=20)

plt.hist(wf_dur[iix], 20)

wf_separation_bins = [-np.inf, -12,  0, 12, 19, np.inf]

nclasses = len(wf_separation_bins) - 1
fig, ax = plt.subplots(1,nclasses, figsize=(10, 3), sharey=True, sharex=True)
wf_class = np.digitize(wf_dur, wf_separation_bins)

for i in np.unique(wf_class):
    ix = (wf_class==i) & iix
    ax[i-1].plot(wfs[:,ix], 'k', alpha = .1)
    ax[i-1].set_title(f'Class {i} n = {np.sum(ix)}')

plt.figure()
for i in np.unique(wf_class):
    ix = (wf_class==i) & iix
    cnt, bins = np.histogram(depth[ix], np.linspace(np.nanmin(depth), np.nanmax(depth), 50), density=True)
    plt.fill_betweenx(bins[:-1], cnt*400 + i, np.ones_like(cnt)*i, alpha=.5, color='k')
    plt.axhline(-150, color='k', linestyle='--')
    plt.axhline(150, color='k', linestyle='--')
    

depth_bins = [-np.inf, -150, 150, np.inf]
depth_class = np.digitize(depth, depth_bins)

waveforms_dict = {'wf': wfs, 'wf_dur': wf_dur, 'wf_class': wf_class, 'depth': depth, 'depth_class': depth_class, 'contamination': contamination}


#%% plot for bps
bps_dict = {}

    
for imodel in range(len(all_results.keys())):

    model_name = list(all_results.keys())[imodel]
    bps_dict[get_model_name(model_name)] = {}

    for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:   
        
        bps_ = np.concatenate(all_results[model_name]['bps'][stim_type]['bps'])
        
        # bps_ = bps_[contamination < 20]
        bps_dict[get_model_name(model_name)][stim_type] = {}
        bps_dict[get_model_name(model_name)][stim_type] = np.maximum(bps_, 0)
        
# fig, axes = plt.subplots(1, nclasses, figsize=(12,3), sharey=True, sharex=True)
#     for iclass in np.unique(wf_class):
#         ax = axes[iclass-1]
#         ax.plot(bps_dict[list(bps_dict.keys())[0]][stim_type][wf_class==iclass], bps_dict[list(bps_dict.keys())[1]][stim_type][wf_class==iclass], '.')
#         ax.plot(plt.xlim(), plt.xlim(), 'k--')
#         ax.set_title(f'Class {iclass}')


#     plt.xlabel(model_names[0])
#     plt.ylabel(model_names[1])
#     plt.show()

#%% plot bps just for back image

stim_type = 'backimage'
base = bps_dict[list(bps_dict.keys())[0]][stim_type]
mod = bps_dict[list(bps_dict.keys())[1]][stim_type]
plt.scatter(base, mod, alpha=.25, color='k', edgecolors='w')
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Base')
plt.ylabel('Mod')
plt.show()

for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
    base = bps_dict[list(bps_dict.keys())[0]][stim_type]
    mod = bps_dict[list(bps_dict.keys())[1]][stim_type]
    h = plt.hist(mod-base, np.linspace(-.5, .5, 100), alpha=.25)
    plt.step(h[1][:-1], h[0], where='post', color=h[2][0].get_facecolor(), alpha=1, linewidth=2)

plt.xlim(-.2, .5)
plt.xlabel('LLR (modulator - vision)')
plt.ylabel('Neuron Count')
plt.savefig(fig_dir / 'bps_comparison.pdf')

#%% saccade modulation
saccades_dict = {}

for model_id in range(len(all_results.keys())):
    model_name = list(all_results.keys())[model_id]
    
    saccades_dict[get_model_name(model_name)] = {}

    for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
        saccades_dict[get_model_name(model_name)][stim_type]={}
        # Get BPS and saccade data for gaborium
        bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
        
        Rdata = get_Rnorm(saccade_robs, saccade_time_bins)
        Rmodel = get_Rnorm(saccade_rhat, saccade_time_bins)

        time_ix = (saccade_time_bins > 0) & (saccade_time_bins < 22)
        tau_trough_robs, val_trough_robs = argmin_subpixel(Rdata[time_ix], 0)
        tau_peak_robs, val_peak_robs = argmax_subpixel(Rdata[time_ix], 0)
        tau_trough_rhat, val_trough_rhat = argmin_subpixel(Rmodel[time_ix], 0)
        tau_peak_rhat, val_peak_rhat = argmax_subpixel(Rmodel[time_ix], 0)

        saccades_dict[get_model_name(model_name)][stim_type] = {'robs': saccade_robs, 'rhat': saccade_rhat, 
                'time_bins': saccade_time_bins, 'cids': cids, 'dids': dids,
                'robs_normed': Rdata, 'rhat_normed': Rmodel,
                'tau_trough_robs': tau_trough_robs, 'val_trough_robs': val_trough_robs,
                'tau_peak_robs': tau_peak_robs, 'val_peak_robs': val_peak_robs,
                'tau_trough_rhat': tau_trough_rhat, 'val_trough_rhat': val_trough_rhat,
                'tau_peak_rhat': tau_peak_rhat, 'val_peak_rhat': val_peak_rhat}

#remove hardcoding and make name based on extract_model_type
#%% STA Latency

ste_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_robs'], -1) / np.concatenate(all_results[list(all_results.keys())[1]]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[list(all_results.keys())[1]]['sta']['norm_dfs'], -1)[None,None,None,:])
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

plt.savefig(fig_dir / 'sta_latency.pdf')

#%%
ste_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_robs'], -1) / np.concatenate(all_results[list(all_results.keys())[1]]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[0]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[list(all_results.keys())[0]]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat0 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[list(all_results.keys())[1]]['sta']['norm_dfs'], -1)[None,None,None,:])

peak_lag = np.argmax(ste_robs.var((1,2)),0)
# sta_rhat1[,...].var((0,1))

inds = np.argsort(sta_robs.var((0,1,2)))
inds = np.array(inds)[::-1].tolist()
inds = inds[200:]

n_cells = sta_robs.shape[-1]

max_plots_per_fig = 20**2

n_cells = np.minimum(n_cells, max_plots_per_fig)

sx = np.floor(np.sqrt(n_cells)).astype(int)
sy = np.ceil(n_cells / sx).astype(int)
fig, axs = plt.subplots(sy, sx, figsize=(16*2, 16))

lag = 4

H = sta_robs.shape[1]

for i in range(n_cells):
    ax = axs.flatten()[i]
    # v = sta_robs[lag,:,:,i].abs().max()
    # I = torch.concat([sta_robs[lag,:,:,i], torch.ones(H,1), sta_rhat[lag,:,:,i]], 1)
    # ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)
    j = inds[i]
    I = torch.concat([sta_robs[lag,:,:,j], torch.zeros(H,1), sta_rhat1[lag,:,:,j]], 1)
    ax.imshow(I, cmap='gray_r', interpolation='none')
    # ax.set_title(f'Cell {i}')
    ax.axis('off')

plt.tight_layout()

plt.show()


#%% plot STA Latency vs. Saccade Peak and tough latency



dt = 1000/120
from scipy.stats import pearsonr

wf_class_breakout = False

for stim_type in ['backimage']: # @, 'gaborium', 'fixrsvp', 'gratings']:

    fig, ax = plt.subplots(1, figsize=(6, 6))
    model_name = list(saccades_dict.keys())[1]

    try:
        
            
        if wf_class_breakout:
            n_loop = np.unique(waveforms_dict['wf_class'])
        else:
            n_loop = [1]

        for iclass in n_loop:
            if wf_class_breakout:
                iix = (waveforms_dict['wf_class']==iclass) & (waveforms_dict['contamination'] < 100)
            else:
                iix = (waveforms_dict['contamination'] < 100)
            
            print(f"n = {iix.sum()}")
        
            saccade_tau_peak = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
            saccade_tau_trough = saccades_dict[model_name][stim_type]['tau_trough_robs'][iix]*dt
            
            # throw out outliers
            id = (saccade_tau_peak < 150) & (saccade_tau_peak > 20)
            saccade_tau_peak = saccade_tau_peak[id]
            saccade_tau_trough = saccade_tau_trough[id]

            sta_tau = lag_robs[iix]*dt
            sta_tau = sta_tau[id]

            ax.scatter(saccade_tau_peak, sta_tau, s=20, edgecolors='w', alpha=.5)
            

            
            ax.set_title('Saccade vs. STA Latency')
            ax.set_xlabel('Saccade Latency (ms)')
            ax.set_ylabel('STA Latency (ms)')
            ax.set_xlim(0, 150)
            ax.set_ylim(0, 100)
            ax.plot([0, 150], [0, 150], 'k--')

            ax.scatter(saccade_tau_trough, sta_tau, s=20, edgecolors='w', alpha=.5)
            
            ax.set_title('Saccade vs. STA Latency')
            ax.set_xlabel('Saccade Latency (ms)')
            ax.set_ylabel('STA Latency (ms)')
            # ax[0].set_xlim(0, 150)
            # ax[0].set_ylim(0, 100)
            # ax[0].plot(plt.xlim(), plt.xlim(), 'k--')
            dsim = ( (saccade_tau_peak + .1) / (sta_tau + .1))
            ix = (dsim < 1) & (dsim > .5)
            ix = ix & (saccade_tau_peak < 100) & (saccade_tau_trough > 0)
            ax.scatter(saccade_tau_trough[ix], sta_tau[ix], s=20, edgecolors='w', alpha=.5)
            ax.scatter(saccade_tau_peak[ix], sta_tau[ix], s=20, edgecolors='w', alpha=1)
            the_list = np.where(iix)[0][id][ix]
            # ax.plot(np.stack([saccade_tau_trough[ix], saccade_tau_peak[ix]]), np.stack([sta_tau[ix], sta_tau[ix]]), 'k')
        plt.savefig(fig_dir / f'sta_saccade_stim_latency_comparison_{stim_type}.pdf')
    except Exception as e:
        print(f"❌ STA vs. Saccade latency comparison failed: {e}")
        import traceback
        traceback.print_exc()


#%%
# the_list = the_list[waveforms_dict['wf_class'][the_list]>2]

saccade_tau_peak = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
saccade_tau_trough = saccades_dict[model_name][stim_type]['tau_trough_robs'][iix]*dt

# throw out outliers
id = (saccade_tau_peak < 150) & (saccade_tau_peak > 20)
saccade_tau_peak = saccade_tau_peak[id]
saccade_tau_trough = saccade_tau_trough[id]

sta_tau = lag_robs[iix]*dt
sta_tau = sta_tau[id]

dep = waveforms_dict['depth'][iix][id]
plt.subplot(1,2,1)
plt.plot(sta_tau, -dep, '.')
plt.plot(lag_robs[the_list]*dt, -waveforms_dict['depth'][the_list], '.')
plt.xlim(0, 100)
plt.axhline(150, color='k', linestyle='--')
plt.axhline(-150, color='k', linestyle='--')
plt.subplot(1,2,2)
plt.plot(saccade_tau_peak, -dep, '.')
plt.plot(saccade_tau_trough, -dep, '.')
plt.plot(sta_tau, -dep, '.')
plt.xlim(0, 100)
plt.axhline(150, color='k', linestyle='--')
plt.axhline(-150, color='k', linestyle='--')

#%%
# the_list = the_list[waveforms_dict['wf_class'][the_list]>2]
plt.plot(waveforms_dict['wf'][:,the_list])

#%%

print(f"Depth n = {len(depth)}")
print(f"Contamination n = {len(contamination)}")
print(f"Waveforms Class n = {len(wf_class)}")
print(f"Depth Class n = {len(depth_class)}")


#%% Does not run. Needs gratings_dict
dt = 1000/120

model_name = list(saccades_dict.keys())[0]
print(f"Model name: {model_name}")
stim_type = 'backimage'
iix = (waveforms_dict['contamination'] < 20) & (waveforms_dict['wf_class'] > 2)
iix = iix & (gratings_dict['ori_snr'] > 1) & (gratings_dict['sf_snr'] > 1)
saccade_tau = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
plt.plot(saccade_tau, gratings_dict['sf_tuning_data'][iix], '.')
plt.ylim(2, 5)

#%%
dt = 1000/120
from scipy.stats import pearsonr

wf_class_breakout = False

for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:

    fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=True, sharex=True)
    model_name = list(saccades_dict.keys())[1]

    try:
        for i, model_name in enumerate(saccades_dict.keys()):
            
            if wf_class_breakout:
                n_loop = np.unique(waveforms_dict['wf_class'])
            else:
                n_loop = [1]

            for iclass in n_loop:
                if wf_class_breakout:
                    iix = (waveforms_dict['wf_class']==iclass) & (waveforms_dict['contamination'] < 100)
                else:
                    iix = (waveforms_dict['contamination'] < 100)
                
                print(f"n = {iix.sum()}")
            
                saccade_tau_data = saccades_dict[model_name][stim_type]['tau_peak_robs'][iix]*dt
                saccade_tau_model = saccades_dict[model_name][stim_type]['tau_peak_rhat'][iix]*dt
                # throw out outliers
                id = (saccade_tau_data < 150) & (saccade_tau_data > 20) & (saccade_tau_model < 150) & (saccade_tau_model > 20)
                saccade_tau_data = saccade_tau_data[id]
                saccade_tau_model = saccade_tau_model[id]
                
                r2 = pearsonr(saccade_tau_data, saccade_tau_model)[0]**2
                print(f"R2 for {model_name} = {r2}")
                # show text for r^2 in figure 
                ax[i].text(0.05, 0.95, f'r² = {r2:.3f}', transform=ax[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax[i].scatter(saccade_tau_data, saccade_tau_model, s=20, edgecolors='w', alpha=.5)
                ax[i].plot(plt.xlim(), plt.xlim(), 'k--')
                ax[i].set_title(get_model_name(model_name))
                ax[i].set_xlabel('Data (ms)')
                ax[i].set_ylabel('Model (ms)')
                ax[i].set_xlim(20, 150)
                ax[i].set_ylim(20, 150)

        plt.savefig(fig_dir / f'saccade_latency_comparison_{stim_type}.pdf')
    except Exception as e:
        print(f"❌ Saccade latency comparison failed: {e}")

#%% Compute saccade metrics
dt = 1000/120
win = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['win'][dset]
sac_time_bins = np.arange(win[0], win[1])*dt

sortby = 'latency'
for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
    fig, ax = plt.subplots(1,3, figsize=(10, 5), sharey=True, sharex=True)

    iix = (waveforms_dict['contamination'] < 100)

    model_name = list(saccades_dict.keys())[1]
    Rdata = saccades_dict[model_name][stim_type]['robs_normed']
    Rmodel = saccades_dict[model_name][stim_type]['rhat_normed']
    iix = np.where(iix & ~np.isnan(np.sum(Rdata,0)))[0]
    Rdata = Rdata[:,iix]
    Rmodel = Rmodel[:,iix]

    if sortby == 'latency':
        ind = np.argsort(np.argmax(Rmodel, 0))
    elif sortby == 'depth':
        ind = np.argsort(waveforms_dict['depth'][iix])

    ax[0].imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[sac_time_bins[0], sac_time_bins[-1], 0, len(ind)])
    ax[0].set_title(f'Data ({stim_type})')
    ax[0].set_xlim(-50, 250)
    ax[0].axvline(0, color='k', linestyle='--')
    ax[0].set_ylabel('Neuron (sorted by peak latency)')
    ax[0].set_xlabel('Time from Saccade Onset (ms)')

    for i, model_name in enumerate(saccades_dict.keys()):
        Rmodel = saccades_dict[model_name][stim_type]['rhat_normed'][:,iix]
        ax[i+1].imshow(Rmodel[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[sac_time_bins[0], sac_time_bins[-1], 0, len(ind)])
        ax[i+1].set_title(get_model_name(model_name))
        ax[i+1].set_xlim(-50, 250)
        ax[i+1].axvline(0, color='k', linestyle='--')

    plt.savefig(fig_dir / f'saccade_sorted_{sortby}_{stim_type}.pdf')

#%% Get some summary stats on saccades

stim_type = 'backimage'
model_id = 1
dset = 0
dt = 1/120
win = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['win'][dset]
sac_time_bins = np.arange(win[0], win[1])*dt

prev = []
next = []
amp = []
dur = []
for dset in range(len(all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['robs'])):
    prev.append([s['time_previous'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])
    next.append([s['time_next'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])
    amp.append([s['A'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])
    dur.append([s['end_time']-s['start_time'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]])

prev = np.concatenate(prev)
next = np.concatenate(next)
amp = np.concatenate(amp)
dur = np.concatenate(dur)

plt.figure(figsize=(3,1))
plt.hist(next+np.random.rand(len(next))/100, np.linspace(0, 1, 150), color='gray')
plt.hist(dur+np.random.rand(len(dur))/100, np.linspace(0, .1, 150), color='k', alpha=.8)

# turn off the spikes
plt.box(False)
plt.xlim(0, .5)
plt.xlabel('Saccade Duration (s)')


plt.figure(figsize=(3,1))
plt.hist(next+np.random.rand(len(next))/100, np.linspace(0, 1, 150), color='gray')
plt.hist(dur+np.random.rand(len(dur))/100, np.linspace(0, .1, 150), color='k', alpha=.8)

plt.savefig(fig_dir / 'saccade_duration.pdf')

# turn off the spikes
plt.box(False)
plt.xlim(-0.05, .25)
plt.xlabel('Saccade Duration (s)')

plt.savefig(fig_dir / 'saccade_duration_clipped.pdf')

#%%
iix = (gratings_dict['ori_snr'] > 1)# & (contamination < 20)
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.plot(gratings_dict['ori_tuning_data'][iix], gratings_dict['ori_tuning_model'][iix], '.', alpha=.1)
plt.xlim(15, 160)
plt.ylim(15, 160)
plt.xlabel('Data (deg)')
plt.ylabel('Model (deg)')
plt.title('Orientation Tuning')

plt.subplot(1,2,2)
plt.plot(gratings_dict['sf_tuning_data'][iix], gratings_dict['sf_tuning_model'][iix], '.', alpha=.1)
plt.xlim(1, 8)
plt.ylim(1, 8)
plt.xlabel('Data (cyc/deg)')
plt.ylabel('Model (cyc/deg)')
plt.title('Spatial Frequency Tuning')

plt.savefig(fig_dir / 'gratings_tuning.pdf')

#%% What is the modulator doing?

fig1, ax1 = plt.subplots(1,3,figsize=(10, 3), sharey=True, sharex=True)
fig2, ax2 = plt.subplots(1,3,figsize=(10, 3), sharey=True, sharex=True)

wf_class_breakout = False

for istim, stim_type in enumerate(['backimage', 'gaborium', 'gratings']):
    
    Rmodel1 = saccades_dict[list(saccades_dict.keys())[1]][stim_type]['rhat']
    Rmodel0 = saccades_dict[list(saccades_dict.keys())[0]][stim_type]['rhat']

    iix = (waveforms_dict['contamination'] < 100) & ~np.isnan(Rmodel1.sum(0))
    Rdelta = Rmodel1[:,iix] / Rmodel0[:,iix]
    ind = np.argsort(np.argmin(Rdelta, 0))
    ax1[istim].imshow(Rdelta[:,ind].T, aspect='auto', interpolation='none', cmap='coolwarm', vmin=.5, vmax=1.5, extent=[sac_time_bins[0], sac_time_bins[-1], 0, len(ind)])
    ax1[istim].set_xlim(-.05, .25)

    wf_class = waveforms_dict['wf_class'][iix]
    depth_class = waveforms_dict['depth_class'][iix]

    
    if wf_class_breakout:
        n_loop = [3, 4, 5] #[1,2]# [2, 3, 4] #np.unique(wf_class)
    else:
        n_loop = [1]

    cmap = plt.cm.get_cmap("tab10", max(n_loop))
    linestyles = ['--', '-', ':']

    for iclass in n_loop:
        for idepth in [1,2,3]: # 4 is nan
            if wf_class_breakout:
                ix = (wf_class==iclass) & (waveforms_dict['depth_class'][iix]==idepth)
            else:
                ix = (waveforms_dict['depth_class'][iix]==idepth)
            # ix = (wf_class==iclass) & (waveforms_dict['depth_class'][iix]==idepth)
            mu = Rdelta[:, ix].mean(1)
            se = Rdelta[:, ix].std(1)/np.sqrt(ix.sum())
            ax2[istim].plot(sac_time_bins, mu, color=cmap(iclass-1), linestyle=linestyles[idepth-1], label=f'Depth {idepth}')
            # plot errorbars as SE mean
            ax2[istim].fill_between(sac_time_bins, mu-se*2, mu+se*2, color=cmap(iclass-1), alpha=.2)
    ax2[istim].axhline(1, color='k', linestyle='--')
    ax2[istim].set_title(stim_type)
    ax1[istim].set_title(stim_type)
    ax2[istim].set_xlim(-.05, .25)
    ax2[istim].set_xlabel('Time from Saccade Onset (s)')
    
    for iax in range(len(ax2)):
        ax2[iax].set_xlim(-.05, .25)
           # plt.plot(sac_time_bins, Rdelta[:, wf_class==iclass].mean(1), label=iclass)
    
fig1.savefig(fig_dir / f'modulator_effect_{wf_class_breakout}.pdf')
fig2.savefig(fig_dir / f'modulator_effect_by_depth_{wf_class_breakout}.pdf')


#%% Recompute CCnorm Values



i = 0
#%%
i += 5
plt.fill_between(np.arange(ccnorm_1['rbarhat'].shape[0]), ccnorm_1['rbar'][:,ind[i]], facecolor='k', alpha=.5, edgecolor='k')
plt.plot(ccnorm_1['rbarhat'][:,ind[i]], 'r')
plt.plot(ccnorm_1['rbarhat_range'][0,:,ind[i]], 'r--')
plt.plot(ccnorm_1['rbarhat_range'][1,:,ind[i]], 'b--')


#%% Same plot but with histogram difference
for stim_type in ['backimage', 'gaborium', 'fixrsvp', 'gratings']:
    bps = []
    model_names = []
    plt.figure(figsize=(5,5))
    for imodel in range(len(all_results.keys())):
        model_name = list(all_results.keys())[imodel]
        bps_ = np.concatenate(all_results[model_name]['bps'][stim_type]['bps'])
        # bps_ = bps_[contamination < 20]
        bps.append(np.maximum(bps_, 0))
        model_names.append(get_model_name(model_name))

    for iclass in np.unique(wf_class):
        ix = (wf_class==iclass) & iix
        cnt, bins = np.histogram(bps[1][ix]-bps[0][ix], np.linspace(-1, 1, 50), density=True)
        cnt = cnt / np.max(cnt)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.fill_between(bin_centers, cnt + iclass, np.ones_like(cnt)*iclass, alpha=.5, color='k')
    plt.axvline(0, color='k', linestyle='--')
    # ix = (wf_class==i) & iix
    # cnt, bins = np.histogram(depth[ix], np.linspace(np.nanmin(depth), np.nanmax(depth), 50), density=True)
    # plt.fill_betweenx(bins[:-1], cnt*400 + i, np.ones_like(cnt)*i, alpha=.5, color='k')
    # plt.axhline(-150, color='k', linestyle='--')
    # plt.axhline(150, color='k', linestyle='--')

#%% plot by depth


fig, axs = plt.subplots(1,4,figsize=(12,3), sharey=True)
for i, stim_type in enumerate(['backimage', 'gaborium', 'fixrsvp', 'gratings']):
    base_bps = np.concatenate(all_results[list(all_results.keys())[0]]['bps'][stim_type]['bps'])
    base_bps = np.maximum(base_bps, -.1)
    n_models = 1
    # n_models = len(all_results.keys())
    
    for imodel in [1]:
        ax = axs[i]
        model_name = list(all_results.keys())[imodel]
        bps_ = np.concatenate(all_results[model_name]['bps'][stim_type]['bps'])
        bps_ = np.maximum(bps_, -.1)
        # bps_ = bps_[contamination < 20]
        bps = (bps_ - base_bps) #/ np.abs(base_bps)
        ax.scatter(bps, -depth, alpha=.5, color='k', edgecolors='w')
        ax.axvline(0, color='k', linestyle='--')
        ax.axhline(0, color='k', linestyle='--')
    
    ax.set_title(stim_type)
    ax.set_xlim(-.5, 1)
    ax.set_ylim(-700, 900)

depth_bins = np.linspace(np.nanmin(depth), np.nanmax(depth), 10)

# %%


n_models = 2
stim_type = 'gaborium'
bps_comparison = {}
for model_id in range(len(all_results.keys())):
    model_name = list(all_results.keys())[model_id]
    # Get BPS and saccade data for gaborium
    bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
    bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}


good_ix = np.where(~np.isnan(np.sum(bps_comparison[0]['robs'],0)))[0]
# good_ix = np.where((contamination < 20) & (wf_class>2))[0]
Rdata = get_Rnorm(bps_comparison[0]['robs'][:,good_ix], saccade_time_bins)
Rmodel = []
for model_id in range(len(all_results.keys())):
    Rmodel.append(get_Rnorm(bps_comparison[model_id]['rhat'][:,good_ix], saccade_time_bins))

ind = np.argmax(Rmodel[0], 0)
ind = np.argsort(ind)
# ind = np.argsort(depth[good_ix])

dt = 1000/120
tbins = saccade_time_bins*dt
plt.figure(figsize=(10, 5))
plt.subplot(1,n_models+1,1)
vmin = np.nanmin(Rdata)
vmax = np.nanmax(Rdata)
plt.imshow(Rdata[:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)], vmin=vmin, vmax=vmax)
plt.title(f'Data {stim_type}')
plt.xlim(-50, 250)
plt.axvline(0, color='k', linestyle='--')

for i in range(n_models):
    
    plt.subplot(1,n_models+1,i+2)
    plt.imshow(Rmodel[i][:, ind].T, aspect='auto', interpolation='none', cmap='coolwarm', extent=[tbins[0], tbins[-1], 0, len(ind)], vmin=vmin, vmax=vmax)
    # plt.set_yticks([])
    plt.title(get_model_name(bps_comparison[i]['name']))
    if i==0:
        plt.xlabel('Time from saccade onset (ms)')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlim(-50, 250)
plt.show()

_ = plt.plot(saccade_time_bins*dt, Rmodel[1]-Rmodel[0], 'k', alpha=.1)
plt.xlim(-50, 250)
plt.axhline(0)

# %%

stim_type = 'gratings'

llrs = []
for dset in range(len(all_results[list(all_results.keys())[0]]['saccade'][stim_type]['rhat'])):
    rhat1 = all_results[list(all_results.keys())[1]]['saccade'][stim_type]['rhat'][dset]
    rhat0 = all_results[list(all_results.keys())[0]]['saccade'][stim_type]['rhat'][dset]
    robs = all_results[list(all_results.keys())[0]]['saccade'][stim_type]['robs'][dset]
    dfs = torch.tensor(all_results[list(all_results.keys())[0]]['saccade'][stim_type]['dfs'][dset])


    loss_1 = poisson_loss(torch.tensor(rhat1), torch.tensor(robs), reduction='none')  # shape [trials, neurons]
    loss_0 = poisson_loss(torch.tensor(rhat0), torch.tensor(robs), reduction='none')

    llr = (dfs *(loss_1 - loss_0)).sum(0) / dfs.sum(0)

    llrs.append(llr.numpy())
    _ = plt.plot(-llr, 'k', alpha=.1)


llr = np.concatenate(llrs, 1)
plt.plot(-llr, 'k', alpha=.1)
plt.show()

# %%
stim_type = 'backimage'
model_id = 1
dset = 0
dt = 1/120
win = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['win'][dset]
sac_time_bins = np.arange(win[0], win[1])*dt

prev = [s['time_previous'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]
next = [s['time_next'].item()*dt for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]
amp = [s['A'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]
dur = [s['end_time']-s['start_time'] for s in all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['saccade_info'][dset]]

robs = all_results[list(all_results.keys())[model_id]]['saccade'][stim_type]['robs'][dset]
rhat_base = all_results[list(all_results.keys())[0]]['saccade'][stim_type]['rhat'][dset]
rhat_gru = all_results[list(all_results.keys())[1]]['saccade'][stim_type]['rhat'][dset]

cid = 0
#%%
sorter = prev
bin_edges = np.percentile(sorter, [0, 25, 50, 75, 100])

cid += 1
fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

cmap = plt.cm.get_cmap("coolwarm", len(bin_edges)-1)
for ibin in range(len(bin_edges)-1):
    
    ind = np.where((sorter > bin_edges[ibin]) & (sorter < bin_edges[ibin+1]))[0]
    
    ax[0].plot(sac_time_bins, np.mean(robs[ind,:,cid], 0), color=cmap(ibin))     
    ax[1].plot(sac_time_bins, np.mean(rhat_base[ind,:,cid], 0), color=cmap(ibin))     
    ax[2].plot(sac_time_bins, np.mean(rhat_gru[ind,:,cid], 0), color=cmap(ibin)) 

ax[0].set_xlim(-.2, .5)
    # plt.imshow(robs[ind,:,10], aspect='auto', interpolation='none', cmap='gray_r')
# plt.hist(prev, np.arange(0, 60)*dt)

# %%
ind = np.argsort(next)
plt.imshow(robs[ind,:,10], aspect='auto', interpolation='none', cmap='gray_r')
# %%


# %%

ste_robs = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_robs'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat1 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:])
ste_rhat0 = torch.from_numpy(np.concatenate(all_results[list(all_results.keys())[1]]['sta']['ste_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:])

lag_robs, _ = argmax_subpixel(ste_robs.var((1,2)), 0)
lag_rhat0, _ = argmax_subpixel(ste_rhat0.var((1,2)), 0)
lag_rhat1, _ = argmax_subpixel(ste_rhat1.var((1,2)), 0)

fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

dt = 1000/120
iix = contamination < 20
ax[0].scatter(lag_robs[iix]*dt, -depth[iix], s=10, facecolor='k', alpha = .25, edgecolor='w')
ax[1].scatter(lag_rhat0[iix]*dt, -depth[iix], s=10, facecolor='k', alpha = .25, edgecolor='w')
ax[2].scatter(lag_rhat1[iix]*dt, -depth[iix], s=10, facecolor='k', alpha = .25, edgecolor='w')
ax[1].set_xlim(10, 60)
# ax[1].set_xlim(0.01, 0.07)


Rdata
# %%
Rmodel = []
for model_id in range(2):
    model_name = list(all_results.keys())[model_id]

    bps, saccade_robs, saccade_rhat, saccade_time_bins, cids, dids = extract_bps_saccade(model_name, stim_type, all_results)
    bps_comparison[model_id] = {'name': model_name.split('_ddp_')[0], 'bps': bps, 'cids': cids, 'dids': dids, 'robs': saccade_robs, 'rhat': saccade_rhat}

    Rdata = get_Rnorm(bps_comparison[0]['robs'], saccade_time_bins)
    rmodel = get_Rnorm(bps_comparison[model_id]['rhat'], saccade_time_bins)
    print(rmodel.shape)
    Rmodel.append(rmodel)

#%%

time_ix = (saccade_time_bins > 0) & (saccade_time_bins < 17)
tau_trough_data, val_trough_data = argmin_subpixel(Rdata[time_ix], 0)
tau_peak_data, val_peak_data = argmax_subpixel(Rdata[time_ix], 0)
tau_trough_base, val_trough_base = argmin_subpixel(Rmodel[0][time_ix], 0)
tau_peak_base, val_peak_base = argmax_subpixel(Rmodel[0][time_ix], 0)
tau_trough_gru, val_trough_gru = argmin_subpixel(Rmodel[1][time_ix], 0)
tau_peak_gru, val_peak_gru = argmax_subpixel(Rmodel[1][time_ix], 0)

fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)
ax[0].plot(tau_trough_data*dt, -val_trough_data, 'r.', alpha=.1)
ax[0].plot(tau_peak_data*dt, val_peak_data, 'b.', alpha=.1)
ax[0].set_title('Data')

ax[1].plot(tau_trough_base*dt, -val_trough_base, 'r.', alpha=.1)
ax[1].plot(tau_peak_base*dt, val_peak_base, 'b.', alpha=.1)
ax[1].set_title('Base')

ax[2].plot(tau_trough_gru*dt, -val_trough_gru, 'r.', alpha=.1)
ax[2].plot(tau_peak_gru*dt, val_peak_gru, 'b.', alpha=.1)
ax[2].set_title('GRU')

ax[1].set_xlim(0, 150)

# %%

fig, ax = plt.subplots(1,3, figsize=(10, 3), sharey=True, sharex=True)

iix = (contamination < 20)
iix = iix & (tau_trough_data > 1)
iix = iix & (tau_peak_data < 15)
ax[0].scatter(tau_trough_data[iix]*dt, -depth[iix], alpha=.1, facecolor='r', edgecolor='w')
ax[0].scatter(tau_peak_data[iix]*dt, -depth[iix], alpha=.1, facecolor='b', edgecolor='w')
ax[0].set_title('Data')

ax[1].scatter(tau_trough_base[iix]*dt, -depth[iix], alpha=.1, facecolor='r', edgecolor='w')
ax[1].scatter(tau_peak_base[iix]*dt, -depth[iix], alpha=.1, facecolor='b', edgecolor='w')
ax[1].set_title('Base')

ax[2].scatter(tau_trough_gru[iix]*dt, -depth[iix], alpha=.1, facecolor='r', edgecolor='w')
ax[2].scatter(tau_peak_gru[iix]*dt, -depth[iix], alpha=.1, facecolor='b', edgecolor='w')
ax[2].set_title('GRU')

# ax[1].set_xlim(0, 150)


# plt.plot(tau_trough_data, lag_robs, '.')
# plt.xlim(0, 20)
# plt.ylim(0, 10)
# %%
plt.plot(tau_trough_data[iix]*dt, tau_peak_data[iix]*dt, '.')
plt.plot(tau_trough_base[iix]*dt, tau_peak_base[iix]*dt, '.')
plt.plot(tau_trough_gru[iix]*dt, tau_peak_gru[iix]*dt, '.')
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Trough (ms)')
plt.ylabel('Peak (ms)')
plt.title('Latency (ms)')
plt.show()
# %%

r = Rdata[:,iix]
plt.plot(r[:,tau_trough_data[iix]>tau_peak_data[iix]])

# %%

#%% track all STAs

sta_robs = np.concatenate(all_results[model_name]['sta']['sta_robs'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]
sta_rhat = np.concatenate(all_results[model_name]['sta']['sta_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]

ste_robs = np.concatenate(all_results[model_name]['sta']['ste_robs'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]
ste_rhat = np.concatenate(all_results[model_name]['sta']['ste_rhat'], -1) / np.concatenate(all_results[model_name]['sta']['norm_dfs'], -1)[None,None,None,:]

sta_robs = torch.from_numpy(sta_robs)
sta_rhat = torch.from_numpy(sta_rhat)
ste_robs = torch.from_numpy(ste_robs)
ste_rhat = torch.from_numpy(ste_rhat)

n_cells = sta_robs.shape[-1]

max_plots_per_fig = 10**2

n_cells = np.minimum(n_cells, max_plots_per_fig)

sx = np.floor(np.sqrt(n_cells)).astype(int)
sy = np.ceil(n_cells / sx).astype(int)
fig, axs = plt.subplots(sy, sx, figsize=(16*2, 16))

lag = 4

H = sta_robs.shape[1]

for i in range(n_cells):
    ax = axs.flatten()[i]
    # v = sta_robs[lag,:,:,i].abs().max()
    # I = torch.concat([sta_robs[lag,:,:,i], torch.ones(H,1), sta_rhat[lag,:,:,i]], 1)
    # ax.imshow(I, cmap='gray_r', interpolation='none', vmin=-v, vmax=v)

    I = torch.concat([ste_robs[lag,:,:,i], torch.zeros(H,1), ste_rhat[lag,:,:,i]], 1)
    ax.imshow(I, cmap='gray_r', interpolation='none')
    # ax.set_title(f'Cell {i}')
    ax.axis('off')

plt.tight_layout()

plt.show()







ccnorms = []
model_names = []
plt.figure(figsize=(6,3))
for imodel in range(len(all_results.keys())):
    model_name = list(all_results.keys())[imodel]
    ccnorm = np.concatenate(all_results[model_name]['ccnorm']['fixrsvp']['ccnorm'])
    iix = (contamination < 20) & (depth > 100) & (depth < 300)
    ccnorm = ccnorm[contamination < 20]
    ccnorms.append(ccnorm)
    model_names.append(get_model_name(model_name))
    

fig, ax = custom_boxplot(
        ccnorms,
        labels=model_names,
        title="CCNORM (fixrsvp)",
        xlabel="Model",
        ylabel="CC NORM",
        sig_spacing_factor=1.5
    )
# plt.ylim(.5,.75)
plt.show()

#%% recompute the CCNORM VALUES


imodel = 1

model_name = list(all_results.keys())[imodel]




#%%


# ccn = ccnorm_1['ccnorm']
# ccn_orig = ccnorm_1['ccnorm_orig']
# plt.plot(np.minimum(ccn, 1.0), ccn_orig, '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
