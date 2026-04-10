
#%%  this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

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

"""
fixrsvp_digitaltwin_performance.py
Generates Figure 3: Digital Twin Performance
"""
import sys
import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append('..')
from scripts.figures_digitaltwin import plot_digitaltwin_paper_figure

# Setup
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def load_data(pkl_path='mcfarland_outputs.pkl'):
    with open(pkl_path, 'rb') as f:
        outputs = dill.load(f)
    return outputs

def aggregate_metrics(outputs, window_idx=1):
    """
    Aggregates metrics and trace dictionaries from the outputs.
    """
    flat_metrics = {
        'var_total': [], 'var_psth': [], 'var_resid': [], 
        'alpha': [], 'cc_norm': [], 'cc_max': [], 'ds_idx': [], 'n_idx': []
    }
    
    # Store full traces per dataset for example plotting
    # Structure: dict where key = ds_idx, value = {'robs': ..., 'rhat': ...}
    all_traces = {}

    total_neurons = 0
    
    for i, out in enumerate(outputs):
        # 1. Traces
        if 'model_traces' not in out:
            print(f"Warning: Dataset {i} missing 'model_traces'. Re-run mcfarland_sim.")
            continue
            
        all_traces[i] = out['model_traces']

        # 2. Scalar Metrics (from LOTC results)
        res = out['results'][window_idx]
        mats_main = out['last_mats'][window_idx]
        mats_resid = out['last_mats_residuals'][window_idx]
        mask = out['neuron_mask']
        
        n_neurons = len(mask)
        
        # Variances
        v_tot = np.diag(mats_main['Total'])
        v_psth = np.diag(mats_main['PSTH'])
        v_resid = np.diag(mats_resid['Total'])
        
        flat_metrics['var_total'].append(v_tot)
        flat_metrics['var_psth'].append(v_psth)
        flat_metrics['var_resid'].append(v_resid)
        flat_metrics['alpha'].append(res['alpha'])
        flat_metrics['cc_norm'].append(out['ccnorm']['ccnorm'])
        flat_metrics['cc_max'].append(out['ccnorm']['ccmax'])

        
        
        # Indices for traceback
        flat_metrics['ds_idx'].append(np.full(n_neurons, i))
        flat_metrics['n_idx'].append(np.arange(n_neurons))
        
    # Concatenate scalars
    for k in flat_metrics:
        flat_metrics[k] = np.concatenate(flat_metrics[k])
        
    # Derived Metrics
    # Var Explained by Model = Var_Total - Var_Residual
    flat_metrics['var_expl_model'] = np.maximum(0, flat_metrics['var_total'] - flat_metrics['var_resid'])
    flat_metrics['var_expl_psth'] = flat_metrics['var_psth']
    
    # Improvement Ratio
    flat_metrics['improvement_ratio'] = flat_metrics['var_expl_model'] / (flat_metrics['var_expl_psth'] + 1e-6)
    
    # FEM Modulation
    flat_metrics['fem_mod'] = 1.0 - flat_metrics['alpha']
    
    return flat_metrics, all_traces

#%%

print("Loading outputs...")
outputs = load_data(pkl_path='mcfarland_outputs_mono.pkl')

print("Aggregating metrics...")
metrics, traces = aggregate_metrics(outputs, window_idx=1) # 20ms window typically

#%% Get all traces
rhats = []
robss = []
ns = []
rhos = []
var_explained_model = []
var_explained_psth = []
dataset_id = []

def var_explained(rhat, rtrue, axis=None):

    residuals = rhat-rtrue
    var_total = np.nanvar(rtrue, axis=axis)
    var_residual = np.nanvar(residuals, axis=axis)
    return 1 - var_residual/var_total

# Loop over datasets and store mean robs and rhat
for idataset, out in enumerate(outputs):        
        
        rhat = out['model_traces']['rhat']
        robs = out['model_traces']['robs']
        dfs = out['model_traces']['dfs']

        rhat[dfs==0] = np.nan
        robs[dfs==0] = np.nan

        # loop over trials and take the mean of all other trials
        rbar = np.zeros_like(robs)
        for i in range(robs.shape[0]):
            trial_ix = np.setdiff1d(np.arange(robs.shape[0]), i)
            rbar[i] = np.nanmean(robs[trial_ix], 0)
        
        var_explained_model.append(var_explained(rhat, robs, axis=(0,1)))
        var_explained_psth.append(var_explained(rbar, robs, axis=(0,1)))

        rhat = np.nansum(out['model_traces']['rhat']*out['model_traces']['dfs'], 0)
        robs = np.nansum(out['model_traces']['robs']*out['model_traces']['dfs'], 0)
        n = np.nansum(out['model_traces']['dfs'], 0)

        rhat = rhat/n
        robs = robs/n

        rhats.append(rhat)
        robss.append(robs)
        ns.append(n)
        rhos.append(np.asarray([np.corrcoef(rhat[n[:,cc]>10,cc], robs[n[:,cc]>10,cc])[0,1] for cc in range(rhat.shape[1]) ]))
        dataset_id.append(idataset*np.ones(rhat.shape[1]))

#%%
var_explained_model = np.concatenate(var_explained_model)
var_explained_psth = np.concatenate(var_explained_psth)
rhos = np.concatenate(rhos)
rhats = np.concatenate(rhats, 1)
robss = np.concatenate(robss, 1)
ns = np.concatenate(ns, 1)
dataset_id = np.concatenate(dataset_id)

ix = np.isfinite(rhos)
rhos = rhos[ix]
rhats = rhats[:,ix]
robss = robss[:,ix]
ns = ns[:,ix]
ccnorm = metrics['cc_norm'][ix] # normalized by ccmax
ccmax = metrics['cc_max'][ix] # reliability of the neuron
alpha = metrics['alpha'][ix]
var_explained_model = var_explained_model[ix]
var_explained_psth = var_explained_psth[ix]
dataset_id = dataset_id[ix]

fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=False)
ccmax_bins = np.arange(0, 1.2, .2)
for i in range(len(ccmax_bins)-1):
    ix = (ccmax > ccmax_bins[i]) & (ccmax <= ccmax_bins[i+1])
    axs[0].plot(rhos[ix], ccnorm[ix], '.', label=f"{np.sum(ix)}")

axs[0].legend()
axs[0].set_xlim(-.1, 1.1)
axs[0].set_ylim(-.1, 1.1)
axs[0].plot(plt.xlim(), plt.xlim(), 'k--', alpha=0.5)
axs[0].set_xlabel('Correlation')
axs[0].set_ylabel('Normalized Correlation')
for i in range(len(ccmax_bins)-1):
    ix = (ccmax > ccmax_bins[i]) & (ccmax <= ccmax_bins[i+1])
    axs[1].plot(ccmax[ix], ccnorm[ix], '.k', label=f"{np.sum(ix)}")
    # plot vertical line across the ccmax bins at the mean ccnorm
    axs[1].plot([ccmax_bins[i] , ccmax_bins[i+1]], ccnorm[ix].mean()*np.ones(2), color='r', alpha=1.0, linewidth=2)

axs[1].set_xlabel('Reliability')
axs[1].set_ylabel('Normalized Correlation')
axs[1].set_xlim(0, 1.1)
axs[1].set_ylim(0, 1.1)

#%%
dt = 1/120
T = robss.shape[0]
tbins = np.arange(T)*dt
ind = np.argsort(rhos)[::-1]
n2show = 25 # top units
sx = int(np.sqrt(n2show))
sy = int(np.ceil(n2show / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= n2show:
        axs.flatten()[i].axis('off')
        continue
    axs.flatten()[i].plot(tbins, robss[:,ind[i]]/dt, 'k')
    axs.flatten()[i].plot(tbins, rhats[:,ind[i]]/dt, 'r')
    axs.flatten()[i].set_title(f'{rhos[ind[i]]:.2f} {ccnorm[ind[i]]:.2f} {ccmax[ind[i]]:.2f}')
    axs.flatten()[i].axis('off')


# plt.imshow(rhats[:,ind[:100]])
#%%
good = np.where(ccmax > 0.85)[0]
print(f"{len(good)} neurons with ccmax > 0.8")
print(f"Median rho: {np.median(rhos[good]):.2f}")
print(f"Median ccnorm: {np.median(ccnorm[good]):.2f}")


good_inds = ind[np.isin(ind, good)]
# pick 4 examples (two from the top 10% and two from the median)
top = ind[:int(len(good)/10)]
med = ind[int(len(good)/2)-1:int(len(good)/2)+3]


n2show = 4
sx = int(np.sqrt(n2show))
sy = int(np.ceil(n2show / sx))
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= n2show:
        axs.flatten()[i].axis('off')
        continue
    axs.flatten()[i].plot(tbins, robss[:,top[i]]/dt, 'k')
    axs.flatten()[i].plot(tbins, rhats[:,top[i]]/dt, 'r')
    axs.flatten()[i].set_title(f'{rhos[top[i]]:.2f} {ccnorm[top[i]]:.2f} {ccmax[top[i]]:.2f}')
    # axs.flatten()[i].axis('off')

fig.savefig("../figures/mcfarland/digital_twin_performance_panel_examples_top.pdf", bbox_inches='tight', dpi=300)

# same for median
fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
for i in range(sx*sy):
    if i >= n2show:
        axs.flatten()[i].axis('off')
        continue
    axs.flatten()[i].plot(tbins, robss[:,med[i]]/dt, 'k')
    axs.flatten()[i].plot(tbins, rhats[:,med[i]]/dt, 'r')
    axs.flatten()[i].set_title(f'{rhos[top[i]]:.2f} {ccnorm[top[i]]:.2f} {ccmax[top[i]]:.2f}')

fig.savefig("../figures/mcfarland/digital_twin_performance_panel_examples_median.pdf", bbox_inches='tight', dpi=300)

#%% plot variance explained by model compared to variance explained by PSTH


cmap = plt.get_cmap('plasma')

# first, sanity check each dataset
fig, ax = plt.subplots(1,1, figsize=(5,5))
for i in np.unique(dataset_id):
    ix = np.where(dataset_id == i)[0]
    good_ix = np.intersect1d(ix, good)
    
    ax.plot(var_explained_psth[good_ix], var_explained_model[good_ix], '.', label=f"Dataset {i}", color=cmap(i/len(np.unique(dataset_id))))

ax.plot(plt.xlim(), plt.xlim(), 'k--')
ax.set_xlabel('Variance explained by PSTH')
ax.set_ylabel('Variance explained by Model')


# Main figure Panel: next, all datasets together (colored by 1-alpha)
fig, ax = plt.subplots(1,1, figsize=(3,2.5))

sc = ax.scatter(
    var_explained_psth[good],
    var_explained_model[good],
    c=(1 - alpha[good]).clip(0, 1),
    s=5
)

ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
ax.set_xlabel('Single trial r^2 (PSTH)')
ax.set_ylabel('Single trial r^2 (Model)')
ax.set_xlim(0, .4)
ax.set_ylim(0, .4)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('1 - alpha')
fig.savefig("../figures/mcfarland/digital_twin_performance_panel_var_explained.pdf", bbox_inches='tight', dpi=300)

#%% wilcoxon rank sum test
y = var_explained_psth[good]
x = var_explained_model[good]

print(f"Median x: {np.median(x):.3f}")
print(f"Median y: {np.median(y):.3f}")

from scipy.stats import wilcoxon

# paired differences
d = x - y

# one-sided test: H1 = x > y
stat, p = wilcoxon(d, alternative='greater')

print(f"Wilcoxon signed-rank: stat={stat:.3g}, p={p:.3g}")

def rank_biserial_from_wilcoxon(x, y):
    d = x - y
    d = d[d != 0]  # Wilcoxon drops zeros
    n = len(d)
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    return (n_pos - n_neg) / n

r_rb = rank_biserial_from_wilcoxon(x, y)
print(f"Rank-biserial r = {r_rb:.3f}")


# Relationship between 1-alpha and model improvement
fig = plt.figure()
plt.plot(1-alpha[good],x/y, '.')
plt.axhline(1, color='k', linestyle='--', alpha=0.5)
plt.ylim(0, 5)
plt.xlabel('1 - alpha')
plt.ylabel('Var explained by model / Var explained by PSTH')
fig.savefig("../figures/mcfarland/digital_twin_performance_panel_improvement.pdf", bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(1,1, figsize=(3,2.5))
ax.hist(ccnorm[good], bins=np.linspace(0, 1, 20), color='gray')
# title is median and IQR
ax.set_title(f"Median = {np.median(ccnorm[good]):.2f}, IQR = {np.quantile(ccnorm[good], 0.25):.2f} - {np.quantile(ccnorm[good], 0.75):.2f}")
ax.set_xlabel('Normalized Correlation')
ax.set_ylabel('Count')
fig.savefig("../figures/mcfarland/digital_twin_performance_panel_ccnorm.pdf", bbox_inches='tight', dpi=300)

#%%

print(f"Generating figure for {len(metrics['alpha'])} neurons...")
fig = plot_digitaltwin_paper_figure(metrics, traces)

savepath = "../figures/mcfarland/digital_twin_performance.pdf"
fig.savefig(savepath, bbox_inches='tight', dpi=300)
print(f"Saved to {savepath}")
# %%

# %%
