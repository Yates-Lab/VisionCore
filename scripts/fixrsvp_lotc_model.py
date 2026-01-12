

"""
This code implements the core analyses based on the law of total covariance decomposition.
Following the framework introduced by McFarland et al., we decompose the total response covariance
into components attributable to fixational eye movements (FEMs) and to the stimulus-locked PSTH.
This decomposition allows us to define a key summary metric: 1-alpha

which quantifies the fraction of rate variability explained by eye movements.
A central goal of this section is therefore to estimate the distribution of this quantity across neurons and populations.

Beyond this primary decomposition, the code performs three additional, complementary analyses.
First, we compute population Fano factors and examine how they scale with the size of the temporal counting window,
providing a population-level characterization of variability across timescales.

Second, we estimate noise correlations, isolating shared variability that is not explained by the mean stimulus response.
Third, we analyze the subspace structure of variability by asking how strongly the covariance associated with fixational
eye movements aligns with the covariance associated with the PSTH.

Third, we evaluate a digital twin model fit to the full population.
We ask whether this model explains a substantial fraction of the observed variance,
and—critically—whether the variance it captures is preferentially aligned with the FEM-related covariance or with the PSTH-related covariance. Together, these analyses allow us to move from a scalar variance decomposition to a geometric, population-level understanding of how eye movements, stimulus structure,
and model predictions shape neural variability.

TODO: we need to generate code to support these analyses with shuffled trial identities so we can
assess whether any of our results are artifactual


"""

#%% Setup and Imports
import contextlib

class TeeWriter:
    """Write to both a file and stdout simultaneously."""
    def __init__(self, file_handle):
        self.file = file_handle
        self.stdout = __import__('sys').stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

@contextlib.contextmanager
def tee_to_file(filepath, mode='w'):
    """Context manager to tee print output to both file and stdout."""
    import sys
    with open(filepath, mode) as f:
        tee = TeeWriter(f)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            yield f
        finally:
            sys.stdout = old_stdout

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
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

from mcfarland_sim import run_mcfarland_on_dataset, extract_metrics
from utils import get_model_and_dataset_configs


#%% Get model and data
model, dataset_configs = get_model_and_dataset_configs()
model = model.to(device)

#%% Run main analysis on all datasets
run_analysis = False
n_shuffles = 100

if run_analysis:
    outputs = []
    analyzers = []

    for dataset_idx in range(len(model.names)):
        print(f"Running on dataset {dataset_idx}")
        try: # some datasets do not have fixrsvp
            output, analyzer = run_mcfarland_on_dataset(model, dataset_idx, plot=False, n_shuffles=n_shuffles)
            outputs.append(output)
            analyzers.append(analyzer)
        except Exception as e:
            print(f"Failed to run on dataset {dataset_idx}: {e}")

    # Save outputs and analyzers in a local file to load so I don't have to always run this every time
    import dill
    with open('mcfarland_outputs.pkl', 'wb') as f:
        dill.dump(outputs, f)
    with open('mcfarland_analyzers.pkl', 'wb') as f:
        dill.dump(analyzers, f)

#%% Load outputs and analyzers from local file
import dill
with open('mcfarland_outputs.pkl', 'rb') as f:
    outputs = dill.load(f)
with open('mcfarland_analyzers.pkl', 'rb') as f:
    analyzers = dill.load(f)


#%% Extract relevant metrics for plotting
metrics = extract_metrics(outputs, min_total_spikes=1000, min_var=.1, eps_rho=1e-3)


#%% Plot covariance decomposition for all sessions

# import cmasher as cmr
# cmap = plt.get_cmap('cmr.prinsenvlag')   # MPL
cmap = plt.get_cmap('RdBu')   # MPL
window_idx = 3

for i in range(len(metrics[0]['Ctotal'])):

    Ctotal = metrics[window_idx]['Ctotal'][i]
    Cpsth = metrics[window_idx]['Cpsth'][i]
    Crate = metrics[window_idx]['Crate'][i]
    CnoiseU = metrics[window_idx]['CnoiseU'][i]
    CnoiseC = metrics[window_idx]['CnoiseC'][i]

    v = np.max(Ctotal.flatten()) * .5
    Cfem = Crate - Cpsth

    fig, axs = plt.subplots(1,3, figsize=(20,5))
    ax = axs[0]
    ax.imshow(Ctotal, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
    ax.set_title('Total')
    ax.axis('off')

    ax = axs[1]
    ax.imshow(Cpsth, cmap=cmap, interpolation='nearest', vmin=-v/2, vmax=v/2)
    ax.set_title('PSTH')
    ax.axis('off')

    ax = axs[2]
    ax.imshow(CnoiseU, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
    ax.set_title('Noise (Uncorrected)')
    ax.axis('off')
    fig.savefig(f'../figures/mcfarland/covariance_decomposition_{i}_{window_idx}_psth.pdf', bbox_inches='tight', dpi=300) 



    fig, axs = plt.subplots(1,4, figsize=(20,5))
    ax = axs[0]
    ax.imshow(Ctotal, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
    ax.set_title('Total')
    ax.axis('off')
    ax = axs[1]
    ax.imshow(Cfem, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
    ax.set_title('FEM')
    ax.axis('off')

    ax = axs[2]
    ax.imshow(Cpsth, cmap=cmap, interpolation='nearest', vmin=-v/2, vmax=v/2)
    ax.set_title('PSTH')
    ax.axis('off')

    ax = axs[3]
    ax.imshow(CnoiseC, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
    ax.set_title('Noise (Corrected)')
    ax.axis('off')
    fig.savefig(f'../figures/mcfarland/covariance_decomposition_{i}_{window_idx}_full.pdf', bbox_inches='tight', dpi=300) 

#%% Fano factors
from scripts.figures_fanofactors import compute_all_fano_stats, plot_fano_paper_figure, plot_supp_var_mean_panels, plot_supp_fano_hist_panels, plot_supp_shuffle_ratio_hist_one_window

stats = compute_all_fano_stats(metrics, alternative="less", fdr_q=0.05, nboot_slope=5000)
fig, axs = plot_fano_paper_figure(stats, title="Fixational eye-movement correction reduces variability")
fig.savefig("../figures/mcfarland/fano_paper_figure.pdf", bbox_inches="tight", dpi=300)

fig, axs = plot_supp_var_mean_panels(metrics, savepath="../figures/mcfarland/var_mean_panels.pdf")
fig, axs = plot_supp_fano_hist_panels(metrics, savepath="../figures/mcfarland/fano_hist_panels.pdf")

#%% plot noise correlations

from scripts.figures_noisecorr import (
    compute_all_noisecorr_stats,
    plot_noisecorr_paper_figure,
    plot_supp_noisecorr_hist_panels,
    plot_supp_noisecorr_shuffle_delta_panels,
    plot_supp_noisecorr_2d_panels,
    plot_supp_noisecorr_shuffle_corrected_panels,
    print_noisecorr_stats,
)

nc_stats = compute_all_noisecorr_stats(metrics, alternative="less", fdr_q=0.05, nboot=5000)

fig, axs = plot_noisecorr_paper_figure(nc_stats, title="Fixational eye-movement correction reduces noise correlations")
fig.savefig("../figures/mcfarland/noisecorr_paper_figure.pdf", bbox_inches="tight", dpi=300)

# supplements
plot_supp_noisecorr_hist_panels(metrics, savepath="../figures/mcfarland/supp_noisecorr_hist_panels.pdf")
plot_supp_noisecorr_shuffle_delta_panels(metrics, savepath="../figures/mcfarland/supp_noisecorr_shuffle_delta_panels.pdf")
plot_supp_noisecorr_2d_panels(metrics, savepath="../figures/mcfarland/supp_noisecorr_2d_panels.pdf")
plot_supp_noisecorr_shuffle_corrected_panels(metrics, savepath="../figures/mcfarland/supp_noisecorr_shuffle_corrected_panels.pdf")

#%% Plot FEM rate modulation and relationships to noise metrics

from scripts.figures_alpha import (
    compute_all_alpha_stats,
    plot_alpha_paper_figure,
    plot_supp_alpha_hist_panels,
    plot_supp_alpha_vs_metrics,
    plot_supp_alpha_shuffle_comparison,
    print_alpha_stats,
)

alpha_stats = compute_all_alpha_stats(metrics, fdr_q=0.05, nboot=5000)

fig, axs = plot_alpha_paper_figure(alpha_stats, title="FEMs account for a substantial fraction of rate modulation")
fig.savefig("../figures/mcfarland/alpha_paper_figure.pdf", bbox_inches="tight", dpi=300)

# supplemental
plot_supp_alpha_hist_panels(metrics, savepath="../figures/mcfarland/supp_alpha_hist_panels.pdf")
plot_supp_alpha_shuffle_comparison(metrics, alpha_stats, savepath="../figures/mcfarland/supp_alpha_shuffle_null_panels.pdf")

plot_supp_alpha_vs_metrics(metrics, which="uncorr", savepath="../figures/mcfarland/supp_ffuncorr_vs_1minusalpha.pdf")
plot_supp_alpha_vs_metrics(metrics, which="corr",   savepath="../figures/mcfarland/supp_ffcorr_vs_1minusalpha.pdf")
plot_supp_alpha_vs_metrics(metrics, which="delta",  savepath="../figures/mcfarland/supp_deltaff_vs_1minusalpha.pdf")

#%% Subspace Structure Alignment

from scripts.figures_subspace import (
    compute_subspace_stats,
    plot_subspace_main_figure,
    plot_subspace_vs_shuffle_figure,
    print_subspace_report,
)

S = compute_subspace_stats(
    outputs,
    model=model,
    window_idx=1,     # e.g. 20 ms window if that's idx=1 in your structure
    rep_k=5,
    min_total_spikes=500,
    psd_eps=0.0,      # start with 0; if still fragile try 1e-10 or small ridge
    n_shuffles_diag=1000,
)

fig, _ = plot_subspace_main_figure(S, title="Subspace structure and alignment")
fig.savefig("../figures/mcfarland/subspace_main.pdf", bbox_inches="tight", dpi=300)

fig, _ = plot_subspace_vs_shuffle_figure(S, title="Subspace alignment vs chance")
fig.savefig("../figures/mcfarland/subspace_vs_shuffle_supp.pdf", bbox_inches="tight", dpi=300)


#%% Create unified main figure (Panels C-K)

from scripts.figure_main_lotc import create_main_figure

# Create the unified figure with all analysis panels
# Using window_idx=0 for single-window panels (typically 10ms window)
fig_main, axs_main = create_main_figure(
    metrics=metrics,
    fano_stats=stats,
    nc_stats=nc_stats,
    alpha_stats=alpha_stats,
    subspace_stats=S,
    window_idx=0,  # 10ms window
    figsize=(11, 9),
    savepath="../figures/mcfarland/main_figure_CK.pdf",
)


#%% CNN performance.


from mcfarland_sim import bootstrap_mean_ci
ccs = []
ccmaxs = []
bpss = []
for j in range(len(outputs)):
    cc = outputs[j]['ccnorm']['ccnorm']
    ccmax = outputs[j]['ccnorm']['ccmax']
    ccmaxs.append(ccmax)
    bps = outputs[j]['bps_results']['gaborium']['bps'][outputs[j]['neuron_mask']]
    ccs.append(cc)
    bpss.append(bps)

cc = np.concatenate(ccs)
ccmax = np.concatenate(ccmaxs)
bps = np.concatenate(bpss)
bins = np.linspace(0, 1, 10)
plt.plot(cc, ccmax, '.')
for i in range(len(bins)-1):
    ix = (ccmax > bins[i]) & (ccmax <= bins[i+1])
    plt.plot(np.nanmean(cc[ix])*np.array([1, 1]), np.array([bins[i], bins[i+1]]), 'r-')
    # add text with the n and the mean cc value
    plt.text(np.nanmean(cc[ix])+.5, (bins[i] + bins[i+1])/2, f"n={np.sum(ix)}, cc={np.nanmean(cc[ix]):.2f}")

plt.xlabel('Performance of model (CC_Norm)')
plt.ylabel('Reliability of neuron (CC_Max)')
plt.title("The model is better on more reliable neurons")

ix = (bps > 0.2) & (ccmax > 0.85)
plt.figure()
plt.hist(cc[ix], bins=np.linspace(0, 1, 50))
mu = np.nanmean(cc[ix])
bootstrap_mean_ci(cc[ix], seed=0)
plt.axvline(mu, color='r', linestyle='--')
plt.title(f"CCNorm (mu={mu:.2f}, n={np.sum(ix)})")

#%% Write all statistics to lotc_stats.txt
# This section prints all analysis statistics to both console and a text file

STATS_OUTPUT_FILE = "lotc_stats.txt"

with tee_to_file(STATS_OUTPUT_FILE, mode='w'):
    print("=" * 80)
    print("LOTC MODEL ANALYSIS STATISTICS")
    print("=" * 80)
    print(f"Generated from: fixrsvp_lotc_model.py")
    print()

    # --- Fano Factor Statistics ---
    print("\n" + "=" * 80)
    print("FANO FACTOR STATISTICS")
    print("=" * 80)

    w = stats["window_ms"]
    order = np.argsort(w)

    for idx in order:
        print(f"\nWindow {int(w[idx])} ms (N valid neurons = {stats['n_valid'][idx]})")
        print(f"  Uncorr: gmean={stats['g_unc'][idx]:.3f}, IQR=[{stats['iqr_unc'][idx,0]:.3f}, {stats['iqr_unc'][idx,1]:.3f}]")
        print(f"  Corr:   gmean={stats['g_cor'][idx]:.3f}, IQR=[{stats['iqr_cor'][idx,0]:.3f}, {stats['iqr_cor'][idx,1]:.3f}]")
        print(f"  Ratio corr/uncorr = {stats['ratio'][idx]:.3f} ({stats['pct_red'][idx]:.1f}% reduction)")
        print(f"  Wilcoxon p={stats['p_wil'][idx]:.3g}, FDR p={stats['p_wil_fdr'][idx]:.3g}")
        print(f"  Shuffle null ratio 95% CI = [{stats['null_ratio_ci95'][idx,0]:.3f}, {stats['null_ratio_ci95'][idx,1]:.3f}]")
        print(f"  Empirical p (null ratio <= real ratio) = {stats['p_emp_ratio'][idx]:.3g}")
        print(f"  Slope-FF uncorr={stats['slope_unc'][idx]:.3f}, corr={stats['slope_cor'][idx]:.3f}, ratio={stats['slope_ratio'][idx]:.3f}")
        lo, hi = stats["slope_diff_ci"][idx]
        print(f"  Slope diff (corr-uncorr) = {stats['slope_diff'][idx]:.3f} with bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  Bootstrap one-sided p(diff>=0) = {stats['p_slope_one'][idx]:.3g}")

    # --- Noise Correlation Statistics ---
    print("\n" + "=" * 80)
    print("NOISE CORRELATION STATISTICS")
    print("=" * 80)
    print_noisecorr_stats(nc_stats)

    # --- Alpha (FEM Modulation) Statistics ---
    print("\n" + "=" * 80)
    print("ALPHA (FEM MODULATION FRACTION) STATISTICS")
    print("=" * 80)
    print_alpha_stats(alpha_stats)

    # --- Subspace Statistics ---
    print("\n" + "=" * 80)
    print("SUBSPACE STRUCTURE STATISTICS")
    print("=" * 80)
    print_subspace_report(S)

    # --- CNN Performance Statistics ---
    print("\n" + "=" * 80)
    print("CNN MODEL PERFORMANCE STATISTICS")
    print("=" * 80)

    ix_reliable = (bps > 0.2) & (ccmax > 0.85)
    mu_ccnorm = np.nanmean(cc[ix_reliable])
    _, (ci_lo, ci_hi) = bootstrap_mean_ci(cc[ix_reliable], seed=0)

    print(f"\nModel Performance Summary:")
    print(f"  Total neurons analyzed: {len(cc)}")
    print(f"  Reliable neurons (bps > 0.2, ccmax > 0.85): {np.sum(ix_reliable)}")
    print(f"  Mean CC_Norm on reliable neurons: {mu_ccnorm:.3f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    print("\n  Performance by reliability bin:")
    for i in range(len(bins)-1):
        ix_bin = (ccmax > bins[i]) & (ccmax <= bins[i+1])
        n_bin = np.sum(ix_bin)
        mean_cc_bin = np.nanmean(cc[ix_bin])
        print(f"    CC_Max [{bins[i]:.1f}, {bins[i+1]:.1f}]: n={n_bin}, mean CC_Norm={mean_cc_bin:.3f}")

    print("\n" + "=" * 80)
    print("END OF STATISTICS")
    print("=" * 80)

print(f"\nStatistics saved to: {STATS_OUTPUT_FILE}")

#%%