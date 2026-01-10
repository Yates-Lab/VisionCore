

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
if run_analysis:
    outputs = []
    analyzers = []

    for dataset_idx in range(len(model.names)):
        print(f"Running on dataset {dataset_idx}")
        try: # some datasets do not have fixrsvp
            output, analyzer = run_mcfarland_on_dataset(model, dataset_idx, plot=False)
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
metrics = extract_metrics(outputs)

    
#%% plot Fano Factors

from mcfarland_sim import plot_slope_estimation, bootstrap_slope_ci

window_ms = np.array([outputs[0]['results'][i]['window_ms'] for i in range(len(outputs[0]['results']))])
n = len(window_ms)

ff_u_slope = np.zeros(n)
ff_c_slope = np.zeros(n)

# bootstrap CI storage (lo, hi)
ff_u_ci = np.zeros((2, n))
ff_c_ci = np.zeros((2, n))

# optional "aleatoric" variability: IQR of per-neuron fano = var/mean
ff_u_iqr = np.zeros((2, n))
ff_c_iqr = np.zeros((2, n))

fig, axs = plt.subplots(1, n, figsize=(3*n, 3))

for i in range(n):    

    # Uncorrected
    res_u, x_u, y_u = plot_slope_estimation(
        axs[i],
        metrics[i]['erate'],
        metrics[i]['uncorr'] * metrics[i]['erate'],
        "",
        "tab:blue",
        label='Uncorrected'
    )
    ff_u_slope[i] = res_u.slope

    # Bootstrap CI for slope (epistemic, fewer assumptions)
    slope_hat, lo, hi, _ = bootstrap_slope_ci(x_u, y_u, nboot=5000, ci=0.95, rng=123 + i)
    ff_u_ci[:, i] = [lo, hi]

    # "Aleatoric" / population variability proxy: spread of per-neuron fano
    fano_u = y_u / x_u
    ff_u_iqr[:, i] = np.quantile(fano_u, [0.25, 0.75])

    # Corrected
    res_c, x_c, y_c = plot_slope_estimation(
        axs[i],
        metrics[i]['erate'],
        metrics[i]['corr'] * metrics[i]['erate'],
        f"Window {metrics[i]['window_ms']}ms",
        "tab:orange",
        label='Corrected'
    )
    ff_c_slope[i] = res_c.slope

    slope_hat, lo, hi, _ = bootstrap_slope_ci(x_c, y_c, nboot=5000, ci=0.95, rng=999 + i)
    ff_c_ci[:, i] = [lo, hi]

    fano_c = y_c / x_c
    ff_c_iqr[:, i] = np.quantile(fano_c, [0.25, 0.75])

# save figure
fig.savefig('../figures/mcfarland/population_fano_window.pdf', bbox_inches='tight', dpi=300) 

# Sort by window_ms so lines don’t zig-zag
order = np.argsort(window_ms)
window_ms = window_ms[order]

ff_u_slope = ff_u_slope[order]
ff_c_slope = ff_c_slope[order]
ff_u_ci = ff_u_ci[:, order]
ff_c_ci = ff_c_ci[:, order]
ff_u_iqr = ff_u_iqr[:, order]
ff_c_iqr = ff_c_iqr[:, order]

#%% plot population summary
# Convert (lo,hi) to asymmetric yerr for matplotlib: (2, n) = [lower_err; upper_err]
u_yerr = np.vstack([ff_u_slope - ff_u_ci[0], ff_u_ci[1] - ff_u_slope])
c_yerr = np.vstack([ff_c_slope - ff_c_ci[0], ff_c_ci[1] - ff_c_slope])

fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
axs[0].errorbar(window_ms, ff_u_slope, yerr=u_yerr, fmt='o-', capsize=3, label='Uncorrected slope (95% bootstrap CI)')
axs[0].errorbar(window_ms, ff_c_slope, yerr=c_yerr, fmt='o-', capsize=3, label='Corrected slope (95% bootstrap CI)')
axs[0].axhline(1.0, color='k', linestyle='--', alpha=0.5)
axs[0].set_xlabel("Window (ms)")
axs[0].set_ylabel("Fano / slope")
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_title("Population Fano")


def geomean(x):
    return np.exp(np.mean(np.log(x)))

ff_u_geomean = np.zeros(n)
ff_c_geomean = np.zeros(n)
for i in range(n):

    x = metrics[i]['uncorr']
    y = metrics[i]['corr']
    ix = np.isfinite(x) & np.isfinite(y)
    ff_u_geomean[i] = geomean(x[ix])
    ff_c_geomean[i] = geomean(y[ix])


axs[1].fill_between(window_ms, ff_u_iqr[0], ff_u_iqr[1], alpha=0.12, label='Uncorr per-neuron Fano IQR')
axs[1].fill_between(window_ms, ff_c_iqr[0], ff_c_iqr[1], alpha=0.12, label='Corr per-neuron Fano IQR')
axs[1].plot(window_ms, ff_u_geomean, 'o-', label='Uncorrected geomean')
axs[1].plot(window_ms, ff_c_geomean, 'o-', label='Corrected geomean')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[1].set_xlabel("Window (ms)")
axs[1].set_ylabel("Fano / geomean")
axs[1].set_title("Per-neuron Fano")
axs[1].axhline(1.0, color='k', linestyle='--', alpha=0.5)

# save fig
fig.savefig('../figures/mcfarland/fano_scaling_summary.pdf', bbox_inches='tight', dpi=300) 


#%% plot noise correlations

fig, axs = plt.subplots(1, n, figsize=(3*n, 3), sharex=True, sharey=True)
bins = np.linspace(-.5, .5, 100)

mu_u, lo_u, hi_u = np.empty(n), np.empty(n), np.empty(n)  # uncorrected
mu_c, lo_c, hi_c = np.empty(n), np.empty(n), np.empty(n)  # corrected

for i in range(n):
    # 2D hist
    cnt, xedges, yedges = np.histogram2d(
        metrics[i]['rho_uncorr'], metrics[i]['rho_corr'],
        bins=[bins, bins]
    )
    cnt = np.log1p(cnt)

    axs[i].imshow(
        cnt.T,
        origin='lower',
        aspect='auto',
        interpolation='none',
        cmap='Blues',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    axs[i].plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)
    axs[i].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[i].axvline(0, color='k', linestyle='--', alpha=0.5)
    axs[i].set_title(f"Window {metrics[i]['window_ms']}ms")
    axs[i].set_xlabel("Correlation (Uncorrected)")
    axs[i].set_ylabel("Correlation (Corrected)")

    # means + bootstrap CIs
    mu_u[i] = np.nanmean(metrics[i]['rho_uncorr'])
    sd = np.nanstd(metrics[i]['rho_uncorr'])
    lo_u[i] = mu_u[i] - sd
    hi_u[i] = mu_u[i] + sd  
    mu_c[i] = np.nanmean(metrics[i]['rho_corr'])
    sd = np.nanstd(metrics[i]['rho_corr'])
    lo_c[i] = mu_c[i] - sd
    hi_c[i] = mu_c[i] + sd
    # mu_u[i], (lo_u[i], hi_u[i]) = bootstrap_mean_ci(ffs[i]['rho_uncorr'], seed=10_000 + i)
    # mu_c[i], (lo_c[i], hi_c[i]) = bootstrap_mean_ci(ffs[i]['rho_corr'],   seed=20_000 + i)
    axs[i].plot([mu_u[i]], [mu_c[i]], 'ro')

fig.savefig('../figures/mcfarland/noise_correlations.pdf', bbox_inches='tight', dpi=300) 

#%%
fig2, ax = plt.subplots(figsize=(5, 3))

ax.errorbar(
    window_ms, mu_u,
    yerr=np.vstack([mu_u - lo_u, hi_u - mu_u]),
    fmt='o-', capsize=3, label='Uncorrected'
)
ax.errorbar(
    window_ms, mu_c,
    yerr=np.vstack([mu_c - lo_c, hi_c - mu_c]),
    fmt='o-', capsize=3, label='Corrected'
)

ax.axhline(0, color='k', lw=1, alpha=0.3)
ax.set_xlabel("Window (ms)")
ax.set_ylabel("Mean noise correlation")
ax.legend(frameon=False)
plt.tight_layout()

fig2.savefig('../figures/mcfarland/noise_correlations_mean.pdf', bbox_inches='tight', dpi=300) 



#%% Plot histogram of alpha
fig, ax = plt.subplots(1, n, figsize=(3*n, 3))

for i in range(n):
    alpha = metrics[i]['alpha']
    ax[i].hist(1 - alpha, bins=np.linspace(0, 1, 50))
    ax[i].axvline(np.nanmean(1-alpha), color='r', linestyle='--', alpha=0.5)
    ax[i].set_xlabel("1 - alpha")
    ax[i].set_ylabel("Count")
    ax[i].set_title(f"Window {metrics[i]['window_ms']}ms")

fig.savefig('../figures/mcfarland/alpha.pdf', bbox_inches='tight', dpi=300) 

# plot fano factor vs 1-alpha

for field in ['corr', 'uncorr']:
    fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
    for i in range(n):
        alpha = metrics[i]['alpha']
        ff = metrics[i][field]
        ax[i].plot(1 - alpha, ff, 'o', alpha=0.1)
        ax[i].set_xlim(0, 1)
        ax[i].set_xlabel("1 - alpha")
        ax[i].set_ylabel("Fano Factor")
        ax[i].set_title(f"Window {metrics[i]['window_ms']}ms")
    fig.savefig(f'../figures/mcfarland/ff_vs_alpha_{field}.pdf', bbox_inches='tight', dpi=300) 


#%% Subspace Structure

def get_finite(cov_matrix):
    # get valid neurons from a covariance matrix
    if hasattr(cov_matrix, "detach"):
        cov_matrix = cov_matrix.detach().cpu().numpy()
    valid = np.isfinite(np.diag(cov_matrix))
    return valid

def index_cov(cov_matrix, indices):
    # index into a square matrix
    return cov_matrix[indices][:, indices]

def get_dominant_subspace(cov_matrix, k):
    # Returns top-k eigenvectors (columns)
    vals, vecs = torch.linalg.eigh(cov_matrix)
    return vecs[:, -k:].flip(1)

def get_spectra(cov_matrix, normalization_factor=1.0):
    # Returns eigenvalues normalized by a global factor (e.g., Total Variance)
    vals = torch.linalg.eigvalsh(cov_matrix)
    vals = vals.flip(0) # Descending order
    vals = vals / normalization_factor 
    return vals.cpu().numpy()

def get_effective_dimensionality(cov_matrix):
    """Computes Participation Ratio: (tr(C))^2 / tr(C^2)."""
    tr = torch.trace(cov_matrix)
    tr_sq = torch.trace(cov_matrix @ cov_matrix)
    return (tr**2 / tr_sq).item()

def symmetric_subspace_overlap(U_p, U_f):
    k = U_p.shape[1]
    cross_proj = U_p.T @ U_f
    return (torch.norm(cross_proj, p='fro') ** 2 / k).item()

def directional_variance_capture(target_cov, source_basis):
    """Fraction of target_cov variance captured by source_basis."""
    captured_var = torch.trace(source_basis.T @ target_cov @ source_basis)
    total_var = torch.trace(target_cov)
    return (captured_var / total_var).item()

# --- Analysis & Data Collection Loop ---
def run_full_analysis(outputs, model, window_idx=1, rep_k=5, max_dims_to_plot=50):
    
    # Containers for Statistics (Text Report)
    stats = {
        'fem_dim': [], 'psth_dim': [],
        'overlap_k1': [], 'overlap_rep': [],
        'var_f_given_p': [], 'var_p_given_f': []
    }
    
    # Containers for Plotting
    plot_data = {
        'spectra_psth': [], 'spectra_fem': [],
        'scatter_x': [], 'scatter_y': [], 'dataset_names': []
    }

    n_sessions = len(outputs)
    
    for i in range(n_sessions):
        # A. Load Matrices
        Cpsth = outputs[i]['last_mats'][window_idx]['PSTH']
        Cfem = outputs[i]['last_mats'][window_idx]['FEM']
        Ctotal = outputs[i]['last_mats'][window_idx]['Total']
        name = model.names[i]
        
        # Sanitize (remove NaNs/Infs)
        valid = get_finite(Cpsth) & get_finite(Cfem)
        Cpsth = torch.from_numpy(index_cov(Cpsth, valid))
        Cfem = torch.from_numpy(index_cov(Cfem, valid))
        Ctotal = torch.from_numpy(index_cov(Ctotal, valid))

        # B. Calculate "Total Shared Variance" for Normalization
        total_shared_var = torch.trace(Ctotal)
        
        # C. Compute Spectra (for Figure 1)
        plot_data['spectra_psth'].append(get_spectra(Cpsth, total_shared_var))
        plot_data['spectra_fem'].append(get_spectra(Cfem, total_shared_var))
        
        # D. Compute Subspace Statistics (for Figure 2 & Text)
        stats['fem_dim'].append(get_effective_dimensionality(Cfem))
        stats['psth_dim'].append(get_effective_dimensionality(Cpsth))
        
        # Subspaces
        curr_k = min(rep_k, Cfem.shape[0])
        Up_1 = get_dominant_subspace(Cpsth, 1)
        Uf_1 = get_dominant_subspace(Cfem, 1)
        Up_k = get_dominant_subspace(Cpsth, curr_k)
        Uf_k = get_dominant_subspace(Cfem, curr_k)
        
        # Metrics
        stats['overlap_k1'].append(symmetric_subspace_overlap(Up_1, Uf_1))
        stats['overlap_rep'].append(symmetric_subspace_overlap(Up_k, Uf_k))
        
        # Variance Capture
        v_f_p = directional_variance_capture(Cfem, Up_k) # FEM var in PSTH space
        v_p_f = directional_variance_capture(Cpsth, Uf_k) # PSTH var in FEM space
        
        stats['var_f_given_p'].append(v_f_p)
        stats['var_p_given_f'].append(v_p_f)
        
        # Store for scatter plot
        plot_data['scatter_y'].append(v_f_p) # Y-axis: FEM explained by PSTH
        plot_data['scatter_x'].append(v_p_f) # X-axis: PSTH explained by FEM
        plot_data['dataset_names'].append(name)

    return stats, plot_data

# --- 3. Execute Analysis ---
final_stats, plot_data = run_full_analysis(outputs, model, window_idx=1, rep_k=5)

# --- 4. Generate Text Report ---
def fmt_stat(data_list):
    arr = np.array(data_list)
    return f"{np.mean(arr):.2f} ± {np.std(arr)/np.sqrt(len(arr)):.2f}"

print("--- RESULTS SUMMARY ---")
text = (
    f"{len(outputs)} sessions.\n"
    f"(Participation Ratio (FEM): {fmt_stat(final_stats['fem_dim'])}). \n"
    f"(Participation Ratio (PSTH): {fmt_stat(final_stats['psth_dim'])}). \n"
    f"Subspace Alignment:\n"
    f"(Overlap k=5: {fmt_stat(final_stats['overlap_rep'])}). \n"
    f"(Overlap k=1: {fmt_stat(final_stats['overlap_k1'])}). \n "
    
    f"PSTH subspace captured more of the FEM variance \n"
    f"(Var explained: {fmt_stat(final_stats['var_f_given_p'])}) than vice versa \n"
    f"(Var explained: {fmt_stat(final_stats['var_p_given_f'])}).\n"
    f"If A >= B, then eye movements operate within the stimulus manifold."
)
print(text)
print("-" * 30)

# --- 5. Generate Figures ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Eigenspectra (Normalized by Total Shared Variance)
ax = axes[0]
max_dims = 50 
# Plot individual lines
for sp_p, sp_f in zip(plot_data['spectra_psth'], plot_data['spectra_fem']):
    limit = min(len(sp_p), max_dims)
    dims = np.arange(1, limit + 1)
    ax.loglog(dims, sp_p[:limit], color='black', alpha=0.1, lw=1)
    ax.loglog(dims, sp_f[:limit], color='red', alpha=0.1, lw=1)

# Plot Averages
min_len = min([len(s) for s in plot_data['spectra_psth']])
limit_avg = min(min_len, max_dims)
dims_avg = np.arange(1, limit_avg + 1)
avg_psth = np.mean([s[:limit_avg] for s in plot_data['spectra_psth']], axis=0)
avg_fem = np.mean([s[:limit_avg] for s in plot_data['spectra_fem']], axis=0)

ax.loglog(dims_avg, avg_psth, color='black', lw=3, label='Stimulus (PSTH)')
ax.loglog(dims_avg, avg_fem, color='red', lw=3, label='Eye Movement (FEM)')
ax.set_xlabel('Dimension (PC)')
ax.set_ylabel('Fraction of Total Variance')
ax.set_title('Eigenspectra magnitude')
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(1, limit_avg)
ax.set_ylim(1e-5, 1)

# Panel 2: Subspace Containment Scatter
ax = axes[1]
x = np.array(plot_data['scatter_x']) # PSTH var explained by FEM (X)
y = np.array(plot_data['scatter_y']) # FEM var explained by PSTH (Y)

# Scatter points
ax.scatter(x, y, color='purple', s=80, alpha=0.8, edgecolors='white')

# Reference Line
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Symmetric')
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel('PSTH Var explained by FEM Subspace\n(Do eyes explain the image?)')
ax.set_ylabel('FEM Var explained by PSTH Subspace\n(Does image explain the eyes?)')
ax.set_title('Subspace Alignment (k=5)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Statistics Box
mean_x = np.mean(x)
mean_y = np.mean(y)
stat_str = f"Mean Y: {mean_y:.2f}\nMean X: {mean_x:.2f}"
ax.text(0.05, 0.95, stat_str, transform=ax.transAxes, va='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

# --- 6. Outlier Debugging ---
print("\n--- OUTLIER INSPECTION ---")
# Identify sessions where "FEM var explained by PSTH" is low (points below the main cluster)
# Threshold chosen visually based on your previous plot (e.g., < 0.70)

outliers = [
    (name, y_val, x_val) 
    for name, y_val, x_val in zip(plot_data['dataset_names'], y, x) 
    if y_val < x_val
]

if outliers:
    print(f"Found {len(outliers)} sessions where FEM subspace captures PSTH more than vice versa:")
    for name, y_val, x_val in outliers:
        print(f"  Dataset: {name} | FEM_exp_by_PSTH (Y): {y_val:.3f} | PSTH_exp_by_FEM (X): {x_val:.3f}")
else:
    print("No strong outliers found below threshold.")

#%% check a pair
# analyzers[0].inspect_neuron_pair(20,20, 20, ax=None, show=True)

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

#%% WHAT IS ALL THIS CODE DOING?

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compute_comparative_stats(CTotal, CResidual, Cpsth, Crate, eps=1e-12):
    """
    Computes comparative variance metrics for Model vs. PSTH.
    """
    # Force numpy and valid subset
    mats = [CTotal, CResidual, Cpsth, Crate]
    valid_mask = np.ones(CTotal.shape[0], dtype=bool)
    for M in mats:
        valid_mask &= np.isfinite(np.diag(M))
    
    def get_tr(M): return np.trace(M[np.ix_(valid_mask, valid_mask)])

    tr_Total = get_tr(CTotal)
    tr_Res   = get_tr(CResidual)
    tr_PSTH  = get_tr(Cpsth)
    tr_Rate  = get_tr(Crate)
    tr_Expl  = tr_Total - tr_Res

    # 1. Total Variance Explained (0 to 1)
    # How much of the raw signal does the model capture?
    ve_model = 1.0 - (tr_Res / (tr_Total + eps))
    
    # How much of the raw signal is just the PSTH?
    ve_psth  = tr_PSTH / (tr_Total + eps)

    # 2. Explainable Variance (Normalized by PSTH)
    # If > 1, the model is capturing non-PSTH (eye movement) variance.
    fev_psth = tr_Expl / (tr_PSTH + eps)

    # 3. Explainable Variance (Normalized by True Rate)
    # Should ideally be 1.0 if model is perfect.
    fev_rate = tr_Expl / (tr_Rate + eps)

    # 4. Alpha (PSTH var / Rate var)
    # Low alpha = Eye movements dominate
    alpha = tr_PSTH / (tr_Rate + eps)

    return {
        've_model': ve_model,
        've_psth': ve_psth,
        'fev_psth': fev_psth,
        'fev_rate': fev_rate,
        'alpha': alpha
    }

# --- Run Loop ---
comparison_data = []
for j, out in enumerate(outputs):
    # Iterate over windows (or pick a specific one, e.g., index 1 for 20ms)
    for w_idx, win_ms in enumerate(out['windows']):
        mats = out['last_mats'][w_idx]
        Cres = out['last_mats_residuals'][w_idx]['Total']
        
        stats = compute_comparative_stats(
            mats['Total'], Cres, mats['PSTH'], mats['Intercept']
        )
        stats['window'] = win_ms
        stats['dataset'] = j
        comparison_data.append(stats)

# Convert to simple arrays for plotting specific windows
def get_data_for_window(target_ms):
    subset = [d for d in comparison_data if d['window'] == target_ms]
    return {k: np.array([d[k] for d in subset]) for k in subset[0].keys()}
# %%
# Select a representative window (e.g., 20ms or 40ms)
win_data = get_data_for_window(40) # Adjust to your preferred window

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot A: Model vs. PSTH (Total Variance) ---
# Question: Does the model explain more raw variance than the PSTH alone?
ax = axes[0]
ax.scatter(win_data['ve_psth'], win_data['ve_model'], c='k', alpha=0.6)
ax.plot([0, 1], [0, 1], 'r--', label='Unity (Model = PSTH)')
ax.set_xlabel('Total Var Explained by PSTH (Signal)')
ax.set_ylabel('Total Var Explained by Model')
ax.set_title('Raw Performance Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
# Interpretation: Points above the line mean the model is learning useful non-PSTH variance.

# --- Plot B: The Ceiling Break (FEV_PSTH vs Alpha) ---
# Question: Does the model break the PSTH ceiling specifically when FEMs are strong?
ax = axes[1]
sc = ax.scatter(win_data['alpha'], win_data['fev_psth'], c=win_data['ve_model'], cmap='viridis')
ax.axhline(1.0, color='r', linestyle='--', label='PSTH Ceiling')
ax.axvline(1.0, color='k', linestyle=':', alpha=0.3)
ax.set_xlabel(r'$\alpha$ (PSTH Var / Total Rate Var)')
ax.set_ylabel(r'$FEV_{PSTH}$ (Model Expl / PSTH Var)')
ax.set_title('The Ceiling Break')
plt.colorbar(sc, ax=ax, label='Total Model $R^2$')
ax.invert_xaxis() # Low alpha (strong eyes) on the right? Or standard left-to-right.
# Let's keep 0 -> 1. Low alpha is on the left.
# Interpretation: We expect points at low alpha (left) to go ABOVE the red line (y > 1).

# --- Plot C: True Model Performance (FEV_Rate vs Alpha) ---
# Question: Is the model consistent across cell types, or does it fail when alpha is low?
ax = axes[2]
ax.scatter(win_data['alpha'], win_data['fev_rate'], c='blue', alpha=0.6)
ax.axhline(1.0, color='r', linestyle='--', label='Perfect Model')
ax.set_xlabel(r'$\alpha$ (PSTH Var / Total Rate Var)')
ax.set_ylabel(r'$FEV_{Rate}$ (Model Expl / Total Rate Var)')
ax.set_title('Normalized Performance')
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3)
# Interpretation: Ideally, this is flat at y=1.0 regardless of alpha. 
# If it drops at low alpha, the model struggles with eye movements.

plt.tight_layout()
plt.show()
# %%
