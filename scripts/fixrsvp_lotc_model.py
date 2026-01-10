

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
import torch
import torch
import numpy as np

# --- Core Math Functions ---
def get_finite(cov_matrix):
    if hasattr(cov_matrix, "detach"):
        cov_matrix = cov_matrix.detach().cpu().numpy()
    valid = np.isfinite(np.diag(cov_matrix))
    return valid

def index_cov(cov_matrix, indices):
    return cov_matrix[indices][:, indices]

def get_dominant_subspace(cov_matrix, k):
    vals, vecs = torch.linalg.eigh(cov_matrix)
    return vecs[:, -k:].flip(1)

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
    captured_var = torch.trace(source_basis.T @ target_cov @ source_basis)
    total_var = torch.trace(target_cov)
    return (captured_var / total_var).item()

# --- Analysis Wrapper ---
def analyze_alignment_stats(outputs, model, window_idx=1, rep_k=5):
    stats = {
        'fem_dim': [],         # Effective dimensionality of FEM
        'overlap_k1': [],      # Overlap of dominant dimension
        'overlap_rep': [],     # Overlap of representative subspace (e.g. k=5)
        'var_f_given_p': [],   # How much FEM var is in PSTH subspace
        'var_p_given_f': []    # How much PSTH var is in FEM subspace
    }
    
    n_sessions = len(outputs)
    
    for i in range(n_sessions):
        # 1. Load and Sanitize
        Cpsth = outputs[i]['last_mats'][window_idx]['PSTH']
        Cfem = outputs[i]['last_mats'][window_idx]['FEM']
        
        valid = get_finite(Cpsth) & get_finite(Cfem)
        Cpsth = torch.from_numpy(index_cov(Cpsth, valid))
        Cfem = torch.from_numpy(index_cov(Cfem, valid))
        
        # 2. Compute Dimensionality
        stats['fem_dim'].append(get_effective_dimensionality(Cfem))
        
        # 3. Compute k=1 stats (Dominant Dimension)
        Up_1 = get_dominant_subspace(Cpsth, 1)
        Uf_1 = get_dominant_subspace(Cfem, 1)
        stats['overlap_k1'].append(symmetric_subspace_overlap(Up_1, Uf_1))
        
        # 4. Compute representative k stats (Subspace)
        # Ensure k doesn't exceed matrix size
        curr_k = min(rep_k, Cfem.shape[0])
        Up_k = get_dominant_subspace(Cpsth, curr_k)
        Uf_k = get_dominant_subspace(Cfem, curr_k)
        
        stats['overlap_rep'].append(symmetric_subspace_overlap(Up_k, Uf_k))
        stats['var_f_given_p'].append(directional_variance_capture(Cfem, Up_k))
        stats['var_p_given_f'].append(directional_variance_capture(Cpsth, Uf_k))

    return stats

# --- Helper to format "Mean +/- SEM" ---
def fmt_stat(data_list):
    arr = np.array(data_list)
    mean = np.mean(arr)
    sem = np.std(arr) / np.sqrt(len(arr))
    return f"{mean:.2f} ± {sem:.2f}"

# --- Execution & Report Generation ---
# Using k=5 as the 'representative' subspace size for the general comparisons
final_stats = analyze_alignment_stats(outputs, model, window_idx=1, rep_k=5)

print("--- Generated Results Text ---")
text = (
    f"Across the {len(outputs)} sessions, Sigma_fem was consistently low rank "
    f"(Participation Ratio: {fmt_stat(final_stats['fem_dim'])}). \n"
    f"We evaluated whether this low rank variability was aligned with the PSTH covariance "
    f"(Sigma_PSTH) by computing the symmetric subspace overlap (see Supplemental Methods). \n"
    f"We found that Sigma_fem was highly aligned to the PSTH subspace "
    f"(Overlap k=5: {fmt_stat(final_stats['overlap_rep'])}). \n"
    f"This supports the interpretation that in our stimulus condition eye movements manifest "
    f"as a shared gain modulator. \n"
    f"The dominant dimension is highly aligned (Overlap k=1: {fmt_stat(final_stats['overlap_k1'])}), "
    f"meaning the single biggest thing FEMs do is modulate the pattern present in the PSTH. \n"
    f"This might explain why many datasets show low dimensional shared variability "
    f"(cite Liska et al). \n"
    f"Further, we found that the subspace of Sigma_PSTH explained more variance about "
    f"Sigma_FEM (Var explained: {fmt_stat(final_stats['var_f_given_p'])}) than vice versa "
    f"(Var explained: {fmt_stat(final_stats['var_p_given_f'])}). \n"
    f"This supports the interpretation that eye movements are not creating new patterns "
    f"orthogonal to the stimulus. This is consistent with an interpretation that FEMs drive "
    f"information-limiting correlations in this stimulus condition (Moreno Bote)."
)
print(text)

#%%
import matplotlib.pyplot as plt
import torch
import numpy as np

# --- Helper Functions (Same as before) ---
def get_finite(cov_matrix):
    if hasattr(cov_matrix, "detach"):
        cov_matrix = cov_matrix.detach().cpu().numpy()
    valid = np.isfinite(np.diag(cov_matrix))
    return valid

def index_cov(cov_matrix, indices):
    return cov_matrix[indices][:, indices]

def get_spectra(cov_matrix):
    # Get eigenvalues, sort descending, normalize by total variance (trace)
    vals = torch.linalg.eigvalsh(cov_matrix)
    vals = vals.flip(0) # Descending
    vals = vals / vals.sum() # Normalize to fraction of variance
    return vals.cpu().numpy()

def directional_variance_capture(target_cov, source_basis):
    captured_var = torch.trace(source_basis.T @ target_cov @ source_basis)
    total_var = torch.trace(target_cov)
    return (captured_var / total_var).item()

def get_dominant_subspace(cov_matrix, k):
    vals, vecs = torch.linalg.eigh(cov_matrix)
    return vecs[:, -k:].flip(1)

# --- Data Collection Container ---
spectra_psth = []
spectra_fem = []
var_capture_data = {'f_given_p': [], 'p_given_f': []}

# Parameters
window_idx = 1
rep_k = 5  # Subspace size for Fig 3
max_dims_to_plot = 50 # Limit x-axis for spectra since N~120

# --- Analysis Loop ---
for i in range(len(outputs)):
    # Load Matrices
    Cpsth = outputs[i]['last_mats'][window_idx]['PSTH']
    Cfem = outputs[i]['last_mats'][window_idx]['FEM']

    # Sanitize
    valid = get_finite(Cpsth) & get_finite(Cfem)
    Cpsth = torch.from_numpy(index_cov(Cpsth, valid))
    Cfem = torch.from_numpy(index_cov(Cfem, valid))

    # 1. Collect Spectra (Fig 1)
    spectra_psth.append(get_spectra(Cpsth))
    spectra_fem.append(get_spectra(Cfem))

    # 2. Collect Variance Capture (Fig 3)
    # Ensure k doesn't exceed matrix size
    curr_k = min(rep_k, Cfem.shape[0])
    
    # Get subspaces
    Up_k = get_dominant_subspace(Cpsth, curr_k)
    Uf_k = get_dominant_subspace(Cfem, curr_k)
    
    # Compute captures
    var_capture_data['f_given_p'].append(directional_variance_capture(Cfem, Up_k))
    var_capture_data['p_given_f'].append(directional_variance_capture(Cpsth, Uf_k))

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Figure 1: Dual Eigenspectra (Log-Log) ---
ax = axes[0]
# Plot individual lines (faint)
for sp_p, sp_f in zip(spectra_psth, spectra_fem):
    # Cut off at max_dims_to_plot or length
    limit = min(len(sp_p), max_dims_to_plot)
    dims = np.arange(1, limit + 1)
    
    ax.loglog(dims, sp_p[:limit], color='black', alpha=0.15, lw=1)
    ax.loglog(dims, sp_f[:limit], color='red', alpha=0.15, lw=1)

# Plot Averages (Bold)
# (Note: averaging spectra of different lengths requires care, here assuming mostly similar N or using min len)
min_len = min([len(s) for s in spectra_psth])
limit_avg = min(min_len, max_dims_to_plot)
dims_avg = np.arange(1, limit_avg + 1)

avg_psth = np.mean([s[:limit_avg] for s in spectra_psth], axis=0)
avg_fem = np.mean([s[:limit_avg] for s in spectra_fem], axis=0)

ax.loglog(dims_avg, avg_psth, color='black', lw=3, label='Stimulus (PSTH)')
ax.loglog(dims_avg, avg_fem, color='red', lw=3, label='Eye Movement (FEM)')

ax.set_xlabel('Dimension (PC)')
ax.set_ylabel('Fraction of Variance')
ax.set_title('Shared Geometry: Stimulus vs. FEM Spectra')
ax.legend()
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(1, limit_avg)
ax.set_ylim(1e-4, 1)

# --- Figure 3: Variance Capture Asymmetry ---
ax = axes[1]
x = var_capture_data['p_given_f'] # PSTH var captured by FEM subspace
y = var_capture_data['f_given_p'] # FEM var captured by PSTH subspace

ax.scatter(x, y, color='purple', s=60, alpha=0.8, edgecolors='w')

# Diagonal line
lims = [0, 1]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Symmetric Capture')

ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel('PSTH Var explained by FEM Subspace\n(Do eyes explain the image?)')
ax.set_ylabel('FEM Var explained by PSTH Subspace\n(Does image explain the eyes?)')
ax.set_title(f'Subspace Containment (k={rep_k})')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add summary stats text
mean_x = np.mean(x)
mean_y = np.mean(y)
ax.text(0.05, 0.95, f"Mean Y: {mean_y:.2f}\nMean X: {mean_x:.2f}", 
        transform=ax.transAxes, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

#%% check a pair
# analyzers[0].inspect_neuron_pair(20,20, 20, ax=None, show=True)

#%% CNN performance. 
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

plt.xlabel('CCNorm')
plt.ylabel('CCMax')
plt.title("The model is better on more reliable neurons")

ix = (bps > 0.2) & (ccmax > 0.85)
plt.figure()
plt.hist(cc[ix], bins=np.linspace(0, 1, 50))
mu = np.nanmean(cc[ix])
bootstrap_mean_ci(cc[ix], seed=0)
plt.axvline(mu, color='r', linestyle='--')
plt.title(f"CCNorm (mu={mu:.2f}, n={np.sum(ix)})")

#%%

# ----------------------------
# helpers
# ----------------------------
def _as_np(A):
    """Convert torch.Tensor or array-like to float64 numpy array."""
    if A is None:
        return None
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
    A = np.asarray(A, dtype=np.float64)
    return A

def _sym(A):
    """Symmetrize (keeps NaNs in place)."""
    return 0.5 * (A + A.T)

def _valid_mask_from_mats(*mats):
    """
    Returns boolean mask of neurons that are finite across all provided matrices
    (both diagonal and entire row/col).
    """
    mats = [m for m in mats if m is not None]
    if len(mats) == 0:
        raise ValueError("No matrices provided.")
    N = mats[0].shape[0]
    ok = np.ones(N, dtype=bool)
    for M in mats:
        if M.shape != (N, N):
            raise ValueError("All matrices must have the same shape (N, N).")
        # require finite diagonal
        ok &= np.isfinite(np.diag(M))
    return ok

def _restrict(M, mask):
    """Restrict square matrix M to mask."""
    return M[np.ix_(mask, mask)]

def _psd_part(A, eps=0.0):
    """
    Project symmetric matrix to its PSD part by zeroing negative eigenvalues.
    Useful when estimates are noisy.
    """
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


# ----------------------------
# 1) fractions explained
# ----------------------------
def fraction_explained(
    CTotal,
    CResidual,
    Cpsth=None,
    Crate=None,
    clip=True,
    eps=1e-12,
    require_finite=True,
):
    """
    Compute per-neuron and global "fraction explained" metrics from covariance matrices.

    Definitions:
      CExpl = CTotal - CResidual

      Raw (total-variance) fraction explained per neuron:
        VE_raw[c] = 1 - diag(CResidual)[c] / diag(CTotal)[c]

      Ceiling-normalized (explainable) fractions:
        FEV_psth[c] = (diag(CTotal)-diag(CResidual)) / diag(Cpsth)
        FEV_rate[c] = (diag(CTotal)-diag(CResidual)) / diag(Crate)

      Global trace versions:
        VE_raw_trace  = 1 - tr(CResidual)/tr(CTotal)
        FEV_psth_trace = tr(CExpl)/tr(Cpsth)
        FEV_rate_trace = tr(CExpl)/tr(Crate)

    Parameters
    ----------
    clip : bool
        If True, clip per-neuron and trace fractions into [0, 1] where applicable.
        (FEV can exceed 1 due to estimation noise; clipping is for display.)
    require_finite : bool
        If True, only compute trace-level metrics on neurons that are finite across
        all matrices provided (CTotal, CResidual, and Cpsth/Crate if not None).

    Returns
    -------
    dict with keys:
      - "VE_raw" (N,)
      - "FEV_psth" (N,) or None
      - "FEV_rate" (N,) or None
      - "VE_raw_trace"
      - "FEV_psth_trace" or None
      - "FEV_rate_trace" or None
      - "mask_used" (N,) bool (only meaningful if require_finite=True)
    """
    CTotal = _as_np(CTotal)
    CResidual = _as_np(CResidual)
    Cpsth = _as_np(Cpsth)
    Crate = _as_np(Crate)

    if CTotal.shape != CResidual.shape or CTotal.ndim != 2 or CTotal.shape[0] != CTotal.shape[1]:
        raise ValueError("CTotal and CResidual must be (N, N) square matrices of same shape.")

    N = CTotal.shape[0]
    dT = np.diag(CTotal)
    dR = np.diag(CResidual)
    CExpl = _sym(CTotal - CResidual)

    # Per-neuron raw VE
    VE_raw = 1.0 - (dR / (dT + eps))

    FEV_psth = None
    if Cpsth is not None:
        if Cpsth.shape != (N, N):
            raise ValueError("Cpsth must have shape (N, N).")
        dP = np.diag(Cpsth)
        FEV_psth = (dT - dR) / (dP + eps)

    FEV_rate = None
    if Crate is not None:
        if Crate.shape != (N, N):
            raise ValueError("Crate must have shape (N, N).")
        dQ = np.diag(Crate)
        FEV_rate = (dT - dR) / (dQ + eps)

    if clip:
        VE_raw = np.clip(VE_raw, 0.0, 1.0)
        if FEV_psth is not None:
            FEV_psth = np.clip(FEV_psth, 0.0, 1.0)
        if FEV_rate is not None:
            FEV_rate = np.clip(FEV_rate, 0.0, 1.0)

    # Trace-level metrics (optionally on common finite subset)
    if require_finite:
        mask = _valid_mask_from_mats(CTotal, CResidual, *(m for m in [Cpsth, Crate] if m is not None))
        CT = _restrict(CTotal, mask)
        CR = _restrict(CResidual, mask)
        CE = _sym(CT - CR)
        CP = _restrict(Cpsth, mask) if Cpsth is not None else None
        CQ = _restrict(Crate, mask) if Crate is not None else None
    else:
        mask = np.ones(N, dtype=bool)
        CT, CR, CE, CP, CQ = CTotal, CResidual, CExpl, Cpsth, Crate

    tr_CT = np.trace(CT)
    tr_CR = np.trace(CR)
    VE_raw_trace = 1.0 - (tr_CR / (tr_CT + eps))

    FEV_psth_trace = None
    if CP is not None:
        FEV_psth_trace = np.trace(CE) / (np.trace(CP) + eps)

    FEV_rate_trace = None
    if CQ is not None:
        FEV_rate_trace = np.trace(CE) / (np.trace(CQ) + eps)

    if clip:
        VE_raw_trace = float(np.clip(VE_raw_trace, 0.0, 1.0))
        if FEV_psth_trace is not None:
            FEV_psth_trace = float(np.clip(FEV_psth_trace, 0.0, 1.0))
        if FEV_rate_trace is not None:
            FEV_rate_trace = float(np.clip(FEV_rate_trace, 0.0, 1.0))

    return {
        "VE_raw": VE_raw,
        "FEV_psth": FEV_psth,
        "FEV_rate": FEV_rate,
        "VE_raw_trace": float(VE_raw_trace),
        "FEV_psth_trace": None if FEV_psth_trace is None else float(FEV_psth_trace),
        "FEV_rate_trace": None if FEV_rate_trace is None else float(FEV_rate_trace),
        "mask_used": mask,
    }


# ----------------------------
# 2) subspace alignment
# ----------------------------
def subspace_alignment(
    CTotal,
    CResidual,
    Cpsth,
    Crate,
    basis="fem",     # "fem" (Crate-Cpsth), "rate" (Crate), "psth" (Cpsth)
    k=5,
    use_psd_explained=False,
    eps=1e-12,
):
    """
    Measure how much of the *explained covariance* (CExpl = CTotal - CResidual)
    lies in a chosen low-dimensional subspace.

    Subspace is defined by the top-k eigenvectors of:
      - basis="fem":  CFEM = Crate - Cpsth
      - basis="rate": Crate
      - basis="psth": Cpsth

    Alignment metric:
      frac_k = tr(P_k * CExpl) / tr(CExpl)
    where P_k = U_k U_k^T projects onto the top-k eigenvectors U_k.

    Notes:
      - This is a *trace / variance* fraction: it’s always interpretable.
      - If CExpl is noisy and has negative eigenvalues, set use_psd_explained=True.

    Returns
    -------
    dict with:
      - "frac_k": float
      - "frac_k_curve": (k_eff,) curve for 1..k_eff
      - "k_eff": int
      - "mask_used": boolean mask of included neurons
    """
    CTotal = _as_np(CTotal)
    CResidual = _as_np(CResidual)
    Cpsth = _as_np(Cpsth)
    Crate = _as_np(Crate)

    # finite subset across all inputs (keeps the eigendecomp sane)
    mask = _valid_mask_from_mats(CTotal, CResidual, Cpsth, Crate)
    CT = _restrict(CTotal, mask)
    CR = _restrict(CResidual, mask)
    CP = _restrict(Cpsth, mask)
    CQ = _restrict(Crate, mask)

    CExpl = _sym(CT - CR)
    if use_psd_explained:
        CExpl_use = _psd_part(CExpl, eps=0.0)
    else:
        CExpl_use = CExpl

    tr_expl = float(np.trace(CExpl_use))
    if not np.isfinite(tr_expl) or tr_expl <= eps:
        return {"frac_k": np.nan, "frac_k_curve": np.array([]), "k_eff": 0, "mask_used": mask}

    if basis == "fem":
        Cbasis = _sym(CQ - CP)
    elif basis == "rate":
        Cbasis = _sym(CQ)
    elif basis == "psth":
        Cbasis = _sym(CP)
    else:
        raise ValueError("basis must be one of {'fem','rate','psth'}")

    # eigenvectors (descending)
    w, V = np.linalg.eigh(Cbasis)
    order = np.argsort(w)[::-1]
    V = V[:, order]

    k_eff = int(min(k, V.shape[1]))
    frac_curve = []
    for kk in range(1, k_eff + 1):
        U = V[:, :kk]                 # (N, kk)
        # tr(P C) = tr(U^T C U)
        numer = float(np.trace(U.T @ CExpl_use @ U))
        frac_curve.append(numer / tr_expl)

    frac_curve = np.array(frac_curve, dtype=np.float64)
    frac_k = float(frac_curve[-1]) if k_eff > 0 else np.nan

    # clip for numerical wobble
    frac_curve = np.clip(frac_curve, 0.0, 1.0)
    frac_k = float(np.clip(frac_k, 0.0, 1.0))

    return {
        "frac_k": frac_k,
        "frac_k_curve": frac_curve,
        "k_eff": k_eff,
        "mask_used": mask,
    }


#%%


#%%
# ----------------------------
# utilities
# ----------------------------
def _as_np(A):
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
    return np.asarray(A)

def cov_from_residuals(residuals, ddof=1):
    """
    residuals: (N_samples, N_cells)
    returns: (N_cells, N_cells) sample covariance
    """
    X = _as_np(residuals).astype(np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    X = X - X.mean(axis=0, keepdims=True)
    return (X.T @ X) / max(1, (X.shape[0] - ddof))

def diag_safe(M):
    M = _as_np(M)
    return np.diag(M)

def clip01(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 0.0, 1.0)

# ----------------------------
# collect metrics per window
# ----------------------------
def collect_window_metrics(results, mats_save, CResidual_list, k_list=(1,2,5,10), basis="fem"):
    """
    results: list of dicts from run_sweep (has window_ms, alpha, etc.)
    mats_save: list of dicts with keys: "Total","PSTH","Intercept","FEM",...
    CResidual_list: list of (N,N) residual covariances, aligned to results/mats_save
    
    returns: dict with arrays + per-window per-neuron vectors
    """
    win_ms = np.array([r["window_ms"] for r in results], dtype=float)

    VE_raw_trace = []
    FEV_psth_trace = []
    FEV_rate_trace = []

    # store per-window per-neuron vectors for later scatter/hist
    per = {
        "alpha": [],
        "VE_raw": [],
        "FEV_psth": [],
        "FEV_rate": [],
        "diag_total": [],
        "diag_res": [],
        "diag_psth": [],
        "diag_rate": [],
    }

    # subspace alignment: frac_k for each k and window
    align = {k: [] for k in k_list}

    for r, mats, Cres in zip(results, mats_save, CResidual_list):
        CTotal = mats["Total"]
        Cpsth  = mats["PSTH"]
        Crate  = mats["Intercept"]

        fe = fraction_explained(CTotal, Cres, Cpsth=Cpsth, Crate=Crate, clip=False, require_finite=True)

        VE_raw_trace.append(fe["VE_raw_trace"])
        FEV_psth_trace.append(fe["FEV_psth_trace"])
        FEV_rate_trace.append(fe["FEV_rate_trace"])

        per["alpha"].append(_as_np(r["alpha"]))
        per["VE_raw"].append(fe["VE_raw"])
        per["FEV_psth"].append(fe["FEV_psth"])
        per["FEV_rate"].append(fe["FEV_rate"])
        per["diag_total"].append(diag_safe(CTotal))
        per["diag_res"].append(diag_safe(Cres))
        per["diag_psth"].append(diag_safe(Cpsth))
        per["diag_rate"].append(diag_safe(Crate))

        for k in k_list:
            al = subspace_alignment(CTotal, Cres, Cpsth, Crate, basis=basis, k=k, use_psd_explained=True)
            align[k].append(al["frac_k"])

    out = {
        "window_ms": win_ms,
        "VE_raw_trace": np.array(VE_raw_trace, float),
        "FEV_psth_trace": np.array(FEV_psth_trace, float),
        "FEV_rate_trace": np.array(FEV_rate_trace, float),
        "per": per,
        "align": {k: np.array(v, float) for k, v in align.items()},
        "basis": basis,
    }
    return out

# ----------------------------
# plotting
# ----------------------------
def plot_trace_metrics(M, clip=True):
    """
    M: output of collect_window_metrics
    """
    w = M["window_ms"]
    ve = M["VE_raw_trace"]
    fp = M["FEV_psth_trace"]
    fr = M["FEV_rate_trace"]

    if clip:
        ve = clip01(ve)
        if fp is not None: fp = clip01(fp)
        if fr is not None: fr = clip01(fr)

    plt.figure()
    plt.plot(w, ve, marker="o", label="VE_raw_trace")
    if fp is not None and np.isfinite(fp).any():
        plt.plot(w, fp, marker="o", label="FEV_psth_trace")
    if fr is not None and np.isfinite(fr).any():
        plt.plot(w, fr, marker="o", label="FEV_rate_trace")
    plt.xlabel("Window size (ms)")
    plt.ylabel("Fraction")
    plt.title("Trace-level explained variance")
    plt.legend()
    plt.tight_layout()

def plot_ceiling_break_scatter(M, which_window=None, clip=True, min_var=1e-12):
    """
    Scatter FEV vs alpha for a chosen window.
    - If which_window is None, picks the median window.
    """
    w = M["window_ms"]
    per = M["per"]
    if which_window is None:
        idx = len(w)//2
    else:
        idx = int(np.argmin(np.abs(w - which_window)))

    alpha = _as_np(per["alpha"][idx])
    fev_p = _as_np(per["FEV_psth"][idx])
    fev_r = _as_np(per["FEV_rate"][idx])

    # keep finite + sensible
    ok = np.isfinite(alpha) & np.isfinite(fev_p) & np.isfinite(fev_r) & (alpha > 0)
    # also require some total variance
    dT = _as_np(per["diag_total"][idx])
    ok &= np.isfinite(dT) & (dT > min_var)

    alpha = alpha[ok]
    fev_p = fev_p[ok]
    fev_r = fev_r[ok]
    if clip:
        fev_p = np.clip(fev_p, 0, 2.0)  # allow ceiling break to show
        fev_r = np.clip(fev_r, 0, 1.2)

    plt.figure()
    plt.scatter(alpha, fev_p, s=12, alpha=0.6, label="FEV_psth (can exceed 1)")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$\alpha = \mathrm{Var}_{PSTH}/\mathrm{Var}_{rate}$")
    plt.ylabel("Fraction explained")
    plt.title(f"Ceiling break vs alpha (window {w[idx]:.0f} ms)")
    plt.tight_layout()

    plt.figure()
    plt.scatter(alpha, fev_r, s=12, alpha=0.6, label="FEV_rate (true ceiling)")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$\alpha = \mathrm{Var}_{PSTH}/\mathrm{Var}_{rate}$")
    plt.ylabel("Fraction explained")
    plt.title(f"True ceiling vs alpha (window {w[idx]:.0f} ms)")
    plt.tight_layout()

def plot_alignment_vs_window(M, k_list=(1,2,5,10)):
    w = M["window_ms"]
    
    for k in k_list:
        y = M["align"].get(k, None)
        if y is None: 
            continue
        plt.plot(w, y, marker="o", label=f"frac_k (k={k})")
    plt.xlabel("Window size (ms)")
    plt.ylabel("Fraction of explained cov in top-k subspace")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Subspace alignment vs window (basis={M['basis']})")
    plt.legend()
    plt.tight_layout()

def plot_diag_sanity(M, which_window=None, min_var=1e-12):
    """
    Compare residual variance to implied noise variances:
      diag(Cnoise_uncorr)=diag(CTotal-Cpsth)
      diag(Cnoise_corr)  =diag(CTotal-Crate)
    """
    w = M["window_ms"]
    per = M["per"]
    if which_window is None:
        idx = len(w)//2
    else:
        idx = int(np.argmin(np.abs(w - which_window)))

    dT = _as_np(per["diag_total"][idx])
    dRes = _as_np(per["diag_res"][idx])
    dP = _as_np(per["diag_psth"][idx])
    dQ = _as_np(per["diag_rate"][idx])

    noise_unc = dT - dP
    noise_cor = dT - dQ

    ok = np.isfinite(dT) & np.isfinite(dRes) & np.isfinite(noise_unc) & np.isfinite(noise_cor) & (dT > min_var)

    plt.figure()
    plt.scatter(noise_unc[ok], dRes[ok], s=12, alpha=0.6)
    mn = np.nanmin([noise_unc[ok].min(), dRes[ok].min()])
    mx = np.nanmax([noise_unc[ok].max(), dRes[ok].max()])
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("diag(CTotal - Cpsth)  (PSTH-defined noise var)")
    plt.ylabel("diag(CResidual)  (model residual var)")
    plt.title(f"Residual variance vs PSTH-noise variance (window {w[idx]:.0f} ms)")
    plt.tight_layout()

    plt.figure()
    plt.scatter(noise_cor[ok], dRes[ok], s=12, alpha=0.6)
    mn = np.nanmin([noise_cor[ok].min(), dRes[ok].min()])
    mx = np.nanmax([noise_cor[ok].max(), dRes[ok].max()])
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("diag(CTotal - Crate)  (rate-defined noise var)")
    plt.ylabel("diag(CResidual)  (model residual var)")
    plt.title(f"Residual variance vs rate-noise variance (window {w[idx]:.0f} ms)")
    plt.tight_layout()

# ----------------------------
# Example usage in your sweep loop
# ----------------------------
# Suppose for each window you have:
#   SpikeCounts (N_samples, C)
#   model_pred_counts (N_samples, C)  # aligned to SpikeCounts
# then:
#
# CResidual = cov_from_residuals(SpikeCounts - model_pred_counts)
# collect all these into CResidual_list in the same order as results/mats_save

# After sweep:
# M = collect_window_metrics(results, mats_save, CResidual_list, k_list=(1,2,5,10), basis="fem")
# plot_trace_metrics(M)
# plot_ceiling_break_scatter(M, which_window=M["window_ms"][len(M["window_ms"])//2])
# plot_alignment_vs_window(M, k_list=(1,2,5,10))
# plot_diag_sanity(M)
# plt.show()

#%%


#%%

j = 0
Cresiduals_list = [outputs[j]['last_mats_residuals'][i]['Total'] for i in range(len(outputs[0]['last_mats_residuals']))]
M = collect_window_metrics(outputs[j]['results'], outputs[j]['last_mats'], Cresiduals_list, k_list=(1,2,5,10), basis="fem")

#%%
plot_trace_metrics(M, clip=False)

#%%
plot_ceiling_break_scatter(M, which_window=M["window_ms"][1], clip=False)

#%%
plot_alignment_vs_window(M, k_list=(1,2,5,10))

#%%
plot_diag_sanity(M)
plt.show()
#%%

#%%


ve_raw_windows = []
windows = outputs[0]['windows']
for i in range(len(outputs[0]['results'])):

    ve_raw = []
    for j in range(len(outputs)):
        Cres = outputs[j]['last_mats_residuals'][i]['Total']
        Ctotal = outputs[j]['last_mats'][i]['Total']
        Cpsth = outputs[j]['last_mats'][i]['PSTH']
        Crate = outputs[j]['last_mats'][i]['Intercept']

        fe = fraction_explained(Ctotal, Cres, Cpsth=Cpsth, Crate=Crate, clip=False)
        ve_raw.append(fe['VE_raw'])
    ve_raw_windows.append(ve_raw)
    # align = subspace_alignment(Ctotal, Cres, Cpsth, Crate, basis="fem", k=5)

# plt.plot(fe['VE_raw'], '.')
mu = np.zeros(len(ve_raw_windows))
for i in range(len(ve_raw_windows)):
    ve = np.concatenate(ve_raw_windows[i])
    mu[i] = np.nanmean(ve)
    plt.hist(ve, bins=np.linspace(0, 1, 50), alpha=0.5)
    
plt.show()

plt.plot(windows, mu, 'o-')
plt.legend()




#%%

var_explained_model = []
var_explained_psth = []
corr_explained_model = []
corr_explained_psth = []
n = len(outputs[0]['results'])
windows = outputs[0]['windows']
fig, axs = plt.subplots(1,n, figsize=(10,5), sharex=True, sharey=True)
for i in range(n):

    for j in range(len(outputs)):
        Cres = outputs[j]['last_mats_residuals'][i]['Total']
        Ctotal = outputs[j]['last_mats'][i]['Total']
        Cpsth = outputs[j]['last_mats'][i]['PSTH']
        model_explained = 1 - np.diag(Cres)/np.diag(Ctotal)
        psth_explained = 1 - np.diag(Ctotal-Cpsth)/np.diag(Ctotal)
        var_explained_model.append(model_explained)
        var_explained_psth.append(psth_explained)

        rho_uncorr = get_upper_triangle(Ctotal-Cpsth)
        rho_corr = get_upper_triangle(Cres)
        corr_explained_model.append(rho_corr)
        corr_explained_psth.append(rho_uncorr)


    model_explained = np.concatenate(var_explained_model)
    psth_explained = np.concatenate(var_explained_psth)
    # model_explained = 1 - np.diag(Cres)/np.diag(Ctotal)
    # psth_explained = 1 - np.diag(Ctotal-Cpsth)/np.diag(Ctotal)
    axs[i].plot(psth_explained, model_explained, '.' , alpha=0.1)
    # plot line of unity
    axs[i].plot(axs[i].get_xlim(), axs[i].get_xlim(), 'k--', alpha=0.5)
    axs[i].set_title(f"Window {windows[i]}ms")
    axs[i].set_xlabel('PSTH Explained')
    axs[i].set_ylabel('Model Explained')
    axs[i].set_title(f"Window {windows[i]}ms")
# plt.colorbar()


#%%



n = len(outputs[0]['results'])
fig, axs = plt.subplots(1,n, figsize=(3*n, 3), sharex=False, sharey=False)
ffs = []
for i in range(n):
    
    ff_uncorrs = []
    ff_corrs = []
    erates = []
    rhos_uncorr = []
    rhos_corr = []
    alphas = []
    for j in range(len(outputs)):
        window_ms = outputs[j]['results'][i]['window_ms']
        ff_uncorr = outputs[j]['results'][i]['ff_uncorr']
        ff_corr = outputs[j]['results'][i]['ff_corr']
        Erates = outputs[j]['results'][i]['Erates']
        alpha = outputs[j]['results'][i]['alpha']
        
        CnoiseU = outputs[j]['last_mats'][i]['NoiseCorrU']
        CnoiseC = outputs[j]['last_mats'][i]['NoiseCorrC']
        rho_uncorr = get_upper_triangle(CnoiseU)
        rho_corr = get_upper_triangle(CnoiseC)

        valid = Erates > 0.1
        ff_uncorrs.append(ff_uncorr[valid])
        ff_corrs.append(ff_corr[valid])
        erates.append(Erates[valid])
        rhos_uncorr.append(rho_uncorr)
        rhos_corr.append(rho_corr)
        alphas.append(alpha[valid])

        axs[i].plot(Erates, ff_uncorr*Erates, 'r.', alpha=0.1)
        axs[i].plot(Erates, ff_corr*Erates, 'b.', alpha=0.1)
        xd = [0, np.percentile(Erates[valid], 99)]
        axs[i].plot(xd, xd, 'k--', alpha=0.5)
        axs[i].set_xlim(xd)
        axs[i].set_ylim(xd[0], xd[1]*2)
        
    
    ffs.append({'window_ms': window_ms,
                'uncorr': np.concatenate(ff_uncorrs),
                'corr': np.concatenate(ff_corrs),
                'erate': np.concatenate(erates),
                'alpha': np.concatenate(alphas),
                'rho_uncorr': np.concatenate(rhos_uncorr),
                'rho_corr': np.concatenate(rhos_corr),
                })
    
    # axs[i].axhline(1.0, color='k', linestyle='--', alpha=0.5)
    axs[i].set_xlabel('Mean')
axs[0].set_ylabel('Variance')
    

plt.show()
#%%


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
        ffs[i]['erate'],
        ffs[i]['uncorr'] * ffs[i]['erate'],
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
        ffs[i]['erate'],
        ffs[i]['corr'] * ffs[i]['erate'],
        f"Window {ffs[i]['window_ms']}ms",
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

    x = ffs[i]['uncorr']
    y = ffs[i]['corr']
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
        ffs[i]['rho_uncorr'], ffs[i]['rho_corr'],
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
    axs[i].set_title(f"Window {ffs[i]['window_ms']}ms")
    axs[i].set_xlabel("Correlation (Uncorrected)")
    axs[i].set_ylabel("Correlation (Corrected)")

    # means + bootstrap CIs
    mu_u[i] = np.nanmean(ffs[i]['rho_uncorr'])
    sd = np.nanstd(ffs[i]['rho_uncorr'])
    lo_u[i] = mu_u[i] - sd
    hi_u[i] = mu_u[i] + sd  
    mu_c[i] = np.nanmean(ffs[i]['rho_corr'])
    sd = np.nanstd(ffs[i]['rho_corr'])
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

#%%
i = 3
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.hist2d(ffs[i]['rho_uncorr'], ffs[i]['rho_corr'], bins=np.linspace(-.2,.2,100)) #, '.', alpha=0.1)
ax.plot(plt.ylim(), plt.ylim(), 'k--', alpha=0.5)
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.set_title(f"Window {ffs[i]['window_ms']}ms")
ax.set_xlabel("Correlation (Uncorrected)")
ax.set_ylabel("Correlation (Corrected)")


#%% Plot histogram of alpha
fig, ax = plt.subplots(1, n, figsize=(3*n, 3))

for i in range(n):
    alpha = ffs[i]['alpha']
    ax[i].hist(1 - alpha, bins=np.linspace(0, 1, 50))
    ax[i].axvline(np.nanmean(1-alpha), color='r', linestyle='--', alpha=0.5)
    ax[i].set_xlabel("1 - alpha")
    ax[i].set_ylabel("Count")
    ax[i].set_title(f"Window {ffs[i]['window_ms']}ms")

fig.savefig('../figures/mcfarland/alpha.pdf', bbox_inches='tight', dpi=300) 

# plot fano factor vs 1-alpha

for field in ['corr', 'uncorr']:
    fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
    for i in range(n):
        alpha = ffs[i]['alpha']
        ff = ffs[i][field]
        ax[i].plot(1 - alpha, ff, 'o', alpha=0.1)
        ax[i].set_xlim(0, 1)
        ax[i].set_xlabel("1 - alpha")
        ax[i].set_ylabel("Fano Factor")
        ax[i].set_title(f"Window {ffs[i]['window_ms']}ms")
    fig.savefig(f'../figures/mcfarland/ff_vs_alpha_{field}.pdf', bbox_inches='tight', dpi=300) 

#%%
j = 0
i = 1
Ctotal = outputs[j]['last_mats'][i]['Total']
Cfem = outputs[j]['last_mats'][i]['FEM']
Cint = outputs[j]['last_mats'][i]['Intercept']
Cpsth = outputs[j]['last_mats'][i]['PSTH']

CnoiseU = Ctotal - Cpsth
CnoiseC = CnoiseU - Cfem
plt.subplot(1,2,1)
plt.imshow(CnoiseU)
plt.subplot(1,2,2)
plt.imshow(CnoiseC)


#%%
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.diag(Ctotal), np.diag(Cint), '.', label='Cintercept')
plt.plot(np.diag(Ctotal), np.diag(Cpsth), '.', label='Cpsth')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Total Variance (diagonal of Ctotal)')
plt.ylabel('Rate Variance (diagonal of Cint, Cpsth)')
plt.legend()
plt.subplot(1,2,2)
plt.plot(get_upper_triangle(Ctotal), get_upper_triangle(Cint), '.')
plt.plot(get_upper_triangle(Ctotal), get_upper_triangle(Cpsth), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Total Covariance (upper triangle of Ctotal)')
plt.ylabel('Rate Covariance (upper triangle of Cint)')

#%%

#%%


v = np.max(Cfem.flatten())
plt.subplot(1,3,1)
plt.imshow(Ctotal, vmin=-v, vmax=v)
plt.title('Total')
plt.subplot(1,3,2)
plt.imshow(Cfem, vmin=-v, vmax=v)
plt.title('Eye')
plt.subplot(1,3,3)
plt.imshow(Cpsth, vmin=-v, vmax=v)
plt.title('PSTH')

plt.figure()
plt.subplot(1,2,1)
v = .2
plt.imshow(CnoiseU, vmin=-v, vmax=v)
plt.colorbar()
plt.title('Noise (Uncorrected))')
plt.subplot(1,2,2)
plt.imshow(CnoiseC, vmin=-v, vmax=v)
plt.colorbar()
plt.title('Noise (Corrected) ')


plt.figure()
plt.plot(FF_uncorr, FF_corr, '.')
plt.axhline(1, color='k', linestyle='--')
plt.axvline(1, color='k', linestyle='--')
plt.plot(np.mean(FF_uncorr), np.mean(FF_corr), 'ro')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')
plt.title(f"FF Window Size ({windows[window_idx]}ms)")

#
def get_upper_triangle(C):
    rows, cols = np.triu_indices_from(C, k=1)
    v = C[rows, cols]
    return v

rho_uncorr = get_upper_triangle(CnoiseU)
rho_corr = get_upper_triangle(CnoiseC)

plt.figure()
plt.plot(rho_uncorr, rho_corr, '.', alpha=0.1)
# plot mean
plt.plot(rho_uncorr.mean(), rho_corr.mean(), 'ro')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Correlation (Uncorrected)')
plt.ylabel('Correlation (Corrected)')
plt.title('Correlation vs Window Size')


# 3. Plot Fano Factor Scaling
window_ms = [results[i]['window_ms'] for i in range(len(results))]
ff_uncorr = np.zeros_like(window_ms, dtype=np.float64)
ff_uncorr_std = np.zeros_like(window_ms, dtype=np.float64)
ff_uncorr_se = np.zeros_like(window_ms, dtype=np.float64)
ff_corr = np.zeros_like(window_ms, dtype=np.float64)
ff_corr_std = np.zeros_like(window_ms, dtype=np.float64)
ff_corr_se = np.zeros_like(window_ms, dtype=np.float64)

for iwindow in range(len(window_ms)):
    ff_uncorr[iwindow] = np.nanmean(results[iwindow]['ff_uncorr'])
    ff_corr[iwindow] = np.nanmean(results[iwindow]['ff_corr'])
    ff_uncorr_std[iwindow] = np.nanstd(results[iwindow]['ff_uncorr'])
    ff_corr_std[iwindow] = np.nanstd(results[iwindow]['ff_corr'])
    ff_uncorr_se[iwindow] = ff_uncorr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_uncorr']))
    ff_corr_se[iwindow] = ff_corr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_corr']))

plt.figure(figsize=(8, 6))
plt.plot(window_ms, ff_uncorr, 'o-', label='Standard (Uncorrected)')
plt.plot(window_ms, ff_corr, 'o-', label='FEM-Corrected')
# plot error bars
plt.fill_between(window_ms, ff_uncorr - ff_uncorr_se, ff_uncorr + ff_uncorr_se, alpha=0.2)
plt.fill_between(window_ms, ff_corr - ff_corr_se, ff_corr + ff_corr_se, alpha=0.2)

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Count Window (ms)')
plt.ylabel('Mean Fano Factor')
plt.title('Integration of Noise: FEM Correction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
window_idx = 0
Sigma_FEM = last_mats[window_idx]['FEM']
u, s, vh = np.linalg.svd(Sigma_FEM)
plt.figure()
plt.plot(s, 'o-', label='FEM')

Sigma_PSTH = last_mats[window_idx]['PSTH']
u, s, vh = np.linalg.svd(Sigma_PSTH)
plt.plot(s, 'o-', label='PSTH')

# same for total covariance
Sigma_Total = last_mats[window_idx]['Total']
u, s, vh = np.linalg.svd(Sigma_Total)
plt.plot(s, 'o-', label='Total')


# # now noise cov
# Sigma_Noise = last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH']
# u, s, vh = np.linalg.svd(Sigma_Noise)
# plt.plot(s, 'o-', label='Noise Uncorrected')

# Sigma_Noise = last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'] - last_mats[window_idx]['PSTH']
# u, s, vh = np.linalg.svd(Sigma_Noise)
# plt.plot(s, 'o-', label='Noise Corrected')
plt.title(f"Singular Values ({windows[window_idx]}ms)")
plt.legend()
# plt.yscale('log')
plt.show()
# %%
for i in range(len(results)):
    plt.plot(results[i]['ff_uncorr'], results[i]['ff_corr'], 'o')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.axvline(1.0, color='k', linestyle='--', alpha=0.5)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Fano Factor (Uncorrected)')
plt.ylabel('Fano Factor (Corrected)')

#%%



fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
for i in range(len(results)):
    
    CvarU = np.diag(last_mats[i]['Total']-last_mats[i]['PSTH'])
    CvarC = np.diag(last_mats[i]['Total']-last_mats[i]['Intercept'])
    mu = results[i]['Erates']
    plot_slope_estimation(axs[0], mu, CvarU, "Uncorrected", "tab:blue")
    plot_slope_estimation(axs[1], mu, CvarC, "Corrected", "tab:red")

#%%
fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
for i in range(len(results)):
    CvarU = np.diag(last_mats[i]['Total']-last_mats[i]['PSTH'])
    CvarC = np.diag(last_mats[i]['Total']-last_mats[i]['Intercept'])
    mu = results[i]['Erates']
    axs[0].plot(mu, CvarU, '.')
    axs[1].plot(mu, CvarC, '.')
    
    # plt.plot(results[i]['Erates'],results[i]['ff_corr'], '.' )
axs[0].set_title('Uncorrected')
axs[1].set_title('Corrected')
axs[0].set_xlabel('Mean Rate (spikes/sec)')
axs[1].set_xlabel('Mean Rate (spikes/sec)')
axs[0].set_ylabel('Variance (spikes^2/sec)')   
# plot line of unity
for ax in axs:
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', alpha=0.5)


#%%
for thresh in [0.05, 0.1, 0.2, 0.5]:
    for i in range(len(results)):
        ff = results[i]['ff_corr'][results[i]['Erates'] > thresh]
        mu = np.mean(ff)
        std = np.std(ff)
        plt.errorbar(i+thresh, mu, yerr=std/np.sqrt(len(ff)), fmt='o')

plt.axvline(.2, color='k', linestyle='--', alpha=0.5)
#%%
results[0]
# %%
# show the total covariance matrix subtracting the diagonal
window_idx = 1
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.imshow(last_mats[window_idx]['Total'] - np.diag(np.diag(last_mats[window_idx]['Total'])))
plt.title(f"Total Covariance ({windows[window_idx]}ms)")

# show FEM
plt.subplot(1,3,2)
plt.imshow(last_mats[window_idx]['FEM'] - np.diag(np.diag(last_mats[window_idx]['FEM'])))
plt.title(f"FEM Covariance ({windows[window_idx]}ms)")

# show Noise_Corr
plt.subplot(1,3,3)
plt.imshow(last_mats[window_idx]['PSTH'] - np.diag(np.diag(last_mats[window_idx]['PSTH'])))
plt.title(f"PSTH Covariance ({windows[window_idx]}ms)")



# %%
plt.subplot(1,2,1)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['PSTH'])
plt.subplot(1,2,2)
plt.imshow(last_mats[window_idx]['Total'] - last_mats[window_idx]['FEM'])

# %%
