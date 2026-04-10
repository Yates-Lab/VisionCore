# %% Imports and configuration
"""
Figure 2: Covariance decomposition reveals a dominant contribution of
fixational eye movements to shared population variability.

Flat, cell-based script. Each cell computes stats and plots its panel(s).
Run interactively with IPython (#%% cells) or as a script with uv run.

Data is loaded directly from the experiment configs and data packages
(no model weights needed). Results are cached after the first run.
"""
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR
from VisionCore.covariance import (
    cov_to_corr,
    project_to_psd,
    get_upper_triangle,
    align_fixrsvp_trials,
    run_covariance_decomposition,
)
from VisionCore.stats import (
    geomean,
    iqr_25_75,
    bootstrap_mean_ci,
    bootstrap_paired_diff_ci,
    fisher_z,
    fisher_z_mean,
    emp_p_one_sided,
    wilcoxon_signed_rank,
    fdr_correct,
    paired_valid,
)
from VisionCore.subspace import (
    participation_ratio,
    symmetric_subspace_overlap,
    directional_variance_capture,
)
from DataYatesV1 import get_free_device


def load_contam_rate(session_name, subject, n_neurons_total):
    """Load per-neuron min contamination rate from QC data.

    Returns array of shape (n_neurons_total,) with min contamination
    proportion per neuron, or None if unavailable.
    """
    if subject in ("Allen", "Logan"):
        from DataYatesV1.utils.io import YatesV1Session
        try:
            sess = YatesV1Session(session_name)
            refractory = np.load(
                sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz'
            )
            min_contam_props = refractory['min_contam_props']
            # min across refractory periods for each neuron
            contam_rate = np.array([
                np.min(min_contam_props[i]) for i in range(len(min_contam_props))
            ])
            return contam_rate
        except Exception as e:
            print(f"  Warning: Could not load QC data: {e}")
            return None
    else:
        raise NotImplementedError(
            f"QC loading not implemented for subject {subject}"
        )


# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
DT = 1 / 240                # seconds per bin (native 240 Hz sampling)
WINDOW_BINS = [2, 4, 8, 16] # counting windows in bins (powers of two)
N_SHUFFLES = 100             # shuffle null iterations
MIN_TOTAL_SPIKES = 500       # neuron inclusion threshold (in align step)
MIN_VAR = 0                  # minimum variance for correlation computation
EPS_RHO = 1e-3               # floor for correlation denominators
SUBJECTS = ["Allen", "Logan", "Luke"]
DEVICE = get_free_device()

# Data config (uses the same configs as model training, no weights needed)
DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"

# Subspace analysis
SUBSPACE_WINDOW_IDX = 1      # second window (4 bins = 16.67 ms)
SUBSPACE_K = 5               # subspace dimensionality for overlap

# Output directories
FIG_DIR = FIGURES_DIR / "fig2"
STAT_DIR = STATS_DIR / "fig2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Detect interactive IPython session
try:
    get_ipython()  # type: ignore[name-defined]
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False


def show_or_close(fig):
    """Show figure in interactive sessions, close otherwise."""
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


def subject_iter(labels):
    """Yield (suffix, mask) for each subject then pooled.

    Parameters
    ----------
    labels : array-like of str
        Subject label per element (neuron, pair, or session).

    Yields
    ------
    suffix : str
        "" for pooled, "_Allen" etc. for per-subject.
    mask : ndarray of bool
        Boolean mask into the labels array.
    """
    labels = np.asarray(labels)
    for subj in SUBJECTS:
        mask = labels == subj
        if mask.any():
            yield f"_{subj}", mask
    yield "", np.ones(len(labels), dtype=bool)


# %% Compute covariance decomposition
# Self-contained: loads raw data via prepare_data(), runs LOTC decomposition.
# Add VisionCore root to path so models.* imports work
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from models.config_loader import load_dataset_configs
from models.data import prepare_data

session_name = "Luke_2025-08-04"
#cids_overwrite = [800, 801, 802, 805, 806, 807, 810, 815, 816, 818, 820, 823, 825, 828, 835, 841, 844, 846, 850, 852, 853, 854, 859, 863, 867, 875, 880, 883, 884, 887, 888, 894, 902, 904, 908, 911, 917, 923, 924, 925, 931, 933, 936, 938, 939, 940, 942, 943, 950]
#session_name = "Luke_2026-03-02"
cids_overwrite = None  # set to list of neuron IDs to overwrite with random data (for testing)


dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
cfg = [cfg for cfg in dataset_configs if cfg["session"] == session_name][0]
if cids_overwrite is not None:
    cfg["cids"] = cids_overwrite
print(f"Example config for session {cfg['session']}:\n{cfg}")

session_name = cfg["session"]
subject = session_name.split("_")[0]

# Set only fixrsvp to load (we only need the raw fixrsvp dataset for this analysis)
cfg["types"] = ["fixrsvp"]
# Ensure fixrsvp is in the types list
#if "fixrsvp" not in cfg["types"]:
    #cfg["types"] = cfg["types"] + ["fixrsvp"]

#train_data, val_data, cfg = prepare_data(cfg, strict=False)

# Get the raw fixRSVP DictDataset
#dset_idx = train_data.get_dataset_index("fixrsvp")
#fixrsvp_dset = train_data.dsets[dset_idx]

#%%
# OK, changing the cids did not seem to modify which units were selected in prepare_data. Seems like a bug somewhere.
# This also makes me realizie that there's another bug where we are downsampling the fixrsvp session, so the actual DT is not 1/240 but instead 1/120. Everywhere it is assumed to be 1/240. 
# Also, since we're not using the model right now we can skip the preprocess data set and just load the raw fixrsvp dataset directly from disk using DictDataset.load. This will avoid any issues with prepare_data modifying the data in unexpected ways.
#fixrsvp_dset.metadata['session']

cids = np.array(cfg['cids'])
from DataRowleyV1V2.data.registry import get_dataset
fixrsvp_dset = get_dataset(session_name, "fixrsvp", eye=cfg['eye'])
print(fixrsvp_dset)
fixrsvp_dset['robs'] = fixrsvp_dset['robs'][:,cids]
print(fixrsvp_dset)


#%%


# Trial-align
robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
    fixrsvp_dset,
    valid_time_bins=480,
    min_fix_dur=20,
    min_total_spikes=50,
)

print(fixrsvp_dset['robs'].shape)
print(robs.shape)

print(f"  Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
      f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")
#%%

for iC in range(robs.shape[2]):
    r = robs[:,:,iC]
    # sort by number of non-nans in each row
    n_non_nan = np.sum(np.isfinite(r), axis=1)
    sort_idx = np.argsort(n_non_nan)[::-1]
    r = r[sort_idx,:]

    plt.figure()
    plt.imshow(r, aspect='auto', cmap='viridis')
    plt.colorbar(label='Spike count')
    plt.show()
#%%

# Run LOTC decomposition
results, mats = run_covariance_decomposition(
    robs, eyepos, valid_mask,
    window_sizes_bins=WINDOW_BINS,
    dt=DT,
    n_shuffles=N_SHUFFLES,
    intercept_mode="lowest_bin",
    seed=42,
    device=str(DEVICE),
)

# Trial-averaged PSTH (neurons passing neuron_mask only)
psth = robs.mean(axis=0)  # (n_time, n_neurons_used)

# QC: contamination rate
try:
    contam_rate = load_contam_rate(
        session_name, subject, meta['n_neurons_total']
    )
except NotImplementedError:
    contam_rate = None
    print(f"  QC: contamination not available for {subject}")

session_results = {
    "session": session_name,
    "subject": subject,
    "results": results,
    "mats": mats,
    "neuron_mask": neuron_mask,
    "meta": meta,
    "psth": psth,
    "qc": {"contam_rate": contam_rate},
}

# Derive window labels and session metadata
WINDOWS_MS = [r["window_ms"] for r in session_results["results"]]
WINDOWS_BINS = [r["window_bins"] for r in session_results["results"]]
print(f"Windows (bins): {WINDOWS_BINS} -> (ms): {[f'{w:.1f}' for w in WINDOWS_MS]}")

# %% Extract per-window metrics
# Single-session version: extract directly from session_results dict.

sr = session_results
n_windows = len(WINDOWS_MS)
metrics = []

for w_idx in range(n_windows):
    res = sr["results"][w_idx]
    mats_w = sr["mats"][w_idx]

    Ctotal = mats_w["Total"]
    Cpsth = mats_w["PSTH"]
    Crate = mats_w["Intercept"]
    Cfem = mats_w["FEM"]

    CnoiseU = Ctotal - Cpsth
    CnoiseC = Ctotal - Crate
    CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
    CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)

    erate = res["Erates"]
    total_spikes = erate * res["n_samples"]

    # Neuron inclusion mask
    valid = (
        np.isfinite(erate)
        & (total_spikes >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    n_valid = valid.sum()
    print(f"Window {WINDOWS_MS[w_idx]:.1f} ms: {n_valid}/{len(erate)} neurons pass inclusion")

    if n_valid < 3:
        metrics.append(None)
        continue

    # Alpha
    diag_psth = np.diag(Cpsth)[valid]
    diag_rate = np.diag(Crate)[valid]
    alpha = np.clip(diag_psth / diag_rate, 0, 1)

    # Fano factors
    ff_u = np.diag(CnoiseU)[valid] / erate[valid]
    ff_c = np.diag(CnoiseC)[valid] / erate[valid]

    # Noise correlations
    NoiseCorrU = cov_to_corr(project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR)
    NoiseCorrC = cov_to_corr(project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR)
    rho_u = get_upper_triangle(NoiseCorrU)
    rho_c = get_upper_triangle(NoiseCorrC)
    pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
    rho_u = rho_u[pair_ok]
    rho_c = rho_c[pair_ok]

    # Shuffle nulls
    shuff_alphas = []
    shuff_rho_c_meanz = []
    shuff_rho_delta_meanz = []
    if "Shuffled_Intercepts" in mats_w and len(mats_w["Shuffled_Intercepts"]) > 0:
        for Crate_shuf in mats_w["Shuffled_Intercepts"]:
            diag_rate_shuf = np.diag(Crate_shuf)[valid]
            alpha_shuf = np.clip(diag_psth / diag_rate_shuf, 0, 1)
            shuff_alphas.append(1 - alpha_shuf)

            CnoiseC_shuf = Ctotal - Crate_shuf
            CnoiseC_shuf = 0.5 * (CnoiseC_shuf + CnoiseC_shuf.T)
            NC_shuf = cov_to_corr(
                project_to_psd(CnoiseC_shuf[np.ix_(valid, valid)]), min_var=MIN_VAR
            )
            rho_c_shuf = get_upper_triangle(NC_shuf)
            ok = np.isfinite(rho_c_shuf) & pair_ok
            if ok.sum() > 0:
                shuff_rho_c_meanz.append(fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO))
                shuff_rho_delta_meanz.append(
                    fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                    - fisher_z_mean(rho_u[ok[:len(rho_u)]], eps=EPS_RHO)
                )

    metrics.append({
        "window_ms": WINDOWS_MS[w_idx],
        "window_bins": WINDOWS_BINS[w_idx],
        "alpha": alpha,
        "uncorr": ff_u,
        "corr": ff_c,
        "erate": erate[valid],
        "rho_uncorr": rho_u,
        "rho_corr": rho_c,
        "Ctotal": Ctotal[np.ix_(valid, valid)],
        "Cpsth": Cpsth[np.ix_(valid, valid)],
        "Crate": Crate[np.ix_(valid, valid)],
        "CnoiseU": CnoiseU[np.ix_(valid, valid)],
        "CnoiseC": CnoiseC[np.ix_(valid, valid)],
        "Cfem": Cfem[np.ix_(valid, valid)],
        "shuff_alphas": shuff_alphas,
        "shuff_rho_delta_meanz": np.array(shuff_rho_delta_meanz),
        "shuff_rho_c_meanz": np.array(shuff_rho_c_meanz),
    })

    print(f"  {n_valid} neurons, {len(rho_u)} pairs, {len(shuff_alphas)} shuffle iterations")

# %% Panel C: FEM modulation fraction (1-alpha)
m_by_window = []
alpha_stats = {}

for w_idx, m_dict in enumerate(metrics):
    if m_dict is None:
        continue
    alpha = m_dict["alpha"]
    m = 1 - alpha
    m_by_window.append(m)

    mean_m, (ci_lo, ci_hi) = bootstrap_mean_ci(m, nboot=5000, seed=0)
    med_m = float(np.nanmedian(m))
    q25, q75 = iqr_25_75(m)

    shuff_m = m_dict["shuff_alphas"]
    if len(shuff_m) > 0:
        null_means = np.array([np.nanmean(s) for s in shuff_m])
        null_mean_ci = (float(np.percentile(null_means, 2.5)),
                        float(np.percentile(null_means, 97.5)))
        p_emp = emp_p_one_sided(null_means, mean_m, direction="less")
    else:
        null_mean_ci = (np.nan, np.nan)
        p_emp = np.nan

    alpha_stats[WINDOWS_MS[w_idx]] = {
        "n": len(m), "mean": mean_m, "ci": (ci_lo, ci_hi),
        "median": med_m, "iqr": (q25, q75),
        "null_ci": null_mean_ci, "p_emp": p_emp,
    }

    print(f"\nWindow {WINDOWS_MS[w_idx]:.1f} ms (N={len(m)}):")
    print(f"  1-alpha: mean={mean_m:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  median={med_m:.3f} IQR=[{q25:.3f}, {q75:.3f}]")
    print(f"  Shuffle null mean 95% CI: [{null_mean_ci[0]:.3f}, {null_mean_ci[1]:.3f}]")
    print(f"  Empirical p={p_emp:.4f}")

# Plot
m0 = m_by_window[0]
s0 = alpha_stats[WINDOWS_MS[0]]
fig_c, ax_c = plt.subplots(figsize=(4, 3))
ax_c.hist(m0, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax_c.axvline(np.nanmedian(m0), color="red", linewidth=2,
             label=f"median={np.nanmedian(m0):.3f}")
ax_c.axvspan(s0["null_ci"][0], s0["null_ci"][1], alpha=0.2, color="gray",
             label="shuffle 95% CI")
ax_c.set_xlabel("1 - α (FEM modulation fraction)")
ax_c.set_ylabel("Neuron count")
ax_c.set_title(f"Panel C: FEM modulation ({WINDOWS_MS[0]:.1f} ms) — {session_name}")
ax_c.legend(frameon=False, fontsize=8)
fig_c.tight_layout()
show_or_close(fig_c)

# %% Panel D: Mean-variance scatter (first window)
fano_stats = {}

for w_idx, m_dict in enumerate(metrics):
    if m_dict is None:
        continue
    ff_u, ff_c, erate = m_dict["uncorr"], m_dict["corr"], m_dict["erate"]
    ff_u_v, ff_c_v, mask = paired_valid(ff_u, ff_c, positive=True)
    erate_v = erate[mask]
    n_valid = len(ff_u_v)

    g_unc = geomean(ff_u_v)
    g_cor = geomean(ff_c_v)
    ratio = g_cor / g_unc
    pct_red = (1 - ratio) * 100
    _, p_wil = wilcoxon_signed_rank(ff_c_v, ff_u_v, alternative="less")

    var_u = ff_u_v * erate_v
    var_c = ff_c_v * erate_v
    slope_unc = float(np.sum(erate_v * var_u) / np.sum(erate_v ** 2))
    slope_cor = float(np.sum(erate_v * var_c) / np.sum(erate_v ** 2))

    fano_stats[WINDOWS_MS[w_idx]] = {
        "n": n_valid, "g_unc": g_unc, "g_cor": g_cor,
        "ratio": ratio, "pct_red": pct_red, "p_wil": p_wil,
        "slope_unc": slope_unc, "slope_cor": slope_cor,
        "erate": erate_v, "var_u": var_u, "var_c": var_c,
    }

    print(f"\nWindow {WINDOWS_MS[w_idx]:.1f} ms (N={n_valid}):")
    print(f"  FF uncorr: gmean={g_unc:.3f}, FF corr: gmean={g_cor:.3f}")
    print(f"  Ratio={ratio:.3f} ({pct_red:.1f}% reduction), Wilcoxon p={p_wil:.3g}")
    print(f"  Population FF: uncorr={slope_unc:.3f}, corr={slope_cor:.3f}")

# Plot Panel D
s0 = fano_stats[WINDOWS_MS[0]]
fig_d, ax_d = plt.subplots(figsize=(4, 3.5))
ax_d.scatter(s0["erate"], s0["var_u"], s=8, alpha=0.3, c="tab:blue",
             label=f"Uncorr FF={s0['slope_unc']:.3f}")
ax_d.scatter(s0["erate"], s0["var_c"], s=8, alpha=0.3, c="tab:red",
             label=f"Corr FF={s0['slope_cor']:.3f}")
x_line = np.linspace(0, s0["erate"].max(), 100)
ax_d.plot(x_line, s0["slope_unc"] * x_line, "b--", linewidth=1.5)
ax_d.plot(x_line, s0["slope_cor"] * x_line, "r--", linewidth=1.5)
ax_d.set_xlabel("Mean rate")
ax_d.set_ylabel("Variance")
ax_d.set_title(f"Panel D: Mean-variance ({WINDOWS_MS[0]:.1f} ms) — {session_name}")
ax_d.legend(frameon=False, fontsize=8)
fig_d.tight_layout()
show_or_close(fig_d)

# %% Panel E: Population FF vs window
slopes_unc_all = [fano_stats[w]["slope_unc"] for w in WINDOWS_MS if w in fano_stats]
slopes_cor_all = [fano_stats[w]["slope_cor"] for w in WINDOWS_MS if w in fano_stats]
windows_valid = [w for w in WINDOWS_MS if w in fano_stats]

fig_e, ax_e = plt.subplots(figsize=(4, 3))
ax_e.plot(windows_valid, slopes_unc_all, "o-", color="tab:blue", label="Uncorrected")
ax_e.plot(windows_valid, slopes_cor_all, "o-", color="tab:red", label="FEM-corrected")
ax_e.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Poisson")
ax_e.set_xlabel("Counting window (ms)")
ax_e.set_ylabel("Population Fano factor")
ax_e.set_title(f"Panel E: Pop. FF vs window — {session_name}")
ax_e.legend(frameon=False, fontsize=8)
ax_e.set_xscale("log")
ax_e.set_xticks(windows_valid)
ax_e.set_xticklabels([f"{w:.1f}" for w in windows_valid])
fig_e.tight_layout()
show_or_close(fig_e)

# %% Panel F: Noise correlation scatter (first window)
nc_stats = {}

for w_idx, m_dict in enumerate(metrics):
    if m_dict is None:
        continue
    rho_u = m_dict["rho_uncorr"]
    rho_c = m_dict["rho_corr"]
    n_pairs = len(rho_u)

    z_u_mean = fisher_z_mean(rho_u, eps=EPS_RHO) if n_pairs > 0 else np.nan
    z_c_mean = fisher_z_mean(rho_c, eps=EPS_RHO) if n_pairs > 0 else np.nan
    dz_mean = z_c_mean - z_u_mean

    shuff_dz = m_dict["shuff_rho_delta_meanz"]
    if len(shuff_dz) > 0:
        null_dz_ci = (float(np.percentile(shuff_dz, 2.5)),
                      float(np.percentile(shuff_dz, 97.5)))
        p_emp_dz = emp_p_one_sided(shuff_dz, dz_mean, direction="less")
    else:
        null_dz_ci = (np.nan, np.nan)
        p_emp_dz = np.nan

    nc_stats[WINDOWS_MS[w_idx]] = {
        "n_pairs": n_pairs,
        "z_u_mean": z_u_mean, "z_c_mean": z_c_mean,
        "dz_mean": dz_mean,
        "null_dz_ci": null_dz_ci, "p_emp_dz": p_emp_dz,
        "rho_u": rho_u, "rho_c": rho_c,
    }

    print(f"\nWindow {WINDOWS_MS[w_idx]:.1f} ms ({n_pairs} pairs):")
    print(f"  z_uncorr = {z_u_mean:.4f}, z_corr = {z_c_mean:.4f}")
    print(f"  delta_z  = {dz_mean:.4f}")
    print(f"  Shuffle null delta_z 95% CI: [{null_dz_ci[0]:.4f}, {null_dz_ci[1]:.4f}]")
    print(f"  Empirical p={p_emp_dz:.4f}")

# Plot Panel F
s0 = nc_stats[WINDOWS_MS[0]]
fig_f, ax_f = plt.subplots(figsize=(4, 4))
ax_f.hist2d(s0["rho_u"], s0["rho_c"], bins=60, cmap="Blues",
            norm=mpl.colors.LogNorm(), range=[[-0.3, 0.3], [-0.3, 0.3]])
ax_f.plot([-0.3, 0.3], [-0.3, 0.3], "k--", alpha=0.3, linewidth=0.5)
ax_f.plot(np.mean(s0["rho_u"]), np.mean(s0["rho_c"]), "ro", markersize=6)
ax_f.set_xlabel("ρ uncorrected")
ax_f.set_ylabel("ρ FEM-corrected")
ax_f.set_title(f"Panel F: Noise correlations ({WINDOWS_MS[0]:.1f} ms) — {session_name}")
fig_f.tight_layout()
show_or_close(fig_f)

# %% Panel G: Mean Fisher z vs window
fig_g, ax_g = plt.subplots(figsize=(4, 3))
z_u_vals = [nc_stats[w]["z_u_mean"] for w in WINDOWS_MS if w in nc_stats]
z_c_vals = [nc_stats[w]["z_c_mean"] for w in WINDOWS_MS if w in nc_stats]
windows_valid_nc = [w for w in WINDOWS_MS if w in nc_stats]
ax_g.plot(windows_valid_nc, z_u_vals, "o-", color="tab:blue", label="Uncorrected")
ax_g.plot(windows_valid_nc, z_c_vals, "o-", color="tab:red", label="FEM-corrected")
ax_g.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax_g.set_xlabel("Counting window (ms)")
ax_g.set_ylabel("Mean Fisher z")
ax_g.set_title(f"Panel G: Noise corr vs window — {session_name}")
ax_g.legend(frameon=False, fontsize=8)
ax_g.set_xscale("log")
ax_g.set_xticks(windows_valid_nc)
ax_g.set_xticklabels([f"{w:.1f}" for w in windows_valid_nc])
fig_g.tight_layout()
show_or_close(fig_g)

# %% Panel H: Effect size (delta z) vs window with shuffle null
fig_h, ax_h = plt.subplots(figsize=(4, 3))
dz_vals = [nc_stats[w]["dz_mean"] for w in windows_valid_nc]
ax_h.plot(windows_valid_nc, dz_vals, "o-", color="black", label="Observed Δz")
null_lo = [nc_stats[w]["null_dz_ci"][0] for w in windows_valid_nc]
null_hi = [nc_stats[w]["null_dz_ci"][1] for w in windows_valid_nc]
ax_h.fill_between(windows_valid_nc, null_lo, null_hi, alpha=0.2, color="gray",
                  label="Shuffle 95% CI")
ax_h.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax_h.set_xlabel("Counting window (ms)")
ax_h.set_ylabel("Δz (corr - uncorr)")
ax_h.set_title(f"Panel H: Effect size — {session_name}")
ax_h.legend(frameon=False, fontsize=8)
ax_h.set_xscale("log")
ax_h.set_xticks(windows_valid_nc)
ax_h.set_xticklabels([f"{w:.1f}" for w in windows_valid_nc])
fig_h.tight_layout()
show_or_close(fig_h)

# %% Panel I-K: Subspace alignment (single session)
w_idx = SUBSPACE_WINDOW_IDX

mats_sub = sr["mats"][w_idx]
Cpsth_s = mats_sub["PSTH"]
Crate_s = mats_sub["Intercept"]
Ctotal_s = mats_sub["Total"]
Cfem_s = Crate_s - Cpsth_s

erate_s = sr["results"][w_idx]["Erates"]
total_spikes_s = erate_s * sr["results"][w_idx]["n_samples"]
valid_s = (
    np.isfinite(erate_s)
    & (total_spikes_s >= MIN_TOTAL_SPIKES)
    & (np.diag(Ctotal_s) > MIN_VAR)
    & np.isfinite(np.diag(Crate_s))
    & np.isfinite(np.diag(Cpsth_s))
)

if valid_s.sum() >= SUBSPACE_K + 1:
    Cpsth_v = Cpsth_s[np.ix_(valid_s, valid_s)]
    Cfem_v = Cfem_s[np.ix_(valid_s, valid_s)]
    Ctotal_v = Ctotal_s[np.ix_(valid_s, valid_s)]

    Cpsth_psd = project_to_psd(Cpsth_v)
    Cfem_psd = project_to_psd(Cfem_v)

    w_psth, V_psth = np.linalg.eigh(Cpsth_psd)
    w_fem, V_fem = np.linalg.eigh(Cfem_psd)
    w_psth, V_psth = w_psth[::-1], V_psth[:, ::-1]
    w_fem, V_fem = w_fem[::-1], V_fem[:, ::-1]

    pr_psth = participation_ratio(Cpsth_psd)
    pr_fem = participation_ratio(Cfem_psd)

    k = min(SUBSPACE_K, valid_s.sum() - 1)
    U_psth = V_psth[:, :k]
    U_fem = V_fem[:, :k]
    overlap_k = symmetric_subspace_overlap(U_psth, U_fem)
    overlap_k1 = symmetric_subspace_overlap(V_psth[:, :1], V_fem[:, :1])

    var_p_given_f = directional_variance_capture(Cpsth_psd, U_fem)
    var_f_given_p = directional_variance_capture(Cfem_psd, U_psth)

    tr_total = np.trace(Ctotal_v)
    spec_psth = w_psth / tr_total
    spec_fem = w_fem / tr_total

    print(f"\nSubspace analysis ({WINDOWS_MS[w_idx]:.1f} ms, {session_name}):")
    print(f"  PR(FEM)={pr_fem:.3f}, PR(PSTH)={pr_psth:.3f}")
    print(f"  Overlap k=1: {overlap_k1:.3f}, k={SUBSPACE_K}: {overlap_k:.3f}")
    print(f"  X (PSTH var in FEM subspace): {var_p_given_f:.3f}")
    print(f"  Y (FEM var in PSTH subspace): {var_f_given_p:.3f}")

    # Plot Panel I: eigenspectra
    fig_i, ax_i = plt.subplots(figsize=(4, 3.5))
    max_dims = min(50, len(spec_psth))
    dims = np.arange(1, max_dims + 1)
    ax_i.plot(dims, spec_psth[:max_dims], color="tab:blue", label="PSTH")
    ax_i.plot(dims, spec_fem[:max_dims], color="tab:red", label="FEM")
    ax_i.set_xscale("log")
    ax_i.set_yscale("log")
    ax_i.set_xlabel("Eigenvalue rank")
    ax_i.set_ylabel("Fraction of total variance")
    ax_i.set_title(f"Panel I: Eigenspectra ({WINDOWS_MS[w_idx]:.1f} ms) — {session_name}")
    ax_i.legend(frameon=False, fontsize=8)
    fig_i.tight_layout()
    show_or_close(fig_i)

    # Plot Panel J: participation ratio bars
    fig_j, ax_j = plt.subplots(figsize=(3, 3))
    ax_j.bar([0, 1], [pr_psth, pr_fem], color=["tab:blue", "tab:red"],
             tick_label=["PSTH", "FEM"])
    ax_j.set_ylabel("Participation ratio")
    ax_j.set_title(f"Panel J: Eff. dimensionality ({WINDOWS_MS[w_idx]:.1f} ms) — {session_name}")
    fig_j.tight_layout()
    show_or_close(fig_j)

    # Plot Panel K: variance capture (single point, annotated)
    fig_k, ax_k = plt.subplots(figsize=(4, 4))
    ax_k.scatter([var_p_given_f], [var_f_given_p], c="tab:blue", s=80,
                 edgecolors="black", linewidths=0.5)
    ax_k.annotate(f"X={var_p_given_f:.3f}\nY={var_f_given_p:.3f}",
                  (var_p_given_f, var_f_given_p), textcoords="offset points",
                  xytext=(10, -10), fontsize=9)
    ax_k.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_k.set_xlabel("X: PSTH var captured by FEM subspace")
    ax_k.set_ylabel("Y: FEM var captured by PSTH subspace")
    ax_k.set_title(f"Panel K: Subspace alignment ({WINDOWS_MS[w_idx]:.1f} ms) — {session_name}")
    ax_k.set_xlim(0, 1)
    ax_k.set_ylim(0, 1)
    fig_k.tight_layout()
    show_or_close(fig_k)
else:
    print(f"\nSubspace analysis: too few neurons ({valid_s.sum()}) for k={SUBSPACE_K}")

# %% Covariance heatmaps
cmap = plt.get_cmap("RdBu")
heatmap_window_idx = 3  # 80 ms

if heatmap_window_idx < len(sr["mats"]):
    mats_hm = sr["mats"][heatmap_window_idx]
    Crate_raw = mats_hm["Intercept"]
    hm_valid = np.isfinite(np.diag(Crate_raw)) & np.isfinite(np.diag(mats_hm["PSTH"]))
    ix = np.ix_(hm_valid, hm_valid)

    Ctotal_hm = project_to_psd(mats_hm["Total"][ix])
    Cpsth_hm = project_to_psd(mats_hm["PSTH"][ix])
    Cfem_hm = project_to_psd(Crate_raw[ix] - mats_hm["PSTH"][ix])
    CnoiseC_hm = project_to_psd(mats_hm["Total"][ix] - Crate_raw[ix])

    v = np.nanmax(np.abs(Ctotal_hm)) * 0.5

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{session_name} ({subject})", fontsize=14)
    for ax, mat, title, vscale in zip(
        axs,
        [Ctotal_hm, Cfem_hm, Cpsth_hm, CnoiseC_hm],
        ["Total", "FEM", "PSTH", "Noise (Corrected)"],
        [1.0, 1.0, 0.5, 1.0],
    ):
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-v * vscale, vmax=v * vscale)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    show_or_close(fig)

print(f"\nDone — {session_name} ({subject})")
