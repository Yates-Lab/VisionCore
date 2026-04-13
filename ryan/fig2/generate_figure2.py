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
import matplotlib.ticker as mticker
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
RECOMPUTE = False # set True to rerun decomposition from raw data
DT = 1 / 120                # seconds per bin (native 240 Hz sampling)
WINDOW_BINS = [1, 2, 4, 8] # counting windows in bins (powers of two)
N_SHUFFLES = 100             # shuffle null iterations
MIN_TOTAL_SPIKES = 200       # neuron inclusion threshold (in align step)
MIN_VAR = 0                  # minimum variance for correlation computation
EPS_RHO = 1e-3               # floor for correlation denominators
SUBJECTS = ["Allen", "Logan", "Luke"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green", "Luke": "tab:orange"}
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



# %% Load or compute covariance decomposition
# Self-contained: loads raw data via prepare_data(), runs LOTC decomposition,
# and caches the results. No model weights or legacy pickles required.

cache_path = CACHE_DIR / "fig2_decomposition.pkl"

if cache_path.exists() and not RECOMPUTE:
    print(f"Loading cached decomposition from {cache_path}")
    with open(cache_path, "rb") as f:
        session_results = dill.load(f)
else:
    # Add VisionCore root to path so models.* imports work
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    from models.config_loader import load_dataset_configs
    from models.data import prepare_data

    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    session_results = []

    for cfg in dataset_configs:
        session_name = cfg["session"]
        subject = session_name.split("_")[0]
        if subject not in SUBJECTS:
            continue

        # Ensure fixrsvp is in the types list
        if "fixrsvp" not in cfg["types"]:
            cfg["types"] = cfg["types"] + ["fixrsvp"]

        print(f"\n--- {session_name} ({subject}) ---")
        try:
            train_data, val_data, cfg = prepare_data(cfg, strict=False)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Get the raw fixRSVP DictDataset
        try:
            dset_idx = train_data.get_dataset_index("fixrsvp")
        except (ValueError, KeyError):
            print("  Skipping: no fixrsvp data")
            continue
        fixrsvp_dset = train_data.dsets[dset_idx]

        # Trial-align
        robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
            fixrsvp_dset,
            valid_time_bins=120,
            min_fix_dur=20,
            min_total_spikes=MIN_TOTAL_SPIKES,
        )
        if robs is None or robs.shape[0] < 10:
            print(f"  Skipping: insufficient data ({meta})")
            continue
        print(f"  Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
              f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")

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

        session_results.append({
            "session": session_name,
            "subject": subject,
            "results": results,
            "mats": mats,
            "neuron_mask": neuron_mask,
            "meta": meta,
            "psth": psth,
            "qc": {"contam_rate": contam_rate},
        })

    # Cache results
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {cache_path}")

# Derive window labels and session metadata
WINDOWS_MS = [r["window_ms"] for r in session_results[0]["results"]]
WINDOWS_BINS = [r["window_bins"] for r in session_results[0]["results"]]
session_names = [sr["session"] for sr in session_results]
subjects = [sr["subject"] for sr in session_results]
n_sessions = len(session_results)
print(f"\nLoaded {n_sessions} sessions: {session_names}")
print(f"Subjects: {sorted(set(subjects))}")
print(f"Windows (bins): {WINDOWS_BINS} -> (ms): {[f'{w:.1f}' for w in WINDOWS_MS]}")

# %% Extract per-window metrics
# Inline the logic so every filtering step is visible.
#
# For each counting window, we aggregate across sessions:
# - Per-neuron: alpha, Fano factors (uncorr/corr), firing rates
# - Per-pair: noise correlations (uncorr/corr)
# - Per-session: covariance matrices, shuffle null distributions
# - Neuron inclusion: min_total_spikes, min_var, eps_rho
# - Subject identity tracked per session

n_windows = len(WINDOWS_MS)
metrics = []

for w_idx in range(n_windows):
    # Accumulators for this window
    all_alpha = []
    all_ff_uncorr = []
    all_ff_corr = []
    all_erate = []
    all_rho_uncorr = []
    all_rho_corr = []
    rho_u_meanz_by_ds = []  # per-dataset mean Fisher z
    rho_c_meanz_by_ds = []
    rho_delta_meanz_by_ds = []
    all_Ctotal = []
    all_Cpsth = []
    all_Crate = []
    all_CnoiseU = []
    all_CnoiseC = []
    all_Cfem = []

    # Shuffle null accumulators
    shuff_alphas = []        # (n_shuffles, n_neurons)
    shuff_rho_delta_meanz = []
    shuff_rho_c_meanz = []
    shuff_rho_subject = []   # subject label per shuffle entry

    subject_by_ds = []  # track subject for each session contributing to this window
    subject_per_neuron = []  # one label per neuron (for alpha, ff, erate)
    subject_per_pair = []    # one label per pair (for rho)

    for ds_idx, sr in enumerate(session_results):
        if w_idx >= len(sr["results"]):
            continue  # session has fewer windows (insufficient data for large windows)
        res = sr["results"][w_idx]
        mats = sr["mats"][w_idx]

        Ctotal = mats["Total"]
        Cpsth = mats["PSTH"]
        Crate = mats["Intercept"]
        Cfem = mats["FEM"]

        CnoiseU = Ctotal - Cpsth
        CnoiseC = Ctotal - Crate
        CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
        CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)

        erate = res["Erates"]
        n_cells = len(erate)

        # Total spike count per neuron (rate * n_samples * dt)
        total_spikes = erate * res["n_samples"]

        # Neuron inclusion mask
        valid = (
            np.isfinite(erate)
            & (total_spikes >= MIN_TOTAL_SPIKES)
            & (np.diag(Ctotal) > MIN_VAR)
            & np.isfinite(np.diag(Crate))
            & np.isfinite(np.diag(Cpsth))
        )
        if valid.sum() < 3:
            continue

        subject_by_ds.append(sr["subject"])

        # Alpha: fraction of rate variance from stimulus
        diag_psth = np.diag(Cpsth)[valid]
        diag_rate = np.diag(Crate)[valid]
        alpha = diag_psth / diag_rate
        alpha = np.clip(alpha, 0, 1)
        all_alpha.append(alpha)
        subject_per_neuron.extend([sr["subject"]] * valid.sum())

        # Fano factors
        ff_u = np.diag(CnoiseU)[valid] / erate[valid]
        ff_c = np.diag(CnoiseC)[valid] / erate[valid]
        all_ff_uncorr.append(ff_u)
        all_ff_corr.append(ff_c)
        all_erate.append(erate[valid])

        # PSD project before correlation (removes negative eigenvalues from estimation noise)
        NoiseCorrU = cov_to_corr(project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR)
        NoiseCorrC = cov_to_corr(project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR)
        rho_u = get_upper_triangle(NoiseCorrU)
        rho_c = get_upper_triangle(NoiseCorrC)

        # Filter to pairs where both are finite
        pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
        rho_u = rho_u[pair_ok]
        rho_c = rho_c[pair_ok]

        all_rho_uncorr.append(rho_u)
        all_rho_corr.append(rho_c)
        subject_per_pair.extend([sr["subject"]] * len(rho_u))

        # Per-dataset Fisher z means (for hierarchical stats)
        if len(rho_u) > 0:
            rho_u_meanz_by_ds.append(fisher_z_mean(rho_u, eps=EPS_RHO))
            rho_c_meanz_by_ds.append(fisher_z_mean(rho_c, eps=EPS_RHO))
            rho_delta_meanz_by_ds.append(
                fisher_z_mean(rho_c, eps=EPS_RHO) - fisher_z_mean(rho_u, eps=EPS_RHO)
            )

        # Store covariance matrices (valid neurons only)
        all_Ctotal.append(Ctotal[np.ix_(valid, valid)])
        all_Cpsth.append(Cpsth[np.ix_(valid, valid)])
        all_Crate.append(Crate[np.ix_(valid, valid)])
        all_CnoiseU.append(CnoiseU[np.ix_(valid, valid)])
        all_CnoiseC.append(CnoiseC[np.ix_(valid, valid)])
        all_Cfem.append(Cfem[np.ix_(valid, valid)])

        # Shuffle nulls (if available)
        if "Shuffled_Intercepts" in mats and len(mats["Shuffled_Intercepts"]) > 0:
            for Crate_shuf in mats["Shuffled_Intercepts"]:
                # Alpha under shuffle
                diag_rate_shuf = np.diag(Crate_shuf)[valid]
                alpha_shuf = diag_psth / diag_rate_shuf
                alpha_shuf = np.clip(alpha_shuf, 0, 1)
                shuff_alphas.append(1 - alpha_shuf)

                # Noise corr under shuffle
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
                    shuff_rho_subject.append(sr["subject"])

    metrics.append({
        "window_ms": WINDOWS_MS[w_idx],
        "window_bins": WINDOWS_BINS[w_idx],
        "alpha": np.concatenate(all_alpha) if all_alpha else np.array([]),
        "uncorr": np.concatenate(all_ff_uncorr) if all_ff_uncorr else np.array([]),
        "corr": np.concatenate(all_ff_corr) if all_ff_corr else np.array([]),
        "erate": np.concatenate(all_erate) if all_erate else np.array([]),
        "rho_uncorr": np.concatenate(all_rho_uncorr) if all_rho_uncorr else np.array([]),
        "rho_corr": np.concatenate(all_rho_corr) if all_rho_corr else np.array([]),
        "rho_u_meanz_by_ds": np.array(rho_u_meanz_by_ds),
        "rho_c_meanz_by_ds": np.array(rho_c_meanz_by_ds),
        "rho_delta_meanz_by_ds": np.array(rho_delta_meanz_by_ds),
        "subject_by_ds": subject_by_ds,
        "subject_per_neuron": np.array(subject_per_neuron),
        "subject_per_pair": np.array(subject_per_pair),
        "Ctotal": all_Ctotal,
        "Cpsth": all_Cpsth,
        "Crate": all_Crate,
        "CnoiseU": all_CnoiseU,
        "CnoiseC": all_CnoiseC,
        "Cfem": all_Cfem,
        "shuff_alphas": shuff_alphas,  # list of arrays (inhomogeneous neuron counts)
        "shuff_rho_delta_meanz": np.array(shuff_rho_delta_meanz),
        "shuff_rho_c_meanz": np.array(shuff_rho_c_meanz),
        "shuff_rho_subject": np.array(shuff_rho_subject),
    })

    m = metrics[-1]
    print(f"Window {WINDOWS_MS[w_idx]:.1f} ms ({WINDOWS_BINS[w_idx]} bins): "
          f"{len(m['alpha'])} neurons, "
          f"{len(m['rho_uncorr'])} pairs, "
          f"{len(m['shuff_alphas'])} shuffle iterations")

# %% Panel C: FEM modulation fraction (1-alpha)
# Fraction of rate variance attributable to fixational eye movements.
# m = 1 - alpha, where alpha = diag(Cpsth) / diag(Crate).
# Higher m means more variance from FEMs.

m_by_window = []
alpha_stats = {}

for w_idx, m_dict in enumerate(metrics):
    alpha = m_dict["alpha"]
    m = 1 - alpha  # FEM modulation fraction
    m_by_window.append(m)

    # Descriptive stats
    mean_m, (ci_lo, ci_hi) = bootstrap_mean_ci(m, nboot=5000, seed=0)
    med_m = float(np.nanmedian(m))
    q25, q75 = iqr_25_75(m)

    # Shuffle null
    shuff_m = m_dict["shuff_alphas"]  # list of 1D arrays, already 1-alpha
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

# Plot: histogram for primary window — per-animal overlay
m0_full = m_by_window[0]
s0 = alpha_stats[WINDOWS_MS[0]]
labels = metrics[0]["subject_per_neuron"]

fig_c, ax_c = plt.subplots(figsize=(4, 3.5))

# Shared bin edges across subjects
valid_m0 = m0_full[np.isfinite(m0_full)]
bins = np.linspace(np.nanmin(valid_m0), np.nanmax(valid_m0), 31)

for subj in SUBJECTS:
    mask = labels == subj
    if not mask.any():
        continue
    m0 = m0_full[mask]
    color = SUBJECT_COLORS[subj]
    ax_c.hist(m0, bins=bins, color=color, edgecolor="white", alpha=0.5)
    ax_c.axvline(np.nanmedian(m0), color=color, linewidth=2, ls=(0, (1,1)),
                 label=f"Median={np.nanmedian(m0):.2f}")

#ax_c.axvspan(s0["null_ci"][0], s0["null_ci"][1], alpha=0.2, color="gray",
             #label="shuffle 95% CI")
ax_c.set_xlabel("Fraction of rate modulation\ndue to FEM (1-α)", fontsize=11)
ax_c.set_ylabel("Count", fontsize=11)
#ax_c.set_title(f"Panel C: FEM modulation ({WINDOWS_MS[0]:.1f} ms)")
ax_c.legend(frameon=False, fontsize=11)
ax_c.grid(True, alpha=0.3, zorder=-1)
# Remove top and right spines
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

fig_c.tight_layout()
fig_c.savefig(FIG_DIR / "panel_c_alpha.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_c)

# %% Panel D-E: Fano factors
# D: mean-variance scatter (single window) with slope fits
# E: population Fano factor vs counting window
#
# Fano factor = Var / Mean for each neuron.
# "Uncorrected" = diag(Ctotal - Cpsth) / E[rate]  (classical noise)
# "Corrected"   = diag(Ctotal - Crate) / E[rate]  (after FEM removal)
# Population FF = slope of variance vs mean across neurons (Churchland et al. 2010)

fano_stats = {}

for w_idx, m_dict in enumerate(metrics):
    ff_u, ff_c, erate = m_dict["uncorr"], m_dict["corr"], m_dict["erate"]

    # Filter to valid neurons (finite, positive rate and FF)
    ff_u_v, ff_c_v, mask = paired_valid(ff_u, ff_c, positive=True)
    erate_v = erate[mask]
    subject_labels_v = m_dict["subject_per_neuron"][mask]
    n_valid = len(ff_u_v)

    # Per-neuron geometric means
    g_unc = geomean(ff_u_v)
    g_cor = geomean(ff_c_v)
    ratio = g_cor / g_unc
    pct_red = (1 - ratio) * 100

    # Wilcoxon signed-rank (corrected < uncorrected?)
    _, p_wil = wilcoxon_signed_rank(ff_c_v, ff_u_v, alternative="less")

    # Population FF: slope of variance vs mean through origin
    # var_uncorr = ff_uncorr * erate, so we need the raw variances
    var_u = ff_u_v * erate_v
    var_c = ff_c_v * erate_v

    # Slope through origin: slope = sum(x*y) / sum(x^2)
    slope_unc = float(np.sum(erate_v * var_u) / np.sum(erate_v ** 2))
    slope_cor = float(np.sum(erate_v * var_c) / np.sum(erate_v ** 2))

    # Bootstrap CI on slopes (paired resampling)
    rng = np.random.default_rng(0)
    nboot = 5000
    slopes_unc_boot = np.empty(nboot)
    slopes_cor_boot = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n_valid, size=n_valid)
        e_b = erate_v[idx]
        slopes_unc_boot[b] = np.sum(e_b * var_u[idx]) / np.sum(e_b ** 2)
        slopes_cor_boot[b] = np.sum(e_b * var_c[idx]) / np.sum(e_b ** 2)

    diff_boot = slopes_unc_boot - slopes_cor_boot
    slope_diff = slope_unc - slope_cor
    slope_diff_ci = (float(np.percentile(diff_boot, 2.5)),
                     float(np.percentile(diff_boot, 97.5)))
    p_slope = float(np.mean(diff_boot <= 0))  # one-sided: diff > 0?

    # Shuffle null ratio — not computed here because shuff_alphas are
    # per-session while ff_u_v is concatenated across sessions.
    # The alpha panel (Panel C) handles the shuffle null directly.
    null_ratio_ci = (np.nan, np.nan)

    fano_stats[WINDOWS_MS[w_idx]] = {
        "n": n_valid, "g_unc": g_unc, "g_cor": g_cor,
        "ratio": ratio, "pct_red": pct_red, "p_wil": p_wil,
        "slope_unc": slope_unc, "slope_cor": slope_cor,
        "slope_diff": slope_diff, "slope_diff_ci": slope_diff_ci,
        "p_slope": p_slope, "null_ratio_ci": null_ratio_ci,
        "erate": erate_v, "var_u": var_u, "var_c": var_c,
        "subject_per_neuron": subject_labels_v,
    }

    print(f"\nWindow {WINDOWS_MS[w_idx]:.1f} ms (N={n_valid}):")
    print(f"  FF uncorr: gmean={g_unc:.3f}")
    print(f"  FF corr:   gmean={g_cor:.3f}")
    print(f"  Ratio={ratio:.3f} ({pct_red:.1f}% reduction), Wilcoxon p={p_wil:.3g}")
    print(f"  Population FF: uncorr={slope_unc:.3f}, corr={slope_cor:.3f}")
    print(f"  Slope diff={slope_diff:.3f} CI [{slope_diff_ci[0]:.3f}, {slope_diff_ci[1]:.3f}]")

# --- Plot Panel D: mean-variance scatter (split Allen / Logan) ---
s0 = fano_stats[WINDOWS_MS[0]]
labels_d = s0["subject_per_neuron"]

fig_d, (ax_da, ax_db) = plt.subplots(1, 2, figsize=(7, 3.5))
for ax_d_sub, subj in [(ax_da, "Allen"), (ax_db, "Logan")]:
    mask = labels_d == subj
    if not mask.any():
        continue
    e_sub = s0["erate"][mask]
    vu_sub = s0["var_u"][mask]
    vc_sub = s0["var_c"][mask]
    color = SUBJECT_COLORS[subj]
    slope_u = float(np.sum(e_sub * vu_sub) / np.sum(e_sub ** 2))
    slope_c = float(np.sum(e_sub * vc_sub) / np.sum(e_sub ** 2))
    ax_d_sub.scatter(e_sub, vu_sub, s=8, alpha=0.3, c=color)
    ax_d_sub.scatter(e_sub, vc_sub, s=8, alpha=0.3, c=color, marker="^")
    x_line = np.linspace(0, e_sub.max(), 100)
    ax_d_sub.plot(x_line, slope_u * x_line, color=color, ls="--", linewidth=1.5,
                  label=f"Uncorr={slope_u:.3f}")
    ax_d_sub.plot(x_line, slope_c * x_line, color=color, ls=":", linewidth=1.5,
                  label=f"Corr={slope_c:.3f}")
    ax_d_sub.set_xlabel("Mean rate")
    ax_d_sub.set_xscale("log")
    ax_d_sub.set_yscale("log")
    ax_d_sub.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d_sub.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d_sub.set_title(subj)
    ax_d_sub.legend(frameon=False, fontsize=8)
    ax_d_sub.grid(True, alpha=0.3)
    ax_d_sub.spines["right"].set_visible(False)
    ax_d_sub.spines["top"].set_visible(False)
ax_da.set_ylabel("Variance")
fig_d.tight_layout()
fig_d.savefig(FIG_DIR / "panel_d_mean_var.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_d)

# --- Plot Panel E: population FF vs window ---
fig_e, ax_e = plt.subplots(figsize=(4, 3))
for subj in SUBJECTS:
    slopes_unc_sub = []
    slopes_cor_sub = []
    for w_idx_e, m_dict in enumerate(metrics):
        n_mask = m_dict["subject_per_neuron"] == subj
        e_sub = m_dict["erate"][n_mask]
        ff_u_sub = m_dict["uncorr"][n_mask]
        ff_c_sub = m_dict["corr"][n_mask]
        ok = (np.isfinite(ff_u_sub) & np.isfinite(ff_c_sub)
              & (ff_u_sub > 0) & (ff_c_sub > 0) & (e_sub > 0))
        e_v = e_sub[ok]
        vu = ff_u_sub[ok] * e_v
        vc = ff_c_sub[ok] * e_v
        if len(e_v) > 0:
            slopes_unc_sub.append(float(np.sum(e_v * vu) / np.sum(e_v ** 2)))
            slopes_cor_sub.append(float(np.sum(e_v * vc) / np.sum(e_v ** 2)))
        else:
            slopes_unc_sub.append(np.nan)
            slopes_cor_sub.append(np.nan)
    if not any(np.isfinite(slopes_unc_sub)):
        continue
    color = SUBJECT_COLORS[subj]
    ax_e.plot(WINDOWS_MS, slopes_unc_sub, "o-", color=color,
              label=f"{subj} Uncorrected")
    ax_e.plot(WINDOWS_MS, slopes_cor_sub, "o--", color=color,
              label=f"{subj} FEM-corrected")
ax_e.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Poisson")
ax_e.set_xlabel("Counting window (ms)")
ax_e.set_ylabel("Population Fano factor")
ax_e.legend(frameon=False, fontsize=8)
ax_e.set_xticks(WINDOWS_MS)
ax_e.set_xticklabels([f"{w:.1f}" for w in WINDOWS_MS])
fig_e.tight_layout()
fig_e.savefig(FIG_DIR / "panel_e_fano_vs_window.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_e)

# %% Panel F-H: Noise correlations
# F: scatter of corrected vs uncorrected noise correlations (single window)
# G: mean Fisher-z noise correlations vs counting window
# H: effect size (delta z) vs window with shuffle null band
#
# Noise correlations are computed per-pair from the off-diagonal of the
# noise covariance matrix normalized to correlation. The "uncorrected" noise
# covariance is Ctotal - Cpsth; "corrected" is Ctotal - Crate.
#
# We use Fisher z-transform for inference because raw rho is bounded and
# the sampling distribution is skewed, especially near 0.

nc_stats = {}

for w_idx, m_dict in enumerate(metrics):
    rho_u = m_dict["rho_uncorr"]
    rho_c = m_dict["rho_corr"]
    n_pairs = len(rho_u)

    # Per-dataset Fisher z means (hierarchical)
    z_u_ds = m_dict["rho_u_meanz_by_ds"]
    z_c_ds = m_dict["rho_c_meanz_by_ds"]
    dz_ds = m_dict["rho_delta_meanz_by_ds"]
    n_ds = len(z_u_ds)

    # Bootstrap CIs on per-dataset means
    z_u_mean, z_u_ci = bootstrap_mean_ci(z_u_ds, nboot=5000, seed=0)
    z_c_mean, z_c_ci = bootstrap_mean_ci(z_c_ds, nboot=5000, seed=0)
    dz_mean, dz_ci = bootstrap_mean_ci(dz_ds, nboot=5000, seed=0)

    # Wilcoxon on paired per-dataset deltas
    if n_ds >= 5:
        _, p_wil = wilcoxon_signed_rank(z_c_ds, z_u_ds, alternative="less")
    else:
        p_wil = np.nan

    # Shuffle null (pooled)
    shuff_dz = m_dict["shuff_rho_delta_meanz"]
    shuff_subj = m_dict["shuff_rho_subject"]
    if len(shuff_dz) > 0:
        null_dz_ci = (float(np.percentile(shuff_dz, 2.5)),
                      float(np.percentile(shuff_dz, 97.5)))
        p_emp_dz = emp_p_one_sided(shuff_dz, dz_mean, direction="less")
    else:
        null_dz_ci = (np.nan, np.nan)
        p_emp_dz = np.nan

    # Shuffle null per subject
    null_dz_ci_by_subject = {}
    for subj in SUBJECTS:
        s_mask = shuff_subj == subj
        if s_mask.sum() > 0:
            null_dz_ci_by_subject[subj] = (
                float(np.percentile(shuff_dz[s_mask], 2.5)),
                float(np.percentile(shuff_dz[s_mask], 97.5)),
            )
        else:
            null_dz_ci_by_subject[subj] = (np.nan, np.nan)

    nc_stats[WINDOWS_MS[w_idx]] = {
        "n_pairs": n_pairs, "n_ds": n_ds,
        "z_u_mean": z_u_mean, "z_u_ci": z_u_ci,
        "z_c_mean": z_c_mean, "z_c_ci": z_c_ci,
        "dz_mean": dz_mean, "dz_ci": dz_ci,
        "p_wil": p_wil, "null_dz_ci": null_dz_ci, "p_emp_dz": p_emp_dz,
        "null_dz_ci_by_subject": null_dz_ci_by_subject,
        "rho_u": rho_u, "rho_c": rho_c,
    }

    print(f"\nWindow {WINDOWS_MS[w_idx]:.1f} ms ({n_pairs} pairs, {n_ds} datasets):")
    print(f"  z_uncorr = {z_u_mean:.4f} [{z_u_ci[0]:.4f}, {z_u_ci[1]:.4f}]")
    print(f"  z_corr   = {z_c_mean:.4f} [{z_c_ci[0]:.4f}, {z_c_ci[1]:.4f}]")
    print(f"  delta_z  = {dz_mean:.4f} [{dz_ci[0]:.4f}, {dz_ci[1]:.4f}]")
    print(f"  Wilcoxon p={p_wil:.3g}")
    print(f"  Shuffle null delta_z 95% CI: [{null_dz_ci[0]:.4f}, {null_dz_ci[1]:.4f}]")
    for subj in SUBJECTS:
        ci_s = null_dz_ci_by_subject[subj]
        print(f"    {subj} shuffle 95% CI: [{ci_s[0]:.4f}, {ci_s[1]:.4f}]")
    print(f"  Empirical p={p_emp_dz:.4f}")

# --- Plot Panel F: noise corr scatter (10 ms) ---
s0 = nc_stats[WINDOWS_MS[0]]
pair_labels = metrics[0]["subject_per_pair"]

fig_f, ax_f = plt.subplots(figsize=(4, 4))
for subj in SUBJECTS:
    mask = pair_labels == subj
    if not mask.any():
        continue
    color = SUBJECT_COLORS[subj]
    ax_f.scatter(s0["rho_u"][mask], s0["rho_c"][mask],
                 s=2, alpha=0.05, c=color, rasterized=True)
    ax_f.plot(np.mean(s0["rho_u"][mask]), np.mean(s0["rho_c"][mask]),
              "o", color=color, markersize=6, markeredgecolor="black",
              markeredgewidth=0.5, label=subj)
ax_f.plot([-0.3, 0.3], [-0.3, 0.3], "k--", alpha=0.3, linewidth=0.5)
ax_f.set_xlim(-0.3, 0.3)
ax_f.set_ylim(-0.3, 0.3)
ax_f.set_xlabel("ρ uncorrected")
ax_f.set_ylabel("ρ FEM-corrected")
ax_f.legend(frameon=False, fontsize=8)
fig_f.tight_layout()
fig_f.savefig(FIG_DIR / "panel_f_noisecorr_scatter.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_f)

# --- Plot Panel G: mean Fisher z vs window ---
fig_g, ax_g = plt.subplots(figsize=(4, 3))
for subj in SUBJECTS:
    for label, key, ls in [("Uncorr", "u", "-"), ("Corr", "c", "--")]:
        means = []
        ci_lo_list = []
        ci_hi_list = []
        for w_idx_g, m_dict in enumerate(metrics):
            ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
            vals = m_dict[f"rho_{key}_meanz_by_ds"][ds_mask]
            if len(vals) > 0:
                mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
                means.append(mn)
                ci_lo_list.append(ci[0])
                ci_hi_list.append(ci[1])
            else:
                means.append(np.nan)
                ci_lo_list.append(np.nan)
                ci_hi_list.append(np.nan)
        if not any(np.isfinite(means)):
            continue
        color = SUBJECT_COLORS[subj]
        ax_g.errorbar(WINDOWS_MS, means,
                      yerr=[np.array(means) - ci_lo_list, np.array(ci_hi_list) - means],
                      fmt=f"o{ls}", color=color, capsize=3,
                      label=f"{subj} {label}")
ax_g.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax_g.set_xlabel("Counting window (ms)")
ax_g.set_ylabel("Mean Fisher z")
ax_g.legend(frameon=False, fontsize=8)
ax_g.set_xticks(WINDOWS_MS)
ax_g.set_xticklabels([f"{w:.1f}" for w in WINDOWS_MS])
fig_g.tight_layout()
fig_g.savefig(FIG_DIR / "panel_g_noisecorr_vs_window.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_g)

# --- Plot Panel H: delta z vs window with shuffle null ---
fig_h, ax_h = plt.subplots(figsize=(4, 3))
for subj in SUBJECTS:
    dz_means_sub = []
    dz_lo_sub = []
    dz_hi_sub = []
    for w_idx_h, m_dict in enumerate(metrics):
        ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
        vals = m_dict["rho_delta_meanz_by_ds"][ds_mask]
        if len(vals) > 0:
            mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
            dz_means_sub.append(mn)
            dz_lo_sub.append(ci[0])
            dz_hi_sub.append(ci[1])
        else:
            dz_means_sub.append(np.nan)
            dz_lo_sub.append(np.nan)
            dz_hi_sub.append(np.nan)
    if not any(np.isfinite(dz_means_sub)):
        continue
    color = SUBJECT_COLORS[subj]
    ax_h.errorbar(WINDOWS_MS, dz_means_sub,
                  yerr=[np.array(dz_means_sub) - dz_lo_sub,
                        np.array(dz_hi_sub) - dz_means_sub],
                  fmt="o-", color=color, capsize=3, label=subj)
    null_lo_sub = [nc_stats[w]["null_dz_ci_by_subject"][subj][0] for w in WINDOWS_MS]
    null_hi_sub = [nc_stats[w]["null_dz_ci_by_subject"][subj][1] for w in WINDOWS_MS]
    ax_h.fill_between(WINDOWS_MS, null_lo_sub, null_hi_sub, alpha=0.15, color=color,
                      label=f"{subj} shuffle 95% CI")
ax_h.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax_h.set_xlabel("Counting window (ms)")
ax_h.set_ylabel("Δz (corr - uncorr)")
ax_h.legend(frameon=False, fontsize=8)
ax_h.set_xticks(WINDOWS_MS)
ax_h.set_xticklabels([f"{w:.1f}" for w in WINDOWS_MS])
fig_h.tight_layout()
fig_h.savefig(FIG_DIR / "panel_h_effect_size.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_h)

# %% Panel I-K: Subspace alignment
# I: eigenspectra of Cpsth and Cfem (log-log) with IQR bands
# J: participation ratio bars per session
# K: subspace alignment scatter (X vs Y) with shuffle controls
#
# For each session, we PSD-project Cpsth and Cfem, compute eigendecompositions,
# participation ratios, and subspace overlap / directional variance capture.

w_idx = SUBSPACE_WINDOW_IDX  # 20 ms window

# Per-session storage
sub_names = []
sub_subjects = []
pr_fem_list = []
pr_psth_list = []
overlap_k1_list = []
overlap_k_list = []
var_f_given_p = []  # FEM variance captured by PSTH subspace (Y)
var_p_given_f = []  # PSTH variance captured by FEM subspace (X)
spectra_psth = []
spectra_fem = []

for ds_idx, sr in enumerate(session_results):
    if w_idx >= len(sr["mats"]):
        continue
    mats = sr["mats"][w_idx]
    Cpsth = mats["PSTH"]
    Crate = mats["Intercept"]
    Ctotal = mats["Total"]
    Cfem = Crate - Cpsth

    # Apply neuron inclusion (same as metrics extraction)
    erate = sr["results"][w_idx]["Erates"]
    total_spikes = erate * sr["results"][w_idx]["n_samples"]
    valid = (
        np.isfinite(erate)
        & (total_spikes >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid.sum() < SUBSPACE_K + 1:
        continue

    Cpsth_v = Cpsth[np.ix_(valid, valid)]
    Cfem_v = Cfem[np.ix_(valid, valid)]
    Ctotal_v = Ctotal[np.ix_(valid, valid)]

    # PSD projection (handles numerical negatives from split-half estimation)
    Cpsth_psd = project_to_psd(Cpsth_v)
    Cfem_psd = project_to_psd(Cfem_v)

    # Eigendecomposition
    w_psth, V_psth = np.linalg.eigh(Cpsth_psd)
    w_fem, V_fem = np.linalg.eigh(Cfem_psd)
    # Sort descending
    w_psth, V_psth = w_psth[::-1], V_psth[:, ::-1]
    w_fem, V_fem = w_fem[::-1], V_fem[:, ::-1]

    # Participation ratios
    pr_psth = participation_ratio(Cpsth_psd)
    pr_fem = participation_ratio(Cfem_psd)
    pr_psth_list.append(pr_psth)
    pr_fem_list.append(pr_fem)

    # Subspace overlap (top-k eigenvectors)
    k = min(SUBSPACE_K, valid.sum() - 1)
    U_psth = V_psth[:, :k]
    U_fem = V_fem[:, :k]
    overlap_k_list.append(symmetric_subspace_overlap(U_psth, U_fem))
    overlap_k1_list.append(symmetric_subspace_overlap(V_psth[:, :1], V_fem[:, :1]))

    # Directional variance capture
    # X = PSTH variance captured by FEM subspace
    # Y = FEM variance captured by PSTH subspace
    var_p_given_f.append(directional_variance_capture(Cpsth_psd, U_fem))
    var_f_given_p.append(directional_variance_capture(Cfem_psd, U_psth))

    # Normalized eigenspectra for plotting
    tr_total = np.trace(Ctotal_v)
    spectra_psth.append(w_psth / tr_total)
    spectra_fem.append(w_fem / tr_total)

    sub_names.append(session_names[ds_idx])
    sub_subjects.append(subjects[ds_idx])

# Summary stats
print(f"\nSubspace analysis ({WINDOWS_MS[w_idx]:.1f} ms, {len(sub_names)} sessions):")
print(f"  PR(FEM):  mean={np.mean(pr_fem_list):.3f} ± {np.std(pr_fem_list)/np.sqrt(len(pr_fem_list)):.3f}")
print(f"  PR(PSTH): mean={np.mean(pr_psth_list):.3f} ± {np.std(pr_psth_list)/np.sqrt(len(pr_psth_list)):.3f}")
print(f"  Overlap k=1: mean={np.mean(overlap_k1_list):.3f}")
print(f"  Overlap k={SUBSPACE_K}: mean={np.mean(overlap_k_list):.3f}")
print(f"  X (PSTH var in FEM subspace): mean={np.mean(var_p_given_f):.3f} ± {np.std(var_p_given_f)/np.sqrt(len(var_p_given_f)):.3f}")
print(f"  Y (FEM var in PSTH subspace): mean={np.mean(var_f_given_p):.3f} ± {np.std(var_f_given_p)/np.sqrt(len(var_f_given_p)):.3f}")

# --- Plot Panel I: eigenspectra ---
fig_i, ax_i = plt.subplots(figsize=(4, 3.5))
max_dims = 12
for subj in SUBJECTS:
    s_mask = np.array(sub_subjects) == subj
    if not s_mask.any():
        continue
    color = SUBJECT_COLORS[subj]
    for spec_list, ls, label_type in [(spectra_psth, "-", "PSTH"),
                                       (spectra_fem, "--", "FEM")]:
        spec_sub = [s for s, m in zip(spec_list, s_mask) if m]
        if not spec_sub:
            continue
        all_spec = np.full((len(spec_sub), max_dims), np.nan)
        for i, s in enumerate(spec_sub):
            L = min(len(s), max_dims)
            all_spec[i, :L] = s[:L]
        median = np.nanmedian(all_spec, axis=0)
        q25 = np.nanpercentile(all_spec, 25, axis=0)
        q75 = np.nanpercentile(all_spec, 75, axis=0)
        dims = np.arange(1, max_dims + 1)
        ax_i.plot(dims, median, color=color, ls=ls, label=f"{subj} {label_type}")
        ax_i.fill_between(dims, q25, q75, color=color, alpha=0.15)
ax_i.set_xlim(1, max_dims)
ax_i.set_xlabel("Eigenvalue rank")
ax_i.set_ylabel("Fraction of total variance")
ax_i.legend(frameon=False, fontsize=8)
fig_i.tight_layout()
fig_i.savefig(FIG_DIR / "panel_i_eigenspectra.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_i)

# --- Plot Panel J: participation ratio scatter ---
fig_j, ax_j = plt.subplots(figsize=(4, 4))
for subj in sorted(set(sub_subjects)):
    s_mask = np.array(sub_subjects) == subj
    ax_j.scatter(np.array(pr_psth_list)[s_mask], np.array(pr_fem_list)[s_mask],
                 c=SUBJECT_COLORS.get(subj, "gray"), s=40,
                 edgecolors="black", linewidths=0.5, label=subj)
pr_max = max(np.max(pr_psth_list), np.max(pr_fem_list)) * 1.1
ax_j.plot([0, pr_max], [0, pr_max], "k--", alpha=0.3)
ax_j.set_xlim(0, pr_max)
ax_j.set_ylim(0, pr_max)
ax_j.set_xlabel("PSTH participation ratio")
ax_j.set_ylabel("FEM participation ratio")
ax_j.legend(frameon=False, fontsize=8)
fig_j.tight_layout()
fig_j.savefig(FIG_DIR / "panel_j_participation_ratio.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_j)

# --- Plot Panel K: X vs Y scatter (color by subject) ---
fig_k, ax_k = plt.subplots(figsize=(4, 4))
for subj in sorted(set(sub_subjects)):
    s_mask = np.array(sub_subjects) == subj
    ax_k.scatter(np.array(var_p_given_f)[s_mask], np.array(var_f_given_p)[s_mask],
                 c=SUBJECT_COLORS.get(subj, "gray"), s=40,
                 edgecolors="black", linewidths=0.5, label=subj)
ax_k.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax_k.set_xlabel("X: PSTH var captured by FEM subspace")
ax_k.set_ylabel("Y: FEM var captured by PSTH subspace")
ax_k.set_xlim(0, 1)
ax_k.set_ylim(0, 1)
ax_k.legend(frameon=False, fontsize=8)
fig_k.tight_layout()
fig_k.savefig(FIG_DIR / "panel_k_subspace_alignment.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_k)

# %% Per-session covariance heatmaps (supplemental)
cmap = plt.get_cmap("RdBu")
heatmap_window_idx = 3  # 80 ms

for ds_idx, sr in enumerate(session_results):
    if heatmap_window_idx >= len(sr["mats"]):
        continue
    mats = sr["mats"][heatmap_window_idx]
    Crate_raw = mats["Intercept"]

    # Exclude neurons with failed intercept fitting (NaN rows/cols)
    hm_valid = np.isfinite(np.diag(Crate_raw)) & np.isfinite(np.diag(mats["PSTH"]))
    ix = np.ix_(hm_valid, hm_valid)

    Ctotal = project_to_psd(mats["Total"][ix])
    Cpsth = project_to_psd(mats["PSTH"][ix])
    Cfem = project_to_psd(Crate_raw[ix] - mats["PSTH"][ix])
    CnoiseC = project_to_psd(mats["Total"][ix] - Crate_raw[ix])

    v = np.nanmax(np.abs(Ctotal)) * 0.5

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{sr['session']} ({sr['subject']})", fontsize=14)
    for ax, mat, title, vscale in zip(
        axs,
        [Ctotal, Cfem, Cpsth, CnoiseC],
        ["Total", "FEM", "PSTH", "Noise (Corrected)"],
        [1.0, 1.0, 0.5, 1.0],
    ):
        ax.imshow(mat, cmap=cmap, interpolation="nearest",
                  vmin=-v * vscale, vmax=v * vscale)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(FIG_DIR / f"cov_decomp_session{ds_idx}.pdf",
                bbox_inches="tight", dpi=300)
    show_or_close(fig)


# %% Save stats report
import sys

stats_file = STAT_DIR / "fig2_stats.txt"


class _Tee:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
    def flush(self):
        self.file.flush()
        self.stream.flush()


with open(stats_file, "w") as f:
    old_stdout = sys.stdout
    sys.stdout = _Tee(f, old_stdout)

    print("=" * 80)
    print("FIGURE 2: LOTC COVARIANCE DECOMPOSITION STATISTICS")
    print("=" * 80)
    print(f"Sessions: {n_sessions} ({', '.join(sorted(set(subjects)))})")
    print(f"Windows (bins): {WINDOWS_BINS} -> (ms): {[f'{w:.1f}' for w in WINDOWS_MS]}")

    print("\n" + "=" * 80)
    print("ALPHA (FEM MODULATION FRACTION)")
    print("=" * 80)
    for w in WINDOWS_MS:
        s = alpha_stats[w]
        print(f"\nWindow {w:.1f} ms (N={s['n']}):")
        print(f"  1-alpha: mean={s['mean']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")
        print(f"  median={s['median']:.3f} IQR=[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}]")
        print(f"  Shuffle null 95% CI: [{s['null_ci'][0]:.3f}, {s['null_ci'][1]:.3f}]")
        print(f"  Empirical p={s['p_emp']:.4f}")

    print("\n" + "=" * 80)
    print("FANO FACTOR STATISTICS")
    print("=" * 80)
    for w in WINDOWS_MS:
        s = fano_stats[w]
        print(f"\nWindow {w:.1f} ms (N={s['n']}):")
        print(f"  Uncorr: gmean={s['g_unc']:.3f}")
        print(f"  Corr:   gmean={s['g_cor']:.3f}")
        print(f"  Ratio={s['ratio']:.3f} ({s['pct_red']:.1f}% reduction)")
        print(f"  Wilcoxon p={s['p_wil']:.3g}")
        print(f"  Pop FF: uncorr={s['slope_unc']:.3f}, corr={s['slope_cor']:.3f}")
        print(f"  Slope diff={s['slope_diff']:.3f} CI [{s['slope_diff_ci'][0]:.3f}, {s['slope_diff_ci'][1]:.3f}]")

    print("\n" + "=" * 80)
    print("NOISE CORRELATION STATISTICS")
    print("=" * 80)
    for w in WINDOWS_MS:
        s = nc_stats[w]
        print(f"\nWindow {w:.1f} ms ({s['n_pairs']} pairs, {s['n_ds']} datasets):")
        print(f"  z_uncorr = {s['z_u_mean']:.4f} [{s['z_u_ci'][0]:.4f}, {s['z_u_ci'][1]:.4f}]")
        print(f"  z_corr   = {s['z_c_mean']:.4f} [{s['z_c_ci'][0]:.4f}, {s['z_c_ci'][1]:.4f}]")
        print(f"  delta_z  = {s['dz_mean']:.4f} [{s['dz_ci'][0]:.4f}, {s['dz_ci'][1]:.4f}]")
        print(f"  Wilcoxon p={s['p_wil']:.3g}, empirical p={s['p_emp_dz']:.4f}")

    print("\n" + "=" * 80)
    print("SUBSPACE STATISTICS")
    print("=" * 80)
    print(f"  PR(FEM):  mean={np.mean(pr_fem_list):.3f}")
    print(f"  PR(PSTH): mean={np.mean(pr_psth_list):.3f}")
    print(f"  X: mean={np.mean(var_p_given_f):.3f}")
    print(f"  Y: mean={np.mean(var_f_given_p):.3f}")

    print("\n" + "=" * 80)
    print("END OF STATISTICS")
    print("=" * 80)

    sys.stdout = old_stdout

print(f"Stats saved to {stats_file}")


# %% Composite figure (assemble panels C-K using GridSpec)
from matplotlib.gridspec import GridSpec

fig_comp = plt.figure(figsize=(14, 12), constrained_layout=True)
gs = GridSpec(3, 12, figure=fig_comp, hspace=0.1, wspace=0.1)

# Top row: C (wide), D_allen (small square), D_logan (small square), E (wide)
ax_C = fig_comp.add_subplot(gs[0, 0:4])
#ax_Da = fig_comp.add_subplot(gs[0, 4:6])
#ax_Db = fig_comp.add_subplot(gs[0, 6:8])
ax_D = fig_comp.add_subplot(gs[0, 4:8])  # for shared x/y labels
ax_E = fig_comp.add_subplot(gs[0, 8:12])

# Middle row: F, G, H
ax_F = fig_comp.add_subplot(gs[1, 0:4])
ax_G = fig_comp.add_subplot(gs[1, 4:8])
ax_H = fig_comp.add_subplot(gs[1, 8:12])

# Bottom row: I, J, K
ax_I = fig_comp.add_subplot(gs[2, 0:4])
ax_J = fig_comp.add_subplot(gs[2, 4:8])
ax_K = fig_comp.add_subplot(gs[2, 8:12])

# --- C: alpha histogram (per-subject overlay) ---
ax = ax_C
m0_comp = m_by_window[0]
labels_comp = metrics[0]["subject_per_neuron"]
valid_m0_comp = m0_comp[np.isfinite(m0_comp)]
bins_comp = np.linspace(np.nanmin(valid_m0_comp), np.nanmax(valid_m0_comp), 31)
for subj in SUBJECTS:
    mask = labels_comp == subj
    if not mask.any():
        continue
    vals = m0_comp[mask]
    color = SUBJECT_COLORS[subj]
    ax.hist(vals, bins=bins_comp, color=color, edgecolor="white", alpha=0.5)
    ax.axvline(np.nanmedian(vals), color=color, linewidth=2, ls=(0, (1, 1)),
               label=f"Median={np.nanmedian(vals):.2f}")
ax.set_xlabel("1 - α (FEM modulation fraction)")
ax.set_ylabel("Neuron count")
ax.set_title("C")
ax.legend(frameon=False, fontsize=7)

# --- D: mean-variance scatter (split Allen / Logan) ---
s0_d = fano_stats[WINDOWS_MS[0]]
labels_d_comp = s0_d["subject_per_neuron"]
for ax_d_comp, subj in [(ax_D, "Allen"), (ax_D, "Logan")]:
    mask = labels_d_comp == subj
    if not mask.any():
        continue
    e_sub = s0_d["erate"][mask]
    vu_sub = s0_d["var_u"][mask]
    vc_sub = s0_d["var_c"][mask]
    color = SUBJECT_COLORS[subj]
    slope_u = float(np.sum(e_sub * vu_sub) / np.sum(e_sub ** 2))
    slope_c = float(np.sum(e_sub * vc_sub) / np.sum(e_sub ** 2))
    #ax_d_comp.scatter(e_sub, vu_sub, s=6, alpha=0.3, c=color)
    #ax_d_comp.scatter(e_sub, vc_sub, s=6, alpha=0.3, c='red', marker="^")
    #x_line = np.linspace(0, e_sub.max(), 100)
    #ax_d_comp.plot(x_line, slope_u * x_line, color=color, ls="--", linewidth=1,
    #               label=f"Uncorr={slope_u:.2f}")
    #ax_d_comp.plot(x_line, slope_c * x_line, color=color, ls=":", linewidth=1,
    #               label=f"Corr={slope_c:.2f}")
    ax_d_comp.scatter(e_sub, vc_sub - vu_sub, s=6, alpha=0.3, c=color)
    ax_d_comp.set_xlabel("Mean rate")
    #ax_d_comp.set_xscale("log")
    #ax_d_comp.set_yscale("log")
    ax_d_comp.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d_comp.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d_comp.set_title('D')
    ax_d_comp.legend(frameon=False, fontsize=6)
    #ax_d_comp.set_box_aspect(1)
ax_D.set_ylabel("$\\Delta$ Variance (Corr - Uncorr)")
ax_D.set_xscale("log")
ax_D.set_yscale("symlog", linthresh=10**-3)
ax_D.spines["top"].set_visible(False)
ax_D.spines["right"].set_visible(False)
ax_D.grid(True, which="both", linestyle=":", alpha=0.5)

# --- E: population FF vs window (per-subject) ---
ax = ax_E
for subj in SUBJECTS:
    slopes_unc_sub = []
    slopes_cor_sub = []
    for w_idx_e, m_dict in enumerate(metrics):
        n_mask = m_dict["subject_per_neuron"] == subj
        e_sub = m_dict["erate"][n_mask]
        ff_u_sub = m_dict["uncorr"][n_mask]
        ff_c_sub = m_dict["corr"][n_mask]
        ok = (np.isfinite(ff_u_sub) & np.isfinite(ff_c_sub)
              & (ff_u_sub > 0) & (ff_c_sub > 0) & (e_sub > 0))
        e_v = e_sub[ok]
        vu = ff_u_sub[ok] * e_v
        vc = ff_c_sub[ok] * e_v
        if len(e_v) > 0:
            slopes_unc_sub.append(float(np.sum(e_v * vu) / np.sum(e_v ** 2)))
            slopes_cor_sub.append(float(np.sum(e_v * vc) / np.sum(e_v ** 2)))
        else:
            slopes_unc_sub.append(np.nan)
            slopes_cor_sub.append(np.nan)
    if not any(np.isfinite(slopes_unc_sub)):
        continue
    color = SUBJECT_COLORS[subj]
    ax.plot(WINDOWS_MS, slopes_unc_sub, "o-", color=color,
            label=f"{subj} Uncorr")
    ax.plot(WINDOWS_MS, slopes_cor_sub, "o--", color=color,
            label=f"{subj} Corr")
ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Poisson")
ax.set_xlabel("Counting window (ms)")
ax.set_ylabel("Population Fano factor")
ax.set_title("E")
ax.legend(frameon=False, fontsize=7)
ax.set_xticks(WINDOWS_MS)
ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])

# --- F: noise corr scatter (per-subject) ---
ax = ax_F
s0_f = nc_stats[WINDOWS_MS[0]]
pair_labels_comp = metrics[0]["subject_per_pair"]
for subj in SUBJECTS:
    mask = pair_labels_comp == subj
    if not mask.any():
        continue
    color = SUBJECT_COLORS[subj]
    ax.scatter(s0_f["rho_u"][mask], s0_f["rho_c"][mask],
               s=1, alpha=0.05, c=color, rasterized=True)
    ax.plot(np.mean(s0_f["rho_u"][mask]), np.mean(s0_f["rho_c"][mask]),
            "o", color=color, markersize=5, markeredgecolor="black",
            markeredgewidth=0.5, label=subj)
ax.plot([-0.3, 0.3], [-0.3, 0.3], "k--", alpha=0.3, linewidth=0.5)
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_xlabel("ρ uncorrected")
ax.set_ylabel("ρ FEM-corrected")
ax.set_title("F")
ax.legend(frameon=False, fontsize=7)

# --- G: mean Fisher z vs window (per-subject) ---
ax = ax_G
for subj in SUBJECTS:
    for label, key, ls in [("Uncorr", "u", "-"), ("Corr", "c", "--")]:
        means = []
        ci_lo_list = []
        ci_hi_list = []
        for w_idx_g, m_dict in enumerate(metrics):
            ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
            vals = m_dict[f"rho_{key}_meanz_by_ds"][ds_mask]
            if len(vals) > 0:
                mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
                means.append(mn)
                ci_lo_list.append(ci[0])
                ci_hi_list.append(ci[1])
            else:
                means.append(np.nan)
                ci_lo_list.append(np.nan)
                ci_hi_list.append(np.nan)
        if not any(np.isfinite(means)):
            continue
        color = SUBJECT_COLORS[subj]
        ax.errorbar(WINDOWS_MS, means,
                    yerr=[np.array(means) - ci_lo_list, np.array(ci_hi_list) - means],
                    fmt=f"o{ls}", color=color, capsize=3,
                    label=f"{subj} {label}")
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Counting window (ms)")
ax.set_ylabel("Mean Fisher z")
ax.set_title("G")
ax.legend(frameon=False, fontsize=7)
ax.set_xticks(WINDOWS_MS)
ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])

# --- H: delta z vs window (per-subject) ---
ax = ax_H
for subj in SUBJECTS:
    dz_means_sub = []
    dz_lo_sub = []
    dz_hi_sub = []
    for w_idx_h, m_dict in enumerate(metrics):
        ds_mask = np.array([s == subj for s in m_dict["subject_by_ds"]])
        vals = m_dict["rho_delta_meanz_by_ds"][ds_mask]
        if len(vals) > 0:
            mn, ci = bootstrap_mean_ci(vals, nboot=5000, seed=0)
            dz_means_sub.append(mn)
            dz_lo_sub.append(ci[0])
            dz_hi_sub.append(ci[1])
        else:
            dz_means_sub.append(np.nan)
            dz_lo_sub.append(np.nan)
            dz_hi_sub.append(np.nan)
    if not any(np.isfinite(dz_means_sub)):
        continue
    color = SUBJECT_COLORS[subj]
    ax.errorbar(WINDOWS_MS, dz_means_sub,
                yerr=[np.array(dz_means_sub) - dz_lo_sub,
                      np.array(dz_hi_sub) - dz_means_sub],
                fmt="o-", color=color, capsize=3, label=subj)
    null_lo_sub = [nc_stats[w]["null_dz_ci_by_subject"][subj][0] for w in WINDOWS_MS]
    null_hi_sub = [nc_stats[w]["null_dz_ci_by_subject"][subj][1] for w in WINDOWS_MS]
    ax.fill_between(WINDOWS_MS, null_lo_sub, null_hi_sub, alpha=0.15, color=color,
                    label=f"{subj} shuffle 95% CI")
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Counting window (ms)")
ax.set_ylabel("Δz (corr - uncorr)")
ax.set_title("H")
ax.legend(frameon=False, fontsize=7)
ax.set_xticks(WINDOWS_MS)
ax.set_xticklabels([f"{w:.0f}" for w in WINDOWS_MS])

# --- I: eigenspectra (per-subject) ---
ax = ax_I
max_dims = 10
for subj in SUBJECTS:
    s_mask = np.array(sub_subjects) == subj
    if not s_mask.any():
        continue
    color = SUBJECT_COLORS[subj]
    for spec_list, ls, label_type in [(spectra_psth, "-", "PSTH"),
                                       (spectra_fem, "--", "FEM")]:
        spec_sub = [s for s, m in zip(spec_list, s_mask) if m]
        if not spec_sub:
            continue
        all_spec = np.full((len(spec_sub), max_dims), np.nan)
        for i, s in enumerate(spec_sub):
            L = min(len(s), max_dims)
            all_spec[i, :L] = s[:L]
        median = np.nanmedian(all_spec, axis=0)
        q25 = np.nanpercentile(all_spec, 25, axis=0)
        q75 = np.nanpercentile(all_spec, 75, axis=0)
        dims = np.arange(1, max_dims + 1)
        ax.plot(dims, median, color=color, ls=ls, label=f"{subj} {label_type}", marker="o", markersize=4)
        ax.fill_between(dims, q25, q75, color=color, alpha=0.15)
ax.set_xlim(1, max_dims)
ax.set_xlabel("Eigenvalue rank")
ax.set_ylabel("Frac. total variance")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.set_title("I")
ax.legend(frameon=False, fontsize=7)

# --- J: participation ratio (scatter) ---
ax = ax_J
for subj in sorted(set(sub_subjects)):
    s_mask = np.array(sub_subjects) == subj
    ax.scatter(np.array(pr_fem_list)[s_mask], np.array(pr_psth_list)[s_mask],
               c=SUBJECT_COLORS.get(subj, "gray"), s=40,
               edgecolors="black", linewidths=0.5, label=subj)
pr_max = max(np.max(pr_psth_list), np.max(pr_fem_list)) * 1.1
ax.plot([0, pr_max], [0, pr_max], "k--", alpha=0.3)
ax.set_xlim(0, pr_max)
ax.set_ylim(0, pr_max)
ax.set_xlabel("FEM PR")
ax.set_ylabel("PSTH PR")
ax.set_title("J")

# --- K: subspace alignment ---
ax = ax_K
for subj in sorted(set(sub_subjects)):
    s_mask = [s == subj for s in sub_subjects]
    ax.scatter(np.array(var_p_given_f)[s_mask], np.array(var_f_given_p)[s_mask],
               c=SUBJECT_COLORS.get(subj, "gray"), s=40,
               edgecolors="black", linewidths=0.5, label=subj)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("X: PSTH var in FEM subspace")
ax.set_ylabel("Y: FEM var in PSTH subspace")
ax.set_title("K")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7)

# Apply right/top spines off and grid to all panels
for ax in [ax_C, ax_D, ax_E, ax_F, ax_G, ax_H, ax_I, ax_J, ax_K]:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.3)

fig_comp.savefig(FIG_DIR / "fig2_composite.pdf", bbox_inches="tight", dpi=300)
show_or_close(fig_comp)

print(f"\nAll panel figures saved to: {FIG_DIR}")
print("Done.")
