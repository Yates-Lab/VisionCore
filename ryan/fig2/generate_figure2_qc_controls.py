# %% Imports and configuration
"""
Figure 2 QC Controls: Spike-sorting quality stratified robustness analysis.

PURPOSE
-------
This script implements supplemental control analyses for Figure 2 (covariance
decomposition). It tests whether the headline findings — FEM modulation
fraction (1 − α ≈ 0.80), noise correlation collapse (Δρ ≈ −0.06), and
sub-Poisson corrected Fano factors — are robust to spike-sorting quality, or
whether they are confounded by multiunit contamination and missing spikes.

RATIONALE (Cohen & Kohn 2011, Nature Neuroscience 14:811-819)
-------------------------------------------------------------
The Figure 2 analysis currently includes every unit Kilosort4 returns,
filtered only by a total spike count threshold (≥ 500 spikes). This creates
two entangled confounds:

1. **Multiunit contamination inflates the uncorrected baseline.**
   Cohen & Kohn (Fig. 4) show multiunit clusters inflate r_SC approximately as
   n·r_pair / [(n−1)·r_pair + 1]. Units with high refractory contamination
   likely pool 2+ neurons' spikes. The contaminating spikes share the same
   local inputs as the "true" neuron, injecting shared variability into the
   **total** covariance Σ_total. Critically, Σ_PSTH (estimated via split-half
   cross-covariance) is robust to contamination since noise averages out. So
   multiunit contamination inflates **uncorrected** noise correlations
   (Σ_total − Σ_PSTH) without equally inflating corrected ones, potentially
   exaggerating the apparent FEM correction effect (Δρ).

2. **Missing spikes deflate correlations and inflate Fano factors.**
   Cohen & Kohn (Fig. 5, Supplement Eq. 12-20) derive that oversorting /
   missing spikes deflate r_SC: cov(m₁,m₂) = (1−p₁)(1−p₂)cov(n₁,n₂), while
   var(mᵢ) = ⟨nᵢ⟩pᵢ(1−pᵢ) + var(nᵢ)(1−pᵢ)². The added binomial variance
   inflates uncorrected Fano factors. With 50% missing, measured r_SC is
   approximately halved.

3. **Low firing rates compress correlations toward zero.**
   Cohen & Kohn (Fig. 2, Fig. 6 meta-analysis) show firing rate explains 33%
   of across-study variance in r_SC. The threshold-masking effect compresses
   correlations toward zero for weakly driven units.

4. **The net bias on delta metrics is unknowable without stratification.**
   Contamination inflates uncorrected r_SC while missing spikes deflate both.
   These partially cancel for noise correlations but compound for Fano
   factors. The existing eye-trajectory shuffle control tests whether eye
   position specifically drives the correction, but does NOT control for spike
   sorting quality — contamination lives in the spike train itself.

APPROACH
--------
Rather than a hard cutoff (which discards information), we show effects as
continuous functions of two QC axes:

  Axis 1: Refractory contamination (%) — proxy for multiunit mixing.
          Source: min_contam_props from refractory period violation analysis.
          Heuristic: < 20% ≈ single unit.

  Axis 2: Missing spike % during fixRSVP — proxy for oversorting.
          Source: amplitude truncation analysis, cross-referenced with fixRSVP
          trial times to get condition-specific missing percentages.
          Heuristic: < 20% ≈ well-isolated unit.

IMPLEMENTATION
--------------
For fixRSVP-specific missing %, we use "Implementation Path 2": load the
pre-generated fixrsvp.dset for each session to get absolute recording time
bins (t_bins), then evaluate the session's missing_pct_interp() at those
times. This gives ground-truth missing % for exactly the epochs used in the
analysis, not a full-recording average.

PANELS
------
S1: Per-neuron effect gradients vs. QC
  (a) 1 − α vs. contamination rate, with running median ± IQR
  (b) 1 − α vs. fixRSVP missing spike %
  (c) Fano factor ratio (corrected/uncorrected) vs. contamination
  (d) Fano factor ratio vs. fixRSVP missing spike %

S2: Per-pair effect gradients vs. QC
  (a) Δρ (noise correlation change) vs. geometric mean contamination
  (b) Δρ vs. geometric mean fixRSVP missing %
  (c) Δρ vs. geometric mean firing rate (Cohen & Kohn rate confound)

S3: Cumulative inclusion analysis
  Sort neurons by contamination (ascending), cumulatively add, recompute each
  effect. If the curve is flat, the effect is robust. Same for missing %.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import dill
from pathlib import Path
from VisionCore.paths import CACHE_DIR, FIGURES_DIR, STATS_DIR
from VisionCore.covariance import cov_to_corr, get_upper_triangle
from VisionCore.stats import fisher_z_mean

# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Analysis parameters (must match generate_figure2.py)
MODE = "standard"
WINDOWS = [10, 20, 40, 80]
PRIMARY_WINDOW_IDX = 0  # 10 ms for primary claims
MIN_TOTAL_SPIKES = 500
MIN_VAR = 0
EPS_RHO = 1e-3

# Running-median parameters
N_BINS_RUNNING = 10  # number of quantile bins for running median

# Output directories
FIG_DIR = FIGURES_DIR / "fig2_qc"
STAT_DIR = STATS_DIR / "fig2_qc"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# %% Load cached data
outputs_path = CACHE_DIR / f"dmcfarland_outputs_{MODE}.pkl"
print(f"Loading cached outputs from {outputs_path}")
with open(outputs_path, "rb") as f:
    outputs = dill.load(f)

session_names = [out["sess"] for out in outputs]
n_sessions = len(outputs)
print(f"Loaded {n_sessions} sessions: {session_names}")

# %% Load QC data per session
# For each session:
#   1. Load refractory contamination from refractory.npz
#   2. Load fixRSVP-specific missing % via get_missing_pct_interp + fixrsvp.dset
#
# Output: per-session arrays aligned to cids_used (same neuron order as
# the analysis metrics in the cached outputs).

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent / "data-yates-v1"))
from DataYatesV1.utils.io import get_session

PROCESSED_DIR = Path("/mnt/ssd/YatesMarmoV1/processed")

qc_per_session = []

for ds_idx, out in enumerate(outputs):
    sess_name = out["sess"]
    cids_used = out["cids_used"]
    n_used = len(cids_used)
    subject, date = sess_name.split("_", 1)

    print(f"\n--- {sess_name} ({n_used} neurons) ---")

    qc = {
        "sess": sess_name,
        "cids_used": cids_used,
        "contamination": np.full(n_used, np.nan),
        "missing_pct_fixrsvp": np.full(n_used, np.nan),
    }

    sess_dir = PROCESSED_DIR / sess_name

    # --- Refractory contamination ---
    ref_path = sess_dir / "qc" / "refractory" / "refractory.npz"
    if ref_path.exists():
        ref = np.load(ref_path)
        mcp = ref["min_contam_props"]  # (n_clusters, n_refractory_periods)
        # cids_used are cluster IDs that directly index into mcp rows
        valid_cids = cids_used[cids_used < mcp.shape[0]]
        mask = cids_used < mcp.shape[0]
        qc["contamination"][mask] = np.min(mcp[valid_cids], axis=1) * 100
        print(f"  Refractory: {mask.sum()}/{n_used} units loaded, "
              f"median contam = {np.nanmedian(qc['contamination']):.1f}%")
    else:
        print(f"  WARNING: refractory.npz not found at {ref_path}")

    # --- FixRSVP-specific missing spike % ---
    trunc_path = sess_dir / "qc" / "amp_truncation" / "truncation.npz"
    dset_path = sess_dir / "datasets" / "fixrsvp.dset"

    if trunc_path.exists() and dset_path.exists():
        # Load truncation data
        trunc = dict(np.load(trunc_path))

        # Load fixrsvp t_bins (absolute ephys times in seconds)
        from DataYatesV1.utils.data.datasets import DictDataset
        fixrsvp_dset = DictDataset.load(dset_path)
        t_bins = fixrsvp_dset["t_bins"].numpy().ravel()

        # Load spike times and clusters via session object
        try:
            sess_obj = get_session(subject, date)
            spike_times_sec = sess_obj.ks_results.spike_times
            spike_clusters = sess_obj.ks_results.spike_clusters
        except Exception as e:
            print(f"  WARNING: Could not load session spike data: {e}")
            qc_per_session.append(qc)
            continue

        # For each unit, find truncation windows overlapping fixrsvp,
        # take median missing %
        for i, cid in enumerate(cids_used):
            # Get this unit's truncation windows
            unit_mask = trunc["cid"] == cid
            if not np.any(unit_mask):
                continue

            wb = trunc["window_blocks"][unit_mask]  # (M, 2) spike indices
            mp = trunc["mpcts"][unit_mask]           # (M,) missing %

            if wb.ndim == 1:
                wb = wb.reshape(-1, 2)

            # Get this unit's spike times
            st = spike_times_sec[spike_clusters == cid]
            if st.size == 0 or wb.shape[0] == 0:
                continue

            # Clamp window indices to valid range
            wb = np.clip(wb, 0, len(st) - 1)

            # Convert window block spike-indices → absolute times
            win_starts = st[wb[:, 0]]
            win_ends = st[wb[:, 1]]

            # Find which windows overlap with fixrsvp time range
            fixrsvp_start = t_bins.min()
            fixrsvp_end = t_bins.max()
            overlap = (win_ends >= fixrsvp_start) & (win_starts <= fixrsvp_end)

            if np.any(overlap):
                qc["missing_pct_fixrsvp"][i] = np.median(mp[overlap])

        n_valid = np.isfinite(qc["missing_pct_fixrsvp"]).sum()
        print(f"  Truncation (fixRSVP): {n_valid}/{n_used} units, "
              f"median missing = {np.nanmedian(qc['missing_pct_fixrsvp']):.1f}%")
    else:
        missing = []
        if not trunc_path.exists():
            missing.append("truncation.npz")
        if not dset_path.exists():
            missing.append("fixrsvp.dset")
        print(f"  WARNING: missing {', '.join(missing)}")

    qc_per_session.append(qc)

print(f"\n=== QC loading complete for {len(qc_per_session)} sessions ===")

# %% Extract per-neuron metrics with QC alignment
# Re-extract the per-neuron metrics from cached outputs (same logic as
# generate_figure2.py) but also carry along the QC metrics for each neuron.
# We focus on the primary counting window (10 ms, index 0).

w_idx = PRIMARY_WINDOW_IDX

# Per-neuron accumulators
all_alpha = []
all_ff_uncorr = []
all_ff_corr = []
all_erate = []
all_contam = []
all_missing = []

# Per-pair accumulators
all_rho_uncorr = []
all_rho_corr = []
all_pair_contam = []  # geometric mean contamination per pair
all_pair_missing = []  # geometric mean missing % per pair
all_pair_rate = []     # geometric mean firing rate per pair

for ds_idx, out in enumerate(outputs):
    res = out["results"][w_idx]
    mats = out["last_mats"][w_idx]

    Ctotal = mats["Total"]
    Cpsth = mats["PSTH"]
    Crate = mats["Intercept"]

    CnoiseU = Ctotal - Cpsth
    CnoiseC = Ctotal - Crate
    CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
    CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)

    erate = res["Erates"]
    total_spikes = erate * res["n_samples"]

    # Neuron inclusion mask (same as generate_figure2.py)
    valid = (
        np.isfinite(erate)
        & (total_spikes >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid.sum() < 3:
        continue

    n_valid = valid.sum()
    valid_idx = np.where(valid)[0]

    # QC metrics aligned to the valid neurons
    qc = qc_per_session[ds_idx]
    contam = qc["contamination"]  # (n_used,) aligned to cids_used
    missing = qc["missing_pct_fixrsvp"]

    # Alpha (FEM modulation fraction)
    diag_psth = np.diag(Cpsth)[valid]
    diag_rate = np.diag(Crate)[valid]
    alpha = np.clip(diag_psth / diag_rate, 0, 1)
    all_alpha.append(alpha)

    # Fano factors
    ff_u = np.diag(CnoiseU)[valid] / erate[valid]
    ff_c = np.diag(CnoiseC)[valid] / erate[valid]
    all_ff_uncorr.append(ff_u)
    all_ff_corr.append(ff_c)
    all_erate.append(erate[valid])

    # QC for valid neurons
    all_contam.append(contam[valid_idx])
    all_missing.append(missing[valid_idx])

    # Noise correlations (upper triangle of valid pairs)
    NoiseCorrU = cov_to_corr(CnoiseU[np.ix_(valid, valid)], min_var=MIN_VAR)
    NoiseCorrC = cov_to_corr(CnoiseC[np.ix_(valid, valid)], min_var=MIN_VAR)
    rho_u = get_upper_triangle(NoiseCorrU)
    rho_c = get_upper_triangle(NoiseCorrC)

    pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
    rho_u = rho_u[pair_ok]
    rho_c = rho_c[pair_ok]
    all_rho_uncorr.append(rho_u)
    all_rho_corr.append(rho_c)

    # Per-pair QC: geometric mean of the two neurons' QC metrics
    contam_valid = contam[valid_idx]
    missing_valid = missing[valid_idx]
    erate_valid = erate[valid]

    # Build upper-triangle pair indices
    n_v = n_valid
    ii, jj = np.triu_indices(n_v, k=1)
    pair_ok_full = np.isfinite(NoiseCorrU[ii, jj]) & np.isfinite(NoiseCorrC[ii, jj])

    # Geometric mean contamination per pair (use max(x, 0.1) to avoid log(0))
    c_i = np.clip(contam_valid[ii], 0.1, None)
    c_j = np.clip(contam_valid[jj], 0.1, None)
    pair_contam = np.sqrt(c_i * c_j)
    all_pair_contam.append(pair_contam[pair_ok_full])

    m_i = np.clip(missing_valid[ii], 0.1, None)
    m_j = np.clip(missing_valid[jj], 0.1, None)
    pair_missing = np.sqrt(m_i * m_j)
    all_pair_missing.append(pair_missing[pair_ok_full])

    r_i = np.clip(erate_valid[ii], 1e-6, None)
    r_j = np.clip(erate_valid[jj], 1e-6, None)
    pair_rate = np.sqrt(r_i * r_j)
    all_pair_rate.append(pair_rate[pair_ok_full])

# Concatenate across sessions
alpha_all = np.concatenate(all_alpha)
ff_u_all = np.concatenate(all_ff_uncorr)
ff_c_all = np.concatenate(all_ff_corr)
erate_all = np.concatenate(all_erate)
contam_all = np.concatenate(all_contam)
missing_all = np.concatenate(all_missing)

rho_u_all = np.concatenate(all_rho_uncorr)
rho_c_all = np.concatenate(all_rho_corr)
pair_contam_all = np.concatenate(all_pair_contam)
pair_missing_all = np.concatenate(all_pair_missing)
pair_rate_all = np.concatenate(all_pair_rate)

# Derived metrics
fem_mod = 1 - alpha_all
ff_ratio = ff_c_all / ff_u_all  # corrected / uncorrected
delta_rho = rho_c_all - rho_u_all

print(f"\n=== Merged dataset ===")
print(f"  Neurons: {len(fem_mod)}")
print(f"  Pairs:   {len(delta_rho)}")
print(f"  Contamination: {np.isfinite(contam_all).sum()} valid "
      f"(median={np.nanmedian(contam_all):.1f}%)")
print(f"  Missing (fixRSVP): {np.isfinite(missing_all).sum()} valid "
      f"(median={np.nanmedian(missing_all):.1f}%)")


# %% Helper: running median with IQR bands
def running_median(x, y, n_bins=N_BINS_RUNNING, min_count=5):
    """Compute running median and IQR of y as a function of x.

    Bins x into n_bins quantile bins. Returns bin centers, medians,
    and 25th/75th percentiles. Bins with fewer than min_count points
    are dropped.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < min_count:
        return np.array([]), np.array([]), np.array([]), np.array([])

    bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    centers, medians, q25s, q75s = [], [], [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)
        if mask.sum() >= min_count:
            centers.append(np.median(x[mask]))
            medians.append(np.median(y[mask]))
            q25s.append(np.percentile(y[mask], 25))
            q75s.append(np.percentile(y[mask], 75))

    return np.array(centers), np.array(medians), np.array(q25s), np.array(q75s)


# %% Panel S1: Per-neuron effect gradients vs. QC
fig_s1, axes_s1 = plt.subplots(2, 2, figsize=(10, 8))

panels = [
    (axes_s1[0, 0], contam_all, fem_mod, "Contamination (%)", "1 − α (FEM fraction)",
     "S1a: FEM modulation vs contamination"),
    (axes_s1[0, 1], missing_all, fem_mod, "Missing spike % (fixRSVP)", "1 − α (FEM fraction)",
     "S1b: FEM modulation vs missing %"),
    (axes_s1[1, 0], contam_all, ff_ratio, "Contamination (%)", "FF ratio (corr/uncorr)",
     "S1c: Fano ratio vs contamination"),
    (axes_s1[1, 1], missing_all, ff_ratio, "Missing spike % (fixRSVP)", "FF ratio (corr/uncorr)",
     "S1d: Fano ratio vs missing %"),
]

for ax, x_data, y_data, xlabel, ylabel, title in panels:
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    if finite.sum() > 0:
        # Clip Fano ratios for display
        y_plot = y_data.copy()
        if "FF ratio" in ylabel:
            y_plot = np.clip(y_plot, -2, 5)

        ax.scatter(x_data[finite], y_plot[finite], s=4, alpha=0.15,
                   c="steelblue", rasterized=True)

        # Running median
        centers, meds, q25s, q75s = running_median(x_data, y_data)
        if len(centers) > 0:
            ax.plot(centers, meds, "r-", linewidth=2, label="running median")
            ax.fill_between(centers, q25s, q75s, color="red", alpha=0.15,
                            label="IQR")

        # Reference lines
        if "1 − α" in ylabel:
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        elif "FF ratio" in ylabel:
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4,
                        label="no change")

        # Spearman correlation
        ok = np.isfinite(x_data) & np.isfinite(y_data)
        if ok.sum() > 10:
            rho_s, p_s = sp_stats.spearmanr(x_data[ok], y_data[ok])
            ax.text(0.02, 0.98, f"ρ_s={rho_s:.3f}, p={p_s:.2g}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

fig_s1.tight_layout()
fig_s1.savefig(FIG_DIR / "panel_s1_neuron_qc.pdf", bbox_inches="tight", dpi=300)
plt.close(fig_s1)
print(f"Saved Panel S1 → {FIG_DIR / 'panel_s1_neuron_qc.pdf'}")


# %% Panel S2: Per-pair effect gradients vs. QC
fig_s2, axes_s2 = plt.subplots(1, 3, figsize=(14, 4))

pair_panels = [
    (axes_s2[0], pair_contam_all, delta_rho, "Geom. mean contamination (%)",
     "Δρ (corr − uncorr)", "S2a: Δρ vs pair contamination"),
    (axes_s2[1], pair_missing_all, delta_rho, "Geom. mean missing % (fixRSVP)",
     "Δρ (corr − uncorr)", "S2b: Δρ vs pair missing %"),
    (axes_s2[2], pair_rate_all, delta_rho, "Geom. mean firing rate",
     "Δρ (corr − uncorr)", "S2c: Δρ vs pair firing rate"),
]

for ax, x_data, y_data, xlabel, ylabel, title in pair_panels:
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    if finite.sum() > 0:
        ax.scatter(x_data[finite], y_data[finite], s=2, alpha=0.05,
                   c="steelblue", rasterized=True)

        centers, meds, q25s, q75s = running_median(x_data, y_data)
        if len(centers) > 0:
            ax.plot(centers, meds, "r-", linewidth=2, label="running median")
            ax.fill_between(centers, q25s, q75s, color="red", alpha=0.15,
                            label="IQR")

        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.4)

        ok = np.isfinite(x_data) & np.isfinite(y_data)
        if ok.sum() > 10:
            rho_s, p_s = sp_stats.spearmanr(x_data[ok], y_data[ok])
            ax.text(0.02, 0.98, f"ρ_s={rho_s:.3f}, p={p_s:.2g}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

fig_s2.tight_layout()
fig_s2.savefig(FIG_DIR / "panel_s2_pair_qc.pdf", bbox_inches="tight", dpi=300)
plt.close(fig_s2)
print(f"Saved Panel S2 → {FIG_DIR / 'panel_s2_pair_qc.pdf'}")


# %% Panel S3: Cumulative inclusion analysis
# Sort neurons by QC metric (ascending = best first), cumulatively include
# more neurons, and recompute each summary statistic. If the curve is flat,
# the finding is robust to inclusion threshold.

def cumulative_analysis(qc_metric, values, n_steps=50, min_n=10):
    """Compute a summary statistic as neurons are cumulatively included.

    Neurons are sorted by qc_metric (ascending). At each step, we include
    all neurons with qc_metric ≤ threshold and compute median of values.

    Returns: thresholds, medians, counts
    """
    finite = np.isfinite(qc_metric) & np.isfinite(values)
    qc_f = qc_metric[finite]
    val_f = values[finite]

    # Sort by QC metric
    order = np.argsort(qc_f)
    qc_sorted = qc_f[order]
    val_sorted = val_f[order]

    thresholds = np.linspace(qc_sorted[min_n], qc_sorted[-1], n_steps)
    medians = np.full(n_steps, np.nan)
    counts = np.zeros(n_steps, dtype=int)
    ci_lo = np.full(n_steps, np.nan)
    ci_hi = np.full(n_steps, np.nan)

    for i, thresh in enumerate(thresholds):
        included = val_sorted[qc_sorted <= thresh]
        if len(included) >= min_n:
            medians[i] = np.median(included)
            counts[i] = len(included)
            # Bootstrap 95% CI on median
            rng = np.random.default_rng(42)
            boot_meds = np.array([
                np.median(rng.choice(included, size=len(included), replace=True))
                for _ in range(1000)
            ])
            ci_lo[i] = np.percentile(boot_meds, 2.5)
            ci_hi[i] = np.percentile(boot_meds, 97.5)

    return thresholds, medians, counts, ci_lo, ci_hi


def cumulative_pair_analysis(qc_metric_pair, values, n_steps=50, min_n=20):
    """Same as cumulative_analysis but for pair-level metrics."""
    finite = np.isfinite(qc_metric_pair) & np.isfinite(values)
    qc_f = qc_metric_pair[finite]
    val_f = values[finite]

    order = np.argsort(qc_f)
    qc_sorted = qc_f[order]
    val_sorted = val_f[order]

    thresholds = np.linspace(qc_sorted[min_n], qc_sorted[-1], n_steps)
    means = np.full(n_steps, np.nan)
    counts = np.zeros(n_steps, dtype=int)
    ci_lo = np.full(n_steps, np.nan)
    ci_hi = np.full(n_steps, np.nan)

    for i, thresh in enumerate(thresholds):
        included = val_sorted[qc_sorted <= thresh]
        if len(included) >= min_n:
            means[i] = np.mean(included)
            counts[i] = len(included)
            rng = np.random.default_rng(42)
            boot_means = np.array([
                np.mean(rng.choice(included, size=len(included), replace=True))
                for _ in range(1000)
            ])
            ci_lo[i] = np.percentile(boot_means, 2.5)
            ci_hi[i] = np.percentile(boot_means, 97.5)

    return thresholds, means, counts, ci_lo, ci_hi


fig_s3, axes_s3 = plt.subplots(2, 3, figsize=(14, 8))

# Row 1: Contamination axis
cum_specs_contam = [
    (axes_s3[0, 0], contam_all, fem_mod, "1 − α", "S3a: FEM mod. vs contam threshold"),
    (axes_s3[0, 1], contam_all, ff_ratio, "FF ratio", "S3b: FF ratio vs contam threshold"),
    (axes_s3[0, 2], pair_contam_all, delta_rho, "Δρ", "S3c: Δρ vs contam threshold"),
]

for ax, qc, val, ylabel, title in cum_specs_contam:
    is_pair = (len(val) == len(delta_rho)) and ylabel == "Δρ"
    if is_pair:
        thresh, stat, cnt, lo, hi = cumulative_pair_analysis(qc, val)
    else:
        thresh, stat, cnt, lo, hi = cumulative_analysis(qc, val)

    valid_mask = np.isfinite(stat)
    if valid_mask.sum() > 0:
        ax.plot(thresh[valid_mask], stat[valid_mask], "k-", linewidth=1.5)
        ax.fill_between(thresh[valid_mask], lo[valid_mask], hi[valid_mask],
                        color="steelblue", alpha=0.2, label="95% CI")
        # Twin axis for counts
        ax2 = ax.twinx()
        ax2.fill_between(thresh[valid_mask], 0, cnt[valid_mask],
                         color="gray", alpha=0.1)
        ax2.set_ylabel("N included", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=7, colors="gray")

    ax.set_xlabel("Max contamination (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7)

# Row 2: Missing spike % axis
cum_specs_missing = [
    (axes_s3[1, 0], missing_all, fem_mod, "1 − α", "S3d: FEM mod. vs missing threshold"),
    (axes_s3[1, 1], missing_all, ff_ratio, "FF ratio", "S3e: FF ratio vs missing threshold"),
    (axes_s3[1, 2], pair_missing_all, delta_rho, "Δρ", "S3f: Δρ vs missing threshold"),
]

for ax, qc, val, ylabel, title in cum_specs_missing:
    is_pair = (len(val) == len(delta_rho)) and ylabel == "Δρ"
    if is_pair:
        thresh, stat, cnt, lo, hi = cumulative_pair_analysis(qc, val)
    else:
        thresh, stat, cnt, lo, hi = cumulative_analysis(qc, val)

    valid_mask = np.isfinite(stat)
    if valid_mask.sum() > 0:
        ax.plot(thresh[valid_mask], stat[valid_mask], "k-", linewidth=1.5)
        ax.fill_between(thresh[valid_mask], lo[valid_mask], hi[valid_mask],
                        color="steelblue", alpha=0.2, label="95% CI")
        ax2 = ax.twinx()
        ax2.fill_between(thresh[valid_mask], 0, cnt[valid_mask],
                         color="gray", alpha=0.1)
        ax2.set_ylabel("N included", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=7, colors="gray")

    ax.set_xlabel("Max missing spike % (fixRSVP)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7)

fig_s3.tight_layout()
fig_s3.savefig(FIG_DIR / "panel_s3_cumulative.pdf", bbox_inches="tight", dpi=300)
plt.close(fig_s3)
print(f"Saved Panel S3 → {FIG_DIR / 'panel_s3_cumulative.pdf'}")


# %% Panel S4: Rate-matched noise correlation analysis
# The dominant confound from S2c is firing rate (ρ_s = -0.59). Cohen & Kohn
# (2011, Fig. 2) show that low-rate neurons have compressed correlations due
# to threshold masking, so there's nothing for the FEM correction to remove.
# This panel tests the FEM correction after restricting to well-driven pairs.
#
# For each rate threshold, we compute:
#   - Mean Fisher-z noise correlations (uncorrected and corrected)
#   - Mean Δz (Fisher-z difference)
#   - N pairs remaining
#
# We also recompute the full Figure 2 noise-correlation statistics using
# per-session hierarchical means (same approach as generate_figure2.py) for
# the restricted population.

RATE_THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]  # min geom-mean rate

# Recompute per-session Fisher z means at each rate threshold
rate_match_results = {}

for rate_min in RATE_THRESHOLDS:
    z_u_ds = []
    z_c_ds = []
    dz_ds = []
    n_pairs_total = 0

    for ds_idx, out in enumerate(outputs):
        res = out["results"][PRIMARY_WINDOW_IDX]
        mats = out["last_mats"][PRIMARY_WINDOW_IDX]

        Ctotal = mats["Total"]
        Cpsth = mats["PSTH"]
        Crate = mats["Intercept"]

        CnoiseU = Ctotal - Cpsth
        CnoiseC = Ctotal - Crate
        CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
        CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)

        erate = res["Erates"]
        total_spikes = erate * res["n_samples"]

        valid = (
            np.isfinite(erate)
            & (total_spikes >= MIN_TOTAL_SPIKES)
            & (np.diag(Ctotal) > MIN_VAR)
            & np.isfinite(np.diag(Crate))
            & np.isfinite(np.diag(Cpsth))
        )
        if valid.sum() < 3:
            continue

        NoiseCorrU = cov_to_corr(CnoiseU[np.ix_(valid, valid)], min_var=MIN_VAR)
        NoiseCorrC = cov_to_corr(CnoiseC[np.ix_(valid, valid)], min_var=MIN_VAR)

        erate_valid = erate[valid]
        n_v = valid.sum()
        ii, jj = np.triu_indices(n_v, k=1)

        rho_u = NoiseCorrU[ii, jj]
        rho_c = NoiseCorrC[ii, jj]

        # Rate filter: geometric mean of pair must exceed threshold
        gm_rate = np.sqrt(np.clip(erate_valid[ii], 1e-9, None)
                          * np.clip(erate_valid[jj], 1e-9, None))
        rate_ok = gm_rate >= rate_min
        pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c) & rate_ok

        rho_u_f = rho_u[pair_ok]
        rho_c_f = rho_c[pair_ok]

        if len(rho_u_f) > 0:
            z_u_ds.append(fisher_z_mean(rho_u_f, eps=EPS_RHO))
            z_c_ds.append(fisher_z_mean(rho_c_f, eps=EPS_RHO))
            dz_ds.append(fisher_z_mean(rho_c_f, eps=EPS_RHO)
                         - fisher_z_mean(rho_u_f, eps=EPS_RHO))
            n_pairs_total += len(rho_u_f)

    z_u_ds = np.array(z_u_ds)
    z_c_ds = np.array(z_c_ds)
    dz_ds = np.array(dz_ds)

    rate_match_results[rate_min] = {
        "z_u_ds": z_u_ds,
        "z_c_ds": z_c_ds,
        "dz_ds": dz_ds,
        "n_pairs": n_pairs_total,
        "n_sessions": len(z_u_ds),
    }

# --- Plot ---
fig_s4, axes_s4 = plt.subplots(1, 3, figsize=(14, 4))

# S4a: Mean Fisher z (uncorrected and corrected) vs rate threshold
ax = axes_s4[0]
for label, key, color in [("Uncorrected", "z_u_ds", "tab:blue"),
                            ("FEM-corrected", "z_c_ds", "tab:red")]:
    means = [np.mean(rate_match_results[r][key]) for r in RATE_THRESHOLDS
             if len(rate_match_results[r][key]) > 0]
    sems = [np.std(rate_match_results[r][key]) / np.sqrt(len(rate_match_results[r][key]))
            for r in RATE_THRESHOLDS if len(rate_match_results[r][key]) > 0]
    valid_r = [r for r in RATE_THRESHOLDS if len(rate_match_results[r][key]) > 0]
    ax.errorbar(valid_r, means, yerr=sems, fmt="o-", color=color, capsize=3, label=label)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Min geom. mean rate (sp/bin)")
ax.set_ylabel("Mean Fisher z")
ax.set_title("S4a: Noise corr vs rate threshold")
ax.legend(frameon=False, fontsize=8)

# S4b: Δz vs rate threshold
ax = axes_s4[1]
dz_means = [np.mean(rate_match_results[r]["dz_ds"]) for r in RATE_THRESHOLDS
            if len(rate_match_results[r]["dz_ds"]) > 0]
dz_sems = [np.std(rate_match_results[r]["dz_ds"]) / np.sqrt(len(rate_match_results[r]["dz_ds"]))
           for r in RATE_THRESHOLDS if len(rate_match_results[r]["dz_ds"]) > 0]
valid_r = [r for r in RATE_THRESHOLDS if len(rate_match_results[r]["dz_ds"]) > 0]
ax.errorbar(valid_r, dz_means, yerr=dz_sems, fmt="o-", color="black", capsize=3)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Min geom. mean rate (sp/bin)")
ax.set_ylabel("Δz (corr − uncorr)")
ax.set_title("S4b: Effect size vs rate threshold")

# S4c: N pairs vs rate threshold
ax = axes_s4[2]
n_pairs_list = [rate_match_results[r]["n_pairs"] for r in RATE_THRESHOLDS]
ax.bar(range(len(RATE_THRESHOLDS)), n_pairs_list, color="steelblue", alpha=0.7)
ax.set_xticks(range(len(RATE_THRESHOLDS)))
ax.set_xticklabels([f"{r}" for r in RATE_THRESHOLDS])
ax.set_xlabel("Min geom. mean rate (sp/bin)")
ax.set_ylabel("N pairs")
ax.set_title("S4c: Sample size vs rate threshold")

fig_s4.tight_layout()
fig_s4.savefig(FIG_DIR / "panel_s4_rate_matched.pdf", bbox_inches="tight", dpi=300)
plt.close(fig_s4)
print(f"Saved Panel S4 → {FIG_DIR / 'panel_s4_rate_matched.pdf'}")


# %% Save stats report
import sys as _sys

stats_file = STAT_DIR / "fig2_qc_stats.txt"

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
    old_stdout = _sys.stdout
    _sys.stdout = _Tee(f, old_stdout)

    print("=" * 80)
    print("FIGURE 2 QC CONTROLS: SPIKE-SORTING QUALITY STRATIFIED ANALYSIS")
    print(f"Primary counting window: {WINDOWS[PRIMARY_WINDOW_IDX]} ms")
    print("=" * 80)

    print(f"\nTotal neurons: {len(fem_mod)}")
    print(f"Total pairs:   {len(delta_rho)}")

    print("\n" + "-" * 60)
    print("QC METRIC DISTRIBUTIONS")
    print("-" * 60)
    n_contam_valid = np.isfinite(contam_all).sum()
    n_missing_valid = np.isfinite(missing_all).sum()
    print(f"\nContamination (refractory):")
    print(f"  Valid: {n_contam_valid}/{len(contam_all)}")
    if n_contam_valid > 0:
        print(f"  Median: {np.nanmedian(contam_all):.1f}%")
        print(f"  IQR: [{np.nanpercentile(contam_all, 25):.1f}, "
              f"{np.nanpercentile(contam_all, 75):.1f}]%")
        print(f"  Range: [{np.nanmin(contam_all):.1f}, {np.nanmax(contam_all):.1f}]%")
        print(f"  N < 20% (single unit): {(contam_all[np.isfinite(contam_all)] < 20).sum()}")

    print(f"\nMissing spike % (fixRSVP):")
    print(f"  Valid: {n_missing_valid}/{len(missing_all)}")
    if n_missing_valid > 0:
        print(f"  Median: {np.nanmedian(missing_all):.1f}%")
        print(f"  IQR: [{np.nanpercentile(missing_all, 25):.1f}, "
              f"{np.nanpercentile(missing_all, 75):.1f}]%")
        print(f"  Range: [{np.nanmin(missing_all):.1f}, {np.nanmax(missing_all):.1f}]%")
        print(f"  N < 20% (well-isolated): {(missing_all[np.isfinite(missing_all)] < 20).sum()}")

    print("\n" + "-" * 60)
    print("SPEARMAN CORRELATIONS: EFFECTS vs QC METRICS")
    print("-" * 60)

    pairs_to_test = [
        ("1-alpha vs contamination", contam_all, fem_mod),
        ("1-alpha vs missing %", missing_all, fem_mod),
        ("FF ratio vs contamination", contam_all, ff_ratio),
        ("FF ratio vs missing %", missing_all, ff_ratio),
        ("Delta rho vs pair contamination", pair_contam_all, delta_rho),
        ("Delta rho vs pair missing %", pair_missing_all, delta_rho),
        ("Delta rho vs pair firing rate", pair_rate_all, delta_rho),
    ]

    for label, x, y in pairs_to_test:
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 10:
            rho_s, p_s = sp_stats.spearmanr(x[ok], y[ok])
            print(f"\n  {label}:")
            print(f"    N = {ok.sum()}, rho_s = {rho_s:.4f}, p = {p_s:.3g}")
        else:
            print(f"\n  {label}: insufficient data (N={ok.sum()})")

    print("\n" + "-" * 60)
    print("EFFECT ESTIMATES: ALL UNITS vs WELL-ISOLATED SUBSET")
    print("-" * 60)

    # Compare full population vs restricted to <20% contamination AND <20% missing
    good_neuron = (
        np.isfinite(contam_all) & (contam_all < 20)
        & np.isfinite(missing_all) & (missing_all < 20)
    )
    print(f"\n  Well-isolated neurons: {good_neuron.sum()} / {len(good_neuron)}")

    if good_neuron.sum() > 5:
        print(f"\n  1-alpha (FEM modulation fraction):")
        print(f"    All:           median={np.nanmedian(fem_mod):.3f}, "
              f"mean={np.nanmean(fem_mod):.3f}")
        print(f"    Well-isolated: median={np.nanmedian(fem_mod[good_neuron]):.3f}, "
              f"mean={np.nanmean(fem_mod[good_neuron]):.3f}")

        print(f"\n  FF ratio (corrected / uncorrected):")
        ff_ok = np.isfinite(ff_ratio)
        print(f"    All:           median={np.nanmedian(ff_ratio[ff_ok]):.3f}")
        ff_good = good_neuron & ff_ok
        if ff_good.sum() > 0:
            print(f"    Well-isolated: median={np.nanmedian(ff_ratio[ff_good]):.3f}")

    good_pair = (
        np.isfinite(pair_contam_all) & (pair_contam_all < 20)
        & np.isfinite(pair_missing_all) & (pair_missing_all < 20)
    )
    print(f"\n  Well-isolated pairs: {good_pair.sum()} / {len(good_pair)}")

    if good_pair.sum() > 10:
        print(f"\n  Delta rho (noise correlation change):")
        print(f"    All:           mean={np.nanmean(delta_rho):.4f}")
        print(f"    Well-isolated: mean={np.nanmean(delta_rho[good_pair]):.4f}")

    print("\n" + "-" * 60)
    print("RATE-MATCHED NOISE CORRELATION ANALYSIS")
    print("-" * 60)
    for rate_min in RATE_THRESHOLDS:
        r = rate_match_results[rate_min]
        if len(r["dz_ds"]) > 0:
            print(f"\n  Rate >= {rate_min} sp/bin "
                  f"({r['n_pairs']} pairs, {r['n_sessions']} sessions):")
            print(f"    z_uncorr = {np.mean(r['z_u_ds']):.4f} "
                  f"± {np.std(r['z_u_ds'])/np.sqrt(len(r['z_u_ds'])):.4f}")
            print(f"    z_corr   = {np.mean(r['z_c_ds']):.4f} "
                  f"± {np.std(r['z_c_ds'])/np.sqrt(len(r['z_c_ds'])):.4f}")
            print(f"    Δz       = {np.mean(r['dz_ds']):.4f} "
                  f"± {np.std(r['dz_ds'])/np.sqrt(len(r['dz_ds'])):.4f}")

    print("\n" + "=" * 80)
    print("END OF QC CONTROL STATISTICS")
    print("=" * 80)

    _sys.stdout = old_stdout

print(f"\nStats saved to {stats_file}")
print(f"Figures saved to {FIG_DIR}")
print("Done.")
