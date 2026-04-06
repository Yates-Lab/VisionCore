# %% Imports and configuration
"""
Pairwise noise correlation as a function of inter-unit distance on the probe.

Loads the cached fig2 covariance decomposition, determines each unit's depth
from its waveform peak-energy channel, and plots corrected noise correlation
(Crate) vs pairwise distance along the probe for Allen and Logan.
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import dill
from pathlib import Path

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR
from VisionCore.covariance import cov_to_corr, get_upper_triangle

# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Must match generate_figure2.py
MIN_TOTAL_SPIKES = 500
MIN_VAR = 0
EPS_RHO = 1e-3
PRIMARY_WINDOW_IDX = 0  # smallest window (2 bins = ~8.3 ms)

PROCESSED_DIR = Path("/mnt/ssd/YatesMarmoV1/processed")
SUBJECTS = ["Allen", "Logan"]

FIG_DIR = FIGURES_DIR / "fig2_pairwise_distance"
STAT_DIR = STATS_DIR / "fig2_pairwise_distance"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Detect interactive IPython session
try:
    get_ipython()  # type: ignore[name-defined]
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False


def show_or_close(fig):
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


# %% Load cached fig2 decomposition and dataset configs
cache_path = CACHE_DIR / "fig2_decomposition.pkl"
print(f"Loading cached decomposition from {cache_path}")
with open(cache_path, "rb") as f:
    session_results = dill.load(f)

# Load dataset configs to get per-session cids
sys.path.insert(0, str(VISIONCORE_ROOT))
from models.config_loader import load_dataset_configs

DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))

# Build lookup: session_name -> cids from config
config_cids = {}
for cfg in dataset_configs:
    sess_name = cfg["session"]
    if cfg.get("cids") is not None:
        config_cids[sess_name] = np.array(cfg["cids"])

print(f"Loaded configs for {len(config_cids)} sessions with cids")

WINDOWS_MS = [r["window_ms"] for r in session_results[0]["results"]]
print(f"Windows (ms): {[f'{w:.1f}' for w in WINDOWS_MS]}")
print(f"Using primary window index {PRIMARY_WINDOW_IDX} ({WINDOWS_MS[PRIMARY_WINDOW_IDX]:.1f} ms)")


# %% Compute unit depths from waveform peak-energy channel
from DataYatesV1.utils.io import get_session


def get_unit_depths(sess_name, cids_used):
    """
    Get probe depth for each unit by finding the peak-energy channel
    in its mean waveform.

    Parameters
    ----------
    sess_name : str
        Session name (e.g., "Allen_2022-02-16")
    cids_used : ndarray of int
        Cluster IDs of the units to get depths for.

    Returns
    -------
    depths : ndarray of float
        Depth in micrometers for each unit, or NaN if unavailable.
    """
    subject, date = sess_name.split("_", 1)
    sess = get_session(subject, date)
    depths = np.full(len(cids_used), np.nan)

    # Load waveforms
    wave_path = sess.sess_dir / "qc" / "waveforms" / "waveforms.npz"
    if not wave_path.exists():
        print(f"  WARNING: waveforms.npz not found for {sess_name}")
        return depths

    waves = np.load(wave_path)
    waveforms = waves["waveforms"]  # (n_all_clusters, n_time_samples, n_channels)

    # Load probe geometry
    try:
        probe_geom = np.array(sess.ephys_metadata["probe_geometry_um"])  # (n_channels, 2)
    except Exception as e:
        print(f"  WARNING: Could not load probe geometry for {sess_name}: {e}")
        return depths

    for i, cid in enumerate(cids_used):
        if cid >= waveforms.shape[0]:
            continue
        wf = waveforms[cid]  # (n_time_samples, n_channels)
        # Energy per channel = sum of squared voltage across time
        energy = np.sum(wf ** 2, axis=0)  # (n_channels,)
        peak_ch = np.argmax(energy)
        if peak_ch < probe_geom.shape[0]:
            depths[i] = probe_geom[peak_ch, 1]  # y-coordinate = depth

    return depths


# %% Extract noise correlations with pairwise distances
w_idx = PRIMARY_WINDOW_IDX

# Per-pair accumulators
all_rho_corr = []
all_distances = []
all_pair_subject = []

for ds_idx, sr in enumerate(session_results):
    sess_name = sr["session"]
    subject = sr["subject"]
    if subject not in SUBJECTS:
        continue

    if w_idx >= len(sr["results"]):
        continue
    res = sr["results"][w_idx]
    mats = sr["mats"][w_idx]

    Ctotal = mats["Total"]
    Cpsth = mats["PSTH"]
    Crate = mats["Intercept"]

    CnoiseC = Ctotal - Crate
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

    # --- Get cluster IDs for valid neurons ---
    neuron_mask = sr["neuron_mask"]  # indices into the config-cids array
    if sess_name not in config_cids:
        print(f"  WARNING: No config cids for {sess_name}, skipping")
        continue

    all_cids = config_cids[sess_name]  # cluster IDs from config (after filtering)

    # neuron_mask indexes into the fixrsvp dataset's neuron axis,
    # which after prepare_data is sliced to config cids
    cids_after_align = all_cids[neuron_mask]

    # valid mask further filters within the aligned neurons
    valid_idx = np.where(valid)[0]
    cids_used = cids_after_align[valid_idx]

    # --- Get depths ---
    depths = get_unit_depths(sess_name, cids_used)

    n_valid = len(cids_used)
    print(f"  {sess_name} ({subject}): {n_valid} valid neurons, "
          f"depth range [{np.nanmin(depths):.0f}, {np.nanmax(depths):.0f}] um")

    # --- Pairwise noise correlations (corrected, Crate) ---
    NoiseCorrC = cov_to_corr(CnoiseC[np.ix_(valid, valid)], min_var=MIN_VAR)
    rho_c = get_upper_triangle(NoiseCorrC)

    # --- Pairwise distances ---
    # Upper triangle indices match get_upper_triangle ordering
    n = n_valid
    i_idx, j_idx = np.triu_indices(n, k=1)
    pairwise_dist = np.abs(depths[i_idx] - depths[j_idx])

    # Filter to pairs where both are finite
    pair_ok = np.isfinite(rho_c) & np.isfinite(pairwise_dist)
    rho_c = rho_c[pair_ok]
    pairwise_dist = pairwise_dist[pair_ok]

    all_rho_corr.append(rho_c)
    all_distances.append(pairwise_dist)
    all_pair_subject.extend([subject] * len(rho_c))

all_rho_corr = np.concatenate(all_rho_corr)
all_distances = np.concatenate(all_distances)
all_pair_subject = np.array(all_pair_subject)

print(f"\nTotal pairs: {len(all_rho_corr)}")
for subj in SUBJECTS:
    mask = all_pair_subject == subj
    print(f"  {subj}: {mask.sum()} pairs")


# %% Plot: noise correlation vs distance
def running_stat(x, y, n_bins=10, stat="median"):
    """Compute running statistic in quantile bins."""
    edges = np.nanpercentile(x, np.linspace(0, 100, n_bins + 1))
    centers = []
    values = []
    ci_lo = []
    ci_hi = []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if i == n_bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        if mask.sum() < 5:
            continue
        centers.append(np.nanmedian(x[mask]))
        if stat == "median":
            values.append(np.nanmedian(y[mask]))
        else:
            values.append(np.nanmean(y[mask]))
        q25, q75 = np.nanpercentile(y[mask], [25, 75])
        ci_lo.append(q25)
        ci_hi.append(q75)
    return np.array(centers), np.array(values), np.array(ci_lo), np.array(ci_hi)


subject_colors = {"Allen": "#1f77b4", "Logan": "#ff7f0e"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"wspace": 0.35})

# --- Panel A: Scatter + running median per subject ---
for ax_idx, subj in enumerate(SUBJECTS):
    ax = axes[ax_idx]
    mask = all_pair_subject == subj
    dist = all_distances[mask]
    rho = all_rho_corr[mask]
    color = subject_colors[subj]

    ax.scatter(dist, rho, s=1, alpha=0.05, color=color, rasterized=True)

    # Running median
    centers, medians, q25, q75 = running_stat(dist, rho, n_bins=10)
    ax.plot(centers, medians, "o-", color=color, lw=2, markersize=5, zorder=5)
    ax.fill_between(centers, q25, q75, alpha=0.2, color=color, zorder=4)

    ax.axhline(0, color="gray", ls="--", lw=0.8, zorder=1)
    ax.set_xlabel("Pairwise distance ($\\mu$m)")
    ax.set_ylabel("Corrected noise corr. ($\\rho_C$)")
    ax.set_title(subj)

    # Spearman correlation
    ok = np.isfinite(dist) & np.isfinite(rho)
    if ok.sum() > 10:
        r_s, p_s = sp_stats.spearmanr(dist[ok], rho[ok])
        ax.text(0.95, 0.95, f"$r_s$ = {r_s:.3f}\np = {p_s:.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    n_pairs = mask.sum()
    ax.text(0.05, 0.95, f"n = {n_pairs:,}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)

# --- Panel C: Pooled ---
ax = axes[2]
for subj in SUBJECTS:
    mask = all_pair_subject == subj
    dist = all_distances[mask]
    rho = all_rho_corr[mask]
    color = subject_colors[subj]
    centers, medians, q25, q75 = running_stat(dist, rho, n_bins=10)
    ax.plot(centers, medians, "o-", color=color, lw=2, markersize=5, label=subj, zorder=5)
    ax.fill_between(centers, q25, q75, alpha=0.15, color=color, zorder=4)

ax.axhline(0, color="gray", ls="--", lw=0.8, zorder=1)
ax.set_xlabel("Pairwise distance ($\\mu$m)")
ax.set_ylabel("Corrected noise corr. ($\\rho_C$)")
ax.set_title("Both subjects")
ax.legend(fontsize=9, loc="upper right")

# Pooled Spearman
ok = np.isfinite(all_distances) & np.isfinite(all_rho_corr)
r_s, p_s = sp_stats.spearmanr(all_distances[ok], all_rho_corr[ok])
ax.text(0.05, 0.95, f"Pooled $r_s$ = {r_s:.3f}\np = {p_s:.2e}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

fig.suptitle("Corrected noise correlation vs. inter-unit distance on probe",
             fontsize=12, y=1.02)
fig.tight_layout()

fig.savefig(FIG_DIR / "noise_corr_vs_distance.pdf", bbox_inches="tight", dpi=200)
fig.savefig(FIG_DIR / "noise_corr_vs_distance.png", bbox_inches="tight", dpi=200)
print(f"\nFigure saved to {FIG_DIR / 'noise_corr_vs_distance.pdf'}")

show_or_close(fig)


# %% Print summary statistics
print("\n=== Summary Statistics ===")
stat_lines = []
for subj in SUBJECTS + ["Pooled"]:
    if subj == "Pooled":
        mask = np.ones(len(all_rho_corr), dtype=bool)
    else:
        mask = all_pair_subject == subj

    dist = all_distances[mask]
    rho = all_rho_corr[mask]
    ok = np.isfinite(dist) & np.isfinite(rho)
    r_s, p_s = sp_stats.spearmanr(dist[ok], rho[ok])
    r_p, p_p = sp_stats.pearsonr(dist[ok], rho[ok])

    line = (f"{subj:8s}: n_pairs={ok.sum():6d}, "
            f"mean_rho={np.mean(rho[ok]):+.4f}, "
            f"Spearman r={r_s:+.4f} (p={p_s:.2e}), "
            f"Pearson r={r_p:+.4f} (p={p_p:.2e})")
    print(line)
    stat_lines.append(line)

# Save stats
with open(STAT_DIR / "noise_corr_vs_distance.txt", "w") as f:
    f.write("Corrected noise correlation vs inter-unit distance\n")
    f.write(f"Primary window: {WINDOWS_MS[PRIMARY_WINDOW_IDX]:.1f} ms\n\n")
    for line in stat_lines:
        f.write(line + "\n")

print(f"\nStats saved to {STAT_DIR / 'noise_corr_vs_distance.txt'}")
