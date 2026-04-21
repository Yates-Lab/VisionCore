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
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
import dill
from pathlib import Path

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR
from VisionCore.covariance import cov_to_corr, get_upper_triangle, project_to_psd

# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# Must match generate_figure2.py
MIN_TOTAL_SPIKES = 200
MIN_VAR = 0
EPS_RHO = 1e-3
PRIMARY_WINDOW_IDX = 0  # smallest window (2 bins = ~8.3 ms)

# Visualization
PITCH_UM = 35.0           # probe row pitch (depth is quantal in multiples of this)
DIST_BIN_STEP_UM = 2 * PITCH_UM  # coarser bins for clearer trend in line plot
RHO_RANGE = 0.2           # y-axis range for 2D histograms: [-RHO_RANGE, +RHO_RANGE]
N_RHO_BINS = 80
MAX_DIST_UM = 800.0       # cap pairwise distance; beyond this gets too noisy

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
    Get probe depth and shank index for each unit by finding the
    peak-energy channel in its mean waveform.

    Parameters
    ----------
    sess_name : str
        Session name (e.g., "Allen_2022-02-16")
    cids_used : ndarray of int
        Cluster IDs of the units to get depths for.

    Returns
    -------
    depths : ndarray of float
        Depth (y-coordinate) in micrometers for each unit, or NaN.
    shanks : ndarray of int
        Shank index for each unit, or -1 if unavailable.
    """
    subject, date = sess_name.split("_", 1)
    sess = get_session(subject, date)
    depths = np.full(len(cids_used), np.nan)
    shanks = np.full(len(cids_used), -1, dtype=int)

    # Load waveforms
    wave_path = sess.sess_dir / "qc" / "waveforms" / "waveforms.npz"
    if not wave_path.exists():
        print(f"  WARNING: waveforms.npz not found for {sess_name}")
        return depths, shanks

    waves = np.load(wave_path)
    waveforms = waves["waveforms"]  # (n_all_clusters, n_time_samples, n_channels)

    # Load probe geometry and shank membership
    try:
        probe_geom = np.array(sess.ephys_metadata["probe_geometry_um"])  # (n_channels, 2)
        shank_inds = [np.asarray(s, dtype=int)
                      for s in sess.ephys_metadata["shank_inds"]]
    except Exception as e:
        print(f"  WARNING: Could not load probe geometry for {sess_name}: {e}")
        return depths, shanks

    # Map channel -> shank index
    ch_to_shank = np.full(probe_geom.shape[0], -1, dtype=int)
    for s_idx, inds in enumerate(shank_inds):
        ch_to_shank[inds] = s_idx

    for i, cid in enumerate(cids_used):
        if cid >= waveforms.shape[0]:
            continue
        wf = waveforms[cid]  # (n_time_samples, n_channels)
        # Energy per channel = sum of squared voltage across time
        energy = np.sum(wf ** 2, axis=0)  # (n_channels,)
        peak_ch = np.argmax(energy)
        if peak_ch < probe_geom.shape[0]:
            depths[i] = probe_geom[peak_ch, 1]
            shanks[i] = ch_to_shank[peak_ch]

    return depths, shanks


# %% Extract noise correlations with pairwise distances
w_idx = PRIMARY_WINDOW_IDX

# Per-pair accumulators — within-shank (same shank, distance = |Δy|)
all_rho_corr = []
all_rho_uncorr = []
all_distances = []
all_pair_subject = []

# Per-pair accumulators — between-shank (different shanks, distance = Euclidean)
between_rho_corr = []
between_rho_uncorr = []
between_distances = []
between_pair_subject = []

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
    CnoiseU = Ctotal - Cpsth
    CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)

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

    # --- Get depths and shank assignments ---
    depths, shanks = get_unit_depths(sess_name, cids_used)

    n_valid = len(cids_used)
    n_per_shank = {s: int((shanks == s).sum()) for s in np.unique(shanks)}
    print(f"  {sess_name} ({subject}): {n_valid} valid neurons, "
          f"depth range [{np.nanmin(depths):.0f}, {np.nanmax(depths):.0f}] um, "
          f"shanks={n_per_shank}")

    # --- Pairwise noise correlations (corrected, Crate; uncorrected, PSTH) ---
    # Project to PSD to handle numerical negatives from split-half estimation
    # (matches generate_figure2.py)
    NoiseCorrC = cov_to_corr(
        project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    NoiseCorrU = cov_to_corr(
        project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    rho_c = get_upper_triangle(NoiseCorrC)
    rho_u = get_upper_triangle(NoiseCorrU)

    # --- Pairwise distances ---
    # Upper triangle indices match get_upper_triangle ordering.
    # Distance = |Δy| (depth difference) for both within- and between-shank
    # pairs. Between-shank pairs have an additional ~200 um lateral offset
    # that is not included in this distance.
    n = n_valid
    i_idx, j_idx = np.triu_indices(n, k=1)
    pairwise_dist = np.abs(depths[i_idx] - depths[j_idx])

    same_shank = (shanks[i_idx] == shanks[j_idx]) & (shanks[i_idx] >= 0)
    diff_shank = (shanks[i_idx] != shanks[j_idx]) & (shanks[i_idx] >= 0) & (shanks[j_idx] >= 0)

    finite_pair = (np.isfinite(rho_c) & np.isfinite(rho_u)
                   & np.isfinite(pairwise_dist))

    # Within-shank
    within_ok = finite_pair & same_shank
    all_rho_corr.append(rho_c[within_ok])
    all_rho_uncorr.append(rho_u[within_ok])
    all_distances.append(pairwise_dist[within_ok])
    all_pair_subject.extend([subject] * int(within_ok.sum()))

    # Between-shank
    between_ok = finite_pair & diff_shank
    between_rho_corr.append(rho_c[between_ok])
    between_rho_uncorr.append(rho_u[between_ok])
    between_distances.append(pairwise_dist[between_ok])
    between_pair_subject.extend([subject] * int(between_ok.sum()))

all_rho_corr = np.concatenate(all_rho_corr)
all_rho_uncorr = np.concatenate(all_rho_uncorr)
all_distances = np.concatenate(all_distances)
all_pair_subject = np.array(all_pair_subject)

between_rho_corr = np.concatenate(between_rho_corr) if between_rho_corr else np.array([])
between_rho_uncorr = np.concatenate(between_rho_uncorr) if between_rho_uncorr else np.array([])
between_distances = np.concatenate(between_distances) if between_distances else np.array([])
between_pair_subject = np.array(between_pair_subject)

print(f"\nWithin-shank pairs: {len(all_rho_corr)}")
for subj in SUBJECTS:
    mask = all_pair_subject == subj
    print(f"  {subj}: {mask.sum()} pairs")
print(f"Between-shank pairs: {len(between_rho_corr)}")
for subj in SUBJECTS:
    mask = between_pair_subject == subj
    print(f"  {subj}: {mask.sum()} pairs")


# %% Plot: noise correlation vs distance
def running_mean_by_bin(x, y, edges, min_count=5):
    """Mean of y per bin with 95% CI (±1.96·SEM). Returns (centers, means, ci_lo, ci_hi)."""
    centers, means, ci_lo, ci_hi = [], [], [], []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            m = (x >= edges[i]) & (x < edges[i + 1])
        yi = y[m]
        yi = yi[np.isfinite(yi)]
        if len(yi) < min_count:
            continue
        mean = float(np.mean(yi))
        sem = float(np.std(yi, ddof=1) / np.sqrt(len(yi)))
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(mean)
        ci_lo.append(mean - 1.96 * sem)
        ci_hi.append(mean + 1.96 * sem)
    return np.array(centers), np.array(means), np.array(ci_lo), np.array(ci_hi)


subject_colors = {"Allen": "#1f77b4", "Logan": "#ff7f0e"}


def _make_dist_edges(distances):
    """Uniform distance bin edges of width DIST_BIN_STEP_UM, capped at MAX_DIST_UM."""
    max_dist = np.nanmax(distances) if len(distances) else DIST_BIN_STEP_UM
    max_dist = min(max_dist, MAX_DIST_UM)
    n_bins = max(int(np.ceil(max_dist / DIST_BIN_STEP_UM)), 1)
    return np.arange(n_bins + 1) * DIST_BIN_STEP_UM


rho_edges = np.linspace(-RHO_RANGE, RHO_RANGE, N_RHO_BINS + 1)


def make_distance_figure(
    distances, pair_subject, rho_rows, out_stem, suptitle,
):
    """
    Figure: N rows of per-subject 2D histograms + full-width bottom line
    panel showing running-median vs distance for every (subject, row) pair.

    rho_rows: list of (row_label, rho_array, tag) tuples. `tag` is the short
              label used in the bottom-panel legend (e.g. "corr.").
    """
    if len(distances) == 0:
        print(f"  No pairs for {out_stem}, skipping figure")
        return

    keep = np.isfinite(distances) & (distances <= MAX_DIST_UM)
    distances = distances[keep]
    pair_subject = pair_subject[keep]
    rho_rows = [(label, rho_all[keep], tag) for (label, rho_all, tag) in rho_rows]

    dist_edges = _make_dist_edges(distances)
    n_rows = len(rho_rows)
    n_cols = len(SUBJECTS)

    fig = plt.figure(figsize=(5 * n_cols, 4.2 * n_rows + 4.5))
    height_ratios = [1] * n_rows + [1.1]
    gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                  height_ratios=height_ratios, hspace=0.45, wspace=0.3)

    for row, (label, rho_all, _tag) in enumerate(rho_rows):
        for col, subj in enumerate(SUBJECTS):
            ax = fig.add_subplot(gs[row, col])
            mask = pair_subject == subj
            dist = distances[mask]
            rho = rho_all[mask]
            color = subject_colors[subj]

            # Column-normalized 2D density
            H, _, _ = np.histogram2d(dist, rho, bins=[dist_edges, rho_edges])
            col_sums = H.sum(axis=1, keepdims=True)
            H_norm = np.where(col_sums > 0, H / np.maximum(col_sums, 1), 0.0)

            if (H_norm > 0).any():
                vmax = np.nanpercentile(H_norm[H_norm > 0], 99)
            else:
                vmax = 1.0
            pcm = ax.pcolormesh(
                dist_edges, rho_edges, H_norm.T,
                cmap="magma", shading="flat", vmin=0, vmax=vmax,
            )

            # Running mean overlay
            centers, means, _ci_lo, _ci_hi = running_mean_by_bin(dist, rho, dist_edges)
            ax.plot(centers, means, "o-", color=color, lw=1.8, markersize=3,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.4)

            ax.axhline(0, color="white", ls="--", lw=0.6, alpha=0.7, zorder=2)
            ax.set_ylim(-RHO_RANGE, RHO_RANGE)
            ax.set_xlim(0, dist_edges[-1])
            if row == n_rows - 1:
                ax.set_xlabel("Pairwise distance, |$\\Delta y$| ($\\mu$m)")
            if col == 0:
                ax.set_ylabel(label)
            ax.set_title(subj)

            ok = np.isfinite(dist) & np.isfinite(rho)
            if ok.sum() > 10:
                r_s, p_s = sp_stats.spearmanr(dist[ok], rho[ok])
                ax.text(0.97, 0.97, f"$r_s$ = {r_s:.3f}\np = {p_s:.1e}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", edgecolor="none", alpha=0.4))
            ax.text(0.03, 0.97, f"n = {int(ok.sum()):,}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color="white",
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.4))

            if col == n_cols - 1:
                cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
                cb.set_label("Density (per distance)", fontsize=8)
                cb.ax.tick_params(labelsize=7)

    # Bottom full-width line panel
    ax_bot = fig.add_subplot(gs[n_rows, :])
    pooled_summaries = []
    linestyles = ["-", "--", ":"]
    for subj in SUBJECTS:
        mask = pair_subject == subj
        dist = distances[mask]
        color = subject_colors[subj]
        for r_idx, (_label, rho_all, tag) in enumerate(rho_rows):
            ls = linestyles[r_idx % len(linestyles)]
            rho = rho_all[mask]
            centers, means, ci_lo, ci_hi = running_mean_by_bin(dist, rho, dist_edges)
            legend_label = f"{subj} ({tag})" if len(rho_rows) > 1 else subj
            ax_bot.plot(centers, means, ls, color=color, lw=2,
                        marker="o", markersize=4,
                        label=legend_label, zorder=5)
            if r_idx == 0:
                ax_bot.fill_between(centers, ci_lo, ci_hi, alpha=0.25,
                                    color=color, zorder=3, linewidth=0)

    for _label, rho_all, tag in rho_rows:
        ok = np.isfinite(distances) & np.isfinite(rho_all)
        r_s, p_s = sp_stats.spearmanr(distances[ok], rho_all[ok])
        pooled_summaries.append(
            f"Pooled $r_s$ ({tag}) = {r_s:+.3f}, p = {p_s:.1e}"
        )

    ax_bot.axhline(0, color="gray", ls=":", lw=0.8, zorder=1)
    ax_bot.set_xlabel("Pairwise distance, |$\\Delta y$| ($\\mu$m)")
    ax_bot.set_ylabel("Noise correlation (mean ± 95% CI)")
    ax_bot.set_title("Running mean vs distance — both subjects")
    ax_bot.set_xlim(0, dist_edges[-1])
    ax_bot.legend(fontsize=9, ncol=len(rho_rows), loc="upper right")
    ax_bot.text(
        0.02, 0.97, "\n".join(pooled_summaries),
        transform=ax_bot.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
    )

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(FIG_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
    print(f"Figure saved to {FIG_DIR / (out_stem + '.pdf')}")
    show_or_close(fig)


# Figure 1: within-shank, corrected
make_distance_figure(
    all_distances, all_pair_subject,
    rho_rows=[
        ("Corrected ($\\rho_C$)", all_rho_corr, "corr."),
    ],
    out_stem="noise_corr_vs_distance",
    suptitle="Noise correlation vs. within-shank pairwise distance",
)

# Figure 2: between-shank, corrected
make_distance_figure(
    between_distances, between_pair_subject,
    rho_rows=[
        ("Corrected ($\\rho_C$)", between_rho_corr, "corr."),
    ],
    out_stem="noise_corr_vs_distance_between_corrected",
    suptitle="Corrected noise correlation vs. between-shank pairwise distance "
             "(|$\\Delta y$| only; +~200 $\\mu$m lateral offset)",
)


# %% Print summary statistics
print("\n=== Summary Statistics ===")
stat_lines = []


def _summary_block(section_title, distances, pair_subject, pairs):
    """pairs: list of (tag, rho_array). Writes lines to stat_lines and prints."""
    stat_lines.append(section_title)
    print(section_title)
    if len(distances) == 0:
        stat_lines.append("  (no pairs)")
        print("  (no pairs)")
        return
    for tag, rho_all in pairs:
        stat_lines.append(f"  [{tag}]")
        print(stat_lines[-1])
        for subj in SUBJECTS + ["Pooled"]:
            if subj == "Pooled":
                mask = np.ones(len(rho_all), dtype=bool)
            else:
                mask = pair_subject == subj
            dist = distances[mask]
            rho = rho_all[mask]
            ok = np.isfinite(dist) & np.isfinite(rho)
            if ok.sum() < 3:
                line = f"    {subj:8s}: n_pairs={ok.sum():6d}  (insufficient)"
            else:
                r_s, p_s = sp_stats.spearmanr(dist[ok], rho[ok])
                r_p, p_p = sp_stats.pearsonr(dist[ok], rho[ok])
                line = (f"    {subj:8s}: n_pairs={ok.sum():6d}, "
                        f"mean_rho={np.mean(rho[ok]):+.4f}, "
                        f"Spearman r={r_s:+.4f} (p={p_s:.2e}), "
                        f"Pearson r={r_p:+.4f} (p={p_p:.2e})")
            print(line)
            stat_lines.append(line)


_summary_block(
    "=== Within-shank pairs ===",
    all_distances, all_pair_subject,
    [("corrected", all_rho_corr), ("uncorrected", all_rho_uncorr)],
)
_summary_block(
    "=== Between-shank pairs ===",
    between_distances, between_pair_subject,
    [("corrected", between_rho_corr), ("uncorrected", between_rho_uncorr)],
)

# Save stats
with open(STAT_DIR / "noise_corr_vs_distance.txt", "w") as f:
    f.write("Noise correlation vs pairwise distance (|Δy|)\n")
    f.write(f"Primary window: {WINDOWS_MS[PRIMARY_WINDOW_IDX]:.1f} ms\n\n")
    for line in stat_lines:
        f.write(line + "\n")

print(f"\nStats saved to {STAT_DIR / 'noise_corr_vs_distance.txt'}")
