# %% Imports and configuration
"""
Pairwise noise correlation as a function of tuning similarity (orientation,
spatial frequency, and both jointly).

Loads the cached fig2 covariance decomposition, runs the gratings analysis
per session to obtain per-unit peak SF / peak orientation / full SF x ori
tuning at the optimal temporal lag, and plots corrected noise correlation
(Crate) against tuning-difference and signal-correlation axes for Allen
and Logan pooled across shanks.
"""
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
import dill

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

# Gratings analysis parameters
GRAT_N_LAGS = 20
GRAT_MIN_SPIKES_SINE = 30  # for sine fitting in gratings_analysis
GRAT_DT = 1 / 120.0

# Pair inclusion thresholds — applied to BOTH members of a pair.
MIN_ORI_SNR = 0.3           # std(ori_tuning) / mean(ori_tuning)
MIN_SF_SNR = 0.3            # std(sf_tuning) / mean(sf_tuning)
MIN_GRAT_SPIKES = 50        # total gratings spikes at preferred SF x ori
MIN_TUNING_ENERGY = 1e-9    # reject flat / all-NaN tuning curves

# Visualization
N_ORI_BINS = 6              # Δθ bins spanning [0, 90] deg
N_SF_BINS = 6               # |Δlog2 SF| bins
N_SIGCORR_BINS = 10         # signal correlation bins
RHO_RANGE = 0.2
N_RHO_BINS = 80
MAX_DLOG_SF = 2.0           # cap |Δlog2 SF| (octaves) for plotting
MIN_SIGCORR = -0.5           # lower bound for signal-correlation axis

SUBJECTS = ["Allen", "Logan"]

CACHE_PATH = CACHE_DIR / "fig2_gratings_tuning.pkl"
FIG_DIR = FIGURES_DIR / "fig2_pairwise_tuning"
STAT_DIR = STATS_DIR / "fig2_pairwise_tuning"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Set True to regenerate the gratings tuning cache.
RECOMPUTE = False

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
fig2_cache = CACHE_DIR / "fig2_decomposition.pkl"
print(f"Loading fig2 decomposition from {fig2_cache}")
with open(fig2_cache, "rb") as f:
    session_results = dill.load(f)

sys.path.insert(0, str(VISIONCORE_ROOT))
from models.config_loader import load_dataset_configs

DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))

# Build lookups
config_cids = {}
config_by_sess = {}
for cfg in dataset_configs:
    sess_name = cfg["session"]
    config_by_sess[sess_name] = cfg
    if cfg.get("cids") is not None:
        config_cids[sess_name] = np.array(cfg["cids"])

print(f"Loaded configs for {len(config_cids)} sessions with cids")

WINDOWS_MS = [r["window_ms"] for r in session_results[0]["results"]]
print(f"Using primary window index {PRIMARY_WINDOW_IDX} "
      f"({WINDOWS_MS[PRIMARY_WINDOW_IDX]:.1f} ms)")


# %% Build per-session gratings tuning cache
def _run_gratings_for_session(sess_name):
    """
    Run gratings_analysis on the session's gratings block. Returns a dict
    with per-unit fields aligned to config_cids[sess_name], or None if the
    session has no gratings data.
    """
    from models.data.loading import prepare_data
    from eval.gratings_analysis import gratings_analysis

    cfg = config_by_sess[sess_name]
    # Ensure gratings is included; keep fixrsvp optional
    cfg = {**cfg}
    if "gratings" not in cfg["types"]:
        cfg["types"] = list(cfg["types"]) + ["gratings"]

    try:
        train_data, _val, cfg = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  prepare_data failed for {sess_name}: {e}")
        return None

    try:
        dset_idx = train_data.get_dataset_index("gratings")
    except (ValueError, KeyError):
        print(f"  No gratings dataset for {sess_name}")
        return None

    gdset = train_data.dsets[dset_idx]
    covs = gdset.covariates if hasattr(gdset, "covariates") else gdset

    # robs is already sliced to cfg['cids'] in prepare_data
    robs = np.asarray(covs["robs"])
    sf = np.asarray(covs["sf"])
    ori = np.asarray(covs["ori"])
    phases = np.asarray(covs["stim_phase"])
    dfs = np.asarray(covs["dfs"]) if "dfs" in covs else None
    if dfs is not None and dfs.ndim == 2 and dfs.shape[1] == 1:
        dfs = dfs.squeeze(-1)

    # gratings_analysis skips invalid ori/sf internally. We just pass dfs.
    result = gratings_analysis(
        robs=robs.astype(np.float64),
        sf=sf,
        ori=ori,
        phases=phases,
        dt=GRAT_DT,
        n_lags=GRAT_N_LAGS,
        min_spikes=GRAT_MIN_SPIKES_SINE,
        dfs=dfs,
    )

    # Full SF x ori tuning at each unit's peak lag (used for signal correlation).
    gratings_sta = result["gratings_sta"]  # (n_units, n_lags, n_sfs, n_oris)
    peak_lag_idx = result["peak_lag_idx"]
    n_units = gratings_sta.shape[0]
    tuning2d = np.full((n_units, gratings_sta.shape[2] * gratings_sta.shape[3]), np.nan)
    for u in range(n_units):
        lag = int(peak_lag_idx[u]) if np.isfinite(peak_lag_idx[u]) else -1
        if lag < 0 or lag >= gratings_sta.shape[1]:
            continue
        tuning2d[u] = gratings_sta[u, lag].reshape(-1)

    return {
        "cids": config_cids[sess_name],
        "peak_sf": result["peak_sf"],          # cyc/deg
        "peak_ori": result["peak_ori"],        # deg
        "sf_tuning": result["sf_tuning"],
        "ori_tuning": result["ori_tuning"],
        "tuning2d": tuning2d,
        "sf_snr": result["sf_snr"],
        "ori_snr": result["ori_snr"],
        "n_spikes_total": result["n_spikes_total"],
        "sfs": result["sfs"],
        "oris": result["oris"],
    }


if CACHE_PATH.exists() and not RECOMPUTE:
    print(f"Loading gratings tuning cache from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        gratings_cache = dill.load(f)
else:
    print(f"Computing gratings tuning cache -> {CACHE_PATH}")
    gratings_cache = {}
    for sr in session_results:
        sess_name = sr["session"]
        subject = sr["subject"]
        if subject not in SUBJECTS:
            continue
        if sess_name not in config_cids:
            print(f"  {sess_name}: no config cids, skipping")
            continue
        print(f"\n--- {sess_name} ({subject}) ---")
        res = _run_gratings_for_session(sess_name)
        if res is None:
            continue
        gratings_cache[sess_name] = res
    with open(CACHE_PATH, "wb") as f:
        dill.dump(gratings_cache, f)
    print(f"\nCached gratings tuning for {len(gratings_cache)} sessions")


# %% Accumulate pair-wise metrics across sessions
def _circular_delta_ori(a, b):
    """Unsigned difference between orientations (mod 180), wrapped to [0, 90]."""
    d = np.mod(a - b, 180.0)
    return np.minimum(d, 180.0 - d)


def _signal_corr_matrix(T):
    """
    Pearson correlation matrix between rows of T, ignoring units whose rows
    are all-NaN or zero-variance. Returns (n_units, n_units) with NaN for
    undefined pairs.
    """
    n = T.shape[0]
    out = np.full((n, n), np.nan)
    mu = np.nanmean(T, axis=1, keepdims=True)
    Tc = T - mu
    sd = np.sqrt(np.nansum(Tc ** 2, axis=1))
    denom = np.outer(sd, sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.nansum(Tc[:, None, :] * Tc[None, :, :], axis=-1)
        out = np.where(denom > 0, num / np.maximum(denom, 1e-30), np.nan)
    return out


w_idx = PRIMARY_WINDOW_IDX

rho_c_all = []
d_ori_all = []
d_logsf_all = []
sig_corr_all = []
pair_subject_all = []

per_session_unit_counts = []

for sr in session_results:
    sess_name = sr["session"]
    subject = sr["subject"]
    if subject not in SUBJECTS:
        continue
    if sess_name not in gratings_cache:
        continue
    if sess_name not in config_cids:
        continue
    if w_idx >= len(sr["results"]):
        continue

    gc = gratings_cache[sess_name]
    all_cids = config_cids[sess_name]
    if len(gc["cids"]) != len(all_cids):
        print(f"  {sess_name}: cid mismatch ({len(gc['cids'])} vs {len(all_cids)}), skip")
        continue

    res = sr["results"][w_idx]
    mats = sr["mats"][w_idx]

    Ctotal = mats["Total"]
    Cpsth = mats["PSTH"]
    Crate = mats["Intercept"]

    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

    erate = res["Erates"]
    total_spikes_psth = erate * res["n_samples"]
    valid_fig2 = (
        np.isfinite(erate)
        & (total_spikes_psth >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid_fig2.sum() < 3:
        continue

    neuron_mask = sr["neuron_mask"]        # indices into config_cids
    # Pull per-unit gratings fields into the fig2 neuron ordering
    peak_ori = gc["peak_ori"][neuron_mask]
    peak_sf = gc["peak_sf"][neuron_mask]
    ori_snr = gc["ori_snr"][neuron_mask]
    sf_snr = gc["sf_snr"][neuron_mask]
    n_grat = gc["n_spikes_total"][neuron_mask]
    tuning2d = gc["tuning2d"][neuron_mask]

    quality = (
        np.isfinite(peak_ori)
        & np.isfinite(peak_sf)
        & np.isfinite(ori_snr) & (ori_snr >= MIN_ORI_SNR)
        & np.isfinite(sf_snr) & (sf_snr >= MIN_SF_SNR)
        & np.isfinite(n_grat) & (n_grat >= MIN_GRAT_SPIKES)
        & (np.nansum(np.abs(tuning2d), axis=1) > MIN_TUNING_ENERGY)
    )

    use = valid_fig2 & quality
    n_use = int(use.sum())
    per_session_unit_counts.append((sess_name, subject, int(valid_fig2.sum()), n_use))
    if n_use < 3:
        continue

    # --- Corrected noise correlation matrix (project to PSD, like fig2) ---
    NoiseCorrC = cov_to_corr(
        project_to_psd(CnoiseC[np.ix_(use, use)]), min_var=MIN_VAR
    )
    rho_c = get_upper_triangle(NoiseCorrC)

    # --- Tuning difference metrics for the same upper-triangle pairs ---
    ori_use = peak_ori[use]
    sf_use = peak_sf[use]
    t2d_use = tuning2d[use]

    n = n_use
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ori = _circular_delta_ori(ori_use[i_idx], ori_use[j_idx])
    with np.errstate(invalid="ignore", divide="ignore"):
        d_logsf = np.abs(np.log2(sf_use[i_idx]) - np.log2(sf_use[j_idx]))

    sig_mat = _signal_corr_matrix(t2d_use)
    sig_corr = sig_mat[i_idx, j_idx]

    finite_pair = (
        np.isfinite(rho_c)
        & np.isfinite(d_ori)
        & np.isfinite(d_logsf)
        & np.isfinite(sig_corr)
    )

    rho_c_all.append(rho_c[finite_pair])
    d_ori_all.append(d_ori[finite_pair])
    d_logsf_all.append(d_logsf[finite_pair])
    sig_corr_all.append(sig_corr[finite_pair])
    pair_subject_all.extend([subject] * int(finite_pair.sum()))

rho_c_all = np.concatenate(rho_c_all) if rho_c_all else np.array([])
d_ori_all = np.concatenate(d_ori_all) if d_ori_all else np.array([])
d_logsf_all = np.concatenate(d_logsf_all) if d_logsf_all else np.array([])
sig_corr_all = np.concatenate(sig_corr_all) if sig_corr_all else np.array([])
pair_subject_all = np.array(pair_subject_all)

print("\nPer-session unit counts (fig2 valid → after gratings filter):")
for sess_name, subject, n_fig2, n_use in per_session_unit_counts:
    print(f"  {sess_name} ({subject}): {n_fig2} → {n_use}")

print(f"\nTotal pairs: {len(rho_c_all)}")
for subj in SUBJECTS:
    m = pair_subject_all == subj
    print(f"  {subj}: {int(m.sum())} pairs")


# %% Plot helpers
subject_colors = {"Allen": "#1f77b4", "Logan": "#ff7f0e"}
rho_edges = np.linspace(-RHO_RANGE, RHO_RANGE, N_RHO_BINS + 1)


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


def make_similarity_figure(
    x, pair_subject, rho, x_edges, xlabel, out_stem, suptitle,
):
    """Per-subject 2D histogram + pooled running-median line panel."""
    if len(x) == 0:
        print(f"  No pairs for {out_stem}, skipping figure")
        return

    n_cols = len(SUBJECTS)
    fig = plt.figure(figsize=(5 * n_cols, 7.5))
    gs = GridSpec(2, n_cols, figure=fig, height_ratios=[1, 1.1],
                  hspace=0.45, wspace=0.3)

    for col, subj in enumerate(SUBJECTS):
        ax = fig.add_subplot(gs[0, col])
        mask = pair_subject == subj
        xs = x[mask]
        ys = rho[mask]
        color = subject_colors[subj]

        H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, rho_edges])
        col_sums = H.sum(axis=1, keepdims=True)
        H_norm = np.where(col_sums > 0, H / np.maximum(col_sums, 1), 0.0)
        vmax = (np.nanpercentile(H_norm[H_norm > 0], 99)
                if (H_norm > 0).any() else 1.0)
        pcm = ax.pcolormesh(x_edges, rho_edges, H_norm.T, cmap="magma",
                            shading="flat", vmin=0, vmax=vmax)

        centers, means, _ci_lo, _ci_hi = running_mean_by_bin(xs, ys, x_edges)
        ax.plot(centers, means, "o-", color=color, lw=1.8, markersize=3,
                zorder=5, markeredgecolor="white", markeredgewidth=0.4)

        ax.axhline(0, color="white", ls="--", lw=0.6, alpha=0.7)
        ax.set_ylim(-RHO_RANGE, RHO_RANGE)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel("Corrected noise corr. ($\\rho_C$)")
        ax.set_title(subj)

        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() > 10:
            r_s, p_s = sp_stats.spearmanr(xs[ok], ys[ok])
            ax.text(0.97, 0.97, f"$r_s$ = {r_s:+.3f}\np = {p_s:.1e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="white",
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.4))
        ax.text(0.03, 0.97, f"n = {int(ok.sum()):,}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="white",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.4))

        if col == n_cols - 1:
            cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
            cb.set_label("Density (per bin)", fontsize=8)
            cb.ax.tick_params(labelsize=7)

    ax_bot = fig.add_subplot(gs[1, :])
    pooled_summaries = []
    for subj in SUBJECTS:
        mask = pair_subject == subj
        xs = x[mask]
        ys = rho[mask]
        color = subject_colors[subj]
        centers, means, ci_lo, ci_hi = running_mean_by_bin(xs, ys, x_edges)
        ax_bot.plot(centers, means, "-", color=color, lw=2,
                    marker="o", markersize=4, label=subj, zorder=5)
        ax_bot.fill_between(centers, ci_lo, ci_hi, alpha=0.25,
                            color=color, zorder=3, linewidth=0)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() > 10:
            r_s, p_s = sp_stats.spearmanr(xs[ok], ys[ok])
            pooled_summaries.append(f"{subj} $r_s$ = {r_s:+.3f}, p = {p_s:.1e}")

    ax_bot.axhline(0, color="gray", ls=":", lw=0.8, zorder=1)
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("Noise correlation (mean ± 95% CI)")
    ax_bot.set_title("Running mean vs similarity — both subjects")
    ax_bot.set_xlim(x_edges[0], x_edges[-1])
    ax_bot.legend(fontsize=9, loc="upper right")
    if pooled_summaries:
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


# %% Plots: noise correlation vs single-axis similarity metrics
ori_edges = np.linspace(0, 90, N_ORI_BINS + 1)
sf_edges = np.linspace(0, MAX_DLOG_SF, N_SF_BINS + 1)
sig_edges = np.linspace(MIN_SIGCORR, 1, N_SIGCORR_BINS + 1)

# Pairs outside the plot range (e.g. |Δlog2 SF| > MAX_DLOG_SF) fall outside
# the bin edges and are dropped naturally by histogram2d / running_mean_by_bin.
# The pooled Spearman/Pearson summaries still see the full data.

make_similarity_figure(
    d_ori_all, pair_subject_all, rho_c_all, ori_edges,
    xlabel="|$\\Delta$ preferred orientation| (deg)",
    out_stem="noise_corr_vs_delta_ori",
    suptitle="Corrected noise correlation vs. orientation-preference difference",
)

make_similarity_figure(
    d_logsf_all, pair_subject_all, rho_c_all, sf_edges,
    xlabel="|$\\Delta \\log_2$ preferred SF| (octaves)",
    out_stem="noise_corr_vs_delta_logsf",
    suptitle="Corrected noise correlation vs. SF-preference difference",
)

make_similarity_figure(
    sig_corr_all, pair_subject_all, rho_c_all, sig_edges,
    xlabel="Signal correlation (SF$\\times$ori tuning, peak lag)",
    out_stem="noise_corr_vs_signal_corr",
    suptitle="Corrected noise correlation vs. tuning-curve signal correlation",
)


# %% Joint plot: rho in (|Δlog2 SF|, |Δori|) grid, per subject + pooled
def make_joint_figure(x_sf, x_ori, pair_subject, rho,
                      sf_edges, ori_edges, out_stem, suptitle):
    if len(rho) == 0:
        print(f"  No pairs for {out_stem}, skipping")
        return

    n_cols = len(SUBJECTS) + 1   # + "Pooled"
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 4.4), squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    # Use pooled data to set a shared color range
    with np.errstate(invalid="ignore"):
        vabs = np.nanpercentile(np.abs(rho), 99)
    vabs = float(vabs if np.isfinite(vabs) else RHO_RANGE)
    vmin, vmax = -vabs, +vabs

    for i, subj in enumerate(SUBJECTS + ["Pooled"]):
        ax = axes[i]
        if subj == "Pooled":
            mask = np.ones(len(rho), dtype=bool)
        else:
            mask = pair_subject == subj

        xs_sf = x_sf[mask]
        xs_ori = x_ori[mask]
        rs = rho[mask]

        count, _, _ = np.histogram2d(xs_sf, xs_ori, bins=[sf_edges, ori_edges])
        sum_rho, _, _ = np.histogram2d(
            xs_sf, xs_ori, bins=[sf_edges, ori_edges], weights=rs
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_rho = np.where(count >= 5, sum_rho / np.maximum(count, 1), np.nan)

        pcm = ax.pcolormesh(sf_edges, ori_edges, mean_rho.T,
                            cmap="RdBu_r", shading="flat",
                            vmin=vmin, vmax=vmax)
        ax.set_xlabel("|$\\Delta \\log_2$ SF| (oct)")
        ax.set_ylabel("|$\\Delta$ ori| (deg)")
        ax.set_title(f"{subj} (n={int(mask.sum()):,})")
        ax.set_xlim(sf_edges[0], sf_edges[-1])
        ax.set_ylim(ori_edges[0], ori_edges[-1])

        cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Mean $\\rho_C$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(FIG_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
    print(f"Figure saved to {FIG_DIR / (out_stem + '.pdf')}")
    show_or_close(fig)


make_joint_figure(
    d_logsf_all, d_ori_all, pair_subject_all, rho_c_all,
    sf_edges, ori_edges,
    out_stem="noise_corr_joint_sf_ori",
    suptitle="Mean corrected noise correlation in (|$\\Delta \\log_2$ SF|, |$\\Delta$ ori|) bins",
)


# %% Summary statistics
print("\n=== Summary Statistics ===")
stat_lines = []


def _summary_block(tag, x, y, subj_array, xlabel):
    stat_lines.append(f"--- {tag} ({xlabel}) ---")
    print(stat_lines[-1])
    if len(x) == 0:
        stat_lines.append("  (no pairs)")
        print("  (no pairs)")
        return
    for subj in SUBJECTS + ["Pooled"]:
        if subj == "Pooled":
            m = np.ones(len(x), dtype=bool)
        else:
            m = subj_array == subj
        ok = m & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            line = f"  {subj:8s}: n={int(ok.sum()):6d}  (insufficient)"
        else:
            r_s, p_s = sp_stats.spearmanr(x[ok], y[ok])
            r_p, p_p = sp_stats.pearsonr(x[ok], y[ok])
            line = (f"  {subj:8s}: n={int(ok.sum()):6d}, "
                    f"mean_rho={np.mean(y[ok]):+.4f}, "
                    f"Spearman r={r_s:+.4f} (p={p_s:.2e}), "
                    f"Pearson r={r_p:+.4f} (p={p_p:.2e})")
        print(line)
        stat_lines.append(line)


_summary_block("Δori", d_ori_all, rho_c_all, pair_subject_all,
               "|Δ preferred orientation| (deg)")
_summary_block("ΔSF", d_logsf_all, rho_c_all, pair_subject_all,
               "|Δ log2 preferred SF| (octaves)")
_summary_block("Signal corr.", sig_corr_all, rho_c_all, pair_subject_all,
               "signal correlation (SF×ori tuning)")

stat_lines.append("")
stat_lines.append("Pair inclusion thresholds:")
stat_lines.append(f"  MIN_ORI_SNR={MIN_ORI_SNR}, MIN_SF_SNR={MIN_SF_SNR}")
stat_lines.append(f"  MIN_GRAT_SPIKES={MIN_GRAT_SPIKES}")
stat_lines.append(f"  MIN_TOTAL_SPIKES (fig2)={MIN_TOTAL_SPIKES}")
stat_lines.append(f"  primary window = {WINDOWS_MS[PRIMARY_WINDOW_IDX]:.1f} ms")

with open(STAT_DIR / "noise_corr_vs_tuning.txt", "w") as f:
    f.write("Noise correlation vs tuning similarity\n")
    f.write(f"Primary window: {WINDOWS_MS[PRIMARY_WINDOW_IDX]:.1f} ms\n\n")
    for line in stat_lines:
        f.write(line + "\n")

print(f"\nStats saved to {STAT_DIR / 'noise_corr_vs_tuning.txt'}")
