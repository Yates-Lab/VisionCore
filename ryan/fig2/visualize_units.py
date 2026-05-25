# %% Imports and configuration
"""
Per-unit visualization to explore inclusion thresholds for Figure 2.

For every neuron passing the dataset's RF-SNR filter (with no spike-count
threshold), produce a 4-panel page:
    1. Per-trial spike raster, sorted by trial duration, red tick at trial end.
    2. Trial-averaged PSTH (spk/s) with SEM band.
    3. (in title) split-half PSTH R^2 over N random trial halvings.
    4. McFarland-style intra-unit covariance vs. delta eye trajectory:
       diagonal Ceye[:, i, i] across distance bins, with dashed horizontal
       lines at total variance (Ctotal[i,i]) and PSTH variance (Cpsth[i,i]);
       title shows 1 - alpha.

One multi-page PDF per session, written to FIGURES_DIR/fig2/unit_viz/.
"""
import sys
import pickle

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR
from VisionCore.covariance import (
    align_fixrsvp_trials,
    extract_valid_segments,
    extract_windows,
    estimate_rate_covariance,
    bagged_split_half_psth_covariance,
)
from DataYatesV1 import get_free_device

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RECOMPUTE = True

DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
DT = 1.0 / 120.0
WINDOW_BINS = 2              # counting window (~16.7 ms at 120 Hz)
T_HIST_BINS = 1              # history window for eye trajectory similarity
N_DIST_BINS = 15
INTERCEPT_MODE = "linear"
INTERCEPT_D_MAX = 0.2        # only fit bins with Δeye <= 0.2 deg
INTERCEPT_KWARGS = {"d_max": INTERCEPT_D_MAX, "eval_at_first_bin": False}
SUBJECTS = ("Allen", "Logan")

N_SPLITS = 100               # random halvings for split-half PSTH R^2
SEED = 42

CACHE_FILE = CACHE_DIR / "fig2_unit_explore.pkl"
OUT_DIR = FIGURES_DIR / "fig2" / "unit_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = get_free_device()


# ---------------------------------------------------------------------------
# Per-session payload
# ---------------------------------------------------------------------------
def compute_session_payload(cfg):
    """Run alignment + decomposition for one session with min_total_spikes=0.

    Returns a dict ready for plotting, or None on failure.
    """
    session_name = cfg["session"]
    subject = session_name.split("_")[0]

    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    from models.data import prepare_data
    try:
        train_data, _val, cfg = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  prepare_data failed: {e}")
        return None

    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (ValueError, KeyError):
        print("  no fixrsvp dataset")
        return None
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        valid_time_bins=120,
        min_fix_dur=20,
        min_total_spikes=0,   # keep every unit; RF-SNR filter is upstream
    )
    if robs is None or robs.shape[0] < 10:
        print(f"  insufficient data: {meta}")
        return None

    n_trials, n_time, n_units = robs.shape
    print(f"  trials={n_trials}, units={n_units}")

    # Run covariance pipeline (mirrors run_covariance_decomposition for one window)
    robs_clean = np.nan_to_num(robs, nan=0.0)
    eyepos_clean = np.nan_to_num(eyepos, nan=0.0)

    robs_t = torch.tensor(robs_clean, dtype=torch.float32, device=DEVICE)
    eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32, device=DEVICE)

    segments = extract_valid_segments(valid_mask, min_len_bins=36)
    if len(segments) == 0:
        print("  no valid segments")
        return None

    SpikeCounts, EyeTraj, T_idx, _ = extract_windows(
        robs_t, eyepos_t, segments,
        t_count=WINDOW_BINS,
        t_hist=max(T_HIST_BINS, WINDOW_BINS),
        device=str(DEVICE),
    )
    if SpikeCounts is None or SpikeCounts.shape[0] < 100:
        print("  too few windows")
        return None

    ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy()

    Crate, Erate, Ceye, bin_centers, count_e, _ = estimate_rate_covariance(
        SpikeCounts, EyeTraj, T_idx, n_bins=N_DIST_BINS,
        Ctotal=Ctotal, intercept_mode=INTERCEPT_MODE,
        intercept_kwargs=INTERCEPT_KWARGS,
    )

    Cpsth, _ = bagged_split_half_psth_covariance(
        SpikeCounts, T_idx, n_boot=20, min_trials_per_time=10,
        seed=SEED, global_mean=Erate,
    )

    # Split-half PSTH reliability (per unit) — R^2 over random trial halvings
    rng = np.random.default_rng(SEED)
    r2_sum = np.zeros(n_units)
    r2_cnt = np.zeros(n_units, dtype=int)
    for _ in range(N_SPLITS):
        perm = rng.permutation(n_trials)
        h = n_trials // 2
        psth_a = np.nanmean(robs[perm[:h]], axis=0)         # (T, U)
        psth_b = np.nanmean(robs[perm[h:2 * h]], axis=0)
        for j in range(n_units):
            a, b = psth_a[:, j], psth_b[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
                continue
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            if np.isfinite(r):
                r2_sum[j] += r * r
                r2_cnt[j] += 1
    psth_r2 = np.where(r2_cnt > 0, r2_sum / np.maximum(r2_cnt, 1), np.nan)

    # Trial duration (valid bins) used to sort the raster
    trial_dur = valid_mask.sum(axis=1)

    return {
        "session": session_name,
        "subject": subject,
        "meta": meta,
        "robs": robs,                  # (n_trials, n_time, n_units)  NaN-padded
        "valid_mask": valid_mask,      # (n_trials, n_time)
        "trial_dur": trial_dur,        # (n_trials,)
        "neuron_mask": neuron_mask,
        "Ctotal": Ctotal,
        "Cpsth": Cpsth,
        "Crate": Crate,
        "Ceye": Ceye,                  # (n_bins, U, U)
        "bin_centers": np.asarray(bin_centers),
        "count_e": np.asarray(count_e),
        "Erate": Erate,
        "psth_r2": psth_r2,
    }


def load_payloads(recompute=False):
    if CACHE_FILE.exists() and not recompute:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs

    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    payloads = []
    for cfg in dataset_configs:
        session_name = cfg["session"]
        subject = session_name.split("_")[0]
        if subject not in SUBJECTS:
            continue
        print(f"\n--- {session_name} ({subject}) ---")
        try:
            p = compute_session_payload(cfg)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        if p is not None:
            payloads.append(p)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payloads, f)
    print(f"\nCached {len(payloads)} sessions to {CACHE_FILE}")
    return payloads


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _weighted_linear_fit(x, y, w):
    """Weighted least-squares linear fit y ~ a + b*x. Returns (a, b) or (nan, nan)."""
    S0 = w.sum()
    Sx = (w * x).sum()
    Sxx = (w * x * x).sum()
    Sy = (w * y).sum()
    Sxy = (w * x * y).sum()
    det = S0 * Sxx - Sx * Sx
    if det <= 0:
        return np.nan, np.nan
    slope = (S0 * Sxy - Sx * Sy) / det
    intercept = (Sxx * Sy - Sx * Sxy) / det
    return float(intercept), float(slope)



def plot_unit_page(p, j, fig):
    """Draw the 4-panel page for unit j of session payload p."""
    robs = p["robs"]
    valid_mask = p["valid_mask"]
    trial_dur = p["trial_dur"]
    Ctotal = p["Ctotal"]
    Cpsth = p["Cpsth"]
    Crate = p["Crate"]
    Ceye = p["Ceye"]
    bin_centers = p["bin_centers"]
    count_e = p["count_e"]
    Erate = p["Erate"]
    psth_r2 = p["psth_r2"]

    n_trials, n_time, _ = robs.shape

    # Sort trials by duration (longest first)
    order = np.argsort(trial_dur)[::-1]
    spikes_unit = robs[order, :, j]              # (n_trials, n_time)
    dur_sorted = trial_dur[order]

    # ------- title metrics -------
    var_tot = float(Ctotal[j, j])
    var_psth = float(Cpsth[j, j])
    var_rate = float(Crate[j, j])
    if np.isfinite(var_rate) and var_rate > 0:
        alpha = var_psth / var_rate
        one_minus_alpha = 1.0 - alpha
    else:
        alpha = np.nan
        one_minus_alpha = np.nan
    r2 = float(psth_r2[j]) if np.isfinite(psth_r2[j]) else np.nan
    n_spikes = float(np.nansum(robs[:, :, j]))
    n_valid_bins = float(np.sum(np.isfinite(robs[:, :, j])))
    rate_hz = n_spikes / (n_valid_bins * DT) if n_valid_bins > 0 else np.nan

    fig.suptitle(
        f"{p['session']}  unit {j} (orig {int(p['neuron_mask'][j])})  "
        f"  rate={rate_hz:.2f} Hz   R²(split-half)={r2:.3f}   "
        f"1-α={one_minus_alpha:.3f}",
        fontsize=11,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax_raster = fig.add_subplot(gs[0, 0])
    ax_psth = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[1, :])

    # ------- raster -------
    ts_ms = np.arange(n_time) * DT * 1000.0
    xs, ys = [], []
    for k in range(n_trials):
        s = spikes_unit[k]
        spk_bins = np.where(s > 0)[0]
        if spk_bins.size == 0:
            continue
        # one tick per spike count (small units typically 0/1)
        for b in spk_bins:
            c = int(s[b])
            xs.extend([ts_ms[b]] * c)
            ys.extend([k] * c)
    if xs:
        ax_raster.plot(xs, ys, "|", color="k", markersize=2, markeredgewidth=0.6)
    # red tick at end of each trial
    end_ms = (dur_sorted - 1) * DT * 1000.0
    ax_raster.plot(end_ms, np.arange(n_trials), "|",
                   color="tab:red", markersize=4, markeredgewidth=0.8)
    ax_raster.set_xlim(0, n_time * DT * 1000.0)
    ax_raster.set_ylim(n_trials, -1)
    ax_raster.set_xlabel("Time from fixation onset (ms)")
    ax_raster.set_ylabel("Trial (sorted by duration)")
    ax_raster.set_title("Raster", fontsize=10)

    # ------- PSTH -------
    with np.errstate(invalid="ignore"):
        psth_mean = np.nanmean(robs[:, :, j], axis=0) / DT
        n_per_t = np.sum(np.isfinite(robs[:, :, j]), axis=0)
        psth_sem = np.nanstd(robs[:, :, j], axis=0) / np.sqrt(np.maximum(n_per_t, 1)) / DT
    t_ms = (np.arange(n_time) + 0.5) * DT * 1000.0
    ax_psth.fill_between(t_ms, psth_mean - psth_sem, psth_mean + psth_sem,
                         color="0.7", alpha=0.5, linewidth=0)
    ax_psth.plot(t_ms, psth_mean, color="k", lw=1.0)
    ax_psth.set_xlim(0, n_time * DT * 1000.0)
    ax_psth.set_ylim(bottom=0)
    ax_psth.set_xlabel("Time from fixation onset (ms)")
    ax_psth.set_ylabel("Spikes / s")
    ax_psth.set_title("PSTH", fontsize=10)

    # ------- covariance vs eye distance (diagonal: variance) -------
    # Ceye is already MM - Erate Erate^T, so Ceye[:, j, j] is the variance per bin.
    var_by_bin = Ceye[:, j, j]
    ok = np.isfinite(var_by_bin) & (count_e > 0)
    if np.any(ok):
        sizes = 8 + 60 * (count_e[ok] / max(count_e.max(), 1))
        ax_cov.scatter(bin_centers[ok], var_by_bin[ok], s=sizes,
                       color="tab:blue", alpha=0.7, edgecolor="k", linewidth=0.3,
                       label="Ceye[k, j, j]")

    # Linear extrapolator over bins with 0 < x <= d_max
    d_max = INTERCEPT_D_MAX
    fit_mask = ok & (bin_centers > 0) & (bin_centers <= d_max)
    if fit_mask.sum() >= 2:
        a_fit, b_fit = _weighted_linear_fit(
            bin_centers[fit_mask], var_by_bin[fit_mask], count_e[fit_mask],
        )
        if np.isfinite(a_fit):
            x_line = np.array([0.0, d_max])
            ax_cov.plot(x_line, a_fit + b_fit * x_line,
                        color="tab:red", ls="--", lw=1.2,
                        label=f"Linear fit (≤{d_max:g}°)")
            ax_cov.scatter([0.0], [a_fit], marker="x", color="tab:red",
                           s=80, linewidths=2.0, zorder=5,
                           label=f"Intercept={a_fit:.3g}")
    if np.isfinite(var_psth):
        ax_cov.axhline(var_psth, color="tab:green", ls="--", lw=1.0,
                       label=f"PSTH var={var_psth:.3g}")
    if np.isfinite(var_rate):
        ax_cov.axhline(var_rate, color="tab:purple", ls=":", lw=1.0,
                       label=f"Rate var (intercept)={var_rate:.3g}")
    ax_cov.set_xlabel("Δ eye trajectory (deg)")
    ax_cov.set_ylabel("Variance")
    ax_cov.set_title(
        f"Within-unit covariance vs. eye distance   "
        f"Total var={var_tot:.3g}   1-α = {one_minus_alpha:.3f}",
        fontsize=10,
    )
    ax_cov.legend(fontsize=7, frameon=False, loc="best")
    ax_cov.grid(True, alpha=0.3)


def render_session_pdf(p):
    out = OUT_DIR / f"{p['session']}.pdf"
    n_units = p["robs"].shape[2]
    with PdfPages(out) as pdf:
        for j in range(n_units):
            fig = plt.figure(figsize=(11, 7))
            plot_unit_page(p, j, fig)
            pdf.savefig(fig, bbox_inches="tight", dpi=120)
            plt.close(fig)
    print(f"  -> {out}  ({n_units} units)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    payloads = load_payloads(recompute=RECOMPUTE)
    print(f"\nRendering {len(payloads)} session PDFs to {OUT_DIR}")
    for p in payloads:
        print(f"[{p['session']}]")
        render_session_pdf(p)
    print("\nDone.")
