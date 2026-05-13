"""
Figure 1 panel F: population-level raster structure driven by gaze location.

Within a single fixation time-segment, trials are clustered into two groups
by their per-trial median gaze position such that the resulting population
PSTHs differ as much as possible. The two gaze clusters then drive a
gaze-sorted, two-block population raster.

Two side-by-side axes:
    (left)  trial-median gaze positions inside the segment, colored by
            cluster (blue = lower-rate cluster, red = higher-rate cluster).
    (right) population raster: cluster-0 trials stacked on top, cluster-1
            on bottom, with cluster-mean PSTHs overlaid.

Data flow follows ``generate_fig1d.py``:
    - One-time per-session load via ``tejas.rsvp_util.get_fixrsvp_data``,
      pickled to ``CACHE_DIR/fig1_population/`` so reruns are tejas-free.
    - Population peak lag from the cached STEs written by ``generate_fig1c.py``
      (no dependency on ``tejas.metrics.gaborium``).

Usage:
    uv run ryan/fig1/generate_fig1f.py
"""

from pathlib import Path
import pickle
import warnings
from itertools import combinations
from math import comb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR, CACHE_DIR

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Configuration  (defaults from fig1_fixrsvp_population.py)
# ---------------------------------------------------------------------------
SUBJECT = "Allen"
DATE = "2022-03-02"
DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)

DT = 1.0 / 240.0

# Segment-of-fixation in which to cluster gaze (bin indices into the
# per-trial fixation-aligned time axis). The exploratory script used a
# single 32-bin segment [46, 78).
SEGMENT_START_BIN = 46
SEGMENT_END_BIN = 78

# Raster window: pad the segment on either side so the cluster split is
# visible against pre- and post-segment activity.
RASTER_PAD_BINS = 30

# Gaze-clustering parameters (mirrors the call site at line ~969 of the
# exploratory script).
NUM_CLUSTERS = 2
CLUSTER_SIZE = 5            # exact size per cluster
MAX_DIST_FROM_CENTROID = 0.10
DIST_BETWEEN_CENTROIDS = (0.02, 0.30)
MIN_INTER_CLUSTER_DIST = 0.0
MICROSACCADE_THRESHOLD = 0.10
RETURN_TOP_K_COMBOS = 10
COMBO_IDX = 1               # script selected the second-best combo (j=1)

# Raster rendering.
PSTH_BIN_SIZE = 0.001       # 1 ms PSTH binning from spike times
RASTER_GAP_BINS = 50

CACHE_FIG_DIR = CACHE_DIR / "fig1_population"
RF_CACHE_DIR = CACHE_DIR / "fig1_rf_contours"
FIG_DIR = FIGURES_DIR / "fig1"
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading: cache the full fixrsvp payload per-session
# ---------------------------------------------------------------------------
def _fixrsvp_cache_path(subject, date):
    return CACHE_FIG_DIR / f"{subject}_{date}_fixrsvp.pkl"


def _load_fixrsvp_data(subject, date, refresh=False):
    """Load the fixrsvp payload for a session. First call goes through
    ``tejas.rsvp_util.get_fixrsvp_data``; subsequent calls are tejas-free."""
    cache = _fixrsvp_cache_path(subject, date)
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    from tejas.rsvp_util import get_fixrsvp_data
    data = get_fixrsvp_data(
        subject, date, DATASET_CONFIGS_PATH,
        use_cached_data=True,
        salvageable_mismatch_time_threshold=25,
        verbose=False,
    )
    payload = {
        "robs": np.asarray(data["robs"]),
        "eyepos": np.asarray(data["eyepos"]),
        "fix_dur": np.asarray(data["fix_dur"]),
        "cids": list(data["cids"]),
        "spike_times_trials": data["spike_times_trials"],
        "trial_t_bins": data["trial_t_bins"],
    }
    with open(cache, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Peak lag from fig1c STE artifacts (population median)
# ---------------------------------------------------------------------------
def _load_ste_for_session(session_name):
    path = RF_CACHE_DIR / f"{session_name}_sta_ste.npz"
    if not path.exists():
        return None
    return np.load(path)


def _population_peak_lag(session_name, robs=None):
    ste_npz = _load_ste_for_session(session_name)
    if ste_npz is not None:
        stes = ste_npz["stes"]
        lags = [int(stes[u].std(axis=(1, 2)).argmax()) for u in range(stes.shape[0])]
        return int(np.median(lags))
    if robs is not None:
        psth = np.nanmean(robs, axis=(0, 2))
        return int(np.nanargmax(psth))
    raise RuntimeError(f"No STE cache for {session_name} and no robs to fall back on")


# ---------------------------------------------------------------------------
# Gaze clustering
# ---------------------------------------------------------------------------
def _microsaccade_exists(trace, threshold=MICROSACCADE_THRESHOLD):
    med = np.nanmedian(trace, axis=0)
    d = np.hypot(trace[:, 0] - med[0], trace[:, 1] - med[1])
    return bool(np.any(d > threshold))


def _get_eyepos_clusters(
    eyepos, robs, start_time, end_time, peak_lag,
    num_clusters=NUM_CLUSTERS,
    cluster_size=CLUSTER_SIZE,
    max_dist_from_centroid=MAX_DIST_FROM_CENTROID,
    dist_between_centroids=DIST_BETWEEN_CENTROIDS,
    min_inter_cluster_dist=MIN_INTER_CLUSTER_DIST,
    return_top_k=RETURN_TOP_K_COMBOS,
):
    """Cluster trials by per-trial median gaze inside the (peak-lag-shifted)
    segment so that the resulting population PSTHs differ as much as possible.

    Trimmed port of ``get_eyepos_clusters`` (line ~48 of
    fig1_fixrsvp_population.py): keeps only ``sort_by_cluster_psth=True``
    with ``method='psth_diff'``, ``cluster_size == min_cluster_size``, and
    deduped top-K combos.

    Returns
    -------
    iix_list, clusters_list : lists of length ``min(return_top_k, K_found)``
        ``iix_list[k]`` is the array of valid trial indices (same for all k),
        ``clusters_list[k][i]`` is the cluster label of trial ``iix_list[k][i]``
        (0-indexed by ascending total spike sum; -1 = unclustered).
    """
    win_len = end_time - start_time
    s = max(start_time - peak_lag, 0)
    e = s + win_len

    # Valid trials: have eyepos data, have at least half the robs window
    # populated, and contain no microsaccade in the shifted window.
    robs_se = robs[:, s:e, :]
    valid = []
    for i in range(eyepos.shape[0]):
        if np.isnan(eyepos[i, s:e, :]).all():
            continue
        if np.isnan(robs_se[i]).sum() > robs_se[i].size // 2:
            continue
        if _microsaccade_exists(eyepos[i, s:e, :]):
            continue
        valid.append(i)
    iix = np.asarray(valid)
    if len(iix) < num_clusters * cluster_size:
        return [iix], [np.full(len(iix), -1)]

    # Per-trial gaze medians and pairwise distances.
    medians = np.array([np.nanmedian(eyepos[i, s:e, :], axis=0) for i in iix])
    pairwise = cdist(medians, medians)

    # Precompute per-trial response summaries for fast PSTH scoring
    # (NaN-aware sum over cells).
    trial_sum_tc = np.nansum(robs_se[iix], axis=2)          # [n_valid, time]
    trial_count_tc = np.sum(~np.isnan(robs_se[iix]), axis=2)
    trial_sum = np.nansum(trial_sum_tc, axis=1)             # [n_valid]

    # Candidate centroids: any point with >= cluster_size neighbors within
    # max_dist_from_centroid (itself included).
    within = [np.flatnonzero(pairwise[i] <= max_dist_from_centroid)
              for i in range(len(medians))]
    candidate_centroids = [i for i in range(len(medians))
                           if len(within[i]) >= cluster_size]
    if len(candidate_centroids) < num_clusters:
        return [iix], [np.full(len(iix), -1)]

    def _psth(members):
        members = np.asarray(members, dtype=int)
        s_tc = np.nansum(trial_sum_tc[members], axis=0)
        c_tc = np.nansum(trial_count_tc[members], axis=0)
        return np.where(c_tc > 0, s_tc / c_tc, np.nan)

    psth_cache = {}

    def _psth_diff(m1, m2):
        k1 = tuple(sorted(int(x) for x in m1))
        k2 = tuple(sorted(int(x) for x in m2))
        if k2 < k1:
            k1, k2 = k2, k1
        key = (k1, k2)
        if key in psth_cache:
            return psth_cache[key]
        p1, p2 = _psth(m1), _psth(m2)
        if np.isnan(p1).sum() > len(p1) // 2 or np.isnan(p2).sum() > len(p2) // 2:
            val = 0.0
        else:
            val = float(np.linalg.norm(p1 - p2))
        psth_cache[key] = val
        return val

    dmin, dmax = dist_between_centroids
    best_by_partition = {}   # canonical (sorted-members) → (score, members)

    centroid_combos = list(combinations(candidate_centroids, num_clusters))
    for c_combo in tqdm(centroid_combos, desc="centroid combos", leave=False):
        # Filter by inter-centroid distance.
        ok = True
        for i, j in combinations(range(num_clusters), 2):
            d_ij = pairwise[c_combo[i], c_combo[j]]
            if d_ij < dmin or d_ij > dmax:
                ok = False
                break
        if not ok:
            continue
        pools = [within[c] for c in c_combo]
        if any(len(p) < cluster_size for p in pools):
            continue

        # Enumerate cluster-size subsets per pool; for num_clusters=2 this is
        # the only loop. (Higher num_clusters would nest combinations, but
        # the call site only uses 2 so we keep it explicit.)
        if num_clusters != 2:
            raise NotImplementedError("num_clusters != 2 not ported")
        n0 = comb(len(pools[0]), cluster_size)
        n1 = comb(len(pools[1]), cluster_size)
        for m0 in combinations(pools[0], cluster_size):
            set0 = set(m0)
            for m1 in combinations(pools[1], cluster_size):
                if any(x in set0 for x in m1):
                    continue
                if min_inter_cluster_dist > 0:
                    if pairwise[np.ix_(m0, m1)].min() < min_inter_cluster_dist:
                        continue
                score = _psth_diff(m0, m1)
                key = tuple(sorted((tuple(sorted(m0)), tuple(sorted(m1)))))
                prev = best_by_partition.get(key)
                if prev is None or score > prev[0]:
                    best_by_partition[key] = (score, (np.asarray(m0, int),
                                                      np.asarray(m1, int)))

    if not best_by_partition:
        return [iix], [np.full(len(iix), -1)]

    top = sorted(best_by_partition.values(), key=lambda x: x[0], reverse=True)
    top = top[:return_top_k]

    iix_out, clusters_out = [], []
    for _, members in top:
        labels = np.full(len(medians), -1, dtype=int)
        for c_idx, mem in enumerate(members):
            labels[mem] = c_idx
        # Order clusters so cluster 0 has the smaller total spike sum.
        sums = [float(np.nansum(robs_se[iix[labels == c]]))
                for c in range(num_clusters)]
        order = np.argsort(sums)
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap.get(c, -1) for c in labels])
        iix_out.append(iix)
        clusters_out.append(labels)
    return iix_out, clusters_out


# ---------------------------------------------------------------------------
# Panel payload (cached)
# ---------------------------------------------------------------------------
def _payload_cache_path(subject, date):
    return CACHE_FIG_DIR / f"{subject}_{date}_panel_f.pkl"


def _compute_panel_payload(subject, date, combo_idx=COMBO_IDX):
    data = _load_fixrsvp_data(subject, date)
    robs = data["robs"]
    eyepos = data["eyepos"]
    session = f"{subject}_{date}"
    peak_lag = _population_peak_lag(session, robs=robs)

    iix_list, clusters_list = _get_eyepos_clusters(
        eyepos, robs, SEGMENT_START_BIN, SEGMENT_END_BIN, peak_lag,
    )
    k = min(combo_idx, len(iix_list) - 1)
    iix = iix_list[k]
    clusters = clusters_list[k]

    raster_start = max(SEGMENT_START_BIN - RASTER_PAD_BINS, 0)
    raster_end = SEGMENT_END_BIN + RASTER_PAD_BINS

    return {
        "subject": subject,
        "date": date,
        "session": session,
        "peak_lag": int(peak_lag),
        "segment_start": int(SEGMENT_START_BIN),
        "segment_end": int(SEGMENT_END_BIN),
        "raster_start": int(raster_start),
        "raster_end": int(raster_end),
        "iix": iix,
        "clusters": clusters,
        # Keep the full session arrays for plotting; cached pickle is large
        # but only loaded on demand.
        "eyepos": eyepos,
        "robs": robs,
        "spike_times_trials": data["spike_times_trials"],
        "trial_t_bins": data["trial_t_bins"],
        "cids": data["cids"],
    }


def load_panel_payload(subject=SUBJECT, date=DATE, refresh=False):
    path = _payload_cache_path(subject, date)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    payload = _compute_panel_payload(subject, date)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _cluster_colors(clusters):
    n = len(set(int(c) for c in clusters if c >= 0))
    base = plt.cm.coolwarm(np.linspace(0, 1, max(n, 1)))
    return [base[c] if c >= 0 else (0.5, 0.5, 0.5, 0.3) for c in clusters]


def plot_eyepos_axis(ax, payload, show_unclustered=False):
    """Left axis: 2D scatter of per-trial median gaze inside the (peak-lag
    shifted) segment, colored by cluster."""
    eye = payload["eyepos"]
    iix = payload["iix"]
    clusters = payload["clusters"]
    s = max(payload["segment_start"] - payload["peak_lag"], 0)
    win_len = payload["segment_end"] - payload["segment_start"]
    e = s + win_len

    colors = _cluster_colors(clusters)
    for idx in range(len(iix)):
        c = clusters[idx]
        if c < 0 and not show_unclustered:
            continue
        trace = eye[iix[idx], s:e, :]
        med = np.nanmedian(trace, axis=0)
        ax.plot(trace[:, 0], trace[:, 1], color=colors[idx], lw=0.7, alpha=0.9)
        ax.scatter(med[0], med[1], color=colors[idx], s=20,
                   edgecolor="k", linewidth=0.7, zorder=3)

    drawn = [iix[i] for i in range(len(iix))
             if clusters[i] >= 0 or show_unclustered]
    if drawn:
        x = eye[drawn, s:e, 0]
        y = eye[drawn, s:e, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
        rng = max(xmax - xmin, ymax - ymin)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        ax.set_xlim(cx - rng / 2 - 0.02, cx + rng / 2 + 0.02)
        ax.set_ylim(cy - rng / 2 - 0.02, cy + rng / 2 + 0.02)

    ax.set_aspect("equal")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    return ax


def _trial_spike_times_in_window(spike_times_trial, t_bins_trial, t0_bin, t1_bin):
    """Return (spike_times_seconds, cell_indices) for a trial, with spike
    times re-zeroed to fixation onset and clipped to [t0_bin, t1_bin)."""
    t_bins = np.asarray(t_bins_trial)
    valid = ~np.isnan(t_bins)
    if not np.any(valid):
        return np.array([]), np.array([], dtype=int)
    vt = t_bins[valid]
    fix_onset = vt[0] - DT / 2          # absolute time of bin-0 left edge
    win_start_s = t0_bin * DT
    win_end_s = t1_bin * DT
    all_t, all_c = [], []
    for cell_idx, sp in enumerate(spike_times_trial):
        sp = np.atleast_1d(np.asarray(sp))
        if sp.size == 0:
            continue
        rel = sp - fix_onset
        mask = (rel >= win_start_s) & (rel < win_end_s)
        if not np.any(mask):
            continue
        all_t.append(rel[mask])
        all_c.append(np.full(mask.sum(), cell_idx, dtype=int))
    if not all_t:
        return np.array([]), np.array([], dtype=int)
    return np.concatenate(all_t), np.concatenate(all_c)


def plot_raster_axis(ax, payload, gap=RASTER_GAP_BINS, show_psth=True,
                    psth_height_factor=1.0):
    """Right axis: cluster-grouped population raster from spike times, with
    cluster-mean PSTHs (1 ms binning) overlaid."""
    iix = payload["iix"]
    clusters = payload["clusters"]
    robs = payload["robs"]
    spike_times = payload["spike_times_trials"]
    t_bins = payload["trial_t_bins"]
    t0_bin = payload["raster_start"]
    t1_bin = payload["raster_end"]
    n_cells = robs.shape[2]

    # Split iix by cluster.
    c0_trials = [iix[i] for i in range(len(iix)) if clusters[i] == 0]
    c1_trials = [iix[i] for i in range(len(iix)) if clusters[i] == 1]
    n0, n1 = len(c0_trials), len(c1_trials)

    psth_height = n_cells * psth_height_factor
    psth_block = (2 * (psth_height + gap)) if show_psth else 0
    block_h = n_cells + gap
    row_c0_start = 0
    row_psth_start = n0 * block_h
    row_c1_start = row_psth_start + psth_block
    total_rows = row_c1_start + n1 * block_h - gap

    win_dur_s = (t1_bin - t0_bin) * DT
    win_dur_ms = win_dur_s * 1000.0
    t0_ms = t0_bin * DT * 1000.0
    t1_ms = t1_bin * DT * 1000.0

    spike_xs, spike_ys, spike_y2s = [], [], []
    tick_positions, tick_labels, tick_colors = [], [], []

    def _add_trial(trial_id, row_top, cluster_label, trial_number):
        times_s, cells = _trial_spike_times_in_window(
            spike_times[trial_id], t_bins[trial_id], t0_bin, t1_bin,
        )
        if times_s.size:
            x_ms = times_s * 1000.0 + t0_ms
            for xt, cell in zip(x_ms, cells):
                spike_xs.append(xt)
                spike_ys.append(row_top + cell)
                spike_y2s.append(row_top + cell + 0.7)
        tick_positions.append(row_top + n_cells / 2)
        tick_labels.append(str(trial_number))
        tick_colors.append("blue" if cluster_label == 0 else "red")

    row = row_c0_start
    trial_n = 1
    for tid in c0_trials:
        _add_trial(tid, row, 0, trial_n)
        row += block_h
        trial_n += 1
    row = row_c1_start
    for tid in c1_trials:
        _add_trial(tid, row, 1, trial_n)
        row += block_h
        trial_n += 1

    if spike_xs:
        xs = np.asarray(spike_xs)
        ys = np.asarray(spike_ys)
        y2s = np.asarray(spike_y2s)
        nan = np.full_like(xs, np.nan)
        seg_x = np.empty(xs.size * 3)
        seg_y = np.empty(xs.size * 3)
        seg_x[0::3] = xs
        seg_x[1::3] = xs
        seg_x[2::3] = nan
        seg_y[0::3] = ys
        seg_y[1::3] = y2s
        seg_y[2::3] = nan
        ax.plot(seg_x, seg_y, color="k", lw=0.5, rasterized=True)

    if show_psth and (n0 > 0 or n1 > 0):
        psth_edges = np.arange(0, win_dur_s + PSTH_BIN_SIZE / 2, PSTH_BIN_SIZE)

        def _mean_psth(trial_ids):
            if not trial_ids:
                return np.zeros(len(psth_edges) - 1)
            rows = []
            for tid in trial_ids:
                times_s, _ = _trial_spike_times_in_window(
                    spike_times[tid], t_bins[tid], t0_bin, t1_bin,
                )
                counts, _ = np.histogram(times_s, bins=psth_edges)
                rows.append(counts)
            return np.mean(rows, axis=0)

        psth0 = _mean_psth(c0_trials)
        psth1 = _mean_psth(c1_trials)
        max_psth = max(psth0.max(), psth1.max(), 1e-10)

        edges_ms = psth_edges * 1000.0 + t0_ms
        x_psth = 0.5 * (edges_ms[:-1] + edges_ms[1:])

        off0 = row_psth_start
        off1 = row_psth_start + (psth_height + gap)
        p0_y = off0 + psth_height - (psth0 / max_psth) * psth_height
        p1_y = off1 + psth_height - (psth1 / max_psth) * psth_height
        ax.fill_between(x_psth, off0 + psth_height, p0_y, color="blue", alpha=0.4)
        ax.fill_between(x_psth, off1 + psth_height, p1_y, color="red", alpha=0.4)

    # Segment-of-cluster boundary lines.
    seg_s_ms = payload["segment_start"] * DT * 1000.0
    seg_e_ms = payload["segment_end"] * DT * 1000.0
    ax.axvline(seg_s_ms, color="0.7", lw=0.5, ls="--", zorder=0)
    ax.axvline(seg_e_ms, color="0.7", lw=0.5, ls="--", zorder=0)

    ax.set_xlim(t0_ms, t1_ms)
    ax.set_ylim(total_rows, 0)
    ax.set_xlabel("Time from fixation onset (ms)")
    ax.set_ylabel("Trial")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    for lbl, col in zip(ax.get_yticklabels(), tick_colors):
        lbl.set_color(col)
    return ax


def plot_panel_f(fig=None, subject=SUBJECT, date=DATE, refresh=False):
    payload = load_panel_payload(subject, date, refresh=refresh)

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5),
                                 gridspec_kw={"width_ratios": [1.0, 1.6]})
    else:
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.6]})

    plot_eyepos_axis(axes[0], payload)
    plot_raster_axis(axes[1], payload)
    axes[0].set_title(
        f"Gaze in segment [{payload['segment_start']},{payload['segment_end']}) bins "
        f"({payload['session']}, N={len(payload['cids'])} cells)",
        fontsize=9,
    )
    axes[1].set_title("Gaze-cluster population raster", fontsize=9)
    return fig, axes


if __name__ == "__main__":
    fig, axes = plot_panel_f()
    fig.tight_layout()
    out = FIG_DIR / "fig1f_population.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
