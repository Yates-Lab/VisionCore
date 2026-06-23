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
    - One-time per-session load via ``eval.fixrsvp.get_fixrsvp_data``,
      pickled to ``CACHE_DIR/fig1_population/`` so reruns avoid the heavy
      fixrsvp extraction.
    - Population peak lag from the shared STE cache produced by
      ``eval.sta_ste.compute_sta_ste``.

Usage:
    uv run ryan/fig1/generate_fig1f.py
"""

from pathlib import Path
import pickle
from itertools import combinations
from math import comb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR, CACHE_DIR
from eval.sta_ste import compute_sta_ste, population_peak_lag

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------------------
# Configuration  (defaults from fig1_fixrsvp_population.py)
# ---------------------------------------------------------------------------
# Flip to True to force fixrsvp/STA-STE/payload caches to be regenerated.
RECALC = False

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

# Display windows (ms from fixation onset).
RASTER_WINDOW_MS = (0.0, 400.0)
# Gray pad on either side of the highlighted (cluster-colored) region in the
# gaze axes. Set to 0 to show only the highlighted window.
GAZE_PAD_MS = 0.0
# Symmetric y-limit for both gaze axes, in degrees.
GAZE_YLIM_HALF = 0.5
GAZE_WINDOW_MS = (
    RASTER_WINDOW_MS[0] - GAZE_PAD_MS,
    RASTER_WINDOW_MS[1] + GAZE_PAD_MS,
)

# Display rows (1-indexed in the combined cluster-0-then-cluster-1 ordering)
# to drop from the figure.
DROP_DISPLAY_ROWS = (2, 7)

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
PSTH_BIN_MS = 5.0           # ms binning for line PSTHs
RASTER_GAP_BINS = 10
RASTER_GROUP_GAP_BINS = 18
GROUP_BLUE = "#6f95ee"
GROUP_RED = "#df6556"
GROUP_BLUE_FILL = "#e8f0ff"
GROUP_RED_FILL = "#fde1dc"
# PSTH line colors propagated from the single-cell PSTH (fig1G) so the
# population PSTH (fig1J) matches it. The raster background fills already
# share fig1F's hex; only their alpha is matched (see plot_raster_axis).
PSTH_BLUE = "#3b6db7"
PSTH_RED = "#c43c3c"
GROUP_FILL_ALPHA = 0.42

CACHE_FIG_DIR = CACHE_DIR / "fig1_population"
FIG_DIR = FIGURES_DIR / "fig1"
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PANEL_LABEL_FONTSIZE_PT = 16


def _fmt_deg_tick(v):
    s = f"{v:.1f}".rstrip("0").rstrip(".")
    if s.startswith("-0."):
        s = "-." + s[3:]
    elif s.startswith("0."):
        s = "." + s[2:]
    return s if s not in ("", "-") else "0"


# ---------------------------------------------------------------------------
# Data loading: cache the full fixrsvp payload per-session
# ---------------------------------------------------------------------------
def _fixrsvp_cache_path(subject, date):
    return CACHE_FIG_DIR / f"{subject}_{date}_fixrsvp.pkl"


def _load_fixrsvp_data(subject, date, refresh=False):
    """Load the fixrsvp payload for a session. First call routes through
    ``eval.fixrsvp.get_fixrsvp_data``; subsequent calls hit the local pickle."""
    cache = _fixrsvp_cache_path(subject, date)
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    from eval.fixrsvp import get_fixrsvp_data
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
# Peak lag from the shared STE cache (population median)
# ---------------------------------------------------------------------------
def _population_peak_lag(session_name, robs=None, recalc=False):
    arrs = compute_sta_ste(session_name, recalc=recalc)
    if arrs is not None:
        return population_peak_lag(arrs["stes"])
    if robs is not None:
        psth = np.nanmean(robs, axis=(0, 2))
        return int(np.nanargmax(psth))
    raise RuntimeError(f"No STA/STE for {session_name} and no robs to fall back on")


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


def _compute_panel_payload(subject, date, combo_idx=COMBO_IDX, recalc=False):
    data = _load_fixrsvp_data(subject, date, refresh=recalc)
    robs = data["robs"]
    eyepos = data["eyepos"]
    session = f"{subject}_{date}"
    peak_lag = _population_peak_lag(session, robs=robs, recalc=recalc)

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


def load_panel_payload(subject=SUBJECT, date=DATE, refresh=None):
    if refresh is None:
        refresh = RECALC
    path = _payload_cache_path(subject, date)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    payload = _compute_panel_payload(subject, date, recalc=refresh)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _ordered_cluster_trials(iix, clusters):
    """Return (c0_trials, c1_trials) in the display order used by the raster."""
    c0 = [iix[i] for i in range(len(iix)) if clusters[i] == 0]
    c1 = [iix[i] for i in range(len(iix)) if clusters[i] == 1]
    return c0, c1


def _apply_drop_rows(c0_trials, c1_trials, drop_rows=DROP_DISPLAY_ROWS):
    """Drop trials at the given 1-indexed display positions in the combined
    cluster-0-then-cluster-1 ordering."""
    combined = list(c0_trials) + list(c1_trials)
    drop = {r - 1 for r in drop_rows}
    n0 = len(c0_trials)
    c0_kept = [t for i, t in enumerate(combined[:n0]) if i not in drop]
    c1_kept = [t for i, t in enumerate(combined[n0:], start=n0) if i not in drop]
    return c0_kept, c1_kept


def _gaze_segment_values(payload, c0_trials, c1_trials,
                         highlight_ms=RASTER_WINDOW_MS):
    """Concatenated highlighted gaze values for each dim, across all selected
    trials. Highlight is the raster display window. Returns (vals_dim0, vals_dim1)."""
    eye = payload["eyepos"]
    seg_s = int(round(highlight_ms[0] / 1000.0 / DT))
    seg_e = int(round(highlight_ms[1] / 1000.0 / DT))
    out = []
    for dim in (0, 1):
        chunks = []
        for trials in (c0_trials, c1_trials):
            for tid in trials:
                t = eye[tid, seg_s:seg_e, dim]
                chunks.append(t[~np.isnan(t)])
        out.append(np.concatenate(chunks) if chunks else np.array([]))
    return out[0], out[1]


def compute_shared_gaze_ylim(payload, c0_trials, c1_trials, pad_factor=2.0,
                             min_half=0.25, highlight_ms=RASTER_WINDOW_MS):
    """Symmetric matched y-limits for both gaze axes, centered on the joint
    highlighted-segment midpoint."""
    v0, v1 = _gaze_segment_values(payload, c0_trials, c1_trials,
                                   highlight_ms=highlight_ms)
    both = np.concatenate([v0, v1]) if (v0.size or v1.size) else np.array([0.0])
    lo, hi = float(np.min(both)), float(np.max(both))
    center = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo) * pad_factor, min_half)
    return center - half, center + half


def plot_gaze_axis(ax, payload, c0_trials, c1_trials, dim,
                   window_ms=GAZE_WINDOW_MS,
                   highlight_ms=RASTER_WINDOW_MS):
    """Plot one gaze dimension over time for the selected trials.
    Full window in gray; portion inside ``highlight_ms`` in cluster colors."""
    eye = payload["eyepos"]

    win_start_bin = int(round(window_ms[0] / 1000.0 / DT))
    win_end_bin = int(round(window_ms[1] / 1000.0 / DT))
    n_bins = win_end_bin - win_start_bin
    t_ms = (win_start_bin + np.arange(n_bins)) * DT * 1000.0

    seg_s = int(round(highlight_ms[0] / 1000.0 / DT))
    seg_e = int(round(highlight_ms[1] / 1000.0 / DT))
    s_idx = max(seg_s - win_start_bin, 0)
    e_idx = min(seg_e - win_start_bin, n_bins)

    for trials, color in [(c0_trials, GROUP_BLUE), (c1_trials, GROUP_RED)]:
        for tid in trials:
            trace = eye[tid, win_start_bin:win_end_bin, dim]
            ax.plot(t_ms, trace, color="0.72", lw=0.45, alpha=0.45)
            if e_idx > s_idx:
                ax.plot(t_ms[s_idx:e_idx], trace[s_idx:e_idx],
                        color=color, lw=0.85, alpha=0.78)

    ax.set_xlim(window_ms[0], window_ms[1])
    label = "Az." if dim == 0 else "El."
    ax.set_ylabel("")
    ax.text(
        0.018, 0.96, f"{label} (°)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="0.1",
    )
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_yticklabels([_fmt_deg_tick(-0.5), "0", _fmt_deg_tick(0.5)])
    # Ticks out: panel H's time axis is shared with the population PSTH (J).
    ax.tick_params(direction="out", length=3, width=0.8, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
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


def _cluster_psth(payload, trials, window_ms, bin_ms=PSTH_BIN_MS):
    """Mean population spike rate (spikes/s/cell) across trials, binned."""
    spike_times = payload["spike_times_trials"]
    t_bins = payload["trial_t_bins"]
    n_cells = payload["robs"].shape[2]
    t0_bin = int(round(window_ms[0] / 1000.0 / DT))
    t1_bin = int(round(window_ms[1] / 1000.0 / DT))
    win_start_s = t0_bin * DT
    win_end_s = t1_bin * DT
    bin_s = bin_ms / 1000.0
    edges = np.arange(win_start_s, win_end_s + bin_s / 2, bin_s)
    centers_ms = 0.5 * (edges[:-1] + edges[1:]) * 1000.0
    if not trials:
        return centers_ms, np.zeros(len(edges) - 1)
    rows = []
    for tid in trials:
        ts, _ = _trial_spike_times_in_window(
            spike_times[tid], t_bins[tid], t0_bin, t1_bin,
        )
        counts, _ = np.histogram(ts, bins=edges)
        rows.append(counts)
    rate = np.mean(rows, axis=0) / bin_s / n_cells
    return centers_ms, rate


def _trial_psth_rows(payload, trials, window_ms, bin_ms=PSTH_BIN_MS):
    """Per-trial population spike-rate rows for mean/SEM summaries."""
    spike_times = payload["spike_times_trials"]
    t_bins = payload["trial_t_bins"]
    n_cells = payload["robs"].shape[2]
    t0_bin = int(round(window_ms[0] / 1000.0 / DT))
    t1_bin = int(round(window_ms[1] / 1000.0 / DT))
    win_start_s = t0_bin * DT
    win_end_s = t1_bin * DT
    bin_s = bin_ms / 1000.0
    edges = np.arange(win_start_s, win_end_s + bin_s / 2, bin_s)
    centers_ms = 0.5 * (edges[:-1] + edges[1:]) * 1000.0
    rows = []
    for tid in trials:
        ts, _ = _trial_spike_times_in_window(
            spike_times[tid], t_bins[tid], t0_bin, t1_bin,
        )
        counts, _ = np.histogram(ts, bins=edges)
        rows.append(counts / bin_s / n_cells)
    if not rows:
        return centers_ms, np.empty((0, len(edges) - 1))
    return centers_ms, np.asarray(rows)


def plot_psth_axis(ax, payload, c0_trials, c1_trials,
                   window_ms=RASTER_WINDOW_MS, bin_ms=PSTH_BIN_MS):
    """Line-plot PSTHs for the two cluster groups."""
    x0, p0 = _cluster_psth(payload, c0_trials, window_ms, bin_ms=bin_ms)
    x1, p1 = _cluster_psth(payload, c1_trials, window_ms, bin_ms=bin_ms)
    x_all, rows = _trial_psth_rows(
        payload, list(c0_trials) + list(c1_trials), window_ms, bin_ms=bin_ms,
    )
    if rows.size:
        mean = np.nanmean(rows, axis=0)
        # Overall (eye-position-agnostic) PSTH removed per PI feedback: it
        # obscures the blue/red difference. Uncomment to restore.
        # ax.plot(x_all, mean, color="0.6", lw=0.75, alpha=0.95, zorder=2)
    ax.plot(x0, p0, color=PSTH_BLUE, lw=1.35, alpha=0.95, zorder=3)
    ax.plot(x1, p1, color=PSTH_RED, lw=1.35, alpha=0.95, zorder=3)

    ax.set_xlim(window_ms[0], window_ms[1])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Spikes/s", fontsize=8)
    ax.tick_params(direction="in", length=3, width=0.8, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def plot_raster_axis(ax, payload, c0_trials=None, c1_trials=None,
                    window_ms=RASTER_WINDOW_MS,
                    gap=RASTER_GAP_BINS,
                    group_gap=RASTER_GROUP_GAP_BINS,
                    scale_units=75):
    """Cluster-grouped population raster with dashed trial dividers.

    ``scale_units`` draws a vertical scale bar at the bottom-right marking that
    many units (rows) within a trial block; pass None to omit it."""
    robs = payload["robs"]
    spike_times = payload["spike_times_trials"]
    t_bins = payload["trial_t_bins"]
    t0_bin = int(round(window_ms[0] / 1000.0 / DT))
    t1_bin = int(round(window_ms[1] / 1000.0 / DT))
    n_cells = robs.shape[2]

    if c0_trials is None or c1_trials is None:
        c0_default, c1_default = _ordered_cluster_trials(
            payload["iix"], payload["clusters"]
        )
        if c0_trials is None:
            c0_trials = c0_default
        if c1_trials is None:
            c1_trials = c1_default
    n0, n1 = len(c0_trials), len(c1_trials)

    block_h = n_cells + gap
    row_c0_start = 0
    row_c1_start = n0 * block_h + group_gap
    total_rows = row_c1_start + n1 * block_h - gap

    t0_ms = t0_bin * DT * 1000.0
    t1_ms = t1_bin * DT * 1000.0

    blue_group_bottom = max(n0 * block_h - gap, 0)
    red_group_top = row_c1_start
    if n0 > 0:
        ax.axhspan(0, blue_group_bottom, color=GROUP_BLUE_FILL,
                   alpha=GROUP_FILL_ALPHA, lw=0, zorder=0)
    if n1 > 0:
        ax.axhspan(red_group_top, total_rows, color=GROUP_RED_FILL,
                   alpha=GROUP_FILL_ALPHA, lw=0, zorder=0)
    # Expose the blue/red region spans (raster data coords) so the gaze->raster
    # arrows can land on each region's centre after final layout.
    ax._gaze_regions = {
        "blue": (0.0, float(blue_group_bottom)) if n0 > 0 else None,
        "red": (float(red_group_top), float(total_rows)) if n1 > 0 else None,
    }

    spike_xs, spike_ys, spike_y2s = [], [], []
    tick_positions, tick_labels, tick_colors = [], [], []

    def _add_trial(trial_id, row_top, cluster_label, trial_number):
        times_s, cells = _trial_spike_times_in_window(
            spike_times[trial_id], t_bins[trial_id], t0_bin, t1_bin,
        )
        if times_s.size:
            x_ms = times_s * 1000.0
            for xt, cell in zip(x_ms, cells):
                spike_xs.append(xt)
                spike_ys.append(row_top + cell)
                spike_y2s.append(row_top + cell + 0.7)
        tick_positions.append(row_top + n_cells / 2)
        tick_labels.append(str(trial_number))
        tick_colors.append(GROUP_BLUE if cluster_label == 0 else GROUP_RED)

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
        ax.plot(seg_x, seg_y, color="k", lw=1.2, rasterized=True)

    # Dashed trial dividers (between trial blocks within each cluster).
    for i in range(1, n0):
        ax.axhline(i * block_h - gap / 2, color="0.6",
                   lw=0.4, ls="--", zorder=0)
    for i in range(1, n1):
        ax.axhline(row_c1_start + i * block_h - gap / 2, color="0.6",
                   lw=0.4, ls="--", zorder=0)
    # Solid black divider centered in the small gap between trajectory groups,
    # marking the boundary between the blue and red sections.
    divider_y = None
    if n0 > 0 and n1 > 0:
        divider_y = n0 * block_h - gap / 2 + group_gap / 2
        ax.axhline(divider_y, color="k", lw=1.0, ls="-", zorder=5)

    ax.set_xlim(t0_ms, t1_ms)
    ax.set_ylim(total_rows, 0)
    ax.set_xlabel("Time from fixation onset (ms)", fontsize=8)
    ax.set_ylabel("Trials, grouped by gaze trajectory", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    for lbl, col in zip(ax.get_yticklabels(), tick_colors):
        lbl.set_color(col)
    ax.tick_params(direction="in", length=3, width=0.8, labelsize=7)
    # No bottom spine and no bottom x ticks; instead place the x tick marks on
    # the black group divider, aligned to panel J's tick positions below.
    ax.tick_params(axis="x", which="both", bottom=False)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    if divider_y is not None:
        af = 1.0 - divider_y / total_rows       # divider in axes fraction
        th = 0.012                              # half tick length (axes frac)
        xtrans = ax.get_xaxis_transform()       # x in data, y in axes frac
        for xt in ax.get_xticks():
            if t0_ms <= xt <= t1_ms:
                ax.plot([xt, xt], [af - th, af + th], transform=xtrans,
                        color="k", lw=0.8, zorder=6, clip_on=False)

    # Vertical scale bar (within-trial units) just outside the right edge, its
    # bottom aligned with the bottom of the plot. It sits off the data, so it's
    # plain black with the label rotated alongside.
    if scale_units:
        xr = t1_ms - t0_ms
        x_bar = t1_ms + 0.0175 * xr
        y_bot = total_rows
        y_top = y_bot - scale_units
        ax.plot([x_bar, x_bar], [y_bot, y_top], color="k", lw=2.2,
                solid_capstyle="butt", zorder=7, clip_on=False)
        # Left-justified in the rotated frame: the first character sits at the
        # bottom of the panel (y_bot), reading upward alongside the bar.
        ax.text(x_bar + 0.03 * xr, y_bot,
                f"{scale_units} units", rotation=90, rotation_mode="anchor",
                ha="left", va="center", fontsize=7, color="k", zorder=7,
                clip_on=False)
    return ax


def _add_block_label(ax, letter, dx=-22, dy=6):
    ax.annotate(
        letter, xy=(0, 1), xycoords="axes fraction",
        xytext=(dx, dy), textcoords="offset points",
        fontsize=PANEL_LABEL_FONTSIZE_PT, fontweight="bold",
        va="bottom", ha="left", annotation_clip=False,
    )


def _corner_arrow(fig, ax_src, ax_dst, src_xy_data, dst_xy_data, color,
                  corridor_offset_pts=16.0):
    """Rounded right-angle connector (right -> vertical -> left) from a data
    point in ``ax_src`` to a data point in ``ax_dst``, matching the fig2 panel
    a->b arrow style. The path detours through a vertical corridor just right
    of both panels; the arrow head lands on the destination point.

    Built in figure coordinates against the *current* axes positions, so call
    after the final layout (a fresh ``fig.canvas.draw()``).
    """
    inv = fig.transFigure.inverted()
    off_px = corridor_offset_pts * fig.dpi / 72.0
    src_disp = ax_src.transData.transform(src_xy_data)
    dst_disp = ax_dst.transData.transform(dst_xy_data)
    right_src = ax_src.transAxes.transform((1.0, 0.0))[0]
    right_dst = ax_dst.transAxes.transform((1.0, 0.0))[0]
    x_corr_disp = max(right_src, right_dst) + off_px

    sx, sy = inv.transform(src_disp)
    ex, ey = inv.transform(dst_disp)
    xc = inv.transform((x_corr_disp, 0.0))[0]

    sign_v = 1.0 if ey >= sy else -1.0
    # Corner radius in figure-fraction; cap to avoid overshooting any leg.
    r = min(0.008, abs(xc - sx) / 2.5, abs(ey - sy) / 2.0, abs(xc - ex) / 2.5)
    verts = [
        (sx, sy),
        (xc - r, sy),
        (xc, sy),                   # CURVE3 control
        (xc, sy + sign_v * r),      # CURVE3 end (turn vertical)
        (xc, ey - sign_v * r),
        (xc, ey),                   # CURVE3 control
        (xc - r, ey),               # CURVE3 end (turn left)
        (ex, ey),
    ]
    codes = [
        Path.MOVETO, Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
    ]
    arrow = FancyArrowPatch(
        path=Path(verts, codes), transform=fig.transFigure,
        arrowstyle="->", mutation_scale=12, lw=1.3, color=color,
        clip_on=False, zorder=8,
    )
    fig.patches.append(arrow)


def draw_gaze_raster_arrows(fig):
    """Draw the two gaze->raster arrows recorded on ``fig`` by ``plot_panel_f``.

    Deferred so the figure-coordinate paths are built against the final axes
    positions (the composite figure repositions every panel after plotting)."""
    data = getattr(fig, "_fig1_arrow_data", None)
    if data is None:
        return
    ax_gaze = data["ax_gaze"]
    ax_raster = data["ax_raster"]
    regions = getattr(ax_raster, "_gaze_regions", None) or {}
    fig.canvas.draw()
    x_src = ax_gaze.get_xlim()[1]          # right edge of the Az gaze plot
    # Heads land at each region's vertical centre but at the panel's right edge
    # (same x the tails start from), so they sit to the right and don't cover
    # the raster. Blue's corridor is pulled left of red's so the two vertical
    # legs don't overlap.
    x_dst = ax_raster.get_xlim()[1]
    links = [
        (GROUP_BLUE, data.get("blue_end_y"), regions.get("blue"), 11.0),
        (GROUP_RED, data.get("red_end_y"), regions.get("red"), 21.0),
    ]
    for color, src_y, region, corridor_pts in links:
        if src_y is None or region is None:
            continue
        y_dst = 0.5 * (region[0] + region[1])
        _corner_arrow(
            fig, ax_gaze, ax_raster,
            (x_src, float(src_y)), (x_dst, y_dst), color,
            corridor_offset_pts=corridor_pts,
        )


def plot_panel_f(fig=None, subject=SUBJECT, date=DATE, refresh=False,
                 panel_letters=("G", "H", "I"), bottom_pad=0.0,
                 raster_height=2.45, gaze_height=1.25, psth_height=0.55):
    payload = load_panel_payload(subject, date, refresh=refresh)

    c0_all, c1_all = _ordered_cluster_trials(payload["iix"], payload["clusters"])
    c0_trials, c1_trials = _apply_drop_rows(c0_all, c1_all)

    if fig is None:
        fig = plt.figure(figsize=(4, 6), constrained_layout=True)
        fig.set_constrained_layout_pads(h_pad=0.02, hspace=0.0)

    if bottom_pad:
        outer = fig.add_gridspec(
            4, 1,
            height_ratios=[gaze_height, raster_height, psth_height, bottom_pad],
            hspace=-0.06,
        )
    else:
        outer = fig.add_gridspec(
            3, 1, height_ratios=[gaze_height, raster_height, psth_height],
            hspace=-0.06,
        )
    gaze_gs = outer[0].subgridspec(2, 1, hspace=-0.04)
    ax_h = fig.add_subplot(gaze_gs[0])
    ax_v = fig.add_subplot(gaze_gs[1], sharex=ax_h)
    ax_raster = fig.add_subplot(outer[1])
    ax_psth = fig.add_subplot(outer[2], sharex=ax_raster)

    plot_gaze_axis(ax_h, payload, c0_trials, c1_trials, dim=0)
    plot_gaze_axis(ax_v, payload, c0_trials, c1_trials, dim=1)
    gaze_ylim = (-GAZE_YLIM_HALF, GAZE_YLIM_HALF)
    ax_h.set_ylim(gaze_ylim)
    ax_v.set_ylim(gaze_ylim)
    plot_raster_axis(ax_raster, payload,
                     c0_trials=c0_trials, c1_trials=c1_trials,
                     window_ms=RASTER_WINDOW_MS)
    ax_raster.text(
        0.03, 1.035, "...can result in very different population responses",
        transform=ax_raster.transAxes, ha="left", va="bottom",
        fontsize=8.5, color="0.10", clip_on=False,
    )
    ax_raster.text(
        1.0, 1.34, "Small differences in eye trajectories...",
        transform=ax_raster.transAxes, ha="right", va="bottom",
        fontsize=8.5, color="0.22", clip_on=False,
    )
    plot_psth_axis(ax_psth, payload, c0_trials, c1_trials)

    # Gaze pair: top plot has no bottom spine or x ticks; bottom plot keeps
    # only left + bottom spines.
    ax_h.spines["bottom"].set_visible(False)
    ax_h.tick_params(bottom=False, labelbottom=False)
    ax_v.tick_params(top=False, labeltop=False, labelbottom=False)

    # PSTH summarizes the raster above, so keep the time labels only at the
    # bottom of this stack.
    ax_raster.tick_params(labelbottom=False)
    ax_raster.set_xlabel("")
    ax_psth.tick_params(labelbottom=True)

    # Record the geometry the gaze->raster arrows need: the mean right-end y of
    # the blue (c0) and red (c1) Az traces. The arrows themselves are drawn by
    # draw_gaze_raster_arrows() after the final layout.
    eye = payload["eyepos"]
    win_start_bin = int(round(GAZE_WINDOW_MS[0] / 1000.0 / DT))
    win_end_bin = int(round(GAZE_WINDOW_MS[1] / 1000.0 / DT))

    def _mean_trace_end(trials):
        ends = []
        for tid in trials:
            tr = eye[tid, win_start_bin:win_end_bin, 0]
            finite = tr[~np.isnan(tr)]
            if finite.size:
                ends.append(float(finite[-1]))
        return float(np.mean(ends)) if ends else None

    fig._fig1_arrow_data = {
        "ax_gaze": ax_h, "ax_raster": ax_raster,
        "blue_end_y": _mean_trace_end(c0_trials),
        "red_end_y": _mean_trace_end(c1_trials),
    }

    if panel_letters is not None:
        _add_block_label(ax_h, panel_letters[0])
        _add_block_label(ax_psth, panel_letters[1])
        _add_block_label(ax_raster, panel_letters[2])

    return fig, {"gaze_h": ax_h, "gaze_v": ax_v,
                 "psth": ax_psth, "raster": ax_raster}


if __name__ == "__main__":
    fig, axes = plot_panel_f()
    draw_gaze_raster_arrows(fig)
    out = FIG_DIR / "fig1f_population.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
