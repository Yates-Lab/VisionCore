"""
Figure 1 panel D: single-cell raster structure driven by gaze location.

Trials are split into short time segments. Within each segment, the
per-trial median gaze is projected onto the axis orthogonal to the cell's
preferred grating orientation and trials are sorted by that projection.
The raster stitches the segments along time, each with its own sort order;
trials are placed by rank (evenly spaced) so spike ticks do not overlap.

Two side-by-side axes:
    (left)  gaze trajectories inside a single example segment (the one
            with the strongest mean response), colored by projection.
    (right) gaze-sorted, segment-stitched spike-time raster, with PSTHs
            split by projection-quantile overlaid.

Reuses ``tejas.rsvp_util.get_fixrsvp_data`` for the heavy data loading
(cached on disk by that module). The per-cell payload is itself cached
to ``CACHE_DIR/fig1_single_cell`` so reruns do no data work at all.

Usage:
    uv run ryan/fig1/generate_fig1d.py
"""

from pathlib import Path
import pickle
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR, CACHE_DIR

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBJECT = "Allen"
DATE = "2022-03-04"
DEFAULT_CELL = 149            # best example from fig1_fixrsvp_single_cell.py
DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)

DT = 1.0 / 240.0              # bin size, seconds
TOTAL_WINDOW_BINS = (0, 100)  # raster x-extent, bins
SEGMENT_LEN_BINS = 25         # gaze sort + stitching segment length
DISTANCE_FROM_LINE_THRESHOLD = 0.3   # degrees
MICROSACCADE_THRESHOLD = 0.3         # degrees
USE_UNIVERSAL_PEAK_LAG = True        # population-median peak lag for sort

CACHE_FIG_DIR = CACHE_DIR / "fig1_single_cell"
RF_CACHE_DIR = CACHE_DIR / "fig1_rf_contours"   # written by generate_fig1c.py
FIG_DIR = FIGURES_DIR / "fig1"
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Peak lag from cached STEs (fig1c artifact); orientation from gratings
# ---------------------------------------------------------------------------
def _load_ste_for_session(session_name):
    path = RF_CACHE_DIR / f"{session_name}_sta_ste.npz"
    if not path.exists():
        return None
    return np.load(path)


def _peak_lag_from_ste(ste_cell):
    return int(ste_cell.std(axis=(1, 2)).argmax())


def _gratings_cache_path(session_name):
    return CACHE_FIG_DIR / f"{session_name}_gratings.npz"


def _compute_gratings_for_session(subject, date):
    """Run the standard gratings analysis on the session's gratings.dset
    (loaded via YatesV1Session). Cache the orientation tuning + peak indices
    per-session so this is paid at most once per session.
    """
    session_name = f"{subject}_{date}"
    cache = _gratings_cache_path(session_name)
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    from DataYatesV1.utils.io import YatesV1Session
    from DataYatesV1.utils.data.filtering import get_valid_dfs
    from eval.gratings_analysis import gratings_analysis

    sess = YatesV1Session(session_name)
    dset = sess.get_dataset("gratings")
    if dset is None:
        raise RuntimeError(f"No gratings dataset for {session_name}")

    n_lags = 20
    dt = 1.0 / 240.0
    dset["dfs"] = get_valid_dfs(dset, n_lags)

    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    robs = _np(dset["robs"])
    sf = _np(dset["sf"]).squeeze()
    ori = _np(dset["ori"]).squeeze()
    phases = _np(dset["stim_phase"])
    if phases.ndim == 3:
        phases = phases[:, phases.shape[1] // 2, phases.shape[2] // 2]
    dfs = _np(dset["dfs"]).squeeze()

    res = gratings_analysis(
        robs=robs, sf=sf, ori=ori, phases=phases, dt=dt,
        n_lags=n_lags, dfs=dfs, min_spikes=30,
    )
    cids = list(dset.metadata.get("cids", np.arange(robs.shape[1])))

    payload = {
        "oris": np.asarray(res["oris"], dtype=np.float64),
        "ori_tuning": np.asarray(res["ori_tuning"], dtype=np.float64),
        "peak_ori": np.asarray(res["peak_ori"], dtype=np.float64),
        "peak_ori_idx": np.asarray(res["peak_ori_idx"], dtype=np.int64),
        "ori_snr": np.asarray(res["ori_snr"], dtype=np.float64),
        "cids": np.asarray(cids),
    }
    np.savez(cache, **payload)
    return payload


# ---------------------------------------------------------------------------
# Gaze sort: project trial-median eye position onto axis orthogonal to ori
# ---------------------------------------------------------------------------
def _microsaccade_present(trial_eyepos, threshold=MICROSACCADE_THRESHOLD):
    med = np.nanmedian(trial_eyepos, axis=0)
    d = np.hypot(trial_eyepos[:, 0] - med[0], trial_eyepos[:, 1] - med[1])
    return np.any(d > threshold)


def _project_onto_orthogonal_line(eyepos, sort_window, max_orientation, peak_lag,
                                  distance_threshold=DISTANCE_FROM_LINE_THRESHOLD):
    """For each trial, take median gaze inside the (peak-lag-shifted) sort
    window, drop trials with a microsaccade, drop trials too far from the
    line, and project the rest onto the line orthogonal to ``max_orientation``
    passing through the across-trial centroid.

    Returns (valid_indices, distances_along_line) sorted by projection.
    """
    s, e = sort_window
    win_len = e - s
    s_shift = max(s - peak_lag, 0)
    e_shift = s_shift + win_len

    win = eyepos[:, s_shift:e_shift, :]
    cx = np.nanmedian(win[..., 0])
    cy = np.nanmedian(win[..., 1])

    ortho = max_orientation + 90.0
    slope = np.tan(np.deg2rad(ortho))
    intercept = cy - slope * cx
    norm = np.sqrt(1 + slope ** 2)

    valid, projections = [], []
    for i in range(eyepos.shape[0]):
        trace = eyepos[i, s_shift:e_shift, :]
        if np.isnan(trace).all():
            continue
        if _microsaccade_present(trace):
            continue
        med = np.nanmedian(trace, axis=0)
        d = abs(slope * med[0] - med[1] + intercept) / norm
        if d >= distance_threshold:
            continue
        x_proj = (med[0] + slope * (med[1] - intercept)) / (1 + slope ** 2)
        valid.append(i)
        projections.append(x_proj)

    if not valid:
        return np.array([], dtype=int), np.array([])
    order = np.argsort(projections)
    iix = np.array(valid)[order]
    proj_sorted = np.array(projections)[order]
    distances = (proj_sorted - proj_sorted[0]) * norm
    return iix, distances


# ---------------------------------------------------------------------------
# Cached per-cell payload
# ---------------------------------------------------------------------------
def _cell_cache_path(subject, date, cell):
    return CACHE_FIG_DIR / f"{subject}_{date}_cell{cell}.pkl"


def _compute_segments(eyepos, max_orientation, peak_lag,
                     total_window=TOTAL_WINDOW_BINS, seg_len=SEGMENT_LEN_BINS):
    """For each contiguous time segment inside ``total_window``, sort trials
    by gaze projection. Returns list of dicts with start/end bins, iix,
    distances."""
    s0, e0 = total_window
    segments = []
    for start in range(s0, e0, seg_len):
        end = min(start + seg_len, e0)
        iix, dist = _project_onto_orthogonal_line(
            eyepos, (start, end), max_orientation, peak_lag,
        )
        segments.append({"start": int(start), "end": int(end),
                         "iix": iix, "distances": dist})
    return segments


def _compute_cell_payload(subject, date, cell, max_orientation=None):
    """Load fixrsvp data, derive orientation + peak lag from the cached STE,
    compute the gaze sort, and slice per-trial spike times.

    ``max_orientation`` (degrees) may be supplied to override the
    STE-derived preferred orientation.
    """
    from tejas.rsvp_util import get_fixrsvp_data

    data = get_fixrsvp_data(
        subject, date, DATASET_CONFIGS_PATH,
        use_cached_data=True,
        salvageable_mismatch_time_threshold=25,
        verbose=False,
    )
    cids = list(data["cids"])
    if cell not in cids:
        raise ValueError(f"cell {cell} not in cids for {subject}_{date}")
    cell_col = cids.index(cell)

    eyepos = data["eyepos"]
    robs_cell = data["robs"][:, :, cell_col]
    spike_times = [
        np.asarray(data["spike_times_trials"][t][cell_col])
        for t in range(len(data["spike_times_trials"]))
    ]
    trial_t_bins = data["trial_t_bins"]

    session_name = f"{subject}_{date}"

    # --- preferred orientation from real gratings analysis (cached) -------
    if max_orientation is None:
        gratings = _compute_gratings_for_session(subject, date)
        gratings_cids = list(gratings["cids"])
        if cell in gratings_cids:
            row = gratings_cids.index(cell)
        else:
            row = cell_col  # assume positional alignment
        max_orientation = float(gratings["peak_ori"][row])
    max_orientation = float(max_orientation)

    # --- peak lag from cached STEs (fig1c artifact) -----------------------
    ste_npz = _load_ste_for_session(session_name)
    if ste_npz is None:
        psth = np.nanmean(robs_cell, axis=0)
        peak_lag_cell = int(np.nanargmax(psth))
        peak_lag = peak_lag_cell
    else:
        stes_all = ste_npz["stes"]
        peak_lag_cell = _peak_lag_from_ste(stes_all[cell_col])
        if USE_UNIVERSAL_PEAK_LAG:
            lags = [int(stes_all[u].std((1, 2)).argmax())
                    for u in range(stes_all.shape[0])]
            peak_lag = int(np.median(lags))
        else:
            peak_lag = peak_lag_cell

    segments = _compute_segments(eyepos, max_orientation, peak_lag)

    # Example segment for the left (gaze) axis = segment with strongest
    # mean response across all trials.
    seg_means = [
        np.nanmean(robs_cell[:, s["start"]:s["end"]]) for s in segments
    ]
    example_idx = int(np.nanargmax(seg_means)) if seg_means else 0

    return {
        "cell": int(cell),
        "session": f"{subject}_{date}",
        "max_orientation": float(max_orientation),
        "peak_lag": int(peak_lag),
        "total_window": np.asarray(TOTAL_WINDOW_BINS, dtype=int),
        "segment_len": int(SEGMENT_LEN_BINS),
        "segments": segments,
        "example_segment_idx": example_idx,
        "eyepos_all": eyepos,
        "spike_times_all": spike_times,
        "trial_t_bins_all": trial_t_bins,
        "robs_cell_all": robs_cell,
    }


def load_cell_payload(subject=SUBJECT, date=DATE, cell=DEFAULT_CELL, refresh=False):
    path = _cell_cache_path(subject, date, cell)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    payload = _compute_cell_payload(subject, date, cell)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_eyepos_axis(ax, payload, segment_idx=None):
    """Left axis: gaze traces inside one example segment, colored by the
    sort projection used for that segment's raster column."""
    seg_i = payload["example_segment_idx"] if segment_idx is None else segment_idx
    seg = payload["segments"][seg_i]
    iix = seg["iix"]
    peak_lag = int(payload["peak_lag"])
    eye_all = payload["eyepos_all"]
    max_orientation = float(payload["max_orientation"])

    win_len = seg["end"] - seg["start"]
    s_shift = max(seg["start"] - peak_lag, 0)
    e_shift = s_shift + win_len
    eye = eye_all[iix]                                # iix-ordered (low→high proj)

    win = eye[:, s_shift:e_shift, :]
    cx = np.nanmedian(win[..., 0])
    cy = np.nanmedian(win[..., 1])
    ortho = max_orientation + 90.0
    slope = np.tan(np.deg2rad(ortho))
    length = 1.0

    ax.plot(
        [cx - length / 2, cx + length / 2],
        [cy - length / 2 * slope, cy + length / 2 * slope],
        "k", lw=1.0,
    )

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(iix)))
    for idx in range(len(iix)):
        trace = eye[idx, s_shift:e_shift, :]
        med = np.nanmedian(trace, axis=0)
        ax.plot(trace[:, 0], trace[:, 1], color=colors[idx], lw=0.4, alpha=0.7)
        ax.scatter(med[0], med[1], s=6, color=colors[idx],
                   edgecolor="k", linewidth=0.3, zorder=3)
        if idx % 10 == 0:
            ax.text(med[0], med[1], str(idx), color="k", fontsize=6,
                    ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    zorder=10)

    # Axes
    xy_max = np.nanmax(np.abs(eye[:, s_shift:e_shift, :]))
    pad = 0.05
    ax.set_xlim(-xy_max - pad, xy_max + pad)
    ax.set_ylim(-xy_max - pad, xy_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    return ax


def _segment_raster_lines(spike_times_list, trial_t_bins_list, trial_indices,
                         seg_start_bin, seg_end_bin, y_positions,
                         dt=DT, height=0.7):
    """Spike ticks for one stitched segment. ``spike_times_list`` /
    ``trial_t_bins_list`` are indexed by GLOBAL trial id; ``trial_indices``
    selects which trials (in iix sort order) to draw; ``y_positions[k]``
    gives the row for ``trial_indices[k]``."""
    seg_start_s = seg_start_bin * dt
    seg_end_s = seg_end_bin * dt
    xs, ys = [], []
    for k, trial_i in enumerate(trial_indices):
        spikes = np.atleast_1d(np.asarray(spike_times_list[trial_i]))
        if spikes.size == 0:
            continue
        t_bins = np.asarray(trial_t_bins_list[trial_i])
        t_bins = t_bins[~np.isnan(t_bins)]
        if t_bins.size == 0:
            continue
        t0 = t_bins[0] - dt / 2          # fixation-onset time, absolute
        rel = spikes - t0                 # seconds since fixation onset
        mask = (rel >= seg_start_s) & (rel < seg_end_s)
        if not np.any(mask):
            continue
        rel_ms = rel[mask] * 1000.0
        y0 = y_positions[k]
        for x in rel_ms:
            xs.extend([x, x, np.nan])
            ys.extend([y0, y0 + height, np.nan])
    return np.asarray(xs), np.asarray(ys)


def plot_raster_axis(ax, payload, n_psth=2, tick_height=0.7, tick_lw=0.8,
                    show_segment_dividers=True):
    """Right axis: gaze-sorted, segment-stitched raster.

    Each segment is sorted independently by trial-median gaze projection.
    Y axis is *rank*, not distance — trials are evenly spaced so spike ticks
    don't superimpose. A secondary right axis shows the projection-distance
    range for the example segment.
    """
    segments = payload["segments"]
    spike_times = payload["spike_times_all"]
    t_bins = payload["trial_t_bins_all"]
    total_start_bin, total_end_bin = payload["total_window"]
    total_dur_ms = (total_end_bin - total_start_bin) * DT * 1000.0
    dt = DT

    n_rows = max((len(s["iix"]) for s in segments), default=0)
    if n_rows == 0:
        ax.set_title("no valid trials")
        return ax

    cmap = plt.cm.coolwarm
    all_xs, all_ys = [], []
    for seg in segments:
        iix = seg["iix"]
        n = len(iix)
        if n == 0:
            continue
        # Rank-based y in [0, n_rows-1] — evenly spaced, regardless of how
        # many trials this segment retained.
        if n == 1:
            y_pos = np.array([0.5 * (n_rows - 1)])
        else:
            y_pos = np.linspace(0, n_rows - 1, n)
        xs, ys = _segment_raster_lines(
            spike_times, t_bins, iix,
            seg["start"], seg["end"], y_pos, dt=dt, height=tick_height,
        )
        if xs.size:
            all_xs.append(xs); all_ys.append(ys)

    if all_xs:
        ax.plot(np.concatenate(all_xs), np.concatenate(all_ys),
                color="k", lw=tick_lw, rasterized=True)

    if show_segment_dividers:
        for seg in segments[1:]:
            x_ms = seg["start"] * dt * 1000.0
            ax.axvline(x_ms, color="0.7", lw=0.5, ls="--", zorder=0)

    ax.set_ylim(n_rows, 0)
    ax.set_xlim(0, total_dur_ms)
    ax.set_xlabel("Time from fixation onset (ms)")
    ax.set_ylabel("Trial (sorted by gaze, per-segment)")

    # PSTH overlay per segment: split iix into n_psth quantile groups, plot
    # the within-group mean response over the segment's time slice.
    if n_psth and n_psth >= 2:
        robs_all = payload["robs_cell_all"]
        # Common psth_max across segments for consistent scaling.
        all_psths_for_scale = []
        for seg in segments:
            iix = seg["iix"]
            if len(iix) == 0:
                continue
            group_size = max(1, int(np.ceil(len(iix) / n_psth)))
            for g in range(n_psth):
                grp = iix[g * group_size:(g + 1) * group_size]
                if len(grp) == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p = np.nanmean(robs_all[grp, seg["start"]:seg["end"]], axis=0)
                if p.size:
                    all_psths_for_scale.append(np.nanmax(p))
        psth_max = max(all_psths_for_scale) if all_psths_for_scale else 1.0
        psth_max = psth_max if psth_max > 0 else 1.0

        for seg in segments:
            iix = seg["iix"]
            n = len(iix)
            if n == 0:
                continue
            group_size = max(1, int(np.ceil(n / n_psth)))
            seg_t_ms = (np.arange(seg["start"], seg["end"]) * dt * 1000.0)
            for g in range(n_psth):
                grp = iix[g * group_size:(g + 1) * group_size]
                if len(grp) == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p = np.nanmean(robs_all[grp, seg["start"]:seg["end"]], axis=0)
                center = (g + 0.5) / n_psth * (n_rows - 1)
                half_band = (n_rows / n_psth) * 0.4
                y = center - p / psth_max * half_band
                color = cmap(g / max(n_psth - 1, 1))
                ax.plot(seg_t_ms, y, color=color, lw=1.2, alpha=0.9)

    # Right-side axis: distance range for the example segment (since the
    # rank-Y axis has no inherent distance scale, and per-segment ranges
    # differ; we show the example to give the reader a feel for the units).
    ex = payload["segments"][payload["example_segment_idx"]]
    if len(ex["distances"]):
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        ax_r.set_ylabel(f"Gaze projection in [{ex['start']}, {ex['end']}) bins (deg)",
                        fontsize=8)
        n_ticks = 5
        n_ex = len(ex["iix"])
        if n_ex > 1:
            tick_y = np.linspace(0, n_rows - 1, n_ticks)
            tick_d = np.linspace(ex["distances"].min(),
                                 ex["distances"].max(), n_ticks)
            ax_r.set_yticks(tick_y)
            ax_r.set_yticklabels([f"{v:.2f}" for v in tick_d])

    return ax


def plot_panel_d(fig=None, subject=SUBJECT, date=DATE, cell=DEFAULT_CELL,
                refresh=False):
    """Compose both axes onto ``fig`` (or a new figure). Returns (fig, axes)."""
    payload = load_cell_payload(subject, date, cell, refresh=refresh)

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3),
                                 gridspec_kw={"width_ratios": [1.0, 1.6]})
    else:
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.6]})

    plot_eyepos_axis(axes[0], payload)
    plot_raster_axis(axes[1], payload)
    ex = payload["segments"][payload["example_segment_idx"]]
    axes[0].set_title(
        f"Gaze in segment [{ex['start']},{ex['end']}) bins "
        f"({payload['session']} cell {int(payload['cell'])})",
        fontsize=9,
    )
    axes[1].set_title(
        f"Gaze-sorted raster (stitched, {payload['segment_len']}-bin segments)",
        fontsize=9,
    )
    return fig, axes


if __name__ == "__main__":
    fig, axes = plot_panel_d()
    fig.tight_layout()
    out = FIG_DIR / "fig1d_single_cell.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
