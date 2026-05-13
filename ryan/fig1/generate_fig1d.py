"""
Figure 1 panel D: single-cell tuning + gaze-driven raster structure.

Layout (4 axes):
    +----------------+----------------+
    |   STA peak     |  Gaze segment  |   shared deg extent
    +----------------+----------------+
    |   PSTH all trials (full width)  |
    +---------------------------------+
    |   Gaze-sorted stitched raster   |   sharex with PSTH
    +---------------------------------+

Reuses ``tejas.rsvp_util.get_fixrsvp_data`` and the STA/STE cache produced
by ``rf_contours.py`` (CACHE_DIR/fig1_rf_contours). Per-cell payload is
cached to CACHE_DIR/fig1_single_cell.

Usage:
    uv run ryan/fig1/generate_fig1d.py
"""

from pathlib import Path
import pickle
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR, CACHE_DIR

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBJECT = "Allen"
DATE = "2022-03-04"
DEFAULT_CELL = 149
DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)

DT = 1.0 / 240.0
TOTAL_WINDOW_BINS = (0, 100)
SEGMENT_LEN_BINS = 25
DISTANCE_FROM_LINE_THRESHOLD = 0.3
MICROSACCADE_THRESHOLD = 0.3
USE_UNIVERSAL_PEAK_LAG = True

CACHE_FIG_DIR = CACHE_DIR / "fig1_single_cell"
RF_CACHE_DIR = CACHE_DIR / "fig1_rf_contours"
FIG_DIR = FIGURES_DIR / "fig1"
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# STA / STE cache access
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


def _gaborium_geometry_cache_path(session_name):
    return CACHE_FIG_DIR / f"{session_name}_gaborium_geom.npz"


def _load_gaborium_geometry(session_name):
    """ROI origin (row, col) in pixels, plus ppd. Cached per-session."""
    path = _gaborium_geometry_cache_path(session_name)
    if path.exists():
        z = np.load(path)
        return float(z["ppd"]), np.asarray(z["roi_origin"], dtype=np.float64)
    from DataYatesV1.utils.io import YatesV1Session
    sess = YatesV1Session(session_name)
    dset = sess.get_dataset("gaborium")
    if dset is None:
        raise RuntimeError(f"No gaborium dataset for {session_name}")
    roi_origin = np.asarray(dset.metadata["roi_src"][:, 0], dtype=np.float64)
    ppd = float(dset.metadata["ppd"])
    np.savez(path, ppd=ppd, roi_origin=roi_origin)
    return ppd, roi_origin


def _gaborium_row_for_cluster(session_name, cluster_id):
    """Map an absolute cluster id (as used by fixrsvp cids) to its row in
    the gaborium STA/STE cache. Gaborium rows are indexed positionally by
    ``YatesV1Session.get_cluster_ids()``."""
    from DataYatesV1.utils.io import YatesV1Session
    sess = YatesV1Session(session_name)
    cluster_ids = np.asarray(sess.get_cluster_ids())
    matches = np.where(cluster_ids == cluster_id)[0]
    if matches.size == 0:
        raise ValueError(
            f"cluster {cluster_id} not in gaborium cluster_ids for {session_name}"
        )
    return int(matches[0])


def _sta_centered_in_degrees(session_name, cluster_id, lag=None):
    """Return (image, extent_in_deg, peak_lag) for the STA at ``lag`` (or
    STE-peak lag if None), centered on the (weighted) RF centroid with
    elevation positive-up. ``cluster_id`` is the absolute cluster id (e.g.
    ``payload["cell"]``), not the rsvp data column.
    """
    z = _load_ste_for_session(session_name)
    if z is None:
        return None
    stas = z["stas"]
    stes = z["stes"]
    row = _gaborium_row_for_cluster(session_name, cluster_id)
    peak_lag = _peak_lag_from_ste(stes[row]) if lag is None else int(lag)
    img = np.asarray(stas[row, peak_lag], dtype=np.float64)

    ppd, roi_origin = _load_gaborium_geometry(session_name)
    h, w = img.shape

    centered = img - np.median(img)
    weights = np.abs(centered)
    rows_grid, cols_grid = np.indices(img.shape)
    if weights.sum() > 0:
        cr = (rows_grid * weights).sum() / weights.sum()
        cc = (cols_grid * weights).sum() / weights.sum()
    else:
        cr = (h - 1) / 2.0
        cc = (w - 1) / 2.0

    az_min = (-0.5 - cc) / ppd
    az_max = (w - 0.5 - cc) / ppd
    el_top = (cr + 0.5) / ppd
    el_bot = (cr - h + 0.5) / ppd
    extent = (az_min, az_max, el_bot, el_top)
    return {"image": centered, "extent": extent, "peak_lag": int(peak_lag)}


# ---------------------------------------------------------------------------
# Gaze sort
# ---------------------------------------------------------------------------
def _microsaccade_present(trial_eyepos, threshold=MICROSACCADE_THRESHOLD):
    med = np.nanmedian(trial_eyepos, axis=0)
    d = np.hypot(trial_eyepos[:, 0] - med[0], trial_eyepos[:, 1] - med[1])
    return np.any(d > threshold)


def _project_onto_orthogonal_line(eyepos, sort_window, max_orientation, peak_lag,
                                  distance_threshold=DISTANCE_FROM_LINE_THRESHOLD):
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


def _compute_segments(eyepos, max_orientation, peak_lag,
                     total_window=TOTAL_WINDOW_BINS, seg_len=SEGMENT_LEN_BINS):
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


# ---------------------------------------------------------------------------
# Cached per-cell payload
# ---------------------------------------------------------------------------
def _cell_cache_path(subject, date, cell):
    return CACHE_FIG_DIR / f"{subject}_{date}_cell{cell}.pkl"


def _compute_cell_payload(subject, date, cell, max_orientation=None):
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

    if max_orientation is None:
        gratings = _compute_gratings_for_session(subject, date)
        gratings_cids = list(gratings["cids"])
        if cell in gratings_cids:
            row = gratings_cids.index(cell)
        else:
            row = cell_col
        max_orientation = float(gratings["peak_ori"][row])
    max_orientation = float(max_orientation)

    ste_npz = _load_ste_for_session(session_name)
    if ste_npz is None:
        psth = np.nanmean(robs_cell, axis=0)
        peak_lag_cell = int(np.nanargmax(psth))
        peak_lag = peak_lag_cell
    else:
        stes_all = ste_npz["stes"]
        sta_row = _gaborium_row_for_cluster(session_name, cell)
        peak_lag_cell = _peak_lag_from_ste(stes_all[sta_row])
        if USE_UNIVERSAL_PEAK_LAG:
            lags = [int(stes_all[u].std((1, 2)).argmax())
                    for u in range(stes_all.shape[0])]
            peak_lag = int(np.median(lags))
        else:
            peak_lag = peak_lag_cell

    segments = _compute_segments(eyepos, max_orientation, peak_lag)
    seg_means = [
        np.nanmean(robs_cell[:, s["start"]:s["end"]]) for s in segments
    ]
    example_idx = int(np.nanargmax(seg_means)) if seg_means else 0

    return {
        "cell": int(cell),
        "cell_col": int(cell_col),
        "session": session_name,
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
            payload = pickle.load(f)
        if "cell_col" not in payload:
            # Legacy cache: recompute on the fly so the STA loader can index.
            from tejas.rsvp_util import get_fixrsvp_data
            data = get_fixrsvp_data(
                subject, date, DATASET_CONFIGS_PATH,
                use_cached_data=True,
                salvageable_mismatch_time_threshold=25,
                verbose=False,
            )
            payload["cell_col"] = int(list(data["cids"]).index(cell))
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        return payload
    payload = _compute_cell_payload(subject, date, cell)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _eyepos_window_geometry(payload, segment_idx=None):
    """Return (eye_iix_ordered, s_shift, e_shift, cx, cy, slope) for the
    example segment used by the gaze axis."""
    seg_i = payload["example_segment_idx"] if segment_idx is None else segment_idx
    seg = payload["segments"][seg_i]
    iix = seg["iix"]
    peak_lag = int(payload["peak_lag"])
    eye_all = payload["eyepos_all"]
    max_orientation = float(payload["max_orientation"])

    win_len = seg["end"] - seg["start"]
    s_shift = max(seg["start"] - peak_lag, 0)
    e_shift = s_shift + win_len
    eye = eye_all[iix]

    win = eye[:, s_shift:e_shift, :]
    cx = float(np.nanmedian(win[..., 0]))
    cy = float(np.nanmedian(win[..., 1]))
    slope = float(np.tan(np.deg2rad(max_orientation + 90.0)))
    return seg, eye, s_shift, e_shift, cx, cy, slope


def _scale_bar(ax, length_deg=0.2, label=None):
    """Draw a small horizontal scale bar in axes-fraction space at lower-right."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = x1 - x0
    span_y = y1 - y0
    x_end = x0 + 0.95 * span_x
    x_start = x_end - length_deg
    y = y0 + 0.07 * span_y
    ax.plot([x_start, x_end], [y, y], "k", lw=1.5, solid_capstyle="butt")
    if label is None:
        label = f"{length_deg:g}°"
    ax.text((x_start + x_end) / 2, y + 0.02 * span_y, label,
            ha="center", va="bottom", fontsize=7)


def plot_sta_axis(ax, payload):
    """Top-left: STA at peak lag, centered on RF centroid, in degrees."""
    sta = _sta_centered_in_degrees(payload["session"], payload["cell"])
    if sta is None:
        ax.text(0.5, 0.5, "STA cache missing", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return ax
    img = sta["image"]
    vmax = float(np.nanmax(np.abs(img)))
    if vmax == 0:
        vmax = 1.0
    ax.imshow(img, extent=sta["extent"], origin="upper",
              cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal")
    return ax


def plot_eyepos_axis(ax, payload, segment_idx=None):
    """Top-right: gaze traces in the example segment, colored by sort rank."""
    seg, eye, s_shift, e_shift, cx, cy, slope = _eyepos_window_geometry(
        payload, segment_idx=segment_idx,
    )
    length = 0.6
    ax.plot(
        [cx - length / 2, cx + length / 2],
        [cy - length / 2 * slope, cy + length / 2 * slope],
        color="0.4", lw=1.0, zorder=2,
    )

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(seg["iix"])))
    for idx in range(len(seg["iix"])):
        trace = eye[idx, s_shift:e_shift, :]
        med = np.nanmedian(trace, axis=0)
        ax.plot(trace[:, 0], trace[:, 1],
                color=colors[idx], lw=0.4, alpha=0.6)
        ax.scatter(med[0], med[1], s=6, color=colors[idx],
                   edgecolor="k", linewidth=0.25, zorder=3)
    ax.set_aspect("equal")
    return ax


def _set_shared_top_limits(ax_sta, ax_eye, payload, segment_idx=None, pad=0.05):
    """Set identical x/y limits on the STA and eyepos axes."""
    # STA half-extent
    sta = _sta_centered_in_degrees(payload["session"], payload["cell"])
    if sta is not None:
        x0, x1, y0, y1 = sta["extent"]
        sta_half = max(abs(x0), abs(x1), abs(y0), abs(y1))
    else:
        sta_half = 0.0

    # Eyepos half-extent (around its own median, centered locally)
    seg, eye, s_shift, e_shift, cx, cy, _slope = _eyepos_window_geometry(
        payload, segment_idx=segment_idx,
    )
    eye_win = eye[:, s_shift:e_shift, :]
    eye_half = float(np.nanmax([
        np.nanmax(np.abs(eye_win[..., 0] - cx)) if eye_win.size else 0.0,
        np.nanmax(np.abs(eye_win[..., 1] - cy)) if eye_win.size else 0.0,
    ]))

    half = max(sta_half, eye_half) + pad

    ax_sta.set_xlim(-half, half)
    ax_sta.set_ylim(-half, half)
    ax_eye.set_xlim(cx - half, cx + half)
    ax_eye.set_ylim(cy - half, cy + half)


def _segment_raster_lines(spike_times_list, trial_t_bins_list, trial_indices,
                         seg_start_bin, seg_end_bin, y_positions,
                         dt=DT, height=0.7):
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
        t0 = t_bins[0] - dt / 2
        rel = spikes - t0
        mask = (rel >= seg_start_s) & (rel < seg_end_s)
        if not np.any(mask):
            continue
        rel_ms = rel[mask] * 1000.0
        y0 = y_positions[k]
        for x in rel_ms:
            xs.extend([x, x, np.nan])
            ys.extend([y0, y0 + height, np.nan])
    return np.asarray(xs), np.asarray(ys)


def plot_psth_axis(ax, payload):
    """Full-width PSTH across all trials within the analysis window."""
    s0, e0 = payload["total_window"]
    robs = payload["robs_cell_all"][:, s0:e0]
    n = robs.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(robs, axis=0) / DT
        sem = np.nanstd(robs, axis=0) / np.sqrt(max(n, 1)) / DT
    t_ms = (np.arange(s0, e0) + 0.5) * DT * 1000.0
    ax.fill_between(t_ms, mean - sem, mean + sem,
                    color="0.6", alpha=0.4, linewidth=0)
    ax.plot(t_ms, mean, color="k", lw=1.0)
    ax.set_xlim(s0 * DT * 1000.0, e0 * DT * 1000.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Spikes/s")
    return ax


def plot_raster_axis(ax, payload, tick_height=0.7, tick_lw=0.6,
                    show_segment_dividers=True):
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

    all_xs, all_ys = [], []
    for seg in segments:
        iix = seg["iix"]
        n = len(iix)
        if n == 0:
            continue
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
            ax.axvline(x_ms, color="0.7", lw=0.4, ls="-", alpha=0.6, zorder=0)

    ax.set_ylim(n_rows, 0)
    ax.set_xlim(0, total_dur_ms)
    ax.set_xlabel("Time from fixation onset (ms)")
    ax.set_yticks([])

    # Right-side: gaze-projection range for the example segment
    ex = payload["segments"][payload["example_segment_idx"]]
    if len(ex["distances"]) > 1:
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        dmin = float(ex["distances"].min())
        dmax = float(ex["distances"].max())
        ax_r.set_yticks([0, n_rows - 1])
        ax_r.set_yticklabels([f"{dmin:.2f}", f"{dmax:.2f}"])
        ax_r.set_ylabel("Gaze proj. (deg)", fontsize=8)
        ax_r.tick_params(axis="y", labelsize=7)
    return ax


def plot_panel_d(fig=None, subject=SUBJECT, date=DATE, cell=DEFAULT_CELL,
                refresh=False):
    payload = load_cell_payload(subject, date, cell, refresh=refresh)

    if fig is None:
        fig = plt.figure(figsize=(4, 6.0), constrained_layout=True)

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 0.55, 1.5],
        width_ratios=[1.0, 1.0],
    )
    ax_sta = fig.add_subplot(gs[0, 0])
    ax_eye = fig.add_subplot(gs[0, 1])
    ax_psth = fig.add_subplot(gs[1, :])
    ax_raster = fig.add_subplot(gs[2, :], sharex=ax_psth)

    plot_sta_axis(ax_sta, payload)
    plot_eyepos_axis(ax_eye, payload)
    _set_shared_top_limits(ax_sta, ax_eye, payload)

    # Strip top-panel tick clutter; show a small scale bar instead.
    for a in (ax_sta, ax_eye):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    _scale_bar(ax_sta, length_deg=0.2)
    _scale_bar(ax_eye, length_deg=0.2)

    plot_psth_axis(ax_psth, payload)
    plot_raster_axis(ax_raster, payload)

    # PSTH shares x with raster; hide its tick labels (raster carries them).
    ax_psth.tick_params(labelbottom=False)
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)
    ax_raster.spines["top"].set_visible(False)

    return fig, {"sta": ax_sta, "eyepos": ax_eye,
                 "psth": ax_psth, "raster": ax_raster}


if __name__ == "__main__":
    fig, axes = plot_panel_d()
    out = FIG_DIR / "fig1d_single_cell.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
