"""
Figure 1 panel D: single-cell tuning + gaze-driven raster structure.

Layout (4 axes):
    +----------------+----------------+
    |   STA peak     |  Gaze segment  |   shared deg extent, centered at 0
    +----------------+----------------+
    |   PSTH all trials (full width)  |
    +---------------------------------+
    |   Gaze-sorted stitched raster   |   sharex with PSTH
    +---------------------------------+

Segments: a single ~20 ms onset bin followed by 6 × 50 ms pulse bins, each
sorted independently by gaze projection onto the axis orthogonal to the
cell's preferred orientation.
"""

from pathlib import Path
import pickle
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR, CACHE_DIR
from eval.sta_ste import (
    compute_sta_ste,
    peak_lag_from_ste,
    population_peak_lag,
)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Flip to True to force STA/STE recomputation and per-cell payload refresh.
RECALC = False

SUBJECT = "Allen"
DATE = "2022-03-04"
DEFAULT_CELL = 149
DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)

DT = 1.0 / 240.0
ONSET_LEN_BINS = 5             # ~20 ms initial response delay
PULSE_LEN_BINS = 12            # 50 ms at 240 Hz (20 Hz pulse rate)
N_PULSES = 6                   # 6 pulses × 50 ms = 300 ms
TOTAL_WINDOW_BINS = (0, ONSET_LEN_BINS + N_PULSES * PULSE_LEN_BINS)  # (0, 77)
DISTANCE_FROM_LINE_THRESHOLD = 0.3
MICROSACCADE_THRESHOLD = 0.3
USE_UNIVERSAL_PEAK_LAG = True

CACHE_FIG_DIR = CACHE_DIR / "fig1_single_cell"
FIG_DIR = FIGURES_DIR / "fig1"
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_ste_for_session(session_name):
    """Cached STA/STE arrays for a session, or None if the gaborium dataset
    is missing. Honors the module-level RECALC flag."""
    return compute_sta_ste(session_name, recalc=RECALC)


def _gratings_cache_path(session_name):
    return CACHE_FIG_DIR / f"{session_name}_gratings.npz"


def _compute_gratings_for_session(subject, date, recalc=False):
    session_name = f"{subject}_{date}"
    cache = _gratings_cache_path(session_name)
    if cache.exists() and not recalc:
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


def _load_gaborium_geometry(session_name, recalc=False):
    path = _gaborium_geometry_cache_path(session_name)
    if path.exists() and not recalc:
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
    z = _load_ste_for_session(session_name)
    if z is None:
        return None
    stas = z["stas"]
    stes = z["stes"]
    row = _gaborium_row_for_cluster(session_name, cluster_id)
    peak_lag = peak_lag_from_ste(stes[row]) if lag is None else int(lag)
    img = np.asarray(stas[row, peak_lag], dtype=np.float64)

    ppd, _roi_origin = _load_gaborium_geometry(session_name)
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
        return {
            "iix": np.array([], dtype=int),
            "distances": np.array([]),
            "signed_proj": np.array([]),
            "cx": float(cx), "cy": float(cy), "slope": float(slope),
        }
    proj_arr = np.array(projections)
    order = np.argsort(proj_arr)
    iix = np.array(valid)[order]
    proj_sorted = proj_arr[order]
    distances = (proj_sorted - proj_sorted[0]) * norm
    med_proj = np.median(proj_sorted)
    signed = (proj_sorted - med_proj) * norm
    return {
        "iix": iix, "distances": distances, "signed_proj": signed,
        "cx": float(cx), "cy": float(cy), "slope": float(slope),
    }


def _segment_bounds(total_window=TOTAL_WINDOW_BINS):
    s0, e0 = total_window
    bounds = [(s0, min(s0 + ONSET_LEN_BINS, e0))]
    cur = bounds[0][1]
    for _ in range(N_PULSES):
        if cur >= e0:
            break
        nxt = min(cur + PULSE_LEN_BINS, e0)
        bounds.append((cur, nxt))
        cur = nxt
    return bounds


def _compute_segments(eyepos, max_orientation, peak_lag,
                     total_window=TOTAL_WINDOW_BINS):
    segments = []
    for (start, end) in _segment_bounds(total_window):
        res = _project_onto_orthogonal_line(
            eyepos, (start, end), max_orientation, peak_lag,
        )
        segments.append({
            "start": int(start), "end": int(end),
            "iix": res["iix"],
            "distances": res["distances"],
            "signed_proj": res["signed_proj"],
            "cx": res["cx"], "cy": res["cy"], "slope": res["slope"],
        })
    return segments


# ---------------------------------------------------------------------------
# Cached per-cell payload
# ---------------------------------------------------------------------------
def _cell_cache_path(subject, date, cell):
    return CACHE_FIG_DIR / f"{subject}_{date}_cell{cell}_v2.pkl"


def _compute_cell_payload(subject, date, cell, max_orientation=None):
    from eval.fixrsvp import get_fixrsvp_data

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
        gratings = _compute_gratings_for_session(subject, date, recalc=RECALC)
        gratings_cids = list(gratings["cids"])
        if cell in gratings_cids:
            row = gratings_cids.index(cell)
        else:
            row = cell_col
        max_orientation = float(gratings["peak_ori"][row])
    max_orientation = float(max_orientation)

    ste_arrs = _load_ste_for_session(session_name)
    if ste_arrs is None:
        psth = np.nanmean(robs_cell, axis=0)
        peak_lag_cell = int(np.nanargmax(psth))
        peak_lag = peak_lag_cell
    else:
        stes_all = ste_arrs["stes"]
        sta_row = _gaborium_row_for_cluster(session_name, cell)
        peak_lag_cell = peak_lag_from_ste(stes_all[sta_row])
        if USE_UNIVERSAL_PEAK_LAG:
            peak_lag = population_peak_lag(stes_all)
        else:
            peak_lag = peak_lag_cell

    segments = _compute_segments(eyepos, max_orientation, peak_lag)
    # Prefer a pulse segment (skip the short onset segment) as the example.
    seg_means = []
    for i, s in enumerate(segments):
        if i == 0:
            seg_means.append(-np.inf)
        else:
            seg_means.append(np.nanmean(robs_cell[:, s["start"]:s["end"]]))
    example_idx = int(np.nanargmax(seg_means)) if seg_means else 0

    return {
        "cell": int(cell),
        "cell_col": int(cell_col),
        "session": session_name,
        "max_orientation": float(max_orientation),
        "peak_lag": int(peak_lag),
        "total_window": np.asarray(TOTAL_WINDOW_BINS, dtype=int),
        "segments": segments,
        "example_segment_idx": example_idx,
        "eyepos_all": eyepos,
        "spike_times_all": spike_times,
        "trial_t_bins_all": trial_t_bins,
        "robs_cell_all": robs_cell,
    }


def load_cell_payload(subject=SUBJECT, date=DATE, cell=DEFAULT_CELL, refresh=None):
    if refresh is None:
        refresh = RECALC
    path = _cell_cache_path(subject, date, cell)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    payload = _compute_cell_payload(subject, date, cell)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _global_proj_vmax(payload):
    vals = []
    for seg in payload["segments"]:
        sp = seg.get("signed_proj", np.array([]))
        if len(sp):
            vals.append(np.max(np.abs(sp)))
    return float(max(vals)) if vals else 1.0


def _eyepos_window_geometry(payload, segment_idx=None):
    seg_i = payload["example_segment_idx"] if segment_idx is None else segment_idx
    seg = payload["segments"][seg_i]
    iix = seg["iix"]
    peak_lag = int(payload["peak_lag"])
    eye_all = payload["eyepos_all"]

    win_len = seg["end"] - seg["start"]
    s_shift = max(seg["start"] - peak_lag, 0)
    e_shift = s_shift + win_len
    eye = eye_all[iix]

    cx = float(seg["cx"]); cy = float(seg["cy"])
    slope = float(seg["slope"])
    return seg, eye, s_shift, e_shift, cx, cy, slope


def _draw_projection_line(ax, slope, lw=1.0, color="0.3"):
    """Draw a line through (0, 0) along the projection direction, sized to fill
    the current axis limits."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = np.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1)))
    theta = np.arctan(slope)
    dx = L * np.cos(theta); dy = L * np.sin(theta)
    ax.plot([-dx, dx], [-dy, dy], color=color, lw=lw, zorder=2)


def _fmt_deg(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _style_top_axis(ax, half, tick=0.5):
    ticks = [-tick, 0.0, tick]
    lbl = [_fmt_deg(-tick), "0", _fmt_deg(tick)]
    ax.set_xticks(ticks); ax.set_xticklabels(lbl)
    ax.set_yticks(ticks); ax.set_yticklabels(lbl)
    ax.tick_params(direction="in", length=3, width=0.8, labelsize=7,
                   top=True, right=True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_sta_axis(ax, payload):
    sta = _sta_centered_in_degrees(payload["session"], payload["cell"])
    if sta is None:
        ax.text(0.5, 0.5, "STA cache missing", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return ax
    img = sta["image"]
    vmax = float(np.nanmax(np.abs(img))) or 1.0
    ax.imshow(img, extent=sta["extent"], origin="upper",
              cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal")
    return ax


def plot_eyepos_axis(ax, payload, segment_idx=None, proj_vmax=None):
    """Gaze traces in the example segment, centered at 0 (median),
    colored by signed projection on a shared coolwarm scale."""
    seg, eye, s_shift, e_shift, cx, cy, slope = _eyepos_window_geometry(
        payload, segment_idx=segment_idx,
    )
    if proj_vmax is None:
        proj_vmax = _global_proj_vmax(payload)
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=-proj_vmax, vmax=proj_vmax)

    sp = seg["signed_proj"]
    for idx in range(len(seg["iix"])):
        trace = eye[idx, s_shift:e_shift, :].copy()
        trace[:, 0] -= cx; trace[:, 1] -= cy
        med = np.nanmedian(trace, axis=0)
        c = cmap(norm(sp[idx]))
        ax.plot(trace[:, 0], trace[:, 1], color=c, lw=0.5, alpha=0.65)
        ax.scatter(med[0], med[1], s=12, color=c,
                   edgecolor="k", linewidth=0.3, zorder=3)
    ax.set_aspect("equal")
    return ax


def _set_shared_top_limits(ax_sta, ax_eye, payload, segment_idx=None, pad=0.05):
    sta = _sta_centered_in_degrees(payload["session"], payload["cell"])
    if sta is not None:
        x0, x1, y0, y1 = sta["extent"]
        sta_half = max(abs(x0), abs(x1), abs(y0), abs(y1))
    else:
        sta_half = 0.0

    seg, eye, s_shift, e_shift, cx, cy, _slope = _eyepos_window_geometry(
        payload, segment_idx=segment_idx,
    )
    eye_win = eye[:, s_shift:e_shift, :]
    eye_half = float(np.nanmax([
        np.nanmax(np.abs(eye_win[..., 0] - cx)) if eye_win.size else 0.0,
        np.nanmax(np.abs(eye_win[..., 1] - cy)) if eye_win.size else 0.0,
    ]))

    half = max(sta_half, eye_half) + pad
    half = min(half, 0.6)  # crop both top axes to central ±0.6 deg
    ax_sta.set_xlim(-half, half); ax_sta.set_ylim(-half, half)
    ax_eye.set_xlim(-half, half); ax_eye.set_ylim(-half, half)
    return half


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
    s0, e0 = payload["total_window"]
    full = payload["robs_cell_all"]
    robs = full[:, s0:e0]
    n = robs.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(robs, axis=0) / DT
        sem = np.nanstd(robs, axis=0) / np.sqrt(max(n, 1)) / DT
    t_ms = (np.arange(s0, e0) + 0.5) * DT * 1000.0

    # Conditional PSTHs by sign of projection in each segment
    mean_neg = np.full(e0 - s0, np.nan)
    mean_pos = np.full(e0 - s0, np.nan)
    for seg in payload["segments"]:
        sst, sen = seg["start"], seg["end"]
        if sen <= s0 or sst >= e0:
            continue
        a = max(sst, s0); b = min(sen, e0)
        iix = seg["iix"]; sp = seg["signed_proj"]
        if len(iix) == 0:
            continue
        neg = iix[sp < 0]; pos = iix[sp > 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if len(neg):
                mean_neg[a - s0:b - s0] = np.nanmean(full[neg, a:b], axis=0) / DT
            if len(pos):
                mean_pos[a - s0:b - s0] = np.nanmean(full[pos, a:b], axis=0) / DT

    ax.plot(t_ms, mean_neg, color="#3b6db7", lw=0.9, alpha=0.85, zorder=1,
            label="proj < 0")
    ax.plot(t_ms, mean_pos, color="#c43c3c", lw=0.9, alpha=0.85, zorder=1,
            label="proj > 0")
    ax.fill_between(t_ms, mean - sem, mean + sem,
                    color="0.6", alpha=0.4, linewidth=0, zorder=2)
    ax.plot(t_ms, mean, color="k", lw=1.0, zorder=3, label="all")
    ax.set_xlim(s0 * DT * 1000.0, e0 * DT * 1000.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Spikes/s")
    ax.tick_params(direction="in", length=3, labelsize=7)
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

    proj_vmax = _global_proj_vmax(payload)
    cmap = plt.cm.coolwarm
    norm_proj = mpl.colors.Normalize(vmin=-proj_vmax, vmax=proj_vmax)

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
                color="k", lw=tick_lw, rasterized=True, zorder=3)

    if show_segment_dividers:
        for seg in segments[1:]:
            x_ms = seg["start"] * dt * 1000.0
            ax.axvline(x_ms, color="0.7", lw=0.4, ls="-", alpha=0.7, zorder=0)

    ax.set_ylim(n_rows, 0)
    ax.set_xlim(0, total_dur_ms)
    ax.set_xlabel("Time from fixation onset (ms)")

    # Left axis: 0 / N_trials with terse title.
    ax.set_yticks([0, n_rows])
    ax.set_yticklabels(["0", str(n_rows)])
    ax.set_ylabel("Trials, gaze ordered", fontsize=8)
    ax.tick_params(axis="y", labelsize=7, direction="in", length=3, left=True)

    # Single color strip to the right of the raster, for the last segment.
    last_seg = next((s for s in reversed(segments) if len(s["iix"])), None)
    if last_seg is not None:
        sp = last_seg["signed_proj"]
        n_last = len(sp)
        if n_last == 1:
            y_pos_last = np.array([0.5 * (n_rows - 1)])
        else:
            y_pos_last = np.linspace(0, n_rows - 1, n_last)

        gap_ms = 0.012 * total_dur_ms
        strip_w_ms = 0.025 * total_dur_ms
        x_left = total_dur_ms + gap_ms
        row_h = n_rows / max(n_last, 1)
        for k in range(n_last):
            c = cmap(norm_proj(sp[k]))
            ax.add_patch(Rectangle(
                (x_left, y_pos_last[k] - row_h / 2),
                strip_w_ms, row_h,
                facecolor=c, edgecolor="none", zorder=4, clip_on=False,
            ))

        # Right axis: top / 0 / bottom labels for last-segment projection.
        ax_r = ax.twinx()
        ax_r.set_ylim(ax.get_ylim())
        if np.any(sp < 0) and np.any(sp > 0):
            j = int(np.argmax(sp >= 0))
            if j == 0:
                y_zero = y_pos_last[0]
            else:
                sp_lo, sp_hi = sp[j - 1], sp[j]
                t = (0.0 - sp_lo) / (sp_hi - sp_lo)
                y_zero = y_pos_last[j - 1] + t * (y_pos_last[j] - y_pos_last[j - 1])
        else:
            y_zero = 0.5 * (y_pos_last[0] + y_pos_last[-1])
        ax_r.set_yticks([y_pos_last[0], y_zero, y_pos_last[-1]])
        ax_r.set_yticklabels([f"{sp[0]:+.2f}", "0", f"{sp[-1]:+.2f}"])
        ax_r.set_ylabel("Gaze proj. (°)", fontsize=8)
        # Pad past the color strip so labels don't overlap it.
        ax_r.tick_params(axis="y", labelsize=7, direction="in", length=3,
                         pad=12)
        for s in ax_r.spines.values():
            s.set_visible(False)

    return ax

def _add_block_label(ax, letter, dx=-22, dy=6):
    ax.annotate(
        letter, xy=(0, 1), xycoords="axes fraction",
        xytext=(dx, dy), textcoords="offset points",
        fontsize=16, fontweight="bold",
        va="bottom", ha="left", annotation_clip=False,
    )


def plot_panel_d(fig=None, subject=SUBJECT, date=DATE, cell=DEFAULT_CELL,
                refresh=False, panel_letters=("D", "E", "F")):
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
    half = _set_shared_top_limits(ax_sta, ax_eye, payload)

    # Projection line on both top axes (drawn after limits so it spans them).
    seg = payload["segments"][payload["example_segment_idx"]]
    slope = float(seg["slope"])
    _draw_projection_line(ax_sta, slope, lw=1.0, color="0.25")
    _draw_projection_line(ax_eye, slope, lw=2.0, color="0.25")

    _style_top_axis(ax_sta, half)
    _style_top_axis(ax_eye, half)

    plot_psth_axis(ax_psth, payload)
    plot_raster_axis(ax_raster, payload)

    ax_psth.tick_params(labelbottom=False)
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)
    ax_raster.spines["top"].set_visible(False)

    if panel_letters is not None:
        _add_block_label(ax_sta, panel_letters[0])
        _add_block_label(ax_psth, panel_letters[1])
        _add_block_label(ax_raster, panel_letters[2])

    return fig, {"sta": ax_sta, "eyepos": ax_eye,
                 "psth": ax_psth, "raster": ax_raster}


if __name__ == "__main__":
    fig, axes = plot_panel_d()
    out = FIG_DIR / "fig1d_single_cell.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
