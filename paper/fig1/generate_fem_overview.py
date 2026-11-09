"""
Standalone overview figure: small fixation differences drive large changes in
the foveal V1 population response.

Data-driven remake of the R01 schematic figure that was assembled by hand from
the Figure 1 work. Panels:

    (top left)   DPI eye image and rig schematic (marmoset drawing reused from
                 poster panel 1)
    (bottom left) the fixRSVP image on screen, per-trial fixation positions for
                  two gaze groups, and the session's population RF placed at
                  each group's mean gaze position
    (right)      population rasters for four trials per gaze group
    (bottom right) group population PSTHs

Data: the same Allen 2022-03-02 session and gaze-grouping search as Figure 1F
(``generate_fig1f.py``), with the distance between cluster centres capped at
0.06 deg (``GROUP_CENTROID_DISTANCE``). Red trials are 36, 39, 69, 84 and 87;
blue trials are 4, 8, 11, 26 and 44. Their mean gaze differs by 0.13 deg, and the
population rate differs about 1.5-fold in the grouping window. Figure 1F's
uncapped search picks the pairs furthest apart (0.33 deg).

What the marks mean (checked against the raw data, 2026-11-09):
    Dots         Each trial's median gaze over 158–292 ms after fixation onset,
                 the window used to group trials. Gaze is (x right, y up) in
                 degrees from screen centre. Gaze is stable within the window
                 (SD about 0.02 deg), and no trial has a microsaccade there.
    Contours     The session's population RF footprint: the half-maximum
                 contour of the mean normalized peak-lag STE of 124 RF-qualified
                 units. The gaborium crop follows gaze, so this map is in
                 retinal coordinates; it is drawn at each group's mean gaze,
                 i.e. where the RF lands on the screen. Its position comes from
                 the fixation trials themselves: 0.41 deg left and 0.06 deg below
                 gaze (95% CI y −0.11 to −0.02), matching Figure 1C's
                 across-session RFs. The mapping session alone put it 0.56 deg
                 left and 0.10 deg up, at the edge of that scatter; see
                 ``_task_rf_offset``. Single-unit RFs are similar in size
                 (median diameter 0.42 deg), so the 0.13 deg gaze shift between
                 groups is about a third of an RF.
    Image        fixRSVP image 4 (on screen 150–200 ms; the same picture on
                 every trial). It was on screen one 33 ms population lag before
                 the largest red–blue response difference
                 (``displayed_image``). Images 5 and 6 also appear during the
                 dot window, so one image is a simplification. Rendering this image at each gaze sample
                 reproduces the dataset's own gaze-contingent stimulus crops.

Two stages:
    1. ``load_assets`` caches the on-screen fixRSVP images, the per-bin image
       IDs and the gaborium RF geometry. Computing it the first time needs the
       optional DataYatesV1 package and the raw session data.
    2. Plotting needs only numpy/scipy/scikit-image/matplotlib plus the
       cached Figure 1F payload and STE cache.

Usage:
    uv run paper/fig1/generate_fem_overview.py
"""

import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import measure

from VisionCore.paths import CACHE_DIR, FIGURES_DIR, VISIONCORE_ROOT

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUBJECT, DATE = "Allen", "2022-03-02"
SESSION = f"{SUBJECT}_{DATE}"
DT = 1.0 / 240.0
DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)

PAYLOAD_PATH = CACHE_DIR / "fig1_population" / f"{SESSION}_panel_f.pkl"
STE_PATH = CACHE_DIR / "fig1_rf_contours" / f"{SESSION}_sta_ste.npz"
ASSET_PATH = CACHE_DIR / "fig1_population" / f"{SESSION}_overview_assets.npz"

# Trial groups: Figure 1F's grouping search (generate_fig1f._get_eyepos_clusters)
# with the distance between the two cluster-centre trials capped at 0.06 deg, so
# the example shows a gaze shift well under one RF (~0.13 deg vs ~0.45 deg).
# The uncapped search (up to 0.30 deg) favours the most distant pairs, which gave
# a 0.33 deg shift. All five trials of each group are shown.
GROUP_CENTROID_DISTANCE = (0.02, 0.06)
GROUPS_PATH = CACHE_DIR / "fig1_population" / f"{SESSION}_overview_groups.npz"

# Raster window, in bins relative to the clustering window [s, e).
RASTER_PAD_BINS = 6
PSTH_BIN_MS = 2.0
PSTH_SMOOTH_BINS = 1.0

# RF inclusion (matches rf_contours.py / generate_fig1c.py).
SNR_THRESH = 9
SPIKE_THRESH = 200
RF_LEVEL = 0.5

VIEW_HALF_DEG = 0.6          # half-width of the zoomed stimulus view
RED, BLUE = "#e3342b", "#3a6fb5"
RED_LIGHT, BLUE_LIGHT = "#f08a83", "#7ea4e0"
TEXT = "#202632"

FIG_DIR = FIGURES_DIR / "fem_overview"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_payload():
    with open(PAYLOAD_PATH, "rb") as f:
        return pickle.load(f)


def _compute_assets(payload):
    """Render on-screen images and read RF geometry (needs DataYatesV1)."""
    from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
    from DataYatesV1.exp.general import get_trial_protocols
    from DataYatesV1.utils.io import YatesV1Session, get_session
    from eval.fixrsvp import get_fixrsvp_data

    data = get_fixrsvp_data(SUBJECT, DATE, DATASET_CONFIGS_PATH, use_cached_data=True,
                            salvageable_mismatch_time_threshold=25, verbose=False)
    robs = np.nan_to_num(np.asarray(data["robs"]))
    if not np.array_equal(robs, np.nan_to_num(payload["robs"])):
        raise RuntimeError("fixRSVP data no longer matches the cached Figure 1F payload")
    image_ids = np.asarray(data["image_ids"]).astype(int)

    exp = get_session(SUBJECT, DATE).exp
    protocols = get_trial_protocols(exp)
    trials = [FixRsvpTrial(exp["D"][i], exp["S"]) for i, p in enumerate(protocols)
              if p == "FixRsvpStim" and FixRsvpTrial.is_valid(exp["D"][i])]
    ppd = float(exp["S"]["pixPerDeg"])
    cx, cy = (int(round(v)) for v in np.asarray(exp["S"]["centerPix"], float).ravel())

    wanted = sorted(set(np.unique(image_ids[:, :120])) - {-1, 0})
    out = {"image_ids": image_ids}
    for image_id in wanted:
        for trial in trials:
            hits = np.flatnonzero(np.asarray(trial.image_ids) == image_id)
            if hits.size:
                half = int(round(float(trial.radius) * ppd))
                roi = np.array([[cy - half, cy + half], [cx - half, cx + half]])
                out[f"im_{image_id}"] = np.asarray(
                    trial.get_rois(int(hits[0]), roi=roi)).squeeze().astype(np.uint8)
                out[f"half_{image_id}"] = half / ppd
                break

    dset = YatesV1Session(SESSION).get_dataset("gaborium")
    out["roi_origin"] = np.asarray(dset.metadata["roi_src"])[:, 0].astype(float)
    out["gaborium_ppd"] = float(dset.metadata["ppd"])
    return out


def _task_rf_offset(assets, lag, n_boot=200, seed=0):
    """RF location relative to gaze, estimated from the fixation task itself.

    The fixRSVP crops follow gaze (crop = gaze + roi_src), so trial-to-trial
    differences in spiking at the same moment of the repeated sequence should
    covary with crop contrast energy where the RF is. Both spikes and energy are
    demeaned within each sequence bin, which removes the shared image structure.
    The half-maximum-weighted centroid of the resulting map is the RF centre in
    gaze-relative degrees. A trial bootstrap gives its 95% CI. Needs DataYatesV1.

    This mapping gave a clearer RF on this session's own trials than the
    separate gaborium session: its offset matches Figure 1C's across-session
    RFs (left of and just below gaze), and recrops with mirrored gaze axes
    explain the residual spiking much less well.
    """
    from DataYatesV1.utils.data import DictDataset
    from DataYatesV1.utils.io import YatesV1Session

    dset = YatesV1Session(SESSION).get_dataset("fixrsvp")
    ppd = float(dset.metadata["ppd"])
    roi = np.asarray(dset.metadata["roi_src"])
    psth, eye = np.asarray(dset["psth_inds"]), np.asarray(dset["eyepos"])
    robs, trials = np.asarray(dset["robs"], dtype=np.float32), np.asarray(dset["trial_inds"])
    # Images 2-10 only, avoiding a bin either side of each 12-bin image change.
    ok = (psth < 106) & (psth % 12 > 1) & (psth % 12 < 10) & (np.hypot(*eye.T) < 0.8)
    idx = np.flatnonzero(ok)
    idx = idx[idx + lag < len(robs)]
    pad = 80
    ims = {i: np.pad(assets[f"im_{i}"].astype(np.float32), pad,
                     constant_values=float(np.median(assets[f"im_{i}"]))) for i in range(2, 11)}
    h_roi, w_roi = roi[0, 1] - roi[0, 0], roi[1, 1] - roi[1, 0]
    crops = np.empty((len(idx), h_roi, w_roi), np.float32)
    for n, k in enumerate(idx):
        im = ims[2 + psth[k] // 12]
        centre = (im.shape[0] - 2 * pad) // 2 + pad
        r0 = int(np.floor(centre - eye[k, 1] * ppd)) + roi[0, 0]
        c0 = int(np.floor(centre + eye[k, 0] * ppd)) + roi[1, 0]
        crops[n] = im[r0:r0 + h_roi, c0:c0 + w_roi]
    groups = psth[idx]

    def demean(x):
        x = x.copy()
        for gid in np.unique(groups):
            sel = groups == gid
            x[sel] -= x[sel].mean(0)
        return x

    energy = demean((crops - crops.mean(0)) ** 2)
    spikes = demean(robs[idx + lag].sum(1))
    rows, cols = np.mgrid[:h_roi, :w_roi]

    def centroid(sel):
        m = gaussian_filter(np.tensordot(spikes[sel], energy[sel], axes=(0, 0)), 1.5)
        w = np.where(m >= 0.5 * m.max(), m - 0.5 * m.max(), 0.0)
        r, c = (w * rows).sum() / w.sum(), (w * cols).sum() / w.sum()
        return np.array([(roi[1, 0] + c) / ppd, -(roi[0, 0] + r) / ppd])   # (x, y), up +

    est = centroid(np.ones(len(idx), bool))
    rng = np.random.default_rng(seed)
    tr, unique = trials[idx], np.unique(trials[idx])
    boot = np.array([centroid(np.isin(tr, rng.choice(unique, len(unique))))
                     for _ in range(n_boot)])
    return est, np.percentile(boot, [2.5, 97.5], axis=0)


def load_assets(payload, refresh=False):
    if ASSET_PATH.exists() and not refresh:
        assets = dict(np.load(ASSET_PATH))
    else:
        assets = _compute_assets(payload)
    if refresh or "task_rf_offset_deg" not in assets:
        est, ci = _task_rf_offset(assets, lag=int(payload["peak_lag"]))
        assets["task_rf_offset_deg"], assets["task_rf_offset_ci"] = est, ci
        np.savez_compressed(ASSET_PATH, **assets)
    return assets


def selected_trials(payload, refresh=False):
    """(low-rate/blue, high-rate/red) trial lists, five trials each. Cached."""
    if GROUPS_PATH.exists() and not refresh:
        z = np.load(GROUPS_PATH)
        return [int(t) for t in z["low"]], [int(t) for t in z["high"]]
    import generate_fig1f as fig1f          # same directory; heavy, so imported lazily
    iix_list, cl_list = fig1f._get_eyepos_clusters(
        payload["eyepos"], payload["robs"], payload["segment_start"],
        payload["segment_end"], payload["peak_lag"],
        dist_between_centroids=GROUP_CENTROID_DISTANCE, return_top_k=1)
    iix, clusters = iix_list[0], cl_list[0]
    groups = [sorted(int(t) for t, c in zip(iix, clusters) if c == label)
              for label in (0, 1)]            # 0 = lower total rate (blue), 1 = red
    np.savez(GROUPS_PATH, low=groups[0], high=groups[1])
    return groups[0], groups[1]


def displayed_image(payload, assets, low, high):
    """Image on screen one population lag before the largest group difference."""
    s, e = clustering_window(payload)
    centers, rate_hi = group_psth(payload, high, s, e + RASTER_PAD_BINS)
    _, rate_lo = group_psth(payload, low, s, e + RASTER_PAD_BINS)
    peak_ms = centers[np.argmax(np.abs(rate_hi - rate_lo))]
    stim_bin = int(peak_ms / 1000 / DT) - payload["peak_lag"]
    image_id = int(np.bincount(assets["image_ids"][high + low, stim_bin].clip(min=0)).argmax())
    return image_id, stim_bin


def clustering_window(payload):
    s = max(payload["segment_start"] - payload["peak_lag"], 0)
    return s, s + (payload["segment_end"] - payload["segment_start"])


def trial_spikes(payload, trial, t0_bin, t1_bin):
    """Spike times (ms from fixation onset) and unit indices inside the window."""
    t_bins = np.asarray(payload["trial_t_bins"][trial])
    onset = t_bins[~np.isnan(t_bins)][0] - DT / 2
    times, units = [], []
    for unit, spikes in enumerate(payload["spike_times_trials"][trial]):
        rel = np.atleast_1d(np.asarray(spikes)) - onset
        rel = rel[(rel >= t0_bin * DT) & (rel < t1_bin * DT)]
        times.append(rel * 1000)
        units.append(np.full(rel.size, unit))
    return np.concatenate(times), np.concatenate(units)


def group_psth(payload, trials, t0_bin, t1_bin):
    edges = np.arange(t0_bin * DT * 1000, t1_bin * DT * 1000 + 1e-9, PSTH_BIN_MS)
    n_units = payload["robs"].shape[2]
    rows = [np.histogram(trial_spikes(payload, t, t0_bin, t1_bin)[0], edges)[0]
            for t in trials]
    rate = np.mean(rows, axis=0) / (PSTH_BIN_MS / 1000) / n_units
    rate = gaussian_filter(rate.astype(float), PSTH_SMOOTH_BINS, mode="nearest")
    return 0.5 * (edges[1:] + edges[:-1]), rate


def trial_gaze(payload, trials, s, e):
    return np.array([np.nanmedian(payload["eyepos"][t, s:e], axis=0) for t in trials])


def population_rf(payload, assets):
    """Population RF outline in gaze-relative degrees (x right, y up).

    Shape and size: half-maximum contour of the mean normalized peak-lag STE of
    the RF-qualified units (gaborium noise mapping, as in Figure 1C).
    Position: centred on the fixation-task estimate from ``_task_rf_offset``.
    That estimate comes from this session's own fixation trials and puts the RF
    left of and just below gaze. The separate mapping session placed it about
    0.13 deg higher, at the edge of the across-session scatter in Figure 1C.
    Adding a gaze position gives the RF's location on the screen.
    """
    mapped, n_units = _mapped_rf_contour(payload, assets)
    centred = mapped - mapped.mean(0) + assets["task_rf_offset_deg"][None, :]
    return centred, n_units


def _mapped_rf_contour(payload, assets):
    """Half-maximum contour of the mean normalized gaborium STE, at its mapped
    position (gaze-relative degrees, x right, y up)."""
    z = np.load(STE_PATH)
    stes, spikes = z["stes"], z["num_spikes"]
    signal = np.abs(stes - np.median(stes, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, [0, 1, 1, 1])
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = signal.max(axis=(2, 3)) / noise[:, None]
    snr, lag = snr_per_lag.max(axis=1), snr_per_lag.argmax(axis=1)

    maps = []
    for uid in payload["cids"]:
        if snr[uid] <= SNR_THRESH or spikes[uid] <= SPIKE_THRESH:
            continue
        img = stes[uid, lag[uid]] - np.median(stes[uid, lag[uid]])
        if img.max() < abs(img.min()) or np.ptp(img) < 1e-8:
            continue
        maps.append((img - img.min()) / np.ptp(img))
    pop = gaussian_filter(np.mean(maps, axis=0), 1.0)
    pop = (pop - pop.min()) / np.ptp(pop)
    contour = max(measure.find_contours(pop, RF_LEVEL), key=len)
    deg = (contour + assets["roi_origin"][None, :]) / assets["gaborium_ppd"]
    deg[:, 0] *= -1                       # up is positive
    return np.column_stack([deg[:, 1], deg[:, 0]]), len(maps)   # (x, y)


# Marmoset line drawing, extracted once from the retired poster panel 1 artwork
# (labels, gaze line and eye-image inset removed; white made transparent).
MARMOSET_PNG = (VISIONCORE_ROOT / "poster" / "typst_poster" / "components"
                / "panel_fem" / "assets" / "marmoset.png")
DPI_IMAGE = VISIONCORE_ROOT / "paper" / "fig1" / "dpieg-levels.png"
NAVY = "#2e3192"
PSTH_RED = "#e8312a"
FRAME_BLUE = "#1aa7d8"

# Reference layout is specified in a 700 x 860 px canvas (100 px per inch).
CANVAS_W, CANVAS_H = 700, 860


def marmoset_artwork():
    """Marmoset head line drawing as an RGBA array."""
    return np.asarray(Image.open(MARMOSET_PNG).convert("RGBA"))


def nice_scale(span, options=(5, 10, 20, 50, 100)):
    return min(options, key=lambda v: abs(v - span / 3))


def make_figure():
    payload = load_payload()
    assets = load_assets(payload)
    low, high = selected_trials(payload)          # low rate → blue, high → red
    s, e = clustering_window(payload)
    t0, t1 = s, e + RASTER_PAD_BINS
    n_units = payload["robs"].shape[2]

    centers, rate_hi = group_psth(payload, high, t0, t1)
    _, rate_lo = group_psth(payload, low, t0, t1)
    image_id, stim_bin = displayed_image(payload, assets, low, high)
    image, half = assets[f"im_{image_id}"], float(assets[f"half_{image_id}"])
    gaze_hi, gaze_lo = trial_gaze(payload, high, s, e), trial_gaze(payload, low, s, e)
    rf, n_rf_units = population_rf(payload, assets)

    fig = plt.figure(figsize=(CANVAS_W / 100, CANVAS_H / 100))

    def box(x0, y0, x1, y1):
        """Axes from a reference-canvas pixel box (top-left origin)."""
        return fig.add_axes([x0 / CANVAS_W, 1 - y1 / CANVAS_H,
                             (x1 - x0) / CANVAS_W, (y1 - y0) / CANVAS_H])

    def P(x, y):
        return (x / CANVAS_W, 1 - y / CANVAS_H)

    def note(x, y, text, size=16, **kw):
        fig.text(*P(x, y), text, style="italic", fontsize=size, va="top",
                 color="black", linespacing=1.2, **kw)

    def arrow(a, b, color, coords_a="figure fraction", coords_b="figure fraction",
              ax_a=None, ax_b=None, lw=1.3, rad=0.0):
        fig.add_artist(ConnectionPatch(a, b, coordsA=coords_a, coordsB=coords_b,
                                       axesA=ax_a, axesB=ax_b, color=color, lw=lw,
                                       arrowstyle="-|>", mutation_scale=12,
                                       shrinkA=3, shrinkB=2,
                                       connectionstyle=f"arc3,rad={rad}"))

    # ---- Eye tracker -------------------------------------------------------
    note(22, 40, "Dual Purkinje\nhigh-resolution\neye tracker")
    ax_dpi = box(230, 10, 395, 165)
    eye = np.asarray(Image.open(DPI_IMAGE).convert("L"))
    ax_dpi.imshow(eye, cmap="gray", vmin=0, vmax=255)
    ax_dpi.add_patch(mpl.patches.Circle((270, 262), 222, fill=False, ec="#c9e8a0", lw=2))
    ax_dpi.add_patch(mpl.patches.Circle((382, 238), 30, fill=False, ec="#f2e27a", lw=1.5))
    for (tx, ty), (px, py), label in (((150, 170), (204, 278), "P4"),
                                      ((205, 400), (262, 330), "Pupil"),
                                      ((345, 395), (382, 268), "CR")):
        ax_dpi.annotate(label, (px, py), (tx, ty), color="#c9e8a0", fontsize=11,
                        arrowprops=dict(arrowstyle="-|>", color="#d9483b", lw=1.6,
                                        mutation_scale=8, shrinkB=2))
    ax_dpi.axis("off")

    # ---- Rig schematic ------------------------------------------------------
    ax_rig = box(0, 170, 330, 420)
    ax_rig.set_xlim(0, 330), ax_rig.set_ylim(420, 170), ax_rig.axis("off")
    ax_rig.add_patch(mpl.patches.Polygon([(38, 255), (150, 225), (150, 330), (38, 355)],
                                         fc="#8c8c8c", ec="none"))
    thumb = 18                                              # px half-width on screen
    ax_rig.imshow(image, cmap="gray", extent=[105 - thumb, 105 + thumb, 312 + thumb,
                                             312 - thumb], zorder=2)
    patch = (95, 302, 115, 322)
    ax_rig.add_patch(mpl.patches.Rectangle(patch[:2], 20, 20, fill=False,
                                           ec=FRAME_BLUE, lw=2.5, zorder=3))
    eye_xy = (216, 268)
    for corner in ((patch[2], patch[1]), (patch[2], patch[3])):
        ax_rig.plot([corner[0], eye_xy[0]], [corner[1], eye_xy[1]], ls=(0, (4, 3)),
                    color=FRAME_BLUE, lw=2, zorder=1)
    ax_rig.plot([178, 198], [300, 270], color="#f2a7a3", lw=4.5, solid_capstyle="butt",
                zorder=2)
    ax_rig.text(92, 178, "hot mirror", rotation=-38, fontsize=14, ha="left", va="top",
                rotation_mode="anchor")
    ax_rig.plot([166, 186], [244, 278], color="#f2a7a3", lw=1.2)
    ax_rig.text(157, 213, "Eye camera", fontsize=14, ha="left", va="bottom")
    ax_rig.add_patch(mpl.patches.Rectangle((196, 224), 26, 22, color="#1a1a1a"))
    ax_rig.add_patch(mpl.patches.Polygon([(201, 246), (217, 246), (213, 262), (205, 262)],
                                         color="#1a1a1a"))
    ax_head = box(193, 232, 313, 380)
    ax_head.imshow(marmoset_artwork(), aspect="auto")
    ax_head.axis("off")

    # ---- Zoomed stimulus with fixations and RFs -----------------------------
    pts = np.vstack([gaze_hi, gaze_lo, rf + gaze_hi.mean(0), rf + gaze_lo.mean(0)])
    cx, cy = (pts.min(0) + pts.max(0)) / 2
    view = np.ptp(pts, axis=0).max() / 2 + 0.06
    ax_im = box(8, 420, 222, 634)
    ax_im.imshow(image, cmap="gray", extent=[-half, half, -half, half], origin="upper",
                 vmin=0, vmax=255, interpolation="bilinear")
    for gaze, color, light in ((gaze_lo, BLUE, BLUE_LIGHT), (gaze_hi, RED, RED_LIGHT)):
        shifted = rf + gaze.mean(0)
        ax_im.plot(*np.vstack([shifted, shifted[:1]]).T, color=color, lw=2.6, zorder=3)
        ax_im.scatter(gaze[:, 0], gaze[:, 1], s=14, color=light, edgecolor=color,
                      linewidth=0.7, zorder=4)
    ax_im.set_xlim(cx - view, cx + view), ax_im.set_ylim(cy - view, cy + view)
    ax_im.set_xticks([]), ax_im.set_yticks([])
    for spine in ax_im.spines.values():
        spine.set_color(FRAME_BLUE), spine.set_linewidth(3.5)
    # Zoom lines from the on-screen patch to the enlarged view.
    for (xa, ya), (xb, yb) in (((patch[0], patch[3]), (8, 420)),
                               ((patch[2], patch[3]), (222, 420))):
        ax_rig.plot([xa, xb], [ya, yb], ls=(0, (4, 3)), color=FRAME_BLUE, lw=2,
                    clip_on=False)
    bar_deg = 1.0 if 214 / (2 * view) <= 214 else 0.5       # 1 deg unless wider than view
    bar_px = bar_deg * 214 / (2 * view)
    fig.add_artist(mpl.lines.Line2D([P(8, 0)[0], P(8 + bar_px, 0)[0]],
                                    [P(0, 649)[1]] * 2, color="black", lw=3.5,
                                    solid_capstyle="butt"))
    fig.text(*P(8 + bar_px, 657), "1 degree" if bar_deg == 1 else "0.5 degree", ha="right",
             va="top", fontsize=12)

    # ---- Rasters --------------------------------------------------------------
    rx0, rx1 = 480, 662
    raster_axes = {}
    n_per = len(high)
    step = 700 // (2 * n_per + 1)
    tops = [i * step for i in range(n_per)] + [(n_per + i) * step + 10 for i in range(n_per)]
    block_h = step - 7
    rows = ([(RED, high, k) for k in range(n_per, 0, -1)]
            + [(NAVY, low, k) for k in range(n_per, 0, -1)])
    for top, (color, group, k) in zip(tops, rows):
        ax = box(rx0, top + 4, rx1, top + 4 + block_h)
        times, units = trial_spikes(payload, group[k - 1], t0, t1)
        ax.scatter(times, units, s=6, marker="|", color="black", linewidths=0.9)
        ax.set_xlim(t0 * DT * 1000, t1 * DT * 1000), ax.set_ylim(n_units, -1)
        ax.axis("off")
        fig.text(*P(470, top + 18), f"Trial {k}", color=PSTH_RED if color == RED else NAVY,
                 ha="right", va="center", fontsize=13)
        raster_axes[(color, k)] = ax
    ax_last = raster_axes[(NAVY, 1)]
    neuron_px = block_h / (n_units + 1)
    y_end = tops[-1] + 4 + block_h
    y_bar = (y_end - 100 * neuron_px, y_end)
    fig.add_artist(mpl.lines.Line2D([P(667, 0)[0]] * 2, [P(0, y)[1] for y in y_bar],
                                    color="black", lw=1.8))
    fig.text(*P(678, y_bar[1]), "100 neurons", rotation=90, ha="left", va="bottom",
             fontsize=14)

    # ---- PSTH -----------------------------------------------------------------
    ax_psth = box(rx0, 733, rx1 - 22, 815)
    ax_psth.plot(centers, rate_hi, color=PSTH_RED, lw=2.4)
    ax_psth.plot(centers, rate_lo, color=NAVY, lw=2.4)
    x_lo, x_hi = t0 * DT * 1000, t1 * DT * 1000
    lo, hi = min(rate_hi.min(), rate_lo.min()), max(rate_hi.max(), rate_lo.max())
    ax_psth.set_xlim(x_lo, x_hi), ax_psth.set_ylim(lo, hi)
    ax_psth.axis("off")
    rate_bar = nice_scale(hi - lo)
    y0p = ax_psth.transData.transform((0, lo))[1]
    y1p = ax_psth.transData.transform((0, lo + rate_bar))[1]
    fh = fig.bbox.height
    fig.add_artist(mpl.lines.Line2D([P(667, 0)[0]] * 2, [y0p / fh, y1p / fh],
                                    color="black", lw=1.8))
    fig.text(*P(678, 815), f"{rate_bar} sp s$^{{-1}}$", rotation=90, ha="left",
             va="bottom", fontsize=14)
    ms_px = (rx1 - 22 - rx0) / (x_hi - x_lo)
    fig.add_artist(mpl.lines.Line2D([P(rx0, 0)[0], P(rx0 + 50 * ms_px, 0)[0]],
                                    [P(0, 826)[1]] * 2, color="black", lw=3.5,
                                    solid_capstyle="butt"))
    fig.text(*P(rx0, 836), "50ms", ha="left", va="top", fontsize=13)

    # ---- Annotations ------------------------------------------------------------
    note(252, 468, "example trials\nfor different\nfixation\npositions", size=15)
    arrow(tuple(gaze_hi.mean(0)), P(400, tops[n_per - 1] + 18), RED, coords_a="data", ax_a=ax_im,
          rad=-0.25)
    arrow(tuple(gaze_lo.mean(0)), P(410, tops[-1] + 18), NAVY, coords_a="data", ax_a=ax_im,
          rad=0.12)
    note(242, 655, "RF shift smaller\nthan RF size")
    arrow(P(236, 672), tuple(rf.mean(0) + gaze_hi.mean(0)), NAVY, coords_b="data",
          ax_b=ax_im)
    note(22, 738, "Small changes in fixation\ndrive large changes in\npopulation response")
    arrow(P(318, 800), P(474, 778), NAVY)

    stem = FIG_DIR / "fem_overview"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    shift = np.hypot(*(gaze_hi.mean(0) - gaze_lo.mean(0)))
    print(f"{SESSION}: trials red={high} blue={low}; window bins {t0}-{t1}; "
          f"image {image_id} (bin {stim_bin}); population RF from {n_rf_units} units; "
          f"gaze shift {shift:.3f} deg vs RF width {np.ptp(rf[:, 0]):.3f} deg; "
          f"rate bar {rate_bar} sp/s")
    print(f"Saved {stem}.pdf/.png/.svg")


if __name__ == "__main__":
    make_figure()
