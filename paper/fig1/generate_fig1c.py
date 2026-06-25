"""
Figure 1 panel C: foveal RF contour map.

Computes (or loads cached) STAs/STEs via the shared
``eval.sta_ste.compute_sta_ste`` module, then extracts convex-hull
contours that pass SNR / spike-count / circularity thresholds.

Set ``RECALC = True`` at the top of the file to force STA/STE
recomputation from raw data.

Usage:
    uv run ryan/fig1/generate_fig1c.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR
from DataYatesV1.utils.io import YatesV1Session
from DataYatesV1.utils.rf import get_contour

from eval.sta_ste import compute_sta_ste, compute_snr, sessions_from_yaml
from generate_fig1b import (
    pick_representative_session, _load_all_fixrsvp_stimuli,
    EXTENT_VIEW_MARGIN_DEG,
)


# Force STA/STE recomputation from raw data (otherwise cached arrays used).
RECALC = False

DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
SUBJECTS = ["Allen", "Logan"]
SNR_THRESH = 9
SPIKE_THRESH = 200
CIRC_THRESH = 0.9

FIG_DIR = FIGURES_DIR / "fig1"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _hull(contour_pts):
    hull = ConvexHull(contour_pts)
    pts = contour_pts[hull.vertices]
    pts = np.vstack([pts, pts[0]])
    perim = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    diam = pdist(pts[:-1]).max()
    circ = perim / (np.pi * diam) if diam > 0 else np.inf
    return pts, circ


def _extract_contours_for_session(session_name, recalc=False):
    """Return list of hull contours (in degrees) for a session that pass
    SNR / spike-count / circularity thresholds."""
    res = compute_sta_ste(session_name, recalc=recalc)
    if res is None:
        return []
    stes = res["stes"]
    spikes = res["num_spikes"]

    sess = YatesV1Session(session_name)
    dset = sess.get_dataset("gaborium")
    if dset is None:
        return []
    roi_origin = dset.metadata["roi_src"][:, 0]
    ppd = dset.metadata["ppd"]
    del dset

    snr, peak_lag, _ = compute_snr(stes)
    hulls = []
    for uid in range(stes.shape[0]):
        if snr[uid] <= SNR_THRESH or spikes[uid] <= SPIKE_THRESH:
            continue
        img = stes[uid, peak_lag[uid]]
        centered = img - np.median(img)
        if centered.max() < abs(centered.min()):
            continue
        ptp = np.ptp(centered)
        if ptp < 1e-8:
            continue
        norm = (centered - centered.min()) / ptp
        try:
            contour, _, _ = get_contour(norm, 0.5)
        except Exception:
            continue
        if len(contour) < 3:
            continue
        try:
            hull_pts, circ = _hull(contour)
        except Exception:
            continue
        if circ < CIRC_THRESH:
            continue
        hull_pix = hull_pts + roi_origin[None, :]
        hull_deg = hull_pix / ppd
        hull_deg[:, 0] *= -1  # up is positive
        hulls.append(hull_deg)
    return hulls


def _gaborium_row_for_cluster(session_name, cluster_id):
    sess = YatesV1Session(session_name)
    cluster_ids = np.asarray(sess.get_cluster_ids())
    matches = np.where(cluster_ids == cluster_id)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _extract_contour_for_cell(session_name, cluster_id, recalc=False):
    """Return the RF hull for one cluster using the same coordinates as C."""
    res = compute_sta_ste(session_name, recalc=recalc)
    if res is None:
        return None
    row = _gaborium_row_for_cluster(session_name, cluster_id)
    if row is None:
        return None

    stes = res["stes"]
    sess = YatesV1Session(session_name)
    dset = sess.get_dataset("gaborium")
    if dset is None:
        return None
    roi_origin = dset.metadata["roi_src"][:, 0]
    ppd = dset.metadata["ppd"]
    del dset

    _, peak_lag, _ = compute_snr(stes)
    img = stes[row, peak_lag[row]]
    centered = img - np.median(img)
    if centered.max() < abs(centered.min()):
        centered = -centered
    ptp = np.ptp(centered)
    if ptp < 1e-8:
        return None
    norm = (centered - centered.min()) / ptp
    try:
        contour, _, _ = get_contour(norm, 0.5)
        hull_pts, _ = _hull(contour)
    except Exception:
        return None
    hull_pix = hull_pts + roi_origin[None, :]
    hull_deg = hull_pix / ppd
    hull_deg[:, 0] *= -1
    return hull_deg


def _load_all_contours(recalc=False):
    by_subject = {s: [] for s in SUBJECTS}
    for name, subject in sessions_from_yaml(DATASET_CONFIGS_PATH, subjects=SUBJECTS):
        hulls = _extract_contours_for_session(name, recalc=recalc)
        if hulls:
            by_subject[subject].append((name, hulls))
    return by_subject


def plot_panel_c(ax=None, refresh=None, roi_extent=None,
                 highlight_session=None, highlight_cell=None,
                 show_extent_circle=False, extent_radius_deg=None,
                 lim=1.5, scale_bar=True,
                 show_stimulus=False, image_id=None, session_name=None):
    """Plot foveal RF contours.

    ``show_extent_circle`` overlays the fixRSVP image maximum extent (a solid
    circle at ``extent_radius_deg``, labelled by diameter) so the panel can be
    compared against the fig1b image-extent version; ``lim`` sets the
    half-width of the field of view in degrees. ``scale_bar`` draws a 0→1°
    bar with edge ticks under the "1°" label so it reads as the *radius* of
    the dashed circle (not the diameter).

    ``show_stimulus`` draws the fixRSVP face (``image_id``, from
    ``session_name`` or the representative session) behind the contours at its
    true on-screen extent, so the RF locations can be read against the
    stimulus; set ``lim`` to that extent so the face fills the panel. Over the
    face, the rings and scale bar switch to white-with-stroke for legibility.
    """
    if refresh is None:
        refresh = RECALC
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    import matplotlib.patheffects as pe
    if show_stimulus:
        anno_color = "white"
        anno_fx = [pe.withStroke(linewidth=1.8, foreground="black")]
    else:
        anno_color = "k"
        anno_fx = []

    # Stimulus backdrop behind the contours, matching the fig1b face.
    if show_stimulus:
        if session_name is None:
            session_name, _ = pick_representative_session()
        stimuli, _ = _load_all_fixrsvp_stimuli(session_name)
        if image_id is not None:
            stim_img, stim_half = stimuli[image_id]
        else:
            stim_img, stim_half = next(iter(stimuli.values()))
        s_lo, s_hi = np.percentile(stim_img, [2, 98])
        ax.imshow(stim_img,
                  extent=[-stim_half, stim_half, -stim_half, stim_half],
                  origin="upper", cmap="gray", vmin=s_lo, vmax=s_hi,
                  interpolation="bilinear", zorder=-2)

    by_subject = _load_all_contours(recalc=refresh)

    cmaps = {"Allen": plt.cm.Blues, "Logan": plt.cm.Greens}
    legend_handles = []
    total = 0
    for subject in SUBJECTS:
        sessions = sorted(by_subject.get(subject, []))
        if not sessions:
            continue
        cmap = cmaps[subject]
        for i, (name, hulls) in enumerate(sessions):
            color = cmap(0.4 + 0.5 * i / max(len(sessions) - 1, 1))
            for hull in hulls:
                ax.plot(hull[:, 1], hull[:, 0], color=color, alpha=0.15, lw=0.6)
                total += 1
        legend_handles.append(
            Line2D([0], [0], color=cmap(0.65), lw=2,
                   label=f"{subject} ({len(sessions)} sess)")
        )

    circle = plt.Circle((0, 0), 1.0, color=anno_color, ls="--", lw=0.8,
                        fill=False, zorder=5)
    circle.set_path_effects(anno_fx)
    ax.add_artist(circle)
    if scale_bar:
        # 0→1° bar (= the dashed circle's radius) lying on the horizontal
        # meridian; the "1°" label reads as the radius.
        bar_y = 0.0
        bar = ax.plot([0, 1], [bar_y, bar_y], color=anno_color, lw=1.5,
                      solid_capstyle="butt", zorder=6)[0]
        bar.set_path_effects(anno_fx)
        ax.text(0.5, bar_y + 0.04, "1°", color=anno_color, fontsize=9,
                ha="center", va="bottom", zorder=6).set_path_effects(anno_fx)
    else:
        ax.text(0.5, 0.05, "1°", color=anno_color, fontsize=9, ha="center",
                va="bottom").set_path_effects(anno_fx)

    # fixRSVP image maximum extent (solid circle, unlabelled), matching the
    # fig1b image-extent version.
    if show_extent_circle and extent_radius_deg is not None:
        ext = plt.Circle((0, 0), extent_radius_deg, color=anno_color, ls="-",
                         lw=1.0, fill=False, zorder=5)
        ext.set_path_effects(anno_fx)
        ax.add_artist(ext)

    # Meridians: match panel B's lighter dotted style for a consistent B/C row.
    cross_color = "white" if show_stimulus else "#888"
    ax.axhline(0, color=cross_color, lw=0.5, ls=":", alpha=0.5, zorder=4.5)
    ax.axvline(0, color=cross_color, lw=0.5, ls=":", alpha=0.5, zorder=4.5)
    if roi_extent is not None:
        x0, x1, y0, y1 = map(float, roi_extent)
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="0.45", linewidth=0.9,
            linestyle="-", alpha=0.75, zorder=6,
        ))
    if highlight_session is not None and highlight_cell is not None:
        hull = _extract_contour_for_cell(
            highlight_session, highlight_cell, recalc=refresh,
        )
        if hull is not None:
            ax.plot(
                hull[:, 1], hull[:, 0],
                color="0.05", alpha=0.95, lw=2.0, zorder=8,
            )
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")

    print(f"Plotted {total} RF contours")
    return fig, ax


if __name__ == "__main__":
    # Match the fig1b image-extent version: same zoom + the (unlabelled) 3°
    # image circle, 1° radius scale bar, no title.
    name, _ = pick_representative_session()
    _, face_radius = _load_all_fixrsvp_stimuli(name)

    fig, ax = plot_panel_c(show_extent_circle=True,
                           extent_radius_deg=face_radius,
                           lim=face_radius + EXTENT_VIEW_MARGIN_DEG,
                           scale_bar=True)
    fig.tight_layout()
    out = FIG_DIR / "fig1c_rf_contours.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
