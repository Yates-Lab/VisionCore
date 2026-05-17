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
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR
from DataYatesV1.utils.io import YatesV1Session
from DataYatesV1.utils.rf import get_contour

from eval.sta_ste import compute_sta_ste, compute_snr, sessions_from_yaml


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


def _load_all_contours(recalc=False):
    by_subject = {s: [] for s in SUBJECTS}
    for name, subject in sessions_from_yaml(DATASET_CONFIGS_PATH, subjects=SUBJECTS):
        hulls = _extract_contours_for_session(name, recalc=recalc)
        if hulls:
            by_subject[subject].append((name, hulls))
    return by_subject


def plot_panel_c(ax=None, refresh=None):
    if refresh is None:
        refresh = RECALC
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

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

    circle = plt.Circle((0, 0), 1.0, color="r", ls="--", lw=1.5, fill=False, zorder=5)
    ax.add_artist(circle)
    ax.text(0.5, 0.05, "1°", color="r", fontsize=9, ha="center", va="bottom")

    ax.axhline(0, color="k", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls="--", zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")

    print(f"Plotted {total} RF contours")
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_panel_c()
    ax.set_title("Receptive fields")
    fig.tight_layout()
    out = FIG_DIR / "fig1c_rf_contours.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
