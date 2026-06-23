#%%
"""
Compute STAs/STEs across gaborium datasets and extract RF contours with
convex-hull smoothing and circularity screening.

Sessions are loaded from the training YAML config. Output paths follow the
fig1 convention (FIGURES_DIR/fig1, CACHE_DIR, STATS_DIR/fig1).

Usage:
    uv run ryan/fig1/rf_contours.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR
from models.config_loader import load_dataset_configs
from DataYatesV1 import calc_sta
from DataYatesV1.utils.io import YatesV1Session
from DataYatesV1.utils.rf import get_contour

#%% ============================================================================
# Configuration
# ==============================================================================

N_LAGS = 20
DT = 1 / 240
SNR_THRESH = 9
SPIKE_THRESH = 200
CIRC_THRESH = 0.9  # min circularity (perimeter / pi / diameter); 1.0 = perfect circle, lower = more elongated
RECALC = False

DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
SUBJECTS = ["Allen", "Logan"]

FIG_DIR = FIGURES_DIR / "fig1"
STAT_DIR = STATS_DIR / "fig1"
CACHE_FIG_DIR = CACHE_DIR / "fig1_rf_contours"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)

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


#%% ============================================================================
# Load session list from YAML config
# ==============================================================================

dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
session_names = []
session_subjects = []
for cfg in dataset_configs:
    name = cfg["session"]
    subject = name.split("_")[0]
    if subject in SUBJECTS:
        session_names.append(name)
        session_subjects.append(subject)

print(f"Found {len(session_names)} sessions from YAML config")

#%% ============================================================================
# Phase 1: Compute and cache STAs/STEs per session
# ==============================================================================

all_stas = []
all_stes = []
all_num_spikes = []

for session_name, subject in zip(session_names, session_subjects):
    cache_path = CACHE_FIG_DIR / f"{session_name}_sta_ste.npz"

    if cache_path.exists() and not RECALC:
        print(f"{session_name}: loading from cache")
        cached = np.load(cache_path)
        all_stas.append(cached["stas"])
        all_stes.append(cached["stes"])
        all_num_spikes.append(cached["num_spikes"])
        continue

    print(f'\n{"="*80}')
    print(f"SESSION: {session_name}")
    print(f'{"="*80}')

    sess = YatesV1Session(session_name)
    dset = sess.get_dataset("gaborium")
    if dset is None:
        print("  No gaborium dataset found, skipping.")
        all_stas.append(None)
        all_stes.append(None)
        all_num_spikes.append(None)
        continue

    dset["stim"] = dset["stim"].float()
    dset["stim"] = (dset["stim"] - dset["stim"].mean()) / dset["stim"].std()

    robs = dset["robs"].numpy() if hasattr(dset["robs"], "numpy") else dset["robs"]
    dpi = dset["dpi_valid"].numpy() if hasattr(dset["dpi_valid"], "numpy") else dset["dpi_valid"]
    if dpi.ndim == 1:
        dpi = dpi[:, None]
    num_spikes = (robs * dpi).sum(axis=0)

    print(f"  Computing STAs ({N_LAGS} lags)...")
    stas = calc_sta(
        dset["stim"], dset["robs"], N_LAGS, dset["dpi_valid"],
        device="cuda", batch_size=10000, progress=True,
    ).cpu().numpy()

    print(f"  Computing STEs ({N_LAGS} lags)...")
    stes = calc_sta(
        dset["stim"], dset["robs"], N_LAGS, dset["dpi_valid"],
        device="cuda", batch_size=10000,
        stim_modifier=lambda x: x**2, progress=True,
    ).cpu().numpy()

    np.savez(cache_path, stas=stas, stes=stes, num_spikes=num_spikes)
    print(f"  Cached to {cache_path}")

    all_stas.append(stas)
    all_stes.append(stes)
    all_num_spikes.append(num_spikes)
    del dset

print(f"\nLoaded {len(all_stas)} sessions")

#%% ============================================================================
# Helper: compute SNR per unit
# ==============================================================================


def compute_snr(stes_arr):
    """Return (cluster_snr, cluster_lag, snr_per_lag) for an array of STEs."""
    signal = np.abs(stes_arr - np.median(stes_arr, axis=(2, 3), keepdims=True))
    signal = gaussian_filter(signal, [0, 1, 1, 1])
    noise = np.median(signal[:, 0], axis=(1, 2))
    snr_per_lag = np.max(signal, axis=(2, 3)) / noise[:, None]
    cluster_snr = snr_per_lag.max(axis=1)
    cluster_lag = snr_per_lag.argmax(axis=1)
    return cluster_snr, cluster_lag, snr_per_lag


#%% ============================================================================
# Helper: convex hull contour, area, and circularity
# ==============================================================================


def convex_hull_contour(contour_pts):
    """Compute convex hull of contour points.

    Returns (hull_pts, area, circularity) where circularity =
    perimeter / (pi * diameter) and diameter is the diameter of the
    minimum bounding circle (max pairwise distance on the hull).
    """
    hull = ConvexHull(contour_pts)
    hull_pts = contour_pts[hull.vertices]
    # Close the polygon
    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    area = hull.volume  # 2D: volume = area
    perimeter = np.sum(np.linalg.norm(np.diff(hull_pts, axis=0), axis=1))
    # Diameter: max distance between any two hull vertices
    from scipy.spatial.distance import pdist
    diameter = pdist(hull_pts[:-1]).max()
    circularity = perimeter / (np.pi * diameter) if diameter > 0 else np.inf
    return hull_pts, area, circularity


#%% ============================================================================
# Phase 2: SNR vs Spike Count scatter (inclusion criteria)
# ==============================================================================

MAX_SNR_PLOT = 150
subject_colors = {"Allen": "tab:blue", "Logan": "tab:green"}

fig, ax = plt.subplots(figsize=(10, 7))

for subject in SUBJECTS:
    idxs = [i for i, s in enumerate(session_subjects) if s == subject and all_stes[i] is not None]
    if not idxs:
        continue
    stes_cat = np.concatenate([all_stes[i] for i in idxs], axis=0)
    spikes_cat = np.concatenate([all_num_spikes[i] for i in idxs])
    cluster_snr, _, _ = compute_snr(stes_cat)
    ax.scatter(spikes_cat, cluster_snr, s=10, alpha=0.5,
               color=subject_colors[subject], label=subject)

ax.set_xscale("log")
ax.axhline(SNR_THRESH, color="red", ls="--", lw=1.5, label=f"SNR thresh = {SNR_THRESH}")
ax.axvline(SPIKE_THRESH, color="gray", ls="--", lw=1.5, label=f"Spike thresh = {SPIKE_THRESH}")
ax.set_xlabel("Number of spikes")
ax.set_ylabel("Max SNR across lags")
ax.set_title("SNR vs Spike Count — Unit Inclusion Criteria")
ax.legend()
ax.set_ylim(0, MAX_SNR_PLOT)
fig.tight_layout()
fig.savefig(FIG_DIR / "snr_vs_spikes.png", dpi=150, bbox_inches="tight")
show_or_close(fig)

#%% ============================================================================
# Phase 3: Per-session peak-lag STE grids + RF contour extraction
# ==============================================================================

all_contours_deg = []  # (session_name, subject, hull_deg)
all_rf_areas_deg2 = []
all_rf_circularities = []
all_rf_session_names = []
all_rf_subjects = []

for sess_i, (session_name, subject) in enumerate(zip(session_names, session_subjects)):
    if all_stes[sess_i] is None:
        continue

    stes_s = all_stes[sess_i]
    stas_s = all_stas[sess_i]
    spikes_s = all_num_spikes[sess_i]
    n_units = stas_s.shape[0]

    cluster_snr, cluster_lag, _ = compute_snr(stes_s)

    # Load ROI metadata for pixel-to-degree conversion
    sess_obj = YatesV1Session(session_name)
    dset = sess_obj.get_dataset("gaborium")
    if dset is None:
        continue
    roi_src = dset.metadata["roi_src"]
    ppd = dset.metadata["ppd"]
    roi_origin = roi_src[:, 0]
    del dset

    # Pre-compute contours for all units (None if excluded)
    unit_hulls = [None] * n_units
    unit_areas_deg2 = [None] * n_units
    unit_circs = [None] * n_units

    for uid in range(n_units):
        if cluster_snr[uid] <= SNR_THRESH or spikes_s[uid] <= SPIKE_THRESH:
            continue

        peak_lag = cluster_lag[uid]
        ste_img = stes_s[uid, peak_lag]
        ste_centered = ste_img - np.median(ste_img)

        # Skip contrast-suppressed units
        if ste_centered.max() < abs(ste_centered.min()):
            continue

        ptp = np.ptp(ste_centered)
        if ptp < 1e-8:
            continue
        ste_norm = (ste_centered - ste_centered.min()) / ptp

        try:
            contour, _, _ = get_contour(ste_norm, 0.5)
        except Exception:
            continue

        if len(contour) < 3:
            continue

        try:
            hull_pts, hull_area_px, circ = convex_hull_contour(contour)
        except Exception:
            continue

        # Convert hull to degrees
        hull_pix = hull_pts + roi_origin[None, :]
        hull_deg = hull_pix / ppd
        hull_deg[:, 0] *= -1  # flip vertical so up is positive

        area_deg2 = hull_area_px / (ppd ** 2)

        unit_hulls[uid] = hull_deg
        unit_areas_deg2[uid] = area_deg2
        unit_circs[uid] = circ

        # Store for combined plot (only if passes circularity threshold)
        if circ >= CIRC_THRESH:
            all_contours_deg.append((session_name, subject, hull_deg))
            all_rf_areas_deg2.append(area_deg2)
            all_rf_circularities.append(circ)
            all_rf_session_names.append(session_name)
            all_rf_subjects.append(subject)

    # ---- Per-session peak-lag STE grid with RF contour overlay ----
    order = np.argsort(-cluster_snr)
    ncols = int(np.ceil(np.sqrt(n_units)))
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"{session_name} — Peak-lag STE (sorted by SNR)", fontsize=12)

    for plot_i, uid in enumerate(order):
        ax = axes.flat[plot_i]
        peak_lag = cluster_lag[uid]
        ste_img = stes_s[uid, peak_lag]
        ste_centered = ste_img - np.median(ste_img)
        vmax = np.max(np.abs(ste_centered))
        ax.imshow(ste_centered, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                  interpolation="none")

        # Overlay convex hull contour if available (in image pixel coords)
        if unit_hulls[uid] is not None:
            # Reverse the deg conversion to get back to image coords for overlay
            hull_img = unit_hulls[uid].copy()
            hull_img[:, 0] *= -1
            hull_img = hull_img * ppd - roi_origin[None, :]
            ax.plot(hull_img[:, 1], hull_img[:, 0], color="lime", lw=1.0, alpha=0.8)

        snr_val = cluster_snr[uid]
        spike_val = spikes_s[uid]
        is_good = (snr_val > SNR_THRESH) and (spike_val > SPIKE_THRESH)
        has_rf = unit_hulls[uid] is not None
        passes_circ = has_rf and unit_circs[uid] >= CIRC_THRESH

        if passes_circ:
            color = "red"
        elif is_good:
            color = "orange"
        else:
            color = "gray"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title: include area and circularity if RF was extracted
        if has_rf:
            ax.set_title(
                f"{uid} SNR={snr_val:.1f} A={unit_areas_deg2[uid]:.2f}° C={unit_circs[uid]:.2f}",
                fontsize=5, color=color,
            )
        else:
            ax.set_title(f"{uid} SNR={snr_val:.1f}", fontsize=5, color=color)

    for plot_i in range(n_units, nrows * ncols):
        axes.flat[plot_i].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CACHE_FIG_DIR / f"{session_name}_peak_ste_grid.png",
                dpi=150, bbox_inches="tight")
    show_or_close(fig)
    print(f"Saved peak-lag STE grid for {session_name}")

#%% ============================================================================
# Phase 4: RF size vs circularity scatter (inclusion criteria)
# ==============================================================================

if all_rf_areas_deg2:
    fig, ax = plt.subplots(figsize=(8, 6))
    areas = np.array(all_rf_areas_deg2)
    circs = np.array(all_rf_circularities)
    subjs = np.array(all_rf_subjects)

    for subject in SUBJECTS:
        mask = subjs == subject
        if not mask.any():
            continue
        ax.scatter(areas[mask], circs[mask], s=15, alpha=0.6,
                   color=subject_colors[subject], label=subject)

    ax.axhline(CIRC_THRESH, color="red", ls="--", lw=1.5,
               label=f"Circularity min = {CIRC_THRESH}")
    ax.set_xlabel("RF area (deg²)")
    ax.set_ylabel("Circularity (perimeter / π / diameter)")
    ax.set_title("RF Size vs Circularity — Inclusion Criteria")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_size_vs_circularity.png", dpi=150, bbox_inches="tight")
    show_or_close(fig)

#%% ============================================================================
# Phase 5: Combined RF contour map (primary output)
# ==============================================================================

if all_contours_deg:
    # Build per-session colors: blues for Allen, greens for Logan
    allen_sessions = sorted(set(n for n, s, _ in all_contours_deg if s == "Allen"))
    logan_sessions = sorted(set(n for n, s, _ in all_contours_deg if s == "Logan"))

    session_color_map = {}
    if allen_sessions:
        blue_cmap = plt.cm.Blues
        for i, name in enumerate(allen_sessions):
            session_color_map[name] = blue_cmap(0.4 + 0.5 * i / max(len(allen_sessions) - 1, 1))
    if logan_sessions:
        green_cmap = plt.cm.Greens
        for i, name in enumerate(logan_sessions):
            session_color_map[name] = green_cmap(0.4 + 0.5 * i / max(len(logan_sessions) - 1, 1))

    fig, ax = plt.subplots(figsize=(4, 4))

    for session_name, subject, hull_deg in all_contours_deg:
        ax.plot(hull_deg[:, 1], hull_deg[:, 0],
                color=session_color_map[session_name], alpha=0.2, linewidth=0.8)

    # Draw an empty dashed circle for reference (1 deg radius)
    circle = plt.Circle((0, 0), 1, color="r", ls="--", lw=1.5, fill=False, alpha=1, zorder=1)
    ax.add_artist(circle)
    ax.text(.5, 0.05, "1° radius", fontsize=16, ha="center", va="bottom")

    # Legend: one entry per monkey (representative color)
    legend_handles = []
    if allen_sessions:
        legend_handles.append(
            Line2D([0], [0], color=plt.cm.Blues(0.65), lw=2, label=f"Allen ({len(allen_sessions)} sessions)")
        )
    if logan_sessions:
        legend_handles.append(
            Line2D([0], [0], color=plt.cm.Greens(0.65), lw=2, label=f"Logan ({len(logan_sessions)} sessions)")
        )
    #ax.legend(handles=legend_handles, fontsize=9, loc="best")

    n_contours = len(all_contours_deg)
    #ax.set_xlabel("Horizontal position (deg)")
    #ax.set_ylabel("Vertical position (deg)")
    ax.set_title(f"RF locations")
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=1.5, ls="--", zorder=0)
    ax.axvline(0, color="k", lw=1.5, ls="--", zorder=0)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    # Turn off ticks and labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_contours_deg_combined.png", dpi=150, bbox_inches="tight")
    show_or_close(fig)
    print(f"Saved combined RF contour plot: {n_contours} contours")
else:
    print("No contours extracted")

#%% ============================================================================
# Summary
# ==============================================================================

print(f'\n{"="*80}')
print("RF CONTOUR ANALYSIS COMPLETE")
print(f'{"="*80}')
print(f"Subjects:           {', '.join(SUBJECTS)}")
print(f"Total sessions:     {len(session_names)}")
print(f"SNR threshold:      {SNR_THRESH}")
print(f"Spike threshold:    {SPIKE_THRESH}")
print(f"Circularity thresh: {CIRC_THRESH}")
print(f"Contours extracted: {len(all_contours_deg)}")
print(f"Figures dir:        {FIG_DIR}")
print(f"Cache dir:          {CACHE_FIG_DIR}")
