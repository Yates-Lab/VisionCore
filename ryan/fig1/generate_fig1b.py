"""
Figure 1 panel B: gaze distribution during fixation for a representative
Allen session.

Picks the Allen session with the most valid fixrsvp eye-position samples,
plots a 2D histogram of gaze (filled percentile contours) with a 1-degree
reference circle.

Usage:
    uv run ryan/fig1/generate_fig1b.py
"""

import numpy as np
import matplotlib.pyplot as plt

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR, CACHE_DIR
from models.config_loader import load_dataset_configs
from DataYatesV1.utils.io import YatesV1Session


DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)
SUBJECT = "Allen"
FIX_RADIUS_DEG = 1.0
# Cumulative-mass band boundaries in %. The lowest (0-5%) is an "outlier"
# band rendered white so the diffuse tail doesn't fill the frame; the
# remaining 5 bands are colored from light to dark.
PERCENTILE_LEVELS = (5, 20, 40, 60, 80)
HIST_BINS = 120
HIST_RANGE_DEG = 1.5
PLOT_LIM_DEG = 1.5

FIG_DIR = FIGURES_DIR / "fig1"
CACHE_FIG_DIR = CACHE_DIR / "fig1_gaze"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _eyepos_deg_from_dataset(dset):
    """Return centered eye position in degrees (N x 2 as [x, y]) plus the
    dpi_valid mask. fixrsvp datasets already provide a degree-space
    'eyepos' field."""
    eyepos = np.asarray(dset["eyepos"], dtype=np.float64)
    valid = np.asarray(dset["dpi_valid"]).astype(bool).reshape(-1)
    return eyepos, valid


def _count_fixation_samples(name):
    sess = YatesV1Session(name)
    dset = sess.get_dataset("fixrsvp")
    if dset is None:
        return 0, None
    eyepos, valid = _eyepos_deg_from_dataset(dset)
    near = (np.abs(eyepos[:, 0]) < FIX_RADIUS_DEG) & (
        np.abs(eyepos[:, 1]) < FIX_RADIUS_DEG
    )
    return int((valid & near).sum()), eyepos[valid & near]


def pick_representative_session():
    """Return (session_name, eyepos_deg) for the Allen session with the
    most valid fixrsvp samples inside the fixation window."""
    cache = CACHE_FIG_DIR / "best_allen_session.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return str(z["session"]), z["eyepos"]

    configs = load_dataset_configs(DATASET_CONFIGS_PATH)
    names = [c["session"] for c in configs if c["session"].startswith(f"{SUBJECT}_")]

    best_name = None
    best_count = -1
    best_eyepos = None
    for name in names:
        try:
            count, eyepos = _count_fixation_samples(name)
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
            continue
        print(f"  {name}: {count} fixation samples")
        if count > best_count:
            best_count = count
            best_name = name
            best_eyepos = eyepos

    if best_name is None:
        raise RuntimeError("No Allen fixrsvp data found.")

    np.savez(cache, session=best_name, eyepos=best_eyepos)
    return best_name, best_eyepos


def _percentile_levels(H, percentiles):
    """Convert a 2D histogram to density-threshold levels enclosing the
    given mass percentiles (lowest threshold encloses the largest mass)."""
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    total = csum[-1]
    levels = []
    for p in percentiles:
        target = total * p / 100.0
        idx = int(np.searchsorted(csum, target))
        idx = min(idx, len(flat) - 1)
        levels.append(flat[idx])
    levels = sorted(set(levels))
    return levels


def plot_panel_b(ax=None, session_name=None, eyepos=None):
    """Draw the gaze-distribution panel on ``ax``.

    If ``session_name`` is None the representative Allen session is chosen.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    if session_name is None or eyepos is None:
        session_name, eyepos = pick_representative_session()

    from scipy.ndimage import gaussian_filter

    # Pass 1: locate the centroid of the >=50% mass region in the raw data.
    pad = 0.5  # generous bins for the initial pass before recentering
    edges0 = np.linspace(-HIST_RANGE_DEG - pad, HIST_RANGE_DEG + pad, HIST_BINS + 1)
    H0, xe0, ye0 = np.histogram2d(eyepos[:, 0], eyepos[:, 1], bins=[edges0, edges0])
    Hs0 = gaussian_filter(H0, sigma=1.5)
    level_50 = _percentile_levels(Hs0, [50])[0]
    mask = Hs0 >= level_50
    xc0 = 0.5 * (xe0[:-1] + xe0[1:])
    yc0 = 0.5 * (ye0[:-1] + ye0[1:])
    Xg, Yg = np.meshgrid(xc0, yc0, indexing="ij")
    w = Hs0[mask]
    centroid = np.array([
        np.average(Xg[mask], weights=w),
        np.average(Yg[mask], weights=w),
    ])
    eyepos_c = eyepos - centroid

    # Pass 2: histogram in the recentered frame, normalized to [0, 1].
    edges = np.linspace(-HIST_RANGE_DEG, HIST_RANGE_DEG, HIST_BINS + 1)
    H, xe, ye = np.histogram2d(eyepos_c[:, 0], eyepos_c[:, 1], bins=[edges, edges])
    Hs = gaussian_filter(H, sigma=1.5)
    Hn = Hs / Hs.max() if Hs.max() > 0 else Hs
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    cmap = plt.cm.viridis
    # Density thresholds at "100 - p"% mass remaining above (i.e., the level
    # below which p% of total mass lies). Sorted low → high density.
    levels = _percentile_levels(Hn, [100 - p for p in PERCENTILE_LEVELS])
    n_bands = len(PERCENTILE_LEVELS) + 1  # 6 bands for 5 boundaries
    # Outermost (0-5% mass, outliers) is white; the rest ramp light → dark.
    inner_colors = [cmap(0.85 - 0.7 * i / max(n_bands - 2, 1))
                    for i in range(n_bands - 1)]
    fill_colors = [(1.0, 1.0, 1.0, 1.0)] + inner_colors
    fill_levels = [0.0] + list(levels) + [float(Hn.max()) + 1e-9]
    ax.contourf(X, Y, Hn, levels=fill_levels, colors=fill_colors, alpha=0.85)
    ax.contour(X, Y, Hn, levels=levels, colors="k", linewidths=0.6, alpha=0.6)

    circle = plt.Circle((0, 0), 1.0, color="r", ls="--", lw=1.5, fill=False, zorder=5)
    ax.add_artist(circle)
    ax.text(0, 1.02, "1°", color="r", fontsize=9, ha="center", va="bottom")

    lim = PLOT_LIM_DEG
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.5)
    ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5)

    # Stepped colorbar matching the discrete percentile bands. Same colors,
    # same ordering: light = low-density band, dark = high-density band.
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    boundaries_pct = [0] + list(PERCENTILE_LEVELS) + [100]
    listed = mpl.colors.ListedColormap(fill_colors)
    norm = mpl.colors.BoundaryNorm(boundaries_pct, listed.N)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=listed)
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 100], spacing="proportional")
    cb.set_ticklabels(["0", "1"])
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    cb.outline.set_linewidth(0.5)

    return fig, ax, session_name


if __name__ == "__main__":
    fig, ax, name = plot_panel_b()
    ax.set_title(f"Gaze during fixation\n({name}, {PERCENTILE_LEVELS} %)", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig1b_gaze.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved {out}")
