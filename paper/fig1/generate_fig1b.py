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
# Radius of the analysis window used downstream (subset of the fixation
# constraint), drawn as a second reference circle.
ANALYSIS_RADIUS_DEG = 0.5
# Preferred (non-aversive) fixRSVP image IDs, tried in order — same curated
# list used by fig4a. The first ID present in the session is rendered.
PREFERRED_FIXRSVP_IMAGE_IDS = [21, 12, 7, 18, 3, 9, 14, 25, 5, 16]
# Cumulative-mass band boundaries in %. The lowest (0-5%) is an "outlier"
# band rendered white so the diffuse tail doesn't fill the frame; the
# remaining 5 bands are colored from light to dark.
PERCENTILE_LEVELS = (5, 20, 40, 60, 80)
HIST_BINS = 120
HIST_RANGE_DEG = 1.5
PLOT_LIM_DEG = 1.5
# Extra view margin (deg) beyond the 3° extent circle when it is shown — gives
# headroom for the "3° image" label above the arc and a gap for the colorbar
# inside the right spine. Shared by fig1c so the B/C row stays matched.
EXTENT_VIEW_MARGIN_DEG = 0.45

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


def _load_all_fixrsvp_stimuli(session_name):
    """Render every preferred fixRSVP image present in the session.

    During fixation every RSVP frame is shown at screen centre (position
    (0, 0)°), so each patch is sampled symmetrically about ``centerPix`` out
    to ``±faceRadius`` degrees — the full footprint the monkey fixates. The
    session is loaded once and all preferred images are rendered together,
    then cached as a single npz.

    Returns ``(stimuli, radius_deg)`` where ``stimuli`` is an ordered dict
    ``{image_id: (image, half_deg)}`` (only IDs found in the session, in
    ``PREFERRED_FIXRSVP_IMAGE_IDS`` order) and ``radius_deg`` is the nominal
    ``faceRadius`` (image maximum extent). ``half_deg`` is the patch's exact
    half-width in degrees, derived from the integer pixel half-width
    (``round(radius·ppd)/ppd``) so the image grid maps exactly onto the gaze
    degree coordinates (rounding makes the true extent a hair under radius).
    """
    cache = CACHE_FIG_DIR / "fixrsvp_stimuli_all.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ids = [int(i) for i in z["ids"]]
        stimuli = {i: (z[f"im_{i}"], float(z[f"half_{i}"])) for i in ids}
        return stimuli, float(z["radius"])

    from DataYatesV1.utils.io import get_session
    from DataYatesV1.exp.general import get_trial_protocols
    from DataYatesV1.exp.fix_rsvp import FixRsvpTrial

    subject, date = session_name.split("_")
    sess = get_session(subject, date)
    exp = sess.exp
    protocols = get_trial_protocols(exp)
    idxs = [i for i, p in enumerate(protocols)
            if p == "FixRsvpStim" and FixRsvpTrial.is_valid(exp["D"][i])]
    if not idxs:
        raise RuntimeError(f"No valid FixRsvp trials in {session_name}")

    ppd = float(exp["S"]["pixPerDeg"])
    center_pix = np.asarray(exp["S"]["centerPix"], dtype=float).ravel()
    cx, cy = int(round(center_pix[0])), int(round(center_pix[1]))

    def _render(trial, frame_idx):
        radius = float(trial.radius)
        half = int(round(radius * ppd))
        roi = np.array([[cy - half, cy + half],
                        [cx - half, cx + half]], dtype=int)
        img = np.asarray(trial.get_rois(int(frame_idx), roi=roi))
        # Exact half-width of the returned patch in degrees: the ROI is
        # `half` px each side of centre, so the patch is 2·half px wide and
        # spans 2·half/ppd deg → half-width = half/ppd.
        return img.squeeze().astype(np.uint8), radius, half / ppd

    stimuli = {}
    face_radius = None
    for desired_id in PREFERRED_FIXRSVP_IMAGE_IDS:
        for iT in idxs:
            trial = FixRsvpTrial(exp["D"][iT], exp["S"])
            hits = np.where(np.asarray(trial.image_ids) == desired_id)[0]
            if len(hits) == 0:
                continue
            image, radius, half_deg = _render(trial, hits[0])
            stimuli[desired_id] = (image, half_deg)
            face_radius = radius
            print(f"  fixrsvp image_id={desired_id:2d}: trial {iT}, "
                  f"radius={radius:.2f}° → {image.shape[0]}×{image.shape[1]}px "
                  f"= ±{half_deg:.3f}°")
            break
        else:
            print(f"  fixrsvp image_id={desired_id:2d}: not present — skipped")

    if not stimuli:
        raise RuntimeError(f"No preferred fixRSVP images found in {session_name}")

    payload = {"ids": np.array(list(stimuli.keys()), dtype=int),
               "radius": float(face_radius)}
    for i, (image, half_deg) in stimuli.items():
        payload[f"im_{i}"] = image
        payload[f"half_{i}"] = float(half_deg)
    np.savez(cache, **payload)
    return stimuli, float(face_radius)


def _load_fixrsvp_stimulus(session_name, image_id=None):
    """Return ``(image, half_deg)`` for one fixRSVP face.

    ``image_id=None`` selects the first preferred image present in the
    session (the default backdrop); otherwise the named image is returned.
    See :func:`_load_all_fixrsvp_stimuli` for the geometry.
    """
    stimuli, _ = _load_all_fixrsvp_stimuli(session_name)
    if image_id is not None:
        if image_id not in stimuli:
            raise KeyError(f"fixRSVP image_id={image_id} not in {session_name}")
        return stimuli[image_id]
    return next(iter(stimuli.values()))


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


def plot_panel_b(ax=None, session_name=None, eyepos=None, *,
                 image_id=None, show_stimulus=False,
                 show_fix_ring=False, show_extent_circle=True,
                 grey_outside_analysis=True, lim=None):
    """Draw the gaze-distribution panel on ``ax``.

    If ``session_name`` is None the representative Allen session is chosen.

    The defaults are the primary figure-1 look: no image backdrop, a 3°
    image-extent circle, and gaze density greyed out beyond the analysis
    window. Pass ``show_stimulus=True`` (and turn off the extent circle /
    greying) for the face-backdrop comparison variants.

    Parameters
    ----------
    image_id : int, optional
        Which fixRSVP face to use as the backdrop. ``None`` picks the first
        preferred image present in the session.
    show_stimulus : bool
        If True, draw the fixRSVP face behind the gaze cloud (mid-gray
        backdrop, white rings). If False (default), the gaze cloud sits on a
        plain white background with dark rings.
    show_fix_ring : bool
        Draw the 1° fixation-constraint ring (off by default — the face
        backdrop / extent circle already convey scale).
    show_extent_circle : bool
        Draw a circle at the image maximum extent (``faceRadius`` radius,
        i.e. a 3° diameter circle). Used by the no-image version.
    grey_outside_analysis : bool
        Render the gaze density outside the analysis window
        (``ANALYSIS_RADIUS_DEG``) in greyscale rather than the warm map, so
        it reads as excluded from analysis. Intended for the no-image
        version (greyscale over a face backdrop would be invisible).
    lim : float, optional
        Half-width of the plotted field of view in degrees. ``None`` (default)
        auto-selects: wide enough for the extent circle when shown, else the
        standard ``PLOT_LIM_DEG``.
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

    # Stimulus geometry (faceRadius = image maximum extent). Loaded even when
    # the backdrop is hidden, so the 3° extent circle uses the true radius.
    stimuli, face_radius = _load_all_fixrsvp_stimuli(session_name)

    if lim is None:
        lim = (face_radius + EXTENT_VIEW_MARGIN_DEG
               if show_extent_circle else PLOT_LIM_DEG)

    # Stimulus backdrop: the fixated fixRSVP face, centered at (0, 0)° and
    # spanning ±half_deg. Drawn first so the gaze cloud overlays it.
    if show_stimulus:
        if image_id is not None:
            stim_img, stim_half = stimuli[image_id]
        else:
            stim_img, stim_half = next(iter(stimuli.values()))
        # Gentle contrast stretch so facial structure reads against the
        # mid-gray screen background (the raw RSVP face is low-contrast).
        s_lo, s_hi = np.percentile(stim_img, [2, 98])
        ax.imshow(stim_img,
                  extent=[-stim_half, stim_half, -stim_half, stim_half],
                  origin="upper", cmap="gray", vmin=s_lo, vmax=s_hi,
                  interpolation="bilinear", zorder=0)

    # Density thresholds at "100 - p"% mass remaining above (i.e., the level
    # below which p% of total mass lies). Sorted low → high density.
    levels = _percentile_levels(Hn, [100 - p for p in PERCENTILE_LEVELS])
    n_bands = len(PERCENTILE_LEVELS) + 1  # 6 bands for 5 boundaries
    fill_levels = [0.0] + list(levels) + [float(Hn.max()) + 1e-9]

    # Per-band RGBA fills from a colormap. The outermost (0-5% mass, outlier)
    # band is transparent so the backdrop shows through the diffuse tail; the
    # rest ramp light → dark. Alpha is baked into each band — a scalar
    # `alpha=` on contourf would override it and paint the transparent band
    # as semi-opaque colour over the whole panel.
    def _band_colors(cmap, alpha, lo=0.2, hi=0.95):
        inner = [cmap(lo + (hi - lo) * i / max(n_bands - 2, 1))
                 for i in range(n_bands - 1)]
        return [(0.0, 0.0, 0.0, 0.0)] + [(c[0], c[1], c[2], alpha) for c in inner]

    # Warm map is the colorbar reference. Opaque when it must fully cover the
    # greyscale layer (no-image split); translucent to overlay a face.
    warm_alpha = 1.0 if grey_outside_analysis else 0.8
    fill_colors = _band_colors(plt.cm.YlOrRd, warm_alpha)

    if grey_outside_analysis:
        # Greyscale density everywhere marks "not analysed"; warm density,
        # clipped to the analysis window, marks what is. Drawn on top so the
        # circular clip boundary reads as the analysis edge.
        grey_colors = _band_colors(plt.cm.Greys, 1.0, lo=0.25, hi=0.65)
        ax.contourf(X, Y, Hn, levels=fill_levels, colors=grey_colors, zorder=1)
        warm_cf = ax.contourf(X, Y, Hn, levels=fill_levels, colors=fill_colors,
                              zorder=1.05)
        warm_cf.set_clip_path(
            plt.Circle((0, 0), ANALYSIS_RADIUS_DEG, transform=ax.transData))
    else:
        ax.contourf(X, Y, Hn, levels=fill_levels, colors=fill_colors, zorder=1)

    ax.contour(X, Y, Hn, levels=levels, colors="k", linewidths=0.5, alpha=0.35,
               zorder=1.1)

    # Reference annotations. Over the grayscale face, white with a dark
    # stroke stays legible; on the plain white background, plain dark lines
    # read better. Always show the 0.5° analysis window; the 1° fixation
    # ring and the 3° image-extent circle are opt-in.
    import matplotlib.patheffects as pe
    if show_stimulus:
        ring_color = "white"
        ring_fx = [pe.withStroke(linewidth=1.8, foreground="black")]
    else:
        ring_color = "#222"
        ring_fx = []

    # Circle labels. The analysis window (0.5° radius) is labelled "1°"
    # (diameter) at the lower-right of its circle; the image-extent circle
    # keeps its diameter label centered above.
    # (radius_deg, label, linestyle, placement) — placement ∈ above/below/lr
    rings = [(ANALYSIS_RADIUS_DEG, f"{2 * ANALYSIS_RADIUS_DEG:g}°", "--", "lr")]
    if show_fix_ring:
        rings.append((FIX_RADIUS_DEG, f"{FIX_RADIUS_DEG:g}°", "--", "above"))
    if show_extent_circle:
        rings.append((face_radius, f"{2 * face_radius:g}° image", "-", "above"))

    for radius_deg, label, ls, placement in rings:
        circ = plt.Circle((0, 0), radius_deg, color=ring_color, ls=ls, lw=1.0,
                          fill=False, zorder=5)
        circ.set_path_effects(ring_fx)
        ax.add_artist(circ)
        if placement == "lr":
            ang = np.deg2rad(-45.0)
            tx = radius_deg * np.cos(ang) + 0.02
            ty = radius_deg * np.sin(ang) - 0.02
            ha, va = "left", "top"
        elif placement == "below":
            tx, ty, ha, va = 0.0, -radius_deg - 0.02, "center", "top"
        else:  # above (sits just on top of the arc; view limits give headroom)
            tx, ty, ha, va = 0.0, radius_deg + 0.02, "center", "bottom"
        txt = ax.text(tx, ty, label, color=ring_color, fontsize=9,
                      ha=ha, va=va, zorder=6)
        txt.set_path_effects(ring_fx)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    cross_color = "white" if show_stimulus else "#888"
    ax.axhline(0, color=cross_color, lw=0.5, ls=":", alpha=0.5, zorder=4.5)
    ax.axvline(0, color=cross_color, lw=0.5, ls=":", alpha=0.5, zorder=4.5)

    # Stepped colorbar matching the discrete percentile bands. Same colors,
    # same ordering: light = low-density band, dark = high-density band.
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if show_extent_circle:
        # Anchor the bar in DATA coordinates just outside the 3° circle. The
        # view limits are expanded past it (EXTENT_VIEW_MARGIN_DEG), so the bar
        # keeps a fixed gap from both the circle and the right spine.
        cb_x0, cb_w, cb_h = face_radius + 0.15, 0.13, 2.2
        cax = inset_axes(
            ax, width="100%", height="100%",
            bbox_to_anchor=(cb_x0, -cb_h / 2.0, cb_w, cb_h),
            bbox_transform=ax.transData, loc="center", borderpad=0,
        )
    else:
        # Face versions (no extent circle): hug the inside of the right spine.
        cax = inset_axes(
            ax, width="4%", height="68%", loc="center right",
            bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes, borderpad=0.1,
        )
    boundaries_pct = [0] + list(PERCENTILE_LEVELS) + [100]
    listed = mpl.colors.ListedColormap(fill_colors)
    norm = mpl.colors.BoundaryNorm(boundaries_pct, listed.N)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=listed)
    cb = fig.colorbar(sm, cax=cax, spacing="proportional")
    # No ticks/labels (the labels collided with the panel's right spine);
    # annotate the ends instead — "1" above the bar, "0" below it.
    cb.set_ticks([])
    cb.ax.tick_params(length=0)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.5)
    cax.text(0.5, 1.04, "1", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=7)
    cax.text(0.5, -0.04, "0", transform=cax.transAxes, ha="center", va="top",
             fontsize=7)

    return fig, ax, session_name


def _save_all(fig, stem):
    """Save a figure as svg/pdf/png next to ``stem`` (a Path without suffix)."""
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
    print(f"Saved {stem}.{{svg,pdf,png}}")


if __name__ == "__main__":
    import math

    name, eyepos = pick_representative_session()
    stimuli, face_radius = _load_all_fixrsvp_stimuli(name)
    ids = list(stimuli.keys())
    print(f"Representative session: {name}; {len(ids)} fixRSVP images: {ids}")

    var_dir = FIG_DIR / "fig1b_variants"
    var_dir.mkdir(parents=True, exist_ok=True)

    # ── Main panel: the default look (no backdrop, 3° extent circle, density
    # greyed outside the analysis window).
    fig, ax, _ = plot_panel_b(session_name=name, eyepos=eyepos)
    ax.set_title(f"Gaze during fixation\n({name})", fontsize=9)
    fig.tight_layout()
    _save_all(fig, FIG_DIR / "fig1b_gaze")
    plt.close(fig)

    # ── Face-backdrop comparison: one version per image in the list, kept for
    # the team to weigh against the default. Individual PNGs + a montage.
    FACE_KW = dict(show_stimulus=True, show_extent_circle=False,
                   grey_outside_analysis=False)
    ncol = min(5, len(ids))
    nrow = math.ceil(len(ids) / ncol)
    mfig, maxes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow),
                               squeeze=False)
    for k, im_id in enumerate(ids):
        fig, ax, _ = plot_panel_b(session_name=name, eyepos=eyepos,
                                  image_id=im_id, **FACE_KW)
        ax.set_title(f"image_id={im_id}", fontsize=9)
        fig.tight_layout()
        fig.savefig(var_dir / f"fig1b_gaze_im{im_id:02d}.png", dpi=200)
        plt.close(fig)

        r, c = divmod(k, ncol)
        plot_panel_b(ax=maxes[r][c], session_name=name, eyepos=eyepos,
                     image_id=im_id, **FACE_KW)
        maxes[r][c].set_title(f"image_id={im_id}", fontsize=9)
    for k in range(len(ids), nrow * ncol):
        r, c = divmod(k, ncol)
        maxes[r][c].axis("off")
    mfig.tight_layout()
    mfig.savefig(var_dir / "fig1b_gaze_montage.png", dpi=150)
    mfig.savefig(var_dir / "fig1b_gaze_montage.pdf")
    plt.close(mfig)
    print(f"Saved {var_dir}/fig1b_gaze_montage.{{png,pdf}} "
          f"and {len(ids)} per-image PNGs")
