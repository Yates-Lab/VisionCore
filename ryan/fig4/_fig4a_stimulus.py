"""Panel A — left half: stimulus example with a consistent 3D viewpoint.

All elements (training screens, test screen, lag cube) live in a single 3D
world frame and project to 2D through one shared **cabinet** projection.

World convention
----------------
- ``+x`` right, ``+y`` up, ``+z`` INTO the page (away from viewer).
- Cabinet projection: ``+z`` shifts by ``DEPTH_VEC`` (up-and-left, 30°,
  half-scale). Front faces (``z = 0`` plane) keep true shape.

Geometry
--------
- Every screen is a yawed 3D rectangle (rotation about world-up) so its
  right edge recedes into the page.
- The lag cube is axis-aligned in the screen's local frame (same yaw):
  its front face is parallel to the screens; time runs along local depth.
- The cube's back-face center is positioned so it projects exactly to the
  centre of the test screen's sampling ROI; magnification lines fan out
  from the four ROI corners to the four back-face corners.
"""
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage, ImageDraw
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from _fig4a_glyphs import (
    CYAN, ARROW_COLOR, TEXT_COLOR,
    _perspective_coeffs,
)


# ──────────────────────────────────────────────────────────────────────────
# Shared cabinet projection
# ──────────────────────────────────────────────────────────────────────────
CABINET_ALPHA = np.deg2rad(45.0)
CABINET_DEPTH = 0.5
DEPTH_VEC = np.array([
    -np.cos(CABINET_ALPHA) * CABINET_DEPTH,
    +np.sin(CABINET_ALPHA) * CABINET_DEPTH,
])

SCREEN_YAW_DEG = -22.0

# Lag-cube has its own rotation triple (yaw / pitch / roll about world Y / X / Z
# applied in that order) so its visual cabinet layout can be tuned independently
# of the screens. Defaults match the screen yaw → original look. The Flask
# tuner (cube_tuner_app.py) monkey-patches these at runtime.
CUBE_YAW_DEG   = -40.0
CUBE_PITCH_DEG =   5.0
CUBE_ROLL_DEG  =  -5.0


def cabinet_project(p3):
    """Project (N×3) (or shape-(3,)) world points to 2D screen coords."""
    p3 = np.asarray(p3, dtype=float)
    return p3[..., :2] + p3[..., 2:3] * DEPTH_VEC


def _R_y(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def _R_x(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def _R_z(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _euler_rotation(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """Composed rotation: world_pt = R_y(yaw) @ R_x(pitch) @ R_z(roll) @ local_pt.

    Roll about local depth → pitch tilts forward/back → yaw spins about
    world-vertical. Picked so yaw matches the prior single-axis behavior
    when pitch=roll=0.
    """
    return _R_y(yaw_deg) @ _R_x(pitch_deg) @ _R_z(roll_deg)


def screen_corners_3d(center, width, height, *, yaw_deg=SCREEN_YAW_DEG):
    """World corners (LL, LR, UR, UL) of an upright, yawed rectangle."""
    cx, cy, cz = center
    w2, h2 = width / 2.0, height / 2.0
    local = np.array([
        [-w2, -h2, 0.0],
        [+w2, -h2, 0.0],
        [+w2, +h2, 0.0],
        [-w2, +h2, 0.0],
    ])
    return local @ _R_y(yaw_deg).T + np.array([cx, cy, cz])


def box_corners_3d(front_center, size, *, yaw_deg=SCREEN_YAW_DEG,
                   pitch_deg=0.0, roll_deg=0.0):
    """World corners of a rotated axis-aligned box.

    `front_center` is the centre of the front face; `size=(w,h,d)`.
    Rotation: R_y(yaw) @ R_x(pitch) @ R_z(roll) applied to local coords
    (see `_euler_rotation`). When pitch=roll=0 this reduces to the prior
    yaw-only behavior.

    Returns 8×3 ordered::

        0 front-LL  1 front-LR  2 front-UR  3 front-UL
        4 back-LL   5 back-LR   6 back-UR   7 back-UL
    """
    cx, cy, cz = front_center
    w, h, d = size
    w2, h2 = w / 2.0, h / 2.0
    local = np.array([
        [-w2, -h2, 0.0], [+w2, -h2, 0.0], [+w2, +h2, 0.0], [-w2, +h2, 0.0],
        [-w2, -h2, d],   [+w2, -h2, d],   [+w2, +h2, d],   [-w2, +h2, d],
    ])
    R = _euler_rotation(yaw_deg, pitch_deg, roll_deg)
    return local @ R.T + np.array([cx, cy, cz])


# Pre-computed projection of the "back-face centre" offset relative to the
# front-face centre, for the shared yaw. Used to invert: given a desired
# projected position of the back-face centre, solve for the cube's
# front-face centre.
def _back_face_center_offset_2d(depth, yaw_deg=SCREEN_YAW_DEG,
                                pitch_deg=0.0, roll_deg=0.0):
    local = np.array([0.0, 0.0, depth])
    world = _euler_rotation(yaw_deg, pitch_deg, roll_deg) @ local
    return cabinet_project(world)


# ──────────────────────────────────────────────────────────────────────────
# Textured-quad rendering
# ──────────────────────────────────────────────────────────────────────────
def _draw_quad_image(ax, image, dst_quad, *, src_corners=None,
                     zorder=2, alpha=1.0, auto_contrast=False,
                     out_res=512):
    """Warp `image` into `dst_quad` (4×2 data coords, LL,LR,UR,UL).

    Default `src_corners` uses the full image extent (PIL pixel coords,
    y-down): LL=(0,H), LR=(W,H), UR=(W,0), UL=(0,0).
    """
    H, W = image.shape[:2]
    if src_corners is None:
        src_corners = np.array([
            [0, H], [W, H], [W, 0], [0, 0],
        ], dtype=np.float64)

    bx0, by0 = dst_quad[:, 0].min(), dst_quad[:, 1].min()
    bx1, by1 = dst_quad[:, 0].max(), dst_quad[:, 1].max()
    sx = out_res / (bx1 - bx0)
    sy = out_res / (by1 - by0)
    dst_px = np.column_stack([
        (dst_quad[:, 0] - bx0) * sx,
        (by1 - dst_quad[:, 1]) * sy,
    ])

    coeffs = _perspective_coeffs(src_corners, dst_px)
    pil = PILImage.fromarray(image.astype(np.uint8))
    if pil.mode != "L":
        pil = pil.convert("L")
    if auto_contrast:
        arr = np.asarray(pil, dtype=np.float32)
        vmin, vmax = np.percentile(arr, [1, 99])
        if vmax > vmin:
            arr = np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255)
            pil = PILImage.fromarray(arr.astype(np.uint8))

    warped = pil.transform(
        (out_res, out_res), PILImage.PERSPECTIVE, coeffs,
        resample=PILImage.BILINEAR,
    )
    mask = PILImage.new("L", (out_res, out_res), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in dst_px], fill=255)
    rgba = np.dstack([np.array(warped)] * 3 + [np.array(mask)])
    ax.imshow(rgba, extent=[bx0, bx1, by0, by1], origin="upper",
              zorder=zorder, interpolation="bilinear", alpha=alpha)


def _solve_homography(src, dst):
    """Solve forward planar homography src→dst (4 pairs)."""
    rows, rhs = [], []
    for (sx, sy), (dx, dy) in zip(src, dst):
        rows.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx])
        rows.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy])
        rhs.append(dx)
        rhs.append(dy)
    coeffs, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return coeffs


def _apply_h(c, pts):
    a, b, e, d, f, g, gh, hh = c
    x, y = pts[:, 0], pts[:, 1]
    denom = gh * x + hh * y + 1.0
    return np.column_stack([(a * x + b * y + e) / denom,
                            (d * x + f * y + g) / denom])


# ROI sequence cadence (samples at 120 Hz):
#   sweep boxes draw every SWEEP_STEP samples (low alpha, thin)
#   snapshot boxes draw every SNAPSHOT_STEP samples (high alpha, crisp)
ROI_SNAPSHOT_ALPHA = 0.85
ROI_SNAPSHOT_LW = 1.0


def _pick_fixation_rois(trace_px, *, fs=120.0, pix_per_deg=None,
                        vel_thresh_deg_s=20.0, min_dur_s=0.10,
                        min_sep_deg=1.0):
    """Pick one sample index per fixation in a gaze trace.

    A "fixation" is a contiguous run of samples below ``vel_thresh_deg_s``
    lasting at least ``min_dur_s``. The median sample of each run is
    chosen as the snapshot index. Indices closer than ``min_sep_deg`` to
    a previously-kept snapshot (in screen-pixel space) are dropped to
    avoid crowding.
    """
    trace = np.asarray(trace_px, dtype=float)
    if len(trace) < 2 or pix_per_deg is None:
        return []
    dxy = np.diff(trace, axis=0, prepend=trace[:1])
    speed_px_per_sample = np.linalg.norm(dxy, axis=1)
    vel_thresh_px = vel_thresh_deg_s * pix_per_deg / fs
    fix = speed_px_per_sample < vel_thresh_px

    min_n = max(2, int(round(min_dur_s * fs)))
    runs = []
    i = 0
    while i < len(fix):
        if fix[i]:
            j = i
            while j < len(fix) and fix[j]:
                j += 1
            if j - i >= min_n:
                runs.append((i, j))
            i = j
        else:
            i += 1

    min_sep_px = min_sep_deg * pix_per_deg
    kept = []
    for a, b in runs:
        mid = (a + b) // 2
        c = trace[mid]
        if any(np.hypot(c[0] - trace[k, 0], c[1] - trace[k, 1]) < min_sep_px
               for k in kept):
            continue
        kept.append(mid)
    return kept


def _project_screen(ax, image, corners3d, *, source_box=None,
                    roi=None, screen_shape=None,
                    eye_trace_px=None, eye_trace_lw=0.8,
                    eye_trace_color="#e6c43a",
                    roi_sequence_px=None,
                    zorder=2, edge_color="#222", edge_width=0.9,
                    roi_color=CYAN, roi_width=2.0):
    """Render a screen face. Optionally crop the image to `source_box` and
    overlay an eye-trace polyline (in source-image pixel coords).

    `source_box`: ((c0, c1), (r0, r1)) — sub-rectangle of `image` to display.
                  Use this to zoom in (e.g. central 5° on the test screen).
                  If None, the full image is shown.
    """
    dst_quad = cabinet_project(corners3d)

    if source_box is not None:
        (c0, c1), (r0, r1) = source_box
        src_corners = np.array([
            [c0, r1], [c1, r1], [c1, r0], [c0, r0],
        ], dtype=np.float64)
        _draw_quad_image(ax, image, dst_quad, src_corners=src_corners,
                         zorder=zorder, auto_contrast=True)
    else:
        _draw_quad_image(ax, image, dst_quad, zorder=zorder, auto_contrast=True)

    ax.add_patch(Polygon(dst_quad, closed=True, fill=False,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.1))

    # Build the image-pixel → projected-quad homography. When `source_box`
    # is given, the *displayed* region is just that box, so the src for the
    # homography is the box (not the full image).
    if source_box is not None:
        (c0, c1), (r0, r1) = source_box
        src_full = np.array([[c0, r1], [c1, r1], [c1, r0], [c0, r0]],
                            dtype=np.float64)
    else:
        H, W = image.shape[:2]
        src_full = np.array([[0, H], [W, H], [W, 0], [0, 0]],
                            dtype=np.float64)
    H_fwd = _solve_homography(src_full, dst_quad)

    roi_quad = None
    if roi is not None and screen_shape is not None:
        r0, r1 = roi[0]
        c0, c1 = roi[1]
        roi_src = np.array([
            [c0, r1], [c1, r1], [c1, r0], [c0, r0],
        ], dtype=np.float64)
        roi_quad = _apply_h(H_fwd, roi_src)
        ax.add_patch(Polygon(roi_quad, closed=True, fill=False,
                             edgecolor=roi_color, linewidth=roi_width,
                             zorder=zorder + 0.2))

    if eye_trace_px is not None and len(eye_trace_px) > 1:
        trace2d = _apply_h(H_fwd, np.asarray(eye_trace_px, dtype=float))
        ax.add_line(Line2D(trace2d[:, 0], trace2d[:, 1],
                           color=eye_trace_color, linewidth=eye_trace_lw,
                           alpha=0.95, zorder=zorder + 0.15,
                           solid_capstyle="round", solid_joinstyle="round"))

    last_snapshot_quad = None
    if roi_sequence_px is not None and len(roi_sequence_px) > 0:
        seq = np.asarray(roi_sequence_px, dtype=float)
        for r in seq:
            r0, r1 = float(r[0, 0]), float(r[0, 1])
            c0, c1 = float(r[1, 0]), float(r[1, 1])
            src = np.array([[c0, r1], [c1, r1], [c1, r0], [c0, r0]],
                           dtype=np.float64)
            quad = _apply_h(H_fwd, src)
            ax.add_patch(Polygon(quad, closed=True, fill=False,
                                 edgecolor=roi_color,
                                 linewidth=ROI_SNAPSHOT_LW,
                                 alpha=ROI_SNAPSHOT_ALPHA,
                                 zorder=zorder + 0.22))
        last_snapshot_quad = quad

    return dst_quad, roi_quad, H_fwd, last_snapshot_quad


# ──────────────────────────────────────────────────────────────────────────
# Lag cube
# ──────────────────────────────────────────────────────────────────────────
def _draw_lag_cube(ax, cube, corners3d, *, outline=CYAN, edge_width=1.4,
                   zorder=4):
    """Texture front, top, and left faces of a 3D box from a (T,H,W) cube."""
    n_lags, H, W = cube.shape

    vmin, vmax = np.percentile(cube, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0

    def _norm(arr):
        a = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        return (a * 255).astype(np.uint8)

    p2 = cabinet_project(corners3d)
    fLL, fLR, fUR, fUL = p2[0], p2[1], p2[2], p2[3]
    bLL, bLR, bUR, bUL = p2[4], p2[5], p2[6], p2[7]

    # Front face: cube[0] is stored row-0-on-top (PIL convention), which
    # is what _draw_quad_image's default src_corners expect — no flip.
    front_img = _norm(cube[0])
    _draw_quad_image(ax, front_img,
                     np.array([fLL, fLR, fUR, fUL]),
                     zorder=zorder + 0.3)

    # Top face: rows = time, cols = x. PIL's row 0 → image top. For the
    # top quad ordered (fUL, fUR, bUR, bUL), the *image bottom* maps to
    # the front edge of the quad (fUL→fUR). We want front of cube to show
    # cube[0] (most recent), so cube[0]'s row sits at the BOTTOM of the
    # image, which means we need row 0 = oldest (cube[-1]).
    top_img = _norm(cube[::-1, 0, :])
    _draw_quad_image(ax, top_img,
                     np.array([fUL, fUR, bUR, bUL]),
                     zorder=zorder + 0.2)

    # Left face. dst order (fLL, bLL, bUL, fUL) → PIL src corners map:
    #   PIL bottom-left (col 0, row H) → fLL (front-lower-left)
    #   PIL bottom-right (col W, row H) → bLL (back-lower-left)
    #   PIL top-right (col W, row 0) → bUL (back-upper-left)
    #   PIL top-left  (col 0, row 0) → fUL (front-upper-left)
    # cube[:, :, 0] has shape (n_lags, H_cube). Transpose → (H_cube, n_lags):
    # row index = cube y (0 = top-of-scene = high in display); col = time
    # (0 = newest = front of cube; n_lags-1 = oldest = back).
    # PIL row 0 (top of image) must map to TOP of cube (fUL/bUL edge).
    # Cube row 0 IS the top-of-scene → no flip needed.
    left_img = _norm(cube[:, :, 0]).T   # (H_cube, n_lags), no flip
    _draw_quad_image(ax, left_img,
                     np.array([fLL, bLL, bUL, fUL]),
                     zorder=zorder + 0.1)

    for quad in (np.array([fLL, fLR, fUR, fUL]),
                 np.array([fUL, fUR, bUR, bUL]),
                 np.array([fLL, bLL, bUL, fUL])):
        ax.add_patch(Polygon(quad, closed=True, fill=False,
                             edgecolor=outline, linewidth=edge_width,
                             zorder=zorder + 0.5))

    return p2


# ──────────────────────────────────────────────────────────────────────────
# Panel layout
# ──────────────────────────────────────────────────────────────────────────
CANVAS_W = 28.0
CANVAS_H = 12.0

# Screen geometry
SCR_W = 5.6
SCR_H = 4.2
SCR_CY = 6.5

# Training stack — fanned in x AND staggered in z. The front screen sits
# far enough right that the back-most screen (offset left + projected
# up-and-left by depth) doesn't clip the canvas's left edge.
TRAIN_CX_FRONT = 7.8       # world x of the front (natural-image) screen
TRAIN_X_STEP = 1.7         # leftward per layer
TRAIN_Z_STEP = 1.7         # back-into-page per layer

# Test screen — moved closer to training stack.
TEST_CX = 13.6
TEST_ZOOM_DEG = 5.0        # extent of the zoomed test view (degrees)

# Lag cube. Kept compact so cabinet projection of the depth axis doesn't
# blow the model-input zone across the test screen.
CUBE_W = 2.35
CUBE_H = 1.80
CUBE_D = 2.45

# Scale bars (deg)
TRAIN_SCALEBAR_DEG = 10.0
TEST_SCALEBAR_DEG = 2.0
SCALEBAR_FONTSIZE = 9.0


def _project_world_dx_to_2d(dx_world):
    """A horizontal world-x offset projects to an x-shift of the same size
    (cabinet projection is parallel; the z-component is unchanged). Used
    when drawing screen-space scale bars whose physical length is defined
    in screen-degrees → screen-world units.
    """
    return dx_world


def plot_panel_a_stimulus(ax, assets):
    """Render the stimulus half of panel A."""
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    ppd = assets.pix_per_deg
    H, W = assets.screen_shape

    # ── Training stimuli (back → front: gratings, gaborium, natural) ────
    train_order = ["gratings", "gaborium", "backimage"]
    train_keys = [k for k in train_order if k in assets.screens]
    n_train = len(train_keys)
    train_dst_quads = []
    front_screen_homography = None
    for i, key in enumerate(train_keys):
        # i=0 → most back (largest z and most leftward), i=n-1 → frontmost
        layer = n_train - 1 - i
        z = layer * TRAIN_Z_STEP
        x = TRAIN_CX_FRONT - layer * TRAIN_X_STEP
        corners = screen_corners_3d((x, SCR_CY, z), SCR_W, SCR_H)
        # Eye trace + ROI sequence overlay on the natural (front) image only.
        eye_trace = assets.freeview_trace_px if key == "backimage" else None
        if key == "backimage":
            fix_idx = _pick_fixation_rois(
                assets.freeview_trace_px,
                pix_per_deg=assets.pix_per_deg,
            )
            roi_seq = (assets.freeview_roi_seq_px[fix_idx]
                       if len(fix_idx) else None)
            print(f"    [roi seq] backimage fixations kept: {len(fix_idx)}")
        else:
            roi_seq = None
        # Static representative ROI suppressed — the sequence's last
        # snapshot now serves as the "current frame" marker.
        dst, _, H_fwd, _ = _project_screen(
            ax, assets.screens[key], corners,
            screen_shape=assets.screen_shape,
            eye_trace_px=eye_trace,
            eye_trace_lw=1.0,
            eye_trace_color="#ffd84d",
            roi_sequence_px=roi_seq,
            roi_width=1.4,
            zorder=2 + i,
        )
        train_dst_quads.append(dst)
        if key == "backimage":
            front_screen_homography = H_fwd

    # ── Test screen, zoomed to central TEST_ZOOM_DEG ────────────────────
    # Crop is centered on the *screen* (not on the ROI) so the fixRSVP
    # image sits centered in the rendered view; the ROI marker may end up
    # off-centre, which is fine.
    test_corners = screen_corners_3d((TEST_CX, SCR_CY, 0.0), SCR_W, SCR_H)
    fixrsvp_roi = assets.rois.get("fixrsvp")
    screen_ccent_px = W / 2.0
    screen_rcent_px = H / 2.0
    half_px = 0.5 * TEST_ZOOM_DEG * ppd
    zoom_box = (
        (screen_ccent_px - half_px, screen_ccent_px + half_px),
        (screen_rcent_px - half_px, screen_rcent_px + half_px),
    )
    # FEM eye trace: gaze is centred on the ROI in degrees; convert to
    # screen pixels using the ROI centre as anchor.
    if fixrsvp_roi is not None:
        roi_rcent = 0.5 * (fixrsvp_roi[0, 0] + fixrsvp_roi[0, 1])
        roi_ccent = 0.5 * (fixrsvp_roi[1, 0] + fixrsvp_roi[1, 1])
    else:
        roi_rcent, roi_ccent = screen_rcent_px, screen_ccent_px
    fem_px = np.column_stack([
        assets.behavior_eyepos[:, 0] * ppd + roi_ccent,
        -assets.behavior_eyepos[:, 1] * ppd + roi_rcent,   # +y deg → up
    ])

    # Test screen carries the single static ROI used to populate cube[-1]
    # — keeps the magnification-line anchor visually identical to the lag
    # cube's front face.
    test_dst, test_roi_quad, _, _ = _project_screen(
        ax, assets.screens["fixrsvp"], test_corners,
        source_box=zoom_box,
        roi=fixrsvp_roi,
        screen_shape=assets.screen_shape,
        eye_trace_px=fem_px,
        eye_trace_lw=1.8,
        eye_trace_color="#ffd84d",
        zorder=4,
    )

    # ── Vertical dashed separator between training & test ───────────────
    train_front_right_x = max(train_dst_quads[-1][:, 0])
    test_left_x = test_dst[:, 0].min()
    sep_x = 0.5 * (train_front_right_x + test_left_x)
    sep_top = SCR_CY + SCR_H / 2 + 0.2
    sep_bot = SCR_CY - SCR_H / 2 - 0.2
    ax.add_line(Line2D([sep_x, sep_x], [sep_bot, sep_top],
                       color="#666", linewidth=1.1,
                       linestyle=(0, (4, 3)), zorder=2.6))

    # ── Lag cube — positioned fully to the right of the test screen ─────
    # Cube depth goes INTO the page (back face up-and-LEFT of front face),
    # so we place the FRONT face well to the right of the screen and the
    # back face naturally ends up just to the right of the test screen,
    # close to the ROI horizontally. Magnification lines (ROI → back-face
    # corners) read as a horizontal "zoom in" connector.
    if test_roi_quad is not None:
        roi_center_2d = test_roi_quad.mean(axis=0)
    else:
        roi_center_2d = np.array([test_dst[:, 0].mean(), SCR_CY])
    test_right_x = test_dst[:, 0].max()
    # Pick a front-face position so the back face sits just to the right
    # of the test screen with a small gap.
    back_off_2d = _back_face_center_offset_2d(
        CUBE_D, CUBE_YAW_DEG, CUBE_PITCH_DEG, CUBE_ROLL_DEG)
    gap = 2.0
    cube_front_cx = test_right_x + gap - back_off_2d[0]   # back face at test_right_x + gap
    cube_front_cy = roi_center_2d[1] - back_off_2d[1]     # back face vertically at ROI

    cube = assets.lag_cube[::-1]
    cube_corners = box_corners_3d(
        (cube_front_cx, cube_front_cy, 0.0),
        (CUBE_W, CUBE_H, CUBE_D),
        yaw_deg=CUBE_YAW_DEG,
        pitch_deg=CUBE_PITCH_DEG,
        roll_deg=CUBE_ROLL_DEG,
    )
    cube_p2 = _draw_lag_cube(ax, cube, cube_corners,
                             outline=CYAN, edge_width=1.4, zorder=5)
    back_corners_2d = cube_p2[4:]   # back LL, LR, UR, UL

    # ── Four magnification lines: ROI corners → back-face corners ───────
    # test_roi_quad order: LL, LR, UR, UL  (image-derived)
    # back_corners_2d  order: LL, LR, UR, UL
    if test_roi_quad is not None:
        for src, dst in zip(test_roi_quad, back_corners_2d):
            ax.add_line(Line2D([src[0], dst[0]], [src[1], dst[1]],
                               color=CYAN, linewidth=0.9, alpha=0.95,
                               zorder=4.6))

    # ── Headers above each zone ─────────────────────────────────────────
    def _quads_top(quads):
        return max(q[:, 1].max() for q in quads)

    header_y = max(
        _quads_top(train_dst_quads),
        test_dst[:, 1].max(),
    ) + 0.55
    sub_header_y = header_y - 0.45

    train_cx_proj = float(np.mean([q[:, 0].mean() for q in train_dst_quads]))
    ax.text(train_cx_proj, header_y, "Training stimuli",
            ha="center", va="bottom",
            fontsize=10, color=TEXT_COLOR, fontweight="bold")
    ax.text(train_cx_proj, sub_header_y,
            "gratings · gabors · natural images",
            ha="center", va="bottom", fontsize=7.0,
            color="#555", style="italic")

    test_cx_proj = test_dst[:, 0].mean()
    ax.text(test_cx_proj, header_y, "Test stimulus",
            ha="center", va="bottom",
            fontsize=10, color=TEXT_COLOR, fontweight="bold")
    ax.text(test_cx_proj, sub_header_y,
            "fixated image sequence",
            ha="center", va="bottom", fontsize=7.0,
            color="#555", style="italic")

    cube_top_2d = cube_p2[[3, 6, 7], 1].max()
    cube_cx_proj = cube_p2[:, 0].mean()
    ax.text(cube_cx_proj, cube_top_2d + 1.10, "Model input",
            ha="center", va="bottom",
            fontsize=10, color=TEXT_COLOR, fontweight="bold")
    ax.text(cube_cx_proj, cube_top_2d + 0.70,
            "space × space × time", ha="center", va="bottom",
            fontsize=7.0, color="#555", style="italic")

    # ── Scale bars under each stimulus zone ─────────────────────────────
    # Each scale bar starts at the LEFT edge of the screen it measures.
    # World→degree mapping:
    #   • Training (front natural screen): SCR_W world units spans the
    #     full screen image, i.e. W pixels = W / ppd degrees. So one
    #     degree = SCR_W * ppd / W world units.
    #   • Test (zoomed view): SCR_W world units spans TEST_ZOOM_DEG, so
    #     one degree = SCR_W / TEST_ZOOM_DEG world units.
    train_front_quad = train_dst_quads[-1]
    train_world_per_deg = SCR_W * ppd / W
    train_bar_len = TRAIN_SCALEBAR_DEG * train_world_per_deg
    train_bar_x0 = train_front_quad[:, 0].min()
    train_bar_x1 = train_bar_x0 + train_bar_len
    train_bar_y = train_front_quad[:, 1].min() - 0.50
    ax.add_line(Line2D([train_bar_x0, train_bar_x1],
                       [train_bar_y, train_bar_y],
                       color="#222", linewidth=2.0, solid_capstyle="butt",
                       zorder=6))
    ax.text(0.5 * (train_bar_x0 + train_bar_x1), train_bar_y - 0.15,
            f"{TRAIN_SCALEBAR_DEG:g}°", ha="center", va="top",
            fontsize=SCALEBAR_FONTSIZE, color="#222")
    print(f"    [scalebar] training: {TRAIN_SCALEBAR_DEG:g}° = "
          f"{train_bar_len:.3f} world units "
          f"(SCR_W={SCR_W}, W={W}px, ppd={ppd:.2f}px/deg → "
          f"full screen = {W/ppd:.2f}°)")

    test_world_per_deg = SCR_W / TEST_ZOOM_DEG
    test_bar_len = TEST_SCALEBAR_DEG * test_world_per_deg
    test_bar_x0 = test_dst[:, 0].min()
    test_bar_x1 = test_bar_x0 + test_bar_len
    test_bar_y = test_dst[:, 1].min() - 0.50
    ax.add_line(Line2D([test_bar_x0, test_bar_x1],
                       [test_bar_y, test_bar_y],
                       color="#222", linewidth=2.0, solid_capstyle="butt",
                       zorder=6))
    ax.text(0.5 * (test_bar_x0 + test_bar_x1), test_bar_y - 0.15,
            f"{TEST_SCALEBAR_DEG:g}°", ha="center", va="top",
            fontsize=SCALEBAR_FONTSIZE, color="#222")
    print(f"    [scalebar] test:     {TEST_SCALEBAR_DEG:g}° = "
          f"{test_bar_len:.3f} world units "
          f"(zoom = {TEST_ZOOM_DEG:g}° across SCR_W={SCR_W})")

    # ── Cube time-axis annotation ───────────────────────────────────────
    n_lags = cube.shape[0]
    ms = n_lags / 120.0 * 1000.0
    p_front_bot = cube_p2[0] + np.array([0.0, -0.40])
    p_back_bot = cube_p2[4] + np.array([0.0, -0.40])
    ax.annotate(
        "", xy=p_back_bot, xytext=p_front_bot,
        arrowprops=dict(arrowstyle="<-", lw=0.9, color=ARROW_COLOR),
    )
    mid = 0.5 * (p_front_bot + p_back_bot) + np.array([0.0, -0.25])
    ax.text(mid[0], mid[1], f"{ms:.0f} ms",
            ha="center", va="top",
            fontsize=7, color=TEXT_COLOR, style="italic")
    ax.text(mid[0], mid[1] - 0.40, "(120 Hz)",
            ha="center", va="top",
            fontsize=6.5, color="#555", style="italic")

    # ── Front-face spatial extent labels (width below, height to left) ──
    # Cube front face spans the same pixel ROI as the model input;
    # convert to degrees using the screen pixels-per-degree.
    cube_h_deg = cube.shape[1] / ppd
    cube_w_deg = cube.shape[2] / ppd
    fLL, fLR, fUR, fUL = cube_p2[0], cube_p2[1], cube_p2[2], cube_p2[3]
    # Width label: below the front face bottom edge, outside the silhouette.
    width_mid = 0.5 * (fLL + fLR) + np.array([0.0, -0.18])
    ax.text(width_mid[0], width_mid[1], f"{cube_w_deg:.1f}°",
            ha="center", va="top", fontsize=7,
            color=TEXT_COLOR, style="italic")
    # Height label: just outside the front face's right edge.
    height_mid = 0.5 * (fLR + fUR) + np.array([0.18, 0.0])
    ax.text(height_mid[0], height_mid[1], f"{cube_h_deg:.1f}°",
            ha="left", va="center", fontsize=7,
            color=TEXT_COLOR, style="italic", rotation=90)

    # ── Tighten axes limits to the actual drawn content ─────────────────
    xs, ys = [], []
    for q in train_dst_quads + [test_dst, cube_p2]:
        xs.extend([q[:, 0].min(), q[:, 0].max()])
        ys.extend([q[:, 1].min(), q[:, 1].max()])
    ys.append(header_y + 0.5)         # top of headers
    ys.append(cube_top_2d + 1.4)      # top of Model input header
    ys.append(min(train_bar_y, test_bar_y) - 0.7)   # below scalebar labels
    ys.append(mid[1] - 0.3)           # below cube time-axis label
    pad_x, pad_y = 0.4, 0.2
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
