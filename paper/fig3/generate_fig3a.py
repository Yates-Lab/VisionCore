"""Figure 3 panel A — integrated digital-twin schematic (single frame).

One matplotlib axes, one shared world coordinate frame, drawn left → right:

  * Stimulus block (upper-left): a shrunk, lightly-staggered training stack
    (gratings · gabors · natural image) with the test stimulus moved BELOW it
    and left-aligned, plus the "Model input" space×space×time lag cube linked
    to the test ROI.
  * Behavior traces (lower-left): the true extraretinal covariates — eye
    position (x, y) and velocity — drawn beneath the model input with labels
    and scale bars, then collected by a bracket.
  * Architecture (right): Frontend → Stem → ResBlock 1 → ResBlock 2 → ConvGRU
    → Readouts as cabinet-projected kernel prisms.
  * Behavior bridge: an arrow runs from the trace bracket across to the
    concatenation marker on the ResBlock2→ConvGRU flow, carrying a
    zero-ablation indicator (the extraretinal input can be zeroed).
  * Readout PSTHs: to the right of each readout, an observed-vs-predicted PSTH
    for one of the best-CCnorm units.

Because everything shares one frame, the behavior signal can be drawn once (as
real traces) and routed across to the model — the point of the figure: the
twin barely needs the extraretinal covariate.

This module replaces the former `_fig3a_stimulus` + `_fig3a_architecture`
split; low-level primitives still live in `_fig3a_glyphs`.

Usage:
    uv run python paper/fig3/generate_fig3a.py [--recompute]
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image as PILImage, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

from _fig3_data import FIG_DIR, configure_matplotlib
from _fig3a_data import load_panel_a_assets
from _fig3a_glyphs import (
    CYAN, ARROW_COLOR, TEXT_COLOR,
    _perspective_coeffs,
    draw_channel_grid, draw_recurrent_loop, draw_feature_weight_block,
    draw_spatial_readout_prism, draw_pool_glyph, draw_arrow_skip,
    draw_op_marker, draw_neuron_trace_panel,
)


# ════════════════════════════════════════════════════════════════════════════
# STIMULUS HALF — cabinet projection + geometry (world +x right, +y up,
# +z INTO the page projecting up-and-LEFT at half scale).
# ════════════════════════════════════════════════════════════════════════════
CABINET_ALPHA = np.deg2rad(45.0)
CABINET_DEPTH = 0.5
DEPTH_VEC = np.array([
    -np.cos(CABINET_ALPHA) * CABINET_DEPTH,
    +np.sin(CABINET_ALPHA) * CABINET_DEPTH,
])

SCREEN_YAW_DEG = -22.0
CUBE_YAW_DEG = -40.0
CUBE_PITCH_DEG = 5.0
CUBE_ROLL_DEG = -5.0


def cabinet_project(p3):
    """Project (N×3) (or shape-(3,)) world points to 2D screen coords."""
    p3 = np.asarray(p3, dtype=float)
    return p3[..., :2] + p3[..., 2:3] * DEPTH_VEC


def _R_y(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _R_x(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _R_z(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_rotation(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    return _R_y(yaw_deg) @ _R_x(pitch_deg) @ _R_z(roll_deg)


def screen_corners_3d(center, width, height, *, yaw_deg=SCREEN_YAW_DEG):
    """World corners (LL, LR, UR, UL) of an upright, yawed rectangle."""
    cx, cy, cz = center
    w2, h2 = width / 2.0, height / 2.0
    local = np.array([
        [-w2, -h2, 0.0], [+w2, -h2, 0.0], [+w2, +h2, 0.0], [-w2, +h2, 0.0],
    ])
    return local @ _R_y(yaw_deg).T + np.array([cx, cy, cz])


def box_corners_3d(front_center, size, *, yaw_deg=SCREEN_YAW_DEG,
                   pitch_deg=0.0, roll_deg=0.0):
    """World corners of a rotated axis-aligned box (front-face center anchor)."""
    cx, cy, cz = front_center
    w, h, d = size
    w2, h2 = w / 2.0, h / 2.0
    local = np.array([
        [-w2, -h2, 0.0], [+w2, -h2, 0.0], [+w2, +h2, 0.0], [-w2, +h2, 0.0],
        [-w2, -h2, d],   [+w2, -h2, d],   [+w2, +h2, d],   [-w2, +h2, d],
    ])
    R = _euler_rotation(yaw_deg, pitch_deg, roll_deg)
    return local @ R.T + np.array([cx, cy, cz])


def _back_face_center_offset_2d(depth, yaw_deg=SCREEN_YAW_DEG,
                                pitch_deg=0.0, roll_deg=0.0):
    local = np.array([0.0, 0.0, depth])
    world = _euler_rotation(yaw_deg, pitch_deg, roll_deg) @ local
    return cabinet_project(world)


# ── Textured-quad rendering ────────────────────────────────────────────────
def _draw_quad_image(ax, image, dst_quad, *, src_corners=None,
                     zorder=2, alpha=1.0, auto_contrast=False, out_res=512):
    """Warp `image` into `dst_quad` (4×2 data coords, LL,LR,UR,UL)."""
    H, W = image.shape[:2]
    if src_corners is None:
        src_corners = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float64)

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

    warped = pil.transform((out_res, out_res), PILImage.PERSPECTIVE, coeffs,
                           resample=PILImage.BILINEAR)
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


ROI_SNAPSHOT_ALPHA = 0.85
ROI_SNAPSHOT_LW = 1.0


def _pick_fixation_rois(trace_px, *, fs=120.0, pix_per_deg=None,
                        vel_thresh_deg_s=20.0, min_dur_s=0.10,
                        min_sep_deg=1.0):
    """Pick one sample index per fixation in a gaze trace (see notes below)."""
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
                    eye_trace_color="#e6c43a", roi_sequence_px=None,
                    zorder=2, edge_color="#222", edge_width=0.9,
                    roi_color=CYAN, roi_width=2.0):
    """Render a screen face, optionally cropped and with an eye-trace overlay."""
    dst_quad = cabinet_project(corners3d)

    if source_box is not None:
        (c0, c1), (r0, r1) = source_box
        src_corners = np.array([[c0, r1], [c1, r1], [c1, r0], [c0, r0]],
                               dtype=np.float64)
        _draw_quad_image(ax, image, dst_quad, src_corners=src_corners,
                         zorder=zorder, auto_contrast=True)
    else:
        _draw_quad_image(ax, image, dst_quad, zorder=zorder, auto_contrast=True)

    ax.add_patch(Polygon(dst_quad, closed=True, fill=False,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.1))

    if source_box is not None:
        (c0, c1), (r0, r1) = source_box
        src_full = np.array([[c0, r1], [c1, r1], [c1, r0], [c0, r0]],
                            dtype=np.float64)
    else:
        H, W = image.shape[:2]
        src_full = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float64)
    H_fwd = _solve_homography(src_full, dst_quad)

    roi_quad = None
    if roi is not None and screen_shape is not None:
        r0, r1 = roi[0]
        c0, c1 = roi[1]
        roi_src = np.array([[c0, r1], [c1, r1], [c1, r0], [c0, r0]],
                           dtype=np.float64)
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
                                 edgecolor=roi_color, linewidth=ROI_SNAPSHOT_LW,
                                 alpha=ROI_SNAPSHOT_ALPHA, zorder=zorder + 0.22))
        last_snapshot_quad = quad

    return dst_quad, roi_quad, H_fwd, last_snapshot_quad


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

    front_img = _norm(cube[0])
    _draw_quad_image(ax, front_img, np.array([fLL, fLR, fUR, fUL]),
                     zorder=zorder + 0.3)
    top_img = _norm(cube[::-1, 0, :])
    _draw_quad_image(ax, top_img, np.array([fUL, fUR, bUR, bUL]),
                     zorder=zorder + 0.2)
    left_img = _norm(cube[:, :, 0]).T
    _draw_quad_image(ax, left_img, np.array([fLL, bLL, bUL, fUL]),
                     zorder=zorder + 0.1)

    for quad in (np.array([fLL, fLR, fUR, fUL]),
                 np.array([fUL, fUR, bUR, bUL]),
                 np.array([fLL, bLL, bUL, fUL])):
        ax.add_patch(Polygon(quad, closed=True, fill=False,
                             edgecolor=outline, linewidth=edge_width,
                             zorder=zorder + 0.5))
    return p2


# ── Stimulus layout constants (shared world units; architecture kernels use
#    S_PIX = 0.03 world units/tap, so screens are sized to read at a matched
#    scale). Training screens now match the test screen and stagger widely so
#    the top row fills horizontally. ─────────────────────────────────────────
TEST_W = 3.35 * 0.9         # screens shrunk ~10% (smaller panel A vs panel B)
TEST_H = 2.55 * 0.9
SCR_W = TEST_W              # training screens match the test screen size
SCR_H = TEST_H
TRAIN_CX_FRONT = 4.2        # world x of the front (natural-image) screen
TRAIN_X_STEP = 1.75         # leftward per back layer (wide horizontal stagger)
TRAIN_Z_STEP = 0.42         # back-into-page per layer (sets the up-left skew)

TEST_ZOOM_DEG = 5.0
TOP_TRAIN_TEST_GAP = 3.2    # 2D gap: training block right edge → test left edge
                            # (leaves room for the dashed train/test divider)
TOP_TITLE_SUB_GAP = 0.42    # gap: image top → subtitle (breathing room)
TOP_TITLE_NAME_GAP = 0.30   # gap: subtitle → bold title

CUBE_W = 1.55               # model-input cube — enlarged to read as a feature
CUBE_H = 1.18
CUBE_D = 1.62
CUBE_GAP = 1.7
MODEL_COL_LIFT = 0.62       # raise the panel-A model-input cube so the cube +
                            # behavior echo column sits centred under the header
                            # (fills the gap above, un-dangles the behavior below)

TRAIN_SCALEBAR_DEG = 10.0
TEST_SCALEBAR_DEG = 2.0
SCALEBAR_FONTSIZE = 6.0
STIM_HEADER_FS = 7.5        # match panel B stage names (ARCH_NAME_FS)
STIM_SUB_FS = 6.0           # match panel B stage sub-captions (ARCH_SUB_FS)


def _draw_cube_block(ax, assets, front_center_2d, *, roi_quad_2d=None,
                     draw_header=True, draw_dims=True, draw_time=None,
                     time_label=None, cube_override=None, outline=CYAN):
    """Draw the model-input lag cube with its front-face centre at
    `front_center_2d` (2D world coords).

    Optionally draws magnification lines from a test-ROI quad to the cube back
    face, the "Model input" header, the depth (time) arrow, and the spatial
    dimension labels. `draw_time` toggles the time arrow (defaults to
    `draw_dims`); `time_label` overrides its caption (defaults to the cube's
    duration). `cube_override` supplies a (n_lags, H, W) volume in place of the
    stored lag cube (e.g. a temporally frozen "stabilized" version), and
    `outline` recolours the cube edges. The cube's 3D pose is fixed so it reads
    identically wherever it is placed. Returns an anchor dict.
    """
    if draw_time is None:
        draw_time = draw_dims
    ppd = assets.pix_per_deg
    fcx, fcy = float(front_center_2d[0]), float(front_center_2d[1])
    cube = (assets.lag_cube if cube_override is None else cube_override)[::-1]
    cube_corners = box_corners_3d(
        (fcx, fcy, 0.0), (CUBE_W, CUBE_H, CUBE_D),
        yaw_deg=CUBE_YAW_DEG, pitch_deg=CUBE_PITCH_DEG, roll_deg=CUBE_ROLL_DEG)
    cube_p2 = _draw_lag_cube(ax, cube, cube_corners, outline=outline,
                             edge_width=1.2, zorder=5)
    back_corners_2d = cube_p2[4:]

    # Magnification lines: ROI corners → back-face corners.
    if roi_quad_2d is not None:
        for src, dst in zip(roi_quad_2d, back_corners_2d):
            ax.add_line(Line2D([src[0], dst[0]], [src[1], dst[1]],
                               color=CYAN, linewidth=0.7, alpha=0.95,
                               zorder=4.6))

    cube_top_2d = float(cube_p2[[3, 6, 7], 1].max())
    cube_cx_proj = float(cube_p2[:, 0].mean())
    y_top = cube_top_2d + 0.15
    if draw_header:
        ax.text(cube_cx_proj, cube_top_2d + 0.72, "Model input",
                ha="center", va="bottom", fontsize=STIM_HEADER_FS,
                color=TEXT_COLOR, fontweight="bold")
        ax.text(cube_cx_proj, cube_top_2d + 0.50, "space × space × time",
                ha="center", va="bottom", fontsize=STIM_SUB_FS,
                color="#555", style="italic")
        y_top = cube_top_2d + 1.05

    y_bottom = float(cube_p2[:, 1].min()) - 0.05
    if draw_time:
        if time_label is None:
            time_label = f"{cube.shape[0] / 120.0 * 1000.0:.0f} ms (120 Hz)"
        p_front_bot = cube_p2[0] + np.array([0.0, -0.22])
        p_back_bot = cube_p2[4] + np.array([0.0, -0.22])
        ax.annotate("", xy=p_back_bot, xytext=p_front_bot,
                    arrowprops=dict(arrowstyle="<-", lw=0.8, color=ARROW_COLOR))
        # Rotate the caption to the arrow's angle and sit it just below the shaft.
        d = p_back_bot - p_front_bot
        L = float(np.hypot(d[0], d[1])) or 1.0
        perp = np.array([d[1], -d[0]]) / L
        if perp[1] > 0:
            perp = -perp
        angle = float(np.degrees(np.arctan2(d[1], d[0])))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        label_pos = 0.5 * (p_front_bot + p_back_bot) + perp * 0.20
        ax.text(label_pos[0], label_pos[1], time_label, ha="center",
                va="center", rotation=angle, rotation_mode="anchor",
                fontsize=6.0, color="#555", style="italic")
        y_bottom = float(label_pos[1] - 0.20)

    if draw_dims:
        cube_w_deg = cube.shape[2] / ppd
        cube_h_deg = cube.shape[1] / ppd
        fLL, fLR, fUR = cube_p2[0], cube_p2[1], cube_p2[2]
        width_mid = 0.5 * (fLL + fLR) + np.array([0.0, -0.10])
        ax.text(width_mid[0], width_mid[1], f"{cube_w_deg:.1f}°", ha="center",
                va="top", fontsize=6.0, color=TEXT_COLOR, style="italic")
        height_mid = 0.5 * (fLR + fUR) + np.array([0.10, 0.0])
        ax.text(height_mid[0], height_mid[1], f"{cube_h_deg:.1f}°", ha="left",
                va="center", fontsize=6.0, color=TEXT_COLOR, style="italic",
                rotation=90)

    return {
        "cube_p2": cube_p2,
        "x_left": float(cube_p2[:, 0].min()),
        "x_right": float(cube_p2[:, 0].max()),
        "mid_y": float(0.5 * (cube_p2[:, 1].min() + cube_p2[:, 1].max())),
        "top_y": y_top,
        "bottom_y": y_bottom,
    }


def _draw_top_row(ax, assets, row_cy):
    """Draw the stimulus-provenance row centred vertically on `row_cy`:
    training stack (left) · test stimulus (middle) · model-input cube (right,
    3rd column) with ROI magnification lines, plus the extraretinal behavior
    covariates drawn *bare* (no labels, scale bar only) beneath the cube. The
    labelled copy of those covariates is drawn again in the model row below.
    Returns the row's bounding box.
    """
    ppd = assets.pix_per_deg
    H, W = assets.screen_shape

    # ── Training stack (back → front: gratings, gabors, natural) ────────────
    train_order = ["gratings", "gaborium", "backimage"]
    train_keys = [k for k in train_order if k in assets.screens]
    n_train = len(train_keys)
    train_dst_quads = []
    for i, key in enumerate(train_keys):
        layer = n_train - 1 - i               # i=0 → back-most, i=n-1 → front
        z = layer * TRAIN_Z_STEP
        x = TRAIN_CX_FRONT - layer * TRAIN_X_STEP
        corners = screen_corners_3d((x, row_cy, z), SCR_W, SCR_H)
        eye_trace = assets.freeview_trace_px if key == "backimage" else None
        if key == "backimage":
            fix_idx = _pick_fixation_rois(assets.freeview_trace_px,
                                          pix_per_deg=assets.pix_per_deg)
            roi_seq = (assets.freeview_roi_seq_px[fix_idx]
                       if len(fix_idx) else None)
        else:
            roi_seq = None
        dst, _, _, _ = _project_screen(
            ax, assets.screens[key], corners,
            screen_shape=assets.screen_shape,
            eye_trace_px=eye_trace, eye_trace_lw=0.9,
            eye_trace_color="#ffd84d", roi_sequence_px=roi_seq,
            roi_width=1.1, zorder=2 + i,
        )
        train_dst_quads.append(dst)

    train_xs = np.concatenate([q[:, 0] for q in train_dst_quads])
    train_ys = np.concatenate([q[:, 1] for q in train_dst_quads])
    train_max_x = float(train_xs.max())
    train_top_y = float(train_ys.max())

    # ── Test screen — to the RIGHT of the training stack, same row centre ───
    test_cx = (train_max_x + TOP_TRAIN_TEST_GAP
               + 0.5 * TEST_W * np.cos(np.deg2rad(SCREEN_YAW_DEG)))
    test_cy = row_cy
    test_corners = screen_corners_3d((test_cx, test_cy, 0.0), TEST_W, TEST_H)

    fixrsvp_roi = assets.rois.get("fixrsvp")
    screen_ccent_px = W / 2.0
    screen_rcent_px = H / 2.0
    half_px = 0.5 * TEST_ZOOM_DEG * ppd
    zoom_box = (
        (screen_ccent_px - half_px, screen_ccent_px + half_px),
        (screen_rcent_px - half_px, screen_rcent_px + half_px),
    )
    if fixrsvp_roi is not None:
        roi_rcent = 0.5 * (fixrsvp_roi[0, 0] + fixrsvp_roi[0, 1])
        roi_ccent = 0.5 * (fixrsvp_roi[1, 0] + fixrsvp_roi[1, 1])
    else:
        roi_rcent, roi_ccent = screen_rcent_px, screen_ccent_px
    fem_px = np.column_stack([
        assets.behavior_eyepos[:, 0] * ppd + roi_ccent,
        -assets.behavior_eyepos[:, 1] * ppd + roi_rcent,
    ])

    test_dst, test_roi_quad, _, _ = _project_screen(
        ax, assets.screens["fixrsvp"], test_corners,
        source_box=zoom_box, roi=fixrsvp_roi,
        screen_shape=assets.screen_shape,
        eye_trace_px=fem_px, eye_trace_lw=1.4,
        eye_trace_color="#ffd84d", zorder=4,
    )

    # ── Lag cube — 3rd column, to the right of the test screen ──────────────
    if test_roi_quad is not None:
        roi_center_2d = test_roi_quad.mean(axis=0)
    else:
        roi_center_2d = np.array([test_dst[:, 0].mean(), test_cy])
    test_right_x = test_dst[:, 0].max()
    back_off_2d = _back_face_center_offset_2d(
        CUBE_D, CUBE_YAW_DEG, CUBE_PITCH_DEG, CUBE_ROLL_DEG)
    cube_front_cx = test_right_x + CUBE_GAP - back_off_2d[0]
    cube_front_cy = roi_center_2d[1] - back_off_2d[1] + MODEL_COL_LIFT
    cube_block = _draw_cube_block(
        ax, assets, (cube_front_cx, cube_front_cy),
        roi_quad_2d=test_roi_quad, draw_header=False, draw_dims=True)

    # ── Dashed divider between the training and test stimulus groups ─────────
    divider_x = 0.5 * (train_max_x + test_dst[:, 0].min())
    divider_half_h = 0.5 * TEST_H + 0.55
    ax.add_line(Line2D([divider_x, divider_x],
                       [row_cy - divider_half_h, row_cy + divider_half_h],
                       color="#b0b0b0", lw=1.0, ls=(0, (4, 4)), zorder=1.5))

    # ── Headers: all three (title + subtitle) share one vertical baseline,
    #    set above the tallest image top so nothing crowds the screens. ───────
    train_cx_proj = float(np.mean([q[:, 0].mean() for q in train_dst_quads]))
    test_cx_proj = float(test_dst[:, 0].mean())
    cube_cx_proj = float(0.5 * (cube_block["x_left"] + cube_block["x_right"]))
    cube_top_2d = float(cube_block["cube_p2"][:, 1].max())

    common_top = max(train_top_y, float(test_dst[:, 1].max()), cube_top_2d)
    sub_y = common_top + TOP_TITLE_SUB_GAP
    title_y = sub_y + TOP_TITLE_NAME_GAP

    def _stim_header(cx, name, sub):
        ax.text(cx, title_y, name, ha="center", va="bottom",
                fontsize=STIM_HEADER_FS, color=TEXT_COLOR, fontweight="bold")
        ax.text(cx, sub_y, sub, ha="center", va="bottom", fontsize=STIM_SUB_FS,
                color="#555", style="italic")

    _stim_header(train_cx_proj, "Training stimuli",
                 "gratings · gabors · natural images")
    _stim_header(test_cx_proj, "Test stimulus", "fixated flashed images")
    _stim_header(cube_cx_proj, "Visual input", "space × space × time")

    # ── Scale bars ──────────────────────────────────────────────────────────
    train_front_quad = train_dst_quads[-1]
    train_world_per_deg = SCR_W * ppd / W
    train_bar_len = TRAIN_SCALEBAR_DEG * train_world_per_deg
    train_bar_x0 = train_front_quad[:, 0].min()
    train_bar_y = train_front_quad[:, 1].min() - 0.02
    ax.add_line(Line2D([train_bar_x0, train_bar_x0 + train_bar_len],
                       [train_bar_y, train_bar_y], color="#222", linewidth=1.8,
                       solid_capstyle="butt", zorder=6))
    ax.text(train_bar_x0 + train_bar_len / 2, train_bar_y - 0.06,
            f"{TRAIN_SCALEBAR_DEG:g}°", ha="center", va="top",
            fontsize=SCALEBAR_FONTSIZE, color="#222")

    test_world_per_deg = TEST_W / TEST_ZOOM_DEG
    test_bar_len = TEST_SCALEBAR_DEG * test_world_per_deg
    test_bar_x0 = test_dst[:, 0].min()
    test_bar_y = test_dst[:, 1].min() - 0.10
    ax.add_line(Line2D([test_bar_x0, test_bar_x0 + test_bar_len],
                       [test_bar_y, test_bar_y], color="#222", linewidth=1.8,
                       solid_capstyle="butt", zorder=6))
    ax.text(test_bar_x0 + test_bar_len / 2, test_bar_y - 0.06,
            f"{TEST_SCALEBAR_DEG:g}°", ha="center", va="top",
            fontsize=SCALEBAR_FONTSIZE, color="#222")

    # ── Behavioral input beneath the visual-input cube ──────────────────────
    # Panel A is where the two model inputs are named and labelled: the visual
    # cube above and the extraretinal behavior signals here (eye position x/y +
    # eye velocity). Panel B shows the same signals bare, flowing into the model.
    beh_x0 = cube_block["x_left"]
    beh_w = cube_block["x_right"] - cube_block["x_left"]
    beh_cx = 0.5 * (cube_block["x_left"] + cube_block["x_right"])
    beh_title_y = cube_block["bottom_y"] - 0.30
    ax.text(beh_cx, beh_title_y, "Behavioral input", ha="center", va="top",
            fontsize=STIM_HEADER_FS, color=TEXT_COLOR, fontweight="bold")
    ax.text(beh_cx, beh_title_y - 0.22, "signals × time", ha="center", va="top",
            fontsize=STIM_SUB_FS, color="#555", style="italic")
    beh_top = beh_title_y - 0.52
    beh_row = _draw_behavior_traces(ax, assets, beh_x0, beh_top, beh_w,
                                    labeled=True, scale_bar=True, box=False,
                                    trace_h=0.95)

    # ── Collect bbox ────────────────────────────────────────────────────────
    xs, ys = [], []
    for q in train_dst_quads + [test_dst, cube_block["cube_p2"]]:
        xs.extend([q[:, 0].min(), q[:, 0].max()])
        ys.extend([q[:, 1].min(), q[:, 1].max()])
    xs.extend([beh_row["x_left"], beh_row["x_right"]])
    ys.append(title_y + 0.35)
    ys.append(cube_block["top_y"])
    ys.append(min(train_bar_y, test_bar_y) - 0.25)
    ys.append(cube_block["bottom_y"])
    ys.append(beh_row["y_bottom"])

    return {
        "x_left": float(min(xs)),
        "x_right": float(max(xs)),
        "y_bottom": float(min(ys)),
        "y_top": float(max(ys)),
    }


# ════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE HALF — kernel prisms in cabinet projection (depth up-and-RIGHT).
# ════════════════════════════════════════════════════════════════════════════
S_PIX = 0.030
S_T = S_PIX
FE_SCALE = 2.0
STEM_SCALE = 1.5
GRID_GAP = 0.04
FE_GAP = 0.06

# Wider inter-stage gaps let the digital-twin row span the full panel width
# and give the stage labels room to grow.
DEFAULT_GAPS = {
    "fe_to_stem": 0.85,
    "stem_to_blk1": 0.70,
    "blk1_to_blk2": 2.15,       # extra room between ResBlock 1 and ResBlock 2
    "blk2_to_gru": 1.55,
    "gru_to_readout": 1.30,
}
ARCH_CENTER_Y = 7.0

# Readout column (feature block + prism) and downstream PSTH prediction scale,
# relative to their base sizes. Enlarged so the stack reads at the full row
# height; the column is then dropped so its title aligns with ResBlock 2's.
READOUT_SCALE = 1.5

# Architecture label font sizes (bumped so the row reads at full width).
ARCH_TITLE_FS = 10.0            # "Digital twin"
ARCH_NAME_FS = 7.5             # stage names (Frontend, ResBlock 1, …)
ARCH_SUB_FS = 6.0             # stage sub-captions (kernel · channels)

LABEL_GAP = 0.18
SUB_GAP = 0.32
HEADER_GAP = 0.55

SKIP_DEPTH = 1.8
SKIP_COLOR = "#222"
SKIP_LW = 1.0
SKIP_CORNER_R = 0.12

# Circled-operator markers (residual sum + / concat ||) and the ablation-switch
# indicator — sized up from the original 0.10/0.13 so the glyphs read clearly.
OP_MARKER_RADIUS = 0.15
ABLATE_RADIUS = 0.18

# Condition colours for the ablation switch, matched to figure3's INTACT/ABLATED
# violin colours so the schematic's two switch states read as the same two
# conditions quantified in panels C/D/E (closed = full, open = ablated).
COND_FULL_COLOR = "#1f77b4"
COND_ABLATED_COLOR = "#d62728"

PAL_FRONTEND = ("#fff2cc", "#e6c97a", "#c9a945", "#7a5e10")
PAL_STEM = ("#d8e4f4", "#9fbdda", "#6a8db0", "#1b3a5b")
PAL_BLOCK1 = ("#cfe2f3", "#7fa4c4", "#4d7396", "#1b3a5b")
PAL_BLOCK2 = ("#b9d3eb", "#6790b8", "#3a6189", "#0e2b4d")
PAL_GRU = ("#ead6f5", "#b685d3", "#7e3f8a", "#3e1a48")

_CAB_ALPHA = np.deg2rad(45.0)
_CAB_DEPTH = 0.50
_CAB_DEPTH_VEC = np.array([
    +np.cos(_CAB_ALPHA) * _CAB_DEPTH,
    +np.sin(_CAB_ALPHA) * _CAB_DEPTH,
])


def _y0_for_center(*, rows, kh, gap, cols=1, kw=None, center_y=ARCH_CENTER_Y):
    if kw is None:
        kw = kh
    front_h = (rows - 1) * (kh + gap) + kh
    z_extent = (cols - 1) * (kw + gap) + kw
    depth_y = abs(_CAB_DEPTH_VEC[1]) * z_extent
    total_h = front_h + depth_y
    return center_y - total_h / 2


def _next_x(grid, gap):
    return grid["bbox2d"][1] + gap


def _stage_label_top(ax, grid, *, name, sub=None, name_fs=ARCH_NAME_FS,
                     sub_fs=ARCH_SUB_FS, y_top_override=None):
    xmin, xmax, _, ymax = grid["bbox2d"]
    front_xmid = (xmin + xmax) / 2
    front_top = y_top_override if y_top_override is not None else ymax
    if sub:
        y_sub = front_top + LABEL_GAP
        y_title = y_sub + SUB_GAP
        ax.text(front_xmid, y_sub, sub, ha="center", va="baseline",
                fontsize=sub_fs, color="#555", style="italic", linespacing=1.05)
    else:
        y_title = front_top + LABEL_GAP
    ax.text(front_xmid, y_title, name, ha="center", va="baseline",
            fontsize=name_fs, color=TEXT_COLOR, fontweight="bold")
    return y_title


def _stage_right_anchor(rec):
    if "grid" in rec:
        xmin, xmax, ymin, ymax = rec["grid"]["bbox2d"]
        return np.array([xmax, 0.5 * (ymin + ymax)])
    ro = rec["ro"]
    return np.array([ro["x_right"], (ro["y_bottom"] + ro["y_top"]) / 2])


def _stage_left_anchor(rec):
    if "grid" in rec:
        g = rec["grid"]
        _, _, ymin, ymax = g["bbox2d"]
        return np.array([g["x_left"], 0.5 * (ymin + ymax)])
    ro = rec["ro"]
    return np.array([ro["x_left"], (ro["y_bottom"] + ro["y_top"]) / 2])


def _connect_stages(ax, stage_records):
    arrows = {}
    for i in range(len(stage_records) - 1):
        a = stage_records[i]
        b = stage_records[i + 1]
        ax0 = _stage_right_anchor(a)
        bx0 = _stage_left_anchor(b)
        x_start = ax0[0] + 0.04
        x_end = bx0[0] - 0.04
        y_mid = 0.5 * (ax0[1] + bx0[1])
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"),
                    zorder=4.8)
        arrows[f"{a['name']}→{b['name']}"] = {
            "x_start": x_start, "x_end": x_end, "y": y_mid}
    return arrows


def _arrow_frac(arrow, frac):
    return (arrow["x_start"] + frac * (arrow["x_end"] - arrow["x_start"]),
            arrow["y"])


def _draw_architecture(ax, assets, *, x_start, gaps=None):
    """Draw the model-architecture pipeline; return anchor dict.

    Unlike the former standalone half, this sets no axis limits, omits the
    local Behavior box (the covariate is now supplied by the trace bridge),
    and returns the concat-marker position, readout-row centers, and bbox so
    the composer can wire the behavior arrow and readout PSTHs.
    """
    g = dict(DEFAULT_GAPS)
    if gaps:
        g.update(gaps)

    arch = assets.arch
    arch_kernels = arch["convnet_kernels"]
    blk1_kt, blk1_kh, blk1_kw = arch_kernels[0]
    blk2_kt, blk2_kh, blk2_kw = arch_kernels[1]
    stem_kt, stem_kh, stem_kw = (1, 7, 7)
    gru_kt, gru_kh, gru_kw = (1, arch["gru_kernel"], arch["gru_kernel"])

    stage_records = []
    label_tops = []
    x_cursor = float(x_start)

    # ── Frontend ────────────────────────────────────────────────────────────
    fe_unit = S_PIX * FE_SCALE
    fe_kt = arch["frontend_k"] * fe_unit
    fe_kh = fe_unit
    fe_kw = fe_unit
    fe_n = arch["frontend_channels"]
    fe_y0 = _y0_for_center(rows=fe_n, kh=fe_kh, gap=FE_GAP, cols=1, kw=fe_kw)
    fe_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=fe_y0, z0=0.0, n_channels=fe_n, rows=fe_n,
        cols=1, kt=fe_kt, kh=fe_kh, kw=fe_kw, gap=FE_GAP, palette=PAL_FRONTEND,
        base_zorder=2.0, edge_width=0.45)

    label_tops.append(_stage_label_top(
        ax, fe_grid, name="Frontend", sub=f"{fe_n} ch · k={arch['frontend_k']}"))
    stage_records.append({"name": "frontend", "grid": fe_grid})
    x_cursor = _next_x(fe_grid, g["fe_to_stem"])

    # ── Stem ──────────────────────────────────────────────────────────────
    stem_kt_w = stem_kt * S_T * STEM_SCALE
    stem_kh_w = stem_kh * S_PIX * STEM_SCALE
    stem_kw_w = stem_kw * S_PIX * STEM_SCALE
    stem_y0 = _y0_for_center(rows=2, kh=stem_kh_w, gap=GRID_GAP, cols=4,
                             kw=stem_kw_w)
    stem_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=stem_y0, z0=0.0, n_channels=8, rows=2, cols=4,
        kt=stem_kt_w, kh=stem_kh_w, kw=stem_kw_w, gap=GRID_GAP, palette=PAL_STEM,
        base_zorder=2.0, edge_width=0.25, hue_jitter=0.08)
    label_tops.append(_stage_label_top(
        ax, stem_grid, name="Stem", sub=f"{stem_kt}×{stem_kh}×{stem_kw} · 8 ch"))
    stage_records.append({"name": "stem", "grid": stem_grid})
    x_cursor = _next_x(stem_grid, g["stem_to_blk1"])

    # ── ResBlock 1 ─────────────────────────────────────────────────────────
    blk1_y0 = _y0_for_center(rows=8, kh=blk1_kh * S_PIX, gap=GRID_GAP, cols=8,
                             kw=blk1_kw * S_PIX)
    blk1_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=blk1_y0, z0=0.0, n_channels=64, rows=8, cols=8,
        kt=blk1_kt * S_T, kh=blk1_kh * S_PIX, kw=blk1_kw * S_PIX, gap=GRID_GAP,
        palette=PAL_BLOCK1, base_zorder=2.0, edge_width=0.20, hue_jitter=0.10)
    label_tops.append(_stage_label_top(
        ax, blk1_grid, name="ResBlock 1",
        sub=f"{blk1_kt}×{blk1_kh}×{blk1_kw} · 64 ch"))
    stage_records.append({"name": "block1", "grid": blk1_grid})
    x_cursor = _next_x(blk1_grid, g["blk1_to_blk2"])

    # ── ResBlock 2 ─────────────────────────────────────────────────────────
    blk2_y0 = _y0_for_center(rows=16, kh=blk2_kh * S_PIX, gap=GRID_GAP, cols=8,
                             kw=blk2_kw * S_PIX)
    blk2_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=blk2_y0, z0=0.0, n_channels=128, rows=16, cols=8,
        kt=blk2_kt * S_T, kh=blk2_kh * S_PIX, kw=blk2_kw * S_PIX, gap=GRID_GAP,
        palette=PAL_BLOCK2, base_zorder=2.0, edge_width=0.18, hue_jitter=0.10)
    label_tops.append(_stage_label_top(
        ax, blk2_grid, name="ResBlock 2",
        sub=f"{blk2_kt}×{blk2_kh}×{blk2_kw} · 128 ch"))
    stage_records.append({"name": "block2", "grid": blk2_grid})
    x_cursor = _next_x(blk2_grid, g["blk2_to_gru"])

    # ── ConvGRU ────────────────────────────────────────────────────────────
    gru_y0 = _y0_for_center(rows=16, kh=gru_kh * S_PIX, gap=GRID_GAP, cols=8,
                            kw=gru_kw * S_PIX)
    gru_grid = draw_channel_grid(
        ax, x_left=x_cursor, y0=gru_y0, z0=0.0, n_channels=128, rows=16, cols=8,
        kt=gru_kt * S_T, kh=gru_kh * S_PIX, kw=gru_kw * S_PIX, gap=GRID_GAP,
        palette=PAL_GRU, base_zorder=2.0, edge_width=0.18, hue_jitter=0.10)
    g_xmin, g_xmax, g_ymin, g_ymax = gru_grid["bbox2d"]
    gru_loop_base = g_ymax - 0.10
    gru_loop_height = 0.37
    gru_loop_shift = gru_loop_height / 2.0 * 0.75
    draw_recurrent_loop(ax, x0=g_xmin + gru_loop_shift, x1=g_xmax + gru_loop_shift,
                        y_top=gru_loop_base, arc_height=gru_loop_height,
                        color="#7e3f8a", lw=1.4, label=None, gap_frac=0.28)
    gru_loop_top = gru_loop_base + gru_loop_height
    label_tops.append(_stage_label_top(
        ax, gru_grid, name="ConvGRU",
        sub=f"{arch['gru_hidden']} ch · k={arch['gru_kernel']}",
        y_top_override=gru_loop_top))
    stage_records.append({"name": "gru", "grid": gru_grid})
    x_cursor = _next_x(gru_grid, g["gru_to_readout"])

    # ── Readouts (2 on top, ⋮, 1 on bottom) ─────────────────────────────────
    # Enlarge the readout column (feature block + prism) and, downstream, the
    # PSTH predictions by READOUT_SCALE, then drop the column so its "Readouts"
    # title lands even with the ConvGRU title. Both titles clear their reference
    # top by the same LABEL_GAP + SUB_GAP, so aligning the top prism's top with
    # the ConvGRU title's reference top (the recurrent-loop top) aligns them.
    examples = assets.example_neurons
    feat_w, feat_h = 0.62, 0.16
    prism_size = 0.50
    feat_to_prism = 0.14
    base_row_pitch = prism_size + 0.30
    base_ellipsis_gap = 0.60

    readout_scale = READOUT_SCALE
    feat_w *= readout_scale
    feat_h *= readout_scale
    prism_size *= readout_scale
    feat_to_prism *= readout_scale
    row_pitch = base_row_pitch * readout_scale
    ellipsis_gap = base_ellipsis_gap * readout_scale

    span = 2 * row_pitch + ellipsis_gap
    # Column centre so the top prism's top (prism half-height + top-face depth
    # above its centre, itself span/2 above the column centre) meets the top of
    # ResBlock 2's grid.
    top_offset = span / 2 + prism_size / 2 + prism_size * abs(_CAB_DEPTH_VEC[1])
    readout_cy = gru_loop_top - top_offset
    centers = [readout_cy + span / 2, readout_cy + span / 2 - row_pitch,
               readout_cy - span / 2]
    ellipsis_y = 0.5 * (centers[1] + centers[2])
    slot_neurons = [examples[min(i, len(examples) - 1)] for i in (2, 1, 0)]

    prism_x = x_cursor + feat_w + feat_to_prism
    readout_records = []
    for cy, neuron in zip(centers, slot_neurons):
        draw_feature_weight_block(ax, neuron["features"], x0=x_cursor,
                                  y0=cy - feat_h / 2, w=feat_w, h=feat_h,
                                  zorder=4.0)
        sp = draw_spatial_readout_prism(ax, neuron["mean"], neuron["std"],
                                        x0=prism_x, y0=cy - prism_size / 2,
                                        size=prism_size, zorder=4.2)
        readout_records.append({"sp": sp, "center_y": cy})

    ell_x = 0.5 * (x_cursor + prism_x + prism_size)
    for dy in (-0.1485, 0.0, 0.1485):
        ax.add_patch(Circle((ell_x, ellipsis_y + dy), 0.0297, facecolor="#666",
                            edgecolor="none", zorder=4.6))
    # Name the repetition index: the ⋮ is "and so on across recorded units", so
    # one readout head is fit per unit. Grey italic, rotated alongside the dots.
    ax.text(ell_x - 0.308, ellipsis_y, "per unit", ha="center", va="center",
            rotation=90, fontsize=ARCH_SUB_FS + 1.0, color="#555", style="italic",
            zorder=4.6)

    col_x_left = x_cursor
    col_x_right = max(r["sp"]["x_right"] for r in readout_records)
    col_y_bottom = min(r["sp"]["y_bottom"] for r in readout_records)
    col_y_top = max(r["sp"]["y_top"] for r in readout_records)

    sub = "factorized"
    title_x = 0.5 * (col_x_left + col_x_right)
    y_sub = col_y_top + LABEL_GAP
    y_title = y_sub + SUB_GAP
    ax.text(title_x, y_sub, sub, ha="center", va="baseline", fontsize=ARCH_SUB_FS,
            color="#555", style="italic")
    ax.text(title_x, y_title, "Readouts", ha="center", va="baseline",
            fontsize=ARCH_NAME_FS, color=TEXT_COLOR, fontweight="bold")
    label_tops.append(y_title)
    stage_records.append({"name": "readout", "ro": {
        "x_left": col_x_left, "x_right": col_x_right,
        "y_bottom": col_y_bottom, "y_top": col_y_top}})

    # ── Flow arrows ─────────────────────────────────────────────────────────
    arrows = _connect_stages(ax, stage_records[:-1])
    gru_anchor = _stage_right_anchor(stage_records[-2])
    ax.annotate("", xy=(col_x_left - 0.04, gru_anchor[1]),
                xytext=(gru_anchor[0] + 0.04, gru_anchor[1]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"),
                zorder=4.8)

    # Residual skips + downsample badge.
    a_in1 = arrows["stem→block1"]
    a_out1 = arrows["block1→block2"]
    x_fork1 = _arrow_frac(a_in1, 0.50)[0]
    x_plus1 = _arrow_frac(a_out1, 1.0 / 5.0)[0]
    draw_arrow_skip(ax, x_fork1, x_plus1, ARCH_CENTER_Y, depth=SKIP_DEPTH,
                    corner_r=SKIP_CORNER_R, color=SKIP_COLOR, lw=SKIP_LW,
                    zorder=4.7, y_end=ARCH_CENTER_Y - OP_MARKER_RADIUS)
    draw_op_marker(ax, x_plus1, ARCH_CENTER_Y, color=SKIP_COLOR,
                   radius=OP_MARKER_RADIUS, lw=0.9, zorder=12.0)
    mx, my = _arrow_frac(a_out1, 0.50)
    draw_pool_glyph(ax, mx, my)

    a_out2 = arrows["block2→gru"]
    x_fork2 = _arrow_frac(a_out1, 0.75)[0]
    x_plus2 = _arrow_frac(a_out2, 1.0 / 3.0)[0]
    draw_arrow_skip(ax, x_fork2, x_plus2, ARCH_CENTER_Y, depth=SKIP_DEPTH,
                    corner_r=SKIP_CORNER_R, color=SKIP_COLOR, lw=SKIP_LW,
                    zorder=4.7, y_end=ARCH_CENTER_Y - OP_MARKER_RADIUS)
    draw_op_marker(ax, x_plus2, ARCH_CENTER_Y, color=SKIP_COLOR,
                   radius=OP_MARKER_RADIUS, lw=0.9, zorder=12.0)

    # Concat marker on the blk2→gru arrow (behavior injected here). The box +
    # vertical stub are intentionally NOT drawn — the trace bridge supplies it.
    x_concat, y_concat = _arrow_frac(a_out2, 2.0 / 3.0)
    draw_op_marker(ax, x_concat, y_concat, color="#222", radius=OP_MARKER_RADIUS,
                   lw=1.0, zorder=12.5, symbol="||")

    # ── bbox ────────────────────────────────────────────────────────────────
    all_xs, all_ys = [], []
    for s in stage_records:
        if "grid" in s:
            xmin, xmax, ymin, ymax = s["grid"]["bbox2d"]
            all_xs.extend([xmin, xmax])
            all_ys.extend([ymin, ymax])
        elif "ro" in s:
            all_xs.extend([s["ro"]["x_left"], s["ro"]["x_right"]])
            all_ys.extend([s["ro"]["y_bottom"], s["ro"]["y_top"]])
    all_ys.extend([col_y_bottom, y_title + 0.35, max(label_tops) + 0.35,
                   ARCH_CENTER_Y - SKIP_DEPTH - 0.20])

    return {
        "x_left": float(min(all_xs)),
        "x_right": float(max(all_xs)),
        "y_bottom": float(min(all_ys)),
        "y_top": float(max(all_ys)),
        "frontend_left_xy": _stage_left_anchor(stage_records[0]),
        "frontend_right_x": float(stage_records[0]["grid"]["bbox2d"][1]),
        "concat_xy": (float(x_concat), float(y_concat)),
        "readout_rows": [r["center_y"] for r in readout_records],
        "readout_x_right": float(col_x_right),
        "readout_ellipsis_y": float(ellipsis_y),
        "readout_scale": float(readout_scale),
    }


# ════════════════════════════════════════════════════════════════════════════
# BEHAVIOR BRIDGE + READOUT PSTHS
# ════════════════════════════════════════════════════════════════════════════
BEH_TRACE_H = 1.55          # total height of the two stacked trace panels
BEH_EYE_COLOR_X = "#1f6feb"
BEH_EYE_COLOR_Y = "#d68900"
BEH_SPEED_COLOR = "#2a9d8f"  # teal — kept distinct from the stabilized purple
                            # (STABILIZED_COLOR) so purple reads only as "stabilized"

# Stabilized (reafferent-ablated) retinal input, colour-matched to figure3's
# "Extraretinal only (stabilized)" condition so the schematic's second cube and
# the panel C/D/E stabilized violins read as the same manipulation.
STABILIZED_COLOR = "#9467bd"


def _mini_trace(ax, t, y_arr, x, y, w, h, *, color, lw=0.9, y2=None,
                color2=None, ymin=None, ymax=None):
    """Plot 1–2 line traces inside a (x, y, w, h) rect (no axes box)."""
    t = np.asarray(t, float)
    t0, t1 = float(t.min()), float(t.max())
    if t1 == t0:
        t1 = t0 + 1.0
    stack = [np.asarray(y_arr, float)]
    if y2 is not None:
        stack.append(np.asarray(y2, float))
    lo = ymin if ymin is not None else min(float(np.nanmin(s)) for s in stack)
    hi = ymax if ymax is not None else max(float(np.nanmax(s)) for s in stack)
    if hi == lo:
        hi = lo + 1.0

    def _xy(arr):
        xs = x + (t - t0) / (t1 - t0) * w
        ys = y + (arr - lo) / (hi - lo) * h * 0.88 + 0.06 * h
        return xs, ys

    xs, ys = _xy(stack[0])
    ax.plot(xs, ys, color=color, lw=lw, zorder=6, solid_capstyle="round")
    if y2 is not None:
        xs2, ys2 = _xy(stack[1])
        ax.plot(xs2, ys2, color=color2, lw=lw, zorder=6, solid_capstyle="round")
    return lo, hi


def _draw_behavior_traces(ax, assets, x0, y_top, w, *, labeled=True,
                          scale_bar=True, box=True, trace_h=BEH_TRACE_H):
    """Draw the eye-position + velocity covariate panels in the rect whose top
    edge is `y_top`, left edge `x0`, width `w`.

    `labeled` adds the left-side "eye position / eye velocity" text and the x/y
    colour key; `scale_bar` adds a 100 ms bar; `box` wraps the traces (and their
    labels) in an unfilled rounded rectangle so the extraretinal covariate reads
    as a discrete model input — like the retinal-input cube — and returns the
    box's right-edge output anchor. `trace_h` sets the stacked panels' total
    height (a smaller value gives the bare panel-A provenance echo). Returns a
    bbox + optional output anchor.
    """
    t = np.asarray(assets.behavior_t, float)
    eye = np.asarray(assets.behavior_eyepos, float)
    speed = np.asarray(assets.behavior_speed, float)

    panel_h = 0.5 * (trace_h - 0.06)
    y_speed = y_top - trace_h
    y_eye = y_speed + panel_h + 0.06

    # Eye position (x & y, shared scale) on top; velocity below.
    _mini_trace(ax, t, eye[:, 0], x0, y_eye, w, panel_h,
                color=BEH_EYE_COLOR_X, y2=eye[:, 1], color2=BEH_EYE_COLOR_Y,
                ymin=float(eye.min()), ymax=float(eye.max()))
    _mini_trace(ax, t, speed, x0, y_speed, w, panel_h, color=BEH_SPEED_COLOR)

    x_left = x0
    if labeled:
        ax.text(x0 - 0.12, y_eye + panel_h / 2, "eye\nposition", ha="right",
                va="center", fontsize=6.0, color="#444", linespacing=1.0)
        ax.text(x0 - 0.12, y_speed + panel_h / 2, "eye\nvelocity", ha="right",
                va="center", fontsize=6.0, color="#444", linespacing=1.0)
        # Tiny x/y color key on the eye-position panel.
        ax.text(x0 + 0.04, y_eye + panel_h - 0.02, "x", ha="left", va="top",
                fontsize=5.5, color=BEH_EYE_COLOR_X, fontweight="bold")
        ax.text(x0 + 0.20, y_eye + panel_h - 0.02, "y", ha="left", va="top",
                fontsize=5.5, color=BEH_EYE_COLOR_Y, fontweight="bold")
        x_left = x0 - 0.55

    bar_y = y_speed - 0.10
    if scale_bar:
        span_s = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
        bar_len = min(0.1 / span_s, 0.5) * w
        bar_x1 = x0 + w
        ax.add_line(Line2D([bar_x1 - bar_len, bar_x1], [bar_y, bar_y],
                           color="#222", lw=1.5, solid_capstyle="butt",
                           zorder=6))
        ax.text(bar_x1 - bar_len / 2, bar_y - 0.05, "100 ms", ha="center",
                va="top", fontsize=6.0, color="#222")

    result = {
        "x_left": x_left,
        "x_right": x0 + w,
        "y_bottom": bar_y - (0.18 if scale_bar else 0.05),
        "y_top": y_top,
    }

    # ── Enclosing module box: an unfilled rounded rectangle around the two
    #    trace panels + their labels, granting the extraretinal covariate the
    #    same object-status as the retinal-input cube. Signal leaves the box's
    #    right-edge midpoint (no separate collecting bracket). ────────────────
    if box:
        box_left = x_left - 0.40
        box_right = x0 + w + 0.14
        box_top = y_top + 0.08
        box_bottom = (bar_y - 0.26) if scale_bar else (y_speed - 0.12)
        ax.add_patch(FancyBboxPatch(
            (box_left, box_bottom), box_right - box_left, box_top - box_bottom,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor="none", edgecolor="#555", linewidth=1.0, zorder=5))
        out_mid_y = 0.5 * (y_speed + y_eye + panel_h)
        result["x_left"] = box_left
        result["x_right"] = box_right
        result["y_top"] = box_top
        result["y_bottom"] = box_bottom
        result["out_x"] = box_right
        result["out_mid_y"] = out_mid_y

    return result


def _route_behavior_to_concat(ax, out_x, out_mid_y, concat_xy):
    """Route the extraretinal signal from the module box rightward, through an
    ablation SWITCH, then up into the concat marker.

    The switch is a single-pole knife switch drawn OPEN (thrown down): the black
    conductor is broken at two contacts, a faint blue segment bridges them to
    show the CLOSED "Full" path, and a red lever droops from the left pivot to
    show the depicted OPEN "Ablated" state. A small double-headed arc marks that
    the switch flips between the two — the model is run either way. The two
    states are colour-matched to the Full/Ablated conditions quantified in
    panels C/D/E and labelled Full (above) / Ablated (below), so the pathway
    reads as optional and compared rather than permanently deleted."""
    cx, cy = concat_xy
    mid = out_mid_y
    elbow_x = cx

    # Switch geometry: two contacts on the wire separated by a gap, centred at
    # 55% along the horizontal run (before the elbow that turns up to concat).
    gx = out_x + 0.55 * (elbow_x - out_x)
    half = 0.42
    xL, xR = gx - half, gx + half

    # ── Conductor: solid black up to each contact, broken across the switch ──
    ax.add_line(Line2D([out_x, xL], [mid, mid], color="#333", lw=1.1, zorder=6))
    ax.add_line(Line2D([xR, elbow_x], [mid, mid], color="#333", lw=1.1, zorder=6))
    ax.add_patch(FancyArrowPatch((elbow_x, mid), (cx, cy - OP_MARKER_RADIUS),
                                 arrowstyle="-|>", lw=1.1, color="#333",
                                 mutation_scale=10, zorder=6,
                                 shrinkA=0, shrinkB=0))

    # ── CLOSED path (Full): solid blue bridge = the alternative (not-thrown)
    #    position that reconnects the two contacts. ──────────────────────────
    ax.add_line(Line2D([xL, xR], [mid, mid], color=COND_FULL_COLOR, lw=2.0,
                       solid_capstyle="round", zorder=6.2))

    # ── OPEN lever (Ablated): red knife thrown down from the right pivot,
    #    extended as a leader to the vertical middle of the right-justified
    #    two-line label sitting to the lower-left. ────────────────────────────
    txt_r = gx - 0.28          # right edge of the right-justified label block
    txt_cy = mid - 0.70        # vertical centre of the two-line label
    tip = (txt_r + 0.08, txt_cy)
    ax.add_line(Line2D([xR, tip[0]], [mid, tip[1]], color=COND_ABLATED_COLOR,
                       lw=2.2, solid_capstyle="round", zorder=6.4))

    # Contacts: two equal filled circles (the lever hinges on the right one).
    for tx in (xL, xR):
        ax.add_patch(Circle((tx, mid), 0.06, facecolor="#333",
                            edgecolor="none", zorder=6.5))

    # ── Labels: Full above (blue, on the bridge), Ablated to the lower-left
    #    (red, right-justified at the end of the extended lever). ─────────────
    ax.text(gx, mid + 0.14, "Full", ha="center", va="bottom",
            fontsize=8.5, color=COND_FULL_COLOR, fontweight="bold", zorder=7)
    ax.text(txt_r, txt_cy + 0.13, "Ablated", ha="right", va="center",
            fontsize=9.0, color=COND_ABLATED_COLOR, fontweight="bold", zorder=7)
    ax.text(txt_r, txt_cy - 0.13, "behavioral input → 0", ha="right", va="center",
            fontsize=6.8, color=COND_ABLATED_COLOR, style="italic", zorder=7)


PSTH_GAP = 0.62
PSTH_W = 1.05
PSTH_H = 0.52
PSTH_OBS_COLOR = "#9a9a9a"   # observed PSTH bars
PSTH_PRED_COLOR = "#d62728"  # prediction line (red)


def _draw_readout_psths(ax, assets, arch):
    """Draw an observed-vs-predicted PSTH to the right of each readout row,
    using the best-CCnorm units. Returns the panels' bounding box."""
    psths = assets.psth_neurons or []
    rows = arch["readout_rows"]
    # Match the readout column's scaling so the predictions fill the same full
    # row height (rows are already spread by the scaled readout layout).
    scale = arch.get("readout_scale", 1.0)
    psth_w = PSTH_W * scale
    psth_h = PSTH_H * scale
    x0 = arch["readout_x_right"] + PSTH_GAP
    xr = x0 + psth_w

    # Small header over the PSTH column.
    ax.text(x0 + psth_w / 2, max(rows) + psth_h / 2 + 0.14,
            "Predictions", ha="center", va="bottom", fontsize=ARCH_NAME_FS,
            color=TEXT_COLOR, fontweight="bold")

    n = min(len(rows), len(psths))
    for k in range(n):
        cy = rows[k]
        p = psths[k]
        y0 = cy - psth_h / 2
        is_bottom = (k == n - 1)
        draw_neuron_trace_panel(
            ax, p["t"], p["robs_rate"], p["rhat_rate"], x0, y0, psth_w, psth_h,
            obs_color=PSTH_OBS_COLOR, pred_color=PSTH_PRED_COLOR,
            obs_lw=0.8, pred_lw=1.2, zorder=5.0,
            label=None,
            show_scale=is_bottom, scale_ms=100,
            scale_sp_s=None)

    # Matching ⋮ between the second and last PSTH (mirrors the readout ⋮).
    if len(rows) >= 3:
        ell_y = arch["readout_ellipsis_y"]
        ell_x = x0 + psth_w / 2
        for dy in (-0.1485, 0.0, 0.1485):
            ax.add_patch(Circle((ell_x, ell_y + dy), 0.0297, facecolor="#666",
                                edgecolor="none", zorder=5.2))

    return {
        "x_left": x0,
        "x_right": xr + 0.05,
        "y_bottom": min(rows) - psth_h / 2 - 0.20,
        "y_top": max(rows) + psth_h / 2 + 0.45,
    }


# ════════════════════════════════════════════════════════════════════════════
# COMPOSITION
# ════════════════════════════════════════════════════════════════════════════
BRIDGE_GAP = 0.7            # world gap: input-cube right → frontend left
PANEL_HEIGHT_IN = 4.0

# ── Two-row layout ──────────────────────────────────────────────────────────
# Bottom row = the model: its own model-input cube + behavior covariates at the
# far left, flowing right through the architecture. Top row = stimulus
# provenance, stacked above with a vertical gap.
BOT_CUBE_CX = 1.35          # bottom-row input-cube front-centre x
BOT_BEH_GAP = 0.95          # gap: bottom-row cube bottom → behavior traces
                            # (lowers the route so it clears the prism bottoms)
ROW_GAP = 0.55              # vertical gap: bottom-row top → top-row lowest ink
TOP_ROW_BELOW = 2.75        # how far the top row extends below its centre line
                            # (now includes the bare behavior echo under the cube)

# Panel-B retinal-input stack: the moving cube sits at ARCH_CENTER_Y; the
# stabilized (reafferent-ablated) cube is dropped below it, and the behavior
# box is pushed beneath the stabilized cube.
STAB_CUBE_DY = 2.15         # vertical drop: moving-cube centre → stabilized-cube centre
STAB_BEH_GAP = 1.15         # gap: stabilized-cube bottom (incl. label) → behavior box top


def _draw_all(ax, assets):
    """Draw the full integrated panel into one axes and set data limits.

    Two rows share one world frame: the model row is drawn first (its own
    model-input cube + behavior at the far left, then the architecture), then
    the stimulus-provenance row is placed above it.
    """
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Bottom row: model input (cube + behavior) → architecture → PSTHs ────
    # The moving (reafferent) cube and the stabilized cube form one vertical
    # group centred on the frontend: the moving cube sits half a step above
    # ARCH_CENTER_Y, the stabilized cube half a step below.
    moving_cy = ARCH_CENTER_Y + STAB_CUBE_DY / 2.0
    stab_cy = ARCH_CENTER_Y - STAB_CUBE_DY / 2.0

    cube_in = _draw_cube_block(ax, assets, (BOT_CUBE_CX, moving_cy),
                               draw_header=False, draw_dims=False,
                               draw_time=True, time_label="time")
    # Title the retinal pathway (mirrors the "Extraretinal behavior" label). The
    # italic sub-line names the reafferent motion the stabilized cube removes.
    cube_cx = 0.5 * (cube_in["x_left"] + cube_in["x_right"])
    ax.text(cube_cx, cube_in["top_y"] + 0.30, "Retinal input", ha="center",
            va="bottom", fontsize=STIM_HEADER_FS, color=TEXT_COLOR,
            fontweight="bold")
    ax.text(cube_cx, cube_in["top_y"] + 0.10, "moves with eye (reafference)",
            ha="center", va="bottom", fontsize=STIM_SUB_FS, color="#555",
            style="italic")

    # Reafferent ablation: the "stabilized" copy of the cube beneath the moving
    # one — the same RSVP frames with the retinal ROI frozen at the trial medoid
    # gaze, so image flashes still update mid-window but the gaze-induced motion
    # is gone (its top/side faces read as frozen bands stepping at each flash,
    # versus the moving cube's continuous drift). Colour-matched to the panel
    # C/D/E "stabilized" condition.
    stab_cube = getattr(assets, "stab_lag_cube", None)
    if stab_cube is None:   # cache predates the field; frozen-frame fallback
        stab_cube = np.repeat(assets.lag_cube[-1][None],
                              assets.lag_cube.shape[0], axis=0)
    stab_block = _draw_cube_block(
        ax, assets, (BOT_CUBE_CX, stab_cy), draw_header=False, draw_dims=False,
        draw_time=False, cube_override=stab_cube, outline=STABILIZED_COLOR)
    stab_cx = 0.5 * (stab_block["x_left"] + stab_block["x_right"])
    ax.text(stab_cx, stab_block["bottom_y"] - 0.04, "Stabilized",
            ha="center", va="top", fontsize=STIM_HEADER_FS - 0.5,
            color=STABILIZED_COLOR, fontweight="bold")
    ax.text(stab_cx, stab_block["bottom_y"] - 0.26, "reafference removed",
            ha="center", va="top", fontsize=STIM_SUB_FS,
            color=STABILIZED_COLOR, style="italic")

    arch = _draw_architecture(ax, assets, x_start=cube_in["x_right"] + BRIDGE_GAP)
    fe_xy = arch["frontend_left_xy"]

    # Both cubes feed the frontend, converging on its input: the moving cube via
    # a solid arrow, the stabilized cube via a dashed purple swap-in arrow
    # (mirrors the behavior box's Full/Ablated grammar — either can drive the
    # retinal pathway).
    ax.annotate("", xy=(fe_xy[0] - 0.06, fe_xy[1]),
                xytext=(cube_in["x_right"] + 0.1, cube_in["mid_y"]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"),
                zorder=4.8)
    ax.annotate("", xy=(fe_xy[0] - 0.10, fe_xy[1] - 0.05),
                xytext=(stab_block["x_right"] + 0.06, stab_block["mid_y"]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=STABILIZED_COLOR,
                                linestyle=(0, (4, 3))),
                zorder=4.75)

    # Behavioral path: labelled covariates beneath the stabilized cube, boxed as
    # a discrete model input and routed across to the concat marker.
    beh_x0 = cube_in["x_left"]
    beh_w = arch["frontend_right_x"] - beh_x0
    beh_top = stab_block["bottom_y"] - STAB_BEH_GAP
    beh = _draw_behavior_traces(ax, assets, beh_x0, beh_top, beh_w,
                                labeled=False, scale_bar=True, box=True)
    # Bold title above the box (mirrors the "Retinal input" title over the cube).
    ax.text(beh_x0 + beh_w / 2, beh["y_top"] + 0.10, "Behavioral input",
            ha="center", va="bottom", fontsize=STIM_HEADER_FS, color=TEXT_COLOR,
            fontweight="bold")
    _route_behavior_to_concat(ax, beh["out_x"], beh["out_mid_y"],
                              arch["concat_xy"])

    psth = _draw_readout_psths(ax, assets, arch)

    bot_xs = [cube_in["x_left"], cube_in["x_right"], arch["x_left"],
              arch["x_right"], beh["x_left"], beh["x_right"],
              psth["x_left"], psth["x_right"],
              stab_block["x_left"], stab_block["x_right"]]
    bot_ys = [cube_in["top_y"], arch["y_bottom"], arch["y_top"],
              beh["y_bottom"], psth["y_bottom"], psth["y_top"],
              stab_block["bottom_y"] - 0.45]
    bot_y_top = max(bot_ys)

    # ── Top row: stimulus provenance, placed above the model row ────────────
    row_cy = bot_y_top + ROW_GAP + TOP_ROW_BELOW
    top = _draw_top_row(ax, assets, row_cy)

    # ── Panel letters A (stimuli) / B (twin) ────────────────────────────────
    letter_x = min(min(bot_xs), top["x_left"])
    ax.text(letter_x, top["y_top"] + 0.35, "A", ha="left", va="top",
            fontsize=10, fontweight="bold", color="#202124")
    ax.text(letter_x, bot_y_top + 0.35, "B", ha="left", va="top",
            fontsize=10, fontweight="bold", color="#202124")

    xs = bot_xs + [top["x_left"], top["x_right"]]
    ys = bot_ys + [top["y_bottom"], top["y_top"], top["y_top"] + 0.35]
    pad_x, pad_y = 0.3, 0.2
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")


def _data_aspect(ax):
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    return (x_hi - x_lo) / (y_hi - y_lo)


def _fit_one_axes_in_rect(fig, ax, rect, aspect):
    """Center a single equal-aspect axes inside `rect` (figure fraction)."""
    fw, fh = fig.get_size_inches()
    slot_w = rect.width * fw
    slot_h = rect.height * fh
    h_in = slot_h
    if h_in * aspect > slot_w:
        h_in = slot_w / aspect
    w_in = h_in * aspect
    x0 = rect.x0 + (rect.width - w_in / fw) / 2.0
    y0 = rect.y0 + (rect.height - h_in / fh) / 2.0
    ax.set_position([x0, y0, w_in / fw, h_in / fh])
    ax.set_in_layout(False)


def render_panel_a(assets=None, *, recompute=False, height_in=PANEL_HEIGHT_IN):
    """Build a standalone panel-A figure sized to the drawn content aspect."""
    if assets is None:
        assets = load_panel_a_assets(recompute=recompute)
    fig = plt.figure(figsize=(3 * height_in, height_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    _draw_all(ax, assets)
    aspect = _data_aspect(ax)
    fig.set_size_inches(height_in * aspect, height_in)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    return fig, ax


def plot_panel_a(*, ax=None, subplotspec=None, fig=None, assets=None,
                 recompute=False):
    """Render panel A into a single axes fitted to an existing figure slot.

    Pass `subplotspec=` (composite GridSpec slot) or `ax=` (the axes is hidden
    and a fitted axes drawn in its rect); with neither, build a standalone
    figure via `render_panel_a`.
    """
    if assets is None:
        assets = load_panel_a_assets(recompute=recompute)

    if subplotspec is not None:
        if fig is None:
            fig = subplotspec.get_gridspec().figure
        rect = subplotspec.get_position(fig)
    elif ax is not None:
        fig = ax.figure
        ss = ax.get_subplotspec()
        rect = ss.get_position(fig) if ss is not None else ax.get_position()
        ax.set_visible(False)
        ax.set_in_layout(False)
    else:
        return render_panel_a(assets)

    draw_ax = fig.add_axes([rect.x0, rect.y0, rect.width, rect.height])
    _draw_all(draw_ax, assets)
    _fit_one_axes_in_rect(fig, draw_ax, rect, _data_aspect(draw_ax))
    return fig, draw_ax


def main(recompute=False):
    configure_matplotlib()
    assets = load_panel_a_assets(recompute=recompute)
    fig, _ = render_panel_a(assets)
    out_png = FIG_DIR / "panel_a_schematic.png"
    out_pdf = FIG_DIR / "panel_a_schematic.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render figure 3 panel A.")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    main(recompute=args.recompute)
