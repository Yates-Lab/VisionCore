"""Drawing primitives for figure 4 panel A.

All primitives operate in matplotlib data coordinates. "3D" looks come from
two techniques:
  * Perspective screens — PIL.PERSPECTIVE warps the source image into a
    trapezoid (right side pinched, simulating receding depth) and matplotlib
    `imshow`s the warped raster.
  * 3D rectangular prisms — three polygons (front + top + right) drawn with
    appropriate shading or with image textures projected onto each face.
"""
from __future__ import annotations

import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
from matplotlib.transforms import Affine2D


CYAN = "#00b8d4"
ARROW_COLOR = "#333333"
TEXT_COLOR = "#222222"


# ──────────────────────────────────────────────────────────────────────────
# Cabinet projection (architecture half)
# +x right, +y up, +z INTO the page → projects to (Δx, Δy) per unit z.
# The stimulus half (_fig4a_stimulus.py) keeps its own CABINET_ALPHA /
# CABINET_DEPTH so the two halves can be tuned independently. Classical
# cabinet projection: depth is foreshortened to half so cube-shaped kernels
# read as square-fronted blocks (slant edge ≈ ½ vertical edge) instead of
# slanted rhombi. α=45° is the standard cabinet angle.
#
# Depth projects UP-AND-RIGHT so each kernel's visible faces are
# front + top + RIGHT — matching the stimulus lag cube's apparent
# orientation and the usual isometric look (block "coming out" toward
# the lower-left rather than receding into the page).
# ──────────────────────────────────────────────────────────────────────────
CAB_ALPHA = np.deg2rad(45.0)
CAB_DEPTH = 0.5
CAB_DEPTH_VEC = np.array([
    +np.cos(CAB_ALPHA) * CAB_DEPTH,
    +np.sin(CAB_ALPHA) * CAB_DEPTH,
])


def cab_project(p3):
    """Project (..., 3) world points to (..., 2) display coords."""
    p3 = np.asarray(p3, dtype=float)
    return p3[..., :2] + p3[..., 2:3] * CAB_DEPTH_VEC


def axis_aligned_box_corners(center, size):
    """8 corners of an axis-aligned box centered at `center=(cx,cy,cz)`
    with extents `size=(sx,sy,sz)`. Ordering::

        0 front-LL  1 front-LR  2 front-UR  3 front-UL    (z = cz - sz/2)
        4 back-LL   5 back-LR   6 back-UR   7 back-UL     (z = cz + sz/2)

    "front" = nearer to viewer (smaller z, since +z is into the page).
    """
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=float)


def draw_axis_aligned_prism(ax, center, size, *,
                            front_color="#cfe2f3", side_color="#7fa4c4",
                            top_color="#4d7396", edge_color="#1b3a5b",
                            edge_width=0.4, zorder=2, alpha=1.0):
    """Draw an axis-aligned rectangular prism in cabinet projection.

    Visible faces: front (z=cz-sz/2), top (y=cy+sy/2), and RIGHT
    (x=cx+sx/2). With +z going into the page (up-and-right), the right
    and top faces face the viewer; the left and bottom faces are hidden.
    """
    c = axis_aligned_box_corners(center, size)
    # corner indices: 0..3 front (LL,LR,UR,UL); 4..7 back
    front = cab_project(c[[0, 1, 2, 3]])
    top   = cab_project(c[[3, 2, 6, 7]])    # front-UL, front-UR, back-UR, back-UL
    right = cab_project(c[[1, 2, 6, 5]])    # front-LR, front-UR, back-UR, back-LR

    ax.add_patch(Polygon(top, closed=True, facecolor=top_color,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.1, alpha=alpha))
    ax.add_patch(Polygon(right, closed=True, facecolor=side_color,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.2, alpha=alpha))
    ax.add_patch(Polygon(front, closed=True, facecolor=front_color,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.3, alpha=alpha))
    return front, top, right


# ──────────────────────────────────────────────────────────────────────────
# Perspective warp utilities
# ──────────────────────────────────────────────────────────────────────────
def _perspective_coeffs(src_corners, dst_corners):
    """Solve for PIL.Image.PERSPECTIVE coefficients.

    PIL maps output(x', y') back to input(x, y) as:
        x = (a x' + b y' + c) / (g x' + h y' + 1)
        y = (d x' + e y' + f) / (g x' + h y' + 1)

    `src_corners`: 4×2 source-image pixel coords (where to sample FROM)
    `dst_corners`: 4×2 output-image pixel coords (target locations in the
                   *output canvas*; we use the bounding box of dst as the
                   canvas size).
    """
    M = []
    for (sx, sy), (dx, dy) in zip(src_corners, dst_corners):
        M.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        M.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(M, dtype=np.float64)
    B = np.array(src_corners, dtype=np.float64).reshape(8)
    coeffs, *_ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs


def _perspective_screen_quad(x, y, w, h, pinch=0.18, lift=0.0):
    """Return target corners (LL, LR, UR, UL) of a trapezoid that looks like
    a screen receding to the RIGHT. `pinch` is the vertical inset on the
    right edge as a fraction of `h`; larger ⇒ deeper perspective."""
    return np.array([
        [x,         y],                     # lower-left (full height)
        [x + w,     y + h * pinch],         # lower-right (lifted)
        [x + w,     y + h - h * pinch],     # upper-right (lowered)
        [x,         y + h],                 # upper-left (full height)
    ])


def draw_perspective_screen(ax, image, x, y, w, h, *, pinch=0.20,
                            roi=None, screen_shape=None,
                            edge_color="#222", edge_width=0.8,
                            roi_color=CYAN, roi_width=2.0, zorder=2,
                            auto_contrast=True, label=None,
                            label_offset=0.18, label_fontsize=8.5):
    """Render a screen tilted into the page on its right side.

    The image is pre-warped to the trapezoid via PIL and placed via imshow
    at the bounding box of that trapezoid. The trapezoid edges are drawn on
    top. If `roi` is provided (screen-pixel coords), it is overlaid as a
    cyan rectangle that follows the same perspective.

    Returns `(target_quad, src_corners)` so callers can attach further
    overlays in screen-pixel space.
    """
    target_quad = _perspective_screen_quad(x, y, w, h, pinch=pinch)
    Himg, Wimg = image.shape[:2]

    # Source corners in image-pixel coords (origin upper-left for PIL).
    # Order MUST match target_quad: LL, LR, UR, UL.
    src_corners = np.array([
        [0,     Himg],
        [Wimg,  Himg],
        [Wimg,  0],
        [0,     0],
    ], dtype=np.float64)

    # Place the trapezoid into a square output canvas at high resolution.
    out_res = 600
    bx0, by0 = target_quad[:, 0].min(), target_quad[:, 1].min()
    bx1, by1 = target_quad[:, 0].max(), target_quad[:, 1].max()
    bw, bh = bx1 - bx0, by1 - by0
    sx = out_res / bw
    sy = out_res / bh
    # Convert target_quad → canvas pixel coords (PIL: y goes DOWN)
    dst_px = np.column_stack([
        (target_quad[:, 0] - bx0) * sx,
        (by1 - target_quad[:, 1]) * sy,  # flip y so up is up in display
    ])

    coeffs = _perspective_coeffs(src_corners, dst_px)
    pil_in = PILImage.fromarray(image.astype(np.uint8))
    if auto_contrast:
        arr = np.asarray(pil_in, dtype=np.float32)
        vmin, vmax = np.percentile(arr, [1, 99])
        if vmax > vmin:
            arr = np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255)
            pil_in = PILImage.fromarray(arr.astype(np.uint8))

    pil_out = pil_in.transform(
        (out_res, out_res), PILImage.PERSPECTIVE, coeffs,
        resample=PILImage.BILINEAR,
    )
    # Mask: paint a polygon-shaped alpha channel so the region outside the
    # trapezoid is transparent.
    mask = PILImage.new("L", (out_res, out_res), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).polygon([tuple(p) for p in dst_px], fill=255)
    rgba = np.dstack([
        np.array(pil_out),
        np.array(pil_out),
        np.array(pil_out),
        np.array(mask),
    ])
    ax.imshow(rgba, extent=[bx0, bx1, by0, by1], origin="upper",
              zorder=zorder, interpolation="bilinear")

    # Edge polygon
    ax.add_patch(Polygon(target_quad, closed=True, fill=False,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.1))

    # ROI rectangle in screen-pixel coords → mapped via the same warp.
    roi_quad = None
    if roi is not None and screen_shape is not None:
        H, W = screen_shape
        r0, r1 = roi[0]
        c0, c1 = roi[1]
        # ROI corners in source-image coords (PIL convention: origin upper-left).
        # Note image_shape = screen_shape for our renders.
        roi_src = np.array([
            [c0, r1],  # LL
            [c1, r1],  # LR
            [c1, r0],  # UR
            [c0, r0],  # UL
        ], dtype=np.float64)
        # Map src → dst by inverting the homography. The PIL coeffs above go
        # dst → src, so we solve src → dst with the reverse fit.
        rev_coeffs = _perspective_coeffs(dst_px, src_corners)
        roi_dst_px = _apply_perspective(rev_coeffs, roi_src)
        # Convert canvas pixel coords → data coords
        roi_quad = np.column_stack([
            roi_dst_px[:, 0] / sx + bx0,
            by1 - roi_dst_px[:, 1] / sy,
        ])
        ax.add_patch(Polygon(roi_quad, closed=True, fill=False,
                             edgecolor=roi_color, linewidth=roi_width,
                             zorder=zorder + 0.2))

    if label is not None:
        cx = target_quad[:, 0].mean()
        top_y = target_quad[2:, 1].max()
        ax.text(cx, top_y + label_offset, label, ha="center", va="bottom",
                fontsize=label_fontsize, color=TEXT_COLOR)

    return target_quad, roi_quad


def _apply_perspective(coeffs, pts):
    """Apply PIL-style perspective coefficients to Nx2 points."""
    a, b, c, d, e, f, g, h = coeffs
    x, y = pts[:, 0], pts[:, 1]
    denom = g * x + h * y + 1
    return np.column_stack([(a * x + b * y + c) / denom,
                            (d * x + e * y + f) / denom])


# ──────────────────────────────────────────────────────────────────────────
# 3D volume box (lag-cube)
# ──────────────────────────────────────────────────────────────────────────
def draw_volume_box(ax, cube, x, y, w, h, *, depth_x=1.6, depth_y=0.9,
                    outline=CYAN, edge_width=1.2, zorder=4,
                    label_dims=("space", "space", "time"),
                    show_dim_labels=True):
    """Draw a 3D rectangular prism whose three visible faces are textured
    from real cube data.

    The cube has shape (n_lags, H, W); the convention:
      front face  = cube[0]                  (the "latest" frame, t = 0)
      right face  = cube[:, :, -1].T (time × y) (time runs along depth axis,
                                                 pointing INTO the page)
      top face    = cube[:, 0, :]   (time × x)

    The prism is oriented so:
      - front face is at (x..x+w, y..y+h) — small, "shortest lag" on the right
      - depth points up-and-LEFT, so longest lag is at the back-left.

    Actually for left-to-right "longest → shortest lag" we draw the prism
    so the FRONT face is on the RIGHT (shortest lag = front), and the depth
    extends to the LEFT (longest lag = back).
    """
    n_lags, Hc, Wc = cube.shape

    # Normalize cube intensities for display.
    vmin, vmax = np.percentile(cube, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0

    def _norm(arr):
        a = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        return (a * 255).astype(np.uint8)

    # Front face = "most recent" frame (shortest lag).
    front_img = _norm(cube[0])

    # Side face: time × y (rows = lags from BACK to FRONT, cols = y). We want
    # the longest-lag time slice on the BACK (left). So order rows so cube[-1]
    # appears at the back-left and cube[0] at the front-right.
    # The side face is along the *left* of the prism (since depth goes left).
    # In display coordinates, time runs from x_back (left) to x_front (right).
    side_img = _norm(cube[::-1, :, Wc // 2])      # (n_lags, H) — rows=time
    # We want the side face to read with time on horizontal axis: transpose.
    side_img = side_img.T  # (H, n_lags) — rows=y, cols=time

    # Top face: time × x along the top of the prism.
    top_img = _norm(cube[::-1, Hc // 2, :])   # (n_lags, W)
    # Top face should have time on horizontal axis: (W, n_lags)? No — we
    # want columns of the top quad to follow depth. The top face stretches
    # across (front-x → back-x). We'll resample below using a quad warp.

    # ── Vertices ─────────────────────────────────────────────────────────
    # Front face (right side of the prism)
    fL, fR = x, x + w
    fB, fT = y, y + h
    # Depth offset: into the page = up-and-left
    dx = -depth_x
    dy = depth_y
    # Back face corners
    bL, bR = fL + dx, fR + dx
    bB, bT = fB + dy, fT + dy

    front_quad = np.array([[fL, fB], [fR, fB], [fR, fT], [fL, fT]])  # LL, LR, UR, UL
    top_quad   = np.array([[fL, fT], [fR, fT], [bR, bT], [bL, bT]])  # front-left, front-right, back-right, back-left
    left_quad  = np.array([[fL, fB], [fL, fT], [bL, bT], [bL, bB]])  # front-bot, front-top, back-top, back-bot

    # ── Draw faces as textured quads ─────────────────────────────────────
    _draw_textured_quad(ax, front_img, front_quad, zorder=zorder + 0.3)
    _draw_textured_quad(ax, top_img,   top_quad,   zorder=zorder + 0.2,
                        src_corners=np.array([
                            [0, top_img.shape[0]],   # front-left of top  ← cube[0]
                            [0, 0],                  # front-right of top ← cube[0]
                            [top_img.shape[1], 0],   # back-right of top  ← cube[-1]
                            [top_img.shape[1], top_img.shape[0]],
                        ]))
    _draw_textured_quad(ax, side_img,  left_quad,  zorder=zorder + 0.1,
                        src_corners=np.array([
                            [side_img.shape[1], side_img.shape[0]],  # front-bot
                            [side_img.shape[1], 0],                  # front-top
                            [0, 0],                                  # back-top
                            [0, side_img.shape[0]],                  # back-bot
                        ]))

    # Outline edges
    for quad in (front_quad, top_quad, left_quad):
        ax.add_patch(Polygon(quad, closed=True, fill=False,
                             edgecolor=outline, linewidth=edge_width,
                             zorder=zorder + 0.5))

    if show_dim_labels:
        # Time arrow under the front face → annotate
        ax.annotate(
            "", xy=(fR, fB - 0.45),
            xytext=(bL, bB - 0.45 + (fB - bB) * 0),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=ARROW_COLOR),
        )
        ms = n_lags / 120.0 * 1000.0
        ax.text((fL + bL) / 2 + (fR - fL) / 2, fB - 0.65,
                f"time ({n_lags} lags · {ms:.0f} ms)",
                ha="center", va="top",
                fontsize=7, color=TEXT_COLOR, style="italic")
        ax.text(fR + 0.15, (fB + fT) / 2, "ROI\n51×51",
                ha="left", va="center", fontsize=7, color=TEXT_COLOR,
                style="italic")


def _draw_textured_quad(ax, image, dst_quad, *, src_corners=None,
                        zorder=2, alpha=1.0):
    """Warp `image` into the quadrilateral `dst_quad` (4×2 in data coords).

    `dst_quad` ordering: corners corresponding to (image LL, LR, UR, UL).
    If `src_corners` is given (4×2 in image-pixel coords), it overrides the
    default which uses the full image extent.
    """
    Himg, Wimg = image.shape[:2]
    if src_corners is None:
        src_corners = np.array([
            [0, Himg], [Wimg, Himg], [Wimg, 0], [0, 0],
        ], dtype=np.float64)

    # Render the warped quad on a small canvas, then imshow that with extent
    # = bounding box of dst_quad.
    bx0, by0 = dst_quad[:, 0].min(), dst_quad[:, 1].min()
    bx1, by1 = dst_quad[:, 0].max(), dst_quad[:, 1].max()
    out_res = 256
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
    warped = pil.transform((out_res, out_res), PILImage.PERSPECTIVE, coeffs,
                            resample=PILImage.BILINEAR)
    mask = PILImage.new("L", (out_res, out_res), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).polygon([tuple(p) for p in dst_px], fill=255)
    rgba = np.dstack([np.array(warped)] * 3 + [np.array(mask)])
    ax.imshow(rgba, extent=[bx0, bx1, by0, by1], origin="upper",
              zorder=zorder, interpolation="bilinear", alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────
# Architecture rectangular prisms (CNN-paper style)
# ──────────────────────────────────────────────────────────────────────────
def draw_arch_prism(ax, x, y, w, h, *, depth_x=0.55, depth_y=0.30,
                    front_color="#cfe2f3", side_color="#9fbdda",
                    top_color="#7fa4c4", edge_color="#1b3a5b",
                    edge_width=0.7, zorder=2, hatch=None,
                    label=None, sublabel=None, label_above_y=None):
    """Draw a rectangular prism with three visible faces (front, top, right).

    Coordinates of the FRONT face: (x, y) lower-left, width=w, height=h.
    Depth extends UP-AND-RIGHT by (depth_x, depth_y).
    """
    fL, fR = x, x + w
    fB, fT = y, y + h
    bL, bR = fL + depth_x, fR + depth_x
    bB, bT = fB + depth_y, fT + depth_y

    front = np.array([[fL, fB], [fR, fB], [fR, fT], [fL, fT]])
    top   = np.array([[fL, fT], [fR, fT], [bR, bT], [bL, bT]])
    right = np.array([[fR, fB], [bR, bB], [bR, bT], [fR, fT]])

    ax.add_patch(Polygon(top, closed=True, facecolor=top_color,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.1, hatch=hatch))
    ax.add_patch(Polygon(right, closed=True, facecolor=side_color,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.2, hatch=hatch))
    ax.add_patch(Polygon(front, closed=True, facecolor=front_color,
                         edgecolor=edge_color, linewidth=edge_width,
                         zorder=zorder + 0.3, hatch=hatch))

    if label is not None:
        ly = label_above_y if label_above_y is not None else (bT + 0.15)
        cx = (fL + bR) / 2
        ax.text(cx, ly, label, ha="center", va="bottom",
                fontsize=9, color=TEXT_COLOR, fontweight="bold",
                zorder=zorder + 1)
        if sublabel:
            ax.text(cx, ly - 0.28, sublabel, ha="center", va="top",
                    fontsize=6.8, color="#555", style="italic",
                    zorder=zorder + 1, linespacing=1.1)
    return front, top, right


# ──────────────────────────────────────────────────────────────────────────
# Behavior trace mini-plot
# ──────────────────────────────────────────────────────────────────────────
def draw_behavior_traces(ax, t, eyepos, speed, x, y, w, h, *,
                         label_color="#222", trace_lw=0.9,
                         eye_color_x="#1f6feb", eye_color_y="#d68900",
                         speed_color="#7e3f8a"):
    """Draw two stacked tiny line plots inside (x, y, w, h):
        top  : eye position x and y (deg) vs time
        bottom: eye speed (deg/s) vs time
    Box outlines are suppressed; only axis labels."""
    # Sub-rects
    pad = 0.05 * h
    sub_h = (h - pad) / 2
    sub_x = x + 0.05 * w
    sub_w = w - 0.10 * w

    # Top: eye position
    py0 = y + sub_h + pad
    _trace_box(ax, t, eyepos[:, 0], sub_x, py0, sub_w, sub_h,
               color=eye_color_x, lw=trace_lw, label="x")
    _trace_overlay(ax, t, eyepos[:, 1], sub_x, py0, sub_w, sub_h,
                   ymin=eyepos.min(), ymax=eyepos.max(),
                   color=eye_color_y, lw=trace_lw, label="y")
    ax.text(sub_x - 0.05 * w, py0 + sub_h / 2, "eye pos\n(deg)",
            ha="right", va="center", fontsize=6.5, color="#444",
            linespacing=1.1)

    # Bottom: speed
    py1 = y
    _trace_box(ax, t, speed, sub_x, py1, sub_w, sub_h,
               color=speed_color, lw=trace_lw, label=None)
    ax.text(sub_x - 0.05 * w, py1 + sub_h / 2, "speed\n(deg/s)",
            ha="right", va="center", fontsize=6.5, color="#444",
            linespacing=1.1)
    # Time scale bar — 100 ms representative interval
    scale_ms = 100
    scale_s = scale_ms / 1000.0
    span_s = float(t[-1] - t[0])
    bar_frac = min(scale_s / span_s, 0.5)
    bar_x1 = sub_x + sub_w - 0.05
    bar_x0 = bar_x1 - bar_frac * sub_w
    bar_y = y - 0.15
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#222", lw=1.5,
            clip_on=False)
    ax.text((bar_x0 + bar_x1) / 2, bar_y - 0.05,
            f"{scale_ms} ms", ha="center", va="top",
            fontsize=6.5, color="#222")


def _trace_box(ax, t, y_arr, x, y, w, h, *, color, lw, label=None):
    """Draw a single line plot inside a rect (no axes)."""
    t0, t1 = t.min(), t.max()
    ymin, ymax = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
    if ymax == ymin:
        ymax = ymin + 1.0
    xs = x + (t - t0) / (t1 - t0) * w
    ys = y + (y_arr - ymin) / (ymax - ymin) * h * 0.9 + 0.05 * h
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
    if label:
        ax.text(xs[-1] + 0.05, ys[-1], label, ha="left", va="center",
                fontsize=6, color=color)


def _trace_overlay(ax, t, y_arr, x, y, w, h, *, ymin, ymax, color, lw, label):
    """Plot another trace on top, sharing the same y-scale as the box owner."""
    t0, t1 = t.min(), t.max()
    if ymax == ymin:
        ymax = ymin + 1.0
    xs = x + (t - t0) / (t1 - t0) * w
    ys = y + (y_arr - ymin) / (ymax - ymin) * h * 0.9 + 0.05 * h
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)
    if label:
        ax.text(xs[-1] + 0.05, ys[-1], label, ha="left", va="center",
                fontsize=6, color=color)


# ──────────────────────────────────────────────────────────────────────────
# Misc primitives
# ──────────────────────────────────────────────────────────────────────────
def flow_arrow(ax, x0, y, x1, *, color=ARROW_COLOR, lw=1.0):
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color),
    )


def slash_separator(ax, x, y_center, height, color="#666", lw=1.2,
                    slant=0.3):
    """A forward-slash ' / ' style separator between two zones."""
    half = height / 2
    ax.plot([x - slant * half, x + slant * half],
            [y_center - half, y_center + half],
            color=color, linewidth=lw, zorder=2.5)


# ──────────────────────────────────────────────────────────────────────────
# Kernel-prism primitives (shared visual language with stimulus cube)
#   Convention for every conv kernel in the architecture half:
#     +x (right)  ← time taps (kt)
#     +y (up)     ← spatial-vertical taps (kh)
#     +z (into)   ← spatial-horizontal taps (kw)
#   So a (kt, kh, kw) kernel becomes an axis-aligned cabinet-projected box
#   with size (kt*S_T, kh*S_PIX, kw*S_PIX). Front face (z=0) shows the
#   newest-time slab; depth goes back-into-page (up-and-LEFT in projection).
# ──────────────────────────────────────────────────────────────────────────


def draw_kernel_prism(ax, front_lower_left_3d, size_3d, *,
                      front_color, side_color, top_color,
                      edge_color="#1b3a5b", edge_width=0.35,
                      zorder=2.0, alpha=1.0):
    """Draw a kernel prism whose front-lower-left corner sits at the given
    3D world point, with extents (sx, sy, sz). Cabinet projection.

    Returns the three projected quads (front, top, side) for use by callers
    that want to attach decorations. `side` is the right face (depth goes
    up-and-right in our cabinet projection).
    """
    fx, fy, fz = front_lower_left_3d
    sx, sy, sz = size_3d
    center = (fx + sx / 2.0, fy + sy / 2.0, fz + sz / 2.0)
    return draw_axis_aligned_prism(
        ax, center, (sx, sy, sz),
        front_color=front_color, side_color=side_color, top_color=top_color,
        edge_color=edge_color, edge_width=edge_width, zorder=zorder,
        alpha=alpha,
    )


def draw_channel_grid(ax, x_left, y0, z0, *, n_channels, rows, cols,
                      kt, kh, kw, gap=0.04,
                      palette=("#cfe2f3", "#7fa4c4", "#4d7396", "#1b3a5b"),
                      base_zorder=2.0, edge_width=0.3,
                      hue_jitter=0.0):
    """Tile `n_channels` kernel prisms in a (rows × cols) grid in the
    (y, z) plane at fixed x stripe `x_left`.

    rows index → y (up); cols index → z (into page).
    Cells are drawn BACK-TO-FRONT (largest z first) so front-most kernels
    correctly occlude rear ones.

    Returns a dict with:
        front_xs : (x_left + kt, ) right face x in front plane
        x_left   : passed back through
        x_right  : x_left + kt (front face right x)
        y_bottom : y0 (front-bottom of grid)
        y_top    : y0 + rows*pitch_y - gap (front-top of grid)
        z_back   : z0 + cols*pitch_z - gap (back face z)
        bbox2d   : (xmin, xmax, ymin, ymax) of the projected grid
        front_center_2d : projected (x, y) center of the front face of the grid
        back_center_2d  : projected (x, y) center of the back-most layer
    """
    pitch_y = kh + gap
    pitch_z = kw + gap

    # collect projected extremes
    xs, ys = [], []

    # back-to-front in z, then any order in y
    drawn = 0
    n_z_layers = cols
    for iz in range(n_z_layers - 1, -1, -1):
        z_layer = z0 + iz * pitch_z
        for iy in range(rows):
            if drawn >= n_channels:
                break
            y_cell = y0 + iy * pitch_y
            front_ll = (x_left, y_cell, z_layer)
            # zorder: rear layers low, front layers high. Within a layer
            # bump slightly by row so adjacency edges resolve cleanly.
            z_local = base_zorder + (n_z_layers - 1 - iz) * 0.5 + iy * 0.01
            fc, sc, tc, ec = _maybe_jitter_palette(palette, hue_jitter,
                                                   iy * cols + iz, drawn)
            front, top, side = draw_kernel_prism(
                ax, front_ll, (kt, kh, kw),
                front_color=fc, side_color=sc, top_color=tc,
                edge_color=ec, edge_width=edge_width,
                zorder=z_local,
            )
            for quad in (front, top, side):
                xs.extend(quad[:, 0])
                ys.extend(quad[:, 1])
            drawn += 1

    # Front-most layer z=z0; back-most layer z=z0 + (cols-1)*pitch_z
    z_front = z0
    z_back = z0 + (n_z_layers - 1) * pitch_z + kw
    y_bot = y0
    y_top = y0 + (rows - 1) * pitch_y + kh

    front_center_3d = np.array([x_left + kt / 2, (y_bot + y_top) / 2, z_front])
    back_center_3d  = np.array([x_left + kt / 2, (y_bot + y_top) / 2, z_back])
    f2 = cab_project(front_center_3d)
    b2 = cab_project(back_center_3d)

    return {
        "x_left": x_left,
        "x_right": x_left + kt,
        "y_bottom": y_bot,
        "y_top": y_top,
        "z_front": z_front,
        "z_back": z_back,
        "bbox2d": (min(xs), max(xs), min(ys), max(ys)),
        "front_center_2d": f2,
        "back_center_2d": b2,
    }


def _maybe_jitter_palette(palette, jitter, seed, drawn):
    """Optional small hue jitter on cell color so a grid of channels reads
    as a sea of slightly-different units rather than one solid block. With
    jitter=0 (default) returns the palette unchanged."""
    if jitter <= 0:
        return palette
    rng = np.random.default_rng(seed * 7 + drawn)
    delta = (rng.random(3) - 0.5) * 2.0 * jitter
    fc = _shift_hex(palette[0], delta[0])
    sc = _shift_hex(palette[1], delta[1])
    tc = _shift_hex(palette[2], delta[2])
    return (fc, sc, tc, palette[3])


def _shift_hex(hexcol, delta):
    """Shift hex color value by `delta` ∈ [-1, 1] (clipped)."""
    hexcol = hexcol.lstrip("#")
    r, g, b = (int(hexcol[i:i+2], 16) for i in (0, 2, 4))
    scale = 1.0 + delta * 0.25
    r = int(np.clip(r * scale, 0, 255))
    g = int(np.clip(g * scale, 0, 255))
    b = int(np.clip(b * scale, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_temporal_weight_traces(ax, weights, x0, y0, w, h, *,
                                gap=0.10, color="#b8860b", lw=0.9,
                                baseline_color="#bbb"):
    """Plot `weights` (C, K) as C small line plots in panels of width `w/C`
    above a row of frontend kernels. Each panel spans `panel_w × h` and
    centers on the corresponding kernel's x-stripe.

    All traces share a common y-scale so amplitudes are directly comparable.
    A faint horizontal baseline marks zero.
    """
    weights = np.asarray(weights, dtype=float)
    C, K = weights.shape
    panel_w = (w - gap * (C - 1)) / C
    ymin = float(weights.min())
    ymax = float(weights.max())
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    ts = np.linspace(0, 1, K)
    for c in range(C):
        px = x0 + c * (panel_w + gap)
        # baseline at y=0
        if ymin < 0 < ymax:
            y_zero = y0 + (0 - ymin) / (ymax - ymin) * h
            ax.plot([px, px + panel_w], [y_zero, y_zero],
                    color=baseline_color, lw=0.5, zorder=3)
        xs = px + ts * panel_w
        ys = y0 + (weights[c] - ymin) / (ymax - ymin) * h
        ax.plot(xs, ys, color=color, lw=lw, zorder=3.5,
                solid_capstyle="round")


def draw_skip_U(ax, x0, x1, y_top, *, depth=0.6, color=CYAN, lw=1.2,
                zorder=1.6, label=None, label_fontsize=6.5):
    """Cyan U-shaped connector from (x0, y_top) down by `depth` and back up
    to (x1, y_top). Implements the residual path of a ResBlock."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    y_bot = y_top - depth
    # Smooth Bezier U: down with curve at top, across, up with curve.
    cx_pad = min(0.25 * (x1 - x0), depth * 0.8)
    verts = [
        (x0, y_top),                          # start
        (x0, y_top - depth * 0.6),            # ctrl down
        (x0 + cx_pad, y_bot),                 # to bottom-left
        (x1 - cx_pad, y_bot),                 # line across
        (x1, y_top - depth * 0.6),            # ctrl up
        (x1, y_top),                          # end
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    ax.add_patch(PathPatch(path, edgecolor=color, facecolor="none",
                           lw=lw, zorder=zorder, capstyle="round"))
    if label is not None:
        # Place label OUTSIDE the U (just under its lowest point), small
        # and unobtrusive — most of the meaning is carried by the cyan
        # arc itself, matching the stimulus-cube color.
        ax.text((x0 + x1) / 2, y_bot - 0.05, label,
                ha="center", va="top", fontsize=label_fontsize,
                color=color, style="italic", zorder=zorder + 0.1)


def draw_skip_staple(ax, x0, x1, y_top, *, depth=0.6, corner_r=0.12,
                     color="#222", lw=1.1, zorder=1.6):
    """Inverted-U (⊔) residual connector: drops from (x0, y_top), runs
    horizontally across the bottom, climbs back up to (x1, y_top). Square
    routing with rounded corners — reads as a hardware-trace style bypass
    rather than a smooth arc."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    r = min(corner_r, depth * 0.5, (x1 - x0) * 0.45)
    y_bot = y_top - depth
    verts = [
        (x0, y_top),               # start at top-left
        (x0, y_bot + r),           # straight down to start of corner
        (x0, y_bot),               # quadratic ctrl: bottom-left corner
        (x0 + r, y_bot),           # corner end
        (x1 - r, y_bot),           # across the bottom
        (x1, y_bot),               # quadratic ctrl: bottom-right corner
        (x1, y_bot + r),           # corner end
        (x1, y_top),               # straight up to end
    ]
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO]
    ax.add_patch(PathPatch(Path(verts, codes),
                           edgecolor=color, facecolor="none",
                           lw=lw, zorder=zorder,
                           capstyle="round", joinstyle="round"))


def draw_recurrent_loop(ax, x0, x1, y_top, *, arc_height=0.9, color="#7e3f8a",
                        lw=1.4, zorder=4.5, label="h$_{t-1}$",
                        label_fontsize=7.0, inset=0.18):
    """Tall closed loop perched on top of a block, indicating recurrence.
    Two near-vertical legs (input on the left, output on the right) connected
    by a flat top, drawn as a single rounded rectangular path with an
    arrowhead on the descending (left) leg.

    `x0` (left) and `x1` (right) are the block's left/right x; `y_top` is the
    block's projected top. `inset` shrinks the loop horizontally so it stays
    visually planted on the block."""
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch
    span = x1 - x0
    lx = x0 + inset * span
    rx = x1 - inset * span
    base_y = y_top + 0.04
    top_y = y_top + arc_height
    corner_r = min(0.18, (rx - lx) * 0.35, arc_height * 0.45)

    # Use a FancyArrowPatch along a custom path to get an arrowhead on the
    # descending leg. Construct an explicit rounded-rectangle path: up the
    # right leg → across the top → down the left leg into the block top.
    verts = [
        (rx, base_y),                       # start: rises from right edge
        (rx, top_y - corner_r),             # straight up
        (rx, top_y),                        # ctrl: top-right round
        (rx - corner_r, top_y),             # corner end
        (lx + corner_r, top_y),             # across the top
        (lx, top_y),                        # ctrl: top-left round
        (lx, top_y - corner_r),             # corner end
        (lx, base_y),                       # straight down (arrow target)
    ]
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO]
    path = Path(verts, codes)
    ax.add_patch(FancyArrowPatch(
        path=path,
        arrowstyle="-|>", lw=lw, color=color,
        zorder=zorder, mutation_scale=12,
        joinstyle="round", capstyle="round",
    ))
    if label is not None:
        ax.text((lx + rx) / 2, top_y + 0.08, label,
                ha="center", va="bottom", fontsize=label_fontsize,
                color=color, style="italic", zorder=zorder + 0.1)


def draw_gaussian_readout(ax, mean, std, features, x0, y0, *,
                          size=0.85, feat_width=0.18, gap=0.12,
                          zorder=4.0, label_color=TEXT_COLOR):
    """Draw the Gaussian readout glyph: a small 2D Gaussian face showing
    the sampling location (mean) and extent (std) in normalized feature-map
    coords, followed by a thin heatmap strip of the depthwise feature
    weights.

    `mean`, `std` are 2-vectors in [-1, 1] feature-map coords. `features`
    is a 1D array of per-feature readout weights.
    """
    res = 96
    grid = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(grid, grid)
    sx = max(float(std[0]), 0.05)
    sy = max(float(std[1]), 0.05)
    G = np.exp(-(((X - float(mean[0])) ** 2) / (2 * sx * sx)
                 + ((Y - float(mean[1])) ** 2) / (2 * sy * sy)))
    G = (G / G.max() * 255).astype(np.uint8)

    # Gaussian face — use a light→saturated green colormap to match readout.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "readout_gauss", ["#f4faf2", "#8cc28c", "#1f5e1f"], N=256,
    )
    ax.imshow(G, extent=(x0, x0 + size, y0, y0 + size),
              origin="lower", cmap=cmap, zorder=zorder,
              interpolation="bilinear")
    # Crisp outline so it reads as a "face" matching nearby prisms.
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False,
                           edgecolor="#1b3a5b", linewidth=0.5,
                           zorder=zorder + 0.2))
    # Mean dot + std ellipse
    cx = x0 + (float(mean[0]) + 1) / 2 * size
    cy = y0 + (float(mean[1]) + 1) / 2 * size
    rx = sx / 2 * size
    ry = sy / 2 * size
    ax.add_patch(Circle((cx, cy), 0.012 * size + 0.02,
                        facecolor="#222", edgecolor="white", linewidth=0.4,
                        zorder=zorder + 0.4))
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fill=False,
                         edgecolor="#222", linewidth=0.7, linestyle=(0, (3, 2)),
                         zorder=zorder + 0.3))

    # Feature-weight strip to the right
    fw_x0 = x0 + size + gap
    feats = np.asarray(features, dtype=float)
    fw_img = feats[::-1, None]   # (n_feat, 1), top = first feature
    vmax = float(np.abs(feats).max()) or 1.0
    ax.imshow(fw_img, extent=(fw_x0, fw_x0 + feat_width, y0, y0 + size),
              origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              aspect="auto", zorder=zorder, interpolation="nearest")
    ax.add_patch(Rectangle((fw_x0, y0), feat_width, size, fill=False,
                           edgecolor="#1b3a5b", linewidth=0.5,
                           zorder=zorder + 0.2))
    ax.text(fw_x0 + feat_width / 2, y0 - 0.08,
            f"{len(feats)}", ha="center", va="top",
            fontsize=6.0, color=label_color, style="italic")
    ax.text(fw_x0 + feat_width / 2, y0 + size + 0.08,
            "ch wts", ha="center", va="bottom",
            fontsize=6.0, color=label_color, style="italic")

    return {
        "x_left": x0,
        "x_right": fw_x0 + feat_width,
        "y_bottom": y0,
        "y_top": y0 + size,
        "feat_strip_x": fw_x0,
    }


def draw_neuron_trace_panel(ax, t, robs_rate, rhat_rate, x0, y0, w, h, *,
                            obs_color="#777", pred_color="#1f5e1f",
                            obs_lw=0.9, pred_lw=1.3, zorder=4.0,
                            label=None, baseline_label=None,
                            show_scale=False, scale_ms=100,
                            scale_sp_s=None, frame_color="#bbb"):
    """Draw a tiny line plot of observed and predicted spike-rate traces.

    `robs_rate` and `rhat_rate` are 1D arrays in sp/s sampled at `t` seconds.
    The panel occupies the data-coord rectangle (x0, y0, w, h). Observed is
    drawn first (in `obs_color`) and predicted is overlaid (in `pred_color`).

    A faint frame is drawn around the panel; optional inline labels and
    scale bars can be enabled for the bottom-most panel.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(robs_rate, dtype=float)
    p = np.asarray(rhat_rate, dtype=float)
    if t.size == 0:
        return
    t0, t1 = float(t[0]), float(t[-1])
    if t1 == t0:
        t1 = t0 + 1.0
    ymin = 0.0
    ymax = float(np.nanmax(np.concatenate([r, p])))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ymax *= 1.10  # headroom

    ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                           edgecolor=frame_color, linewidth=0.4,
                           zorder=zorder - 0.1))

    xs = x0 + (t - t0) / (t1 - t0) * w
    ys_obs = y0 + (r - ymin) / (ymax - ymin) * h
    ys_pred = y0 + (p - ymin) / (ymax - ymin) * h
    ax.plot(xs, ys_obs, color=obs_color, lw=obs_lw, zorder=zorder,
            solid_capstyle="round")
    ax.plot(xs, ys_pred, color=pred_color, lw=pred_lw, zorder=zorder + 0.1,
            solid_capstyle="round")

    if label is not None:
        ax.text(x0 + w - 0.04, y0 + h - 0.04, label,
                ha="right", va="top", fontsize=6.0, color="#333",
                zorder=zorder + 0.5)
    if baseline_label is not None:
        ax.text(x0 + 0.04, y0 + h - 0.04, baseline_label,
                ha="left", va="top", fontsize=6.0, color="#555",
                style="italic", zorder=zorder + 0.5)

    if show_scale:
        scale_s = scale_ms / 1000.0
        bar_frac = min(scale_s / (t1 - t0), 0.5)
        bar_x1 = x0 + w
        bar_x0 = bar_x1 - bar_frac * w
        bar_y = y0 - 0.08
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#222", lw=1.3,
                clip_on=False, zorder=zorder + 0.5)
        ax.text((bar_x0 + bar_x1) / 2, bar_y - 0.04,
                f"{scale_ms} ms", ha="center", va="top",
                fontsize=6.0, color="#222", clip_on=False)
        if scale_sp_s is not None:
            sb_x = x0 - 0.06
            sb_h_frac = min(scale_sp_s / (ymax - ymin), 0.9)
            sb_y0 = y0
            sb_y1 = y0 + sb_h_frac * h
            ax.plot([sb_x, sb_x], [sb_y0, sb_y1], color="#222", lw=1.3,
                    clip_on=False, zorder=zorder + 0.5)
            ax.text(sb_x - 0.04, (sb_y0 + sb_y1) / 2,
                    f"{int(scale_sp_s)}\nsp/s",
                    ha="right", va="center", fontsize=6.0, color="#222",
                    linespacing=1.0, clip_on=False)


def draw_pool_glyph(ax, x, y, *, color="#222", fontsize=7.5, zorder=12.0):
    """Small '↓2×' badge marking a 2× spatial downsample on a flow arrow.
    Default zorder sits above kernel back-layers so the white bbox cleanly
    obscures any kernels behind the badge."""
    ax.text(x, y, "↓2×", ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold",
            zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="white", edgecolor=color, linewidth=0.7))
