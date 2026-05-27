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
