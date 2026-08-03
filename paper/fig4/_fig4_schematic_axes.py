"""Loading the cached panel-A/C schematic payload, and annotating crop axes.

The bottom layer of what used to be one 2,900-line `_fig4_ssi_common`: reading
`fig4_schematic_final_maps.npz` and friends, deriving the contour-window and
center-zoom geometry from them, and the toolkit that draws that geometry onto a
stimulus crop axis (contour window, contour axis line, trajectory span arrows,
center-zoom box, ROI connectors, image labels).

Everything here takes an axis and some geometry and draws on it, or reads a
cache and returns numbers. Nothing here knows what panel it is drawing, which
is what makes it separable from `_fig4_ssi_common`, and the dependency runs one
way: that module imports this one.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

import _fig4_paths as _paths
from _fig4_style import GRAY, INK

try:
    import _fig4_contour_schematic as ssi_schematic

    SCHEMATIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback path is visual, not unit-tested.
    ssi_schematic = None
    SCHEMATIC_IMPORT_ERROR = exc


CYAN = "#00A9C8"

TRACE_COLOR = "#8B1E3F"

CONTOUR_WINDOW = "#E6A700"

ZOOM_BOX = "#E8118A"  # vivid magenta -- center-zoom marker box, connectors, and the crop border all use this

CROP_BORDER_LW = 2.6

CENTER_ZOOM_HALF_DEG = 0.25

CENTER_ZOOM_TRACE_PAD_DEG = 0.04

CONTOUR_AXIS_DASH = (0, (2.1, 1.5))

D_IMAGE_LABEL_FS = 5.8

D_CENTER_ZOOM_BOX_LW = 1.05

D_CONNECTOR_LW = 0.95

def read_schematic_payload() -> dict | None:
    """Panel A's schematic payload.

    The bare `except: return None` this replaces swallowed every failure --
    a missing cache, a corrupt npz, a genuine bug in the loader -- and turned
    all of them into a silently substituted synthetic schematic. Errors now
    propagate; only explicitly-enabled degraded rendering yields None.
    """
    if ssi_schematic is None:
        if _paths.ALLOW_MISSING:
            return None
        raise _paths.Fig4MissingInput(
            "panel A's contour schematic module failed to import, so the panel "
            "cannot be drawn from real data.\n"
            f"  underlying error: {SCHEMATIC_IMPORT_ERROR!r}"
        )
    return ssi_schematic.load_real_payload()

def _finite_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)

def contour_window_metadata(payload: dict | None) -> dict[str, float | str]:
    """Return display metadata for the local contour-axis/coherence aperture."""
    default_ppd = 37.50476617
    if ssi_schematic is not None:
        default_ppd = _finite_float(getattr(ssi_schematic, "MODEL_PPD", default_ppd), default_ppd)

    row = {}
    if isinstance(payload, dict) and isinstance(payload.get("stimulus_row"), dict):
        row = payload["stimulus_row"]

    radius_px = _finite_float(row.get("image_patch_radius_px"), default_ppd)
    crop_size_px = _finite_float((payload or {}).get("stimulus_crop_size_px"), 151.0)
    if crop_size_px <= 0:
        crop_size_px = 151.0
    ppd = _finite_float(row.get("ppd"), default_ppd)
    if ppd <= 0:
        ppd = default_ppd
    radius_deg = radius_px / ppd
    zoom_half_px = CENTER_ZOOM_HALF_DEG * ppd
    axis_image_deg = _finite_float((payload or {}).get("contour_axis_image_deg"), 10.352312)
    coherence = _finite_float(row.get("image_orientation_coherence"), float("nan"))
    radius_label = "1 deg radius" if 0.93 <= radius_deg <= 1.08 else f"{radius_deg:.2f} deg radius"
    return {
        "radius_px": radius_px,
        "radius_deg": radius_deg,
        "radius_fraction": min(0.48, max(0.08, radius_px / crop_size_px)),
        "crop_size_px": crop_size_px,
        "ppd": ppd,
        "center_zoom_half_deg": CENTER_ZOOM_HALF_DEG,
        "center_zoom_half_px": zoom_half_px,
        "center_zoom_fraction": min(0.48, max(0.03, zoom_half_px / crop_size_px)),
        "center_zoom_mode": "minimum",
        "axis_image_deg": axis_image_deg,
        "coherence": coherence,
        "radius_label": radius_label,
    }

def trace_fit_center_zoom_metadata(
    metadata: dict[str, float | str],
    motion_eye: object | None,
) -> dict[str, float | str]:
    """Expand the center zoom just enough to contain the displayed traces."""
    if not isinstance(motion_eye, dict):
        return metadata

    trace_extents: list[float] = []
    for key in ("small_xy_px", "large_xy_px"):
        try:
            trace = np.asarray(motion_eye.get(key), dtype=np.float64)
        except Exception:
            continue
        if trace.ndim != 2 or trace.shape[1] != 2 or trace.size == 0:
            continue
        finite = trace[np.isfinite(trace)]
        if finite.size:
            trace_extents.append(float(np.nanmax(np.abs(finite))))
    if not trace_extents:
        return metadata

    updated = dict(metadata)
    ppd = _finite_float(updated.get("ppd"), 37.50476617)
    crop_size_px = _finite_float(updated.get("crop_size_px"), 151.0)
    min_half_px = CENTER_ZOOM_HALF_DEG * ppd
    trace_max_px = max(trace_extents)
    half_px = max(min_half_px, trace_max_px + CENTER_ZOOM_TRACE_PAD_DEG * ppd)
    updated["center_zoom_half_px"] = half_px
    updated["center_zoom_half_deg"] = half_px / ppd if ppd > 0 else CENTER_ZOOM_HALF_DEG
    updated["center_zoom_fraction"] = min(0.48, max(0.03, half_px / crop_size_px))
    updated["center_zoom_trace_max_px"] = trace_max_px
    updated["center_zoom_trace_max_deg"] = trace_max_px / ppd if ppd > 0 else float("nan")
    updated["center_zoom_mode"] = "trace_fit" if half_px > min_half_px else "minimum"
    return updated

def data_width_for_physical_aspect(ax: plt.Axes, height: float, width_over_height: float) -> float:
    """Convert a desired physical aspect into parent-axis data coordinates."""
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    axis_w = float(bbox.width) * float(fig_w)
    axis_h = float(bbox.height) * float(fig_h)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = abs(float(x1) - float(x0))
    yspan = abs(float(y1) - float(y0))
    if axis_w <= 0 or axis_h <= 0 or xspan <= 0 or yspan <= 0:
        return float(height) * float(width_over_height)
    return float(width_over_height) * float(height) * (axis_h / axis_w) * (xspan / yspan)

def stimulus_canvas_aspect(payload: dict | None) -> float:
    canvas = (payload or {}).get("stimulus_canvas")
    try:
        height, width = canvas.shape[:2]
        if float(height) > 0:
            return float(width) / float(height)
    except Exception:
        pass
    return 16.0 / 9.0

def _axis_vector_image(axis_image_deg: float) -> tuple[float, float]:
    if ssi_schematic is not None and hasattr(ssi_schematic, "axis_vector_image"):
        try:
            vec = ssi_schematic.axis_vector_image(float(axis_image_deg))
            return float(vec[0]), float(vec[1])
        except Exception:
            pass
    theta = math.radians(float(axis_image_deg))
    return math.cos(theta), math.sin(theta)

def add_contour_window_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Draw the real 1-deg contour-analysis aperture in crop pixel coordinates."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    norm = math.hypot(dx, dy)
    if norm <= 0:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / norm, dy / norm

    crop_ax.add_patch(
        patches.Circle(
            (center, center),
            radius,
            fill=False,
            edgecolor=CONTOUR_WINDOW,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.2)),
            zorder=8,
        )
    )
    local_half = radius * 0.84
    crop_ax.plot(
        [center - dx * local_half, center + dx * local_half],
        [center - dy * local_half, center + dy * local_half],
        color="white",
        lw=1.5,
        ls=CONTOUR_AXIS_DASH,
        alpha=0.95,
        solid_capstyle="round",
        zorder=9,
    )

def add_contour_axis_line_to_crop_axis(
    crop_ax: plt.Axes,
    metadata: dict[str, float | str],
    *,
    zoomed: bool = False,
) -> None:
    """Draw only the local contour axis line, useful for the zoomed crop."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    radius = _finite_float(metadata.get("radius_px"), 38.0)
    half = radius * 0.84
    if zoomed:
        half = min(half, _finite_float(metadata.get("center_zoom_half_px"), half) * 0.78)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    norm = math.hypot(dx, dy)
    if norm <= 0:
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx / norm, dy / norm
    crop_ax.plot(
        [center - dx * half, center + dx * half],
        [center - dy * half, center + dy * half],
        color="white",
        lw=1.5,
        ls=CONTOUR_AXIS_DASH,
        alpha=0.95,
        solid_capstyle="round",
        zorder=22,
    )

def add_trace_path_without_marker(ax: plt.Axes, center: np.ndarray, trace_xy_px: object, color: str, *, lw: float, zorder: float) -> None:
    """Draw the eye trajectory path without the endpoint dot used upstream."""
    trace = np.asarray(trace_xy_px, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] < 2:
        return
    points = np.asarray(center, dtype=np.float64)[None, :] + trace
    ax.plot(
        points[:, 0],
        points[:, 1],
        color="white",
        lw=float(lw) + 0.85,
        alpha=0.72,
        solid_capstyle="round",
        zorder=zorder - 0.1,
    )
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        lw=float(lw),
        alpha=0.92,
        solid_capstyle="round",
        zorder=zorder,
    )

def add_trajectory_span_arrows_to_crop_axis(
    crop_ax: plt.Axes,
    metadata: dict[str, float | str],
    motion_eye: dict | None,
) -> None:
    """Draw double-headed along/across arrows spanning the displayed trace."""
    if not isinstance(motion_eye, dict) or "large_xy_px" not in motion_eye:
        return
    trace = np.asarray(motion_eye["large_xy_px"], dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 2:
        return
    trace = trace[np.all(np.isfinite(trace), axis=1)]
    if trace.shape[0] < 2:
        return

    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = np.array([0.5 * (n - 1.0), 0.5 * (n - 1.0)], dtype=np.float64)
    dx, dy = _axis_vector_image(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    tangent = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm <= 0:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    along = trace @ tangent
    across = trace @ normal
    along_span = max(float(np.nanmax(along) - np.nanmin(along)), 1.5)
    across_span = max(float(np.nanmax(across) - np.nanmin(across)), 1.5)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), 18.8) * 0.82

    along_min = max(float(np.nanmin(along) - 0.08 * along_span), -half_px)
    along_max = min(float(np.nanmax(along) + 0.08 * along_span), half_px)
    along_cross = float(np.nanmedian(across) - 0.34 * across_span)
    along_cross = float(np.clip(along_cross, -half_px * 0.62, half_px * 0.62))

    across_min = max(float(np.nanmin(across) - 0.12 * across_span), -half_px)
    across_max = min(float(np.nanmax(across) + 0.12 * across_span), half_px)
    across_along = float(np.nanmax(along) + 0.34 * along_span)
    across_along = float(np.clip(across_along, -half_px * 0.62, half_px * 0.76))

    def point(a: float, c: float) -> tuple[float, float]:
        p = center + tangent * a + normal * c
        return float(p[0]), float(p[1])

    def display_angle(vector: np.ndarray) -> float:
        origin_disp = crop_ax.transData.transform(center)
        end_disp = crop_ax.transData.transform(center + vector)
        angle = math.degrees(math.atan2(end_disp[1] - origin_disp[1], end_disp[0] - origin_disp[0]))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        return angle

    def shift_up(point_xy: tuple[float, float], amount_px: float) -> tuple[float, float]:
        return (point_xy[0], point_xy[1] - amount_px)

    def shift_right(point_xy: tuple[float, float], amount_px: float) -> tuple[float, float]:
        return (point_xy[0] + amount_px, point_xy[1])

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        for color, lw, alpha, zorder in [("black", 2.7, 0.62, 27), ("white", 1.25, 0.98, 28)]:
            crop_ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw, mutation_scale=8, shrinkA=0, shrinkB=0),
                alpha=alpha,
                zorder=zorder,
            )

    along_up_shift = min(half_px * 0.24, 4.6)
    across_right_shift = min(half_px * 0.16, 3.4)
    along_start = shift_up(point(along_min, along_cross), along_up_shift)
    along_end = shift_up(point(along_max, along_cross), along_up_shift)
    across_start = shift_right(point(across_along, across_min), across_right_shift)
    across_end = shift_right(point(across_along, across_max), across_right_shift)
    arrow(along_start, along_end)
    arrow(across_start, across_end)

    label_style = dict(
        fontsize=5.8,
        color="white",
        ha="center",
        va="center",
        rotation_mode="anchor",
        clip_on=False,
        zorder=31,
    )
    along_label = shift_up(point(0.5 * (along_min + along_max), along_cross - 0.15 * across_span), along_up_shift)
    crop_ax.text(*along_label, "along", rotation=display_angle(tangent), **label_style)
    crop_ax.text(
        1.075,
        0.46,
        "across",
        transform=crop_ax.transAxes,
        color=INK,
        rotation=display_angle(normal),
        **{key: value for key, value in label_style.items() if key != "color"},
    )

def restore_trace_orientation(motion_eye: dict | None) -> dict | None:
    """Undo the extra display rotation baked into the "large"/long-path
    trace by _fig4_contour_schematic.py's selected_real_panel_a_trace_pair
    (PANEL_A_LARGE_TRACE_ROTATION_DEG = 90 deg, PANEL_A_SMALL_TRACE_ROTATION_DEG
    = 0 deg) for that module's own Panel A schematic. D/G want the traces at
    their real recorded orientation, not that rotation, so rotate back by the
    same angle in reverse. Only applies when the real trace bank was used
    (large_trace_index >= 0); the synthetic fallback trace was never rotated.
    """
    if not isinstance(motion_eye, dict) or ssi_schematic is None:
        return motion_eye
    if motion_eye.get("large_trace_index", -1) is None or motion_eye.get("large_trace_index", -1) < 0:
        return motion_eye
    rotate = getattr(ssi_schematic, "rotate_trace_xy_px", None)
    if rotate is None:
        return motion_eye
    restored = dict(motion_eye)
    large_rotation_deg = getattr(ssi_schematic, "PANEL_A_LARGE_TRACE_ROTATION_DEG", 0.0)
    if large_rotation_deg and "large_xy_px" in restored:
        restored["large_xy_px"] = rotate(restored["large_xy_px"], -large_rotation_deg)
    small_rotation_deg = getattr(ssi_schematic, "PANEL_A_SMALL_TRACE_ROTATION_DEG", 0.0)
    if small_rotation_deg and "small_xy_px" in restored:
        restored["small_xy_px"] = rotate(restored["small_xy_px"], -small_rotation_deg)
    return restored

def add_center_zoom_box_to_crop_axis(crop_ax: plt.Axes, metadata: dict[str, float | str]) -> None:
    """Mark the tight central zoom window on the 151 px crop."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    crop_ax.add_patch(
        patches.Rectangle(
            (center - half_px, center - half_px),
            2.0 * half_px,
            2.0 * half_px,
            fill=False,
            edgecolor=ZOOM_BOX,
            linewidth=D_CENTER_ZOOM_BOX_LW,
            zorder=12,
        )
    )

def draw_plain_crop(
    crop_ax: plt.Axes,
    patch: object,
    *,
    trace_xy_px: object | None = None,
    trace_color: str | None = None,
) -> int:
    """Render a crop image directly instead of via ssi_schematic.add_stimulus.

    add_stimulus always draws a red+blue trace pair (or, with motion_eye=None,
    a red/blue placeholder double-arrow) with colors hardcoded to its own
    module -- there's no way to show only one trace, or to recolor it, through
    that function. This replicates just its image/border rendering, then
    optionally draws a single trace in a caller-chosen color via
    ssi_schematic.add_panel_a_trace_path directly. Returns the crop's pixel
    size (patches are square).
    """
    image = ssi_schematic.normalize_image(patch)
    n = int(image.shape[0])
    crop_ax.imshow(image, cmap="gray", interpolation="bicubic")
    crop_ax.set_aspect("equal", adjustable="box")
    hide_axis_completely(crop_ax)
    crop_ax.set_xlim(0, n - 1)
    crop_ax.set_ylim(n - 1, 0)
    crop_ax.add_patch(patches.Rectangle((0, 0), n - 1, n - 1, fill=False, lw=1.0, ec=INK))
    if trace_xy_px is not None and trace_color is not None:
        axis_center = np.array([0.5 * (n - 1), 0.5 * (n - 1)], dtype=np.float64)
        add_trace_path_without_marker(crop_ax, axis_center, trace_xy_px, trace_color, lw=1.85, zorder=4)
    return n

def add_upper_left_image_label(image_ax: plt.Axes, label: str) -> None:
    """Caption an inset image from its rendered upper-left corner."""
    x_left = float(image_ax.get_xlim()[0])
    y_top = float(image_ax.get_ylim()[1])
    image_ax.annotate(
        label,
        xy=(x_left, y_top),
        xycoords="data",
        xytext=(0, 3.6),
        textcoords="offset points",
        fontsize=D_IMAGE_LABEL_FS,
        color=GRAY,
        ha="left",
        va="bottom",
        linespacing=0.95,
        annotation_clip=False,
        clip_on=False,
        zorder=40,
    )

def add_lower_left_image_label(image_ax: plt.Axes, label: str) -> None:
    """Caption an inset image from its rendered lower-left corner."""
    x_left = float(image_ax.get_xlim()[0])
    y_bottom = float(image_ax.get_ylim()[0])
    image_ax.annotate(
        label,
        xy=(x_left, y_bottom),
        xycoords="data",
        xytext=(0, -3.6),
        textcoords="offset points",
        fontsize=D_IMAGE_LABEL_FS,
        color=GRAY,
        ha="left",
        va="top",
        linespacing=0.95,
        annotation_clip=False,
        clip_on=False,
        zorder=40,
    )

def add_zoomed_crop_view(
    zoom_ax: plt.Axes,
    schematic_payload: dict,
    motion_eye: dict | None,
    metadata: dict[str, float | str],
    *,
    trace_color: str = TRACE_COLOR,
    border_color: str = ZOOM_BOX,
    border_lw: float = CROP_BORDER_LW,
) -> None:
    """Draw the crop again, zoomed to the central +/-0.25 deg window."""
    if ssi_schematic is None:
        raise RuntimeError("SSI schematic helpers are unavailable")
    trace_xy_px = motion_eye.get("large_xy_px") if isinstance(motion_eye, dict) else None
    draw_plain_crop(zoom_ax, schematic_payload.get("patch"), trace_xy_px=trace_xy_px, trace_color=trace_color)
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    zoom_ax.set_xlim(center - half_px, center + half_px)
    zoom_ax.set_ylim(center + half_px, center - half_px)
    zoom_ax.add_patch(
        patches.Rectangle(
            (center - half_px, center - half_px),
            2.0 * half_px,
            2.0 * half_px,
            fill=False,
            edgecolor=border_color,
            linewidth=border_lw,
            zorder=20,
        )
    )

def add_roi_to_crop_connectors(
    ax: plt.Axes,
    full_ax: plt.Axes,
    crop_ax: plt.Axes,
    crop_center_xy: object,
    crop_size_px: object,
    *,
    color: str = CYAN,
) -> None:
    """Connect the source-image ROI square to the rendered crop border."""
    center = np.asarray(crop_center_xy, dtype=np.float64).reshape(-1)
    if center.size < 2:
        return
    size = _finite_float(crop_size_px, np.nan)
    if not np.isfinite(size) or size <= 0:
        return

    cx, cy = float(center[0]), float(center[1])
    right = cx + size / 2.0
    top = cy - size / 2.0
    bottom = cy + size / 2.0

    crop_x_left = float(crop_ax.get_xlim()[0])
    crop_y_bottom, crop_y_top = [float(v) for v in crop_ax.get_ylim()]
    endpoint_pairs = [
        ((right, top), (crop_x_left, crop_y_top)),
        ((right, bottom), (crop_x_left, crop_y_bottom)),
    ]
    for source_xy, target_xy in endpoint_pairs:
        connector = patches.ConnectionPatch(
            xyA=source_xy,
            xyB=target_xy,
            coordsA="data",
            coordsB="data",
            axesA=full_ax,
            axesB=crop_ax,
            arrowstyle="-",
            color=color,
            lw=D_CONNECTOR_LW,
            ls=(0, (3, 3)),
            alpha=0.58,
            clip_on=False,
            zorder=3,
        )
        ax.add_artist(connector)

def add_center_zoom_to_zoom_connectors(
    ax: plt.Axes,
    crop_ax: plt.Axes,
    zoom_ax: plt.Axes,
    metadata: dict[str, float | str],
    *,
    color: str = ZOOM_BOX,
) -> None:
    """Connect the center zoom box to the rendered zoomed-crop border."""
    n = _finite_float(metadata.get("crop_size_px"), 151.0)
    center = 0.5 * (n - 1.0)
    half_px = _finite_float(metadata.get("center_zoom_half_px"), CENTER_ZOOM_HALF_DEG * 37.50476617)
    if not np.isfinite(half_px) or half_px <= 0:
        return

    source_right = center + half_px
    source_top = center - half_px
    source_bottom = center + half_px
    target_left = float(zoom_ax.get_xlim()[0])
    target_bottom, target_top = [float(v) for v in zoom_ax.get_ylim()]
    endpoint_pairs = [
        ((source_right, source_top), (target_left, target_top)),
        ((source_right, source_bottom), (target_left, target_bottom)),
    ]
    for source_xy, target_xy in endpoint_pairs:
        connector = patches.ConnectionPatch(
            xyA=source_xy,
            xyB=target_xy,
            coordsA="data",
            coordsB="data",
            axesA=crop_ax,
            axesB=zoom_ax,
            arrowstyle="-",
            color=color,
            lw=D_CONNECTOR_LW,
            ls=(0, (3, 3)),
            alpha=0.70,
            clip_on=False,
            zorder=7,
        )
        ax.add_artist(connector)

def add_contour_window_parent_overlay(
    ax: plt.Axes,
    crop_x: float,
    crop_y: float,
    crop_w: float,
    crop_h: float,
    metadata: dict[str, float | str],
) -> None:
    """Fallback aperture overlay when the real crop inset is unavailable."""
    center_x = crop_x + 0.5 * crop_w
    center_y = crop_y + 0.5 * crop_h
    radius_fraction = _finite_float(metadata.get("radius_fraction"), 38.0 / 151.0)
    radius_x = crop_w * radius_fraction
    radius_y = crop_h * radius_fraction
    theta = math.radians(_finite_float(metadata.get("axis_image_deg"), 10.352312))
    dx = math.cos(theta)
    dy = -math.sin(theta)
    line_scale = min(0.46 / max(abs(dx), 1e-6), 0.46 / max(abs(dy), 1e-6))
    local_scale = radius_fraction * 0.84

    ax.plot(
        [crop_x + crop_w * (0.5 - dx * line_scale), crop_x + crop_w * (0.5 + dx * line_scale)],
        [crop_y + crop_h * (0.5 - dy * line_scale), crop_y + crop_h * (0.5 + dy * line_scale)],
        color=CYAN,
        lw=0.95,
        ls=(0, (4, 3)),
        zorder=18,
    )
    ax.add_patch(
        patches.Ellipse(
            (center_x, center_y),
            width=2.0 * radius_x,
            height=2.0 * radius_y,
            fill=False,
            edgecolor=CONTOUR_WINDOW,
            linewidth=0.9,
            linestyle=(0, (3.0, 2.0)),
            zorder=19,
        )
    )
    ax.plot(
        [crop_x + crop_w * (0.5 - dx * local_scale), crop_x + crop_w * (0.5 + dx * local_scale)],
        [crop_y + crop_h * (0.5 - dy * local_scale), crop_y + crop_h * (0.5 + dy * local_scale)],
        color=CYAN,
        lw=1.3,
        solid_capstyle="round",
        zorder=20,
    )
    ax.plot([center_x, center_x + radius_x], [center_y, center_y], color=CONTOUR_WINDOW, lw=0.65, zorder=21)
    ax.scatter([center_x], [center_y], s=15, facecolor="white", edgecolor=INK, linewidth=0.65, zorder=22)

def schematic_response_maps(
    payload: dict | None,
) -> tuple[dict[str, object], tuple[float, float] | None, dict[str, float]]:
    """Real/stabilized response maps for panel A, plus the scalar SSI (bits
    per spike) for each -- schematic_rr100_final_map_unit_metrics.csv's
    real_final_map_ssi_bits_per_spike / stable_final_map_ssi_bits_per_spike
    columns for whichever unit choose_right_panel_real_unit selected. SSI is
    one number per map, not a per-pixel quantity -- the map's own pixels are
    the model's predicted firing rate (spikes/s), mean-centered for display.
    """
    if ssi_schematic is None or payload is None:
        return {}, None, {}
    maps = payload.get("schematic_rr100_final_maps")
    condition_ids = payload.get("schematic_rr100_final_condition_id")
    unit_row = ssi_schematic.choose_right_panel_real_unit(payload)
    if maps is None or condition_ids is None or unit_row is None:
        return {}, None, {}
    try:
        ids = [str(x) for x in condition_ids]
        real_idx = ids.index("real_trace_final")
        stable_idx = ids.index("endpoint_stabilized_final")
        unit_idx = int(unit_row["unit_index"])
        real_maps = {
            "fem": maps[real_idx, unit_idx],
            "stable": maps[stable_idx, unit_idx],
        }
        ssi_bits_per_spike = {
            "fem": _finite_float(unit_row.get("real_final_map_ssi_bits_per_spike"), float("nan")),
            "stable": _finite_float(unit_row.get("stable_final_map_ssi_bits_per_spike"), float("nan")),
        }
        return real_maps, ssi_schematic.panel_b_map_pair_limits(real_maps.values()), ssi_bits_per_spike
    except Exception:
        return {}, None, {}

def hide_axis_completely(ax: plt.Axes) -> None:
    """``ax.set_axis_off()`` hides ticks/spines visually, but the default
    tick-label Text artists it leaves behind (e.g. an untouched 0-1 axes
    still carries '0.0'..'1.0' tick labels) keep reporting real bounding
    boxes to ``fig.get_tightbbox()`` -- matplotlib's axis-off draw path
    skips *drawing* them but doesn't check visibility when measuring tight
    bbox. In this file's v3 per-panel builds that made ``bbox_inches="tight"``
    silently grow a panel's saved page well past its intended size (e.g.
    panel D grew 0.51in taller from an invisible x-axis sitting below its
    own y=0), pushing it into whichever neighbor happened to be composited
    on top. Clearing the ticks outright (not just hiding the axis) removes
    the phantom Text artists so nothing is left to measure.
    """
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
