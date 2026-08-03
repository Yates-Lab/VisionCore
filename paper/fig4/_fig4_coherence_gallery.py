#!/usr/bin/env python3
"""D's lower-left gallery: what different levels of local edge coherence
actually look like as image crops.

Reads the cache built by build_coherence_gallery_cache.py (one real
BackImage crop per COHERENCE_ORDER bin, picked closest to that bin's
midpoint coherence value) and draws them as a small labeled row of
thumbnails, each with the same dashed local-contour aperture circle D's own
crop shows. Uses the same green low-to-high coherence sequence as panel I's
coherence-bin legend (panel_f_unwrapped_edge_coherence.COLORS), keeping the
low-SF/high-SF blue/orange palette reserved for SF population identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import panel_f_unwrapped_edge_coherence as panel_h  # noqa: E402

CACHE_NPZ = _paths.COHERENCE_GALLERY_NPZ
# Same low -> high green sequence as panel I's coherence-bin legend.
BIN_COLORS = panel_h.COLORS
CONTOUR_WINDOW = "#E6A700"
GRAY = "#6B6F75"
INK = "#111111"


def load_gallery(cache_npz: Path = CACHE_NPZ) -> dict | None:
    if not cache_npz.exists():
        return None
    with np.load(cache_npz, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.files}


def draw_gallery(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    gap: float,
    header_y: float,
    header_text: str | None = "local edge coherence",
    gallery: dict | None = None,
) -> bool:
    """Draw the coherence-example thumbnail row into ``ax``'s data coords.

    Returns True if real thumbnails were drawn, False if the cache was
    unavailable (caller should fall back to a placeholder).
    """
    gallery = load_gallery() if gallery is None else gallery
    if gallery is None:
        return False

    patches_arr = gallery["patches"]
    coherence_values = gallery["coherence_values"]
    radius_px = gallery["radius_px"]
    bin_labels = [str(v) for v in gallery["bin_labels"]]
    crop_size_px = int(np.asarray(gallery["crop_size_px"]).reshape(-1)[0])
    n = patches_arr.shape[0]

    if header_text:
        ax.text(x0, header_y, header_text, fontsize=7.0, color=GRAY, ha="left")

    thumb_w = (w - (n - 1) * gap) / n
    for i in range(n):
        thumb_x = x0 + i * (thumb_w + gap)
        color = BIN_COLORS[min(i, len(BIN_COLORS) - 1)]
        thumb_ax = ax.inset_axes([thumb_x, y0, thumb_w, h], transform=ax.transData)
        thumb_ax.imshow(patches_arr[i], cmap="gray", interpolation="bicubic")
        thumb_ax.set_xlim(0, crop_size_px - 1)
        thumb_ax.set_ylim(crop_size_px - 1, 0)
        thumb_ax.set_xticks([])
        thumb_ax.set_yticks([])
        for spine in thumb_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2.2)
        center = 0.5 * (crop_size_px - 1)
        thumb_ax.add_patch(
            patches.Circle(
                (center, center),
                float(radius_px[i]),
                fill=False,
                edgecolor=CONTOUR_WINDOW,
                linewidth=1.0,
                linestyle=(0, (3.0, 2.2)),
                zorder=8,
            )
        )
        ax.text(
            thumb_x + thumb_w / 2,
            y0 - 0.016,
            bin_labels[i],
            fontsize=6.0,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
        )
        ax.text(
            thumb_x + thumb_w / 2,
            y0 - 0.016 - 0.030,
            f"coh {coherence_values[i]:.2f}",
            fontsize=5.2,
            color=GRAY,
            ha="center",
            va="top",
        )
    return True


def draw_gallery_placeholder(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    header_y: float,
    gap: float = 0.0,
    header_text: str | None = "local edge coherence",
) -> None:
    if header_text:
        ax.text(x0, header_y, header_text, fontsize=7.0, color=GRAY, ha="left")
    ax.add_patch(
        patches.Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            edgecolor="#B9BFC6",
            linewidth=0.9,
            hatch="...",
        )
    )
    ax.text(
        x0 + w / 2,
        y0 + h / 2,
        "coherence gallery\ncache not built\n(run build_coherence_gallery_cache.py)",
        fontsize=5.6,
        color=GRAY,
        ha="center",
        va="center",
    )
