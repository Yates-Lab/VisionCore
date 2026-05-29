"""Natural-image selection and pyramid crop-hotspot visualization.

This module owns step 2 of the cleaned analysis:

1. Load every available natural image.
2. Reconstruct each image through a four-level complex steerable pyramid.
3. Score candidate crop centers by local contrast energy in the two middle
   spatial-frequency bands.
4. Save human-readable QC sheets and a CSV of crop-center offsets used by the
   retinal movie and information analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .common import OUT_SIZE, PPD
from .io_utils import write_csv
from .retinal_examples import (
    _copy_pyramid_coeffs,
    _padded_even_patch,
    _patch_to_tensor,
    _phase_scramble_pyramid_coeffs,
    _reconstruct_pyramid_patch,
    _selected_band_coeffs,
    _steerable_pyramid,
    model_crop_centers_px,
    pyramid_local_image_controls,
)
from .stimuli import load_natural_images


PYRAMID_HEIGHT = 4
PYRAMID_ORDER = 3
SF_BANDS_4 = ("sf_low", "sf_mid_low", "sf_mid_high", "sf_high")
MIDDLE_ENERGY_BANDS = ("sf_mid_low", "sf_mid_high")


@dataclass(frozen=True)
class ImageCrop:
    """One selected source-image crop center, represented as a model offset."""

    image_index: int
    crop_rank: int
    center_x_px: float
    center_y_px: float
    offset_x_px: float
    offset_y_px: float
    energy_value: float
    energy_percentile: float
    trace_margin_x_px: float = 0.0
    trace_margin_y_px: float = 0.0


def _full_pyramid_coeffs(image: np.ndarray):
    """Return full-image pyramid coeffs and padding metadata."""
    patch, padding = _padded_even_patch(np.asarray(image, dtype=np.float32))
    pyr = _steerable_pyramid(patch.shape, height=PYRAMID_HEIGHT, order=PYRAMID_ORDER)
    coeffs = pyr(_patch_to_tensor(patch))
    return pyr, coeffs, padding


def pyramid_reconstruction(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct an image through the analysis pyramid and return residual."""
    pyr, coeffs, padding = _full_pyramid_coeffs(image)
    recon = _reconstruct_pyramid_patch(pyr, _copy_pyramid_coeffs(coeffs), padding)
    residual = np.asarray(image, dtype=np.float32) - recon.astype(np.float32)
    return recon.astype(np.float32), residual.astype(np.float32)


def pyramid_band_image(image: np.ndarray, band: str) -> np.ndarray:
    """Reconstruct one spatial-frequency band from the four-level pyramid."""
    pyr, coeffs, padding = _full_pyramid_coeffs(image)
    component = _reconstruct_pyramid_patch(pyr, _selected_band_coeffs(coeffs, band), padding)
    if band == "sf_low":
        return component.astype(np.float32)
    return (component + float(np.mean(image))).astype(np.float32)


def middle_band_energy_map(image: np.ndarray, *, smooth_sigma_px: float | None = None) -> np.ndarray:
    """Local contrast-energy map from the two middle pyramid bands.

    The smoothing scale is deliberately tied to the model crop.  A hotspot is
    therefore a region likely to expose many 151x151 retinal crops to useful
    middle-band structure, not a single-pixel edge detector.
    """
    if smooth_sigma_px is None:
        smooth_sigma_px = max(2.0, float(OUT_SIZE[0]) / 8.0)
    pyr, coeffs, padding = _full_pyramid_coeffs(image)
    energy = np.zeros_like(np.asarray(image, dtype=np.float32), dtype=np.float32)
    for band in MIDDLE_ENERGY_BANDS:
        component = _reconstruct_pyramid_patch(pyr, _selected_band_coeffs(coeffs, band), padding)
        component = component - float(np.mean(component))
        energy += (component * component).astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        energy = gaussian_filter(energy, sigma=float(smooth_sigma_px), mode="nearest")
    except Exception:
        pass
    return energy.astype(np.float32)


def _trace_padding_margins_px(
    traces: Sequence[np.ndarray] | None,
    *,
    ppd: float = PPD,
) -> tuple[float, float]:
    """Return extra x/y crop margins needed by the selected eye traces.

    Crop selection scores a static source image, but the retinal movie samples
    a moving 151x151 footprint.  A hotspot must therefore be far enough from
    the image edge for the full trace excursion, otherwise the model sees
    artificial edge padding in the first or last movie frames.
    """
    if not traces:
        return 0.0, 0.0
    dx_values: list[np.ndarray] = []
    dy_values: list[np.ndarray] = []
    for trace in traces:
        tr = np.asarray(trace, dtype=np.float32)
        if tr.size == 0:
            continue
        # Keep the same axis convention as model_crop_centers_px.
        dx_values.append(-tr[:, 1] * float(ppd))
        dy_values.append(tr[:, 0] * float(ppd))
    if not dx_values:
        return 0.0, 0.0
    dx = np.concatenate(dx_values)
    dy = np.concatenate(dy_values)
    margin_x = float(max(abs(float(np.min(dx))), abs(float(np.max(dx)))))
    margin_y = float(max(abs(float(np.min(dy))), abs(float(np.max(dy)))))
    return margin_x, margin_y


def _eligible_energy(
    energy: np.ndarray,
    *,
    trace_margin_x_px: float = 0.0,
    trace_margin_y_px: float = 0.0,
) -> np.ndarray:
    """Mask pixels that cannot hold the full eye-movement-swept model crop."""
    out = np.asarray(energy, dtype=np.float32).copy()
    h, w = out.shape
    half_h = int(np.ceil(OUT_SIZE[0] / 2.0 + float(trace_margin_y_px)))
    half_w = int(np.ceil(OUT_SIZE[1] / 2.0 + float(trace_margin_x_px)))
    out[:half_h, :] = -np.inf
    out[h - half_h :, :] = -np.inf
    out[:, :half_w] = -np.inf
    out[:, w - half_w :] = -np.inf
    return out


def select_hotspot_crops(
    image: np.ndarray,
    *,
    image_index: int,
    n_crops: int = 3,
    exclusion_radius_px: int | None = None,
    trace_margin_x_px: float = 0.0,
    trace_margin_y_px: float = 0.0,
) -> list[ImageCrop]:
    """Select non-overlapping crop centers with high middle-band energy."""
    if exclusion_radius_px is None:
        exclusion_radius_px = int(OUT_SIZE[0])
    energy = middle_band_energy_map(image)
    work = _eligible_energy(
        energy,
        trace_margin_x_px=trace_margin_x_px,
        trace_margin_y_px=trace_margin_y_px,
    )
    h, w = work.shape
    center_x0 = (w - 1) / 2.0
    center_y0 = (h - 1) / 2.0
    finite_energy = energy[np.isfinite(work)]
    crops: list[ImageCrop] = []
    for rank in range(int(n_crops)):
        if not np.any(np.isfinite(work)):
            break
        flat_idx = int(np.nanargmax(work))
        y, x = np.unravel_index(flat_idx, work.shape)
        value = float(energy[y, x])
        percentile = float(100.0 * np.mean(finite_energy <= value)) if finite_energy.size else float("nan")
        crops.append(ImageCrop(
            image_index=int(image_index),
            crop_rank=rank,
            center_x_px=float(x),
            center_y_px=float(y),
            offset_x_px=float(x - center_x0),
            offset_y_px=float(y - center_y0),
            energy_value=value,
            energy_percentile=percentile,
            trace_margin_x_px=float(trace_margin_x_px),
            trace_margin_y_px=float(trace_margin_y_px),
        ))
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= float(exclusion_radius_px) ** 2
        work[mask] = -np.inf
    return crops


def local_pyramid_phase_image(
    image: np.ndarray,
    crop: ImageCrop,
    *,
    seed: int,
    t_max: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Phase-scramble the source ROI sampled by a zero-centered trace at crop."""
    trace = np.zeros((int(t_max), 2), dtype=np.float32)
    controls, audits = pyramid_local_image_controls(
        image,
        trace,
        np.random.default_rng(seed + int(crop.image_index) * 1009 + int(crop.crop_rank)),
        crop_center_offset_px=(crop.offset_x_px, crop.offset_y_px),
        height=PYRAMID_HEIGHT,
        order=PYRAMID_ORDER,
        sf_bands=SF_BANDS_4,
    )
    audit = next(row for row in audits if row["control"] == "pyramid_phase_scrambled")
    return controls["pyramid_phase_scrambled"], dict(audit)


def _image_panel_limits(image: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(image, dtype=np.float32), [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _draw_crop_boxes(ax, crops: list[ImageCrop], *, color: str = "#d62728") -> None:
    crop_h, crop_w = int(OUT_SIZE[0]), int(OUT_SIZE[1])
    for crop in crops:
        ax.add_patch(Rectangle(
            (crop.center_x_px - crop_w / 2.0, crop.center_y_px - crop_h / 2.0),
            crop_w,
            crop_h,
            fill=False,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.95,
        ))
        ax.text(
            crop.center_x_px,
            crop.center_y_px,
            str(crop.crop_rank + 1),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        )


def write_image_selection_qc(
    images: list[tuple[int, np.ndarray]],
    crops_by_image: dict[int, list[ImageCrop]],
    figure_dir: Path,
    *,
    seed: int,
    t_max: int,
    rows_per_page: int = 10,
) -> list[str]:
    """Save paged QC figures for all images and selected crop hotspots."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    for page_start in range(0, len(images), rows_per_page):
        page = images[page_start : page_start + rows_per_page]
        fig, axs = plt.subplots(len(page), 5, figsize=(16.0, 3.1 * len(page)), squeeze=False)
        for row_i, (image_index, image) in enumerate(page):
            crops = crops_by_image[image_index]
            recon, residual = pyramid_reconstruction(image)
            phase_img, phase_audit = local_pyramid_phase_image(
                image,
                crops[0],
                seed=seed,
                t_max=t_max,
            )
            energy = middle_band_energy_map(image)
            vmin, vmax = _image_panel_limits(image)
            panels = (
                (image, "original", "gray", vmin, vmax),
                (recon, "4-layer pyramid recon", "gray", vmin, vmax),
                (residual, "recon residual", "coolwarm", None, None),
                (phase_img, "local pyramid phase scramble", "gray", vmin, vmax),
                (energy, "middle-band energy + crops", "magma", None, None),
            )
            for ax, (panel, title, cmap, lo, hi) in zip(axs[row_i], panels, strict=True):
                if lo is None or hi is None:
                    ax.imshow(panel, cmap=cmap, interpolation="nearest")
                else:
                    ax.imshow(panel, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_title(f"image {image_index}: {title}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            _draw_crop_boxes(axs[row_i, 0], crops)
            _draw_crop_boxes(axs[row_i, 3], [crops[0]], color="#1f77b4")
            _draw_crop_boxes(axs[row_i, 4], crops, color="#ffffff")
            x0, x1 = int(phase_audit["roi_x0"]), int(phase_audit["roi_x1"])
            y0, y1 = int(phase_audit["roi_y0"]), int(phase_audit["roi_y1"])
            axs[row_i, 3].add_patch(Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#d62728",
                linewidth=1.0,
            ))
        fig.subplots_adjust(hspace=0.35, wspace=0.05, top=0.98)
        path = figure_dir / f"02_image_selection_page_{page_start // rows_per_page + 1:02d}.pdf"
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figure_paths.append(str(path))
    return figure_paths


def select_image_crops(
    *,
    image_indices: tuple[int, ...] | None,
    n_crops_per_image: int,
    figure_dir: Path | None = None,
    metadata_path: Path | None = None,
    seed: int = 0,
    t_max: int = 128,
    trace_arrays: Sequence[np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], list[ImageCrop], list[str]]:
    """Run image selection and optionally write the QC artifacts."""
    if image_indices is None:
        from scripts.mcfarland_sim import get_fixrsvp_stack

        n_total = int(get_fixrsvp_stack(frames_per_im=1, prefix="nat").shape[0])
        image_indices = tuple(range(n_total))
    loaded = load_natural_images(len(image_indices), indices=tuple(int(v) for v in image_indices))
    images = [(int(spec.image_index), image) for spec, image in loaded if spec.image_index is not None]
    image_by_index = {image_index: image for image_index, image in images}
    trace_margin_x_px, trace_margin_y_px = _trace_padding_margins_px(trace_arrays)
    all_crops: list[ImageCrop] = []
    crops_by_image: dict[int, list[ImageCrop]] = {}
    for image_index, image in images:
        crops = select_hotspot_crops(
            image,
            image_index=image_index,
            n_crops=int(n_crops_per_image),
            trace_margin_x_px=trace_margin_x_px,
            trace_margin_y_px=trace_margin_y_px,
        )
        crops_by_image[image_index] = crops
        all_crops.extend(crops)
    figures: list[str] = []
    if metadata_path is not None:
        write_csv([crop.__dict__ for crop in all_crops], metadata_path)
    if figure_dir is not None:
        figures = write_image_selection_qc(
            images,
            crops_by_image,
            figure_dir,
            seed=seed,
            t_max=t_max,
        )
    return image_by_index, all_crops, figures


def crop_offset_specs(crops: list[ImageCrop]) -> tuple[str, ...]:
    """Serialize crop offsets for CLI-compatible metadata."""
    return tuple(
        f"{crop.image_index}:{crop.crop_rank}:{crop.offset_x_px:.3f},{crop.offset_y_px:.3f}"
        for crop in crops
    )


def crop_rows(crops: list[ImageCrop]) -> list[dict[str, Any]]:
    return [crop.__dict__ for crop in crops]
