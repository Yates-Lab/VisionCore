"""Retinal stimulus movie generation.

This module owns step 3 of the production pipeline: visual MP4 checks of the
actual 151x151 retinal inputs sent to the model under real, stabilized,
pyramid-phase-scrambled, and spatial-frequency-band conditions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .image_selection import PYRAMID_HEIGHT, PYRAMID_ORDER, SF_BANDS_4
from .retinal_examples import TraceExample, pyramid_local_image_controls, retinal_movie_from_image_trace


CONDITION_LABELS = {
    "real": "real",
    "stabilized": "stabilized",
    "pyramid_phase_scrambled": "phase shuffled",
    "sf_low": "low SF",
    "sf_mid_low": "mid-low SF",
    "sf_mid_high": "mid-high SF",
    "sf_high": "high SF",
}


def stimulus_condition_movies(
    image: np.ndarray,
    trace: np.ndarray,
    *,
    t_max: int,
    seed: int,
    crop_center_offset_px: tuple[float, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Render retinal movies for real, stabilized, phase, and SF conditions."""
    control_images, audits = pyramid_local_image_controls(
        image,
        trace,
        np.random.default_rng(seed),
        crop_center_offset_px=crop_center_offset_px,
        height=PYRAMID_HEIGHT,
        order=PYRAMID_ORDER,
        sf_bands=SF_BANDS_4,
    )
    stable_trace = np.repeat(np.mean(trace[:t_max], axis=0, keepdims=True), t_max, axis=0).astype(np.float32)
    movies: dict[str, np.ndarray] = {
        "real": retinal_movie_from_image_trace(
            image,
            trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        ),
        "stabilized": retinal_movie_from_image_trace(
            image,
            stable_trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        ),
    }
    for condition, control_image in control_images.items():
        movies[condition] = retinal_movie_from_image_trace(
            control_image,
            trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
    return movies, control_images, audits


def _movie_scale_limits(movies: dict[str, np.ndarray], condition_order: tuple[str, ...]) -> tuple[float, float]:
    vals = np.concatenate([
        np.asarray(movies[key], dtype=np.float32).ravel()[:: max(1, movies[key].size // 50000)]
        for key in condition_order
    ])
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_stimulus_comparison_mp4(
    path: Path,
    movies: dict[str, np.ndarray],
    *,
    condition_order: tuple[str, ...],
    title: str,
    fps: int,
) -> None:
    """Write a side-by-side retinal movie with a shared grayscale scale."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _movie_scale_limits(movies, condition_order)
    n_frames = min(int(movies[key].shape[0]) for key in condition_order)
    fig, axs = plt.subplots(1, len(condition_order), figsize=(4.0 * len(condition_order), 4.48), squeeze=False)
    axs = axs[0]
    ims = []
    for ax, condition in zip(axs, condition_order, strict=True):
        im = ax.imshow(
            movies[condition][0],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(CONDITION_LABELS.get(condition, condition), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.84, wspace=0.05)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in range(n_frames):
            for im, condition in zip(ims, condition_order, strict=True):
                im.set_data(movies[condition][frame])
            fig.suptitle(f"{title} - frame {frame}/{n_frames - 1}", fontsize=12)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)


def plot_movie_audit(movie_paths: list[Path], path: Path) -> None:
    """Save a compact thumbnail audit of generated MP4s."""
    import imageio.v2 as imageio

    if not movie_paths:
        return
    frame_fracs = (0.0, 0.5, 0.95)
    fig, axs = plt.subplots(len(movie_paths), len(frame_fracs), figsize=(12.0, 3.0 * len(movie_paths)))
    axs = np.asarray(axs).reshape(len(movie_paths), len(frame_fracs))
    for r, movie_path in enumerate(movie_paths):
        reader = imageio.get_reader(movie_path)
        n_frames = reader.count_frames()
        label = movie_path.stem.replace("stimulus_", "")
        for c, frac in enumerate(frame_fracs):
            idx = min(n_frames - 1, int(round(frac * (n_frames - 1))))
            axs[r, c].imshow(reader.get_data(idx))
            axs[r, c].set_title(f"{label}\nframe {idx}", fontsize=8)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
        reader.close()
    fig.suptitle("Retinal stimulus movie audit frames", fontsize=12)
    fig.subplots_adjust(hspace=0.55, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_representative_stimulus_movies(
    *,
    examples: list[TraceExample],
    image_by_index: dict[int, np.ndarray],
    crop_rows: list[dict[str, Any]],
    movie_dir: Path,
    figure_dir: Path,
    seed: int,
    t_max: int,
    fps: int,
    max_examples_per_kind: int,
) -> list[str]:
    """Render a few fixation and microsaccade movies for visual review."""
    movie_paths: list[str] = []
    selected_examples: list[TraceExample] = []
    for kind in ("fixation", "microsaccade"):
        selected_examples.extend([ex for ex in examples if ex.kind == kind][:max_examples_per_kind])
    selected_crops = crop_rows[: max(1, min(len(crop_rows), max_examples_per_kind))]
    for example in selected_examples:
        for crop in selected_crops:
            image_index = int(crop["image_index"])
            image = image_by_index[image_index]
            crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
            movies, _control_images, _audits = stimulus_condition_movies(
                image,
                example.trace,
                t_max=t_max,
                seed=seed + image_index * 1009 + int(crop["crop_rank"]) * 37,
                crop_center_offset_px=crop_offset,
            )
            base = f"{example.example_id}_image{image_index:02d}_crop{int(crop['crop_rank']):02d}"
            phase_path = movie_dir / f"stimulus_phase_{base}.mp4"
            save_stimulus_comparison_mp4(
                phase_path,
                movies,
                condition_order=("real", "stabilized", "pyramid_phase_scrambled"),
                title=f"{example.kind} {example.example_id} image {image_index} crop {crop['crop_rank']}: phase control",
                fps=fps,
            )
            movie_paths.append(str(phase_path))
            sf_path = movie_dir / f"stimulus_sf_bands_{base}.mp4"
            save_stimulus_comparison_mp4(
                sf_path,
                movies,
                condition_order=SF_BANDS_4,
                title=f"{example.kind} {example.example_id} image {image_index} crop {crop['crop_rank']}: SF bands",
                fps=fps,
            )
            movie_paths.append(str(sf_path))
    plot_movie_audit([Path(p) for p in movie_paths], figure_dir / "03_retinal_stimulus_movie_audit.pdf")
    return movie_paths
