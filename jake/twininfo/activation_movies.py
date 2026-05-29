"""Activation-map movies for representative retinal examples.

This is step 4 of the production pipeline.  It shows the spatial readout maps
for nine representative units as the same retinal movie examples are played
through the digital twin.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .image_selection import PYRAMID_HEIGHT, PYRAMID_ORDER, SF_BANDS_4
from .lagcube_information import run_lag_cube_rates
from .retinal_examples import TraceExample, model_lag_cubes_from_image_trace, pyramid_local_image_controls
from .retinal_movies import CONDITION_LABELS


ACTIVATION_CONDITIONS = ("real", "stabilized", "pyramid_phase_scrambled")


def _activation_limits(rate_maps: list[np.ndarray], unit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use shared display limits so condition movies are visually comparable."""
    lows = np.zeros((len(unit_indices),), dtype=np.float32)
    highs = np.zeros((len(unit_indices),), dtype=np.float32)
    for i, unit in enumerate(unit_indices):
        samples = []
        for rate_map in rate_maps:
            vals = np.asarray(rate_map[:, unit], dtype=np.float32).ravel()
            samples.append(vals[:: max(1, vals.size // 100000)])
        pooled = np.concatenate(samples)
        lo, hi = np.percentile(pooled, [1.0, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        lows[i] = float(lo)
        highs[i] = float(hi)
    return lows, highs


def save_activation_map_mp4(
    path: Path,
    rate_map: np.ndarray,
    unit_indices: np.ndarray,
    *,
    title: str,
    fps: int,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> None:
    """Write a 3x3 grid movie of spatial activation maps."""
    import imageio.v2 as imageio

    maps = np.asarray(rate_map, dtype=np.float32)
    units = np.asarray(unit_indices, dtype=np.int64)[:9]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 3, figsize=(8.0, 8.0))
    ims = []
    for i, (ax, unit) in enumerate(zip(axs.ravel(), units, strict=True)):
        im = ax.imshow(maps[0, unit], cmap="gray", vmin=float(vmin[i]), vmax=float(vmax[i]), interpolation="nearest")
        ax.set_title(f"unit {int(unit)}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.02, hspace=0.15)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as writer:
        for frame in range(maps.shape[0]):
            for im, unit in zip(ims, units, strict=True):
                im.set_data(maps[frame, unit])
            fig.suptitle(f"{title} - frame {frame}/{maps.shape[0] - 1}", fontsize=13)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)


def plot_activation_movie_audit(movie_paths: list[Path], path: Path) -> None:
    """Save uncluttered thumbnails from activation-map MP4s."""
    import imageio.v2 as imageio

    if not movie_paths:
        return
    frame_fracs = (0.0, 0.5, 0.95)
    fig, axs = plt.subplots(len(movie_paths), len(frame_fracs), figsize=(10.0, 2.35 * len(movie_paths)))
    axs = np.asarray(axs).reshape(len(movie_paths), len(frame_fracs))
    for r, movie_path in enumerate(movie_paths):
        reader = imageio.get_reader(movie_path)
        n_frames = reader.count_frames()
        label = movie_path.stem.replace("activation_maps_", "")
        label = label.replace("_pyramid_phase_scrambled", "\nphase shuffled")
        label = label.replace("_stabilized", "\nstabilized")
        label = label.replace("_real", "\nreal")
        label = label.replace("_image", "\nimage")
        label = label.replace("_crop", " crop")
        for c, frac in enumerate(frame_fracs):
            idx = min(n_frames - 1, int(round(frac * (n_frames - 1))))
            axs[r, c].imshow(reader.get_data(idx))
            if r == 0:
                axs[r, c].set_title(f"frame {idx}", fontsize=9)
            if c == 0:
                axs[r, c].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
        reader.close()
    fig.suptitle("Activation-map movie audit frames", fontsize=12, y=0.995)
    fig.subplots_adjust(hspace=0.18, wspace=0.04, top=0.965, left=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def condition_lag_cubes(
    image: np.ndarray,
    trace: np.ndarray,
    *,
    condition: str,
    seed: int,
    t_max: int,
    crop_center_offset_px: tuple[float, float],
) -> np.ndarray:
    """Build the exact model lag cubes for one condition."""
    if condition == "real":
        return model_lag_cubes_from_image_trace(
            image,
            trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
    if condition == "stabilized":
        stable_trace = np.repeat(np.mean(trace[:t_max], axis=0, keepdims=True), t_max, axis=0).astype(np.float32)
        return model_lag_cubes_from_image_trace(
            image,
            stable_trace,
            t_max=t_max,
            crop_center_offset_px=crop_center_offset_px,
        )
    control_images, _audits = pyramid_local_image_controls(
        image,
        trace,
        np.random.default_rng(seed),
        crop_center_offset_px=crop_center_offset_px,
        height=PYRAMID_HEIGHT,
        order=PYRAMID_ORDER,
        sf_bands=SF_BANDS_4,
    )
    if condition not in control_images:
        raise ValueError(f"Unsupported activation condition: {condition}")
    return model_lag_cubes_from_image_trace(
        control_images[condition],
        trace,
        t_max=t_max,
        crop_center_offset_px=crop_center_offset_px,
    )


def make_activation_movies(
    *,
    model: Any,
    population: Any,
    device: Any,
    examples: list[TraceExample],
    image_by_index: dict[int, np.ndarray],
    crop_rows: list[dict[str, Any]],
    unit_scores: dict[tuple[str, int, int, str], np.ndarray],
    movie_dir: Path,
    figure_dir: Path,
    seed: int,
    t_max: int,
    batch_size: int,
    fps: int,
    max_examples_per_kind: int,
) -> list[str]:
    """Render activation-map movies for representative traces and crops."""
    selected_examples: list[TraceExample] = []
    for kind in ("fixation", "microsaccade"):
        selected_examples.extend([ex for ex in examples if ex.kind == kind][:max_examples_per_kind])
    selected_crops = crop_rows[: max(1, min(len(crop_rows), max_examples_per_kind))]
    movie_paths: list[str] = []
    for example in selected_examples:
        for crop in selected_crops:
            image_index = int(crop["image_index"])
            crop_rank = int(crop["crop_rank"])
            image = image_by_index[image_index]
            crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))
            rate_maps: dict[str, np.ndarray] = {}
            for condition in ACTIVATION_CONDITIONS:
                cubes = condition_lag_cubes(
                    image,
                    example.trace,
                    condition=condition,
                    seed=seed + image_index * 1009 + crop_rank * 37,
                    t_max=t_max,
                    crop_center_offset_px=crop_offset,
                )
                _rates, rate_map = run_lag_cube_rates(
                    model,
                    population,
                    device,
                    cubes,
                    batch_size=batch_size,
                    return_rate_map=True,
                )
                if rate_map is not None:
                    rate_maps[condition] = rate_map
            if "real" not in rate_maps:
                continue
            scores = unit_scores.get((example.example_id, image_index, crop_rank, "real"))
            if scores is None:
                scores = np.nanvar(rate_maps["real"], axis=(0, 2, 3))
            units = np.argsort(scores)[-9:][::-1]
            vmin, vmax = _activation_limits(list(rate_maps.values()), units)
            for condition, rate_map in rate_maps.items():
                path = movie_dir / (
                    f"activation_maps_{example.example_id}_image{image_index:02d}_"
                    f"crop{crop_rank:02d}_{condition}.mp4"
                )
                save_activation_map_mp4(
                    path,
                    rate_map,
                    units,
                    title=(
                        f"{example.kind} {example.example_id} image {image_index} "
                        f"crop {crop_rank} {CONDITION_LABELS.get(condition, condition)}"
                    ),
                    fps=fps,
                    vmin=vmin,
                    vmax=vmax,
                )
                movie_paths.append(str(path))
    plot_activation_movie_audit([Path(p) for p in movie_paths], figure_dir / "04_activation_movie_audit.pdf")
    return movie_paths
