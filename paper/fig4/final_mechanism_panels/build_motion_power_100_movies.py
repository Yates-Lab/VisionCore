#!/usr/bin/env python3
"""Average native-240 retinal SF x TF power over 100 natural-image movies."""

from __future__ import annotations

import argparse
import base64
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from paper.fig4.final_mechanism_panels.build_motion_mechanism_preview import (
    BRANCHES,
    BRANCH_LABELS,
    COLORS,
    N_NATIVE,
    RATE_HZ,
    SCALES,
    SOURCE_WINDOWS,
    STEM_SPECTRA,
    as_numpy,
    configure,
    movie_sftf_power,
    normalized_distribution,
    retinal_movies,
)
from paper.fig4.fixation_stats.backimage_canvas import _backimage_canvas, _clip_patch


IMAGE_TABLE = (
    ROOT
    / "outputs/dekel240_paper/m66_final_snapshot/fig4_trace_bank_merged/"
    "image_feature_table.csv"
)
DATASET = Path(
    "/mnt/ssd/YatesMarmoV1/processed/Allen_2022-04-01/datasets/backimage.dset"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "motion_power_100_movies"
)
DEFAULT_VIS = Path(
    "/home/jake/.codex/visualizations/2026/08/14/"
    "019ffe32-dd4b-7ab1-ba87-15e22155ada2/"
    "eye-motion-power-100-movies.html"
)
N_MOVIES = 100
PATCH_SIZE = 540


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--visualization", type=Path, default=DEFAULT_VIS)
    parser.add_argument("--n-movies", type=int, default=N_MOVIES)
    return parser.parse_args()


def normalize_patch(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    lo, hi = np.nanpercentile(value, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_image_patches(n_movies: int) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    table = pd.read_csv(IMAGE_TABLE).sort_values("image_index").head(int(n_movies))
    patches: list[np.ndarray] = []
    ppd_values = []
    image_indices = []
    for row in table.itertuples(index=False):
        canvas, ppd, _screen_shape = _backimage_canvas(str(row.session), int(row.trial_idx))
        center = (float(row.image_patch_center_x_px), float(row.image_patch_center_y_px))
        patch = _clip_patch(canvas, center, PATCH_SIZE)
        patches.append(normalize_patch(patch))
        ppd_values.append(float(ppd))
        image_indices.append(int(row.image_index))
    return patches, np.asarray(ppd_values, dtype=np.float64), image_indices


def load_trace_bank(n_movies: int) -> tuple[np.ndarray, float, list[int]]:
    from models.data.datasets import DictDataset

    dataset = DictDataset.load(str(DATASET))
    eyepos = as_numpy(dataset["eyepos"]).astype(np.float64)
    t_bins = as_numpy(dataset["t_bins"]).astype(np.float64)
    rows = pd.read_csv(SOURCE_WINDOWS).reset_index(names="source_row")
    rows = rows.loc[
        rows["session"].astype(str).eq("Allen_2022-04-01")
        & ((rows["global_stop"] - rows["global_start"]) >= N_NATIVE)
    ].copy()
    if len(rows) < int(n_movies):
        raise RuntimeError(f"Only {len(rows)} eligible native trace windows are available.")
    chosen = np.linspace(0, len(rows) - 1, int(n_movies)).round().astype(int)
    selected = rows.iloc[chosen]
    traces = []
    starts = []
    source_rows = []
    for row in selected.itertuples(index=False):
        window_start = int(row.global_start)
        window_stop = int(row.global_stop)
        start = window_start + (window_stop - window_start - N_NATIVE) // 2
        stop = start + N_NATIVE
        trace = eyepos[start:stop].copy()
        trace -= np.mean(trace, axis=0, keepdims=True)
        traces.append(trace.astype(np.float32))
        starts.append(start)
        source_rows.append(int(row.source_row))
    dt = float(
        np.median(
            np.concatenate([np.diff(t_bins[start : start + N_NATIVE]) for start in starts])
        )
    )
    if not np.isclose(dt, 1.0 / RATE_HZ, rtol=0, atol=2e-5):
        raise RuntimeError(f"Raw trace bank is not 240 Hz: median dt={dt}.")
    del dataset, eyepos, t_bins
    gc.collect()
    return np.asarray(traces), dt, source_rows


def accumulate_spectra(
    patches: list[np.ndarray],
    ppd_values: np.ndarray,
    traces: np.ndarray,
    sf_edges: np.ndarray,
    tf_edges: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[float, float]]:
    powers = {float(scale): np.zeros((len(tf_edges) - 1, len(sf_edges) - 1)) for scale in SCALES}
    total_ac = {float(scale): 0.0 for scale in SCALES}
    n_movies = len(patches)
    for index, (patch, ppd, trace) in enumerate(zip(patches, ppd_values, traces, strict=True)):
        # The helper uses the canonical PPD constant.  BackImage sessions use
        # the same display geometry; fail rather than silently mix grids.
        if not np.isclose(float(ppd), 37.50476617, rtol=0, atol=0.02):
            raise RuntimeError(f"Unexpected BackImage pixels/degree {ppd}.")
        movies = retinal_movies(patch, trace)
        for scale, movie in movies.items():
            spectrum, ac = movie_sftf_power(movie, sf_edges=sf_edges, tf_edges=tf_edges)
            powers[float(scale)] += spectrum
            total_ac[float(scale)] += ac
        if (index + 1) % 10 == 0:
            print(f"retinal spectra {index + 1}/{n_movies}", flush=True)
    for scale in SCALES:
        powers[float(scale)] /= float(n_movies)
        total_ac[float(scale)] /= float(n_movies)
    return powers, total_ac


def contour_levels(arrays: list[np.ndarray], *, low: float, high: float, n: int = 24) -> np.ndarray:
    positive = np.concatenate([value[value > 0] for value in arrays])
    lo = float(np.quantile(np.log10(positive), low))
    hi = float(np.quantile(np.log10(positive), high))
    return np.linspace(lo, hi, int(n))


def draw_contour(
    axis: plt.Axes,
    value: np.ndarray,
    sf_centers: np.ndarray,
    tf_centers: np.ndarray,
    *,
    levels: np.ndarray,
    cmap: str,
):
    floor = 10.0 ** float(levels[0])
    return axis.contourf(
        sf_centers,
        tf_centers,
        np.log10(np.maximum(value, floor)),
        levels=levels,
        cmap=cmap,
        extend="both",
        antialiased=True,
    )


def render(
    out_dir: Path,
    powers: dict[float, np.ndarray],
    total_ac: dict[float, float],
    stem: dict[str, np.ndarray],
    sf_edges: np.ndarray,
    tf_edges: np.ndarray,
    n_movies: int,
) -> tuple[Path, dict[str, object]]:
    sf_centers = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    tf_centers = 0.5 * (tf_edges[:-1] + tf_edges[1:])
    retinal_levels = contour_levels([powers[scale] for scale in (0.5, 1.0, 3.0)], low=0.03, high=0.995)

    products = {
        name: normalized_distribution(powers[1.0])
        * normalized_distribution(stem[f"{name}_power"])
        for name in BRANCHES
    }
    product_levels = contour_levels(list(products.values()), low=0.03, high=0.997)

    overlaps: dict[str, np.ndarray] = {}
    matched_drive: dict[str, np.ndarray] = {}
    for name in BRANCHES:
        values = np.asarray(
            [
                np.sum(
                    normalized_distribution(powers[float(scale)])
                    * normalized_distribution(stem[f"{name}_power"])
                )
                for scale in SCALES
            ]
        )
        overlaps[name] = values / max(float(np.max(values)), 1e-30)
        stem_distribution = normalized_distribution(stem[f"{name}_power"])
        drive = np.asarray(
            [
                np.sum(np.asarray(powers[float(scale)], dtype=np.float64) * stem_distribution)
                for scale in SCALES
            ]
        )
        matched_drive[name] = drive / max(float(drive[np.where(SCALES == 1.0)[0][0]]), 1e-30)

    configure()
    figure = plt.figure(figsize=(9.2, 5.0))
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1.08, 1.05), hspace=0.47, wspace=0.42)
    retinal_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    marginal_axis = figure.add_subplot(grid[0, 3])
    product_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    engagement_axis = figure.add_subplot(grid[1, 3])

    retinal_contour = None
    for axis, scale in zip(retinal_axes, (0.5, 1.0, 3.0)):
        retinal_contour = draw_contour(
            axis,
            powers[scale],
            sf_centers,
            tf_centers,
            levels=retinal_levels,
            cmap="magma",
        )
        axis.set_title(
            f"{scale:g}× motion\nAC power {total_ac[scale] / total_ac[1.0]:.2f}×",
            pad=4,
        )
        axis.set_xlim(0, 12)
        axis.set_ylim(0, 120)
        axis.set_xlabel("SF (cycles/deg)")
    retinal_axes[0].set_ylabel("TF (Hz)")
    colorbar = figure.colorbar(retinal_contour, ax=retinal_axes, fraction=0.025, pad=0.02)
    colorbar.set_label("log₁₀ temporal AC power")

    for scale in SCALES[1:]:
        marginal = powers[float(scale)].sum(axis=1)
        marginal /= max(float(marginal.sum()), 1e-30)
        marginal_axis.plot(
            tf_centers,
            marginal,
            color=COLORS[float(scale)],
            lw=1.8,
            label=f"{scale:g}×",
        )
    marginal_axis.set_xlim(0, 120)
    marginal_axis.set_xlabel("TF (Hz)")
    marginal_axis.set_ylabel("fraction of AC power")
    marginal_axis.set_title("Temporal marginal")
    marginal_axis.legend(frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8)
    marginal_axis.grid(axis="y", color="#DDDDDD", lw=0.6)

    for axis, name in zip(product_axes, BRANCHES):
        draw_contour(
            axis,
            products[name],
            sf_centers,
            tf_centers,
            levels=product_levels,
            cmap="viridis",
        )
        axis.set_title(f"1× retina × {BRANCH_LABELS[name]} stem", pad=4)
        axis.set_xlim(0, 12)
        axis.set_ylim(0, 120)
        axis.set_xlabel("SF (cycles/deg)")
    product_axes[0].set_ylabel("TF (Hz)")

    for name in BRANCHES:
        engagement_axis.plot(
            SCALES,
            matched_drive[name],
            marker="o",
            ms=3.5,
            lw=1.7,
            color=COLORS[name],
            label=BRANCH_LABELS[name],
        )
    engagement_axis.set_xticks(SCALES)
    engagement_axis.set_xlabel("motion scale")
    engagement_axis.set_ylabel("stem-matched input power\n(relative to 1×)")
    engagement_axis.set_title("First-layer drive proxy")
    engagement_axis.axhline(1.0, color="#777777", lw=0.7, zorder=0)
    engagement_axis.legend(frameon=False, handlelength=1.4)
    engagement_axis.grid(axis="y", color="#DDDDDD", lw=0.6)

    figure.suptitle(
        f"Native 240-Hz retinal power averaged across {n_movies} natural-image movies",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=10,
        fontweight="semibold",
    )
    figure.text(
        0.985,
        0.015,
        "filled contours of unsmoothed binned averages; spectral drive is not causal attribution",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.84)
    stem_name = f"native240_retinal_power_{n_movies}_movies_contours"
    path = out_dir / f"{stem_name}.png"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            out_dir / f"{stem_name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    tf_centroids = {}
    for scale in SCALES[1:]:
        marginal = powers[float(scale)].sum(axis=1)
        tf_centroids[str(scale)] = float(
            np.sum(tf_centers * marginal) / max(float(marginal.sum()), 1e-30)
        )
    return path, {
        "temporal_centroid_hz": tf_centroids,
        "temporal_ac_power_relative_to_1x": {
            str(scale): float(total_ac[float(scale)] / total_ac[1.0]) for scale in SCALES
        },
        "normalized_retina_stem_overlap": {
            name: {
                str(scale): float(value) for scale, value in zip(SCALES, overlaps[name])
            }
            for name in BRANCHES
        },
        "stem_matched_input_power_relative_to_1x": {
            name: {
                str(scale): float(value)
                for scale, value in zip(SCALES, matched_drive[name])
            }
            for name in BRANCHES
        },
    }


def write_visualization(path: Path, png_path: Path, n_movies: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    path.write_text(
        f'''<div id="eye-motion-power-{n_movies}" style="width:100%;color:var(--foreground)">
  <figure style="margin:0;display:grid;gap:6px">
    <figcaption style="font-weight:500">Retinal SF×TF power and M66 first-layer drive</figcaption>
    <img src="data:image/png;base64,{data}" alt="Native 240 hertz retinal spatial-temporal power averaged over {n_movies} natural-image movies, displayed as filled contours with temporal marginals and a spectral drive proxy for each M66 visual stem." style="width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:var(--card)">
  </figure>
</div>
''',
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patches, ppd_values, image_indices = load_image_patches(int(args.n_movies))
    traces, dt, trace_rows = load_trace_bank(int(args.n_movies))
    with np.load(STEM_SPECTRA, allow_pickle=False) as archive:
        stem = {key: np.asarray(archive[key]) for key in archive.files}
    sf_edges = np.asarray(stem["base_sf_edges"], dtype=np.float64)
    tf_edges = np.asarray(stem["base_tf_edges"], dtype=np.float64)
    powers, total_ac = accumulate_spectra(
        patches,
        ppd_values,
        traces,
        sf_edges,
        tf_edges,
    )
    png, metrics = render(
        args.out_dir,
        powers,
        total_ac,
        stem,
        sf_edges,
        tf_edges,
        int(args.n_movies),
    )
    np.savez_compressed(
        args.out_dir / f"native240_retinal_power_{int(args.n_movies)}_movies_arrays.npz",
        scales=SCALES,
        sf_edges=sf_edges,
        tf_edges=tf_edges,
        image_indices=np.asarray(image_indices),
        trace_source_rows=np.asarray(trace_rows),
        **{
            f"retinal_power_scale_{str(scale).replace('.', 'p')}": powers[float(scale)]
            for scale in SCALES
        },
    )
    report = {
        "analysis": "native-240 retinal SFxTF power averaged over distinct natural-image movies",
        "n_movies": int(args.n_movies),
        "n_distinct_image_patches": int(len(set(image_indices))),
        "n_distinct_native_eye_traces": int(len(set(trace_rows))),
        "image_table": str(IMAGE_TABLE),
        "eye_trace_dataset": str(DATASET),
        "eye_trace_sampling_rate_hz": float(1.0 / dt),
        "movie_samples": int(N_NATIVE),
        "movie_endpoint_span_s": float((N_NATIVE - 1) * dt),
        "native_frequency_resolution_hz": float(RATE_HZ / N_NATIVE),
        "image_normalization": "per-patch 0.5-to-99.5 percentile range mapped to [0,1]",
        "averaging": "one distinct selected Figure-4 image patch paired with one distinct native Allen_2022-04-01 eye trace; power accumulated before display",
        "display": "filled contours of unsmoothed SFxTF bin averages; 256-point temporal zero-padding changes display sampling but not 3-Hz native resolution",
        **metrics,
        "claim_boundary": "stem spectral overlap is descriptive support, not measured activation or causal attribution",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    write_visualization(args.visualization, png, int(args.n_movies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
