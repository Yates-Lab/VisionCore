#!/usr/bin/env python3
"""Build a compact M66/native-240 eye-motion mechanism preview.

The response-map panel uses the already frozen M66 endpoint-stabilized
schematic cache.  The retinal spectra are recomputed from 80 consecutive raw
240-Hz eye samples and the retained natural-image patch; no temporal samples
are interpolated.  The final panel compares those spectra with the complete
60-frame M66 stem-filter spectra.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/motion_mechanism_preview"
DEFAULT_VIS = Path(
    "/home/jake/.codex/visualizations/2026/08/14/"
    "019ffe32-dd4b-7ab1-ba87-15e22155ada2/eye-motion-mechanism.html"
)
MAP_CACHE = (
    ROOT
    / "outputs/dekel240_paper/m66_final_snapshot/schematic_maps_m66/cache/"
    "schematic_rr100_final_maps.npz"
)
MAP_METRICS = (
    ROOT
    / "outputs/dekel240_paper/m66_final_snapshot/schematic_maps_m66/"
    "schematic_rr100_final_map_unit_metrics.csv"
)
SOURCE_WINDOWS = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/"
    "backimage_image_fem_windows.csv"
)
PATCH_PATH = (
    ROOT
    / "outputs/dekel240_paper/m66_final_snapshot/unit_maps_m66/cache/selected_patch.npy"
)
STEM_SPECTRA = (
    ROOT
    / "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "panel_e_m66_stem_sftf_arrays.npz"
)
DATASET = Path(
    "/mnt/ssd/YatesMarmoV1/processed/Allen_2022-04-01/datasets/backimage.dset"
)

PPD = 37.50476617
RATE_HZ = 240.0
SOURCE_START = 67544
SOURCE_STOP = 67672
N_NATIVE = 80
N_SPECTRAL_TRACES = 24
TEMPORAL_FFT_SIZE = 256
SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0])
UNITS = (63, 93, 11)
BRANCHES = ("base", "auxiliary", "residual")
BRANCH_LABELS = {"base": "main", "auxiliary": "auxiliary", "residual": "residual"}
COLORS = {
    0.5: "#6BAED6",
    1.0: "#2F78B7",
    2.0: "#F28E2B",
    3.0: "#C84C36",
    "base": "#2F78B7",
    "auxiliary": "#59A14F",
    "residual": "#C84C36",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--visualization", type=Path, default=DEFAULT_VIS)
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def load_native_traces() -> tuple[np.ndarray, float, list[int]]:
    from models.data.datasets import DictDataset

    dataset = DictDataset.load(str(DATASET))
    eyepos = as_numpy(dataset["eyepos"]).astype(np.float64)
    t_bins = as_numpy(dataset["t_bins"]).astype(np.float64)
    rows = pd.read_csv(SOURCE_WINDOWS).reset_index(names="source_row")
    rows = rows.loc[
        rows["session"].astype(str).eq("Allen_2022-04-01")
        & ((rows["global_stop"] - rows["global_start"]) >= N_NATIVE)
    ].copy()
    if len(rows) < N_SPECTRAL_TRACES:
        raise RuntimeError(f"Only {len(rows)} eligible native windows are available.")
    chosen = np.linspace(0, len(rows) - 1, N_SPECTRAL_TRACES).round().astype(int)
    selected = rows.iloc[chosen]
    traces = []
    selected_rows = []
    selected_starts = []
    for row in selected.itertuples(index=False):
        window_start = int(row.global_start)
        window_stop = int(row.global_stop)
        offset = (window_stop - window_start - N_NATIVE) // 2
        start = window_start + offset
        stop = start + N_NATIVE
        trace = eyepos[start:stop].copy()
        trace -= np.mean(trace, axis=0, keepdims=True)
        if trace.shape != (N_NATIVE, 2):
            raise RuntimeError(f"Expected native trace {(N_NATIVE, 2)}, got {trace.shape}.")
        traces.append(trace)
        selected_rows.append(int(row.source_row))
        selected_starts.append(start)
    dt = float(np.median(np.concatenate([np.diff(t_bins[start : start + N_NATIVE]) for start in selected_starts])))
    if not np.isclose(dt, 1.0 / RATE_HZ, rtol=0, atol=2e-5):
        raise RuntimeError(f"Raw trace is not 240 Hz: median dt={dt}.")
    return np.asarray(traces, dtype=np.float32), dt, selected_rows


def retinal_movies(patch: np.ndarray, trace: np.ndarray) -> dict[float, np.ndarray]:
    out_size = 151
    half = (out_size - 1) / 2.0
    center_y = (patch.shape[0] - 1) / 2.0
    center_x = (patch.shape[1] - 1) / 2.0
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    movies: dict[float, np.ndarray] = {}
    for scale in SCALES:
        frames = []
        for eye_x, eye_y in trace * float(scale):
            # The sign convention changes phase, not the SF x |TF| power used
            # below.  This convention matches a retinal counter-shift.
            sample_y = center_y + y - float(eye_y) * PPD
            sample_x = center_x + x + float(eye_x) * PPD
            frames.append(
                map_coordinates(
                    patch,
                    (sample_y, sample_x),
                    order=1,
                    mode="reflect",
                    prefilter=False,
                )
            )
        movies[float(scale)] = np.asarray(frames, dtype=np.float32)
    return movies


def movie_sftf_power(
    movie: np.ndarray,
    *,
    sf_edges: np.ndarray,
    tf_edges: np.ndarray,
) -> tuple[np.ndarray, float]:
    value = np.asarray(movie, dtype=np.float64)
    value -= value.mean(axis=0, keepdims=True)
    temporal_window = np.hanning(value.shape[0])[:, None, None]
    spatial_window = np.outer(np.hanning(value.shape[1]), np.hanning(value.shape[2]))[None]
    transformed = np.fft.fftn(
        value * temporal_window * spatial_window,
        s=(TEMPORAL_FFT_SIZE, value.shape[1], value.shape[2]),
        axes=(0, 1, 2),
    )
    power = np.abs(transformed) ** 2
    temporal = np.fft.fftfreq(TEMPORAL_FFT_SIZE, d=1.0 / RATE_HZ)
    spatial_y = np.fft.fftfreq(value.shape[1], d=1.0 / PPD)
    spatial_x = np.fft.fftfreq(value.shape[2], d=1.0 / PPD)
    fy, fx = np.meshgrid(spatial_y, spatial_x, indexing="ij")
    radial = np.sqrt(fx**2 + fy**2)
    positive = temporal > 0
    tf_grid = np.broadcast_to(temporal[positive, None, None], power[positive].shape)
    sf_grid = np.broadcast_to(radial[None], power[positive].shape)
    hist, _, _ = np.histogram2d(
        tf_grid.ravel(),
        sf_grid.ravel(),
        bins=(tf_edges, sf_edges),
        weights=power[positive].ravel(),
    )
    return hist, float(np.sum(power[positive]))


def normalized_distribution(value: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    return result / max(float(np.sum(result)), 1e-30)


def ssi(rate_map: np.ndarray) -> float:
    rate = np.maximum(np.asarray(rate_map, dtype=np.float64), 0.0)
    mean = float(rate.mean())
    if mean <= 0:
        return 0.0
    gain = rate / mean
    return float(np.mean(gain * np.log2(np.maximum(gain, 1e-30))))


def render_activation_maps(out_dir: Path) -> tuple[Path, dict[str, dict[str, float]]]:
    with np.load(MAP_CACHE, allow_pickle=False) as archive:
        maps = np.asarray(archive["final_maps"], dtype=np.float64)
        condition_ids = archive["condition_id"].astype(str).tolist()
    moving_idx = condition_ids.index("real_trace_final")
    stable_idx = condition_ids.index("endpoint_stabilized_final")
    metrics = pd.read_csv(MAP_METRICS).set_index("unit_index")
    configure()
    figure, axes = plt.subplots(
        len(UNITS),
        3,
        figsize=(6.6, 5.2),
        gridspec_kw={"wspace": 0.07, "hspace": 0.28},
    )
    summary: dict[str, dict[str, float]] = {}
    for row, unit in enumerate(UNITS):
        moving = maps[moving_idx, unit]
        stable = maps[stable_idx, unit]
        delta = moving - stable
        lo, hi = np.quantile(np.concatenate([moving.ravel(), stable.ravel()]), [0.01, 0.99])
        dlim = float(np.quantile(np.abs(delta), 0.995))
        for column, (image, cmap, norm) in enumerate(
            (
                (stable, "viridis", None),
                (moving, "viridis", None),
                (delta, "RdBu_r", TwoSlopeNorm(vcenter=0.0, vmin=-dlim, vmax=dlim)),
            )
        ):
            axis = axes[row, column]
            axis.imshow(
                image,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=lo if norm is None else None,
                vmax=hi if norm is None else None,
                norm=norm,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        stable_ssi = ssi(stable)
        moving_ssi = ssi(moving)
        group = str(metrics.loc[unit, "sf_group"])
        axes[row, 0].set_ylabel(f"u{unit:03d}  {group.replace('_', ' ')}", labelpad=6)
        axes[row, 1].text(
            0.5,
            -0.08,
            f"SSI {stable_ssi:.3f} → {moving_ssi:.3f} bits/spike",
            transform=axes[row, 1].transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
        )
        summary[f"u{unit:03d}"] = {
            "stable_ssi_bits_per_spike": stable_ssi,
            "moving_ssi_bits_per_spike": moving_ssi,
            "delta_ssi_bits_per_spike": moving_ssi - stable_ssi,
            "stable_mean_rate": float(np.mean(stable)),
            "moving_mean_rate": float(np.mean(moving)),
        }
    for axis, title in zip(axes[0], ("stabilized", "measured motion", "motion − stable")):
        axis.set_title(title, pad=5)
    figure.suptitle(
        "M66 response maps remain spatially coherent; motion sharpens selected peaks",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="semibold",
    )
    figure.text(
        0.99,
        0.01,
        "raw 30×30 position maps; no display smoothing",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.06, top=0.91)
    path = out_dir / "activation_maps_stable_vs_motion.png"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out_dir / f"activation_maps_stable_vs_motion.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path, summary


def render_spectra(
    out_dir: Path,
    powers: dict[float, np.ndarray],
    total_ac: dict[float, float],
    stem: dict[str, np.ndarray],
    sf_edges: np.ndarray,
    tf_edges: np.ndarray,
) -> tuple[Path, dict[str, object]]:
    sf_centers = 0.5 * (sf_edges[:-1] + sf_edges[1:])
    tf_centers = 0.5 * (tf_edges[:-1] + tf_edges[1:])
    moving_positive = np.concatenate([powers[s][powers[s] > 0] for s in SCALES[1:]])
    vmin = max(float(np.quantile(moving_positive, 0.08)), float(np.max(moving_positive)) * 1e-6)
    vmax = float(np.quantile(moving_positive, 0.997))
    model_bank = sum(
        np.asarray(stem[f"{name}_power"], dtype=np.float64)
        * float(np.asarray(stem[f"{name}_n_filters"]))
        for name in BRANCHES
    ) / sum(float(np.asarray(stem[f"{name}_n_filters"])) for name in BRANCHES)
    model_bank = normalized_distribution(model_bank)

    overlaps: dict[str, list[float]] = {name: [] for name in (*BRANCHES, "combined")}
    for scale in SCALES:
        retinal = normalized_distribution(powers[float(scale)])
        for name in BRANCHES:
            overlaps[name].append(
                float(np.sum(retinal * normalized_distribution(stem[f"{name}_power"])))
            )
        overlaps["combined"].append(float(np.sum(retinal * model_bank)))
    normalized_overlap = {
        name: np.asarray(values) / max(float(np.max(values)), 1e-30)
        for name, values in overlaps.items()
    }

    configure()
    figure = plt.figure(figsize=(9.2, 5.0))
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1.08, 1.05), hspace=0.47, wspace=0.42)
    map_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    curve_axis = figure.add_subplot(grid[0, 3])
    overlap_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    scale_axis = figure.add_subplot(grid[1, 3])

    mesh = None
    for axis, scale in zip(map_axes, (0.5, 1.0, 3.0)):
        mesh = axis.pcolormesh(
            sf_edges,
            tf_edges,
            powers[scale],
            shading="auto",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        axis.set_title(
            f"{scale:g}× motion\nAC power {total_ac[scale] / max(total_ac[1.0], 1e-30):.2f}×",
            pad=4,
        )
        axis.set_xlim(0, 12)
        axis.set_ylim(0, 120)
        axis.set_xlabel("SF (cycles/deg)")
    map_axes[0].set_ylabel("TF (Hz)")
    figure.colorbar(mesh, ax=map_axes, fraction=0.025, pad=0.02, label="temporal AC power")

    for scale in SCALES[1:]:
        marginal = powers[float(scale)].sum(axis=1)
        marginal /= max(float(marginal.sum()), 1e-30)
        curve_axis.plot(tf_centers, marginal, color=COLORS[float(scale)], lw=1.7, label=f"{scale:g}×")
    curve_axis.set_xlim(0, 120)
    curve_axis.set_xlabel("TF (Hz)")
    curve_axis.set_ylabel("fraction of AC power")
    curve_axis.set_title("Temporal marginal")
    curve_axis.legend(frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8)
    curve_axis.grid(axis="y", color="#DDDDDD", lw=0.6)

    retinal_one = normalized_distribution(powers[1.0])
    positive_product = []
    products = {}
    for name in BRANCHES:
        product = retinal_one * normalized_distribution(stem[f"{name}_power"])
        products[name] = product
        positive_product.append(product[product > 0])
    product_values = np.concatenate(positive_product)
    product_vmin = max(float(np.quantile(product_values, 0.05)), float(np.max(product_values)) * 1e-5)
    product_vmax = float(np.quantile(product_values, 0.997))
    for axis, name in zip(overlap_axes, BRANCHES):
        axis.pcolormesh(
            sf_edges,
            tf_edges,
            products[name],
            shading="auto",
            cmap="viridis",
            norm=LogNorm(vmin=product_vmin, vmax=product_vmax),
            rasterized=True,
        )
        axis.set_title(f"1× retina × {BRANCH_LABELS[name]} stem", pad=4)
        axis.set_xlim(0, 12)
        axis.set_ylim(0, 120)
        axis.set_xlabel("SF (cycles/deg)")
    overlap_axes[0].set_ylabel("TF (Hz)")

    for name in BRANCHES:
        scale_axis.plot(
            SCALES,
            normalized_overlap[name],
            marker="o",
            ms=3.5,
            lw=1.7,
            color=COLORS[name],
            label=BRANCH_LABELS[name],
        )
    scale_axis.set_xticks(SCALES)
    scale_axis.set_ylim(0, 1.08)
    scale_axis.set_xlabel("motion scale")
    scale_axis.set_ylabel("retina–stem overlap\n(normalized within branch)")
    scale_axis.set_title("Passband engagement")
    scale_axis.legend(frameon=False, handlelength=1.4)
    scale_axis.grid(axis="y", color="#DDDDDD", lw=0.6)

    figure.suptitle(
        f"Native 240-Hz retinal motion redistributes image power into M66 channels ({N_SPECTRAL_TRACES} traces)",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=10,
        fontweight="semibold",
    )
    figure.text(
        0.985,
        0.015,
        "spectral support overlap; not causal attribution",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.84)
    path = out_dir / "native240_retinal_power_and_stem_overlap.png"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out_dir / f"native240_retinal_power_and_stem_overlap.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    tf_centroids = {}
    for scale in SCALES[1:]:
        marginal = powers[float(scale)].sum(axis=1)
        tf_centroids[str(scale)] = float(np.sum(tf_centers * marginal) / max(float(marginal.sum()), 1e-30))
    return path, {
        "temporal_centroid_hz": tf_centroids,
        "temporal_ac_power_relative_to_1x": {
            str(scale): float(total_ac[float(scale)] / max(total_ac[1.0], 1e-30))
            for scale in SCALES
        },
        "normalized_retina_stem_overlap": {
            name: {str(scale): float(value) for scale, value in zip(SCALES, normalized_overlap[name])}
            for name in normalized_overlap
        },
    }


def png_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_visualization(path: Path, activation_png: Path, spectra_png: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fragment = f"""<div id="eye-motion-mechanism" style="display:grid;gap:12px;width:100%;color:var(--foreground)">
  <figure style="margin:0;display:grid;gap:6px">
    <figcaption style="font-weight:500">1. Spatial response maps</figcaption>
    <img src="{png_data_uri(activation_png)}" alt="Three example M66 units showing stabilized, measured-motion, and motion-minus-stabilized spatial response maps." style="width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:var(--card)">
  </figure>
  <figure style="margin:0;display:grid;gap:6px">
    <figcaption style="font-weight:500">2–3. Native retinal power and overlap with learned stem channels</figcaption>
    <img src="{png_data_uri(spectra_png)}" alt="Native 240 hertz retinal spatial-temporal power at several motion scales, temporal marginals, retinal-times-stem overlap maps, and engagement versus motion scale." style="width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:var(--card)">
  </figure>
</div>
"""
    path.write_text(fragment, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces, dt, selected_rows = load_native_traces()
    patch = np.load(PATCH_PATH).astype(np.float32)
    with np.load(STEM_SPECTRA, allow_pickle=False) as archive:
        stem = {key: np.asarray(archive[key]) for key in archive.files}
    sf_edges = np.asarray(stem["base_sf_edges"], dtype=np.float64)
    tf_edges = np.asarray(stem["base_tf_edges"], dtype=np.float64)
    powers: dict[float, np.ndarray] = {}
    total_ac: dict[float, float] = {}
    for trace in traces:
        movies = retinal_movies(patch, trace)
        for scale, movie in movies.items():
            trace_power, trace_total = movie_sftf_power(
                movie,
                sf_edges=sf_edges,
                tf_edges=tf_edges,
            )
            powers[scale] = powers.get(scale, np.zeros_like(trace_power)) + trace_power
            total_ac[scale] = total_ac.get(scale, 0.0) + trace_total
    for scale in powers:
        powers[scale] /= float(len(traces))
        total_ac[scale] /= float(len(traces))
    activation_png, activation_summary = render_activation_maps(args.out_dir)
    spectra_png, spectra_summary = render_spectra(
        args.out_dir,
        powers,
        total_ac,
        stem,
        sf_edges,
        tf_edges,
    )
    np.savez_compressed(
        args.out_dir / "native240_motion_mechanism_arrays.npz",
        trace_xy=traces,
        scales=SCALES,
        sf_edges=sf_edges,
        tf_edges=tf_edges,
        **{f"retinal_power_scale_{str(scale).replace('.', 'p')}": powers[float(scale)] for scale in SCALES},
    )
    report = {
        "analysis": "M66 response maps plus native-240 retinal SFxTF power and M66 stem overlap",
        "activation_map_source": str(MAP_CACHE),
        "retinal_patch_source": str(PATCH_PATH),
        "raw_dataset": str(DATASET),
        "retinal_spectrum_trace_samples": N_NATIVE,
        "n_native_traces_averaged": int(len(traces)),
        "source_window_rows": selected_rows,
        "raw_trace_median_dt_s": dt,
        "raw_trace_rate_hz": 1.0 / dt,
        "movie_duration_endpoint_span_s": (N_NATIVE - 1) * dt,
        "temporal_frequency_resolution_hz": RATE_HZ / N_NATIVE,
        "temporal_fft_display_size": TEMPORAL_FFT_SIZE,
        "retinal_power_definition": "temporal-mean-removed 80x151x151 movie, Hann window in time and space, temporal zero-padding to 256 for display only, full 3D FFT, positive temporal frequencies, radial spatial collapse",
        "activation_maps": activation_summary,
        **spectra_summary,
        "claim_boundary": "retina-times-stem spectral overlap is a support/engagement diagnostic, not causal branch attribution; activation maps are M66 (240-Hz input, 120-Hz output) pending the winning true-240 checkpoint",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_visualization(args.visualization, activation_png, spectra_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
