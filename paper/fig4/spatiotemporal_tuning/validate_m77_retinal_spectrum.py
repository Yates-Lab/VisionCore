#!/usr/bin/env python3
"""Validate complete-trajectory spectra against exact M77 retinal rendering.

This is a signal-processing claim gate, not the primary power estimator.  It
compares the ideal Fourier translation identity, image FFT times the temporal
spectrum of exp(-i 2π k·X(t)), with direct movies rendered through the exact
151-pixel scorer geometry.  The primary analysis remains direct rendered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import (
    mode_to_grid_matrix,
    render_movies,
)
from paper.fig4.spatiotemporal_tuning.analyze_image_specific_joint_engagement import trajectory_phase_spectra
from paper.fig4.spatiotemporal_tuning.audit_m77_nonlinear_sharpening import evenly_spaced_rows
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import frequency_grid
from paper.fig4.spatiotemporal_tuning.run_m77_retinal_causal_chain import (
    DEFAULT_CHAIN,
    DEFAULT_MATRIX,
    load_tuning_tensors,
)
from paper.fig4.spatiotemporal_tuning.validate_trajectory_phase_spectrum import (
    distribution_metrics,
    folded_multitaper_power,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_MATRIX / "image_feature_table.csv")
    parser.add_argument("--trace-bank", type=Path, default=DEFAULT_CHAIN / "fixation_bank")
    parser.add_argument("--tuning-table", type=Path, default=DEFAULT_CHAIN / "dense_tuning/frequency_tuning_grouped.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-kind", choices=("filtered", "raw"), default="filtered")
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--n-traces", type=int, default=8)
    parser.add_argument("--analysis-samples", type=int, default=240)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-cosine-gate", type=float, default=0.97)
    parser.add_argument("--pairwise-median-cosine-gate", type=float, default=0.85)
    return parser.parse_args()


def direct_mode_power(movie: np.ndarray, flat_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = (np.asarray(movie, dtype=np.float64) - 127.0) / 255.0
    coefficient = np.fft.fft2(value, axes=(-2, -1), norm="ortho")
    selected = coefficient.reshape(len(value), -1)[:, flat_index].T
    return folded_multitaper_power(selected, 240.0)


def cube_from_modes(
    power: np.ndarray,
    distributor,
    n_spatial: int,
    n_orientation: int,
) -> np.ndarray:
    flat = distributor.T @ power
    return np.asarray(flat).reshape(n_spatial, n_orientation, power.shape[1]).transpose(0, 2, 1)


def render_validation_figure(
    path: Path,
    *,
    spatial: np.ndarray,
    temporal: np.ndarray,
    actual_cube: np.ndarray,
    ideal_cube: np.ndarray,
    pairs: pd.DataFrame,
) -> None:
    actual = actual_cube.sum(axis=-1)
    ideal = ideal_cube.sum(axis=-1)
    actual /= max(float(actual.sum()), EPS)
    ideal /= max(float(ideal.sum()), EPS)
    positive = np.concatenate((actual[actual > 0], ideal[ideal > 0]))
    floor = max(float(np.quantile(positive, 0.02)), EPS)
    ceiling = float(np.quantile(positive, 0.995))
    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.2), constrained_layout=True)
    contour = None
    for axis, value, title in (
        (axes[0], actual, "A  Exact rendered movies"),
        (axes[1], ideal, "B  Image FFT × full-trajectory phase"),
    ):
        contour = axis.contourf(
            spatial,
            temporal,
            np.log10(np.maximum(value.T, floor)),
            levels=np.linspace(np.log10(floor), np.log10(ceiling), 13),
            cmap="magma",
            extend="both",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("spatial frequency (cycles/degree)")
        axis.set_ylabel("temporal frequency (Hz)")
        axis.set_title(title, loc="left", fontweight="semibold")
    if contour is not None:
        figure.colorbar(contour, ax=axes[:2], shrink=0.78, label="log10 fraction of dynamic power")
    ratio = np.log2(np.maximum(actual, EPS) / np.maximum(ideal, EPS))
    image = axes[2].contourf(spatial, temporal, ratio.T, levels=np.linspace(-3, 3, 13), cmap="RdBu_r", extend="both")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log", base=2)
    axes[2].set_xlabel("spatial frequency (cycles/degree)")
    axes[2].set_ylabel("temporal frequency (Hz)")
    axes[2].set_title("C  Rendered / ideal", loc="left", fontweight="semibold")
    figure.colorbar(image, ax=axes[2], shrink=0.78, label="log2 power ratio")
    axes[3].scatter(np.arange(len(pairs)), pairs.distribution_cosine, s=24, alpha=0.75, color="#2E7D49")
    median = float(pairs.distribution_cosine.median())
    axes[3].axhline(median, color="#2E7D49", lw=1.7)
    axes[3].axhline(0.85, color="0.45", lw=0.9, ls="--")
    axes[3].set_xlabel("image–trace pair")
    axes[3].set_ylabel("spectral cosine similarity")
    axes[3].set_ylim(0, 1.02)
    axes[3].set_title(f"D  Pairwise validation\nmedian={median:.3f}", loc="left", fontweight="semibold")
    figure.suptitle("Complete trajectory spectra reproduce direct retinal rendering", fontsize=14, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    images = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    trace_table = pd.read_csv(args.trace_bank / "trace_table.csv").sort_values("trace_index").reset_index(drop=True)
    traces = np.load(args.trace_bank / f"trace_xy_{args.trace_kind}.npy", mmap_mode="r")
    image_rows = evenly_spaced_rows(len(images), args.n_images)
    trace_rows = evenly_spaced_rows(len(trace_table), args.n_traces)
    selected_traces = np.asarray(traces[trace_rows, -args.analysis_samples :], dtype=np.float32)
    tuning = load_tuning_tensors(args.tuning_table)
    spatial = np.asarray(tuning["spatial_cpd"], dtype=float)
    orientations = np.asarray(tuning["orientation_deg"], dtype=float)
    grid = frequency_grid()
    kxy = np.asarray(grid["kxy"], dtype=float)
    flat_index = np.asarray(grid["flat_index"], dtype=int)
    distributor, resolved_modes = mode_to_grid_matrix(kxy, spatial, orientations)

    phase_power = []
    temporal = None
    for ordinal, trace in enumerate(selected_traces):
        frequency, value = trajectory_phase_spectra(kxy, trace[None], 240.0, chunk_size=512)
        if temporal is None:
            temporal = frequency
        elif not np.array_equal(temporal, frequency):
            raise RuntimeError("trajectory temporal grids differ")
        phase_power.append(value)
        print(f"trajectory validation trace {ordinal + 1}/{len(selected_traces)}", flush=True)
    phase_power = np.asarray(phase_power)
    temporal = np.asarray(temporal)
    valid_tf = temporal <= 96.0
    valid_mode = np.asarray(resolved_modes, dtype=bool)
    aggregate_actual_modes = np.zeros_like(phase_power[0], dtype=np.float64)
    aggregate_ideal_modes = np.zeros_like(aggregate_actual_modes)
    rows = []
    canvas_cache: dict = {}
    for image_ordinal, image_row in enumerate(image_rows):
        patch, _ = extract_patch(images.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=540)
        stabilized = render_movies(patch, np.zeros((1, 1, 2), dtype=np.float32), device=args.device)[0, 0]
        base = np.fft.fft2((stabilized - 127.0) / 255.0, norm="ortho")
        base_power = np.square(np.abs(base.ravel()[flat_index]))
        for trace_ordinal, trace in enumerate(selected_traces):
            movie = render_movies(patch, trace[None], device=args.device)[0]
            direct_hz, actual = direct_mode_power(movie, flat_index)
            if not np.array_equal(direct_hz, temporal):
                raise RuntimeError("direct and trajectory temporal grids differ")
            ideal = base_power[:, None] * phase_power[trace_ordinal]
            rows.append(
                {
                    "image_index": int(images.iloc[int(image_row)].image_index),
                    "trace_index": int(trace_table.iloc[int(trace_rows[trace_ordinal])].trace_index),
                    **distribution_metrics(actual[valid_mode][:, valid_tf], ideal[valid_mode][:, valid_tf]),
                }
            )
            aggregate_actual_modes += actual
            aggregate_ideal_modes += ideal
        print(f"direct spectrum validation image {image_ordinal + 1}/{len(image_rows)}", flush=True)
    pair_table = pd.DataFrame(rows)
    aggregate_metrics = distribution_metrics(
        aggregate_actual_modes[valid_mode][:, valid_tf],
        aggregate_ideal_modes[valid_mode][:, valid_tf],
    )
    actual_cube = cube_from_modes(aggregate_actual_modes, distributor, len(spatial), len(orientations))
    ideal_cube = cube_from_modes(aggregate_ideal_modes, distributor, len(spatial), len(orientations))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(args.out_dir / "pair_metrics.csv", index=False)
    np.savez_compressed(
        args.out_dir / "retinal_spectrum_validation.npz",
        actual_cube=actual_cube.astype(np.float32),
        ideal_cube=ideal_cube.astype(np.float32),
        spatial_cpd=spatial,
        temporal_hz=temporal,
        orientation_deg=orientations,
        image_rows=image_rows,
        trace_rows=trace_rows,
    )
    figure = args.out_dir / "m77_retinal_spectrum_validation.png"
    render_validation_figure(
        figure,
        spatial=spatial,
        temporal=temporal,
        actual_cube=actual_cube,
        ideal_cube=ideal_cube,
        pairs=pair_table,
    )
    pair_median = float(pair_table.distribution_cosine.median())
    summary = {
        "analysis": "complete-trajectory Fourier phase spectrum versus direct M77 retinal rendering",
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_pairs": int(len(pair_table)),
        "trace_kind": args.trace_kind,
        "aggregate": aggregate_metrics,
        "pairwise_median_distribution_cosine": pair_median,
        "aggregate_cosine_gate": float(args.aggregate_cosine_gate),
        "pairwise_median_cosine_gate": float(args.pairwise_median_cosine_gate),
        "aggregate_gate_pass": bool(aggregate_metrics["distribution_cosine"] >= args.aggregate_cosine_gate),
        "pairwise_gate_pass": bool(pair_median >= args.pairwise_median_cosine_gate),
        "all_claim_gates_pass": bool(
            aggregate_metrics["distribution_cosine"] >= args.aggregate_cosine_gate
            and pair_median >= args.pairwise_median_cosine_gate
        ),
        "instantaneous_velocity_used": False,
        "claim_boundary": "This validates the pure-translation spectrum against finite scorer rendering. Primary causal projections use direct-rendered movies with their declared spatial Tukey window.",
        "figure": str(figure.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
