#!/usr/bin/env python3
"""Validate the Kuang phase-carrier spectrum against exact retinal rendering.

This audit has two deliberately separate comparisons.  Periodic Fourier
translation must reproduce ``I(k)Q(k,w)`` essentially exactly.  The finite
151-pixel model scorer additionally includes bilinear resampling, crop
replacement, and padding, so its agreement is measured and bounded rather
than assumed.  Only a filtered 240-Hz fixation bank is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
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

from paper.fig4.spatiotemporal_tuning.build_rucci_ensemble_power import (  # noqa: E402
    load_filtered_fixation_bank,
    production_spectral_grid,
)
from paper.fig4.spatiotemporal_tuning.retinal_replay import (  # noqa: E402
    evenly_spaced_rows,
    render_movies,
)
from paper.fig4.spatiotemporal_tuning.spectral_power import (  # noqa: E402
    EPS,
    folded_dpss_mode_power,
    frequency_grid,
    mode_to_grid_matrix,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch  # noqa: E402
from paper.fig4.upstream.real_trace_matrix.model import PPD  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, required=True)
    parser.add_argument("--fixation-bank", type=Path, required=True)
    parser.add_argument("--tuning-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--n-traces", type=int, default=8)
    parser.add_argument("--analysis-samples", type=int, default=240)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode-chunk-size", type=int, default=512)
    parser.add_argument("--maximum-tested-tf-hz", type=float, default=96.0)
    parser.add_argument("--periodic-aggregate-cosine-gate", type=float, default=0.999)
    parser.add_argument(
        "--periodic-pairwise-median-cosine-gate", type=float, default=0.995
    )
    parser.add_argument("--finite-aggregate-cosine-gate", type=float, default=0.95)
    parser.add_argument(
        "--finite-pairwise-median-cosine-gate", type=float, default=0.85
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_metrics(actual: np.ndarray, ideal: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(ideal, dtype=np.float64).ravel()
    total_a = float(a.sum())
    total_b = float(b.sum())
    if total_a <= EPS or total_b <= EPS:
        raise ValueError("both compared spectra must contain positive dynamic power")
    probability_a = a / total_a
    probability_b = b / total_b
    gain = total_a / total_b
    return {
        "actual_over_ideal_power": gain,
        "distribution_total_variation": float(
            0.5 * np.abs(probability_a - probability_b).sum()
        ),
        "distribution_cosine": float(
            np.dot(probability_a, probability_b)
            / max(np.linalg.norm(probability_a) * np.linalg.norm(probability_b), EPS)
        ),
        "relative_l1_after_gain_match": float(
            np.abs(a - gain * b).sum() / total_a
        ),
    }


def trajectory_phase_power(
    traces_xy_deg: np.ndarray,
    kxy_cpd: np.ndarray,
    *,
    frame_rate_hz: float,
    mode_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete-order ``Q(k,w)`` for each trace and Fourier mode."""
    traces = np.asarray(traces_xy_deg, dtype=np.float64)
    kxy = np.asarray(kxy_cpd, dtype=np.float64)
    if traces.ndim != 3 or traces.shape[-1] != 2 or traces.shape[1] < 8:
        raise ValueError(f"traces must be [trace,time>=8,2], got {traces.shape}")
    if kxy.ndim != 2 or kxy.shape[1] != 2:
        raise ValueError(f"kxy must be [mode,2], got {kxy.shape}")
    output = np.empty((len(traces), len(kxy), traces.shape[1] // 2), dtype=np.float32)
    temporal_hz: np.ndarray | None = None
    for trace_index, trace in enumerate(traces):
        for start in range(0, len(kxy), int(mode_chunk_size)):
            stop = min(start + int(mode_chunk_size), len(kxy))
            phase = -2j * np.pi * (kxy[start:stop] @ trace.T)
            frequency, value = folded_dpss_mode_power(
                np.exp(phase), float(frame_rate_hz)
            )
            if temporal_hz is None:
                temporal_hz = frequency
            elif not np.array_equal(temporal_hz, frequency):
                raise RuntimeError("trajectory chunks produced different TF grids")
            output[trace_index, start:stop] = value.astype(np.float32)
        print(f"trajectory spectrum {trace_index + 1}/{len(traces)}", flush=True)
    if temporal_hz is None:
        raise RuntimeError("no trajectory spectrum was computed")
    return temporal_hz, output


def direct_mode_power(
    movie: np.ndarray, flat_index: np.ndarray, *, frame_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    value = (np.asarray(movie, dtype=np.float64) - 127.0) / 255.0
    coefficient = np.fft.fft2(value, axes=(-2, -1), norm="ortho")
    selected = coefficient.reshape(len(value), -1)[:, flat_index].T
    return folded_dpss_mode_power(selected, float(frame_rate_hz))


def periodic_translation_movie(
    stabilized_frame: np.ndarray, trace_xy_deg: np.ndarray, *, ppd: float
) -> np.ndarray:
    """Translate a square image exactly under periodic Fourier boundaries."""
    frame = np.asarray(stabilized_frame, dtype=np.float64)
    trace = np.asarray(trace_xy_deg, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[0] != frame.shape[1]:
        raise ValueError(f"stabilized_frame must be square, got {frame.shape}")
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError(f"trace_xy_deg must be [time,2], got {trace.shape}")
    axis = np.fft.fftfreq(frame.shape[0], d=1.0 / float(ppd))
    ky_image, kx = np.meshgrid(axis, axis, indexing="ij")
    ky = -ky_image
    dot = trace[:, 0, None, None] * kx[None] + trace[:, 1, None, None] * ky[None]
    coefficient = np.fft.fft2(frame, norm="ortho")
    movie = np.fft.ifft2(
        coefficient[None] * np.exp(-2j * np.pi * dot),
        axes=(-2, -1),
        norm="ortho",
    )
    imaginary_fraction = float(
        np.linalg.norm(movie.imag) / max(np.linalg.norm(movie.real), EPS)
    )
    if imaginary_fraction > 1e-10:
        raise RuntimeError(
            f"periodic translation lost real-valued Fourier symmetry: {imaginary_fraction}"
        )
    return movie.real


def cube_from_modes(
    power: np.ndarray, distributor, n_spatial: int, n_orientation: int
) -> np.ndarray:
    flat = distributor.T @ power
    return np.asarray(flat).reshape(
        n_spatial, n_orientation, power.shape[1]
    ).transpose(0, 2, 1)


def render_validation_figure(
    path: Path,
    *,
    spatial: np.ndarray,
    temporal: np.ndarray,
    actual_cube: np.ndarray,
    periodic_cube: np.ndarray,
    ideal_cube: np.ndarray,
    pairs: pd.DataFrame,
) -> None:
    actual = actual_cube.sum(axis=-1)
    periodic = periodic_cube.sum(axis=-1)
    ideal = ideal_cube.sum(axis=-1)
    actual /= max(float(actual.sum()), EPS)
    periodic /= max(float(periodic.sum()), EPS)
    ideal /= max(float(ideal.sum()), EPS)
    positive = np.concatenate(
        (actual[actual > 0], periodic[periodic > 0], ideal[ideal > 0])
    )
    floor = max(float(np.quantile(positive, 0.02)), EPS)
    ceiling = float(np.quantile(positive, 0.995))
    figure, axes = plt.subplots(1, 5, figsize=(20.0, 4.0), constrained_layout=True)
    contour = None
    for axis, value, title in (
        (axes[0], periodic, "periodic translation"),
        (axes[1], ideal, "image FFT × trajectory phase"),
        (axes[2], actual, "finite scorer rendering"),
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
        axis.set(xlabel="SF (cycles/deg)", ylabel="TF (Hz)", title=title)
    if contour is not None:
        figure.colorbar(
            contour, ax=list(axes[:3]), shrink=0.78, label="log10 dynamic power fraction"
        )
    ratio = np.log2(np.maximum(actual, EPS) / np.maximum(ideal, EPS))
    ratio_image = axes[3].contourf(
        spatial,
        temporal,
        ratio.T,
        levels=np.linspace(-3, 3, 13),
        cmap="RdBu_r",
        extend="both",
    )
    axes[3].set_xscale("log", base=2)
    axes[3].set_yscale("log", base=2)
    axes[3].set(xlabel="SF (cycles/deg)", ylabel="TF (Hz)", title="finite / ideal")
    figure.colorbar(ratio_image, ax=axes[3], shrink=0.78, label="log2 power ratio")
    index = np.arange(len(pairs))
    axes[4].scatter(
        index,
        pairs.periodic_distribution_cosine,
        s=18,
        alpha=0.7,
        color="#2E7D49",
        label="periodic",
    )
    axes[4].scatter(
        index,
        pairs.finite_distribution_cosine,
        s=18,
        alpha=0.7,
        color="#D55E00",
        label="finite scorer",
    )
    axes[4].set(
        xlabel="image–trace pair",
        ylabel="spectral cosine similarity",
        ylim=(0, 1.02),
        title="pairwise agreement",
    )
    axes[4].legend(frameon=False)
    figure.suptitle("Kuang translation identity and finite-aperture boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.n_images < 1 or args.n_traces < 1:
        raise ValueError("n_images and n_traces must be positive")
    manifest, all_trace_table, all_traces, all_trace_rows = load_filtered_fixation_bank(
        args.fixation_bank,
        analysis_samples=int(args.analysis_samples),
        requested_traces=0,
    )
    trace_positions = evenly_spaced_rows(
        len(all_trace_table), min(int(args.n_traces), len(all_trace_table))
    )
    trace_table = all_trace_table.iloc[trace_positions].reset_index(drop=True)
    traces = np.asarray(all_traces[trace_positions], dtype=np.float32)
    trace_rows = np.asarray(all_trace_rows[trace_positions], dtype=int)
    frame_rate_hz = float(manifest["target_rate_hz"])
    images = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    image_rows = evenly_spaced_rows(len(images), min(int(args.n_images), len(images)))
    spatial, orientations = production_spectral_grid(args.tuning_table)
    grid = frequency_grid()
    kxy = np.asarray(grid["kxy"], dtype=float)
    flat_index = np.asarray(grid["flat_index"], dtype=int)
    distributor, resolved_modes = mode_to_grid_matrix(kxy, spatial, orientations)
    temporal, phase_power = trajectory_phase_power(
        traces,
        kxy,
        frame_rate_hz=frame_rate_hz,
        mode_chunk_size=int(args.mode_chunk_size),
    )
    valid_tf = temporal <= float(args.maximum_tested_tf_hz)
    valid_mode = np.asarray(resolved_modes, dtype=bool)
    aggregate_actual = np.zeros_like(phase_power[0], dtype=np.float64)
    aggregate_periodic = np.zeros_like(aggregate_actual)
    aggregate_ideal = np.zeros_like(aggregate_actual)
    rows: list[dict[str, float | int]] = []
    canvas_cache: dict = {}
    for image_ordinal, image_row in enumerate(image_rows):
        patch, _ = extract_patch(
            images.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=540
        )
        stabilized = render_movies(
            patch, np.zeros((1, 1, 2), dtype=np.float32), device=args.device
        )[0, 0]
        base = np.fft.fft2((stabilized - 127.0) / 255.0, norm="ortho")
        base_power = np.square(np.abs(base.ravel()[flat_index]))
        for trace_ordinal, trace in enumerate(traces):
            movie = render_movies(patch, trace[None], device=args.device)[0]
            direct_hz, actual = direct_mode_power(
                movie, flat_index, frame_rate_hz=frame_rate_hz
            )
            periodic_movie = periodic_translation_movie(stabilized, trace, ppd=float(PPD))
            periodic_hz, periodic = direct_mode_power(
                periodic_movie, flat_index, frame_rate_hz=frame_rate_hz
            )
            if not np.array_equal(direct_hz, temporal) or not np.array_equal(
                periodic_hz, temporal
            ):
                raise RuntimeError("rendered and phase-carrier TF grids differ")
            ideal = base_power[:, None] * phase_power[trace_ordinal]
            finite_metrics = distribution_metrics(
                actual[valid_mode][:, valid_tf], ideal[valid_mode][:, valid_tf]
            )
            periodic_metrics = distribution_metrics(
                periodic[valid_mode][:, valid_tf], ideal[valid_mode][:, valid_tf]
            )
            rows.append(
                {
                    "image_index": int(images.iloc[int(image_row)].image_index),
                    "trace_index": int(trace_table.iloc[trace_ordinal].trace_index),
                    **{f"finite_{key}": value for key, value in finite_metrics.items()},
                    **{f"periodic_{key}": value for key, value in periodic_metrics.items()},
                }
            )
            aggregate_actual += actual
            aggregate_periodic += periodic
            aggregate_ideal += ideal
        canvas_cache.clear()
        print(
            f"exact retinal-spectrum validation image {image_ordinal + 1}/{len(image_rows)}",
            flush=True,
        )
    pair_table = pd.DataFrame(rows)
    finite_aggregate = distribution_metrics(
        aggregate_actual[valid_mode][:, valid_tf],
        aggregate_ideal[valid_mode][:, valid_tf],
    )
    periodic_aggregate = distribution_metrics(
        aggregate_periodic[valid_mode][:, valid_tf],
        aggregate_ideal[valid_mode][:, valid_tf],
    )
    actual_cube = cube_from_modes(
        aggregate_actual, distributor, len(spatial), len(orientations)
    )
    periodic_cube = cube_from_modes(
        aggregate_periodic, distributor, len(spatial), len(orientations)
    )
    ideal_cube = cube_from_modes(
        aggregate_ideal, distributor, len(spatial), len(orientations)
    )
    periodic_pair_median = float(pair_table.periodic_distribution_cosine.median())
    finite_pair_median = float(pair_table.finite_distribution_cosine.median())
    gates = {
        "periodic_aggregate": bool(
            periodic_aggregate["distribution_cosine"]
            >= float(args.periodic_aggregate_cosine_gate)
        ),
        "periodic_pairwise_median": bool(
            periodic_pair_median >= float(args.periodic_pairwise_median_cosine_gate)
        ),
        "finite_aggregate": bool(
            finite_aggregate["distribution_cosine"]
            >= float(args.finite_aggregate_cosine_gate)
        ),
        "finite_pairwise_median": bool(
            finite_pair_median >= float(args.finite_pairwise_median_cosine_gate)
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(args.out_dir / "pair_metrics.csv", index=False)
    np.savez_compressed(
        args.out_dir / "retinal_spectrum_validation.npz",
        actual_cube=actual_cube.astype(np.float32),
        periodic_cube=periodic_cube.astype(np.float32),
        ideal_cube=ideal_cube.astype(np.float32),
        spatial_cpd=spatial,
        temporal_hz=temporal,
        orientation_deg=orientations,
        image_rows=image_rows,
        trace_rows=trace_rows,
    )
    figure = args.out_dir / "retinal_spectrum_validation.png"
    render_validation_figure(
        figure,
        spatial=spatial,
        temporal=temporal,
        actual_cube=actual_cube,
        periodic_cube=periodic_cube,
        ideal_cube=ideal_cube,
        pairs=pair_table,
    )
    summary = {
        "analysis": "Kuang complete-trajectory identity and finite-aperture scorer boundary",
        "inputs": {
            "fixation_bank": str(args.fixation_bank.resolve()),
            "fixation_bank_manifest_sha256": sha256(args.fixation_bank / "manifest.json"),
            "filtered_trace_sha256": sha256(
                args.fixation_bank / "trace_xy_filtered.npy"
            ),
            "image_table": str(args.image_table.resolve()),
            "image_table_sha256": sha256(args.image_table),
            "tuning_table": str(args.tuning_table.resolve()),
            "tuning_table_sha256": sha256(args.tuning_table),
        },
        "frame_rate_hz": frame_rate_hz,
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_pairs": int(len(pair_table)),
        "analysis_samples": int(args.analysis_samples),
        "trace_kind": "filtered",
        "periodic_translation_control": {
            "aggregate": periodic_aggregate,
            "pairwise_median_distribution_cosine": periodic_pair_median,
            "aggregate_cosine_gate": float(args.periodic_aggregate_cosine_gate),
            "pairwise_median_cosine_gate": float(
                args.periodic_pairwise_median_cosine_gate
            ),
            "aggregate_gate_pass": gates["periodic_aggregate"],
            "pairwise_gate_pass": gates["periodic_pairwise_median"],
        },
        "finite_scorer_comparison": {
            "aggregate": finite_aggregate,
            "pairwise_median_distribution_cosine": finite_pair_median,
            "aggregate_cosine_gate": float(args.finite_aggregate_cosine_gate),
            "pairwise_median_cosine_gate": float(
                args.finite_pairwise_median_cosine_gate
            ),
            "aggregate_gate_pass": gates["finite_aggregate"],
            "pairwise_gate_pass": gates["finite_pairwise_median"],
        },
        "claim_gates": gates,
        "all_claim_gates_pass": bool(all(gates.values())),
        "instantaneous_velocity_used": False,
        "claim_boundary": (
            "The Kuang factorization is exact for periodic translation. The finite "
            "scorer comparison separately includes aperture replacement and bilinear "
            "interpolation; model response analyses use direct-rendered movies."
        ),
        "figure": str(figure.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_claim_gates_pass"]:
        raise RuntimeError(f"retinal-spectrum validation failed: {gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
