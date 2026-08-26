#!/usr/bin/env python3
"""Validate ideal trajectory-phase spectra against exactly rendered retinal movies.

For an infinite translated image, spatial mode k evolves as
``I_k exp(-i 2 pi k dot X(t))``.  The exact Figure 4 renderer additionally has
finite-crop, bilinear-interpolation, and padding effects.  This audit compares
the DPSS temporal spectrum predicted by the ideal phase carrier with the same
DPSS spectrum measured after rendering actual movies through the scorer's crop
and interpolation path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.signal.windows import dpss


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_image_specific_joint_engagement import (
    TF_DISPLAY_EDGES_HZ,
    trajectory_phase_spectra,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (
    EPS,
    frequency_grid,
    load_matrix_trace_bank,
    log_edges,
    observed_tuning_tensor,
    render_stabilized_frame,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    OUT_SIZE,
    PPD,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)


DEFAULT_MATRIX = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/figure4_real_trace_pilot_corrected/merged"
)
DEFAULT_GROUPED = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/periodic_tuning_respaced/"
    "frequency_tuning_grouped.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/figure4_real_trace_pilot_corrected/"
    "phase_spectrum_renderer_validation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-images", type=int, default=2)
    parser.add_argument("--n-traces", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def folded_multitaper_power(signal: np.ndarray, frame_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return two-sided direction-collapsed non-DC DPSS power for [mode,time]."""
    value = np.asarray(signal)
    if value.ndim != 2 or value.shape[1] < 4:
        raise ValueError(f"signal must have shape [mode,time], got {value.shape}")
    value = value - value.mean(axis=1, keepdims=True)
    n_time = int(value.shape[1])
    signed = np.fft.fftfreq(n_time, d=1.0 / float(frame_rate_hz))
    positive = np.fft.rfftfreq(n_time, d=1.0 / float(frame_rate_hz))[1:]
    tapers = dpss(n_time, NW=1.5, Kmax=2, sym=False)
    power = np.zeros(value.shape, dtype=np.float64)
    for taper in tapers:
        transformed = np.fft.fft(value * taper[None], axis=1, norm="ortho")
        power += np.abs(transformed) ** 2
    power /= float(len(tapers))
    folded = np.column_stack(
        [power[:, np.isclose(np.abs(signed), frequency)].sum(axis=1) for frequency in positive]
    )
    return positive.astype(np.float64), folded


def render_movie(patch: np.ndarray, trace_xy: np.ndarray, *, device: str) -> np.ndarray:
    """Render one native-output-grid trace through the exact scorer geometry."""
    import torch

    image = _standardize_uint_like(patch)
    eye = torch.from_numpy(np.asarray(trace_xy, dtype=np.float32)).to(device)
    eye_norm = _eye_deg_to_norm(
        eye, ppd=PPD, img_size=image.shape, torch=torch
    )
    base = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    repeated = base.unsqueeze(0).expand(len(trace_xy), -1, -1)
    with torch.no_grad():
        shifted = _shift_movie_with_eye(
            repeated,
            eye_norm,
            out_size=OUT_SIZE,
            scale_factor=1.0,
            torch=torch,
        )
    return shifted.cpu().numpy().astype(np.float32)


def movie_mode_power(
    movie: np.ndarray,
    flat_index: np.ndarray,
    frame_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.fft.fft2(movie, axes=(-2, -1)) / float(movie.shape[-2] * movie.shape[-1])
    selected = coefficient.reshape(len(movie), -1)[:, flat_index].T
    return folded_multitaper_power(selected, frame_rate_hz)


def distribution_metrics(actual: np.ndarray, ideal: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(ideal, dtype=np.float64).ravel()
    total_a = float(a.sum())
    total_b = float(b.sum())
    if total_a <= EPS or total_b <= EPS:
        return {
            "actual_over_ideal_power": np.nan,
            "distribution_total_variation": np.nan,
            "distribution_cosine": np.nan,
            "relative_l1_after_gain_match": np.nan,
        }
    pa = a / total_a
    pb = b / total_b
    gain = total_a / total_b
    return {
        "actual_over_ideal_power": gain,
        "distribution_total_variation": float(0.5 * np.abs(pa - pb).sum()),
        "distribution_cosine": float(np.dot(pa, pb) / max(np.linalg.norm(pa) * np.linalg.norm(pb), EPS)),
        "relative_l1_after_gain_match": float(np.abs(a - gain * b).sum() / total_a),
    }


def orientation_bin(kxy: np.ndarray, orientations_deg: np.ndarray) -> np.ndarray:
    normal = np.degrees(np.arctan2(kxy[:, 1], kxy[:, 0]))
    bar = np.mod(normal - 90.0, 180.0)
    distance = np.abs(((bar[:, None] - orientations_deg[None] + 90.0) % 180.0) - 90.0)
    return np.argmin(distance, axis=1)


def aggregate_cube(
    power: np.ndarray,
    *,
    kxy: np.ndarray,
    temporal_hz: np.ndarray,
    spatial_cpd: np.ndarray,
    orientations_deg: np.ndarray,
) -> np.ndarray:
    radial = np.linalg.norm(kxy, axis=1)
    edges = log_edges(spatial_cpd)
    sf_index = np.digitize(np.log2(np.maximum(radial, EPS)), edges) - 1
    ori_index = orientation_bin(kxy, orientations_deg)
    cube = np.zeros(
        (len(spatial_cpd), len(TF_DISPLAY_EDGES_HZ) - 1, len(orientations_deg)),
        dtype=np.float64,
    )
    for mode in range(len(kxy)):
        sf = int(sf_index[mode])
        if sf < 0 or sf >= len(spatial_cpd):
            continue
        for band, (low, high) in enumerate(
            zip(TF_DISPLAY_EDGES_HZ[:-1], TF_DISPLAY_EDGES_HZ[1:])
        ):
            keep = (temporal_hz >= low) & (temporal_hz < high)
            if np.any(keep):
                cube[sf, band, ori_index[mode]] += float(power[mode, keep].sum())
    return cube


def render_figure(
    path: Path,
    *,
    spatial: np.ndarray,
    actual: np.ndarray,
    ideal: np.ndarray,
    pair_table: pd.DataFrame,
) -> None:
    actual_map = actual.sum(axis=-1).T
    ideal_map = ideal.sum(axis=-1).T
    actual_probability = actual_map / max(float(actual_map.sum()), EPS)
    ideal_probability = ideal_map / max(float(ideal_map.sum()), EPS)
    centers = np.asarray(
        [0.5 * high if low == 0 else np.sqrt(low * high) for low, high in zip(TF_DISPLAY_EDGES_HZ[:-1], TF_DISPLAY_EDGES_HZ[1:])]
    )
    labels = [
        f"{low:g}–{high:g}" for low, high in zip(TF_DISPLAY_EDGES_HZ[:-1], TF_DISPLAY_EDGES_HZ[1:])
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), constrained_layout=True)

    positive = np.concatenate((actual_probability[actual_probability > 0], ideal_probability[ideal_probability > 0]))
    vmin = max(float(np.quantile(positive, 0.02)), float(positive.max()) * 1e-5)
    vmax = float(positive.max())
    for axis, value, title in (
        (axes[0, 0], actual_probability, "A  Exact rendered retinal movies"),
        (axes[0, 1], ideal_probability, "B  Image FFT × trajectory-phase spectrum"),
    ):
        mesh = axis.pcolormesh(spatial, centers, np.maximum(value, vmin), shading="nearest", cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax))
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_yticks(centers, labels)
        axis.set(xlabel="spatial frequency (cycles/deg)", ylabel="broad TF band (Hz)", title=title)
    figure.colorbar(mesh, ax=axes[0, :2], label="fraction of dynamic spectral power", shrink=0.78)

    ratio = np.divide(actual_probability, ideal_probability, out=np.full_like(actual_probability, np.nan), where=ideal_probability > 0)
    ratio_image = axes[0, 2].pcolormesh(spatial, centers, np.log2(np.maximum(ratio, 2**-4)), shading="nearest", cmap="RdBu_r", vmin=-4, vmax=4)
    axes[0, 2].set_xscale("log", base=2)
    axes[0, 2].set_yscale("log", base=2)
    axes[0, 2].set_yticks(centers, labels)
    axes[0, 2].set(xlabel="spatial frequency (cycles/deg)", ylabel="broad TF band (Hz)", title="C  Rendered / ideal distribution")
    figure.colorbar(ratio_image, ax=axes[0, 2], label="log2 ratio", shrink=0.78)

    x = ideal_probability.ravel()
    y = actual_probability.ravel()
    keep = (x > 0) & (y > 0)
    axes[1, 0].scatter(x[keep], y[keep], s=25, alpha=0.7, color="#2F78B7")
    limits = [min(float(x[keep].min()), float(y[keep].min())), max(float(x[keep].max()), float(y[keep].max()))]
    axes[1, 0].plot(limits, limits, color="0.45", lw=1)
    axes[1, 0].set(xscale="log", yscale="log", xlabel="ideal phase-carrier power fraction", ylabel="rendered-movie power fraction", title="D  Broad SF×TF cells")

    axes[1, 1].scatter(np.arange(len(pair_table)), pair_table.distribution_cosine, color="#2E7D49", s=32)
    axes[1, 1].axhline(pair_table.distribution_cosine.median(), color="#2E7D49", lw=1.5)
    axes[1, 1].set(xlabel="image–trace pair", ylabel="cosine similarity", title=f"E  Pairwise spectral-shape agreement\nmedian={pair_table.distribution_cosine.median():.3f}")

    axes[1, 2].scatter(np.arange(len(pair_table)), pair_table.distribution_total_variation, color="#C84C36", s=32)
    axes[1, 2].axhline(pair_table.distribution_total_variation.median(), color="#C84C36", lw=1.5)
    axes[1, 2].set(xlabel="image–trace pair", ylabel="total-variation distance", title=f"F  Residual distribution error\nmedian={pair_table.distribution_total_variation.median():.3f}")
    for axis in axes[1]:
        axis.grid(alpha=0.16)
    figure.suptitle("Does the ideal trajectory-phase calculation reproduce the exact rendered retinal-movie spectrum?", fontsize=13.5, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = args.matrix_dir.resolve()
    images = pd.read_csv(matrix_dir / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    image_rows = np.unique(
        np.round(np.linspace(0, len(images) - 1, min(int(args.n_images), len(images)))).astype(int)
    )
    traces, dt, trace_rows, trace_contract = load_matrix_trace_bank(
        matrix_dir, int(args.n_traces)
    )
    frame_rate_hz = 1.0 / dt
    grid = frequency_grid()
    kxy = np.asarray(grid["kxy"], dtype=np.float64)
    flat_index = np.asarray(grid["flat_index"], dtype=int)
    _, spatial, _, orientations, _, _ = observed_tuning_tensor(pd.read_csv(args.grouped_csv))

    phase_power = []
    temporal_hz = None
    for trace_index, trace in enumerate(traces):
        frequency, power = trajectory_phase_spectra(
            kxy, trace[None], frame_rate_hz, chunk_size=512
        )
        if temporal_hz is None:
            temporal_hz = frequency
        elif not np.array_equal(temporal_hz, frequency):
            raise RuntimeError("trace spectra returned inconsistent temporal grids")
        phase_power.append(power)
        print(f"ideal phase spectrum trace {trace_index + 1}/{len(traces)}", flush=True)
    phase_power = np.asarray(phase_power)
    temporal_hz = np.asarray(temporal_hz)
    resolved_tf = temporal_hz <= 90.50966799187816
    resolved_sf = (
        np.log2(np.linalg.norm(kxy, axis=1)) >= log_edges(spatial)[0]
    ) & (
        np.log2(np.linalg.norm(kxy, axis=1)) <= log_edges(spatial)[-1]
    )

    rows = []
    aggregate_actual = np.zeros((len(spatial), len(TF_DISPLAY_EDGES_HZ) - 1, len(orientations)))
    aggregate_ideal = np.zeros_like(aggregate_actual)
    canvas_cache = {}
    for image_ordinal, image_row in enumerate(image_rows):
        patch, _ = extract_patch(
            images.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=540
        )
        stabilized = render_stabilized_frame(patch)
        base_coefficient = np.fft.fft2(stabilized) / stabilized.size
        base_power = np.abs(base_coefficient.ravel()[flat_index]) ** 2
        for trace_ordinal, trace in enumerate(traces):
            movie = render_movie(patch, trace, device=args.device)
            direct_hz, actual = movie_mode_power(movie, flat_index, frame_rate_hz)
            if not np.array_equal(direct_hz, temporal_hz):
                raise RuntimeError("rendered and ideal spectra use different temporal grids")
            ideal = base_power[:, None] * phase_power[trace_ordinal]
            metrics = distribution_metrics(
                actual[resolved_sf][:, resolved_tf], ideal[resolved_sf][:, resolved_tf]
            )
            rows.append(
                {
                    "image_row": int(image_row),
                    "trace_row": int(trace_rows[trace_ordinal]),
                    **metrics,
                }
            )
            aggregate_actual += aggregate_cube(
                actual,
                kxy=kxy,
                temporal_hz=temporal_hz,
                spatial_cpd=spatial,
                orientations_deg=orientations,
            )
            aggregate_ideal += aggregate_cube(
                ideal,
                kxy=kxy,
                temporal_hz=temporal_hz,
                spatial_cpd=spatial,
                orientations_deg=orientations,
            )
        print(f"exact rendered validation image {image_ordinal + 1}/{len(image_rows)}", flush=True)

    pair_table = pd.DataFrame(rows)
    pair_table.to_csv(args.out_dir / "phase_spectrum_renderer_pair_metrics.csv", index=False)
    aggregate_actual /= max(len(rows), 1)
    aggregate_ideal /= max(len(rows), 1)
    np.savez_compressed(
        args.out_dir / "phase_spectrum_renderer_validation.npz",
        actual_cube=aggregate_actual.astype(np.float32),
        ideal_cube=aggregate_ideal.astype(np.float32),
        spatial_cpd=spatial,
        temporal_hz=temporal_hz,
        tf_band_edges_hz=TF_DISPLAY_EDGES_HZ,
        orientation_deg=orientations,
        image_rows=image_rows,
        trace_rows=np.asarray(trace_rows),
    )
    aggregate_metrics = distribution_metrics(aggregate_actual, aggregate_ideal)
    figure_path = args.out_dir / "m77_phase_spectrum_renderer_validation.png"
    render_figure(
        figure_path,
        spatial=spatial,
        actual=aggregate_actual,
        ideal=aggregate_ideal,
        pair_table=pair_table,
    )
    summary = {
        "analysis": "ideal trajectory-phase spectrum versus exact rendered retinal movies",
        "matrix_dir": str(matrix_dir),
        "n_images": int(len(image_rows)),
        "n_traces": int(len(traces)),
        "n_image_trace_pairs": int(len(rows)),
        "frame_rate_hz": float(frame_rate_hz),
        "n_time_samples": int(traces.shape[1]),
        "temporal_resolution_hz": float(frame_rate_hz / traces.shape[1]),
        "trace_contract": trace_contract,
        "aggregate": aggregate_metrics,
        "pairwise_median": {
            column: float(pair_table[column].median())
            for column in (
                "actual_over_ideal_power",
                "distribution_total_variation",
                "distribution_cosine",
                "relative_l1_after_gain_match",
            )
        },
        "claim_boundary": "Agreement validates the ideal pure-translation phase-carrier approximation against the exact finite crop, bilinear interpolation, and zero padding for the sampled model retinal movies. It does not validate direction pooling or any downstream neural interpretation.",
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
