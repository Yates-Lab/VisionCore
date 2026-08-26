#!/usr/bin/env python3
"""Test the Rucci mechanism with image-specific joint SF×TF×orientation overlap.

Preferred TF alone is not a valid mechanistic predictor.  Under an arbitrary
eye-position trajectory ``X(t)``, image mode ``k`` acquires the complete phase
carrier ``exp(-i 2 pi k dot X(t))``; only constant-velocity translation reduces
this to a single frequency ``k dot v``.  This analysis estimates the carrier's
finite-trace temporal power, preserves SF, TF, and orientation until the final
dot product with a unit's *observed* periodic tuning tensor, and asks whether
the resulting image-specific engagement predicts the exact causal
moving-minus-stabilized model response and spatial-selectivity change.
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
from scipy.signal.windows import dpss
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (
    EPS,
    image_mode_power_matrix_from_matrix,
    load_matrix_trace_bank,
    log_edges,
    observed_tuning_tensor,
)


DEFAULT_MATRIX = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/figure4_real_trace_pilot_corrected/merged"
)
DEFAULT_GROUPED = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/periodic_tuning_respaced/"
    "frequency_tuning_grouped.csv"
)
DEFAULT_AUDIT = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/periodic_tuning_respaced/fit_audit/"
    "m77_tuning_fit_audit.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/figure4_real_trace_pilot_corrected/"
    "image_specific_joint_engagement"
)

TF_DISPLAY_EDGES_HZ = np.asarray([0.0, 6.0, 12.0, 24.0, 48.0, 120.0001])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_matrix_contract(matrix_dir: Path) -> dict[str, object]:
    matrix_dir = Path(matrix_dir)
    summary_path = matrix_dir / "summary.json"
    stabilized_path = matrix_dir / "stabilized_baseline_summary.json"
    if not summary_path.is_file() or not stabilized_path.is_file():
        raise FileNotFoundError(
            "Corrected analysis requires merged and stabilized matrix summaries"
        )
    summary = json.loads(summary_path.read_text())
    n_images = len(pd.read_csv(matrix_dir / "image_feature_table.csv"))
    n_traces = len(pd.read_csv(matrix_dir / "trace_feature_table.csv"))
    n_units = len(pd.read_csv(matrix_dir / "unit_feature_table.csv"))
    declared = {
        "n_images": int(summary["n_images"]),
        "n_traces": int(summary["n_traces"]),
        "n_units": int(summary["n_units"]),
    }
    observed = {"n_images": n_images, "n_traces": n_traces, "n_units": n_units}
    if declared != observed:
        raise RuntimeError(
            f"Merged matrix coordinate mismatch: declared={declared}, observed={observed}"
        )
    expected_movie_shape = (n_images * n_traces, n_units)
    shapes = {}
    for name in ("ssi_matrix", "expected_spikes_matrix", "mean_rate_matrix"):
        shape = tuple(np.load(matrix_dir / f"{name}.npy", mmap_mode="r").shape)
        shapes[name] = list(shape)
        if shape != expected_movie_shape:
            raise RuntimeError(
                f"{name} shape {shape} does not match {expected_movie_shape}"
            )
    expected_stable_shape = (n_images, n_units)
    for name in (
        "stabilized_ssi_by_image",
        "stabilized_expected_spikes_by_image",
        "stabilized_mean_rate_by_image",
    ):
        shape = tuple(np.load(matrix_dir / f"{name}.npy", mmap_mode="r").shape)
        shapes[name] = list(shape)
        if shape != expected_stable_shape:
            raise RuntimeError(
                f"{name} shape {shape} does not match {expected_stable_shape}"
            )
    trace_shape = tuple(np.load(matrix_dir / "trace_xy.npy", mmap_mode="r").shape)
    shapes["trace_xy"] = list(trace_shape)
    if trace_shape[0] != n_traces or trace_shape[-1] != 2:
        raise RuntimeError(f"trace_xy shape {trace_shape} does not match trace table")
    integrity = summary.get("integrity_checks")
    if integrity is not None and not all(bool(value) for value in integrity.values()):
        raise RuntimeError(f"Merged matrix integrity checks failed: {integrity}")
    return {
        "declared_coordinates": declared,
        "array_shapes": shapes,
        "merge_integrity_checks": integrity,
        "summary_sha256": sha256(summary_path),
        "stabilized_summary_sha256": sha256(stabilized_path),
        "validated_common_provenance": summary.get("validated_common_provenance"),
    }


def validate_native240_trace_contract(contract: dict[str, object]) -> None:
    """Refuse mixed-rate caches that do not match the declared model replay."""
    expected_integer = {
        "source_trace_rate_hz": 120,
        "source_trace_samples": 40,
        "model_output_rate_hz": 240,
        "scored_samples_per_source_trace_sample": 2,
        "scored_trace_samples": 80,
    }
    for key, expected in expected_integer.items():
        observed = contract.get(key)
        if observed is None or int(observed) != expected:
            raise RuntimeError(
                f"Native-240 trace contract mismatch for {key}: "
                f"{observed!r} versus {expected!r}"
            )
    expected_float = {
        "scored_bin_seconds": 1.0 / 240.0,
        "analysis_interval_seconds": 1.0 / 3.0,
    }
    for key, expected in expected_float.items():
        observed = contract.get(key)
        if observed is None or not np.isclose(float(observed), expected):
            raise RuntimeError(
                f"Native-240 trace contract mismatch for {key}: "
                f"{observed!r} versus {expected!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unit", type=int, default=25)
    parser.add_argument(
        "--n-images",
        type=int,
        default=0,
        help="Evenly sample this many matrix images; 0 uses every image.",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=0,
        help="Evenly sample this many retained eye traces; 0 uses every trace.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--expected-dataset-configs-sha256", default=None)
    parser.add_argument(
        "--require-native-240-contract",
        action="store_true",
        help="Require 40 source samples at 120 Hz scored as 80 bins at 240 Hz.",
    )
    return parser.parse_args()


def circular_orientation_weights(
    kxy: np.ndarray, orientations_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate Fourier-vector bar orientation on a 180° circle."""
    orientations = np.asarray(orientations_deg, dtype=np.float64)
    if len(orientations) < 2 or not np.allclose(
        np.diff(orientations), np.diff(orientations)[0]
    ):
        raise ValueError("orientation probes must be uniformly spaced")
    step = float(np.diff(orientations)[0])
    if not np.isclose(step * len(orientations), 180.0):
        raise ValueError("orientation probes must tile the 180-degree period")
    normal = np.degrees(np.arctan2(kxy[:, 1], kxy[:, 0]))
    bar = np.mod(normal - 90.0 - orientations[0], 180.0)
    position = bar / step
    lower = np.floor(position).astype(int) % len(orientations)
    fraction = position - np.floor(position)
    upper = (lower + 1) % len(orientations)
    return lower, upper, 1.0 - fraction, fraction


def log_interpolation_weights(
    values: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate positive values between log-frequency probes."""
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    log_centers = np.log2(centers)
    edges = log_edges(centers)
    positive = np.isfinite(values) & (values > 0)
    log_values = np.full(values.shape, np.nan, dtype=np.float64)
    log_values[positive] = np.log2(values[positive])
    valid = positive & (log_values >= edges[0]) & (log_values <= edges[-1])
    position = np.interp(
        np.where(valid, log_values, log_centers[0]),
        log_centers,
        np.arange(len(centers), dtype=np.float64),
    )
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(centers) - 1)
    fraction = position - lower
    fraction[upper == lower] = 0.0
    return lower, upper, 1.0 - fraction, fraction, valid


def trajectory_phase_spectra(
    kxy: np.ndarray,
    traces: np.ndarray,
    frame_rate_hz: float,
    *,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direction-collapsed temporal power of each trajectory phase carrier.

    For a translated image, spatial Fourier mode ``k`` is multiplied by
    ``exp(-i 2 pi k dot X(t))``.  Its temporal spectrum depends on the complete
    displacement trajectory, not on a histogram of instantaneous velocities.
    We remove each finite trace's temporal mean (the static/DC component), use
    two DPSS tapers, average power over traces, and fold positive and negative
    temporal frequencies because the current grating bank does not contain the
    two motion directions separately.
    """
    trace = np.asarray(traces, dtype=np.float64)
    if trace.ndim != 3 or trace.shape[-1] != 2 or trace.shape[1] < 4:
        raise ValueError(f"traces must have shape [trace,time,2], got {trace.shape}")
    n_time = int(trace.shape[1])
    signed_hz = np.fft.fftfreq(n_time, d=1.0 / float(frame_rate_hz))
    positive_hz = np.fft.rfftfreq(n_time, d=1.0 / float(frame_rate_hz))[1:]
    fold_indices = [np.flatnonzero(np.isclose(np.abs(signed_hz), value)) for value in positive_hz]
    if any(len(index) not in (1, 2) for index in fold_indices):
        raise RuntimeError("Unable to pair positive and negative temporal FFT bins")
    tapers = dpss(n_time, NW=1.5, Kmax=2, sym=False).astype(np.float64)
    spectra = np.zeros((len(kxy), len(positive_hz)), dtype=np.float32)
    for start in range(0, len(kxy), int(chunk_size)):
        stop = min(start + int(chunk_size), len(kxy))
        dot = np.einsum(
            "md,ntd->nmt", kxy[start:stop], trace, optimize=True
        )
        carrier = np.exp(-2j * np.pi * dot)
        carrier -= carrier.mean(axis=-1, keepdims=True)
        power = np.zeros((stop - start, n_time), dtype=np.float64)
        for taper in tapers:
            transformed = np.fft.fft(
                carrier * taper[None, None, :], axis=-1, norm="ortho"
            )
            power += np.mean(np.abs(transformed) ** 2, axis=0)
        power /= float(len(tapers))
        spectra[start:stop] = np.column_stack(
            [power[:, index].sum(axis=1) for index in fold_indices]
        ).astype(np.float32)
        if start == 0 or stop == len(kxy) or (start // int(chunk_size) + 1) % 10 == 0:
            print(f"trajectory-phase spectra {stop}/{len(kxy)}", flush=True)
    return positive_hz.astype(np.float64), spectra


def interpolate_tuning_to_fft_bins(
    tuning: np.ndarray,
    probe_temporal_hz: np.ndarray,
    fft_temporal_hz: np.ndarray,
) -> np.ndarray:
    """Log-linearly interpolate raw sampled tuning onto resolvable FFT bins.

    This is interpolation between observed grating responses, not a fitted peak.
    Frequencies outside the measured grating grid are assigned zero sensitivity.
    The resulting tensor is normalized per unit only to remove arbitrary response
    scale from the spectral-support score.
    """
    value = np.asarray(tuning, dtype=np.float64)
    probe = np.asarray(probe_temporal_hz, dtype=np.float64)
    target = np.asarray(fft_temporal_hz, dtype=np.float64)
    if value.ndim != 4 or value.shape[2] != len(probe):
        raise ValueError("tuning must have shape [unit,sf,probe_tf,orientation]")
    output = np.zeros(
        (value.shape[0], value.shape[1], len(target), value.shape[3]),
        dtype=np.float64,
    )
    valid = (target >= probe[0]) & (target <= probe[-1])
    if np.any(valid):
        position = np.interp(
            np.log2(target[valid]),
            np.log2(probe),
            np.arange(len(probe), dtype=np.float64),
        )
        lower = np.floor(position).astype(int)
        upper = np.minimum(lower + 1, len(probe) - 1)
        fraction = position - lower
        output[:, :, valid, :] = (
            value[:, :, lower, :] * (1.0 - fraction)[None, None, :, None]
            + value[:, :, upper, :] * fraction[None, None, :, None]
        )
    output = np.clip(output, 0.0, None)
    return output / np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), EPS)


def mode_contract(
    *,
    kxy: np.ndarray,
    traces: np.ndarray,
    frame_rate_hz: float,
    spatial_cpd: np.ndarray,
    orientation_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    radial = np.linalg.norm(kxy, axis=1)
    sf0, sf1, sw0, sw1, resolved = log_interpolation_weights(radial, spatial_cpd)
    ori0, ori1, ow0, ow1 = circular_orientation_weights(kxy, orientation_deg)
    temporal_hz, temporal = trajectory_phase_spectra(kxy, traces, frame_rate_hz)
    temporal[~resolved] = 0.0
    return {
        "sf0": sf0,
        "sf1": sf1,
        "sw0": sw0,
        "sw1": sw1,
        "ori0": ori0,
        "ori1": ori1,
        "ow0": ow0,
        "ow1": ow1,
        "resolved": resolved,
        "temporal_hz": temporal_hz,
        "temporal": temporal,
    }


def image_unit_engagement(
    mode_power: np.ndarray,
    normalized_tuning: np.ndarray,
    contract: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute joint and control image-by-unit spectral engagement scores.

    The separable control preserves each unit's SF-by-orientation and TF
    marginals while removing their measured coupling.  It therefore provides
    the direct control for whether the particular SF--TF pairing matters, not
    merely whether a unit is spatially and temporally tuned.
    """
    n_modes = mode_power.shape[1]
    n_units = normalized_tuning.shape[0]
    joint_match = np.zeros((n_modes, n_units), dtype=np.float32)
    separable_match = np.zeros_like(joint_match)
    sf_match = np.zeros_like(joint_match)
    temporal = np.asarray(contract["temporal"], dtype=np.float32)
    tuning_sf_orientation = normalized_tuning.sum(axis=2)
    tuning_tf = normalized_tuning.sum(axis=(1, 3))
    for mode in np.flatnonzero(contract["resolved"]):
        weighted_tuning = np.zeros(
            (n_units, normalized_tuning.shape[2]), dtype=np.float64
        )
        weighted_sf = np.zeros(n_units, dtype=np.float64)
        for sf_index, sf_weight in (
            (int(contract["sf0"][mode]), float(contract["sw0"][mode])),
            (int(contract["sf1"][mode]), float(contract["sw1"][mode])),
        ):
            for ori_index, ori_weight in (
                (int(contract["ori0"][mode]), float(contract["ow0"][mode])),
                (int(contract["ori1"][mode]), float(contract["ow1"][mode])),
            ):
                weight = sf_weight * ori_weight
                if weight == 0:
                    continue
                weighted_tuning += weight * normalized_tuning[:, sf_index, :, ori_index]
                weighted_sf += weight * tuning_sf_orientation[:, sf_index, ori_index]
        joint_match[mode] = weighted_tuning @ temporal[mode]
        separable_match[mode] = weighted_sf * (tuning_tf @ temporal[mode])
        sf_match[mode] = weighted_sf
    joint = np.asarray(mode_power @ joint_match, dtype=np.float64)
    separable = np.asarray(mode_power @ separable_match, dtype=np.float64)
    sf_only = np.asarray(mode_power @ sf_match, dtype=np.float64)
    temporal_power = np.asarray(mode_power @ temporal, dtype=np.float64)
    tf_only = np.asarray(temporal_power @ tuning_tf.T, dtype=np.float64)
    total_dynamic_power = temporal_power.sum(axis=1)
    total_image_power = mode_power[:, np.asarray(contract["resolved"], dtype=bool)].sum(
        axis=1
    )

    def divide_by_image_total(values: np.ndarray, total: np.ndarray) -> np.ndarray:
        return np.divide(
            values,
            total[:, None],
            out=np.zeros_like(values, dtype=np.float64),
            where=total[:, None] > 0,
        )

    return {
        "joint": joint,
        "separable": separable,
        "tf_marginal": tf_only,
        "sf_marginal": sf_only,
        "joint_fraction": divide_by_image_total(joint, total_dynamic_power),
        "separable_fraction": divide_by_image_total(
            separable, total_dynamic_power
        ),
        "tf_marginal_fraction": divide_by_image_total(
            tf_only, total_dynamic_power
        ),
        "sf_marginal_fraction": divide_by_image_total(
            sf_only, total_image_power
        ),
        "total_dynamic_power": np.broadcast_to(
            total_dynamic_power[:, None], joint.shape
        ).copy(),
        "total_image_power": np.broadcast_to(
            total_image_power[:, None], joint.shape
        ).copy(),
    }, joint_match


def actual_causal_effects(matrix_dir: Path, unit_indices: np.ndarray) -> dict[str, np.ndarray]:
    units = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    n_images = len(pd.read_csv(matrix_dir / "image_feature_table.csv"))
    n_traces = len(pd.read_csv(matrix_dir / "trace_feature_table.csv"))
    n_units = len(units)
    if not np.array_equal(units.unit_index.to_numpy(int), np.arange(n_units)):
        raise ValueError("matrix unit coordinate is not contiguous unit_index order")
    moving_rate = np.load(matrix_dir / "mean_rate_matrix.npy").reshape(
        n_images, n_traces, n_units
    ).mean(axis=1)
    stable_rate = np.load(matrix_dir / "stabilized_mean_rate_by_image.npy")
    moving_ssi = np.load(matrix_dir / "ssi_matrix.npy").reshape(
        n_images, n_traces, n_units
    )
    moving_expected = np.load(matrix_dir / "expected_spikes_matrix.npy").reshape(
        n_images, n_traces, n_units
    )
    stable_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy")
    stable_expected = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy")

    def weighted(values: np.ndarray, weights: np.ndarray, axes) -> np.ndarray:
        numerator = np.sum(values * weights, axis=axes, dtype=np.float64)
        denominator = np.sum(weights, axis=axes, dtype=np.float64)
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )

    moving_unit_ssi = weighted(moving_ssi, moving_expected, axes=(0, 1))
    stable_unit_ssi = weighted(stable_ssi, stable_expected, axes=0)
    delta_ssi = moving_unit_ssi - stable_unit_ssi
    percent_ssi = np.divide(
        100.0 * delta_ssi,
        stable_unit_ssi,
        out=np.full_like(delta_ssi, np.nan),
        where=stable_unit_ssi > 0,
    )
    selected = np.asarray(unit_indices, dtype=int)
    return {
        "moving_rate": moving_rate[:, selected],
        "stable_rate": stable_rate[:, selected],
        "rate_delta": (moving_rate - stable_rate)[:, selected],
        "log_rate_ratio": np.log(np.maximum(moving_rate[:, selected], EPS))
        - np.log(np.maximum(stable_rate[:, selected], EPS)),
        "moving_ssi": moving_unit_ssi[selected],
        "stable_ssi": stable_unit_ssi[selected],
        "ssi_delta": delta_ssi[selected],
        "ssi_percent": percent_ssi[selected],
    }


def selectivity_bits(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = values.sum(axis=0, keepdims=True)
    probability = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
    uniform_ratio = probability * values.shape[0]
    term = np.zeros_like(values)
    positive = probability > 0
    term[positive] = probability[positive] * np.log2(uniform_ratio[positive])
    return term.sum(axis=0)


def unit_correlations(predictor: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    if predictor.shape != outcome.shape:
        raise ValueError(f"shape mismatch: {predictor.shape} versus {outcome.shape}")
    output = np.full(predictor.shape[1], np.nan, dtype=np.float64)
    for unit in range(predictor.shape[1]):
        finite = np.isfinite(predictor[:, unit]) & np.isfinite(outcome[:, unit])
        if np.count_nonzero(finite) >= 5:
            output[unit] = float(spearmanr(predictor[finite, unit], outcome[finite, unit]).statistic)
    return output


def bootstrap_median(
    values: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, tuple[float, float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.median(
        values[rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))],
        axis=1,
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.median(values)), (float(low), float(high))


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, tuple[float, float]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[finite]
    y = np.asarray(y, dtype=np.float64)[finite]
    observed = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        rows = rng.integers(0, len(x), size=len(x))
        if np.unique(x[rows]).size < 2 or np.unique(y[rows]).size < 2:
            continue
        value = float(spearmanr(x[rows], y[rows]).statistic)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        interval = (float("nan"), float("nan"))
    else:
        low, high = np.quantile(np.asarray(draws), [0.025, 0.975])
        interval = (float(low), float(high))
    return float(observed.statistic), float(observed.pvalue), interval


def example_occupancy(
    mode_power: np.ndarray,
    contract: dict[str, np.ndarray],
    shape: tuple[int, int, int],
) -> np.ndarray:
    occupancy = np.zeros(shape, dtype=np.float64)
    temporal = contract["temporal"]
    for mode in np.flatnonzero(contract["resolved"]):
        for sf_index, sf_weight in (
            (int(contract["sf0"][mode]), float(contract["sw0"][mode])),
            (int(contract["sf1"][mode]), float(contract["sw1"][mode])),
        ):
            for ori_index, ori_weight in (
                (int(contract["ori0"][mode]), float(contract["ow0"][mode])),
                (int(contract["ori1"][mode]), float(contract["ow1"][mode])),
            ):
                weight = float(mode_power[mode]) * sf_weight * ori_weight
                if weight:
                    occupancy[sf_index, :, ori_index] += weight * temporal[mode]
    return occupancy


def render(
    *,
    output: Path,
    unit_index: int,
    image_index: int,
    spatial: np.ndarray,
    temporal: np.ndarray,
    tuning: np.ndarray,
    occupancy: np.ndarray,
    engagement: dict[str, np.ndarray],
    causal: dict[str, np.ndarray],
    unit_correlations_by_predictor: dict[str, np.ndarray],
    engagement_ssi: np.ndarray,
    coupling_excess_ssi: np.ndarray,
    ssi_summary: dict,
    trusted: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.6, 8.2), constrained_layout=True)
    unit_row = int(np.flatnonzero(ssi_summary["unit_indices"] == unit_index)[0])
    def broad_tf_bands(values: np.ndarray, *, reduction: str) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        centers = []
        for low, high in zip(TF_DISPLAY_EDGES_HZ[:-1], TF_DISPLAY_EDGES_HZ[1:]):
            keep = (temporal >= low) & (temporal < high)
            if not np.any(keep):
                continue
            selected = values[:, keep]
            rows.append(selected.mean(axis=1) if reduction == "mean" else selected.sum(axis=1))
            centers.append(0.5 * high if low == 0 else np.sqrt(low * high))
        return np.stack(rows, axis=1), np.asarray(centers, dtype=np.float64)

    tuning_fine = tuning[unit_row].max(axis=-1)
    occupancy_fine = occupancy.sum(axis=-1)
    contribution_fine = (occupancy * tuning[unit_row]).sum(axis=-1)
    tuning_map, display_temporal = broad_tf_bands(tuning_fine, reduction="mean")
    occupancy_map, _ = broad_tf_bands(occupancy_fine, reduction="mean")
    contribution, _ = broad_tf_bands(contribution_fine, reduction="mean")
    tuning_map = tuning_map.T
    occupancy_map = occupancy_map.T
    contribution = contribution.T

    def contour(axis: plt.Axes, values: np.ndarray, *, title: str, cmap: str) -> None:
        positive = values[values > 0]
        floor = float(np.quantile(positive, 0.02)) if len(positive) else EPS
        maximum = max(float(values.max()), EPS)
        floor_fraction = max(floor / maximum, 1e-6)
        normalized = np.maximum(values / maximum, floor_fraction)
        levels = np.linspace(np.log10(floor_fraction), 0, 18)
        axis.contourf(
            spatial,
            display_temporal,
            np.log10(np.maximum(normalized, 1e-6)),
            levels=levels,
            cmap=cmap,
            extend="min",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        sf_tick_indices = np.unique(
            np.round(np.linspace(0, len(spatial) - 1, min(5, len(spatial)))).astype(int)
        )
        sf_ticks = spatial[sf_tick_indices]
        tf_ticks = display_temporal
        axis.set_xticks(sf_ticks)
        axis.set_xticklabels([f"{value:.2g}" for value in sf_ticks])
        axis.set_yticks(tf_ticks)
        axis.set_yticklabels(
            [
                f"{low:g}–{high:g}"
                for low, high in zip(TF_DISPLAY_EDGES_HZ[:-1], TF_DISPLAY_EDGES_HZ[1:])
                if np.any((temporal >= low) & (temporal < high))
            ]
        )
        axis.set_xlabel("spatial frequency (cycles/deg)")
        axis.set_ylabel("broad temporal-frequency band (Hz)")
        axis.set_title(title, loc="left")

    contour(
        axes[0, 0],
        tuning_map,
        title=f"A  u{unit_index:03d}: measured joint passband (max orientation)",
        cmap="viridis",
    )
    contour(
        axes[0, 1],
        occupancy_map,
        title=f"B  Image {image_index}: trajectory-induced SF×TF power density",
        cmap="magma",
    )
    contour(
        axes[0, 2],
        contribution,
        title="C  Passband-weighted power density",
        cmap="inferno",
    )

    example_x = engagement["joint"][:, unit_row]
    example_y = causal["rate_delta"][:, unit_row]
    example_rho = float(spearmanr(example_x, example_y).statistic)
    axes[1, 0].scatter(example_x, example_y, s=35, color="#2F78B7", alpha=0.8)
    for row, (x, y) in enumerate(zip(example_x, example_y)):
        axes[1, 0].annotate(str(row), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=6)
    axes[1, 0].axhline(0, color="0.5", lw=0.8)
    axes[1, 0].set(
        title=f"D  Same-image causal test for u{unit_index:03d}: ρ={example_rho:.2f}",
        xlabel="fraction of dynamic power aligned to joint tuning",
        ylabel="moving − stabilized mean rate",
    )

    predictor_order = ("joint", "separable", "tf_marginal", "sf_marginal")
    predictor_label = {
        "joint": "joint\nSF×TF×ori",
        "separable": "separable\n(SF×ori)×TF",
        "tf_marginal": "TF\nmarginal",
        "sf_marginal": "SF×ori\nmarginal",
    }
    rng = np.random.default_rng(20260817)
    for x_position, name in enumerate(predictor_order):
        values = unit_correlations_by_predictor[name]
        finite = values[np.isfinite(values)]
        jitter = rng.uniform(-0.15, 0.15, size=len(finite))
        axes[1, 1].scatter(
            x_position + jitter,
            finite,
            s=13,
            color="0.65",
            alpha=0.28,
            edgecolor="none",
        )
        median, interval = ssi_summary["predictor_within_unit_rate_delta"][name]
        axes[1, 1].errorbar(
            x_position,
            median,
            yerr=[[median - interval[0]], [interval[1] - median]],
            fmt="D",
            ms=6,
            capsize=3,
            color="#1F77B4" if name == "joint" else "#777777",
            mec="white",
            mew=0.7,
            zorder=4,
        )
    axes[1, 1].axhline(0, color="0.5", lw=0.8)
    axes[1, 1].set(
        title="E  Does tuning-specific alignment predict causal rate changes?",
        ylabel="within-unit, across-image Spearman ρ",
        xticks=np.arange(len(predictor_order)),
        xticklabels=[predictor_label[name] for name in predictor_order],
    )

    axes[1, 2].scatter(
        coupling_excess_ssi[~trusted],
        causal["ssi_percent"][~trusted],
        s=23,
        color="0.75",
        alpha=0.45,
        edgecolor="none",
        label="continuous peak uncertain",
    )
    axes[1, 2].scatter(
        coupling_excess_ssi[trusted],
        causal["ssi_percent"][trusted],
        s=31,
        color="#2E7D49",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
        label="joint peak audit-trusted",
    )
    axes[1, 2].axhline(0, color="0.5", lw=0.8)
    axes[1, 2].set(
        title=(
            "F  Does SF–TF coupling beyond the marginals explain ΔSSI?\n"
            f"all units: ρ={ssi_summary['coupling_excess_ssi_vs_ssi_percent']['rho']:.2f}, "
            f"p={ssi_summary['coupling_excess_ssi_vs_ssi_percent']['p']:.2g}"
        ),
        xlabel="joint − separable alignment selectivity (bits)",
        ylabel="moving versus stabilized SSI (%)",
    )
    axes[1, 2].legend(frameon=False, fontsize=7, loc="best")
    for axis in axes[1]:
        axis.grid(alpha=0.16)
    figure.suptitle(
        "Rucci audit uses full trajectory-phase spectra (motion direction collapsed)",
        fontsize=13.5,
        fontweight="semibold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=210, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = args.matrix_dir.resolve()
    matrix_audit = validate_matrix_contract(matrix_dir)
    provenance = matrix_audit.get("validated_common_provenance") or {}
    expected_provenance = {
        "model_provenance.model.checkpoint_sha256": args.expected_checkpoint_sha256,
        "model_provenance.model.dataset_configs_sha256": args.expected_dataset_configs_sha256,
    }
    for key, expected in expected_provenance.items():
        if expected is not None and provenance.get(key) != expected:
            raise RuntimeError(
                f"Merged matrix provenance mismatch for {key}: "
                f"{provenance.get(key)!r} versus expected {expected!r}"
            )
    images = pd.read_csv(matrix_dir / "image_feature_table.csv")
    traces_table = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    n_images = len(images) if int(args.n_images) <= 0 else min(int(args.n_images), len(images))
    n_traces = len(traces_table) if int(args.n_traces) <= 0 else min(int(args.n_traces), len(traces_table))
    units, spatial, probe_temporal, orientation, raw_tuning, _ = observed_tuning_tensor(
        pd.read_csv(args.grouped_csv)
    )
    mode_power, kxy, image_audit = image_mode_power_matrix_from_matrix(
        matrix_dir, n_images
    )
    if mode_power.shape[0] != len(images):
        raise RuntimeError(
            "Causal image matching requires every matrix image; omit --n-images or "
            "set it to the full matrix image count."
        )
    traces, dt, trace_rows, trace_contract = load_matrix_trace_bank(
        matrix_dir, n_traces
    )
    if args.require_native_240_contract:
        validate_native240_trace_contract(trace_contract)
    contract = mode_contract(
        kxy=kxy,
        traces=traces,
        frame_rate_hz=1.0 / dt,
        spatial_cpd=spatial,
        orientation_deg=orientation,
    )
    temporal = np.asarray(contract["temporal_hz"], dtype=np.float64)
    normalized_tuning = interpolate_tuning_to_fft_bins(
        raw_tuning, probe_temporal, temporal
    )
    engagement, _ = image_unit_engagement(mode_power, normalized_tuning, contract)
    causal = actual_causal_effects(matrix_dir, units)
    correlations = {
        name: unit_correlations(values, causal["rate_delta"])
        for name, values in engagement.items()
    }
    predictor_summary = {
        name: bootstrap_median(
            values, n_bootstrap=args.n_bootstrap, seed=args.seed + index
        )
        for index, (name, values) in enumerate(correlations.items())
    }
    spectral_alignment = {
        "joint": engagement["joint_fraction"],
        "separable": engagement["separable_fraction"],
        "tf_marginal": engagement["tf_marginal_fraction"],
        "sf_marginal": engagement["sf_marginal_fraction"],
    }
    alignment_correlations = {
        name: correlations[f"{name}_fraction"] for name in spectral_alignment
    }
    engagement_ssi = selectivity_bits(spectral_alignment["joint"])
    separable_ssi = selectivity_bits(spectral_alignment["separable"])
    coupling_excess_ssi = engagement_ssi - separable_ssi
    finite = np.isfinite(engagement_ssi) & np.isfinite(causal["ssi_percent"])
    population = bootstrap_spearman(
        engagement_ssi,
        causal["ssi_percent"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 20,
    )
    coupling_finite = np.isfinite(coupling_excess_ssi) & np.isfinite(
        causal["ssi_percent"]
    )
    coupling_population = bootstrap_spearman(
        coupling_excess_ssi,
        causal["ssi_percent"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 21,
    )
    audit = pd.read_csv(args.audit_csv).set_index("unit_index")
    trusted = np.asarray(
        [
            unit in audit.index and str(audit.loc[unit, "audit_category"]) == "trusted"
            for unit in units
        ],
        dtype=bool,
    )
    trusted_finite = coupling_finite & trusted
    trusted_population = (
        bootstrap_spearman(
            coupling_excess_ssi[trusted_finite],
            causal["ssi_percent"][trusted_finite],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 22,
        )
        if np.count_nonzero(trusted_finite) >= 5
        else None
    )
    unit_table = pd.DataFrame(
        {
            "unit_index": units,
            "joint_engagement_ssi_bits": engagement_ssi,
            "separable_engagement_ssi_bits": separable_ssi,
            "joint_minus_separable_engagement_ssi_bits": coupling_excess_ssi,
            "ssi_percent_vs_stabilized": causal["ssi_percent"],
            "ssi_delta_bits_per_spike": causal["ssi_delta"],
            "joint_engagement_vs_rate_delta_spearman": correlations["joint"],
            "joint_alignment_fraction_vs_rate_delta_spearman": correlations[
                "joint_fraction"
            ],
            "separable_alignment_fraction_vs_rate_delta_spearman": correlations[
                "separable_fraction"
            ],
            "tf_marginal_alignment_fraction_vs_rate_delta_spearman": correlations[
                "tf_marginal_fraction"
            ],
            "sf_marginal_alignment_fraction_vs_rate_delta_spearman": correlations[
                "sf_marginal_fraction"
            ],
            "total_dynamic_power_vs_rate_delta_spearman": correlations[
                "total_dynamic_power"
            ],
            "separable_engagement_vs_rate_delta_spearman": correlations["separable"],
            "tf_marginal_vs_rate_delta_spearman": correlations["tf_marginal"],
            "sf_marginal_vs_rate_delta_spearman": correlations["sf_marginal"],
            "joint_peak_audit_trusted": trusted,
        }
    )
    unit_table.to_csv(args.out_dir / "unit_image_specific_joint_engagement.csv", index=False)
    example_image = int(
        np.argmax(
            spectral_alignment["joint"][:, np.flatnonzero(units == args.unit)[0]]
        )
    )
    example_phase_power = example_occupancy(
        mode_power[example_image],
        contract,
        (len(spatial), len(temporal), len(orientation)),
    )
    image_phase_power = np.stack(
        [
            example_occupancy(
                power,
                contract,
                (len(spatial), len(temporal), len(orientation)),
            )
            for power in mode_power
        ],
        axis=0,
    )
    np.savez_compressed(
        args.out_dir / "image_specific_joint_engagement.npz",
        unit_indices=units,
        spatial_cpd=spatial,
        temporal_hz=temporal,
        periodic_probe_temporal_hz=probe_temporal,
        display_tf_band_edges_hz=TF_DISPLAY_EDGES_HZ,
        orientation_deg=orientation,
        joint_engagement=engagement["joint"].astype(np.float32),
        separable_engagement=engagement["separable"].astype(np.float32),
        tf_marginal_engagement=engagement["tf_marginal"].astype(np.float32),
        sf_marginal_engagement=engagement["sf_marginal"].astype(np.float32),
        joint_alignment_fraction=engagement["joint_fraction"].astype(np.float32),
        separable_alignment_fraction=engagement["separable_fraction"].astype(np.float32),
        tf_marginal_alignment_fraction=engagement["tf_marginal_fraction"].astype(np.float32),
        sf_marginal_alignment_fraction=engagement["sf_marginal_fraction"].astype(np.float32),
        total_dynamic_power=engagement["total_dynamic_power"][:, 0].astype(np.float32),
        total_image_power=engagement["total_image_power"][:, 0].astype(np.float32),
        moving_rate=causal["moving_rate"].astype(np.float32),
        stabilized_rate=causal["stable_rate"].astype(np.float32),
        example_unit_index=np.asarray(args.unit, dtype=np.int64),
        example_image_index=np.asarray(example_image, dtype=np.int64),
        example_trajectory_phase_power=example_phase_power.astype(np.float32),
        image_trajectory_phase_power=image_phase_power.astype(np.float32),
        spectrum_method_version=np.asarray("trajectory_phase_dpss_nw1p5_k2_v1"),
    )
    summary = {
        "analysis": "image-specific joint SFxTFxorientation Rucci engagement",
        "definition": "per-image Fourier-mode power times the two-DPSS, finite-trace temporal power spectrum of mean-removed exp(-i2pi k dot X(t)); positive and negative temporal frequencies are folded; raw periodic responses are log-linearly interpolated onto resolvable FFT bins before the final dot product",
        "normalization": "Raw engagement estimates passband-weighted dynamic power. Tuning-specific alignment divides joint, separable, and TF scores by each image's total trajectory-induced dynamic power; the SF-only control is divided by total resolved image Fourier power. Total dynamic and image power are reported separately.",
        "direction_assumption": "Positive and negative temporal Fourier power are folded because the existing periodic bank measured one temporal direction per bar orientation; a bidirectional grating bank is required before claiming direction-specific matching.",
        "spectral_estimator": {
            "trajectory_carrier": "exp(-i2pi k dot X(t))",
            "static_component": "per-trace temporal mean removed before spectral estimation",
            "tapers": "DPSS NW=1.5 K=2",
            "native_frequency_resolution_hz": float(1.0 / (len(traces[0]) * dt)),
            "resolvable_positive_fft_hz": temporal.tolist(),
            "periodic_probe_temporal_hz": probe_temporal.tolist(),
            "display_tf_band_edges_hz": TF_DISPLAY_EDGES_HZ.tolist(),
        },
        "separable_control": "outer product of each unit's observed SF-by-orientation and TF marginals; preserves both marginals but removes measured SF--TF coupling",
        "matrix_dir": str(matrix_dir),
        "matrix_audit": matrix_audit,
        "grouped_tuning_csv": str(args.grouped_csv.resolve()),
        "grouped_tuning_sha256": sha256(args.grouped_csv),
        "tuning_audit_sha256": sha256(args.audit_csv),
        "n_images": int(len(images)),
        "n_native_eye_traces": int(len(traces)),
        "n_units": int(len(units)),
        "n_joint_peak_audit_trusted": int(trusted.sum()),
        "image_audit": image_audit,
        "trace_rows": trace_rows,
        "trace_contract": trace_contract,
        "fraction_fourier_modes_resolved_on_sf_grid": float(contract["resolved"].mean()),
        "predictor_within_unit_rate_delta": {
            name: {
                "median_spearman": float(value[0]),
                "bootstrap_ci95": [float(x) for x in value[1]],
            }
            for name, value in predictor_summary.items()
        },
        "primary_alignment_predictors": [
            "joint_fraction",
            "separable_fraction",
            "tf_marginal_fraction",
            "sf_marginal_fraction",
        ],
        "engagement_ssi_vs_ssi_percent": {
            "n_units": int(np.count_nonzero(finite)),
            "rho": population[0],
            "p": population[1],
            "bootstrap_ci95": list(population[2]),
        },
        "coupling_excess_ssi_vs_ssi_percent": {
            "definition": "image-selectivity bits of joint dynamic-power fraction minus image-selectivity bits of the matched separable-marginals dynamic-power fraction",
            "n_units": int(np.count_nonzero(coupling_finite)),
            "rho": coupling_population[0],
            "p": coupling_population[1],
            "bootstrap_ci95": list(coupling_population[2]),
        },
        "trusted_peak_sensitivity": (
            {
                "n_units": int(np.count_nonzero(trusted_finite)),
                "rho": trusted_population[0],
                "p": trusted_population[1],
                "bootstrap_ci95": list(trusted_population[2]),
            }
            if trusted_population is not None
            else None
        ),
        "claim_boundary": "The score uses the complete eye-position trajectory and is a direction-collapsed, second-order spectral-support predictor. Raw passband-weighted drive and total dynamic power are separated from the fraction aligned to each unit's tuning. Image Fourier phase cancels from the ideal translated-field power spectrum; finite crop and interpolation effects are checked separately against directly rendered retinal movies. Causal evidence comes from the exact moving-minus-stabilized model replay; a fitted preferred SF or TF is never used in the score.",
    }
    # Convert the summary back to the tuple form expected by the plotting code.
    plot_summary = {
        "unit_indices": units,
        "predictor_within_unit_rate_delta": {
            name: (
                predictor_summary[f"{name}_fraction"][0],
                predictor_summary[f"{name}_fraction"][1],
            )
            for name in spectral_alignment
        },
        "engagement_ssi_vs_ssi_percent": summary["engagement_ssi_vs_ssi_percent"],
        "coupling_excess_ssi_vs_ssi_percent": summary[
            "coupling_excess_ssi_vs_ssi_percent"
        ],
    }
    figure_path = args.out_dir / "m77_image_specific_joint_engagement.png"
    render(
        output=figure_path,
        unit_index=int(args.unit),
        image_index=example_image,
        spatial=spatial,
        temporal=temporal,
        tuning=normalized_tuning,
        occupancy=example_phase_power,
        engagement=spectral_alignment,
        causal=causal,
        unit_correlations_by_predictor=alignment_correlations,
        engagement_ssi=engagement_ssi,
        coupling_excess_ssi=coupling_excess_ssi,
        ssi_summary=plot_summary,
        trusted=trusted,
    )
    summary["figure"] = str(figure_path.resolve())
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
