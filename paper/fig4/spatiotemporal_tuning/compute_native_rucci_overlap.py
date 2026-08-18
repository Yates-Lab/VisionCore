#!/usr/bin/env python3
"""Deprecated instantaneous-velocity diagnostic; this is not retinal power.

Every natural-image Fourier mode ``k`` and native 240-Hz eye-velocity sample
``v(t)`` contributes at the instantaneous temporal frequency
``|k dot v(t)|``. This discards temporal order and displacement correlations,
so it is not the temporal PSD of the translated movie except in special
constant-velocity cases. It is retained for audit/reproducibility only.

Production analyses must instead Fourier-transform the complete carrier
``exp(-i 2pi k.X(t))``; see ``analyze_image_specific_joint_engagement.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float64)
EPS = 1e-30


def matrix_trace_time_contract(
    matrix_dir: Path,
    trace_xy: np.ndarray,
    *,
    model_output_rate_hz: int,
) -> dict[str, Any]:
    """Recover and validate the retained-trace timing used by the matrix."""
    summary_path = Path(matrix_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing matrix timing provenance: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = dict(summary.get("trace_time_contract") or {})
    if not contract:
        shard_contracts = [
            dict(item.get("trace_time_contract") or {})
            for item in (summary.get("shard_summaries") or [])
        ]
        shard_contracts = [item for item in shard_contracts if item]
        if shard_contracts:
            contract = shard_contracts[0]
            if any(item != contract for item in shard_contracts[1:]):
                raise ValueError(
                    f"Merged matrix shards disagree on trace timing in {summary_path}"
                )
    source_rate_hz = int(contract.get("source_trace_rate_hz", 0))
    source_samples = int(contract.get("source_trace_samples", 0))
    declared_output_rate_hz = int(contract.get("model_output_rate_hz", 0))
    if source_rate_hz < 1 or source_samples < 1 or declared_output_rate_hz < 1:
        raise ValueError(f"Incomplete matrix trace_time_contract in {summary_path}")
    trace_xy = np.asarray(trace_xy)
    if trace_xy.ndim != 3 or trace_xy.shape[1:] != (source_samples, 2):
        raise ValueError(
            "trace_xy.npy disagrees with the matrix timing contract: "
            f"shape={trace_xy.shape}, expected [trace, {source_samples}, 2]."
        )
    if declared_output_rate_hz != int(model_output_rate_hz):
        raise ValueError(
            "Replay output rate differs from the matrix: "
            f"{model_output_rate_hz} versus {declared_output_rate_hz} Hz."
        )
    if declared_output_rate_hz < source_rate_hz or (
        declared_output_rate_hz % source_rate_hz
    ):
        raise ValueError(
            "Matrix trace rate must divide model output rate; got "
            f"{source_rate_hz} -> {declared_output_rate_hz} Hz."
        )
    factor = declared_output_rate_hz // source_rate_hz
    expected_scored = source_samples * factor
    declared_scored = int(contract.get("scored_trace_samples", expected_scored))
    if declared_scored != expected_scored:
        raise ValueError(
            "Matrix scored_trace_samples is inconsistent with its rates: "
            f"{declared_scored} versus {expected_scored}."
        )
    return {
        **contract,
        "source_trace_rate_hz": source_rate_hz,
        "source_trace_samples": source_samples,
        "model_output_rate_hz": declared_output_rate_hz,
        "scored_samples_per_source_trace_sample": factor,
        "scored_trace_samples": expected_scored,
        "analysis_interval_seconds": source_samples / float(source_rate_hz),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grouped_tuning_csv", type=Path)
    parser.add_argument("--robust-summary", type=Path, required=True)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help=(
            "Use the exact retained traces and selected images from a scored "
            "matrix. Retained 120-Hz traces are expanded to the native 240-Hz "
            "model grid before computing velocity."
        ),
    )
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--n-traces", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def observed_tuning_tensor(table: pd.DataFrame):
    dynamic = table.loc[table.temporal_hz.gt(0)].copy()
    units = np.sort(dynamic.unit_index.unique().astype(int))
    sf = np.sort(dynamic.spatial_cpd.unique().astype(float))
    tf = np.sort(dynamic.temporal_hz.unique().astype(float))
    orientation = np.sort(dynamic.probe_orientation_deg.unique().astype(float))
    expected_rows = len(units) * len(sf) * len(tf) * len(orientation)
    if len(dynamic) != expected_rows:
        raise RuntimeError(
            f"Tuning table is incomplete: {len(dynamic)} rows, expected {expected_rows}"
        )
    unit_map = {value: index for index, value in enumerate(units)}
    sf_map = {value: index for index, value in enumerate(sf)}
    tf_map = {value: index for index, value in enumerate(tf)}
    orientation_map = {value: index for index, value in enumerate(orientation)}
    tuning = np.full(
        (len(units), len(sf), len(tf), len(orientation)), np.nan, dtype=np.float64
    )
    for row in dynamic.itertuples(index=False):
        tuning[
            unit_map[int(row.unit_index)],
            sf_map[float(row.spatial_cpd)],
            tf_map[float(row.temporal_hz)],
            orientation_map[float(row.probe_orientation_deg)],
        ] = float(row.response_amp_rms)
    if not np.all(np.isfinite(tuning)):
        raise RuntimeError("Tuning tensor contains missing or non-finite conditions")
    tuning = np.clip(tuning, 0.0, None)
    normalized = tuning / np.maximum(tuning.sum(axis=(1, 2, 3), keepdims=True), EPS)
    return units, sf, tf, orientation, tuning, normalized


def nearest_axis_orientation(
    kxy: np.ndarray,
    orientations_deg: np.ndarray,
) -> np.ndarray:
    """Map Fourier wave-vector direction to the closest bar-axis orientation."""
    normal_deg = np.degrees(np.arctan2(kxy[:, 1], kxy[:, 0]))
    bar_deg = np.mod(normal_deg - 90.0, 180.0)
    distance = np.abs(
        ((bar_deg[:, None] - orientations_deg[None] + 90.0) % 180.0) - 90.0
    )
    return np.argmin(distance, axis=1)


def log_edges(centers_linear: np.ndarray) -> np.ndarray:
    """Return half-octave boundaries around positive log-spaced centers."""
    centers = np.log2(np.asarray(centers_linear, dtype=np.float64))
    if centers.ndim != 1 or len(centers) < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("Log-spaced frequency centers must be a finite 1-D array")
    return np.concatenate(
        (
            [centers[0] - 0.5 * (centers[1] - centers[0])],
            0.5 * (centers[:-1] + centers[1:]),
            [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
        )
    )


def log_bin_indices(
    radial: np.ndarray,
    spatial_cpd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign positive radial frequencies to the measured SF grid."""
    radial = np.asarray(radial, dtype=np.float64)
    edges = log_edges(spatial_cpd)
    index = np.full(radial.shape, -1, dtype=int)
    positive = radial > 0
    index[positive] = np.digitize(np.log2(radial[positive]), edges) - 1
    keep = positive & (index >= 0) & (index < len(spatial_cpd))
    return index, keep


def frequency_grid() -> dict[str, np.ndarray]:
    """Return physical Fourier modes supported by the exact model crop."""
    from paper.fig4.upstream.real_trace_matrix.model import OUT_SIZE, PPD

    height, width = OUT_SIZE
    if height != width:
        raise ValueError(OUT_SIZE)
    n = int(height)
    f_axis = np.fft.fftfreq(n, d=1.0 / float(PPD))
    fy, fx = np.meshgrid(f_axis, f_axis, indexing="ij")
    kx = fx
    ky = -fy  # image rows increase downward; physical y increases upward
    radius = np.hypot(kx, ky)
    df = float(PPD) / n
    nyquist = float(PPD) / 2.0
    mask = (radius >= 0.5 * df) & (radius <= nyquist)
    flat = np.flatnonzero(mask.ravel())
    return {
        "flat_index": flat,
        "kxy": np.column_stack((kx.ravel()[flat], ky.ravel()[flat])),
        "radial": radius.ravel()[flat],
        "df": np.asarray(df),
        "nyquist": np.asarray(nyquist),
    }


def render_stabilized_frame(patch: np.ndarray) -> np.ndarray:
    """Render the exact central model crop at zero retinal displacement."""
    import torch

    from paper.fig4.upstream.real_trace_matrix.model import (
        OUT_SIZE,
        PPD,
        _eye_deg_to_norm,
        _shift_movie_with_eye,
        _standardize_uint_like,
    )

    image = _standardize_uint_like(patch)
    eye = torch.zeros((1, 2), dtype=torch.float32)
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    base = torch.from_numpy(image).to(dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        shifted = _shift_movie_with_eye(
            base,
            eye_norm,
            out_size=OUT_SIZE,
            scale_factor=1.0,
            torch=torch,
        )
    return shifted[0].cpu().numpy().astype(np.float32)


def load_matrix_trace_bank(
    matrix_dir: Path,
    n_traces: int,
) -> tuple[np.ndarray, float, list[int], dict]:
    """Load exact scored traces and put them on the native model-output grid."""
    import torch

    from paper.fig4.upstream.real_trace_matrix.model import _trace_on_output_grid

    trace_xy = np.load(Path(matrix_dir) / "trace_xy.npy")
    contract = matrix_trace_time_contract(
        Path(matrix_dir),
        trace_xy,
        model_output_rate_hz=240,
    )
    count = min(int(n_traces), len(trace_xy))
    if count < 1:
        raise ValueError("n_traces must select at least one retained trace")
    selected = np.unique(
        np.round(np.linspace(0, len(trace_xy) - 1, count)).astype(int)
    )
    traces = np.stack(
        [
            _trace_on_output_grid(
                trace_xy[index],
                source_rate_hz=int(contract["source_trace_rate_hz"]),
                output_rate_hz=int(contract["model_output_rate_hz"]),
                torch=torch,
            )
            for index in selected
        ],
        axis=0,
    )
    return (
        traces,
        1.0 / float(contract["model_output_rate_hz"]),
        selected.tolist(),
        contract,
    )


def image_mode_power_matrix_from_matrix(
    matrix_dir: Path,
    n_images: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Measure per-image Fourier-mode power on exact stabilized model crops."""
    from paper.fig4.upstream.real_trace_matrix.core import extract_patch

    images = pd.read_csv(Path(matrix_dir) / "image_feature_table.csv")
    images = images.sort_values("image_index").reset_index(drop=True)
    count = min(int(n_images), len(images))
    if count < 1:
        raise ValueError("n_images must select at least one matrix image")
    selected = np.unique(
        np.round(np.linspace(0, len(images) - 1, count)).astype(int)
    )
    grid = frequency_grid()
    powers = []
    canvas_cache = {}
    for ordinal, row_index in enumerate(selected):
        patch, _ = extract_patch(
            images.iloc[int(row_index)],
            canvas_cache=canvas_cache,
            patch_size_px=540,
        )
        frame = render_stabilized_frame(patch)
        coefficient = np.fft.fft2(frame) / frame.size
        powers.append(
            np.abs(coefficient.ravel()[grid["flat_index"]]) ** 2
        )
        if ordinal == 0 or (ordinal + 1) % 10 == 0 or ordinal + 1 == len(selected):
            print(f"matrix image Fourier power {ordinal + 1}/{len(selected)}", flush=True)
    return (
        np.asarray(powers, dtype=np.float64),
        np.asarray(grid["kxy"], dtype=np.float64),
        {
            "matrix_image_rows": selected.tolist(),
            "n_images": int(len(selected)),
            "normalization": "fft2(exact stabilized 151x151 model crop) / crop.size",
        },
    )


def image_mode_power_from_matrix(
    matrix_dir: Path,
    n_images: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Measure mean Fourier-mode power from exact stabilized model crops."""
    powers, kxy, audit = image_mode_power_matrix_from_matrix(matrix_dir, n_images)
    return powers.mean(axis=0), kxy, audit


def native_kinematic_occupancy(
    *,
    mode_power: np.ndarray,
    kxy: np.ndarray,
    traces: np.ndarray,
    frame_rate_hz: float,
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    orientation_deg: np.ndarray,
) -> tuple[np.ndarray, dict]:
    mode_power = np.asarray(mode_power, dtype=np.float64)
    kxy = np.asarray(kxy, dtype=np.float64)
    if len(mode_power) != len(kxy):
        raise RuntimeError("Natural-image Fourier power and k vectors differ in length")
    velocity = np.diff(np.asarray(traces, dtype=np.float64), axis=1) * float(frame_rate_hz)
    radial = np.linalg.norm(kxy, axis=1)
    sf_bin, resolved = log_bin_indices(radial, spatial_cpd)
    orientation_bin = nearest_axis_orientation(kxy, orientation_deg)
    temporal_edges = log_edges(temporal_hz)
    occupancy = np.zeros(
        (len(SCALES), len(spatial_cpd), len(temporal_hz), len(orientation_deg)),
        dtype=np.float64,
    )
    total_weight = float(np.sum(mode_power))
    resolved_power = float(np.sum(mode_power[resolved]))
    in_tf_grid = np.zeros(len(SCALES), dtype=np.float64)
    positive_dynamic = np.zeros(len(SCALES), dtype=np.float64)
    kept = np.flatnonzero(resolved)
    for start in range(0, len(kept), 256):
        mode_ids = kept[start : start + 256]
        instantaneous = np.abs(
            np.einsum("md,ntd->nmt", kxy[mode_ids], velocity, optimize=True)
        )
        for scale_index, scale in enumerate(SCALES[1:], start=1):
            hz = float(scale) * instantaneous
            for local_index, mode_id in enumerate(mode_ids):
                values = hz[:, local_index].reshape(-1)
                positive = values > 0
                weight = float(mode_power[mode_id])
                positive_dynamic[scale_index] += weight * int(np.count_nonzero(positive))
                if not np.any(positive):
                    continue
                temporal_bin = np.digitize(
                    np.log2(values[positive]), temporal_edges
                ) - 1
                valid = (temporal_bin >= 0) & (temporal_bin < len(temporal_hz))
                counts = np.bincount(
                    temporal_bin[valid], minlength=len(temporal_hz)
                )[: len(temporal_hz)]
                occupancy[
                    scale_index,
                    sf_bin[mode_id],
                    :,
                    orientation_bin[mode_id],
                ] += weight * counts
                in_tf_grid[scale_index] += weight * int(counts.sum())
        if start == 0 or (start // 256 + 1) % 10 == 0 or start + 256 >= len(kept):
            print(
                f"kinematic occupancy modes {min(start + 256, len(kept))}/{len(kept)}",
                flush=True,
            )
    audit = {
        "n_image_fourier_modes": int(len(kxy)),
        "n_spatially_resolved_modes": int(len(kept)),
        "fraction_image_mode_power_in_resolved_sf_grid": resolved_power / max(total_weight, EPS),
        "fraction_positive_dynamic_weight_in_tf_grid_by_scale": {
            str(float(scale)): float(in_tf_grid[index] / max(positive_dynamic[index], EPS))
            if index > 0
            else 0.0
            for index, scale in enumerate(SCALES)
        },
        "definition": "mean natural-image Fourier-mode power weighted counts of native 240-Hz framewise ft=scale*abs(k dot v(t))",
    }
    return occupancy, audit


def overlap_table(
    units: np.ndarray,
    tuning: np.ndarray,
    occupancy: np.ndarray,
    robust_summary: pd.DataFrame,
) -> pd.DataFrame:
    raw = np.einsum("stfo,utfo->su", occupancy, tuning, optimize=True)
    one_x = raw[int(np.flatnonzero(np.isclose(SCALES, 1.0))[0])]
    relative = raw / np.maximum(one_x[None], EPS)
    summary = robust_summary.set_index("unit_index")
    ordered = robust_summary.loc[
        robust_summary.unit_index.isin(units),
        ["unit_index", "weighted_center_sf_cpd"],
    ].copy()
    ordered["weighted_center_sf_cpd"] = pd.to_numeric(
        ordered["weighted_center_sf_cpd"], errors="coerce"
    )
    if not np.isfinite(ordered.weighted_center_sf_cpd.to_numpy(float)).all():
        raise RuntimeError("Active tuning units lack a finite weighted-center SF")
    ordered = ordered.sort_values(
        ["weighted_center_sf_cpd", "unit_index"], kind="mergesort"
    ).reset_index(drop=True)
    n_tail = len(ordered) // 3
    if n_tail < 1:
        raise RuntimeError("Need at least three active units for SF tertiles")
    ordered["sf_group"] = "middle_sf"
    ordered.loc[ordered.index < n_tail, "sf_group"] = "low_sf"
    ordered.loc[ordered.index >= len(ordered) - n_tail, "sf_group"] = "high_sf"
    sf_group = ordered.set_index("unit_index").sf_group
    rows = []
    for unit_row, unit in enumerate(units):
        if int(unit) not in summary.index:
            raise RuntimeError(f"Unit {unit} is absent from robust tuning summary")
        metadata = summary.loc[int(unit)]
        for scale_index, scale in enumerate(SCALES):
            rows.append(
                {
                    "unit_index": int(unit),
                    "motion_scale": float(scale),
                    "observed_passband_overlap": float(raw[scale_index, unit_row]),
                    "passband_overlap_relative_to_1x": float(relative[scale_index, unit_row]),
                    "fit_r2": float(metadata.fit_r2),
                    "peak_censoring": str(metadata.peak_censoring),
                    "low_sf_censored": bool(metadata.low_sf_censored),
                    "weighted_center_sf_cpd": float(
                        metadata.weighted_center_sf_cpd
                    ),
                    "sf_group": str(sf_group.loc[int(unit)]),
                    "preferred_tf_hz_if_resolved": (
                        float(metadata.preferred_tf_hz)
                        if not bool(metadata.low_tf_censored) and not bool(metadata.high_tf_censored)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_median(values: np.ndarray, rng: np.random.Generator, n: int = 4000):
    values = np.asarray(values, dtype=np.float64)
    draws = np.median(values[rng.integers(0, len(values), size=(n, len(values)))], axis=1)
    return float(np.median(values)), [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def render(
    occupancy: np.ndarray,
    spatial: np.ndarray,
    temporal: np.ndarray,
    overlaps: pd.DataFrame,
    output: Path,
    *,
    trace_label: str,
) -> dict:
    marginal = occupancy.sum(axis=-1)
    positive = marginal[1:][marginal[1:] > 0]
    floor = float(np.quantile(positive, 0.02))
    ceiling = float(np.quantile(positive, 0.995))
    levels = np.linspace(np.log10(floor), np.log10(ceiling), 22)
    figure = plt.figure(figsize=(13.8, 6.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.32))
    contour_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    temporal_axis = figure.add_subplot(grid[0, 3])
    engagement_axis = figure.add_subplot(grid[1, :2])
    lowpass_axis = figure.add_subplot(grid[1, 2:])
    contour = None
    for axis, scale in zip(contour_axes, (0.5, 1.0, 3.0), strict=True):
        scale_index = int(np.flatnonzero(np.isclose(SCALES, scale))[0])
        value = marginal[scale_index].T
        contour = axis.contourf(
            spatial,
            temporal,
            np.log10(np.maximum(value, floor)),
            levels=levels,
            cmap="magma",
            extend="both",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_title(f"{scale:g}× measured motion")
        axis.set_xlabel("resolved SF (cpd)")
    contour_axes[0].set_ylabel("instantaneous TF (Hz)")
    figure.colorbar(
        contour,
        ax=contour_axes,
        fraction=0.035,
        pad=0.018,
        label="log weighted occupancy",
    )

    for scale, color in ((0.5, "#6baed6"), (1.0, "#2f78b7"), (2.0, "#f28e2b"), (3.0, "#c84c36")):
        index = int(np.flatnonzero(np.isclose(SCALES, scale))[0])
        value = marginal[index].sum(axis=0)
        value = value / max(float(value.sum()), EPS)
        temporal_axis.plot(temporal, value, marker="o", ms=2.5, lw=1.4, label=f"{scale:g}×", color=color)
    temporal_axis.set_xscale("log", base=2)
    temporal_axis.set_xlabel("instantaneous TF (Hz)")
    temporal_axis.set_ylabel("fraction of resolved occupancy")
    temporal_axis.set_title("Kinematic temporal marginal")
    temporal_axis.legend(frameon=False, ncol=2, loc="upper left")

    rng = np.random.default_rng(20260816)
    summaries = []
    for scale, group in overlaps.loc[overlaps.motion_scale.gt(0)].groupby("motion_scale"):
        median, interval = _bootstrap_median(
            group.passband_overlap_relative_to_1x.to_numpy(float), rng
        )
        summaries.append((scale, median, *interval))
    summary_array = np.asarray(summaries)
    engagement_axis.fill_between(
        summary_array[:, 0], summary_array[:, 2], summary_array[:, 3], color="#4c78a8", alpha=0.22
    )
    engagement_axis.plot(summary_array[:, 0], summary_array[:, 1], "o-", color="#2f6fa3", lw=2)
    engagement_axis.axhline(1.0, color="0.5", linestyle=":")
    engagement_axis.set(
        xlabel="motion scale",
        ylabel="observed passband overlap (relative to 1×)",
        title="Population passband engagement",
        xticks=SCALES[1:],
    )

    for label, group_name, color in (
        ("low SF tertile", "low_sf", "#7b6fd0"),
        ("high SF tertile", "high_sf", "#31a354"),
    ):
        curve = overlaps.loc[
            overlaps.sf_group.eq(group_name) & overlaps.motion_scale.gt(0)
        ].groupby("motion_scale").passband_overlap_relative_to_1x.median()
        lowpass_axis.plot(curve.index, curve.values, "o-", lw=1.8, label=label, color=color)
    lowpass_axis.axhline(1.0, color="0.5", linestyle=":")
    lowpass_axis.set(
        xlabel="motion scale",
        ylabel="median overlap relative to 1×",
        title="Cycle-valid SF tertiles; observed tuning surfaces",
        xticks=SCALES[1:],
    )
    lowpass_axis.legend(frameon=False)
    figure.suptitle(
        "Native eye motion moves resolved image structure through measured RR100 passbands\n"
        f"$f_t(t)=|k\\cdot v(t)|$; {trace_label}; no short-window FFT",
        fontsize=12,
        fontweight="bold",
    )
    figure.savefig(output, dpi=190, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return {
        str(float(scale)): {
            "median_relative_to_1x": float(median),
            "ci95": [float(low), float(high)],
        }
        for scale, median, low, high in summaries
    }


def main() -> None:
    args = parse_args()
    tuning_table = pd.read_csv(args.grouped_tuning_csv)
    robust_summary = pd.read_csv(args.robust_summary)
    units, spatial, temporal, orientation, raw_tuning, normalized_tuning = observed_tuning_tensor(
        tuning_table
    )
    matrix_dir = args.matrix_dir.resolve()
    mode_power, kxy, image_audit = image_mode_power_from_matrix(
        matrix_dir,
        int(args.n_images),
    )
    traces, dt, trace_rows, trace_contract = load_matrix_trace_bank(
        matrix_dir,
        int(args.n_traces),
    )
    trace_label = (
        f"{len(traces)} retained 120-Hz traces endpoint-interpolated "
        "onto the 240-Hz model grid"
    )
    image_source = f"exact stabilized crops from {matrix_dir}"
    occupancy, occupancy_audit = native_kinematic_occupancy(
        mode_power=mode_power,
        kxy=kxy,
        traces=traces,
        frame_rate_hz=1.0 / dt,
        spatial_cpd=spatial,
        temporal_hz=temporal,
        orientation_deg=orientation,
    )
    overlaps = overlap_table(units, normalized_tuning, occupancy, robust_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlaps.to_csv(args.out_dir / "unit_passband_overlap.csv", index=False)
    np.savez_compressed(
        args.out_dir / "native_kinematic_occupancy.npz",
        occupancy=occupancy.astype(np.float32),
        scales=SCALES,
        spatial_cpd=spatial,
        temporal_hz=temporal,
        orientation_deg=orientation,
        unit_indices=units,
        observed_tuning=raw_tuning.astype(np.float32),
        normalized_tuning=normalized_tuning.astype(np.float32),
        trace_source_rows=np.asarray(trace_rows, dtype=np.int64),
    )
    figure_path = args.out_dir / "native_rucci_passband_overlap.png"
    population_summary = render(
        occupancy,
        spatial,
        temporal,
        overlaps,
        figure_path,
        trace_label=trace_label,
    )
    report = {
        "analysis": "deprecated instantaneous-velocity occupancy diagnostic",
        "grouped_tuning_csv": str(args.grouped_tuning_csv.resolve()),
        "robust_tuning_summary": str(args.robust_summary.resolve()),
        "image_power": image_source,
        "matrix_dir": str(matrix_dir),
        "matrix_image_audit": image_audit,
        "matrix_trace_time_contract": trace_contract,
        "n_native_eye_traces": int(len(traces)),
        "eye_trace_rate_hz": float(1.0 / dt),
        "n_active_rr100_units": int(len(units)),
        "occupancy_audit": occupancy_audit,
        "population_passband_overlap": population_summary,
        "claim_boundary": "This |k dot v(t)| histogram is not a retinal temporal PSD because it discards temporal order and displacement correlations. It is retained only for historical reproducibility and must not enter a production spectral or causal claim.",
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "native_rucci_overlap_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
