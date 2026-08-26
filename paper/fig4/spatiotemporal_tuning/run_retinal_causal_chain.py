#!/usr/bin/env python3
"""Score a renderer-faithful retinal-motion causal chain with a selected model.

This command consumes an independently selected natural-image table, the
real-fixation bank, and a dense zero-behavior population tuning table.
For every requested image/trace/motion-scale condition it computes the exact
retinal movie spectrum, projects that spectrum onto measured population tuning, and
plays the identical movie through the selected model to obtain rate, expected spikes, and
spatial-map SSI.  Behavior is always a fixed all-zero vector.

The command is shardable over image and trace rows.  It stores summaries and
spectral projections, never the full retinal movies or full response maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal.windows import dpss
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import (
    mode_to_grid_matrix,
    render_movies,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import frequency_grid
from paper.fig4.spatiotemporal_tuning.retinal_motion_conditions import phase_scramble_trace
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    PPD,
    RealTraceMatrixScorer,
)


EPS = 1e-12
DEFAULT_OUTPUT_RATE_HZ = 240.0
# Generic compatibility roots for plotting/analysis helpers. Model execution
# itself requires explicit checkpoint, dataset, population, image, trace, and
# tuning inputs; no model identity is encoded here.
DEFAULT_CHAIN = ROOT / "outputs/retinal_causal_chain"
DEFAULT_MATRIX = DEFAULT_CHAIN / "matrix"
DEFAULT_MCFARLAND = ROOT / "scripts/mcfarland_outputs_mono.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, required=True)
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--tuning-table", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-label",
        default=None,
        help="Human-readable label stored in metadata and figures; defaults to checkpoint stem.",
    )
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, required=True)
    parser.add_argument(
        "--population-version",
        required=True,
        help="Named population view to load from --population-spec-dir.",
    )
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-kind", choices=("filtered", "raw"), default="filtered")
    parser.add_argument(
        "--trace-transform",
        choices=(
            "identity",
            "rotate90",
            "time_reverse",
            "phase_scramble",
            "movie_phase_scramble",
        ),
        default="identity",
        help=(
            "Matched trajectory control. Time reversal preserves the complete-trajectory "
            "power spectrum. movie_phase_scramble is the production phase-only "
            "control: it preserves the rendered movie's complete SF×TF amplitude and "
            "mean image while randomizing dynamic joint phase. "
            "phase_scramble is the legacy coordinate-spectrum sensitivity control."
        ),
    )
    parser.add_argument("--trace-transform-seed", type=int, default=20260819)
    parser.add_argument("--motion-scales", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0))
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-stop", type=int, default=0)
    parser.add_argument("--trace-start", type=int, default=0)
    parser.add_argument("--trace-stop", type=int, default=0)
    parser.add_argument(
        "--image-indices-file",
        type=Path,
        default=None,
        help="CSV containing an explicit, unique image_index column.",
    )
    parser.add_argument(
        "--trace-indices-file",
        type=Path,
        default=None,
        help="CSV containing an explicit, unique trace_index column.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument(
        "--sample-images-evenly",
        type=int,
        default=0,
        help="After applying the image range, sample this many rows across its full extent.",
    )
    parser.add_argument(
        "--sample-traces-evenly",
        type=int,
        default=0,
        help="After applying the trace range, sample this many rows across its full extent.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--history-samples", type=int, default=60)
    parser.add_argument("--analysis-samples", type=int, default=240)
    parser.add_argument("--save-example-maps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tuning_tensors(
    table: pd.DataFrame, *, output_rate_hz: float = DEFAULT_OUTPUT_RATE_HZ
) -> dict[str, np.ndarray]:
    """Return complete signed-rate and validated-passband tuning tensors.

    Exact-CID Figure-4 tables provide ``passband_weight`` from the released Yu
    SFxTF fit multiplied by the unit's acquired axial-orientation tuning.  That
    explicit column takes precedence over the historical phase-RMS proxy.
    """
    required = {
        "unit_index",
        "spatial_cpd",
        "temporal_hz",
        "probe_orientation_deg",
        "mean_rate",
        "response_amp_rms",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"tuning table is missing columns: {missing}")
    units = np.sort(table.unit_index.unique().astype(int))
    spatial = np.sort(table.spatial_cpd.unique().astype(float))
    temporal = np.sort(table.loc[table.temporal_hz.gt(0), "temporal_hz"].unique().astype(float))
    orientation = np.sort(table.probe_orientation_deg.unique().astype(float))
    shape = (len(units), len(spatial), len(temporal), len(orientation))
    keys = ["unit_index", "spatial_cpd", "temporal_hz", "probe_orientation_deg"]
    if table.duplicated(keys).any():
        raise RuntimeError("tuning table contains duplicated grid cells")
    dynamic = table.loc[table.temporal_hz.gt(0)].sort_values(keys)
    expected_dynamic = int(np.prod(shape))
    if len(dynamic) != expected_dynamic:
        raise RuntimeError(
            f"dynamic tuning grid has {len(dynamic)} rows; expected {expected_dynamic}"
        )
    mean_rate = dynamic.mean_rate.to_numpy(dtype=np.float64).reshape(shape)
    passband_column = (
        "passband_weight" if "passband_weight" in dynamic else "response_amp_rms"
    )
    amplitude = dynamic[passband_column].to_numpy(dtype=np.float64).reshape(shape)
    static_shape = (len(units), len(spatial), len(orientation))
    static_rows = table.loc[np.isclose(table.temporal_hz, 0.0)].sort_values(
        ["unit_index", "spatial_cpd", "probe_orientation_deg"]
    )
    if len(static_rows) != int(np.prod(static_shape)):
        raise RuntimeError(
            f"static tuning grid has {len(static_rows)} rows; expected {int(np.prod(static_shape))}"
        )
    static = static_rows.mean_rate.to_numpy(dtype=np.float64).reshape(static_shape)
    for name, value in (("dynamic mean rate", mean_rate), ("phase RMS", amplitude), ("static mean rate", static)):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} tuning grid is incomplete")
    # The Poisson model predicts expected spikes per native 4.17-ms bin.
    # Express signed sensitivity and RMS modulation as spikes/s so the causal
    # projection and the replay outcome share physical rate units.
    rate_hz = float(output_rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("output_rate_hz must be positive")
    signed = rate_hz * (mean_rate - static[:, :, None, :])
    passband = rate_hz * np.clip(amplitude, 0.0, None)
    normalized_passband = passband / np.maximum(
        passband.sum(axis=(1, 2, 3), keepdims=True), EPS
    )
    return {
        "unit_indices": units,
        "spatial_cpd": spatial,
        "temporal_hz": temporal,
        "orientation_deg": orientation,
        "signed_rate_sensitivity": signed,
        "phase_rms": passband,
        "normalized_phase_rms": normalized_passband,
        "static_mean_rate": static,
        "passband_source": np.asarray(passband_column),
    }


def load_tuning_tensors(
    path: Path, *, output_rate_hz: float = DEFAULT_OUTPUT_RATE_HZ
) -> dict[str, np.ndarray]:
    """Load a validated tensor cache, building it once from the grouped CSV."""
    source = Path(path)
    rate_tag = f"{float(output_rate_hz):g}".replace(".", "p")
    cache = (
        source
        if source.suffix == ".npz"
        else source.parent / f"population_tuning_tensors_{rate_tag}hz.npz"
    )
    if cache.exists() and (
        source.suffix == ".npz" or cache.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        with np.load(cache, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    if source.suffix == ".npz":
        raise FileNotFoundError(source)
    result = tuning_tensors(pd.read_csv(source), output_rate_hz=output_rate_hz)
    np.savez_compressed(cache, **result)
    return result


def folded_dpss_mode_power(
    selected_coefficients: np.ndarray,
    frame_rate_hz: float,
    *,
    nw: float = 1.5,
    n_tapers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return folded non-DC temporal power for complex spatial Fourier modes."""
    value = np.asarray(selected_coefficients, dtype=np.complex128)
    if value.ndim != 2 or value.shape[1] < 8:
        raise ValueError("coefficients must have shape [mode,time>=8]")
    value = value - value.mean(axis=1, keepdims=True)
    n_time = value.shape[1]
    signed_hz = np.fft.fftfreq(n_time, d=1.0 / float(frame_rate_hz))
    positive_hz = np.fft.rfftfreq(n_time, d=1.0 / float(frame_rate_hz))[1:]
    tapers = dpss(n_time, NW=float(nw), Kmax=int(n_tapers), sym=False)
    raw = np.zeros((len(value), n_time), dtype=np.float64)
    for taper in tapers:
        transformed = np.fft.fft(value * taper[None], axis=1, norm="ortho")
        raw += np.square(np.abs(transformed))
    raw /= len(tapers)
    folded = np.column_stack(
        [raw[:, np.flatnonzero(np.isclose(np.abs(signed_hz), hz))].sum(axis=1) for hz in positive_hz]
    )
    return positive_hz, folded


def movie_power_cube(
    movie: np.ndarray,
    *,
    flat_index: np.ndarray,
    mode_to_grid: sparse.csr_matrix,
    n_spatial: int,
    n_orientation: int,
    frame_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact rendered SF×TF×orientation dynamic power."""
    value = np.asarray(movie, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError("movie must have shape [time,square y,square x]")
    # Match the model's fixed luminance units rather than dividing out each
    # image's contrast; this preserves the contrast available to drive rate.
    value = (value - 127.0) / 255.0
    value -= value.mean(axis=0, keepdims=True)
    from scipy.signal.windows import tukey

    window_1d = tukey(value.shape[1], alpha=0.15, sym=False)
    spatial_window = np.outer(window_1d, window_1d)
    coefficient = np.fft.fft2(
        value * spatial_window[None], axes=(-2, -1), norm="ortho"
    )
    selected = coefficient.reshape(len(value), -1)[:, flat_index].T
    temporal_hz, mode_power = folded_dpss_mode_power(selected, frame_rate_hz)
    flat_cube = mode_to_grid.T @ mode_power
    cube = np.asarray(flat_cube).reshape(
        n_spatial, n_orientation, len(temporal_hz)
    ).transpose(0, 2, 1)
    return temporal_hz, cube


def interpolate_tuning_temporal(
    tuning: np.ndarray,
    source_hz: np.ndarray,
    target_hz: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    value = np.asarray(tuning, dtype=np.float64)
    source = np.asarray(source_hz, dtype=np.float64)
    target = np.asarray(target_hz, dtype=np.float64)
    output = np.zeros((value.shape[0], value.shape[1], len(target), value.shape[3]))
    valid = (target >= source[0]) & (target <= source[-1])
    if np.any(valid):
        position = np.interp(
            np.log2(target[valid]), np.log2(source), np.arange(len(source), dtype=float)
        )
        lower = np.floor(position).astype(int)
        upper = np.minimum(lower + 1, len(source) - 1)
        fraction = position - lower
        output[:, :, valid] = (
            value[:, :, lower] * (1.0 - fraction)[None, None, :, None]
            + value[:, :, upper] * fraction[None, None, :, None]
        )
    if normalize:
        output = np.clip(output, 0.0, None)
        output /= np.maximum(output.sum(axis=(1, 2, 3), keepdims=True), EPS)
    return output


def spectral_predictors(
    cube: np.ndarray,
    signed_tuning: np.ndarray,
    passband_tuning: np.ndarray,
    signed_controls: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Project one movie spectrum onto full and controlled unit sensitivities."""
    power = np.asarray(cube, dtype=np.float64)
    signed = np.asarray(signed_tuning, dtype=np.float64)
    passband = np.asarray(passband_tuning, dtype=np.float64)
    if signed.shape[1:] != power.shape or passband.shape != signed.shape:
        raise ValueError("power and tuning grids do not match")
    total = float(power.sum())
    joint_signed = np.einsum("sto,usto->u", power, signed, optimize=True)
    joint = np.einsum("sto,usto->u", power, passband, optimize=True)
    controls = signed_controls or signed_projection_controls(signed)
    tf_tuning = controls["tf_marginal"]
    sf_ori_tuning = controls["sf_orientation_marginal"]
    separable_tuning = controls["separable"]
    tf_power = power.sum(axis=(0, 2))
    sf_ori_power = power.sum(axis=1)
    tf = np.einsum("t,ut->u", tf_power, tf_tuning, optimize=True)
    sf_ori = np.einsum("so,uso->u", sf_ori_power, sf_ori_tuning, optimize=True)
    separable = np.einsum("sto,usto->u", power, separable_tuning, optimize=True)
    return {
        "total_dynamic_power": np.full(len(signed), total),
        "joint_signed_rate_drive": joint_signed,
        "joint_passband_power": joint,
        "tf_marginal_power": tf,
        "sf_orientation_marginal_power": sf_ori,
        "separable_passband_power": separable,
    }


def signed_projection_controls(signed_tuning: np.ndarray) -> dict[str, np.ndarray]:
    """Build matched signed marginal and best rank-1 separable controls."""
    signed = np.asarray(signed_tuning, dtype=np.float64)
    if signed.ndim != 4:
        raise ValueError("signed tuning must have shape [unit,sf,tf,orientation]")
    tf = signed.mean(axis=(1, 3))
    sf_orientation = signed.mean(axis=2)
    separable = np.zeros_like(signed)
    for unit in range(len(signed)):
        matrix = signed[unit].transpose(0, 2, 1).reshape(-1, signed.shape[2])
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        rank_one = singular[0] * np.outer(left[:, 0], right[0])
        separable[unit] = rank_one.reshape(
            signed.shape[1], signed.shape[3], signed.shape[2]
        ).transpose(0, 2, 1)
    return {
        "tf_marginal": tf,
        "sf_orientation_marginal": sf_orientation,
        "separable": separable,
    }


def load_signed_projection_controls(
    cache: Path,
    signed_tuning: np.ndarray,
    temporal_hz: np.ndarray,
    *,
    source: Path,
) -> dict[str, np.ndarray]:
    cache = Path(cache)
    if cache.exists() and cache.stat().st_mtime_ns >= Path(source).stat().st_mtime_ns:
        with np.load(cache, allow_pickle=False) as handle:
            if np.array_equal(handle["temporal_hz"], temporal_hz):
                return {
                    "tf_marginal": handle["tf_marginal"],
                    "sf_orientation_marginal": handle["sf_orientation_marginal"],
                    "separable": handle["separable"],
                }
    result = signed_projection_controls(signed_tuning)
    np.savez_compressed(cache, temporal_hz=temporal_hz, **result)
    return result


def map_ssi(rate_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame/unit spatial SSI and mean rate."""
    rate = np.clip(np.asarray(rate_map, dtype=np.float64), 0.0, None)
    mean = rate.mean(axis=(-2, -1))
    gain = rate / np.maximum(mean[..., None, None], EPS)
    information = np.mean(gain * np.log2(np.maximum(gain, EPS)), axis=(-2, -1))
    return information, mean


def lagged_movie_view(movie: torch.Tensor, n_lags: int) -> torch.Tensor:
    """Return newest-first lag histories without materializing all windows."""
    value = movie.unsqueeze(1) if movie.ndim == 3 else movie
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("movie must have shape [time,y,x] or [time,1,y,x]")
    if len(value) < int(n_lags):
        raise ValueError("movie is shorter than its requested lag history")
    # Tensor.unfold appends the window dimension: [endpoint,channel,y,x,lag].
    # Reverse that final axis so lag zero is the current frame, exactly as in
    # the training embedder, while retaining a cheap strided view.
    return value.unfold(0, int(n_lags), 1).permute(0, 1, 4, 2, 3).flip(2)


@torch.no_grad()
def score_retinal_movie(
    scorer: RealTraceMatrixScorer,
    movie: np.ndarray,
    unit_indices: np.ndarray,
    *,
    history_samples: int,
    analysis_samples: int,
    batch_size: int,
    retain_example: bool,
) -> dict[str, np.ndarray]:
    """Score the final analysis interval using the supplied real history."""
    value = torch.from_numpy(np.asarray(movie, dtype=np.float32)).to(scorer.device)
    histories = lagged_movie_view(value, int(history_samples))
    if len(histories) < analysis_samples + 1:
        raise RuntimeError("retinal movie does not contain real history plus analysis")
    # The first embedded endpoint is the last history sample.  The following
    # 240 endpoints are exactly the one-second analysis interval.
    histories = histories[1 : 1 + int(analysis_samples)]
    histories = (histories - 127.0) / 255.0
    information_chunks, rate_chunks = [], []
    example = None
    unit_tensor = torch.as_tensor(unit_indices, device=scorer.device, dtype=torch.long)
    for start in range(0, len(histories), int(batch_size)):
        x = histories[start : start + int(batch_size)]
        full = scorer._compute_rate_map(x)
        rr = scorer.apply_population_view(full, scorer.population_view).clamp_min(0.0)
        rr = rr.index_select(1, unit_tensor).float().cpu().numpy()
        information, mean = map_ssi(rr)
        information_chunks.append(information)
        rate_chunks.append(mean)
        if retain_example and example is None:
            example = rr[0]
    information = np.concatenate(information_chunks)
    rate = np.concatenate(rate_chunks)
    expected = rate.sum(axis=0)
    pooled_ssi = np.sum(information * rate, axis=0) / np.maximum(rate.sum(axis=0), EPS)
    return {
        "mean_rate": float(scorer.output_rate_hz) * rate.mean(axis=0),
        "expected_spikes": expected,
        "map_ssi": pooled_ssi,
        "example_rate_map": np.asarray(example) if example is not None else np.empty((0,)),
    }


def selected_rows(frame: pd.DataFrame, start: int, stop: int, maximum: int) -> pd.DataFrame:
    left = max(int(start), 0)
    right = len(frame) if int(stop) <= 0 else min(int(stop), len(frame))
    result = frame.iloc[left:right].copy()
    if maximum > 0:
        result = result.iloc[: int(maximum)].copy()
    return result


def evenly_sampled_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic rows spanning the complete supplied frame."""
    requested = int(count)
    if requested <= 0 or requested >= len(frame):
        return frame.copy()
    indices = np.unique(np.round(np.linspace(0, len(frame) - 1, requested)).astype(int))
    if len(indices) != requested:
        raise RuntimeError("even sampling produced duplicated row indices")
    return frame.iloc[indices].copy()


def explicitly_selected_rows(
    frame: pd.DataFrame,
    selection_file: Path,
    *,
    index_column: str,
) -> pd.DataFrame:
    """Resolve an explicit selection while preserving its declared order."""
    selection = pd.read_csv(selection_file)
    if index_column not in selection.columns:
        raise ValueError(f"{selection_file} must contain {index_column}")
    requested = selection[index_column].to_numpy(dtype=int)
    if len(requested) == 0:
        raise ValueError(f"{selection_file} contains no {index_column} values")
    if len(np.unique(requested)) != len(requested):
        raise ValueError(f"{selection_file} contains duplicate {index_column} values")
    if index_column not in frame.columns:
        raise ValueError(f"source table must contain {index_column}")
    if frame[index_column].duplicated().any():
        raise ValueError(f"source table contains duplicate {index_column} values")
    indexed = frame.set_index(index_column, drop=False)
    missing = [int(value) for value in requested if int(value) not in indexed.index]
    if missing:
        raise ValueError(
            f"{selection_file} requests missing {index_column} values: {missing}"
        )
    return indexed.loc[requested].reset_index(drop=True)


def transform_trace_bank(
    traces_xy: np.ndarray,
    trace_indices: np.ndarray,
    *,
    transform: str,
    history_samples: int,
    analysis_samples: int,
    seed: int,
) -> np.ndarray:
    """Apply a matched spatial/temporal control to a trace bank.

    Temporal controls operate on the exact scored interval. Their causal
    prefix is the preceding part of a circular extension, so the transformed
    interval has no artificial boundary jump and retains its declared Fourier
    invariant.
    """
    traces = np.asarray(traces_xy, dtype=np.float32)
    indices = np.asarray(trace_indices, dtype=int)
    if traces.ndim != 3 or traces.shape[-1] != 2 or len(traces) != len(indices):
        raise ValueError("traces and indices must match [trace,time,2]")
    history = int(history_samples)
    analysis = int(analysis_samples)
    if traces.shape[1] < history + analysis:
        raise ValueError("traces are shorter than history plus analysis")
    if transform in ("identity", "movie_phase_scramble"):
        return traces.copy()
    if transform == "rotate90":
        return np.stack((-traces[..., 1], traces[..., 0]), axis=-1)
    scored = traces[:, -analysis:]
    if transform == "time_reverse":
        transformed = scored[:, ::-1].copy()
    elif transform == "phase_scramble":
        transformed = np.stack(
            [
                phase_scramble_trace(trace, seed=int(seed) + int(trace_index))
                for trace_index, trace in zip(indices, scored)
            ]
        ).astype(np.float32)
    else:
        raise ValueError(f"unknown trace transform: {transform}")
    return np.concatenate((transformed[:, -history:], transformed), axis=1)


def spatiotemporal_phase_scramble_movie(
    analysis_movie: np.ndarray,
    *,
    seed: int,
    device: str | None = None,
) -> tuple[np.ndarray, float]:
    """Randomize dynamic joint phase while preserving every SF×TF amplitude.

    The phase of a real white-noise 3-D Fourier transform supplies an exactly
    Hermitian phase mask, so inverse transformation remains real. The TF=0
    plane is left untouched: the time-averaged natural image is therefore
    identical, while dynamic phase is independently changed across spatial and
    temporal modes.
    """
    movie = np.asarray(analysis_movie, dtype=np.float64)
    if movie.ndim != 3 or movie.shape[0] < 8:
        raise ValueError("analysis_movie must have shape [time>=8,y,x]")
    if device is not None:
        value = torch.as_tensor(movie, dtype=torch.float64, device=device)
        coefficient = torch.fft.fftn(value)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            value.shape, dtype=torch.float64, device=device, generator=generator
        )
        noise_coefficient = torch.fft.fftn(noise)
        phase = noise_coefficient / noise_coefficient.abs().clamp_min(EPS)
        phase[0] = torch.ones_like(phase[0])
        scrambled_complex = torch.fft.ifftn(coefficient * phase)
        imaginary_error = float(scrambled_complex.imag.abs().max().item())
        if imaginary_error > 1e-8:
            raise RuntimeError(
                "Hermitian phase mask failed to produce a real movie: "
                f"max imaginary residual={imaginary_error:.3g}"
            )
        scrambled_tensor = scrambled_complex.real.float()
        observed = torch.fft.fftn(scrambled_tensor.double()).abs()
        expected = coefficient.abs()
        relative_error = float(
            (observed - expected).abs().max().div(expected.max().clamp_min(EPS)).item()
        )
        return scrambled_tensor.cpu().numpy(), relative_error
    coefficient = np.fft.fftn(movie)
    rng = np.random.default_rng(int(seed))
    noise_coefficient = np.fft.fftn(rng.standard_normal(movie.shape))
    phase = noise_coefficient / np.maximum(np.abs(noise_coefficient), EPS)
    phase[0] = 1.0
    scrambled_complex = np.fft.ifftn(coefficient * phase)
    imaginary_error = float(np.max(np.abs(scrambled_complex.imag)))
    if imaginary_error > 1e-8:
        raise RuntimeError(
            "Hermitian phase mask failed to produce a real movie: "
            f"max imaginary residual={imaginary_error:.3g}"
        )
    scrambled = scrambled_complex.real.astype(np.float32)
    observed = np.abs(np.fft.fftn(scrambled.astype(np.float64)))
    expected = np.abs(coefficient)
    relative_error = float(
        np.max(np.abs(observed - expected)) / max(float(np.max(expected)), EPS)
    )
    return scrambled, relative_error


def main() -> int:
    args = parse_args()
    scales = np.asarray(args.motion_scales, dtype=float)
    if len(scales) < 2 or scales[0] != 0.0 or np.any(scales < 0):
        raise ValueError("motion scales must begin with stabilized 0 and be nonnegative")
    image_table = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    trace_table = pd.read_csv(args.trace_bank / "trace_table.csv").sort_values("trace_index").reset_index(drop=True)
    trace_path = args.trace_bank / f"trace_xy_{args.trace_kind}.npy"
    traces = np.load(trace_path, mmap_mode="r")
    if traces.shape != (len(trace_table), args.history_samples + args.analysis_samples, 2):
        raise ValueError(f"unexpected trace-bank shape {traces.shape}")
    if args.image_indices_file is not None:
        if any(
            value
            for value in (
                args.image_start,
                args.image_stop,
                args.max_images,
                args.sample_images_evenly,
            )
        ):
            raise ValueError(
                "--image-indices-file cannot be combined with image range or sampling options"
            )
        images = explicitly_selected_rows(
            image_table, args.image_indices_file, index_column="image_index"
        )
    else:
        images = selected_rows(image_table, args.image_start, args.image_stop, args.max_images)
        images = evenly_sampled_rows(images, int(args.sample_images_evenly))
    if args.trace_indices_file is not None:
        if any(
            value
            for value in (
                args.trace_start,
                args.trace_stop,
                args.max_traces,
                args.sample_traces_evenly,
            )
        ):
            raise ValueError(
                "--trace-indices-file cannot be combined with trace range or sampling options"
            )
        trace_rows = explicitly_selected_rows(
            trace_table, args.trace_indices_file, index_column="trace_index"
        )
    else:
        trace_rows = selected_rows(trace_table, args.trace_start, args.trace_stop, args.max_traces)
        trace_rows = evenly_sampled_rows(trace_rows, int(args.sample_traces_evenly))
    trace_indices = trace_rows.trace_index.to_numpy(dtype=int)
    selected_traces = np.asarray(traces[trace_indices], dtype=np.float32)
    selected_traces = transform_trace_bank(
        selected_traces,
        trace_indices,
        transform=str(args.trace_transform),
        history_samples=int(args.history_samples),
        analysis_samples=int(args.analysis_samples),
        seed=int(args.trace_transform_seed),
    )
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_config.resolve(),
        population_spec_dir=args.population_spec_dir.resolve(),
        population_version=str(args.population_version),
        device=args.device,
        strict=True,
        mcfarland_outputs=args.mcfarland_outputs.resolve(),
    )
    if scorer.input_rate_hz != scorer.output_rate_hz:
        raise RuntimeError(
            "direct causal-chain replay currently requires equal model input and output "
            f"rates; selected model has {scorer.input_rate_hz} Hz input and "
            f"{scorer.output_rate_hz} Hz output"
        )
    if int(args.history_samples) != int(scorer.n_lags):
        raise RuntimeError(
            f"--history-samples={args.history_samples} does not match the selected "
            f"model history ({scorer.n_lags})"
        )
    tuning = load_tuning_tensors(
        args.tuning_table, output_rate_hz=float(scorer.output_rate_hz)
    )
    units = tuning["unit_indices"]
    if len(units) == 0 or int(np.min(units)) < 0 or int(np.max(units)) >= scorer.n_units:
        raise RuntimeError(
            "tuning-table unit indices do not fit the selected population view "
            f"({scorer.n_units} units)"
        )

    grid = frequency_grid()
    distributor, resolved_modes = mode_to_grid_matrix(
        np.asarray(grid["kxy"]), tuning["spatial_cpd"], tuning["orientation_deg"]
    )
    shape = (len(images), len(trace_rows), len(scales), len(units))
    predictor_names = (
        "total_dynamic_power",
        "joint_signed_rate_drive",
        "joint_passband_power",
        "tf_marginal_power",
        "sf_orientation_marginal_power",
        "separable_passband_power",
    )
    predictors = {name: np.zeros(shape, dtype=np.float32) for name in predictor_names}
    mean_rate = np.zeros(shape, dtype=np.float32)
    expected_spikes = np.zeros(shape, dtype=np.float32)
    map_information = np.zeros(shape, dtype=np.float32)
    average_power = None
    temporal_hz = None
    example_maps = []
    example_power = []
    example_movie_frames = None
    transform_invariant_errors: list[float] = []
    phase_out_of_range_fractions: list[float] = []
    phase_max_normalized_excursions: list[float] = []
    phase_mean_image_errors: list[float] = []
    canvas_cache: dict = {}
    for image_local, (_, image_row) in enumerate(images.iterrows()):
        patch, _ = extract_patch(image_row, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px))
        for trace_local, trace in enumerate(selected_traces):
            for scale_index, scale in enumerate(scales):
                # With behavior neutralized, stabilization is identical for
                # every trace paired with the same image. Score it once and
                # broadcast across the factorial trace bank.
                if scale_index == 0 and trace_local > 0:
                    for name in predictor_names:
                        predictors[name][image_local, trace_local, scale_index] = predictors[name][image_local, 0, scale_index]
                    mean_rate[image_local, trace_local, scale_index] = mean_rate[image_local, 0, scale_index]
                    expected_spikes[image_local, trace_local, scale_index] = expected_spikes[image_local, 0, scale_index]
                    map_information[image_local, trace_local, scale_index] = map_information[image_local, 0, scale_index]
                    average_power[scale_index] += 0.0
                    continue
                movie = render_movies(
                    patch,
                    (trace[None] * float(scale)),
                    device=args.device,
                )[0]
                if args.trace_transform == "movie_phase_scramble":
                    scrambled, invariant_error = spatiotemporal_phase_scramble_movie(
                        movie[-int(args.analysis_samples) :],
                        seed=(
                            int(args.trace_transform_seed)
                            + 1000003 * int(image_row.image_index)
                            + 1009 * int(trace_indices[trace_local])
                            + int(scale_index)
                        ),
                        device=args.device,
                    )
                    transform_invariant_errors.append(invariant_error)
                    phase_out_of_range_fractions.append(
                        float(np.mean((scrambled < 0.0) | (scrambled > 255.0)))
                    )
                    phase_max_normalized_excursions.append(
                        float(
                            max(
                                np.max(np.maximum(-scrambled, 0.0)),
                                np.max(np.maximum(scrambled - 255.0, 0.0)),
                            )
                            / 255.0
                        )
                    )
                    phase_mean_image_errors.append(
                        float(
                            np.max(
                                np.abs(
                                    scrambled.mean(axis=0)
                                    - movie[-int(args.analysis_samples) :].mean(axis=0)
                                )
                            )
                            / 255.0
                        )
                    )
                    movie = np.concatenate(
                        (scrambled[-int(args.history_samples) :], scrambled), axis=0
                    )
                analysis_movie = movie[-int(args.analysis_samples) :]
                frequency, cube = movie_power_cube(
                    analysis_movie,
                    flat_index=np.asarray(grid["flat_index"], dtype=int),
                    mode_to_grid=distributor,
                    n_spatial=len(tuning["spatial_cpd"]),
                    n_orientation=len(tuning["orientation_deg"]),
                    frame_rate_hz=float(scorer.input_rate_hz),
                )
                if temporal_hz is None:
                    temporal_hz = frequency
                    signed = interpolate_tuning_temporal(
                        tuning["signed_rate_sensitivity"], tuning["temporal_hz"], frequency, normalize=False
                    )
                    passband = interpolate_tuning_temporal(
                        tuning["phase_rms"], tuning["temporal_hz"], frequency, normalize=True
                    )
                    signed_controls = load_signed_projection_controls(
                        args.tuning_table.parent
                        / f"population_signed_projection_controls_{scorer.input_rate_hz}hz.npz",
                        signed,
                        frequency,
                        source=args.tuning_table,
                    )
                    average_power = np.zeros((len(scales), *cube.shape), dtype=np.float64)
                elif not np.array_equal(temporal_hz, frequency):
                    raise RuntimeError("temporal spectrum grid changed")
                projection = spectral_predictors(
                    cube, signed, passband, signed_controls=signed_controls
                )
                for name, value in projection.items():
                    predictors[name][image_local, trace_local, scale_index] = value
                score = score_retinal_movie(
                    scorer,
                    movie,
                    units,
                    history_samples=int(args.history_samples),
                    analysis_samples=int(args.analysis_samples),
                    batch_size=int(args.frame_batch_size),
                    retain_example=bool(args.save_example_maps and image_local == 0 and trace_local == 0),
                )
                mean_rate[image_local, trace_local, scale_index] = score["mean_rate"]
                expected_spikes[image_local, trace_local, scale_index] = score["expected_spikes"]
                map_information[image_local, trace_local, scale_index] = score["map_ssi"]
                average_power[scale_index] += cube
                if args.save_example_maps and image_local == 0 and trace_local == 0:
                    example_maps.append(score["example_rate_map"])
                    example_power.append(cube)
                    if np.isclose(scale, 1.0):
                        frame_rows = np.unique(
                            np.round(np.linspace(0, len(analysis_movie) - 1, 5)).astype(int)
                        )
                        example_movie_frames = analysis_movie[frame_rows].astype(np.float32)
            print(
                f"causal chain image {image_local + 1}/{len(images)} trace "
                f"{trace_local + 1}/{len(trace_rows)}",
                flush=True,
            )
    if average_power is None or temporal_hz is None:
        raise RuntimeError("no causal-chain conditions were scored")
    average_power /= float(len(images) * len(trace_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "causal_chain_shard.npz"
    np.savez_compressed(
        archive,
        image_indices=images.image_index.to_numpy(dtype=int),
        trace_indices=trace_indices,
        motion_scales=scales,
        unit_indices=units,
        spatial_cpd=tuning["spatial_cpd"],
        temporal_hz=temporal_hz,
        orientation_deg=tuning["orientation_deg"],
        mean_rate=mean_rate,
        expected_spikes=expected_spikes,
        map_ssi=map_information,
        average_power=average_power.astype(np.float32),
        example_power=np.asarray(example_power, dtype=np.float32),
        example_rate_maps=np.asarray(example_maps, dtype=np.float32),
        example_movie_frames=(
            np.asarray(example_movie_frames, dtype=np.float32)
            if example_movie_frames is not None
            else np.empty((0,), dtype=np.float32)
        ),
        example_trace_xy=np.asarray(selected_traces[0], dtype=np.float32),
        **predictors,
    )
    summary = {
        "analysis": "renderer-faithful zero-behavior retinal causal-chain shard",
        "model_label": str(args.model_label or args.checkpoint.stem),
        "population_version": str(args.population_version),
        "behavior": "fixed all-zero 42-dimensional vector for every condition",
        "trace_kind": args.trace_kind,
        "trace_transform": args.trace_transform,
        "trace_transform_seed": (
            int(args.trace_transform_seed)
            if args.trace_transform in ("phase_scramble", "movie_phase_scramble")
            else None
        ),
        "transform_history_policy": (
            "measured history"
            if args.trace_transform in ("identity", "rotate90")
            else "circular prefix from transformed analysis interval"
        ),
        "transform_invariant": (
            {
                "quantity": "every rendered-movie joint SFx-SFy-TF amplitude; TF=0 mean image also unchanged",
                "maximum_relative_error": float(max(transform_invariant_errors, default=0.0)),
                "gate": 1e-6,
                "gate_pass": bool(max(transform_invariant_errors, default=0.0) < 1e-6),
                "mean_image_max_absolute_error_normalized": float(
                    max(phase_mean_image_errors, default=0.0)
                ),
                "mean_image_gate_pass": bool(
                    max(phase_mean_image_errors, default=0.0) < 1e-6
                ),
                "median_fraction_outside_original_uint8_range": float(
                    np.median(phase_out_of_range_fractions)
                ),
                "maximum_normalized_range_overshoot": float(
                    max(phase_max_normalized_excursions, default=0.0)
                ),
                "range_support_gate": 0.15,
                "range_support_gate_pass": bool(
                    max(phase_max_normalized_excursions, default=0.0) <= 0.15
                ),
            }
            if args.trace_transform == "movie_phase_scramble"
            else None
        ),
        "n_images": int(len(images)),
        "n_traces": int(len(trace_rows)),
        "n_units": int(len(units)),
        "image_indices_file": (
            str(args.image_indices_file.resolve())
            if args.image_indices_file is not None
            else None
        ),
        "trace_indices_file": (
            str(args.trace_indices_file.resolve())
            if args.trace_indices_file is not None
            else None
        ),
        "motion_scales": scales.tolist(),
        "image_indices": images.image_index.astype(int).tolist(),
        "trace_indices": trace_indices.tolist(),
        "resolved_spatial_fourier_modes": int(np.count_nonzero(resolved_modes)),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256(args.dataset_config.resolve()),
        "tuning_table": str(args.tuning_table.resolve()),
        "tuning_table_sha256": sha256(args.tuning_table.resolve()),
        "trace_bank_manifest": str((args.trace_bank / "manifest.json").resolve()),
        "spectrum": "exact rendered 151px movie; spatial Tukey; temporal mean removed; DPSS NW=1.5 K=2; folded directions",
        "model_input_rate_hz": int(scorer.input_rate_hz),
        "model_output_rate_hz": int(scorer.output_rate_hz),
        "model_history_samples": int(scorer.n_lags),
        "analysis_samples": int(args.analysis_samples),
        "analysis_seconds": float(args.analysis_samples / scorer.output_rate_hz),
        "rate_units": (
            "spikes/s; native Poisson outputs multiplied by the selected model's "
            "output rate"
        ),
        "expected_spike_definition": (
            "sum of native expected spike counts across the exact analysis interval"
        ),
        "rate_projection_controls": (
            "all rate predictors derive from the identical signed dynamic-minus-TF0 "
            "tuning tensor; separable is the best rank-1 (SF×orientation)-by-TF "
            "approximation per unit; phase RMS is used only for nonnegative passband support"
        ),
        "archive": str(archive.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
