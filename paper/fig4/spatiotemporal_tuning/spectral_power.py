"""Exact rendered-movie SF×TF power and measured-passband projections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal.windows import dpss, tukey

from paper.fig4.upstream.real_trace_matrix.model import OUT_SIZE, PPD


EPS = 1.0e-12
DEFAULT_OUTPUT_RATE_HZ = 240.0


def log_edges(centers_linear: np.ndarray) -> np.ndarray:
    centers = np.log2(np.asarray(centers_linear, dtype=np.float64))
    if centers.ndim != 1 or len(centers) < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("log-frequency centers must be a finite one-dimensional array")
    return np.concatenate(
        (
            [centers[0] - 0.5 * (centers[1] - centers[0])],
            0.5 * (centers[:-1] + centers[1:]),
            [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
        )
    )


def log_interpolation_weights(
    values: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def circular_orientation_weights(
    kxy: np.ndarray, orientations_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orientations = np.asarray(orientations_deg, dtype=np.float64)
    if len(orientations) < 2 or not np.allclose(
        np.diff(orientations), np.diff(orientations)[0]
    ):
        raise ValueError("orientation probes must be uniformly spaced")
    step = float(np.diff(orientations)[0])
    if not np.isclose(step * len(orientations), 180.0):
        raise ValueError("orientation probes must tile the 180-degree period")
    normal = np.degrees(np.arctan2(kxy[:, 1], kxy[:, 0]))
    position = np.mod(normal - 90.0 - orientations[0], 180.0) / step
    lower = np.floor(position).astype(int) % len(orientations)
    fraction = position - np.floor(position)
    upper = (lower + 1) % len(orientations)
    return lower, upper, 1.0 - fraction, fraction


def frequency_grid() -> dict[str, np.ndarray]:
    """Return physical Fourier modes supported by the exact model crop."""
    height, width = OUT_SIZE
    if height != width:
        raise ValueError(OUT_SIZE)
    n = int(height)
    axis = np.fft.fftfreq(n, d=1.0 / float(PPD))
    fy, fx = np.meshgrid(axis, axis, indexing="ij")
    kx, ky = fx, -fy
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


def spatial_frequency_grid(
    size: int, *, maximum_cpd: float
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.fft.fftfreq(int(size), d=1.0 / float(PPD))
    ky, kx = np.meshgrid(-axis, axis, indexing="ij")
    kxy = np.column_stack((kx.ravel(), ky.ravel()))
    radial = np.linalg.norm(kxy, axis=1)
    keep = (radial > 0) & (radial <= float(maximum_cpd) * 1.15)
    return kxy[keep], np.flatnonzero(keep)


def mode_to_grid_matrix(
    kxy: np.ndarray,
    spatial_cpd: np.ndarray,
    orientations_deg: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    radial = np.linalg.norm(kxy, axis=1)
    sf0, sf1, sw0, sw1, resolved = log_interpolation_weights(radial, spatial_cpd)
    ori0, ori1, ow0, ow1 = circular_orientation_weights(kxy, orientations_deg)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    mode = np.arange(len(kxy), dtype=int)
    for sf_index, sf_weight in ((sf0, sw0), (sf1, sw1)):
        for ori_index, ori_weight in ((ori0, ow0), (ori1, ow1)):
            weight = sf_weight * ori_weight * resolved
            keep = weight > 0
            rows.append(mode[keep])
            columns.append(sf_index[keep] * len(orientations_deg) + ori_index[keep])
            values.append(weight[keep])
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(kxy), len(spatial_cpd) * len(orientations_deg)),
    ).tocsr()
    return matrix, resolved


def tuning_tensors(
    table: pd.DataFrame, *, output_rate_hz: float = DEFAULT_OUTPUT_RATE_HZ
) -> dict[str, np.ndarray]:
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
    temporal = np.sort(
        table.loc[table.temporal_hz.gt(0), "temporal_hz"].unique().astype(float)
    )
    orientation = np.sort(table.probe_orientation_deg.unique().astype(float))
    shape = (len(units), len(spatial), len(temporal), len(orientation))
    keys = ["unit_index", "spatial_cpd", "temporal_hz", "probe_orientation_deg"]
    if table.duplicated(keys).any():
        raise RuntimeError("tuning table contains duplicated grid cells")
    dynamic = table.loc[table.temporal_hz.gt(0)].sort_values(keys)
    if len(dynamic) != int(np.prod(shape)):
        raise RuntimeError("dynamic tuning grid is incomplete")
    mean_rate = dynamic.mean_rate.to_numpy(dtype=np.float64).reshape(shape)
    passband_column = "passband_weight" if "passband_weight" in dynamic else "response_amp_rms"
    amplitude = dynamic[passband_column].to_numpy(dtype=np.float64).reshape(shape)
    static_shape = (len(units), len(spatial), len(orientation))
    static_rows = table.loc[np.isclose(table.temporal_hz, 0.0)].sort_values(
        ["unit_index", "spatial_cpd", "probe_orientation_deg"]
    )
    if len(static_rows) != int(np.prod(static_shape)):
        raise RuntimeError("static tuning grid is incomplete")
    static = static_rows.mean_rate.to_numpy(dtype=np.float64).reshape(static_shape)
    for name, value in (("dynamic mean rate", mean_rate), ("passband", amplitude), ("static mean rate", static)):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} tuning grid is incomplete")
    rate_hz = float(output_rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("output_rate_hz must be positive")
    signed = rate_hz * (mean_rate - static[:, :, None, :])
    passband = rate_hz * np.clip(amplitude, 0.0, None)
    normalized = passband / np.maximum(passband.sum(axis=(1, 2, 3), keepdims=True), EPS)
    return {
        "unit_indices": units,
        "spatial_cpd": spatial,
        "temporal_hz": temporal,
        "orientation_deg": orientation,
        "signed_rate_sensitivity": signed,
        "phase_rms": passband,
        "normalized_phase_rms": normalized,
        "static_mean_rate": static,
        "passband_source": np.asarray(passband_column),
    }


def load_tuning_tensors(
    path: Path, *, output_rate_hz: float = DEFAULT_OUTPUT_RATE_HZ
) -> dict[str, np.ndarray]:
    source = Path(path)
    rate_tag = f"{float(output_rate_hz):g}".replace(".", "p")
    cache = source if source.suffix == ".npz" else source.parent / f"population_tuning_tensors_{rate_tag}hz.npz"
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
    value = (value - 127.0) / 255.0
    value -= value.mean(axis=0, keepdims=True)
    window_1d = tukey(value.shape[1], alpha=0.15, sym=False)
    coefficient = np.fft.fft2(
        value * np.outer(window_1d, window_1d)[None],
        axes=(-2, -1),
        norm="ortho",
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


def signed_projection_controls(signed_tuning: np.ndarray) -> dict[str, np.ndarray]:
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


def spectral_predictors(
    cube: np.ndarray,
    signed_tuning: np.ndarray,
    passband_tuning: np.ndarray,
    signed_controls: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    power = np.asarray(cube, dtype=np.float64)
    signed = np.asarray(signed_tuning, dtype=np.float64)
    passband = np.asarray(passband_tuning, dtype=np.float64)
    if signed.shape[1:] != power.shape or passband.shape != signed.shape:
        raise ValueError("power and tuning grids do not match")
    total = float(power.sum())
    controls = signed_controls or signed_projection_controls(signed)
    return {
        "total_dynamic_power": np.full(len(signed), total),
        "joint_signed_rate_drive": np.einsum("sto,usto->u", power, signed, optimize=True),
        "joint_passband_power": np.einsum("sto,usto->u", power, passband, optimize=True),
        "tf_marginal_power": np.einsum(
            "t,ut->u", power.sum(axis=(0, 2)), controls["tf_marginal"], optimize=True
        ),
        "sf_orientation_marginal_power": np.einsum(
            "so,uso->u",
            power.sum(axis=1),
            controls["sf_orientation_marginal"],
            optimize=True,
        ),
        "separable_passband_power": np.einsum(
            "sto,usto->u", power, controls["separable"], optimize=True
        ),
    }
