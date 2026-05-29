"""Model lag-cube response and cumulative information helpers."""
from __future__ import annotations

from typing import Any

import numpy as np

from .common import DT, N_LAGS, PPD, compute_rate_map_batched
from .information import (
    event_code_information,
    event_code_information_pattern_only,
    expected_counts_from_rates,
    finite_difference_derivatives,
    fisher_by_time,
    spatial_single_spike_information,
)


def gather_population_rates(rate_map: Any, unit_ids: np.ndarray, t_max: int) -> np.ndarray:
    """Gather sampled ``(unit, row, col)`` rates from a full model rate map."""
    import torch

    unit_ids = np.asarray(unit_ids, dtype=np.int64)
    u = torch.as_tensor(unit_ids[:, 0], dtype=torch.long)
    r = torch.as_tensor(unit_ids[:, 1], dtype=torch.long)
    c = torch.as_tensor(unit_ids[:, 2], dtype=torch.long)
    return rate_map[:t_max, u, r, c].detach().cpu().numpy().astype(np.float32)


def block_current_samples(t_max: int, n_lags: int = N_LAGS) -> np.ndarray:
    """Current-sample indices for overlapping model-lag cubes.

    The retinal rendering code pads the lag history before alignment, so a
    128-sample eye trace yields 128 valid model inputs.  If callers pass raw,
    unpadded lag embeddings, the equivalent trace length is already reflected
    in ``t_max`` as ``movie_length - n_lags + 1``.
    """
    del n_lags
    if int(t_max) <= 0:
        raise ValueError(f"Need t_max > 0, got t_max={t_max}")
    return np.arange(int(t_max), dtype=np.int32)


def block_endpoint_lag_cubes(cubes: np.ndarray, n_lags: int = N_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """Return every aligned lag cube and its current-frame sample index.

    This function keeps the old name for compatibility with existing scripts,
    but it no longer decimates to non-overlapping 32-frame blocks.
    """
    arr = np.asarray(cubes, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected lag cubes with shape (time, lag, height, width), got {arr.shape}")
    if arr.shape[1] != n_lags:
        raise ValueError(f"Expected {n_lags} lag frames, got {arr.shape[1]}")
    current = block_current_samples(arr.shape[0], n_lags=n_lags)
    return arr.astype(np.float32), current


def lag_cubes_to_stim(cubes: np.ndarray):
    """Convert pixel-valued lag cubes to the normalized tensor expected by the twin."""
    import torch

    arr = np.asarray(cubes, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected lag cubes with shape (time, lag, height, width), got {arr.shape}")
    return (torch.from_numpy(arr[:, None]) - 127.0) / 255.0


def lag_cubes_to_shifted_stim(
    cubes: np.ndarray,
    dx_deg: float,
    dy_deg: float,
    *,
    device: Any | None = None,
    ppd: float = PPD,
):
    """Normalize lag cubes after a small spatial shift on the model device."""
    import torch
    import torch.nn.functional as F

    arr = np.asarray(cubes, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected lag cubes with shape (time, lag, height, width), got {arr.shape}")
    tensor = torch.from_numpy(arr)
    if device is not None:
        tensor = tensor.to(device)
    if float(dx_deg) != 0.0 or float(dy_deg) != 0.0:
        t_max, n_lags, height, width = tensor.shape
        flat = tensor.reshape(t_max * n_lags, 1, height, width)
        dtype = flat.dtype
        dev = flat.device
        y = torch.linspace(-1.0, 1.0, height, device=dev, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=dev, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dx_norm = 2.0 * float(dx_deg) * ppd / max(width - 1, 1)
        dy_norm = 2.0 * float(dy_deg) * ppd / max(height - 1, 1)
        grid = torch.stack((xx - dx_norm, yy - dy_norm), dim=-1)
        grid = grid.unsqueeze(0).expand(flat.shape[0], -1, -1, -1)
        flat = F.grid_sample(
            flat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        tensor = flat.reshape(t_max, n_lags, height, width)
    return (tensor[:, None] - 127.0) / 255.0


def run_lag_cube_rates(
    model: Any,
    population: Any,
    device: Any,
    cubes: np.ndarray,
    *,
    batch_size: int,
    return_rate_map: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run one lag-cube tensor through the twin and gather population rates."""
    stim = lag_cubes_to_stim(cubes)
    rate_map = compute_rate_map_batched(
        model,
        population.readout.to(device),
        stim,
        batch_size=batch_size,
    )
    rates = gather_population_rates(rate_map, population.unit_ids, int(cubes.shape[0]))
    if return_rate_map:
        return rates.astype(np.float32), rate_map.detach().cpu().numpy().astype(np.float32)
    del rate_map
    return rates.astype(np.float32), None


def _shift_key(dx: float, dy: float) -> tuple[float, float]:
    return (round(float(dx), 8), round(float(dy), 8))


def unique_shifts(*shift_groups: np.ndarray) -> np.ndarray:
    """Return ordered unique ``(dx, dy)`` rows."""
    rows: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for shifts in shift_groups:
        for dx, dy in np.asarray(shifts, dtype=np.float64).reshape(-1, 2):
            key = _shift_key(float(dx), float(dy))
            if key not in seen:
                seen.add(key)
                rows.append(key)
    return np.asarray(rows, dtype=np.float64)


def run_shifted_lag_cube_rates(
    model: Any,
    population: Any,
    device: Any,
    cubes: np.ndarray,
    shifts_deg: np.ndarray,
    *,
    batch_size: int,
) -> dict[tuple[float, float], np.ndarray]:
    """Run shifted lag cubes through the twin for FI and SSI state grids."""
    import torch

    arr = np.asarray(cubes, dtype=np.float32)
    readout = population.readout.to(device)
    out: dict[tuple[float, float], np.ndarray] = {}
    for dx, dy in unique_shifts(np.asarray(shifts_deg, dtype=np.float64)):
        stim = lag_cubes_to_shifted_stim(arr, float(dx), float(dy), device=device)
        rate_map = compute_rate_map_batched(
            model,
            readout,
            stim,
            batch_size=batch_size,
        )
        out[_shift_key(dx, dy)] = gather_population_rates(
            rate_map,
            population.unit_ids,
            int(arr.shape[0]),
        ).astype(np.float32)
        del rate_map, stim
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def run_shifted_lag_cube_rate_maps(
    model: Any,
    population: Any,
    device: Any,
    cubes: np.ndarray,
    shifts_deg: np.ndarray,
    *,
    batch_size: int,
) -> dict[tuple[float, float], np.ndarray]:
    """Run shifted lag cubes and keep the full convolutional rate maps."""
    import torch

    arr = np.asarray(cubes, dtype=np.float32)
    readout = population.readout.to(device)
    out: dict[tuple[float, float], np.ndarray] = {}
    for dx, dy in unique_shifts(np.asarray(shifts_deg, dtype=np.float64)):
        stim = lag_cubes_to_shifted_stim(arr, float(dx), float(dy), device=device)
        rate_map = compute_rate_map_batched(
            model,
            readout,
            stim,
            batch_size=batch_size,
        )
        out[_shift_key(dx, dy)] = rate_map.detach().cpu().numpy().astype(np.float32)
        del rate_map, stim
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def cross_shift_grid(max_arcmin: float, step_arcmin: float) -> np.ndarray:
    """Return center plus cardinal-axis shifts in degrees."""
    max_deg = float(max_arcmin) / 60.0
    step_deg = float(step_arcmin) / 60.0
    vals = np.arange(step_deg, max_deg + 0.5 * step_deg, step_deg, dtype=np.float64)
    rows = [(0.0, 0.0)]
    for v in vals:
        rows.extend([(float(v), 0.0), (-float(v), 0.0), (0.0, float(v)), (0.0, -float(v))])
    return np.asarray(rows, dtype=np.float64)


def square_shift_grid(max_arcmin: float, step_arcmin: float) -> np.ndarray:
    """Return a square shift grid in degrees."""
    max_deg = float(max_arcmin) / 60.0
    step_deg = float(step_arcmin) / 60.0
    vals = np.arange(-max_deg, max_deg + 0.5 * step_deg, step_deg, dtype=np.float64)
    xx, yy = np.meshgrid(vals, vals, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)


def finite_difference_shift_set(fisher_step_arcmin: float) -> np.ndarray:
    """Small x/y offsets used to estimate local Fisher derivatives."""
    h = float(fisher_step_arcmin) / 60.0
    return np.asarray([(h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h), (0.0, 0.0)], dtype=np.float64)


def cumulative_pattern_fisher(
    rates_by_shift: dict[tuple[float, float], np.ndarray],
    *,
    fisher_step_arcmin: float,
    dt: float = DT,
) -> dict[str, np.ndarray]:
    """Compute cumulative count/pattern Fisher traces over time.

    Arrays may be sampled responses ``(T, N)`` or full convolutional maps
    ``(T, N, H, W)``.
    """
    h_deg = float(fisher_step_arcmin) / 60.0
    mu0 = expected_counts_from_rates(rates_by_shift[_shift_key(0.0, 0.0)], dt)
    dmu = finite_difference_derivatives(
        expected_counts_from_rates(rates_by_shift[_shift_key(h_deg, 0.0)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(-h_deg, 0.0)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(0.0, h_deg)], dt),
        expected_counts_from_rates(rates_by_shift[_shift_key(0.0, -h_deg)], dt),
        h_deg,
    )
    total_by_t, total_cum, pattern_by_t, pattern_cum = fisher_by_time(mu0, dmu)
    total_trace = np.trace(total_cum, axis1=1, axis2=2)
    pattern_trace = np.trace(pattern_cum, axis1=1, axis2=2)
    expected_spikes = np.cumsum(np.sum(mu0, axis=tuple(range(1, mu0.ndim))))
    return {
        "fisher_total_by_time": total_by_t.astype(np.float32),
        "fisher_pattern_by_time": pattern_by_t.astype(np.float32),
        "cumulative_fisher_total": total_trace.astype(np.float32),
        "cumulative_fisher_pattern": pattern_trace.astype(np.float32),
        "cumulative_fisher_pattern_per_spike": (pattern_trace / np.maximum(expected_spikes, 1e-12)).astype(np.float32),
        "cumulative_expected_spikes": expected_spikes.astype(np.float32),
        "mu0": mu0.astype(np.float32),
        "dmu": dmu.astype(np.float32),
    }


def cumulative_spatial_ssi(
    rate_map: np.ndarray,
    *,
    dt: float = DT,
) -> dict[str, np.ndarray]:
    """Compute cumulative spatial-information traces from a rate map.

    ``cumulative_spatial_ssi_bits`` is the additive prefix information over
    movie time. ``cumulative_spatial_ssi_bits_per_spike`` is retained as the
    prefix-normalized single-spike value; it is a ratio and is not expected to
    be monotone.
    """
    info = spatial_single_spike_information(rate_map, dt=dt)
    bits_per_spike = np.asarray(info["bits_per_spike"], dtype=np.float32)
    bits_per_second = np.asarray(info["bits_per_second"], dtype=np.float32)
    bits_per_bin = np.asarray(info["bits_per_bin"], dtype=np.float32)
    expected = np.asarray(info["expected_spikes"], dtype=np.float32)
    cumulative_bits = np.asarray(info["cumulative_bits"], dtype=np.float32)
    cumulative_bits_per_spike = np.asarray(info["cumulative_bits_per_spike"], dtype=np.float32)
    cumulative_bits_per_second = np.asarray(info["cumulative_bits_per_second"], dtype=np.float32)
    cumulative_expected = np.asarray(info["cumulative_expected_spikes"], dtype=np.float32)
    unit_bits = np.asarray(info["unit_bits_per_spike"], dtype=np.float32)

    return {
        "spatial_ssi_bits_per_spike": bits_per_spike,
        "spatial_ssi_bits_per_second": bits_per_second,
        "spatial_ssi_bits_per_bin": bits_per_bin,
        "spatial_ssi_expected_spikes": expected,
        "spatial_ssi_unit_bits_per_spike": unit_bits,
        "spatial_ssi_mean_unit_bits_per_spike": np.asarray(info["mean_unit_bits_per_spike"], dtype=np.float32),
        "cumulative_spatial_ssi_bits": cumulative_bits,
        "cumulative_spatial_ssi_bits_per_spike": cumulative_bits_per_spike,
        "prefix_spatial_ssi_bits_per_spike": cumulative_bits_per_spike,
        "cumulative_spatial_ssi_bits_per_second": cumulative_bits_per_second,
        "cumulative_spatial_ssi_expected_spikes": cumulative_expected,
        # Compatibility aliases used by older plotting/metadata code. These now
        # refer to spatial single-spike information, not shift-grid state info.
        "cumulative_ssi_total_bits_per_window": cumulative_bits,
        "cumulative_ssi_pattern_bits_per_window": cumulative_bits,
        "prefix_ssi_total_bits_per_spike": cumulative_bits_per_spike,
        "prefix_ssi_pattern_bits_per_spike": cumulative_bits_per_spike,
        "cumulative_ssi_bits_per_second": cumulative_bits_per_second,
        "cumulative_ssi_expected_spikes": cumulative_expected,
    }


def cumulative_shift_grid_ssi(
    rates_by_shift: dict[tuple[float, float], np.ndarray],
    shift_grid_deg: np.ndarray,
    *,
    dt: float = DT,
) -> dict[str, np.ndarray]:
    """Compute cumulative shift-grid SSI over prefixes."""
    shifts = np.asarray(shift_grid_deg, dtype=np.float64)
    mu_states = np.stack(
        [expected_counts_from_rates(rates_by_shift[_shift_key(float(dx), float(dy))], dt) for dx, dy in shifts],
        axis=0,
    )
    t_max = int(mu_states.shape[1])
    total_spike = np.zeros((t_max,), dtype=np.float32)
    pattern_spike = np.zeros((t_max,), dtype=np.float32)
    total_window = np.zeros((t_max,), dtype=np.float32)
    pattern_window = np.zeros((t_max,), dtype=np.float32)
    bits_per_second = np.zeros((t_max,), dtype=np.float32)
    expected = np.zeros((t_max,), dtype=np.float32)
    for t in range(t_max):
        prefix = mu_states[:, :t + 1]
        info_total = event_code_information(prefix)
        info_pattern = event_code_information_pattern_only(prefix)
        total_spike[t] = float(info_total["bits_per_spike_total"])
        pattern_spike[t] = float(info_pattern["bits_per_spike_pattern"])
        total_window[t] = float(info_total["bits_per_window_total"])
        pattern_window[t] = float(info_pattern["bits_per_spike_pattern"]) * float(info_pattern["mean_expected_spikes"])
        expected[t] = float(info_total["average_expected_spikes"])
        bits_per_second[t] = float(info_total["bits_per_window_total"]) / max((t + 1) * dt, 1e-12)
    return {
        "cumulative_ssi_total_bits_per_window": total_window,
        "cumulative_ssi_pattern_bits_per_window": pattern_window,
        "prefix_ssi_total_bits_per_spike": total_spike,
        "prefix_ssi_pattern_bits_per_spike": pattern_spike,
        "cumulative_ssi_bits_per_second": bits_per_second,
        "cumulative_ssi_expected_spikes": expected,
        "shift_grid_deg": shifts.astype(np.float32),
    }


def approximate_unit_fisher_scores(mu0: np.ndarray, dmu: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rank units by their local shift-sensitivity contribution."""
    mu = np.asarray(mu0, dtype=np.float64)
    jac = np.asarray(dmu, dtype=np.float64)
    if mu.ndim < 2 or jac.ndim != mu.ndim + 1 or jac.shape[:-1] != mu.shape:
        raise ValueError(
            "Expected mu with shape (T, U, ...) and dmu with shape "
            f"mu.shape + (D,), got {mu.shape} and {jac.shape}"
        )
    contribution = np.sum(jac * jac, axis=-1) / np.clip(mu, eps, None)
    reduce_axes = tuple(axis for axis in range(contribution.ndim) if axis != 1)
    return np.sum(contribution, axis=reduce_axes).astype(np.float32)


def final_metric_row(
    *,
    example_id: str,
    kind: str,
    image_index: int,
    condition: str,
    fisher: dict[str, np.ndarray],
    ssi: dict[str, np.ndarray],
    psd_errors: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize final cumulative values for CSV review."""
    spatial_bits = ssi.get("cumulative_spatial_ssi_bits", ssi["cumulative_ssi_total_bits_per_window"])
    spatial_bps = ssi.get("cumulative_spatial_ssi_bits_per_spike", ssi["prefix_ssi_total_bits_per_spike"])
    spatial_rate = ssi.get("cumulative_spatial_ssi_bits_per_second", ssi["cumulative_ssi_bits_per_second"])
    spatial_inst = ssi.get("spatial_ssi_bits_per_spike", ssi["prefix_ssi_total_bits_per_spike"])
    return {
        "example_id": example_id,
        "kind": kind,
        "image_index": int(image_index),
        "condition": condition,
        "final_cumulative_fisher_pattern": float(fisher["cumulative_fisher_pattern"][-1]),
        "final_cumulative_fisher_pattern_per_spike": float(fisher["cumulative_fisher_pattern_per_spike"][-1]),
        "final_cumulative_expected_spikes": float(fisher["cumulative_expected_spikes"][-1]),
        "final_cumulative_ssi_total_bits_per_window": float(ssi["cumulative_ssi_total_bits_per_window"][-1]),
        "final_cumulative_ssi_pattern_bits_per_window": float(ssi["cumulative_ssi_pattern_bits_per_window"][-1]),
        "final_prefix_ssi_total_bits_per_spike": float(ssi["prefix_ssi_total_bits_per_spike"][-1]),
        "final_prefix_ssi_pattern_bits_per_spike": float(ssi["prefix_ssi_pattern_bits_per_spike"][-1]),
        "final_cumulative_ssi_bits_per_second": float(ssi["cumulative_ssi_bits_per_second"][-1]),
        "final_cumulative_spatial_ssi_bits": float(spatial_bits[-1]),
        "final_cumulative_spatial_ssi_bits_per_spike": float(spatial_bps[-1]),
        "final_cumulative_spatial_ssi_bits_per_second": float(spatial_rate[-1]),
        "final_mean_spatial_ssi_bits_per_spike": float(np.mean(spatial_inst)),
        "mean_psd_relative_error": float(np.mean(psd_errors)) if psd_errors is not None else np.nan,
        "max_psd_relative_error": float(np.max(psd_errors)) if psd_errors is not None else np.nan,
    }
