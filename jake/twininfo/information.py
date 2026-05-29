"""Information metrics for cache-backed digital-twin analyses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StimulusRecord:
    stimulus_id: str
    family: str
    source_image_index: int | None
    variant: str
    seed: int
    image_shape: tuple[int, int]
    image_min: float
    image_max: float
    image_mean: float
    image_std: float
    spatial_band_power: dict[str, float]
    high_sf_fraction: float


@dataclass(frozen=True)
class EyeTraceRecord:
    trace_id: str
    source_trace_index: int
    rank: int
    condition: str
    seed: int
    t_max: int
    dt: float
    rms_displacement_deg: float
    path_length_deg: float
    mean_speed_deg_s: float
    median_speed_deg_s: float
    max_speed_deg_s: float
    microsaccade_fraction: float | None
    psd_summary: dict[str, float]


@dataclass(frozen=True)
class TwinResponseRecord:
    response_id: str
    stimulus_id: str
    trace_id: str
    population_id: str
    n_units: int
    t_max: int
    rate_shape: tuple[int, int]
    total_expected_spikes: float
    mean_rate: float
    median_unit_rate: float
    n_dead_units: int


@dataclass(frozen=True)
class InformationRecord:
    info_id: str
    response_id: str
    metric: str
    shift_step_deg: float | None
    total_expected_spikes: float
    fisher_total: list[list[float]] | None
    fisher_count: list[list[float]] | None
    fisher_pattern: list[list[float]] | None
    fisher_trace_total: float | None
    fisher_logdet_total: float | None
    fisher_ellipse_area_total: float | None
    fisher_trace_pattern: float | None
    fisher_logdet_pattern: float | None
    fisher_ellipse_area_pattern: float | None
    info_bits_per_spike: float | None
    info_bits_per_second: float | None


def expected_counts_from_rates(rates: np.ndarray, dt: float) -> np.ndarray:
    """Convert model rates in spikes/s to expected counts per time bin."""
    return np.asarray(rates, dtype=np.float64) * float(dt)


def _flatten_mu_j(
    mu: np.ndarray,
    dmu_dtheta: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mu_arr = np.asarray(mu, dtype=np.float64)
    j_arr = np.asarray(dmu_dtheta, dtype=np.float64)
    if j_arr.shape[:-1] != mu_arr.shape:
        raise ValueError(f"mu shape {mu_arr.shape} incompatible with J shape {j_arr.shape}")
    mu_flat = mu_arr.reshape(-1)
    j_flat = j_arr.reshape(-1, j_arr.shape[-1])
    if mask is not None:
        mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_flat.shape != mu_flat.shape:
            raise ValueError(f"mask shape {mask_flat.shape} incompatible with mu shape {mu_flat.shape}")
        mu_flat = mu_flat[mask_flat]
        j_flat = j_flat[mask_flat]
    good = np.isfinite(mu_flat) & np.all(np.isfinite(j_flat), axis=1)
    return mu_flat[good], j_flat[good]


def poisson_fisher_from_counts(
    mu: np.ndarray,
    dmu_dtheta: np.ndarray,
    eps: float = 1e-12,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Poisson Fisher information from expected counts.

    Parameters
    ----------
    mu:
        Expected counts with arbitrary leading shape.
    dmu_dtheta:
        Derivatives with shape ``mu.shape + (D,)``.
    eps:
        Expected-count floor used only in the denominator.
    mask:
        Optional boolean inclusion mask with shape ``mu.shape``.
    """
    mu_flat, j_flat = _flatten_mu_j(mu, dmu_dtheta, mask=mask)
    if j_flat.size == 0:
        d = np.asarray(dmu_dtheta).shape[-1]
        return np.zeros((d, d), dtype=np.float64)
    mu_safe = np.clip(mu_flat, eps, None)
    fisher = j_flat.T @ (j_flat / mu_safe[:, None])
    return 0.5 * (fisher + fisher.T)


def poisson_fisher_count_pattern_decomposition(
    mu: np.ndarray,
    dmu_dtheta: np.ndarray,
    eps: float = 1e-12,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return total, count, and pattern Fisher information.

    ``F_count`` is the information in total expected spike count, and
    ``F_pattern`` is the information in the normalized unit x time pattern at a
    fixed spike budget. The returned matrices satisfy ``F_total ~= F_count +
    F_pattern`` up to numerical floor effects.
    """
    mu_flat, j_flat = _flatten_mu_j(mu, dmu_dtheta, mask=mask)
    d = np.asarray(dmu_dtheta).shape[-1]
    if j_flat.size == 0:
        z = np.zeros((d, d), dtype=np.float64)
        return z, z, z, {"Lambda": 0.0, "n_valid": 0, "n_floored": 0}

    n_floored = int(np.sum(mu_flat < eps))
    mu_safe = np.clip(mu_flat, eps, None)
    f_total = j_flat.T @ (j_flat / mu_safe[:, None])
    lam = float(np.sum(mu_safe))
    if lam <= eps:
        z = np.zeros((d, d), dtype=np.float64)
        return z, z, z, {"Lambda": lam, "n_valid": int(mu_flat.size), "n_floored": n_floored}

    d_lam = np.sum(j_flat, axis=0)
    f_count = np.outer(d_lam, d_lam) / lam
    p = mu_safe / lam
    dp = (j_flat * lam - mu_safe[:, None] * d_lam[None, :]) / (lam * lam)
    f_pattern = lam * (dp.T @ (dp / np.clip(p, eps, None)[:, None]))

    f_total = 0.5 * (f_total + f_total.T)
    f_count = 0.5 * (f_count + f_count.T)
    f_pattern = 0.5 * (f_pattern + f_pattern.T)
    residual = f_total - f_count - f_pattern
    info = {
        "Lambda": lam,
        "n_valid": int(mu_flat.size),
        "n_floored": n_floored,
        "floor_fraction": float(n_floored / max(mu_flat.size, 1)),
        "decomposition_abs_error": float(np.max(np.abs(residual))),
        "decomposition_rel_error": float(np.linalg.norm(residual) / (np.linalg.norm(f_total) + eps)),
    }
    return f_total, f_count, f_pattern, info


def fisher_by_time(
    mu: np.ndarray,
    dmu_dtheta: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute instantaneous and cumulative Fisher matrices by time bin.

    ``mu`` may be ``(T, N)`` sampled responses or a full convolutional rate map
    such as ``(T, N, H, W)``. All non-time axes are flattened within each time
    bin by the Poisson information helper.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    j_arr = np.asarray(dmu_dtheta, dtype=np.float64)
    if mu_arr.ndim < 2 or j_arr.ndim != mu_arr.ndim + 1 or j_arr.shape[:-1] != mu_arr.shape:
        raise ValueError(
            "Expected mu with shape (T, ...) and dmu_dtheta with shape "
            f"mu.shape + (D,), got {mu_arr.shape} and {j_arr.shape}"
        )
    t_max = mu_arr.shape[0]
    d = j_arr.shape[-1]
    total = np.zeros((t_max, d, d), dtype=np.float64)
    pattern = np.zeros_like(total)
    for t in range(t_max):
        total[t], _, pattern[t], _ = poisson_fisher_count_pattern_decomposition(
            mu_arr[t], j_arr[t], eps=eps
        )
    return total, np.cumsum(total, axis=0), pattern, np.cumsum(pattern, axis=0)


def spatial_single_spike_information(
    rate_map: np.ndarray,
    dt: float = 1.0,
    eps: float = 1e-8,
    log_base: float = 2.0,
    spike_weighted: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Compute spatial single-spike information from a convolutional rate map.

    This is the NumPy equivalent of ``scripts/spatial_info.py``'s
    ``spatial_ssi_population``.  Spatial positions, not artificial shift-grid
    states, define the stimulus ensemble.  For each time and unit, the rate map
    is normalized by that unit's spatial mean, yielding
    ``E_x[(r(x) / rbar) log2(r(x) / rbar)]`` in bits/spike.

    Parameters
    ----------
    rate_map:
        Model rates in spikes/s with shape ``(T, N, H, W)``.
    dt:
        Seconds per time bin. Used only to convert rates into expected spikes
        for population spike-weighted summaries.
    eps:
        Numerical floor for divisions and logs.
    log_base:
        Use ``2.0`` for bits. Other bases are supported for completeness.
    spike_weighted:
        If true, population bits/spike at each time is weighted by the expected
        spike count of each unit. If false, units are averaged equally.
    """
    y = np.asarray(rate_map, dtype=np.float64)
    if y.ndim != 4:
        raise ValueError(f"Expected rate_map with shape (T, N, H, W), got {y.shape}")
    if np.any(y < 0):
        raise ValueError("rate_map must be non-negative")
    t_max, n_units, height, width = y.shape
    r = y.reshape(t_max, n_units, height * width)
    rbar = np.mean(r, axis=2)
    gain = r / (rbar[..., None] + eps)
    if float(log_base) == 2.0:
        log_gain = np.log2(gain + eps)
    else:
        log_gain = np.log(gain + eps) / np.log(float(log_base))
    unit_bits_per_spike = np.mean(gain * log_gain, axis=2)
    spikes_tn = rbar * float(dt)
    bits_tn = spikes_tn * unit_bits_per_spike
    bits_per_second_tn = rbar * unit_bits_per_spike

    if spike_weighted:
        spikes_t = np.sum(spikes_tn, axis=1)
        bits_t = np.sum(bits_tn, axis=1)
        bits_per_spike_t = bits_t / np.maximum(spikes_t, eps)
    else:
        spikes_t = np.sum(spikes_tn, axis=1)
        bits_t = np.sum(bits_tn, axis=1)
        bits_per_spike_t = np.mean(unit_bits_per_spike, axis=1)

    bits_per_second_t = np.sum(bits_per_second_tn, axis=1)
    cumulative_bits = np.cumsum(bits_t)
    cumulative_spikes = np.cumsum(spikes_t)
    cumulative_bits_per_spike = cumulative_bits / np.maximum(cumulative_spikes, eps)
    elapsed_s = np.arange(1, t_max + 1, dtype=np.float64) * float(dt)
    return {
        "bits_per_spike": bits_per_spike_t.astype(np.float32),
        "bits_per_second": bits_per_second_t.astype(np.float32),
        "bits_per_bin": bits_t.astype(np.float32),
        "expected_spikes": spikes_t.astype(np.float32),
        "cumulative_bits": cumulative_bits.astype(np.float32),
        "cumulative_bits_per_spike": cumulative_bits_per_spike.astype(np.float32),
        "cumulative_bits_per_second": (cumulative_bits / np.maximum(elapsed_s, eps)).astype(np.float32),
        "cumulative_expected_spikes": cumulative_spikes.astype(np.float32),
        "unit_bits_per_spike": unit_bits_per_spike.astype(np.float32),
        "unit_mean_rate": rbar.astype(np.float32),
        "mean_unit_bits_per_spike": np.mean(unit_bits_per_spike, axis=1).astype(np.float32),
        "n_time": int(t_max),
        "n_units": int(n_units),
        "n_spatial_bins": int(height * width),
    }


def fisher_scalars(F: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Return stable scalar summaries for a Fisher matrix."""
    mat = np.asarray(F, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square Fisher matrix, got {mat.shape}")
    mat = 0.5 * (mat + mat.T)
    evals = np.linalg.eigvalsh(mat)
    evals_clip = np.clip(evals, eps, None)
    det = float(np.prod(evals_clip))
    logdet = float(np.sum(np.log(evals_clip)))
    trace = float(np.trace(mat))
    ellipse_area = float(np.pi / np.sqrt(det))
    cond = float(evals_clip[-1] / evals_clip[0]) if evals_clip.size else float("nan")
    return {
        "trace": trace,
        "det": det,
        "logdet": logdet,
        "eig_min": float(evals[0]) if evals.size else float("nan"),
        "eig_max": float(evals[-1]) if evals.size else float("nan"),
        "ellipse_area": ellipse_area,
        "anisotropy": cond,
        "condition_number": cond,
    }


def _state_prior(n_states: int, prior: np.ndarray | None) -> np.ndarray:
    if prior is None:
        return np.full(n_states, 1.0 / n_states, dtype=np.float64)
    prior_arr = np.asarray(prior, dtype=np.float64)
    if prior_arr.shape != (n_states,):
        raise ValueError(f"prior shape {prior_arr.shape} incompatible with {n_states} states")
    total = float(np.sum(prior_arr))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("prior must have a positive finite sum")
    return prior_arr / total


def event_code_information(
    mu_by_state: np.ndarray,
    prior: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute event-code information from expected counts.

    The leading axis indexes discrete states. All trailing axes are flattened
    into the event index, usually time x unit. ``bits_per_spike_total`` is the
    single-spike/event information. ``bits_per_window_total`` is the expected
    information in the full response window.
    """
    mu = np.asarray(mu_by_state, dtype=np.float64)
    if mu.ndim < 2:
        raise ValueError("Expected mu_by_state with a leading state axis")
    s = mu.shape[0]
    mu = mu.reshape(s, -1)
    prior_arr = _state_prior(s, prior)
    mu_bar = np.sum(prior_arr[:, None] * mu, axis=0)
    lam_bar = float(np.sum(mu_bar))
    ratio = np.clip(mu, eps, None) / np.clip(mu_bar[None, :], eps, None)
    numer = float(np.sum(prior_arr[:, None] * mu * np.log2(ratio)))
    return {
        "bits_per_spike_total": float(numer / max(lam_bar, eps)),
        "bits_per_window_total": numer,
        "average_expected_spikes": lam_bar,
        "n_states": int(s),
        "n_events": int(mu.shape[1]),
    }


def event_code_information_pattern_only(
    mu_by_state: np.ndarray,
    prior: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute pattern-only event information after normalizing each state.

    This controls for state-dependent total expected spike count and asks how
    much the normalized event pattern carries about the state.
    """
    mu = np.asarray(mu_by_state, dtype=np.float64)
    if mu.ndim < 2:
        raise ValueError("Expected mu_by_state with a leading state axis")
    s = mu.shape[0]
    mu = mu.reshape(s, -1)
    prior_arr = _state_prior(s, prior)
    lam = np.sum(mu, axis=1)
    p_shift = mu / np.clip(lam[:, None], eps, None)
    p_bar = np.sum(prior_arr[:, None] * p_shift, axis=0)
    ratio = np.clip(p_shift, eps, None) / np.clip(p_bar[None, :], eps, None)
    bits = float(np.sum(prior_arr[:, None] * p_shift * np.log2(ratio)))
    return {
        "bits_per_spike_pattern": bits,
        "mean_expected_spikes": float(np.sum(prior_arr * lam)),
        "n_states": int(s),
        "n_events": int(mu.shape[1]),
    }


def single_spike_info_event_code(
    mu_by_shift: np.ndarray,
    prior: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Backward-compatible shift-grid event-code information wrapper."""
    out = event_code_information(mu_by_shift, prior=prior, eps=eps)
    out["bits_per_spike"] = out["bits_per_spike_total"]
    out["bits_per_window"] = out["bits_per_window_total"]
    return out


def single_spike_info_pattern_only(
    mu_by_shift: np.ndarray,
    prior: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Backward-compatible pattern-only shift-grid information wrapper."""
    return event_code_information_pattern_only(mu_by_shift, prior=prior, eps=eps)


def image_shift_grid(max_shift_deg: float, step_deg: float) -> np.ndarray:
    """Return a square grid of ``(dx, dy)`` shifts in degrees."""
    vals = np.arange(-max_shift_deg, max_shift_deg + 0.5 * step_deg, step_deg, dtype=np.float64)
    xx, yy = np.meshgrid(vals, vals, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def finite_difference_derivatives(
    mu_x_plus: np.ndarray,
    mu_x_minus: np.ndarray,
    mu_y_plus: np.ndarray,
    mu_y_minus: np.ndarray,
    h_deg: float,
) -> np.ndarray:
    """Stack central finite-difference derivatives for x/y shifts."""
    h = float(h_deg)
    if h <= 0:
        raise ValueError("h_deg must be positive")
    dmu_dx = (np.asarray(mu_x_plus, dtype=np.float64) - np.asarray(mu_x_minus, dtype=np.float64)) / (2.0 * h)
    dmu_dy = (np.asarray(mu_y_plus, dtype=np.float64) - np.asarray(mu_y_minus, dtype=np.float64)) / (2.0 * h)
    return np.stack([dmu_dx, dmu_dy], axis=-1)
