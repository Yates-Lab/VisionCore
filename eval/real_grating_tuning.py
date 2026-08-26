"""Recorded-grating tuning measurements for observed and predicted responses.

The forage-grating experiment varies spatial frequency and orientation while
gaze changes the retinal phase.  It does *not* independently vary temporal
frequency.  Accordingly this module measures SF, orientation, retinal phase,
and the response-lag profile; it never labels response latency as TF tuning.

All model comparisons are made at data-defined conditions.  Independent model
preferences are also returned, but they are a secondary diagnostic rather than
the score used to say that a model reproduced a recorded tuning curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def stable_trial_mask(trial_values: np.ndarray, min_run: int = 4) -> np.ndarray:
    """Keep genuine trial runs after continuous-covariate downsampling.

    The generic rate converter averages ``trial_inds`` just like a continuous
    signal.  Boundary pairs can therefore create not only half-integers but
    also spurious *integer* IDs (for example, averaging a probe value of -1
    with trial 3 yields 1).  A physical trial occupies a long constant run;
    an averaged boundary artifact is isolated.  This mask retains only integer
    IDs belonging to a contiguous run of at least ``min_run`` samples.
    """
    values = np.asarray(trial_values, dtype=np.float64).ravel()
    rounded = np.rint(values)
    integer = (
        np.isfinite(values)
        & np.isclose(values, rounded, atol=1e-7, rtol=0.0)
        & (rounded >= 0)
    )
    result = np.zeros(len(values), dtype=bool)
    start = 0
    while start < len(values):
        if not integer[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and integer[stop] and rounded[stop] == rounded[start]:
            stop += 1
        if stop - start >= int(min_run):
            result[start:stop] = True
        start = stop
    return result


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with finite-pair and zero-variance guards."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return float("nan")
    x = x[good] - np.mean(x[good])
    y = y[good] - np.mean(y[good])
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > EPS else float("nan")


def local_quadratic_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Peak from a three-point local quadratic, with boundary censoring.

    The quadratic is fit only around the sampled maximum.  If the maximum is
    on a boundary, or the local parabola is not concave with its vertex inside
    the neighboring samples, the sampled maximum is returned and ``boundary``
    is true.  ``x`` may be log-spaced coordinates such as log2(SF).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return float("nan"), True
    valid_idx = np.flatnonzero(good)
    i = int(valid_idx[np.argmax(y[good])])
    if i == 0 or i == len(y) - 1 or not np.all(good[i - 1 : i + 2]):
        return float(x[i]), True
    coef = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
    if not np.all(np.isfinite(coef)) or coef[0] >= 0:
        return float(x[i]), True
    vertex = float(-coef[1] / (2.0 * coef[0]))
    if not (x[i - 1] <= vertex <= x[i + 1]):
        return float(x[i]), True
    return vertex, False


def periodic_local_quadratic_peak(values: np.ndarray, period: float = 180.0) -> float:
    """Sub-bin peak of uniformly sampled periodic values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).any():
        return float("nan")
    filled = np.where(np.isfinite(values), values, -np.inf)
    i = int(np.argmax(filled))
    ym, y0, yp = values[(i - 1) % len(values)], values[i], values[(i + 1) % len(values)]
    denom = ym - 2.0 * y0 + yp
    offset = 0.0
    if np.isfinite(denom) and denom < -EPS:
        offset = float(np.clip(0.5 * (ym - yp) / denom, -1.0, 1.0))
    return float(((i + offset) * period / len(values)) % period)


def axial_error_deg(a: float, b: float, period: float = 180.0) -> float:
    """Absolute circular error for an axial quantity such as orientation."""
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    delta = abs((float(a) - float(b)) % period)
    return float(min(delta, period - delta))


def circular_error_deg(a: float, b: float, period: float = 360.0) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    delta = abs((float(a) - float(b)) % period)
    return float(min(delta, period - delta))


def harmonic_fit(phases_rad: np.ndarray, response: np.ndarray) -> dict[str, float]:
    """Fit ``C + K sin(phase) + L cos(phase)`` and return standard F1/F0."""
    phase = np.asarray(phases_rad, dtype=np.float64).ravel()
    response = np.asarray(response, dtype=np.float64).ravel()
    good = np.isfinite(phase) & np.isfinite(response) & (phase >= 0)
    if good.sum() < 8:
        return {k: float("nan") for k in (
            "sin_coeff", "cos_coeff", "offset", "amplitude", "f1_f0", "phase_deg", "r2"
        )}
    phase = np.mod(phase[good], 2.0 * np.pi)
    response = response[good]
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    k, l, offset = (float(v) for v in beta)
    fitted = design @ beta
    amplitude = float(np.hypot(k, l))
    total = float(np.sum((response - np.mean(response)) ** 2))
    residual = float(np.sum((response - fitted) ** 2))
    r2 = 1.0 - residual / total if total > EPS else float("nan")
    return {
        "sin_coeff": k,
        "cos_coeff": l,
        "offset": offset,
        "amplitude": amplitude,
        "f1_f0": amplitude / offset if offset > EPS else float("nan"),
        "phase_deg": float(np.degrees(np.arctan2(l, k)) % 360.0),
        "r2": r2,
    }


def cross_half_harmonic(
    first: dict[str, float],
    second: dict[str, float],
    *,
    offset: float | None = None,
) -> dict[str, float]:
    """Debias harmonic amplitude using independent repeat halves.

    The usual ``hypot(K, L)`` is positive even when both fitted coefficients
    contain only noise.  The dot product of independently estimated harmonic
    vectors has zero expectation for independent noise, while retaining the
    shared phase-locked component.  This uses all repeats for the model/data
    curves; the half split estimates neural measurement noise only.
    """
    a = np.asarray([first.get("sin_coeff"), first.get("cos_coeff")], dtype=np.float64)
    b = np.asarray([second.get("sin_coeff"), second.get("cos_coeff")], dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {k: float("nan") for k in (
            "cross_power", "amplitude", "f1_f0", "phase_consistency"
        )}
    cross_power = float(np.dot(a, b))
    amplitude = float(np.sqrt(max(cross_power, 0.0)))
    if offset is None:
        offset = float(np.nanmean([first.get("offset"), second.get("offset")]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "cross_power": cross_power,
        "amplitude": amplitude,
        "f1_f0": amplitude / float(offset) if np.isfinite(offset) and offset > EPS else float("nan"),
        "phase_consistency": cross_power / denom if denom > EPS else float("nan"),
    }


def _condition_codes(sf: np.ndarray, ori: np.ndarray, sfs: np.ndarray, oris: np.ndarray) -> np.ndarray:
    sf = np.asarray(sf)
    ori = np.asarray(ori)
    codes = np.full(len(sf), -1, dtype=np.int64)
    codes[np.isclose(sf, 0.0)] = 0
    for si, value in enumerate(sfs):
        sf_mask = np.isclose(sf, value)
        for oi, angle in enumerate(oris):
            codes[sf_mask & np.isclose(ori, angle)] = 1 + si * len(oris) + oi
    return codes


def _lag_pairs(raw_indices: np.ndarray, trials: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows containing stimulus t and response t+lag within one trial."""
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    targets = raw_indices + int(lag)
    response_rows = np.searchsorted(raw_indices, targets)
    valid = response_rows < len(raw_indices)
    valid[valid] &= raw_indices[response_rows[valid]] == targets[valid]
    valid[valid] &= trials[response_rows[valid]] == trials[np.flatnonzero(valid)]
    stimulus_rows = np.flatnonzero(valid)
    return stimulus_rows, response_rows[valid]


def condition_lag_tensors(
    *,
    robs: np.ndarray,
    rhat: np.ndarray,
    dfs: np.ndarray,
    sf: np.ndarray,
    ori: np.ndarray,
    trials: np.ndarray,
    raw_indices: np.ndarray,
    dt: float,
    max_lag_ms: float = 125.0,
) -> dict[str, np.ndarray]:
    """Compute rate STAs for blank plus every SF×orientation condition.

    Arrays are returned for all selected trials and for a deterministic
    even/odd split of their trial IDs.  The caller may select either the
    frozen test split or every genuine physical repeat; this keeps neural
    split-half reliability separate from model-defined inclusion.
    """
    robs = np.asarray(robs, dtype=np.float64)
    rhat = np.asarray(rhat, dtype=np.float64)
    dfs = np.asarray(dfs, dtype=np.float64)
    if dfs.ndim == 1:
        dfs = np.repeat(dfs[:, None], robs.shape[1], axis=1)
    if not (robs.shape == rhat.shape == dfs.shape):
        raise ValueError((robs.shape, rhat.shape, dfs.shape))
    sfs = np.sort(np.unique(np.asarray(sf)[np.asarray(sf) > 0]))
    oris = np.sort(np.unique(np.asarray(ori)))
    codes = _condition_codes(sf, ori, sfs, oris)
    n_conditions = 1 + len(sfs) * len(oris)
    n_lags = max(3, int(round(float(max_lag_ms) / (1000.0 * float(dt)))) + 1)
    lags_ms = np.arange(n_lags, dtype=np.float64) * float(dt) * 1000.0
    unique_trials = np.unique(trials)
    trial_half = {float(t): i % 2 for i, t in enumerate(unique_trials)}
    split_names = ("all", "half0", "half1")
    shape = (len(split_names), robs.shape[1], n_lags, n_conditions)
    observed = np.full(shape, np.nan, dtype=np.float32)
    predicted = np.full(shape, np.nan, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.float32)

    for lag in range(n_lags):
        stim_rows, resp_rows = _lag_pairs(raw_indices, trials, lag)
        if len(stim_rows) == 0:
            continue
        pair_codes = codes[stim_rows]
        pair_trials = trials[stim_rows]
        valid_code = pair_codes >= 0
        for split_idx, split_name in enumerate(split_names):
            keep = valid_code.copy()
            if split_name != "all":
                half = int(split_name[-1])
                keep &= np.asarray([trial_half[float(t)] == half for t in pair_trials])
            sr = stim_rows[keep]
            rr = resp_rows[keep]
            cc = pair_codes[keep]
            if len(sr) == 0:
                continue
            one_hot = np.eye(n_conditions, dtype=np.float64)[cc]
            valid = np.isfinite(dfs[rr]) & (dfs[rr] > 0)
            denom = one_hot.T @ valid.astype(np.float64)
            obs_sum = one_hot.T @ np.where(valid, robs[rr], 0.0)
            pred_sum = one_hot.T @ np.where(valid, rhat[rr], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                obs_rate = obs_sum / denom / float(dt)
                pred_rate = pred_sum / denom / float(dt)
            observed[split_idx, :, lag, :] = obs_rate.T.astype(np.float32)
            predicted[split_idx, :, lag, :] = pred_rate.T.astype(np.float32)
            counts[split_idx, :, lag, :] = denom.T.astype(np.float32)

    return {
        "observed": observed,
        "predicted": predicted,
        "counts": counts,
        "sfs": sfs.astype(np.float64),
        "oris": oris.astype(np.float64),
        "lags_ms": lags_ms,
    }


def _dynamic_joint(tensor: np.ndarray, n_sfs: int, n_oris: int) -> np.ndarray:
    blank = tensor[..., :1]
    joint = tensor[..., 1:].reshape(*tensor.shape[:-1], n_sfs, n_oris)
    return joint - blank[..., None]


@dataclass(frozen=True)
class UnitSelection:
    lag_idx: int
    lag_ms: float
    lag_boundary: bool
    sf_idx: int
    ori_idx: int
    preferred_sf: float
    sf_boundary: bool
    preferred_ori: float


def select_unit(dynamic: np.ndarray, sfs: np.ndarray, oris: np.ndarray, lags_ms: np.ndarray) -> UnitSelection:
    """Select lag and condition from a unit's baseline-subtracted tensor."""
    temporal = np.nanstd(dynamic, axis=(1, 2))
    peak_lag_ms, lag_boundary = local_quadratic_peak(lags_ms, temporal)
    if not np.isfinite(peak_lag_ms):
        return UnitSelection(0, float("nan"), True, 0, 0, float("nan"), True, float("nan"))
    lag_idx = int(np.argmin(np.abs(lags_ms - peak_lag_ms)))
    plane = dynamic[lag_idx]
    if not np.isfinite(plane).any():
        return UnitSelection(lag_idx, peak_lag_ms, lag_boundary, 0, 0, float("nan"), True, float("nan"))
    sf_idx, ori_idx = np.unravel_index(np.nanargmax(plane), plane.shape)
    sf_curve = plane[:, ori_idx]
    log_peak, sf_boundary = local_quadratic_peak(np.log2(sfs), sf_curve)
    preferred_sf = float(2.0 ** log_peak) if np.isfinite(log_peak) else float("nan")
    preferred_ori = periodic_local_quadratic_peak(plane[sf_idx], period=180.0)
    # The experiment's first orientation is 11.25°, not 0°.
    if np.isfinite(preferred_ori):
        preferred_ori = float((preferred_ori + float(oris[0])) % 180.0)
    return UnitSelection(
        lag_idx=lag_idx,
        lag_ms=float(peak_lag_ms),
        lag_boundary=bool(lag_boundary),
        sf_idx=int(sf_idx),
        ori_idx=int(ori_idx),
        preferred_sf=preferred_sf,
        sf_boundary=bool(sf_boundary),
        preferred_ori=preferred_ori,
    )


def binned_phase_curve(phases: np.ndarray, response: np.ndarray, valid: np.ndarray, n_bins: int = 12) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(phases) & (phases >= 0) & np.isfinite(response)
    curve = np.full(int(n_bins), np.nan, dtype=np.float64)
    if not np.any(valid):
        return curve
    phase = np.mod(phases[valid], 2.0 * np.pi)
    values = response[valid]
    bins = np.floor(phase / (2.0 * np.pi) * n_bins).astype(int) % n_bins
    for idx in range(n_bins):
        if np.any(bins == idx):
            curve[idx] = np.mean(values[bins == idx])
    return curve


def phase_harmonic_targets(
    *,
    lag_tensors: dict[str, np.ndarray],
    robs: np.ndarray,
    dfs: np.ndarray,
    sf: np.ndarray,
    ori: np.ndarray,
    phase: np.ndarray,
    trials: np.ndarray,
    raw_indices: np.ndarray,
    dt: float,
    min_phase_consistency: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build denoised, all-repeat phase targets at each unit's preferred condition.

    The pointwise Poisson objective is statistically correct, but a small
    phase-locked modulation can be overwhelmed by baseline rate and trial
    noise.  This helper expresses the *same recorded spikes* as a compact
    descriptive target for a calibration loss:

    1. choose each unit's lag, SF, and orientation from the observed
       condition tensor;
    2. estimate the harmonic direction and offset using every selected
       physical repeat;
    3. use the cross-half harmonic amplitude to remove the positive F1 bias;
    4. assign that denoised sinusoidal expected count to the corresponding
       response rows.

    No model prediction enters the target.  Units whose two repeat halves do
    not agree are omitted rather than fitting noise.  Returned row indices are
    positions in the caller's selected arrays (``raw_indices`` maps them back
    to the underlying dataset).
    """
    observed = np.asarray(lag_tensors["observed"], dtype=np.float64)
    sfs = np.asarray(lag_tensors["sfs"], dtype=np.float64)
    oris = np.asarray(lag_tensors["oris"], dtype=np.float64)
    lags_ms = np.asarray(lag_tensors["lags_ms"], dtype=np.float64)
    robs = np.asarray(robs, dtype=np.float64)
    dfs = np.asarray(dfs, dtype=np.float64)
    if dfs.ndim == 1:
        dfs = np.repeat(dfs[:, None], robs.shape[1], axis=1)
    if robs.shape != dfs.shape:
        raise ValueError(f"robs/dfs shape mismatch: {robs.shape} versus {dfs.shape}")
    if observed.shape[1] != robs.shape[1]:
        raise ValueError("lag tensor and response array have different unit counts")

    sf = np.asarray(sf, dtype=np.float64).ravel()
    ori = np.asarray(ori, dtype=np.float64).ravel()
    phase = np.asarray(phase, dtype=np.float64).ravel()
    trials = np.asarray(trials).ravel()
    raw_indices = np.asarray(raw_indices, dtype=np.int64).ravel()
    n_rows, n_units = robs.shape
    if not all(len(value) == n_rows for value in (sf, ori, phase, trials, raw_indices)):
        raise ValueError("all sample covariates must align with robs")

    targets = np.full((n_rows, n_units), np.nan, dtype=np.float32)
    mask = np.zeros((n_rows, n_units), dtype=bool)
    preferred_lag = np.full(n_units, -1, dtype=np.int64)
    preferred_sf = np.full(n_units, np.nan, dtype=np.float64)
    preferred_ori = np.full(n_units, np.nan, dtype=np.float64)
    phase_consistency = np.full(n_units, np.nan, dtype=np.float64)
    target_f1_f0 = np.full(n_units, np.nan, dtype=np.float64)

    data_dyn = _dynamic_joint(observed, len(sfs), len(oris))
    unique_trials = np.unique(trials)
    trial_half_lookup = {float(value): idx % 2 for idx, value in enumerate(unique_trials)}
    trial_halves = np.asarray([trial_half_lookup[float(value)] for value in trials])

    for unit in range(n_units):
        selection = select_unit(data_dyn[0, unit], sfs, oris, lags_ms)
        li, si, oi = selection.lag_idx, selection.sf_idx, selection.ori_idx
        stim_rows, response_rows = _lag_pairs(raw_indices, trials, li)
        condition = (
            np.isclose(sf[stim_rows], sfs[si])
            & np.isclose(ori[stim_rows], oris[oi])
        )
        stimulus_rows = stim_rows[condition]
        response_rows = response_rows[condition]
        valid = (
            np.isfinite(dfs[response_rows, unit])
            & (dfs[response_rows, unit] > 0)
            & np.isfinite(phase[stimulus_rows])
            & (phase[stimulus_rows] >= 0)
        )
        if valid.sum() < 16:
            continue
        stimulus_rows = stimulus_rows[valid]
        response_rows = response_rows[valid]
        phases = phase[stimulus_rows]
        rates = robs[response_rows, unit] / float(dt)
        full_fit = harmonic_fit(phases, rates)
        half_fits = [
            harmonic_fit(phases[trial_halves[stimulus_rows] == half],
                         rates[trial_halves[stimulus_rows] == half])
            for half in (0, 1)
        ]
        debiased = cross_half_harmonic(
            half_fits[0], half_fits[1], offset=full_fit["offset"]
        )
        consistency = float(debiased["phase_consistency"])
        amplitude = float(debiased["amplitude"])
        full_amplitude = float(full_fit["amplitude"])
        offset = float(full_fit["offset"])
        if (
            not np.isfinite(consistency)
            or consistency < float(min_phase_consistency)
            or not np.isfinite(amplitude)
            or not np.isfinite(full_amplitude)
            or full_amplitude <= EPS
            or not np.isfinite(offset)
            or offset <= EPS
        ):
            continue

        scale = amplitude / full_amplitude
        expected_rate = (
            offset
            + scale * float(full_fit["sin_coeff"]) * np.sin(phases)
            + scale * float(full_fit["cos_coeff"]) * np.cos(phases)
        )
        # A Poisson mean cannot be negative.  The floor is far below one spike
        # per sample and only matters for an unusually deep fitted trough.
        expected_count = np.maximum(expected_rate * float(dt), 1.0e-8)
        targets[response_rows, unit] = expected_count.astype(np.float32)
        mask[response_rows, unit] = True
        preferred_lag[unit] = int(li)
        preferred_sf[unit] = float(sfs[si])
        preferred_ori[unit] = float(oris[oi])
        phase_consistency[unit] = consistency
        target_f1_f0[unit] = amplitude / offset

    return {
        "target_counts": targets,
        "target_mask": mask,
        "selected_rows": np.flatnonzero(mask.any(axis=1)).astype(np.int64),
        "raw_indices": raw_indices,
        "preferred_lag_idx": preferred_lag,
        "preferred_sf_cpd": preferred_sf,
        "preferred_ori_deg": preferred_ori,
        "phase_consistency": phase_consistency,
        "target_f1_f0": target_f1_f0,
    }


def phase_harmonic_grid_targets(
    *,
    lag_tensors: dict[str, np.ndarray],
    robs: np.ndarray,
    dfs: np.ndarray,
    sf: np.ndarray,
    ori: np.ndarray,
    phase: np.ndarray,
    trials: np.ndarray,
    raw_indices: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    """Build smooth all-repeat targets over the complete SF×orientation grid.

    Each unit keeps the response lag selected from its recorded condition
    tensor. At that lag, an independent first-harmonic function is fitted to
    every measured SF×orientation condition using all physical repeats. This
    differs from phase_harmonic_targets, which deliberately targets only
    reliable units at their preferred condition. The grid target is intended
    as a calibration constraint: it prevents a phase branch from matching one
    preferred curve by silently damaging off-condition firing rates, SF
    tuning, orientation tuning, or full-grating likelihood.

    The target is derived only from recorded spikes. It uses the ordinary
    all-repeat harmonic amplitude because the purpose is to reproduce the
    complete training-set tuning surface, not to estimate a noise-free
    population statistic.
    """
    observed = np.asarray(lag_tensors["observed"], dtype=np.float64)
    sfs = np.asarray(lag_tensors["sfs"], dtype=np.float64)
    oris = np.asarray(lag_tensors["oris"], dtype=np.float64)
    lags_ms = np.asarray(lag_tensors["lags_ms"], dtype=np.float64)
    robs = np.asarray(robs, dtype=np.float64)
    dfs = np.asarray(dfs, dtype=np.float64)
    if dfs.ndim == 1:
        dfs = np.repeat(dfs[:, None], robs.shape[1], axis=1)
    if robs.shape != dfs.shape:
        raise ValueError(f"robs/dfs shape mismatch: {robs.shape} versus {dfs.shape}")
    if observed.shape[1] != robs.shape[1]:
        raise ValueError("lag tensor and response array have different unit counts")

    sf = np.asarray(sf, dtype=np.float64).ravel()
    ori = np.asarray(ori, dtype=np.float64).ravel()
    phase = np.asarray(phase, dtype=np.float64).ravel()
    trials = np.asarray(trials).ravel()
    raw_indices = np.asarray(raw_indices, dtype=np.int64).ravel()
    n_rows, n_units = robs.shape
    if not all(len(value) == n_rows for value in (sf, ori, phase, trials, raw_indices)):
        raise ValueError("all sample covariates must align with robs")

    targets = np.full((n_rows, n_units), np.nan, dtype=np.float32)
    mask = np.zeros((n_rows, n_units), dtype=bool)
    preferred_lag = np.full(n_units, -1, dtype=np.int64)
    fit_r2 = np.full((n_units, len(sfs), len(oris)), np.nan, dtype=np.float32)
    fit_f1_f0 = np.full_like(fit_r2, np.nan)
    data_dyn = _dynamic_joint(observed, len(sfs), len(oris))

    for unit in range(n_units):
        selection = select_unit(data_dyn[0, unit], sfs, oris, lags_ms)
        lag_idx = int(selection.lag_idx)
        stimulus_rows, response_rows = _lag_pairs(
            raw_indices, trials, lag_idx
        )
        preferred_lag[unit] = lag_idx
        for sf_idx, sf_value in enumerate(sfs):
            for ori_idx, ori_value in enumerate(oris):
                condition = (
                    np.isclose(sf[stimulus_rows], sf_value)
                    & np.isclose(ori[stimulus_rows], ori_value)
                )
                sr = stimulus_rows[condition]
                rr = response_rows[condition]
                valid = (
                    np.isfinite(dfs[rr, unit])
                    & (dfs[rr, unit] > 0)
                    & np.isfinite(robs[rr, unit])
                    & np.isfinite(phase[sr])
                    & (phase[sr] >= 0)
                )
                if valid.sum() < 8:
                    continue
                sr = sr[valid]
                rr = rr[valid]
                phases = phase[sr]
                rates = robs[rr, unit] / float(dt)
                fitted = harmonic_fit(phases, rates)
                coefficients = np.asarray(
                    [
                        fitted["sin_coeff"],
                        fitted["cos_coeff"],
                        fitted["offset"],
                    ],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(coefficients)):
                    continue
                expected_rate = (
                    coefficients[2]
                    + coefficients[0] * np.sin(phases)
                    + coefficients[1] * np.cos(phases)
                )
                expected_count = np.maximum(
                    expected_rate * float(dt), 1.0e-8
                )
                targets[rr, unit] = expected_count.astype(np.float32)
                mask[rr, unit] = True
                fit_r2[unit, sf_idx, ori_idx] = fitted["r2"]
                fit_f1_f0[unit, sf_idx, ori_idx] = fitted["f1_f0"]

    return {
        "target_counts": targets,
        "target_mask": mask,
        "selected_rows": np.flatnonzero(mask.any(axis=1)).astype(np.int64),
        "raw_indices": raw_indices,
        "preferred_lag_idx": preferred_lag,
        "fit_r2": fit_r2,
        "fit_f1_f0": fit_f1_f0,
        "sfs": sfs,
        "oris": oris,
    }


def analyze_tuning(
    *,
    lag_tensors: dict[str, np.ndarray],
    robs: np.ndarray,
    rhat: np.ndarray,
    dfs: np.ndarray,
    sf: np.ndarray,
    ori: np.ndarray,
    phase: np.ndarray,
    trials: np.ndarray,
    raw_indices: np.ndarray,
    dt: float,
    n_phase_bins: int = 12,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Extract data-defined curves and one compact metric row per unit."""
    observed = np.asarray(lag_tensors["observed"], dtype=np.float64)
    predicted = np.asarray(lag_tensors["predicted"], dtype=np.float64)
    sfs = np.asarray(lag_tensors["sfs"], dtype=np.float64)
    oris = np.asarray(lag_tensors["oris"], dtype=np.float64)
    lags_ms = np.asarray(lag_tensors["lags_ms"], dtype=np.float64)
    n_units = observed.shape[1]
    data_dyn = _dynamic_joint(observed, len(sfs), len(oris))
    model_dyn = _dynamic_joint(predicted, len(sfs), len(oris))
    robs = np.asarray(robs, dtype=np.float64)
    rhat = np.asarray(rhat, dtype=np.float64)
    dfs = np.asarray(dfs, dtype=np.float64)
    if dfs.ndim == 1:
        dfs = np.repeat(dfs[:, None], n_units, axis=1)
    unique_trials = np.unique(trials)
    trial_halves = np.asarray([{float(t): i % 2 for i, t in enumerate(unique_trials)}[float(t)] for t in trials])

    data_temporal = np.nanstd(data_dyn[0], axis=(2, 3))
    model_temporal = np.nanstd(model_dyn[0], axis=(2, 3))
    shape_sf = (n_units, len(sfs))
    shape_ori = (n_units, len(oris))
    data_sf = np.full(shape_sf, np.nan, dtype=np.float32)
    model_sf = np.full(shape_sf, np.nan, dtype=np.float32)
    data_ori = np.full(shape_ori, np.nan, dtype=np.float32)
    model_ori = np.full(shape_ori, np.nan, dtype=np.float32)
    data_phase = np.full((n_units, n_phase_bins), np.nan, dtype=np.float32)
    model_phase = np.full_like(data_phase, np.nan)
    data_phase_halves = np.full((2, n_units, n_phase_bins), np.nan, dtype=np.float32)
    model_phase_halves = np.full_like(data_phase_halves, np.nan)
    metrics: list[dict[str, float]] = []

    for unit in range(n_units):
        data_sel = select_unit(data_dyn[0, unit], sfs, oris, lags_ms)
        model_sel = select_unit(model_dyn[0, unit], sfs, oris, lags_ms)
        li, si, oi = data_sel.lag_idx, data_sel.sf_idx, data_sel.ori_idx
        data_sf[unit] = data_dyn[0, unit, li, :, oi]
        model_sf[unit] = model_dyn[0, unit, li, :, oi]
        data_ori[unit] = data_dyn[0, unit, li, si, :]
        model_ori[unit] = model_dyn[0, unit, li, si, :]

        stim_rows, resp_rows = _lag_pairs(raw_indices, trials, li)
        cond = np.isclose(sf[stim_rows], sfs[si]) & np.isclose(ori[stim_rows], oris[oi])
        sr, rr = stim_rows[cond], resp_rows[cond]
        valid = np.isfinite(dfs[rr, unit]) & (dfs[rr, unit] > 0) & (phase[sr] >= 0)
        obs_rate = robs[rr, unit] / float(dt)
        pred_rate = rhat[rr, unit] / float(dt)
        data_phase[unit] = binned_phase_curve(phase[sr], obs_rate, valid, n_phase_bins)
        model_phase[unit] = binned_phase_curve(phase[sr], pred_rate, valid, n_phase_bins)
        data_half_fits = []
        model_half_fits = []
        for half in (0, 1):
            hvalid = valid & (trial_halves[sr] == half)
            data_phase_halves[half, unit] = binned_phase_curve(phase[sr], obs_rate, hvalid, n_phase_bins)
            model_phase_halves[half, unit] = binned_phase_curve(phase[sr], pred_rate, hvalid, n_phase_bins)
            data_half_fits.append(harmonic_fit(phase[sr][hvalid], obs_rate[hvalid]))
            model_half_fits.append(harmonic_fit(phase[sr][hvalid], pred_rate[hvalid]))
        data_fit = harmonic_fit(phase[sr][valid], obs_rate[valid])
        model_fit = harmonic_fit(phase[sr][valid], pred_rate[valid])
        data_debiased = cross_half_harmonic(
            data_half_fits[0], data_half_fits[1], offset=data_fit["offset"]
        )
        model_debiased = cross_half_harmonic(
            model_half_fits[0], model_half_fits[1], offset=model_fit["offset"]
        )

        sf_rel = safe_corr(
            data_dyn[1, unit, li, :, oi], data_dyn[2, unit, li, :, oi]
        )
        ori_rel = safe_corr(
            data_dyn[1, unit, li, si, :], data_dyn[2, unit, li, si, :]
        )
        temporal_rel = safe_corr(
            np.nanstd(data_dyn[1, unit], axis=(1, 2)),
            np.nanstd(data_dyn[2, unit], axis=(1, 2)),
        )
        phase_rel = safe_corr(data_phase_halves[0, unit], data_phase_halves[1, unit])
        metrics.append({
            "unit": int(unit),
            # This is the spike count in the caller-selected assay scope.  The
            # production capture assay supplies every genuine grating repeat,
            # so a ``test_*`` name is actively misleading here.
            "n_spikes": float(np.nansum(robs[:, unit] * (dfs[:, unit] > 0))),
            "data_peak_lag_ms": data_sel.lag_ms,
            "model_peak_lag_ms": model_sel.lag_ms,
            "lag_error_ms": abs(model_sel.lag_ms - data_sel.lag_ms),
            "data_lag_boundary": float(data_sel.lag_boundary),
            "model_lag_boundary": float(model_sel.lag_boundary),
            "temporal_curve_corr": safe_corr(data_temporal[unit], model_temporal[unit]),
            "temporal_split_half": temporal_rel,
            "data_preferred_sf_cpd": data_sel.preferred_sf,
            "model_preferred_sf_cpd": model_sel.preferred_sf,
            "sf_error_octaves": abs(np.log2(model_sel.preferred_sf / data_sel.preferred_sf))
            if data_sel.preferred_sf > 0 and model_sel.preferred_sf > 0 else float("nan"),
            "data_sf_boundary": float(data_sel.sf_boundary),
            "model_sf_boundary": float(model_sel.sf_boundary),
            "sf_curve_corr": safe_corr(data_sf[unit], model_sf[unit]),
            "sf_split_half": sf_rel,
            "data_preferred_ori_deg": data_sel.preferred_ori,
            "model_preferred_ori_deg": model_sel.preferred_ori,
            "ori_error_deg": axial_error_deg(model_sel.preferred_ori, data_sel.preferred_ori),
            "ori_curve_corr": safe_corr(data_ori[unit], model_ori[unit]),
            "ori_split_half": ori_rel,
            "phase_curve_corr": safe_corr(data_phase[unit], model_phase[unit]),
            "phase_split_half": phase_rel,
            "data_f1_f0": data_fit["f1_f0"],
            "model_f1_f0": model_fit["f1_f0"],
            "data_debiased_f1_f0": data_debiased["f1_f0"],
            "model_cross_half_f1_f0": model_debiased["f1_f0"],
            "data_phase_vector_consistency": data_debiased["phase_consistency"],
            "data_phase_fit_r2": data_fit["r2"],
            "model_phase_fit_r2": model_fit["r2"],
            "data_preferred_phase_deg": data_fit["phase_deg"],
            "model_preferred_phase_deg": model_fit["phase_deg"],
            "phase_error_deg": circular_error_deg(model_fit["phase_deg"], data_fit["phase_deg"]),
        })

    curves = {
        "sfs": sfs,
        "oris": oris,
        "lags_ms": lags_ms,
        "phase_bins_deg": (np.arange(n_phase_bins) + 0.5) * 360.0 / n_phase_bins,
        "data_temporal": data_temporal.astype(np.float32),
        "model_temporal": model_temporal.astype(np.float32),
        "data_sf": data_sf,
        "model_sf": model_sf,
        "data_ori": data_ori,
        "model_ori": model_ori,
        "data_phase": data_phase,
        "model_phase": model_phase,
        "data_phase_halves": data_phase_halves,
        "model_phase_halves": model_phase_halves,
    }
    return curves, metrics
