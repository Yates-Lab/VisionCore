"""Explainable-rate-variance scoring for Figure 3.

The score is computed in count-variance units:

    Q = [Var(y) - Var(y - yhat)] / diag(Crate)_fig2.

The numerator is measured on Figure 2's one-bin window contract intersected with
the twin's valid support (`dfs > 0`, which also removes each trial's model
history warm-up). The denominator is read from the Figure 2 covariance
decomposition at the same one-bin counting window and is NOT re-estimated here.

Re-estimating diag(Crate) on the model-valid subset was tried and abandoned: the
33-frame history mask removes most of the close trial pairs the estimator needs,
which left the denominator undefined for half the population and swung the rest
by a factor of five (5-95% of matched/Figure 2 = [0.10, 5.19]). Figure 2's
estimate uses every bin it has and is the same number panel E and figure 2
report, so it is the stable denominator. `estimate_matched_rate_variance` is
retained as a diagnostic of that decision; it never gates the score.

The two quantities are therefore measured on different sample sets: matched
Var(y) is a median 0.88 of Figure 2's diag(Ctotal). `total_variance_ratio`
reports that mismatch per unit instead of asserting it away.

A non-positive or non-finite Figure 2 rate-variance estimate is undefined.
Values above one are retained.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from VisionCore.covariance import extract_valid_segments


FIG2_FIXATION_RADIUS = 0.5
FIG2_MIN_SEGMENT_BINS = 36


def _window_contract():
    """Read the shared Figure 2 / Figure 3 counting-window contract."""
    import sys
    from pathlib import Path
    from VisionCore.paths import VISIONCORE_ROOT
    covd = str(Path(VISIONCORE_ROOT) / "paper" / "covariance_decomposition")
    if covd not in sys.path:
        sys.path.insert(0, covd)
    import fig3_windows
    return fig3_windows


# Figure 2 uses a fixed matching history at every counting window, so the
# one-bin window this module scores on carries a 3-bin history, not a 1-bin one.
# Both come from `covariance_decomposition/fig3_windows.py` -- restating them
# here is what let them drift apart from Figure 2 once already.
FIG2_HISTORY_BINS = _window_contract().fig2_history_bins()
FIG2_COUNT_BINS = _window_contract().FIG3_SINGLETRIAL_WINDOW_BINS

# A unit's numerator is a difference of two sample variances over the same bins,
# so its relative sampling error scales as sqrt(2/(n-1)) -- about 20% at this
# floor. The floor is here to reject a degenerate variance, not to select
# sessions: it sits well below every session's window count, and the difference
# in precision across the plausible range for it (14% at 100 samples, 20% at 50)
# is far smaller than the spread the panel already shows. Below it the captured
# variance is reported as undefined rather than plotted.
MIN_SCORED_WINDOWS = 50


def matched_count_indices(
    valid_mask,
    *,
    min_segment_bins=FIG2_MIN_SEGMENT_BINS,
    history_bins=FIG2_HISTORY_BINS,
    count_bins=FIG2_COUNT_BINS,
):
    """Return trial and count-bin indices using Figure 2 window semantics."""
    segments = extract_valid_segments(valid_mask, min_len_bins=min_segment_bins)
    total_bins = history_bins + count_bins
    trial_indices = []
    time_indices = []
    for trial, start, stop in segments:
        if stop - start < total_bins:
            continue
        starts = np.arange(start, stop - total_bins + 1, count_bins, dtype=int)
        count_offsets = np.arange(history_bins, total_bins, dtype=int)
        for t0 in starts:
            trial_indices.extend([trial] * count_bins)
            time_indices.extend((t0 + count_offsets).tolist())
    return np.asarray(trial_indices, dtype=int), np.asarray(time_indices, dtype=int)


def figure2_valid_mask(robs, eyepos, *, fixation_radius=FIG2_FIXATION_RADIUS):
    """Reconstruct Figure 2's shared finite-response and fixation mask."""
    robs = np.asarray(robs)
    eyepos = np.asarray(eyepos)
    if robs.ndim != 3 or eyepos.shape != robs.shape[:2] + (2,):
        raise ValueError("expected robs=(trial,time,unit), eyepos=(trial,time,2)")
    finite_response = np.isfinite(robs).all(axis=2)
    finite_eye = np.isfinite(eyepos).all(axis=2)
    central = np.hypot(eyepos[..., 0], eyepos[..., 1]) < fixation_radius
    return finite_response & finite_eye & central


def matched_model_valid_mask(robs, eyepos, dfs):
    """Figure 2 fixation support intersected with shared model validity."""
    robs = np.asarray(robs)
    dfs = np.asarray(dfs)
    if dfs.shape != robs.shape:
        raise ValueError("dfs must match robs")
    model_valid = np.isfinite(dfs).all(axis=2) & (dfs > 0).all(axis=2)
    return figure2_valid_mask(robs, eyepos) & model_valid


def matched_model_sample_indices(
    robs,
    eyepos,
    model_valid,
    *,
    min_segment_bins=FIG2_MIN_SEGMENT_BINS,
):
    """Select Figure 2 windows first, then apply a model-validity mask."""
    model_valid = np.asarray(model_valid, dtype=bool)
    if model_valid.shape != np.asarray(robs).shape[:2]:
        raise ValueError("model_valid must have shape (trial,time)")
    trial, time = matched_count_indices(
        figure2_valid_mask(robs, eyepos),
        min_segment_bins=min_segment_bins,
    )
    keep = model_valid[trial, time]
    return trial[keep], time[keep]


def _sample_variance(values):
    """Sample variance of a 1-D array, NaN with fewer than two values."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.nan
    return float(np.var(values, ddof=1))


def compute_matched_captured_variance(
    robs,
    predictions: Mapping[str, np.ndarray],
    eyepos,
    dfs,
    *,
    min_segment_bins=FIG2_MIN_SEGMENT_BINS,
    count_bins=FIG2_COUNT_BINS,
    min_scored_windows=MIN_SCORED_WINDOWS,
):
    """Captured count variance per unit on Figure 2-matched, model-valid windows.

    Figure 2's segment rule is applied to the fixation frame first and model
    validity filters the extracted samples afterwards, so the twin's 33-frame
    history does not split an otherwise eligible fixation segment.

    Every condition is scored on one common per-unit window set -- the windows
    where the response, the data filter, and all predictions are valid. Var(y)
    and Var(y - yhat) then refer to the same samples, which is what makes their
    difference a captured variance, and the conditions stay comparable.

    ``count_bins`` is Figure 2's counting window. Above one bin, the response and
    every prediction are SUMMED over the window's bins, because Figure 2's
    ``extract_windows`` sums ``robs`` over the counting window and so
    ``diag(Crate)`` -- the denominator this numerator is divided by -- is a
    variance of summed counts. Averaging instead would put the numerator a factor
    ``count_bins**2`` below its denominator. A window is scored only if every one
    of its bins is valid, so a summed count and its summed prediction always
    refer to the same complete window.
    """
    robs = np.asarray(robs, dtype=float)
    eyepos = np.asarray(eyepos, dtype=float)
    dfs = np.asarray(dfs)
    predictions = {key: np.asarray(value, dtype=float)
                   for key, value in predictions.items()}
    if robs.ndim != 3 or dfs.shape != robs.shape:
        raise ValueError("robs and dfs must both be (trial,time,unit)")
    if any(value.shape != robs.shape for value in predictions.values()):
        raise ValueError("every prediction must match robs")
    n_units = robs.shape[2]

    trial, time = matched_count_indices(
        figure2_valid_mask(robs, eyepos), min_segment_bins=min_segment_bins,
        count_bins=count_bins,
    )
    n_base_windows = int(trial.size // count_bins)

    var_y = np.full(n_units, np.nan)
    n_windows = np.zeros(n_units, dtype=int)
    captured = {key: np.full(n_units, np.nan) for key in predictions}
    var_residual = {key: np.full(n_units, np.nan) for key in predictions}

    if n_base_windows:
        def _by_window(arr):
            """(n_base_windows, count_bins, n_units) view of the chosen bins.

            `matched_count_indices` emits each window's bins contiguously, so the
            reshape recovers the window grouping.
            """
            return arr[trial, time].reshape(n_base_windows, count_bins, n_units)

        y_bins = _by_window(robs)
        pred_bins = {key: _by_window(value) for key, value in predictions.items()}
        df_bins = _by_window(dfs)

        bin_valid = np.isfinite(y_bins) & np.isfinite(df_bins) & (df_bins > 0)
        for value in pred_bins.values():
            bin_valid &= np.isfinite(value)
        score_valid = bin_valid.all(axis=1)

        y = y_bins.sum(axis=1)
        pred = {key: value.sum(axis=1) for key, value in pred_bins.items()}

        for unit in range(n_units):
            keep = score_valid[:, unit]
            n_windows[unit] = int(keep.sum())
            if n_windows[unit] < min_scored_windows:
                continue
            y_unit = y[keep, unit]
            var_y[unit] = _sample_variance(y_unit)
            for key, value in pred.items():
                var_residual[key][unit] = _sample_variance(
                    y_unit - value[keep, unit]
                )
                captured[key][unit] = var_y[unit] - var_residual[key][unit]

    return {
        "captured_variance": captured,
        "var_residual": var_residual,
        "var_y": var_y,
        "n_windows": n_windows,
        "n_base_windows": n_base_windows,
        "n_units_below_floor": int(
            np.sum(n_windows < min_scored_windows)
        ),
    }


def explainable_fraction(captured_variance, c_rate):
    """Divide captured variance by a rate variance, NaN where it is unusable."""
    c_rate = np.asarray(c_rate, dtype=float)
    out = {}
    usable = np.isfinite(c_rate) & (c_rate > 0)
    for key, captured in captured_variance.items():
        captured = np.asarray(captured, dtype=float)
        if captured.shape != c_rate.shape:
            raise ValueError("captured variance and c_rate shapes differ")
        out[key] = np.divide(
            captured, c_rate, out=np.full_like(c_rate, np.nan), where=usable
        )
    return out


def total_variance_ratio(var_y, c_total):
    """Matched Var(y) over Figure 2 diag(Ctotal): the sampling-frame diagnostic.

    The two are measured on different sample sets -- the numerator drops every
    bin the twin cannot predict -- so this ratio is reported, not asserted.
    """
    var_y = np.asarray(var_y, dtype=float)
    c_total = np.asarray(c_total, dtype=float)
    if var_y.shape != c_total.shape:
        raise ValueError("var_y and c_total shapes differ")
    usable = np.isfinite(var_y) & np.isfinite(c_total) & (c_total > 0)
    return np.divide(
        var_y, c_total, out=np.full_like(c_total, np.nan), where=usable
    )


def _estimate_rate_variance_from_samples(counts, trajectories, time_index):
    """Apply Figure 2's production estimator to an already filtered sample set."""
    import sys
    from pathlib import Path

    from VisionCore.covariance import decompose_trajectory
    from VisionCore.paths import VISIONCORE_ROOT

    package_dir = Path(VISIONCORE_ROOT) / "paper" / "covariance_decomposition"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from decompose import (
        CLOSEPAIR_DENSITY_DEFAULT,
        CPSTH_METHOD_DEFAULT,
        N_BOOT_DEFAULT,
        THRESHOLD_DEFAULT,
        TIME_BIN_WEIGHTING_DEFAULT,
        WEIGHT_CLIP_DEFAULT,
        _uncentred_crate,
    )

    real = decompose_trajectory(
        counts,
        trajectories,
        time_index,
        target="full",
        threshold=THRESHOLD_DEFAULT,
        weight_clip=WEIGHT_CLIP_DEFAULT,
        time_bin_weighting=TIME_BIN_WEIGHTING_DEFAULT,
        cpsth_method=CPSTH_METHOD_DEFAULT,
        n_boot=N_BOOT_DEFAULT,
        seed=42,
        min_trials_per_time_bin=10,
        closepair_density=CLOSEPAIR_DENSITY_DEFAULT,
    )
    # No shuffle null here, so `pair_density_ok` is informational only; the real
    # close-pair set is never small enough to make it False (see _fit_pair_density).
    c_rate, n_pairs, _, _, _, _pair_density_ok = _uncentred_crate(
        counts,
        trajectories,
        time_index,
        "full",
        THRESHOLD_DEFAULT,
        Erate=real["Erate"],
        time_bin_weighting=TIME_BIN_WEIGHTING_DEFAULT,
        weight_clip=WEIGHT_CLIP_DEFAULT,
        closepair_density=CLOSEPAIR_DENSITY_DEFAULT,
    )
    # Ctotal under Figure 2's pair-count time-bin weighting, centred on the same
    # weighted Erate as Cpsth/Crate (decompose_trajectory computes it that way).
    # Matches paper/covariance_decomposition/decompose.py; the previous
    # unweighted np.cov here left Figure 3 on a different Ctotal convention
    # than Figure 2.
    return real["Ctotal"], c_rate, int(n_pairs)


def estimate_matched_rate_variance(
    robs,
    eyepos,
    dfs,
    *,
    min_segment_bins=FIG2_MIN_SEGMENT_BINS,
    min_group_windows=MIN_SCORED_WINDOWS,
):
    """DIAGNOSTIC ONLY: re-run Figure 2's estimator on model-valid samples.

    This is the abandoned denominator, kept so each sweep can report how far it
    drifts from Figure 2's. It is estimated per unit-validity group because the
    close-pair estimator needs one shared sample set across the units it scores.
    A group that cannot support the estimator is left NaN; nothing in the
    production score depends on the result.
    """
    robs = np.asarray(robs, dtype=float)
    eyepos = np.asarray(eyepos, dtype=float)
    dfs = np.asarray(dfs)
    if robs.ndim != 3 or dfs.shape != robs.shape:
        raise ValueError("robs and dfs must both be (trial,time,unit)")
    n_units = robs.shape[2]

    base_trial, base_time = matched_count_indices(
        figure2_valid_mask(robs, eyepos), min_segment_bins=min_segment_bins
    )
    c_total = np.full(n_units, np.nan)
    c_rate = np.full(n_units, np.nan)
    n_windows = np.zeros(n_units, dtype=int)
    n_close_pairs = np.zeros(n_units, dtype=int)
    n_groups = 0
    n_groups_excluded = 0
    if base_trial.size == 0:
        return {
            "c_total": c_total, "c_rate": c_rate, "n_windows": n_windows,
            "n_close_pairs": n_close_pairs, "n_validity_groups": 0,
            "n_validity_groups_excluded": 0,
        }

    counts = robs[base_trial, base_time]
    trajectories = np.stack(
        [eyepos[base_trial, base_time - 1], eyepos[base_trial, base_time]],
        axis=1,
    )
    model_valid = np.isfinite(dfs[base_trial, base_time]) & (
        dfs[base_trial, base_time] > 0
    )
    groups_by_mask = {}
    for unit in range(n_units):
        groups_by_mask.setdefault(model_valid[:, unit].tobytes(), []).append(unit)

    for units in groups_by_mask.values():
        n_groups += 1
        group = np.asarray(units, dtype=int)
        keep = model_valid[:, int(group[0])]
        n_windows[group] = int(keep.sum())
        group_time = base_time[keep]
        if keep.sum() < min_group_windows:
            n_groups_excluded += 1
            continue
        _, trials_per_phase = np.unique(group_time, return_counts=True)
        if not np.any(trials_per_phase >= 10):
            n_groups_excluded += 1
            continue
        reference_total, reference_rate, pairs = _estimate_rate_variance_from_samples(
            counts[keep], trajectories[keep], group_time
        )
        c_total[group] = np.diag(reference_total)[group]
        c_rate[group] = np.diag(reference_rate)[group]
        n_close_pairs[group] = pairs

    return {
        "c_total": c_total,
        "c_rate": c_rate,
        "n_windows": n_windows,
        "n_close_pairs": n_close_pairs,
        "n_validity_groups": n_groups,
        "n_validity_groups_excluded": n_groups_excluded,
    }
