"""Tests for Figure 3's explainable-rate-variance score.

The production helper lives beside the figure scripts. These tests pin the
sampling and normalization contracts without loading model or data caches.
The score is a captured count variance measured on Figure 2-matched,
model-valid windows over Figure 2's own diag(Crate); nothing in the scored path
re-estimates that denominator.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG3_DIR = REPO_ROOT / "paper" / "fig3"
for _path in (str(REPO_ROOT), str(FIG3_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def test_matched_count_indices_follow_figure2_window_semantics():
    """One-bin counts follow one history bin within contiguous valid segments."""
    from _fig3_explainable_variance import matched_count_indices

    valid = np.array([
        [False, True, True, True, True, True, False],
        [False, True, True, True, False, False, False],
    ])

    trial, time = matched_count_indices(
        valid, min_segment_bins=4, history_bins=1, count_bins=1
    )

    assert_allclose(trial, [0, 0, 0, 0])
    assert_allclose(time, [2, 3, 4, 5])


def test_model_valid_mask_is_shared_across_scored_units():
    """The shared mask the diagnostic estimator needs drops any invalid unit."""
    from _fig3_explainable_variance import matched_model_valid_mask

    robs = np.ones((1, 5, 2))
    eyepos = np.zeros((1, 5, 2))
    dfs = np.ones_like(robs)
    dfs[0, 2, 1] = 0
    eyepos[0, 3, 0] = 0.6

    valid = matched_model_valid_mask(robs, eyepos, dfs)

    assert valid.tolist() == [[True, True, False, False, True]]


def test_model_validity_filters_windows_after_fixation_segment_selection():
    """History-invalid bins do not split an otherwise eligible fixation segment."""
    from _fig3_explainable_variance import matched_model_sample_indices

    robs = np.ones((1, 6, 1))
    eyepos = np.zeros((1, 6, 2))
    dfs = np.zeros_like(robs)
    dfs[:, 3:, :] = 1

    trial, time = matched_model_sample_indices(
        robs, eyepos, dfs[..., 0], min_segment_bins=4
    )

    assert trial.tolist() == [0, 0, 0]
    assert time.tolist() == [3, 4, 5]


def _one_unit_case():
    """One trial, one unit, six bins; the first three supply Figure 2 history."""
    robs = np.array([[[0.0], [1.0], [0.0], [2.0], [1.0], [3.0]]])
    eyepos = np.zeros(robs.shape[:2] + (2,))
    dfs = np.ones_like(robs)
    return robs, eyepos, dfs


def test_captured_variance_has_oracle_and_constant_references():
    """An oracle captures all count variance; a constant captures none."""
    from _fig3_explainable_variance import (
        compute_matched_captured_variance, explainable_fraction,
    )

    robs, eyepos, dfs = _one_unit_case()
    var_y = np.var(robs[0, 3:, 0], ddof=1)

    scored = compute_matched_captured_variance(
        robs,
        {"oracle": robs.copy(), "constant": np.full_like(robs, 0.5)},
        eyepos,
        dfs,
        min_segment_bins=2,
        min_scored_windows=2,
    )
    fraction = explainable_fraction(scored["captured_variance"], [var_y])

    assert_allclose(scored["var_y"], [var_y])
    assert_allclose(scored["n_windows"], [3])
    assert_allclose(scored["captured_variance"]["oracle"], [var_y])
    assert_allclose(scored["captured_variance"]["constant"], [0.0])
    assert_allclose(fraction["oracle"], [1.0])
    assert_allclose(fraction["constant"], [0.0])


def test_every_condition_is_scored_on_one_common_bin_set():
    """A bin missing from one prediction leaves Var(y) and every residual."""
    from _fig3_explainable_variance import compute_matched_captured_variance

    robs, eyepos, dfs = _one_unit_case()
    partial = robs.copy()
    partial[0, 3, 0] = np.nan          # one condition cannot predict this bin
    kept = np.array([4, 5])

    scored = compute_matched_captured_variance(
        robs,
        {"full": robs.copy(), "partial": partial},
        eyepos,
        dfs,
        min_segment_bins=2,
        min_scored_windows=2,
    )

    assert_allclose(scored["n_windows"], [2])
    assert_allclose(scored["var_y"], [np.var(robs[0, kept, 0], ddof=1)])
    # Both conditions are oracles on the bins that survive, so both capture all
    # of the common-mask Var(y) -- the conditions remain comparable.
    assert_allclose(scored["captured_variance"]["full"], scored["var_y"])
    assert_allclose(scored["captured_variance"]["partial"], scored["var_y"])


def test_each_unit_keeps_its_own_model_valid_bins():
    """Numerators are per unit: one unit's missing bin does not shrink another."""
    from _fig3_explainable_variance import compute_matched_captured_variance

    robs, eyepos, dfs = _one_unit_case()
    robs = np.repeat(robs, 2, axis=2)
    dfs = np.repeat(dfs, 2, axis=2)
    dfs[0, 3, 1] = 0

    scored = compute_matched_captured_variance(
        robs, {"oracle": robs.copy()}, eyepos, dfs,
        min_segment_bins=2, min_scored_windows=2,
    )

    assert scored["n_windows"].tolist() == [3, 2]
    assert_allclose(scored["var_y"][0], np.var(robs[0, 3:, 0], ddof=1))
    assert_allclose(
        scored["var_y"][1], np.var(robs[0, [4, 5], 1], ddof=1)
    )


def test_units_below_the_window_floor_are_undefined():
    """Too few scored bins is reported as undefined rather than plotted."""
    from _fig3_explainable_variance import compute_matched_captured_variance

    robs, eyepos, dfs = _one_unit_case()

    scored = compute_matched_captured_variance(
        robs, {"oracle": robs.copy()}, eyepos, dfs,
        min_segment_bins=2, min_scored_windows=6,
    )

    assert scored["n_windows"].tolist() == [3]
    assert scored["n_units_below_floor"] == 1
    assert np.isnan(scored["var_y"][0])
    assert np.isnan(scored["captured_variance"]["oracle"][0])


def test_nonpositive_rate_variance_is_undefined():
    """An unusable Figure 2 denominator is excluded rather than floored."""
    from _fig3_explainable_variance import explainable_fraction

    fraction = explainable_fraction(
        {"oracle": np.array([1.0, 1.0, 1.0])}, np.array([0.0, -0.5, np.nan])
    )

    assert np.all(np.isnan(fraction["oracle"]))


def test_fraction_above_one_is_retained():
    """Finite-sample values above one remain visible rather than being clipped."""
    from _fig3_explainable_variance import explainable_fraction

    fraction = explainable_fraction({"oracle": np.array([1.0])}, np.array([0.5]))

    assert_allclose(fraction["oracle"], [2.0])


def test_total_variance_ratio_reports_the_sampling_mismatch():
    """Matched Var(y) over Figure 2 Ctotal is reported, not asserted."""
    from _fig3_explainable_variance import total_variance_ratio

    ratio = total_variance_ratio(
        np.array([0.9, 2.0, np.nan]), np.array([1.0, 0.0, 3.0])
    )

    assert_allclose(ratio[0], 0.9)
    assert np.isnan(ratio[1])
    assert np.isnan(ratio[2])


def test_matched_rate_estimator_is_diagnostic_and_never_raises():
    """The abandoned denominator is reported per validity group, NaN if unusable."""
    from _fig3_explainable_variance import (
        estimate_matched_rate_variance, matched_count_indices,
        matched_model_valid_mask,
    )

    rng = np.random.default_rng(8)
    n_trials, n_time, n_units = 12, 12, 3
    rates = np.linspace(0.1, 0.8, n_time)[None, :, None]
    robs = rng.poisson(
        np.broadcast_to(rates, (n_trials, n_time, n_units))
    ).astype(float)
    eyepos = rng.normal(0, 0.005, size=(n_trials, n_time, 2))
    dfs = np.ones_like(robs)
    dfs[:, :, 2] = 0                    # this unit has no model-valid support

    estimated = estimate_matched_rate_variance(
        robs, eyepos, dfs, min_segment_bins=4, min_group_windows=10
    )
    trial, time = matched_count_indices(
        matched_model_valid_mask(robs[:, :, :2], eyepos, dfs[:, :, :2]),
        min_segment_bins=4,
    )

    assert estimated["n_validity_groups"] == 2
    assert estimated["n_validity_groups_excluded"] == 1
    assert_allclose(
        estimated["c_total"][:2],
        np.var(robs[trial, time][:, :2], axis=0, ddof=1),
        rtol=1e-12,
    )
    assert np.isnan(estimated["c_rate"][2])
    assert estimated["n_windows"][2] == 0


def _ablation_result(with_explainable=True):
    conditions = ("intact", "zeroed", "stabilized")
    result = {
        "session": "Allen_test",
        "subject": "Allen",
        "neuron_mask": np.array([3, 7]),
        "n_neurons": 2,
        "ve": {c: np.array([0.1, 0.2]) for c in conditions},
        "ve_psth": np.array([0.05, 0.06]),
        "ccmax": np.array([0.8, 0.9]),
        "alpha": np.array([0.3, 0.4]),
    }
    if with_explainable:
        order = ("psth",) + conditions
        result.update({
            "explainable_fraction": {
                c: np.array([i + 0.1, i + 0.2]) for i, c in enumerate(order)
            },
            "captured_variance": {
                c: np.array([i + 0.01, i + 0.02]) for i, c in enumerate(order)
            },
            "matched_var_y": np.array([0.9, 1.8]),
            "matched_n_windows": np.array([500, 400]),
            "fig2_c_rate": np.array([0.25, 0.45]),
            "fig2_c_total": np.array([1.0, 2.0]),
            "matched_c_rate": np.array([0.2, -0.1]),
            "matched_c_total": np.array([0.9, 1.8]),
            "matched_n_close_pairs": np.array([120, 130]),
        })
    return result


def test_ablation_aggregate_carries_scores_sensitivity_and_sessions():
    """The render cache exposes PSTH and model fractions in aligned cell order."""
    from _fig3_ablation_data import aggregate

    agg = aggregate([_ablation_result()])

    assert set(agg["explainable_fraction"]) == {
        "psth", "intact", "zeroed", "stabilized"
    }
    assert_allclose(agg["explainable_fraction"]["psth"], [0.1, 0.2])
    assert_allclose(agg["explainable_fraction"]["stabilized"], [3.1, 3.2])
    assert_allclose(agg["fig2_c_rate"], [0.25, 0.45])
    # The sensitivity view divides the same numerators by the abandoned
    # denominator, and is undefined where that estimate is not a variance.
    assert_allclose(agg["explainable_fraction_matched"]["psth"][0], 0.01 / 0.2)
    assert np.isnan(agg["explainable_fraction_matched"]["psth"][1])
    assert_allclose(agg["total_variance_ratio"], [0.9, 0.9])
    assert agg["sessions"].tolist() == ["Allen_test", "Allen_test"]


def test_ablation_aggregate_rejects_cache_without_matched_scores():
    """An old summary cache cannot reconstruct the metric without predictions."""
    from _fig3_ablation_data import aggregate

    with pytest.raises(RuntimeError, match="--recompute"):
        aggregate([_ablation_result(with_explainable=False)])
