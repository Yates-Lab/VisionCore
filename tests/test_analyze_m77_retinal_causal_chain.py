from __future__ import annotations

import numpy as np

from paper.fig4.spatiotemporal_tuning.analyze_m77_retinal_causal_chain import (
    crossed_predictions,
    hierarchical_effect_ci,
    matched_motion_delta,
    paired_bootstrap_median_difference,
    prediction_metrics,
)


def test_matched_motion_delta_uses_each_pairs_stabilized_condition() -> None:
    value = np.asarray(
        [
            [
                [[10.0], [12.0], [15.0]],
                [[20.0], [21.0], [24.0]],
            ]
        ]
    )
    result = matched_motion_delta(value)
    np.testing.assert_allclose(result[..., 0], [[[2.0, 5.0], [1.0, 4.0]]])


def test_crossed_prediction_generalizes_additive_scalar_relationship() -> None:
    image = np.linspace(-1.0, 1.0, 10)[:, None, None]
    trace = np.linspace(-0.5, 0.5, 10)[None, :, None]
    scale = np.asarray((0.5, 1.0, 2.0))[None, None]
    feature = (image + trace + 0.3) * scale
    outcome = 1.2 + 2.5 * feature
    prediction = crossed_predictions(feature, outcome, image_folds=5, trace_folds=5)
    assert np.isfinite(prediction).all()
    metrics = prediction_metrics(outcome, prediction)
    assert metrics["spearman"] > 0.999
    assert metrics["cv_r2"] > 0.999
    assert abs(metrics["calibration_slope"] - 1.0) < 1e-6


def test_crossed_prediction_does_not_leak_image_or_trace_offsets() -> None:
    rng = np.random.default_rng(4)
    feature = rng.normal(size=(10, 10, 3))
    image_offset = rng.normal(size=(10, 1, 1))
    trace_offset = rng.normal(size=(1, 10, 1))
    outcome = np.broadcast_to(image_offset + trace_offset, feature.shape)
    prediction = crossed_predictions(feature, outcome, image_folds=5, trace_folds=5)
    metrics = prediction_metrics(outcome, prediction)
    assert metrics["cv_r2"] < 0.1


def test_hierarchical_effect_ci_contains_constant_effect() -> None:
    value = np.full((8, 9, 7), 3.5)
    center, low, high = hierarchical_effect_ci(
        value, n_bootstrap=50, rng=np.random.default_rng(1)
    )
    assert center == low == high == 3.5


def test_paired_bootstrap_difference_preserves_pairing() -> None:
    first = np.arange(20, dtype=float) + 2.0
    second = np.arange(20, dtype=float)
    center, low, high = paired_bootstrap_median_difference(
        first, second, n_bootstrap=100, rng=np.random.default_rng(2)
    )
    assert center == low == high == 2.0
