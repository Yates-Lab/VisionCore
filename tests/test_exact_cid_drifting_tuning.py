from __future__ import annotations

import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning.run_exact_cid_drifting_tuning import (
    _conditions,
    drifting_histories,
    response_metrics,
)
from paper.fig4.spatiotemporal_tuning.audit_exact_cid_drifting_tuning import (
    assign_crossed_groups,
    connected_peak_lasso,
    fit_yu_passband,
    preferred_direction,
    response_cube,
)
from paper.fig4.spatiotemporal_tuning._figure4_rendering import (
    yu_surface,
)


def _histories(**overrides) -> np.ndarray:
    arguments = {
        "spatial_cpd": 4.0,
        "temporal_hz": 8.0,
        "motion_direction_deg": 30.0,
        "endpoint_phases_rad": np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False),
        "n_lags": 60,
        "input_size": 35,
        "ppd": 37.50476617,
        "frame_rate_hz": 240.0,
        "contrast": 0.25,
        "device": "cpu",
    }
    arguments.update(overrides)
    return drifting_histories(**arguments).numpy()


def test_drifting_histories_have_native_lag_order_and_amplitude() -> None:
    phase = np.asarray([0.37])
    values = _histories(endpoint_phases_rad=phase)
    assert values.shape == (1, 1, 60, 35, 35)
    assert np.max(np.abs(values)) <= 0.25 + 1e-6

    center = values[0, 0, :, 17, 17]
    expected = 0.25 * np.cos(
        phase[0] + 2.0 * np.pi * 8.0 * np.arange(60) / 240.0
    )
    np.testing.assert_allclose(center, expected, atol=2e-6)


def test_static_opposite_directions_are_phase_relabelings() -> None:
    phases = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    forward = _histories(
        temporal_hz=0.0,
        motion_direction_deg=25.0,
        endpoint_phases_rad=phases,
    )
    opposite = _histories(
        temporal_hz=0.0,
        motion_direction_deg=205.0,
        endpoint_phases_rad=(-phases) % (2.0 * np.pi),
    )
    np.testing.assert_allclose(forward, opposite, atol=2e-6)


def test_response_metrics_keep_f0_and_phase_locking_separate() -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    rates = np.column_stack(
        (
            3.0 + 1.25 * np.cos(phase) + 0.4 * np.cos(2.0 * phase),
            5.0 + 0.75 * np.sin(phase),
        )
    )
    metrics = response_metrics(phase, rates, np.asarray([2.0, 5.5]))
    np.testing.assert_allclose(
        metrics["f0_expected_count"], [3.0, 5.0], atol=1e-12
    )
    np.testing.assert_allclose(
        metrics["delta_f0_expected_count"], [1.0, -0.5], atol=1e-12
    )
    np.testing.assert_allclose(
        metrics["f1_expected_count_amplitude"], [1.25, 0.75], atol=1e-12
    )
    np.testing.assert_allclose(
        metrics["f2_expected_count_amplitude"], [0.4, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        metrics["half_phase_f0_expected_count"],
        metrics["f0_expected_count"],
        atol=1e-12,
    )


def test_condition_table_covers_full_direction_circle_without_duplicate_endpoint() -> None:
    directions = np.linspace(0.0, 360.0, 18, endpoint=False)
    table = _conditions(np.asarray([1.0, 2.0]), np.asarray([0.0, 4.0]), directions)
    assert len(table) == 2 * 2 * 18
    assert table.motion_direction_deg.nunique() == 18
    assert table.motion_direction_deg.min() == 0.0
    assert table.motion_direction_deg.max() < 360.0
    assert table.bar_orientation_deg.nunique() == 9
    assert table.bar_orientation_deg.min() >= 0.0
    assert table.bar_orientation_deg.max() < 180.0


def test_response_cube_uses_physical_condition_coordinates() -> None:
    directions = np.asarray([0.0, 180.0])
    table = _conditions(np.asarray([1.0, 2.0]), np.asarray([0.0, 4.0]), directions)
    values = np.column_stack((table.condition_index, table.condition_index + 100.0))
    sf, tf, recovered_direction, cube = response_cube(table, values)
    np.testing.assert_array_equal(sf, [1.0, 2.0])
    np.testing.assert_array_equal(tf, [0.0, 4.0])
    np.testing.assert_array_equal(recovered_direction, directions)
    assert cube.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(cube[0, 0, 0], values[0])


def test_preferred_direction_and_lasso_preserve_one_measured_passband() -> None:
    surface = np.zeros((5, 4, 3))
    surface[1:4, 1:3, 2] = np.asarray(
        [[0.4, 0.5], [0.8, 1.0], [0.4, 0.5]]
    )
    assert preferred_direction(surface, np.asarray([False, True, True, True])) == 2
    smoothed, lasso = connected_peak_lasso(surface[:, 1:, 2].T)
    assert smoothed.shape == lasso.shape == (3, 5)
    assert lasso[1, 2]
    assert lasso.sum() >= 2


def test_yu_passband_fit_recovers_clean_interior_surface() -> None:
    sf = 2.0 ** np.arange(0.0, 4.5, 0.5)
    tf = 2.0 ** np.arange(0.0, 7.0, 0.5)
    sf_grid, tf_grid = np.meshgrid(sf, tf)
    parameters = np.asarray([1.0, np.log2(2.0), 0.7, 0.0, np.log2(8.0), 0.8, 0.0])
    surface = yu_surface(
        parameters, np.log2(sf_grid), np.log2(tf_grid), inseparable=False
    )
    result = fit_yu_passband(sf, tf, surface)
    assert result["fit_success"]
    assert result["selected_r2"] > 0.999
    assert abs(np.log2(result["preferred_sf_cpd"] / 2.0)) < 0.05
    assert abs(np.log2(result["preferred_tf_hz"] / 8.0)) < 0.05


def test_crossed_groups_require_recorded_and_twin_sf_agreement() -> None:
    n = 90
    yu_sf = np.geomspace(1.0, 8.0, n)
    yu_tf = np.geomspace(32.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "recorded_data_preferred_sf_cpd": yu_sf,
            "yu_preferred_sf_cpd": yu_sf,
            "yu_preferred_tf_hz": yu_tf,
            "recorded_sf_data_reliable": True,
            "validated_model_sf_tf": True,
            "validated_for_figure4": True,
        }
    )
    grouped, report = assign_crossed_groups(frame)
    assert report["validated_counts"]["low recorded SF / high twin TF"] > 0
    assert report["validated_counts"]["high recorded SF / low twin TF"] > 0
    assert grouped.loc[0, "crossed_group"] == "low recorded SF / high twin TF"
    assert grouped.loc[n - 1, "crossed_group"] == "high recorded SF / low twin TF"

    contradicted = frame.copy()
    contradicted["recorded_data_preferred_sf_cpd"] = contradicted[
        "recorded_data_preferred_sf_cpd"
    ].to_numpy()[::-1]
    regrouped, _ = assign_crossed_groups(contradicted)
    assert regrouped.crossed_group.eq("middle").all()
