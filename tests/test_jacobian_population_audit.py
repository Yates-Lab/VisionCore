import numpy as np

from paper.model_selection.audit_jacobian_population import (
    _auto_units,
    _bootstrap_median,
    _bootstrap_paired_median_difference,
)


def test_auto_unit_panel_is_even_and_uses_only_validity() -> None:
    valid = np.ones((32, 11), dtype=bool)
    valid[:, 5] = False

    selected = _auto_units(valid, 5)

    assert np.array_equal(selected, np.array([0, 2, 4, 8, 10]))
    assert 5 not in selected


def test_bootstrap_median_is_deterministic_and_contains_point_estimate() -> None:
    values = np.arange(1.0, 10.0)

    first = _bootstrap_median(values, seed=4, n_boot=500)
    second = _bootstrap_median(values, seed=4, n_boot=500)

    assert first == second
    assert first["median"] == 5.0
    assert first["ci95_low"] <= first["median"] <= first["ci95_high"]


def test_paired_bootstrap_uses_only_common_finite_units() -> None:
    reference = np.asarray([1.0, 2.0, np.nan, 4.0])
    candidate = np.asarray([1.5, 1.0, 8.0, 5.0])

    result = _bootstrap_paired_median_difference(
        candidate, reference, seed=9, n_boot=500
    )

    assert result["n_paired_units"] == 3
    assert result["median_difference"] == 0.5
    assert result["ci95_low"] <= result["median_difference"] <= result["ci95_high"]
