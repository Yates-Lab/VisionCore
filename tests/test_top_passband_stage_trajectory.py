import numpy as np

from paper.fig4.spatiotemporal_tuning.analyze_top_passband_stage_trajectory import (
    select_top_traces,
    unit_top_bin_effects,
    unit_top_bin_normalized_effects,
)


def test_select_top_traces_stratifies_full_top_bin_and_covers_units():
    percentile = np.tile(np.linspace(0.25, 99.75, 200)[:, None], (1, 5))
    rows, membership = select_top_traces(
        percentile,
        n_traces=10,
        threshold=80.0,
    )
    selected = percentile[rows, 0]
    assert len(np.unique(rows)) == 10
    assert selected.min() < 84.0
    assert selected.max() > 97.0
    assert membership[rows].all(axis=0).all()


def test_unit_top_bin_effects_pool_images_before_percent_change():
    rate = np.ones((2, 3, 2, 3, 4), dtype=float)
    expected = np.ones_like(rate)
    ssi = np.ones_like(rate)
    rate[:, :, 1] *= 1.10
    ssi[:, :, 1] *= 1.20
    membership = np.ones((3, 4), dtype=bool)
    unit_rate, unit_ssi = unit_top_bin_effects(
        rate,
        expected,
        ssi,
        membership,
    )
    np.testing.assert_allclose(unit_rate, 10.0)
    np.testing.assert_allclose(unit_ssi, 20.0)


def test_normalized_effects_report_modulation_points_and_absolute_ssi():
    temporal = np.zeros((2, 3, 2, 3, 4), dtype=float)
    temporal[:, :, 0] = 0.05
    temporal[:, :, 1] = 0.15
    expected = np.ones_like(temporal)
    ssi = np.full_like(temporal, 0.25)
    ssi[:, :, 1] = 0.30
    membership = np.ones((3, 4), dtype=bool)
    unit_temporal, unit_ssi = unit_top_bin_normalized_effects(
        temporal,
        expected,
        ssi,
        membership,
    )
    np.testing.assert_allclose(unit_temporal, 10.0)
    np.testing.assert_allclose(unit_ssi, 0.05)
