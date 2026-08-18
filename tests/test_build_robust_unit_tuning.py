import numpy as np
import pandas as pd
import pytest

from paper.fig4.upstream.build_robust_unit_tuning import (
    build_robust_tuning_table,
    production_sf_mask,
    robust_orientation_metadata,
)


def _tables():
    base = pd.DataFrame(
        {
            "unit_index": list(range(8)),
            "unit_label": [f"u{i:03d}" for i in range(8)],
            "rr100_active": [True] * 6 + [False, False],
        }
    )
    robust = pd.DataFrame(
        {
            "unit_index": [5, 4, 3, 2, 1, 0],
            "weighted_center_sf_cpd": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "preferred_sf_cpd": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "preferred_tf_hz": [8.0] * 6,
            "peak_censored": [False, False, False, False, True, True],
            "peak_censoring": ["none", "none", "none", "none", "low_sf", "low_sf"],
        }
    )
    return base, robust


def test_robust_groups_are_stable_tertile_tails_and_keep_inactive_rows():
    output, audit = build_robust_tuning_table(*_tables())
    by_unit = output.set_index("unit_index")

    assert by_unit.loc[[0, 1], "sf_group"].tolist() == ["low_sf", "low_sf"]
    assert by_unit.loc[[2, 3], "sf_group"].tolist() == ["middle_sf", "middle_sf"]
    assert by_unit.loc[[4, 5], "sf_group"].tolist() == ["high_sf", "high_sf"]
    assert by_unit.loc[[6, 7], "sf_group"].tolist() == ["inactive", "inactive"]
    assert np.isnan(by_unit.loc[6, "sf_split_metric"])
    assert audit["n_active_units"] == 6
    assert audit["n_tail"] == 2
    assert audit["fraction_peak_censored"] == pytest.approx(2 / 6)


def test_robust_groups_reject_missing_active_unit():
    base, robust = _tables()
    robust = robust[robust["unit_index"] != 2]
    with pytest.raises(ValueError, match="missing active RR100"):
        build_robust_tuning_table(base, robust)


def test_robust_groups_reject_tuning_for_inactive_unit():
    base, robust = _tables()
    robust = pd.concat(
        [robust, robust.iloc[[0]].assign(unit_index=6)], ignore_index=True
    )
    with pytest.raises(ValueError, match="inactive RR100"):
        build_robust_tuning_table(base, robust)


def test_production_mask_table_mode_uses_explicit_tertiles():
    units = pd.DataFrame(
        {
            "sf_group": ["low_sf", "middle_sf", "high_sf", "inactive"],
            # Deliberately contradict the historical thresholds. The explicit
            # selected-twin grouping must win in table mode.
            "sf_split_metric": [9.0, 0.1, 0.2, 20.0],
        }
    )
    assert production_sf_mask(units, group="low", mode="table_tertiles").tolist() == [
        True, False, False, False
    ]
    assert production_sf_mask(units, group="high", mode="table_tertiles").tolist() == [
        False, False, True, False
    ]


def test_production_mask_default_mode_preserves_absolute_thresholds():
    units = pd.DataFrame(
        {
            "sf_group": ["high_sf", "low_sf", "middle_sf"],
            "sf_split_metric": [0.2, 0.6, 0.9],
        }
    )
    assert production_sf_mask(
        units, group="low", mode="absolute_cpd", low_max_cpd=0.5
    ).tolist() == [True, False, False]
    assert production_sf_mask(
        units, group="high", mode="absolute_cpd", high_min_cpd=0.75
    ).tolist() == [False, False, True]


def test_orientation_metadata_uses_dynamic_rms_and_doubled_angle_vector():
    grouped = pd.DataFrame(
        {
            "unit_index": [0] * 5,
            "probe_orientation_deg": [0.0, 0.0, 45.0, 90.0, 135.0],
            "temporal_hz": [0.0, 8.0, 8.0, 8.0, 8.0],
            "response_amp_rms": [100.0, 4.0, 1.0, 0.0, 1.0],
        }
    )
    row = robust_orientation_metadata(grouped).iloc[0]
    assert row["prior_preferred_orientation_deg"] == 0.0
    # Dynamic weights are [4, 1, 0, 1], whose doubled-angle vector has
    # magnitude 4/6. The huge TF=0 row must not enter the estimate.
    assert row["prior_orientation_selectivity_index"] == pytest.approx(4 / 6)
