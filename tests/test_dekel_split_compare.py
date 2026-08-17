import numpy as np
import pytest

from paper.model_selection.compare_dekel_split_evaluations import (
    align_per_unit,
    clipped_session_difference,
    complementarity_summary,
    hierarchical_bootstrap,
)
from paper.model_selection.evaluate_dekel_split import (
    rate_contract,
    write_per_unit_archive,
)


def test_rate_contract_supports_legacy_and_mixed_rate_configs():
    assert rate_contract(
        {"sampling": {"source_rate": 240, "target_rate": 120}}
    ) == (120, 120, "dataset-rate spike-count targets")
    assert rate_contract(
        {
            "sampling": {"source_rate": 240, "target_rate": 240},
            "supervision": {"target_rate": 120, "phase": 1},
        }
    ) == (240, 120, "pre-binned causal spike-count targets")


def test_align_per_unit_matches_cids_and_drops_nonfinite_pairs():
    reference = {
        "s1": {10: 0.1, 20: np.nan, 30: -0.2},
        "s2": {4: 0.5},
    }
    candidate = {
        "s1": {30: 0.3, 20: 0.2, 10: 0.4, 99: 1.0},
        "s2": {4: 0.25},
    }

    aligned = align_per_unit(reference, candidate)

    assert aligned["s1"]["cids"].tolist() == [10, 30]
    assert aligned["s1"]["reference"].tolist() == pytest.approx([0.1, -0.2])
    assert aligned["s1"]["candidate"].tolist() == pytest.approx([0.4, 0.3])
    assert aligned["s2"]["cids"].tolist() == [4]


def test_clipped_difference_matches_training_reduction():
    reference = np.asarray([-2.0, 0.2, 0.4])
    candidate = np.asarray([-1.0, 0.5, 0.1])

    difference = clipped_session_difference(reference, candidate)

    assert difference == pytest.approx((0.0 + 0.5 + 0.1) / 3 - (0.0 + 0.2 + 0.4) / 3)


def test_hierarchical_bootstrap_reports_exact_observed_session_mean():
    aligned = {
        "s1": {
            "reference": np.asarray([0.1, 0.2]),
            "candidate": np.asarray([0.3, 0.4]),
        },
        "s2": {
            "reference": np.asarray([-0.1, 0.5]),
            "candidate": np.asarray([0.2, 0.6]),
        },
    }

    report = hierarchical_bootstrap(aligned, n_bootstrap=200, seed=7)

    expected_s1 = 0.2
    expected_s2 = ((0.2 + 0.6) - (0.0 + 0.5)) / 2
    assert report["observed_paired_difference"] == pytest.approx(
        (expected_s1 + expected_s2) / 2
    )
    assert report["n_bootstrap"] == 200
    assert report["seed"] == 7
    assert len(report["bootstrap_ci95"]) == 2


def test_complementarity_summary_uses_equal_session_oracle_reduction():
    aligned = {
        "large": {
            "reference": np.asarray([-1.0, 0.2, 0.5]),
            "candidate": np.asarray([0.1, 0.4, 0.3]),
        },
        "small": {
            "reference": np.asarray([0.8]),
            "candidate": np.asarray([0.7]),
        },
    }

    report = complementarity_summary(aligned)

    expected_reference = ((0.0 + 0.2 + 0.5) / 3 + 0.8) / 2
    expected_candidate = ((0.1 + 0.4 + 0.3) / 3 + 0.7) / 2
    expected_oracle = ((0.1 + 0.4 + 0.5) / 3 + 0.8) / 2
    assert report["aligned_reference_bps_overall"] == pytest.approx(
        expected_reference
    )
    assert report["aligned_candidate_bps_overall"] == pytest.approx(
        expected_candidate
    )
    assert report["unit_oracle_bps_overall"] == pytest.approx(expected_oracle)
    assert report["candidate_better_units"] == 2
    assert report["reference_better_units"] == 2
    assert report["tied_units"] == 0


def test_per_unit_archive_preserves_session_order_cids_and_values(tmp_path):
    path = tmp_path / "per_unit.npz"
    names = write_per_unit_archive(
        path,
        {"s2": np.asarray([0.4]), "s1": np.asarray([0.1, np.nan])},
        {"s1": [10, 11], "s2": [20]},
        ["s1", "missing", "s2"],
    )

    assert names == ["s1", "s2"]
    with np.load(path, allow_pickle=False) as archive:
        assert archive["session_names"].astype(str).tolist() == ["s1", "s2"]
        assert archive["cids_0"].tolist() == [10, 11]
        assert archive["bps_0"][0] == pytest.approx(0.1)
        assert np.isnan(archive["bps_0"][1])
        assert archive["cids_1"].tolist() == [20]
        assert archive["bps_1"].tolist() == pytest.approx([0.4])
