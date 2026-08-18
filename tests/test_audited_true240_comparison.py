import json

import numpy as np
import pytest

from paper.model_selection.evaluate_paired_validation_models import (
    common_support_bps,
    independent_bits_per_spike,
    intersect_source_support,
    paired_native_filter,
    score_support_digest,
)
from paper.model_selection.render_audited_true240_comparison import (
    load_paired_validation,
    load_validation,
)


def _write_report(tmp_path, name, *, rate=120, count=3):
    npz = tmp_path / f"{name}.npz"
    np.savez_compressed(
        npz,
        session_names=np.asarray(["session"]),
        cids_0=np.asarray([4, 7]),
        bps_0=np.asarray([0.2, 0.4]) if name == "left" else np.asarray([0.3, 0.35]),
    )
    report = tmp_path / f"{name}.json"
    report.write_text(
        json.dumps(
            {
                "split": "val",
                "score_rate_hz": rate,
                "samples_by_session": {"session": count},
                "per_unit_bps_npz": str(npz),
            }
        ),
        encoding="utf-8",
    )
    return report


def test_validation_comparison_accepts_shared_120hz_support(tmp_path) -> None:
    left = _write_report(tmp_path, "left")
    right = _write_report(tmp_path, "right")

    table, _, _ = load_validation(left, right)

    assert table.shape[0] == 2
    assert table.metric.eq("validation_bps").all()


def test_validation_comparison_rejects_different_bin_rates(tmp_path) -> None:
    left = _write_report(tmp_path, "left")
    right = _write_report(tmp_path, "right", rate=240)

    with pytest.raises(RuntimeError, match="same 120-Hz count grid"):
        load_validation(left, right)


def test_validation_comparison_rejects_different_sample_support(tmp_path) -> None:
    left = _write_report(tmp_path, "left")
    right = _write_report(tmp_path, "right", count=2)

    with pytest.raises(RuntimeError, match="sample supports differ"):
        load_validation(left, right)


def test_source_support_intersection_preserves_both_row_maps() -> None:
    shared, candidate_rows, reference_rows = intersect_source_support(
        np.asarray([1, 3, 4, 8]), np.asarray([0, 1, 4, 6, 8])
    )
    np.testing.assert_array_equal(shared, [1, 4, 8])
    np.testing.assert_array_equal(candidate_rows, [0, 2, 3])
    np.testing.assert_array_equal(reference_rows, [1, 2, 4])


def test_common_support_bps_uses_one_mask_for_both_models() -> None:
    observation = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    candidate = np.asarray([[0.2, 0.9], [1.1, 0.2], [1.8, 1.0]])
    reference = np.asarray([[0.3, 1.1], [0.9, 0.1], [2.1, 0.8]])
    candidate_filter = np.ones_like(observation)
    reference_filter = np.ones_like(observation)
    candidate_filter[0, 0] = 0
    reference_filter[2, 1] = 0

    candidate_bps, reference_bps, counts = common_support_bps(
        candidate,
        reference,
        observation,
        candidate_filter,
        reference_filter,
    )
    np.testing.assert_array_equal(counts, [2, 2])
    assert np.all(np.isfinite(candidate_bps))
    assert np.all(np.isfinite(reference_bps))

    direct = independent_bits_per_spike(
        candidate,
        observation,
        (candidate_filter > 0) & (reference_filter > 0),
    )
    np.testing.assert_allclose(candidate_bps, direct, rtol=0, atol=2e-12)


def test_common_support_bps_records_cross_implementation_error() -> None:
    rng = np.random.default_rng(12)
    observation = rng.poisson(0.2, size=(20_000, 4)).astype(np.float64)
    candidate = rng.lognormal(-1.5, 0.5, size=observation.shape)
    reference = rng.lognormal(-1.4, 0.4, size=observation.shape)
    data_filter = rng.random(observation.shape) > 0.1

    *_, audit = common_support_bps(
        candidate,
        reference,
        observation,
        data_filter,
        data_filter,
        return_audit=True,
    )

    assert audit["candidate_max_abs_error"] <= audit["absolute_tolerance"]
    assert audit["reference_max_abs_error"] <= audit["absolute_tolerance"]


def test_common_support_bps_rejects_missing_model_prediction() -> None:
    observation = np.ones((3, 2))
    candidate = np.ones_like(observation)
    reference = np.ones_like(observation)
    candidate[1, 0] = np.nan
    with pytest.raises(RuntimeError, match="candidate=1, reference=0"):
        common_support_bps(
            candidate,
            reference,
            observation,
            np.ones_like(observation),
            np.ones_like(observation),
        )


def test_score_support_digest_is_prediction_independent_and_mask_sensitive() -> None:
    observation = np.arange(6, dtype=np.float32).reshape(3, 2)
    left = np.ones_like(observation)
    right = np.ones_like(observation)
    first = score_support_digest(observation, left, right)
    right[1, 0] = 0
    second = score_support_digest(observation, left, right)
    assert first[0] != second[0]
    assert first[1] != second[1]


def test_paired_native_filter_rejects_pair_with_one_invalid_bin() -> None:
    import torch

    data_filter = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    paired = paired_native_filter(data_filter, n_pairs=2)
    np.testing.assert_array_equal(paired.numpy(), [[0.0, 1.0], [1.0, 0.0]])


def test_load_paired_validation_uses_audited_archive(tmp_path) -> None:
    archive = tmp_path / "paired.npz"
    np.savez_compressed(
        archive,
        session_names=np.asarray(["session"]),
        cids_0=np.asarray([4, 7]),
        candidate_bps_0=np.asarray([0.4, np.nan]),
        reference_bps_0=np.asarray([0.3, 0.2]),
        common_valid_count_0=np.asarray([12, 0]),
    )
    report = tmp_path / "paired.json"
    report.write_text(
        json.dumps(
            {
                "split": "val",
                "score_rate_hz": 120,
                "per_unit_npz": str(archive),
                "shared_geometry_by_session": {"session": 15},
            }
        ),
        encoding="utf-8",
    )

    table, loaded = load_paired_validation(report)

    assert loaded["score_rate_hz"] == 120
    assert table.shape[0] == 1
    assert table.iloc[0].unit_id == 4
    assert table.iloc[0].ryan == pytest.approx(0.3)
    assert table.iloc[0].candidate == pytest.approx(0.4)
