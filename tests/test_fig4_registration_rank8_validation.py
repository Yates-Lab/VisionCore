from __future__ import annotations

import numpy as np

from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation import (
    common as rank8_common,
)

from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.finalize_rank8_validation import (
    consensus_from_projectors,
    fold_stability_gate,
    generalization_gate,
    random_overlap_distribution,
    shared_high_gate,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.fit_rank8_folds import (
    planned_fits,
)


def _basis(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value, _ = np.linalg.qr(rng.standard_normal((128, 8)), mode="reduced")
    return value.astype(np.float32)


def test_rank8_continuation_plan_cannot_expand_into_another_rank_sweep() -> None:
    plan = planned_fits()
    assert len(plan) == 9
    assert {row["rank"] for row in plan} == {8}
    assert {row["fold"] for row in plan} == {1, 2, 3}
    assert {row["contrast"] for row in plan} == {
        "low_0_to_2",
        "high_0_to_1",
        "high_1_to_3",
    }


def test_consensus_uses_projectors_and_is_invariant_to_basis_rotation() -> None:
    basis = _basis(3)
    rng = np.random.default_rng(4)
    projectors = []
    for _ in range(4):
        rotation, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        rotated = basis @ rotation
        projectors.append(rotated @ rotated.T)
    consensus_basis, consensus_projector, eigenvalues = consensus_from_projectors(projectors)
    np.testing.assert_allclose(consensus_basis.T @ consensus_basis, np.eye(8), atol=1e-6)
    np.testing.assert_allclose(consensus_projector, basis @ basis.T, atol=2e-6)
    np.testing.assert_allclose(consensus_projector @ consensus_projector, consensus_projector, atol=2e-6)
    np.testing.assert_allclose(eigenvalues[:8], np.ones(8), atol=2e-6)


def test_random_overlap_null_matches_rank_over_channel_expectation() -> None:
    values = random_overlap_distribution(1_000, seed=91)
    assert values.shape == (1_000,)
    assert abs(float(values.mean()) - 8 / 128) < 0.004
    assert float(np.percentile(values, 97.5)) > 8 / 128


def test_fold_stability_requires_every_pair_above_random_bound() -> None:
    rows = [
        {
            "comparison_type": "within_contrast_across_folds",
            "contrast_a": "low_0_to_2",
            "projector_overlap": 0.40 + pair / 100,
        }
        for pair in range(6)
    ]
    passing = fold_stability_gate(rows, "low_0_to_2", null_high=0.10)
    assert passing["complete"] is True
    assert passing["clearly_above_random"] is True
    rows[-1]["projector_overlap"] = 0.09
    failing = fold_stability_gate(rows, "low_0_to_2", null_high=0.10)
    assert failing["clearly_above_random"] is False


def test_generalization_gate_uses_both_intervention_directions_and_readout_control() -> None:
    rows = []
    for fold, recovery in enumerate((0.75, 0.72, 0.68, 0.35)):
        rows.append(
            {
                "contrast": "high_0_to_1",
                "method": "learned",
                "fold": fold,
                "map_r2_sufficiency": recovery,
                "map_r2_necessity": recovery - 0.01,
                "learned_minus_readout_complete_map_mean": 0.05,
            }
        )
    gate = generalization_gate(rows, "high_0_to_1")
    assert gate["generalizes"] is True
    assert gate["folds_passing"] == 3
    assert gate["learned_outperforms_readout_consistently"] is True
    rows[2]["map_r2_necessity"] = 0.30
    assert generalization_gate(rows, "high_0_to_1")["generalizes"] is False


def test_shared_high_consensus_gate_requires_overlap_stability_and_bidirectional_transfer() -> None:
    stability = [
        {
            "comparison_type": "higher_sf_between_contrasts_within_fold",
            "projector_overlap": value,
        }
        for value in (0.80, 0.76, 0.73, 0.71)
    ]
    transfer = [
        {"map_r2_sufficiency": 0.70, "map_r2_necessity": 0.65}
        for _ in range(8)
    ]
    per_contrast = {
        "high_0_to_1": {"clearly_above_random": True},
        "high_1_to_3": {"clearly_above_random": True},
    }
    passing = shared_high_gate(stability, transfer, per_contrast, null_high=0.10)
    assert passing["allowed"] is True
    transfer[-1]["map_r2_necessity"] = 0.39
    assert shared_high_gate(stability, transfer, per_contrast, null_high=0.10)["allowed"] is False


def test_overnight_budget_uses_one_authoritative_cumulative_ledger(
    tmp_path, monkeypatch
) -> None:
    ledger = tmp_path / "gpu_budget.json"
    ledger.write_text(
        '{"hard_limit_hours": 4.0, "cache_accelerator_hours": 0.1, '
        '"gpu_stage_wall_hours": 2.9, "total_conservative_gpu_hours": 3.0, "events": []}\n',
        encoding="utf-8",
    )
    out = tmp_path / "registration"
    monkeypatch.setattr(rank8_common, "OUT", out)
    monkeypatch.setattr(rank8_common, "RANK8_OUT", out / "rank8_validation")
    monkeypatch.setattr(rank8_common, "CROSS_TRANSFER_OUT", out / "rank8_validation/cross_transfer")
    monkeypatch.setattr(rank8_common, "SOURCE_GPU_BUDGET", ledger)
    monkeypatch.setattr(rank8_common, "GPU_BUDGET", ledger)

    before = rank8_common.load_budget()
    assert before["total_conservative_gpu_hours"] == 3.0
    assert before["hard_limit_hours"] == 10.0
    observed = rank8_common.record_gpu_time("test", 3600.0, {"completed": True})
    assert observed["gpu_stage_wall_hours"] == 3.9
    assert observed["total_conservative_gpu_hours"] == 4.0
    assert rank8_common.load_budget()["total_conservative_gpu_hours"] == 4.0
