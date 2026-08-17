from __future__ import annotations

import json

import numpy as np
import pandas as pd

from paper.fig4.mechanism_audit_v1.causal_low_rank.infer_and_report import (
    BOOTSTRAP_METRICS,
    CELL_SUM_FIELDS,
    DECISION_LABELS,
    FoldCells,
    _cross_contrast_evidence,
    crossed_bootstrap,
    crossed_resample_weights,
    select_scientific_rank,
    strict_decision_label,
)


def test_crossed_bootstrap_weight_is_cartesian_not_independent_pairs() -> None:
    weight = crossed_resample_weights(3, 4, np.random.default_rng(91))
    assert weight.shape == (3, 4)
    assert int(weight.sum()) == 12
    # An outer product has matrix rank one (or zero, which cannot occur here).
    assert np.linalg.matrix_rank(weight) == 1
    image_marginal = weight.sum(axis=1)
    trajectory_marginal = weight.sum(axis=0)
    np.testing.assert_array_equal(weight * weight.sum(), np.multiply.outer(image_marginal, trajectory_marginal))


def test_crossed_bootstrap_preserves_paired_intervention_statistics() -> None:
    template = {name: np.ones((2, 6), dtype=np.float64) for name in CELL_SUM_FIELDS}
    template.update(
        {
            "map_effect": np.full((2, 6), 2.0),
            "map_suff_residual": np.full((2, 6), 0.5),
            "map_nec_residual": np.full((2, 6), 0.5),
            "z_effect": np.full((2, 6), 4.0),
            "z_suff_residual": np.full((2, 6), 1.0),
            "z_nec_residual": np.full((2, 6), 1.0),
            "ssi_numerator_a": np.zeros((2, 6)),
            "ssi_numerator_b": np.ones((2, 6)),
            "ssi_numerator_suff": np.full((2, 6), 0.75),
            "ssi_numerator_nec": np.full((2, 6), 0.25),
            "mean_rate_sum_a": np.ones((2, 6)),
            "mean_rate_sum_b": np.full((2, 6), 2.0),
            "mean_rate_sum_suff": np.full((2, 6), 1.75),
            "mean_rate_sum_nec": np.full((2, 6), 1.25),
            "n_unit_records": np.ones((2, 6)),
        }
    )
    fold = FoldCells(0, np.asarray([0, 1]), np.arange(6), template)
    point, samples = crossed_bootstrap([fold], 50, 191)
    expected = np.full(len(BOOTSTRAP_METRICS), 0.75)
    np.testing.assert_allclose([point[name] for name in BOOTSTRAP_METRICS], expected)
    np.testing.assert_allclose(samples, np.broadcast_to(expected, samples.shape))


def test_rank_selection_uses_lowest_compact_validation_qualified_rank() -> None:
    rows = []
    for rank, recovery in ((4, 0.68), (8, 0.82), (16, 0.91)):
        for fold in range(4):
            rows.append({"rank": rank, "fold": fold, "best_validation_loss": 1.0 - recovery})
    result = select_scientific_rank(pd.DataFrame(rows), [4, 8, 16])
    assert result["selected_rank"] == 4
    assert result["selection_used_test_metrics"] is False


def test_rank_selection_falls_back_to_best_preregistered_validation_rank() -> None:
    rows = []
    for rank, recovery in ((4, 0.40), (8, 0.60), (16, 0.72)):
        for fold in range(4):
            rows.append({"rank": rank, "fold": fold, "best_validation_loss": 1.0 - recovery})
    result = select_scientific_rank(pd.DataFrame(rows), [4, 8, 16])
    assert result["selected_rank"] == 16


def test_cross_contrast_uses_source_selected_rank() -> None:
    selections = {
        "low_0_to_2": {"selected_rank": 4},
        "high_0_to_1": {"selected_rank": 8},
        "high_1_to_3": {"selected_rank": 16},
    }
    rows = []
    for source, source_rank in (("low_0_to_2", 4), ("high_0_to_1", 8), ("high_1_to_3", 16)):
        for target in selections:
            if source == target:
                continue
            for fold in range(4):
                rows.append(
                    {
                        "source_contrast": source,
                        "target_contrast": target,
                        "fold": fold,
                        "rank": source_rank,
                        "projector_overlap": 0.1,
                        "map_r2_sufficiency": 0.7,
                        "map_r2_necessity": 0.7,
                        "ssi_fraction_transferred": 0.7,
                        "ssi_fraction_removed": 0.7,
                    }
                )
    result = _cross_contrast_evidence(pd.DataFrame(rows), selections)
    assert result["complete"] is True
    direction = result["directions"]["low_0_to_2->high_0_to_1"]
    assert direction["source_rank_evaluated"] == 4
    assert direction["target_selected_rank"] == 8
    assert direction["geometry_comparable_at_selected_ranks"] is False


def test_bootstrap_summary_json_is_plot_loader_compatible(tmp_path) -> None:
    from paper.fig4.mechanism_audit_v1.causal_low_rank.plot_results import load_bootstrap_summary

    records = [
        {
            "contrast": "low_0_to_2",
            "rank": 4,
            "method": "learned",
            "metric": "map_r2_sufficiency",
            "mean": 0.7,
            "ci_low": 0.6,
            "ci_high": 0.8,
            "n_bootstrap": 4000,
        }
    ]
    path = tmp_path / "bootstrap_results.npz"
    np.savez_compressed(path, summary_json=np.asarray(json.dumps(records)))
    result = load_bootstrap_summary(path)
    assert len(result) == 1
    assert result.iloc[0]["contrast"] == "low_0_to_2"
    assert result.iloc[0]["rank"] == 4


def _evidence(rank: int, value: float, ci: float) -> dict[str, object]:
    return {
        "rank": rank,
        "map_suff": value,
        "map_nec": value,
        "ssi_suff": value,
        "ssi_nec": value,
        "ci_map_suff": ci,
        "ci_map_nec": ci,
        "ci_ssi_suff": ci,
        "ci_ssi_nec": ci,
        "beats_pca": True,
        "beats_readout": True,
        "beats_random95": True,
        "beats_shuffle95": True,
        "stable": True,
        "cross_scale": True,
        "no_train_test_collapse": True,
        "loo_robust": True,
    }


def test_strict_decision_hierarchy_and_incomplete_guard() -> None:
    evidence = {
        "low_0_to_2": _evidence(4, 0.85, 0.65),
        "high_0_to_1": _evidence(4, 0.85, 0.65),
        "high_1_to_3": _evidence(4, 0.85, 0.65),
    }
    assert strict_decision_label(evidence, True) == DECISION_LABELS[0]
    evidence["high_1_to_3"] = _evidence(8, 0.70, 0.55)
    assert strict_decision_label(evidence, True) == DECISION_LABELS[1]
    evidence["high_1_to_3"]["stable"] = False
    assert strict_decision_label(evidence, True) == DECISION_LABELS[3]
    assert strict_decision_label(evidence, False) is None
