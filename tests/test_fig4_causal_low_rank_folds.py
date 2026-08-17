from __future__ import annotations

from itertools import product

from paper.fig4.mechanism_audit_v1.causal_low_rank.prepare_analysis import (
    ANALYSIS_SEED,
    build_fold_assignments,
)


def _pairs(split: dict) -> set[tuple[int, int]]:
    if "pair_positions" in split:
        return {tuple(value) for value in split["pair_positions"]}
    return set(product(split["image_positions"], split["trajectory_positions"]))


def test_fixed_image_folds_use_sorted_stable_identifiers() -> None:
    payload = build_fold_assignments()
    observed = [fold["image_ids"] for fold in payload["image_folds"]]
    assert observed == [[13, 15], [27, 29], [34, 43], [56, 90]]
    assert sorted(value for fold in observed for value in fold) == [13, 15, 27, 29, 34, 43, 56, 90]


def test_trajectory_folds_are_seeded_stratified_and_exhaustive() -> None:
    first = build_fold_assignments(seed=ANALYSIS_SEED)
    second = build_fold_assignments(seed=ANALYSIS_SEED)
    assert first == second

    folds = first["trajectory_folds"]
    assert all(len(fold["trajectory_positions"]) == 6 for fold in folds)
    assert all(sorted(fold["path_strata"]) == list(range(6)) for fold in folds)
    positions = [position for fold in folds for position in fold["trajectory_positions"]]
    assert sorted(positions) == list(range(24))
    assert len(set(positions)) == 24


def test_outer_crossed_folds_have_no_identity_leakage_and_cover_all_test_ids() -> None:
    payload = build_fold_assignments()
    seen_test_images: list[int] = []
    seen_test_trajectories: list[int] = []
    for fold in payload["folds"]:
        outer = fold["outer_train"]
        test = fold["test"]
        assert len(test["image_positions"]) == 2
        assert len(test["trajectory_positions"]) == 6
        assert test["n_image_trajectory_pairs"] == 12
        assert len(outer["image_positions"]) == 6
        assert len(outer["trajectory_positions"]) == 18
        assert outer["n_image_trajectory_pairs"] == 108
        assert set(test["image_positions"]).isdisjoint(outer["image_positions"])
        assert set(test["trajectory_positions"]).isdisjoint(outer["trajectory_positions"])
        seen_test_images.extend(test["image_positions"])
        seen_test_trajectories.extend(test["trajectory_positions"])

    assert sorted(seen_test_images) == list(range(8))
    assert sorted(seen_test_trajectories) == list(range(24))


def test_inner_validation_is_fixed_inside_outer_train_and_pair_disjoint() -> None:
    payload = build_fold_assignments()
    for fold in payload["folds"]:
        outer = fold["outer_train"]
        train = fold["train"]
        validation = fold["validation"]
        test = fold["test"]

        assert len(validation["image_positions"]) == 6
        assert len(validation["trajectory_positions"]) == 6
        assert validation["n_image_trajectory_pairs"] == 6
        assert set(validation["image_positions"]) <= set(outer["image_positions"])
        assert set(validation["trajectory_positions"]) <= set(outer["trajectory_positions"])
        assert set(train["image_positions"]) == set(outer["image_positions"])
        assert set(train["trajectory_positions"]) == set(outer["trajectory_positions"])
        assert _pairs(train).isdisjoint(_pairs(validation))
        assert _pairs(train).isdisjoint(_pairs(test))
        assert _pairs(validation).isdisjoint(_pairs(test))
        assert _pairs(train) | _pairs(validation) == _pairs(outer)
        assert train["n_image_trajectory_pairs"] == 102
        assert fold["unused_outer_train_cross_pairs"] == 0
