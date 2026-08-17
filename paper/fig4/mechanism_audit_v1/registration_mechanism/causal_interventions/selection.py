"""Outcome-blind, fold-balanced pilot and complete-subset job definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.pilot_selection import (
    IMAGE_TABLE,
    TRAJECTORY_TABLE,
    select_pilot,
)


Scope = Literal["pilot", "full"]


@dataclass(frozen=True)
class InterventionSelection:
    """Transport and held-out realignment jobs for one scope."""

    scope: Scope
    image_positions: tuple[int, ...]
    trajectory_positions: tuple[int, ...]
    transport_pairs: tuple[tuple[int, int], ...]
    heldout_realign_pairs: tuple[tuple[int, int, int], ...]
    rows: tuple[dict[str, Any], ...]


def frozen_stratified_pilot(
    image_table: Path = IMAGE_TABLE,
    trajectory_table: Path = TRAJECTORY_TABLE,
) -> InterventionSelection:
    """Reuse the already frozen outcome-blind 4-by-12 pilot exactly.

    The transport jobs are the complete Cartesian product of the four image
    and twelve path-length medoids selected before any mechanism outcome was
    observed.  P-specific inference is restricted to the cells in that fixed
    grid for which one crossed-validation fold held out both identities.  The
    frozen grid happens to contain 12 such cells (fold counts 4, 6, 0, 2).
    Fold 2 is therefore absent from the *pilot* realignment gate rather than
    being filled with a newly selected, outcome-dependent example; the gated
    full confirmation contains all four folds.
    """
    images = pd.read_csv(image_table).reset_index(drop=True)
    trajectories = pd.read_csv(trajectory_table).reset_index(drop=True)
    if len(images) != 8 or len(trajectories) != 24:
        raise RuntimeError(
            f"Expected the frozen 8x24 subset tables, found {len(images)}x{len(trajectories)}"
        )
    frozen = select_pilot(image_table=image_table, trajectory_table=trajectory_table)
    selected_images = [int(row["position"]) for row in frozen["images"]]
    selected_trajectories = [int(row["position"]) for row in frozen["trajectories"]]
    transport = tuple(
        (image, trajectory)
        for image in selected_images
        for trajectory in selected_trajectories
    )
    heldout_pairs: list[tuple[int, int, int]] = []
    rows: list[dict[str, Any]] = [
        {
            "scope": "pilot",
            "kind": str(row["kind"]),
            "fold": -1,
            "within_fold_stratum": int(row["stratum"]),
            "position": int(row["position"]),
            "stable_id": int(row["stable_id"]),
            "metric": str(row["metric"]),
            "metric_value": float(row["metric_value"]),
            "selection_rule": str(row["selection_rule"]),
            "outcome_used": False,
        }
        for row in (*frozen["images"], *frozen["trajectories"])
    ]
    for fold_index in range(4):
        test = load_fold(fold_index).test
        test_pairs = set(test.pairs)
        heldout_pairs.extend(
            (fold_index, image, trajectory)
            for image, trajectory in transport
            if (image, trajectory) in test_pairs
        )
    selection = InterventionSelection(
        scope="pilot",
        image_positions=tuple(selected_images),
        trajectory_positions=tuple(selected_trajectories),
        transport_pairs=transport,
        heldout_realign_pairs=tuple(heldout_pairs),
        rows=tuple(rows),
    )
    validate_selection(selection)
    return selection


def complete_selection() -> InterventionSelection:
    """Return the frozen 8x24 design and all 48 crossed held-out P jobs."""
    heldout: list[tuple[int, int, int]] = []
    rows: list[dict[str, Any]] = []
    for fold_index in range(4):
        test = load_fold(fold_index).test
        heldout.extend(
            (fold_index, int(image), int(trajectory))
            for image in test.image_positions
            for trajectory in test.trajectory_positions
        )
    images = tuple(range(8))
    trajectories = tuple(range(24))
    for kind, positions in (("image", images), ("trajectory", trajectories)):
        rows.extend(
            {
                "scope": "full",
                "kind": kind,
                "fold": -1,
                "within_fold_stratum": -1,
                "position": int(position),
                "stable_id": -1,
                "metric": "frozen complete subset",
                "metric_value": np.nan,
                "selection_rule": "all positions in the frozen exact 8-image x 24-trajectory subset",
                "outcome_used": False,
            }
            for position in positions
        )
    selection = InterventionSelection(
        scope="full",
        image_positions=images,
        trajectory_positions=trajectories,
        transport_pairs=tuple((image, trajectory) for image in images for trajectory in trajectories),
        heldout_realign_pairs=tuple(heldout),
        rows=tuple(rows),
    )
    validate_selection(selection)
    return selection


def selection_for_scope(scope: Scope) -> InterventionSelection:
    return frozen_stratified_pilot() if scope == "pilot" else complete_selection()


def validate_selection(selection: InterventionSelection) -> None:
    expected = (4, 12, 48, 12) if selection.scope == "pilot" else (8, 24, 192, 48)
    observed = (
        len(selection.image_positions),
        len(selection.trajectory_positions),
        len(selection.transport_pairs),
        len(selection.heldout_realign_pairs),
    )
    if observed != expected:
        raise RuntimeError(f"{selection.scope} selection shape changed: {observed} != {expected}")
    if len(set(selection.transport_pairs)) != len(selection.transport_pairs):
        raise RuntimeError("Transport selection contains duplicate pairs")
    for fold_index, image, trajectory in selection.heldout_realign_pairs:
        test_pairs = set(load_fold(int(fold_index)).test.pairs)
        if (int(image), int(trajectory)) not in test_pairs:
            raise RuntimeError(
                f"P-specific realignment pair {(image, trajectory)} is not held out by fold {fold_index}"
            )
    if selection.scope == "pilot":
        fold_counts = np.bincount(
            np.asarray([row[0] for row in selection.heldout_realign_pairs], dtype=np.int64),
            minlength=4,
        )
        if not np.array_equal(fold_counts, np.asarray([4, 6, 0, 2])):
            raise RuntimeError(
                "Frozen pilot held-out realignment intersections changed: "
                f"{fold_counts.tolist()} != [4, 6, 0, 2]"
            )
