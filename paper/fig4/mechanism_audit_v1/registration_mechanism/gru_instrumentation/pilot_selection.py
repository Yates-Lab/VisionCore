"""Outcome-blind 4-image x 12-trajectory pilot selection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
SELECTION_DIR = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1/selection"
)
IMAGE_TABLE = SELECTION_DIR / "selected_images.csv"
TRAJECTORY_TABLE = SELECTION_DIR / "selected_traces.csv"
IMAGE_STRATIFICATION_METRIC = "image_oriented_8plus_power_proxy"
TRAJECTORY_STRATIFICATION_METRIC = "rendered_path_length_arcmin"


def stratified_medoids(
    values: Sequence[float],
    stable_ids: Sequence[int],
    *,
    n_strata: int,
) -> list[dict[str, float | int]]:
    """Select the closest observation to each equal-count stratum median."""
    value = np.asarray(values, dtype=np.float64)
    identifiers = np.asarray(stable_ids, dtype=np.int64)
    if value.ndim != 1 or identifiers.shape != value.shape:
        raise ValueError("values and stable_ids must be matching vectors")
    if not np.isfinite(value).all():
        raise ValueError("Stratification values contain nonfinite entries")
    if len(value) % int(n_strata) != 0:
        raise ValueError("Equal-count strata require sample count divisible by n_strata")
    order = np.lexsort((identifiers, value))
    chunks = np.split(order, int(n_strata))
    output: list[dict[str, float | int]] = []
    for stratum, positions in enumerate(chunks):
        target = float(np.median(value[positions]))
        # lexsort makes argmin ties deterministic by metric then stable id.
        chosen = int(positions[int(np.argmin(np.abs(value[positions] - target)))])
        output.append(
            {
                "stratum": int(stratum),
                "position": chosen,
                "stable_id": int(identifiers[chosen]),
                "metric_value": float(value[chosen]),
                "stratum_target_median": target,
                "stratum_min": float(value[positions].min()),
                "stratum_max": float(value[positions].max()),
            }
        )
    return output


def select_pilot(
    image_table: Path = IMAGE_TABLE,
    trajectory_table: Path = TRAJECTORY_TABLE,
) -> dict[str, list[dict[str, float | int | str]]]:
    images = pd.read_csv(image_table)
    trajectories = pd.read_csv(trajectory_table)
    image_rows = stratified_medoids(
        images[IMAGE_STRATIFICATION_METRIC], images["image_index"], n_strata=4
    )
    trajectory_rows = stratified_medoids(
        trajectories[TRAJECTORY_STRATIFICATION_METRIC],
        trajectories["trace_bank_index"],
        n_strata=12,
    )
    for row in image_rows:
        row["kind"] = "image"
        row["metric"] = IMAGE_STRATIFICATION_METRIC
        row["selection_rule"] = "closest to median within four equal-count metric strata; stable-id tie break"
    for row in trajectory_rows:
        row["kind"] = "trajectory"
        row["metric"] = TRAJECTORY_STRATIFICATION_METRIC
        row["selection_rule"] = "closest to median within twelve equal-count path-length strata; stable-id tie break"
    return {"images": image_rows, "trajectories": trajectory_rows}


def write_pilot_selection(path: Path, selection: dict[str, list[dict[str, object]]]) -> None:
    rows = [*selection["images"], *selection["trajectories"]]
    if len(selection["images"]) != 4 or len(selection["trajectories"]) != 12:
        raise RuntimeError("Pilot selection is not exactly 4 images x 12 trajectories")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
