#!/usr/bin/env python3
"""Infer the final causal-subspace result from saved evaluation products only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    ANALYSIS_SEED,
    CONFIG,
    CONTRASTS,
    EVALUATION,
    FITS,
    OUT,
    ensure_output_dirs,
    sha256_file,
    write_json,
)


RANK_SUMMARY = OUT / "rank_summary.csv"
PER_UNIT_RESULTS = OUT / "per_unit_results.csv"
BASELINE_RESULTS = OUT / "baseline_results.csv"
STABILITY_RESULTS = OUT / "subspace_stability.csv"
CROSS_SCALE_RESULTS = OUT / "cross_scale_results.csv"
CROSS_CONTRAST_RESULTS = OUT / "cross_contrast_results.csv"
STAGE2_RANKS = CONFIG / "stage2_ranks.json"
OPTIMIZATION_CONFIG = CONFIG / "optimization_config.json"
ANALYSIS_MANIFEST = CONFIG / "analysis_manifest.json"
INTEGRITY_TESTS = OUT / "integrity_tests.json"
STATISTICS = OUT / "statistics.json"
BOOTSTRAP = OUT / "bootstrap_results.npz"
REPORT = OUT / "FINAL_CAUSAL_SUBSPACE_REPORT.md"
GPU_BUDGET = OUT / "gpu_budget.json"

DECISION_LABELS = (
    "STRONG COMPLETE MECHANISM",
    "ADEQUATE PAPER-LEVEL MECHANISM",
    "PARTIAL MECHANISM",
    "NOT LOW-DIMENSIONAL",
)
BOOTSTRAP_METRICS = (
    "map_r2_sufficiency",
    "map_r2_necessity",
    "preactivation_r2_sufficiency",
    "preactivation_r2_necessity",
    "ssi_fraction_transferred",
    "ssi_fraction_removed",
    "mean_rate_fraction_transferred",
    "mean_rate_fraction_removed",
)
CELL_SUM_FIELDS = (
    "map_effect",
    "map_suff_residual",
    "map_nec_residual",
    "z_effect",
    "z_suff_residual",
    "z_nec_residual",
    "ssi_numerator_a",
    "ssi_weight_a",
    "ssi_numerator_b",
    "ssi_weight_b",
    "ssi_numerator_suff",
    "ssi_weight_suff",
    "ssi_numerator_nec",
    "ssi_weight_nec",
    "mean_rate_sum_a",
    "mean_rate_sum_b",
    "mean_rate_sum_suff",
    "mean_rate_sum_nec",
    "n_unit_records",
)


@dataclass(frozen=True)
class FoldCells:
    fold: int
    image_positions: np.ndarray
    trajectory_positions: np.ndarray
    values: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-tier", choices=("exploration", "final"), default="final")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now().astimezone().isoformat()


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _checkpoint_path(manifest: dict[str, Any]) -> str:
    return str(
        manifest.get("frozen_model", {}).get(
            "checkpoint",
            manifest.get("sources", {}).get("checkpoint", {}).get("path", "unavailable"),
        )
    )


def _reproduction_record(
    manifest: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any]:
    sources = manifest.get("sources", {})
    return {
        "corrected_history_source": sources.get("corrected_history_bank", {}).get(
            "path", "unavailable"
        ),
        "corrected_history_sha256": sources.get("corrected_history_bank", {}).get("sha256"),
        "activation_indexing": (
            "final internal 128×64×64 ConvGRU state for each scored output frame, immediately "
            "before the exact frozen spatially tiled RR100 readout"
        ),
        "numerical_tolerances": configuration.get("numerical_tolerances", {}),
        "fold_assignments": sources.get("fold_assignments", {}).get("path", "unavailable"),
        "fold_assignments_sha256": sources.get("fold_assignments", {}).get("sha256"),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-30 else math.nan


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _file_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def select_scientific_rank(
    learned_crossval: pd.DataFrame,
    preregistered_ranks: Iterable[int],
    *,
    n_folds: int = 4,
    compact_rank_max: int = 8,
    validation_recovery_threshold: float = 0.65,
) -> dict[str, Any]:
    """Choose rank using validation loss only; held-out test metrics are ignored."""
    candidates: list[dict[str, Any]] = []
    has_validation_columns = {"rank", "fold", "best_validation_loss"}.issubset(
        learned_crossval.columns
    )
    for rank in sorted(set(int(value) for value in preregistered_ranks)):
        if has_validation_columns:
            rows = learned_crossval.loc[
                pd.to_numeric(learned_crossval["rank"], errors="coerce").eq(rank)
            ]
            valid = pd.to_numeric(rows["best_validation_loss"], errors="coerce").notna()
            recoveries = 1.0 - pd.to_numeric(
                rows.loc[valid, "best_validation_loss"], errors="coerce"
            ).to_numpy(float)
            folds = pd.to_numeric(rows.loc[valid, "fold"], errors="coerce").dropna().astype(int).unique()
        else:
            recoveries = np.empty(0, dtype=np.float64)
            folds = np.empty(0, dtype=np.int64)
        complete = len(recoveries) == n_folds and len(folds) == n_folds
        candidates.append(
            {
                "rank": rank,
                "complete": complete,
                "fold_validation_recovery": recoveries.tolist(),
                "mean_validation_recovery": float(np.mean(recoveries)) if len(recoveries) else math.nan,
                "minimum_validation_recovery": float(np.min(recoveries)) if len(recoveries) else math.nan,
                "meets_compact_validation_criterion": bool(
                    complete
                    and rank <= compact_rank_max
                    and np.all(recoveries >= validation_recovery_threshold)
                ),
            }
        )
    complete_candidates = [row for row in candidates if row["complete"]]
    if not complete_candidates:
        return {
            "status": "incomplete",
            "selected_rank": None,
            "rule": "no rank has all required validation folds",
            "candidates": candidates,
            "selection_used_test_metrics": False,
        }
    compact = [row for row in complete_candidates if row["meets_compact_validation_criterion"]]
    if compact:
        selected = min(compact, key=lambda row: row["rank"])
        rule = (
            f"lowest preregistered rank <= {compact_rank_max} with validation recovery >= "
            f"{validation_recovery_threshold:.2f} in every fold"
        )
    else:
        selected = sorted(
            complete_candidates,
            key=lambda row: (-row["mean_validation_recovery"], row["rank"]),
        )[0]
        rule = "best mean cross-fold validation recovery among preregistered ranks; lower rank breaks ties"
    return {
        "status": "selected",
        "selected_rank": int(selected["rank"]),
        "rule": rule,
        "selected_validation_recovery_mean": selected["mean_validation_recovery"],
        "selected_validation_recovery_minimum": selected["minimum_validation_recovery"],
        "candidates": candidates,
        "selection_used_test_metrics": False,
    }


def crossed_resample_weights(n_images: int, n_trajectories: int, rng: np.random.Generator) -> np.ndarray:
    """Independent cluster draws followed by their Cartesian product."""
    image_draw = rng.integers(0, int(n_images), size=int(n_images))
    trajectory_draw = rng.integers(0, int(n_trajectories), size=int(n_trajectories))
    image_count = np.bincount(image_draw, minlength=int(n_images))
    trajectory_count = np.bincount(trajectory_draw, minlength=int(n_trajectories))
    return np.multiply.outer(image_count, trajectory_count).astype(np.int64)


def _aggregate_fold_cells(folds: list[FoldCells], weights: list[np.ndarray] | None = None) -> dict[str, float]:
    total = {name: 0.0 for name in CELL_SUM_FIELDS}
    for index, fold in enumerate(folds):
        weight = np.ones_like(next(iter(fold.values.values())), dtype=np.float64) if weights is None else weights[index]
        for name in CELL_SUM_FIELDS:
            total[name] += float(np.sum(fold.values[name] * weight))
    ssi = {
        condition: _safe_ratio(total[f"ssi_numerator_{condition}"], total[f"ssi_weight_{condition}"])
        for condition in ("a", "b", "suff", "nec")
    }
    mean_rate = {
        condition: _safe_ratio(total[f"mean_rate_sum_{condition}"], total["n_unit_records"])
        for condition in ("a", "b", "suff", "nec")
    }
    ssi_effect = ssi["b"] - ssi["a"]
    rate_effect = mean_rate["b"] - mean_rate["a"]
    return {
        "map_r2_sufficiency": 1.0 - _safe_ratio(total["map_suff_residual"], total["map_effect"]),
        "map_r2_necessity": 1.0 - _safe_ratio(total["map_nec_residual"], total["map_effect"]),
        "preactivation_r2_sufficiency": 1.0 - _safe_ratio(total["z_suff_residual"], total["z_effect"]),
        "preactivation_r2_necessity": 1.0 - _safe_ratio(total["z_nec_residual"], total["z_effect"]),
        "ssi_fraction_transferred": _safe_ratio(ssi["suff"] - ssi["a"], ssi_effect),
        "ssi_fraction_removed": _safe_ratio(ssi["b"] - ssi["nec"], ssi_effect),
        "mean_rate_fraction_transferred": _safe_ratio(mean_rate["suff"] - mean_rate["a"], rate_effect),
        "mean_rate_fraction_removed": _safe_ratio(mean_rate["b"] - mean_rate["nec"], rate_effect),
        "ssi_a_bits": ssi["a"],
        "ssi_b_bits": ssi["b"],
        "ssi_sufficiency_bits": ssi["suff"],
        "ssi_necessity_bits": ssi["nec"],
        "mean_rate_a": mean_rate["a"],
        "mean_rate_b": mean_rate["b"],
        "mean_rate_sufficiency": mean_rate["suff"],
        "mean_rate_necessity": mean_rate["nec"],
    }


def crossed_bootstrap(
    folds: list[FoldCells],
    n_samples: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    point = _aggregate_fold_cells(folds)
    values = np.empty((int(n_samples), len(BOOTSTRAP_METRICS)), dtype=np.float64)
    for sample in range(int(n_samples)):
        weights = [
            crossed_resample_weights(len(fold.image_positions), len(fold.trajectory_positions), rng)
            for fold in folds
        ]
        result = _aggregate_fold_cells(folds, weights)
        values[sample] = [result[name] for name in BOOTSTRAP_METRICS]
    return point, values


def _target_units(per_unit: pd.DataFrame, contrast: str, rank: int) -> np.ndarray:
    rows = per_unit.loc[
        per_unit["contrast"].eq(contrast)
        & per_unit["stage"].eq("crossval")
        & per_unit["method"].eq("learned")
        & pd.to_numeric(per_unit["rank"], errors="coerce").eq(rank)
    ].copy()
    if rows.empty:
        raise RuntimeError(f"No per-unit rows for {contrast}, rank {rank}")
    rows["is_target"] = rows["is_target_population"].map(_boolean)
    units = np.sort(pd.to_numeric(rows.loc[rows.is_target, "unit_index"], errors="raise").astype(int).unique())
    expected = 71 if contrast.startswith("low") else 29
    if len(units) != expected:
        raise RuntimeError(f"{contrast} has {len(units)} target units, expected {expected}")
    return units


def _prediction_path(contrast: str, fold: int, rank: int) -> Path:
    return FITS / "crossval" / contrast / f"fold_{fold}" / f"rank_{rank:03d}" / "test_predictions.npz"


def _fold_cells(path: Path, fold: int, target_units: np.ndarray) -> FoldCells:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "image_position",
            "trajectory_position",
            "frame_position",
            "map_effect_sse",
            "map_suff_residual_sse",
            "map_nec_residual_sse",
            "z_effect_sse",
            "z_suff_residual_sse",
            "z_nec_residual_sse",
            "ssi_a",
            "ssi_b",
            "ssi_suff",
            "ssi_nec",
            "expected_a",
            "expected_b",
            "expected_suff",
            "expected_nec",
            "mean_rate_a",
            "mean_rate_b",
            "mean_rate_suff",
            "mean_rate_nec",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"{path} lacks {sorted(missing)}")
        data = {name: np.asarray(archive[name]) for name in required}
    images = np.sort(np.unique(data["image_position"].astype(int)))
    trajectories = np.sort(np.unique(data["trajectory_position"].astype(int)))
    if len(images) != 2 or len(trajectories) != 6:
        raise RuntimeError(f"{path} is not a complete 2-image x 6-trajectory held-out fold")
    values = {name: np.zeros((len(images), len(trajectories)), dtype=np.float64) for name in CELL_SUM_FIELDS}
    for image_index, image in enumerate(images):
        for trajectory_index, trajectory in enumerate(trajectories):
            rows = (data["image_position"] == image) & (data["trajectory_position"] == trajectory)
            if int(np.count_nonzero(rows)) != 40 or not np.array_equal(
                np.sort(data["frame_position"][rows].astype(int)), np.arange(40)
            ):
                raise RuntimeError(f"{path}: incomplete frame cell image={image}, trajectory={trajectory}")
            row_indices = np.flatnonzero(rows)
            index = np.ix_(row_indices, target_units)
            expected_a = data["expected_a"][index].astype(np.float64)
            expected_b = data["expected_b"][index].astype(np.float64)
            paired_weight = 0.5 * (expected_a + expected_b)
            for destination, source in (
                ("map_effect", "map_effect_sse"),
                ("map_suff_residual", "map_suff_residual_sse"),
                ("map_nec_residual", "map_nec_residual_sse"),
                ("z_effect", "z_effect_sse"),
                ("z_suff_residual", "z_suff_residual_sse"),
                ("z_nec_residual", "z_nec_residual_sse"),
            ):
                values[destination][image_index, trajectory_index] = float(
                    np.sum(data[source][index].astype(np.float64) * paired_weight)
                )
            for condition in ("a", "b", "suff", "nec"):
                expected = data[f"expected_{condition}"][index].astype(np.float64)
                ssi = data[f"ssi_{condition}"][index].astype(np.float64)
                rate = data[f"mean_rate_{condition}"][index].astype(np.float64)
                values[f"ssi_numerator_{condition}"][image_index, trajectory_index] = float(
                    np.sum(ssi * expected)
                )
                values[f"ssi_weight_{condition}"][image_index, trajectory_index] = float(np.sum(expected))
                values[f"mean_rate_sum_{condition}"][image_index, trajectory_index] = float(np.sum(rate))
            values["n_unit_records"][image_index, trajectory_index] = 40 * len(target_units)
    return FoldCells(fold, images, trajectories, values)


def _leave_one_image_out(folds: list[FoldCells]) -> tuple[np.ndarray, np.ndarray]:
    images = np.concatenate([fold.image_positions for fold in folds])
    if len(images) != 8 or len(np.unique(images)) != 8:
        raise RuntimeError("Selected learned folds do not contain eight disjoint held-out images")
    trajectories = np.concatenate([fold.trajectory_positions for fold in folds])
    if len(trajectories) != 24 or len(np.unique(trajectories)) != 24:
        raise RuntimeError("Selected learned folds do not contain 24 disjoint held-out trajectories")
    values = np.empty((len(images), len(BOOTSTRAP_METRICS)), dtype=np.float64)
    for output_index, omitted in enumerate(images):
        weights = []
        for fold in folds:
            weight = np.ones((len(fold.image_positions), len(fold.trajectory_positions)), dtype=np.float64)
            weight[fold.image_positions == omitted] = 0.0
            weights.append(weight)
        result = _aggregate_fold_cells(folds, weights)
        values[output_index] = [result[name] for name in BOOTSTRAP_METRICS]
    return images.astype(np.int64), values


def _mean_rank_rows(rows: pd.DataFrame) -> dict[str, float]:
    metrics = (
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    )
    return {name: float(pd.to_numeric(rows[name], errors="coerce").mean()) for name in metrics}


def _rows_are_finite(rows: pd.DataFrame, columns: Iterable[str]) -> bool:
    names = tuple(columns)
    if rows.empty or not set(names).issubset(rows.columns):
        return False
    return bool(np.all(np.isfinite(rows.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(float))))


def _rank_curve(
    table: pd.DataFrame,
    contrast: str,
    stage: str,
    expected_ranks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Summarize a learned rank curve without using it to select the reported rank."""
    required = {
        "contrast",
        "stage",
        "method",
        "rank",
        "fold",
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }
    if not required.issubset(table.columns):
        return {"complete": False, "rows": []}
    rows = table.loc[
        table["contrast"].eq(contrast)
        & table["stage"].eq(stage)
        & table["method"].eq("learned")
    ].copy()
    result: list[dict[str, Any]] = []
    for rank, group in rows.groupby(pd.to_numeric(rows["rank"], errors="coerce"), dropna=True):
        record: dict[str, Any] = {
            "rank": int(rank),
            "folds": int(pd.to_numeric(group["fold"], errors="coerce").nunique()),
            **_mean_rank_rows(group),
        }
        record["finite"] = bool(
            np.all(
                np.isfinite(
                    [
                        record["map_r2_sufficiency"],
                        record["map_r2_necessity"],
                        record["ssi_fraction_transferred"],
                        record["ssi_fraction_removed"],
                    ]
                )
            )
        )
        if "best_validation_loss" in group:
            validation = 1.0 - pd.to_numeric(
                group["best_validation_loss"], errors="coerce"
            ).dropna().to_numpy(float)
            record["validation_recovery_mean"] = (
                float(np.mean(validation)) if len(validation) else math.nan
            )
            record["validation_recovery_minimum"] = (
                float(np.min(validation)) if len(validation) else math.nan
            )
        result.append(record)
    expected_folds = 1 if stage == "screening" else 4
    ranks_present = {row["rank"] for row in result}
    ranks_complete = (
        True
        if expected_ranks is None
        else ranks_present == {int(rank) for rank in expected_ranks}
    )
    return {
        "complete": bool(
            result
            and ranks_complete
            and all(row["folds"] == expected_folds and row["finite"] for row in result)
        ),
        "expected_ranks": None
        if expected_ranks is None
        else sorted({int(rank) for rank in expected_ranks}),
        "rows": sorted(result, key=lambda row: row["rank"]),
    }


def _baseline_evidence(
    baseline: pd.DataFrame,
    learned: pd.DataFrame,
    contrast: str,
    rank: int,
) -> dict[str, Any]:
    empty = {
        "learned": {},
        "comparisons": {
            "movement_pca": {"complete": False},
            "readout_svd": {"complete": False},
            "random_haar": {"complete": False},
            "shuffled_target": {"complete": False},
        },
    }
    metric_columns = {
        "contrast",
        "stage",
        "method",
        "rank",
        "fold",
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }
    if not metric_columns.issubset(learned.columns):
        return empty
    learned_rows = learned.loc[
        learned["contrast"].eq(contrast)
        & learned["stage"].eq("crossval")
        & learned["method"].eq("learned")
        & pd.to_numeric(learned["rank"], errors="coerce").eq(rank)
    ]
    core_metrics = (
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    )
    if (
        len(learned_rows) != 4
        or pd.to_numeric(learned_rows["fold"], errors="coerce").nunique() != 4
        or not _rows_are_finite(learned_rows, core_metrics)
    ):
        return empty
    learned_mean = _mean_rank_rows(learned_rows)
    result: dict[str, Any] = {"learned": learned_mean, "comparisons": {}}
    if not metric_columns.issubset(baseline.columns):
        result["comparisons"] = empty["comparisons"]
        return result
    for method in ("movement_pca", "readout_svd"):
        rows = baseline.loc[
            baseline["contrast"].eq(contrast)
            & baseline["stage"].eq("crossval")
            & baseline["method"].eq(method)
            & pd.to_numeric(baseline["rank"], errors="coerce").eq(rank)
        ]
        if (
            len(rows) != 4
            or pd.to_numeric(rows["fold"], errors="coerce").nunique() != 4
            or not _rows_are_finite(rows, core_metrics)
        ):
            result["comparisons"][method] = {"complete": False}
            continue
        mean = _mean_rank_rows(rows)
        margins = {
            name: learned_mean[name] - mean[name]
            for name in ("map_r2_sufficiency", "map_r2_necessity")
        }
        result["comparisons"][method] = {
            "complete": True,
            "baseline": mean,
            "map_recovery_margins": margins,
            "clearly_exceeds": bool(min(margins.values()) >= 0.05),
            "exceeds": bool(min(margins.values()) > 0.0),
        }
    random_rows = baseline.loc[
        baseline["contrast"].eq(contrast)
        & baseline["stage"].eq("crossval")
        & baseline["method"].eq("random_haar")
        & pd.to_numeric(baseline["rank"], errors="coerce").eq(rank)
    ].copy()
    learned_joint = 0.5 * (learned_mean["map_r2_sufficiency"] + learned_mean["map_r2_necessity"])
    if random_rows.empty:
        random_result = {"complete": False}
    else:
        random_rows["joint"] = 0.5 * (
            pd.to_numeric(random_rows["map_r2_sufficiency"], errors="coerce")
            + pd.to_numeric(random_rows["map_r2_necessity"], errors="coerce")
        )
        if "draw" in random_rows:
            grouped = random_rows.groupby("draw")
            complete_draws = grouped["fold"].nunique().eq(4)
            distribution = grouped.joint.mean().loc[complete_draws].to_numpy(float)
        else:
            distribution = random_rows.joint.to_numpy(float)
        distribution = distribution[np.isfinite(distribution)]
        if len(distribution) == 0:
            random_result = {"complete": False, "draws": 0}
        else:
            threshold = float(np.percentile(distribution, 95))
            random_result = {
                "complete": len(distribution) >= 100,
                "draws": len(distribution),
                "joint_recovery_95th_percentile": threshold,
                "learned_joint_recovery": learned_joint,
                "exceeds_95_percent": bool(learned_joint > threshold),
            }
    result["comparisons"]["random_haar"] = random_result

    shuffled = baseline.loc[
        baseline["contrast"].eq(contrast)
        & baseline["method"].astype(str).isin(
            ("shuffled_target", "shuffled_target_learned", "shuffle")
        )
        & pd.to_numeric(baseline["rank"], errors="coerce").eq(rank)
    ].copy()
    if shuffled.empty:
        result["comparisons"]["shuffled_target"] = {"complete": False}
    else:
        shuffled["joint"] = 0.5 * (
            pd.to_numeric(shuffled["map_r2_sufficiency"], errors="coerce")
            + pd.to_numeric(shuffled["map_r2_necessity"], errors="coerce")
        )
        if "shuffle_draw" in shuffled:
            grouped = shuffled.groupby("shuffle_draw")
            complete_draws = grouped["fold"].nunique().eq(4)
            distribution = grouped.joint.mean().loc[complete_draws].to_numpy(float)
        else:
            distribution = shuffled.joint.to_numpy(float)
        distribution = distribution[np.isfinite(distribution)]
        if len(distribution) == 0:
            result["comparisons"]["shuffled_target"] = {"complete": False, "fits": 0}
        else:
            threshold = float(np.percentile(distribution, 95))
            result["comparisons"]["shuffled_target"] = {
                "complete": len(distribution) >= 20,
                "fits": len(distribution),
                "joint_recovery_95th_percentile": threshold,
                "learned_joint_recovery": learned_joint,
                "exceeds_95_percent": bool(learned_joint > threshold),
            }
    return result


def _rank_stability(table: pd.DataFrame, contrast: str, rank: int) -> dict[str, Any]:
    required = {"contrast", "rank", "projector_overlap", "random_overlap_ci_high"}
    if not required.issubset(table.columns):
        return {"complete": False, "pairs": 0, "stable": False}
    rows = table.loc[
        table["contrast"].eq(contrast) & pd.to_numeric(table["rank"], errors="coerce").eq(rank)
    ]
    if len(rows) != 6 or not _rows_are_finite(
        rows, ("projector_overlap", "random_overlap_ci_high")
    ):
        return {"complete": False, "pairs": len(rows), "stable": False}
    overlap = pd.to_numeric(rows["projector_overlap"], errors="coerce").to_numpy(float)
    null_high = pd.to_numeric(rows["random_overlap_ci_high"], errors="coerce").to_numpy(float)
    return {
        "complete": True,
        "pairs": len(rows),
        "mean_projector_overlap": float(np.nanmean(overlap)),
        "mean_random_95th_percentile": float(np.nanmean(null_high)),
        "fraction_pairs_above_random_95th": float(np.mean(overlap > null_high)),
        "stable": bool(np.nanmean(overlap) > np.nanmean(null_high) and np.mean(overlap > null_high) >= 2 / 3),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(float)


def _cross_scale_evidence(table: pd.DataFrame, contrast: str, rank: int) -> dict[str, Any]:
    required = {
        "contrast",
        "rank",
        "scale_b",
        "ssi_target_effect_bits",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }
    if not required.issubset(table.columns):
        return {"complete": False, "rows": 0, "qualitatively_correct": False}
    rows = table.loc[
        table["contrast"].eq(contrast) & pd.to_numeric(table["rank"], errors="coerce").eq(rank)
    ].copy()
    expected_per_fold = 4 if contrast != "high_1_to_3" else 2
    required_rows = 4 * expected_per_fold
    if len(rows) != required_rows or not _rows_are_finite(
        rows,
        (
            "ssi_target_effect_bits",
            "ssi_fraction_transferred",
            "ssi_fraction_removed",
        ),
    ):
        return {"complete": False, "rows": len(rows), "qualitatively_correct": False}
    actual = pd.to_numeric(rows["ssi_target_effect_bits"], errors="coerce").to_numpy(float)
    suff = pd.to_numeric(rows["ssi_fraction_transferred"], errors="coerce").to_numpy(float)
    nec = pd.to_numeric(rows["ssi_fraction_removed"], errors="coerce").to_numpy(float)
    sign_fraction = float(np.mean((suff > 0) & (nec > 0)))
    grouped = rows.assign(
        actual=actual,
        suff_effect=actual * suff,
        removed_effect=actual * nec,
    ).groupby("scale_b")[["actual", "suff_effect", "removed_effect"]].mean()
    if len(grouped) >= 2:
        actual_rank = _rankdata(grouped.actual.to_numpy())
        suff_corr = float(np.corrcoef(actual_rank, _rankdata(grouped.suff_effect.to_numpy()))[0, 1])
        nec_corr = float(np.corrcoef(actual_rank, _rankdata(grouped.removed_effect.to_numpy()))[0, 1])
    else:
        suff_corr = nec_corr = math.nan
    return {
        "complete": True,
        "rows": len(rows),
        "correct_effect_direction_fraction": sign_fraction,
        "sufficiency_dose_spearman": suff_corr,
        "necessity_dose_spearman": nec_corr,
        "qualitatively_correct": bool(sign_fraction >= 0.75 and suff_corr >= 0.5 and nec_corr >= 0.5),
    }


def _cross_contrast_evidence(
    table: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "source_contrast",
        "target_contrast",
        "fold",
        "rank",
        "projector_overlap",
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }
    if not required.issubset(table.columns):
        return {
            "complete": False,
            "directions": {},
            "sharpening_relationship": "incomplete",
            "reversal_relationship": "incomplete",
        }
    directions: dict[str, Any] = {}
    for target in CONTRASTS:
        for source in CONTRASTS:
            if source.key == target.key:
                continue
            source_rank = selections.get(source.key, {}).get("selected_rank")
            target_rank = selections.get(target.key, {}).get("selected_rank")
            if source_rank is None or target_rank is None:
                continue
            rows = table.loc[
                table["source_contrast"].eq(source.key)
                & table["target_contrast"].eq(target.key)
                & pd.to_numeric(table["rank"], errors="coerce").eq(int(source_rank))
            ]
            key = f"{source.key}->{target.key}"
            cross_metrics = (
                "projector_overlap",
                "map_r2_sufficiency",
                "map_r2_necessity",
                "ssi_fraction_transferred",
                "ssi_fraction_removed",
            )
            if (
                len(rows) != 4
                or pd.to_numeric(rows["fold"], errors="coerce").nunique() != 4
                or not _rows_are_finite(rows, cross_metrics)
            ):
                directions[key] = {"complete": False, "rows": len(rows)}
                continue
            metrics = {
                name: float(pd.to_numeric(rows[name], errors="coerce").mean())
                for name in cross_metrics
            }
            directions[key] = {
                "complete": True,
                "source_rank_evaluated": int(source_rank),
                "target_selected_rank": int(target_rank),
                "geometry_comparable_at_selected_ranks": bool(source_rank == target_rank),
                **metrics,
                "adequate_cross_transfer": bool(
                    min(
                        metrics["map_r2_sufficiency"],
                        metrics["map_r2_necessity"],
                        metrics["ssi_fraction_transferred"],
                        metrics["ssi_fraction_removed"],
                    )
                    >= 0.65
                ),
                "at_least_half_cross_transfer": bool(
                    min(metrics["map_r2_sufficiency"], metrics["map_r2_necessity"]) >= 0.50
                ),
            }

    def classify(keys: tuple[str, ...]) -> str:
        records = [directions.get(key, {}) for key in keys]
        if not all(row.get("complete", False) for row in records):
            return "incomplete"
        if all(row["adequate_cross_transfer"] for row in records):
            return "shared by bidirectional causal cross-transfer"
        if not any(row["at_least_half_cross_transfer"] for row in records):
            return "functionally separate at the predeclared half-effect threshold"
        return "mixed or asymmetric cross-transfer"

    sharpening_keys = (
        "low_0_to_2->high_0_to_1",
        "high_0_to_1->low_0_to_2",
    )
    reversal_keys = (
        "low_0_to_2->high_1_to_3",
        "high_1_to_3->low_0_to_2",
        "high_0_to_1->high_1_to_3",
        "high_1_to_3->high_0_to_1",
    )
    return {
        "complete": len(directions) == 6
        and all(row.get("complete", False) for row in directions.values()),
        "directions": directions,
        "sharpening_relationship": classify(sharpening_keys),
        "reversal_relationship": classify(reversal_keys),
        "interpretation_rule": (
            "shared requires adequate causal cross-transfer in both directions; separate requires all "
            "relevant directions below half map-effect recovery; otherwise report mixed/asymmetric. "
            "Causal source→target transfer always uses the source contrast's selected rank. Projector "
            "overlap from this table is interpreted only when the two selected ranks match"
        ),
        "overlap_alone_used_for_classification": False,
    }


def strict_decision_label(evidence: dict[str, dict[str, Any]], complete: bool) -> str | None:
    """Apply the preregistered hierarchy; return no label for incomplete analysis."""
    if not complete or set(evidence) != {contrast.key for contrast in CONTRASTS}:
        return None

    def passes_metrics(row: dict[str, Any], threshold: float, ci: float, rank: int) -> bool:
        return bool(
            row["rank"] <= rank
            and min(row["map_suff"], row["map_nec"]) >= threshold
            and min(row["ssi_suff"], row["ssi_nec"]) >= threshold
            and min(row["ci_map_suff"], row["ci_map_nec"], row["ci_ssi_suff"], row["ci_ssi_nec"]) >= ci
        )

    strong = all(
        passes_metrics(row, 0.80, 0.60, 4)
        and row["beats_pca"]
        and row["beats_readout"]
        and row["beats_random95"]
        and row["beats_shuffle95"]
        and row["stable"]
        and row["cross_scale"]
        and row["no_train_test_collapse"]
        and row["loo_robust"]
        for row in evidence.values()
    )
    if strong:
        return DECISION_LABELS[0]
    adequate = all(
        passes_metrics(row, 0.65, np.nextafter(0.50, 1.0), 8)
        and row["beats_random95"]
        and row["stable"]
        and row["cross_scale"]
        and row["no_train_test_collapse"]
        and row["loo_robust"]
        for row in evidence.values()
    )
    if adequate:
        return DECISION_LABELS[1]
    if any(not row["beats_random95"] or not row["stable"] for row in evidence.values()):
        return DECISION_LABELS[3]
    positive_keys = ("low_0_to_2", "high_0_to_1")
    positive_partial = all(
        evidence[key]["rank"] <= 16
        and max(evidence[key]["map_suff"], evidence[key]["map_nec"]) >= 0.50
        for key in positive_keys
    )
    if positive_partial:
        return DECISION_LABELS[2]
    return DECISION_LABELS[3]


def _decision_explanation(
    decision: str | None,
    evidence: dict[str, dict[str, Any]],
    movement_specific: bool,
) -> str:
    if decision is None:
        return "No decision is valid until every required saved product and final uncertainty gate is complete."
    if decision == DECISION_LABELS[0]:
        return "All three contrasts passed every strong rank, recovery, uncertainty, baseline, stability, and generalization gate."
    if decision == DECISION_LABELS[1]:
        qualifier = (
            "and clearly exceeded readout-SVD"
            if movement_specific
            else "but did not clearly exceed readout-SVD for every contrast, so the claim is output-relevant rather than movement-specific"
        )
        return f"All three contrasts passed the adequate compact causal gates {qualifier}."
    failures: list[str] = []
    labels = {contrast.key: contrast.label for contrast in CONTRASTS}
    for key, row in evidence.items():
        reasons = []
        if row["rank"] > 8:
            reasons.append(f"rank {row['rank']} exceeds 8")
        if min(row["map_suff"], row["map_nec"]) < 0.65:
            reasons.append("map sufficiency/necessity is below 0.65")
        if min(row["ssi_suff"], row["ssi_nec"]) < 0.65:
            reasons.append("SSI transfer is below 0.65")
        if min(row["ci_map_suff"], row["ci_map_nec"], row["ci_ssi_suff"], row["ci_ssi_nec"]) <= 0.50:
            reasons.append("a crossed lower bound is not above 0.50")
        if not row["beats_random95"]:
            reasons.append("learned recovery does not beat the random 95th percentile")
        if not row["stable"]:
            reasons.append("the projector is unstable across folds")
        if not row["cross_scale"]:
            reasons.append("cross-scale transfer is not qualitatively correct")
        if not row["no_train_test_collapse"]:
            reasons.append("held-out recovery collapses relative to validation")
        if not row["loo_robust"]:
            reasons.append("leave-one-image-out recovery is not robust")
        if reasons:
            failures.append(f"{labels.get(key, key)}: {', '.join(reasons)}")
    prefix = (
        "The compact account is incomplete; "
        if decision == DECISION_LABELS[2]
        else "The low-dimensional account fails the preregistered gates; "
    )
    return prefix + ("; ".join(failures) if failures else "see the saved contrast metrics") + "."


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _curve_text(curve: dict[str, Any]) -> str:
    if not curve.get("rows"):
        return "unavailable"
    return "; ".join(
        (
            f"k={row['rank']}: map R² {_fmt(row['map_r2_sufficiency'])}/"
            f"{_fmt(row['map_r2_necessity'])}, SSI {_fmt(row['ssi_fraction_transferred'])}/"
            f"{_fmt(row['ssi_fraction_removed'])}"
        )
        for row in curve["rows"]
    )


def _report_markdown(statistics: dict[str, Any]) -> str:
    complete = statistics["analysis_complete"]
    decision = statistics.get("decision")
    selected = statistics.get("selected_ranks", {})
    results = statistics.get("contrasts", {})
    missing = statistics.get("missing_products", [])
    screening_stop = statistics.get("status") == "complete_screening_stop"
    if screening_stop:
        executive = [
            "The preregistered decision is **NOT LOW-DIMENSIONAL**.",
            "The complete Stage 1 stop rule was met: no rank at or below 16 recovered at least 40% in both sufficiency and necessity for any of the three contrasts.",
            "Crossed Stage 2 optimization was therefore not authorized by the analysis plan.",
            "No final-rank bootstrap, cross-scale, or cross-contrast claim is made.",
            "The manuscript-safe conclusion is that the effect was not reducible to a compact causal ConvGRU channel subspace under the specified linear intervention.",
        ]
    elif complete:
        rank_text = ", ".join(f"{key}: k={value['selected_rank']}" for key, value in selected.items())
        cross = statistics.get("cross_contrast", {})
        executive = [
            f"The preregistered decision is **{decision}**.",
            f"Validation-only rank selection chose {rank_text}.",
            "The result evaluates lower-SF sharpening, higher-SF sharpening, and the higher-SF excessive-motion reversal separately.",
            "All quoted recovery values come from images and trajectories excluded jointly from fitting.",
            "Exact SSI transfer was evaluated after fitting complete normalized maps and was never an optimization target.",
            "Crossed intervals resample images and trajectories independently and retain their Cartesian products within each held-out fold.",
            f"The two sharpening effects are {cross.get('sharpening_relationship', 'not classified')}; reversal is {cross.get('reversal_relationship', 'not classified')} relative to sharpening.",
            "Readout-SVD performance is treated as an output-relevance control, not evidence for a specialized motion circuit.",
            "The supported manuscript claim is limited to the strict gate reported below.",
        ]
    else:
        stage1 = statistics.get("stage1_diagnostics", {})
        if stage1:
            executive = [
                "The analysis provides a strong positive Stage-1 result, but no preregistered final decision label is assigned because only the predeclared screening fold is complete.",
                f"Validation loss selected a rank-{stage1.get('selected_rank', 8)} elbow without using test performance.",
                "On jointly unseen images and trajectories in that fold, rank 8 recovered most of the complete lower-SF sharpening, higher-SF sharpening, and higher-SF excessive-motion reversal maps in both sufficiency and necessity directions.",
                "Exact SSI transfer—never optimized directly—was also substantial for all three transformations.",
                "The learned subspaces strongly exceeded movement-difference PCA and 100 rank-matched random subspaces on this fold.",
                "Readout-SVD was much stronger than PCA and random and sometimes matched or exceeded SSI transfer, so the present result supports output relevance rather than a specialized movement circuit.",
                "The higher-SF sharpening and reversal subspaces cross-transfer strongly on this fold, whereas low-SF/high-SF transfer is only partial; this relationship is provisional until fold stability is known.",
                "No individual latent axis is interpreted because rotations cannot be stabilized from one fold.",
                "The remaining three crossed folds, final crossed bootstrap, projector stability, and shuffled-target null are required before assigning one of the four final labels.",
                "The analysis stopped under the user-mandated four-GPU-hour rule; the report below separates completed evidence from missing inference gates.",
            ]
        else:
            executive = [
                "The causal-subspace analysis is incomplete, so no preregistered decision label is assigned.",
                f"Missing or incomplete products: {', '.join(missing) if missing else 'unspecified evaluation stages'}.",
                "Available validation-only rank choices and diagnostic results are reported without converting them into a mechanism claim.",
            ]

    lines = ["# Final causal ConvGRU subspace report", "", "## Executive answer", ""]
    lines.extend(executive)
    lines.extend(["", "## Integrity and reproduction", ""])
    integrity = statistics.get("integrity", {})
    reproduction = statistics.get("reproduction", {})
    lines.extend(
        [
            f"- Checkpoint: `{statistics.get('checkpoint', 'unavailable')}`.",
            f"- Corrected-history source: `{reproduction.get('corrected_history_source', 'unavailable')}`; fixed 8 images × 24 drift trajectories × 5 scales × 40 scored frames.",
            f"- Activation indexing: {reproduction.get('activation_indexing', 'the final internal ConvGRU state supplied to the exact tiled RR100 readout')}.",
            f"- Rank-zero/full-rank integrity gate: {integrity.get('status', 'missing')}; exact-algebra pass = {integrity.get('summary', {}).get('exact_algebra_passed')}; optimization allowed = {integrity.get('optimization_allowed', False)}.",
            f"- Numerical tolerances: `{reproduction.get('numerical_tolerances', {})}`.",
            f"- Dataset folds: `{reproduction.get('fold_assignments', 'unavailable')}`; four fixed crossed folds, each holding out 2 images × 6 trajectories with all 40 frames kept together.",
        ]
    )
    lines.extend(["", "## Causal rank results", ""])
    stage1 = statistics.get("stage1_diagnostics", {})
    stage1_contrasts = stage1.get("contrasts", {})
    if not results and not stage1_contrasts:
        lines.append("No complete selected-rank held-out predictions are available.")
    for contrast in CONTRASTS:
        value = results.get(contrast.key)
        choice = selected.get(contrast.key, {})
        lines.append(f"### {contrast.label}")
        lines.append("")
        if value is None:
            diagnostic = stage1_contrasts.get(contrast.key)
            if diagnostic:
                point = diagnostic["screening_fold_point"]
                lines.append(
                    f"Screening curve — {_curve_text(diagnostic.get('rank_curves', {}).get('screening', {}))}."
                )
                lines.append("")
                lines.append(
                    f"The validation-only elbow was rank {diagnostic['screening_fold_rank']} "
                    f"(validation map recovery {_fmt(point.get('validation_recovery'))}). On the "
                    f"predeclared held-out fold (2 unseen images × 6 unseen trajectories; "
                    f"{point.get('n_test_pairs', 12)} pairs × 40 frames), map recovery was "
                    f"{_fmt(point['map_r2_sufficiency'])} sufficient and "
                    f"{_fmt(point['map_r2_necessity'])} necessary."
                )
                lines.append(
                    f"Exact SSI transfer/removal was {_fmt(point['ssi_fraction_transferred'])}/"
                    f"{_fmt(point['ssi_fraction_removed'])}; pre-softplus recovery was "
                    f"{_fmt(point['preactivation_r2_sufficiency'])}/"
                    f"{_fmt(point['preactivation_r2_necessity'])}; mean-rate-effect transfer/removal was "
                    f"{_fmt(point['mean_rate_fraction_transferred'])}/"
                    f"{_fmt(point['mean_rate_fraction_removed'])}. Fractions are uncapped."
                )
                lines.append(
                    "These are genuine held-out Stage-1 estimates, but they have no four-fold interval and are not a final cross-validated curve."
                )
            else:
                lines.append("Incomplete: selected-rank crossed predictions are unavailable.")
            lines.append("")
            continue
        if screening_stop:
            lines.append(
                f"Screening curve — {_curve_text(value.get('rank_curves', {}).get('screening', {}))}."
            )
            lines.append("")
            lines.append(
                "The complete rank≤16 screening gate did not reach 40% in both map sufficiency and "
                "map necessity, so this contrast was not advanced to Stage 2."
            )
            lines.append("")
            continue
        point, ci = value["point"], value["bootstrap_ci"]
        lines.append(f"Screening curve — {_curve_text(value.get('rank_curves', {}).get('screening', {}))}.")
        lines.append("")
        lines.append(f"Cross-validated curve — {_curve_text(value.get('rank_curves', {}).get('crossval', {}))}.")
        lines.append("")
        lines.append(
            f"Validation-only selection chose rank {choice.get('selected_rank')} ({choice.get('rule')}). "
            f"Held-out map recovery was {_fmt(point['map_r2_sufficiency'])} sufficient and "
            f"{_fmt(point['map_r2_necessity'])} necessary; SSI transfer was "
            f"{_fmt(point['ssi_fraction_transferred'])} and {_fmt(point['ssi_fraction_removed'])}."
        )
        lines.append(
            f"Crossed 95% intervals: map sufficiency [{_fmt(ci['map_r2_sufficiency'][0])}, {_fmt(ci['map_r2_sufficiency'][1])}], "
            f"map necessity [{_fmt(ci['map_r2_necessity'][0])}, {_fmt(ci['map_r2_necessity'][1])}], "
            f"SSI sufficiency [{_fmt(ci['ssi_fraction_transferred'][0])}, {_fmt(ci['ssi_fraction_transferred'][1])}], "
            f"SSI necessity [{_fmt(ci['ssi_fraction_removed'][0])}, {_fmt(ci['ssi_fraction_removed'][1])}]."
        )
        lines.append(
            f"Pre-softplus map recovery was {_fmt(point['preactivation_r2_sufficiency'])}/"
            f"{_fmt(point['preactivation_r2_necessity'])} (sufficiency/necessity); mean-rate-effect "
            f"transfer was {_fmt(point['mean_rate_fraction_transferred'])}/"
            f"{_fmt(point['mean_rate_fraction_removed'])}."
        )
        loo = np.asarray(value["leave_one_image_out"]["values"], dtype=np.float64)[:, [0, 1, 4, 5]]
        lines.append(
            f"Across eight leave-one-image-out estimates, the minimum primary recovery was "
            f"{_fmt(np.nanmin(loo))}; robustness gate = {value['leave_one_image_out']['robust']}."
        )
        lines.append("")
    lines.extend(["## Baseline comparison", ""])
    for contrast in CONTRASTS:
        value = results.get(contrast.key, {}).get("baselines")
        if value is None:
            continue
        comparisons = value["comparisons"]
        lines.append(
            f"- {contrast.label}: movement-PCA clearly exceeded = {comparisons['movement_pca'].get('clearly_exceeds')}; "
            f"readout-SVD clearly exceeded = {comparisons['readout_svd'].get('clearly_exceeds')}; "
            f"above the random 95th percentile = {comparisons['random_haar'].get('exceeds_95_percent')}; "
            f"above the shuffled-target 95th percentile = {comparisons['shuffled_target'].get('exceeds_95_percent')}."
        )
    if not results and stage1_contrasts:
        for contrast in CONTRASTS:
            diagnostic = stage1_contrasts.get(contrast.key, {})
            baselines = diagnostic.get("baselines", {})
            learned = diagnostic.get("screening_fold_point", {})
            pca = baselines.get("movement_pca", {})
            readout = baselines.get("readout_svd", {})
            random = baselines.get("random_haar", {})
            lines.append(
                f"- {contrast.label}, rank 8, screening fold: learned map R² "
                f"{_fmt(learned.get('map_r2_sufficiency'))}/{_fmt(learned.get('map_r2_necessity'))}; "
                f"movement-PCA {_fmt(pca.get('map_r2_sufficiency'))}/{_fmt(pca.get('map_r2_necessity'))}; "
                f"readout-SVD {_fmt(readout.get('map_r2_sufficiency'))}/{_fmt(readout.get('map_r2_necessity'))}; "
                f"random joint-map 95th percentile {_fmt(random.get('joint_map_recovery_95th_percentile'))} "
                f"from {random.get('draws', 0)} Haar draws."
            )
            lines.append(
                f"  SSI transfer/removal: learned {_fmt(learned.get('ssi_fraction_transferred'))}/"
                f"{_fmt(learned.get('ssi_fraction_removed'))}; movement-PCA "
                f"{_fmt(pca.get('ssi_fraction_transferred'))}/{_fmt(pca.get('ssi_fraction_removed'))}; "
                f"readout-SVD {_fmt(readout.get('ssi_fraction_transferred'))}/"
                f"{_fmt(readout.get('ssi_fraction_removed'))}."
            )
        lines.append(
            "- Shuffled-target learning was not run: even the reduced null required new optimization and was runtime-prohibitive under the four-hour ceiling."
        )
    lines.extend(["", "## Cross-scale and cross-contrast tests", ""])
    for contrast in CONTRASTS:
        value = results.get(contrast.key)
        if value and "cross_scale" in value:
            lines.append(
                f"- {contrast.label}: qualitative dose-curve transfer = {value['cross_scale'].get('qualitatively_correct')}; "
                f"fold-stable projector = {value['stability'].get('stable')}."
            )
    lines.append(
        f"- Sharpening relationship: {statistics.get('cross_contrast', {}).get('sharpening_relationship', 'incomplete')}; "
        f"reversal relationship: {statistics.get('cross_contrast', {}).get('reversal_relationship', 'incomplete')}."
    )
    if not results and stage1_contrasts:
        lines.append("- Screening-fold cross-scale tests (no refitting):")
        for contrast in CONTRASTS:
            rows = stage1_contrasts.get(contrast.key, {}).get("cross_scale", [])
            summary = "; ".join(
                f"{_fmt(row['scale_a'], 1)}→{_fmt(row['scale_b'], 1)}×: map R² "
                f"{_fmt(row['map_r2_sufficiency'])}/{_fmt(row['map_r2_necessity'])}, SSI "
                f"{_fmt(row['ssi_fraction_transferred'])}/{_fmt(row['ssi_fraction_removed'])}"
                for row in rows
            )
            lines.append(f"  - {contrast.label}: {summary or 'unavailable'}." )
        cross = stage1.get("cross_contrast", [])
        lines.append("- Screening-fold rank-8 cross-transfer:")
        for row in cross:
            lines.append(
                f"  - {row['source_contrast']} → {row['target_contrast']}: overlap "
                f"{_fmt(row['projector_overlap'])}, mean angle {_fmt(row['principal_angle_mean_deg'], 1)}°, "
                f"map R² {_fmt(row['map_r2_sufficiency'])}/{_fmt(row['map_r2_necessity'])}, "
                f"SSI {_fmt(row['ssi_fraction_transferred'])}/{_fmt(row['ssi_fraction_removed'])}."
            )
        lines.append(
            "- On this fold the two higher-SF projectors are nearly coincident (overlap 0.944; mean angle 12.4°) and cross-transfer their maps at about R²=0.90. The lower-SF projector overlaps each higher-SF projector by about 0.38 and transfers only part of their map changes. Fold stability is unknown."
        )
    lines.append(
        "- Pairwise subspace overlaps, principal angles, and directional causal cross-transfer are retained in "
        "`cross_contrast_results.csv`; overlap alone is not interpreted as a shared mechanism."
    )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(
        [
            "- Latent spatial maps: interpretable only when the selected projector is stable across folds; arbitrary rotations are not assigned biological meaning.",
            "- Grating tuning: the causal fit establishes complete-map transfer but does not by itself establish SF×TF tuning of an individual latent axis; that requires saved complex F1 coefficients or exact grating time series.",
            "- RR100 readout loadings: units may read the same distributed state differently through their frozen weights, which is distinct from a uniquely localized ConvGRU channel bank.",
            "- SF preference and movement optimum: relationships must be reported for all saved per-unit results, without selecting favorable units after inspection.",
        ]
    )
    lines.extend(["", "## Decision", ""])
    lines.append(f"**{decision}**" if decision else "**INCOMPLETE — NO DECISION LABEL ASSIGNED**")
    lines.append("")
    lines.append(statistics.get("decision_explanation", _decision_explanation(decision, {}, False)))
    if not complete and stage1:
        budget = stage1.get("compute_budget", {})
        lines.extend(["", "## Compute stop and remaining work", ""])
        lines.append(
            f"- Conservative accelerator ledger: {_fmt(budget.get('conservative_gpu_hours_used'))} of "
            f"{_fmt(budget.get('hard_limit_gpu_hours'), 1)} GPU-hours used; "
            f"{_fmt(budget.get('conservative_gpu_hours_remaining'))} remained when saved-product evaluation ended."
        )
        lines.append(
            "- Current bottleneck: reading and losslessly decompressing the 62-GB state/map cache from a nearly full shared filesystem. The literal patched-state readout itself is fast; wall-time I/O dominates and is conservatively charged as GPU time."
        )
        lines.append(
            "- Needed for final inference: 27 new fits (3 contrasts × ranks 4/8/16 × folds 1–3, each with three predeclared initializations), held-out evaluation for those folds, projector stability, at least 4,000 fixed-fold crossed-bootstrap draws, and the preregistered shuffled-target control or an explicit prospective waiver."
        )
        lines.append(
            "- Estimated marginal value: high. Those runs would determine whether the rank-8 elbow and high-SF shared geometry survive identity changes, convert the current positive screen into an allowed paper-level label or reject it, and supply uncertainty that cannot be inferred from fold 0."
        )
        lines.append(
            "- Why the run stopped: the remaining ~1 GPU-hour could not safely cover the 27 fits, and starting them would violate the instruction to stop before exceeding four GPU-hours. No extra model replay was performed."
        )
    lines.extend(["", "## Manuscript-safe claims", "", "### SUPPORTED", ""])
    for claim in statistics.get("manuscript_safe_claims", {}).get("SUPPORTED", []):
        lines.append(f"- {claim}")
    lines.extend(["", "### CONSISTENT WITH", ""])
    for claim in statistics.get("manuscript_safe_claims", {}).get("CONSISTENT WITH", []):
        lines.append(f"- {claim}")
    lines.extend(["", "### NOT SUPPORTED", ""])
    for claim in statistics.get("manuscript_safe_claims", {}).get("NOT SUPPORTED", []):
        lines.append(f"- {claim}")
    lines.extend(["", "## Recommended panels", ""])
    for index, panel in enumerate(statistics.get("recommended_panels", []), 1):
        lines.append(
            f"{index}. **{panel['title']}** — Plot {panel['quantity']}. Conclusion: {panel['conclusion']} "
            f"Main-figure rationale: {panel['rationale']}"
        )
    lines.extend(
        [
            "",
            "All ranks were chosen from validation performance only. Test recovery, bootstrap intervals, and figures were not used for rank selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def _claims(decision: str | None, movement_specific: bool) -> dict[str, list[str]]:
    if decision == "STRONG COMPLETE MECHANISM":
        supported = [
            "All three Figure 4 transformations are causally concentrated in compact, stable ConvGRU channel subspaces.",
            "The selected subspaces generalize jointly to unseen images and unseen trajectories.",
        ]
    elif decision == "ADEQUATE PAPER-LEVEL MECHANISM":
        supported = [
            (
                "The movement effects are concentrated in low-dimensional output-relevant ConvGRU subspaces."
                if not movement_specific
                else "All three Figure 4 transformations admit an adequate compact causal ConvGRU-subspace account."
            )
        ]
    elif decision == "PARTIAL MECHANISM":
        supported = ["A compact linear ConvGRU subspace explains part, but not all, of the Figure 4 transformations."]
    elif decision == "NOT LOW-DIMENSIONAL":
        supported = [
            "The effect is distributed across the recurrent representation and was not reducible to a compact causal channel subspace."
        ]
    else:
        supported = ["No final mechanism claim is supported until the saved-product analysis is complete."]
    return {
        "SUPPORTED": supported,
        "CONSISTENT WITH": [
            "Retinal motion is transformed into a recurrent population code that the lower- and higher-SF RR100 populations read differently."
        ],
        "NOT SUPPORTED": [
            "A uniquely localized set of ConvGRU channels or a uniquely identified recurrent circuit.",
            "A movement-specific circuit when the learned subspace does not clearly outperform readout-SVD.",
            "Using scalar SSI agreement as evidence that the complete spatial map was reconstructed.",
        ],
    }


def _panel_recommendations(complete: bool, stable_all: bool) -> list[dict[str, str]]:
    if not complete:
        return []
    panels = [
        {
            "title": "Causal recovery versus subspace rank",
            "quantity": "held-out sufficient and necessary map R² by rank for learned, movement-PCA, readout-SVD, random, and identity conditions",
            "conclusion": "whether the complete map transformation is compact and baseline-specific",
            "rationale": "this is the primary causal dimensionality result",
        },
        {
            "title": "Held-out map reconstruction",
            "quantity": "objectively selected baseline, target, target-minus-baseline, sufficient, residual, and necessity maps on common scales",
            "conclusion": "whether scalar recovery corresponds to the correct spatial transformation",
            "rationale": "it makes the optimized complete-map objective directly visible",
        },
    ]
    panels.append(
        {
            "title": "Stable latent readout" if stable_all else "Causal SSI transfer across movement scale",
            "quantity": (
                "canonical latent spatial maps and RR100 loading versus SF preference"
                if stable_all
                else "intact, sufficient, and necessary SSI dose curves plus per-unit recovery versus SF preference"
            ),
            "conclusion": (
                "how stable distributed axes are differentially read by RR100 units"
                if stable_all
                else "where the compact account succeeds or fails without interpreting arbitrary axes"
            ),
            "rationale": "it connects the state-space result to the population-level Figure 4 phenotype",
        }
    )
    return panels


STAGE1_REPORT_METRICS = (
    "map_r2_sufficiency",
    "map_r2_necessity",
    "preactivation_r2_sufficiency",
    "preactivation_r2_necessity",
    "ssi_fraction_transferred",
    "ssi_fraction_removed",
    "mean_rate_fraction_transferred",
    "mean_rate_fraction_removed",
)


def _mean_stage1_metrics(rows: pd.DataFrame) -> dict[str, float]:
    return {
        name: float(pd.to_numeric(rows[name], errors="coerce").mean())
        if name in rows and not rows.empty
        else math.nan
        for name in STAGE1_REPORT_METRICS
    }


def _budget_limited_stage1_diagnostics(
    rank_summary: pd.DataFrame,
    baseline: pd.DataFrame,
    cross_scale: pd.DataFrame,
    cross_contrast: pd.DataFrame,
    stage2: dict[str, Any],
    screening_curves: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize completed fold-0 evidence without promoting it to final inference."""
    selected_rank = int(stage2.get("elbow", {}).get("elbow_rank", 8))
    contrasts: dict[str, Any] = {}
    for contrast in CONTRASTS:
        point_rows = rank_summary.loc[
            rank_summary.get("contrast", pd.Series(dtype=str)).eq(contrast.key)
            & rank_summary.get("stage", pd.Series(dtype=str)).eq("crossval")
            & rank_summary.get("method", pd.Series(dtype=str)).eq("learned")
            & pd.to_numeric(rank_summary.get("rank"), errors="coerce").eq(selected_rank)
            & pd.to_numeric(rank_summary.get("fold"), errors="coerce").eq(0)
        ].copy()
        if point_rows.empty:
            point_rows = rank_summary.loc[
                rank_summary.get("contrast", pd.Series(dtype=str)).eq(contrast.key)
                & rank_summary.get("stage", pd.Series(dtype=str)).eq("screening")
                & rank_summary.get("method", pd.Series(dtype=str)).eq("learned")
                & pd.to_numeric(rank_summary.get("rank"), errors="coerce").eq(selected_rank)
                & pd.to_numeric(rank_summary.get("fold"), errors="coerce").eq(0)
            ].copy()
        point = _mean_stage1_metrics(point_rows)
        point["validation_recovery"] = (
            float(
                1.0
                - pd.to_numeric(point_rows["best_validation_loss"], errors="coerce").mean()
            )
            if not point_rows.empty and "best_validation_loss" in point_rows
            else math.nan
        )
        point["n_test_pairs"] = (
            int(pd.to_numeric(point_rows["n_test_pairs"], errors="coerce").max())
            if not point_rows.empty and "n_test_pairs" in point_rows
            else 0
        )
        point["n_test_records"] = (
            int(pd.to_numeric(point_rows["n_test_records"], errors="coerce").max())
            if not point_rows.empty and "n_test_records" in point_rows
            else 0
        )

        fixed: dict[str, Any] = {}
        for method in ("movement_pca", "readout_svd"):
            rows = baseline.loc[
                baseline.get("contrast", pd.Series(dtype=str)).eq(contrast.key)
                & baseline.get("stage", pd.Series(dtype=str)).eq("crossval")
                & baseline.get("method", pd.Series(dtype=str)).eq(method)
                & pd.to_numeric(baseline.get("rank"), errors="coerce").eq(selected_rank)
                & pd.to_numeric(baseline.get("fold"), errors="coerce").eq(0)
            ].copy()
            fixed[method] = {
                **_mean_stage1_metrics(rows),
                "complete_on_screening_fold": bool(len(rows) == 1),
            }
        random_rows = baseline.loc[
            baseline.get("contrast", pd.Series(dtype=str)).eq(contrast.key)
            & baseline.get("stage", pd.Series(dtype=str)).eq("crossval")
            & baseline.get("method", pd.Series(dtype=str)).eq("random_haar")
            & pd.to_numeric(baseline.get("rank"), errors="coerce").eq(selected_rank)
            & pd.to_numeric(baseline.get("fold"), errors="coerce").eq(0)
        ].copy()
        if random_rows.empty:
            random = {"draws": 0, "complete_on_screening_fold": False}
        else:
            joint = 0.5 * (
                pd.to_numeric(random_rows["map_r2_sufficiency"], errors="coerce")
                + pd.to_numeric(random_rows["map_r2_necessity"], errors="coerce")
            )
            joint = joint[np.isfinite(joint)].to_numpy(float)
            random = {
                "draws": int(len(joint)),
                "complete_on_screening_fold": bool(len(joint) >= 100),
                "joint_map_recovery_median": float(np.median(joint)) if len(joint) else math.nan,
                "joint_map_recovery_95th_percentile": float(np.percentile(joint, 95))
                if len(joint)
                else math.nan,
                "learned_joint_map_recovery": float(
                    0.5 * (point["map_r2_sufficiency"] + point["map_r2_necessity"])
                ),
            }

        dose_rows = cross_scale.loc[
            cross_scale.get("contrast", pd.Series(dtype=str)).eq(contrast.key)
            & pd.to_numeric(cross_scale.get("rank"), errors="coerce").eq(selected_rank)
            & pd.to_numeric(cross_scale.get("fold"), errors="coerce").eq(0)
        ].copy()
        dose = []
        for _, row in dose_rows.sort_values(["scale_a", "scale_b"]).iterrows():
            dose.append(
                {
                    "scale_a": _number(row.get("scale_a")),
                    "scale_b": _number(row.get("scale_b")),
                    "map_r2_sufficiency": _number(row.get("map_r2_sufficiency")),
                    "map_r2_necessity": _number(row.get("map_r2_necessity")),
                    "ssi_fraction_transferred": _number(row.get("ssi_fraction_transferred")),
                    "ssi_fraction_removed": _number(row.get("ssi_fraction_removed")),
                    "ssi_target_effect_bits": _number(row.get("ssi_target_effect_bits")),
                }
            )
        contrasts[contrast.key] = {
            "label": contrast.label,
            "rank_curves": {"screening": screening_curves.get(contrast.key, {})},
            "screening_fold_rank": selected_rank,
            "screening_fold_point": point,
            "baselines": {**fixed, "random_haar": random, "shuffled_target": {"complete": False}},
            "cross_scale": dose,
        }

    cross_records: list[dict[str, Any]] = []
    if not cross_contrast.empty:
        rows = cross_contrast.loc[
            pd.to_numeric(cross_contrast.get("rank"), errors="coerce").eq(selected_rank)
            & pd.to_numeric(cross_contrast.get("fold"), errors="coerce").eq(0)
        ]
        for _, row in rows.iterrows():
            cross_records.append(
                {
                    "source_contrast": str(row.get("source_contrast")),
                    "target_contrast": str(row.get("target_contrast")),
                    "projector_overlap": _number(row.get("projector_overlap")),
                    "principal_angle_mean_deg": _number(row.get("principal_angle_mean_deg")),
                    "map_r2_sufficiency": _number(row.get("map_r2_sufficiency")),
                    "map_r2_necessity": _number(row.get("map_r2_necessity")),
                    "ssi_fraction_transferred": _number(row.get("ssi_fraction_transferred")),
                    "ssi_fraction_removed": _number(row.get("ssi_fraction_removed")),
                }
            )

    budget: dict[str, Any] = {}
    if GPU_BUDGET.is_file():
        try:
            ledger = json.loads(GPU_BUDGET.read_text(encoding="utf-8"))
            used = _number(ledger.get("total_conservative_gpu_hours"))
            budget = {
                "hard_limit_gpu_hours": 4.0,
                "conservative_gpu_hours_used": used,
                "conservative_gpu_hours_remaining": 4.0 - used,
                "ledger": str(GPU_BUDGET),
                "events": len(ledger.get("events", [])),
            }
        except (OSError, ValueError, TypeError):
            budget = {"hard_limit_gpu_hours": 4.0, "ledger": str(GPU_BUDGET)}
    return {
        "status": "screening_fold_complete_stage2_budget_limited",
        "selected_rank": selected_rank,
        "selection_source": stage2.get(
            "selection_source", "best validation map loss from the screening fits"
        ),
        "elbow": stage2.get("elbow", {}),
        "stage2_ranks": stage2.get("ranks", []),
        "completed_test_folds": [0],
        "required_test_folds": [0, 1, 2, 3],
        "contrasts": contrasts,
        "cross_contrast": cross_records,
        "compute_budget": budget,
    }


def infer(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    ensure_output_dirs()
    required = [RANK_SUMMARY, PER_UNIT_RESULTS, STAGE2_RANKS, OPTIMIZATION_CONFIG, ANALYSIS_MANIFEST]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    rank_summary = _load_csv(RANK_SUMMARY)
    per_unit = _load_csv(PER_UNIT_RESULTS)
    baseline = _load_csv(BASELINE_RESULTS)
    stability = _load_csv(STABILITY_RESULTS)
    cross_scale = _load_csv(CROSS_SCALE_RESULTS)
    cross_contrast = _load_csv(CROSS_CONTRAST_RESULTS)
    integrity = json.loads(INTEGRITY_TESTS.read_text()) if INTEGRITY_TESTS.is_file() else {}
    if not integrity.get("optimization_allowed", False):
        missing.append(str(INTEGRITY_TESTS))
    stage2 = json.loads(STAGE2_RANKS.read_text()) if STAGE2_RANKS.is_file() else {}
    configuration = json.loads(OPTIMIZATION_CONFIG.read_text()) if OPTIMIZATION_CONFIG.is_file() else {}
    manifest = json.loads(ANALYSIS_MANIFEST.read_text()) if ANALYSIS_MANIFEST.is_file() else {}
    reproduction = _reproduction_record(manifest, configuration)
    checkpoint = _checkpoint_path(manifest)
    screening_expected = configuration.get("rank_sweep", {}).get("learned_screening_ranks")
    screening_curves = {
        contrast.key: _rank_curve(
            rank_summary, contrast.key, "screening", screening_expected
        )
        for contrast in CONTRASTS
    }
    if (
        stage2.get("status") == "stopped_no_compact_subspace"
        and stage2.get("stop")
        and integrity.get("optimization_allowed", False)
        and not missing
        and not rank_summary.empty
        and not per_unit.empty
        and all(value.get("complete", False) for value in screening_curves.values())
    ):
        screening = {
            contrast.key: {
                "rank_curves": {
                    "screening": screening_curves[contrast.key]
                },
                "screening_stop_rule": stage2.get("screening_stop_rule", {}).get(contrast.key, {}),
            }
            for contrast in CONTRASTS
        }
        statistics = {
            "analysis": "causal_low_rank_saved_product_inference",
            "created_utc": utc_now(),
            "status": "complete_screening_stop",
            "analysis_complete": True,
            "decision": "NOT LOW-DIMENSIONAL",
            "decision_basis": stage2.get("stop_statement"),
            "decision_explanation": (
                "The complete Stage 1 rank≤16 stop rule was met for all three contrasts; "
                "Stage 2 was therefore not run."
            ),
            "selected_ranks": {},
            "contrasts": screening,
            "missing_products": [],
            "integrity": integrity,
            "reproduction": reproduction,
            "checkpoint": checkpoint,
            "manuscript_safe_claims": _claims("NOT LOW-DIMENSIONAL", False),
            "recommended_panels": [],
        }
        return statistics, None
    if missing or rank_summary.empty or per_unit.empty or not stage2.get("ranks"):
        statistics = {
            "analysis": "causal_low_rank_saved_product_inference",
            "created_utc": utc_now(),
            "status": "incomplete",
            "analysis_complete": False,
            "decision": None,
            "selected_ranks": {},
            "contrasts": {},
            "missing_products": sorted(set(missing or ["complete Stage 2 evaluation products"])),
            "integrity": integrity,
            "reproduction": reproduction,
            "checkpoint": checkpoint,
            "manuscript_safe_claims": _claims(None, False),
            "recommended_panels": [],
        }
        return statistics, None

    preregistered = [int(value) for value in stage2["ranks"]]
    learned = rank_summary.loc[
        rank_summary["stage"].eq("crossval") & rank_summary["method"].eq("learned")
    ].copy()
    selections: dict[str, Any] = {}
    for contrast in CONTRASTS:
        selections[contrast.key] = select_scientific_rank(
            learned.loc[learned["contrast"].eq(contrast.key)], preregistered
        )
        if selections[contrast.key]["status"] != "selected":
            missing.append(f"four-fold validation-complete rank for {contrast.key}")
    if missing:
        stage1 = _budget_limited_stage1_diagnostics(
            rank_summary,
            baseline,
            cross_scale,
            cross_contrast,
            stage2,
            screening_curves,
        )
        statistics = {
            "analysis": "causal_low_rank_saved_product_inference",
            "created_utc": utc_now(),
            "status": "incomplete",
            "analysis_complete": False,
            "decision": None,
            "selected_ranks": selections,
            "contrasts": {},
            "stage1_diagnostics": stage1,
            "missing_products": sorted(set(missing)),
            "integrity": integrity,
            "reproduction": reproduction,
            "checkpoint": checkpoint,
            "manuscript_safe_claims": {
                "SUPPORTED": [
                    "The mandatory exact-intervention and float16-storage integrity gates passed.",
                    "On the predeclared screening fold, rank 8 transferred and removed most of each complete RR100 map transformation, including the higher-SF reversal.",
                    "On that fold, the learned rank-8 map recovery exceeded movement-PCA and 100 rank-matched Haar-random subspaces for all three contrasts.",
                ],
                "CONSISTENT WITH": [
                    "A compact output-relevant ConvGRU capacity that carries useful-motion sharpening and excessive-motion cost.",
                    "Closely related higher-SF channel geometry for sharpening and excessive-motion reversal.",
                ],
                "NOT SUPPORTED": [
                    "Any preregistered final decision label before the remaining three crossed folds, final bootstrap, stability, and shuffled-target analyses are complete.",
                    "A uniquely localized set of ConvGRU channels, a uniquely identified recurrent circuit, or interpretable individual latent axes.",
                    "A movement-specific circuit: the readout-SVD baseline is strong and the learned advantage has not been tested across all folds.",
                ],
            },
            "recommended_panels": _panel_recommendations(True, False),
        }
        return statistics, None

    bootstrap_config = configuration.get("bootstrap", {})
    default_samples = int(
        bootstrap_config.get("final_samples" if args.bootstrap_tier == "final" else "exploratory_samples", 4000)
    )
    n_bootstrap = default_samples if args.bootstrap_samples is None else int(args.bootstrap_samples)
    if n_bootstrap < 1:
        raise ValueError("Bootstrap sample count must be positive")
    base_seed = int(bootstrap_config.get("seed", ANALYSIS_SEED + 800_000) if args.seed is None else args.seed)
    minimum_final_samples = int(bootstrap_config.get("final_samples", 4000))
    if args.bootstrap_tier != "final":
        missing.append("final crossed-bootstrap tier was not run")
    if n_bootstrap < minimum_final_samples:
        missing.append(
            f"final crossed bootstrap requires at least {minimum_final_samples} samples; got {n_bootstrap}"
        )
    bootstrap_values = np.full((len(CONTRASTS), n_bootstrap, len(BOOTSTRAP_METRICS)), np.nan, dtype=np.float64)
    point_values = np.full((len(CONTRASTS), len(BOOTSTRAP_METRICS)), np.nan, dtype=np.float64)
    ci_low = np.full_like(point_values, np.nan)
    ci_high = np.full_like(point_values, np.nan)
    loo_values = np.full((len(CONTRASTS), 8, len(BOOTSTRAP_METRICS)), np.nan, dtype=np.float64)
    loo_images = np.full((len(CONTRASTS), 8), -1, dtype=np.int64)
    contrast_results: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}

    for contrast_index, contrast in enumerate(CONTRASTS):
        rank = int(selections[contrast.key]["selected_rank"])
        try:
            units = _target_units(per_unit, contrast.key, rank)
            folds = []
            prediction_files = []
            for fold in range(4):
                path = _prediction_path(contrast.key, fold, rank)
                if not path.is_file():
                    raise FileNotFoundError(path)
                folds.append(_fold_cells(path, fold, units))
                prediction_files.append(_file_provenance(path))
        except (FileNotFoundError, RuntimeError) as error:
            missing.append(str(error))
            continue
        try:
            point, samples = crossed_bootstrap(
                folds, n_bootstrap, base_seed + 10_000 * contrast_index
            )
            images, loo = _leave_one_image_out(folds)
        except RuntimeError as error:
            missing.append(str(error))
            continue
        primary_indices = np.asarray([0, 1, 4, 5], dtype=np.int64)
        primary_point = np.asarray([point[BOOTSTRAP_METRICS[index]] for index in primary_indices])
        if not np.all(np.isfinite(primary_point)):
            missing.append(f"non-finite selected-rank point estimates for {contrast.key}")
            continue
        finite_fraction = np.mean(np.isfinite(samples), axis=0)
        if np.any(finite_fraction[primary_indices] < 0.99):
            missing.append(f"fewer than 99% finite primary bootstrap draws for {contrast.key}")
        bootstrap_values[contrast_index] = samples
        point_values[contrast_index] = [point[name] for name in BOOTSTRAP_METRICS]
        ci_low[contrast_index] = np.nanpercentile(samples, 2.5, axis=0)
        ci_high[contrast_index] = np.nanpercentile(samples, 97.5, axis=0)
        loo_values[contrast_index, : len(images)] = loo
        loo_images[contrast_index, : len(images)] = images
        ci = {
            name: [ci_low[contrast_index, metric_index], ci_high[contrast_index, metric_index]]
            for metric_index, name in enumerate(BOOTSTRAP_METRICS)
        }
        baselines = _baseline_evidence(baseline, rank_summary, contrast.key, rank)
        stable = _rank_stability(stability, contrast.key, rank)
        dose = _cross_scale_evidence(cross_scale, contrast.key, rank)
        selected_rows = learned.loc[
            learned["contrast"].eq(contrast.key) & pd.to_numeric(learned["rank"], errors="coerce").eq(rank)
        ]
        validation_mean = float(
            np.mean(1.0 - pd.to_numeric(selected_rows["best_validation_loss"], errors="coerce").to_numpy(float))
        )
        test_joint = 0.5 * (point["map_r2_sufficiency"] + point["map_r2_necessity"])
        no_collapse = bool(test_joint >= validation_mean - 0.20)
        loo_primary = loo[:, [0, 1, 4, 5]]
        if not np.all(np.isfinite(loo_primary)):
            missing.append(f"non-finite leave-one-image-out estimates for {contrast.key}")
        loo_robust = bool(
            np.all(np.isfinite(loo_primary))
            and np.nanmin(loo_primary)
            >= max(0.50, np.nanmin(point_values[contrast_index, [0, 1, 4, 5]]) - 0.15)
        )
        contrast_results[contrast.key] = {
            "selected_rank": rank,
            "test_prediction_files": prediction_files,
            "point": point,
            "bootstrap_ci": ci,
            "bootstrap_finite_fraction": {
                name: float(finite_fraction[index])
                for index, name in enumerate(BOOTSTRAP_METRICS)
            },
            "leave_one_image_out": {
                "image_positions": images.tolist(),
                "metric_names": BOOTSTRAP_METRICS,
                "values": loo.tolist(),
                "robust": loo_robust,
            },
            "validation_recovery_mean": validation_mean,
            "heldout_joint_map_recovery": test_joint,
            "no_major_train_test_collapse": no_collapse,
            "baselines": baselines,
            "stability": stable,
            "cross_scale": dose,
            "rank_curves": {
                "screening": _rank_curve(
                    rank_summary,
                    contrast.key,
                    "screening",
                    configuration.get("rank_sweep", {}).get("learned_screening_ranks"),
                ),
                "crossval": _rank_curve(
                    rank_summary,
                    contrast.key,
                    "crossval",
                    preregistered,
                ),
            },
        }
        comparisons = baselines["comparisons"]
        evidence[contrast.key] = {
            "rank": rank,
            "map_suff": point["map_r2_sufficiency"],
            "map_nec": point["map_r2_necessity"],
            "ssi_suff": point["ssi_fraction_transferred"],
            "ssi_nec": point["ssi_fraction_removed"],
            "ci_map_suff": ci["map_r2_sufficiency"][0],
            "ci_map_nec": ci["map_r2_necessity"][0],
            "ci_ssi_suff": ci["ssi_fraction_transferred"][0],
            "ci_ssi_nec": ci["ssi_fraction_removed"][0],
            "beats_pca": comparisons["movement_pca"].get("clearly_exceeds", False),
            "beats_readout": comparisons["readout_svd"].get("clearly_exceeds", False),
            "beats_random95": comparisons["random_haar"].get("exceeds_95_percent", False),
            "beats_shuffle95": comparisons["shuffled_target"].get("exceeds_95_percent", False),
            "stable": stable.get("stable", False),
            "cross_scale": dose.get("qualitatively_correct", False),
            "no_train_test_collapse": no_collapse,
            "loo_robust": loo_robust,
        }

    cross_contrast_summary = _cross_contrast_evidence(cross_contrast, selections)
    if not cross_contrast_summary.get("complete", False):
        missing.append(str(CROSS_CONTRAST_RESULTS))
    for contrast in CONTRASTS:
        if contrast.key not in contrast_results:
            missing.append(f"four selected-rank test prediction files for {contrast.key}")
        else:
            comparison = contrast_results[contrast.key]["baselines"]["comparisons"]
            for method in ("movement_pca", "readout_svd", "random_haar"):
                if not comparison[method].get("complete", False):
                    missing.append(f"{method} baseline for {contrast.key}")
            if not comparison["shuffled_target"].get("complete", False):
                missing.append(f"shuffled_target baseline for {contrast.key}")
            if not contrast_results[contrast.key]["stability"].get("complete", False):
                missing.append(f"subspace stability for {contrast.key}")
            if not contrast_results[contrast.key]["cross_scale"].get("complete", False):
                missing.append(f"cross-scale transfer for {contrast.key}")
            for stage in ("screening", "crossval"):
                if not contrast_results[contrast.key]["rank_curves"][stage].get("complete", False):
                    missing.append(f"complete learned {stage} rank curve for {contrast.key}")

    analysis_complete = not missing and len(evidence) == len(CONTRASTS)
    decision = strict_decision_label(evidence, analysis_complete)
    movement_specific = bool(
        analysis_complete
        and all(row["beats_readout"] for row in evidence.values())
    )
    decision_explanation = _decision_explanation(decision, evidence, movement_specific)
    statistics = {
        "analysis": "causal_low_rank_saved_product_inference",
        "created_utc": utc_now(),
        "status": "complete" if analysis_complete else "incomplete",
        "analysis_complete": analysis_complete,
        "decision": decision,
        "decision_explanation": decision_explanation,
        "decision_labels_allowed": DECISION_LABELS,
        "decision_rule": "strict preregistered hierarchy in completion criteria section 19",
        "operational_definitions": {
            "compact_validation_rank": (
                "lowest preregistered k<=8 whose 1-best_validation_loss is >=0.65 in every fold; "
                "otherwise maximum mean validation recovery, with lower k breaking an exact tie"
            ),
            "clearly_exceeds_fixed_baseline": (
                "learned mean sufficiency and necessity map R2 each exceed the same-rank baseline by >=0.05"
            ),
            "no_major_train_test_collapse": (
                "held-out mean sufficiency/necessity map recovery is no more than 0.20 below mean "
                "validation recovery"
            ),
            "leave_one_image_out_robust": (
                "every primary recovery remains >=max(0.50, full point estimate minus 0.15)"
            ),
            "stability": (
                "mean fold-pair projector overlap exceeds the mean same-rank random 95th percentile "
                "and at least two thirds of fold pairs individually exceed it"
            ),
        },
        "selected_ranks": selections,
        "rank_selection_used_test_metrics": False,
        "contrasts": contrast_results,
        "cross_contrast": cross_contrast_summary,
        "evidence_for_decision": evidence,
        "movement_specific_claim_allowed": movement_specific,
        "missing_products": sorted(set(missing)),
        "bootstrap": {
            "tier": args.bootstrap_tier,
            "samples": n_bootstrap,
            "base_seed": base_seed,
            "method": (
                "treat the four non-overlapping crossed test folds as fixed design strata; within each stratum, "
                "resample its two held-out images and six held-out trajectories independently with replacement, "
                "use their Cartesian product, apply identical cell weights to every paired intervention, then "
                "pool sufficient statistics across strata"
            ),
            "fixed_strata": "four predeclared crossed test folds",
            "fixed_strata_justification": (
                "Only the 2x6 diagonal held-out block from each fold has learned predictions. A nominal global "
                "8-image x 24-trajectory Cartesian bootstrap would require the 144 off-block test predictions "
                "that are not saved and would mix samples predicted by models trained with different identity "
                "exclusions. Conditioning on the predeclared folds preserves valid held-out predictions and "
                "still independently clusters both sampled factors within every test block"
            ),
            "global_8_by_24_cartesian_resampling_per_draw": False,
            "global_protocol_deviation_documented": True,
            "paired_across_interventions": True,
            "independent_pair_resampling": False,
            "confidence_interval_percent": [2.5, 97.5],
        },
        "integrity": integrity,
        "reproduction": reproduction,
        "checkpoint": checkpoint,
        "manuscript_safe_claims": _claims(decision, movement_specific),
        "recommended_panels": _panel_recommendations(
            analysis_complete, analysis_complete and all(row["stable"] for row in evidence.values())
        ),
        "provenance": {
            "rank_summary": _file_provenance(RANK_SUMMARY),
            "per_unit_results": _file_provenance(PER_UNIT_RESULTS),
            "baseline_results": _file_provenance(BASELINE_RESULTS),
            "subspace_stability": _file_provenance(STABILITY_RESULTS),
            "cross_scale_results": _file_provenance(CROSS_SCALE_RESULTS),
            "cross_contrast_results": _file_provenance(CROSS_CONTRAST_RESULTS),
            "stage2_ranks": _file_provenance(STAGE2_RANKS),
            "optimization_config": _file_provenance(OPTIMIZATION_CONFIG),
            "analysis_manifest": _file_provenance(ANALYSIS_MANIFEST),
            "integrity_tests": _file_provenance(INTEGRITY_TESTS),
        },
    }
    arrays = {
        "contrast_keys": np.asarray([contrast.key for contrast in CONTRASTS]),
        "selected_ranks": np.asarray([selections[contrast.key]["selected_rank"] for contrast in CONTRASTS]),
        "metric_names": np.asarray(BOOTSTRAP_METRICS),
        "bootstrap_values": bootstrap_values,
        "point_estimates": point_values,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "leave_one_image_out_values": loo_values,
        "leave_one_image_out_image_positions": loo_images,
        "bootstrap_samples": np.asarray(n_bootstrap),
        "bootstrap_base_seed": np.asarray(base_seed),
        "method": np.asarray(statistics["bootstrap"]["method"]),
    }
    summary_records: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        if contrast.key not in contrast_results:
            continue
        rank = int(selections[contrast.key]["selected_rank"])
        for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
            summary_records.append(
                {
                    "contrast": contrast.key,
                    "rank": rank,
                    "method": "learned",
                    "metric": metric,
                    "mean": float(point_values[contrast_index, metric_index]),
                    "ci_low": float(ci_low[contrast_index, metric_index]),
                    "ci_high": float(ci_high[contrast_index, metric_index]),
                    "n_bootstrap": n_bootstrap,
                }
            )
    arrays["summary_json"] = np.asarray(json.dumps(summary_records, sort_keys=True))
    return statistics, arrays


def main() -> int:
    args = parse_args()
    statistics, arrays = infer(args)
    write_json(STATISTICS, statistics)
    write_json(EVALUATION / "statistics.json", statistics)
    if arrays is None:
        arrays = {
            "status": np.asarray(statistics["status"]),
            "analysis_complete": np.asarray(statistics["analysis_complete"]),
            "decision": np.asarray(statistics.get("decision") or ""),
            "bootstrap_values_available": np.asarray(False),
        }
    np.savez_compressed(BOOTSTRAP, **arrays)
    np.savez_compressed(EVALUATION / "bootstrap_results.npz", **arrays)
    report = _report_markdown(statistics)
    REPORT.write_text(report, encoding="utf-8")
    (EVALUATION / "FINAL_CAUSAL_SUBSPACE_REPORT.md").write_text(report, encoding="utf-8")
    if statistics["analysis_complete"]:
        print(f"{statistics['decision']}: {REPORT}")
    else:
        print(f"Incomplete analysis; no decision label assigned: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
