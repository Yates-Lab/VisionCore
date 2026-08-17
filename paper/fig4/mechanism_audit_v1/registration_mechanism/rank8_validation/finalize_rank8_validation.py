#!/usr/bin/env python3
"""Consolidate saved rank-8 validation products and gate consensus projectors.

This is a CPU-only saved-product stage.  Consensus bases are eigenvectors of
the mean projector, never averages of the fold-specific basis vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    CONTRAST_BY_KEY,
    principal_angles_deg,
    projector,
    sha256_file,
    subspace_overlap,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold
from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.common import (
    CAUSAL_OUT,
    CONTRASTS,
    FOLDS,
    HIGH_CONTRASTS,
    OUT,
    RANK,
    RANK8_OUT,
    cross_transfer_path,
    evaluation_directory,
    fit_directory,
    load_budget,
    preflight,
    validate_basis,
    write_json,
)


NULL_DRAWS = 20_000
NULL_SEED = 20261731
GENERALIZATION_MIN_MAP_R2 = 0.40
GENERALIZATION_REQUIRED_FOLDS = 3
SHARED_HIGH_MIN_MEDIAN_OVERLAP = 0.50
SHARED_HIGH_MIN_CROSS_TRANSFER_R2 = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON product: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if columns:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def required_products() -> list[Path]:
    paths: list[Path] = []
    for contrast in CONTRASTS:
        for fold in FOLDS:
            fit = fit_directory(contrast, fold)
            paths.extend(
                [
                    fit / "U.npy",
                    fit / "P.npy",
                    fit / "fit_metadata.json",
                    fit / "test_predictions.npz",
                    evaluation_directory(contrast, fold) / "learned.json",
                    evaluation_directory(contrast, fold) / "readout_svd.json",
                ]
            )
    for fold in FOLDS:
        for source, target in (HIGH_CONTRASTS, tuple(reversed(HIGH_CONTRASTS))):
            paths.append(cross_transfer_path(source, target, fold))
    return paths


def readiness() -> dict[str, Any]:
    paths = required_products()
    missing = [str(path) for path in paths if not path.is_file()]
    return {
        "status": "complete" if not missing else "incomplete",
        "rank": RANK,
        "folds": list(FOLDS),
        "contrasts": list(CONTRASTS),
        "required_product_count": len(paths),
        "present_product_count": len(paths) - len(missing),
        "missing": missing,
    }


def _load_aggregate(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(path)
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise RuntimeError(f"Missing aggregate result at {path}")
    for key, expected_value in expected.items():
        observed = aggregate.get(key)
        if isinstance(expected_value, int):
            match = int(observed) == expected_value
        else:
            match = str(observed) == str(expected_value)
        if not match:
            raise RuntimeError(
                f"Result provenance mismatch at {path}: {key}={observed!r}, "
                f"expected {expected_value!r}"
            )
    for metric in ("map_r2_sufficiency", "map_r2_necessity"):
        if not np.isfinite(float(aggregate.get(metric, np.nan))):
            raise RuntimeError(f"Nonfinite primary metric {metric} at {path}")
    return aggregate


def load_fold_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrast_key in CONTRASTS:
        contrast = CONTRAST_BY_KEY[contrast_key]
        for fold_index in FOLDS:
            method_results: dict[str, dict[str, Any]] = {}
            for method in ("learned", "readout_svd"):
                path = evaluation_directory(contrast_key, fold_index) / f"{method}.json"
                aggregate = _load_aggregate(
                    path,
                    {
                        "stage": "crossval",
                        "contrast": contrast_key,
                        "fold": fold_index,
                        "rank": RANK,
                        "method": method,
                    },
                )
                method_results[method] = aggregate
            learned_mean = float(
                np.mean(
                    [
                        method_results["learned"]["map_r2_sufficiency"],
                        method_results["learned"]["map_r2_necessity"],
                    ]
                )
            )
            readout_mean = float(
                np.mean(
                    [
                        method_results["readout_svd"]["map_r2_sufficiency"],
                        method_results["readout_svd"]["map_r2_necessity"],
                    ]
                )
            )
            fold = load_fold(fold_index)
            for method, aggregate in method_results.items():
                primary_mean = float(
                    np.mean(
                        [
                            aggregate["map_r2_sufficiency"],
                            aggregate["map_r2_necessity"],
                        ]
                    )
                )
                rows.append(
                    {
                        "contrast": contrast_key,
                        "contrast_label": contrast.label,
                        "target_group": contrast.group,
                        "fold": fold_index,
                        "rank": RANK,
                        "method": method,
                        "test_image_positions": ";".join(map(str, fold.test.image_positions)),
                        "test_trajectory_positions": ";".join(
                            map(str, fold.test.trajectory_positions)
                        ),
                        "n_test_pairs": int(aggregate["n_test_pairs"]),
                        "n_test_records": int(aggregate["n_test_records"]),
                        "map_r2_sufficiency": float(aggregate["map_r2_sufficiency"]),
                        "map_r2_necessity": float(aggregate["map_r2_necessity"]),
                        "complete_map_recovery_mean": primary_mean,
                        "preactivation_r2_sufficiency": float(
                            aggregate["preactivation_r2_sufficiency"]
                        ),
                        "preactivation_r2_necessity": float(
                            aggregate["preactivation_r2_necessity"]
                        ),
                        "ssi_fraction_transferred": float(
                            aggregate["ssi_fraction_transferred"]
                        ),
                        "ssi_fraction_removed": float(aggregate["ssi_fraction_removed"]),
                        "learned_minus_readout_complete_map_mean": (
                            learned_mean - readout_mean if method == "learned" else np.nan
                        ),
                        "learned_outperforms_readout_this_fold": (
                            learned_mean > readout_mean if method == "learned" else ""
                        ),
                        "basis_path": (
                            fit_directory(contrast_key, fold_index) / "U.npy"
                            if method == "learned"
                            else "frozen target-population readout SVD"
                        ),
                        "heldout_prediction_path": (
                            fit_directory(contrast_key, fold_index) / "test_predictions.npz"
                            if method == "learned"
                            else ""
                        ),
                    }
                )
    return rows


def random_overlap_distribution(draws: int, seed: int = NULL_SEED) -> np.ndarray:
    """Monte Carlo null for tr(P1 P2)/8 using rotational invariance."""
    if int(draws) < 100:
        raise ValueError("At least 100 random-overlap draws are required")
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        candidate, _ = np.linalg.qr(rng.standard_normal((128, RANK)), mode="reduced")
        # The first projector can be fixed to the first eight coordinate axes.
        values[draw] = float(np.square(candidate[:RANK]).sum() / RANK)
    return values


def _load_projectors() -> tuple[dict[tuple[str, int], np.ndarray], list[dict[str, Any]]]:
    bases: dict[tuple[str, int], np.ndarray] = {}
    inventory: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for fold in FOLDS:
            directory = fit_directory(contrast, fold)
            basis_path = directory / "U.npy"
            projector_path = directory / "P.npy"
            basis = validate_basis(basis_path)
            saved_projector = np.asarray(np.load(projector_path), dtype=np.float64)
            expected = projector(basis).astype(np.float64)
            if saved_projector.shape != (128, 128) or not np.allclose(
                saved_projector, expected, atol=3e-5, rtol=3e-5
            ):
                raise RuntimeError(f"Saved projector does not equal U U^T: {projector_path}")
            bases[(contrast, fold)] = basis
            inventory.append(
                {
                    "contrast": contrast,
                    "fold": fold,
                    "rank": RANK,
                    "basis_path": basis_path,
                    "basis_sha256": sha256_file(basis_path),
                    "projector_path": projector_path,
                    "projector_sha256": sha256_file(projector_path),
                    "heldout_predictions_path": directory / "test_predictions.npz",
                    "heldout_predictions_sha256": sha256_file(
                        directory / "test_predictions.npz"
                    ),
                }
            )
    return bases, inventory


def stability_rows(
    bases: dict[tuple[str, int], np.ndarray],
    null: np.ndarray,
) -> list[dict[str, Any]]:
    null_low, null_high = np.percentile(null, [2.5, 97.5])
    common = {
        "rank": RANK,
        "overlap_definition": "tr(P_a P_b) / 8",
        "random_overlap_mean": float(np.mean(null)),
        "random_overlap_ci_low": float(null_low),
        "random_overlap_ci_high": float(null_high),
        "analytic_random_expectation": RANK / 128.0,
    }
    rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for fold_a, fold_b in combinations(FOLDS, 2):
            first = bases[(contrast, fold_a)]
            second = bases[(contrast, fold_b)]
            angles = principal_angles_deg(first, second)
            overlap = subspace_overlap(first, second)
            rows.append(
                {
                    "comparison_type": "within_contrast_across_folds",
                    "contrast_a": contrast,
                    "contrast_b": contrast,
                    "fold_a": fold_a,
                    "fold_b": fold_b,
                    "projector_overlap": overlap,
                    "overlap_above_random_97_5": overlap > null_high,
                    "principal_angle_mean_deg": float(np.mean(angles)),
                    "principal_angle_median_deg": float(np.median(angles)),
                    "principal_angle_max_deg": float(np.max(angles)),
                    "principal_angles_deg": ";".join(f"{value:.8g}" for value in angles),
                    **common,
                }
            )
    for fold in FOLDS:
        first = bases[(HIGH_CONTRASTS[0], fold)]
        second = bases[(HIGH_CONTRASTS[1], fold)]
        angles = principal_angles_deg(first, second)
        overlap = subspace_overlap(first, second)
        rows.append(
            {
                "comparison_type": "higher_sf_between_contrasts_within_fold",
                "contrast_a": HIGH_CONTRASTS[0],
                "contrast_b": HIGH_CONTRASTS[1],
                "fold_a": fold,
                "fold_b": fold,
                "projector_overlap": overlap,
                "overlap_above_random_97_5": overlap > null_high,
                "principal_angle_mean_deg": float(np.mean(angles)),
                "principal_angle_median_deg": float(np.median(angles)),
                "principal_angle_max_deg": float(np.max(angles)),
                "principal_angles_deg": ";".join(f"{value:.8g}" for value in angles),
                **common,
            }
        )
    return rows


def load_cross_transfer_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for source, target in (HIGH_CONTRASTS, tuple(reversed(HIGH_CONTRASTS))):
            path = cross_transfer_path(source, target, fold)
            payload = _read_json(path)
            aggregate = payload.get("aggregate")
            if not isinstance(aggregate, dict):
                raise RuntimeError(f"Missing cross-transfer aggregate: {path}")
            for metric in ("map_r2_sufficiency", "map_r2_necessity"):
                if not np.isfinite(float(aggregate.get(metric, np.nan))):
                    raise RuntimeError(f"Nonfinite {metric} in {path}")
            rows.append(
                {
                    "source_contrast": source,
                    "target_contrast": target,
                    "fold": fold,
                    "rank": RANK,
                    "map_r2_sufficiency": float(aggregate["map_r2_sufficiency"]),
                    "map_r2_necessity": float(aggregate["map_r2_necessity"]),
                    "complete_map_recovery_mean": float(
                        np.mean(
                            [
                                aggregate["map_r2_sufficiency"],
                                aggregate["map_r2_necessity"],
                            ]
                        )
                    ),
                    "preactivation_r2_sufficiency": float(
                        aggregate["preactivation_r2_sufficiency"]
                    ),
                    "preactivation_r2_necessity": float(
                        aggregate["preactivation_r2_necessity"]
                    ),
                    "ssi_fraction_transferred": float(
                        aggregate["ssi_fraction_transferred"]
                    ),
                    "ssi_fraction_removed": float(aggregate["ssi_fraction_removed"]),
                    "product_path": path,
                    "adopted_from_prior_fold0_evaluation": payload.get("adopted_from")
                    is not None,
                }
            )
    return rows


def fold_stability_gate(
    rows: Iterable[dict[str, Any]], contrast: str, null_high: float
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["comparison_type"] == "within_contrast_across_folds"
        and row["contrast_a"] == contrast
    ]
    overlaps = np.asarray([row["projector_overlap"] for row in selected], dtype=float)
    complete = len(selected) == 6
    passes = bool(complete and np.all(overlaps > float(null_high)))
    return {
        "complete": complete,
        "n_fold_pairs": len(selected),
        "criterion": "all six pairwise overlaps exceed the rank-matched random 97.5th percentile",
        "random_97_5_threshold": float(null_high),
        "minimum_observed_overlap": float(np.min(overlaps)) if len(overlaps) else np.nan,
        "median_observed_overlap": float(np.median(overlaps)) if len(overlaps) else np.nan,
        "clearly_above_random": passes,
    }


def generalization_gate(fold_rows: list[dict[str, Any]], contrast: str) -> dict[str, Any]:
    selected = [
        row
        for row in fold_rows
        if row["contrast"] == contrast and row["method"] == "learned"
    ]
    suff = np.asarray([row["map_r2_sufficiency"] for row in selected], dtype=float)
    nec = np.asarray([row["map_r2_necessity"] for row in selected], dtype=float)
    fold_pass = np.minimum(suff, nec) >= GENERALIZATION_MIN_MAP_R2
    complete = len(selected) == len(FOLDS)
    passes = bool(
        complete
        and int(fold_pass.sum()) >= GENERALIZATION_REQUIRED_FOLDS
        and np.median(suff) >= GENERALIZATION_MIN_MAP_R2
        and np.median(nec) >= GENERALIZATION_MIN_MAP_R2
    )
    readout_deltas = np.asarray(
        [row["learned_minus_readout_complete_map_mean"] for row in selected],
        dtype=float,
    )
    readout_wins = int(np.sum(readout_deltas > 0.0))
    consistent_advantage = bool(
        complete and readout_wins >= 3 and float(np.median(readout_deltas)) > 0.0
    )
    return {
        "complete": complete,
        "criterion": (
            "both held-out complete-map sufficiency and necessity R2 >= 0.40 in at least "
            "three of four folds, with both fold medians >= 0.40"
        ),
        "folds_passing": int(fold_pass.sum()),
        "fold_pass": fold_pass.tolist(),
        "median_map_r2_sufficiency": float(np.median(suff)) if len(suff) else np.nan,
        "median_map_r2_necessity": float(np.median(nec)) if len(nec) else np.nan,
        "generalizes": passes,
        "learned_minus_readout_median": float(np.median(readout_deltas))
        if len(readout_deltas)
        else np.nan,
        "learned_outperforms_readout_fold_count": readout_wins,
        "learned_outperforms_readout_consistently": consistent_advantage,
        "recommended_term": (
            "candidate movement subspace (movement specificity remains to be tested)"
            if consistent_advantage
            else "compact output-relevant subspace"
        ),
        "allowed_term_if_readout_not_consistently_outperformed": "compact output-relevant subspace",
    }


def shared_high_gate(
    stability: list[dict[str, Any]],
    cross_transfer: list[dict[str, Any]],
    per_contrast_stability: dict[str, dict[str, Any]],
    null_high: float,
) -> dict[str, Any]:
    overlap_rows = [
        row
        for row in stability
        if row["comparison_type"] == "higher_sf_between_contrasts_within_fold"
    ]
    overlaps = np.asarray([row["projector_overlap"] for row in overlap_rows], dtype=float)
    transfer_minima = np.asarray(
        [
            min(row["map_r2_sufficiency"], row["map_r2_necessity"])
            for row in cross_transfer
        ],
        dtype=float,
    )
    overlap_complete = len(overlap_rows) == len(FOLDS)
    transfer_complete = len(cross_transfer) == 2 * len(FOLDS)
    strong_overlap = bool(
        overlap_complete
        and np.all(overlaps > float(null_high))
        and float(np.median(overlaps)) >= SHARED_HIGH_MIN_MEDIAN_OVERLAP
    )
    transfer_replicates = bool(
        transfer_complete
        and np.all(transfer_minima >= SHARED_HIGH_MIN_CROSS_TRANSFER_R2)
    )
    separate_stable = all(
        per_contrast_stability[key]["clearly_above_random"] for key in HIGH_CONTRASTS
    )
    allowed = bool(strong_overlap and transfer_replicates and separate_stable)
    return {
        "allowed": allowed,
        "criterion": (
            "both high-SF projectors stable across folds; all four within-fold high-SF "
            "overlaps exceed the random 97.5th percentile; median overlap >= 0.50; "
            "and both sufficiency and necessity complete-map R2 >= 0.40 for both transfer "
            "directions in every fold"
        ),
        "overlap_complete": overlap_complete,
        "transfer_complete": transfer_complete,
        "separate_high_projectors_stable": separate_stable,
        "all_high_overlaps_above_random_97_5": bool(
            overlap_complete and np.all(overlaps > float(null_high))
        ),
        "median_high_overlap": float(np.median(overlaps)) if len(overlaps) else np.nan,
        "strong_overlap": strong_overlap,
        "minimum_cross_transfer_r2": float(np.min(transfer_minima))
        if len(transfer_minima)
        else np.nan,
        "cross_transfer_replicates": transfer_replicates,
        "failure_action": "retain separate high-SF sharpening and reversal projectors",
    }


def consensus_from_projectors(projectors: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [np.asarray(value, dtype=np.float64) for value in projectors]
    if not values:
        raise ValueError("At least one projector is required")
    for value in values:
        if value.shape != (128, 128):
            raise ValueError(f"Projector shape must be 128x128, got {value.shape}")
    mean_projector = np.mean(values, axis=0)
    mean_projector = 0.5 * (mean_projector + mean_projector.T)
    eigenvalues, eigenvectors = np.linalg.eigh(mean_projector)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:RANK]]
    # Canonical signs make the visualization product byte-stable; P is sign-invariant.
    for column in range(RANK):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
    basis = basis.astype(np.float32)
    consensus_projector = projector(basis)
    return basis, consensus_projector, eigenvalues[order].astype(np.float32)


def _save_consensus(
    bases: dict[tuple[str, int], np.ndarray],
    stability_gates: dict[str, dict[str, Any]],
    high_gate: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "schema_version": "fig4-registration-consensus-projectors-v1",
        "rank": RANK,
        "construction": (
            "top eight eigenvectors of the arithmetic mean of fold projectors; "
            "raw bases are never averaged"
        ),
        "visualization_only": True,
        "foldwise_projectors_required_for_semantic_inference": True,
        "contrasts": {},
    }
    for contrast in CONTRASTS:
        allowed = bool(stability_gates[contrast]["clearly_above_random"])
        entry: dict[str, Any] = {"saved": allowed, "gate": stability_gates[contrast]}
        if allowed:
            projectors = [projector(bases[(contrast, fold)]) for fold in FOLDS]
            u, p, eigenvalues = consensus_from_projectors(projectors)
            arrays[f"U__{contrast}"] = u
            arrays[f"P__{contrast}"] = p
            arrays[f"mean_projector_eigenvalues__{contrast}"] = eigenvalues
            entry["source_folds"] = list(FOLDS)
        metadata["contrasts"][contrast] = entry
    metadata["shared_higher_sf"] = {"saved": bool(high_gate["allowed"]), "gate": high_gate}
    if high_gate["allowed"]:
        high_projectors = [
            projector(bases[(contrast, fold)])
            for contrast in HIGH_CONTRASTS
            for fold in FOLDS
        ]
        u, p, eigenvalues = consensus_from_projectors(high_projectors)
        arrays["U__shared_higher_sf"] = u
        arrays["P__shared_higher_sf"] = p
        arrays["mean_projector_eigenvalues__shared_higher_sf"] = eigenvalues
        metadata["shared_higher_sf"]["source_contrasts"] = list(HIGH_CONTRASTS)
        metadata["shared_higher_sf"]["source_folds"] = list(FOLDS)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination = OUT / "consensus_projectors.npz"
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    return arrays, metadata


def run(args: argparse.Namespace) -> int:
    provenance = preflight()
    ready = readiness()
    write_json(RANK8_OUT / "readiness.json", ready)
    if ready["status"] != "complete":
        if args.allow_incomplete:
            print(
                f"rank-8 validation incomplete: {ready['present_product_count']}/"
                f"{ready['required_product_count']} required products"
            )
            return 0
        raise RuntimeError(
            f"Rank-8 validation products are incomplete; see {RANK8_OUT / 'readiness.json'}"
        )

    fold_rows = load_fold_results()
    bases, inventory = _load_projectors()
    null = random_overlap_distribution(int(args.null_draws))
    null_low, null_high = np.percentile(null, [2.5, 97.5])
    stability = stability_rows(bases, null)
    cross_transfer = load_cross_transfer_results()
    stability_gates = {
        contrast: fold_stability_gate(stability, contrast, float(null_high))
        for contrast in CONTRASTS
    }
    generalization_gates = {
        contrast: generalization_gate(fold_rows, contrast) for contrast in CONTRASTS
    }
    high_gate = shared_high_gate(
        stability, cross_transfer, stability_gates, float(null_high)
    )
    _, consensus_metadata = _save_consensus(bases, stability_gates, high_gate)

    generalizes = all(value["generalizes"] for value in generalization_gates.values())
    gate = {
        "status": "complete",
        "rank_fixed_prospectively": RANK,
        "rank_selection_used_test_metrics": False,
        "heldout_generalization": generalization_gates,
        "all_three_contrasts_generalize": generalizes,
        "fold_projector_stability": stability_gates,
        "shared_higher_sf_consensus": high_gate,
        "stop_downstream_mechanism_audit": not generalizes,
        "stop_reason": (
            None
            if generalizes
            else "rank-8 complete-map effects failed the predeclared crossed-fold recovery gate"
        ),
        "terminology_boundary": (
            "If learned rank-8 does not outperform readout-SVD consistently, use "
            "'compact output-relevant subspace', not 'movement-specific subspace'. "
            "Even consistent readout-SVD superiority does not establish movement specificity; "
            "that requires the separate fold-wise P/Q motion-enrichment analysis."
        ),
        "random_overlap_null": {
            "draws": int(args.null_draws),
            "seed": NULL_SEED,
            "mean": float(np.mean(null)),
            "ci_low": float(null_low),
            "ci_high": float(null_high),
            "analytic_expectation": RANK / 128.0,
        },
        "consensus_metadata": consensus_metadata,
    }

    _write_csv(OUT / "fold_rank8_results.csv", fold_rows)
    _write_csv(RANK8_OUT / "fold_rank8_results.csv", fold_rows)
    _write_csv(OUT / "projector_stability.csv", stability)
    _write_csv(RANK8_OUT / "projector_stability.csv", stability)
    _write_csv(RANK8_OUT / "higher_sf_cross_transfer.csv", cross_transfer)
    write_json(OUT / "rank8_validation_gate.json", gate)
    write_json(RANK8_OUT / "rank8_validation_gate.json", gate)
    write_json(
        RANK8_OUT / "rank8_projector_inventory.json",
        {
            "schema_version": "fig4-registration-rank8-projector-inventory-v1",
            "rank": RANK,
            "fold_projectors": inventory,
            "consensus_path": OUT / "consensus_projectors.npz",
            "consensus_for_visualization_only": True,
        },
    )
    write_json(
        RANK8_OUT / "analysis_manifest.json",
        {
            "schema_version": "fig4-registration-rank8-validation-v1",
            "completed_at_unix": time.time(),
            "rank": RANK,
            "folds": list(FOLDS),
            "contrasts": list(CONTRASTS),
            "source_causal_analysis": CAUSAL_OUT,
            "screening_rank_selection": provenance["stage2"]["elbow"],
            "integrity_gate": CAUSAL_OUT / "integrity_tests.json",
            "uses_saved_cache_only_for_evaluation": True,
            "consensus_is_visualization_only": True,
            "outputs": {
                "fold_results": OUT / "fold_rank8_results.csv",
                "stability": OUT / "projector_stability.csv",
                "consensus": OUT / "consensus_projectors.npz",
                "gate": OUT / "rank8_validation_gate.json",
                "inventory": RANK8_OUT / "rank8_projector_inventory.json",
                "cross_transfer": RANK8_OUT / "higher_sf_cross_transfer.csv",
            },
            "gpu_budget": load_budget(),
        },
    )
    print(f"rank-8 validation products written under {RANK8_OUT}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
