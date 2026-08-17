#!/usr/bin/env python3
"""Evaluate rank-8 learned/readout-SVD projectors and high-SF cross-transfer.

Only saved ConvGRU states and the frozen cached tiled readout are used.  This
script never imports or runs the image-to-ConvGRU model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank import evaluate_subspaces as evaluator
from paper.fig4.mechanism_audit_v1.causal_low_rank.common import CONTRAST_BY_KEY
from paper.fig4.mechanism_audit_v1.causal_low_rank.data import load_fold
from paper.fig4.mechanism_audit_v1.causal_low_rank.evaluate_subspaces import (
    EvaluationRequest,
    _evaluate_one,
    evaluate_basis,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.common import (
    CAUSAL_OUT,
    CONTINUATION_FOLDS,
    CONTRASTS,
    CROSS_TRANSFER_OUT,
    FOLDS,
    HIGH_CONTRASTS,
    RANK,
    RANK8_OUT,
    budget_deadline,
    cross_transfer_path,
    evaluation_directory,
    exclusive_gpu_lock,
    fit_directory,
    load_budget,
    preflight,
    record_gpu_time,
    validate_basis,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _coerce_csv_value(value: str) -> Any:
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _adopt_existing_fold0_cross_transfer(source: str, target: str) -> bool:
    """Adopt the already evaluated fold-0 transfer without another GPU pass."""
    destination = cross_transfer_path(source, target, 0)
    if destination.is_file():
        return True
    candidates = (
        CAUSAL_OUT / "evaluation/cross_contrast_results.csv",
        CAUSAL_OUT / "cross_contrast_results.csv",
    )
    for candidate in candidates:
        if not candidate.is_file() or candidate.stat().st_size == 0:
            continue
        with candidate.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if (
                    raw.get("source_contrast") == source
                    and raw.get("target_contrast") == target
                    and int(raw.get("fold", -1)) == 0
                    and int(raw.get("rank", -1)) == RANK
                ):
                    aggregate = {key: _coerce_csv_value(value) for key, value in raw.items()}
                    write_json(
                        destination,
                        {
                            "schema_version": "fig4-rank8-high-cross-transfer-v1",
                            "aggregate": aggregate,
                            "pair_metrics": None,
                            "basis_path": fit_directory(source, 0) / "U.npy",
                            "adopted_from": candidate,
                            "adoption_reason": (
                                "same saved fold-0 rank-8 source basis, held-out pairs, target "
                                "contrast, literal intervention, and exact tiled readout"
                            ),
                        },
                    )
                    return True
    return False


def planned_products() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for fold in FOLDS:
            for method in ("learned", "readout_svd"):
                rows.append(
                    {
                        "kind": "heldout_self",
                        "contrast": contrast,
                        "fold": fold,
                        "method": method,
                        "destination": evaluation_directory(contrast, fold) / f"{method}.json",
                    }
                )
    for fold in FOLDS:
        for source, target in (HIGH_CONTRASTS, tuple(reversed(HIGH_CONTRASTS))):
            rows.append(
                {
                    "kind": "high_cross_transfer",
                    "source": source,
                    "target": target,
                    "fold": fold,
                    "destination": cross_transfer_path(source, target, fold),
                }
            )
    return rows


def _evaluate_self(
    contrast_key: str,
    fold_index: int,
    method: str,
    args: argparse.Namespace,
) -> bool:
    fold = load_fold(fold_index)
    contrast = CONTRAST_BY_KEY[contrast_key]
    request = EvaluationRequest(
        "crossval",
        contrast_key,
        fold_index,
        RANK,
        method,
        contrast.scale_a,
        contrast.scale_b,
    )
    learned_predictions = fit_directory(contrast_key, fold_index) / "test_predictions.npz"
    force = bool(args.overwrite) or (method == "learned" and not learned_predictions.is_file())
    return _evaluate_one(
        request,
        fold.test.pairs,
        fold.train.pairs,
        str(args.device),
        int(args.frame_batch_size),
        force,
    )


def _evaluate_cross_transfer(
    source: str,
    target: str,
    fold_index: int,
    args: argparse.Namespace,
) -> bool:
    destination = cross_transfer_path(source, target, fold_index)
    if destination.is_file() and not args.overwrite:
        return False
    if fold_index == 0 and not args.overwrite:
        if _adopt_existing_fold0_cross_transfer(source, target):
            return False
    source_basis_path = fit_directory(source, fold_index) / "U.npy"
    basis = validate_basis(source_basis_path)
    target_contrast = CONTRAST_BY_KEY[target]
    fold = load_fold(fold_index)
    request = EvaluationRequest(
        "rank8_cross_transfer",
        target,
        fold_index,
        RANK,
        f"source_{source}",
        target_contrast.scale_a,
        target_contrast.scale_b,
    )
    aggregate, _, payload = evaluate_basis(
        request,
        fold.test.pairs,
        basis,
        str(args.device),
        int(args.frame_batch_size),
        save_predictions=False,
    )
    aggregate.update(
        {
            "source_contrast": source,
            "target_contrast": target,
            "source_rank": RANK,
        }
    )
    write_json(
        destination,
        {
            "schema_version": "fig4-rank8-high-cross-transfer-v1",
            "aggregate": aggregate,
            "pair_metrics": payload["pair_metrics"],
            "basis_path": source_basis_path,
            "adopted_from": None,
        },
    )
    return True


def _timed_gpu_call(stage: str, details: dict[str, Any], function) -> bool:
    deadline = budget_deadline()
    evaluator._GPU_DEADLINE_MONOTONIC = deadline
    started = time.monotonic()
    computed = False
    completed = False
    try:
        computed = bool(function())
        completed = True
        return computed
    finally:
        evaluator._GPU_DEADLINE_MONOTONIC = float("inf")
        if computed or not completed:
            record_gpu_time(
                stage,
                time.monotonic() - started,
                {**details, "completed": completed},
            )


def _self_product_complete(contrast: str, fold: int, method: str) -> bool:
    result = evaluation_directory(contrast, fold) / f"{method}.json"
    if not result.is_file():
        return False
    if method == "learned":
        return (fit_directory(contrast, fold) / "test_predictions.npz").is_file()
    return True


def run(args: argparse.Namespace) -> int:
    preflight()
    plan = planned_products()
    write_json(
        RANK8_OUT / "rank8_evaluation_plan.json",
        {
            "status": "dry_run" if args.dry_run else "started",
            "uses_saved_state_cache_only": True,
            "runs_image_to_convgru_model": False,
            "rank": RANK,
            "self_methods": ["learned", "readout_svd"],
            "fold0_existing_products_reused": True,
            "products": plan,
            "budget_before": load_budget(),
        },
    )
    if args.dry_run:
        for row in plan:
            print(f"{row['kind']} fold={row['fold']} -> {row['destination']}")
        return 0
    if not str(args.device).startswith("cuda"):
        raise ValueError("Production held-out evaluation must use an explicitly selected CUDA device")

    completed_rows: list[dict[str, Any]] = []
    if not args.overwrite:
        for source, target in (HIGH_CONTRASTS, tuple(reversed(HIGH_CONTRASTS))):
            _adopt_existing_fold0_cross_transfer(source, target)
    with exclusive_gpu_lock():
        # Finish one complete held-out fold before starting the next so any
        # ten-hour checkpoint leaves maximally interpretable saved products.
        for fold in FOLDS:
            for contrast in CONTRASTS:
                validate_basis(fit_directory(contrast, fold) / "U.npy")
                for method in ("learned", "readout_svd"):
                    details = {
                        "contrast": contrast,
                        "fold": fold,
                        "rank": RANK,
                        "method": method,
                    }
                    if not args.overwrite and _self_product_complete(
                        contrast, fold, method
                    ):
                        completed_rows.append({**details, "computed": False})
                        continue
                    computed = _timed_gpu_call(
                        "rank8_heldout_evaluation",
                        details,
                        lambda c=contrast, f=fold, m=method: _evaluate_self(c, f, m, args),
                    )
                    completed_rows.append({**details, "computed": computed})

            for source, target in (HIGH_CONTRASTS, tuple(reversed(HIGH_CONTRASTS))):
                details = {
                    "source_contrast": source,
                    "target_contrast": target,
                    "fold": fold,
                    "rank": RANK,
                }
                if not args.overwrite and cross_transfer_path(
                    source, target, fold
                ).is_file():
                    completed_rows.append({**details, "computed": False})
                    continue
                computed = _timed_gpu_call(
                    "rank8_high_cross_transfer",
                    details,
                    lambda s=source, t=target, f=fold: _evaluate_cross_transfer(s, t, f, args),
                )
                completed_rows.append({**details, "computed": computed})

    write_json(
        RANK8_OUT / "rank8_evaluation_run.json",
        {
            "status": "complete",
            "completed_products": completed_rows,
            "budget_after": load_budget(),
        },
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
