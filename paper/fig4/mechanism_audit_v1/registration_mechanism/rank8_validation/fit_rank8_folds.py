#!/usr/bin/env python3
"""Continue only the prospectively fixed rank-8 fits on folds 1--3.

This entry point deliberately has no rank or contrast-selection option.  It
reuses the already materialized fold-0 fit and runs exactly the nine missing
fits required by the Overnight ConvGRU Mechanism Audit.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.causal_low_rank.common import CONTRAST_BY_KEY
from paper.fig4.mechanism_audit_v1.causal_low_rank.optimize_subspaces import (
    _TRAIN_RESIDENT_CACHE,
    GlobalGPUBudgetReached,
    load_configuration,
    train_one,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.rank8_validation.common import (
    CONTINUATION_FOLDS,
    CONTRASTS,
    HARD_LIMIT_HOURS,
    RANK,
    RANK8_OUT,
    budget_deadline,
    exclusive_gpu_lock,
    fit_directory,
    load_budget,
    preflight,
    record_gpu_time,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _optimizer_namespace(args: argparse.Namespace, deadline: float) -> argparse.Namespace:
    """Build exactly the unchanged optimizer settings expected by train_one."""
    return argparse.Namespace(
        stage="crossval",
        device=str(args.device),
        contrasts=list(CONTRASTS),
        folds=list(CONTINUATION_FOLDS),
        ranks=[RANK],
        max_steps=None,
        frame_batch_size=None,
        validation_interval=None,
        patience_evaluations=None,
        learning_rate=None,
        overwrite=bool(args.overwrite),
        dry_run=False,
        _gpu_deadline_monotonic=float(deadline),
    )


def planned_fits() -> list[dict[str, object]]:
    return [
        {
            "contrast": contrast,
            "fold": fold,
            "rank": RANK,
            "destination": fit_directory(contrast, fold),
        }
        for fold in CONTINUATION_FOLDS
        for contrast in CONTRASTS
    ]


def run(args: argparse.Namespace) -> int:
    provenance = preflight()
    plan = planned_fits()
    budget = load_budget()
    plan_product = {
        "status": "dry_run" if args.dry_run else "started",
        "rank_fixed_prospectively": RANK,
        "rank_selection": provenance["stage2"]["elbow"],
        "fold0_action": "reuse_existing_validated_screening-fold fit",
        "continuation_folds": list(CONTINUATION_FOLDS),
        "contrasts": list(CONTRASTS),
        "three_predeclared_initializations": True,
        "optimizer_overrides": None,
        "budget_before": budget,
        "hard_limit_hours_including_prior_work": HARD_LIMIT_HOURS,
        "fits": plan,
    }
    write_json(RANK8_OUT / "rank8_fit_plan.json", plan_product)
    if args.dry_run:
        for row in plan:
            print(
                f"rank8 continuation {row['contrast']} fold={row['fold']} -> {row['destination']}"
            )
        return 0

    if not str(args.device).startswith("cuda"):
        raise ValueError("Production rank-8 fitting must use an explicitly selected CUDA device")
    config = load_configuration()
    summaries: list[dict[str, object]] = []

    def save_progress(status: str) -> None:
        write_json(
            RANK8_OUT / "rank8_fit_run.json",
            {
                "status": status,
                "rank": RANK,
                "fold0_reused": True,
                "summaries": summaries,
                "budget_after": load_budget(),
            },
        )

    with exclusive_gpu_lock():
        contrast_ordinals = {key: ordinal for ordinal, key in enumerate(CONTRASTS)}
        # Complete every contrast for one held-out fold before moving on.  If
        # the ten-hour reporting boundary is reached, this maximizes the
        # number of scientifically interpretable complete crossed folds.
        for fold in CONTINUATION_FOLDS:
            for contrast_key in CONTRASTS:
                contrast_ordinal = contrast_ordinals[contrast_key]
                contrast = CONTRAST_BY_KEY[contrast_key]
                current_budget = load_budget()
                if float(current_budget["remaining_gpu_hours"]) <= 0.0:
                    raise RuntimeError(
                        "Ten cumulative GPU-hours reached before the next rank-8 fit; "
                        "write the required interim report"
                    )
                destination = fit_directory(contrast_key, fold)
                existing = all(
                    (destination / name).is_file()
                    for name in ("U.npy", "P.npy", "training_curve.csv", "fit_metadata.json")
                )
                if existing and not args.overwrite:
                    summaries.append(
                        {
                            "status": "exists",
                            "contrast": contrast_key,
                            "fold": fold,
                            "rank": RANK,
                            "path": destination,
                        }
                    )
                    save_progress("in_progress")
                    continue

                started = time.monotonic()
                completed = False
                progress_status = "in_progress"
                optimizer_args = _optimizer_namespace(args, budget_deadline())
                try:
                    summary = train_one(
                        stage="crossval",
                        contrast=contrast,
                        contrast_ordinal=contrast_ordinal,
                        fold_index=fold,
                        rank=RANK,
                        config=config,
                        args=optimizer_args,
                    )
                    completed = True
                    summaries.append(summary)
                except GlobalGPUBudgetReached as error:
                    progress_status = "budget_reached"
                    summaries.append(
                        {
                            "status": "budget_reached",
                            "contrast": contrast_key,
                            "fold": fold,
                            "rank": RANK,
                            "error": str(error),
                        }
                    )
                    raise
                except Exception as error:
                    progress_status = "interrupted"
                    summaries.append(
                        {
                            "status": "interrupted",
                            "contrast": contrast_key,
                            "fold": fold,
                            "rank": RANK,
                            "error": repr(error),
                        }
                    )
                    raise
                finally:
                    elapsed = time.monotonic() - started
                    record_gpu_time(
                        "rank8_fit",
                        elapsed,
                        {
                            "contrast": contrast_key,
                            "fold": fold,
                            "rank": RANK,
                            "completed": completed,
                        },
                    )
                    # Resident arrays are intentionally cached across ranks by
                    # the original sweep.  This continuation has one rank per
                    # split, so retaining ~20 GiB per fold would only risk OOM.
                    _TRAIN_RESIDENT_CACHE.clear()
                    gc.collect()
                    save_progress(progress_status)

    save_progress("complete")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
