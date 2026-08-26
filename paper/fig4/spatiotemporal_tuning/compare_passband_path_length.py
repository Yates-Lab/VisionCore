#!/usr/bin/env python3
"""Compare passband engagement with path length on matched Figure-4 trials.

The comparison is deliberately within unit.  For each validated exact-CID
unit, it relates the 200 fixation-wise response modulations to (1) that unit's
motion-induced passband-power rank and (2) the same fixation's filtered path
length.  The paired population summary therefore asks whether tuning-matched
spectral engagement adds descriptive resolution beyond a scalar motion dose
without confounding stable differences between units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning._spectral_shards import (  # noqa: E402
    load_and_merge_shards,
)
from paper.fig4.spatiotemporal_tuning._figure4_renderer import (  # noqa: E402
    _direct_mechanism_values,
)
from paper.fig4.spatiotemporal_tuning._figure4_rendering import (  # noqa: E402
    _summary_checkpoint_digest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--trace-table", type=Path, required=True)
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument("--tuning-contract-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--population-policy",
        choices=("validated", "all_checkpoint_available"),
        default="validated",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def paired_unit_bootstrap(
    differences: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("too few finite units for paired bootstrap")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))
    distribution = np.median(values[draws], axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    population = load_and_merge_shards(args.population_shards)
    values = _direct_mechanism_values(population)
    unit_indices = np.asarray(population["unit_indices"], dtype=int)
    trace_indices = np.asarray(population["trace_indices"], dtype=int)

    tuning = pd.read_csv(args.tuning_summary)
    if args.population_policy == "all_checkpoint_available":
        selected = tuning.loc[
            tuning.included_in_exploratory_population.astype(bool), "unit_index"
        ].to_numpy(dtype=int)
    else:
        selected = tuning.loc[
            tuning.audit_category.eq("trusted")
            & tuning.validated_tuning.astype(bool),
            "unit_index",
        ].to_numpy(dtype=int)
    if not np.array_equal(np.sort(unit_indices), np.sort(selected)):
        raise ValueError("mechanism replay and selected tuning population differ")

    traces = pd.read_csv(args.trace_table)
    if traces.trace_bank_index.duplicated().any():
        raise ValueError("trace table contains duplicate trace-bank identities")
    indexed = traces.set_index("trace_bank_index")
    if not set(trace_indices).issubset(indexed.index):
        raise ValueError("mechanism replay references absent trace identities")
    path_length = indexed.loc[
        trace_indices, "rendered_path_length_arcmin"
    ].to_numpy(dtype=float)
    if not np.isfinite(path_length).all():
        raise ValueError("path-length comparison contains non-finite values")

    rows: list[dict[str, float | int | str]] = []
    passband = np.asarray(values["passband_percentile"], dtype=float)
    for outcome in ("rate_percent", "ssi_percent"):
        response = np.asarray(values[outcome], dtype=float)
        if response.shape != passband.shape:
            raise ValueError(f"{outcome} and passband arrays differ")
        for column, unit_index in enumerate(unit_indices):
            rows.append(
                {
                    "unit_index": int(unit_index),
                    "outcome": outcome,
                    "passband_spearman": float(
                        spearmanr(passband[:, column], response[:, column]).statistic
                    ),
                    "path_length_spearman": float(
                        spearmanr(path_length, response[:, column]).statistic
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table["passband_minus_path_spearman"] = (
        table.passband_spearman - table.path_length_spearman
    )
    table.to_csv(args.out_dir / "per_unit_comparison.csv", index=False)

    contract = json.loads(
        args.tuning_contract_summary.read_text(encoding="utf-8")
    )
    shard_summaries = [
        json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        for path in args.population_shards
    ]
    checkpoint = str(contract["checkpoint_sha256"])
    if any(_summary_checkpoint_digest(item) != checkpoint for item in shard_summaries):
        raise ValueError("tuning contract and mechanism replay checkpoints differ")

    outcomes: dict[str, object] = {}
    for index, outcome in enumerate(("rate_percent", "ssi_percent")):
        frame = table.loc[table.outcome.eq(outcome)]
        center, low, high = paired_unit_bootstrap(
            frame.passband_minus_path_spearman.to_numpy(dtype=float),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + index,
        )
        outcomes[outcome] = {
            "median_within_unit_passband_spearman": float(
                frame.passband_spearman.median()
            ),
            "median_within_unit_path_length_spearman": float(
                frame.path_length_spearman.median()
            ),
            "median_paired_difference": center,
            "paired_unit_bootstrap_ci95": [low, high],
            "n_units": int(len(frame)),
        }
    report = {
        "analysis": "within-unit passband-engagement versus path-length comparison",
        "checkpoint_sha256": checkpoint,
        "population": str(args.population_policy),
        "population_policy": str(args.population_policy),
        "n_units": int(len(unit_indices)),
        "n_traces": int(len(trace_indices)),
        "response_definition": (
            "measured-motion minus stabilized percent modulation, averaged across "
            "matched natural images"
        ),
        "passband_definition": (
            "within-unit rank of measured-minus-stabilized joint SFxTF passband power"
        ),
        "path_length_definition": "filtered rendered path length over the same 250-ms window",
        "inference": "paired bootstrap across exact units of the within-unit Spearman difference",
        "outcomes": outcomes,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
