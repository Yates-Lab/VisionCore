#!/usr/bin/env python3
"""Audit Figure 3's unified intact/ablation inference cache.

The audit is intentionally independent of plotting.  It proves that every
counterfactual uses the selected intact trace's exact neuron order and
data-only reliability ceiling, verifies ``CCnorm = CCabs / CCmax`` on the
stable population, and (for schema v7+) proves that the observed FEM
decomposition is invariant across model conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import dill
import numpy as np


CONDITIONS = ("intact", "zeroed", "stabilized")


def _exact(left: Any, right: Any) -> bool:
    return bool(
        np.array_equal(
            np.asarray(left),
            np.asarray(right),
            equal_nan=True,
        )
    )


def audit_ablation_payload(
    payload: dict[str, Any],
    intact_trace: list[dict[str, Any]],
    *,
    require_femfraction: bool = True,
    expected_femfraction_count_bins: int | None = 3,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Ablation cache must be a dict containing a results list")
    if payload.get("complete") is False:
        raise ValueError("Ablation cache is a resumable partial, not a completed cache")
    schema = int(payload.get("schema_version", 0))
    if require_femfraction and schema < 7:
        raise ValueError(f"FEM-fraction audit requires schema >=7, found {schema}")
    if (
        schema >= 7
        and expected_femfraction_count_bins is not None
        and int(payload.get("femfraction_count_bins", -1))
        != int(expected_femfraction_count_bins)
    ):
        raise ValueError(
            "FEM-fraction counting window does not match the Figure-2 contract: "
            f"expected {expected_femfraction_count_bins}, found "
            f"{payload.get('femfraction_count_bins')}"
        )
    intact_by_session = {str(record["session"]): record for record in intact_trace}
    if len(intact_by_session) != len(intact_trace):
        raise ValueError("Intact trace contains duplicate sessions")

    payload_names = [str(record["session"]) for record in payload["results"]]
    if len(set(payload_names)) != len(payload_names):
        raise ValueError("Ablation cache contains duplicate sessions")

    identity_max = 0.0
    session_rows: list[dict[str, Any]] = []
    for record in payload["results"]:
        session = str(record["session"])
        if session not in intact_by_session:
            raise ValueError(f"Ablation session {session} is absent from the intact trace")
        anchor = intact_by_session[session]
        neuron_mask = np.asarray(record["neuron_mask"])
        n_units = int(neuron_mask.size)
        neuron_exact = _exact(record["neuron_mask"], anchor["neuron_mask"])
        ccmax_exact = _exact(record["ccmax"], anchor["ccmax"])
        unstable_exact = _exact(record["ccnorm_unstable"], anchor["ccnorm_unstable"])
        intact_ccabs_exact = _exact(record["ccabs"]["intact"], anchor["ccabs"])
        intact_ccnorm_exact = _exact(record["ccnorm"]["intact"], anchor["ccnorm"])
        if not all(
            (
                neuron_exact,
                ccmax_exact,
                unstable_exact,
                intact_ccabs_exact,
                intact_ccnorm_exact,
            )
        ):
            raise AssertionError(
                f"{session}: ablation cache does not share the exact intact metric anchor"
            )

        ccmax = np.asarray(record["ccmax"], dtype=np.float64)
        unstable = np.asarray(record["ccnorm_unstable"], dtype=bool)
        if ccmax.shape != (n_units,) or unstable.shape != (n_units,):
            raise ValueError(
                f"{session}: CCmax/stability shapes do not match the neuron population"
            )
        for condition in CONDITIONS:
            ccabs = np.asarray(record["ccabs"][condition], dtype=np.float64)
            ccnorm = np.asarray(record["ccnorm"][condition], dtype=np.float64)
            if ccabs.shape != (n_units,) or ccnorm.shape != (n_units,):
                raise ValueError(
                    f"{session}/{condition}: CC metric shapes do not match the neuron population"
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                identity = ccabs / ccmax
            identity[unstable] = np.nan
            finite = np.isfinite(identity) & np.isfinite(ccnorm)
            if np.any(finite):
                identity_max = max(
                    identity_max,
                    float(np.max(np.abs(identity[finite] - ccnorm[finite]))),
                )
            if not _exact(np.isnan(identity), np.isnan(ccnorm)) or not np.allclose(
                identity,
                ccnorm,
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(f"{session}/{condition}: CCnorm identity failed")

        fem_observed_exact = None
        if schema >= 7:
            fem = record.get("femfraction")
            if not isinstance(fem, dict) or any(condition not in fem for condition in CONDITIONS):
                raise ValueError(f"{session}: schema-v7 cache lacks complete FEM fractions")
            fem_observed_exact = True
            for condition in CONDITIONS:
                for key in ("B_obs", "B_obs_uncl", "B_model", "B_model_uncl"):
                    if key not in fem[condition]:
                        raise ValueError(
                            f"{session}/{condition}: schema-v7 cache lacks {key}"
                        )
                    if np.asarray(fem[condition][key]).shape != (n_units,):
                        raise ValueError(
                            f"{session}/{condition}/{key}: FEM vector does not match the neuron population"
                        )
            for key in ("B_obs", "B_obs_uncl"):
                anchor_observed = np.asarray(fem["intact"][key])
                for condition in CONDITIONS[1:]:
                    observed = np.asarray(fem[condition][key])
                    fem_observed_exact &= _exact(anchor_observed, observed)
            if not fem_observed_exact:
                raise AssertionError(
                    f"{session}: data-only FEM decomposition changed across model conditions"
                )

        session_rows.append(
            {
                "session": session,
                "n_units": n_units,
                "neuron_order_exact": neuron_exact,
                "ccmax_exact": ccmax_exact,
                "stability_mask_exact": unstable_exact,
                "intact_ccabs_exact": intact_ccabs_exact,
                "intact_ccnorm_exact": intact_ccnorm_exact,
                "fem_observed_exact_across_conditions": fem_observed_exact,
            }
        )

    payload_sessions = {row["session"] for row in session_rows}
    intact_sessions = set(intact_by_session)
    if payload_sessions != intact_sessions:
        missing = sorted(intact_sessions - payload_sessions)
        extra = sorted(payload_sessions - intact_sessions)
        raise AssertionError(
            f"Ablation/intact session sets differ; missing={missing}, extra={extra}"
        )
    report = {
        "analysis": "figure3_ablation_metric_anchor_audit",
        "schema_version": schema,
        "n_sessions": len(session_rows),
        "n_units": int(sum(row["n_units"] for row in session_rows)),
        "shared_anchor_exact": True,
        "ccnorm_identity_max_abs_error": identity_max,
        "fem_observed_exact_across_conditions": (
            all(row["fem_observed_exact_across_conditions"] for row in session_rows)
            if schema >= 7
            else None
        ),
        "femfraction_count_bins": payload.get("femfraction_count_bins"),
        "sessions": session_rows,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation_cache", type=Path)
    parser.add_argument("intact_trace_cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-pre-v7",
        action="store_true",
        help="Audit only CCnorm/anchor fields when FEM fractions are not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.ablation_cache.open("rb") as handle:
        payload = dill.load(handle)
    with args.intact_trace_cache.open("rb") as handle:
        intact_trace = dill.load(handle)
    report = audit_ablation_payload(
        payload,
        intact_trace,
        require_femfraction=not args.allow_pre_v7,
    )
    report.update(
        {
            "ablation_cache": str(args.ablation_cache.resolve()),
            "intact_trace_cache": str(args.intact_trace_cache.resolve()),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
