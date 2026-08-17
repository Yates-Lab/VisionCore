#!/usr/bin/env python3
"""Paired cell-level comparison of two exhaustive Dekel evaluations.

Each evaluation JSON points to the lossless per-unit BPS NPZ written by
``evaluate_dekel_split.py``.  Units are aligned by (session, cid).  The
reported hierarchical bootstrap resamples sessions and then cells within each
resampled session, applying the same per-unit zero clipping and equal-session
averaging used by the training metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_per_unit(report_path: Path):
    report = json.loads(report_path.read_text())
    npz_path = Path(report["per_unit_bps_npz"])
    if not npz_path.is_absolute():
        npz_path = report_path.parent / npz_path
    with np.load(npz_path, allow_pickle=False) as archive:
        names = archive["session_names"].astype(str).tolist()
        values = {}
        for session_index, name in enumerate(names):
            cids = archive[f"cids_{session_index}"].astype(np.int64)
            bps = archive[f"bps_{session_index}"].astype(np.float64)
            if len(cids) != len(bps):
                raise RuntimeError(
                    f"{name}: {len(cids)} cids but {len(bps)} BPS values"
                )
            if len(np.unique(cids)) != len(cids):
                raise RuntimeError(f"{name}: duplicate cids in {npz_path}")
            values[name] = dict(zip(cids.tolist(), bps.tolist()))
    return report, values


def align_per_unit(reference, candidate):
    sessions = sorted(set(reference) & set(candidate))
    if not sessions:
        raise RuntimeError("The evaluations have no sessions in common")
    aligned = {}
    for session in sessions:
        cids = sorted(set(reference[session]) & set(candidate[session]))
        ref = np.asarray([reference[session][cid] for cid in cids], dtype=float)
        cand = np.asarray([candidate[session][cid] for cid in cids], dtype=float)
        valid = np.isfinite(ref) & np.isfinite(cand)
        if valid.any():
            aligned[session] = {
                "cids": np.asarray(cids, dtype=np.int64)[valid],
                "reference": ref[valid],
                "candidate": cand[valid],
            }
    if not aligned:
        raise RuntimeError("No matched units have finite BPS in both evaluations")
    return aligned


def clipped_session_difference(reference, candidate):
    return float(
        np.clip(candidate, 0.0, None).mean()
        - np.clip(reference, 0.0, None).mean()
    )


def complementarity_summary(aligned):
    """Summarize model complementarity with the evaluator's reduction.

    The oracle is deliberately descriptive rather than a deployable score: it
    chooses the larger held-out BPS independently for every matched unit.  Its
    distance above the better complete model is useful evidence for whether a
    unit-selective refinement has meaningful headroom.
    """
    by_session = {}
    pooled_differences = []
    for name, values in aligned.items():
        reference = np.clip(values["reference"], 0.0, None)
        candidate = np.clip(values["candidate"], 0.0, None)
        difference = values["candidate"] - values["reference"]
        pooled_differences.append(difference)
        by_session[name] = {
            "n_units": int(len(reference)),
            "reference_bps": float(reference.mean()),
            "candidate_bps": float(candidate.mean()),
            "oracle_bps": float(np.maximum(reference, candidate).mean()),
            "candidate_better_fraction": float(np.mean(difference > 0)),
            "reference_better_fraction": float(np.mean(difference < 0)),
        }

    reference_overall = float(
        np.mean([values["reference_bps"] for values in by_session.values()])
    )
    candidate_overall = float(
        np.mean([values["candidate_bps"] for values in by_session.values()])
    )
    oracle_overall = float(
        np.mean([values["oracle_bps"] for values in by_session.values()])
    )
    differences = np.concatenate(pooled_differences)
    return {
        "aligned_reference_bps_overall": reference_overall,
        "aligned_candidate_bps_overall": candidate_overall,
        "unit_oracle_bps_overall": oracle_overall,
        "unit_oracle_gain_over_reference": oracle_overall - reference_overall,
        "unit_oracle_gain_over_candidate": oracle_overall - candidate_overall,
        "candidate_better_units": int(np.sum(differences > 0)),
        "reference_better_units": int(np.sum(differences < 0)),
        "tied_units": int(np.sum(differences == 0)),
        "candidate_better_fraction": float(np.mean(differences > 0)),
        "reference_better_fraction": float(np.mean(differences < 0)),
        "complementarity_by_session": by_session,
    }


def hierarchical_bootstrap(aligned, n_bootstrap=20_000, seed=42):
    rng = np.random.default_rng(seed)
    sessions = list(aligned)
    observed_by_session = {
        name: clipped_session_difference(
            aligned[name]["reference"], aligned[name]["candidate"]
        )
        for name in sessions
    }
    observed = float(np.mean(list(observed_by_session.values())))

    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw_index in range(n_bootstrap):
        sampled_sessions = rng.integers(0, len(sessions), size=len(sessions))
        session_differences = []
        for sampled_index in sampled_sessions:
            values = aligned[sessions[int(sampled_index)]]
            n_units = len(values["reference"])
            unit_indices = rng.integers(0, n_units, size=n_units)
            session_differences.append(
                clipped_session_difference(
                    values["reference"][unit_indices],
                    values["candidate"][unit_indices],
                )
            )
        draws[draw_index] = np.mean(session_differences)
    return {
        "observed_paired_difference": observed,
        "difference_by_session": observed_by_session,
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_probability_candidate_le_reference": float(
            np.mean(draws <= 0)
        ),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reference_report, reference_values = load_per_unit(args.reference.resolve())
    candidate_report, candidate_values = load_per_unit(args.candidate.resolve())
    if reference_report["split"] != candidate_report["split"]:
        raise RuntimeError("Cannot compare evaluations from different splits")
    aligned = align_per_unit(reference_values, candidate_values)
    bootstrap = hierarchical_bootstrap(
        aligned, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    complementarity = complementarity_summary(aligned)
    report = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "split": reference_report["split"],
        "reference_bps_overall": reference_report["bps_overall"],
        "candidate_bps_overall": candidate_report["bps_overall"],
        "reported_overall_difference": (
            candidate_report["bps_overall"] - reference_report["bps_overall"]
        ),
        "matched_sessions": len(aligned),
        "matched_units": int(
            sum(len(values["reference"]) for values in aligned.values())
        ),
        **complementarity,
        **bootstrap,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
