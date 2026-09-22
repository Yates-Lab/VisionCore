#!/usr/bin/env python3
"""Independently audit saved FixRSVP traces and CCnorm bookkeeping.

This deliberately does not import the production CCnorm helper.  It rebuilds
the data-only support, PSTH correlation, split-half noise ceiling, and the
``CCnorm = CCabs / CCmax`` identity directly from the saved trial tensors.  The
audit is intended as a second implementation that can be run on a one-session
sentinel or on the final all-session trace cache.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


MIN_TRIALS_PER_BIN = 20
MIN_TIME_BINS = 20
MIN_TRIALS_PER_HALF = 10


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.mean(a * a) * np.mean(b * b))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.mean(a * b) / denom)


def _data_support(robs, dfs):
    robs = np.asarray(robs)
    dfs = np.asarray(dfs)
    if robs.shape != dfs.shape:
        raise ValueError(f"robs/dfs shape mismatch: {robs.shape} versus {dfs.shape}")
    return np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)


def _masked_time_mean(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    numerator = np.where(mask, values, 0.0).sum(axis=0)
    denominator = mask.sum(axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def independent_ccabs(robs, prediction, support):
    """Recompute full-PSTH correlations without production evaluation code."""
    robs = np.asarray(robs, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if robs.shape != prediction.shape or robs.shape != support.shape:
        raise ValueError("robs, prediction, and support must have one shape")
    result = np.full(robs.shape[-1], np.nan, dtype=np.float64)
    for unit in range(robs.shape[-1]):
        mask = support[..., unit]
        eligible = mask.sum(axis=0) >= MIN_TRIALS_PER_BIN
        if int(eligible.sum()) < MIN_TIME_BINS:
            continue
        y = _masked_time_mean(robs[..., unit], mask)[eligible]
        p = _masked_time_mean(prediction[..., unit], mask)[eligible]
        finite = np.isfinite(y) & np.isfinite(p)
        if int(finite.sum()) >= MIN_TIME_BINS:
            result[unit] = _pearson(y[finite], p[finite])
    return result


def independent_ccmax(robs, support, *, seed, n_splits):
    """Recompute the data-only split-half ceiling without model predictions."""
    robs = np.asarray(robs, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if robs.shape != support.shape:
        raise ValueError("robs and support must have one shape")
    n_trials, _, n_units = robs.shape
    rng = np.random.default_rng(int(seed))
    splits = []
    for _ in range(int(n_splits)):
        permutation = rng.permutation(n_trials)
        splits.append((permutation[: n_trials // 2], permutation[n_trials // 2 :]))

    ceilings = np.full(n_units, np.nan, dtype=np.float64)
    for unit in range(n_units):
        response = robs[..., unit]
        mask = support[..., unit]
        eligible = mask.sum(axis=0) >= MIN_TRIALS_PER_BIN
        if int(eligible.sum()) < MIN_TIME_BINS:
            continue
        half_correlations = []
        for first, second in splits:
            mask_first = mask[first]
            mask_second = mask[second]
            good = (
                eligible
                & (mask_first.sum(axis=0) >= MIN_TRIALS_PER_HALF)
                & (mask_second.sum(axis=0) >= MIN_TRIALS_PER_HALF)
            )
            if int(good.sum()) < MIN_TIME_BINS:
                continue
            first_mean = _masked_time_mean(response[first], mask_first)[good]
            second_mean = _masked_time_mean(response[second], mask_second)[good]
            finite = np.isfinite(first_mean) & np.isfinite(second_mean)
            if int(finite.sum()) < MIN_TIME_BINS:
                continue
            correlation = _pearson(first_mean[finite], second_mean[finite])
            if np.isfinite(correlation):
                half_correlations.append(correlation)
        if not half_correlations:
            continue
        split_half = float(np.mean(half_correlations))
        if np.isfinite(split_half) and split_half > 0:
            ceilings[unit] = np.sqrt(2.0 * split_half / (1.0 + split_half))
    return ceilings


def _max_abs_difference(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def audit_trace(trace, *, n_splits=500, affine_scale=1.7, affine_offset=0.3):
    robs = np.asarray(trace["robs_used"], dtype=np.float64)
    prediction = np.asarray(trace["rhat_used"], dtype=np.float64)
    dfs = np.asarray(trace["dfs_used"])
    stored_support = np.asarray(trace["ccnorm_support"], dtype=bool)
    expected_support = _data_support(robs, dfs)
    if not np.array_equal(stored_support, expected_support):
        raise AssertionError("stored CCnorm support is not the data-only support")
    if np.any(expected_support & ~np.isfinite(prediction)):
        raise AssertionError("prediction is missing on data-only support")

    ccabs = independent_ccabs(robs, prediction, expected_support)
    stored_ccabs = np.asarray(trace["ccabs"], dtype=np.float64)
    if not np.allclose(ccabs, stored_ccabs, rtol=0, atol=2e-12, equal_nan=True):
        raise AssertionError("independent CCabs does not match stored CCabs")

    transformed = affine_scale * prediction + affine_offset
    transformed_ccabs = independent_ccabs(robs, transformed, expected_support)
    if not np.allclose(ccabs, transformed_ccabs, rtol=0, atol=2e-12, equal_nan=True):
        raise AssertionError("positive affine transform changed PSTH correlation")

    ceilings = [
        independent_ccmax(robs, expected_support, seed=seed, n_splits=n_splits)
        for seed in (42, 43)
    ]
    ccmax = 0.5 * (ceilings[0] + ceilings[1])
    stored_ccmax = np.asarray(trace["ccmax"], dtype=np.float64)
    if not np.allclose(ccmax, stored_ccmax, rtol=0, atol=2e-12, equal_nan=True):
        raise AssertionError("independent data-only CCmax does not match stored CCmax")

    unstable = (ceilings[0] - ceilings[1]) ** 2 > 0.01
    stored_unstable = np.asarray(trace["ccnorm_unstable"], dtype=bool)
    if not np.array_equal(unstable, stored_unstable):
        raise AssertionError("independent stability mask does not match stored mask")

    with np.errstate(divide="ignore", invalid="ignore"):
        identity = stored_ccabs / stored_ccmax
    identity[stored_unstable] = np.nan
    stored_ccnorm = np.asarray(trace["ccnorm"], dtype=np.float64)
    if not np.allclose(identity, stored_ccnorm, rtol=0, atol=2e-12, equal_nan=True):
        raise AssertionError("stored CCnorm is not exactly CCabs / CCmax")

    return {
        "session": str(trace["session"]),
        "n_units": int(robs.shape[-1]),
        "n_supported_samples": int(expected_support.sum()),
        "n_stable_units": int(np.isfinite(stored_ccnorm).sum()),
        "support_exact_match": True,
        "prediction_complete_on_support": True,
        "ccabs_max_abs_error": _max_abs_difference(ccabs, stored_ccabs),
        "positive_affine_ccabs_max_abs_error": _max_abs_difference(
            ccabs, transformed_ccabs
        ),
        "ccmax_max_abs_error": _max_abs_difference(ccmax, stored_ccmax),
        "stability_mask_exact_match": True,
        "ccnorm_identity_max_abs_error": _max_abs_difference(identity, stored_ccnorm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_cache", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-splits", type=int, default=500)
    args = parser.parse_args()

    with args.trace_cache.open("rb") as stream:
        traces = pickle.load(stream)
    if not isinstance(traces, list) or not traces:
        raise ValueError("trace cache must contain a non-empty list of sessions")

    sessions = [audit_trace(trace, n_splits=args.n_splits) for trace in traces]
    result = {
        "trace_cache": str(args.trace_cache.resolve()),
        "independent_implementation": True,
        "n_splits_per_seed": int(args.n_splits),
        "seeds": [42, 43],
        "n_sessions": len(sessions),
        "n_units": int(sum(row["n_units"] for row in sessions)),
        "all_checks_passed": True,
        "sessions": sessions,
    }

    if args.report is not None:
        report = json.loads(args.report.read_text())
        reported_audits = {
            row["session"]: row for row in report.get("metric_audits", [])
        }
        for row in sessions:
            reported = reported_audits.get(row["session"])
            if reported is None:
                raise AssertionError(f"report lacks metric audit for {row['session']}")
            if not (
                reported.get("support_exact_match")
                and reported.get("ccmax_exact_match")
                and reported.get("stability_mask_exact_match")
            ):
                raise AssertionError(f"candidate/reference-model audit failed for {row['session']}")
            if int(reported["n_supported_samples"]) != row["n_supported_samples"]:
                raise AssertionError(f"support count differs for {row['session']}")
        result["candidate_reference_report_audits_passed"] = True
        result["report"] = str(args.report.resolve())

    payload = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
