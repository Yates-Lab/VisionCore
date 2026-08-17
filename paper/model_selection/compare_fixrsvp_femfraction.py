#!/usr/bin/env python3
"""Compare Ryan and a selected twin's FEM fraction on identical FixRSVP data.

The production Figure-3 cache builders historically inherited model-specific
time-grid conventions.  This utility instead starts from the observation-
aligned trace cache written by ``evaluate_dekel_fixrsvp.py`` and pairs it with
the exact Figure-3 Ryan predictions.  Both models are therefore decomposed on
the same trials, bins, cells, eye positions, and data filters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import dill
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "paper" / "covariance_decomposition"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_decompose import (  # noqa: E402
    decompose_model_session,
    one_minus_alpha_windowed,
)


def load_records(path: Path):
    try:
        value = np.load(path, allow_pickle=True)
    except Exception:
        with path.open("rb") as stream:
            value = dill.load(stream)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected a list of session records in {path}, got {type(value)}")
    return value


def finite_unit_interval(values):
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values >= 0.0) & (values <= 1.0)


def summary(values, *, unit_interval=True):
    values = np.asarray(values, dtype=float)
    values = values[
        finite_unit_interval(values) if unit_interval else np.isfinite(values)
    ]
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else None,
        "q25": float(np.quantile(values, 0.25)) if values.size else None,
        "q75": float(np.quantile(values, 0.75)) if values.size else None,
    }


def model_fem_fraction(rhat, eye, valid_mask, dfs, *, count_bins):
    """Compute only the model side, avoiding a duplicate observation pass."""
    rhat = np.asarray(rhat, dtype=float)
    dfs = np.asarray(dfs, dtype=float)
    values = np.full(rhat.shape[2], np.nan, dtype=float)
    for unit in range(rhat.shape[2]):
        valid = np.asarray(valid_mask, dtype=bool) & (dfs[:, :, unit] != 0)
        result = one_minus_alpha_windowed(
            rhat[:, :, unit : unit + 1],
            eye,
            valid,
            t_count=count_bins,
        )
        values[unit] = result["one_minus_alpha"]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-traces", type=Path, required=True)
    parser.add_argument("--ryan-cache", type=Path, required=True)
    parser.add_argument("--aligned-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count-bins", type=int, default=3)
    parser.add_argument("--fixation-radius", type=float, default=0.5)
    parser.add_argument("--min-session-units", type=int, default=10)
    args = parser.parse_args()

    model_records = load_records(args.model_traces)
    ryan_by_session = {row["session"]: row for row in load_records(args.ryan_cache)}
    aligned_by_session = {
        row["session"]: row for row in load_records(args.aligned_cache)
    }

    rows = []
    observation_max_abs = 0.0
    for model in model_records:
        session = model["session"]
        if session not in ryan_by_session or session not in aligned_by_session:
            continue
        ryan = ryan_by_session[session]
        aligned = aligned_by_session[session]

        aligned_ids = np.asarray(aligned["neuron_mask"], dtype=np.int64)
        rate = np.asarray(aligned["rate_hz"], dtype=float)
        psth_r2 = np.asarray(aligned["psth_r2"], dtype=float)
        included_ids = set(
            aligned_ids[
                np.isfinite(rate)
                & (rate > 2.0)
                & np.isfinite(psth_r2)
                & (psth_r2 > 0.10)
            ].tolist()
        )

        model_ids = np.asarray(model["neuron_mask"], dtype=np.int64)
        ryan_ids = np.asarray(ryan["neuron_mask"], dtype=np.int64)
        ryan_column = {int(unit): index for index, unit in enumerate(ryan_ids)}
        selected = [
            (index, ryan_column[int(unit)], int(unit))
            for index, unit in enumerate(model_ids)
            if int(unit) in included_ids and int(unit) in ryan_column
        ]
        if len(selected) < args.min_session_units:
            continue
        model_columns = np.asarray([item[0] for item in selected], dtype=np.int64)
        ryan_columns = np.asarray([item[1] for item in selected], dtype=np.int64)
        unit_ids = np.asarray([item[2] for item in selected], dtype=np.int64)

        robs_model = np.asarray(model["robs_used"], dtype=float)[:, :, model_columns]
        robs_ryan = np.asarray(ryan["robs_used"], dtype=float)[:, :, ryan_columns]
        if robs_model.shape != robs_ryan.shape:
            raise RuntimeError(
                f"{session}: observation shapes differ {robs_model.shape} != {robs_ryan.shape}"
            )
        same_finite = np.array_equal(np.isfinite(robs_model), np.isfinite(robs_ryan))
        overlap = np.isfinite(robs_model) & np.isfinite(robs_ryan)
        max_abs = (
            float(np.max(np.abs(robs_model[overlap] - robs_ryan[overlap])))
            if overlap.any()
            else float("inf")
        )
        observation_max_abs = max(observation_max_abs, max_abs)
        if not same_finite or max_abs != 0.0:
            raise RuntimeError(
                f"{session}: model and Ryan observations are not exact "
                f"(finite={same_finite}, max_abs={max_abs})"
            )

        dfs = np.asarray(model["dfs_used"], dtype=float)[:, :, model_columns]
        eye = np.asarray(model["eyepos_used"], dtype=float)
        valid = np.isfinite(eye).all(axis=-1)
        valid &= np.hypot(eye[..., 0], eye[..., 1]) < args.fixation_radius
        model_decomp = decompose_model_session(
            np.asarray(model["rhat_used"], dtype=float)[:, :, model_columns],
            robs_model,
            eye,
            valid,
            dfs,
            count_bins=args.count_bins,
        )
        ryan_model_uncl = model_fem_fraction(
            np.asarray(ryan["rhat_used"], dtype=float)[:, :, ryan_columns],
            eye,
            valid,
            dfs,
            count_bins=args.count_bins,
        )

        for index, unit in enumerate(unit_ids):
            rows.append(
                {
                    "session": session,
                    "unit_id": int(unit),
                    "empirical": float(model_decomp["B_obs_uncl"][index]),
                    "model": float(model_decomp["B_model_uncl"][index]),
                    "ryan": float(ryan_model_uncl[index]),
                }
            )
        print(f"{session}: {len(unit_ids)} units", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "paired_femfraction.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["session", "unit_id", "empirical", "model", "ryan"]
        )
        writer.writeheader()
        writer.writerows(rows)

    arrays = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in ("empirical", "model", "ryan")
    }
    paired_model = finite_unit_interval(arrays["empirical"]) & finite_unit_interval(
        arrays["model"]
    )
    paired_ryan = finite_unit_interval(arrays["empirical"]) & finite_unit_interval(
        arrays["ryan"]
    )
    report = {
        "model_traces": str(args.model_traces.resolve()),
        "ryan_cache": str(args.ryan_cache.resolve()),
        "aligned_cache": str(args.aligned_cache.resolve()),
        "count_bins": args.count_bins,
        "fixation_radius": args.fixation_radius,
        "min_session_units": args.min_session_units,
        "n_sessions": len({row["session"] for row in rows}),
        "n_units": len(rows),
        "observation_max_abs_difference": observation_max_abs,
        "summaries": {key: summary(value) for key, value in arrays.items()},
        "paired": {
            "empirical_minus_model": summary(
                arrays["empirical"][paired_model] - arrays["model"][paired_model],
                unit_interval=False,
            ),
            "empirical_minus_ryan": summary(
                arrays["empirical"][paired_ryan] - arrays["ryan"][paired_ryan],
                unit_interval=False,
            ),
        },
        "csv": str(csv_path.resolve()),
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
