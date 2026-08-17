#!/usr/bin/env python3
"""Audit the temporal prefix actually used by the production Figure 4 scorer.

This is intentionally a narrow integrity-gate analysis.  It records the exact
index mapping, quantifies the artificial sample-31 -> sample-0 discontinuity,
and stops.  It does not repair the scorer or continue to mechanism tests.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MATRIX_DIR = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)
PREVIOUS_DIR = ROOT / "outputs/figures/fig4/spatiotemporal_tuning"
OUT_DIR = ROOT / "outputs/figures/fig4/mechanism_audit_v1"
SCORER_SOURCE = ROOT / "paper/fig4/upstream/real_trace_matrix/model.py"
PRODUCTION_HELPER = ROOT / "ryan/digital-twin-fem/_common.py"
FRAME_RATE_HZ = 120.0
N_LAGS = 32
N_SCORED = 40
SCALES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
START_GIT_COMMIT = "1d3cbe538a8f403bec39e58c1e56a95827dca261"
START_GIT_STATUS = ["?? paper/fig4/spatiotemporal_tuning/"]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def history_mapping() -> pd.DataFrame:
    """Map each retained response to the source indices in its lag window."""
    # make_counterfactual_stim constructs the eye sequence as
    # [trace[0:32], trace[0:40]].  Lag embedding creates 41 outputs.  The
    # production scorer drops output 0 and retains embedded outputs 1:41.
    padded_source_index = np.concatenate((np.arange(N_LAGS), np.arange(N_SCORED)))
    rows: list[dict[str, Any]] = []
    for scored_index in range(N_SCORED):
        embedded_index = scored_index + 1
        current_padded_index = N_LAGS - 1 + embedded_index
        chronological = padded_source_index[
            current_padded_index - N_LAGS + 1 : current_padded_index + 1
        ]
        future = chronological[chronological > scored_index]
        wraps = np.flatnonzero(np.diff(chronological) < 0)
        rows.append(
            {
                "scored_output_index": scored_index,
                "embedded_tensor_index": embedded_index,
                "nominal_current_source_index": int(chronological[-1]),
                "history_source_indices_oldest_to_current": ",".join(map(str, chronological)),
                "n_future_source_samples": int(len(future)),
                "contains_future_samples": bool(len(future)),
                "contains_sample31_to_sample0_wrap": bool(len(wraps)),
                "wrap_time_before_current_ms": (
                    float(scored_index / FRAME_RATE_HZ * 1000.0) if len(wraps) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def trace_artifacts(trace_table: pd.DataFrame, trace_xy: np.ndarray) -> pd.DataFrame:
    table = trace_table.sort_values("trace_bank_index").reset_index(drop=True).copy()
    if trace_xy.shape != (len(table), N_SCORED, 2):
        raise ValueError(f"Unexpected trace bank shape: {trace_xy.shape}")
    native_speed = np.linalg.norm(np.diff(trace_xy, axis=1), axis=2) * FRAME_RATE_HZ
    wrap_displacement = np.linalg.norm(trace_xy[:, 0] - trace_xy[:, N_LAGS - 1], axis=1)
    wrap_speed = wrap_displacement * FRAME_RATE_HZ
    threshold = pd.to_numeric(table["rendered_microsaccade_threshold_dps"], errors="coerce").to_numpy()
    cached_ms = (
        pd.to_numeric(table["rendered_n_microsaccade_events"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
        > 0
    )
    out = pd.DataFrame(
        {
            "trace_bank_index": table["trace_bank_index"].to_numpy(dtype=int),
            "cached_interval_has_microsaccade": cached_ms,
            "cached_interval_microsaccade_count": pd.to_numeric(
                table["rendered_n_microsaccade_events"], errors="coerce"
            ).fillna(0).to_numpy(dtype=int),
            "microsaccade_threshold_deg_s": threshold,
            "artificial_wrap_displacement_deg": wrap_displacement,
            "artificial_wrap_speed_deg_s": wrap_speed,
            "native_max_speed_deg_s": np.max(native_speed, axis=1),
            "native_p95_speed_deg_s": np.quantile(native_speed, 0.95, axis=1),
            "artificial_wrap_exceeds_event_threshold": wrap_speed >= threshold,
            "artificial_wrap_exceeds_native_max": wrap_speed > np.max(native_speed, axis=1),
        }
    )
    for scale in SCALES:
        out[f"artificial_wrap_speed_scale_{scale:g}_deg_s"] = wrap_speed * scale
    return out


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=float), [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(zip(("min", "p05", "median", "p95", "max"), map(float, q)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = history_mapping()
    mapping.to_csv(OUT_DIR / "history_index_mapping.csv", index=False)
    trace_table = pd.read_csv(MATRIX_DIR / "trace_feature_table.csv")
    trace_xy = np.load(MATRIX_DIR / "trace_xy.npy")
    artifacts = trace_artifacts(trace_table, trace_xy)
    artifacts.to_csv(OUT_DIR / "trace_prefix_artifact.csv", index=False)

    drift = ~artifacts["cached_interval_has_microsaccade"].to_numpy(dtype=bool)
    microsaccade = ~drift
    wrap_speed = artifacts["artificial_wrap_speed_deg_s"].to_numpy(dtype=float)
    above_threshold = artifacts["artificial_wrap_exceeds_event_threshold"].to_numpy(dtype=bool)
    above_native = artifacts["artificial_wrap_exceeds_native_max"].to_numpy(dtype=bool)
    previous_stats = json.loads((PREVIOUS_DIR / "analysis/statistics.json").read_text(encoding="utf-8"))
    replay = json.loads(
        (PREVIOUS_DIR / "controlled_scaling/replay_validation.json").read_text(encoding="utf-8")
    )
    statistics = {
        "audit_status": "STOPPED_NEW_FIGURE4_HISTORY_BUG",
        "n_scored_outputs": N_SCORED,
        "n_outputs_with_noncausal_wrapped_history": int(mapping["contains_future_samples"].sum()),
        "fraction_outputs_with_noncausal_wrapped_history": float(mapping["contains_future_samples"].mean()),
        "n_outputs_with_valid_sliding_history": int((~mapping["contains_future_samples"]).sum()),
        "artificial_wrap_time_range_ms": [0.0, 250.0],
        "n_trajectories": int(len(artifacts)),
        "n_cached_drift_only_trajectories": int(drift.sum()),
        "n_cached_microsaccade_trajectories": int(microsaccade.sum()),
        "artificial_wrap_speed_deg_s": {
            "all": quantiles(wrap_speed),
            "cached_drift_only": quantiles(wrap_speed[drift]),
            "cached_microsaccade": quantiles(wrap_speed[microsaccade]),
        },
        "cached_drift_only_artificial_wrap_exceeds_event_threshold": {
            "n": int(np.sum(above_threshold & drift)),
            "fraction": float(np.mean(above_threshold[drift])),
        },
        "cached_drift_only_artificial_wrap_exceeds_native_max_speed": {
            "n": int(np.sum(above_native & drift)),
            "fraction": float(np.mean(above_native[drift])),
        },
        "previous_reproduction_gate": {
            "n_units": previous_stats["n_units"],
            "n_low_sf": previous_stats["figure4b_low_sf_units"],
            "n_high_sf": previous_stats["figure4b_high_sf_units"],
            "n_interior_tf": previous_stats["n_interior_tf_preference"],
            "n_boundary_tf": previous_stats["n_boundary_tf_preference"],
            "median_vstar_low_deg_s": previous_stats["median_predicted_speed_low_deg_s"],
            "median_vstar_high_deg_s": previous_stats["median_predicted_speed_high_deg_s"],
            "figure4b_max_abs_error_percentage_points": previous_stats[
                "figure4b_max_absolute_reproduction_error_percent_points"
            ],
            "controlled_scaling_replay_validation": replay,
        },
    }
    write_json(OUT_DIR / "statistics.json", statistics)

    current_commit = git_output("rev-parse", "HEAD")
    current_status = [line for line in git_output("status", "--short").splitlines() if line]
    manifest = {
        "analysis": "figure4_mechanism_audit_v1_blocking_history_integrity_gate",
        "status": "stopped_before_mechanistic_interpretation",
        "stop_condition": "new bug in old Figure 4 temporal history construction",
        "repo_state_at_audit_start": {
            "commit": START_GIT_COMMIT,
            "status_short": START_GIT_STATUS,
        },
        "repo_state_when_generated": {"commit": current_commit, "status_short": current_status},
        "inputs": {
            "matrix_dir": MATRIX_DIR,
            "trace_xy_sha256": sha256_file(MATRIX_DIR / "trace_xy.npy"),
            "trace_feature_table_sha256": sha256_file(MATRIX_DIR / "trace_feature_table.csv"),
            "production_scorer_source": SCORER_SOURCE,
            "production_scorer_source_sha256": sha256_file(SCORER_SOURCE),
            "original_production_helper": PRODUCTION_HELPER,
            "original_production_helper_sha256": sha256_file(PRODUCTION_HELPER),
            "previous_statistics": PREVIOUS_DIR / "analysis/statistics.json",
            "controlled_scaling_replay_validation": PREVIOUS_DIR
            / "controlled_scaling/replay_validation.json",
        },
        "exact_sequence_contract": {
            "input_trace_samples": N_SCORED,
            "model_lags": N_LAGS,
            "padded_eye_source_indices": list(range(N_LAGS)) + list(range(N_SCORED)),
            "lag_embedded_outputs": N_SCORED + 1,
            "discarded_embedded_outputs": [0],
            "retained_embedded_outputs": list(range(1, N_SCORED + 1)),
        },
        "outputs": [
            "README.md",
            "MECHANISM_REPORT.md",
            "statistics.json",
            "analysis_manifest.json",
            "history_index_mapping.csv",
            "trace_prefix_artifact.csv",
        ],
        "deferred_by_stop_condition": [
            "q and SF-identifiability audit",
            "output TF time courses",
            "retinal space-to-time spectra",
            "learned frontend characterization",
            "history-conditioned controlled experiments",
            "layer hooks and frontend interventions",
            "Figures A-D and final mechanistic claims",
        ],
    }
    write_json(OUT_DIR / "analysis_manifest.json", manifest)

    readme = """# Figure 4 mechanism audit v1 outputs

Status: **stopped at the temporal-history integrity gate**.

`history_index_mapping.csv` gives the exact source-sample history for every
retained response. `trace_prefix_artifact.csv` quantifies the artificial
sample-31 to sample-0 wrap for all 1,000 trajectories. See
`MECHANISM_REPORT.md` for the blocking finding.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    drift_wrap = statistics["artificial_wrap_speed_deg_s"]["cached_drift_only"]
    report = f"""# FIGURE 4 MECHANISM AUDIT — BLOCKING REPORT

## Executive summary

The audit stopped before mechanistic interpretation because the production Figure 4 scorer does not provide a causal 32-frame prehistory for the 40-sample FEM snippets. It constructs the eye-position sequence as `trace[0:32] + trace[0:40]`, lag-embeds that sequence, drops the first of 41 outputs, and scores the remaining 40. Consequently, the first 31 scored responses ({100.0 * statistics['fraction_outputs_with_noncausal_wrapped_history']:.1f}%) contain future samples and a circular sample-31 to sample-0 transition; only the final 9 responses have an ordinary causal sliding history. The artificial transition is presented at apparent times 0--250 ms before the current output, exactly overlapping the requested recent-event analysis window. This construction is present in both the original production helper and the scorer that reproduces the frozen cache. Therefore the existing Figure 4 and controlled-scaling results cannot currently distinguish natural drift/history effects from this boundary artifact. Per the requested scientific-integrity stop condition, no q rescue, layerwise mechanism claim, or synthetic-history experiment was run after confirming the bug.

## Reproduction gate

The starting state was commit `{START_GIT_COMMIT}`. Existing products reproduce 100 RR units (71 low-SF, 29 high-SF), 79 interior and 21 boundary TF peaks, median inferred speeds of {previous_stats['median_predicted_speed_low_deg_s']:.6g} deg/s (low-SF) and {previous_stats['median_predicted_speed_high_deg_s']:.6g} deg/s (high-SF), and Figure 4B to a maximum absolute error of {previous_stats['figure4b_max_absolute_reproduction_error_percent_points']:.3g} percentage points. The controlled scale-1 replay differs from the cached SSI by at most {replay['scale1_max_abs_ssi_error_vs_cached']:.6g}. This establishes that the audited scorer is the scorer underlying the reproduced result, not a newly introduced implementation.

## Exact indexing failure

Let the stored trajectory samples be `e[0] ... e[39]`. The helper supplies:

```text
e[0], e[1], ..., e[31], e[0], e[1], ..., e[39]
```

After lag embedding and removal of the extra first output, nominal scored output 0 sees chronological source indices `1,2,...,31,0`; output 1 sees `2,3,...,31,0,1`; and output 30 sees `31,0,1,...,30`. These histories contain future samples and the same circular wrap. Outputs 31--39 finally see valid sliding windows `0:31` through `8:39`. The exact mapping for every output is saved in `history_index_mapping.csv`.

## Size of the artificial event

Across the 800 trajectories labeled drift-only in the scored interval, the implied wrap speed has median {drift_wrap['median']:.3f} deg/s and 95th percentile {drift_wrap['p95']:.3f} deg/s. It exceeds each trace's stored microsaccade threshold for {statistics['cached_drift_only_artificial_wrap_exceeds_event_threshold']['n']} of 800 traces ({100.0 * statistics['cached_drift_only_artificial_wrap_exceeds_event_threshold']['fraction']:.1f}%) and exceeds the largest native adjacent-sample speed for {statistics['cached_drift_only_artificial_wrap_exceeds_native_max_speed']['n']} traces ({100.0 * statistics['cached_drift_only_artificial_wrap_exceeds_native_max_speed']['fraction']:.1f}%). Even below the event threshold, it is noncausal for all first 31 outputs and is scaled along with the trajectory in the controlled-amplitude experiment.

## Scientific consequence

The current bank contains no genuine pre-snippet history in `trace_xy.npy`; the scorer manufactures one from the snippet itself. Therefore the requested categories “no fast event in entire causal history” and “time since last fast event” are not identifiable from the frozen scored inputs as currently constructed. A history dependence found with the present renderer could be caused by the artificial wrap, while a drift effect could be contaminated by it. The old SSI values remain exactly reproducible, but their mechanistic interpretation is blocked.

## Required decision before continuing

Continuing requires an explicitly authorized new counterfactual bank with a valid history convention, followed by a sensitivity comparison against the frozen published bank. Viable conventions include recovering the real 32 samples preceding each snippet from the source trace or, if unavailable, using an explicitly labeled hold-at-initial-position prefix. Either choice changes model inputs and thus cannot be made silently under the instruction to preserve the 100 x 1,000 bank. The original values should remain frozen and be reported alongside, not overwritten.

## Manuscript-safe status

SUPPORTED: the frozen Figure 4 computation is reproducible; retinal motion changes SSI in that computation; the production scorer uses the documented wrapped prefix.

CONSISTENT WITH: some observed effect may reflect real drift-induced temporal modulation, a transient-memory effect, or both.

NOT SUPPORTED: that the scored “drift-only” condition contains only genuine drift throughout its causal input; that preceding-event timing can be inferred from the present bank; or that the current controlled scaling isolates sustained velocity from transient history.
"""
    (OUT_DIR / "MECHANISM_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote blocking audit to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
