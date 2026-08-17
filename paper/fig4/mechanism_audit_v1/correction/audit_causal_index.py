#!/usr/bin/env python3
"""Prove the exact causal index convention with index-valued synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    N_LAGS,
    N_PRECEDING,
    N_SCORED,
    OUT_DIR,
    lag_windows_from_sequence,
    validate_monotone_causal_windows,
    write_json,
)


def exact_torch_windows(sequence: np.ndarray) -> np.ndarray:
    """Run an index-valued movie through the production lag embedder."""
    import torch

    from paper.fig4.upstream.real_trace_matrix.model import _embed_time_lags

    values = np.asarray(sequence, dtype=np.float32)
    movie = torch.from_numpy(values[:, None, None])
    embedded = _embed_time_lags(movie, n_lags=N_LAGS, torch=torch)
    return embedded[:, 0, :, 0, 0].numpy().astype(np.int64)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Synthetic eye positions equal their source-time indices.  Negative
    # indices are the true preceding samples, 0..39 are the scored interval.
    corrected_sequence = np.arange(-N_PRECEDING, N_SCORED, dtype=np.int64)
    expected = lag_windows_from_sequence(corrected_sequence)
    observed = exact_torch_windows(corrected_sequence)
    if not np.array_equal(observed, expected):
        raise AssertionError("Production `_embed_time_lags` differs from the explicit index mapping")
    output_times = np.arange(N_SCORED, dtype=np.int64)
    validation = validate_monotone_causal_windows(observed, output_times)
    if validation != {
        "total_outputs": 40,
        "outputs_with_future_samples": 0,
        "outputs_with_nonmonotonic_indices": 0,
        "outputs_with_nonunit_index_steps": 0,
    }:
        raise AssertionError(validation)

    legacy_sequence = np.concatenate((np.arange(N_LAGS), np.arange(N_SCORED)))
    legacy_all = exact_torch_windows(legacy_sequence)
    # The production scorer discards embedded output zero.
    legacy = legacy_all[1:]
    rows: list[dict[str, object]] = []
    for convention, windows in (("corrected_true_history", observed), ("legacy_wrapped_prefix", legacy)):
        for output_index, lag_order in enumerate(windows):
            chronological = lag_order[::-1]
            rows.append(
                {
                    "convention": convention,
                    "output_index": output_index,
                    "output_time": output_index,
                    "model_lag_order_current_to_oldest": ",".join(map(str, lag_order)),
                    "chronological_oldest_to_current": ",".join(map(str, chronological)),
                    "current_index": int(lag_order[0]),
                    "oldest_index": int(lag_order[-1]),
                    "contains_future": bool(np.any(lag_order > output_index)),
                    "chronological": bool(np.all(np.diff(chronological) >= 0)),
                    "contains_wrap": bool(np.any(np.diff(chronological) < 0)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "causal_index_mapping.csv", index=False)
    write_json(
        OUT_DIR / "causal_index_test_output.json",
        {
            "synthetic_eye_definition": "e[t] = t",
            "exact_embedder": "paper.fig4.upstream.real_trace_matrix.model._embed_time_lags",
            "model_lag_axis": "lag 0=current; lag 31=oldest",
            "corrected_input_frames_consumed": int(len(corrected_sequence)),
            "corrected_outputs": int(len(observed)),
            "preceding_samples_required": N_PRECEDING,
            "current_sample_included": True,
            "intended_window_at_output_t": "t-31 ... t",
            "validation": validation,
            "first_corrected_output_model_lag_order": observed[0],
            "first_corrected_output_chronological": observed[0, ::-1],
            "last_corrected_output_model_lag_order": observed[-1],
            "last_corrected_output_chronological": observed[-1, ::-1],
        },
    )
    report = f"""# Causal index audit

The exact production lag embedder was run on a synthetic scalar trajectory whose sample value equals its source index, `e[t] = t`. The model input lag axis includes the current sample: lag 0 is time `t`, and lag 31 is time `t-31`. Therefore a scored output at source time `t` consumes the ordered retinal samples `t-31 ... t`. It requires **31 preceding samples plus the current sample**, not 32 preceding samples.

A corrected 40-output trajectory consumes 71 retinal frames: true source times `-31 ... 39`. The first output sees chronological indices `-31 ... 0`; the last sees `8 ... 39`. All {validation['total_outputs']} corrected outputs passed: future-index count {validation['outputs_with_future_samples']}, nonmonotonic-index count {validation['outputs_with_nonmonotonic_indices']}, and non-unit-step count {validation['outputs_with_nonunit_index_steps']}.

The legacy helper instead creates `[0...31] + [0...39]`, produces 41 embedded outputs, and the scorer drops output zero. Its first 31 retained outputs contain future samples and a `31 -> 0` wrap. The full per-output comparison is saved in `causal_index_mapping.csv`.
"""
    (OUT_DIR / "CAUSAL_INDEX_AUDIT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
