#!/usr/bin/env python3
"""Benchmark corrected scoring and verify direct RR100 readout equivalence."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, LEGACY_MATRIX_DIR, OUT_DIR, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import (
    build_direct_rr100_readout,
    make_corrected_causal_stims,
    score_corrected_histories_for_patch,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-traces", type=int, default=32)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    args = parser.parse_args()
    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    patch, _ = extract_patch(images.iloc[0], canvas_cache={}, patch_size_px=540)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        histories = np.asarray(archive["true_history_xy"][: args.n_traces], dtype=np.float32)
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=args.device,
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    direct = build_direct_rr100_readout(scorer)
    # One batch proves the optimized readout is numerically equivalent.
    stim = (make_corrected_causal_stims(_standardize_uint_like(patch), histories[:1], torch=scorer.torch)[:1] - 127.0) / 255.0
    x = stim.to(scorer.device)
    with scorer.torch.no_grad():
        full = scorer._compute_rate_map(x)
        expected = scorer.apply_population_view(full, scorer.population_view)
        from scripts.spatial_info import compute_rate_map

        observed = compute_rate_map(scorer.model, direct, x, behavior=scorer._zero_behavior(1, x.dtype))
    equivalence_error = float(scorer.torch.max(scorer.torch.abs(expected - observed)).item())
    del x, full, expected, observed, stim
    if equivalence_error > 2e-6:
        raise AssertionError(f"Direct RR100 readout mismatch: {equivalence_error}")
    start = time.perf_counter()
    score_corrected_histories_for_patch(
        scorer,
        patch,
        histories,
        trace_batch_size=8,
        frame_batch_size=args.frame_batch_size,
        direct_rr_readout=direct,
    )
    elapsed = time.perf_counter() - start
    result = {
        "device": args.device,
        "n_traces": args.n_traces,
        "frame_batch_size": args.frame_batch_size,
        "elapsed_s": elapsed,
        "seconds_per_movie": elapsed / args.n_traces,
        "movies_per_s": args.n_traces / elapsed,
        "direct_readout_max_abs_equivalence_error": equivalence_error,
    }
    write_json(OUT_DIR / f"benchmark_framebatch{args.frame_batch_size}.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
