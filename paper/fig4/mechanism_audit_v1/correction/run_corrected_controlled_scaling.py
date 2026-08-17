#!/usr/bin/env python3
"""Rerun controlled amplitude scaling with corrected true and held histories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    CONTROLLED_DIR,
    CORE_DIR,
    LEGACY_MATRIX_DIR,
    N_PRECEDING,
    PREVIOUS_ANALYSIS_DIR,
    SCALE_FACTORS,
    sha256_file,
    write_json,
)
from paper.fig4.mechanism_audit_v1.correction.scoring import (
    build_direct_rr100_readout,
    score_corrected_histories_for_patch,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


BANK_NAMES = ("real_trace_true_history_v1", "real_trace_held_initial_history_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scaled_histories(base: np.ndarray, bank_index: int) -> np.ndarray:
    out = []
    for trace in base:
        e0 = trace[N_PRECEDING].copy()
        scored_displacement = trace[N_PRECEDING:] - e0[None]
        for scale in SCALE_FACTORS:
            history = trace.copy()
            history[N_PRECEDING:] = e0[None] + float(scale) * scored_displacement
            if bank_index == 1:
                history[:N_PRECEDING] = e0
            out.append(history)
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    args = parse_args()
    CONTROLLED_DIR.mkdir(parents=True, exist_ok=True)
    output = CONTROLLED_DIR / "corrected_controlled_scaling_response.npz"
    if output.exists() and not args.force:
        print(f"Using existing {output}")
        return 0
    selected_images = pd.read_csv(PREVIOUS_ANALYSIS_DIR / "controlled_scaling/selected_images.csv")
    selected_traces = pd.read_csv(PREVIOUS_ANALYSIS_DIR / "controlled_scaling/selected_traces.csv")
    image_ids = selected_images["image_index"].to_numpy(dtype=int)
    trace_ids = selected_traces["trace_bank_index"].to_numpy(dtype=int)
    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        true_base = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)
        held_base = np.asarray(archive["held_initial_history_xy"][trace_ids], dtype=np.float32)
    all_histories = [scaled_histories(true_base, 0), scaled_histories(held_base, 1)]
    shape = (2, len(image_ids), len(trace_ids), len(SCALE_FACTORS), 100)
    ssi = np.full(shape, np.nan, dtype=np.float32)
    expected = np.full_like(ssi, np.nan)
    rate = np.full_like(ssi, np.nan)
    population = np.full(shape[:-1], np.nan, dtype=np.float32)
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
    canvas_cache = {}
    for image_ordinal, image_id in enumerate(image_ids):
        patch, _ = extract_patch(images.iloc[image_id], canvas_cache=canvas_cache, patch_size_px=540)
        for bank_index, histories in enumerate(all_histories):
            values = score_corrected_histories_for_patch(
                scorer,
                patch,
                histories,
                trace_batch_size=args.trace_batch_size,
                frame_batch_size=args.frame_batch_size,
                direct_rr_readout=direct,
            )
            ssi[bank_index, image_ordinal] = values[0].reshape(len(trace_ids), len(SCALE_FACTORS), 100)
            expected[bank_index, image_ordinal] = values[1].reshape(len(trace_ids), len(SCALE_FACTORS), 100)
            rate[bank_index, image_ordinal] = values[2].reshape(len(trace_ids), len(SCALE_FACTORS), 100)
            population[bank_index, image_ordinal] = values[3].reshape(len(trace_ids), len(SCALE_FACTORS))
        print(f"corrected controlled scaling image {image_ordinal + 1}/{len(image_ids)}", flush=True)
    np.savez_compressed(
        output,
        ssi=ssi,
        expected_spikes=expected,
        mean_rate=rate,
        population_ssi=population,
        bank_names=np.asarray(BANK_NAMES),
        scale_factors=SCALE_FACTORS,
        selected_image_index=image_ids,
        selected_trace_index=trace_ids,
        intervention=np.asarray(
            "preserve true prefix; scale only scored displacement relative to e0; held bank replaces prefix by e0"
        ),
    )
    scale1 = int(np.flatnonzero(np.isclose(SCALE_FACTORS, 1.0))[0])
    validation = {}
    for bank_index, bank in enumerate(BANK_NAMES):
        core_path = CORE_DIR / bank / "merged/ssi_matrix.npy"
        if core_path.is_file():
            core = np.load(core_path)
            reference = core[np.ix_(image_ids, trace_ids, np.arange(100))]
            validation[f"{bank}_scale1_max_abs_ssi_error_vs_core"] = float(
                np.max(np.abs(ssi[bank_index, :, :, scale1] - reference))
            )
        else:
            # The same-day first pass intentionally precedes the complete
            # 100-image held-prefix core bank.  Do not fabricate a validation
            # target; retain an explicit machine-readable status instead.
            validation[f"{bank}_scale1_core_validation_status"] = "not_available_full_core_bank_deferred"
    write_json(CONTROLLED_DIR / "validation.json", validation)
    write_json(
        CONTROLLED_DIR / "manifest.json",
        {
            "analysis": "corrected_history_primary_within_window_controlled_scaling",
            "banks": BANK_NAMES,
            "legacy_bank": "not_interpreted; previous controlled scaling inherited wrapped prefix",
            "intervention": "prefix fixed exactly; scored e(t)=e0+s*(e(t)-e0)",
            "scale_factors": SCALE_FACTORS,
            "selected_image_ids": image_ids,
            "selected_trace_ids": trace_ids,
            "output": output,
            "output_sha256": sha256_file(output),
            "validation": validation,
            "model": scorer.provenance,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
