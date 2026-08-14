#!/usr/bin/env python3
"""Score zero-motion stabilized SSI baselines for a real-trace SSI matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

UPSTREAM_DIR = Path(__file__).resolve().parent
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from real_trace_matrix.core import score_stabilized_images, write_csv, write_json
from real_trace_matrix.model import RealTraceMatrixScorer


ROOT = Path(__file__).resolve().parents[3]
RUN_STEM = "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_history32_v2"
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)
DEFAULT_MATRIX_DIR = ROOT / "outputs/active_sensing_movie_information" / RUN_STEM / "merged"
CHECKPOINT_ENV = "FIG4_TWIN_CHECKPOINT"
MODEL_CHECKPOINT_FILENAME = "epoch=147-val_bps_overall=0.5702.ckpt"
STAGED_MODEL_CHECKPOINT_PATH = ROOT / "outputs/artifacts/model_checkpoints/fig4_twin" / MODEL_CHECKPOINT_FILENAME


def default_checkpoint_path() -> Path:
    if CHECKPOINT_ENV in os.environ:
        return Path(os.environ[CHECKPOINT_ENV])
    return STAGED_MODEL_CHECKPOINT_PATH
DEFAULT_DATASET_CONFIGS = Path(
    os.environ.get(
        "FIG4_DATASET_CONFIGS",
        str(UPSTREAM_DIR / "dataset_configs" / "multi_basic_120_long.yaml"),
    )
)
DEFAULT_POPULATION_SPEC_DIR = Path(
    os.environ.get(
        "FIG4_RR100_POPULATION_SPEC_DIR",
        str(ROOT / "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"),
    )
)
DEFAULT_MCFARLAND_OUTPUTS = os.environ.get("FIG4_MCFARLAND_OUTPUTS")
OUTPUT_FILES = {
    "ssi": "stabilized_ssi_by_image.npy",
    "expected": "stabilized_expected_spikes_by_image.npy",
    "mean_rate": "stabilized_mean_rate_by_image.npy",
    "population": "stabilized_population_ssi_by_image.npy",
    "table": "stabilized_movie_feature_table.csv",
    "summary": "stabilized_baseline_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rr100-version", type=str, default=RR100_VERSION)
    parser.add_argument("--checkpoint-path", type=Path, default=default_checkpoint_path())
    parser.add_argument("--dataset-configs", type=Path, default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument(
        "--mcfarland-outputs",
        type=Path,
        default=Path(DEFAULT_MCFARLAND_OUTPUTS) if DEFAULT_MCFARLAND_OUTPUTS else None,
    )
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--history-burn-in-samples", type=int, default=32)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def existing_outputs(out_dir: Path) -> list[Path]:
    return [out_dir / name for name in OUTPUT_FILES.values() if (out_dir / name).exists()]


def read_merged_summary_defaults(matrix_dir: Path) -> dict[str, Any]:
    path = matrix_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    shard_summaries = summary.get("shard_summaries")
    if isinstance(shard_summaries, list) and shard_summaries:
        first = shard_summaries[0]
        if isinstance(first, dict):
            return first
    return summary if isinstance(summary, dict) else {}


def main() -> int:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else matrix_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    present = existing_outputs(out_dir)
    if present and not bool(args.force):
        names = ", ".join(path.name for path in present)
        raise FileExistsError(f"Baseline outputs already exist in {out_dir}: {names}. Pass --force to overwrite.")

    summary_defaults = read_merged_summary_defaults(matrix_dir)
    n_timepoints = int(summary_defaults.get("n_timepoints", args.n_timepoints))
    history_burn_in_samples = int(
        summary_defaults.get("history_burn_in_samples", args.history_burn_in_samples)
    )
    bin_seconds = float(summary_defaults.get("bin_seconds", args.bin_seconds))
    patch_size_px = int(summary_defaults.get("patch_size_px", args.patch_size_px))
    rr100_version = str(summary_defaults.get("rr100_version", args.rr100_version))

    image_path = matrix_dir / "image_feature_table.csv"
    unit_path = matrix_dir / "unit_feature_table.csv"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing selected image table: {image_path}")
    if not unit_path.exists():
        raise FileNotFoundError(f"Missing unit feature table: {unit_path}")
    images = pd.read_csv(image_path)
    units = pd.read_csv(unit_path)
    if "image_index" not in images.columns:
        raise ValueError(f"{image_path} must contain image_index.")

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=Path(args.checkpoint_path),
        dataset_configs=Path(args.dataset_configs),
        population_spec_dir=Path(args.population_spec_dir),
        rr100_version=rr100_version,
        device=str(args.device),
        mcfarland_outputs=Path(args.mcfarland_outputs) if args.mcfarland_outputs is not None else None,
    )
    if int(scorer.n_units) != int(units.shape[0]):
        raise ValueError(f"RR100 scorer has {scorer.n_units} units but unit_feature_table has {units.shape[0]} rows.")

    ssi, expected, mean_rate, population, rows, timing = score_stabilized_images(
        scorer=scorer,
        images=images,
        frame_batch_size=int(args.frame_batch_size),
        n_timepoints=n_timepoints,
        bin_seconds=bin_seconds,
        patch_size_px=patch_size_px,
        history_burn_in_samples=history_burn_in_samples,
    )
    np.save(out_dir / OUTPUT_FILES["ssi"], ssi)
    np.save(out_dir / OUTPUT_FILES["expected"], expected)
    np.save(out_dir / OUTPUT_FILES["mean_rate"], mean_rate)
    np.save(out_dir / OUTPUT_FILES["population"], population)
    write_csv(out_dir / OUTPUT_FILES["table"], rows)

    payload = {
        "analysis": "backimage_real_trace_stabilized_baseline",
        "matrix_dir": matrix_dir,
        "out_dir": out_dir,
        "rr100_version": rr100_version,
        "n_images": int(images.shape[0]),
        "n_units": int(scorer.n_units),
        "n_timepoints": n_timepoints,
        "history_burn_in_samples": history_burn_in_samples,
        "scored_trace_samples": n_timepoints,
        "model_trace_samples": history_burn_in_samples + n_timepoints,
        "bin_seconds": bin_seconds,
        "patch_size_px": patch_size_px,
        "device": str(args.device),
        "frame_batch_size": int(args.frame_batch_size),
        **timing,
        "images_per_s": float(images.shape[0] / timing["elapsed_s"]) if timing["elapsed_s"] > 0.0 else None,
        "model_provenance": scorer.provenance,
        "outputs": {key: out_dir / name for key, name in OUTPUT_FILES.items()},
        "contract": (
            "Rows are selected images in image_feature_table order. Each row is a counterfactually stabilized "
            "zero-motion movie scored with the same time-resolved spatial SSI calculation and RR100 population "
            "view as the real-trace image x trace matrix."
        ),
    }
    write_json(out_dir / OUTPUT_FILES["summary"], payload)
    print(f"[fig4-stabilized-baseline] wrote stabilized baseline to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
