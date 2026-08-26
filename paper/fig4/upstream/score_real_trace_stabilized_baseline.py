#!/usr/bin/env python3
"""Score stabilized baselines for an exact-unit fixation response matrix."""

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
DEFAULT_MCFARLAND_OUTPUTS = ROOT / "scripts/mcfarland_outputs_mono.pkl"
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
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--population-version", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-configs", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, required=True)
    parser.add_argument(
        "--mcfarland-outputs",
        type=Path,
        default=DEFAULT_MCFARLAND_OUTPUTS,
    )
    parser.add_argument("--n-timepoints", type=int, default=40)
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


def selected_image_table_path(matrix_dir: Path) -> Path:
    """Use only images scored in a shard; merged matrices use the full table."""
    scored = Path(matrix_dir) / "scored_image_feature_table.csv"
    if scored.exists():
        return scored
    return Path(matrix_dir) / "image_feature_table.csv"


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
    bin_seconds = float(summary_defaults.get("bin_seconds", args.bin_seconds))
    patch_size_px = int(summary_defaults.get("patch_size_px", args.patch_size_px))
    population_version = str(summary_defaults.get("population_version", args.population_version))
    if population_version != str(args.population_version):
        raise ValueError(
            "matrix population version does not match the requested exact-unit population"
        )

    image_path = selected_image_table_path(matrix_dir)
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
        population_version=population_version,
        device=str(args.device),
        mcfarland_outputs=Path(args.mcfarland_outputs) if args.mcfarland_outputs is not None else None,
    )
    if int(scorer.n_units) != int(units.shape[0]):
        raise ValueError(f"Population scorer has {scorer.n_units} units but unit_feature_table has {units.shape[0]} rows.")

    ssi, expected, mean_rate, population, rows, timing = score_stabilized_images(
        scorer=scorer,
        images=images,
        frame_batch_size=int(args.frame_batch_size),
        n_timepoints=n_timepoints,
        bin_seconds=bin_seconds,
        patch_size_px=patch_size_px,
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
        "image_table": image_path,
        "image_indices": images["image_index"].astype(int).tolist(),
        "population_version": population_version,
        "n_images": int(images.shape[0]),
        "n_units": int(scorer.n_units),
        "n_timepoints": n_timepoints,
        "bin_seconds": bin_seconds,
        "patch_size_px": patch_size_px,
        "device": str(args.device),
        "frame_batch_size": int(args.frame_batch_size),
        **timing,
        "images_per_s": float(images.shape[0] / timing["elapsed_s"]) if timing["elapsed_s"] > 0.0 else None,
        "model_provenance": scorer.provenance,
        "outputs": {key: out_dir / name for key, name in OUTPUT_FILES.items()},
        "contract": (
            "Rows are selected images in the recorded image-table order. Each row is a counterfactually stabilized "
            "zero-motion movie scored with the same time-resolved spatial SSI calculation and declared population "
            "view as the real-trace image x trace matrix."
        ),
    }
    write_json(out_dir / OUTPUT_FILES["summary"], payload)
    print(f"[fig4-stabilized-baseline] wrote stabilized baseline to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
