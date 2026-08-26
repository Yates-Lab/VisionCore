#!/usr/bin/env python3
"""Plan or execute the native-240 Figure 4 response-matrix replay.

The launcher has no checkpoint, dataset, or population fallback. The model is
resolved from the canonical production-model YAML, and the exact-unit
population is supplied by an explicit versioned JSON/NPZ pair. Dry-run is the
default; ``--execute`` runs deterministic image shards, merges them, and scores
the matched stabilized baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_SPEC = ROOT / "paper/model_selection/production_model.yaml"
SCORER = Path(__file__).resolve().parent / "score_real_trace_matrix.py"
MERGER = Path(__file__).resolve().parent / "merge_backimage_real_trace_ssi_matrix_shards.py"
BASELINE = Path(__file__).resolve().parent / "score_real_trace_stabilized_baseline.py"

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "n_images": 1,
        "n_traces": 2,
        "image_shard_size": 1,
        "device": "cpu",
        "frame_batch_size": 4,
        "trace_batch_size": 1,
    },
    "release": {
        "n_images": 40,
        "n_traces": 200,
        "image_shard_size": 10,
        "device": "cuda:0",
        "frame_batch_size": 32,
        "trace_batch_size": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--replay-matrix-dir", type=Path, required=True)
    parser.add_argument("--unit-table", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, required=True)
    parser.add_argument("--population-version", required=True)
    parser.add_argument(
        "--mcfarland-outputs",
        type=Path,
        default=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="release")
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--n-traces", type=int, default=None)
    parser.add_argument("--image-shard-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-batch-size", type=int, default=None)
    parser.add_argument("--trace-batch-size", type=int, default=None)
    parser.add_argument(
        "--only-stage",
        choices=("score", "merge", "baseline"),
        action="append",
        default=[],
        help="Execute only selected stages; may be repeated. Default is all stages.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the recorded commands. Without this flag, only plan them.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} is missing: {value}")
    return value


def require_dir(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"{label} is missing: {value}")
    return value


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_model_contract(spec_path: Path) -> dict[str, Any]:
    spec_path = require_file(spec_path, "production model spec")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    checkpoint_record = spec.get("checkpoint", {})
    dataset_record = (
        spec.get("training", {}).get("datasets", {}).get("descriptive_all_gratings", {})
    )
    checkpoint = require_file(Path(checkpoint_record.get("path", "")), "checkpoint")
    dataset = require_file(
        resolve_repo_path(dataset_record.get("path", "")), "native-240 dataset config"
    )
    if sha256(checkpoint) != str(checkpoint_record.get("sha256", "")):
        raise ValueError("checkpoint digest does not match the production model spec")
    if sha256(dataset) != str(dataset_record.get("sha256", "")):
        raise ValueError("dataset digest does not match the production model spec")

    config = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
    sampling = config.get("sampling", {}) or {}
    supervision = config.get("supervision", {}) or {}
    input_rate = int(sampling.get("target_rate", sampling.get("source_rate", 0)))
    output_rate = int(supervision.get("target_rate", input_rate))
    lags = np.asarray((config.get("keys_lags", {}) or {}).get("stim", []), dtype=int)
    if input_rate != 240 or output_rate != 240:
        raise ValueError(f"response replay requires native 240->240 Hz; got {input_rate}->{output_rate}")
    if not np.array_equal(lags, np.arange(60, dtype=int)):
        raise ValueError("response replay requires exactly 60 consecutive stimulus lags")
    return {
        "label": str(spec.get("label", "model")),
        "spec": spec_path,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "dataset": dataset,
        "dataset_sha256": sha256(dataset),
        "n_timepoints": 60,
        "bin_seconds": 1.0 / 240.0,
    }


def population_contract(spec_dir: Path, version: str) -> dict[str, Path]:
    directory = require_dir(spec_dir, "population-spec directory")
    if not str(version).strip():
        raise ValueError("population version must be non-empty")
    json_path = require_file(
        directory / f"population_spec_{version}.json", "population JSON"
    )
    npz_path = require_file(
        directory / f"population_spec_{version}.npz", "population NPZ"
    )
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    pooling_mode = str(metadata.get("pooling_mode", metadata.get("meta", {}).get("pooling_mode", "")))
    if pooling_mode != "exact_identity":
        raise ValueError(
            f"Figure 4 production requires exact_identity population membership; got {pooling_mode!r}"
        )
    return {"json": json_path, "npz": npz_path}


def validate_replay_bank(directory: Path, *, n_images: int, n_traces: int) -> dict[str, Path]:
    directory = require_dir(directory, "filtered replay bank")
    members = {
        "images": require_file(directory / "image_feature_table.csv", "replay image table"),
        "traces": require_file(directory / "trace_feature_table.csv", "replay trace table"),
        "trace_xy": require_file(directory / "trace_xy.npy", "filtered replay traces"),
        "provenance": require_file(directory / "trace_provenance.json", "trace provenance"),
    }
    image_table = pd.read_csv(members["images"])
    trace_table = pd.read_csv(members["traces"])
    trace_xy = np.load(members["trace_xy"], mmap_mode="r")
    if len(image_table) < int(n_images):
        raise ValueError(f"replay bank has {len(image_table)} images; requested {n_images}")
    if len(trace_table) < int(n_traces) or trace_xy.shape[0] < int(n_traces):
        raise ValueError(
            f"replay bank has {min(len(trace_table), trace_xy.shape[0])} traces; requested {n_traces}"
        )
    if trace_xy.ndim != 3 or trace_xy.shape[1:] != (60, 2):
        raise ValueError(f"filtered replay traces must have shape [trace, 60, 2], got {trace_xy.shape}")
    return members


def selected_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = dict(PROFILES[str(args.profile)])
    for key in (
        "n_images",
        "n_traces",
        "image_shard_size",
        "device",
        "frame_batch_size",
        "trace_batch_size",
    ):
        value = getattr(args, key)
        if value is not None:
            profile[key] = value
    for key in ("n_images", "n_traces", "image_shard_size", "frame_batch_size", "trace_batch_size"):
        if int(profile[key]) <= 0:
            raise ValueError(f"{key} must be positive")
        profile[key] = int(profile[key])
    profile["device"] = str(profile["device"])
    return profile


def shard_bounds(n_images: int, shard_size: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, min(start + int(shard_size), int(n_images)))
        for start in range(0, int(n_images), int(shard_size))
    )


def build_commands(
    *,
    contract: dict[str, Any],
    args: argparse.Namespace,
    profile: dict[str, Any],
    output_root: Path,
) -> dict[str, list[list[str]]]:
    shared = [
        "--population-version", str(args.population_version),
        "--checkpoint-path", str(contract["checkpoint"]),
        "--dataset-configs", str(contract["dataset"]),
        "--population-spec-dir", str(args.population_spec_dir),
        "--mcfarland-outputs", str(args.mcfarland_outputs),
    ]
    score_commands: list[list[str]] = []
    shard_dirs: list[Path] = []
    for start, stop in shard_bounds(profile["n_images"], profile["image_shard_size"]):
        shard_dir = output_root / "shards" / f"images_{start:03d}_{stop:03d}"
        shard_dirs.append(shard_dir)
        command = [
            sys.executable, str(SCORER),
            "--replay-matrix-dir", str(args.replay_matrix_dir),
            "--unit-tuning-csv", str(args.unit_table),
            "--out-dir", str(shard_dir),
            *shared,
            "--n-images", str(profile["n_images"]),
            "--n-traces", str(profile["n_traces"]),
            "--n-timepoints", str(contract["n_timepoints"]),
            "--bin-seconds", repr(contract["bin_seconds"]),
            "--patch-size-px", "540",
            "--trace-scale-metric", "rendered_path_length_arcmin",
            "--min-microsaccade-traces", "0",
            "--device", str(profile["device"]),
            "--pilot-frame-batch-size", str(profile["frame_batch_size"]),
            "--pilot-trace-batch-size", str(profile["trace_batch_size"]),
            "--image-shard-start", str(start),
            "--image-shard-stop", str(stop),
            "--skip-benchmark",
        ]
        if args.force:
            command.append("--force")
        score_commands.append(command)

    merged = output_root / "merged"
    merge_command = [
        sys.executable,
        str(MERGER),
        "--out-dir",
        str(merged),
        *map(str, shard_dirs),
    ]
    baseline_command = [
        sys.executable, str(BASELINE),
        "--matrix-dir", str(merged),
        "--out-dir", str(merged),
        *shared,
        "--n-timepoints", str(contract["n_timepoints"]),
        "--bin-seconds", repr(contract["bin_seconds"]),
        "--patch-size-px", "540",
        "--device", str(profile["device"]),
        "--frame-batch-size", str(profile["frame_batch_size"]),
    ]
    if args.force:
        merge_command.append("--force")
        baseline_command.append("--force")
    return {"score": score_commands, "merge": [merge_command], "baseline": [baseline_command]}


def require_execution_environment(*, execute: bool, environment: str | None) -> None:
    if execute and environment != "yatesfv":
        raise RuntimeError("Figure 4 response replay must run in conda environment 'yatesfv'.")


def git_provenance() -> dict[str, Any]:
    def run(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_files": run("status", "--short").splitlines(),
    }


def main() -> int:
    args = parse_args()
    require_execution_environment(
        execute=bool(args.execute), environment=os.environ.get("CONDA_DEFAULT_ENV")
    )
    contract = load_model_contract(args.model_spec)
    profile = selected_profile(args)
    population = population_contract(args.population_spec_dir, args.population_version)
    replay = validate_replay_bank(
        args.replay_matrix_dir,
        n_images=profile["n_images"],
        n_traces=profile["n_traces"],
    )
    unit_table = require_file(args.unit_table, "exact-unit table")
    mcfarland = require_file(args.mcfarland_outputs, "McFarland readout metadata")
    args.replay_matrix_dir = Path(args.replay_matrix_dir).expanduser().resolve()
    args.population_spec_dir = Path(args.population_spec_dir).expanduser().resolve()
    args.unit_table = unit_table
    args.mcfarland_outputs = mcfarland
    output_root = args.output_root.expanduser().resolve()
    commands = build_commands(
        contract=contract,
        args=args,
        profile=profile,
        output_root=output_root,
    )
    selected_stages = tuple(args.only_stage) or ("score", "merge", "baseline")
    source_files = (
        Path(__file__).resolve(),
        SCORER,
        MERGER,
        BASELINE,
        Path(__file__).resolve().parent / "real_trace_matrix/core.py",
        Path(__file__).resolve().parent / "real_trace_matrix/model.py",
    )
    inputs = {
        "model_spec": contract["spec"],
        "checkpoint": contract["checkpoint"],
        "dataset_config": contract["dataset"],
        "population_json": population["json"],
        "population_npz": population["npz"],
        "unit_table": unit_table,
        "mcfarland_outputs": mcfarland,
        **{f"replay_{key}": value for key, value in replay.items()},
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "native-240 exact-unit Figure 4 response replay",
        "status": "planned" if not args.execute else "running",
        "profile": args.profile,
        "profile_parameters": profile,
        "population_version": str(args.population_version),
        "population_contract": "exact_identity",
        "model_label": contract["label"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "dataset_config_sha256": contract["dataset_sha256"],
        "temporal_contract": {
            "input_rate_hz": 240,
            "output_rate_hz": 240,
            "history_frames": 60,
            "scored_trace_samples": 60,
            "analysis_interval_seconds": 0.25,
        },
        "selected_stages": list(selected_stages),
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in inputs.items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_files
        },
        "commands": {
            stage: [shlex.join(command) for command in stage_commands]
            for stage, stage_commands in commands.items()
        },
        "git": git_provenance(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.execute:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["MPLCONFIGDIR"] = "/tmp/mpl-figure4-response-matrix"
    try:
        for stage in ("score", "merge", "baseline"):
            if stage not in selected_stages:
                continue
            for command in commands[stage]:
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
