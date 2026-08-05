#!/usr/bin/env python3
"""Plan or launch the Figure 4 BackImage real-trace SSI matrix.

The production matrix is expensive: 100 selected BackImage patches crossed with
1000 native fixation traces, split into two 50-image shards, then merged and
scored against a stabilized zero-motion baseline. This launcher is deliberately
safe by default: without --run-* flags it only writes a run plan/provenance JSON
and prints the commands that would be run.

The scorer body lives in this repo under upstream/real_trace_matrix. The
launcher still preflights every direct input before it will execute: the raw
BackImage window table, the McFarland readout artifact, the model checkpoint,
the frozen dataset config, and the RR100 population spec are data/model assets,
not git-tracked code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FIG4_DIR = ROOT / "paper" / "fig4"
UPSTREAM_DIR = FIG4_DIR / "upstream"

RUN_STEM = "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75_"
    "medoidPosthocminRepcomplete0p45_movieMedoid"
)
MODEL_CHECKPOINT_FILENAME = "epoch=147-val_bps_overall=0.5702.ckpt"
MODEL_CHECKPOINT_PATH = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/"
    "checkpoints/learned_resnet_none_convgru_gaussian_ddp_bs128_ds30_lr1e-3_wd1e-4_"
    f"corelrscale.5_warmup5/{MODEL_CHECKPOINT_FILENAME}"
)
STAGED_MODEL_CHECKPOINT_PATH = ROOT / "outputs/artifacts/model_checkpoints/fig4_twin" / MODEL_CHECKPOINT_FILENAME
MODEL_CHECKPOINT_SHA256 = "55d084aa0beb7d65614aecb9122edf7ad49c5799d370dbbd5dcf60b815c62de3"
DATASET_CONFIGS_REV = "e6c85ae"
DATASET_CONFIGS_MAIN_SHA256 = "c42906b90c340d64d35247baaa6b715e452d043dcff6439e02147aed6b7322d8"
RR100_SPEC_JSON_SHA256 = "d599fb0718faa363520a91b8f0819edafbff74ec501899b590e4061fef557f08"
RR100_SPEC_NPZ_SHA256 = "ffdbf4deee0d2bf4cc82d1bb7363e4271ee61be2cb6e87962bf6909a5b80c3d4"
WINDOW_FEATURES_SHA256 = "e8e2fa28c39d4d0222502bbe73fc221210260212fbed25bdc6c2e6c6217f73ba"
DIRECT_MATRIX_SOURCE_CSV_SHA256 = "ac2364e22ede162940a9ba7de5c8ab2c2ef2ea9c9985d851e302769d17c12567"
UNIT_TUNING_CSV_SHA256 = "7a506b617ccbda563cab1e7f10173f9015448f88ce71a1abec7b05dc8aaa92f2"

DEFAULT_SOURCE_CSV = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/backimage_image_fem_windows.csv"
)
DEFAULT_UNIT_TUNING_CSV = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_rr100_frequency_tuning_center_pixel_all_rr100_fast_nyquist_v1/"
    "sf_group_ssi_modulation_dynamic_log_gaussian_marginal_threshold_low0p05_high0p5_v1/"
    "dynamic_log_gaussian_marginal_sf_tuning_unit_groups.csv"
)
DEFAULT_WINDOW_FEATURES_CSV = (
    ROOT / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/window_features.csv"
)
DEFAULT_PRODUCTION_OUT_ROOT = ROOT / "outputs/active_sensing_movie_information" / RUN_STEM
DEFAULT_SMOKE_OUT_ROOT = ROOT / "outputs/figures/fig4/smoke/real_trace_matrix_smoke"
DEFAULT_DATASET_CONFIGS = UPSTREAM_DIR / "dataset_configs" / "multi_basic_120_long.yaml"
DEFAULT_POPULATION_SPEC_DIR = (
    ROOT / "outputs/redundancy_resolved_v1_twin/step1_activation_fingerprints"
)
MERGE_RUNNER = UPSTREAM_DIR / "merge_backimage_real_trace_ssi_matrix_shards.py"
DEFAULT_MATRIX_RUNNER = UPSTREAM_DIR / "score_real_trace_matrix.py"
DEFAULT_BASELINE_RUNNER = UPSTREAM_DIR / "score_real_trace_stabilized_baseline.py"
DEFAULT_MCFARLAND_OUTPUT_CANDIDATES = (
    ROOT / "scripts/mcfarland_outputs_mono.pkl",
    ROOT / "scripts/mcfarland_outputs.pkl",
)

MATRIX_RUNNER_ENV = "FIG4_REAL_TRACE_MATRIX_RUNNER"
BASELINE_RUNNER_ENV = "FIG4_STABILIZED_BASELINE_RUNNER"
CHECKPOINT_ENV = "FIG4_TWIN_CHECKPOINT"
DATASET_CONFIGS_ENV = "FIG4_DATASET_CONFIGS"
POPULATION_SPEC_ENV = "FIG4_RR100_POPULATION_SPEC_DIR"
MCFARLAND_OUTPUTS_ENV = "FIG4_MCFARLAND_OUTPUTS"


PROFILES: dict[str, dict[str, Any]] = {
    "production": {
        "n_images": 100,
        "n_traces": 1000,
        "seed": 20260717,
        "n_timepoints": 40,
        "bin_seconds": 1.0 / 120.0,
        "patch_size_px": 540,
        "image_contrast_quantile": 0.75,
        "image_min_orientation_coherence": 0.2,
        "image_min_drift_anisotropy": 0.0,
        "min_strong_contour_images": 40,
        "strong_contour_orientation_coherence_min": 0.5,
        "max_trace_path_length_arcmin": 350.0,
        "trace_scale_metric": "rendered_path_length_arcmin",
        "trace_sampling": "quantile",
        "min_microsaccade_traces": 200,
        "session_filter": "",
        "device": "cuda:1",
        "frame_batch_size": 16,
        "trace_batch_size": 8,
        "skip_benchmark": True,
        "shards": ((0, 50), (50, 100)),
        "expected_runtime": "Recovered run took about 10 hours per 50-image shard.",
    },
    "smoke": {
        "n_images": 1,
        "n_traces": 2,
        "seed": 20260717,
        "n_timepoints": 40,
        "bin_seconds": 1.0 / 120.0,
        "patch_size_px": 540,
        "image_contrast_quantile": 0.75,
        "image_min_orientation_coherence": 0.2,
        "image_min_drift_anisotropy": 0.0,
        "min_strong_contour_images": 0,
        "strong_contour_orientation_coherence_min": 0.5,
        "max_trace_path_length_arcmin": 350.0,
        "trace_scale_metric": "rendered_path_length_arcmin",
        "trace_sampling": "quantile",
        "min_microsaccade_traces": 0,
        "session_filter": "Allen_2022-02-16",
        "device": "cpu",
        "frame_batch_size": 4,
        "trace_batch_size": 1,
        "skip_benchmark": True,
        "shards": ((0, 1),),
        "expected_runtime": "Tiny scorer smoke target; intended to test schemas, not values.",
    },
}


@dataclass(frozen=True)
class CommandPlan:
    stage: str
    label: str
    argv: list[str]
    exec_argv: list[str] | None
    env: dict[str, str]
    cwd: Path
    ready: bool
    blocker: str | None
    hard_blocker: str | None
    expected_outputs: tuple[Path, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label": self.label,
            "argv": self.argv,
            "command": shlex.join(self.argv),
            "env": self.env,
            "cwd": str(self.cwd),
            "ready": self.ready,
            "blocker": self.blocker,
            "hard_blocker": self.hard_blocker,
            "expected_outputs": [str(path) for path in self.expected_outputs],
        }


def default_checkpoint_path() -> Path:
    if CHECKPOINT_ENV in os.environ:
        return Path(os.environ[CHECKPOINT_ENV])
    if STAGED_MODEL_CHECKPOINT_PATH.exists():
        return STAGED_MODEL_CHECKPOINT_PATH
    return MODEL_CHECKPOINT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--unit-tuning-csv", type=Path, default=DEFAULT_UNIT_TUNING_CSV)
    parser.add_argument("--window-features-csv", type=Path, default=DEFAULT_WINDOW_FEATURES_CSV)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument(
        "--matrix-runner",
        type=Path,
        default=Path(os.environ.get(MATRIX_RUNNER_ENV, str(DEFAULT_MATRIX_RUNNER))),
    )
    parser.add_argument(
        "--baseline-runner",
        type=Path,
        default=Path(os.environ.get(BASELINE_RUNNER_ENV, str(DEFAULT_BASELINE_RUNNER))),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=default_checkpoint_path(),
    )
    parser.add_argument(
        "--dataset-configs",
        type=Path,
        default=Path(os.environ.get(DATASET_CONFIGS_ENV, str(DEFAULT_DATASET_CONFIGS))),
    )
    parser.add_argument(
        "--population-spec-dir",
        type=Path,
        default=Path(os.environ.get(POPULATION_SPEC_ENV, str(DEFAULT_POPULATION_SPEC_DIR))),
    )
    parser.add_argument(
        "--mcfarland-outputs",
        type=Path,
        default=Path(os.environ[MCFARLAND_OUTPUTS_ENV]) if MCFARLAND_OUTPUTS_ENV in os.environ else None,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the profile's device.")
    parser.add_argument(
        "--session-filter",
        type=str,
        default=None,
        help="Override the profile's comma-separated source-session filter.",
    )
    parser.add_argument(
        "--only-shard",
        action="append",
        default=[],
        metavar="START:STOP",
        help="Restrict planned/launched score shards. May be repeated.",
    )
    parser.add_argument("--force", action="store_true", help="Pass --force to producers that support it.")
    parser.add_argument("--run-shards", action="store_true", help="Execute shard scoring commands.")
    parser.add_argument("--run-merge", action="store_true", help="Execute the in-repo shard merge.")
    parser.add_argument("--run-baseline", action="store_true", help="Execute the stabilized baseline command.")
    parser.add_argument("--run-all", action="store_true", help="Execute shards, merge, and baseline.")
    parser.add_argument(
        "--print-commands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a short preflight and shell commands.",
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Also hash large direct input CSVs that do not have fixed expected hashes.",
    )
    return parser.parse_args()


def cli_value(value: Any) -> str:
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_check(
    *,
    label: str,
    path: Path,
    required_for: tuple[str, ...],
    expected_sha256: str | None = None,
    hash_if_present: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    exists = path.exists()
    observed_sha256 = None
    if exists and (expected_sha256 is not None or hash_if_present):
        observed_sha256 = sha256_file(path)
    status = "present" if exists else "missing"
    if exists and expected_sha256 is not None:
        status = "ok" if observed_sha256 == expected_sha256 else "hash-mismatch"
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "status": status,
        "required_for": list(required_for),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "note": note,
    }


def git_info() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def parse_shard(text: str) -> tuple[int, int]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"Shard must be START:STOP, got {text!r}.")
    start_text, stop_text = text.split(":", 1)
    start = int(start_text)
    stop = int(stop_text)
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError(f"Invalid shard bounds: {text!r}.")
    return start, stop


def selected_shards(profile: dict[str, Any], only_shards: list[str]) -> tuple[tuple[int, int], ...]:
    if not only_shards:
        return tuple((int(a), int(b)) for a, b in profile["shards"])
    return tuple(parse_shard(text) for text in only_shards)


def shard_name(bounds: tuple[int, int]) -> str:
    start, stop = bounds
    return f"images_{int(start):03d}_{int(stop):03d}"


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = {
        DATASET_CONFIGS_ENV: str(Path(args.dataset_configs)),
        POPULATION_SPEC_ENV: str(Path(args.population_spec_dir)),
        CHECKPOINT_ENV: str(Path(args.checkpoint_path)),
        "MPLBACKEND": "Agg",
    }
    if args.mcfarland_outputs is not None:
        env[MCFARLAND_OUTPUTS_ENV] = str(Path(args.mcfarland_outputs))
    return env


def population_spec_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    spec_dir = Path(args.population_spec_dir)
    return (
        spec_dir / f"population_spec_{RR100_VERSION}.json",
        spec_dir / f"population_spec_{RR100_VERSION}.npz",
    )


def mcfarland_output_candidates(args: argparse.Namespace) -> tuple[Path, ...]:
    if args.mcfarland_outputs is not None:
        return (Path(args.mcfarland_outputs),)
    return DEFAULT_MCFARLAND_OUTPUT_CANDIDATES


def mcfarland_outputs_exist(args: argparse.Namespace) -> bool:
    return any(path.exists() for path in mcfarland_output_candidates(args))


def score_asset_blocker(args: argparse.Namespace, *, include_source_tables: bool) -> str | None:
    spec_json, spec_npz = population_spec_paths(args)
    missing: list[str] = []
    if include_source_tables:
        for label, path in (
            ("source CSV", Path(args.source_csv)),
            ("unit tuning CSV", Path(args.unit_tuning_csv)),
        ):
            if not path.exists():
                missing.append(f"{label}: {path}")
    for label, path in (
        ("model checkpoint", Path(args.checkpoint_path)),
        ("dataset config", Path(args.dataset_configs)),
        ("RR100 population JSON", spec_json),
        ("RR100 population NPZ", spec_npz),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    if not mcfarland_outputs_exist(args):
        candidates = ", ".join(str(path) for path in mcfarland_output_candidates(args))
        missing.append(f"McFarland outputs: {candidates}")
    if not missing:
        return None
    return "Missing required source/model asset(s): " + "; ".join(missing)


def runner_supports_source_asset_flags(runner: Path | None) -> bool:
    if runner is None:
        return False
    try:
        resolved = Path(runner).resolve()
    except OSError:
        resolved = Path(runner)
    return resolved in {DEFAULT_MATRIX_RUNNER.resolve(), DEFAULT_BASELINE_RUNNER.resolve()}


def source_asset_cli_args(args: argparse.Namespace, *, include_mcfarland: bool = True) -> list[str]:
    out = [
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--dataset-configs",
        str(args.dataset_configs),
        "--population-spec-dir",
        str(args.population_spec_dir),
    ]
    if include_mcfarland and args.mcfarland_outputs is not None:
        out.extend(["--mcfarland-outputs", str(args.mcfarland_outputs)])
    return out


def matrix_command(
    *,
    runner: Path | None,
    profile: dict[str, Any],
    shard: tuple[int, int],
    out_root: Path,
    args: argparse.Namespace,
) -> CommandPlan:
    start, stop = shard
    shard_dir = out_root / "shards" / shard_name(shard)
    runner_token = str(runner) if runner is not None else f"<{MATRIX_RUNNER_ENV}>"
    producer_args: list[str] = [
        "--source-csv",
        str(args.source_csv),
        "--unit-tuning-csv",
        str(args.unit_tuning_csv),
        "--out-dir",
        str(shard_dir),
        "--rr100-version",
        RR100_VERSION,
        "--n-images",
        cli_value(profile["n_images"]),
        "--n-traces",
        cli_value(profile["n_traces"]),
        "--seed",
        cli_value(profile["seed"]),
        "--n-timepoints",
        cli_value(profile["n_timepoints"]),
        "--bin-seconds",
        cli_value(profile["bin_seconds"]),
        "--patch-size-px",
        cli_value(profile["patch_size_px"]),
        "--image-contrast-quantile",
        cli_value(profile["image_contrast_quantile"]),
        "--image-min-orientation-coherence",
        cli_value(profile["image_min_orientation_coherence"]),
        "--image-min-drift-anisotropy",
        cli_value(profile["image_min_drift_anisotropy"]),
        "--min-strong-contour-images",
        cli_value(profile["min_strong_contour_images"]),
        "--strong-contour-orientation-coherence-min",
        cli_value(profile["strong_contour_orientation_coherence_min"]),
        "--max-trace-path-length-arcmin",
        cli_value(profile["max_trace_path_length_arcmin"]),
        "--trace-scale-metric",
        str(profile["trace_scale_metric"]),
        "--trace-sampling",
        str(profile["trace_sampling"]),
        "--min-microsaccade-traces",
        cli_value(profile["min_microsaccade_traces"]),
        "--device",
        str(profile["device"]),
        "--pilot-frame-batch-size",
        cli_value(profile["frame_batch_size"]),
        "--pilot-trace-batch-size",
        cli_value(profile["trace_batch_size"]),
        "--image-shard-start",
        cli_value(start),
        "--image-shard-stop",
        cli_value(stop),
        "--skip-benchmark",
    ]
    if bool(args.force):
        producer_args.append("--force")
    if runner_supports_source_asset_flags(runner):
        producer_args.extend(source_asset_cli_args(args))
    if str(profile.get("session_filter", "")).strip():
        producer_args.extend(["--session-filter", str(profile["session_filter"])])

    display_argv = ["uv", "run", "python", runner_token, *producer_args]
    runner_ready = runner is not None and Path(runner).exists()
    asset_blocker = score_asset_blocker(args, include_source_tables=True)
    ready = runner_ready and asset_blocker is None
    blocker = None
    hard_blocker = None
    if not runner_ready:
        blocker = f"Missing matrix scorer script: {runner_token}"
        hard_blocker = blocker
    elif asset_blocker is not None:
        blocker = asset_blocker
        hard_blocker = asset_blocker
    return CommandPlan(
        stage="score_shard",
        label=shard_name(shard),
        argv=display_argv,
        exec_argv=[sys.executable, str(runner), *producer_args] if runner is not None else None,
        env=base_env(args),
        cwd=ROOT,
        ready=ready,
        blocker=blocker,
        hard_blocker=hard_blocker,
        expected_outputs=(
            shard_dir / "summary.json",
            shard_dir / "ssi_matrix.npy",
            shard_dir / "expected_spikes_matrix.npy",
            shard_dir / "mean_rate_matrix.npy",
            shard_dir / "population_ssi.npy",
            shard_dir / "movie_feature_table.csv",
            shard_dir / "image_feature_table.csv",
            shard_dir / "trace_feature_table.csv",
            shard_dir / "unit_feature_table.csv",
            shard_dir / "trace_xy.npy",
        ),
    )


def merge_command(
    *,
    shards: tuple[tuple[int, int], ...],
    out_root: Path,
    args: argparse.Namespace,
) -> CommandPlan:
    shard_dirs = [out_root / "shards" / shard_name(shard) for shard in shards]
    merged_dir = out_root / "merged"
    producer_args = ["--out-dir", str(merged_dir), *[str(path) for path in shard_dirs]]
    if bool(args.force):
        producer_args.append("--force")
    display_argv = ["uv", "run", "python", str(MERGE_RUNNER), *producer_args]
    missing = [path for path in shard_dirs if not path.exists()]
    ready = MERGE_RUNNER.exists() and not missing
    blocker = None
    if not MERGE_RUNNER.exists():
        blocker = f"Missing in-repo merge helper: {MERGE_RUNNER}"
    elif missing:
        blocker = "Missing shard dir(s): " + ", ".join(str(path) for path in missing)
    return CommandPlan(
        stage="merge_shards",
        label="merged",
        argv=display_argv,
        exec_argv=[sys.executable, str(MERGE_RUNNER), *producer_args],
        env=base_env(args),
        cwd=ROOT,
        ready=ready,
        blocker=blocker,
        hard_blocker=None if MERGE_RUNNER.exists() else blocker,
        expected_outputs=(
            merged_dir / "summary.json",
            merged_dir / "ssi_matrix.npy",
            merged_dir / "expected_spikes_matrix.npy",
            merged_dir / "mean_rate_matrix.npy",
            merged_dir / "population_ssi.npy",
            merged_dir / "movie_feature_table.csv",
            merged_dir / "image_feature_table.csv",
            merged_dir / "trace_feature_table.csv",
            merged_dir / "unit_feature_table.csv",
            merged_dir / "trace_xy.npy",
        ),
    )


def baseline_command(
    *,
    runner: Path | None,
    profile: dict[str, Any],
    out_root: Path,
    args: argparse.Namespace,
) -> CommandPlan:
    merged_dir = out_root / "merged"
    runner_token = str(runner) if runner is not None else f"<{BASELINE_RUNNER_ENV}>"
    producer_args = [
        "--matrix-dir",
        str(merged_dir),
        "--out-dir",
        str(merged_dir),
        "--rr100-version",
        RR100_VERSION,
        "--n-timepoints",
        cli_value(profile["n_timepoints"]),
        "--bin-seconds",
        cli_value(profile["bin_seconds"]),
        "--patch-size-px",
        cli_value(profile["patch_size_px"]),
        "--device",
        str(profile["device"]),
        "--frame-batch-size",
        cli_value(profile["frame_batch_size"]),
    ]
    if bool(args.force):
        producer_args.append("--force")
    if runner_supports_source_asset_flags(runner):
        producer_args.extend(source_asset_cli_args(args))
    display_argv = ["uv", "run", "python", runner_token, *producer_args]
    runner_missing = runner is None or not Path(runner).exists()
    asset_blocker = score_asset_blocker(args, include_source_tables=False)
    ready = not runner_missing and merged_dir.exists() and asset_blocker is None
    blocker = None
    hard_blocker = None
    if runner_missing:
        blocker = f"Missing stabilized-baseline script: {runner_token}"
        hard_blocker = blocker
    elif asset_blocker is not None:
        blocker = asset_blocker
        hard_blocker = asset_blocker
    elif not merged_dir.exists():
        blocker = f"Missing merged matrix dir: {merged_dir}"
    return CommandPlan(
        stage="stabilized_baseline",
        label="merged",
        argv=display_argv,
        exec_argv=[sys.executable, str(runner), *producer_args] if runner is not None else None,
        env=base_env(args),
        cwd=ROOT,
        ready=ready,
        blocker=blocker,
        hard_blocker=hard_blocker,
        expected_outputs=(
            merged_dir / "stabilized_baseline_summary.json",
            merged_dir / "stabilized_ssi_by_image.npy",
            merged_dir / "stabilized_expected_spikes_by_image.npy",
            merged_dir / "stabilized_mean_rate_by_image.npy",
            merged_dir / "stabilized_population_ssi_by_image.npy",
            merged_dir / "stabilized_movie_feature_table.csv",
        ),
    )


def input_checks(args: argparse.Namespace, *, hash_inputs: bool) -> list[dict[str, Any]]:
    spec_json, spec_npz = population_spec_paths(args)
    mcfarland_candidates = mcfarland_output_candidates(args)
    mcfarland_present = [path for path in mcfarland_candidates if path.exists()]
    return [
        path_check(
            label="direct_matrix_source_csv",
            path=Path(args.source_csv),
            required_for=("score_shard",),
            expected_sha256=DIRECT_MATRIX_SOURCE_CSV_SHA256,
            note="Direct input opened by the matrix scorer.",
        ),
        path_check(
            label="unit_tuning_csv",
            path=Path(args.unit_tuning_csv),
            required_for=("score_shard",),
            expected_sha256=UNIT_TUNING_CSV_SHA256,
            note="Adds SF/group metadata to unit_feature_table.csv.",
        ),
        path_check(
            label="window_features_csv",
            path=Path(args.window_features_csv),
            required_for=("upstream_boundary",),
            expected_sha256=WINDOW_FEATURES_SHA256,
            note="True upstream fixation-window input; not produced by this compact Fig. 4 module.",
        ),
        path_check(
            label="model_checkpoint",
            path=Path(args.checkpoint_path),
            required_for=("score_shard", "stabilized_baseline"),
            expected_sha256=MODEL_CHECKPOINT_SHA256,
            note="Recovered from production logs; the old summaries recorded only the RR100 version.",
        ),
        {
            "label": "mcfarland_outputs",
            "path": str(mcfarland_present[0] if mcfarland_present else mcfarland_candidates[0]),
            "candidate_paths": [str(path) for path in mcfarland_candidates],
            "exists": bool(mcfarland_present),
            "status": "present" if mcfarland_present else "missing",
            "required_for": ["score_shard", "stabilized_baseline"],
            "expected_sha256": None,
            "observed_sha256": sha256_file(mcfarland_present[0]) if hash_inputs and mcfarland_present else None,
            "size_bytes": mcfarland_present[0].stat().st_size if mcfarland_present and mcfarland_present[0].is_file() else None,
            "note": "McFarland output metadata used to assemble the canonical spatial readout.",
        },
        path_check(
            label="dataset_configs_main",
            path=Path(args.dataset_configs),
            required_for=("score_shard", "stabilized_baseline"),
            expected_sha256=DATASET_CONFIGS_MAIN_SHA256,
            note=f"Frozen model-readout pin from VisionCore {DATASET_CONFIGS_REV}.",
        ),
        path_check(
            label="rr100_population_spec_json",
            path=spec_json,
            required_for=("score_shard", "stabilized_baseline"),
            expected_sha256=RR100_SPEC_JSON_SHA256,
        ),
        path_check(
            label="rr100_population_spec_npz",
            path=spec_npz,
            required_for=("score_shard", "stabilized_baseline"),
            expected_sha256=RR100_SPEC_NPZ_SHA256,
        ),
    ]


def build_manifest(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    out_root: Path,
    shards: tuple[tuple[int, int], ...],
    commands: list[CommandPlan],
) -> dict[str, Any]:
    return {
        "analysis": "fig4_backimage_real_trace_ssi_matrix_run_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "git": git_info(),
        "profile": args.profile,
        "run_stem": RUN_STEM,
        "out_root": str(out_root),
        "rr100_version": RR100_VERSION,
        "profile_parameters": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in profile.items()
        },
        "temporal_contract": {
            "model_history_frames": 32,
            "scored_trace_samples": int(profile["n_timepoints"]),
            "bin_seconds": float(profile["bin_seconds"]),
            "boundary": (
                "The twin stimulus uses a 32-frame model history, but the SSI "
                "analysis scores the 40 native FEM samples after lag alignment."
            ),
        },
        "recovered_production_facts": {
            "candidate_rows_after_gates": 2140,
            "selected_reliable_contour_images": 100,
            "selected_strong_contour_images": 53,
            "trace_bank_eligible_rows": 11683,
            "selected_microsaccade_traces": 200,
            "n_units": 100,
            "production_shards": ["images_000_050", "images_050_100"],
            "movies_per_production_shard": 50000,
            "total_movies": 100000,
        },
        "implementation_boundary": {
            "launcher_status": "in_visioncoremain",
            "merge_runner": str(MERGE_RUNNER),
            "scorer_runner_status": "in_visioncoremain",
            "matrix_runner": str(args.matrix_runner),
            "stabilized_baseline_runner_status": "in_visioncoremain",
            "baseline_runner": str(args.baseline_runner),
            "external_runner_override_env": {
                "matrix": MATRIX_RUNNER_ENV,
                "baseline": BASELINE_RUNNER_ENV,
            },
        },
        "inputs": input_checks(args, hash_inputs=bool(args.hash_inputs)),
        "commands": [command.manifest() for command in commands],
        "selected_shards": [shard_name(shard) for shard in shards],
        "execution_requested": {
            "run_shards": bool(args.run_shards or args.run_all),
            "run_merge": bool(args.run_merge or args.run_all),
            "run_baseline": bool(args.run_baseline or args.run_all),
        },
    }


def default_out_root(profile_name: str) -> Path:
    if profile_name == "smoke":
        return DEFAULT_SMOKE_OUT_ROOT
    return DEFAULT_PRODUCTION_OUT_ROOT


def default_plan_json(profile_name: str) -> Path:
    return ROOT / "outputs/figures/fig4/provenance" / f"real_trace_matrix_{profile_name}_plan.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_preflight(plan_json: Path, manifest: dict[str, Any]) -> None:
    print("FIGURE 4 REAL-TRACE MATRIX PLAN")
    print(f"profile : {manifest['profile']}")
    print(f"out_root: {manifest['out_root']}")
    print(f"plan    : {plan_json}")
    print("")
    print("inputs:")
    for check in manifest["inputs"]:
        status = check["status"]
        label = check["label"]
        path = check["path"]
        print(f"  [{status:13}] {label}: {path}")
    print("")
    print("commands:")
    for command in manifest["commands"]:
        status = "ready" if command["ready"] else "blocked"
        print(f"  [{status:7}] {command['stage']} {command['label']}")
        if command["blocker"]:
            print(f"      blocker: {command['blocker']}")
        print(f"      {command['command']}")


def requested_commands(args: argparse.Namespace, commands: list[CommandPlan]) -> list[CommandPlan]:
    run_shards = bool(args.run_shards or args.run_all)
    run_merge = bool(args.run_merge or args.run_all)
    run_baseline = bool(args.run_baseline or args.run_all)
    out: list[CommandPlan] = []
    for command in commands:
        if command.stage == "score_shard" and run_shards:
            out.append(command)
        elif command.stage == "merge_shards" and run_merge:
            out.append(command)
        elif command.stage == "stabilized_baseline" and run_baseline:
            out.append(command)
    return out


def execution_blocker(command: CommandPlan, args: argparse.Namespace) -> str | None:
    if command.hard_blocker is not None:
        return command.hard_blocker
    if command.ready:
        return None
    if command.stage == "merge_shards" and bool(args.run_shards or args.run_all):
        return None
    if command.stage == "stabilized_baseline" and bool(args.run_merge or args.run_all):
        return None
    return command.blocker


def run_command(command: CommandPlan) -> None:
    if command.exec_argv is None:
        raise RuntimeError(f"No executable argv for {command.stage} {command.label}.")
    env = dict(os.environ)
    env.update(command.env)
    print(f"[run] {command.stage} {command.label}: {shlex.join(command.argv)}", flush=True)
    subprocess.run(command.exec_argv, cwd=str(command.cwd), env=env, check=True)


def main() -> int:
    args = parse_args()
    profile = dict(PROFILES[args.profile])
    if args.device is not None:
        profile["device"] = str(args.device)
    if args.session_filter is not None:
        profile["session_filter"] = str(args.session_filter)
    out_root = Path(args.out_root) if args.out_root is not None else default_out_root(args.profile)
    plan_json = Path(args.plan_json) if args.plan_json is not None else default_plan_json(args.profile)
    shards = selected_shards(profile, list(args.only_shard))
    matrix_runner = Path(args.matrix_runner) if args.matrix_runner is not None else None
    baseline_runner = Path(args.baseline_runner) if args.baseline_runner is not None else None

    commands = [
        *[
            matrix_command(
                runner=matrix_runner,
                profile=profile,
                shard=shard,
                out_root=out_root,
                args=args,
            )
            for shard in shards
        ],
        merge_command(shards=shards, out_root=out_root, args=args),
        baseline_command(runner=baseline_runner, profile=profile, out_root=out_root, args=args),
    ]
    manifest = build_manifest(
        args=args,
        profile=profile,
        out_root=out_root,
        shards=shards,
        commands=commands,
    )
    write_json(plan_json, manifest)
    if bool(args.print_commands):
        print_preflight(plan_json, manifest)

    to_run = requested_commands(args, commands)
    if not to_run:
        return 0
    blocked = [(command, execution_blocker(command, args)) for command in to_run]
    blocked = [(command, blocker) for command, blocker in blocked if blocker is not None]
    if blocked:
        print("", file=sys.stderr)
        print("Refusing to run because requested commands are blocked:", file=sys.stderr)
        for command, blocker in blocked:
            print(f"  {command.stage} {command.label}: {blocker}", file=sys.stderr)
        return 2
    for command in to_run:
        run_command(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
