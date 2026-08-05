#!/usr/bin/env python3
"""Plan or run a replay audit against a historical Figure 4 real-trace matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UPSTREAM_DIR = Path(__file__).resolve().parent
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

from run_real_trace_matrix import (  # noqa: E402
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_MCFARLAND_OUTPUT_CANDIDATES,
    DEFAULT_POPULATION_SPEC_DIR,
    DEFAULT_UNIT_TUNING_CSV,
    MODEL_CHECKPOINT_PATH,
    ROOT,
    RR100_VERSION,
    RUN_STEM,
)


SCORER = UPSTREAM_DIR / "score_real_trace_matrix.py"
COMPARATOR = UPSTREAM_DIR / "compare_real_trace_matrix_outputs.py"
DEFAULT_REFERENCE_DIR = ROOT / "outputs/active_sensing_movie_information" / RUN_STEM / "merged"
DEFAULT_AUDIT_ROOT = ROOT / "outputs/figures/fig4/replay_audit"


@dataclass(frozen=True)
class CommandPlan:
    label: str
    argv: list[str]
    exec_argv: list[str]
    cwd: Path

    def manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "argv": self.argv,
            "command": shlex.join(self.argv),
            "cwd": str(self.cwd),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-stop", type=int, default=1)
    parser.add_argument("--trace-start", type=int, default=0)
    parser.add_argument("--trace-stop", type=int, default=100)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--bin-seconds", type=float, default=1.0 / 120.0)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--unit-tuning-csv", type=Path, default=DEFAULT_UNIT_TUNING_CSV)
    parser.add_argument("--checkpoint-path", type=Path, default=MODEL_CHECKPOINT_PATH)
    parser.add_argument("--dataset-configs", type=Path, default=DEFAULT_DATASET_CONFIGS)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument("--mcfarland-outputs", type=Path, default=None)
    parser.add_argument("--run", action="store_true", help="Execute the replay scorer and comparator.")
    parser.add_argument("--force", action="store_true", help="Pass --force to the replay scorer.")
    parser.add_argument(
        "--print-commands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print planned commands.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_bounds(start: int, stop: int, label: str) -> None:
    if int(start) < 0 or int(stop) <= int(start):
        raise ValueError(f"Invalid {label} bounds: {start}:{stop}.")


def safe_device_token(device: str) -> str:
    return str(device).replace(":", "").replace("/", "_")


def default_out_dir(args: argparse.Namespace) -> Path:
    return (
        DEFAULT_AUDIT_ROOT
        / f"images_{int(args.image_start):03d}_{int(args.image_stop):03d}"
        / f"traces_{int(args.trace_start):04d}_{int(args.trace_stop):04d}_{safe_device_token(args.device)}"
    )


def default_plan_json(out_dir: Path) -> Path:
    return out_dir / "replay_audit_plan.json"


def default_report_json(out_dir: Path) -> Path:
    return out_dir / "replay_comparison.json"


def mcfarland_path(args: argparse.Namespace) -> Path | None:
    if args.mcfarland_outputs is not None:
        return Path(args.mcfarland_outputs)
    for path in DEFAULT_MCFARLAND_OUTPUT_CANDIDATES:
        if path.exists():
            return path
    return None


def build_commands(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    report_json: Path,
    mcfarland_outputs: Path | None,
) -> list[CommandPlan]:
    n_images = int(args.image_stop) - int(args.image_start)
    n_traces = int(args.trace_stop) - int(args.trace_start)
    scorer_args = [
        "--replay-matrix-dir",
        str(args.reference_dir),
        "--unit-tuning-csv",
        str(args.unit_tuning_csv),
        "--out-dir",
        str(out_dir),
        "--rr100-version",
        RR100_VERSION,
        "--n-images",
        str(n_images),
        "--n-traces",
        str(n_traces),
        "--image-shard-start",
        str(args.image_start),
        "--image-shard-stop",
        str(args.image_stop),
        "--trace-shard-start",
        str(args.trace_start),
        "--trace-shard-stop",
        str(args.trace_stop),
        "--n-timepoints",
        str(args.n_timepoints),
        "--bin-seconds",
        repr(float(args.bin_seconds)),
        "--patch-size-px",
        str(args.patch_size_px),
        "--device",
        str(args.device),
        "--pilot-frame-batch-size",
        str(args.frame_batch_size),
        "--pilot-trace-batch-size",
        str(args.trace_batch_size),
        "--skip-benchmark",
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--dataset-configs",
        str(args.dataset_configs),
        "--population-spec-dir",
        str(args.population_spec_dir),
    ]
    if mcfarland_outputs is not None:
        scorer_args.extend(["--mcfarland-outputs", str(mcfarland_outputs)])
    if bool(args.force):
        scorer_args.append("--force")

    compare_args = [
        "--candidate-dir",
        str(out_dir),
        "--reference-dir",
        str(args.reference_dir),
        "--report-json",
        str(report_json),
        "--atol",
        repr(float(args.atol)),
        "--rtol",
        repr(float(args.rtol)),
    ]
    return [
        CommandPlan(
            label="replay_score",
            argv=["uv", "run", "python", str(SCORER), *scorer_args],
            exec_argv=[sys.executable, str(SCORER), *scorer_args],
            cwd=ROOT,
        ),
        CommandPlan(
            label="compare",
            argv=["uv", "run", "python", str(COMPARATOR), *compare_args],
            exec_argv=[sys.executable, str(COMPARATOR), *compare_args],
            cwd=ROOT,
        ),
    ]


def input_status(args: argparse.Namespace, *, out_dir: Path, mcfarland_outputs: Path | None) -> list[dict[str, Any]]:
    checks = [
        ("reference_dir", Path(args.reference_dir), True),
        ("unit_tuning_csv", Path(args.unit_tuning_csv), True),
        ("checkpoint_path", Path(args.checkpoint_path), True),
        ("dataset_configs", Path(args.dataset_configs), True),
        ("population_spec_dir", Path(args.population_spec_dir), True),
        ("mcfarland_outputs", mcfarland_outputs, True),
        ("out_dir", out_dir, False),
    ]
    rows: list[dict[str, Any]] = []
    for label, path, required in checks:
        rows.append(
            {
                "label": label,
                "path": str(path) if path is not None else None,
                "exists": bool(path is not None and Path(path).exists()),
                "required": bool(required),
            }
        )
    return rows


def print_plan(manifest: dict[str, Any]) -> None:
    print("FIGURE 4 REAL-TRACE REPLAY AUDIT")
    print(f"reference: {manifest['reference_dir']}")
    print(f"out_dir  : {manifest['out_dir']}")
    print(f"plan     : {manifest['plan_json']}")
    print(f"report   : {manifest['report_json']}")
    print("")
    print("inputs:")
    for row in manifest["inputs"]:
        status = "present" if row["exists"] else "missing"
        print(f"  [{status:7}] {row['label']}: {row['path']}")
    print("")
    print("commands:")
    for command in manifest["commands"]:
        print(f"  {command['label']}:")
        print(f"    {command['command']}")


def run_command(command: CommandPlan) -> None:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    print(f"[run] {command.label}: {shlex.join(command.argv)}", flush=True)
    subprocess.run(command.exec_argv, cwd=str(command.cwd), env=env, check=True)


def main() -> int:
    args = parse_args()
    check_bounds(args.image_start, args.image_stop, "image")
    check_bounds(args.trace_start, args.trace_stop, "trace")
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir(args)
    plan_json = Path(args.plan_json) if args.plan_json is not None else default_plan_json(out_dir)
    report_json = Path(args.report_json) if args.report_json is not None else default_report_json(out_dir)
    mcfarland_outputs = mcfarland_path(args)
    commands = build_commands(args, out_dir=out_dir, report_json=report_json, mcfarland_outputs=mcfarland_outputs)
    manifest = {
        "analysis": "fig4_real_trace_matrix_replay_audit_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dir": Path(args.reference_dir),
        "out_dir": out_dir,
        "plan_json": plan_json,
        "report_json": report_json,
        "image_bounds": [int(args.image_start), int(args.image_stop)],
        "trace_bounds": [int(args.trace_start), int(args.trace_stop)],
        "device": str(args.device),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "inputs": input_status(args, out_dir=out_dir, mcfarland_outputs=mcfarland_outputs),
        "commands": [command.manifest() for command in commands],
        "execution_requested": bool(args.run),
    }
    write_json(plan_json, manifest)
    if bool(args.print_commands):
        print_plan(manifest)
    if not bool(args.run):
        return 0
    missing = [row for row in manifest["inputs"] if bool(row["required"]) and not bool(row["exists"])]
    if missing:
        print("Refusing to run because required inputs are missing:", file=sys.stderr)
        for row in missing:
            print(f"  {row['label']}: {row['path']}", file=sys.stderr)
        return 2
    for command in commands:
        run_command(command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
