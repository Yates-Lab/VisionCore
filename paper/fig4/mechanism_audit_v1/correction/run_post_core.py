#!/usr/bin/env python3
"""Run the correction decision gate, then only gate-authorized follow-up work."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CORRECTION_DIR = ROOT / "paper/fig4/mechanism_audit_v1/correction"
OUT_DIR = ROOT / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1"


def run(script: str, *arguments: str, env: dict[str, str]) -> None:
    command = [sys.executable, str(CORRECTION_DIR / script), *arguments]
    print("RUNNING", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    data_root = "/home/jake/repos/DataYatesV1"
    env["PYTHONPATH"] = data_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # The report is deliberately written before any downstream mechanism run.
    run("analyze_core_correction.py", env=env)
    run("plot_core_correction.py", env=env)
    run("write_correction_report.py", env=env)
    statistics = json.loads((OUT_DIR / "statistics.json").read_text(encoding="utf-8"))
    decision = str(statistics["decision"])
    print(f"CORRECTION GATE: {decision}", flush=True)
    if decision != "CORE RESULT SURVIVES":
        print("STOPPING at the authorized correction gate; no mechanism analysis was run.", flush=True)
    else:
        run(
            "run_corrected_controlled_scaling.py",
            "--device",
            str(args.device),
            "--frame-batch-size",
            str(args.frame_batch_size),
            "--trace-batch-size",
            str(args.trace_batch_size),
            env=env,
        )
        run("analyze_controlled_scaling.py", env=env)
        run("plot_controlled_scaling.py", env=env)
        # Refresh the manifest after gate-authorized controlled outputs exist.
        run("write_correction_report.py", env=env)

    # Write a machine-readable completion audit, refresh the manifest so that
    # it includes that audit, then verify the final hashes without mutation.
    run("verify_correction_outputs.py", "--write", env=env)
    run("write_correction_report.py", env=env)
    run("verify_correction_outputs.py", env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
