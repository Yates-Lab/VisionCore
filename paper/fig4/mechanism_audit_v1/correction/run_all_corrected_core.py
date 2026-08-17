#!/usr/bin/env python3
"""Run all four corrected core shards sequentially, then merge both banks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "paper/fig4/mechanism_audit_v1/correction/run_corrected_core.py"
MERGER = ROOT / "paper/fig4/mechanism_audit_v1/correction/merge_corrected_core.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--trace-batch-size", type=int, default=8)
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    data_root = "/home/jake/repos/DataYatesV1"
    env["PYTHONPATH"] = data_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for bank in ("true_history", "held_initial"):
        for start, stop in ((0, 50), (50, 100)):
            command = [
                sys.executable,
                str(RUNNER),
                "--bank",
                bank,
                "--image-start",
                str(start),
                "--image-stop",
                str(stop),
                "--device",
                args.device,
                "--frame-batch-size",
                str(args.frame_batch_size),
                "--trace-batch-size",
                str(args.trace_batch_size),
            ]
            print("RUNNING", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        subprocess.run(
            [sys.executable, str(MERGER), "--bank", bank], cwd=ROOT, env=env, check=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
