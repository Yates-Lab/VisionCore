#!/usr/bin/env python3
"""Stop the long run after true history, merge it, and refresh the checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CORRECTION_DIR = ROOT / "paper/fig4/mechanism_audit_v1/correction"
OUT_DIR = ROOT / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1"
SHARD_HASHES = (
    OUT_DIR
    / "core_ssi/real_trace_true_history_v1/images_050_100/hashes.json"
)


def run(*command: str, check: bool = True) -> None:
    print("RUNNING", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=check)


def main() -> int:
    print(f"Waiting for atomically complete true-history shard: {SHARD_HASHES}", flush=True)
    while not SHARD_HASHES.is_file():
        time.sleep(15)
    print("True-history shard is complete; stopping long-run orchestration before held scoring", flush=True)
    run("systemctl", "--user", "stop", "fig4-corrected-watchdog-v1.service", check=False)
    run("systemctl", "--user", "stop", "fig4-corrected-core-v1.service", check=False)
    python = sys.executable
    run(python, str(CORRECTION_DIR / "merge_corrected_core.py"), "--bank", "true_history")
    run(python, str(CORRECTION_DIR / "analyze_preliminary_true_only.py"))
    run(python, str(CORRECTION_DIR / "plot_preliminary_true_only.py"))
    run(python, str(CORRECTION_DIR / "write_true_history_primary_report.py"))
    print("COMPLETE_TRUE_HISTORY_PRIMARY__HELD_CONTROL_PENDING", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
