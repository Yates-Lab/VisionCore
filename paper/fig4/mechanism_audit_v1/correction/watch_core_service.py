#!/usr/bin/env python3
"""Restart the resumable core scorer if its managed service exits early."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def service_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--completion-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    completion = Path(args.completion_path)
    while not completion.exists():
        if not service_active(str(args.unit)):
            print(f"Restarting inactive resumable service {args.unit}", flush=True)
            subprocess.run(
                ["systemctl", "--user", "restart", str(args.unit)],
                check=True,
            )
        time.sleep(max(10.0, float(args.poll_seconds)))
    print(f"Completion marker found: {completion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
