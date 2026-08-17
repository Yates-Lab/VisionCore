#!/usr/bin/env python3
"""Preserve selected ``last.ckpt`` epochs while long fits keep rotating top-k."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch


def checkpoint_epoch(path: Path) -> int:
    """Return a completed checkpoint's epoch, raising while a write is incomplete."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["epoch"])


def preserve_available(run_dir: Path, milestones: set[int]) -> set[int]:
    """Copy the current endpoint when it lands exactly on a requested milestone."""
    source = run_dir / "last.ckpt"
    if not source.exists():
        return set()
    epoch = checkpoint_epoch(source)
    if epoch not in milestones:
        return set()

    destination = run_dir / "analysis_candidates" / f"epoch={epoch:03d}-endpoint.ckpt"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
        print(f"Preserved {destination}", flush=True)
    return {epoch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--milestones",
        default="31,63,95,127,159,191,223,255",
        help="Comma-separated zero-based checkpoint epochs",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stop-after-hours", type=float, default=12.0)
    args = parser.parse_args()

    milestones = {int(value) for value in args.milestones.split(",") if value.strip()}
    if not milestones:
        raise ValueError("At least one milestone is required")
    run_dirs = [path.resolve() for path in args.run_dirs]
    missing = [str(path) for path in run_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Run directories do not exist: {missing}")

    found = {
        run_dir: {
            epoch
            for epoch in milestones
            if (run_dir / "analysis_candidates" / f"epoch={epoch:03d}-endpoint.ckpt").exists()
        }
        for run_dir in run_dirs
    }
    deadline = time.monotonic() + args.stop_after_hours * 3600.0
    while time.monotonic() < deadline:
        for run_dir in run_dirs:
            try:
                found[run_dir].update(
                    preserve_available(run_dir, milestones - found[run_dir])
                )
            except (EOFError, OSError, RuntimeError, KeyError) as error:
                print(f"Waiting for a complete checkpoint in {run_dir}: {error}", flush=True)
        if all(milestones.issubset(found[run_dir]) for run_dir in run_dirs):
            return
        time.sleep(args.poll_seconds)

    outstanding = {
        str(run_dir): sorted(milestones - found[run_dir]) for run_dir in run_dirs
    }
    print(f"Milestone monitor reached its time limit; outstanding={outstanding}", flush=True)


if __name__ == "__main__":
    main()
