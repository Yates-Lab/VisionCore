"""Small filesystem helpers for the production twininfo pipeline."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .common import to_jsonable


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    """Create the standard output tree used by every pipeline step."""
    dirs = {
        "cache": run_dir / "cache",
        "figures": run_dir / "figures",
        "metadata": run_dir / "metadata",
        "movies": run_dir / "movies",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dictionaries without pulling in pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write compressed arrays and create the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
