#!/usr/bin/env python3
"""Audit the executable source graphs for fitting and Figures 3--4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.run_production_figure4 import (  # noqa: E402
    PRODUCTION_ENTRYPOINTS as FIGURE4_ENTRYPOINTS,
    production_source_closure as figure4_source_closure,
)
from paper.model_selection.run_production_figure3 import (  # noqa: E402
    FIGURE3_ENTRYPOINTS,
    production_source_closure as figure3_source_closure,
)
from training.run_three_stage_curriculum import (  # noqa: E402
    TRAINING_ENTRYPOINTS,
    production_source_closure as training_source_closure,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(entrypoints: tuple[Path, ...], closure: tuple[Path, ...]) -> dict:
    entrypoint_set = {path.resolve() for path in entrypoints}
    closure_set = set(closure)
    if not entrypoint_set <= closure_set:
        raise RuntimeError("a declared production entry point is absent from its closure")
    files = [str(path.relative_to(ROOT)) for path in closure]
    return {
        "entrypoints": [str(path.resolve().relative_to(ROOT)) for path in entrypoints],
        "source_files": files,
        "n_source_files": len(files),
        "source_lines": int(
            sum(len(path.read_text(encoding="utf-8").splitlines()) for path in closure)
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in closure
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    closures = {
        "three_stage_fitting": record(TRAINING_ENTRYPOINTS, training_source_closure()),
        "figure3": record(FIGURE3_ENTRYPOINTS, figure3_source_closure()),
        "figure4": record(FIGURE4_ENTRYPOINTS, figure4_source_closure()),
    }
    figure4_dir = ROOT / "paper/fig4/spatiotemporal_tuning"
    scoped = {path.resolve() for path in figure4_dir.glob("*.py")}
    declared = set(figure4_source_closure())
    orphans = sorted(str(path.relative_to(ROOT)) for path in scoped - declared)
    payload = {
        "schema_version": 1,
        "release_ready": not orphans,
        "pipelines": closures,
        "figure4_directory_orphans": orphans,
        "policy": (
            "explicit executable roots plus transitive repository-local imports; "
            "directory globs are audit inputs only and never define provenance"
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
