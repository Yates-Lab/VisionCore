#!/usr/bin/env python3
"""Regenerate, audit, and render Figure 3 from the production model spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.production_source_closure import source_closure  # noqa: E402


DEFAULT_SPEC = ROOT / "paper/model_selection/production_model.yaml"


FIGURE3_ENTRYPOINTS = (
    Path(__file__).resolve(),
    ROOT / "paper/model_selection/regen_fig3_caches.py",
    ROOT / "paper/fig3/audit_ablation_cache.py",
    ROOT / "paper/fig3/generate_figure3.py",
)


def production_source_closure() -> tuple[Path, ...]:
    return source_closure(ROOT, FIGURE3_ENTRYPOINTS)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(record: dict[str, Any], label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256(path)
    expected = str(record["sha256"])
    if observed != expected:
        raise ValueError(
            f"{label} digest mismatch: expected {expected}, observed {observed}"
        )
    return path


def existing(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def git_provenance() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_files": run("status", "--short").splitlines(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--canonical-observation-cache", type=Path, required=True)
    parser.add_argument("--covdecomp-cache", type=Path, required=True)
    parser.add_argument("--covdecomp-derived-cache", type=Path, required=True)
    parser.add_argument("--covdecomp-aligned-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layout", choices=("production", "manuscript"), default="production")
    parser.add_argument("--reuse-existing-caches", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = existing(args.model_spec, "production model spec")
    spec = yaml.safe_load(spec_path.read_text())
    checkpoint = verified(spec["checkpoint"], "production checkpoint")
    dataset = verified(
        spec["training"]["datasets"]["descriptive_all_gratings"],
        "Figure-3 dataset config",
    )
    canonical = existing(
        args.canonical_observation_cache, "canonical observation cache"
    )
    covdecomp = existing(args.covdecomp_cache, "covariance cache")
    covdecomp_derived = existing(
        args.covdecomp_derived_cache, "derived covariance cache"
    )
    covdecomp_aligned = existing(
        args.covdecomp_aligned_cache, "aligned covariance cache"
    )
    analysis_code = production_source_closure()
    if not args.dry_run and os.environ.get("CONDA_DEFAULT_ENV") != "yatesfv":
        raise RuntimeError(
            "Production Figure 3 must run in conda environment 'yatesfv'."
        )

    output = args.output_root.expanduser().resolve()
    cache_dir = output / "cache"
    figure_dir = output / "figures"
    stats_dir = output / "stats"
    model_cache = cache_dir / "fig3_model.pkl"
    ablation_cache = cache_dir / "fig3_ablation_inference.pkl"
    panel_a_cache = cache_dir / "fig3a_assets.pkl"
    audit_path = cache_dir / "ablation_audit.json"

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "FIG3_MODEL_CHECKPOINT": str(checkpoint),
            "FIG3_DATASET_CONFIGS": str(dataset),
            "FIG3_REFERENCE_CACHE": str(canonical),
            "FIG3_COVDECOMP_CACHE_PATH": str(covdecomp),
            "FIG3_COVDECOMP_DERIVED_CACHE_PATH": str(covdecomp_derived),
            "COVDECOMP_ALIGNED_CACHE_PATH": str(covdecomp_aligned),
            "FIG3_CACHE_PATH": str(model_cache),
            "FIG3_ABLATION_CACHE_PATH": str(ablation_cache),
            "FIG3_PANEL_A_CACHE_PATH": str(panel_a_cache),
            "FIG3_FIG_DIR": str(figure_dir),
            "FIG3_STAT_DIR": str(stats_dir),
            "FIG3_GPU": str(args.gpu),
            "FIG3_REUSE_EXISTING_CACHES": (
                "1" if args.reuse_existing_caches else "0"
            ),
            "MPLCONFIGDIR": str(Path("/tmp") / "mpl-production-figure3"),
        }
    )
    commands = [
        [sys.executable, str(ROOT / "paper/model_selection/regen_fig3_caches.py")],
        [
            sys.executable,
            str(ROOT / "paper/fig3/audit_ablation_cache.py"),
            str(ablation_cache),
            str(model_cache),
            "--out",
            str(audit_path),
        ],
        [sys.executable, str(ROOT / "paper/fig3/generate_figure3.py"), "--layout", args.layout],
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "production Figure 3 regeneration",
        "model_spec": str(spec_path),
        "model_spec_sha256": sha256(spec_path),
        "model_label": spec["label"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_config": str(dataset),
        "dataset_config_sha256": sha256(dataset),
        "canonical_observation_cache": str(canonical),
        "canonical_observation_cache_sha256": sha256(canonical),
        "covdecomp_cache": str(covdecomp),
        "covdecomp_cache_sha256": sha256(covdecomp),
        "covdecomp_derived_cache": str(covdecomp_derived),
        "covdecomp_derived_cache_sha256": sha256(covdecomp_derived),
        "covdecomp_aligned_cache": str(covdecomp_aligned),
        "covdecomp_aligned_cache_sha256": sha256(covdecomp_aligned),
        "analysis_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in analysis_code
        },
        "git": git_provenance(),
        "output_root": str(output),
        "environment": {
            key: environment[key]
            for key in environment
            if key.startswith("FIG3_") or key == "COVDECOMP_ALIGNED_CACHE_PATH"
        },
        "commands": [shlex.join(command) for command in commands],
        "status": "dry_run" if args.dry_run else "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise

    artifacts = {
        "model_cache": model_cache,
        "ablation_cache": ablation_cache,
        "panel_a_cache": panel_a_cache,
        "audit": audit_path,
        "figure_pdf": figure_dir / "figure3.pdf",
        "figure_png": figure_dir / "figure3.png",
    }
    for label, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"Expected {label} was not produced: {path}")
    manifest["status"] = "completed"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["artifacts"] = {
        label: {"path": str(path), "sha256": sha256(path)}
        for label, path in artifacts.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
