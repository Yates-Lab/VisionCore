#!/usr/bin/env python3
"""Render, audit, and provenance-bind Figure 4 from explicit artifacts.

Expensive measurements are produced by the dedicated commands documented in
``FIGURE4_PRODUCTION.md``. This runner is the sole manuscript-facing assembly
entry point: it verifies the model identity and every declared input, records
the source closure, renders the locked layout, runs the fail-closed audit, and
only writes results provenance after a release audit passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.production_source_closure import source_closure  # noqa: E402


DEFAULT_MODEL_SPEC = ROOT / "paper/model_selection/production_model.yaml"

PRODUCTION_ENTRYPOINTS = (
    Path(__file__).resolve(),
    ROOT / "paper/fig4/spatiotemporal_tuning/run_exact_cid_drifting_tuning.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/audit_exact_cid_drifting_tuning.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_exact_cid_figure4_contract.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_all_available_population_spec.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_real_fixation_bank.py",
    ROOT / "paper/fig4/upstream/run_real_trace_matrix.py",
    ROOT / "paper/fig4/upstream/score_real_trace_matrix.py",
    ROOT / "paper/fig4/upstream/merge_backimage_real_trace_ssi_matrix_shards.py",
    ROOT / "paper/fig4/upstream/score_real_trace_stabilized_baseline.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/audit_panel_a_exemplars.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_panel_b_population_path_length.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_rucci_ensemble_power.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_matrix_spectral_replay.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/compare_passband_path_length.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/analyze_top_passband_stage_trajectory.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/audit_eye_trace_filter.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_figure4.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/audit_revised_figure4_release.py",
    ROOT / "paper/fig4/spatiotemporal_tuning/build_figure4_results_provenance.py",
)


def production_source_closure() -> tuple[Path, ...]:
    """Return the executable Figure-4 source graph from declared roots."""
    return source_closure(ROOT, PRODUCTION_ENTRYPOINTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--panel-a-audit", type=Path, required=True)
    parser.add_argument("--panel-b-reduction", type=Path, required=True)
    parser.add_argument("--tuning-release-audit", type=Path, required=True)
    parser.add_argument("--tuning-visual-audit", type=Path, required=True)
    parser.add_argument("--tuning-contract", type=Path, required=True)
    parser.add_argument("--example-contract", type=Path, required=True)
    parser.add_argument("--population-spec", type=Path, required=True)
    parser.add_argument("--rucci-ensemble", type=Path, required=True)
    parser.add_argument("--passband-comparison", type=Path, required=True)
    parser.add_argument("--population-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--spectral-replay-tuning-table", type=Path, required=True)
    parser.add_argument("--stage-trajectory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--population-policy",
        choices=("validated", "all_checkpoint_available"),
        default="all_checkpoint_available",
    )
    parser.add_argument("--mode", choices=("smoke", "release"), default="release")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the recorded commands. Without this flag, emit a dry-run manifest.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} is missing: {value}")
    return value


def require_dir(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"{label} is missing: {value}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require_file(path, str(path)).read_text(encoding="utf-8"))


def resolve_contract_files(contract_dir: Path) -> dict[str, Path]:
    summary = load_json(contract_dir / "summary.json")
    files = summary.get("files", {})

    def resolve(key: str, fallback: str) -> Path:
        raw = files.get(key, contract_dir / fallback)
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        return require_file(path, f"tuning contract {key}")

    return {
        "tuning_table": resolve("tuning_table", "frequency_tuning_grouped.csv"),
        "tuning_summary": resolve("tuning_summary", "tuning_summary.csv"),
        "all_fits": resolve(
            "all_yu_fits" if "all_yu_fits" in files else "all_fits",
            "all_yu_fits.csv" if "all_yu_fits" in files else "all_validated_yu_fits.csv",
        ),
    }


def model_identity(spec_path: Path) -> tuple[dict[str, Any], Path, Path]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    checkpoint = require_file(Path(spec["checkpoint"]["path"]), "model checkpoint")
    dataset_record = spec["training"]["datasets"]["descriptive_all_gratings"]
    dataset = Path(dataset_record["path"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    dataset = require_file(dataset, "descriptive all-gratings dataset config")
    if sha256(checkpoint) != str(spec["checkpoint"]["sha256"]):
        raise ValueError("model checkpoint digest does not match the production spec")
    if sha256(dataset) != str(dataset_record["sha256"]):
        raise ValueError("dataset-config digest does not match the production spec")
    return spec, checkpoint, dataset


def git_provenance() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_files": run("status", "--short").splitlines(),
    }


def require_execution_environment(*, execute: bool, environment: str | None) -> None:
    """Keep dry runs portable while making every mutating run environment-exact."""
    if execute and environment != "yatesfv":
        raise RuntimeError("Production Figure 4 must run in conda environment 'yatesfv'.")


def validate_panel_h_sample_size(*, mode: str, n_movies: int) -> None:
    """Refuse to call the 40-movie smoke artifact a production release."""
    if mode == "release" and int(n_movies) < 100:
        raise ValueError(
            f"release mode requires at least 100 crossed Panel-H movies; found {n_movies}"
        )


def build_commands(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    figure_dir: Path,
    audit_dir: Path,
) -> list[list[str]]:
    example_fits = require_file(
        paths["example_contract"] / "crossed_example_fits.csv",
        "example fit contract",
    )
    render = [
        sys.executable,
        str(ROOT / "paper/fig4/spatiotemporal_tuning/build_figure4.py"),
        "--model-spec", str(paths["model_spec"]),
        "--panel-a-audit", str(paths["panel_a"]),
        "--panel-b-reduction", str(paths["panel_b"]),
        "--tuning-table", str(paths["tuning_table"]),
        "--tuning-summary", str(paths["tuning_summary"]),
        "--example-fits", str(example_fits),
        "--all-fits", str(paths["all_fits"]),
        "--rucci-ensemble", str(paths["rucci"]),
        "--population-shards", *map(str, paths["shards"]),
        "--stage-trajectory", str(paths["trajectory"]),
        "--out-dir", str(figure_dir),
        "--population-policy", args.population_policy,
        "--n-bootstrap", str(args.n_bootstrap),
        "--seed", str(args.seed),
    ]
    audit = [
        sys.executable,
        str(ROOT / "paper/fig4/spatiotemporal_tuning/audit_revised_figure4_release.py"),
        "--figure-dir", str(figure_dir),
        "--panel-a-audit", str(paths["panel_a"]),
        "--panel-b-reduction", str(paths["panel_b"]),
        "--tuning-release-audit", str(paths["tuning_release"]),
        "--tuning-visual-audit", str(paths["tuning_visual"]),
        "--tuning-contract", str(paths["tuning_contract"]),
        "--example-contract", str(paths["example_contract"]),
        "--population-spec", str(paths["population_spec"]),
        "--rucci-ensemble", str(paths["rucci"]),
        "--passband-comparison", str(paths["passband"]),
        "--population-shards", *map(str, paths["shards"]),
        "--spectral-replay-tuning-table", str(paths["spectral_tuning"]),
        "--stage-trajectory", str(paths["trajectory"]),
        "--out-dir", str(audit_dir),
    ]
    provenance = [
        sys.executable,
        str(ROOT / "paper/fig4/spatiotemporal_tuning/build_figure4_results_provenance.py"),
        "--production-audit", str(audit_dir / "production_audit.json"),
        "--panel-a-audit", str(paths["panel_a"]),
        "--panel-b-reduction", str(paths["panel_b"]),
        "--tuning-release-audit", str(paths["tuning_release"]),
        "--tuning-contract", str(paths["tuning_contract"]),
        "--rucci-ensemble", str(paths["rucci"]),
        "--passband-comparison", str(paths["passband"]),
        "--stage-trajectory", str(paths["trajectory"]),
        "--figure-dir", str(figure_dir),
        "--output", str(figure_dir / "results_provenance.json"),
    ]
    return [render, audit, provenance]


def main() -> int:
    args = parse_args()
    require_execution_environment(
        execute=bool(args.execute), environment=os.environ.get("CONDA_DEFAULT_ENV")
    )
    model_spec = require_file(args.model_spec, "production model spec")
    spec, checkpoint, dataset = model_identity(model_spec)
    tuning_contract = require_dir(args.tuning_contract, "tuning contract")
    contract_files = resolve_contract_files(tuning_contract)
    paths: dict[str, Any] = {
        "model_spec": model_spec,
        "panel_a": require_dir(args.panel_a_audit, "Panel-A audit"),
        "panel_b": require_dir(args.panel_b_reduction, "Panel-B reduction"),
        "tuning_release": require_dir(args.tuning_release_audit, "tuning release audit"),
        "tuning_visual": require_file(args.tuning_visual_audit, "tuning visual audit"),
        "tuning_contract": tuning_contract,
        "example_contract": require_dir(args.example_contract, "example contract"),
        "population_spec": require_file(args.population_spec, "population spec"),
        "rucci": require_dir(args.rucci_ensemble, "Rucci ensemble"),
        "passband": require_dir(args.passband_comparison, "passband comparison"),
        "shards": [require_file(path, "spectral replay shard") for path in args.population_shards],
        "spectral_tuning": require_file(
            args.spectral_replay_tuning_table, "spectral replay tuning table"
        ),
        "trajectory": require_dir(args.stage_trajectory, "stage trajectory"),
        **contract_files,
    }
    required_members = (
        paths["panel_a"] / "selected_example.npz",
        paths["panel_a"] / "summary.json",
        paths["panel_b"] / "binned_curves.csv",
        paths["panel_b"] / "unit_effects.npz",
        paths["panel_b"] / "summary.json",
        paths["tuning_release"] / "release_audit.json",
        paths["tuning_release"] / "unit_measurement_audit.csv",
        paths["rucci"] / "rucci_ensemble_power.npz",
        paths["rucci"] / "summary.json",
        paths["passband"] / "summary.json",
        paths["trajectory"] / "top_passband_stage_trajectory.npz",
        paths["trajectory"] / "summary.json",
    )
    for member in required_members:
        require_file(member, str(member))
    analysis_code = production_source_closure()

    trajectory_summary = load_json(paths["trajectory"] / "summary.json")
    n_movies = int(trajectory_summary.get("n_image_trace_pairs", 0))
    validate_panel_h_sample_size(mode=str(args.mode), n_movies=n_movies)

    output = args.output_root.expanduser().resolve()
    figure_dir = output / "figure"
    audit_dir = output / "audit"
    commands = build_commands(args=args, paths=paths, figure_dir=figure_dir, audit_dir=audit_dir)
    hashed_inputs = {
        "model_spec": model_spec,
        "checkpoint": checkpoint,
        "dataset_config": dataset,
        "population_spec": paths["population_spec"],
        "tuning_table": paths["tuning_table"],
        "tuning_summary": paths["tuning_summary"],
        "all_fits": paths["all_fits"],
        "spectral_replay_tuning_table": paths["spectral_tuning"],
        **{f"spectral_shard_{index}": path for index, path in enumerate(paths["shards"])},
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "production Figure 4 assembly and audit",
        "status": "planned" if not args.execute else "running",
        "mode": args.mode,
        "population_policy": args.population_policy,
        "model_label": spec["label"],
        "checkpoint_sha256": sha256(checkpoint),
        "panel_h_crossed_movies": n_movies,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in hashed_inputs.items()
        },
        "analysis_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in analysis_code
        },
        "commands": [shlex.join(command) for command in commands],
        "git": git_provenance(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.execute:
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["MPLCONFIGDIR"] = "/tmp/mpl-production-figure4"
    try:
        subprocess.run(commands[0], cwd=ROOT, env=environment, check=True)
        audit_result = subprocess.run(commands[1], cwd=ROOT, env=environment, check=False)
        audit = load_json(audit_dir / "production_audit.json")
        if args.mode == "release":
            if audit_result.returncode or not audit.get("release_ready"):
                raise RuntimeError("Figure-4 production audit did not pass")
            subprocess.run(commands[2], cwd=ROOT, env=environment, check=True)
            manifest["status"] = "completed_release_ready"
        else:
            failed = [gate for gate in audit.get("gates", []) if not gate.get("passed")]
            unexpected = [
                gate for gate in failed
                if "at least 100 crossed movies" not in str(gate.get("gate", ""))
            ]
            if unexpected:
                raise RuntimeError(
                    "smoke assembly exposed non-sample-size audit failures: "
                    + ", ".join(str(gate.get("gate")) for gate in unexpected)
                )
            manifest["status"] = "completed_smoke_not_release_ready"
        artifacts = [figure_dir / f"figure4.{suffix}" for suffix in ("pdf", "png", "svg")]
        artifacts.append(audit_dir / "production_audit.json")
        manifest["artifacts"] = {
            path.name: {"path": str(path), "sha256": sha256(path)} for path in artifacts
        }
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
