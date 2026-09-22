#!/usr/bin/env python3
"""Run the reproducible three-stage native-240-Hz production curriculum.

The stages intentionally start new optimizers.  A completed stage contributes
only model parameters to the next stage; optimizer and scheduler state never
leak across a curriculum boundary.  Re-running an interrupted stage resumes
its own ``last.ckpt`` with the original optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.production_source_closure import source_closure  # noqa: E402


TRAINING_ENTRYPOINTS = (
    Path(__file__).resolve(),
    REPO_ROOT / "training/train_multidataset.py",
)
VAL_BPS_PATTERN = re.compile(r"val_bps_overall=([-+]?(?:\d+(?:\.\d*)?|\.\d+))\.ckpt$")
ALLOWED_REGULARIZERS = {
    "laplacian",
    "proximal_l1",
    "proximal_unit_norm",
    "proximal_clamp",
    "proximal_clamp_positive",
    "proximal_clamp_min",
}


def production_source_closure() -> tuple[Path, ...]:
    return source_closure(REPO_ROOT, TRAINING_ENTRYPOINTS)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint_val_bps(path: Path) -> float | None:
    match = VAL_BPS_PATTERN.search(path.name)
    return None if match is None else float(match.group(1))


def select_handoff_checkpoint(stage_dir: Path, policy: str) -> Path:
    if policy == "last":
        checkpoint = stage_dir / "last.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing final checkpoint: {checkpoint}")
        return checkpoint
    if policy != "best_val_bps":
        raise ValueError(f"Unknown handoff policy {policy!r}")
    candidates = [
        (score, path)
        for path in stage_dir.glob("*.ckpt")
        if (score := checkpoint_val_bps(path)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No validation-scored checkpoints in {stage_dir}")
    return max(candidates, key=lambda item: (item[0], item[1].name))[1]


def validate_curriculum(spec: dict[str, Any]) -> list[dict[str, Any]]:
    stages = spec.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("A production curriculum must contain exactly three stages")
    if [stage.get("key") for stage in stages] != ["core", "sparse_readout", "finetune"]:
        raise ValueError("Stage order must be core, sparse_readout, finetune")

    configs = []
    for stage in stages:
        model_path = _resolve(stage["model_config"])
        dataset_path = _resolve(stage["dataset_config"])
        if not model_path.is_file() or not dataset_path.is_file():
            raise FileNotFoundError(f"Missing model or dataset config for {stage['key']}")
        config = _read_yaml(model_path)
        if config.get("sampling_rate") != 240:
            raise ValueError(f"{stage['key']} is not native 240 Hz")
        if "stimulus_loss_weights" in config:
            raise ValueError("Condition loss weighting is outside the clean curriculum")
        forbidden = {"teacher", "distillation", "calibration", "auxiliary_loss"} & set(config)
        if forbidden:
            raise ValueError(f"Forbidden objective keys in {stage['key']}: {sorted(forbidden)}")
        regularizer_types = {term["type"] for term in config.get("regularization", [])}
        unexpected = regularizer_types - ALLOWED_REGULARIZERS
        if unexpected:
            raise ValueError(f"Unexpected regularizers in {stage['key']}: {sorted(unexpected)}")
        frequency_mask = config["convnet"]["params"].get("frequency_mask", {})
        if not frequency_mask.get("temporal") or not frequency_mask.get("stem_spatial"):
            raise ValueError(f"{stage['key']} must retain both frequency masks")
        configs.append(config)

    core, sparse, finetune = configs
    if "phase_readout" in core or "trainable_components" in core:
        raise ValueError("The scratch core stage must use the ordinary readout only")
    readout_families = ["readouts"]
    if sparse.get("phase_readout") is not None:
        readout_families.append("phase_readouts")
        sparse_rank = sparse["phase_readout"].get("params", {}).get("rank")
        if not isinstance(sparse_rank, int) or sparse_rank < 1:
            raise ValueError("Stage 2 phase-readout rank must be a positive integer")
    if sparse.get("trainable_components") != readout_families:
        raise ValueError("Stage 2 must freeze everything except its configured readout families")
    if sparse["readout"]["type"] == "sparse_gaussian_low_rank":
        rank = sparse["readout"].get("params", {}).get("rank")
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("Stage 2 ordinary-readout rank must be a positive integer")
    if "trainable_components" in finetune:
        raise ValueError("Stage 3 must unfreeze the entire model")
    if sparse.get("readout") != finetune.get("readout"):
        raise ValueError("Deep readout architecture changes between stages 2 and 3")
    if sparse.get("phase_readout") != finetune.get("phase_readout"):
        raise ValueError("Phase readout architecture changes between stages 2 and 3")
    if sparse["convnet"]["params"] != finetune["convnet"]["params"]:
        raise ValueError("Core architecture changes between stages 2 and 3")
    if float(stages[2]["core_lr_scale"]) >= 1.0:
        raise ValueError("Stage 3 must use a smaller core than readout learning rate")
    return configs


def stage_experiment_name(run_id: str, index: int, stage: dict[str, Any]) -> str:
    return f"{run_id}_{index + 1:02d}_{stage['key']}"


def build_stage_command(
    spec: dict[str, Any],
    run_id: str,
    index: int,
    gpu: int,
    run_root: Path,
    previous_checkpoint: Path | str | None,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    stage = spec["stages"][index]
    common = spec["common"]
    command = [
        sys.executable,
        str(_resolve(spec.get("trainer", "training/train_multidataset.py"))),
        "--model_config", str(_resolve(stage["model_config"])),
        "--dataset_configs_path", str(_resolve(stage["dataset_config"])),
        "--max_datasets", str(common["max_datasets"]),
        "--batch_size", str(common["batch_size"]),
        "--learning_rate", str(stage["learning_rate"]),
        "--core_lr_scale", str(stage["core_lr_scale"]),
        "--weight_decay", str(common["weight_decay"]),
        "--lr_scheduler", str(stage["lr_scheduler"]),
        "--warmup_epochs", str(stage["warmup_epochs"]),
        "--max_epochs", str(stage["max_epochs"]),
        "--accumulate_grad_batches", str(common["accumulate_grad_batches"]),
        "--steps_per_epoch", str(common["steps_per_epoch"]),
        "--gradient_clip_val", str(common["gradient_clip_val"]),
        "--precision", str(common["precision"]),
        "--dset_dtype", str(common["dset_dtype"]),
        "--num_workers", str(common["num_workers"]),
        "--limit_val_batches", str(common["limit_val_batches"]),
        "--check_val_every_n_epoch", str(stage["check_val_every_n_epoch"]),
        "--seed", str(common["seed"]),
        "--gpu", str(gpu),
        "--checkpoint_dir", str(run_root),
        "--project_name", str(spec["project_name"]),
        "--experiment_name", stage_experiment_name(run_id, index, stage),
        "--homogeneous_batches" if common["homogeneous_batches"] else "--no-homogeneous_batches",
        "--early_stopping" if common["early_stopping"] else "--no-early_stopping",
    ]
    if resume_checkpoint is not None:
        command.extend(["--ckpt_path", str(resume_checkpoint)])
    elif stage["initialization"] == "previous":
        if previous_checkpoint is None:
            raise ValueError(f"Stage {stage['key']} requires a previous checkpoint")
        command.extend(["--pretrained_checkpoint", str(previous_checkpoint)])
        if stage.get("pretrained_load_heads", False):
            command.append("--pretrained_load_heads")
    elif stage["initialization"] != "scratch":
        raise ValueError(f"Unknown initialization {stage['initialization']!r}")
    return command


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_files": run("status", "--short").splitlines(),
    }


def build_manifest(spec_path: Path, spec: dict[str, Any], run_id: str, gpu: int) -> dict[str, Any]:
    tracked = [spec_path]
    for stage in spec["stages"]:
        tracked.extend([_resolve(stage["model_config"]), _resolve(stage["dataset_config"])])
    tracked.extend(production_source_closure())
    hashes = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in dict.fromkeys(tracked)}
    return {
        "schema_version": 1,
        "pipeline": spec["name"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "git": _git_metadata(),
        "file_sha256": hashes,
        "stages": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="experiments/curricula/native240_sparse_phase_v1.yaml",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Run; default is a dry-run")
    parser.add_argument("--start-stage", choices=["core", "sparse_readout", "finetune"])
    parser.add_argument("--stop-after-stage", choices=["core", "sparse_readout", "finetune"])
    args = parser.parse_args()

    if args.execute and os.environ.get("CONDA_DEFAULT_ENV") != "yatesfv":
        raise RuntimeError(
            "Production training must run inside the yatesfv conda environment"
        )

    spec_path = _resolve(args.spec)
    spec = _read_yaml(spec_path)
    validate_curriculum(spec)
    run_root = _resolve(spec["checkpoint_root"]) / args.run_id
    stage_keys = [stage["key"] for stage in spec["stages"]]
    start_index = 0 if args.start_stage is None else stage_keys.index(args.start_stage)
    stop_index = len(stage_keys) - 1 if args.stop_after_stage is None else stage_keys.index(args.stop_after_stage)
    if start_index > stop_index:
        parser.error("--start-stage follows --stop-after-stage")

    manifest_path = run_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("pipeline") != spec["name"] or manifest.get("run_id") != args.run_id:
            raise ValueError("Existing run manifest belongs to another pipeline")
    else:
        manifest = build_manifest(spec_path, spec, args.run_id, args.gpu)

    previous: Path | str | None = None
    for index, stage in enumerate(spec["stages"]):
        name = stage_experiment_name(args.run_id, index, stage)
        stage_dir = run_root / name
        record = manifest["stages"].get(stage["key"], {})
        selected = record.get("handoff_checkpoint")
        if index < start_index:
            if selected is None or not Path(selected).is_file():
                selected = str(select_handoff_checkpoint(stage_dir, stage["handoff"]))
            previous = Path(selected)
            continue
        if index > stop_index:
            break

        if record.get("status") == "completed" and selected and Path(selected).is_file():
            print(f"[skip] {stage['key']}: {selected}")
            previous = Path(selected)
            continue
        resume = stage_dir / "last.ckpt"
        resume = resume if resume.is_file() else None
        prior_for_command: Path | str | None = previous
        if stage["initialization"] == "previous" and prior_for_command is None:
            prior_for_command = f"<previous:{stage_keys[index - 1]}>"
        command = build_stage_command(
            spec, args.run_id, index, args.gpu, run_root,
            prior_for_command, resume_checkpoint=resume,
        )
        print(f"\n[{index + 1}/3] {stage['key']}\n{shlex.join(command)}", flush=True)
        if not args.execute:
            previous = f"<handoff:{stage['key']}>"
            continue

        run_root.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "running",
            "experiment_name": name,
            "command": command,
            "resumed_from": None if resume is None else str(resume),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["stages"][stage["key"]] = record
        _atomic_json(manifest_path, manifest)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except BaseException:
            record["status"] = "interrupted"
            record["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(manifest_path, manifest)
            raise
        selected_path = select_handoff_checkpoint(stage_dir, stage["handoff"])
        record.update(
            status="completed",
            ended_utc=datetime.now(timezone.utc).isoformat(),
            handoff_policy=stage["handoff"],
            handoff_checkpoint=str(selected_path),
            handoff_sha256=_sha256(selected_path),
            handoff_val_bps=checkpoint_val_bps(selected_path),
        )
        _atomic_json(manifest_path, manifest)
        previous = selected_path

    if not args.execute:
        print("\nDry-run only. Add --execute inside the yatesfv environment to train.")
    elif previous is not None:
        print(f"\nCurriculum handoff: {previous}")


if __name__ == "__main__":
    main()
