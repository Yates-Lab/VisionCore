#!/usr/bin/env python3
"""Fail-closed provenance and capacity audit for a production model spec."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from training.pl_modules.multidataset_model import MultiDatasetModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(record: dict[str, Any], *, label: str) -> Path:
    path = resolve(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(
            f"{label} digest mismatch: expected {record['sha256']}, got {observed}"
        )
    return path


def load_checkpoint_model(path: Path, *, verbose: bool = False) -> MultiDatasetModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hyperparameters = dict(checkpoint.get("hyper_parameters") or {})
    required = {"cfg_dir", "model_config_dict", "max_ds", "lr", "wd"}
    missing = sorted(required - set(hyperparameters))
    if missing:
        raise ValueError(f"Checkpoint lacks self-contained fields: {missing}")
    capture = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with capture:
        module = MultiDatasetModel(
            model_cfg=str(hyperparameters.get("model_cfg", "checkpoint-contained")),
            cfg_dir=str(hyperparameters["cfg_dir"]),
            lr=float(hyperparameters["lr"]),
            wd=float(hyperparameters["wd"]),
            max_ds=int(hyperparameters["max_ds"]),
            freeze_vision=bool(hyperparameters.get("freeze_vision", False)),
            compile_model=False,
            model_config_dict=hyperparameters["model_config_dict"],
            pretrained_load_heads=bool(
                hyperparameters.get("pretrained_load_heads", False)
            ),
            core_lr_scale=float(hyperparameters.get("core_lr_scale", 1.0)),
            selected_sessions=hyperparameters.get("selected_sessions"),
        )
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    stored_cids = hyperparameters.get("dataset_cids")
    if not isinstance(stored_cids, dict):
        raise ValueError("Checkpoint does not carry an exact session-to-CID table")
    current_cids = {
        name: list(config.get("cids", []))
        for name, config in zip(module.names, module.cfgs)
    }
    if current_cids != {str(k): list(v) for k, v in stored_cids.items()}:
        raise ValueError("Checkpoint CID table differs from the current dataset configs")
    return module


def architecture_summary(module: MultiDatasetModel) -> dict[str, Any]:
    parameters = dict(module.model.named_parameters())

    def count(*prefixes: str) -> int:
        return sum(
            parameter.numel()
            for name, parameter in parameters.items()
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        )

    phase_readouts = getattr(module.model, "phase_readouts", None)
    ranks = sorted(
        {int(readout.rank) for readout in phase_readouts or [] if hasattr(readout, "rank")}
    )
    if phase_readouts and len(ranks) != 1:
        raise ValueError(f"Expected one phase-readout rank across sessions, got {ranks}")
    readout_ranks = {int(getattr(readout, "rank", 1)) for readout in module.model.readouts}
    if len(readout_ranks) != 1:
        raise ValueError(f"Ordinary readout ranks differ across sessions: {readout_ranks}")
    return {
        "total": sum(parameter.numel() for parameter in parameters.values()),
        "visual_core": count("adapters", "frontend", "convnet", "recurrent"),
        "behavior_modulator": count("modulator"),
        "deep_readouts": count("readouts"),
        "phase_readouts": count("phase_readouts"),
        "phase_readout_rank": ranks[0] if ranks else 0,
        "readout_rank": readout_ranks.pop(),
        "output_channels": sum(len(config.get("cids", [])) for config in module.cfgs),
        "sessions": len(module.cfgs),
    }


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {observed!r}")


def audit(spec_path: Path, *, verbose: bool = False) -> dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text())
    checkpoint_path = verify_file(spec["checkpoint"], label="production checkpoint")
    run_manifest = verify_file(
        spec["training"]["run_manifest"], label="training run manifest"
    )
    verify_file(spec["training"]["curriculum"], label="curriculum")
    for index, record in enumerate(spec["training"]["stages"], start=1):
        verify_file(record, label=f"stage-{index} config")
    for name, record in spec["training"]["datasets"].items():
        verify_file(record, label=f"{name} dataset config")

    module = load_checkpoint_model(checkpoint_path, verbose=verbose)
    observed = architecture_summary(module)
    expected = spec["architecture"]
    require_equal(observed["phase_readout_rank"], expected["phase_readout_rank"], "rank")
    if "readout_rank" in expected:
        require_equal(observed["readout_rank"], expected["readout_rank"], "ordinary rank")
    require_equal(observed["output_channels"], expected["output_channels"], "outputs")
    for name, value in expected["parameters"].items():
        require_equal(observed[name], int(value), f"parameter count {name}")

    comparator_spec = spec["capacity_audit"]["comparator"]
    comparator_path = verify_file(
        {
            "path": comparator_spec["checkpoint"],
            "sha256": comparator_spec["checkpoint_sha256"],
        },
        label="capacity comparator checkpoint",
    )
    comparator = architecture_summary(
        load_checkpoint_model(comparator_path, verbose=verbose)
    )
    require_equal(
        comparator["phase_readout_rank"],
        int(comparator_spec["phase_readout_rank"]),
        "comparator rank",
    )
    for name, value in comparator_spec["parameters"].items():
        require_equal(comparator[name], int(value), f"comparator parameter count {name}")

    return {
        "schema_version": 1,
        "status": "passed",
        "spec": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "run_manifest": str(run_manifest),
        "production": {"label": spec["label"], **observed},
        "capacity_comparator": {"label": comparator_spec["label"], **comparator},
        "generalization_status": "requires matched Figure-3 evaluation caches",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "paper/model_selection/production_model.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = audit(args.spec.resolve(), verbose=args.verbose)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
