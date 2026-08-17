#!/usr/bin/env python3
"""Add identity-initialized residual components to a mature checkpoint.

The output is a self-contained Lightning checkpoint.  Every tensor inherited
from the parent must be bitwise identical, every new state tensor must belong
to an explicitly supported residual component, and each new output branch must
expose a zero-valued control parameter.  Existing residual components in a
parent checkpoint are loaded exactly; only components absent from the parent
are left at their identity initialization.  These checks make chained
architectural upgrades auditable identities rather than implicit partial
checkpoint loads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESIDUAL_COMPONENTS = (
    "residual_readouts",
    "auxiliary_residual_readouts",
    "residual_convnet",
    "residual_visual_readouts",
)


def missing_residual_components(parent_state) -> tuple[str, ...]:
    """Return residual components absent from a Lightning parent state.

    A chained upgrade may already contain one or more residual branches.  The
    complete pretrained loader must retain those branches and exclude only the
    genuinely new ones, whose output controls are initialized to zero.
    """
    normalized_names = {
        name[len("model.") :] if name.startswith("model.") else name
        for name in parent_state
    }
    return tuple(
        component
        for component in RESIDUAL_COMPONENTS
        if not any(
            name == component or name.startswith(component + ".")
            for name in normalized_names
        )
    )


def validate_upgrade_state(parent_state, child_state) -> dict:
    """Prove exact inheritance and zero initialization of the new branch."""
    missing = sorted(set(parent_state) - set(child_state))
    if missing:
        raise RuntimeError(f"Upgraded checkpoint lost parent tensors: {missing[:10]}")
    changed = [
        name
        for name, value in parent_state.items()
        if not torch.equal(value, child_state[name])
    ]
    if changed:
        raise RuntimeError(
            "Architectural upgrade changed inherited tensors: "
            f"{changed[:10]}"
        )
    added = sorted(set(child_state) - set(parent_state))
    allowed_prefixes = (
        "model.residual_readouts.",
        "model.auxiliary_residual_readouts.",
        "model.residual_convnet.",
        "model.residual_visual_readouts.",
    )
    unexpected = [
        name for name in added if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected new checkpoint tensors: {unexpected[:10]}")
    if not added:
        raise RuntimeError("Model config did not add residual-readout tensors")

    control_suffixes = (
        ".features.weight",
        ".base_scale",
        ".population_left",
    )
    controls = [name for name in added if name.endswith(control_suffixes)]
    if not controls:
        raise RuntimeError("Residual readout has no recognized identity control")
    nonzero = [
        name
        for name in controls
        if torch.count_nonzero(child_state[name]).item() != 0
    ]
    if nonzero:
        raise RuntimeError(
            "Residual identity controls were not initialized to zero: "
            f"{nonzero[:10]}"
        )
    dataset_indices = {
        int(name.split(".")[2]) for name in controls
    }
    return {
        "inherited_tensors_exact": len(parent_state),
        "new_residual_tensors": len(added),
        "zero_identity_controls": len(controls),
        "residual_datasets": len(dataset_indices),
        "added_tensor_names": added,
        "identity_control_names": controls,
    }


def initialize_checkpoint(
    parent_path: Path,
    model_config_path: Path,
    output_path: Path,
) -> dict:
    from models.config_loader import load_config
    from training.pl_modules import MultiDatasetModel

    parent_path = parent_path.resolve()
    model_config_path = model_config_path.resolve()
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_hparams = dict(parent.get("hyper_parameters", {}) or {})
    cfg_dir = parent_hparams.get("cfg_dir")
    if cfg_dir is None:
        raise ValueError("Parent checkpoint does not record cfg_dir")
    model_config = load_config(model_config_path)

    excluded_components = missing_residual_components(parent["state_dict"])
    wrapper = MultiDatasetModel(
        model_cfg=str(model_config_path),
        cfg_dir=str(cfg_dir),
        lr=float(parent_hparams.get("lr", 1.0e-4)),
        wd=float(parent_hparams.get("wd", 1.0e-5)),
        max_ds=int(parent_hparams.get("max_ds", 30)),
        pretrained_checkpoint=str(parent_path),
        pretrained_scope="complete",
        freeze_pretrained=True,
        pretrained_exclude_prefixes=",".join(excluded_components),
        model_config_dict=model_config,
    )
    child_state = {
        name: value.detach().cpu().clone()
        for name, value in wrapper.state_dict().items()
    }
    validation = validate_upgrade_state(parent["state_dict"], child_state)

    hparams = dict(wrapper.hparams)
    # The checkpoint contains every tensor it needs.  Loading it for inference
    # must not recursively reload the parent or reapply selective trainability.
    hparams.update(
        {
            "pretrained_checkpoint": None,
            "freeze_vision": False,
            "freeze_pretrained": False,
            "trainable_parameter_patterns": None,
            "pretrained_exclude_prefixes": None,
            "model_cfg": str(model_config_path),
            "model_config_dict": model_config,
            "core_lr_scale": parent_hparams.get("core_lr_scale", 1.0),
            "lr_scheduler": parent_hparams.get("lr_scheduler", "cosine_warmup"),
            "warmup_epochs": parent_hparams.get("warmup_epochs", 2),
            "restart_period": parent_hparams.get("restart_period"),
        }
    )
    metadata = {
        "parent_checkpoint": str(parent_path),
        "model_config": str(model_config_path),
        "new_residual_components": list(excluded_components),
        **validation,
    }
    artifact = dict(parent)
    artifact["state_dict"] = child_state
    artifact["hyper_parameters"] = hparams
    artifact["optimizer_states"] = []
    artifact["lr_schedulers"] = []
    artifact["callbacks"] = {}
    artifact.pop("selective_ryan_distillation", None)
    artifact["architecture_upgrade"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return {"output_checkpoint": str(output_path.resolve()), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = initialize_checkpoint(
        args.parent_checkpoint,
        args.model_config,
        args.output,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
