#!/usr/bin/env python3
"""Create one self-contained weight-averaged Lightning checkpoint.

This is intended for checkpoints from a single optimization trajectory.  It
validates the model/data identity, averages every floating state tensor, and
requires caller-declared frozen prefixes to be bit-identical.  Optimizer and
loop state are removed so the result is unambiguously an evaluation artifact,
not a resumable training checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _validate_identity(checkpoints, paths):
    reference = checkpoints[0].get("hyper_parameters", {}) or {}
    fields = ("model_config_dict", "dataset_cids", "cfg_dir", "max_ds")
    for path, checkpoint in zip(paths[1:], checkpoints[1:]):
        candidate = checkpoint.get("hyper_parameters", {}) or {}
        mismatched = [
            field for field in fields
            if _canonical(candidate.get(field)) != _canonical(reference.get(field))
        ]
        if mismatched:
            raise ValueError(
                f"Checkpoint identity mismatch for {path}: {mismatched}"
            )


def average_state_dicts(state_dicts, equal_prefixes=()):
    reference = state_dicts[0]
    reference_keys = tuple(reference)
    for index, state in enumerate(state_dicts[1:], start=1):
        if tuple(state) != reference_keys:
            missing = sorted(set(reference) - set(state))
            extra = sorted(set(state) - set(reference))
            raise ValueError(
                f"State dict {index} has different keys; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )

    averaged = {}
    for key in reference_keys:
        tensors = [state[key] for state in state_dicts]
        shape = tuple(tensors[0].shape)
        if any(tuple(tensor.shape) != shape for tensor in tensors[1:]):
            raise ValueError(f"Shape mismatch for {key}")
        require_equal = any(key.startswith(prefix) for prefix in equal_prefixes)
        if require_equal and any(
            not torch.equal(tensors[0], tensor) for tensor in tensors[1:]
        ):
            raise ValueError(f"Required-equal state changed: {key}")

        if tensors[0].is_floating_point() or tensors[0].is_complex():
            accumulator_dtype = (
                torch.complex128 if tensors[0].is_complex() else torch.float64
            )
            value = torch.stack(
                [tensor.to(dtype=accumulator_dtype) for tensor in tensors]
            ).mean(dim=0)
            averaged[key] = value.to(dtype=tensors[0].dtype)
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating state changed: {key}")
            averaged[key] = tensors[0].clone()
    return averaged


def build_soup(paths, equal_prefixes=()):
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    if any("state_dict" not in checkpoint for checkpoint in checkpoints):
        raise ValueError("Every input must be a Lightning checkpoint with state_dict")
    _validate_identity(checkpoints, paths)

    output = copy.deepcopy(checkpoints[0])
    output["state_dict"] = average_state_dicts(
        [checkpoint["state_dict"] for checkpoint in checkpoints],
        equal_prefixes=equal_prefixes,
    )
    for key in ("optimizer_states", "lr_schedulers", "callbacks", "loops"):
        output.pop(key, None)
    output["epoch"] = -1
    output["global_step"] = -1
    output["checkpoint_soup"] = {
        "method": "uniform_parameter_mean",
        "inputs": [str(path.resolve()) for path in paths],
        "input_epochs": [checkpoint.get("epoch") for checkpoint in checkpoints],
        "equal_prefixes": list(equal_prefixes),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--equal-prefix",
        action="append",
        default=[],
        help="State prefix that must be bit-identical across every input",
    )
    args = parser.parse_args()

    resolved = [path.resolve() for path in args.checkpoints]
    if len(resolved) < 2:
        parser.error("at least two checkpoints are required")
    if len(set(resolved)) != len(resolved):
        parser.error("checkpoint paths must be unique")
    for path in resolved:
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    output = build_soup(resolved, tuple(args.equal_prefix))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "inputs": output["checkpoint_soup"]["inputs"],
                "input_epochs": output["checkpoint_soup"]["input_epochs"],
                "equal_prefixes": output["checkpoint_soup"]["equal_prefixes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
