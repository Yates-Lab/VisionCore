#!/usr/bin/env python3
"""Import coordinated dual-source residual readouts into a compatible twin.

The recipient supplies the complete target architecture, including its chosen
smooth residual visual branch.  The donor supplies only the independently
localized residual readouts on the mature base and auxiliary feature banks.
Every non-residual tensor shared by the checkpoints must be bitwise identical;
the donor's own residual visual branch is deliberately ignored because it may
have a different width from the recipient's branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


READOUT_PREFIXES = (
    "model.residual_readouts.",
    "model.auxiliary_residual_readouts.",
)
ALL_RESIDUAL_PREFIXES = READOUT_PREFIXES + (
    "model.residual_convnet.",
    "model.residual_visual_readouts.",
)


def compose_readout_states(recipient_state, donor_state):
    """Copy only the two mature-feature residual readout families."""
    donor_names = sorted(
        name for name in donor_state if name.startswith(READOUT_PREFIXES)
    )
    if not donor_names:
        raise RuntimeError("Donor checkpoint contains no dual-source readouts")
    missing = [name for name in donor_names if name not in recipient_state]
    if missing:
        raise RuntimeError(
            f"Recipient architecture lacks donor readout tensors: {missing[:10]}"
        )
    shape_errors = [
        (name, tuple(donor_state[name].shape), tuple(recipient_state[name].shape))
        for name in donor_names
        if tuple(donor_state[name].shape) != tuple(recipient_state[name].shape)
    ]
    if shape_errors:
        raise RuntimeError(f"Residual readout shape mismatch: {shape_errors[:10]}")

    shared_parent_names = sorted(
        name
        for name in set(recipient_state) & set(donor_state)
        if not name.startswith(ALL_RESIDUAL_PREFIXES)
    )
    drifted = [
        name
        for name in shared_parent_names
        if not torch.equal(recipient_state[name], donor_state[name])
    ]
    if drifted:
        raise RuntimeError(
            "Cannot compose checkpoints with different shared parents; "
            f"drifted tensors: {drifted[:10]}"
        )

    composed = {
        name: value.detach().cpu().clone()
        for name, value in recipient_state.items()
    }
    for name in donor_names:
        composed[name] = donor_state[name].detach().cpu().clone()
    changed = [
        name
        for name in donor_names
        if not torch.equal(recipient_state[name], composed[name])
    ]
    if not changed:
        raise RuntimeError("Donor readouts are identical to the recipient")
    return composed, {
        "shared_parent_tensors_exact": len(shared_parent_names),
        "donor_readout_tensors_copied": len(donor_names),
        "donor_readout_tensors_changed": len(changed),
        "copied_tensor_names": donor_names,
    }


def compose_checkpoints(
    recipient_path: Path,
    donor_path: Path,
    output_path: Path,
) -> dict:
    recipient_path = recipient_path.resolve()
    donor_path = donor_path.resolve()
    recipient = torch.load(recipient_path, map_location="cpu", weights_only=False)
    donor = torch.load(donor_path, map_location="cpu", weights_only=False)
    recipient_state = recipient.get("state_dict", recipient)
    donor_state = donor.get("state_dict", donor)
    composed_state, audit = compose_readout_states(recipient_state, donor_state)

    artifact = dict(recipient)
    artifact["state_dict"] = composed_state
    artifact["optimizer_states"] = []
    artifact["lr_schedulers"] = []
    artifact["callbacks"] = {}
    artifact.pop("selective_ryan_distillation", None)
    metadata = {
        "recipient_checkpoint": str(recipient_path),
        "readout_donor_checkpoint": str(donor_path),
        **audit,
    }
    artifact["residual_readout_composition"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return {"output_checkpoint": str(output_path.resolve()), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipient_checkpoint", type=Path)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = compose_checkpoints(
        args.recipient_checkpoint,
        args.donor,
        args.output,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
