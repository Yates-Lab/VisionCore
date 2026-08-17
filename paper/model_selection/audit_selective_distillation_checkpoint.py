#!/usr/bin/env python3
"""Audit exact inheritance guarantees of a selective-distillation checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection.train_selective_ryan_distillation import (
    load_bps_archive,
    unit_teacher_weights,
)


UNIT_PARAMETER_PATTERN = re.compile(
    r"^model\.(?:readouts|auxiliary_readouts|residual_readouts|"
    r"auxiliary_residual_readouts|residual_visual_readouts)\.(\d+)\."
    r"|^model\.(?:output_modulator|distilled_output_modulator)\."
    r"(?:offset_layers|gain_layers)\.(\d+)\."
)


def audit_checkpoint(child_path: Path, parent_path: Path | None = None) -> dict:
    child_path = child_path.resolve()
    child = torch.load(child_path, map_location="cpu", weights_only=False)
    metadata = child.get("selective_ryan_distillation")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{child_path} has no selective-distillation metadata")

    recorded_parent = Path(metadata["student_checkpoint"]).resolve()
    if parent_path is None:
        parent_path = recorded_parent
    else:
        parent_path = parent_path.resolve()
        if parent_path != recorded_parent:
            raise RuntimeError(
                f"Requested parent {parent_path} differs from recorded {recorded_parent}"
            )
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_state = parent["state_dict"]
    child_state = child["state_dict"]
    if set(parent_state) != set(child_state):
        missing = sorted(set(parent_state) - set(child_state))[:10]
        extra = sorted(set(child_state) - set(parent_state))[:10]
        raise RuntimeError(f"State keys changed; missing={missing}, extra={extra}")

    trainable_names = set(metadata["trainable_names"])
    missing_trainable = sorted(trainable_names - set(child_state))
    if missing_trainable:
        raise RuntimeError(
            f"Recorded trainable names are absent from state: {missing_trainable[:10]}"
        )
    changed = [
        name
        for name in child_state
        if not torch.equal(parent_state[name], child_state[name])
    ]
    unexpected_changed = sorted(set(changed) - trainable_names)
    if unexpected_changed:
        raise RuntimeError(
            "Frozen tensors changed during refinement: "
            f"{unexpected_changed[:10]}"
        )

    isolation_report = metadata.get("unit_isolation_report", {})
    frozen_unit_rows = 0
    isolated_tensors = 0
    if isolation_report:
        student_archive = load_bps_archive(Path(metadata["student_evaluation"]))
        sessions = list(metadata["unit_weight_report"])
        cids_by_session = {
            session: student_archive[session][0].tolist() for session in sessions
        }
        weights, _ = unit_teacher_weights(
            Path(metadata["teacher_evaluation"]),
            Path(metadata["student_evaluation"]),
            sessions,
            cids_by_session,
            float(metadata["teacher_advantage_margin"]),
            float(metadata["teacher_advantage_scale"]),
        )
        for name, recorded in isolation_report.items():
            match = UNIT_PARAMETER_PATTERN.match(name)
            if match is None:
                raise RuntimeError(f"Unexpected isolated parameter name: {name}")
            dataset_idx = int(match.group(1) or match.group(2))
            if dataset_idx != int(recorded["dataset_idx"]):
                raise RuntimeError(f"Isolation dataset index changed for {name}")
            selected = weights[dataset_idx] > 0
            if int(selected.sum()) != int(recorded["n_trainable_rows"]):
                raise RuntimeError(f"Isolation selection count changed for {name}")
            frozen = ~selected
            if not torch.equal(child_state[name][frozen], parent_state[name][frozen]):
                raise RuntimeError(f"Unselected unit rows changed in {name}")
            frozen_unit_rows += int(frozen.sum())
            isolated_tensors += 1

    return {
        "child_checkpoint": str(child_path),
        "parent_checkpoint": str(parent_path),
        "state_tensors": len(child_state),
        "recorded_trainable_tensors": len(trainable_names),
        "changed_tensors": len(changed),
        "exactly_preserved_tensors": len(child_state) - len(changed),
        "unexpected_changed_tensors": 0,
        "unit_isolation_enabled": bool(isolation_report),
        "isolated_tensors": isolated_tensors,
        "exactly_preserved_unit_rows_across_tensors": frozen_unit_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("child_checkpoint", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = audit_checkpoint(args.child_checkpoint, args.parent_checkpoint)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
