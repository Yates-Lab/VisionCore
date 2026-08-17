#!/usr/bin/env python3
"""Cache Ryan-twin rates on the exact native endpoints served to a student.

Ryan's production twin consumes 120-Hz, 51 x 51 stimulus histories whereas
the smooth Dekel student consumes native 240-Hz, 35 x 35 histories and is
supervised at 120 Hz.  The two data paths must therefore never be paired by
DataLoader position.  This script pairs them by the underlying dataset type
and physical 240-Hz endpoint:

    teacher index k  <->  student endpoint 2*k + supervision_phase

For the current causal factor-two contract the phase is one, so both models
predict the spike count accumulated over raw frames [2*k, 2*k+1].  Cached
rates are stored in the student's split order and can be consumed without
loading or differentiating through the teacher during student refinement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_tensor(value: torch.Tensor) -> str:
    """Hash a CPU tensor's dtype, shape, and contiguous bytes."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def unwrap(dataset):
    """Return the CombinedEmbeddedDataset under an optional Float32View."""
    return dataset.base if hasattr(dataset, "base") else dataset


def prepare_uint8_session(config: dict):
    """Prepare one session with the same memory/data contract as the main DM."""
    from DataYatesV1.utils.data.loading import remove_pixel_norm
    from models.data import prepare_data
    from models.data.datasets import CombinedEmbeddedDataset
    from training.utils import Float32View

    config, norm_removed = remove_pixel_norm(copy.deepcopy(config))
    train, val, test, _ = prepare_data(config, strict=True, return_test=True)
    for dataset in (train, val, test):
        if norm_removed:
            dataset.cast(torch.uint8, target_keys=["stim"])
    # The two models' support-dependent validity filters can remove different
    # trials before seeded splitting.  Consequently a physical endpoint may
    # be in the student's train split but Ryan's val/test split.  Teacher
    # inference therefore uses the disjoint union of all teacher splits and is
    # mapped into whichever student split is being cached.  The student's
    # split remains authoritative; no student response from another split is
    # ever exposed to refinement.
    all_indices = []
    for type_index in range(train.n_dsets):
        all_indices.append(
            torch.unique(
                torch.cat(
                    [
                        split.dset_inds[type_index].long()
                        for split in (train, val, test)
                    ]
                ),
                sorted=True,
            )
        )
    all_data = CombinedEmbeddedDataset(
        train.dsets, all_indices, train.keys_lags, device=train.device
    )
    return {
        "train": Float32View(train, norm_removed, float16=False),
        "val": Float32View(val, norm_removed, float16=False),
        "test": Float32View(test, norm_removed, float16=False),
        "all": Float32View(all_data, norm_removed, float16=False),
    }


def teacher_to_student_positions(
    teacher_dataset,
    student_dataset,
    factor: int,
    phase: int,
) -> torch.Tensor:
    """Map every teacher split position to a student split position or -1."""
    teacher = unwrap(teacher_dataset)
    student = unwrap(student_dataset)
    if teacher.n_dsets != student.n_dsets:
        raise RuntimeError("Teacher and student contain different dataset types")

    mapping = torch.full((len(teacher),), -1, dtype=torch.long)
    for type_index, (teacher_raw, student_raw) in enumerate(
        zip(teacher.dsets, student.dsets)
    ):
        teacher_name = teacher_raw.metadata.get("name")
        student_name = student_raw.metadata.get("name")
        if teacher_name != student_name:
            raise RuntimeError(
                f"Dataset type order differs: {teacher_name!r} != {student_name!r}"
            )

        student_positions = torch.nonzero(
            student.inds[:, 0] == type_index, as_tuple=True
        )[0]
        student_indices = student.inds[student_positions, 1].long()
        teacher_positions = torch.nonzero(
            teacher.inds[:, 0] == type_index, as_tuple=True
        )[0]
        teacher_indices = teacher.inds[teacher_positions, 1].long()
        endpoints = factor * teacher_indices + phase

        lookup_size = max(
            int(student_indices.max()) + 1 if student_indices.numel() else 0,
            int(endpoints.max()) + 1 if endpoints.numel() else 0,
        )
        lookup = torch.full((lookup_size,), -1, dtype=torch.long)
        lookup[student_indices] = student_positions
        valid = endpoints < lookup.numel()
        mapped = torch.full_like(endpoints, -1)
        mapped[valid] = lookup[endpoints[valid]]
        mapping[teacher_positions] = mapped

    nonnegative = mapping[mapping >= 0]
    if nonnegative.numel() != torch.unique(nonnegative).numel():
        raise RuntimeError("Teacher-to-student endpoint mapping is not one-to-one")
    return mapping


def verify_supervision_contract(
    teacher_dataset,
    student_dataset,
    mapping: torch.Tensor,
    max_examples: int = 1024,
) -> dict[str, float]:
    """Check response, mask, and behavior equality at mapped endpoints."""
    teacher_positions = torch.nonzero(mapping >= 0, as_tuple=True)[0]
    if not teacher_positions.numel():
        raise RuntimeError("No teacher/student endpoints overlap")
    if teacher_positions.numel() > max_examples:
        select = torch.linspace(
            0, teacher_positions.numel() - 1, max_examples
        ).round().long()
        teacher_positions = teacher_positions[select]
    student_positions = mapping[teacher_positions]
    teacher_batch = teacher_dataset[teacher_positions]
    student_batch = student_dataset[student_positions]

    result = {}
    for key in ("robs", "dfs", "behavior"):
        if key not in teacher_batch or key not in student_batch:
            continue
        teacher_value = teacher_batch[key].float()
        student_value = student_batch[key].float()
        if teacher_value.shape != student_value.shape:
            raise RuntimeError(
                f"Mapped {key} shapes differ: {teacher_value.shape} != "
                f"{student_value.shape}"
            )
        difference = (teacher_value - student_value).abs()
        maximum = float(difference.max()) if difference.numel() else 0.0
        result[f"{key}_max_abs_difference"] = maximum
        # The validity masks are deliberately not identical: Ryan constructs
        # them after global 120-Hz downsampling with 32 lags, whereas the
        # native-rate student constructs them before causal supervision
        # binning with its own support.  Record that disagreement, but use the
        # student's dfs during refinement.  Responses and behavior, in
        # contrast, must describe the identical physical supervision bin.
        tolerance = 0.0 if key == "robs" else 2.0e-5
        if key != "dfs" and maximum > tolerance:
            raise RuntimeError(
                f"Mapped {key} contract differs (max abs {maximum:g} > {tolerance:g})"
            )
    return result


def cache_split(
    teacher,
    dataset_idx: int,
    teacher_dataset,
    student_dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    factor: int,
    phase: int,
):
    """Run the teacher once and return rates aligned to student split order."""
    mapping = teacher_to_student_positions(
        teacher_dataset, student_dataset, factor=factor, phase=phase
    )
    contract = verify_supervision_contract(
        teacher_dataset, student_dataset, mapping
    )
    n_units = int(student_dataset[0]["robs"].shape[-1])
    rates = torch.full(
        (len(student_dataset), n_units), float("nan"), dtype=torch.bfloat16
    )
    loader = DataLoader(
        teacher_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    offset = 0
    teacher.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            count = int(batch["stim"].shape[0])
            student_positions = mapping[offset : offset + count]
            valid = student_positions >= 0
            if valid.any():
                batch = {
                    key: value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                    for key, value in batch.items()
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = teacher(
                        batch["stim"],
                        dataset_idx,
                        batch.get("behavior"),
                        batch.get("history"),
                        batch.get("output_behavior"),
                    )
                rates[student_positions[valid]] = prediction[valid].bfloat16().cpu()
            offset += count
            if (batch_index + 1) % 100 == 0:
                print(
                    f"  cached {offset:,}/{len(teacher_dataset):,} teacher rows",
                    flush=True,
                )
    if offset != len(teacher_dataset):
        raise RuntimeError(f"Served {offset} teacher rows, expected {len(teacher_dataset)}")

    covered = torch.isfinite(rates).all(dim=1)
    partial = torch.isfinite(rates).any(dim=1) & ~covered
    if partial.any():
        raise RuntimeError("Some student rows received only a subset of teacher units")
    return rates, {
        "student_rows": len(student_dataset),
        "teacher_rows": len(teacher_dataset),
        "matched_rows": int(covered.sum()),
        "coverage_fraction": float(covered.float().mean()),
        "student_inds_sha256": sha256_tensor(unwrap(student_dataset).inds),
        **contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_checkpoint", type=Path)
    parser.add_argument("--teacher-dataset-config", type=Path, required=True)
    parser.add_argument("--student-dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--endpoint-factor",
        type=int,
        default=None,
        help=(
            "Override the teacher-index to student-endpoint factor. The "
            "default is the historical Ryan-to-native causal factor. Use 1 "
            "when caching a same-grid parent for prediction preservation."
        ),
    )
    parser.add_argument(
        "--endpoint-phase",
        type=int,
        default=None,
        help=(
            "Override the student endpoint phase. The default is the "
            "student supervision phase; same-grid parent caches use 0."
        ),
    )
    parser.add_argument(
        "--splits", default="train", help="Comma-separated train,val,test"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from eval.load_twin import load_twin
    from models.config_loader import load_dataset_configs

    requested_splits = tuple(
        value.strip() for value in args.splits.split(",") if value.strip()
    )
    if not requested_splits or not set(requested_splits) <= {"train", "val", "test"}:
        raise ValueError(f"Invalid --splits value {args.splits!r}")

    teacher_configs = load_dataset_configs(args.teacher_dataset_config.resolve())
    student_configs = load_dataset_configs(args.student_dataset_config.resolve())
    teacher_configs = teacher_configs[: args.max_datasets]
    student_configs = student_configs[: args.max_datasets]
    teacher_names = [config["session"] for config in teacher_configs]
    student_names = [config["session"] for config in student_configs]
    if teacher_names != student_names:
        raise RuntimeError("Teacher and student session order differs")
    for teacher_config, student_config in zip(teacher_configs, student_configs):
        if list(teacher_config["cids"]) != list(student_config["cids"]):
            raise RuntimeError(
                f"Cell identities differ for {teacher_config['session']}"
            )

    first_student = student_configs[0]
    sampling = first_student.get("sampling") or {}
    supervision = first_student.get("supervision") or {}
    source_rate = int(sampling.get("source_rate", 240))
    target_rate = int(supervision.get("target_rate", source_rate))
    if source_rate % target_rate:
        raise ValueError("Student supervision rate must divide source rate")
    factor = source_rate // target_rate
    phase = int(supervision.get("phase", factor - 1))
    if args.endpoint_factor is not None:
        factor = int(args.endpoint_factor)
    if args.endpoint_phase is not None:
        phase = int(args.endpoint_phase)
    if factor <= 0:
        raise ValueError("Endpoint factor must be positive")
    if not 0 <= phase < factor:
        raise ValueError(
            f"Endpoint phase must satisfy 0 <= phase < factor; got {phase}, {factor}"
        )
    same_dataset_config = (
        args.teacher_dataset_config.resolve()
        == args.student_dataset_config.resolve()
    )
    same_grid_cache = same_dataset_config and factor == 1 and phase == 0

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    teacher, info = load_twin(
        args.teacher_checkpoint.resolve(),
        device=str(device),
        dataset_configs_path=args.teacher_dataset_config.resolve(),
        verbose=False,
    )
    if teacher.names[: len(teacher_names)] != teacher_names:
        raise RuntimeError("Teacher checkpoint and requested sessions differ")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_dataset_config": str(args.teacher_dataset_config.resolve()),
        "student_dataset_config": str(args.student_dataset_config.resolve()),
        "factor": factor,
        "phase": phase,
        "same_grid_cache": same_grid_cache,
        "splits": list(requested_splits),
        "sessions": {},
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text())
        contract_keys = (
            "teacher_checkpoint",
            "teacher_dataset_config",
            "student_dataset_config",
            "factor",
            "phase",
        )
        mismatched = [
            key for key in contract_keys if existing.get(key) != manifest.get(key)
        ]
        if mismatched:
            raise RuntimeError(
                "Existing teacher cache has a different contract for: "
                + ", ".join(mismatched)
            )
        manifest["splits"] = sorted(
            set(existing.get("splits") or ()) | set(requested_splits)
        )
        manifest["sessions"].update(existing.get("sessions") or {})
    for dataset_idx, (teacher_config, student_config) in enumerate(
        zip(teacher_configs, student_configs)
    ):
        name = teacher_config["session"]
        print(f"[{dataset_idx + 1}/{len(teacher_configs)}] {name}", flush=True)
        teacher_splits = prepare_uint8_session(teacher_config)
        student_splits = (
            teacher_splits
            if same_dataset_config
            else prepare_uint8_session(student_config)
        )
        session_report = {}
        for split in requested_splits:
            output = args.output_dir / f"{dataset_idx:02d}_{name}_{split}.pt"
            if output.exists() and not args.overwrite:
                artifact = torch.load(output, map_location="cpu", weights_only=False)
                session_report[split] = artifact["metadata"]
                print(f"  reusing {output}", flush=True)
                continue
            rates, metadata = cache_split(
                teacher,
                dataset_idx,
                (
                    teacher_splits[split]
                    if same_grid_cache
                    else teacher_splits["all"]
                ),
                student_splits[split],
                device,
                args.batch_size,
                args.num_workers,
                factor,
                phase,
            )
            artifact = {
                "rates": rates,
                "metadata": {
                    "session": name,
                    "dataset_idx": dataset_idx,
                    "split": split,
                    "cids": list(student_config["cids"]),
                    **metadata,
                },
            }
            torch.save(artifact, output)
            session_report[split] = artifact["metadata"]
            print(
                f"  wrote {output} ({metadata['coverage_fraction']:.3%} coverage)",
                flush=True,
            )
        manifest["sessions"][name] = session_report
        manifest_path.write_text(json.dumps(manifest, indent=2))
        del teacher_splits, student_splits

    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
