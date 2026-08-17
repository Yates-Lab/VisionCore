#!/usr/bin/env python3
"""Deterministically score every available target in a Dekel checkpoint split.

The mixed-rate data path already aggregates spike-count supervision to 120 Hz
while retaining native 240 Hz stimulus histories. This evaluator therefore
does not re-bin predictions: it exhausts the requested validation or test split
and applies the same per-unit, clipped-per-session BPS reduction as training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def rate_contract(config):
    """Return stimulus/supervision rates for legacy and mixed-rate configs."""
    sampling = config.get("sampling") or {}
    stimulus_rate_hz = int(
        sampling.get("target_rate", sampling.get("source_rate", 120))
    )
    supervision = config.get("supervision") or {}
    supervision_rate_hz = int(
        supervision.get("target_rate", stimulus_rate_hz)
    )
    aggregation = (
        "pre-binned causal spike-count targets"
        if supervision
        else "dataset-rate spike-count targets"
    )
    return stimulus_rate_hz, supervision_rate_hz, aggregation


def iter_session_split_batches(datamodule, split, model_names=None):
    """Yield every example once, grouped by session, in deterministic order.

    The homogeneous training sampler chooses dataset batches with replacement.
    Iterating it to exhaustion therefore does *not* exhaust a validation/test
    split.  Evaluation instead walks each session dataset directly with no
    shuffle and keeps the final partial batch.
    """
    datasets = (
        datamodule.val_dsets if split == "val" else datamodule.test_dsets
    )
    workers = int(getattr(datamodule, "workers", 0))
    batch_size = int(getattr(datamodule, "batch", 256))
    if model_names is None:
        model_names = datamodule.names
    model_name_to_index = {name: idx for idx, name in enumerate(model_names)}
    for name in datamodule.names:
        dataset = datasets.get(name)
        if dataset is None:
            continue
        dataset_idx = model_name_to_index[name]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=workers > 0,
        )
        for batch in loader:
            yield dataset_idx, name, batch


def score_split(model, datamodule, device, split):
    from models.losses import PoissonBPSAggregator
    from paper.model_selection.evaluate import overall_bps

    aggregates = {name: PoissonBPSAggregator() for name in model.names}
    sample_counts = {name: 0 for name in model.names}
    identity_activation = isinstance(model.model.activation, nn.Identity)
    expected_sessions = sum(
        1 for name in model.names
        if name in (
            datamodule.val_dsets if split == "val" else datamodule.test_dsets
        )
    )
    active_name = None
    active_started = None
    completed_sessions = 0

    def report_complete(name):
        nonlocal completed_sessions
        completed_sessions += 1
        elapsed = time.perf_counter() - active_started
        print(
            f"Scored {completed_sessions}/{expected_sessions}: {name} "
            f"({sample_counts[name]:,} samples, {elapsed:.1f}s)",
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        for dataset_idx, name, raw_batch in iter_session_split_batches(
            datamodule, split, model.names
        ):
            if name != active_name:
                if active_name is not None:
                    report_complete(active_name)
                active_name = name
                active_started = time.perf_counter()
            batch = _to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    batch["stim"],
                    dataset_idx,
                    batch.get("behavior"),
                    batch.get("history"),
                    batch.get("output_behavior"),
                )
            prediction = prediction.float()
            if identity_activation:
                prediction = prediction.exp()
            aggregates[name](
                {
                    "rhat": prediction,
                    "robs": batch["robs"].float(),
                    "dfs": batch["dfs"].float(),
                }
            )
            sample_counts[name] += int(batch["stim"].shape[0])
    if active_name is not None:
        report_complete(active_name)

    per_unit = {}
    for name, aggregate in aggregates.items():
        if not aggregate.robs:
            continue
        values = aggregate.closure()
        if values is not None:
            per_unit[name] = values.detach().cpu().numpy()

    overall, per_session = overall_bps(per_unit)
    finite_unit_counts = {
        name: int(np.isfinite(values).sum()) for name, values in per_unit.items()
    }
    return {
        "bps_overall": overall,
        "bps_by_session": per_session,
        "finite_units_by_session": finite_unit_counts,
        "samples_by_session": {
            name: count for name, count in sample_counts.items() if count > 0
        },
        # Kept out of the printed JSON report below and written losslessly to a
        # compact NPZ.  This enables paired, cell-level checkpoint comparisons
        # without repeating the expensive exhaustive forward pass.
        "_bps_per_unit": per_unit,
    }


def write_per_unit_archive(path, per_unit, cids_by_session, ordered_sessions):
    """Write aligned per-cell BPS and cids without using pickle objects."""
    session_names = [name for name in ordered_sessions if name in per_unit]
    payload = {"session_names": np.asarray(session_names)}
    for session_index, name in enumerate(session_names):
        bps = np.asarray(per_unit[name])
        cids = np.asarray(cids_by_session[name], dtype=np.int64)
        if len(bps) != len(cids):
            raise RuntimeError(
                f"{name}: {len(bps)} per-unit BPS values but {len(cids)} cids"
            )
        payload[f"bps_{session_index}"] = bps
        payload[f"cids_{session_index}"] = cids
    np.savez_compressed(path, **payload)
    return session_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Load and score only this exact session name (repeatable).",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--supervision-phase",
        type=int,
        default=None,
        help=(
            "Evaluation-only native endpoint phase override. For M66, phase 1 "
            "is trained and phase 0 probes the interleaved 240-Hz endpoints."
        ),
    )
    parser.add_argument(
        "--zero-auxiliary-readouts",
        action="store_true",
        help=(
            "Diagnostic ablation: zero every additive projection from the "
            "auxiliary visual core"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from eval.load_twin import load_twin
    from training.pl_modules import MultiDatasetDM

    checkpoint_path = args.checkpoint.resolve()
    raw_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg_dir = (raw_checkpoint.get("hyper_parameters", {}) or {}).get("cfg_dir")
    if cfg_dir is None:
        raise ValueError("Checkpoint does not record cfg_dir")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    model, model_info = load_twin(
        checkpoint_path, device=str(device), verbose=False
    )
    if args.zero_auxiliary_readouts:
        auxiliary_readouts = getattr(model.model, "auxiliary_readouts", None)
        if auxiliary_readouts is None:
            raise ValueError(
                "--zero-auxiliary-readouts requires an auxiliary visual branch"
            )
        auxiliary_collections = [auxiliary_readouts]
        auxiliary_residuals = getattr(
            model.model, "auxiliary_residual_readouts", None
        )
        if auxiliary_residuals is not None:
            auxiliary_collections.append(auxiliary_residuals)
        with torch.no_grad():
            for readout in (
                item
                for collection in auxiliary_collections
                for item in collection
            ):
                if readout.features is not None:
                    readout.features.weight.zero_()
                if readout.base_scale is not None:
                    readout.base_scale.zero_()
                if readout.population_left is not None:
                    readout.population_left.zero_()
    datamodule = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=args.max_datasets,
        batch=args.batch_size,
        workers=args.num_workers,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
        supervision_phase_override=args.supervision_phase,
        dataset_names=args.session,
    )
    datamodule.setup("fit")
    if any(name not in model.names for name in datamodule.names):
        raise RuntimeError("A data-config session is absent from the checkpoint")

    result = score_split(model, datamodule, device, args.split)
    per_unit = result.pop("_bps_per_unit")
    first_config = datamodule.cfgs[0]
    stimulus_rate_hz, supervision_rate_hz, aggregation = rate_contract(
        first_config
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(raw_checkpoint.get("epoch", -1)),
        "split": args.split,
        "stimulus_rate_hz": stimulus_rate_hz,
        "supervision_rate_hz": supervision_rate_hz,
        "aggregation": aggregation,
        "zero_auxiliary_readouts": bool(args.zero_auxiliary_readouts),
        "supervision_phase": (
            int(args.supervision_phase)
            if args.supervision_phase is not None
            else int((first_config.get("supervision") or {}).get("phase", 0))
        ),
        **result,
    }

    output = args.out or (
        ROOT
        / "outputs"
        / "dekel240_evaluation"
        / checkpoint_path.parent.name
        / f"epoch_{report['checkpoint_epoch']:03d}_{args.split}_full.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    per_unit_path = output.with_name(f"{output.stem}_per_unit.npz")
    write_per_unit_archive(
        per_unit_path,
        per_unit,
        model_info["cids_by_session"],
        datamodule.names,
    )
    report["per_unit_bps_npz"] = str(per_unit_path.resolve())
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
