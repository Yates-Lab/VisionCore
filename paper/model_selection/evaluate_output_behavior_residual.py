#!/usr/bin/env python3
"""Paired evaluation of an output behavior residual and its inherited twin.

The visual core and legacy M16 behavior path are evaluated exactly once per
batch.  Several cheap output-head conditions are then scored on identical
examples, eliminating validation-subsample noise from the comparison:

``inherited``
    Bypass the new output residual (the exact M16 prediction for an M20 warm
    start).
``intact``
    Gain plus additive output residual.
``additive_only`` / ``gain_only``
    Component ablations of the learned residual.
``residual_behavior_zero``
    Feed zeros only to the new residual; the inherited M16 behavior path stays
    intact.
``residual_behavior_shifted``
    Deterministically roll behavior within each batch to break its alignment
    with the neural target while preserving its marginal distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection.evaluate_dekel_split import (  # noqa: E402
    _to_device,
    iter_session_split_batches,
    write_per_unit_archive,
)


CONDITIONS = (
    "inherited",
    "intact",
    "additive_only",
    "gain_only",
    "residual_behavior_zero",
    "residual_behavior_shifted",
)


def _rates_from_logits(base_model, logits, dataset_idx):
    output = base_model.activation(logits)
    if base_model.baseline_enabled:
        output = output + base_model.baseline_activation(
            base_model.baselines[dataset_idx]
        )
    if isinstance(base_model.activation, nn.Identity):
        output = output.clamp_min(-20).exp()
    else:
        output = output.clamp_min(1e-8)
    return output.float()


def output_residual_conditions(
    base_model,
    stimulus,
    behavior,
    dataset_idx,
    output_behavior=None,
):
    """Return all paired rate conditions after one inherited-core forward."""
    residual = base_model.output_modulator
    if residual is None:
        raise ValueError("Checkpoint has no output_modulator")

    x = base_model.adapters[dataset_idx](stimulus)
    feature_behavior = base_model.resolve_feature_behavior(
        behavior, output_behavior
    )
    features = base_model.core_forward(x, feature_behavior)
    logits = base_model.readouts[dataset_idx](features)

    residual_behavior = base_model.resolve_output_behavior(
        behavior, output_behavior
    )
    gain, offset = residual.gain_offset(residual_behavior, dataset_idx)
    zero_gain, zero_offset = residual.gain_offset(
        torch.zeros_like(residual_behavior), dataset_idx
    )

    # A roll is deterministic, avoids a device-specific RNG, and has no fixed
    # points for ordinary batches.  The final singleton batch is necessarily
    # unshuffled and is negligible in an exhaustive split.
    shifted_behavior = residual_behavior.roll(
        shifts=max(1, residual_behavior.shape[0] // 2), dims=0
    )
    shifted_gain, shifted_offset = residual.gain_offset(
        shifted_behavior, dataset_idx
    )

    condition_logits = {
        "inherited": logits,
        "intact": logits * (1.0 + gain) + offset,
        "additive_only": logits + offset,
        "gain_only": logits * (1.0 + gain),
        "residual_behavior_zero": logits * (1.0 + zero_gain) + zero_offset,
        "residual_behavior_shifted": (
            logits * (1.0 + shifted_gain) + shifted_offset
        ),
    }
    rates = {
        name: _rates_from_logits(base_model, values, dataset_idx)
        for name, values in condition_logits.items()
    }
    return rates, gain.float(), offset.float()


def score_paired(model, datamodule, device, split):
    from models.losses import PoissonBPSAggregator
    from paper.model_selection.evaluate import overall_bps

    aggregates = {
        condition: {
            name: PoissonBPSAggregator() for name in model.names
        }
        for condition in CONDITIONS
    }
    sample_counts = {name: 0 for name in model.names}
    residual_stats = {
        "count": 0,
        "gain_sum": 0.0,
        "gain_sq_sum": 0.0,
        "gain_abs_max": 0.0,
        "gain_near_bound_count": 0,
        "offset_sum": 0.0,
        "offset_sq_sum": 0.0,
        "offset_abs_max": 0.0,
    }

    model.eval()
    with torch.no_grad():
        for dataset_idx, name, raw_batch in iter_session_split_batches(
            datamodule, split
        ):
            batch = _to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                predictions, gain, offset = output_residual_conditions(
                    model.model,
                    batch["stim"],
                    batch["behavior"],
                    dataset_idx,
                    batch.get("output_behavior"),
                )
            observations = batch["robs"].float()
            data_filters = batch["dfs"].float()
            for condition, prediction in predictions.items():
                aggregates[condition][name](
                    {
                        "rhat": prediction,
                        "robs": observations,
                        "dfs": data_filters,
                    }
                )
            sample_counts[name] += int(batch["stim"].shape[0])

            residual_stats["count"] += gain.numel()
            residual_stats["gain_sum"] += gain.sum().item()
            residual_stats["gain_sq_sum"] += gain.square().sum().item()
            residual_stats["gain_abs_max"] = max(
                residual_stats["gain_abs_max"], gain.abs().max().item()
            )
            residual_stats["gain_near_bound_count"] += int(
                (gain.abs() >= 0.9 * model.model.output_modulator.max_gain)
                .sum()
                .item()
            )
            residual_stats["offset_sum"] += offset.sum().item()
            residual_stats["offset_sq_sum"] += offset.square().sum().item()
            residual_stats["offset_abs_max"] = max(
                residual_stats["offset_abs_max"], offset.abs().max().item()
            )

    per_unit_by_condition = {}
    condition_reports = {}
    for condition in CONDITIONS:
        per_unit = {}
        for name, aggregate in aggregates[condition].items():
            if aggregate.robs:
                values = aggregate.closure()
                if values is not None:
                    per_unit[name] = values.detach().cpu().numpy()
        overall, per_session = overall_bps(per_unit)
        per_unit_by_condition[condition] = per_unit
        condition_reports[condition] = {
            "bps_overall": overall,
            "bps_by_session": per_session,
        }

    paired_deltas = {}
    inherited = per_unit_by_condition["inherited"]
    for condition in CONDITIONS[1:]:
        differences = []
        for name in inherited:
            if name not in per_unit_by_condition[condition]:
                continue
            difference = (
                per_unit_by_condition[condition][name] - inherited[name]
            )
            differences.append(difference[np.isfinite(difference)])
        finite = np.concatenate(differences) if differences else np.empty(0)
        paired_deltas[condition] = {
            "mean_per_unit_bps_delta": (
                float(np.mean(finite)) if finite.size else float("nan")
            ),
            "median_per_unit_bps_delta": (
                float(np.median(finite)) if finite.size else float("nan")
            ),
            "fraction_units_improved": (
                float(np.mean(finite > 0)) if finite.size else float("nan")
            ),
            "finite_units": int(finite.size),
        }

    count = max(1, residual_stats.pop("count"))
    gain_mean = residual_stats.pop("gain_sum") / count
    offset_mean = residual_stats.pop("offset_sum") / count
    gain_sq_mean = residual_stats.pop("gain_sq_sum") / count
    offset_sq_mean = residual_stats.pop("offset_sq_sum") / count
    residual_stats.update(
        {
            "gain_mean": gain_mean,
            "gain_std": max(0.0, gain_sq_mean - gain_mean**2) ** 0.5,
            "gain_near_bound_fraction": (
                residual_stats.pop("gain_near_bound_count") / count
            ),
            "offset_mean": offset_mean,
            "offset_std": max(0.0, offset_sq_mean - offset_mean**2) ** 0.5,
        }
    )
    return {
        "conditions": condition_reports,
        "paired_vs_inherited": paired_deltas,
        "residual_stats": residual_stats,
        "samples_by_session": {
            name: count for name, count in sample_counts.items() if count > 0
        },
    }, per_unit_by_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
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
    if model.model.output_modulator is None:
        raise ValueError("Checkpoint has no output behavior residual")

    datamodule = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=args.max_datasets,
        batch=args.batch_size,
        workers=args.num_workers,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
    )
    datamodule.setup("fit")
    if datamodule.names != model.names[: len(datamodule.names)]:
        raise RuntimeError("Dataset order differs between checkpoint and data config")

    result, per_unit_by_condition = score_paired(
        model, datamodule, device, args.split
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(raw_checkpoint.get("epoch", -1)),
        "split": args.split,
        **result,
    }

    output = args.out or (
        ROOT
        / "outputs"
        / "dekel240_evaluation"
        / checkpoint_path.parent.name
        / f"epoch_{report['checkpoint_epoch']:03d}_{args.split}_output_residual.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_paths = {}
    for condition, per_unit in per_unit_by_condition.items():
        archive = output.with_name(f"{output.stem}_{condition}_per_unit.npz")
        write_per_unit_archive(
            archive,
            per_unit,
            model_info["cids_by_session"],
            datamodule.names,
        )
        archive_paths[condition] = str(archive.resolve())
    report["per_unit_bps_npz"] = archive_paths
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
