#!/usr/bin/env python3
"""Distill only Ryan's behavior-dependent residual into a feed-forward head.

The teacher is evaluated twice on its native 120-Hz training batches: once
with the recorded behavior and once with the behavior tensor set to zero.  A
standalone neuron-specific gain/offset head learns to map the zero-behavior
teacher logits to the intact teacher rates.  No teacher visual parameter or
activation is copied.  The resulting module can therefore be attached after
an independently trained smooth visual twin without transferring the
teacher's stimulus Jacobian geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def inverse_softplus(rate: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive predicted rates."""
    rate = rate.clamp_min(eps)
    return rate + torch.log(-torch.expm1(-rate))


def poisson_cross_entropy(pred_rate: torch.Tensor, target_rate: torch.Tensor) -> torch.Tensor:
    """Poisson cross-entropy up to the target-only constant."""
    return (pred_rate - target_rate * pred_rate.clamp_min(1.0e-8).log()).mean()


def residual_metrics(
    zero_rate: torch.Tensor,
    intact_rate: torch.Tensor,
    adapted_rate: torch.Tensor,
) -> dict[str, float]:
    """Return interpretable batch metrics for behavior-residual fidelity."""
    zero_logit = inverse_softplus(zero_rate.float())
    intact_logit = inverse_softplus(intact_rate.float())
    adapted_logit = inverse_softplus(adapted_rate.float())
    target = (intact_logit - zero_logit).flatten()
    prediction = (adapted_logit - zero_logit).flatten()
    centered = target - target.mean()
    residual = target - prediction
    variance = centered.square().mean()
    r2 = 1.0 - residual.square().mean() / variance.clamp_min(1.0e-12)
    if target.numel() > 1 and target.std() > 0 and prediction.std() > 0:
        corr = torch.corrcoef(torch.stack((target, prediction)))[0, 1]
    else:
        corr = target.new_tensor(float("nan"))
    return {
        "identity_nll": float(poisson_cross_entropy(zero_rate, intact_rate)),
        "adapted_nll": float(poisson_cross_entropy(adapted_rate, intact_rate)),
        "residual_logit_r2": float(r2),
        "residual_logit_correlation": float(corr),
    }


def improves_behavior_residual(metrics: dict, best: dict | None) -> bool:
    """Rank checkpoints by gain over their matched identity prediction.

    Absolute teacher NLL varies with session mix and firing rate, so it is not
    a valid cross-checkpoint criterion unless the validation batches are
    identical.  The matched improvement is the quantity this adapter is meant
    to maximize and remains interpretable even in saved reports.
    """
    return best is None or metrics["nll_improvement"] > best["nll_improvement"]


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _set_loader_epoch(loader, epoch: int):
    sampler = getattr(loader, "batch_sampler", None)
    setter = getattr(sampler, "set_epoch", None)
    if setter is not None:
        setter(epoch)


def teacher_pair(teacher, batch, device):
    """Return zero-behavior and intact teacher rates on one homogeneous batch."""
    batch = _to_device(batch, device)
    dataset_idx = int(batch["dataset_idx"][0])
    behavior = batch["behavior"].float()
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )
    with torch.no_grad(), autocast:
        intact = teacher(batch["stim"], dataset_idx, behavior)
        zero = teacher(batch["stim"], dataset_idx, torch.zeros_like(behavior))
    return dataset_idx, behavior, zero.float(), intact.float()


def evaluate(adapter, teacher, loader, device, max_batches: int):
    adapter.eval()
    sums = {
        "identity_nll": 0.0,
        "adapted_nll": 0.0,
        "residual_logit_r2": 0.0,
        "residual_logit_correlation": 0.0,
    }
    counts = {key: 0 for key in sums}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            dataset_idx, behavior, zero_rate, intact_rate = teacher_pair(
                teacher, batch, device
            )
            zero_logit = inverse_softplus(zero_rate)
            adapted_rate = F.softplus(adapter(zero_logit, behavior, dataset_idx))
            metrics = residual_metrics(zero_rate, intact_rate, adapted_rate)
            for key, value in metrics.items():
                if math.isfinite(value):
                    sums[key] += value
                    counts[key] += 1
    result = {
        key: sums[key] / counts[key] if counts[key] else float("nan")
        for key in sums
    }
    result["n_batches"] = min(batch_index + 1, max_batches) if 'batch_index' in locals() else 0
    result["nll_improvement"] = result["identity_nll"] - result["adapted_nll"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_checkpoint", type=Path)
    parser.add_argument("--dataset-config", type=Path,
                        default=ROOT / "paper/model_selection/configs/multi_120_long_split3.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--logit-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=201)
    args = parser.parse_args()

    from eval.load_twin import load_twin
    from models.modules.modulator import MultiDatasetBehaviorOutputModulator
    from training.pl_modules import MultiDatasetDM

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    teacher, info = load_twin(
        args.teacher_checkpoint.resolve(),
        device=str(device),
        dataset_configs_path=args.dataset_config.resolve(),
        verbose=False,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    config = {
        "behavior_dim": int(info.get("behavior_dim") or 42),
        "hidden_dims": [128, 64],
        "bottleneck_dim": 64,
        "activation": "gelu",
        "dropout": 0.0,
        "input_norm": False,
        "use_gain": True,
        "max_gain": 0.75,
    }
    unit_counts = info["readout_sizes"][: args.max_datasets]
    adapter = MultiDatasetBehaviorOutputModulator(config, unit_counts).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )

    dm = MultiDatasetDM(
        cfg_dir=str(args.dataset_config.resolve()),
        max_ds=args.max_datasets,
        batch=args.batch_size,
        workers=args.num_workers,
        steps_per_epoch=args.steps_per_epoch,
        dset_dtype="uint8",
        homogeneous_batches=True,
    )
    dm.setup("fit")
    if dm.names != teacher.names[: len(dm.names)]:
        raise RuntimeError("Teacher and distillation dataset order differ")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    for epoch in range(args.epochs):
        _set_loader_epoch(train_loader, epoch)
        adapter.train()
        train_sum = 0.0
        iterator = iter(train_loader)
        for step in range(args.steps_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            dataset_idx, behavior, zero_rate, intact_rate = teacher_pair(
                teacher, batch, device
            )
            zero_logit = inverse_softplus(zero_rate)
            intact_logit = inverse_softplus(intact_rate)
            predicted_logit = adapter(zero_logit, behavior, dataset_idx)
            predicted_rate = F.softplus(predicted_logit)
            data_loss = poisson_cross_entropy(predicted_rate, intact_rate)
            logit_loss = F.smooth_l1_loss(
                predicted_logit - zero_logit,
                intact_logit - zero_logit,
            )
            zero_behavior = torch.zeros_like(behavior)
            zero_gain, zero_offset = adapter.gain_offset(zero_behavior, dataset_idx)
            anchor_loss = zero_gain.square().mean() + zero_offset.square().mean()
            loss = data_loss + args.logit_weight * logit_loss + args.anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 10.0)
            optimizer.step()
            train_sum += float(loss.detach())
        scheduler.step()

        # Keep the held-out panel identical across epochs.  Rotating the
        # validation sampler changes the session/firing-rate mixture and makes
        # absolute NLL values incomparable for checkpoint selection.
        _set_loader_epoch(val_loader, 0)
        metrics = evaluate(
            adapter, teacher, val_loader, device, args.validation_batches
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_sum / args.steps_per_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if improves_behavior_residual(metrics, best):
            best = dict(record)
            torch.save(
                {
                    "state_dict": adapter.state_dict(),
                    "config": config,
                    "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                    "dataset_config": str(args.dataset_config.resolve()),
                    "dataset_names": dm.names,
                    "cids_by_session": {
                        name: info["cids_by_session"][name] for name in dm.names
                    },
                    "best": best,
                    "history": history,
                },
                args.output,
            )

    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({"best": best, "history": history}, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
