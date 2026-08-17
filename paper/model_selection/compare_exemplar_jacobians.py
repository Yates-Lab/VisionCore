#!/usr/bin/env python3
"""Compare one unit's exact Jacobian across checkpoints and shared contexts.

The context examples are chosen once from a deterministic validation batch by
the 20th, 50th, and 80th percentiles of the models' mean standardized log rate.
Every model is then differentiated at exactly those stimulus and behavior
examples.  The gradient target is log predicted rate.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_one_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) != 1:
        raise ValueError("Expected one homogeneous-session validation batch")
    return batch[0]


def _to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _choose_contexts(rates, valid, quantiles):
    """Choose shared low/median/high ensemble-drive examples."""
    log_rate = np.log(np.maximum(rates, 1e-8))
    standardized = np.full_like(log_rate, np.nan)
    for model_idx in range(log_rate.shape[0]):
        values = log_rate[model_idx, valid]
        scale = values.std()
        standardized[model_idx] = (
            (log_rate[model_idx] - values.mean()) / max(scale, 1e-8)
        )
    drive = np.nanmean(standardized, axis=0)
    candidates = np.flatnonzero(valid & np.isfinite(drive))
    targets = np.quantile(drive[candidates], quantiles)
    chosen = []
    for target in targets:
        available = np.asarray([idx for idx in candidates if idx not in chosen])
        chosen.append(int(available[np.argmin(np.abs(drive[available] - target))]))
    return np.asarray(chosen), drive, targets


def _exact_jacobian(model, batch, rows, unit_index):
    stimulus = batch["stim"][rows].float().detach().requires_grad_(True)
    behavior = batch.get("behavior")
    if behavior is not None:
        behavior = behavior[rows].float()
    output_behavior = batch.get("output_behavior")
    if output_behavior is not None:
        output_behavior = output_behavior[rows].float()
    dataset_idx = int(batch["dataset_idx"][0])
    prediction = model(
        stimulus,
        dataset_idx,
        behavior,
        None,
        output_behavior,
    )
    target = prediction[:, unit_index].clamp_min(1e-8).log().sum()
    gradient = torch.autograd.grad(target, stimulus, create_graph=False)[0][:, 0]
    return gradient.detach().float().cpu(), prediction[:, unit_index].detach().cpu()


def _roughness(jacobian, sampling_rate=240.0):
    value = jacobian.float()
    denom = value.square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    temporal = value.diff(n=2, dim=1).square().mean(dim=(1, 2, 3)) / denom

    flat = value.flatten(0, 1).unsqueeze(1)
    padded = torch.nn.functional.pad(flat, (1, 1, 1, 1))
    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    laplacian = torch.nn.functional.conv2d(padded, kernel).reshape_as(value)
    spatial = laplacian.square().mean(dim=(1, 2, 3)) / denom

    frequency = torch.fft.rfftfreq(value.shape[1], d=1.0 / sampling_rate)
    temporal_power = torch.fft.rfft(value, dim=1).abs().square().sum(dim=(2, 3))
    temporal_high = temporal_power[:, frequency >= sampling_rate / 4].sum(dim=1)
    temporal_high /= temporal_power.sum(dim=1).clamp_min(1e-12)

    fy = torch.fft.fftfreq(value.shape[2])[:, None]
    fx = torch.fft.fftfreq(value.shape[3])[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    spatial_power = torch.fft.fft2(value, dim=(-2, -1)).abs().square().sum(dim=1)
    spatial_high = spatial_power[:, radius >= 0.25].sum(dim=1)
    spatial_high /= spatial_power.sum(dim=(1, 2)).clamp_min(1e-12)
    return {
        "rms": value.square().mean(dim=(1, 2, 3)).sqrt().tolist(),
        "temporal_second_difference_ratio": temporal.tolist(),
        "spatial_laplacian_ratio": spatial.tolist(),
        "temporal_power_at_or_above_60hz": temporal_high.tolist(),
        "spatial_power_at_or_above_0p25_cycles_per_pixel": spatial_high.tolist(),
    }


def _plot(
    jacobians,
    model_names,
    context_names,
    predictions,
    out_path,
    sampling_rate,
):
    import matplotlib.pyplot as plt

    n_models, n_contexts, n_lags, _, _ = jacobians.shape
    lag_ms = np.arange(n_lags) * 1000.0 / float(sampling_rate)
    colors = plt.get_cmap("tab10").colors[:n_contexts]
    figure, axes = plt.subplots(
        n_models,
        n_contexts + 1,
        figsize=(3.15 * (n_contexts + 1), 2.7 * n_models),
        squeeze=False,
    )

    for model_idx, model_name in enumerate(model_names):
        temporal_energy = np.sqrt(np.mean(jacobians[model_idx] ** 2, axis=(-2, -1)))
        for context_idx, context_name in enumerate(context_names):
            energy = temporal_energy[context_idx]
            axes[model_idx, 0].plot(
                lag_ms,
                energy / max(float(energy.max()), 1e-12),
                color=colors[context_idx],
                label=context_name,
                linewidth=1.8,
            )
        axes[model_idx, 0].set_ylim(-0.04, 1.08)
        axes[model_idx, 0].set_ylabel(f"{model_name}\nnormalized RMS")
        axes[model_idx, 0].spines[["top", "right"]].set_visible(False)
        if model_idx == 0:
            axes[model_idx, 0].set_title("Temporal Jacobian energy")
            axes[model_idx, 0].legend(frameon=False, fontsize=8)
        if model_idx == n_models - 1:
            axes[model_idx, 0].set_xlabel("lag before response (ms)")

        for context_idx, context_name in enumerate(context_names):
            peak = int(np.argmax(temporal_energy[context_idx]))
            spatial = jacobians[model_idx, context_idx, peak]
            vmax = max(float(np.max(np.abs(spatial))), 1e-12)
            axes[model_idx, context_idx + 1].imshow(
                spatial,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
            )
            axes[model_idx, context_idx + 1].set_xticks([])
            axes[model_idx, context_idx + 1].set_yticks([])
            axes[model_idx, context_idx + 1].set_title(
                f"{context_name}: {predictions[model_idx, context_idx] * sampling_rate:.1f} sp/s\n"
                f"peak lag {lag_ms[peak]:.0f} ms",
                fontsize=9,
            )

    figure.suptitle(
        "Exact stimulus Jacobian of log predicted rate\n"
        "shared held-out contexts; each spatial map uses its own symmetric scale",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--unit-index", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--context-rows",
        type=int,
        nargs="+",
        default=None,
        help="Use these deterministic validation-batch rows instead of rate quantiles",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.model_names):
        parser.error("--model-names must have one entry per checkpoint")

    from eval.load_twin import load_twin
    from training.pl_modules import MultiDatasetDM

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint_meta = torch.load(
        args.checkpoints[0], map_location="cpu", weights_only=False
    )
    hyper_parameters = checkpoint_meta["hyper_parameters"]
    cfg_dir = hyper_parameters["cfg_dir"]
    checkpoint_datasets = list(hyper_parameters.get("dataset_cids") or ())
    if not checkpoint_datasets:
        raise RuntimeError("Checkpoint does not record its ordered dataset heads")
    if not 0 <= args.dataset_idx < len(checkpoint_datasets):
        raise IndexError(
            f"Dataset index {args.dataset_idx} is outside "
            f"{len(checkpoint_datasets)} checkpoint heads"
        )
    dataset_name = checkpoint_datasets[args.dataset_idx]
    dm = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=1,
        batch=args.batch_size,
        workers=0,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
        dataset_names=[dataset_name],
    )
    dm.setup("fit")
    if dm.names != [dataset_name]:
        raise RuntimeError(
            f"Targeted data load returned {dm.names}, expected {[dataset_name]}"
        )
    first_config = dm.cfgs[0]
    sampling = first_config.get("sampling") or {}
    supervision = first_config.get("supervision") or {}
    sampling_rate = float(
        supervision.get(
            "target_rate",
            sampling.get("target_rate", sampling.get("source_rate", 120)),
        )
    )
    batch = _to_device(_as_one_batch(next(iter(dm.val_dataloader()))), device)
    # The targeted DataModule labels this sole session as local dataset 0.
    # Restore the checkpoint's original head index before every model call.
    batch["dataset_idx"] = torch.full_like(
        batch["dataset_idx"], args.dataset_idx
    )

    models = []
    all_rates = []
    for checkpoint in args.checkpoints:
        model, _ = load_twin(checkpoint.resolve(), device=str(device), verbose=False)
        model.eval()
        if model.names[args.dataset_idx] != dataset_name:
            raise RuntimeError(
                "Dataset order differs across checkpoint and requested data: "
                f"{model.names[args.dataset_idx]} != {dataset_name}"
            )
        with torch.no_grad():
            prediction = model(
                batch["stim"].float(),
                args.dataset_idx,
                batch.get("behavior").float() if batch.get("behavior") is not None else None,
                None,
                batch.get("output_behavior").float()
                if batch.get("output_behavior") is not None
                else None,
            )
        if args.unit_index >= prediction.shape[1]:
            raise IndexError(
                f"Unit {args.unit_index} is outside {prediction.shape[1]} outputs"
            )
        models.append(model)
        all_rates.append(prediction[:, args.unit_index].float().cpu().numpy())

    rates = np.stack(all_rates)
    valid = batch["dfs"][:, args.unit_index].detach().cpu().numpy() > 0
    if args.context_rows is None:
        rows, drive, targets = _choose_contexts(rates, valid, [0.2, 0.5, 0.8])
        context_names = ["low drive", "median drive", "high drive"]
        selection_rule = (
            "Nearest valid examples to the 20th, 50th, and 80th percentiles "
            "of mean standardized log predicted rate across models"
        )
    else:
        rows = np.asarray(args.context_rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) != 3:
            parser.error("--context-rows currently requires exactly three rows")
        if rows.min() < 0 or rows.max() >= rates.shape[1]:
            parser.error("--context-rows contains a row outside the validation batch")
        if not valid[rows].all():
            parser.error("--context-rows contains an invalid example for this unit")
        drive = np.full(rates.shape[1], np.nan)
        targets = np.full(3, np.nan)
        context_names = ["fixed context 1", "fixed context 2", "fixed context 3"]
        selection_rule = "Explicit deterministic validation-batch rows"

    all_jacobians = []
    selected_rates = []
    reports = {}
    for model_name, model in zip(args.model_names, models):
        jacobian, model_rates = _exact_jacobian(
            model, batch, rows, args.unit_index
        )
        all_jacobians.append(jacobian.numpy())
        selected_rates.append(model_rates.numpy())
        reports[model_name] = _roughness(jacobian, sampling_rate=sampling_rate)

    jacobians = np.stack(all_jacobians)
    selected_rates = np.stack(selected_rates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "shared_context_jacobians.npz",
        jacobians=jacobians,
        rates=selected_rates,
        context_rows=rows,
        context_drive=drive[rows],
        model_names=np.asarray(args.model_names),
        unit_index=args.unit_index,
    )
    _plot(
        jacobians,
        args.model_names,
        context_names,
        selected_rates,
        args.out_dir / "shared_context_jacobians.png",
        sampling_rate,
    )
    report = {
        "dataset": dataset_name,
        "dataset_idx": args.dataset_idx,
        "sampling_rate_hz": sampling_rate,
        "unit_index": args.unit_index,
        "selection_rule": selection_rule,
        "context_names": context_names,
        "context_rows": rows.tolist(),
        "context_drive": (
            drive[rows].tolist() if args.context_rows is None else None
        ),
        "target_drive_quantiles": (
            targets.tolist() if args.context_rows is None else None
        ),
        "predicted_rate_counts_per_bin": {
            name: values.tolist()
            for name, values in zip(args.model_names, selected_rates)
        },
        "jacobian_metrics": reports,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
    }
    (args.out_dir / "shared_context_jacobians.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))
    del models, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
