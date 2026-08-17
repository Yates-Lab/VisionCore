#!/usr/bin/env python3
"""Audit exact stimulus Jacobians for a fixed, non-cherry-picked unit panel.

The dataset and checkpoints are loaded once.  Each unit is differentiated in
the same low/median/high held-out movie contexts across every model.  The
result is a paired population comparison rather than a single attractive
example.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection.compare_exemplar_jacobians import (
    _as_one_batch,
    _choose_contexts,
    _exact_jacobian,
    _roughness,
    _to_device,
)


METRICS = (
    "temporal_second_difference_ratio",
    "spatial_laplacian_ratio",
    "temporal_power_at_or_above_60hz",
    "spatial_power_at_or_above_0p25_cycles_per_pixel",
)
METRIC_LABELS = {
    "temporal_second_difference_ratio": "temporal second-difference / energy",
    "spatial_laplacian_ratio": "spatial Laplacian / energy",
    "temporal_power_at_or_above_60hz": "temporal energy >=60 Hz",
    "spatial_power_at_or_above_0p25_cycles_per_pixel": "spatial energy >=0.25 cyc/pixel",
}


def _auto_units(valid: np.ndarray, requested: int) -> np.ndarray:
    """Evenly cover the checkpoint head after a data-only validity gate."""
    minimum_valid = 16
    eligible = np.flatnonzero(np.sum(valid, axis=0) >= minimum_valid)
    if len(eligible) < requested:
        raise RuntimeError(
            f"Only {len(eligible)} units have at least {minimum_valid} valid "
            f"held-out samples; requested {requested}"
        )
    positions = np.linspace(0, len(eligible) - 1, requested)
    chosen = eligible[np.round(positions).astype(int)]
    return np.unique(chosen)


def _bootstrap_median(values: np.ndarray, *, seed: int, n_boot: int = 2000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = np.median(values[indices], axis=1)
    return {
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
    }


def _bootstrap_paired_median_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, float | int]:
    """Bootstrap the paired unit-level median(candidate - reference)."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Paired arrays must have matching shape, got {candidate.shape} and "
            f"{reference.shape}."
        )
    valid = np.isfinite(candidate) & np.isfinite(reference)
    difference = candidate[valid] - reference[valid]
    if difference.size == 0:
        return {
            "median_difference": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_paired_units": 0,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(n_boot, difference.size))
    boot = np.median(difference[indices], axis=1)
    return {
        "median_difference": float(np.median(difference)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "n_paired_units": int(difference.size),
    }


def _plot_summary(table: pd.DataFrame, model_names: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    rng = np.random.default_rng(20260816)
    for axis, metric in zip(axes.flat, METRICS, strict=True):
        unit_values = (
            table.groupby(["model", "unit_index"], sort=False)[metric]
            .median()
            .rename("value")
            .reset_index()
        )
        groups = [
            unit_values.loc[unit_values.model.eq(name), "value"].to_numpy(dtype=float)
            for name in model_names
        ]
        axis.boxplot(groups, positions=np.arange(len(model_names)), widths=0.5, showfliers=False)
        for index, values in enumerate(groups):
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            axis.scatter(
                index + jitter,
                values,
                s=18,
                alpha=0.65,
                color=colors[index],
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
        axis.set_xticks(np.arange(len(model_names)), model_names, rotation=18, ha="right")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if metric in METRICS[2:]:
            axis.set_yscale("log")
    figure.suptitle(
        "Exact held-out Jacobian roughness across a fixed unit panel\n"
        "each point is one unit, summarized across three shared contexts",
        fontsize=13,
    )
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_gallery(
    jacobians: np.ndarray,
    unit_indices: np.ndarray,
    model_names: list[str],
    sampling_rate: float,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if len(unit_indices) > 8:
        keep = np.round(np.linspace(0, len(unit_indices) - 1, 8)).astype(int)
    else:
        keep = np.arange(len(unit_indices))
    values = jacobians[:, keep, -1]
    shown_units = unit_indices[keep]
    n_models, n_units, n_lags, _, _ = values.shape
    lag_ms = np.arange(n_lags) * 1000.0 / float(sampling_rate)
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(
        n_units,
        n_models + 1,
        figsize=(2.45 * (n_models + 1), 2.1 * n_units),
        squeeze=False,
        constrained_layout=True,
    )
    for unit_pos, unit_index in enumerate(shown_units):
        for model_pos, model_name in enumerate(model_names):
            energy = np.sqrt(np.mean(values[model_pos, unit_pos] ** 2, axis=(-2, -1)))
            axes[unit_pos, 0].plot(
                lag_ms,
                energy / max(float(energy.max()), 1e-12),
                color=colors[model_pos],
                linewidth=1.5,
                label=model_name,
            )
            peak = int(np.argmax(energy))
            spatial = values[model_pos, unit_pos, peak]
            vmax = max(float(np.max(np.abs(spatial))), 1e-12)
            axes[unit_pos, model_pos + 1].imshow(
                spatial,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
            )
            axes[unit_pos, model_pos + 1].set_xticks([])
            axes[unit_pos, model_pos + 1].set_yticks([])
            axes[unit_pos, model_pos + 1].set_title(
                f"{model_name}; peak {lag_ms[peak]:.0f} ms", fontsize=8.5
            )
        axes[unit_pos, 0].set_ylim(-0.03, 1.05)
        axes[unit_pos, 0].set_ylabel(f"unit {int(unit_index)}")
        axes[unit_pos, 0].spines[["top", "right"]].set_visible(False)
        if unit_pos == 0:
            axes[unit_pos, 0].legend(frameon=False, fontsize=7.5)
        if unit_pos == n_units - 1:
            axes[unit_pos, 0].set_xlabel("lag before response (ms)")
    figure.suptitle(
        "High-drive exact Jacobians; rows are evenly spaced valid units\n"
        "each spatial map uses its own symmetric color scale",
        fontsize=13,
    )
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--dataset-idx", type=int, default=0)
    parser.add_argument("--unit-indices", type=int, nargs="+", default=None)
    parser.add_argument("--n-auto-units", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) != len(args.model_names):
        raise ValueError("--model-names must have one entry per checkpoint")
    missing_checkpoints = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Checkpoint preflight failed before loading the dataset: "
            + ", ".join(missing_checkpoints)
        )

    from eval.load_twin import load_twin
    from training.pl_modules import MultiDatasetDM

    device = torch.device(args.device)
    meta = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    hyper = meta["hyper_parameters"]
    checkpoint_datasets = list(hyper.get("dataset_cids") or ())
    if not 0 <= args.dataset_idx < len(checkpoint_datasets):
        raise IndexError(f"dataset index {args.dataset_idx} is unavailable")
    dataset_name = checkpoint_datasets[args.dataset_idx]
    dm = MultiDatasetDM(
        cfg_dir=hyper["cfg_dir"],
        max_ds=1,
        batch=args.batch_size,
        workers=0,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
        dataset_names=[dataset_name],
    )
    dm.setup("fit")
    batch = _to_device(_as_one_batch(next(iter(dm.val_dataloader()))), device)
    batch["dataset_idx"] = torch.full_like(batch["dataset_idx"], args.dataset_idx)
    config = dm.cfgs[0]
    sampling = config.get("sampling") or {}
    supervision = config.get("supervision") or {}
    sampling_rate = float(
        supervision.get(
            "target_rate", sampling.get("target_rate", sampling.get("source_rate", 120))
        )
    )

    models: list[torch.nn.Module] = []
    predictions: list[np.ndarray] = []
    for checkpoint in args.checkpoints:
        model, _ = load_twin(checkpoint.resolve(), device=str(device), verbose=False)
        model.eval()
        if model.names[args.dataset_idx] != dataset_name:
            raise RuntimeError("dataset head order differs across checkpoints")
        with torch.no_grad():
            pred = model(
                batch["stim"].float(),
                args.dataset_idx,
                batch.get("behavior").float() if batch.get("behavior") is not None else None,
                None,
                batch.get("output_behavior").float()
                if batch.get("output_behavior") is not None
                else None,
            )
        models.append(model)
        predictions.append(pred.float().cpu().numpy())
    rates = np.stack(predictions)
    valid = batch["dfs"].detach().cpu().numpy() > 0
    unit_indices = (
        np.asarray(args.unit_indices, dtype=int)
        if args.unit_indices is not None
        else _auto_units(valid, int(args.n_auto_units))
    )
    if unit_indices.min() < 0 or unit_indices.max() >= rates.shape[-1]:
        raise IndexError("requested unit is outside the checkpoint head")

    context_rows = np.empty((len(unit_indices), 3), dtype=np.int64)
    context_drive = np.empty((len(unit_indices), 3), dtype=np.float32)
    target_quantiles = np.empty((len(unit_indices), 3), dtype=np.float32)
    jacobians: list[list[np.ndarray]] = [[] for _ in models]
    selected_rates = np.empty((len(models), len(unit_indices), 3), dtype=np.float32)
    rows_out: list[dict[str, float | int | str]] = []
    for unit_pos, unit_index in enumerate(unit_indices):
        rows, drive, targets = _choose_contexts(
            rates[:, :, unit_index], valid[:, unit_index], [0.2, 0.5, 0.8]
        )
        context_rows[unit_pos] = rows
        context_drive[unit_pos] = drive[rows]
        target_quantiles[unit_pos] = targets
        for model_pos, (model_name, model) in enumerate(zip(args.model_names, models)):
            gradient, model_rates = _exact_jacobian(model, batch, rows, int(unit_index))
            jacobians[model_pos].append(gradient.numpy())
            selected_rates[model_pos, unit_pos] = model_rates.numpy()
            report = _roughness(gradient, sampling_rate=sampling_rate)
            for context_pos in range(3):
                record: dict[str, float | int | str] = {
                    "model": model_name,
                    "unit_index": int(unit_index),
                    "context_index": context_pos,
                    "context_row": int(rows[context_pos]),
                    "predicted_count_per_bin": float(model_rates[context_pos]),
                }
                for metric, values in report.items():
                    record[metric] = float(values[context_pos])
                rows_out.append(record)
        print(f"completed unit {int(unit_index)} ({unit_pos + 1}/{len(unit_indices)})", flush=True)

    jacobian_array = np.stack([np.stack(values) for values in jacobians])
    table = pd.DataFrame(rows_out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "jacobian_unit_context_metrics.csv", index=False)
    np.savez_compressed(
        args.out_dir / "jacobian_population.npz",
        jacobians=jacobian_array,
        rates=selected_rates,
        unit_indices=unit_indices,
        context_rows=context_rows,
        context_drive=context_drive,
        target_drive_quantiles=target_quantiles,
        model_names=np.asarray(args.model_names),
    )
    _plot_summary(table, list(args.model_names), args.out_dir / "jacobian_population_summary.png")
    _plot_gallery(
        jacobian_array,
        unit_indices,
        list(args.model_names),
        sampling_rate,
        args.out_dir / "jacobian_high_drive_gallery.png",
    )

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for model_pos, model_name in enumerate(args.model_names):
        summaries[model_name] = {}
        selected = table[table.model.eq(model_name)]
        for metric_pos, metric in enumerate(METRICS):
            per_unit = selected.groupby("unit_index")[metric].median().to_numpy(dtype=float)
            summaries[model_name][metric] = _bootstrap_median(
                per_unit, seed=20260816 + 100 * model_pos + metric_pos
            )
    paired_differences: dict[str, dict[str, dict[str, float | int]]] = {}
    reference_name = args.model_names[0]
    reference_table = (
        table.loc[table.model.eq(reference_name)]
        .groupby("unit_index", sort=True)[list(METRICS)]
        .median()
    )
    for model_pos, model_name in enumerate(args.model_names[1:], start=1):
        candidate_table = (
            table.loc[table.model.eq(model_name)]
            .groupby("unit_index", sort=True)[list(METRICS)]
            .median()
        )
        common_units = reference_table.index.intersection(candidate_table.index)
        paired_differences[model_name] = {}
        for metric_pos, metric in enumerate(METRICS):
            paired_differences[model_name][metric] = (
                _bootstrap_paired_median_difference(
                    candidate_table.loc[common_units, metric].to_numpy(dtype=float),
                    reference_table.loc[common_units, metric].to_numpy(dtype=float),
                    seed=20260816 + 1000 * model_pos + metric_pos,
                )
            )
    payload = {
        "dataset": dataset_name,
        "dataset_idx": int(args.dataset_idx),
        "sampling_rate_hz": sampling_rate,
        "unit_selection": (
            "explicit" if args.unit_indices is not None else "evenly_spaced_after_data_only_validity_gate"
        ),
        "unit_indices": unit_indices.tolist(),
        "context_rule": (
            "per unit: nearest valid samples to 20th, 50th, and 80th percentiles "
            "of mean standardized log predicted rate across models"
        ),
        "summaries": summaries,
        "paired_difference_reference": reference_name,
        "paired_differences": paired_differences,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
    }
    (args.out_dir / "jacobian_population_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    del models, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
