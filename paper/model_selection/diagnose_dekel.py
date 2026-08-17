#!/usr/bin/env python3
"""Checkpoint diagnostics for the 240 Hz feed-forward Dekel twins.

Produces three compact artifacts from a pinned checkpoint:

* held-out bits/spike on a deterministic subset of one session;
* learned first-layer temporal and spatial components; and
* exact input Jacobians for several units at multiple real validation contexts.

The Jacobian target is log predicted rate, which removes arbitrary firing-rate
scale while preserving the local stimulus direction.  No finite differences
or surrogate model are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _as_one_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) != 1:
        raise ValueError("Diagnostic loader must return one homogeneous session")
    return batch[0]


def _checkpoint_score(checkpoint):
    scores = []
    for state in (checkpoint.get("callbacks", {}) or {}).values():
        score = state.get("best_model_score") if isinstance(state, dict) else None
        if torch.is_tensor(score) and score.numel() == 1:
            scores.append(float(score))
        elif isinstance(score, (float, int)):
            scores.append(float(score))
    return max(scores) if scores else None


def summarize_readout_widths(model, model_config):
    """Summarize Gaussian standard deviations in scaffold-pixel units."""
    readouts = getattr(model.model, "readouts", ())
    widths = [
        readout.std.detach().float().cpu().flatten()
        for readout in readouts
        if hasattr(readout, "std")
    ]
    if not widths:
        return {"available": False}

    values = torch.cat(widths)
    floor = None
    for spec in model_config.get("regularization", []):
        if spec.get("type") != "proximal_clamp_min":
            continue
        if any("readouts/std" in target for target in spec.get("apply_to", [])):
            candidate = float(spec["lambda"])
            floor = candidate if floor is None else max(floor, candidate)

    quantiles = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
    summary = {
        "available": True,
        "units": "scaffold_pixels",
        "n_axis_widths": int(values.numel()),
        "minimum": float(values.min()),
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(values.mean()),
        "p90": float(quantiles[2]),
        "maximum": float(values.max()),
        "configured_floor": floor,
    }
    if floor is not None:
        summary["fraction_at_floor"] = float(
            (values <= floor + 1.0e-4).float().mean()
        )
    return summary


def summarize_residual_readouts(model, attribute="residual_readouts"):
    """Describe the learned strength and bounded geometry of second RFs."""
    residuals = getattr(model.model, attribute, None)
    bases = getattr(model.model, "readouts", ())
    if residuals is None:
        return {"available": False}
    if len(residuals) != len(bases):
        raise RuntimeError("Residual/base readout session count differs")

    feature_norms = []
    displacements = []
    width_ratios = []
    orientation_deltas = []
    modes = set()
    for residual, base in zip(residuals, bases):
        modes.add(residual.feature_mode)
        mean, std, theta = residual.effective_geometry(base)
        displacements.append(
            (mean.detach() - base.mean.detach()).float().norm(dim=1).cpu()
        )
        width_ratios.append(
            (
                std.detach().float()
                / base.std.detach().float().clamp_min(1.0e-6)
            ).cpu()
        )
        orientation_deltas.append(
            (theta.detach() - base.theta.detach()).float().abs().cpu()
        )
        if residual.feature_mode == "independent":
            strength = residual.features.weight.detach().float().flatten(1).norm(dim=1)
        elif residual.feature_mode == "base_scaled":
            strength = residual.max_base_scale * torch.tanh(
                residual.base_scale.detach().float()
            ).abs()
        else:
            mixing = (
                residual.population_left.detach().float()
                @ residual.population_right.detach().float().T
            )
            mixing = residual.max_population_mix * torch.tanh(
                mixing / residual.population_rank ** 0.5
            )
            strength = mixing.norm(dim=1)
        feature_norms.append(strength.cpu())

    feature_norm = torch.cat(feature_norms)
    displacement = torch.cat(displacements)
    width_ratio = torch.cat(width_ratios).flatten()
    orientation_delta = torch.cat(orientation_deltas)
    active = feature_norm > 1.0e-8

    def quantiles(value):
        if not value.numel():
            return None
        q = torch.quantile(value, value.new_tensor([0.1, 0.5, 0.9]))
        return {
            "minimum": float(value.min()),
            "p10": float(q[0]),
            "median": float(q[1]),
            "p90": float(q[2]),
            "maximum": float(value.max()),
        }

    active_axes = active[:, None].expand(-1, 2).reshape(-1)
    return {
        "available": True,
        "feature_modes": sorted(modes),
        "n_units": int(feature_norm.numel()),
        "n_active_units": int(active.sum()),
        "fraction_active": float(active.float().mean()),
        "feature_strength": quantiles(feature_norm),
        "active_center_displacement_scaffold_pixels": quantiles(
            displacement[active]
        ),
        "active_width_ratio": quantiles(width_ratio[active_axes]),
        "active_absolute_orientation_delta_radians": quantiles(
            orientation_delta[active]
        ),
    }


def summarize_exact_zeros(model):
    """Report exact sparsity induced by proximal competitive penalties."""

    def summarize(tensors):
        tensors = [value.detach() for value in tensors]
        if not tensors:
            return {"available": False}
        fractions = [float((value == 0).float().mean()) for value in tensors]
        total = sum(value.numel() for value in tensors)
        zeros = sum(int((value == 0).sum()) for value in tensors)
        return {
            "available": True,
            "fraction": zeros / total,
            "minimum_per_tensor": min(fractions),
            "median_per_tensor": float(np.median(fractions)),
            "maximum_per_tensor": max(fractions),
            "n_tensors": len(tensors),
        }

    readout_weights = [
        readout.features.weight
        for readout in getattr(model.model, "readouts", ())
        if hasattr(readout, "features") and hasattr(readout.features, "weight")
    ]
    hidden_weights = [
        module.conv.weight
        for name in ("stage1_conv", "stage2_conv", "stage3_conv")
        if (module := getattr(model.model.convnet, name, None)) is not None
    ]
    return {
        "readout_feature_weights": summarize(readout_weights),
        "hidden_spatial_weights": summarize(hidden_weights),
    }


def score_validation_subset(
    model, loader, device, max_batches, dataset_idx_override=None
):
    from models.losses import PoissonBPSAggregator

    aggregate = PoissonBPSAggregator()
    batches_scored = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = _to_device(_as_one_batch(raw_batch), device)
            dataset_idx = (
                int(dataset_idx_override)
                if dataset_idx_override is not None
                else int(batch["dataset_idx"][0])
            )
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
            aggregate(
                {
                    "rhat": prediction.float(),
                    "robs": batch["robs"].float(),
                    "dfs": batch["dfs"].float(),
                }
            )
            batches_scored += 1

    per_unit = aggregate.closure()
    if per_unit is None:
        return {"bps_mean_clipped": None, "bps_median": None, "n_units": 0}
    values = per_unit.detach().cpu().numpy()
    finite = values[np.isfinite(values)]
    return {
        "bps_mean_clipped": float(np.maximum(finite, 0).mean()) if finite.size else None,
        "bps_median": float(np.median(finite)) if finite.size else None,
        "n_units": int(finite.size),
        "n_batches": batches_scored,
    }


def score_behavior_ablation_subset(
    model, loader, device, max_batches, dataset_idx_override=None
):
    """Compare behavior conditions on exactly the same held-out examples."""
    from models.losses import PoissonBPSAggregator

    aggregates = {
        "observed": PoissonBPSAggregator(),
        "permuted_within_batch": PoissonBPSAggregator(),
        "zeroed": PoissonBPSAggregator(),
    }
    batches_scored = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = _to_device(_as_one_batch(raw_batch), device)
            behavior = batch.get("behavior")
            if behavior is None:
                return {"available": False, "reason": "checkpoint has no behavior input"}
            output_behavior = batch.get("output_behavior")
            dataset_idx = (
                int(dataset_idx_override)
                if dataset_idx_override is not None
                else int(batch["dataset_idx"][0])
            )
            conditions = {
                "observed": (
                    behavior,
                    output_behavior,
                ),
                "permuted_within_batch": (
                    behavior.roll(shifts=1, dims=0),
                    output_behavior.roll(shifts=1, dims=0)
                    if output_behavior is not None
                    else None,
                ),
                "zeroed": (
                    torch.zeros_like(behavior),
                    torch.zeros_like(output_behavior)
                    if output_behavior is not None
                    else None,
                ),
            }
            for name, (condition, output_condition) in conditions.items():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(
                        batch["stim"],
                        dataset_idx,
                        condition,
                        batch.get("history"),
                        output_condition,
                    )
                aggregates[name](
                    {
                        "rhat": prediction.float(),
                        "robs": batch["robs"].float(),
                        "dfs": batch["dfs"].float(),
                    }
                )
            batches_scored += 1

    summaries = {}
    for name, aggregate in aggregates.items():
        per_unit = aggregate.closure()
        values = per_unit.detach().cpu().numpy()
        finite = values[np.isfinite(values)]
        summaries[name] = {
            "bps_mean_clipped": (
                float(np.maximum(finite, 0).mean()) if finite.size else None
            ),
            "bps_median": float(np.median(finite)) if finite.size else None,
            "n_units": int(finite.size),
        }

    observed = summaries["observed"]["bps_mean_clipped"]
    for name in ("permuted_within_batch", "zeroed"):
        value = summaries[name]["bps_mean_clipped"]
        summaries[name]["delta_from_observed"] = (
            value - observed if value is not None and observed is not None else None
        )
    return {"available": True, "n_batches": batches_scored, "conditions": summaries}


def jacobian_metrics(jacobian, sampling_rate=240.0):
    """Dimensionless roughness, high-frequency energy, and context rank."""
    value = jacobian.float()
    denom = value.square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    temporal_roughness = value.diff(n=2, dim=1).square().mean(dim=(1, 2, 3)) / denom

    padded = torch.nn.functional.pad(value.flatten(0, 1).unsqueeze(1), (1, 1, 1, 1))
    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    laplacian = torch.nn.functional.conv2d(padded, kernel).reshape_as(value)
    spatial_roughness = laplacian.square().mean(dim=(1, 2, 3)) / denom

    temporal_frequency = torch.fft.rfftfreq(value.shape[1], d=1.0 / sampling_rate)
    temporal_power = torch.fft.rfft(value, dim=1).abs().square().sum(dim=(2, 3))
    temporal_high = temporal_power[:, temporal_frequency >= sampling_rate / 4].sum(dim=1)
    temporal_high = temporal_high / temporal_power.sum(dim=1).clamp_min(1e-12)

    fy = torch.fft.fftfreq(value.shape[2], device=value.device)[:, None]
    fx = torch.fft.fftfreq(value.shape[3], device=value.device)[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    spatial_power = torch.fft.fft2(value, dim=(-2, -1)).abs().square().sum(dim=1)
    spatial_high = spatial_power[:, radius >= 0.25].sum(dim=1)
    spatial_high = spatial_high / spatial_power.sum(dim=(1, 2)).clamp_min(1e-12)

    flat = value.flatten(1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    singular_values = torch.linalg.svdvals(flat)
    cumulative = singular_values.square().cumsum(0) / singular_values.square().sum()
    rank80 = int(torch.searchsorted(cumulative, value.new_tensor(0.8)).item() + 1)

    return {
        "temporal_second_difference_ratio": temporal_roughness.cpu().tolist(),
        "spatial_laplacian_ratio": spatial_roughness.cpu().tolist(),
        "temporal_power_at_or_above_60hz": temporal_high.cpu().tolist(),
        "spatial_power_at_or_above_0p25_cycles_per_pixel": spatial_high.cpu().tolist(),
        "normalized_context_singular_values": singular_values.cpu().tolist(),
        "rank_for_80pct_context_jacobian_energy": rank80,
    }


def first_layer_metrics(core, sampling_rate=240.0):
    """Apply the same roughness audit to the effective 3-D stem filters."""
    weight = core.effective_temporal_weight().detach().float().cpu()
    if weight.ndim != 5 or weight.shape[1] != 1:
        raise ValueError(
            "First-layer metrics require filters shaped (filters, 1, time, y, x)"
        )
    _, _, separability = core.first_layer_separable_components()
    metrics = jacobian_metrics(weight[:, 0], sampling_rate=sampling_rate)
    metrics["rank1_separability"] = separability.float().cpu().tolist()
    metrics["rank_for_80pct_filter_bank_energy"] = metrics.pop(
        "rank_for_80pct_context_jacobian_energy"
    )
    metrics["normalized_filter_bank_singular_values"] = metrics.pop(
        "normalized_context_singular_values"
    )
    return metrics


def exact_jacobians(
    model,
    raw_batch,
    device,
    n_units,
    contexts_per_unit,
    unit_indices=None,
    dataset_idx_override=None,
):
    batch = _to_device(_as_one_batch(raw_batch), device)
    needed = n_units * contexts_per_unit
    if batch["stim"].shape[0] < needed:
        raise ValueError(f"Need {needed} examples, batch has {batch['stim'].shape[0]}")

    stimulus = batch["stim"][:needed].float().detach().requires_grad_(True)
    behavior = batch.get("behavior")
    if behavior is not None:
        behavior = behavior[:needed].float()
    output_behavior = batch.get("output_behavior")
    if output_behavior is not None:
        output_behavior = output_behavior[:needed].float()
    dataset_idx = (
        int(dataset_idx_override)
        if dataset_idx_override is not None
        else int(batch["dataset_idx"][0])
    )

    model.eval()
    prediction = model(
        stimulus,
        dataset_idx,
        behavior,
        None,
        output_behavior,
    )
    valid = batch["dfs"][:needed].sum(dim=0) > 0
    score = prediction.detach().mean(dim=0)
    score[~valid] = -torch.inf
    if unit_indices is None:
        selected_units = score.topk(min(n_units, int(valid.sum()))).indices
    else:
        selected_units = torch.as_tensor(unit_indices, device=device, dtype=torch.long)
        if selected_units.numel() != n_units:
            raise ValueError(
                f"Received {selected_units.numel()} fixed units; expected {n_units}"
            )
        if selected_units.min() < 0 or selected_units.max() >= prediction.shape[1]:
            raise ValueError("Fixed unit index is outside the model output range")
        if not valid[selected_units].all():
            raise ValueError("At least one fixed unit is invalid in this batch")
    if selected_units.numel() != n_units:
        raise ValueError(f"Only {selected_units.numel()} valid units; requested {n_units}")

    unit_for_context = selected_units.repeat_interleave(contexts_per_unit)
    rows = torch.arange(needed, device=device)
    target = prediction[rows, unit_for_context].clamp_min(1e-8).log().sum()
    jacobian = torch.autograd.grad(target, stimulus, create_graph=False)[0][:, 0]

    return (
        jacobian.detach().cpu(),
        selected_units.detach().cpu(),
        prediction.detach().cpu(),
    )


def plot_jacobians(jacobian, units, contexts_per_unit, out_path, sampling_rate=240.0):
    import matplotlib.pyplot as plt

    n_units = len(units)
    figure, axes = plt.subplots(n_units, 3, figsize=(12, 2.7 * n_units), squeeze=False)
    time_ms = np.arange(jacobian.shape[1]) * 1000.0 / sampling_rate

    for row, unit in enumerate(units.tolist()):
        contexts = jacobian[
            row * contexts_per_unit : (row + 1) * contexts_per_unit
        ]
        temporal_energy = contexts.square().mean(dim=(-2, -1)).sqrt()
        for context_idx in range(contexts_per_unit):
            axes[row, 0].plot(
                time_ms,
                temporal_energy[context_idx],
                label=f"context {context_idx + 1}",
            )
        axes[row, 0].set_ylabel(f"unit {unit}\nRMS |d log rate/dx|")
        if row == 0:
            axes[row, 0].set_title("Temporal Jacobian energy")
        if row == n_units - 1:
            axes[row, 0].set_xlabel("history position (ms)")

        for context_idx in range(min(2, contexts_per_unit)):
            temporal_index = int(temporal_energy[context_idx].argmax())
            image = contexts[context_idx, temporal_index]
            vmax = float(image.abs().max().clamp_min(1e-12))
            axes[row, context_idx + 1].imshow(
                image, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest"
            )
            axes[row, context_idx + 1].set_title(
                f"context {context_idx + 1}, peak {temporal_index * 1000 / sampling_rate:.0f} ms"
            )
            axes[row, context_idx + 1].axis("off")

    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Exact validation-context input Jacobians (log predicted rate)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset-idx", type=int, default=0)
    parser.add_argument("--n-units", type=int, default=4)
    parser.add_argument("--contexts-per-unit", type=int, default=2)
    parser.add_argument(
        "--unit-indices",
        type=int,
        nargs="+",
        default=None,
        help="Use these units in this exact order instead of selecting top-rate units",
    )
    parser.add_argument("--performance-batches", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.unit_indices is not None:
        args.n_units = len(args.unit_indices)

    from eval.load_twin import load_twin
    from training.pl_modules import MultiDatasetDM

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dir = (checkpoint.get("hyper_parameters", {}) or {}).get("cfg_dir")
    if cfg_dir is None:
        raise ValueError("Checkpoint does not record cfg_dir")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    model, info = load_twin(checkpoint_path, device=str(device), verbose=True)

    dataset_name = model.names[args.dataset_idx]
    dm = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=1,
        batch=max(64, args.n_units * args.contexts_per_unit),
        workers=0,
        steps_per_epoch=1,
        dset_dtype="uint8",
        homogeneous_batches=True,
        dataset_names=[dataset_name],
    )
    dm.setup("fit")
    if dm.names != [dataset_name]:
        raise RuntimeError(
            f"Targeted dataset load drifted: requested={dataset_name}, data={dm.names}"
        )

    out_dir = args.out_dir or (
        ROOT / "outputs" / "dekel240_diagnostics" / checkpoint_path.parent.name
        / f"epoch_{int(checkpoint.get('epoch', -1)):03d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    core = model.model.convnet
    core_variants = {"base": core}
    for core_name, attribute in (
        ("auxiliary", "auxiliary_convnet"),
        ("residual", "residual_convnet"),
    ):
        candidate = getattr(model.model, attribute, None)
        if candidate is not None:
            core_variants[core_name] = candidate

    first_layer_reports = {}
    first_layer_artifacts = {}
    for core_name, selected_core in core_variants.items():
        core_out_dir = out_dir if core_name == "base" else out_dir / f"{core_name}_core"
        core_out_dir.mkdir(parents=True, exist_ok=True)
        temporal_figure = selected_core.plot_temporal_filters(sampling_rate=240.0)
        temporal_path = core_out_dir / "first_layer_temporal_and_frequency.png"
        temporal_figure.savefig(temporal_path, dpi=180, bbox_inches="tight")
        spatial_figure = selected_core.plot_first_layer_spatial_filters()
        spatial_path = core_out_dir / "first_layer_spatial.png"
        spatial_figure.savefig(spatial_path, dpi=180, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(temporal_figure)
        plt.close(spatial_figure)
        first_layer_reports[core_name] = first_layer_metrics(selected_core)
        first_layer_artifacts[core_name] = {
            "first_layer_temporal": str(temporal_path),
            "first_layer_spatial": str(spatial_path),
        }

    validation_loader = dm.val_dataloader()
    raw_batch = next(iter(validation_loader))
    jacobian, units, _ = exact_jacobians(
        model,
        raw_batch,
        device,
        n_units=args.n_units,
        contexts_per_unit=args.contexts_per_unit,
        unit_indices=args.unit_indices,
        dataset_idx_override=args.dataset_idx,
    )
    jacobian_path = out_dir / "exact_input_jacobians.png"
    plot_jacobians(
        jacobian,
        units,
        args.contexts_per_unit,
        jacobian_path,
    )

    performance = score_validation_subset(
        model,
        dm.val_dataloader(),
        device,
        args.performance_batches,
        dataset_idx_override=args.dataset_idx,
    )
    behavior_ablation = score_behavior_ablation_subset(
        model,
        dm.val_dataloader(),
        device,
        args.performance_batches,
        dataset_idx_override=args.dataset_idx,
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_bps": _checkpoint_score(checkpoint),
        "dataset_idx": args.dataset_idx,
        "dataset": dataset_name,
        "model_units_all_sessions": info["n_units"],
        "selected_unit_indices": units.tolist(),
        "performance_subset": performance,
        "behavior_ablation_subset": behavior_ablation,
        "readout_widths": summarize_readout_widths(
            model, checkpoint["hyper_parameters"]["model_config_dict"]
        ),
        "residual_readouts": summarize_residual_readouts(model),
        "auxiliary_residual_readouts": summarize_residual_readouts(
            model, "auxiliary_residual_readouts"
        ),
        "residual_visual_readouts": summarize_residual_readouts(
            model, "residual_visual_readouts"
        ),
        "exact_parameter_zeros": summarize_exact_zeros(model),
        # Keep the original scalar field for downstream consumers while also
        # reporting every visual branch in newer additive-core checkpoints.
        "first_layer": first_layer_reports["base"],
        "first_layers": first_layer_reports,
        "jacobian": jacobian_metrics(jacobian),
        "artifacts": {
            **first_layer_artifacts["base"],
            "first_layers": first_layer_artifacts,
            "jacobians": str(jacobian_path),
        },
    }
    report_path = out_dir / "diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
