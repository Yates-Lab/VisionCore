#!/usr/bin/env python3
"""Track motion-related spatial information through an actual native-240 twin.

This audit replays matched measured-motion and stabilized natural-image movies
through the unmodified model.  It contains no linearized, affine, ablated, or
counterfactual network.  Every internal value is an activation produced during
the ordinary forward pass.

Signed pre-normalization activations are represented as their positive and
negative half-wave channels before spatial information is measured.  That is
the same representation materialized by the model's split-ReLU and makes the
pre/post comparisons independent of an arbitrary sign convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    PPD,
    RealTraceMatrixScorer,
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
    make_counterfactual_stim,
)
from paper.fig4.spatiotemporal_tuning.run_retinal_causal_chain import (
    DEFAULT_MCFARLAND,
)


EPS = 1e-12
STAGES = (
    "temporal pre-normalization",
    "temporal after GroupNorm/LRN",
    "temporal split-ReLU",
    "spatial stage 1 pre-normalization",
    "spatial stage 1 after GroupNorm/LRN",
    "spatial stage 1 split-ReLU",
    "spatial stage 2 pre-normalization",
    "spatial stage 2 after GroupNorm/LRN",
    "spatial stage 2 split-ReLU",
    "spatial stage 3 pre-normalization",
    "spatial stage 3 after GroupNorm/LRN",
    "spatial stage 3 split-ReLU",
    "population output",
)
SIGNED_STAGE_INDICES = frozenset((0, 1, 3, 4, 6, 7, 9, 10))
BLOCKS = (
    ("temporal", 0, 2),
    ("S1", 3, 5),
    ("S2", 6, 8),
    ("S3", 9, 11),
)
OUTPUT_TAPS = (
    ("temporal", 2),
    ("S1", 5),
    ("S2", 8),
    ("S3", 11),
    ("output", 12),
)
METRIC_NAMES = ("mean_activation", "spatial_information")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_rows(length: int, requested: int) -> np.ndarray:
    if length < 1 or requested < 1:
        raise ValueError("available and requested row counts must be positive")
    rows = np.unique(
        np.round(np.linspace(0, length - 1, min(length, requested))).astype(int)
    )
    if len(rows) != min(length, requested):
        raise RuntimeError("deterministic row selection produced duplicates")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, required=True)
    parser.add_argument("--trace-array", type=Path, required=True)
    parser.add_argument("--trace-table", type=Path, required=True)
    parser.add_argument("--trace-provenance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, required=True)
    parser.add_argument("--population-version", required=True)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND)
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--n-traces", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--patch-size-px", type=int, default=540)
    return parser.parse_args()


def positive_representation(value: torch.Tensor, *, signed: bool) -> torch.Tensor:
    """Return the nonnegative event channels used by the information metric."""
    raw = value.float()
    if not signed:
        if torch.any(raw < -1e-6):
            raise ValueError("a declared nonnegative stage contains negative values")
        return raw.clamp_min(0.0)
    # The selected core materializes exactly these two half-wave channels after each
    # signed GroupNorm/LRN coordinate.  Channel order is irrelevant below.
    return torch.cat((F.relu(raw), F.relu(-raw)), dim=1)


def information_components(
    value: torch.Tensor, *, signed: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return information numerator, activation mass, total value, and pixels."""
    positive = positive_representation(value, signed=signed)
    flat = positive.flatten(start_dim=2).double()
    mean = flat.mean(dim=2)
    gain = flat / mean[..., None].clamp_min(EPS)
    bits = (gain * gain.clamp_min(EPS).log2()).mean(dim=2)
    return (
        (bits * mean).sum(),
        mean.sum(),
        flat.sum(),
        int(flat.numel()),
    )


def finalize_components(
    numerator: np.ndarray,
    mass: np.ndarray,
    total: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    """Convert exactly pooled sufficient statistics into saved stage metrics."""
    information = numerator / np.maximum(mass, EPS)
    mean_activation = total / np.maximum(count, 1)
    return np.stack((mean_activation, information), axis=-1)


def hierarchy_function(scorer: RealTraceMatrixScorer):
    """Expose actual core taps while retaining the exact population output."""
    core = scorer.model.model.convnet

    def forward(stimulus: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temporal_pre = core.temporal_conv(stimulus)
        if temporal_pre.shape[2] != 1:
            raise RuntimeError("the complete temporal history did not collapse time")
        temporal_pre = temporal_pre.squeeze(2)
        stem, temporal_signed = core.temporal_nonlinearity.forward_with_signed(
            temporal_pre
        )
        stage1_pre = core.stage1_conv(stem)
        stage1, stage1_signed = core.stage1_nonlinearity.forward_with_signed(stage1_pre)
        stage2_pre = core.stage2_conv(core._downsample(stage1))
        stage2, stage2_signed = core.stage2_nonlinearity.forward_with_signed(stage2_pre)
        stage3_pre = core.stage3_conv(core._downsample(stage2))
        stage3, stage3_signed = core.stage3_nonlinearity.forward_with_signed(stage3_pre)
        output = scorer.apply_population_view(
            scorer._compute_rate_map(stimulus), scorer.population_view
        ).clamp_min(0.0)
        return (
            temporal_pre,
            temporal_signed,
            stem,
            stage1_pre,
            stage1_signed,
            stage1,
            stage2_pre,
            stage2_signed,
            stage2,
            stage3_pre,
            stage3_signed,
            stage3,
            output,
        )

    return forward


def score_condition(
    forward,
    histories: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    numerator = np.zeros(len(STAGES), dtype=np.float64)
    mass = np.zeros(len(STAGES), dtype=np.float64)
    total = np.zeros(len(STAGES), dtype=np.float64)
    count = np.zeros(len(STAGES), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(histories), int(batch_size)):
            batch = histories[start : start + int(batch_size)].to(device)
            outputs = forward(batch)
            if len(outputs) != len(STAGES):
                raise RuntimeError("hierarchy tap count changed")
            for stage, value in enumerate(outputs):
                parts = information_components(
                    value, signed=stage in SIGNED_STAGE_INDICES
                )
                numerator[stage] += float(parts[0].detach().cpu())
                mass[stage] += float(parts[1].detach().cpu())
                total[stage] += float(parts[2].detach().cpu())
                count[stage] += int(parts[3])
    return numerator, mass, total, count


def causal_histories(
    patch: np.ndarray,
    trace_xy: np.ndarray,
    *,
    n_lags: int,
    out_size: tuple[int, int],
    temporal_factor: int,
    supervision_phase: int,
) -> torch.Tensor:
    """Use the identical causal held-initial-gaze policy as the response matrix."""
    image = _standardize_uint_like(patch)
    trace = np.asarray(trace_xy, dtype=np.float32)
    full_stack = np.broadcast_to(
        image[None],
        (len(trace) * int(temporal_factor) + int(n_lags) + 1, *image.shape),
    ).copy()
    eye = torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    histories = make_counterfactual_stim(
        full_stack,
        eye,
        ppd=PPD,
        scale_factor=1.0,
        n_lags=int(n_lags),
        out_size=tuple(map(int, out_size)),
        temporal_factor=int(temporal_factor),
        supervision_phase=int(supervision_phase),
    )
    return (histories - 127.0) / 255.0


def crossed_bootstrap_median(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(values, dtype=float)
    if value.ndim != 3:
        raise ValueError("crossed bootstrap expects [image, trace, measure]")
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_bootstrap), value.shape[2]), dtype=float)
    for draw in range(int(n_bootstrap)):
        images = rng.integers(0, value.shape[0], value.shape[0])
        traces = rng.integers(0, value.shape[1], value.shape[1])
        draws[draw] = np.nanmedian(value[np.ix_(images, traces)], axis=(0, 1))
    return (
        np.nanmedian(value, axis=(0, 1)),
        np.nanquantile(draws, 0.025, axis=0),
        np.nanquantile(draws, 0.975, axis=0),
    )


def crossed_bootstrap_pooled_percent(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a crossed-bootstrap percent change of a pooled ratio.

    ``numerator`` and ``denominator`` must be aligned arrays with dimensions
    ``[image, trace, condition, stage]`` and condition order
    ``(stabilized, measured motion)``.  Images and traces are pooled before
    forming the ratio and its percent change.  This is deliberately different
    from taking the median of pairwise percentages: the latter does not
    reproduce a pooled population estimand.
    """
    numer = np.asarray(numerator, dtype=np.float64)
    weight = np.asarray(denominator, dtype=np.float64)
    if numer.shape != weight.shape or numer.ndim != 4 or numer.shape[2] != 2:
        raise ValueError(
            "pooled information bootstrap expects aligned "
            "[image, trace, 2 conditions, stage] arrays"
        )
    if (
        np.any(~np.isfinite(numer))
        or np.any(~np.isfinite(weight))
        or np.any(weight < 0)
    ):
        raise ValueError("pooled information components must be finite and nonnegative")

    def effect(image_rows: np.ndarray, trace_rows: np.ndarray) -> np.ndarray:
        selected_numerator = numer[image_rows][:, trace_rows]
        selected_mass = weight[image_rows][:, trace_rows]
        pooled_mass = selected_mass.sum(axis=(0, 1))
        if np.any(pooled_mass <= EPS):
            raise ValueError("pooled information requires positive activation mass")
        information = selected_numerator.sum(axis=(0, 1)) / pooled_mass
        baseline = information[0]
        if np.any(baseline <= EPS):
            raise ValueError("pooled percent change requires positive baseline information")
        return 100.0 * (information[1] - baseline) / baseline

    image_rows = np.arange(numer.shape[0], dtype=int)
    trace_rows = np.arange(numer.shape[1], dtype=int)
    center = effect(image_rows, trace_rows)
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_bootstrap), numer.shape[3]), dtype=np.float64)
    for draw in range(int(n_bootstrap)):
        images = rng.integers(0, numer.shape[0], size=numer.shape[0])
        traces = rng.integers(0, numer.shape[1], size=numer.shape[1])
        draws[draw] = effect(images, traces)
    return (
        center,
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
    )


def crossed_bootstrap_pooled_information_percent(
    numerator: np.ndarray,
    mass: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Specialized public name for pooled spatial-information modulation."""
    return crossed_bootstrap_pooled_percent(
        numerator,
        mass,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def render_audit(
    path: Path,
    baseline: np.ndarray,
    actual: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    n_pairs: int,
) -> dict[str, object]:
    metric = METRIC_NAMES.index("spatial_information")
    effect = actual[..., metric] - baseline[..., metric]
    center, low, high = crossed_bootstrap_median(
        effect, n_bootstrap=n_bootstrap, seed=seed
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)

    tap_indices = np.asarray([stage for _, stage in OUTPUT_TAPS], dtype=int)
    x = np.arange(len(tap_indices))
    colors = ("0.58", "#93C47D", "#5BAE61", "#2E8540", "#6A51A3")
    axes[0].errorbar(
        x,
        center[tap_indices],
        yerr=np.vstack(
            (center[tap_indices] - low[tap_indices], high[tap_indices] - center[tap_indices])
        ),
        fmt="o-",
        color="#2E8540",
        markerfacecolor="white",
        markeredgewidth=1.2,
        lw=1.6,
        capsize=2.5,
    )
    for location, value, color in zip(x, center[tap_indices], colors):
        axes[0].scatter(location, value, s=34, color=color, edgecolor="white", zorder=4)
    axes[0].set_xticks(x, [label for label, _ in OUTPUT_TAPS])
    axes[0].set_title("actual stage outputs", fontsize=9.0, fontweight="semibold")

    block_labels = [label for label, _, _ in BLOCKS]
    pre_indices = np.asarray([pre for _, pre, _ in BLOCKS], dtype=int)
    post_indices = np.asarray([post for _, _, post in BLOCKS], dtype=int)
    x = np.arange(len(BLOCKS), dtype=float)
    for location, before, after in zip(x, center[pre_indices], center[post_indices]):
        axes[1].plot(
            (location - 0.13, location + 0.13),
            (before, after),
            color="0.55",
            lw=1.0,
            zorder=1,
        )
    for indices, offset, color, label in (
        (pre_indices, -0.13, "0.58", "before normalization"),
        (post_indices, 0.13, "#2E8540", "after normalization + split"),
    ):
        axes[1].errorbar(
            x + offset,
            center[indices],
            yerr=np.vstack((center[indices] - low[indices], high[indices] - center[indices])),
            fmt="o",
            color=color,
            markerfacecolor=color,
            lw=1.0,
            capsize=2.2,
            label=label,
            zorder=3,
        )
    axes[1].set_xticks(x, block_labels)
    axes[1].set_title("before and after each nonlinear block", fontsize=9.0, fontweight="semibold")
    axes[1].legend(frameon=False, fontsize=7.0, loc="best")

    for axis in axes:
        axis.axhline(0, color="0.5", lw=0.8)
        axis.set_ylabel("motion − stabilized spatial information\n(bits/event)")
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Where motion-related sharpening emerges in the actual twin · {n_pairs} matched movies",
        fontsize=11.0,
        fontweight="semibold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=260, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    increments = effect[..., post_indices] - effect[..., pre_indices]
    inc_center, inc_low, inc_high = crossed_bootstrap_median(
        increments, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    return {
        "stage_effect_bits_per_event": center.tolist(),
        "stage_effect_ci95": np.column_stack((low, high)).tolist(),
        "block_increment_bits_per_event": {
            label: {
                "median": float(value),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
            for label, value, lo, hi in zip(block_labels, inc_center, inc_low, inc_high)
        },
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.image_table,
        args.trace_array,
        args.trace_table,
        args.trace_provenance,
        args.checkpoint,
        args.dataset_config,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    images = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    trace_table = pd.read_csv(args.trace_table).reset_index(drop=True)
    traces = np.load(args.trace_array, mmap_mode="r")
    if traces.ndim != 3 or traces.shape[-1] != 2 or len(trace_table) != len(traces):
        raise ValueError("trace table and [trace,time,2] array are not aligned")
    provenance = json.loads(args.trace_provenance.read_text(encoding="utf-8"))
    filter_kind = str((provenance.get("filter") or {}).get("kind", ""))
    if "zero-phase" not in filter_kind.lower():
        raise ValueError("the nonlinear audit requires zero-phase-filtered eye traces")
    if int(provenance.get("sample_rate_hz", 0)) != 240:
        raise ValueError("the nonlinear audit requires the native 240-Hz trace replay")

    image_rows = evenly_spaced_rows(len(images), int(args.n_images))
    trace_rows = evenly_spaced_rows(len(trace_table), int(args.n_traces))
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_config.resolve(),
        population_spec_dir=args.population_spec_dir.resolve(),
        population_version=str(args.population_version),
        device=str(args.device),
        strict=True,
        mcfarland_outputs=args.mcfarland_outputs.resolve(),
    )
    if scorer.input_rate_hz != 240 or scorer.output_rate_hz != 240:
        raise ValueError("this audit requires a native 240-Hz input/output contract")
    scorer.model.model.eval()
    scorer.readout.eval()
    forward = hierarchy_function(scorer)

    shape = (len(image_rows), len(trace_rows), 2, len(STAGES))
    numerator = np.zeros(shape, dtype=np.float64)
    mass = np.zeros(shape, dtype=np.float64)
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)
    canvas_cache: dict = {}
    for image_position, image_row in enumerate(image_rows):
        patch, _ = extract_patch(
            images.iloc[int(image_row)],
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        for trace_position, trace_row in enumerate(trace_rows):
            trace = np.asarray(traces[int(trace_row)], dtype=np.float32)
            for condition, condition_trace in enumerate((np.zeros_like(trace), trace)):
                histories = causal_histories(
                    patch,
                    condition_trace,
                    n_lags=scorer.n_lags,
                    out_size=scorer.out_size,
                    temporal_factor=scorer.temporal_factor,
                    supervision_phase=scorer.supervision_phase,
                )
                parts = score_condition(
                    forward,
                    histories,
                    device=scorer.device,
                    batch_size=int(args.batch_size),
                )
                numerator[image_position, trace_position, condition] = parts[0]
                mass[image_position, trace_position, condition] = parts[1]
                total[image_position, trace_position, condition] = parts[2]
                count[image_position, trace_position, condition] = parts[3]
        print(
            f"actual nonlinear audit image {image_position + 1}/{len(image_rows)}",
            flush=True,
        )

    metrics = finalize_components(numerator, mass, total, count)
    baseline = metrics[:, :, 0]
    actual = metrics[:, :, 1]
    # The signed coordinate after each presplit normalization and its explicit
    # split-ReLU carry identical event maps.  Treat a violation as an
    # architecture/measurement bug, not as a scientific result.
    split_pairs = ((1, 2), (4, 5), (7, 8), (10, 11))
    split_error = max(
        float(np.max(np.abs(metrics[..., left, :] - metrics[..., right, :])))
        for left, right in split_pairs
    )
    if split_error > 2e-5:
        raise RuntimeError(
            f"signed-to-split representation gate failed: max error {split_error:.3g}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "nonlinear_sharpening.npz"
    np.savez_compressed(
        archive,
        baseline=baseline.astype(np.float32),
        actual=actual.astype(np.float32),
        actual_motion_effect=(actual - baseline).astype(np.float32),
        metric_names=np.asarray(METRIC_NAMES),
        stage_names=np.asarray(STAGES),
        information_numerator=numerator,
        activation_mass=mass,
        activation_total=total,
        activation_count=count,
        condition_names=np.asarray(("stabilized", "measured motion")),
        image_rows=image_rows,
        trace_rows=trace_rows,
    )
    figure = args.out_dir / "actual_nonlinear_sharpening.png"
    report = render_audit(
        figure,
        baseline,
        actual,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        n_pairs=int(len(image_rows) * len(trace_rows)),
    )
    summary = {
        "analysis": "actual measured-motion versus stabilized activations through the unmodified network",
        "model_label": str(args.model_label),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256(args.dataset_config.resolve()),
        "population_version": str(args.population_version),
        "population_n_units": int(scorer.n_units),
        "behavior": "identical all-zero 42-dimensional vector",
        "trace_filter": filter_kind,
        "history_prefix_policy": "hold initial gaze for n_lags minus 1; identical to production response matrix",
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_image_trace_pairs": int(len(image_rows) * len(trace_rows)),
        "n_scored_timepoints_per_pair": int(traces.shape[1]),
        "image_rows": image_rows.tolist(),
        "trace_rows": trace_rows.tolist(),
        "inputs": {
            "image_table": str(args.image_table.resolve()),
            "image_table_sha256": sha256(args.image_table),
            "trace_array": str(args.trace_array.resolve()),
            "trace_array_sha256": sha256(args.trace_array),
            "trace_table": str(args.trace_table.resolve()),
            "trace_table_sha256": sha256(args.trace_table),
            "trace_provenance": str(args.trace_provenance.resolve()),
            "trace_provenance_sha256": sha256(args.trace_provenance),
        },
        "metric_definition": (
            "activation-mass-weighted spatial information; signed coordinates are "
            "represented by their positive and negative half-wave event channels"
        ),
        "synthetic_network_reference": False,
        "actual_activations_only": True,
        "signed_to_split_max_metric_error": split_error,
        "inference": "crossed image and trace bootstrap of the median",
        "archive": str(archive.resolve()),
        "figure": str(figure.resolve()),
        **report,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
