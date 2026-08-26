#!/usr/bin/env python3
"""Trace top-passband motion effects through the model's additive output readout.

This analysis does not linearize, ablate, or replace any network block.  It
uses the three real Dekel-core feature groups and the trained additive spatial
readout to form cumulative output logits:

    bias/zero-behavior baseline + S1 + phase, then + S2, then + S3.

The ordinary softplus output link is applied after every cumulative sum.  The
last cumulative map is therefore required to be numerically identical to the
unmodified model output.  Movies are selected with the exact unit-specific
passband percentiles used by Figure 4G; only unit--trace pairs in the top 20%
of their own passband distribution contribute to the displayed trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

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

from paper.fig4.spatiotemporal_tuning._spectral_shards import (  # noqa: E402
    load_and_merge_shards,
)
from paper.fig4.spatiotemporal_tuning.audit_actual_nonlinear_sharpening import (  # noqa: E402
    causal_histories,
    evenly_spaced_rows,
)
from paper.fig4.spatiotemporal_tuning._figure4_renderer import (  # noqa: E402
    _direct_mechanism_values,
)
from paper.fig4.spatiotemporal_tuning.run_retinal_causal_chain import (  # noqa: E402
    DEFAULT_MCFARLAND,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch  # noqa: E402
from paper.fig4.upstream.real_trace_matrix.model import (  # noqa: E402
    RealTraceMatrixScorer,
)


EPS = 1.0e-12
STAGE_NAMES = ("S1 + phase", "+ S2", "+ S3 / output")
CONDITION_NAMES = ("stabilized", "measured motion")
RATE_HZ = 240.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, required=True)
    parser.add_argument("--trace-array", type=Path, required=True)
    parser.add_argument("--trace-table", type=Path, required=True)
    parser.add_argument("--trace-provenance", type=Path, required=True)
    parser.add_argument("--spectral-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, required=True)
    parser.add_argument("--population-version", required=True)
    parser.add_argument("--mcfarland-outputs", type=Path, default=DEFAULT_MCFARLAND)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--n-traces", type=int, default=10)
    parser.add_argument("--top-percentile", type=float, default=80.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_top_traces(
    percentile: np.ndarray,
    *,
    n_traces: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify traces across the full top bin without looking at responses."""
    rank = np.asarray(percentile, dtype=float)
    if rank.ndim != 2 or not np.all(np.isfinite(rank)):
        raise ValueError("passband percentile must be finite [trace, unit]")
    if not 0.0 < float(threshold) < 100.0:
        raise ValueError("top-percentile threshold must lie in (0, 100)")
    membership = rank > float(threshold)
    counts = membership.sum(axis=1)
    if int(n_traces) > len(counts):
        raise ValueError("requested more traces than the spectral replay contains")
    conditional_median = np.asarray(
        [
            np.median(rank[row, membership[row]])
            if counts[row]
            else np.nan
            for row in range(len(rank))
        ],
        dtype=float,
    )
    # A smoke test should represent the whole 80--100th-percentile bin, not
    # merely its most extreme tail.  Restrict to traces that contribute to a
    # substantial fraction of units, then match evenly spaced percentile
    # targets.  Counts break ties and unit responses are never consulted.
    minimum_count = max(1, int(math.ceil(0.35 * rank.shape[1])))
    targets = np.linspace(
        float(threshold) + (100.0 - float(threshold)) / int(n_traces),
        100.0 - (100.0 - float(threshold)) / (2.0 * int(n_traces)),
        int(n_traces),
    )
    selected: list[int] = []
    for target in targets:
        candidates = np.flatnonzero(
            (counts >= minimum_count)
            & ~np.isin(np.arange(len(counts)), np.asarray(selected, dtype=int))
        )
        if not len(candidates):
            raise RuntimeError("too few broadly shared traces to stratify the top bin")
        selected.append(
            min(
                candidates.tolist(),
                key=lambda row: (
                    abs(float(conditional_median[row]) - float(target)),
                    -int(counts[row]),
                    int(row),
                ),
            )
        )
    selected_array = np.asarray(selected, dtype=int)
    covered = membership[selected_array].any(axis=0)
    if not np.all(covered):
        missing = np.flatnonzero(~covered)
        raise RuntimeError(
            "selected smoke-test traces do not cover every unit's top bin; "
            f"missing unit rows {missing.tolist()}"
        )
    return selected_array, membership


def _aligned_stages(core: Any, stages: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    target = tuple(stages[-1].shape[-2:])
    kwargs: dict[str, Any] = {}
    if core.scaffold_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return tuple(
        value
        if tuple(value.shape[-2:]) == target
        else F.interpolate(value, size=target, mode=core.scaffold_mode, **kwargs)
        for value in stages
    )


def _deep_map(readout: Any, feature: torch.Tensor) -> torch.Tensor:
    return readout._factorized_map(
        feature[:, :, -1],
        readout.features,
        readout.space_weights,
        readout.rank,
        readout.output_scale,
    )


def _phase_map(readout: Any, phase: torch.Tensor) -> torch.Tensor:
    if not readout.has_phase_branch:
        raise ValueError("the production cumulative trajectory requires its phase branch")
    value = readout._factorized_map(
        phase,
        readout.phase_features,
        readout.phase_space_weights,
        readout.phase_rank,
        readout.phase_output_scale,
    )
    return value[..., :: readout.phase_stride, :: readout.phase_stride]


def cumulative_rate_maps(
    scorer: RealTraceMatrixScorer,
    stimulus: torch.Tensor,
    *,
    check_identity: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return three cumulative real-output maps and exactness diagnostics."""
    module = scorer.model.model
    core = module.convnet
    front = module.frontend(stimulus)
    stages, phase = core._forward_stages_with_phase(front, strict_spatial=False)
    aligned = _aligned_stages(core, tuple(stages))
    scaffold = torch.cat(aligned, dim=1).unsqueeze(2)
    behavior = scorer._zero_behavior(len(scaffold), scaffold.dtype)

    # At fixed zero behavior the production modulator is channelwise affine in
    # the visual scaffold and appends a spatially constant behavior code.  The
    # zero-scaffold output contains that additive code exactly once.
    zero = torch.zeros_like(scaffold)
    if module.modulator is None:
        base = zero
        full = scaffold
        pieces = []
        start = 0
        for value in aligned:
            stop = start + value.shape[1]
            piece = torch.zeros_like(scaffold)
            piece[:, start:stop] = value[:, :, None]
            pieces.append(piece)
            start = stop
    else:
        base = module.modulator(zero, behavior)
        full = module.modulator(scaffold, behavior)
        pieces = []
        start = 0
        for value in aligned:
            stop = start + value.shape[1]
            piece = torch.zeros_like(scaffold)
            piece[:, start:stop] = value[:, :, None]
            pieces.append(module.modulator(piece, behavior) - base)
            start = stop

    # The released architecture's recurrent module is an identity. Keep the
    # check explicit so a later architecture cannot silently invalidate
    # additive attribution.
    base_post = module.recurrent(base)
    full_post = module.recurrent(full)
    piece_post = tuple(module.recurrent(base + value) - base_post for value in pieces)
    reconstruction = base_post + sum(piece_post)
    scaffold_error = float((reconstruction - full_post).abs().max().detach().cpu())
    if scaffold_error > 2.0e-6:
        raise RuntimeError(
            "fixed-behavior scaffold is not additively decomposable: "
            f"max error {scaffold_error:.3g}"
        )

    base_logit = _deep_map(scorer.readout, base_post)
    contributions = [_deep_map(scorer.readout, value) for value in piece_post]
    contributions[0] = contributions[0] + _phase_map(scorer.readout, phase)
    cumulative_logits = []
    current = base_logit + scorer.readout.bias[None, :, None, None]
    for contribution in contributions:
        current = current + contribution
        cumulative_logits.append(current)
    full_logit = scorer.readout(full_post[:, :, -1], phase)
    logit_error = float((cumulative_logits[-1] - full_logit).abs().max().detach().cpu())
    if logit_error > 2.0e-5:
        raise RuntimeError(
            f"cumulative readout does not reconstruct full logits: {logit_error:.3g}"
        )

    rates = torch.stack(
        [
            scorer.apply_population_view(module.activation(value), scorer.population_view)
            for value in cumulative_logits
        ],
        dim=0,
    ).clamp_min(0.0)
    output_error = 0.0
    if check_identity:
        ordinary = scorer.apply_population_view(
            scorer._compute_rate_map(stimulus), scorer.population_view
        ).clamp_min(0.0)
        output_error = float((rates[-1] - ordinary).abs().max().detach().cpu())
        if output_error > 2.0e-5:
            raise RuntimeError(
                f"final cumulative map differs from ordinary model: {output_error:.3g}"
            )
    return rates, {
        "scaffold_reconstruction_max_abs": scaffold_error,
        "logit_reconstruction_max_abs": logit_error,
        "ordinary_output_max_abs": output_error,
    }


def score_histories(
    scorer: RealTraceMatrixScorer,
    histories: torch.Tensor,
    *,
    batch_size: int,
    check_identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n_stages = len(STAGE_NAMES)
    n_units = scorer.n_units
    rate_sum = np.zeros((n_stages, n_units), dtype=np.float64)
    expected = np.zeros_like(rate_sum)
    information_numerator = np.zeros_like(rate_sum)
    temporal_sum: np.ndarray | None = None
    temporal_sum_squares: np.ndarray | None = None
    frames = 0
    diagnostic = {
        "scaffold_reconstruction_max_abs": 0.0,
        "logit_reconstruction_max_abs": 0.0,
        "ordinary_output_max_abs": 0.0,
    }
    with torch.no_grad():
        for start in range(0, len(histories), int(batch_size)):
            batch = histories[start : start + int(batch_size)].to(scorer.device)
            rate, check = cumulative_rate_maps(
                scorer,
                batch,
                check_identity=bool(check_identity and start == 0),
            )
            flat = rate.double().reshape(n_stages, rate.shape[1], rate.shape[2], -1)
            mean = flat.mean(dim=-1)
            gain = flat / mean[..., None].clamp_min(1.0e-8)
            bits = (gain * gain.clamp_min(1.0e-8).log2()).mean(dim=-1)
            rate_sum += mean.sum(dim=1).detach().cpu().numpy()
            expected += (mean.sum(dim=1) / RATE_HZ).detach().cpu().numpy()
            information_numerator += (
                (mean * bits).sum(dim=1) / RATE_HZ
            ).detach().cpu().numpy()
            rate_cpu = rate.double().detach().cpu().numpy()
            batch_sum = rate_cpu.sum(axis=1)
            batch_sum_squares = np.square(rate_cpu).sum(axis=1)
            if temporal_sum is None:
                temporal_sum = batch_sum
                temporal_sum_squares = batch_sum_squares
            else:
                temporal_sum += batch_sum
                temporal_sum_squares += batch_sum_squares
            frames += int(rate.shape[1])
            for key, value in check.items():
                diagnostic[key] = max(float(diagnostic[key]), float(value))
            del batch, rate, flat, mean, gain, bits
    mean_rate_hz = RATE_HZ * rate_sum / max(frames, 1)
    ssi = information_numerator / np.maximum(expected, EPS)
    if temporal_sum is None or temporal_sum_squares is None:
        raise RuntimeError("no response frames were scored")
    temporal_mean_map = temporal_sum / max(frames, 1)
    temporal_variance = np.maximum(
        temporal_sum_squares / max(frames, 1) - np.square(temporal_mean_map),
        0.0,
    )
    # Remove the time mean separately at every spatial location, then express
    # the remaining RMS temporal fluctuation relative to the movie-wide mean.
    # A scalar gain applied to the entire response cancels exactly.
    temporal_modulation = np.sqrt(temporal_variance.mean(axis=(-2, -1))) / np.maximum(
        temporal_mean_map.mean(axis=(-2, -1)), EPS
    )
    return mean_rate_hz, expected, ssi, temporal_modulation, diagnostic


def unit_top_bin_effects(
    rate: np.ndarray,
    expected: np.ndarray,
    ssi: np.ndarray,
    membership: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror Figure 4G's image pooling then trace/unit top-bin reduction."""
    if rate.ndim != 5 or rate.shape[2] != 2:
        raise ValueError("rate must be [image, trace, condition, stage, unit]")
    stable_rate = rate[:, :, 0].mean(axis=0)
    motion_rate = rate[:, :, 1].mean(axis=0)
    rate_percent = 100.0 * (motion_rate - stable_rate) / np.maximum(stable_rate, EPS)
    stable_ssi = np.sum(expected[:, :, 0] * ssi[:, :, 0], axis=0) / np.maximum(
        np.sum(expected[:, :, 0], axis=0), EPS
    )
    motion_ssi = np.sum(expected[:, :, 1] * ssi[:, :, 1], axis=0) / np.maximum(
        np.sum(expected[:, :, 1], axis=0), EPS
    )
    ssi_percent = 100.0 * (motion_ssi - stable_ssi) / np.maximum(stable_ssi, EPS)
    if membership.shape != rate_percent.shape[::2]:
        # rate_percent is [trace, stage, unit]; compare trace and unit axes.
        if membership.shape != (rate_percent.shape[0], rate_percent.shape[2]):
            raise ValueError("top-bin membership does not match trace/unit dimensions")
    unit_rate = np.full((rate_percent.shape[1], rate_percent.shape[2]), np.nan)
    unit_ssi = np.full_like(unit_rate, np.nan)
    for unit in range(rate_percent.shape[2]):
        mask = membership[:, unit]
        if np.any(mask):
            unit_rate[:, unit] = np.median(rate_percent[mask, :, unit], axis=0)
            unit_ssi[:, unit] = np.median(ssi_percent[mask, :, unit], axis=0)
    return unit_rate, unit_ssi


def unit_top_bin_normalized_effects(
    temporal_modulation: np.ndarray,
    expected: np.ndarray,
    ssi: np.ndarray,
    membership: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gain-invariant temporal modulation and absolute SSI changes."""
    temporal = np.asarray(temporal_modulation, dtype=float)
    if temporal.ndim != 5 or temporal.shape[2] != 2:
        raise ValueError(
            "temporal modulation must be [image, trace, condition, stage, unit]"
        )
    stable_temporal = temporal[:, :, 0].mean(axis=0)
    motion_temporal = temporal[:, :, 1].mean(axis=0)
    # Percentage points of the mean response, not percent change relative to a
    # near-zero stabilized denominator.
    temporal_points = 100.0 * (motion_temporal - stable_temporal)
    stable_ssi = np.sum(expected[:, :, 0] * ssi[:, :, 0], axis=0) / np.maximum(
        np.sum(expected[:, :, 0], axis=0), EPS
    )
    motion_ssi = np.sum(expected[:, :, 1] * ssi[:, :, 1], axis=0) / np.maximum(
        np.sum(expected[:, :, 1], axis=0), EPS
    )
    ssi_delta = motion_ssi - stable_ssi
    if membership.shape != (temporal.shape[1], temporal.shape[-1]):
        raise ValueError("top-bin membership does not match trace/unit dimensions")
    unit_temporal = np.full((temporal.shape[3], temporal.shape[4]), np.nan)
    unit_ssi = np.full_like(unit_temporal, np.nan)
    for unit in range(temporal.shape[4]):
        mask = membership[:, unit]
        if np.any(mask):
            unit_temporal[:, unit] = np.median(
                temporal_points[mask, :, unit], axis=0
            )
            unit_ssi[:, unit] = np.median(ssi_delta[mask, :, unit], axis=0)
    return unit_temporal, unit_ssi


def bootstrap_population(
    unit_values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(unit_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("unit values must be [stage, unit]")
    center = np.nanmedian(values, axis=1)
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_bootstrap), values.shape[0]), dtype=float)
    for draw in range(int(n_bootstrap)):
        units = rng.integers(0, values.shape[1], size=values.shape[1])
        draws[draw] = np.nanmedian(values[:, units], axis=1)
    return center, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def render_figure(
    path: Path,
    unit_temporal: np.ndarray,
    unit_ssi: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    colors = ("#D55E00", "#6A51A3")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    report: dict[str, Any] = {}
    x = np.arange(len(STAGE_NAMES))
    for axis, values, label, color, offset in zip(
        axes,
        (unit_temporal, unit_ssi),
        ("temporal modulation", "spatial sharpening"),
        colors,
        (0, 1000),
    ):
        center, low, high = bootstrap_population(
            values, n_bootstrap=n_bootstrap, seed=seed + offset
        )
        box = axis.boxplot(
            [row[np.isfinite(row)] for row in values],
            positions=x,
            widths=0.46,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
        )
        for artist in box["boxes"]:
            artist.set(facecolor=color, edgecolor=color, alpha=0.16, linewidth=0.8)
        for artist in (*box["whiskers"], *box["caps"]):
            artist.set(color=color, alpha=0.5, linewidth=0.7)
        for artist in box["medians"]:
            artist.set(color=color, linewidth=1.1)
        axis.errorbar(
            x,
            center,
            yerr=np.vstack((center - low, high - center)),
            fmt="o-",
            color=color,
            lw=1.7,
            capsize=2.5,
            zorder=4,
        )
        axis.axhline(0, color="0.55", lw=0.8)
        axis.set_title(label, fontsize=10, fontweight="bold")
        axis.set_ylabel(
            "motion − stabilized\n(% of mean response)"
            if label == "temporal modulation"
            else "motion − stabilized\n(bits/spike)"
        )
        axis.set_xticks(x, STAGE_NAMES, rotation=18, ha="right")
        axis.grid(axis="y", color="0.9", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        report[label] = {
            "center_percent": center.tolist(),
            "ci_low": low.tolist(),
            "ci_high": high.tolist(),
            "n_units": int(np.isfinite(values[-1]).sum()),
        }
    fig.suptitle(
        "Gain-invariant effects of top-passband movies across the cumulative readout",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return report


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace_provenance = json.loads(args.trace_provenance.read_text(encoding="utf-8"))
    filter_kind = str(trace_provenance.get("filter", {}).get("kind", ""))
    if "zero-phase" not in filter_kind.lower():
        raise ValueError("the selected fixation traces are not zero-phase filtered")
    if not bool(
        trace_provenance.get("spectral_filter_qc", {}).get(
            "stopband_suppression_gate", False
        )
    ):
        raise ValueError("the fixation-filter suppression gate did not pass")

    images = pd.read_csv(args.image_table)
    traces = np.load(args.trace_array)
    trace_table = pd.read_csv(args.trace_table)
    spectral = load_and_merge_shards(args.spectral_shards)
    direct = _direct_mechanism_values(spectral)
    percentile = np.asarray(direct["passband_percentile"], dtype=float)
    trace_rows, all_membership = select_top_traces(
        percentile,
        n_traces=int(args.n_traces),
        threshold=float(args.top_percentile),
    )
    image_rows = evenly_spaced_rows(len(images), int(args.n_images))
    membership = all_membership[trace_rows]
    if len(images) != spectral["mean_rate"].shape[0] or len(traces) != percentile.shape[0]:
        raise ValueError("spectral replay and image/trace inputs have different dimensions")
    if len(trace_table) != len(traces):
        raise ValueError("trace table and trace array disagree")

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=args.checkpoint,
        dataset_configs=args.dataset_config,
        population_spec_dir=args.population_spec_dir,
        population_version=str(args.population_version),
        device=str(args.device),
        strict=True,
        mcfarland_outputs=args.mcfarland_outputs,
    )
    if scorer.n_units != percentile.shape[1]:
        raise ValueError(
            f"population has {scorer.n_units} units but G has {percentile.shape[1]}"
        )
    scorer.model.model.eval()
    scorer.readout.eval()

    shape = (len(image_rows), len(trace_rows), 2, len(STAGE_NAMES), scorer.n_units)
    rate = np.empty(shape, dtype=np.float32)
    expected = np.empty(shape, dtype=np.float32)
    ssi = np.empty(shape, dtype=np.float32)
    temporal_modulation = np.empty(shape, dtype=np.float32)
    identity = {
        "scaffold_reconstruction_max_abs": 0.0,
        "logit_reconstruction_max_abs": 0.0,
        "ordinary_output_max_abs": 0.0,
    }
    canvas_cache: dict[Any, Any] = {}
    for image_position, image_row in enumerate(image_rows):
        patch, _ = extract_patch(
            images.iloc[int(image_row)],
            canvas_cache=canvas_cache,
            patch_size_px=int(args.patch_size_px),
        )
        zero_trace = np.zeros_like(traces[int(trace_rows[0])], dtype=np.float32)
        stable_histories = causal_histories(
            patch,
            zero_trace,
            n_lags=scorer.n_lags,
            out_size=scorer.out_size,
            temporal_factor=scorer.temporal_factor,
            supervision_phase=scorer.supervision_phase,
        )
        stable = score_histories(
            scorer,
            stable_histories,
            batch_size=int(args.batch_size),
            check_identity=(image_position == 0),
        )
        for key, value in stable[4].items():
            identity[key] = max(identity[key], float(value))
        for trace_position, trace_row in enumerate(trace_rows):
            rate[image_position, trace_position, 0] = stable[0]
            expected[image_position, trace_position, 0] = stable[1]
            ssi[image_position, trace_position, 0] = stable[2]
            temporal_modulation[image_position, trace_position, 0] = stable[3]
            histories = causal_histories(
                patch,
                np.asarray(traces[int(trace_row)], dtype=np.float32),
                n_lags=scorer.n_lags,
                out_size=scorer.out_size,
                temporal_factor=scorer.temporal_factor,
                supervision_phase=scorer.supervision_phase,
            )
            moving = score_histories(
                scorer,
                histories,
                batch_size=int(args.batch_size),
                check_identity=(image_position == 0 and trace_position == 0),
            )
            rate[image_position, trace_position, 1] = moving[0]
            expected[image_position, trace_position, 1] = moving[1]
            ssi[image_position, trace_position, 1] = moving[2]
            temporal_modulation[image_position, trace_position, 1] = moving[3]
            for key, value in moving[4].items():
                identity[key] = max(identity[key], float(value))
        print(
            f"top-passband stage trajectory image {image_position + 1}/{len(image_rows)}",
            flush=True,
        )

    # Literal cached-output audit for the full cumulative stage.
    cached_rate = spectral["mean_rate"][np.ix_(image_rows, trace_rows, (0, 1), np.arange(scorer.n_units))]
    cached_expected = spectral["expected_spikes"][np.ix_(image_rows, trace_rows, (0, 1), np.arange(scorer.n_units))]
    cached_ssi = spectral["map_ssi"][np.ix_(image_rows, trace_rows, (0, 1), np.arange(scorer.n_units))]
    cache_errors = {
        "mean_rate_spikes_s_max_abs": float(np.max(np.abs(rate[..., -1, :] - cached_rate))),
        "expected_spikes_max_abs": float(np.max(np.abs(expected[..., -1, :] - cached_expected))),
        "ssi_bits_per_spike_max_abs": float(np.max(np.abs(ssi[..., -1, :] - cached_ssi))),
    }
    if cache_errors["mean_rate_spikes_s_max_abs"] > 2.0e-3:
        raise RuntimeError(f"final rates do not reproduce G cache: {cache_errors}")
    if cache_errors["expected_spikes_max_abs"] > 2.0e-5:
        raise RuntimeError(f"final expected spikes do not reproduce G cache: {cache_errors}")
    if cache_errors["ssi_bits_per_spike_max_abs"] > 2.0e-4:
        raise RuntimeError(f"final SSI does not reproduce G cache: {cache_errors}")

    unit_rate, unit_ssi_percent = unit_top_bin_effects(
        rate, expected, ssi, membership
    )
    unit_temporal, unit_ssi = unit_top_bin_normalized_effects(
        temporal_modulation, expected, ssi, membership
    )
    archive = args.out_dir / "top_passband_stage_trajectory.npz"
    np.savez_compressed(
        archive,
        mean_rate_spikes_s=rate,
        expected_spikes=expected,
        ssi_bits_per_spike=ssi,
        temporal_modulation_depth=temporal_modulation,
        unit_rate_percent=unit_rate,
        unit_ssi_percent=unit_ssi_percent,
        unit_temporal_modulation_points=unit_temporal,
        unit_ssi_delta_bits_per_spike=unit_ssi,
        passband_percentile=percentile[trace_rows],
        top_membership=membership,
        image_rows=image_rows,
        trace_rows=trace_rows,
        stage_names=np.asarray(STAGE_NAMES),
        condition_names=np.asarray(CONDITION_NAMES),
    )
    figure = args.out_dir / "top_passband_stage_trajectory.png"
    plot_report = render_figure(
        figure,
        unit_temporal,
        unit_ssi,
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    summary = {
        "analysis": "top-passband motion effect through the trained cumulative output readout",
        "model_label": str(args.model_label),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256(args.dataset_config),
        "population_version": str(args.population_version),
        "population_n_units": int(scorer.n_units),
        "movie_selection": {
            "basis": "exact unit-specific Figure-4G passband percentile; response-blind",
            "top_percentile_threshold": float(args.top_percentile),
            "n_images": int(len(image_rows)),
            "n_traces": int(len(trace_rows)),
            "n_unique_image_trace_movies": int(len(image_rows) * len(trace_rows)),
            "image_rows": image_rows.tolist(),
            "trace_rows": trace_rows.tolist(),
            "top_memberships_per_trace": membership.sum(axis=1).astype(int).tolist(),
            "all_units_covered": bool(np.all(membership.any(axis=0))),
        },
        "readout_trajectory": {
            "stages": list(STAGE_NAMES),
            "definition": (
                "trained deep-readout logits are added by their S1, S2, and S3 "
                "feature-channel groups; the trained phase branch is assigned to S1; "
                "fixed zero-behavior additive channels and readout bias enter once; "
                "ordinary softplus is applied after each cumulative sum"
            ),
            "synthetic_reference": False,
            "affine_or_tangent_model": False,
            "final_stage_is_ordinary_model": True,
        },
        "normalization": {
            "temporal_modulation": (
                "for each unit and movie, subtract the time mean independently at "
                "each spatial location; RMS the residual across time and space; "
                "divide by the movie-wide mean response; report measured-minus-"
                "stabilized percentage points"
            ),
            "spatial_sharpening": (
                "spike-weighted spatial single-spike information; report the "
                "absolute measured-minus-stabilized difference in bits/spike"
            ),
            "both_gain_invariant": True,
            "mean_rate_gain_plotted": False,
        },
        "identity_checks": identity,
        "cached_G_output_checks": cache_errors,
        "trace_filter": filter_kind,
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_image_trace_pairs": int(len(image_rows) * len(trace_rows)),
        "n_scored_timepoints_per_pair": int(traces.shape[1]),
        "inputs": {
            "image_table": str(args.image_table.resolve()),
            "trace_array": str(args.trace_array.resolve()),
            "trace_table": str(args.trace_table.resolve()),
            "trace_provenance": str(args.trace_provenance.resolve()),
            "spectral_shards": [str(Path(value).resolve()) for value in args.spectral_shards],
        },
        "plot": plot_report,
        "archive": str(archive.resolve()),
        "figure": str(figure.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
