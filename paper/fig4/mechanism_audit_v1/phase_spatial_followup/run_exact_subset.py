#!/usr/bin/env python3
"""Exact phase-preserving activation audit on a small corrected-history subset.

The script streams exact activations through hooks and saves spatial metrics,
not the prohibitively large full activation tensors.  Exact signed maps are
saved for one predetermined example.  The full 100-image bank is never run.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    LEGACY_MATRIX_DIR,
    N_PRECEDING,
    sha256_file,
    write_json,
)
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
SELECTION = OUT / "selection"
CONTROLLED = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling"
LINEAR_DATA = ROOT / "outputs/figures/fig4/mechanism_audit_v1/targeted_frontend_v1/plot_data"
SOURCE = ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"

SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
EXAMPLE_SCALES = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
N_TRACES = 24
N_SCORED = 40
METRICS = ("total_energy", "kl_bits_uniform", "entropy_bits", "effective_area_fraction")
STAGES = (
    "retinal_input",
    "temporal_frontend",
    "stem_preactivation",
    "post_stem_splitrelu",
    "resblock1_preactivation",
    "resblock1_normalized",
    "resblock1_splitrelu",
    "resblock1_main_pooled",
    "resblock1_shortcut",
    "resblock1_output",
    "resblock2_output",
    "convgru",
    "convgru_low_readout_weighted",
    "convgru_high_readout_weighted",
    "final_rate_maps_all",
    "final_rate_maps_low",
    "final_rate_maps_high",
)


class DirectPopulationReadout(nn.Module):
    """Dependency-light copy of the exact one-hot RR100 population readout."""

    def __init__(self, feature_weights: torch.Tensor, bias: torch.Tensor, space_weights: torch.Tensor):
        super().__init__()
        self.features = nn.Conv2d(feature_weights.shape[1], feature_weights.shape[0], 1, bias=False)
        self.features.weight = nn.Parameter(feature_weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.space_weights = nn.Parameter(space_weights[:, None], requires_grad=False)
        self.n_units = int(len(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(x)
        spatial = F.conv2d(feature_map, self.space_weights, groups=self.n_units, padding="valid")
        return spatial + self.bias[None, :, None, None]


def build_direct_readout(scorer: RealTraceMatrixScorer) -> DirectPopulationReadout:
    membership = np.asarray(scorer.population_view.membership, dtype=np.float64)
    selected = np.argmax(np.abs(membership), axis=1)
    expected = np.zeros_like(membership)
    expected[np.arange(len(selected)), selected] = 1.0
    if not np.array_equal(membership, expected):
        raise ValueError("Expected exact positive one-hot RR100 membership")
    source = scorer.readout
    return DirectPopulationReadout(
        source.features.weight.detach()[selected].clone(),
        source.bias.detach()[selected].clone(),
        source.space_weights.detach()[selected, 0].clone(),
    ).to(scorer.device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_traces(table: pd.DataFrame, n_trace: int = N_TRACES) -> pd.DataFrame:
    drift = table.loc[table.rendered_n_microsaccade_events.fillna(0).eq(0)].copy()
    drift = drift.sort_values(["rendered_path_length_arcmin", "trace_bank_index"]).reset_index(drop=True)
    quantiles = np.linspace(0.02, 0.98, n_trace)
    targets = drift.rendered_path_length_arcmin.quantile(quantiles).to_numpy(float)
    chosen: list[int] = []
    for target in targets:
        distance = np.abs(drift.rendered_path_length_arcmin.to_numpy(float) - target)
        if chosen:
            distance[np.isin(drift.trace_bank_index.to_numpy(int), chosen)] = np.inf
        chosen.append(int(drift.iloc[int(np.argmin(distance))].trace_bank_index))
    out = drift.set_index("trace_bank_index").loc[chosen].reset_index()
    out.insert(1, "selection_quantile", quantiles)
    out.insert(2, "selection_target_path_arcmin", targets)
    out["selection_rule"] = "nearest unique drift-only trace to evenly spaced 0.02..0.98 rendered-path quantiles"
    return out


def scaled_histories(base: np.ndarray, scales: np.ndarray = SCALES) -> np.ndarray:
    rows = []
    for trace in np.asarray(base, dtype=np.float32):
        anchor = trace[N_PRECEDING].copy()
        displacement = trace[N_PRECEDING:] - anchor
        for scale in scales:
            history = trace.copy()
            history[N_PRECEDING:] = anchor + float(scale) * displacement
            rows.append(history)
    return np.asarray(rows, dtype=np.float32)


def spatial_metrics(
    value: torch.Tensor,
    *,
    channel_weights: torch.Tensor | None = None,
    per_channel: bool = False,
) -> torch.Tensor:
    """Time-mean energy and spatial concentration metrics for each sample.

    For a 5-D activation, every metric is first evaluated independently at
    each internal time point from the channel-summed spatial energy map and
    is only then averaged over time.  This preserves translations across the
    internal temporal axis instead of blurring them together before spatial
    normalization.
    """
    x = value.detach().float()
    if x.ndim == 5:
        has_time = True
        if per_channel:
            energy_map = x.square()  # B, C, T, H, W
        elif channel_weights is not None:
            weight = channel_weights.to(x.device, x.dtype).view(1, -1, 1, 1, 1)
            energy_map = (x.square() * weight).sum(dim=1)  # B, T, H, W
        else:
            energy_map = x.square().sum(dim=1)  # B, T, H, W
    elif x.ndim == 4:
        has_time = False
        if per_channel:
            energy_map = x.square()
        elif channel_weights is not None:
            weight = channel_weights.to(x.device, x.dtype).view(1, -1, 1, 1)
            energy_map = (x.square() * weight).sum(dim=1)
        else:
            energy_map = x.square().sum(dim=1)
    else:
        raise ValueError(tuple(x.shape))

    if per_channel:
        flat = energy_map.flatten(start_dim=-2)
    else:
        flat = energy_map.flatten(start_dim=-2)
    total = flat.sum(dim=-1)
    p = flat / total[..., None].clamp_min(1e-30)
    n = flat.shape[-1]
    p = torch.where(total[..., None] > 1e-30, p, torch.full_like(p, 1.0 / n))
    entropy = -(p * torch.log2(p.clamp_min(1e-30))).sum(dim=-1)
    kl = math.log2(n) - entropy
    effective_fraction = 1.0 / (p.square().sum(dim=-1).clamp_min(1e-30) * n)
    metrics = torch.stack((total, kl, entropy, effective_fraction), dim=-1)
    if has_time:
        time_dim = 2 if per_channel else 1
        metrics = metrics.mean(dim=time_dim)
    return metrics.cpu()


def unit_spatial_metrics(rate_map: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    value = rate_map.clamp_min(0).to(torch.float64)
    flat = value.reshape(len(value), value.shape[1], -1)
    mean_rate = flat.mean(dim=2)
    gain = flat / (mean_rate[..., None] + 1e-8)
    bits = (gain * (gain + 1e-8).log() / math.log(2.0)).mean(dim=2)
    return bits.cpu().numpy().astype(np.float32), (mean_rate / 120.0).cpu().numpy().astype(np.float32)


class MetricCollector:
    def __init__(
        self,
        model: Any,
        readout_group_weights: dict[str, torch.Tensor],
    ):
        modules = {
            "temporal_frontend": model.frontend,
            "stem_preactivation": model.convnet.stem.components["conv"],
            "post_stem_splitrelu": model.convnet.stem.components["act"],
            "resblock1_preactivation": model.convnet.layers[0].main_block.components["conv"],
            "resblock1_normalized": model.convnet.layers[0].main_block.components["norm"],
            "resblock1_splitrelu": model.convnet.layers[0].main_block.components["act"],
            "resblock1_main_pooled": model.convnet.layers[0].main_block.components["pool"],
            "resblock1_shortcut": model.convnet.layers[0].shortcut,
            "resblock1_output": model.convnet.layers[0],
            "resblock2_output": model.convnet.layers[1],
            "convgru": model.recurrent,
        }
        self.values: dict[str, list[torch.Tensor]] = {stage: [] for stage in modules}
        self.values["convgru_low_readout_weighted"] = []
        self.values["convgru_high_readout_weighted"] = []
        self.channel_values: list[torch.Tensor] = []
        self.handles = []
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name, readout_group_weights)))

    def _hook(self, name: str, readout_group_weights: dict[str, torch.Tensor]):
        def fn(module, args, output):
            self.values[name].append(spatial_metrics(output))
            if name == "resblock1_preactivation":
                self.channel_values.append(spatial_metrics(output, per_channel=True))
            if name == "convgru":
                self.values["convgru_low_readout_weighted"].append(
                    spatial_metrics(output, channel_weights=readout_group_weights["low"])
                )
                self.values["convgru_high_readout_weighted"].append(
                    spatial_metrics(output, channel_weights=readout_group_weights["high"])
                )
        return fn

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def aggregate_scored_samples(values: np.ndarray, n_condition: int) -> np.ndarray:
    return np.asarray(values).reshape(n_condition, N_SCORED, *values.shape[1:]).mean(axis=1)


def score_image(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    patch: np.ndarray,
    histories: np.ndarray,
    groups: dict[str, np.ndarray],
    group_channel_weights: dict[str, torch.Tensor],
    history_batch: int,
    frame_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = scorer.model.model
    dtype = next(model.parameters()).dtype
    collector = MetricCollector(model, group_channel_weights)
    manual: dict[str, list[torch.Tensor]] = {
        "retinal_input": [],
        "final_rate_maps_all": [],
        "final_rate_maps_low": [],
        "final_rate_maps_high": [],
    }
    bits_chunks: list[np.ndarray] = []
    expected_chunks: list[np.ndarray] = []
    image = _standardize_uint_like(patch)
    with torch.no_grad():
        for history_start in range(0, len(histories), history_batch):
            chunk = histories[history_start : history_start + history_batch]
            stims = (make_corrected_causal_stims(image, chunk, torch=scorer.torch) - 127.0) / 255.0
            for frame_start in range(0, len(stims), frame_batch):
                x = stims[frame_start : frame_start + frame_batch].to(scorer.device)
                manual["retinal_input"].append(spatial_metrics(x))
                behavior = scorer._zero_behavior(len(x), dtype)
                core = model.core_forward(x, behavior)
                rate_map = model.activation(readout(core[:, :, -1]))
                manual["final_rate_maps_all"].append(spatial_metrics(rate_map))
                manual["final_rate_maps_low"].append(spatial_metrics(rate_map[:, groups["low"]]))
                manual["final_rate_maps_high"].append(spatial_metrics(rate_map[:, groups["high"]]))
                bits, expected = unit_spatial_metrics(rate_map)
                bits_chunks.append(bits)
                expected_chunks.append(expected)
    collector.close()

    n_condition = len(histories)
    stage_values = []
    for stage in STAGES:
        chunks = manual[stage] if stage in manual else collector.values[stage]
        joined = torch.cat(chunks).numpy()
        stage_values.append(aggregate_scored_samples(joined, n_condition))
    stage_metrics = np.stack(stage_values, axis=1).astype(np.float32)
    channel_joined = torch.cat(collector.channel_values).numpy()
    channel_metrics = aggregate_scored_samples(channel_joined, n_condition).astype(np.float32)
    bits = np.concatenate(bits_chunks).reshape(n_condition, N_SCORED, 100)
    expected = np.concatenate(expected_chunks).reshape(n_condition, N_SCORED, 100)
    unit_ssi = np.sum(bits * expected, axis=1) / np.maximum(np.sum(expected, axis=1), 1e-12)
    unit_expected = np.sum(expected, axis=1)
    return unit_ssi.astype(np.float32), unit_expected.astype(np.float32), stage_metrics, channel_metrics


def select_example(
    images: pd.DataFrame,
    selected_images: pd.DataFrame,
    selected_traces: pd.DataFrame,
) -> tuple[int, int, int, pd.DataFrame]:
    candidates = images.set_index("image_index").loc[selected_images.image_index.to_numpy(int)].reset_index()
    feature = "image_high_freq_power_fraction"
    target = float(candidates[feature].quantile(0.75))
    image_id = int(candidates.iloc[int(np.argmin(np.abs(candidates[feature].to_numpy(float) - target)))].image_index)
    median_path = float(selected_traces.rendered_path_length_arcmin.median())
    trace_id = int(
        selected_traces.iloc[
            int(np.argmin(np.abs(selected_traces.rendered_path_length_arcmin.to_numpy(float) - median_path)))
        ].trace_bank_index
    )
    gain = pd.read_csv(LINEAR_DATA / "first_resnet_64_mixed_output_gain.csv.gz")
    gain = gain.loc[gain.temporal_hz.gt(0) & gain.spatial_cpd.le(16)].copy()
    gain["weighted_sf"] = gain.spatial_cpd * gain.mixed_linear_fundamental_gain
    summary = gain.groupby("first_resnet_output").agg(
        gain_sum=("mixed_linear_fundamental_gain", "sum"),
        weighted_sf_sum=("weighted_sf", "sum"),
    ).reset_index()
    summary["sf_centroid_cpd"] = summary.weighted_sf_sum / summary.gain_sum.clip(lower=1e-30)
    channel_id = int(summary.iloc[int(np.argmax(summary.sf_centroid_cpd.to_numpy(float)))].first_resnet_output)
    return image_id, trace_id, channel_id, summary


def capture_example(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    patch: np.ndarray,
    base_history: np.ndarray,
    groups: dict[str, np.ndarray],
    group_channel_weights: dict[str, torch.Tensor],
    channel_id: int,
) -> dict[str, np.ndarray]:
    model = scorer.model.model
    histories = scaled_histories(base_history[None], EXAMPLE_SCALES)
    image = _standardize_uint_like(patch)
    stims = (make_corrected_causal_stims(image, histories, torch=scorer.torch) - 127.0) / 255.0
    indices = np.arange(len(EXAMPLE_SCALES)) * N_SCORED + N_SCORED // 2
    x = stims[indices].to(scorer.device)
    capture_names = {
        "resblock1_preactivation": model.convnet.layers[0].main_block.components["conv"],
        "resblock1_normalized": model.convnet.layers[0].main_block.components["norm"],
        "resblock1_splitrelu": model.convnet.layers[0].main_block.components["act"],
        "resblock1_main_pooled": model.convnet.layers[0].main_block.components["pool"],
        "resblock1_shortcut": model.convnet.layers[0].shortcut,
        "resblock1_output": model.convnet.layers[0],
        "resblock2_output": model.convnet.layers[1],
        "convgru": model.recurrent,
    }
    captured: dict[str, torch.Tensor] = {}
    conv_input: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in capture_names.items():
        def hook(module, args, output, name=name):
            captured[name] = output.detach().clone()
            if name == "resblock1_preactivation":
                conv_input["value"] = args[0].detach().clone()
        handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        core = model.core_forward(x, scorer._zero_behavior(len(x), next(model.parameters()).dtype))
        rate_map = model.activation(readout(core[:, :, -1]))
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        replay = model.convnet.layers[0].main_block.components["conv"](conv_input["value"])
    replay_error = torch.max(torch.abs(replay - captured["resblock1_preactivation"]), dim=0).values.max().item()

    signed_all = captured["resblock1_preactivation"][:, :, -1].cpu().numpy()
    signed = signed_all[:, channel_id]
    convgru_value = captured["convgru"]
    payload: dict[str, np.ndarray] = {
        "scales": EXAMPLE_SCALES,
        "retinal_xt_slice": x[:, 0, :, x.shape[-2] // 2].detach().cpu().numpy(),
        "first_conv_signed_map": signed,
        "first_conv_squared_map": signed**2,
        "first_conv_signed_maps_all64": signed_all,
        "final_low_population_map": rate_map[:, groups["low"]].mean(dim=1).cpu().numpy(),
        "final_high_population_map": rate_map[:, groups["high"]].mean(dim=1).cpu().numpy(),
        "trajectory_xy": histories[:, N_PRECEDING:],
        "scored_output_index": np.asarray(N_SCORED // 2),
        "first_conv_internal_time_index": np.asarray(-1),
        "first_conv_channel": np.asarray(channel_id),
        "conv_replay_max_abs_error": np.asarray(replay_error),
    }
    for group_name in ("low", "high"):
        weight = group_channel_weights[group_name].to(convgru_value.device, convgru_value.dtype)
        payload[f"convgru_{group_name}_weighted_spatial_energy_map"] = (
            convgru_value.square() * weight.view(1, -1, 1, 1, 1)
        ).sum(dim=(1, 2)).cpu().numpy()
    for name, value in captured.items():
        payload[f"{name}_aggregate_metrics"] = spatial_metrics(value).numpy()
        if value.ndim == 5:
            # Preserve the exact spatial distribution of activation energy for
            # the representative conditions.  These maps remain in native
            # coordinates; the plotting code applies one common colour scale
            # across movement scales within each stage.
            payload[f"{name}_spatial_energy_map"] = (
                value.square().sum(dim=(1, 2)).detach().cpu().numpy()
            )
    return payload


def save_readout_mapping(
    readout: DirectPopulationReadout,
    unit_table: pd.DataFrame,
    groups: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    weight = readout.features.weight.detach().cpu().numpy()[:, :, 0, 0]
    rows = []
    for unit_index in range(weight.shape[0]):
        for channel_index in range(weight.shape[1]):
            rows.append(
                {
                    "unit_index": unit_index,
                    "figure4_sf_group": "low" if unit_index in set(groups["low"].tolist()) else "high",
                    "convgru_channel": channel_index,
                    "readout_feature_weight": weight[unit_index, channel_index],
                    "squared_weight": weight[unit_index, channel_index] ** 2,
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "rr100_convgru_readout_feature_weights.csv.gz", index=False)
    group_weights: dict[str, torch.Tensor] = {}
    summary_rows = []
    for name, indices in groups.items():
        value = np.mean(weight[indices] ** 2, axis=0)
        value /= max(float(value.sum()), 1e-30)
        group_weights[name] = torch.from_numpy(value.astype(np.float32))
        for channel_index, channel_weight in enumerate(value):
            summary_rows.append(
                {
                    "figure4_sf_group": name,
                    "convgru_channel": channel_index,
                    "normalized_mean_squared_readout_weight": channel_weight,
                }
            )
    pd.DataFrame(summary_rows).to_csv(DATA / "convgru_channel_group_association.csv", index=False)
    return group_weights


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    SELECTION.mkdir(parents=True, exist_ok=True)
    output = RAW / "exact_phase_spatial_metrics.npz"
    if output.is_file() and not args.overwrite:
        print(f"Using existing {output}")
        return 0
    start = time.time()

    selected_images = pd.read_csv(CONTROLLED / "selected_images.csv")
    if len(selected_images) != 8:
        raise ValueError(len(selected_images))
    trace_table = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index")
    selected_traces = select_traces(trace_table)
    selected_images.to_csv(SELECTION / "selected_images.csv", index=False)
    selected_traces.to_csv(SELECTION / "selected_traces.csv", index=False)
    image_ids = selected_images.image_index.to_numpy(int)
    trace_ids = selected_traces.trace_bank_index.to_numpy(int)

    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    unit_table = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index").reset_index(drop=True)
    sf = pd.to_numeric(unit_table.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"low": np.flatnonzero(sf < 0.5), "high": np.flatnonzero(sf >= 0.5)}
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        base_histories = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)
    histories = scaled_histories(base_histories)

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=str(args.device),
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    scorer.model.model.eval()
    readout = build_direct_readout(scorer)
    readout.eval()
    group_channel_weights = save_readout_mapping(readout, unit_table, groups)

    shape = (len(image_ids), len(trace_ids), len(SCALES))
    ssi = np.full((*shape, 100), np.nan, dtype=np.float32)
    expected = np.full_like(ssi, np.nan)
    stage_metrics = np.full((*shape, len(STAGES), len(METRICS)), np.nan, dtype=np.float32)
    channel_metrics = np.full((*shape, 64, len(METRICS)), np.nan, dtype=np.float32)
    canvas_cache: dict[Any, Any] = {}
    for image_ordinal, image_id in enumerate(image_ids):
        patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
        unit_ssi, unit_expected, stages, channels = score_image(
            scorer,
            readout,
            patch,
            histories,
            groups,
            group_channel_weights,
            int(args.history_batch_size),
            int(args.frame_batch_size),
        )
        ssi[image_ordinal] = unit_ssi.reshape(len(trace_ids), len(SCALES), 100)
        expected[image_ordinal] = unit_expected.reshape(len(trace_ids), len(SCALES), 100)
        stage_metrics[image_ordinal] = stages.reshape(len(trace_ids), len(SCALES), len(STAGES), len(METRICS))
        channel_metrics[image_ordinal] = channels.reshape(len(trace_ids), len(SCALES), 64, len(METRICS))
        print(f"exact phase/spatial image {image_ordinal + 1}/{len(image_ids)} id={image_id}", flush=True)

    example_image, example_trace, example_channel, channel_sf = select_example(images, selected_images, selected_traces)
    channel_sf.to_csv(DATA / "first_resnet_channel_sf_centroids.csv", index=False)
    example_patch, _ = extract_patch(images.iloc[example_image], canvas_cache=canvas_cache, patch_size_px=540)
    example_base = base_histories[np.flatnonzero(trace_ids == example_trace)[0]]
    example = capture_example(
        scorer, readout, example_patch, example_base, groups, group_channel_weights, example_channel
    )
    example.update(
        {
            "image_index": np.asarray(example_image),
            "trace_index": np.asarray(example_trace),
            "selection_rule": np.asarray(
                "image nearest 75th percentile high-frequency-power fraction among predetermined 8; trace nearest median path among predetermined 24; channel maximum non-DC SF centroid"
            ),
        }
    )
    np.savez_compressed(RAW / "representative_signed_maps.npz", **example)
    np.savez_compressed(
        output,
        ssi=ssi,
        expected_spikes=expected,
        stage_metrics=stage_metrics,
        first_resnet_channel_metrics=channel_metrics,
        stages=np.asarray(STAGES),
        metrics=np.asarray(METRICS),
        scales=SCALES,
        selected_image_index=image_ids,
        selected_trace_index=trace_ids,
        low_unit_indices=groups["low"],
        high_unit_indices=groups["high"],
    )
    write_json(
        OUT / "run_manifest.json",
        {
            "analysis": "exact_phase_preserving_spatial_representation_subset_v1",
            "n_images": len(image_ids),
            "n_trajectories": len(trace_ids),
            "n_scales": len(SCALES),
            "scales": SCALES,
            "selection": "8 predeclared controlled-scaling images; 24 unique drift-only trajectories nearest evenly spaced 0.02..0.98 rendered-path quantiles",
            "temporal_aggregation": "for each corrected 32-lag scored output, channel-summed spatial energy and its normalized spatial concentration metrics are computed separately at every internal model time; those metrics are arithmetic-mean averaged over internal time and then over the 40 scored outputs",
            "activation_storage": "exact activations streamed through hooks; full tensors not persisted because they are multi-terabyte at this scope; exact metrics for every condition and signed tensors for one predetermined example are persisted",
            "stages": STAGES,
            "metrics": METRICS,
            "checkpoint": MODEL_CHECKPOINT_PATH,
            "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
            "output": output,
            "output_sha256": sha256_file(output),
            "representative_signed_maps": RAW / "representative_signed_maps.npz",
            "elapsed_minutes": (time.time() - start) / 60.0,
        },
    )
    print(f"completed exact phase/spatial subset in {(time.time() - start) / 60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
