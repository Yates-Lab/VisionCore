#!/usr/bin/env python3
"""Exact rate-map decomposition and motion-matched null controls for Figure 4.

This analysis uses the authoritative historical Figure 4 populations (71 low
SF, 29 high SF).  Channel masks are selected without natural-image SSI from
corrected-FEM × ConvGRU-F1 overlap and frozen readout weights.  It saves exact
rate-map information metrics, linear pre-softplus pathway alignment, causal
necessity/sufficiency curves, and association-matched permutation-null masks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, LEGACY_MATRIX_DIR, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    DirectPopulationReadout,
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/ssi_mechanism_v2"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
LAYERWISE = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
PHASE_OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
SOURCE = ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
TARGET_SCALE = {"low": 3.0, "high": 1.0}
OPPOSITE_SCALE = {"low": 1.0, "high": 3.0}
N_SCORED = 40
MASK_SIZE = 16
N_NULL_MASKS = 63
NULL_ASSOCIATION_TOLERANCE = 0.10
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--null-mask-batch-size", type=int, default=7)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def load_model(device: str) -> tuple[RealTraceMatrixScorer, DirectPopulationReadout]:
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=device,
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    scorer.model.model.eval()
    return scorer, build_direct_readout(scorer).eval()


def historical_groups() -> dict[str, np.ndarray]:
    table = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    sf = pd.to_numeric(table.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"low": np.flatnonzero(sf < 0.5), "high": np.flatnonzero(sf >= 0.5)}
    if len(groups["low"]) != 71 or len(groups["high"]) != 29:
        raise ValueError({key: len(value) for key, value in groups.items()})
    return groups


def channel_selection(
    groups: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    weights = pd.read_csv(PHASE_OUT / "plot_data/rr100_convgru_readout_feature_weights.csv.gz")
    overlap = pd.read_csv(LAYERWISE / "plot_data/layerwise_corrected_fem_tuning_overlap.csv.gz")
    overlap = overlap.loc[
        overlap.stage.eq("convgru") & overlap.response_metric.eq("f1_amplitude")
    ].pivot(index="channel", columns="scale", values="normalized_overlap")
    selected: dict[str, np.ndarray] = {}
    readout_only: dict[str, np.ndarray] = {}
    null_masks: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260812)
    for group, units in groups.items():
        association = (
            weights.loc[weights.unit_index.isin(units)]
            .groupby("convgru_channel")
            .squared_weight.mean()
            .reindex(np.arange(128), fill_value=0.0)
            .to_numpy(float)
        )
        target_overlap = overlap[TARGET_SCALE[group]].reindex(np.arange(128)).to_numpy(float)
        opposite_overlap = overlap[OPPOSITE_SCALE[group]].reindex(np.arange(128)).to_numpy(float)
        contrast = np.clip(target_overlap - opposite_overlap, 0, None)
        score = association * contrast
        selected[group] = np.argsort(score)[-MASK_SIZE:][::-1].astype(int)
        readout_only[group] = np.argsort(association)[-MASK_SIZE:][::-1].astype(int)
        observed_association = float(association[selected[group]].sum())

        accepted: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = {tuple(sorted(selected[group].tolist()))}
        attempts = 0
        while len(accepted) < N_NULL_MASKS and attempts < 1_000_000:
            attempts += 1
            permuted = rng.permutation(contrast)
            ids = np.argsort(association * permuted)[-MASK_SIZE:].astype(int)
            key = tuple(sorted(ids.tolist()))
            association_ratio = float(association[ids].sum() / max(observed_association, EPS))
            if key in seen or abs(association_ratio - 1.0) > NULL_ASSOCIATION_TOLERANCE:
                continue
            seen.add(key)
            accepted.append(np.asarray(key, dtype=int))
        if len(accepted) != N_NULL_MASKS:
            raise RuntimeError((group, len(accepted), attempts))
        null_masks[group] = np.stack(accepted)
        selected_set = set(selected[group].tolist())
        readout_set = set(readout_only[group].tolist())
        for channel in range(128):
            rows.append(
                {
                    "figure4_sf_group": group,
                    "convgru_channel": channel,
                    "target_scale": TARGET_SCALE[group],
                    "opposite_scale": OPPOSITE_SCALE[group],
                    "mean_squared_readout_weight": association[channel],
                    "fem_f1_overlap_target": target_overlap[channel],
                    "fem_f1_overlap_opposite": opposite_overlap[channel],
                    "positive_overlap_contrast": contrast[channel],
                    "selection_score": score[channel],
                    "selected_motion_matched": channel in selected_set,
                    "selected_readout_only": channel in readout_set,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "figure4_convgru_channel_selection.csv", index=False)
    np.savez_compressed(
        RAW / "association_matched_permutation_null_masks.npz",
        low_masks=null_masks["low"],
        high_masks=null_masks["high"],
        low_selected=selected["low"],
        high_selected=selected["high"],
        low_readout_only=readout_only["low"],
        high_readout_only=readout_only["high"],
        mask_size=np.asarray(MASK_SIZE),
        association_tolerance=np.asarray(NULL_ASSOCIATION_TOLERANCE),
        selection_rule=np.asarray(
            "permute the corrected-FEM overlap contrast over ConvGRU channels, rerank association*permuted_contrast, and retain unique 16-channel masks whose total true readout association is within 10% of the observed motion-matched mask"
        ),
    )
    return selected, readout_only, null_masks, table


def rate_components(rate_map: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = rate_map.clamp_min(0).double().flatten(start_dim=2)
    mean_rate = flat.mean(dim=2)
    gain = flat / (mean_rate[..., None] + 1e-8)
    bits = (gain * torch.log2(gain + 1e-8)).mean(dim=2)
    cv2 = ((gain - 1.0) ** 2).mean(dim=2)
    quadratic_bits = cv2 / (2.0 * math.log(2.0))
    return {
        "mean_rate": mean_rate,
        "gain": gain,
        "ssi": bits,
        "cv2": cv2,
        "quadratic_bits": quadratic_bits,
    }


def append_group_metrics(
    rows: list[dict[str, object]],
    *,
    image_id: int,
    trace_id: int,
    condition: str,
    scale: float,
    components: dict[str, torch.Tensor],
    groups: dict[str, np.ndarray],
) -> None:
    expected = components["mean_rate"] / 120.0
    for group, units in groups.items():
        weight = expected[:, units]
        denominator = float(weight.sum().detach().cpu())
        row: dict[str, object] = {
            "image_index": image_id,
            "trace_index": trace_id,
            "condition": condition,
            "scale": scale,
            "figure4_sf_group": group,
            "expected_spikes": denominator,
            "mean_rate_sum": float(components["mean_rate"][:, units].sum().detach().cpu()),
            "n_rate_values": int(components["mean_rate"][:, units].numel()),
        }
        for metric in ("ssi", "cv2", "quadratic_bits"):
            row[f"{metric}_weighted_numerator"] = float(
                (components[metric][:, units] * weight).sum().detach().cpu()
            )
        rows.append(row)


def selected_preactivation(
    readout: DirectPopulationReadout, recurrent: torch.Tensor, channels: np.ndarray
) -> torch.Tensor:
    latest = recurrent[:, :, -1]
    selected = torch.zeros_like(latest)
    selected[:, channels] = latest[:, channels]
    feature_map = readout.features(selected)
    return F.conv2d(feature_map, readout.space_weights, groups=readout.n_units, padding="valid")


def full_preactivation(readout: DirectPopulationReadout, recurrent: torch.Tensor) -> torch.Tensor:
    return readout(recurrent[:, :, -1])


def spatial_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = a.double().flatten(start_dim=2)
    bb = b.double().flatten(start_dim=2)
    aa = aa - aa.mean(dim=2, keepdim=True)
    bb = bb - bb.mean(dim=2, keepdim=True)
    return (aa * bb).sum(dim=2) / (
        torch.linalg.vector_norm(aa, dim=2) * torch.linalg.vector_norm(bb, dim=2)
    ).clamp_min(1e-12)


def hybrid(anchor: torch.Tensor, donor: torch.Tensor, channels: np.ndarray) -> torch.Tensor:
    value = anchor.clone()
    value[:, channels] = donor[:, channels]
    return value


def run_pair(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    stims: torch.Tensor,
    *,
    image_id: int,
    trace_id: int,
    groups: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
    readout_only: dict[str, np.ndarray],
    null_masks: dict[str, np.ndarray],
    frame_batch: int,
    null_batch: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model = scorer.model.model
    dtype = next(model.parameters()).dtype
    movies = stims.reshape(len(SCALES), N_SCORED, *stims.shape[1:])
    metric_chunks: dict[tuple[str, float], dict[str, list[torch.Tensor]]] = {}
    alignment_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        group: {"selected_total_cosine": [], "selected_gain_cosine": [], "selected_projection": []}
        for group in groups
    }
    unit_target: dict[tuple[str, str], dict[str, list[torch.Tensor]]] = {}
    null_accumulator: dict[tuple[str, int], list[float]] = {
        (group, null_id): [0.0, 0.0] for group in groups for null_id in range(N_NULL_MASKS)
    }

    def save_condition(condition: str, scale: float, rate: torch.Tensor) -> None:
        comp = rate_components(rate)
        bucket = metric_chunks.setdefault((condition, scale), {key: [] for key in comp if key != "gain"})
        for key in bucket:
            bucket[key].append(comp[key].detach().cpu())

    with torch.no_grad():
        for start in range(0, N_SCORED, frame_batch):
            stop = min(start + frame_batch, N_SCORED)
            batch = stop - start
            all_x = movies[:, start:stop].reshape(len(SCALES) * batch, *movies.shape[2:]).to(scorer.device)
            recurrent_flat = model.core_forward(all_x, scorer._zero_behavior(len(all_x), dtype))
            recurrent = recurrent_flat.reshape(len(SCALES), batch, *recurrent_flat.shape[1:])
            stable = recurrent[0]
            stable_rate = model.activation(full_preactivation(readout, stable))
            for scale in SCALES:
                save_condition("normal_stable", float(scale), stable_rate)
            save_condition("normal_moving", 0.0, stable_rate)
            for target in groups:
                save_condition(f"{target}_motion_matched_necessary", 0.0, stable_rate)
                save_condition(f"{target}_motion_matched_sufficient", 0.0, stable_rate)
                save_condition(f"{target}_readout_only_necessary", 0.0, stable_rate)

            for scale_i in range(1, len(SCALES)):
                scale = float(SCALES[scale_i])
                moving = recurrent[scale_i]
                moving_rate = model.activation(full_preactivation(readout, moving))
                save_condition("normal_moving", scale, moving_rate)
                for target in groups:
                    necessary = hybrid(moving, stable, selected[target])
                    sufficient = hybrid(stable, moving, selected[target])
                    control = hybrid(moving, stable, readout_only[target])
                    save_condition(
                        f"{target}_motion_matched_necessary", scale,
                        model.activation(full_preactivation(readout, necessary)),
                    )
                    save_condition(
                        f"{target}_motion_matched_sufficient", scale,
                        model.activation(full_preactivation(readout, sufficient)),
                    )
                    save_condition(
                        f"{target}_readout_only_necessary", scale,
                        model.activation(full_preactivation(readout, control)),
                    )

                for group in groups:
                    if not np.isclose(scale, TARGET_SCALE[group]):
                        continue
                    z_stable = full_preactivation(readout, stable)
                    z_moving = full_preactivation(readout, moving)
                    dz_total = z_moving - z_stable
                    dz_selected = selected_preactivation(readout, moving, selected[group]) - selected_preactivation(
                        readout, stable, selected[group]
                    )
                    gain_stable = rate_components(stable_rate)["gain"].reshape(batch, 100, *stable_rate.shape[-2:])
                    gain_moving = rate_components(moving_rate)["gain"].reshape(batch, 100, *moving_rate.shape[-2:])
                    alignment_chunks[group]["selected_total_cosine"].append(spatial_cosine(dz_selected, dz_total).cpu())
                    alignment_chunks[group]["selected_gain_cosine"].append(
                        spatial_cosine(dz_selected, gain_moving - gain_stable).cpu()
                    )
                    selected_flat = dz_selected.double().flatten(start_dim=2)
                    total_flat = dz_total.double().flatten(start_dim=2)
                    selected_flat -= selected_flat.mean(dim=2, keepdim=True)
                    total_flat -= total_flat.mean(dim=2, keepdim=True)
                    projection = (selected_flat * total_flat).sum(dim=2) / total_flat.square().sum(dim=2).clamp_min(EPS)
                    alignment_chunks[group]["selected_projection"].append(projection.cpu())

                    necessary_rate = model.activation(
                        full_preactivation(readout, hybrid(moving, stable, selected[group]))
                    )
                    for name, rate in (
                        ("stable", stable_rate),
                        ("moving", moving_rate),
                        ("necessary", necessary_rate),
                    ):
                        comp = rate_components(rate)
                        bucket = unit_target.setdefault((group, name), {"ssi": [], "expected": []})
                        bucket["ssi"].append(comp["ssi"].cpu())
                        bucket["expected"].append((comp["mean_rate"] / 120.0).cpu())

                    masks = null_masks[group]
                    for null_start in range(0, N_NULL_MASKS, null_batch):
                        null_stop = min(null_start + null_batch, N_NULL_MASKS)
                        hybrids = [hybrid(moving, stable, mask) for mask in masks[null_start:null_stop]]
                        joined = torch.cat(hybrids, dim=0)
                        rate = model.activation(full_preactivation(readout, joined))
                        comp = rate_components(rate)
                        for local, null_id in enumerate(range(null_start, null_stop)):
                            sl = slice(local * batch, (local + 1) * batch)
                            units = groups[group]
                            expected = comp["mean_rate"][sl][:, units] / 120.0
                            null_accumulator[(group, null_id)][0] += float(
                                (comp["ssi"][sl][:, units] * expected).sum().cpu()
                            )
                            null_accumulator[(group, null_id)][1] += float(expected.sum().cpu())

    metric_rows: list[dict[str, object]] = []
    for (condition, scale), chunks in metric_chunks.items():
        comp = {key: torch.cat(value) for key, value in chunks.items()}
        append_group_metrics(
            metric_rows,
            image_id=image_id,
            trace_id=trace_id,
            condition=condition,
            scale=scale,
            components=comp,
            groups=groups,
        )

    alignment_rows: list[dict[str, object]] = []
    for group, metrics in alignment_chunks.items():
        units = groups[group]
        for metric, chunks in metrics.items():
            value = torch.cat(chunks)[:, units].numpy()
            alignment_rows.append(
                {
                    "image_index": image_id,
                    "trace_index": trace_id,
                    "figure4_sf_group": group,
                    "target_scale": TARGET_SCALE[group],
                    "metric": metric,
                    "mean": float(np.nanmean(value)),
                    "median": float(np.nanmedian(value)),
                    "n_frame_units": int(np.isfinite(value).sum()),
                }
            )

    unit_rows: list[dict[str, object]] = []
    for group in groups:
        units = groups[group]
        joined = {
            name: {key: torch.cat(value) for key, value in unit_target[(group, name)].items()}
            for name in ("stable", "moving", "necessary")
        }
        for unit in units:
            row: dict[str, object] = {
                "image_index": image_id,
                "trace_index": trace_id,
                "figure4_sf_group": group,
                "target_scale": TARGET_SCALE[group],
                "unit_index": int(unit),
            }
            for name in ("stable", "moving", "necessary"):
                expected = joined[name]["expected"][:, unit]
                row[f"{name}_ssi"] = float(
                    (joined[name]["ssi"][:, unit] * expected).sum() / expected.sum().clamp_min(EPS)
                )
                row[f"{name}_expected_spikes"] = float(expected.sum())
            unit_rows.append(row)

    null_rows = [
        {
            "image_index": image_id,
            "trace_index": trace_id,
            "figure4_sf_group": group,
            "target_scale": TARGET_SCALE[group],
            "null_mask_index": null_id,
            "ssi_weighted_numerator": values[0],
            "expected_spikes": values[1],
        }
        for (group, null_id), values in null_accumulator.items()
    ]
    return metric_rows, alignment_rows, unit_rows, null_rows


def choose_examples(unit_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, frame in unit_table.groupby("figure4_sf_group"):
        value = frame.copy()
        value["movement_delta"] = value.moving_ssi - value.stable_ssi
        value["necessary_delta"] = value.necessary_ssi - value.moving_ssi
        unit_summary = value.groupby("unit_index").agg(
            movement_delta=("movement_delta", "median"),
            necessary_delta=("necessary_delta", "median"),
        )
        center = unit_summary.median()
        unit_scale = unit_summary.mad() if hasattr(unit_summary, "mad") else (unit_summary - center).abs().median()
        unit_scale = unit_scale.replace(0, 1.0)
        distance = (((unit_summary - center) / unit_scale) ** 2).sum(axis=1)
        unit = int(distance.idxmin())
        pair = value.loc[value.unit_index.eq(unit)].copy()
        pair_center = pair[["movement_delta", "necessary_delta"]].median()
        pair_scale = (pair[["movement_delta", "necessary_delta"]] - pair_center).abs().median().replace(0, 1.0)
        pair_distance = (((pair[["movement_delta", "necessary_delta"]] - pair_center) / pair_scale) ** 2).sum(axis=1)
        chosen = pair.iloc[int(np.argmin(pair_distance.to_numpy(float)))]
        rows.append(
            {
                "figure4_sf_group": group,
                "unit_index": unit,
                "image_index": int(chosen.image_index),
                "trace_index": int(chosen.trace_index),
                "target_scale": float(chosen.target_scale),
                "selection_rule": "unit nearest bivariate median of unit-median movement and necessity effects; image-trajectory pair nearest bivariate median for that unit",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "representative_map_selection.csv", index=False)
    return result


def capture_examples(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    selection: pd.DataFrame,
    selected_channels: dict[str, np.ndarray],
    images: pd.DataFrame,
    base_histories_by_trace: dict[int, np.ndarray],
) -> None:
    model = scorer.model.model
    dtype = next(model.parameters()).dtype
    payload: dict[str, np.ndarray] = {}
    canvas_cache: dict[Any, Any] = {}
    for _, row in selection.iterrows():
        group = str(row.figure4_sf_group)
        unit = int(row.unit_index)
        image_id = int(row.image_index)
        trace_id = int(row.trace_index)
        scale = float(row.target_scale)
        patch, _ = extract_patch(images.iloc[image_id], canvas_cache=canvas_cache, patch_size_px=540)
        histories = scaled_histories(base_histories_by_trace[trace_id][None], np.asarray([0.0, scale], dtype=np.float32))
        stims = (make_corrected_causal_stims(_standardize_uint_like(patch), histories, torch=torch) - 127.0) / 255.0
        movies = stims.reshape(2, N_SCORED, *stims.shape[1:])
        with torch.no_grad():
            recurrent = []
            for scale_i in range(2):
                chunks = []
                for start in range(0, N_SCORED, 8):
                    x = movies[scale_i, start : start + 8].to(scorer.device)
                    chunks.append(
                        model.core_forward(
                            x, scorer._zero_behavior(len(x), dtype)
                        )
                    )
                recurrent.append(torch.cat(chunks))
            stable, moving = recurrent
            necessary = hybrid(moving, stable, selected_channels[group])
            z_stable = full_preactivation(readout, stable)
            z_moving = full_preactivation(readout, moving)
            z_necessary = full_preactivation(readout, necessary)
            rates = [model.activation(value) for value in (z_stable, z_moving, z_necessary)]
            dz_selected = selected_preactivation(readout, moving, selected_channels[group]) - selected_preactivation(
                readout, stable, selected_channels[group]
            )
            components = [rate_components(rate) for rate in rates]
            frame_delta = components[1]["ssi"][:, unit] - components[0]["ssi"][:, unit]
            expected = components[1]["mean_rate"][:, unit] / 120.0
            pair_delta = float((frame_delta * expected).sum() / expected.sum().clamp_min(EPS))
            frame = int(torch.argmin(torch.abs(frame_delta - pair_delta)).cpu())
            prefix = f"{group}_"
            payload[prefix + "unit_index"] = np.asarray(unit)
            payload[prefix + "image_index"] = np.asarray(image_id)
            payload[prefix + "trace_index"] = np.asarray(trace_id)
            payload[prefix + "target_scale"] = np.asarray(scale)
            payload[prefix + "frame_index"] = np.asarray(frame)
            for name, rate, comp in zip(("stable", "moving", "necessary"), rates, components):
                rate_map = rate[frame, unit].detach().cpu().numpy()
                payload[prefix + name + "_rate_map"] = rate_map.astype(np.float32)
                payload[prefix + name + "_gain_map"] = (rate_map / max(float(rate_map.mean()), EPS)).astype(np.float32)
                payload[prefix + name + "_ssi"] = np.asarray(float(comp["ssi"][frame, unit].cpu()))
                payload[prefix + name + "_cv2"] = np.asarray(float(comp["cv2"][frame, unit].cpu()))
            payload[prefix + "selected_delta_preactivation_map"] = dz_selected[frame, unit].cpu().numpy().astype(np.float32)
            payload[prefix + "total_delta_preactivation_map"] = (z_moving - z_stable)[frame, unit].cpu().numpy().astype(np.float32)
    np.savez_compressed(RAW / "representative_rate_map_decomposition.npz", **payload)


def main() -> int:
    args = parse_args()
    start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    groups = historical_groups()
    selected, readout_only, null_masks, selection_table = channel_selection(groups)
    with np.load(PHASE_OUT / "exact_arrays/exact_phase_spatial_metrics.npz") as archive:
        image_ids = np.asarray(archive["selected_image_index"], dtype=int)
        trace_ids = np.asarray(archive["selected_trace_index"], dtype=int)
    if int(args.max_images) > 0:
        image_ids = image_ids[: int(args.max_images)]
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        base_histories = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)
    base_by_trace = {int(trace): history for trace, history in zip(trace_ids, base_histories)}
    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    scorer, readout = load_model(str(args.device))
    metric_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    canvas_cache: dict[Any, Any] = {}
    for image_ordinal, image_id in enumerate(image_ids):
        patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
        image = _standardize_uint_like(patch)
        for trace_ordinal, (trace_id, base_history) in enumerate(zip(trace_ids, base_histories)):
            histories = scaled_histories(base_history[None], SCALES)
            stims = (make_corrected_causal_stims(image, histories, torch=torch) - 127.0) / 255.0
            result = run_pair(
                scorer,
                readout,
                stims,
                image_id=int(image_id),
                trace_id=int(trace_id),
                groups=groups,
                selected=selected,
                readout_only=readout_only,
                null_masks=null_masks,
                frame_batch=int(args.frame_batch_size),
                null_batch=int(args.null_mask_batch_size),
            )
            metric_rows.extend(result[0]); alignment_rows.extend(result[1]); unit_rows.extend(result[2]); null_rows.extend(result[3])
            print(
                f"exact SSI decomposition image {image_ordinal + 1}/{len(image_ids)} "
                f"trace {trace_ordinal + 1}/{len(trace_ids)}",
                flush=True,
            )
    metrics = pd.DataFrame(metric_rows)
    alignment = pd.DataFrame(alignment_rows)
    units = pd.DataFrame(unit_rows)
    null = pd.DataFrame(null_rows)
    metrics.to_csv(DATA / "exact_rate_map_metric_components.csv.gz", index=False)
    alignment.to_csv(DATA / "selected_pathway_spatial_alignment.csv", index=False)
    units.to_csv(DATA / "target_scale_unit_effects.csv.gz", index=False)
    null.to_csv(DATA / "association_matched_null_trace_components.csv.gz", index=False)
    examples = choose_examples(units)
    capture_examples(scorer, readout, examples, selected, images, base_by_trace)
    write_json(
        OUT / "run_manifest.json",
        {
            "analysis": "exact_figure4_population_ssi_rate_map_decomposition_v2",
            "groups": {key: value for key, value in groups.items()},
            "group_counts": {key: len(value) for key, value in groups.items()},
            "scales": SCALES,
            "target_scale": TARGET_SCALE,
            "selected_channels": selected,
            "readout_only_channels": readout_only,
            "n_association_matched_permutation_null_masks": N_NULL_MASKS,
            "null_association_tolerance": NULL_ASSOCIATION_TOLERANCE,
            "selection_used_natural_image_ssi": False,
            "selection_rule": "mean squared readout weight to authoritative Figure 4 population times positive corrected-FEM/ConvGRU-F1 overlap contrast at target versus opposite-population scale",
            "n_images": len(image_ids),
            "n_corrected_drift_trajectories": len(trace_ids),
            "n_scored_frames": N_SCORED,
            "checkpoint": MODEL_CHECKPOINT_PATH,
            "rr100_version": RR100_VERSION,
            "elapsed_minutes": (time.time() - start) / 60.0,
        },
    )
    print(json.dumps({"selected": {key: value.tolist() for key, value in selected.items()}, "elapsed_minutes": (time.time() - start) / 60.0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
