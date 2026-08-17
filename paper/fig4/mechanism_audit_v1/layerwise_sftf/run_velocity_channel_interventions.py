#!/usr/bin/env python3
"""Causal ConvGRU channel tests selected without using natural-image SSI.

Channels are selected from three frozen quantities only: layerwise grating
F1 speed tuning, RR100-quartile readout weights, and (for the evaluation-scale
declaration) the corrected-FEM finite-trajectory phase spectrum.  The held-out outcome is
the exact natural-image SSI response to corrected eye-movement histories.
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


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, write_json
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


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
PHASE_OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
SOURCE = ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
N_SCORED = 40
MASK_SIZE = 16
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
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


def channel_selection(
    evaluation_scale: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    fits = pd.read_csv(DATA / "layerwise_channel_sftf_fits.csv.gz")
    quartiles = pd.read_csv(DATA / "rr100_f0_sf_quartiles.csv")
    weights = pd.read_csv(
        PHASE_OUT / "plot_data/rr100_convgru_readout_feature_weights.csv.gz"
    )
    conv = fits.loc[
        fits.stage.eq("convgru")
        & fits.response_metric.eq("f1_amplitude")
        & fits.valid_tuned.fillna(False)
        & fits.sf_censoring.eq("none")
        & fits.tf_censoring.eq("none")
    ].copy()
    overlap = pd.read_csv(DATA / "layerwise_corrected_fem_tuning_overlap.csv.gz")
    conv_overlap = overlap.loc[
        overlap.stage.eq("convgru") & overlap.response_metric.eq("f1_amplitude")
    ].pivot(index="channel", columns="scale", values="normalized_overlap")
    n_channels = int(weights.convgru_channel.max()) + 1
    rows: list[dict[str, object]] = []
    selected: dict[str, np.ndarray] = {}
    association_only: dict[str, np.ndarray] = {}
    target_speed: dict[str, float] = {}
    for quartile in ("Q1", "Q4"):
        units = quartiles.loc[quartiles.sf_quartile.eq(quartile), "channel"].to_numpy(int)
        target_speed[quartile] = float(
            quartiles.loc[quartiles.sf_quartile.eq(quartile), "preferred_speed_dps"].median()
        )
        association = (
            weights.loc[weights.unit_index.isin(units)]
            .groupby("convgru_channel")
            .squared_weight.mean()
            .reindex(np.arange(n_channels), fill_value=0.0)
            .to_numpy(float)
        )
        association /= max(float(association.sum()), EPS)
        speed = np.full(n_channels, np.nan)
        speed[conv.channel.to_numpy(int)] = conv.preferred_speed_dps.to_numpy(float)
        opposite = "Q4" if quartile == "Q1" else "Q1"
        target_overlap = conv_overlap[float(evaluation_scale[quartile])].reindex(
            np.arange(n_channels)
        ).to_numpy(float)
        opposite_overlap = conv_overlap[float(evaluation_scale[opposite])].reindex(
            np.arange(n_channels)
        ).to_numpy(float)
        speed_match = np.clip(target_overlap - opposite_overlap, 0, None)
        score = association * speed_match
        selected[quartile] = np.argsort(score)[-MASK_SIZE:][::-1].astype(int)
        association_only[quartile] = np.argsort(association)[-MASK_SIZE:][::-1].astype(int)
        selected_set = set(selected[quartile].tolist())
        association_set = set(association_only[quartile].tolist())
        for channel in range(n_channels):
            rows.append(
                {
                    "target_sf_quartile": quartile,
                    "convgru_channel": channel,
                    "target_speed_dps": target_speed[quartile],
                    "target_corrected_fem_scale": evaluation_scale[quartile],
                    "opposite_corrected_fem_scale": evaluation_scale[opposite],
                    "convgru_f1_preferred_speed_dps": speed[channel],
                    "mean_squared_readout_weight": association[channel],
                    "corrected_fem_f1_overlap_at_target_scale": target_overlap[channel],
                    "corrected_fem_f1_overlap_at_opposite_scale": opposite_overlap[channel],
                    "positive_target_vs_opposite_overlap_contrast": speed_match[channel],
                    "selection_score": score[channel],
                    "selected_velocity_matched": channel in selected_set,
                    "selected_readout_association_only_control": channel in association_set,
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "convgru_velocity_channel_selection.csv", index=False)
    return selected, association_only, target_speed


def declared_evaluation_scales() -> dict[str, float]:
    overlap = pd.read_csv(DATA / "layerwise_corrected_fem_tuning_overlap.csv.gz")
    quartiles = pd.read_csv(DATA / "rr100_f0_sf_quartiles.csv")
    rr = overlap.loc[
        overlap.stage.eq("rr100") & overlap.response_metric.eq("f0_signed_mean") & overlap.scale.gt(0)
    ].merge(quartiles[["channel", "sf_quartile"]], on="channel", validate="many_to_one")
    mean = rr.groupby(["sf_quartile", "scale"], as_index=False).normalized_overlap.mean()
    return {
        quartile: float(frame.loc[frame.normalized_overlap.idxmax(), "scale"])
        for quartile, frame in mean.loc[mean.sf_quartile.isin(["Q1", "Q4"])].groupby("sf_quartile")
    }


def output_rate(model: Any, readout: DirectPopulationReadout, recurrent: torch.Tensor) -> torch.Tensor:
    return model.activation(readout(recurrent[:, :, -1]))


def group_components(rate_map: torch.Tensor, groups: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    value = rate_map.clamp_min(0).double().flatten(start_dim=2)
    mean_rate = value.mean(dim=2)
    gain = value / (mean_rate[..., None] + 1e-8)
    bits = (gain * torch.log2(gain + 1e-8)).mean(dim=2)
    expected = mean_rate / 120.0
    result: dict[str, tuple[float, float]] = {}
    for name, indices in groups.items():
        result[name] = (
            float((bits[:, indices] * expected[:, indices]).sum().detach().cpu()),
            float(expected[:, indices].sum().detach().cpu()),
        )
    return result


def add_components(
    accumulator: dict[tuple[str, float, str], list[float]],
    condition: str,
    scale: float,
    components: dict[str, tuple[float, float]],
) -> None:
    for group, (numerator, denominator) in components.items():
        value = accumulator.setdefault((condition, float(scale), group), [0.0, 0.0])
        value[0] += numerator
        value[1] += denominator


def hybrid_recurrent(
    anchor: torch.Tensor, donor: torch.Tensor, channels: np.ndarray
) -> torch.Tensor:
    value = anchor.clone()
    value[:, channels] = donor[:, channels]
    return value


def score_trace(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    stims: torch.Tensor,
    groups: dict[str, np.ndarray],
    selected: dict[str, np.ndarray],
    association_only: dict[str, np.ndarray],
    frame_batch: int,
) -> dict[tuple[str, float, str], list[float]]:
    model = scorer.model.model
    dtype = next(model.parameters()).dtype
    movies = stims.reshape(len(SCALES), N_SCORED, *stims.shape[1:])
    accumulator: dict[tuple[str, float, str], list[float]] = {}
    with torch.no_grad():
        for start in range(0, N_SCORED, frame_batch):
            stop = min(start + frame_batch, N_SCORED)
            batch_per_scale = stop - start
            # The five amplitudes share one batched core call.  This is exactly
            # equivalent to five independent calls because ConvGRU state is
            # confined to each sample's 32-lag causal window.
            all_x = movies[:, start:stop].reshape(
                len(SCALES) * batch_per_scale, *movies.shape[2:]
            ).to(scorer.device)
            flat_recurrent = model.core_forward(
                all_x, scorer._zero_behavior(len(all_x), dtype)
            )
            # Avoid encoding architecture-specific dimensions in the reshape.
            recurrent = flat_recurrent.reshape(
                len(SCALES), batch_per_scale, *flat_recurrent.shape[1:]
            )
            stable = recurrent[0]
            stable_components = group_components(output_rate(model, readout, stable), groups)
            for scale in SCALES:
                add_components(accumulator, "normal_stable", float(scale), stable_components)
            for scale_i in range(1, len(SCALES)):
                scale = float(SCALES[scale_i])
                moving = recurrent[scale_i]
                conditions: list[tuple[str, torch.Tensor]] = [("normal_moving", moving)]
                for target in ("Q1", "Q4"):
                    necessary = hybrid_recurrent(moving, stable, selected[target])
                    sufficient = hybrid_recurrent(stable, moving, selected[target])
                    control = hybrid_recurrent(moving, stable, association_only[target])
                    conditions.extend(
                        [
                            (f"{target}_velocity_matched_necessary", necessary),
                            (f"{target}_velocity_matched_sufficient", sufficient),
                            (f"{target}_readout_weight_only_necessary_control", control),
                        ]
                    )
                joined_rate = output_rate(
                    model, readout, torch.cat([value for _, value in conditions], dim=0)
                )
                for (condition, _), rate in zip(
                    conditions, joined_rate.split(batch_per_scale, dim=0)
                ):
                    add_components(
                        accumulator,
                        condition,
                        scale,
                        group_components(rate, groups),
                    )
    # Moving at 0x is identical to stabilized and is useful for complete curves.
    for group in groups:
        stable_value = accumulator[("normal_stable", 0.0, group)]
        accumulator[("normal_moving", 0.0, group)] = stable_value.copy()
        for target in ("Q1", "Q4"):
            for suffix in (
                "velocity_matched_necessary",
                "velocity_matched_sufficient",
                "readout_weight_only_necessary_control",
            ):
                accumulator[(f"{target}_{suffix}", 0.0, group)] = stable_value.copy()
    return accumulator


def main() -> int:
    args = parse_args()
    start_time = time.time()
    evaluation_scale = declared_evaluation_scales()
    selected, association_only, target_speed = channel_selection(evaluation_scale)
    quartiles = pd.read_csv(DATA / "rr100_f0_sf_quartiles.csv")
    groups = {
        quartile: frame.channel.to_numpy(int)
        for quartile, frame in quartiles.groupby("sf_quartile")
    }
    with np.load(PHASE_OUT / "exact_arrays/exact_phase_spatial_metrics.npz") as archive:
        image_ids = np.asarray(archive["selected_image_index"], dtype=int)
        trace_ids = np.asarray(archive["selected_trace_index"], dtype=int)
    if int(args.max_images) > 0:
        image_ids = image_ids[: int(args.max_images)]
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        base_histories = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)
    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    scorer, readout = load_model(str(args.device))
    rows: list[dict[str, object]] = []
    canvas_cache: dict[Any, Any] = {}
    for image_ordinal, image_id in enumerate(image_ids):
        patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
        image = _standardize_uint_like(patch)
        for trace_ordinal, (trace_id, base_history) in enumerate(zip(trace_ids, base_histories)):
            histories = scaled_histories(base_history[None], SCALES)
            stims = (make_corrected_causal_stims(image, histories, torch=torch) - 127.0) / 255.0
            values = score_trace(
                scorer,
                readout,
                stims,
                groups,
                selected,
                association_only,
                int(args.frame_batch_size),
            )
            for (condition, scale, group), (numerator, denominator) in values.items():
                rows.append(
                    {
                        "image_index": int(image_id),
                        "trace_index": int(trace_id),
                        "scale": float(scale),
                        "condition": condition,
                        "sf_quartile": group,
                        "ssi_numerator": numerator,
                        "expected_spikes": denominator,
                        "ssi": numerator / max(denominator, EPS),
                    }
                )
            print(
                f"velocity-channel intervention image {image_ordinal + 1}/{len(image_ids)} "
                f"trace {trace_ordinal + 1}/{len(trace_ids)}",
                flush=True,
            )
    trace_table = pd.DataFrame(rows)
    trace_table.to_csv(DATA / "convgru_velocity_channel_intervention_trace_results.csv.gz", index=False)
    aggregate = (
        trace_table.groupby(["condition", "sf_quartile", "scale"], as_index=False)
        .agg(ssi_numerator=("ssi_numerator", "sum"), expected_spikes=("expected_spikes", "sum"))
    )
    aggregate["ssi"] = aggregate.ssi_numerator / aggregate.expected_spikes.clip(lower=EPS)
    aggregate.to_csv(DATA / "convgru_velocity_channel_intervention_aggregate.csv", index=False)
    write_json(
        OUT / "velocity_channel_intervention_manifest.json",
        {
            "selection_independence": "No natural-image SSI value was used for channel selection. The Q1/Q4 target scale is the maximum of the corrected-FEM finite-trajectory phase spectrum times measured RR100 F0 tuning. ConvGRU channels are ranked by mean squared frozen readout weight to that quartile times the positive contrast between their grating-F1/corrected-FEM overlap at the target versus opposite-quartile scale.",
            "selected_channels": selected,
            "readout_weight_only_control_channels": association_only,
            "mask_size": MASK_SIZE,
            "target_speed_dps": target_speed,
            "declared_evaluation_scale_from_corrected_fem_rr100_overlap": evaluation_scale,
            "n_images": len(image_ids),
            "n_corrected_drift_trajectories": len(trace_ids),
            "scales": SCALES,
            "necessity_definition": "moving ConvGRU activation with selected channels replaced by their paired 0x activation",
            "sufficiency_definition": "0x ConvGRU activation with selected channels replaced by their paired moving activation",
            "control_definition": "same-size top readout-association mask selected without velocity matching; necessity direction only",
            "elapsed_minutes": (time.time() - start_time) / 60.0,
        },
    )
    print(json.dumps({"selected": {k: v.tolist() for k, v in selected.items()}, "evaluation_scale": evaluation_scale}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
