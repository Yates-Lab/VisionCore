#!/usr/bin/env python3
"""Targeted gradient mapping and exact activation-swap controls for Figure 4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules.conv_blocks import _minimal_crop_like
from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, LEGACY_MATRIX_DIR, N_PRECEDING, write_json
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


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
SOURCE = ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
SCALES = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
N_SCORED = 40
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
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
    readout = build_direct_readout(scorer).eval()
    return scorer, readout


def output_rate(model: Any, readout: DirectPopulationReadout, recurrent: torch.Tensor) -> torch.Tensor:
    return model.activation(readout(recurrent[:, :, -1]))


def group_frame_components(rate_map: torch.Tensor, groups: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    value = rate_map.clamp_min(0).double()
    flat = value.flatten(start_dim=2)
    mean_rate = flat.mean(dim=2)
    gain = flat / (mean_rate[..., None] + 1e-8)
    bits = (gain * torch.log2(gain + 1e-8)).mean(dim=2)
    expected = mean_rate / 120.0
    out = {}
    for name, indices in groups.items():
        out[name] = (
            (bits[:, indices] * expected[:, indices]).sum(dim=1).detach().cpu().numpy(),
            expected[:, indices].sum(dim=1).detach().cpu().numpy(),
        )
    return out


def prefix(model: Any, x: torch.Tensor) -> dict[str, torch.Tensor]:
    frontend = model.frontend(x)
    stem = model.convnet.stem(frontend)
    rb1 = model.convnet.layers[0]
    rb1_main = rb1.main_block(stem)
    rb1_shortcut = _minimal_crop_like(rb1.shortcut(stem), rb1_main, causal_time=rb1.causal_time)
    rb1_output = rb1.post_add_activation(rb1_main + rb1_shortcut)
    rb2 = model.convnet.layers[1]
    rb2_main = rb2.main_block(rb1_output)
    rb2_shortcut = _minimal_crop_like(rb2.shortcut(rb1_output), rb2_main, causal_time=rb2.causal_time)
    rb2_output = rb2.post_add_activation(rb2_main + rb2_shortcut)
    convgru = model.recurrent(rb2_output)
    return {
        "rb1_main": rb1_main,
        "rb1_shortcut": rb1_shortcut,
        "rb1_output": rb1_output,
        "rb2_main": rb2_main,
        "rb2_shortcut": rb2_shortcut,
        "rb2_output": rb2_output,
        "convgru": convgru,
    }


def from_rb1(model: Any, value: torch.Tensor) -> torch.Tensor:
    return model.recurrent(model.convnet.layers[1](value))


def from_rb2(model: Any, value: torch.Tensor) -> torch.Tensor:
    return model.recurrent(value)


def collect_condition(
    storage: dict[str, dict[str, list[np.ndarray]]], condition: str, rate: torch.Tensor, groups: dict[str, np.ndarray]
) -> None:
    components = group_frame_components(rate, groups)
    for group, (num, den) in components.items():
        storage.setdefault(condition, {}).setdefault(f"{group}_num", []).append(num)
        storage.setdefault(condition, {}).setdefault(f"{group}_den", []).append(den)


def run_swaps(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    movies: torch.Tensor,
    groups: dict[str, np.ndarray],
    frame_batch: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = scorer.model.model
    movies = movies.reshape(len(SCALES), N_SCORED, *movies.shape[1:])
    results: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for scale_index, scale in enumerate(SCALES):
        storage: dict[str, dict[str, list[np.ndarray]]] = {}
        max_errors = {f"{stage}_{direction}": 0.0 for stage in ("rb1", "rb2", "convgru") for direction in ("stable_into_moving", "moving_into_stable")}
        for start in range(0, N_SCORED, frame_batch):
            stable_x = movies[0, start : start + frame_batch].to(scorer.device)
            moving_x = movies[scale_index, start : start + frame_batch].to(scorer.device)
            with torch.no_grad():
                stable = prefix(model, stable_x)
                moving = prefix(model, moving_x)
                stable_rate = output_rate(model, readout, stable["convgru"])
                moving_rate = output_rate(model, readout, moving["convgru"])
                collect_condition(storage, "normal_stable", stable_rate, groups)
                collect_condition(storage, "normal_moving", moving_rate, groups)

                rb1_stable_in_moving = output_rate(model, readout, from_rb1(model, stable["rb1_output"]))
                rb1_moving_in_stable = output_rate(model, readout, from_rb1(model, moving["rb1_output"]))
                rb2_stable_in_moving = output_rate(model, readout, from_rb2(model, stable["rb2_output"]))
                rb2_moving_in_stable = output_rate(model, readout, from_rb2(model, moving["rb2_output"]))
                gru_stable_in_moving = output_rate(model, readout, stable["convgru"])
                gru_moving_in_stable = output_rate(model, readout, moving["convgru"])
                for stage, a, b in (
                    ("rb1", rb1_stable_in_moving, rb1_moving_in_stable),
                    ("rb2", rb2_stable_in_moving, rb2_moving_in_stable),
                    ("convgru", gru_stable_in_moving, gru_moving_in_stable),
                ):
                    max_errors[f"{stage}_stable_into_moving"] = max(max_errors[f"{stage}_stable_into_moving"], float((a - stable_rate).abs().max()))
                    max_errors[f"{stage}_moving_into_stable"] = max(max_errors[f"{stage}_moving_into_stable"], float((b - moving_rate).abs().max()))

                rb1_move_main = model.convnet.layers[0].post_add_activation(moving["rb1_main"] + stable["rb1_shortcut"])
                rb1_move_short = model.convnet.layers[0].post_add_activation(stable["rb1_main"] + moving["rb1_shortcut"])
                collect_condition(storage, "rb1_moving_main_stable_shortcut", output_rate(model, readout, from_rb1(model, rb1_move_main)), groups)
                collect_condition(storage, "rb1_stable_main_moving_shortcut", output_rate(model, readout, from_rb1(model, rb1_move_short)), groups)

                rb2_move_main = model.convnet.layers[1].post_add_activation(moving["rb2_main"] + stable["rb2_shortcut"])
                rb2_move_short = model.convnet.layers[1].post_add_activation(stable["rb2_main"] + moving["rb2_shortcut"])
                collect_condition(storage, "rb2_moving_main_stable_shortcut", output_rate(model, readout, from_rb2(model, rb2_move_main)), groups)
                collect_condition(storage, "rb2_stable_main_moving_shortcut", output_rate(model, readout, from_rb2(model, rb2_move_short)), groups)
        for condition, values in storage.items():
            for group in groups:
                num = np.concatenate(values[f"{group}_num"]).sum()
                den = np.concatenate(values[f"{group}_den"]).sum()
                results.append({"scale": float(scale), "condition": condition, "sf_group": group, "ssi": float(num / max(float(den), EPS))})
        for key, error in max_errors.items():
            stage, direction = key.split("_", 1)
            validation_rows.append({"scale": float(scale), "stage": stage, "swap_direction": direction, "max_abs_rate_map_error_vs_exact_anchor": error})
    table = pd.DataFrame(results)
    baseline = table.loc[table.condition.eq("normal_stable"), ["scale", "sf_group", "ssi"]].rename(columns={"ssi": "stable_ssi"})
    table = table.merge(baseline, on=["scale", "sf_group"], validate="many_to_one")
    table["ssi_percent_vs_stable"] = 100 * (table.ssi - table.stable_ssi) / table.stable_ssi.abs().clip(lower=EPS)
    return table, pd.DataFrame(validation_rows)


def gradient_mapping(
    scorer: RealTraceMatrixScorer,
    readout: DirectPopulationReadout,
    movies: torch.Tensor,
    groups: dict[str, np.ndarray],
) -> pd.DataFrame:
    model = scorer.model.model
    midpoint = N_SCORED // 2
    x = movies.reshape(len(SCALES), N_SCORED, *movies.shape[1:])[:, midpoint].to(scorer.device)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    modules = {
        "first_conv_preactivation": model.convnet.layers[0].main_block.components["conv"],
        "resblock1_output": model.convnet.layers[0],
        "resblock2_output": model.convnet.layers[1],
        "convgru": model.recurrent,
    }
    for name, module in modules.items():
        handles.append(module.register_forward_hook(lambda module, args, output, name=name: captured.setdefault(name, output)))
    core = model.core_forward(x, scorer._zero_behavior(len(x), next(model.parameters()).dtype))
    rate = model.activation(readout(core[:, :, -1]))
    for handle in handles:
        handle.remove()
    value = rate.double().flatten(start_dim=2)
    mean_rate = value.mean(dim=2)
    gain = value / (mean_rate[..., None] + 1e-8)
    bits = (gain * torch.log2(gain + 1e-8)).mean(dim=2)
    expected = mean_rate / 120.0
    rows = []
    targets = list(captured.values())
    for group_index, (group, indices) in enumerate(groups.items()):
        group_ssi = (bits[:, indices] * expected[:, indices]).sum(dim=1) / expected[:, indices].sum(dim=1).clamp_min(EPS)
        gradients = torch.autograd.grad(group_ssi.sum(), targets, retain_graph=group_index == 0)
        for (stage, activation), gradient in zip(captured.items(), gradients):
            importance = (activation * gradient).abs()
            if importance.ndim == 5:
                importance = importance.mean(dim=(2, 3, 4))
            else:
                importance = importance.mean(dim=(2, 3))
            for scale_index, scale in enumerate(SCALES):
                normalized = importance[scale_index] / importance[scale_index].sum().clamp_min(EPS)
                for channel, score in enumerate(normalized.detach().cpu().numpy()):
                    rows.append({"sf_group": group, "scale": float(scale), "stage": stage, "channel": channel, "normalized_abs_gradient_times_activation": float(score)})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    with np.load(RAW / "representative_signed_maps.npz") as archive:
        image_index = int(archive["image_index"])
        trace_index = int(archive["trace_index"])
    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    patch, _ = extract_patch(images.iloc[image_index], canvas_cache={}, patch_size_px=540)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        history = np.asarray(archive["true_history_xy"][trace_index], dtype=np.float32)
    histories = scaled_histories(history[None], SCALES)
    stims = (make_corrected_causal_stims(_standardize_uint_like(patch), histories, torch=torch) - 127.0) / 255.0
    unit = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    sf = pd.to_numeric(unit.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"low": np.flatnonzero(sf < .5), "high": np.flatnonzero(sf >= .5)}
    scorer, readout = load_model(str(args.device))
    swaps, validation = run_swaps(scorer, readout, stims, groups, int(args.frame_batch_size))
    gradients = gradient_mapping(scorer, readout, stims, groups)
    swaps.to_csv(DATA / "representative_activation_branch_swaps.csv", index=False)
    validation.to_csv(DATA / "full_activation_swap_validation.csv", index=False)
    gradients.to_csv(DATA / "representative_gradient_channel_mapping.csv.gz", index=False)
    write_json(
        OUT / "mapping_intervention_manifest.json",
        {
            "image_index": image_index,
            "trace_index": trace_index,
            "scales": SCALES,
            "gradient_mapping": "absolute gradient-times-activation of low/high final population SSI at the objectively selected midpoint frame, normalized over channels per stage and scale",
            "full_swap_interpretation": "full-stage replacements are exact shape-compatible tests but algebraically reproduce the downstream output belonging to the injected complete activation; their numerical role is validation, not localization",
            "branch_hybrids": "moving main branch plus stabilized shortcut, and stabilized main branch plus moving shortcut, separately in ResBlocks 1 and 2",
            "max_full_swap_anchor_error": float(validation.max_abs_rate_map_error_vs_exact_anchor.max()),
        },
    )
    print(f"maximum full-swap anchor error: {validation.max_abs_rate_map_error_vs_exact_anchor.max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
