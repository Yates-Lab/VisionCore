#!/usr/bin/env python3
"""First-pass layerwise concentration and matched frontend-replacement experiment."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, N_PRECEDING, OUT_DIR, sha256_file, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import build_direct_rr100_readout, make_corrected_causal_stims
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


OUT = OUT_DIR / "first_pass_v1" / "layerwise_pilot"
SCALES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0], np.float32)
STAGES = ("retinal_input", "frontend", "stem", "resblock1", "resblock2", "convgru")


def scaled_histories(base: np.ndarray) -> np.ndarray:
    rows = []
    for trace in base:
        e0 = trace[N_PRECEDING].copy()
        displacement = trace[N_PRECEDING:] - e0
        for scale in SCALES:
            x = trace.copy()
            x[N_PRECEDING:] = e0 + float(scale) * displacement
            rows.append(x)
    return np.asarray(rows, np.float32)


def spatial_concentration(x) -> tuple[np.ndarray, np.ndarray]:
    value = x.detach().float()
    reduce_dims = tuple(range(1, value.ndim - 2))
    energy = value.square().sum(dim=reduce_dims)
    flat = energy.reshape(len(energy), -1)
    p = flat / (flat.sum(dim=1, keepdim=True) + 1e-12)
    n = flat.shape[1]
    kl = (p * (p * n + 1e-12).log() / math.log(2.0)).sum(dim=1)
    effective_fraction = 1.0 / ((p.square().sum(dim=1) + 1e-12) * n)
    return kl.cpu().numpy().astype(np.float32), effective_fraction.cpu().numpy().astype(np.float32)


class LayerCollector:
    def __init__(self, modules):
        self.values = {name: {"kl": [], "effective_fraction": []} for name in modules}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, args, output):
            kl, effective = spatial_concentration(output)
            self.values[name]["kl"].append(kl)
            self.values[name]["effective_fraction"].append(effective)
        return fn

    def close(self):
        for handle in self.handles:
            handle.remove()


def unit_spatial_metrics(rate_map, torch):
    rate_map = rate_map.clamp_min(0).to(torch.float64)
    flat = rate_map.reshape(len(rate_map), rate_map.shape[1], -1)
    rbar = flat.mean(dim=2)
    gain = flat / (rbar[..., None] + 1e-8)
    bits = (gain * (gain + 1e-8).log() / math.log(2.0)).mean(dim=2)
    return bits.cpu().numpy().astype(np.float32), (rbar / 120.0).cpu().numpy().astype(np.float32)


def score_normal(scorer, readout, patch, histories, history_batch, frame_batch):
    from scripts.spatial_info import compute_rate_map

    modules = {
        "frontend": scorer.model.model.frontend,
        "stem": scorer.model.model.convnet.stem,
        "resblock1": scorer.model.model.convnet.layers[0],
        "resblock2": scorer.model.model.convnet.layers[1],
        "convgru": scorer.model.model.recurrent,
    }
    bits_chunks, expected_chunks = [], []
    raw_kl, raw_effective = [], []
    collector = LayerCollector(modules)
    dtype = next(scorer.model.model.parameters()).dtype
    with scorer.torch.no_grad():
        for history_start in range(0, len(histories), history_batch):
            chunk = histories[history_start : history_start + history_batch]
            stims = (make_corrected_causal_stims(_standardize_uint_like(patch), chunk, torch=scorer.torch) - 127.0) / 255.0
            for start in range(0, len(stims), frame_batch):
                x = stims[start : start + frame_batch].to(scorer.device)
                kl, effective = spatial_concentration(x)
                raw_kl.append(kl); raw_effective.append(effective)
                behavior = scorer._zero_behavior(len(x), dtype)
                rate_map = compute_rate_map(scorer.model, readout, x, behavior=behavior)
                bits, expected = unit_spatial_metrics(rate_map, scorer.torch)
                bits_chunks.append(bits); expected_chunks.append(expected)
    collector.close()
    layer_kl = {"retinal_input": np.concatenate(raw_kl)}
    layer_effective = {"retinal_input": np.concatenate(raw_effective)}
    for name in modules:
        layer_kl[name] = np.concatenate(collector.values[name]["kl"])
        layer_effective[name] = np.concatenate(collector.values[name]["effective_fraction"])
    return np.concatenate(bits_chunks), np.concatenate(expected_chunks), layer_kl, layer_effective


def score_frontend_replacement(scorer, readout, patch, stable_histories, moving_histories, history_batch, frame_batch):
    """Moving input with its frontend output replaced by the matched 0x output."""
    from scripts.spatial_info import compute_rate_map

    dtype = next(scorer.model.model.parameters()).dtype
    bit_chunks, expected_chunks = [], []
    with scorer.torch.no_grad():
        for history_start in range(0, len(stable_histories), history_batch):
            stable_chunk = stable_histories[history_start : history_start + history_batch]
            moving_chunk = moving_histories[history_start : history_start + history_batch]
            stable_stims = (make_corrected_causal_stims(_standardize_uint_like(patch), stable_chunk, torch=scorer.torch) - 127.0) / 255.0
            moving_stims = (make_corrected_causal_stims(_standardize_uint_like(patch), moving_chunk, torch=scorer.torch) - 127.0) / 255.0
            for start in range(0, len(stable_stims), frame_batch):
                stop = min(start + frame_batch, len(stable_stims))
                x0 = stable_stims[start:stop].to(scorer.device)
                x1 = moving_stims[start:stop].to(scorer.device)
                captured = {}
                capture_handle = scorer.model.model.frontend.register_forward_hook(
                    lambda module, args, output: captured.setdefault("value", output.detach().clone())
                )
                behavior = scorer._zero_behavior(len(x0), dtype)
                _ = compute_rate_map(scorer.model, readout, x0, behavior=behavior)
                capture_handle.remove()
                stable_frontend = captured["value"]

                def replace(module, args, output):
                    if output.shape != stable_frontend.shape:
                        raise ValueError((output.shape, stable_frontend.shape))
                    return stable_frontend

                replacement_handle = scorer.model.model.frontend.register_forward_hook(replace)
                replaced_map = compute_rate_map(scorer.model, readout, x1, behavior=behavior)
                replacement_handle.remove()
                bits, expected = unit_spatial_metrics(replaced_map, scorer.torch)
                bit_chunks.append(bits); expected_chunks.append(expected)
    return np.concatenate(bit_chunks), np.concatenate(expected_chunks)


def aggregate_time(bits: np.ndarray, expected: np.ndarray, n_conditions: int):
    bits = bits.reshape(n_conditions, 40, 100)
    expected = expected.reshape(n_conditions, 40, 100)
    return (
        np.sum(bits * expected, axis=1) / np.maximum(np.sum(expected, axis=1), 1e-8),
        np.sum(expected, axis=1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "layerwise_response.npz"
    if output.is_file():
        print(f"Using existing {output}")
        return 0
    selected_images = pd.read_csv(
        ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling/selected_images.csv"
    )
    selected_traces = pd.read_csv(
        ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling/selected_traces.csv"
    )
    image_ids = selected_images.image_index.to_numpy(int)
    trace_ids = selected_traces.trace_bank_index.to_numpy(int)
    images = pd.read_csv(
        ROOT / "outputs/active_sensing_movie_information/backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged/image_feature_table.csv"
    ).sort_values("image_index").reset_index(drop=True)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        base = np.asarray(archive["true_history_xy"][trace_ids], np.float32)
    histories = scaled_histories(base)
    n_conditions = len(histories)
    shape = (len(image_ids), len(trace_ids), len(SCALES), 100)
    ssi = np.full(shape, np.nan, np.float32)
    expected = np.full_like(ssi, np.nan)
    replacement_ssi = np.full((len(image_ids), len(trace_ids), 100), np.nan, np.float32)
    replacement_expected = np.full_like(replacement_ssi, np.nan)
    layer_kl = np.full((len(image_ids), len(trace_ids), len(SCALES), len(STAGES)), np.nan, np.float32)
    layer_effective = np.full_like(layer_kl, np.nan)
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH, dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR, rr100_version=RR100_VERSION,
        device=str(args.device), strict=True, mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    readout = build_direct_rr100_readout(scorer)
    canvas_cache = {}
    scale0 = histories.reshape(len(trace_ids), len(SCALES), 71, 2)[:, 0]
    scale1 = histories.reshape(len(trace_ids), len(SCALES), 71, 2)[:, 2]
    for image_ordinal, image_id in enumerate(image_ids):
        patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
        bits, exp, kl, effective = score_normal(
            scorer, readout, patch, histories,
            history_batch=int(args.history_batch_size), frame_batch=int(args.frame_batch_size)
        )
        unit_ssi, unit_expected = aggregate_time(bits, exp, n_conditions)
        ssi[image_ordinal] = unit_ssi.reshape(len(trace_ids), len(SCALES), 100)
        expected[image_ordinal] = unit_expected.reshape(len(trace_ids), len(SCALES), 100)
        for stage_index, stage in enumerate(STAGES):
            layer_kl[image_ordinal, :, :, stage_index] = kl[stage].reshape(n_conditions, 40).mean(axis=1).reshape(len(trace_ids), len(SCALES))
            layer_effective[image_ordinal, :, :, stage_index] = effective[stage].reshape(n_conditions, 40).mean(axis=1).reshape(len(trace_ids), len(SCALES))
        rb, re = score_frontend_replacement(
            scorer, readout, patch, scale0, scale1,
            history_batch=int(args.history_batch_size), frame_batch=int(args.frame_batch_size)
        )
        replacement_ssi[image_ordinal], replacement_expected[image_ordinal] = aggregate_time(rb, re, len(trace_ids))
        print(f"layerwise/intervention image {image_ordinal + 1}/{len(image_ids)} id={image_id}", flush=True)
    np.savez_compressed(
        output, ssi=ssi, expected_spikes=expected,
        frontend_replacement_ssi=replacement_ssi,
        frontend_replacement_expected_spikes=replacement_expected,
        layer_kl_bits_from_uniform=layer_kl,
        layer_effective_area_fraction=layer_effective,
        stages=np.asarray(STAGES), scales=SCALES,
        selected_image_index=image_ids, selected_trace_index=trace_ids,
    )
    write_json(
        OUT / "manifest.json",
        {
            "analysis": "layerwise_spatial_concentration_and_matched_frontend_replacement_pilot",
            "status": "pilot", "n_images": len(image_ids), "n_trajectories": len(trace_ids),
            "scales": SCALES, "stages": STAGES,
            "concentration": "E(x,y)=sum over feature/time dimensions of activation^2; p=E/sum(E); KL(p||uniform) and effective area fraction",
            "intervention": "moving 1x input; replace learned frontend output at every scored output with matched same-image/same-trace 0x frontend activation; downstream weights unchanged",
            "checkpoint": MODEL_CHECKPOINT_PATH, "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
            "output": output, "output_sha256": sha256_file(output),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
