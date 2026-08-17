#!/usr/bin/env python3
"""Calibrate a reduced twin's response nonlinearity on natural/FEM contexts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import (
    build_direct_readout,
    scaled_histories,
)
from paper.fig4.nonlinear_phase_causal.common import N_SCORED
from paper.fig4.nonlinear_phase_causal.reduced_twin import ReducedTwin, ReducedTwinState
from paper.fig4.nonlinear_phase_causal.run_experiment import SOURCE, _selection, preactivation
from paper.fig4.nonlinear_phase_causal.run_reduced_twin_pilot import (
    architecture_name,
    fit_decoder,
    parse_architectures,
    sampled_metrics,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


DEFAULT_STATE = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/reduced_twin_full_frequency/"
    "reduced_twin_rank128_mlp_128x64.pt"
)
DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/reduced_twin_natural_calibration"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--locations-per-frame", type=int, default=32)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--calibration-scales", default="0,1")
    parser.add_argument("--architectures", default="64,32;128,64;256,128")
    parser.add_argument("--seeds", default="17,29")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--overwrite-bank", action="store_true")
    return parser.parse_args()


def split_spec(selected_images, selected_traces):
    image_ids = selected_images.image_index.to_numpy(int)
    trace_ids = selected_traces.trace_bank_index.to_numpy(int)
    return {
        "train": (image_ids[:4], trace_ids[:12]),
        "val": (image_ids[4:5], trace_ids[12:18]),
        "test": (image_ids[5:8], trace_ids[18:24]),
    }


@torch.no_grad()
def gather_locations(value, ids):
    flat = value.permute(0, 2, 3, 1).reshape(len(value), -1, value.shape[1])
    batch = torch.arange(len(value), device=value.device)[:, None]
    return flat[batch, ids]


@torch.no_grad()
def build_bank(args, reduced, scorer, readout, selection, selected_images, selected_traces):
    bank = args.output_dir / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    split_rows = split_spec(selected_images, selected_traces)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        all_histories = np.asarray(archive["true_history_xy"], dtype=np.float32)
    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    canvas_cache = {}
    scales = np.asarray(
        [float(value) for value in args.calibration_scales.split(",") if value.strip()],
        dtype=np.float32,
    )
    for split_number, (split, (image_ids, trace_ids)) in enumerate(split_rows.items()):
        n_rows = len(image_ids) * len(trace_ids) * len(scales) * N_SCORED * args.locations_per_frame
        generator_path = bank / f"{split}_generators.npy"
        target_path = bank / f"{split}_rates.npy"
        if (
            generator_path.exists()
            and target_path.exists()
            and not args.overwrite_bank
            and np.load(generator_path, mmap_mode="r").shape == (n_rows, reduced.rank)
        ):
            print(f"natural calibration {split}: reusing bank", flush=True)
            continue
        generators_out = np.lib.format.open_memmap(
            generator_path, mode="w+", dtype=np.float16, shape=(n_rows, reduced.rank)
        )
        rates_out = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.float16, shape=(n_rows, len(selection))
        )
        cursor = 0
        for image_ordinal, image_id in enumerate(image_ids):
            patch, _ = extract_patch(images.iloc[int(image_id)], canvas_cache=canvas_cache, patch_size_px=540)
            image = _standardize_uint_like(patch)
            for trace_ordinal, trace_id in enumerate(trace_ids):
                histories = scaled_histories(all_histories[int(trace_id) : int(trace_id) + 1], scales)
                stims = (make_corrected_causal_stims(image, histories, torch=scorer.torch) - 127.0) / 255.0
                for start in range(0, len(stims), args.frame_batch_size):
                    x = stims[start : start + args.frame_batch_size].to(scorer.device)
                    behavior = scorer._zero_behavior(len(x), x.dtype)
                    exact_rate = scorer.model.model.activation(
                        preactivation(scorer.model.model, readout, x, behavior)[:, selection]
                    )
                    generator_maps = reduced.movie_generators(x)
                    rng = np.random.default_rng(
                        20260813 + split_number * 10_000_000 + int(image_id) * 100_003
                        + int(trace_id) * 101 + start
                    )
                    location_ids = torch.from_numpy(
                        np.stack(
                            [
                                rng.choice(51 * 51, args.locations_per_frame, replace=False)
                                for _ in range(len(x))
                            ]
                        )
                    ).to(x.device)
                    generators = gather_locations(generator_maps, location_ids)
                    targets = gather_locations(exact_rate, location_ids)
                    count = generators.shape[0] * generators.shape[1]
                    generators_out[cursor : cursor + count] = generators.reshape(count, -1).cpu().numpy().astype(np.float16)
                    rates_out[cursor : cursor + count] = targets.reshape(count, -1).cpu().numpy().astype(np.float16)
                    cursor += count
                print(
                    f"natural calibration {split}: image {image_ordinal + 1}/{len(image_ids)} "
                    f"trace {trace_ordinal + 1}/{len(trace_ids)}",
                    flush=True,
                )
        if cursor != n_rows:
            raise RuntimeError((cursor, n_rows))
        generators_out.flush(); rates_out.flush()
    return split_rows, scales


def load_bank(output_dir, device):
    result = {}
    for split in ("train", "val", "test"):
        x = torch.from_numpy(np.asarray(np.load(output_dir / "bank" / f"{split}_generators.npy"))).to(device).float()
        y = torch.from_numpy(np.asarray(np.load(output_dir / "bank" / f"{split}_rates.npy"))).to(device).float()
        result[split] = (x, y)
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_state = torch.load(args.state, map_location="cpu", weights_only=False)
    reduced = ReducedTwin(source_state).to(args.device).eval()
    selection_table = pd.read_csv(
        ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot/unit_selection.csv"
    )
    selection = selection_table.unit_index.to_numpy(int)
    selected_images, selected_traces = _selection()
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=args.device,
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    scorer.model.model.eval()
    readout = build_direct_readout(scorer).eval()
    splits, calibration_scales = build_bank(
        args, reduced, scorer, readout, selection, selected_images, selected_traces
    )
    del scorer, readout
    data = load_bank(args.output_dir, args.device)
    architectures = parse_architectures(args.architectures)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for hidden in architectures:
        candidates = []
        for seed in seeds:
            fitted = fit_decoder(
                data,
                reduced.rank,
                hidden,
                dropout=0.1,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
            )
            candidates.append((fitted[4], seed, fitted))
        _, seed, fitted = min(candidates, key=lambda item: item[0])
        decoder, generator_mean, generator_std, _, val_loss, best_epoch, history = fitted
        state = ReducedTwinState(
            rank=reduced.rank,
            unit_indices=source_state.unit_indices,
            basis=source_state.basis,
            feature_mean=source_state.feature_mean,
            feature_std=source_state.feature_std,
            generator_mean=generator_mean.detach().cpu(),
            generator_std=generator_std.detach().cpu(),
            hidden_dims=hidden,
            dropout=0.1,
            decoder_state={key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        )
        name = architecture_name(hidden)
        metric = sampled_metrics(
            decoder, data, generator_mean, generator_std, reduced.rank, selection_table
        )
        metric.to_csv(args.output_dir / f"test_response_fidelity_{name}.csv", index=False)
        history.to_csv(args.output_dir / f"training_{name}.csv", index=False)
        torch.save(state, args.output_dir / f"natural_calibrated_{name}.pt")
        rows.append(
            {
                "architecture": name,
                "seed": seed,
                "best_epoch": best_epoch,
                "val_normalized_mse": val_loss,
                "median_test_rate_r2": metric.sampled_rate_r2.median(),
                "min_test_rate_r2": metric.sampled_rate_r2.min(),
                "median_test_rate_correlation": metric.sampled_rate_correlation.median(),
            }
        )
        print(name, rows[-1], flush=True)
    summary = pd.DataFrame(rows).sort_values(
        ["median_test_rate_r2", "val_normalized_mse"], ascending=[False, True]
    )
    summary.to_csv(args.output_dir / "model_selection.csv", index=False)
    best = summary.iloc[0]
    payload = {
        "source_state": str(args.state.resolve()),
        "split": {
            key: {"image_ids": value[0].tolist(), "trace_ids": value[1].tolist()}
            for key, value in splits.items()
        },
        "calibration_scales": calibration_scales.tolist(),
        "selected_model": best.to_dict(),
        "selected_state": str((args.output_dir / f"natural_calibrated_{best.architecture}.pt").resolve()),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
