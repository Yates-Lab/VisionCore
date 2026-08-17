#!/usr/bin/env python3
"""Test whether a response-distilled twin transfers to natural FEM movies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from paper.fig4.nonlinear_phase_causal.common import N_SCORED, unit_rate_metrics
from paper.fig4.nonlinear_phase_causal.reduced_twin import ReducedTwin
from paper.fig4.nonlinear_phase_causal.run_experiment import (
    PHASE_SOURCE,
    SOURCE,
    _selection,
    preactivation,
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
DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/reduced_twin_fem_transfer"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-traces", type=int, default=6)
    parser.add_argument("--selection-split", choices=("pilot", "test", "all"), default="test")
    parser.add_argument("--scales", default="0,0.5,1,2,3")
    parser.add_argument("--frame-batch-size", type=int, default=4)
    return parser.parse_args()


def empty_moments(n_units):
    return {
        key: np.zeros(n_units, dtype=np.float64)
        for key in ("sum_y", "sum_p", "sum_y2", "sum_p2", "sum_yp", "sse")
    } | {"n": 0}


def update_moments(moment, target, prediction):
    y = target.detach().double().flatten(start_dim=2)
    p = prediction.detach().double().flatten(start_dim=2)
    moment["sum_y"] += y.sum((0, 2)).cpu().numpy()
    moment["sum_p"] += p.sum((0, 2)).cpu().numpy()
    moment["sum_y2"] += y.square().sum((0, 2)).cpu().numpy()
    moment["sum_p2"] += p.square().sum((0, 2)).cpu().numpy()
    moment["sum_yp"] += (y * p).sum((0, 2)).cpu().numpy()
    moment["sse"] += (y - p).square().sum((0, 2)).cpu().numpy()
    moment["n"] += y.shape[0] * y.shape[2]


def finish_moments(moment):
    n = max(int(moment["n"]), 1)
    target_ss = moment["sum_y2"] - moment["sum_y"] ** 2 / n
    pred_ss = moment["sum_p2"] - moment["sum_p"] ** 2 / n
    covariance = moment["sum_yp"] - moment["sum_y"] * moment["sum_p"] / n
    return (
        1.0 - moment["sse"] / np.maximum(target_ss, 1e-12),
        covariance / np.sqrt(np.maximum(target_ss * pred_ss, 1e-24)),
    )


def empty_metric_accumulator(n_units):
    return {
        "ssi_weighted": np.zeros(n_units, dtype=np.float64),
        "expected": np.zeros(n_units, dtype=np.float64),
        "mean_rate_sum": np.zeros(n_units, dtype=np.float64),
        "n_frames": 0,
    }


def update_metrics(accumulator, rate):
    metric = unit_rate_metrics(rate)
    expected = metric["expected_spikes"].detach().cpu().numpy()
    accumulator["ssi_weighted"] += (metric["ssi"].detach().cpu().numpy() * expected).sum(0)
    accumulator["expected"] += expected.sum(0)
    accumulator["mean_rate_sum"] += metric["mean_rate_hz"].detach().cpu().numpy().sum(0)
    accumulator["n_frames"] += len(rate)


def score_pair(scorer, readout, reduced, stable, moving, selection, batch_size):
    result = {}
    for condition, stims in (("stable", stable), ("moving", moving)):
        moment = empty_moments(len(selection))
        exact_metric = empty_metric_accumulator(len(selection))
        reduced_metric = empty_metric_accumulator(len(selection))
        for start in range(0, N_SCORED, batch_size):
            x = stims[start : start + batch_size].to(scorer.device)
            behavior = scorer._zero_behavior(len(x), x.dtype)
            with torch.no_grad():
                exact = scorer.model.model.activation(
                    preactivation(scorer.model.model, readout, x, behavior)[:, selection]
                )
                predicted = reduced(x)
            update_moments(moment, exact, predicted)
            update_metrics(exact_metric, exact)
            update_metrics(reduced_metric, predicted)
        r2, correlation = finish_moments(moment)
        result[condition] = {
            "r2": r2,
            "correlation": correlation,
            "exact_ssi": exact_metric["ssi_weighted"] / np.maximum(exact_metric["expected"], 1e-12),
            "reduced_ssi": reduced_metric["ssi_weighted"] / np.maximum(reduced_metric["expected"], 1e-12),
            "exact_rate": exact_metric["mean_rate_sum"] / exact_metric["n_frames"],
            "reduced_rate": reduced_metric["mean_rate_sum"] / reduced_metric["n_frames"],
        }
    return result


def render(output_dir, rows):
    moving = rows.loc[rows.condition.eq("moving")].copy()
    curves = moving.groupby(["response_subspace_group", "scale"])[
        ["exact_ssi_delta", "reduced_ssi_delta"]
    ].mean().reset_index()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    unit = moving.groupby("unit_index").map_rate_r2.median().sort_values()
    axes[0].bar(np.arange(len(unit)), unit, color="#3178a8")
    axes[0].axhline(0.8, color="0.4", ls="--", lw=1)
    axes[0].set(xlabel="RR100 unit", ylabel="natural/FEM map response $R^2$", xticks=np.arange(len(unit)), xticklabels=[f"u{x:03d}" for x in unit.index], ylim=(-1, 1))
    axes[0].tick_params(axis="x", rotation=55)
    axes[1].scatter(moving.exact_ssi_delta, moving.reduced_ssi_delta, s=8, alpha=0.35)
    limit = np.nanpercentile(np.abs(np.r_[moving.exact_ssi_delta, moving.reduced_ssi_delta]), 99)
    axes[1].plot([-limit, limit], [-limit, limit], color="0.4", ls="--", lw=1)
    axes[1].set(xlabel=r"exact twin $\Delta$SSI", ylabel=r"reduced twin $\Delta$SSI", xlim=(-limit, limit), ylim=(-limit, limit))
    colors = {"lower-SF": "#2676b8", "higher-SF": "#d85832"}
    for group, part in curves.groupby("response_subspace_group"):
        axes[2].plot(part.scale, part.exact_ssi_delta, "o-", color=colors[group], label=f"{group} exact")
        axes[2].plot(part.scale, part.reduced_ssi_delta, "o--", color=colors[group], label=f"{group} reduced")
    axes[2].axhline(0, color="0.5", lw=1)
    axes[2].set(xlabel="FEM amplitude (x measured)", ylabel=r"mean $\Delta$SSI vs stabilization")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "natural_fem_transfer.png", dpi=220)
    fig.savefig(output_dir / "natural_fem_transfer.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    reduced = ReducedTwin(state).to(args.device).eval()
    selection_table = pd.read_csv(
        ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot/unit_selection.csv"
    )
    selection = selection_table.unit_index.to_numpy(int)
    if not np.array_equal(selection, state.unit_indices.numpy()):
        raise ValueError("State and predeclared unit selection differ")
    selected_images, selected_traces = _selection()
    if args.selection_split == "test":
        selected_images = selected_images.iloc[5:8]
        selected_traces = selected_traces.iloc[18:24]
    elif args.selection_split == "pilot":
        selected_images = selected_images.iloc[:2]
        selected_traces = selected_traces.iloc[:6]
    selected_images = selected_images.iloc[: args.max_images]
    selected_traces = selected_traces.iloc[: args.max_traces]
    trace_ids = selected_traces.trace_bank_index.to_numpy(int)
    scales = np.asarray([float(x) for x in args.scales.split(",") if x.strip()], dtype=np.float32)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        histories = np.asarray(archive["true_history_xy"][trace_ids], dtype=np.float32)
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
    images = pd.read_csv(SOURCE / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    canvas_cache = {}
    rows = []
    for _, image_row in selected_images.iterrows():
        image_id = int(image_row.image_index)
        patch, _ = extract_patch(images.iloc[image_id], canvas_cache=canvas_cache, patch_size_px=540)
        image = _standardize_uint_like(patch)
        for trace_ordinal, (trace_id, history) in enumerate(zip(trace_ids, histories, strict=True)):
            stable_by_unit = None
            for scale in scales:
                pair = scaled_histories(history[None], np.asarray([0.0, scale], dtype=np.float32))
                stims = (make_corrected_causal_stims(image, pair, torch=scorer.torch) - 127.0) / 255.0
                result = score_pair(
                    scorer, readout, reduced, stims[:N_SCORED], stims[N_SCORED:],
                    selection, args.frame_batch_size,
                )
                for condition in ("stable", "moving"):
                    values = result[condition]
                    for subset_index, unit_index in enumerate(selection):
                        rows.append(
                            {
                                "image_index": image_id,
                                "trace_index": int(trace_id),
                                "scale": float(scale),
                                "condition": condition,
                                "subset_index": subset_index,
                                "unit_index": int(unit_index),
                                "response_subspace_group": selection_table.response_subspace_group.iloc[subset_index],
                                "map_rate_r2": values["r2"][subset_index],
                                "map_rate_correlation": values["correlation"][subset_index],
                                "exact_ssi": values["exact_ssi"][subset_index],
                                "reduced_ssi": values["reduced_ssi"][subset_index],
                                "exact_mean_rate": values["exact_rate"][subset_index],
                                "reduced_mean_rate": values["reduced_rate"][subset_index],
                            }
                        )
                print(f"image={image_id} trace={trace_ordinal + 1}/{len(trace_ids)} scale={scale:g}", flush=True)
    rows = pd.DataFrame(rows)
    stable = rows.loc[rows.condition.eq("stable"), ["image_index", "trace_index", "scale", "unit_index", "exact_ssi", "reduced_ssi"]].rename(columns={"exact_ssi": "exact_stable_ssi", "reduced_ssi": "reduced_stable_ssi"})
    rows = rows.merge(stable, on=["image_index", "trace_index", "scale", "unit_index"], how="left")
    rows["exact_ssi_delta"] = rows.exact_ssi - rows.exact_stable_ssi
    rows["reduced_ssi_delta"] = rows.reduced_ssi - rows.reduced_stable_ssi
    rows.to_csv(args.output_dir / "natural_fem_response_fidelity.csv", index=False)
    moving = rows.loc[rows.condition.eq("moving")]
    summary = {
        "state": str(args.state.resolve()),
        "n_images": int(len(selected_images)),
        "n_traces": int(len(selected_traces)),
        "median_natural_fem_map_rate_r2": float(moving.map_rate_r2.median()),
        "median_natural_fem_map_rate_correlation": float(moving.map_rate_correlation.median()),
        "ssi_delta_correlation": float(np.corrcoef(moving.exact_ssi_delta, moving.reduced_ssi_delta)[0, 1]),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(args.output_dir, rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
