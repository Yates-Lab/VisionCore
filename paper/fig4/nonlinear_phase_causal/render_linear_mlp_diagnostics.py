#!/usr/bin/env python3
"""Show the learned linear subspace and held-out predictions of a reduced twin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
from paper.fig4.nonlinear_phase_causal.reduced_twin import ReducedTwin
from paper.fig4.nonlinear_phase_causal.response_subspace import dct_filters_to_movies, r2_score
from paper.fig4.nonlinear_phase_causal.run_experiment import SOURCE, _selection, preactivation
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer, _standardize_uint_like
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


DEFAULT_RUN = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/linear_mlp_reduction"
DEFAULT_BANK = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/"
    "reduced_twin_natural_calibration_all_scales/bank"
)
SELECTION = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot/"
    "unit_selection.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--architecture", default="mlp_64x32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scatter-samples", type=int, default=5000)
    return parser.parse_args()


@torch.no_grad()
def reduced_bank_predictions(reduced, encoder, bank_dir, device):
    raw = torch.from_numpy(np.asarray(np.load(bank_dir / "test_generators.npy"))).to(device).float()
    target = np.asarray(np.load(bank_dir / "test_rates.npy"), dtype=np.float32)
    source_std = torch.from_numpy(encoder["source_generator_std"]).to(device).float()
    weight = torch.from_numpy(encoder["weight"]).to(device).float()
    transform = weight / source_std[None]
    predictions = []
    encoded_rows = []
    for start in range(0, len(raw), 4096):
        encoded = raw[start : start + 4096] @ transform.T
        encoded_rows.append(encoded.cpu())
        predictions.append(reduced.decode_generators(encoded).cpu())
    return target, torch.cat(predictions).numpy(), torch.cat(encoded_rows)


@torch.no_grad()
def mode_importance(reduced, encoded, target, sample_count=8192):
    count = min(sample_count, len(encoded))
    ids = torch.linspace(0, len(encoded) - 1, count).long()
    value = encoded[ids].to(reduced.basis.device)
    target_std = torch.from_numpy(target.std(0).clip(1e-6)).to(value.device)
    baseline = reduced.decode_generators(value)
    importance = []
    mean = reduced.generator_mean
    for mode in range(reduced.rank):
        ablated = value.clone()
        ablated[:, mode] = mean[mode]
        prediction = reduced.decode_generators(ablated)
        importance.append(((baseline - prediction) / target_std).square().mean(0).cpu().numpy())
    return np.stack(importance)


def render_subspace(output_dir, state, importance):
    physical = state.basis / state.feature_std[None]
    filters = dct_filters_to_movies(physical).numpy()
    total_importance = importance.sum(1)
    order = np.argsort(total_importance)[::-1]
    shown = order[: min(16, len(order))]
    fig, axes = plt.subplots(4, 8, figsize=(14.2, 6.5), squeeze=False)
    temporal_rows = (1, 3)
    for display_index, mode in enumerate(shown):
        block = display_index // 8
        column = display_index % 8
        image_axis = axes[2 * block, column]
        trace_axis = axes[2 * block + 1, column]
        movie = filters[mode]
        energy = np.square(movie).sum((1, 2))
        peak = int(np.argmax(energy))
        spatial = movie[peak]
        limit = np.percentile(np.abs(spatial), 99)
        image_axis.imshow(spatial, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        image_axis.axhline(12, color="0.3", lw=0.4, alpha=0.4)
        image_axis.axvline(12, color="0.3", lw=0.4, alpha=0.4)
        image_axis.set_title(f"mode {mode + 1} | lag {31 - peak}", fontsize=8)
        image_axis.set_xticks([]); image_axis.set_yticks([])
        template = spatial / np.sqrt(np.square(spatial).sum()).clip(1e-10)
        signed_temporal = np.einsum("thw,hw->t", movie, template)
        trace_axis.plot(np.arange(31, -1, -1), signed_temporal, color="#2f6f9f", lw=1)
        trace_axis.axhline(0, color="0.6", lw=0.5)
        trace_axis.invert_xaxis()
        trace_axis.set_xticks([31, 15, 0])
        trace_axis.tick_params(labelsize=6)
        if column == 0:
            trace_axis.set_ylabel("projection", fontsize=7)
        if block == 1:
            trace_axis.set_xlabel("lag (frames)", fontsize=7)
    for row in temporal_rows:
        for axis in axes[row]:
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Supervised rank-32 linear encoder: spatial filter at peak-energy lag and signed temporal profile", y=0.998)
    fig.tight_layout()
    fig.savefig(output_dir / "learned_subspace_top16.png", dpi=220)
    fig.savefig(output_dir / "learned_subspace_top16.pdf")
    plt.close(fig)

    units = pd.read_csv(SELECTION).unit_index.to_numpy(int)
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    image = axis.imshow(importance[shown], aspect="auto", cmap="magma")
    axis.set(
        xlabel="RR100 unit",
        ylabel="learned encoder mode (importance order)",
        xticks=np.arange(len(units)),
        xticklabels=[f"u{x:03d}" for x in units],
        yticks=np.arange(len(shown)),
        yticklabels=[f"m{x + 1}" for x in shown],
    )
    axis.tick_params(axis="x", rotation=55)
    fig.colorbar(image, ax=axis, label="normalized prediction change when mode is ablated")
    fig.tight_layout()
    fig.savefig(output_dir / "learned_mode_unit_importance.png", dpi=220)
    fig.savefig(output_dir / "learned_mode_unit_importance.pdf")
    plt.close(fig)
    return order, filters


def render_scatter(output_dir, target, prediction, selection, sample_count):
    rng = np.random.default_rng(20260813)
    ids = rng.choice(len(target), min(sample_count, len(target)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.0), squeeze=False)
    rows = []
    for unit, axis in enumerate(axes.ravel()):
        true, pred = target[:, unit], prediction[:, unit]
        r2 = float(r2_score(true, pred, axis=0))
        corr = float(np.corrcoef(true, pred)[0, 1])
        limit = np.quantile(np.r_[true[ids], pred[ids]], [0.002, 0.998])
        axis.scatter(true[ids], pred[ids], s=3, alpha=0.18, color="#2878a8", rasterized=True)
        axis.plot(limit, limit, color="0.35", ls="--", lw=1)
        axis.set(xlim=limit, ylim=limit, title=f"u{int(selection.unit_index.iloc[unit]):03d}  $R^2$={r2:.2f}, r={corr:.2f}")
        if unit // 4 == 2:
            axis.set_xlabel("exact rate")
        if unit % 4 == 0:
            axis.set_ylabel("reduced rate")
        rows.append({"unit_index": int(selection.unit_index.iloc[unit]), "test_rate_r2": r2, "test_rate_correlation": corr})
    fig.suptitle("Held-out images and trajectories: exact versus rank-32 linear-MLP responses", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_response_scatter_all_units.png", dpi=220)
    fig.savefig(output_dir / "heldout_response_scatter_all_units.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "heldout_response_scatter_metrics.csv", index=False)


@torch.no_grad()
def render_partial_dependence(output_dir, reduced, importance, selection):
    coordinates = torch.linspace(-3.5, 3.5, 141, device=reduced.basis.device)
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.8), squeeze=False)
    records = []
    for unit, axis in enumerate(axes.ravel()):
        modes = np.argsort(importance[:, unit])[::-1][:4]
        for mode in modes:
            normalized = torch.zeros(len(coordinates), reduced.rank, device=coordinates.device)
            normalized[:, mode] = coordinates
            rate = reduced.decoder(normalized)[:, unit].cpu().numpy()
            axis.plot(coordinates.cpu(), rate, lw=1.5, label=f"m{mode + 1}")
            records.extend(
                {
                    "unit_index": int(selection.unit_index.iloc[unit]),
                    "mode": int(mode + 1),
                    "normalized_generator": float(value),
                    "predicted_rate": float(response),
                }
                for value, response in zip(coordinates.cpu().numpy(), rate, strict=True)
            )
        axis.axvline(0, color="0.65", lw=0.6)
        axis.set_title(f"u{int(selection.unit_index.iloc[unit]):03d}")
        if unit // 4 == 2:
            axis.set_xlabel("one normalized generator")
        if unit % 4 == 0:
            axis.set_ylabel("MLP-predicted rate")
        axis.legend(frameon=False, fontsize=7, ncol=2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Learned response nonlinearity: one-mode partial dependence at the mean generator state", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "learned_nonlinearity_partial_dependence.png", dpi=220)
    fig.savefig(output_dir / "learned_nonlinearity_partial_dependence.pdf")
    plt.close(fig)
    pd.DataFrame(records).to_csv(output_dir / "learned_nonlinearity_partial_dependence.csv", index=False)


@torch.no_grad()
def heldout_example(args, reduced, selection):
    selected_images, selected_traces = _selection()
    image_id = int(selected_images.iloc[5].image_index)
    trace_id = int(selected_traces.iloc[18].trace_bank_index)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        history = np.asarray(archive["true_history_xy"][trace_id], dtype=np.float32)
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
    patch, _ = extract_patch(images.iloc[image_id], canvas_cache={}, patch_size_px=540)
    image = _standardize_uint_like(patch)
    scales = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    exact_by_scale, reduced_by_scale = [], []
    for scale in scales:
        scaled = scaled_histories(history[None], np.asarray([scale], dtype=np.float32))
        stims = (make_corrected_causal_stims(image, scaled, torch=scorer.torch) - 127.0) / 255.0
        exact_chunks, reduced_chunks = [], []
        for start in range(0, N_SCORED, 8):
            x = stims[start : start + 8].to(scorer.device)
            behavior = scorer._zero_behavior(len(x), x.dtype)
            exact_chunks.append(
                scorer.model.model.activation(
                    preactivation(scorer.model.model, readout, x, behavior)[:, selection]
                ).cpu()
            )
            reduced_chunks.append(reduced(x).cpu())
        exact_by_scale.append(torch.cat(exact_chunks).numpy())
        reduced_by_scale.append(torch.cat(reduced_chunks).numpy())
    return image_id, trace_id, scales, np.stack(exact_by_scale), np.stack(reduced_by_scale)


def render_map_examples(output_dir, selection, image_id, trace_id, scales, exact, prediction):
    frame = N_SCORED // 2
    units = selection.unit_index.to_numpy(int)
    np.savez_compressed(
        output_dir / "heldout_map_example_arrays.npz",
        image_index=image_id,
        trace_index=trace_id,
        scales=scales,
        frame_index=frame,
        unit_indices=units,
        exact_rate_maps=exact,
        reduced_rate_maps=prediction,
    )
    with PdfPages(output_dir / "heldout_map_examples_all_units.pdf") as pdf:
        for unit, unit_index in enumerate(units):
            fig, axes = plt.subplots(len(scales), 3, figsize=(7.6, 7.0), squeeze=False)
            rate_limit = np.quantile(np.r_[exact[:, frame, unit], prediction[:, frame, unit]], 0.995)
            residual_limit = np.quantile(np.abs(prediction[:, frame, unit] - exact[:, frame, unit]), 0.995)
            for row, scale in enumerate(scales):
                values = (exact[row, frame, unit], prediction[row, frame, unit])
                residual = values[1] - values[0]
                axes[row, 0].imshow(values[0], cmap="viridis", vmin=0, vmax=rate_limit)
                axes[row, 1].imshow(values[1], cmap="viridis", vmin=0, vmax=rate_limit)
                axes[row, 2].imshow(residual, cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit)
                axes[row, 0].set_ylabel(f"{scale:g}x")
                map_r2 = float(r2_score(values[0].ravel(), values[1].ravel(), axis=0))
                axes[row, 1].text(0.03, 0.04, f"map $R^2$={map_r2:.2f}", color="white", transform=axes[row, 1].transAxes, fontsize=8)
                for axis in axes[row]:
                    axis.set_xticks([]); axis.set_yticks([])
            axes[0, 0].set_title("exact twin")
            axes[0, 1].set_title("rank-32 linear-MLP")
            axes[0, 2].set_title("reduced - exact")
            fig.suptitle(f"Held-out image {image_id}, trajectory {trace_id}, unit u{unit_index:03d}, output frame {frame}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    scale_index = int(np.flatnonzero(np.isclose(scales, 1.0))[0])
    fig, axes = plt.subplots(len(units), 3, figsize=(6.8, 1.85 * len(units)), squeeze=False)
    for unit, unit_index in enumerate(units):
        true = exact[scale_index, frame, unit]
        pred = prediction[scale_index, frame, unit]
        rate_limit = np.quantile(np.r_[true, pred], 0.995)
        error_limit = np.quantile(np.abs(pred - true), 0.995)
        axes[unit, 0].imshow(true, cmap="viridis", vmin=0, vmax=rate_limit)
        axes[unit, 1].imshow(pred, cmap="viridis", vmin=0, vmax=rate_limit)
        axes[unit, 2].imshow(pred - true, cmap="RdBu_r", vmin=-error_limit, vmax=error_limit)
        axes[unit, 0].set_ylabel(f"u{unit_index:03d}")
        axes[unit, 1].text(0.03, 0.04, f"$R^2$={r2_score(true.ravel(), pred.ravel(), axis=0):.2f}", color="white", fontsize=7, transform=axes[unit, 1].transAxes)
        for axis in axes[unit]:
            axis.set_xticks([]); axis.set_yticks([])
    axes[0, 0].set_title("exact 1x")
    axes[0, 1].set_title("rank-32 prediction")
    axes[0, 2].set_title("residual")
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_map_examples_scale1_all_units.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.run_dir / f"diagnostics_rank{args.rank}_{args.architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / f"linear_mlp_rank{args.rank}_{args.architecture}.pt"
    encoder_path = args.run_dir / f"linear_encoder_rank{args.rank}_{args.architecture}.npz"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    reduced = ReducedTwin(state).to(args.device).eval()
    encoder = np.load(encoder_path)
    selection = pd.read_csv(SELECTION)
    target, prediction, encoded = reduced_bank_predictions(reduced, encoder, args.bank_dir, args.device)
    importance = mode_importance(reduced, encoded, target)
    order, _ = render_subspace(output_dir, state, importance)
    render_scatter(output_dir, target, prediction, selection, args.scatter_samples)
    render_partial_dependence(output_dir, reduced, importance, selection)
    image_id, trace_id, scales, exact, predicted_maps = heldout_example(
        args, reduced, selection.unit_index.to_numpy(int)
    )
    render_map_examples(output_dir, selection, image_id, trace_id, scales, exact, predicted_maps)
    summary = {
        "model_state": str(state_path.resolve()),
        "rank": args.rank,
        "architecture": args.architecture,
        "heldout_image_index": image_id,
        "heldout_trace_index": trace_id,
        "mode_importance_order_zero_based": order.tolist(),
        "median_sampled_test_r2": float(
            np.median([r2_score(target[:, unit], prediction[:, unit], axis=0) for unit in range(target.shape[1])])
        ),
    }
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
