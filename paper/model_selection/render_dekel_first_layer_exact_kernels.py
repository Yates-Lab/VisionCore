#!/usr/bin/env python3
"""Render exact first-layer spatiotemporal kernels from a Dekel checkpoint.

For every stem channel, the spatial panel is the effective 7x7 kernel at the
lag with the largest spatial RMS.  The adjacent temporal panel follows the
kernel through time at the most positive and most negative pixels in that
spatial slice without assuming the learned kernel is separable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sampling-rate", type=float, default=240.0)
    parser.add_argument("--model-label", default="M77")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from eval.load_twin import load_twin

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    model, _ = load_twin(checkpoint_path, device="cpu", verbose=False)

    core = getattr(model.model, "convnet", None)
    if core is None or not hasattr(core, "effective_temporal_weight"):
        raise TypeError(f"{checkpoint_path} has no compatible Dekel core")

    # This is the exact parameter used by the forward pass, after both the
    # fixed spatial envelope and the configured frequency-domain mask.
    weight = core.effective_temporal_weight().detach().float().cpu().numpy()
    if weight.ndim != 5 or weight.shape[1] != 1:
        raise ValueError(f"Expected [channel, 1, lag, y, x], got {weight.shape}")
    weight = weight[:, 0]
    n_channels, n_lags, height, width = weight.shape

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    positive_color = "#0072B2"
    negative_color = "#D55E00"
    neutral = "#4A4A4A"
    lag_ms = -np.arange(n_lags, dtype=float) * (1000.0 / args.sampling_rate)

    # Two channel pairs per row keeps both the 7x7 map and 60-point waveform
    # legible in the desktop preview.
    n_pairs = 2
    n_rows = int(np.ceil(n_channels / n_pairs))
    fig = plt.figure(figsize=(15.2, 2.65 * n_rows + 1.45), constrained_layout=False)
    grid = fig.add_gridspec(
        n_rows,
        4,
        width_ratios=(1.0, 2.05, 1.0, 2.05),
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.81,
        wspace=0.27,
        hspace=0.62,
    )

    records = []
    for channel in range(n_channels):
        row = channel % n_rows
        pair = channel // n_rows
        spatial_ax = fig.add_subplot(grid[row, 2 * pair])
        temporal_ax = fig.add_subplot(grid[row, 2 * pair + 1])

        kernel = weight[channel]
        lag_rms = np.sqrt(np.mean(kernel**2, axis=(1, 2)))
        peak_lag = int(np.argmax(lag_rms))
        spatial = kernel[peak_lag]
        positive_yx = tuple(int(v) for v in np.unravel_index(np.argmax(spatial), spatial.shape))
        negative_yx = tuple(int(v) for v in np.unravel_index(np.argmin(spatial), spatial.shape))
        positive_trace = kernel[:, positive_yx[0], positive_yx[1]]
        negative_trace = kernel[:, negative_yx[0], negative_yx[1]]

        spatial_scale = float(np.max(np.abs(spatial)))
        if spatial_scale == 0:
            spatial_scale = 1.0
        trace_scale = float(
            max(np.max(np.abs(positive_trace)), np.max(np.abs(negative_trace)), 1e-12)
        )

        spatial_ax.imshow(
            spatial / spatial_scale,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            origin="upper",
            interpolation="nearest",
        )
        spatial_ax.scatter(
            positive_yx[1], positive_yx[0], s=66, marker="o",
            facecolors="none", edgecolors=positive_color, linewidths=1.8,
        )
        spatial_ax.scatter(
            negative_yx[1], negative_yx[0], s=72, marker="s",
            facecolors="none", edgecolors=negative_color, linewidths=1.8,
        )
        spatial_ax.set_xticks([])
        spatial_ax.set_yticks([])
        for spine in spatial_ax.spines.values():
            spine.set_color("#B8B8B8")
            spine.set_linewidth(0.7)
        spatial_ax.set_title(
            f"f{channel:02d}  space at {lag_ms[peak_lag]:.0f} ms",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=4,
        )

        temporal_ax.plot(
            lag_ms,
            positive_trace / trace_scale,
            color=positive_color,
            linewidth=1.9,
        )
        temporal_ax.plot(
            lag_ms,
            negative_trace / trace_scale,
            color=negative_color,
            linewidth=1.9,
        )
        temporal_ax.axhline(0, color="#A8A8A8", linewidth=0.7)
        temporal_ax.axvline(lag_ms[peak_lag], color="#888888", linewidth=0.8, linestyle=":")
        temporal_ax.scatter(
            [lag_ms[peak_lag]],
            [positive_trace[peak_lag] / trace_scale],
            color=positive_color,
            s=18,
            zorder=3,
        )
        temporal_ax.scatter(
            [lag_ms[peak_lag]],
            [negative_trace[peak_lag] / trace_scale],
            color=negative_color,
            s=22,
            marker="s",
            zorder=3,
        )
        temporal_ax.set_xlim(lag_ms[-1], 0)
        temporal_ax.set_ylim(-1.08, 1.08)
        temporal_ax.set_xticks([-240, -180, -120, -60, 0])
        temporal_ax.set_yticks([-1, 0, 1])
        temporal_ax.tick_params(labelsize=8)
        temporal_ax.spines[["top", "right"]].set_visible(False)
        temporal_ax.spines[["left", "bottom"]].set_color("#777777")
        temporal_ax.set_title(
            f"time at spatial peak / trough  ·  RMS {lag_rms[peak_lag]:.3g}",
            loc="left",
            fontsize=9.2,
            pad=4,
            color=neutral,
        )
        if row == n_rows - 1:
            temporal_ax.set_xlabel("time relative to prediction (ms)", fontsize=9)
        if pair == 0:
            temporal_ax.set_ylabel("normalized weight", fontsize=9)

        records.append(
            {
                "channel": channel,
                "peak_energy_lag_index": peak_lag,
                "peak_energy_time_relative_to_prediction_ms": float(lag_ms[peak_lag]),
                "peak_spatial_yx": list(positive_yx),
                "trough_spatial_yx": list(negative_yx),
                "peak_lag_spatial_rms": float(lag_rms[peak_lag]),
                "peak_lag_spatial_scale": float(spatial_scale),
                "temporal_trace_scale": float(trace_scale),
            }
        )

    fig.suptitle(
        f"{args.model_label} first-layer spatiotemporal kernels",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.925,
        "Exact effective 60 × 7 × 7 weights · spatial slice chosen by maximum RMS across space · each channel normalized independently",
        ha="left",
        va="top",
        fontsize=10.2,
        color=neutral,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=positive_color, marker="o", markerfacecolor="none", label="temporal trace at spatial peak"),
            Line2D([0], [0], color=negative_color, marker="s", markerfacecolor="none", label="temporal trace at spatial trough"),
            Line2D([0], [0], color="#888888", linestyle=":", label="peak-energy lag"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.045, 0.885),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / "m77_first_layer_exact_kernels.png"
    pdf_path = args.out_dir / "m77_first_layer_exact_kernels.pdf"
    json_path = args.out_dir / "m77_first_layer_exact_kernels.json"
    fig.savefig(png_path, dpi=200, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "epoch": epoch,
        "model_label": args.model_label,
        "sampling_rate_hz": args.sampling_rate,
        "effective_weight_shape": list(weight.shape),
        "lag_convention": "kernel index 0 is the current frame (0 ms); larger indices are older",
        "spatial_slice_rule": "lag maximizing spatial RMS within each channel",
        "normalization": "spatial map and paired temporal traces normalized independently within channel for display",
        "channels": records,
        "artifacts": {"png": str(png_path.resolve()), "pdf": str(pdf_path.resolve())},
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
