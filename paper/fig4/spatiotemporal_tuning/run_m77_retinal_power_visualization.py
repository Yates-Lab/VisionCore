#!/usr/bin/env python3
"""Clean auxiliary-aperture retinal SF×TF×orientation power maps for M77.

The 255-pixel aperture is used only to reduce finite-crop spectral variance in
population visualizations.  Rate prediction always uses the exact 151-pixel
scorer aperture in ``run_m77_retinal_causal_chain.py``.  Every spectrum here
comes from a directly rendered movie and the complete eye trajectory; no
instantaneous velocity or k·v approximation is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import mode_to_grid_matrix
from paper.fig4.spatiotemporal_tuning.audit_m77_nonlinear_sharpening import evenly_spaced_rows
from paper.fig4.spatiotemporal_tuning.run_m77_retinal_causal_chain import (
    DEFAULT_CHAIN,
    DEFAULT_MATRIX,
    load_tuning_tensors,
    movie_power_cube,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    PPD,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, default=DEFAULT_MATRIX / "image_feature_table.csv")
    parser.add_argument("--trace-bank", type=Path, default=DEFAULT_CHAIN / "fixation_bank")
    parser.add_argument("--tuning-table", type=Path, default=DEFAULT_CHAIN / "dense_tuning/frequency_tuning_grouped.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trace-kind", choices=("filtered", "raw"), default="filtered")
    parser.add_argument("--n-images", type=int, default=20)
    parser.add_argument("--n-traces", type=int, default=20)
    parser.add_argument("--motion-scales", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0))
    parser.add_argument("--aperture-size-px", type=int, default=255)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--analysis-samples", type=int, default=240)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def spatial_frequency_grid(size: int, *, maximum_cpd: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.fft.fftfreq(int(size), d=1.0 / float(PPD))
    ky, kx = np.meshgrid(-axis, axis, indexing="ij")
    kxy = np.column_stack((kx.ravel(), ky.ravel()))
    radial = np.linalg.norm(kxy, axis=1)
    keep = (radial > 0) & (radial <= float(maximum_cpd) * 1.15)
    return kxy[keep], np.flatnonzero(keep)


def render_movie_size(
    patch: np.ndarray,
    trace: np.ndarray,
    *,
    out_size: int,
    device: str,
) -> np.ndarray:
    image = _standardize_uint_like(patch)
    eye = torch.from_numpy(np.asarray(trace, dtype=np.float32)).to(device)
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    source = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    repeated = source.unsqueeze(0).expand(len(trace), -1, -1)
    with torch.no_grad():
        movie = _shift_movie_with_eye(
            repeated,
            eye_norm,
            out_size=(int(out_size), int(out_size)),
            scale_factor=1.0,
            torch=torch,
        )
    return movie.float().cpu().numpy()


def render_power_figure(
    path: Path,
    *,
    scales: np.ndarray,
    spatial: np.ndarray,
    temporal: np.ndarray,
    power: np.ndarray,
) -> None:
    marginal = power.sum(axis=-1)
    total = marginal.sum(axis=(1, 2))
    measured = int(np.argmin(np.abs(scales - 1.0)))
    relative = total / max(float(total[measured]), EPS)
    normalized = marginal / np.maximum(total[:, None, None], EPS)
    positive = normalized[normalized > 0]
    floor = max(float(np.quantile(positive, 0.02)), EPS)
    ceiling = max(float(np.quantile(positive, 0.995)), floor * 10)
    figure = plt.figure(figsize=(4.0 * len(scales), 7.3), constrained_layout=True)
    grid = figure.add_gridspec(2, len(scales), height_ratios=(1.0, 0.43))
    contour = None
    for index, scale in enumerate(scales):
        axis = figure.add_subplot(grid[0, index])
        if total[index] <= EPS:
            axis.set_facecolor("#0A061E")
            axis.text(0.5, 0.5, "no dynamic power", color="white", ha="center", va="center", transform=axis.transAxes)
        else:
            display = np.log10(np.maximum(normalized[index].T, floor))
            contour = axis.contourf(
                spatial,
                temporal,
                display,
                levels=np.linspace(np.log10(floor), np.log10(ceiling), 14),
                cmap="magma",
                extend="both",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlim(float(spatial[0]), float(spatial[-1]))
        axis.set_ylim(float(temporal[0]), float(temporal[-1]))
        axis.set_title(f"{scale:g}× motion\nAC power {relative[index]:.2f}×")
        axis.set_xlabel("spatial frequency (cycles/degree)")
        if index == 0:
            axis.set_ylabel("temporal frequency (Hz)")
    if contour is not None:
        bar = figure.colorbar(contour, ax=figure.axes[: len(scales)], shrink=0.78, pad=0.012)
        bar.set_label("log10 fraction of dynamic power")

    temporal_marginal = normalized.sum(axis=1)
    axis = figure.add_subplot(grid[1, : len(scales) // 2 or 1])
    for index, scale in enumerate(scales):
        if total[index] > EPS:
            axis.plot(temporal, temporal_marginal[index], lw=1.8, label=f"{scale:g}×")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("temporal frequency (Hz)")
    axis.set_ylabel("fraction of dynamic power")
    axis.set_title("Temporal marginal")
    axis.legend(frameon=False, ncol=2)

    axis = figure.add_subplot(grid[1, len(scales) // 2 or 1 :])
    axis.plot(scales, relative, "o-", color="#D95F30", lw=2)
    axis.axhline(1, color="0.5", lw=0.8, ls="--")
    axis.set_xlabel("motion scale")
    axis.set_ylabel("total dynamic power relative to measured motion")
    axis.set_title("Motion changes both spectral location and total power")
    figure.suptitle(
        f"Direct-rendered {args.aperture_size_px}-pixel retinal movies · complete real trajectories",
        fontsize=15,
        fontweight="semibold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.aperture_size_px < 151 or args.aperture_size_px % 2 != 1:
        raise ValueError("auxiliary aperture must be odd and at least 151 pixels")
    scales = np.asarray(args.motion_scales, dtype=float)
    if scales[0] != 0 or np.any(scales < 0):
        raise ValueError("motion scales must begin at zero and remain nonnegative")
    images = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    trace_table = pd.read_csv(args.trace_bank / "trace_table.csv").sort_values("trace_index").reset_index(drop=True)
    traces = np.load(args.trace_bank / f"trace_xy_{args.trace_kind}.npy", mmap_mode="r")
    tuning = load_tuning_tensors(args.tuning_table)
    image_rows = evenly_spaced_rows(len(images), args.n_images)
    trace_rows = evenly_spaced_rows(len(trace_table), args.n_traces)
    kxy, flat_index = spatial_frequency_grid(
        args.aperture_size_px, maximum_cpd=float(tuning["spatial_cpd"][-1])
    )
    distributor, resolved = mode_to_grid_matrix(
        kxy, tuning["spatial_cpd"], tuning["orientation_deg"]
    )
    accumulator = None
    temporal = None
    canvas_cache: dict = {}
    for image_position, image_row in enumerate(image_rows):
        patch, _ = extract_patch(images.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=args.patch_size_px)
        for trace_row in trace_rows:
            trace = np.asarray(traces[int(trace_row)], dtype=np.float32)
            for scale_index, scale in enumerate(scales):
                movie = render_movie_size(
                    patch,
                    trace * float(scale),
                    out_size=args.aperture_size_px,
                    device=args.device,
                )[-args.analysis_samples :]
                frequency, cube = movie_power_cube(
                    movie,
                    flat_index=flat_index,
                    mode_to_grid=distributor,
                    n_spatial=len(tuning["spatial_cpd"]),
                    n_orientation=len(tuning["orientation_deg"]),
                    frame_rate_hz=240.0,
                )
                if accumulator is None:
                    temporal = frequency
                    accumulator = np.zeros((len(scales), *cube.shape), dtype=np.float64)
                elif not np.array_equal(temporal, frequency):
                    raise RuntimeError("temporal FFT grid changed")
                accumulator[scale_index] += cube
        print(f"auxiliary retinal power image {image_position + 1}/{len(image_rows)}", flush=True)
    if accumulator is None or temporal is None:
        raise RuntimeError("no auxiliary retinal movies were analyzed")
    accumulator /= float(len(image_rows) * len(trace_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "auxiliary_retinal_power.npz"
    np.savez_compressed(
        archive,
        power=accumulator.astype(np.float32),
        motion_scales=scales,
        spatial_cpd=tuning["spatial_cpd"],
        temporal_hz=temporal,
        orientation_deg=tuning["orientation_deg"],
        image_rows=image_rows,
        trace_rows=trace_rows,
    )
    figure = args.out_dir / "m77_auxiliary_retinal_power.png"
    render_power_figure(
        figure,
        scales=scales,
        spatial=tuning["spatial_cpd"],
        temporal=temporal,
        power=accumulator,
    )
    summary = {
        "analysis": "direct-rendered auxiliary-aperture real retinal-movie power",
        "behavior": "not evaluated; visual stimulus analysis only",
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_image_trace_pairs": int(len(image_rows) * len(trace_rows)),
        "trace_kind": args.trace_kind,
        "aperture_size_px": int(args.aperture_size_px),
        "frame_rate_hz": 240,
        "analysis_seconds": float(args.analysis_samples / 240.0),
        "spectrum": "exact directly rendered movies; spatial Tukey alpha=0.15; temporal mean removal; DPSS NW=1.5 K=2; positive/negative TF folded; complete trajectories",
        "instantaneous_velocity_used": False,
        "resolved_spatial_fourier_modes": int(np.count_nonzero(resolved)),
        "archive": str(archive.resolve()),
        "figure": str(figure.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
