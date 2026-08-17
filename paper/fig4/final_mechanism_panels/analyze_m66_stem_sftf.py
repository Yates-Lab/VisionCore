#!/usr/bin/env python3
"""Visualize the full spatiotemporal spectra of M66's three temporal stems.

Unlike a temporal marginal or rank-1 approximation, this diagnostic Fourier
transforms every complete 60 x 7 x 7 effective kernel.  Spatial orientation is
collapsed only after power is computed, leaving a radial-SF x absolute-TF
description of the feature bank present at the point where time disappears as
an explicit tensor axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/"
    "D240M66a_m63e31_m64be63_readout_coord_teacher0p4_s201/"
    "analysis_candidates/epoch=031-endpoint.ckpt"
)
DEFAULT_OUT = ROOT / "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sampling-rate", type=float, default=240.0)
    parser.add_argument("--pixels-per-degree", type=float, default=37.50476617)
    parser.add_argument("--temporal-fft-size", type=int, default=256)
    parser.add_argument("--spatial-fft-size", type=int, default=64)
    parser.add_argument("--max-spatial-frequency", type=float, default=12.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_spectrum(
    core,
    *,
    sampling_rate: float,
    pixels_per_degree: float,
    temporal_fft_size: int,
    spatial_fft_size: int,
    max_spatial_frequency: float,
) -> dict[str, np.ndarray | int]:
    weights = core.effective_temporal_weight().detach().float().cpu()
    if weights.shape[1] != 1:
        raise ValueError(f"Expected grayscale stem weights, got {tuple(weights.shape)}")
    weights = weights[:, 0]
    spectrum = torch.fft.fftn(
        weights,
        s=(temporal_fft_size, spatial_fft_size, spatial_fft_size),
        dim=(-3, -2, -1),
    )
    power = spectrum.abs().square().numpy()

    temporal_hz = np.abs(
        np.fft.fftfreq(temporal_fft_size, d=1.0 / sampling_rate)
    )
    spatial_axis = np.fft.fftfreq(
        spatial_fft_size, d=1.0 / pixels_per_degree
    )
    fy, fx = np.meshgrid(spatial_axis, spatial_axis, indexing="ij")
    spatial_cpd = np.sqrt(fx**2 + fy**2)
    tf_grid = np.broadcast_to(
        temporal_hz[:, None, None],
        (temporal_fft_size, spatial_fft_size, spatial_fft_size),
    )
    sf_grid = np.broadcast_to(spatial_cpd[None], tf_grid.shape)

    sf_edges = np.linspace(0.0, max_spatial_frequency, 49)
    tf_edges = np.linspace(0.0, sampling_rate / 2.0, 61)
    bank = np.zeros((len(tf_edges) - 1, len(sf_edges) - 1), dtype=np.float64)
    centroids = []
    for filter_power in power:
        normalized = filter_power.astype(np.float64)
        normalized /= max(float(np.sum(normalized)), 1e-12)
        hist, _, _ = np.histogram2d(
            tf_grid.ravel(),
            sf_grid.ravel(),
            bins=(tf_edges, sf_edges),
            weights=normalized.ravel(),
        )
        bank += hist
        in_view = sf_grid <= max_spatial_frequency
        selected_power = normalized[in_view]
        denominator = max(float(np.sum(selected_power)), 1e-12)
        centroids.append(
            [
                float(np.sum(sf_grid[in_view] * selected_power) / denominator),
                float(np.sum(tf_grid[in_view] * selected_power) / denominator),
            ]
        )
    bank /= int(weights.shape[0])
    return {
        "n_filters": int(weights.shape[0]),
        "power": bank,
        "sf_edges": sf_edges,
        "tf_edges": tf_edges,
        "centroids": np.asarray(centroids, dtype=np.float64),
    }


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def render(results: dict[str, dict[str, np.ndarray | int]], out_dir: Path) -> None:
    configure()
    positive = np.concatenate(
        [np.asarray(result["power"])[np.asarray(result["power"]) > 0] for result in results.values()]
    )
    vmax = float(np.max(positive))
    vmin = max(float(np.quantile(positive, 0.10)), vmax * 1e-5)
    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad(color_map(0.0))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.55),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    titles = {
        "base": "main core",
        "auxiliary": "auxiliary core",
        "residual": "residual core",
    }
    mesh = None
    for axis, (name, result) in zip(axes, results.items()):
        mesh = axis.pcolormesh(
            result["sf_edges"],
            result["tf_edges"],
            result["power"],
            shading="auto",
            cmap=color_map,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            rasterized=True,
        )
        centroids = np.asarray(result["centroids"])
        axis.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=13,
            facecolors="none",
            edgecolors="white",
            linewidths=0.65,
        )
        axis.set_title(f"{titles[name]} ({result['n_filters']} filters)", pad=5)
        axis.set_xlim(0, float(np.max(result["sf_edges"])))
        axis.set_ylim(0, float(np.max(result["tf_edges"])))
        axis.set_xlabel("spatial frequency (cycles/deg)")
    axes[0].set_ylabel("temporal frequency |f| (Hz)")
    colorbar = figure.colorbar(mesh, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("mean within-filter power")
    figure.suptitle(
        "M66 retains spatiotemporal frequency channels at the temporal-collapse stem",
        x=0.07,
        y=1.01,
        ha="left",
        fontsize=9.5,
        fontweight="semibold",
    )
    figure.text(0.012, 1.01, "E", ha="left", va="top", fontsize=11, fontweight="bold")
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.20, top=0.82)
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(
            out_dir / f"figure4_panel_e_m66_stem_sftf.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from eval.load_twin import load_twin

    model, _ = load_twin(args.checkpoint.resolve(), device="cpu", verbose=False)
    cores = {
        "base": model.model.convnet,
        "auxiliary": model.model.auxiliary_convnet,
        "residual": model.model.residual_convnet,
    }
    results = {
        name: core_spectrum(
            core,
            sampling_rate=args.sampling_rate,
            pixels_per_degree=args.pixels_per_degree,
            temporal_fft_size=args.temporal_fft_size,
            spatial_fft_size=args.spatial_fft_size,
            max_spatial_frequency=args.max_spatial_frequency,
        )
        for name, core in cores.items()
    }
    render(results, args.out_dir)
    np.savez_compressed(
        args.out_dir / "panel_e_m66_stem_sftf_arrays.npz",
        **{
            f"{name}_{field}": value
            for name, result in results.items()
            for field, value in result.items()
        },
    )
    report = {
        "analysis": "complete effective-kernel SF x absolute-TF spectra for every M66 temporal stem",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "sampling_rate_hz": args.sampling_rate,
        "pixels_per_degree": args.pixels_per_degree,
        "temporal_fft_size": args.temporal_fft_size,
        "spatial_fft_size": args.spatial_fft_size,
        "max_displayed_spatial_frequency_cpd": args.max_spatial_frequency,
        "power_normalization": "each complete 60x7x7 filter is unit-power before branch averaging",
        "orientation_handling": "power is computed in the full 3D Fourier domain before radial spatial-orientation collapse",
        "filter_centroids_sf_cpd_tf_hz": {
            name: np.asarray(result["centroids"]).tolist()
            for name, result in results.items()
        },
        "claim_boundary": "filter-bank support alone does not establish that eye-motion-driven activation in these channels causes the final SSI change",
    }
    (args.out_dir / "panel_e_m66_stem_sftf_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
