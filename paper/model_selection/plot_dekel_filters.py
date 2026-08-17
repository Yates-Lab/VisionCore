#!/usr/bin/env python3
"""Write first-layer filter panels from a Dekel checkpoint without loading data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sampling-rate", type=float, default=240.0)
    parser.add_argument(
        "--core",
        choices=("base", "auxiliary", "residual"),
        default="base",
        help=(
            "Plot the mature base core, the frozen auxiliary core, or the "
            "trainable smooth residual core"
        ),
    )
    args = parser.parse_args()

    from eval.load_twin import load_twin

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    default_out_dir = (
        ROOT / "outputs" / "dekel240_diagnostics" / checkpoint_path.parent.name
        / f"epoch_{epoch:03d}"
    )
    if args.core != "base":
        default_out_dir = default_out_dir / f"{args.core}_core"
    out_dir = args.out_dir or default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_twin(checkpoint_path, device="cpu", verbose=False)
    core_attributes = {
        "base": "convnet",
        "auxiliary": "auxiliary_convnet",
        "residual": "residual_convnet",
    }
    core = getattr(model.model, core_attributes[args.core], None)
    if core is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no {args.core} Dekel core"
        )
    if not hasattr(core, "first_layer_separable_components"):
        raise TypeError(
            f"Selected {args.core} core does not expose Dekel filter diagnostics"
        )
    temporal, spatial, separability = core.first_layer_separable_components()

    temporal_figure = core.plot_temporal_filters(sampling_rate=args.sampling_rate)
    temporal_path = out_dir / "first_layer_temporal_and_frequency.png"
    temporal_figure.savefig(temporal_path, dpi=180, bbox_inches="tight")
    temporal_figure.savefig(
        out_dir / "first_layer_temporal_and_frequency.pdf", bbox_inches="tight"
    )
    spatial_figure = core.plot_first_layer_spatial_filters()
    spatial_path = out_dir / "first_layer_spatial.png"
    spatial_figure.savefig(spatial_path, dpi=180, bbox_inches="tight")
    spatial_figure.savefig(out_dir / "first_layer_spatial.pdf", bbox_inches="tight")

    temporal_frequency = torch.fft.rfftfreq(
        temporal.shape[1], d=1.0 / args.sampling_rate
    )
    temporal_power = torch.fft.rfft(temporal, dim=1).abs().square()

    # Compact one-panel versions render more reliably in the desktop preview
    # than the original wide two-panel contact sheet.
    import matplotlib.pyplot as plt

    time_ms = torch.arange(temporal.shape[1]) * (1000.0 / args.sampling_rate)
    profile_figure, profile_axis = plt.subplots(figsize=(7.0, 4.2))
    spectrum_figure, spectrum_axis = plt.subplots(figsize=(7.0, 4.2))
    for idx in range(temporal.shape[0]):
        profile_axis.plot(
            time_ms.numpy(),
            temporal[idx].numpy(),
            label=f"f{idx} ({100 * separability[idx]:.0f}% rank-1)",
        )
        spectrum_axis.plot(
            temporal_frequency.numpy(), temporal_power[idx].numpy(), label=f"f{idx}"
        )
    profile_axis.axhline(0, color="0.6", linewidth=0.8)
    profile_axis.set(
        xlabel="history position (ms)",
        ylabel="leading SVD component",
        title="First-layer temporal profiles",
    )
    profile_axis.legend(ncol=2, fontsize=8, frameon=False)
    profile_axis.spines[["top", "right"]].set_visible(False)
    spectrum_axis.set(
        xlabel="temporal frequency (Hz)",
        ylabel="power",
        title="First-layer temporal spectra",
        xlim=(0, args.sampling_rate / 2),
    )
    spectrum_axis.spines[["top", "right"]].set_visible(False)
    profile_figure.tight_layout()
    spectrum_figure.tight_layout()
    profile_path = out_dir / "first_layer_temporal_profiles.png"
    spectrum_path = out_dir / "first_layer_temporal_spectra.png"
    profile_figure.savefig(profile_path, dpi=160, bbox_inches="tight", facecolor="white")
    profile_figure.savefig(
        out_dir / "first_layer_temporal_profiles.pdf", bbox_inches="tight"
    )
    spectrum_figure.savefig(
        spectrum_path, dpi=160, bbox_inches="tight", facecolor="white"
    )
    spectrum_figure.savefig(
        out_dir / "first_layer_temporal_spectra.pdf", bbox_inches="tight"
    )
    plt.close(temporal_figure)
    plt.close(spatial_figure)
    plt.close(profile_figure)
    plt.close(spectrum_figure)

    temporal_high = temporal_power[:, temporal_frequency >= args.sampling_rate / 4].sum(1)
    temporal_high = temporal_high / temporal_power.sum(1).clamp_min(1e-12)

    temporal_conv = core.temporal_conv
    raw_weight = temporal_conv.conv.weight.detach().float()
    hamming_weight = raw_weight * temporal_conv.spatial_window.detach().float()
    effective_weight = temporal_conv.weight.detach().float()

    def full_weight_temporal_high_fraction(weight):
        power = torch.fft.rfft(weight, dim=2).abs().square()
        high = power[:, :, temporal_frequency >= args.sampling_rate / 4].sum()
        return float(high / power.sum().clamp_min(1e-12))

    fy = torch.fft.fftfreq(spatial.shape[-2])[:, None]
    fx = torch.fft.fftfreq(spatial.shape[-1])[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    spatial_power = torch.fft.fft2(spatial, dim=(-2, -1)).abs().square()
    spatial_high = spatial_power[:, radius >= 0.25].sum(dim=1)
    spatial_high = spatial_high / spatial_power.sum(dim=(1, 2)).clamp_min(1e-12)

    report = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "core": args.core,
        "sampling_rate_hz": float(args.sampling_rate),
        "effective_parameterization": {
            "spatial_hamming_active": True,
            "frequency_mask_axes": list(temporal_conv.frequency_mask_axes),
            "frequency_window": temporal_conv.frequency_window,
            "frequency_fft_pad": temporal_conv.frequency_fft_pad,
            "raw_weight_temporal_high_frequency_fraction": (
                full_weight_temporal_high_fraction(raw_weight)
            ),
            "hamming_only_temporal_high_frequency_fraction": (
                full_weight_temporal_high_fraction(hamming_weight)
            ),
            "effective_weight_temporal_high_frequency_fraction": (
                full_weight_temporal_high_fraction(effective_weight)
            ),
            "effective_differs_from_hamming_only": bool(
                not torch.equal(effective_weight, hamming_weight)
            ),
        },
        "temporal_components": temporal.tolist(),
        "spatial_components": spatial.tolist(),
        "temporal_high_frequency_fraction_per_filter": temporal_high.tolist(),
        "temporal_high_frequency_fraction_mean": float(temporal_high.mean()),
        "spatial_high_frequency_fraction_per_filter": spatial_high.tolist(),
        "spatial_high_frequency_fraction_mean": float(spatial_high.mean()),
        "rank1_fraction_per_filter": separability.tolist(),
        "rank1_fraction_mean": float(separability.mean()),
        "artifacts": {
            "first_layer_temporal": str(temporal_path),
            "first_layer_temporal_profiles": str(profile_path),
            "first_layer_temporal_spectra": str(spectrum_path),
            "first_layer_spatial": str(spatial_path),
        },
    }
    report_path = out_dir / "first_layer_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
