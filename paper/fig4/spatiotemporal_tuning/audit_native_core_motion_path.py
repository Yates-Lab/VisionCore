#!/usr/bin/env python3
"""Track controlled retinal motion through a native-240 Dekel visual core.

This is the activation-level counterpart to the kinematic SF/TF calculation.
It replays the exact selected natural images and centered trajectories used by
the controlled SSI experiment, samples causal 60-frame histories, and measures
spatial information after the temporal stem and every spatial stage.  Because
the trained core applies GroupNorm/LRN after the temporal convolution, total
linear passband energy alone need not predict the nonlinear spatial code.
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
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EPS = 1e-10
LAYER_NAMES = ("temporal stem", "spatial stage 1", "spatial stage 2", "spatial stage 3")


def json_ready(value):
    """Recursively convert provenance values to stable JSON primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def selected_causal_histories(
    image: np.ndarray,
    trace_xy: np.ndarray,
    endpoint_indices: np.ndarray,
    *,
    n_lags: int,
    out_size: tuple[int, int],
    ppd: float,
) -> torch.Tensor:
    """Render only requested causal histories with the production geometry."""
    from paper.fig4.upstream.real_trace_matrix.model import (
        _eye_deg_to_norm,
        _shift_movie_with_eye,
    )

    trace = torch.as_tensor(np.asarray(trace_xy, dtype=np.float32))
    endpoints = torch.as_tensor(np.asarray(endpoint_indices, dtype=np.int64))
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError("trace_xy must have shape [time, 2]")
    if endpoints.ndim != 1 or torch.any(endpoints < 0) or torch.any(endpoints >= len(trace)):
        raise ValueError("endpoint index is outside the trace")
    prefix = trace[:1].repeat(int(n_lags) - 1, 1)
    padded = torch.cat((prefix, trace), dim=0)
    lag = torch.arange(int(n_lags), dtype=torch.long)
    # Lag zero is the newest frame, matching _embed_time_lags.
    history_rows = endpoints[:, None] + (int(n_lags) - 1) - lag[None]
    eye = padded[history_rows].reshape(-1, 2)
    eye_norm = _eye_deg_to_norm(
        eye,
        ppd=float(ppd),
        img_size=tuple(np.asarray(image).shape),
        torch=torch,
    )
    source = torch.from_numpy(np.asarray(image, dtype=np.float32))
    movie = source[None].expand(len(eye), -1, -1)
    shifted = _shift_movie_with_eye(
        movie,
        eye_norm,
        out_size=tuple(map(int, out_size)),
        scale_factor=1.0,
        torch=torch,
    )
    return shifted.reshape(len(endpoints), int(n_lags), *out_size).unsqueeze(1)


def spatial_information_components(activation: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Expected-activation-weighted spatial information by batch and channel."""
    if activation.ndim != 4 or torch.any(activation < 0):
        raise ValueError("activation must be nonnegative [batch, channel, y, x]")
    value = activation.double().flatten(start_dim=2)
    mean = value.mean(dim=2)
    gain = value / mean[..., None].clamp_min(EPS)
    bits = (gain * gain.clamp_min(EPS).log2()).mean(dim=2)
    return (bits * mean).cpu().numpy(), mean.cpu().numpy()


def percent_curve(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    pooled = np.sum(numerator, axis=tuple(range(numerator.ndim - 1))) / np.maximum(
        np.sum(denominator, axis=tuple(range(denominator.ndim - 1))), EPS
    )
    return 100.0 * (pooled - pooled[0]) / max(float(pooled[0]), EPS)


def filter_joint_spectra(core, *, ppd: float) -> dict[str, np.ndarray]:
    """Return joint SF/TF support of the learned 3-D stem filters.

    Spatial radialization happens only after the full 3-D Fourier power is
    computed, so inseparable direction/speed structure is retained in the
    per-channel SF/TF surface.  Zero-padding interpolates the response of the
    finite learned kernel; it does not add measurements or frequency content.
    """
    weight = core.effective_temporal_weight().detach().float().cpu().numpy()[:, 0]
    temporal_frequency = np.fft.rfftfreq(256, d=1.0 / 240.0)
    temporal_fft = np.fft.rfft(weight, n=256, axis=1)
    joint_fft = np.fft.fft2(temporal_fft, s=(128, 128), axes=(2, 3))
    power = np.abs(joint_fft) ** 2

    spatial_axis = np.fft.fftfreq(128, d=1.0 / float(ppd))
    fy, fx = np.meshgrid(spatial_axis, spatial_axis, indexing="ij")
    radius = np.hypot(fx, fy)
    edges = np.linspace(0.0, float(radius.max()) + 1e-9, 97)
    spatial_frequency = 0.5 * (edges[:-1] + edges[1:])
    radial_bin = np.digitize(radius.ravel(), edges) - 1
    joint = np.zeros(
        (len(weight), len(temporal_frequency), len(spatial_frequency)),
        dtype=np.float64,
    )
    flat_power = power.reshape(len(weight), len(temporal_frequency), -1)
    for index in range(len(spatial_frequency)):
        select = radial_bin == index
        if np.any(select):
            joint[:, :, index] = flat_power[:, :, select].sum(axis=2)
    joint /= np.maximum(joint.sum(axis=(1, 2), keepdims=True), EPS)
    temporal = joint.sum(axis=2)
    spatial = joint.sum(axis=1)
    median_tf = temporal_frequency[
        np.argmax(np.cumsum(temporal, axis=1) >= 0.5, axis=1)
    ]
    median_sf = spatial_frequency[
        np.argmax(np.cumsum(spatial, axis=1) >= 0.5, axis=1)
    ]
    return {
        "temporal_frequency_hz": temporal_frequency,
        "spatial_frequency_cpd": spatial_frequency,
        "joint_power": joint,
        "temporal_power": temporal,
        "spatial_power": spatial,
        "median_tf_hz": median_tf,
        "median_sf_cpd": median_sf,
    }


def temporal_filter_drive_from_moments(
    activation_sum: torch.Tensor,
    activation_square_sum: torch.Tensor,
    n_samples: int,
) -> np.ndarray:
    """Return per-channel temporal variance averaged over spatial positions."""
    if int(n_samples) < 2:
        raise ValueError("at least two samples are required")
    if activation_sum.shape != activation_square_sum.shape or activation_sum.ndim != 3:
        raise ValueError("activation moments must share [channel, y, x] shape")
    variance = (
        activation_square_sum / int(n_samples)
        - (activation_sum / int(n_samples)).square()
    ).clamp_min(0)
    return variance.mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def score_core(
    core,
    stimulus: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return information components and exact temporal-filter drive.

    Filter drive is the temporal variance of the signed temporal-convolution
    output at each channel and spatial location, averaged over space.  It is an
    exact projection through the learned spatiotemporal filters and therefore
    avoids estimating a short-fixation Fourier spectrum.
    """
    layer_numer, layer_denom = [], []
    pair_numer, pair_denom = [], []
    pre_sum = None
    pre_square_sum = None
    n_pre = 0
    for start in range(0, len(stimulus), int(batch_size)):
        x = stimulus[start : start + int(batch_size)].to(device)
        pre = core.temporal_conv(x)
        if pre.shape[2] != 1:
            raise RuntimeError("sampled histories must collapse to one temporal output")
        signed = pre.squeeze(2).detach().double().cpu()
        batch_sum = signed.sum(dim=0)
        batch_square_sum = signed.square().sum(dim=0)
        pre_sum = batch_sum if pre_sum is None else pre_sum + batch_sum
        pre_square_sum = (
            batch_square_sum
            if pre_square_sum is None
            else pre_square_sum + batch_square_sum
        )
        n_pre += len(signed)
        stem = core.temporal_nonlinearity(pre.squeeze(2))
        stage1 = core.stage1_nonlinearity(core.stage1_conv(stem))
        stage2 = core.stage2_nonlinearity(core.stage2_conv(core._downsample(stage1)))
        stage3 = core.stage3_nonlinearity(core.stage3_conv(core._downsample(stage2)))
        item_numer, item_denom = [], []
        for activation in (stem, stage1, stage2, stage3):
            numerator, denominator = spatial_information_components(activation)
            item_numer.append(np.sum(numerator, axis=1))
            item_denom.append(np.sum(denominator, axis=1))
        layer_numer.append(np.stack(item_numer, axis=1))
        layer_denom.append(np.stack(item_denom, axis=1))
        numerator, denominator = spatial_information_components(stem)
        if numerator.shape[1] % 2:
            raise RuntimeError("sign-split temporal stem must have channel pairs")
        pair_numer.append(numerator.reshape(len(numerator), -1, 2).sum(axis=2))
        pair_denom.append(denominator.reshape(len(denominator), -1, 2).sum(axis=2))
    if n_pre < 2 or pre_sum is None or pre_square_sum is None:
        raise RuntimeError("at least two causal endpoints are required for filter drive")
    filter_drive = temporal_filter_drive_from_moments(
        pre_sum, pre_square_sum, n_pre
    )
    return (
        np.concatenate(layer_numer),
        np.concatenate(layer_denom),
        np.concatenate(pair_numer),
        np.concatenate(pair_denom),
        filter_drive,
    )


def output_information_components(
    ssi: np.ndarray,
    expected: np.ndarray,
    image_rows: np.ndarray,
    trace_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_ssi = ssi[np.ix_(image_rows, trace_rows, np.arange(ssi.shape[2]), np.arange(ssi.shape[3]))]
    selected_expected = expected[
        np.ix_(image_rows, trace_rows, np.arange(expected.shape[2]), np.arange(expected.shape[3]))
    ]
    return (
        np.sum(selected_ssi * selected_expected, axis=-1),
        np.sum(selected_expected, axis=-1),
    )


def require_matching_controlled_model(
    manifest_path: Path,
    checkpoint: Path,
    dataset_configs: Path,
) -> dict:
    """Abort before scoring if the activation core differs from SSI's model."""
    from paper.fig4.upstream.real_trace_matrix.core import sha256_file

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    model = ((manifest.get("model") or {}).get("model") or {})
    expected_checkpoint = str(model.get("checkpoint_sha256", ""))
    expected_dataset = str(model.get("dataset_configs_sha256", ""))
    observed_checkpoint = sha256_file(Path(checkpoint))
    observed_dataset = sha256_file(Path(dataset_configs))
    if observed_checkpoint != expected_checkpoint:
        raise RuntimeError(
            "Controlled SSI and activation audit checkpoints differ: "
            f"expected {expected_checkpoint}, observed {observed_checkpoint}"
        )
    if observed_dataset != expected_dataset:
        raise RuntimeError(
            "Controlled SSI and activation audit dataset configs differ: "
            f"expected {expected_dataset}, observed {observed_dataset}"
        )
    return manifest


def render(
    scales: np.ndarray,
    layer_percent: np.ndarray,
    output_percent: np.ndarray,
    temporal_frequency: np.ndarray,
    spatial_frequency: np.ndarray,
    joint_spectra: np.ndarray,
    median_tf: np.ndarray,
    median_sf: np.ndarray,
    filter_drive_relative: np.ndarray,
    measured_filter_drive_fraction: np.ndarray,
    out_path: Path,
    model_label: str,
    n_image_trace_pairs: int,
) -> dict:
    figure, axes = plt.subplots(1, 4, figsize=(17.4, 4.2), constrained_layout=True)
    axes[0].plot(
        scales,
        filter_drive_relative,
        "o-",
        color="#2F78B7",
        lw=2.1,
        ms=4.5,
    )
    axes[0].axhline(1, color="0.55", lw=0.8, linestyle=":")
    axes[0].axvline(1, color="0.55", lw=0.8, linestyle=":")
    axes[0].set(
        xlabel="retinal trajectory amplitude (× measured)",
        ylabel="temporal-filter drive (relative to 1×)",
        title="A  Motion drives learned temporal filters",
        xticks=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    )
    for index, name in enumerate(LAYER_NAMES):
        axes[1].plot(scales, layer_percent[:, index], "o-", lw=1.8, ms=4, label=name)
    axes[1].plot(scales, output_percent, "s-", color="black", lw=2.2, ms=4.5, label="RR100 output")
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].axvline(1, color="0.55", linestyle=":")
    axes[1].set(
        xlabel="retinal trajectory amplitude (× measured)",
        ylabel="spatial information change from 0× (%)",
        title="B  Nonlinear stages create selective activity",
        xticks=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    )
    axes[1].legend(frameon=False, fontsize=7.5)

    population_joint = joint_spectra.sum(axis=0)
    population_joint /= max(float(population_joint.max()), EPS)
    log_joint = np.log10(np.maximum(population_joint, 1e-5))
    image = axes[2].contourf(
        spatial_frequency,
        temporal_frequency,
        log_joint,
        levels=np.linspace(-5.0, 0.0, 16),
        cmap="magma",
        extend="min",
    )
    axes[2].set(
        xlim=(0, min(18.0, float(spatial_frequency[-1]))),
        ylim=(0, 120),
        xlabel="spatial frequency (cycles/deg)",
        ylabel="temporal frequency (Hz)",
        title="C  Learned joint SF/TF passbands",
    )
    figure.colorbar(
        image,
        ax=axes[2],
        fraction=0.045,
        label="log10 normalized filter power",
    )

    marker_size = 25.0 + 1150.0 * measured_filter_drive_fraction
    scatter = axes[3].scatter(
        median_sf,
        median_tf,
        s=marker_size,
        c=measured_filter_drive_fraction,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.6,
    )
    axes[3].set(
        xlim=(0, max(10.0, min(18.0, float(np.max(median_sf)) + 1.5))),
        ylim=(0, max(30.0, float(np.max(median_tf)) + 3.0)),
        xlabel="stem-filter median SF (cycles/deg)",
        ylabel="stem-filter median TF (Hz)",
        title="D  Exact motion engagement of passbands",
    )
    # Label only the filters carrying the upper half of measured-motion drive.
    # This criterion is data-driven and keeps nearby low-drive channels from
    # turning the passband map into an unreadable label cloud.
    labeled_channels = set(
        np.argsort(measured_filter_drive_fraction)[
            -max(1, len(measured_filter_drive_fraction) // 2) :
        ].tolist()
    )
    for channel, (x, y) in enumerate(zip(median_sf, median_tf)):
        if channel not in labeled_channels:
            continue
        axes[3].annotate(
            f"f{channel}",
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[3].text(
        0.98,
        0.02,
        "area/color = exact 1× drive; labels = upper half",
        transform=axes[3].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    figure.colorbar(
        scatter,
        ax=axes[3],
        fraction=0.045,
        label="fraction of 1× filter drive",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"{model_label}: tracing native retinal motion through the feed-forward visual core\n"
        f"{int(n_image_trace_pairs)} controlled image–trace pairs",
        fontsize=13,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=220, facecolor="white")
    figure.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    tf_correlation = spearmanr(median_tf, measured_filter_drive_fraction).statistic
    sf_correlation = spearmanr(median_sf, measured_filter_drive_fraction).statistic
    return {
        "stem_median_tf_vs_filter_drive_fraction_spearman": float(tf_correlation),
        "stem_median_sf_vs_filter_drive_fraction_spearman": float(sf_correlation),
        "fraction_filter_drive_in_channels_above_10hz": float(
            np.sum(measured_filter_drive_fraction[median_tf >= 10.0])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-configs", type=Path, required=True)
    parser.add_argument("--controlled-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--n-traces", type=int, default=4)
    parser.add_argument("--n-timepoints", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-label", default="native-240 twin")
    args = parser.parse_args()

    from paper.fig4.upstream.real_trace_matrix.core import extract_patch
    from paper.fig4.upstream.real_trace_matrix.model import (
        PPD,
        _standardize_uint_like,
        load_pinned_multidataset_model,
    )

    response_path = args.controlled_dir / "controlled_scaling_response.npz"
    controlled_manifest = require_matching_controlled_model(
        args.controlled_dir / "manifest.json",
        args.checkpoint.resolve(),
        args.dataset_configs.resolve(),
    )
    with np.load(response_path) as archive:
        ssi = np.asarray(archive["ssi"], dtype=np.float64)
        expected = np.asarray(archive["expected_spikes"], dtype=np.float64)
        scales = np.asarray(archive["scale_factors"], dtype=np.float64)
        trace_xy = np.asarray(archive["base_trace_xy"], dtype=np.float32)
        archive_source_rate_hz = (
            int(round(float(archive["base_trace_source_rate_hz"])))
            if "base_trace_source_rate_hz" in archive.files
            else None
        )
    images = pd.read_csv(args.controlled_dir / "selected_images.csv")
    if len(images) != ssi.shape[0] or len(trace_xy) != ssi.shape[1]:
        raise ValueError("controlled response and selection tables disagree")
    image_rows = np.unique(np.round(np.linspace(0, len(images) - 1, args.n_images)).astype(int))
    trace_rows = np.unique(np.round(np.linspace(0, len(trace_xy) - 1, args.n_traces)).astype(int))

    model, model_info = load_pinned_multidataset_model(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_configs.resolve(),
        device=args.device,
        strict=True,
    )
    core = model.model.convnet.eval()
    n_lags = int(core.temporal_support)
    time_contract = dict(controlled_manifest.get("time_contract") or {})
    source_rate_hz = int(
        archive_source_rate_hz
        if archive_source_rate_hz is not None
        else time_contract.get("source_trace_rate_hz", 0)
    )
    output_rate_hz = int(time_contract.get("model_output_rate_hz", 0))
    if source_rate_hz < 1 or output_rate_hz < 1:
        raise ValueError("Controlled response is missing its source/output trace rates")
    if output_rate_hz != 240:
        raise ValueError(
            f"Native core audit requires a 240-Hz output grid, got {output_rate_hz} Hz."
        )
    from paper.fig4.upstream.real_trace_matrix.model import _trace_on_output_grid

    trace_xy_output = np.stack(
        [
            _trace_on_output_grid(
                trace,
                source_rate_hz=source_rate_hz,
                output_rate_hz=output_rate_hz,
                torch=torch,
            )
            for trace in trace_xy
        ],
        axis=0,
    )
    endpoints = np.unique(
        np.round(
            np.linspace(0, trace_xy_output.shape[1] - 1, args.n_timepoints)
        ).astype(int)
    )
    out_size = (151, 151)
    shape = (len(image_rows), len(trace_rows), len(scales))
    layer_numer = np.zeros((*shape, len(LAYER_NAMES)), dtype=np.float64)
    layer_denom = np.zeros_like(layer_numer)
    pair_numer = np.zeros((*shape, core.temporal_channels), dtype=np.float64)
    pair_denom = np.zeros_like(pair_numer)
    preactivation_filter_drive = np.zeros_like(pair_numer)
    canvas_cache = {}
    for image_position, image_row in enumerate(image_rows):
        patch, _ = extract_patch(images.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=540)
        standardized = _standardize_uint_like(patch)
        for trace_position, trace_row in enumerate(trace_rows):
            base_trace = trace_xy_output[int(trace_row)]
            for scale_position, scale in enumerate(scales):
                stimulus = selected_causal_histories(
                    standardized,
                    base_trace * float(scale),
                    endpoints,
                    n_lags=n_lags,
                    out_size=out_size,
                    ppd=PPD,
                )
                stimulus = (stimulus - 127.0) / 255.0
                ln, ld, pn, pair_den, filter_drive = score_core(
                    core,
                    stimulus,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                layer_numer[image_position, trace_position, scale_position] = ln.sum(axis=0)
                layer_denom[image_position, trace_position, scale_position] = ld.sum(axis=0)
                pair_numer[image_position, trace_position, scale_position] = pn.sum(axis=0)
                pair_denom[image_position, trace_position, scale_position] = pair_den.sum(axis=0)
                preactivation_filter_drive[
                    image_position, trace_position, scale_position
                ] = filter_drive
                del stimulus
        print(f"core motion path image {image_position + 1}/{len(image_rows)}", flush=True)

    layer_percent = np.column_stack([
        percent_curve(layer_numer[..., index], layer_denom[..., index])
        for index in range(len(LAYER_NAMES))
    ])
    output_numer, output_denom = output_information_components(ssi, expected, image_rows, trace_rows)
    output_percent = percent_curve(output_numer, output_denom)
    stem_percent = np.column_stack([
        percent_curve(pair_numer[..., index], pair_denom[..., index])
        for index in range(core.temporal_channels)
    ])
    filter_spectra = filter_joint_spectra(core, ppd=PPD)
    frequency = filter_spectra["temporal_frequency_hz"]
    spatial_frequency = filter_spectra["spatial_frequency_cpd"]
    joint_spectra = filter_spectra["joint_power"]
    spectra = filter_spectra["temporal_power"]
    spatial_spectra = filter_spectra["spatial_power"]
    median_tf = filter_spectra["median_tf_hz"]
    median_sf = filter_spectra["median_sf_cpd"]
    measured_index = int(np.flatnonzero(np.isclose(scales, 1.0))[0])
    pooled_filter_drive = preactivation_filter_drive.sum(axis=(0, 1, 3))
    filter_drive_relative = pooled_filter_drive / max(
        float(pooled_filter_drive[measured_index]), EPS
    )
    measured_filter_drive = preactivation_filter_drive[:, :, measured_index].sum(axis=(0, 1))
    measured_filter_drive_fraction = measured_filter_drive / max(
        float(measured_filter_drive.sum()), EPS
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "native_core_motion_path.npz",
        scales=scales,
        layer_numerator=layer_numer,
        layer_denominator=layer_denom,
        layer_percent_vs_zero=layer_percent,
        stem_pair_numerator=pair_numer,
        stem_pair_denominator=pair_denom,
        stem_pair_percent_vs_zero=stem_percent,
        preactivation_filter_drive=preactivation_filter_drive,
        preactivation_filter_drive_relative_to_measured=filter_drive_relative,
        measured_filter_drive_fraction_by_channel=measured_filter_drive_fraction,
        output_percent_vs_zero=output_percent,
        temporal_frequency_hz=frequency,
        temporal_filter_power=spectra,
        temporal_filter_median_hz=median_tf,
        spatial_frequency_cpd=spatial_frequency,
        spatial_filter_power=spatial_spectra,
        spatial_filter_median_cpd=median_sf,
        joint_sf_tf_filter_power=joint_spectra,
        selected_image_rows=image_rows,
        selected_trace_rows=trace_rows,
        selected_timepoints=endpoints,
        source_trace_rate_hz=np.asarray(source_rate_hz),
        model_output_rate_hz=np.asarray(output_rate_hz),
    )
    figure_path = args.out_dir / "native_core_motion_path.png"
    statistics = render(
        scales,
        layer_percent,
        output_percent,
        frequency,
        spatial_frequency,
        joint_spectra,
        median_tf,
        median_sf,
        filter_drive_relative,
        measured_filter_drive_fraction,
        figure_path,
        args.model_label,
        int(len(image_rows) * len(trace_rows)),
    )
    full_controlled_bank = (
        len(image_rows) == len(images) and len(trace_rows) == len(trace_xy)
    )
    report = {
        "analysis": "activation-level native-240 retinal-motion path audit",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_configs": str(args.dataset_configs.resolve()),
        "controlled_response": str(response_path.resolve()),
        "controlled_manifest_analysis": controlled_manifest.get("analysis"),
        "model_info": model_info,
        "n_images": int(len(image_rows)),
        "n_traces": int(len(trace_rows)),
        "n_scored_timepoints": int(len(endpoints)),
        "source_trace_rate_hz": source_rate_hz,
        "source_trace_samples": int(trace_xy.shape[1]),
        "model_output_rate_hz": output_rate_hz,
        "model_output_trace_samples": int(trace_xy_output.shape[1]),
        "history_frames": n_lags,
        "history_rate_hz": output_rate_hz,
        "estimator": "expected-activation-weighted spatial information after exact nonlinear core stages",
        "filter_drive_estimator": (
            "temporal variance across sampled endpoints of the signed learned "
            "temporal-convolution output, averaged over space; exact convolution, "
            "no short-window spectral estimate"
        ),
        "preactivation_filter_drive_relative_to_measured": {
            str(float(scale)): float(value)
            for scale, value in zip(scales, filter_drive_relative)
        },
        "layer_percent_vs_stabilized_at_measured_motion": {
            name: float(layer_percent[measured_index, index])
            for index, name in enumerate(LAYER_NAMES)
        },
        "rr100_output_percent_vs_stabilized_at_measured_motion": float(output_percent[measured_index]),
        **statistics,
        "sample_scope": (
            "complete controlled bank"
            if full_controlled_bank
            else "evenly spaced subset of the controlled bank"
        ),
        "claim_boundary": (
            f"{len(image_rows) * len(trace_rows)} controlled image-trace pairs; "
            "a production-scale natural-image/eye-trace bank would still be larger"
        ),
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "native_core_motion_path_summary.json").write_text(
        json.dumps(json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_ready(report), indent=2))


if __name__ == "__main__":
    main()
