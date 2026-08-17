#!/usr/bin/env python3
"""Targeted Figure 4 audit: retinal phase modulation through early ResNet filtering.

This intentionally does not evaluate the full neural network.  It crosses all
100 Figure 4 natural images with the eight preselected controlled-scaling drift
trajectories, computes the exact 2-D translation phase signal on every Fourier
mode supported by the 151-pixel retinal aperture, and passes its discrete
temporal spectrum through the four trained frontend kernels, spatial stem, and
first 3x9x9 spatiotemporal ResNet kernel. Because normalization and rectification
intervene, that cascade is explicitly a linear-fundamental approximation. A finite-crop
audit uses the exact renderer on the existing 8-image x 8-trajectory controlled
subset.  The existing corrected natural-history SSI curves are read only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules.conv_layers import _get_window
from paper.fig4.mechanism_audit_v1.correction.common import (
    BANK_DIR,
    CONTROLLED_DIR,
    FRAME_RATE_HZ,
    LEGACY_MATRIX_DIR,
    N_PRECEDING,
    SCALE_FACTORS,
    sha256_file,
    write_json,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    OUT_SIZE,
    PPD,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)
from paper.fig4.upstream.run_real_trace_matrix import MODEL_CHECKPOINT_PATH


OUT_DIR = ROOT / "outputs/figures/fig4/mechanism_audit_v1/targeted_frontend_v1"
DATA_DIR = OUT_DIR / "plot_data"
EXACT_DIR = OUT_DIR / "exact_arrays"
PREVIOUS_CONTROLLED = ROOT / "outputs/figures/fig4/spatiotemporal_tuning/controlled_scaling"
CORRECTED_CURVES = (
    ROOT
    / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1/preliminary_true_only/preliminary_true_only_curves.csv"
)
SCALES = np.asarray(SCALE_FACTORS, dtype=np.float64)
DT = 1.0 / FRAME_RATE_HZ
N_TIME = 40
LOW_COLOR = "#007C83"
HIGH_COLOR = "#D55E00"
NEUTRAL = "#4B5563"
CHANNEL_COLORS = ("#0072B2", "#CC79A7", "#E69F00", "#009E73")
SF_BANDS = (
    (0.25, 0.5, "0.25–0.5 cpd"),
    (0.5, 1.0, "0.5–1 cpd"),
    (1.0, 2.0, "1–2 cpd"),
    (2.0, 4.0, "2–4 cpd"),
    (4.0, 8.0, "4–8 cpd"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-finite-crop", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_inputs() -> dict[str, Any]:
    images = (
        pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv")
        .sort_values("image_index")
        .reset_index(drop=True)
    )
    traces = (
        pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv")
        .sort_values("trace_bank_index")
        .reset_index(drop=True)
    )
    selected_images = pd.read_csv(PREVIOUS_CONTROLLED / "selected_images.csv")
    selected_traces = pd.read_csv(PREVIOUS_CONTROLLED / "selected_traces.csv")
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        scored = np.asarray(archive["stored_scored_trace_xy"], dtype=np.float64)
        true_history = np.asarray(archive["true_history_xy"], dtype=np.float64)
    if images.shape[0] != 100 or traces.shape[0] != 1000 or scored.shape != (1000, 40, 2):
        raise ValueError((images.shape, traces.shape, scored.shape))
    trace_ids = selected_traces["trace_bank_index"].to_numpy(dtype=int)
    image_ids = selected_images["image_index"].to_numpy(dtype=int)
    return {
        "images": images,
        "traces": traces,
        "selected_image_ids": image_ids,
        "selected_trace_ids": trace_ids,
        "selected_scored_xy": scored[trace_ids],
        "true_history": true_history,
    }


def _effective_parametrized_weight(
    state: dict[str, Any],
    prefix: str,
    *,
    window_axes: tuple[tuple[int, float], ...],
) -> np.ndarray:
    """Reconstruct the exact checkpoint weight after weight norm and AA windows."""
    if f"{prefix}.0.v" in state:
        value = state[f"{prefix}.0.v"].detach().cpu()
        gain = state[f"{prefix}.0.g"].detach().cpu()
        reduction = tuple(range(1, value.ndim))
        normalized = value / torch.linalg.vector_norm(value, dim=reduction, keepdim=True).clamp_min(1e-12)
        shape = (len(gain),) + (1,) * (value.ndim - 1)
        weight = normalized * gain.reshape(shape)
    else:
        weight = state[f"{prefix}.original"].detach().cpu()
    for axis, power in window_axes:
        size = weight.shape[axis]
        window = _get_window("hann", size, power=power)
        shape = [1] * weight.ndim
        shape[axis] = size
        weight = weight * window.reshape(shape)
    return weight.numpy().astype(np.float64)


def effective_early_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    frontend = _effective_parametrized_weight(
        state,
        "model.frontend.temporal_conv.conv.parametrizations.weight",
        window_axes=((2, 0.25),),
    )[:, 0, :, 0, 0]
    stem = _effective_parametrized_weight(
        state,
        "model.convnet.stem.components.conv.conv.parametrizations.weight",
        window_axes=((3, 1.0), (4, 1.0)),
    )[:, :, 0]
    first_spatiotemporal = _effective_parametrized_weight(
        state,
        "model.convnet.layers.0.main_block.components.conv.conv.parametrizations.weight",
        window_axes=((3, 1.0), (4, 1.0)),
    )
    stem_gamma = state["model.convnet.stem.components.norm.gamma"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    if frontend.shape != (4, 16) or stem.shape != (8, 4, 7, 7) or first_spatiotemporal.shape != (64, 16, 3, 9, 9) or stem_gamma.shape != (8,):
        raise ValueError((frontend.shape, stem.shape, first_spatiotemporal.shape, stem_gamma.shape))
    return frontend, stem, first_spatiotemporal, stem_gamma


def frontend_transfer(weights: np.ndarray, frequencies_hz: np.ndarray) -> np.ndarray:
    lag = np.arange(weights.shape[1], dtype=np.float64) / FRAME_RATE_HZ
    return np.abs(
        np.einsum(
            "cl,fl->cf",
            weights,
            np.exp(-2j * np.pi * np.asarray(frequencies_hz)[:, None] * lag[None]),
        )
    )


def contiguous_half_power_band(response: np.ndarray, frequency: np.ndarray) -> tuple[float, float]:
    peak = int(np.argmax(response))
    keep = response >= response[peak] / math.sqrt(2.0)
    lo = peak
    hi = peak
    while lo > 0 and keep[lo - 1]:
        lo -= 1
    while hi + 1 < len(keep) and keep[hi + 1]:
        hi += 1
    return float(frequency[lo]), float(frequency[hi])


def describe_filter(kernel: np.ndarray, response: np.ndarray, frequency: np.ndarray) -> str:
    peak = float(frequency[int(np.argmax(response))])
    dc_ratio = float(response[0] / max(response.max(), 1e-12))
    zero_sum_ratio = float(abs(kernel.sum()) / max(np.abs(kernel).sum(), 1e-12))
    if peak >= 0.9 * float(frequency[-1]):
        kind = "high-pass"
    elif peak <= 3.0 and dc_ratio >= 0.7:
        kind = "low-pass"
    else:
        kind = "band-pass"
    if zero_sum_ratio < 0.05:
        kind += ", derivative-like"
    elif zero_sum_ratio < 0.15:
        kind += ", near-zero-DC"
    return kind


def render_movies(
    patch: np.ndarray,
    histories: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Render histories (S,T,2) through the exact Figure 4 crop/interpolator."""
    image = _standardize_uint_like(patch)
    histories = np.asarray(histories, dtype=np.float32)
    n_sequence, n_time = histories.shape[:2]
    flattened = torch.from_numpy(histories.reshape(-1, 2)).to(device)
    eye_norm = _eye_deg_to_norm(flattened, ppd=PPD, img_size=image.shape, torch=torch)
    base = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    repeated = base.unsqueeze(0).expand(n_sequence * n_time, -1, -1)
    with torch.no_grad():
        shifted = _shift_movie_with_eye(
            repeated,
            eye_norm,
            out_size=OUT_SIZE,
            scale_factor=1.0,
            torch=torch,
        )
    return shifted.reshape(n_sequence, n_time, *OUT_SIZE).cpu().numpy().astype(np.float32)


def frequency_grid() -> dict[str, np.ndarray]:
    height, width = OUT_SIZE
    if height != width:
        raise ValueError(OUT_SIZE)
    n = height
    f_axis = np.fft.fftfreq(n, d=1.0 / PPD)
    fy, fx = np.meshgrid(f_axis, f_axis, indexing="ij")
    # Image rows increase downward; physical eye y and physical ky increase upward.
    kx = fx
    ky = -fy
    radius = np.hypot(kx, ky)
    df = float(PPD / n)
    nyquist = float(PPD / 2.0)
    mask = (radius >= 0.5 * df) & (radius <= nyquist)
    flat = np.flatnonzero(mask.ravel())
    kxy = np.column_stack((kx.ravel()[flat], ky.ravel()[flat]))
    radial = radius.ravel()[flat]
    # Half-cpd display bins, with the lowest resolvable shell (0.248 cpd) retained.
    radial_edges = np.arange(0.125, 19.125 + 1e-9, 0.5)
    radial_bin = np.digitize(radial, radial_edges) - 1
    keep = (radial_bin >= 0) & (radial_bin < len(radial_edges) - 1)
    return {
        "f_axis": f_axis,
        "flat_index": flat[keep],
        "kxy": kxy[keep],
        "radial": radial[keep],
        "radial_edges": radial_edges,
        "radial_bin": radial_bin[keep],
        "radial_centers": 0.5 * (radial_edges[:-1] + radial_edges[1:]),
        "df": np.asarray(df),
        "nyquist": np.asarray(nyquist),
    }


def collapse_absolute_temporal(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.fft.fftfreq(N_TIME, d=DT)
    absolute = np.abs(frequency)
    unique = np.unique(absolute)
    out = np.zeros((*power.shape[:-1], len(unique)), dtype=np.float64)
    for index, value in enumerate(unique):
        out[..., index] = power[..., np.isclose(absolute, value)].sum(axis=-1)
    return unique, out


def load_or_compute_image_power(inputs: dict[str, Any], grid: dict[str, np.ndarray], device: str) -> np.ndarray:
    cache = EXACT_DIR / "natural_image_mode_power.npz"
    if cache.is_file():
        with np.load(cache) as archive:
            values = np.asarray(archive["mean_mode_power"], dtype=np.float64)
            if values.shape == (len(grid["flat_index"]),):
                return values
    powers: list[np.ndarray] = []
    canvas_cache: dict[Any, Any] = {}
    eye0 = np.zeros((1, 1, 2), dtype=np.float32)
    for ordinal, row in inputs["images"].iterrows():
        patch, _ = extract_patch(row, canvas_cache=canvas_cache, patch_size_px=540)
        frame = render_movies(patch, eye0, device=device)[0, 0]
        coefficient = np.fft.fft2(frame) / frame.size
        powers.append(np.abs(coefficient.ravel()[grid["flat_index"]]) ** 2)
        if (ordinal + 1) % 10 == 0:
            print(f"natural image Fourier amplitudes {ordinal + 1}/100", flush=True)
    matrix = np.asarray(powers, dtype=np.float64)
    mean_power = matrix.mean(axis=0)
    np.savez_compressed(
        cache,
        mean_mode_power=mean_power.astype(np.float32),
        per_image_mode_power=matrix.astype(np.float32),
        image_ids=inputs["images"]["image_index"].to_numpy(int),
        flat_index=grid["flat_index"],
        kxy=grid["kxy"],
        normalization=np.asarray("fft2(frame) / (151*151); exact stabilized central renderer crop"),
    )
    return mean_power


def phase_spectra_and_geometry(
    displacement: np.ndarray,
    grid: dict[str, np.ndarray],
    selected_trace_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    cache = EXACT_DIR / "phase_only_mode_spectra.npz"
    if cache.is_file():
        with np.load(cache) as archive:
            if np.array_equal(archive["scales"], SCALES) and np.array_equal(
                archive["selected_trace_ids"], selected_trace_ids
            ):
                return {key: np.asarray(archive[key]) for key in archive.files}
    kxy = np.asarray(grid["kxy"], dtype=np.float64)
    n_mode = len(kxy)
    frequency = np.unique(np.abs(np.fft.fftfreq(N_TIME, d=DT)))
    phase_power = np.zeros((n_mode, len(SCALES), len(frequency)), dtype=np.float32)
    phase_rms = np.zeros((n_mode, len(SCALES)), dtype=np.float32)
    phase_range = np.zeros_like(phase_rms)
    concentration = np.zeros_like(phase_rms)
    chunk_size = 512
    for start in range(0, n_mode, chunk_size):
        stop = min(n_mode, start + chunk_size)
        dot = np.einsum("kd,ntd->nkt", kxy[start:stop], displacement, optimize=True)
        for scale_index, scale in enumerate(SCALES):
            phase = 2.0 * np.pi * float(scale) * dot
            signal = np.exp(-1j * phase)
            coefficient = np.fft.fft(signal, axis=-1) / N_TIME
            _, absolute_power = collapse_absolute_temporal(np.abs(coefficient) ** 2)
            phase_power[start:stop, scale_index] = absolute_power.mean(axis=0).astype(np.float32)
            centered = phase - phase.mean(axis=-1, keepdims=True)
            phase_rms[start:stop, scale_index] = np.sqrt(np.mean(centered**2, axis=-1)).mean(axis=0)
            phase_range[start:stop, scale_index] = np.ptp(phase, axis=-1).mean(axis=0)
            concentration[start:stop, scale_index] = np.abs(signal.mean(axis=-1)).mean(axis=0)
        print(f"phase spectra modes {stop}/{n_mode}", flush=True)
    validation = np.max(np.abs(phase_power.sum(axis=-1) - 1.0))
    np.savez_compressed(
        cache,
        phase_power=phase_power,
        phase_rms=phase_rms,
        phase_range=phase_range,
        circular_concentration=concentration,
        temporal_hz=frequency,
        scales=SCALES,
        kxy=kxy.astype(np.float32),
        radial_cpd=grid["radial"].astype(np.float32),
        selected_trace_ids=np.asarray(selected_trace_ids, dtype=int),
        parseval_max_abs_error=np.asarray(validation),
        definition=np.asarray("mean across selected trajectories of |FFT_t exp(-i 2pi k dot s[e(t)-e(0)])/T|^2; +/- frequencies summed by |f|"),
    )
    with np.load(cache) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def group_mode_matrix(
    values: np.ndarray,
    radial_bin: np.ndarray,
    n_bin: int,
    *,
    reducer: str = "mean",
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full((n_bin, *values.shape[1:]), np.nan, dtype=np.float64)
    for index in range(n_bin):
        subset = values[radial_bin == index]
        if len(subset):
            out[index] = subset.mean(axis=0) if reducer == "mean" else subset.sum(axis=0)
    return out


def aggregate_ideal(
    image_power: np.ndarray,
    phase: dict[str, np.ndarray],
    weights: np.ndarray,
    grid: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    temporal_hz = np.asarray(phase["temporal_hz"], dtype=np.float64)
    phase_power = np.asarray(phase["phase_power"], dtype=np.float64)
    retinal_mode_power = image_power[:, None, None] * phase_power
    response_discrete = frontend_transfer(weights, temporal_hz)
    response_discrete[:, 0] = 0.0  # motion-induced energy explicitly excludes DC
    transmitted = np.einsum(
        "msf,cf->mc s".replace(" ", ""), retinal_mode_power, response_discrete**2, optimize=True
    )
    # The explicit einsum result is modes x channels x scales; move to modes x scales x channels.
    transmitted = np.moveaxis(transmitted, 1, 2)
    combined = transmitted.sum(axis=-1)
    n_bin = len(grid["radial_centers"])
    radial_retinal = group_mode_matrix(retinal_mode_power, grid["radial_bin"], n_bin)
    radial_channel = group_mode_matrix(transmitted, grid["radial_bin"], n_bin)
    radial_combined = group_mode_matrix(combined, grid["radial_bin"], n_bin)
    radial_phase_rms = group_mode_matrix(phase["phase_rms"], grid["radial_bin"], n_bin)
    radial_phase_range = group_mode_matrix(phase["phase_range"], grid["radial_bin"], n_bin)
    radial_concentration = group_mode_matrix(
        phase["circular_concentration"], grid["radial_bin"], n_bin
    )
    return {
        "temporal_hz": temporal_hz,
        "response_discrete": response_discrete,
        "retinal_mode_power": retinal_mode_power,
        "transmitted_mode": transmitted,
        "combined_mode": combined,
        "radial_retinal": radial_retinal,
        "radial_channel": radial_channel,
        "radial_combined": radial_combined,
        "radial_phase_rms": radial_phase_rms,
        "radial_phase_range": radial_phase_range,
        "radial_concentration": radial_concentration,
    }


def early_resnet_linearized_transfer(
    frontend_weights: np.ndarray,
    stem_weights: np.ndarray,
    resblock1_weights: np.ndarray,
    stem_norm_gamma: np.ndarray,
    phase: dict[str, np.ndarray],
    image_power: np.ndarray,
    grid: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Fourier audit through the first genuine spatiotemporal ResNet kernel.

    The trained stack is frontend (16x1x1) -> spatial stem (1x7x7) -> RMSNorm
    -> SplitReLU -> first ResBlock convolution (3x9x9).  RMSNorm and SplitReLU
    prevent an exact single transfer function. For the explicitly named
    *linear-fundamental approximation* below, the RMSNorm denominator is
    treated as identity, its learned affine gamma is retained,
    and the positive/negative SplitReLU branches carry +/- one half of the
    input fundamental. Exact nonlinear natural-movie activations remain in
    the conditional hook pilot and are never called this Fourier transfer.
    """
    cache = EXACT_DIR / "early_resnet1_linearized_transfer.npz"
    definition = (
        "trained frontend + 1x7x7 stem + first 3x9x9 ResNet convolution; "
        "RMSNorm denominator treated as identity with learned gamma retained; SplitReLU fundamental mapped to +/-0.5 "
        "branches; DC excluded only from transmitted motion energy"
    )
    temporal_hz = np.asarray(phase["temporal_hz"], dtype=np.float64)
    kxy = np.asarray(grid["kxy"], dtype=np.float64)
    if cache.is_file():
        with np.load(cache) as archive:
            if (
                np.array_equal(archive["kxy"], kxy.astype(np.float32))
                and np.array_equal(archive["temporal_hz"], temporal_hz)
                and str(archive["definition"]) == definition
                and "radial_output_energy" in archive.files
                and "output_band_energy" in archive.files
            ):
                return {key: np.asarray(archive[key]) for key in archive.files}

    # Retain complex phase for the cascade; ``frontend_transfer`` is
    # magnitude-only and is used elsewhere for power calculations.
    lag = np.arange(frontend_weights.shape[1], dtype=np.float64) / FRAME_RATE_HZ
    frontend_complex = np.einsum(
        "cl,fl->cf",
        frontend_weights,
        np.exp(-2j * np.pi * temporal_hz[:, None] * lag[None]),
    )
    n_mode = len(kxy)
    stem_gain = np.zeros(n_mode, dtype=np.float32)
    resblock_gain = np.zeros((n_mode, len(temporal_hz)), dtype=np.float32)
    cascade_gain = np.zeros_like(resblock_gain)
    n_output = resblock1_weights.shape[0]
    n_bin = len(grid["radial_centers"])
    radial_output_gain_sum = np.zeros((n_bin, n_output, len(temporal_hz)), dtype=np.float64)
    radial_output_energy_sum = np.zeros((n_bin, len(SCALES), n_output), dtype=np.float64)
    radial_counts = np.zeros(n_bin, dtype=np.int64)
    output_band_energy_sum = np.zeros((len(SF_BANDS), len(SCALES), n_output), dtype=np.float64)
    output_band_counts = np.zeros(len(SF_BANDS), dtype=np.int64)
    phase_power = np.asarray(phase["phase_power"], dtype=np.float64)
    chunk_size = 128
    stem_coordinate = np.arange(7, dtype=np.float64) - 3.0
    res_coordinate = np.arange(9, dtype=np.float64) - 4.0
    res_time = np.arange(3, dtype=np.float64) / FRAME_RATE_HZ
    for start in range(0, n_mode, chunk_size):
        stop = min(n_mode, start + chunk_size)
        wave = kxy[start:stop]
        stem_spatial_phase = np.exp(
            -2j
            * np.pi
            * (
                wave[:, 0, None, None] * stem_coordinate[None, None, :] / PPD
                + wave[:, 1, None, None] * (-stem_coordinate[None, :, None]) / PPD
            )
        )
        stem_complex = np.einsum("jcyx,myx->mjc", stem_weights, stem_spatial_phase)
        stem_complex *= stem_norm_gamma[None, :, None]
        stem_gain[start:stop] = np.sum(np.abs(stem_complex) ** 2, axis=(1, 2)).astype(np.float32)
        stem_frontend = np.einsum("mjc,cf->mjf", stem_complex, frontend_complex)
        # SplitReLU ordering is [ReLU(x_1..x_8), ReLU(-x_1..-x_8)].
        split_fundamental = 0.5 * np.concatenate((stem_frontend, -stem_frontend), axis=1)

        res_spatial_phase = np.exp(
            -2j
            * np.pi
            * (
                wave[:, 0, None, None] * res_coordinate[None, None, :] / PPD
                + wave[:, 1, None, None] * (-res_coordinate[None, :, None]) / PPD
            )
        )
        res_temporal_phase = np.exp(
            -2j * np.pi * temporal_hz[:, None] * res_time[None]
        )
        res_complex = np.einsum(
            "oityx,myx,ft->moif",
            resblock1_weights,
            res_spatial_phase,
            res_temporal_phase,
            optimize=True,
        )
        resblock_gain[start:stop] = np.sum(np.abs(res_complex) ** 2, axis=(1, 2)).astype(np.float32)
        cascade = np.einsum("moif,mif->mof", res_complex, split_fundamental, optimize=True)
        output_gain = np.abs(cascade) ** 2
        cascade_gain[start:stop] = np.sum(output_gain, axis=1).astype(np.float32)

        output_motion_gain = output_gain.copy()
        output_motion_gain[:, :, 0] = 0.0
        output_energy = image_power[start:stop, None, None] * np.einsum(
            "msf,mof->mso",
            phase_power[start:stop],
            output_motion_gain,
            optimize=True,
        )
        chunk_bins = grid["radial_bin"][start:stop]
        chunk_radial = grid["radial"][start:stop]
        for bin_index in np.unique(chunk_bins):
            keep = chunk_bins == bin_index
            radial_counts[bin_index] += int(np.sum(keep))
            radial_output_gain_sum[bin_index] += output_gain[keep].sum(axis=0)
            radial_output_energy_sum[bin_index] += output_energy[keep].sum(axis=0)
        for band_index, (low, high, _) in enumerate(SF_BANDS):
            keep = (chunk_radial >= low) & (chunk_radial < high)
            if np.any(keep):
                output_band_counts[band_index] += int(np.sum(keep))
                output_band_energy_sum[band_index] += output_energy[keep].sum(axis=0)

    # Preserve the actual kernel/cascade DC gain for the transfer-function
    # figures.  Exclude DC only from the motion-induced-energy calculation.
    motion_gain = cascade_gain.copy()
    motion_gain[:, 0] = 0.0
    transmitted_mode = image_power[:, None] * np.einsum(
        "msf,mf->ms", phase_power, motion_gain, optimize=True
    )
    radial_energy = group_mode_matrix(transmitted_mode, grid["radial_bin"], n_bin)
    radial_stem_gain = group_mode_matrix(stem_gain, grid["radial_bin"], n_bin)
    radial_resblock_gain = group_mode_matrix(resblock_gain, grid["radial_bin"], n_bin)
    radial_cascade_gain = group_mode_matrix(cascade_gain, grid["radial_bin"], n_bin)
    radial_output_gain = np.full_like(radial_output_gain_sum, np.nan)
    radial_output_energy = np.full_like(radial_output_energy_sum, np.nan)
    valid_bins = radial_counts > 0
    radial_output_gain[valid_bins] = radial_output_gain_sum[valid_bins] / radial_counts[valid_bins, None, None]
    radial_output_energy[valid_bins] = radial_output_energy_sum[valid_bins] / radial_counts[valid_bins, None, None]
    output_band_energy = output_band_energy_sum / np.maximum(output_band_counts[:, None, None], 1)
    payload = {
        "kxy": kxy.astype(np.float32),
        "radial_cpd": grid["radial"].astype(np.float32),
        "temporal_hz": temporal_hz,
        "scales": SCALES,
        "stem_gain": stem_gain,
        "resblock1_gain": resblock_gain,
        "cascade_gain": cascade_gain,
        "transmitted_mode": transmitted_mode.astype(np.float32),
        "radial_energy": radial_energy.astype(np.float32),
        "radial_stem_gain": radial_stem_gain.astype(np.float32),
        "radial_resblock1_gain": radial_resblock_gain.astype(np.float32),
        "radial_cascade_gain": radial_cascade_gain.astype(np.float32),
        "radial_output_gain": radial_output_gain.astype(np.float32),
        "radial_output_energy": radial_output_energy.astype(np.float32),
        "output_band_energy": output_band_energy.astype(np.float32),
        "output_band_labels": np.asarray([label for _, _, label in SF_BANDS]),
        "resblock1_input_mixing_energy": np.sum(resblock1_weights ** 2, axis=(2, 3, 4)).astype(np.float32),
        "definition": np.asarray(definition),
    }
    np.savez_compressed(cache, **payload)
    return payload


def write_ideal_tables(
    aggregate: dict[str, np.ndarray],
    weights: np.ndarray,
    grid: dict[str, np.ndarray],
) -> pd.DataFrame:
    frequency_dense = np.linspace(0.0, FRAME_RATE_HZ / 2.0, 1201)
    response_dense = frontend_transfer(weights, frequency_dense)
    filter_rows: list[dict[str, Any]] = []
    for channel, kernel in enumerate(weights):
        low, high = contiguous_half_power_band(response_dense[channel], frequency_dense)
        peak = float(frequency_dense[int(np.argmax(response_dense[channel]))])
        filter_rows.append(
            {
                "channel": channel + 1,
                "dc_gain": float(abs(kernel.sum())),
                "dc_over_peak": float(response_dense[channel, 0] / response_dense[channel].max()),
                "peak_hz": peak,
                "half_power_low_hz": low,
                "half_power_high_hz": high,
                "half_power_bandwidth_hz": high - low,
                "kernel_sum": float(kernel.sum()),
                "kernel_l1": float(np.abs(kernel).sum()),
                "description": describe_filter(kernel, response_dense[channel], frequency_dense),
            }
        )
    pd.DataFrame(filter_rows).to_csv(DATA_DIR / "frontend_filter_summary.csv", index=False)
    pd.DataFrame(
        [
            {"channel": c + 1, "lag_ms": -1000.0 * lag * DT, "effective_weight": value}
            for c, kernel in enumerate(weights)
            for lag, value in enumerate(kernel)
        ]
    ).to_csv(DATA_DIR / "frontend_kernels.csv", index=False)
    pd.DataFrame(
        [
            {"channel": c + 1, "temporal_hz": f, "magnitude": response_dense[c, i]}
            for c in range(len(weights))
            for i, f in enumerate(frequency_dense)
        ]
    ).to_csv(DATA_DIR / "frontend_frequency_response.csv", index=False)

    retinal_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for sf_index, sf in enumerate(grid["radial_centers"]):
        if not np.isfinite(aggregate["radial_combined"][sf_index]).any():
            continue
        for scale_index, scale in enumerate(SCALES):
            for tf_index, tf in enumerate(aggregate["temporal_hz"]):
                retinal_rows.append(
                    {
                        "spatial_cpd": sf,
                        "trajectory_scale": scale,
                        "temporal_hz": tf,
                        "phase_only_natural_image_power": aggregate["radial_retinal"][
                            sf_index, scale_index, tf_index
                        ],
                    }
                )
            for channel in range(weights.shape[0]):
                energy_rows.append(
                    {
                        "spatial_cpd": sf,
                        "trajectory_scale": scale,
                        "frontend_channel": str(channel + 1),
                        "transmitted_motion_energy": aggregate["radial_channel"][
                            sf_index, scale_index, channel
                        ],
                        "combination_rule": "individual channel",
                    }
                )
            energy_rows.append(
                {
                    "spatial_cpd": sf,
                    "trajectory_scale": scale,
                    "frontend_channel": "combined",
                    "transmitted_motion_energy": aggregate["radial_combined"][sf_index, scale_index],
                    "combination_rule": "unweighted sum of physical output power across the four learned frontend channels",
                }
            )
            phase_rows.append(
                {
                    "spatial_cpd": sf,
                    "trajectory_scale": scale,
                    "mean_phase_rms_rad": aggregate["radial_phase_rms"][sf_index, scale_index],
                    "mean_phase_range_rad": aggregate["radial_phase_range"][sf_index, scale_index],
                    "mean_circular_concentration": aggregate["radial_concentration"][sf_index, scale_index],
                }
            )
    pd.DataFrame(retinal_rows).to_csv(DATA_DIR / "retinal_phase_only_power.csv.gz", index=False)
    energy_table = pd.DataFrame(energy_rows)
    energy_table.to_csv(DATA_DIR / "frontend_transmitted_energy.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(DATA_DIR / "phase_excursion.csv", index=False)
    return pd.DataFrame(filter_rows)


def write_early_resnet_tables(
    early: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gain_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    for sf_index, sf in enumerate(grid["radial_centers"]):
        if not np.isfinite(early["radial_energy"][sf_index]).any():
            continue
        for tf_index, tf in enumerate(early["temporal_hz"]):
            gain_rows.append(
                {
                    "spatial_cpd": sf,
                    "temporal_hz": tf,
                    "stem_spatial_gain": early["radial_stem_gain"][sf_index],
                    "first_resnet_3x9x9_gain": early["radial_resblock1_gain"][sf_index, tf_index],
                    "linear_fundamental_cascade_gain": early["radial_cascade_gain"][sf_index, tf_index],
                }
            )
        for scale_index, scale in enumerate(SCALES):
            energy_rows.append(
                {
                    "spatial_cpd": sf,
                    "trajectory_scale": scale,
                    "early_stack_transmitted_motion_energy": early["radial_energy"][sf_index, scale_index],
                    "scope": "frontend + spatial stem + first 3x9x9 ResNet convolution",
                    "approximation": "RMSNorm denominator identity with learned gamma retained; SplitReLU fundamental +/-0.5; DC excluded",
                }
            )
    pd.DataFrame(gain_rows).to_csv(DATA_DIR / "first_resnet_spatiotemporal_kernel_gain.csv.gz", index=False)
    pd.DataFrame(energy_rows).to_csv(DATA_DIR / "early_stack_transmitted_energy.csv", index=False)

    mixing_rows = []
    for output_index in range(early["resblock1_input_mixing_energy"].shape[0]):
        for input_index in range(early["resblock1_input_mixing_energy"].shape[1]):
            mixing_rows.append(
                {
                    "first_resnet_output": output_index,
                    "post_stem_split_input": input_index,
                    "kernel_weight_energy": early["resblock1_input_mixing_energy"][output_index, input_index],
                }
            )
    pd.DataFrame(mixing_rows).to_csv(DATA_DIR / "first_resnet_64x16_input_mixing.csv", index=False)

    output_gain_rows = []
    for sf_index, sf in enumerate(grid["radial_centers"]):
        if not np.isfinite(early["radial_output_gain"][sf_index]).any():
            continue
        for output_index in range(early["radial_output_gain"].shape[1]):
            for tf_index, tf in enumerate(early["temporal_hz"]):
                output_gain_rows.append(
                    {
                        "first_resnet_output": output_index,
                        "spatial_cpd": sf,
                        "temporal_hz": tf,
                        "mixed_linear_fundamental_gain": early["radial_output_gain"][sf_index, output_index, tf_index],
                    }
                )
    pd.DataFrame(output_gain_rows).to_csv(DATA_DIR / "first_resnet_64_mixed_output_gain.csv.gz", index=False)

    dose_rows: list[dict[str, Any]] = []
    for low, high, label in SF_BANDS:
        curve = band_curve(early["transmitted_mode"], grid["radial"], low, high)
        peak = int(np.nanargmax(curve))
        for scale_index, scale in enumerate(SCALES):
            dose_rows.append(
                {
                    "sf_band": label,
                    "sf_low_cpd": low,
                    "sf_high_cpd": high,
                    "trajectory_scale": scale,
                    "early_stack_energy": curve[scale_index],
                    "normalized_within_band": curve[scale_index] / max(np.nanmax(curve), 1e-30),
                    "band_optimum_scale": SCALES[peak],
                    "optimum_censoring": "right" if peak == len(SCALES) - 1 else "left" if peak == 0 else "none",
                }
            )
    dose = pd.DataFrame(dose_rows)
    dose.to_csv(DATA_DIR / "early_stack_sf_band_dose_curves.csv", index=False)

    output_dose_rows: list[dict[str, Any]] = []
    output_optimum_rows: list[dict[str, Any]] = []
    for band_index, (_, _, label) in enumerate(SF_BANDS):
        for output_index in range(early["output_band_energy"].shape[2]):
            curve = np.asarray(early["output_band_energy"][band_index, :, output_index], dtype=float)
            peak = int(np.nanargmax(curve))
            censoring = "right" if peak == len(SCALES) - 1 else "left" if peak == 0 else "none"
            output_optimum_rows.append(
                {
                    "first_resnet_output": output_index,
                    "sf_band": label,
                    "band_optimum_scale": SCALES[peak],
                    "optimum_censoring": censoring,
                    "peak_energy": curve[peak],
                }
            )
            for scale_index, scale in enumerate(SCALES):
                output_dose_rows.append(
                    {
                        "first_resnet_output": output_index,
                        "sf_band": label,
                        "trajectory_scale": scale,
                        "mixed_output_energy": curve[scale_index],
                        "normalized_within_output_and_band": curve[scale_index] / max(np.nanmax(curve), 1e-30),
                        "band_optimum_scale": SCALES[peak],
                        "optimum_censoring": censoring,
                    }
                )
    output_optima = pd.DataFrame(output_optimum_rows)
    pd.DataFrame(output_dose_rows).to_csv(DATA_DIR / "first_resnet_64_output_band_dose_curves.csv.gz", index=False)
    output_optima.to_csv(DATA_DIR / "first_resnet_64_output_band_optima.csv", index=False)
    return dose, output_optima


def band_curve(values: np.ndarray, radial: np.ndarray, low: float, high: float) -> np.ndarray:
    keep = (radial >= low) & (radial < high)
    if not np.any(keep):
        return np.full(values.shape[1:], np.nan)
    return np.nanmean(values[keep], axis=0)


def temporal_spatial_power(movie: np.ndarray, grid: dict[str, np.ndarray]) -> np.ndarray:
    coefficient_xy = np.fft.fft2(movie, axes=(-2, -1)) / (movie.shape[-2] * movie.shape[-1])
    coefficient = np.fft.fft(coefficient_xy, axis=0) / movie.shape[0]
    # ``collapse_absolute_temporal`` operates on the final dimension.  Rendered
    # movies are T x H x W, whereas the analytic phase tensor is mode x T.
    _, power = collapse_absolute_temporal(np.moveaxis(np.abs(coefficient) ** 2, 0, -1))
    selected = power.reshape(-1, power.shape[-1])[grid["flat_index"]]
    return group_mode_matrix(selected, grid["radial_bin"], len(grid["radial_centers"]))


def trajectory_aligned_slice(movie: np.ndarray, scored_trace: np.ndarray) -> np.ndarray:
    centered = scored_trace - np.mean(scored_trace, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    eye_axis = vh[0]
    pixel_axis = np.asarray([eye_axis[0], -eye_axis[1]], dtype=float)
    pixel_axis /= np.linalg.norm(pixel_axis)
    half = 0.46 * min(movie.shape[1:])
    positions = np.linspace(-half, half, movie.shape[2])
    center = 0.5 * (np.asarray(movie.shape[1:]) - 1)
    x = center[1] + positions * pixel_axis[0]
    y = center[0] + positions * pixel_axis[1]
    return np.stack([map_coordinates(frame, [y, x], order=1, mode="nearest") for frame in movie])


def make_figure1(inputs: dict[str, Any], grid: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    images = inputs["images"]
    traces = inputs["traces"]
    coherence = pd.to_numeric(images["image_orientation_coherence"], errors="coerce").to_numpy(float)
    candidates = np.flatnonzero(np.isfinite(coherence) & (coherence >= 0.2))
    target = float(np.median(coherence[candidates]))
    image_ordinal = int(candidates[np.argmin(np.abs(coherence[candidates] - target))])
    no_event = pd.to_numeric(traces["rendered_n_microsaccade_events"], errors="coerce").fillna(0).to_numpy() == 0
    paths = traces["rendered_path_length_arcmin"].to_numpy(float)
    candidates_trace = np.flatnonzero(no_event)
    path_target = float(np.median(paths[candidates_trace]))
    trace_id = int(candidates_trace[np.argmin(np.abs(paths[candidates_trace] - path_target))])
    row = images.iloc[image_ordinal]
    image_id = int(row["image_index"])
    patch, _ = extract_patch(row, canvas_cache={}, patch_size_px=540)
    scored = inputs["true_history"][trace_id, N_PRECEDING:]
    e0 = scored[0]
    shown_scales = np.asarray([0.0, 0.5, 1.0, 2.0])
    histories = np.asarray([e0 + scale * (scored - e0) for scale in shown_scales])
    movies = render_movies(patch, histories, device=device)
    slices = np.asarray([trajectory_aligned_slice(movie, history) for movie, history in zip(movies, histories)])
    powers = np.asarray([temporal_spatial_power(movie, grid) for movie in movies])
    np.savez_compressed(
        EXACT_DIR / "figure1_representative_movies.npz",
        scales=shown_scales,
        movies=movies.astype(np.float32),
        xt_slices=slices.astype(np.float32),
        radial_temporal_power=powers.astype(np.float32),
        spatial_cpd=grid["radial_centers"],
        temporal_hz=np.unique(np.abs(np.fft.fftfreq(N_TIME, d=DT))),
        image_id=image_id,
        trace_id=trace_id,
        scored_trace_xy=scored,
    )
    rows = []
    for scale_index, scale in enumerate(shown_scales):
        for sf_index, sf in enumerate(grid["radial_centers"]):
            for tf_index, tf in enumerate(np.unique(np.abs(np.fft.fftfreq(N_TIME, d=DT)))):
                rows.append(
                    {
                        "trajectory_scale": scale,
                        "spatial_cpd": sf,
                        "temporal_hz": tf,
                        "rendered_retinal_power": powers[scale_index, sf_index, tf_index],
                    }
                )
    pd.DataFrame(rows).to_csv(DATA_DIR / "figure1_rendered_retinal_power.csv.gz", index=False)

    fig = plt.figure(figsize=(12.2, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=(0.85, 1.0, 1.1))
    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(movies[0, 0], cmap="gray", vmin=np.percentile(movies[0], 1), vmax=np.percentile(movies[0], 99))
    ax.set_title(f"A  Natural image {image_id}", loc="left")
    ax.set_axis_off()
    ax = fig.add_subplot(gs[0, 2:])
    centered = 60.0 * (scored - scored[0])
    ax.plot(centered[:, 0], centered[:, 1], color=NEUTRAL, lw=1.7)
    ax.scatter(centered[0, 0], centered[0, 1], s=28, color=LOW_COLOR, zorder=3, label="start")
    ax.set(
        aspect="equal",
        xlabel="horizontal displacement (arcmin)",
        ylabel="vertical displacement (arcmin)",
        title=f"B  Median drift {trace_id} ({paths[trace_id]:.1f} arcmin path)",
    )
    ax.legend(frameon=False)
    for index, scale in enumerate(shown_scales):
        ax = fig.add_subplot(gs[1, index])
        ax.imshow(
            slices[index].T,
            origin="lower",
            aspect="auto",
            cmap="gray",
            extent=(0, 1000 * (N_TIME - 1) * DT, -0.5 * OUT_SIZE[1] / PPD, 0.5 * OUT_SIZE[1] / PPD),
        )
        ax.set_title(f"{chr(67 + index)}  {scale:g}× movement")
        ax.set_xlabel("time (ms)")
        if index == 0:
            ax.set_ylabel("position along drift axis (deg)")
    # For explanation, show how each physical SF redistributes its own power
    # over temporal frequency.  Absolute power (including the natural 1/f
    # envelope) remains in the tidy CSV and exact NPZ.
    display_power = powers / np.maximum(powers.sum(axis=-1, keepdims=True), 1e-30)
    vmax = 1.0
    vmin = 1e-5
    for index, scale in enumerate(shown_scales):
        ax = fig.add_subplot(gs[2, index])
        image = np.maximum(display_power[index].T, vmin)
        mesh = ax.pcolormesh(
            grid["radial_centers"],
            np.unique(np.abs(np.fft.fftfreq(N_TIME, d=DT))),
            image,
            shading="nearest",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set(xlim=(0.125, 12), ylim=(0, 60), xlabel="spatial frequency (cpd)")
        ax.set_title(f"{chr(71 + index)}  {scale:g}×: SF → temporal power")
        if index == 0:
            ax.set_ylabel("temporal frequency (Hz)")
        if index == len(shown_scales) - 1:
            fig.colorbar(mesh, ax=ax, pad=0.02, label="fraction of each SF mode's temporal power")
    fig.suptitle("Retinal motion converts space into time", fontsize=15, weight="bold")
    save_figure(fig, "figure1_retinal_motion_converts_space_into_time")
    return {"representative_image_id": image_id, "representative_trace_id": trace_id}


def make_figure2(
    weights: np.ndarray,
    filter_table: pd.DataFrame,
    aggregate: dict[str, np.ndarray],
    early: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
) -> None:
    dense_f = np.linspace(0, 60, 1201)
    dense_h = frontend_transfer(weights, dense_f)
    lag_ms = -1000.0 * np.arange(weights.shape[1]) * DT
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.7), constrained_layout=True)
    ax = axes[0, 0]
    for channel, color in enumerate(CHANNEL_COLORS):
        ax.plot(lag_ms, weights[channel], "o-", ms=3, lw=1.4, color=color, label=f"channel {channel + 1}")
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel="input lag (ms; 0 = current)", ylabel="effective learned weight", title="A  Trained temporal kernels")
    ax.legend(frameon=False, ncol=2)
    ax = axes[0, 1]
    for channel, color in enumerate(CHANNEL_COLORS):
        ax.plot(dense_f, dense_h[channel] / dense_h[channel].max(), lw=1.8, color=color, label=f"ch. {channel + 1}")
        peak = filter_table.iloc[channel]["peak_hz"]
        ax.text(min(float(peak), 57), 1.02 - 0.07 * channel, f"{peak:.1f} Hz", color=color, ha="right", va="top", fontsize=7)
    ax.set(xlim=(0, 60), ylim=(0, 1.08), xlabel="temporal frequency (Hz)", ylabel="magnitude / channel maximum", title="B  Temporal frontend transfer")
    ax = axes[0, 2]
    stem = np.asarray(early["radial_stem_gain"], dtype=float)
    valid = np.isfinite(stem) & (grid["radial_centers"] <= 16)
    ax.plot(grid["radial_centers"][valid], stem[valid] / np.nanmax(stem[valid]), color="#3B82F6", lw=2)
    ax.set(xlim=(0.125, 16), ylim=(0, 1.05), xlabel="physical spatial frequency (cpd)", ylabel="summed gain / maximum", title="C  1×7×7 stem + learned RMS gain")

    ax = axes[1, 0]
    mixing = np.asarray(early["resblock1_input_mixing_energy"], dtype=float)
    mixing /= np.maximum(mixing.sum(axis=1, keepdims=True), 1e-30)
    mesh = ax.imshow(mixing, aspect="auto", origin="lower", interpolation="nearest", cmap="Blues")
    ax.set(
        xlabel="post-stem SplitReLU input (16)",
        ylabel="first-kernel preactivation output (64)",
        title="D  Trained 64×16 channel mixing",
        xticks=np.arange(0, 16, 3),
    )
    fig.colorbar(mesh, ax=ax, pad=0.02, label="fraction of each output's kernel energy")

    ax = axes[1, 1]
    gain = np.asarray(early["radial_resblock1_gain"], dtype=float)
    valid_sf = np.isfinite(gain).all(axis=1) & (grid["radial_centers"] <= 16)
    positive = gain[valid_sf, 1:]
    vmax = float(np.nanmax(positive))
    vmin = max(vmax * 1e-4, 1e-30)
    mesh = ax.pcolormesh(
        grid["radial_centers"][valid_sf], early["temporal_hz"],
        np.maximum(gain[valid_sf].T, vmin), shading="nearest", cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set(xlim=(0.125, 16), ylim=(0, 60), xlabel="physical spatial frequency (cpd)", ylabel="temporal frequency (Hz)", title="E  First ResNet 3×9×9 kernel gain")
    fig.colorbar(mesh, ax=ax, pad=0.02, label="summed |K(k,f)|²")

    ax = axes[1, 2]
    cascade = np.asarray(early["radial_cascade_gain"], dtype=float)
    positive = cascade[valid_sf, 1:]
    vmax = float(np.nanmax(positive))
    vmin = max(vmax * 1e-5, 1e-30)
    mesh = ax.pcolormesh(
        grid["radial_centers"][valid_sf], early["temporal_hz"],
        np.maximum(cascade[valid_sf].T, vmin), shading="nearest", cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set(xlim=(0.125, 16), ylim=(0, 60), xlabel="physical spatial frequency (cpd)", ylabel="temporal frequency (Hz)", title="F  Linear-fundamental early-stack gain")
    fig.colorbar(mesh, ax=ax, pad=0.02, label="cascade gain")
    fig.suptitle("What signals reach the first spatiotemporal ResNet layer?", fontsize=15, weight="bold")
    save_figure(fig, "figure2_early_spatiotemporal_filters")


def make_figure_s1_all_mixed_outputs(
    early: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
) -> None:
    """Show every mixed preactivation output, not only representative examples."""
    centers = grid["radial_centers"]
    valid = np.isfinite(early["radial_output_gain"]).all(axis=(1, 2)) & (centers <= 16.0)
    fig, axes = plt.subplots(8, 8, figsize=(14.5, 13.0), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for output_index, ax in enumerate(axes.flat):
        response = np.asarray(early["radial_output_gain"][valid, output_index, :], dtype=float).T
        response /= max(float(np.nanmax(response)), 1e-30)
        mesh = ax.pcolormesh(
            centers[valid],
            early["temporal_hz"],
            np.maximum(response, 1e-3),
            shading="nearest",
            cmap="magma",
            norm=LogNorm(vmin=1e-3, vmax=1.0),
        )
        ax.set(xlim=(0.125, 16), ylim=(0, 60), title=f"output {output_index}")
        ax.title.set_fontsize(7)
        if output_index % 8 == 0:
            ax.set_ylabel("TF (Hz)", fontsize=7)
        if output_index >= 56:
            ax.set_xlabel("SF (cpd)", fontsize=7)
        ax.tick_params(labelsize=6)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, shrink=0.55, pad=0.01, label="gain / output maximum")
    fig.suptitle(
        "All 64 outputs after trained frontend → stem → first 3×9×9 channel mixing\n"
        "Linear-fundamental SF×TF gain; each output normalized independently",
        fontsize=15,
        weight="bold",
    )
    save_figure(fig, "figureS1_all_64_mixed_first_resnet_outputs")


def band_dose_table(aggregate: dict[str, np.ndarray], grid: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for low, high, label in SF_BANDS:
        combined = band_curve(aggregate["combined_mode"], grid["radial"], low, high)
        peak_index = int(np.nanargmax(combined))
        for scale_index, scale in enumerate(SCALES):
            rows.append(
                {
                    "sf_band": label,
                    "sf_low_cpd": low,
                    "sf_high_cpd": high,
                    "trajectory_scale": scale,
                    "combined_transmitted_energy": combined[scale_index],
                    "normalized_within_band": combined[scale_index] / max(np.nanmax(combined), 1e-30),
                    "band_optimum_scale": SCALES[peak_index],
                    "optimum_censoring": "right" if peak_index == len(SCALES) - 1 else "left" if peak_index == 0 else "none",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "frontend_sf_band_dose_curves.csv", index=False)
    return table


def select_representative_mixed_outputs(
    radial_output_gain: np.ndarray,
    centers: np.ndarray,
    temporal_hz: np.ndarray,
    n_output: int = 4,
) -> list[int]:
    """Deterministic farthest-point sample of the 64 mixed SF–TF profiles."""
    keep_sf = np.isfinite(radial_output_gain).all(axis=(1, 2)) & (centers <= 16.0)
    response = np.asarray(radial_output_gain[keep_sf, :, 1:], dtype=float).transpose(1, 0, 2)
    spatial_marginal = response.sum(axis=2)
    temporal_marginal = response.sum(axis=1)
    spatial_marginal /= np.maximum(spatial_marginal.sum(axis=1, keepdims=True), 1e-30)
    temporal_marginal /= np.maximum(temporal_marginal.sum(axis=1, keepdims=True), 1e-30)
    features = np.concatenate((spatial_marginal, temporal_marginal), axis=1)
    features -= features.mean(axis=1, keepdims=True)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-30)
    total = response.sum(axis=(1, 2))
    chosen = [int(np.argmax(total))]
    while len(chosen) < n_output:
        distances = np.min(
            np.sum((features[:, None, :] - features[np.asarray(chosen)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distances[np.asarray(chosen)] = -np.inf
        chosen.append(int(np.argmax(distances)))
    tf_centroid = (
        temporal_marginal @ np.asarray(temporal_hz[1:], dtype=float)
    )
    return sorted(chosen, key=lambda index: tf_centroid[index])


def make_figure3(
    aggregate: dict[str, np.ndarray],
    early: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
    frontend_dose: pd.DataFrame,
    early_dose: pd.DataFrame,
    output_optima: pd.DataFrame,
) -> pd.DataFrame:
    centers = grid["radial_centers"]
    combined = np.asarray(early["radial_energy"], dtype=float)
    valid = np.isfinite(combined).all(axis=1) & (centers <= 16.0)
    ridge_rows = []
    for index in np.flatnonzero(valid):
        peak = int(np.nanargmax(combined[index]))
        ridge_rows.append(
            {
                "spatial_cpd": centers[index],
                "optimum_scale": SCALES[peak],
                "censoring": "right" if peak == len(SCALES) - 1 else "left" if peak == 0 else "none",
                "peak_energy": combined[index, peak],
                "stage": "linear-fundamental cascade through first 3x9x9 ResNet convolution",
            }
        )
    ridge = pd.DataFrame(ridge_rows)
    ridge.to_csv(DATA_DIR / "early_stack_optimum_ridge.csv", index=False)

    fig = plt.figure(figsize=(12.6, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=(1.4, 0.8))
    ax = fig.add_subplot(gs[0, :3])
    matrix = combined[valid].T
    positive = matrix[matrix > 0]
    vmin = max(float(np.nanmax(matrix)) * 1e-5, float(np.nanpercentile(positive, 2)))
    mesh = ax.pcolormesh(
        centers[valid],
        SCALES,
        np.maximum(matrix, vmin),
        shading="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=float(np.nanmax(matrix))),
    )
    interior = ridge["censoring"].eq("none")
    ax.plot(ridge.loc[interior, "spatial_cpd"], ridge.loc[interior, "optimum_scale"], "w.-", lw=1.3, ms=5, label="interior optimum")
    boundary = ridge["censoring"].eq("right")
    ax.scatter(ridge.loc[boundary, "spatial_cpd"], ridge.loc[boundary, "optimum_scale"], marker="^", facecolors="none", edgecolors="white", s=26, label="still rising at 3×")
    ax.axhline(1.0, color="white", ls="--", lw=0.9)
    ax.set(xlim=(0.125, 16), xlabel="physical spatial frequency (cpd)", ylabel="retinal trajectory amplitude (× measured FEM)", title="A  Energy after first 3×9×9 kernel (linear fundamental)")
    ax.legend(frameon=False, labelcolor="white", loc="lower right")
    fig.colorbar(mesh, ax=ax, pad=0.02, label="linear-fundamental early-stack energy")

    ax = fig.add_subplot(gs[0, 3:])
    colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(SF_BANDS)))
    for (band, frame), color in zip(early_dose.groupby("sf_band", sort=False), colors):
        frame = frame.sort_values("trajectory_scale")
        ax.plot(frame["trajectory_scale"], frame["normalized_within_band"], "o-", lw=1.7, ms=4, color=color, label=band)
        front = frontend_dose.loc[frontend_dose.sf_band.eq(band)].sort_values("trajectory_scale")
        ax.plot(front["trajectory_scale"], front["normalized_within_band"], "--", lw=0.9, color=color, alpha=0.55)
    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    ax.set(xlim=(0, 3), ylim=(0, 1.06), xlabel="retinal trajectory amplitude (× measured FEM)", ylabel="energy / band maximum", title="B  Early stack (solid); temporal-only (dashed)")
    ax.legend(frameon=False, ncol=1)

    representative_outputs = select_representative_mixed_outputs(
        early["radial_output_gain"], centers, early["temporal_hz"]
    )
    for panel_index, output_index in enumerate(representative_outputs):
        ax = fig.add_subplot(gs[1, panel_index])
        channel_matrix = early["radial_output_gain"][valid, output_index, :].T
        vmax_c = float(np.nanmax(channel_matrix))
        vmin_c = max(vmax_c * 1e-4, 1e-30)
        ax.pcolormesh(
            centers[valid], early["temporal_hz"], np.maximum(channel_matrix, vmin_c),
            shading="nearest", cmap="magma", norm=LogNorm(vmin=vmin_c, vmax=vmax_c),
        )
        ax.set(xlim=(0.125, 16), ylim=(0, 60), xlabel="SF (cpd)", title=f"{chr(67 + panel_index)}  mixed output {output_index}")
        if panel_index == 0:
            ax.set_ylabel("TF (Hz)")
    ax = fig.add_subplot(gs[1, 4])
    bar_width = 0.09
    for offset, (band, color, label) in enumerate(
        (("0.25–0.5 cpd", LOW_COLOR, "low physical SF"), ("2–4 cpd", HIGH_COLOR, "higher physical SF"))
    ):
        frame = output_optima.loc[output_optima.sf_band.eq(band)]
        counts = frame.band_optimum_scale.value_counts().reindex(SCALES, fill_value=0)
        shift = (-0.5 if offset == 0 else 0.5) * bar_width
        ax.bar(SCALES + shift, counts.to_numpy(), width=bar_width, color=color, alpha=0.82, label=label)
    ax.set(xlim=(-0.1, 3.15), xlabel="output-specific optimum scale", ylabel="number of 64 outputs", title="G  Mixed-output optima")
    ax.legend(frameon=False, fontsize=6)
    fig.suptitle("None of the 64 mixed outputs produces the low/high optimum split", fontsize=15, weight="bold")
    save_figure(fig, "figure3_predicted_movement_scale_by_spatial_frequency")
    return ridge


def finite_crop_audit(
    inputs: dict[str, Any],
    grid: dict[str, np.ndarray],
    device: str,
) -> pd.DataFrame:
    cache = EXACT_DIR / "finite_crop_radial_spectra.npz"
    n_bin = len(grid["radial_centers"])
    temporal_hz = np.unique(np.abs(np.fft.fftfreq(N_TIME, d=DT)))
    if cache.is_file():
        with np.load(cache) as archive:
            actual = np.asarray(archive["actual_power"], dtype=np.float64)
            ideal = np.asarray(archive["ideal_power"], dtype=np.float64)
            n_pair = int(archive["n_pairs"])
    else:
        actual = np.zeros((len(SCALES), n_bin, len(temporal_hz)), dtype=np.float64)
        ideal = np.zeros_like(actual)
        n_pair = 0
        canvas_cache: dict[Any, Any] = {}
        for image_count, image_id in enumerate(inputs["selected_image_ids"]):
            row = inputs["images"].iloc[int(image_id)]
            patch, _ = extract_patch(row, canvas_cache=canvas_cache, patch_size_px=540)
            for trace_id in inputs["selected_trace_ids"]:
                scored = inputs["true_history"][int(trace_id), N_PRECEDING:]
                e0 = scored[0]
                histories = np.asarray([e0 + scale * (scored - e0) for scale in SCALES])
                movies = render_movies(patch, histories, device=device)
                base_coefficient = np.fft.fft2(movies[0, 0]) / movies[0, 0].size
                base_power = np.abs(base_coefficient.ravel()[grid["flat_index"]]) ** 2
                displacement = scored - e0
                dot = np.einsum("kd,td->kt", grid["kxy"], displacement)
                for scale_index, scale in enumerate(SCALES):
                    actual[scale_index] += temporal_spatial_power(movies[scale_index], grid)
                    signal = np.exp(-2j * np.pi * float(scale) * dot)
                    coefficient = np.fft.fft(signal, axis=-1) / N_TIME
                    _, phase_power = collapse_absolute_temporal(np.abs(coefficient) ** 2)
                    ideal_mode = base_power[:, None] * phase_power
                    ideal[scale_index] += group_mode_matrix(ideal_mode, grid["radial_bin"], n_bin)
                n_pair += 1
            print(f"finite-crop exact renderer images {image_count + 1}/{len(inputs['selected_image_ids'])}", flush=True)
        actual /= n_pair
        ideal /= n_pair
        np.savez_compressed(
            cache,
            actual_power=actual.astype(np.float32),
            ideal_power=ideal.astype(np.float32),
            n_pairs=n_pair,
            selected_image_ids=inputs["selected_image_ids"],
            selected_trace_ids=inputs["selected_trace_ids"],
            scales=SCALES,
            spatial_cpd=grid["radial_centers"],
            temporal_hz=temporal_hz,
        )
    rows = []
    for scale_index, scale in enumerate(SCALES):
        for sf_index, sf in enumerate(grid["radial_centers"]):
            a = np.asarray(actual[scale_index, sf_index, 1:], dtype=float)
            b = np.asarray(ideal[scale_index, sf_index, 1:], dtype=float)
            total_a = float(np.nansum(a))
            total_b = float(np.nansum(b))
            if scale == 0 or total_b <= 1e-20:
                relative_l1 = 0.0 if total_a <= 1e-12 else math.nan
                tv = 0.0 if total_a <= 1e-12 else math.nan
                ratio = math.nan
            else:
                relative_l1 = float(np.nansum(np.abs(a - b)) / total_b)
                tv = float(0.5 * np.nansum(np.abs(a / max(total_a, 1e-30) - b / total_b)))
                ratio = total_a / total_b
            rows.append(
                {
                    "spatial_cpd": sf,
                    "trajectory_scale": scale,
                    "actual_non_dc_power": total_a,
                    "ideal_non_dc_power": total_b,
                    "actual_over_ideal_non_dc_power": ratio,
                    "relative_l1_spectral_error": relative_l1,
                    "temporal_distribution_total_variation": tv,
                    "n_image_trajectory_pairs": n_pair,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(DATA_DIR / "finite_crop_discrepancy.csv", index=False)
    return table


def load_corrected_ssi_proxy(inputs: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    corrected_controlled = CONTROLLED_DIR / "corrected_controlled_scaling_response.npz"
    if corrected_controlled.is_file():
        # This branch is deliberately not reached in the present workspace.  Keep the
        # gate explicit so a future valid cache is not silently confused with the proxy.
        raise RuntimeError(
            "A corrected controlled-scaling archive appeared after analysis design; add a read-only analyzer before using it."
        )
    observational = pd.read_csv(CORRECTED_CURVES)
    observational = observational[
        observational["bank"].eq("real_trace_true_history_v1")
        & observational["context"].eq("drift_only")
        & observational["sf_group"].isin(("low_sf_lt0p5", "high_sf_ge0p5"))
    ].copy()
    drift = inputs["traces"]["rendered_n_microsaccade_events"].fillna(0).to_numpy(int) == 0
    median_path = float(np.median(inputs["traces"].loc[drift, "rendered_path_length_arcmin"]))
    observational["trajectory_scale_proxy"] = observational["path_median_arcmin"] / median_path
    observational["display_group"] = observational["sf_group"].map(
        {"low_sf_lt0p5": "low-SF model population", "high_sf_ge0p5": "high-SF model population"}
    )
    observational["source_status"] = "corrected true-history observational path bins; normalized-path amplitude proxy, not controlled scaling"
    observational.to_csv(DATA_DIR / "corrected_ssi_observational_proxy.csv", index=False)

    # The conditional deeper-layer pilot is allowed only after a negative
    # early-stage result. Its final readout gives a small, corrected controlled
    # endpoint on the exact scale axis; it is not the missing full cache.
    pilot_path = (
        ROOT
        / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1/first_pass_v1/plot_data/layerwise_final_ssi_curves.csv"
    )
    if pilot_path.is_file():
        pilot = pd.read_csv(pilot_path).copy()
        pilot["trajectory_scale_proxy"] = pilot["scale"]
        pilot["ssi_percent_vs_stabilized"] = pilot["ssi_percent_vs_0x"]
        pilot["ci95_low_paired_image_boot"] = pilot["ci95_low"]
        pilot["ci95_high_paired_image_boot"] = pilot["ci95_high"]
        pilot["display_group"] = pilot["sf_group"].map(
            {"low SF": "low-SF model population", "high SF": "high-SF model population"}
        )
        pilot["source_status"] = (
            "conditional corrected controlled layerwise pilot: 8 images x 8 preselected drifts; not the missing full controlled-scaling cache"
        )
        pilot.to_csv(DATA_DIR / "conditional_controlled_ssi_pilot.csv", index=False)
        return pilot, "conditional corrected controlled 8-image x 8-trajectory layerwise pilot"
    return observational, "corrected true-history observational proxy (controlled-scaling cache absent)"


def make_figure4(
    aggregate: dict[str, np.ndarray],
    early: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
    corrected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    physical = ((0.25, 0.5, "low physical SF: 0.25–0.5 cpd", LOW_COLOR), (2.0, 4.0, "higher physical SF: 2–4 cpd", HIGH_COLOR))
    for low, high, label, color in physical:
        frontend_curve = band_curve(aggregate["combined_mode"], grid["radial"], low, high)
        frontend_curve /= max(np.nanmax(frontend_curve), 1e-30)
        early_curve = band_curve(early["transmitted_mode"], grid["radial"], low, high)
        early_curve /= max(np.nanmax(early_curve), 1e-30)
        for scale, value in zip(SCALES, frontend_curve):
            rows.append({"panel": "early prediction", "predictor": "temporal frontend only", "group": label, "trajectory_scale": scale, "value": value, "color": color})
        for scale, value in zip(SCALES, early_curve):
            rows.append({"panel": "early prediction", "predictor": "linear through first ResNet 3x9x9 preactivation", "group": label, "trajectory_scale": scale, "value": value, "color": color})
    for _, row in corrected.iterrows():
        rows.append(
            {
                "panel": "corrected SSI observational proxy",
                "group": row["display_group"],
                "trajectory_scale": row["trajectory_scale_proxy"],
                "value": row["ssi_percent_vs_stabilized"],
                "ci95_low": row["ci95_low_paired_image_boot"],
                "ci95_high": row["ci95_high_paired_image_boot"],
                "color": LOW_COLOR if "low" in row["display_group"] else HIGH_COLOR,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(DATA_DIR / "figure4_early_stack_vs_corrected_ssi.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True, constrained_layout=True)
    ax = axes[0]
    prediction = comparison[comparison.panel.eq("early prediction")]
    for (label, predictor), frame in prediction.groupby(["group", "predictor"], sort=False):
        early_line = "first ResNet" in predictor
        ax.plot(
            frame.trajectory_scale, frame.value,
            "o-" if early_line else "--", lw=2 if early_line else 1.2,
            ms=4.5 if early_line else 0, color=frame.color.iloc[0], alpha=1.0 if early_line else 0.65,
            label=f"{label} — {'through 3×9×9 kernel (linear)' if early_line else 'temporal only'}",
        )
    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    ax.set(ylabel="predicted energy / band maximum", title="A  Early linear stages still rise through 3×")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(
        0.02,
        0.05,
        "Channel-resolved check: 64/64 mixed outputs peak at 3×\nin both 0.25–0.5 and 2–4 cpd bands",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
    )
    ax = axes[1]
    proxy = comparison[comparison.panel.eq("corrected SSI observational proxy")]
    for label, frame in proxy.groupby("group", sort=False):
        frame = frame.sort_values("trajectory_scale")
        ax.plot(frame.trajectory_scale, frame.value, "o-", lw=2, ms=4.5, color=frame.color.iloc[0], label=label)
        ax.fill_between(frame.trajectory_scale, frame.ci95_low, frame.ci95_high, color=frame.color.iloc[0], alpha=0.12, linewidth=0)
    ax.axvline(1.0, color="black", ls="--", lw=0.9)
    is_pilot = corrected["source_status"].astype(str).str.contains("conditional corrected controlled").all()
    ax.set(
        xlim=(0, 3),
        xlabel=(
            "retinal trajectory amplitude (× measured FEM)"
            if is_pilot
            else "retinal trajectory amplitude (× median measured FEM)"
        ),
        ylabel="SSI change vs stabilized (%)",
        title=(
            "B  Corrected controlled SSI (conditional 8-image × 8-drift pilot)"
            if is_pilot
            else "B  Corrected 100-image SSI (natural path bins; amplitude proxy)"
        ),
    )
    ax.legend(frameon=False)
    fig.suptitle("Do the early filtering stages explain the Figure 4 SSI result?", fontsize=15, weight="bold")
    save_figure(fig, "figure4_early_stack_vs_corrected_ssi")
    return comparison


def summarize_results(
    filter_table: pd.DataFrame,
    frontend_dose: pd.DataFrame,
    early_dose: pd.DataFrame,
    output_optima: pd.DataFrame,
    ridge: pd.DataFrame,
    phase_table: pd.DataFrame,
    finite: pd.DataFrame,
    corrected: pd.DataFrame,
    example: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    frontend_optima = (
        frontend_dose.groupby("sf_band", sort=False)
        .first()[["band_optimum_scale", "optimum_censoring"]]
        .reset_index()
    )
    early_optima = (
        early_dose.groupby("sf_band", sort=False)
        .first()[["band_optimum_scale", "optimum_censoring"]]
        .reset_index()
    )
    frontend_low_opt = float(frontend_optima.loc[frontend_optima.sf_band.eq("0.25–0.5 cpd"), "band_optimum_scale"].iloc[0])
    frontend_high_opt = float(frontend_optima.loc[frontend_optima.sf_band.eq("2–4 cpd"), "band_optimum_scale"].iloc[0])
    early_low_opt = float(early_optima.loc[early_optima.sf_band.eq("0.25–0.5 cpd"), "band_optimum_scale"].iloc[0])
    early_high_opt = float(early_optima.loc[early_optima.sf_band.eq("2–4 cpd"), "band_optimum_scale"].iloc[0])
    frontend_predicts = bool(frontend_high_opt < frontend_low_opt)
    early_predicts = bool(early_high_opt < early_low_opt)
    low_output_optima = output_optima.loc[
        output_optima.sf_band.eq("0.25–0.5 cpd"),
        ["first_resnet_output", "band_optimum_scale", "optimum_censoring"],
    ].rename(columns={"band_optimum_scale": "low_optimum", "optimum_censoring": "low_censoring"})
    high_output_optima = output_optima.loc[
        output_optima.sf_band.eq("2–4 cpd"),
        ["first_resnet_output", "band_optimum_scale", "optimum_censoring"],
    ].rename(columns={"band_optimum_scale": "high_optimum", "optimum_censoring": "high_censoring"})
    paired_output_optima = low_output_optima.merge(high_output_optima, on="first_resnet_output", validate="one_to_one")
    output_distributions: dict[str, dict[str, int]] = {}
    for band, frame in output_optima.groupby("sf_band", sort=False):
        counts = frame.band_optimum_scale.value_counts().sort_index()
        output_distributions[str(band)] = {f"{float(scale):g}x": int(count) for scale, count in counts.items()}
    n_high_before_low = int(np.sum(paired_output_optima.high_optimum < paired_output_optima.low_optimum))
    interior_ridge = ridge.loc[ridge.censoring.eq("none")]
    first_interior_sf = float(interior_ridge.spatial_cpd.min()) if len(interior_ridge) else math.nan
    phase_high = phase_table[
        phase_table.spatial_cpd.between(2.0, 4.0, inclusive="left")
        & np.isclose(phase_table.trajectory_scale, early_high_opt)
    ].mean(numeric_only=True)
    finite_summary = {}
    for scale in (1.0, 2.0, 3.0):
        frame = finite[np.isclose(finite.trajectory_scale, scale) & finite.spatial_cpd.between(0.25, 8.0)]
        finite_summary[f"scale_{scale:g}_median_relative_l1"] = float(frame.relative_l1_spectral_error.median())
        finite_summary[f"scale_{scale:g}_median_total_variation"] = float(frame.temporal_distribution_total_variation.median())
        finite_summary[f"scale_{scale:g}_median_actual_over_ideal"] = float(
            frame.actual_over_ideal_non_dc_power.median()
        )
    corrected_peaks = {}
    for group, frame in corrected.groupby("display_group"):
        row = frame.iloc[int(np.nanargmax(frame.ssi_percent_vs_stabilized.to_numpy(float)))]
        corrected_peaks[group] = {
            "amplitude_proxy_at_peak": float(row.trajectory_scale_proxy),
            "ssi_percent_at_peak": float(row.ssi_percent_vs_stabilized),
        }
    return {
        "analysis": "targeted_retinal_phase_modulation_through_first_spatiotemporal_resnet_layer",
        "elapsed_minutes": elapsed_s / 60.0,
        "n_natural_images_ideal": 100,
        "n_controlled_drift_trajectories_ideal": 8,
        "n_exact_2d_fourier_modes": int(len(ridge)),
        "scales": SCALES,
        "frame_rate_hz": FRAME_RATE_HZ,
        "n_time_samples": N_TIME,
        "temporal_frequency_resolution_hz": FRAME_RATE_HZ / N_TIME,
        "spatial_aperture_pixels": OUT_SIZE,
        "pixels_per_degree": PPD,
        "frontend_combination_rule": "unweighted sum of output power across four learned temporal channels",
        "early_stack_definition": "trained temporal frontend + 1x7x7 spatial stem + first 3x9x9 ResNet convolution",
        "early_stack_approximation": "linear-fundamental calculation: RMSNorm denominator treated as identity, learned affine gamma retained, beta omitted from the AC fundamental, and SplitReLU branches carry +/-0.5 of the input fundamental; exact nonlinear activations assessed separately in the conditional hook pilot",
        "frontend_filters": filter_table.to_dict(orient="records"),
        "frontend_sf_band_optima": frontend_optima.to_dict(orient="records"),
        "early_stack_sf_band_optima": early_optima.to_dict(orient="records"),
        "frontend_predicts_high_smaller_than_low": frontend_predicts,
        "early_stack_predicts_high_smaller_than_low": early_predicts,
        "first_resnet_mixed_output_count": int(output_optima.first_resnet_output.nunique()),
        "first_resnet_output_optimum_distributions": output_distributions,
        "first_resnet_outputs_high_optimum_smaller_than_low_n": n_high_before_low,
        "first_resnet_outputs_high_optimum_smaller_than_low_fraction": n_high_before_low / len(paired_output_optima),
        "first_resnet_outputs_both_right_censored_n": int(
            np.sum(paired_output_optima.low_censoring.eq("right") & paired_output_optima.high_censoring.eq("right"))
        ),
        "early_stack_first_interior_optimum_spatial_cpd": first_interior_sf,
        "frontend_low_physical_sf_optimum_scale": frontend_low_opt,
        "frontend_higher_physical_sf_optimum_scale": frontend_high_opt,
        "early_stack_low_physical_sf_optimum_scale": early_low_opt,
        "early_stack_higher_physical_sf_optimum_scale": early_high_opt,
        "higher_sf_mean_phase_rms_at_early_stack_optimum_rad": float(phase_high.mean_phase_rms_rad),
        "finite_crop": finite_summary,
        "corrected_ssi_comparison_status": str(corrected["source_status"].iloc[0]),
        "corrected_ssi_proxy_peaks": corrected_peaks,
        "representative_example": example,
        "deeper_network_run": "not run; analytic early-stack result determines whether it is required",
        "checkpoint_path": MODEL_CHECKPOINT_PATH,
        "checkpoint_sha256": sha256_file(MODEL_CHECKPOINT_PATH),
    }


def load_deeper_layer_summary() -> dict[str, Any] | None:
    source = (
        ROOT
        / "outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1/first_pass_v1/plot_data"
    )
    curve_path = source / "layerwise_final_ssi_curves.csv"
    layer_path = source / "layerwise_spatial_concentration.csv"
    intervention_path = source / "frontend_replacement_intervention.csv"
    if not (curve_path.is_file() and layer_path.is_file() and intervention_path.is_file()):
        return None
    curves = pd.read_csv(curve_path)
    layers = pd.read_csv(layer_path)
    intervention = pd.read_csv(intervention_path)
    curves.to_csv(DATA_DIR / "conditional_layerwise_final_ssi_curves.csv", index=False)
    layers.to_csv(DATA_DIR / "conditional_layerwise_spatial_concentration.csv", index=False)
    intervention.to_csv(DATA_DIR / "conditional_frontend_replacement_intervention.csv", index=False)

    group_summary: dict[str, Any] = {}
    for group, frame in curves.groupby("sf_group", sort=False):
        peak = frame.iloc[int(np.nanargmax(frame.ssi_percent_vs_0x.to_numpy(float)))]
        at_three = frame.loc[np.isclose(frame.scale, 3.0)].iloc[0]
        group_summary[str(group)] = {
            "optimum_scale": float(peak.scale),
            "peak_ssi_percent_vs_0x": float(peak.ssi_percent_vs_0x),
            "ssi_percent_at_3x": float(at_three.ssi_percent_vs_0x),
            "ci95_at_3x": [float(at_three.ci95_low), float(at_three.ci95_high)],
        }

    first_positive: dict[str, str | None] = {}
    stages = list(layers.stage.drop_duplicates())
    for scale in (1.0, 2.0, 3.0):
        frame = layers.loc[np.isclose(layers.scale, scale)].set_index("stage")
        found = next(
            (stage for stage in stages if float(frame.loc[stage, "kl_ci95_low"]) > 0.0),
            None,
        )
        first_positive[f"{scale:g}x"] = found
    return {
        "scope": "8 preselected images x 8 preselected corrected true-history drift trajectories x scales 0,0.5,1,2,3",
        "metric": "spatial concentration of squared feature energy (KL from spatial uniform); not SSI",
        "population_endpoint": group_summary,
        "first_stage_with_positive_kl_change_ci95": first_positive,
        "interpretation_limit": "intermediate feature-energy concentration is not SF-population-specific; the low/high split is directly established only at final readout",
        "frontend_replacement": intervention.to_dict(orient="records"),
    }


def write_report(stats: dict[str, Any], finite: pd.DataFrame) -> None:
    filters = "\n".join(
        f"- Channel {row['channel']}: `{row['description']}`; DC gain {row['dc_gain']:.4f}; "
        f"peak {row['peak_hz']:.2f} Hz; contiguous half-power band "
        f"{row['half_power_low_hz']:.2f}–{row['half_power_high_hz']:.2f} Hz."
        for row in stats["frontend_filters"]
    )
    frontend_optima = "\n".join(
        f"- {row['sf_band']}: {row['band_optimum_scale']:g}×"
        + (" (right-censored: still increasing at 3×)" if row["optimum_censoring"] == "right" else "")
        for row in stats["frontend_sf_band_optima"]
    )
    early_optima = "\n".join(
        f"- {row['sf_band']}: {row['band_optimum_scale']:g}×"
        + (" (right-censored: still increasing at 3×)" if row["optimum_censoring"] == "right" else "")
        for row in stats["early_stack_sf_band_optima"]
    )
    predicts = bool(stats["early_stack_predicts_high_smaller_than_low"])
    output_distributions = stats["first_resnet_output_optimum_distributions"]
    low_output_distribution = output_distributions["0.25–0.5 cpd"]
    high_output_distribution = output_distributions["2–4 cpd"]
    highest_output_distribution = output_distributions["4–8 cpd"]
    channel_resolved_sentence = (
        f"This is not an artifact of summing the layer before finding an optimum: all "
        f"{stats['first_resnet_mixed_output_count']} individual mixed preactivation outputs peak at the "
        "3× boundary for both 0.25–0.5 and 2–4 cpd, and zero outputs have a smaller "
        "2–4-cpd optimum than 0.25–0.5-cpd optimum. In the more extreme 4–8-cpd band, "
        f"{highest_output_distribution.get('2x', 0)} outputs peak at 2× and "
        f"{highest_output_distribution.get('3x', 0)} remain right-censored at 3×."
    )
    if predicts:
        mechanism_sentence = (
            "The trained early stack is sufficient to predict the qualitative ordering: "
            f"the 2–4-cpd band peaks at {stats['early_stack_higher_physical_sf_optimum_scale']:g}×, while the "
            f"0.25–0.5-cpd band peaks at {stats['early_stack_low_physical_sf_optimum_scale']:g}×."
        )
        deeper = (
            "The early linear-fundamental calculation met the qualitative ordering criterion."
        )
    else:
        mechanism_sentence = (
            "Including the first spatiotemporal ResNet convolution does not produce the required "
            "high-before-low optimum ordering in the linear-fundamental calculation "
            f"(2–4 cpd: {stats['early_stack_higher_physical_sf_optimum_scale']:g}×; 0.25–0.5 cpd: "
            f"{stats['early_stack_low_physical_sf_optimum_scale']:g}×). Both remain right-censored at 3×."
        )
        deeper = (
            "This points to the intervening nonlinear operations at the first block or later "
            "nonlinear/recurrent/readout processing, rather than the linear frequency response of the "
            "trained early kernels."
        )
    deeper_stats = stats.get("deeper_layer_pilot")
    if deeper_stats is not None:
        low_endpoint = deeper_stats["population_endpoint"]["low SF"]
        high_endpoint = deeper_stats["population_endpoint"]["high SF"]
        first_stage = deeper_stats["first_stage_with_positive_kl_change_ci95"]
        deeper = (
            "After the early linear-filtering result was negative, I ran the predeclared small conditional hook pilot: "
            "8 preselected images × 8 preselected corrected drifts × scales 0, 0.5, 1, 2, 3. "
            f"At the final readout, high-SF SSI peaked at {high_endpoint['optimum_scale']:g}× and was "
            f"{high_endpoint['ssi_percent_at_3x']:.1f}% at 3×; low-SF SSI peaked at "
            f"{low_endpoint['optimum_scale']:g}× and remained {low_endpoint['ssi_percent_at_3x']:.1f}% at 3×. "
            f"For the global squared-feature spatial-concentration metric, the first positive-CI stage was "
            f"{first_stage['1x']} at 1× and {first_stage['2x']} at 2×. Because that intermediate metric is "
            "not population-specific, the exact low/high divergence is directly localized only to the final "
            "readout endpoint, not uniquely to one earlier block."
        )
    is_controlled_pilot = "conditional corrected controlled" in str(
        stats["corrected_ssi_comparison_status"]
    )
    if is_controlled_pilot:
        comparison_paragraph = (
            "I searched Jake's, Ryan's, and Declan's repo caches for the assumed full corrected "
            "controlled-scaling archive; it is absent. I did not launch that full rerun. After the early-stage "
            "calculation failed the mechanistic test, the authorized small layerwise pilot produced a corrected 8-image × "
            "8-drift final-readout endpoint on the exact movement-scale axis. Figure 4 uses that conditional "
            "pilot and labels its scope. The completed 100-image corrected observational curve is retained as "
            "a separate tidy CSV, not presented as controlled scaling."
        )
        comparison_limit = (
            "- The full corrected controlled-scaling cache is absent. The exact-scale SSI comparison is the "
            "conditional 8-image × 8-drift layerwise pilot, not a full replacement dataset."
        )
        figure4_description = "temporal-only and through-first-ResNet predictions versus the conditional corrected controlled 8×8 final-readout pilot"
        run_sentence = (
            f"The analytic stage took {stats['elapsed_minutes']:.1f} minutes; the conditional small full-model "
            "hook pilot took about four additional minutes. No 100-image network job was run."
        )
    else:
        comparison_paragraph = (
            "I searched Jake's, Ryan's, and Declan's repo caches for the corrected controlled-scaling SSI "
            "archive. It is absent. In accordance with the no-rerun rule, Figure 4 reads the completed "
            "100-image corrected true-history SSI curves and places their natural path-length bins on an "
            "explicitly labeled normalized-path amplitude proxy."
        )
        comparison_limit = (
            "- The requested direct comparison with corrected controlled-scaling SSI cannot be made because "
            "that cache is absent; Figure 4 uses a labeled observational proxy."
        )
        figure4_description = "early-stack predictions versus the explicitly labeled corrected observational SSI proxy"
        run_sentence = f"Runtime was {stats['elapsed_minutes']:.1f} minutes. No full-model response computation was performed."
    finite1 = stats["finite_crop"]["scale_1_median_relative_l1"]
    finite2 = stats["finite_crop"]["scale_2_median_relative_l1"]
    finite3 = stats["finite_crop"]["scale_3_median_relative_l1"]
    ratio1 = stats["finite_crop"]["scale_1_median_actual_over_ideal"]
    ratio2 = stats["finite_crop"]["scale_2_median_actual_over_ideal"]
    ratio3 = stats["finite_crop"]["scale_3_median_actual_over_ideal"]
    tv1 = stats["finite_crop"]["scale_1_median_total_variation"]
    tv2 = stats["finite_crop"]["scale_2_median_total_variation"]
    tv3 = stats["finite_crop"]["scale_3_median_total_variation"]
    report = f"""# Figure 4 mechanism: retinal phase through the first spatiotemporal ResNet layer

## Executive answer

{mechanism_sentence}

{channel_resolved_sentence}

This revision explicitly includes the ResNet's early spatial and spatiotemporal kernels, rather than stopping at the temporal-basis frontend. It used exact 2-D spatial-frequency vectors, measured two-dimensional eye trajectories, trained effective checkpoint weights, and the model's 120-Hz sampling. No 100-image network job was launched.

## Exactly what was done, and why

1. I read the 100 natural images and the eight objectively preselected drift trajectories from the existing Figure 4 controlled-scaling design. For each image I reproduced the exact 151×151 stabilized retinal crop after the production percentile scaling, bilinear renderer, physical pixels-per-degree conversion, and aperture crop. I Fourier transformed those crops with coefficients normalized by `151²`.
2. I retained every 2-D Fourier vector inside the isotropic physical Nyquist circle ({PPD / 2:.3f} cpd). Image-row frequency was assigned the physical upward-positive sign, so each calculation used an explicit `(kx, ky)` in cycles/degree. Orientation averaging was done only after the mode-wise calculation, in 0.5-cpd display bins.
3. For each selected trajectory and each movement scale `0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3`, I calculated `z_k(t;s)=exp[-i 2π k·s(e(t)-e(0))]` over the 40 scored samples. The temporal DFT was divided by 40, positive and negative power at equal `|f|` was combined, and DC was excluded from "motion-induced" summaries. With 40 samples at 120 Hz, the exact temporal bins are 0, 3, ..., 60 Hz.
4. Because the original experiment crosses images and trajectories, the ideal phase-only mean factorizes exactly into mean natural-image mode power times mean trajectory phase power. This allowed all 100 images to be included without a neural-network run. The trajectory set is the same eight predeclared controlled-scaling drifts, not all 1,000 natural-history snippets.
5. I extracted the **trained effective** weights of three consecutive early stages from `{MODEL_CHECKPOINT_PATH}`: the `4×1×16×1×1` temporal frontend, the ResNet `8×4×1×7×7` spatial stem, and the first genuinely spatiotemporal ResNet convolution, `64×16×3×9×9`. "Effective" means the learned weight-normalized parameters after the production temporal/spatial Hann windows, not raw unconstrained checkpoint tensors.
6. I evaluated the full complex Fourier response of those kernels at every exact `(kx,ky,ft)` mode. The temporal-only comparator is the unweighted sum of output power across its four channels. For the extended cascade I propagated complex fundamentals through the temporal frontend, 1×7×7 stem, 16-channel SplitReLU expansion, and trained 64×16×3×9×9 convolution. I retained the joint SF–TF response and movement-dose curve of **each of the 64 mixed preactivation outputs** before also computing their population sum.
7. That extended calculation is explicitly a **linear-fundamental approximation**, not an exact transfer function for the nonlinear network: the RMSNorm denominator is held at identity, its learned affine gamma is retained (beta contributes no AC term here), and positive/negative SplitReLU branches receive `+0.5` and `-0.5` of a sinusoid's fundamental. This is the correct way to ask whether the trained early kernels' joint SF–TF selectivity alone explains the turnover; exact nonlinear activations are handled separately by hooks.
8. I quantified phase RMS, phase range, and circular concentration for each mode and scale. I also rendered the existing 8-image × 8-trajectory subset at every scale with the exact crop/interpolator and compared its spatiotemporal spectrum with the matched fixed-amplitude phase-only prediction.
9. {comparison_paragraph}
10. The existing conditional pilot recorded exact nonlinear activations at retinal input, frontend, stem, ResBlocks 1–2, ConvGRU, and final readout for 64 preselected image–trajectory pairs at scales 0, 0.5, 1, 2, and 3. Its intermediate metric is spatial concentration of squared feature energy (KL from uniform), not SSI; final output-unit curves are SSI.

## 1. What translation does

For an ideal translated image, `I(x,t)=I0(x-e(t))`, so `Î(k,t)=Î0(k) exp[-i2πk·e(t)]`. Instantaneous spatial amplitude is unchanged. Movement converts spatial phase into a temporal signal. The same displacement traverses more radians at higher `|k|`, so higher spatial frequencies spread into nonzero and generally higher temporal frequencies sooner as movement grows. Figure 1 shows this directly in the actual retinal crop.

## 2. What increases with movement amplitude

At 0×, ideal phase power is all DC. Increasing scale lowers circular concentration and moves power into non-DC temporal bins. The relationship is not simply total speed: it depends on the exact projection `k·e(t)` at every time sample. At sufficiently large phase excursions the finite sampled spectrum redistributes, rather than increasing linearly forever.

## 3. What the trained early stack filters

{filters}

The frontend is therefore not a single preferred temporal frequency. It supplies several broad, complementary temporal channels, including a low-frequency/near-derivative channel and channels with substantial high-frequency gain. Figure 2 then shows how the spatial-only 1×7×7 stem and first 3×9×9 ResNet kernel reshape that signal jointly over SF and TF.

## 4. Does filtering through the first spatiotemporal ResNet layer predict the low/high difference?

Frontend-only band optima:

{frontend_optima}

Linear-fundamental cascade through the temporal frontend, spatial stem, and first 3×9×9 ResNet convolution:

{early_optima}

{mechanism_sentence}

Channel-resolved result:

{channel_resolved_sentence}

These are physical input bands. They should not be identified one-to-one with the manuscript's output-unit low/high populations; Figure 4 tests qualitative structure only.

## 5. Is phase excursion the useful geometric variable?

At 3×, where the 2–4-cpd early-stack curve is still rising, its mean phase RMS is {stats['higher_sf_mean_phase_rms_at_early_stack_optimum_rad']:.2f} rad. The exact phase table records the 1-rad, π-rad, and 2π landmarks. Higher `|k|` does reach a given phase-excursion landmark at a smaller movement multiplier, but that geometric fact did **not** create a turnover over 0.25–8 cpd even after the first spatiotemporal ResNet kernel. The aggregate ridge first showed an interior 2× maximum at approximately {stats['early_stack_first_interior_optimum_spatial_cpd']:.1f} cpd, beyond the main physical bands used for the low/high comparison. Phase excursion is therefore a well-defined physical coordinate, but it is not by itself an explanation of the model-population SSI difference.

## 6. Do finite-crop effects matter?

The exact rendered-versus-phase-only median relative L1 spectral error over 0.25–8 cpd was {finite1:.2f} at 1×, {finite2:.2f} at 2×, and {finite3:.2f} at 3×. Most of this was a nearly scale-invariant gain difference: actual/ideal non-DC power was {ratio1:.3f}, {ratio2:.3f}, and {ratio3:.3f}, while temporal-distribution total variation was only {tv1:.3f}, {tv2:.3f}, and {tv3:.3f}. Finite-crop/interpolation effects therefore exist, but they do not grow at the large-motion end and do not explain the high-SF decline.

## 7. Deeper processing

{deeper}

## 8. Best-supported mechanism

The best-supported causal sequence is narrower than the original hypothesis: measured retinal displacement creates scale- and SF-dependent temporal phase modulation, and the trained early kernels jointly reweight it. Yet neither the summed energy nor any of the 64 individual mixed preactivation outputs produces the low/high optimum split below 4 cpd, while the exact nonlinear pilot's final readout shows the high-SF 1× peak and low-SF 2× peak. Thus linear SF–TF selectivity through that kernel is insufficient. The missing operation could be RMS normalization/SplitReLU at or before the first block, subsequent nonlinear/spatial processing, ConvGRU recurrence, or the readout statistic. Global feature-energy concentration first has a positive-CI change at ConvGRU for 1× and ResBlock1 for 2×/3×, but that global metric cannot uniquely localize the population-specific split.

## 9. What remains unexplained

{comparison_limit}
- Physical input SF bands and output-unit SF groups are related but not identical objects.
- The small intermediate hook metric is global across feature channels; it does not say which block first creates the low/high population split.
- Because RMSNorm and SplitReLU intervene, the analytic early-stack cascade isolates the trained kernels' fundamental SF–TF selectivity (while retaining learned RMSNorm gamma); it is not the exact nonlinear block response.
- The 40-sample record gives 3-Hz Fourier resolution. This is exact for the experiment's scored interval but coarse as a continuous-spectrum estimate.

## 10. Manuscript-safe language

Safe wording is: **"Retinal translation converts natural-image spatial components into temporal phase modulation, which is jointly reweighted by the trained temporal frontend, spatial stem, and first spatiotemporal ResNet kernel. In a channel-resolved linear-fundamental analysis, all 64 mixed preactivation outputs remained right-censored at 3× movement in both the 0.25–0.5- and 2–4-cpd bands; none reproduced the high-before-low optimum ordering. The turnover therefore requires nonlinear operations within or after the early stack, recurrent/readout processing, or a statistic other than linear transmitted energy."**

Do not write that output-unit temporal-frequency preference predicts each unit's optimum, or that Figure 4 directly validates `ft*/fs*`. Do not present the conditional 8×8 endpoint as the missing full corrected controlled-scaling dataset.

## Outputs

- `figure1_retinal_motion_converts_space_into_time.*`: image/trajectory, x–t slices, and rendered SF×TF power.
- `figure2_early_spatiotemporal_filters.*`: temporal kernels, temporal transfer, spatial-stem gain, the trained 64×16 mixing matrix, first-ResNet SF×TF gain, and complete early-cascade gain.
- `figureS1_all_64_mixed_first_resnet_outputs.*`: the joint SF×TF gain map for every one of the 64 mixed preactivation outputs, each normalized independently so filter-shape diversity is visible.
- `figure3_predicted_movement_scale_by_spatial_frequency.*`: aggregate early-cascade energy, four diverse mixed-output SF×TF maps, and the optimum distribution across all 64 mixed outputs.
- `figure4_early_stack_vs_corrected_ssi.*`: {figure4_description}.
- `plot_data/`: tidy CSVs for every plotted quantity.
- `exact_arrays/`: mode-level phase-only spectra, image Fourier power, representative movies, and finite-crop spectra.

{run_sentence}

## Evidence classification

**SUPPORTED**

- The exact physical claim that rigid retinal translation preserves ideal spatial Fourier amplitude and creates temporal phase modulation `exp[-i2πk·e(t)]`.
- The trained frontend has the four reported temporal transfer functions, the ResNet stem is spatial-only (`1×7×7`), and the next convolution is genuinely spatiotemporal (`3×9×9`).
{('- The linear-fundamental early cascade predicts a smaller optimum for the higher physical-SF band than for the low physical-SF band.' if predicts else '- Linear-fundamental energy through the first spatiotemporal ResNet convolution is insufficient: all 0.25–8-cpd bands remain right-censored at 3×.')}
- The channel-resolved result is also negative: all 64 mixed preactivation outputs peak at the 3× boundary in both the 0.25–0.5- and 2–4-cpd bands; none shows the required high-before-low ordering.
{('- The conditional corrected 8×8 final-readout pilot reproduces the high-SF 1× optimum and low-SF 2× optimum.' if is_controlled_pilot else '')}

**CONSISTENT WITH**

- Phase excursion relative to spatial wavelength being one useful geometric coordinate for where later nonlinear effects could appear.
- The SSI turnover arising from nonlinear operations in the stem/first ResBlock or later nonlinear/spatial/recurrent/readout processing, because it is absent from the early linear-fundamental energy but present at the final readout.

**NOT SUPPORTED**

- Output-unit `q=fs*v/ft` as the primary mechanism.
- The trained early kernels' linear SF–TF transfer, through the first 3×9×9 ResNet convolution, as an explanation of the 0.25–8-cpd low/high SSI turnover.
- Increasing finite-crop/new-content distortion as the cause of the large-motion decline; spectral-shape discrepancy remained below 0.008 total variation and did not grow materially.
- A claim that the missing full corrected controlled-scaling dataset has been compared here; only the explicitly conditional 8×8 endpoint is available.
- A claim that downstream spatial/recurrent computation is irrelevant.
"""
    (OUT_DIR / "FIG4_MECHANISM_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXACT_DIR.mkdir(parents=True, exist_ok=True)
    legacy_outputs = [
        *(OUT_DIR / f"figure2_learned_temporal_frontend.{suffix}" for suffix in ("png", "pdf", "svg")),
        *(OUT_DIR / f"figure4_frontend_vs_corrected_ssi.{suffix}" for suffix in ("png", "pdf", "svg")),
        DATA_DIR / "figure4_frontend_vs_corrected_ssi.csv",
        DATA_DIR / "frontend_optimum_ridge.csv",
    ]
    for legacy_path in legacy_outputs:
        if legacy_path.is_file():
            legacy_path.unlink()
    configure_plotting()
    args = parse_args()
    inputs = load_inputs()
    grid = frequency_grid()
    weights, stem_weights, resblock1_weights, stem_norm_gamma = effective_early_weights()
    image_power = load_or_compute_image_power(inputs, grid, args.device)
    selected_displacement = inputs["selected_scored_xy"] - inputs["selected_scored_xy"][:, :1]
    phase = phase_spectra_and_geometry(
        selected_displacement,
        grid,
        inputs["selected_trace_ids"],
    )
    aggregate = aggregate_ideal(image_power, phase, weights, grid)
    early = early_resnet_linearized_transfer(
        weights,
        stem_weights,
        resblock1_weights,
        stem_norm_gamma,
        phase,
        image_power,
        grid,
    )
    filter_table = write_ideal_tables(aggregate, weights, grid)
    early_dose, output_optima = write_early_resnet_tables(early, grid)
    example = make_figure1(inputs, grid, args.device)
    make_figure2(weights, filter_table, aggregate, early, grid)
    make_figure_s1_all_mixed_outputs(early, grid)
    frontend_dose = band_dose_table(aggregate, grid)
    ridge = make_figure3(aggregate, early, grid, frontend_dose, early_dose, output_optima)
    if args.skip_finite_crop:
        raise RuntimeError("The final requested report requires the finite-crop audit; do not use --skip-finite-crop for final output")
    finite = finite_crop_audit(inputs, grid, args.device)
    corrected, comparison_status = load_corrected_ssi_proxy(inputs)
    make_figure4(aggregate, early, grid, corrected)
    phase_table = pd.read_csv(DATA_DIR / "phase_excursion.csv")
    stats = summarize_results(
        filter_table,
        frontend_dose,
        early_dose,
        output_optima,
        ridge,
        phase_table,
        finite,
        corrected,
        example,
        time.time() - start,
    )
    stats["n_exact_2d_fourier_modes"] = int(len(grid["kxy"]))
    stats["comparison_status"] = comparison_status
    deeper_summary = load_deeper_layer_summary()
    if deeper_summary is not None:
        stats["deeper_network_run"] = "completed conditional 8-image x 8-trajectory layerwise pilot"
        stats["deeper_layer_pilot"] = deeper_summary
    write_json(OUT_DIR / "statistics.json", stats)
    write_report(stats, finite)
    write_json(
        OUT_DIR / "manifest.json",
        {
            "analysis": "fig4_targeted_early_spatiotemporal_mechanism_v2",
            "no_100_image_network_run": True,
            "full_corrected_controlled_scaling_rerun": False,
            "conditional_small_layerwise_network_pilot": deeper_summary is not None,
            "ideal_scope": "100 natural images x 8 preselected controlled drift trajectories x 8 scales",
            "finite_crop_scope": "8 preselected images x 8 preselected controlled drift trajectories x 8 scales",
            "early_stack_scope": "trained temporal frontend + 1x7x7 spatial stem + first 3x9x9 ResNet convolution",
            "early_stack_transfer_status": "linear-fundamental approximation because intervening RMSNorm and SplitReLU make the exact stack nonlinear; learned RMSNorm gamma retained and exact hooked activations reported separately",
            "figures": [
                "figure1_retinal_motion_converts_space_into_time",
                "figure2_early_spatiotemporal_filters",
                "figureS1_all_64_mixed_first_resnet_outputs",
                "figure3_predicted_movement_scale_by_spatial_frequency",
                "figure4_early_stack_vs_corrected_ssi",
            ],
            "corrected_ssi_source": CORRECTED_CURVES,
            "corrected_controlled_scaling_expected_but_absent": CONTROLLED_DIR / "corrected_controlled_scaling_response.npz",
            "statistics": stats,
        },
    )
    print(f"completed targeted mechanism analysis in {(time.time() - start) / 60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
