#!/usr/bin/env python3
"""Build a Rucci-style ensemble estimate of retinal SF×TF power.

The estimator follows the factorization used by Kuang et al. (2012),
``S(k, w) = I(k) Q(k, w)``. ``I`` is measured from real BackImage patches and
``Q`` from real intersaccadic eye-position snippets whose complete temporal
ordering is retained. One central snippet is retained per intersaccadic epoch
so long fixations are not overrepresented.
Motion strata are defined by within-animal speed thirds, and spectra are first
averaged within animal and then across animals.

This analysis is model-output blind. The tuning table supplies only the spatial
frequency grid used by the downstream Figure 4 overlay.
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
from scipy.signal.windows import dpss, tukey
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import (  # noqa: E402
    mode_to_grid_matrix,
)
from paper.fig4.spatiotemporal_tuning.run_retinal_power_visualization import (  # noqa: E402
    spatial_frequency_grid,
)
from paper.fig4.upstream.real_trace_matrix.core import (  # noqa: E402
    build_native_snippet_trace_bank,
    extract_patch,
    load_backimage_eyepos_by_session,
    load_source_rows,
    microsaccade_event_count,
)


EPS = np.finfo(np.float64).eps
DEFAULT_SOURCE = (
    ROOT
    / "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/"
    "backimage_image_fem_windows.csv"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-windows", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tuning-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-timepoints", type=int, default=40)
    parser.add_argument("--trace-rate-hz", type=float, default=120.0)
    parser.add_argument("--maximum-path-length-arcmin", type=float, default=350.0)
    parser.add_argument("--n-directions", type=int, default=8)
    parser.add_argument("--trace-chunk-size", type=int, default=256)
    parser.add_argument("--mode-chunk-size", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--aperture-size-px", type=int, default=151)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def central_epoch_rows(rows: pd.DataFrame, *, n_timepoints: int) -> pd.DataFrame:
    """Choose one source window nearest the center of every fixation epoch."""
    required = {
        "session",
        "trial_idx",
        "epoch_start_local",
        "epoch_stop_local",
        "local_start",
        "local_stop",
        "n_samples",
        "source_row",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"source fixation table is missing columns: {missing}")
    work = rows.loc[pd.to_numeric(rows.n_samples, errors="coerce") >= n_timepoints].copy()
    work["epoch_midpoint"] = 0.5 * (
        work.epoch_start_local.to_numpy(float)
        + work.epoch_stop_local.to_numpy(float)
    )
    work["window_midpoint"] = 0.5 * (
        work.local_start.to_numpy(float) + work.local_stop.to_numpy(float)
    )
    work["center_distance"] = np.abs(work.window_midpoint - work.epoch_midpoint)
    keys = ("session", "trial_idx", "epoch_start_local", "epoch_stop_local")
    return (
        work.sort_values(["center_distance", "source_row"], kind="mergesort")
        .drop_duplicates(list(keys))
        .sort_values(list(keys), kind="mergesort")
        .reset_index(drop=True)
    )


def trajectory_phase_power_per_trace(
    traces_xy_deg: np.ndarray,
    spatial_cpd: np.ndarray,
    *,
    frame_rate_hz: float,
    n_directions: int,
    trace_chunk_size: int = 256,
    mode_chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Q(k,w) for every complete measured trajectory.

    The translation carrier ``exp(-i 2π k·X(t))`` retains temporal order. The
    complex carrier is temporally demeaned, tapered with two DPSS windows, and
    folded across positive and negative temporal frequencies before radial
    direction averaging.
    """
    traces = np.asarray(traces_xy_deg, dtype=np.float64)
    spatial = np.asarray(spatial_cpd, dtype=np.float64)
    if traces.ndim != 3 or traces.shape[-1] != 2 or traces.shape[1] < 8:
        raise ValueError(f"traces must have shape [trace,time>=8,2], got {traces.shape}")
    if frame_rate_hz <= 0 or np.any(spatial <= 0) or n_directions < 2:
        raise ValueError("positive sampling rate/SF and at least two directions are required")
    directions = np.linspace(0.0, np.pi, int(n_directions), endpoint=False)
    kxy = np.stack(
        (
            spatial[:, None] * np.cos(directions),
            spatial[:, None] * np.sin(directions),
        ),
        axis=-1,
    ).reshape(-1, 2)
    n_time = traces.shape[1]
    signed_hz = np.fft.fftfreq(n_time, d=1.0 / float(frame_rate_hz))
    temporal_hz = np.fft.rfftfreq(n_time, d=1.0 / float(frame_rate_hz))[1:]
    fold_indices = [
        np.flatnonzero(np.isclose(np.abs(signed_hz), frequency))
        for frequency in temporal_hz
    ]
    tapers = dpss(n_time, NW=1.5, Kmax=2, sym=False).astype(np.float64)
    spectra = np.zeros(
        (len(traces), len(kxy), len(temporal_hz)), dtype=np.float32
    )
    for trace_start in range(0, len(traces), int(trace_chunk_size)):
        trace_stop = min(trace_start + int(trace_chunk_size), len(traces))
        trace_batch = traces[trace_start:trace_stop]
        for mode_start in range(0, len(kxy), int(mode_chunk_size)):
            mode_stop = min(mode_start + int(mode_chunk_size), len(kxy))
            dot = np.einsum(
                "md,ntd->nmt", kxy[mode_start:mode_stop], trace_batch, optimize=True
            )
            carrier = np.exp(-2j * np.pi * dot)
            carrier -= carrier.mean(axis=-1, keepdims=True)
            power = np.zeros(
                (len(trace_batch), mode_stop - mode_start, n_time),
                dtype=np.float64,
            )
            for taper_values in tapers:
                transformed = np.fft.fft(
                    carrier * taper_values[None, None, :], axis=-1, norm="ortho"
                )
                power += np.square(np.abs(transformed))
            power /= float(len(tapers))
            spectra[trace_start:trace_stop, mode_start:mode_stop] = np.stack(
                [power[:, :, indices].sum(axis=-1) for indices in fold_indices],
                axis=-1,
            ).astype(np.float32)
    radial = spectra.reshape(
        len(traces), len(spatial), len(directions), len(temporal_hz)
    ).mean(axis=2)
    return temporal_hz, radial


def trajectory_phase_power_budget_per_trace(
    traces_xy_deg: np.ndarray,
    spatial_cpd: np.ndarray,
    *,
    n_directions: int,
    trace_chunk_size: int = 256,
    mode_chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact TF=0 and TF>0 carrier-energy fractions.

    For each spatial Fourier mode, translation multiplies the image coefficient
    by the unit-modulus carrier ``exp(-i 2π k·X(t))``.  The squared magnitude
    of the carrier's temporal mean is therefore the TF=0 fraction, and its
    complement is the total fraction redistributed to nonzero temporal
    frequencies.  This untapered identity is exact and makes explicit that eye
    motion redistributes, rather than creates, complete movie energy.
    """
    traces = np.asarray(traces_xy_deg, dtype=np.float64)
    spatial = np.asarray(spatial_cpd, dtype=np.float64)
    if traces.ndim != 3 or traces.shape[-1] != 2 or traces.shape[1] < 2:
        raise ValueError(f"traces must have shape [trace,time>=2,2], got {traces.shape}")
    if np.any(spatial <= 0) or n_directions < 2:
        raise ValueError("positive SF and at least two directions are required")
    directions = np.linspace(0.0, np.pi, int(n_directions), endpoint=False)
    kxy = np.stack(
        (
            spatial[:, None] * np.cos(directions),
            spatial[:, None] * np.sin(directions),
        ),
        axis=-1,
    ).reshape(-1, 2)
    static = np.empty((len(traces), len(kxy)), dtype=np.float64)
    for trace_start in range(0, len(traces), int(trace_chunk_size)):
        trace_stop = min(trace_start + int(trace_chunk_size), len(traces))
        trace_batch = traces[trace_start:trace_stop]
        for mode_start in range(0, len(kxy), int(mode_chunk_size)):
            mode_stop = min(mode_start + int(mode_chunk_size), len(kxy))
            dot = np.einsum(
                "md,ntd->nmt", kxy[mode_start:mode_stop], trace_batch, optimize=True
            )
            carrier_mean = np.exp(-2j * np.pi * dot).mean(axis=-1)
            static[trace_start:trace_stop, mode_start:mode_stop] = np.square(
                np.abs(carrier_mean)
            )
    static = static.reshape(
        len(traces), len(spatial), len(directions)
    ).mean(axis=2)
    static = np.clip(static, 0.0, 1.0)
    dynamic = 1.0 - static
    return static, dynamic


def within_subject_metric_tails(
    subject: np.ndarray,
    metric: np.ndarray,
    *,
    tail_fraction: float,
    metric_name: str,
    metric_units: str,
    low_name: str,
    high_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Label equal lower/upper metric tails independently within each animal."""
    subject = np.asarray(subject).astype(str)
    value = np.asarray(metric, dtype=float)
    if subject.shape != value.shape:
        raise ValueError("subject and metric must be aligned one-dimensional arrays")
    if not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must be strictly between zero and one half")
    if not np.isfinite(value).all():
        raise ValueError("stratification metric must be finite")
    code = np.full(len(value), -1, dtype=np.int8)
    thresholds: dict[str, list[float]] = {}
    for animal in np.unique(subject):
        indices = np.flatnonzero(subject == animal)
        percentile = rankdata(value[indices], method="average") / (len(indices) + 1.0)
        code[indices[percentile <= float(tail_fraction)]] = 0
        code[indices[percentile >= 1.0 - float(tail_fraction)]] = 1
        thresholds[str(animal)] = [
            float(np.quantile(value[indices], float(tail_fraction))),
            float(np.quantile(value[indices], 1.0 - float(tail_fraction))),
        ]
        if not np.any(code[indices] == 0) or not np.any(code[indices] == 1):
            raise ValueError(f"{animal} has an empty requested metric tail")
    return code, {
        "rule": (
            f"{low_name} and {high_name} within-animal tails of {metric_name}; "
            "selection reads no neural or model response"
        ),
        "metric": metric_name,
        "metric_units": metric_units,
        "tail_fraction": float(tail_fraction),
        "regime_names": [low_name, high_name],
        "within_animal_tail_boundaries": thresholds,
    }


def within_subject_speed_strata(
    subject: np.ndarray, speed_deg_s: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Label the slowest and fastest thirds within each animal."""
    code, report = within_subject_metric_tails(
        subject,
        speed_deg_s,
        tail_fraction=1.0 / 3.0,
        metric_name="mean eye speed",
        metric_units="deg/s",
        low_name="slowest third",
        high_name="fastest third",
    )
    report["within_animal_speed_tertile_boundaries_deg_s"] = report[
        "within_animal_tail_boundaries"
    ]
    report["rule"] = (
        "slowest and fastest within-animal thirds by mean eye speed; "
        "selection reads no neural or model response"
    )
    return code, report


def animal_balanced_mean(
    values: np.ndarray, subject: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """Average observations within animal, then give animals equal weight."""
    value = np.asarray(values, dtype=np.float64)
    subject = np.asarray(subject).astype(str)
    keep = np.asarray(keep, dtype=bool)
    means = [value[keep & (subject == animal)].mean(axis=0) for animal in np.unique(subject)]
    if any(not np.isfinite(item).all() for item in means):
        raise ValueError("a requested motion stratum is empty for at least one animal")
    return np.mean(means, axis=0)


def select_balanced_image_rows(
    rows: pd.DataFrame, *, n_images: int, seed: int
) -> pd.DataFrame:
    """Sample image patches across each animal's observed contrast distribution."""
    work = rows.loc[rows.image_feature_ok.fillna(False)].copy()
    work["subject"] = work.session.astype(str).str.split("_").str[0]
    work = work.drop_duplicates(["session", "trial_idx"]).reset_index(drop=True)
    animals = sorted(work.subject.unique())
    if n_images < len(animals) or n_images % len(animals):
        raise ValueError("n_images must be a positive multiple of the animal count")
    rng = np.random.default_rng(int(seed))
    per_animal = n_images // len(animals)
    selected = []
    for animal in animals:
        group = work.loc[work.subject.eq(animal)].sort_values(
            "image_patch_rms_contrast", kind="mergesort"
        )
        if len(group) < per_animal:
            raise ValueError(f"{animal} has fewer than {per_animal} image trials")
        chunks = np.array_split(group.index.to_numpy(dtype=int), per_animal)
        selected.extend(int(rng.choice(chunk)) for chunk in chunks)
    return work.loc[selected].sort_values(["subject", "session", "trial_idx"]).reset_index(drop=True)


def average_image_radial_power(
    image_rows: pd.DataFrame,
    spatial_cpd: np.ndarray,
    *,
    aperture_size_px: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Measure animal-balanced radial power from actual BackImage patches."""
    spatial = np.asarray(spatial_cpd, dtype=float)
    orientations = np.linspace(0.0, 180.0, 12, endpoint=False)
    kxy, flat_index = spatial_frequency_grid(
        int(aperture_size_px), maximum_cpd=float(spatial.max())
    )
    distributor, _ = mode_to_grid_matrix(kxy, spatial, orientations)
    denominator = np.asarray(distributor.sum(axis=0)).ravel().reshape(
        len(spatial), len(orientations)
    ).sum(axis=1)
    window = tukey(int(aperture_size_px), alpha=0.15, sym=False)
    window_2d = np.outer(window, window)
    canvas_cache: dict = {}
    per_image = []
    subjects = []
    start = (540 - int(aperture_size_px)) // 2
    stop = start + int(aperture_size_px)
    for ordinal, (_, row) in enumerate(image_rows.iterrows(), start=1):
        patch, _ = extract_patch(row, canvas_cache=canvas_cache, patch_size_px=540)
        crop = np.asarray(patch[start:stop, start:stop], dtype=np.float64)
        contrast = (crop - 127.0) / 255.0
        contrast -= float(np.mean(contrast))
        coefficient = np.fft.fft2(contrast * window_2d, norm="ortho")
        mode_power = np.square(np.abs(coefficient.reshape(-1)[flat_index]))
        gridded = np.asarray(mode_power @ distributor).reshape(
            len(spatial), len(orientations)
        ).sum(axis=1)
        per_image.append(gridded / np.maximum(denominator, EPS))
        subjects.append(str(row.subject))
        # Rows are unique trials, so retaining full canvases only increases the
        # peak memory footprint without creating cache hits.
        canvas_cache.clear()
        if ordinal % 10 == 0 or ordinal == len(image_rows):
            print(f"BackImage radial power {ordinal}/{len(image_rows)}", flush=True)
    image_power = animal_balanced_mean(
        np.stack(per_image), np.asarray(subjects), np.ones(len(subjects), dtype=bool)
    )
    return image_power, {
        "n_images": int(len(image_rows)),
        "n_animals": int(image_rows.subject.nunique()),
        "animal_counts": {
            str(key): int(value)
            for key, value in image_rows.subject.value_counts().sort_index().items()
        },
        "selection": (
            "one patch per trial, sampled across each animal's observed RMS-contrast "
            "distribution; no neural or motion response used"
        ),
        "spatial_spectrum": "radial mean of Tukey-windowed 151-px BackImage FFT power",
    }


def render_diagnostic(
    path: Path,
    *,
    speed: np.ndarray,
    microsaccade: np.ndarray,
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    power: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(10.7, 3.45), gridspec_kw={"width_ratios": (0.65, 1, 1)},
        constrained_layout=True,
    )
    axes[0].hist(speed, bins=28, orientation="horizontal", color="0.72")
    axes[0].scatter(
        np.full(np.sum(microsaccade), axes[0].get_xlim()[1] * 0.96),
        speed[microsaccade], s=3, color="#D55E00", alpha=0.35,
    )
    axes[0].set(xlabel="count", ylabel="mean eye speed (deg/s)", title="fixation ensemble")
    normalized = power / max(float(np.max(power)), EPS)
    display = np.log10(np.maximum(normalized, 1e-6))
    levels = np.linspace(max(-5.0, float(np.quantile(display, 0.03))), 0.0, 13)
    for axis, name, value in zip(axes[1:], ("slowest third", "fastest third"), display):
        contour = axis.contourf(
            spatial_cpd, temporal_hz, value.T, levels=levels, cmap="magma", extend="both"
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set(xlabel="SF (cycles/deg)", title=name)
    axes[1].set_ylabel("TF (Hz)")
    figure.colorbar(contour, ax=list(axes[1:]), label="log10 ensemble power / shared peak")
    figure.suptitle("Real fixation distributions reformat average BackImage power")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.n_timepoints < 8 or args.trace_rate_hz <= 0:
        raise ValueError("valid trace length and rate are required")
    rows = load_source_rows(args.source_windows.resolve())
    epoch_rows = central_epoch_rows(rows, n_timepoints=int(args.n_timepoints))
    print(
        f"selected {len(epoch_rows)} central intersaccadic windows from {len(rows)} rows",
        flush=True,
    )
    sessions = epoch_rows.session.astype(str).unique().tolist()
    eye_by_session = load_backimage_eyepos_by_session(sessions)
    bank, bank_report = build_native_snippet_trace_bank(
        epoch_rows,
        eye_by_session,
        int(args.n_timepoints),
        dt=1.0 / float(args.trace_rate_hz),
        microsaccade_speed_threshold_dps=None,
        microsaccade_threshold_z=6.0,
        microsaccade_pad_frames=1,
    )
    bank = [
        item
        for item in bank
        if float(item["rendered_path_length_arcmin"])
        <= float(args.maximum_path_length_arcmin)
    ]
    print(f"retained {len(bank)} fixation epochs after path gate", flush=True)
    traces = np.stack([np.asarray(item["trace"], dtype=np.float32) for item in bank])
    speed = np.asarray([item["rendered_speed_mean_deg_s"] for item in bank], dtype=float)
    path_length = np.asarray(
        [item["rendered_path_length_arcmin"] for item in bank], dtype=float
    )
    microsaccade_count = np.asarray(
        [microsaccade_event_count(item) for item in bank], dtype=int
    )
    subject = np.asarray([str(item["session"]).split("_")[0] for item in bank])
    source_row = np.asarray([item["source_row"] for item in bank], dtype=int)
    tuning = pd.read_csv(args.tuning_table)
    spatial_cpd = np.sort(
        tuning.loc[tuning.temporal_hz.gt(0), "spatial_cpd"].unique().astype(float)
    )
    temporal_hz, per_trace_q = trajectory_phase_power_per_trace(
        traces,
        spatial_cpd,
        frame_rate_hz=float(args.trace_rate_hz),
        n_directions=int(args.n_directions),
        trace_chunk_size=int(args.trace_chunk_size),
        mode_chunk_size=int(args.mode_chunk_size),
    )
    print("finished trajectory redistribution spectra", flush=True)
    stratum_code, stratum_report = within_subject_speed_strata(subject, speed)
    q_power = np.stack(
        [
            animal_balanced_mean(per_trace_q, subject, stratum_code == code)
            for code in (0, 1)
        ]
    )
    image_rows = select_balanced_image_rows(
        rows, n_images=int(args.n_images), seed=int(args.seed)
    )
    print(f"selected {len(image_rows)} animal-balanced BackImage patches", flush=True)
    image_power, image_report = average_image_radial_power(
        image_rows,
        spatial_cpd,
        aperture_size_px=int(args.aperture_size_px),
    )
    power = q_power * image_power[None, :, None]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "rucci_ensemble_power.npz"
    np.savez_compressed(
        archive,
        spatial_cpd=spatial_cpd,
        temporal_hz=temporal_hz,
        power=power.astype(np.float32),
        q_power=q_power.astype(np.float32),
        image_radial_power=image_power.astype(np.float32),
        trace_xy=traces,
        speed_deg_s=speed.astype(np.float32),
        path_length_arcmin=path_length.astype(np.float32),
        microsaccade_count=microsaccade_count.astype(np.int16),
        subject=subject.astype("U16"),
        source_row=source_row,
        stratum_code=stratum_code,
    )
    stratum_rows = []
    for code, name in enumerate(("slowest third", "fastest third")):
        keep = stratum_code == code
        stratum_rows.append(
            {
                "name": name,
                "n_fixation_epochs": int(np.sum(keep)),
                "median_speed_deg_s": float(np.median(speed[keep])),
                "median_path_length_arcmin": float(np.median(path_length[keep])),
                "microsaccade_fraction": float(np.mean(microsaccade_count[keep] > 0)),
                "animal_counts": {
                    str(animal): int(np.sum(keep & (subject == animal)))
                    for animal in np.unique(subject)
                },
            }
        )
    summary = {
        "analysis": "Rucci-style real-fixation ensemble retinal power",
        "mechanism": "S(k,w) = mean_image I(k) * mean_fixation Q(k,w)",
        "source_windows": str(args.source_windows.resolve()),
        "tuning_grid_source": str(args.tuning_table.resolve()),
        "selection": (
            "one central native snippet per intersaccadic epoch; path-length gate; "
            "no model output used"
        ),
        "n_source_rows": int(len(rows)),
        "n_unique_intersaccadic_epochs_before_path_gate": int(len(epoch_rows)),
        "n_fixation_epochs": int(len(bank)),
        "n_animals": int(len(np.unique(subject))),
        "animal_counts": {
            str(animal): int(np.sum(subject == animal)) for animal in np.unique(subject)
        },
        "trace_rate_hz": float(args.trace_rate_hz),
        "trace_samples": int(args.n_timepoints),
        "fft_window_seconds": float(args.n_timepoints / args.trace_rate_hz),
        "maximum_path_length_arcmin": float(args.maximum_path_length_arcmin),
        "microsaccade_fraction": float(np.mean(microsaccade_count > 0)),
        "motion_distribution": {
            "speed_deg_s_quantiles": {
                str(q): float(np.quantile(speed, q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "path_length_arcmin_quantiles": {
                str(q): float(np.quantile(path_length, q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
        },
        "stratification": stratum_report,
        "strata": stratum_rows,
        "averaging": (
            "trajectory spectra averaged within animal then equally across animals; "
            "image spectra likewise animal-balanced"
        ),
        "image_ensemble": image_report,
        "trajectory_spectrum": (
            "complete-order translation carrier; temporal mean removal; DPSS NW=1.5 K=2; "
            "positive/negative TF folded; radial direction average"
        ),
        "bank_builder": bank_report,
        "archive": str(archive.resolve()),
        "diagnostic_figure": str((args.out_dir / "rucci_ensemble_power.png").resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_diagnostic(
        args.out_dir / "rucci_ensemble_power.png",
        speed=speed,
        microsaccade=microsaccade_count > 0,
        spatial_cpd=spatial_cpd,
        temporal_hz=temporal_hz,
        power=power,
    )
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
