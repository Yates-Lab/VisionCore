#!/usr/bin/env python3
"""Build the production Kuang/Rucci retinal-power ensemble for Figure 4.

The estimator is the analytic factorization ``S(k,w) = I(k) Q(k,w)``. ``I`` is
the radial spectrum of real BackImage patches and ``Q`` is the complete-order
translation-carrier spectrum of real DDPI fixation traces.  The traces must
come from the separately audited, continuously zero-phase-filtered 240-Hz
bank; this module never reconstructs or filters eye position itself.

The two displayed regimes are response-blind within-animal quartiles of the
geometric temporal-frequency centroid.  Every trace is normalized to equal
integrated TF>0 power before regime averaging, so the maps compare *where*
motion places dynamic power rather than how much power it creates.  A separate
exact carrier budget verifies that TF=0 plus TF>0 energy remains one at every
spatial frequency.
"""
from __future__ import annotations

import argparse
import hashlib
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

from paper.fig4.spatiotemporal_tuning.retinal_replay import (  # noqa: E402
    evenly_spaced_rows,
)
from paper.fig4.spatiotemporal_tuning.spectral_power import (  # noqa: E402
    mode_to_grid_matrix,
    spatial_frequency_grid,
)
from paper.fig4.upstream.real_trace_matrix.core import (  # noqa: E402
    extract_patch,
)
from paper.fig4.spatiotemporal_tuning.eye_trace_filter import filter_contract_valid, filter_qc_passed


EPS = np.finfo(np.float64).eps
PRODUCTION_N_SPATIAL = 25
PRODUCTION_MAX_SPATIAL_CPD = 12.0
PRODUCTION_N_ORIENTATIONS = 8


def json_ready(value):
    """Recursively convert NumPy scalar metadata to strict JSON primitives."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, required=True)
    parser.add_argument("--fixation-bank", type=Path, required=True)
    parser.add_argument("--tuning-table", type=Path, required=True)
    parser.add_argument(
        "--spectrum-validation-summary",
        type=Path,
        required=True,
        help="Passed periodic-identity and finite-renderer validation summary.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument(
        "--n-traces", type=int, default=0, help="0 uses the complete filtered bank."
    )
    parser.add_argument("--analysis-samples", type=int, default=240)
    parser.add_argument("--n-directions", type=int, default=12)
    parser.add_argument("--trace-chunk-size", type=int, default=256)
    parser.add_argument("--mode-chunk-size", type=int, default=64)
    parser.add_argument("--aperture-size-px", type=int, default=151)
    return parser.parse_args()


def radial_temporal_quadrature(
    spatial_cpd: np.ndarray, temporal_hz: np.ndarray
) -> np.ndarray:
    """Integration weights for a radial 2-D SF spectrum over positive TF."""
    spatial = np.asarray(spatial_cpd, dtype=np.float64)
    temporal = np.asarray(temporal_hz, dtype=np.float64)
    if spatial.ndim != 1 or temporal.ndim != 1 or len(spatial) < 2 or len(temporal) < 2:
        raise ValueError("spatial and temporal grids must be one-dimensional")
    if np.any(np.diff(spatial) <= 0) or np.any(np.diff(temporal) <= 0):
        raise ValueError("spatial and temporal grids must be strictly increasing")
    return (
        2.0
        * np.pi
        * spatial[:, None]
        * np.gradient(spatial)[:, None]
        * np.gradient(temporal)[None, :]
    )


def production_spectral_grid(tuning_table: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the locked dense power grid inside the measured assay support.

    The grating assay remains on its deliberately low-resolution acquired grid.
    Kuang power is an analytic continuous-frequency calculation, so Panels C/F
    use a denser integration/display grid without interpolating measured neural
    responses.  The lower bound is read from the exact assay and the 12-cpd
    upper bound lies within its tested support.
    """
    table = pd.read_csv(tuning_table, usecols=["spatial_cpd"])
    acquired = np.sort(table.spatial_cpd.dropna().unique().astype(float))
    if len(acquired) < 2 or np.any(acquired <= 0):
        raise ValueError("tuning table has no valid positive SF support")
    lower = float(acquired[0])
    upper = float(PRODUCTION_MAX_SPATIAL_CPD)
    if upper > float(acquired[-1]):
        raise ValueError(
            f"locked {upper:g}-cpd power grid exceeds acquired support {acquired[-1]:g}"
        )
    spatial = np.geomspace(lower, upper, PRODUCTION_N_SPATIAL)
    orientation = np.linspace(
        0.0, 180.0, PRODUCTION_N_ORIENTATIONS, endpoint=False
    )
    return spatial, orientation


def normalize_dynamic_power_per_trace(
    power_per_trace: np.ndarray,
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each TF>0 map and return mass and geometric TF centroid."""
    power = np.asarray(power_per_trace, dtype=np.float64)
    weights = radial_temporal_quadrature(spatial_cpd, temporal_hz)
    if power.ndim != 3 or power.shape[1:] != weights.shape:
        raise ValueError(
            "power_per_trace must have shape [trace, spatial, temporal] matching grids"
        )
    if np.any(power < 0) or not np.isfinite(power).all():
        raise ValueError("dynamic power must be finite and nonnegative")
    mass = np.sum(power * weights[None, :, :], axis=(1, 2))
    if np.any(mass <= EPS):
        raise ValueError("every trace must contain positive TF>0 power")
    distribution = power / mass[:, None, None]
    log_centroid = np.sum(
        distribution
        * weights[None, :, :]
        * np.log2(np.asarray(temporal_hz, dtype=np.float64))[None, None, :],
        axis=(1, 2),
    )
    return distribution, mass, 2.0**log_centroid


def balanced_trace_rows(table: pd.DataFrame, requested: int) -> np.ndarray:
    """Select a deterministic subset spanning each animal's speed distribution."""
    if requested <= 0 or requested >= len(table):
        return np.arange(len(table), dtype=int)
    subject = table.session.astype(str).str.split("_").str[0].to_numpy()
    selected: list[int] = []
    animals = sorted(np.unique(subject))
    base, extra = divmod(int(requested), len(animals))
    for ordinal, animal in enumerate(animals):
        rows = np.flatnonzero(subject == animal)
        order = np.argsort(
            table.iloc[rows].analysis_speed_mean_deg_s.to_numpy(float),
            kind="mergesort",
        )
        rows = rows[order]
        count = min(len(rows), base + int(ordinal < extra))
        if count:
            selected.extend(rows[evenly_spaced_rows(len(rows), count)].tolist())
    if len(selected) != requested:
        raise RuntimeError(
            f"balanced trace selection returned {len(selected)} rows, expected {requested}"
        )
    return np.asarray(sorted(selected), dtype=int)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_defined_regimes(table: pd.DataFrame, subject: np.ndarray):
    """Select matched-duration event classes without consulting their spectra."""
    required = {"event_class", "event_coverage_pass", "unmatched_rapid_motion",
                "verified_microsaccade_count", "verified_microsaccade_max_amplitude_deg"}
    if missing := required.difference(table.columns):
        raise ValueError(f"event classification lacks audit fields: {sorted(missing)}")
    codes = table.event_class.map({"drift": 0, "microsaccade": 1, "excluded": -1})
    if codes.isna().any():
        raise ValueError("unrecognized event class")
    codes = codes.to_numpy(dtype=np.int8)
    for code in (0, 1):
        keep = codes == code
        if not table.loc[keep, "event_coverage_pass"].eq(True).all():
            raise ValueError("an event group contains unverified coverage")
        if not table.loc[keep, "unmatched_rapid_motion"].eq(False).all():
            raise ValueError("an event group contains unmatched rapid motion")
        if any(not np.any(keep & (subject == animal)) for animal in np.unique(subject)):
            raise ValueError("both event groups must contain each animal")
    micro = codes == 1
    if not ((table.loc[micro, "verified_microsaccade_count"] > 0).all()
            and table.loc[micro, "verified_microsaccade_max_amplitude_deg"].between(0, 1, inclusive="neither").all()
            and (table.loc[codes == 0, "verified_microsaccade_count"] == 0).all()):
        raise ValueError("drift/microsaccade labels disagree with the event audit")
    return codes, {
        "rule": "audited event-free drift windows versus windows containing verified microsaccades below 1 degree",
        "regime_names": ["drift", "microsaccades"],
        "selected_by_microsaccade_label": True,
        "selected_by_spectral_centroid": False,
        "excluded_n_traces": int(np.sum(codes < 0)),
        "animal_counts": {str(animal): [int(np.sum((subject == animal) & (codes == c))) for c in (0, 1)]
                          for animal in np.unique(subject)},
    }


def load_filtered_fixation_bank(
    directory: Path, *, analysis_samples: int, requested_traces: int
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and fail closed on the instrument-valid Figure 4 trace contract."""
    root = directory.resolve()
    manifest_path = root / "manifest.json"
    table_path = root / "trace_table.csv"
    trace_path = root / "trace_xy_filtered.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filter_spec = dict(manifest.get("filter", {}))
    gates = {
        "native_240_hz": np.isclose(float(manifest.get("target_rate_hz", np.nan)), 240.0),
        "zero_phase_filter": "zero-phase" in str(filter_spec.get("kind", "")),
        "declared_filter_contract": filter_contract_valid(filter_spec),
        "analysis_interval_available": int(manifest.get("analysis_samples", -1))
        >= int(analysis_samples),
    }
    if filter_spec.get("family") == "gaussian":
        gates["gaussian_filter_validation"] = filter_qc_passed(manifest)
    failed = sorted(name for name, passed in gates.items() if not bool(passed))
    if failed:
        raise ValueError(f"fixation-bank contract failed: {failed}")
    table = pd.read_csv(table_path).sort_values("trace_index").reset_index(drop=True)
    if int(requested_traces) > len(table):
        raise ValueError(
            f"requested {requested_traces} traces from a bank containing {len(table)}"
        )
    expected_index = np.arange(len(table), dtype=int)
    if not np.array_equal(table.trace_index.to_numpy(dtype=int), expected_index):
        raise ValueError("fixation trace_table is not in complete trace_index order")
    required = {
        "session",
        "analysis_speed_mean_deg_s",
        "analysis_path_length_deg",
        "saved_microsaccade_count",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"fixation trace_table is missing columns: {missing}")
    traces_all = np.load(trace_path, mmap_mode="r")
    if traces_all.ndim != 3 or traces_all.shape[-1] != 2 or len(traces_all) != len(table):
        raise ValueError(
            "filtered trace array and trace_table disagree: "
            f"traces={traces_all.shape}, rows={len(table)}"
        )
    rows = balanced_trace_rows(table, int(requested_traces))
    selected = np.asarray(traces_all[rows, -int(analysis_samples) :], dtype=np.float32)
    if selected.shape != (len(rows), int(analysis_samples), 2):
        raise ValueError(f"unexpected selected trace shape {selected.shape}")
    if not np.isfinite(selected).all():
        raise ValueError("filtered fixation bank contains non-finite coordinates")
    return manifest, table.iloc[rows].reset_index(drop=True), selected, rows


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
    path_length: np.ndarray,
    spatial_cpd: np.ndarray,
    temporal_hz: np.ndarray,
    regime_power: np.ndarray,
    regime_names: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(13.8, 3.5),
        gridspec_kw={"width_ratios": (0.78, 1, 1, 1)},
        constrained_layout=True,
    )
    axes[0].scatter(speed, path_length, s=16, color="0.55", alpha=0.6)
    axes[0].set(
        xlabel="mean eye speed (deg/s)",
        ylabel="path length (arcmin)",
        title="filtered fixation ensemble",
    )
    radial_display = (
        regime_power
        * (2.0 * np.pi * np.square(spatial_cpd))[None, :, None]
        * temporal_hz[None, None, :]
    )
    peak = max(float(np.max(radial_display)), EPS)
    positive = radial_display[radial_display > 0] / peak
    floor = max(float(np.quantile(positive, 0.02)), 1e-7)
    levels = np.linspace(np.log10(floor), 0.0, 13)
    contour = None
    for axis, name, value in zip(axes[1:3], regime_names, radial_display):
        contour = axis.contourf(
            spatial_cpd,
            temporal_hz,
            np.log10(np.maximum(value.T / peak, floor)),
            levels=levels,
            cmap="magma",
            extend="min",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set(xlabel="SF (cycles/deg)", title=str(name))
    axes[1].set_ylabel("TF (Hz)")
    ratio = np.log2(
        np.maximum(radial_display[1], EPS)
        / np.maximum(radial_display[0], EPS)
    )
    limit = min(4.0, max(1.0, float(np.quantile(np.abs(ratio), 0.98))))
    difference = axes[3].contourf(
        spatial_cpd,
        temporal_hz,
        ratio.T,
        levels=np.linspace(-limit, limit, 13),
        cmap="RdBu_r",
        extend="both",
    )
    axes[3].set_xscale("log", base=2)
    axes[3].set_yscale("log", base=2)
    axes[3].set(xlabel="SF (cycles/deg)", title="rapid / drift")
    if contour is not None:
        figure.colorbar(
            contour,
            ax=list(axes[1:3]),
            label="log10 equal-mass power / shared peak",
        )
    figure.colorbar(difference, ax=axes[3], label="log2 power ratio")
    figure.suptitle("Filtered real fixations redistribute natural-image power")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if int(args.analysis_samples) < 8:
        raise ValueError("analysis_samples must be at least eight")
    manifest, trace_table, traces, trace_rows = load_filtered_fixation_bank(
        args.fixation_bank,
        analysis_samples=int(args.analysis_samples),
        requested_traces=int(args.n_traces),
    )
    frame_rate_hz = float(manifest["target_rate_hz"])
    validation = json.loads(
        args.spectrum_validation_summary.read_text(encoding="utf-8")
    )
    if not bool(validation.get("all_claim_gates_pass", False)):
        raise RuntimeError("the supplied Kuang/direct-renderer validation did not pass")
    validation_contract = {
        "filtered_trace": validation.get("trace_kind") == "filtered",
        "native_240_hz": np.isclose(
            float(validation.get("frame_rate_hz", np.nan)), frame_rate_hz
        ),
        "same_analysis_interval": int(validation.get("analysis_samples", -1))
        == int(args.analysis_samples),
        "at_least_8_images": int(validation.get("n_images", 0)) >= 8,
        "at_least_8_traces": int(validation.get("n_traces", 0)) >= 8,
        "at_least_64_crossed_pairs": int(validation.get("n_pairs", 0)) >= 64,
    }
    failed_validation_contract = sorted(
        name for name, passed in validation_contract.items() if not bool(passed)
    )
    if failed_validation_contract:
        raise RuntimeError(
            "spectrum validation is a smoke result, not a production audit: "
            f"{failed_validation_contract}"
        )
    validation_inputs = dict(validation.get("inputs", {}))
    expected_hashes = {
        "fixation_bank_manifest_sha256": sha256(args.fixation_bank / "manifest.json"),
        "filtered_trace_sha256": sha256(args.fixation_bank / "trace_xy_filtered.npy"),
        "image_table_sha256": sha256(args.image_table),
        "tuning_table_sha256": sha256(args.tuning_table),
    }
    mismatched = sorted(
        name
        for name, expected in expected_hashes.items()
        if validation_inputs.get(name) != expected
    )
    if mismatched:
        raise RuntimeError(
            "spectrum validation does not cover the supplied production inputs: "
            f"{mismatched}"
        )
    spatial_cpd, orientation_deg = production_spectral_grid(args.tuning_table)
    speed = trace_table.analysis_speed_mean_deg_s.to_numpy(dtype=float)
    path_length = 60.0 * trace_table.analysis_path_length_deg.to_numpy(dtype=float)
    microsaccade_count = trace_table.get("verified_microsaccade_count", trace_table.saved_microsaccade_count).to_numpy(dtype=int)
    subject = trace_table.session.astype(str).str.split("_").str[0].to_numpy()
    temporal_hz, per_trace_q = trajectory_phase_power_per_trace(
        traces,
        spatial_cpd,
        frame_rate_hz=frame_rate_hz,
        n_directions=int(args.n_directions),
        trace_chunk_size=int(args.trace_chunk_size),
        mode_chunk_size=int(args.mode_chunk_size),
    )
    print("finished complete-order filtered trajectory spectra", flush=True)
    stratum_code, stratum_report = within_subject_speed_strata(subject, speed)
    kuang_q = np.stack(
        [
            animal_balanced_mean(per_trace_q, subject, stratum_code == code)
            for code in (0, 1)
        ]
    )
    images = pd.read_csv(args.image_table).sort_values("image_index").reset_index(drop=True)
    image_rows = evenly_spaced_rows(len(images), min(int(args.n_images), len(images)))
    selected_images = images.iloc[image_rows].copy()
    selected_images["subject"] = selected_images.session.astype(str).str.split("_").str[0]
    print(f"selected {len(selected_images)} production BackImage patches", flush=True)
    image_power, image_report = average_image_radial_power(
        selected_images,
        spatial_cpd,
        aperture_size_px=int(args.aperture_size_px),
    )
    kuang_power = kuang_q * image_power[None, :, None]
    per_trace_dynamic_power = per_trace_q * image_power[None, :, None]
    (
        per_trace_power_distribution,
        per_trace_dynamic_mass,
        per_trace_power_centroid_hz,
    ) = normalize_dynamic_power_per_trace(
        per_trace_dynamic_power, spatial_cpd, temporal_hz
    )
    if "event_class" in trace_table:
        regime_names = np.asarray(("drift", "microsaccades"), dtype="U32")
        spectral_regime_code, spectral_regime_report = event_defined_regimes(trace_table, subject)
        spectral_regime_report["event_classification"] = manifest["event_classification"]
    else:
        regime_names = np.asarray(("drift-rich quartile", "rapid-transient quartile"), dtype="U32")
        spectral_regime_code, spectral_regime_report = within_subject_metric_tails(
            subject, per_trace_power_centroid_hz, tail_fraction=0.25,
            metric_name="geometric TF centroid of the normalized dynamic spectrum",
            metric_units="Hz", low_name=str(regime_names[0]), high_name=str(regime_names[1]),
        )
    regime_power_distribution = np.stack(
        [
            animal_balanced_mean(
                per_trace_power_distribution,
                subject,
                spectral_regime_code == code,
            )
            for code in (0, 1)
        ]
    )
    quadrature = radial_temporal_quadrature(spatial_cpd, temporal_hz)
    regime_integrals = np.sum(
        regime_power_distribution * quadrature[None, :, :], axis=(1, 2)
    )
    if not np.allclose(regime_integrals, 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "shape-normalized motion-regime maps do not integrate to one: "
            f"{regime_integrals.tolist()}"
        )
    carrier_static_per_trace, carrier_dynamic_per_trace = (
        trajectory_phase_power_budget_per_trace(
            traces,
            spatial_cpd,
            n_directions=int(args.n_directions),
            trace_chunk_size=int(args.trace_chunk_size),
            mode_chunk_size=int(args.mode_chunk_size),
        )
    )
    stratum_static_fraction = np.stack(
        [
            animal_balanced_mean(
                carrier_static_per_trace, subject, stratum_code == code
            )
            for code in (0, 1)
        ]
    )
    stratum_dynamic_fraction = np.stack(
        [
            animal_balanced_mean(
                carrier_dynamic_per_trace, subject, stratum_code == code
            )
            for code in (0, 1)
        ]
    )
    conservation_error = float(
        np.max(np.abs(stratum_static_fraction + stratum_dynamic_fraction - 1.0))
    )
    if conservation_error >= 1e-12:
        raise RuntimeError(
            "the exact trajectory carrier budget does not conserve power: "
            f"maximum error={conservation_error:.3g}"
        )
    all_traces = np.ones(len(subject), dtype=bool)
    full_q = animal_balanced_mean(per_trace_q, subject, all_traces)[None]
    full_power = full_q * image_power[None, :, None]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "rucci_ensemble_power.npz"
    payload = dict(
        spatial_cpd=spatial_cpd,
        temporal_hz=temporal_hz,
        orientation_deg=orientation_deg,
        power=kuang_power.astype(np.float32),
        kuang_power=kuang_power.astype(np.float32),
        kuang_q=kuang_q.astype(np.float32),
        image_radial_power=image_power.astype(np.float32),
        carrier_static_fraction_per_trace=carrier_static_per_trace.astype(np.float32),
        carrier_dynamic_fraction_per_trace=carrier_dynamic_per_trace.astype(np.float32),
        stratum_static_fraction=stratum_static_fraction.astype(np.float32),
        stratum_dynamic_fraction=stratum_dynamic_fraction.astype(np.float32),
        condition_names=np.asarray(("full_filtered",), dtype="U32"),
        condition_power=full_power.astype(np.float32),
        condition_q=full_q.astype(np.float32),
        per_trace_dynamic_power_mass=per_trace_dynamic_mass.astype(np.float32),
        per_trace_power_centroid_hz=per_trace_power_centroid_hz.astype(np.float32),
        spectral_regime_names=regime_names,
        spectral_regime_selection=np.asarray("events" if "event_class" in trace_table else "centroid_quartiles"),
        spectral_regime_code=spectral_regime_code,
        spectral_regime_power_distribution=regime_power_distribution.astype(np.float32),
        spectral_regime_integrals=regime_integrals.astype(np.float32),
        trace_xy=traces,
        speed_deg_s=speed.astype(np.float32),
        path_length_arcmin=path_length.astype(np.float32),
        microsaccade_count=microsaccade_count.astype(np.int16),
        subject=subject.astype("U16"),
        trace_rows=trace_rows,
        stratum_code=stratum_code,
        image_rows=image_rows,
    )
    np.savez_compressed(archive, **payload)
    compatibility_archive = args.out_dir / "retinal_motion_ensemble.npz"
    np.savez_compressed(compatibility_archive, **payload)
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
        "analysis": "Kuang/Rucci natural-image retinal-motion ensemble",
        "primary_estimator": (
            "mean BackImage I(k) times animal-balanced complete-order filtered "
            "fixation phase Q(k,w)"
        ),
        "trace_source": str((args.fixation_bank / "trace_xy_filtered.npy").resolve()),
        "trace_filter": manifest["filter"],
        "frame_rate_hz": frame_rate_hz,
        "analysis_samples": int(args.analysis_samples),
        "n_fixation_epochs": int(len(traces)),
        "n_animals": int(len(np.unique(subject))),
        "animal_counts": {
            str(animal): int(np.sum(subject == animal)) for animal in np.unique(subject)
        },
        "tuning_grid_source": str(args.tuning_table.resolve()),
        "fft_window_seconds": float(args.analysis_samples / frame_rate_hz),
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
        "spectral_shape_regimes": {
            **spectral_regime_report,
            "n_fixation_epochs": [
                int(np.sum(spectral_regime_code == code)) for code in (0, 1)
            ],
            "median_power_centroid_hz": [
                float(
                    np.median(
                        per_trace_power_centroid_hz[spectral_regime_code == code]
                    )
                )
                for code in (0, 1)
            ],
            "median_eye_speed_deg_s": [
                float(np.median(speed[spectral_regime_code == code]))
                for code in (0, 1)
            ],
            "median_dynamic_power_mass_before_normalization": [
                float(np.median(per_trace_dynamic_mass[spectral_regime_code == code]))
                for code in (0, 1)
            ],
            "map_integrals_after_normalization": regime_integrals.tolist(),
            "normalization": (
                "each trace is divided by its integrated TF>0 power before "
                "animal-balanced averaging; each displayed regime therefore "
                "integrates to one and reports spectral shape, not scalar gain"
            ),
            "selection_reads_neural_or_model_responses": False,
        },
        "image_ensemble": image_report,
        "trajectory_spectrum": (
            "complete-order translation carrier; temporal mean removal; DPSS NW=1.5 K=2; "
            "positive/negative TF folded; radial direction average"
        ),
        "conditions": ["full_filtered"],
        "component_claim_boundary": (
            "Complete equal-duration trajectories, not a decomposition into additive motion components. "
            + ("Event-defined drift-only versus microsaccade-containing windows; no spectral-centroid selection."
               if "event_class" in trace_table else "No microsaccade label selects a regime.")
        ),
        "kuang_estimator": {
            "definition": (
                "factorial image-by-fixation expectation: mean radial BackImage I(k) "
                "times animal-balanced complete-order Q(k,w)"
            ),
            "instantaneous_velocity_used": False,
            "validation_summary": str(args.spectrum_validation_summary.resolve()),
            "validation_all_gates_pass": True,
            "validation_release_contract": validation_contract,
            "validation_input_hashes": expected_hashes,
        },
        "complete_power_budget": {
            "definition": (
                "for every spatial Fourier mode, TF=0 fraction is the squared "
                "magnitude of the translation carrier mean and TF>0 fraction is "
                "its complement"
            ),
            "maximum_static_plus_dynamic_error": conservation_error,
            "all_power_is_redistributed_not_created": True,
            "displayed_maps_omit_tf_zero": True,
        },
        "archive": str(archive.resolve()),
        "compatibility_archive": str(compatibility_archive.resolve()),
        "diagnostic_figure": str((args.out_dir / "rucci_ensemble_power.png").resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    render_diagnostic(
        args.out_dir / "rucci_ensemble_power.png",
        speed=speed,
        path_length=path_length,
        spatial_cpd=spatial_cpd,
        temporal_hz=temporal_hz,
        regime_power=regime_power_distribution,
        regime_names=regime_names,
    )
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
