#!/usr/bin/env python3
"""Audit the usable bandwidth of the 240-Hz DDPI fixation bank.

This command treats the existing 100/118-Hz trace as an instrument-level
anti-alias product, estimates the population PSD breakpoint, and evaluates
candidate zero-phase analysis filters.  It does not select a cutoff from how a
retinal-power or neural-response figure looks.
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
from scipy import signal


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.run_retinal_causal_chain import DEFAULT_CHAIN


EPS = np.finfo(np.float64).eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixation-bank", type=Path, default=DEFAULT_CHAIN / "fixation_bank")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-passbands-hz", type=float, nargs="+", default=(10, 15, 20, 30, 40, 60, 80, 100))
    parser.add_argument("--primary-passband-hz", type=float, default=20.0)
    parser.add_argument("--primary-stopband-hz", type=float, default=30.0)
    parser.add_argument("--analysis-start", type=int, default=60)
    return parser.parse_args()


def population_position_psd(
    traces_xy: np.ndarray, *, frame_rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(traces_xy, dtype=np.float64)
    frequency, power = signal.welch(
        value - value.mean(axis=1, keepdims=True),
        fs=float(frame_rate_hz),
        axis=1,
        nperseg=value.shape[1],
        detrend=False,
        window="hann",
    )
    return frequency, np.median(power.sum(axis=2), axis=0)


def piecewise_psd_breakpoint(
    frequency_hz: np.ndarray,
    power: np.ndarray,
    *,
    minimum_hz: float = 2.0,
    maximum_hz: float = 80.0,
    candidate_minimum_hz: int = 8,
    candidate_maximum_hz: int = 60,
) -> dict[str, float]:
    """Fit two log-log lines and return their minimum-SSE breakpoint."""
    frequency = np.asarray(frequency_hz, dtype=float)
    spectrum = np.asarray(power, dtype=float)
    valid = (
        (frequency >= float(minimum_hz))
        & (frequency <= float(maximum_hz))
        & np.isfinite(spectrum)
        & (spectrum > 0)
    )
    indices = np.flatnonzero(valid)
    x = np.full_like(frequency, np.nan, dtype=float)
    y = np.full_like(spectrum, np.nan, dtype=float)
    x[valid] = np.log2(frequency[valid])
    y[valid] = np.log2(np.maximum(spectrum[valid], EPS))
    candidates: list[tuple[float, float, float, float]] = []
    for breakpoint in range(int(candidate_minimum_hz), int(candidate_maximum_hz) + 1):
        left = indices[frequency[indices] <= breakpoint]
        right = indices[frequency[indices] >= breakpoint]
        if len(left) < 5 or len(right) < 5:
            continue
        low_fit = np.polyfit(x[left], y[left], 1)
        high_fit = np.polyfit(x[right], y[right], 1)
        error = float(
            np.square(y[left] - np.polyval(low_fit, x[left])).sum()
            + np.square(y[right] - np.polyval(high_fit, x[right])).sum()
        )
        candidates.append((error, float(breakpoint), float(low_fit[0]), float(high_fit[0])))
    if not candidates:
        raise RuntimeError("no valid PSD breakpoint candidates")
    error, breakpoint, low_slope, high_slope = min(candidates)
    return {
        "breakpoint_hz": breakpoint,
        "low_frequency_loglog_slope": low_slope,
        "high_frequency_loglog_slope": high_slope,
        "piecewise_sse": error,
    }


def zero_phase_filter(
    traces_xy: np.ndarray,
    *,
    frame_rate_hz: float,
    passband_hz: float,
    stopband_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces_xy, dtype=np.float64)
    sos = signal.iirdesign(
        wp=float(passband_hz),
        ws=float(stopband_hz),
        gpass=0.1,
        gstop=60.0,
        fs=float(frame_rate_hz),
        output="sos",
    )
    center = np.median(traces, axis=1, keepdims=True)
    return center + signal.sosfiltfilt(sos, traces - center, axis=1), sos


def trace_metrics(traces_xy: np.ndarray, *, frame_rate_hz: float) -> dict[str, np.ndarray]:
    traces = np.asarray(traces_xy, dtype=float)
    step = np.linalg.norm(np.diff(traces, axis=1), axis=2)
    centered = traces - traces.mean(axis=1, keepdims=True)
    return {
        "mean_speed_deg_s": step.mean(axis=1) * float(frame_rate_hz),
        "peak_speed_deg_s": step.max(axis=1) * float(frame_rate_hz),
        "path_length_arcmin": step.sum(axis=1) * 60.0,
        "rms_radius_arcmin": np.sqrt(np.mean(np.sum(np.square(centered), axis=2), axis=1)) * 60.0,
    }


def candidate_table(
    traces_xy: np.ndarray,
    *,
    frame_rate_hz: float,
    passbands_hz: np.ndarray,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    traces = np.asarray(traces_xy, dtype=float)
    reference = trace_metrics(traces, frame_rate_hz=frame_rate_hz)
    rows = []
    filtered: dict[float, np.ndarray] = {}
    for passband in np.asarray(passbands_hz, dtype=float):
        stopband = min(0.5 * frame_rate_hz - 2.0, max(passband + 10.0, 1.25 * passband))
        if passband >= stopband:
            continue
        value, _ = zero_phase_filter(
            traces,
            frame_rate_hz=frame_rate_hz,
            passband_hz=float(passband),
            stopband_hz=float(stopband),
        )
        filtered[float(passband)] = value
        metrics = trace_metrics(value, frame_rate_hz=frame_rate_hz)
        frequency, psd = population_position_psd(value, frame_rate_hz=frame_rate_hz)
        rows.append(
            {
                "passband_hz": float(passband),
                "stopband_hz": float(stopband),
                "median_speed_deg_s": float(np.median(metrics["mean_speed_deg_s"])),
                "median_peak_speed_deg_s": float(np.median(metrics["peak_speed_deg_s"])),
                "median_path_length_arcmin": float(np.median(metrics["path_length_arcmin"])),
                "median_rms_radius_arcmin": float(np.median(metrics["rms_radius_arcmin"])),
                "median_rms_radius_fraction_of_instrument_trace": float(
                    np.median(metrics["rms_radius_arcmin"])
                    / max(float(np.median(reference["rms_radius_arcmin"])), EPS)
                ),
                "median_path_fraction_of_instrument_trace": float(
                    np.median(metrics["path_length_arcmin"])
                    / max(float(np.median(reference["path_length_arcmin"])), EPS)
                ),
                "population_psd_fraction_above_30hz": float(
                    psd[frequency >= 30].sum() / max(float(psd.sum()), EPS)
                ),
            }
        )
    return pd.DataFrame(rows), filtered


def render_audit(
    path: Path,
    *,
    traces: np.ndarray,
    frame_rate_hz: float,
    frequency: np.ndarray,
    psd: np.ndarray,
    breakpoint: dict[str, float],
    candidates: pd.DataFrame,
    filtered: dict[float, np.ndarray],
    primary_passband_hz: float,
) -> None:
    raw_metrics = trace_metrics(traces, frame_rate_hz=frame_rate_hz)
    order = np.argsort(raw_metrics["mean_speed_deg_s"])
    examples = order[np.asarray((len(order) // 10, len(order) // 2, 9 * len(order) // 10))]
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), constrained_layout=True)
    axis = axes[0, 0]
    axis.loglog(frequency[1:], psd[1:], color="black", lw=1.8)
    axis.axvline(breakpoint["breakpoint_hz"], color="#D55E00", ls="--", label=f"breakpoint {breakpoint['breakpoint_hz']:.0f} Hz")
    axis.axvspan(30, 100, color="#D55E00", alpha=0.08, label="rising high-frequency floor")
    axis.set(xlabel="frequency (Hz)", ylabel="median position PSD", title="A  Instrument-filtered population PSD")
    axis.legend(frameon=False, fontsize=7)

    axis = axes[0, 1]
    axis.plot(candidates.passband_hz, candidates.median_speed_deg_s, "o-", label="mean speed")
    axis.plot(candidates.passband_hz, candidates.median_peak_speed_deg_s, "o-", label="peak speed")
    axis.axvline(primary_passband_hz, color="black", ls="--")
    axis.set(xlabel="analysis-filter passband (Hz)", ylabel="population median (deg/s)", title="B  Velocity sensitivity")
    axis.legend(frameon=False)

    axis = axes[0, 2]
    axis.plot(candidates.passband_hz, candidates.median_rms_radius_fraction_of_instrument_trace, "o-", label="RMS radius")
    axis.plot(candidates.passband_hz, candidates.median_path_fraction_of_instrument_trace, "o-", label="path length")
    axis.axvline(primary_passband_hz, color="black", ls="--")
    axis.axhline(1, color="0.6", lw=0.8)
    axis.set(xlabel="analysis-filter passband (Hz)", ylabel="fraction of 100/118-Hz trace", title="C  Displacement retained; jitter removed")
    axis.legend(frameon=False)

    primary = filtered[float(primary_passband_hz)]
    for column, row in enumerate(examples):
        axis = axes[1, column]
        raw = (traces[row] - traces[row].mean(axis=0)) * 60.0
        smooth = (primary[row] - primary[row].mean(axis=0)) * 60.0
        axis.plot(raw[:, 0], raw[:, 1], color="0.75", lw=0.65, label="100/118-Hz anti-alias")
        axis.plot(smooth[:, 0], smooth[:, 1], color="#0072B2", lw=1.5, label=f"{primary_passband_hz:g}/{primary_passband_hz + 10:g}-Hz analysis filter")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set(xlabel="horizontal (arcmin)", ylabel="vertical (arcmin)", title=f"{chr(68 + column)}  speed quantile {column + 1}/3")
        if column == 0:
            axis.legend(frameon=False, fontsize=7)
    figure.suptitle("Eye-trace bandwidth audit before retinal replay", fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.fixation_bank / "manifest.json").read_text(encoding="utf-8"))
    frame_rate_hz = float(manifest["target_rate_hz"])
    traces = np.load(args.fixation_bank / "trace_xy_filtered.npy")[:, int(args.analysis_start) :]
    frequency, psd = population_position_psd(traces, frame_rate_hz=frame_rate_hz)
    breakpoint = piecewise_psd_breakpoint(frequency, psd)
    candidates, filtered = candidate_table(
        traces,
        frame_rate_hz=frame_rate_hz,
        passbands_hz=np.asarray(args.candidate_passbands_hz, dtype=float),
    )
    primary_rows = candidates.loc[np.isclose(candidates.passband_hz, args.primary_passband_hz)]
    if len(primary_rows) != 1:
        raise ValueError("primary passband must appear exactly once among candidate passbands")
    primary = primary_rows.iloc[0]
    gates = {
        "psd_breakpoint_between_18_and_30_hz": bool(18 <= breakpoint["breakpoint_hz"] <= 30),
        "high_frequency_psd_slope_is_positive": bool(breakpoint["high_frequency_loglog_slope"] > 0),
        "primary_retains_at_least_85_percent_rms_radius": bool(primary.median_rms_radius_fraction_of_instrument_trace >= 0.85),
        "primary_removes_at_least_80_percent_path_length": bool(primary.median_path_fraction_of_instrument_trace <= 0.20),
        "primary_median_speed_below_1_deg_s": bool(primary.median_speed_deg_s < 1.0),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.out_dir / "filter_candidates.csv", index=False)
    np.savez_compressed(args.out_dir / "filter_audit.npz", frequency_hz=frequency, population_psd=psd, primary_filtered_trace_xy=filtered[float(args.primary_passband_hz)])
    figure = args.out_dir / "eye_trace_filter_audit.png"
    render_audit(
        figure,
        traces=traces,
        frame_rate_hz=frame_rate_hz,
        frequency=frequency,
        psd=psd,
        breakpoint=breakpoint,
        candidates=candidates,
        filtered=filtered,
        primary_passband_hz=float(args.primary_passband_hz),
    )
    summary = {
        "analysis": "DDPI eye-trace bandwidth and analysis-filter audit",
        "input_filter": manifest["filter"],
        "input_interpretation": "instrument anti-alias trace, not the final analysis smoothing",
        "n_traces": int(len(traces)),
        "frame_rate_hz": frame_rate_hz,
        "psd_piecewise_fit": breakpoint,
        "selected_analysis_filter": {
            "kind": "zero-phase IIR applied continuously before final 240-Hz sampling in the rebuilt bank",
            "passband_hz": float(args.primary_passband_hz),
            "stopband_hz": float(args.primary_stopband_hz),
            "basis": "population PSD changes from decaying signal-like spectrum to a flat/rising high-frequency floor near 24 Hz",
        },
        "selected_candidate_metrics": {key: float(value) for key, value in primary.to_dict().items()},
        "claim_boundary": "The PSD breakpoint supports removal of a high-frequency measurement floor; it does not identify physiological tremor independently of instrument noise.",
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "figure": str(figure.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["all_gates_pass"]:
        raise RuntimeError(f"eye-trace filter audit failed: {gates}")
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
