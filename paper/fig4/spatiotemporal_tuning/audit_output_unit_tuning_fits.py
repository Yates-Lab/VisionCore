#!/usr/bin/env python3
"""Audit continuous output-unit SF/TF peaks against raw grating responses.

The existing M77 summary fits a single correlated log-Gaussian to phase-RMS
modulation at the strongest orientation.  This audit instead estimates the
peak with a local joint quadratic around the sampled maximum and asks whether
that vertex is stable to omitted local points, smoothing scale, response metric
(all-harmonic RMS versus phase-locked F1), and orientation selection.  The
global log-Gaussian is retained only as a descriptive shape check; its R² is
not treated as peak validation.
It produces population diagnostics plus galleries that never hide the sampled
responses behind the fitted surface.
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
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    EPS,
    fit_local_quadratic_peak,
    fit_log_gaussian_surface,
)


DEFAULT_GROUPED = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/periodic_tuning_respaced/"
    "frequency_tuning_grouped.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/periodic_tuning_respaced/fit_audit"
)
METRICS = {
    "rms": ("response_amp_rms", "phase RMS (all harmonics)"),
    "f1": ("f1_amplitude", "phase-locked F1"),
}
SMOOTHING_SIGMAS = (0.5, 1.0, 1.5)
JACKKNIFE_TF_ROWS = (0, 2, 4, 6, 8, 10, 12, 13)
CATEGORY_COLORS = {
    "trusted": "#2E7D49",
    "usable with caution": "#D88B21",
    "metric-dependent": "#8A5AA5",
    "unstable/censored": "#C64B4B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def complete_surface(
    frame: pd.DataFrame, value_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.sort(frame.spatial_cpd.unique().astype(float))
    tf = np.sort(frame.temporal_hz.unique().astype(float))
    values = (
        frame.pivot_table(
            index="temporal_hz",
            columns="spatial_cpd",
            values=value_column,
            aggfunc="mean",
        )
        .reindex(index=tf, columns=sf)
        .to_numpy(dtype=float)
    )
    return sf, tf, values


def orientation_score(frame: pd.DataFrame, value_column: str) -> pd.Series:
    return frame.groupby("probe_orientation_deg")[value_column].apply(
        lambda values: float(np.sqrt(np.nanmean(np.square(values.to_numpy(dtype=float)))))
    )


def heldout_r2(sf: np.ndarray, tf: np.ndarray, response: np.ndarray) -> float:
    row, column = np.indices(response.shape)
    scores: list[float] = []
    for phase in (0, 1):
        holdout = (row + column) % 2 == phase
        train = response.copy()
        train[holdout] = np.nan
        fit = fit_log_gaussian_surface(sf, tf, train)
        prediction = np.asarray(fit["fitted"], dtype=float)
        observed = response[holdout]
        predicted = prediction[holdout]
        denominator = float(np.sum(np.square(observed - np.mean(observed))))
        scores.append(
            1.0 - float(np.sum(np.square(observed - predicted))) / max(denominator, EPS)
        )
    return float(np.mean(scores))


def scale_space_peaks(
    sf: np.ndarray, tf: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    peak_sf, peak_tf = [], []
    for sigma in SMOOTHING_SIGMAS:
        smoothed = gaussian_filter(response, sigma=sigma, mode="nearest")
        tf_index, sf_index = np.unravel_index(int(np.nanargmax(smoothed)), smoothed.shape)
        peak_sf.append(float(sf[sf_index]))
        peak_tf.append(float(tf[tf_index]))
    return np.asarray(peak_sf), np.asarray(peak_tf)


def local_peak_jackknife(
    sf: np.ndarray, tf: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Delete each point around the observed maximum and refit its vertex."""
    sf_peaks: list[float] = []
    tf_peaks: list[float] = []
    for local_index in range(9):
        peak = fit_local_quadratic_peak(
            sf, tf, response, omitted_local_index=local_index
        )
        if peak.get("peak_status") != "ok":
            continue
        sf_peak = float(peak["preferred_sf_cpd"])
        tf_peak = float(peak["preferred_tf_hz"])
        if np.isfinite(sf_peak) and np.isfinite(tf_peak):
            sf_peaks.append(sf_peak)
            tf_peaks.append(tf_peak)
    return np.asarray(sf_peaks, dtype=float), np.asarray(tf_peaks, dtype=float)


def finite_quantile(values: np.ndarray, quantile: float) -> float:
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected)]
    return float(np.quantile(selected, quantile)) if len(selected) else np.nan


def octave_span(values: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return np.inf
    quantiles = np.quantile(np.log2(values), [low, high])
    return float(quantiles[1] - quantiles[0])


def orientation_peak_span(
    unit: pd.DataFrame,
    value_column: str,
    scores: pd.Series,
) -> tuple[float, float, list[float], list[float]]:
    maximum = float(scores.max())
    sf_peaks: list[float] = []
    tf_peaks: list[float] = []
    for orientation, score in scores.items():
        if float(score) < 0.75 * maximum:
            continue
        selected = unit.loc[np.isclose(unit.probe_orientation_deg, float(orientation))]
        sf, tf, response = complete_surface(selected, value_column)
        peak = fit_local_quadratic_peak(sf, tf, response)
        if (
            peak.get("peak_status") == "ok"
            and float(peak["local_fit_r2"]) >= 0.35
        ):
            sf_peaks.append(float(peak["preferred_sf_cpd"]))
            tf_peaks.append(float(peak["preferred_tf_hz"]))
    return (
        octave_span(np.asarray(sf_peaks)),
        octave_span(np.asarray(tf_peaks)),
        sf_peaks,
        tf_peaks,
    )


def analyze_metric(unit: pd.DataFrame, prefix: str) -> tuple[dict, dict]:
    value_column, label = METRICS[prefix]
    scores = orientation_score(unit, value_column)
    best_orientation = float(scores.idxmax())
    selected = unit.loc[np.isclose(unit.probe_orientation_deg, best_orientation)]
    sf, tf, response = complete_surface(selected, value_column)
    global_fit = fit_log_gaussian_surface(sf, tf, response)
    peak = fit_local_quadratic_peak(sf, tf, response)
    smooth_sf, smooth_tf = scale_space_peaks(sf, tf, response)
    jackknife_sf, jackknife_tf = local_peak_jackknife(sf, tf, response)
    (
        orientation_sf_span,
        orientation_tf_span,
        orientation_sf_peaks,
        orientation_tf_peaks,
    ) = orientation_peak_span(
        unit, value_column, scores
    )
    preferred_sf = float(peak.get("preferred_sf_cpd", np.nan))
    preferred_tf = float(peak.get("preferred_tf_hz", np.nan))
    tf_index = int(peak["discrete_peak_tf_index"])
    sf_index = int(peak["discrete_peak_sf_index"])
    record = {
        f"{prefix}_best_orientation_deg": best_orientation,
        f"{prefix}_orientation_dominance": float(scores.max() / max(float(scores.nlargest(2).iloc[-1]), EPS)),
        f"{prefix}_peak_status": str(peak["peak_status"]),
        f"{prefix}_local_peak_r2": float(peak.get("local_fit_r2", np.nan)),
        f"{prefix}_fit_r2": float(global_fit["fit_r2"]),
        f"{prefix}_heldout_r2": heldout_r2(sf, tf, response),
        f"{prefix}_preferred_sf_cpd": preferred_sf,
        f"{prefix}_preferred_tf_hz": preferred_tf,
        f"{prefix}_global_center_sf_cpd": float(global_fit["preferred_sf_cpd"]),
        f"{prefix}_global_center_tf_hz": float(global_fit["preferred_tf_hz"]),
        f"{prefix}_discrete_peak_sf_cpd": float(peak["discrete_peak_sf_cpd"]),
        f"{prefix}_discrete_peak_tf_hz": float(peak["discrete_peak_tf_hz"]),
        f"{prefix}_low_tf_censored": bool(tf_index == 0),
        f"{prefix}_high_tf_censored": bool(tf_index == len(tf) - 1),
        f"{prefix}_low_sf_censored": bool(sf_index == 0),
        f"{prefix}_high_sf_censored": bool(sf_index == len(sf) - 1),
        f"{prefix}_smooth_sf_median_cpd": float(np.median(smooth_sf)),
        f"{prefix}_smooth_sf_span_octaves": octave_span(smooth_sf, 0.0, 1.0),
        f"{prefix}_fit_vs_smooth_sf_octaves": float(
            abs(np.log2(preferred_sf / np.median(smooth_sf)))
        ),
        f"{prefix}_smooth_tf_median_hz": float(np.median(smooth_tf)),
        f"{prefix}_smooth_tf_span_octaves": octave_span(smooth_tf, 0.0, 1.0),
        f"{prefix}_fit_vs_smooth_octaves": float(
            abs(np.log2(preferred_tf / np.median(smooth_tf)))
        ),
        f"{prefix}_jackknife_sf_median_cpd": finite_quantile(jackknife_sf, 0.5),
        f"{prefix}_jackknife_sf_p10_cpd": finite_quantile(jackknife_sf, 0.1),
        f"{prefix}_jackknife_sf_p90_cpd": finite_quantile(jackknife_sf, 0.9),
        f"{prefix}_jackknife_sf_span_octaves": octave_span(jackknife_sf),
        f"{prefix}_jackknife_tf_median_hz": finite_quantile(jackknife_tf, 0.5),
        f"{prefix}_jackknife_tf_p10_hz": finite_quantile(jackknife_tf, 0.1),
        f"{prefix}_jackknife_tf_p90_hz": finite_quantile(jackknife_tf, 0.9),
        f"{prefix}_jackknife_tf_span_octaves": octave_span(jackknife_tf),
        f"{prefix}_n_local_jackknife_peaks": int(len(jackknife_tf)),
        f"{prefix}_orientation_sf_span_octaves": orientation_sf_span,
        f"{prefix}_orientation_tf_span_octaves": orientation_tf_span,
        f"{prefix}_n_strong_orientation_peaks": int(len(orientation_tf_peaks)),
    }
    details = {
        "label": label,
        "value_column": value_column,
        "scores": scores,
        "best_orientation": best_orientation,
        "sf": sf,
        "tf": tf,
        "response": response,
        "fit": global_fit,
        "peak": peak,
        "smooth_sf": smooth_sf,
        "smooth_tf": smooth_tf,
        "jackknife_sf": jackknife_sf,
        "jackknife_tf": jackknife_tf,
        "orientation_sf": np.asarray(orientation_sf_peaks, dtype=float),
        "orientation_tf": np.asarray(orientation_tf_peaks, dtype=float),
    }
    return record, details


def classify(frame: pd.DataFrame) -> pd.Series:
    boundary_censored = (
        frame.rms_low_tf_censored
        | frame.rms_high_tf_censored
        | frame.f1_low_tf_censored
        | frame.f1_high_tf_censored
        | frame.rms_low_sf_censored
        | frame.rms_high_sf_censored
        | frame.f1_low_sf_censored
        | frame.f1_high_sf_censored
    )
    peak_invalid = (
        frame.rms_peak_status.ne("ok")
        | frame.f1_peak_status.ne("ok")
        | frame.rms_preferred_tf_hz.isna()
        | frame.f1_preferred_tf_hz.isna()
        | frame.rms_preferred_sf_cpd.isna()
        | frame.f1_preferred_sf_cpd.isna()
    )
    grossly_unstable = (
        frame[["rms_local_peak_r2", "f1_local_peak_r2"]].min(axis=1).lt(0.35)
        | frame[["rms_n_local_jackknife_peaks", "f1_n_local_jackknife_peaks"]]
        .min(axis=1)
        .lt(6)
        | frame[["rms_jackknife_tf_span_octaves", "f1_jackknife_tf_span_octaves"]]
        .max(axis=1)
        .gt(1.5)
        | frame[["rms_jackknife_sf_span_octaves", "f1_jackknife_sf_span_octaves"]]
        .max(axis=1)
        .gt(1.5)
        | frame[["rms_orientation_tf_span_octaves", "f1_orientation_tf_span_octaves"]]
        .max(axis=1)
        .gt(1.5)
        | frame[["rms_orientation_sf_span_octaves", "f1_orientation_sf_span_octaves"]]
        .max(axis=1)
        .gt(1.5)
    )
    metric_dependent = frame[
        ["rms_vs_f1_tf_difference_octaves", "rms_vs_f1_sf_difference_octaves"]
    ].max(axis=1).gt(0.75)
    trusted = (
        ~boundary_censored
        & ~peak_invalid
        & ~grossly_unstable
        & ~metric_dependent
        & frame[["rms_local_peak_r2", "f1_local_peak_r2"]].min(axis=1).ge(0.50)
        & frame[["rms_jackknife_tf_span_octaves", "f1_jackknife_tf_span_octaves"]]
        .max(axis=1)
        .le(0.75)
        & frame[["rms_jackknife_sf_span_octaves", "f1_jackknife_sf_span_octaves"]]
        .max(axis=1)
        .le(0.75)
        & frame[["rms_fit_vs_smooth_octaves", "f1_fit_vs_smooth_octaves"]]
        .max(axis=1)
        .le(0.75)
        & frame[["rms_fit_vs_smooth_sf_octaves", "f1_fit_vs_smooth_sf_octaves"]]
        .max(axis=1)
        .le(0.75)
    )
    category = pd.Series("usable with caution", index=frame.index, dtype=object)
    category.loc[
        metric_dependent & ~boundary_censored & ~peak_invalid & ~grossly_unstable
    ] = "metric-dependent"
    category.loc[boundary_censored | peak_invalid | grossly_unstable] = "unstable/censored"
    category.loc[trusted] = "trusted"
    return category


def analyze(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict]]:
    dynamic = table.loc[table.temporal_hz.gt(0)].copy()
    rows: list[dict] = []
    details: dict[int, dict] = {}
    for unit_index, unit in dynamic.groupby("unit_index", sort=True):
        record = {"unit_index": int(unit_index)}
        unit_details: dict[str, dict] = {}
        for prefix in METRICS:
            metric_record, metric_details = analyze_metric(unit, prefix)
            record.update(metric_record)
            unit_details[prefix] = metric_details
        record["rms_vs_f1_tf_difference_octaves"] = float(
            abs(
                np.log2(
                    record["rms_preferred_tf_hz"]
                    / record["f1_preferred_tf_hz"]
                )
            )
        )
        record["rms_vs_f1_sf_difference_octaves"] = float(
            abs(
                np.log2(
                    record["rms_preferred_sf_cpd"]
                    / record["f1_preferred_sf_cpd"]
                )
            )
        )
        orientation_difference = abs(
            record["rms_best_orientation_deg"] - record["f1_best_orientation_deg"]
        )
        record["rms_vs_f1_orientation_difference_deg"] = float(
            min(orientation_difference, 180.0 - orientation_difference)
        )
        rows.append(record)
        details[int(unit_index)] = unit_details
    summary = pd.DataFrame(rows)
    summary["audit_category"] = classify(summary)
    return summary, details


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def population_figure(summary: pd.DataFrame, out_dir: Path) -> Path:
    configure()
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 7.8), constrained_layout=True)
    exemplar = summary.loc[summary.unit_index.eq(25)].iloc[0]

    def metric_comparison(
        axis: plt.Axes,
        *,
        rms_column: str,
        f1_column: str,
        limits: tuple[float, float],
        xlabel: str,
        ylabel: str,
        title: str,
        show_legend: bool = False,
    ) -> None:
        for category, color in CATEGORY_COLORS.items():
            selected = summary.loc[summary.audit_category.eq(category)]
            axis.scatter(
                selected[rms_column],
                selected[f1_column],
                s=29,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.4,
                label=f"{category} (n={len(selected)})",
            )
        axis.plot(limits, limits, color="0.65", linestyle=":")
        axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.scatter(
            [exemplar[rms_column]],
            [exemplar[f1_column]],
            marker="*",
            s=85,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        axis.text(
            exemplar[rms_column] * 1.05,
            exemplar[f1_column],
            "u025",
            fontsize=7,
            va="center",
        )
        if show_legend:
            axis.legend(frameon=False, fontsize=6.4, loc="lower right")

    metric_comparison(
        axes[0, 0],
        rms_column="rms_preferred_tf_hz",
        f1_column="f1_preferred_tf_hz",
        limits=(0.8, 100),
        xlabel="continuous phase-RMS peak TF (Hz)",
        ylabel="continuous F1 peak TF (Hz)",
        title="A  Does preferred TF depend on the response metric?",
        show_legend=True,
    )
    sf_min = 0.85 * float(
        summary[["rms_preferred_sf_cpd", "f1_preferred_sf_cpd"]].min().min()
    )
    sf_max = 1.15 * float(
        summary[["rms_preferred_sf_cpd", "f1_preferred_sf_cpd"]].max().max()
    )
    metric_comparison(
        axes[0, 1],
        rms_column="rms_preferred_sf_cpd",
        f1_column="f1_preferred_sf_cpd",
        limits=(sf_min, sf_max),
        xlabel="continuous phase-RMS peak SF (cycles/deg)",
        ylabel="continuous F1 peak SF (cycles/deg)",
        title="B  Does preferred SF depend on the response metric?",
    )

    axes[0, 2].scatter(
        summary.rms_fit_r2,
        summary.rms_heldout_r2,
        s=25,
        color="#4C78A8",
        alpha=0.55,
        label="phase RMS",
    )
    axes[0, 2].scatter(
        summary.f1_fit_r2,
        summary.f1_heldout_r2,
        s=25,
        color="#E07B39",
        alpha=0.55,
        label="F1",
    )
    finite_cv = np.r_[summary.rms_heldout_r2, summary.f1_heldout_r2]
    lower = max(-1.0, float(np.nanquantile(finite_cv, 0.02)))
    axes[0, 2].axhline(0, color="0.55", linewidth=0.8)
    axes[0, 2].plot([-0.1, 1], [-0.1, 1], color="0.75", linestyle=":")
    axes[0, 2].set_xlim(-0.1, 1.0)
    axes[0, 2].set_ylim(lower, 1.0)
    axes[0, 2].set_xlabel("global-shape in-sample $R^2$")
    axes[0, 2].set_ylabel("global-shape checkerboard held-out $R^2$")
    axes[0, 2].set_title("C  Shape fit is descriptive, not peak validation", loc="left")
    axes[0, 2].legend(frameon=False)
    axes[0, 2].scatter(
        [exemplar.rms_fit_r2, exemplar.f1_fit_r2],
        [exemplar.rms_heldout_r2, exemplar.f1_heldout_r2],
        marker="*",
        s=95,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    axes[0, 2].text(
        exemplar.f1_fit_r2 + 0.015,
        exemplar.f1_heldout_r2,
        "u025",
        fontsize=7,
        va="center",
    )

    maximum_tf_jackknife = summary[
        ["rms_jackknife_tf_span_octaves", "f1_jackknife_tf_span_octaves"]
    ].max(axis=1)
    for category, color in CATEGORY_COLORS.items():
        selected = summary.audit_category.eq(category)
        axes[1, 0].scatter(
            summary.loc[selected, "rms_vs_f1_tf_difference_octaves"],
            maximum_tf_jackknife[selected],
            s=29,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
    axes[1, 0].axvline(0.75, color="0.55", linestyle=":")
    axes[1, 0].axhline(1.5, color="0.55", linestyle=":")
    axes[1, 0].set_xlabel("RMS–F1 peak disagreement (octaves)")
    axes[1, 0].set_ylabel("largest leave-one-band TF span (octaves)")
    axes[1, 0].set_title("D  Is the preferred TF estimator-stable?", loc="left")
    axes[1, 0].set_xlim(left=-0.05)
    axes[1, 0].set_ylim(bottom=-0.05)

    maximum_sf_jackknife = summary[
        ["rms_jackknife_sf_span_octaves", "f1_jackknife_sf_span_octaves"]
    ].max(axis=1)
    for category, color in CATEGORY_COLORS.items():
        selected = summary.audit_category.eq(category)
        axes[1, 1].scatter(
            summary.loc[selected, "rms_vs_f1_sf_difference_octaves"],
            maximum_sf_jackknife[selected],
            s=29,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
    axes[1, 1].axvline(0.75, color="0.55", linestyle=":")
    axes[1, 1].axhline(1.5, color="0.55", linestyle=":")
    axes[1, 1].set_xlabel("RMS–F1 peak disagreement (octaves)")
    axes[1, 1].set_ylabel("largest leave-one-band SF span (octaves)")
    axes[1, 1].set_title("E  Is the preferred SF estimator-stable?", loc="left")
    axes[1, 1].set_xlim(left=-0.05)
    axes[1, 1].set_ylim(bottom=-0.05)

    counts = summary.audit_category.value_counts().reindex(CATEGORY_COLORS, fill_value=0)
    bars = axes[1, 2].bar(
        np.arange(len(counts)),
        counts.to_numpy(),
        color=[CATEGORY_COLORS[name] for name in counts.index],
    )
    axes[1, 2].set_xticks(
        np.arange(len(counts)),
        ["trusted", "usable\nwith caution", "metric-\ndependent", "unstable/\ncensored"],
    )
    axes[1, 2].set_ylabel("output units")
    axes[1, 2].set_title("F  Joint SF/TF audit outcome", loc="left")
    for bar, count in zip(bars, counts):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.7,
            str(int(count)),
            ha="center",
            va="bottom",
        )
    axes[1, 2].set_ylim(0, max(counts) * 1.24)
    axes[1, 2].text(
        0.98,
        0.96,
        "Trusted requires a concave local maximum\nand agreement across metric, smoothing,\nlocal delete-one fits, and orientations.",
        transform=axes[1, 2].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )

    figure.suptitle(
        "M77 joint SF/TF tuning audit: local interpolated maxima must follow the raw peak",
        fontsize=13,
        fontweight="semibold",
    )
    path = out_dir / "m77_tuning_fit_audit_population.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def normalized_fit_surface(details: dict) -> np.ndarray:
    fit = details["fit"]
    return np.maximum(
        np.asarray(fit["fitted"], dtype=float) - float(fit["baseline"]), 0.0
    ) / max(float(fit["amplitude"]), EPS)


def normalized_observed(details: dict) -> np.ndarray:
    values = np.asarray(details["response"], dtype=float)
    # Normalize the samples independently of the parametric fit.  Otherwise a
    # failed near-zero-amplitude fit saturates the entire raw heatmap and hides
    # precisely the structure this audit is intended to expose.
    shifted = values - float(np.nanmin(values))
    return shifted / max(float(np.nanmax(shifted)), EPS)


def example_gallery(
    summary: pd.DataFrame,
    details: dict[int, dict],
    units: list[int],
    out_dir: Path,
    page: int,
) -> Path:
    configure()
    figure, axes = plt.subplots(
        len(units),
        5,
        figsize=(17.0, 2.45 * len(units)),
        squeeze=False,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.18, 0.9, 0.9)},
    )
    for row_index, unit_index in enumerate(units):
        record = summary.loc[summary.unit_index.eq(unit_index)].iloc[0]
        for column, prefix in enumerate(("rms", "f1")):
            item = details[unit_index][prefix]
            observed = normalized_observed(item)
            fitted = normalized_fit_surface(item)
            axis = axes[row_index, column]
            axis.imshow(
                np.clip(observed, 0, 1),
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            axis.contour(
                fitted,
                levels=[0.25, 0.5, 0.75],
                colors="#55DDE0",
                linewidths=0.8,
            )
            sf = item["sf"]
            tf = item["tf"]
            fit = item["fit"]
            peak = item["peak"]
            if peak.get("peak_status") == "ok":
                sf_index = np.interp(
                    np.log2(float(peak["preferred_sf_cpd"])),
                    np.log2(sf),
                    np.arange(len(sf)),
                )
                tf_index = np.interp(
                    np.log2(float(peak["preferred_tf_hz"])),
                    np.log2(tf),
                    np.arange(len(tf)),
                )
                axis.scatter([sf_index], [tf_index], marker="*", s=52, color="#F4A261", edgecolor="white", linewidth=0.5)
            axis.set_xticks(np.arange(0, len(sf), 2), [f"{value:.2g}" for value in sf[::2]])
            tf_ticks = np.asarray([0, 2, 4, 6, 8, 10, 12, 13])
            axis.set_yticks(tf_ticks, [f"{tf[index]:.2g}" for index in tf_ticks])
            axis.set_xlabel("SF (cpd)")
            if column == 0:
                axis.set_ylabel(f"u{unit_index:03d}\nTF (Hz)")
            metric_name = "phase RMS" if prefix == "rms" else "F1"
            peak_label = (
                f"local {float(peak['preferred_sf_cpd']):.2g} cpd, "
                f"{float(peak['preferred_tf_hz']):.1f} Hz"
                if peak.get("peak_status") == "ok"
                else f"local peak rejected ({peak.get('peak_status')})"
            )
            axis.set_title(
                f"{metric_name} raw + global shape contour\n"
                f"ori {item['best_orientation']:g}°, {peak_label}"
            )

        rms = details[unit_index]["rms"]
        f1 = details[unit_index]["f1"]
        axis = axes[row_index, 2]
        for item, color, label in (
            (rms, "#4C78A8", "phase RMS"),
            (f1, "#E07B39", "F1"),
        ):
            observed_marginal = np.sqrt(np.mean(np.square(item["response"]), axis=1))
            fitted_marginal = np.sqrt(np.mean(np.square(item["fit"]["fitted"]), axis=1))
            scale = max(float(observed_marginal.max()), EPS)
            axis.plot(item["tf"], observed_marginal / scale, "o-", color=color, ms=3.2, lw=1.2, label=f"{label} raw")
            axis.plot(item["tf"], fitted_marginal / scale, "-", color=color, lw=1.8, alpha=0.7)
            if item["peak"].get("peak_status") == "ok":
                axis.axvline(float(item["peak"]["preferred_tf_hz"]), color=color, linestyle=":", lw=1.0)
        axis.set_xscale("log", base=2)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("stimulus TF (Hz)")
        axis.set_ylabel("normalized SF-marginal response")
        axis.set_title(
            f"raw profiles; global shape fits only\n"
            f"local $R^2$: RMS {record.rms_local_peak_r2:.2f}, F1 {record.f1_local_peak_r2:.2f}"
        )
        if row_index == 0:
            axis.legend(frameon=False, fontsize=6.6, ncol=2)
        axis.grid(alpha=0.18)

        category = str(record.audit_category)

        def plot_estimators(
            axis: plt.Axes,
            *,
            coordinate: str,
            xlim: tuple[float, float],
            xlabel: str,
            disagreement: float,
            show_category: bool,
        ) -> None:
            suffix = "tf" if coordinate == "tf" else "sf"
            estimators = [
                ("RMS local peak", np.asarray([record[f"rms_preferred_{suffix}_{'hz' if suffix == 'tf' else 'cpd'}"]]), "#4C78A8"),
                ("RMS smooth", rms[f"smooth_{coordinate}"], "#4C78A8"),
                ("RMS jackknife", rms[f"jackknife_{coordinate}"], "#4C78A8"),
                ("F1 local peak", np.asarray([record[f"f1_preferred_{suffix}_{'hz' if suffix == 'tf' else 'cpd'}"]]), "#E07B39"),
                ("F1 smooth", f1[f"smooth_{coordinate}"], "#E07B39"),
                ("F1 jackknife", f1[f"jackknife_{coordinate}"], "#E07B39"),
            ]
            for y, (label, values, color) in enumerate(estimators):
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values) & (values > 0)]
                if not len(values):
                    continue
                axis.plot([values.min(), values.max()], [y, y], color=color, lw=1.5, alpha=0.7)
                axis.scatter(values, np.full(len(values), y), s=15, color=color, alpha=0.62, edgecolor="none")
                axis.scatter([np.median(values)], [y], s=30, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            axis.set_xscale("log", base=2)
            axis.set_xlim(*xlim)
            axis.set_yticks(np.arange(len(estimators)), [item[0] for item in estimators])
            axis.invert_yaxis()
            axis.set_xlabel(xlabel)
            title = (
                f"{category}\n" if show_category else ""
            ) + f"RMS–F1 difference {disagreement:.2f} oct"
            axis.set_title(title, color=CATEGORY_COLORS[category] if show_category else "black")
            axis.grid(axis="x", alpha=0.18)

        plot_estimators(
            axes[row_index, 3],
            coordinate="tf",
            xlim=(0.8, 100),
            xlabel="preferred TF estimate (Hz)",
            disagreement=float(record.rms_vs_f1_tf_difference_octaves),
            show_category=True,
        )
        plot_estimators(
            axes[row_index, 4],
            coordinate="sf",
            xlim=(0.9, 18),
            xlabel="preferred SF estimate (cpd)",
            disagreement=float(record.rms_vs_f1_sf_difference_octaves),
            show_category=False,
        )

    figure.suptitle(
        "M77 tuning-fit audit: raw samples, fitted shape, and estimator sensitivity",
        fontsize=13,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.5, w_pad=1.0)
    path = out_dir / f"m77_tuning_fit_audit_examples_page{page}.png"
    figure.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.grouped_csv)
    summary, details = analyze(table)
    summary.to_csv(args.out_dir / "m77_tuning_fit_audit.csv", index=False)
    population_path = population_figure(summary, args.out_dir)
    # Include the two units that exposed the global-center bias, plus two more
    # high-TF examples where the same failure mode is especially visible.
    trusted_units = [unit for unit in (88, 25, 0, 67) if unit in details]
    questionable_units = [39, 55, 51, 23]
    galleries = [
        example_gallery(summary, details, trusted_units, args.out_dir, 1),
        example_gallery(summary, details, questionable_units, args.out_dir, 2),
    ]
    counts = summary.audit_category.value_counts().reindex(CATEGORY_COLORS, fill_value=0)
    exemplar = summary.loc[summary.unit_index.eq(25)].iloc[0]
    report = {
        "analysis": "M77 continuous SF/TF peak audit",
        "n_units": int(len(summary)),
        "response_metrics": {
            "phase_rms": "all phase-locked and higher-harmonic modulation",
            "f1": "least-squares response component at the stimulus temporal frequency",
        },
        "audit_counts": {key: int(value) for key, value in counts.items()},
        "population": {
            "median_rms_vs_f1_tf_difference_octaves": float(
                summary.rms_vs_f1_tf_difference_octaves.median()
            ),
            "median_rms_vs_f1_sf_difference_octaves": float(
                summary.rms_vs_f1_sf_difference_octaves.median()
            ),
            "fraction_tf_within_half_octave": float(
                summary.rms_vs_f1_tf_difference_octaves.le(0.5).mean()
            ),
            "fraction_sf_within_half_octave": float(
                summary.rms_vs_f1_sf_difference_octaves.le(0.5).mean()
            ),
            "fraction_positive_heldout_r2_rms": float(summary.rms_heldout_r2.gt(0).mean()),
            "fraction_positive_heldout_r2_f1": float(summary.f1_heldout_r2.gt(0).mean()),
            "n_with_any_sf_boundary_censoring": int(
                summary[
                    [
                        "rms_low_sf_censored",
                        "rms_high_sf_censored",
                        "f1_low_sf_censored",
                        "f1_high_sf_censored",
                    ]
                ].any(axis=1).sum()
            ),
            "n_with_any_tf_boundary_censoring": int(
                summary[
                    [
                        "rms_low_tf_censored",
                        "rms_high_tf_censored",
                        "f1_low_tf_censored",
                        "f1_high_tf_censored",
                    ]
                ].any(axis=1).sum()
            ),
        },
        "exemplar_u025": {
            key: (bool(value) if isinstance(value, np.bool_) else value.item() if isinstance(value, np.generic) else value)
            for key, value in exemplar.to_dict().items()
        },
        "classification": {
            "trusted": "interior concave local joint maximum; RMS and F1 local R2 >= .5, both coordinate differences <= .75 octave, both local delete-one spans <= .75 octave, and both local-peak-to-smoothing differences <= .75 octave",
            "metric-dependent": "individually stable fits but RMS and F1 SF or TF peaks differ by > .75 octave",
            "unstable/censored": "boundary/nonconcave local maximum, local R2 < .35, fewer than six valid local delete-one fits, or >1.5-octave SF/TF local-jackknife/orientation instability",
            "usable_with_caution": "passes gross failure gates but not every conservative trusted gate",
        },
        "claim_boundary": (
            "A continuous joint SF/TF peak is reported only when a concave quadratic fitted to the 3x3 neighborhood of the sampled maximum has its vertex inside that neighborhood. "
            "The global log-Gaussian and its held-out R2 describe the broader surface but do not validate peak location. "
            "Phase RMS and F1 answer different questions once nonlinearities create harmonics; "
            "neither is a direct causal measure of retinal-motion contribution to SSI."
        ),
        "source": str(args.grouped_csv.resolve()),
        "figures": [str(population_path.resolve()), *[str(path.resolve()) for path in galleries]],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["audit_counts"], indent=2))
    print(json.dumps(report["population"], indent=2))
    print(json.dumps({"u025": report["exemplar_u025"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
