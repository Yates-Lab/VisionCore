#!/usr/bin/env python3
"""Quantify the bridge between real BackImage behavior and model dose curves."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
import _fig4_paths as _paths

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from _fig4_style import COHERENCE_ORDER, GRAY, configure_matplotlib
from fixation_stats import plot_backimage_contour_motion_components as contour_motion


OUT_DIR = _paths.FIG_DIR / "behavior_model_bridge"
OUT_STEM = "behavior_model_bridge"
MODEL_VALUES_CSV = _paths.PATH_BINS_VALUES_CSV
MODEL_REFERENCE_CSV = _paths.PATH_BINS_TRACE_BANK_REFERENCE_CSV
MODEL_POPULATIONS_CSV = _paths.PATH_BINS_POPULATIONS_CSV
BEHAVIOR_WINDOWS_CSV = _paths.BEHAVIOR_PATH_WINDOWS_CSV

PANEL_G_SNIPPET_N_SAMPLES = 40
DT = 1.0 / 120.0
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 119
EPS = 1e-12
DISPLAY_BEHAVIOR_QUANTILE = 99.0

COHERENCE_COLORS = {"0-0.2": "#9aa5b1", "0.8-1": "#0b4f83"}
GREEN = "#1b7f5c"
PURPLE = "#7a3b9a"
GRID = "#d8dde3"

METRIC_FAMILIES = (
    {
        "key": "component_path",
        "title": "Unsigned Component Path",
        "xlabel": "component path, central 0.325 s snippet (arcmin)",
        "across_behavior": "across_snippet_path_arcmin",
        "along_behavior": "along_snippet_path_arcmin",
    },
    {
        "key": "component_rms",
        "title": "RMS Excursion",
        "xlabel": "component RMS, central 0.325 s snippet (arcmin)",
        "across_behavior": "across_snippet_rms_arcmin",
        "along_behavior": "along_snippet_rms_arcmin",
    },
    {
        "key": "component_range",
        "title": "Projected Range",
        "xlabel": "component peak-to-peak range, central 0.325 s snippet (arcmin)",
        "across_behavior": "across_snippet_range_arcmin",
        "along_behavior": "along_snippet_range_arcmin",
    },
    {
        "key": "path_per_range",
        "title": "Tortuosity Proxy",
        "xlabel": "component path / range, central 0.325 s snippet",
        "across_behavior": "across_snippet_path_per_range",
        "along_behavior": "along_snippet_path_per_range",
    },
)

COMPONENTS = (
    ("across", "contour-normal", PURPLE, "-", "o"),
    ("along", "contour-parallel", GREEN, (0, (4.2, 2.0)), "s"),
)

REQUIRED_BEHAVIOR_METRIC_COLUMNS = [
    "along_snippet_path_arcmin",
    "across_snippet_path_arcmin",
    "along_snippet_rms_arcmin",
    "across_snippet_rms_arcmin",
    "along_snippet_range_arcmin",
    "across_snippet_range_arcmin",
    "along_snippet_path_per_range",
    "across_snippet_path_per_range",
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _central_snippet(trace: np.ndarray, n_samples: int = PANEL_G_SNIPPET_N_SAMPLES) -> np.ndarray:
    arr = np.asarray(trace, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < int(n_samples):
        return np.empty((0, 2), dtype=np.float64)
    start = max(0, (arr.shape[0] - int(n_samples)) // 2)
    return arr[start : start + int(n_samples)].copy()


def _snippet_metrics_for_row(row: pd.Series) -> dict[str, float | int]:
    trace = contour_motion._window_trace(row)
    snippet = _central_snippet(trace)
    if snippet.shape[0] < PANEL_G_SNIPPET_N_SAMPLES or not np.isfinite(snippet).all():
        return {
            "snippet_n_samples_used": int(snippet.shape[0]),
            "snippet_duration_s": float("nan"),
            **{col: float("nan") for col in REQUIRED_BEHAVIOR_METRIC_COLUMNS},
        }
    edge_axis = float(row["image_edge_axis_deg"])
    along_vec, across_vec = contour_motion._axis_vectors(np.asarray([edge_axis], dtype=np.float64))
    along = along_vec[0]
    across = across_vec[0]
    steps = np.diff(snippet, axis=0)
    centered = snippet - np.nanmean(snippet, axis=0, keepdims=True)

    along_step = steps @ along
    across_step = steps @ across
    along_pos = centered @ along
    across_pos = centered @ across

    along_path = float(np.nansum(np.abs(along_step)) * 60.0)
    across_path = float(np.nansum(np.abs(across_step)) * 60.0)
    along_rms = float(np.sqrt(np.nanmean(along_pos * along_pos)) * 60.0)
    across_rms = float(np.sqrt(np.nanmean(across_pos * across_pos)) * 60.0)
    along_range = float((np.nanmax(along_pos) - np.nanmin(along_pos)) * 60.0)
    across_range = float((np.nanmax(across_pos) - np.nanmin(across_pos)) * 60.0)

    return {
        "snippet_n_samples_used": int(snippet.shape[0]),
        "snippet_duration_s": float((snippet.shape[0] - 1) * DT),
        "along_snippet_path_arcmin": along_path,
        "across_snippet_path_arcmin": across_path,
        "along_snippet_rms_arcmin": along_rms,
        "across_snippet_rms_arcmin": across_rms,
        "along_snippet_range_arcmin": along_range,
        "across_snippet_range_arcmin": across_range,
        "along_snippet_path_per_range": along_path / along_range if along_range > EPS else float("nan"),
        "across_snippet_path_per_range": across_path / across_range if across_range > EPS else float("nan"),
    }


def compute_behavior_metrics(
    *,
    cache_csv: Path,
    force_recompute: bool = False,
    behavior_windows_csv: Path = BEHAVIOR_WINDOWS_CSV,
) -> pd.DataFrame:
    if cache_csv.exists() and not force_recompute:
        cached = pd.read_csv(cache_csv)
        if all(col in cached.columns for col in REQUIRED_BEHAVIOR_METRIC_COLUMNS):
            return cached

    windows = pd.read_csv(behavior_windows_csv)
    rows: list[dict[str, Any]] = []
    total = len(windows)
    for idx, row in enumerate(windows.itertuples(index=False), start=1):
        series = pd.Series(row._asdict())
        rows.append(_snippet_metrics_for_row(series))
        if idx % 2000 == 0:
            print(f"computed bridge behavior snippet metrics for {idx}/{total} windows", flush=True)
    metrics = pd.DataFrame(rows)
    out = pd.concat([windows.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False)
    return out


def _curve_for(
    model_values: pd.DataFrame,
    *,
    population_key: str,
    metric_family: str,
    component: str,
) -> pd.DataFrame:
    sub = model_values[
        model_values["population_key"].astype(str).eq(str(population_key))
        & model_values["metric_family"].astype(str).eq(str(metric_family))
        & model_values["component"].astype(str).eq(str(component))
    ].copy()
    sub = sub.sort_values("plot_median")
    sub = sub[np.isfinite(sub["plot_median"].to_numpy(dtype=float))]
    sub = sub.drop_duplicates("plot_median", keep="last")
    return sub


def _interpolate_curve(x: np.ndarray, curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_model = curve["plot_median"].to_numpy(dtype=float)
    y_model = curve["ssi_percent_vs_cell_baseline"].to_numpy(dtype=float)
    ok = np.isfinite(x_model) & np.isfinite(y_model)
    x_model = x_model[ok]
    y_model = y_model[ok]
    order = np.argsort(x_model)
    x_model = x_model[order]
    y_model = y_model[order]
    x_values = np.asarray(x, dtype=float)
    pred = np.interp(x_values, x_model, y_model, left=np.nan, right=np.nan)
    outside = (~np.isfinite(x_values)) | (x_values < x_model[0]) | (x_values > x_model[-1])
    return pred, outside


def _bootstrap_mean(values: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(arr))
    if arr.size <= 1 or n_bootstrap <= 0:
        return point, float("nan"), float("nan")
    sample = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    boots = np.mean(arr[sample], axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _region_for_values(values: np.ndarray, *, q25: float, q75: float, tail_low: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, "outside_or_nan", dtype=object)
    finite = np.isfinite(arr)
    out[finite & (arr < q25)] = "below_q25"
    out[finite & (arr >= q25) & (arr <= q75)] = "q25_q75"
    out[finite & (arr > q75) & (arr < tail_low)] = "q75_to_tail"
    out[finite & (arr >= tail_low)] = "final_tail"
    return out


def make_predictions(
    behavior: pd.DataFrame,
    model_values: pd.DataFrame,
    model_reference: pd.DataFrame,
    populations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    tail_rows: list[dict[str, Any]] = []
    ref_lookup = {str(row.metric_family): row for row in model_reference.itertuples(index=False)}

    base_cols = ["session", "subject", "coherence_bin", "coherence_band_order", "image_orientation_coherence"]
    for population_key in populations["population_key"].astype(str):
        for family in METRIC_FAMILIES:
            family_key = str(family["key"])
            ref = ref_lookup[family_key]
            for component, component_label, _color, _linestyle, _marker in COMPONENTS:
                behavior_col = str(family["across_behavior"] if component == "across" else family["along_behavior"])
                curve = _curve_for(
                    model_values,
                    population_key=population_key,
                    metric_family=family_key,
                    component=component,
                )
                x = pd.to_numeric(behavior[behavior_col], errors="coerce").to_numpy(dtype=float)
                pred, outside = _interpolate_curve(x, curve)
                last_bin_low = float(np.nanmax(pd.to_numeric(curve["component_min"], errors="coerce").to_numpy(dtype=float)))
                regions = _region_for_values(
                    x,
                    q25=float(ref.q25),
                    q75=float(ref.q75),
                    tail_low=last_bin_low,
                )
                frame = behavior[base_cols].copy()
                frame["population_key"] = population_key
                frame["metric_family"] = family_key
                frame["metric_title"] = str(family["title"])
                frame["component"] = component
                frame["component_label"] = component_label
                frame["behavior_metric"] = behavior_col
                frame["behavior_dose"] = x
                frame["predicted_ssi_residual"] = pred
                frame["outside_model_range"] = outside
                frame["dose_region"] = regions
                prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    summary = summarize_predictions(predictions)

    group_cols = ["population_key", "metric_family", "component", "coherence_bin", "dose_region"]
    for keys, sub in predictions.groupby(group_cols, observed=True, sort=False):
        population_key, metric_family, component, coherence_bin, region = keys
        all_group = predictions[
            predictions["population_key"].astype(str).eq(str(population_key))
            & predictions["metric_family"].astype(str).eq(str(metric_family))
            & predictions["component"].astype(str).eq(str(component))
            & predictions["coherence_bin"].astype(str).eq(str(coherence_bin))
        ]
        n_total = max(int(len(all_group)), 1)
        all_pred = pd.to_numeric(all_group["predicted_ssi_residual"], errors="coerce").to_numpy(dtype=float)
        n_valid_total = int(np.count_nonzero(np.isfinite(all_pred)))
        pred = pd.to_numeric(sub["predicted_ssi_residual"], errors="coerce").to_numpy(dtype=float)
        outside_region = sub["outside_model_range"].to_numpy(dtype=bool)
        tail_rows.append(
            {
                "population_key": population_key,
                "metric_family": metric_family,
                "component": component,
                "coherence_bin": coherence_bin,
                "dose_region": region,
                "n_windows": int(len(sub)),
                "n_valid_predictions": int(np.count_nonzero(np.isfinite(pred))),
                "fraction_windows": float(len(sub) / n_total),
                "fraction_outside_model_range": float(np.mean(outside_region)) if outside_region.size else float("nan"),
                "mean_predicted_residual": float(np.nanmean(pred)) if np.isfinite(pred).any() else float("nan"),
                "contribution_to_valid_prediction_mean": float(np.nansum(pred) / n_valid_total)
                if n_valid_total > 0
                else float("nan"),
                "contribution_to_all_windows_zero_for_unmodeled": float(np.nansum(pred) / n_total)
                if pred.size
                else float("nan"),
            }
        )
    tail = pd.DataFrame(tail_rows)
    contrasts = summarize_high_low_contrasts(predictions)
    return summary, contrasts, tail


def summarize_predictions(predictions: pd.DataFrame, *, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["population_key", "metric_family", "component", "coherence_bin"]
    for idx, (keys, sub) in enumerate(predictions.groupby(group_cols, observed=True, sort=False)):
        population_key, metric_family, component, coherence_bin = keys
        session_values = (
            sub.groupby("session", observed=True)["predicted_ssi_residual"]
            .mean(numeric_only=True)
            .to_numpy(dtype=float)
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED + idx)
        point, lo, hi = _bootstrap_mean(session_values, rng=rng, n_bootstrap=n_bootstrap)
        doses = pd.to_numeric(sub["behavior_dose"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(sub["predicted_ssi_residual"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "population_key": population_key,
                "metric_family": metric_family,
                "component": component,
                "coherence_bin": coherence_bin,
                "coherence_band_order": int(pd.to_numeric(sub["coherence_band_order"], errors="coerce").dropna().iloc[0])
                if sub["coherence_band_order"].notna().any()
                else -1,
                "session_mean_predicted_residual": point,
                "session_ci95_low": lo,
                "session_ci95_high": hi,
                "window_mean_predicted_residual": float(np.nanmean(pred)) if np.isfinite(pred).any() else float("nan"),
                "behavior_dose_q25": float(np.nanpercentile(doses, 25.0)) if np.isfinite(doses).any() else float("nan"),
                "behavior_dose_median": float(np.nanmedian(doses)) if np.isfinite(doses).any() else float("nan"),
                "behavior_dose_q75": float(np.nanpercentile(doses, 75.0)) if np.isfinite(doses).any() else float("nan"),
                "n_windows": int(len(sub)),
                "n_sessions": int(sub["session"].nunique()),
                "outside_model_range_fraction": float(np.nanmean(sub["outside_model_range"].to_numpy(dtype=bool))),
            }
        )
    return pd.DataFrame(rows)


def summarize_high_low_contrasts(predictions: pd.DataFrame, *, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["population_key", "metric_family", "component"]
    for idx, (keys, sub) in enumerate(predictions.groupby(group_cols, observed=True, sort=False)):
        population_key, metric_family, component = keys
        session_bin = (
            sub.groupby(["session", "coherence_bin"], observed=True)["predicted_ssi_residual"]
            .mean(numeric_only=True)
            .reset_index()
        )
        piv = session_bin.pivot(index="session", columns="coherence_bin", values="predicted_ssi_residual")
        if "0-0.2" not in piv.columns or "0.8-1" not in piv.columns:
            continue
        diff = (piv["0.8-1"] - piv["0-0.2"]).to_numpy(dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 5000 + idx)
        point, lo, hi = _bootstrap_mean(diff, rng=rng, n_bootstrap=n_bootstrap)
        rows.append(
            {
                "population_key": population_key,
                "metric_family": metric_family,
                "component": component,
                "contrast": "high_coherence_0p8_1_minus_low_0_0p2",
                "session_mean_delta": point,
                "session_ci95_low": lo,
                "session_ci95_high": hi,
                "n_sessions_paired": int(np.count_nonzero(np.isfinite(diff))),
            }
        )
    return pd.DataFrame(rows)


def _add_reference_band(ax: plt.Axes, reference: pd.DataFrame, metric_family: str) -> None:
    ref = reference[reference["metric_family"].astype(str).eq(str(metric_family))]
    if ref.empty:
        return
    row = ref.iloc[0]
    low = float(row["q25"])
    high = float(row["q75"])
    if np.isfinite(low) and np.isfinite(high) and high > low:
        ax.axvspan(low, high, color=GRAY, alpha=0.12, lw=0, zorder=0, label="trace-bank q25-q75")


def _set_distribution_xlim(
    ax: plt.Axes,
    behavior: pd.DataFrame,
    family: dict[str, Any],
    curves: list[pd.DataFrame],
) -> None:
    behavior_cols = [str(family["across_behavior"]), str(family["along_behavior"])]
    behavior_values = np.concatenate(
        [
            pd.to_numeric(behavior[col], errors="coerce").to_numpy(dtype=float)
            for col in behavior_cols
            if col in behavior.columns
        ]
    )
    behavior_values = behavior_values[np.isfinite(behavior_values)]
    model_values = np.concatenate(
        [
            pd.to_numeric(curve["plot_median"], errors="coerce").to_numpy(dtype=float)
            for curve in curves
            if not curve.empty
        ]
    )
    model_values = model_values[np.isfinite(model_values)]
    candidates = np.concatenate([behavior_values, model_values]) if model_values.size else behavior_values
    if candidates.size == 0:
        return

    x_hi = (
        float(np.nanpercentile(behavior_values, DISPLAY_BEHAVIOR_QUANTILE))
        if behavior_values.size
        else float(np.nanmax(candidates))
    )
    if model_values.size:
        x_hi = max(x_hi, float(np.nanmax(model_values)))
    x_hi = max(x_hi, 1.0)

    if str(family["key"]) == "path_per_range":
        x_lo = float(np.nanpercentile(behavior_values, 0.5)) if behavior_values.size else float(np.nanmin(candidates))
        if model_values.size:
            x_lo = min(x_lo, float(np.nanmin(model_values)))
        pad = 0.04 * max(x_hi - x_lo, 1.0)
        ax.set_xlim(max(0.0, x_lo - pad), x_hi + pad)
    else:
        ax.set_xlim(0.0, x_hi * 1.04)


def plot_distribution_on_curves(
    behavior: pd.DataFrame,
    model_values: pd.DataFrame,
    model_reference: pd.DataFrame,
    populations: pd.DataFrame,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    configure_matplotlib()
    pdf = out_dir / f"{OUT_STEM}_distribution_on_curves.pdf"
    png_paths: dict[str, Path] = {}
    with PdfPages(pdf) as pages:
        for pop in populations.itertuples(index=False):
            population_key = str(pop.population_key)
            fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.1), sharey=False, constrained_layout=True)
            for ax, family in zip(axes.ravel(), METRIC_FAMILIES, strict=True):
                family_key = str(family["key"])
                _add_reference_band(ax, model_reference, family_key)
                ax_hist = ax.twinx()
                ax_hist.set_zorder(ax.get_zorder() - 1)
                ax.set_zorder(ax_hist.get_zorder() + 1)
                ax.patch.set_visible(False)
                normal_col = str(family["across_behavior"])
                for coherence, color in COHERENCE_COLORS.items():
                    vals = pd.to_numeric(
                        behavior.loc[behavior["coherence_bin"].astype(str).eq(coherence), normal_col],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        ax_hist.hist(vals, bins=28, density=True, color=color, alpha=0.22, label=f"normal behavior {coherence}")
                ax_hist.set_yticks([])
                for spine in ax_hist.spines.values():
                    spine.set_visible(False)
                component_curves: list[pd.DataFrame] = []
                for component, label, color, linestyle, marker in COMPONENTS:
                    curve = _curve_for(model_values, population_key=population_key, metric_family=family_key, component=component)
                    component_curves.append(curve)
                    ax.plot(
                        curve["plot_median"],
                        curve["ssi_percent_vs_cell_baseline"],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markerfacecolor="white",
                        lw=1.7,
                        markersize=3.5,
                        label=label,
                    )
                _set_distribution_xlim(ax, behavior, family, component_curves)
                ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
                ax.set_title(str(family["title"]), loc="left", fontweight="bold")
                ax.set_xlabel(str(family["xlabel"]))
                ax.grid(axis="y", color=GRID, lw=0.75)
                _clean_axis(ax)
            axes[0, 0].set_ylabel("model SSI residual (% vs cell baseline)")
            axes[1, 0].set_ylabel("model SSI residual (% vs cell baseline)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            handles.extend(
                [
                    Patch(facecolor=COHERENCE_COLORS["0-0.2"], alpha=0.22, label="behavior normal 0-0.2"),
                    Patch(facecolor=COHERENCE_COLORS["0.8-1"], alpha=0.22, label="behavior normal 0.8-1"),
                ]
            )
            labels.extend(["behavior normal 0-0.2", "behavior normal 0.8-1"])
            axes[0, 0].legend(handles, labels, frameon=False, fontsize=7, loc="best")
            fig.suptitle(
                f"Behavior distributions on model curves: {pop.population_title}\n{pop.population_subtitle}",
                fontsize=12.2,
                fontweight="bold",
            )
            png = out_dir / f"{OUT_STEM}_distribution_on_curves_{population_key}.png"
            fig.savefig(png, dpi=230)
            pages.savefig(fig)
            png_paths[population_key] = png
            plt.close(fig)
    return {"pdf": pdf, **{f"png_{k}": v for k, v in png_paths.items()}}


def plot_prediction_pages(
    summary: pd.DataFrame,
    populations: pd.DataFrame,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    configure_matplotlib()
    pdf = out_dir / f"{OUT_STEM}_predicted_ssi_by_coherence.pdf"
    png_paths: dict[str, Path] = {}
    x = np.arange(len(COHERENCE_ORDER), dtype=float)
    with PdfPages(pdf) as pages:
        for pop in populations.itertuples(index=False):
            population_key = str(pop.population_key)
            fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.0), sharey=True, constrained_layout=True)
            for ax, family in zip(axes.ravel(), METRIC_FAMILIES, strict=True):
                family_key = str(family["key"])
                frame = summary[
                    summary["population_key"].astype(str).eq(population_key)
                    & summary["metric_family"].astype(str).eq(family_key)
                ].copy()
                for component, label, color, linestyle, marker in COMPONENTS:
                    sub = frame[frame["component"].astype(str).eq(component)].copy()
                    sub["coherence_bin"] = pd.Categorical(sub["coherence_bin"], categories=COHERENCE_ORDER, ordered=True)
                    sub = sub.sort_values("coherence_bin")
                    y = sub["session_mean_predicted_residual"].to_numpy(dtype=float)
                    lo = sub["session_ci95_low"].to_numpy(dtype=float)
                    hi = sub["session_ci95_high"].to_numpy(dtype=float)
                    ax.errorbar(
                        x,
                        y,
                        yerr=np.vstack([y - lo, hi - y]),
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markerfacecolor="white",
                        lw=1.75,
                        capsize=2.5,
                        label=label,
                    )
                ax.axhline(0.0, color="0.35", lw=0.85, ls=":")
                ax.set_xticks(x)
                ax.set_xticklabels(COHERENCE_ORDER)
                ax.set_xlabel("behavior local edge coherence")
                ax.set_title(str(family["title"]), loc="left", fontweight="bold")
                ax.grid(axis="y", color=GRID, lw=0.75)
                _clean_axis(ax)
            axes[0, 0].set_ylabel("behavior-weighted model SSI residual (%)")
            axes[1, 0].set_ylabel("behavior-weighted model SSI residual (%)")
            axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
            fig.suptitle(
                f"Behavior-weighted model prediction: {pop.population_title}\n{pop.population_subtitle}",
                fontsize=12.2,
                fontweight="bold",
            )
            png = out_dir / f"{OUT_STEM}_predicted_ssi_by_coherence_{population_key}.png"
            fig.savefig(png, dpi=230)
            pages.savefig(fig)
            png_paths[population_key] = png
            plt.close(fig)
    return {"pdf": pdf, **{f"png_{k}": v for k, v in png_paths.items()}}


def plot_tail_contributions(tail: pd.DataFrame, *, out_dir: Path) -> dict[str, Path]:
    configure_matplotlib()
    region_order = ["below_q25", "q25_q75", "q75_to_tail", "final_tail"]
    metric_order = ["component_path", "component_rms", "component_range"]
    frame = tail[
        tail["population_key"].astype(str).eq("high_sf_aligned")
        & tail["component"].astype(str).eq("across")
        & tail["metric_family"].astype(str).isin(metric_order)
        & tail["coherence_bin"].astype(str).isin(["0-0.2", "0.8-1"])
    ].copy()
    fig, axes = plt.subplots(1, len(metric_order), figsize=(11.0, 3.3), sharey=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    x = np.arange(len(region_order), dtype=float)
    for ax, metric in zip(axes_arr, metric_order, strict=True):
        sub_metric = frame[frame["metric_family"].astype(str).eq(metric)].copy()
        for coherence, color, offset in [("0-0.2", "#9aa5b1", -0.18), ("0.8-1", "#0b4f83", 0.18)]:
            sub = sub_metric[sub_metric["coherence_bin"].astype(str).eq(coherence)].set_index("dose_region")
            vals = [float(sub.loc[region, "fraction_windows"]) if region in sub.index else 0.0 for region in region_order]
            ax.bar(x + offset, vals, width=0.32, color=color, alpha=0.75, label=coherence)
        ax.set_xticks(x)
        ax.set_xticklabels(["<q25", "q25-q75", "q75-tail", "tail"], rotation=20)
        ax.set_title(metric.replace("_", " "), loc="left", fontweight="bold")
        ax.set_xlabel("normal-dose region")
        ax.grid(axis="y", color=GRID, lw=0.75)
        _clean_axis(ax)
    axes_arr[0].set_ylabel("fraction of behavior windows")
    axes_arr[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Aligned high-SF: behavior normal-dose occupancy by tail region", fontsize=12.0, fontweight="bold")
    png = out_dir / f"{OUT_STEM}_tail_region_occupancy_high_sf_aligned.png"
    pdf = out_dir / f"{OUT_STEM}_tail_region_occupancy_high_sf_aligned.pdf"
    fig.savefig(png, dpi=230)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def build(
    out_dir: Path = OUT_DIR,
    *,
    force_recompute_behavior: bool = False,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_metrics_csv = out_dir / f"{OUT_STEM}_behavior_snippet_metrics.csv"
    prediction_summary_csv = out_dir / f"{OUT_STEM}_prediction_summary.csv"
    coherence_contrasts_csv = out_dir / f"{OUT_STEM}_coherence_contrasts.csv"
    tail_contributions_csv = out_dir / f"{OUT_STEM}_tail_contributions.csv"
    provenance_json = out_dir / f"{OUT_STEM}_provenance.json"

    behavior = compute_behavior_metrics(cache_csv=behavior_metrics_csv, force_recompute=force_recompute_behavior)
    model_values = pd.read_csv(MODEL_VALUES_CSV)
    model_reference = pd.read_csv(MODEL_REFERENCE_CSV)
    populations = pd.read_csv(MODEL_POPULATIONS_CSV)
    summary, contrasts, tail = make_predictions(behavior, model_values, model_reference, populations)
    summary.to_csv(prediction_summary_csv, index=False)
    contrasts.to_csv(coherence_contrasts_csv, index=False)
    tail.to_csv(tail_contributions_csv, index=False)

    distribution_paths = plot_distribution_on_curves(behavior, model_values, model_reference, populations, out_dir=out_dir)
    prediction_paths = plot_prediction_pages(summary, populations, out_dir=out_dir)
    tail_paths = plot_tail_contributions(tail, out_dir=out_dir)

    outputs = {
        "behavior_metrics_csv": behavior_metrics_csv,
        "prediction_summary_csv": prediction_summary_csv,
        "coherence_contrasts_csv": coherence_contrasts_csv,
        "tail_contributions_csv": tail_contributions_csv,
        "distribution_on_curves_pdf": distribution_paths["pdf"],
        "predicted_ssi_by_coherence_pdf": prediction_paths["pdf"],
        "tail_region_occupancy_png": tail_paths["png"],
        "tail_region_occupancy_pdf": tail_paths["pdf"],
        "provenance_json": provenance_json,
    }
    _write_json(
        provenance_json,
        {
            "analysis": OUT_STEM,
            "inputs": {
                "model_values_csv": MODEL_VALUES_CSV,
                "model_reference_csv": MODEL_REFERENCE_CSV,
                "model_populations_csv": MODEL_POPULATIONS_CSV,
                "behavior_windows_csv": BEHAVIOR_WINDOWS_CSV,
            },
            "behavior_metrics": {
                "snippet_n_samples": PANEL_G_SNIPPET_N_SAMPLES,
                "snippet_duration_s": (PANEL_G_SNIPPET_N_SAMPLES - 1) * DT,
                "definition": "central 40-sample behavior snippet from each reviewed BackImage window, projected onto local contour tangent/normal axes",
                "n_windows": int(len(behavior)),
                "n_sessions": int(behavior["session"].nunique()),
            },
            "prediction": {
                "interpolation": "piecewise-linear interpolation through Panel-G alternative-axis model bin medians; outside model range set to NaN and reported in summaries",
                "summary_unit": "session mean, bootstrapped over sessions",
                "n_bootstrap": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "coherence_bins": COHERENCE_ORDER,
            },
            "display": {
                "distribution_on_curves_xlim": (
                    "model curve range plus behavior "
                    f"{DISPLAY_BEHAVIOR_QUANTILE:g}th percentile; full behavior doses remain in CSVs"
                )
            },
            "outputs": outputs,
        },
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force-recompute-behavior", action="store_true")
    args = parser.parse_args()
    paths = build(args.out_dir, force_recompute_behavior=bool(args.force_recompute_behavior))
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
