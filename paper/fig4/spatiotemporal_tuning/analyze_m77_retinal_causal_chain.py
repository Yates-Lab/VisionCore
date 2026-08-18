#!/usr/bin/env python3
"""Merge M77 retinal causal-chain shards and evaluate held-out predictions.

The analysis keeps image and eye-trace identity explicit.  Five image folds
are crossed with five trace folds; a test cell is predicted only by a model
fit without either its images or its traces.  Motion effects are always
defined relative to the matched stabilized rendering of the same image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EPS = 1e-12
PREDICTORS = {
    "total power": "total_dynamic_power",
    "TF marginal": "tf_marginal_power",
    "SF×orientation marginal": "sf_orientation_marginal_power",
    "separable SF×TF×orientation": "separable_passband_power",
    "full joint passband": "joint_passband_power",
    "signed joint rate drive": "joint_signed_rate_drive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-shards", type=Path, nargs="*", default=())
    parser.add_argument("--rotated-shards", type=Path, nargs="*", default=())
    parser.add_argument("--image-folds", type=int, default=5)
    parser.add_argument("--trace-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--exemplar-unit", type=int, default=25)
    return parser.parse_args()


def _archive_path(path: Path) -> Path:
    return path / "causal_chain_shard.npz" if path.is_dir() else path


def load_and_merge_shards(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    archives = []
    for raw_path in paths:
        path = _archive_path(Path(raw_path))
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as handle:
            archives.append({key: handle[key] for key in handle.files})
    if not archives:
        raise ValueError("at least one shard is required")
    archives.sort(key=lambda item: int(np.min(item["image_indices"])))
    reference = archives[0]
    fixed = (
        "trace_indices",
        "motion_scales",
        "unit_indices",
        "spatial_cpd",
        "temporal_hz",
        "orientation_deg",
    )
    for archive in archives[1:]:
        for key in fixed:
            if not np.array_equal(reference[key], archive[key]):
                raise ValueError(f"shards disagree on {key}")
    image_indices = np.concatenate([item["image_indices"] for item in archives])
    if len(np.unique(image_indices)) != len(image_indices):
        raise ValueError("shards contain duplicate image rows")
    order = np.argsort(image_indices)
    result: dict[str, np.ndarray] = {key: reference[key] for key in fixed}
    result["image_indices"] = image_indices[order]
    image_keys = (
        "mean_rate",
        "expected_spikes",
        "map_ssi",
        *PREDICTORS.values(),
    )
    for key in image_keys:
        value = np.concatenate([item[key] for item in archives], axis=0)
        result[key] = value[order]
    weights = np.asarray([len(item["image_indices"]) for item in archives], dtype=float)
    result["average_power"] = np.average(
        np.stack([item["average_power"] for item in archives]), axis=0, weights=weights
    )
    result["example_power"] = reference.get("example_power", np.empty((0,)))
    result["example_rate_maps"] = reference.get("example_rate_maps", np.empty((0,)))
    result["example_movie_frames"] = reference.get("example_movie_frames", np.empty((0,)))
    result["example_trace_xy"] = reference.get("example_trace_xy", np.empty((0,)))
    return result


def matched_motion_delta(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError("motion value must have shape [image,trace,scale,unit]")
    return array[:, :, 1:] - array[:, :, :1]


def crossed_predictions(
    feature: np.ndarray,
    outcome: np.ndarray,
    *,
    image_folds: int,
    trace_folds: int,
) -> np.ndarray:
    """Return crossed image×trace held-out scalar-regression predictions."""
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("feature and outcome must match [image,trace,condition]")
    n_image, n_trace, _ = x.shape
    n_ifold = min(max(int(image_folds), 2), n_image)
    n_tfold = min(max(int(trace_folds), 2), n_trace)
    if n_image < 2 or n_trace < 2:
        raise ValueError("crossed validation requires at least two images and traces")
    image_groups = np.array_split(np.arange(n_image), n_ifold)
    trace_groups = np.array_split(np.arange(n_trace), n_tfold)
    prediction = np.full_like(y, np.nan)
    for held_images in image_groups:
        for held_traces in trace_groups:
            image_test = np.zeros(n_image, dtype=bool)
            trace_test = np.zeros(n_trace, dtype=bool)
            image_test[held_images] = True
            trace_test[held_traces] = True
            train_mask = (~image_test)[:, None] & (~trace_test)[None, :]
            test_mask = image_test[:, None] & trace_test[None, :]
            train_x = x[train_mask].reshape(-1)
            train_y = y[train_mask].reshape(-1)
            valid = np.isfinite(train_x) & np.isfinite(train_y)
            train_x, train_y = train_x[valid], train_y[valid]
            if len(train_x) < 4:
                continue
            center = float(np.mean(train_x))
            scale = float(np.std(train_x))
            if scale <= EPS:
                coefficients = np.asarray((float(np.mean(train_y)), 0.0))
            else:
                design = np.column_stack((np.ones(len(train_x)), (train_x - center) / scale))
                coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
            test_x = x[test_mask].reshape(-1)
            test_prediction = coefficients[0] + coefficients[1] * (test_x - center) / max(scale, EPS)
            prediction[test_mask] = test_prediction.reshape(prediction[test_mask].shape)
    return prediction


def prediction_metrics(outcome: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = np.asarray(outcome, dtype=np.float64).reshape(-1)
    fitted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(observed) & np.isfinite(fitted)
    observed, fitted = observed[valid], fitted[valid]
    if len(observed) < 4:
        return {"spearman": np.nan, "cv_r2": np.nan, "calibration_slope": np.nan}
    correlation = float(spearmanr(observed, fitted).statistic)
    denominator = float(np.sum(np.square(observed - observed.mean())))
    r2 = 1.0 - float(np.sum(np.square(observed - fitted))) / max(denominator, EPS)
    calibration = float(
        np.linalg.lstsq(
            np.column_stack((np.ones(len(fitted)), fitted)), observed, rcond=None
        )[0][1]
    )
    return {"spearman": correlation, "cv_r2": r2, "calibration_slope": calibration}


def evaluate_predictor(
    feature: np.ndarray,
    outcome: np.ndarray,
    *,
    image_folds: int,
    trace_folds: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if feature.shape != outcome.shape or feature.ndim != 4:
        raise ValueError("feature and outcome must match [image,trace,scale,unit]")
    prediction = np.full_like(outcome, np.nan, dtype=np.float64)
    rows = []
    for unit in range(outcome.shape[-1]):
        prediction[..., unit] = crossed_predictions(
            feature[..., unit],
            outcome[..., unit],
            image_folds=image_folds,
            trace_folds=trace_folds,
        )
        rows.append({"unit_local_index": unit, **prediction_metrics(outcome[..., unit], prediction[..., unit])})
    return prediction, pd.DataFrame(rows)


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return np.nan, np.nan, np.nan
    draws = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        draws[index] = np.median(rng.choice(value, size=len(value), replace=True))
    return float(np.median(value)), *map(float, np.quantile(draws, (0.025, 0.975)))


def paired_bootstrap_median_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return np.nan, np.nan, np.nan
    draws = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        draws[index] = np.median(rng.choice(difference, size=len(difference), replace=True))
    return float(np.median(difference)), *map(float, np.quantile(draws, (0.025, 0.975)))


def hierarchical_effect_ci(
    effect: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Resample image, trace, and unit clusters for a population median."""
    value = np.asarray(effect, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("effect must have shape [image,trace,unit]")
    n_image, n_trace, n_unit = value.shape
    observed = float(np.nanmedian(value))
    draws = np.empty(int(n_bootstrap), dtype=float)
    for draw in range(int(n_bootstrap)):
        image_index = rng.integers(0, n_image, n_image)
        trace_index = rng.integers(0, n_trace, n_trace)
        unit_index = rng.integers(0, n_unit, n_unit)
        sampled = value[np.ix_(image_index, trace_index, unit_index)]
        draws[draw] = np.nanmedian(sampled)
    return observed, *map(float, np.quantile(draws, (0.025, 0.975)))


def evaluate_dataset(data: dict[str, np.ndarray], args: argparse.Namespace):
    rate_delta = matched_motion_delta(data["mean_rate"])
    spike_delta = matched_motion_delta(data["expected_spikes"])
    ssi_delta = matched_motion_delta(data["map_ssi"])
    results: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for label, key in PREDICTORS.items():
        feature = matched_motion_delta(data[key])
        results[label] = evaluate_predictor(
            feature,
            rate_delta,
            image_folds=args.image_folds,
            trace_folds=args.trace_folds,
        )
    return rate_delta, spike_delta, ssi_delta, results


def save_prediction_table(
    data: dict[str, np.ndarray],
    results: dict[str, tuple[np.ndarray, pd.DataFrame]],
    path: Path,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    units = data["unit_indices"].astype(int)
    for label, (_, metrics) in results.items():
        for metric in ("spearman", "cv_r2", "calibration_slope"):
            center, low, high = bootstrap_median_ci(
                metrics[metric].to_numpy(float), n_bootstrap=n_bootstrap, rng=rng
            )
            rows.append(
                {
                    "predictor": label,
                    "metric": metric,
                    "median": center,
                    "ci_low": low,
                    "ci_high": high,
                    "n_units": int(len(units)),
                }
            )
        unit_frame = metrics.copy()
        unit_frame["unit_index"] = units[unit_frame.unit_local_index.to_numpy(int)]
        unit_frame["predictor"] = label
        unit_frame.to_csv(path.parent / f"per_unit_{label.lower().replace(' ', '_').replace('×', 'x')}.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(path, index=False)
    return summary


def effect_table(
    data: dict[str, np.ndarray],
    rate_delta: np.ndarray,
    spike_delta: np.ndarray,
    ssi_delta: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for scale_index, scale in enumerate(data["motion_scales"][1:]):
        for label, value in (
            ("mean_rate_hz", rate_delta[:, :, scale_index]),
            ("expected_spikes_1s", spike_delta[:, :, scale_index]),
            ("map_ssi_bits_per_spike", ssi_delta[:, :, scale_index]),
        ):
            center, low, high = hierarchical_effect_ci(
                value, n_bootstrap=n_bootstrap, rng=rng
            )
            rows.append(
                {
                    "motion_scale": float(scale),
                    "measure": label,
                    "median_change": center,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_tf_sf(power: np.ndarray) -> np.ndarray:
    value = np.asarray(power, dtype=float).sum(axis=-1)
    return value / max(float(np.sum(value)), EPS)


def render_overview(
    data: dict[str, np.ndarray],
    rate_delta: np.ndarray,
    ssi_delta: np.ndarray,
    results: dict[str, tuple[np.ndarray, pd.DataFrame]],
    predictor_summary: pd.DataFrame,
    effects: pd.DataFrame,
    path: Path,
    exemplar_unit: int,
) -> None:
    spatial = data["spatial_cpd"]
    temporal = data["temporal_hz"]
    scales = data["motion_scales"]
    figure = plt.figure(figsize=(15.8, 10.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.88))
    top = grid[0, :].subgridspec(1, len(scales))
    maps = [_aggregate_tf_sf(power) for power in data["average_power"]]
    positive = np.concatenate([item[item > 0] for item in maps[1:] if np.any(item > 0)])
    floor = max(float(np.quantile(positive, 0.02)), EPS)
    ceiling = max(float(np.quantile(positive, 0.995)), floor * 10)
    contour = None
    for index, scale in enumerate(scales):
        axis = figure.add_subplot(top[0, index])
        display = np.log10(np.maximum(maps[index].T, floor))
        contour = axis.contourf(
            spatial,
            temporal,
            display,
            levels=np.linspace(np.log10(floor), np.log10(ceiling), 13),
            cmap="magma",
            extend="both",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_title(f"{scale:g}× measured motion")
        axis.set_xlabel("spatial frequency (cycles/degree)")
        if index == 0:
            axis.set_ylabel("temporal frequency (Hz)")
    if contour is not None:
        bar = figure.colorbar(contour, ax=figure.axes[: len(scales)], shrink=0.75, pad=0.012)
        bar.set_label("log10 fraction of dynamic power")

    axis = figure.add_subplot(grid[1, 0])
    unit_matches = np.flatnonzero(data["unit_indices"].astype(int) == int(exemplar_unit))
    unit = int(unit_matches[0]) if len(unit_matches) else 0
    observed = rate_delta[..., unit].reshape(-1)
    prediction = results["signed joint rate drive"][0][..., unit].reshape(-1)
    valid = np.isfinite(observed) & np.isfinite(prediction)
    if np.any(valid):
        axis.scatter(prediction[valid], observed[valid], s=7, alpha=0.16, color="#2774AE", rasterized=True)
        low = min(float(np.min(prediction[valid])), float(np.min(observed[valid])))
        high = max(float(np.max(prediction[valid])), float(np.max(observed[valid])))
        axis.plot((low, high), (low, high), color="0.25", lw=1)
    else:
        axis.text(0.5, 0.5, "smoke bank too small for crossed CV", ha="center", va="center", transform=axis.transAxes)
    metrics = prediction_metrics(observed[valid], prediction[valid])
    axis.set_title(f"u{int(data['unit_indices'][unit]):03d}: joint projection → rate\nheld-out ρ={metrics['spearman']:.2f}, R²={metrics['cv_r2']:.2f}")
    axis.set_xlabel("held-out predicted Δ rate (Hz)")
    axis.set_ylabel("M77 replay Δ rate (Hz)")

    axis = figure.add_subplot(grid[1, 1])
    r2 = predictor_summary.loc[predictor_summary.metric.eq("cv_r2")].copy()
    order = list(PREDICTORS)
    r2["order"] = r2.predictor.map({name: i for i, name in enumerate(order)})
    r2 = r2.sort_values("order")
    positions = np.arange(len(r2))
    axis.errorbar(
        positions,
        r2["median"],
        yerr=np.vstack((r2["median"] - r2["ci_low"], r2["ci_high"] - r2["median"])),
        fmt="o",
        color="#1F77B4",
        capsize=3,
    )
    axis.axhline(0, color="0.55", lw=0.8)
    axis.set_xticks(positions, [name.replace(" ", "\n", 1) for name in r2.predictor], rotation=24, ha="right")
    axis.set_ylabel("cross-validated R² across units")
    axis.set_title("Joint tuning must beat power-only controls")

    axis = figure.add_subplot(grid[1, 2])
    rate_effect = effects.loc[effects.measure.eq("mean_rate_hz")].sort_values("motion_scale")
    rate_line = axis.errorbar(
        rate_effect.motion_scale,
        rate_effect.median_change,
        yerr=np.vstack((rate_effect.median_change - rate_effect.ci_low, rate_effect.ci_high - rate_effect.median_change)),
        marker="o",
        capsize=3,
        color="#E8752E",
        label="rate / expected spikes in 1 s",
    )
    twin = axis.twinx()
    ssi_effect = effects.loc[effects.measure.eq("map_ssi_bits_per_spike")].sort_values("motion_scale")
    ssi_line = twin.errorbar(
        ssi_effect.motion_scale,
        ssi_effect.median_change,
        yerr=np.vstack((ssi_effect.median_change - ssi_effect.ci_low, ssi_effect.ci_high - ssi_effect.median_change)),
        marker="s",
        capsize=3,
        color="#6A51A3",
        label="map SSI",
    )
    axis.axhline(0, color="0.45", lw=0.8, ls="--")
    axis.set_xlabel("measured-motion scale")
    axis.set_ylabel("Δ rate (spikes/s) = Δ spikes in 1 s", color="#E8752E")
    twin.set_ylabel("Δ map SSI (bits/spike)", color="#6A51A3")
    axis.set_title("Motion simultaneously changes spike count and SSI")
    axis.legend([rate_line, ssi_line], ["rate / 1-s spikes", "map SSI"], frameon=False)
    figure.suptitle("M77 retinal-motion causal chain · zero behavior", fontsize=17, fontweight="semibold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_per_unit_pages(
    path: Path,
    *,
    data: dict[str, np.ndarray],
    rate_delta: np.ndarray,
    ssi_delta: np.ndarray,
    results: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> None:
    scales = data["motion_scales"][1:]
    joint_prediction = results["signed joint rate drive"][0]
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, len(scales)))
    with PdfPages(path) as pdf:
        for unit_local, unit_index in enumerate(data["unit_indices"].astype(int)):
            figure, axes = plt.subplots(2, 2, figsize=(10.2, 8.1), constrained_layout=True)
            observed = rate_delta[..., unit_local]
            predicted = joint_prediction[..., unit_local]
            for scale_index, scale in enumerate(scales):
                axes[0, 0].scatter(
                    predicted[:, :, scale_index].ravel(),
                    observed[:, :, scale_index].ravel(),
                    s=5,
                    alpha=0.12,
                    color=colors[scale_index],
                    rasterized=True,
                    label=f"{scale:g}×",
                )
            valid = np.isfinite(observed) & np.isfinite(predicted)
            if np.any(valid):
                low = min(float(observed[valid].min()), float(predicted[valid].min()))
                high = max(float(observed[valid].max()), float(predicted[valid].max()))
                axes[0, 0].plot((low, high), (low, high), color="0.3", lw=1)
            metrics = prediction_metrics(observed, predicted)
            axes[0, 0].set(
                xlabel="held-out projected Δ rate (spikes/s)",
                ylabel="exact M77 replay Δ rate (spikes/s)",
                title=f"Joint SF×TF×orientation prediction\nρ={metrics['spearman']:.2f}, R²={metrics['cv_r2']:.2f}, slope={metrics['calibration_slope']:.2f}",
            )
            axes[0, 0].legend(frameon=False, markerscale=2)

            for axis, value, ylabel, title in (
                (axes[0, 1], observed, "moving − stabilized rate (spikes/s)", "Rate effect"),
                (axes[1, 0], ssi_delta[..., unit_local], "moving − stabilized SSI (bits/spike)", "Spatial-information effect"),
            ):
                medians = np.nanmedian(value, axis=(0, 1))
                low = np.nanquantile(value, 0.25, axis=(0, 1))
                high = np.nanquantile(value, 0.75, axis=(0, 1))
                axis.errorbar(scales, medians, yerr=np.vstack((medians - low, high - medians)), marker="o", capsize=3)
                axis.axhline(0, color="0.5", lw=0.8, ls="--")
                axis.set(xlabel="motion scale", ylabel=ylabel, title=title)

            labels = list(PREDICTORS)
            values = [float(results[label][1].iloc[unit_local].cv_r2) for label in labels]
            axes[1, 1].barh(np.arange(len(labels)), values, color="#3B7EA1")
            axes[1, 1].axvline(0, color="0.5", lw=0.8)
            axes[1, 1].set_yticks(np.arange(len(labels)), labels)
            axes[1, 1].set_xlabel("cross-validated R²")
            axes[1, 1].set_title("Power/tuning predictor controls")
            for axis in axes.flat:
                axis.spines[["top", "right"]].set_visible(False)
            figure.suptitle(f"M77 RR100 u{unit_index:03d} · zero behavior", fontsize=15, fontweight="semibold")
            pdf.savefig(figure, bbox_inches="tight", facecolor="white")
            plt.close(figure)


def main() -> int:
    args = parse_args()
    data = load_and_merge_shards(args.shards)
    if len(data["image_indices"]) < 2 or len(data["trace_indices"]) < 2:
        raise ValueError("production analysis requires at least two images and traces")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rate_delta, spike_delta, ssi_delta, results = evaluate_dataset(data, args)
    ssi_spectral_prediction, ssi_spectral_metrics = evaluate_predictor(
        matched_motion_delta(data["joint_passband_power"]),
        ssi_delta,
        image_folds=args.image_folds,
        trace_folds=args.trace_folds,
    )
    ssi_spectral_metrics["unit_index"] = data["unit_indices"].astype(int)
    ssi_spectral_metrics.to_csv(args.out_dir / "spectral_prediction_of_ssi_gain.csv", index=False)
    predictor_summary = save_prediction_table(
        data,
        results,
        args.out_dir / "predictor_summary.csv",
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )
    effects = effect_table(
        data,
        rate_delta,
        spike_delta,
        ssi_delta,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )
    effects.to_csv(args.out_dir / "motion_effects.csv", index=False)
    np.savez_compressed(
        args.out_dir / "analysis_arrays.npz",
        image_indices=data["image_indices"],
        trace_indices=data["trace_indices"],
        unit_indices=data["unit_indices"],
        motion_scales=data["motion_scales"],
        spatial_cpd=data["spatial_cpd"],
        temporal_hz=data["temporal_hz"],
        orientation_deg=data["orientation_deg"],
        average_power=data["average_power"],
        rate_delta=rate_delta.astype(np.float32),
        spike_delta=spike_delta.astype(np.float32),
        ssi_delta=ssi_delta.astype(np.float32),
        signed_joint_prediction=results["signed joint rate drive"][0].astype(np.float32),
        example_power=data["example_power"],
        example_rate_maps=data["example_rate_maps"],
        example_movie_frames=data["example_movie_frames"],
        example_trace_xy=data["example_trace_xy"],
    )

    permutation = rng.permutation(len(data["unit_indices"]))
    shuffled = matched_motion_delta(data["joint_signed_rate_drive"])[..., permutation]
    _, shuffled_metrics = evaluate_predictor(
        shuffled,
        rate_delta,
        image_folds=args.image_folds,
        trace_folds=args.trace_folds,
    )
    shuffled_metrics["unit_index"] = data["unit_indices"].astype(int)
    shuffled_metrics.to_csv(args.out_dir / "shuffled_tuning_control.csv", index=False)

    stabilized_max = float(np.max(np.abs(data["average_power"][0])))
    joint_metrics = results["signed joint rate drive"][1]
    total_metrics = results["total power"][1]
    separable_metrics = results["separable SF×TF×orientation"][1]
    joint_over_total = paired_bootstrap_median_difference(
        joint_metrics.cv_r2,
        total_metrics.cv_r2,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )
    joint_over_separable = paired_bootstrap_median_difference(
        joint_metrics.cv_r2,
        separable_metrics.cv_r2,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )
    joint_spearman = bootstrap_median_ci(
        joint_metrics.spearman, n_bootstrap=args.n_bootstrap, rng=rng
    )
    ssi_spectral_r2 = bootstrap_median_ci(
        ssi_spectral_metrics.cv_r2, n_bootstrap=args.n_bootstrap, rng=rng
    )
    measured_index = int(np.argmin(np.abs(data["motion_scales"][1:] - 1.0)))
    measured_scale = float(data["motion_scales"][1:][measured_index])
    measured_effects = effects.loc[np.isclose(effects.motion_scale, measured_scale)]
    rate_effect = measured_effects.loc[measured_effects.measure.eq("mean_rate_hz")].iloc[0]
    ssi_effect = measured_effects.loc[measured_effects.measure.eq("map_ssi_bits_per_spike")].iloc[0]
    summary = {
        "analysis": "M77 retinal-motion causal chain; all behavior fixed to zero",
        "n_images": int(len(data["image_indices"])),
        "n_traces": int(len(data["trace_indices"])),
        "n_units": int(len(data["unit_indices"])),
        "cross_validation": f"{min(args.image_folds, len(data['image_indices']))} image folds × {min(args.trace_folds, len(data['trace_indices']))} trace folds",
        "stabilized_dynamic_power_max": stabilized_max,
        "stabilized_dynamic_power_gate_pass": bool(stabilized_max < 1e-16),
        "behavior": "identical all-zero 42-dimensional vector",
        "raw_control_analyzed": bool(args.raw_shards),
        "rotated_control_analyzed": bool(args.rotated_shards),
        "signed_joint_rate_prediction_median_spearman_ci": list(joint_spearman),
        "joint_minus_total_power_median_cv_r2_ci": list(joint_over_total),
        "joint_minus_separable_median_cv_r2_ci": list(joint_over_separable),
        "joint_passband_prediction_of_ssi_median_cv_r2_ci": list(ssi_spectral_r2),
        "claim_gates": {
            "stabilized_dynamic_power_is_zero": bool(stabilized_max < 1e-16),
            "joint_tuning_predicts_held_out_rate": bool(joint_spearman[1] > 0),
            "joint_improves_over_total_power": bool(joint_over_total[1] > 0),
            "joint_improves_over_separable_tuning": bool(joint_over_separable[1] > 0),
            "measured_motion_increases_rate": bool(float(rate_effect.ci_low) > 0),
            "measured_motion_increases_ssi": bool(float(ssi_effect.ci_low) > 0),
            "spectral_engagement_does_not_by_itself_explain_ssi": bool(ssi_spectral_r2[2] <= 0.05),
        },
    }
    if args.raw_shards:
        raw = load_and_merge_shards(args.raw_shards)
        raw_rate, _, raw_ssi, raw_results = evaluate_dataset(raw, args)
        summary["raw_signed_joint_median_r2"] = float(
            np.nanmedian(raw_results["signed joint rate drive"][1].cv_r2)
        )
        summary["raw_median_rate_delta_1x"] = float(np.nanmedian(raw_rate[:, :, np.argmin(np.abs(raw["motion_scales"][1:] - 1.0))]))
        summary["raw_median_ssi_delta_1x"] = float(np.nanmedian(raw_ssi[:, :, np.argmin(np.abs(raw["motion_scales"][1:] - 1.0))]))
    if args.rotated_shards:
        rotated = load_and_merge_shards(args.rotated_shards)
        rotated_rate, _, _, rotated_results = evaluate_dataset(rotated, args)
        summary["rotated_signed_joint_median_r2"] = float(
            np.nanmedian(rotated_results["signed joint rate drive"][1].cv_r2)
        )
        summary["rotated_median_rate_delta_1x"] = float(np.nanmedian(rotated_rate[:, :, np.argmin(np.abs(rotated["motion_scales"][1:] - 1.0))]))

    render_overview(
        data,
        rate_delta,
        ssi_delta,
        results,
        predictor_summary,
        effects,
        args.out_dir / "m77_retinal_causal_chain_overview.png",
        args.exemplar_unit,
    )
    render_per_unit_pages(
        args.out_dir / "m77_per_unit_projected_vs_actual.pdf",
        data=data,
        rate_delta=rate_delta,
        ssi_delta=ssi_delta,
        results=results,
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "m77_retinal_causal_chain_overview.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
