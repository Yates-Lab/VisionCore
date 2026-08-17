#!/usr/bin/env python3
"""Render checkpoint-verified RR100 rate maps for stabilized and moving retinae.

The gallery reuses the exact image and centered drift trajectories from a
controlled-scaling run.  It selects a representative image/trajectory pair by
the median pooled SSI change at measured motion, selects units near fixed
positive-effect quantiles, and uses one shared causal timepoint for every row.
No displayed map is spatially smoothed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.audit_native_core_motion_path import (
    json_ready,
    require_matching_controlled_model,
    selected_causal_histories,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    PPD,
    RealTraceMatrixScorer,
    _standardize_uint_like,
    _trace_on_output_grid,
)
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_POPULATION_SPEC_DIR,
    RR100_VERSION,
)


EPS = 1e-10


def controlled_trace_on_model_grid(
    trace: np.ndarray,
    manifest: dict,
    *,
    archive_source_rate_hz: int | None,
    model_output_rate_hz: int,
    torch,
) -> np.ndarray:
    """Expand a retained controlled trace onto the model's output grid."""
    contract = dict(manifest.get("time_contract") or {})
    source_rate_hz = int(
        archive_source_rate_hz
        if archive_source_rate_hz is not None
        else contract.get("source_trace_rate_hz", 0)
    )
    declared_output_rate_hz = int(contract.get("model_output_rate_hz", 0))
    if source_rate_hz < 1 or declared_output_rate_hz < 1:
        raise ValueError("Controlled response is missing source/output trace rates")
    if declared_output_rate_hz != int(model_output_rate_hz):
        raise ValueError(
            "Activation-gallery model rate differs from the controlled response: "
            f"{model_output_rate_hz} versus {declared_output_rate_hz} Hz."
        )
    return _trace_on_output_grid(
        np.asarray(trace, dtype=np.float32),
        source_rate_hz=source_rate_hz,
        output_rate_hz=declared_output_rate_hz,
        torch=torch,
    )


def map_ssi(rate_map: np.ndarray) -> np.ndarray:
    """Return spatial single-spike information for [..., y, x] rate maps."""
    rate = np.clip(np.asarray(rate_map, dtype=np.float64), 0.0, None)
    mean = rate.mean(axis=(-2, -1))
    gain = rate / np.maximum(mean[..., None, None], EPS)
    return np.mean(gain * np.log2(np.maximum(gain, EPS)), axis=(-2, -1))


def pooled_pair_effect(ssi: np.ndarray, expected: np.ndarray, one_index: int) -> np.ndarray:
    """Expected-spike-weighted SSI percent change for every image/trace pair."""
    stable = np.sum(ssi[:, :, 0] * expected[:, :, 0], axis=-1) / np.maximum(
        np.sum(expected[:, :, 0], axis=-1), EPS
    )
    moving = np.sum(ssi[:, :, one_index] * expected[:, :, one_index], axis=-1) / np.maximum(
        np.sum(expected[:, :, one_index], axis=-1), EPS
    )
    return 100.0 * (moving - stable) / np.maximum(stable, EPS)


def representative_pair(effect: np.ndarray) -> tuple[int, int]:
    """Choose the finite pair closest to the population median effect."""
    value = np.asarray(effect, dtype=np.float64)
    finite = np.isfinite(value)
    if not np.any(finite):
        raise ValueError("No finite image/trajectory effects are available")
    target = float(np.median(value[finite]))
    distance = np.where(finite, np.abs(value - target), np.inf)
    return tuple(int(v) for v in np.unravel_index(np.argmin(distance), value.shape))


def representative_units(
    ssi: np.ndarray,
    expected: np.ndarray,
    image_row: int,
    trace_row: int,
    one_index: int,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Select unique units nearest fixed quantiles of positive SSI change."""
    stable = np.asarray(ssi[image_row, trace_row, 0], dtype=np.float64)
    moving = np.asarray(ssi[image_row, trace_row, one_index], dtype=np.float64)
    support = (
        np.isfinite(stable)
        & np.isfinite(moving)
        & (expected[image_row, trace_row, 0] > 0)
        & (expected[image_row, trace_row, one_index] > 0)
    )
    delta = moving - stable
    candidates = np.flatnonzero(support & (delta > 0))
    if len(candidates) < len(quantiles):
        candidates = np.flatnonzero(support)
    if not len(candidates):
        raise ValueError("No supported units are available for the activation gallery")
    ordered = candidates[np.argsort(delta[candidates], kind="mergesort")]
    positions = np.clip(
        np.round(np.asarray(quantiles, dtype=np.float64) * (len(ordered) - 1)).astype(int),
        0,
        len(ordered) - 1,
    )
    selected = []
    for position in positions:
        unit = int(ordered[position])
        if unit not in selected:
            selected.append(unit)
    if len(selected) < len(quantiles):
        for unit in ordered[::-1]:
            if int(unit) not in selected:
                selected.append(int(unit))
            if len(selected) == len(quantiles):
                break
    return np.asarray(selected, dtype=int)


def score_selected_maps(
    scorer: RealTraceMatrixScorer,
    image: np.ndarray,
    trace: np.ndarray,
    units: np.ndarray,
    *,
    frame_batch_size: int,
) -> np.ndarray:
    """Return [condition, time, selected unit, y, x] RR100 maps."""
    core = scorer.model.model.convnet
    endpoints = np.arange(len(trace), dtype=np.int64)
    condition_maps: list[np.ndarray] = []
    for scale in (0.0, 1.0):
        chunks = []
        for start in range(0, len(endpoints), int(frame_batch_size)):
            rows = endpoints[start : start + int(frame_batch_size)]
            stimulus = selected_causal_histories(
                image,
                trace * float(scale),
                rows,
                n_lags=int(core.temporal_support),
                out_size=tuple(scorer.out_size),
                ppd=PPD,
            )
            x = ((stimulus - 127.0) / 255.0).to(scorer.device)
            with scorer.torch.no_grad():
                full = scorer._compute_rate_map(x)
                rr = scorer.apply_population_view(full, scorer.population_view).clamp_min(0.0)
            chunks.append(rr[:, units].detach().cpu().numpy().astype(np.float32))
            del stimulus, x, full, rr
            if str(scorer.device).startswith("cuda"):
                scorer.torch.cuda.empty_cache()
        condition_maps.append(np.concatenate(chunks, axis=0))
    return np.stack(condition_maps, axis=0)


def shared_timepoint(maps: np.ndarray) -> int:
    """Choose the time nearest the upper quartile of pooled positive SSI change."""
    information = map_ssi(maps)
    delta = np.mean(information[1] - information[0], axis=1)
    positive = delta[np.isfinite(delta) & (delta > 0)]
    target = float(np.quantile(positive, 0.75)) if len(positive) else float(np.nanmedian(delta))
    return int(np.nanargmin(np.abs(delta - target)))


def render(
    maps: np.ndarray,
    units: np.ndarray,
    endpoint: int,
    output: Path,
    *,
    model_label: str,
    pair_effect: float,
) -> list[dict[str, float | int]]:
    n_units = len(units)
    figure, axes = plt.subplots(
        n_units,
        3,
        figsize=(7.2, 2.05 * n_units + 0.7),
        gridspec_kw={"wspace": 0.07, "hspace": 0.32},
        squeeze=False,
    )
    rows = []
    for row, unit in enumerate(units):
        stable = maps[0, endpoint, row]
        moving = maps[1, endpoint, row]
        delta = moving - stable
        lo, high = np.quantile(np.concatenate((stable.ravel(), moving.ravel())), [0.01, 0.99])
        limit = max(float(np.quantile(np.abs(delta), 0.995)), EPS)
        for column, (value, cmap, norm) in enumerate(
            (
                (stable, "viridis", None),
                (moving, "viridis", None),
                (delta, "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)),
            )
        ):
            axis = axes[row, column]
            axis.imshow(
                value,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=lo if norm is None else None,
                vmax=high if norm is None else None,
                norm=norm,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        stable_ssi = float(map_ssi(stable))
        moving_ssi = float(map_ssi(moving))
        pattern_correlation = float(np.corrcoef(stable.ravel(), moving.ravel())[0, 1])
        delta_alignment = float(np.corrcoef(stable.ravel(), delta.ravel())[0, 1])
        axes[row, 0].set_ylabel(f"RR100 u{int(unit):03d}", labelpad=7)
        axes[row, 1].text(
            0.5,
            -0.08,
            (
                f"map SSI {stable_ssi:.3f} → {moving_ssi:.3f}; "
                f"pattern r={pattern_correlation:.2f}"
            ),
            transform=axes[row, 1].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        rows.append({
            "unit_index": int(unit),
            "stabilized_map_ssi_bits_per_spike": stable_ssi,
            "moving_map_ssi_bits_per_spike": moving_ssi,
            "delta_map_ssi_bits_per_spike": moving_ssi - stable_ssi,
            "stable_vs_moving_spatial_correlation": pattern_correlation,
            "stable_map_vs_delta_spatial_correlation": delta_alignment,
            "mean_rate_gain": float(np.mean(moving) / max(float(np.mean(stable)), EPS)),
        })
    for axis, title in zip(axes[0], ("stabilized retina", "measured motion", "motion − stabilized")):
        axis.set_title(title, pad=5)
    figure.suptitle(
        f"{model_label}: exact causal RR100 rate maps; representative pair effect {pair_effect:+.1f}%",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.99,
        0.005,
        "one shared causal timepoint; raw maps; no display smoothing",
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return rows


def measured_unit_tuning_surface(
    grouped: pd.DataFrame,
    unit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orientation-envelope SF×TF response on the measured probe grid."""
    selected = grouped.loc[
        grouped.unit_index.eq(int(unit)) & grouped.temporal_hz.gt(0)
    ]
    if selected.empty:
        raise ValueError(f"No dynamic periodic-tuning rows for unit {unit}")
    table = (
        selected.groupby(["spatial_cpd", "temporal_hz"], sort=True)
        .response_amp_rms.max()
        .unstack()
    )
    surface = table.to_numpy(dtype=np.float64).T
    if not np.all(np.isfinite(surface)) or float(surface.max()) <= 0:
        raise ValueError(f"Invalid periodic-tuning surface for unit {unit}")
    return (
        table.index.to_numpy(dtype=np.float64),
        table.columns.to_numpy(dtype=np.float64),
        surface / float(surface.max()),
    )


def render_tuning_map_gallery(
    maps: np.ndarray,
    units: np.ndarray,
    endpoint: int,
    grouped_tuning_csv: Path,
    robust_tuning_summary: Path,
    output: Path,
    *,
    model_label: str,
    pair_effect: float,
) -> list[dict[str, float | int | bool]]:
    """Put each output unit's measured passband beside its causal rate maps."""
    grouped = pd.read_csv(grouped_tuning_csv)
    robust = pd.read_csv(robust_tuning_summary).set_index("unit_index")
    figure, axes = plt.subplots(
        len(units),
        4,
        figsize=(10.8, 2.55 * len(units) + 0.9),
        gridspec_kw={
            "width_ratios": (1.18, 1.0, 1.0, 1.0),
            "wspace": 0.16,
            "hspace": 0.42,
        },
        squeeze=False,
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.90)
    rows = []
    tuning_contour = None
    for row, unit in enumerate(units):
        sf, tf, tuning = measured_unit_tuning_surface(grouped, int(unit))
        tuning_contour = axes[row, 0].contourf(
            sf,
            tf,
            tuning,
            levels=np.linspace(0, 1, 11),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axes[row, 0].scatter(
            *np.meshgrid(sf, tf, indexing="xy"),
            s=3,
            color="white",
            alpha=0.45,
            linewidth=0,
        )
        axes[row, 0].set_xscale("log", base=2)
        axes[row, 0].set_yscale("log", base=2)
        axes[row, 0].set_xlabel(
            "SF (cycles/deg)" if row == len(units) - 1 else ""
        )
        axes[row, 0].set_ylabel("TF (Hz)")
        metadata = robust.loc[int(unit)]
        axes[row, 0].set_title(
            f"u{int(unit):03d}: {float(metadata.preferred_sf_cpd):.2g} c/deg, "
            f"{float(metadata.preferred_tf_hz):.1f} Hz",
            fontsize=8.5,
            pad=4,
        )

        stable = maps[0, endpoint, row]
        moving = maps[1, endpoint, row]
        delta = moving - stable
        lo, high = np.quantile(np.concatenate((stable.ravel(), moving.ravel())), [0.01, 0.99])
        limit = max(float(np.quantile(np.abs(delta), 0.995)), EPS)
        for column, (value, cmap, norm) in enumerate(
            (
                (stable, "viridis", None),
                (moving, "viridis", None),
                (delta, "RdBu_r", TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)),
            ),
            start=1,
        ):
            axis = axes[row, column]
            axis.imshow(
                value,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=lo if norm is None else None,
                vmax=high if norm is None else None,
                norm=norm,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        stable_ssi = float(map_ssi(stable))
        moving_ssi = float(map_ssi(moving))
        axes[row, 2].text(
            0.5,
            -0.08,
            f"map SSI {stable_ssi:.3f} → {moving_ssi:.3f}",
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
        )
        rows.append({
            "unit_index": int(unit),
            "preferred_sf_cpd": float(metadata.preferred_sf_cpd),
            "preferred_tf_hz": float(metadata.preferred_tf_hz),
            "low_sf_censored": bool(metadata.low_sf_censored),
            "fit_r2": float(metadata.fit_r2),
            "stabilized_map_ssi_bits_per_spike": stable_ssi,
            "moving_map_ssi_bits_per_spike": moving_ssi,
        })
    for axis, title in zip(
        axes[0, 1:],
        ("stabilized retina", "measured motion", "motion − stabilized"),
    ):
        axis.set_title(title, pad=5)
    figure.canvas.draw()
    tuning_box = axes[0, 0].get_position()
    figure.text(
        0.5 * (tuning_box.x0 + tuning_box.x1),
        0.925,
        "measured output-unit SF×TF tuning",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    if tuning_contour is not None:
        cbar_axis = figure.add_axes(
            [tuning_box.x0, 0.055, tuning_box.width, 0.016]
        )
        colorbar = figure.colorbar(
            tuning_contour,
            cax=cbar_axis,
            orientation="horizontal",
        )
        colorbar.set_label("normalized response amplitude", fontsize=7.5)
        colorbar.ax.tick_params(labelsize=7, length=2)
    figure.suptitle(
        f"{model_label}: measured passbands and exact motion-driven spatial maps "
        f"(representative pair {pair_effect:+.1f}% SSI)",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.99,
        0.006,
        "same causal timepoint in every row; raw maps; periodic tuning measured on a respaced grid",
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-configs", type=Path, required=True)
    parser.add_argument("--controlled-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, default=DEFAULT_POPULATION_SPEC_DIR)
    parser.add_argument("--rr100-version", default=RR100_VERSION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--model-label", default="native-240 twin")
    parser.add_argument("--periodic-tuning-csv", type=Path, default=None)
    parser.add_argument("--robust-tuning-summary", type=Path, default=None)
    args = parser.parse_args()

    controlled_dir = args.controlled_dir.resolve()
    manifest = require_matching_controlled_model(
        controlled_dir / "manifest.json",
        args.checkpoint.resolve(),
        args.dataset_configs.resolve(),
    )
    with np.load(controlled_dir / "controlled_scaling_response.npz") as archive:
        ssi = np.asarray(archive["ssi"], dtype=np.float64)
        expected = np.asarray(archive["expected_spikes"], dtype=np.float64)
        scales = np.asarray(archive["scale_factors"], dtype=np.float64)
        traces = np.asarray(archive["base_trace_xy"], dtype=np.float32)
        archive_source_rate_hz = (
            int(round(float(archive["base_trace_source_rate_hz"])))
            if "base_trace_source_rate_hz" in archive.files
            else None
        )
    one_index = int(np.flatnonzero(np.isclose(scales, 1.0))[0])
    pair_effect = pooled_pair_effect(ssi, expected, one_index)
    image_row, trace_row = representative_pair(pair_effect)
    unit_quantiles = np.asarray([0.5, 0.75, 0.9], dtype=np.float64)
    units = representative_units(
        ssi, expected, image_row, trace_row, one_index, unit_quantiles
    )

    images = pd.read_csv(controlled_dir / "selected_images.csv")
    patch, patch_meta = extract_patch(images.iloc[image_row], canvas_cache={}, patch_size_px=540)
    image = _standardize_uint_like(patch)
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_configs.resolve(),
        population_spec_dir=args.population_spec_dir.resolve(),
        rr100_version=str(args.rr100_version),
        device=str(args.device),
        strict=True,
    )
    model_trace = controlled_trace_on_model_grid(
        traces[trace_row],
        manifest,
        archive_source_rate_hz=archive_source_rate_hz,
        model_output_rate_hz=int(scorer.output_rate_hz),
        torch=scorer.torch,
    )
    maps = score_selected_maps(
        scorer,
        image,
        model_trace,
        units,
        frame_batch_size=int(args.frame_batch_size),
    )
    endpoint = shared_timepoint(maps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "native_motion_activation_gallery.npz",
        maps=maps,
        unit_indices=units,
        image_row=np.asarray(image_row),
        trace_row=np.asarray(trace_row),
        endpoint=np.asarray(endpoint),
        source_trace_xy=traces[trace_row],
        model_output_trace_xy=model_trace,
    )
    figure_path = args.out_dir / "native_motion_activation_gallery.png"
    unit_rows = render(
        maps,
        units,
        endpoint,
        figure_path,
        model_label=str(args.model_label),
        pair_effect=float(pair_effect[image_row, trace_row]),
    )
    tuning_map_rows = None
    tuning_map_figure = None
    if (args.periodic_tuning_csv is None) != (args.robust_tuning_summary is None):
        raise ValueError(
            "--periodic-tuning-csv and --robust-tuning-summary must be supplied together"
        )
    if args.periodic_tuning_csv is not None:
        tuning_map_figure = args.out_dir / "native_motion_tuning_map_gallery.png"
        tuning_map_rows = render_tuning_map_gallery(
            maps,
            units,
            endpoint,
            args.periodic_tuning_csv.resolve(),
            args.robust_tuning_summary.resolve(),
            tuning_map_figure,
            model_label=str(args.model_label),
            pair_effect=float(pair_effect[image_row, trace_row]),
        )
    report = {
        "analysis": "checkpoint-verified native retinal-motion RR100 activation gallery",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_configs": str(args.dataset_configs.resolve()),
        "controlled_manifest_analysis": manifest.get("analysis"),
        "selection": {
            "pair_rule": "finite image/trajectory pair nearest median pooled SSI percent change",
            "unit_rule": "nearest 50th, 75th, and 90th percentiles of positive per-unit SSI change",
            "time_rule": "shared endpoint nearest upper quartile of positive mean selected-unit map-SSI change",
            "image_row": int(image_row),
            "trace_row": int(trace_row),
            "pair_ssi_percent_change": float(pair_effect[image_row, trace_row]),
            "unit_indices": units.tolist(),
            "endpoint_native_index": int(endpoint),
            "endpoint_ms": 1000.0 * float(endpoint) / float(scorer.output_rate_hz),
            "source_trace_samples": int(traces.shape[1]),
            "model_output_trace_samples": int(len(model_trace)),
        },
        "patch_metadata": patch_meta,
        "unit_map_metrics": unit_rows,
        "tuning_map_metrics": tuning_map_rows,
        "tuning_map_figure": (
            str(tuning_map_figure.resolve()) if tuning_map_figure is not None else None
        ),
        "display": "raw rate maps with nearest-neighbor display; stable and moving share row color limits",
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "native_motion_activation_gallery_summary.json").write_text(
        json.dumps(json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_ready(report), indent=2))


if __name__ == "__main__":
    main()
