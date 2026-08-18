#!/usr/bin/env python3
"""Compose the production M77 retinal-motion causal-chain figure."""
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


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.run_m77_retinal_causal_chain import (
    DEFAULT_CHAIN,
    interpolate_tuning_temporal,
    load_tuning_tensors,
)


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--tuning-table", type=Path, default=DEFAULT_CHAIN / "dense_tuning/frequency_tuning_grouped.csv")
    parser.add_argument("--auxiliary-power", type=Path, default=DEFAULT_CHAIN / "auxiliary_power/auxiliary_retinal_power.npz")
    parser.add_argument("--nonlinear-audit", type=Path, default=DEFAULT_CHAIN / "nonlinear_audit/nonlinear_sharpening.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exemplar-unit", type=int, default=25)
    return parser.parse_args()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.08, label, transform=axis.transAxes, fontsize=13, fontweight="bold", va="bottom")


def contour_power(
    axis: plt.Axes,
    spatial: np.ndarray,
    temporal: np.ndarray,
    value: np.ndarray,
    *,
    floor: float,
    ceiling: float,
    title: str,
):
    contour = axis.contourf(
        spatial,
        temporal,
        np.log10(np.maximum(value.T, floor)),
        levels=np.linspace(np.log10(floor), np.log10(ceiling), 13),
        cmap="magma",
        extend="both",
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_title(title, fontsize=9.5)
    axis.set_xlabel("SF (cycles/deg)")
    return contour


def main() -> int:
    args = parse_args()
    with np.load(args.analysis_dir / "analysis_arrays.npz") as handle:
        data = {key: handle[key] for key in handle.files}
    tuning = load_tuning_tensors(args.tuning_table)
    unit_matches = np.flatnonzero(data["unit_indices"].astype(int) == args.exemplar_unit)
    if not len(unit_matches):
        raise ValueError(f"exemplar unit {args.exemplar_unit} is not active")
    unit = int(unit_matches[0])
    tuning_unit = int(np.flatnonzero(tuning["unit_indices"] == args.exemplar_unit)[0])
    orientation_score = tuning["phase_rms"][tuning_unit].sum(axis=(0, 1))
    orientation_index = int(np.argmax(orientation_score))

    if args.auxiliary_power.exists():
        with np.load(args.auxiliary_power) as handle:
            power = np.asarray(handle["power"], dtype=float)
            power_scales = np.asarray(handle["motion_scales"], dtype=float)
            power_spatial = np.asarray(handle["spatial_cpd"], dtype=float)
            power_temporal = np.asarray(handle["temporal_hz"], dtype=float)
    else:
        power = np.asarray(data["average_power"], dtype=float)
        power_scales = np.asarray(data["motion_scales"], dtype=float)
        power_spatial = np.asarray(data["spatial_cpd"], dtype=float)
        power_temporal = np.asarray(data["temporal_hz"], dtype=float)
    marginal_power = power.sum(axis=-1)
    total_power = marginal_power.sum(axis=(1, 2))
    normalized_power = marginal_power / np.maximum(total_power[:, None, None], EPS)
    positive = normalized_power[normalized_power > 0]
    floor = max(float(np.quantile(positive, 0.02)), EPS)
    ceiling = float(np.quantile(positive, 0.995))
    measured = int(np.argmin(np.abs(power_scales - 1.0)))

    interpolated_signed = interpolate_tuning_temporal(
        tuning["signed_rate_sensitivity"], tuning["temporal_hz"], power_temporal, normalize=False
    )[tuning_unit]
    interpolated_passband = interpolate_tuning_temporal(
        tuning["phase_rms"], tuning["temporal_hz"], power_temporal, normalize=True
    )[tuning_unit]
    contribution = power[measured] * interpolated_signed
    contribution_map = contribution.sum(axis=-1)
    contribution_limit = max(float(np.quantile(np.abs(contribution_map), 0.995)), EPS)

    figure = plt.figure(figsize=(17.2, 13.2), constrained_layout=True)
    outer = figure.add_gridspec(4, 4, height_ratios=(0.78, 1.0, 1.0, 0.95))

    trajectory_axis = figure.add_subplot(outer[0, 0])
    trace = np.asarray(data.get("example_trace_xy", np.empty((0,))), dtype=float)
    if trace.size:
        scored = trace[-240:]
        trajectory_axis.plot(scored[:, 0] * 60, scored[:, 1] * 60, color="#00A6C8", lw=1.4)
        trajectory_axis.scatter(scored[0, 0] * 60, scored[0, 1] * 60, s=24, color="black", zorder=3)
        trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set(xlabel="horizontal position (arcmin)", ylabel="vertical position (arcmin)", title="Real filtered fixation")
    panel_label(trajectory_axis, "A")

    movie_grid = outer[0, 1:].subgridspec(1, 5)
    frames = np.asarray(data.get("example_movie_frames", np.empty((0,))), dtype=float)
    for index in range(5):
        axis = figure.add_subplot(movie_grid[0, index])
        if len(frames) > index:
            axis.imshow(frames[index], cmap="gray", vmin=0, vmax=255)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{250 * index:g} ms", fontsize=9)
        if index == 0:
            panel_label(axis, "B")
            axis.set_ylabel("retinal movie", fontsize=9)

    power_grid = outer[1, :].subgridspec(1, len(power_scales))
    power_contour = None
    for index, scale in enumerate(power_scales):
        axis = figure.add_subplot(power_grid[0, index])
        if total_power[index] <= EPS:
            axis.set_facecolor("#0A061E")
            axis.text(0.5, 0.5, "no dynamic power", color="white", ha="center", va="center", transform=axis.transAxes)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log", base=2)
            axis.set_xlabel("SF (cycles/deg)")
        else:
            power_contour = contour_power(
                axis,
                power_spatial,
                power_temporal,
                normalized_power[index],
                floor=floor,
                ceiling=ceiling,
                title=f"{scale:g}× motion · power {total_power[index] / max(total_power[measured], EPS):.2f}×",
            )
        if index == 0:
            axis.set_ylabel("temporal frequency (Hz)")
            panel_label(axis, "C")
    if power_contour is not None:
        figure.colorbar(power_contour, ax=figure.axes[-len(power_scales) :], shrink=0.75, label="log10 fraction of dynamic power")

    tuning_axis = figure.add_subplot(outer[2, 0])
    surface = tuning["phase_rms"][tuning_unit, :, :, orientation_index].T
    surface /= max(float(surface.max()), EPS)
    tuning_contour = tuning_axis.contourf(
        tuning["spatial_cpd"], tuning["temporal_hz"], surface,
        levels=np.linspace(0, 1, 13), cmap="viridis",
    )
    tuning_axis.contour(tuning["spatial_cpd"], tuning["temporal_hz"], surface, levels=[0.5], colors="white", linewidths=1.2)
    tuning_axis.set_xscale("log", base=2)
    tuning_axis.set_yscale("log", base=2)
    tuning_axis.set(xlabel="SF (cycles/deg)", ylabel="TF (Hz)", title=f"u{args.exemplar_unit:03d} measured passband · {tuning['orientation_deg'][orientation_index]:g}°")
    figure.colorbar(tuning_contour, ax=tuning_axis, shrink=0.73, label="normalized phase RMS")
    panel_label(tuning_axis, "D")

    engagement_axis = figure.add_subplot(outer[2, 1])
    measured_map = normalized_power[measured]
    engagement_contour = contour_power(
        engagement_axis,
        power_spatial,
        power_temporal,
        measured_map,
        floor=floor,
        ceiling=ceiling,
        title="Measured retinal power + u025 half-max passband",
    )
    passband_map = interpolated_passband.sum(axis=-1).T
    if passband_map.max() > 0:
        engagement_axis.contour(power_spatial, power_temporal, passband_map, levels=[0.5 * passband_map.max()], colors="#52D1DC", linewidths=1.8)
    figure.colorbar(engagement_contour, ax=engagement_axis, shrink=0.73, label="log10 power fraction")
    panel_label(engagement_axis, "E")

    contribution_axis = figure.add_subplot(outer[2, 2])
    contribution_image = contribution_axis.contourf(
        power_spatial,
        power_temporal,
        contribution_map.T,
        levels=np.linspace(-contribution_limit, contribution_limit, 15),
        cmap="RdBu_r",
        extend="both",
    )
    contribution_axis.set_xscale("log", base=2)
    contribution_axis.set_yscale("log", base=2)
    contribution_axis.set(xlabel="SF (cycles/deg)", ylabel="TF (Hz)", title="Power × signed rate sensitivity")
    figure.colorbar(contribution_image, ax=contribution_axis, shrink=0.73, label="signed projected contribution")
    panel_label(contribution_axis, "F")

    scatter_axis = figure.add_subplot(outer[2, 3])
    observed = data["rate_delta"][..., unit].ravel()
    predicted = data["signed_joint_prediction"][..., unit].ravel()
    valid = np.isfinite(observed) & np.isfinite(predicted)
    scatter_axis.scatter(predicted[valid], observed[valid], s=6, alpha=0.15, color="#2878B5", rasterized=True)
    if np.any(valid):
        limits = (min(float(predicted[valid].min()), float(observed[valid].min())), max(float(predicted[valid].max()), float(observed[valid].max())))
        scatter_axis.plot(limits, limits, color="0.35", lw=1)
    scatter_axis.set(xlabel="held-out projected Δ rate", ylabel="exact M77 Δ rate", title=f"u{args.exemplar_unit:03d}: projection predicts replay")
    panel_label(scatter_axis, "G")

    bottom = outer[3, :].subgridspec(1, 3, width_ratios=(1.45, 1.1, 2.2))
    predictor_axis = figure.add_subplot(bottom[0, 0])
    predictor = pd.read_csv(args.analysis_dir / "predictor_summary.csv")
    predictor = predictor.loc[predictor.metric.eq("cv_r2")]
    x = np.arange(len(predictor))
    predictor_axis.errorbar(x, predictor["median"], yerr=np.vstack((predictor["median"] - predictor.ci_low, predictor.ci_high - predictor["median"])), fmt="o", capsize=3)
    predictor_axis.axhline(0, color="0.5", lw=0.8)
    predictor_axis.set_xticks(
        x,
        ("total", "TF", "SF×ori", "separable", "joint\npassband", "signed\njoint"),
        rotation=18,
        ha="right",
    )
    predictor_axis.set(ylabel="held-out R²", title="Joint tuning versus controls")
    panel_label(predictor_axis, "H")

    effect_axis = figure.add_subplot(bottom[0, 1])
    effects = pd.read_csv(args.analysis_dir / "motion_effects.csv")
    rate = effects.loc[effects.measure.eq("mean_rate_hz")].sort_values("motion_scale")
    ssi = effects.loc[effects.measure.eq("map_ssi_bits_per_spike")].sort_values("motion_scale")
    effect_axis.errorbar(rate.motion_scale, rate.median_change, yerr=np.vstack((rate.median_change - rate.ci_low, rate.ci_high - rate.median_change)), marker="o", color="#D95F30", capsize=3)
    effect_axis.set(
        xlabel="motion scale",
        ylabel="Δ rate (spikes/s)\n= Δ expected spikes in 1 s",
        title="Motion raises spike count and per-spike information",
    )
    twin = effect_axis.twinx()
    twin.errorbar(ssi.motion_scale, ssi.median_change, yerr=np.vstack((ssi.median_change - ssi.ci_low, ssi.ci_high - ssi.median_change)), marker="s", color="#6A51A3", capsize=3)
    twin.set_ylabel("Δ SSI (bits/spike)", color="#6A51A3")
    effect_axis.axhline(0, color="0.5", lw=0.8, ls="--")
    panel_label(effect_axis, "I")

    nonlinear_axis = figure.add_subplot(bottom[0, 2])
    if args.nonlinear_audit.exists():
        with np.load(args.nonlinear_audit) as handle:
            metric_names = list(handle["metric_names"])
            stage_names = handle["stage_names"]
            baseline = handle["baseline"]
            actual = handle["actual"]
            tangent = handle["tangent"]
        metric_index = metric_names.index("spatial_information")
        actual_effect = np.nanmedian(actual[..., metric_index] - baseline[..., metric_index], axis=(0, 1))
        tangent_effect = np.nanmedian(tangent[..., metric_index] - baseline[..., metric_index], axis=(0, 1))
        stage_x = np.arange(len(stage_names))
        nonlinear_axis.plot(stage_x, actual_effect, "o-", color="#C94C4C", lw=2, label="full replay")
        nonlinear_axis.plot(stage_x, tangent_effect, "o--", color="#3178B5", lw=1.8, label="local tangent")
        nonlinear_axis.set_xticks(stage_x, ["signed\nstem", "GN/LRN/\nsplit-ReLU", "spatial\n1", "spatial\n2", "spatial\n3", "RR100\noutput"])
        nonlinear_axis.legend(frameon=False)
    else:
        nonlinear_axis.text(0.5, 0.5, "nonlinear audit pending", ha="center", va="center", transform=nonlinear_axis.transAxes)
    nonlinear_axis.axhline(0, color="0.5", lw=0.8)
    nonlinear_axis.set(ylabel="motion − stabilized spatial information", title="Where nonlinear sharpening emerges")
    panel_label(nonlinear_axis, "J")

    engagement_axis.set_title(
        f"Measured retinal power + u{args.exemplar_unit:03d} half-max passband"
    )
    for axis in figure.axes:
        if hasattr(axis, "spines"):
            axis.spines[["top", "right"]].set_visible(False)
    summary_path = args.analysis_dir / "summary.json"
    claims = {}
    if summary_path.exists():
        claims = (json.loads(summary_path.read_text(encoding="utf-8")).get("claim_gates") or {})
    rate_chain_pass = all(
        claims.get(key, False)
        for key in (
            "joint_tuning_predicts_held_out_rate",
            "joint_improves_over_total_power",
            "joint_improves_over_separable_tuning",
            "measured_motion_increases_rate",
            "measured_motion_increases_ssi",
        )
    )
    nonlinear_claims = {}
    nonlinear_summary = args.nonlinear_audit.parent / "summary.json"
    if nonlinear_summary.exists():
        nonlinear_claims = (
            json.loads(nonlinear_summary.read_text(encoding="utf-8")).get("claim_gates")
            or {}
        )
    nonlinear_pass = all(
        nonlinear_claims.get(key, False)
        for key in (
            "full_replay_increases_rr100_spatial_information",
            "tangent_underestimates_rr100_spatial_sharpening",
        )
    )
    title = (
        "M77: retinal motion increases firing while nonlinear stages sharpen spatial coding"
        if rate_chain_pass and nonlinear_pass
        else "M77 retinal-motion causal chain · production claim audit"
    )
    figure.suptitle(title, fontsize=18, fontweight="semibold")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "m77_retinal_motion_causal_chain.png"
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
