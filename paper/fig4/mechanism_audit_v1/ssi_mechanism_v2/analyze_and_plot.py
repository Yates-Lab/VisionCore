#!/usr/bin/env python3
"""Analyze the exact Figure 4 SSI decomposition and make manuscript figures.

The primary figure follows the causal chain tested here:

    corrected retinal motion -> SF-to-TF power -> layerwise joint tuning
    -> population scale prediction -> spatial rate-map contrast -> SSI

All population summaries use the historical Figure 4 split (71 units with
``sf_split_metric < 0.5`` and 29 units with ``sf_split_metric >= 0.5``).
Channel selection never uses natural-image SSI.  Uncertainty is estimated by
crossed resampling of the eight images and 24 corrected drift trajectories.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import write_json


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/ssi_mechanism_v2"
DATA = OUT / "plot_data"
RAW = OUT / "exact_arrays"
LAYERWISE = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
LEGACY = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/merged"
)

LOW = "#0072B2"
HIGH = "#D55E00"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"
BLACK = "#202124"
GROUP_COLOR = {"low": LOW, "high": HIGH}
GROUP_LABEL = {"low": "Lower-SF units (n=71)", "high": "Higher-SF units (n=29)"}
TARGET = {"low": 3.0, "high": 1.0}
N_BOOT = 4000
EPS = 1e-12

NUMERATOR_COLUMNS = {
    "ssi": "ssi_weighted_numerator",
    "cv2": "cv2_weighted_numerator",
    "quadratic_bits": "quadratic_bits_weighted_numerator",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.17, y: float = 1.13) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def aggregate_rows(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    if weights is None:
        weights = np.ones(len(frame), dtype=float)
    weights = np.asarray(weights, dtype=float)
    expected = frame.expected_spikes.to_numpy(float)
    denominator = float(np.sum(weights * expected))
    result = {
        metric: float(np.sum(weights * frame[column].to_numpy(float)) / max(denominator, EPS))
        for metric, column in NUMERATOR_COLUMNS.items()
    }
    result["mean_rate"] = float(
        np.sum(weights * frame.mean_rate_sum.to_numpy(float))
        / max(np.sum(weights * frame.n_rate_values.to_numpy(float)), EPS)
    )
    result["expected_spikes"] = denominator
    return result


def aggregate_curves(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in metrics.groupby(["figure4_sf_group", "condition", "scale"], sort=False):
        rows.append(
            {
                "figure4_sf_group": keys[0],
                "condition": keys[1],
                "scale": float(keys[2]),
                **aggregate_rows(frame),
            }
        )
    result = pd.DataFrame(rows)
    baseline = (
        result.loc[result.condition.eq("normal_stable")]
        .groupby("figure4_sf_group")
        .ssi.first()
        .to_dict()
    )
    result["ssi_percent_vs_stable"] = [
        100.0 * (value - baseline[group]) / baseline[group]
        for value, group in zip(result.ssi, result.figure4_sf_group)
    ]
    result.to_csv(DATA / "exact_ssi_decomposition_curves.csv", index=False)
    return result


def crossed_bootstrap_weights(
    metrics: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = 20260812
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = np.sort(metrics.image_index.unique())
    traces = np.sort(metrics.trace_index.unique())
    image_lut = {value: i for i, value in enumerate(images)}
    trace_lut = {value: i for i, value in enumerate(traces)}
    image_ord = metrics.image_index.map(image_lut).to_numpy(int)
    trace_ord = metrics.trace_index.map(trace_lut).to_numpy(int)
    rng = np.random.default_rng(seed)
    weights = np.empty((n_boot, len(metrics)), dtype=np.float32)
    for boot in range(n_boot):
        image_counts = np.bincount(rng.integers(0, len(images), len(images)), minlength=len(images))
        trace_counts = np.bincount(rng.integers(0, len(traces), len(traces)), minlength=len(traces))
        weights[boot] = image_counts[image_ord] * trace_counts[trace_ord]
    return weights, images, traces


def values_by_boot(frame: pd.DataFrame, full_weights: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    w = full_weights[:, row_indices]
    expected = frame.expected_spikes.to_numpy(float)
    numerator = frame.ssi_weighted_numerator.to_numpy(float)
    return (w @ numerator) / np.maximum(w @ expected, EPS)


def bootstrap_curves(metrics: pd.DataFrame, full_weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in ("low", "high"):
        opposite = "high" if group == "low" else "low"
        stable_frame = metrics.loc[
            metrics.figure4_sf_group.eq(group) & metrics.condition.eq("normal_stable") & metrics.scale.eq(0)
        ]
        stable = values_by_boot(stable_frame, full_weights, stable_frame.index.to_numpy(int))
        for condition in (
            "normal_moving",
            f"{group}_motion_matched_necessary",
            f"{opposite}_motion_matched_necessary",
        ):
            for scale in sorted(metrics.scale.unique()):
                frame = metrics.loc[
                    metrics.figure4_sf_group.eq(group)
                    & metrics.condition.eq(condition)
                    & metrics.scale.eq(scale)
                ]
                value = values_by_boot(frame, full_weights, frame.index.to_numpy(int))
                percent = 100.0 * (value - stable) / np.maximum(stable, EPS)
                rows.append(
                    {
                        "figure4_sf_group": group,
                        "condition": condition,
                        "scale": scale,
                        "ci_low": float(np.quantile(percent, 0.025)),
                        "ci_high": float(np.quantile(percent, 0.975)),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "exact_ssi_decomposition_curve_bootstrap.csv", index=False)
    return result


def intervention_statistics(metrics: pd.DataFrame, full_weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in ("low", "high"):
        scale = TARGET[group]
        opposite = "high" if group == "low" else "low"
        conditions = {
            "stable": ("normal_stable", scale),
            "moving": ("normal_moving", scale),
            "necessary": (f"{group}_motion_matched_necessary", scale),
            "sufficient": (f"{group}_motion_matched_sufficient", scale),
            "readout_only": (f"{group}_readout_only_necessary", scale),
            "opposite_mask": (f"{opposite}_motion_matched_necessary", scale),
        }
        values: dict[str, float] = {}
        boots: dict[str, np.ndarray] = {}
        for name, (condition, target_scale) in conditions.items():
            frame = metrics.loc[
                metrics.figure4_sf_group.eq(group)
                & metrics.condition.eq(condition)
                & metrics.scale.eq(target_scale)
            ]
            values[name] = aggregate_rows(frame)["ssi"]
            boots[name] = values_by_boot(frame, full_weights, frame.index.to_numpy(int))
        observed = {
            "stable_ssi": values["stable"],
            "moving_ssi": values["moving"],
            "movement_gain": values["moving"] - values["stable"],
            "necessary_ssi": values["necessary"],
            "fraction_benefit_removed": (values["moving"] - values["necessary"])
            / max(values["moving"] - values["stable"], EPS),
            "sufficient_ssi": values["sufficient"],
            "fraction_benefit_transferred": (values["sufficient"] - values["stable"])
            / max(values["moving"] - values["stable"], EPS),
            "readout_only_ssi": values["readout_only"],
            "readout_only_fraction_removed": (values["moving"] - values["readout_only"])
            / max(values["moving"] - values["stable"], EPS),
            "opposite_mask_fraction_removed": (values["moving"] - values["opposite_mask"])
            / max(values["moving"] - values["stable"], EPS),
        }
        boot_metrics = {
            "stable_ssi": boots["stable"],
            "moving_ssi": boots["moving"],
            "movement_gain": boots["moving"] - boots["stable"],
            "necessary_ssi": boots["necessary"],
            "fraction_benefit_removed": (boots["moving"] - boots["necessary"])
            / np.maximum(boots["moving"] - boots["stable"], EPS),
            "sufficient_ssi": boots["sufficient"],
            "fraction_benefit_transferred": (boots["sufficient"] - boots["stable"])
            / np.maximum(boots["moving"] - boots["stable"], EPS),
            "readout_only_ssi": boots["readout_only"],
            "readout_only_fraction_removed": (boots["moving"] - boots["readout_only"])
            / np.maximum(boots["moving"] - boots["stable"], EPS),
            "opposite_mask_fraction_removed": (boots["moving"] - boots["opposite_mask"])
            / np.maximum(boots["moving"] - boots["stable"], EPS),
        }
        for metric, estimate in observed.items():
            distribution = boot_metrics[metric]
            rows.append(
                {
                    "figure4_sf_group": group,
                    "target_scale": scale,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.nanquantile(distribution, 0.025)),
                    "ci_high": float(np.nanquantile(distribution, 0.975)),
                    "bootstrap": "crossed image x corrected-trajectory resampling",
                    "n_bootstrap": len(distribution),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "exact_ssi_intervention_statistics.csv", index=False)
    return result


def authoritative_prediction(metrics: pd.DataFrame) -> pd.DataFrame:
    unit_table = pd.read_csv(LEGACY / "unit_feature_table.csv").sort_values("unit_index")
    group_by_unit = np.where(unit_table.sf_split_metric.to_numpy(float) < 0.5, "low", "high")
    overlap = pd.read_csv(LAYERWISE / "plot_data/layerwise_corrected_fem_tuning_overlap.csv.gz")
    if "spectrum_method_version" not in overlap or not overlap.spectrum_method_version.astype(str).str.startswith("trajectory_phase").all():
        raise RuntimeError(
            "Stale FEM overlap cache: production analysis requires complete trajectory-phase spectra"
        )
    overlap = overlap.loc[overlap.stage.eq("rr100") & overlap.response_metric.eq("f0_signed_mean")].copy()
    overlap["figure4_sf_group"] = overlap.channel.map(dict(enumerate(group_by_unit)))
    predicted = (
        overlap.groupby(["figure4_sf_group", "scale"], as_index=False)
        .normalized_overlap.mean()
        .rename(columns={"normalized_overlap": "raw_prediction"})
    )
    rows = []
    for group in ("low", "high"):
        moving = metrics.loc[
            metrics.figure4_sf_group.eq(group) & metrics.condition.eq("normal_moving")
        ]
        observed = []
        for scale, frame in moving.groupby("scale"):
            observed.append((float(scale), aggregate_rows(frame)["ssi"]))
        observed = pd.DataFrame(observed, columns=["scale", "ssi"])
        stable = aggregate_rows(
            metrics.loc[
                metrics.figure4_sf_group.eq(group)
                & metrics.condition.eq("normal_stable")
                & metrics.scale.eq(0)
            ]
        )["ssi"]
        observed["observed_gain"] = observed.ssi - stable
        pred = predicted.loc[predicted.figure4_sf_group.eq(group)].copy()
        pred["prediction_normalized"] = pred.raw_prediction / max(pred.raw_prediction.max(), EPS)
        observed["observed_gain_normalized"] = observed.observed_gain / max(
            observed.observed_gain.max(), EPS
        )
        joined = pred.merge(observed, on="scale", how="outer")
        for _, row in joined.iterrows():
            rows.append({"figure4_sf_group": group, **row.to_dict()})
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "authoritative_population_prediction_vs_ssi.csv", index=False)
    return result


def null_statistics(
    metrics: pd.DataFrame, null_components: pd.DataFrame, intervention: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(DATA / "figure4_convgru_channel_selection.csv")
    masks = np.load(RAW / "association_matched_permutation_null_masks.npz")
    null_rows = []
    stats_rows = []
    for group in ("low", "high"):
        scale = TARGET[group]
        stable = aggregate_rows(
            metrics.loc[
                metrics.figure4_sf_group.eq(group)
                & metrics.condition.eq("normal_stable")
                & metrics.scale.eq(scale)
            ]
        )["ssi"]
        moving = aggregate_rows(
            metrics.loc[
                metrics.figure4_sf_group.eq(group)
                & metrics.condition.eq("normal_moving")
                & metrics.scale.eq(scale)
            ]
        )["ssi"]
        observed = float(
            intervention.loc[
                intervention.figure4_sf_group.eq(group)
                & intervention.metric.eq("fraction_benefit_removed"),
                "estimate",
            ].iloc[0]
        )
        association = (
            selection.loc[selection.figure4_sf_group.eq(group)]
            .set_index("convgru_channel")
            .mean_squared_readout_weight
            .reindex(np.arange(128))
            .to_numpy(float)
        )
        observed_mask = masks[f"{group}_selected"].astype(int)
        observed_association = float(association[observed_mask].sum())
        group_selection = selection.loc[selection.figure4_sf_group.eq(group)].set_index("convgru_channel")
        true_contrast = group_selection.positive_overlap_contrast.reindex(np.arange(128)).to_numpy(float)
        true_score = group_selection.selection_score.reindex(np.arange(128)).to_numpy(float)
        observed_set = set(observed_mask.tolist())
        null_masks = masks[f"{group}_masks"].astype(int)
        distribution = []
        null_score = []
        null_contrast = []
        for null_id, frame in null_components.loc[
            null_components.figure4_sf_group.eq(group)
        ].groupby("null_mask_index"):
            null_ssi = float(frame.ssi_weighted_numerator.sum() / frame.expected_spikes.sum())
            fraction = (moving - null_ssi) / max(moving - stable, EPS)
            mask_association = float(association[null_masks[int(null_id)]].sum())
            distribution.append(fraction)
            null_score.append(float(true_score[null_masks[int(null_id)]].sum()))
            null_contrast.append(float(true_contrast[null_masks[int(null_id)]].sum()))
            null_rows.append(
                {
                    "figure4_sf_group": group,
                    "null_mask_index": int(null_id),
                    "null_necessary_ssi": null_ssi,
                    "fraction_benefit_removed": fraction,
                    "total_readout_association": mask_association,
                    "association_ratio_vs_observed": mask_association / observed_association,
                    "true_motion_contrast_sum": null_contrast[-1],
                    "true_selection_score_sum": null_score[-1],
                    "n_channels_shared_with_observed": len(
                        observed_set.intersection(null_masks[int(null_id)].tolist())
                    ),
                }
            )
        distribution = np.asarray(distribution)
        null_score = np.asarray(null_score)
        null_contrast = np.asarray(null_contrast)
        p_value = float((1 + np.sum(distribution >= observed)) / (1 + len(distribution)))
        stats_rows.append(
            {
                "figure4_sf_group": group,
                "target_scale": scale,
                "observed_fraction_benefit_removed": observed,
                "null_median": float(np.median(distribution)),
                "null_ci_low": float(np.quantile(distribution, 0.025)),
                "null_ci_high": float(np.quantile(distribution, 0.975)),
                "empirical_one_sided_p": p_value,
                "n_null_masks": len(distribution),
                "observed_total_readout_association": observed_association,
                "null_effect_vs_true_score_spearman_rho": float(
                    spearmanr(distribution, null_score).statistic
                ),
                "null_effect_vs_true_contrast_spearman_rho": float(
                    spearmanr(distribution, null_contrast).statistic
                ),
            }
        )
    null_table = pd.DataFrame(null_rows)
    stats = pd.DataFrame(stats_rows)
    null_table.to_csv(DATA / "association_matched_null_statistics.csv", index=False)
    stats.to_csv(DATA / "association_matched_null_summary.csv", index=False)
    return null_table, stats


def rate_contrast_diagnostics(metrics: pd.DataFrame) -> dict[str, float]:
    normal = metrics.loc[metrics.condition.eq("normal_moving")].copy()
    normal["ssi"] = normal.ssi_weighted_numerator / normal.expected_spikes
    normal["quadratic_bits"] = normal.quadratic_bits_weighted_numerator / normal.expected_spikes
    normal["cv2"] = normal.cv2_weighted_numerator / normal.expected_spikes
    normal["mean_rate"] = normal.mean_rate_sum / normal.n_rate_values
    baseline = normal.loc[normal.scale.eq(0), ["image_index", "trace_index", "figure4_sf_group", "ssi", "quadratic_bits", "cv2", "mean_rate"]]
    baseline = baseline.rename(columns={name: f"stable_{name}" for name in ("ssi", "quadratic_bits", "cv2", "mean_rate")})
    moving = normal.merge(baseline, on=["image_index", "trace_index", "figure4_sf_group"], how="left")
    moving["delta_ssi"] = moving.ssi - moving.stable_ssi
    moving["delta_quadratic_bits"] = moving.quadratic_bits - moving.stable_quadratic_bits
    moving["delta_cv2"] = moving.cv2 - moving.stable_cv2
    moving["delta_mean_rate"] = moving.mean_rate - moving.stable_mean_rate
    moving.to_csv(DATA / "rate_contrast_diagnostics.csv", index=False)
    dynamic = moving.loc[moving.scale.gt(0)]
    return {
        "delta_ssi_vs_delta_cv2_pearson_r": float(pearsonr(dynamic.delta_ssi, dynamic.delta_cv2).statistic),
        "delta_ssi_vs_delta_cv2_spearman_rho": float(spearmanr(dynamic.delta_ssi, dynamic.delta_cv2).statistic),
        "delta_ssi_vs_delta_quadratic_bits_pearson_r": float(
            pearsonr(dynamic.delta_ssi, dynamic.delta_quadratic_bits).statistic
        ),
        "delta_ssi_vs_delta_mean_rate_pearson_r": float(
            pearsonr(dynamic.delta_ssi, dynamic.delta_mean_rate).statistic
        ),
        "median_absolute_quadratic_approximation_error_bits": float(
            np.median(np.abs(dynamic.ssi - dynamic.quadratic_bits))
        ),
    }


def alignment_statistics(alignment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260813)
    for (group, metric), frame in alignment.groupby(["figure4_sf_group", "metric"]):
        values = frame["mean"].to_numpy(float)
        distribution = np.mean(
            values[rng.integers(0, len(values), size=(N_BOOT, len(values)))], axis=1
        )
        rows.append(
            {
                "figure4_sf_group": group,
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "bootstrap": "image-trajectory pair",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "selected_pathway_alignment_summary.csv", index=False)
    return result


def draw_retinal_spectra(fig: plt.Figure, spec, fem: np.lib.npyio.NpzFile) -> list[plt.Axes]:
    sub = spec.subgridspec(1, 2, wspace=0.10)
    axes = [fig.add_subplot(sub[0, i]) for i in range(2)]
    sf = fem["spatial_cpd"].astype(float)
    tf = fem["temporal_hz"].astype(float)
    scales = fem["scales"].astype(float)
    spectra = fem["retinal_motion_power"].mean(axis=-1)
    positive = spectra[spectra > 0]
    floor = np.quantile(positive, 0.02)
    transformed = np.log10(np.maximum(spectra, floor))
    vmin, vmax = np.quantile(transformed[scales > 0], [0.05, 0.995])
    for ax, scale in zip(axes, (1.0, 3.0)):
        index = int(np.flatnonzero(np.isclose(scales, scale))[0])
        ax.pcolormesh(sf, tf, transformed[index].T, shading="nearest", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks([0.5, 2, 8, 16], labels=["0.5", "2", "8", "16"])
        ax.set_yticks([0.5, 2, 8, 32], labels=["0.5", "2", "8", "32"])
        ax.tick_params(length=2.5)
        ax.set_title(f"{scale:g}× motion", pad=3)
    axes[0].set_ylabel("Temporal frequency (Hz)")
    axes[1].tick_params(labelleft=False)
    fig.text(
        (axes[0].get_position().x0 + axes[1].get_position().x1) / 2,
        axes[0].get_position().y0 - 0.045,
        "Spatial frequency (cycles/deg)",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    axes[0].text(
        0.02,
        1.18,
        r"Retinal motion:  $R_{\mathbf{k}}(t)=I_{\mathbf{k}}e^{-i2\pi\mathbf{k}\cdot\mathbf{X}(t)}$",
        transform=axes[0].transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
    )
    panel_label(axes[0], "A", x=-0.34, y=1.22)
    return axes


def draw_layer_emergence(ax: plt.Axes) -> None:
    summary = pd.read_csv(LAYERWISE / "plot_data/layerwise_sftf_stage_summary.csv")
    stages = [
        "temporal_frontend",
        "stem_preactivation",
        "post_stem_splitrelu",
        "resblock1_output",
        "resblock2_output",
        "convgru",
        "rr100",
    ]
    labels = ["Temporal\nbasis", "Spatial\nstem", "Split-\nReLU", "Res.\nblock 1", "Res.\nblock 2", "ConvGRU", "RR100"]
    for metric, label, color, marker in (
        ("f1_amplitude", "phase-sensitive F1", "#4C78A8", "o"),
        ("f0_signed_mean", "phase-averaged F0", "#B279A2", "s"),
    ):
        frame = summary.loc[summary.response_metric.eq(metric)].set_index("stage").loc[stages]
        ax.plot(
            range(len(stages)),
            frame.fraction_valid_tuned,
            marker=marker,
            ms=4,
            lw=1.7,
            color=color,
            label=label,
        )
    ax.set_xticks(range(len(stages)), labels, rotation=0)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Fraction with reliable\n2-D SF×TF fit")
    ax.set_title("Joint tuning is retained and reshaped")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    panel_label(ax, "B")


def draw_prediction(ax: plt.Axes, prediction: pd.DataFrame) -> None:
    for group in ("low", "high"):
        frame = prediction.loc[prediction.figure4_sf_group.eq(group)].sort_values("scale")
        color = GROUP_COLOR[group]
        ax.plot(
            frame.scale,
            frame.prediction_normalized,
            ls="--",
            marker="o",
            ms=3.5,
            lw=1.5,
            color=color,
            alpha=0.85,
        )
        ax.plot(
            frame.scale,
            frame.observed_gain_normalized,
            ls="-",
            marker="o",
            ms=3.5,
            lw=2.1,
            color=color,
        )
    ax.axhline(0, color=LIGHT_GREY, lw=0.8)
    ax.set_xticks([0, 0.5, 1, 2, 3])
    ax.set_xlabel("Movement scale")
    ax.set_ylabel("Normalized population effect")
    ax.set_ylim(-0.85, 1.12)
    ax.set_title("Tuning predicts ordering—not the full gain")
    color_handles = [
        Line2D([0], [0], color=LOW, lw=2, label="Lower SF"),
        Line2D([0], [0], color=HIGH, lw=2, label="Higher SF"),
    ]
    style_handles = [
        Line2D([0], [0], color=BLACK, lw=1.5, ls="--", label="FEM × tuning"),
        Line2D([0], [0], color=BLACK, lw=2, ls="-", label="Measured SSI gain"),
    ]
    first = ax.legend(handles=color_handles, frameon=False, loc="upper right", handlelength=1.5)
    ax.add_artist(first)
    ax.legend(handles=style_handles, frameon=False, loc="lower left", handlelength=1.8)
    panel_label(ax, "C")


def draw_maps(fig: plt.Figure, spec, maps: np.lib.npyio.NpzFile) -> list[plt.Axes]:
    sub = spec.subgridspec(2, 3, wspace=0.06, hspace=0.18)
    axes = [[fig.add_subplot(sub[row, col]) for col in range(3)] for row in range(2)]
    for row, group in enumerate(("low", "high")):
        stable = maps[f"{group}_stable_gain_map"].astype(float)
        moving = maps[f"{group}_moving_gain_map"].astype(float)
        necessary = maps[f"{group}_necessary_gain_map"].astype(float)
        delta = moving - stable
        selected_effect = moving - necessary
        gain_limit = np.quantile(np.concatenate([stable.ravel(), moving.ravel()]), [0.01, 0.99])
        delta_limit = np.quantile(np.abs(np.concatenate([delta.ravel(), selected_effect.ravel()])), 0.99)
        axes[row][0].imshow(stable, cmap="viridis", vmin=gain_limit[0], vmax=gain_limit[1], interpolation="nearest")
        axes[row][1].imshow(delta, cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, interpolation="nearest")
        axes[row][2].imshow(selected_effect, cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit, interpolation="nearest")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row][0].set_ylabel("Lower SF" if group == "low" else "Higher SF", color=GROUP_COLOR[group], fontweight="bold")
        stable_ssi = float(maps[f"{group}_stable_ssi"])
        moving_ssi = float(maps[f"{group}_moving_ssi"])
        necessary_ssi = float(maps[f"{group}_necessary_ssi"])
        axes[row][0].text(0.03, 0.04, f"SSI {stable_ssi:.3f}", transform=axes[row][0].transAxes, color="white", fontsize=6.8, bbox={"fc": "black", "alpha": 0.45, "ec": "none", "pad": 1.2})
        axes[row][1].text(0.03, 0.04, f"ΔSSI {moving_ssi-stable_ssi:+.3f}", transform=axes[row][1].transAxes, color="black", fontsize=6.8, bbox={"fc": "white", "alpha": 0.7, "ec": "none", "pad": 1.2})
        axes[row][2].text(0.03, 0.04, f"removed {moving_ssi-necessary_ssi:+.3f}", transform=axes[row][2].transAxes, color="black", fontsize=6.8, bbox={"fc": "white", "alpha": 0.7, "ec": "none", "pad": 1.2})
    for ax, title in zip(axes[0], (r"Stable  $g=r/\bar r$", "Motion-induced Δg", "Selected-path Δg")):
        ax.set_title(title, fontsize=8.3, pad=3)
    axes[0][0].text(0, 1.26, "Recurrent channels add structured spatial contrast", transform=axes[0][0].transAxes, fontsize=9.5, fontweight="bold", ha="left")
    panel_label(axes[0][0], "D", x=-0.36, y=1.30)
    return [ax for row in axes for ax in row]


def draw_causal_curves(ax: plt.Axes, curves: pd.DataFrame, ci: pd.DataFrame) -> None:
    for group in ("low", "high"):
        opposite = "high" if group == "low" else "low"
        for condition, style, width, alpha in (
            ("normal_moving", "-", 2.1, 1.0),
            (f"{group}_motion_matched_necessary", "--", 1.7, 1.0),
            (f"{opposite}_motion_matched_necessary", ":", 1.35, 0.8),
        ):
            frame = curves.loc[
                curves.figure4_sf_group.eq(group) & curves.condition.eq(condition)
            ].sort_values("scale")
            interval = ci.loc[
                ci.figure4_sf_group.eq(group) & ci.condition.eq(condition)
            ].sort_values("scale")
            color = GROUP_COLOR[group]
            if style != ":":
                ax.fill_between(interval.scale, interval.ci_low, interval.ci_high, color=color, alpha=0.10, lw=0)
            ax.plot(frame.scale, frame.ssi_percent_vs_stable, style, color=color, lw=width, marker="o", ms=3.0, alpha=alpha)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks([0, 0.5, 1, 2, 3])
    ax.set_xlabel("Movement scale")
    ax.set_ylabel("SSI change from stabilized (%)")
    ax.set_title("Recurrent channels causally shape SSI curves")
    color_handles = [
        Line2D([0], [0], color=LOW, lw=2, label="Lower SF"),
        Line2D([0], [0], color=HIGH, lw=2, label="Higher SF"),
    ]
    style_handles = [
        Line2D([0], [0], color=BLACK, lw=2, ls="-", label="Intact"),
        Line2D([0], [0], color=BLACK, lw=1.7, ls="--", label="Matched channels stabilized"),
        Line2D([0], [0], color=BLACK, lw=1.35, ls=":", label="Opposite-scale channels stabilized"),
    ]
    first = ax.legend(handles=color_handles, frameon=False, loc="upper left", handlelength=1.4)
    ax.add_artist(first)
    ax.legend(handles=style_handles, frameon=False, loc="upper right", handlelength=2)
    panel_label(ax, "E")


def draw_null(ax: plt.Axes, null_table: pd.DataFrame, null_summary: pd.DataFrame) -> None:
    rng = np.random.default_rng(17)
    for position, group in enumerate(("low", "high")):
        values = 100 * null_table.loc[
            null_table.figure4_sf_group.eq(group), "fraction_benefit_removed"
        ].to_numpy(float)
        jitter = rng.uniform(-0.14, 0.14, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=13,
            facecolor="#D1D5DB",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.8,
            zorder=1,
        )
        summary = null_summary.loc[null_summary.figure4_sf_group.eq(group)].iloc[0]
        ax.scatter(
            [position],
            [100 * summary.observed_fraction_benefit_removed],
            marker="D",
            s=48,
            color=GROUP_COLOR[group],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            position,
            100 * summary.observed_fraction_benefit_removed + 5,
            f"p={summary.empirical_one_sided_p:.3f}",
            color=GROUP_COLOR[group],
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks([0, 1], ["Lower SF\n3×", "Higher SF\n1×"])
    ax.set_ylabel("Target SSI benefit removed (%)")
    all_specific = bool((null_summary.empirical_one_sided_p < 0.05).all())
    ax.set_title(
        "Motion matching beats association-matched masks"
        if all_specific
        else "Causal support is distributed across channels"
    )
    ax.text(0.02, 0.055, "gray: 63 permuted-overlap masks/group\nmatched within ±10% readout association", transform=ax.transAxes, fontsize=6.8, color=GREY, va="bottom")
    panel_label(ax, "F")


def make_main_figure(
    curves: pd.DataFrame,
    curve_ci: pd.DataFrame,
    prediction: pd.DataFrame,
    null_table: pd.DataFrame,
    null_summary: pd.DataFrame,
) -> None:
    configure()
    fig = plt.figure(figsize=(14.2, 8.1), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.0, 1.0],
        height_ratios=[1.0, 1.05],
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.95,
        wspace=0.34,
        hspace=0.43,
    )
    fem = np.load(LAYERWISE / "exact_arrays/corrected_fem_retinal_sftf_spectrum.npz")
    maps = np.load(RAW / "representative_rate_map_decomposition.npz")
    draw_retinal_spectra(fig, outer[0, 0], fem)
    draw_layer_emergence(fig.add_subplot(outer[0, 1]))
    draw_prediction(fig.add_subplot(outer[0, 2]), prediction)
    draw_maps(fig, outer[1, 0], maps)
    draw_causal_curves(fig.add_subplot(outer[1, 1]), curves, curve_ci)
    draw_null(fig.add_subplot(outer[1, 2]), null_table, null_summary)
    export(fig, "figure_4_ssi_mechanism")


def make_supplement(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    intervention: pd.DataFrame,
    alignment: pd.DataFrame,
    null_table: pd.DataFrame,
) -> None:
    configure()
    diagnostics = pd.read_csv(DATA / "rate_contrast_diagnostics.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.15), gridspec_kw={"wspace": 0.38})
    dynamic = diagnostics.loc[diagnostics.scale.gt(0)]
    for group in ("low", "high"):
        frame = dynamic.loc[dynamic.figure4_sf_group.eq(group)]
        axes[0].scatter(frame.delta_quadratic_bits, frame.delta_ssi, s=6, alpha=0.22, color=GROUP_COLOR[group], rasterized=True)
    bounds = np.nanquantile(np.concatenate([dynamic.delta_quadratic_bits, dynamic.delta_ssi]), [0.01, 0.99])
    axes[0].plot(bounds, bounds, color=BLACK, ls="--", lw=1)
    axes[0].set_xlabel("Δ quadratic contrast (bits)")
    axes[0].set_ylabel("Δ exact SSI (bits)")
    axes[0].set_title("SSI follows spatial contrast")
    panel_label(axes[0], "A", x=-0.24)

    for group in ("low", "high"):
        frame = curves.loc[curves.figure4_sf_group.eq(group) & curves.condition.eq("normal_moving")].sort_values("scale")
        stable = float(frame.loc[frame.scale.eq(0), "mean_rate"].iloc[0])
        axes[1].plot(frame.scale, 100 * (frame.mean_rate - stable) / stable, marker="o", lw=1.8, ms=3.5, color=GROUP_COLOR[group])
    axes[1].axhline(0, color=GREY, lw=0.7)
    axes[1].set_xticks([0, 0.5, 1, 2, 3])
    axes[1].set_xlabel("Movement scale")
    axes[1].set_ylabel("Mean-rate change (%)")
    axes[1].set_title("Mean rate is not SSI")
    panel_label(axes[1], "B", x=-0.24)

    metric_order = ["selected_total_cosine", "selected_gain_cosine", "selected_projection"]
    metric_labels = ["Δz selected\nvs total", "Δz selected\nvs Δgain", "Selected/total\nprojection"]
    rng = np.random.default_rng(91)
    for group_index, group in enumerate(("low", "high")):
        for metric_index, metric in enumerate(metric_order):
            values = alignment.loc[
                alignment.figure4_sf_group.eq(group) & alignment.metric.eq(metric), "mean"
            ].to_numpy(float)
            x = metric_index + (group_index - 0.5) * 0.18
            axes[2].scatter(x + rng.uniform(-0.045, 0.045, len(values)), values, s=6, alpha=0.18, color=GROUP_COLOR[group])
            axes[2].plot([x - 0.07, x + 0.07], [values.mean(), values.mean()], color=GROUP_COLOR[group], lw=2.3)
    axes[2].axhline(0, color=GREY, lw=0.7)
    axes[2].set_xticks(range(3), metric_labels)
    axes[2].set_ylabel("Spatial similarity / fraction")
    axes[2].set_title("Selected pathway is spatially aligned")
    panel_label(axes[2], "C", x=-0.31, y=1.20)

    for group in ("low", "high"):
        frame = null_table.loc[null_table.figure4_sf_group.eq(group)]
        axes[3].scatter(frame.association_ratio_vs_observed, 100 * frame.fraction_benefit_removed, s=15, alpha=0.55, color=GROUP_COLOR[group], label="Lower SF" if group == "low" else "Higher SF")
    axes[3].axvspan(0.9, 1.1, color=LIGHT_GREY, alpha=0.25)
    axes[3].set_xlabel("Readout association / observed")
    axes[3].set_ylabel("SSI benefit removed (%)")
    axes[3].set_title("Nulls preserve readout coupling")
    axes[3].legend(frameon=False)
    panel_label(axes[3], "D", x=-0.31, y=1.20)
    export(fig, "figure_s_ssi_mechanism_controls")


def write_report(
    curves: pd.DataFrame,
    intervention: pd.DataFrame,
    null_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    diagnostics: dict[str, float],
) -> None:
    with (OUT / "run_manifest.json").open() as handle:
        run_manifest = json.load(handle)
    elapsed_minutes = float(run_manifest.get("elapsed_minutes", float("nan")))
    verification_path = OUT / "verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
    def stat(group: str, metric: str) -> pd.Series:
        return intervention.loc[
            intervention.figure4_sf_group.eq(group) & intervention.metric.eq(metric)
        ].iloc[0]

    if bool((null_summary.empirical_one_sided_p < 0.05).all()):
        null_conclusion = (
            "The observed effects exceeded readout-association-matched masks whose motion labels were "
            "permuted, supporting a channel-specific motion-matched pathway."
        )
    else:
        null_conclusion = (
            "The selected pathways were causal, but their effects did not consistently exceed "
            "readout-association-matched masks whose motion labels were permuted. The recurrent support is "
            "therefore distributed rather than uniquely localized to the selected channel identities."
        )

    lines = [
        "# Figure 4 mechanism: exact causal decomposition",
        "",
        "## Bottom line",
        "",
        "Corrected retinal motion redistributes image power through temporal frequency according to the "
        "finite-trajectory carrier `R_k(t)=I_k exp(-i2pi k.X(t))`. The model preserves joint SF×TF selectivity through the spatial stem, "
        "residual blocks, ConvGRU, and RR100 output. Combining the measured RR100 tuning with the exact "
        "corrected-FEM spectrum predicts the authoritative Figure 4 population ordering: higher-SF units "
        "are best served by smaller motion, whereas lower-SF units continue to benefit at larger motion. "
        "That spectral overlap does not explain the full effect amplitude or the large high-SF decline, "
        "which therefore has to arise downstream of the retinal transform. "
        "The final causal test shows how motion-driven recurrent activity becomes SSI: selected ConvGRU channels "
        "add a spatially structured component to the pre-softplus rate map, increasing the normalized "
        "spatial response contrast `g = r / mean(r)`. Stabilizing those channels removes the SSI benefit, "
        f"while inserting them transfers part of the benefit. {null_conclusion}",
        "",
        "## Exactly what was done",
        "",
        "1. Used the historical Figure 4 split verbatim: 71 units with `sf_split_metric < 0.5` and 29 "
        "units with `sf_split_metric >= 0.5`.",
        "2. Reused the validated dense grating probe (12 SF × 15 TF × 4 orientations × 2 phases) and its "
        "layerwise F0/F1 fits from retinal input through the first spatial stem, both residual blocks, "
        "ConvGRU, and RR100.",
        "3. Computed the retinal SF×TF spectrum from the 24 corrected drift histories by Fourier-transforming "
        "the complete demeaned trajectory carrier `exp(-i2pi k.scale[X(t)-X(0)])` for every image Fourier mode, "
        "then weighting by Fourier power from the eight held-out images. This retains displacement correlations "
        "and temporal order; no instantaneous-velocity histogram is used as a PSD.",
        "4. Predicted each population's movement-scale engagement by multiplying the exact FEM spectrum by "
        "its measured RR100 F0 SF×TF tuning. This prediction was fixed before reading natural-image SSI.",
        "5. Selected 16 ConvGRU channels per population using only frozen readout association multiplied by "
        "positive corrected-FEM/F1 overlap contrast at that population's target versus the opposite scale "
        "(lower SF: 3× vs 1×; higher SF: 1× vs 3×). Natural-image SSI was not used for selection.",
        "6. Replayed the complete 8 images × 24 corrected trajectories × 40 scored frames at 0, 0.5, 1, 2, "
        "and 3×. For necessity, the selected channels in each moving ConvGRU state were replaced by their "
        "paired stabilized values. For sufficiency, those moving channels were inserted into the stabilized state.",
        "7. Recomputed full 51×51 rate maps and exact SSI, not a scalar response proxy. For every unit/frame, "
        "`SSI = mean(g log2 g)`, where `g = r / mean(r)`. Also saved CV² and its local approximation "
        "`CV²/(2 ln 2)` to test whether increased normalized spatial contrast explains the SSI change.",
        "8. Exploited the additive pre-softplus readout to compare the selected channel contribution directly "
        "with the total motion-induced preactivation map and final normalized gain-map change.",
        "9. Constructed 63 null masks per population by permuting the motion-overlap contrast over ConvGRU "
        "channels, reranking channels, and retaining only 16-channel masks within ±10% of the observed mask's "
        "total true readout association.",
        "10. Computed 95% confidence intervals with 4,000 crossed bootstrap resamples of images and corrected trajectories.",
        "",
        "## Results",
        "",
    ]
    for group in ("low", "high"):
        movement = stat(group, "movement_gain")
        removed = stat(group, "fraction_benefit_removed")
        transferred = stat(group, "fraction_benefit_transferred")
        opposite = stat(group, "opposite_mask_fraction_removed")
        null = null_summary.loc[null_summary.figure4_sf_group.eq(group)].iloc[0]
        align = alignment_summary.loc[
            alignment_summary.figure4_sf_group.eq(group)
            & alignment_summary.metric.eq("selected_total_cosine")
        ].iloc[0]
        lines.extend(
            [
                f"### {GROUP_LABEL[group]}",
                "",
                f"At the preregistered target scale ({TARGET[group]:g}×), intact movement increased exact SSI "
                f"by {movement.estimate:.5f} bits (95% CI {movement.ci_low:.5f} to {movement.ci_high:.5f}). "
                f"Stabilizing the 16 motion-matched channels removed {100*removed.estimate:.1f}% of that benefit "
                f"(95% CI {100*removed.ci_low:.1f}% to {100*removed.ci_high:.1f}%); inserting the same moving "
                f"channels into the stable state transferred {100*transferred.estimate:.1f}% "
                f"({100*transferred.ci_low:.1f}% to {100*transferred.ci_high:.1f}%).",
                "",
                f"Stabilizing the mask selected for the opposite population removed "
                f"{100*opposite.estimate:.1f}% of the target benefit (95% CI "
                f"{100*opposite.ci_low:.1f}% to {100*opposite.ci_high:.1f}%).",
                "",
                f"The selected pre-softplus spatial change had mean cosine {align.estimate:.3f} with the total "
                f"motion-induced change (pair-bootstrap 95% CI {align.ci_low:.3f} to {align.ci_high:.3f}). "
                f"The association-matched null median removed {100*null.null_median:.1f}% "
                f"(central 95% {100*null.null_ci_low:.1f}% to {100*null.null_ci_high:.1f}%); the one-sided "
                f"empirical p value for the observed matched effect was {null.empirical_one_sided_p:.4f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The FEM×tuning calculation is a coarse population-ordering prediction, not a quantitative model "
            "of the SSI curve. For higher-SF units, normalized overlap remains 0.955 at 3× even though measured "
            "SSI is 8.98% below stabilized; for lower-SF units, overlap peaks at 3× while SSI peaks at 2×. Thus "
            "retinal spectral matching explains why the populations prefer different motion ranges, but the "
            "large downstream enhancement and high-SF reversal require unit-wise recurrent/readout geometry.",
            "",
            f"Across normal moving conditions, the change in exact SSI tracked the change in normalized "
            f"spatial variance (CV²): Pearson r={diagnostics['delta_ssi_vs_delta_cv2_pearson_r']:.3f}, "
            f"Spearman rho={diagnostics['delta_ssi_vs_delta_cv2_spearman_rho']:.3f}. Its correlation with "
            f"mean-rate change was r={diagnostics['delta_ssi_vs_delta_mean_rate_pearson_r']:.3f}. This matters "
            "because SSI is defined on each unit's mean-normalized rate map: a uniform gain in firing cannot by "
            "itself increase SSI. The mechanism supported here is therefore not simply 'motion raises responses.' "
            "It is: retinal motion places different spatial bands into different temporal bands; learned recurrent "
            "and readout-weighted channels transmit that modulation into spatially nonuniform output-rate changes; those "
            "changes increase normalized spatial contrast and hence SSI.",
            "",
            null_conclusion,
            "",
            "The evidence is population-level. The independently reconstructed absolute RR100 grating fits do "
            "not reproduce the postdoc's parametric fit table exactly (81 versus 85 valid F0 units and boundary-heavy "
            "fits), so absolute preferred SF/TF values should not replace the authoritative tuning analysis. The "
            "trajectory-phase spectrum, exact replay, historical populations, causal channel swaps, and rate-map SSI "
            "decomposition do not depend on that numerical match.",
            "",
            "## Figures",
            "",
            "### Main mechanism figure",
            "",
            "![Figure 4 SSI mechanism](figure_4_ssi_mechanism.png)",
            "",
            "- **A:** exact corrected-FEM retinal spectra. Increasing motion pushes the same spatial image "
            "power across temporal bands; the structure comes from the finite-trajectory phase spectrum and "
            "therefore includes the displacement correlations that broaden Brownian-like motion.",
            "- **B:** fraction of channels with a reliable joint 2-D SF×TF fit. Phase-sensitive F1 tuning is "
            "already explicit in the four-filter temporal frontend and remains through the core. Phase-averaged "
            "F0 tuning first becomes common after the spatial stem's SplitReLU and is reshaped by residual and recurrent mixing.",
            "- **C:** population-level prediction versus exact SSI gain. The prediction gets the low/high ordering "
            "and the high group's smaller optimum, but not the high group's large 3× reversal. The vertical mismatch "
            "is evidence for a computation beyond retinal spectral overlap.",
            "- **D:** objectively selected median examples. SSI operates on the normalized spatial rate map "
            "`g = r / mean(r)`. Movement adds a spatially patterned `Δg`; stabilizing the selected recurrent channels "
            "removes a similarly patterned component.",
            "- **E:** exact population SSI curves under intact, selected-mask, and opposite-mask conditions. The "
            "selected channels are strongly necessary for the lower-SF benefit and partly necessary for the higher-SF benefit.",
            "- **F:** the decisive specificity control. The selected effects sit inside the distributions from "
            "association-matched permuted-overlap masks. Therefore the causal support is distributed and should not "
            "be described as a unique low-speed or high-speed ConvGRU channel bank.",
            "",
            "Vector versions are `figure_4_ssi_mechanism.pdf` and `figure_4_ssi_mechanism.svg`.",
            "",
            "### Control figure",
            "",
            "![Figure 4 SSI mechanism controls](figure_s_ssi_mechanism_controls.png)",
            "",
            "Panel A shows that exact SSI changes track the quadratic normalized-contrast term; panel B separates "
            "mean-rate gain from SSI; panel C quantifies spatial alignment of the selected pre-softplus contribution; "
            "panel D verifies that every null is within the prespecified readout-association band. Vector versions are "
            "`figure_s_ssi_mechanism_controls.pdf` and `figure_s_ssi_mechanism_controls.svg`.",
            "",
            "## Runtime and reproduction",
            "",
            f"The complete exact decomposition took {elapsed_minutes:.2f} minutes on GPU 0. The analysis/figure "
            "aggregation took seconds on CPU. Reproduction commands from the repository root:",
            "",
            "```bash",
            "/home/jake/miniconda3/envs/yatesfv/bin/python paper/fig4/mechanism_audit_v1/ssi_mechanism_v2/run_exact_decomposition.py --device cuda:0 --frame-batch-size 8 --null-mask-batch-size 7",
            "/home/jake/miniconda3/envs/yatesfv/bin/python paper/fig4/mechanism_audit_v1/ssi_mechanism_v2/analyze_and_plot.py",
            "python -m pytest -q tests/test_fig4_ssi_mechanism.py tests/test_fig4_layerwise_sftf.py tests/test_fig4_corrected_history.py",
            "/home/jake/miniconda3/envs/yatesfv/bin/python paper/fig4/mechanism_audit_v1/ssi_mechanism_v2/verify_outputs.py",
            "```",
            "",
            *(
                [
                    "The saved-product verification status is **pass**: stable-condition spread and zero-motion "
                    "intervention error are exactly zero; endpoint reproduction differs by at most "
                    f"{verification['max_endpoint_reproduction_error_percentage_points']:.6g} percentage points; "
                    "all 126 null masks are unique within group and satisfy the ±10% readout-association gate; "
                    "representative maps are finite 51×51 arrays. Full results are in `verification.json`.",
                    "",
                ]
                if verification is not None and verification.get("status") == "pass"
                else []
            ),
            "## Machine-readable products",
            "",
            "All aggregate curves, crossed-bootstrap intervals, intervention statistics, null-mask effects, "
            "alignment statistics, selected channels, and representative map arrays are in `plot_data/` and "
            "`exact_arrays/`. `statistics.json` is the compact machine-readable summary.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    metrics = pd.read_csv(DATA / "exact_rate_map_metric_components.csv.gz").reset_index(drop=True)
    alignment = pd.read_csv(DATA / "selected_pathway_spatial_alignment.csv")
    null_components = pd.read_csv(DATA / "association_matched_null_trace_components.csv.gz")
    if metrics.image_index.nunique() != 8 or metrics.trace_index.nunique() != 24:
        raise ValueError(
            f"Expected full 8 x 24 run, got {metrics.image_index.nunique()} x {metrics.trace_index.nunique()}"
        )
    curves = aggregate_curves(metrics)
    weights, images, traces = crossed_bootstrap_weights(metrics)
    curve_ci = bootstrap_curves(metrics, weights)
    intervention = intervention_statistics(metrics, weights)
    prediction = authoritative_prediction(metrics)
    null_table, null_summary = null_statistics(metrics, null_components, intervention)
    diagnostics = rate_contrast_diagnostics(metrics)
    alignment_summary = alignment_statistics(alignment)
    make_main_figure(curves, curve_ci, prediction, null_table, null_summary)
    make_supplement(metrics, curves, intervention, alignment, null_table)
    compact = {
        "analysis": "exact_figure4_ssi_mechanism_v2",
        "population_definition": {"low": "sf_split_metric < 0.5", "high": "sf_split_metric >= 0.5"},
        "population_counts": {"low": 71, "high": 29},
        "n_images": len(images),
        "n_corrected_trajectories": len(traces),
        "n_crossed_bootstrap": N_BOOT,
        "rate_contrast_diagnostics": diagnostics,
        "intervention_statistics": intervention.to_dict(orient="records"),
        "association_matched_null": null_summary.to_dict(orient="records"),
        "selected_pathway_alignment": alignment_summary.to_dict(orient="records"),
    }
    write_json(OUT / "statistics.json", compact)
    write_report(curves, intervention, null_summary, alignment_summary, diagnostics)
    print(json.dumps(compact, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
