#!/usr/bin/env python3
"""Analyze and visualize held-out velocity-channel necessity/sufficiency tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import write_json


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/layerwise_sftf_v1"
DATA = OUT / "plot_data"
EPS = 1e-12
COLORS = {"Q1": "#0072B2", "Q4": "#D55E00"}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def aggregate_rows(frame: pd.DataFrame) -> float:
    return float(frame.ssi_numerator.sum() / max(float(frame.expected_spikes.sum()), EPS))


def bootstrap_statistics(trace: pd.DataFrame, evaluation_scales: dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(20260811)
    clusters = trace[["image_index", "trace_index"]].drop_duplicates().to_numpy(int)
    rows: list[dict[str, object]] = []
    for quartile in ("Q1", "Q4"):
        scale = float(evaluation_scales[quartile])
        names = {
            "stable": "normal_stable",
            "moving": "normal_moving",
            "necessary": f"{quartile}_velocity_matched_necessary",
            "sufficient": f"{quartile}_velocity_matched_sufficient",
            "readout_only_control": f"{quartile}_readout_weight_only_necessary_control",
            "opposite_speed_control": f"{'Q4' if quartile == 'Q1' else 'Q1'}_velocity_matched_necessary",
        }
        frame = trace.loc[trace.sf_quartile.eq(quartile) & np.isclose(trace.scale, scale)].copy()

        cluster_index = pd.MultiIndex.from_arrays(clusters.T, names=["image_index", "trace_index"])
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, condition in names.items():
            grouped = (
                frame.loc[frame.condition.eq(condition)]
                .groupby(["image_index", "trace_index"])[["ssi_numerator", "expected_spikes"]]
                .sum()
                .reindex(cluster_index)
            )
            if grouped.isna().any().any():
                raise ValueError(f"Missing cluster for {quartile} {condition}")
            arrays[key] = (
                grouped.ssi_numerator.to_numpy(float),
                grouped.expected_spikes.to_numpy(float),
            )

        def derive(value: dict[str, float | np.ndarray]) -> dict[str, float | np.ndarray]:
            benefit = value["moving"] - value["stable"]
            value["movement_benefit"] = benefit
            value["fraction_benefit_removed_by_necessary_swap"] = (
                value["moving"] - value["necessary"]
            ) / np.where(np.abs(benefit) > EPS, benefit, np.nan)
            value["fraction_benefit_transferred_by_sufficient_swap"] = (
                value["sufficient"] - value["stable"]
            ) / np.where(np.abs(benefit) > EPS, benefit, np.nan)
            value["necessary_delta_vs_moving"] = value["necessary"] - value["moving"]
            value["sufficient_delta_vs_stable"] = value["sufficient"] - value["stable"]
            value["readout_only_control_delta_vs_moving"] = value["readout_only_control"] - value["moving"]
            value["opposite_speed_control_delta_vs_moving"] = value["opposite_speed_control"] - value["moving"]
            return value

        observed = derive(
            {key: float(numerator.sum() / max(float(denominator.sum()), EPS)) for key, (numerator, denominator) in arrays.items()}
        )
        sample_indices = rng.integers(0, len(clusters), size=(1000, len(clusters)))
        bootstrap = derive(
            {
                key: numerator[sample_indices].sum(axis=1)
                / np.maximum(denominator[sample_indices].sum(axis=1), EPS)
                for key, (numerator, denominator) in arrays.items()
            }
        )
        for metric, estimate in observed.items():
            distribution = np.asarray(bootstrap[metric], dtype=float)
            rows.append(
                {
                    "sf_quartile": quartile,
                    "evaluation_scale": scale,
                    "metric": metric,
                    "estimate": estimate,
                    "bootstrap_ci_low": float(np.nanpercentile(distribution, 2.5)),
                    "bootstrap_ci_high": float(np.nanpercentile(distribution, 97.5)),
                    "bootstrap_unit": "paired image-by-trajectory cluster",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "convgru_velocity_channel_intervention_statistics.csv", index=False)
    return result


def curve_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    stable = aggregate.loc[aggregate.condition.eq("normal_stable"), ["sf_quartile", "scale", "ssi"]].rename(
        columns={"ssi": "stable_ssi"}
    )
    result = aggregate.merge(stable, on=["sf_quartile", "scale"], validate="many_to_one")
    result["ssi_percent_vs_stable"] = 100 * (result.ssi - result.stable_ssi) / result.stable_ssi.abs().clip(lower=EPS)
    result.to_csv(DATA / "convgru_velocity_channel_intervention_curves.csv", index=False)
    return result


def plot_selection(ax: plt.Axes, selection: pd.DataFrame, target: str) -> None:
    frame = selection.loc[selection.target_sf_quartile.eq(target)].copy()
    # Recover SF and TF coordinates from the grating fits so the selected
    # pathway is visibly a joint SFxTF/speed selection rather than a label.
    fits = pd.read_csv(DATA / "layerwise_channel_sftf_fits.csv.gz")
    fits = fits.loc[
        fits.stage.eq("convgru") & fits.response_metric.eq("f1_amplitude"),
        ["channel", "preferred_sf_cpd", "preferred_tf_hz"],
    ]
    frame = frame.merge(fits, left_on="convgru_channel", right_on="channel", how="left")
    valid = frame.convgru_f1_preferred_speed_dps.notna().to_numpy(bool)
    ax.scatter(
        frame.loc[valid, "preferred_sf_cpd"],
        frame.loc[valid, "preferred_tf_hz"],
        s=13,
        color="0.78",
        alpha=.65,
        label="other ConvGRU channels",
    )
    chosen = frame.selected_velocity_matched.fillna(False)
    ax.scatter(
        frame.loc[chosen, "preferred_sf_cpd"],
        frame.loc[chosen, "preferred_tf_hz"],
        s=42,
        color=COLORS[target],
        edgecolor="white",
        linewidth=.5,
        label=f"selected {target} pathway (n={int(chosen.sum())})",
        zorder=3,
    )
    ax.set(xscale="log", yscale="log", xlim=(.35, 19), ylim=(.35, 60), xticks=[.4, 1, 4, 16], yticks=[.4, 1, 4, 16, 51.2])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set(xlabel="ConvGRU F1 preferred SF (cpd)", ylabel="preferred TF (Hz)", title=f"{target}: FEM-overlap/readout selection")
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_curves(ax: plt.Axes, curves: pd.DataFrame, target: str, evaluation_scale: float) -> None:
    frame = curves.loc[curves.sf_quartile.eq(target)]
    opposite = "Q4" if target == "Q1" else "Q1"
    specifications = [
        ("normal_moving", "normal moving", "black", "-", 2.2),
        (f"{target}_velocity_matched_necessary", "matched channels stabilized (necessity)", COLORS[target], "--", 2.0),
        (f"{target}_velocity_matched_sufficient", "matched channels moved into 0x (sufficiency)", COLORS[target], ":", 2.0),
        (f"{opposite}_velocity_matched_necessary", "opposite-speed mask", "0.5", "--", 1.3),
        (f"{target}_readout_weight_only_necessary_control", "readout-weight-only mask", "0.5", "-.", 1.3),
    ]
    for condition, label, color, style, width in specifications:
        value = frame.loc[frame.condition.eq(condition)].sort_values("scale")
        ax.plot(value.scale, value.ssi_percent_vs_stable, marker="o", ms=3.5, color=color, ls=style, lw=width, label=label)
    ax.axhline(0, color="0.2", lw=.7)
    ax.axvline(evaluation_scale, color=COLORS[target], alpha=.28, lw=5, zorder=0, label="declared FEM evaluation scale")
    ax.set(xlabel="corrected trajectory amplitude", ylabel="SSI change vs paired 0x (%)", title=f"Held-out natural movies: {target} RR100 units")
    ax.grid(alpha=.16)
    ax.legend(frameon=False, fontsize=7)


def figure(selection: pd.DataFrame, curves: pd.DataFrame, evaluation_scales: dict[str, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.1), constrained_layout=True)
    for col, target in enumerate(("Q1", "Q4")):
        plot_selection(axes[0, col], selection, target)
        plot_curves(axes[1, col], curves, target, evaluation_scales[target])
    fig.suptitle(
        "Causal test: do velocity-matched ConvGRU channels carry the eye-movement SSI effect?",
        fontsize=14,
        weight="bold",
    )
    export(fig, "figure_4_velocity_channel_necessity_sufficiency")


def main() -> int:
    configure()
    trace = pd.read_csv(DATA / "convgru_velocity_channel_intervention_trace_results.csv.gz")
    aggregate = pd.read_csv(DATA / "convgru_velocity_channel_intervention_aggregate.csv")
    selection = pd.read_csv(DATA / "convgru_velocity_channel_selection.csv")
    with (OUT / "velocity_channel_intervention_manifest.json").open() as handle:
        manifest = json.load(handle)
    evaluation_scales = {
        key: float(value)
        for key, value in manifest["declared_evaluation_scale_from_corrected_fem_rr100_overlap"].items()
    }
    curves = curve_table(aggregate)
    statistics = bootstrap_statistics(trace, evaluation_scales)
    figure(selection, curves, evaluation_scales)
    mask_efficiency = []
    for quartile in ("Q1", "Q4"):
        frame = selection.loc[selection.target_sf_quartile.eq(quartile)]
        matched_weight = float(frame.loc[frame.selected_velocity_matched, "mean_squared_readout_weight"].sum())
        control_weight = float(
            frame.loc[
                frame.selected_readout_association_only_control,
                "mean_squared_readout_weight",
            ].sum()
        )
        stat = statistics.loc[statistics.sf_quartile.eq(quartile)].set_index("metric")
        matched_effect = abs(float(stat.loc["necessary_delta_vs_moving", "estimate"]))
        control_effect = abs(float(stat.loc["readout_only_control_delta_vs_moving", "estimate"]))
        mask_efficiency.append(
            {
                "sf_quartile": quartile,
                "velocity_matched_mask_readout_association_sum": matched_weight,
                "readout_only_mask_readout_association_sum": control_weight,
                "velocity_matched_necessary_abs_effect_per_association": matched_effect / matched_weight,
                "readout_only_necessary_abs_effect_per_association": control_effect / control_weight,
                "velocity_matched_vs_readout_only_efficiency_ratio": (matched_effect / matched_weight)
                / (control_effect / control_weight),
            }
        )
    summary = {
        "selection_used_natural_image_ssi": False,
        "evaluation_scales": evaluation_scales,
        "mask_efficiency": mask_efficiency,
        "key_statistics": statistics.loc[
            statistics.metric.isin(
                [
                    "movement_benefit",
                    "fraction_benefit_removed_by_necessary_swap",
                    "fraction_benefit_transferred_by_sufficient_swap",
                    "necessary_delta_vs_moving",
                    "sufficient_delta_vs_stable",
                ]
            )
        ].to_dict(orient="records"),
    }
    write_json(OUT / "velocity_channel_intervention_statistics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
