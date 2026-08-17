#!/usr/bin/env python3
"""Analyze and visualize the exact phase-preserving Figure 4 subset."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import write_json


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
RAW = OUT / "exact_arrays"
DATA = OUT / "plot_data"
ARRAY = RAW / "exact_phase_spatial_metrics.npz"
EXAMPLE = RAW / "representative_signed_maps.npz"
LINEAR_GAIN = ROOT / "outputs/figures/fig4/mechanism_audit_v1/targeted_frontend_v1/plot_data/first_resnet_64_mixed_output_gain.csv.gz"
LOW = "#007C83"
HIGH = "#D55E00"
N_BOOT = 5000
EPS = 1e-12


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def paired_mean_delta(values: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float, float]:
    delta = np.asarray(values - baseline, dtype=float)
    point = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, delta.shape[0], size=(N_BOOT, delta.shape[0]))
    tt = rng.integers(0, delta.shape[1], size=(N_BOOT, delta.shape[1]))
    boot = delta[ii[:, :, None], tt[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_percent(values: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    point = 100.0 * (float(values.mean()) - float(baseline.mean())) / max(abs(float(baseline.mean())), EPS)
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, values.shape[0], size=(N_BOOT, values.shape[0]))
    tt = rng.integers(0, values.shape[1], size=(N_BOOT, values.shape[1]))
    a = values[ii[:, :, None], tt[:, None, :]].mean(axis=(1, 2))
    b = baseline[ii[:, :, None], tt[:, None, :]].mean(axis=(1, 2))
    boot = 100.0 * (a - b) / np.maximum(np.abs(b), EPS)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def group_contributions(ssi: np.ndarray, expected: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.sum(ssi[..., units] * expected[..., units], axis=-1), np.sum(expected[..., units], axis=-1)


def paired_ssi_percent(
    point: tuple[np.ndarray, np.ndarray], baseline: tuple[np.ndarray, np.ndarray], seed: int
) -> tuple[float, float, float]:
    pnum, pden = point
    bnum, bden = baseline
    p = float(pnum.sum() / max(float(pden.sum()), EPS))
    b = float(bnum.sum() / max(float(bden.sum()), EPS))
    estimate = 100.0 * (p - b) / max(abs(b), EPS)
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, pnum.shape[0], size=(N_BOOT, pnum.shape[0]))
    tt = rng.integers(0, pnum.shape[1], size=(N_BOOT, pnum.shape[1]))
    pnum_b = pnum[ii[:, :, None], tt[:, None, :]].sum(axis=(1, 2))
    pden_b = pden[ii[:, :, None], tt[:, None, :]].sum(axis=(1, 2))
    bnum_b = bnum[ii[:, :, None], tt[:, None, :]].sum(axis=(1, 2))
    bden_b = bden[ii[:, :, None], tt[:, None, :]].sum(axis=(1, 2))
    pv = pnum_b / np.maximum(pden_b, EPS)
    bv = bnum_b / np.maximum(bden_b, EPS)
    boot = 100.0 * (pv - bv) / np.maximum(np.abs(bv), EPS)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def read_arrays() -> dict[str, np.ndarray]:
    with np.load(ARRAY) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def make_stage_tables(payload: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stages = payload["stages"].astype(str)
    metrics = payload["metrics"].astype(str)
    scales = payload["scales"].astype(float)
    value = payload["stage_metrics"].astype(float)
    stage_rows: list[dict[str, object]] = []
    signature_rows: list[dict[str, object]] = []
    for stage_index, stage in enumerate(stages):
        for scale_index, scale in enumerate(scales):
            energy = value[:, :, scale_index, stage_index, np.flatnonzero(metrics == "total_energy")[0]]
            base_energy = value[:, :, 0, stage_index, np.flatnonzero(metrics == "total_energy")[0]]
            kl = value[:, :, scale_index, stage_index, np.flatnonzero(metrics == "kl_bits_uniform")[0]]
            base_kl = value[:, :, 0, stage_index, np.flatnonzero(metrics == "kl_bits_uniform")[0]]
            ep, elo, ehi = paired_percent(energy, base_energy, 18000 + stage_index * 100 + scale_index)
            kp, klo, khi = paired_mean_delta(kl, base_kl, 28000 + stage_index * 100 + scale_index)
            stage_rows.append(
                {
                    "stage": stage,
                    "scale": scale,
                    "mean_total_energy": float(energy.mean()),
                    "energy_percent_vs_0x": ep,
                    "energy_ci95_low": elo,
                    "energy_ci95_high": ehi,
                    "mean_kl_bits_uniform": float(kl.mean()),
                    "kl_change_bits_vs_0x": kp,
                    "kl_ci95_low": klo,
                    "kl_ci95_high": khi,
                }
            )
        interior = np.arange(1, len(scales) - 1)
        means = value[:, :, :, stage_index, np.flatnonzero(metrics == "kl_bits_uniform")[0]].mean(axis=(0, 1))
        peak_index = int(interior[np.argmax(means[interior])])
        peak = value[:, :, peak_index, stage_index, np.flatnonzero(metrics == "kl_bits_uniform")[0]]
        last = value[:, :, -1, stage_index, np.flatnonzero(metrics == "kl_bits_uniform")[0]]
        point, lo, hi = paired_mean_delta(peak, last, 38000 + stage_index)
        signature_rows.append(
            {
                "stage": stage,
                "interior_peak_scale": float(scales[peak_index]),
                "interior_peak_minus_3x_kl_bits": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "turnover_positive_ci": bool(lo > 0),
                "scope": "low/high readout-associated" if "low" in stage or "high" in stage else "global feature population",
            }
        )

    ssi = payload["ssi"].astype(float)
    expected = payload["expected_spikes"].astype(float)
    final_rows: list[dict[str, object]] = []
    for group_index, (group, units) in enumerate(
        (("low SF", payload["low_unit_indices"].astype(int)), ("high SF", payload["high_unit_indices"].astype(int)))
    ):
        numerator, denominator = group_contributions(ssi, expected, units)
        for scale_index, scale in enumerate(scales):
            point, lo, hi = paired_ssi_percent(
                (numerator[:, :, scale_index], denominator[:, :, scale_index]),
                (numerator[:, :, 0], denominator[:, :, 0]),
                48000 + group_index * 100 + scale_index,
            )
            final_rows.append(
                {"sf_group": group, "scale": scale, "ssi_percent_vs_0x": point, "ci95_low": lo, "ci95_high": hi}
            )
    stages_table = pd.DataFrame(stage_rows)
    final_table = pd.DataFrame(final_rows)
    signature_table = pd.DataFrame(signature_rows)
    stages_table.to_csv(DATA / "exact_stage_energy_concentration.csv", index=False)
    final_table.to_csv(DATA / "exact_final_ssi_curves.csv", index=False)
    signature_table.to_csv(DATA / "spatial_concentration_turnover_signatures.csv", index=False)
    return stages_table, final_table, signature_table


def channel_tables(payload: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    scales = payload["scales"].astype(float)
    metrics = payload["metrics"].astype(str)
    values = payload["first_resnet_channel_metrics"].astype(float)
    energy_i = int(np.flatnonzero(metrics == "total_energy")[0])
    kl_i = int(np.flatnonzero(metrics == "kl_bits_uniform")[0])
    gain = pd.read_csv(LINEAR_GAIN)
    gain = gain.loc[gain.temporal_hz.gt(0) & gain.spatial_cpd.le(16)].copy()
    collapsed = gain.groupby(["first_resnet_output", "spatial_cpd"], as_index=False).mixed_linear_fundamental_gain.sum()
    sf_rows = []
    for channel, frame in collapsed.groupby("first_resnet_output"):
        weights = frame.mixed_linear_fundamental_gain.to_numpy(float)
        sf = frame.spatial_cpd.to_numpy(float)
        sf_rows.append(
            {
                "first_resnet_output": int(channel),
                "characteristic_sf_centroid_cpd": float(np.sum(sf * weights) / max(float(weights.sum()), EPS)),
                "peak_physical_sf_cpd": float(sf[int(np.argmax(weights))]),
            }
        )
    sf_table = pd.DataFrame(sf_rows)
    dose_rows = []
    optimum_rows = []
    for channel in range(values.shape[3]):
        energy_curve = values[:, :, :, channel, energy_i].mean(axis=(0, 1))
        kl_curve = values[:, :, :, channel, kl_i].mean(axis=(0, 1))
        optimum = int(np.argmax(kl_curve))
        censor = "left" if optimum == 0 else "right" if optimum == len(scales) - 1 else "none"
        optimum_rows.append(
            {
                "first_resnet_output": channel,
                "concentration_optimum_scale": float(scales[optimum]),
                "optimum_censoring": censor,
                "peak_mean_kl_bits_uniform": float(kl_curve[optimum]),
            }
        )
        for scale_index, scale in enumerate(scales):
            dose_rows.append(
                {
                    "first_resnet_output": channel,
                    "scale": scale,
                    "mean_total_energy": float(energy_curve[scale_index]),
                    "energy_percent_vs_0x": float(100 * (energy_curve[scale_index] - energy_curve[0]) / max(abs(energy_curve[0]), EPS)),
                    "mean_kl_bits_uniform": float(kl_curve[scale_index]),
                    "kl_change_bits_vs_0x": float(kl_curve[scale_index] - kl_curve[0]),
                }
            )
    dose = pd.DataFrame(dose_rows)
    optimum = pd.DataFrame(optimum_rows).merge(sf_table, on="first_resnet_output", validate="one_to_one")
    uncensored = optimum.optimum_censoring.eq("none")
    rho_all, p_all = spearmanr(optimum.characteristic_sf_centroid_cpd, optimum.concentration_optimum_scale)
    if uncensored.sum() >= 3:
        rho_unc, p_unc = spearmanr(
            optimum.loc[uncensored, "characteristic_sf_centroid_cpd"],
            optimum.loc[uncensored, "concentration_optimum_scale"],
        )
    else:
        rho_unc, p_unc = math.nan, math.nan
    stats = {
        "spearman_all_channels": float(rho_all),
        "spearman_all_p": float(p_all),
        "spearman_uncensored": float(rho_unc),
        "spearman_uncensored_p": float(p_unc),
        "n_channels": int(len(optimum)),
        "n_left_censored": int(optimum.optimum_censoring.eq("left").sum()),
        "n_right_censored": int(optimum.optimum_censoring.eq("right").sum()),
        "n_uncensored": int(uncensored.sum()),
    }
    dose.to_csv(DATA / "first_conv_channel_dose_curves.csv", index=False)
    optimum.to_csv(DATA / "first_conv_channel_concentration_optima.csv", index=False)
    return dose, optimum, stats


def figure_a(payload: dict[str, np.ndarray]) -> None:
    with np.load(EXAMPLE) as archive:
        ex = {key: np.asarray(archive[key]) for key in archive.files}
    scales = ex["scales"].astype(float)
    signed = ex["first_conv_signed_map"].astype(float)
    squared = ex["first_conv_squared_map"].astype(float)
    final = ex["final_high_population_map"].astype(float)
    xt = ex["retinal_xt_slice"].astype(float)
    signed_lim = float(np.quantile(np.abs(signed), 0.997))
    energy_lim = float(np.quantile(squared, 0.997))
    final_min, final_max = [float(x) for x in np.quantile(final, [0.003, 0.997])]

    image_ids = payload["selected_image_index"].astype(int)
    trace_ids = payload["selected_trace_index"].astype(int)
    ii = int(np.flatnonzero(image_ids == int(ex["image_index"]))[0])
    tt = int(np.flatnonzero(trace_ids == int(ex["trace_index"]))[0])
    all_scales = payload["scales"].astype(float)
    stages = payload["stages"].astype(str)
    metrics = payload["metrics"].astype(str)
    pre_i = int(np.flatnonzero(stages == "resblock1_preactivation")[0])
    e_i = int(np.flatnonzero(metrics == "total_energy")[0])
    k_i = int(np.flatnonzero(metrics == "kl_bits_uniform")[0])
    high = payload["high_unit_indices"].astype(int)
    fig, axes = plt.subplots(4, len(scales), figsize=(13.2, 11.2), constrained_layout=True)
    for column, scale in enumerate(scales):
        si = int(np.flatnonzero(np.isclose(all_scales, scale))[0])
        rate_num = np.sum(payload["ssi"][ii, tt, si, high] * payload["expected_spikes"][ii, tt, si, high])
        rate_den = np.sum(payload["expected_spikes"][ii, tt, si, high])
        final_ssi = float(rate_num / max(float(rate_den), EPS))
        energy = float(payload["stage_metrics"][ii, tt, si, pre_i, e_i])
        kl = float(payload["stage_metrics"][ii, tt, si, pre_i, k_i])
        axes[0, column].imshow(xt[column], cmap="gray", aspect="auto", vmin=float(xt.min()), vmax=float(xt.max()))
        axes[0, column].set_title(f"{scale:g}× movement", weight="bold")
        axes[1, column].imshow(signed[column], cmap="RdBu_r", vmin=-signed_lim, vmax=signed_lim)
        axes[2, column].imshow(squared[column], cmap="magma", vmin=0, vmax=energy_lim)
        axes[3, column].imshow(final[column], cmap="viridis", vmin=final_min, vmax=final_max)
        axes[1, column].text(0.02, 0.02, f"mean E={energy:.3g}\nmean KL={kl:.3f} bits", transform=axes[1, column].transAxes, va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=.75, edgecolor="none"))
        axes[3, column].text(0.02, 0.02, f"high-SF SSI={final_ssi:.3f}", transform=axes[3, column].transAxes, va="bottom", fontsize=8, color="white", bbox=dict(facecolor="black", alpha=.55, edgecolor="none"))
        for row in range(4):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    for row, label in enumerate(("retinal x–t slice", "signed first-conv feature", "same feature squared energy", "final high-SF population map")):
        axes[row, 0].set_ylabel(label, fontsize=10)
    fig.suptitle("Figure A — More early feature energy need not mean a more localized final representation\n(common scales across movement conditions; same image, trace, time, and channel)", fontsize=15, weight="bold")
    export(fig, "figure_A_phase_preserved_example")


def figure_b(stage: pd.DataFrame, final: pd.DataFrame) -> None:
    selected = ["temporal_frontend", "resblock1_preactivation", "resblock1_output", "resblock2_output", "convgru"]
    labels = ["temporal frontend", "first conv preactivation", "ResBlock 1 output", "ResBlock 2 output", "ConvGRU"]
    colors = plt.get_cmap("viridis")(np.linspace(.1, .9, len(selected)))
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 11.2), sharex=True, constrained_layout=True)
    for name, label, color in zip(selected, labels, colors):
        sub = stage.loc[stage.stage.eq(name)].sort_values("scale")
        axes[0].plot(sub.scale, sub.energy_percent_vs_0x, marker="o", color=color, label=label)
        axes[1].plot(sub.scale, sub.kl_change_bits_vs_0x, marker="o", color=color, label=label)
    for name, label, color, marker in (
        ("convgru_low_readout_weighted", "ConvGRU weighted by low-SF readout", LOW, "o"),
        ("convgru_high_readout_weighted", "ConvGRU weighted by high-SF readout", HIGH, "s"),
    ):
        sub = stage.loc[stage.stage.eq(name)].sort_values("scale")
        axes[1].plot(sub.scale, sub.kl_change_bits_vs_0x, marker=marker, color=color, lw=2.6, ls="--", label=label)
    for group, color, marker in (("low SF", LOW, "o"), ("high SF", HIGH, "s")):
        sub = final.loc[final.sf_group.eq(group)].sort_values("scale")
        y = sub.ssi_percent_vs_0x.to_numpy(float)
        axes[2].errorbar(sub.scale, y, yerr=np.vstack((y - sub.ci95_low, sub.ci95_high - y)), marker=marker, capsize=3, color=color, lw=2.4, label=group)
    for ax in axes:
        ax.axhline(0, color="black", lw=.7)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("total feature energy\nchange vs 0× (%)")
    axes[1].set_ylabel("spatial concentration\nchange vs 0× (KL bits)")
    axes[2].set_ylabel("final SSI change\nvs 0× (%)")
    axes[2].set_xlabel("trajectory amplitude (× measured drift)")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[2].legend(frameon=False)
    fig.suptitle("Figure B — Where does the high-SF turnover first emerge?", fontsize=15, weight="bold")
    export(fig, "figure_B_layerwise_energy_and_concentration")


def figure_channel_scatter(optimum: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    style = {"none": ("o", "#2563EB"), "left": ("<", "#6B7280"), "right": (">", "#6B7280")}
    for censor, frame in optimum.groupby("optimum_censoring"):
        marker, color = style[censor]
        ax.scatter(frame.characteristic_sf_centroid_cpd, frame.concentration_optimum_scale, marker=marker, color=color, s=52, alpha=.85, label=f"{censor} censored" if censor != "none" else "interior optimum")
    ax.set(xscale="log", xlabel="first-conv characteristic physical SF (cpd)", ylabel="movement scale maximizing spatial concentration", yticks=[0, .5, 1, 2, 3])
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.set_title("Channel SF tuning versus concentration-optimal movement\n(boundary optima shown as censored)")
    export(fig, "supplement_channel_sf_vs_concentration_optimum")


def mapping_intervention_summary(optimum: pd.DataFrame) -> dict[str, object]:
    swaps = pd.read_csv(DATA / "representative_activation_branch_swaps.csv")
    validation = pd.read_csv(DATA / "full_activation_swap_validation.csv")
    gradients = pd.read_csv(DATA / "representative_gradient_channel_mapping.csv.gz")
    association = pd.read_csv(DATA / "convgru_channel_group_association.csv").pivot(
        index="convgru_channel", columns="figure4_sf_group", values="normalized_mean_squared_readout_weight"
    )
    rho_readout, p_readout = spearmanr(association.low, association.high)
    mixture = (association.low + association.high) / 2
    js = 0.5 * np.sum(association.low * np.log2(association.low / mixture)) + 0.5 * np.sum(
        association.high * np.log2(association.high / mixture)
    )
    tv = 0.5 * np.abs(association.low - association.high).sum()
    first = gradients.loc[gradients.stage.eq("first_conv_preactivation")].merge(
        optimum[["first_resnet_output", "characteristic_sf_centroid_cpd", "peak_physical_sf_cpd"]],
        left_on="channel",
        right_on="first_resnet_output",
        validate="many_to_one",
    )
    gradient_rows = []
    for (group, scale), frame in first.groupby(["sf_group", "scale"]):
        weight = frame.normalized_abs_gradient_times_activation.to_numpy(float)
        gradient_rows.append(
            {
                "sf_group": group,
                "scale": float(scale),
                "gradient_weighted_characteristic_sf_cpd": float(np.sum(weight * frame.characteristic_sf_centroid_cpd)),
                "gradient_weighted_peak_sf_cpd": float(np.sum(weight * frame.peak_physical_sf_cpd)),
            }
        )
    gradient_summary = pd.DataFrame(gradient_rows)
    gradient_summary.to_csv(DATA / "first_conv_gradient_weighted_sf_mapping.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True, sharey=True, constrained_layout=True)
    styles = (
        ("normal_moving", "normal moving", "black", "o", "-"),
        ("rb1_moving_main_stable_shortcut", "RB1 moving main + stable shortcut", "#7C3AED", "s", "-"),
        ("rb1_stable_main_moving_shortcut", "RB1 stable main + moving shortcut", "#7C3AED", "s", "--"),
        ("rb2_moving_main_stable_shortcut", "RB2 moving main + stable shortcut", "#2563EB", "^", "-"),
        ("rb2_stable_main_moving_shortcut", "RB2 stable main + moving shortcut", "#2563EB", "^", "--"),
    )
    for ax, group, group_label in zip(axes, ("low", "high"), ("low-SF final group", "high-SF final group")):
        for condition, label, color, marker, linestyle in styles:
            frame = swaps.loc[swaps.sf_group.eq(group) & swaps.condition.eq(condition)].sort_values("scale")
            ax.plot(frame.scale, frame.ssi_percent_vs_stable, color=color, marker=marker, ls=linestyle, lw=2, label=label)
        ax.axhline(0, color="black", lw=.7)
        ax.grid(alpha=.18)
        ax.set_title(group_label)
        ax.set_xlabel("trajectory amplitude")
    axes[0].set_ylabel("final SSI change vs stabilized (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Supplement — Residual-branch hybrids in one predetermined example\n(diagnostic localization; no sampling uncertainty)", fontsize=14, weight="bold")
    export(fig, "supplement_residual_branch_hybrids")
    return {
        "max_full_stage_swap_error": float(validation.max_abs_rate_map_error_vs_exact_anchor.max()),
        "full_stage_swap_warning": "A complete activation deterministically fixes all downstream output, so these zero-error swaps validate the implementation but cannot localize where information was computed.",
        "convgru_low_high_squared_readout_weight_spearman": float(rho_readout),
        "convgru_low_high_squared_readout_weight_spearman_p": float(p_readout),
        "convgru_low_high_readout_weight_js_bits": float(js),
        "convgru_low_high_readout_weight_total_variation": float(tv),
        "convgru_top10_channel_overlap": int(
            len(set(association.low.nlargest(10).index) & set(association.high.nlargest(10).index))
        ),
        "branch_swap_scope": "one objectively selected image, median-path selected drift, all 40 scored frames; diagnostic only",
    }


def figure_c_if_localized(signature: pd.DataFrame, stage: pd.DataFrame) -> dict[str, object]:
    order = ["resblock1_preactivation", "resblock1_normalized", "resblock1_splitrelu", "resblock1_main_pooled", "resblock1_output", "resblock2_output", "convgru"]
    ordered = signature.set_index("stage").reindex(order).dropna().reset_index()
    hits = ordered.loc[ordered.turnover_positive_ci]
    if hits.empty:
        return {"made": False, "reason": "No global internal stage had an interior spatial-concentration peak reliably above 3x."}
    current = str(hits.iloc[0].stage)
    position = order.index(current)
    if position == 0:
        return {"made": False, "reason": "Turnover was already present at first-conv preactivation; no captured earlier same-coordinate transition is available."}
    previous = order[position - 1]
    with np.load(EXAMPLE) as archive:
        ex = {key: np.asarray(archive[key]) for key in archive.files}
    prev_key = f"{previous}_spatial_energy_map"
    curr_key = f"{current}_spatial_energy_map"
    if prev_key not in ex or curr_key not in ex:
        return {"made": False, "reason": f"Representative maps unavailable for {previous} -> {current}."}
    scales = ex["scales"].astype(float)
    before = ex[prev_key].astype(float)
    after = ex[curr_key].astype(float)
    fig = plt.figure(figsize=(14.2, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.35])
    lim_before = float(np.quantile(before, .997)); lim_after = float(np.quantile(after, .997))
    for column, scale in enumerate(scales):
        ax = fig.add_subplot(grid[0, column]); ax.imshow(before[column], cmap="magma", vmin=0, vmax=lim_before); ax.set_title(f"{scale:g}×"); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(grid[1, column]); ax.imshow(after[column], cmap="magma", vmin=0, vmax=lim_after); ax.set_xticks([]); ax.set_yticks([])
        if column == 0:
            fig.axes[-2].set_ylabel(previous.replace("_", " "))
            fig.axes[-1].set_ylabel(current.replace("_", " "))
    ax = fig.add_subplot(grid[:, 4])
    for name, color, marker in ((previous, "#2563EB", "o"), (current, "#7C3AED", "s")):
        sub = stage.loc[stage.stage.eq(name)].sort_values("scale")
        ax.plot(sub.scale, sub.kl_change_bits_vs_0x, color=color, marker=marker, lw=2.4, label=name.replace("_", " "))
    ax.axhline(0, color="black", lw=.7); ax.grid(alpha=.2)
    ax.set(xlabel="trajectory amplitude", ylabel="KL concentration change vs 0× (bits)")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Figure C — Candidate turnover-generating transition: {previous.replace('_', ' ')} → {current.replace('_', ' ')}\n(common colour scale across movement within each row)", fontsize=14, weight="bold")
    export(fig, "figure_C_candidate_turnover_transition")
    return {"made": True, "previous_stage": previous, "current_stage": current}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    payload = read_arrays()
    stage, final, signature = make_stage_tables(payload)
    _, optimum, channel_stats = channel_tables(payload)
    figure_a(payload)
    figure_b(stage, final)
    figure_channel_scatter(optimum)
    figure_c = figure_c_if_localized(signature, stage)
    mapping = mapping_intervention_summary(optimum)
    endpoint = {
        group: {
            str(float(row.scale)): {"percent_vs_0x": float(row.ssi_percent_vs_0x), "ci95": [float(row.ci95_low), float(row.ci95_high)]}
            for _, row in frame.iterrows()
        }
        for group, frame in final.groupby("sf_group")
    }
    summary = {
        "status": "complete_exact_8_image_24_drift_subset",
        "endpoint": endpoint,
        "endpoint_gate_low_peak_scale": float(final.loc[final.sf_group.eq("low SF")].sort_values("ssi_percent_vs_0x").iloc[-1].scale),
        "endpoint_gate_high_peak_scale": float(final.loc[final.sf_group.eq("high SF")].sort_values("ssi_percent_vs_0x").iloc[-1].scale),
        "channel_sf_vs_concentration_optimum": channel_stats,
        "mapping_and_interventions": mapping,
        "figure_c": figure_c,
        "bootstrap": "5000 paired crossed resamples of image and trajectory indices; model and RR100 units fixed",
        "metric_warning": "KL, entropy, and effective area describe spatial concentration of squared feature energy; they are not SSI or mutual information.",
    }
    write_json(OUT / "analysis_statistics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
