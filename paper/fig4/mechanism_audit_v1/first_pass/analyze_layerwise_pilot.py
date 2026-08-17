#!/usr/bin/env python3
"""Analyze and plot the layerwise/frontend-replacement first-pass pilot."""

from __future__ import annotations

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

from paper.fig4.mechanism_audit_v1.correction.common import LEGACY_MATRIX_DIR, OUT_DIR, write_json


FIRST = OUT_DIR / "first_pass_v1"
OUT = FIRST / "layerwise_pilot"
DATA = FIRST / "plot_data"
LOW = "#007C83"
HIGH = "#D55E00"
EPS = 1e-8


def paired_bootstrap(values: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float, float]:
    delta = np.asarray(values - baseline, float)
    point = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    image_ids = rng.integers(0, delta.shape[0], size=(10000, delta.shape[0]))
    trace_ids = rng.integers(0, delta.shape[1], size=(10000, delta.shape[1]))
    boot = np.empty(10000)
    for b in range(10000):
        boot[b] = np.mean(delta[np.ix_(image_ids[b], trace_ids[b])])
    return point, *[float(x) for x in np.percentile(boot, [2.5, 97.5])]


def group_ssi_contributions(ssi: np.ndarray, expected: np.ndarray, units: np.ndarray):
    return (
        # Preserve image, trajectory, and scale; pool only the selected units.
        np.sum(ssi[..., units] * expected[..., units], axis=-1),
        np.sum(expected[..., units], axis=-1),
    )


def ratio_delta_bootstrap(point, base, seed):
    point_value = point[0].sum() / max(point[1].sum(), EPS)
    base_value = base[0].sum() / max(base[1].sum(), EPS)
    estimate = 100 * (point_value - base_value) / base_value
    rng = np.random.default_rng(seed)
    n_images, n_traces = point[0].shape
    boot = np.empty(10000)
    for b in range(10000):
        ii = rng.integers(0, n_images, n_images)
        tt = rng.integers(0, n_traces, n_traces)
        ix = np.ix_(ii, tt)
        p = point[0][ix].sum() / max(point[1][ix].sum(), EPS)
        q = base[0][ix].sum() / max(base[1][ix].sum(), EPS)
        boot[b] = 100 * (p - q) / q
    return float(estimate), *[float(x) for x in np.percentile(boot, [2.5, 97.5])]


def export(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIRST / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    with np.load(OUT / "layerwise_response.npz") as archive:
        ssi = np.asarray(archive["ssi"], float)
        expected = np.asarray(archive["expected_spikes"], float)
        replacement_ssi = np.asarray(archive["frontend_replacement_ssi"], float)
        replacement_expected = np.asarray(archive["frontend_replacement_expected_spikes"], float)
        layer_kl = np.asarray(archive["layer_kl_bits_from_uniform"], float)
        layer_effective = np.asarray(archive["layer_effective_area_fraction"], float)
        stages = archive["stages"].astype(str)
        scales = np.asarray(archive["scales"], float)
    unit = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index")
    sf = pd.to_numeric(unit.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"low SF": np.flatnonzero(sf < .5), "high SF": np.flatnonzero(sf >= .5)}
    layer_rows = []
    for stage_index, stage in enumerate(stages):
        for scale_index, scale in enumerate(scales):
            value, low, high = paired_bootstrap(
                layer_kl[:, :, scale_index, stage_index], layer_kl[:, :, 0, stage_index],
                12000 + stage_index * 100 + scale_index,
            )
            eff, eff_low, eff_high = paired_bootstrap(
                layer_effective[:, :, scale_index, stage_index], layer_effective[:, :, 0, stage_index],
                13000 + stage_index * 100 + scale_index,
            )
            layer_rows.append({
                "stage": stage, "scale": scale, "kl_change_bits": value,
                "kl_ci95_low": low, "kl_ci95_high": high,
                "effective_area_fraction_change": eff,
                "effective_area_ci95_low": eff_low, "effective_area_ci95_high": eff_high,
            })
    layer_table = pd.DataFrame(layer_rows)
    curve_rows = []
    intervention_rows = []
    for group_index, (group, units) in enumerate(groups.items()):
        num, den = group_ssi_contributions(ssi, expected, units)
        for scale_index, scale in enumerate(scales):
            value, low, high = ratio_delta_bootstrap(
                (num[:, :, scale_index], den[:, :, scale_index]), (num[:, :, 0], den[:, :, 0]),
                14000 + group_index * 100 + scale_index,
            )
            curve_rows.append({"sf_group": group, "scale": scale, "ssi_percent_vs_0x": value, "ci95_low": low, "ci95_high": high})
        rnum, rden = group_ssi_contributions(replacement_ssi[:, :, None, :], replacement_expected[:, :, None, :], units)
        normal_1x = (num[:, :, 2], den[:, :, 2])
        replacement = (rnum[:, :, 0], rden[:, :, 0])
        baseline = (num[:, :, 0], den[:, :, 0])
        for label, point, seed in (("normal moving 1x", normal_1x, 15000 + group_index), ("moving 1x + stabilized frontend", replacement, 15100 + group_index)):
            value, low, high = ratio_delta_bootstrap(point, baseline, seed)
            intervention_rows.append({"sf_group": group, "condition": label, "ssi_percent_vs_0x": value, "ci95_low": low, "ci95_high": high})
    curves = pd.DataFrame(curve_rows)
    intervention = pd.DataFrame(intervention_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    layer_table.to_csv(DATA / "layerwise_spatial_concentration.csv", index=False)
    curves.to_csv(DATA / "layerwise_final_ssi_curves.csv", index=False)
    intervention.to_csv(DATA / "frontend_replacement_intervention.csv", index=False)

    fig = plt.figure(figsize=(13.3, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, :])
    xpos = np.arange(len(stages))
    for scale, color in ((.5, "#9CA3AF"), (1., "#2563EB"), (2., "#7C3AED")):
        sub = layer_table.loc[layer_table.scale == scale].set_index("stage").loc[stages].reset_index()
        ax.errorbar(xpos, sub.kl_change_bits, yerr=np.vstack((sub.kl_change_bits - sub.kl_ci95_low, sub.kl_ci95_high - sub.kl_change_bits)), marker="o", capsize=2, color=color, label=f"{scale:g}×")
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(xpos, ["retinal\ninput", "temporal\nfrontend", "stem", "ResBlock 1", "ResBlock 2", "ConvGRU"])
    ax.set(ylabel="change in feature-energy concentration\nKL from spatial uniform (bits)", title="A  Where does motion first make internal feature energy more spatially concentrated?")
    ax.legend(frameon=False, ncol=3)

    ax = fig.add_subplot(grid[1, 0])
    for group, color, marker in (("low SF", LOW, "o"), ("high SF", HIGH, "s")):
        sub = curves.loc[curves.sf_group == group].sort_values("scale")
        y = sub.ssi_percent_vs_0x.to_numpy(float)
        ax.errorbar(sub.scale, y, yerr=np.vstack((y - sub.ci95_low, sub.ci95_high - y)), color=color, marker=marker, capsize=2, label=group)
    ax.axhline(0, color="black", lw=.7); ax.axvline(1, color="#6B7280", ls=":")
    ax.set(xlabel="within-window trajectory amplitude", ylabel="final SSI change vs 0× (%)", title="B  Final readout endpoint")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[1, 1])
    labels = []
    x = np.arange(4)
    y = []
    lo = []; hi = []; colors = []
    for group in ("low SF", "high SF"):
        for condition in ("normal moving 1x", "moving 1x + stabilized frontend"):
            row = intervention.loc[(intervention.sf_group == group) & (intervention.condition == condition)].iloc[0]
            labels.append(f"{group}\n{'normal' if condition.startswith('normal') else 'frontend replaced'}")
            y.append(row.ssi_percent_vs_0x); lo.append(row.ci95_low); hi.append(row.ci95_high)
            colors.append(LOW if group == "low SF" else HIGH)
    y = np.asarray(y); lo = np.asarray(lo); hi = np.asarray(hi)
    bars = ax.bar(x, y, color=colors)
    for bar, alpha in zip(bars, (1.0, 0.5, 1.0, 0.5)):
        bar.set_alpha(alpha)
    ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), fmt="none", color="black", capsize=2)
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set(ylabel="SSI change vs matched 0× (%)", title="C  Matched frontend replacement")

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    ax.text(.02, .90, "Intervention", fontsize=12, weight="bold", transform=ax.transAxes)
    ax.text(.02, .78, "moving 1× retinal input", color="#2563EB", transform=ax.transAxes)
    ax.text(.02, .67, "↓", transform=ax.transAxes)
    ax.text(.02, .56, "replace its frontend activation", transform=ax.transAxes)
    ax.text(.02, .46, "with the matched 0× activation", color="#7C3AED", transform=ax.transAxes)
    ax.text(.02, .35, "↓ unchanged stem / ResNet / ConvGRU", transform=ax.transAxes)
    ax.text(.02, .23, "↓ final SSI", transform=ax.transAxes)
    ax.text(.02, .06, "Causal within the fitted model;\nnot a biological intervention.", fontsize=9, color="#4B5563", transform=ax.transAxes)
    fig.suptitle("Figure G — Layerwise localization and a matched temporal-frontend intervention", fontsize=15, weight="bold")
    export(fig, "fig_G_layerwise_frontend_intervention")

    stats = {"status": "eight_image_eight_trajectory_pilot", "normal_and_replacement": {}}
    for _, row in intervention.iterrows():
        stats["normal_and_replacement"][f"{row.sf_group}__{row.condition}"] = {
            "point_percent": float(row.ssi_percent_vs_0x), "ci95": [float(row.ci95_low), float(row.ci95_high)]
        }
    first_positive = None
    one_x = layer_table.loc[layer_table.scale == 1].set_index("stage").loc[stages]
    for stage, row in one_x.iterrows():
        if row.kl_ci95_low > 0:
            first_positive = stage
            break
    stats["first_stage_with_positive_1x_kl_change_ci"] = first_positive
    stats["metric_warning"] = "Intermediate metric is spatial concentration of squared feature energy, not SSI or mutual information."
    write_json(FIRST / "layerwise_statistics.json", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
