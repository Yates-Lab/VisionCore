#!/usr/bin/env python3
"""Analyze the completed 16-image held-prefix pilot against matched true/legacy data."""

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

from paper.fig4.mechanism_audit_v1.correction.analyze_core_correction import (
    baseline_contributions,
    image_contributions,
    paired_bank_difference_percent,
    paired_image_bootstrap_percent,
)
from paper.fig4.mechanism_audit_v1.correction.common import CORE_DIR, LEGACY_MATRIX_DIR, OUT_DIR, write_json


FIRST_DIR = OUT_DIR / "first_pass_v1"
PILOT_DIR = FIRST_DIR / "held_pilot"
DATA_DIR = FIRST_DIR / "plot_data"
BANK_STYLE = {
    "legacy wrapped": ("#C2413B", ":", "x"),
    "corrected true history": ("#007C83", "-", "o"),
    "held initial history": ("#6B7280", "--", "s"),
}


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIRST_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    completed = np.load(PILOT_DIR / "completed_images.npy").astype(bool)
    selected = pd.read_csv(PILOT_DIR / "selected_images.csv").sort_values("pilot_ordinal").reset_index(drop=True)
    if not np.all(completed):
        raise RuntimeError(f"Held pilot incomplete: {completed.sum()}/{len(completed)} images")
    image_ids = selected.image_index.to_numpy(int)
    held_ssi = np.asarray(np.load(PILOT_DIR / "ssi_matrix.npy"), float)
    held_expected = np.asarray(np.load(PILOT_DIR / "expected_spikes_matrix.npy"), float)
    true_ssi = np.asarray(np.load(CORE_DIR / "real_trace_true_history_v1/merged/ssi_matrix.npy"), float)[image_ids]
    true_expected = np.asarray(np.load(CORE_DIR / "real_trace_true_history_v1/merged/expected_spikes_matrix.npy"), float)[image_ids]
    legacy_ssi = np.asarray(np.load(LEGACY_MATRIX_DIR / "ssi_matrix.npy"), float).reshape(100, 1000, 100)[image_ids]
    legacy_expected = np.asarray(np.load(LEGACY_MATRIX_DIR / "expected_spikes_matrix.npy"), float).reshape(100, 1000, 100)[image_ids]
    base_ssi = np.asarray(np.load(LEGACY_MATRIX_DIR / "stabilized_ssi_by_image.npy"), float)[image_ids]
    base_expected = np.asarray(np.load(LEGACY_MATRIX_DIR / "stabilized_expected_spikes_by_image.npy"), float)[image_ids]
    banks = {
        "legacy wrapped": (legacy_ssi, legacy_expected),
        "corrected true history": (true_ssi, true_expected),
        "held initial history": (held_ssi, held_expected),
    }
    for name, (ssi, expected) in banks.items():
        if ssi.shape != (len(image_ids), 1000, 100) or not np.isfinite(ssi).all() or not np.isfinite(expected).all():
            raise ValueError(f"Invalid {name} pilot arrays: {ssi.shape}")

    trace = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    unit = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index").reset_index(drop=True)
    path = pd.to_numeric(trace.rendered_path_length_arcmin, errors="coerce").to_numpy(float)
    has_ms = pd.to_numeric(trace.rendered_n_microsaccade_events, errors="coerce").fillna(0).to_numpy(int) > 0
    sf = pd.to_numeric(unit.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {"all units": np.arange(100), "low SF": np.flatnonzero(sf < .5), "high SF": np.flatnonzero(sf >= .5)}
    contexts = {"drift only": np.flatnonzero(~has_ms), "microsaccade-containing": np.flatnonzero(has_ms)}
    curve_rows = []
    aggregate_rows = []
    aggregate_contributions = {}
    baselines = {}
    for context_index, (context, trace_ids) in enumerate(contexts.items()):
        ordered = trace_ids[np.argsort(path[trace_ids], kind="mergesort")]
        chunks = np.array_split(ordered, 8 if context == "drift only" else 5)
        for group_index, (group, unit_ids) in enumerate(groups.items()):
            base = baseline_contributions(base_ssi, base_expected, unit_ids)
            baselines[(context, group)] = base
            for bank_index, (bank, (ssi, expected)) in enumerate(banks.items()):
                aggregate = image_contributions(ssi, expected, trace_ids, unit_ids)
                aggregate_contributions[(bank, context, group)] = aggregate
                point, low, high, p = paired_image_bootstrap_percent(
                    *aggregate, *base, seed=9100 + context_index * 1000 + group_index * 100 + bank_index
                )
                aggregate_rows.append({
                    "bank": bank, "context": context, "sf_group": group,
                    "ssi_percent_vs_stabilized": point, "ci95_low": low, "ci95_high": high,
                    "p_paired_image_boot_sign": p, "n_images": len(image_ids),
                    "n_trajectories": len(trace_ids), "n_units": len(unit_ids),
                })
                for bin_index, ids in enumerate(chunks):
                    moving = image_contributions(ssi, expected, ids, unit_ids)
                    point, low, high, p = paired_image_bootstrap_percent(
                        *moving, *base,
                        seed=9200 + context_index * 10000 + group_index * 1000 + bank_index * 100 + bin_index,
                    )
                    curve_rows.append({
                        "bank": bank, "context": context, "sf_group": group, "bin_index": bin_index,
                        "path_median_arcmin": float(np.median(path[ids])), "n_images": len(image_ids),
                        "n_trajectories": len(ids), "n_units": len(unit_ids),
                        "ssi_percent_vs_stabilized": point, "ci95_low": low, "ci95_high": high,
                        "p_paired_image_boot_sign": p,
                    })
    curves = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(aggregate_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(DATA_DIR / "held_pilot_ssi_curves.csv", index=False)
    summary.to_csv(DATA_DIR / "held_pilot_ssi_summary.csv", index=False)

    differences = {}
    for context_index, context in enumerate(contexts):
        differences[context] = {}
        for group_index, group in enumerate(groups):
            differences[context][group] = paired_bank_difference_percent(
                aggregate_contributions[("corrected true history", context, group)],
                aggregate_contributions[("held initial history", context, group)],
                baselines[(context, group)],
                seed=9300 + context_index * 100 + group_index,
            )

    fig = plt.figure(figsize=(12.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    curve_specs = [
        ("drift only", "low SF", "A  Drift: low SF"),
        ("drift only", "high SF", "B  Drift: high SF"),
        ("microsaccade-containing", "low SF", "C  Microsaccade window: low SF"),
        ("microsaccade-containing", "high SF", "D  Microsaccade window: high SF"),
    ]
    for ax, (context, group, title) in zip((fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])), curve_specs):
        for bank, (color, ls, marker) in BANK_STYLE.items():
            sub = curves.loc[(curves.context == context) & (curves.sf_group == group) & (curves.bank == bank)].sort_values("bin_index")
            y = sub.ssi_percent_vs_stabilized.to_numpy(float)
            ax.errorbar(
                sub.path_median_arcmin, y,
                yerr=np.vstack((y - sub.ci95_low, sub.ci95_high - y)),
                color=color, ls=ls, marker=marker, ms=4, capsize=2, label=bank,
            )
        ax.axhline(0, color="black", lw=.7)
        ax.set(xlabel="path length (arcmin)", ylabel="SSI change vs stabilization (%)", title=title)
    fig.axes[0].legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    drift_summary = summary.loc[summary.context == "drift only"]
    labels = ["all units", "low SF", "high SF"]
    xpos = np.arange(3)
    width = .25
    for offset, bank in zip((-.25, 0, .25), BANK_STYLE):
        sub = drift_summary.set_index(["bank", "sf_group"]).loc[[(bank, x) for x in labels]].reset_index()
        y = sub.ssi_percent_vs_stabilized.to_numpy(float)
        ax.bar(xpos + offset, y, width, color=BANK_STYLE[bank][0], label=bank)
        ax.errorbar(xpos + offset, y, yerr=np.vstack((y - sub.ci95_low, sub.ci95_high - y)), fmt="none", color="black", lw=.8, capsize=2)
    ax.set_xticks(xpos, labels)
    ax.set(ylabel="SSI change vs stabilization (%)", title="E  Matched 16-image drift summary")

    ax = fig.add_subplot(grid[1, 2])
    rows = []
    for context in contexts:
        for group in groups:
            row = {"context": context, "sf_group": group, **differences[context][group]}
            rows.append(row)
    difference_table = pd.DataFrame(rows)
    difference_table.to_csv(DATA_DIR / "held_pilot_true_minus_held.csv", index=False)
    sub = difference_table.loc[difference_table.sf_group.isin(["low SF", "high SF"])].copy()
    labels = [f"{r.context}\n{r.sf_group}" for _, r in sub.iterrows()]
    y = sub.point_percent_points.to_numpy(float)
    colors = ["#007C83" if x == "low SF" else "#D55E00" for x in sub.sf_group]
    ax.bar(np.arange(len(sub)), y, color=colors)
    ax.errorbar(
        np.arange(len(sub)), y,
        yerr=np.vstack((y - sub.ci95_low_percent_points, sub.ci95_high_percent_points - y)),
        fmt="none", color="black", lw=.8, capsize=2,
    )
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(np.arange(len(sub)), labels, rotation=20, ha="right")
    ax.set(ylabel="true − held benefit (percentage points)", title="F  Does real prehistory change the result?")
    fig.suptitle("Figure D — Held-prefix pilot: real prehistory has little effect on the core interaction", fontsize=15, weight="bold")
    export(fig, "fig_D_held_control_pilot")

    stats = {
        "status": "16_image_pilot_not_final_100_image_gate",
        "selection_rule": "orientation coherence >=0.2; 16 equal coherence strata; median image per stratum",
        "image_ids": image_ids.tolist(),
        "n_images": len(image_ids), "n_trajectories": 1000, "n_units": 100,
        "paired_image_bootstrap_draws": 10000,
        "true_minus_held": differences,
        "interpretation_rule": "If true-minus-held intervals are small and core low/high curve shapes agree, real prehistory is not the primary source of the interaction; this remains provisional until 100 images.",
    }
    write_json(FIRST_DIR / "held_pilot_statistics.json", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
