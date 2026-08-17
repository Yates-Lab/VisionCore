#!/usr/bin/env python3
"""First-pass real-event history audit using the corrected 71-frame bank."""

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
    paired_image_bootstrap_percent,
)
from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, CORE_DIR, LEGACY_MATRIX_DIR, OUT_DIR, write_json


FIRST_DIR = OUT_DIR / "first_pass_v1"
DATA_DIR = FIRST_DIR / "plot_data"
LOW = "#007C83"
HIGH = "#D55E00"


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIRST_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        history = np.asarray(archive["true_history_xy"], float)
    trace = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    unit = pd.read_csv(LEGACY_MATRIX_DIR / "unit_feature_table.csv").sort_values("unit_index").reset_index(drop=True)
    ssi = np.asarray(np.load(CORE_DIR / "real_trace_true_history_v1/merged/ssi_matrix.npy"), float)
    expected = np.asarray(np.load(CORE_DIR / "real_trace_true_history_v1/merged/expected_spikes_matrix.npy"), float)
    base_ssi = np.asarray(np.load(LEGACY_MATRIX_DIR / "stabilized_ssi_by_image.npy"), float)
    base_expected = np.asarray(np.load(LEGACY_MATRIX_DIR / "stabilized_expected_spikes_by_image.npy"), float)

    speed = np.linalg.norm(np.diff(history, axis=1), axis=2) * 120.0
    threshold = pd.to_numeric(trace.rendered_microsaccade_threshold_dps, errors="coerce").to_numpy(float)
    fast_step = speed >= threshold[:, None]
    scored_fast = fast_step[:, 31:].any(axis=1)
    prefix_fast = fast_step[:, :31].any(axis=1)
    cached_ms = pd.to_numeric(trace.rendered_n_microsaccade_events, errors="coerce").fillna(0).to_numpy(int) > 0
    if not np.array_equal(scored_fast, cached_ms):
        raise AssertionError(
            f"Recovered scored event classification differs for {np.count_nonzero(scored_fast != cached_ms)} traces"
        )
    strict_drift = ~fast_step.any(axis=1)
    prefix_only = prefix_fast & ~scored_fast
    event_class = np.full(len(trace), "scored_fast_event", object)
    event_class[strict_drift] = "no_fast_event_full_71"
    event_class[prefix_only] = "prefix_only_fast_event"

    per_output_rows = []
    for output in range(40):
        # For output t, its 32 input positions are position indices t..t+31,
        # hence its 31 within-window displacement steps are t..t+30.
        window = fast_step[:, output : output + 31]
        for trace_id in range(len(trace)):
            event_indices = np.flatnonzero(window[trace_id])
            if len(event_indices):
                lag_frames = 30 - int(event_indices[-1])
                lag_ms = 1000.0 * lag_frames / 120.0
                if lag_ms < 100:
                    category = "<100 ms"
                elif lag_ms < 200:
                    category = "100–200 ms"
                else:
                    category = "200–258 ms"
            else:
                lag_frames = -1
                lag_ms = np.nan
                category = "none in 32-frame history"
            per_output_rows.append({
                "trajectory_id": trace_id,
                "output_index": output,
                "output_time_ms": 1000.0 * output / 120.0,
                "cached_scored_class": "microsaccade" if cached_ms[trace_id] else "drift_only",
                "last_fast_event_category": category,
                "last_fast_event_lag_frames": lag_frames,
                "last_fast_event_lag_ms": lag_ms,
                "event_class_full_71": event_class[trace_id],
            })
    per_output = pd.DataFrame(per_output_rows)
    per_output.to_csv(DATA_DIR / "causal_event_history_per_output.csv.gz", index=False, compression="gzip")
    trajectory_table = trace[["trace_bank_index", "rendered_path_length_arcmin", "rendered_n_microsaccade_events"]].copy()
    trajectory_table["prefix_has_fast_step"] = prefix_fast
    trajectory_table["scored_has_fast_step"] = scored_fast
    trajectory_table["full_71_has_fast_step"] = fast_step.any(axis=1)
    trajectory_table["event_class_full_71"] = event_class
    trajectory_table["max_full_history_speed_deg_s"] = speed.max(axis=1)
    trajectory_table["event_threshold_deg_s"] = threshold
    trajectory_table.to_csv(DATA_DIR / "causal_event_history_trajectories.csv", index=False)

    sf = pd.to_numeric(unit.sf_split_metric, errors="coerce").to_numpy(float)
    groups = {
        "low SF": np.flatnonzero(sf < 0.5),
        "high SF": np.flatnonzero(sf >= 0.5),
    }
    contexts = {
        "historical drift-only (n=800)": np.flatnonzero(~cached_ms),
        f"strict no-fast-event drift (n={strict_drift.sum()})": np.flatnonzero(strict_drift),
        f"prefix-only fast event (n={prefix_only.sum()})": np.flatnonzero(prefix_only),
        f"scored fast event (n={scored_fast.sum()})": np.flatnonzero(scored_fast),
    }
    curve_rows = []
    aggregate_rows = []
    path = pd.to_numeric(trace.rendered_path_length_arcmin, errors="coerce").to_numpy(float)
    for context_index, (context, trace_ids) in enumerate(contexts.items()):
        if len(trace_ids) == 0:
            continue
        ordered = trace_ids[np.argsort(path[trace_ids], kind="mergesort")]
        n_bins = 8 if len(trace_ids) >= 100 else min(5, len(trace_ids))
        chunks = np.array_split(ordered, n_bins)
        for group_index, (group, unit_ids) in enumerate(groups.items()):
            base_num, base_den = baseline_contributions(base_ssi, base_expected, unit_ids)
            moving_num, moving_den = image_contributions(ssi, expected, trace_ids, unit_ids)
            point, low, high, _ = paired_image_bootstrap_percent(
                moving_num, moving_den, base_num, base_den, seed=8100 + context_index * 100 + group_index
            )
            aggregate_rows.append({
                "event_history_group": context, "sf_group": group, "n_trajectories": len(trace_ids),
                "ssi_percent_vs_stabilized": point, "ci95_low": low, "ci95_high": high,
            })
            for bin_index, ids in enumerate(chunks):
                moving_num, moving_den = image_contributions(ssi, expected, ids, unit_ids)
                point, low, high, _ = paired_image_bootstrap_percent(
                    moving_num, moving_den, base_num, base_den,
                    seed=8200 + context_index * 1000 + group_index * 100 + bin_index,
                )
                curve_rows.append({
                    "event_history_group": context, "sf_group": group, "bin_index": bin_index,
                    "n_trajectories": len(ids), "path_median_arcmin": float(np.median(path[ids])),
                    "ssi_percent_vs_stabilized": point, "ci95_low": low, "ci95_high": high,
                })
    curves = pd.DataFrame(curve_rows)
    aggregates = pd.DataFrame(aggregate_rows)
    curves.to_csv(DATA_DIR / "event_history_ssi_curves.csv", index=False)
    aggregates.to_csv(DATA_DIR / "event_history_ssi_summary.csv", index=False)

    fig = plt.figure(figsize=(14.2, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax = fig.add_subplot(grid[0, 0])
    counts = pd.Series(event_class).value_counts().reindex(
        ["no_fast_event_full_71", "prefix_only_fast_event", "scored_fast_event"]
    )
    ax.bar(["no fast event\nanywhere", "prefix event\nonly", "scored-window\nevent"], counts, color=["#4B5563", "#8B5CF6", "#C2413B"])
    for i, v in enumerate(counts):
        ax.text(i, v + 10, str(int(v)), ha="center")
    ax.set(ylabel="trajectories", title="A  Event classes across 71 frames")
    ax.title.set_fontsize(10)

    ax = fig.add_subplot(grid[0, 1])
    count_by_output = (
        per_output.assign(has_event=per_output.last_fast_event_category != "none in 32-frame history")
        .groupby("output_time_ms", as_index=False).has_event.sum()
    )
    ax.plot(count_by_output.output_time_ms, count_by_output.has_event, color="#8B5CF6", lw=2)
    ax.set(xlabel="scored output time (ms)", ylabel="histories containing a fast event", title="B  Events inside each 32-frame input")
    ax.title.set_fontsize(10)

    ax = fig.add_subplot(grid[0, 2])
    order = ["none in 32-frame history", "200–258 ms", "100–200 ms", "<100 ms"]
    category_counts = per_output.last_fast_event_category.value_counts().reindex(order).fillna(0)
    ax.barh(order, category_counts, color=["#6B7280", "#B8A1E3", "#8B5CF6", "#5B21B6"])
    ax.set(xlabel="trajectory × output histories", title="C  Time since the last fast event")
    ax.title.set_fontsize(10)

    selected_contexts = [list(contexts)[0], list(contexts)[1]]
    for col, group in enumerate(("low SF", "high SF")):
        ax = fig.add_subplot(grid[1, col])
        color = LOW if group == "low SF" else HIGH
        for context, marker, ls in ((selected_contexts[0], "o", ":"), (selected_contexts[1], "s", "-")):
            sub = curves.loc[(curves.event_history_group == context) & (curves.sf_group == group)].sort_values("bin_index")
            ax.errorbar(
                sub.path_median_arcmin, sub.ssi_percent_vs_stabilized,
                yerr=np.vstack((sub.ssi_percent_vs_stabilized - sub.ci95_low, sub.ci95_high - sub.ssi_percent_vs_stabilized)),
                color=color, alpha=0.65 if ":" == ls else 1, marker=marker, ls=ls, capsize=2,
                label="historical drift label" if ":" == ls else "strict event-free history",
            )
        ax.axhline(0, color="black", lw=.7)
        ax.set(xlabel="path length (arcmin)", ylabel="SSI change vs stabilization (%)", title=f"{'D' if col == 0 else 'E'}  {group}: does strict drift retain the curve?")
        ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[1, 2])
    show_order = list(contexts)
    width = .34
    xpos = np.arange(len(show_order))
    for offset, (group, color) in zip((-.17, .17), (("low SF", LOW), ("high SF", HIGH))):
        sub = aggregates.set_index(["event_history_group", "sf_group"]).loc[[(c, group) for c in show_order]].reset_index()
        y = sub.ssi_percent_vs_stabilized.to_numpy(float)
        ax.bar(xpos + offset, y, width=width, color=color, label=group)
        ax.errorbar(xpos + offset, y, yerr=np.vstack((y - sub.ci95_low, sub.ci95_high - y)), fmt="none", ecolor="black", lw=.8, capsize=2)
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(xpos, ["historical\ndrift", "strict\ndrift", "prefix-only\nevent", "scored\nevent"], rotation=0)
    ax.set(ylabel="SSI change vs stabilization (%)", title="F  Aggregate corrected SSI by real history")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure C — The low-SF movement benefit survives genuinely event-free causal histories", fontsize=15, weight="bold")
    export(fig, "fig_C_temporal_history")

    strict_low = aggregates.loc[
        aggregates.event_history_group.str.startswith("strict") & (aggregates.sf_group == "low SF")
    ].iloc[0]
    strict_high = aggregates.loc[
        aggregates.event_history_group.str.startswith("strict") & (aggregates.sf_group == "high SF")
    ].iloc[0]
    stats = {
        "status": "first_pass_movie_aggregate_not_output_rescored",
        "event_definition": "sample-to-sample speed >= each trajectory's frozen rendered microsaccade threshold",
        "scored_event_class_matches_frozen_labels": True,
        "n_no_fast_event_full_71": int(strict_drift.sum()),
        "n_prefix_only_fast_event": int(prefix_only.sum()),
        "n_scored_fast_event": int(scored_fast.sum()),
        "historical_drift_traces_reclassified_due_to_prefix_event": int(np.count_nonzero((~cached_ms) & prefix_fast)),
        "strict_drift_low_sf_percent": float(strict_low.ssi_percent_vs_stabilized),
        "strict_drift_low_sf_ci95": [float(strict_low.ci95_low), float(strict_low.ci95_high)],
        "strict_drift_high_sf_percent": float(strict_high.ssi_percent_vs_stabilized),
        "strict_drift_high_sf_ci95": [float(strict_high.ci95_low), float(strict_high.ci95_high)],
        "limitation": "SSI is the completed 40-output movie aggregate; per-output SSI stratification still requires a time-resolved rescore",
    }
    write_json(FIRST_DIR / "event_history_statistics.json", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
