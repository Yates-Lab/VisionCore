#!/usr/bin/env python3
"""Create correction-stage figures entirely from saved model outputs."""

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

from paper.fig4.mechanism_audit_v1.correction.common import DT_S, LEGACY_MATRIX_DIR, N_PRECEDING, OUT_DIR


LOW = "#2F6B9A"
HIGH = "#D97706"
BANK_STYLE = {
    "legacy_wrapped_prefix": {"label": "legacy wrapped", "color": "#C2413B", "ls": ":", "marker": "x"},
    "real_trace_true_history_v1": {"label": "true history", "color": "#167D75", "ls": "-", "marker": "o"},
    "real_trace_held_initial_history_v1": {"label": "held prefix", "color": "#6B7280", "ls": "--", "marker": "s"},
}


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None)


def plot_curve(ax: plt.Axes, data: pd.DataFrame, group: str, context: str) -> None:
    for bank, style in BANK_STYLE.items():
        sub = data[
            data["bank"].eq(bank) & data["sf_group"].eq(group) & data["context"].eq(context)
        ].sort_values("bin_index")
        x = sub["path_median_arcmin"].to_numpy(dtype=float)
        y = sub["ssi_percent_vs_stabilized"].to_numpy(dtype=float)
        lo = sub["ci95_low_paired_image_boot"].to_numpy(dtype=float)
        hi = sub["ci95_high_paired_image_boot"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - lo, hi - y)),
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            ms=4,
            lw=1.5,
            capsize=2,
            label=style["label"],
        )
    ax.axhline(0, color="#111827", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("trajectory path length (arcmin)")
    ax.set_ylabel("SSI change vs stabilization (%)")


def core_figure(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.5), sharey="row", constrained_layout=True)
    specs = [
        ("drift_only", "all_units", "Drift only — all units (n=100)"),
        ("drift_only", "low_sf_lt0p5", "Drift only — low SF (n=71)"),
        ("drift_only", "high_sf_ge0p5", "Drift only — high SF (n=29)"),
        ("microsaccade", "all_units", "Microsaccade — all units (n=100)"),
        ("microsaccade", "low_sf_lt0p5", "Microsaccade — low SF (n=71)"),
        ("microsaccade", "high_sf_ge0p5", "Microsaccade — high SF (n=29)"),
    ]
    for ax, (context, group, title) in zip(axes.flat, specs):
        plot_curve(ax, curves, group, context)
        color = LOW if "low" in group else HIGH if "high" in group else "#111827"
        ax.set_title(title, color=color, weight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Does the core Figure 4 movement-scale result survive causal history correction?", weight="bold")
    export(fig, "fig_core_result_corrected")
    plt.close(fig)


def choose_median_wrap(legacy: np.ndarray) -> int:
    speed = np.linalg.norm(legacy[:, 31] - legacy[:, 0], axis=1) / DT_S
    target = float(np.median(speed))
    return int(np.argmin(np.abs(speed - target)))


def bug_impact_figure(curves: pd.DataFrame, effects: pd.DataFrame, stats: dict) -> None:
    with np.load(OUT_DIR / "banks/corrected_history_trajectory_banks.npz") as archive:
        true_xy = np.asarray(archive["true_history_xy"])
        legacy = np.asarray(archive["stored_scored_trace_xy"])
    trace = pd.read_csv(LEGACY_MATRIX_DIR / "trace_feature_table.csv").sort_values("trace_bank_index").reset_index(drop=True)
    representative = choose_median_wrap(legacy)
    legacy_sequence = np.concatenate((legacy[representative, :32], legacy[representative]), axis=0)
    true_sequence = true_xy[representative]
    time_true = np.arange(-N_PRECEDING, 40) * DT_S * 1000.0
    time_legacy = np.arange(-32, 40) * DT_S * 1000.0
    time_scored = np.arange(40) * DT_S * 1000.0
    scored_x = legacy[representative, :, 0] * 60.0
    legacy_x = legacy_sequence[:, 0] * 60.0
    true_x = true_sequence[:, 0] * 60.0
    trace_span = max(
        float(np.ptp(scored_x)),
        float(np.ptp(legacy_x)),
        float(np.ptp(true_x)),
        1.0,
    )
    row_gap = trace_span + max(3.0, 0.25 * trace_span)
    legacy_plot_x = legacy_x - row_gap
    true_plot_x = true_x - 2.0 * row_gap
    plot_a = pd.DataFrame(
        {
            "scored_time_ms": np.pad(time_scored, (0, 32), constant_values=np.nan),
            "scored_x_arcmin": np.pad(scored_x, (0, 32), constant_values=np.nan),
            "true_time_ms": np.pad(time_true, (0, 1), constant_values=np.nan),
            "true_x_arcmin": np.pad(true_x, (0, 1), constant_values=np.nan),
            "true_plot_x_arcmin": np.pad(true_plot_x, (0, 1), constant_values=np.nan),
            "legacy_time_ms": time_legacy,
            "legacy_x_arcmin": legacy_x,
            "legacy_plot_x_arcmin": legacy_plot_x,
            "row_gap_arcmin": row_gap,
        }
    )
    plot_a.to_csv(OUT_DIR / "plot_data/fig_history_bug_explained_trace.csv", index=False)
    causal_rows = []
    for label, output_index, contaminated in (
        ("early", 0, 31),
        ("middle", 15, 16),
        ("late", 39, 0),
    ):
        causal_rows.append(
            {
                "window_label": label,
                "output_index": output_index,
                "output_time_ms": output_index * DT_S * 1000.0,
                "history_start_time_ms": (output_index - N_PRECEDING) * DT_S * 1000.0,
                "legacy_future_samples": contaminated,
            }
        )
    pd.DataFrame(causal_rows).to_csv(
        OUT_DIR / "plot_data/fig_history_bug_explained_causal_windows.csv", index=False
    )

    native_speed_by_trace = np.linalg.norm(np.diff(legacy, axis=1), axis=2) / DT_S
    wrap_speed = np.linalg.norm(legacy[:, 31] - legacy[:, 0], axis=1) / DT_S
    has_ms = pd.to_numeric(trace["rendered_n_microsaccade_events"], errors="coerce").fillna(0).to_numpy(dtype=int) > 0
    threshold = pd.to_numeric(trace["rendered_microsaccade_threshold_dps"], errors="coerce").to_numpy(dtype=float)
    dist = pd.concat(
        [
            pd.DataFrame(
                {
                    "distribution": "native adjacent steps — drift labels",
                    "speed_deg_s": native_speed_by_trace[~has_ms].ravel(),
                }
            ),
            pd.DataFrame(
                {
                    "distribution": "native adjacent steps — microsaccade labels",
                    "speed_deg_s": native_speed_by_trace[has_ms].ravel(),
                }
            ),
            pd.DataFrame({"distribution": "legacy wrap — drift labels", "speed_deg_s": wrap_speed[~has_ms]}),
            pd.DataFrame({"distribution": "legacy wrap — microsaccade labels", "speed_deg_s": wrap_speed[has_ms]}),
            pd.DataFrame({"distribution": "event threshold", "speed_deg_s": threshold}),
        ],
        ignore_index=True,
    )
    dist.to_csv(OUT_DIR / "plot_data/fig_history_bug_explained_speed_distributions.csv", index=False)

    fig = plt.figure(figsize=(11.2, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    ax = axes[0]
    ax.plot(time_scored, scored_x, color="#374151", lw=1.2)
    ax.plot(time_legacy, legacy_plot_x, color="#C2413B", lw=1.2)
    ax.plot(time_true, true_plot_x, color="#167D75", lw=1.2)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.annotate(
        "artificial e[31]→e[0] wrap",
        xy=(0, legacy_plot_x[32]),
        xytext=(-150, float(np.max(legacy_plot_x)) + 0.20 * row_gap),
        fontsize=8,
        color="#C2413B",
        arrowprops={"arrowstyle": "->", "color": "#C2413B"},
    )
    ax.text(time_scored[1], float(np.max(scored_x)) + 0.06 * row_gap, "scored e[0:40]", color="#374151", fontsize=8)
    ax.text(time_legacy[1], float(np.max(legacy_plot_x)) + 0.06 * row_gap, "legacy e[0:32] + e[0:40]", color="#C2413B", fontsize=8)
    ax.text(time_true[1], float(np.max(true_plot_x)) + 0.06 * row_gap, "true e[-31:40]", color="#167D75", fontsize=8)
    window_band_bottom = float(np.min(true_plot_x)) - 0.56 * row_gap
    for ordinal, (window_label, output_index, contaminated) in enumerate(
        (("early", 0, 31), ("middle", 15, 16), ("late", 39, 0))
    ):
        output_ms = output_index * DT_S * 1000.0
        history_start_ms = (output_index - N_PRECEDING) * DT_S * 1000.0
        window_y = window_band_bottom + ordinal * 0.17 * row_gap
        ax.annotate(
            "",
            xy=(output_ms, window_y),
            xytext=(history_start_ms, window_y),
            arrowprops={"arrowstyle": "|-|", "lw": 0.8, "color": "#111827"},
        )
        ax.text(
            0.5 * (history_start_ms + output_ms),
            window_y + 0.025 * row_gap,
            f"{window_label} output {output_index}: {contaminated} future",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_title("A  The legacy prefix injects future samples", loc="left", weight="bold")
    ax.set_xlabel("time from scored onset (ms)")
    ax.set_ylabel("horizontal position (row offsets)")
    ax.set_yticks([])
    ax.set_ylim(window_band_bottom - 0.04 * row_gap, float(np.max(scored_x)) + 0.18 * row_gap)

    ax = axes[1]
    bins = np.geomspace(max(0.05, np.nanmin(dist.speed_deg_s[dist.speed_deg_s > 0])), np.nanmax(dist.speed_deg_s), 45)
    for label, color, ls in (
        ("native adjacent steps — drift labels", "#374151", "-"),
        ("native adjacent steps — microsaccade labels", "#6B7280", "--"),
        ("legacy wrap — drift labels", "#C2413B", "-"),
        ("legacy wrap — microsaccade labels", "#D97706", "--"),
    ):
        values = dist.loc[dist.distribution.eq(label), "speed_deg_s"].to_numpy(dtype=float)
        hist, edges = np.histogram(values, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax.plot(centers, hist, color=color, ls=ls, lw=1.5, label=label)
    ax.axvline(np.nanmedian(threshold), color="#111827", lw=1, ls=":")
    ax.text(np.nanmedian(threshold) * 1.08, ax.get_ylim()[1] * 0.75, "median event threshold", fontsize=7)
    ax.set_xscale("log")
    ax.set_title("B  The artificial wrap can be a fast event", loc="left", weight="bold")
    ax.set_xlabel("implied speed (deg/s)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7)

    for ax, group, title in (
        (axes[2], "low_sf_lt0p5", "C  Low-SF SSI"),
        (axes[3], "high_sf_ge0p5", "D  High-SF SSI"),
    ):
        plot_curve(ax, curves, group, "drift_only")
        ax.set_title(title, loc="left", weight="bold", color=LOW if "Low" in title else HIGH)
    axes[2].legend(frameon=False, fontsize=7)

    ax = axes[4]
    for group, color, label in (("low_sf_lt0p5", LOW, "low SF"), ("high_sf_ge0p5", HIGH, "high SF")):
        sub = effects[effects["sf_group"].eq(group)]
        ax.scatter(
            sub["legacy_wrapped_prefix_benefit_percent"],
            sub["real_trace_true_history_v1_benefit_percent"],
            s=18,
            color=color,
            alpha=0.8,
            label=label,
        )
    limits = np.nanpercentile(
        np.concatenate(
            (
                effects["legacy_wrapped_prefix_benefit_percent"].to_numpy(),
                effects["real_trace_true_history_v1_benefit_percent"].to_numpy(),
            )
        ),
        [1, 99],
    )
    pad = 0.1 * (limits[1] - limits[0])
    ax.plot([limits[0] - pad, limits[1] + pad], [limits[0] - pad, limits[1] + pad], color="#111827", lw=0.8)
    ax.set_xlim(limits[0] - pad, limits[1] + pad)
    ax.set_ylim(limits[0] - pad, limits[1] + pad)
    ax.set_xlabel("legacy drift benefit (%)")
    ax.set_ylabel("true-history drift benefit (%)")
    ax.set_title("E  Per-unit correction impact", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[5]
    ax.axis("off")
    true = stats["true_history"]["overall_drift"]
    legacy_stats = stats["legacy"]["overall_drift"]
    held = stats["held_initial"]["overall_drift"]
    legacy_minus_true = stats["correction_impact"]["legacy_minus_true_drift"]["all_units"]
    true_minus_held = stats["correction_impact"]["true_minus_held_drift"]["all_units"]
    unit_summary = stats["per_unit_drift_effect_summary"]
    text = (
        f"{stats['decision']}\n\n"
        f"Overall drift benefit\n"
        f"legacy: {legacy_stats['ssi_percent_vs_stabilized']:.1f}%\n"
        f"true history: {true['ssi_percent_vs_stabilized']:.1f}% "
        f"[{true['ci95_low_paired_image_boot']:.1f}, {true['ci95_high_paired_image_boot']:.1f}]\n"
        f"held prefix: {held['ssi_percent_vs_stabilized']:.1f}%\n\n"
        f"legacy − true: {legacy_minus_true['point_percent_points']:.1f} pp "
        f"[{legacy_minus_true['ci95_low_percent_points']:.1f}, "
        f"{legacy_minus_true['ci95_high_percent_points']:.1f}]\n"
        f"true − held: {true_minus_held['point_percent_points']:.1f} pp "
        f"[{true_minus_held['ci95_low_percent_points']:.1f}, "
        f"{true_minus_held['ci95_high_percent_points']:.1f}]\n\n"
        f"Per-unit true benefit mean / median:\n"
        f"{unit_summary['real_trace_true_history_v1_benefit_percent_mean']:.1f}% / "
        f"{unit_summary['real_trace_true_history_v1_benefit_percent_median']:.1f}%\n\n"
        f"Low-SF progressive: {stats['true_history']['low_curve_spearman_rho'] > 0.8}\n"
        f"High-SF interior peak bin: {stats['true_history']['high_peak_bin_index']}"
    )
    ax.text(0.03, 0.97, text, va="top", fontsize=11, linespacing=1.4)
    ax.set_title("F  Correction summary", loc="left", weight="bold")
    for ax in axes[:5]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("How much did the wrapped-prefix bug change Figure 4?", fontsize=15, weight="bold")
    export(fig, "fig_history_bug_explained")
    plt.close(fig)
    pd.DataFrame([{"representative_trajectory_id": representative}]).to_csv(
        OUT_DIR / "plot_data/fig_history_bug_explained_example.csv", index=False
    )


def main() -> int:
    curves = pd.read_csv(OUT_DIR / "corrected_core_ssi_curves.csv")
    effects = pd.read_csv(OUT_DIR / "legacy_vs_corrected_unit_effects.csv")
    stats = json.loads((OUT_DIR / "statistics.json").read_text(encoding="utf-8"))
    core_figure(curves)
    bug_impact_figure(curves, effects, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
