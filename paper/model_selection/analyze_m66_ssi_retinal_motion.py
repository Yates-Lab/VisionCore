#!/usr/bin/env python3
"""Audit whether a selected twin's SSI changes under retinal eye motion.

The production Figure-4 scorer feeds zero behavior for both moving and
stabilized movies.  This script therefore compares the saved real-trace matrix
with its image-matched zero-motion baseline, using the same expected-spike
weighted population SSI estimand and an image bootstrap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "outputs/dekel240_paper/m66_final_snapshot/fig4_trace_bank_merged"
DEFAULT_OUT = ROOT / "outputs/dekel240_paper/m66_ssi_retinal_motion_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--n-path-bins", type=int, default=10)
    parser.add_argument(
        "--model-label",
        default="selected twin",
        help="Human-readable model name used in the audit figure.",
    )
    parser.add_argument(
        "--robust-tuning-summary",
        type=Path,
        default=None,
        help=(
            "Optional selected-twin periodic-tuning summary. When supplied, "
            "SF groups are low-edge-censored versus resolved observed peaks "
            "rather than historical labels."
        ),
    )
    parser.add_argument(
        "--sf-group-mode",
        choices=("auto", "table_tertiles", "censoring", "historical"),
        default="auto",
        help=(
            "SF grouping contract. 'auto' prefers selected-twin low_sf/high_sf "
            "tertiles in unit_feature_table.csv, then falls back to robust-fit "
            "edge censoring or the historical absolute split."
        ),
    )
    return parser.parse_args()


def population_ssi(ssi: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.sum(expected, dtype=np.float64))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(ssi * expected, dtype=np.float64) / denominator)


def pooled_effect(
    moving_ssi: np.ndarray,
    moving_expected: np.ndarray,
    stable_ssi: np.ndarray,
    stable_expected: np.ndarray,
    image_indices: np.ndarray,
    trace_indices: np.ndarray,
    unit_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    unit_indices = np.flatnonzero(unit_mask)
    m_ssi = moving_ssi[np.ix_(image_indices, trace_indices, unit_indices)]
    m_exp = moving_expected[np.ix_(image_indices, trace_indices, unit_indices)]
    s_ssi = stable_ssi[np.ix_(image_indices, unit_indices)]
    s_exp = stable_expected[np.ix_(image_indices, unit_indices)]
    moving = population_ssi(m_ssi, m_exp)
    stable = population_ssi(s_ssi, s_exp)
    delta = moving - stable
    percent = 100.0 * delta / stable if stable > 0 else float("nan")
    return moving, stable, delta, percent


def image_bootstrap(
    moving_ssi: np.ndarray,
    moving_expected: np.ndarray,
    stable_ssi: np.ndarray,
    stable_expected: np.ndarray,
    trace_indices: np.ndarray,
    unit_mask: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_images = moving_ssi.shape[0]
    unit_indices = np.flatnonzero(unit_mask)
    m_ssi = moving_ssi[np.ix_(np.arange(n_images), trace_indices, unit_indices)]
    m_exp = moving_expected[np.ix_(np.arange(n_images), trace_indices, unit_indices)]
    s_ssi = stable_ssi[np.ix_(np.arange(n_images), unit_indices)]
    s_exp = stable_expected[np.ix_(np.arange(n_images), unit_indices)]
    moving_numerator = np.sum(m_ssi * m_exp, axis=(1, 2), dtype=np.float64)
    moving_denominator = np.sum(m_exp, axis=(1, 2), dtype=np.float64)
    stable_numerator = np.sum(s_ssi * s_exp, axis=1, dtype=np.float64)
    stable_denominator = np.sum(s_exp, axis=1, dtype=np.float64)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        images = rng.integers(0, n_images, size=n_images)
        moving = np.sum(moving_numerator[images]) / np.sum(moving_denominator[images])
        stable = np.sum(stable_numerator[images]) / np.sum(stable_denominator[images])
        draws[draw] = 100.0 * (moving - stable) / stable
    return draws


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return float("nan")
    return float(np.corrcoef(ranks(x[finite]), ranks(y[finite]))[0, 1])


def build_sf_groups(
    units: pd.DataFrame,
    robust_tuning_summary: Path | None,
    *,
    mode: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Resolve low/high SF populations without transferring stale cutoffs."""
    n_units = len(units)
    labels = (
        units["sf_group"].astype(str).str.lower()
        if "sf_group" in units
        else pd.Series([""] * n_units)
    )


    has_tertiles = bool(labels.eq("low_sf").any() and labels.eq("high_sf").any())
    if mode == "auto":
        if has_tertiles:
            mode = "table_tertiles"
        elif robust_tuning_summary is not None:
            mode = "censoring"
        else:
            mode = "historical"

    if mode == "table_tertiles":
        if not has_tertiles:
            raise ValueError(
                "table_tertiles requires low_sf and high_sf labels in unit_feature_table.csv"
            )
        groups = {
            "all": np.ones(n_units, dtype=bool),
            "lower_sf": labels.eq("low_sf").to_numpy(bool),
            "higher_sf": labels.eq("high_sf").to_numpy(bool),
        }
        definition = {
            "mode": mode,
            "lower_sf": "selected-twin lower tertile of cycle-valid weighted-center SF",
            "higher_sf": "selected-twin upper tertile of cycle-valid weighted-center SF",
            "middle_sf_omitted_from_group_contrast": int(labels.eq("middle_sf").sum()),
            "inactive_units_omitted_from_group_contrast": int(labels.eq("inactive").sum()),
        }
        if robust_tuning_summary is not None:
            definition["robust_tuning_summary"] = str(
                robust_tuning_summary.resolve()
            )
        return groups, definition

    if mode == "censoring":
        if robust_tuning_summary is None:
            raise ValueError("censoring mode requires --robust-tuning-summary")
        tuning = pd.read_csv(robust_tuning_summary)
        if tuning.unit_index.duplicated().any():
            raise ValueError("Robust tuning summary contains duplicate unit_index rows")
        low_by_unit = tuning.set_index("unit_index").low_sf_censored.astype(bool)
        low = units.unit_index.map(low_by_unit).fillna(False).to_numpy(bool)
        tuned = units.unit_index.isin(low_by_unit.index).to_numpy(bool)
        return (
            {
                "all": np.ones(n_units, dtype=bool),
                "lower_sf": tuned & low,
                "higher_sf": tuned & ~low,
            },
            {
                "mode": mode,
                "lower_sf": "selected-twin SF peak censored at the lowest aperture-resolved SF",
                "higher_sf": "selected-twin observed SF peak resolved above the lowest grid value",
                "n_units_with_selected_twin_tuning": int(tuned.sum()),
                "robust_tuning_summary": str(robust_tuning_summary.resolve()),
            },
        )

    if mode != "historical":
        raise ValueError(f"Unknown SF group mode {mode!r}")
    sf = pd.to_numeric(units["sf_split_metric"], errors="coerce").to_numpy(dtype=float)
    return (
        {
            "all": np.ones(n_units, dtype=bool),
            "lower_sf": np.isfinite(sf) & (sf < 0.5),
            "higher_sf": np.isfinite(sf) & (sf >= 0.5),
        },
        {
            "mode": mode,
            "lower_sf": "historical sf_split_metric < 0.5",
            "higher_sf": "historical sf_split_metric >= 0.5",
        },
    )


def render_summary_figure(
    summary_groups: dict[str, dict[str, object]],
    image_table: pd.DataFrame,
    path_table: pd.DataFrame,
    *,
    output: Path,
    model_label: str,
) -> None:
    """Render the causal SSI result and its path-length dose response."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_order = [
        name
        for name in ("all", "lower_sf", "higher_sf")
        if name in summary_groups
    ]
    group_labels = {
        "all": "all active",
        "lower_sf": "lower-SF tertile",
        "higher_sf": "higher-SF tertile",
    }
    colors = {
        "all": "#4D4D4D",
        "lower_sf": "#2C7FB8",
        "higher_sf": "#E66101",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.15))

    all_images = image_table.loc[image_table.group.eq("all")]
    stable = all_images["stabilized_ssi_bits_per_spike"].to_numpy(float)
    moving = all_images["moving_ssi_bits_per_spike"].to_numpy(float)
    finite = np.isfinite(stable) & np.isfinite(moving)
    values = np.concatenate((stable[finite], moving[finite]))
    lo, hi = np.quantile(values, [0.01, 0.99]) if values.size else (0.0, 1.0)
    pad = max(0.05 * float(hi - lo), 1e-4)
    axes[0].plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad], color="0.55", lw=1
    )
    axes[0].scatter(
        stable[finite],
        moving[finite],
        s=22,
        alpha=0.55,
        color=colors["all"],
        edgecolors="none",
    )
    axes[0].set(
        title="Same-image causal replay",
        xlabel="stabilized SSI (bits/spike)",
        ylabel="measured-motion SSI (bits/spike)",
        xlim=(lo - pad, hi + pad),
        ylim=(lo - pad, hi + pad),
    )
    axes[0].set_aspect("equal", adjustable="box")
    fraction_positive = (
        float(np.mean(moving[finite] > stable[finite])) if finite.any() else float("nan")
    )
    axes[0].text(
        0.04,
        0.96,
        f"{fraction_positive:.0%} of images above identity",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )

    for x_position, group in enumerate(group_order):
        rows = image_table.loc[image_table.group.eq(group)]
        effects = rows["ssi_percent_vs_stabilized"].to_numpy(float)
        effects = effects[np.isfinite(effects)]
        offsets = (
            np.linspace(-0.16, 0.16, len(effects))
            if len(effects)
            else np.asarray([])
        )
        axes[1].scatter(
            x_position + offsets,
            effects,
            s=13,
            alpha=0.28,
            color=colors[group],
            edgecolors="none",
        )
        pooled = summary_groups[group]
        estimate = float(pooled["percent_vs_stabilized"])
        low_ci, high_ci = [
            float(value) for value in pooled["percent_ci95_image_bootstrap"]
        ]
        axes[1].errorbar(
            x_position,
            estimate,
            yerr=[
                [max(0.0, estimate - low_ci)],
                [max(0.0, high_ci - estimate)],
            ],
            fmt="D",
            ms=6,
            capsize=3,
            color=colors[group],
            mec="white",
            mew=0.7,
            zorder=4,
        )
    axes[1].axhline(0.0, color="0.35", lw=0.9, ls=":")
    axes[1].set(
        title="Pooled SSI increase",
        ylabel="moving vs stabilized SSI (%)",
        xticks=np.arange(len(group_order)),
        xticklabels=[group_labels[group] for group in group_order],
    )
    axes[1].tick_params(axis="x", rotation=18)

    for group in group_order:
        rows = path_table.loc[path_table.group.eq(group)].sort_values("path_bin")
        x_values = rows["path_median_arcmin"].to_numpy(float)
        y_values = rows["ssi_percent_vs_stabilized"].to_numpy(float)
        low_ci = rows["ci95_low"].to_numpy(float)
        high_ci = rows["ci95_high"].to_numpy(float)
        axes[2].plot(
            x_values,
            y_values,
            marker="o",
            ms=4,
            lw=1.6,
            color=colors[group],
            label=group_labels[group],
        )
        axes[2].fill_between(
            x_values,
            low_ci,
            high_ci,
            color=colors[group],
            alpha=0.12,
            linewidth=0,
        )
    axes[2].axhline(0.0, color="0.35", lw=0.9, ls=":")
    axes[2].set(
        title="Measured motion dose response",
        xlabel="retinal path length (arcmin)",
        ylabel="moving vs stabilized SSI (%)",
    )
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.16)
    fig.suptitle(
        f"{model_label}: eye motion causally increases spatial information",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(pad=0.8, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.08,
    }
    fig.savefig(output, dpi=190, **options)
    fig.savefig(output.with_suffix(".pdf"), **options)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    matrix_dir = args.matrix_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    units = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    traces = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    images = pd.read_csv(matrix_dir / "image_feature_table.csv")
    moving_ssi_flat = np.load(matrix_dir / "ssi_matrix.npy")
    moving_expected_flat = np.load(matrix_dir / "expected_spikes_matrix.npy")
    stable_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy")
    stable_expected = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy")
    with (matrix_dir / "stabilized_baseline_summary.json").open() as stream:
        baseline_summary = json.load(stream)

    n_images = len(images)
    n_traces = len(traces)
    n_units = len(units)
    moving_ssi = moving_ssi_flat.reshape(n_images, n_traces, n_units)
    moving_expected = moving_expected_flat.reshape(n_images, n_traces, n_units)
    all_images = np.arange(n_images, dtype=int)
    all_traces = np.arange(n_traces, dtype=int)

    groups, group_definition = build_sf_groups(
        units,
        args.robust_tuning_summary,
        mode=args.sf_group_mode,
    )

    summary_groups: dict[str, dict[str, float | int | list[float]]] = {}
    image_rows: list[dict[str, float | int | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []
    path_rows: list[dict[str, float | int | str | list[float]]] = []

    path = pd.to_numeric(traces["rendered_path_length_arcmin"], errors="coerce").to_numpy(dtype=float)
    rms = pd.to_numeric(traces["rendered_rms_radius_arcmin"], errors="coerce").to_numpy(dtype=float)
    microsaccades = pd.to_numeric(traces["rendered_n_microsaccade_events"], errors="coerce").to_numpy(dtype=float)
    quantile_edges = np.unique(np.nanquantile(path, np.linspace(0.0, 1.0, args.n_path_bins + 1)))
    bin_id = np.clip(np.digitize(path, quantile_edges[1:-1], right=True), 0, len(quantile_edges) - 2)

    for group_index, (group_name, unit_mask) in enumerate(groups.items()):
        moving, stable, delta, percent = pooled_effect(
            moving_ssi,
            moving_expected,
            stable_ssi,
            stable_expected,
            all_images,
            all_traces,
            unit_mask,
        )
        draws = image_bootstrap(
            moving_ssi,
            moving_expected,
            stable_ssi,
            stable_expected,
            all_traces,
            unit_mask,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + group_index,
        )

        per_image_percent = np.empty(n_images, dtype=float)
        for image_index in range(n_images):
            effect = pooled_effect(
                moving_ssi,
                moving_expected,
                stable_ssi,
                stable_expected,
                np.asarray([image_index]),
                all_traces,
                unit_mask,
            )
            per_image_percent[image_index] = effect[3]
            image_rows.append(
                {
                    "group": group_name,
                    "image_index": image_index,
                    "image_session": str(images.iloc[image_index]["session"]),
                    "moving_ssi_bits_per_spike": effect[0],
                    "stabilized_ssi_bits_per_spike": effect[1],
                    "delta_bits_per_spike": effect[2],
                    "ssi_percent_vs_stabilized": effect[3],
                }
            )

        per_trace_percent = np.empty(n_traces, dtype=float)
        for trace_index in range(n_traces):
            effect = pooled_effect(
                moving_ssi,
                moving_expected,
                stable_ssi,
                stable_expected,
                all_images,
                np.asarray([trace_index]),
                unit_mask,
            )
            per_trace_percent[trace_index] = effect[3]
            trace_rows.append(
                {
                    "group": group_name,
                    "trace_index": trace_index,
                    "trace_session": str(traces.iloc[trace_index]["session"]),
                    "path_length_arcmin": path[trace_index],
                    "rms_radius_arcmin": rms[trace_index],
                    "n_microsaccade_events": microsaccades[trace_index],
                    "ssi_percent_vs_stabilized": effect[3],
                }
            )

        summary_groups[group_name] = {
            "n_units": int(np.sum(unit_mask)),
            "moving_ssi_bits_per_spike": moving,
            "stabilized_ssi_bits_per_spike": stable,
            "delta_bits_per_spike": delta,
            "percent_vs_stabilized": percent,
            "percent_ci95_image_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
            "fraction_images_positive": float(np.mean(per_image_percent > 0.0)),
            "fraction_traces_positive": float(np.mean(per_trace_percent > 0.0)),
            "trace_spearman_path_length": spearman(path, per_trace_percent),
            "trace_spearman_rms_radius": spearman(rms, per_trace_percent),
            "mean_percent_no_microsaccade": float(np.mean(per_trace_percent[microsaccades == 0])),
            "mean_percent_with_microsaccade": float(np.mean(per_trace_percent[microsaccades > 0])),
        }

        for path_bin in range(len(quantile_edges) - 1):
            selected_traces = np.flatnonzero(bin_id == path_bin)
            effect = pooled_effect(
                moving_ssi,
                moving_expected,
                stable_ssi,
                stable_expected,
                all_images,
                selected_traces,
                unit_mask,
            )
            bin_draws = image_bootstrap(
                moving_ssi,
                moving_expected,
                stable_ssi,
                stable_expected,
                selected_traces,
                unit_mask,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + 100 + group_index * 20 + path_bin,
            )
            path_rows.append(
                {
                    "group": group_name,
                    "path_bin": path_bin + 1,
                    "n_traces": len(selected_traces),
                    "path_min_arcmin": float(np.min(path[selected_traces])),
                    "path_median_arcmin": float(np.median(path[selected_traces])),
                    "path_max_arcmin": float(np.max(path[selected_traces])),
                    "moving_ssi_bits_per_spike": effect[0],
                    "stabilized_ssi_bits_per_spike": effect[1],
                    "ssi_percent_vs_stabilized": effect[3],
                    "ci95_low": float(np.quantile(bin_draws, 0.025)),
                    "ci95_high": float(np.quantile(bin_draws, 0.975)),
                }
            )

    contract = baseline_summary["model_provenance"]["stimulus"]
    summary = {
        "analysis": "selected-twin retinal-motion versus stabilized SSI audit",
        "matrix_dir": str(matrix_dir),
        "n_images": n_images,
        "n_traces": n_traces,
        "n_movies": n_images * n_traces,
        "n_units": n_units,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_unit": "images; every draw recomputes pooled expected-spike-weighted SSI",
        "group_definition": group_definition,
        "causal_contract": {
            "moving_condition": "measured eye trace shifts a static natural image on the retinal input",
            "stabilized_condition": "same image with the eye trace set identically to zero",
            "behavior_input_both_conditions": "all zeros",
            "interpretation": "moving-minus-stabilized isolates retinal image motion in this replay",
            "model_time_contract": contract,
        },
        "groups": summary_groups,
    }
    path_table = pd.DataFrame(path_rows)
    image_table = pd.DataFrame(image_rows)
    trace_table = pd.DataFrame(trace_rows)
    path_table.to_csv(out_dir / "path_dose_curves.csv", index=False)
    image_table.to_csv(out_dir / "per_image_effects.csv", index=False)
    trace_table.to_csv(out_dir / "per_trace_effects.csv", index=False)
    figure_path = out_dir / "causal_ssi_retinal_motion.png"
    render_summary_figure(
        summary_groups,
        image_table,
        path_table,
        output=figure_path,
        model_label=str(args.model_label),
    )
    summary["figure"] = str(figure_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
