#!/usr/bin/env python3
"""Summarize native-240 controlled retinal-motion scaling with paired CIs.

The response archive contains the same images and trajectories at every motion
scale.  We therefore keep units and trajectories fixed and bootstrap paired
images only.  SSI is pooled with expected-spike weights before taking a percent
change from the matched zero-motion replay; averaging per-unit percentages
would answer a different question.  SF groups default to deterministic
tertiles of the cycle-valid response-weighted SF center, avoiding a fragile
split based on whether a fitted peak landed on the probe boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


def pooled_components(
    ssi: np.ndarray,
    expected: np.ndarray,
    unit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return expected-SSI numerator and expected denominator by image/scale."""
    ssi = np.asarray(ssi, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    units = np.asarray(unit_indices, dtype=int)
    if ssi.shape != expected.shape or ssi.ndim != 4:
        raise ValueError("ssi and expected must share [image, trace, scale, unit] shape")
    if units.ndim != 1 or units.size == 0:
        raise ValueError("unit_indices must be a nonempty one-dimensional array")
    if np.any(units < 0) or np.any(units >= ssi.shape[-1]):
        raise ValueError("unit index is outside the response archive")
    if not np.all(np.isfinite(ssi)) or not np.all(np.isfinite(expected)):
        raise ValueError("controlled-scaling archive contains non-finite values")
    if np.any(expected < 0):
        raise ValueError("expected-spike weights must be nonnegative")
    numerator = np.sum(ssi[..., units] * expected[..., units], axis=(1, 3))
    denominator = np.sum(expected[..., units], axis=(1, 3))
    return numerator, denominator


def percent_from_zero(
    numerator: np.ndarray,
    denominator: np.ndarray,
    image_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Pool selected images and return percent change from scale zero."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 2:
        raise ValueError("components must share [image, scale] shape")
    if image_rows is None:
        image_rows = np.arange(len(numerator), dtype=int)
    pooled = np.sum(numerator[image_rows], axis=0) / np.maximum(
        np.sum(denominator[image_rows], axis=0), EPS
    )
    return 100.0 * (pooled - pooled[0]) / max(float(pooled[0]), EPS)


def paired_image_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap paired images while preserving every within-image condition."""
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_bootstrap), numerator.shape[1]), dtype=np.float64)
    for draw in range(int(n_bootstrap)):
        rows = rng.integers(0, len(numerator), size=len(numerator))
        draws[draw] = percent_from_zero(numerator, denominator, rows)
    return np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def tuning_groups(
    path: Path,
    n_units: int,
    *,
    mode: str = "weighted_center_tertiles",
) -> dict[str, np.ndarray]:
    table = pd.read_csv(path)
    required = {"unit_index", "low_sf_censored", "weighted_center_sf_cpd"}
    if not required.issubset(table.columns):
        raise ValueError(f"robust tuning summary lacks {sorted(required - set(table.columns))}")
    unit = table.unit_index.to_numpy(dtype=int)
    low = table.low_sf_censored.astype("boolean").fillna(True).to_numpy(dtype=bool)
    if len(np.unique(unit)) != len(unit):
        raise ValueError("robust tuning summary contains duplicate units")
    if np.any(unit < 0) or np.any(unit >= int(n_units)):
        raise ValueError("robust tuning unit is outside the response archive")
    if mode == "censoring":
        return {
            "all active RR100": np.sort(unit),
            "low-SF censored": np.sort(unit[low]),
            "resolved SF peak": np.sort(unit[~low]),
        }
    if mode != "weighted_center_tertiles":
        raise ValueError(f"Unsupported SF grouping mode: {mode}")
    weighted_center = pd.to_numeric(
        table.weighted_center_sf_cpd, errors="coerce"
    ).to_numpy(float)
    if not np.all(np.isfinite(weighted_center) & (weighted_center > 0)):
        raise ValueError("robust tuning units require finite positive weighted-center SF")
    order = np.lexsort((unit, weighted_center))
    ordered_unit = unit[order]
    n_tail = len(ordered_unit) // 3
    if n_tail < 1:
        raise ValueError("Need at least three active units for SF tertiles")
    return {
        "all active RR100": np.sort(unit),
        "lower-SF tertile": np.sort(ordered_unit[:n_tail]),
        "middle-SF tertile": np.sort(ordered_unit[n_tail:-n_tail]),
        "higher-SF tertile": np.sort(ordered_unit[-n_tail:]),
    }


def render(table: pd.DataFrame, output: Path, model_label: str) -> None:
    styles = {
        "all active RR100": ("#4C4C4C", "o"),
        "low-SF censored": ("#4C78A8", "o"),
        "resolved SF peak": ("#E26A00", "s"),
        "lower-SF tertile": ("#7566CC", "o"),
        "middle-SF tertile": ("#7A7A7A", "s"),
        "higher-SF tertile": ("#2CA25F", "D"),
    }
    figure, axis = plt.subplots(figsize=(9.4, 4.6), constrained_layout=True)
    for name, subset in table.groupby("group", sort=False):
        subset = subset.sort_values("motion_scale")
        color, marker = styles[name]
        x = subset.motion_scale.to_numpy(float)
        y = subset.ssi_percent_vs_zero.to_numpy(float)
        low = subset.ci95_low.to_numpy(float)
        high = subset.ci95_high.to_numpy(float)
        axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        axis.plot(x, y, marker=marker, color=color, lw=2, ms=5, label=f"{name} (n={int(subset.n_units.iloc[0])})")
    axis.axhline(0, color="0.45", lw=1)
    axis.axvline(1, color="0.55", lw=1, linestyle=":")
    axis.text(1, axis.get_ylim()[1], " measured", color="0.35", va="top", ha="left")
    axis.set(
        xlabel="retinal trajectory amplitude (× measured)",
        ylabel="pooled SSI change from stabilized (%)",
        title=f"{model_label}: causal retinal-motion dose response",
        xticks=np.sort(table.motion_scale.unique()),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_npz", type=Path)
    parser.add_argument("--robust-tuning-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="native-240 twin")
    parser.add_argument(
        "--sf-group-mode",
        choices=("weighted_center_tertiles", "censoring"),
        default="weighted_center_tertiles",
        help=(
            "Use robust weighted-center SF tertiles (default) or the older "
            "boundary-censoring split."
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    with np.load(args.response_npz) as archive:
        ssi = np.asarray(archive["ssi"], dtype=np.float64)
        expected = np.asarray(archive["expected_spikes"], dtype=np.float64)
        scales = np.asarray(archive["scale_factors"], dtype=np.float64)
    if ssi.shape != expected.shape or ssi.ndim != 4 or ssi.shape[2] != len(scales):
        raise ValueError("controlled-scaling archive has inconsistent dimensions")
    if not np.isclose(scales[0], 0.0) or np.any(np.diff(scales) <= 0):
        raise ValueError("motion scales must increase strictly from zero")

    rows = []
    groups = tuning_groups(
        args.robust_tuning_summary,
        ssi.shape[-1],
        mode=args.sf_group_mode,
    )
    for group_index, (name, units) in enumerate(groups.items()):
        numerator, denominator = pooled_components(ssi, expected, units)
        point = percent_from_zero(numerator, denominator)
        low, high = paired_image_interval(
            numerator,
            denominator,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + group_index,
        )
        for index, scale in enumerate(scales):
            rows.append({
                "group": name,
                "motion_scale": float(scale),
                "ssi_percent_vs_zero": float(point[index]),
                "ci95_low": float(low[index]),
                "ci95_high": float(high[index]),
                "n_images": int(ssi.shape[0]),
                "n_traces": int(ssi.shape[1]),
                "n_units": int(len(units)),
            })
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "native_controlled_scaling_curves.csv"
    figure_path = args.out_dir / "native_controlled_scaling.png"
    table.to_csv(csv_path, index=False)
    render(table, figure_path, args.model_label)
    report = {
        "analysis": "native-240 paired controlled retinal-motion scaling",
        "response_npz": str(args.response_npz.resolve()),
        "robust_tuning_summary": str(args.robust_tuning_summary.resolve()),
        "sf_group_mode": str(args.sf_group_mode),
        "estimator": "expected-spike-weighted pooled SSI; paired image bootstrap; units and trajectories fixed",
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "n_images": int(ssi.shape[0]),
        "n_traces": int(ssi.shape[1]),
        "motion_scales": scales.tolist(),
        "csv": str(csv_path.resolve()),
        "figure": str(figure_path.resolve()),
    }
    (args.out_dir / "native_controlled_scaling_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
