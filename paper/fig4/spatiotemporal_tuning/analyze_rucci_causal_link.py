#!/usr/bin/env python3
"""Deprecated link from instantaneous-velocity occupancy to SSI changes.

The kinematic overlap is neither a retinal temporal PSD nor causal. This
analysis joins its measured-motion (1x) unit scores to the exact same selected
twin's moving-versus-stabilized replay and asks whether better passband
engagement identifies units with a larger SSI gain. It is retained only for
historical reproducibility; production uses complete trajectory-phase spectra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
GROUP_ORDER = ("low_sf", "middle_sf", "high_sf")
GROUP_LABEL = {
    "low_sf": "lower-SF tertile",
    "middle_sf": "middle-SF tertile",
    "high_sf": "higher-SF tertile",
}
GROUP_COLOR = {
    "low_sf": "#7566CC",
    "middle_sf": "#777777",
    "high_sf": "#2CA25F",
}


def _weighted_ssi(ssi: np.ndarray, expected: np.ndarray, axes) -> np.ndarray:
    numerator = np.sum(ssi * expected, axis=axes, dtype=np.float64)
    denominator = np.sum(expected, axis=axes, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def unit_causal_effects(matrix_dir: Path, overlap_csv: Path) -> pd.DataFrame:
    """Return one row per active unit on an exact shared unit coordinate."""
    matrix_dir = Path(matrix_dir)
    units = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    images = pd.read_csv(matrix_dir / "image_feature_table.csv")
    traces = pd.read_csv(matrix_dir / "trace_feature_table.csv")
    if units.unit_index.duplicated().any():
        raise ValueError("Matrix unit_feature_table contains duplicate unit_index rows")
    n_images, n_traces, n_units = len(images), len(traces), len(units)

    moving_ssi = np.load(matrix_dir / "ssi_matrix.npy")
    moving_expected = np.load(matrix_dir / "expected_spikes_matrix.npy")
    stable_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy")
    stable_expected = np.load(matrix_dir / "stabilized_expected_spikes_by_image.npy")
    expected_moving_shape = (n_images, n_traces, n_units)
    expected_stable_shape = (n_images, n_units)
    if moving_ssi.size != int(np.prod(expected_moving_shape)):
        raise ValueError(
            f"Moving SSI has {moving_ssi.size} values; expected "
            f"{int(np.prod(expected_moving_shape))}"
        )
    if moving_expected.size != moving_ssi.size:
        raise ValueError("Moving SSI and expected-spike matrices differ in size")
    if stable_ssi.shape != expected_stable_shape or stable_expected.shape != expected_stable_shape:
        raise ValueError(
            "Stabilized arrays do not match image/unit geometry: "
            f"{stable_ssi.shape}, {stable_expected.shape}, expected {expected_stable_shape}"
        )
    moving_ssi = moving_ssi.reshape(expected_moving_shape)
    moving_expected = moving_expected.reshape(expected_moving_shape)
    moving = _weighted_ssi(moving_ssi, moving_expected, axes=(0, 1))
    stable = _weighted_ssi(stable_ssi, stable_expected, axes=0)
    delta = moving - stable
    percent = np.divide(
        100.0 * delta,
        stable,
        out=np.full_like(delta, np.nan),
        where=stable > 0,
    )
    causal = pd.DataFrame(
        {
            "unit_index": units.unit_index.to_numpy(int),
            "moving_ssi_bits_per_spike": moving,
            "stabilized_ssi_bits_per_spike": stable,
            "ssi_delta_bits_per_spike": delta,
            "ssi_percent_vs_stabilized": percent,
        }
    )

    overlap = pd.read_csv(overlap_csv)
    overlap = overlap.loc[np.isclose(overlap.motion_scale.to_numpy(float), 1.0)].copy()
    if overlap.unit_index.duplicated().any():
        raise ValueError("The 1x overlap table contains duplicate unit rows")
    required = {
        "unit_index",
        "observed_passband_overlap",
        "weighted_center_sf_cpd",
        "sf_group",
    }
    missing = required - set(overlap.columns)
    if missing:
        raise ValueError(f"Overlap table lacks required columns: {sorted(missing)}")
    joined = overlap.merge(causal, on="unit_index", how="left", validate="one_to_one")
    if joined["moving_ssi_bits_per_spike"].isna().any():
        missing_units = joined.loc[
            joined["moving_ssi_bits_per_spike"].isna(), "unit_index"
        ].astype(int).tolist()
        raise ValueError(f"Overlap units are absent from the matrix: {missing_units[:8]}")
    engagement = pd.to_numeric(
        joined["observed_passband_overlap"], errors="coerce"
    ).to_numpy(float)
    finite_positive = engagement[np.isfinite(engagement) & (engagement > 0)]
    if not len(finite_positive):
        raise ValueError("No finite positive measured-motion passband overlaps")
    joined["passband_overlap_relative_to_population_median"] = (
        engagement / np.median(finite_positive)
    )
    return joined


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()


def spearman_with_resampling(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[finite]
    y = np.asarray(y, dtype=np.float64)[finite]
    if len(x) < 5:
        raise ValueError("Need at least five finite units for a rank-correlation audit")
    rx, ry = _rank(x), _rank(y)
    observed = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(n_resamples), dtype=np.float64)
    permutation = np.empty(int(n_resamples), dtype=np.float64)
    for draw in range(int(n_resamples)):
        rows = rng.integers(0, len(x), size=len(x))
        bootstrap[draw] = np.corrcoef(_rank(x[rows]), _rank(y[rows]))[0, 1]
        permutation[draw] = np.corrcoef(rx, ry[rng.permutation(len(y))])[0, 1]
    return {
        "n_units": int(len(x)),
        "spearman_rho": observed,
        "bootstrap_ci95": [
            float(value) for value in np.nanquantile(bootstrap, [0.025, 0.975])
        ],
        "two_sided_permutation_p": float(
            (1 + np.count_nonzero(np.abs(permutation) >= abs(observed)))
            / (len(permutation) + 1)
        ),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def group_summaries(table: pd.DataFrame, *, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    output = {}
    for group in GROUP_ORDER:
        values = table.loc[
            table.sf_group.astype(str).eq(group), "ssi_percent_vs_stabilized"
        ].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        draws = np.median(
            values[rng.integers(0, len(values), size=(int(n_bootstrap), len(values)))],
            axis=1,
        )
        output[group] = {
            "label": GROUP_LABEL[group],
            "n_units": int(len(values)),
            "median_percent": float(np.median(values)),
            "median_ci95": [
                float(value) for value in np.quantile(draws, [0.025, 0.975])
            ],
            "fraction_positive": float(np.mean(values > 0)),
        }
    return output


def render(table: pd.DataFrame, summary: dict, output: Path, *, model_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for group in GROUP_ORDER:
        rows = table.loc[table.sf_group.astype(str).eq(group)]
        if rows.empty:
            continue
        axes[0].scatter(
            rows.passband_overlap_relative_to_population_median,
            rows.ssi_percent_vs_stabilized,
            s=28,
            alpha=0.68,
            color=GROUP_COLOR[group],
            edgecolors="none",
            label=GROUP_LABEL[group],
        )
        axes[1].scatter(
            rows.weighted_center_sf_cpd,
            rows.ssi_percent_vs_stabilized,
            s=28,
            alpha=0.68,
            color=GROUP_COLOR[group],
            edgecolors="none",
        )
    axes[0].axhline(0.0, color="0.35", lw=0.9, ls=":")
    axes[0].set(
        title="Rucci prediction versus causal effect",
        xlabel="1x passband overlap / population median",
        ylabel="moving vs stabilized SSI (%)",
    )
    correlation = summary["passband_overlap_vs_ssi_percent"]
    axes[0].text(
        0.04,
        0.96,
        (
            f"Spearman ρ={correlation['spearman_rho']:.2f}\n"
            f"95% CI [{correlation['bootstrap_ci95'][0]:.2f}, "
            f"{correlation['bootstrap_ci95'][1]:.2f}]\n"
            f"permutation p={correlation['two_sided_permutation_p']:.3f}"
        ),
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].axhline(0.0, color="0.35", lw=0.9, ls=":")
    axes[1].set_xscale("log", base=2)
    axes[1].set(
        title="Causal effect across measured SF tuning",
        xlabel="cycle-valid weighted-center SF (cycles/deg)",
        ylabel="moving vs stabilized SSI (%)",
    )

    group_summary = summary["sf_tertiles"]
    plotted_groups = [group for group in GROUP_ORDER if group in group_summary]
    for x_position, group in enumerate(plotted_groups):
        values = table.loc[
            table.sf_group.astype(str).eq(group), "ssi_percent_vs_stabilized"
        ].to_numpy(float)
        values = values[np.isfinite(values)]
        offsets = np.linspace(-0.16, 0.16, len(values)) if len(values) else np.asarray([])
        axes[2].scatter(
            x_position + offsets,
            values,
            s=17,
            alpha=0.35,
            color=GROUP_COLOR[group],
            edgecolors="none",
        )
        row = group_summary[group]
        estimate = float(row["median_percent"])
        low_ci, high_ci = [float(value) for value in row["median_ci95"]]
        axes[2].errorbar(
            x_position,
            estimate,
            yerr=[
                [max(0.0, estimate - low_ci)],
                [max(0.0, high_ci - estimate)],
            ],
            fmt="D",
            ms=6,
            capsize=3,
            color=GROUP_COLOR[group],
            mec="white",
            mew=0.7,
            zorder=4,
        )
    axes[2].axhline(0.0, color="0.35", lw=0.9, ls=":")
    axes[2].set(
        title="Causal SSI effect by SF tertile",
        ylabel="moving vs stabilized SSI (%)",
        xticks=np.arange(len(plotted_groups)),
        xticklabels=[GROUP_LABEL[group] for group in plotted_groups],
    )
    axes[2].tick_params(axis="x", rotation=18)

    for axis in axes:
        axis.grid(alpha=0.16)
    fig.suptitle(
        f"{model_label}: does measured SF×TF engagement explain the SSI gain?",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(pad=0.8, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.08}
    fig.savefig(output, dpi=190, **options)
    fig.savefig(output.with_suffix(".pdf"), **options)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--overlap-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="selected twin")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = unit_causal_effects(args.matrix_dir, args.overlap_csv)
    summary = {
        "analysis": "deprecated instantaneous-velocity overlap versus causal SSI effect",
        "matrix_dir": str(args.matrix_dir.resolve()),
        "overlap_csv": str(args.overlap_csv.resolve()),
        "causal_contrast": "measured retinal motion minus exact stabilized replay; zero behavior in both",
        "passband_prediction": "natural-image Fourier power weighted by native framewise TF=abs(k dot v), dotted with each unit's observed normalized SFxTFxorientation tuning",
        "method_warning": "The input overlap is not based on a valid retinal temporal PSD and is retained only for historical reproducibility. Production uses the complete trajectory-phase spectrum.",
        "passband_overlap_vs_ssi_percent": spearman_with_resampling(
            table.passband_overlap_relative_to_population_median.to_numpy(float),
            table.ssi_percent_vs_stabilized.to_numpy(float),
            n_resamples=args.n_resamples,
            seed=args.seed,
        ),
        "passband_overlap_vs_ssi_delta": spearman_with_resampling(
            table.passband_overlap_relative_to_population_median.to_numpy(float),
            table.ssi_delta_bits_per_spike.to_numpy(float),
            n_resamples=args.n_resamples,
            seed=args.seed + 1,
        ),
        "weighted_center_sf_vs_ssi_percent": spearman_with_resampling(
            table.weighted_center_sf_cpd.to_numpy(float),
            table.ssi_percent_vs_stabilized.to_numpy(float),
            n_resamples=args.n_resamples,
            seed=args.seed + 2,
        ),
        "sf_tertiles": group_summaries(
            table,
            n_bootstrap=args.n_resamples,
            seed=args.seed + 3,
        ),
    }
    figure_path = args.out_dir / "rucci_passband_causal_ssi_link.png"
    summary["figure"] = str(figure_path.resolve())
    table.to_csv(args.out_dir / "unit_rucci_causal_effects.csv", index=False)
    render(table, summary, figure_path, model_label=str(args.model_label))
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
