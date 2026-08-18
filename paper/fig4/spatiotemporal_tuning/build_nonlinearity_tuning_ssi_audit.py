#!/usr/bin/env python3
"""Relate temporal response harmonics, nonlinear core stages, and SSI gain.

For each grating condition, phase-modulation power is decomposed exactly as

    phase RMS^2 = F1 amplitude^2 / 2 + non-F1 harmonic power.

The decomposition distinguishes nonlinear temporal distortion from the later
spatial-information amplification measured by the causal motion replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/dekel240_paper/m77_epoch279"
PILOT = BASE / "figure4_real_trace_pilot_corrected"
DEFAULT_GROUPED = BASE / "periodic_tuning_respaced/frequency_tuning_grouped.csv"
DEFAULT_AUDIT = BASE / "periodic_tuning_respaced/fit_audit/m77_tuning_fit_audit.csv"
DEFAULT_MATRIX = PILOT / "merged"
DEFAULT_CORE = PILOT / "core_motion_path_full8x8/native_core_motion_path.npz"
DEFAULT_OUT = PILOT / "nonlinearity_tuning_ssi"
EPS = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--core-path", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--unit", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def surface(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sf = np.sort(frame.spatial_cpd.unique().astype(float))
    tf = np.sort(frame.temporal_hz.unique().astype(float))
    values = (
        frame.pivot(index="temporal_hz", columns="spatial_cpd", values=column)
        .reindex(index=tf, columns=sf)
        .to_numpy(dtype=float)
    )
    return sf, tf, values


def harmonic_components(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sf, tf, rms = surface(frame, "response_amp_rms")
    _, _, f1_amplitude = surface(frame, "f1_amplitude")
    total = np.square(rms)
    f1 = 0.5 * np.square(f1_amplitude)
    residual = np.maximum(total - f1, 0.0)
    return sf, tf, total, f1, residual


def population_harmonics(grouped: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dynamic = grouped.loc[grouped.temporal_hz.gt(0)]
    for unit_index, unit in dynamic.groupby("unit_index", sort=True):
        audit_row = audit.loc[audit.unit_index.eq(unit_index)].iloc[0]
        selected = unit.loc[
            np.isclose(unit.probe_orientation_deg, float(audit_row.rms_best_orientation_deg))
        ]
        _, _, total, f1, residual = harmonic_components(selected)
        preferred_sf = float(audit_row.rms_preferred_sf_cpd)
        preferred_tf = float(audit_row.rms_preferred_tf_hz)
        sf = np.sort(selected.spatial_cpd.unique().astype(float))
        tf = np.sort(selected.temporal_hz.unique().astype(float))
        near = (
            np.abs(np.log2(tf[:, None] / preferred_tf)) <= 0.75
        ) & (
            np.abs(np.log2(sf[None, :] / preferred_sf)) <= 0.75
        )
        discrete = np.unravel_index(int(np.argmax(total)), total.shape)
        rows.append(
            {
                "unit_index": int(unit_index),
                "audit_category": str(audit_row.audit_category),
                "rms_best_orientation_deg": float(audit_row.rms_best_orientation_deg),
                "rms_preferred_sf_cpd": preferred_sf,
                "rms_preferred_tf_hz": preferred_tf,
                "non_f1_fraction_surface": float(np.sum(residual) / max(float(np.sum(total)), EPS)),
                "non_f1_fraction_near_peak": float(
                    np.sum(residual[near]) / max(float(np.sum(total[near])), EPS)
                ),
                "non_f1_fraction_discrete_peak": float(
                    residual[discrete] / max(float(total[discrete]), EPS)
                ),
            }
        )
    return pd.DataFrame(rows)


def ssi_effect(matrix_dir: Path) -> tuple[pd.DataFrame, int, int]:
    n_units = len(pd.read_csv(matrix_dir / "unit_feature_table.csv"))
    n_images = len(pd.read_csv(matrix_dir / "image_feature_table.csv"))
    n_traces = len(pd.read_csv(matrix_dir / "trace_feature_table.csv"))
    moving_ssi = np.load(matrix_dir / "ssi_matrix.npy", mmap_mode="r").reshape(
        n_images, n_traces, n_units
    )
    moving_expected = np.load(
        matrix_dir / "expected_spikes_matrix.npy", mmap_mode="r"
    ).reshape(n_images, n_traces, n_units)
    stable_ssi = np.load(matrix_dir / "stabilized_ssi_by_image.npy", mmap_mode="r")
    stable_expected = np.load(
        matrix_dir / "stabilized_expected_spikes_by_image.npy", mmap_mode="r"
    )
    moving = np.sum(moving_ssi * moving_expected, axis=(0, 1), dtype=np.float64) / np.maximum(
        np.sum(moving_expected, axis=(0, 1), dtype=np.float64), EPS
    )
    stable = np.sum(stable_ssi * stable_expected, axis=0, dtype=np.float64) / np.maximum(
        np.sum(stable_expected, axis=0, dtype=np.float64), EPS
    )
    return (
        pd.DataFrame(
            {
                "unit_index": np.arange(n_units),
                "stable_ssi": stable,
                "ssi_change_percent": 100.0 * (moving - stable) / np.maximum(stable, EPS),
            }
        ),
        n_images,
        n_traces,
    )


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def heatmap(
    axis,
    sf: np.ndarray,
    tf: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    vmax: float,
    cmap: str,
) -> None:
    mesh = axis.pcolormesh(sf, tf, values, shading="nearest", cmap=cmap, norm=Normalize(0, vmax))
    axis.set(xscale="log", yscale="log")
    axis.set_xlim(sf.min(), sf.max())
    axis.set_ylim(tf.min(), tf.max())
    axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    axis.set_yticks([1, 2, 4, 8, 16, 32, 64, 90], ["1", "2", "4", "8", "16", "32", "64", "90"])
    axis.set_xlabel("spatial frequency (cycles/deg)")
    axis.set_ylabel("stimulus temporal frequency (Hz)")
    axis.set_title(title, loc="left", pad=6)
    return mesh


def render(args: argparse.Namespace) -> tuple[Path, dict]:
    configure()
    grouped = pd.read_csv(args.grouped_csv)
    audit = pd.read_csv(args.audit_csv)
    harmonics = population_harmonics(grouped, audit)
    effects, n_images, n_traces = ssi_effect(args.matrix_dir)
    population = harmonics.merge(effects, on="unit_index", validate="one_to_one")
    eligible = population.loc[
        population.stable_ssi.gt(1e-5)
        & np.isfinite(population.non_f1_fraction_surface)
        & np.isfinite(population.ssi_change_percent)
    ].copy()
    trusted = eligible.loc[eligible.audit_category.eq("trusted")].copy()
    rho, p_value = spearmanr(
        eligible.non_f1_fraction_surface, eligible.ssi_change_percent
    )
    trusted_rho, trusted_p = spearmanr(
        trusted.non_f1_fraction_surface, trusted.ssi_change_percent
    )

    unit = int(args.unit)
    audit_row = audit.loc[audit.unit_index.eq(unit)].iloc[0]
    selected = grouped.loc[
        grouped.unit_index.eq(unit)
        & grouped.temporal_hz.gt(0)
        & np.isclose(
            grouped.probe_orientation_deg, float(audit_row.rms_best_orientation_deg)
        )
    ]
    sf, tf, total, f1, residual = harmonic_components(selected)
    scale = max(float(np.max(total)), EPS)
    unit_row = population.loc[population.unit_index.eq(unit)].iloc[0]

    with np.load(args.core_path, allow_pickle=False) as archive:
        scales = np.asarray(archive["scales"], dtype=float)
        layer_percent = np.asarray(archive["layer_percent_vs_zero"], dtype=float)
        output_percent = np.asarray(archive["output_percent_vs_zero"], dtype=float)
    measured_index = int(np.argmin(np.abs(scales - 1.0)))
    stage_values = np.r_[layer_percent[measured_index], output_percent[measured_index]]

    figure = plt.figure(figsize=(13.3, 7.4), facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.985,
        bottom=0.105,
        top=0.88,
        wspace=0.34,
        hspace=0.43,
    )

    mesh = heatmap(
        figure.add_subplot(grid[0, 0]), sf, tf, total / scale,
        title=f"A  u{unit:03d}: total phase-modulation power", vmax=1.0, cmap="magma"
    )
    mesh = heatmap(
        figure.add_subplot(grid[0, 1]), sf, tf, f1 / scale,
        title="B  Power phase-locked to the stimulus F1", vmax=1.0, cmap="magma"
    )
    mesh = heatmap(
        figure.add_subplot(grid[0, 2]), sf, tf, residual / scale,
        title="C  Non-F1 power created by nonlinear harmonics", vmax=1.0, cmap="magma"
    )
    colorbar = figure.colorbar(mesh, ax=figure.axes[:3], fraction=0.018, pad=0.012)
    colorbar.set_label("fraction of u025's maximum total modulation power")

    axis = figure.add_subplot(grid[1, 0])
    profiles = [
        (np.sum(total, axis=1), "total RMS²", "#222222"),
        (np.sum(f1, axis=1), "F1² / 2", "#0072B2"),
        (np.sum(residual, axis=1), "non-F1 residual", "#D55E00"),
    ]
    profile_scale = max(float(np.max(profiles[0][0])), EPS)
    for values, label, color in profiles:
        axis.plot(tf, values / profile_scale, "o-", color=color, lw=1.6, ms=3.5, label=label)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32, 64, 90], ["1", "2", "4", "8", "16", "32", "64", "90"])
    axis.set_xlabel("stimulus temporal frequency (Hz)")
    axis.set_ylabel("SF-summed modulation power (normalized)")
    axis.set_title("D  u025: F1 and nonlinear harmonic power", loc="left", pad=6)
    axis.legend(frameon=False, fontsize=7)
    axis.grid(alpha=0.18)
    axis.text(
        0.98,
        0.95,
        f"non-F1 fraction over surface: {100 * unit_row.non_f1_fraction_surface:.0f}%\n"
        f"near preferred region: {100 * unit_row.non_f1_fraction_near_peak:.0f}%\n"
        f"at sampled maximum: {100 * unit_row.non_f1_fraction_discrete_peak:.0f}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )

    axis = figure.add_subplot(grid[1, 1])
    uncertain = eligible.loc[~eligible.audit_category.eq("trusted")]
    axis.scatter(
        100 * uncertain.non_f1_fraction_surface,
        uncertain.ssi_change_percent,
        s=13,
        color="#AFAFAF",
        alpha=0.38,
        linewidths=0,
        label="continuous peak uncertain",
        rasterized=True,
    )
    axis.scatter(
        100 * trusted.non_f1_fraction_surface,
        trusted.ssi_change_percent,
        s=24,
        color="#2E7D49",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        label="joint peak audit-trusted",
        rasterized=True,
    )
    axis.scatter(
        [100 * unit_row.non_f1_fraction_surface],
        [unit_row.ssi_change_percent],
        marker="D",
        s=52,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    axis.annotate(
        f"u{unit:03d}",
        (100 * unit_row.non_f1_fraction_surface, unit_row.ssi_change_percent),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7,
    )
    axis.axhline(0, color="#888888", lw=0.75)
    axis.set_xlabel("non-F1 fraction of grating modulation power (%)")
    axis.set_ylabel("SSI change: measured motion vs stabilized (%)")
    axis.set_title("E  Harmonic fraction does not predict SSI gain", loc="left", pad=6)
    axis.text(
        0.03,
        0.97,
        f"all {len(eligible)} raw surfaces: Spearman $\\rho$={rho:.2f}, p={p_value:.2f}\n"
        f"trusted-peak sensitivity n={len(trusted)}: $\\rho$={trusted_rho:.2f}, p={trusted_p:.2f}\n"
        f"{n_images} images × {n_traces} traces; no fitted peak enters x-axis",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    axis.legend(frameon=False, loc="lower left", fontsize=6.7)

    axis = figure.add_subplot(grid[1, 2])
    names = ["temporal\nstem", "spatial\nstage 1", "spatial\nstage 2", "spatial\nstage 3", "RR100\noutput"]
    colors = ["#999999", "#7AAE61", "#4E9D57", "#2E7D49", "#0072B2"]
    bars = axis.bar(np.arange(len(names)), stage_values, color=colors, width=0.72)
    axis.axhline(0, color="#777777", lw=0.75)
    axis.set_xticks(np.arange(len(names)), names)
    axis.set_ylabel("spatial-information change at measured motion (%)")
    axis.set_title("F  SSI gain emerges in later spatial stages", loc="left", pad=6)
    for bar, value in zip(bars, stage_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.2 if value >= 0 else -0.2),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.8,
        )
    axis.set_ylim(min(-1.2, float(stage_values.min()) - 0.7), float(stage_values.max()) + 1.3)

    figure.suptitle(
        "M77 nonlinearities have two distinct roles: temporal harmonics and spatial-information amplification",
        x=0.03,
        y=0.965,
        ha="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    figure.text(
        0.985,
        0.025,
        "Harmonic decomposition uses the same 32-phase periodic probe · stage audit uses exact moving-versus-stabilized core activations (8 images × 8 traces)",
        ha="right",
        fontsize=6.6,
        color="#555555",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "m77_nonlinearity_tuning_ssi.png"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.out_dir / f"m77_nonlinearity_tuning_ssi.{suffix}",
            dpi=260 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    population.to_csv(args.out_dir / "per_unit_harmonic_ssi.csv", index=False)
    report = {
        "analysis": "M77 temporal-harmonic and spatial-information nonlinearity audit",
        "definition": "total_phase_power=response_amp_rms^2; f1_power=f1_amplitude^2/2; non_f1=max(total-f1,0)",
        "unit": unit,
        "unit_non_f1_fraction_surface": float(unit_row.non_f1_fraction_surface),
        "unit_non_f1_fraction_near_peak": float(unit_row.non_f1_fraction_near_peak),
        "unit_non_f1_fraction_discrete_peak": float(unit_row.non_f1_fraction_discrete_peak),
        "raw_surface_population": {
            "n": int(len(eligible)),
            "spearman_rho_non_f1_fraction_vs_ssi_change": float(rho),
            "spearman_p": float(p_value),
            "n_images": n_images,
            "n_traces": n_traces,
        },
        "trusted_continuous_peak_sensitivity": {
            "n": int(len(trusted)),
            "spearman_rho_non_f1_fraction_vs_ssi_change": float(trusted_rho),
            "spearman_p": float(trusted_p),
        },
        "stage_percent_vs_stabilized_at_measured_motion": {
            name: float(value) for name, value in zip(names, stage_values)
        },
        "interpretation": (
            "Nonlinear harmonics contribute to the apparent temporal tuning, but their unit-level "
            "fraction does not explain SSI gain. Spatial-information amplification emerges after "
            "the later nonlinear spatial stages, so harmonic distortion and SSI amplification "
            "must not be treated as the same mechanism."
        ),
        "claim_boundary": (
            f"SSI correlations use the supplied {n_images}-image x {n_traces}-trace "
            "causal replay. The layerwise stage-localization values are inherited "
            "from the separate 8-image x 8-trace exact core replay and are therefore "
            "a mechanistic localization audit, not a production-scale effect estimate."
        ),
        "figure": str(output.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return output, report


def main() -> int:
    args = parse_args()
    output, report = render(args)
    print(output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
