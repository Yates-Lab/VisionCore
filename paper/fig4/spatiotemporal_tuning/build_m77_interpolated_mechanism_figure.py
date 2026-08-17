#!/usr/bin/env python3
"""Build an M77 retinal-motion mechanism figure with joint SF/TF analyses.

The panels keep four logically distinct quantities separate: an independently
measured output-unit tuning surface, renderer-faithful retinal spectral power,
the causal moving-versus-stabilized response, and the layer at which spatial
information is created.  When supplied, the population panel uses per-image
joint SF×TF×orientation engagement rather than a scalar preferred TF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/dekel240_paper/m77_epoch279"
DEFAULT_TUNING = BASE / "periodic_tuning_respaced/frequency_tuning_grouped.csv"
DEFAULT_ROBUST = BASE / "periodic_tuning_respaced/robust/robust_tuning_summary.csv"
DEFAULT_TUNING_AUDIT = (
    BASE / "periodic_tuning_respaced/fit_audit/m77_tuning_fit_audit.csv"
)
DEFAULT_PILOT = BASE / "figure4_real_trace_pilot_corrected"
DEFAULT_MATRIX = DEFAULT_PILOT / "merged"
DEFAULT_OCCUPANCY = (
    DEFAULT_PILOT
    / "image_specific_joint_engagement_phase_spectrum/image_specific_joint_engagement.npz"
)
DEFAULT_ACTIVATION = (
    DEFAULT_PILOT / "activation_gallery/native_motion_activation_gallery.npz"
)
DEFAULT_ACTIVATION_SUMMARY = (
    DEFAULT_PILOT / "activation_gallery/native_motion_activation_gallery_summary.json"
)
DEFAULT_CORE = (
    DEFAULT_PILOT / "core_motion_path_full8x8/native_core_motion_path.npz"
)
DEFAULT_CORE_SUMMARY = (
    DEFAULT_PILOT / "core_motion_path_full8x8/native_core_motion_path_summary.json"
)
DEFAULT_OUT = DEFAULT_PILOT / "m77_interpolated_mechanism"
EPS = 1e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=int, default=25)
    parser.add_argument("--tuning-table", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--robust-summary", type=Path, default=DEFAULT_ROBUST)
    parser.add_argument("--tuning-audit", type=Path, default=DEFAULT_TUNING_AUDIT)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--joint-engagement",
        type=Path,
        default=None,
        help=(
            "Optional unit_image_specific_joint_engagement.csv. When present, "
            "Panel D tests the full SFxTFxorientation predictor; otherwise it "
            "renders the older preferred-TF diagnostic."
        ),
    )
    parser.add_argument("--occupancy", type=Path, default=DEFAULT_OCCUPANCY)
    parser.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument(
        "--activation-summary", type=Path, default=DEFAULT_ACTIVATION_SUMMARY
    )
    parser.add_argument("--core-path", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--core-summary", type=Path, default=DEFAULT_CORE_SUMMARY)
    parser.add_argument(
        "--causal-audit-summary",
        type=Path,
        default=None,
        help=(
            "Optional production retinal-motion versus stabilized SSI audit. "
            "When supplied, Panel C reports its pooled image-bootstrap effect."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-label", default="M77 e279")
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fitted_surface(row: pd.Series, sf: np.ndarray, tf: np.ndarray) -> np.ndarray:
    """Evaluate the robust 2-D log-Gaussian fit as normalized signal amplitude."""
    xx, yy = np.meshgrid(np.log2(sf), np.log2(tf))
    mux = np.log2(float(row.preferred_sf_cpd))
    muy = np.log2(float(row.preferred_tf_hz))
    sx = float(row.sf_bandwidth_sigma_octaves)
    sy = float(row.tf_bandwidth_sigma_octaves)
    rho = float(row.sf_tf_log_correlation)
    dx, dy = (xx - mux) / sx, (yy - muy) / sy
    exponent = -0.5 * (dx * dx + dy * dy - 2.0 * rho * dx * dy) / max(
        1.0 - rho * rho, 1e-4
    )
    return np.exp(exponent)


def measured_surface(
    observed: pd.DataFrame,
    sf: np.ndarray,
    tf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-bilinearly interpolate the measured RMS surface without a shape model."""
    observed_sf = np.sort(observed.spatial_cpd.unique().astype(float))
    observed_tf = np.sort(observed.temporal_hz.unique().astype(float))
    response = (
        observed.pivot_table(
            index="temporal_hz",
            columns="spatial_cpd",
            values="response_amp_rms",
            aggfunc="mean",
        )
        .reindex(index=observed_tf, columns=observed_sf)
        .to_numpy(dtype=float)
    )
    signal = np.clip(response - float(np.nanmin(response)), 0.0, None)
    signal /= max(float(np.nanmax(signal)), EPS)
    interpolator = RegularGridInterpolator(
        (np.log2(observed_tf), np.log2(observed_sf)),
        signal,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    xx, yy = np.meshgrid(np.log2(sf), np.log2(tf))
    dense = interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(
        len(tf), len(sf)
    )
    return signal, dense, response


def load_tuning(
    table_path: Path, summary_path: Path, unit: int
) -> tuple[pd.Series, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    rows = summary.loc[summary.unit_index.eq(unit)]
    if len(rows) != 1:
        raise ValueError(f"expected one robust tuning row for unit {unit}, found {len(rows)}")
    row = rows.iloc[0]
    if not bool(row.fit_success):
        raise ValueError(f"unit {unit} lacks a successful robust tuning fit")
    grouped = pd.read_csv(table_path)
    observed = grouped.loc[
        grouped.unit_index.eq(unit)
        & grouped.temporal_hz.gt(0)
        & np.isclose(grouped.probe_orientation_deg, float(row.best_orientation_deg))
    ].copy()
    if observed.empty:
        raise ValueError(f"no observed tuning samples for unit {unit}")
    return row, observed


def population_effects(
    matrix_dir: Path, robust_path: Path, audit_path: Path
) -> pd.DataFrame:
    units = pd.read_csv(matrix_dir / "unit_feature_table.csv")
    n_units = len(units)
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
    moving_denominator = np.sum(moving_expected, axis=(0, 1), dtype=np.float64)
    stable_denominator = np.sum(stable_expected, axis=0, dtype=np.float64)
    moving = np.sum(
        moving_ssi * moving_expected, axis=(0, 1), dtype=np.float64
    ) / np.maximum(moving_denominator, EPS)
    stable = np.sum(
        stable_ssi * stable_expected, axis=0, dtype=np.float64
    ) / np.maximum(stable_denominator, EPS)
    metrics = pd.DataFrame(
        {
            "unit_index": np.arange(n_units),
            "stable_ssi": stable,
            "moving_ssi": moving,
            "ssi_change_percent": 100.0 * (moving - stable) / np.maximum(stable, EPS),
            "moving_expected_spikes": moving_denominator,
        }
    )
    tuning = pd.read_csv(robust_path)
    result = tuning.merge(metrics, on="unit_index", validate="one_to_one")
    audit_columns = [
        "unit_index",
        "audit_category",
        "rms_peak_status",
        "f1_peak_status",
        "rms_best_orientation_deg",
        "rms_preferred_sf_cpd",
        "rms_preferred_tf_hz",
        "rms_discrete_peak_sf_cpd",
        "rms_discrete_peak_tf_hz",
        "rms_local_peak_r2",
        "f1_preferred_sf_cpd",
        "f1_preferred_tf_hz",
        "f1_local_peak_r2",
        "rms_global_center_tf_hz",
        "rms_fit_r2",
        "rms_heldout_r2",
        "f1_fit_r2",
        "f1_heldout_r2",
    ]
    audit = pd.read_csv(audit_path)[audit_columns]
    result = result.merge(audit, on="unit_index", validate="one_to_one")
    result["tf_uncensored"] = (
        result.fit_success.astype(bool)
        & ~result.low_tf_censored.astype(bool)
        & ~result.high_tf_censored.astype(bool)
        & np.isfinite(result.preferred_tf_hz)
        & np.isfinite(result.ssi_change_percent)
        & result.stable_ssi.gt(1e-5)
    )
    result["well_fit"] = result.tf_uncensored & result.fit_r2.ge(0.35)
    result["audit_qualified"] = result.audit_category.eq("trusted")
    result.attrs.update(n_images=n_images, n_traces=n_traces, n_units=n_units)
    return result


def bootstrap_tf_quartiles(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame.audit_qualified].copy()
    selected["tf_quartile"] = pd.qcut(
        np.log2(selected.rms_preferred_tf_hz), 4, labels=False, duplicates="drop"
    )
    rng = np.random.default_rng(20260817)
    rows: list[dict[str, float | int]] = []
    for quartile, group in selected.groupby("tf_quartile", sort=True):
        values = group.ssi_change_percent.to_numpy(dtype=float)
        draws = np.empty(4000, dtype=float)
        for index in range(len(draws)):
            draws[index] = np.median(
                values[rng.integers(0, len(values), size=len(values))]
            )
        rows.append(
            {
                "tf_quartile": int(quartile),
                "n_units": len(values),
                "preferred_tf_hz": float(
                    2.0 ** np.median(np.log2(group.rms_preferred_tf_hz))
                ),
                "median_ssi_change_percent": float(np.median(values)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_joint_quartiles(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        np.isfinite(frame.joint_minus_separable_engagement_ssi_bits)
        & np.isfinite(frame.ssi_change_percent)
    ].copy()
    selected["engagement_quartile"] = pd.qcut(
        selected.joint_minus_separable_engagement_ssi_bits,
        4,
        labels=False,
        duplicates="drop",
    )
    rng = np.random.default_rng(20260817)
    rows: list[dict[str, float | int]] = []
    for quartile, group in selected.groupby("engagement_quartile", sort=True):
        values = group.ssi_change_percent.to_numpy(dtype=float)
        draws = np.median(
            values[rng.integers(0, len(values), size=(4000, len(values)))], axis=1
        )
        rows.append(
            {
                "engagement_quartile": int(quartile),
                "n_units": int(len(values)),
                "joint_minus_separable_engagement_ssi_bits": float(
                    np.median(group.joint_minus_separable_engagement_ssi_bits)
                ),
                "median_ssi_change_percent": float(np.median(values)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def attach_joint_engagement(
    population: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, dict]:
    population_attrs = dict(population.attrs)
    joint = pd.read_csv(path)
    # The renderer-faithful estimator uses clearer "alignment_selectivity"
    # names for the same image-specific quantities used by the earlier
    # trajectory-phase implementation.  Normalize the schema here so this
    # figure can compare estimators without silently mixing their values.
    direct_aliases = {
        "joint_alignment_selectivity_bits": "joint_engagement_ssi_bits",
        "joint_minus_separable_selectivity_bits": "joint_minus_separable_engagement_ssi_bits",
        "joint_alignment_vs_rate_delta_spearman": "joint_alignment_fraction_vs_rate_delta_spearman",
        "separable_alignment_vs_rate_delta_spearman": "separable_alignment_fraction_vs_rate_delta_spearman",
    }
    for source, destination in direct_aliases.items():
        if destination not in joint.columns and source in joint.columns:
            joint[destination] = joint[source]
    required = {
        "unit_index",
        "joint_engagement_ssi_bits",
        "joint_minus_separable_engagement_ssi_bits",
        "ssi_percent_vs_stabilized",
        "joint_peak_audit_trusted",
        "joint_alignment_fraction_vs_rate_delta_spearman",
        "separable_alignment_fraction_vs_rate_delta_spearman",
    }
    missing = required - set(joint.columns)
    if missing:
        raise ValueError(f"joint engagement table lacks {sorted(missing)}")
    columns = [
        "unit_index",
        "joint_engagement_ssi_bits",
        "joint_minus_separable_engagement_ssi_bits",
        "ssi_percent_vs_stabilized",
        "joint_peak_audit_trusted",
        "joint_alignment_fraction_vs_rate_delta_spearman",
        "separable_alignment_fraction_vs_rate_delta_spearman",
    ]
    optional_rate_columns = [
        column
        for column in (
            "tf_alignment_vs_rate_delta_spearman",
            "total_dynamic_power_vs_rate_delta_spearman",
            "joint_passband_power_vs_rate_delta_spearman",
        )
        if column in joint.columns
    ]
    columns.extend(optional_rate_columns)
    result = population.merge(joint[columns], on="unit_index", validate="one_to_one")
    result.attrs.update(population_attrs)
    disagreement = np.nanmax(
        np.abs(result.ssi_change_percent - result.ssi_percent_vs_stabilized)
    )
    if not np.isfinite(disagreement) or disagreement > 1e-4:
        raise ValueError(
            "joint-engagement and matrix SSI effects do not share the same causal bank "
            f"(maximum disagreement {disagreement:g})"
        )
    finite = np.isfinite(result.joint_minus_separable_engagement_ssi_bits) & np.isfinite(
        result.ssi_change_percent
    )
    all_test = spearmanr(
        result.loc[finite, "joint_minus_separable_engagement_ssi_bits"],
        result.loc[finite, "ssi_change_percent"],
    )
    trusted = finite & result.joint_peak_audit_trusted.astype(bool)
    trusted_test = (
        spearmanr(
            result.loc[trusted, "joint_minus_separable_engagement_ssi_bits"],
            result.loc[trusted, "ssi_change_percent"],
        )
        if int(trusted.sum()) >= 5
        else None
    )
    summary = {
        "all_units": {
            "n": int(finite.sum()),
            "rho": float(all_test.statistic),
            "p": float(all_test.pvalue),
        },
        "trusted_peak_sensitivity": (
            {
                "n": int(trusted.sum()),
                "rho": float(trusted_test.statistic),
                "p": float(trusted_test.pvalue),
            }
            if trusted_test is not None
            else None
        ),
        "same_image_rate_delta_median_spearman": {
            "joint": float(
                np.nanmedian(result.joint_alignment_fraction_vs_rate_delta_spearman)
            ),
            "separable_marginals": float(
                np.nanmedian(result.separable_alignment_fraction_vs_rate_delta_spearman)
            ),
            **(
                {
                    "tf_aligned_fraction": float(
                        np.nanmedian(result.tf_alignment_vs_rate_delta_spearman)
                    )
                }
                if "tf_alignment_vs_rate_delta_spearman" in result
                else {}
            ),
            **(
                {
                    "total_dynamic_power": float(
                        np.nanmedian(result.total_dynamic_power_vs_rate_delta_spearman)
                    )
                }
                if "total_dynamic_power_vs_rate_delta_spearman" in result
                else {}
            ),
        },
    }
    return result, summary


def load_activation(
    archive_path: Path, summary_path: Path, unit: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with np.load(archive_path, allow_pickle=False) as archive:
        units = archive["unit_indices"].astype(int)
        positions = np.flatnonzero(units == unit)
        if len(positions) != 1:
            raise ValueError(f"unit {unit} is absent from the activation gallery")
        position = int(positions[0])
        endpoint = int(archive["endpoint"])
        maps = np.asarray(archive["maps"], dtype=float)
        stable = maps[0, endpoint, position]
        moving = maps[1, endpoint, position]
    report = json.loads(summary_path.read_text())
    candidates = [
        item for item in report["unit_map_metrics"] if int(item["unit_index"]) == unit
    ]
    if len(candidates) != 1:
        raise ValueError(f"missing activation metrics for unit {unit}")
    return stable, moving, candidates[0]


def correlation(
    frame: pd.DataFrame, mask: str, x_column: str
) -> tuple[float, float, int]:
    selected = frame.loc[frame[mask]].copy()
    rho, p_value = spearmanr(
        np.log2(selected[x_column]), selected.ssi_change_percent
    )
    return float(rho), float(p_value), int(len(selected))


def render(args: argparse.Namespace) -> tuple[Path, dict]:
    configure_plotting()
    unit = int(args.unit)
    fit, observed = load_tuning(args.tuning_table, args.robust_summary, unit)
    population = population_effects(
        args.matrix_dir, args.robust_summary, args.tuning_audit
    )
    quartiles = bootstrap_tf_quartiles(population)
    joint_summary = None
    joint_quartiles = None
    if args.joint_engagement is not None:
        population, joint_summary = attach_joint_engagement(
            population, args.joint_engagement
        )
        joint_quartiles = bootstrap_joint_quartiles(population)
    exemplar = population.loc[population.unit_index.eq(unit)].iloc[0]
    stable_map, moving_map, map_metrics = load_activation(
        args.activation, args.activation_summary, unit
    )
    core_summary = json.loads(args.core_summary.read_text())
    causal_audit = None
    if args.causal_audit_summary is not None:
        causal_audit = json.loads(args.causal_audit_summary.read_text())
        if int(causal_audit["n_images"]) != int(population.attrs["n_images"]):
            raise ValueError("causal-audit image count does not match the matrix")
        if int(causal_audit["n_traces"]) != int(population.attrs["n_traces"]):
            raise ValueError("causal-audit trace count does not match the matrix")
    with np.load(args.core_path, allow_pickle=False) as archive:
        motion_scales = np.asarray(archive["scales"], dtype=float)
        layer_percent = np.asarray(archive["layer_percent_vs_zero"], dtype=float)
        output_percent = np.asarray(archive["output_percent_vs_zero"], dtype=float)
    with np.load(args.occupancy, allow_pickle=False) as archive:
        spectrum_method_version = str(archive["spectrum_method_version"].item())
        occupancy_sf = np.asarray(archive["spatial_cpd"], dtype=float)
        occupancy_tf = np.asarray(archive["temporal_hz"], dtype=float)
        orientations = np.asarray(archive["orientation_deg"], dtype=float)
        if "image_rendered_movie_power" in archive.files:
            spectrum_units = np.asarray(archive["unit_indices"], dtype=int)
            unit_row = int(np.flatnonzero(spectrum_units == unit)[0])
            joint_fraction = np.asarray(
                archive["joint_alignment_fraction"], dtype=float
            )
            spectrum_image_index = int(np.argmax(joint_fraction[:, unit_row]))
            occupancy = np.asarray(
                archive["image_rendered_movie_power"], dtype=float
            )[spectrum_image_index : spectrum_image_index + 1]
            occupancy_scales = np.asarray([1.0], dtype=float)
            spectrum_kind = "direct rendered retinal movie"
            spectrum_formula = (
                "spatial FFT of rendered frames, then two-DPSS temporal power"
            )
            spectrum_n_images = int(
                np.asarray(archive["image_rendered_movie_power"]).shape[0]
            )
            spectrum_n_traces = int(np.asarray(archive["trace_rows"]).size)
            # The absolute joint score retains dynamic-power magnitude while
            # weighting it by the measured SF×TF×orientation response.  Keep
            # it separate from the normalized alignment fraction, which asks
            # a different question and can fall when total drive grows.
            if {
                "joint_engagement",
                "moving_rate",
                "stabilized_rate",
            }.issubset(archive.files):
                raw_joint = np.asarray(archive["joint_engagement"], dtype=float)
                raw_rate_delta = np.asarray(archive["moving_rate"], dtype=float) - np.asarray(
                    archive["stabilized_rate"], dtype=float
                )
                raw_joint_correlation = np.asarray(
                    [
                        spearmanr(raw_joint[:, index], raw_rate_delta[:, index]).statistic
                        for index in range(raw_joint.shape[1])
                    ],
                    dtype=float,
                )
                correlation_by_unit = dict(zip(spectrum_units, raw_joint_correlation))
                population["joint_passband_power_vs_rate_delta_spearman"] = (
                    population.unit_index.map(correlation_by_unit)
                )
                if joint_summary is not None:
                    joint_summary["same_image_rate_delta_median_spearman"][
                        "raw_joint_passband_power"
                    ] = float(np.nanmedian(raw_joint_correlation))
        elif "example_trajectory_phase_power" in archive.files:
            occupancy = np.asarray(
                archive["example_trajectory_phase_power"], dtype=float
            )[None]
            occupancy_scales = np.asarray([1.0], dtype=float)
            spectrum_image_index = None
            spectrum_kind = "ideal trajectory-phase calculation"
            spectrum_formula = (
                "$|\\mathcal{F}_t\\{e^{-i2\\pi\\mathbf{k}\\cdot\\mathbf{X}(t)}\\}|^2$"
            )
            spectrum_n_images = int(population.attrs["n_images"])
            spectrum_n_traces = int(population.attrs["n_traces"])
        else:
            raise RuntimeError(
                "The supplied retinal spectrum contains neither direct rendered-movie "
                "power nor a complete trajectory-phase spectrum. Instantaneous |k.v| "
                "occupancy is not a valid temporal PSD and is no longer accepted."
            )

    best_orientation_index = int(
        np.argmin(np.abs(orientations - float(exemplar.rms_best_orientation_deg)))
    )
    measured_scale_index = int(np.argmin(np.abs(occupancy_scales - 1.0)))
    matched_occupancy = occupancy[
        measured_scale_index, :, :, best_orientation_index
    ].T
    sf_dense = np.geomspace(occupancy_sf.min(), occupancy_sf.max(), 240)
    tf_dense = np.geomspace(occupancy_tf.min(), occupancy_tf.max(), 240)
    _, tuning_dense, _ = measured_surface(
        observed, sf_dense, tf_dense
    )
    _, tuning_on_occupancy, _ = measured_surface(
        observed, occupancy_sf, occupancy_tf
    )
    halfmax_fraction = float(
        np.sum(matched_occupancy[tuning_on_occupancy >= 0.5])
        / max(float(np.sum(matched_occupancy)), EPS)
    )

    fig = plt.figure(figsize=(14.2, 7.9), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 1.02),
        height_ratios=(1.0, 0.92),
        left=0.055,
        right=0.985,
        bottom=0.105,
        top=0.89,
        wspace=0.31,
        hspace=0.43,
    )

    # A: measured samples and shape-preserving interpolation.  The marker is
    # the separately validated local joint peak, never a global-fit center.
    ax = fig.add_subplot(grid[0, 0])
    contour = ax.contourf(
        sf_dense,
        tf_dense,
        tuning_dense,
        levels=np.linspace(0, 1, 11),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    observed_signal = np.clip(
        observed.response_amp_rms.to_numpy(dtype=float)
        - float(observed.response_amp_rms.min()),
        0.0,
        None,
    )
    observed_signal /= max(float(np.max(observed_signal)), EPS)
    ax.scatter(
        observed.spatial_cpd,
        observed.temporal_hz,
        c=np.clip(observed_signal, 0, 1),
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=17,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    ax.scatter(
        [float(exemplar.rms_discrete_peak_sf_cpd)],
        [float(exemplar.rms_discrete_peak_tf_hz)],
        marker="x",
        s=42,
        color="white",
        linewidth=1.3,
        zorder=5,
        label=f"sampled max: {float(exemplar.rms_discrete_peak_tf_hz):.1f} Hz",
    )
    ax.scatter(
        [float(exemplar.rms_preferred_sf_cpd)],
        [float(exemplar.rms_preferred_tf_hz)],
        marker="*",
        s=105,
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label=f"local RMS peak: {float(exemplar.rms_preferred_tf_hz):.1f} Hz",
    )
    ax.set(xscale="log", yscale="log")
    ax.set_xlim(occupancy_sf.min(), occupancy_sf.max())
    ax.set_ylim(occupancy_tf.min(), occupancy_tf.max())
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 90], ["1", "2", "4", "8", "16", "32", "64", "90"])
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")
    ax.set_title(
        f"A  Audit-qualified u{unit:03d} peaks around "
        f"{float(exemplar.f1_preferred_tf_hz):.0f}–{float(exemplar.rms_preferred_tf_hz):.0f} Hz",
        loc="left",
        pad=6,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=6.7, handletextpad=0.4)
    ax.text(
        0.03,
        0.97,
        f"preferred SF {float(exemplar.rms_preferred_sf_cpd):.2f} c/deg\n"
        f"phase RMS {float(exemplar.rms_preferred_tf_hz):.1f} Hz; F1 "
        f"{float(exemplar.f1_preferred_tf_hz):.1f} Hz\n"
        f"local peak $R^2$: {float(exemplar.rms_local_peak_r2):.2f} / "
        f"{float(exemplar.f1_local_peak_r2):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=6.7,
        bbox={"facecolor": "#111111", "alpha": 0.62, "edgecolor": "none", "pad": 2.2},
    )
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("normalized measured response")
    colorbar.set_ticks([0, 0.5, 1])

    # B: renderer-faithful or complete-trajectory spectrum, never |k dot v|.
    ax = fig.add_subplot(grid[0, 1])
    positive = matched_occupancy[matched_occupancy > 0]
    floor = max(float(np.quantile(positive, 0.02)), float(np.max(positive)) * 1e-5)
    ceiling = float(np.quantile(positive, 0.995))
    power = ax.contourf(
        occupancy_sf,
        occupancy_tf,
        np.maximum(matched_occupancy, floor),
        levels=np.geomspace(floor, ceiling, 12),
        norm=LogNorm(vmin=floor, vmax=ceiling),
        cmap="magma",
        extend="both",
    )
    ax.contour(sf_dense, tf_dense, tuning_dense, levels=[0.5], colors="#222222", linewidths=2.8)
    ax.contour(sf_dense, tf_dense, tuning_dense, levels=[0.5], colors="white", linewidths=1.35)
    ax.scatter(
        [float(exemplar.rms_preferred_sf_cpd)],
        [float(exemplar.rms_preferred_tf_hz)],
        marker="*",
        s=90,
        color="#56B4E9",
        edgecolor="#222222",
        linewidth=0.65,
        zorder=4,
    )
    ax.set(xscale="log", yscale="log")
    ax.set_xlim(occupancy_sf.min(), occupancy_sf.max())
    ax.set_ylim(occupancy_tf.min(), occupancy_tf.max())
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 90], ["1", "2", "4", "8", "16", "32", "64", "90"])
    ax.set_xlabel("spatial frequency (cycles/deg)")
    ax.set_ylabel("temporal frequency (Hz)")
    ax.set_title(
        f"B  Retinal-movie power versus u{unit:03d}'s passband",
        loc="left",
        pad=6,
    )
    ax.text(
        0.03,
        0.97,
        f"{100 * halfmax_fraction:.0f}% of orientation-matched dynamic power\n"
        "inside measured half-max contour\n"
        f"{spectrum_formula}\n"
        f"{spectrum_n_images} images × {spectrum_n_traces} measured traces",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=6.7,
        bbox={"facecolor": "#111111", "alpha": 0.68, "edgecolor": "none", "pad": 2.2},
    )
    colorbar = fig.colorbar(power, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label(f"{spectrum_kind} power (log)")
    colorbar.set_ticks([])

    # C: actual output-map change for the same unit and endpoint.
    parent = fig.add_subplot(grid[0, 2])
    parent.set_axis_off()
    relative_map_change = 100.0 * float(map_metrics["delta_map_ssi_bits_per_spike"]) / max(
        float(map_metrics["stabilized_map_ssi_bits_per_spike"]), EPS
    )
    if causal_audit is None:
        panel_c_title = (
            f"C  Motion sharpens u{unit:03d}'s spatial response "
            f"(+{relative_map_change:.0f}% SSI)"
        )
        panel_c_note = "mean-normalized predicted rate; one matched image, trace, and endpoint"
    else:
        pooled = causal_audit["groups"]["all"]
        pooled_percent = float(pooled["percent_vs_stabilized"])
        pooled_ci = np.asarray(pooled["percent_ci95_image_bootstrap"], dtype=float)
        panel_c_title = (
            "C  Measured retinal motion raises pooled SSI "
            f"{pooled_percent:+.1f}% [{pooled_ci[0]:.1f}, {pooled_ci[1]:.1f}]"
        )
        panel_c_note = (
            f"u{unit:03d} example endpoint: {relative_map_change:+.0f}% SSI; "
            f"production replay: {int(causal_audit['n_images'])} images × "
            f"{int(causal_audit['n_traces']):,} traces"
        )
    parent.set_title(panel_c_title, loc="left", pad=6)
    map_grid = grid[0, 2].subgridspec(1, 3, wspace=0.07)
    stable_gain = stable_map / max(float(np.mean(stable_map)), EPS)
    moving_gain = moving_map / max(float(np.mean(moving_map)), EPS)
    delta_gain = moving_gain - stable_gain
    common_max = float(np.quantile(np.r_[stable_gain.ravel(), moving_gain.ravel()], 0.995))
    delta_limit = max(float(np.quantile(np.abs(delta_gain), 0.995)), 1e-8)
    map_axes = [fig.add_subplot(map_grid[0, index]) for index in range(3)]
    for axis, values, title, ssi in (
        (
            map_axes[0],
            stable_gain,
            "stabilized",
            float(map_metrics["stabilized_map_ssi_bits_per_spike"]),
        ),
        (
            map_axes[1],
            moving_gain,
            "measured motion",
            float(map_metrics["moving_map_ssi_bits_per_spike"]),
        ),
    ):
        axis.imshow(values, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=common_max)
        axis.text(
            0.04,
            0.96,
            f"{title}\nSSI {ssi:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=6.2,
            bbox={"facecolor": "#111111", "alpha": 0.68, "edgecolor": "none", "pad": 1.8},
        )
    map_axes[2].imshow(
        delta_gain,
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit),
    )
    map_axes[2].text(
        0.04,
        0.96,
        "motion − stable\nnormalized rate",
        transform=map_axes[2].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=6.2,
        bbox={"facecolor": "#111111", "alpha": 0.68, "edgecolor": "none", "pad": 1.8},
    )
    for axis in map_axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    parent.text(
        0.5,
        0.03,
        panel_c_note,
        transform=parent.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#555555",
    )

    # D/E: distinguish added dynamic drive from the separate question of SSI
    # sharpening. The first panel asks what predicts rate within a unit; the
    # second asks whether normalized spectral alignment explains ΔSSI.
    if joint_summary is not None and joint_quartiles is not None:
        mechanism_grid = grid[1, :2].subgridspec(
            1, 2, width_ratios=(0.86, 1.14), wspace=0.42
        )
        rate_axis = fig.add_subplot(mechanism_grid[0, 0])
        rate_specs = [
            (
                "total\ndynamic",
                "total_dynamic_power_vs_rate_delta_spearman",
                "#4C78A8",
            ),
            (
                "joint-weighted\ndynamic",
                "joint_passband_power_vs_rate_delta_spearman",
                "#2A9D8F",
            ),
            (
                "TF-aligned\nfraction",
                "tf_alignment_vs_rate_delta_spearman",
                "#59A14F",
            ),
            (
                "joint-aligned\nfraction",
                "joint_alignment_fraction_vs_rate_delta_spearman",
                "#E15759",
            ),
        ]
        rng = np.random.default_rng(20260817)
        rate_medians, rate_low, rate_high, rate_labels, rate_colors = [], [], [], [], []
        for label, column, color in rate_specs:
            if column not in population:
                continue
            values = population[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            draws = np.median(
                values[rng.integers(0, len(values), size=(4000, len(values)))],
                axis=1,
            )
            median = float(np.median(values))
            rate_labels.append(label)
            rate_colors.append(color)
            rate_medians.append(median)
            rate_low.append(median - float(np.quantile(draws, 0.025)))
            rate_high.append(float(np.quantile(draws, 0.975)) - median)
        positions = np.arange(len(rate_medians))
        rate_axis.bar(positions, rate_medians, color=rate_colors, width=0.68)
        rate_axis.errorbar(
            positions,
            rate_medians,
            yerr=np.vstack((rate_low, rate_high)),
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=2.5,
        )
        rate_axis.axhline(0, color="#777777", lw=0.75)
        rate_axis.set_xticks(positions, rate_labels)
        rate_axis.set_ylabel(
            "within-unit Spearman correlation\nwith moving − stabilized rate"
        )
        rate_axis.set_title(
            "D  Dynamic and passband power predict added drive", loc="left", pad=6
        )
        rate_axis.text(
            0.97,
            0.97,
            f"median across {len(population)} units\n95% unit-bootstrap intervals",
            transform=rate_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
        )
        for position, value in zip(positions, rate_medians):
            rate_axis.text(
                position,
                value + (0.012 if value >= 0 else -0.012),
                f"{value:+.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=6.7,
            )
        ax = fig.add_subplot(mechanism_grid[0, 1])
    else:
        ax = fig.add_subplot(grid[1, :2])
    rho_rms, p_rms, n_qualified = correlation(
        population, "audit_qualified", "rms_preferred_tf_hz"
    )
    rho_f1, p_f1, _ = correlation(
        population, "audit_qualified", "f1_preferred_tf_hz"
    )
    if joint_summary is not None and joint_quartiles is not None:
        trusted = population.joint_peak_audit_trusted.astype(bool)
        ax.scatter(
            population.loc[~trusted, "joint_minus_separable_engagement_ssi_bits"],
            population.loc[~trusted, "ssi_change_percent"],
            s=14,
            color="#D0D0D0",
            alpha=0.38,
            linewidths=0,
            label="peak uncertain",
            rasterized=True,
        )
        ax.scatter(
            population.loc[trusted, "joint_minus_separable_engagement_ssi_bits"],
            population.loc[trusted, "ssi_change_percent"],
            s=24,
            color="#2E7D49",
            alpha=0.68,
            edgecolor="white",
            linewidth=0.35,
            label="trusted peak",
            rasterized=True,
        )
        qx = joint_quartiles.joint_minus_separable_engagement_ssi_bits.to_numpy(dtype=float)
        qy = joint_quartiles.median_ssi_change_percent.to_numpy(dtype=float)
        qlow = qy - joint_quartiles.ci95_low.to_numpy(dtype=float)
        qhigh = joint_quartiles.ci95_high.to_numpy(dtype=float) - qy
        ax.plot(qx, qy, color="#0072B2", linewidth=1.4, zorder=3)
        ax.errorbar(
            qx,
            qy,
            yerr=np.vstack((qlow, qhigh)),
            fmt="o",
            color="#0072B2",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.2,
            label="quartile median ± 95% CI",
            zorder=4,
        )
        exemplar_x = float(exemplar.joint_minus_separable_engagement_ssi_bits)
        ax.scatter(
            [exemplar_x],
            [float(exemplar.ssi_change_percent)],
            marker="D",
            s=55,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"u{unit:03d}",
            (exemplar_x, float(exemplar.ssi_change_percent)),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
        )
        ax.set_xlabel("extra image selectivity from joint versus separable alignment (bits)")
        ax.set_title(
            "E  Passband alignment alone does not explain nonlinear ΔSSI",
            loc="left",
            pad=6,
        )
        trusted_joint = joint_summary["trusted_peak_sensitivity"]
        sensitivity = (
            f"trusted-peak sensitivity n={trusted_joint['n']}: "
            f"$\\rho$={trusted_joint['rho']:.2f}, p={trusted_joint['p']:.2f}"
            if trusted_joint is not None
            else "too few trusted continuous peaks for sensitivity test"
        )
        ax.text(
            0.02,
            0.97,
            f"all {joint_summary['all_units']['n']} raw tuning surfaces: "
            f"$\\rho$={joint_summary['all_units']['rho']:.2f}, "
            f"p={joint_summary['all_units']['p']:.2f}\n"
            f"{sensitivity}\n"
            "alignment divides out total dynamic power; joint score retains SF×TF×orientation coupling",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        ax.legend(loc="lower left", fontsize=6.5, frameon=True, ncol=3)
    else:
        excluded = population.loc[~population.audit_qualified]
        qualified = population.loc[population.audit_qualified]
        ax.scatter(
            excluded.rms_preferred_tf_hz,
            excluded.ssi_change_percent,
            s=13,
            color="#D5D5D5",
            alpha=0.32,
            linewidths=0,
            label="excluded by tuning audit",
            rasterized=True,
        )
        ax.scatter(
            qualified.rms_preferred_tf_hz,
            qualified.ssi_change_percent,
            s=23,
            color="#555555",
            alpha=0.55,
            linewidths=0,
            label="audit-qualified units",
            rasterized=True,
        )
        qx = quartiles.preferred_tf_hz.to_numpy(dtype=float)
        qy = quartiles.median_ssi_change_percent.to_numpy(dtype=float)
        qlow = qy - quartiles.ci95_low.to_numpy(dtype=float)
        qhigh = quartiles.ci95_high.to_numpy(dtype=float) - qy
        ax.errorbar(
            qx,
            qy,
            yerr=np.vstack((qlow, qhigh)),
            fmt="o-",
            color="#0072B2",
            markersize=5.5,
            capsize=2.5,
            linewidth=1.2,
        )
        ax.set_xscale("log")
        ax.set_xlabel("continuous phase-RMS peak TF (diagnostic only)")
        ax.set_title("D  Preferred TF alone is an incomplete diagnostic", loc="left", pad=6)
        ax.text(
            0.02,
            0.97,
            f"audit-qualified n={n_qualified}: phase RMS $\\rho$={rho_rms:.2f}, p={p_rms:.2f}\n"
            f"F1 sensitivity: $\\rho$={rho_f1:.2f}, p={p_f1:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
    ax.axhline(0, color="#888888", linewidth=0.75, zorder=0)
    ax.set_ylabel("SSI change: measured motion vs stabilized (%)")

    # F: exact activation audit through every nonlinear stage.
    ax = fig.add_subplot(grid[1, 2])
    measured_index = int(np.argmin(np.abs(motion_scales - 1.0)))
    stage_names = ["temporal\nstem", "spatial\nstage 1", "spatial\nstage 2", "spatial\nstage 3", "RR100\noutput"]
    measured_values = np.r_[layer_percent[measured_index], output_percent[measured_index]]
    colors = ["#999999", "#7AAE61", "#4E9D57", "#2E7D49", "#0072B2"]
    bars = ax.bar(np.arange(len(stage_names)), measured_values, color=colors, width=0.72)
    ax.axhline(0, color="#777777", linewidth=0.75)
    ax.set_xticks(np.arange(len(stage_names)), stage_names)
    ax.set_ylabel("spatial-information change at measured motion (%)")
    ax.set_title("F  SSI-like spatial modulation emerges downstream", loc="left", pad=6)
    for bar, value in zip(bars, measured_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.24 if value >= 0 else -0.24),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.8,
        )
    ax.set_ylim(min(-1.25, measured_values.min() - 0.8), measured_values.max() + 1.7)
    drive_fraction = float(core_summary["fraction_filter_drive_in_channels_above_10hz"])
    ax.text(
        0.03,
        0.97,
        f"Before these nonlinearities, {100 * drive_fraction:.0f}% of measured\n"
        "filter-drive variance is in stem filters ≥10 Hz.\n"
        "The stem itself does not yet show an SSI gain.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
    )

    fig.suptitle(
        "M77 mechanism: retinal motion adds dynamic drive; nonlinear stages create spatial sharpening",
        x=0.03,
        y=0.97,
        ha="left",
        fontsize=12.2,
        fontweight="semibold",
    )
    fig.text(
        0.985,
        0.022,
        f"{args.model_label}: native 240-Hz retinal input and output · peaks: local joint quadratic around sampled maximum · "
        "causal contrast: measured motion vs stabilization · nonlinearity audit: 8 images × 8 traces",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "m77_interpolated_mechanism.png"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.out_dir / f"m77_interpolated_mechanism.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    population.to_csv(args.out_dir / "per_unit_continuous_tf_ssi.csv", index=False)
    quartiles.to_csv(args.out_dir / "continuous_tf_quartile_summary.csv", index=False)
    if joint_quartiles is not None:
        joint_quartiles.to_csv(
            args.out_dir / "joint_engagement_quartile_summary.csv", index=False
        )
    report = {
        "analysis": "M77 interpolated unit-specific retinal-motion mechanism figure",
        "model_label": args.model_label,
        "exemplar_unit": unit,
        "continuous_tuning_fit": {
            "estimator": "local joint quadratic in log2 SF/TF over the sampled maximum's 3x3 neighborhood",
            "preferred_sf_cpd": float(exemplar.rms_preferred_sf_cpd),
            "preferred_tf_hz": float(exemplar.rms_preferred_tf_hz),
            "discrete_peak_sf_cpd": float(exemplar.rms_discrete_peak_sf_cpd),
            "discrete_peak_tf_hz": float(exemplar.rms_discrete_peak_tf_hz),
            "best_orientation_deg": float(exemplar.rms_best_orientation_deg),
            "local_peak_r2": float(exemplar.rms_local_peak_r2),
            "peak_status": str(exemplar.rms_peak_status),
            "old_global_center_tf_hz": float(exemplar.rms_global_center_tf_hz),
            "audit_category": str(exemplar.audit_category),
            "f1_preferred_tf_hz": float(exemplar.f1_preferred_tf_hz),
            "rms_heldout_r2": float(exemplar.rms_heldout_r2),
            "f1_heldout_r2": float(exemplar.f1_heldout_r2),
        },
        "retinal_spectrum": {
            "orientation_deg": float(orientations[best_orientation_index]),
            "halfmax_fraction": halfmax_fraction,
            "method_version": spectrum_method_version,
            "estimator": spectrum_kind,
            "image_index": spectrum_image_index,
            "n_images": spectrum_n_images,
            "n_traces": spectrum_n_traces,
            "temporal_frequency_definition": spectrum_formula,
        },
        "exemplar_map": {
            **map_metrics,
            "relative_ssi_change_percent": relative_map_change,
        },
        "population": {
            "n_images": int(population.attrs["n_images"]),
            "n_traces": int(population.attrs["n_traces"]),
            "tuning_audit": {
                "qualified_category": "trusted",
                "n_qualified": n_qualified,
                "n_excluded": int(len(population) - n_qualified),
            },
            "qualified_phase_rms_peak_tf": {
                "rho": rho_rms,
                "p": p_rms,
                "n": n_qualified,
            },
            "qualified_f1_peak_tf": {
                "rho": rho_f1,
                "p": p_f1,
                "n": n_qualified,
            },
            "joint_sfx_tfx_orientation_engagement": joint_summary,
            "exemplar_ssi_change_percent": float(exemplar.ssi_change_percent),
            "causal_retinal_motion_audit": (
                causal_audit["groups"]["all"] if causal_audit is not None else None
            ),
        },
        "nonlinearity": {
            "measured_motion_scale": float(motion_scales[measured_index]),
            "layer_percent_vs_stabilized": {
                name: float(value)
                for name, value in zip(stage_names, measured_values)
            },
            "fraction_filter_drive_in_channels_above_10hz": drive_fraction,
        },
        "sources": {
            str(path): sha256(path)
            for path in [
                args.tuning_table,
                args.robust_summary,
                args.tuning_audit,
                args.occupancy,
                args.activation,
                args.activation_summary,
                args.core_path,
                args.core_summary,
                *([args.joint_engagement] if args.joint_engagement is not None else []),
                *([args.causal_audit_summary] if args.causal_audit_summary is not None else []),
            ]
        },
        "claim_boundary": (
            "Joint passband alignment is a spectral-support predictor, not a "
            "frequency-band ablation. Its causal test uses the same-image moving-minus-"
            "stabilized replay and separates total dynamic power from its fraction aligned "
            "to tuning. Preferred SF/TF fits are visual summaries only and never enter the score."
        ),
        "figure": str(output.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return output, report


def main() -> int:
    args = parse_args()
    output, report = render(args)
    print(output)
    print(json.dumps(report["continuous_tuning_fit"], indent=2))
    print(json.dumps(report["population"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
