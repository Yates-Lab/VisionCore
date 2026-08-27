#!/usr/bin/env python3
"""Audit exact-CID drifting-grating measurements before Figure 4 can use them.

The audit keeps the raw low-resolution SF x TF measurements visible and draws
a half-maximum lasso from a Yu R0/R1 fit only when that fit follows the raw
surface.  A unit is not released for Figure 4 unless phase quadrature and a
second-contrast replay converge, the raw peak is interior and coherent, and
the exact biological unit has reliable recorded static-grating SF tuning.  The
recorded assay is never mislabeled as a biological TF measurement: it used
brief randomized gratings and supplies SF, orientation, phase, and latency,
whereas TF comes only from the controlled matched-twin replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label, maximum_filter


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning._figure4_rendering import (
    fit_probability,
    fit_yu_surface,
    yu_surface,
)


EXPECTED_COUNT_METRICS = {
    "blank_expected_count",
    "f0_expected_count",
    "delta_f0_expected_count",
    "half_phase_f0_expected_count",
    "half_phase_evoked_rms_expected_count",
    "evoked_rms_expected_count",
    "phase_modulation_rms_expected_count",
    "f1_expected_count_amplitude",
    "f2_expected_count_amplitude",
    "minimum_expected_count",
    "maximum_expected_count",
}

ATLAS_UNITS_PER_PAGE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_dir", type=Path)
    parser.add_argument("--recorded-static-dir", type=Path, required=True)
    parser.add_argument(
        "--comparison-measurement-dir",
        type=Path,
        default=None,
        help=(
            "Independent repeat at a second contrast. It must use the same exact "
            "units and physical grid; Figure 4 release remains blocked without it."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return np.nan
    x, y = x[keep], y[keep]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _octave_error(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or left <= 0 or right <= 0:
        return np.nan
    return float(abs(np.log2(left / right)))


def _circular_error_deg(left: float, right: float) -> float:
    return float(abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0))


def load_measurement(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    provenance_path = path / "provenance.json"
    units_path = path / "units.csv"
    conditions_path = path / "conditions.csv"
    responses_path = path / "responses.npz"
    for required in (provenance_path, units_path, conditions_path, responses_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not bool(provenance.get("complete_grid", False)):
        raise ValueError("the exact-CID tuning grid is incomplete")
    if "not a recorded biological" not in str(provenance.get("source_kind", "")):
        raise ValueError("provenance does not distinguish model replay from biological TF")
    if "no reconstructed population head" not in str(
        provenance.get("readout_identity_audit", {}).get("readout_path", "")
    ):
        raise ValueError("tuning did not use checkpoint-native exact-CID readouts")
    units = pd.read_csv(units_path)
    conditions = pd.read_csv(conditions_path)
    if units.duplicated(["session", "cid"]).any():
        raise ValueError("unit table contains duplicate biological identities")
    if units.canonical_channel.duplicated().any():
        raise ValueError("unit table contains duplicate canonical channels")
    archive = dict(np.load(responses_path))
    missing = sorted(EXPECTED_COUNT_METRICS - set(archive))
    if missing:
        raise ValueError(f"response archive lacks {missing}")
    expected = (len(conditions), len(units))
    for key, value in archive.items():
        target = (len(units),) if key == "blank_expected_count" else expected
        if value.shape != target or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid {key} array: {value.shape}, expected {target}")
    return provenance, units, conditions, archive


def response_cube(
    conditions: pd.DataFrame, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sf = np.sort(conditions.spatial_cpd.unique().astype(float))
    tf = np.sort(conditions.temporal_hz.unique().astype(float))
    direction = np.sort(conditions.motion_direction_deg.unique().astype(float))
    cube = np.full((len(sf), len(tf), len(direction), values.shape[1]), np.nan)
    sf_index = {value: index for index, value in enumerate(sf)}
    tf_index = {value: index for index, value in enumerate(tf)}
    direction_index = {value: index for index, value in enumerate(direction)}
    for row, condition in enumerate(conditions.itertuples(index=False)):
        index = (
            sf_index[float(condition.spatial_cpd)],
            tf_index[float(condition.temporal_hz)],
            direction_index[float(condition.motion_direction_deg)],
        )
        if np.any(np.isfinite(cube[index])):
            raise ValueError(f"duplicate tuning condition {index}")
        cube[index] = values[row]
    if not np.all(np.isfinite(cube)):
        raise ValueError("condition table does not form a complete SF x TF x direction cube")
    return sf, tf, direction, cube


def preferred_direction(surface: np.ndarray, dynamic_tf: np.ndarray) -> int:
    """Select one direction by its mean of the six strongest dynamic cells."""
    dynamic = np.maximum(surface[:, dynamic_tf, :], 0.0)
    flattened = dynamic.reshape(-1, dynamic.shape[-1])
    count = min(6, len(flattened))
    scores = np.mean(np.sort(flattened, axis=0)[-count:], axis=0)
    return int(np.argmax(scores))


def connected_peak_lasso(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed display rule: smoothed half-max component containing the raw peak."""
    values = np.maximum(np.asarray(surface, dtype=float), 0.0)
    smoothed = gaussian_filter(values, sigma=0.5, mode="nearest")
    peak = np.unravel_index(int(np.argmax(values)), values.shape)
    if smoothed[peak] <= 0:
        return smoothed, np.zeros_like(smoothed, dtype=bool)
    components, _ = label(smoothed >= 0.5 * smoothed[peak])
    component = int(components[peak])
    return smoothed, components == component


def recorded_static_curves(path: Path) -> tuple[pd.DataFrame, dict[int, dict]]:
    metrics_path = path / "unit_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    curves: dict[int, dict] = {}
    for archive_path in sorted((path / "sessions").glob("*_curves.npz")):
        archive = np.load(archive_path)
        channels = archive["canonical_channels"].astype(int)
        for row, channel in enumerate(channels):
            if int(channel) in curves:
                raise ValueError(f"duplicate recorded curve for channel {channel}")
            curves[int(channel)] = {
                "sf": archive["sfs"].astype(float),
                "data_sf": archive["data_sf"][row].astype(float),
                "model_sf": archive["model_sf"][row].astype(float),
            }
    return metrics, curves


def recorded_sf_anchor(
    unit: pd.Series,
    metrics: pd.DataFrame,
    curves: dict[int, dict],
    sf: np.ndarray,
    directions: np.ndarray,
    dynamic_cube: np.ndarray,
) -> dict:
    channel = int(unit.canonical_channel)
    match = metrics.loc[metrics.canonical_channel.eq(channel)]
    curve = curves.get(channel)
    if len(match) != 1 or curve is None:
        return {
            "recorded_sf_anchor_available": False,
            "recorded_sf_curve_corr": np.nan,
            "recorded_sf_peak_error_octaves": np.nan,
            "recorded_sf_data_reliable": False,
            "recorded_sf_twin_match": False,
        }
    row = match.iloc[0]
    preferred_bar = float(row.model_preferred_ori_deg) % 180.0
    bar = (directions + 90.0) % 180.0
    axial_error = np.abs(((bar - preferred_bar + 90.0) % 180.0) - 90.0)
    selected = np.flatnonzero(np.isclose(axial_error, axial_error.min()))
    # Match the recorded preferred axial orientation, average the two opposite
    # motion directions, and retain the strongest dynamic TF at each SF.  This
    # yields a spatial envelope from the motion assay without pretending that
    # its steady TF=0 history reproduces a brief randomized-grating onset.
    oriented = np.take(dynamic_cube, selected, axis=2).mean(axis=2)
    synthetic = np.maximum(oriented, 0.0).max(axis=1)
    recorded_sf = np.asarray(curve["sf"], dtype=float)
    recorded = np.asarray(curve["model_sf"], dtype=float)
    interpolated = np.interp(
        np.log2(recorded_sf), np.log2(sf), synthetic, left=np.nan, right=np.nan
    )
    return {
        "recorded_sf_anchor_available": True,
        "recorded_sf_curve_corr": _correlation(recorded, interpolated),
        "recorded_sf_peak_error_octaves": _octave_error(
            float(row.model_preferred_sf_cpd), float(sf[int(np.argmax(synthetic))])
        ),
        "recorded_model_preferred_sf_cpd": float(row.model_preferred_sf_cpd),
        "recorded_data_preferred_sf_cpd": float(row.data_preferred_sf_cpd),
        "recorded_n_spikes": float(row.n_spikes),
        "recorded_data_sf_boundary": bool(row.data_sf_boundary),
        "recorded_model_sf_boundary": bool(row.model_sf_boundary),
        "recorded_sf_split_half": float(row.sf_split_half),
        "recorded_model_data_sf_curve_corr": float(row.sf_curve_corr),
        "recorded_model_data_sf_peak_error_octaves": float(row.sf_error_octaves),
        "recorded_sf_data_reliable": bool(
            row.n_spikes >= 50
            and row.sf_split_half >= 0.2
            and not bool(row.data_sf_boundary)
        ),
        "recorded_sf_twin_match": bool(
            row.sf_curve_corr >= 0.5 and row.sf_error_octaves <= 1.0
        ),
        "synthetic_motion_envelope_preferred_sf_cpd": float(
            sf[int(np.argmax(synthetic))]
        ),
    }


def fit_yu_passband(
    sf: np.ndarray, dynamic_tf: np.ndarray, surface: np.ndarray
) -> dict:
    """Fit Yu R0/R1 to every acquired dynamic cell with explicit blank subtraction."""
    sf_grid, tf_grid = np.meshgrid(sf, dynamic_tf, indexing="xy")
    try:
        r0 = fit_yu_surface(
            np.log2(sf_grid), np.log2(tf_grid), surface, inseparable=False
        )
        normalized_r0 = r0.parameters.copy()
        normalized_r0[0] /= max(float(np.max(surface)), 1e-10)
        r1 = fit_yu_surface(
            np.log2(sf_grid),
            np.log2(tf_grid),
            surface,
            inseparable=True,
            initial_from_r0=normalized_r0,
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {
            "fit_success": False,
            "fit_error": f"{type(error).__name__}: {error}",
        }
    probability_r1 = fit_probability(r0.bic_dense, r1.bic_dense)
    selected = r1 if probability_r1 > 0.5 else r0
    parameters = selected.parameters
    fitted_sf = float(2.0 ** parameters[1])
    fitted_tf = float(2.0 ** parameters[4])
    raw_tf_index, raw_sf_index = np.unravel_index(
        int(np.argmax(surface)), surface.shape
    )
    prediction = yu_surface(
        parameters,
        np.log2(sf_grid),
        np.log2(tf_grid),
        inseparable=selected.model == "R1",
    )
    return {
        "fit_success": bool(selected.success),
        "fit_error": "none",
        "selected_model": selected.model,
        "p_r1": probability_r1,
        "r0_r2": float(r0.r2),
        "r1_r2": float(r1.r2),
        "selected_r2": float(selected.r2),
        "preferred_sf_cpd": fitted_sf,
        "preferred_tf_hz": fitted_tf,
        "inside_grid": bool(
            sf[0] < fitted_sf < sf[-1]
            and dynamic_tf[0] < fitted_tf < dynamic_tf[-1]
        ),
        "raw_peak_sf_error_octaves": _octave_error(
            fitted_sf, float(sf[raw_sf_index])
        ),
        "raw_peak_tf_error_octaves": _octave_error(
            fitted_tf, float(dynamic_tf[raw_tf_index])
        ),
        "parameters": parameters,
        "prediction": prediction,
    }


def analyze(
    units: pd.DataFrame,
    sf: np.ndarray,
    tf: np.ndarray,
    directions: np.ndarray,
    delta_cube: np.ndarray,
    f0_cube: np.ndarray,
    half_f0_cube: np.ndarray,
    blank_expected_count: np.ndarray,
    recorded_metrics: pd.DataFrame,
    recorded_curves: dict[int, dict],
    comparison_delta_cube: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[int, dict]]:
    dynamic = tf > 0
    rows: list[dict] = []
    displays: dict[int, dict] = {}
    for unit_index, unit in units.iterrows():
        unit_delta = delta_cube[..., unit_index]
        unit_f0 = f0_cube[..., unit_index]
        unit_half_f0 = half_f0_cube[..., unit_index]
        direction_index = preferred_direction(unit_delta, dynamic)
        half_delta = unit_half_f0 - blank_expected_count[unit_index]
        half_direction_index = preferred_direction(half_delta, dynamic)
        surface = np.maximum(
            unit_delta[:, :, direction_index][:, dynamic], 0.0
        ).T
        half_surface = np.maximum(
            half_delta[:, :, half_direction_index][:, dynamic],
            0.0,
        ).T
        peak_tf_index, peak_sf_index = np.unravel_index(
            int(np.argmax(surface)), surface.shape
        )
        half_tf_index, half_sf_index = np.unravel_index(
            int(np.argmax(half_surface)), half_surface.shape
        )
        dynamic_tf = tf[dynamic]
        span = float(np.ptp(surface))
        roughness = (
            float(np.mean(np.abs(np.diff(surface, axis=0))))
            + float(np.mean(np.abs(np.diff(surface, axis=1))))
        ) / max(span, 1e-12)
        _, lasso = connected_peak_lasso(surface)
        local_maximum = surface == maximum_filter(surface, size=3, mode="nearest")
        high_mode_count = int(np.count_nonzero(local_maximum & (surface >= 0.75 * surface.max())))
        phase_corr = _correlation(
            unit_f0.ravel(), unit_half_f0.ravel()
        )
        peak_sf_error = _octave_error(sf[peak_sf_index], sf[half_sf_index])
        peak_tf_error = _octave_error(dynamic_tf[peak_tf_index], dynamic_tf[half_tf_index])
        direction_error = _circular_error_deg(
            directions[direction_index], directions[half_direction_index]
        )
        boundary = bool(
            peak_sf_index in (0, len(sf) - 1)
            or peak_tf_index in (0, len(dynamic_tf) - 1)
        )
        anchor = recorded_sf_anchor(
            unit,
            recorded_metrics,
            recorded_curves,
            sf,
            directions,
            unit_delta[:, dynamic, :],
        )
        peak_delta = float(surface[peak_tf_index, peak_sf_index])
        yu = fit_yu_passband(sf, dynamic_tf, surface)
        yu_stable = bool(
            yu.get("fit_success", False)
            and yu.get("selected_r2", -np.inf) >= 0.8
            and yu.get("inside_grid", False)
            and yu.get("raw_peak_sf_error_octaves", np.inf) <= 0.75
            and yu.get("raw_peak_tf_error_octaves", np.inf) <= 0.75
        )

        contrast = {
            "contrast_repeat_available": comparison_delta_cube is not None,
            "contrast_surface_corr": np.nan,
            "contrast_peak_sf_error_octaves": np.nan,
            "contrast_peak_tf_error_octaves": np.nan,
            "contrast_direction_error_deg": np.nan,
            "contrast_repeat_peak_delta_expected_count": np.nan,
            "contrast_stable": False,
        }
        if comparison_delta_cube is not None:
            repeat_delta = comparison_delta_cube[..., unit_index]
            repeat_direction_index = preferred_direction(repeat_delta, dynamic)
            repeat_surface = np.maximum(
                repeat_delta[:, :, repeat_direction_index][:, dynamic], 0.0
            ).T
            repeat_peak_tf, repeat_peak_sf = np.unravel_index(
                int(np.argmax(repeat_surface)), repeat_surface.shape
            )
            contrast.update(
                {
                    "contrast_surface_corr": _correlation(
                        surface.ravel(), repeat_surface.ravel()
                    ),
                    "contrast_peak_sf_error_octaves": _octave_error(
                        sf[peak_sf_index], sf[repeat_peak_sf]
                    ),
                    "contrast_peak_tf_error_octaves": _octave_error(
                        dynamic_tf[peak_tf_index], dynamic_tf[repeat_peak_tf]
                    ),
                    "contrast_direction_error_deg": _circular_error_deg(
                        directions[direction_index],
                        directions[repeat_direction_index],
                    ),
                    "contrast_repeat_peak_delta_expected_count": float(
                        repeat_surface[repeat_peak_tf, repeat_peak_sf]
                    ),
                }
            )
            contrast["contrast_stable"] = bool(
                np.isfinite(contrast["contrast_surface_corr"])
                and contrast["contrast_surface_corr"] >= 0.8
                and contrast["contrast_peak_sf_error_octaves"] <= 0.75
                and contrast["contrast_peak_tf_error_octaves"] <= 0.75
                and contrast["contrast_direction_error_deg"] <= 40.1
                and contrast["contrast_repeat_peak_delta_expected_count"] >= 0.001
            )

        checks = {
            "phase_converged": bool(
                np.isfinite(phase_corr)
                and phase_corr >= 0.995
                and peak_sf_error <= 0.51
                and peak_tf_error <= 0.51
                and direction_error <= 20.1
            ),
            "positive_drive": bool(peak_delta >= 0.002),
            "interior_peak": not boundary,
            "coherent_surface": bool(roughness <= 0.35 and high_mode_count <= 2 and lasso.any()),
            "yu_fit_stable": yu_stable,
            "contrast_stable": bool(contrast["contrast_stable"]),
            "recorded_sf_data_reliable": bool(anchor["recorded_sf_data_reliable"]),
            "recorded_sf_twin_match": bool(anchor["recorded_sf_twin_match"]),
        }
        failed = [name for name, passed in checks.items() if not passed]
        model_checks = {
            key: checks[key]
            for key in (
                "phase_converged",
                "positive_drive",
                "interior_peak",
                "coherent_surface",
                "yu_fit_stable",
                "contrast_stable",
            )
        }
        rows.append(
            {
                **unit.to_dict(),
                "preferred_motion_direction_deg": float(directions[direction_index]),
                "preferred_sf_cpd": float(sf[peak_sf_index]),
                "preferred_tf_hz": float(dynamic_tf[peak_tf_index]),
                "peak_delta_f0_expected_count": peak_delta,
                "blank_expected_count": float(blank_expected_count[unit_index]),
                "phase_count_curve_corr": phase_corr,
                "phase_count_peak_sf_error_octaves": peak_sf_error,
                "phase_count_peak_tf_error_octaves": peak_tf_error,
                "phase_count_direction_error_deg": direction_error,
                "surface_neighbor_roughness": roughness,
                "high_mode_count": high_mode_count,
                "boundary_peak": boundary,
                "passband_cells": int(lasso.sum()),
                **anchor,
                "yu_selected_model": yu.get("selected_model", "failed"),
                "yu_p_r1": yu.get("p_r1", np.nan),
                "yu_selected_r2": yu.get("selected_r2", np.nan),
                "yu_r0_r2": yu.get("r0_r2", np.nan),
                "yu_r1_r2": yu.get("r1_r2", np.nan),
                "yu_preferred_sf_cpd": yu.get("preferred_sf_cpd", np.nan),
                "yu_preferred_tf_hz": yu.get("preferred_tf_hz", np.nan),
                "yu_peak_inside_grid": bool(yu.get("inside_grid", False)),
                "yu_raw_peak_sf_error_octaves": yu.get(
                    "raw_peak_sf_error_octaves", np.nan
                ),
                "yu_raw_peak_tf_error_octaves": yu.get(
                    "raw_peak_tf_error_octaves", np.nan
                ),
                "yu_fit_error": yu.get("fit_error", "unknown"),
                **contrast,
                **checks,
                "validated_model_sf_tf": bool(all(model_checks.values())),
                "validated_for_figure4": bool(all(checks.values())),
                "failed_checks": "+".join(failed) if failed else "none",
            }
        )
        displays[int(unit_index)] = {
            "surface": surface,
            "lasso": lasso,
            "yu_prediction": yu.get("prediction"),
            "yu_parameters": yu.get("parameters"),
            "yu_model": yu.get("selected_model"),
            "direction_curve": np.maximum(
                unit_delta[peak_sf_index, np.flatnonzero(dynamic)[peak_tf_index], :],
                0.0,
            ),
            "recorded_curve": recorded_curves.get(int(unit.canonical_channel)),
        }
    return pd.DataFrame(rows), displays


def render_summary(summary: pd.DataFrame, out_dir: Path) -> Path:
    plt.rcParams.update(
        {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 5, figsize=(16.2, 3.1), constrained_layout=True)
    axes[0].hist(summary.phase_count_curve_corr.dropna(), bins=30, color="#2679B8")
    axes[0].axvline(0.995, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="24-phase vs 12-phase curve correlation", ylabel="units")

    axes[1].hist(summary.yu_selected_r2.dropna(), bins=30, color="#6A51A3")
    axes[1].axvline(0.8, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Yu R0/R1 full-grid $R^2$", ylabel="units")

    axes[2].hist(summary.contrast_surface_corr.dropna(), bins=30, color="#D95F02")
    axes[2].axvline(0.8, color="black", linestyle="--", linewidth=1)
    axes[2].set(xlabel="surface correlation across contrast", ylabel="units")

    accepted = summary.loc[summary.validated_for_figure4]
    rejected = summary.loc[~summary.validated_for_figure4]
    axes[3].scatter(
        rejected.yu_preferred_sf_cpd,
        rejected.yu_preferred_tf_hz,
        s=9,
        facecolors="none",
        edgecolors="0.7",
        linewidths=0.5,
    )
    axes[3].scatter(
        accepted.yu_preferred_sf_cpd,
        accepted.yu_preferred_tf_hz,
        s=12,
        color="#2E8B57",
    )
    axes[3].set_xscale("log", base=2)
    axes[3].set_yscale("log", base=2)
    axes[3].set(
        xlabel="exact-CID Yu preferred SF (cycles/deg)",
        ylabel="exact-CID Yu preferred TF (Hz)",
    )

    checks = [
        "phase_converged",
        "positive_drive",
        "interior_peak",
        "coherent_surface",
        "yu_fit_stable",
        "contrast_stable",
        "recorded_sf_data_reliable",
        "recorded_sf_twin_match",
    ]
    failures = [int((~summary[key]).sum()) for key in checks]
    axes[4].barh(np.arange(len(checks)), failures, color="#C75B4B")
    axes[4].set_yticks(np.arange(len(checks)), [key.replace("_", " ") for key in checks])
    axes[4].set(xlabel="units failing check")
    path = out_dir / "population_measurement_audit.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def render_atlas(
    summary: pd.DataFrame,
    displays: dict[int, dict],
    sf: np.ndarray,
    tf: np.ndarray,
    out_dir: Path,
    *,
    filename: str = "all_exact_units_raw_tuning_atlas.pdf",
    title: str = "Raw exact-CID F0 SF x TF surfaces; white = Yu-fit half-max lasso",
) -> Path:
    dynamic_tf = tf[tf > 0]
    path = out_dir / filename
    with PdfPages(path) as pdf:
        for start in range(0, len(summary), ATLAS_UNITS_PER_PAGE):
            page = summary.iloc[start : start + ATLAS_UNITS_PER_PAGE]
            figure, axes = plt.subplots(4, 5, figsize=(12.5, 9.2), squeeze=False)
            for axis, (_, row) in zip(axes.flat, page.iterrows()):
                display = displays[int(row.unit_index)]
                surface = display["surface"]
                vmax = max(float(surface.max()), 1e-12)
                axis.imshow(
                    surface / vmax,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
                prediction = display["yu_prediction"]
                if prediction is not None and float(np.nanmax(prediction)) > 0:
                    axis.contour(
                        prediction / float(np.nanmax(prediction)),
                        levels=[0.5],
                        colors="white",
                        linewidths=1.0,
                    )
                sf_index = int(np.argmin(np.abs(sf - row.preferred_sf_cpd)))
                tf_index = int(np.argmin(np.abs(dynamic_tf - row.preferred_tf_hz)))
                axis.plot(sf_index, tf_index, marker="*", color="#FFD84D", markersize=6)
                if np.isfinite(row.yu_preferred_sf_cpd) and np.isfinite(
                    row.yu_preferred_tf_hz
                ):
                    fit_sf_index = np.interp(
                        np.log2(row.yu_preferred_sf_cpd),
                        np.log2(sf),
                        np.arange(len(sf)),
                    )
                    fit_tf_index = np.interp(
                        np.log2(row.yu_preferred_tf_hz),
                        np.log2(dynamic_tf),
                        np.arange(len(dynamic_tf)),
                    )
                    axis.plot(
                        fit_sf_index,
                        fit_tf_index,
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor="white",
                        markersize=4,
                    )
                axis.set_xticks(
                    np.arange(0, len(sf), 2),
                    [f"{value:.2g}" for value in sf[::2]],
                    fontsize=5.5,
                )
                tf_tick = np.arange(0, len(dynamic_tf), 2)
                axis.set_yticks(
                    tf_tick,
                    [f"{dynamic_tf[index]:.2g}" for index in tf_tick],
                    fontsize=5.5,
                )
                status = "PASS" if bool(row.validated_for_figure4) else str(row.failed_checks)
                axis.set_title(
                    f"ch{int(row.canonical_channel):03d} {row.session} cid{int(row.cid)}\n{status}",
                    fontsize=6.2,
                )
            for axis in axes.flat[len(page) :]:
                axis.axis("off")
            figure.suptitle(
                title,
                fontsize=11,
            )
            figure.supxlabel("spatial frequency (cycles/deg)", fontsize=9)
            figure.supylabel("temporal frequency (Hz)", fontsize=9)
            figure.tight_layout(rect=(0.025, 0.025, 1, 0.97))
            pdf.savefig(figure)
            plt.close(figure)
    return path


def assign_crossed_groups(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Define crossed examples without letting a model fit invent biological SF.

    Recorded-neuron SF and exact-twin Yu SF must agree on the same outer SF
    third.  The twin TF preference must then occupy the opposite outer third.
    The reference distributions are defined before intersecting all Figure-4
    gates: reliable recorded SF defines the SF cutoffs and otherwise-valid
    model SFxTF measurements define the TF cutoffs.  This avoids moving the
    thresholds merely because an unrelated downstream gate was tightened.
    """
    result = summary.copy()
    eligible = result.validated_for_figure4.astype(bool)
    sf_reference = result.loc[
        result.recorded_sf_data_reliable.astype(bool),
        "recorded_data_preferred_sf_cpd",
    ].dropna()
    tf_reference = result.loc[
        result.validated_model_sf_tf.astype(bool), "yu_preferred_tf_hz"
    ].dropna()
    if len(sf_reference) < 30 or len(tf_reference) < 30:
        raise RuntimeError("too few reliable units to define crossed SF/TF thirds")
    sf_low, sf_high = np.quantile(sf_reference, [1.0 / 3.0, 2.0 / 3.0])
    tf_low, tf_high = np.quantile(tf_reference, [1.0 / 3.0, 2.0 / 3.0])
    result["crossed_group"] = "middle"
    low = (
        eligible
        & result.recorded_data_preferred_sf_cpd.le(sf_low)
        & result.yu_preferred_sf_cpd.le(sf_low)
        & result.yu_preferred_tf_hz.ge(tf_high)
    )
    high = (
        eligible
        & result.recorded_data_preferred_sf_cpd.ge(sf_high)
        & result.yu_preferred_sf_cpd.ge(sf_high)
        & result.yu_preferred_tf_hz.le(tf_low)
    )
    result.loc[low, "crossed_group"] = "low recorded SF / high twin TF"
    result.loc[high, "crossed_group"] = "high recorded SF / low twin TF"
    result["crossed_extremity_octaves"] = np.nan
    result.loc[low, "crossed_extremity_octaves"] = (
        np.log2(sf_low / result.loc[low, "recorded_data_preferred_sf_cpd"])
        + np.log2(sf_low / result.loc[low, "yu_preferred_sf_cpd"])
        + np.log2(result.loc[low, "yu_preferred_tf_hz"] / tf_high)
    )
    result.loc[high, "crossed_extremity_octaves"] = (
        np.log2(result.loc[high, "recorded_data_preferred_sf_cpd"] / sf_high)
        + np.log2(result.loc[high, "yu_preferred_sf_cpd"] / sf_high)
        + np.log2(tf_low / result.loc[high, "yu_preferred_tf_hz"])
    )
    counts = {
        group: int((eligible & result.crossed_group.eq(group)).sum())
        for group in (
            "low recorded SF / high twin TF",
            "high recorded SF / low twin TF",
        )
    }
    return result, {
        "definition": (
            "outer population thirds: both reliable recorded-neuron SF and "
            "exact-twin Yu-fit SF must fall in the same SF third; twin TF must "
            "fall in the opposite outer TF third"
        ),
        "recorded_sf_tercile_cutoffs_cpd": [float(sf_low), float(sf_high)],
        "twin_tf_tercile_cutoffs_hz": [float(tf_low), float(tf_high)],
        "validated_counts": counts,
    }


def select_candidate_units(summary: pd.DataFrame, n_per_group: int = 6) -> pd.DataFrame:
    selected = []
    for group in (
        "low recorded SF / high twin TF",
        "high recorded SF / low twin TF",
    ):
        candidates = summary.loc[
            summary.validated_for_figure4 & summary.crossed_group.eq(group)
        ].copy()
        candidates = candidates.sort_values(
            [
                "crossed_extremity_octaves",
                "high_mode_count",
                "yu_selected_r2",
                "contrast_surface_corr",
                "peak_delta_f0_expected_count",
            ],
            ascending=[False, True, False, False, False],
        )
        candidates["candidate_rank_within_group"] = np.arange(1, len(candidates) + 1)
        selected.append(candidates.head(int(n_per_group)))
    if not selected:
        return pd.DataFrame()
    result = pd.concat(selected, ignore_index=True)
    order = {
        "low recorded SF / high twin TF": 0,
        "high recorded SF / low twin TF": 1,
    }
    result["candidate_group_order"] = result.crossed_group.map(order)
    return result.sort_values(
        ["candidate_rank_within_group", "candidate_group_order"]
    ).reset_index(drop=True)


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    low, high = float(np.nanmin(value)), float(np.nanmax(value))
    return (value - low) / max(high - low, 1e-12)


def render_candidate_diagnostics(
    candidates: pd.DataFrame,
    displays: dict[int, dict],
    sf: np.ndarray,
    tf: np.ndarray,
    directions: np.ndarray,
    out_dir: Path,
) -> tuple[Path, Path | None]:
    """Expose raw surfaces, recorded SF, TF slice, and direction curve for selection."""
    dynamic_tf = tf[tf > 0]
    pdf_path = out_dir / "crossed_candidate_diagnostics.pdf"
    preview_path: Path | None = None
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(candidates), 3):
            page = candidates.iloc[page_start : page_start + 3]
            figure, axes = plt.subplots(
                len(page), 4, figsize=(14.5, 3.15 * len(page)), squeeze=False,
                constrained_layout=True,
            )
            for row_axes, (_, row) in zip(axes, page.iterrows()):
                display = displays[int(row.unit_index)]
                surface = display["surface"]
                surface_max = max(float(np.nanmax(surface)), 1e-12)
                axis = row_axes[0]
                mesh = axis.pcolormesh(
                    sf,
                    dynamic_tf,
                    surface / surface_max,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    shading="nearest",
                )
                parameters = display["yu_parameters"]
                if parameters is not None:
                    fine_sf = np.geomspace(sf[0], sf[-1], 180)
                    fine_tf = np.geomspace(dynamic_tf[0], dynamic_tf[-1], 180)
                    fine_sf_grid, fine_tf_grid = np.meshgrid(fine_sf, fine_tf)
                    fitted = yu_surface(
                        parameters,
                        np.log2(fine_sf_grid),
                        np.log2(fine_tf_grid),
                        inseparable=display["yu_model"] == "R1",
                    )
                    if float(np.nanmax(fitted)) > 0:
                        axis.contour(
                            fine_sf,
                            fine_tf,
                            fitted / float(np.nanmax(fitted)),
                            levels=[0.5],
                            colors="white",
                            linewidths=1.4,
                        )
                axis.plot(
                    row.preferred_sf_cpd,
                    row.preferred_tf_hz,
                    marker="*",
                    color="#FFD84D",
                    markeredgecolor="black",
                    markersize=8,
                )
                axis.set_xscale("log", base=2)
                axis.set_yscale("log", base=2)
                axis.set(xlabel="SF (cycles/deg)", ylabel="TF (Hz)")
                axis.set_title(
                    f"ch{int(row.canonical_channel):03d} {row.session} cid{int(row.cid)}\n"
                    f"{row.crossed_group}; Yu {row.yu_selected_model} $R^2$={row.yu_selected_r2:.2f}",
                    fontsize=8,
                )

                axis = row_axes[1]
                recorded = display["recorded_curve"]
                if recorded is not None:
                    axis.plot(
                        recorded["sf"],
                        _normalize_curve(recorded["data_sf"]),
                        marker="o",
                        label="recorded neuron",
                    )
                    axis.plot(
                        recorded["sf"],
                        _normalize_curve(recorded["model_sf"]),
                        marker="s",
                        label="twin on recorded trials",
                    )
                axis.set_xscale("log", base=2)
                axis.set(
                    xlabel="recorded-grating SF (cycles/deg)",
                    ylabel="normalized response",
                    ylim=(-0.05, 1.05),
                )
                axis.legend(frameon=False, fontsize=6.5)

                axis = row_axes[2]
                nearest_sf = int(
                    np.argmin(np.abs(np.log2(sf / row.preferred_sf_cpd)))
                )
                axis.plot(
                    dynamic_tf,
                    surface[:, nearest_sf] / surface_max,
                    marker="o",
                    label="measured grid",
                )
                prediction = display["yu_prediction"]
                if prediction is not None:
                    axis.plot(
                        dynamic_tf,
                        prediction[:, nearest_sf] / max(float(np.max(prediction)), 1e-12),
                        label="Yu fit",
                    )
                axis.set_xscale("log", base=2)
                axis.set(
                    xlabel=f"TF at {sf[nearest_sf]:.2g} cycles/deg",
                    ylabel="normalized response",
                    ylim=(-0.05, 1.05),
                )
                axis.legend(frameon=False, fontsize=6.5)

                axis = row_axes[3]
                direction_curve = np.asarray(display["direction_curve"], dtype=float)
                axis.plot(
                    np.r_[directions, 360.0],
                    np.r_[direction_curve, direction_curve[0]]
                    / max(float(np.max(direction_curve)), 1e-12),
                    marker="o",
                    markersize=3,
                )
                axis.axvline(
                    row.preferred_motion_direction_deg,
                    color="0.3",
                    linestyle="--",
                    linewidth=0.8,
                )
                axis.set(
                    xlabel="motion direction (deg)",
                    ylabel="normalized response",
                    xlim=(0, 360),
                    ylim=(-0.05, 1.05),
                )
            figure.suptitle(
                "Crossed SF/TF candidates: every intermediate measurement is visible",
                fontsize=12,
                fontweight="bold",
            )
            pdf.savefig(figure)
            if preview_path is None:
                preview_path = out_dir / "crossed_candidate_diagnostics_preview.png"
                figure.savefig(preview_path, dpi=180, bbox_inches="tight")
            plt.close(figure)
    return pdf_path, preview_path


def main() -> None:
    args = parse_args()
    provenance, units, conditions, archive = load_measurement(args.measurement_dir)
    sf, tf, directions, delta_cube = response_cube(
        conditions, archive["delta_f0_expected_count"]
    )
    sf2, tf2, directions2, f0_cube = response_cube(
        conditions, archive["f0_expected_count"]
    )
    sf3, tf3, directions3, half_cube = response_cube(
        conditions, archive["half_phase_f0_expected_count"]
    )
    if not (
        np.array_equal(sf, sf2)
        and np.array_equal(sf, sf3)
        and np.array_equal(tf, tf2)
        and np.array_equal(tf, tf3)
        and np.array_equal(directions, directions2)
        and np.array_equal(directions, directions3)
    ):
        raise RuntimeError("response cubes disagree on their physical axes")
    if np.count_nonzero(tf == 0) != 1:
        raise ValueError("audit requires exactly one static TF=0 slice")
    opposite = (directions + 180.0) % 360.0
    if not all(np.any(np.isclose(directions, value)) for value in opposite):
        raise ValueError("motion directions do not contain explicit opposite pairs")

    comparison_provenance = None
    comparison_delta_cube = None
    if args.comparison_measurement_dir is not None:
        (
            comparison_provenance,
            comparison_units,
            comparison_conditions,
            comparison_archive,
        ) = load_measurement(args.comparison_measurement_dir)
        pd.testing.assert_frame_equal(
            units[["canonical_channel", "session", "cid"]].reset_index(drop=True),
            comparison_units[["canonical_channel", "session", "cid"]].reset_index(
                drop=True
            ),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            conditions.reset_index(drop=True),
            comparison_conditions.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
        if np.isclose(
            provenance["contrast_amplitude"],
            comparison_provenance["contrast_amplitude"],
        ):
            raise ValueError("contrast-repeat audit requires two different contrasts")
        comparison_sf, comparison_tf, comparison_directions, comparison_delta_cube = (
            response_cube(
                comparison_conditions,
                comparison_archive["delta_f0_expected_count"],
            )
        )
        if not (
            np.array_equal(sf, comparison_sf)
            and np.array_equal(tf, comparison_tf)
            and np.array_equal(directions, comparison_directions)
        ):
            raise RuntimeError("contrast-repeat physical axes differ")

    recorded_metrics, recorded_curves = recorded_static_curves(
        args.recorded_static_dir
    )
    summary, displays = analyze(
        units,
        sf,
        tf,
        directions,
        delta_cube,
        f0_cube,
        half_cube,
        archive["blank_expected_count"],
        recorded_metrics,
        recorded_curves,
        comparison_delta_cube,
    )
    summary, group_definition = assign_crossed_groups(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "unit_measurement_audit.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = render_summary(summary, args.out_dir)
    atlas_path = render_atlas(summary, displays, sf, tf, args.out_dir)
    validated_atlas_path = render_atlas(
        summary.loc[summary.validated_for_figure4].reset_index(drop=True),
        displays,
        sf,
        tf,
        args.out_dir,
        filename="validated_exact_units_raw_tuning_atlas.pdf",
        title=(
            "Figure-4-eligible exact-CID F0 SF x TF surfaces; "
            "white = Yu-fit half-max lasso"
        ),
    )
    candidates = select_candidate_units(summary)
    candidate_path = args.out_dir / "crossed_candidate_units.csv"
    candidates.to_csv(candidate_path, index=False)
    candidate_pdf, candidate_preview = render_candidate_diagnostics(
        candidates, displays, sf, tf, directions, args.out_dir
    )
    validated_count = int(summary.validated_for_figure4.sum())
    crossed_counts = group_definition["validated_counts"]
    release_ready = bool(
        comparison_provenance is not None
        and validated_count >= 100
        and all(count >= 4 for count in crossed_counts.values())
    )
    payload = {
        "analysis": "exact-CID drifting-grating release audit",
        "source_measurement": str(args.measurement_dir.resolve()),
        "source_provenance": provenance,
        "contrast_repeat_measurement": (
            str(args.comparison_measurement_dir.resolve())
            if args.comparison_measurement_dir is not None
            else None
        ),
        "contrast_repeat_provenance": comparison_provenance,
        "recorded_grating_sf_anchor": str(args.recorded_static_dir.resolve()),
        "interpretation_contract": {
            "recorded_biology": (
                "all genuine recorded static-forage-grating trials; descriptive SF, "
                "orientation, retinal phase, and response-latency tuning; not a TF assay"
            ),
            "model_sf_tf": (
                "controlled synthetic drifting gratings through exact checkpoint-native "
                "matched readouts; TF is a twin inference, not a recorded-neuron measurement"
            ),
            "figure4_population_sf_groups": (
                "must be defined by reliable recorded-neuron SF, never by RR100 or the "
                "synthetic drifting bank"
            ),
            "figure4_passbands": (
                "may use only validated exact-twin Yu SFxTF fits and must retain the "
                "raw measured grid beside every exemplar fit"
            ),
            "rr100_use": "none in measurement, validation, grouping, or example selection",
        },
        "response_metric_contract": {
            "primary": (
                "phase-averaged model expected count (F0) minus explicit gray blank"
            ),
            "reason": "Yu-style drifting-grating mean response and the Figure 4 rate mechanism",
            "units": "expected spikes per 1/240-s output bin; multiply by 240 for spikes/s",
            "diagnostics_only": "F1, F2, phase RMS, all-harmonic RMS, min, and max",
        },
        "n_units": int(len(summary)),
        "n_validated_model_sf_tf": int(summary.validated_model_sf_tf.sum()),
        "n_validated_for_figure4": validated_count,
        "validation_fraction": float(summary.validated_for_figure4.mean()),
        "visual_audit_contract": {
            "atlas_units_per_page": ATLAS_UNITS_PER_PAGE,
            "expected_validated_atlas_pages": list(
                range(
                    1,
                    (validated_count + ATLAS_UNITS_PER_PAGE - 1)
                    // ATLAS_UNITS_PER_PAGE
                    + 1,
                )
            ),
            "inspection_scope": "every page of validated_exact_units_raw_tuning_atlas.pdf",
        },
        "failure_counts": {
            key: int((~summary[key]).sum())
            for key in (
                "phase_converged",
                "positive_drive",
                "interior_peak",
                "coherent_surface",
                "yu_fit_stable",
                "contrast_stable",
                "recorded_sf_data_reliable",
                "recorded_sf_twin_match",
            )
        },
        "crossed_group_definition": group_definition,
        "release_rule": (
            "exact identity and complete grid; 24-vs-12 phase convergence; positive "
            "blank-subtracted F0; interior coherent raw peak; full-grid Yu R0/R1 "
            "R^2 >= 0.80 with raw/fit peak agreement; stable peak and surface at a "
            "second contrast; reliable non-boundary recorded-neuron SF; matched-twin "
            "recorded-SF agreement. Release additionally requires >=100 units and "
            ">=4 independently validated candidates in each crossed outer-third "
            "SF/TF group, so no example depends on a lone hand-picked unit."
        ),
        "figure4_unblocked": release_ready,
        "files": {
            "unit_audit": str(summary_path.resolve()),
            "population_figure": str(figure_path.resolve()),
            "all_unit_atlas": str(atlas_path.resolve()),
            "validated_unit_atlas": str(validated_atlas_path.resolve()),
            "candidate_table": str(candidate_path.resolve()),
            "candidate_diagnostics": str(candidate_pdf.resolve()),
            "candidate_preview": (
                str(candidate_preview.resolve()) if candidate_preview else None
            ),
        },
    }
    report_path = args.out_dir / "release_audit.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
