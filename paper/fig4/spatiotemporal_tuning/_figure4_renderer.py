#!/usr/bin/env python3
"""Compose the canonical eight-panel Figure-4 narrative.

This builder keeps effect estimation, conditional retinal power, measured
tuning, signal routing, and network-stage analysis as separate visual steps.
It is model-agnostic and requires explicit, provenance-carrying inputs for
every data-dependent panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning._spectral_shards import (  # noqa: E402
    load_and_merge_shards,
)
from paper.fig4.spatiotemporal_tuning._figure4_rendering import (  # noqa: E402
    EPS,
    PASSBAND_RESPONSE_FRACTION,
    PURPLE,
    ROLE_COLORS,
    ROLE_NAMES,
    _compose_page,
    _exemplar_tuning,
    _frequency_axes,
    _load_panel_a_audit,
    _load_shard_summaries,
    _overlay_passband,
    _overlay_passbands,
    _panel_title,
    _render_page_png,
    _render_panel,
    _summary_checkpoint_digest,
    configure,
    draw_panel_a,
    routing_metrics,
    select_population_units,
    support_limited_yu_tuning_surface,
)
PAGE_SIZE = (12.0, 10.0)
EFFECT_GREEN = "#009E73"
PASSBAND_DENSITY_DARKEST_GRAY = 0.50
PANEL_LAYOUT = {
    "A": (0.08, 0.10, 5.72, 3.45),
    "B": (5.92, 0.10, 5.98, 3.45),
    "C": (0.15, 3.72, 4.16, 2.78),
    "D": (4.39, 3.72, 4.08, 2.78),
    "E": (8.55, 3.72, 3.29, 2.78),
    "F": (0.22, 6.62, 3.00, 2.72),
    "G": (3.34, 6.62, 4.15, 2.72),
    "H": (7.62, 6.62, 4.22, 2.72),
}
MANUSCRIPT_PAGE_SIZE = (6.5, 8.5)
MANUSCRIPT_PANEL_LAYOUT = {
    "A": (0.04, 0.04, 6.40, 2.35),
    "B": (0.04, 2.46, 3.20, 1.90),
    "C": (3.30, 2.46, 3.15, 1.90),
    "D": (0.04, 4.46, 3.20, 1.85),
    "E": (3.30, 4.46, 1.50, 1.85),
    "F": (4.91, 4.46, 1.54, 1.85),
    "G": (0.04, 6.42, 3.20, 1.98),
    "H": (3.30, 6.42, 3.15, 1.98),
}
MANUSCRIPT_LABELS = {
    "Population response versus fixational path length": "Effect of fixation path length",
    "firing-rate change": "Rate change",
    "single-spike information change": "SSI change",
    "filtered fixation path length (arcmin)": "Path length (arcmin)",
    "motion − stabilized (%)": "Change (%)",
    "firing-rate change\nmotion − stabilized (%)": "Rate change (%)",
    "single-spike information change\nmotion − stabilized (%)": "SSI change (%)",
    "passband-power percentile": "Engagement\npercentile",
    "cumulative readout": "Cumulative\nreadout",
    "SF (cycles/deg)": "SF (cpd)",
    "log₁₀ conditional power density": "log₁₀ power density",
    "power shifts to\nhigher TF": "Higher TF",
    "motion − stabilized\n(% of mean response)": "Temporal change\n(% mean response)",
    "motion − stabilized\n(bits/spike)": "SSI change\n(bits/spike)",
}


def _panel_label(subfigure, label: str) -> None:
    """Draw only the panel letter; C--H use in-axis annotations, not titles."""
    subfigure.text(
        0.0,
        0.985,
        label,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--panel-a-audit", type=Path, required=True)
    parser.add_argument("--panel-b-reduction", type=Path, required=True)
    parser.add_argument("--tuning-table", type=Path, required=True)
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument("--example-fits", type=Path, required=True)
    parser.add_argument(
        "--all-fits",
        type=Path,
        required=True,
        help="Released Yu R0/R1 parameter table for every Panel-E unit.",
    )
    parser.add_argument("--rucci-ensemble", type=Path, required=True)
    parser.add_argument("--population-shards", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--stage-trajectory",
        type=Path,
        required=True,
        help="Audited top-passband cumulative-readout trajectory for Panel H.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("production", "manuscript"), default="production")
    parser.add_argument(
        "--population-policy",
        choices=("validated", "all_checkpoint_available"),
        default="validated",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _draw_panel_b(
    subfigure,
    curves: pd.DataFrame,
    unit_effects: dict[str, np.ndarray],
) -> dict[str, object]:
    _panel_title(subfigure, "B", "Population response versus fixational path length")
    axes = subfigure.subplots(1, 2, gridspec_kw={"wspace": 0.34})
    reports: dict[str, object] = {}
    for axis, outcome, title, color in zip(
        axes,
        ("rate", "SSI"),
        ("firing-rate change", "single-spike information change"),
        (EFFECT_GREEN, PURPLE),
    ):
        frame = curves.loc[curves.outcome.eq(outcome)].sort_values("bin_index")
        if frame.empty:
            raise ValueError(f"Panel B lacks {outcome} rows")
        distribution_key = "rate_percent" if outcome == "rate" else "ssi_percent"
        distributions = np.asarray(unit_effects[distribution_key], dtype=float)
        if distributions.shape != (len(frame), int(frame.n_units.iloc[0])):
            raise ValueError(
                f"Panel-B {outcome} unit distribution has shape "
                f"{distributions.shape}, expected {(len(frame), int(frame.n_units.iloc[0]))}"
            )
        positions = frame.x_median.to_numpy(dtype=float)
        if not np.allclose(
            positions,
            np.asarray(unit_effects["x_median"], dtype=float),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"Panel-B {outcome} distribution bins do not match its curve")
        gaps = np.diff(positions)
        # Keep crowded bins separate without collapsing boxes at sparse paths.
        widths = 0.42 * np.minimum(
            np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]]
        )
        box = axis.boxplot(
            [row[np.isfinite(row)] for row in distributions],
            positions=positions,
            widths=widths,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            zorder=1,
        )
        for artist in box["boxes"]:
            artist.set(facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.75)
        for artist in (*box["whiskers"], *box["caps"]):
            artist.set(color=color, alpha=0.55, linewidth=0.65)
        for artist in box["medians"]:
            artist.set(color=color, linewidth=1.25)
        center = frame.effect_percent.to_numpy(dtype=float)
        axis.errorbar(
            frame.x_median,
            center,
            yerr=np.vstack(
                (
                    center - frame.ci_low.to_numpy(dtype=float),
                    frame.ci_high.to_numpy(dtype=float) - center,
                )
            ),
            fmt="o-",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=1.7,
            ms=4.6,
            capsize=2.2,
            zorder=3,
        )
        axis.axhline(0, color="0.5", lw=0.75)
        axis.set_title(title, fontsize=6.9)
        if outcome == "rate":
            axis.set_ylabel("motion − stabilized (%)")
        axis.grid(axis="y", alpha=0.16)
        reports[outcome] = {
            "x_median": frame.x_median.to_numpy(dtype=float).tolist(),
            "effect_percent": center.tolist(),
            "ci95": frame[["ci_low", "ci_high"]].to_numpy(dtype=float).tolist(),
            "unit_distribution": {
                "n_units": int(distributions.shape[1]),
                "median": np.median(distributions, axis=1).tolist(),
                "iqr": np.quantile(distributions, (0.25, 0.75), axis=1).T.tolist(),
                "whisker_5_95": np.quantile(
                    distributions, (0.05, 0.95), axis=1
                ).T.tolist(),
                "fraction_positive": np.mean(distributions > 0, axis=1).tolist(),
            },
        }
    subfigure.supxlabel("filtered fixation path length (arcmin)", x=0.56, y=0.045,
                       fontsize=6.8)
    return {
        "curves": reports,
        "distribution_display": (
            "per-unit boxes show interquartile range and median; whiskers show "
            "5th--95th percentiles; colored line and CI show the spike-weighted "
            "population estimand"
        ),
    }


def _shared_frequency_limits(
    tuning: dict[str, object], metrics: dict[str, object]
) -> tuple[tuple[float, float], tuple[float, float]]:
    spatial = np.asarray(metrics["spatial"], dtype=float)
    temporal = np.asarray(metrics["temporal"], dtype=float)
    measured_spatial = [
        np.asarray(item["spatial"], dtype=float) for item in tuning["measured"]
    ]
    measured_temporal = [
        np.asarray(item["temporal"], dtype=float) for item in tuning["measured"]
    ]
    sf_limits = (
        max(float(np.min(spatial)), *(float(np.min(v)) for v in measured_spatial)),
        min(float(np.max(spatial)), *(float(np.max(v)) for v in measured_spatial)),
    )
    tf_limits = (
        max(float(np.min(temporal)), *(float(np.min(v)) for v in measured_temporal)),
        min(float(np.max(temporal)), *(float(np.max(v)) for v in measured_temporal)),
    )
    if sf_limits[0] >= sf_limits[1] or tf_limits[0] >= tf_limits[1]:
        raise ValueError("tuning and power grids have no shared SF×TF domain")
    return sf_limits, tf_limits


def _draw_panel_d_tuning(
    subfigure,
    tuning: dict[str, object],
    metrics: dict[str, object],
    fits: pd.DataFrame,
) -> dict[str, object]:
    _panel_label(subfigure, "D")
    axes = subfigure.subplots(1, 2, gridspec_kw={"wspace": 0.26})
    spatial = np.asarray(metrics["spatial"], dtype=float)
    temporal = np.asarray(metrics["temporal"], dtype=float)
    sf_limits, tf_limits = _shared_frequency_limits(tuning, metrics)
    for column, (axis, unit, color, surface, measured) in enumerate(
        zip(
            axes,
            tuning["units"],
            ROLE_COLORS,
            metrics["surfaces"],
            tuning["measured"],
        )
    ):
        response = np.asarray(measured["response"], dtype=float)
        axis.pcolormesh(
            measured["spatial"],
            measured["temporal"],
            response / max(float(np.nanmax(response)), EPS),
            cmap="viridis",
            vmin=0,
            vmax=1,
            shading="nearest",
        )
        _overlay_passband(axis, spatial, temporal, np.asarray(surface), color)
        row = fits.loc[int(unit)]
        axis.scatter(
            float(row.preferred_sf_cpd),
            float(row.preferred_tf_hz),
            marker="*",
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        _frequency_axes(axis, show_y=column == 0)
        axis.set_xlim(*sf_limits)
        axis.set_ylim(*tf_limits)
    return {
        "units": list(map(int, tuning["units"])),
        "source_units": [
            int(fits.loc[int(unit)].get("source_unit_index", unit))
            for unit in tuning["units"]
        ],
        "shared_sf_limits": list(sf_limits),
        "shared_tf_limits": list(tf_limits),
        "passbands_share_authoritative_contour_code_with_panel_f": True,
    }


def _draw_panel_c_power(
    subfigure,
    tuning: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    """Show the two equal-mass conditional power distributions without tuning."""
    _panel_label(subfigure, "C")
    axes = subfigure.subplots(1, 2, gridspec_kw={"wspace": 0.24})
    spatial = np.asarray(metrics["spatial"], dtype=float)
    temporal = np.asarray(metrics["temporal"], dtype=float)
    sf_limits, tf_limits = _shared_frequency_limits(tuning, metrics)
    log_power = np.asarray(metrics["log_power"], dtype=float)
    low, high = map(float, metrics["display_limits"])
    names = tuple(metrics.get("regime_names", ("drift-rich", "rapid-transient")))
    event_groups = metrics.get("regime_selection") == "events"
    levels = np.linspace(low, high, 12)
    rendered = None
    for index in range(2):
        rendered = axes[index].contourf(
            spatial,
            temporal,
            log_power[index].T,
            levels=levels,
            cmap="magma",
            extend="both",
        )
        _frequency_axes(axes[index], show_y=index == 0)
        axes[index].set_xlim(*sf_limits)
        axes[index].set_ylim(*tf_limits)
        axes[index].set_title(names[index], fontsize=6.5)
    if not event_groups:
        axes[1].annotate(
            "power shifts to\nhigher TF",
            xy=(0.76, 0.79),
            xytext=(0.76, 0.40),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=5.2,
            fontweight="semibold",
            color="white",
            arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.2},
        )
    if rendered is None:
        raise RuntimeError("Panel C failed to render conditional power")
    colorbar = subfigure.colorbar(
        rendered,
        ax=list(axes),
        location="right",
        fraction=0.040,
        pad=0.025,
        aspect=18,
    )
    colorbar.set_label("log₁₀ conditional power density", fontsize=5.3)
    colorbar.set_ticks(np.arange(np.ceil(low), np.floor(high) + 1, 2))
    colorbar.ax.tick_params(labelsize=4.8, length=2)
    metadata_keys = (
        "regime_code",
        "per_trace_power_centroid_hz",
        "speed_deg_s",
        "path_length_arcmin",
        "microsaccade_count",
    )
    if missing := [key for key in metadata_keys if key not in metrics]:
        raise ValueError(f"Panel C lacks regime-selection metadata: {missing}")
    regime_code = np.asarray(metrics["regime_code"], dtype=int)
    centroid = np.asarray(metrics["per_trace_power_centroid_hz"], dtype=float)
    speed = np.asarray(metrics["speed_deg_s"], dtype=float)
    path_length = np.asarray(metrics["path_length_arcmin"], dtype=float)
    microsaccades = np.asarray(metrics["microsaccade_count"], dtype=int)
    if not all(
        len(values) == len(regime_code)
        for values in (centroid, speed, path_length, microsaccades)
    ):
        raise ValueError("Panel-C regime metadata arrays are not aligned")
    regime_reports = []
    for code, name in enumerate(names):
        keep = regime_code == code
        if not np.any(keep):
            raise ValueError(f"Panel C has no {name} fixation epochs")
        regime_reports.append(
            {
                "name": name,
                "n_fixation_epochs": int(np.sum(keep)),
                "median_tf_centroid_hz": float(np.median(centroid[keep])),
                "median_eye_speed_deg_s": float(np.median(speed[keep])),
                "median_path_length_arcmin": float(np.median(path_length[keep])),
                "microsaccade_positive_fraction": float(
                    np.mean(microsaccades[keep] > 0)
                ),
            }
        )
    return {
        "data_dependent": True,
        "source": "Kuang-factorized natural-image × filtered real-fixation ensemble",
        "conditions": list(names),
        "conditional_power_integrals": np.asarray(metrics["integrals"], dtype=float).tolist(),
        "equal_dynamic_mass_before_comparison": bool(
            np.allclose(np.asarray(metrics["integrals"], dtype=float), 1.0, atol=1e-6, rtol=0.0)
        ),
        "passband_contours_drawn": False,
        "selection_rule": (
            "audited event-free drift windows versus windows containing verified microsaccades below 1 degree"
            if event_groups else
            "within each animal, lower and upper quartiles of each filtered "
            "fixation's geometric temporal-frequency centroid after normalizing "
            "that fixation's TF>0 spectrum to unit mass"
        ),
        "selected_by_microsaccade_label": event_groups,
        "regimes": regime_reports,
        "annotation": "" if event_groups else "rapid-transient fixation dynamics redistribute conditional power toward higher temporal frequencies",
    }


def _draw_panel_f_contrast(
    subfigure,
    tuning: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    """Show the rapid/drift power contrast with both measured passbands."""
    _panel_label(subfigure, "F")
    axis = subfigure.subplots(1, 1)
    spatial = np.asarray(metrics["spatial"], dtype=float)
    temporal = np.asarray(metrics["temporal"], dtype=float)
    sf_limits, tf_limits = _shared_frequency_limits(tuning, metrics)
    limit = float(metrics["contrast_limit"])
    rendered = axis.contourf(
        spatial,
        temporal,
        np.asarray(metrics["contrast"], dtype=float).T,
        levels=np.linspace(-limit, limit, 13),
        cmap="RdBu_r",
        extend="both",
    )
    _overlay_passbands(axis, spatial, temporal, metrics["surfaces"])
    _frequency_axes(axis, show_y=True)
    axis.set_xlim(*sf_limits)
    axis.set_ylim(*tf_limits)
    colorbar = subfigure.colorbar(
        rendered,
        ax=axis,
        location="right",
        fraction=0.055,
        pad=0.035,
        aspect=18,
    )
    colorbar.set_label("log₂ microsaccade / drift" if metrics.get("regime_selection") == "events" else "log₂ rapid / drift", fontsize=5.3)
    colorbar.set_ticks([-limit, 0, limit])
    colorbar.set_ticklabels([f"{-limit:g}", "0", f"{limit:g}"])
    colorbar.ax.tick_params(labelsize=4.8, length=2)
    return {
        "units": list(map(int, tuning["units"])),
        "lasso_fraction": np.asarray(metrics["lasso_fraction"], dtype=float).tolist(),
        "routing_separation": float(metrics["routing_separation"]),
        "passbands_share_authoritative_contour_code_with_panel_d": True,
    }


def _draw_panel_e_population(
    subfigure,
    tuning_summary: pd.DataFrame,
    example_fits: pd.DataFrame,
    all_fits: pd.DataFrame,
    *,
    population_policy: str,
) -> dict[str, object]:
    all_units = str(population_policy) == "all_checkpoint_available"
    _panel_label(subfigure, "E")
    axis = subfigure.subplots(1, 1)
    required = {"unit_index", "exact_twin_yu_preferred_sf_cpd", "exact_twin_yu_preferred_tf_hz"}
    missing = required.difference(tuning_summary.columns)
    if missing:
        raise ValueError(f"Panel E tuning table lacks {sorted(missing)}")
    if all_units:
        if "included_in_exploratory_population" not in tuning_summary:
            raise ValueError("all-unit Panel E lacks its explicit inclusion flag")
        frame = tuning_summary.loc[
            tuning_summary.included_in_exploratory_population.astype(bool)
        ].copy()
    else:
        validated_required = {
            "audit_category",
            "validated_tuning",
            "validated_preferred_sf_cpd",
            "validated_preferred_tf_hz",
        }
        if missing := validated_required.difference(tuning_summary.columns):
            raise ValueError(f"validated Panel E lacks {sorted(missing)}")
        frame = tuning_summary.loc[
            tuning_summary.audit_category.eq("trusted")
            & tuning_summary.validated_tuning.astype(bool)
        ].copy()
        for validated, source in (
            ("validated_preferred_sf_cpd", "exact_twin_yu_preferred_sf_cpd"),
            ("validated_preferred_tf_hz", "exact_twin_yu_preferred_tf_hz"),
        ):
            if not np.allclose(frame[validated], frame[source], atol=0, rtol=0):
                raise ValueError(f"{validated} diverges from released Yu coordinate")
    sf = frame.exact_twin_yu_preferred_sf_cpd.to_numpy(dtype=float)
    tf = frame.exact_twin_yu_preferred_tf_hz.to_numpy(dtype=float)
    if not np.isfinite(np.column_stack((sf, tf))).all():
        raise ValueError("Panel E contains non-finite Yu coordinates")

    required_fit_columns = {
        "unit_index",
        "source_unit_index",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "selected_model",
        "optimizer_success",
        "full_support_r2",
        "sigma_s",
        "zeta_s",
        "sigma_t",
        "zeta_t",
        "q",
        "measured_min_sf_cpd",
        "measured_max_sf_cpd",
        "measured_min_tf_hz",
        "measured_max_tf_hz",
    }
    if missing := required_fit_columns.difference(all_fits.columns):
        raise ValueError(f"Panel E all-fit table lacks {sorted(missing)}")
    released_fits = all_fits.loc[all_fits.unit_index.isin(frame.unit_index)].copy()
    if set(released_fits.unit_index.astype(int)) != set(frame.unit_index.astype(int)):
        raise ValueError("Panel E contours and released tuning population differ")
    released_fits = released_fits.set_index("unit_index").loc[
        frame.unit_index.astype(int)
    ]
    if not all_units and not released_fits.optimizer_success.astype(bool).all():
        raise ValueError("Panel E cannot draw a non-converged Yu contour")
    fit_parameters = released_fits[
        ["sigma_s", "zeta_s", "sigma_t", "zeta_t", "q"]
    ].to_numpy(dtype=float)
    if not np.isfinite(fit_parameters).all():
        raise ValueError("Panel E cannot draw a non-finite Yu fit")
    if not np.allclose(
        released_fits.preferred_sf_cpd.to_numpy(dtype=float), sf, atol=1e-12, rtol=0
    ) or not np.allclose(
        released_fits.preferred_tf_hz.to_numpy(dtype=float), tf, atol=1e-12, rtol=0
    ):
        raise ValueError("Panel E contour fits diverge from released Yu centroids")
    if "source_unit_index" in frame and not np.array_equal(
        released_fits.source_unit_index.to_numpy(dtype=int),
        frame.source_unit_index.to_numpy(dtype=int),
    ):
        raise ValueError("Panel E contour source identities do not match centroids")

    spatial = np.geomspace(
        float(released_fits.measured_min_sf_cpd.min()),
        float(released_fits.measured_max_sf_cpd.max()),
        180,
    )
    temporal = np.geomspace(
        float(released_fits.measured_min_tf_hz.min()),
        float(released_fits.measured_max_tf_hz.max()),
        180,
    )
    density_count = np.zeros((len(spatial), len(temporal)), dtype=np.int32)
    contour_count = 0
    density_fill_count = 0
    for _, fit in released_fits.iterrows():
        surface = support_limited_yu_tuning_surface(fit, spatial, temporal)
        interior = np.asarray(surface >= PASSBAND_RESPONSE_FRACTION, dtype=bool)
        contributes = bool(np.any(interior))
        density_count += interior
        contour_count += int(contributes)
        density_fill_count += int(contributes)
    density_fraction = density_count.astype(float) / float(len(released_fits))
    maximum_density = float(np.max(density_fraction))
    if not maximum_density > 0:
        raise ValueError("Panel E passband occupancy density is empty")
    density_cmap = LinearSegmentedColormap.from_list(
        "passband_occupancy",
        [
            (1.0, 1.0, 1.0),
            (PASSBAND_DENSITY_DARKEST_GRAY,) * 3,
        ],
    )
    axis.pcolormesh(
        spatial,
        temporal,
        density_fraction.T,
        cmap=density_cmap,
        vmin=0.0,
        vmax=maximum_density,
        shading="gouraud",
        rasterized=True,
        zorder=1,
    )
    example_units = (
        example_fits.source_unit_index.to_numpy(dtype=int)
        if all_units
        else example_fits.index.to_numpy(dtype=int)
    )
    for unit, color in zip(example_units, ROLE_COLORS):
        if int(unit) not in released_fits.index:
            raise ValueError(f"exemplar u{unit:03d} is absent from Panel E contours")
        surface = support_limited_yu_tuning_surface(
            released_fits.loc[int(unit)], spatial, temporal
        )
        axis.contour(
            spatial,
            temporal,
            surface.T,
            levels=(PASSBAND_RESPONSE_FRACTION,),
            colors=("white",),
            linewidths=2.0,
            zorder=4,
        )
        axis.contour(
            spatial,
            temporal,
            surface.T,
            levels=(PASSBAND_RESPONSE_FRACTION,),
            colors=(color,),
            linewidths=1.05,
            zorder=5,
        )
    _frequency_axes(axis, show_y=True)
    axis.set_xlim(float(spatial.min()), float(spatial.max()))
    axis.set_ylim(float(temporal.min()), float(temporal.max()))
    return {
        "n_units": int(len(frame)),
        "unit_indices": frame.unit_index.astype(int).tolist(),
        "population_visualization": (
            "transparent filled overlay of every released Yu 55%-response "
            "passband; accumulated opacity depicts passband occupancy density"
        ),
        "passband_response_fraction": float(PASSBAND_RESPONSE_FRACTION),
        "n_passband_contours": int(contour_count),
        "n_filled_passbands": int(density_fill_count),
        "density_maximum_overlap_units": int(np.max(density_count)),
        "density_maximum_overlap_fraction": maximum_density,
        "density_darkest_gray": float(PASSBAND_DENSITY_DARKEST_GRAY),
        "density_definition": (
            "overlap count of released unit passband interiors; no KDE or "
            "additional smoothing"
        ),
        "density_rendering": (
            "single linear grayscale occupancy map with display interpolation; "
            "avoids backend quantization of hundreds of sub-percent alpha layers"
        ),
        "example_contours_highlighted": list(map(int, example_units)),
        "contours_share_authoritative_code_with_panels_d_and_f": True,
        "population_policy": str(population_policy),
        "n_optimizer_converged_fits": int(
            released_fits.optimizer_success.astype(bool).sum()
        ),
        "n_finite_nonconverged_fits": int(
            (~released_fits.optimizer_success.astype(bool)).sum()
        ),
    }


def _direct_mechanism_values(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scales = np.asarray(data["motion_scales"], dtype=float)
    stable_matches = np.flatnonzero(np.isclose(scales, 0.0))
    motion_matches = np.flatnonzero(np.isclose(scales, 1.0))
    if len(stable_matches) != 1 or len(motion_matches) != 1:
        raise ValueError("mechanism replay needs one stabilized and one measured condition")
    stable, motion = int(stable_matches[0]), int(motion_matches[0])
    rate = np.asarray(data["mean_rate"], dtype=float)
    spikes = np.asarray(data["expected_spikes"], dtype=float)
    ssi = np.asarray(data["map_ssi"], dtype=float)
    power = np.asarray(data["joint_passband_power"], dtype=float)
    stable_rate = np.nanmean(rate[:, :, stable], axis=0)
    motion_rate = np.nanmean(rate[:, :, motion], axis=0)
    rate_percent = 100.0 * (motion_rate - stable_rate) / np.maximum(stable_rate, EPS)
    stable_ssi = np.nansum(spikes[:, :, stable] * ssi[:, :, stable], axis=0) / np.maximum(
        np.nansum(spikes[:, :, stable], axis=0), EPS
    )
    motion_ssi = np.nansum(spikes[:, :, motion] * ssi[:, :, motion], axis=0) / np.maximum(
        np.nansum(spikes[:, :, motion], axis=0), EPS
    )
    ssi_percent = 100.0 * (motion_ssi - stable_ssi) / np.maximum(stable_ssi, EPS)
    passband_change = np.nanmedian(power[:, :, motion] - power[:, :, stable], axis=0)
    percentile = np.column_stack(
        [
            100.0 * (rankdata(passband_change[:, unit]) - 0.5) / passband_change.shape[0]
            for unit in range(passband_change.shape[1])
        ]
    )
    return {
        "passband_percentile": percentile,
        "rate_percent": rate_percent,
        "ssi_percent": ssi_percent,
    }


def _clustered_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray((10, 30, 50, 70, 90), dtype=float)
    unit_bin_values: list[list[np.ndarray]] = []
    for left, right in zip((0, 20, 40, 60, 80), (20, 40, 60, 80, 100)):
        unit_bin_values.append(
            [
                y[(x[:, unit] > left) & (x[:, unit] <= right), unit][
                    np.isfinite(y[(x[:, unit] > left) & (x[:, unit] <= right), unit])
                ]
                for unit in range(y.shape[1])
            ]
        )
    center = np.asarray(
        [
            np.nanmedian([np.median(value) if len(value) else np.nan for value in values])
            for values in unit_bin_values
        ]
    )
    draws = np.empty((int(n_bootstrap), len(centers)), dtype=float)
    rng = np.random.default_rng(int(seed))
    sampled_units = rng.integers(
        0,
        y.shape[1],
        size=(int(n_bootstrap), y.shape[1]),
    )
    # Preserve the original two-level bootstrap (units, then traces within
    # unit), but evaluate it in bounded vectorized batches so the all-unit
    # population does not create millions of Python-level loops.
    draw_batch_size = 32
    for bin_index, values in enumerate(unit_bin_values):
        counts = np.asarray([len(value) for value in values], dtype=int)
        max_count = int(counts.max(initial=0))
        if max_count == 0:
            draws[:, bin_index] = np.nan
            continue
        packed = np.full((y.shape[1], max_count), np.nan, dtype=float)
        for unit, value in enumerate(values):
            packed[unit, : len(value)] = value
        for start in range(0, int(n_bootstrap), draw_batch_size):
            stop = min(start + draw_batch_size, int(n_bootstrap))
            units = sampled_units[start:stop]
            selected = packed[units]
            selected_counts = counts[units]
            random_index = np.floor(
                rng.random(selected.shape) * np.maximum(selected_counts[..., None], 1)
            ).astype(int)
            resampled = np.take_along_axis(selected, random_index, axis=2)
            valid_slot = (
                np.arange(max_count, dtype=int)[None, None, :]
                < selected_counts[..., None]
            )
            resampled[~valid_slot] = np.nan
            unit_medians = np.nanmedian(resampled, axis=2)
            draws[start:stop, bin_index] = np.nanmedian(unit_medians, axis=1)
    return centers, center, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _draw_panel_g_boundary(
    subfigure,
    data: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    _panel_label(subfigure, "G")
    values = _direct_mechanism_values(data)
    x = values["passband_percentile"]
    axes = subfigure.subplots(1, 2, gridspec_kw={"wspace": 0.42})
    reports: dict[str, object] = {}
    shared_limits: list[tuple[float, float]] = []
    for index, (axis, key, title, color) in enumerate(
        zip(
            axes,
            ("rate_percent", "ssi_percent"),
            ("firing-rate change", "single-spike information change"),
            (EFFECT_GREEN, PURPLE),
        )
    ):
        y = values[key]
        valid = np.isfinite(x) & np.isfinite(y)
        centers, center, low, high = _clustered_curve(
            x, y, n_bootstrap=int(n_bootstrap), seed=int(seed) + index
        )
        unit_distributions = []
        for left, right in zip((0, 20, 40, 60, 80), (20, 40, 60, 80, 100)):
            medians = np.asarray(
                [
                    np.nanmedian(y[(x[:, unit] > left) & (x[:, unit] <= right), unit])
                    for unit in range(y.shape[1])
                ],
                dtype=float,
            )
            unit_distributions.append(medians[np.isfinite(medians)])
        axis.boxplot(
            unit_distributions,
            positions=centers,
            widths=12.0,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.12, "linewidth": 0.7},
            medianprops={"color": color, "alpha": 0.42, "linewidth": 0.8},
            whiskerprops={"color": color, "alpha": 0.24, "linewidth": 0.65},
            capprops={"color": color, "alpha": 0.24, "linewidth": 0.65},
            zorder=1,
        )
        axis.errorbar(
            centers,
            center,
            yerr=np.vstack((center - low, high - center)),
            fmt="o-",
            color=color,
            lw=1.6,
            ms=4.0,
            capsize=2.2,
        )
        axis.axhline(0, color="0.5", lw=0.75)
        # Preserve a small internal margin so the terminal percentile tick is
        # not clipped when the two compact axes are composed into Panel G.
        axis.set_xlim(-2, 102)
        lower = min(0.0, float(np.nanmin(low)))
        upper = max(0.0, float(np.nanmax(high)))
        padding = max(2.0, 0.10 * (upper - lower))
        axis.set_ylim(lower - padding, upper + padding)
        shared_limits.append((lower - padding, upper + padding))
        axis.set_xlabel("passband-power percentile")
        axis.set_ylabel(f"{title}\nmotion − stabilized (%)")
        axis.grid(axis="y", alpha=0.15)
        correlation = spearmanr(x[valid], y[valid])
        reports[key] = {
            "spearman_rho": float(correlation.statistic),
            "spearman_p": float(correlation.pvalue),
            "binned_center": center.tolist(),
            "binned_ci95": np.column_stack((low, high)).tolist(),
            "n_finite_unit_trace_pairs": int(np.count_nonzero(valid)),
            "n_unit_medians_per_bin": [int(len(value)) for value in unit_distributions],
        }
    shared_y = (
        min(limit[0] for limit in shared_limits),
        max(limit[1] for limit in shared_limits),
    )
    for axis in axes:
        axis.set_ylim(*shared_y)
    return {
        "n_units": int(len(np.asarray(data["unit_indices"]))),
        "x_definition": "within-unit percentile of measured-minus-stabilized joint passband power, median across matched images",
        "y_definition": (
            "direct measured-motion minus stabilized modulation, expressed as percent "
            "of the matched stabilized firing rate or single-spike information"
        ),
        "fraction_of_total_effect_explained": False,
        "claim_boundary": (
            "descriptive within-unit association; values are response modulation, "
            "not variance explained or a decomposition of the total effect"
        ),
        "visual_summary": "boxes show distributions of unit-specific bin medians; colored curves show the unit-clustered population median and bootstrap interval",
        "shared_y_limits_percent": list(shared_y),
        **reports,
    }


def _draw_panel_h_normalized(
    subfigure,
    trajectory: dict[str, np.ndarray],
    trajectory_summary: dict[str, object],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Show gain-invariant modulation and sharpening across the readout."""
    _panel_label(subfigure, "H")
    stage_names = np.asarray(trajectory["stage_names"]).astype(str)
    has_phase = trajectory_summary.get("readout_trajectory", {}).get("has_phase_branch", True)
    expected_stages = np.asarray(("S1 + phase" if has_phase else "S1", "+ S2", "+ S3 / output"))
    if not np.array_equal(stage_names, expected_stages):
        raise ValueError(f"Panel-H cumulative stages changed: {stage_names.tolist()}")
    unit_temporal = np.asarray(
        trajectory["unit_temporal_modulation_points"], dtype=float
    )
    unit_ssi = np.asarray(
        trajectory["unit_ssi_delta_bits_per_spike"], dtype=float
    )
    if unit_temporal.shape != unit_ssi.shape or unit_temporal.ndim != 2:
        raise ValueError("Panel-H unit trajectories must align as [stage, unit]")
    if unit_temporal.shape[0] != len(stage_names):
        raise ValueError("Panel-H stage labels and unit trajectories disagree")
    if np.any(~np.isfinite(unit_temporal)) or np.any(~np.isfinite(unit_ssi)):
        raise ValueError("Panel-H released unit trajectories must be finite")

    def bootstrap(values: np.ndarray, offset: int):
        center = np.median(values, axis=1)
        rng = np.random.default_rng(int(seed) + int(offset))
        units = rng.integers(
            0,
            values.shape[1],
            size=(int(n_bootstrap), values.shape[1]),
        )
        draws = np.median(values[:, units], axis=2)
        return (
            center,
            np.quantile(draws, 0.025, axis=1),
            np.quantile(draws, 0.975, axis=1),
        )

    axes = subfigure.subplots(1, 2, gridspec_kw={"wspace": 0.72})
    x = np.arange(len(stage_names), dtype=float)
    reports: dict[str, object] = {}
    for index, (axis, values, name, ylabel, color) in enumerate(
        zip(
            axes,
            (unit_temporal, unit_ssi),
            ("temporal_modulation", "spatial_sharpening"),
            (
                "motion − stabilized\n(% of mean response)",
                "motion − stabilized\n(bits/spike)",
            ),
            (EFFECT_GREEN, PURPLE),
        )
    ):
        center, low, high = bootstrap(values, 1000 * index)
        box = axis.boxplot(
            [row for row in values],
            positions=x,
            widths=0.46,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        for artist in box["boxes"]:
            artist.set(
                facecolor=color,
                edgecolor=color,
                alpha=0.16,
                linewidth=0.7,
            )
        for artist in (*box["whiskers"], *box["caps"]):
            artist.set(color=color, alpha=0.48, linewidth=0.65)
        for artist in box["medians"]:
            artist.set(color=color, linewidth=1.05)
        axis.errorbar(
            x,
            center,
            yerr=np.vstack((center - low, high - center)),
            fmt="o-",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            lw=1.55,
            ms=4.0,
            capsize=2.0,
            zorder=4,
        )
        axis.set_xticks(
            x,
            (stage_names[0], "+ S2", "output"),
            rotation=18,
            ha="right",
        )
        axis.axhline(0, color="0.5", lw=0.7)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("cumulative readout")
        axis.grid(axis="y", alpha=0.15)
        lower = min(0.0, float(np.nanpercentile(values, 2)))
        upper = max(0.0, float(np.nanpercentile(values, 98)))
        padding = max(0.01, 0.10 * (upper - lower))
        axis.set_ylim(lower - padding, upper + padding)
        reports[name] = {
            "center": center.tolist(),
            "ci95": np.column_stack((low, high)).tolist(),
            "n_units": int(values.shape[1]),
        }

    return {
        "n_images": int(trajectory_summary["n_images"]),
        "n_traces": int(trajectory_summary["n_traces"]),
        "n_image_trace_pairs": int(trajectory_summary["n_image_trace_pairs"]),
        "n_units": int(unit_temporal.shape[1]),
        "cumulative_stage_labels": stage_names.tolist(),
        "synthetic_network_reference": False,
        "affine_or_tangent_model": False,
        "final_stage_is_ordinary_model": bool(
            trajectory_summary.get("readout_trajectory", {}).get(
                "final_stage_is_ordinary_model", False
            )
        ),
        "ordinary_output_max_abs": float(
            trajectory_summary.get("identity_checks", {}).get(
                "ordinary_output_max_abs", np.inf
            )
        ),
        "cached_G_output_checks": trajectory_summary.get(
            "cached_G_output_checks", {}
        ),
        "visualization": (
            "unit-distribution boxes and unit-bootstrap medians for gain-invariant "
            "temporal modulation and absolute spatial-information sharpening"
        ),
        "normalization": (
            "temporal modulation subtracts each spatial location's time mean, RMSes "
            "the residual, and divides by the movie-wide mean; SSI uses the "
            "mean-normalized spatial rate map"
        ),
        "mean_rate_gain_plotted": False,
        "intermediate_definition": (
            f"trained output logits accumulated by {stage_names[0].replace(' + ', ' plus ')}, S2, and S3 feature "
            "groups, with the ordinary softplus applied after each cumulative sum"
        ),
        **reports,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _model_identity(spec: dict[str, object]) -> tuple[str, str]:
    """Read label and checkpoint digest from the canonical production spec."""
    checkpoint = spec.get("checkpoint", {})
    if isinstance(checkpoint, dict):
        digest = str(checkpoint.get("sha256", ""))
    else:
        digest = str(spec.get("checkpoint_sha256", ""))
    label = str(spec.get("label", ""))
    if not label or len(digest) != 64:
        raise ValueError("model spec lacks a label or valid checkpoint SHA-256")
    return label, digest


def main() -> int:
    args = parse_args()
    configure()
    manuscript_layout = args.layout == "manuscript"
    page_size = MANUSCRIPT_PAGE_SIZE if manuscript_layout else PAGE_SIZE
    panel_layout = MANUSCRIPT_PANEL_LAYOUT if manuscript_layout else PANEL_LAYOUT
    model_spec = yaml.safe_load(args.model_spec.read_text(encoding="utf-8"))
    required = (
        args.panel_a_audit / "selected_example.npz",
        args.panel_a_audit / "summary.json",
        args.panel_b_reduction / "binned_curves.csv",
        args.panel_b_reduction / "unit_effects.npz",
        args.panel_b_reduction / "summary.json",
        args.tuning_table,
        args.tuning_summary,
        args.tuning_summary.parent / "summary.json",
        args.example_fits,
        args.all_fits,
        args.rucci_ensemble / "rucci_ensemble_power.npz",
        args.stage_trajectory / "top_passband_stage_trajectory.npz",
        args.stage_trajectory / "summary.json",
        *args.population_shards,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing revised Figure-4 inputs:\n" + "\n".join(map(str, missing)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    panel_a, panel_a_summary = _load_panel_a_audit(args.panel_a_audit)
    panel_b = pd.read_csv(args.panel_b_reduction / "binned_curves.csv")
    panel_b_unit_effects = _load_npz(
        args.panel_b_reduction / "unit_effects.npz"
    )
    panel_b_summary = json.loads(
        (args.panel_b_reduction / "summary.json").read_text(encoding="utf-8")
    )
    tuning_table = pd.read_csv(args.tuning_table)
    tuning_summary = pd.read_csv(args.tuning_summary).sort_values("unit_index")
    tuning_contract_summary = json.loads(
        (args.tuning_summary.parent / "summary.json").read_text(encoding="utf-8")
    )
    fits = pd.read_csv(args.example_fits).set_index("unit_index")
    fits = fits.loc[[int(fits.index[fits.role.eq(role)][0]) for role in ROLE_NAMES]]
    all_fits = pd.read_csv(args.all_fits)
    if args.population_policy == "all_checkpoint_available":
        # The validated contract numbers examples in its compact 145-unit view,
        # whereas the all-unit assay uses the checkpoint's 725 native indices.
        # Re-identify the same physical units through the explicit source-unit
        # key, then take every fit parameter from the all-unit Yu release.
        example_roles = fits.reset_index()[["source_unit_index", "role"]]
        if example_roles.source_unit_index.duplicated().any():
            raise ValueError("all-unit exemplar source identities are not unique")
        fits = all_fits.merge(
            example_roles,
            on="source_unit_index",
            how="inner",
            validate="one_to_one",
        ).set_index("unit_index")
        fits = fits.loc[[int(fits.index[fits.role.eq(role)][0]) for role in ROLE_NAMES]]
        if len(fits) != len(ROLE_NAMES):
            raise ValueError("all-unit Yu release is missing a selected exemplar")
    ensemble = _load_npz(args.rucci_ensemble / "rucci_ensemble_power.npz")
    population = load_and_merge_shards(args.population_shards)
    population_shard_summaries = _load_shard_summaries(args.population_shards)
    if args.population_policy == "all_checkpoint_available":
        if str(tuning_contract_summary.get("population_policy", "")) != "all_checkpoint_available":
            raise ValueError("all-unit figure requires an explicit all-unit tuning contract")
        analysis_units = tuning_summary.loc[
            tuning_summary.included_in_exploratory_population.astype(bool), "unit_index"
        ].astype(int).tolist()
    else:
        analysis_units = tuning_summary.loc[
            tuning_summary.audit_category.eq("trusted")
            & tuning_summary.validated_tuning.astype(bool),
            "unit_index",
        ].astype(int).tolist()
    population = select_population_units(population, analysis_units)
    trajectory = _load_npz(
        args.stage_trajectory / "top_passband_stage_trajectory.npz"
    )
    trajectory_summary = json.loads(
        (args.stage_trajectory / "summary.json").read_text(encoding="utf-8")
    )
    trajectory_units = int(
        np.asarray(trajectory["unit_temporal_modulation_points"]).shape[1]
    )
    if trajectory_units != len(analysis_units):
        raise ValueError(
            "Panel H and the selected figure population differ: "
            f"{trajectory_units} stage-trajectory units versus "
            f"{len(analysis_units)} analysis units"
        )
    model_label, expected_digest = _model_identity(model_spec)
    provenance_summaries = {
        "Panel A": panel_a_summary,
        "Yu tuning contract": tuning_contract_summary,
        "Panel B": panel_b_summary,
        "Panel H": trajectory_summary,
        **{
            f"spectral replay shard {index}": summary
            for index, summary in enumerate(population_shard_summaries)
        },
    }
    wrong_checkpoint = {
        name: _summary_checkpoint_digest(summary)
        for name, summary in provenance_summaries.items()
        if _summary_checkpoint_digest(summary) != expected_digest
    }
    if wrong_checkpoint:
        raise ValueError(
            "cross-panel checkpoint chain failed: "
            + ", ".join(f"{name}={digest!r}" for name, digest in wrong_checkpoint.items())
        )
    if args.population_policy == "all_checkpoint_available":
        coordinate_assay = str(tuning_contract_summary.get("coordinate_assay", ""))
        contract_units = tuning_contract_summary.get("unit_indices", [])
    else:
        tuning_contract = tuning_contract_summary.get("trusted_tuning_contract", {})
        coordinate_assay = str(tuning_contract.get("coordinate_assay", ""))
        contract_units = tuning_contract.get("unit_indices", [])
    if coordinate_assay != "exact_cid_yu_sf_tf":
        raise ValueError("centroid and passband coordinates are not exact-CID Yu fits")
    if set(map(int, contract_units)) != set(analysis_units):
        raise ValueError("Yu contract and selected tuning rows differ")
    panel_b_units = int(panel_b_summary["n_units"])
    if panel_b_units < len(analysis_units):
        raise ValueError(
            "Panel B population cannot be smaller than the validated tuning population"
        )
    if set(map(int, population["unit_indices"])) != set(analysis_units):
        raise ValueError("mechanism replay and tuning-summary populations differ")

    exemplar_tuning = _exemplar_tuning(tuning_table, fits)
    mechanism = routing_metrics(ensemble, fits)
    reports: dict[str, object] = {}
    drawers = {
        "A": (draw_panel_a, (panel_a,), {}, (0.015, 0.95, 0.12, 0.91)),
        "B": (
            _draw_panel_b,
            (panel_b, panel_b_unit_effects),
            {},
            (0.09, 0.99, 0.18, 0.88),
        ),
        "C": (
            _draw_panel_c_power,
            (exemplar_tuning, mechanism),
            {},
            (0.10, 0.965, 0.19, 0.84),
        ),
        "D": (
            _draw_panel_d_tuning,
            (exemplar_tuning, mechanism, fits),
            {},
            (0.12, 0.985, 0.19, 0.84),
        ),
        "E": (
            _draw_panel_e_population,
            (tuning_summary, fits, all_fits),
            {"population_policy": str(args.population_policy)},
            (0.19, 0.97, 0.17, 0.82),
        ),
        "F": (
            _draw_panel_f_contrast,
            (exemplar_tuning, mechanism),
            {},
            (0.15, 0.96, 0.19, 0.84),
        ),
        "G": (
            _draw_panel_g_boundary,
            (population,),
            {"n_bootstrap": int(args.n_bootstrap), "seed": int(args.seed)},
            (0.13, 0.985, 0.20, 0.82),
        ),
        "H": (
            _draw_panel_h_normalized,
            (trajectory, trajectory_summary),
            {
                "n_bootstrap": int(args.n_bootstrap),
                "seed": int(args.seed) + 20,
            },
            (0.14, 0.97, 0.18, 0.82),
        ),
    }
    panel_paths: dict[str, Path] = {}
    manuscript_margins = {
        "A": (0.025, 0.96, 0.12, 0.88),
        "B": (0.16, 0.985, 0.24, 0.76),
        "C": (0.14, 0.84, 0.24, 0.82),
        "D": (0.14, 0.98, 0.24, 0.82),
        "E": (0.28, 0.95, 0.24, 0.83),
        "F": (0.28, 0.77, 0.24, 0.83),
        "G": (0.18, 0.985, 0.26, 0.85),
        "H": (0.18, 0.985, 0.26, 0.85),
    }
    for label, (draw, positional, keyword, margins) in drawers.items():
        path = panels_dir / f"panel_{label.lower()}.pdf"
        panel_paths[label] = path
        reports[label] = _render_panel(
            path,
            panel_layout[label][2:],
            draw,
            *positional,
            margins=manuscript_margins[label] if manuscript_layout else margins,
            text_replacements=MANUSCRIPT_LABELS if manuscript_layout else None,
            **keyword,
        )
    pdf = args.out_dir / "figure4.pdf"
    png = args.out_dir / "figure4.png"
    svg = args.out_dir / "figure4.svg"
    _compose_page(
        pdf,
        [(panel_paths[label], *panel_layout[label][:2]) for label in "ABCDEFGH"],
        page_width_in=page_size[0],
        page_height_in=page_size[1],
    )
    _render_page_png(pdf, png)
    executable = shutil.which("pdftocairo")
    if executable is not None:
        subprocess.run(
            [executable, "-svg", str(pdf), str(svg)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    trajectory_pairs = int(trajectory_summary.get("n_image_trace_pairs", 0))
    trajectory_inference_ready = trajectory_pairs >= 100
    report = {
        "artifact_status": (
            "production candidate" if trajectory_inference_ready else
            "smoke preview; Panel H requires at least 100 matched movies"
        ),
        "model_label": model_label,
        "checkpoint_sha256": expected_digest,
        "layout": args.layout,
        "page_size_inches": list(page_size),
        "panel_layout_inches": {key: list(value) for key, value in panel_layout.items()},
        "analysis_population_units": len(analysis_units),
        "panel_populations": {
            "A": "one audited exact-CID exemplar",
            "B": {
                "n_units": panel_b_units,
                "selection": "all checkpoint-available exact-CID readouts",
            },
            "D_E_F_G": {
                "n_units": len(analysis_units),
                "selection": (
                    "all checkpoint-available units with finite exact-CID Yu fits"
                    if args.population_policy == "all_checkpoint_available"
                    else "units passing the prespecified SFxTF release audit"
                ),
            },
            "H": {
                "n_units": trajectory_units,
                "selection": (
                    "all checkpoint-available exact-CID units with unit-specific "
                    "top-passband trace membership"
                    if args.population_policy == "all_checkpoint_available"
                    else "validated exact-CID units with unit-specific top-passband "
                    "trace membership"
                ),
            },
        },
        "population_policy": str(args.population_policy),
        "chain_of_custody": {
            "all_model_panels_match_checkpoint": True,
            "input_checkpoint_digests": {
                name: _summary_checkpoint_digest(summary)
                for name, summary in provenance_summaries.items()
            },
            "tuning_coordinate_assay": coordinate_assay,
            "unit_identity_match_across_tuning_response_and_mechanism": True,
            "panel_b_uses_tuning_quality_gate": False,
        },
        "panel_b": panel_b_summary,
        "panel_h": {
            "n_images": int(trajectory_summary["n_images"]),
            "n_traces": int(trajectory_summary["n_traces"]),
            "n_timepoints": int(trajectory_summary["n_scored_timepoints_per_pair"]),
            "n_image_trace_pairs": trajectory_pairs,
            "inference_ready": trajectory_inference_ready,
        },
        "panels": reports,
        "panel_narrative": {
            "C": "filtered fixation dynamics redistribute natural-image power toward higher temporal frequencies",
            "D": "measured Yu SFxTF tuning defines example-unit passbands",
            "E": "the rapid-minus-drift power contrast intersects those passbands differently",
            "F": "validated foveal units span SFxTF tuning space",
            "G": (
                "passband engagement covaries descriptively with both firing-rate and "
                "single-spike-information modulation; it is not an explained-fraction plot"
            ),
            "H": (
                "top-passband movies generate gain-invariant temporal modulation and "
                "spatial sharpening across cumulative trained readout branches"
            ),
        },
        "figure_pdf": str(pdf.resolve()),
        "figure_png": str(png.resolve()),
        "production_figure_pdf": str(pdf.resolve()),
        "production_figure_png": str(png.resolve()),
        "production_figure_svg": (
            str(svg.resolve()) if svg.exists() else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
