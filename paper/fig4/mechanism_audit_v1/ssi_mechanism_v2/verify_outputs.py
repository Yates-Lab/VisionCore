#!/usr/bin/env python3
"""Verify the saved exact Figure 4 SSI mechanism products."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import write_json


OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/ssi_mechanism_v2"
DATA = OUT / "plot_data"
RAW = OUT / "exact_arrays"
PHASE = ROOT / "outputs/figures/fig4/mechanism_audit_v1/phase_spatial_followup_v1"
EPS = 1e-12


def main() -> int:
    metrics = pd.read_csv(DATA / "exact_rate_map_metric_components.csv.gz")
    curves = pd.read_csv(DATA / "exact_ssi_decomposition_curves.csv")
    null_table = pd.read_csv(DATA / "association_matched_null_statistics.csv")
    selection = pd.read_csv(DATA / "figure4_convgru_channel_selection.csv")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    checks: dict[str, object] = {}

    checks["n_images"] = int(metrics.image_index.nunique())
    checks["n_trajectories"] = int(metrics.trace_index.nunique())
    checks["n_scales"] = int(metrics.scale.nunique())
    checks["n_conditions"] = int(metrics.condition.nunique())
    checks["n_groups"] = int(metrics.figure4_sf_group.nunique())
    assert checks["n_images"] == 8
    assert checks["n_trajectories"] == 24
    assert checks["n_scales"] == 5
    assert checks["n_conditions"] == 8
    assert checks["n_groups"] == 2
    assert np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()

    metric_columns = [
        "expected_spikes",
        "mean_rate_sum",
        "n_rate_values",
        "ssi_weighted_numerator",
        "cv2_weighted_numerator",
        "quadratic_bits_weighted_numerator",
    ]
    stable = metrics.loc[metrics.condition.eq("normal_stable")]
    stable_spread = stable.groupby(
        ["image_index", "trace_index", "figure4_sf_group"]
    )[metric_columns].agg(lambda x: float(np.ptp(x.to_numpy(float))))
    checks["max_stable_spread_across_nominal_scales"] = float(stable_spread.to_numpy().max())
    assert checks["max_stable_spread_across_nominal_scales"] < 1e-12

    zero = metrics.loc[metrics.scale.eq(0)].copy()
    anchor = zero.loc[zero.condition.eq("normal_moving")].set_index(
        ["image_index", "trace_index", "figure4_sf_group"]
    )
    zero_errors = []
    for condition, frame in zero.groupby("condition"):
        indexed = frame.set_index(["image_index", "trace_index", "figure4_sf_group"])
        zero_errors.append(
            float(np.max(np.abs(indexed[metric_columns].to_numpy(float) - anchor[metric_columns].to_numpy(float))))
        )
    checks["max_zero_motion_intervention_component_error"] = max(zero_errors)
    assert checks["max_zero_motion_intervention_component_error"] < 1e-12

    prior = pd.read_csv(PHASE / "plot_data/exact_final_ssi_curves.csv")
    group_map = {"low SF": "low", "high SF": "high"}
    prior["figure4_sf_group"] = prior.sf_group.map(group_map)
    current = curves.loc[curves.condition.eq("normal_moving"), [
        "figure4_sf_group", "scale", "ssi_percent_vs_stable"
    ]]
    comparison = prior.merge(current, on=["figure4_sf_group", "scale"], validate="one_to_one")
    checks["max_endpoint_reproduction_error_percentage_points"] = float(
        np.max(np.abs(comparison.ssi_percent_vs_0x - comparison.ssi_percent_vs_stable))
    )
    assert checks["max_endpoint_reproduction_error_percentage_points"] < 5e-4

    selected_association = {}
    for group in ("low", "high"):
        group_selection = selection.loc[selection.figure4_sf_group.eq(group)]
        observed = float(
            group_selection.loc[group_selection.selected_motion_matched, "mean_squared_readout_weight"].sum()
        )
        ratios = null_table.loc[
            null_table.figure4_sf_group.eq(group), "association_ratio_vs_observed"
        ].to_numpy(float)
        checks[f"{group}_n_unique_null_masks"] = int(
            null_table.loc[null_table.figure4_sf_group.eq(group), "null_mask_index"].nunique()
        )
        checks[f"{group}_null_association_ratio_min"] = float(ratios.min())
        checks[f"{group}_null_association_ratio_max"] = float(ratios.max())
        selected_association[group] = observed
        assert checks[f"{group}_n_unique_null_masks"] == 63
        assert ratios.min() >= 0.9 - 1e-12 and ratios.max() <= 1.1 + 1e-12

    with np.load(RAW / "representative_rate_map_decomposition.npz") as maps:
        map_arrays = [maps[key] for key in maps.files if key.endswith("_map")]
        checks["representative_map_shapes"] = sorted({tuple(value.shape) for value in map_arrays})
        checks["all_representative_maps_finite"] = bool(
            all(np.isfinite(value).all() for value in map_arrays)
        )
        assert checks["representative_map_shapes"] == [(51, 51)]
        assert checks["all_representative_maps_finite"]

    checks["selection_used_natural_image_ssi"] = bool(
        manifest["selection_used_natural_image_ssi"]
    )
    assert not checks["selection_used_natural_image_ssi"]
    required = [
        "figure_4_ssi_mechanism.png",
        "figure_4_ssi_mechanism.pdf",
        "figure_4_ssi_mechanism.svg",
        "figure_s_ssi_mechanism_controls.png",
        "figure_s_ssi_mechanism_controls.pdf",
        "figure_s_ssi_mechanism_controls.svg",
        "REPORT.md",
        "statistics.json",
    ]
    checks["required_products_present"] = all((OUT / name).is_file() for name in required)
    assert checks["required_products_present"]
    checks["status"] = "pass"
    write_json(OUT / "verification.json", checks)
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
