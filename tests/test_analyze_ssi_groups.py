from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.model_selection.analyze_m66_ssi_retinal_motion import (
    build_sf_groups,
    render_summary_figure,
)


def test_ssi_audit_auto_prefers_selected_twin_table_tertiles():
    units = pd.DataFrame(
        {
            "unit_index": np.arange(8),
            "sf_group": [
                "low_sf",
                "middle_sf",
                "high_sf",
                "inactive",
                "low_sf",
                "middle_sf",
                "high_sf",
                "inactive",
            ],
            "sf_split_metric": np.linspace(0.1, 0.8, 8),
        }
    )
    groups, definition = build_sf_groups(
        units,
        Path("robust.csv"),
        mode="auto",
    )
    np.testing.assert_array_equal(np.flatnonzero(groups["lower_sf"]), [0, 4])
    np.testing.assert_array_equal(np.flatnonzero(groups["higher_sf"]), [2, 6])
    assert definition["mode"] == "table_tertiles"
    assert definition["middle_sf_omitted_from_group_contrast"] == 2


def test_ssi_audit_table_tertiles_fails_without_explicit_labels():
    units = pd.DataFrame(
        {"unit_index": [0, 1], "sf_group": ["inactive", "middle_sf"]}
    )
    with pytest.raises(ValueError, match="requires low_sf and high_sf"):
        build_sf_groups(units, None, mode="table_tertiles")


def test_ssi_audit_summary_figure_renders_causal_and_dose_panels(tmp_path):
    groups = {
        name: {
            "percent_vs_stabilized": estimate,
            "percent_ci95_image_bootstrap": [estimate - 1.0, estimate + 1.0],
        }
        for name, estimate in (("all", 3.0), ("lower_sf", 2.0), ("higher_sf", 4.0))
    }
    image_rows = []
    path_rows = []
    for group_index, group in enumerate(groups):
        for image_index in range(4):
            stable = 0.02 + 0.001 * image_index
            moving = stable * (1.0 + 0.02 + 0.01 * group_index)
            image_rows.append(
                {
                    "group": group,
                    "stabilized_ssi_bits_per_spike": stable,
                    "moving_ssi_bits_per_spike": moving,
                    "ssi_percent_vs_stabilized": 100.0 * (moving - stable) / stable,
                }
            )
        for path_bin in range(3):
            estimate = 1.0 + path_bin + group_index
            path_rows.append(
                {
                    "group": group,
                    "path_bin": path_bin + 1,
                    "path_median_arcmin": 20.0 + 20.0 * path_bin,
                    "ssi_percent_vs_stabilized": estimate,
                    "ci95_low": estimate - 0.5,
                    "ci95_high": estimate + 0.5,
                }
            )
    output = tmp_path / "ssi.png"
    render_summary_figure(
        groups,
        pd.DataFrame(image_rows),
        pd.DataFrame(path_rows),
        output=output,
        model_label="M77",
    )
    assert output.exists() and output.stat().st_size > 0
    assert output.with_suffix(".pdf").exists()
