from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from paper.fig4.final_mechanism_panels import build_final_mechanism_panels as panels


def _write_retinal_cache(path, *, include_phase: bool) -> None:
    payload = {
        "scales": np.asarray([1.0, 3.0]),
        "spatial_cpd": np.asarray([1.0, 2.0]),
        "temporal_hz": np.asarray([3.0, 6.0]),
        "orientation_deg": np.asarray([0.0]),
        "selected_image_index": np.asarray([0]),
        "selected_trace_index": np.asarray([0]),
        "definition": np.asarray("test"),
        "instantaneous_ft_occupancy": np.ones((2, 2, 2, 1)),
    }
    if include_phase:
        payload["trajectory_phase_modulation_power"] = np.asarray(
            [
                [[[1.0], [2.0]], [[3.0], [4.0]]],
                [[[4.0], [3.0]], [[2.0], [1.0]]],
            ]
        )
        payload["method_version"] = np.asarray("trajectory_phase_test")
    np.savez(path, **payload)


def test_panel_d_rejects_instantaneous_velocity_histogram_as_power(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "instantaneous_only.npz"
    _write_retinal_cache(source, include_phase=False)
    monkeypatch.setattr(panels, "FEM_SOURCE", source)
    with pytest.raises(RuntimeError, match="not a temporal PSD"):
        panels.load_panel_d()


def test_panel_d_uses_complete_trajectory_phase_spectrum(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "phase.npz"
    _write_retinal_cache(source, include_phase=True)
    monkeypatch.setattr(panels, "FEM_SOURCE", source)
    table, metadata = panels.load_panel_d()
    assert "trajectory_phase_power_percent_in_grid" in table
    np.testing.assert_allclose(
        table.groupby("movement_scale").trajectory_phase_power_percent_in_grid.sum(),
        [100.0, 100.0],
    )
    assert metadata["method_version"] == "trajectory_phase_test"


def test_panel_e_claim_is_limited_to_coarse_peak_ordering() -> None:
    table = pd.DataFrame(
        {
            "sf_group": ["lower SF"] * 5 + ["higher SF"] * 5,
            "scale": [0.0, 0.5, 1.0, 2.0, 3.0] * 2,
            "prediction_normalized": [0.0, 0.81, 0.92, 0.99, 1.0, 0.0, 0.94, 1.0, 0.995, 0.955],
            "ssi_percent_vs_stabilized": [0.0, 3.5, 13.9, 23.6, 21.8, 0.0, 7.3, 11.8, 2.9, -9.0],
            "ssi_percent_ci95_low": [0.0] * 9 + [-16.5],
            "ssi_percent_ci95_high": [0.0] * 9 + [-3.4],
        }
    )
    result = panels.summarize_panel_e_claim(table)
    assert result["highest_sampled_overlap_scale"] == {"lower SF": 3.0, "higher SF": 1.0}
    assert result["highest_sampled_ssi_scale"] == {"lower SF": 2.0, "higher SF": 1.0}
    assert result["higher_sf_at_3x"]["prediction_normalized"] > 0.9
    assert result["higher_sf_at_3x"]["ssi_percent_vs_stabilized"] < 0.0
    assert "does not predict the useful movement range" in result["claim_boundary"]


def test_panel_e_title_and_caption_do_not_overclaim() -> None:
    source = inspect.getsource(panels)
    assert "Spatiotemporal overlap captures coarse peak ordering" in source
    assert "Measured spatiotemporal tuning predicts the useful movement range" not in source
    assert "continued predicted engagement" in source
    assert "observed reversal" in source
    assert "cannot predict a below-stabilization reversal by construction" in source
