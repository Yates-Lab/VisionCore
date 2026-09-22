import json

import numpy as np
import pandas as pd
import pytest

from paper.fig4.spatiotemporal_tuning.build_rucci_ensemble_power import (
    json_ready,
    load_filtered_fixation_bank,
    normalize_dynamic_power_per_trace,
    production_spectral_grid,
    radial_temporal_quadrature,
    trajectory_phase_power_budget_per_trace,
)
from paper.fig4.spatiotemporal_tuning.validate_retinal_spectrum import (
    direct_mode_power,
    distribution_metrics,
    periodic_translation_movie,
    trajectory_phase_power,
)
from paper.fig4.spatiotemporal_tuning.spectral_power import frequency_grid
from paper.fig4.upstream.real_trace_matrix.model import OUT_SIZE, PPD


def _write_fixation_bank(path, *, filter_kind="zero-phase analysis IIR"):
    path.mkdir()
    manifest = {
        "target_rate_hz": 240.0,
        "analysis_samples": 12,
        "filter": {
            "kind": filter_kind,
            "passband_hz": 20.0,
            "stopband_hz": 30.0,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    table = pd.DataFrame(
        {
            "trace_index": np.arange(4),
            "session": ["Allen_a", "Allen_b", "Logan_a", "Logan_b"],
            "analysis_speed_mean_deg_s": [1.0, 2.0, 3.0, 4.0],
            "analysis_path_length_deg": [0.1, 0.2, 0.3, 0.4],
            "saved_microsaccade_count": [0, 1, 0, 1],
        }
    )
    table.to_csv(path / "trace_table.csv", index=False)
    np.save(path / "trace_xy_filtered.npy", np.zeros((4, 12, 2), dtype=np.float32))


def test_filtered_bank_contract_rejects_non_zero_phase_source(tmp_path):
    bank = tmp_path / "bad"
    _write_fixation_bank(bank, filter_kind="unfiltered")
    with pytest.raises(ValueError, match="zero_phase_filter"):
        load_filtered_fixation_bank(bank, analysis_samples=12, requested_traces=0)


def test_summary_metadata_converts_numpy_scalars_to_strict_json():
    payload = {
        "gate": np.bool_(True),
        "count": np.int64(7),
        "nested": [np.float32(0.5)],
    }
    encoded = json.dumps(json_ready(payload), allow_nan=False)
    assert json.loads(encoded) == {"gate": True, "count": 7, "nested": [0.5]}


def test_filtered_bank_contract_preserves_trace_identity(tmp_path):
    bank = tmp_path / "good"
    _write_fixation_bank(bank)
    _, table, traces, rows = load_filtered_fixation_bank(
        bank, analysis_samples=12, requested_traces=2
    )
    assert traces.shape == (2, 12, 2)
    assert len(np.unique(table.session.str.split("_").str[0])) == 2
    assert np.array_equal(table.trace_index.to_numpy(), rows)


def test_analytic_power_grid_is_dense_but_stays_inside_measured_support(tmp_path):
    table = tmp_path / "tuning.csv"
    pd.DataFrame(
        {"spatial_cpd": np.geomspace(1.07156475, 15.93952562, 9)}
    ).to_csv(table, index=False)
    spatial, orientation = production_spectral_grid(table)
    assert len(spatial) == 25
    assert np.isclose(spatial[0], 1.07156475)
    assert np.isclose(spatial[-1], 12.0)
    assert len(orientation) == 8
    assert np.allclose(orientation, np.arange(8) * 22.5)


def test_equal_mass_maps_integrate_to_one_and_preserve_centroid_order():
    spatial = np.asarray([1.0, 2.0, 4.0])
    temporal = np.asarray([1.0, 2.0, 4.0, 8.0])
    power = np.ones((2, len(spatial), len(temporal)))
    power[0, :, 2:] = 0.01
    power[1, :, :2] = 0.01
    distribution, mass, centroid = normalize_dynamic_power_per_trace(
        power, spatial, temporal
    )
    weights = radial_temporal_quadrature(spatial, temporal)
    assert np.allclose(np.sum(distribution * weights[None], axis=(1, 2)), 1.0)
    assert np.all(mass > 0)
    assert centroid[0] < centroid[1]


def test_complete_carrier_budget_cannot_create_power():
    rng = np.random.default_rng(3)
    traces = np.cumsum(rng.normal(scale=0.001, size=(3, 24, 2)), axis=1)
    static, dynamic = trajectory_phase_power_budget_per_trace(
        traces, np.asarray([1.0, 2.0, 4.0]), n_directions=6
    )
    assert np.allclose(static + dynamic, 1.0, atol=1e-14, rtol=0.0)
    assert np.all((static >= 0) & (static <= 1))
    assert np.all((dynamic >= 0) & (dynamic <= 1))


def test_periodic_renderer_matches_complete_phase_carrier():
    rng = np.random.default_rng(4)
    frame = rng.normal(127.0, 25.0, OUT_SIZE)
    time = np.arange(32) / 240.0
    trace = np.column_stack(
        (0.04 * np.sin(2 * np.pi * 7 * time), 0.03 * np.cos(2 * np.pi * 5 * time))
    )
    grid = frequency_grid()
    temporal, phase = trajectory_phase_power(
        trace[None],
        grid["kxy"],
        frame_rate_hz=240.0,
        mode_chunk_size=1024,
    )
    movie = periodic_translation_movie(frame, trace, ppd=float(PPD))
    rendered_temporal, actual = direct_mode_power(
        movie, grid["flat_index"], frame_rate_hz=240.0
    )
    base = np.fft.fft2((frame - 127.0) / 255.0, norm="ortho")
    ideal = (
        np.square(np.abs(base.ravel()[grid["flat_index"]]))[:, None] * phase[0]
    )
    metrics = distribution_metrics(actual, ideal)
    assert np.array_equal(rendered_temporal, temporal)
    assert metrics["distribution_cosine"] > 0.999999
    assert metrics["distribution_total_variation"] < 1e-5
