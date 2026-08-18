import numpy as np
import torch
import json
from pathlib import Path
import pandas as pd

from paper.fig4.spatiotemporal_tuning.audit_native_core_motion_path import (
    filter_joint_spectra,
    json_ready,
    percent_curve,
    require_matching_controlled_model,
    selected_causal_histories,
    spatial_information_components,
    temporal_filter_drive_from_moments,
)
from paper.fig4.spatiotemporal_tuning.render_native_motion_activation_gallery import (
    controlled_trace_on_model_grid,
    map_ssi,
    measured_unit_tuning_surface,
    pooled_pair_effect,
    representative_pair,
    representative_units,
)
from paper.fig4.upstream.real_trace_matrix.model import (
    _trace_xy_to_twin_helper_order,
    make_counterfactual_stim,
)


def test_selected_causal_histories_hold_initial_gaze_and_use_newest_first():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    trace = np.zeros((3, 2), dtype=np.float32)
    result = selected_causal_histories(
        image,
        trace,
        np.asarray([0, 2]),
        n_lags=3,
        out_size=(3, 3),
        ppd=1.0,
    )
    assert result.shape == (2, 1, 3, 3, 3)
    torch.testing.assert_close(result[0, 0, 0], result[0, 0, 1])
    torch.testing.assert_close(result[0, 0, 1], result[0, 0, 2])


def test_temporal_filter_drive_is_zero_for_static_and_tracks_variance():
    static = torch.full((4, 2, 3, 3), 2.0, dtype=torch.float64)
    static_drive = temporal_filter_drive_from_moments(
        static.sum(dim=0), static.square().sum(dim=0), len(static)
    )
    np.testing.assert_allclose(static_drive, 0.0, atol=0.0)

    moving = static.clone()
    moving[:, 0] += torch.arange(4, dtype=torch.float64)[:, None, None]
    moving_drive = temporal_filter_drive_from_moments(
        moving.sum(dim=0), moving.square().sum(dim=0), len(moving)
    )
    assert moving_drive[0] > 0
    assert moving_drive[1] == 0


def test_joint_filter_spectra_are_normalized_and_frequency_calibrated():
    class Core:
        @staticmethod
        def effective_temporal_weight():
            # Two deliberately different finite 3-D filters.
            value = torch.zeros(2, 1, 5, 3, 3)
            value[0, 0, :, 1, 1] = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0])
            value[1, 0, 2, 1, :] = torch.tensor([-1.0, 0.0, 1.0])
            return value

    result = filter_joint_spectra(Core(), ppd=40.0)
    assert result["joint_power"].shape == (2, 129, 96)
    np.testing.assert_allclose(result["joint_power"].sum(axis=(1, 2)), 1.0)
    assert result["temporal_frequency_hz"][-1] == 120.0
    assert np.all(np.isfinite(result["median_tf_hz"]))
    assert np.all(np.isfinite(result["median_sf_cpd"]))


def test_selected_causal_histories_match_production_renderer_exactly():
    rng = np.random.default_rng(9)
    image = rng.normal(size=(11, 13)).astype(np.float32)
    trace = np.asarray(
        [[0.00, 0.00], [0.04, -0.02], [0.07, 0.03], [-0.01, 0.06], [0.02, -0.04]],
        dtype=np.float32,
    )
    endpoints = np.asarray([0, 2, 4], dtype=np.int64)
    n_lags = 4
    out_size = (7, 9)
    ppd = 6.5

    selected = selected_causal_histories(
        image,
        trace,
        endpoints,
        n_lags=n_lags,
        out_size=out_size,
        ppd=ppd,
    )
    full_stack = np.repeat(image[None], len(trace) + n_lags - 1, axis=0)
    helper_order = torch.from_numpy(_trace_xy_to_twin_helper_order(trace))
    production = make_counterfactual_stim(
        full_stack,
        helper_order,
        ppd=ppd,
        n_lags=n_lags,
        out_size=out_size,
    )
    torch.testing.assert_close(selected, production.index_select(0, torch.from_numpy(endpoints)))


def test_spatial_information_components_are_zero_for_uniform_maps():
    numerator, denominator = spatial_information_components(torch.ones(2, 3, 4, 5))
    np.testing.assert_allclose(numerator, 0.0)
    np.testing.assert_allclose(denominator, 1.0)


def test_percent_curve_pools_all_axes_before_scale():
    numerator = np.asarray([[[1.0, 2.0]], [[3.0, 9.0]]])
    denominator = np.ones_like(numerator)
    np.testing.assert_allclose(percent_curve(numerator, denominator), [0.0, 175.0])


def test_core_motion_path_rejects_checkpoint_mismatch(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    dataset = tmp_path / "dataset.yaml"
    checkpoint.write_bytes(b"checkpoint-a")
    dataset.write_bytes(b"dataset-a")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model": {"model": {
            "checkpoint_sha256": "not-the-model-hash",
            "dataset_configs_sha256": "not-the-dataset-hash",
        }}
    }))
    try:
        require_matching_controlled_model(manifest, checkpoint, dataset)
    except RuntimeError as error:
        assert "checkpoints differ" in str(error)
    else:
        raise AssertionError("checkpoint mismatch was accepted")


def test_core_motion_path_report_provenance_is_json_serializable():
    payload = {
        "path": Path("checkpoint.ckpt"),
        "array": np.asarray([1, 2]),
        "scalar": np.float32(0.5),
    }
    assert json.loads(json.dumps(json_ready(payload))) == {
        "path": "checkpoint.ckpt",
        "array": [1, 2],
        "scalar": 0.5,
    }


def test_activation_gallery_map_ssi_is_zero_for_uniform_maps():
    np.testing.assert_allclose(map_ssi(np.ones((2, 3, 4))), 0.0)


def test_activation_gallery_expands_retained_trace_to_model_grid():
    trace = np.asarray(
        [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]], dtype=np.float32
    )
    output = controlled_trace_on_model_grid(
        trace,
        {
            "time_contract": {
                "source_trace_rate_hz": 120,
                "model_output_rate_hz": 240,
            }
        },
        archive_source_rate_hz=120,
        model_output_rate_hz=240,
        torch=torch,
    )
    assert output.shape == (6, 2)
    np.testing.assert_allclose(output[[1, 3, 5]], trace)


def test_activation_gallery_tuning_surface_uses_orientation_envelope():
    rows = []
    for sf in (1.0, 2.0):
        for tf in (4.0, 8.0):
            for orientation, offset in ((0.0, 0.0), (90.0, 10.0)):
                rows.append({
                    "unit_index": 3,
                    "spatial_cpd": sf,
                    "temporal_hz": tf,
                    "probe_orientation_deg": orientation,
                    "response_amp_rms": sf + tf + offset,
                })
    sf, tf, surface = measured_unit_tuning_surface(pd.DataFrame(rows), 3)
    np.testing.assert_array_equal(sf, [1.0, 2.0])
    np.testing.assert_array_equal(tf, [4.0, 8.0])
    assert surface.shape == (2, 2)
    assert surface[-1, -1] == 1.0


def test_activation_gallery_pair_and_unit_selection_are_deterministic():
    ssi = np.ones((2, 2, 2, 4), dtype=float)
    expected = np.ones_like(ssi)
    # Pair effects are 10%, 20%, 30%, and 40%; the lower median tie wins in
    # row-major order. Unit deltas on that pair are strictly increasing.
    for flat, percent in enumerate((10.0, 20.0, 30.0, 40.0)):
        row = np.unravel_index(flat, (2, 2))
        ssi[row[0], row[1], 1] = 1.0 + percent / 100.0
    effect = pooled_pair_effect(ssi, expected, 1)
    assert representative_pair(effect) == (0, 1)

    ssi[0, 1, 1] = np.asarray([1.1, 1.2, 1.3, 1.4])
    units = representative_units(
        ssi,
        expected,
        0,
        1,
        1,
        np.asarray([0.0, 0.5, 1.0]),
    )
    np.testing.assert_array_equal(units, [0, 2, 3])
