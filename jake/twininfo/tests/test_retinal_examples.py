from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from jake.twininfo.retinal_examples import (
    aligned_current_retinal_movie,
    aligned_model_lag_cubes,
    local_phase_scramble_roi,
    model_crop_centers_px,
    model_lag_cubes_from_image_trace,
    retinal_movie_from_image_trace,
    pyramid_local_image_controls,
    select_trace_examples,
    _phase_scramble_pyramid_coeffs,
)
from jake.twininfo.lagcube_information import block_endpoint_lag_cubes


def test_aligned_current_retinal_movie_uses_lag_zero_and_skips_history_frame():
    retinal = np.zeros((6, 1, 3, 2, 2), dtype=np.float32)
    for t in range(retinal.shape[0]):
        retinal[t, 0, 0] = 100 + t
        retinal[t, 0, 2] = 200 + t

    movie = aligned_current_retinal_movie(retinal, t_max=4)

    assert movie.shape == (4, 2, 2)
    assert np.all(movie[:, 0, 0] == np.array([101, 102, 103, 104], dtype=np.float32))


def test_aligned_model_lag_cubes_use_lag_zero_as_current_frame():
    retinal = np.zeros((6, 1, 3, 2, 2), dtype=np.float32)
    for t in range(retinal.shape[0]):
        for lag in range(retinal.shape[2]):
            retinal[t, 0, lag] = 100 * t + lag

    cubes = aligned_model_lag_cubes(retinal, t_max=4, n_lags=3)

    assert cubes.shape == (4, 3, 2, 2)
    assert cubes[0, 0, 0, 0] == 100
    assert cubes[0, 2, 0, 0] == 102


def test_model_crop_centers_match_model_axis_flip_convention():
    trace = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    centers = model_crop_centers_px(trace, (101, 201), ppd=10.0)

    assert np.allclose(centers[0], [100.0, 50.0])
    assert np.allclose(centers[1], [80.0, 60.0])


def test_model_crop_centers_apply_source_image_crop_offset():
    trace = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    centers = model_crop_centers_px(
        trace,
        (101, 201),
        ppd=10.0,
        crop_center_offset_px=(25.0, -5.0),
    )

    assert np.allclose(centers[0], [125.0, 45.0])
    assert np.allclose(centers[1], [105.0, 55.0])


def test_retinal_movie_source_crop_offset_moves_sampled_region():
    y, x = np.mgrid[:240, :260]
    image = (x + 10 * y).astype(np.float32)
    trace = np.zeros((40, 2), dtype=np.float32)

    centered = retinal_movie_from_image_trace(image, trace, t_max=40)
    shifted = retinal_movie_from_image_trace(
        image,
        trace,
        t_max=40,
        crop_center_offset_px=(30.0, 0.0),
    )

    assert centered.shape == shifted.shape == (40, 151, 151)
    assert float(np.mean(shifted - centered)) > 20.0


def test_stabilized_trace_is_constant_across_all_embedded_lags():
    rng = np.random.default_rng(3)
    y, x = np.mgrid[:240, :260]
    image = (127.0 + 35.0 * np.sin(x / 13.0) + 20.0 * np.cos(y / 17.0)).astype(np.float32)
    moving_trace = rng.normal(scale=0.02, size=(64, 2)).astype(np.float32)
    stable_trace = np.repeat(moving_trace.mean(axis=0, keepdims=True), moving_trace.shape[0], axis=0)

    cubes = model_lag_cubes_from_image_trace(image, stable_trace, t_max=64)
    blocks, _current = block_endpoint_lag_cubes(cubes)

    assert float(np.max(np.abs(cubes - cubes[:1]))) == 0.0
    assert float(np.max(np.abs(blocks - blocks[:, :1]))) == 0.0


def test_local_phase_scramble_roi_contains_all_trace_crops():
    trace = np.array([[0.0, 0.0], [0.1, -0.2], [-0.1, 0.15]], dtype=np.float32)
    roi = local_phase_scramble_roi(trace, (240, 260), ppd=20.0, out_size=(41, 41), margin_px=10)
    centers = model_crop_centers_px(trace, (240, 260), ppd=20.0)

    assert roi["roi_x0"] <= np.floor(np.min(centers[:, 0] - 41 / 2.0))
    assert roi["roi_x1"] >= np.ceil(np.max(centers[:, 0] + 41 / 2.0))
    assert roi["roi_y0"] <= np.floor(np.min(centers[:, 1] - 41 / 2.0))
    assert roi["roi_y1"] >= np.ceil(np.max(centers[:, 1] + 41 / 2.0))


def test_pyramid_local_controls_change_only_roi_and_preserve_band_energy():
    pytest.importorskip("plenoptic")
    rng = np.random.default_rng(9)
    y, x = np.mgrid[:180, :200]
    image = (
        127.0
        + 35.0 * np.sin(x / 7.0)
        + 20.0 * np.cos(y / 11.0)
        + 8.0 * rng.normal(size=(180, 200))
    ).astype(np.float32)
    trace = np.array([[0.0, 0.0], [0.05, -0.05], [-0.05, 0.04]], dtype=np.float32)

    controls, audits = pyramid_local_image_controls(
        image,
        trace,
        rng,
        out_size=(41, 41),
        margin_px=12,
    )

    assert set(controls) == {"pyramid_phase_scrambled", "sf_low", "sf_mid", "sf_high"}
    assert {row["control"] for row in audits} == set(controls)
    phase_audit = next(row for row in audits if row["control"] == "pyramid_phase_scrambled")
    y0, y1 = int(phase_audit["roi_y0"]), int(phase_audit["roi_y1"])
    x0, x1 = int(phase_audit["roi_x0"]), int(phase_audit["roi_x1"])
    outside = np.ones(image.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False

    for control, out in controls.items():
        assert out.shape == image.shape
        assert np.all(out >= float(np.min(image)))
        assert np.all(out <= float(np.max(image)))
        assert np.allclose(out[outside], image[outside])
        assert not np.allclose(out[y0:y1, x0:x1], image[y0:y1, x0:x1])
        audit = next(row for row in audits if row["control"] == control)
        assert audit["outside_roi_changed_fraction"] == 0.0
        assert np.isfinite(float(audit["pyramid_reconstruction_relative_error"]))
        assert float(audit["pyramid_reconstruction_relative_error"]) < 1e-3

    band_rows = [row for row in audits if str(row["control"]).startswith("sf_")]
    assert sum(float(row["band_energy_fraction"]) for row in band_rows) > 0.5


def test_four_level_pyramid_controls_expose_four_sf_bands():
    pytest.importorskip("plenoptic")
    rng = np.random.default_rng(10)
    y, x = np.mgrid[:180, :200]
    image = (
        127.0
        + 30.0 * np.sin(x / 5.0)
        + 25.0 * np.cos(y / 17.0)
        + 6.0 * rng.normal(size=(180, 200))
    ).astype(np.float32)
    trace = np.zeros((8, 2), dtype=np.float32)

    controls, audits = pyramid_local_image_controls(
        image,
        trace,
        rng,
        out_size=(41, 41),
        margin_px=12,
        height=4,
        sf_bands=("sf_low", "sf_mid_low", "sf_mid_high", "sf_high"),
    )

    assert set(controls) == {
        "pyramid_phase_scrambled",
        "sf_low",
        "sf_mid_low",
        "sf_mid_high",
        "sf_high",
    }
    assert {row["control"] for row in audits} == set(controls)


def test_pyramid_phase_scramble_preserves_residuals_and_complex_amplitudes():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    real_residual = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    complex_band = torch.complex(
        torch.ones((1, 1, 3, 3), dtype=torch.float32),
        torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3) + 1,
    )
    coeffs = OrderedDict([
        ("residual_highpass", real_residual),
        ((0, 0), complex_band),
        ("residual_lowpass", real_residual + 10.0),
    ])

    scrambled, mag_error, n_complex = _phase_scramble_pyramid_coeffs(coeffs, rng)

    assert n_complex == 1
    assert mag_error < 1e-6
    assert torch.allclose(scrambled["residual_highpass"], coeffs["residual_highpass"])
    assert torch.allclose(scrambled["residual_lowpass"], coeffs["residual_lowpass"])
    assert torch.allclose(torch.abs(scrambled[(0, 0)]), torch.abs(coeffs[(0, 0)]))
    assert not torch.allclose(scrambled[(0, 0)], coeffs[(0, 0)])


def test_trace_example_selection_splits_fixations_and_microsaccades():
    t = np.arange(100, dtype=np.float32)
    fixation = np.column_stack([0.0002 * t, 0.0001 * t]).astype(np.float32)
    microsaccade = fixation.copy()
    microsaccade[50:, 0] += 0.3
    eye_traces = np.stack([fixation, microsaccade], axis=0)
    durations = np.array([100, 100])

    examples = select_trace_examples(
        eye_traces,
        durations,
        t_max=40,
        n_each=1,
        seed=0,
        stride=5,
    )

    by_kind = {example.kind: example for example in examples}
    assert by_kind["fixation"].n_events_in_window == 0
    assert by_kind["fixation"].event_onset is None
    assert by_kind["microsaccade"].n_events_in_window == 1
    assert by_kind["microsaccade"].event_onset is not None
