import numpy as np

from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    _surface_prediction,
    fit_local_quadratic_peak,
    fit_log_gaussian_surface,
    recommended_native_grid,
)
from paper.fig4.spatiotemporal_tuning.spectral_power import (
    circular_orientation_weights,
    folded_dpss_mode_power,
    interpolate_tuning_temporal,
    log_interpolation_weights,
    spectral_predictors,
)


def test_recommended_grid_is_cycle_valid_and_below_nyquist() -> None:
    grid = recommended_native_grid()
    assert grid["minimum_cycles_across_aperture"] >= 1.0
    assert grid["minimum_pixels_per_spatial_cycle"] > 2.0
    assert grid["minimum_frames_per_temporal_cycle"] > 2.0
    assert grid["minimum_analyzed_cycles"] >= 2.0
    assert grid["spatial_cpd"][-1] < grid["spatial_nyquist_cpd"]
    assert grid["dynamic_temporal_hz"][-1] < grid["temporal_nyquist_hz"]


def test_recommended_grid_removes_subcycle_bins_for_native_35px_crop() -> None:
    grid = recommended_native_grid(image_size=35)
    assert grid["spatial_cpd"][0] > 1.0
    assert np.isclose(grid["minimum_cycles_across_aperture"], 1.0)
    assert grid["spatial_grid_anchor"] == "one cycle across the model input aperture"
    assert np.max(np.diff(np.log2(grid["spatial_cpd"]))) <= 0.5 + 1e-12
    assert grid["minimum_pixels_per_spatial_cycle"] > 2.0
    assert len(grid["spatial_cpd"]) >= 5


def test_log_gaussian_fit_recovers_interior_continuous_peak() -> None:
    spatial = 0.5 * np.sqrt(2.0) ** np.arange(11)
    temporal = np.sqrt(2.0) ** np.arange(14)
    xx, yy = np.meshgrid(np.log2(spatial), np.log2(temporal))
    target_sf, target_tf = 3.1, 13.2
    params = np.asarray(
        [
            0.08,
            np.log(1.7),
            np.log2(target_sf),
            np.log2(target_tf),
            np.log(0.8),
            np.log(1.0),
            0.2,
        ]
    )
    response = _surface_prediction(params, xx, yy)
    fit = fit_log_gaussian_surface(spatial, temporal, response)
    assert fit["fit_status"] == "ok"
    assert fit["fit_r2"] > 0.999
    assert np.isclose(fit["preferred_sf_cpd"], target_sf, rtol=0.01)
    assert np.isclose(fit["preferred_tf_hz"], target_tf, rtol=0.01)
    assert not fit["peak_censored"]


def test_boundary_peak_is_never_reported_as_uncensored() -> None:
    spatial = 0.5 * np.sqrt(2.0) ** np.arange(11)
    temporal = np.sqrt(2.0) ** np.arange(14)
    xx, yy = np.meshgrid(np.log2(spatial), np.log2(temporal))
    params = np.asarray(
        [
            0.05,
            np.log(1.0),
            np.log2(spatial[-1]),
            np.log2(temporal[-1]),
            np.log(0.7),
            np.log(0.7),
            0.0,
        ]
    )
    fit = fit_log_gaussian_surface(spatial, temporal, _surface_prediction(params, xx, yy))
    assert fit["discrete_peak_boundary"]
    assert fit["peak_censored"]


def test_local_quadratic_peak_recovers_joint_interior_maximum() -> None:
    spatial = np.sqrt(2.0) ** np.arange(7)
    temporal = np.sqrt(2.0) ** np.arange(10)
    xx, yy = np.meshgrid(np.log2(spatial), np.log2(temporal))
    target_sf, target_tf = 3.6, 18.5
    response = 0.1 + np.exp(
        -0.5
        * (
            np.square((xx - np.log2(target_sf)) / 0.72)
            + np.square((yy - np.log2(target_tf)) / 0.84)
            - 0.35
            * (xx - np.log2(target_sf))
            * (yy - np.log2(target_tf))
        )
    )
    peak = fit_local_quadratic_peak(spatial, temporal, response)
    assert peak["peak_status"] == "ok"
    assert peak["local_fit_r2"] > 0.9
    assert abs(np.log2(peak["preferred_sf_cpd"] / target_sf)) < 0.12
    assert abs(np.log2(peak["preferred_tf_hz"] / target_tf)) < 0.12


def test_local_peak_tracks_observed_maximum_with_a_shoulder() -> None:
    spatial = np.sqrt(2.0) ** np.arange(7)
    temporal = np.sqrt(2.0) ** np.arange(13)
    xx, yy = np.meshgrid(np.log2(spatial), np.log2(temporal))
    narrow = np.exp(
        -0.5
        * (
            np.square((xx - np.log2(2.8)) / 0.55)
            + np.square((yy - np.log2(22.0)) / 0.52)
        )
    )
    broad = 0.72 * np.exp(
        -0.5
        * (
            np.square((xx - np.log2(2.0)) / 1.2)
            + np.square((yy - np.log2(5.0)) / 2.0)
        )
    )
    response = 0.04 + narrow + broad
    global_fit = fit_log_gaussian_surface(spatial, temporal, response)
    local_peak = fit_local_quadratic_peak(spatial, temporal, response)
    assert local_peak["peak_status"] == "ok"
    assert local_peak["preferred_tf_hz"] > global_fit["preferred_tf_hz"]
    assert abs(np.log2(local_peak["preferred_tf_hz"] / 22.0)) < 0.3


def test_local_peak_rejects_a_boundary_maximum() -> None:
    spatial = np.sqrt(2.0) ** np.arange(7)
    temporal = np.sqrt(2.0) ** np.arange(10)
    peak = fit_local_quadratic_peak(spatial, temporal, np.add.outer(temporal, spatial))
    assert peak["peak_status"] == "boundary"
    assert peak["peak_censored"]


def test_spectral_grid_interpolates_log_frequency_and_orientation() -> None:
    lower, upper, low, high, valid = log_interpolation_weights(
        np.asarray([np.sqrt(2.0)]), np.asarray([1.0, 2.0, 4.0])
    )
    assert valid[0]
    assert (lower[0], upper[0]) == (0, 1)
    np.testing.assert_allclose([low[0], high[0]], [0.5, 0.5])
    o0, o1, w0, w1 = circular_orientation_weights(
        np.asarray([[0.0, 2.0]]), np.asarray([0.0, 45.0, 90.0, 135.0])
    )
    assert (o0[0], o1[0]) == (0, 1)
    np.testing.assert_allclose([w0[0], w1[0]], [1.0, 0.0])


def test_folded_dpss_power_recovers_constant_translation_frequency() -> None:
    rate_hz = 128.0
    time = np.arange(128, dtype=float) / rate_hz
    coefficient = np.exp(-2j * np.pi * 8.0 * time)[None]
    frequency, power = folded_dpss_mode_power(coefficient, rate_hz)
    assert frequency[np.argmax(power[0])] == 8.0


def test_temporal_interpolation_is_zero_outside_measured_grid() -> None:
    source = np.asarray([1.0, 2.0, 4.0, 8.0])
    target = np.asarray([0.5, 2.0, np.sqrt(8.0), 16.0])
    tuning = np.zeros((1, 1, len(source), 1), dtype=float)
    tuning[0, 0, :, 0] = [0.0, 1.0, 3.0, 0.0]
    result = interpolate_tuning_temporal(
        tuning, source, target, normalize=False
    )[0, 0, :, 0]
    assert result[0] == 0.0
    assert result[-1] == 0.0
    assert result[1] > 0.0
    assert result[2] > result[1]


def test_spectral_predictor_preserves_joint_sf_tf_pairing() -> None:
    power = np.zeros((2, 2, 1), dtype=float)
    power[0, 0, 0] = 1.0
    power[1, 1, 0] = 1.0
    matched = np.zeros((1, 2, 2, 1), dtype=float)
    matched[0, 0, 0, 0] = 0.5
    matched[0, 1, 1, 0] = 0.5
    crossed = np.zeros_like(matched)
    crossed[0, 0, 1, 0] = 0.5
    crossed[0, 1, 0, 0] = 0.5
    matched_score = spectral_predictors(power, matched, matched)
    crossed_score = spectral_predictors(power, crossed, crossed)
    assert matched_score["joint_passband_power"][0] > crossed_score[
        "joint_passband_power"
    ][0]
