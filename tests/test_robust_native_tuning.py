import json

import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning.robust_native_tuning import (
    _surface_prediction,
    fit_local_quadratic_peak,
    fit_log_gaussian_surface,
    recommended_native_grid,
)
from paper.fig4.spatiotemporal_tuning.run_native_periodic_tuning import (
    periodic_histories,
    periodic_response_metrics,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (
    native_kinematic_occupancy,
    overlap_table,
)
from paper.fig4.spatiotemporal_tuning.analyze_image_specific_joint_engagement import (
    circular_orientation_weights,
    image_unit_engagement,
    interpolate_tuning_to_fft_bins,
    log_interpolation_weights,
    trajectory_phase_spectra,
    validate_matrix_contract,
    validate_native240_trace_contract,
)
from paper.fig4.spatiotemporal_tuning.analyze_direct_rendered_joint_engagement import (
    engagement_from_cubes,
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
    # The last anti-alias guard point may be slightly closer than half an
    # octave; no adjacent probe bins may be farther apart than half an octave.
    assert np.max(np.diff(np.log2(grid["spatial_cpd"]))) <= 0.5 + 1e-12
    assert grid["minimum_pixels_per_spatial_cycle"] > 2.0
    assert len(grid["spatial_cpd"]) >= 5


def test_log_gaussian_fit_recovers_interior_continuous_peak() -> None:
    spatial = 0.5 * np.sqrt(2.0) ** np.arange(11)
    temporal = np.sqrt(2.0) ** np.arange(14)
    xx, yy = np.meshgrid(np.log2(spatial), np.log2(temporal))
    target_sf, target_tf = 3.1, 13.2
    params = np.asarray(
        [0.08, np.log(1.7), np.log2(target_sf), np.log2(target_tf), np.log(0.8), np.log(1.0), 0.2]
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
        [0.05, np.log(1.0), np.log2(spatial[-1]), np.log2(temporal[-1]), np.log(0.7), np.log(0.7), 0.0]
    )
    response = _surface_prediction(params, xx, yy)

    fit = fit_log_gaussian_surface(spatial, temporal, response)

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


def test_local_peak_tracks_observed_maximum_when_global_surface_has_a_shoulder() -> None:
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
    broad_shoulder = 0.72 * np.exp(
        -0.5
        * (
            np.square((xx - np.log2(2.0)) / 1.2)
            + np.square((yy - np.log2(5.0)) / 2.0)
        )
    )
    response = 0.04 + narrow + broad_shoulder

    global_fit = fit_log_gaussian_surface(spatial, temporal, response)
    local_peak = fit_local_quadratic_peak(spatial, temporal, response)

    assert local_peak["peak_status"] == "ok"
    assert local_peak["preferred_tf_hz"] > global_fit["preferred_tf_hz"]
    assert abs(np.log2(local_peak["preferred_tf_hz"] / 22.0)) < 0.3


def test_local_peak_rejects_a_boundary_maximum() -> None:
    spatial = np.sqrt(2.0) ** np.arange(7)
    temporal = np.sqrt(2.0) ** np.arange(10)
    response = np.add.outer(temporal, spatial)

    peak = fit_local_quadratic_peak(spatial, temporal, response)

    assert peak["peak_status"] == "boundary"
    assert peak["peak_censored"]


def test_periodic_histories_have_exact_temporal_phase_advance() -> None:
    phases = np.asarray([0.0, np.pi / 2])
    histories = periodic_histories(
        spatial_cpd=2.0,
        temporal_hz=30.0,
        bar_orientation_deg=0.0,
        phases_rad=phases,
        n_lags=3,
        input_size=5,
        ppd=10.0,
        frame_rate_hz=240.0,
        contrast=0.25,
        device="cpu",
    ).numpy()

    assert histories.shape == (2, 1, 3, 5, 5)
    # At the center pixel spatial phase is zero; one lag advances backward by pi/4.
    np.testing.assert_allclose(
        histories[0, 0, :, 2, 2],
        0.25 * np.cos([0, -np.pi / 4, -np.pi / 2]),
        atol=1e-7,
    )


def test_periodic_response_metrics_recovers_first_harmonic() -> None:
    phase = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rates = np.column_stack((2.0 + 0.7 * np.cos(phase), 1.0 + 0.4 * np.sin(phase)))

    metrics = periodic_response_metrics(phase, rates)

    np.testing.assert_allclose(metrics["mean_rate"], [2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(metrics["f1_amplitude"], [0.7, 0.4], atol=1e-12)
    np.testing.assert_allclose(
        metrics["response_amp_rms"], np.asarray([0.7, 0.4]) / np.sqrt(2), atol=1e-12
    )


def test_native_kinematic_occupancy_maps_k_dot_v_without_fft(tmp_path) -> None:
    # One degree/second along x gives |[2,0] dot [1,0]| = 2 Hz at 1x.
    time = np.arange(6, dtype=float) / 240.0
    traces = np.stack((time, np.zeros_like(time)), axis=1)[None]

    occupancy, audit = native_kinematic_occupancy(
        mode_power=np.asarray([3.0]),
        kxy=np.asarray([[2.0, 0.0]]),
        traces=traces,
        frame_rate_hz=240.0,
        spatial_cpd=np.asarray([np.sqrt(2), 2.0, 2 * np.sqrt(2)]),
        temporal_hz=np.asarray([1.0, 2.0, 4.0, 8.0]),
        orientation_deg=np.asarray([0.0, 90.0]),
    )

    assert occupancy[2, 1, 1, 1] > 0  # 1x motion, 2 cpd, 2 Hz, vertical bars
    assert occupancy[3, 1, 2, 1] > 0  # 2x motion moves the same mode to 4 Hz
    assert audit["fraction_image_mode_power_in_resolved_sf_grid"] == 1.0


def test_rucci_overlap_uses_stable_weighted_center_sf_tertile_tails() -> None:
    units = np.arange(6)
    tuning = np.ones((6, 1, 1, 1), dtype=float)
    occupancy = np.ones((5, 1, 1, 1), dtype=float)
    robust = pd.DataFrame(
        {
            "unit_index": units,
            "weighted_center_sf_cpd": [3.0, 1.0, 6.0, 2.0, 5.0, 4.0],
            "fit_r2": np.ones(6),
            "peak_censoring": ["none"] * 6,
            "low_sf_censored": [False] * 6,
            "preferred_tf_hz": np.ones(6),
            "low_tf_censored": [False] * 6,
            "high_tf_censored": [False] * 6,
        }
    )

    overlaps = overlap_table(units, tuning, occupancy, robust)
    labels = overlaps.drop_duplicates("unit_index").set_index("unit_index").sf_group

    assert set(labels[labels.eq("low_sf")].index) == {1, 3}
    assert set(labels[labels.eq("high_sf")].index) == {2, 4}
    assert set(labels[labels.eq("middle_sf")].index) == {0, 5}


def test_joint_engagement_interpolates_log_frequency_and_orientation() -> None:
    lower, upper, low_weight, high_weight, valid = log_interpolation_weights(
        np.asarray([np.sqrt(2.0)]), np.asarray([1.0, 2.0, 4.0])
    )
    assert valid[0]
    assert (lower[0], upper[0]) == (0, 1)
    np.testing.assert_allclose([low_weight[0], high_weight[0]], [0.5, 0.5])

    # k points along physical +y, so its grating-bar axis is 0 degrees.
    o0, o1, w0, w1 = circular_orientation_weights(
        np.asarray([[0.0, 2.0]]), np.asarray([0.0, 45.0, 90.0, 135.0])
    )
    assert (o0[0], o1[0]) == (0, 1)
    np.testing.assert_allclose([w0[0], w1[0]], [1.0, 0.0])


def test_trajectory_phase_spectrum_recovers_constant_translation_frequency() -> None:
    rate_hz = 128.0
    n_time = 128
    time = np.arange(n_time, dtype=float) / rate_hz
    trace = np.zeros((1, n_time, 2), dtype=float)
    trace[0, :, 0] = 4.0 * time

    temporal_hz, power = trajectory_phase_spectra(
        np.asarray([[2.0, 0.0]]), trace, rate_hz, chunk_size=1
    )

    # A 2-cpd Fourier mode translated at 4 deg/s is an 8-Hz carrier.
    assert temporal_hz[np.argmax(power[0])] == 8.0


def test_trajectory_phase_spectrum_uses_temporal_order_not_velocity_histogram() -> None:
    rate_hz = 64.0
    increments_blocked = np.tile([0.01, 0.01, -0.01, -0.01], 16)
    increments_alternating = np.tile([0.01, -0.01, 0.01, -0.01], 16)
    traces = np.zeros((2, len(increments_blocked), 2), dtype=float)
    traces[0, :, 0] = np.cumsum(increments_blocked)
    traces[1, :, 0] = np.cumsum(increments_alternating)

    # Estimate the two paths separately: their increment/velocity histograms
    # are identical, but the temporal ordering and therefore phase spectra differ.
    _, blocked = trajectory_phase_spectra(
        np.asarray([[8.0, 0.0]]), traces[:1], rate_hz, chunk_size=1
    )
    _, alternating = trajectory_phase_spectra(
        np.asarray([[8.0, 0.0]]), traces[1:], rate_hz, chunk_size=1
    )

    assert not np.allclose(blocked, alternating, rtol=0.05, atol=1e-8)


def test_brownian_phase_spectrum_broadens_with_spatial_frequency() -> None:
    rng = np.random.default_rng(17)
    rate_hz = 128.0
    n_traces, n_time = 96, 128
    diffusion = 0.015
    increments = rng.normal(
        scale=np.sqrt(2.0 * diffusion / rate_hz),
        size=(n_traces, n_time, 2),
    )
    traces = np.cumsum(increments, axis=1)

    temporal_hz, power = trajectory_phase_spectra(
        np.asarray([[1.0, 0.0], [8.0, 0.0]]),
        traces,
        rate_hz,
        chunk_size=2,
    )
    centroid = (power * temporal_hz[None]).sum(axis=1) / power.sum(axis=1)

    assert centroid[1] > 2.0 * centroid[0]


def test_raw_tuning_is_log_interpolated_only_on_resolvable_fft_bins() -> None:
    probe = np.asarray([1.0, 2.0, 4.0, 8.0])
    target = np.asarray([0.5, 2.0, np.sqrt(8.0), 16.0])
    tuning = np.zeros((1, 1, len(probe), 1), dtype=float)
    tuning[0, 0, :, 0] = [0.0, 1.0, 3.0, 0.0]

    interpolated = interpolate_tuning_to_fft_bins(tuning, probe, target)[0, 0, :, 0]

    assert interpolated[0] == 0.0
    assert interpolated[-1] == 0.0
    assert interpolated[1] > 0.0
    assert interpolated[2] > interpolated[1]


def test_joint_engagement_preserves_sf_tf_pairing_until_final_dot_product() -> None:
    # Mode 0 occupies low-SF/low-TF; mode 1 occupies high-SF/high-TF.
    # The unit responds to the matched diagonal only. Two images exchange which
    # spatial mode carries power, so their joint engagement must differ.
    mode_power = np.asarray([[3.0, 1.0], [1.0, 4.0]])
    tuning = np.zeros((1, 2, 2, 1), dtype=float)
    tuning[0, 0, 0, 0] = 0.5
    tuning[0, 1, 1, 0] = 0.5
    contract = {
        "sf0": np.asarray([0, 1]),
        "sf1": np.asarray([0, 1]),
        "sw0": np.ones(2),
        "sw1": np.zeros(2),
        "ori0": np.zeros(2, dtype=int),
        "ori1": np.zeros(2, dtype=int),
        "ow0": np.ones(2),
        "ow1": np.zeros(2),
        "resolved": np.ones(2, dtype=bool),
        "temporal": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    }

    engagement, _ = image_unit_engagement(mode_power, tuning, contract)

    np.testing.assert_allclose(engagement["joint"][:, 0], [2.0, 2.5])
    np.testing.assert_allclose(engagement["separable"][:, 0], [1.0, 1.25])


def test_joint_alignment_fraction_removes_global_image_power_scale() -> None:
    mode_power = np.asarray([[1.0, 2.0], [7.0, 14.0]])
    tuning = np.zeros((1, 2, 2, 1), dtype=float)
    tuning[0, 0, 0, 0] = 0.7
    tuning[0, 1, 1, 0] = 0.3
    contract = {
        "sf0": np.asarray([0, 1]),
        "sf1": np.asarray([0, 1]),
        "sw0": np.ones(2),
        "sw1": np.zeros(2),
        "ori0": np.zeros(2, dtype=int),
        "ori1": np.zeros(2, dtype=int),
        "ow0": np.ones(2),
        "ow1": np.zeros(2),
        "resolved": np.ones(2, dtype=bool),
        "temporal": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    }
    engagement, _ = image_unit_engagement(mode_power, tuning, contract)
    assert engagement["joint"][1, 0] == 7.0 * engagement["joint"][0, 0]
    np.testing.assert_allclose(
        engagement["joint_fraction"][0, 0],
        engagement["joint_fraction"][1, 0],
    )


def test_direct_rendered_joint_score_preserves_pairing_and_separable_control() -> None:
    cubes = np.zeros((2, 2, 2, 1), dtype=float)
    cubes[0, 0, 0, 0] = 1.0
    cubes[0, 1, 1, 0] = 1.0
    cubes[1, 0, 1, 0] = 1.0
    cubes[1, 1, 0, 0] = 1.0
    tuning = np.zeros((1, 2, 2, 1), dtype=float)
    tuning[0, 0, 0, 0] = 0.5
    tuning[0, 1, 1, 0] = 0.5

    scores = engagement_from_cubes(cubes, tuning)

    assert scores["joint_fraction"][0, 0] > scores["joint_fraction"][1, 0]
    np.testing.assert_allclose(
        scores["separable_fraction"][0, 0],
        scores["separable_fraction"][1, 0],
    )
    scaled = engagement_from_cubes(cubes * np.asarray([3.0, 7.0])[:, None, None, None], tuning)
    np.testing.assert_allclose(scaled["joint_fraction"], scores["joint_fraction"])


def test_matrix_contract_validates_coordinates_shapes_and_integrity(tmp_path) -> None:
    n_images, n_traces, n_units = 2, 3, 4
    pd.DataFrame({"image_index": range(n_images)}).to_csv(
        tmp_path / "image_feature_table.csv", index=False
    )
    pd.DataFrame({"trace_index": range(n_traces)}).to_csv(
        tmp_path / "trace_feature_table.csv", index=False
    )
    pd.DataFrame({"unit_index": range(n_units)}).to_csv(
        tmp_path / "unit_feature_table.csv", index=False
    )
    movie_shape = (n_images * n_traces, n_units)
    for name in ("ssi_matrix", "expected_spikes_matrix", "mean_rate_matrix"):
        np.save(tmp_path / f"{name}.npy", np.ones(movie_shape))
    stable_shape = (n_images, n_units)
    for name in (
        "stabilized_ssi_by_image",
        "stabilized_expected_spikes_by_image",
        "stabilized_mean_rate_by_image",
    ):
        np.save(tmp_path / f"{name}.npy", np.ones(stable_shape))
    np.save(tmp_path / "trace_xy.npy", np.zeros((n_traces, 5, 2)))
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "n_images": n_images,
                "n_traces": n_traces,
                "n_units": n_units,
                "integrity_checks": {"finite": True, "nonnegative": True},
            }
        )
    )
    (tmp_path / "stabilized_baseline_summary.json").write_text("{}")

    audit = validate_matrix_contract(tmp_path)
    assert audit["declared_coordinates"] == {
        "n_images": n_images,
        "n_traces": n_traces,
        "n_units": n_units,
    }
    assert all(audit["merge_integrity_checks"].values())


def test_native240_trace_contract_requires_true_scored_grid() -> None:
    contract = {
        "source_trace_rate_hz": 120,
        "source_trace_samples": 40,
        "model_output_rate_hz": 240,
        "scored_samples_per_source_trace_sample": 2,
        "scored_trace_samples": 80,
        "scored_bin_seconds": 1.0 / 240.0,
        "analysis_interval_seconds": 1.0 / 3.0,
    }
    validate_native240_trace_contract(contract)
    contract["scored_trace_samples"] = 40
    with np.testing.assert_raises(RuntimeError):
        validate_native240_trace_contract(contract)


def test_separable_control_preserves_marginals_but_removes_sf_tf_coupling() -> None:
    mode_power = np.eye(2, dtype=float)
    diagonal = np.zeros((1, 2, 2, 1), dtype=float)
    diagonal[0, 0, 0, 0] = 0.5
    diagonal[0, 1, 1, 0] = 0.5
    anti_diagonal = np.zeros_like(diagonal)
    anti_diagonal[0, 0, 1, 0] = 0.5
    anti_diagonal[0, 1, 0, 0] = 0.5
    contract = {
        "sf0": np.asarray([0, 1]),
        "sf1": np.asarray([0, 1]),
        "sw0": np.ones(2),
        "sw1": np.zeros(2),
        "ori0": np.zeros(2, dtype=int),
        "ori1": np.zeros(2, dtype=int),
        "ow0": np.ones(2),
        "ow1": np.zeros(2),
        "resolved": np.ones(2, dtype=bool),
        "temporal": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    }

    diagonal_scores, _ = image_unit_engagement(mode_power, diagonal, contract)
    anti_diagonal_scores, _ = image_unit_engagement(mode_power, anti_diagonal, contract)

    # The two tensors have identical SF, TF, and orientation marginals, hence
    # identical separable controls, but opposite joint compatibility.
    np.testing.assert_allclose(
        diagonal_scores["separable"], anti_diagonal_scores["separable"]
    )
    np.testing.assert_allclose(diagonal_scores["separable"][:, 0], [0.25, 0.25])
    np.testing.assert_allclose(diagonal_scores["joint"][:, 0], [0.5, 0.5])
    np.testing.assert_allclose(anti_diagonal_scores["joint"][:, 0], [0.0, 0.0])
