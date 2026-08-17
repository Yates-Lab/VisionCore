import numpy as np

from paper.fig4.mechanism_audit_v1.layerwise_sftf.analyze_layerwise_tuning import (
    phase_orientation_surface,
)
from paper.fig4.mechanism_audit_v1.layerwise_sftf.compute_fem_tuning_overlap import (
    instantaneous_temporal_frequency,
)
from paper.fig4.mechanism_audit_v1.layerwise_sftf.run_layerwise_grating_probe import (
    harmonic_amplitude,
)


def test_constant_velocity_obeys_spatial_to_temporal_mapping() -> None:
    # A 4-cpd component moving at 1.5 deg/s across its bars must generate 6 Hz.
    modes = np.asarray([[4.0, 0.0], [0.0, 4.0]])
    velocity = np.asarray([[[1.5, 0.0], [0.0, -2.0]]])
    observed = instantaneous_temporal_frequency(modes, velocity)
    expected = np.asarray([[[6.0, 0.0], [0.0, 8.0]]])
    np.testing.assert_allclose(observed, expected)


def test_harmonic_fit_recovers_amplitude_with_offset_and_phase() -> None:
    time = np.arange(360, dtype=float) / 120.0
    values = 3.2 + 1.7 * np.sin(2 * np.pi * 7.3 * time + 0.61)
    observed = harmonic_amplitude(values[:, None], 7.3, time)
    np.testing.assert_allclose(observed, [1.7], atol=1e-10)


def test_amplitude_phase_collapse_is_quadrature_safe() -> None:
    # Shape: channel, SF, TF, orientation, phase.  RMS phase aggregation
    # preserves equal-amplitude responses with phase-dependent signs.
    values = np.asarray([[[[[2.0, -2.0], [2.0, -2.0]]]]])
    collapsed, oriented = phase_orientation_surface(values, "f1_amplitude")
    np.testing.assert_allclose(oriented, 2.0)
    np.testing.assert_allclose(collapsed, 2.0)
