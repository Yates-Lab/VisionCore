import numpy as np

from paper.fig4.spatiotemporal_tuning.audit_eye_trace_filter import (
    piecewise_psd_breakpoint,
    trace_metrics,
    zero_phase_filter,
)


def test_piecewise_psd_breakpoint_recovers_signal_to_noise_transition() -> None:
    frequency = np.arange(1.0, 101.0)
    power = np.where(frequency <= 24, frequency**-2.0, (24.0**-2.0) * (frequency / 24.0) ** 1.5)
    result = piecewise_psd_breakpoint(frequency, power)
    assert abs(result["breakpoint_hz"] - 24.0) <= 2.0
    assert result["low_frequency_loglog_slope"] < -1.5
    assert result["high_frequency_loglog_slope"] > 1.0


def test_analysis_filter_removes_high_frequency_jitter_but_retains_drift_radius() -> None:
    time = np.arange(300) / 240.0
    drift = 0.05 * np.sin(2 * np.pi * 3 * time)
    jitter = 0.01 * np.sin(2 * np.pi * 70 * time)
    trace = np.column_stack((drift + jitter, 0.7 * drift - jitter))[None]
    filtered, _ = zero_phase_filter(trace, frame_rate_hz=240, passband_hz=20, stopband_hz=30)
    original_metrics = trace_metrics(trace, frame_rate_hz=240)
    filtered_metrics = trace_metrics(filtered, frame_rate_hz=240)
    assert filtered_metrics["mean_speed_deg_s"][0] < 0.25 * original_metrics["mean_speed_deg_s"][0]
    assert filtered_metrics["rms_radius_arcmin"][0] > 0.9 * original_metrics["rms_radius_arcmin"][0]
