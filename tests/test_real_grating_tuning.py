import numpy as np

from eval.real_grating_tuning import (
    _lag_pairs,
    axial_error_deg,
    binned_phase_curve,
    cross_half_harmonic,
    harmonic_fit,
    local_quadratic_peak,
    phase_harmonic_grid_targets,
    phase_harmonic_targets,
    periodic_local_quadratic_peak,
    stable_trial_mask,
)


def test_lag_pairs_do_not_cross_trials_or_missing_rows():
    raw = np.array([10, 11, 12, 20, 21, 23])
    trials = np.array([1, 1, 1, 2, 2, 2])
    stim, resp = _lag_pairs(raw, trials, lag=1)
    assert np.array_equal(stim, [0, 1, 3])
    assert np.array_equal(resp, [1, 2, 4])


def test_stable_trial_mask_rejects_integer_boundary_averages():
    # The isolated 1 can arise from averaging probe=-1 with trial=3.  Merely
    # checking that it is integer-valued would incorrectly treat it as trial 1.
    values = np.array([-1.0, -1.0, -1.0, -1.0, 0.5, 1.0, 3.0, 3.0, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0])
    mask = stable_trial_mask(values, min_run=4)
    assert np.array_equal(mask, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])


def test_phase_zero_is_a_valid_sample():
    phase = np.array([0.0, np.pi / 2, np.pi, -1.0])
    response = np.array([4.0, 2.0, 1.0, 100.0])
    curve = binned_phase_curve(phase, response, np.ones(4, dtype=bool), n_bins=4)
    assert curve[0] == 4.0
    assert curve[1] == 2.0
    assert curve[2] == 1.0
    assert np.isnan(curve[3])


def test_harmonic_fit_reports_standard_f1_over_f0():
    phase = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    response = 4.0 + 2.0 * np.sin(phase + 0.3)
    fit = harmonic_fit(phase, response)
    assert np.isclose(fit["offset"], 4.0)
    assert np.isclose(fit["amplitude"], 2.0)
    assert np.isclose(fit["f1_f0"], 0.5)
    assert np.isclose(fit["r2"], 1.0)


def test_cross_half_harmonic_removes_unshared_noise_power():
    phase = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    first = harmonic_fit(phase, 4.0 + 2.0 * np.sin(phase + 0.3))
    second = harmonic_fit(phase, 4.0 + 2.0 * np.sin(phase + 0.3))
    shared = cross_half_harmonic(first, second, offset=4.0)
    assert np.isclose(shared["amplitude"], 2.0)
    assert np.isclose(shared["f1_f0"], 0.5)
    assert np.isclose(shared["phase_consistency"], 1.0)

    opposite = harmonic_fit(phase, 4.0 - 2.0 * np.sin(phase + 0.3))
    unshared = cross_half_harmonic(first, opposite, offset=4.0)
    assert unshared["cross_power"] < 0
    assert unshared["amplitude"] == 0.0


def test_local_quadratic_peak_and_boundary_censor():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = -(x - 1.25) ** 2
    peak, boundary = local_quadratic_peak(x, y)
    assert np.isclose(peak, 1.25)
    assert not boundary
    edge, edge_boundary = local_quadratic_peak(x, x)
    assert edge == 3.0
    assert edge_boundary


def test_periodic_peak_wraps_and_orientation_error_is_axial():
    values = np.array([1.0, 0.0, 0.0, 0.5])
    peak = periodic_local_quadratic_peak(values, period=180.0)
    assert 160.0 < peak < 180.0 or 0.0 <= peak < 20.0
    assert axial_error_deg(175.0, 5.0) == 10.0


def test_phase_harmonic_targets_use_debiased_all_repeat_amplitude():
    n_trials, trial_length, lag = 8, 96, 2
    n_rows = n_trials * trial_length
    raw_indices = np.arange(n_rows)
    trials = np.repeat(np.arange(n_trials), trial_length)
    phase = np.tile(
        np.linspace(0.0, 2.0 * np.pi, trial_length, endpoint=False), n_trials
    )
    sf = np.tile(np.resize(np.array([1.0, 2.0]), trial_length), n_trials)
    ori = np.tile(np.resize(np.array([0.0, 0.0, 90.0, 90.0]), trial_length), n_trials)
    robs = np.zeros((n_rows, 1), dtype=np.float64)
    dfs = np.ones_like(robs)

    # The preferred condition is SF=2, orientation=90. Responses are placed
    # two samples later and are identical in even/odd repeat halves, so the
    # cross-half estimate is exactly the generating amplitude.
    stim, resp = _lag_pairs(raw_indices, trials, lag)
    preferred = np.isclose(sf[stim], 2.0) & np.isclose(ori[stim], 90.0)
    rates = 20.0 + 5.0 * np.sin(phase[stim[preferred]] + 0.2)
    robs[resp[preferred], 0] = rates * 0.01

    # [split, unit, lag, blank + 2 SF x 2 orientation]. Only the preferred
    # condition varies across lag, with a unique maximum at lag=2.
    observed = np.zeros((3, 1, 4, 5), dtype=np.float64)
    observed[:, 0, lag, 4] = 10.0
    lag_tensors = {
        "observed": observed,
        "sfs": np.array([1.0, 2.0]),
        "oris": np.array([0.0, 90.0]),
        "lags_ms": np.arange(4, dtype=np.float64) * 10.0,
    }
    result = phase_harmonic_targets(
        lag_tensors=lag_tensors,
        robs=robs,
        dfs=dfs,
        sf=sf,
        ori=ori,
        phase=phase,
        trials=trials,
        raw_indices=raw_indices,
        dt=0.01,
    )
    assert result["preferred_lag_idx"][0] == lag
    assert result["preferred_sf_cpd"][0] == 2.0
    assert result["preferred_ori_deg"][0] == 90.0
    assert result["phase_consistency"][0] > 0.999
    assert np.isclose(result["target_f1_f0"][0], 0.25, atol=1e-6)
    selected = result["target_mask"][:, 0]
    assert selected.sum() == preferred.sum()
    expected = 0.01 * (20.0 + 5.0 * np.sin(phase[stim[preferred]] + 0.2))
    assert np.allclose(result["target_counts"][selected, 0], expected, atol=1e-6)


def test_phase_harmonic_grid_targets_cover_every_measured_condition():
    n_trials, trial_length, lag = 4, 64, 2
    n_rows = n_trials * trial_length
    raw_indices = np.arange(n_rows)
    trials = np.repeat(np.arange(n_trials), trial_length)
    phase = np.tile(
        np.linspace(0.0, 2.0 * np.pi, trial_length, endpoint=False), n_trials
    )
    sf_pattern = np.resize(np.array([1.0, 1.0, 2.0, 2.0]), trial_length)
    ori_pattern = np.resize(np.array([0.0, 90.0, 0.0, 90.0]), trial_length)
    sf = np.tile(sf_pattern, n_trials)
    ori = np.tile(ori_pattern, n_trials)
    robs = np.zeros((n_rows, 1), dtype=np.float64)
    dfs = np.ones_like(robs)

    stim, resp = _lag_pairs(raw_indices, trials, lag)
    offsets = 10.0 + 2.0 * sf[stim] + ori[stim] / 45.0
    amplitudes = 1.0 + 0.5 * sf[stim] + ori[stim] / 90.0
    expected_rates = offsets + amplitudes * np.sin(phase[stim] + 0.3)
    robs[resp, 0] = expected_rates * 0.01

    observed = np.zeros((3, 1, 4, 5), dtype=np.float64)
    observed[:, 0, lag, 1:] = [1.0, 2.0, 3.0, 6.0]
    result = phase_harmonic_grid_targets(
        lag_tensors={
            "observed": observed,
            "sfs": np.array([1.0, 2.0]),
            "oris": np.array([0.0, 90.0]),
            "lags_ms": np.arange(4, dtype=np.float64) * 10.0,
        },
        robs=robs,
        dfs=dfs,
        sf=sf,
        ori=ori,
        phase=phase,
        trials=trials,
        raw_indices=raw_indices,
        dt=0.01,
    )
    assert result["preferred_lag_idx"][0] == lag
    assert result["target_mask"][:, 0].sum() == len(resp)
    assert np.allclose(
        result["target_counts"][resp, 0],
        expected_rates * 0.01,
        atol=1e-6,
    )
    assert np.allclose(result["fit_r2"][0], 1.0)
