"""Behaviour of `paper/model_selection/evaluate.py`.

`protocol.py` declares METRICS = (bps, ccnorm, single_trial_r2) but training
computes only BPS, and nothing in the repo ever called `test_dataloader()`.
These tests pin the parts of the evaluation pass that do not need a GPU or the
data volume: the single-trial score itself, protocol inclusion, and the guard
that stops a run from being read under a protocol it was not produced under.

The heavy passes (the test-split BPS sweep and the held-out fixrsvp inference)
are verified by running them, not here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODEL_SELECTION = HERE.parent / "paper" / "model_selection"
if str(MODEL_SELECTION) not in sys.path:
    sys.path.insert(0, str(MODEL_SELECTION))

import evaluate  # noqa: E402
import protocol  # noqa: E402


# ---------------------------------------------------------------------------
# single-trial r^2
# ---------------------------------------------------------------------------
def test_single_trial_r2_is_one_for_exact_prediction_and_zero_for_the_mean():
    """The score is per unit, and anchored at the two references that matter.

    Figures 3 and 4 rest on this number, and its sign convention is what makes
    a twin "better than the PSTH" meaningful: predicting each trial exactly
    scores 1, predicting the unit's mean on every trial scores 0.
    """
    rng = np.random.default_rng(0)
    robs = rng.poisson(3.0, size=(20, 30, 4)).astype(float)
    dfs = np.ones_like(robs)

    exact = evaluate.single_trial_r2(robs.copy(), robs, dfs)
    mean_only = evaluate.single_trial_r2(
        np.broadcast_to(robs.mean(axis=(0, 1)), robs.shape).copy(), robs, dfs
    )

    assert exact.shape == (4,)
    np.testing.assert_allclose(exact, np.ones(4), atol=1e-12)
    np.testing.assert_allclose(mean_only, np.zeros(4), atol=1e-12)


def test_single_trial_r2_ignores_bins_the_data_filter_rejects():
    """`dfs == 0` bins must not enter either variance.

    The twin emits a prediction for every bin, including bins no unit was
    recorded on. Scoring those would silently mix unrecorded samples into a
    number the figures report.
    """
    rng = np.random.default_rng(1)
    robs = rng.poisson(3.0, size=(20, 30, 2)).astype(float)
    rhat = robs.copy()
    dfs = np.ones_like(robs)

    # Reject a quarter of the bins for unit 0 and fill them with garbage that
    # would wreck the score if it were counted.
    dfs[:5, :, 0] = 0.0
    rhat[:5, :, 0] = 1e3

    scored = evaluate.single_trial_r2(rhat, robs, dfs)

    np.testing.assert_allclose(scored, np.ones(2), atol=1e-12)


def test_bps_per_unit_survives_the_nan_padding_of_assembled_trials():
    """Unfilled (trial, bin) slots must not poison BPS.

    The fixrsvp pass assembles ragged trials into a full (trial, time, unit)
    array and leaves the unreached slots NaN. `bits_per_spike` sanitises NaN in
    the rates but not in `dfs`, so a NaN there makes `T = dfs.sum(0)` NaN and
    every unit scores NaN -- silently, as an absent metric rather than an error.
    """
    rng = np.random.default_rng(2)
    robs = rng.poisson(2.0, size=(12, 20, 3)).astype(float)
    rhat = np.full_like(robs, 2.0)
    dfs = np.ones_like(robs)

    # The last five bins of every trial were never reached.
    robs[:, 15:, :] = np.nan
    rhat[:, 15:, :] = np.nan
    dfs[:, 15:, :] = np.nan

    bps = evaluate.bps_per_unit(rhat, robs, dfs)

    assert bps.shape == (3,)
    assert np.isfinite(bps).all()


# ---------------------------------------------------------------------------
# BPS reduction
# ---------------------------------------------------------------------------
def test_overall_bps_matches_the_training_reduction():
    """Mean over datasets of each dataset's mean over units, NaN units dropped.

    This has to agree with `MultiDatasetModel.on_validation_epoch_end`, or a
    run's reported test BPS is not comparable to the validation BPS it was
    selected on. That reduction drops NaN units (cells with no samples) and
    clamps a negative per-unit BPS to zero before averaging.
    """
    per_dataset = {
        "Allen_2022-02-16": np.array([0.4, 0.6, np.nan]),   # -> 0.5
        "Logan_2020-01-06": np.array([-0.2, 0.4]),           # -> 0.2 after clamp
    }

    overall, per_ds_mean = evaluate.overall_bps(per_dataset)

    assert per_ds_mean["Allen_2022-02-16"] == pytest.approx(0.5)
    assert per_ds_mean["Logan_2020-01-06"] == pytest.approx(0.2)
    assert overall == pytest.approx(0.35)


def test_a_dataset_with_no_scored_units_is_skipped_not_counted_as_zero():
    """An empty readout must not drag the mean down.

    Counting it as 0.0 would make a run look worse purely because one session
    contributed no valid samples to the pass.
    """
    per_dataset = {
        "Allen_2022-02-16": np.array([0.4, 0.6]),
        "Logan_2020-01-06": np.array([np.nan, np.nan]),
    }

    overall, per_ds_mean = evaluate.overall_bps(per_dataset)

    assert "Logan_2020-01-06" not in per_ds_mean
    assert overall == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# protocol inclusion
# ---------------------------------------------------------------------------
def test_inclusion_applies_every_protocol_criterion_including_the_session_floor():
    """Units below the spike or reliability floor go, then thin sessions go.

    The order matters: the session floor counts units that already passed unit
    inclusion, so a session can be dropped by cells it lost rather than by the
    cells it started with.
    """
    n_units = protocol.MIN_SESSION_UNITS + 2

    rich = {
        "session": "Allen_2022-02-16",
        "total_spikes": np.full(n_units, protocol.MIN_TOTAL_SPIKES + 1.0),
        "psth_r2": np.full(n_units, protocol.MIN_PSTH_R2 + 0.05),
    }
    # Same size, but two cells fail each criterion, dropping it under the floor.
    thin = {
        "session": "Logan_2020-01-06",
        "total_spikes": np.full(n_units, protocol.MIN_TOTAL_SPIKES + 1.0),
        "psth_r2": np.full(n_units, protocol.MIN_PSTH_R2 + 0.05),
    }
    thin["total_spikes"][:2] = protocol.MIN_TOTAL_SPIKES - 1
    thin["psth_r2"][2] = protocol.MIN_PSTH_R2 - 0.01

    kept = evaluate.apply_protocol_inclusion([rich, thin])

    assert [s["session"] for s in kept] == ["Allen_2022-02-16"]
    assert kept[0]["include"].sum() == n_units


# ---------------------------------------------------------------------------
# the pooling guard
# ---------------------------------------------------------------------------
def test_a_run_from_a_different_protocol_is_refused(tmp_path):
    """Reading a foreign-protocol run must fail loudly, naming both hashes.

    The whole point of stamping PROTOCOL_HASH into every manifest is that a
    sweep spanning days cannot silently mix a run trained under one split with
    a run trained under another.
    """
    run_dir = tmp_path / "E1a"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run": "E1a",
        "protocol_hash": "deadbeefcafe",
        "spec": {"name": "E1a"},
    }))

    with pytest.raises(ValueError) as excinfo:
        evaluate.load_run_manifest(run_dir)

    message = str(excinfo.value)
    assert "deadbeefcafe" in message
    assert protocol.PROTOCOL_HASH in message


def test_a_run_from_the_current_protocol_loads(tmp_path):
    """The guard must not reject the runs it is meant to admit."""
    run_dir = tmp_path / "E1a"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run": "E1a",
        "protocol_hash": protocol.PROTOCOL_HASH,
        "spec": {"name": "E1a", "seed": 101},
    }))

    manifest = evaluate.load_run_manifest(run_dir)

    assert manifest["run"] == "E1a"
    assert manifest["spec"]["seed"] == 101
