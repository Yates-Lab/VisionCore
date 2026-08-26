import numpy as np
import pytest

from paper.fig4.spatiotemporal_tuning import population_response as population


def test_quantile_bins_assign_every_trace_once_with_balanced_counts() -> None:
    values = np.asarray((8.0, 1.0, 3.0, 4.0, 2.0, 7.0, 6.0, 5.0))
    labels = population.quantile_bins(values, 3)
    assert set(labels.tolist()) == {0, 1, 2}
    assert int(np.ptp(np.bincount(labels))) <= 1
    assert np.max(values[labels == 0]) <= np.min(values[labels == 1])


def test_population_effect_uses_spike_weighted_ssi_and_matched_baseline() -> None:
    moving_spikes = np.asarray([[[2.0], [4.0]], [[2.0], [4.0]]])
    arrays = {
        "moving_spikes": moving_spikes,
        "moving_ssi": np.full_like(moving_spikes, 0.5),
        "stable_spikes": np.asarray([[2.0], [2.0]]),
        "stable_ssi": np.asarray([[0.25], [0.25]]),
    }
    reduced = population._population_arrays(arrays, np.asarray([0]))
    rate, ssi = population.population_effects(reduced, np.asarray((0, 1)))
    assert rate == 50.0
    assert ssi == 100.0


def test_population_bootstrap_resamples_units() -> None:
    moving_spikes = np.asarray([[[1.0, 3.0]]])
    arrays = {
        "moving_spikes": moving_spikes,
        "moving_ssi": np.ones_like(moving_spikes),
        "stable_spikes": np.ones((1, 2)),
        "stable_ssi": np.ones((1, 2)),
    }
    reduced = population._population_arrays(arrays, np.asarray([0, 1]))
    center, low, high = population.crossed_population_bootstrap(
        reduced,
        np.asarray([0]),
        n_bootstrap=500,
        rng=np.random.default_rng(11),
    )
    assert center[0] == pytest.approx(100.0)
    assert low[0] < center[0] < high[0]


def test_matrix_filter_gate_requires_zero_phase_provenance() -> None:
    summary = {
        "shard_summaries": [
            {
                "trace_bank": {
                    "trace_provenance": {
                        "filter": {
                            "kind": "zero-phase analysis IIR before 240-Hz sampling",
                            "passband_hz": 20.0,
                            "stopband_hz": 30.0,
                        }
                    }
                }
            }
        ]
    }
    assert population.matrix_trace_filter(summary)["passband_hz"] == 20.0
    assert population.matrix_trace_filter({"shard_summaries": [{}]}) is None


def test_selected_model_gate_reads_canonical_nested_checkpoint_digest(tmp_path) -> None:
    model_spec = tmp_path / "production_model.yaml"
    model_spec.write_text("checkpoint:\n  sha256: selected-digest\n", encoding="utf-8")
    summary = {
        "shard_summaries": [
            {
                "model_provenance": {
                    "model": {
                        "checkpoint_path": "/models/model/epoch.ckpt",
                        "checkpoint_sha256": "other-digest",
                    }
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="does not match"):
        population.assert_selected_matrix(summary, model_spec=model_spec)
    summary["shard_summaries"][0]["model_provenance"]["model"][
        "checkpoint_sha256"
    ] = "selected-digest"
    _, digest = population.assert_selected_matrix(summary, model_spec=model_spec)
    assert digest == "selected-digest"
