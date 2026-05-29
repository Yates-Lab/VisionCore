from __future__ import annotations

import numpy as np
import pytest

from jake.twininfo.population import ranked_units_from_session_results


def test_ranked_units_filter_by_ccmax_and_sort_by_performance():
    session_results = [
        {
            "session": "A",
            "neuron_mask": np.array([10, 11, 12]),
            "ccmax": np.array([0.9, 0.7, 0.95]),
            "ccnorm": np.array([0.5, 0.99, 0.4]),
            "rhos": np.array([0.2, 0.3, 0.1]),
            "ccabs": np.array([0.45, 0.69, 0.38]),
            "ve_model": np.array([0.1, 0.2, 0.05]),
            "ve_psth": np.array([0.09, 0.19, 0.04]),
        },
        {
            "session": "B",
            "neuron_mask": np.array([20, 21]),
            "ccmax": np.array([0.91, np.nan]),
            "ccnorm": np.array([0.8, 0.95]),
            "rhos": np.array([0.7, 0.8]),
            "ccabs": np.array([0.72, 0.76]),
            "ve_model": np.array([0.3, 0.4]),
            "ve_psth": np.array([0.25, 0.35]),
        },
    ]

    rows = ranked_units_from_session_results(
        session_results,
        performance_metric="ccnorm",
        ccmax_threshold=0.8,
        model_session_names=("A", "B"),
    )

    assert [(row["session_name"], row["original_neuron_id"]) for row in rows] == [
        ("B", 20),
        ("A", 10),
        ("A", 12),
    ]
    assert [row["performance_score"] for row in rows] == [0.8, 0.5, 0.4]


def test_ranked_units_filter_by_min_performance_score():
    session_results = [
        {
            "session": "A",
            "neuron_mask": np.array([10, 11, 12]),
            "ccmax": np.array([0.9, 0.91, 0.92]),
            "ccnorm": np.array([0.7, 0.5, 0.3]),
        },
    ]

    rows = ranked_units_from_session_results(
        session_results,
        performance_metric="ccnorm",
        min_performance_score=0.5,
        ccmax_threshold=0.8,
        model_session_names=("A",),
    )

    assert [row["original_neuron_id"] for row in rows] == [10, 11]


def test_ranked_units_ignore_sessions_not_in_model():
    session_results = [
        {
            "session": "missing",
            "neuron_mask": np.array([1]),
            "ccmax": np.array([0.99]),
            "ccnorm": np.array([0.99]),
        },
        {
            "session": "present",
            "neuron_mask": np.array([2]),
            "ccmax": np.array([0.9]),
            "ccnorm": np.array([0.2]),
        },
    ]

    rows = ranked_units_from_session_results(
        session_results,
        performance_metric="ccnorm",
        ccmax_threshold=0.8,
        model_session_names=("present",),
    )

    assert len(rows) == 1
    assert rows[0]["session_name"] == "present"
    assert rows[0]["original_neuron_id"] == 2


def test_ranked_units_reject_unknown_metric():
    with pytest.raises(ValueError, match="Unknown performance metric"):
        ranked_units_from_session_results([], performance_metric="not_a_metric")
