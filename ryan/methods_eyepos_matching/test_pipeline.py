"""Tests for the methods-folder parallel pipeline.

Two core checks:

1. ``test_extract_windows_matches_legacy`` — the numpy windowing in
   ``pipeline._extract_windows_numpy`` produces bit-identical (counts,
   trajectories, T_idx) to the legacy torch ``extract_windows``. If this
   diverges, every downstream equivalence claim is suspect.

2. ``test_decompose_session_schema`` — the per-session driver runs end-to-end
   on a tiny aligned record and returns the expected schema (keys, shapes,
   shuffles).

Plus a sanity check that the legacy adapter (``decompose_session_legacy``) runs
on the same record without errors -- the apples-to-apples comparison path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pipeline import (                                              # noqa: E402
    _extract_windows_numpy, decompose_session, decompose_session_legacy,
)
from legacy.covariance import extract_valid_segments, extract_windows  # noqa: E402


def _make_aligned_session(seed=0, n_trials=20, n_time=80, n_cells=6,
                          sigma=0.15):
    """Tiny synthetic aligned-session record. Healthy enough that close pairs
    exist and the inclusion filter retains a few cells."""
    rng = np.random.default_rng(seed)
    robs = rng.poisson(0.5, size=(n_trials, n_time, n_cells)).astype(np.float32)
    eye = rng.normal(0, sigma, size=(n_trials, n_time, 2)).astype(np.float32)
    valid = np.ones((n_trials, n_time), bool)
    return {
        "session": "synthetic_test",
        "subject": "Allen",
        "robs": robs, "eyepos": eye, "valid_mask": valid,
        "neuron_mask": np.arange(n_cells),
        "rate_hz": np.full(n_cells, 5.0),
        "psth_r2": np.full(n_cells, 0.2),
        "contam_rate": None,
        "n_trials_total": n_trials, "n_trials_good": n_trials,
        "n_neurons_total": n_cells, "n_neurons_used": n_cells,
    }


def test_extract_windows_matches_legacy():
    """Numpy port and legacy torch implementation produce identical outputs.

    Verifies: counts, trajectories, T_idx all match for a 3-window sweep on a
    synthetic 30-trial fixture. If this drifts, equivalence on real data is
    not trustworthy.
    """
    import torch

    rng = np.random.default_rng(123)
    robs = rng.poisson(0.4, size=(30, 100, 5)).astype(np.float32)
    eye = rng.normal(0, 0.2, size=(30, 100, 2)).astype(np.float32)
    valid = np.ones((30, 100), bool)

    segments = extract_valid_segments(valid, min_len_bins=36)
    assert len(segments) == 30

    for t_count in (1, 2, 6):
        t_hist = max(1, t_count)

        # Numpy port
        counts_np, traj_np, t_np = _extract_windows_numpy(
            robs, eye, segments, t_count, t_hist
        )

        # Legacy torch path (CPU)
        robs_t = torch.tensor(robs, dtype=torch.float32)
        eye_t = torch.tensor(eye, dtype=torch.float32)
        sc, et, ti, _ = extract_windows(
            robs_t, eye_t, segments, t_count, t_hist, device="cpu"
        )
        counts_l = sc.detach().cpu().numpy()
        traj_l = et.detach().cpu().numpy()
        t_l = ti.detach().cpu().numpy()

        assert counts_np.shape == counts_l.shape, (
            f"t_count={t_count}: counts shape mismatch {counts_np.shape} vs {counts_l.shape}"
        )
        np.testing.assert_allclose(counts_np, counts_l, atol=1e-6,
                                   err_msg=f"counts mismatch at t_count={t_count}")
        np.testing.assert_allclose(traj_np, traj_l, atol=1e-6,
                                   err_msg=f"trajectories mismatch at t_count={t_count}")
        np.testing.assert_array_equal(t_np, t_l,
                                      err_msg=f"T_idx mismatch at t_count={t_count}")


def test_decompose_session_schema():
    """decompose_session returns the expected schema (keys, shapes, shuffles)
    and produces finite Crate/Cpsth on a session with healthy close pairs.
    """
    aligned = _make_aligned_session(seed=1, n_trials=40, n_time=80,
                                    n_cells=4, sigma=0.15)
    out = decompose_session(
        aligned, windows_bins=(2,), targets=("naive", "full", "central"),
        n_shuffles=3, threshold=0.4,  # wide threshold -> guaranteed pairs
    )

    assert out["session"] == "synthetic_test"
    assert out["subject"] == "Allen"
    assert len(out["windows"]) == 1
    w = out["windows"][0]
    for key in ("window_bins", "window_ms", "n_samples", "n_close_pairs",
                "Ctotal", "targets"):
        assert key in w, f"missing window key: {key}"
    assert w["window_bins"] == 2
    assert w["n_close_pairs"] > 0
    assert w["Ctotal"].shape == (4, 4)
    assert np.isfinite(w["Ctotal"]).all()

    for tgt in ("naive", "full", "central"):
        assert tgt in w["targets"], f"missing target: {tgt}"
        block = w["targets"][tgt]
        for key in ("Crate", "Cpsth", "Erate", "one_minus_alpha",
                    "Shuffled_Crates"):
            assert key in block, f"target={tgt} missing key={key}"
        assert block["Crate"].shape == (4, 4)
        assert block["Cpsth"].shape == (4, 4)
        assert block["Erate"].shape == (4,)
        assert block["one_minus_alpha"].shape == (4,)

    # Shuffles now run for every target (naive fast path; full/central via the
    # target-reweighted close-pair estimator on shuffled trajectories).
    for tgt in ("naive", "full", "central"):
        cs_list = w["targets"][tgt]["Shuffled_Crates"]
        assert len(cs_list) == 3, f"target={tgt}: expected 3 shuffles"
        for cs in cs_list:
            assert cs.shape == (4, 4)
            assert np.isfinite(cs).all(), f"target={tgt}: non-finite shuffled Crate"


def test_legacy_adapter_runs():
    """The legacy adapter accepts an aligned record and returns the legacy
    schema; this is the comparator path used by compute_methods_data --legacy.
    The legacy snapshot's INTERCEPT_KWARGS pins threshold=0.05, so we need a
    tight enough sigma that close pairs exist on the synthetic fixture.
    """
    aligned = _make_aligned_session(seed=2, n_trials=40, n_time=80,
                                    n_cells=4, sigma=0.03)
    out = decompose_session_legacy(
        aligned, device="cpu", windows_bins=(2,), n_shuffles=2,
    )
    assert "results" in out and "mats" in out
    assert len(out["results"]) == 1
    assert len(out["mats"]) == 1
    res, mats = out["results"][0], out["mats"][0]
    assert "Erates" in res
    assert "Total" in mats and "PSTH" in mats and "Intercept" in mats
    assert mats["Total"].shape == (4, 4)
    assert len(mats["Shuffled_Intercepts"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
