"""Tests for the per-unit single-trial ceiling extracted from the LOTC
decomposition (`paper/covariance_decomposition/ceiling.py`).

These pin the table's contracts -- the join key, the undefined-not-clipped rule,
and the shuffle-null p -- against hand-built stage-1 records. They do not test
the close-pair estimator itself; `Crate`/`Ctotal` are produced by
`decompose.py`, which fig2 already rests on.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COVDECOMP_DIR = REPO_ROOT / "paper" / "covariance_decomposition"
for _p in (str(REPO_ROOT), str(COVDECOMP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _stage1_record(session, neuron_mask, c_tot, c_rate, c_psth,
                   shuffled_crates=()):
    """Minimal stage-1 shaped record carrying only the diagonals under test."""
    n = len(neuron_mask)
    return {
        "session": session,
        "subject": session.split("_")[0],
        "neuron_mask": np.asarray(neuron_mask),
        "rate_hz": np.full(n, 10.0),
        "psth_r2": np.full(n, 0.5),
        "qc": {"contam_rate": np.zeros(n)},
        "meta": {},
        "windows": [{
            "window_bins": 1,
            "window_ms": 1000 / 120,
            "n_samples": 1000,
            "n_close_pairs": 500,
            "Ctotal": np.diag(np.asarray(c_tot, dtype=float)),
            "targets": {"full": {
                "Crate": np.diag(np.asarray(c_rate, dtype=float)),
                "Cpsth": np.diag(np.asarray(c_psth, dtype=float)),
                "Erate": np.full(n, 0.3),
                "one_minus_alpha": np.full(n, np.nan),
                "Shuffled_Crates": [np.diag(np.asarray(s, dtype=float))
                                    for s in shuffled_crates],
            }},
        }],
    }


def test_table_is_keyed_by_original_neuron_ids():
    """Keys carry the dataset's neuron ids, not the compacted column index."""
    from ceiling import build_ceiling_table

    sr = _stage1_record("Allen_2022-02-16", [3, 7, 11],
                        c_tot=[1.0, 2.0, 4.0],
                        c_rate=[0.2, 0.5, 1.0],
                        c_psth=[0.1, 0.1, 0.4])

    table = build_ceiling_table([sr])

    assert set(table) == {("Allen_2022-02-16", 3),
                          ("Allen_2022-02-16", 7),
                          ("Allen_2022-02-16", 11)}
    assert table[("Allen_2022-02-16", 7)][1]["r2_max"] == pytest.approx(0.25)
    assert table[("Allen_2022-02-16", 11)][1]["one_minus_alpha"] == pytest.approx(0.6)


def test_nonpositive_rate_variance_is_undefined_not_clipped():
    """A unit whose estimated rate variance is <= 0 gets NaN, so the figure
    excludes it rather than plotting a clipped or infinite ratio."""
    from ceiling import build_ceiling_table

    sr = _stage1_record("Allen_2022-02-16", [0, 1],
                        c_tot=[1.0, 0.0],
                        c_rate=[-0.05, 0.5],
                        c_psth=[0.0, 0.1])

    table = build_ceiling_table([sr])

    assert np.isnan(table[("Allen_2022-02-16", 0)][1]["r2_max"])
    assert np.isnan(table[("Allen_2022-02-16", 1)][1]["r2_max"])


def test_shuffle_null_gives_a_one_sided_p_for_rate_variance():
    """`p_rate` is the fraction of eye-shuffles whose rate variance reaches the
    real one -- the resolvability rule the figure excludes on."""
    from ceiling import build_ceiling_table

    sr = _stage1_record("Allen_2022-02-16", [0, 1],
                        c_tot=[1.0, 1.0],
                        c_rate=[0.5, 0.02],
                        c_psth=[0.1, 0.01],
                        shuffled_crates=[[0.01, 0.03], [0.02, 0.04],
                                         [0.03, 0.05], [0.04, 0.01]])

    table = build_ceiling_table([sr])

    assert table[("Allen_2022-02-16", 0)][1]["p_rate"] == pytest.approx(0.0)
    assert table[("Allen_2022-02-16", 1)][1]["p_rate"] == pytest.approx(0.75)
