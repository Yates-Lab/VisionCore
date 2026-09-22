import numpy as np

from paper.model_selection.audit_fixrsvp_trace_cache import (
    _data_support,
    independent_ccabs,
    independent_ccmax,
)


def test_independent_ccabs_is_positive_affine_invariant_and_data_masked():
    rng = np.random.default_rng(9)
    signal = np.sin(np.linspace(0, 4 * np.pi, 36))[None, :, None]
    robs = 1.5 + signal + 0.25 * rng.standard_normal((42, 36, 2))
    prediction = np.broadcast_to(2.0 + 0.8 * signal, robs.shape).copy()
    dfs = np.ones_like(robs)
    dfs[0, 0, 0] = np.nan
    dfs[1, 1, 1] = 0
    support = _data_support(robs, dfs)

    baseline = independent_ccabs(robs, prediction, support)
    transformed = independent_ccabs(robs, 2.3 * prediction + 0.7, support)

    np.testing.assert_allclose(baseline, transformed, rtol=0, atol=2e-12)
    assert not support[0, 0, 0]
    assert not support[1, 1, 1]


def test_independent_ccmax_depends_only_on_data_and_mask():
    rng = np.random.default_rng(19)
    signal = np.cos(np.linspace(0, 3 * np.pi, 32))[None, :, None]
    robs = 2.0 + signal + 0.4 * rng.standard_normal((40, 32, 2))
    support = np.ones_like(robs, dtype=bool)

    first = independent_ccmax(robs, support, seed=42, n_splits=7)
    second = independent_ccmax(robs.copy(), support.copy(), seed=42, n_splits=7)

    np.testing.assert_allclose(first, second, rtol=0, atol=0, equal_nan=True)
