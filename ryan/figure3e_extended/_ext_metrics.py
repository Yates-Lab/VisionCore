"""Single-trial Poisson metrics for the figure3e_extended ladder.

Split out of `_ext_data.py` so the metrics can be tested without importing the
renderer, the data packages, or a GPU checkpoint.

Two scores share one preparation step (valid-bin mask + positive-rate floor) so
that they describe exactly the same bins:

`bits_per_spike`
    The existing row: `models.losses.calc_poisson_bits_per_spike`, i.e. the
    Poisson log-likelihood gain over each unit's own mean-rate null, per
    observed spike.

`poisson_self_consistency`
    The observed likelihood gain divided by the gain expected if the predicted
    rates themselves generated independent Poisson counts,

        S = G_obs / G_self,
        G_obs  = sum_i [ y_i log(q_i/q0_i) - q_i + q0_i ]
        G_self = sum_i [ q_i log(q_i/q0_i) - q_i + q0_i ]
               = sum_i KL[ Poisson(q_i) || Poisson(q0_i) ].

    `G_self` is `E[G_obs]` under `Y_i ~ Poisson(q_i)` with the null held fixed,
    which follows from `E[Y_i] = q_i` and the fact that `G` is affine in `y`.
    No sampling is involved; the Monte Carlo check lives in the unit tests.

    `S = 1` in expectation when `q` is the true conditional Poisson mean,
    `0 < S < 1` when the counts realize less gain than the model claims under
    its own rates, and `S < 0` when the null predicts better. `S > 1` is
    possible: the denominator is an expectation, not a hard bound.

CAVEAT (deliberate, for comparability with the bits/spike row). The affine
rescaling applied upstream and the mean-rate null used here are both estimated
from the very responses being scored, so this is not a cross-fitted estimate and
the expected-one result holds only up to that in-sample calibration. The
alternative -- cross-fitting the calibration and the null -- would change the
bits/spike row too, so it is left as a separate decision.

The numerator is deliberately the same quantity `calc_poisson_bits_per_spike`
integrates, down to its `1e-8` log offset, so the two rows are two
normalizations of one likelihood gain. Because `G` is affine in `y` with the
coefficient `log(q + EPS) - log(q0 + EPS)` held fixed, carrying that offset into
both terms leaves `E[G_obs] = G_self` exact.
"""
import sys

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))


# Log offset inside `calc_poisson_bits_per_spike`; mirrored here so the observed
# gain reproduces that function's numerator exactly.
EPS = 1e-8

# Poisson likelihood needs a strictly positive rate; the affine rescaling that
# makes the r^2 axis comparable across conditions can push a few bins
# non-positive.
BPS_FLOOR = 1e-6

# `G_self` is a sum of KL divergences and vanishes only when the prediction is
# the null in every valid bin. Below this fraction of the total predicted count
# the ratio is numerically meaningless, so it is reported as undefined rather
# than clipped.
SELF_GAIN_REL_TOL = 1e-12


def _prepare(rhat, robs, dfs):
    """Flatten to (n_bins, n_units), mask, and floor the predicted rate.

    Bins that are NaN in either array, or masked by `dfs`, are excluded.
    Returns `(q, y, valid, n_floored)`, where `n_floored` counts valid bins whose
    predicted rate had to be raised to `BPS_FLOOR` for the log."""
    n_units = robs.shape[-1]
    rhat = np.asarray(rhat, dtype=np.float64).reshape(-1, n_units)
    robs = np.asarray(robs, dtype=np.float64).reshape(-1, n_units)
    dfs = np.asarray(dfs, dtype=np.float64).reshape(-1, n_units)

    valid = np.isfinite(robs) & np.isfinite(rhat) & np.isfinite(dfs) & (dfs > 0)
    q = np.where(valid, rhat, 1.0)
    n_floored = int(((q < BPS_FLOOR) & valid).sum())
    q = np.maximum(q, BPS_FLOOR)
    y = np.where(valid, robs, 0.0)
    return q, y, valid, n_floored


def _mean_rate_null(y, valid):
    """The per-unit mean-rate null `calc_poisson_bits_per_spike` builds: total
    spikes over total valid bins, each clamped at 1 exactly as that function
    clamps them. Returns `(q0, n_spikes)`, both (n_units,)."""
    n_bins = np.maximum(valid.sum(axis=0).astype(np.float64), 1.0)
    n_spikes = np.maximum((valid * y).sum(axis=0), 1.0)
    return n_spikes / n_bins, n_spikes


def poisson_likelihood_gain(q, y, q0, valid):
    """Observed and self-referential Poisson log-likelihood gain, in nats.

    Parameters
    ----------
    q, y, valid : (n_bins, n_units)
        Predicted mean counts (strictly positive), observed counts, and the
        valid-bin mask. Entries outside the mask are ignored.
    q0 : broadcastable to (n_bins, n_units)
        The null prediction. It must be fixed independently of `y` for
        `g_self` to be the expectation of `g_obs`.

    Returns
    -------
    g_obs, g_self : (n_units,)
        `sum_i [y_i a_i - q_i + q0_i]` and `sum_i [q_i a_i - q_i + q0_i]` with
        `a_i = log(q_i + EPS) - log(q0_i + EPS)`. The second is
        `sum_i KL[Poisson(q_i) || Poisson(q0_i)]` and equals `E[g_obs]` under
        `Y_i ~ Poisson(q_i)`.
    """
    a = np.log(q + EPS) - np.log(q0 + EPS)
    base = q0 - q
    g_obs = np.where(valid, y * a + base, 0.0).sum(axis=0)
    g_self = np.where(valid, q * a + base, 0.0).sum(axis=0)
    return g_obs, g_self


def bits_per_spike(rhat, robs, dfs):
    """Per-neuron Poisson bits/spike. Arrays are (..., n_units); bins that are
    NaN in either array, or masked by `dfs`, are excluded.

    Returns `(bps, n_floored)`."""
    import torch
    from models.losses import calc_poisson_bits_per_spike

    q, y, valid, n_floored = _prepare(rhat, robs, dfs)
    bps = calc_poisson_bits_per_spike(
        torch.from_numpy(q),
        torch.from_numpy(y),
        torch.from_numpy(valid.astype(np.float64)),
    ).numpy()
    return bps, n_floored


def poisson_self_consistency(rhat, robs, dfs):
    """Per-neuron Poisson self-consistency fraction `S = G_obs / G_self`.

    Same arrays, same valid-bin mask, same positive-rate floor, and the same
    mean-rate null as `bits_per_spike`, so the numerator here is that row's
    likelihood gain before its per-spike normalization.

    Returns `(S, g_obs, g_self, n_floored)`. `S` is NaN for units whose
    denominator is zero or numerically indistinguishable from it -- a prediction
    that is the null carries no self-reference, and the ratio is left undefined
    rather than clipped."""
    q, y, valid, n_floored = _prepare(rhat, robs, dfs)
    q0, _ = _mean_rate_null(y, valid)
    g_obs, g_self = poisson_likelihood_gain(q, y, q0[None, :], valid)

    tol = SELF_GAIN_REL_TOL * np.maximum(1.0, np.where(valid, q, 0.0).sum(axis=0))
    defined = g_self > tol
    S = np.divide(g_obs, g_self, out=np.full_like(g_self, np.nan), where=defined)
    return S, g_obs, g_self, n_floored
