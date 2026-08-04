"""Per-unit single-trial ceiling, extracted from the stage-1 LOTC decomposition.

For a neuron with observed counts ``y`` and conditional rate
``lambda = E[y | stimulus, gaze]``, the law of total variance gives

    Var(y) = Var(lambda) + E[Var(y | z)],

and any predictor that is a function of the conditioning variables satisfies

    E[(y - yhat)^2] = E[(lambda - yhat)^2] + E[Var(y | z)] >= E[Var(y | z)],

with equality only at ``yhat = lambda``. So single-trial r^2 is bounded by

    R2_max = Var(lambda) / Var(y) = diag(Crate) / diag(Ctotal),

which is exact -- it assumes only that the residual is conditionally mean-zero
given ``z``, not that it is Poisson. Stage 1 already estimates both terms:
``Ctotal`` is the sample covariance of windowed counts and ``Crate`` is the
close-pair estimator, reweighted (``target='full'``) so that the rate variance
is measured under the actual viewing distribution -- the same distribution the
single-trial r^2 averages over, which is what makes the ratio a fraction of
anything.

This module does no estimation. It reshapes numbers already in
``covdecomp_empirical.pkl`` into a small table keyed by ``(session, neuron_id)``
so downstream figures can normalize an r^2 without opening the 8 GB stage-1
cache. ``neuron_id`` is the dataset's own neuron index (from ``neuron_mask``),
matching the key ``_fig2_inclusion()`` builds.

Undefined rather than clipped: a unit whose estimated rate variance is
non-positive (the close-pair estimator is noisy and unconstrained) gets NaN, so
callers exclude it instead of plotting a clipped ratio. ``one_minus_alpha`` is
likewise left unclipped, matching fig2 panel E rather than the clipped
``decompose.py`` field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from VisionCore.paths import CACHE_DIR

TARGET = "full"

# The twin runs at 120 Hz (dataset configs decimate 240 -> 120), so one
# decomposition bin is the model's own resolution and the window at which a
# per-bin single-trial r^2 must be normalized. R2_max is strongly
# window-dependent -- summing counts averages private noise down faster than
# rate variance -- so this is not a free choice.
from fig3_windows import FIG3_SINGLETRIAL_WINDOW_BINS

PRODUCTION_WINDOW_BINS = FIG3_SINGLETRIAL_WINDOW_BINS

CEILING_CACHE = CACHE_DIR / "covdecomp_ceiling.pkl"


def _entry(c_tot, c_rate, c_psth, shuffled_rates):
    """Per-neuron ceiling record for one counting window."""
    r2_max = c_rate / c_tot if (c_tot > 0 and c_rate > 0) else np.nan
    r2_psth = c_psth / c_tot if c_tot > 0 else np.nan
    one_minus_alpha = 1.0 - c_psth / c_rate if c_rate > 0 else np.nan

    if len(shuffled_rates):
        p_rate = float(np.mean(np.asarray(shuffled_rates) >= c_rate))
    else:
        p_rate = np.nan

    return {
        "c_tot": float(c_tot),
        "c_rate": float(c_rate),
        "c_psth": float(c_psth),
        "r2_max": float(r2_max),
        "r2_psth": float(r2_psth),
        "one_minus_alpha": float(one_minus_alpha),
        "p_rate": p_rate,
    }


def ceiling_from_session(sr, target=TARGET):
    """``{neuron_id: {window_bins: entry}}`` for one stage-1 session record."""
    neuron_ids = np.asarray(sr["neuron_mask"])
    out = {int(nid): {} for nid in neuron_ids}

    for w in sr["windows"]:
        blk = w["targets"][target]
        c_tot = np.diag(w["Ctotal"]).astype(float)
        c_rate = np.diag(blk["Crate"]).astype(float)
        c_psth = np.diag(blk["Cpsth"]).astype(float)
        shuf = np.asarray([np.diag(s) for s in blk.get("Shuffled_Crates", [])],
                          dtype=float)

        for k, nid in enumerate(neuron_ids):
            out[int(nid)][int(w["window_bins"])] = _entry(
                c_tot[k], c_rate[k], c_psth[k],
                shuf[:, k] if shuf.size else np.empty(0))
    return out


def build_ceiling_table(session_results, target=TARGET):
    """``{(session, neuron_id): {window_bins: entry}}`` across sessions."""
    table = {}
    for sr in session_results:
        for nid, per_window in ceiling_from_session(sr, target=target).items():
            table[(sr["session"], nid)] = per_window
    return table


def load_ceiling(refresh=False, cache_path=CEILING_CACHE):
    """Load the small ceiling table, building it from stage 1 if needed."""
    import dill

    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            return dill.load(f)

    from decompose import compute_decomposition

    print("Building ceiling table from the stage-1 decomposition "
          "(loads the multi-GB empirical cache once) ...")
    table = build_ceiling_table(compute_decomposition())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        dill.dump(table, f)
    print(f"Cached {len(table)} (session, neuron) ceilings to {cache_path}")
    return table


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the per-unit ceiling table.")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    table = load_ceiling(refresh=args.refresh)
    w = PRODUCTION_WINDOW_BINS
    r2 = np.array([e[w]["r2_max"] for e in table.values() if w in e])
    print(f"{len(table)} units; window_bins={w}: "
          f"R2_max median={np.nanmedian(r2):.4f}, "
          f"{np.mean(~np.isfinite(r2)) * 100:.1f}% undefined")
