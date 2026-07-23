"""Shared augmentation harness for the variance hunt.

Goal: raise the fraction of the twin's extraretinal ablation gap
``y = full - ablated`` recovered on held-out whole trials, above the ~0.40
saccade-kernel baseline, using ONLY interpretable design columns.

This module is the *single* evaluator every hypothesis plugs into so that all
comparisons share the SAME reliable set, the SAME 5-fold trial CV (seed 0), and
the SAME metric ``recovered = 1 - Var_heldout(y - yhat)/Var_heldout(y)``.

A hypothesis is expressed as one or more **column builders**:

    def my_cols(rec, designs) -> np.ndarray | None   # shape (n_samples, k)

`rec` is a per-neuron record (see `_neuron_records` in `_supp_saccade_model`);
its samples are the valid (trial, bin) pairs `rec["tr"], rec["b"]`. A column
builder returns interpretable features aligned to those samples (or None to
contribute nothing). The evaluator fits, per reliable neuron and on the same
folds:

    baseline:   X_base            = [1, Aadd(L), drive*Aadd(L)]        (the "both" design)
    augmented:  X_aug             = [X_base | extra_columns]

and reports median held-out recovered for both plus the median incremental
Δrecovered = recovered(aug) - recovered(base). ALWAYS report the incremental
number: base is recomputed with the identical code path so the two are exactly
comparable.

Signals exposed on each `designs[session]` dict (all per trial x bin) for use
inside column builders:
    "full", "abl"      (T,B)-per-neuron via rec; rec["drive"] = ablated rate
    "Aadd"     (T,B,L) saccade-onset lag indicators (baseline kernel design)
    "eyepos"   (T,B,2) gaze x,y in deg (NaN outside the 0.5 deg fixation gate)
    "speed"    (T,B)   drift speed deg/s (central diff of eyepos; NaN at edges)
    "sacc_trial","sacc_bin"  int[] microsaccade onsets (trial-pos, bin)
    "B"        int     bins per trial (time-in-trial = rec["b"])

Usage (baseline sanity check -- no extra columns => Δ=0, base ~0.403):
    uv run python paper/supp_twin_saccade_modulation/_supp_saccade_augment.py

To evaluate a hypothesis from another script:
    from _supp_saccade_augment import evaluate_augmentation, load_augment_context
    ctx = load_augment_context()
    res = evaluate_augmentation([my_cols], "my hypothesis", ctx)
    print(res["summary"])
"""
from __future__ import annotations

import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation" / "variance_hunt"))

from _supp_saccade_data import (  # noqa: E402
    load_conditions_cache, build_saccade_alignment, DT, _eye_speed,
)
from _supp_saccade_model import (  # noqa: E402
    _build_designs, _neuron_records, _design_cols, _ols, _recovered,
    N_FOLDS, L, MODEL_CACHE,
)


def _attach_signals(designs, cond, align):
    """Attach per-session eyepos / drift speed / saccade-onset lists to designs."""
    cache_by_session = {r["session"]: r for r in cond}
    for s, d in designs.items():
        eyepos = cache_by_session[s]["eyepos_used"]          # (T, B, 2)
        d["eyepos"] = eyepos
        d["speed"] = _eye_speed(eyepos)                       # (T, B)
        d["sacc_trial"] = align[s]["sacc_trial"]
        d["sacc_bin"] = align[s]["sacc_bin"]
    return designs


def load_augment_context():
    """Build everything a hypothesis needs, once. Reuses the model bundle's
    reliable-unit mask so the reliable set is identical to the baseline paper."""
    cond = load_conditions_cache()
    align = build_saccade_alignment()
    designs = _build_designs(cond, align)
    _attach_signals(designs, cond, align)
    records = _neuron_records(designs)
    valid_recs = [r for r in records if r is not None]

    if not MODEL_CACHE.exists():
        raise FileNotFoundError(
            f"{MODEL_CACHE} missing -- run _supp_saccade_model.py first to build "
            "the reliable-unit mask."
        )
    with open(MODEL_CACHE, "rb") as f:
        bundle = dill.load(f)

    # bundle arrays are in valid_recs order; verify alignment before trusting it.
    b_sessions = bundle["sessions"]
    b_ids = bundle["neuron_ids"]
    assert len(valid_recs) == len(b_sessions), (
        f"record count {len(valid_recs)} != bundle {len(b_sessions)}")
    for r, bs, bid in zip(valid_recs, b_sessions, b_ids):
        assert r["session"] == bs and r["neuron_id"] == int(bid), (
            "valid_recs order does not match model bundle -- caches out of sync")
    good = np.asarray(bundle["good"], dtype=bool)
    reliable_recs = [r for r, g in zip(valid_recs, good) if g]
    return {
        "designs": designs, "valid_recs": valid_recs,
        "reliable_recs": reliable_recs, "good": good,
        "rec_both_baseline": bundle["rec_both"],  # per-neuron paper baseline
    }


def _cv_recovered(X, y, fold):
    """5-fold CV held-out recovered for one neuron given design X."""
    yhat = np.full_like(y, np.nan)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        if tr.sum() < X.shape[1] + 2 or te.sum() == 0:
            continue
        beta = _ols(X[tr], y[tr])
        yhat[te] = X[te] @ beta
    ok = np.isfinite(yhat)
    if ok.sum() <= 10:
        return np.nan
    return _recovered(y[ok], yhat[ok])


def _extra_columns(rec, designs, col_builders):
    """Stack all hypothesis columns for one neuron; returns (S, k) or (S, 0)."""
    S = len(rec["tr"])
    cols = []
    for fn in col_builders:
        c = fn(rec, designs)
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        if c.ndim == 1:
            c = c[:, None]
        assert c.shape[0] == S, (
            f"column builder returned {c.shape[0]} rows, expected {S}")
        # guard against all-NaN / constant-NaN columns leaking in
        c = np.where(np.isfinite(c), c, 0.0)
        cols.append(c)
    if not cols:
        return np.zeros((S, 0))
    return np.hstack(cols)


def evaluate_augmentation(col_builders, label, ctx=None, base_model="both",
                          verbose=True):
    """Fit baseline vs baseline+extra columns per reliable neuron, same folds.

    Parameters
    ----------
    col_builders : list of callables ``f(rec, designs) -> (S,k) | (S,) | None``
    label : str, for logging.
    ctx : dict from `load_augment_context` (built if None).
    base_model : "both" (default), "add", or "mult" -- baseline design form.

    Returns dict with per-neuron arrays and a printable `summary`.
    """
    if ctx is None:
        ctx = load_augment_context()
    designs = ctx["designs"]
    recs = ctx["reliable_recs"]

    base = np.full(len(recs), np.nan)
    aug = np.full(len(recs), np.nan)
    n_cols = 0
    for i, rec in enumerate(recs):
        Xb, _ = _design_cols(rec, designs, base_model)
        y, fold = rec["y"], rec["fold"]
        extra = _extra_columns(rec, designs, col_builders)
        n_cols = extra.shape[1]
        base[i] = _cv_recovered(Xb, y, fold)
        if n_cols:
            aug[i] = _cv_recovered(np.hstack([Xb, extra]), y, fold)
        else:
            aug[i] = base[i]

    delta = aug - base
    med_base = float(np.nanmedian(base))
    med_aug = float(np.nanmedian(aug))
    med_delta = float(np.nanmedian(delta))
    mean_delta = float(np.nanmean(delta))
    frac_improved = float(np.mean(delta[np.isfinite(delta)] > 0))
    # bootstrap-free significance proxy: sign test on paired deltas
    finite = delta[np.isfinite(delta)]
    n_pos = int((finite > 0).sum())
    n_neg = int((finite < 0).sum())

    summary = (
        f"[{label}] extra cols/neuron = {n_cols}, reliable N = {len(recs)}\n"
        f"  median recovered: base {med_base:.4f} -> aug {med_aug:.4f}"
        f"  (Δ median {med_delta:+.4f}, mean {mean_delta:+.4f})\n"
        f"  neurons improved: {frac_improved*100:.0f}%  "
        f"(sign test +{n_pos}/-{n_neg})"
    )
    if verbose:
        print(summary)
    return {
        "label": label, "n_cols": n_cols, "n_neurons": len(recs),
        "base": base, "aug": aug, "delta": delta,
        "med_base": med_base, "med_aug": med_aug, "med_delta": med_delta,
        "mean_delta": mean_delta, "frac_improved": frac_improved,
        "n_pos": n_pos, "n_neg": n_neg, "summary": summary,
    }


# --- example column builders (also serve as the baseline sanity check) ------
def cols_time_in_trial(rec, designs):
    """Interpretable example: linear + quadratic time-in-trial (bin index)."""
    B = designs[rec["session"]]["B"]
    t = rec["b"].astype(np.float64) / B          # normalized [0,1)
    return np.column_stack([t, t * t])


if __name__ == "__main__":
    ctx = load_augment_context()
    print(f"reliable neurons: {len(ctx['reliable_recs'])}")
    # (1) no extra columns => Δ must be ~0 and base must reproduce paper 0.403
    res0 = evaluate_augmentation([], "baseline-check (no extra cols)", ctx)
    paper = float(np.nanmedian(ctx["rec_both_baseline"][ctx["good"]]))
    print(f"  paper baseline (from model bundle) = {paper:.4f}")
    print(f"  harness base                        = {res0['med_base']:.4f}")
    # (2) a trivial interpretable augmentation, as a plumbing demo
    evaluate_augmentation([cols_time_in_trial], "demo: time-in-trial (quad)", ctx)
