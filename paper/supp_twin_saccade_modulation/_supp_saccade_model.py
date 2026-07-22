"""Model layer: does a simple saccade modulator reproduce the twin's learned one?

Reference: McFarland et al. 2015 (Nat Commun) — saccades drive V1 through early
divisive gain suppression + a later additive offset. Here we ask the same of the
twin's extraretinal contribution: treat the twin as the "neuron", the ablated
(behavior-zeroed) rate as the stimulus drive, and fit a saccade-locked
gain + offset modulator to the gap ``y = full - ablated``:

    y_i(t) = drive_i(t) * Σ_k g_i(t - t_k)  +  Σ_k a_i(t - t_k)              (both)

with per-neuron kernels g_i (gain, multiplicative) and a_i (offset, additive),
convolved over that trial's (micro)saccade onsets t_k (linear superposition).
Nested variants: additive-only (g≡0), multiplicative-only (a≡0).

Model ladder (all scored the same way):
  - per-neuron   : free per-lag kernels g_i(τ), a_i(τ)      (ceiling)
  - pooled       : rank-1 shared shapes  g_i = wᵍ_i·g₀(τ),  a_i = wᵃ_i·a₀(τ)
  - pooled+δ     : + per-neuron latency shift δ_i (grid search)

Headline metric — fraction of the ablation gap recovered on **held-out whole
trials** (5-fold CV over trials within session):

    recovered = 1 - Var_heldout(y - ŷ) / Var_heldout(y)

recovered(per-neuron)   = how much of the twin's extraretinal effect is
                          microsaccade-*locked* at all.
recovered(pooled)/ceiling = how *stereotyped* (simple) that locked part is.

Reuses the caches from _supp_saccade_data (0.5° conditions cache + bit-wise-gated
saccade→(trial,bin) alignment + reliability). No new inference.

Usage:
    uv run python paper/supp_twin_saccade_modulation/_supp_saccade_model.py [--force]
"""
from __future__ import annotations

import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _supp_saccade_data import (  # noqa: E402
    load_conditions_cache, build_saccade_alignment, compute_sta_bundle,
    STA_LAGS, DT, RELIABLE_THRESHOLD, EXAMPLE_SESSION, EXAMPLE_NEURON_ID,
)

L = len(STA_LAGS)
N_FOLDS = 5
FOLD_SEED = 0
ALS_ITERS = 12
DELTA_GRID = np.arange(-6, 7)          # bins (~±50 ms) for the latency tier
RIDGE = 1e-6                            # numerical stabilizer (≈ OLS)
MODEL_CACHE = CACHE_DIR / "supp_saccade_model.pkl"


# --- linear algebra helpers ------------------------------------------------
def _ols(X, y, ridge=RIDGE):
    XtX = X.T @ X
    XtX[np.diag_indices_from(XtX)] += ridge * (np.trace(XtX) / X.shape[1] + 1e-12)
    return np.linalg.solve(XtX, X.T @ y)


def _recovered(y, yhat):
    ybar = y.mean()
    ss = ((y - ybar) ** 2).sum()
    return 1.0 - ((y - yhat) ** 2).sum() / (ss + 1e-12)


def _shift_kernel(k, delta):
    """Shift kernel so its features peak δ bins later (zero-filled edges)."""
    if delta == 0:
        return k
    out = np.zeros_like(k)
    if delta > 0:
        out[delta:] = k[:-delta]
    else:
        out[:delta] = k[-delta:]
    return out


# --- session designs -------------------------------------------------------
def _build_designs(cond, align):
    """Per session: additive design Aadd (T,B,L), rates, valid mask, fold ids."""
    designs = {}
    for rec in cond:
        s = rec["session"]
        if s not in align:
            continue
        a = align[s]
        if len(a["sacc_trial"]) == 0:
            continue
        full = rec["rhat_used"]["intact"] / DT
        abl = rec["rhat_used"]["zeroed"] / DT
        dfs = rec["dfs_used"]
        T, B, N = full.shape
        Aadd = np.zeros((T, B, L), dtype=np.float32)
        for tr, b0 in zip(a["sacc_trial"], a["sacc_bin"]):
            for li, lag in enumerate(STA_LAGS):
                b = b0 + lag
                if 0 <= b < B:
                    Aadd[tr, b, li] += 1.0
        rng = np.random.default_rng(FOLD_SEED)
        fold_of_trial = rng.integers(0, N_FOLDS, size=T)
        designs[s] = {
            "full": full, "abl": abl, "dfs": dfs, "Aadd": Aadd,
            "neuron_mask": rec["neuron_mask"], "subject": rec["subject"],
            "fold_of_trial": fold_of_trial, "T": T, "B": B, "N": N,
        }
    return designs


def _neuron_records(designs):
    """Flatten to per-neuron records: coords, y (gap), drive, fold."""
    records = []
    for s, d in designs.items():
        full, abl, dfs = d["full"], d["abl"], d["dfs"]
        valid_all = (dfs > 0) & np.isfinite(full) & np.isfinite(abl)
        for ni in range(d["N"]):
            v = valid_all[:, :, ni]
            tr, b = np.where(v)
            if len(tr) < 200:
                records.append(None)
                continue
            records.append({
                "session": s, "ni": ni, "tr": tr.astype(np.int32),
                "b": b.astype(np.int32),
                "y": (full[tr, b, ni] - abl[tr, b, ni]).astype(np.float64),
                "drive": abl[tr, b, ni].astype(np.float64),
                "fold": d["fold_of_trial"][tr].astype(np.int8),
                "neuron_id": int(d["neuron_mask"][ni]),
                "subject": d["subject"],
            })
    return records


def _design_cols(rec, designs, model):
    """Return the design matrix (samples × p) for `model` and the kernel-column
    slice(s) needed to read kernels back."""
    Aadd_s = designs[rec["session"]]["Aadd"][rec["tr"], rec["b"], :]  # (S, L)
    ones = np.ones((Aadd_s.shape[0], 1))
    if model == "add":
        return np.hstack([ones, Aadd_s]), Aadd_s
    if model == "mult":
        return np.hstack([ones, rec["drive"][:, None] * Aadd_s]), Aadd_s
    Madd_s = rec["drive"][:, None] * Aadd_s
    return np.hstack([ones, Aadd_s, Madd_s]), Aadd_s


# --- per-neuron tier (ceiling) --------------------------------------------
def _fit_per_neuron(rec, designs, model="both"):
    """5-fold CV recovered for one neuron; also returns full-data kernels (both)."""
    X, _ = _design_cols(rec, designs, model)
    y, fold = rec["y"], rec["fold"]
    yhat = np.full_like(y, np.nan)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        if tr.sum() < X.shape[1] + 2 or te.sum() == 0:
            continue
        beta = _ols(X[tr], y[tr])
        yhat[te] = X[te] @ beta
    ok = np.isfinite(yhat)
    rec_cv = _recovered(y[ok], yhat[ok]) if ok.sum() > 10 else np.nan
    kernels = None
    if model == "both":
        beta = _ols(X, y)
        kernels = {"a": beta[1:1 + L], "g": beta[1 + L:1 + 2 * L],
                   "intercept": beta[0]}
    return rec_cv, kernels


# --- pooled tier (rank-1 shared waveforms via ALS) -------------------------
def _als(recs, designs, a0, g0, iters=ALS_ITERS):
    """Alternating least squares for shared g₀,a₀ + per-neuron (c,wa,wg)."""
    # Precompute per-neuron train features once (Aadd_s, Madd_s, y).
    feats = []
    for r in recs:
        Aadd_s = designs[r["session"]]["Aadd"][r["tr"], r["b"], :]
        feats.append((Aadd_s, r["drive"][:, None] * Aadd_s, r["y"]))
    params = np.zeros((len(recs), 3))  # c, wa, wg
    for _ in range(iters):
        # weight step: per-neuron 3-param OLS given shapes
        for i, (Aadd_s, Madd_s, y) in enumerate(feats):
            fa = Aadd_s @ a0
            fg = Madd_s @ g0
            X = np.column_stack([np.ones_like(fa), fa, fg])
            params[i] = _ols(X, y)
        # waveform step: pooled 2L OLS given weights
        XtX = np.zeros((2 * L, 2 * L))
        Xty = np.zeros(2 * L)
        for i, (Aadd_s, Madd_s, y) in enumerate(feats):
            c, wa, wg = params[i]
            D = np.hstack([wa * Aadd_s, wg * Madd_s])
            XtX += D.T @ D
            Xty += D.T @ (y - c)
        XtX[np.diag_indices_from(XtX)] += RIDGE * (np.trace(XtX) / (2 * L) + 1e-12)
        sol = np.linalg.solve(XtX, Xty)
        a0, g0 = sol[:L], sol[L:]
        na, ng = np.linalg.norm(a0) + 1e-12, np.linalg.norm(g0) + 1e-12
        a0, g0 = a0 / na, g0 / ng
        params[:, 1] *= na
        params[:, 2] *= ng
    return a0, g0, params, feats


def _pooled_eval(rec, designs, a0, g0, use_delta):
    """Fit this neuron's (c,wa,wg[,δ]) on train folds, score held-out folds."""
    Aadd_s = designs[rec["session"]]["Aadd"][rec["tr"], rec["b"], :]
    Madd_s = rec["drive"][:, None] * Aadd_s
    y, fold = rec["y"], rec["fold"]
    yhat = np.full_like(y, np.nan)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        if tr.sum() < 10 or te.sum() == 0:
            continue
        best = None
        deltas = DELTA_GRID if use_delta else [0]
        for dl in deltas:
            aS = _shift_kernel(a0, dl)
            gS = _shift_kernel(g0, dl)
            fa = Aadd_s @ aS
            fg = Madd_s @ gS
            X = np.column_stack([np.ones_like(fa), fa, fg])
            beta = _ols(X[tr], y[tr])
            sse = ((y[tr] - X[tr] @ beta) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, beta, X)
        yhat[te] = best[2][te] @ best[1]
    ok = np.isfinite(yhat)
    return _recovered(y[ok], yhat[ok]) if ok.sum() > 10 else np.nan


def _fit_pooled_cv(recs, designs, per_neuron_kernels, use_delta):
    """Pooled recovered per neuron, CV over trials. Shared shapes are refit on
    each fold's TRAIN trials (leak-free) via ALS, initialized from the mean of
    the per-neuron kernels."""
    # init shapes from the mean of per-neuron 'both' kernels
    A = np.array([k["a"] for k in per_neuron_kernels if k is not None])
    G = np.array([k["g"] for k in per_neuron_kernels if k is not None])
    a_init = A.mean(0) / (np.linalg.norm(A.mean(0)) + 1e-12)
    g_init = G.mean(0) / (np.linalg.norm(G.mean(0)) + 1e-12)

    yhat_store = [np.full_like(r["y"], np.nan) for r in recs]
    for f in range(N_FOLDS):
        # ALS on all reliable neurons' train samples for this fold
        train_recs = [{**r, "tr": r["tr"][r["fold"] != f],
                       "b": r["b"][r["fold"] != f],
                       "y": r["y"][r["fold"] != f],
                       "drive": r["drive"][r["fold"] != f]} for r in recs]
        a0, g0, _, _ = _als(train_recs, designs, a_init.copy(), g_init.copy())
        for i, r in enumerate(recs):
            te = r["fold"] == f
            tr = r["fold"] != f
            if tr.sum() < 10 or te.sum() == 0:
                continue
            Aadd_s = designs[r["session"]]["Aadd"][r["tr"], r["b"], :]
            Madd_s = r["drive"][:, None] * Aadd_s
            best = None
            for dl in (DELTA_GRID if use_delta else [0]):
                aS, gS = _shift_kernel(a0, dl), _shift_kernel(g0, dl)
                fa, fg = Aadd_s @ aS, Madd_s @ gS
                X = np.column_stack([np.ones_like(fa), fa, fg])
                beta = _ols(X[tr], r["y"][tr])
                sse = ((r["y"][tr] - X[tr] @ beta) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, beta, X)
            yhat_store[i][te] = best[2][te] @ best[1]
    out = np.full(len(recs), np.nan)
    for i, r in enumerate(recs):
        ok = np.isfinite(yhat_store[i])
        if ok.sum() > 10:
            out[i] = _recovered(r["y"][ok], yhat_store[i][ok])
    return out, (a_init, g_init)


# --- orchestration ---------------------------------------------------------
def compute_model_bundle(force=False):
    if MODEL_CACHE.exists() and not force:
        print(f"Loading model bundle from {MODEL_CACHE}")
        with open(MODEL_CACHE, "rb") as f:
            return dill.load(f)

    cond = load_conditions_cache()
    align = build_saccade_alignment()
    sta = compute_sta_bundle()
    designs = _build_designs(cond, align)
    records = _neuron_records(designs)

    # reliability aligned to records (same session/neuron order as STA bundle)
    rel = sta["reliability"]
    # STA bundle iterated cond sessions in order, all neurons; _neuron_records
    # skips <200-sample neurons (None). Build a parallel reliability array.
    rel_by_key = {}
    idx = 0
    for rec in cond:
        s = rec["session"]
        if s not in align or len(align[s]["sacc_trial"]) == 0:
            continue  # STA bundle skipped these without advancing its index
        for ni in range(rec["n_neurons"]):
            rel_by_key[(s, ni)] = rel[idx]
            idx += 1

    valid_recs = [r for r in records if r is not None]
    reliability = np.array([rel_by_key.get((r["session"], r["ni"]), np.nan)
                            for r in valid_recs])
    subjects = np.array([r["subject"] for r in valid_recs])
    good = np.isfinite(reliability) & (reliability > RELIABLE_THRESHOLD)
    print(f"neurons: {len(valid_recs)} total, {int(good.sum())} reliable")

    # --- per-neuron tier (ceiling) for all neurons, all three model forms ---
    print("Fitting per-neuron tier (add / mult / both) ...")
    rec_add = np.full(len(valid_recs), np.nan)
    rec_mult = np.full(len(valid_recs), np.nan)
    rec_both = np.full(len(valid_recs), np.nan)
    kernels = [None] * len(valid_recs)
    for i, r in enumerate(valid_recs):
        rec_add[i], _ = _fit_per_neuron(r, designs, "add")
        rec_mult[i], _ = _fit_per_neuron(r, designs, "mult")
        rec_both[i], kernels[i] = _fit_per_neuron(r, designs, "both")
        if i % 300 == 0:
            print(f"  {i}/{len(valid_recs)}")

    # --- pooled tiers on reliable neurons ---
    good_recs = [r for r, g in zip(valid_recs, good) if g]
    good_kernels = [k for k, g in zip(kernels, good) if g]
    print(f"Pooled ALS on {len(good_recs)} reliable neurons (no-δ) ...")
    rec_pooled_g, (a_init, g_init) = _fit_pooled_cv(
        good_recs, designs, good_kernels, use_delta=False)
    print("Pooled ALS (+δ) ...")
    rec_pooled_d_g, _ = _fit_pooled_cv(
        good_recs, designs, good_kernels, use_delta=True)

    # map pooled results back to full-length arrays
    rec_pooled = np.full(len(valid_recs), np.nan)
    rec_pooled_d = np.full(len(valid_recs), np.nan)
    rec_pooled[np.where(good)[0]] = rec_pooled_g
    rec_pooled_d[np.where(good)[0]] = rec_pooled_d_g

    # --- shared waveforms on ALL reliable data (display) ---
    print("Fitting display waveforms (all reliable data) ...")
    a0, g0, params, _ = _als(good_recs, designs, a_init.copy(), g_init.copy())
    med_wa = np.median(params[:, 1])
    med_wg = np.median(params[:, 2])

    # --- example neuron payload ---
    example = _example_payload(valid_recs, designs, kernels, a0, g0)

    bundle = {
        "reliability": reliability, "subjects": subjects, "good": good,
        "neuron_ids": np.array([r["neuron_id"] for r in valid_recs]),
        "sessions": np.array([r["session"] for r in valid_recs]),
        "rec_add": rec_add, "rec_mult": rec_mult, "rec_both": rec_both,
        "rec_pooled": rec_pooled, "rec_pooled_delta": rec_pooled_d,
        "waveform_a0": a0, "waveform_g0": g0, "wa": params[:, 1],
        "wg": params[:, 2], "med_wa": med_wa, "med_wg": med_wg,
        "lags_ms": STA_LAGS * DT * 1000.0,
        "example": example,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CACHE, "wb") as f:
        dill.dump(bundle, f)
    print(f"Cached model bundle to {MODEL_CACHE}")
    return bundle


def _example_payload(valid_recs, designs, kernels, a0, g0):
    idx = next((i for i, r in enumerate(valid_recs)
                if r["session"] == EXAMPLE_SESSION
                and r["neuron_id"] == EXAMPLE_NEURON_ID), None)
    if idx is None:
        return None
    r = valid_recs[idx]
    k = kernels[idx]
    d = designs[r["session"]]
    Aadd = d["Aadd"]                                   # (T, B, L)
    li0 = int(np.where(STA_LAGS == 0)[0][0])
    onset_tr, onset_b = np.where(Aadd[:, :, li0] > 0)  # saccade onsets
    full = d["full"][:, :, r["ni"]]
    abl = d["abl"][:, :, r["ni"]]
    gap = full - abl
    valid = d["dfs"][:, :, r["ni"]] > 0

    # Per-sample model prediction maps (exact; no mean-drive reconstruction).
    add_map = Aadd @ k["a"]                             # (T, B)
    mult_map = abl * (Aadd @ k["g"])
    pred_map = k["intercept"] + add_map + mult_map

    def _sta(map2d):
        acc = np.zeros(L); cnt = np.zeros(L)
        for tr, b0 in zip(onset_tr, onset_b):
            for li, lag in enumerate(STA_LAGS):
                b = b0 + lag
                if 0 <= b < d["B"] and valid[tr, b]:
                    acc[li] += map2d[tr, b]; cnt[li] += 1
        with np.errstate(invalid="ignore"):
            return acc / cnt

    sta_gap = _sta(gap)
    fit_full = _sta(pred_map)
    fit_add = _sta(add_map)
    fit_mult = _sta(mult_map)

    # example held-out trial: refit pooled excluding that trial's fold
    ex_trial = int(onset_tr[np.argmax([(onset_tr == t).sum() for t in onset_tr])]) \
        if len(onset_tr) else None
    trial_payload = _example_trial(valid_recs, designs, a0, g0, idx, ex_trial)

    return {
        "session": r["session"], "neuron_id": r["neuron_id"],
        "lags_ms": STA_LAGS * DT * 1000.0,
        "sta_gap": sta_gap, "fit_full": fit_full,
        "fit_add": fit_add, "fit_mult": fit_mult,
        "trial": trial_payload,
    }


def _example_trial(valid_recs, designs, a0, g0, ex_idx, ex_trial):
    if ex_trial is None:
        return None
    r = valid_recs[ex_idx]
    d = designs[r["session"]]
    f_hold = int(d["fold_of_trial"][ex_trial])
    # fit this neuron's pooled params on all OTHER folds
    Aadd_full = d["Aadd"]
    tr_mask = d["fold_of_trial"][r["tr"]] != f_hold
    Aadd_s = Aadd_full[r["tr"], r["b"], :]
    Madd_s = r["drive"][:, None] * Aadd_s
    fa, fg = Aadd_s @ a0, Madd_s @ g0
    X = np.column_stack([np.ones_like(fa), fa, fg])
    beta = _ols(X[tr_mask], r["y"][tr_mask])
    # reconstruct the whole held-out trial
    B = d["B"]
    full = d["full"][ex_trial, :, r["ni"]]
    abl = d["abl"][ex_trial, :, r["ni"]]
    Aadd_t = Aadd_full[ex_trial]                # (B, L)
    fa_t = Aadd_t @ a0
    fg_t = (abl[:, None] * Aadd_t) @ g0
    yhat = beta[0] + beta[1] * fa_t + beta[2] * fg_t
    pred_full = abl + yhat
    li0 = int(np.where(STA_LAGS == 0)[0][0])
    sacc_bins = np.where(Aadd_t[:, li0] > 0)[0]
    valid = d["dfs"][ex_trial, :, r["ni"]] > 0
    t_ms = np.arange(B) * DT * 1000.0
    return {"t_ms": t_ms, "full": np.where(valid, full, np.nan),
            "abl": np.where(valid, abl, np.nan),
            "pred": np.where(valid, pred_full, np.nan),
            "sacc_bins": sacc_bins, "trial": ex_trial,
            "sacc_ms": sacc_bins * DT * 1000.0}


def print_model_stats(bundle=None):
    if bundle is None:
        bundle = compute_model_bundle()
    g = bundle["good"]

    def med(x):
        return np.nanmedian(x[g])

    print("\n=== Saccade-modulator model (reliable units, CV recovered-gap) ===")
    print(f"reliable N = {int(g.sum())}")
    print(f"per-neuron  additive-only : median recovered {med(bundle['rec_add']):.3f}")
    print(f"per-neuron  mult-only     : median recovered {med(bundle['rec_mult']):.3f}")
    print(f"per-neuron  both (ceiling): median recovered {med(bundle['rec_both']):.3f}")
    print(f"pooled      both          : median recovered {med(bundle['rec_pooled']):.3f}")
    print(f"pooled+δ    both          : median recovered {med(bundle['rec_pooled_delta']):.3f}")
    ceil = med(bundle["rec_both"])
    pool = med(bundle["rec_pooled_delta"])
    print(f"pooled+δ / ceiling = {pool / ceil:.2f}  "
          "(→1 means a simple shared modulator captures the learned one)")
    # all-units headline for context
    allm = np.isfinite(bundle["rec_both"])
    print(f"[all units] per-neuron both median recovered "
          f"{np.nanmedian(bundle['rec_both'][allm]):.3f} (N={int(allm.sum())})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    bundle = compute_model_bundle(force=args.force)
    print_model_stats(bundle)
